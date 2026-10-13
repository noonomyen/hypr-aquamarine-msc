#include "ScanoutCopy.hpp"

#include <aquamarine/backend/DRM.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string_view>
#include <hyprutils/utils/ScopeGuard.hpp>
#include <vector>
#include <drm_fourcc.h>
#include <format>
#include <mutex>
#include <poll.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

using namespace Aquamarine;
using namespace Hyprutils::Memory;
using Hyprutils::Utils::CScopeGuard;
#define SP CSharedPointer
#define WP CWeakPointer

namespace {
    // The NVIDIA Vulkan ICD opens an X display while it initialises and blocks
    // forever when that display accepts but never answers. Hyprland sets DISPLAY
    // for its Xwayland early in initServer, and the first commit can land before
    // Xwayland is serving. The copy engine never needs X, so hide DISPLAY while
    // the instance and device are created.
    struct SDisplayGuard {
        std::string saved;
        bool        had = false;

        SDisplayGuard() {
            if (const char* d = getenv("DISPLAY"); d != nullptr) {
                had   = true;
                saved = d;
            }
            unsetenv("DISPLAY");
        }

        ~SDisplayGuard() {
            if (had)
                setenv("DISPLAY", saved.c_str(), 1);
        }
    };
}

//
CMSCBuffer::CMSCBuffer(SP<CMSCTransfer> transfer, unsigned width, unsigned height, uint32_t fourcc) : m_transfer(transfer) {
    const auto VKFMT = mscVkFormat(fourcc);
    if (VKFMT == VK_FORMAT_UNDEFINED || !m_transfer) {
        m_attrs.success = false;
        return;
    }

    m_storage.stride         = mscStrideFor(width);
    const VkDeviceSize BYTES = (VkDeviceSize)m_storage.stride * height + 512;

    if (!m_transfer->allocateDestinationStorage(BYTES, m_storage)) {
        m_attrs.success = false;
        return;
    }

    size               = Hyprutils::Math::Vector2D{(double)width, (double)height};
    opaque             = false;
    m_attrs.success    = true;
    m_attrs.size       = size;
    m_attrs.format     = fourcc;
    m_attrs.modifier   = DRM_FORMAT_MOD_LINEAR;
    m_attrs.planes     = 1;
    m_attrs.fds        = {-1, -1, -1, -1};
    m_attrs.fds[0]     = m_storage.fd;
    m_attrs.strides[0] = m_storage.stride;
    m_attrs.offsets[0] = 0;
}

CMSCBuffer::~CMSCBuffer() {
    // CDRMFB hangs its destroy handler on this: without it a framebuffer keeps a
    // dangling buffer reference (see CDRMFB::import / the GBM buffer for parity).
    events.destroy.emit();

    if (m_transfer)
        m_transfer->freeDestinationStorage(m_storage);
}

eBufferCapability CMSCBuffer::caps() {
    return BUFFER_CAPABILITY_NONE;
}

eBufferType CMSCBuffer::type() {
    return BUFFER_TYPE_DMABUF;
}

void CMSCBuffer::update(const Hyprutils::Math::CRegion&) {
    ;
}

bool CMSCBuffer::isSynchronous() {
    return false;
}

bool CMSCBuffer::good() {
    return m_attrs.success;
}

SDMABUFAttrs CMSCBuffer::dmabuf() {
    return m_attrs;
}

SMSCBufferStorage& CMSCBuffer::storage() {
    return m_storage;
}

//
struct CMSCTransfer::SImpl {
    VkInstance                     instance         = VK_NULL_HANDLE;
    VkPhysicalDevice               physical         = VK_NULL_HANDLE;
    VkDevice                       device           = VK_NULL_HANDLE;
    VkQueue                        queue            = VK_NULL_HANDLE;
    uint32_t                       queueFamily      = 0;
    VkCommandPool                  commandPool      = VK_NULL_HANDLE;
    PFN_vkImportSemaphoreFdKHR     importSemaphore  = nullptr;
    PFN_vkGetSemaphoreFdKHR        getSemaphoreFd   = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR getMemoryFdProps = nullptr;

    // A copy records into the slot whose fence has already signalled, so the copy's own
    // completion never has to be waited on the host before the next one can be recorded.
    // The KMS in-fence makes the flip retire behind the copy, so the slot the display just
    // consumed is normally signalled again by the time it comes around, which leaves
    // headroom for three slots.
    struct SSlot {
        VkCommandBuffer command  = VK_NULL_HANDLE;
        VkFence         fence    = VK_NULL_HANDLE;
        VkSemaphore     outgoing = VK_NULL_HANDLE;
        // Imported from the incoming fence once per copy so the queue can wait for the
        // producer on the GPU instead of the compositor main thread waiting on the host.
        // One per slot: a slot is only reused once its fence signalled, so its previous
        // import is guaranteed to have been consumed by then.
        VkSemaphore incoming = VK_NULL_HANDLE;
        // The imported source of this slot's last copy: it may only be released once that
        // copy has completed, which the slot fence now tracks instead of a host wait.
        VkImage        srcImage  = VK_NULL_HANDLE;
        VkDeviceMemory srcMemory = VK_NULL_HANDLE;
    };
    SSlot    slots[3];
    size_t   nextSlot = 0;
    uint64_t copies = 0, gpuWaits = 0, hostWaits = 0, slotWaits = 0;

    // Copies are strictly serialized: one mutex for the whole transfer. Per-output
    // objects if two outputs ever need to copy concurrently.
    std::mutex mutex;
};

SP<CMSCTransfer> CMSCTransfer::create(WP<CDRMBackend> primary, SP<CBackend> logBackend) {
    auto transfer  = SP<CMSCTransfer>(new CMSCTransfer(primary, logBackend));
    transfer->self = transfer;
    if (!transfer->good())
        return nullptr;
    return transfer;
}

CMSCTransfer::CMSCTransfer(WP<CDRMBackend> primary, SP<CBackend> logBackend) : m_primary(primary), m_backend(logBackend) {
    m_impl = std::make_unique<SImpl>();

    if (getenv("DISPLAY") != nullptr)
        log(AQ_LOG_DEBUG, "drm: MSC: hiding DISPLAY while the copy engine initialises (the NVIDIA ICD opens X during init)");
    const SDisplayGuard DISPLAYGUARD;

    const int           RENDERFD = m_primary->drmRenderNodeFD();
    if (RENDERFD < 0) {
        log(AQ_LOG_ERROR, "drm: MSC: the compose backend has no render node");
        return;
    }

    struct stat st;
    if (fstat(RENDERFD, &st)) {
        log(AQ_LOG_ERROR, "drm: MSC: cannot stat the compose render node");
        return;
    }
    const uint32_t TARGETMAJOR = major(st.st_rdev), TARGETMINOR = minor(st.st_rdev);

    // Enable only the instance extensions the loader actually advertises:
    // VK_EXT_physical_device_drm is promoted to core in Vulkan 1.4 and is then no
    // longer listed, while requesting an absent extension fails the instance.
    uint32_t apiVersion = VK_API_VERSION_1_1;
    vkEnumerateInstanceVersion(&apiVersion);
    if (apiVersion < VK_API_VERSION_1_1)
        apiVersion = VK_API_VERSION_1_1;

    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, available.data());

    const char*              wanted[] = {"VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore_capabilities", "VK_EXT_physical_device_drm"};
    std::vector<const char*> enabled;
    for (const auto* W : wanted) {
        const bool HAS = std::ranges::any_of(available, [W](const auto& e) { return std::string_view{e.extensionName} == W; });
        if (HAS)
            enabled.push_back(W);
        else
            log(AQ_LOG_DEBUG, std::format("drm: MSC: instance extension {} not advertised, assuming core (Vulkan 1.2+)", W));
    }

    VkApplicationInfo    app  = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = apiVersion};
    VkInstanceCreateInfo inst = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app, .enabledExtensionCount = (uint32_t)enabled.size(), .ppEnabledExtensionNames = enabled.data()};
    if (vkCreateInstance(&inst, nullptr, &m_impl->instance) != VK_SUCCESS) {
        log(AQ_LOG_ERROR, "drm: MSC: vkCreateInstance failed (is the Vulkan loader installed?)");
        return;
    }

    VkPhysicalDevice devices[16];
    uint32_t         count = 16;
    vkEnumeratePhysicalDevices(m_impl->instance, &count, devices);
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceDrmPropertiesEXT drm   = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2      props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drm};
        vkGetPhysicalDeviceProperties2(devices[i], &props);
        if (drm.hasRender && drm.renderMajor == TARGETMAJOR && drm.renderMinor == TARGETMINOR) {
            m_impl->physical = devices[i];
            break;
        }
    }

    // No guessing: the copy engine must be the compose GPU itself.
    if (!m_impl->physical) {
        log(AQ_LOG_ERROR, std::format("drm: MSC: no Vulkan device for the compose render node {}:{}, refusing to use another GPU", TARGETMAJOR, TARGETMINOR));
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceDrmPropertiesEXT drm   = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2      props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drm};
            vkGetPhysicalDeviceProperties2(devices[i], &props);
            log(AQ_LOG_ERROR,
                std::format("drm: MSC:   candidate {} vendor 0x{:x} render {}:{}", props.properties.deviceName, props.properties.vendorID, drm.hasRender ? drm.renderMajor : -1,
                            drm.hasRender ? drm.renderMinor : -1));
        }
        return;
    }

    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_impl->physical, &props);
        log(AQ_LOG_DEBUG, std::format("drm: MSC: copy engine {} (vendor 0x{:x}) on render node {}:{}", props.deviceName, props.vendorID, TARGETMAJOR, TARGETMINOR));
    }

    VkQueueFamilyProperties families[32];
    count = 32;
    vkGetPhysicalDeviceQueueFamilyProperties(m_impl->physical, &count, families);
    for (uint32_t i = 0; i < count; ++i)
        if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            m_impl->queueFamily = i;

    float                   priority = 0;
    VkDeviceQueueCreateInfo queue = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = m_impl->queueFamily, .queueCount = 1, .pQueuePriorities = &priority};
    const char*             devExts[] = {"VK_KHR_external_memory_fd",        "VK_KHR_external_semaphore_fd", "VK_EXT_external_memory_dma_buf",
                                         "VK_EXT_image_drm_format_modifier", "VK_KHR_external_fence_fd",     "VK_EXT_queue_family_foreign"};
    VkDeviceCreateInfo      dev       = {.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                         .queueCreateInfoCount    = 1,
                                         .pQueueCreateInfos       = &queue,
                                         .enabledExtensionCount   = (uint32_t)std::size(devExts),
                                         .ppEnabledExtensionNames = devExts};
    if (vkCreateDevice(m_impl->physical, &dev, nullptr, &m_impl->device) != VK_SUCCESS) {
        log(AQ_LOG_ERROR,
            "drm: MSC: vkCreateDevice failed (needs VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd, VK_EXT_external_memory_dma_buf, "
            "VK_EXT_image_drm_format_modifier, VK_KHR_external_fence_fd and VK_EXT_queue_family_foreign)");
        return;
    }
    vkGetDeviceQueue(m_impl->device, m_impl->queueFamily, 0, &m_impl->queue);

    VkCommandPoolCreateInfo pool = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = m_impl->queueFamily};
    if (vkCreateCommandPool(m_impl->device, &pool, nullptr, &m_impl->commandPool) != VK_SUCCESS)
        return;
    VkCommandBufferAllocateInfo cmd = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = m_impl->commandPool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 3};
    VkCommandBuffer commands[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (vkAllocateCommandBuffers(m_impl->device, &cmd, commands) != VK_SUCCESS)
        return;
    VkExportSemaphoreCreateInfo exportSem = {.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO, .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    VkSemaphoreCreateInfo       sem       = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &exportSem};
    VkSemaphoreCreateInfo       incoming  = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    // Signalled fences: a slot starts out free, so the first three copies never wait.
    VkFenceCreateInfo fence = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    for (size_t i = 0; i < 3; ++i) {
        m_impl->slots[i].command = commands[i];
        if (vkCreateFence(m_impl->device, &fence, nullptr, &m_impl->slots[i].fence) != VK_SUCCESS)
            return;
        if (vkCreateSemaphore(m_impl->device, &sem, nullptr, &m_impl->slots[i].outgoing) != VK_SUCCESS)
            return;
        if (vkCreateSemaphore(m_impl->device, &incoming, nullptr, &m_impl->slots[i].incoming) != VK_SUCCESS)
            return;
    }

    m_impl->importSemaphore  = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(m_impl->device, "vkImportSemaphoreFdKHR");
    m_impl->getSemaphoreFd   = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(m_impl->device, "vkGetSemaphoreFdKHR");
    m_impl->getMemoryFdProps = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(m_impl->device, "vkGetMemoryFdPropertiesKHR");

    m_bad = false;
}

CMSCTransfer::~CMSCTransfer() {
    if (!m_impl)
        return;
    if (m_impl->device) {
        for (const auto& SLOT : m_impl->slots) {
            if (SLOT.outgoing)
                vkDestroySemaphore(m_impl->device, SLOT.outgoing, nullptr);
            if (SLOT.incoming)
                vkDestroySemaphore(m_impl->device, SLOT.incoming, nullptr);
            if (SLOT.fence)
                vkDestroyFence(m_impl->device, SLOT.fence, nullptr);
            if (SLOT.srcImage)
                vkDestroyImage(m_impl->device, SLOT.srcImage, nullptr);
            if (SLOT.srcMemory)
                vkFreeMemory(m_impl->device, SLOT.srcMemory, nullptr);
        }
        if (m_impl->commandPool)
            vkDestroyCommandPool(m_impl->device, m_impl->commandPool, nullptr);
        vkDestroyDevice(m_impl->device, nullptr);
    }
    if (m_impl->instance)
        vkDestroyInstance(m_impl->instance, nullptr);
}

bool CMSCTransfer::good() {
    return !m_bad && m_impl && m_impl->device;
}

WP<CDRMBackend> CMSCTransfer::primary() {
    return m_primary;
}

bool CMSCTransfer::allocateDestinationStorage(VkDeviceSize bytes, SMSCBufferStorage& out) {
    out.device = m_impl->device;

    VkExternalMemoryBufferCreateInfo external = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo               opts     = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external, .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(m_impl->device, &opts, nullptr, &out.buffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_impl->device, out.buffer, &req);

    // The destination is scanned out by the display GPU, so it must live in
    // system memory: a host-visible *device-local* heap (BAR/VRAM on a discrete
    // GPU) cannot be fetched by i915 and its commit fails. Prefer a
    // non-device-local coherent heap and only fall back to device-local.
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(m_impl->physical, &mem);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        const auto FLAGS = mem.memoryTypes[i].propertyFlags;
        if (!(req.memoryTypeBits & (1u << i)) || !(FLAGS & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || !(FLAGS & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            continue;

        const bool DEVICELOCAL = (mem.memoryHeaps[mem.memoryTypes[i].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        if (!DEVICELOCAL) {
            type = i;
            break;
        }
        if (type == UINT32_MAX)
            type = i;
    }
    if (type != UINT32_MAX)
        log(AQ_LOG_DEBUG,
            std::format("drm: MSC dst memory type {} heap {} device-local {} (system memory is required for scanout)", type, mem.memoryTypes[type].heapIndex,
                        (mem.memoryHeaps[mem.memoryTypes[type].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0));
    if (type == UINT32_MAX) {
        vkDestroyBuffer(m_impl->device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }

    VkExportMemoryAllocateInfo    exportInfo = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkMemoryDedicatedAllocateInfo dedicated  = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &exportInfo, .buffer = out.buffer};
    VkMemoryAllocateInfo          alloc      = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &dedicated, .allocationSize = req.size, .memoryTypeIndex = type};
    if (vkAllocateMemory(m_impl->device, &alloc, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyBuffer(m_impl->device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(m_impl->device, out.buffer, out.memory, 0) != VK_SUCCESS) {
        freeDestinationStorage(out);
        return false;
    }

    PFN_vkGetMemoryFdKHR get  = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_impl->device, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR info = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = out.memory, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    if (!get || get(m_impl->device, &info, &out.fd) != VK_SUCCESS) {
        freeDestinationStorage(out);
        return false;
    }

    return true;
}

void CMSCTransfer::freeDestinationStorage(SMSCBufferStorage& s) {
    if (s.fd >= 0)
        close(s.fd);
    if (s.device) {
        if (s.buffer)
            vkDestroyBuffer(s.device, s.buffer, nullptr);
        if (s.memory)
            vkFreeMemory(s.device, s.memory, nullptr);
    }
    s.fd     = -1;
    s.buffer = VK_NULL_HANDLE;
    s.memory = VK_NULL_HANDLE;
}

SP<IBuffer> CMSCTransfer::allocateDestination(uint32_t fourcc, const Hyprutils::Math::Vector2D& size) {
    if (!good() || !self)
        return nullptr;
    return SP<IBuffer>(new CMSCBuffer(self.lock(), (unsigned)size.x, (unsigned)size.y, fourcc));
}

bool CMSCTransfer::copy(SP<IBuffer> src, SP<IBuffer> dst, int waitFd, Hyprutils::OS::CFileDescriptor& outFence) {
    if (!good() || !src || !dst)
        return false;

    auto mscDst = static_cast<CMSCBuffer*>(dst.get());
    if (!mscDst->good())
        return false;

    const auto SRCDMA = src->dmabuf();
    if (!SRCDMA.success || SRCDMA.planes != 1 || SRCDMA.offsets[0] != 0)
        return false;

    const auto SRCFMT = mscVkFormat(SRCDMA.format);
    if (SRCFMT == VK_FORMAT_UNDEFINED)
        return false;

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    // Reuse the oldest slot: three copies of pipeline depth. The KMS in-fence retires the
    // flip behind its copy, so the slot is normally signalled by the time it comes around,
    // and the wait below is the only place the main thread can still wait for the GPU.
    auto& SLOT       = m_impl->slots[m_impl->nextSlot];
    m_impl->nextSlot = (m_impl->nextSlot + 1) % 3;
    if (vkGetFenceStatus(m_impl->device, SLOT.fence) == VK_NOT_READY) {
        ++m_impl->slotWaits;
        if (vkWaitForFences(m_impl->device, 1, &SLOT.fence, VK_TRUE, 1000000000) != VK_SUCCESS) {
            m_bad = true;
            log(AQ_LOG_ERROR, "drm: MSC: a copy slot never signalled, disabling the copy engine");
            return false;
        }
    }

    // Wait for the incoming fence on the GPU rather than on the host: import it as a
    // semaphore and let the queue wait on it. Host-waiting it here would block the
    // compositor main thread for the whole producer render. Importing is best
    // effort; when the driver refuses the fd, fall back to the host wait.
    bool gpuWait = false;
    if (waitFd >= 0) {
        const int IMPORTFD = m_impl->importSemaphore ? dup(waitFd) : -1;
        if (IMPORTFD >= 0) {
            VkImportSemaphoreFdInfoKHR import = {.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                                                 .semaphore  = SLOT.incoming,
                                                 .flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                                 .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
                                                 .fd         = IMPORTFD};
            gpuWait                           = m_impl->importSemaphore(m_impl->device, &import) == VK_SUCCESS;
            if (!gpuWait)
                close(IMPORTFD); // on success the import consumes the fd
        }

        static bool loggedGpuWait = false, loggedHostWait = false;
        if (gpuWait && !loggedGpuWait) {
            loggedGpuWait = true;
            log(AQ_LOG_DEBUG, "drm: MSC: the incoming fence is waited on the GPU, not on the host");
        } else if (!gpuWait && !loggedHostWait) {
            loggedHostWait = true;
            log(AQ_LOG_DEBUG, "drm: MSC: the incoming fence could not be imported, waiting on the host");
        }

        if (!gpuWait) {
            struct pollfd p = {.fd = waitFd, .events = POLLIN};
            if (poll(&p, 1, 1000) != 1)
                return false;
        }
    }

    // The slot's previous source image is only safe to release now that its copy completed.
    if (SLOT.srcImage) {
        vkDestroyImage(m_impl->device, SLOT.srcImage, nullptr);
        SLOT.srcImage = VK_NULL_HANDLE;
    }
    if (SLOT.srcMemory) {
        vkFreeMemory(m_impl->device, SLOT.srcMemory, nullptr);
        SLOT.srcMemory = VK_NULL_HANDLE;
    }

    VkSubresourceLayout                           plane        = {.offset = SRCDMA.offsets[0], .rowPitch = SRCDMA.strides[0]};
    VkImageDrmFormatModifierExplicitCreateInfoEXT modifierInfo = {.sType                       = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                                                                  .drmFormatModifier           = SRCDMA.modifier,
                                                                  .drmFormatModifierPlaneCount = 1,
                                                                  .pPlaneLayouts               = &plane};
    VkExternalMemoryImageCreateInfo               external     = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &modifierInfo, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo imgInfo = {.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                 .pNext         = &external,
                                 .imageType     = VK_IMAGE_TYPE_2D,
                                 .format        = SRCFMT,
                                 .extent        = {(uint32_t)SRCDMA.size.x, (uint32_t)SRCDMA.size.y, 1},
                                 .mipLevels     = 1,
                                 .arrayLayers   = 1,
                                 .samples       = VK_SAMPLE_COUNT_1_BIT,
                                 .tiling        = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                                 .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                 .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                                 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage           image   = VK_NULL_HANDLE;
    if (vkCreateImage(m_impl->device, &imgInfo, nullptr, &image) != VK_SUCCESS)
        return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_impl->device, image, &req);
    VkMemoryFdPropertiesKHR fdProps = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (!m_impl->getMemoryFdProps || m_impl->getMemoryFdProps(m_impl->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, SRCDMA.fds[0], &fdProps) != VK_SUCCESS) {
        vkDestroyImage(m_impl->device, image, nullptr);
        return false;
    }

    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(m_impl->physical, &mem);
    const uint32_t allowed = req.memoryTypeBits & fdProps.memoryTypeBits;
    VkDeviceMemory srcMem  = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < mem.memoryTypeCount && !srcMem; ++i) {
        if (!(allowed & (1u << i)))
            continue;
        const int IMPORTFD = dup(SRCDMA.fds[0]);
        if (IMPORTFD < 0)
            continue;
        VkImportMemoryFdInfoKHR       import = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = IMPORTFD};
        VkMemoryDedicatedAllocateInfo dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &import, .image = image};
        VkMemoryAllocateInfo          alloc     = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &dedicated, .allocationSize = req.size, .memoryTypeIndex = i};
        if (vkAllocateMemory(m_impl->device, &alloc, nullptr, &srcMem) != VK_SUCCESS) {
            close(IMPORTFD);
            srcMem = VK_NULL_HANDLE;
            continue;
        }
    }
    if (!srcMem || vkBindImageMemory(m_impl->device, image, srcMem, 0) != VK_SUCCESS) {
        if (srcMem)
            vkFreeMemory(m_impl->device, srcMem, nullptr);
        vkDestroyImage(m_impl->device, image, nullptr);
        return false;
    }

    // Release the imported source on every exit path. The one exception is a
    // fence timeout below: the GPU may still be reading it, freeing then would be
    // a use-after-free, so those resources ride along until the device dies.
    bool                     keepSource = false;
    CScopeGuard              sourceGuard([&] {
        if (keepSource)
            return;
        vkDestroyImage(m_impl->device, image, nullptr);
        vkFreeMemory(m_impl->device, srcMem, nullptr);
    });

    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (vkResetCommandBuffer(SLOT.command, 0) != VK_SUCCESS || vkBeginCommandBuffer(SLOT.command, &begin) != VK_SUCCESS)
        return false;

    VkImageMemoryBarrier acquire = {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
                                    .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
                                    .dstQueueFamilyIndex = m_impl->queueFamily,
                                    .image               = image,
                                    .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(SLOT.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire);

    VkBufferImageCopy region = {.bufferOffset      = 0,
                                .bufferRowLength   = mscDst->storage().stride / 4,
                                .bufferImageHeight = (uint32_t)SRCDMA.size.y,
                                .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent       = {(uint32_t)SRCDMA.size.x, (uint32_t)SRCDMA.size.y, 1}};
    vkCmdCopyImageToBuffer(SLOT.command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mscDst->storage().buffer, 1, &region);
    if (vkEndCommandBuffer(SLOT.command) != VK_SUCCESS)
        return false;

    VkSemaphore          WAITSEM   = SLOT.incoming;
    VkPipelineStageFlags WAITSTAGE = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo         submit    = {.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .waitSemaphoreCount   = gpuWait ? 1u : 0u,
                                      .pWaitSemaphores      = gpuWait ? &WAITSEM : nullptr,
                                      .pWaitDstStageMask    = gpuWait ? &WAITSTAGE : nullptr,
                                      .commandBufferCount   = 1,
                                      .pCommandBuffers      = &SLOT.command,
                                      .signalSemaphoreCount = 1,
                                      .pSignalSemaphores    = &SLOT.outgoing};
    if (vkResetFences(m_impl->device, 1, &SLOT.fence) != VK_SUCCESS || vkQueueSubmit(m_impl->queue, 1, &submit, SLOT.fence) != VK_SUCCESS)
        return false;

    // The copy is in flight, so the imported source now belongs to the slot: it is released
    // when the slot is reused, after its fence has signalled. Nothing here waits on the host.
    SLOT.srcImage  = image;
    SLOT.srcMemory = srcMem;
    keepSource     = true;

    VkSemaphoreGetFdInfoKHR info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = SLOT.outgoing, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    int                     fd   = -1;
    if (!m_impl->getSemaphoreFd || m_impl->getSemaphoreFd(m_impl->device, &info, &fd) != VK_SUCCESS)
        return false;
    outFence = Hyprutils::OS::CFileDescriptor{fd};

    ++m_impl->copies;
    if (gpuWait)
        ++m_impl->gpuWaits;
    else
        ++m_impl->hostWaits;
    if (m_impl->copies % 300 == 0)
        log(AQ_LOG_DEBUG,
            std::format("drm: MSC copy stats: {} copies, {} gpu waits, {} host waits, {} slot waits", m_impl->copies, m_impl->gpuWaits, m_impl->hostWaits, m_impl->slotWaits));

    return true;
}

void CMSCTransfer::log(eBackendLogLevel level, const std::string& msg) {
    if (auto backend = m_backend.lock())
        backend->log(level, msg);
}

//
SP<CMSCAllocator> CMSCAllocator::create(SP<CMSCTransfer> transfer, SP<CBackend> backend) {
    return SP<CMSCAllocator>(new CMSCAllocator(transfer, backend));
}

CMSCAllocator::CMSCAllocator(SP<CMSCTransfer> transfer, SP<CBackend> backend) : m_transfer(transfer), m_backend(backend) {
    ;
}

SP<IBuffer> CMSCAllocator::acquire(const SAllocatorBufferParams& params, SP<CSwapchain> swapchain) {
    if (!m_transfer)
        return nullptr;

    const auto BUFFER = m_transfer->allocateDestination(params.format == DRM_FORMAT_INVALID ? DRM_FORMAT_ARGB8888 : params.format, params.size);

    // A broken destination would poison the swapchain for the rest of the session.
    return BUFFER && BUFFER->good() ? BUFFER : nullptr;
}

SP<CBackend> CMSCAllocator::getBackend() {
    return m_backend.lock();
}

int CMSCAllocator::drmFD() {
    if (!m_transfer)
        return -1;

    const auto PRIMARY = m_transfer->primary().lock();
    return PRIMARY ? PRIMARY->drmFD() : -1;
}

eAllocatorType CMSCAllocator::type() {
    return AQ_ALLOCATOR_TYPE_GBM;
}
