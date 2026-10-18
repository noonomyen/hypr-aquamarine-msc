#include "ScanoutCopy.hpp"
#include "Shared.hpp"

#include <aquamarine/backend/DRM.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <iterator>
#include <string_view>
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
#define SP CSharedPointer
#define WP CWeakPointer

namespace {
    enum class ECopyMode {
        Full,
        Partial,
        Skip
    };

    // Beyond this many boxes, or this much of the frame, a full copy is both simpler and no
    // slower, so the tracking gives up on that frame.
    constexpr size_t MSC_DAMAGE_MAX_RECTS     = 16;
    constexpr double MSC_DAMAGE_FULL_FRACTION = 0.5;

    // AQ_MSC_DAMAGE=0 copies whole frames; on (the default) the copy follows the damage the
    // compositor reports, skipping frames that changed nothing and moving only the boxes that did.
    bool trackDamage = [] { return !envExplicitlyDisabled("AQ_MSC_DAMAGE"); }();

    // The NVIDIA Vulkan ICD can dlclose its core library while the process tears down, so a Vulkan
    // call from a destructor that runs after that (these objects are owned by Hyprland's global
    // backend, which is constructed before this transfer and destroyed after it) faults on an
    // unmapped address. Registering std::atexit here, at transfer creation, runs it before those
    // globals: mark the exit and leak the Vulkan objects instead. The kernel reclaims them and the
    // process is going away anyway.
    // Deliberate simplification: the leak is exit-only; a deterministic backend teardown would
    // remove the need for it.
    std::atomic<bool> driverTearingDown = false;

    void              noteDriverTearingDown() {
        driverTearingDown = true;
    }

    size_t damageBucket(double fraction) {
        static const double BOUNDS[] = {0.000001, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.000001};
        for (size_t i = 0; i < std::size(BOUNDS); ++i)
            if (fraction < BOUNDS[i])
                return i;
        return std::size(BOUNDS) - 1;
    }

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
    const VkDeviceSize BYTES = (VkDeviceSize)m_storage.stride * height;

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

const Hyprutils::Math::CRegion& CMSCBuffer::pendingDamage() const {
    return m_pendingDamage;
}

bool CMSCBuffer::seeded() const {
    return m_seeded;
}

void CMSCBuffer::markSynced() {
    m_pendingDamage.clear();
    m_seeded = true;
}

void CMSCBuffer::accumulateDamage(const Hyprutils::Math::CRegion& region) {
    m_pendingDamage.add(region);
}

//
struct CMSCTransfer::SImpl {
    VkInstance                       instance         = VK_NULL_HANDLE;
    VkPhysicalDevice                 physical         = VK_NULL_HANDLE;
    VkDevice                         device           = VK_NULL_HANDLE;
    VkQueue                          queue            = VK_NULL_HANDLE;
    uint32_t                         queueFamily      = 0;
    bool                             useCopyEngine    = true;
    VkCommandPool                    commandPool      = VK_NULL_HANDLE;
    PFN_vkImportSemaphoreFdKHR       importSemaphore  = nullptr;
    PFN_vkGetSemaphoreFdKHR          getSemaphoreFd   = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR   getMemoryFdProps = nullptr;
    PFN_vkGetMemoryFdKHR             getMemoryFd      = nullptr;
    VkPhysicalDeviceMemoryProperties memProps{};

    // One imported source per compositor buffer, kept between copies. src is the identity
    // of the producer's buffer, so an entry stays valid exactly as long as that buffer
    // does; slotsUsing counts the copies whose slot has not come round yet.
    struct SSource {
        WP<IBuffer>    src;
        VkImage        image    = VK_NULL_HANDLE;
        VkDeviceMemory memory   = VK_NULL_HANDLE;
        uint32_t       format   = 0;
        uint64_t       modifier = 0;
        uint64_t       width = 0, height = 0, stride = 0, offset = 0;
        int            slotsUsing = 0;
        uint64_t       lastUse    = 0;
        uint64_t       id         = 0;

        void           destroy(VkDevice dev) {
            if (image)
                vkDestroyImage(dev, image, nullptr);
            if (memory)
                vkFreeMemory(dev, memory, nullptr);
            image  = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
        }
    };

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
        // Which source import this slot's last copy read, so the entry can be marked idle
        // once that copy has completed (the slot fence tracks that, not a host wait). The
        // id, not an index: evicting an entry shifts the ones behind it.
        uint64_t sourceId = 0;
    };
    // Every destination the frame copy has written, so each frame's damage can be handed to all of
    // them. Weak: a swapchain that is reallocated simply drops out.
    std::vector<WP<IBuffer>> frameDestinations;
    std::vector<SSource>     sources;
    uint64_t                 sourceCounter = 0, useCounter = 0;
    SSlot                    slots[3];
    size_t                   nextSlot = 0;
    uint64_t                 copies = 0, gpuWaits = 0, hostWaits = 0, slotWaits = 0;

    // What the damage tracking did, and how much of each frame it moved. The distribution of that
    // fraction is the number that decides whether the bookkeeping pays for itself.
    uint64_t damageFull = 0, damagePartial = 0, damageSkipped = 0;
    uint64_t damageRects = 0, damagePixels = 0, damageFramePixels = 0;
    uint64_t damageBuckets[8] = {};
    // More than three copies can never be in flight, so a cache of this size always has an
    // evictable entry.
    static constexpr size_t MAXSOURCES = 16;

    // Copies are strictly serialized: one mutex for the whole transfer. Per-output
    // objects if two outputs ever need to copy concurrently.
    std::mutex mutex;
};

SP<CMSCTransfer> CMSCTransfer::create(WP<CDRMBackend> primary, SP<CBackend> logBackend, bool useCopyEngine) {
    auto transfer  = SP<CMSCTransfer>(new CMSCTransfer(primary, logBackend, useCopyEngine));
    transfer->self = transfer;
    if (!transfer->good())
        return nullptr;
    return transfer;
}

CMSCTransfer::CMSCTransfer(WP<CDRMBackend> primary, SP<CBackend> logBackend, bool useCopyEngine) : m_primary(primary), m_backend(logBackend) {
    m_impl                = std::make_unique<SImpl>();
    m_impl->useCopyEngine = useCopyEngine;

    // Armed once, at creation, so it pops before the globals that own this transfer during exit.
    static const bool ATEXIT_ARMED = [] {
        std::atexit(noteDriverTearingDown);
        return true;
    }();
    (void)ATEXIT_ARMED;

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

    // Enable only the advertised instance extensions (an absent one fails the instance, and both
    // of these are core in 1.1). VK_EXT_physical_device_drm is a device extension, queried below.
    uint32_t apiVersion = VK_API_VERSION_1_1;
    vkEnumerateInstanceVersion(&apiVersion);
    if (apiVersion < VK_API_VERSION_1_1)
        apiVersion = VK_API_VERSION_1_1;

    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, available.data());

    const char*              wanted[] = {"VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore_capabilities"};
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
    uint32_t         count      = (uint32_t)std::size(devices);
    const VkResult   ENUMRESULT = vkEnumeratePhysicalDevices(m_impl->instance, &count, devices);
    if (ENUMRESULT != VK_SUCCESS && ENUMRESULT != VK_INCOMPLETE) {
        log(AQ_LOG_ERROR, std::format("drm: MSC: enumerating Vulkan devices failed (VkResult {})", (int)ENUMRESULT));
        return;
    }
    if (ENUMRESULT == VK_INCOMPLETE)
        log(AQ_LOG_DEBUG,
            std::format("drm: MSC: more than {} Vulkan devices present, considering the first {} (the compose render node is matched by DRM major:minor)", std::size(devices),
                        count));
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

    // AQ_MSC_QUEUE picks the family the copy is submitted to: transfer asks for the async
    // copy engine (dedicated DMA hardware, so the copies do not compete with the compositor's
    // rendering for the graphics engine), graphics for the graphics engine. Neither falls back.
    VkQueueFamilyProperties families[32];
    count = (uint32_t)std::size(families);
    vkGetPhysicalDeviceQueueFamilyProperties(m_impl->physical, &count, families);
    bool found = false;
    if (m_impl->useCopyEngine) {
        for (uint32_t i = 0; i < count; ++i) {
            if (!families[i].queueCount)
                continue;
            if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                m_impl->queueFamily = i;
                found               = true;
                break;
            }
        }
        if (!found)
            log(AQ_LOG_ERROR, "drm: MSC: AQ_MSC_QUEUE=transfer, but this device exposes no transfer-only queue family; refusing to fall back to graphics");
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                m_impl->queueFamily = i;
                found               = true;
            }
        }
        if (!found)
            log(AQ_LOG_ERROR, "drm: MSC: AQ_MSC_QUEUE=graphics, but this device exposes no graphics queue family");
    }
    if (!found)
        return;

    log(AQ_LOG_DEBUG, std::format("drm: MSC: damage tracking {}", trackDamage ? "on" : "off (AQ_MSC_DAMAGE=0)"));

    // A transfer-only family executes transfer commands only: the copy and its barriers are
    // transfer commands, but vkCmdClearColorImage is not, and it loses the device there.
    log(AQ_LOG_DEBUG, std::format("drm: MSC: copy queue family {} ({})", m_impl->queueFamily, m_impl->useCopyEngine ? "transfer-only, the async copy engine" : "graphics"));

    float                   priority = 0;
    VkDeviceQueueCreateInfo queue = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = m_impl->queueFamily, .queueCount = 1, .pQueuePriorities = &priority};
    const char*             devExts[] = {"VK_KHR_external_memory_fd", "VK_KHR_external_semaphore_fd", "VK_EXT_external_memory_dma_buf", "VK_EXT_image_drm_format_modifier",
                                         "VK_EXT_queue_family_foreign"};
    VkDeviceCreateInfo      dev       = {.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                         .queueCreateInfoCount    = 1,
                                         .pQueueCreateInfos       = &queue,
                                         .enabledExtensionCount   = (uint32_t)std::size(devExts),
                                         .ppEnabledExtensionNames = devExts};
    if (vkCreateDevice(m_impl->physical, &dev, nullptr, &m_impl->device) != VK_SUCCESS) {
        log(AQ_LOG_ERROR,
            "drm: MSC: vkCreateDevice failed (needs VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd, VK_EXT_external_memory_dma_buf, "
            "VK_EXT_image_drm_format_modifier and VK_EXT_queue_family_foreign)");
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
    m_impl->getMemoryFd      = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_impl->device, "vkGetMemoryFdKHR");
    vkGetPhysicalDeviceMemoryProperties(m_impl->physical, &m_impl->memProps);

    m_bad = false;
}

CMSCTransfer::~CMSCTransfer() {
    if (!m_impl)
        return;
    // Once the process is exiting the driver may already be unloading; skip the Vulkan teardown
    // and leak, rather than jump into an unloaded ICD.
    if (m_impl->device && !driverTearingDown) {
        // The rebuild path can drop this transfer while a copy is still in flight, so make sure
        // the device is idle before its objects go (a lost device returns at once).
        vkDeviceWaitIdle(m_impl->device);

        for (const auto& SLOT : m_impl->slots) {
            if (SLOT.outgoing)
                vkDestroySemaphore(m_impl->device, SLOT.outgoing, nullptr);
            if (SLOT.incoming)
                vkDestroySemaphore(m_impl->device, SLOT.incoming, nullptr);
            if (SLOT.fence)
                vkDestroyFence(m_impl->device, SLOT.fence, nullptr);
        }
        for (auto& SOURCE : m_impl->sources)
            SOURCE.destroy(m_impl->device);
        m_impl->sources.clear();
        if (m_impl->commandPool)
            vkDestroyCommandPool(m_impl->device, m_impl->commandPool, nullptr);
        vkDestroyDevice(m_impl->device, nullptr);
    }
    if (m_impl->instance && !driverTearingDown)
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
    const auto& mem  = m_impl->memProps;
    uint32_t    type = UINT32_MAX;
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

    const auto           get  = m_impl->getMemoryFd;
    VkMemoryGetFdInfoKHR info = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = out.memory, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    int                  fd   = -1;
    if (!get || get(m_impl->device, &info, &fd) != VK_SUCCESS) {
        freeDestinationStorage(out);
        return false;
    }
    out.fd = fd;

    return true;
}

void CMSCTransfer::freeDestinationStorage(SMSCBufferStorage& s) {
    if (s.fd >= 0)
        close(s.fd);
    // Same as the destructor: after the exit handler ran, the ICD may be unloaded, so leak the
    // Vulkan objects rather than fault on an unmapped dispatch target.
    if (s.device && !driverTearingDown) {
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

// One import per producer buffer. An entry a slot still points at is never evicted, or a
// later copy would read memory that has already been freed and handed out again.
int CMSCTransfer::acquireSource(const SP<IBuffer>& src, const SDMABUFAttrs& attrs, VkFormat format) {
    for (size_t i = 0; i < m_impl->sources.size(); ++i) {
        const auto& SOURCE = m_impl->sources[i];
        if (SOURCE.src == src && SOURCE.format == attrs.format && SOURCE.modifier == attrs.modifier && SOURCE.width == (uint64_t)attrs.size.x &&
            SOURCE.height == (uint64_t)attrs.size.y && SOURCE.stride == attrs.strides[0] && SOURCE.offset == attrs.offsets[0])
            return (int)i;
    }

    // Not seen before: make room for one more.
    if (m_impl->sources.size() >= SImpl::MAXSOURCES) {
        bool   haveVictim = false;
        size_t victim     = 0;
        for (size_t i = 0; i < m_impl->sources.size(); ++i) {
            const auto& SOURCE = m_impl->sources[i];
            if (SOURCE.slotsUsing > 0)
                continue;
            if (!haveVictim || SOURCE.lastUse < m_impl->sources[victim].lastUse) {
                victim     = i;
                haveVictim = true;
            }
        }
        if (!haveVictim) {
            log(AQ_LOG_ERROR, "drm: MSC: the source import cache is full and every entry is still in flight");
            return -1;
        }
        m_impl->sources[victim].destroy(m_impl->device);
        m_impl->sources.erase(m_impl->sources.begin() + (long)victim);
    }

    VkSubresourceLayout                           plane        = {.offset = attrs.offsets[0], .rowPitch = attrs.strides[0]};
    VkImageDrmFormatModifierExplicitCreateInfoEXT modifierInfo = {.sType                       = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                                                                  .drmFormatModifier           = attrs.modifier,
                                                                  .drmFormatModifierPlaneCount = 1,
                                                                  .pPlaneLayouts               = &plane};
    VkExternalMemoryImageCreateInfo               external     = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &modifierInfo, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo imgInfo = {.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                 .pNext         = &external,
                                 .imageType     = VK_IMAGE_TYPE_2D,
                                 .format        = format,
                                 .extent        = {(uint32_t)attrs.size.x, (uint32_t)attrs.size.y, 1},
                                 .mipLevels     = 1,
                                 .arrayLayers   = 1,
                                 .samples       = VK_SAMPLE_COUNT_1_BIT,
                                 .tiling        = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                                 .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                 .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                                 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage           image   = VK_NULL_HANDLE;
    if (!checkResult(vkCreateImage(m_impl->device, &imgInfo, nullptr, &image), "importing a source image"))
        return -1;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_impl->device, image, &req);
    VkMemoryFdPropertiesKHR fdProps = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (!m_impl->getMemoryFdProps ||
        !checkResult(m_impl->getMemoryFdProps(m_impl->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, attrs.fds[0], &fdProps), "querying the source dma-buf")) {
        if (image)
            vkDestroyImage(m_impl->device, image, nullptr);
        return -1;
    }

    const auto&    mem     = m_impl->memProps;
    const uint32_t allowed = req.memoryTypeBits & fdProps.memoryTypeBits;
    VkDeviceMemory srcMem  = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < mem.memoryTypeCount && !srcMem; ++i) {
        if (!(allowed & (1u << i)))
            continue;
        const int IMPORTFD = dup(attrs.fds[0]);
        if (IMPORTFD < 0)
            continue;
        VkImportMemoryFdInfoKHR       import = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = IMPORTFD};
        VkMemoryDedicatedAllocateInfo dedicated  = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &import, .image = image};
        VkMemoryAllocateInfo          alloc      = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &dedicated, .allocationSize = req.size, .memoryTypeIndex = i};
        const VkResult                ALLOCSTATE = vkAllocateMemory(m_impl->device, &alloc, nullptr, &srcMem);
        if (ALLOCSTATE != VK_SUCCESS) {
            close(IMPORTFD);
            srcMem = VK_NULL_HANDLE;
            if (ALLOCSTATE == VK_ERROR_DEVICE_LOST) {
                // trying another memory type is normal, a lost device is not
                checkResult(ALLOCSTATE, "importing the source dma-buf");
                vkDestroyImage(m_impl->device, image, nullptr);
                return -1;
            }
            continue;
        }
    }
    if (!srcMem || !checkResult(vkBindImageMemory(m_impl->device, image, srcMem, 0), "binding the source image")) {
        if (srcMem)
            vkFreeMemory(m_impl->device, srcMem, nullptr);
        vkDestroyImage(m_impl->device, image, nullptr);
        return -1;
    }

    SImpl::SSource SOURCE;
    SOURCE.src      = src;
    SOURCE.image    = image;
    SOURCE.memory   = srcMem;
    SOURCE.format   = attrs.format;
    SOURCE.modifier = attrs.modifier;
    SOURCE.width    = (uint64_t)attrs.size.x;
    SOURCE.height   = (uint64_t)attrs.size.y;
    SOURCE.stride   = attrs.strides[0];
    SOURCE.offset   = attrs.offsets[0];
    SOURCE.id       = ++m_impl->sourceCounter;
    m_impl->sources.emplace_back(std::move(SOURCE));
    return (int)(m_impl->sources.size() - 1);
}

void CMSCTransfer::evictIdleSources() {
    std::erase_if(m_impl->sources, [this](auto& SOURCE) {
        if (SOURCE.slotsUsing > 0 || SOURCE.src.valid())
            return false;
        SOURCE.destroy(m_impl->device);
        return true;
    });
}

// A lost device cannot be used again, and the backend rebuilds the whole engine from this flag.
// Every other failure is per-frame: log it, let the next frame try again.
bool CMSCTransfer::checkResult(VkResult result, const char* what) {
    if (result == VK_SUCCESS)
        return true;

    log(AQ_LOG_ERROR, std::format("drm: MSC: {} failed (VkResult {})", what, (int)result));
    if (result == VK_ERROR_DEVICE_LOST) {
        m_bad = true;
        log(AQ_LOG_ERROR, "drm: MSC: the compose device was lost, the copy engine will be rebuilt");
    }
    return false;
}

uint64_t CMSCTransfer::copyCount() const {
    return m_impl ? m_impl->copies : 0;
}

bool CMSCTransfer::copy(SP<IBuffer> src, SP<IBuffer> dst, int waitFd, Hyprutils::OS::CFileDescriptor& outFence, const Hyprutils::Math::CRegion* damage) {
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

    const uint32_t WIDTH  = (uint32_t)SRCDMA.size.x;
    const uint32_t HEIGHT = (uint32_t)SRCDMA.size.y;
    // The copy converts strides, so only the pixels have to line up for the damage rects to mean
    // the same thing on both sides.
    const bool SAMESIZE = WIDTH == (uint32_t)mscDst->size.x && HEIGHT == (uint32_t)mscDst->size.y;

    // The damage bookkeeping below and the Vulkan recording further down both touch the transfer's
    // shared state, so take the copy lock before either.
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    // This frame's damage, or the whole frame when the compositor reported none: with no damage to
    // trust, every destination takes the full frame next time so nothing it changed is missed.
    Hyprutils::Math::CRegion tracked;
    if (SAMESIZE) {
        if (damage) {
            tracked.set(*damage);
            tracked.intersect(0, 0, (double)WIDTH, (double)HEIGHT);
        } else
            tracked.set(Hyprutils::Math::CRegion(0, 0, (double)WIDTH, (double)HEIGHT));
    }

    // The destination is one of a rotating set and every buffer in it has to catch up on the frames
    // it sat out, so this frame's damage is added to all of them, the one being written included;
    // each forgets it once written. Only destinations of this frame's size consume it: the cursor
    // destination and any other resolution would otherwise accumulate damage they never clear.
    // (Recording it only in the destination being written measured the same copy volume; it stays
    // for a consumer whose damage is not accumulated over the buffer's age.)
    if (trackDamage && SAMESIZE) {
        bool known = false;
        for (auto it = m_impl->frameDestinations.begin(); it != m_impl->frameDestinations.end();) {
            if (it->expired()) {
                it = m_impl->frameDestinations.erase(it);
                continue;
            }
            if (it->lock().get() == dst.get())
                known = true;
            ++it;
        }
        if (!known)
            m_impl->frameDestinations.emplace_back(dst);

        const Hyprutils::Math::Vector2D FRAMESIZE{(double)WIDTH, (double)HEIGHT};
        for (auto& DEST : m_impl->frameDestinations)
            if (auto BUF = DEST.lock(); BUF && BUF->size == FRAMESIZE)
                static_cast<CMSCBuffer*>(BUF.get())->accumulateDamage(tracked);
    }

    Hyprutils::Math::CRegion pending = mscDst->pendingDamage();
    // The owed set is shared by every output; keep only what belongs to this frame, so a box can
    // never describe another output (or the cursor) and run past the destination buffer.
    pending.intersect(0, 0, (double)WIDTH, (double)HEIGHT);

    // Extract the boxes and their area once: pixman's rect array is already disjoint and sorted,
    // while getRects() would build a fresh vector on every call.
    int                   damageRectCount = 0;
    const pixman_box32_t* DAMAGE          = pixman_region32_rectangles(pending.pixman(), &damageRectCount);
    uint64_t              damageArea      = 0;
    for (int i = 0; i < damageRectCount; ++i)
        damageArea += (uint64_t)(DAMAGE[i].x2 - DAMAGE[i].x1) * (uint64_t)(DAMAGE[i].y2 - DAMAGE[i].y1);

    ECopyMode mode;
    if (!trackDamage || !SAMESIZE || !mscDst->seeded() || !damage)
        mode = ECopyMode::Full;
    else if (pending.empty())
        mode = ECopyMode::Skip;
    else if (damageRectCount > (int)MSC_DAMAGE_MAX_RECTS || (double)damageArea > MSC_DAMAGE_FULL_FRACTION * (double)WIDTH * (double)HEIGHT)
        mode = ECopyMode::Full; // past the point where the bookkeeping pays for itself
    else
        mode = ECopyMode::Partial;

    const uint64_t FRAMEPIXELS = (uint64_t)WIDTH * (uint64_t)HEIGHT;
    const auto     ACCOUNT     = [&](ECopyMode taken, size_t rects) {
        const uint64_t MOVED = taken == ECopyMode::Full ? FRAMEPIXELS : damageArea;
        m_impl->damageFramePixels += FRAMEPIXELS;
        m_impl->damagePixels += MOVED;
        m_impl->damageRects += rects;

        if (taken == ECopyMode::Full)
            m_impl->damageFull++;
        else if (taken == ECopyMode::Partial)
            m_impl->damagePartial++;
        else
            m_impl->damageSkipped++;

        m_impl->damageBuckets[damageBucket(FRAMEPIXELS ? (double)MOVED / (double)FRAMEPIXELS : 0.0)]++;
        if ((m_impl->damageFull + m_impl->damagePartial + m_impl->damageSkipped) % 300 == 0)
            logStats();
    };

    if (mode == ECopyMode::Skip) {
        // Nothing changed, so the destination already holds this frame. There is no copy and no
        // fence: the caller has nothing to wait for.
        ACCOUNT(mode, 0);
        mscDst->markSynced();
        return true;
    }

    // Reuse the oldest slot: three copies of pipeline depth. The KMS in-fence retires the
    // flip behind its copy, so the slot is normally signalled by the time it comes around,
    // and the wait below is the only place the main thread can still wait for the GPU.
    auto& SLOT       = m_impl->slots[m_impl->nextSlot];
    m_impl->nextSlot = (m_impl->nextSlot + 1) % 3;
    if (vkGetFenceStatus(m_impl->device, SLOT.fence) != VK_SUCCESS) {
        ++m_impl->slotWaits;
        // A slot that does not come back is either a lost device or a wedged copy engine: neither
        // can be salvaged here, and the backend rebuilds the engine from the flag either way.
        const VkResult WAITSTATE = vkWaitForFences(m_impl->device, 1, &SLOT.fence, VK_TRUE, 1000000000);
        if (WAITSTATE != VK_SUCCESS) {
            m_bad = true;
            if (WAITSTATE == VK_ERROR_DEVICE_LOST)
                log(AQ_LOG_ERROR, "drm: MSC: the compose device was lost, the copy engine will be rebuilt");
            else
                log(AQ_LOG_ERROR, std::format("drm: MSC: a copy slot never signalled (VkResult {}), the copy engine will be rebuilt", (int)WAITSTATE));
            return false;
        }
    }

    // The slot's previous copy has completed, so the source it read is idle again. Look the
    // entry up by its id: the vector may have shifted since the copy was recorded.
    if (SLOT.sourceId) {
        for (auto& SOURCE : m_impl->sources) {
            if (SOURCE.id == SLOT.sourceId) {
                SOURCE.slotsUsing--;
                break;
            }
        }
        SLOT.sourceId = 0;
    }
    evictIdleSources();

    const int SOURCEINDEX = acquireSource(src, SRCDMA, SRCFMT);
    if (SOURCEINDEX < 0)
        return false;
    auto& SOURCE = m_impl->sources[(size_t)SOURCEINDEX];

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
            // An errored producer fence is a failure, not a ready one, and a signal in this process
            // must not drop the frame (same rule as mscWaitCopyFence).
            struct pollfd p = {.fd = waitFd, .events = POLLIN};
            while (poll(&p, 1, 1000) == -1 && errno == EINTR) {}
            if (!(p.revents & POLLIN))
                return false;
        }
    }

    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (!checkResult(vkResetCommandBuffer(SLOT.command, 0), "resetting a copy command buffer") ||
        !checkResult(vkBeginCommandBuffer(SLOT.command, &begin), "beginning a copy command buffer"))
        return false;

    // The cached image is owned by the foreign family between copies: each copy acquires it
    // and the release below hands it back, since a repeated acquire without a release in
    // between is not valid.
    VkImageMemoryBarrier acquire = {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
                                    .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
                                    .dstQueueFamilyIndex = m_impl->queueFamily,
                                    .image               = SOURCE.image,
                                    .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(SLOT.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquire);

    // The rect count is bounded: Partial is only taken when there are at most this many, so the
    // regions can live on the stack instead of a heap vector.
    std::array<VkBufferImageCopy, MSC_DAMAGE_MAX_RECTS> REGIONS{};
    uint32_t                                            regionCount = 0;
    if (mode == ECopyMode::Full) {
        REGIONS[regionCount++] = {.bufferOffset      = 0,
                                  .bufferRowLength   = mscDst->storage().stride / 4,
                                  .bufferImageHeight = HEIGHT,
                                  .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                  .imageExtent       = {WIDTH, HEIGHT, 1}};
    } else {
        for (int i = 0; i < damageRectCount; ++i) {
            const auto& BOX = DAMAGE[i];
            // Both offsets are needed: imageOffset selects the box in the source frame and
            // bufferOffset places it at the same coordinates in the destination buffer. Without
            // imageOffset every box is fed the frame's top-left corner.
            REGIONS[regionCount++] = {.bufferOffset      = (VkDeviceSize)BOX.y1 * mscDst->storage().stride + (VkDeviceSize)BOX.x1 * 4,
                                      .bufferRowLength   = mscDst->storage().stride / 4,
                                      .bufferImageHeight = HEIGHT,
                                      .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                      .imageOffset       = {(int32_t)BOX.x1, (int32_t)BOX.y1, 0},
                                      .imageExtent       = {(uint32_t)(BOX.x2 - BOX.x1), (uint32_t)(BOX.y2 - BOX.y1), 1}};
        }
    }
    vkCmdCopyImageToBuffer(SLOT.command, SOURCE.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mscDst->storage().buffer, regionCount, REGIONS.data());

    VkImageMemoryBarrier release = {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    .srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
                                    .oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    .srcQueueFamilyIndex = m_impl->queueFamily,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
                                    .image               = SOURCE.image,
                                    .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(SLOT.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &release);
    if (!checkResult(vkEndCommandBuffer(SLOT.command), "ending a copy command buffer"))
        return false;

    VkSemaphore WAITSEM = SLOT.incoming;
    // ALL_COMMANDS is the only stage that synchronizes with the QFOT acquire below: acquire and
    // release do not run in a defined stage, so a wait at TRANSFER would not order it.
    VkPipelineStageFlags WAITSTAGE = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo         submit    = {.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .waitSemaphoreCount   = gpuWait ? 1u : 0u,
                                      .pWaitSemaphores      = gpuWait ? &WAITSEM : nullptr,
                                      .pWaitDstStageMask    = gpuWait ? &WAITSTAGE : nullptr,
                                      .commandBufferCount   = 1,
                                      .pCommandBuffers      = &SLOT.command,
                                      .signalSemaphoreCount = 1,
                                      .pSignalSemaphores    = &SLOT.outgoing};
    // A fence cannot be signalled by hand: a failed reset, or a submit that failed after one, leaves
    // a slot nothing will signal, so rebuild rather than stall on it three copies later.
    const VkResult RESETSTATE = vkResetFences(m_impl->device, 1, &SLOT.fence);
    if (RESETSTATE != VK_SUCCESS) {
        checkResult(RESETSTATE, "resetting a copy slot fence");
        m_bad = true;
        return false;
    }

    const VkResult SUBMITSTATE = vkQueueSubmit(m_impl->queue, 1, &submit, SLOT.fence);
    if (!checkResult(SUBMITSTATE, "submitting the copy")) {
        if (SUBMITSTATE != VK_ERROR_DEVICE_LOST)
            m_bad = true;
        return false;
    }

    // The copy is in flight, so the import belongs to this slot until its fence signals. markSynced
    // records the destination as holding this frame at submit; the KMS in-fence, not the
    // bookkeeping, is what keeps scanout behind the write.
    SLOT.sourceId = SOURCE.id;
    SOURCE.slotsUsing++;
    SOURCE.lastUse = ++m_impl->useCounter;
    mscDst->markSynced();

    VkSemaphoreGetFdInfoKHR info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = SLOT.outgoing, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    int                     fd   = -1;
    if (!m_impl->getSemaphoreFd || m_impl->getSemaphoreFd(m_impl->device, &info, &fd) != VK_SUCCESS) {
        // Exporting the semaphore is what resets it (SYNC_FD is copy transference), so a slot that
        // failed to export must not be signalled again (VUID-vkQueueSubmit-pSignalSemaphores-00067).
        log(AQ_LOG_ERROR, "drm: MSC: exporting the copy completion fence failed, the copy engine will be rebuilt");
        m_bad = true;
        return false;
    }
    outFence = Hyprutils::OS::CFileDescriptor{fd};

    ++m_impl->copies;
    if (gpuWait)
        ++m_impl->gpuWaits;
    else
        ++m_impl->hostWaits;

    ACCOUNT(mode, mode == ECopyMode::Partial ? (size_t)regionCount : 0);

    return true;
}

void CMSCTransfer::logStats() {
    const uint64_t FRAMES = m_impl->damageFull + m_impl->damagePartial + m_impl->damageSkipped;
    const double   MEAN   = m_impl->damageFramePixels ? (double)m_impl->damagePixels / (double)m_impl->damageFramePixels * 100.0 : 0.0;

    log(AQ_LOG_DEBUG,
        std::format("drm: MSC copy stats: {} copies, {} gpu waits, {} host waits, {} slot waits, {} source imports cached", m_impl->copies, m_impl->gpuWaits, m_impl->hostWaits,
                    m_impl->slotWaits, m_impl->sources.size()));
    log(AQ_LOG_DEBUG,
        std::format("drm: MSC damage over {} frames: partial {} full {} skipped {}, mean {:.2f}% of the frame moved, {} damage rects, buckets 0:{} 1%:{} 2%:{} 5%:{} 10%:{} 25%:{} "
                    "50%:{} 100%:{}",
                    FRAMES, m_impl->damagePartial, m_impl->damageFull, m_impl->damageSkipped, MEAN, m_impl->damageRects, m_impl->damageBuckets[0], m_impl->damageBuckets[1],
                    m_impl->damageBuckets[2], m_impl->damageBuckets[3], m_impl->damageBuckets[4], m_impl->damageBuckets[5], m_impl->damageBuckets[6], m_impl->damageBuckets[7]));
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
