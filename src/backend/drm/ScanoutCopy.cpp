#include "ScanoutCopy.hpp"
#include "MSCScanGridSpirv.hpp"
#include "Shared.hpp"

#include <aquamarine/backend/DRM.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
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

    // Beyond this many boxes, or this much of the frame, a full copy is both simpler and no slower:
    // a partial copy is one box per damage rect and a box's rows are contiguous, so a nearly
    // whole-frame box costs about what a whole frame costs.
    constexpr double MSC_DAMAGE_FULL_FRACTION = 0.9;

    // AQ_MSC_DAMAGE picks what the copy follows:
    //   0 | off              copy whole frames
    //   compositor           follow the damage the compositor reports
    //   compositor+grid      follow it, then scan the reported area against the previous frame and
    //                        copy only what actually changed (the default)
    //   grid                 scan the whole frame and ignore the compositor's damage
    // An unknown value is refused at startup.
    enum class EDamageMode {
        Off,
        Compositor,
        CompositorGrid,
        Grid
    };

    EDamageMode parseDamageMode() {
        const char* value = getenv("AQ_MSC_DAMAGE");
        if (!value || !*value)
            return EDamageMode::Compositor; // follow the reported damage; the scan is opt-in
        if (!strcmp(value, "0") || !strcmp(value, "off"))
            return EDamageMode::Off;
        if (!strcmp(value, "1"))
            return EDamageMode::Compositor; // legacy "damage on"
        if (!strcmp(value, "compositor+grid"))
            return EDamageMode::CompositorGrid;
        if (!strcmp(value, "grid"))
            return EDamageMode::Grid;
        return EDamageMode::Compositor;
    }

    // Parsed once: the mode cannot change while the process runs, and the copy path asks for it on
    // every frame.
    EDamageMode damageMode() {
        static const EDamageMode MODE = parseDamageMode();
        return MODE;
    }

    const bool trackDamage = damageMode() != EDamageMode::Off;

    // How often the copy counters are logged: AQ_MSC_STATS=<seconds>, default 60. An unset value
    // means the default; zero, a negative number or an unparsable one turns them off.
    int statsSeconds() {
        const char* value = getenv("AQ_MSC_STATS");
        if (!value)
            return 60;
        const int seconds = atoi(value);
        return seconds > 0 ? seconds : 0;
    }
    const int MSC_STATS_SECONDS = statsSeconds();

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
        VkImageView    view     = VK_NULL_HANDLE;
        uint32_t       format   = 0;
        uint64_t       modifier = 0;
        uint64_t       width = 0, height = 0, stride = 0, offset = 0;
        int            slotsUsing = 0;
        uint64_t       lastUse    = 0;
        uint64_t       id         = 0;

        void           destroy(VkDevice dev) {
            if (view)
                vkDestroyImageView(dev, view, nullptr);
            if (image)
                vkDestroyImage(dev, image, nullptr);
            if (memory)
                vkFreeMemory(dev, memory, nullptr);
            view   = VK_NULL_HANDLE;
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
    // Why a frame was copied whole, and what the tracking asked for on the frames it could be used
    // on: the partial/full counts alone cannot say which of the two decides a frame.
    uint64_t damageFullNoTracking = 0, damageFullSizeMismatch = 0, damageFullUnseeded = 0, damageFullNoDamage = 0, damageFullManyRects = 0, damageFullMuchArea = 0;
    uint64_t damageTrackedFrames = 0, damageOwedPixels = 0, damageTrackedMovedPixels = 0, damageTrackedFramePixels = 0;
    std::chrono::steady_clock::time_point lastStats{}, started{};
    // More than three copies can never be in flight, so a cache of this size always has an
    // evictable entry.
    static constexpr size_t MAXSOURCES = 16;

    // Copies are strictly serialized: one mutex for the whole transfer. Per-output
    // objects if two outputs ever need to copy concurrently.
    std::mutex mutex;

    // Grid damage scan (AQ_MSC_DAMAGE=compositor+grid|grid). The scan runs on a compute queue,
    // the copy stays on the family picked for it, and the two are ordered by the host: the scan's
    // result has to be back before the copy can be recorded.
    uint32_t              scanFamily   = UINT32_MAX;
    VkQueue               scanQueue    = VK_NULL_HANDLE;
    VkCommandPool         scanPool     = VK_NULL_HANDLE;
    VkCommandBuffer       scanCmd      = VK_NULL_HANDLE;
    VkFence               scanFence    = VK_NULL_HANDLE;
    VkSemaphore           scanIncoming = VK_NULL_HANDLE;
    VkSampler             scanSampler  = VK_NULL_HANDLE;
    bool                  scanGuard    = false;

    VkDescriptorSetLayout scanDSL      = VK_NULL_HANDLE;
    VkDescriptorPool      scanDPool    = VK_NULL_HANDLE;
    VkDescriptorSet       scanDS       = VK_NULL_HANDLE;
    VkPipelineLayout      scanLayout   = VK_NULL_HANDLE;
    VkPipeline            scanPipeline = VK_NULL_HANDLE;

    VkImage               shadow     = VK_NULL_HANDLE;
    VkDeviceMemory        shadowMem  = VK_NULL_HANDLE;
    VkImageView           shadowView = VK_NULL_HANDLE;
    uint32_t              shadowW = 0, shadowH = 0, shadowFourcc = 0;
    bool                  shadowFresh = true;

    VkBuffer              scanDirty     = VK_NULL_HANDLE;
    VkDeviceMemory        scanDirtyMem  = VK_NULL_HANDLE;
    void*                 scanDirtyPtr  = nullptr;
    VkDeviceSize          scanDirtySize = 0;
    VkDeviceSize          scanCandsSize = 0;
    VkBuffer              scanCands     = VK_NULL_HANDLE;
    VkDeviceMemory        scanCandsMem  = VK_NULL_HANDLE;
    void*                 scanCandsPtr  = nullptr;

    VkImage               blinkImage = VK_NULL_HANDLE;
    VkDeviceMemory        blinkMem   = VK_NULL_HANDLE;
    VkImageView           blinkView  = VK_NULL_HANDLE;
    uint32_t              blinkW = 0, blinkH = 0, blinkFourcc = 0;
    bool                  blinkEnabled = false;

    uint32_t              cellSize = 64;

    uint64_t              scanFrames = 0, scanCells = 0, scanDirtyCells = 0, scanNs = 0;
};

std::vector<Aquamarine::SMSCDamageRect> Aquamarine::mscMergeCells(const std::vector<SMSCDamageRect>& cells, uint32_t cap, uint32_t frameW, uint32_t frameH) {
    std::vector<SMSCDamageRect> runs;
    std::vector<SMSCDamageRect> out;

    if (cells.empty() || frameW == 0 || frameH == 0)
        return out;

    std::vector<SMSCDamageRect> sorted;
    sorted.reserve(cells.size());
    for (const auto& C : cells) {
        if (C.w <= 0 || C.h <= 0 || C.x < 0 || C.y < 0 || C.x >= (int32_t)frameW || C.y >= (int32_t)frameH)
            continue;
        sorted.push_back({C.x, C.y, std::min<int32_t>(C.w, (int32_t)frameW - C.x), std::min<int32_t>(C.h, (int32_t)frameH - C.y)});
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
    sorted.erase(std::unique(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h; }), sorted.end());

    // Horizontal runs inside each cell row. Cells of one row share a height, so a run that meets a
    // cell of another height starts a new rectangle instead of silently changing the row height.
    for (size_t i = 0; i < sorted.size();) {
        const int32_t y    = sorted[i].y;
        const int32_t rowH = sorted[i].h;
        size_t        j    = i;
        while (j < sorted.size() && sorted[j].y == y)
            ++j;

        for (size_t k = i; k < j;) {
            const int32_t X = sorted[k].x;
            int32_t       w = sorted[k].w;
            size_t        m = k + 1;
            while (m < j && sorted[m].h == rowH && sorted[m].x == X + w && X + w < (int32_t)frameW) {
                w += std::min<int32_t>(sorted[m].w, (int32_t)frameW - (X + w));
                ++m;
            }
            runs.push_back({X, y, w, rowH});
            k = m;
        }
        i = j;
    }

    // Vertical merge: runs that share an x range and touch become one taller rectangle. Free in
    // bytes, fewer copy regions.
    for (size_t a = 0; a < runs.size(); ++a) {
        if (runs[a].h == 0)
            continue;
        for (size_t b = a + 1; b < runs.size(); ++b) {
            if (runs[b].h == 0)
                continue;
            if (runs[b].x == runs[a].x && runs[b].w == runs[a].w && runs[b].y == runs[a].y + runs[a].h) {
                runs[a].h += runs[b].h;
                runs[b].h = 0;
            }
        }
    }

    for (const auto& R : runs) {
        if (R.h > 0)
            out.push_back(R);
    }

    // Cap: merge the pair whose bounding box adds the least area. A bounding box of everything
    // would be simpler and is a superset, but it turns scattered damage into a whole-frame copy.
    while (out.size() > cap && out.size() > 1) {
        uint64_t best = UINT64_MAX;
        size_t   bi = 0, bj = 1;
        for (size_t i = 0; i < out.size(); ++i) {
            for (size_t j = i + 1; j < out.size(); ++j) {
                const int32_t  x0   = std::min(out[i].x, out[j].x);
                const int32_t  y0   = std::min(out[i].y, out[j].y);
                const int32_t  x1   = std::max(out[i].x + out[i].w, out[j].x + out[j].w);
                const int32_t  y1   = std::max(out[i].y + out[i].h, out[j].y + out[j].h);
                const uint64_t area = (uint64_t)(x1 - x0) * (uint64_t)(y1 - y0);
                const uint64_t own  = (uint64_t)out[i].w * out[i].h + (uint64_t)out[j].w * out[j].h;
                if (area - own < best) {
                    best = area - own;
                    bi   = i;
                    bj   = j;
                }
            }
        }

        const SMSCDamageRect A = out[bi], B = out[bj];
        const int32_t        x0 = std::min(A.x, B.x), y0 = std::min(A.y, B.y);
        const int32_t        x1 = std::min<int32_t>(std::max(A.x + A.w, B.x + B.w), (int32_t)frameW);
        const int32_t        y1 = std::min<int32_t>(std::max(A.y + A.h, B.y + B.h), (int32_t)frameH);
        out[bi]                 = {x0, y0, x1 - x0, y1 - y0};
        out.erase(out.begin() + (long)bj);
    }

    return out;
}

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
    m_impl->started = m_impl->lastStats = std::chrono::steady_clock::now();

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

    const EDamageMode DMODE = damageMode();
    const bool        SCAN  = DMODE == EDamageMode::CompositorGrid || DMODE == EDamageMode::Grid;
    m_impl->blinkEnabled =
        [] {
            const char* value = getenv("AQ_MSC_DAMAGE_BLINK");
            return value && *value && std::string{value} != "0";
        }() &&
        trackDamage;
    const bool NEEDCS = SCAN || m_impl->blinkEnabled;

    // The grid scan and the alpha-blended blink pass need a compute-capable family. The copy keeps
    // its own family; the compute pass gets a queue of its own so the two can overlap. Asking for a
    // mode the device cannot run refuses to start, the same way an unavailable copy family does.
    if (NEEDCS) {
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                m_impl->scanFamily = i;
                break;
            }
        }
        if (m_impl->scanFamily == UINT32_MAX) {
            log(AQ_LOG_ERROR, "drm: MSC: AQ_MSC_DAMAGE/BLINK needs a compute queue, and this device has none; refusing to start");
            return;
        }

        if (SCAN) {
            if (const char* CELL = getenv("AQ_MSC_GRID_DAMAGE_CELL")) {
                const int VALUE = atoi(CELL);
                // Any power of two from 8 up works; the ceiling keeps one workgroup's pixel loop sane.
                if (VALUE < 8 || VALUE > 512 || (VALUE & (VALUE - 1)) != 0) {
                    log(AQ_LOG_ERROR, std::format("drm: MSC: AQ_MSC_GRID_DAMAGE_CELL must be a power of two from 8 to 512 (got {}), refusing to start", CELL));
                    return;
                }
                m_impl->cellSize = (uint32_t)VALUE;
            }
        }
    }

    const std::string_view MODENAME = DMODE == EDamageMode::Off ? "off" :
        DMODE == EDamageMode::Grid                              ? "on, whole-frame grid scan" :
        DMODE == EDamageMode::CompositorGrid                    ? "on, compositor damage refined by the grid scan" :
                                                                  "on, from the compositor's damage";
    log(AQ_LOG_DEBUG, std::format("drm: MSC: damage tracking {}", MODENAME));

    // A transfer-only family executes transfer commands only: the copy and its barriers are
    // transfer commands, but vkCmdClearColorImage is not, and it loses the device there.
    log(AQ_LOG_DEBUG, std::format("drm: MSC: copy queue family {} ({})", m_impl->queueFamily, m_impl->useCopyEngine ? "transfer-only, the async copy engine" : "graphics"));

    const float             priorities[2] = {0.0f, 0.0f};
    VkDeviceQueueCreateInfo queues[2]     = {};
    uint32_t                queueCount    = 1;
    const bool              SHAREDFAMILY  = NEEDCS && m_impl->scanFamily == m_impl->queueFamily;
    const uint32_t          SCANQUEUEIDX  = SHAREDFAMILY && families[m_impl->queueFamily].queueCount >= 2 ? 1u : 0u;
    queues[0] = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = m_impl->queueFamily, .queueCount = 1u + SCANQUEUEIDX, .pQueuePriorities = priorities};
    if (NEEDCS && !SHAREDFAMILY) {
        queues[1]  = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = m_impl->scanFamily, .queueCount = 1, .pQueuePriorities = priorities};
        queueCount = 2;
    }
    const char*        devExts[] = {"VK_KHR_external_memory_fd", "VK_KHR_external_semaphore_fd", "VK_EXT_external_memory_dma_buf", "VK_EXT_image_drm_format_modifier",
                                    "VK_EXT_queue_family_foreign"};
    VkDeviceCreateInfo dev       = {.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                    .queueCreateInfoCount    = queueCount,
                                    .pQueueCreateInfos       = queues,
                                    .enabledExtensionCount   = (uint32_t)std::size(devExts),
                                    .ppEnabledExtensionNames = devExts};
    if (vkCreateDevice(m_impl->physical, &dev, nullptr, &m_impl->device) != VK_SUCCESS) {
        log(AQ_LOG_ERROR,
            "drm: MSC: vkCreateDevice failed (needs VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd, VK_EXT_external_memory_dma_buf, "
            "VK_EXT_image_drm_format_modifier and VK_EXT_queue_family_foreign)");
        return;
    }
    vkGetDeviceQueue(m_impl->device, m_impl->queueFamily, 0, &m_impl->queue);
    if (NEEDCS)
        vkGetDeviceQueue(m_impl->device, m_impl->scanFamily, SCANQUEUEIDX, &m_impl->scanQueue);

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

    m_impl->scanGuard = NEEDCS;
    if (NEEDCS && !scanInit()) {
        fprintf(stderr, "MSC: the grid damage pass could not be created; refusing to start\n");
        return;
    }

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
        scanDestroy();
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
                                 .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (m_impl->scanGuard ? (VkImageUsageFlags)VK_IMAGE_USAGE_SAMPLED_BIT : 0u),
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

    VkImageView view = VK_NULL_HANDLE;
    if (m_impl->scanGuard) {
        VkImageViewCreateInfo viewInfo = {.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                          .image            = image,
                                          .viewType         = VK_IMAGE_VIEW_TYPE_2D,
                                          .format           = format,
                                          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (!checkResult(vkCreateImageView(m_impl->device, &viewInfo, nullptr, &view), "creating the source view")) {
            vkDestroyImage(m_impl->device, image, nullptr);
            vkFreeMemory(m_impl->device, srcMem, nullptr);
            return -1;
        }
    }

    SImpl::SSource SOURCE;
    SOURCE.src      = src;
    SOURCE.image    = image;
    SOURCE.view     = view;
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
    if (result != VK_SUCCESS && result != VK_TIMEOUT)
        fprintf(stderr, "MSC: %s failed: VkResult %d\n", what, (int)result);
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

void CMSCTransfer::scanDestroy() {
    const VkDevice DEV = m_impl->device;
    if (!DEV)
        return;

    if (m_impl->scanPipeline) {
        vkDestroyPipeline(DEV, m_impl->scanPipeline, nullptr);
        m_impl->scanPipeline = VK_NULL_HANDLE;
    }
    if (m_impl->scanLayout) {
        vkDestroyPipelineLayout(DEV, m_impl->scanLayout, nullptr);
        m_impl->scanLayout = VK_NULL_HANDLE;
    }
    if (m_impl->scanDPool) {
        vkDestroyDescriptorPool(DEV, m_impl->scanDPool, nullptr);
        m_impl->scanDPool = VK_NULL_HANDLE;
        m_impl->scanDS    = VK_NULL_HANDLE;
    }
    if (m_impl->scanDSL) {
        vkDestroyDescriptorSetLayout(DEV, m_impl->scanDSL, nullptr);
        m_impl->scanDSL = VK_NULL_HANDLE;
    }
    if (m_impl->scanSampler) {
        vkDestroySampler(DEV, m_impl->scanSampler, nullptr);
        m_impl->scanSampler = VK_NULL_HANDLE;
    }
    if (m_impl->scanPool) {
        vkDestroyCommandPool(DEV, m_impl->scanPool, nullptr);
        m_impl->scanPool = VK_NULL_HANDLE;
        m_impl->scanCmd  = VK_NULL_HANDLE;
    }
    if (m_impl->scanFence) {
        vkDestroyFence(DEV, m_impl->scanFence, nullptr);
        m_impl->scanFence = VK_NULL_HANDLE;
    }
    if (m_impl->scanIncoming) {
        vkDestroySemaphore(DEV, m_impl->scanIncoming, nullptr);
        m_impl->scanIncoming = VK_NULL_HANDLE;
    }
    if (m_impl->shadowView) {
        vkDestroyImageView(DEV, m_impl->shadowView, nullptr);
        m_impl->shadowView = VK_NULL_HANDLE;
    }
    if (m_impl->shadow) {
        vkDestroyImage(DEV, m_impl->shadow, nullptr);
        m_impl->shadow = VK_NULL_HANDLE;
    }
    if (m_impl->shadowMem) {
        vkFreeMemory(DEV, m_impl->shadowMem, nullptr);
        m_impl->shadowMem = VK_NULL_HANDLE;
    }
    m_impl->shadowW = m_impl->shadowH = m_impl->shadowFourcc = 0;
    m_impl->shadowFresh                                      = true;

    for (VkBuffer* B : {&m_impl->scanDirty, &m_impl->scanCands}) {
        if (*B) {
            vkDestroyBuffer(DEV, *B, nullptr);
            *B = VK_NULL_HANDLE;
        }
    }
    m_impl->scanDirtyPtr  = nullptr;
    m_impl->scanCandsPtr  = nullptr;
    m_impl->scanDirtySize = 0;
    m_impl->scanCandsSize = 0;
    if (m_impl->scanDirtyMem) {
        vkFreeMemory(DEV, m_impl->scanDirtyMem, nullptr);
        m_impl->scanDirtyMem = VK_NULL_HANDLE;
    }
    if (m_impl->scanCandsMem) {
        vkFreeMemory(DEV, m_impl->scanCandsMem, nullptr);
        m_impl->scanCandsMem = VK_NULL_HANDLE;
    }
    if (m_impl->blinkView) {
        vkDestroyImageView(DEV, m_impl->blinkView, nullptr);
        m_impl->blinkView = VK_NULL_HANDLE;
    }
    if (m_impl->blinkImage) {
        vkDestroyImage(DEV, m_impl->blinkImage, nullptr);
        m_impl->blinkImage = VK_NULL_HANDLE;
    }
    if (m_impl->blinkMem) {
        vkFreeMemory(DEV, m_impl->blinkMem, nullptr);
        m_impl->blinkMem = VK_NULL_HANDLE;
    }
    m_impl->blinkW = m_impl->blinkH = m_impl->blinkFourcc = 0;
}

bool CMSCTransfer::scanInit() {
    const VkDevice           DEV = m_impl->device;

    VkShaderModuleCreateInfo sInfo  = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof(MSC_SCAN_GRID_SPIRV), .pCode = MSC_SCAN_GRID_SPIRV};
    VkShaderModule           module = VK_NULL_HANDLE;
    if (!checkResult(vkCreateShaderModule(DEV, &sInfo, nullptr, &module), "creating the grid scan shader"))
        return false;

    VkSamplerCreateInfo          samp = {.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                         .magFilter    = VK_FILTER_NEAREST,
                                         .minFilter    = VK_FILTER_NEAREST,
                                         .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                         .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                         .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};

    VkDescriptorSetLayoutBinding bindings[4] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dsl   = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = bindings};
    VkPushConstantRange             range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 6 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo      pl    = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &m_impl->scanDSL, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VkComputePipelineCreateInfo cp = {
        .sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage              = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex  = -1};
    VkDescriptorPoolSize        sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo  dpi      = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 3, .pPoolSizes = sizes};
    VkDescriptorSetAllocateInfo dsai     = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorSetCount = 1, .pSetLayouts = &m_impl->scanDSL};
    VkCommandPoolCreateInfo     cpi      = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = m_impl->scanFamily};
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkFenceCreateInfo           fci  = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkSemaphoreCreateInfo       sci  = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    bool                        OK = checkResult(vkCreateSampler(DEV, &samp, nullptr, &m_impl->scanSampler), "creating the grid scan sampler") &&
        checkResult(vkCreateDescriptorSetLayout(DEV, &dsl, nullptr, &m_impl->scanDSL), "creating the grid scan set layout") &&
        checkResult(vkCreatePipelineLayout(DEV, &pl, nullptr, &m_impl->scanLayout), "creating the grid scan pipeline layout") &&
        checkResult(vkCreateDescriptorPool(DEV, &dpi, nullptr, &m_impl->scanDPool), "creating the grid scan descriptor pool");
    // The pool handle is a value in this struct, so it has to be filled in after the pool exists.
    dsai.descriptorPool = m_impl->scanDPool;
    OK                  = OK && checkResult(vkAllocateDescriptorSets(DEV, &dsai, &m_impl->scanDS), "allocating the grid scan descriptor set") &&
        checkResult(vkCreateCommandPool(DEV, &cpi, nullptr, &m_impl->scanPool), "creating the grid scan command pool");
    // The pool handle is a value here too: fill it in once the pool exists.
    cbai.commandPool = m_impl->scanPool;
    OK               = OK && checkResult(vkAllocateCommandBuffers(DEV, &cbai, &m_impl->scanCmd), "allocating the grid scan command buffer") &&
        checkResult(vkCreateFence(DEV, &fci, nullptr, &m_impl->scanFence), "creating the grid scan fence") &&
        checkResult(vkCreateSemaphore(DEV, &sci, nullptr, &m_impl->scanIncoming), "creating the grid scan incoming semaphore");

    bool ok = OK;
    if (ok) {
        cp.layout = m_impl->scanLayout;
        ok        = checkResult(vkCreateComputePipelines(DEV, VK_NULL_HANDLE, 1, &cp, nullptr, &m_impl->scanPipeline), "creating the grid scan pipeline");
    }
    vkDestroyShaderModule(DEV, module, nullptr);
    if (!ok) {
        scanDestroy();
        return false;
    }
    return true;
}

bool CMSCTransfer::ensureBlink(const Hyprutils::Math::CRegion& paint, uint32_t width, uint32_t height, uint32_t fourcc, VkImage srcImage, VkImageView srcView, int waitFd,
                               bool& waitedOnProducer) {
    if (!m_impl->scanPipeline || !srcImage || !srcView)
        return false;

    const VkDevice DEV = m_impl->device;
    // Wait for any in-flight copy slots that may still be reading blinkImage on the transfer queue
    // before resizing or writing to it on the compute queue.
    for (auto& s : m_impl->slots) {
        if (vkGetFenceStatus(DEV, s.fence) != VK_SUCCESS) {
            if (!checkResult(vkWaitForFences(DEV, 1, &s.fence, VK_TRUE, 100000000), "waiting for in-flight copies before updating the blink frame"))
                return false;
        }
    }

    if (!m_impl->blinkImage || m_impl->blinkW != width || m_impl->blinkH != height || m_impl->blinkFourcc != fourcc) {
        if (m_impl->blinkView) {
            vkDestroyImageView(DEV, m_impl->blinkView, nullptr);
            m_impl->blinkView = VK_NULL_HANDLE;
        }
        if (m_impl->blinkImage) {
            vkDestroyImage(DEV, m_impl->blinkImage, nullptr);
            m_impl->blinkImage = VK_NULL_HANDLE;
        }
        if (m_impl->blinkMem) {
            vkFreeMemory(DEV, m_impl->blinkMem, nullptr);
            m_impl->blinkMem = VK_NULL_HANDLE;
        }

        const bool        CONCURRENT       = m_impl->scanFamily != UINT32_MAX && m_impl->scanFamily != m_impl->queueFamily;
        const uint32_t    blinkFamilies[2] = {m_impl->queueFamily, m_impl->scanFamily};
        VkImageCreateInfo ci               = {.sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                              .imageType             = VK_IMAGE_TYPE_2D,
                                              .format                = VK_FORMAT_R8G8B8A8_UNORM,
                                              .extent                = {width, height, 1},
                                              .mipLevels             = 1,
                                              .arrayLayers           = 1,
                                              .samples               = VK_SAMPLE_COUNT_1_BIT,
                                              .tiling                = VK_IMAGE_TILING_OPTIMAL,
                                              .usage                 = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                              .sharingMode           = CONCURRENT ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
                                              .queueFamilyIndexCount = CONCURRENT ? 2u : 0u,
                                              .pQueueFamilyIndices   = CONCURRENT ? blinkFamilies : nullptr,
                                              .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED};
        if (!checkResult(vkCreateImage(DEV, &ci, nullptr, &m_impl->blinkImage), "creating the blink frame"))
            return false;

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(DEV, m_impl->blinkImage, &req);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < m_impl->memProps.memoryTypeCount; ++i) {
            const auto FLAGS = m_impl->memProps.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) && (FLAGS & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(FLAGS & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX) {
            log(AQ_LOG_ERROR, "drm: MSC: no device-local memory for the blink frame");
            return false;
        }
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = type};
        if (!checkResult(vkAllocateMemory(DEV, &ai, nullptr, &m_impl->blinkMem), "allocating the blink frame") ||
            !checkResult(vkBindImageMemory(DEV, m_impl->blinkImage, m_impl->blinkMem, 0), "binding the blink frame"))
            return false;

        VkImageViewCreateInfo vi = {.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                    .image            = m_impl->blinkImage,
                                    .viewType         = VK_IMAGE_VIEW_TYPE_2D,
                                    .format           = VK_FORMAT_R8G8B8A8_UNORM,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (!checkResult(vkCreateImageView(DEV, &vi, nullptr, &m_impl->blinkView), "creating the blink frame view"))
            return false;

        m_impl->blinkW      = width;
        m_impl->blinkH      = height;
        m_impl->blinkFourcc = fourcc;
    }

    if (paint.empty())
        return true;

    uint32_t blinkMode = 2u;
    switch (mscVkFormat(fourcc)) {
        case VK_FORMAT_B8G8R8A8_UNORM: blinkMode = 2u; break;
        case VK_FORMAT_R8G8B8A8_UNORM: blinkMode = 3u; break;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32: blinkMode = 4u; break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: blinkMode = 5u; break;
        default: return false;
    }

    const uint32_t     CELL  = 16;
    const uint32_t     COLS  = (width + CELL - 1) / CELL;
    const uint32_t     ROWS  = (height + CELL - 1) / CELL;
    const VkDeviceSize DIRTY = (VkDeviceSize)COLS * (VkDeviceSize)ROWS * sizeof(uint32_t);

    const auto         HOSTBUFFER = [&](VkBuffer& buf, VkDeviceMemory& mem, void*& ptr, VkDeviceSize bytes, VkDeviceSize& have) {
        if (buf && have >= bytes)
            return true;
        if (buf) {
            vkDestroyBuffer(DEV, buf, nullptr);
            buf = VK_NULL_HANDLE;
        }
        if (mem) {
            vkFreeMemory(DEV, mem, nullptr);
            mem = VK_NULL_HANDLE;
        }
        ptr                   = nullptr;
        VkBufferCreateInfo bi = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(DEV, &bi, nullptr, &buf) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(DEV, buf, &req);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < m_impl->memProps.memoryTypeCount; ++i) {
            const auto FLAGS = m_impl->memProps.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) && (FLAGS & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (FLAGS & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX)
            return false;
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = type};
        if (vkAllocateMemory(DEV, &ai, nullptr, &mem) != VK_SUCCESS || vkBindBufferMemory(DEV, buf, mem, 0) != VK_SUCCESS)
            return false;
        if (vkMapMemory(DEV, mem, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS)
            return false;
        have = bytes;
        return true;
    };
    if (!HOSTBUFFER(m_impl->scanDirty, m_impl->scanDirtyMem, m_impl->scanDirtyPtr, DIRTY, m_impl->scanDirtySize))
        return false;
    if (!HOSTBUFFER(m_impl->scanCands, m_impl->scanCandsMem, m_impl->scanCandsPtr, 16 * 4 * sizeof(uint32_t), m_impl->scanCandsSize))
        return false;

    struct SCand {
        uint32_t cx, cy, cw, ch;
    };
    SCand                    cands[16] = {};
    uint32_t                 ncands = 0, total = 0;
    int                      cnt   = 0;
    const pixman_box32_t*    boxes = pixman_region32_rectangles(paint.pixman(), &cnt);
    Hyprutils::Math::CRegion cellRegion;
    for (int i = 0; i < cnt; ++i) {
        const int32_t x0 = std::max<int32_t>(0, boxes[i].x1), y0 = std::max<int32_t>(0, boxes[i].y1);
        const int32_t x1 = std::min<int32_t>(boxes[i].x2, (int32_t)width), y1 = std::min<int32_t>(boxes[i].y2, (int32_t)height);
        if (x1 <= x0 || y1 <= y0)
            continue;
        const uint32_t CX = (uint32_t)x0 / CELL, CY = (uint32_t)y0 / CELL;
        const uint32_t CW = ((uint32_t)x1 - 1) / CELL - CX + 1, CH = ((uint32_t)y1 - 1) / CELL - CY + 1;
        cellRegion.add(Hyprutils::Math::CRegion(CX, CY, CW, CH));
    }
    int                   cellCnt   = 0;
    const pixman_box32_t* cellBoxes = pixman_region32_rectangles(cellRegion.pixman(), &cellCnt);
    if (cellCnt > 16) {
        const pixman_box32_t* ext = pixman_region32_extents(cellRegion.pixman());
        if (ext->x2 > ext->x1 && ext->y2 > ext->y1)
            cands[ncands++] = {(uint32_t)ext->x1, (uint32_t)ext->y1, (uint32_t)(ext->x2 - ext->x1), (uint32_t)(ext->y2 - ext->y1)};
    } else {
        for (int i = 0; i < cellCnt; ++i)
            cands[ncands++] = {(uint32_t)cellBoxes[i].x1, (uint32_t)cellBoxes[i].y1, (uint32_t)(cellBoxes[i].x2 - cellBoxes[i].x1), (uint32_t)(cellBoxes[i].y2 - cellBoxes[i].y1)};
    }
    for (uint32_t i = 0; i < ncands; ++i)
        total += cands[i].cw * cands[i].ch;
    if (!total)
        return true;

    bool gpuWait = false;
    if (waitFd >= 0 && !waitedOnProducer) {
        const int IMPORTFD = m_impl->importSemaphore ? dup(waitFd) : -1;
        if (IMPORTFD >= 0) {
            VkImportSemaphoreFdInfoKHR import = {.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                                                 .semaphore  = m_impl->scanIncoming,
                                                 .flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                                 .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
                                                 .fd         = IMPORTFD};
            gpuWait                           = m_impl->importSemaphore(DEV, &import) == VK_SUCCESS;
            if (!gpuWait)
                close(IMPORTFD);
        }
        if (!gpuWait) {
            struct pollfd p = {.fd = waitFd, .events = POLLIN};
            while (poll(&p, 1, 1000) == -1 && errno == EINTR) {}
            if (!(p.revents & POLLIN))
                return false;
            waitedOnProducer = true;
            ++m_impl->hostWaits;
        }
    }

    memcpy(m_impl->scanCandsPtr, cands, sizeof(cands));

    VkDescriptorImageInfo  srcInfo   = {.sampler = m_impl->scanSampler, .imageView = srcView, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo  blinkInfo = {.imageView = m_impl->blinkView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dirtyInfo = {.buffer = m_impl->scanDirty, .offset = 0, .range = DIRTY};
    VkDescriptorBufferInfo candInfo  = {.buffer = m_impl->scanCands, .offset = 0, .range = sizeof(cands)};
    VkWriteDescriptorSet   writes[4] = {
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 0,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo      = &srcInfo},
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 1,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo      = &blinkInfo},
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 2,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo     = &dirtyInfo},
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 3,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo     = &candInfo},
    };
    vkUpdateDescriptorSets(DEV, 4, writes, 0, nullptr);

    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (!checkResult(vkResetCommandBuffer(m_impl->scanCmd, 0), "resetting the blink command buffer") ||
        !checkResult(vkBeginCommandBuffer(m_impl->scanCmd, &bi), "beginning the blink command buffer"))
        return false;

    VkImageMemoryBarrier acquires[2] = {
        {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask       = 0,
         .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
         .dstQueueFamilyIndex = m_impl->scanFamily,
         .image               = srcImage,
         .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask       = 0,
         .dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
         .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image               = m_impl->blinkImage,
         .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
    };
    vkCmdPipelineBarrier(m_impl->scanCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, acquires);

    vkCmdBindPipeline(m_impl->scanCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->scanPipeline);
    vkCmdBindDescriptorSets(m_impl->scanCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->scanLayout, 0, 1, &m_impl->scanDS, 0, nullptr);
    const uint32_t push[6] = {CELL, COLS, ROWS, blinkMode, width, height};
    vkCmdPushConstants(m_impl->scanCmd, m_impl->scanLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    VkPhysicalDeviceProperties SCANPROPS;
    vkGetPhysicalDeviceProperties(m_impl->physical, &SCANPROPS);
    if (total > SCANPROPS.limits.maxComputeWorkGroupCount[0]) {
        log(AQ_LOG_DEBUG, "drm: MSC: the grid exceeds one dispatch; skipping the pass on this frame");
        return false;
    }
    vkCmdDispatch(m_impl->scanCmd, total, 1, 1);

    VkImageMemoryBarrier releases[2] = {
        {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask       = VK_ACCESS_SHADER_READ_BIT,
         .dstAccessMask       = 0,
         .oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = m_impl->scanFamily,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
         .image               = srcImage,
         .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask       = 0,
         .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image               = m_impl->blinkImage,
         .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
    };
    vkCmdPipelineBarrier(m_impl->scanCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 2, releases);

    if (!checkResult(vkEndCommandBuffer(m_impl->scanCmd), "ending the blink command buffer"))
        return false;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo         submit    = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .waitSemaphoreCount = gpuWait ? 1u : 0u,
                                      .pWaitSemaphores    = gpuWait ? &m_impl->scanIncoming : nullptr,
                                      .pWaitDstStageMask  = gpuWait ? &waitStage : nullptr,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers    = &m_impl->scanCmd};
    if (!checkResult(vkQueueSubmit(m_impl->scanQueue, 1, &submit, m_impl->scanFence), "submitting the blink pass"))
        return false;
    const VkResult BLINKWAIT = vkWaitForFences(DEV, 1, &m_impl->scanFence, VK_TRUE, 100000000);
    if (BLINKWAIT != VK_SUCCESS) {
        // The pass has to retire before its command buffer, its fence or the source image's
        // ownership can be touched again, so a slow pass gets one longer bound and is only then
        // skipped. A pass that never finishes is a lost engine.
        const VkResult RETIRE = BLINKWAIT == VK_TIMEOUT ? vkWaitForFences(DEV, 1, &m_impl->scanFence, VK_TRUE, 1000000000) : BLINKWAIT;
        if (RETIRE != VK_SUCCESS) {
            checkResult(RETIRE, "waiting for the blink pass");
            m_bad = true;
            return false;
        }
        log(AQ_LOG_DEBUG, "drm: MSC: the blink pass was slow; skipping it for this frame");
    }
    if (!checkResult(vkResetFences(DEV, 1, &m_impl->scanFence), "resetting the blink fence"))
        return false;
    if (gpuWait) {
        waitedOnProducer = true;
        ++m_impl->gpuWaits;
    }
    return true;
}

bool CMSCTransfer::scanRun(const Hyprutils::Math::CRegion* hint, uint32_t width, uint32_t height, uint32_t fourcc, VkImage srcImage, VkImageView srcView, int waitFd,
                           bool& waitedOnProducer, std::vector<SMSCDamageRect>& out) {
    out.clear();
    const VkDevice     DEV   = m_impl->device;
    const uint32_t     CELL  = m_impl->cellSize;
    const uint32_t     COLS  = (width + CELL - 1) / CELL;
    const uint32_t     ROWS  = (height + CELL - 1) / CELL;
    const VkDeviceSize DIRTY = (VkDeviceSize)COLS * (VkDeviceSize)ROWS * sizeof(uint32_t);

    // The shadow is the frame we scanned last. It is (re)created with the frame and starts out
    // "fresh": the first scan stores the source without comparing, so it can never miss a change.
    if (m_impl->shadowW != width || m_impl->shadowH != height || m_impl->shadowFourcc != fourcc) {
        for (auto& s : m_impl->slots) {
            if (vkGetFenceStatus(DEV, s.fence) != VK_SUCCESS) {
                if (!checkResult(vkWaitForFences(DEV, 1, &s.fence, VK_TRUE, 100000000), "waiting for in-flight copies before resizing the damage shadow"))
                    return false;
            }
        }
        if (m_impl->shadowView) {
            vkDestroyImageView(DEV, m_impl->shadowView, nullptr);
            m_impl->shadowView = VK_NULL_HANDLE;
        }
        if (m_impl->shadow) {
            vkDestroyImage(DEV, m_impl->shadow, nullptr);
            m_impl->shadow = VK_NULL_HANDLE;
        }
        if (m_impl->shadowMem) {
            vkFreeMemory(DEV, m_impl->shadowMem, nullptr);
            m_impl->shadowMem = VK_NULL_HANDLE;
        }

        VkImageCreateInfo ci = {.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                .imageType     = VK_IMAGE_TYPE_2D,
                                .format        = VK_FORMAT_R8G8B8A8_UNORM,
                                .extent        = {width, height, 1},
                                .mipLevels     = 1,
                                .arrayLayers   = 1,
                                .samples       = VK_SAMPLE_COUNT_1_BIT,
                                .tiling        = VK_IMAGE_TILING_OPTIMAL,
                                .usage         = VK_IMAGE_USAGE_STORAGE_BIT,
                                .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        if (!checkResult(vkCreateImage(DEV, &ci, nullptr, &m_impl->shadow), "creating the damage shadow"))
            return false;

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(DEV, m_impl->shadow, &req);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < m_impl->memProps.memoryTypeCount; ++i) {
            const auto FLAGS = m_impl->memProps.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) && (FLAGS & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(FLAGS & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX) {
            log(AQ_LOG_ERROR, "drm: MSC: no device-local memory for the damage shadow");
            return false;
        }
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = type};
        if (!checkResult(vkAllocateMemory(DEV, &ai, nullptr, &m_impl->shadowMem), "allocating the damage shadow") ||
            !checkResult(vkBindImageMemory(DEV, m_impl->shadow, m_impl->shadowMem, 0), "binding the damage shadow"))
            return false;

        VkImageViewCreateInfo vi = {.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                    .image            = m_impl->shadow,
                                    .viewType         = VK_IMAGE_VIEW_TYPE_2D,
                                    .format           = VK_FORMAT_R8G8B8A8_UNORM,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (!checkResult(vkCreateImageView(DEV, &vi, nullptr, &m_impl->shadowView), "creating the damage shadow view"))
            return false;

        m_impl->shadowW      = width;
        m_impl->shadowH      = height;
        m_impl->shadowFourcc = fourcc;
        m_impl->shadowFresh  = true;
    }

    // Host-visible scratch: one dirty word per cell plus the candidate rectangles.
    const auto HOSTBUFFER = [&](VkBuffer& buf, VkDeviceMemory& mem, void*& ptr, VkDeviceSize bytes, VkDeviceSize& have) {
        if (buf && have >= bytes)
            return true;
        if (buf) {
            vkDestroyBuffer(DEV, buf, nullptr);
            buf = VK_NULL_HANDLE;
        }
        if (mem) {
            vkFreeMemory(DEV, mem, nullptr);
            mem = VK_NULL_HANDLE;
        }
        ptr                   = nullptr;
        VkBufferCreateInfo bi = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(DEV, &bi, nullptr, &buf) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(DEV, buf, &req);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < m_impl->memProps.memoryTypeCount; ++i) {
            const auto FLAGS = m_impl->memProps.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) && (FLAGS & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (FLAGS & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX)
            return false;
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = type};
        if (vkAllocateMemory(DEV, &ai, nullptr, &mem) != VK_SUCCESS || vkBindBufferMemory(DEV, buf, mem, 0) != VK_SUCCESS)
            return false;
        if (vkMapMemory(DEV, mem, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS)
            return false;
        have = bytes;
        return true;
    };
    if (!HOSTBUFFER(m_impl->scanDirty, m_impl->scanDirtyMem, m_impl->scanDirtyPtr, DIRTY, m_impl->scanDirtySize))
        return false;
    if (!HOSTBUFFER(m_impl->scanCands, m_impl->scanCandsMem, m_impl->scanCandsPtr, 16 * 4 * sizeof(uint32_t), m_impl->scanCandsSize))
        return false;

    // Candidate cells: the cell rectangles the hint covers, or the whole grid when seeding or in grid mode.
    struct SCand {
        uint32_t cx, cy, cw, ch;
    };
    SCand    cands[16] = {};
    uint32_t ncands = 0, total = 0;
    if (!m_impl->shadowFresh && hint) {
        if (hint->empty())
            return true;
        int                      cnt   = 0;
        const pixman_box32_t*    boxes = pixman_region32_rectangles(hint->pixman(), &cnt);
        Hyprutils::Math::CRegion cellRegion;
        for (int i = 0; i < cnt; ++i) {
            const int32_t x0 = std::max<int32_t>(0, boxes[i].x1), y0 = std::max<int32_t>(0, boxes[i].y1);
            const int32_t x1 = std::min<int32_t>(boxes[i].x2, (int32_t)width), y1 = std::min<int32_t>(boxes[i].y2, (int32_t)height);
            if (x1 <= x0 || y1 <= y0)
                continue;
            const uint32_t CX = (uint32_t)x0 / CELL, CY = (uint32_t)y0 / CELL;
            const uint32_t CW = ((uint32_t)x1 - 1) / CELL - CX + 1, CH = ((uint32_t)y1 - 1) / CELL - CY + 1;
            cellRegion.add(Hyprutils::Math::CRegion(CX, CY, CW, CH));
        }
        int                   cellCnt   = 0;
        const pixman_box32_t* cellBoxes = pixman_region32_rectangles(cellRegion.pixman(), &cellCnt);
        if (cellCnt > 16) {
            const pixman_box32_t* ext = pixman_region32_extents(cellRegion.pixman());
            if (ext->x2 > ext->x1 && ext->y2 > ext->y1)
                cands[ncands++] = {(uint32_t)ext->x1, (uint32_t)ext->y1, (uint32_t)(ext->x2 - ext->x1), (uint32_t)(ext->y2 - ext->y1)};
        } else {
            for (int i = 0; i < cellCnt; ++i)
                cands[ncands++] = {(uint32_t)cellBoxes[i].x1, (uint32_t)cellBoxes[i].y1, (uint32_t)(cellBoxes[i].x2 - cellBoxes[i].x1),
                                   (uint32_t)(cellBoxes[i].y2 - cellBoxes[i].y1)};
        }
    } else
        cands[ncands++] = {0, 0, COLS, ROWS};
    for (uint32_t i = 0; i < ncands; ++i)
        total += cands[i].cw * cands[i].ch;
    if (!total)
        return true;

    bool gpuWait = false;
    if (waitFd >= 0 && !waitedOnProducer) {
        const int IMPORTFD = m_impl->importSemaphore ? dup(waitFd) : -1;
        if (IMPORTFD >= 0) {
            VkImportSemaphoreFdInfoKHR import = {.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                                                 .semaphore  = m_impl->scanIncoming,
                                                 .flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                                 .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
                                                 .fd         = IMPORTFD};
            gpuWait                           = m_impl->importSemaphore(DEV, &import) == VK_SUCCESS;
            if (!gpuWait)
                close(IMPORTFD);
        }
        if (!gpuWait) {
            struct pollfd p = {.fd = waitFd, .events = POLLIN};
            while (poll(&p, 1, 1000) == -1 && errno == EINTR) {}
            if (!(p.revents & POLLIN))
                return false;
            waitedOnProducer = true;
            ++m_impl->hostWaits;
        }
    }

    memset(m_impl->scanDirtyPtr, 0, (size_t)DIRTY);
    memcpy(m_impl->scanCandsPtr, cands, sizeof(cands));

    VkDescriptorImageInfo  srcInfo    = {.sampler = m_impl->scanSampler, .imageView = srcView, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo  shadowInfo = {.imageView = m_impl->shadowView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dirtyInfo  = {.buffer = m_impl->scanDirty, .offset = 0, .range = DIRTY};
    VkDescriptorBufferInfo candInfo   = {.buffer = m_impl->scanCands, .offset = 0, .range = sizeof(cands)};
    VkWriteDescriptorSet   writes[4]  = {
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 0,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo      = &srcInfo},
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 1,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo      = &shadowInfo},
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 2,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo     = &dirtyInfo},
        {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet          = m_impl->scanDS,
         .dstBinding      = 3,
         .descriptorCount = 1,
         .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo     = &candInfo},
    };
    vkUpdateDescriptorSets(DEV, 4, writes, 0, nullptr);

    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (!checkResult(vkResetCommandBuffer(m_impl->scanCmd, 0), "resetting the grid scan command buffer") ||
        !checkResult(vkBeginCommandBuffer(m_impl->scanCmd, &bi), "beginning the grid scan command buffer"))
        return false;

    VkImageMemoryBarrier acquires[2] = {
        {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask       = 0,
         .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
         .dstQueueFamilyIndex = m_impl->scanFamily,
         .image               = srcImage,
         .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask       = m_impl->shadowFresh ? 0u : (VkAccessFlags)VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
         .oldLayout           = m_impl->shadowFresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
         .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image               = m_impl->shadow,
         .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
    };
    vkCmdPipelineBarrier(m_impl->scanCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         2, acquires);

    vkCmdBindPipeline(m_impl->scanCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->scanPipeline);
    vkCmdBindDescriptorSets(m_impl->scanCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->scanLayout, 0, 1, &m_impl->scanDS, 0, nullptr);
    const uint32_t push[6] = {CELL, COLS, ROWS, m_impl->shadowFresh ? 1u : 0u, width, height};
    vkCmdPushConstants(m_impl->scanCmd, m_impl->scanLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    VkPhysicalDeviceProperties SCANPROPS;
    vkGetPhysicalDeviceProperties(m_impl->physical, &SCANPROPS);
    if (total > SCANPROPS.limits.maxComputeWorkGroupCount[0]) {
        log(AQ_LOG_DEBUG, "drm: MSC: the grid exceeds one dispatch; skipping the pass on this frame");
        return false;
    }
    vkCmdDispatch(m_impl->scanCmd, total, 1, 1);

    VkImageMemoryBarrier  release  = {.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                      .srcAccessMask       = VK_ACCESS_SHADER_READ_BIT,
                                      .dstAccessMask       = 0,
                                      .oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      .newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      .srcQueueFamilyIndex = m_impl->scanFamily,
                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
                                      .image               = srcImage,
                                      .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkBufferMemoryBarrier dirtyOut = {.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                      .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
                                      .dstAccessMask       = VK_ACCESS_HOST_READ_BIT,
                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                      .buffer              = m_impl->scanDirty,
                                      .offset              = 0,
                                      .size                = DIRTY};
    vkCmdPipelineBarrier(m_impl->scanCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &release);
    vkCmdPipelineBarrier(m_impl->scanCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &dirtyOut, 0, nullptr);

    if (!checkResult(vkEndCommandBuffer(m_impl->scanCmd), "ending the grid scan command buffer"))
        return false;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo         submit    = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .waitSemaphoreCount = gpuWait ? 1u : 0u,
                                      .pWaitSemaphores    = gpuWait ? &m_impl->scanIncoming : nullptr,
                                      .pWaitDstStageMask  = gpuWait ? &waitStage : nullptr,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers    = &m_impl->scanCmd};
    const auto           BEGAN     = std::chrono::steady_clock::now();
    if (!checkResult(vkQueueSubmit(m_impl->scanQueue, 1, &submit, m_impl->scanFence), "submitting the grid scan"))
        return false;
    const VkResult SCANWAIT = vkWaitForFences(DEV, 1, &m_impl->scanFence, VK_TRUE, 100000000);
    if (SCANWAIT != VK_SUCCESS) {
        // Same rule as the blink pass: retire it before anything reuses the command buffer, the
        // fence or the source image's ownership.
        const VkResult RETIRE = SCANWAIT == VK_TIMEOUT ? vkWaitForFences(DEV, 1, &m_impl->scanFence, VK_TRUE, 1000000000) : SCANWAIT;
        if (RETIRE != VK_SUCCESS) {
            checkResult(RETIRE, "waiting for the grid scan");
            m_bad = true;
            return false;
        }
        log(AQ_LOG_DEBUG, "drm: MSC: the grid scan was slow; skipping it for this frame");
    }
    if (!checkResult(vkResetFences(DEV, 1, &m_impl->scanFence), "resetting the grid scan fence"))
        return false;
    if (gpuWait) {
        waitedOnProducer = true;
        ++m_impl->gpuWaits;
    }

    m_impl->scanNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - BEGAN).count();
    m_impl->scanFrames++;
    m_impl->scanCells += total;
    m_impl->shadowFresh = false;

    const uint32_t*             DIRTYWORDS = (const uint32_t*)m_impl->scanDirtyPtr;
    std::vector<SMSCDamageRect> cells;
    uint32_t                    dirtyCells = 0;
    for (uint32_t i = 0; i < ncands; ++i) {
        for (uint32_t y = 0; y < cands[i].ch; ++y) {
            for (uint32_t x = 0; x < cands[i].cw; ++x) {
                const uint32_t CX = cands[i].cx + x, CY = cands[i].cy + y;
                if (CX >= COLS || CY >= ROWS)
                    continue;
                if (!DIRTYWORDS[CY * COLS + CX])
                    continue;
                ++dirtyCells;
                const int32_t X0 = (int32_t)(CX * CELL), Y0 = (int32_t)(CY * CELL);
                const int32_t W = std::min<int32_t>((int32_t)CELL, (int32_t)width - X0), H = std::min<int32_t>((int32_t)CELL, (int32_t)height - Y0);
                cells.push_back({X0, Y0, W, H});
            }
        }
    }
    m_impl->scanDirtyCells += dirtyCells;
    // The list is bounded by the candidate cells and by a command-buffer budget, not by a fixed
    // count: one rectangle per candidate cell is the finest split the scan can produce.
    const size_t GRIDBOUND = (size_t)((width + CELL - 1) / CELL) * (size_t)((height + CELL - 1) / CELL);
    out                    = mscMergeCells(cells, (uint32_t)std::clamp<size_t>(GRIDBOUND, 64, ((size_t)1 << 20) / sizeof(VkBufferImageCopy)), width, height);
    return true;
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
    // the same thing on both sides. Cursor plane copies arrive from the main thread with no damage
    // or waitFd and must not resize the primary plane's shadow or blink images.
    const bool SAMESIZE  = WIDTH == (uint32_t)mscDst->size.x && HEIGHT == (uint32_t)mscDst->size.y;
    const bool ISCURSOR  = !damage && waitFd < 0 && WIDTH <= 256 && HEIGHT <= 256;
    const bool TRACKABLE = SAMESIZE && !ISCURSOR;

    // The damage bookkeeping below and the Vulkan recording further down both touch the transfer's
    // shared state, so take the copy lock before either.
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    for (auto& s : m_impl->slots) {
        if (s.sourceId && vkGetFenceStatus(m_impl->device, s.fence) == VK_SUCCESS) {
            for (auto& SRC : m_impl->sources) {
                if (SRC.id == s.sourceId) {
                    SRC.slotsUsing--;
                    break;
                }
            }
            s.sourceId = 0;
        }
    }
    evictIdleSources();

    // This frame's damage, or the whole frame when the compositor reported none: with no damage to
    // trust, every destination takes the full frame next time so nothing it changed is missed.
    Hyprutils::Math::CRegion tracked;
    if (TRACKABLE) {
        if (damage) {
            tracked.set(*damage);
            tracked.intersect(0, 0, (double)WIDTH, (double)HEIGHT);
        } else
            tracked.set(Hyprutils::Math::CRegion(0, 0, (double)WIDTH, (double)HEIGHT));
    }

    // The grid scan turns "what the compositor reported" into "what the pixels actually show". It
    // runs on its own queue and its result is back before the copy is recorded, so the two need no
    // semaphore between them. A scan that cannot run leaves the reported damage in place (or the
    // whole frame when there was none) - never "nothing changed".
    bool              waitedOnProducer = false;
    bool              scanSucceeded    = false;
    const EDamageMode MODE             = damageMode();
    const bool SCAN = (MODE == EDamageMode::CompositorGrid || MODE == EDamageMode::Grid) && TRACKABLE && trackDamage && mscDst->seeded() && (MODE == EDamageMode::Grid || damage);
    if (SCAN) {
        const int SRCID = acquireSource(src, SRCDMA, SRCFMT);
        if (SRCID >= 0 && (size_t)SRCID < m_impl->sources.size()) {
            const uint64_t TARGETID = m_impl->sources[SRCID].id;
            bool           idle     = true;
            for (auto& s : m_impl->slots) {
                if (s.sourceId == TARGETID) {
                    if (!checkResult(vkWaitForFences(m_impl->device, 1, &s.fence, VK_TRUE, 100000000), "waiting for an in-flight copy of the same source")) {
                        idle = false;
                        break;
                    }
                    m_impl->sources[SRCID].slotsUsing--;
                    s.sourceId = 0;
                }
            }
            std::vector<SMSCDamageRect> scanned;
            if (idle &&
                scanRun(MODE == EDamageMode::Grid ? nullptr : damage, WIDTH, HEIGHT, SRCDMA.format, m_impl->sources[SRCID].image, m_impl->sources[SRCID].view, waitFd,
                        waitedOnProducer, scanned)) {
                scanSucceeded = true;
                if (MODE == EDamageMode::Grid || damage) {
                    tracked.clear();
                    for (const auto& R : scanned)
                        tracked.add(Hyprutils::Math::CRegion(R.x, R.y, R.w, R.h));
                }
            } else
                m_impl->shadowFresh = true;
        } else
            m_impl->shadowFresh = true;
    }

    // Blink paints this frame's damage over the copy, and restores the regions it painted last time
    // by copying them from the real source again - otherwise the ring would accumulate tint forever.
    Hyprutils::Math::CRegion paint = tracked;

    // The destination is one of a rotating set and every buffer in it has to catch up on the frames
    // it sat out, so this frame's damage is added to all of them, the one being written included;
    // each forgets it once written. Only destinations of this frame's size consume it: the cursor
    // destination and any other resolution would otherwise accumulate damage they never clear.
    // Every destination of this size accumulates the frame's damage: each has to catch up on the
    // frames it sat out, whichever producer the damage came from.
    if (trackDamage && TRACKABLE) {
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

        if (m_impl->blinkEnabled && !mscDst->m_tintedSet.empty())
            mscDst->accumulateDamage(mscDst->m_tintedSet);
    }

    Hyprutils::Math::CRegion pending = mscDst->pendingDamage();
    // The owed set is shared by every output; keep only what belongs to this frame, so a box can
    // never describe another output (or the cursor) and run past the destination buffer.
    pending.intersect(0, 0, (double)WIDTH, (double)HEIGHT);

    // Extract the boxes and their area once: pixman's rect array is already disjoint and sorted,
    // while getRects() would build a fresh vector on every call.
    int                   damageRectCount = 0;
    const pixman_box32_t* DAMAGE          = pixman_region32_rectangles(pending.pixman(), &damageRectCount);

    // One region per rectangle, so the list is bounded by the grid (the finest split the scan can
    // produce) and by a command-buffer budget, not by a fixed count.
    const uint32_t CELLSIZE    = m_impl->cellSize ? m_impl->cellSize : 64;
    const size_t   GRIDBOUND   = (size_t)((WIDTH + CELLSIZE - 1) / CELLSIZE) * (size_t)((HEIGHT + CELLSIZE - 1) / CELLSIZE);
    const size_t   BUDGETBOUND = ((size_t)1 << 20) / sizeof(VkBufferImageCopy);
    const size_t   MAXREGIONS  = std::clamp<size_t>(GRIDBOUND, 64, BUDGETBOUND);

    // The owed set is a union across the rotating destinations, so it often degenerates into many
    // strips. Collapse it into the same bounded rectangle list the scan produces instead of copying
    // a whole frame: the merged list is a superset, so it is always correct.
    if (damageRectCount > (int)MAXREGIONS) {
        std::vector<SMSCDamageRect> STRIPS;
        STRIPS.reserve((size_t)damageRectCount);
        for (int i = 0; i < damageRectCount; ++i)
            STRIPS.push_back({DAMAGE[i].x1, DAMAGE[i].y1, DAMAGE[i].x2 - DAMAGE[i].x1, DAMAGE[i].y2 - DAMAGE[i].y1});
        const auto MERGED = mscMergeCells(STRIPS, (uint32_t)MAXREGIONS, WIDTH, HEIGHT);
        pending.clear();
        for (const auto& R : MERGED)
            pending.add(Hyprutils::Math::CRegion(R.x, R.y, R.w, R.h));
        DAMAGE = pixman_region32_rectangles(pending.pixman(), &damageRectCount);
    }

    uint64_t damageArea = 0;
    for (int i = 0; i < damageRectCount; ++i)
        damageArea += (uint64_t)(DAMAGE[i].x2 - DAMAGE[i].x1) * (uint64_t)(DAMAGE[i].y2 - DAMAGE[i].y1);

    // Which of the untrusted cases forced a whole-frame copy.
    enum class EFullReason : uint8_t {
        None,
        NoTracking,
        SizeMismatch,
        Unseeded,
        NoDamage,
        ManyRects,
        MuchArea
    };
    EFullReason    fullReason = EFullReason::None;

    const uint64_t FRAMEPIXELS = (uint64_t)WIDTH * (uint64_t)HEIGHT;

    // The frames the tracking could be used on at all: the fallback rules do not apply to them. For
    // those, record what the owed set asked for and what was copied, so the two can be compared.
    const bool HAVE_DAMAGE = damage != nullptr || (MODE == EDamageMode::Grid && scanSucceeded);
    const bool TRACKED     = trackDamage && TRACKABLE && mscDst->seeded() && HAVE_DAMAGE;
    if (TRACKED) {
        m_impl->damageTrackedFrames++;
        m_impl->damageOwedPixels += damageArea;
        m_impl->damageTrackedFramePixels += FRAMEPIXELS;
    }

    ECopyMode mode;
    if (!trackDamage) {
        mode       = ECopyMode::Full;
        fullReason = EFullReason::NoTracking;
    } else if (!TRACKABLE) {
        mode       = ECopyMode::Full;
        fullReason = EFullReason::SizeMismatch;
    } else if (!mscDst->seeded()) {
        mode       = ECopyMode::Full;
        fullReason = EFullReason::Unseeded;
    } else if (!HAVE_DAMAGE) {
        mode       = ECopyMode::Full;
        fullReason = EFullReason::NoDamage;
    } else if (pending.empty())
        mode = ECopyMode::Skip;
    else if (damageRectCount > (int)MAXREGIONS) {
        mode       = ECopyMode::Full;
        fullReason = EFullReason::ManyRects;
    } else if ((double)damageArea > MSC_DAMAGE_FULL_FRACTION * (double)WIDTH * (double)HEIGHT) {
        mode       = ECopyMode::Full;
        fullReason = EFullReason::MuchArea;
    } else
        mode = ECopyMode::Partial;

    const auto ACCOUNT = [&](ECopyMode taken, size_t rects) {
        const uint64_t MOVED = taken == ECopyMode::Full ? FRAMEPIXELS : damageArea;
        m_impl->damageFramePixels += FRAMEPIXELS;
        m_impl->damagePixels += MOVED;
        m_impl->damageRects += rects;
        if (TRACKED)
            m_impl->damageTrackedMovedPixels += MOVED;

        if (taken == ECopyMode::Full) {
            m_impl->damageFull++;
            switch (fullReason) {
                case EFullReason::NoTracking: m_impl->damageFullNoTracking++; break;
                case EFullReason::SizeMismatch: m_impl->damageFullSizeMismatch++; break;
                case EFullReason::Unseeded: m_impl->damageFullUnseeded++; break;
                case EFullReason::NoDamage: m_impl->damageFullNoDamage++; break;
                case EFullReason::ManyRects: m_impl->damageFullManyRects++; break;
                case EFullReason::MuchArea: m_impl->damageFullMuchArea++; break;
                case EFullReason::None: break;
            }
        } else if (taken == ECopyMode::Partial)
            m_impl->damagePartial++;
        else
            m_impl->damageSkipped++;

        m_impl->damageBuckets[damageBucket(FRAMEPIXELS ? (double)MOVED / (double)FRAMEPIXELS : 0.0)]++;

        if (MSC_STATS_SECONDS > 0) {
            const auto NOW = std::chrono::steady_clock::now();
            if (NOW - m_impl->lastStats >= std::chrono::seconds(MSC_STATS_SECONDS)) {
                m_impl->lastStats = NOW;
                logStats();
            }
        }
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
    auto&      SOURCE = m_impl->sources[(size_t)SOURCEINDEX];

    const bool BLINKREADY =
        m_impl->blinkEnabled && TRACKABLE && (paint.empty() || ensureBlink(paint, WIDTH, HEIGHT, SRCDMA.format, SOURCE.image, SOURCE.view, waitFd, waitedOnProducer));

    // Wait for the incoming fence on the GPU rather than on the host: import it as a
    // semaphore and let the queue wait on it. Host-waiting it here would block the
    // compositor main thread for the whole producer render. Importing is best
    // effort; when the driver refuses the fd, fall back to the host wait.
    bool gpuWait = false;
    if (waitFd >= 0 && !waitedOnProducer) {
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

    // One region per rectangle, so the list is bounded by the grid (the finest split the scan can
    // produce) and by a command-buffer budget, not by a fixed count.
    std::vector<VkBufferImageCopy> REGIONS;
    REGIONS.reserve(mode == ECopyMode::Full ? 1 : (size_t)damageRectCount);
    if (mode == ECopyMode::Full) {
        REGIONS.push_back({.bufferOffset      = 0,
                           .bufferRowLength   = mscDst->storage().stride / 4,
                           .bufferImageHeight = HEIGHT,
                           .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           .imageExtent       = {WIDTH, HEIGHT, 1}});
    } else {
        for (int i = 0; i < damageRectCount; ++i) {
            const auto& BOX = DAMAGE[i];
            // Both offsets are needed: imageOffset selects the box in the source frame and
            // bufferOffset places it at the same coordinates in the destination buffer. Without
            // imageOffset every box is fed the frame's top-left corner.
            REGIONS.push_back({.bufferOffset      = (VkDeviceSize)BOX.y1 * mscDst->storage().stride + (VkDeviceSize)BOX.x1 * 4,
                               .bufferRowLength   = mscDst->storage().stride / 4,
                               .bufferImageHeight = HEIGHT,
                               .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                               .imageOffset       = {(int32_t)BOX.x1, (int32_t)BOX.y1, 0},
                               .imageExtent       = {(uint32_t)(BOX.x2 - BOX.x1), (uint32_t)(BOX.y2 - BOX.y1), 1}});
        }
    }
    vkCmdCopyImageToBuffer(SLOT.command, SOURCE.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mscDst->storage().buffer, (uint32_t)REGIONS.size(), REGIONS.data());

    if (BLINKREADY) {
        if (!paint.empty()) {
            int                            paintRects = 0;
            const pixman_box32_t*          PAINT      = pixman_region32_rectangles(paint.pixman(), &paintRects);
            std::vector<VkBufferImageCopy> TINTED;
            TINTED.reserve((size_t)paintRects);
            for (int i = 0; i < paintRects; ++i) {
                const int32_t X0 = std::max<int32_t>(0, PAINT[i].x1), Y0 = std::max<int32_t>(0, PAINT[i].y1);
                const int32_t X1 = std::min<int32_t>(PAINT[i].x2, (int32_t)WIDTH), Y1 = std::min<int32_t>(PAINT[i].y2, (int32_t)HEIGHT);
                if (X1 <= X0 || Y1 <= Y0)
                    continue;
                TINTED.push_back({.bufferOffset      = (VkDeviceSize)Y0 * mscDst->storage().stride + (VkDeviceSize)X0 * 4,
                                  .bufferRowLength   = mscDst->storage().stride / 4,
                                  .bufferImageHeight = HEIGHT,
                                  .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                  .imageOffset       = {X0, Y0, 0},
                                  .imageExtent       = {(uint32_t)(X1 - X0), (uint32_t)(Y1 - Y0), 1}});
            }
            if (!TINTED.empty()) {
                VkBufferMemoryBarrier tintBarrier = {.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                                     .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
                                                     .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
                                                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                     .buffer              = mscDst->storage().buffer,
                                                     .offset              = 0,
                                                     .size                = VK_WHOLE_SIZE};
                vkCmdPipelineBarrier(SLOT.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &tintBarrier, 0, nullptr);
                vkCmdCopyImageToBuffer(SLOT.command, m_impl->blinkImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mscDst->storage().buffer, (uint32_t)TINTED.size(), TINTED.data());
            }
        }
        mscDst->m_tintedSet = paint;
    }

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
                                      .signalSemaphoreCount = 1u,
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
    if (!waitedOnProducer) {
        if (gpuWait)
            ++m_impl->gpuWaits;
        else
            ++m_impl->hostWaits;
    }

    ACCOUNT(mode, mode == ECopyMode::Partial ? REGIONS.size() : 0);

    return true;
}

void CMSCTransfer::logStats() {
    const uint64_t  FRAMES = m_impl->damageFull + m_impl->damagePartial + m_impl->damageSkipped;
    const double    MEAN   = m_impl->damageFramePixels ? (double)m_impl->damagePixels / (double)m_impl->damageFramePixels * 100.0 : 0.0;
    const long long AGE    = (long long)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_impl->started).count();

    log(AQ_LOG_DEBUG,
        std::format("drm: MSC copy stats (t+{}s): {} copies, {} gpu waits, {} host waits, {} slot waits, {} source imports cached", AGE, m_impl->copies, m_impl->gpuWaits,
                    m_impl->hostWaits, m_impl->slotWaits, m_impl->sources.size()));
    log(AQ_LOG_DEBUG,
        std::format("drm: MSC damage over {} frames (t+{}s): partial {} full {} skipped {}, mean {:.2f}% of the frame moved, {} damage rects, buckets 0:{} <1%:{} <2%:{} <5%:{} "
                    "<10%:{} <25%:{} "
                    "<50%:{} >=50%:{}",
                    FRAMES, AGE, m_impl->damagePartial, m_impl->damageFull, m_impl->damageSkipped, MEAN, m_impl->damageRects, m_impl->damageBuckets[0], m_impl->damageBuckets[1],
                    m_impl->damageBuckets[2], m_impl->damageBuckets[3], m_impl->damageBuckets[4], m_impl->damageBuckets[5], m_impl->damageBuckets[6], m_impl->damageBuckets[7]));
    // Both means are over the same frames - the ones the tracking could be used on - so the gap
    // between them is exactly what the whole-frame fallbacks cost. The first line reports the
    // fraction of every frame the copy actually moved, whatever mode it took.
    const double OWED  = m_impl->damageTrackedFramePixels ? (double)m_impl->damageOwedPixels / (double)m_impl->damageTrackedFramePixels * 100.0 : 0.0;
    const double MOVED = m_impl->damageTrackedFramePixels ? (double)m_impl->damageTrackedMovedPixels / (double)m_impl->damageTrackedFramePixels * 100.0 : 0.0;
    if (m_impl->scanFrames)
        log(AQ_LOG_DEBUG,
            std::format("drm: MSC damage grid (t+{}s): {} frames, {} cells scanned, {} dirty, {:.1f}% of the scanned cells, {:.1f} us per frame", AGE, m_impl->scanFrames,
                        m_impl->scanCells, m_impl->scanDirtyCells, m_impl->scanCells ? 100.0 * (double)m_impl->scanDirtyCells / (double)m_impl->scanCells : 0.0,
                        m_impl->scanFrames ? (double)m_impl->scanNs / (double)m_impl->scanFrames / 1000.0 : 0.0));
    log(AQ_LOG_DEBUG,
        std::format("drm: MSC damage full reasons: no tracking {}, size mismatch {}, unseeded {}, no reported damage {}, over the region budget {}, over {:.0f}% of the frame {}; "
                    "owed {:.2f}% vs moved {:.2f}% over {} tracked frames",
                    m_impl->damageFullNoTracking, m_impl->damageFullSizeMismatch, m_impl->damageFullUnseeded, m_impl->damageFullNoDamage, m_impl->damageFullManyRects,
                    MSC_DAMAGE_FULL_FRACTION * 100.0, m_impl->damageFullMuchArea, OWED, MOVED, m_impl->damageTrackedFrames));
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
