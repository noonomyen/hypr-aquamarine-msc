#pragma once

#include <aquamarine/allocator/Allocator.hpp>
#include <aquamarine/backend/Backend.hpp>
#include <aquamarine/buffer/Buffer.hpp>
#include <hyprutils/math/Region.hpp>
#include <hyprutils/memory/SharedPtr.hpp>
#include <hyprutils/memory/WeakPtr.hpp>
#include <hyprutils/os/FileDescriptor.hpp>
#include <drm_fourcc.h>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace Aquamarine {
    class CDRMBackend;
    class CMSCBuffer;
    class CMSCTransfer;

    struct SMSCBufferStorage {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDevice       device = VK_NULL_HANDLE;
        int            fd     = -1;
        uint32_t       stride = 0;
    };

    // MSC (Muxless Scanout Copy): copies a buffer produced on the primary (rendering)
    // GPU into a linear, host-visible destination the secondary (scanout) GPU can
    // display, using a Vulkan copy and a native completion fence. Replaces the
    // secondary-GPU EGL blit when AQ_MSC=1.
    //
    // Why not reuse the existing cross-GPU blit? The probes showed there is no
    // modifier that is both NVIDIA-EGL-renderable and i915-scannable, so an EGL
    // blit on the display GPU cannot land in a buffer it can also scan out: the
    // blit path degrades to a CPU readback (EGL_BAD_MATCH on the source import).
    // The Vulkan copy writes into system memory the display GPU can actually
    // fetch, which is the whole point.
    // Pure helpers (kept here so the unit test can reach them).

    // The destination stride: 256-aligned so the same buffer is valid on the
    // primary plane (64-byte linear rule) and the cursor plane (i915 wants cursor
    // pitches 256-aligned).
    // Deliberate simplification: 4 bytes per pixel, single plane; derive from the format when a
    // 16bpp or multi-plane source ever shows up.
    inline uint32_t mscStrideFor(uint32_t width) {
        return ((width * 4 + 255) / 256) * 256;
    }

    // The bpc a scanout format carries. Under MSC the destination format is the compositor's,
    // so this is the honest ceiling for the link: a wider one would carry no more bits.
    inline uint8_t mscBpcForFormat(uint32_t fourcc) {
        switch (fourcc) {
            case DRM_FORMAT_XRGB2101010:
            case DRM_FORMAT_XBGR2101010:
            case DRM_FORMAT_RGBX1010102:
            case DRM_FORMAT_BGRX1010102:
            case DRM_FORMAT_ARGB2101010:
            case DRM_FORMAT_ABGR2101010:
            case DRM_FORMAT_RGBA1010102:
            case DRM_FORMAT_BGRA1010102: return 10;

            case DRM_FORMAT_XRGB16161616:
            case DRM_FORMAT_XBGR16161616:
            case DRM_FORMAT_ARGB16161616:
            case DRM_FORMAT_ABGR16161616:

            case DRM_FORMAT_XRGB16161616F:
            case DRM_FORMAT_XBGR16161616F:
            case DRM_FORMAT_ARGB16161616F:
            case DRM_FORMAT_ABGR16161616F: return 16;

            case DRM_FORMAT_RGB565: return 6;

            default: return 8;
        }
    }

    inline VkFormat mscVkFormat(uint32_t fourcc) {
        switch (fourcc) {
            case DRM_FORMAT_ARGB8888:
            case DRM_FORMAT_XRGB8888: return VK_FORMAT_B8G8R8A8_UNORM;
            case DRM_FORMAT_ABGR8888:
            case DRM_FORMAT_XBGR8888: return VK_FORMAT_R8G8B8A8_UNORM;
            case DRM_FORMAT_ARGB2101010:
            case DRM_FORMAT_XRGB2101010: return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
            case DRM_FORMAT_ABGR2101010:
            case DRM_FORMAT_XBGR2101010: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            default: return VK_FORMAT_UNDEFINED;
        }
    }

    // The grid scan's shadow format. One representation per depth so neither path pays for the
    // other, and only for the formats the copy itself carries: 8-bit keeps the byte shadow its
    // compare was built on, 10-bit gets a packed word per pixel so the compare sees every bit the
    // scanout displays. A format the copy refuses has no shadow either.
    inline VkFormat mscShadowFormat(uint32_t fourcc) {
        switch (mscVkFormat(fourcc)) {
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
            case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return VK_FORMAT_R32_UINT;
            default: return VK_FORMAT_UNDEFINED;
        }
    }

    // The family the scan runs on: a compute family without the graphics engine first, so the scan
    // does not compete with the compositor's renderer, and any compute family as the fallback.
    inline uint32_t mscPickScanFamily(const VkQueueFamilyProperties* families, uint32_t count) {
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t i = 0; i < count; ++i) {
                if (!families[i].queueCount || !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
                    continue;
                if (pass == 0 && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                    continue;
                return i;
            }
        }
        return UINT32_MAX;
    }

    // A host-visible memory type for the scan's readback buffers, cached preferred: the dirty map
    // is read word by word on the CPU, and the write-combined type costs an uncached read per word.
    // The fallback keeps the buffers coherent, which the read path relies on.
    inline uint32_t mscHostMemoryType(const VkPhysicalDeviceMemoryProperties& memProps, uint32_t bits) {
        for (const VkMemoryPropertyFlags REQUIRED : {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT}) {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((bits & (1u << i)) && ((memProps.memoryTypes[i].propertyFlags & REQUIRED) == REQUIRED))
                    return i;
            }
        }
        return UINT32_MAX;
    }

    struct SMSCDamageRect {
        int32_t x = 0, y = 0, w = 0, h = 0;
    };

    // Turn dirty grid cells into copy rectangles: horizontal runs inside each cell row, then a
    // vertical merge of runs that share an x range, then cheapest-pair merges until the rectangle
    // count fits `cap`. Every rectangle is clipped to the frame, because the last grid row and
    // column are short and a copy outside the image faults the device.
    std::vector<SMSCDamageRect> mscMergeCells(const std::vector<SMSCDamageRect>& cells, uint32_t cap, uint32_t frameW, uint32_t frameH);

    class CMSCTransfer {
      public:
        static Hyprutils::Memory::CSharedPointer<CMSCTransfer> create(Hyprutils::Memory::CWeakPointer<CDRMBackend> primary, Hyprutils::Memory::CSharedPointer<CBackend> logBackend,
                                                                      bool useCopyEngine);
        ~CMSCTransfer();

        bool                                       good();
        uint64_t                                   copyCount() const;

        Hyprutils::Memory::CSharedPointer<IBuffer> allocateDestination(uint32_t fourcc, const Hyprutils::Math::Vector2D& size);

        // Copy `src` into `dst` (a CMSCBuffer). When `waitFd` is >= 0 it is imported as a
        // semaphore and waited on the GPU (with a host-wait fallback if the import fails), and
        // `outFence` is set to the copy-completion sync file, or left invalid when the frame was
        // skipped (the destination already held it). `damage` is the region that changed since the
        // last frame, or null when the caller has nothing to trust, in which case every
        // destination takes a whole frame.
        bool copy(Hyprutils::Memory::CSharedPointer<IBuffer> src, Hyprutils::Memory::CSharedPointer<IBuffer> dst, int waitFd, Hyprutils::OS::CFileDescriptor& outFence,
                  const Hyprutils::Math::CRegion* damage = nullptr);

        Hyprutils::Memory::CWeakPointer<CDRMBackend>  primary();

        Hyprutils::Memory::CWeakPointer<CMSCTransfer> self;

      private:
        friend class CMSCBuffer;

        CMSCTransfer(Hyprutils::Memory::CWeakPointer<CDRMBackend> primary, Hyprutils::Memory::CSharedPointer<CBackend> logBackend, bool useCopyEngine);

        bool allocateDestinationStorage(VkDeviceSize bytes, SMSCBufferStorage& out);
        void freeDestinationStorage(SMSCBufferStorage& storage);

        // Grid damage scan: create the pass, run it for one frame and return the merged copy
        // rectangles. `hint` is the compositor's damage (a superset of what changed), or null to
        // scan the whole frame; the shadow is refreshed with what the pass saw.
        bool scanInit();
        bool scanRun(const Hyprutils::Math::CRegion* hint, uint32_t width, uint32_t height, uint32_t fourcc, VkImage srcImage, VkImageView srcView, int waitFd,
                     bool& waitedOnProducer, std::vector<SMSCDamageRect>& out);
        void scanDestroy();

        // Blink: an image holding the alpha-blended red highlight for the decided damage, used as
        // the copy source for the damaged boxes so the regions the copy touched are visible.
        bool ensureBlink(const Hyprutils::Math::CRegion& paint, uint32_t width, uint32_t height, uint32_t fourcc, VkImage srcImage, VkImageView srcView, int waitFd,
                         bool& waitedOnProducer);

        // One import per producer buffer, kept while the compositor still holds it: importing
        // one costs a vkCreateImage plus a dedicated allocation of its dma-buf.
        int  acquireSource(const Hyprutils::Memory::CSharedPointer<IBuffer>& src, const SDMABUFAttrs& attrs, VkFormat format);
        void evictIdleSources();
        void logStats();

        // Logs a failed Vulkan call and latches a lost device, which is the only failure the
        // backend can recover from by rebuilding the engine.
        bool checkResult(VkResult result, const char* what);

        void log(eBackendLogLevel level, const std::string& msg);

        struct SImpl;
        std::unique_ptr<SImpl>                       m_impl;
        Hyprutils::Memory::CWeakPointer<CDRMBackend> m_primary;
        Hyprutils::Memory::CWeakPointer<CBackend>    m_backend; // log target
        // Fail-closed until the constructor has a working copy engine: it clears this only when
        // every copy object exists, and sets it again if the optional scan cannot be built.
        bool m_bad = true;
    };

    class CMSCBuffer : public IBuffer {
      public:
        CMSCBuffer(Hyprutils::Memory::CSharedPointer<CMSCTransfer> transfer, unsigned width, unsigned height, uint32_t fourcc);
        ~CMSCBuffer() override;

        eBufferCapability caps() override;
        eBufferType       type() override;
        void              update(const Hyprutils::Math::CRegion&) override;
        bool              isSynchronous() override;
        bool              good() override;
        SDMABUFAttrs      dmabuf() override;

      private:
        friend class CMSCTransfer;

        // Regions this destination had painted with the blink colour last time it was written, so
        // the next write can restore them from the real source before painting the new damage.
        Hyprutils::Math::CRegion                        m_tintedSet;

        SMSCBufferStorage&                              storage();
        Hyprutils::Memory::CSharedPointer<CMSCTransfer> m_transfer;
        SDMABUFAttrs                                    m_attrs;
        SMSCBufferStorage                               m_storage;
        // The damage this destination still owes: what a copy has to move the next time this
        // buffer is written. It belongs to the buffer, because the ring is what makes a frame
        // miss damage, and a buffer that has never been filled is not in sync with anything.
        Hyprutils::Math::CRegion        m_pendingDamage;
        bool                            m_seeded = false;

        const Hyprutils::Math::CRegion& pendingDamage() const;
        bool                            seeded() const;
        void                            markSynced();
        // Every frame's damage goes to every destination in the rotating set, not only the one
        // being written; each buffer forgets what it owes once it has been written again.
        void accumulateDamage(const Hyprutils::Math::CRegion& region);
    };

    class CMSCAllocator : public IAllocator {
      public:
        static Hyprutils::Memory::CSharedPointer<CMSCAllocator> create(Hyprutils::Memory::CSharedPointer<CMSCTransfer> transfer,
                                                                       Hyprutils::Memory::CSharedPointer<CBackend>     backend);
        ~CMSCAllocator() override = default;

        Hyprutils::Memory::CSharedPointer<IBuffer>  acquire(const SAllocatorBufferParams& params, Hyprutils::Memory::CSharedPointer<CSwapchain> swapchain) override;
        Hyprutils::Memory::CSharedPointer<CBackend> getBackend() override;
        int                                         drmFD() override;
        eAllocatorType                              type() override;

      private:
        CMSCAllocator(Hyprutils::Memory::CSharedPointer<CMSCTransfer> transfer, Hyprutils::Memory::CSharedPointer<CBackend> backend);

        Hyprutils::Memory::CSharedPointer<CMSCTransfer> m_transfer;
        Hyprutils::Memory::CWeakPointer<CBackend>       m_backend;
    };
};
