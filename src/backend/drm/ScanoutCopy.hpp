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
        bool                                         m_bad = true;
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

        SMSCBufferStorage&                              storage();
        Hyprutils::Memory::CSharedPointer<CMSCTransfer> m_transfer;
        SDMABUFAttrs                                    m_attrs;
        SMSCBufferStorage                               m_storage;
        // The damage this destination still owes: what a copy has to move the next time this
        // buffer is written. It belongs to the buffer, because the ring is what makes a frame
        // miss damage, and a buffer that has never been filled is not in sync with anything.
        Hyprutils::Math::CRegion m_pendingDamage;
        bool                     m_seeded = false;

        Hyprutils::Math::CRegion pendingDamage();
        bool                     seeded() const;
        void                     markSynced();
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
