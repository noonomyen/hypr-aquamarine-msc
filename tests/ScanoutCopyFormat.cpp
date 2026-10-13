// Small check for the two pure helpers of the muxless scanout copy path. Both
// have already been wrong once (the 10-bit channel order), so they keep a check.
#include "ScanoutCopy.hpp"

#include <cassert>
#include <cstdio>

#include <drm_fourcc.h>

int main() {
    // The destination stride must fit the row and stay 256-aligned: the display
    // engine wants 64-byte alignment for linear, and i915's cursor plane wants a
    // 256-aligned pitch.
    assert(Aquamarine::mscStrideFor(1920) == 7680);
    assert(Aquamarine::mscStrideFor(1920) % 256 == 0);
    assert(Aquamarine::mscStrideFor(1080) >= 1080 * 4);
    assert(Aquamarine::mscStrideFor(1080) % 256 == 0);
    assert(Aquamarine::mscStrideFor(1) == 256);

    // 8-bit: XRGB8888/ARGB8888 keep the byte order, so BGRA in Vulkan.
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_XRGB8888) == VK_FORMAT_B8G8R8A8_UNORM);
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_ARGB8888) == VK_FORMAT_B8G8R8A8_UNORM);
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_XBGR8888) == VK_FORMAT_R8G8B8A8_UNORM);

    // 10-bit: ARGB2101010 packs R in the high bits after alpha, so it maps to
    // A2R10G10B10, and ABGR2101010 to A2B10G10R10. Swapping these was a real bug.
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_ARGB2101010) == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_XRGB2101010) == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_ABGR2101010) == VK_FORMAT_A2B10G10R10_UNORM_PACK32);

    // Anything else is refused rather than silently copied with a wrong format.
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_NV12) == VK_FORMAT_UNDEFINED);
    assert(Aquamarine::mscVkFormat(DRM_FORMAT_ABGR16161616F) == VK_FORMAT_UNDEFINED);

    std::printf("ScanoutCopyFormat: ok\n");
    return 0;
}
