// Checks for the pure helpers of the muxless scanout copy path. The format tables have already
// been wrong once (the 10-bit channel order), and the scan's family and memory-type choices decide
// how expensive the readback is, so they keep a check. CHECK aborts in every build type: assert()
// is compiled out under the documented Release configuration, which made these checks vacuous.
#include "ScanoutCopy.hpp"

#include <cstdio>
#include <cstdlib>

#include <drm_fourcc.h>

#define CHECK(cond)                                                                                                                                                                \
    do {                                                                                                                                                                           \
        if (!(cond)) {                                                                                                                                                             \
            std::fprintf(stderr, "ScanoutCopyFormat: FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                                              \
            std::exit(1);                                                                                                                                                          \
        }                                                                                                                                                                          \
    } while (0)

int main() {
    // The destination stride must fit the row and stay 256-aligned: the display engine wants
    // 64-byte alignment for linear, and i915's cursor plane wants a 256-aligned pitch.
    CHECK(Aquamarine::mscStrideFor(1920) == 7680);
    CHECK(Aquamarine::mscStrideFor(1920) % 256 == 0);
    CHECK(Aquamarine::mscStrideFor(1080) >= 1080 * 4);
    CHECK(Aquamarine::mscStrideFor(1080) % 256 == 0);
    CHECK(Aquamarine::mscStrideFor(1) == 256);

    // 8-bit: XRGB8888/ARGB8888 keep the byte order, so BGRA in Vulkan.
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_XRGB8888) == VK_FORMAT_B8G8R8A8_UNORM);
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_ARGB8888) == VK_FORMAT_B8G8R8A8_UNORM);
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_XBGR8888) == VK_FORMAT_R8G8B8A8_UNORM);

    // 10-bit: ARGB2101010 packs R in the high bits after alpha, so it maps to A2R10G10B10, and
    // ABGR2101010 to A2B10G10R10. Swapping these was a real bug.
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_ARGB2101010) == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_XRGB2101010) == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_ABGR2101010) == VK_FORMAT_A2B10G10R10_UNORM_PACK32);

    // Anything else is refused rather than silently copied with a wrong format.
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_NV12) == VK_FORMAT_UNDEFINED);
    CHECK(Aquamarine::mscVkFormat(DRM_FORMAT_ABGR16161616F) == VK_FORMAT_UNDEFINED);

    // The MSC max_bpc ceiling follows the pixel depth of the scanout format.
    CHECK(Aquamarine::mscBpcForFormat(DRM_FORMAT_XRGB8888) == 8);
    CHECK(Aquamarine::mscBpcForFormat(DRM_FORMAT_ARGB2101010) == 10);
    CHECK(Aquamarine::mscBpcForFormat(DRM_FORMAT_XRGB16161616) == 16);
    CHECK(Aquamarine::mscBpcForFormat(DRM_FORMAT_XRGB16161616F) == 16);
    CHECK(Aquamarine::mscBpcForFormat(DRM_FORMAT_RGB565) == 6);

    // The grid scan's shadow follows the depth as well: bytes for 8-bit, one packed word for
    // 10-bit, and no shadow for a format the copy path does not carry.
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_XRGB8888) == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_XBGR8888) == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_XRGB2101010) == VK_FORMAT_R32_UINT);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_ABGR2101010) == VK_FORMAT_R32_UINT);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_RGBX1010102) == VK_FORMAT_UNDEFINED);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_XRGB16161616F) == VK_FORMAT_UNDEFINED);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_NV12) == VK_FORMAT_UNDEFINED);
    CHECK(Aquamarine::mscShadowFormat(DRM_FORMAT_RGB565) == VK_FORMAT_UNDEFINED);

    // The scan prefers a compute-only family (the dedicated async compute engine) even when it is
    // listed after a family that shares graphics, and falls back to the shared family when the
    // device has no compute-only one. A family without compute or without queues is never picked.
    {
        const VkQueueFamilyProperties FAMILIES[] = {
            {.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, .queueCount = 16},
            {.queueFlags = VK_QUEUE_TRANSFER_BIT, .queueCount = 2},
            {.queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, .queueCount = 8},
            {.queueFlags = VK_QUEUE_COMPUTE_BIT, .queueCount = 0},
        };
        CHECK(Aquamarine::mscPickScanFamily(FAMILIES, 4) == 2);
        CHECK(Aquamarine::mscPickScanFamily(FAMILIES, 0) == UINT32_MAX);
    }
    {
        const VkQueueFamilyProperties FAMILIES[] = {
            {.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, .queueCount = 1},
            {.queueFlags = VK_QUEUE_TRANSFER_BIT, .queueCount = 1},
        };
        CHECK(Aquamarine::mscPickScanFamily(FAMILIES, 2) == 0);
    }
    {
        const VkQueueFamilyProperties FAMILIES[] = {
            {.queueFlags = VK_QUEUE_GRAPHICS_BIT, .queueCount = 1},
            {.queueFlags = VK_QUEUE_COMPUTE_BIT, .queueCount = 0},
        };
        CHECK(Aquamarine::mscPickScanFamily(FAMILIES, 2) == UINT32_MAX);
    }

    // The readback buffers take the cached host-visible type when the device has one, because the
    // dirty map is read word by word; the write-combined type is the fallback and the buffers must
    // stay coherent either way.
    {
        VkPhysicalDeviceMemoryProperties MEMORIES{};
        MEMORIES.memoryTypeCount              = 3;
        MEMORIES.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        MEMORIES.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        MEMORIES.memoryTypes[2].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        CHECK(Aquamarine::mscHostMemoryType(MEMORIES, 0x7) == 1);
        CHECK(Aquamarine::mscHostMemoryType(MEMORIES, 0x3) == 1);
        CHECK(Aquamarine::mscHostMemoryType(MEMORIES, 0x1) == 0);
        CHECK(Aquamarine::mscHostMemoryType(MEMORIES, 0x4) == UINT32_MAX);
        CHECK(Aquamarine::mscHostMemoryType(MEMORIES, 0x0) == UINT32_MAX);

        // No cached type at all: still visible and coherent, never the device-local one.
        MEMORIES.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        CHECK(Aquamarine::mscHostMemoryType(MEMORIES, 0x7) == 0);
    }

    std::printf("ScanoutCopyFormat: ok\n");
    return 0;
}
