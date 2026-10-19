// Check for the damage-cell merge of the muxless scanout copy path: the function that turns dirty
// grid cells into the copy rectangle list. It is pure logic, so it is checked here rather than on
// hardware - the one failure mode that has to be impossible is a rectangle that runs outside the
// frame, which faults the device instead of returning an error.
#include "ScanoutCopy.hpp"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using Aquamarine::SMSCDamageRect;

static bool covers(const std::vector<SMSCDamageRect>& rects, int32_t x, int32_t y) {
    for (const auto& R : rects) {
        if (x >= R.x && x < R.x + R.w && y >= R.y && y < R.y + R.h)
            return true;
    }
    return false;
}

static bool allInside(const std::vector<SMSCDamageRect>& rects, uint32_t w, uint32_t h) {
    for (const auto& R : rects) {
        if (R.w <= 0 || R.h <= 0 || R.x < 0 || R.y < 0 || R.x + R.w > (int32_t)w || R.y + R.h > (int32_t)h)
            return false;
    }
    return true;
}

static void checkCellsCovered(const std::vector<SMSCDamageRect>& cells, const std::vector<SMSCDamageRect>& merged) {
    for (const auto& C : cells) {
        assert(covers(merged, C.x, C.y));
        assert(covers(merged, C.x + C.w - 1, C.y + C.h - 1));
        assert(covers(merged, C.x + C.w / 2, C.y + C.h / 2));
    }
}

int main() {
    const uint32_t W = 2560, H = 1440, CAP = 16;

    // Nothing dirty, nothing to copy.
    assert(Aquamarine::mscMergeCells({}, CAP, W, H).empty());

    // One cell stays one cell.
    {
        const auto R = Aquamarine::mscMergeCells({{64, 96, 16, 16}}, CAP, W, H);
        assert(R.size() == 1 && R[0].x == 64 && R[0].y == 96 && R[0].w == 16 && R[0].h == 16);
    }

    // A text line becomes one run, not one region per cell.
    {
        std::vector<SMSCDamageRect> cells;
        for (int x = 0; x < 38; ++x)
            cells.push_back({100 + x * 16, 600, 16, 16});
        const auto R = Aquamarine::mscMergeCells(cells, CAP, W, H);
        assert(R.size() == 1 && R[0].x == 100 && R[0].y == 600 && R[0].w == 38 * 16 && R[0].h == 16);
    }

    // Two stacked runs with the same x range merge vertically.
    {
        const auto R = Aquamarine::mscMergeCells({{0, 0, 16, 16}, {0, 16, 16, 16}, {0, 32, 16, 16}}, CAP, W, H);
        assert(R.size() == 1 && R[0].h == 48);
    }

    // Runs that do not share an x range stay separate.
    {
        const auto R = Aquamarine::mscMergeCells({{0, 0, 16, 16}, {32, 16, 16, 16}}, CAP, W, H);
        assert(R.size() == 2);
    }

    // A 2D block (many rows, many columns) collapses to one rectangle.
    {
        std::vector<SMSCDamageRect> cells;
        for (int y = 0; y < 20; ++y)
            for (int x = 0; x < 20; ++x)
                cells.push_back({200 + x * 16, 300 + y * 16, 16, 16});
        const auto R = Aquamarine::mscMergeCells(cells, CAP, W, H);
        assert(R.size() == 1 && R[0].w == 320 && R[0].h == 320);
    }

    // The last grid row and column are short (1440 = 90 * 16 exactly, 2560 / 16 = 160 exactly, so
    // use a frame that is not a multiple of the cell size for the interesting case).
    {
        const uint32_t              w = 1920, h = 1080; // 1080 = 67.5 * 16 -> the last row is 8 rows tall
        std::vector<SMSCDamageRect> cells;
        for (int x = 0; x < 4; ++x)
            cells.push_back({x * 16, 1072, 16, 8});
        const auto R = Aquamarine::mscMergeCells(cells, CAP, w, h);
        assert(allInside(R, w, h));
        assert(R.size() == 1 && R[0].y == 1072 && R[0].h == 8 && R[0].w == 64);
    }

    // Cells handed in already running past the frame (a clipped short column) must not produce a
    // rectangle outside it.
    {
        const uint32_t              w = 1920, h = 1080;
        std::vector<SMSCDamageRect> cells = {{1904, 1064, 16, 16}, {1904, 1072, 16, 8}, {1919, 0, 16, 16}};
        const auto                  R     = Aquamarine::mscMergeCells(cells, CAP, w, h);
        assert(allInside(R, w, h));
    }

    // Scattered damage: every patch must stay covered (a superset is correct, a miss is not) and
    // the rectangle count must respect the cap.
    {
        std::vector<SMSCDamageRect> cells;
        for (int i = 0; i < 20; ++i) {
            const int px = 80 + (i % 4) * 500, py = 72 + (i / 4) * 200;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                    cells.push_back({px + x * 16, py + y * 16, 16, 16});
        }
        const auto R = Aquamarine::mscMergeCells(cells, CAP, W, H);
        assert(R.size() <= CAP);
        assert(allInside(R, W, H));
        checkCellsCovered(cells, R);
    }

    // With the cap forced low, the merge must still cover every cell (it may only add area).
    {
        std::vector<SMSCDamageRect> cells;
        for (int i = 0; i < 40; ++i)
            cells.push_back({(i % 8) * 32, i * 32, 16, 16});
        const auto R = Aquamarine::mscMergeCells(cells, 4, W, H);
        assert(R.size() <= 4);
        assert(allInside(R, W, H));
        checkCellsCovered(cells, R);
    }

    // Duplicate and empty cells are harmless.
    {
        const auto R = Aquamarine::mscMergeCells({{0, 0, 16, 16}, {0, 0, 16, 16}, {0, 0, 0, 16}, {0, 0, 16, 0}}, CAP, W, H);
        assert(R.size() == 1);
    }

    std::printf("ScanoutCopyDamage: ok\n");
    return 0;
}
