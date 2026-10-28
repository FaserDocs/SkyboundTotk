#include <cassert>
#include <cstdint>
#include <vector>

#include "../render/rx_bcn.h"

int main() {
    // BC1: red endpoint selected for all texels (565 red = 0xf800).
    const uint8_t red_blocks[32] = {0x00, 0xf8, 0x00, 0x00, 0, 0, 0, 0,
                                    0x00, 0xf8, 0x00, 0x00, 0, 0, 0, 0,
                                    0x00, 0xf8, 0x00, 0x00, 0, 0, 0, 0};
    uint8_t rgba[5 * 3 * 4]{};
    rx::bcn::decode(red_blocks, 5, 3, 1, rx::bcn::Format::BC1, rgba);
    for (size_t i = 0; i < 15; ++i) {
        assert(rgba[i * 4] == 255);
        assert(rgba[i * 4 + 1] == 0);
        assert(rgba[i * 4 + 2] == 0);
        assert(rgba[i * 4 + 3] == 255);
    }

    // BC4 with endpoints 200 and 100; all indices select endpoint 0.
    const uint8_t channel_block[8] = {200, 100, 0, 0, 0, 0, 0, 0};
    uint8_t channel_rgba[4 * 4 * 4]{};
    rx::bcn::decode(channel_block, 4, 4, 1, rx::bcn::Format::BC4, channel_rgba);
    for (size_t i = 0; i < 16; ++i) {
        assert(channel_rgba[i * 4] == 200);
        assert(channel_rgba[i * 4 + 1] == 0);
        assert(channel_rgba[i * 4 + 2] == 0);
        assert(channel_rgba[i * 4 + 3] == 255);
    }

    // 1x1x2 checks 3D slice addressing and clipping, including signed BC4 bit patterns.
    const uint8_t signed_blocks[16] = {0x80, 0x7f, 0, 0, 0, 0, 0, 0,
                                       0x40, 0xc0, 0, 0, 0, 0, 0, 0};
    uint8_t signed_rgba[2 * 4]{};
    rx::bcn::decode(signed_blocks, 1, 1, 2, rx::bcn::Format::BC4Snorm, signed_rgba);
    assert(signed_rgba[0] == 0x81); // -127 encoded as signed-normalized byte
    assert(signed_rgba[4] == 0x40);
}
