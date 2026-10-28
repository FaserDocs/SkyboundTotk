// BC1 through BC5 CPU fallback for compressed textures WebGPU cannot represent directly.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rx::bcn {
enum class Format { BC1, BC2, BC3, BC4, BC4Snorm, BC5, BC5Snorm };

inline uint16_t read16(const uint8_t* p) {
    return uint16_t(p[0] | uint16_t(p[1]) << 8);
}
inline uint32_t read32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

inline void decode_bc1_block(const uint8_t* src, uint8_t dst[16][4], bool force_opaque = false) {
    const uint16_t c0 = read16(src), c1 = read16(src + 2);
    uint8_t palette[4][4]{};
    auto unpack565 = [](uint16_t c, uint8_t* rgba) {
        rgba[0] = uint8_t(((c >> 11) * 255 + 15) / 31);
        rgba[1] = uint8_t((((c >> 5) & 63) * 255 + 31) / 63);
        rgba[2] = uint8_t(((c & 31) * 255 + 15) / 31);
        rgba[3] = 255;
    };
    unpack565(c0, palette[0]);
    unpack565(c1, palette[1]);
    for (unsigned c = 0; c < 3; ++c) {
        if (c0 > c1 || force_opaque) {
            palette[2][c] = uint8_t((2 * palette[0][c] + palette[1][c]) / 3);
            palette[3][c] = uint8_t((palette[0][c] + 2 * palette[1][c]) / 3);
        } else {
            palette[2][c] = uint8_t((palette[0][c] + palette[1][c]) / 2);
            palette[3][c] = 0;
        }
    }
    palette[2][3] = 255;
    palette[3][3] = c0 > c1 || force_opaque ? 255 : 0;
    const uint32_t indices = read32(src + 4);
    for (unsigned i = 0; i < 16; ++i)
        std::memcpy(dst[i], palette[(indices >> (2 * i)) & 3], 4);
}

inline void decode_bc4_block(const uint8_t* src, uint8_t dst[16], bool signed_values) {
    int palette[8];
    palette[0] = signed_values ? std::max(-127, int(int8_t(src[0]))) : src[0];
    palette[1] = signed_values ? std::max(-127, int(int8_t(src[1]))) : src[1];
    if (palette[0] > palette[1]) {
        for (int i = 2; i < 8; ++i)
            palette[i] = ((8 - i) * palette[0] + (i - 1) * palette[1]) / 7;
    } else {
        for (int i = 2; i < 6; ++i)
            palette[i] = ((6 - i) * palette[0] + (i - 1) * palette[1]) / 5;
        palette[6] = signed_values ? -127 : 0;
        palette[7] = signed_values ? 127 : 255;
    }
    uint64_t indices = 0;
    for (unsigned i = 0; i < 6; ++i)
        indices |= uint64_t(src[i + 2]) << (8 * i);
    for (unsigned i = 0; i < 16; ++i)
        dst[i] = uint8_t(palette[(indices >> (3 * i)) & 7]);
}

/// Decodes source block bytes into tightly packed RGBA8 texels and clips partial edge blocks.
inline void decode(const uint8_t* src, unsigned width, unsigned height, unsigned depth, Format format,
                   uint8_t* dst) {
    const size_t blocks_x = (width + 3) / 4;
    const size_t blocks_y = (height + 3) / 4;
    const bool color_format = format == Format::BC1 || format == Format::BC2 || format == Format::BC3;
    const bool two_channel = format == Format::BC5 || format == Format::BC5Snorm;
    const bool signed_values = format == Format::BC4Snorm || format == Format::BC5Snorm;
    const size_t block_bytes = format == Format::BC1 || format == Format::BC4 || format == Format::BC4Snorm ? 8 : 16;

    for (unsigned z = 0; z < depth; ++z) {
        for (size_t by = 0; by < blocks_y; ++by) {
            for (size_t bx = 0; bx < blocks_x; ++bx) {
                const uint8_t* block = src + ((size_t(z) * blocks_y + by) * blocks_x + bx) * block_bytes;
                uint8_t rgba[16][4]{};
                for (unsigned i = 0; i < 16; ++i) rgba[i][3] = signed_values ? 127 : 255;
                if (color_format) {
                    const uint8_t* color = block + (format == Format::BC1 ? 0 : 8);
                    decode_bc1_block(color, rgba, format != Format::BC1);
                }
                if (format == Format::BC2) {
                    for (unsigned i = 0; i < 16; ++i)
                        rgba[i][3] = uint8_t(((block[i / 2] >> ((i & 1) * 4)) & 15) * 17);
                } else if (format == Format::BC3) {
                    uint8_t alpha[16];
                    decode_bc4_block(block, alpha, false);
                    for (unsigned i = 0; i < 16; ++i) rgba[i][3] = alpha[i];
                } else if (!color_format) {
                    uint8_t red[16];
                    decode_bc4_block(block, red, signed_values);
                    for (unsigned i = 0; i < 16; ++i) rgba[i][0] = red[i];
                    if (two_channel) {
                        uint8_t green[16];
                        decode_bc4_block(block + 8, green, signed_values);
                        for (unsigned i = 0; i < 16; ++i) rgba[i][1] = green[i];
                    }
                }
                for (unsigned iy = 0; iy < 4 && by * 4 + iy < height; ++iy) {
                    for (unsigned ix = 0; ix < 4 && bx * 4 + ix < width; ++ix) {
                        const size_t pixel = ((size_t(z) * height + by * 4 + iy) * width + bx * 4 + ix);
                        std::memcpy(dst + pixel * 4, rgba[iy * 4 + ix], 4);
                    }
                }
            }
        }
    }
}
} // namespace rx::bcn
