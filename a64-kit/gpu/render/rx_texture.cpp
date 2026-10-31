// Textures and samplers: a shader's texture handle -> the texture header (TIC) and sampler header (TSC) in the bound
// pools -> a WebGPU texture view and sampler. A texture is the render target that memory holds when the GPU drew
// it, or else is decoded from guest memory (block-linear layout, ASTC, packed 16-bit formats) and kept while that
// memory does not change. Layout arithmetic is suyu's (video_core/texture_cache/util.cpp, image_info.cpp).
#include <algorithm>
#include <unordered_set>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstdint>
#include <map>
#include <numeric>
#include <span>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/workers.h"

#include "rx_bcn.h"
#include "rx_internal.h"

namespace rx {
namespace {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::DefaultBlockHeight;
using VideoCore::Surface::DefaultBlockWidth;
using Tegra::Texture::GOB_SIZE_SHIFT;
using Tegra::Texture::GOB_SIZE_X;
using Tegra::Texture::GOB_SIZE_X_SHIFT;
using Tegra::Texture::GOB_SIZE_Y;
using Tegra::Texture::GOB_SIZE_Y_SHIFT;
using Tegra::Texture::GOB_SIZE_Z;

// ---- suyu's block-linear layout arithmetic (texture_cache/util.cpp) ----
struct Extent2D {
    u32 width, height;
};
struct Extent3D {
    u32 width, height, depth;
};
struct LevelInfo {
    Extent3D size;
    Extent3D block;
    Extent2D tile_size;
    u32 bpp_log2;
    u32 tile_width_spacing;
    u32 num_levels;
};

constexpr u32 AdjustTileSize(u32 shift, u32 unit_factor, u32 dimension) {
    if (shift == 0) return 0;
    u32 x = unit_factor << (shift - 1);
    if (x >= dimension) {
        while (--shift) {
            x >>= 1;
            if (x < dimension) break;
        }
    }
    return shift;
}
constexpr u32 AdjustMipSize(u32 size, u32 level) {
    return std::max<u32>(size >> level, 1);
}
constexpr Extent3D AdjustMipSize(Extent3D size, u32 level) {
    return {AdjustMipSize(size.width, level), AdjustMipSize(size.height, level), AdjustMipSize(size.depth, level)};
}
template <u32 GOB_EXTENT>
constexpr u32 AdjustMipBlockSize(u32 num_tiles, u32 block_size, u32 level) {
    do {
        while (block_size > 0 && num_tiles <= (1U << (block_size - 1)) * GOB_EXTENT) --block_size;
    } while (level--);
    return block_size;
}
constexpr Extent3D AdjustMipBlockSize(Extent3D num_tiles, Extent3D block_size, u32 level, u32 num_levels) {
    return {AdjustMipBlockSize<GOB_SIZE_X>(num_tiles.width, block_size.width, level),
            AdjustMipBlockSize<GOB_SIZE_Y>(num_tiles.height, block_size.height, level),
            level == 0 && num_levels == 1 ? block_size.depth
                                          : AdjustMipBlockSize<GOB_SIZE_Z>(num_tiles.depth, block_size.depth, level)};
}
constexpr Extent3D AdjustTileSize(Extent3D size, Extent2D tile) {
    return {Common::DivCeil(size.width, tile.width), Common::DivCeil(size.height, tile.height), size.depth};
}
constexpr u32 BytesPerBlockLog2(u32 bytes) {
    return std::countl_zero(bytes) ^ 0x1F;
}
constexpr u32 AdjustSize(u32 size, u32 level, u32 block_size) {
    return Common::DivCeil(AdjustMipSize(size, level), block_size);
}
constexpr Extent3D NumLevelBlocks(const LevelInfo& info, u32 level) {
    return {AdjustSize(info.size.width, level, info.tile_size.width) << info.bpp_log2,
            AdjustSize(info.size.height, level, info.tile_size.height), AdjustMipSize(info.size.depth, level)};
}
constexpr Extent3D TileShift(const LevelInfo& info, u32 level) {
    if (level == 0 && info.num_levels == 1) return info.block;
    const Extent3D blocks = NumLevelBlocks(info, level);
    return {AdjustTileSize(info.block.width, GOB_SIZE_X, blocks.width),
            AdjustTileSize(info.block.height, GOB_SIZE_Y, blocks.height),
            AdjustTileSize(info.block.depth, GOB_SIZE_Z, blocks.depth)};
}
constexpr Extent2D GobSize(u32 bpp_log2, u32 block_height, u32 tile_width_spacing) {
    return {GOB_SIZE_X_SHIFT - bpp_log2 + tile_width_spacing, GOB_SIZE_Y_SHIFT + block_height};
}
constexpr bool IsSmallerThanGobSize(Extent3D num_tiles, Extent2D gob, u32 block_depth) {
    return num_tiles.width <= (1U << gob.width) || num_tiles.height <= (1U << gob.height) ||
           num_tiles.depth < (1U << block_depth);
}
constexpr u32 StrideAlignment(Extent3D num_tiles, Extent3D block, Extent2D gob, u32 bpp_log2) {
    return IsSmallerThanGobSize(num_tiles, gob, block.depth) ? GOB_SIZE_X_SHIFT - bpp_log2 : gob.width;
}
constexpr Extent2D NumGobs(const LevelInfo& info, u32 level) {
    const Extent3D blocks = NumLevelBlocks(info, level);
    const Extent2D gobs{Common::DivCeilLog2(blocks.width, GOB_SIZE_X_SHIFT),
                        Common::DivCeilLog2(blocks.height, GOB_SIZE_Y_SHIFT)};
    const Extent2D gob = GobSize(info.bpp_log2, info.block.height, info.tile_width_spacing);
    const bool is_small = IsSmallerThanGobSize(blocks, gob, info.block.depth);
    const u32 alignment = is_small ? 0 : info.tile_width_spacing;
    return {Common::AlignUpLog2(gobs.width, alignment), gobs.height};
}
constexpr Extent3D LevelTiles(const LevelInfo& info, u32 level) {
    const Extent3D blocks = NumLevelBlocks(info, level);
    const Extent3D tile_shift = TileShift(info, level);
    const Extent2D gobs = NumGobs(info, level);
    return {Common::DivCeilLog2(gobs.width, tile_shift.width), Common::DivCeilLog2(gobs.height, tile_shift.height),
            Common::DivCeilLog2(blocks.depth, tile_shift.depth)};
}
constexpr u32 CalculateLevelSize(const LevelInfo& info, u32 level) {
    const Extent3D tile_shift = TileShift(info, level);
    const Extent3D tiles = LevelTiles(info, level);
    const u32 num_tiles = tiles.width * tiles.height * tiles.depth;
    return num_tiles << (GOB_SIZE_SHIFT + tile_shift.width + tile_shift.height + tile_shift.depth);
}
constexpr u32 AlignLayerSize(u32 size_bytes, Extent3D size, Extent3D block, u32 tile_size_y,
                             u32 tile_width_spacing) {
    if (tile_width_spacing > 0) {
        const u32 alignment_log2 = GOB_SIZE_SHIFT + tile_width_spacing + block.height + block.depth;
        return Common::AlignUpLog2(size_bytes, alignment_log2);
    }
    const u32 aligned_height = Common::AlignUp(size.height, tile_size_y);
    while (block.height != 0 && aligned_height <= (1U << (block.height - 1)) * GOB_SIZE_Y) --block.height;
    while (block.depth != 0 && size.depth <= (1U << (block.depth - 1))) --block.depth;
    const u32 block_shift = GOB_SIZE_SHIFT + block.height + block.depth;
    const u32 num_blocks = size_bytes >> block_shift;
    return size_bytes != num_blocks << block_shift ? (num_blocks + 1) << block_shift : size_bytes;
}

// ---- texture descriptions ----
struct TexInfo {
    PixelFormat format{};
    WGPUTextureDimension dimension = WGPUTextureDimension_2D;
    Extent3D size{1, 1, 1};   // depth: 3D textures only
    u32 layers = 1, levels = 1;
    Extent3D block{0, 0, 0};
    u32 tile_width_spacing = 0;
    bool linear = false;
    u32 pitch = 0;
    bool valid = false;
};

TexInfo tex_info(const TICEntry& t) {
    TexInfo i;
    i.format = VideoCommon::PixelFormatFromTextureInfo(t.format, t.r_type, t.g_type, t.b_type, t.a_type,
                                                       t.srgb_conversion);
    i.levels = std::min<u32>(t.max_mip_level + 1, 16);
    i.linear = t.IsPitchLinear();
    if (i.linear) i.pitch = t.Pitch();
    else i.block = {t.block_width, t.block_height, t.block_depth};
    i.tile_width_spacing = t.tile_width_spacing;
    using T = Tegra::Texture::TextureType;
    T type = t.texture_type;
    if (t.Depth() > 1 || t.BaseLayer() != 0) {
        if (type == T::Texture1D) type = T::Texture1DArray;
        else if (type == T::Texture2D) type = T::Texture2DArray;
        else if (type == T::TextureCubemap) type = T::TextureCubeArray;
    }
    i.size = {t.Width(), t.Height(), 1};
    switch (type) {
    case T::Texture1D:
        i.dimension = WGPUTextureDimension_1D;
        i.size.height = 1;
        break;
    case T::Texture1DArray:
        i.dimension = WGPUTextureDimension_1D;
        i.size.height = 1;
        i.layers = t.BaseLayer() + t.Depth();
        break;
    case T::Texture2D:
    case T::Texture2DNoMipmap:
        i.layers = t.BaseLayer() + 1;
        break;
    case T::Texture2DArray:
        i.layers = t.BaseLayer() + t.Depth();
        break;
    case T::TextureCubemap:
        i.layers = t.BaseLayer() + 6;
        break;
    case T::TextureCubeArray:
        i.layers = t.BaseLayer() + t.Depth() * 6;
        break;
    case T::Texture3D:
        i.dimension = WGPUTextureDimension_3D;
        i.size.depth = t.Depth();
        break;
    default:
        return i;   // buffers: not textures
    }
    if (i.linear) i.levels = 1;
    if (i.dimension == WGPUTextureDimension_1D) i.levels = 1;   // WebGPU: 1D textures have one level
    i.valid = i.size.width && i.size.height;
    return i;
}

/// What a decoded texture is stored as when its format has no WebGPU equivalent.
enum class Convert { None, Astc, Packed16, Rgb32f, Bc1, Bc2, Bc3, Bc4, Bc4Snorm, Bc5, Bc5Snorm, Unsupported };
Convert conversion(PixelFormat f) {
    using P = PixelFormat;
    if (VideoCore::Surface::IsPixelFormatASTC(f)) return Convert::Astc;
    switch (f) {
    case P::R5G6B5_UNORM:
    case P::B5G6R5_UNORM:
    case P::A1R5G5B5_UNORM:
    case P::A1B5G5R5_UNORM:
    case P::A5B5G5R1_UNORM:
    case P::A4B4G4R4_UNORM:
    case P::G4R4_UNORM:
        return Convert::Packed16;
    case P::R32G32B32_FLOAT:
        return Convert::Rgb32f;
    case P::R16G16B16A16_UNORM: case P::R16G16B16A16_SNORM: case P::R16_UNORM: case P::R16_SNORM:
    case P::R16G16_UNORM: case P::R16G16_SNORM:
        return dev.norm16 ? Convert::None : Convert::Unsupported;
    case P::BC1_RGBA_UNORM: case P::BC1_RGBA_SRGB:
        return dev.bc ? Convert::None : Convert::Bc1;
    case P::BC2_UNORM: case P::BC2_SRGB:
        return dev.bc ? Convert::None : Convert::Bc2;
    case P::BC3_UNORM: case P::BC3_SRGB:
        return dev.bc ? Convert::None : Convert::Bc3;
    case P::BC4_UNORM:
        return dev.bc ? Convert::None : Convert::Bc4;
    case P::BC4_SNORM:
        return dev.bc ? Convert::None : Convert::Bc4Snorm;
    case P::BC5_UNORM:
        return dev.bc ? Convert::None : Convert::Bc5;
    case P::BC5_SNORM:
        return dev.bc ? Convert::None : Convert::Bc5Snorm;
    case P::BC7_UNORM: case P::BC7_SRGB: case P::BC6H_UFLOAT: case P::BC6H_SFLOAT:
        return dev.bc ? Convert::None : Convert::Unsupported;
    default:
        if (VideoCore::Surface::GetFormatType(f) != VideoCore::Surface::SurfaceType::ColorTexture &&
            !is_depth_format(wgpu_format(f)))
            return Convert::Unsupported;
        return wgpu_format(f) == WGPUTextureFormat_Undefined ? Convert::Unsupported : Convert::None;
    }
}

uint8_t expand(uint32_t v, int bits) {
    return bits == 0 ? 0 : (uint8_t)((v * 255u + ((1u << bits) - 1) / 2) / ((1u << bits) - 1));
}

void convert_packed16(PixelFormat f, const uint8_t* src, uint8_t* dst, size_t pixels) {
    using P = PixelFormat;
    for (size_t i = 0; i < pixels; i++) {
        uint32_t v = f == P::G4R4_UNORM ? src[i] : (uint32_t)(src[2 * i] | src[2 * i + 1] << 8);
        uint8_t r = 0, g = 0, b = 0, a = 255;
        switch (f) {
        case P::R5G6B5_UNORM: r = expand(v >> 11, 5); g = expand(v >> 5 & 63, 6); b = expand(v & 31, 5); break;
        case P::B5G6R5_UNORM: b = expand(v >> 11, 5); g = expand(v >> 5 & 63, 6); r = expand(v & 31, 5); break;
        case P::A1R5G5B5_UNORM:
            a = v >> 15 ? 255 : 0; r = expand(v >> 10 & 31, 5); g = expand(v >> 5 & 31, 5); b = expand(v & 31, 5);
            break;
        case P::A1B5G5R5_UNORM:
            a = v >> 15 ? 255 : 0; b = expand(v >> 10 & 31, 5); g = expand(v >> 5 & 31, 5); r = expand(v & 31, 5);
            break;
        case P::A5B5G5R1_UNORM:
            r = v & 1 ? 255 : 0; g = expand(v >> 1 & 31, 5); b = expand(v >> 6 & 31, 5); a = expand(v >> 11, 5);
            break;
        case P::A4B4G4R4_UNORM:
            r = expand(v & 15, 4); g = expand(v >> 4 & 15, 4); b = expand(v >> 8 & 15, 4); a = expand(v >> 12, 4);
            break;
        case P::G4R4_UNORM: r = expand(v & 15, 4); g = expand(v >> 4, 4); break;
        default: break;
        }
        dst[4 * i] = r;
        dst[4 * i + 1] = g;
        dst[4 * i + 2] = b;
        dst[4 * i + 3] = a;
    }
}

/// Applies the header's component swizzle to RGBA8 texels.
void swizzle_rgba8(const TICEntry& t, uint8_t* p, size_t pixels) {
    using S = Tegra::Texture::SwizzleSource;
    const S src[4] = {t.x_source, t.y_source, t.z_source, t.w_source};
    if (src[0] == S::R && src[1] == S::G && src[2] == S::B && src[3] == S::A) return;
    for (size_t i = 0; i < pixels; i++) {
        uint8_t in[4] = {p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]};
        for (int c = 0; c < 4; c++) {
            switch (src[c]) {
            case S::Zero: p[4 * i + c] = 0; break;
            case S::R: p[4 * i + c] = in[0]; break;
            case S::G: p[4 * i + c] = in[1]; break;
            case S::B: p[4 * i + c] = in[2]; break;
            case S::A: p[4 * i + c] = in[3]; break;
            default: p[4 * i + c] = 255; break;
            }
        }
    }
}

std::unordered_map<uint64_t, std::unique_ptr<Image>> decoded;   // by TIC contents
uint64_t decoded_generation;   // bumped whenever the decoded-texture map changes
// The same, by the 1 MB pieces of guest memory each covers (textures_written runs per written page: a walk back
// over every texture within the largest one's size was ~4% of the GPU thread)
constexpr uint64_t kBucketShift = 20;
std::unordered_map<uint64_t, std::vector<Image*>> decoded_by_bucket;

void remember(Image* im) {
    const uint64_t last = (im->guest + std::max<uint64_t>(im->guest_bytes, 1) - 1) >> kBucketShift;
    for (uint64_t b = im->guest >> kBucketShift; b <= last; b++) decoded_by_bucket[b].push_back(im);
}

void forget(Image* im) {
    const uint64_t last = (im->guest + std::max<uint64_t>(im->guest_bytes, 1) - 1) >> kBucketShift;
    for (uint64_t b = im->guest >> kBucketShift; b <= last; b++) {
        auto it = decoded_by_bucket.find(b);
        if (it == decoded_by_bucket.end()) continue;
        auto& v = it->second;
        v.erase(std::remove(v.begin(), v.end(), im), v.end());
        if (v.empty()) decoded_by_bucket.erase(it);
    }
}
std::unordered_map<uint64_t, WGPUSampler> samplers;

} // namespace

namespace { uint64_t sample_hash(uint64_t va, uint64_t size); }

// RX_DUMP_TEX=texels: the textures the game last wrote with WriteTexels (rx_nvn.cpp) instead of a list of addresses
static std::mutex texels_mu;
static std::vector<uint64_t> texels_written;
void note_texels_written(uint64_t guest) {
    static const bool on = getenv("RX_DUMP_TEX") && !strcmp(getenv("RX_DUMP_TEX"), "texels");
    if (!on || !guest) return;
    std::lock_guard lk(texels_mu);
    if (std::find(texels_written.begin(), texels_written.end(), guest) != texels_written.end()) return;
    texels_written.push_back(guest);
    if (texels_written.size() > 16) texels_written.erase(texels_written.begin());
}

void dump_tex_list(const char* dir) {
    const char* s = getenv("RX_DUMP_TEX");
    if (!s) return;
    std::string from_texels;
    if (!strcmp(s, "texels")) {
        std::lock_guard lk(texels_mu);
        for (uint64_t g : texels_written) {
            char b[24];
            std::snprintf(b, sizeof b, "%llx,", (unsigned long long)g);
            from_texels += b;
        }
        s = from_texels.c_str();
    }
    int order = 800;
    for (const char* p = s; *p;) {
        char* end;
        const uint64_t va = strtoull(p, &end, 16);
        if (end == p) break;
        p = *end == ',' ? end + 1 : end;
        for (auto& [key, im] : decoded)
            if (im && (im->gpu_va == va || im->guest == va) && im->tex) {
                dump_texture(im->tex, va, im->width, im->height, im->format, dir, order++);
                // what guest memory holds now, against the decoded copy
                std::vector<uint8_t> mem(im->guest_bytes);
                gpu_read(im->gpu_va, mem.data(), mem.size());
                size_t nonzero = 0;
                for (uint8_t b : mem) nonzero += b != 0;
                char path[512];
                std::snprintf(path, sizeof path, "%s/tex_%llx_guest.bin", dir, (unsigned long long)va);
                if (FILE* f = std::fopen(path, "wb")) { std::fwrite(mem.data(), 1, mem.size(), f); std::fclose(f); }
                hos_log("rx-frame: texture %#llx (gpu %#llx) %ux%u format %d saved; guest %llu bytes, %zu nonzero; stale %d "
                        "decoded_seq %llu now %llu; hash %s; rendered %d; layers %u dim %d levels %u", (unsigned long long)va, (unsigned long long)im->gpu_va, im->width,
                        im->height, (int)im->format, (unsigned long long)im->guest_bytes, nonzero, (int)im->stale,
                        (unsigned long long)im->decoded_seq, (unsigned long long)memory_seq(),
                        sample_hash(im->gpu_va, im->guest_bytes) == im->content_hash ? "same" : "changed",
                        (int)im->rendered, im->layers, (int)im->dimension, im->levels);
            }
        // a render target there: what its guest memory holds (the GPU's image is what draws see)
        for (const auto& [g, im] : all_images())
            if (im && (im->gpu_va == va || im->guest == va)) {
                const uint64_t bytes = (uint64_t)im->width * im->height * BytesPerBlock(im->pixel_format);
                std::vector<uint8_t> mem(bytes);
                gpu_read(im->gpu_va, mem.data(), mem.size());
                size_t nonzero = 0;
                for (uint8_t b : mem) nonzero += b != 0;
                char path[512];
                std::snprintf(path, sizeof path, "%s/rt_%llx_guest.bin", dir, (unsigned long long)va);
                if (FILE* f = std::fopen(path, "wb")) { std::fwrite(mem.data(), 1, mem.size(), f); std::fclose(f); }
                hos_log("rx-frame: render target %#llx (gpu %#llx) %ux%u pixel format %d: guest %llu bytes, %zu nonzero; "
                        "rendered %d version %llu", (unsigned long long)va, (unsigned long long)im->gpu_va, im->width,
                        im->height, (int)im->pixel_format, (unsigned long long)bytes, nonzero, (int)im->rendered,
                        (unsigned long long)im->version);
            }
    }
}

namespace {

/// A few samples of a texture's memory: what tells that the game wrote new contents.
uint64_t sample_hash(uint64_t va, uint64_t size) {
    uint64_t h = 0xcbf29ce484222325ull;
    uint8_t buf[64];
    const uint64_t n = std::min<uint64_t>(size, 64);
    for (int i = 0; i < 16; i++) {
        const uint64_t off = size > n ? (size - n) * i / 15 & ~7ull : 0;
        gpu_read(va + off, buf, n);
        h = hash_bytes(buf, n, h);
    }
    return h;
}

/// Where a (layer, level) of a block-linear texture starts, from its address.
uint64_t subresource_offset(const TexInfo& info, uint32_t layer, uint32_t level) {
    const u32 bpp = BytesPerBlock(info.format);
    const Extent2D tile{DefaultBlockWidth(info.format), DefaultBlockHeight(info.format)};
    const LevelInfo li{info.size, info.block, tile, BytesPerBlockLog2(bpp), info.tile_width_spacing, info.levels};
    u32 layer_size = 0, offset = 0;
    for (u32 l = 0; l < info.levels; l++) {
        if (l == level) offset = layer_size;
        layer_size += CalculateLevelSize(li, l);
    }
    const u32 stride = AlignLayerSize(layer_size, info.size, li.block, tile.height, info.tile_width_spacing);
    return (uint64_t)layer * stride + offset;
}

uint64_t guest_size(const TexInfo& info) {
    const u32 bpp = BytesPerBlock(info.format);
    const Extent2D tile{DefaultBlockWidth(info.format), DefaultBlockHeight(info.format)};
    if (info.linear) return (uint64_t)info.pitch * Common::DivCeil(info.size.height, tile.height);
    const LevelInfo li{info.size, info.block, tile, BytesPerBlockLog2(bpp), info.tile_width_spacing, info.levels};
    u32 layer = 0;
    for (u32 l = 0; l < info.levels; l++) layer += CalculateLevelSize(li, l);
    const u32 stride = AlignLayerSize(layer, info.size, li.block, tile.height, info.tile_width_spacing);
    return (uint64_t)stride * (info.layers - 1) + layer;
}

/// The browser: WebGPU objects belong to the render thread, so a decode worker describes the texture
/// (thread_local deferring) and streams its decoded layers to the render thread, which makes the texture at the first
/// one and writes them at each present. At most kDeferredBudget bytes wait (the host heap above the guest window is
/// ~800 MB: a 350 MB ASTC array decoded whole ran it out), so a big texture arrives over a few frames.
struct DeferredUpload {
    uint32_t level = 0, origin_y = 0, origin_z = 0;
    WGPUTexelCopyBufferLayout layout{};
    WGPUExtent3D extent{};
    std::vector<uint8_t> data;
};
struct DeferredTexture {
    WGPUTextureDescriptor desc{};
    WGPUTextureFormat pair = WGPUTextureFormat_Undefined;
    WGPUTexture tex = nullptr;   // (render thread)
};
struct StreamedUpload {
    DeferredTexture* texture;
    DeferredUpload upload;
};
thread_local DeferredTexture* deferring;
thread_local bool gpu_force_cpu;   // (RX_GPU_VERIFY: the CPU decode to compare with)
std::unordered_set<WGPUSampler> nonfiltering_samplers;   // (debugging the browser's sampler checks)
std::unordered_set<uint64_t> rewritten_textures;   // guest addresses whose background decode was rewritten meanwhile
thread_local char decoding_what[160];   // (debugging: HOS_BIG_ALLOC_MB names the texture being decoded)
extern "C" const char* rx_decoding_what() { return decoding_what; }
#ifdef __EMSCRIPTEN__
extern "C" void* sbrk(intptr_t);
uintptr_t heap_top() { return (uintptr_t)sbrk(0); }
#else
uintptr_t heap_top() { return 0; }
#endif
std::mutex decode_mu;
std::condition_variable decode_cv;
std::deque<StreamedUpload> upload_stream;   // (decode_mu)
std::atomic<int64_t> deferred_bytes{0};
constexpr int64_t kDeferredBudget = 32ll << 20;

void stream_upload(DeferredUpload&& u) {
    const int64_t n = (int64_t)u.data.size();
    std::unique_lock lk(decode_mu);
    decode_cv.wait(lk, [n] {
        const int64_t b = deferred_bytes.load();
        return b == 0 || b + n <= kDeferredBudget;
    });
    deferred_bytes += n;
    upload_stream.push_back({deferring, std::move(u)});
}

WGPUTexture deferred_texture(DeferredTexture& t) {   // (render thread)
    if (!t.tex) {
        if (t.pair != WGPUTextureFormat_Undefined) t.desc.viewFormats = &t.pair;
        t.tex = wgpuDeviceCreateTexture(dev.device, &t.desc);
    }
    return t.tex;
}

/// BC textures whose size is not a multiple of 4 stay BC: the texture is made a whole number of blocks (3686 ->
/// 3688: 0.05% of stretch) and each level's copy fits its padded size. Decoded to RGBA8 on the CPU they were 8x the
/// memory and a 108 MB buffer for TOTK's 7373x3686 maps (more than the browser build's heap had free).
/// RX_BC_DECODE_ODD=1: decoded as before.
bool odd_bc_decoded() {
    static const bool on = env_flag("RX_BC_DECODE_ODD");
    return on;
}

/// Decodes the texture into a new WebGPU texture.
std::unique_ptr<Image> decode(const TICEntry& tic, const TexInfo& info, uint64_t va) {
    const Convert cv = conversion(info.format);
    if (cv == Convert::Unsupported) {
        log_limited("tex_format", 24, "rx: texture format %d not decoded", (int)info.format);
        return nullptr;
    }
    const WGPUTextureFormat guest_format = wgpu_format(info.format);
    const u32 bpp = BytesPerBlock(info.format);
    const Extent2D tile{DefaultBlockWidth(info.format), DefaultBlockHeight(info.format)};
    if (info.size.width > dev.limits.maxTextureDimension2D || info.size.height > dev.limits.maxTextureDimension2D ||
        info.layers > dev.limits.maxTextureArrayLayers)
        return nullptr;
    const bool guest_compressed = VideoCore::Surface::IsPixelFormatBCn(info.format);
    const u32 physical_width = guest_compressed ? Common::AlignUp(info.size.width, 4) : info.size.width;
    const u32 physical_height = guest_compressed ? Common::AlignUp(info.size.height, 4) : info.size.height;
    const bool has_cpu_bc_decoder =
        info.format == PixelFormat::BC1_RGBA_UNORM || info.format == PixelFormat::BC1_RGBA_SRGB ||
        info.format == PixelFormat::BC2_UNORM || info.format == PixelFormat::BC2_SRGB ||
        info.format == PixelFormat::BC3_UNORM || info.format == PixelFormat::BC3_SRGB ||
        info.format == PixelFormat::BC4_UNORM || info.format == PixelFormat::BC4_SNORM ||
        info.format == PixelFormat::BC5_UNORM || info.format == PixelFormat::BC5_SNORM;
    const bool decode_bc = cv == Convert::Bc1 || cv == Convert::Bc2 || cv == Convert::Bc3 || cv == Convert::Bc4 ||
                           cv == Convert::Bc4Snorm || cv == Convert::Bc5 || cv == Convert::Bc5Snorm ||
                           (guest_compressed && has_cpu_bc_decoder &&
                            ((odd_bc_decoded() && ((info.size.width & 3) || (info.size.height & 3))) ||
                             info.dimension == WGPUTextureDimension_3D));
    const bool decode_any_compressed = decode_bc;
    static const bool astc_bc7 = env_flag("RX_ASTC_BC7");
    const bool to_bc7 = astc_bc7 && dev.bc && cv == Convert::Astc && info.dimension == WGPUTextureDimension_2D &&
                        !(info.size.width & 3) && !(info.size.height & 3);
    const WGPUTextureFormat format = decode_any_compressed
                                         ? (VideoCore::Surface::IsPixelFormatSRGB(info.format)
                                                ? WGPUTextureFormat_RGBA8UnormSrgb
                                                : WGPUTextureFormat_RGBA8Unorm)
                                     : to_bc7 ? (VideoCore::Surface::IsPixelFormatSRGB(info.format)
                                                     ? WGPUTextureFormat_BC7RGBAUnormSrgb
                                                     : WGPUTextureFormat_BC7RGBAUnorm)
                                              : guest_format;

    const uint64_t total = guest_size(info);
    // made by the GPU from the guest's bytes as they are (rx_texture_gpu.cpp), 2D (arrays) in block-linear layout:
    // - ASTC decoded there: the texture is RGBA8Unorm, as storage textures cannot be sRGB, with the sRGB encoding as
    //   its view format (its Image keeps the guest's encoding: views and copies are made in it)
    // - other textures uploaded as they are (BC, 4-16 byte texels), from 256 KB: unswizzled there
    const bool gpu_layout_ok = !gpu_force_cpu && !deferring && !info.linear && info.dimension == WGPUTextureDimension_2D &&
                               info.size.depth == 1 && total <= dev.limits.maxStorageBufferBindingSize;
    const bool astc_gpu = gpu_layout_ok && cv == Convert::Astc && !to_bc7 && astc_gpu_on();
    const bool swizzle_gpu = gpu_layout_ok && cv == Convert::None && !decode_any_compressed && !to_bc7 &&
                             (bpp == 4 || bpp == 8 || bpp == 16) && !is_depth_format(format) && total >= (256u << 10) &&
                             unswizzle_gpu_on();
    std::vector<uint8_t> scratch;
    std::snprintf(decoding_what, sizeof decoding_what, "%ux%u x%u depth %u levels %u format %d linear %d total %llu",
                  info.size.width, info.size.height, info.layers, info.size.depth, info.levels, (int)info.format,
                  (int)info.linear, (unsigned long long)total);
    const uint8_t* src = gpu_span(va, total);
    if (total > (32u << 20)) {
        static int logged;
        if (logged++ < 20)
            hos_log("rx: big texture %ux%u x%u (depth %u) levels %u format %d: %llu MB in guest memory%s", info.size.width,
                    info.size.height, info.layers, info.size.depth, info.levels, (int)info.format,
                    (unsigned long long)(total >> 20), src ? "" : " (copied: not one mapping)");
    }
    if (!src) {
        scratch.resize(total);
        gpu_read(va, scratch.data(), total);
        src = scratch.data();
    }

    auto im = std::make_unique<Image>();
    im->gpu_va = va;
    im->guest = hos_gpu_translate(va);
    im->width = info.size.width;
    im->height = info.size.height;
    im->layers = info.dimension == WGPUTextureDimension_3D ? info.size.depth : info.layers;
    im->levels = info.levels;
    im->pixel_format = info.format;
    im->format = format;
    im->storage = astc_gpu;
    im->dimension = info.dimension;
    WGPUTextureDescriptor d{};
    d.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc |
              (astc_gpu ? WGPUTextureUsage_StorageBinding : 0);
    d.dimension = info.dimension;
    d.size = {decode_any_compressed ? info.size.width : physical_width,
              decode_any_compressed ? info.size.height : physical_height, im->layers};
    d.format = astc_gpu ? WGPUTextureFormat_RGBA8Unorm : format;
    d.mipLevelCount = info.levels;
    d.sampleCount = 1;
    const WGPUTextureFormat pair = srgb_pair(d.format);
    if (pair != WGPUTextureFormat_Undefined) {
        d.viewFormatCount = 1;
        d.viewFormats = &pair;
    }
    if (deferring) {
        deferring->desc = d;
        deferring->pair = pair;
    } else {
        scope_push();
        im->tex = wgpuDeviceCreateTexture(dev.device, &d);
    }
    if (const std::string error = deferring ? std::string() : scope_pop(); !error.empty()) {
        log_limited("tex_create", 16, "rx: texture %ux%u x%u format %d not created: %s", info.size.width,
                    info.size.height, info.layers, (int)info.format, error.c_str());
        return nullptr;
    }

    const LevelInfo li{info.size, info.block, tile, BytesPerBlockLog2(bpp), info.tile_width_spacing, info.levels};
    u32 layer_size = 0;
    for (u32 l = 0; l < info.levels; l++) layer_size += CalculateLevelSize(li, l);
    const u32 layer_stride = AlignLayerSize(layer_size, info.size, li.block, tile.height, info.tile_width_spacing);
    const Extent2D gob = GobSize(li.bpp_log2, info.block.height, info.tile_width_spacing);
    std::vector<uint8_t> level_data, converted;
    size_t level_offset = 0;
    const u32 copy_layers = info.dimension == WGPUTextureDimension_3D ? 1 : info.layers;
    bool gpu_decoded = false;
    if (astc_gpu || swizzle_gpu) {
        std::vector<BlockLinearLevel> lv;
        size_t off = 0;
        for (u32 level = 0; level < info.levels; level++) {
            const Extent3D tiles = AdjustTileSize(AdjustMipSize(info.size, level), tile);
            const Extent3D block = AdjustMipBlockSize(tiles, li.block, level, li.num_levels);
            const u32 stride = Common::AlignUpLog2(tiles.width, StrideAlignment(tiles, info.block, gob, li.bpp_log2)) * bpp;
            const u32 x_shift = 9 + block.height + block.depth;   // (a GOB is 512 bytes, 64 wide)
            lv.push_back({(uint32_t)off, tiles.width, tiles.height, ((stride + 63) >> 6) << x_shift, x_shift, block.height});
            off += CalculateLevelSize(li, level);
        }
        const int64_t t0 = hos_now_ns();
        gpu_decoded = astc_gpu ? astc_gpu_decode(im->tex, src, total, lv.data(), (int)lv.size(), copy_layers,
                                                 layer_stride, tile.width, tile.height)
                               : unswizzle_gpu(im->tex, src, total, lv.data(), (int)lv.size(), copy_layers, layer_stride,
                                               li.bpp_log2, tile.width, tile.height);
        if (static const bool log_inplace = env_flag("RX_LOG_INPLACE"); log_inplace && total > (1u << 20))
            hos_log("rx: %s %ux%u x%u levels %u format %d on the GPU: %.2f ms, %llu KB", astc_gpu ? "ASTC" : "unswizzle",
                    info.size.width, info.size.height, copy_layers, info.levels, (int)info.format,
                    (hos_now_ns() - t0) / 1e6, (unsigned long long)(total >> 10));
        // RX_GPU_VERIFY=1 (or RX_ASTC_VERIFY=1): each texture also made by the CPU, and the two compared (level by
        // level, first and last layer; ASTC within 1 of 255: the GPU rounds where the CPU truncates)
        static const bool verify = env_flag("RX_GPU_VERIFY") || env_flag("RX_ASTC_VERIFY");
        if (verify && gpu_decoded) {
            gpu_force_cpu = true;
            const auto cpu = decode(tic, info, va);
            gpu_force_cpu = false;
            for (u32 level = 0; cpu && level < info.levels; level++) {
                const Extent3D ls = AdjustMipSize(info.size, level);
                const Extent3D tiles = AdjustTileSize(ls, tile);
                // (RGBA8 texels for ASTC; whole blocks as they are otherwise)
                const uint32_t w = astc_gpu ? ls.width : tiles.width * tile.width;
                const uint32_t h = astc_gpu ? ls.height : tiles.height * tile.height;
                const uint32_t row = astc_gpu ? ls.width * 4 : tiles.width * bpp, rows = astc_gpu ? ls.height : tiles.height;
                const int tolerance = astc_gpu ? 1 : 0;
                for (u32 layer : {0u, copy_layers - 1}) {
                    const auto a = read_level(im->tex, level, layer, w, h, row, rows);
                    const auto b = read_level(cpu->tex, level, layer, w, h, row, rows);
                    size_t bad = 0, first = SIZE_MAX;
                    int worst = 0;
                    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
                        const int diff = std::abs((int)a[i] - (int)b[i]);
                        if (diff > tolerance) {
                            bad++;
                            if (first == SIZE_MAX) first = i;
                        }
                        worst = std::max(worst, diff);
                    }
                    static int logged;
                    if ((bad || a.size() != b.size()) && logged++ < 60)
                        hos_log("rx: GPU verify (%s) %ux%u x%u format %d level %u layer %u: %zu of %zu bytes differ "
                                "(worst %d), first at byte %zu of row %zu", astc_gpu ? "ASTC" : "unswizzle",
                                info.size.width, info.size.height, copy_layers, (int)info.format, level, layer, bad,
                                a.size(), worst, first == SIZE_MAX ? 0 : first % row,
                                first == SIZE_MAX ? 0 : first / row);
                    static std::atomic<int64_t> compared, differ;
                    const int64_t n = ++compared, nd = differ += bad || a.size() != b.size() || a.empty();
                    if (n % 200 == 0 || n == 20)
                        hos_log("rx: GPU verify: %lld level/layer pairs compared, %lld differ", (long long)n,
                                (long long)nd);
                    if (copy_layers == 1) break;
                }
            }
        }
        if (gpu_decoded && !rx_side_thread) spike.upload_bytes += (int64_t)total;
    }
    for (u32 level = 0; !gpu_decoded && level < info.levels; level++) {
        const Extent3D level_size = AdjustMipSize(info.size, level);
        const Extent3D tiles = AdjustTileSize(level_size, tile);
        const size_t layer_bytes = (size_t)tiles.width * tiles.height * tiles.depth * bpp;
        // ASTC arrays: the layers of a level decoded at once, each by a thread of its own (unswizzle, decode,
        // upload). One after another, with the ASTC pool waiting between small slices, the terrain's 1024x1024 x121
        // material array took 1.2 s of the title screen's first frame (RX_SERIAL_LAYERS=1: the old way). Not on the
        // web: there only the render thread may call WebGPU, and each worker uploads its own layer.
        static const bool serial_layers = RX_WEB || env_flag("RX_SERIAL_LAYERS");
        if (!serial_layers && cv == Convert::Astc && !deferring && !to_bc7 && !decode_any_compressed && !info.linear &&
            copy_layers >= 4 && level_size.depth == 1 && (size_t)level_size.width * level_size.height * 4 <= (8u << 20)) {
            const Extent3D block = AdjustMipBlockSize(tiles, li.block, level, li.num_levels);
            const u32 stride_alignment = StrideAlignment(tiles, info.block, gob, li.bpp_log2);
            const unsigned nthreads = std::max(1u, std::min<unsigned>(copy_layers, std::thread::hardware_concurrency() - 1));
            std::atomic<u32> next_layer{0};
            static std::atomic<int64_t> t_swz, t_dec, t_up;
            auto work = [&] {
                std::vector<uint8_t> swz, rgba((size_t)level_size.width * level_size.height * 4);
                for (u32 layer; (layer = next_layer++) < copy_layers;) {
                    const uint64_t off = level_offset + (uint64_t)layer * layer_stride;
                    if (off >= total) continue;
                    const int64_t a0 = hos_now_ns();
                    swz.assign(layer_bytes, 0);
                    Tegra::Texture::UnswizzleTexture(swz, std::span<const u8>(src + off, total - off), bpp, tiles.width,
                                                     tiles.height, tiles.depth, block.height, block.depth,
                                                     stride_alignment);
                    const int64_t a1 = hos_now_ns();
                    Tegra::Texture::ASTC::DecompressSerial(swz, level_size.width, level_size.height, 1, tile.width,
                                                           tile.height, rgba);
                    const int64_t a2 = hos_now_ns();
                    t_swz += a1 - a0;
                    t_dec += a2 - a1;
                    WGPUTexelCopyTextureInfo d{};
                    d.texture = im->tex;
                    d.mipLevel = level;
                    d.origin = {0, 0, layer};
                    d.aspect = WGPUTextureAspect_All;
                    WGPUTexelCopyBufferLayout lay{};
                    lay.bytesPerRow = level_size.width * 4;
                    lay.rowsPerImage = level_size.height;
                    const WGPUExtent3D ext{level_size.width, level_size.height, 1};
                    const int64_t a3 = hos_now_ns();
                    wgpuQueueWriteTexture(dev.queue, &d, rgba.data(), rgba.size(), &lay, &ext);
                    t_up += hos_now_ns() - a3;
                }
            };
            std::vector<std::thread> threads;
            for (unsigned t = 1; t < nthreads; t++) threads.emplace_back(work);
            work();
            for (std::thread& t : threads) t.join();
            if (static const bool log_inplace = env_flag("RX_LOG_INPLACE"); log_inplace && level == info.levels - 1)
                hos_log("rx: parallel layers %ux%u x%u: thread time unswizzle %.0f ms, ASTC %.0f ms, upload %.0f ms "
                        "(%u threads)", info.size.width, info.size.height, copy_layers, t_swz.exchange(0) / 1e6,
                        t_dec.exchange(0) / 1e6, t_up.exchange(0) / 1e6, nthreads);
            if (!rx_side_thread) spike.upload_bytes += (int64_t)level_size.width * level_size.height * 4 * copy_layers;
            level_offset += CalculateLevelSize(li, level);
            continue;
        }
        for (u32 layer = 0; layer < copy_layers; layer++) {
            level_data.assign(layer_bytes, 0);
            if (info.linear) {
                for (u32 y = 0; y < tiles.height; y++) {
                    const uint64_t off = (uint64_t)y * info.pitch;
                    if (off + (uint64_t)tiles.width * bpp <= total)
                        std::memcpy(level_data.data() + (size_t)y * tiles.width * bpp, src + off,
                                    (size_t)tiles.width * bpp);
                }
            } else {
                const Extent3D block = AdjustMipBlockSize(tiles, li.block, level, li.num_levels);
                const u32 stride_alignment = StrideAlignment(tiles, info.block, gob, li.bpp_log2);
                const uint64_t off = level_offset + (uint64_t)layer * layer_stride;
                if (off >= total) break;
                Tegra::Texture::UnswizzleTexture(level_data, std::span<const u8>(src + off, total - off), bpp,
                                                 tiles.width, tiles.height, tiles.depth, block.height, block.depth,
                                                 stride_alignment);
            }
            const uint8_t* upload = level_data.data();
            uint32_t row_bytes = tiles.width * bpp, rows = tiles.height;
            if (decode_any_compressed) {
                const size_t pixels = (size_t)level_size.width * level_size.height * level_size.depth;
                converted.assign(pixels * 4, 0);
                bcn::Format bc_format;
                switch (info.format) {
                case PixelFormat::BC1_RGBA_UNORM: case PixelFormat::BC1_RGBA_SRGB: bc_format = bcn::Format::BC1; break;
                case PixelFormat::BC2_UNORM: case PixelFormat::BC2_SRGB: bc_format = bcn::Format::BC2; break;
                case PixelFormat::BC3_UNORM: case PixelFormat::BC3_SRGB: bc_format = bcn::Format::BC3; break;
                case PixelFormat::BC4_SNORM: bc_format = bcn::Format::BC4Snorm; break;
                case PixelFormat::BC5_UNORM: bc_format = bcn::Format::BC5; break;
                case PixelFormat::BC5_SNORM: bc_format = bcn::Format::BC5Snorm; break;
                default: bc_format = bcn::Format::BC4; break;
                }
                bcn::decode(level_data.data(), level_size.width, level_size.height, level_size.depth, bc_format,
                            converted.data());
                upload = converted.data();
                row_bytes = level_size.width * 4;
                rows = level_size.height;
            }
            // a band of rows (y0, nrows) of RGBA8 for this level and layer, to the texture (or the render thread)
            auto write_band = [&](std::vector<uint8_t>& data, uint32_t y0, uint32_t nrows) {
                WGPUTexelCopyBufferLayout lay{};
                lay.bytesPerRow = level_size.width * 4;
                lay.rowsPerImage = nrows;
                const WGPUExtent3D ext{level_size.width, nrows, 1};
                const uint32_t z = info.dimension == WGPUTextureDimension_3D ? 0 : layer;
                if (deferring) {
                    DeferredUpload u;
                    u.level = level;
                    u.origin_y = y0;
                    u.origin_z = z;
                    u.layout = lay;
                    u.extent = ext;
                    u.data = std::move(data);
                    stream_upload(std::move(u));
                } else {
                    WGPUTexelCopyTextureInfo d{};
                    d.texture = im->tex;
                    d.mipLevel = level;
                    d.origin = {0, y0, z};
                    d.aspect = WGPUTextureAspect_All;
                    wgpuQueueWriteTexture(dev.queue, &d, data.data(), data.size(), &lay, &ext);
                }
            };
            auto write_bc7_band = [&](const std::vector<uint8_t>& rgba, uint32_t y0, uint32_t nrows) {
                const uint32_t bw = (level_size.width + 3) / 4, bh = (nrows + 3) / 4;
                std::vector<uint8_t> bc((size_t)bw * bh * 16);
                bc7_encode(rgba.data(), level_size.width, nrows, level_size.width * 4, bc.data());
                WGPUTexelCopyBufferLayout lay{};
                lay.bytesPerRow = bw * 16;
                lay.rowsPerImage = bh;
                const WGPUExtent3D ext{bw * 4, bh * 4, 1};
                if (deferring) {
                    DeferredUpload u;
                    u.level = level;
                    u.origin_y = y0;
                    u.origin_z = layer;
                    u.layout = lay;
                    u.extent = ext;
                    u.data = std::move(bc);
                    stream_upload(std::move(u));
                } else {
                    WGPUTexelCopyTextureInfo d{};
                    d.texture = im->tex;
                    d.mipLevel = level;
                    d.origin = {0, y0, layer};
                    d.aspect = WGPUTextureAspect_All;
                    wgpuQueueWriteTexture(dev.queue, &d, bc.data(), bc.size(), &lay, &ext);
                }
            };
            bool banded = false;
            switch (decode_any_compressed ? Convert::None : cv) {
            case Convert::Astc:
                if (level_size.depth == 1 && (size_t)level_size.width * level_size.height * 4 > (8u << 20)) {
                    // (in bands of ~8 MB: a 7373x3686 level is 108 MB of RGBA8, more than the browser build's host
                    // heap had free at once)
                    uint32_t brows = std::max<uint32_t>(1, (8u << 20) / (level_size.width * 4 * tile.height));
                    if (to_bc7) {
                        const uint32_t step = 4 / std::gcd(tile.height, 4u);
                        brows = std::max(step, brows / step * step);
                    }
                    for (uint32_t by = 0; by < tiles.height; by += brows) {
                        const uint32_t nb = std::min(brows, tiles.height - by);
                        const uint32_t y0 = by * tile.height;
                        const uint32_t h = std::min(nb * tile.height, level_size.height - y0);
                        const size_t off = (size_t)by * tiles.width * 16, len = (size_t)nb * tiles.width * 16;
                        if (off + len > level_data.size()) break;
                        std::vector<uint8_t> band((size_t)level_size.width * h * 4);
                        Tegra::Texture::ASTC::Decompress(std::span<const uint8_t>(level_data.data() + off, len),
                                                         level_size.width, h, 1, tile.width, tile.height, band);
                        if (to_bc7) write_bc7_band(band, y0, h);
                        else write_band(band, y0, h);
                    }
                    banded = true;
                    break;
                }
                converted.assign((size_t)level_size.width * level_size.height * level_size.depth * 4, 0);
                Tegra::Texture::ASTC::Decompress(level_data, level_size.width, level_size.height, level_size.depth,
                                                 tile.width, tile.height, converted);
                upload = converted.data();
                row_bytes = level_size.width * 4;
                rows = level_size.height;
                if (to_bc7) {
                    const uint32_t bw = (level_size.width + 3) / 4, bh = (level_size.height + 3) / 4;
                    std::vector<uint8_t> bc((size_t)bw * bh * 16);
                    bc7_encode(converted.data(), level_size.width, level_size.height, level_size.width * 4, bc.data());
                    converted.swap(bc);
                    upload = converted.data();
                    row_bytes = bw * 16;
                    rows = bh;
                }
                break;
            case Convert::Packed16:
                converted.assign((size_t)tiles.width * tiles.height * tiles.depth * 4, 0);
                convert_packed16(info.format, level_data.data(), converted.data(),
                                 (size_t)tiles.width * tiles.height * tiles.depth);
                upload = converted.data();
                row_bytes = tiles.width * 4;
                break;
            case Convert::Rgb32f: {
                const size_t n = (size_t)tiles.width * tiles.height * tiles.depth;
                converted.assign(n * 16, 0);
                for (size_t i = 0; i < n; i++) {
                    std::memcpy(converted.data() + i * 16, level_data.data() + i * 12, 12);
                    const float one = 1.0f;
                    std::memcpy(converted.data() + i * 16 + 12, &one, 4);
                }
                upload = converted.data();
                row_bytes = tiles.width * 16;
                break;
            }
            case Convert::Bc1:
            case Convert::Bc2:
            case Convert::Bc3:
            case Convert::Bc4:
            case Convert::Bc4Snorm:
            case Convert::Bc5:
            case Convert::Bc5Snorm: {
                upload = converted.data();
                row_bytes = level_size.width * 4;
                rows = level_size.height;
                break;
            }
            default:
                break;
            }
            if (banded) continue;
            // Component mapping is applied by the shader for every format (including render targets).
            WGPUTexelCopyTextureInfo dst{};
            dst.texture = im->tex;
            dst.mipLevel = level;
            dst.origin = {0, 0, info.dimension == WGPUTextureDimension_3D ? 0 : layer};
            dst.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyBufferLayout layout{};
            layout.bytesPerRow = row_bytes;
            layout.rowsPerImage = rows;
            // compressed formats are copied in whole blocks: the extent is the block-aligned size
            const bool compressed = !decode_any_compressed && cv == Convert::None && tile.width > 1;
            WGPUExtent3D extent{compressed ? tiles.width * tile.width : to_bc7 ? (level_size.width + 3) / 4 * 4 : level_size.width,
                                compressed ? tiles.height * tile.height : to_bc7 ? (level_size.height + 3) / 4 * 4 : level_size.height,
                                info.dimension == WGPUTextureDimension_3D ? level_size.depth : 1};
            const size_t upload_bytes = (size_t)row_bytes * rows * extent.depthOrArrayLayers;
            if (deferring) {
                DeferredUpload u;
                u.level = level;
                u.origin_z = dst.origin.z;
                u.layout = layout;
                u.extent = extent;
                // (the decoded level itself, not a copy: one big texture's copy ran the host heap out)
                std::vector<uint8_t>& from = upload == converted.data() ? converted : level_data;
                if (upload == from.data() && from.size() >= upload_bytes) {
                    u.data = std::move(from);
                    from = {};
                } else {
                    u.data.assign(upload, upload + upload_bytes);
                }
                stream_upload(std::move(u));
            } else {
                wgpuQueueWriteTexture(dev.queue, &dst, upload, upload_bytes, &layout, &extent);
            }
            if (!rx_side_thread) spike.upload_bytes += (int64_t)row_bytes * rows * extent.depthOrArrayLayers;
        }
        level_offset += CalculateLevelSize(li, level);
    }
    im->content_hash = sample_hash(va, total);
    im->guest_bytes = total;
    if (!rx_side_thread) {   // (a worker's texture gets these when it is installed)
        im->checked_frame = frame_number;
        im->decoded_seq = memory_seq();
    }
    return im;
}

// ---- textures decoded on a worker thread: a texture first drawn is decoded there (ASTC above all: the GPU has no
// ASTC, and new areas brought 40-60 ms frames of decoding) while its draws get the placeholder, as for programs and
// pipelines still being made; it is installed at the next present. A texture whose memory was written meanwhile is
// dropped (and decoded again when drawn). Used on the web only by default (decode_async). ----
struct DecodeJob {
    uint64_t key = 0, va = 0, seq = 0;
    TICEntry tic{};
    TexInfo info{};
    std::unique_ptr<Image> im;
    std::unique_ptr<DeferredTexture> deferred;   // (web)
};
std::deque<DecodeJob*> decode_queue;
std::vector<DecodeJob*> decode_done;
std::atomic<int> decodes_done{0};
std::unordered_map<uint64_t, DecodeJob*> decode_pending;   // by key (render thread)
bool decode_pending_result;   // the last texture_for_handle_uncached gave the placeholder for a texture on its way

const bool astc_workers_in_background = (Common::worker_thread_start = &background_thread, true);

void decode_worker() {
    rx_side_thread = true;
    background_thread();
    for (;;) {
        DecodeJob* j;
        {
            std::unique_lock lk(decode_mu);
            decode_cv.wait(lk, [] { return !decode_queue.empty() && deferred_bytes.load() < kDeferredBudget; });
            j = decode_queue.front();
            decode_queue.pop_front();
        }
        if (RX_WEB) {
            j->deferred = std::make_unique<DeferredTexture>();
            deferring = j->deferred.get();
        }
        j->im = decode(j->tic, j->info, j->va);
        deferring = nullptr;
        static const bool trace = env_flag("RX_TRACE_DECODE");
        if (trace)
            hos_log("rx: decoded %ux%u x%u levels %u format %d (streamed %lld MB, heap top %u MB)", j->info.size.width,
                    j->info.size.height, j->info.layers, j->info.levels, (int)j->info.format,
                    (long long)(deferred_bytes.load() >> 20), (unsigned)(heap_top() >> 20));
        std::lock_guard lk(decode_mu);
        decode_done.push_back(j);
        decodes_done++;
    }
}

// Natively textures decode in place by default: ASTC and big block-linear textures are made by the GPU now (a few ms
// at most), and a placeholder drawn into anything the game keeps (the loading screen's map, terrain tiles baked once:
// light boxes on the horizon for seconds after a load) stays wrong. RX_ASYNC_DECODE=1: the worker, as before (the web
// keeps it: its CPU decodes and heap are the constraint).
bool decode_async() {
    static const bool sync = RX_WEB ? env_flag("RX_SYNC_DECODE") : !env_flag("RX_ASYNC_DECODE");
    return !sync && !frame_dump_active();
}

void queue_decode(uint64_t key, const TICEntry& tic, const TexInfo& info, uint64_t va) {
    if (decode_pending.count(key)) return;
    auto* j = new DecodeJob();
    j->key = key;
    j->va = va;
    j->seq = memory_seq();
    j->tic = tic;
    j->info = info;
    decode_pending[key] = j;
    static bool started;
    std::lock_guard lk(decode_mu);
    if (!started) {
        started = true;
        // (one: ASTC fans out over its own thread pool; in the browser four workers each holding a whole decoded
        // texture array ran the host heap out)
        std::thread(decode_worker).detach();
    }
    decode_queue.push_back(j);
    decode_cv.notify_one();
}
}   // namespace

/// (the browser's memory log: decoded texture bytes waiting for the render thread, textures queued and decoded)
extern "C" void rx_texture_memory(int64_t* waiting, int64_t* queued, int64_t* pending) {
    *waiting = deferred_bytes.load();
    std::lock_guard lk(decode_mu);
    *queued = (int64_t)decode_queue.size();
    *pending = (int64_t)decode_done.size();
}

bool sampler_nonfiltering(WGPUSampler s) { return nonfiltering_samplers.count(s) != 0; }

void install_decoded_textures() {
    std::vector<DecodeJob*> done;
    std::deque<StreamedUpload> uploads;
    if (RX_WEB) {   // (one lock for both: a finished job's uploads all come with it)
        std::lock_guard lk(decode_mu);
        uploads.swap(upload_stream);
        done.swap(decode_done);
        decodes_done = 0;
    } else {
        if (!decodes_done.load(std::memory_order_relaxed)) return;
        std::lock_guard lk(decode_mu);
        done.swap(decode_done);
        decodes_done = 0;
    }
    if (!uploads.empty()) {
        int64_t n = 0;
        for (StreamedUpload& su : uploads) {
            const DeferredUpload& u = su.upload;
            WGPUTexelCopyTextureInfo dst{};
            dst.texture = deferred_texture(*su.texture);
            dst.mipLevel = u.level;
            dst.origin = {0, u.origin_y, u.origin_z};
            dst.aspect = WGPUTextureAspect_All;
            wgpuQueueWriteTexture(dev.queue, &dst, u.data.data(), u.data.size(), &u.layout, &u.extent);
            n += (int64_t)u.data.size();
        }
        uploads.clear();
        deferred_bytes -= n;
        decode_cv.notify_all();   // (a worker waiting for room)
    }
    for (DecodeJob* j : done) {
        decode_pending.erase(j->key);
        Image* im = j->im.get();
        const bool written = im && j->seq && memory_written_since(im->guest, std::max<uint64_t>(im->guest_bytes, 1), j->seq);
        if (written || decoded.count(j->key)) {
            count("tex_decode_dropped");
            // (rewritten while it decoded: a texture the CPU or a copy rewrites every frame or two, such as a
            // cutscene video's frames, would be dropped again and again and the draw would sample the gray
            // placeholder for the whole movie; from now on it decodes where it is drawn)
            if (written && rewritten_textures.insert(im->guest).second) {
                static int logged;
                if (logged++ < 20)
                    hos_log("rx: texture %ux%u format %d at %08llx was rewritten while it decoded: decoded where drawn "
                            "from now on", j->info.size.width, j->info.size.height, (int)j->info.format,
                            (unsigned long long)im->guest);
            }
        } else {
            if (im && j->deferred) {   // (web: the texture its layers were streamed into)
                im->tex = deferred_texture(*j->deferred);
                j->deferred->tex = nullptr;
            }
            if (im) {
                im->checked_frame = frame_number;
                im->decoded_seq = j->seq;
                remember(im);
                static int logged;
                if (logged++ < 40)
                    hos_log("rx: texture %ux%u x%u levels %u format %d at GPU %#llx (decoded on the worker)",
                            j->info.size.width, j->info.size.height, j->info.layers, j->info.levels,
                            (int)j->info.format, (unsigned long long)j->va);
            }
            decoded[j->key] = std::move(j->im);   // (null: known undecodable)
            decoded_generation++;
            count("tex_decoded_async");
        }
        if (j->deferred && j->deferred->tex) wgpuTextureRelease(j->deferred->tex);   // (dropped)
        delete j;   // (a dropped texture goes with it: no draw used it)
    }
}

namespace {

TexKind kind_of(WGPUTextureFormat f, PixelFormat pf) {
    if (is_depth_format(f)) return TexKind::Depth;
    if (VideoCore::Surface::IsPixelFormatInteger(pf))
        return VideoCore::Surface::IsPixelFormatSignedInteger(pf) ? TexKind::Sint : TexKind::Uint;
    const bool f32 = f == WGPUTextureFormat_R32Float || f == WGPUTextureFormat_RG32Float ||
                     f == WGPUTextureFormat_RGBA32Float;
    // (the browser: 16-bit normalized formats (texture-formats-tier1) are not filterable there; bound as filterable,
    // the bind group was invalid and its whole batch was dropped: half of Lookout Landing missing)
    const bool norm16 = RX_WEB && (f == WGPUTextureFormat_R16Unorm || f == WGPUTextureFormat_RG16Unorm ||
                                   f == WGPUTextureFormat_RGBA16Unorm || f == WGPUTextureFormat_R16Snorm ||
                                   f == WGPUTextureFormat_RG16Snorm || f == WGPUTextureFormat_RGBA16Snorm);
    return (f32 && !dev.float32_filterable) || norm16 ? TexKind::UnfilterableFloat : TexKind::Float;
}

WGPUAddressMode address_mode(Tegra::Texture::WrapMode m) {
    using W = Tegra::Texture::WrapMode;
    switch (m) {
    case W::Wrap: return WGPUAddressMode_Repeat;
    case W::Mirror: return WGPUAddressMode_MirrorRepeat;
    default: return WGPUAddressMode_ClampToEdge;
    }
}

WGPUCompareFunction compare_function(Tegra::Texture::DepthCompareFunc f) {
    using D = Tegra::Texture::DepthCompareFunc;
    switch (f) {
    case D::Never: return WGPUCompareFunction_Never;
    case D::Less: return WGPUCompareFunction_Less;
    case D::Equal: return WGPUCompareFunction_Equal;
    case D::LessEqual: return WGPUCompareFunction_LessEqual;
    case D::Greater: return WGPUCompareFunction_Greater;
    case D::NotEqual: return WGPUCompareFunction_NotEqual;
    case D::GreaterEqual: return WGPUCompareFunction_GreaterEqual;
    default: return WGPUCompareFunction_Always;
    }
}

/// mode: 0 filtering as the header says, 1 never filtering (unfilterable textures), 2 comparison
WGPUSampler sampler_for(const TSCEntry& tsc, int mode) {
    const uint64_t key = hash_bytes(&tsc, sizeof tsc, (uint64_t)mode);
    auto it = samplers.find(key);
    if (it != samplers.end()) {
        if (mode == 1 && !nonfiltering_samplers.count(it->second))
            log_limited("sampler_key", 6, "rx: sampler key %016llx (mode 1) found %p, not non-filtering (map %zu)",
                        (unsigned long long)key, (void*)it->second, samplers.size());
        return it->second;
    }
    using F = Tegra::Texture::TextureFilter;
    using M = Tegra::Texture::TextureMipmapFilter;
    WGPUSamplerDescriptor d{};
    d.addressModeU = address_mode(tsc.wrap_u);
    d.addressModeV = address_mode(tsc.wrap_v);
    d.addressModeW = address_mode(tsc.wrap_p);
    const bool filter = mode != 1;
    d.magFilter = filter && tsc.mag_filter == F::Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    d.minFilter = filter && tsc.min_filter == F::Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    d.mipmapFilter = filter && tsc.mipmap_filter == M::Linear ? WGPUMipmapFilterMode_Linear
                                                              : WGPUMipmapFilterMode_Nearest;
    d.lodMinClamp = std::max(0.0f, tsc.MinLod());
    d.lodMaxClamp = tsc.mipmap_filter == M::None ? d.lodMinClamp : std::max(d.lodMinClamp, tsc.MaxLod());
    d.compare = mode == 2 ? (tsc.depth_compare_enabled ? compare_function(tsc.depth_compare_func)
                                                       : WGPUCompareFunction_LessEqual)
                          : WGPUCompareFunction_Undefined;
    static const uint16_t aniso[8] = {1, 2, 4, 6, 8, 10, 12, 16};
    const bool all_linear = d.magFilter == WGPUFilterMode_Linear && d.minFilter == WGPUFilterMode_Linear &&
                            d.mipmapFilter == WGPUMipmapFilterMode_Linear;
    d.maxAnisotropy = all_linear ? aniso[tsc.max_anisotropy & 7] : 1;
    WGPUSampler s = wgpuDeviceCreateSampler(dev.device, &d);
    samplers[key] = s;
    if (mode == 1) nonfiltering_samplers.insert(s);
    return s;
}

struct Dummy {
    Image d2, d1, d3, depth, uint2, sint2;
};
Dummy* dummies;

void make_dummy(Image& im, WGPUTextureDimension dim, WGPUTextureFormat f, uint32_t layers, const void* texel,
                uint32_t texel_bytes) {
    im.format = f;
    im.dimension = dim;
    im.width = im.height = 1;
    im.layers = layers;
    WGPUTextureDescriptor d{};
    d.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst |
              (is_depth_format(f) ? WGPUTextureUsage_RenderAttachment : 0);
    d.dimension = dim;
    d.size = {1, 1, layers};
    d.format = f;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    im.tex = wgpuDeviceCreateTexture(dev.device, &d);
    if (is_depth_format(f)) {   // depth textures cannot be written: cleared by a render pass
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
        for (uint32_t l = 0; l < layers; l++) {
            WGPUTextureViewDescriptor vd{};
            vd.format = f;
            vd.dimension = WGPUTextureViewDimension_2D;
            vd.mipLevelCount = 1;
            vd.baseArrayLayer = l;
            vd.arrayLayerCount = 1;
            vd.aspect = WGPUTextureAspect_All;
            WGPUTextureView v = wgpuTextureCreateView(im.tex, &vd);
            WGPURenderPassDepthStencilAttachment da{};
            da.view = v;
            da.depthLoadOp = WGPULoadOp_Clear;
            da.depthStoreOp = WGPUStoreOp_Store;
            da.depthClearValue = 1.0f;
            WGPURenderPassDescriptor pd{};
            pd.depthStencilAttachment = &da;
            WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(e, &pd);
            wgpuRenderPassEncoderEnd(p);
            wgpuRenderPassEncoderRelease(p);
            wgpuTextureViewRelease(v);
        }
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(dev.queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(e);
        return;
    }
    for (uint32_t l = 0; l < layers; l++) {
        WGPUTexelCopyTextureInfo dst{};
        dst.texture = im.tex;
        dst.origin = {0, 0, l};
        dst.aspect = WGPUTextureAspect_All;
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = texel_bytes;
        layout.rowsPerImage = 1;
        WGPUExtent3D ext{1, 1, 1};
        wgpuQueueWriteTexture(dev.queue, &dst, texel, texel_bytes, &layout, &ext);
    }
}

Dummy& dummy() {
    if (!dummies) {
        dummies = new Dummy();
        const uint8_t gray[4] = {128, 128, 128, 255};
        const uint32_t zero[4] = {0, 0, 0, 0};
        make_dummy(dummies->d2, WGPUTextureDimension_2D, WGPUTextureFormat_RGBA8Unorm, 6, gray, 4);
        make_dummy(dummies->d1, WGPUTextureDimension_1D, WGPUTextureFormat_RGBA8Unorm, 1, gray, 4);
        make_dummy(dummies->d3, WGPUTextureDimension_3D, WGPUTextureFormat_RGBA8Unorm, 1, gray, 4);
        make_dummy(dummies->depth, WGPUTextureDimension_2D, WGPUTextureFormat_Depth32Float, 6, nullptr, 0);
        make_dummy(dummies->uint2, WGPUTextureDimension_2D, WGPUTextureFormat_RGBA32Uint, 6, zero, 16);
        make_dummy(dummies->sint2, WGPUTextureDimension_2D, WGPUTextureFormat_RGBA32Sint, 6, zero, 16);
    }
    return *dummies;
}

TSCEntry read_tsc(const Regs& r, uint32_t index) {
    TSCEntry e{};
    if (index <= r.tex_sampler.limit) gpu_read(r.tex_sampler.Address() + (uint64_t)index * sizeof e, &e, sizeof e);
    return e;
}

bool layers_fit(WGPUTextureViewDimension dim, uint32_t layers) {
    switch (dim) {
    case WGPUTextureViewDimension_Cube: return layers >= 6;
    case WGPUTextureViewDimension_CubeArray: return layers >= 6 && layers % 6 == 0;
    default: return true;
    }
}

/// A layered or mipmapped texture whose layers/levels the GPU rendered one at a time (array layers, cube faces,
/// mip chains drawn as separate render targets at their places in the texture's memory): those subresources come
/// from the render targets (copied when they were drawn again since), the rest from guest memory.
/// A 3D texture decoded from memory takes the slices drawn as render targets (nvn_target files them under
/// nvn_slice_key): each slice whose image was drawn since the last copy is copied into its depth slice (level 0).
void assemble_3d_slices(Image& im, const TexInfo& info, uint64_t guest) {
    if (!guest || im.dimension != WGPUTextureDimension_3D || im.assembled_passes == pass_count()) return;
    im.assembled_passes = pass_count();
    const uint32_t depth = std::min<uint32_t>(info.size.depth, im.layers);
    if (im.sub_versions.size() != depth) im.sub_versions.assign(depth, 0);
    bool ended = false;
    for (uint32_t z = 0; z < depth; z++) {
        Image* rt = image_at(nvn_slice_key(guest, z));
        if (!rt || !rt->rendered || rt->width != im.width || rt->height != im.height || rt->format != im.format) continue;
        if (im.sub_versions[z] == rt->version) continue;
        im.sub_versions[z] = rt->version;
        if (!ended) end_render_pass("pe_assemble"), ended = true;
        WGPUTexelCopyTextureInfo src{}, dst{};
        src.texture = rt->tex;
        dst.texture = im.tex;
        dst.origin = {0, 0, z};
        const WGPUExtent3D one{im.width, im.height, 1};
        ce_copy_texture_to_texture(encoder(), &src, &dst, &one);
        count("assembled_3d_slices");
    }
}

void assemble_from_render_targets(Image& im, const TexInfo& info, uint64_t guest) {
    static const bool always = env_flag("RX_ASSEMBLE_ALWAYS");   // (debugging: every look copies every level again)
    if (!guest || (im.assembled_passes == pass_count() && !always)) return;   // no render target drawn since the last look
    im.assembled_passes = pass_count();
    const uint32_t layers = std::min(info.layers, im.layers), levels = std::min(info.levels, im.levels);
    if (im.sub_versions.size() != (size_t)layers * levels) {
        im.sub_versions.assign((size_t)layers * levels, 0);
        im.sub_addresses.resize((size_t)layers * levels);
        im.sub_end = 0;
        for (uint32_t layer = 0; layer < layers; layer++)
            for (uint32_t level = 0; level < levels; level++) {
                const uint64_t a = guest + subresource_offset(info, layer, level);
                im.sub_addresses[(size_t)layer * levels + level] = a;
                im.sub_end = std::max(im.sub_end, a + 1);
            }
        im.sub_images.clear();
    }
    // image_at per subresource, again only when an image appeared or went away inside this texture's memory (render
    // targets change tens of times a frame elsewhere; a 121-layer, 11-level array is 1331 lookups)
    static const bool no_subcache = env_flag("RX_NO_SUBCACHE");
    if (im.sub_images_generation != images_generation() || no_subcache || im.sub_images.empty()) {
        if (im.sub_images.empty() || no_subcache || images_changed_in(im.sub_images_generation, guest, im.sub_end)) {
            im.sub_images.resize(im.sub_addresses.size());
            im.sub_present.clear();
            for (size_t i = 0; i < im.sub_addresses.size(); i++)
                if ((im.sub_images[i] = image_at(im.sub_addresses[i]))) im.sub_present.push_back((uint32_t)i);
        }
        im.sub_images_generation = images_generation();
    }
    bool ended = false;
    // (RX_DUMP_ASSEMBLE=1: frame dumps log the missing ones too, every time: a 121-layer array is 1331 lines)
    static const bool dump_missing = env_flag("RX_DUMP_ASSEMBLE");
    const bool all = dump_missing && frame_dump_active();
    const size_t n = all ? im.sub_images.size() : im.sub_present.size();
    for (size_t k = 0; k < n; k++) {
            const uint32_t i = all ? (uint32_t)k : im.sub_present[k];
            const uint32_t layer = i / levels, level = i % levels;
            Image* rt = im.sub_images[i];
            const uint32_t w = std::max(1u, im.width >> level), h = std::max(1u, im.height >> level);
            if (all && (!rt || !rt->rendered || rt->width != w || rt->height != h || rt->format != im.format)) {
                char b[256];
                std::snprintf(b, sizeof b, "assemble %08llx layer %u level %u at %08llx: %s %ux%u fmt %x (want %ux%u fmt %x)",
                              (unsigned long long)im.guest, layer, level,
                              (unsigned long long)im.sub_addresses[(size_t)layer * levels + level],
                              rt ? (rt->rendered ? "rt" : "unrendered image") : "no image", rt ? rt->width : 0,
                              rt ? rt->height : 0, rt ? (unsigned)rt->format : 0, w, h, (unsigned)im.format);
                dump_draw(b);
            }
            if (!rt) continue;
            if (!rt || !rt->rendered || rt->width != w || rt->height != h || rt->format != im.format) continue;
            uint64_t& seen = im.sub_versions[(size_t)layer * levels + level];
            if (seen == rt->version && !always) continue;
            static const uint64_t freeze = getenv("RX_ASSEMBLE_FREEZE") ? strtoull(getenv("RX_ASSEMBLE_FREEZE"), nullptr, 16) : 0;
            if (freeze && im.guest == freeze && seen) continue;   // (debugging: copied once, then never again)
            seen = rt->version;
            if (!ended) {   // copies cannot be in a render pass
                end_render_pass("pe_assemble");
                ended = true;
            }
            WGPUTexelCopyTextureInfo src{}, dst{};
            src.texture = rt->tex;
            src.aspect = WGPUTextureAspect_All;
            dst.texture = im.tex;
            dst.mipLevel = level;
            dst.origin = {0, 0, layer};
            dst.aspect = WGPUTextureAspect_All;
            const WGPUExtent3D extent{w, h, 1};
            ce_copy_texture_to_texture(encoder(), &src, &dst, &extent);
            count("assembled_copies");
            if (frame_dump_active()) {
                char b[256];
                std::snprintf(b, sizeof b, "assemble %08llx %ux%u x%u levels %u fmt %x: layer %u level %u <- rt %08llx",
                              (unsigned long long)im.guest, im.width, im.height, im.layers, im.levels,
                              (unsigned)im.format, layer, level, (unsigned long long)rt->guest);
                dump_draw(b);
            }
        }
}

} // namespace

uint64_t image_bytes(const Image& im) {
    double per_texel;
    switch (im.format) {
    case WGPUTextureFormat_BC1RGBAUnorm: case WGPUTextureFormat_BC1RGBAUnormSrgb:
    case WGPUTextureFormat_BC4RUnorm: case WGPUTextureFormat_BC4RSnorm:
        per_texel = 0.5;
        break;
    case WGPUTextureFormat_BC2RGBAUnorm: case WGPUTextureFormat_BC2RGBAUnormSrgb:
    case WGPUTextureFormat_BC3RGBAUnorm: case WGPUTextureFormat_BC3RGBAUnormSrgb:
    case WGPUTextureFormat_BC5RGUnorm: case WGPUTextureFormat_BC5RGSnorm:
    case WGPUTextureFormat_BC6HRGBUfloat: case WGPUTextureFormat_BC6HRGBFloat:
    case WGPUTextureFormat_BC7RGBAUnorm: case WGPUTextureFormat_BC7RGBAUnormSrgb:
    case WGPUTextureFormat_R8Unorm: case WGPUTextureFormat_R8Snorm:
        per_texel = 1;
        break;
    case WGPUTextureFormat_RG8Unorm: case WGPUTextureFormat_RG8Snorm: case WGPUTextureFormat_R16Float:
    case WGPUTextureFormat_R16Unorm: case WGPUTextureFormat_Depth16Unorm:
        per_texel = 2;
        break;
    case WGPUTextureFormat_RGBA16Float: case WGPUTextureFormat_RG32Float: case WGPUTextureFormat_RGBA16Unorm:
        per_texel = 8;
        break;
    case WGPUTextureFormat_RGBA32Float:
        per_texel = 16;
        break;
    default:
        per_texel = 4;
        break;
    }
    const double texels = (double)im.width * im.height * im.layers * (im.levels > 1 ? 4.0 / 3.0 : 1.0);
    return (uint64_t)(texels * per_texel);
}

void decoded_texture_stats(size_t* n, uint64_t* bytes) {
    *n = 0;
    *bytes = 0;
    for (const auto& [key, im] : decoded)
        if (im) {
            (*n)++;
            *bytes += image_bytes(*im);
        }
}

void trim_decoded_textures() {
    static const uint64_t budget = [] {
        const char* e = getenv("RX_TEXTURE_BUDGET_MB");
        return (uint64_t)(e && atoi(e) > 0 ? atoi(e) : RX_WEB ? 640 : 2048) << 20;
    }();
    if (frame_number % 600 != 0) return;
    uint64_t total = 0;
    int dropped = 0;
    std::vector<std::pair<int64_t, uint64_t>> old;
    for (auto it = decoded.begin(); it != decoded.end();) {
        Image* im = it->second.get();
        if (!im) {
            ++it;
            continue;
        }
        if (im->stale && im->decoded_seq && im->last_use < frame_number - 60) {
            forget(im);
            it = decoded.erase(it);
            dropped++;
            continue;
        }
        const uint64_t bytes = image_bytes(*im);
        total += bytes;
        if (im->last_use < frame_number - 900) old.emplace_back(im->last_use, it->first);
        ++it;
    }
    // RX_LOG_DECODED_TOP=1 (debugging): the largest decoded textures, when over budget (once)
    if (static const bool top = env_flag("RX_LOG_DECODED_TOP"); top && total > budget) {
        static bool done;
        if (!done) {
            done = true;
            std::vector<std::pair<uint64_t, const Image*>> v;
            for (const auto& [k, p] : decoded)
                if (p) v.push_back({image_bytes(*p), p.get()});
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t i = 0; i < v.size() && i < 24; i++)
                hos_log("rx: decoded %3zu: %llu MB %ux%u x%u levels %u format %#x at %llx, last use %lld", i,
                        (unsigned long long)(v[i].first >> 20), v[i].second->width, v[i].second->height,
                        v[i].second->layers, v[i].second->levels, (unsigned)v[i].second->format,
                        (unsigned long long)v[i].second->guest, (long long)v[i].second->last_use);
        }
    }
    uint64_t freed = 0;
    if (total > budget && !old.empty()) {
        std::sort(old.begin(), old.end());
        for (const auto& [use, key] : old) {
            if (total - freed <= budget - budget / 8) break;
            auto it = decoded.find(key);
            if (it == decoded.end() || !it->second) continue;
            freed += image_bytes(*it->second);
            forget(it->second.get());
            decoded.erase(it);
            dropped++;
        }
    }
    if (dropped) {
        decoded_generation++;
        count("tex_trimmed", dropped);
        static int64_t logged_at = -100000;
        if (frame_number - logged_at > 3600) {
            logged_at = frame_number;
            hos_log("rx: decoded textures %llu MB (budget %llu MB): %d dropped (rewritten or unused), %llu MB freed",
                    (unsigned long long)((total - freed) >> 20), (unsigned long long)(budget >> 20), dropped,
                    (unsigned long long)(freed >> 20));
        }
    }
}

void textures_written(uint64_t guest, uint64_t size) {
    if (decoded_by_bucket.empty() || !size) return;
    for (uint64_t b = guest >> kBucketShift; b <= (guest + size - 1) >> kBucketShift; b++) {
        auto it = decoded_by_bucket.find(b);
        if (it == decoded_by_bucket.end()) continue;
        for (Image* im : it->second)
            if (im->guest < guest + size && im->guest + im->guest_bytes > guest) im->stale = true;
    }
}

WGPUTextureViewDimension view_dimension(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D: return WGPUTextureViewDimension_1D;
    case Shader::TextureType::ColorArray2D: return WGPUTextureViewDimension_2DArray;
    case Shader::TextureType::Color3D: return WGPUTextureViewDimension_3D;
    case Shader::TextureType::ColorCube: return WGPUTextureViewDimension_Cube;
    case Shader::TextureType::ColorArrayCube: return WGPUTextureViewDimension_CubeArray;
    default: return WGPUTextureViewDimension_2D;
    }
}

WGPUTextureView Image::view(WGPUTextureViewDimension dim, bool depth_aspect, uint32_t base_layer,
                                 uint32_t base_level, uint32_t level_count, WGPUTextureFormat view_format) {
    if (view_format != srgb_pair(format)) view_format = format;   // (only the other encoding is a view format)
    base_layer = std::min(base_layer, layers - 1);
    base_level = std::min(base_level, levels - 1);
    level_count = level_count ? std::min(level_count, levels - base_level) : levels - base_level;
    const uint64_t key = (uint64_t)dim | (depth_aspect ? 0x100ull : 0) | (view_format != format ? 0x200ull : 0) |
                         ((uint64_t)base_layer << 16) | ((uint64_t)base_level << 40) | ((uint64_t)level_count << 48);
    for (auto& [k, v] : views)
        if (k == key) return v;
    WGPUTextureViewDescriptor d{};
    d.format = view_format;
    d.dimension = dim;
    d.baseMipLevel = base_level;
    d.mipLevelCount = level_count;
    d.baseArrayLayer = base_layer;
    switch (dim) {
    case WGPUTextureViewDimension_1D:
    case WGPUTextureViewDimension_2D:
    case WGPUTextureViewDimension_3D: d.arrayLayerCount = 1; break;
    case WGPUTextureViewDimension_Cube: d.arrayLayerCount = 6; break;
    default: d.arrayLayerCount = dimension == WGPUTextureDimension_3D ? 1 : layers - base_layer; break;
    }
    d.aspect = depth_aspect ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
    if (storage) d.usage = WGPUTextureUsage_TextureBinding;   // (an sRGB view cannot have the storage usage)
    WGPUTextureView v = wgpuTextureCreateView(tex, &d);
    views.emplace_back(key, v);
    return v;
}

void texture_slots(uint8_t* dst, const TICEntry& tic, const TSCEntry& tsc) {
    using W = Tegra::Texture::WrapMode;
    auto border = [](W w) { return w == W::Border || w == W::MirrorOnceBorder; };
    static const bool no_border = env_flag("RX_NO_BORDER");   // (debugging: border wrap as clamp)
    const uint32_t axes = no_border ? 0u
                                    : (border(tsc.wrap_u) ? 1u : 0u) | (border(tsc.wrap_v) ? 2u : 0u) |
                                          (border(tsc.wrap_p) ? 4u : 0u);
    const uint32_t compare = tsc.depth_compare_enabled ? (uint32_t)tsc.depth_compare_func.Value() & 7u : 3u;
    const int32_t bias = (int32_t)std::lround(tsc.LodBias() * 256.0f);
    const uint32_t slot[4] = {(uint32_t)tic.x_source.Value() | axes << 3, (uint32_t)tic.y_source.Value() | compare << 3,
                              (uint32_t)tic.z_source.Value() | (uint32_t)bias << 8, (uint32_t)tic.w_source.Value()};
    std::memcpy(dst, slot, 16);
    std::memcpy(dst + 16, tsc.border_color.data(), 16);
}

Tegra::Texture::TSCEntry read_tsc_for_dump(const Regs& r, uint32_t raw_handle) {
    const auto [tic_index, tsc_index] =
        Tegra::Texture::TexturePair(raw_handle, r.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding);
    (void)tic_index;
    return read_tsc(r, tsc_index);
}

Tegra::Texture::TICEntry read_tic(const Regs& r, uint32_t index) {
    TICEntry e{};
    if (index <= r.tex_header.limit) gpu_read(r.tex_header.Address() + (uint64_t)index * sizeof e, &e, sizeof e);
    return e;
}

BoundTexture dummy_texture(const Shader::TextureDescriptor& desc) {
    Dummy& d = dummy();
    const WGPUTextureViewDimension dim = view_dimension(desc.type);
    BoundTexture b;
    TSCEntry tsc{};
    if (desc.is_depth) {
        b.kind = TexKind::Depth;
        b.view = d.depth.view(dim == WGPUTextureViewDimension_1D || dim == WGPUTextureViewDimension_3D
                                  ? WGPUTextureViewDimension_2D
                                  : dim,
                              false);
        b.sampler = sampler_for(tsc, 2);
        return b;
    }
    Image& im = dim == WGPUTextureViewDimension_1D ? d.d1 : dim == WGPUTextureViewDimension_3D ? d.d3 : d.d2;
    b.view = im.view(dim, false);
    b.kind = TexKind::Float;
    b.sampler = sampler_for(tsc, 0);
    return b;
}

namespace {

Image* last_bound_image;       // the image texture_for_handle_uncached bound (null: a placeholder)
bool last_assembled;           // ...and it is assembled from render targets (assemble_from_render_targets)
BoundTexture texture_for_handle_uncached(const Regs& r, uint32_t raw, const Shader::TextureDescriptor& desc);
} // namespace

/// Draws bind the same textures again and again, frame after frame: the same header pair gives the same binding
/// while the image at the texture's address is the same one (and did not become rendered), no texture was decoded
/// or dropped, no GPU mapping changed and the texture's memory was not written (a decoded texture's stale flag).
/// A texture assembled from render targets still copies what was drawn since; a decoded texture without write
/// tracking still checks its memory once a frame (RX_NO_TEX_MEMO=1: always the full path; frame dumps always take
/// it).
BoundTexture texture_for_handle(const Regs& r, uint32_t raw, const Shader::TextureDescriptor& desc) {
    static const bool no_memo = env_flag("RX_NO_TEX_MEMO");
    if (no_memo || frame_dump_active()) {
        const auto [tic_index, tsc_index] =
            Tegra::Texture::TexturePair(raw, r.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding);
        BoundTexture b = texture_for_handle_uncached(r, raw, desc);
        texture_slots(b.slots, read_tic(r, tic_index), read_tsc(r, tsc_index));
        return b;
    }
    const bool via_header = r.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding;
    const auto [tic_index, tsc_index] = Tegra::Texture::TexturePair(raw, via_header);
    TICEntry tic{};
    TSCEntry tsc{};
    bool have_headers = false;
    auto load_headers = [&] {
        if (have_headers) return;
        tic = read_tic(r, tic_index);
        tsc = read_tsc(r, tsc_index);
        have_headers = true;
    };
    // The memo key is a hash of both headers: reading and hashing them for each of ~20k bindings a frame was itself
    // ~2 ms. A front cache knows a handle's key while the headers' memory was not written since (page tracking) and
    // the pools and mappings are the same.
    struct Front {
        uint64_t fkey = 0, key = 0, seq = 0, tic_guest = 0, tsc_guest = 0;
        long maps = 0;
    };
    static Front front[8192];
    const uint64_t pools[4] = {r.tex_header.Address() | (uint64_t)r.tex_header.limit << 44,
                               r.tex_sampler.Address() | (uint64_t)r.tex_sampler.limit << 44,
                               (uint64_t)raw << 32 | (uint64_t)desc.type << 8 | (desc.is_depth ? 2u : 0u) |
                                   (via_header ? 1u : 0u),
                               0};
    const uint64_t fkey = hash_bytes(pools, sizeof pools) | 1;
    Front& f = front[fkey & 8191];
    uint64_t key;
    if (f.fkey == fkey && f.maps == maps_generation && !memory_written_since(f.tic_guest, sizeof(TICEntry), f.seq) &&
        !memory_written_since(f.tsc_guest, sizeof(TSCEntry), f.seq)) {
        key = f.key;
    } else {
        const uint64_t seq = memory_seq();
        load_headers();
        key = hash_bytes(&tic, sizeof tic, hash_bytes(&tsc, sizeof tsc, (uint64_t)desc.type << 8 | (desc.is_depth ? 1u : 0u)));
        const uint64_t tic_guest = tic_index <= r.tex_header.limit
                                       ? hos_gpu_translate(r.tex_header.Address() + (uint64_t)tic_index * sizeof(TICEntry)) : 0;
        const uint64_t tsc_guest = tsc_index <= r.tex_sampler.limit
                                       ? hos_gpu_translate(r.tex_sampler.Address() + (uint64_t)tsc_index * sizeof(TSCEntry)) : 0;
        if (tic_guest && tsc_guest) f = {fkey, key, seq, tic_guest, tsc_guest, maps_generation};
        else f.fkey = 0;
    }
    struct Memo {
        uint64_t key = 0, frame = ~0ull, decoded = 0, guest = 0;
        long maps = 0;             // the GPU mappings' generation (guest = the texture's VA translated then)
        Image* src = nullptr;      // a decoded texture (its stale flag), or null
        uint64_t src_id = 0, tic_key = 0;   // ...its id, and its key in the decoded map
        uint64_t at_id = 0;        // image_at(guest) then (its id), and whether it was rendered
        bool at_rendered = false;
        Image* assembled = nullptr;   // a texture assembled from render targets, and its header's info
        TexInfo info;
        BoundTexture b;
    };
    static Memo memo[8192];
    Memo& m = memo[key & 8191];
    if (m.key == key && m.decoded != decoded_generation && m.maps == maps_generation) {
        // a texture was decoded or dropped since: that touches this binding only if it was its own (one texture the
        // CPU rewrites every frame used to drop every memo entry every frame)
        bool same = true;
        if (m.src_id) {
            auto it = decoded.find(m.tic_key);
            same = it != decoded.end() && it->second && it->second->id == m.src_id;
            if (same) m.src = it->second.get();
        }
        if (same) m.decoded = decoded_generation;
    }
    if (m.key == key && m.decoded == decoded_generation && m.maps == maps_generation) {
        Image* at = m.guest ? image_at(m.guest) : nullptr;
        // a later frame: what the uncached path does once a frame (a decoded texture without write tracking
        // checks its memory's samples; images in use are stamped so they are not dropped)
        const bool new_frame = m.frame != (uint64_t)frame_number;
        if ((at ? at->id : 0) == m.at_id && (!at || at->rendered == m.at_rendered) && (!m.src || !m.src->stale) &&
            !(new_frame && m.src && !m.src->decoded_seq)) {
            if (new_frame) {
                m.frame = (uint64_t)frame_number;
                if (m.src) m.src->last_use = frame_number;
                if (m.b.image) m.b.image->last_use = frame_number;
            }
            if (m.assembled) assemble_from_render_targets(*m.assembled, m.info, m.guest);
            count("tex_memo_hits");
            if (RX_WEB && m.b.kind == TexKind::UnfilterableFloat && !nonfiltering_samplers.count(m.b.sampler)) {
                load_headers();   // (the browser validates the pair: see below)
                m.b.sampler = sampler_for(tsc, 1);
                log_limited("memo_sampler", 4, "rx: memo held a filtering sampler for an unfilterable texture");
            }
            return m.b;   // (its slots too: the key holds both headers)
        }
    }
    load_headers();
    count(m.key != key ? "tex_memo_miss_key" : m.decoded != decoded_generation ? "tex_memo_miss_decoded"
          : m.maps != maps_generation ? "tex_memo_miss_maps" : "tex_memo_miss_image");
    last_bound_image = nullptr;
    last_assembled = false;
    decode_pending_result = false;
    BoundTexture b = texture_for_handle_uncached(r, raw, desc);
    texture_slots(b.slots, tic, tsc);
    // (the browser rejects a filtering sampler with an unfilterable texture, and the batch with it)
    if (RX_WEB && b.kind == TexKind::UnfilterableFloat && !nonfiltering_samplers.count(b.sampler)) {
        const WGPUSampler before = b.sampler;
        b.sampler = sampler_for(tsc, 1);
        log_limited("uncached_sampler", 4, "rx: unfilterable texture %ux%u format %d came with a filtering sampler %p "
                    "-> %p (in set %d)", b.width, b.height, b.format, (void*)before, (void*)b.sampler,
                    (int)nonfiltering_samplers.count(b.sampler));
    }
    if (decode_pending_result) {   // (the placeholder is not remembered: the texture comes in a frame or two)
        m.key = 0;
        return b;
    }
    const uint64_t va = (uint64_t)tic.address_high << 32 | tic.address_low;
    m.key = key;
    m.frame = (uint64_t)frame_number;
    m.decoded = decoded_generation;
    m.maps = maps_generation;
    m.guest = va ? hos_gpu_translate(va) : 0;
    Image* at = m.guest ? image_at(m.guest) : nullptr;
    m.at_id = at ? at->id : 0;
    m.at_rendered = at && at->rendered;
    m.src = last_bound_image && !last_bound_image->rendered ? last_bound_image : nullptr;
    m.src_id = m.src ? m.src->id : 0;
    m.tic_key = hash_bytes(&tic, sizeof tic);   // (as the decoded map keys it)
    m.assembled = last_assembled ? last_bound_image : nullptr;
    if (m.assembled) m.info = tex_info(tic);
    m.b = b;
    return b;
}

namespace {
BoundTexture texture_for_handle_uncached(const Regs& r, uint32_t raw, const Shader::TextureDescriptor& desc) {
    const bool via_header = r.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding;
    const auto [tic_index, tsc_index] = Tegra::Texture::TexturePair(raw, via_header);
    const TICEntry tic = read_tic(r, tic_index);
    const TSCEntry tsc = read_tsc(r, tsc_index);
    const uint64_t va = (uint64_t)tic.address_high << 32 | tic.address_low;
    static int texture_trace_count;
    static const bool trace_textures = env_flag("RX_TRACE_TEXTURES");
    if (trace_textures && texture_trace_count++ < 24)
        hos_log("rx: texture binding raw %#x TIC %u TSC %u VA %#llx TIC[%016llx %016llx %016llx %016llx]",
                raw, tic_index, tsc_index, (unsigned long long)va, (unsigned long long)tic.raw[0],
                (unsigned long long)tic.raw[1], (unsigned long long)tic.raw[2],
                (unsigned long long)tic.raw[3]);
    if (!va) return dummy_texture(desc);
    const TexInfo info = tex_info(tic);
    if (trace_textures && texture_trace_count <= 24)
        hos_log("rx: texture info valid %u %ux%u x%u levels %u format %d", (unsigned)info.valid, info.size.width,
                info.size.height, info.layers, info.levels, (int)info.format);
    if (!info.valid) return dummy_texture(desc);
    const WGPUTextureViewDimension dim = view_dimension(desc.type);

    Image* im = nullptr;
    // Padded attachments can have a larger backing extent than the TIC's logical sampling extent.
    const uint64_t guest = hos_gpu_translate(va);
    {   // RX_TIC_LOG=<guest hex>: the header of the texture there, once (debugging)
        static const uint64_t tic_log = getenv("RX_TIC_LOG") ? strtoull(getenv("RX_TIC_LOG"), nullptr, 16) : 0;
        static int logged;
        if (tic_log && guest == tic_log && logged++ < 2)
            hos_log("rx: TIC at %#llx: %016llx %016llx %016llx %016llx: %ux%ux%u layers %u levels %u block %u,%u,%u "
                    "dim %d linear %d format %d view %d", (unsigned long long)guest, (unsigned long long)tic.raw[0],
                    (unsigned long long)tic.raw[1], (unsigned long long)tic.raw[2], (unsigned long long)tic.raw[3],
                    info.size.width, info.size.height, info.size.depth, info.layers, info.levels, info.block.width,
                    info.block.height, info.block.depth, (int)info.dimension, (int)info.linear, (int)info.format,
                    (int)dim);
    }
    if (Image* rt = guest ? image_at(guest) : nullptr;
        rt && rt->rendered && rt->width >= info.size.width && rt->height >= info.size.height &&
        rt->dimension == info.dimension && rt->layers >= info.layers && rt->levels >= info.levels &&
        layers_fit(dim, rt->layers) &&
        ((dim == WGPUTextureViewDimension_3D) == (rt->dimension == WGPUTextureDimension_3D)) &&
        ((dim == WGPUTextureViewDimension_1D) == (rt->dimension == WGPUTextureDimension_1D)))
        im = rt;   // (a mipmapped one whose levels were drawn one by one, a bloom pyramid: assembled below; sampled as
                   // its first level alone, its blurred levels were the unblurred one: particles as squares of glow)
    if (!im && frame_dump_active())
        if (Image* rt = guest ? image_at(guest) : nullptr) {
            char b[256];
            std::snprintf(b, sizeof b,
                          "render target %08llx %ux%u x%u dim %d rendered %d not taken for a TIC %ux%u x%u dim %d "
                          "(view %d, format %d)",
                          (unsigned long long)guest, rt->width, rt->height, rt->layers, (int)rt->dimension,
                          (int)rt->rendered, info.size.width, info.size.height, info.layers, (int)info.dimension,
                          (int)dim, (int)info.format);
            dump_draw(b);
        }
    if (!im) {
        const uint64_t key = hash_bytes(&tic, sizeof tic);
        auto it = decoded.find(key);
        if (it != decoded.end()) {
            if (!it->second) return dummy_texture(desc);   // known undecodable
            im = it->second.get();
            static const uint64_t redecode = getenv("RX_REDECODE") ? strtoull(getenv("RX_REDECODE"), nullptr, 16) : 0;
            if ((redecode == va || redecode == im->guest) && im->checked_frame != frame_number) im->stale = true;   // (debugging)
            if (im->decoded_seq) {   // its pages were written since (textures_written)
                if (im->stale) {
                    forget(im);
                    decoded.erase(it);
                    decoded_generation++;
                    im = nullptr;
                }
            } else if (im->checked_frame != frame_number) {   // first use this frame: did the samples change?
                im->checked_frame = frame_number;
                if (sample_hash(va, guest_size(info)) != im->content_hash) {
                    forget(im);
                    decoded.erase(it);
                    decoded_generation++;
                    im = nullptr;
                }
            }
        }
        // (a one-off draw decodes in place: a placeholder would stay in what it renders, a map made once; not big
        // textures: whole in memory at once, where the background decode streams them, and the web's heap is small)
        static const bool linear_async = env_flag("RX_ASYNC_LINEAR");
        const bool cheap = !linear_async && (info.linear || rewritten_textures.count(guest)) &&
                           guest_size(info) <= (RX_WEB ? (4u << 20) : (16u << 20));
        if (!im && decode_async() && !cheap &&
            !(draw_must_run && guest_size(info) <= (RX_WEB ? (4u << 20) : (64u << 20)))) {
            queue_decode(key, tic, info, va);
            decode_pending_result = true;
            count("tex_decoding");
            return dummy_texture(desc);
        }
        if (!im) {
            const int64_t t0 = hos_now_ns();
            auto fresh = decode(tic, info, va);
            count("t_decode", hos_now_ns() - t0);
            spike.decode_ns += hos_now_ns() - t0;
            if (static const bool log_inplace = env_flag("RX_LOG_INPLACE"); log_inplace && hos_now_ns() - t0 > 5000000)
                hos_log("rx: decoded in place (%s) %ux%u x%u levels %u format %d linear %d: %.1f ms, %llu KB guest",
                        draw_must_run ? "one-off draw" : decode_async() ? "cheap" : "sync", info.size.width,
                        info.size.height, info.layers, info.levels, (int)info.format, (int)info.linear,
                        (hos_now_ns() - t0) / 1e6, (unsigned long long)(guest_size(info) >> 10));
            im = fresh.get();
            if (im) remember(im);
            decoded[key] = std::move(fresh);
            decoded_generation++;
            if (!im) return dummy_texture(desc);
            static int logged;
            if (logged++ < 40)
                hos_log("rx: texture %ux%u x%u levels %u format %d at GPU %#llx", info.size.width, info.size.height,
                        info.layers, info.levels, (int)info.format, (unsigned long long)va);
        }
    }
    im->last_use = frame_number;
    last_bound_image = im;
    if (!im->rendered && !info.linear && info.dimension != WGPUTextureDimension_3D && (info.layers > 1 || info.levels > 1)) {
        assemble_from_render_targets(*im, info, guest);
        last_assembled = guest != 0;
    }
    const uint32_t base_layer = im->dimension == WGPUTextureDimension_3D ? 0 : tic.BaseLayer();
    const uint32_t base_level = tic.res_min_mip_level;
    const uint32_t level_count = (uint32_t)tic.res_max_mip_level >= base_level
                                    ? (uint32_t)tic.res_max_mip_level - base_level + 1 : 1;
    // the encoding the header asks for (a render target drawn as sRGB can be read raw, and the other way round)
    const WGPUTextureFormat wanted = wgpu_format(info.format);
    if (im->rendered && wanted != im->format && wanted != srgb_pair(im->format) && !is_depth_format(im->format)) {
        static std::unordered_map<uint64_t, bool> seen;   // (each target and format pair once)
        if (seen.size() < 200 && seen.emplace(im->guest ^ ((uint64_t)wanted << 40) ^ ((uint64_t)im->format << 48), true).second)
            hos_log("rx: render target %08llx %ux%u format %#x sampled as format %#x (pixel format %d)",
                    (unsigned long long)im->guest, im->width, im->height, (unsigned)im->format, (unsigned)wanted,
                    (int)info.format);
    }
    auto texture_view = [&](bool depth) {
        return im->view(dim, depth, base_layer, base_level, level_count, wanted);
    };
    BoundTexture b;
    b.image = im->rendered ? im : nullptr;
    b.source = im->rendered ? 'r' : 't';
    if (im->rendered) dump_sampled(im);
    b.guest = im->guest;
    b.width = im->width;
    b.height = im->height;
    b.format = (int)im->format;
    const TexKind kind = kind_of(im->format, im->pixel_format);
    if (!layers_fit(dim, im->layers) || (dim == WGPUTextureViewDimension_3D) != (im->dimension == WGPUTextureDimension_3D) ||
        (dim == WGPUTextureViewDimension_1D) != (im->dimension == WGPUTextureDimension_1D)) {
        log_limited("tex_dim", 16, "rx: texture of %u layers (dimension %d) bound as view dimension %d: %ux%u format %d "
                    "at %#llx (TIC %ux%ux%u layers %u dimension %d, %s)", im->layers, (int)im->dimension, (int)dim,
                    im->width, im->height, (int)im->format, (unsigned long long)im->guest, info.size.width,
                    info.size.height, info.size.depth, info.layers, (int)info.dimension,
                    im->rendered ? "rendered" : "decoded");
        count("tex_dim_dummy");
        return dummy_texture(desc);
    }
    if (desc.is_depth) {
        if (kind != TexKind::Depth) return dummy_texture(desc);
        b.kind = TexKind::Depth;
        b.view = texture_view(has_stencil(im->format));
        b.sampler = sampler_for(tsc, 2);
        return b;
    }
    switch (kind) {
    case TexKind::Float:
        b.kind = TexKind::Float;
        b.view = texture_view(false);
        b.sampler = sampler_for(tsc, 0);
        return b;
    case TexKind::UnfilterableFloat:
    case TexKind::Depth:   // a depth buffer read as a color texture
        b.kind = TexKind::UnfilterableFloat;
        b.view = texture_view(kind == TexKind::Depth && has_stencil(im->format));
        b.sampler = sampler_for(tsc, 1);
        return b;
    default:   // integer textures: the WGSL backend declares every color texture as f32 for now
        log_limited("tex_int", 8, "rx: integer texture (format %d) bound to a float binding", (int)im->pixel_format);
        return dummy_texture(desc);
    }
}
} // namespace

// ---- the native NVN layer: textures and samplers from NVN objects ----
bool nvn_sampler_of(int id, nvn_samp_par* p);   // rx_nvn.cpp
namespace {

WGPUAddressMode nvn_address_mode(uint8_t w) {
    switch (w) {
    case NVN_WRAP_MODE_REPEAT: return WGPUAddressMode_Repeat;
    case NVN_WRAP_MODE_MIRRORED_REPEAT: return WGPUAddressMode_MirrorRepeat;
    default: return WGPUAddressMode_ClampToEdge;   // (border: the WGSL backend tests the coordinates, see the slots)
    }
}

WGPUCompareFunction nvn_compare_function(uint8_t f) {
    switch (f) {
    case NVN_COMPARE_FUNC_NEVER: return WGPUCompareFunction_Never;
    case NVN_COMPARE_FUNC_LESS: return WGPUCompareFunction_Less;
    case NVN_COMPARE_FUNC_EQUAL: return WGPUCompareFunction_Equal;
    case NVN_COMPARE_FUNC_LEQUAL: return WGPUCompareFunction_LessEqual;
    case NVN_COMPARE_FUNC_GREATER: return WGPUCompareFunction_Greater;
    case NVN_COMPARE_FUNC_NOTEQUAL: return WGPUCompareFunction_NotEqual;
    case NVN_COMPARE_FUNC_GEQUAL: return WGPUCompareFunction_GreaterEqual;
    default: return WGPUCompareFunction_Always;
    }
}

/// mode: 0 filtering as the sampler says, 1 never filtering (unfilterable textures), 2 comparison
WGPUSampler nvn_sampler(const nvn_samp_par& p, int mode) {
    static std::unordered_map<uint64_t, WGPUSampler> cache;
    const uint64_t key = hash_bytes(&p, sizeof p, 0x5a1d0 + (uint64_t)mode);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    WGPUSamplerDescriptor d{};
    d.addressModeU = nvn_address_mode(p.ws);
    d.addressModeV = nvn_address_mode(p.wt);
    d.addressModeW = nvn_address_mode(p.wr);
    const bool filter = mode != 1;
    const bool min_linear = p.minf == NVN_MIN_FILTER_LINEAR || p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST ||
                            p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
    const bool mips = p.minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST;
    const bool mip_linear = p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR || p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
    d.magFilter = filter && p.magf == NVN_MAG_FILTER_LINEAR ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    d.minFilter = filter && min_linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    d.mipmapFilter = filter && mip_linear ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
    d.lodMinClamp = std::clamp(p.lod_min, 0.0f, 32.0f);
    d.lodMaxClamp = mips ? std::clamp(p.lod_max, d.lodMinClamp, 32.0f) : d.lodMinClamp;
    d.compare = mode == 2 ? (p.cmp_mode ? nvn_compare_function(p.cmp_func) : WGPUCompareFunction_LessEqual)
                          : WGPUCompareFunction_Undefined;
    const bool all_linear = d.magFilter == WGPUFilterMode_Linear && d.minFilter == WGPUFilterMode_Linear &&
                            d.mipmapFilter == WGPUMipmapFilterMode_Linear;
    d.maxAnisotropy = all_linear ? (uint16_t)std::clamp(p.aniso, 1.0f, 16.0f) : 1;
    if (p.reduction) log_limited("nvn_reduction", 4, "rx: NVN min/max sampler reduction: averaged");
    WGPUSampler s = wgpuDeviceCreateSampler(dev.device, &d);
    cache[key] = s;
    if (mode == 1) nonfiltering_samplers.insert(s);
    return s;
}

/// A texture's two uniform block slots (as texture_slots makes them from a header and sampler): the component
/// swizzle with the sampler's border axes, depth compare function and LOD bias, then the border color
void nvn_texture_slots(uint8_t* dst, const nvn_tex& t, const nvn_view* v, const nvn_samp_par& p) {
    uint8_t sw[4] = {(uint8_t)t.d.swizzle_r, (uint8_t)t.d.swizzle_g, (uint8_t)t.d.swizzle_b, (uint8_t)t.d.swizzle_a};
    if (v && v->set_swizzle) std::memcpy(sw, v->swz, 4);
    uint32_t src[4];
    for (int i = 0; i < 4; i++)   // NVN ZERO 0, ONE 1, R..A 2..5 -> the header's sources (ONE: one as a float, 7)
        src[i] = sw[i] == NVN_TEXTURE_SWIZZLE_ONE ? 7u : sw[i];
    auto border = [](uint8_t w) { return w == NVN_WRAP_MODE_CLAMP_TO_BORDER || w == NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER; };
    const uint32_t axes = (border(p.ws) ? 1u : 0u) | (border(p.wt) ? 2u : 0u) | (border(p.wr) ? 4u : 0u);
    const uint32_t compare = p.cmp_mode ? (uint32_t)(p.cmp_func - 1) & 7u : 3u;
    const int32_t bias = (int32_t)std::lround(p.lod_bias * 256.0f);
    const uint32_t slot[4] = {src[0] | axes << 3, src[1] | compare << 3, src[2] | (uint32_t)bias << 8, src[3]};
    std::memcpy(dst, slot, 16);
    std::memcpy(dst + 16, p.border, 16);
}

/// The texture description an NVN texture and view give (the texture's whole storage; the view picks levels and
/// layers). valid false: not a texture the renderer samples (buffer textures, unknown formats).
TexInfo nvn_tex_info(const nvn_tex& t, const nvn_view* v) {
    TexInfo i;
    const nvn_texdesc& d = t.d;
    i.format = nvn_pixel_format(v && v->set_format ? v->format : d.format);
    if (i.format == PixelFormat::Invalid) return i;
    i.levels = std::clamp<u32>(d.levels, 1, 16);
    i.linear = nvn_tex_is_linear(&d);
    if (i.linear) i.pitch = nvn_tex_linear_stride(&d);
    else i.block = {0, t.L.tileH, t.L.tileD};
    i.size = {std::max(1u, d.width), std::max(1u, d.height), 1};
    switch (d.target) {
    case NVN_TEXTURE_TARGET_1D:
        i.dimension = WGPUTextureDimension_1D;
        i.size.height = 1;
        break;
    case NVN_TEXTURE_TARGET_1D_ARRAY:
        i.dimension = WGPUTextureDimension_1D;
        i.layers = std::max(1u, d.height);
        i.size.height = 1;
        break;
    case NVN_TEXTURE_TARGET_2D: case NVN_TEXTURE_TARGET_RECTANGLE: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE:
        break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
        i.layers = std::max(1u, d.depth);
        break;
    case NVN_TEXTURE_TARGET_CUBEMAP: i.layers = 6; break;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: i.layers = std::max(6u, d.depth); break;   // (depth counts layer-faces)
    case NVN_TEXTURE_TARGET_3D:
        i.dimension = WGPUTextureDimension_3D;
        i.size.depth = std::max(1u, d.depth);
        break;
    default: return i;   // buffer textures
    }
    if (i.linear) i.levels = 1;
    if (i.dimension == WGPUTextureDimension_1D) i.levels = 1;
    i.valid = true;
    return i;
}

}  // namespace

/// The levels a view of the texture reaches (a view of its first level alone can sample the render target itself)
static uint32_t nvn_levels_used(const TexInfo& info, const nvn_view* v) {
    if (v && v->set_levels && v->nlevels) return std::min<uint32_t>(info.levels, (uint32_t)v->base_level + v->nlevels);
    return info.levels;
}

/// Decoded textures being decoded again (their memory was rewritten): the old image, drawn until the new one is in
static std::unordered_map<uint64_t, std::unique_ptr<Image>> nvn_stale_images;

/// What nvn_texture_resolve settled on, for the memo in nvn_texture: the texture's guest address (the image there is
/// checked again), the decoded image it took (its stale flag), the image assembled from render targets (assembled
/// again at each use) and its info.
struct NvnTexResolved {
    uint64_t guest = 0;
    Image* src = nullptr;
    Image* assembled = nullptr;
    Image* used = nullptr;   // (the image bound: its last use is renewed on memo hits)
    uint64_t dec_key = 0;    // (src's key in decoded: a hit checks it is still the image there)
    TexInfo info{};
    bool memo_ok = false;   // (a placeholder or a stale image drawn while decoding: not remembered)
};

BoundTexture nvn_texture_resolve(const NvnState& s, uint64_t tex_handle, uint64_t samp_handle,
                                 const Shader::TextureDescriptor& desc, NvnTexResolved& res) {
    (void)s;
    const int tid = (int)(tex_handle & 0xFFFFF);
    const int sid = (int)(((samp_handle ? samp_handle : tex_handle) >> 20) & 0xFFF);
    nvn_tex t;
    nvn_view v;
    bool has_view = false;
    if (!tid || !nvn_texture_of(tid, &t, &v, &has_view) || !t.gpu) return dummy_texture(desc);
    nvn_samp_par sp{};
    if (!nvn_sampler_of(sid, &sp)) {
        sp.minf = NVN_MIN_FILTER_LINEAR, sp.magf = NVN_MAG_FILTER_LINEAR;
        sp.ws = sp.wt = sp.wr = NVN_WRAP_MODE_CLAMP_TO_EDGE;
        sp.lod_max = 1000.0f, sp.aniso = 1.0f;
    }
    const nvn_view* vp = has_view ? &v : nullptr;
    const TexInfo info = nvn_tex_info(t, vp);
    if (!info.valid) return dummy_texture(desc);
    const WGPUTextureViewDimension dim = view_dimension(desc.type);
    const uint64_t va = t.gpu, guest = nvn_gpu_guest(va);
    res.guest = guest;
    res.info = info;
    bool from_stale = false;
    Image* im = nullptr;
    // drawn by the GPU: the render target at the texture's storage
    if (Image* rt = guest ? image_at(guest) : nullptr;
        rt && rt->rendered && rt->width >= info.size.width && rt->height >= info.size.height &&
        rt->dimension == info.dimension && rt->layers >= info.layers && rt->levels >= nvn_levels_used(info, vp) &&
        layers_fit(dim, rt->layers) &&
        ((dim == WGPUTextureViewDimension_3D) == (rt->dimension == WGPUTextureDimension_3D)) &&
        ((dim == WGPUTextureViewDimension_1D) == (rt->dimension == WGPUTextureDimension_1D)))
        im = rt;   // (a texture whose layers or levels were drawn one by one is assembled below, as in texture_for_handle)
    if (!im) {   // from memory: decoded (and cached by what it was decoded from)
        // (field by field: TexInfo's padding bytes are not initialized; the texture object's own fields, such as its
        // swizzle, do not change what is decoded: one decode per layout, as for a TIC)
        const uint32_t fields[] = {(uint32_t)info.format, (uint32_t)info.dimension, info.size.width, info.size.height,
                                   info.size.depth, info.layers, info.levels, info.block.width, info.block.height,
                                   info.block.depth, info.tile_width_spacing, info.linear ? 1u : 0u, info.pitch};
        const uint64_t key = hash_bytes(fields, sizeof fields, va ^ 0x4e564e00ull);
        res.dec_key = key;
        auto it = decoded.find(key);
        if (it != decoded.end()) {
            if (!it->second) return dummy_texture(desc);
            im = it->second.get();
            // RX_REDECODE=<guest hex> (debugging): that texture decoded again every frame
            static const uint64_t redecode = getenv("RX_REDECODE") ? strtoull(getenv("RX_REDECODE"), nullptr, 16) : 0;
            if (redecode == guest && im->decoded_seq && im->checked_frame != frame_number)
                im->checked_frame = frame_number, im->stale = true;
            if (im->decoded_seq) {
                if (im->stale) {   // (rewritten: decoded again; the old one is drawn until the new one is in)
                    forget(im);
                    nvn_stale_images[key] = std::move(it->second);
                    decoded.erase(it);
                    decoded_generation++;
                    im = nullptr;
                }
            } else if (im->checked_frame != frame_number) {
                im->checked_frame = frame_number;
                if (sample_hash(va, guest_size(info)) != im->content_hash) {
                    forget(im);
                    decoded.erase(it);
                    decoded_generation++;
                    im = nullptr;
                }
            }
        }
        const TICEntry no_tic{};
        const bool cheap = (info.linear || rewritten_textures.count(guest)) && guest_size(info) <= (16u << 20);
        if (!im && decode_async() && !cheap && !(draw_must_run && guest_size(info) <= (64u << 20))) {
            queue_decode(key, no_tic, info, va);
            decode_pending_result = true;
            count("tex_decoding");
            // (a texture decoded again after the CPU rewrote it: its old image meanwhile, not a placeholder (a flash
            // of black or gray for the frames the decode takes))
            auto f = nvn_stale_images.find(key);
            if (f == nvn_stale_images.end() || !f->second) return dummy_texture(desc);
            im = f->second.get();
            from_stale = true;
            count("tex_stale_drawn");
        } else if (im && !nvn_stale_images.empty()) {
            nvn_stale_images.erase(key);   // (its new image is in)
        }
        if (!im) {
            // (a render target drew this memory on the GPU only and does not fit as the texture: the decode reads
            // what guest memory held before, unless it was written back. The gloom noise volumes were such a case:
            // block views, written back now. Logged once an address, to find others)
            const Image* under = guest ? image_at(guest) : nullptr;
            if (!under && guest && info.dimension == WGPUTextureDimension_3D) under = image_at(nvn_slice_key(guest, 0));
            // (layers, levels and 3D slices of the texture's own size and kind are assembled from their targets)
            const bool assembled = under && under->width == info.size.width && under->height == info.size.height &&
                                   !VideoCore::Surface::IsPixelFormatBCn(info.format) &&
                                   !VideoCore::Surface::IsPixelFormatASTC(info.format);
            if (under && under->rendered && !under->block_view && !assembled) {
                static std::unordered_set<uint64_t> told;
                if (told.size() < 256 && told.insert(guest).second)
                    hos_log("rx: texture %08llx %ux%ux%u format %d decoded from memory a render target drew on the GPU "
                            "(%ux%u format %#x%s)", (unsigned long long)guest, info.size.width, info.size.height,
                            std::max(info.size.depth, info.layers), (int)info.format, under->width, under->height,
                            (unsigned)under->format, under->guest >> 32 ? ", 3D slices" : "");
            }
            const int64_t t0 = hos_now_ns();
            auto fresh = decode(no_tic, info, va);
            count("t_decode", hos_now_ns() - t0);
            spike.decode_ns += hos_now_ns() - t0;
            im = fresh.get();
            if (im) remember(im);
            decoded[key] = std::move(fresh);
            decoded_generation++;
            if (!im) return dummy_texture(desc);
        }
    }
    im->last_use = frame_number;
    res.used = im;
    if (!im->rendered && !info.linear && info.dimension != WGPUTextureDimension_3D && (info.layers > 1 || info.levels > 1)) {
        assemble_from_render_targets(*im, info, guest);
        res.assembled = im;
    } else if (!im->rendered && info.dimension == WGPUTextureDimension_3D) {
        assemble_3d_slices(*im, info, guest);
        res.assembled = im;
    }
    if (!im->rendered) res.src = im;
    res.memo_ok = !from_stale;
    const uint32_t base_layer = im->dimension == WGPUTextureDimension_3D ? 0 : (vp && vp->set_layers ? vp->min_layer : 0);
    const uint32_t base_level = vp && vp->set_levels ? vp->base_level : 0;
    const uint32_t level_count = vp && vp->set_levels && vp->nlevels ? vp->nlevels : 0;
    const WGPUTextureFormat wanted = wgpu_format(info.format);
    auto texture_view = [&](bool depth) {
        return im->view(dim, depth, base_layer, base_level, level_count, wanted);
    };
    if (im->rendered && wanted != im->format && frame_dump_active()) {   // (diagnostic: drawn as one format, read as another)
        char b[160];
        std::snprintf(b, sizeof b, "texture %llx %ux%u read as format %#x, drawn as %#x", (unsigned long long)guest,
                      info.size.width, info.size.height, (unsigned)wanted, (unsigned)im->format);
        dump_draw(b);
    }
    BoundTexture b;
    b.image = im->rendered ? im : nullptr;
    b.source = im->rendered ? 'r' : 't';
    if (im->rendered) dump_sampled(im);
    b.guest = im->guest;
    b.width = im->width;
    b.height = im->height;
    b.format = (int)im->format;
    nvn_texture_slots(b.slots, t, vp, sp);
    const TexKind kind = kind_of(im->format, im->pixel_format);
    if (!layers_fit(dim, im->layers) || (dim == WGPUTextureViewDimension_3D) != (im->dimension == WGPUTextureDimension_3D) ||
        (dim == WGPUTextureViewDimension_1D) != (im->dimension == WGPUTextureDimension_1D)) {
        count("tex_dim_dummy");
        return dummy_texture(desc);
    }
    if (desc.is_depth) {
        if (kind != TexKind::Depth) return dummy_texture(desc);
        b.kind = TexKind::Depth;
        b.view = texture_view(has_stencil(im->format));
        b.sampler = nvn_sampler(sp, 2);
        return b;
    }
    switch (kind) {
    case TexKind::Float:
        b.kind = TexKind::Float;
        b.view = texture_view(false);
        b.sampler = nvn_sampler(sp, 0);
        return b;
    case TexKind::UnfilterableFloat:
    case TexKind::Depth:
        b.kind = TexKind::UnfilterableFloat;
        b.view = texture_view(kind == TexKind::Depth && has_stencil(im->format));
        b.sampler = nvn_sampler(sp, 1);
        return b;
    default:
        log_limited("tex_int", 8, "rx: integer texture (format %d) bound to a float binding", (int)im->pixel_format);
        return dummy_texture(desc);
    }
}


/// The native path's texture bindings, remembered (rx_draw's per-draw lookups were 16% of the replay thread): the same
/// handles and view type give the same binding while no texture was decoded or dropped, no image changed address, the
/// image at the texture's memory is the same one (and as rendered or not), and the decoded image is not stale. Across
/// frames too while neither handle was registered again or had its texture finalized (per-handle versions,
/// rx_nvn.cpp; 914 of ~1500 misses a frame were the frame's first use), except for a handle that follows its storage
/// (a finalized transient target: what lives there can change any frame) and an image decoded without write tracking
/// (checked by its samples once a frame). RX_NO_NVN_TEX_MEMO=1: every lookup resolves again.
BoundTexture nvn_texture(const NvnState& s, uint64_t tex_handle, uint64_t samp_handle,
                         const Shader::TextureDescriptor& desc) {
    static const bool no_memo = env_flag("RX_NO_NVN_TEX_MEMO");
    NvnTexResolved res;
    if (no_memo || frame_dump_active()) return nvn_texture_resolve(s, tex_handle, samp_handle, desc, res);
    struct Memo {
        uint64_t tex = 0, samp = 0, frame = ~0ull, decoded = 0, images = 0, guest = 0, at_id = 0;
        uint64_t src_checked = ~0ull;   // (decoded_generation when src was last found in the map)
        uint32_t dkey = 0, tver = 0, sver = 0;
        uint64_t dec_key = 0;
        bool at_rendered = false, valid = false, per_frame = false;
        Image* src = nullptr;
        Image* assembled = nullptr;
        Image* used = nullptr;
        TexInfo info{};
        BoundTexture b;
    };
    static Memo memo[4096];
    const uint32_t dkey = (uint32_t)desc.type | (desc.is_depth ? 0x100u : 0u) | (desc.is_multisample ? 0x200u : 0u);
    const uint64_t key = hash_bytes(&tex_handle, 8, samp_handle ^ (uint64_t)dkey << 40);
    Memo& m = memo[key & 4095];
    const int tid = (int)(tex_handle & 0xFFFFF), sid = (int)(((samp_handle ? samp_handle : tex_handle) >> 20) & 0xFFF);
    static const bool frame_only = env_flag("RX_NVN_TEX_MEMO_FRAME");   // (the memo within a frame only, as before)
    // an entry holds while: the same handles and view type, neither handle changed (versions); then a bound image
    // (decoded or rendered) is checked on its own: a decoded one is still the decode of its key and not stale, the image
    // at the texture's memory is the same one (both below). A binding without one (a dummy) holds within the frame
    // while no texture was decoded or dropped and no image moved, as before.
    bool ok = m.valid && m.tex == tex_handle && m.samp == samp_handle && m.dkey == dkey &&
              m.tver == nvn_texture_version(tid) && m.sver == nvn_sampler_version(sid);
    if (ok && (!m.used || m.per_frame || frame_only))
        ok = m.frame == (uint64_t)frame_number && (m.used || (m.decoded == decoded_generation && m.images == images_generation()));
    if (ok && m.src && m.src_checked != decoded_generation) {   // (the map unchanged since: src is still its entry)
        auto it = decoded.find(m.dec_key);
        ok = it != decoded.end() && it->second.get() == m.src;
        if (ok) m.src_checked = decoded_generation;
    }
    if (ok) {
        Image* at = m.guest ? image_at(m.guest) : nullptr;
        if ((at ? at->id : 0) == m.at_id && (!at || at->rendered == m.at_rendered) && (!m.src || !m.src->stale)) {
            if (m.assembled) {
                if (m.info.dimension == WGPUTextureDimension_3D) assemble_3d_slices(*m.assembled, m.info, m.guest);
                else assemble_from_render_targets(*m.assembled, m.info, m.guest);
            }
            count("nvn_tex_memo_hits");
            if (m.used) m.used->last_use = frame_number;
            if (m.b.image) dump_sampled(m.b.image);
            m.frame = (uint64_t)frame_number;
            return m.b;
        }
    }
    const uint32_t tver = nvn_texture_version(tid), sver = nvn_sampler_version(sid);   // (before: a change meanwhile misses)
    const BoundTexture b = nvn_texture_resolve(s, tex_handle, samp_handle, desc, res);
    m.valid = res.memo_ok && b.view != nullptr;
    if (!m.valid) return b;
    m.tver = tver, m.sver = sver, m.used = res.used, m.dec_key = res.dec_key;
    m.per_frame = nvn_texture_follows_storage(tid) || (res.src && !res.src->decoded_seq);
    Image* at = res.guest ? image_at(res.guest) : nullptr;
    m.tex = tex_handle, m.samp = samp_handle, m.dkey = dkey, m.frame = (uint64_t)frame_number;
    m.decoded = decoded_generation, m.images = images_generation();
    m.guest = res.guest, m.at_id = at ? at->id : 0, m.at_rendered = at && at->rendered;
    m.src = res.src, m.assembled = res.assembled, m.info = res.info, m.b = b;
    m.src_checked = ~0ull;
    if (m.src) {   // (checked as a hit checks it: a source not in the map is looked up each time, as before)
        auto it = decoded.find(m.dec_key);
        if (it != decoded.end() && it->second.get() == m.src) m.src_checked = decoded_generation;
    }
    return b;
}

} // namespace rx
