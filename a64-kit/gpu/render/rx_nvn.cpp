// The native NVN layer's replay (HOS_NVN_NATIVE=1): the command records hos/nvn_native.c made, run at the submit with
// the renderer. See rx_nvn.h. Each record becomes renderer work built from NVN state directly: render targets from
// NVN texture objects, clears, draws and dispatches from the bound programs and state objects. Effects the CPU can see
// (events, counters, uniform updates, buffer copies) happen in the order of the recordings.
#include <atomic>
#include <vector>
#include <thread>
#include <deque>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

#include "rx_internal.h"
#include "rx_nvn.h"
#if RX_WEB
#include <emscripten/threading.h>
#include <pthread.h>
#endif

extern "C" {
uint32_t* rx_collect_written(int64_t* count);
void rx_begin_submission_written(uint32_t* pages, int64_t n);
void rx_flush(void);
void rx_memory_written(uint64_t gpu_va, uint64_t size);
int rx_gpu_writes_pending(uint64_t gpu_va, uint64_t size);
void rx_sync_gpu_writes(uint64_t gpu_va, uint64_t size);
int rx_present(uint64_t guest, uint32_t width, uint32_t height);
void rx_wait_idle(void);
void hos_host_thread_register_self(const char* name);
void hos_thread_boost(int level);
#if RX_WEB
void rx_web_idle(volatile uint32_t* wake, uint32_t seen);
#endif
#if defined(A64_WRITE_TRACK) && !defined(__EMSCRIPTEN__)
void hos_mark_written(const void* p, size_t n);
#endif
}

namespace rx {
/// Guest memory this code wrote with the CPU: the soft write watch's mark (Windows' write watch sees it by itself)
inline void rx_mark_written(const void* p, size_t n) {
#if defined(A64_WRITE_TRACK) && !defined(__EMSCRIPTEN__)
    hos_mark_written(p, n);
#else
    (void)p, (void)n;
#endif
}

// ---- formats ----
PixelFormat nvn_pixel_format(int f) {
    using P = PixelFormat;
    switch (f) {
    case NVN_FORMAT_R8: return P::R8_UNORM;
    case NVN_FORMAT_R8SN: return P::R8_SNORM;
    case NVN_FORMAT_R8UI: return P::R8_UINT;
    case NVN_FORMAT_R8I: return P::R8_SINT;
    case NVN_FORMAT_R16F: return P::R16_FLOAT;
    case NVN_FORMAT_R16: return P::R16_UNORM;
    case NVN_FORMAT_R16SN: return P::R16_SNORM;
    case NVN_FORMAT_R16UI: return P::R16_UINT;
    case NVN_FORMAT_R16I: return P::R16_SINT;
    case NVN_FORMAT_R32F: return P::R32_FLOAT;
    case NVN_FORMAT_R32UI: return P::R32_UINT;
    case NVN_FORMAT_R32I: return P::R32_SINT;
    case NVN_FORMAT_RG8: return P::R8G8_UNORM;
    case NVN_FORMAT_RG8SN: return P::R8G8_SNORM;
    case NVN_FORMAT_RG8UI: return P::R8G8_UINT;
    case NVN_FORMAT_RG8I: return P::R8G8_SINT;
    case NVN_FORMAT_RG16F: return P::R16G16_FLOAT;
    case NVN_FORMAT_RG16: return P::R16G16_UNORM;
    case NVN_FORMAT_RG16SN: return P::R16G16_SNORM;
    case NVN_FORMAT_RG16UI: return P::R16G16_UINT;
    case NVN_FORMAT_RG16I: return P::R16G16_SINT;
    case NVN_FORMAT_RG32F: return P::R32G32_FLOAT;
    case NVN_FORMAT_RG32UI: return P::R32G32_UINT;
    case NVN_FORMAT_RG32I: return P::R32G32_SINT;
    case NVN_FORMAT_RGB32F: return P::R32G32B32_FLOAT;
    case NVN_FORMAT_RGBA8: case NVN_FORMAT_RGBX8: return P::A8B8G8R8_UNORM;
    case NVN_FORMAT_RGBA8SN: case NVN_FORMAT_RGBX8SN: return P::A8B8G8R8_SNORM;
    case NVN_FORMAT_RGBA8UI: case NVN_FORMAT_RGBX8UI: return P::A8B8G8R8_UINT;
    case NVN_FORMAT_RGBA8I: case NVN_FORMAT_RGBX8I: return P::A8B8G8R8_SINT;
    case NVN_FORMAT_RGBA16F: return P::R16G16B16A16_FLOAT;
    case NVN_FORMAT_RGBX16F: return P::R16G16B16X16_FLOAT;
    case NVN_FORMAT_RGBA16: case NVN_FORMAT_RGBX16: return P::R16G16B16A16_UNORM;
    case NVN_FORMAT_RGBA16SN: case NVN_FORMAT_RGBX16SN: return P::R16G16B16A16_SNORM;
    case NVN_FORMAT_RGBA16UI: case NVN_FORMAT_RGBX16UI: return P::R16G16B16A16_UINT;
    case NVN_FORMAT_RGBA16I: case NVN_FORMAT_RGBX16I: return P::R16G16B16A16_SINT;
    case NVN_FORMAT_RGBA32F: case NVN_FORMAT_RGBX32F: return P::R32G32B32A32_FLOAT;
    case NVN_FORMAT_RGBA32UI: case NVN_FORMAT_RGBX32UI: return P::R32G32B32A32_UINT;
    case NVN_FORMAT_RGBA32I: case NVN_FORMAT_RGBX32I: return P::R32G32B32A32_SINT;
    case NVN_FORMAT_STENCIL8: return P::S8_UINT;
    case NVN_FORMAT_DEPTH16: return P::D16_UNORM;
    case NVN_FORMAT_DEPTH24: return P::X8_D24_UNORM;
    case NVN_FORMAT_DEPTH32F: return P::D32_FLOAT;
    case NVN_FORMAT_DEPTH24_STENCIL8: return P::S8_UINT_D24_UNORM;
    case NVN_FORMAT_DEPTH32F_STENCIL8: return P::D32_FLOAT_S8_UINT;
    case NVN_FORMAT_RGBX8_SRGB: case NVN_FORMAT_RGBA8_SRGB: return P::A8B8G8R8_SRGB;
    case NVN_FORMAT_RGBA4: return P::A4B4G4R4_UNORM;
    case NVN_FORMAT_RGB5A1: return P::A1B5G5R5_UNORM;
    case NVN_FORMAT_RGB565: return P::B5G6R5_UNORM;
    case NVN_FORMAT_BGR565: return P::R5G6B5_UNORM;
    case NVN_FORMAT_RGB10A2: return P::A2B10G10R10_UNORM;
    case NVN_FORMAT_RGB10A2UI: return P::A2B10G10R10_UINT;
    case NVN_FORMAT_R11G11B10F: return P::B10G11R11_FLOAT;
    case NVN_FORMAT_RGB9E5F: return P::E5B9G9R9_FLOAT;
    case NVN_FORMAT_RGB_DXT1: case NVN_FORMAT_RGBA_DXT1: return P::BC1_RGBA_UNORM;
    case NVN_FORMAT_RGB_DXT1_SRGB: case NVN_FORMAT_RGBA_DXT1_SRGB: return P::BC1_RGBA_SRGB;
    case NVN_FORMAT_RGBA_DXT3: return P::BC2_UNORM;
    case NVN_FORMAT_RGBA_DXT3_SRGB: return P::BC2_SRGB;
    case NVN_FORMAT_RGBA_DXT5: return P::BC3_UNORM;
    case NVN_FORMAT_RGBA_DXT5_SRGB: return P::BC3_SRGB;
    case NVN_FORMAT_RGTC1_UNORM: return P::BC4_UNORM;
    case NVN_FORMAT_RGTC1_SNORM: return P::BC4_SNORM;
    case NVN_FORMAT_RGTC2_UNORM: return P::BC5_UNORM;
    case NVN_FORMAT_RGTC2_SNORM: return P::BC5_SNORM;
    case NVN_FORMAT_BPTC_UNORM: return P::BC7_UNORM;
    case NVN_FORMAT_BPTC_UNORM_SRGB: return P::BC7_SRGB;
    case NVN_FORMAT_BPTC_SFLOAT: return P::BC6H_SFLOAT;
    case NVN_FORMAT_BPTC_UFLOAT: return P::BC6H_UFLOAT;
    case NVN_FORMAT_RGBA_ASTC_4x4: return P::ASTC_2D_4X4_UNORM;
    case NVN_FORMAT_RGBA_ASTC_5x4: return P::ASTC_2D_5X4_UNORM;
    case NVN_FORMAT_RGBA_ASTC_5x5: return P::ASTC_2D_5X5_UNORM;
    case NVN_FORMAT_RGBA_ASTC_6x5: return P::ASTC_2D_6X5_UNORM;
    case NVN_FORMAT_RGBA_ASTC_6x6: return P::ASTC_2D_6X6_UNORM;
    case NVN_FORMAT_RGBA_ASTC_8x5: return P::ASTC_2D_8X5_UNORM;
    case NVN_FORMAT_RGBA_ASTC_8x6: return P::ASTC_2D_8X6_UNORM;
    case NVN_FORMAT_RGBA_ASTC_8x8: return P::ASTC_2D_8X8_UNORM;
    case NVN_FORMAT_RGBA_ASTC_10x5: return P::ASTC_2D_10X5_UNORM;
    case NVN_FORMAT_RGBA_ASTC_10x6: return P::ASTC_2D_10X6_UNORM;
    case NVN_FORMAT_RGBA_ASTC_10x8: return P::ASTC_2D_10X8_UNORM;
    case NVN_FORMAT_RGBA_ASTC_10x10: return P::ASTC_2D_10X10_UNORM;
    case NVN_FORMAT_RGBA_ASTC_12x10: return P::ASTC_2D_12X10_UNORM;
    case NVN_FORMAT_RGBA_ASTC_12x12: return P::ASTC_2D_12X12_UNORM;
    case NVN_FORMAT_RGBA_ASTC_4x4_SRGB: return P::ASTC_2D_4X4_SRGB;
    case NVN_FORMAT_RGBA_ASTC_5x4_SRGB: return P::ASTC_2D_5X4_SRGB;
    case NVN_FORMAT_RGBA_ASTC_5x5_SRGB: return P::ASTC_2D_5X5_SRGB;
    case NVN_FORMAT_RGBA_ASTC_6x5_SRGB: return P::ASTC_2D_6X5_SRGB;
    case NVN_FORMAT_RGBA_ASTC_6x6_SRGB: return P::ASTC_2D_6X6_SRGB;
    case NVN_FORMAT_RGBA_ASTC_8x5_SRGB: return P::ASTC_2D_8X5_SRGB;
    case NVN_FORMAT_RGBA_ASTC_8x6_SRGB: return P::ASTC_2D_8X6_SRGB;
    case NVN_FORMAT_RGBA_ASTC_8x8_SRGB: return P::ASTC_2D_8X8_SRGB;
    case NVN_FORMAT_RGBA_ASTC_10x5_SRGB: return P::ASTC_2D_10X5_SRGB;
    case NVN_FORMAT_RGBA_ASTC_10x6_SRGB: return P::ASTC_2D_10X6_SRGB;
    case NVN_FORMAT_RGBA_ASTC_10x8_SRGB: return P::ASTC_2D_10X8_SRGB;
    case NVN_FORMAT_RGBA_ASTC_10x10_SRGB: return P::ASTC_2D_10X10_SRGB;
    case NVN_FORMAT_RGBA_ASTC_12x10_SRGB: return P::ASTC_2D_12X10_SRGB;
    case NVN_FORMAT_RGBA_ASTC_12x12_SRGB: return P::ASTC_2D_12X12_SRGB;
    case NVN_FORMAT_BGRA8: case NVN_FORMAT_BGRX8: return P::B8G8R8A8_UNORM;
    case NVN_FORMAT_BGRA8_SRGB: case NVN_FORMAT_BGRX8_SRGB: return P::B8G8R8A8_SRGB;
    default: return P::Invalid;
    }
}

// ---- texture and sampler descriptor registrations ----
namespace {
struct TexReg {
    nvn_tex t{};
    nvn_view v{};
    bool has_view = false, ok = false;
    bool released = false;   // (its texture was finalized: the registration names the one living at its storage)
};
struct SampReg {
    nvn_samp_par p{};
    bool ok = false;
};
std::mutex reg_mu;   // (registrations come from any game thread)
std::vector<TexReg> tex_regs;
std::vector<SampReg> samp_regs;
// registrations by their texture's serial: a finalized texture marks the ones naming it (a set of every finalized
// serial grew for the whole session: TOTK finalizes transient textures every frame)
std::unordered_map<uint64_t, std::vector<int>> regs_by_serial;
// per handle, bumped whenever what the handle resolves to may change (registered again, its texture finalized): the
// texture binding memo (rx_texture.cpp) holds across frames while it stays (read without the lock: fixed arrays)
std::atomic<uint32_t> tex_version[1 << 20], samp_version[1 << 12];
}  // namespace

uint32_t nvn_texture_version(int id) { return id >= 0 && id < (1 << 20) ? tex_version[id].load(std::memory_order_relaxed) : 0; }
uint32_t nvn_sampler_version(int id) { return id >= 0 && id < (1 << 12) ? samp_version[id].load(std::memory_order_relaxed) : 0; }
bool nvn_texture_follows_storage(int id) {
    std::lock_guard lk(reg_mu);
    return id >= 0 && (size_t)id < tex_regs.size() && tex_regs[id].ok && tex_regs[id].released;
}

void nvn_register_texture(int id, const nvn_tex& t, const nvn_view* v) {
    if (id < 0 || id >= (1 << 20)) return;
    std::lock_guard lk(reg_mu);
    if ((size_t)id >= tex_regs.size()) tex_regs.resize((size_t)id + 1024);
    TexReg& r = tex_regs[id];
    if (r.ok) {   // (no longer named by this handle)
        auto it = regs_by_serial.find(r.t.serial);
        if (it != regs_by_serial.end()) {
            std::erase(it->second, id);
            if (it->second.empty()) regs_by_serial.erase(it);
        }
    }
    r.t = t;
    r.has_view = v != nullptr;
    r.v = v ? *v : nvn_view{};
    r.ok = true;
    r.released = false;
    regs_by_serial[t.serial].push_back(id);
    tex_version[id].fetch_add(1, std::memory_order_relaxed);
}
void nvn_register_sampler(int id, const nvn_samp_par& p) {
    if (id < 0 || id >= (1 << 12)) return;
    std::lock_guard lk(reg_mu);
    if ((size_t)id >= samp_regs.size()) samp_regs.resize((size_t)id + 256);
    samp_regs[id].p = p;
    samp_regs[id].ok = true;
    samp_version[id].fetch_add(1, std::memory_order_relaxed);
}
void nvn_texture_released(uint64_t serial) {
    std::lock_guard lk(reg_mu);
    auto it = regs_by_serial.find(serial);
    if (it == regs_by_serial.end()) return;
    for (int id : it->second) {
        tex_regs[id].released = true;
        tex_version[id].fetch_add(1, std::memory_order_relaxed);
    }
}
bool nvn_texture_of(int id, nvn_tex* t, nvn_view* v, bool* has_view) {
    std::lock_guard lk(reg_mu);
    if (id < 0 || (size_t)id >= tex_regs.size() || !tex_regs[id].ok) return false;
    const TexReg& r = tex_regs[id];
    *t = r.t;
    *v = r.v;
    *has_view = r.has_view;
    // the game re-initialises transient targets every frame at the same storage without registering them again (a
    // descriptor names memory on the console): a finalized texture's registration names the one living there now
    if (r.released) {
        nvn_tex live;
        if (nvn_tex_live_at(r.t.gpu, &live)) *t = live;
    }
    return true;
}
bool nvn_sampler_of(int id, nvn_samp_par* p) {
    std::lock_guard lk(reg_mu);
    if (id < 0 || (size_t)id >= samp_regs.size() || !samp_regs[id].ok) return false;
    *p = samp_regs[id].p;
    return true;
}

// ---- the driver's constant banks (what NVN-compiled programs read for their bindings) ----
// c0: the storage buffer table (16 bytes a slot: address, size) at 0x110 (vertex), 0x510 (fragment), 0x310 (compute);
// compute also has its grid at 0x200 and its uniform buffers' table at 0x210; vertex programs clamp the point size to
// 0x610. c2: texture handles at 0x168 + 8 slot and 0x20 + 8 slot (BindTexture's combined handles, 32 slots), sampler
// handles at 0x568 + 8 slot, image handles at 0x120 + 8 slot. (Measured on TOTK by the MIT-licensed reference layer.)
void nvn_driver_bank(const NvnState& s, int bank, int stage, uint8_t d[0xA00]) {
    std::memset(d, 0, 0xA00);
    if (stage < 0 || stage > 5) return;
    if (bank == 0) {
        const int base = stage == 0 ? 0x110 : stage == 5 ? 0x310 : stage == 1 ? 0x510 : -1;
        if (base >= 0)
            for (int r = 0; r < 16; r++) {
                const uint64_t a = s.ssbo[stage][r].addr;
                const uint32_t n = (uint32_t)s.ssbo[stage][r].size;
                std::memcpy(d + base + 16 * r, &a, 8);
                std::memcpy(d + base + 16 * r + 8, &n, 4);
            }
        if (stage == 5) {
            std::memcpy(d + 0x200, s.grid, sizeof s.grid);
            for (int r = 0; r < 16; r++) {
                const uint64_t a = s.ubo[5][r].addr;
                const uint32_t n = (uint32_t)s.ubo[5][r].size;
                std::memcpy(d + 0x210 + 16 * r, &a, 8);
                std::memcpy(d + 0x210 + 16 * r + 8, &n, 4);
            }
        } else if (stage == 0) {
            const float point_max = 2047.9375f;
            std::memcpy(d + 0x610, &point_max, 4);
        }
    } else if (bank == 2) {
        for (int r = 0; r < 128; r++) {
            const uint32_t th = (uint32_t)s.tex[stage][r], sh = (uint32_t)s.samp[stage][r];
            std::memcpy(d + 0x168 + 8 * r, &th, 4);
            std::memcpy(d + 0x568 + 8 * r, &sh, 4);
        }
        for (int r = 0; r < 32; r++) {
            const uint32_t th = (uint32_t)s.tex[stage][r];
            std::memcpy(d + 0x20 + 8 * r, &th, 4);
        }
        for (int r = 0; r < 9; r++) {
            const uint32_t ih = (uint32_t)s.img[stage][r];
            std::memcpy(d + 0x120 + 8 * r, &ih, 4);
        }
    }
}
/// One word of a driver bank, as nvn_driver_bank lays it out (without building the 2.5 KB bank: the programs' handle
/// and buffer reads, several per draw)
uint32_t nvn_driver_bank_word(const NvnState& s, int bank, int stage, uint32_t offset) {
    if (stage < 0 || stage > 5 || offset + 4 > 0xA00 || (offset & 3)) {
        uint8_t d[0xA00];
        nvn_driver_bank(s, bank, stage, d);
        uint32_t w = 0;
        if (offset + 4 <= sizeof d) std::memcpy(&w, d + offset, 4);
        return w;
    }
    auto lo = [](uint64_t v) { return (uint32_t)v; };
    auto hi = [](uint64_t v) { return (uint32_t)(v >> 32); };
    if (bank == 2) {
        if (offset & 7) return 0;   // (the upper words of the 8-byte slots)
        if (offset >= 0x568 && offset < 0x568 + 8 * 128) return lo(s.samp[stage][(offset - 0x568) / 8]);
        if (offset >= 0x168 && offset < 0x168 + 8 * 128) return lo(s.tex[stage][(offset - 0x168) / 8]);
        if (offset >= 0x120 && offset < 0x120 + 8 * 9) return lo(s.img[stage][(offset - 0x120) / 8]);
        if (offset >= 0x20 && offset < 0x20 + 8 * 32) return lo(s.tex[stage][(offset - 0x20) / 8]);
        return 0;
    }
    if (bank != 0) return 0;
    const uint32_t base = stage == 0 ? 0x110 : stage == 5 ? 0x310 : stage == 1 ? 0x510 : 0;
    auto table = [&](const nvn_p_bind_buf* t, uint32_t at) -> uint32_t {   // 16-byte entries: address, size, 0
        const nvn_p_bind_buf& e = t[(offset - at) / 16];
        switch ((offset - at) % 16) {
        case 0: return lo(e.addr);
        case 4: return hi(e.addr);
        case 8: return (uint32_t)e.size;
        default: return 0;
        }
    };
    if (stage == 5) {
        if (offset >= 0x200 && offset < 0x200 + sizeof s.grid) return s.grid[(offset - 0x200) / 4];
        if (offset >= 0x210 && offset < 0x210 + 16 * 16) return table(s.ubo[5], 0x210);
    }
    if (base && offset >= base && offset < base + 16 * 16) return table(s.ssbo[stage], base);
    if (stage == 0 && offset == 0x610) return std::bit_cast<uint32_t>(2047.9375f);
    return 0;
}


namespace {

std::mutex nvn_mu;                 // one replay at a time (submits come from several game threads)
std::atomic<uint64_t> submitted{0};
NvnState state;                    // the queue's state

/* Tegra GPU ticks (19.2 MHz) of now: what counters' timestamps hold */
/* Counter timestamps: within a frame the time runs at RX_NVN_TS_SCALE (default 0.1), from the real time at its start.
   TOTK's dynamic resolution reads the frame's GPU time from them, and the replay's CPU time is not the GPU's (several
   times the GPU's work): after every load it dropped to 1280x720 for ~1.5 s, and in heavy scenes for good. */
bool ts_new_frame = true;
uint64_t gpu_ticks() {
    static const double scale = getenv("RX_NVN_TS_SCALE") ? atof(getenv("RX_NVN_TS_SCALE")) : 0.1;
    const uint64_t real = (uint64_t)hos_now_ns();
    static uint64_t origin_real, origin_out, last_out;
    if (ts_new_frame) {
        ts_new_frame = false;
        origin_real = real;
        origin_out = std::max(real, last_out);
    }
    const uint64_t out = origin_out + (uint64_t)((double)(real - origin_real) * scale);
    last_out = std::max(last_out, out);
    return last_out * 12 / 625;
}

/* a record's payload: variable-length records (bindings of n items) are shorter than their type (the rest: zeros) */
template <class T>
T payload(const uint8_t* a, uint32_t bytes) {
    T v{};
    std::memcpy(&v, a, std::min<size_t>(sizeof v, bytes));
    return v;
}

// ---- copies: on guest memory, as the console's copy engine makes them (textures in their own layout) ----
// A copy reads what guest memory holds: a source a shader wrote on the GPU is synced first; one a render pass drew
// (a render target) is counted (nvn_copy_from_rt) until copies of those run on the GPU.

/* one level of a texture as the copy engine addresses it: sizes in blocks (a compressed format's 4x4 texels) */
struct CopySurface {
    uint64_t base = 0, layer_size = 0, level_size = 0;   // GPU address of the level in the first layer; bytes
    uint32_t bpb = 0, bw = 1, bh = 1;                     // bytes per block, block size in texels
    uint32_t width = 0, height = 0, depth = 1;            // the level in blocks (depth: a 3D texture's slices)
    uint32_t base_layer = 0;
    int target = 0;
    bool linear = false;
    uint32_t stride = 0;                                  // linear: bytes per row
    unsigned tile_h = 0, tile_d = 0;                      // block linear: block height / depth (log2 GOBs)
};

/* a GOB is 64 bytes x 8 rows (512 bytes) swizzled inside; a block is 1 GOB wide, 2^bh GOBs tall and 2^bd deep;
   blocks run left to right, then top to bottom, then front to back (x in bytes, y in rows) */
uint64_t bl_offset(uint32_t x, uint32_t y, uint32_t z, uint32_t width_bytes, uint32_t height, uint32_t bh, uint32_t bd) {
    const uint64_t gobs_x = (width_bytes + 63) >> 6;
    const uint64_t blocks_y = (height + (8u << bh) - 1) >> (3 + bh);
    const uint64_t block = 512ull << (bh + bd);
    const uint64_t gob = ((x & 63) >> 5) << 8 | ((y & 7) >> 1) << 6 | ((x & 31) >> 4) << 5 | (y & 1) << 4 | (x & 15);
    return (uint64_t)(z >> bd) * gobs_x * blocks_y * block + (uint64_t)(y >> (3 + bh)) * gobs_x * block +
           (uint64_t)(x >> 6) * block + ((uint64_t)(z & ((1u << bd) - 1)) << (9 + bh)) +
           ((uint64_t)((y >> 3) & ((1u << bh) - 1)) << 9) + gob;
}

bool copy_surface(const nvn_tex& t, const nvn_view* v, CopySurface& o) {
    int bw = 1, bh = 1;
    const int bpb = nvn_fmt_bpb(t.d.format, &bw, &bh);
    if (bpb <= 0 || !t.gpu) return false;
    const uint32_t level = v && v->set_levels ? v->base_level : 0;
    const bool one_d = t.d.target == NVN_TEXTURE_TARGET_1D || t.d.target == NVN_TEXTURE_TARGET_1D_ARRAY;
    o.target = t.d.target;
    o.bpb = (uint32_t)bpb, o.bw = (uint32_t)bw, o.bh = (uint32_t)bh;
    o.width = (std::max(1u, t.d.width >> level) + o.bw - 1) / o.bw;
    o.height = one_d ? 1 : (std::max(1u, t.d.height >> level) + o.bh - 1) / o.bh;
    o.depth = t.d.target == NVN_TEXTURE_TARGET_3D ? std::max(1u, t.d.depth >> level) : 1;
    o.base = t.gpu + nvn_tex_level_offset(&t.d, level);
    o.layer_size = t.L.layer_size;
    o.base_layer = v && v->set_layers ? v->min_layer : 0;
    o.linear = nvn_tex_is_linear(&t.d) != 0;
    if (o.linear) {
        o.stride = t.d.stride ? t.d.stride : nvn_tex_linear_stride(&t.d);
        o.level_size = (uint64_t)o.stride * o.height;
    } else {
        nvn_tex_level_tiles(&t.d, level, &o.tile_h, &o.tile_d);
        const uint64_t gobs_x = (o.width * o.bpb + 63) >> 6;
        const uint64_t blocks_y = (o.height + (8u << o.tile_h) - 1) >> (3 + o.tile_h);
        const uint64_t blocks_z = (o.depth + (1u << o.tile_d) - 1) >> o.tile_d;
        o.level_size = gobs_x * blocks_y * blocks_z * (512ull << (o.tile_h + o.tile_d));
    }
    return true;
}

/* a copy region in blocks: x, y, w, h; z, d (3D slices, or array layers when `layers`) */
struct CopyBox {
    uint32_t x = 0, y = 0, w = 0, h = 0, z = 0, d = 0;
    bool layers = false;
};
CopyBox copy_box(const CopySurface& s, const NVNcopyRegion& r) {
    CopyBox b;
    const auto clampi = [](int v) { return (uint32_t)std::max(v, 0); };
    b.x = clampi(r.xoffset) / s.bw;
    b.w = (clampi(r.width) + s.bw - 1) / s.bw;
    b.y = clampi(r.yoffset) / s.bh;
    b.h = (clampi(r.height) + s.bh - 1) / s.bh;
    b.z = clampi(r.zoffset);
    b.d = std::max(1u, clampi(r.depth));
    switch (s.target) {
    case NVN_TEXTURE_TARGET_1D: b.y = 0, b.h = 1, b.z = 0, b.d = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY:
        b.z = clampi(r.yoffset), b.d = std::max(1u, clampi(r.height)), b.layers = true;
        b.y = 0, b.h = 1;
        break;
    case NVN_TEXTURE_TARGET_3D: break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: case NVN_TEXTURE_TARGET_CUBEMAP:
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY:
        b.layers = true;
        break;
    default: b.z = 0, b.d = 1; break;
    }
    // (clipped to the level)
    b.w = b.x < s.width ? std::min(b.w, s.width - b.x) : 0;
    b.h = b.y < s.height ? std::min(b.h, s.height - b.y) : 0;
    if (!b.layers) b.d = b.z < s.depth ? std::min(b.d, s.depth - b.z) : 0;
    return b;
}

/* host memory of GPU addresses for a run of copies: a 64 KiB page at a time (virtual pools map pages that size) */
struct HostPages {
    uint64_t page = ~0ull;
    uint8_t* host = nullptr;
    uint8_t* at(uint64_t gpu) {
        if (gpu >> 16 != page) {
            page = gpu >> 16;
            host = nvn_gpu_host(page << 16);
            if (host && nvn_gpu_contiguous(page << 16, 0x10000) < 0x10000) host = nullptr;
        }
        return host ? host + (gpu & 0xFFFF) : nullptr;
    }
};

/* one row of blocks of the box (row `row` of slice / layer k): fn(GPU address, byte offset in the row, bytes) for
   each run contiguous in the texture's layout (16 bytes inside a GOB; a linear texture's whole row) */
template <class Fn>
void surface_row(const CopySurface& s, const CopyBox& b, uint32_t row, uint32_t k, Fn&& fn) {
    const uint64_t layer = b.layers ? (uint64_t)(s.base_layer + b.z + k) : s.base_layer;
    const uint32_t z = b.layers ? 0 : b.z + k;
    const uint64_t base = s.base + layer * s.layer_size;
    const uint32_t x0 = b.x * s.bpb, n = b.w * s.bpb, y = b.y + row;
    if (s.linear) {
        fn(base + (uint64_t)y * s.stride + x0, 0u, n);
        return;
    }
    for (uint32_t x = 0; x < n;) {
        const uint32_t bx = x0 + x, run = std::min(16 - (bx & 15), n - x);
        fn(base + bl_offset(bx, y, z, s.width * s.bpb, s.height, s.tile_h, s.tile_d), x, run);
        x += run;
    }
}

/* the source's memory holds what the GPU wrote (a shader's stores reach guest memory after their batch ran) */
void copy_source_synced(uint64_t gpu, uint64_t size) {
    if (size && rx_gpu_writes_pending(gpu, size)) rx_sync_gpu_writes(gpu, size);
}
/* a copy out of a render target the GPU drew: its pixels into guest memory first, where the copy reads them (they
   are only in its texture). TOTK renders each fused weapon's inventory icon into a 256x144 target, copies it to a
   buffer, crops it on the CPU and uploads it with WriteTexels: read from guest memory, the icons were black. (The
   threaded web build writes targets back after the batch: not in time for this copy.) */
void sync_rendered_source(const CopySurface& s) {
    if (RX_WEB_MT) return;
    const uint64_t guest = nvn_gpu_guest(s.base);
    Image* im = guest ? image_at(guest) : nullptr;
    if (!im || !im->rendered) return;
    queue_target_writeback(im);
    flush_batch();
    writeback_targets();
    count("nvn_copy_rt_synced");
}
void copy_note_target(const CopySurface& s, bool to) {
    const uint64_t guest = nvn_gpu_guest(s.base);
    if (Image* im = guest ? image_at(guest) : nullptr; im && im->rendered) {
        const char* what = to ? "nvn_copy_to_rt" : "nvn_copy_from_rt";
        count(what);
        log_limited(what, 4, "rx: NVN copy %s render target %ux%u at %#llx: through guest memory", to ? "to" : "from",
                    im->width, im->height, (unsigned long long)s.base);
    }
}
uint64_t copy_from(const CopySurface& s, const CopyBox& b) {
    return s.base + (uint64_t)(s.base_layer + (b.layers ? b.z : 0)) * s.layer_size;
}
uint64_t copy_span(const CopySurface& s, const CopyBox& b) {
    return b.layers ? (uint64_t)(b.d - 1) * s.layer_size + s.level_size : s.level_size;
}

/* buffer <-> texture: the buffer's rows `row_stride` apart (0: packed), its slices / layers row_stride * rows apart */
void nvn_copy_buffer_texture(const nvn_p_copy_bt& c, int64_t row_stride, bool to_texture) {
    CopySurface s;
    if (!copy_surface(*c.tex, c.has_view ? &c.view : nullptr, s)) {
        log_limited("nvn_copy_fmt", 8, "rx: NVN copy of texture format %d: no layout", c.tex->d.format);
        return;
    }
    const CopyBox b = copy_box(s, c.r);
    if (!b.w || !b.h || !b.d) return;
    const uint64_t pitch = row_stride > 0 ? (uint64_t)row_stride : (uint64_t)b.w * s.bpb;
    const uint64_t image = pitch * b.h, total = image * b.d;
    if (to_texture) copy_source_synced(c.buf, total);
    else copy_source_synced(copy_from(s, b), copy_span(s, b)), sync_rendered_source(s);
    copy_note_target(s, to_texture);
    static std::vector<uint8_t> row;
    row.resize((size_t)b.w * s.bpb);
    HostPages tex;
    for (uint32_t k = 0; k < b.d; k++)
        for (uint32_t y = 0; y < b.h; y++) {
            const uint64_t at = c.buf + k * image + y * pitch;
            if (to_texture && !nvn_gpu_read(at, row.data(), row.size())) continue;
            surface_row(s, b, y, k, [&](uint64_t gpu, uint32_t off, uint32_t n) {
                uint8_t* h = tex.at(gpu);
                if (h && (gpu & 0xFFFF) + n <= 0x10000) {
                    if (to_texture) std::memcpy(h, row.data() + off, n);
                    else std::memcpy(row.data() + off, h, n);
                } else if (to_texture) {
                    nvn_gpu_write(gpu, row.data() + off, n);
                } else {
                    nvn_gpu_read(gpu, row.data() + off, n);
                }
            });
            if (!to_texture) nvn_gpu_write(at, row.data(), row.size());
        }
    if (to_texture) rx_memory_written(copy_from(s, b), copy_span(s, b));
    else rx_memory_written(c.buf, total);
    count(to_texture ? "nvn_copies_b2t" : "nvn_copies_t2b");
    if (static const bool trace = getenv("RX_NVN_TEXELS_LOG") != nullptr; trace) {
        const uint64_t guest = nvn_gpu_guest(s.base);
        const Image* im = guest ? image_at(guest) : nullptr;
        uint64_t nz = 0;
        std::vector<uint8_t> all((size_t)std::min<uint64_t>(total, 64u << 20));
        if (nvn_gpu_read(c.buf, all.data(), all.size()))
            for (uint8_t v : all) nz += v != 0;
        log_limited("nvn_texels", 100000, "rx: NVN copy %s %08llx %ux%u format %d (%s %ux%u) %s buffer %llx at %d,%d %dx%d "
                    "(%llu nonzero)", to_texture ? "to texture" : "from texture", (unsigned long long)guest,
                    c.tex->d.width, c.tex->d.height, c.tex->d.format, im && im->rendered ? "a render target" : "memory",
                    im ? im->width : 0, im ? im->height : 0, to_texture ? "<-" : "->", (unsigned long long)c.buf,
                    c.r.xoffset, c.r.yoffset, c.r.width, c.r.height, (unsigned long long)nz);
    }
}

/* texture -> texture: block for block; a scaled or mirrored copy takes the nearest source block */
void nvn_copy_texture_texture(const nvn_p_copy_tt& c) {
    CopySurface ss, ds;
    if (!copy_surface(*c.src, c.has_sv ? &c.sv : nullptr, ss) || !copy_surface(*c.dst, c.has_dv ? &c.dv : nullptr, ds))
        return;
    if (ss.bpb != ds.bpb || ss.bw != ds.bw || ss.bh != ds.bh) {
        log_limited("nvn_copy_tt_fmt", 8, "rx: NVN texture copy from format %d to %d: blocks differ, not copied",
                    c.src->d.format, c.dst->d.format);
        count("nvn_copies_skipped");
        return;
    }
    const CopyBox sb = copy_box(ss, c.sr), db = copy_box(ds, c.dr);
    if (!sb.w || !sb.h || !sb.d || !db.w || !db.h || !db.d) return;
    copy_source_synced(copy_from(ss, sb), copy_span(ss, sb));
    sync_rendered_source(ss);
    copy_note_target(ss, false);
    copy_note_target(ds, true);
    const bool mirror_x = c.flags & NVN_COPY_FLAGS_MIRROR_X_BIT;
    const bool scaled = sb.w != db.w || sb.h != db.h || sb.d != db.d;
    if (scaled || (c.flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT | NVN_COPY_FLAGS_MIRROR_Z_BIT)))
        count("nvn_copies_scaled");
    static std::vector<uint8_t> src_row, row;
    src_row.resize((size_t)sb.w * ss.bpb);
    row.resize((size_t)db.w * ds.bpb);
    HostPages sp, dp;
    for (uint32_t k = 0; k < db.d; k++)
        for (uint32_t y = 0; y < db.h; y++) {
            uint32_t sk = (uint32_t)((uint64_t)k * sb.d / db.d), sy = (uint32_t)((uint64_t)y * sb.h / db.h);
            if (c.flags & NVN_COPY_FLAGS_MIRROR_Z_BIT) sk = sb.d - 1 - sk;
            if (c.flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) sy = sb.h - 1 - sy;
            surface_row(ss, sb, sy, sk, [&](uint64_t gpu, uint32_t off, uint32_t n) {
                uint8_t* h = sp.at(gpu);
                if (h && (gpu & 0xFFFF) + n <= 0x10000) std::memcpy(src_row.data() + off, h, n);
                else nvn_gpu_read(gpu, src_row.data() + off, n);
            });
            if (!scaled && !mirror_x) {
                row = src_row;
            } else {
                for (uint32_t x = 0; x < db.w; x++) {
                    uint32_t sx = (uint32_t)((uint64_t)x * sb.w / db.w);
                    if (mirror_x) sx = sb.w - 1 - sx;
                    std::memcpy(row.data() + (size_t)x * ds.bpb, src_row.data() + (size_t)sx * ss.bpb, ds.bpb);
                }
            }
            surface_row(ds, db, y, k, [&](uint64_t gpu, uint32_t off, uint32_t n) {
                uint8_t* h = dp.at(gpu);
                if (h && (gpu & 0xFFFF) + n <= 0x10000) std::memcpy(h, row.data() + off, n);
                else nvn_gpu_write(gpu, row.data() + off, n);
            });
        }
    rx_memory_written(copy_from(ds, db), copy_span(ds, db));
    count("nvn_copies_t2t");
}

// RX_NVN_COND_RECHECK=1 (debugging): the counters render conditions read are read again at the next present; the
// ones that changed in between are logged (the CPU wrote them after the replay read them)
struct CondRead { uint64_t addr, x, y; int64_t frame; };
std::vector<CondRead> cond_reads;
std::mutex cond_mu;
void cond_recheck_note(uint64_t addr, uint64_t x, uint64_t y) {
    std::lock_guard lk(cond_mu);
    if (cond_reads.size() < 4096) cond_reads.push_back({addr, x, y, frame_number});
}
void cond_recheck_present() {
    std::vector<CondRead> v;
    {
        std::lock_guard lk(cond_mu);
        v.swap(cond_reads);
    }
    static int logged = 0;
    for (const CondRead& c : v) {
        uint64_t x = 0, y = 0;
        nvn_gpu_read(c.addr, &x, 8);
        nvn_gpu_read(c.addr + 16, &y, 8);
        if (c.x == c.y && x != y && logged++ < 300)   // (equal at the replay, different later: written after it)
            hos_log("rx: render condition at %llx read %llx / %llx at the replay, %llx / %llx at the present (frame %lld)",
                    (unsigned long long)c.addr, (unsigned long long)c.x, (unsigned long long)c.y, (unsigned long long)x,
                    (unsigned long long)y, (long long)c.frame);
    }
}

[[gnu::noinline]] void replay(NvnState& s, const uint8_t* p) {   // (its own frame in crash reports)
    for (;;) {
        nvn_op_hdr h;
        std::memcpy(&h, p, sizeof h);
        const uint8_t* a = p + 8;
        const uint32_t avail = h.size - 8;
        switch (h.op) {
        case NOP_END: return;
        case NOP_BEGIN: case NOP_JUMP: case NOP_CALL: break;
        // ---- what the CPU sees ----
        case NOP_SIGNAL_EVENT: {
            const auto e = payload<nvn_p_signal>(a, avail);
            uint32_t v = 0;
            if (e.mode == NVN_EVENT_SIGNAL_MODE_ADD) nvn_gpu_read(e.gpu, &v, 4);
            v = e.mode == NVN_EVENT_SIGNAL_MODE_ADD ? v + e.value : e.value;
            nvn_gpu_write(e.gpu, &v, 4);
            if (frame_dump_active()) {
                char b[96];
                std::snprintf(b, sizeof b, "nvn event %llx = %u", (unsigned long long)e.gpu, v);
                dump_draw(b);
            }
            break;
        }
        case NOP_REPORT_COUNTER: {
            const auto c = payload<nvn_p_counter>(a, avail);
            // occlusion counts: what was drawn since the last reset (NvnState::samples_passed). Games subtract two
            // reports around a draw: a constant made every object hidden (Ultrahand never found its target).
            // RX_OCCLUSION_CONSTANT=1: the old constant
            static const bool constant = getenv("RX_OCCLUSION_CONSTANT") != nullptr;
            const uint64_t samples = constant ? 0x10000ull : s.samples_passed;
            const uint64_t rec[2] = {c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? samples : 0, gpu_ticks()};
            nvn_gpu_write(c.addr, rec, sizeof rec);
            if (frame_dump_active()) {
                char b[96];
                std::snprintf(b, sizeof b, "nvn counter %llu at %llx", (unsigned long long)c.type, (unsigned long long)c.addr);
                dump_draw(b);
            }
            break;
        }
        case NOP_RESET_COUNTER: {
            const auto c = payload<nvn_p_counter>(a, avail);
            if (c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED) s.samples_passed = 0;
            break;
        }
        case NOP_COPY_B2B: {
            const auto c = payload<nvn_p_copy_bb>(a, avail);
            if (frame_dump_active()) {
                char b[160];
                std::snprintf(b, sizeof b, "nvn copy buffer %llx -> %llx, %llu bytes", (unsigned long long)c.src,
                              (unsigned long long)c.dst, (unsigned long long)c.size);
                dump_draw(b);
            }
            for (uint64_t done = 0; done < c.size;) {
                uint8_t buf[65536];
                const uint64_t k = std::min<uint64_t>(sizeof buf, c.size - done);
                if (!nvn_gpu_read(c.src + done, buf, k) || !nvn_gpu_write(c.dst + done, buf, k)) break;
                done += k;
            }
            rx_memory_written(c.dst, c.size);
            break;
        }
        case NOP_UPDATE_UB: {
            const auto u = payload<nvn_p_update_ub>(a, avail);
            nvn_gpu_write(u.dst, a + sizeof u, u.size);
            rx_memory_written(u.dst, u.size);
            break;
        }
        // ---- state ----
        case NOP_SET_RT: s.rt = payload<nvn_p_set_rt>(a, avail); s.at_dirty = true; break;
        case NOP_VIEWPORTS: {
            const auto v = payload<nvn_p_viewports>(a, avail);
            for (int i = 0; i < v.n && v.first + i < 16; i++) std::memcpy(s.viewport[v.first + i], v.v[i], 16);
            break;
        }
        case NOP_SCISSOR: {
            const auto c = payload<nvn_p_scissor>(a, avail);
            s.scissor[0] = c.x, s.scissor[1] = c.y, s.scissor[2] = c.w, s.scissor[3] = c.h;
            break;
        }
        case NOP_DEPTH_RANGE: {
            const auto d = payload<nvn_p_depth_range>(a, avail);
            s.depth_near = d.n, s.depth_far = d.f;
            break;
        }
        case NOP_BIND_PROGRAM: {
            const auto b = payload<nvn_p_bind_program>(a, avail);
            if (b.prog) s.prog = *b.prog;
            s.has_prog = b.prog != nullptr && b.prog->nstage > 0;
            s.prog_epoch++;
            break;
        }
        case NOP_BIND_STATE: {
            const auto b = payload<nvn_p_bind_state>(a, avail);
            switch (b.kind) {
            case 0: {
                nvn_blend_st st;
                std::memcpy(&st, b.bytes, sizeof st);
                s.blend[st.target & 7] = st;
                break;
            }
            case 1: std::memcpy(&s.color, b.bytes, sizeof s.color); break;
            case 2: std::memcpy(&s.chmask, b.bytes, sizeof s.chmask); break;
            case 3: std::memcpy(&s.ms, b.bytes, sizeof s.ms); break;
            case 4: std::memcpy(&s.poly, b.bytes, sizeof s.poly); break;
            case 5: std::memcpy(&s.ds, b.bytes, sizeof s.ds); break;
            default: break;
            }
            break;
        }
        case NOP_BIND_VATTR: {
            const auto v = payload<nvn_p_bind_vattr>(a, avail);
            decltype(s.vattr) old;
            std::memcpy(old, s.vattr, sizeof old);
            std::memset(s.vattr, 0, sizeof s.vattr);
            for (int i = 0; i < v.n && i < 16; i++) s.vattr[i] = v.st[i];
            const int n = std::min(v.n, 16);
            if (n != s.nvattr || std::memcmp(old, s.vattr, sizeof old)) {
                s.nvattr = n;
                s.prog_epoch++;
                s.vtx_hash = nvn_vertex_hash(s);
            }
            break;
        }
        case NOP_BIND_VSTREAM: {
            const auto v = payload<nvn_p_bind_vstream>(a, avail);
            bool changed = false;
            for (int i = 0; i < v.n && i < 16; i++) {
                changed |= std::memcmp(&s.vstream[i], &v.st[i], sizeof v.st[i]) != 0;
                s.vstream[i] = v.st[i];
            }
            if (changed) s.prog_epoch++, s.vtx_hash = nvn_vertex_hash(s);
            break;
        }
        case NOP_BIND_VBUF: {
            const auto v = payload<nvn_p_bind_vbuf>(a, avail);
            for (int i = 0; i < v.n && v.first + i < 16; i++) s.vbuf[v.first + i] = v.r[i];
            break;
        }
        case NOP_BIND_BUFS: {
            const auto b = payload<nvn_p_bind_bufs>(a, avail);
            for (int i = 0; i < b.n && i < 16; i++) {
                const int index = b.first + i;
                if (b.stage < 0 || b.stage >= 6 || index < 0 || index >= 16) continue;
                nvn_p_bind_buf& u = (b.ssbo ? s.ssbo : s.ubo)[b.stage][index];
                u.stage = b.stage, u.index = index, u.addr = b.r[i].address, u.size = b.r[i].size;
            }
            break;
        }
        case NOP_BIND_UBO: case NOP_BIND_SSBO: {
            const auto b = payload<nvn_p_bind_buf>(a, avail);
            if (b.stage >= 0 && b.stage < 6 && b.index >= 0 && b.index < 16)
                (h.op == NOP_BIND_UBO ? s.ubo : s.ssbo)[b.stage][b.index] = b;
            break;
        }
        case NOP_BIND_TEX: case NOP_BIND_SAMP: case NOP_BIND_IMG: {
            nvn_p_bind_handles b;
            std::memset(&b, 0, sizeof b);
            std::memcpy(&b, a, std::min<size_t>(sizeof b, h.size - 8));
            if (b.stage < 0 || b.stage > 5 || b.first < 0) break;
            uint64_t* to = h.op == NOP_BIND_TEX ? s.tex[b.stage] : h.op == NOP_BIND_SAMP ? s.samp[b.stage] : s.img[b.stage];
            const int lim = h.op == NOP_BIND_IMG ? 32 : 128;
            bool changed = false;
            for (int k = 0; k < b.n && b.first + k < lim; k++) {
                changed |= to[b.first + k] != b.h[k];
                to[b.first + k] = b.h[k];
            }
            if (changed) s.prog_epoch++;
            break;
        }
        case NOP_STENCIL: {
            const auto st = payload<nvn_p_stencil>(a, avail);
            for (int f = 0; f < 2; f++) {
                if (!(st.face & (1 << f))) continue;
                if (st.which == 0) s.stencil_value_mask[f] = (uint32_t)st.value_mask;
                else if (st.which == 1) s.stencil_mask[f] = (uint32_t)st.mask;
                else s.stencil_ref[f] = (uint32_t)st.ref;
            }
            break;
        }
        case NOP_POLY_OFFSET: {
            const auto o = payload<nvn_p_poly_offset>(a, avail);
            s.poly_offset[0] = o.factor, s.poly_offset[1] = o.units, s.poly_offset[2] = o.clamp;
            break;
        }
        case NOP_BLEND_COLOR: std::memcpy(s.blend_color, payload<nvn_p_blend_color>(a, avail).rgba, 16); break;
        case NOP_PRIMITIVE_RESTART: {
            const auto r = payload<nvn_p_restart>(a, avail);
            s.restart = r.enable != 0, s.restart_index = (uint32_t)r.index;
            break;
        }
        case NOP_DEPTH_CLAMP: s.depth_clamp = payload<nvn_p_flag>(a, avail).value != 0; break;
        case NOP_RASTER_DISCARD: s.raster_discard = payload<nvn_p_flag>(a, avail).value != 0; break;
        case NOP_RENDER_ENABLE:
            s.render_enabled = payload<nvn_p_flag>(a, avail).value != 0;
            if (frame_dump_active()) dump_draw(s.render_enabled ? "nvn render on" : "nvn render off");
            break;
        case NOP_RENDER_COND: {
            const auto r = payload<nvn_p_render_cond>(a, avail);
            uint64_t x = 0, y = 0;
            nvn_gpu_read(r.addr, &x, 8);
            nvn_gpu_read(r.addr + 16, &y, 8);
            s.render_enabled = r.mode == 0 ? x == y : x != y;
            if (static const bool recheck = getenv("RX_NVN_COND_RECHECK") != nullptr; recheck) cond_recheck_note(r.addr, x, y);
            if (frame_dump_active()) {
                char b[128];
                std::snprintf(b, sizeof b, "nvn render if %s at %llx: %llx / %llx: %s", r.mode ? "not equal" : "equal",
                              (unsigned long long)r.addr, (unsigned long long)x, (unsigned long long)y,
                              s.render_enabled ? "on" : "off");
                dump_draw(b);
            }
            break;
        }
        case NOP_SET_TEX_POOL: case NOP_SET_SAMP_POOL: break;
        case NOP_COPY_STRIDE: s.copy_stride = payload<nvn_p_copy_stride>(a, avail).stride; break;
        case NOP_BARRIER: {   // (what was drawn is visible to texture reads after it: feedback snapshots are taken again)
            int32_t flags = 0;
            std::memcpy(&flags, a, std::min<uint32_t>(4, avail));
            // The pass goes on: what a barrier makes visible is only read through a feedback snapshot (taken again
            // after it, which ends the pass then: snapshot_for_feedback) or storage the pass wrote (draw_storage ends
            // it first). Ending it at every barrier was ~97 pass breaks a frame at the castle, each one a render pass
            // begun again in wgpu and the driver and its attachments stored and loaded. RX_BARRIER_ENDS_PASS=1: old.
            if (static const bool ends = env_flag("RX_BARRIER_ENDS_PASS"); ends) end_render_pass("pe_nvn_barrier");
            texture_barrier(0xB000, (uint32_t)flags);
            break;
        }
        // ---- work ----
        case NOP_CLEAR_COLOR: {
            if (!s.render_enabled) {
                if (frame_dump_active()) dump_draw("nvn clear skipped (render off)");
                break;
            }
            const auto c = payload<nvn_p_clear_color>(a, avail);
            const Attachments at = nvn_attachments(s);
            if (frame_dump_active() && c.index >= 0 && c.index < 8 && at.color[c.index]) {
                char b[128];
                std::snprintf(b, sizeof b, "nvn clear c%d %llx to %g %g %g %g mask %x", c.index,
                              (unsigned long long)at.color[c.index]->guest, c.rgba[0], c.rgba[1], c.rgba[2], c.rgba[3], c.mask);
                dump_draw(b);
            }
            if (c.index >= 0 && c.index < 8) nvn_clear_color(at.color[c.index], c.rgba, (uint32_t)c.mask, s.scissor);
            break;
        }
        case NOP_CLEAR_DS: {
            if (!s.render_enabled) break;
            const auto c = payload<nvn_p_clear_ds>(a, avail);
            const Attachments at = nvn_attachments(s);
            nvn_clear_depth_stencil(at.depth, c.depth_mask != 0, c.depth, c.stencil_mask != 0, (uint32_t)c.stencil,
                                    (uint32_t)c.stencil_mask, s.scissor);
            break;
        }
        case NOP_CLEAR_TEX: {
            if (!s.render_enabled) break;
            const auto c = payload<nvn_p_clear_tex>(a, avail);
            Image* im = nvn_target(c.tex, c.has_view ? &c.view : nullptr);
            if (frame_dump_active()) {
                char b[200];
                std::snprintf(b, sizeof b, "nvn clear texture %08llx %ux%u at %d,%d %dx%d to %g %g %g %g mask %x%s",
                              c.tex ? (unsigned long long)nvn_gpu_guest(c.tex->gpu) : 0ull, c.tex ? c.tex->d.width : 0,
                              c.tex ? c.tex->d.height : 0, c.r.xoffset, c.r.yoffset, c.r.width, c.r.height, c.c.f[0],
                              c.c.f[1], c.c.f[2], c.c.f[3], (unsigned)c.mask, im ? "" : " (no target)");
                dump_draw(b);
            }
            if (!im) break;
            const int32_t rect[4] = {c.r.xoffset, c.r.yoffset, c.r.width, c.r.height};
            if (is_depth_format(im->format)) nvn_clear_depth_stencil(im, true, c.c.f[0], false, 0, 0, rect);
            else nvn_clear_color(im, c.c.f, (uint32_t)c.mask, rect);
            break;
        }
        case NOP_DRAW: {
            if (!s.render_enabled || s.raster_discard) {
                if (frame_dump_active()) dump_draw(s.raster_discard ? "nvn draw skipped (raster discard)" : "nvn draw skipped (render off)");
                break;
            }
            const auto d = payload<nvn_p_draw>(a, avail);
            NvnDraw w;
            w.prim = d.prim, w.first = (uint32_t)d.first, w.count = (uint32_t)d.count;
            w.base_instance = (uint32_t)d.base_instance, w.instances = (uint32_t)std::max(1, d.instances);
            s.samples_passed += (uint64_t)std::max(1u, w.count) * w.instances * 16;   // (an estimate: some passed)
            nvn_draw(s, w);
            break;
        }
        case NOP_DRAW_IDX: {
            if (!s.render_enabled || s.raster_discard) {
                if (frame_dump_active()) dump_draw(s.raster_discard ? "nvn draw skipped (raster discard)" : "nvn draw skipped (render off)");
                break;
            }
            const auto d = payload<nvn_p_draw_idx>(a, avail);
            NvnDraw w;
            w.indexed = true;
            w.prim = d.prim, w.index_type = d.itype, w.count = (uint32_t)d.count, w.index_gpu = d.ibuf;
            w.base_vertex = d.base_vertex, w.base_instance = (uint32_t)d.base_instance;
            w.instances = (uint32_t)std::max(1, d.instances);
            s.samples_passed += (uint64_t)std::max(1u, w.count) * w.instances * 16;
            nvn_draw(s, w);
            break;
        }
        case NOP_DISPATCH: {
            const auto d = payload<nvn_p_dispatch>(a, avail);
            if (frame_dump_active()) {
                char b[96];
                std::snprintf(b, sizeof b, "nvn dispatch %d x %d x %d", d.x, d.y, d.z);
                dump_draw(b);
            }
            nvn_dispatch(s, (uint32_t)d.x, (uint32_t)d.y, (uint32_t)d.z);
            break;
        }
        case NOP_COPY_B2T: case NOP_COPY_T2B: {
            const auto c = payload<nvn_p_copy_bt>(a, avail);
            const bool to_texture = h.op == NOP_COPY_B2T;
            if (frame_dump_active()) {
                char b[200];
                std::snprintf(b, sizeof b, "nvn copy %s buffer %llx %s texture %08llx %ux%u at %d,%d,%d %dx%dx%d",
                              to_texture ? "from" : "to", (unsigned long long)c.buf, to_texture ? "->" : "<-",
                              c.tex ? (unsigned long long)nvn_gpu_guest(c.tex->gpu) : 0ull, c.tex ? c.tex->d.width : 0,
                              c.tex ? c.tex->d.height : 0, c.r.xoffset, c.r.yoffset, c.r.zoffset, c.r.width, c.r.height,
                              c.r.depth);
                dump_draw(b);
            }
            nvn_copy_buffer_texture(c, s.copy_stride, to_texture);
            break;
        }
        case NOP_COPY_T2T: {
            const auto c = payload<nvn_p_copy_tt>(a, avail);
            if (frame_dump_active()) {
                char b[220];
                std::snprintf(b, sizeof b, "nvn copy texture %08llx %ux%u (%d,%d %dx%d) -> %08llx %ux%u (%d,%d %dx%d) flags %x",
                              c.src ? (unsigned long long)nvn_gpu_guest(c.src->gpu) : 0ull, c.src ? c.src->d.width : 0,
                              c.src ? c.src->d.height : 0, c.sr.xoffset, c.sr.yoffset, c.sr.width, c.sr.height,
                              c.dst ? (unsigned long long)nvn_gpu_guest(c.dst->gpu) : 0ull, c.dst ? c.dst->d.width : 0,
                              c.dst ? c.dst->d.height : 0, c.dr.xoffset, c.dr.yoffset, c.dr.width, c.dr.height,
                              (unsigned)c.flags);
                dump_draw(b);
            }
            nvn_copy_texture_texture(c);
            break;
        }
        default:
            log_limited("nvn_op", 16, "rx: NVN record %u not replayed", h.op);
            break;
        }
        p += h.size;
    }
}

}  // namespace
}  // namespace rx

using namespace rx;

extern "C" {

// ---- submits and presents: replayed in order on the replay thread ----
// A submit used to replay on the game thread that called it (TOTK's frame thread: ~14 ms a frame in the open world,
// the game waiting for its own draws). The submits and presents now queue for one "NVN replay" thread, as the
// console's GPU front end takes them, and a fence is reached when the replay passed it: the game runs up to a frame
// ahead of the replay. The game's command recordings stay alive until the replay passed every submit made before
// they were released (hos/nvn_native.c). RX_NVN_SYNC=1: replayed at the call, as before. On the web the replay thread
// is also the render thread (WebGPU objects belong to the worker that made them): it takes the page's canvas, starts
// the renderer and waits for jobs in its event loop (rx_web_idle: WebGPU's callbacks run there).
}  // extern "C"
namespace rx {
namespace {
struct NvnJob {
    std::vector<uint64_t> handles;
    uint64_t scan = 0;                 // (a submit: the write scan its push started; 0: scanned when replayed)
    bool present = false;
    uint64_t present_guest = 0;
    uint32_t present_w = 0, present_h = 0;
};
std::mutex job_mu;
std::condition_variable job_cv, job_done_cv;
std::deque<NvnJob> jobs;
uint64_t jobs_pushed = 0;              // (under job_mu)
int presents_queued = 0;               // (under job_mu)
std::atomic<uint64_t> jobs_replayed{0};

/* RX_NVN_FRAMES_AHEAD=<n>: presents the game may queue ahead of the replay (default 1) */
int max_presents_ahead() {
    static const int n = getenv("RX_NVN_FRAMES_AHEAD") ? std::max(1, atoi(getenv("RX_NVN_FRAMES_AHEAD"))) : 1;
    return n;
}

bool replay_sync() {
    static const bool sync = !RX_WEB && getenv("RX_NVN_SYNC") != nullptr;
    return sync;
}

/* The CPU's writes before a submit are collected on the "NVN write scan" thread, started when the game submits:
   GetWriteWatch over the ~2 GB the GPU can see costs ~2 ms, which the replay thread spent at the start of every
   submit (11% of its time while running through the castle). A submit's replay waits for the scan its push started;
   one scan may answer several pushes (its pages go to the first replay that takes them: invalidating early is
   harmless). RX_NVN_SCAN_INLINE=1: scanned on the replay thread, as before. */
std::mutex scan_mu;
std::condition_variable scan_cv, scan_done_cv;
uint64_t scan_requested = 0, scan_done = 0;   // (under scan_mu)
std::vector<uint32_t> scan_pages;             // collected, not taken yet (under scan_mu)
bool scan_failed = false;                     // (under scan_mu: no write watch; every replay scans for itself)

void scan_main() {
    hos_host_thread_register_self("NVN write scan");
    hos_thread_boost(1);
    for (;;) {
        uint64_t target;
        {
            std::unique_lock lk(scan_mu);
            scan_cv.wait(lk, [] { return scan_requested > scan_done; });
            target = scan_requested;
        }
        int64_t n = -1;
        uint32_t* pages = rx_collect_written(&n);
        {
            std::lock_guard lk(scan_mu);
            if (n < 0) scan_failed = true;
            else scan_pages.insert(scan_pages.end(), pages, pages + n);
            scan_done = target;
        }
        free(pages);
        scan_done_cv.notify_all();
    }
}

uint64_t request_scan() {
    static const bool inline_scan = getenv("RX_NVN_SCAN_INLINE") != nullptr;
    if (inline_scan) return 0;
    static std::once_flag started;
    std::call_once(started, [] { std::thread(scan_main).detach(); });
    uint64_t serial;
    {
        std::lock_guard lk(scan_mu);
        serial = ++scan_requested;
    }
    scan_cv.notify_one();
    return serial;
}

void run_submit(const uint64_t* handles, size_t n, uint64_t scan = 0) {
    std::lock_guard lk(nvn_mu);
    {   // what the CPU wrote since the last submit: the buffer cache and decoded textures learn of it
        int64_t nw = -1;
        uint32_t* pages = nullptr;
        bool scanned = false;
        if (scan) {
            const int64_t t0 = hos_now_ns();
            std::unique_lock sl(scan_mu);
            scan_done_cv.wait(sl, [&] { return scan_done >= scan; });
            if (!scan_failed) {
                nw = (int64_t)scan_pages.size();
                pages = static_cast<uint32_t*>(malloc((scan_pages.size() + 1) * sizeof(uint32_t)));
                if (nw) std::memcpy(pages, scan_pages.data(), scan_pages.size() * sizeof(uint32_t));
                scan_pages.clear();
                scanned = true;
            }
            sl.unlock();
            count("t_nvn_scan_wait", hos_now_ns() - t0);
        }
        if (!scanned) pages = rx_collect_written(&nw);
        rx_begin_submission_written(pages, nw);
    }
    {
        std::lock_guard rl(mu);
        for (size_t i = 0; i < n; i++) replay(state, reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(handles[i])));
    }
    rx_flush();
    submitted++;
}

void run_present(uint64_t guest, uint32_t w, uint32_t h) {
    ts_new_frame = true;
    if (static const bool recheck = getenv("RX_NVN_COND_RECHECK") != nullptr; recheck) cond_recheck_present();
    std::lock_guard lk(nvn_mu);
    rx_present(guest, w, h);
}

#if RX_WEB
uint32_t job_seq;   // (bumped at each push: the replay thread's rx_web_idle wakes on it)
#endif
std::mutex ready_mu;
std::condition_variable ready_cv;
bool ready = false;   // (the replay thread has started the renderer: under ready_mu)

void replay_main() {
    hos_host_thread_register_self("NVN replay");
    hos_thread_boost(1);
#if RX_WEB
    rx_start(nullptr);   // (the device is made on the thread that makes every WebGPU call)
#endif
    {
        std::lock_guard lk(ready_mu);
        ready = true;
    }
    ready_cv.notify_all();
    for (;;) {
        NvnJob j;
        {
            std::unique_lock lk(job_mu);
#if RX_WEB
            while (jobs.empty()) {
                const uint32_t seen = __atomic_load_n(&job_seq, __ATOMIC_SEQ_CST);
                lk.unlock();
                rx_web_idle(&job_seq, seen);
                lk.lock();
            }
#else
            job_cv.wait(lk, [] { return !jobs.empty(); });
#endif
            j = std::move(jobs.front());
            jobs.pop_front();
        }
        if (j.present) run_present(j.present_guest, j.present_w, j.present_h);
        else run_submit(j.handles.data(), j.handles.size(), j.scan);
        {
            std::lock_guard lk(job_mu);
            if (j.present) presents_queued--;
            jobs_replayed++;
        }
        job_done_cv.notify_all();
    }
}

void* replay_entry(void*) {
    replay_main();
    return nullptr;
}

/* The replay thread, started once (and on the web before the first submit or present: until it has started the
   renderer, rx_active() is false there and the work would be dropped) */
void start_replay() {
    static std::once_flag once;
    std::call_once(once, [] {
#if RX_WEB
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, 1 << 20);
        if (!RX_WEB_MT)   // (the canvas the page shows; the threaded build: the WebGPU thread takes it, rx_web_mt.cpp)
            emscripten_pthread_attr_settransferredcanvases(&a, "#screen");
        pthread_t t;
        pthread_create(&t, &a, replay_entry, nullptr);
        pthread_attr_destroy(&a);
        pthread_detach(t);
#else
        std::thread(replay_main).detach();
#endif
        std::unique_lock lk(ready_mu);
        ready_cv.wait(lk, [] { return ready; });
    });
}

void push_job(NvnJob&& j) {
    start_replay();
    std::unique_lock lk(job_mu);
    // (the game at most one frame and 64 submits ahead of the replay)
    const auto room = [&] { return jobs.size() < 64 && (!j.present || presents_queued < max_presents_ahead()); };
    if (!room()) {
        const int64_t t0 = hos_now_ns();
        job_done_cv.wait(lk, room);
        count(j.present ? "t_nvn_present_wait" : "t_nvn_submit_wait", hos_now_ns() - t0);
    }
    if (j.present) presents_queued++;
    jobs.push_back(std::move(j));
    jobs_pushed++;
    lk.unlock();
    job_cv.notify_one();
#if RX_WEB
    __atomic_add_fetch(&job_seq, 1, __ATOMIC_SEQ_CST);
    emscripten_futex_wake(&job_seq, 1);
#endif
}

/* The threaded web build reads render targets back to guest memory after the replay (rx_core.cpp writeback_targets):
   a fence or a drain also waits for those */
void wait_writebacks() {
    if (!RX_WEB_MT || rx_target_writebacks_pending() <= 0) return;
    const int64_t t0 = hos_now_ns();
    while (rx_target_writebacks_pending() > 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
#if RX_WEB_MT
    mt_note_wait("(fences and drains: writebacks)", hos_now_ns() - t0);
#endif
}

/* every job pushed so far replayed (the game is about to look at what they wrote, or to replace what they read) */
void drain() {
    if (replay_sync()) return;
    {
        std::unique_lock lk(job_mu);
        job_done_cv.wait(lk, [] { return jobs_replayed.load() >= jobs_pushed; });
    }
    wait_writebacks();
}
}  // namespace
}  // namespace rx
extern "C" {

void rx_nvn_submit(const uint64_t* handles, int n) {
    if (RX_WEB) rx::start_replay();
    if (!rx_active() || n <= 0) return;
    if (replay_sync()) {
        run_submit(handles, (size_t)n);
        std::lock_guard lk(job_mu);
        jobs_pushed++;
        jobs_replayed++;
        return;
    }
    NvnJob j;
    j.handles.assign(handles, handles + n);
    j.scan = request_scan();
    push_job(std::move(j));
}
uint64_t rx_nvn_pushed(void) {
    std::lock_guard lk(job_mu);
    return jobs_pushed;
}
uint64_t rx_nvn_replayed(void) { return jobs_replayed.load(); }
/* A fence: the submits made so far. It is reached when they were replayed, as the emulated GPU's syncpoints are when
   its command processor parsed the work: what the CPU changes after that does not reach work already recorded
   (uniforms are copied at the draw, memory uploads go to the queue in order), and what shaders write comes back to
   guest memory by itself. RX_NVN_GPU_FENCES=1: also the GPU ran them (the replay waits for it). */
uint64_t rx_nvn_fence(void) { return rx_nvn_pushed(); }
int rx_nvn_wait(uint64_t fence, uint64_t timeout_ns) {
    // RX_NVN_EARLY_FENCES=1 (experiment): reached at the submit, before the replay
    static const bool early = getenv("RX_NVN_EARLY_FENCES") != nullptr;
    if (early) return 1;
    if (jobs_replayed.load() < fence) {
        if (!timeout_ns) return 0;
        const int64_t t0 = hos_now_ns();
        std::unique_lock lk(job_mu);
        job_done_cv.wait(lk, [&] { return jobs_replayed.load() >= fence; });
        count("nvn_fence_waits");
        count("t_nvn_fence_wait", hos_now_ns() - t0);
#if RX_WEB_MT
        mt_note_wait("(fences: the replay)", hos_now_ns() - t0);
#endif
    }
    // (writebacks: the game's memcpy waits for the ones it reads, rx_wait_writebacks_reading; a fence waited for all
    // of them, ~20 ms each, about the GPU's round trip in the browser. RX_FENCE_WRITEBACKS=1: as before)
    static const bool fence_writebacks = env_flag("RX_FENCE_WRITEBACKS");
    if (RX_WEB_MT && fence_writebacks && rx_target_writebacks_pending() > 0) {
        if (!timeout_ns) return 0;
        wait_writebacks();
    }
    static const bool gpu_fences = getenv("RX_NVN_GPU_FENCES") != nullptr;
    if (gpu_fences) wait_batch(last_batch_serial());
    return 1;
}
void rx_nvn_finish(void) {
    drain();
    rx_wait_idle();
}
void rx_nvn_present(const nvn_tex* t, const int crop[4]) {
    (void)crop;
    if (RX_WEB) rx::start_replay();
    if (!t || !rx_active()) return;
    const uint64_t guest = nvn_gpu_guest(t->gpu);
    if (replay_sync()) {
        run_present(guest, t->d.width, t->d.height);
        return;
    }
    NvnJob j;
    j.present = true;
    j.present_guest = guest;
    j.present_w = t->d.width;
    j.present_h = t->d.height;
    push_job(std::move(j));
}
void rx_nvn_register_texture(int id, const nvn_tex* t, const nvn_view* v) {
    if (t) nvn_register_texture(id, *t, v);
}
void rx_nvn_register_sampler(int id, const nvn_samp_par* p) {
    if (p) nvn_register_sampler(id, *p);
}
void rx_nvn_set_texture_pool_size(int n) { (void)n; }
void rx_nvn_texture_released(const nvn_tex* t) {
    if (t && t->serial) nvn_texture_released(t->serial);
}
void rx_nvn_program_released(const nvn_prog* p) { (void)p; }
/* ReadTexels / WriteTexels: the texture's memory in its own layout, through the copy engine's addressing (a texture
   the GPU rendered is read as guest memory holds it: nvn_copy_from_rt counts those) */
static void texel_copy(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, uint8_t* data,
                       size_t row, size_t img, bool write) {
    nvn_view v{};
    v.set_levels = 1, v.base_level = (uint8_t)level, v.nlevels = 1;
    v.set_layers = 1, v.min_layer = (uint16_t)layer0, v.nlayers = 1;
    CopySurface s;
    if (!copy_surface(*t, &v, s)) {
        log_limited("nvn_texels_fmt", 8, "rx: NVN %sTexels of texture format %d: no layout", write ? "Write" : "Read",
                    t->d.format);
        return;
    }
    const CopyBox b = copy_box(s, *r);
    if (!b.w || !b.h || !b.d) return;
    const size_t pitch = row ? row : (size_t)b.w * s.bpb, image = img ? img : pitch * b.h;
    if (!write) {
        copy_source_synced(copy_from(s, b), copy_span(s, b));
        copy_note_target(s, false);
    }
    HostPages tex;
    for (uint32_t k = 0; k < b.d; k++)
        for (uint32_t y = 0; y < b.h; y++) {
            uint8_t* line = data + k * image + y * pitch;
            surface_row(s, b, y, k, [&](uint64_t gpu, uint32_t off, uint32_t n) {
                uint8_t* h = tex.at(gpu);
                if (h && (gpu & 0xFFFF) + n <= 0x10000) {
                    if (write) std::memcpy(h, line + off, n), rx_mark_written(h, n);
                    else std::memcpy(line + off, h, n);
                } else if (write) {
                    nvn_gpu_write(gpu, line + off, n);
                } else {
                    nvn_gpu_read(gpu, line + off, n);
                }
            });
        }
    // (a CPU write: the write watch reports it at the next submit)
    count(write ? "nvn_write_texels" : "nvn_read_texels");
    if (write) note_texels_written(nvn_gpu_guest(t->gpu));   // (RX_DUMP_TEX=texels: the last ones are dumped)
    if (static const bool trace = getenv("RX_NVN_TEXELS_LOG") != nullptr; trace)
        log_limited("nvn_texels", 100000, "rx: NVN %sTexels %08llx %ux%u format %d level %u layer %u at %d,%d,%d %dx%dx%d",
                    write ? "Write" : "Read", (unsigned long long)nvn_gpu_guest(t->gpu), t->d.width, t->d.height,
                    t->d.format, level, layer0, r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth);
}
void rx_nvn_read_texels(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, uint8_t* dst,
                        size_t row, size_t img) {
    drain();   // (what the submits before it wrote)
    std::lock_guard lk(nvn_mu);
    texel_copy(t, level, layer0, r, dst, row, img, false);
}
void rx_nvn_write_texels(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, const uint8_t* src,
                         size_t row, size_t img) {
    drain();   // (after the submits before it read the texture)
    std::lock_guard lk(nvn_mu);
    texel_copy(t, level, layer0, r, const_cast<uint8_t*>(src), row, img, true);
}
/* the CPU is about to read memory a shader may have written on the GPU (InvalidateMappedRange): nothing to do here.
   TOTK invalidates ranges every frame; a GPU sync at each (as rx_sync_gpu_writes makes) halved the frame rate, and the
   emulated path never synced there either: shader writes come back to guest memory by themselves (readbacks). */
void rx_nvn_invalidate(uint64_t gpu, uint64_t size) { (void)gpu, (void)size; }

}  // extern "C"
