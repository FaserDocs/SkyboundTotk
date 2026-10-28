// Frame dumps (debugging): RX_DUMP_FRAME=<n> (or RX_DUMP_AT=<seconds>) logs frame n's passes in order (their targets, how many draws went
// into each, dispatches, clears) and, at its present, saves every render target as a PNG in RX_DUMP_DIR (default
// "rx_frame"): <order of last use>_<guest address>_<width>x<height>_<format>.png. Float formats are clamped to
// [0, 1]; depth is shown as gray.
#include <algorithm>
#include <array>
#include <map>
#include <tuple>
#include <cstring>
#include <cmath>
#include <cstdlib>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "renderdoc_app.h"
#include "rx_internal.h"

namespace rx {
namespace {

int64_t dump_frame = -2;
int pass_draws;
bool pass_open;
/// The render targets in the order passes used them (their textures held: an image can be replaced mid-frame
/// when the game reuses its memory for another surface).
struct Touched {
    WGPUTexture tex;
    uint64_t guest;
    uint32_t width, height;
    WGPUTextureFormat format;
};
std::vector<Touched> touched;

float half(uint16_t h) {
    const uint32_t s = h >> 15, e = h >> 10 & 31, m = h & 1023;
    const float f = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? 65504.0f : std::ldexp(1.0f + m / 1024.0f, (int)e - 15);
    return s ? -f : f;
}
float small_float(uint32_t v, int mbits) {
    const uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return std::ldexp((float)m, -14 - mbits);
    if (e == 31) return 65504.0f;
    return std::ldexp(1.0f + (float)m / (1u << mbits), (int)e - 15);
}
uint8_t to8(float f) {
    return (uint8_t)std::lround(std::clamp(f, 0.0f, 1.0f) * 255.0f);
}

/// The texel at p in RGBA8 (false: a format this does not show).
bool texel(WGPUTextureFormat f, const uint8_t* p, uint8_t out[4]) {
    out[3] = 255;
    switch (f) {
    case WGPUTextureFormat_RGBA8Unorm:
    case WGPUTextureFormat_RGBA8UnormSrgb:
        std::memcpy(out, p, 4);
        out[3] = 255;
        return true;
    case WGPUTextureFormat_BGRA8Unorm:
    case WGPUTextureFormat_BGRA8UnormSrgb:
        out[0] = p[2], out[1] = p[1], out[2] = p[0];
        return true;
    case WGPUTextureFormat_R8Unorm:
        out[0] = out[1] = out[2] = p[0];
        return true;
    case WGPUTextureFormat_RG8Unorm:
        out[0] = p[0], out[1] = p[1], out[2] = 0;
        return true;
    case WGPUTextureFormat_RGBA16Float: {
        uint16_t c[4];
        std::memcpy(c, p, 8);
        for (int i = 0; i < 3; i++) out[i] = to8(half(c[i]));
        return true;
    }
    case WGPUTextureFormat_RG16Float: {
        uint16_t c[2];
        std::memcpy(c, p, 4);
        out[0] = to8(half(c[0])), out[1] = to8(half(c[1])), out[2] = 0;
        return true;
    }
    case WGPUTextureFormat_R16Float: {
        uint16_t c;
        std::memcpy(&c, p, 2);
        out[0] = out[1] = out[2] = to8(half(c));
        return true;
    }
    case WGPUTextureFormat_Depth16Unorm: {
        uint16_t v;
        std::memcpy(&v, p, 2);
        out[0] = out[1] = out[2] = (uint8_t)(v >> 8);
        return true;
    }
    case WGPUTextureFormat_R32Float:
    case WGPUTextureFormat_Depth32Float: {
        float v;
        std::memcpy(&v, p, 4);
        out[0] = out[1] = out[2] = to8(v);
        return true;
    }
    case WGPUTextureFormat_RG11B10Ufloat: {
        uint32_t v;
        std::memcpy(&v, p, 4);
        out[0] = to8(small_float(v & 0x7FF, 6)), out[1] = to8(small_float(v >> 11 & 0x7FF, 6));
        out[2] = to8(small_float(v >> 22, 5));
        return true;
    }
    case WGPUTextureFormat_RGB10A2Unorm: {
        uint32_t v;
        std::memcpy(&v, p, 4);
        out[0] = (uint8_t)((v & 1023) >> 2), out[1] = (uint8_t)((v >> 10 & 1023) >> 2);
        out[2] = (uint8_t)((v >> 20 & 1023) >> 2);
        return true;
    }
    default:
        return false;
    }
}

uint32_t texel_bytes(WGPUTextureFormat f) {
    switch (f) {
    case WGPUTextureFormat_R8Unorm: return 1;
    case WGPUTextureFormat_RG8Unorm: case WGPUTextureFormat_R16Float: case WGPUTextureFormat_Depth16Unorm: return 2;
    case WGPUTextureFormat_RGBA16Float: return 8;
    default: return 4;
    }
}

void save(const Touched* im, const char* dir, int order) {
    uint8_t probe[4];
    const uint8_t zero[8]{};
    if (!texel(im->format, zero, probe)) return;
    const uint32_t bpp = texel_bytes(im->format);
    const size_t stride = ((size_t)im->width * bpp + 255) & ~size_t(255), size = stride * im->height;
    WGPUBufferDescriptor bd{};
    bd.size = size;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUTexelCopyTextureInfo src{};
    src.texture = im->tex;
    src.aspect = is_depth_format(im->format) ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = buffer;
    dst.layout.bytesPerRow = (uint32_t)stride;
    dst.layout.rowsPerImage = im->height;
    const WGPUExtent3D extent{im->width, im->height, 1};
    wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    struct Done {
        bool complete = false;
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    } done;
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        auto* d = static_cast<Done*>(u1);
        d->status = status;
        d->complete = true;
    };
    mc.userdata1 = &done;
    wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, size, mc);
    while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
    if (done.status == WGPUMapAsyncStatus_Success) {
        const auto* px = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(buffer, 0, size));
        std::vector<uint8_t> rgba((size_t)im->width * im->height * 4);
        for (uint32_t y = 0; y < im->height; y++)
            for (uint32_t x = 0; x < im->width; x++)
                texel(im->format, px + y * stride + (size_t)x * bpp, &rgba[((size_t)y * im->width + x) * 4]);
        char path[512];
        std::snprintf(path, sizeof path, "%s/%03d_%08llx_%ux%u_%x.png", dir, order, (unsigned long long)im->guest,
                      im->width, im->height, (unsigned)im->format);
        write_png(path, rgba.data(), im->width, im->height, (size_t)im->width * 4);
        if (im->format == WGPUTextureFormat_R32Float || im->format == WGPUTextureFormat_Depth32Float ||
            im->format == WGPUTextureFormat_RGBA16Float || im->format == WGPUTextureFormat_RG11B10Ufloat ||
            im->format == WGPUTextureFormat_RGBA8Unorm || im->format == WGPUTextureFormat_RGBA8UnormSrgb ||
            im->format == WGPUTextureFormat_Depth16Unorm || im->format == WGPUTextureFormat_RG16Float ||
            im->format == WGPUTextureFormat_R16Float) {   // raw too
            std::snprintf(path, sizeof path, "%s/%03d_%08llx_%ux%u_%x.raw", dir, order, (unsigned long long)im->guest,
                          im->width, im->height, (unsigned)im->format);
            if (FILE* f = std::fopen(path, "wb")) {
                for (uint32_t y = 0; y < im->height; y++) std::fwrite(px + y * stride, 1, (size_t)im->width * bpp, f);
                std::fclose(f);
            }
        }
        wgpuBufferUnmap(buffer);
    }
    wgpuBufferRelease(buffer);
}

void touch(const Image* im) {
    if (!im) return;
    for (const Touched& t : touched)
        if (t.tex == im->tex) return;
    wgpuTextureAddRef(im->tex);
    touched.push_back({im->tex, im->guest, im->width, im->height, im->format});
}

std::vector<std::pair<Touched, std::string>> snapshots;   // RX_DUMP_AFTER: copies taken after chosen draws

} // namespace

/// RX_DUMP_AFTER=<fs address>[,...] (with a frame dump): color target 0 is copied after each draw of those pixel
/// programs and saved with the frame (snap_<n>_<fs>.png); with RX_DUMP_AFTER_DEPTH=1 the depth target instead.
void dump_after_draw(uint64_t fs_address, Image* target) {
    if (!frame_dump_active() || !target) return;
    static std::vector<uint64_t> wanted;
    static bool parsed;
    if (!parsed) {
        parsed = true;
        if (const char* s = getenv("RX_DUMP_AFTER"))
            for (const char* p = s; *p;) {
                char* end;
                wanted.push_back(strtoull(p, &end, 16));
                p = *end == ',' ? end + 1 : end;
                if (end == p && !*end) break;
            }
    }
    // RX_DUMP_AFTER_TARGET=<guest hex> (with a frame dump): after every draw into that target, whatever its program
    static const uint64_t target_guest = getenv("RX_DUMP_AFTER_TARGET") ? strtoull(getenv("RX_DUMP_AFTER_TARGET"), nullptr, 16) : 0;
    if (!(target_guest && target->guest == target_guest) &&
        std::find(wanted.begin(), wanted.end(), fs_address) == wanted.end())
        return;
    end_render_pass();
    WGPUTextureDescriptor d{};
    d.usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    d.dimension = WGPUTextureDimension_2D;
    d.size = {target->width, target->height, 1};
    d.format = target->format;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    WGPUTexture copy = wgpuDeviceCreateTexture(dev.device, &d);
    WGPUTexelCopyTextureInfo src{}, dst{};
    src.texture = target->tex;
    dst.texture = copy;
    const WGPUExtent3D extent{target->width, target->height, 1};
    ce_copy_texture_to_texture(encoder(), &src, &dst, &extent);
    char label[64];
    std::snprintf(label, sizeof label, "snap_%03zu_%llx", snapshots.size(), (unsigned long long)fs_address);
    snapshots.push_back({{copy, target->guest, target->width, target->height, target->format}, label});
}

namespace {
struct PixelRead {
    WGPUTextureFormat format;
    uint64_t guest;
    int slot;
    uint32_t x, y;
    std::string what;
};
std::vector<PixelRead> pixel_reads;
WGPUBuffer pixel_buffer = nullptr;
constexpr uint32_t kPixelReads = 16384;
}  // namespace

/// RX_PIXEL=<x>,<y> (with a frame dump; 1920x1080 coordinates, scaled to each target's size): after every draw each
/// colour attachment's texel there is read back and logged with the draw at the frame's end (rx-pixel lines): which
/// draw put a mark on the screen.
void dump_pixel_after(const Attachments& at, const char* what) {
    // (RX_PIXEL=<x>,<y>[;<x>,<y>...]: up to 8 points)
    static std::vector<std::pair<int, int>> points;
    static const bool parsed = [] {
        if (const char* s = getenv("RX_PIXEL"))
            for (const char* p = s; *p;) {
                int x, y;
                if (std::sscanf(p, "%d,%d", &x, &y) != 2) break;
                points.push_back({x, y});
                const char* semi = std::strchr(p, ';');
                if (!semi || points.size() >= 8) break;
                p = semi + 1;
            }
        return true;
    }();
    (void)parsed;
    if (points.empty() || !frame_dump_active()) return;
    // RX_PIXEL_TARGET=<guest hex>: only after draws into that target (the passes before it are left as they are)
    static const uint64_t only = getenv("RX_PIXEL_TARGET") ? strtoull(getenv("RX_PIXEL_TARGET"), nullptr, 16) : 0;
    // RX_PIXEL_FROM=<text>: only from the first draw whose description contains it on (every target after it)
    static const char* from = getenv("RX_PIXEL_FROM");
    static int64_t from_frame = -1;
    if (from) {
        if (from_frame != frame_number) {
            if (!what || !std::strstr(what, from)) return;
            from_frame = frame_number;
        }
    } else if (only) {
        bool hit = false;
        for (const Image* c : at.color) hit |= c && c->guest == only;
        if (!hit) return;
    }
    if (!pixel_buffer) {
        WGPUBufferDescriptor bd{};
        bd.size = (uint64_t)kPixelReads * 256;
        bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        pixel_buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
    }
    bool ended = false;
    for (int i = 0; i < 8; i++) {
        const Image* t = at.color[i];
        if (!t || pixel_reads.size() >= kPixelReads) continue;
        uint8_t probe[4];
        const uint8_t zero[8]{};
        if (!texel(t->format, zero, probe)) continue;
        if (!ended) end_render_pass(), ended = true;
        for (const auto& [px, py] : points) {
        if (pixel_reads.size() >= kPixelReads) break;
        const uint32_t x = std::min<uint32_t>(t->width - 1, (uint32_t)((uint64_t)px * t->width / 1920));
        const uint32_t y = std::min<uint32_t>(t->height - 1, (uint32_t)((uint64_t)py * t->height / 1080));
        WGPUTexelCopyTextureInfo src{};
        src.texture = t->tex;
        src.origin = {x, y, 0};
        WGPUTexelCopyBufferInfo dst{};
        dst.buffer = pixel_buffer;
        dst.layout.offset = (uint64_t)pixel_reads.size() * 256;
        dst.layout.bytesPerRow = 256;
        dst.layout.rowsPerImage = 1;
        const WGPUExtent3D one{1, 1, 1};
        ce_copy_texture_to_buffer(encoder(), &src, &dst, &one);
        pixel_reads.push_back({t->format, t->guest, i, x, y, what ? what : ""});
        }
    }
}

namespace {
uint64_t watch_va, watch_size;
int64_t watch_frames = 600;
void watch_parse() {
    static const bool parsed = [] {
        if (const char* s = getenv("RX_WATCH_BUF")) {
            char* e = nullptr;
            watch_va = strtoull(s, &e, 16);
            if (e && *e == ':') watch_size = strtoull(e + 1, &e, 16);
            if (e && *e == ':') watch_frames = atoll(e + 1);
        }
        return true;
    }();
    (void)parsed;
}
// the watched range's buffer cache copy read back; the 16-byte records changed since the last read logged (rx-buf
// lines, after <when>), and the copy appended to RX_WATCH_BUF_FILE under <tag>
void watch_buffer_read(const char* when, int64_t tag) {
    const BufferRef ref = guest_buffer(watch_va, watch_size);
    if (!ref.buffer) return;
    flush_batch();   // (the uploads guest_buffer queued)
    wait_submitted();
    WGPUBufferDescriptor bd{};
    bd.size = (watch_size + 3) & ~3ull;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(dev.device, &bd);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(e, ref.buffer, ref.offset, staging, 0, bd.size);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    struct Done {
        bool complete = false;
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    } done;
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        auto* d = static_cast<Done*>(u1);
        d->status = status;
        d->complete = true;
    };
    mc.userdata1 = &done;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bd.size, mc);
    while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
    static std::vector<uint32_t> last;
    if (done.status == WGPUMapAsyncStatus_Success) {
        const auto* w = static_cast<const uint32_t*>(wgpuBufferGetConstMappedRange(staging, 0, bd.size));
        const size_t n = bd.size / 4;
        if (last.size() != n) last.assign(n, 0xDEADBEEFu);
        size_t nonzero = 0, changed = 0;
        std::string line;
        for (size_t i = 0; i < n; i++) nonzero += w[i] != 0;
        for (size_t r = 0; r + 4 <= n; r += 4) {
            if (!std::memcmp(w + r, last.data() + r, 16)) continue;
            if (changed++ < 48) {
                char b[80];
                std::snprintf(b, sizeof b, " %zx:%x,%x,%x,%x", r * 4, w[r], w[r + 1], w[r + 2], w[r + 3]);
                line += b;
            }
        }
        if (changed || tag < (1ll << 40))   // (after draws: only the ones that changed it)
            hos_log("rx-buf: %s (tag %lld): %zu nonzero words, %zu records changed:%s%s", when, (long long)tag, nonzero,
                    changed, line.c_str(), changed > 48 ? " ..." : "");
        // RX_WATCH_BUF_FILE=<path>: every copy appended (its tag, 8 bytes, then the range)
        if (static FILE* f = getenv("RX_WATCH_BUF_FILE") ? std::fopen(getenv("RX_WATCH_BUF_FILE"), "wb") : nullptr; f) {
            std::fwrite(&tag, 8, 1, f);
            std::fwrite(w, 4, n, f);
            std::fflush(f);
        }
        last.assign(w, w + n);
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
}
}  // namespace

/// RX_WATCH_BUF=<GPU VA hex>:<size hex>[:<frames>] (debugging): at each present the buffer cache's copy of that range
/// (what the GPU's draws read) is read back, and the 16-byte records that changed since the last present are logged
/// (rx-buf lines), for that many frames (default 600) from RX_WATCH_BUF_AFTER=<seconds> on.
void watch_buffer_frame() {
    watch_parse();
    if (!watch_va || !watch_size || watch_frames <= 0) return;
    static const int64_t after = getenv("RX_WATCH_BUF_AFTER") ? (int64_t)(atof(getenv("RX_WATCH_BUF_AFTER")) * 1e9) : 0;
    static const int64_t start = hos_now_ns();
    if (hos_now_ns() - start < after) return;
    char when[48];
    std::snprintf(when, sizeof when, "frame %lld", (long long)frame_number);
    watch_buffer_read(when, frame_number);
    watch_frames--;
}

/// RX_WATCH_BUF_DRAWS=1 (with RX_WATCH_BUF and a frame dump): the range is also read back after each draw of the
/// dumped frame (ending its pass), and the draws that changed it are logged (rx-buf: after <draw>; tagged 1 << 40 +
/// the draw's number in the frame, in RX_WATCH_BUF_FILE): which draw wrote a record, and what the next one read
void watch_buffer_after_draw(const char* what) {
    static const bool per_draw = env_flag("RX_WATCH_BUF_DRAWS");
    if (!per_draw || !frame_dump_active()) return;
    watch_parse();
    if (!watch_va || !watch_size) return;
    static int64_t draw_frame = -1, draw = 0;
    if (draw_frame != frame_number) draw_frame = frame_number, draw = 0;
    end_render_pass();
    std::string when = "after ";
    when += what ? what : "?";
    watch_buffer_read(when.c_str(), (1ll << 40) + draw++);
}

/// RX_DUMP_BEFORE=<fs address>[,...] (with a frame dump): every colour attachment and the depth are copied just
/// before each draw of those pixel programs (what the draw reads through feedback, what it blends with), saved as
/// pre_<n>_<fs>_c<slot>.png/.raw
void dump_before_draw(uint64_t fs_address, const Attachments& at) {
    if (!frame_dump_active()) return;
    static std::vector<uint64_t> wanted;
    static bool parsed;
    if (!parsed) {
        parsed = true;
        if (const char* s = getenv("RX_DUMP_BEFORE"))
            for (const char* p = s; *p;) {
                char* end;
                wanted.push_back(strtoull(p, &end, 16));
                if (end == p) break;
                p = *end == ',' ? end + 1 : end;
            }
    }
    if (std::find(wanted.begin(), wanted.end(), fs_address) == wanted.end()) return;
    end_render_pass();
    static int n;
    for (int i = 0; i < 9; i++) {
        const Image* t = i < 8 ? at.color[i] : at.depth;
        if (!t) continue;
        WGPUTextureDescriptor d{};
        d.usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
        d.dimension = WGPUTextureDimension_2D;
        d.size = {t->width, t->height, 1};
        d.format = t->format;
        d.mipLevelCount = 1;
        d.sampleCount = 1;
        WGPUTexture copy = wgpuDeviceCreateTexture(dev.device, &d);
        WGPUTexelCopyTextureInfo src{}, dst{};
        src.texture = t->tex;
        src.aspect = is_depth_format(t->format) ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
        dst.texture = copy;
        dst.aspect = src.aspect;
        const WGPUExtent3D extent{t->width, t->height, 1};
        ce_copy_texture_to_texture(encoder(), &src, &dst, &extent);
        char label[64];
        std::snprintf(label, sizeof label, "pre_%03d_%llx_%s%d", n, (unsigned long long)fs_address, i < 8 ? "c" : "z",
                      i < 8 ? i : 0);
        snapshots.push_back({{copy, t->guest, t->width, t->height, t->format}, label});
    }
    n++;
}

/// RX_DUMP_ON_RT=<GPU VA hex> (with RX_DUMP_FRAME set to a far frame): the dump starts at the first draw into that
/// color target, mid-frame (a target drawn once, at loading, say)
void dump_start_now() {
    if (dump_frame < 0 || dump_frame > frame_number) dump_frame = frame_number;
}

bool frame_dump_enabled = false;
std::atomic<int> dump_requested{0};   // rx_request_frame_dump (the live input file's "!dump")

bool frame_dump_active_slow() {
    if (dump_frame == -2) {
        const char* s = getenv("RX_DUMP_FRAME");
        dump_frame = s ? atoll(s) : -1;
    }
    if (dump_requested.exchange(0)) dump_frame = frame_number + 1;
    // RX_DUMP_AT=<seconds>: the frame that starts after that much time (since the renderer started)
    static const char* at = getenv("RX_DUMP_AT");
    static int64_t start = hos_now_ns();
    if (at && dump_frame < 0 && hos_now_ns() - start >= (int64_t)(atof(at) * 1e9)) dump_frame = frame_number + 1;
    // RX_DUMP_COUNT=<n>: n frames from there (each in a folder of its own)
    static const int64_t n = getenv("RX_DUMP_COUNT") ? std::max(1, atoi(getenv("RX_DUMP_COUNT"))) : 1;
    return dump_frame >= 0 && frame_number >= dump_frame && frame_number < dump_frame + n;
}

/// RX_DUMP_ON_GUEST=<guest hex> (with RX_DUMP_FRAME set to a far frame): the dump starts at the first draw or pass into
/// that target (either path; RX_DUMP_ON_RT takes the emulated path's GPU VA). Native draws check before they bind.
/// RX_DUMP_ON_SIZE=<width>x<height>: the same at the first color target of that size (both: that target at that
/// size). RX_DUMP_ON_ARM=<seconds>: not
/// before that much time (past the title screen's scene); RX_DUMP_ON_DELAY=<frames>: that many frames later.
void dump_on_targets(const Attachments& at) {
    static const uint64_t on_guest = getenv("RX_DUMP_ON_GUEST") ? strtoull(getenv("RX_DUMP_ON_GUEST"), nullptr, 16) : 0;
    static uint32_t on_w = 0, on_h = 0;
    static const bool size_read = [] {
        if (const char* s = getenv("RX_DUMP_ON_SIZE")) std::sscanf(s, "%ux%u", &on_w, &on_h);
        return true;
    }();
    (void)size_read;
    if ((!on_guest && !on_w) || !frame_dump_enabled) return;
    static const int64_t arm_ns = getenv("RX_DUMP_ON_ARM") ? (int64_t)(atof(getenv("RX_DUMP_ON_ARM")) * 1e9) : 0;
    static const int64_t start = hos_now_ns();
    if (arm_ns && hos_now_ns() - start < arm_ns) return;
    static const int64_t delay = getenv("RX_DUMP_ON_DELAY") ? atoll(getenv("RX_DUMP_ON_DELAY")) : 0;
    static int64_t fired_at = -1;
    if (fired_at >= 0) {
        if (frame_number >= fired_at + delay) dump_start_now();
        return;
    }
    for (int i = 0; i < 8; i++)
        if (at.color[i] && (!on_guest || at.color[i]->guest == on_guest) &&
            (!on_w || (at.color[i]->width == on_w && at.color[i]->height == on_h))) {
            fired_at = frame_number;
            if (!delay) dump_start_now();
            return;
        }
}

void dump_pass(const Attachments& at, const char* kind) {
    // RX_WATCH_RT=<guest hex>: every pass into that target is logged (any frame)
    static const uint64_t watch = getenv("RX_WATCH_RT") ? strtoull(getenv("RX_WATCH_RT"), nullptr, 16) : 0;
    if (watch) {
        static int logged[2];
        for (int i = 0; i < 8; i++)
            if (at.color[i] && at.color[i]->guest == watch && logged[kind[0] == 'd']++ < 400)
                hos_log("rx: watch %08llx: %s pass %ux%u as c%d (format %#x, pixel format %d) frame %lld",
                        (unsigned long long)watch, kind, at.width, at.height, i, (unsigned)at.color[i]->format,
                        (int)at.color[i]->pixel_format, (long long)frame_number);
    }
    dump_on_targets(at);
    if (!frame_dump_active()) return;
    if (pass_open) hos_log("rx-frame:   %d draws", pass_draws);
    pass_draws = 0;
    pass_open = true;
    char b[512];
    int n = std::snprintf(b, sizeof b, "rx-frame: %s pass %ux%u:", kind, at.width, at.height);
    for (int i = 0; i < 8; i++)
        if (at.color[i]) {
            touch(at.color[i]);
            n += std::snprintf(b + n, sizeof b - n, " c%d %08llx@%04x fmt %x", i, (unsigned long long)at.color[i]->guest,
                               (unsigned)(((uintptr_t)at.color[i] >> 4) & 0xFFFF),
                               (unsigned)at.color[i]->format);
        }
    if (at.depth) {
        touch(at.depth);
        std::snprintf(b + n, sizeof b - n, " z %08llx fmt %x", (unsigned long long)at.depth->guest,
                      (unsigned)at.depth->format);
    }
    hos_log("%s", b);
}

void dump_sampled(const Image* im) {   // RX_DUMP_SAMPLED=1: render targets read as textures are saved too
    static const bool on = env_flag("RX_DUMP_SAMPLED");
    if (on && frame_dump_active()) touch(im);
}

void dump_draw(const char* what) {
    if (!frame_dump_active()) return;
    if (what) hos_log("rx-frame:   %s", what);
    else pass_draws++;
}

void dump_texture(WGPUTexture tex, uint64_t va, uint32_t width, uint32_t height, WGPUTextureFormat format,
                  const char* dir, int order) {
    const Touched t{tex, va, width, height, format};
    save(&t, dir, order);
}

void dump_frame_images(const char* dir) {
    if (!frame_dump_active()) return;
    if (pass_open) hos_log("rx-frame:   %d draws", pass_draws);
    pass_open = false;
    CreateDirectoryA(dir, nullptr);
    char sub[512];
    if (getenv("RX_DUMP_COUNT")) {
        std::snprintf(sub, sizeof sub, "%s/%lld", dir, (long long)frame_number);
        CreateDirectoryA(sub, nullptr);
        dir = sub;
        hos_log("rx-frame: end of frame %lld", (long long)frame_number);
    }
    int order = 0;
    // RX_DUMP_ONLY=<guest hex>[,...]: only those targets are saved
    static std::vector<uint64_t> only;
    static bool only_parsed;
    if (!only_parsed) {
        only_parsed = true;
        if (const char* s = getenv("RX_DUMP_ONLY"))
            for (const char* p = s; *p;) {
                char* end;
                only.push_back(strtoull(p, &end, 16));
                if (end == p) break;
                p = *end == ',' ? end + 1 : end;
            }
    }
    for (const Touched& t : touched) {
        if (only.empty() || std::find(only.begin(), only.end(), t.guest) != only.end()) save(&t, dir, order);
        order++;
        wgpuTextureRelease(t.tex);
    }
    hos_log("rx-frame: %zu render targets saved in %s", touched.size(), dir);
    touched.clear();
    dump_tex_list(dir);
    for (auto& [t, label] : snapshots) {
        save(&t, dir, 900 + order++);
        char from[512], to[512];
        std::snprintf(from, sizeof from, "%s/%03d_%08llx_%ux%u_%x.png", dir, 900 + order - 1, (unsigned long long)t.guest,
                      t.width, t.height, (unsigned)t.format);
        std::snprintf(to, sizeof to, "%s/%s.png", dir, label.c_str());
        MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING);
        std::strcpy(from + std::strlen(from) - 4, ".raw");   // (and the raw texels, when the format has them)
        std::snprintf(to, sizeof to, "%s/%s.raw", dir, label.c_str());
        MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING);
        wgpuTextureRelease(t.tex);
    }
    snapshots.clear();
    if (!pixel_reads.empty()) {   // (RX_PIXEL: the frame's work was submitted)
        struct Done {
            bool complete = false;
            WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        } done;
        WGPUBufferMapCallbackInfo mc{};
        mc.mode = WGPUCallbackMode_AllowSpontaneous;
        mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
            auto* d = static_cast<Done*>(u1);
            d->status = status;
            d->complete = true;
        };
        mc.userdata1 = &done;
        const uint64_t size = (uint64_t)pixel_reads.size() * 256;
        wgpuBufferMapAsync(pixel_buffer, WGPUMapMode_Read, 0, size, mc);
        while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
        if (done.status == WGPUMapAsyncStatus_Success) {
            const auto* px = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(pixel_buffer, 0, size));
            std::map<std::tuple<uint64_t, uint32_t, uint32_t>, std::array<uint8_t, 16>> last;
            for (size_t i = 0; i < pixel_reads.size(); i++) {
                const PixelRead& r = pixel_reads[i];
                const uint8_t* v = px + i * 256;
                const uint32_t bpp = texel_bytes(r.format);
                // (only what changed: a draw that did not touch the texel is left out)
                auto& l = last[{r.guest, r.x, r.y}];
                if (!std::memcmp(l.data(), v, bpp) && i) continue;
                std::memcpy(l.data(), v, bpp);
                uint8_t c[4];
                texel(r.format, v, c);
                char hex[40] = {};
                for (uint32_t k = 0; k < bpp && k < 16; k++) std::snprintf(hex + 2 * k, 3, "%02x", v[k]);
                hos_log("rx-pixel: %4zu c%d %08llx (%u,%u) = %3u %3u %3u %3u [%s] after %s", i, r.slot,
                        (unsigned long long)r.guest, r.x, r.y, c[0], c[1], c[2], c[3], hex, r.what.c_str());
            }
            wgpuBufferUnmap(pixel_buffer);
        }
        pixel_reads.clear();
    }
}

/// RenderDoc captures of the renderer's own frames (debugging). Run with RenderDoc's Vulkan layer loaded through the
/// environment (VK_ADD_IMPLICIT_LAYER_PATH=<RenderDoc's folder> ENABLE_VULKAN_RENDERDOC_CAPTURE=1: the log and the live
/// input keep working, headless too); then the live input's "!rdoc [frames]" or RX_RDOC_AT=<frame number> captures
/// that many frames (default 1) from the next present on, every batch submitted in between, into RX_RDOC_FILE=<path
/// template> (RenderDoc's default folder otherwise).
std::atomic<int> rdoc_requested{0};

#if RX_WEB
void rdoc_present() {}
#else
RENDERDOC_API_1_1_2* rdoc_api() {
    static RENDERDOC_API_1_1_2* const api = [] {
        RENDERDOC_API_1_1_2* a = nullptr;
        if (HMODULE m = GetModuleHandleA("renderdoc.dll"))
            if (auto get = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(m, "RENDERDOC_GetAPI")))
                if (!get(eRENDERDOC_API_Version_1_1_2, reinterpret_cast<void**>(&a))) a = nullptr;
        if (a && getenv("RX_RDOC_FILE")) a->SetCaptureFilePathTemplate(getenv("RX_RDOC_FILE"));
        return a;
    }();
    return api;
}

void rdoc_present() {
    static int left = 0;
    static const int64_t at = getenv("RX_RDOC_AT") ? atoll(getenv("RX_RDOC_AT")) : -1;
    if (left > 0) {
        if (--left == 0) {
            wait_submitted();   // (all of the captured frames' batches reached the queue)
            rdoc_api()->EndFrameCapture(nullptr, nullptr);
            hos_log("rx: RenderDoc capture ended at frame %lld", (long long)frame_number);
        }
        return;
    }
    int n = rdoc_requested.exchange(0);
    if (frame_number == at) n = 1;
    if (n <= 0) return;
    if (!rdoc_api()) {
        log_limited("rdoc", 4, "rx: no RenderDoc capture: its Vulkan layer is not loaded (ENABLE_VULKAN_RENDERDOC_CAPTURE)");
        return;
    }
    wait_submitted();   // (nothing recorded before the capture lands in it)
    rdoc_api()->StartFrameCapture(nullptr, nullptr);
    left = n;
    hos_log("rx: RenderDoc capture of %d frame(s) from frame %lld", n, (long long)frame_number);
}
#endif

} // namespace rx

/// A frame dump of the next frame, as RX_DUMP_AT would make (the live input file's "!dump" line: hos/nn_hid.c).
extern "C" void rx_request_frame_dump(void) {
    rx::frame_dump_enabled = true;
    rx::dump_requested = 1;
}

/// RenderDoc captures the next `frames` frames (the live input file's "!rdoc [frames]" line: rdoc_present).
extern "C" void rx_request_rdoc_capture(int frames) {
    rx::rdoc_requested = frames > 0 ? frames : 1;
}
