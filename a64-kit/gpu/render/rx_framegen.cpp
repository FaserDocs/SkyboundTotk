// Frame interpolation (F7, RX_FRAMEGEN=1): frames made between the game's, from its own two last frames, shown in
// between by a display thread, so a game at 60 fps is shown at 120 (RX_FRAMEGEN_MULT=3: 180) on a fast display.
// TOTK hands out no motion vectors, so the motion is estimated from the images (optical flow):
//   1. each frame is copied (it may be drawn over next frame) and reduced to quarter-size luma;
//   2. block matching: for every 16x16 pixels, the offset (within +-32 pixels) where the last frame's 8x8 quarter-size
//      patch matches the new frame's best (a small penalty for long vectors keeps still areas still), then a 3x3
//      median of those vectors;
//   3. each in-between frame at time t: every pixel takes the last frame back along t of its motion and the new frame
//      forward along 1 - t, blended by t.
// The display thread shows the in-between frames, then the new one, evenly over the game's frame time (its recent
// mean): what is shown runs about one frame behind the game. The HUD over moving scenery and things moving faster
// than the search reaches smear a little; loading screens (frames over 100 ms apart) are not interpolated.
// The threaded web build too: only its WebGPU thread may present to the canvas, and it may not sleep (the batches go
// through it), so there the made frames are shown by that thread's loop as they come due (framegen_web_tick). The
// single-threaded web build: not done (its one render thread is the replay).
#include "rx_internal.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

extern "C" void hos_host_thread_register_self(const char* name);

namespace rx {
namespace {

const char* kCopyWgsl = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = textureDimensions(dst);
  if (g.x >= d.x || g.y >= d.y) { return; }
  textureStore(dst, g.xy, vec4<f32>(textureLoad(src, g.xy, 0).rgb, 1.0));
}
)";

const char* kLumaWgsl = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = textureDimensions(dst);
  if (g.x >= d.x || g.y >= d.y) { return; }
  let s = textureDimensions(src) - vec2<u32>(1u);
  var sum = 0.0;
  for (var y = 0u; y < 4u; y++) {
    for (var x = 0u; x < 4u; x++) {
      sum += dot(textureLoad(src, min(g.xy * 4u + vec2<u32>(x, y), s), 0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    }
  }
  textureStore(dst, g.xy, vec4<f32>(sum / 16.0, 0.0, 0.0, 1.0));
}
)";

// one vector per 4x4 quarter-size texels (16x16 pixels); the search covers +-8 quarter texels (+-32 pixels)
const char* kFlowWgsl = R"(
@group(0) @binding(0) var la: texture_2d<f32>;
@group(0) @binding(1) var lb: texture_2d<f32>;
@group(0) @binding(2) var flow: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let fd = textureDimensions(flow);
  if (g.x >= fd.x || g.y >= fd.y) { return; }
  let lm = vec2<i32>(textureDimensions(la)) - vec2<i32>(1);
  let c = vec2<i32>(g.xy) * 4 + vec2<i32>(2);
  var pa: array<f32, 64>;
  for (var i = 0; i < 64; i++) {
    pa[i] = textureLoad(la, clamp(c + vec2<i32>(i % 8 - 4, i / 8 - 4), vec2<i32>(0), lm), 0).r;
  }
  var best = 1e9;
  var bv = vec2<i32>(0);
  for (var dy = -RADIUS; dy <= RADIUS; dy++) {
    for (var dx = -RADIUS; dx <= RADIUS; dx++) {
      var sad = 0.0;
      for (var i = 0; i < 64; i++) {
        let q = clamp(c + vec2<i32>(i % 8 - 4 + dx, i / 8 - 4 + dy), vec2<i32>(0), lm);
        sad += abs(pa[i] - textureLoad(lb, q, 0).r);
      }
      let cost = sad + 0.04 * f32(abs(dx) + abs(dy));
      if (cost < best) { best = cost; bv = vec2<i32>(dx, dy); }
    }
  }
  textureStore(flow, g.xy, vec4<f32>(vec2<f32>(bv) * 4.0, best / 64.0, 1.0));
}
)";

const char* kMedianWgsl = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
fn med9(v: array<f32, 9>) -> f32 {
  var a = v;
  for (var i = 0; i < 5; i++) {   // (the 5 smallest in order: the 5th is the median)
    for (var j = i + 1; j < 9; j++) {
      if (a[j] < a[i]) { let t = a[i]; a[i] = a[j]; a[j] = t; }
    }
  }
  return a[4];
}
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = textureDimensions(dst);
  if (g.x >= d.x || g.y >= d.y) { return; }
  let m = vec2<i32>(d) - vec2<i32>(1);
  var xs: array<f32, 9>;
  var ys: array<f32, 9>;
  var e = 0.0;
  for (var i = 0; i < 9; i++) {
    let f = textureLoad(src, clamp(vec2<i32>(g.xy) + vec2<i32>(i % 3 - 1, i / 3 - 1), vec2<i32>(0), m), 0);
    xs[i] = f.x;
    ys[i] = f.y;
    e = max(e, f.z);
  }
  textureStore(dst, g.xy, vec4<f32>(med9(xs), med9(ys), e, 1.0));
}
)";

const char* kSynthWgsl = R"(
@group(0) @binding(0) var fa: texture_2d<f32>;
@group(0) @binding(1) var fb: texture_2d<f32>;
@group(0) @binding(2) var flow: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<uniform> u: vec4<f32>;   // t
fn fl(q: vec2<i32>) -> vec2<f32> {
  return textureLoad(flow, clamp(q, vec2<i32>(0), vec2<i32>(textureDimensions(flow)) - vec2<i32>(1)), 0).xy;
}
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = textureDimensions(dst);
  if (g.x >= d.x || g.y >= d.y) { return; }
  let t = u.x;
  let p = vec2<f32>(g.xy) + 0.5;
  // the motion here: the vectors (one per 16x16 pixels, at their centres) bilinearly
  let fp = p / 16.0 - 0.5;
  let i = vec2<i32>(floor(fp));
  let w = fp - floor(fp);
  let v = mix(mix(fl(i), fl(i + vec2<i32>(1, 0)), w.x), mix(fl(i + vec2<i32>(0, 1)), fl(i + vec2<i32>(1, 1)), w.x), w.y);
  let size = vec2<f32>(d);
  let a = textureSampleLevel(fa, samp, (p - t * v) / size, 0.0).rgb;
  let b = textureSampleLevel(fb, samp, (p + (1.0 - t) * v) / size, 0.0).rgb;
  textureStore(dst, g.xy, vec4<f32>(mix(a, b, t), 1.0));
}
)";

constexpr int kMaxMult = 4;

struct Tex {
    WGPUTexture tex = nullptr;
    WGPUTextureView view = nullptr;
};
Tex make_tex(uint32_t w, uint32_t h, WGPUTextureFormat f, WGPUTextureUsage usage, const char* label) {
    WGPUTextureDescriptor d{};
    d.label = sv(label);
    d.usage = usage;
    d.dimension = WGPUTextureDimension_2D;
    d.size = {w, h, 1};
    d.format = f;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    Tex t;
    t.tex = wgpuDeviceCreateTexture(dev.device, &d);
    t.view = wgpuTextureCreateView(t.tex, nullptr);
    return t;
}
void drop_tex(Tex& t) {
    if (t.view) wgpuTextureViewRelease(t.view);
    if (t.tex) wgpuTextureRelease(t.tex);
    t = {};
}

/// What the display thread shows for one game frame: the made frames, then the game's (its own copy each)
struct OutSet {
    Tex frame[kMaxMult];
    int made = 0;   // (how many of frame[] are made frames; frame[made] is the game's)
    bool busy = false;
};

struct Job {
    OutSet* set;
    uint32_t sw, sh;
    int32_t ww, wh;
    int64_t at;
};

struct State {
    bool init = false;
    uint32_t w = 0, h = 0;
    WGPUComputePipeline copy = nullptr, luma = nullptr, flow = nullptr, median = nullptr, synth = nullptr;
    WGPUSampler sampler = nullptr;
    WGPUBuffer t_buf[kMaxMult] = {};
    Tex frames[2], lumas[2], flows[2];
    int cur = 0;
    bool have_prev = false;
    int64_t last_at = 0;
    OutSet sets[2];
    int next_set = 0;
    // the display thread's
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    bool thread_started = false;
} fg;

WGPUComputePipeline make_compute(const char* wgsl, const char* label) {
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = sv(wgsl);
    WGPUShaderModuleDescriptor md{};
    md.nextInChain = &src.chain;
    md.label = sv(label);
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(dev.device, &md);
    WGPUComputePipelineDescriptor cd{};
    cd.label = sv(label);
    cd.compute.module = mod;
    cd.compute.entryPoint = sv("main");
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(dev.device, &cd);   // (layout: from the shader)
    wgpuShaderModuleRelease(mod);
    return p;
}

void init_pipelines() {
    if (fg.init) return;
    fg.init = true;
    fg.copy = make_compute(kCopyWgsl, "rx framegen copy");
    fg.luma = make_compute(kLumaWgsl, "rx framegen luma");
    {   // RX_FRAMEGEN_RADIUS=<quarter-size texels> searched each way (default 8: +-32 pixels)
        const char* e = getenv("RX_FRAMEGEN_RADIUS");
        const int r = e ? std::clamp(atoi(e), 1, 16) : 8;
        std::string src = std::string("const RADIUS: i32 = ") + std::to_string(r) + ";\n" + kFlowWgsl;
        fg.flow = make_compute(src.c_str(), "rx framegen flow");
    }
    fg.median = make_compute(kMedianWgsl, "rx framegen median");
    fg.synth = make_compute(kSynthWgsl, "rx framegen synth");
    WGPUSamplerDescriptor sd{};
    sd.addressModeU = sd.addressModeV = sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = sd.minFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.lodMaxClamp = 1.0f;
    sd.maxAnisotropy = 1;
    fg.sampler = wgpuDeviceCreateSampler(dev.device, &sd);
    for (WGPUBuffer& b : fg.t_buf) {
        WGPUBufferDescriptor d{};
        d.size = 16;
        d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        b = wgpuDeviceCreateBuffer(dev.device, &d);
    }
}

void ensure_size(uint32_t w, uint32_t h) {
    if (fg.w == w && fg.h == h) return;
    {   // (the display thread is done with the old sets first)
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return fg.jobs.empty() && !fg.sets[0].busy && !fg.sets[1].busy; });
    }
    for (int i = 0; i < 2; i++) {
        drop_tex(fg.frames[i]);
        drop_tex(fg.lumas[i]);
        drop_tex(fg.flows[i]);
        for (Tex& t : fg.sets[i].frame) drop_tex(t);
    }
    fg.w = w, fg.h = h;
    const uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    const WGPUTextureUsage rw = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
    for (int i = 0; i < 2; i++) {
        fg.frames[i] = make_tex(w, h, WGPUTextureFormat_RGBA8Unorm, rw | WGPUTextureUsage_CopySrc, "rx framegen frame");
        fg.lumas[i] = make_tex(qw, qh, WGPUTextureFormat_RGBA16Float, rw, "rx framegen luma");
        fg.flows[i] = make_tex(cw, ch, WGPUTextureFormat_RGBA16Float, rw, "rx framegen flow");
        for (Tex& t : fg.sets[i].frame)
            t = make_tex(w, h, WGPUTextureFormat_RGBA8Unorm, rw | WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc,
                         "rx framegen shown");
    }
    fg.have_prev = false;
}

/// A compute pass: `pipe` over x * y groups of 8x8 with these bindings (textures, then a sampler / buffer)
void dispatch(WGPUCommandEncoder e, WGPUComputePipeline pipe, std::initializer_list<WGPUBindGroupEntry> entries,
              uint32_t w, uint32_t h) {
    std::vector<WGPUBindGroupEntry> be(entries);
    for (size_t i = 0; i < be.size(); i++) be[i].binding = (uint32_t)i;
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipe, 0);
    WGPUBindGroupDescriptor bd{};
    bd.layout = layout;
    bd.entryCount = be.size();
    bd.entries = be.data();
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(dev.device, &bd);
    WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(e, nullptr);
    wgpuComputePassEncoderSetPipeline(p, pipe);
    wgpuComputePassEncoderSetBindGroup(p, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(p, (w + 7) / 8, (h + 7) / 8, 1);
    wgpuComputePassEncoderEnd(p);
    wgpuComputePassEncoderRelease(p);
    wgpuBindGroupRelease(bg);
    wgpuBindGroupLayoutRelease(layout);
}
WGPUBindGroupEntry tex_entry(WGPUTextureView v) {
    WGPUBindGroupEntry e{};
    e.textureView = v;
    return e;
}

/// RX_FRAMEGEN_DUMP=<seconds> (checking): that long after interpolation started, the last frame, the first made one and the new
/// one as fg_prev.png, fg_mid.png, fg_cur.png (the GPU waited for)
void dump_png(WGPUTexture tex, uint32_t w, uint32_t h, const char* path) {
    const uint32_t row = (w * 4 + 255) & ~255u;
    WGPUBufferDescriptor bd{};
    bd.size = (uint64_t)row * h;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(dev.device, &bd);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUTexelCopyTextureInfo src{};
    src.texture = tex;
    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = buf;
    dst.layout.bytesPerRow = row;
    dst.layout.rowsPerImage = h;
    const WGPUExtent3D extent{w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    struct Done { volatile bool done = false; WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; } done;
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* u, void*) {
        static_cast<Done*>(u)->status = st;
        static_cast<Done*>(u)->done = true;
    };
    mc.userdata1 = &done;
    wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, bd.size, mc);
    while (!done.done) wgpuDevicePoll(dev.device, true, nullptr);
    if (done.status == WGPUMapAsyncStatus_Success) {
        write_png(path, static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(buf, 0, bd.size)), w, h, row);
        wgpuBufferUnmap(buf);
        hos_log("rx: frame interpolation: wrote %s", path);
    }
    wgpuBufferRelease(buf);
}

int multiplier() {
    static const int m = [] {
        const char* e = getenv("RX_FRAMEGEN_MULT");
        return e ? std::clamp(atoi(e), 2, kMaxMult) : 2;
    }();
    return m;
}

/// The display thread: each set's made frames, then the game's, spread over the game's recent frame time
void display_main() {
    hos_host_thread_register_self("rx display");
    int64_t interval = 16666667, last_at = 0;
    struct { int on_time, late, skipped; int64_t since; } stat{};   // (logged every 5 s)
    for (;;) {
        Job j;
        {
            std::unique_lock lk(fg.mu);
            fg.cv.wait(lk, [] { return !fg.jobs.empty(); });
            j = fg.jobs.front();
            fg.jobs.pop_front();
        }
        if (last_at) {
            const int64_t d = j.at - last_at;
            if (d > 0 && d < 100000000) interval = (interval * 7 + std::clamp<int64_t>(d, 4000000, 50000000)) / 8;
        }
        last_at = j.at;
        if (!stat.since) stat.since = j.at;
        if (j.at - stat.since > 5000000000LL) {
            log_limited("framegen_pace", 40, "rx: frame interpolation: %d frames shown on time, %d late (over 2 ms), %d "
                        "left out (the game's next frame came first)", stat.on_time, stat.late, stat.skipped);
            stat = {0, 0, 0, j.at};
        }
        const int shown = j.set->made + 1;
        const int64_t step = interval / shown;
        for (int k = 0; k <= j.set->made; k++) {
            wgpuTextureViewAddRef(j.set->frame[k].view);
            surface_present(j.set->frame[k].view, j.sw, j.sh, j.ww, j.wh, false);
            if (k == j.set->made) break;
            // the next one at its time; if the game's next frame comes first, the rest are left out (shown at once,
            // two presents back to back: the display shows one of them)
            const int64_t due = j.at + step * (k + 1);
            std::unique_lock lk(fg.mu);
            if (fg.cv.wait_until(lk, std::chrono::steady_clock::now() +
                                         std::chrono::nanoseconds(std::max<int64_t>(0, due - hos_now_ns())),
                                 [] { return !fg.jobs.empty(); })) {
                stat.skipped += j.set->made - k;
                break;
            }
            (hos_now_ns() - due > 2000000 ? stat.late : stat.on_time)++;
        }
        std::lock_guard lk(fg.mu);
        j.set->busy = false;
        fg.cv.notify_all();
    }
}

#if RX_WEB
// ---- the web (the WebGPU thread): the set being shown, frame by frame as they come due ----
struct WebShow {
    bool on = false;
    Job j{};
    int k = 0;
    int64_t step = 0;
} ws;
int64_t web_interval = 16666667, web_last_at = 0;
struct { int on_time, late, skipped; int64_t since; } wstat;   // (logged every 5 s)
void web_present_k(int64_t due) {
    if (due) (hos_now_ns() - due > 2000000 ? wstat.late : wstat.on_time)++;
    const Tex& t = ws.j.set->frame[ws.k];
    wgpuTextureViewAddRef(t.view);
    surface_present(t.view, ws.j.sw, ws.j.sh, ws.j.ww, ws.j.wh, false);
}
void web_finish() {
    ws.on = false;
    ws.j.set->busy = false;
    framegen_web_due = 0;
}
/// the game's next frame is here: the last set's frames not shown yet are left out (shown now, they would go out
/// back to back with the next set's first, and the browser shows one of the two)
void web_flush() {
    if (!ws.on) return;
    wstat.skipped += ws.j.set->made - ws.k;
    web_finish();
}
void web_start(const Job& j) {
    if (web_last_at) {
        const int64_t d = j.at - web_last_at;
        if (d > 0 && d < 100000000) web_interval = (web_interval * 7 + std::clamp<int64_t>(d, 4000000, 50000000)) / 8;
    }
    web_last_at = j.at;
    if (!wstat.since) wstat.since = j.at;
    if (j.at - wstat.since > 5000000000LL) {
        log_limited("framegen_pace", 40, "rx: frame interpolation: %d frames shown on time, %d late (over 2 ms), %d "
                    "left out (the WebGPU thread busy when they were due)", wstat.on_time, wstat.late, wstat.skipped);
        wstat = {0, 0, 0, j.at};
    }
    ws.on = true;
    ws.j = j;
    ws.k = 0;
    ws.step = web_interval / (j.set->made + 1);
    web_present_k(0);
    if (!j.set->made) web_finish();
    else framegen_web_due = j.at + ws.step;
}
#endif
}  // namespace

#if RX_WEB
int64_t framegen_web_due;

int64_t framegen_web_tick() {
    static bool in_tick;   // (a present yields to the event loop)
    if (!ws.on || in_tick) return -1;
    const int64_t due = ws.j.at + ws.step * (ws.k + 1), now = hos_now_ns();
    if (now < due) return due - now;
    in_tick = true;
    ws.k++;
    web_present_k(due);
    in_tick = false;
    if (ws.k >= ws.j.set->made) {
        web_finish();
        return -1;
    }
    framegen_web_due = due + ws.step;
    return std::max<int64_t>(0, framegen_web_due - hos_now_ns());
}
#endif

bool framegen_frame(WGPUTextureView view, uint32_t sw, uint32_t sh, int32_t ww, int32_t wh) {
#if RX_WEB && !RX_WEB_MT
    (void)view, (void)sw, (void)sh, (void)ww, (void)wh;
    log_limited("framegen_web", 1, "rx: frame interpolation: the threaded web build only (index_mt.html)");
    return false;
#else
    if (!sw || !sh) return false;
#if RX_WEB
    web_flush();   // (the set still being shown: done now; this thread is the one that would free it)
#endif
    init_pipelines();
    ensure_size(sw, sh);
    if (!fg.thread_started) {
        fg.thread_started = true;
#if !RX_WEB
        std::thread(display_main).detach();
#endif
        hos_log("rx: frame interpolation on: %dx the game's frames (made from its last two, shown evenly)", multiplier());
    }

    const int64_t now = hos_now_ns();
    const bool gap = fg.last_at && now - fg.last_at > 100000000;   // (a loading hitch: nothing to blend)
    fg.last_at = now;
    // a set the display thread is not showing
    OutSet* set;
    static int64_t waited, frames, since;
    const int64_t w0 = hos_now_ns();
    {
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return !fg.sets[fg.next_set].busy; });
        set = &fg.sets[fg.next_set];
        set->busy = true;
        fg.next_set ^= 1;
    }
    waited += hos_now_ns() - w0;
    frames++;
    if (!since) since = w0;
#if !RX_WEB   // (the web's sets are free by now: web_flush)
    if (w0 - since > 5000000000LL) {   // (the submit thread's wait for a set the display thread is done with)
        log_limited("framegen_wait", 40, "rx: frame interpolation: %.2f ms a frame waiting for the display thread",
                    waited / 1e6 / frames);
        waited = frames = 0;
        since = w0;
    }
#endif
    const int cur = fg.cur, prev = cur ^ 1;
    const int m = multiplier();
    const bool blend = fg.have_prev && !gap;
    for (int k = 1; k < m && blend; k++) {
        const float t[4] = {(float)k / (float)m, 0, 0, 0};
        wgpuQueueWriteBuffer(dev.queue, fg.t_buf[k], 0, t, sizeof t);
    }
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    const uint32_t qw = (sw + 3) / 4, qh = (sh + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    dispatch(e, fg.copy, {tex_entry(view), tex_entry(fg.frames[cur].view)}, sw, sh);
    dispatch(e, fg.luma, {tex_entry(fg.frames[cur].view), tex_entry(fg.lumas[cur].view)}, qw, qh);
    set->made = 0;
    if (blend) {
        dispatch(e, fg.flow, {tex_entry(fg.lumas[prev].view), tex_entry(fg.lumas[cur].view), tex_entry(fg.flows[0].view)},
                 cw, ch);
        dispatch(e, fg.median, {tex_entry(fg.flows[0].view), tex_entry(fg.flows[1].view)}, cw, ch);
        for (int k = 1; k < m; k++) {
            WGPUBindGroupEntry s{}, b{};
            s.sampler = fg.sampler;
            b.buffer = fg.t_buf[k];
            b.size = 16;
            dispatch(e, fg.synth,
                     {tex_entry(fg.frames[prev].view), tex_entry(fg.frames[cur].view), tex_entry(fg.flows[1].view), s,
                      tex_entry(set->frame[k - 1].view), b},
                     sw, sh);
        }
        set->made = m - 1;
    }
    WGPUTexelCopyTextureInfo src{}, dst{};
    src.texture = fg.frames[cur].tex;
    dst.texture = set->frame[set->made].tex;
    const WGPUExtent3D extent{sw, sh, 1};
    wgpuCommandEncoderCopyTextureToTexture(e, &src, &dst, &extent);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    wgpuTextureViewRelease(view);
    if (static const double dump_at = getenv("RX_FRAMEGEN_DUMP") ? atof(getenv("RX_FRAMEGEN_DUMP")) : 0; dump_at && blend) {
        static const int64_t t0 = now;
        static bool dumped;
        if (!dumped && now - t0 > (int64_t)(dump_at * 1e9)) {
            dumped = true;
            dump_png(fg.frames[prev].tex, sw, sh, "fg_prev.png");
            dump_png(set->frame[0].tex, sw, sh, "fg_mid.png");
            dump_png(fg.frames[cur].tex, sw, sh, "fg_cur.png");
        }
    }
    fg.have_prev = true;
    fg.cur = prev;
#if RX_WEB
    web_start({set, sw, sh, ww, wh, now});
#else
    {
        std::lock_guard lk(fg.mu);
        fg.jobs.push_back({set, sw, sh, ww, wh, now});
        fg.cv.notify_all();
    }
#endif
    return true;
#endif
}
}  // namespace rx
