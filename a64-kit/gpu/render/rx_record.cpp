// Deferred command encoding: the render thread records a batch's encoder commands (render and compute passes,
// copies) into a plain list, and the submit thread replays them into a WebGPU command encoder just before it finishes
// it. wgpu's encoding calls (SetBindGroup above all) were ~10% of the render thread, the frame rate's limit; the submit
// thread had room. Objects the render thread may release before the replay are held (AddRef) until it is done: texture
// views and textures (passes, copies) and compute bind groups (released after their dispatch). Pipelines and buffers
// live as long as the renderer, and the cached render bind groups are only released after a flush and a wait
// (trim_bind_groups): an AddRef each was as costly as the call it deferred. RX_DEFER_ENCODING=0: the calls go to the
// encoder at once, as before.
#include "rx_internal.h"

#include <algorithm>
#include <map>

namespace rx {
namespace {

enum Op : uint64_t {
    kBeginRenderPass = 1, kEndRenderPass, kSetViewport, kSetPipeline, kSetBindGroup, kSetVertexBuffer, kSetIndexBuffer,
    kSetScissor, kSetBlendConstant, kSetStencilReference, kDraw, kDrawIndexed, kCopyBufferToBuffer,
    kCopyTextureToTexture, kBeginComputePass, kComputeSetPipeline, kComputeSetBindGroup, kDispatch, kEndComputePass,
    kCopyTextureToBuffer, kComputeSetBindGroupDesc,
};
enum RefKind : uint8_t { kView, kTexture, kBuffer, kRenderPipeline, kComputePipeline, kBindGroup };

WGPURenderPassEncoder const kFakePass = reinterpret_cast<WGPURenderPassEncoder>(uintptr_t{0x10});
WGPUComputePassEncoder const kFakeCompute = reinterpret_cast<WGPUComputePassEncoder>(uintptr_t{0x20});
WGPUCommandEncoder const kFakeEncoder = reinterpret_cast<WGPUCommandEncoder>(uintptr_t{0x30});

}   // namespace

struct Recording {
    std::vector<uint64_t> words;
    std::vector<size_t> pass_ends;   // where each pass ended (in words): the places a replay can be split
    std::vector<std::pair<RefKind, void*>> refs;   // released after the replay
};

namespace {
Recording* open_rec;   // the open batch's commands (render thread)
size_t last_words, last_refs, last_passes;   // (the last batch's sizes: a new one starts at them instead of growing)

Recording& rec() {
    if (!open_rec) {
        open_rec = new Recording();
        open_rec->words.reserve(last_words);
        open_rec->refs.reserve(last_refs);
        open_rec->pass_ends.reserve(last_passes);
    }
    return *open_rec;
}
void put(uint64_t w) { rec().words.push_back(w); }
// RX_REC_STATS=1: the commands recorded, by kind, logged every 600 frames (per frame): what each draw costs the
// encoding threads in wgpu calls
uint64_t op_counts[32];
const char* const op_names[32] = {"", "begin pass", "end pass", "viewport", "pipeline", "bind group", "vertex buffer",
                                  "index buffer", "scissor", "blend constant", "stencil ref", "draw", "draw indexed",
                                  "copy b2b", "copy t2t", "begin compute", "compute pipeline", "compute bind group",
                                  "dispatch", "end compute", "copy t2b", "compute bind group desc"};
void put_op(Op o) {
    static const bool stats = env_flag("RX_REC_STATS");
    if (stats) {
        op_counts[o & 31]++;
        static int64_t since = frame_number;
        if (frame_number - since >= 600) {
            std::string line;
            for (int i = 1; i < 32; i++)
                if (op_counts[i]) {
                    char b[64];
                    std::snprintf(b, sizeof b, " %s %.1f;", op_names[i], (double)op_counts[i] / (double)(frame_number - since));
                    line += b;
                }
            hos_log("rx: recorded per frame:%s", line.c_str());
            std::memset(op_counts, 0, sizeof op_counts);
            since = frame_number;
        }
    }
    put(o);
}
void put_ptr(const void* p) { put(reinterpret_cast<uintptr_t>(p)); }
uint64_t bits(float f) { return std::bit_cast<uint32_t>(f); }
uint64_t bits(double d) { return std::bit_cast<uint64_t>(d); }
void hold(RefKind k, void* h) {
    if (!h) return;
    switch (k) {
    case kView: wgpuTextureViewAddRef(static_cast<WGPUTextureView>(h)); break;
    case kTexture: wgpuTextureAddRef(static_cast<WGPUTexture>(h)); break;
    case kBuffer: wgpuBufferAddRef(static_cast<WGPUBuffer>(h)); break;
    case kRenderPipeline: wgpuRenderPipelineAddRef(static_cast<WGPURenderPipeline>(h)); break;
    case kComputePipeline: wgpuComputePipelineAddRef(static_cast<WGPUComputePipeline>(h)); break;
    case kBindGroup: wgpuBindGroupAddRef(static_cast<WGPUBindGroup>(h)); break;
    }
    rec().refs.push_back({k, h});
}
void drop(RefKind k, void* h) {
    switch (k) {
    case kView: wgpuTextureViewRelease(static_cast<WGPUTextureView>(h)); break;
    case kTexture: wgpuTextureRelease(static_cast<WGPUTexture>(h)); break;
    case kBuffer: wgpuBufferRelease(static_cast<WGPUBuffer>(h)); break;
    case kRenderPipeline: wgpuRenderPipelineRelease(static_cast<WGPURenderPipeline>(h)); break;
    case kComputePipeline: wgpuComputePipelineRelease(static_cast<WGPUComputePipeline>(h)); break;
    case kBindGroup: wgpuBindGroupRelease(static_cast<WGPUBindGroup>(h)); break;
    }
}
template <class T> T as(uint64_t w) { return reinterpret_cast<T>(static_cast<uintptr_t>(w)); }
float as_float(uint64_t w) { return std::bit_cast<float>(static_cast<uint32_t>(w)); }
double as_double(uint64_t w) { return std::bit_cast<double>(w); }
void put_copy_info(const WGPUTexelCopyTextureInfo& t) {
    hold(kTexture, t.texture);
    put_ptr(t.texture);
    put(t.mipLevel);
    put(uint64_t{t.origin.x} | uint64_t{t.origin.y} << 32);
    put(uint64_t{t.origin.z} | uint64_t{static_cast<uint32_t>(t.aspect)} << 32);
}
WGPUTexelCopyTextureInfo get_copy_info(const uint64_t*& w) {
    WGPUTexelCopyTextureInfo t{};
    t.texture = as<WGPUTexture>(*w++);
    t.mipLevel = static_cast<uint32_t>(*w++);
    t.origin.x = static_cast<uint32_t>(*w);
    t.origin.y = static_cast<uint32_t>(*w++ >> 32);
    t.origin.z = static_cast<uint32_t>(*w);
    t.aspect = static_cast<WGPUTextureAspect>(*w++ >> 32);
    return t;
}
}   // namespace

bool encoding_deferred() {
    static const bool on = [] {
        const char* e = getenv("RX_DEFER_ENCODING");
        return e ? std::atoi(e) != 0 : !RX_WEB || RX_WEB_MT;   // (the threaded web build records, as natively)
    }();
    return on;
}

WGPUCommandEncoder fake_encoder() { return kFakeEncoder; }

Recording* recording_take() {
    Recording* r = open_rec;
    open_rec = nullptr;
    if (r) {
        last_words = r->words.size();
        last_refs = r->refs.size();
        last_passes = r->pass_ends.size();
    }
    return r;
}

std::vector<std::pair<size_t, size_t>> recording_parts(const Recording* r, uint32_t parts, size_t min_words) {
    std::vector<std::pair<size_t, size_t>> out;
    const size_t total = r->words.size();
    if (parts > 1 && total >= 2 * min_words) {
        // each cut at the pass end nearest its share (k * total / parts): cutting at the first one past the target left
        // the last part the remainder (5.1, 4.3 and 0.4 ms a frame at the castle)
        size_t from = 0, k = 1;
        const std::vector<size_t>& e = r->pass_ends;
        for (size_t i = 0; i < e.size() && k < parts; i++) {
            const size_t share = total * k / parts;
            if (e[i] < share && i + 1 < e.size() && e[i + 1] <= share) continue;   // (a later one is nearer)
            size_t cut = e[i];
            if (cut < share && i + 1 < e.size() && e[i + 1] - share < share - cut) cut = e[++i];
            if (cut - from < min_words || total - cut < min_words) continue;
            out.push_back({from, cut});
            from = cut;
            k++;
        }
        out.push_back({from, total});
    } else {
        out.push_back({0, total});
    }
    return out;
}

// ---- RX_GPU_TIMING=1 (profiling): timestamps around each pass, summed by the pass's tag ----

struct GpuTimer {
    static constexpr uint32_t kQueries = 1024;   // two a pass
    WGPUQuerySet set = nullptr;
    WGPUBuffer resolve = nullptr, read = nullptr;
    struct Pass {
        uint64_t va, desc;   // desc 0: a compute pass
        uint32_t draws;
    };
    std::vector<Pass> passes;
    /// The next pass's timestamp writes, or null when the timer is full.
    const WGPUPassTimestampWrites* begin(uint64_t va, uint64_t desc) {
        if (passes.size() * 2 + 2 > kQueries) return nullptr;
        tw.querySet = set;
        tw.beginningOfPassWriteIndex = static_cast<uint32_t>(passes.size() * 2);
        tw.endOfPassWriteIndex = tw.beginningOfPassWriteIndex + 1;
        passes.push_back({va, desc, 0});
        return &tw;
    }
    WGPUPassTimestampWrites tw{};
};

namespace {
uint64_t tag_va, tag_desc;   // the next render pass's (render thread)
std::mutex timer_mu;
std::vector<GpuTimer*> timer_pool;
struct TimeStat {
    uint64_t ns = 0, passes = 0, draws = 0;
};
std::map<std::pair<uint64_t, uint64_t>, TimeStat> time_stats;
uint64_t time_total_ns = 0;
int64_t time_since = 0, time_frame0 = 0;

void gpu_timer_done(GpuTimer* t, WGPUMapAsyncStatus status) {
    const size_t n = t->passes.size() * 2;
    if (status == WGPUMapAsyncStatus_Success) {
#if RX_WEB
        static const double period = 1.0;   // (the browser's timestamps are nanoseconds already: no period call there)
#else
        static const double period = wgpuQueueGetTimestampPeriod(dev.queue);
#endif
        const auto* ts = static_cast<const uint64_t*>(wgpuBufferGetConstMappedRange(t->read, 0, n * 8));
        std::lock_guard lk(timer_mu);
        for (size_t i = 0; ts && i < t->passes.size(); i++) {
            const uint64_t a = ts[2 * i], b = ts[2 * i + 1];
            const uint64_t ns = b > a ? static_cast<uint64_t>((b - a) * period) : 0;
            if (ns > 1000000000) continue;   // (a pass the queue split oddly: not a real time)
            TimeStat& s = time_stats[{t->passes[i].va, t->passes[i].desc}];
            s.ns += ns;
            s.passes++;
            s.draws += t->passes[i].draws;
            time_total_ns += ns;
        }
        wgpuBufferUnmap(t->read);
        const int64_t now = hos_now_ns();
        if (!time_since) {
            time_since = now;
            time_frame0 = frame_number;
        } else if (now - time_since > 5000000000) {
            const double frames = std::max<double>(1.0, static_cast<double>(frame_number - time_frame0));
            std::vector<std::pair<std::pair<uint64_t, uint64_t>, TimeStat>> v(time_stats.begin(), time_stats.end());
            std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.second.ns > y.second.ns; });
            hos_log("rx: gpu time: %.2f ms a frame in passes over %.0f frames (%zu kinds)", time_total_ns / 1e6 / frames,
                    frames, v.size());
            for (size_t i = 0; i < v.size() && i < 30; i++) {
                const auto& [key, s] = v[i];
                const uint64_t d = key.second;
                if (!d)
                    hos_log("rx: gpu time %5.2f ms  x%-5.1f compute          %6.1f dispatches", s.ns / 1e6 / frames,
                            s.passes / frames, static_cast<double>(s.draws) / s.passes);
                else
                    hos_log("rx: gpu time %5.2f ms  x%-5.1f %4llux%-4llu c%llu f%-3llu d%-3llu %6.1f draws  va %llx",
                            s.ns / 1e6 / frames, s.passes / frames, (unsigned long long)(d >> 48 & 0x7FFF),
                            (unsigned long long)(d >> 32 & 0xFFFF), (unsigned long long)(d >> 24 & 0xF),
                            (unsigned long long)(d >> 8 & 0xFFFF), (unsigned long long)(d & 0xFF),
                            static_cast<double>(s.draws) / s.passes, (unsigned long long)key.first);
            }
            time_stats.clear();
            time_total_ns = 0;
            time_since = now;
            time_frame0 = frame_number;
        }
    }
    t->passes.clear();
    std::lock_guard lk(timer_mu);
    timer_pool.push_back(t);
}
}   // namespace

bool gpu_timing() {
    static const bool on = env_flag("RX_GPU_TIMING") && dev.timestamps && encoding_deferred();
    return on;
}

void ce_pass_tag(uint64_t va, uint64_t desc) {
    tag_va = va;
    tag_desc = desc;
}

GpuTimer* gpu_timer_take() {
    {
        std::lock_guard lk(timer_mu);
        if (!timer_pool.empty()) {
            GpuTimer* t = timer_pool.back();
            timer_pool.pop_back();
            return t;
        }
    }
    auto* t = new GpuTimer();
    WGPUQuerySetDescriptor qd{};
    qd.type = WGPUQueryType_Timestamp;
    qd.count = GpuTimer::kQueries;
    t->set = wgpuDeviceCreateQuerySet(dev.device, &qd);
    WGPUBufferDescriptor bd{};
    bd.size = GpuTimer::kQueries * 8;
    bd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
    t->resolve = wgpuDeviceCreateBuffer(dev.device, &bd);
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    t->read = wgpuDeviceCreateBuffer(dev.device, &bd);
    return t;
}

void gpu_timer_resolve(GpuTimer* t, WGPUCommandEncoder e) {
    if (!t || t->passes.empty()) return;
    const uint32_t n = static_cast<uint32_t>(t->passes.size() * 2);
    wgpuCommandEncoderResolveQuerySet(e, t->set, 0, n, t->resolve, 0);
    wgpuCommandEncoderCopyBufferToBuffer(e, t->resolve, 0, t->read, 0, uint64_t{n} * 8);
}

void gpu_timer_read(GpuTimer* t) {
    if (!t) return;
    if (t->passes.empty()) {
        std::lock_guard lk(timer_mu);
        timer_pool.push_back(t);
        return;
    }
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        gpu_timer_done(static_cast<GpuTimer*>(u1), status);
    };
    mc.userdata1 = t;
    wgpuBufferMapAsync(t->read, WGPUMapMode_Read, 0, t->passes.size() * 16, mc);
}

void recording_replay(Recording* r, WGPUCommandEncoder e) {
    recording_replay(r, e, 0, r->words.size());
}

void recording_replay(Recording* r, WGPUCommandEncoder e, size_t from, size_t to, GpuTimer* t) {
    WGPURenderPassEncoder pass = nullptr;
    WGPUComputePassEncoder compute = nullptr;
    GpuTimer::Pass* timed = nullptr;   // the open pass's entry in t (a pointer into a vector reserved below)
    if (t) t->passes.reserve(GpuTimer::kQueries / 2);
    const bool tagged = gpu_timing();
    const uint64_t* w = r->words.data() + from;
    const uint64_t* end = r->words.data() + to;
#if RX_WEB_MT
    uint32_t ops = 0;
#endif
    while (w < end) {
#if RX_WEB_MT
        serve_gpu_calls_now_if_any();   // (creations other threads wait for: not after the whole batch)
        // (frame interpolation: a made frame shown when it is due, not after the batch: rx_framegen.cpp)
        if (framegen_web_due && !(++ops & 31) && hos_now_ns() >= framegen_web_due) framegen_web_tick();
#endif
        switch (static_cast<Op>(*w++)) {
        case kBeginRenderPass: {
            uint64_t va = 0, desc = 0;
            if (tagged) {
                va = *w++;
                desc = *w++;
            }
            const uint32_t ncolor = static_cast<uint32_t>(*w++);
            WGPURenderPassColorAttachment colors[8]{};
            for (uint32_t i = 0; i < ncolor; i++) {
                WGPURenderPassColorAttachment& c = colors[i];
                c.view = as<WGPUTextureView>(*w++);
                c.depthSlice = static_cast<uint32_t>(*w);
                c.loadOp = static_cast<WGPULoadOp>(*w >> 32 & 0xFFFF);
                c.storeOp = static_cast<WGPUStoreOp>(*w++ >> 48);
                c.clearValue = {as_double(w[0]), as_double(w[1]), as_double(w[2]), as_double(w[3])};
                w += 4;
            }
            WGPURenderPassDepthStencilAttachment depth{};
            const bool has_depth = *w++ != 0;
            if (has_depth) {
                depth.view = as<WGPUTextureView>(*w++);
                depth.depthLoadOp = static_cast<WGPULoadOp>(*w & 0xFF);
                depth.depthStoreOp = static_cast<WGPUStoreOp>(*w >> 8 & 0xFF);
                depth.stencilLoadOp = static_cast<WGPULoadOp>(*w >> 16 & 0xFF);
                depth.stencilStoreOp = static_cast<WGPUStoreOp>(*w >> 24 & 0xFF);
                depth.depthReadOnly = static_cast<WGPUBool>(*w >> 32 & 1);
                depth.stencilReadOnly = static_cast<WGPUBool>(*w++ >> 33 & 1);
                depth.depthClearValue = as_float(*w);
                depth.stencilClearValue = static_cast<uint32_t>(*w++ >> 32);
            }
            WGPURenderPassDescriptor pd{};
            pd.colorAttachmentCount = ncolor;
            pd.colorAttachments = colors;
            pd.depthStencilAttachment = has_depth ? &depth : nullptr;
            if (t) {
                pd.timestampWrites = t->begin(va, desc | 1ull << 63);
                timed = pd.timestampWrites ? &t->passes.back() : nullptr;
            }
            pass = wgpuCommandEncoderBeginRenderPass(e, &pd);
            break;
        }
        case kEndRenderPass:
            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
            pass = nullptr;
            timed = nullptr;
            break;
        case kSetViewport:
            wgpuRenderPassEncoderSetViewport(pass, as_float(w[0]), as_float(w[1]), as_float(w[2]), as_float(w[3]),
                                             as_float(w[4]), as_float(w[5]));
            w += 6;
            break;
        case kSetPipeline: wgpuRenderPassEncoderSetPipeline(pass, as<WGPURenderPipeline>(*w++)); break;
        case kSetBindGroup: {
            const uint32_t index = static_cast<uint32_t>(w[0]), n = static_cast<uint32_t>(w[0] >> 32);
            WGPUBindGroup g = as<WGPUBindGroup>(w[1]);
            w += 2;
            uint32_t offsets[64];
            for (uint32_t i = 0; i < n; i++) offsets[i] = static_cast<uint32_t>(*w++);
            wgpuRenderPassEncoderSetBindGroup(pass, index, g, n, offsets);
            break;
        }
        case kSetVertexBuffer:
            wgpuRenderPassEncoderSetVertexBuffer(pass, static_cast<uint32_t>(w[0]), as<WGPUBuffer>(w[1]), w[2], w[3]);
            w += 4;
            break;
        case kSetIndexBuffer:
            wgpuRenderPassEncoderSetIndexBuffer(pass, as<WGPUBuffer>(w[0]), static_cast<WGPUIndexFormat>(w[1]), w[2],
                                                w[3]);
            w += 4;
            break;
        case kSetScissor:
            wgpuRenderPassEncoderSetScissorRect(pass, static_cast<uint32_t>(w[0]), static_cast<uint32_t>(w[0] >> 32),
                                                static_cast<uint32_t>(w[1]), static_cast<uint32_t>(w[1] >> 32));
            w += 2;
            break;
        case kSetBlendConstant: {
            const WGPUColor c{as_double(w[0]), as_double(w[1]), as_double(w[2]), as_double(w[3])};
            w += 4;
            wgpuRenderPassEncoderSetBlendConstant(pass, &c);
            break;
        }
        case kSetStencilReference: wgpuRenderPassEncoderSetStencilReference(pass, static_cast<uint32_t>(*w++)); break;
        case kDraw:
            if (timed) timed->draws++;
            wgpuRenderPassEncoderDraw(pass, static_cast<uint32_t>(w[0]), static_cast<uint32_t>(w[0] >> 32),
                                      static_cast<uint32_t>(w[1]), static_cast<uint32_t>(w[1] >> 32));
            w += 2;
            break;
        case kDrawIndexed:
            if (timed) timed->draws++;
            wgpuRenderPassEncoderDrawIndexed(pass, static_cast<uint32_t>(w[0]), static_cast<uint32_t>(w[0] >> 32),
                                             static_cast<uint32_t>(w[1]), static_cast<int32_t>(w[1] >> 32),
                                             static_cast<uint32_t>(w[2]));
            w += 3;
            break;
        case kCopyBufferToBuffer:
            wgpuCommandEncoderCopyBufferToBuffer(e, as<WGPUBuffer>(w[0]), w[1], as<WGPUBuffer>(w[2]), w[3], w[4]);
            w += 5;
            break;
        case kCopyTextureToTexture: {
            const WGPUTexelCopyTextureInfo src = get_copy_info(w), dst = get_copy_info(w);
            const WGPUExtent3D extent{static_cast<uint32_t>(w[0]), static_cast<uint32_t>(w[0] >> 32),
                                      static_cast<uint32_t>(w[1])};
            w += 2;
            wgpuCommandEncoderCopyTextureToTexture(e, &src, &dst, &extent);
            break;
        }
        case kCopyTextureToBuffer: {
            const WGPUTexelCopyTextureInfo src = get_copy_info(w);
            WGPUTexelCopyBufferInfo dst{};
            dst.buffer = as<WGPUBuffer>(w[0]);
            dst.layout.offset = w[1];
            dst.layout.bytesPerRow = static_cast<uint32_t>(w[2]);
            dst.layout.rowsPerImage = static_cast<uint32_t>(w[2] >> 32);
            const WGPUExtent3D extent{static_cast<uint32_t>(w[3]), static_cast<uint32_t>(w[3] >> 32),
                                      static_cast<uint32_t>(w[4])};
            w += 5;
            wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &extent);
            break;
        }
        case kBeginComputePass: {
            WGPUComputePassDescriptor cd{};
            if (t) {
                cd.timestampWrites = t->begin(0, 0);
                timed = cd.timestampWrites ? &t->passes.back() : nullptr;
            }
            compute = wgpuCommandEncoderBeginComputePass(e, &cd);
            break;
        }
        case kComputeSetPipeline: wgpuComputePassEncoderSetPipeline(compute, as<WGPUComputePipeline>(*w++)); break;
        case kComputeSetBindGroupDesc: {   // (made here, from what cp_set_bind_group_desc recorded)
            const uint32_t index = static_cast<uint32_t>(w[0]), n = static_cast<uint32_t>(w[0] >> 32);
            WGPUBindGroupDescriptor bd{};
            bd.layout = as<WGPUBindGroupLayout>(w[1]);
            w += 2;
            WGPUBindGroupEntry entries[64]{};
            for (uint32_t i = 0; i < n && i < 64; i++, w += 6) {
                entries[i].binding = static_cast<uint32_t>(w[0]);
                entries[i].buffer = as<WGPUBuffer>(w[1]);
                entries[i].offset = w[2];
                entries[i].size = w[3];
                entries[i].sampler = as<WGPUSampler>(w[4]);
                entries[i].textureView = as<WGPUTextureView>(w[5]);
            }
            bd.entryCount = std::min<uint32_t>(n, 64);
            bd.entries = entries;
            WGPUBindGroup g = wgpuDeviceCreateBindGroup(dev.device, &bd);
            wgpuComputePassEncoderSetBindGroup(compute, index, g, 0, nullptr);
            wgpuBindGroupRelease(g);
            break;
        }
        case kComputeSetBindGroup: {
            const uint32_t index = static_cast<uint32_t>(w[0]), n = static_cast<uint32_t>(w[0] >> 32);
            WGPUBindGroup g = as<WGPUBindGroup>(w[1]);
            w += 2;
            uint32_t offsets[64];
            for (uint32_t i = 0; i < n; i++) offsets[i] = static_cast<uint32_t>(*w++);
            wgpuComputePassEncoderSetBindGroup(compute, index, g, n, offsets);
            break;
        }
        case kDispatch:
            if (timed) timed->draws++;
            wgpuComputePassEncoderDispatchWorkgroups(compute, static_cast<uint32_t>(w[0]), static_cast<uint32_t>(w[1]),
                                                     static_cast<uint32_t>(w[2]));
            w += 3;
            break;
        case kEndComputePass:
            wgpuComputePassEncoderEnd(compute);
            wgpuComputePassEncoderRelease(compute);
            compute = nullptr;
            timed = nullptr;
            break;
        default:
            hos_log("rx: recording: unknown op %llu", static_cast<unsigned long long>(w[-1]));
            return;
        }
    }
}

void recording_free(Recording* r) {
    if (!r) return;
    for (auto& [k, h] : r->refs) drop(k, h);
    delete r;
}

// ---- the calls: recorded when deferred, made at once otherwise ----

WGPURenderPassEncoder ce_begin_render_pass(WGPUCommandEncoder e, const WGPURenderPassDescriptor* pd) {
    if (!encoding_deferred()) return wgpuCommandEncoderBeginRenderPass(e, pd);
    put_op(kBeginRenderPass);
    if (gpu_timing()) {
        put(tag_va);
        put(tag_desc);
        tag_va = tag_desc = 0;
    }
    put(pd->colorAttachmentCount);
    for (uint32_t i = 0; i < pd->colorAttachmentCount; i++) {
        const WGPURenderPassColorAttachment& c = pd->colorAttachments[i];
        hold(kView, c.view);
        put_ptr(c.view);
        put(uint64_t{c.depthSlice} | uint64_t{static_cast<uint16_t>(c.loadOp)} << 32 |
            uint64_t{static_cast<uint16_t>(c.storeOp)} << 48);
        put(bits(c.clearValue.r));
        put(bits(c.clearValue.g));
        put(bits(c.clearValue.b));
        put(bits(c.clearValue.a));
    }
    const WGPURenderPassDepthStencilAttachment* d = pd->depthStencilAttachment;
    put(d ? 1 : 0);
    if (d) {
        hold(kView, d->view);
        put_ptr(d->view);
        put(uint64_t{static_cast<uint8_t>(d->depthLoadOp)} | uint64_t{static_cast<uint8_t>(d->depthStoreOp)} << 8 |
            uint64_t{static_cast<uint8_t>(d->stencilLoadOp)} << 16 |
            uint64_t{static_cast<uint8_t>(d->stencilStoreOp)} << 24 | uint64_t{d->depthReadOnly ? 1u : 0u} << 32 |
            uint64_t{d->stencilReadOnly ? 1u : 0u} << 33);
        put(bits(d->depthClearValue) | uint64_t{d->stencilClearValue} << 32);
    }
    return kFakePass;
}

void rp_end(WGPURenderPassEncoder p) {
    if (!encoding_deferred()) {
        wgpuRenderPassEncoderEnd(p);
        wgpuRenderPassEncoderRelease(p);
        return;
    }
    put_op(kEndRenderPass);
    rec().pass_ends.push_back(rec().words.size());
}

void rp_set_viewport(WGPURenderPassEncoder p, float x, float y, float w, float h, float zmin, float zmax) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetViewport(p, x, y, w, h, zmin, zmax);
    put_op(kSetViewport);
    for (float f : {x, y, w, h, zmin, zmax}) put(bits(f));
}

void rp_set_pipeline(WGPURenderPassEncoder p, WGPURenderPipeline pipeline) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetPipeline(p, pipeline);
    put_op(kSetPipeline);
    put_ptr(pipeline);
}

void rp_set_bind_group(WGPURenderPassEncoder p, uint32_t index, WGPUBindGroup g, size_t n, const uint32_t* offsets) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetBindGroup(p, index, g, n, offsets);
    put_op(kSetBindGroup);
    put(uint64_t{index} | uint64_t{static_cast<uint32_t>(n)} << 32);
    put_ptr(g);
    for (size_t i = 0; i < n; i++) put(offsets[i]);
}

void rp_set_vertex_buffer(WGPURenderPassEncoder p, uint32_t slot, WGPUBuffer b, uint64_t offset, uint64_t size) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetVertexBuffer(p, slot, b, offset, size);
    put_op(kSetVertexBuffer);
    put(slot);
    put_ptr(b);
    put(offset);
    put(size);
}

void rp_set_index_buffer(WGPURenderPassEncoder p, WGPUBuffer b, WGPUIndexFormat format, uint64_t offset,
                         uint64_t size) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetIndexBuffer(p, b, format, offset, size);
    put_op(kSetIndexBuffer);
    put_ptr(b);
    put(static_cast<uint64_t>(format));
    put(offset);
    put(size);
}

void rp_set_scissor(WGPURenderPassEncoder p, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetScissorRect(p, x, y, w, h);
    put_op(kSetScissor);
    put(uint64_t{x} | uint64_t{y} << 32);
    put(uint64_t{w} | uint64_t{h} << 32);
}

void rp_set_blend_constant(WGPURenderPassEncoder p, const WGPUColor* c) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetBlendConstant(p, c);
    put_op(kSetBlendConstant);
    put(bits(c->r));
    put(bits(c->g));
    put(bits(c->b));
    put(bits(c->a));
}

void rp_set_stencil_reference(WGPURenderPassEncoder p, uint32_t ref) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderSetStencilReference(p, ref);
    put_op(kSetStencilReference);
    put(ref);
}

void rp_draw(WGPURenderPassEncoder p, uint32_t vertices, uint32_t instances, uint32_t first, uint32_t first_instance) {
    if (!encoding_deferred()) return wgpuRenderPassEncoderDraw(p, vertices, instances, first, first_instance);
    put_op(kDraw);
    put(uint64_t{vertices} | uint64_t{instances} << 32);
    put(uint64_t{first} | uint64_t{first_instance} << 32);
}

void rp_draw_indexed(WGPURenderPassEncoder p, uint32_t indices, uint32_t instances, uint32_t first_index,
                     int32_t base_vertex, uint32_t first_instance) {
    if (!encoding_deferred())
        return wgpuRenderPassEncoderDrawIndexed(p, indices, instances, first_index, base_vertex, first_instance);
    put_op(kDrawIndexed);
    put(uint64_t{indices} | uint64_t{instances} << 32);
    put(uint64_t{first_index} | uint64_t{static_cast<uint32_t>(base_vertex)} << 32);
    put(first_instance);
}

void ce_copy_buffer_to_buffer(WGPUCommandEncoder e, WGPUBuffer src, uint64_t src_offset, WGPUBuffer dst,
                              uint64_t dst_offset, uint64_t size) {
    if (!encoding_deferred()) return wgpuCommandEncoderCopyBufferToBuffer(e, src, src_offset, dst, dst_offset, size);
    put_op(kCopyBufferToBuffer);
    put_ptr(src);
    put(src_offset);
    put_ptr(dst);
    put(dst_offset);
    put(size);
}

void ce_copy_texture_to_texture(WGPUCommandEncoder e, const WGPUTexelCopyTextureInfo* src,
                                const WGPUTexelCopyTextureInfo* dst, const WGPUExtent3D* extent) {
    if (!encoding_deferred()) return wgpuCommandEncoderCopyTextureToTexture(e, src, dst, extent);
    put_op(kCopyTextureToTexture);
    put_copy_info(*src);
    put_copy_info(*dst);
    put(uint64_t{extent->width} | uint64_t{extent->height} << 32);
    put(extent->depthOrArrayLayers);
}

void ce_copy_texture_to_buffer(WGPUCommandEncoder e, const WGPUTexelCopyTextureInfo* src,
                               const WGPUTexelCopyBufferInfo* dst, const WGPUExtent3D* extent) {
    if (!encoding_deferred()) return wgpuCommandEncoderCopyTextureToBuffer(e, src, dst, extent);
    put_op(kCopyTextureToBuffer);
    put_copy_info(*src);
    hold(kBuffer, dst->buffer);
    put_ptr(dst->buffer);
    put(dst->layout.offset);
    put(uint64_t{dst->layout.bytesPerRow} | uint64_t{dst->layout.rowsPerImage} << 32);
    put(uint64_t{extent->width} | uint64_t{extent->height} << 32);
    put(extent->depthOrArrayLayers);
}

WGPUComputePassEncoder ce_begin_compute_pass(WGPUCommandEncoder e) {
    if (!encoding_deferred()) return wgpuCommandEncoderBeginComputePass(e, nullptr);
    put_op(kBeginComputePass);
    return kFakeCompute;
}

void cp_set_pipeline(WGPUComputePassEncoder p, WGPUComputePipeline pipeline) {
    if (!encoding_deferred()) return wgpuComputePassEncoderSetPipeline(p, pipeline);
    put_op(kComputeSetPipeline);
    put_ptr(pipeline);
}

void cp_set_bind_group(WGPUComputePassEncoder p, uint32_t index, WGPUBindGroup g, size_t n, const uint32_t* offsets) {
    if (!encoding_deferred()) return wgpuComputePassEncoderSetBindGroup(p, index, g, n, offsets);
    hold(kBindGroup, g);
    put_op(kComputeSetBindGroup);
    put(uint64_t{index} | uint64_t{static_cast<uint32_t>(n)} << 32);
    put_ptr(g);
    for (size_t i = 0; i < n; i++) put(offsets[i]);
}

/// A bind group made from its entries where the pass is replayed (the threaded web build: a group that lives for one
/// dispatch is not made on the renderer's thread, which would wait for the WebGPU thread each time). The entries'
/// buffers and views are held until then; the layout and samplers live in their caches.
void cp_set_bind_group_desc(WGPUComputePassEncoder p, uint32_t index, WGPUBindGroupLayout layout,
                            const WGPUBindGroupEntry* entries, size_t n) {
    if (!encoding_deferred()) {
        WGPUBindGroupDescriptor bd{};
        bd.layout = layout;
        bd.entryCount = n;
        bd.entries = entries;
        WGPUBindGroup g = wgpuDeviceCreateBindGroup(dev.device, &bd);
        wgpuComputePassEncoderSetBindGroup(p, index, g, 0, nullptr);
        wgpuBindGroupRelease(g);
        return;
    }
    put_op(kComputeSetBindGroupDesc);
    put(uint64_t{index} | uint64_t{static_cast<uint32_t>(n)} << 32);
    put_ptr(layout);
    for (size_t i = 0; i < n; i++) {
        const WGPUBindGroupEntry& e = entries[i];
        hold(kBuffer, e.buffer);
        hold(kView, e.textureView);
        put(e.binding);
        put_ptr(e.buffer);
        put(e.offset);
        put(e.size);
        put_ptr(e.sampler);
        put_ptr(e.textureView);
    }
}

void cp_dispatch(WGPUComputePassEncoder p, uint32_t x, uint32_t y, uint32_t z) {
    if (!encoding_deferred()) return wgpuComputePassEncoderDispatchWorkgroups(p, x, y, z);
    put_op(kDispatch);
    put(x);
    put(y);
    put(z);
}

void cp_end(WGPUComputePassEncoder p) {
    if (!encoding_deferred()) {
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        return;
    }
    put_op(kEndComputePass);
    rec().pass_ends.push_back(rec().words.size());
}

}   // namespace rx
