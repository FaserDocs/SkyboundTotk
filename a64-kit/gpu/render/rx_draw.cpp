// Draws: the 3D engine's state at a draw (programs, vertex input, blending, depth/stencil, viewport, scissor,
// constant buffers, textures) -> a WebGPU render pipeline (cached), bind groups and a draw in the current pass.
// Vertex and index data are copied from guest memory into the batch's upload stream for each draw (only the range
// the draw reads).
#include <unordered_set>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>
#include <bit>
#include <cmath>
#include <cstdlib>

#include "common/div_ceil.h"
#include "shader_recompiler/runtime_info.h"

#include "rx_internal.h"

namespace rx {
namespace {

using Topology = Regs::PrimitiveTopology;

WGPUCompareFunction compare_op(Regs::ComparisonOp op) {
    using C = Regs::ComparisonOp;
    switch (op) {
    case C::Never_D3D: case C::Never_GL: return WGPUCompareFunction_Never;
    case C::Less_D3D: case C::Less_GL: return WGPUCompareFunction_Less;
    case C::Equal_D3D: case C::Equal_GL: return WGPUCompareFunction_Equal;
    case C::LessEqual_D3D: case C::LessEqual_GL: return WGPUCompareFunction_LessEqual;
    case C::Greater_D3D: case C::Greater_GL: return WGPUCompareFunction_Greater;
    case C::NotEqual_D3D: case C::NotEqual_GL: return WGPUCompareFunction_NotEqual;
    case C::GreaterEqual_D3D: case C::GreaterEqual_GL: return WGPUCompareFunction_GreaterEqual;
    default: return WGPUCompareFunction_Always;
    }
}

WGPUStencilOperation stencil_op(Regs::StencilOp::Op op) {
    using O = Regs::StencilOp::Op;
    switch (op) {
    case O::Zero_D3D: case O::Zero_GL: return WGPUStencilOperation_Zero;
    case O::Replace_D3D: case O::Replace_GL: return WGPUStencilOperation_Replace;
    case O::IncrSaturate_D3D: case O::IncrSaturate_GL: return WGPUStencilOperation_IncrementClamp;
    case O::DecrSaturate_D3D: case O::DecrSaturate_GL: return WGPUStencilOperation_DecrementClamp;
    case O::Invert_D3D: case O::Invert_GL: return WGPUStencilOperation_Invert;
    case O::Incr_D3D: case O::Incr_GL: return WGPUStencilOperation_IncrementWrap;
    case O::Decr_D3D: case O::Decr_GL: return WGPUStencilOperation_DecrementWrap;
    default: return WGPUStencilOperation_Keep;
    }
}

WGPUBlendOperation blend_op(Regs::Blend::Equation e) {
    using E = Regs::Blend::Equation;
    switch (e) {
    case E::Subtract_D3D: case E::Subtract_GL: return WGPUBlendOperation_Subtract;
    case E::ReverseSubtract_D3D: case E::ReverseSubtract_GL: return WGPUBlendOperation_ReverseSubtract;
    case E::Min_D3D: case E::Min_GL: return WGPUBlendOperation_Min;
    case E::Max_D3D: case E::Max_GL: return WGPUBlendOperation_Max;
    default: return WGPUBlendOperation_Add;
    }
}

WGPUBlendFactor blend_factor(Regs::Blend::Factor f) {
    using F = Regs::Blend::Factor;
    switch (f) {
    case F::Zero_D3D: case F::Zero_GL: return WGPUBlendFactor_Zero;
    case F::One_D3D: case F::One_GL: return WGPUBlendFactor_One;
    case F::SourceColor_D3D: case F::SourceColor_GL: return WGPUBlendFactor_Src;
    case F::OneMinusSourceColor_D3D: case F::OneMinusSourceColor_GL: return WGPUBlendFactor_OneMinusSrc;
    case F::SourceAlpha_D3D: case F::SourceAlpha_GL: return WGPUBlendFactor_SrcAlpha;
    case F::OneMinusSourceAlpha_D3D: case F::OneMinusSourceAlpha_GL: return WGPUBlendFactor_OneMinusSrcAlpha;
    case F::DestAlpha_D3D: case F::DestAlpha_GL: return WGPUBlendFactor_DstAlpha;
    case F::OneMinusDestAlpha_D3D: case F::OneMinusDestAlpha_GL: return WGPUBlendFactor_OneMinusDstAlpha;
    case F::DestColor_D3D: case F::DestColor_GL: return WGPUBlendFactor_Dst;
    case F::OneMinusDestColor_D3D: case F::OneMinusDestColor_GL: return WGPUBlendFactor_OneMinusDst;
    case F::SourceAlphaSaturate_D3D: case F::SourceAlphaSaturate_GL: return WGPUBlendFactor_SrcAlphaSaturated;
    case F::BlendFactor_D3D: case F::ConstantColor_GL: case F::ConstantAlpha_GL: return WGPUBlendFactor_Constant;
    case F::OneMinusBlendFactor_D3D: case F::OneMinusConstantColor_GL: case F::OneMinusConstantAlpha_GL:
        return WGPUBlendFactor_OneMinusConstant;
    // dual source blending: approximated with the first source
    case F::Source1Color_D3D: case F::Source1Color_GL:
        log_limited("dual_source", 4, "rx: a blend uses the second source (dual-source blending): approximated");
        return WGPUBlendFactor_Src;
    case F::OneMinusSource1Color_D3D: case F::OneMinusSource1Color_GL:
        log_limited("dual_source", 4, "rx: a blend uses the second source (dual-source blending): approximated");
        return WGPUBlendFactor_OneMinusSrc;
    case F::Source1Alpha_D3D: case F::Source1Alpha_GL:
        log_limited("dual_source", 4, "rx: a blend uses the second source (dual-source blending): approximated");
        return WGPUBlendFactor_SrcAlpha;
    case F::OneMinusSource1Alpha_D3D: case F::OneMinusSource1Alpha_GL:
        log_limited("dual_source", 4, "rx: a blend uses the second source (dual-source blending): approximated");
        return WGPUBlendFactor_OneMinusSrcAlpha;
    default: return WGPUBlendFactor_One;
    }
}

/// The WebGPU vertex format of an attribute, and its size in bytes (Undefined: none).
WGPUVertexFormat vertex_format(const Regs::VertexAttribute& a, uint32_t& size) {
    using S = Regs::VertexAttribute::Size;
    using T = Regs::VertexAttribute::Type;
    const T t = a.type;
    const bool f = t == T::Float, sn = t == T::SNorm, un = t == T::UNorm, si = t == T::SInt || t == T::SScaled,
               ui = t == T::UInt || t == T::UScaled;
    auto pick = [&](WGPUVertexFormat vf, WGPUVertexFormat vs, WGPUVertexFormat vu, WGPUVertexFormat vsn,
                    WGPUVertexFormat vun, uint32_t bytes) {
        size = bytes;
        return f ? vf : si ? vs : ui ? vu : sn ? vsn : un ? vun : WGPUVertexFormat(0);
    };
    constexpr WGPUVertexFormat X = WGPUVertexFormat(0);
    switch (a.size) {
    case S::Size_R32_G32_B32_A32:
        return pick(WGPUVertexFormat_Float32x4, WGPUVertexFormat_Sint32x4, WGPUVertexFormat_Uint32x4, X, X, 16);
    case S::Size_R32_G32_B32:
        return pick(WGPUVertexFormat_Float32x3, WGPUVertexFormat_Sint32x3, WGPUVertexFormat_Uint32x3, X, X, 12);
    case S::Size_R32_G32:
        return pick(WGPUVertexFormat_Float32x2, WGPUVertexFormat_Sint32x2, WGPUVertexFormat_Uint32x2, X, X, 8);
    case S::Size_R32:
        return pick(WGPUVertexFormat_Float32, WGPUVertexFormat_Sint32, WGPUVertexFormat_Uint32, X, X, 4);
    case S::Size_R16_G16_B16_A16:
    case S::Size_R16_G16_B16:   // read as 4 components (the 4th is whatever follows)
        return pick(WGPUVertexFormat_Float16x4, WGPUVertexFormat_Sint16x4, WGPUVertexFormat_Uint16x4,
                    WGPUVertexFormat_Snorm16x4, WGPUVertexFormat_Unorm16x4, 8);
    case S::Size_R16_G16:
        return pick(WGPUVertexFormat_Float16x2, WGPUVertexFormat_Sint16x2, WGPUVertexFormat_Uint16x2,
                    WGPUVertexFormat_Snorm16x2, WGPUVertexFormat_Unorm16x2, 4);
    case S::Size_R16:
        return pick(WGPUVertexFormat_Float16, WGPUVertexFormat_Sint16, WGPUVertexFormat_Uint16,
                    WGPUVertexFormat_Snorm16, WGPUVertexFormat_Unorm16, 2);
    case S::Size_R8_G8_B8_A8:
    case S::Size_X8_B8_G8_R8:
    case S::Size_R8_G8_B8:
        if (a.bgra && un) {
            size = 4;
            return WGPUVertexFormat_Unorm8x4BGRA;
        }
        return pick(X, WGPUVertexFormat_Sint8x4, WGPUVertexFormat_Uint8x4, WGPUVertexFormat_Snorm8x4,
                    WGPUVertexFormat_Unorm8x4, 4);
    case S::Size_R8_G8:
    case S::Size_G8_R8:
        return pick(X, WGPUVertexFormat_Sint8x2, WGPUVertexFormat_Uint8x2, WGPUVertexFormat_Snorm8x2,
                    WGPUVertexFormat_Unorm8x2, 2);
    case S::Size_R8:
    case S::Size_A8:
        return pick(X, WGPUVertexFormat_Sint8, WGPUVertexFormat_Uint8, WGPUVertexFormat_Snorm8, WGPUVertexFormat_Unorm8,
                    1);
    case S::Size_A2_B10_G10_R10:
        size = 4;
        return un ? WGPUVertexFormat_Unorm10_10_10_2 : X;
    default:
        size = 0;
        return X;
    }
}

/// One attribute element decoded on the CPU, for formats WebGPU lacks (A2B10G10R10 other than unorm, B10G11R11,
/// 3-component formats that would read past the stride): four floats, or four integers for integer attributes.
float unpack_small_float(uint32_t v, int mantissa) {
    const uint32_t e = v >> mantissa, m = v & ((1u << mantissa) - 1);
    if (e == 0) return std::ldexp((float)m, -14 - mantissa);
    if (e == 31) return m ? NAN : INFINITY;
    return std::ldexp(1.0f + (float)m / (1u << mantissa), (int)e - 15);
}
float half_to_float(uint16_t h) {
    const uint32_t s = h >> 15, e = h >> 10 & 31, m = h & 1023;
    float f = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1.0f + m / 1024.0f, (int)e - 15);
    return s ? -f : f;
}
void decode_attribute(const uint8_t* p, const Regs::VertexAttribute& a, uint32_t out[4]) {
    using S = Regs::VertexAttribute::Size;
    using T = Regs::VertexAttribute::Type;
    int bits[4] = {0, 0, 0, 0};
    switch (a.size) {
    case S::Size_R32_G32_B32_A32: bits[0] = bits[1] = bits[2] = bits[3] = 32; break;
    case S::Size_R32_G32_B32: bits[0] = bits[1] = bits[2] = 32; break;
    case S::Size_R32_G32: bits[0] = bits[1] = 32; break;
    case S::Size_R32: bits[0] = 32; break;
    case S::Size_R16_G16_B16_A16: bits[0] = bits[1] = bits[2] = bits[3] = 16; break;
    case S::Size_R16_G16_B16: bits[0] = bits[1] = bits[2] = 16; break;
    case S::Size_R16_G16: bits[0] = bits[1] = 16; break;
    case S::Size_R16: bits[0] = 16; break;
    case S::Size_R8_G8_B8_A8: case S::Size_X8_B8_G8_R8: bits[0] = bits[1] = bits[2] = bits[3] = 8; break;
    case S::Size_R8_G8_B8: bits[0] = bits[1] = bits[2] = 8; break;
    case S::Size_R8_G8: case S::Size_G8_R8: bits[0] = bits[1] = 8; break;
    case S::Size_R8: case S::Size_A8: bits[0] = 8; break;
    case S::Size_A2_B10_G10_R10: bits[0] = bits[1] = bits[2] = 10; bits[3] = 2; break;
    case S::Size_B10_G11_R11: bits[0] = bits[1] = 11; bits[2] = 10; break;
    default: break;
    }
    const bool integer = a.type == T::SInt || a.type == T::UInt || a.type == T::SScaled || a.type == T::UScaled;
    const bool packed = a.size == S::Size_A2_B10_G10_R10 || a.size == S::Size_B10_G11_R11;
    uint32_t word = 0;
    if (packed) std::memcpy(&word, p, 4);
    int shift = 0, byte = 0;
    for (int c = 0; c < 4; c++) {
        const float def = c == 3 ? 1.0f : 0.0f;
        if (!bits[c]) {
            out[c] = integer ? (c == 3 ? 1u : 0u) : std::bit_cast<uint32_t>(def);
            continue;
        }
        uint32_t v;
        if (packed) {
            v = word >> shift & ((1u << bits[c]) - 1);
            shift += bits[c];
        } else {
            v = 0;
            std::memcpy(&v, p + byte, bits[c] / 8);
            byte += bits[c] / 8;
        }
        const int n = bits[c];
        const int32_t sv = n == 32 ? (int32_t)v : (int32_t)(v << (32 - n)) >> (32 - n);
        if (integer) {
            out[c] = a.type == T::SInt || a.type == T::SScaled ? (uint32_t)sv : v;
            continue;
        }
        float f;
        switch (a.type) {
        case T::SNorm: f = std::max(-1.0f, (float)sv / (float)((1u << (n - 1)) - 1)); break;
        case T::UNorm: f = n == 32 ? (float)v / 4294967295.0f : (float)v / (float)((1u << n) - 1); break;
        default:   // float
            f = a.size == S::Size_B10_G11_R11 ? unpack_small_float(v, n - 5)
                : n == 16                     ? half_to_float((uint16_t)v)
                : n == 32                     ? std::bit_cast<float>(v)
                                              : (float)v;
            break;
        }
        out[c] = std::bit_cast<uint32_t>(f);
    }
    if (a.size == S::Size_G8_R8) std::swap(out[0], out[1]);
}

} // namespace

/// Whether WebGPU's vertex fetch can read the attribute (strides multiples of 4, offsets multiples of min(4, the
/// format's size), the attribute inside its stride, a format it has).
bool fetchable(const Regs::VertexAttribute& a, const Regs::VertexStream& vs) {
    uint32_t size = 0;
    WGPUVertexFormat f = vertex_format(a, size);
    if (vertex_unpack(a)) {
        f = WGPUVertexFormat_Uint32;
        size = 4;
    }
    const uint32_t stride = vs.stride;
    return f && !(stride & 3) && !(size && a.offset % std::min(4u, size)) && !(stride && a.offset + size > stride) &&
           !(!stride && a.offset + size > 2048);
}

namespace {

struct Pipeline {
    WGPURenderPipeline pipeline = nullptr;
    WGPUBindGroupLayout layouts[2]{};
    std::atomic<int> state{0};   // 0 compiling, 1 ready, 2 failed
    int64_t last_used = 0;
    bool cache_used = false;   // a draw used it this session (the native pipeline cache keeps what sessions use)
};

/// What a pipeline's creation reads, owned: pipelines are compiled on worker threads (a driver compile takes tens
/// of milliseconds; draws that need one are skipped until it is ready). RX_SYNC_PIPELINES=1: compiled in place.
struct PipelineJob {
    Pipeline* target = nullptr;
    WGPUPipelineLayout layout = nullptr;
    WGPUShaderModule vs = nullptr, fs = nullptr;
    ShaderModule *vs_module = nullptr, *fs_module = nullptr;   // (created here first: ensure_module)
    std::vector<WGPUVertexBufferLayout> buffers;
    std::vector<std::vector<WGPUVertexAttribute>> attributes;
    std::vector<WGPUColorTargetState> targets;
    std::array<WGPUBlendState, 8> blends{};
    WGPUPrimitiveState primitive{};
    bool has_depth = false;
    WGPUDepthStencilState depth{};
    std::vector<WGPUBindGroupLayoutEntry> bgl[2];   // (the browser's pipeline cache records them)
    int nbgl = 1;
};

std::atomic<int64_t> st_module_ns, st_pipeline_ns, st_pipelines_made;   // (pipeline compilers' time)

void compile(PipelineJob& j) {
    const int64_t t0 = hos_now_ns();
    const bool vs_ok = ensure_module(*j.vs_module), fs_ok = !j.fs_module || ensure_module(*j.fs_module);
    const int64_t t1 = hos_now_ns();
    st_module_ns += t1 - t0;
    struct Timer {
        int64_t t1;
        ~Timer() {
            st_pipeline_ns += hos_now_ns() - t1;
            st_pipelines_made++;
            static std::atomic<int64_t> last;
            const int64_t now = hos_now_ns();
            if (now - last > 5000000000LL) {
                last = now;
                hos_log("rx: pipeline compilers: %lld pipelines, modules %.0f ms, pipelines %.0f ms (total so far)",
                        (long long)st_pipelines_made.load(), st_module_ns.load() / 1e6, st_pipeline_ns.load() / 1e6);
            }
        }
    } timer{t1};
    if (!vs_ok || !fs_ok) {
        wgpuPipelineLayoutRelease(j.layout);
        j.target->state = 2;
        return;
    }
    j.vs = j.vs_module->module;
    j.fs = j.fs_module ? j.fs_module->module : nullptr;
    for (size_t i = 0; i < j.buffers.size(); i++) {
        j.buffers[i].attributes = j.attributes[i].data();
        j.buffers[i].attributeCount = j.attributes[i].size();
    }
    for (size_t i = 0; i < j.targets.size(); i++)
        if (j.targets[i].blend) j.targets[i].blend = &j.blends[i];
    WGPUFragmentState fs{};
    if (j.fs) {
        fs.module = j.fs;
        fs.entryPoint = sv("main");
        fs.targetCount = j.targets.size();
        fs.targets = j.targets.data();
    }
    WGPURenderPipelineDescriptor d{};
    d.layout = j.layout;
    d.vertex.module = j.vs;
    d.vertex.entryPoint = sv("main");
    d.vertex.bufferCount = j.buffers.size();
    d.vertex.buffers = j.buffers.data();
    d.primitive = j.primitive;
    d.depthStencil = j.has_depth ? &j.depth : nullptr;
    d.multisample.count = 1;
    d.multisample.mask = 0xFFFFFFFF;
    d.fragment = j.fs ? &fs : nullptr;
#if RX_WEB
    pipecache_record_render(d, *j.vs_module, j.fs_module, j.bgl, j.nbgl);
    for (ShaderModule* m : {j.vs_module, j.fs_module})
        if (m && m->module && m->cache_hash && pipecache_has_module(m->cache_hash)) release_module_text(*m);
    // (the browser: the GPU process compiles it on its own threads (Tint -> HLSL -> the D3D compiler, tens to
    // hundreds of ms each); its draws are skipped until the callback, which wgpuInstanceProcessEvents runs, at present.
    // Made in place, a load's hundreds of pipelines were compiled one by one while the game waited on the GPU.)
    WGPUCreateRenderPipelineAsyncCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUCreatePipelineAsyncStatus status, WGPURenderPipeline pipeline, WGPUStringView message,
                     void* userdata1, void*) {
        auto* target = static_cast<Pipeline*>(userdata1);
        if (status == WGPUCreatePipelineAsyncStatus_Success) {
            target->pipeline = pipeline;
            target->state = 1;
        } else {
            static int logged;
            if (logged++ < 30)
                hos_log("rx: render pipeline not created: %.*s", (int)(message.data ? message.length : 0),
                        message.data ? message.data : "");
            target->state = 2;
        }
    };
    cb.userdata1 = j.target;
    wgpuDeviceCreateRenderPipelineAsync(dev.device, &d, cb);
    wgpuPipelineLayoutRelease(j.layout);
    return;
#endif
    scope_push();
    j.target->pipeline = wgpuDeviceCreateRenderPipeline(dev.device, &d);
    const std::string error = scope_pop();
    wgpuPipelineLayoutRelease(j.layout);
    if (!error.empty()) {
        static std::atomic<int> logged;
        if (logged++ < 30) hos_log("rx: render pipeline not created: %s", error.c_str());
        j.target->state = 2;
    } else {
        j.target->state = 1;
    }
}

struct PipelineWorkers {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::unique_ptr<PipelineJob>> jobs;
    bool started = false;
    void submit(std::unique_ptr<PipelineJob> job) {
        {
            std::lock_guard lk(m);
            jobs.push_back(std::move(job));
            if (!started) {
                started = true;
                // (2: with 6, everything deadlocked a few seconds into the Great Sky Island, 2026-10-13)
                const char* want = getenv("RX_COMPILERS");
                const int n = env_flag("RX_COMPILERS_6") ? 6 : want ? std::clamp(atoi(want), 1, 8) : 2;
                for (int i = 0; i < n; i++)
                    std::thread([this] {
                        hos_host_thread_register_self("pipeline compiler");
                        background_thread();
                        run();
                    }).detach();
            }
        }
        cv.notify_one();
    }
    std::unique_ptr<PipelineJob> steal(Pipeline* target) {
        std::lock_guard lk(m);
        for (auto it = jobs.begin(); it != jobs.end(); ++it)
            if ((*it)->target == target) {
                std::unique_ptr<PipelineJob> job = std::move(*it);
                jobs.erase(it);
                return job;
            }
        return nullptr;
    }
    void run() {
        for (;;) {
            std::unique_ptr<PipelineJob> job;
            {
                std::unique_lock lk(m);
                cv.wait(lk, [this] { return !jobs.empty(); });
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            compile(*job);
        }
    }
} pipeline_workers;
std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> pipelines;
WGPUBuffer const_vertex_buffer;

std::string dump_textures;   // (RX_DUMP_FRAME) the textures of the stage being bound
uint64_t dump_fs_program;    // (RX_DUMP_UB) the pixel program being bound

/// One stage's resources for this draw: its bind group layout, entries (dynamic bindings at offset 0) and the
/// dynamic offsets (in binding order).
/// A bind group layout entry packed in 32 bits (the pipeline key hashes these; the WebGPU entries are only made for a
/// new pipeline): binding 0-9, visibility 10-11, kind 12-13 (1 buffer, 2 texture, 3 sampler), then per kind: buffer
/// type 14-17 and dynamic offset 18; sample type 14-17, view dimension 18-21, multisampled 22; sampler type 14-17.
uint32_t pack_layout(const WGPUBindGroupLayoutEntry& e) {
    uint32_t v = (e.binding & 1023) | ((uint32_t)e.visibility & 3) << 10;
    if (e.buffer.type) v |= 1u << 12 | ((uint32_t)e.buffer.type & 15) << 14 | (e.buffer.hasDynamicOffset ? 1u : 0u) << 18;
    else if (e.texture.sampleType)
        v |= 2u << 12 | ((uint32_t)e.texture.sampleType & 15) << 14 | ((uint32_t)e.texture.viewDimension & 15) << 18 |
             (e.texture.multisampled ? 1u : 0u) << 22;
    else v |= 3u << 12 | ((uint32_t)e.sampler.type & 15) << 14;
    return v;
}
WGPUBindGroupLayoutEntry unpack_layout(uint32_t v) {
    WGPUBindGroupLayoutEntry e{};
    e.binding = v & 1023;
    e.visibility = (WGPUShaderStage)(v >> 10 & 3);
    switch (v >> 12 & 3) {
    case 1:
        e.buffer.type = (WGPUBufferBindingType)(v >> 14 & 15);
        e.buffer.hasDynamicOffset = v >> 18 & 1;
        break;
    case 2:
        e.texture.sampleType = (WGPUTextureSampleType)(v >> 14 & 15);
        e.texture.viewDimension = (WGPUTextureViewDimension)(v >> 18 & 15);
        e.texture.multisampled = v >> 22 & 1;
        break;
    default: e.sampler.type = (WGPUSamplerBindingType)(v >> 14 & 15); break;
    }
    return e;
}

struct StageResources {
    std::vector<uint32_t> layout;   // packed (pack_layout)
    std::vector<WGPUBindGroupEntry> entries;
    std::vector<uint32_t> dynamic_offsets;
    uint32_t next_binding = 0;
    bool ok = true;
};

uint32_t read_handle(const Channel& ch, uint32_t stage, const Shader::TextureDescriptor& d) {
    const auto& cbufs = ch.cbufs[stage];
    const uint64_t addr = cbufs[d.cbuf_index % 18].address + d.cbuf_offset;
    if (d.has_secondary) {
        const uint64_t sec = cbufs[d.secondary_cbuf_index % 18].address + d.secondary_cbuf_offset;
        return (gpu_get<uint32_t>(addr) << d.shift_left) | (gpu_get<uint32_t>(sec) << d.secondary_shift_left);
    }
    return gpu_get<uint32_t>(addr);
}

/// One stage's uniform block, constant/storage buffers and textures (the layout follows the WGSL backend: binding 0
/// the uniform block [rescale (4 vec4), the textures' swizzles, the constant buffers that fit], then the other
/// constant buffers (storage), storage buffers, textures (texture, sampler)).
/// dynamic_storage: how many more storage bindings may take dynamic offsets (WebGPU's per-layout limit).
/// RX_WATCH_TEX_SIZE=<width>x<height> (debugging): draws sampling a texture of that size are logged (any frame)
void watch_tex_size(const Attachments& at, const BoundTexture& t) {
    static uint32_t ww = 0, wh = 0;
    static const bool read = [] {
        if (const char* e = getenv("RX_WATCH_TEX_SIZE")) std::sscanf(e, "%ux%u", &ww, &wh);
        return true;
    }();
    (void)read;
    if (!ww || t.width != ww || t.height != wh) return;
    const Image* c0 = at.color[0];
    log_limited("watch_tex_size", 300, "rx: watch: frame %lld draws into %08llx %ux%u sampling %c%08llx %ux%u",
                (long long)frame_number, c0 ? (unsigned long long)c0->guest : 0ull, at.width, at.height, t.source,
                (unsigned long long)t.guest, t.width, t.height);
}

/// A storage buffer range a draw binds: a read-only one in place in the buffer cache; a written one copied into the
/// read-write stream once per batch, the draws that write it sharing the copy as they share the memory (TOTK's
/// material shaders all set flags in one visibility buffer: a copy each was 8 MB a frame, uploaded and read back),
/// started from the buffer cache when the GPU wrote it last (after a dispatch wrote storage, a copy from before it
/// would hide what it wrote, then overwrite it in the buffer cache at the pass's end: TOTK's exposure dispatch before
/// the lighting pass, the open world black). in_stream: the range is in a stream (it may take a dynamic offset).
struct StorageBinding {
    WGPUBuffer buffer = nullptr;
    uint64_t offset = 0, size = 0;
    bool in_stream = false, ok = false;
};
StorageBinding draw_storage(uint64_t va, uint64_t size, bool writable) {
    StorageBinding out;
    static std::unordered_map<uint64_t, uint64_t> rw_copies;
    static int64_t rw_copies_batch = -1;
    static uint64_t rw_copies_epoch = 0;
    if (rw_copies_batch != batch_number() || rw_copies_epoch != dispatch_write_epoch) {
        rw_copies.clear();
        rw_copies_batch = batch_number();
        rw_copies_epoch = dispatch_write_epoch;
    }
    if (static const bool log_ro = env_flag("RX_RO_AFTER_RW_LOG"); log_ro && !writable && va)   // (experiment)
        for (const auto& [k, o] : rw_copies) {
            const uint64_t kva = k & ((1ull << 44) - 1), ksize = k >> 44;
            if (kva < va + size && kva + ksize > va && storage_written_this_pass(va, size)) {
                count("ssbo_ro_after_rw");
                log_limited("ro_after_rw", 3000, "rx: read-only storage %llx+%llx reads memory this batch's writable copy "
                            "of %llx+%llx holds (frame %lld)", (unsigned long long)va, (unsigned long long)size,
                            (unsigned long long)kva, (unsigned long long)ksize, (long long)frame_number);
                break;
            }
        }
    // A read of storage a pixel program of the open pass wrote (into the batch's copy: the buffer cache gets it when
    // the pass ends): the pass ends first, so this draw sees those writes, as draws after it do on the console. TOTK's
    // tiled lighting: one variant's pixels flag tiles, the next variant's vertex program reads the flags to pick its
    // tiles; from the pass's start state it picked stale ones (light boxes on the horizon after a load).
    // RX_NO_STORAGE_HAZARD_BREAK=1: as before
    static const bool no_hazard_break = env_flag("RX_NO_STORAGE_HAZARD_BREAK");
    if (!writable && va && !no_hazard_break && storage_written_this_pass(va, size)) {
        end_render_pass("pe_storage_hazard");
        count("storage_hazard_breaks");
    }
    if (!writable && va && buffer_cache_active()) {
        const BufferRef c = guest_buffer(va, size);
        if (c.buffer && c.offset % dev.limits.minStorageBufferOffsetAlignment == 0) {
            out.buffer = c.buffer, out.offset = c.offset, out.size = c.size, out.ok = true;
            return out;
        }
        // a range too big for the buffer cache's segments (> 8 MB): a GPU buffer of its own, uploaded again only when
        // the CPU wrote it (a walk through the castle once copied 218 such ranges a frame, 133 MB, for every draw).
        // Not while a dispatch's results are newer on the GPU (gpu_written: the copy below takes them from there).
        // RX_NO_BIG_SSBO_CACHE=1: copied, as before
        static const bool no_big = env_flag("RX_NO_BIG_SSBO_CACHE");
        if (!c.buffer && size > (8u << 20) && !no_big && !gpu_written(va, size)) {
            uint64_t avail = 0;
            const uint8_t* host = hos_gpu_ptr(va, &avail);
            if (host && avail >= size) {
                struct Big { WGPUBuffer buffer = nullptr; uint64_t seq = 0; int64_t frame = 0; };
                static std::unordered_map<uint64_t, Big> bigs;
                const uint64_t guest = hos_gpu_translate(va);
                Big& b = bigs[guest ^ size << 44];
                if (!b.buffer) {
                    WGPUBufferDescriptor bd{};
                    bd.label = sv("rx big storage range");
                    bd.size = (size + 3) & ~3ull;
                    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
                    b.buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
                    b.seq = 0;
                }
                if (!b.seq) {
                    b.seq = memory_seq();
                    queue_upload(b.buffer, 0, host, (size + 3) & ~3ull);
                    count("big_ssbo_uploads", (int64_t)size);
                } else if (memory_written_since(guest, size, b.seq)) {   // (the pages written since, in runs)
                    const uint64_t seq = b.seq;
                    b.seq = memory_seq();
                    const uint64_t first = guest & ~4095ull, end = (guest + size + 4095) & ~4095ull;
                    for (uint64_t p = first; p < end;) {
                        if (!memory_written_since(p, 4096, seq)) {
                            p += 4096;
                            continue;
                        }
                        uint64_t q = p + 4096;
                        while (q < end && memory_written_since(q, 4096, seq)) q += 4096;
                        const uint64_t lo = std::max(p, guest), hi = std::min(q, guest + size);
                        const uint64_t off = (lo - guest) & ~3ull, n = ((hi - guest + 3) & ~3ull) - off;
                        queue_upload(b.buffer, off, host + off, n);
                        count("big_ssbo_uploads", (int64_t)n);
                        p = q;
                    }
                }
                b.frame = frame_number;
                if (bigs.size() > 64)   // (unused for a minute: released)
                    for (auto it = bigs.begin(); it != bigs.end();)
                        if (frame_number - it->second.frame > 3600) {
                            release_buffer_later(it->second.buffer);
                            it = bigs.erase(it);
                        } else {
                            ++it;
                        }
                out.buffer = b.buffer, out.offset = 0, out.size = size, out.ok = true;
                count("big_ssbo_bound");
                return out;
            }
        }
        count(!c.buffer ? (size > (8u << 20) ? "ssbo_copy_big" : "ssbo_copy_unmapped") : "ssbo_copy_unaligned");
        if (static const bool log_copies = env_flag("RX_LOG_SSBO_COPIES"); log_copies)
            log_limited("ssbo_copy", 200, "rx: read-only storage %llx+%llx copied (%s)", (unsigned long long)va,
                        (unsigned long long)size, !c.buffer ? (size > (8u << 20) ? "big" : "unmapped") : "unaligned");
    } else if (writable) {
        count("ssbo_copy_writable");
    }
    Stream& s = writable ? stream_rw : stream_ro;
    const uint64_t rw_key = va ^ size << 44;
    uint64_t off = UINT64_MAX;
    if (writable)
        if (auto it = rw_copies.find(rw_key); it != rw_copies.end()) off = it->second;
    if (off == UINT64_MAX) {
        static const bool overlap_log = env_flag("RX_RW_OVERLAP_LOG");   // (debugging: another copy of overlapping memory)
        if (writable && va && overlap_log)
            for (const auto& [k, o] : rw_copies) {
                const uint64_t kva = k & ((1ull << 44) - 1), ksize = k >> 44;
                if (kva < va + size && kva + ksize > va)
                    log_limited("rw_overlap", 40, "rx: writable storage %llx+%llx overlaps this batch's copy of %llx+%llx",
                                (unsigned long long)va, (unsigned long long)size, (unsigned long long)kva,
                                (unsigned long long)ksize);
            }
        off = s.alloc(size, dev.limits.minStorageBufferOffsetAlignment);
        if (off == UINT64_MAX) return out;
        if (va) gpu_read(va, s.at(off), size);
        else std::memset(s.at(off), 0, size);
        count("up_ssbo", (int64_t)size);
        if (writable && va) {
            // newer on the GPU (a dispatch or an earlier pass wrote it): the copy starts from the buffer cache
            if (gpu_written(va, size))
                if (const BufferRef ref = guest_buffer(va, size); ref.buffer) {
                    end_render_pass("pe_ssbo_copy");
                    ce_copy_buffer_to_buffer(encoder(), ref.buffer, ref.offset, s.buffer, off, size);
                    count("ssbo_from_cache", (int64_t)size);
                }
            writeback_after_batch(off, size, va);
            rw_copies.emplace(rw_key, off);
        }
    }
    if (writable && va) storage_written_in_pass(off, size, va);
    out.buffer = s.buffer, out.offset = off, out.size = size, out.in_stream = true, out.ok = true;
    return out;
}

void stage_resources(StageResources& out, const Channel& ch, const Regs& r, const Shader::Info& info,
                     uint32_t stage, WGPUShaderStage vis, const float rescale[16], const Attachments& at,
                     uint32_t& dynamic_storage) {
    auto buffer_entry = [&](uint32_t b, WGPUBufferBindingType type, bool dynamic, WGPUBuffer buf, uint64_t off,
                            uint64_t size) {
        WGPUBindGroupLayoutEntry l{};
        l.binding = b;
        l.visibility = vis;
        l.buffer.type = type;
        l.buffer.hasDynamicOffset = dynamic;
        out.layout.push_back(pack_layout(l));
        WGPUBindGroupEntry e{};
        e.binding = b;
        e.buffer = buf;
        e.offset = dynamic ? 0 : off;
        e.size = size;
        out.entries.push_back(e);
        if (dynamic) out.dynamic_offsets.push_back((uint32_t)off);
    };
    // the uniform block's layout (as the WGSL backend computes it)
    const bool indirect = info.uses_global_memory || info.uses_cbuf_indirect;
    std::array<uint32_t, 18> base{};
    std::array<bool, 18> in_ub{};
    uint32_t ub_size = 4 + 2 * (uint32_t)info.texture_descriptors.size();
    if (!indirect)
        for (const auto& d : info.constant_buffer_descriptors) {
            const uint32_t size = std::max(Common::DivCeil(info.constant_buffer_used_sizes[d.index], 16U), 1U);
            if (ub_size + size > 0x1000) continue;
            in_ub[d.index % 18] = true;
            base[d.index % 18] = ub_size;
            ub_size += size;
        }
    // (the block is made at the end, once the texture slots are known: a draw whose block would hold what the last one
    // made for this program holds reuses it; RX_NO_UB_REUSE=1 always makes a new one)
    static std::vector<uint8_t> slot_bytes;
    slot_bytes.assign(info.texture_descriptors.size() * 32, 0);
    buffer_entry(0, WGPUBufferBindingType_Uniform, true, stream_ro.buffer, 0, (uint64_t)ub_size * 16);
    const size_t ub_dynamic = out.dynamic_offsets.size() - 1;
    uint32_t b = 1;
    for (const auto& d : info.constant_buffer_descriptors) {
        if (in_ub[d.index % 18]) continue;
        // the others: read-only storage of the bound size, copied once per batch while their memory is unchanged
        const ConstBuffer& cb = ch.cbufs[stage][d.index % 18];
        const uint64_t size = std::clamp<uint64_t>((cb.enabled ? cb.size : 0) + 15 & ~15ull, 16, 0x10000);
        static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> copies;   // address ^ size -> offset, seq
        static int64_t copies_batch = -1;
        if (copies_batch != batch_number()) {
            copies.clear();
            copies_batch = batch_number();
        }
        const uint64_t ckey = cb.address ^ size << 44;
        const uint64_t guest = memory_seq() ? hos_gpu_translate(cb.address) : 0;
        uint64_t off = UINT64_MAX;
        if (auto it = copies.find(ckey); it != copies.end() && guest &&
                                         !memory_written_since(guest, size, it->second.second))
            off = it->second.first;
        if (off == UINT64_MAX) {
            off = stream_ro.alloc(size, dev.limits.minStorageBufferOffsetAlignment);
            if (off == UINT64_MAX) {
                out.ok = false;
                return;
            }
            const uint64_t n = cb.enabled ? std::min<uint64_t>(size, cb.size) : 0;
            if (n) gpu_read(cb.address, stream_ro.at(off), n);
            if (n < size) std::memset(stream_ro.at(off + n), 0, size - n);
            count("up_cbuf_storage", (int64_t)size);
            if (guest) copies[ckey] = {off, memory_seq()};
        }
        const bool dynamic = dynamic_storage > 0;
        dynamic_storage -= dynamic;
        buffer_entry(b, WGPUBufferBindingType_ReadOnlyStorage, dynamic, stream_ro.buffer, off, size);
        b += d.count;
    }
    for (const auto& d : info.storage_buffers_descriptors) {
        const ConstBuffer& cb = ch.cbufs[stage][d.cbuf_index % 18];
        // the program reads from the address aligned down to the storage offset alignment (the frontend's
        // min_ssbo_alignment): the binding starts there
        const uint64_t raw_va = gpu_get<uint64_t>(cb.address + d.cbuf_offset);
        const uint64_t va = raw_va & ~(uint64_t)(dev.limits.minStorageBufferOffsetAlignment - 1);
        uint64_t size = gpu_get<uint32_t>(cb.address + d.cbuf_offset + 8) + (raw_va - va);
        size = std::clamp<uint64_t>((size + 3) & ~3ull, 16, 16ull << 20);
        const bool writable = d.is_written && stage == 4;
        if (frame_dump_active())
            if (const char* want = getenv("RX_DUMP_UB"); want && strtoull(want, nullptr, 16) == dump_fs_program) {
                uint32_t w[16]{};
                if (const char* at = getenv("RX_DUMP_SSBO_AT")) {   // (the guest's copy, before the GPU's is read back)
                    uint32_t v[8]{};
                    gpu_read(raw_va + strtoull(at, nullptr, 16), v, sizeof v);
                    hos_log("rx-frame:       guest +%s: %08x %08x %08x %08x %08x %08x %08x %08x (gpu-written %d)", at, v[0],
                            v[1], v[2], v[3], v[4], v[5], v[6], v[7], (int)gpu_written(raw_va, size));
                }
                if (gpu_written(raw_va, size)) {   // (what the GPU holds: submit, wait, read back)
                    flush_batch();
                    wait_submitted();
                    buffer_cache_sync_to_guest(raw_va, size);
                }
                gpu_read(raw_va, w, sizeof w);
                hos_log("rx-frame:     %s ssbo %#llx size %#llx gpu-written %d: %08x %08x %08x %08x %08x %08x %08x %08x "
                        "%08x %08x %08x %08x", stage ? "fs" : "vs", (unsigned long long)raw_va,
                        (unsigned long long)size, (int)gpu_written(raw_va, size), w[0], w[1], w[2], w[3], w[4], w[5],
                        w[6], w[7], w[8], w[9], w[10], w[11]);
                // RX_DUMP_SSBO_AT=<hex byte offset>: also 8 words from there (as hex and as floats)
                if (const char* at = getenv("RX_DUMP_SSBO_AT")) {
                    const uint64_t off = strtoull(at, nullptr, 16);
                    uint32_t v[8]{};
                    gpu_read(raw_va + off, v, sizeof v);
                    float f[8];
                    std::memcpy(f, v, sizeof f);
                    hos_log("rx-frame:       +%#llx: %08x %08x %08x %08x %08x %08x %08x %08x  (%g %g %g %g %g %g %g %g)",
                            (unsigned long long)off, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], f[0], f[1], f[2],
                            f[3], f[4], f[5], f[6], f[7]);
                }
            }
        // RX_CS_WATCH=<GPU VA hex> (debugging): draws whose programs write storage covering it are logged too
        if (static const uint64_t watch = getenv("RX_CS_WATCH") ? strtoull(getenv("RX_CS_WATCH"), nullptr, 16) : 0;
            (watch && writable && va <= watch && watch < va + size) || (writable && guest_watch_hit(va, size))) {
            static int logged;
            if (logged++ < 100)
                hos_log("rx: draw (fs %#llx) writes storage %#llx+%#llx (watch %#llx)",
                        (unsigned long long)(r.program_region.Address() + r.pipelines[5].offset), (unsigned long long)va,
                        (unsigned long long)size, (unsigned long long)watch);
        }
        const StorageBinding sb = draw_storage(va, size, writable);
        if (!sb.ok) {
            out.ok = false;
            return;
        }
        const bool dynamic = sb.in_stream && dynamic_storage > 0;
        dynamic_storage -= dynamic;
        buffer_entry(b, writable ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage, dynamic,
                     sb.buffer, sb.offset, sb.size);
        b += d.count;
    }
    if (!info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty() ||
        !info.image_descriptors.empty()) {
        log_limited("images", 8, "rx: a graphics program uses images or texture buffers: not drawn yet");
        out.ok = false;
        return;
    }
    uint32_t t_index = 0;
    if (frame_dump_active()) dump_textures.clear();
    // (web: from the 17th texture on, a texture shares the sampler of the last earlier one of its kind that has its
    // own, as the WGSL backend declares them: browsers allow 16 samplers a stage)
    uint32_t last_own[2] = {0, 0};
    for (const auto& d : info.texture_descriptors) {
        const bool shared_sampler = RX_WEB && t_index >= 16 && last_own[d.is_depth ? 1 : 0];
        if (!shared_sampler) last_own[d.is_depth ? 1 : 0] = b + 1;
        const uint32_t handle = read_handle(ch, stage, d);
        if (static const bool trace_bindings = env_flag("RX_TRACE_BINDINGS"); trace_bindings) {
            static std::unordered_map<uint64_t, bool> seen;
            const ConstBuffer& cb = ch.cbufs[stage][d.cbuf_index % 18];
            const uint64_t key = handle ^ cb.address ^ ((uint64_t)d.cbuf_offset << 40) ^ ((uint64_t)stage << 56);
            if (seen.size() < 160 && seen.emplace(key, true).second)
                hos_log("rx: texture descriptor stage %u slot %u offset %#x handle %#x TIC pool %#llx limit %u", stage,
                        d.cbuf_index, d.cbuf_offset, handle, (unsigned long long)r.tex_header.Address(),
                        r.tex_header.limit);
        }
        const auto [tic_index, tsc_index] =
            Tegra::Texture::TexturePair(handle, r.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding);
        BoundTexture t = texture_for_handle(r, handle, d);
        watch_tex_size(at, t);
        if (t.image) {   // sampling an image this pass draws into: WebGPU forbids it
            bool attached = at.depth == t.image;
            for (Image* c : at.color) attached |= c == t.image;
            if (attached) {
                Image* source = t.image;
                if (Image* snapshot = snapshot_for_feedback(*source)) {
                    const auto tic = read_tic(r, tic_index);
                    const uint32_t base_level = tic.res_min_mip_level;
                    const uint32_t level_count = (uint32_t)tic.res_max_mip_level >= base_level
                                                     ? (uint32_t)tic.res_max_mip_level - base_level + 1
                                                     : 1;
                    t.view = snapshot->view(view_dimension(d.type), has_stencil(source->format), tic.BaseLayer(),
                                            base_level, level_count);
                    t.image = nullptr;
                    log_limited("feedback_snapshot", 8,
                                "rx: attachment feedback samples a pre-draw snapshot (%ux%u format %#x)",
                                source->width, source->height, (unsigned)source->format);
                } else {
                    out.ok = false;
                    return;
                }
            }
        }
        if (frame_dump_active()) {
            char tb[128];
            const auto tic = read_tic(r, tic_index);
            const auto tsc = read_tsc_for_dump(r, handle);
            std::snprintf(tb, sizeof tb, " %c%s%08llx@%04x:%ux%u/%x[%s%s%s L%u-%u w%u%u%u b%.2f]", t.source,
                          t.kind == TexKind::Depth ? "z" : "", (unsigned long long)t.guest,
                          (unsigned)(((uintptr_t)t.image >> 4) & 0xFFFF), t.width, t.height, t.format,
                          tsc.mag_filter == Tegra::Texture::TextureFilter::Linear ? "l" : "n",
                          tsc.min_filter == Tegra::Texture::TextureFilter::Linear ? "l" : "n",
                          tsc.mipmap_filter == Tegra::Texture::TextureMipmapFilter::Linear ? "l"
                          : tsc.mipmap_filter == Tegra::Texture::TextureMipmapFilter::Nearest ? "n" : "-",
                          (unsigned)tic.res_min_mip_level, (unsigned)tic.max_mip_level, (unsigned)tsc.wrap_u.Value(),
                          (unsigned)tsc.wrap_v.Value(), (unsigned)tsc.wrap_p.Value(), tsc.LodBias());
            dump_textures += tb;
            // the header's format: TextureFormat, component types (r g b a), sRGB, swizzle
            std::snprintf(tb, sizeof tb, "{f%x t%u%u%u%u s%u w%u%u%u%u}", (unsigned)tic.format.Value(),
                          (unsigned)tic.r_type.Value(), (unsigned)tic.g_type.Value(), (unsigned)tic.b_type.Value(),
                          (unsigned)tic.a_type.Value(), (unsigned)tic.srgb_conversion.Value(),
                          (unsigned)tic.x_source.Value(), (unsigned)tic.y_source.Value(), (unsigned)tic.z_source.Value(),
                          (unsigned)tic.w_source.Value());
            dump_textures += tb;
            // RX_DUMP_TEXINFO=<vertex program hex> (debugging): for that program's first draw in the dump, where each
            // texture's memory stands: nonzero bytes, dispatch-written, render targets over it
            if (static const uint64_t want = getenv("RX_DUMP_TEXINFO") ? strtoull(getenv("RX_DUMP_TEXINFO"), nullptr, 16) : 0;
                want && want == r.program_region.Address() + r.pipelines[1].offset) {
                static int64_t done_frame = -1;
                static uint64_t done_handle_mask = 0;
                if (done_frame != frame_number) done_frame = frame_number, done_handle_mask = 0;
                const uint64_t bit = 1ull << (t_index % 64);
                // (RX_DUMP_TEXINFO_ALL=1: every draw, not only the first)
                static const bool every = env_flag("RX_DUMP_TEXINFO_ALL");
                if (!(done_handle_mask & bit) || every) {
                    done_handle_mask |= bit;
                    const uint64_t gva = tic.Address();
                    const uint64_t n = std::min<uint64_t>((uint64_t)t.width * t.height * 4, 4u << 20);
                    std::vector<uint8_t> mem(n);
                    gpu_read(gva, mem.data(), n);
                    size_t nz = 0;
                    for (uint8_t v : mem) nz += v != 0;
                    std::string over;
                    for (const auto& [g, im] : all_images())
                        if (im && im->guest < t.guest + n && t.guest < im->guest + (uint64_t)im->width * im->height * 4) {
                            char ob[96];
                            std::snprintf(ob, sizeof ob, " rt %08llx (gpu %#llx) %ux%u f%d r%d", (unsigned long long)im->guest, (unsigned long long)im->gpu_va,
                                          im->width, im->height, (int)im->format, (int)im->rendered);
                            over += ob;
                        }
                    if (over.size() > 300) over.resize(300);
                    {   // the texture header pool's entries near that memory (index: address)
                        const uint64_t pool = r.tex_header.Address();
                        const uint32_t entries = std::min<uint32_t>(r.tex_header.limit + 1, 1u << 16);
                        std::vector<Tegra::Texture::TICEntry> tics(entries);
                        gpu_read(pool, tics.data(), (uint64_t)entries * sizeof(Tegra::Texture::TICEntry));
                        over += " | tic " + std::to_string(tic_index) + " of " + std::to_string(entries) + ":";
                        {   // where the handle came from
                            const auto& cbh = ch.cbufs[stage][d.cbuf_index % 18];
                            char hb[200];
                            std::snprintf(hb, sizeof hb, " [handle %#x from cbuf %u+%#x at %#llx size %#x, dispatch-written %d,"
                                          " cpu-written-since-1 %d]", handle, d.cbuf_index, d.cbuf_offset,
                                          (unsigned long long)cbh.address, cbh.size, (int)gpu_written(cbh.address, cbh.size),
                                          (int)memory_written_since(hos_gpu_translate(cbh.address), cbh.size, 1));
                            over += hb;
                        }
                        // (RX_DUMP_TICS=<file>: every entry's index, address, size and format, once)
                        static bool tics_done;
                        if (const char* f = getenv("RX_DUMP_TICS"); f && !tics_done) {
                            tics_done = true;
                            if (FILE* o = std::fopen(f, "w")) {
                                for (uint32_t k = 0; k < entries; k++)
                                    std::fprintf(o, "%u %#llx %ux%u f%#x\n", k, (unsigned long long)tics[k].Address(),
                                                 tics[k].Width(), tics[k].Height(), (unsigned)tics[k].format.Value());
                                std::fprintf(o, "history\n%s\n", rt_history_find(0, ~0ull).c_str());
                                std::fclose(o);
                            }
                        }
                        int shown = 0;
                        for (uint32_t k = 0; k < entries && shown < 12; k++) {
                            const uint64_t a = tics[k].Address();
                            if (a + 0x40000 > gva && a < gva + 0x40000) {
                                char ob[64];
                                std::snprintf(ob, sizeof ob, " %u:%#llx", k, (unsigned long long)a);
                                over += ob;
                                shown++;
                            }
                        }
                    }
                    const uint64_t g = hos_gpu_translate(gva);
                    int64_t first_seq = -1, last_seq = -1;
                    for (uint64_t q = 1; q < (1ull << 40); q <<= 1)   // (the newest write: the smallest q not passed)
                        if (!memory_written_since(g, n, q)) { last_seq = q; break; }
                    first_seq = memory_written_since(g, n, 0) ? 1 : 0;
                    hos_log("rx-frame:     texinfo %c%08llx gpu %#llx %ux%u: %zu of %llu bytes nonzero, dispatch-written %d,"
                            " cpu-written %lld (last write seq < %lld, now %llu);%s;"
                            " history:%s", t.source, (unsigned long long)t.guest, (unsigned long long)gva, t.width,
                            t.height, nz, (unsigned long long)n, (int)gpu_written(gva, n), (long long)first_seq,
                            (long long)last_seq, (unsigned long long)memory_seq(), over.c_str(),
                            "");
                    const std::string hist = rt_history_find(gva - 0x40000, gva + 0x40000);
                    for (size_t k = 0; k < hist.size(); k += 900)
                        hos_log("rx-frame:       history %s", hist.substr(k, 900).c_str());
                }
            }
        }
        // RX_LOG_TILE_SRC=<width> (debugging): the textures draws into a target that wide (color 0) sample
        if (static const uint32_t tw = getenv("RX_LOG_TILE_SRC") ? atoi(getenv("RX_LOG_TILE_SRC")) : 0;
            tw && r.rt[0].width == tw && r.rt[0].height == 260 && r.rt[0].format != Tegra::RenderTargetFormat::NONE) {
            static int logged;
            if (logged++ < 3000)
                hos_log("rx: tile target %#llx samples %c%08llx (gpu %#llx) %ux%u frame %lld",
                        (unsigned long long)r.rt[0].Address(), t.source, (unsigned long long)t.guest,
                        (unsigned long long)read_tic(r, tic_index).Address(), t.width, t.height, (long long)frame_number);
        }
        // the header's component swizzle and the sampler's border, in the uniform block
        std::memcpy(slot_bytes.data() + (size_t)t_index++ * 32, t.slots, sizeof t.slots);
        WGPUBindGroupLayoutEntry tl{};
        tl.binding = b;
        tl.visibility = vis;
        tl.texture.sampleType = t.kind == TexKind::Float               ? WGPUTextureSampleType_Float
                                : t.kind == TexKind::UnfilterableFloat ? WGPUTextureSampleType_UnfilterableFloat
                                : t.kind == TexKind::Depth             ? WGPUTextureSampleType_Depth
                                : t.kind == TexKind::Uint              ? WGPUTextureSampleType_Uint
                                                                       : WGPUTextureSampleType_Sint;
        tl.texture.viewDimension = view_dimension(d.type);
        tl.texture.multisampled = d.is_multisample;
        out.layout.push_back(pack_layout(tl));
        WGPUBindGroupLayoutEntry sl{};
        sl.binding = b + 1;
        sl.visibility = vis;
        sl.sampler.type = t.kind == TexKind::Depth             ? WGPUSamplerBindingType_Comparison
                          : t.kind == TexKind::UnfilterableFloat ? WGPUSamplerBindingType_NonFiltering
                                                                 : WGPUSamplerBindingType_Filtering;
        if (!shared_sampler) out.layout.push_back(pack_layout(sl));
        if (RX_WEB && !shared_sampler && sl.sampler.type == WGPUSamplerBindingType_NonFiltering &&
            !sampler_nonfiltering(t.sampler))
            log_limited("sampler_mismatch", 12, "rx: non-filtering binding %u (stage %u) gets a filtering sampler: "
                        "texture %ux%u format %d kind %d source %c image %p sampler %p", b + 1, stage, t.width, t.height,
                        t.format, (int)t.kind, t.source, (void*)t.image, (void*)t.sampler);
        WGPUBindGroupEntry te{};
        te.binding = b;
        te.textureView = t.view;
        out.entries.push_back(te);
        WGPUBindGroupEntry se{};
        se.binding = b + 1;
        se.sampler = t.sampler;
        if (!shared_sampler) out.entries.push_back(se);
        b += 2;
    }
    out.next_binding = b;

    // ---- the uniform block: rescale, the texture slots, the constant buffers that fit ----
    struct UbMemo {
        const Shader::Info* info = nullptr;
        int64_t batch = -1;
        uint64_t seq = 0, ub = 0;
        uint32_t ub_size = 0;
        uint8_t rescale[64]{};
        std::vector<uint8_t> slots;
        std::array<std::array<uint64_t, 2>, 18> cbufs{};
    };
    static UbMemo memos[256];
    static const bool no_reuse = env_flag("RX_NO_UB_REUSE");
    std::array<std::array<uint64_t, 2>, 18> cbs{};
    for (const auto& d : info.constant_buffer_descriptors)
        if (in_ub[d.index % 18]) {
            const ConstBuffer& cb = ch.cbufs[stage][d.index % 18];
            cbs[d.index % 18] = {cb.address, uint64_t{cb.size} | uint64_t{cb.enabled ? 1u : 0u} << 32};
        }
    UbMemo& m = memos[((uintptr_t)&info >> 4 ^ stage) & 255];
    bool reuse = !no_reuse && !frame_dump_active() && m.info == &info && m.batch == batch_number() &&
                 m.ub_size == ub_size && m.cbufs == cbs && !std::memcmp(m.rescale, rescale, 64) && m.slots == slot_bytes;
    if (reuse)
        for (const auto& d : info.constant_buffer_descriptors) {   // (and none of their memory written since)
            if (!in_ub[d.index % 18]) continue;
            const ConstBuffer& cb = ch.cbufs[stage][d.index % 18];
            if (!cb.enabled) continue;
            const uint64_t size = (uint64_t)std::max(Common::DivCeil(info.constant_buffer_used_sizes[d.index], 16U), 1U) * 16;
            const uint64_t guest = hos_gpu_translate(cb.address);
            if (!guest || memory_written_since(guest, std::min<uint64_t>(size, cb.size), m.seq)) {
                reuse = false;
                break;
            }
        }
    if (reuse) {
        out.dynamic_offsets[ub_dynamic] = (uint32_t)m.ub;
        count("ub_reused");
        return;
    }
    const uint64_t seq = memory_seq();
    const uint64_t used_before = stream_ro.used;
    const uint64_t ub = stream_ro.alloc((uint64_t)ub_size * 16, dev.limits.minUniformBufferOffsetAlignment);
    if (ub == UINT64_MAX) {
        out.ok = false;
        return;
    }
    uint8_t* block = stream_ro.at(ub);
    std::memcpy(block, rescale, 64);
    for (const auto& d : info.constant_buffer_descriptors) {
        if (!in_ub[d.index % 18]) continue;
        const ConstBuffer& cb = ch.cbufs[stage][d.index % 18];
        if (cb.enabled && frame_dump_active() && gpu_written(cb.address, cb.size)) {   // (diagnostic)
            count("cbuf_gpu_written");
            log_limited("cbuf_gpu", 8, "rx: constant buffer %#llx (%#x bytes, stage %u slot %u) holds what a dispatch wrote",
                        (unsigned long long)cb.address, cb.size, stage, d.index);
        }
        const uint64_t size = (uint64_t)std::max(Common::DivCeil(info.constant_buffer_used_sizes[d.index], 16U), 1U) * 16;
        const uint64_t n = cb.enabled ? std::min<uint64_t>(size, cb.size) : 0;
        uint8_t* dst = block + (uint64_t)base[d.index % 18] * 16;
        if (n) gpu_read(cb.address, dst, n);
        if (n < size) std::memset(dst + n, 0, size - n);
        count("up_cbuf", (int64_t)size);
    }
    if (!slot_bytes.empty()) std::memcpy(block + 64, slot_bytes.data(), slot_bytes.size());
    // the same bytes as a block this batch already holds: that one is bound and this space given back (many objects
    // carry equal constants at their own addresses: less to upload; RX_NO_UB_DEDUPE=1 off)
    uint64_t bound = ub;
    static const bool no_dedupe = env_flag("RX_NO_UB_DEDUPE");
    if (!no_dedupe) {
        // (a direct-mapped table tagged with the batch: no allocation per block and nothing to clear; a slot taken
        // by another block only misses a dedupe. An unordered_map's node per block was ~3% of the replay thread)
        struct Same { uint64_t h = 0, ub = 0; int64_t batch = -1; };
        static std::vector<Same> same(1u << 14);
        const uint64_t bytes = (uint64_t)ub_size * 16;
        const uint64_t h = hash_bytes(block, bytes, bytes);
        Same& e = same[(h ^ h >> 29) & ((1u << 14) - 1)];
        if (e.batch == batch_number() && e.h == h) {
            if (!std::memcmp(stream_ro.at(e.ub), block, bytes) && stream_ro.used == ub + bytes) {
                bound = e.ub;
                stream_ro.used = used_before;
                count("ub_deduped", (int64_t)bytes);
            }
        } else {
            e = {h, ub, batch_number()};
        }
    }
    out.dynamic_offsets[ub_dynamic] = (uint32_t)bound;
    m.info = &info;
    m.batch = batch_number();
    m.seq = seq;
    m.ub = bound;
    m.ub_size = ub_size;
    std::memcpy(m.rescale, rescale, 64);
    m.slots = slot_bytes;
    m.cbufs = cbs;
    // RX_DUMP_UB=<pixel program hex> (with a frame dump): that program's uniform block is logged
    if (frame_dump_active())
        if (const char* want = getenv("RX_DUMP_UB"); want && strtoull(want, nullptr, 16) == dump_fs_program) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(block);
            for (const auto& d : info.constant_buffer_descriptors)   // (where each part comes from)
                if (in_ub[d.index % 18]) {
                    const ConstBuffer& cb = ch.cbufs[stage][d.index % 18];
                    hos_log("rx-frame:     %s ub[%u..] = cbuf %u at gpu %#llx (guest %#llx) size %#x", stage ? "fs" : "vs",
                            base[d.index % 18], d.index, (unsigned long long)cb.address,
                            (unsigned long long)hos_gpu_translate(cb.address), cb.size);
                }
            for (uint32_t i = 0; i < std::min<uint32_t>(ub_size, 1024); i++) {
                float f[4];
                std::memcpy(f, w + i * 4, 16);
                hos_log("rx-frame:     %s ub[%u] %08x %08x %08x %08x  (%g %g %g %g)", stage ? "fs" : "vs", i, w[i * 4], w[i * 4 + 1],
                        w[i * 4 + 2], w[i * 4 + 3], f[0], f[1], f[2], f[3]);
            }
        }
}

/// The pulled vertex streams (see PullPlan), after the vertex stage's other resources: a uniform (where each stream
/// starts in its binding; the instance base) and each stream as read-only storage from the buffer cache.
bool pulled_streams(StageResources& out, const Regs& r, const PullPlan& plan, const DrawParams& p, bool cached) {
    if (!plan.count) return true;
    uint32_t pull[4] = {0, 0, 0, cached ? 0u : p.base_instance};
    struct Bind {
        BufferRef ref;
        uint64_t offset, size;
    } binds[3];
    const uint64_t align = dev.limits.minStorageBufferOffsetAlignment;
    for (uint32_t k = 0; k < plan.count; k++) {
        const uint32_t s = plan.streams[k];
        const auto& vs = r.vertex_streams[s];
        const uint64_t addr = vs.Address(), limit = r.vertex_stream_limits[s].Address();
        uint64_t bytes = limit >= addr ? limit - addr + 1 : 0;
        const bool instanced = r.vertex_stream_instances.IsInstancingEnabled(s);
        if (!p.indexed && !instanced) bytes = std::min<uint64_t>(bytes, (uint64_t)(p.first + p.count) * vs.stride + 64);
        if (instanced) bytes = std::min<uint64_t>(bytes, (uint64_t)(p.base_instance + p.instance_count) * vs.stride + 64);
        // (no limit set, or more than the buffer cache binds at once: an 8 MB window, as far as the memory is mapped;
        // the shader's reads past it come back as zeros)
        uint64_t avail = 0;
        if (!hos_gpu_ptr(addr, &avail)) return count("pull_unmapped"), false;
        if (!bytes || bytes > (8ull << 20)) bytes = 8ull << 20;
        bytes = std::min<uint64_t>(bytes, avail);
        if (!bytes) return count("pull_empty"), false;
        const BufferRef ref = guest_buffer(addr, bytes);
        if (!ref.buffer) return count(bytes > (8ull << 20) ? "pull_big" : "pull_no_buffer"), false;
        const uint64_t off = ref.offset & ~(align - 1);
        pull[k] = (uint32_t)(ref.offset - off);
        uint64_t size = (ref.size + pull[k] + 3) & ~3ull;
        size = std::min<uint64_t>(size, wgpuBufferGetSize(ref.buffer) - off);
        binds[k] = {ref, off, size};
    }
    const uint64_t u = stream_ro.alloc(16, dev.limits.minUniformBufferOffsetAlignment);
    if (u == UINT64_MAX) return count("pull_space"), false;
    std::memcpy(stream_ro.at(u), pull, 16);
    uint32_t b = out.next_binding;
    WGPUBindGroupLayoutEntry l{};
    l.binding = b;
    l.visibility = WGPUShaderStage_Vertex;
    l.buffer.type = WGPUBufferBindingType_Uniform;
    l.buffer.hasDynamicOffset = true;
    out.layout.push_back(pack_layout(l));
    WGPUBindGroupEntry e{};
    e.binding = b++;
    e.buffer = stream_ro.buffer;
    e.size = 16;
    out.entries.push_back(e);
    out.dynamic_offsets.push_back((uint32_t)u);
    for (uint32_t k = 0; k < plan.count; k++) {
        WGPUBindGroupLayoutEntry sl{};
        sl.binding = b;
        sl.visibility = WGPUShaderStage_Vertex;
        sl.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        out.layout.push_back(pack_layout(sl));
        WGPUBindGroupEntry se{};
        se.binding = b++;
        se.buffer = binds[k].ref.buffer;
        se.offset = binds[k].offset;
        se.size = binds[k].size;
        out.entries.push_back(se);
    }
    count("pulled_draws");
    return true;
}

/// Bind groups by contents (dynamic bindings are at offset 0 in their entries, so a group serves every draw with
/// the same buffers, textures and samplers). Looked up by hash first: no key is built for a hit.
struct CachedBindGroup {
    WGPUBindGroupLayout layout;
    std::vector<WGPUBindGroupEntry> entries;
    WGPUBindGroup group;
    int64_t last_frame = 0;
};
std::unordered_multimap<uint64_t, CachedBindGroup> bind_groups;
std::unordered_map<WGPUTextureView, std::vector<uint64_t>> groups_by_view;
size_t groups_by_view_entries = 0, group_views_live = 0;

size_t index_group_views(uint64_t h, const std::vector<WGPUBindGroupEntry>& entries) {
    size_t n = 0;
    for (const WGPUBindGroupEntry& e : entries)
        if (e.textureView) {
            groups_by_view[e.textureView].push_back(h);
            n++;
        }
    groups_by_view_entries += n;
    return n;
}

size_t group_views(const CachedBindGroup& g) {
    size_t n = 0;
    for (const WGPUBindGroupEntry& e : g.entries) n += e.textureView != nullptr;
    return n;
}

void rebuild_group_index() {
    groups_by_view.clear();
    groups_by_view_entries = group_views_live = 0;
    for (const auto& [h, g] : bind_groups) group_views_live += index_group_views(h, g.entries);
}

/// The cache is emptied when large (or when views in its keys are about to be released): at the start of a draw,
/// never between a draw's bind_group calls (that released the vertex stage's group before the draw bound it: wgpu
/// panicked "BindGroup ... is no longer alive").
uint64_t bind_group_trims;   // bumped when the cached groups are released (bind_group's last-group check)
/// Only the groups that name a view about to be released leave the cache (its pointer could come back as another
/// view's), or all of them when it is large; they are released after the batches recorded so far are replayed. (It
/// used to wait for those batches and drop every group: a 25 ms frame every ~3 s, rebuilding ~1500 groups.)
void trim_bind_groups() {
    const bool big = bind_groups.size() > 20000, buffers = segments_retired_pending();
    if (!big && !buffers && !retired_views_pending()) return;
    const int64_t t0 = hos_now_ns();
    int64_t dropped = 0;
    static const bool full_scan = env_flag("RX_TRIM_FULL_SCAN");
    if (big) {
        for (auto it = bind_groups.begin(); it != bind_groups.end();) {
            if (it->second.last_frame < frame_number - 600) {
                release_bind_group_later(it->second.group);
                it = bind_groups.erase(it);
                dropped++;
            } else {
                ++it;
            }
        }
    }
    const bool all = bind_groups.size() > 20000;
    if (all || full_scan || buffers || big) {
        std::unordered_set<WGPUTextureView> gone(retired_view_list().begin(), retired_view_list().end());
        const std::vector<WGPUBuffer> retired_buffers = take_retired_segment_buffers();
        const std::unordered_set<WGPUBuffer> gone_buffers(retired_buffers.begin(), retired_buffers.end());
        for (auto it = bind_groups.begin(); it != bind_groups.end();) {
            bool drop = all;
            for (size_t i = 0; i < it->second.entries.size() && !drop; i++)
                drop = (it->second.entries[i].textureView && gone.count(it->second.entries[i].textureView)) ||
                       (it->second.entries[i].buffer && gone_buffers.count(it->second.entries[i].buffer));
            if (drop) {
                release_bind_group_later(it->second.group);
                it = bind_groups.erase(it);
                dropped++;
            } else {
                ++it;
            }
        }
        rebuild_group_index();
        for (WGPUBuffer x : retired_buffers) release_buffer_later(x);
    } else {
        for (WGPUTextureView v : retired_view_list()) {
            auto vi = groups_by_view.find(v);
            if (vi == groups_by_view.end()) continue;
            for (uint64_t h : vi->second) {
                auto [a, b] = bind_groups.equal_range(h);
                for (auto it = a; it != b;) {
                    bool names = false;
                    for (const WGPUBindGroupEntry& e : it->second.entries) names |= e.textureView == v;
                    if (names) {
                        release_bind_group_later(it->second.group);
                        group_views_live -= group_views(it->second);
                        it = bind_groups.erase(it);
                        dropped++;
                    } else {
                        ++it;
                    }
                }
            }
            groups_by_view_entries -= vi->second.size();
            groups_by_view.erase(vi);
        }
        if (groups_by_view_entries > 2 * group_views_live + 65536) rebuild_group_index();
    }
    bind_group_trims++;
    release_retired_views();
    count("bind_groups_trimmed", dropped);
    count("t_trim", hos_now_ns() - t0);
    spike.trim_ns += hos_now_ns() - t0;
}

WGPUBindGroup bind_group(WGPUBindGroupLayout layout, const std::vector<WGPUBindGroupEntry>& entries) {
    const size_t bytes = entries.size() * sizeof(WGPUBindGroupEntry);
    // the previous group made with this layout (consecutive draws of one material share theirs): no hash then
    struct Last {
        WGPUBindGroupLayout layout = nullptr;
        std::vector<WGPUBindGroupEntry> entries;
        WGPUBindGroup group = nullptr;
        CachedBindGroup* entry = nullptr;
        uint64_t trims = ~0ull;
    };
    static Last last[64];
    Last& l = last[((uintptr_t)layout >> 4) & 63];
    if (l.layout == layout && l.trims == bind_group_trims && l.entries.size() == entries.size() &&
        !std::memcmp(l.entries.data(), entries.data(), bytes)) {
        if (l.entry) l.entry->last_frame = frame_number;
        return l.group;
    }
    const uint64_t h = hash_bytes(entries.data(), bytes, (uint64_t)(uintptr_t)layout);
    auto found = [&](CachedBindGroup& e) {
        e.last_frame = frame_number;
        l.layout = layout;
        l.entries = entries;
        l.group = e.group;
        l.entry = &e;
        l.trims = bind_group_trims;
        return e.group;
    };
    // the group last found for this hash (the map's buckets cost ~3% of the replay thread in cache misses)
    struct Recent {
        uint64_t h = 0, trims = ~0ull;
        CachedBindGroup* entry = nullptr;
    };
    static Recent recent[4096];
    Recent& r = recent[(h ^ h >> 32) & 4095];
    if (r.entry && r.h == h && r.trims == bind_group_trims && r.entry->layout == layout &&
        r.entry->entries.size() == entries.size() && !std::memcmp(r.entry->entries.data(), entries.data(), bytes))
        return found(*r.entry);
    auto [a, b] = bind_groups.equal_range(h);
    for (auto it = a; it != b; ++it)
        if (it->second.layout == layout && it->second.entries.size() == entries.size() &&
            !std::memcmp(it->second.entries.data(), entries.data(), bytes)) {
            r = {h, bind_group_trims, &it->second};
            return found(it->second);
        }
    WGPUBindGroupDescriptor bd{};
    bd.layout = layout;
    bd.entryCount = entries.size();
    bd.entries = entries.data();
    WGPUBindGroup g = wgpuDeviceCreateBindGroup(dev.device, &bd);
    spike.bind_groups++;
    count("bind_groups");
    auto inserted = bind_groups.emplace(h, CachedBindGroup{layout, entries, g, frame_number});
    group_views_live += index_group_views(h, entries);
    r = {h, bind_group_trims, &inserted->second};
    l.layout = layout;
    l.entries = entries;
    l.group = g;
    l.entry = &inserted->second;
    l.trims = bind_group_trims;
    return g;
}

struct VertexInput {
    std::vector<WGPUVertexBufferLayout> layouts;
    // one list per layout; the first nattr are this draw's (the lists are kept and reused: allocations per stream
    // and draw cost the GPU thread several percent)
    std::vector<std::vector<WGPUVertexAttribute>> attributes;
    size_t nattr = 0;
    std::vector<WGPUVertexAttribute>& next_attributes() {
        if (nattr == attributes.size()) attributes.emplace_back();
        auto& v = attributes[nattr++];
        v.clear();
        return v;
    }
    void clear() {
        layouts.clear();
        nattr = 0;
        bindings.clear();
        key.clear();
    }
    struct Binding {
        WGPUBuffer buffer;
        uint64_t offset, size;
    };
    std::vector<Binding> bindings;
    std::vector<uint32_t> key;
};

/// The vertex buffers of a draw: each stream the program reads, copied for vertices [v0, v1] (instanced streams:
/// the draw's instances).
/// Cached mode (the buffer cache): streams are bound where they are in guest memory, for the elements the draw
/// can read: [0, first + count) vertices of an array draw, up to the stream's limit for an indexed draw.
struct CachedDraw {
    bool on = false, indexed = false;
    uint32_t first = 0, count = 0;
};

bool vertex_input(VertexInput& vi, const Regs& r, const Shader::Info& info, uint32_t v0, uint32_t v1,
                  uint32_t base_instance, uint32_t instances, const CachedDraw& cached = {},
                  const PullPlan* pull = nullptr) {
    struct Attr {
        uint32_t location, offset, size;
        WGPUVertexFormat format;
    };
    // (reused across calls: the GPU thread's allocations were ~10% of its time)
    static std::array<std::vector<Attr>, 32> streams;
    for (auto& st : streams) st.clear();
    static std::vector<WGPUVertexAttribute> constants;
    constants.clear();
    static std::vector<uint32_t> converted;   // attributes decoded on the CPU
    converted.clear();
    for (uint32_t i = 0; i < 32; i++) {
        if (!info.loads.Generic(i)) continue;
        const auto& a = r.vertex_attrib_format[i];
        uint32_t size = 0;
        WGPUVertexFormat f = vertex_format(a, size);
        if (vertex_unpack(a)) {   // fetched as one u32, unpacked by the vertex stage
            f = WGPUVertexFormat_Uint32;
            size = 4;
        }
        const auto& vs = r.vertex_streams[a.buffer % 16];
        const uint32_t stride = vs.stride;
        const bool constant = a.constant || !vs.IsEnabled();
        // WebGPU: strides are multiples of 4, offsets multiples of min(4, the format's size), and an attribute ends
        // within its stride
        if (!constant && pull && pull->attr[i].enabled) continue;   // read by the vertex stage itself
        if (!constant && (!f || (stride & 3) || (size && a.offset % std::min(4u, size)) ||
                          (stride && a.offset + size > stride) || (!stride && a.offset + size > 2048))) {
            log_limited("vconv", 12, "rx: vertex attribute %u (format %#x, offset %u, stride %u) decoded on the CPU",
                        i, a.hex, (unsigned)a.offset, stride);
            converted.push_back(i);
            continue;
        }
        if (constant) {
            WGPUVertexAttribute c{};
            c.shaderLocation = i;
            c.offset = 0;
            using T = Regs::VertexAttribute::Type;
            c.format = a.type == T::SInt || a.type == T::SScaled   ? WGPUVertexFormat_Sint32x4
                       : a.type == T::UInt || a.type == T::UScaled ? WGPUVertexFormat_Uint32x4
                                                                   : WGPUVertexFormat_Float32x4;
            if (a.constant) c.format = WGPUVertexFormat_Float32x4;
            constants.push_back(c);
            vi.key.insert(vi.key.end(), {0xC0000000u | i, (uint32_t)c.format});
            continue;
        }
        streams[a.buffer % 16].push_back({i, a.offset, size, f});
    }
    if (cached.on && !converted.empty()) {
        rx::count("nc_converted");
        return false;
    }
    for (uint32_t s = 0; s < 16; s++) {
        if (streams[s].empty()) continue;
        const auto& vs = r.vertex_streams[s];
        const uint32_t stride = vs.stride;
        const bool instanced = r.vertex_stream_instances.IsInstancingEnabled(s);
        if (instanced && vs.frequency > 1)
            log_limited("vdivisor", 4, "rx: instance divisor %u taken as 1", (unsigned)vs.frequency);
        if (cached.on) {
            uint32_t end = 0;
            for (const Attr& a : streams[s]) end = std::max(end, a.offset + a.size);
            const uint64_t addr = vs.Address(), limit = r.vertex_stream_limits[s].Address();
            uint64_t bytes;
            if (!stride) bytes = end;
            else if (instanced) bytes = (uint64_t)(base_instance + instances - 1) * stride + end;
            else if (!cached.indexed) bytes = (uint64_t)(cached.first + cached.count - 1) * stride + end;
            else if (limit >= addr) bytes = std::max<uint64_t>(limit - addr + 1, end);
            else {
                rx::count("nc_limit");
                return false;
            }
            bytes = (bytes + 3) & ~3ull;
            if (addr & 3) {
                rx::count("nc_align");
                return false;
            }
            const BufferRef b = guest_buffer(addr, bytes);
            if (!b.buffer) {
                rx::count(bytes > (8u << 20) ? "nc_big" : "nc_unmapped");
                return false;
            }
            auto& attrs = vi.next_attributes();
            for (const Attr& a : streams[s]) {
                WGPUVertexAttribute w{};
                w.format = a.format;
                w.offset = a.offset;
                w.shaderLocation = a.location;
                attrs.push_back(w);
                vi.key.insert(vi.key.end(), {s << 16 | a.location, (uint32_t)a.format, a.offset});
            }
            vi.key.insert(vi.key.end(), {0x80000000u | s, stride, instanced ? 1u : 0u});
            WGPUVertexBufferLayout l{};
            l.stepMode = instanced ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
            l.arrayStride = stride;
            vi.layouts.push_back(l);
            vi.bindings.push_back({b.buffer, b.offset, b.size});
            continue;
        }
        const uint64_t first = instanced ? base_instance : v0;
        const uint64_t count = instanced ? instances : (uint64_t)v1 - v0 + 1;
        uint32_t end = 0;
        for (const Attr& a : streams[s]) end = std::max(end, a.offset + a.size);
        uint64_t bytes = stride ? (count - 1) * stride + end : end;
        bytes = (bytes + 3) & ~3ull;
        if (bytes > (64ull << 20)) {
            log_limited("vbig", 8, "rx: %llu bytes of vertices for one draw: skipped", (unsigned long long)bytes);
            return false;
        }
        const uint64_t off = stream_ro.alloc(bytes, 16);
        if (off == UINT64_MAX) return false;
        const uint64_t src = vs.Address() + first * stride;
        const uint64_t limit = r.vertex_stream_limits[s].Address();
        uint64_t n = bytes;
        if (limit >= src && limit - src + 1 < n) n = limit - src + 1;
        if (limit < src) n = 0;
        if (n) gpu_read(src, stream_ro.at(off), n);
        if (n < bytes) std::memset(stream_ro.at(off + n), 0, bytes - n);
        rx::count("up_vertex", (int64_t)bytes);
        auto& attrs = vi.next_attributes();
        for (const Attr& a : streams[s]) {
            WGPUVertexAttribute w{};
            w.format = a.format;
            w.offset = a.offset;
            w.shaderLocation = a.location;
            attrs.push_back(w);
            vi.key.insert(vi.key.end(), {s << 16 | a.location, (uint32_t)a.format, a.offset});
        }
        vi.key.insert(vi.key.end(), {0x80000000u | s, stride, instanced ? 1u : 0u});
        WGPUVertexBufferLayout l{};
        l.stepMode = instanced ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
        l.arrayStride = stride;
        vi.layouts.push_back(l);
        vi.bindings.push_back({stream_ro.buffer, off, bytes});
    }
    for (uint32_t i : converted) {
        const auto& a = r.vertex_attrib_format[i];
        const auto& vs = r.vertex_streams[a.buffer % 16];
        const uint32_t stride = vs.stride;
        const bool instanced = r.vertex_stream_instances.IsInstancingEnabled(a.buffer % 16);
        const uint64_t first = instanced ? base_instance : v0;
        const uint64_t count = instanced ? instances : (uint64_t)v1 - v0 + 1;
        const uint64_t off = stream_ro.alloc(count * 16, 16);
        if (off == UINT64_MAX) return false;
        std::vector<uint8_t> src((size_t)((count - 1) * stride + a.offset + 16));
        gpu_read(vs.Address() + first * stride, src.data(), src.size());
        uint8_t* dst = stream_ro.at(off);
        for (uint64_t e = 0; e < count; e++) {
            uint32_t v[4];
            decode_attribute(src.data() + e * stride + a.offset, a, v);
            std::memcpy(dst + e * 16, v, 16);
        }
        using T = Regs::VertexAttribute::Type;
        WGPUVertexAttribute w{};
        w.shaderLocation = i;
        w.offset = 0;
        w.format = a.type == T::SInt || a.type == T::SScaled   ? WGPUVertexFormat_Sint32x4
                   : a.type == T::UInt || a.type == T::UScaled ? WGPUVertexFormat_Uint32x4
                                                               : WGPUVertexFormat_Float32x4;
        vi.next_attributes().push_back(w);
        WGPUVertexBufferLayout l{};
        l.stepMode = instanced ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
        l.arrayStride = 16;
        vi.layouts.push_back(l);
        vi.bindings.push_back({stream_ro.buffer, off, count * 16});
        vi.key.insert(vi.key.end(), {0xA0000000u | i, (uint32_t)w.format, instanced ? 1u : 0u});
    }
    if (!constants.empty()) {
        auto& ca = vi.next_attributes();
        ca.assign(constants.begin(), constants.end());
        WGPUVertexBufferLayout l{};
        l.stepMode = WGPUVertexStepMode_Vertex;
        l.arrayStride = 0;
        vi.layouts.push_back(l);
        vi.bindings.push_back({const_vertex_buffer, 0, 16});
    }
    for (size_t i = 0; i < vi.layouts.size(); i++) {
        vi.layouts[i].attributeCount = vi.attributes[i].size();
        vi.layouts[i].attributes = vi.attributes[i].data();
    }
    if (vi.layouts.size() > dev.limits.maxVertexBuffers) {
        log_limited("vbuffers", 4, "rx: %zu vertex buffers (WebGPU allows %u)", vi.layouts.size(),
                    dev.limits.maxVertexBuffers);
        return false;
    }
    return true;
}

/// The draw's vertex numbers in the order WebGPU draws them, for topologies WebGPU lacks (as lists).
bool expand_topology(uint32_t topology, std::vector<uint32_t>& v, WGPUPrimitiveTopology& out) {
    std::vector<uint32_t> o;
    const size_t n = v.size();
    switch ((Topology)topology) {
    case Topology::Quads:
        for (size_t q = 0; q + 3 < n; q += 4) o.insert(o.end(), {v[q], v[q + 1], v[q + 2], v[q], v[q + 2], v[q + 3]});
        out = WGPUPrimitiveTopology_TriangleList;
        break;
    case Topology::QuadStrip:
        for (size_t k = 0; k + 3 < n; k += 2)
            o.insert(o.end(), {v[k], v[k + 1], v[k + 3], v[k], v[k + 3], v[k + 2]});
        out = WGPUPrimitiveTopology_TriangleList;
        break;
    case Topology::TriangleFan:
    case Topology::Polygon:
        for (size_t i = 1; i + 1 < n; i++) o.insert(o.end(), {v[0], v[i], v[i + 1]});
        out = WGPUPrimitiveTopology_TriangleList;
        break;
    case Topology::LineLoop:
        for (size_t i = 0; i < n; i++) o.insert(o.end(), {v[i], v[(i + 1) % n]});
        out = WGPUPrimitiveTopology_LineList;
        break;
    default:
        return false;
    }
    v.swap(o);
    return true;
}

bool direct_topology(uint32_t topology, WGPUPrimitiveTopology& out) {
    switch ((Topology)topology) {
    case Topology::Points: out = WGPUPrimitiveTopology_PointList; return true;
    case Topology::Lines: out = WGPUPrimitiveTopology_LineList; return true;
    case Topology::LineStrip: out = WGPUPrimitiveTopology_LineStrip; return true;
    case Topology::Triangles: out = WGPUPrimitiveTopology_TriangleList; return true;
    case Topology::TriangleStrip: out = WGPUPrimitiveTopology_TriangleStrip; return true;
    default: return false;
    }
}

WGPUBindGroupLayout shared_layout(const std::vector<uint32_t>& packed, const WGPUBindGroupLayoutDescriptor& ld) {
    static const bool own = env_flag("RX_NO_SHARED_LAYOUTS");
    if (own) return wgpuDeviceCreateBindGroupLayout(dev.device, &ld);
    static std::unordered_multimap<uint64_t, std::pair<std::vector<uint32_t>, WGPUBindGroupLayout>> layouts;
    const uint64_t h = hash_bytes(packed.data(), packed.size() * sizeof(uint32_t), packed.size());
    auto [a, b] = layouts.equal_range(h);
    for (auto it = a; it != b; ++it)
        if (it->second.first == packed) return it->second.second;
    WGPUBindGroupLayout l = wgpuDeviceCreateBindGroupLayout(dev.device, &ld);
    if (l) layouts.emplace(h, std::make_pair(packed, l));
    count("bind_group_layouts");
    return l;
}

Pipeline* get_pipeline(const std::vector<uint32_t>& key, ProgramPair& prog, const VertexInput& vi,
                       const StageResources& vs_res, const StageResources* fs_res, const WGPUPrimitiveState& prim,
                       const WGPUDepthStencilState* ds, const std::vector<WGPUColorTargetState>& targets,
                       const std::array<WGPUBlendState, 8>& blends) {
    const uint64_t h = hash_bytes(key.data(), key.size() * 4);
    auto it = pipelines.find(h);
    if (it != pipelines.end()) {
        it->second->last_used = frame_number;
        if (!it->second->cache_used) {
            it->second->cache_used = true;
            native_cache_used(h);
        }
        return it->second->state == 2 ? nullptr : it->second.get();
    }
    struct PipelineTimer {
        int64_t t0 = hos_now_ns();
        ~PipelineTimer() { count("t_pipeline", hos_now_ns() - t0); count("pipelines"); }
    } pipeline_timer;
    auto p = std::make_unique<Pipeline>();
    p->last_used = frame_number;
    p->cache_used = true;
    native_cache_used(h);
    WGPUBindGroupLayoutDescriptor ld{};
    std::vector<WGPUBindGroupLayoutEntry> entries;
    for (uint32_t v : vs_res.layout) entries.push_back(unpack_layout(v));
    ld.entryCount = entries.size();
    ld.entries = entries.data();
    p->layouts[0] = shared_layout(vs_res.layout, ld);
    uint32_t nlayouts = 1;
    std::vector<WGPUBindGroupLayoutEntry> vs_entries;
    if (RX_WEB) vs_entries = entries;
    if (fs_res) {
        entries.clear();
        for (uint32_t v : fs_res->layout) entries.push_back(unpack_layout(v));
        ld.entryCount = entries.size();
        ld.entries = entries.data();
        p->layouts[1] = shared_layout(fs_res->layout, ld);
        nlayouts = 2;
    }
    WGPUPipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = nlayouts;
    pld.bindGroupLayouts = p->layouts;
    auto job = std::make_unique<PipelineJob>();
    job->target = p.get();
    job->layout = wgpuDeviceCreatePipelineLayout(dev.device, &pld);
    if (RX_WEB) {
        job->bgl[0] = std::move(vs_entries);
        if (nlayouts == 2) job->bgl[1] = entries;
        job->nbgl = (int)nlayouts;
    }
    job->vs_module = &prog.vs;
    job->fs_module = prog.has_fs ? &prog.fs : nullptr;
    job->buffers = vi.layouts;
    job->attributes.assign(vi.attributes.begin(), vi.attributes.begin() + (ptrdiff_t)vi.nattr);
    job->targets = targets;
    job->blends = blends;
    job->primitive = prim;
    job->has_depth = ds != nullptr;
    if (ds) job->depth = *ds;
    st_pipelines++;
    if (!RX_WEB && native_cache_on() && prog.vs.text_hash) {   // (the native pipeline cache: made at the next start)
        PipelineDesc pd;
        pd.key = h;
        pd.vs_text = prog.vs.text_hash;
        pd.fs_text = prog.has_fs ? prog.fs.text_hash : 0;
        pd.layout[0] = vs_res.layout;
        if (fs_res) pd.layout[1] = fs_res->layout;
        pd.nlayouts = (int)nlayouts;
        pd.buffers = job->buffers;
        pd.attributes = job->attributes;
        pd.targets = targets;
        for (const WGPUColorTargetState& t : targets) pd.has_blend.push_back(t.blend != nullptr);
        pd.blends = blends;
        pd.primitive = prim;
        pd.has_depth = ds != nullptr;
        if (ds) pd.depth = *ds;
        native_cache_pipeline(pd);
    }
    Pipeline* raw = p.get();
    pipelines[h] = std::move(p);
    static const bool sync = RX_WEB || getenv("RX_SYNC_PIPELINES") != nullptr;
    if (sync) compile(*job);
    else pipeline_workers.submit(std::move(job));
    return raw->state == 2 ? nullptr : raw;
}

std::vector<Pipeline*> prewarming;   // (cached pipelines submitted, until they are compiled)

} // namespace

int prewarm_in_flight() {
    prewarming.erase(std::remove_if(prewarming.begin(), prewarming.end(), [](Pipeline* p) { return p->state != 0; }),
                     prewarming.end());
    return (int)prewarming.size();
}

bool prewarm_pipeline(const PipelineDesc& d, ShaderModule* vs, ShaderModule* fs) {
    if (!vs || pipelines.count(d.key)) return false;
    auto p = std::make_unique<Pipeline>();
    p->last_used = frame_number;
    std::vector<WGPUBindGroupLayoutEntry> entries;
    for (int i = 0; i < d.nlayouts && i < 2; i++) {
        entries.clear();
        for (uint32_t v : d.layout[i]) entries.push_back(unpack_layout(v));
        WGPUBindGroupLayoutDescriptor ld{};
        ld.entryCount = entries.size();
        ld.entries = entries.data();
        p->layouts[i] = shared_layout(d.layout[i], ld);
    }
    WGPUPipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = (size_t)d.nlayouts;
    pld.bindGroupLayouts = p->layouts;
    auto job = std::make_unique<PipelineJob>();
    job->target = p.get();
    job->layout = wgpuDeviceCreatePipelineLayout(dev.device, &pld);
    job->vs_module = vs;
    job->fs_module = fs;
    job->buffers = d.buffers;
    job->attributes = d.attributes;
    job->targets = d.targets;
    static const WGPUBlendState marker{};   // (compile() points a target that blends at its blend state)
    for (size_t i = 0; i < job->targets.size(); i++)
        job->targets[i].blend = i < d.has_blend.size() && d.has_blend[i] ? &marker : nullptr;
    job->blends = d.blends;
    job->primitive = d.primitive;
    job->has_depth = d.has_depth;
    job->depth = d.depth;
    st_pipelines++;
    prewarming.push_back(p.get());
    pipelines[d.key] = std::move(p);
    pipeline_workers.submit(std::move(job));
    return true;
}

namespace {

/// The viewport transform of the draw in clip space (see the WGSL backend's `rescale`), and the pixels it covers.
struct ViewTransform {
    float ax, bx, ay, by, az, bz;
    float x0, y0, x1, y1;   // the viewport's rectangle in target pixels (rows from the top)
};
ViewTransform view_transform(const Regs& r, uint32_t W, uint32_t H) {
    float sx, tx, sy, ty, sz, tz;
    if (r.viewport_scale_offset_enabled) {
        const auto& v = r.viewport_transform[0];
        sx = v.scale_x;
        tx = v.translate_x;
        sy = v.scale_y;
        ty = v.translate_y;
        sz = v.scale_z;
        tz = v.translate_z;
    } else {   // positions are window coordinates (the program uses the render area): the whole surface
        const auto& c = r.surface_clip;
        sx = c.width * 0.5f;
        tx = c.x + c.width * 0.5f;
        sy = c.height * 0.5f;
        ty = c.y + c.height * 0.5f;
        sz = 1.0f;
        tz = 0.0f;
    }
    const bool lower_left = r.window_origin.mode != Regs::WindowOrigin::Mode::UpperLeft;
    // target row of a vertex: c + k * y_ndc
    const float clip_h = (float)r.surface_clip.height;
    const float c = lower_left ? clip_h - ty : ty;
    const float k = lower_left ? -sy : sy;
    ViewTransform t;
    t.ax = 2.0f * sx / W;
    t.bx = 2.0f * tx / W - 1.0f;
    t.ay = -2.0f * k / H;
    t.by = 1.0f - 2.0f * c / H;
    t.az = sz;
    t.bz = tz;
    t.x0 = tx - std::fabs(sx);
    t.x1 = tx + std::fabs(sx);
    t.y0 = c - std::fabs(k);
    t.y1 = c + std::fabs(k);
    return t;
}

int trace = -1;

} // namespace

void draw(Channel& ch, const Regs& r, const DrawParams& p) {
    struct DrawTimer {
        int64_t t0 = clock_ns();
        ~DrawTimer() { count("t_draw", clock_ns() - t0); spike.draws++; }
    } draw_timer;
    if (trace < 0) trace = getenv("RX_TRACE") ? atoi(getenv("RX_TRACE")) : 0;
    if (!const_vertex_buffer) {
        WGPUBufferDescriptor bd{};
        bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
        bd.size = 16;
        const_vertex_buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
        const float v[4] = {0, 0, 0, 1};
        wgpuQueueWriteBuffer(dev.queue, const_vertex_buffer, 0, v, 16);
    }
    auto skip = [&](const char* why) {
        st_skipped++;
        count("skipped");
        count(why);
        if (trace > 0 && st_skipped < trace) hos_log("rx: draw skipped: %s", why);
        static std::unordered_map<uint64_t, bool> logged_skips;   // RX_LOG_SKIPS=1: each program pair once
        const uint64_t skip_key = (r.program_region.Address() + r.pipelines[1].offset) * 31 +
                                  r.pipelines[5].offset + (uint64_t)(uintptr_t)why;
        static const bool log_skips = env_flag("RX_LOG_SKIPS");
        const bool log_skip = log_skips && logged_skips.emplace(skip_key, true).second;
        if (frame_dump_active() || log_skip) {
            char b[512];
            int n = std::snprintf(b, sizeof b, "skipped (%s): vs %llx fs %llx rtctl count %u", why,
                                  (unsigned long long)(r.program_region.Address() + r.pipelines[1].offset),
                                  (unsigned long long)(r.program_region.Address() + r.pipelines[5].offset),
                                  (unsigned)r.rt_control.count);
            for (uint32_t i = 0; i < std::min<uint32_t>(r.rt_control.count, 8) && n < 400; i++) {
                const auto& rt = r.rt[r.rt_control.Map(i)];
                n += std::snprintf(b + n, sizeof b - n, " [%u->%u %#llx %ux%u f%u]", i, r.rt_control.Map(i),
                                   (unsigned long long)rt.Address(), rt.width, rt.height, (unsigned)rt.format);
            }
            std::snprintf(b + n, sizeof b - n, " zeta %u %#llx f%u", (unsigned)r.zeta_enable,
                          (unsigned long long)r.zeta.Address(), (unsigned)r.zeta.format);
            if (log_skip) hos_log("rx: %s", b);
            else dump_draw(b);
        }
    };
    if (!ch.render_on) {
        // RX_WATCH_RT=<guest hex>: draws into that target skipped by the render enable condition are logged
        static const uint64_t watch = getenv("RX_WATCH_RT") ? strtoull(getenv("RX_WATCH_RT"), nullptr, 16) : 0;
        if (watch) {
            static int logged;
            for (uint32_t i = 0; i < 8; i++)
                if ((uint32_t)r.rt[i].Address() == (uint32_t)watch && r.rt[i].format != Tegra::RenderTargetFormat::NONE &&
                    logged++ < 200) {
                    uint64_t v[4]{};
                    gpu_read(r.render_enable.Address(), v, sizeof v);
                    hos_log("rx: watch %08llx: draw skipped (render enable %u override %u at %llx: %llx / %llx) frame %lld",
                            (unsigned long long)watch, (unsigned)r.render_enable.mode,
                            (unsigned)r.render_enable_override, (unsigned long long)r.render_enable.Address(),
                            (unsigned long long)v[0], (unsigned long long)v[2], (long long)frame_number);
                }
        }
        return skip("render enable condition false");
    }
    trim_bind_groups();
    ProgramPair* prog = programs_for_draw(ch, r);
    if (!prog) return skip("programs");
    Attachments at = draw_attachments(r);
    if (!at.width) {
        if (trace > 0 && st_skipped < trace)
            hos_log("rx: no-target registers rtctrl count %u target0 %u rt0 %#llx %ux%u fmt %u; zeta enable %u %#llx "
                    "%ux%u fmt %#x; programs vs %#x fs %#x", static_cast<uint32_t>(r.rt_control.count),
                    r.rt_control.Map(0), (unsigned long long)r.rt[0].Address(), r.rt[0].width, r.rt[0].height,
                    (unsigned)r.rt[0].format, (unsigned)r.zeta_enable, (unsigned long long)r.zeta.Address(),
                    r.zeta_size.width, r.zeta_size.height, (unsigned)r.zeta.format, r.pipelines[1].offset,
                    r.pipelines[5].offset);
        return skip("no render target");
    }
    // RX_DUMP_GEOM=<pixel program hex> (with a frame dump): the first such draw's indices and attribute 0 (3 floats)
    if (frame_dump_active())
        if (static const char* want = getenv("RX_DUMP_GEOM");
            want && strtoull(want, nullptr, 16) == r.program_region.Address() + r.pipelines[5].offset) {
            static int64_t dumped_frame = -1;
            if (dumped_frame != frame_number) {
                dumped_frame = frame_number;
                const auto& a = r.vertex_attrib_format[0];
                const auto& vs = r.vertex_streams[a.buffer % 16];
                std::string line = "geom";
                for (uint32_t k = 0; k < std::min<uint32_t>(p.count, 36); k++) {
                    uint32_t idx = p.first + k;
                    if (p.indexed && !p.inline_index) {
                        uint32_t v = 0;
                        gpu_read(p.index_va + (uint64_t)(p.first + k) * p.index_size, &v, p.index_size);
                        idx = v + p.base_vertex;
                    }
                    float xyz[3]{};
                    gpu_read(vs.Address() + (uint64_t)idx * vs.stride + a.offset, xyz, 12);
                    char b[96];
                    std::snprintf(b, sizeof b, " %u:(%g,%g,%g)", idx, xyz[0], xyz[1], xyz[2]);
                    line += b;
                }
                hos_log("rx-frame:   %s", line.c_str());
            }
        }

    // ---- vertices and indices ----
    WGPUPrimitiveTopology topology{};
    const bool direct = direct_topology(p.topology, topology);
    std::vector<uint32_t> indices;
    uint32_t vmin = 0, vmax = 0;
    bool indexed = p.indexed;
    const bool restart = r.primitive_restart.enabled != 0;
    const uint32_t restart_index = r.primitive_restart.index;
    std::bitset<32> vs_loads;
    for (uint32_t i = 0; i < 32; i++) vs_loads[i] = prog->vs.info.loads.Generic(i);
    const PullPlan plan = pull_plan(r, vs_loads);
    static VertexInput vi;   // (reused: see vertex_input)
    vi.clear();
    uint64_t index_off = 0, index_bytes = 0;
    WGPUIndexFormat index_format = WGPUIndexFormat_Undefined;
    // the buffer cache: vertex and index data bound where it is in guest memory (no copy, no index scan)
    bool cached = false;
    BufferRef index_ref;
    const bool restart_ok = !restart || !p.indexed ||
                            (p.index_size == 2 ? restart_index == 0xFFFF : restart_index == 0xFFFFFFFF);
    if (!direct) count("nc_topology");
    else if (!restart_ok) count("nc_restart");
    else if (p.indexed && (p.inline_index || p.index_size == 1)) count("nc_index");
    if (direct && restart_ok && buffer_cache_active() && (!p.indexed || (!p.inline_index && p.index_size != 1))) {
        // room for the draw's uniforms first: a flush must not come between the cache's bindings and the draw
        uint64_t estimate = 64 * 1024;
        for (const auto* info : {&prog->vs.info, prog->has_fs ? &prog->fs.info : nullptr}) {
            if (!info) continue;
            estimate += info->constant_buffer_descriptors.size() * 0x10100;
            estimate += info->storage_buffers_descriptors.size() * (1 << 20);
        }
        if (stream_ro.used + estimate > stream_ro.capacity || stream_rw.used + (8u << 20) > stream_rw.capacity)
            flush_batch();
        CachedDraw cd;
        cd.on = true;
        cd.indexed = p.indexed;
        cd.first = p.first;
        cd.count = p.count;
        cached = vertex_input(vi, r, prog->vs.info, 0, 0, p.base_instance, p.instance_count, cd, &plan);
        if (cached && p.indexed) {
            index_ref = guest_buffer(p.index_va + (uint64_t)p.first * p.index_size, (uint64_t)p.count * p.index_size);
            cached = index_ref.buffer && index_ref.offset % p.index_size == 0;
            index_format = p.index_size == 2 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
        }
        if (!cached) vi.clear();
        else count("draws_cached");
    }
    if (!cached) {
        if (p.indexed) {
            indices.resize(p.count);
            if (p.inline_index) {
                const size_t n = std::min<size_t>(p.count, p.inline_index->size() / 4);
                std::memcpy(indices.data(), p.inline_index->data(), n * 4);
            } else {
                std::vector<uint8_t> raw((size_t)p.count * p.index_size);
                gpu_read(p.index_va + (uint64_t)p.first * p.index_size, raw.data(), raw.size());
                for (uint32_t i = 0; i < p.count; i++) {
                    const uint8_t* s = raw.data() + (size_t)i * p.index_size;
                    indices[i] = p.index_size == 1 ? s[0] : p.index_size == 2 ? (uint32_t)(s[0] | s[1] << 8)
                                                                               : (uint32_t)(s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24);
                }
            }
            vmin = UINT32_MAX;
            for (uint32_t& i : indices) {
                if (restart && i == restart_index) {
                    i = 0xFFFFFFFF;
                    continue;
                }
                const uint32_t v = (uint32_t)((int64_t)i + p.base_vertex);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
            if (vmin == UINT32_MAX) return skip("only restart indices");
        } else {
            vmin = p.first;
            vmax = p.first + p.count - 1;
        }
        if (!direct) {
            if (!p.indexed) {
                indices.resize(p.count);
                for (uint32_t i = 0; i < p.count; i++) indices[i] = p.first + i;
                indexed = true;
            } else {
                for (uint32_t& i : indices)
                    if (i != 0xFFFFFFFF) i = (uint32_t)((int64_t)i + p.base_vertex);
            }
            if (!expand_topology(p.topology, indices, topology)) {
                log_limited("topology", 8, "rx: topology %u not drawn", p.topology);
                return skip("topology");
            }
        }
        if (vmax - vmin > (16u << 20)) return skip("vertex range");

        // upload space: estimated first, so that a flush cannot happen in the middle of the draw's uploads
        {
            uint64_t estimate = 64 * 1024 + (uint64_t)indices.size() * 4;
            for (const auto* info : {&prog->vs.info, prog->has_fs ? &prog->fs.info : nullptr}) {
                if (!info) continue;
                estimate += info->constant_buffer_descriptors.size() * (info->uses_cbuf_indirect ? 0x10100 : 0x2100);
                estimate += info->storage_buffers_descriptors.size() * (1 << 20);
            }
            for (uint32_t s = 0; s < 16; s++)
                if (r.vertex_streams[s].IsEnabled())
                    estimate += ((uint64_t)vmax - vmin + 1) * r.vertex_streams[s].stride + 256 +
                                (uint64_t)p.instance_count * r.vertex_streams[s].stride;
            if (stream_ro.used + estimate > stream_ro.capacity || stream_rw.used + (8u << 20) > stream_rw.capacity)
                flush_batch();
            if (estimate > stream_ro.capacity) return skip("too much data");
        }

        if (!vertex_input(vi, r, prog->vs.info, vmin, vmax, p.base_instance, p.instance_count, {}, &plan))
            return skip("vertex input");

        if (indexed) {
            // vertex numbers relative to the uploaded range: idx + base_vertex - vmin (expanded lists hold numbers)
            const int64_t adjust = direct ? (int64_t)p.base_vertex - vmin : -(int64_t)vmin;
            const bool small = vmax - vmin < 0xFFFF;
            index_format = small ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
            index_bytes = ((uint64_t)indices.size() * (small ? 2 : 4) + 3) & ~3ull;
            index_off = stream_ro.alloc(index_bytes, 16);
            if (index_off == UINT64_MAX) return skip("index space");
            count("up_index", (int64_t)index_bytes);
            uint8_t* dst = stream_ro.at(index_off);
            for (size_t i = 0; i < indices.size(); i++) {
                const uint32_t v = indices[i] == 0xFFFFFFFF ? 0xFFFFFFFF : (uint32_t)((int64_t)indices[i] + adjust);
                if (small) {
                    const uint16_t s = indices[i] == 0xFFFFFFFF ? 0xFFFF : (uint16_t)v;
                    std::memcpy(dst + i * 2, &s, 2);
                } else {
                    std::memcpy(dst + i * 4, &v, 4);
                }
            }
            if (indices.size() & 1 && small) std::memset(dst + indices.size() * 2, 0, 2);
        }
    }

    // ---- per-draw uniforms (the start of each stage's uniform block: see the WGSL backend) ----
    const ViewTransform vt = view_transform(r, at.width, at.height);
    float rescale[16] = {0, 0, 1.0f, 0, (float)r.surface_clip.width, (float)r.surface_clip.height, 0, 0,
                         vt.ax, vt.bx, vt.ay, vt.by, vt.az, vt.bz, 0, 0};
    {
        const int32_t vertex_offset = cached ? 0 : (int32_t)vmin;
        const int32_t instance_offset = cached ? -(int32_t)p.base_instance : 0;
        std::memcpy(&rescale[14], &vertex_offset, 4);
        std::memcpy(&rescale[15], &instance_offset, 4);
    }
    // a 16-bit depth target: depths on its steps (d16q_ in the WGSL backend; RX_NO_D16Q=1: off)
    static const bool no_d16q = env_flag("RX_NO_D16Q");
    if (at.depth && at.depth->format == WGPUTextureFormat_Depth16Unorm && !no_d16q) rescale[6] = 1.0f;
    // colour targets whose 16-bit UNORM / SNORM format is kept as float: the pixel shader clamps what it writes
    // there, as the hardware's store would (RX_NO_NORM_CLAMP=1: off)
    static const bool no_norm_clamp = env_flag("RX_NO_NORM_CLAMP");
    if (!no_norm_clamp) {
        uint32_t mask = 0;
        for (int i = 0; i < 8; i++) {
            const Image* c = at.color[i];
            if (!c || (c->format != WGPUTextureFormat_R16Float && c->format != WGPUTextureFormat_RG16Float &&
                       c->format != WGPUTextureFormat_RGBA16Float && c->format != WGPUTextureFormat_RGBA32Float))
                continue;
            switch (c->pixel_format) {
            case PixelFormat::R16_UNORM: case PixelFormat::R16G16_UNORM: case PixelFormat::R16G16B16A16_UNORM:
                mask |= 1u << i;
                break;
            case PixelFormat::R16_SNORM: case PixelFormat::R16G16_SNORM: case PixelFormat::R16G16B16A16_SNORM:
                mask |= 0x100u << i;
                break;
            default: break;
            }
        }
        rescale[7] = (float)mask;
    }

    if (static const bool trace_draw_state = env_flag("RX_TRACE_DRAW_STATE"); trace_draw_state) {
        static std::unordered_map<uint64_t, bool> seen;
        const uint64_t state_key = (uint64_t)(uintptr_t)prog ^ ((uint64_t)at.width << 32) ^ at.height;
        if (seen.size() < 160 && seen.emplace(state_key, true).second) {
            hos_log("rx: state vs %#llx fs %#llx target %ux%u first %u count %u vertices %u..%u depth %u/%u/%#x clip %u,%u %ux%u viewport %u origin %u transform %g,%g %g,%g %g,%g scissor %u %u..%u %u..%u mask %#x",
                    (unsigned long long)(r.program_region.Address() + r.pipelines[1].offset),
                    (unsigned long long)(r.program_region.Address() + r.pipelines[5].offset), at.width, at.height,
                    p.first, p.count, vmin, vmax, r.depth_test_enable, r.depth_write_enabled, (unsigned)r.depth_test_func,
                    (unsigned)r.surface_clip.x, (unsigned)r.surface_clip.y, (unsigned)r.surface_clip.width,
                    (unsigned)r.surface_clip.height, r.viewport_scale_offset_enabled, static_cast<unsigned>(r.window_origin.mode.Value()),
                    vt.ax, vt.bx, vt.ay, vt.by, vt.az, vt.bz, r.scissor_test[0].enable,
                    (unsigned)r.scissor_test[0].min_x, (unsigned)r.scissor_test[0].max_x,
                    (unsigned)r.scissor_test[0].min_y, (unsigned)r.scissor_test[0].max_y, r.color_mask[0].raw);
            for (uint32_t i = 0; i < 32; i++) {
                if (!prog->vs.info.loads.Generic(i)) continue;
                const auto& a = r.vertex_attrib_format[i];
                const auto& stream = r.vertex_streams[a.buffer % 16];
                uint8_t raw[16]{};
                gpu_read(stream.Address() + (uint64_t)vmin * (uint32_t)stream.stride + (uint32_t)a.offset, raw, 16);
                uint32_t values[4];
                decode_attribute(raw, a, values);
                hos_log("rx: attribute %u config %#x stream %#llx stride %u offset %u enabled %u value %g,%g,%g,%g",
                        i, a.hex, (unsigned long long)stream.Address(), (unsigned)stream.stride, (unsigned)a.offset,
                        (unsigned)stream.IsEnabled(), std::bit_cast<float>(values[0]), std::bit_cast<float>(values[1]),
                        std::bit_cast<float>(values[2]), std::bit_cast<float>(values[3]));
            }
        }
    }

    const int64_t t_res = clock_ns();
    std::string vs_textures;
    // (reused across draws: their allocations were ~10% of the GPU thread)
    static StageResources vs_res, fs_res;
    for (StageResources* sr : {&vs_res, &fs_res}) {
        sr->layout.clear();
        sr->entries.clear();
        sr->dynamic_offsets.clear();
        sr->next_binding = 0;
        sr->ok = true;
    }
    uint32_t dynamic_storage = dev.limits.maxDynamicStorageBuffersPerPipelineLayout;
    dump_fs_program = r.program_region.Address() + r.pipelines[5].offset;
    stage_resources(vs_res, ch, r, prog->vs.info, 0, WGPUShaderStage_Vertex, rescale, at, dynamic_storage);
    if (!vs_res.ok) return skip("vertex resources");
    if (!pulled_streams(vs_res, r, plan, p, cached)) return skip("pulled vertices");
    if (frame_dump_active()) vs_textures = dump_textures;
    if (prog->has_fs) {
        stage_resources(fs_res, ch, r, prog->fs.info, 4, WGPUShaderStage_Fragment, rescale, at, dynamic_storage);
        if (!fs_res.ok) return skip("pixel resources");
    } else {
        for (Image* c : at.color)
            if (c) return skip("color targets without a pixel program");
    }

    count("t_resources", clock_ns() - t_res);
    const int64_t t_state = clock_ns();
    // ---- fixed-function state ----
    static const bool normalize = !env_flag("RX_NO_PIPELINE_NORMALIZE");
    WGPUPrimitiveState prim{};
    prim.topology = topology;
    const bool strip = topology == WGPUPrimitiveTopology_TriangleStrip || topology == WGPUPrimitiveTopology_LineStrip;
    prim.stripIndexFormat = strip && indexed ? index_format : WGPUIndexFormat_Undefined;
    // Winding: the viewport transform keeps the guest's window rows (flipped for a lower-left origin, which flips
    // the winding too); with TOTK's upper-left origin the guest's CCW is WebGPU's CCW (checked: culling as the game
    // asks removes nothing visible)
    {
        const bool lower_left = r.window_origin.mode != Regs::WindowOrigin::Mode::UpperLeft;
        const bool ccw = r.gl_front_face != Regs::FrontFace::ClockWise;
        prim.frontFace = ccw != lower_left ? WGPUFrontFace_CCW : WGPUFrontFace_CW;
        static const uint64_t flip_face = getenv("RX_DEBUG_FLIP_FACE") ? strtoull(getenv("RX_DEBUG_FLIP_FACE"), nullptr, 16) : 0;
        if (flip_face && flip_face == r.program_region.Address() + r.pipelines[5].offset)   // (debugging)
            prim.frontFace = prim.frontFace == WGPUFrontFace_CCW ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
        // depth clamp instead of clipping at the near and far planes unless the clip mode clips z (as suyu): TOTK
        // draws its cave volumes as boxes reaching past the far plane, back faces only
        {
            using GC = Regs::ViewportClipControl::GeometryClip;
            const GC clip = r.viewport_clip_control.geometry_clip;
            prim.unclippedDepth = dev.depth_clip_control && clip != GC::FrustumXYZ && clip != GC::FrustumZ;
            static const uint64_t unclipped = getenv("RX_DEBUG_UNCLIPPED") ? strtoull(getenv("RX_DEBUG_UNCLIPPED"), nullptr, 16) : 0;
            if (unclipped && unclipped == r.program_region.Address() + r.pipelines[5].offset)   // (debugging)
                prim.unclippedDepth = dev.depth_clip_control;
        }
        prim.cullMode = WGPUCullMode_None;
        static const bool no_cull = env_flag("RX_NO_CULL");
        if (r.gl_cull_test_enabled && !no_cull) {
            if (r.gl_cull_face == Regs::CullFace::FrontAndBack) return skip("culled");
            prim.cullMode = r.gl_cull_face == Regs::CullFace::Front ? WGPUCullMode_Front : WGPUCullMode_Back;
        }
    }
    WGPUDepthStencilState ds{};
    if (at.depth) {
        ds.format = at.depth->format;
        const bool test = r.depth_test_enable != 0;
        ds.depthWriteEnabled = test && r.depth_write_enabled ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        ds.depthCompare = test ? compare_op(r.depth_test_func) : WGPUCompareFunction_Always;
        // RX_DEBUG_DEPTH_ALWAYS=<pixel program hex> (debugging): that program's draws pass the depth test
        static const uint64_t depth_always = getenv("RX_DEBUG_DEPTH_ALWAYS") ? strtoull(getenv("RX_DEBUG_DEPTH_ALWAYS"), nullptr, 16) : 0;
        if (depth_always && depth_always == r.program_region.Address() + r.pipelines[5].offset)
            ds.depthCompare = WGPUCompareFunction_Always;
        if (r.stencil_enable && has_stencil(ds.format)) {
            const auto& f = r.stencil_front_op;
            ds.stencilFront = {compare_op(f.func), stencil_op(f.fail), stencil_op(f.zfail), stencil_op(f.zpass)};
            const auto& b = r.stencil_two_side_enable ? r.stencil_back_op : r.stencil_front_op;
            ds.stencilBack = {compare_op(b.func), stencil_op(b.fail), stencil_op(b.zfail), stencil_op(b.zpass)};
            ds.stencilReadMask = r.stencil_front_func_mask & 0xFF;
            ds.stencilWriteMask = r.stencil_front_mask & 0xFF;
        } else {
            ds.stencilFront = ds.stencilBack = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep,
                                                WGPUStencilOperation_Keep, WGPUStencilOperation_Keep};
            ds.stencilReadMask = ds.stencilWriteMask = 0;
        }
        if (r.polygon_offset_fill_enable && (test || !normalize) && topology != WGPUPrimitiveTopology_PointList &&
            topology != WGPUPrimitiveTopology_LineList && topology != WGPUPrimitiveTopology_LineStrip) {
            ds.depthBias = (int32_t)std::lround(r.depth_bias);
            ds.depthBiasSlopeScale = r.slope_scale_depth_bias;
            ds.depthBiasClamp = r.depth_bias_clamp;
        }
    }
    if (normalize && at.depth) {
        for (WGPUStencilFaceState* f : {&ds.stencilFront, &ds.stencilBack})
            if (!ds.stencilWriteMask)
                f->failOp = f->depthFailOp = f->passOp = WGPUStencilOperation_Keep;
        auto inert = [](const WGPUStencilFaceState& f) {
            return f.compare == WGPUCompareFunction_Always && f.failOp == WGPUStencilOperation_Keep &&
                   f.depthFailOp == WGPUStencilOperation_Keep && f.passOp == WGPUStencilOperation_Keep;
        };
        if (inert(ds.stencilFront) && inert(ds.stencilBack)) ds.stencilReadMask = ds.stencilWriteMask = 0;
    }
    if (normalize && prim.cullMode == WGPUCullMode_None &&
        (!prog->has_fs || (!prog->fs.info.loads.mask[(size_t)Shader::IR::Attribute::FrontFace] &&
                           !prog->fs.info.loads_indexed_attributes)) &&
        (!at.depth || !std::memcmp(&ds.stencilFront, &ds.stencilBack, sizeof ds.stencilFront)))
        prim.frontFace = WGPUFrontFace_CCW;
    static std::vector<WGPUColorTargetState> targets;
    targets.clear();
    std::array<WGPUBlendState, 8> blends{};
    uint32_t ntargets = 0;
    for (uint32_t i = 0; i < 8; i++)
        if (at.color[i]) ntargets = i + 1;
    static std::vector<uint32_t> key;
    key.clear();
    // (by WGSL text: many program pairs translate to the same modules)
    static const bool by_pair = env_flag("RX_NO_MODULE_SHARING");
    if (by_pair) {
        key.push_back((uint32_t)(uintptr_t)prog);
        key.push_back((uint32_t)((uintptr_t)prog >> 32));
    } else {
        // (by the texts' hashes: the same key in every session, for the native pipeline cache)
        const uint64_t vh = prog->vs.text_hash, fh = prog->has_fs ? prog->fs.text_hash : 0;
        key.insert(key.end(), {(uint32_t)vh, (uint32_t)(vh >> 32), (uint32_t)fh, (uint32_t)(fh >> 32)});
    }
    for (uint32_t i = 0; i < ntargets; i++) {
        WGPUColorTargetState t{};
        Image* im = at.color[i];
        t.format = im ? im->format : WGPUTextureFormat_Undefined;
        if (im) {
            const uint32_t target = r.rt_control.Map(i);
            const auto& m = r.color_mask[r.color_mask_common ? 0 : target];
            t.writeMask = (m.R ? WGPUColorWriteMask_Red : 0) | (m.G ? WGPUColorWriteMask_Green : 0) |
                          (m.B ? WGPUColorWriteMask_Blue : 0) | (m.A ? WGPUColorWriteMask_Alpha : 0);
            if (!prog->fs.info.stores_frag_color[i]) t.writeMask = WGPUColorWriteMask_None;
            const bool integer = VideoCore::Surface::IsPixelFormatInteger(im->pixel_format);
            const bool blend_on = r.blend.enable[target] != 0 && !integer &&
                                  (t.writeMask != WGPUColorWriteMask_None || !normalize);
            if (blend_on) {
                WGPUBlendState& b = blends[i];
                // the alpha registers always (as yuzu / Eden): TOTK's per-target blends leave separate_alpha clear
                // with their own alpha factors set (the rain fog's transmittance multiplies down; using the colour
                // factors for alpha kept it near 1: sky islands and clouds showed through the fog, the native NVN
                // layer and Eden both fog them out). RX_SEPARATE_ALPHA_FLAG=1: the old behaviour.
                static const bool use_flag = env_flag("RX_SEPARATE_ALPHA_FLAG");
                if (r.blend_per_target_enabled) {
                    const auto& bt = r.blend_per_target[target];
                    b.color = {blend_op(bt.color_op), blend_factor(bt.color_source), blend_factor(bt.color_dest)};
                    b.alpha = !use_flag || bt.separate_alpha
                                  ? WGPUBlendComponent{blend_op(bt.alpha_op), blend_factor(bt.alpha_source),
                                                       blend_factor(bt.alpha_dest)}
                                  : b.color;
                } else {
                    const auto& bc = r.blend;
                    b.color = {blend_op(bc.color_op), blend_factor(bc.color_source), blend_factor(bc.color_dest)};
                    b.alpha = !use_flag || bc.separate_alpha
                                  ? WGPUBlendComponent{blend_op(bc.alpha_op), blend_factor(bc.alpha_source),
                                                       blend_factor(bc.alpha_dest)}
                                  : b.color;
                }
                // WebGPU: min/max take factors of one
                for (WGPUBlendComponent* c : {&b.color, &b.alpha})
                    if (c->operation == WGPUBlendOperation_Min || c->operation == WGPUBlendOperation_Max)
                        c->srcFactor = c->dstFactor = WGPUBlendFactor_One;
                t.blend = &blends[i];
                key.insert(key.end(), {(uint32_t)b.color.operation, (uint32_t)b.color.srcFactor,
                                       (uint32_t)b.color.dstFactor, (uint32_t)b.alpha.operation,
                                       (uint32_t)b.alpha.srcFactor, (uint32_t)b.alpha.dstFactor});
            }
        }
        targets.push_back(t);
        key.insert(key.end(), {(uint32_t)t.format, (uint32_t)t.writeMask, t.blend ? 1u : 0u});
    }
    key.insert(key.end(), vi.key.begin(), vi.key.end());
    key.insert(key.end(), {0xDDDD, (uint32_t)prim.topology, (uint32_t)prim.stripIndexFormat, (uint32_t)prim.frontFace,
                           (uint32_t)prim.cullMode, (uint32_t)prim.unclippedDepth});
    const std::vector<WGPUColorTargetState>& targets_state_for_dump = targets;
    if (at.depth)
        key.insert(key.end(),
                   {(uint32_t)ds.format, (uint32_t)ds.depthWriteEnabled, (uint32_t)ds.depthCompare,
                    (uint32_t)ds.stencilFront.compare, (uint32_t)ds.stencilFront.failOp,
                    (uint32_t)ds.stencilFront.depthFailOp, (uint32_t)ds.stencilFront.passOp,
                    (uint32_t)ds.stencilBack.compare, (uint32_t)ds.stencilBack.failOp,
                    (uint32_t)ds.stencilBack.depthFailOp, (uint32_t)ds.stencilBack.passOp, ds.stencilReadMask,
                    ds.stencilWriteMask, (uint32_t)ds.depthBias, std::bit_cast<uint32_t>(ds.depthBiasSlopeScale),
                    std::bit_cast<uint32_t>(ds.depthBiasClamp)});
    key.push_back(0xEEEE);
    for (const StageResources* res : {&vs_res, &fs_res}) {
        const uint64_t h = hash_bytes(res->layout.data(), res->layout.size() * sizeof(uint32_t));
        key.insert(key.end(), {(uint32_t)h, (uint32_t)(h >> 32), (uint32_t)res->layout.size()});
    }

    Pipeline* pipe = get_pipeline(key, *prog, vi, vs_res, prog->has_fs ? &fs_res : nullptr, prim,
                                  at.depth ? &ds : nullptr, targets, blends);
    if (!pipe) return skip("pipeline");
    if (pipe->state.load() == 0 && draw_must_run) {   // (a one-off draw waits for its pipeline, up to 5 s)
        const int64_t w0 = hos_now_ns();
        if (!RX_WEB)
            if (std::unique_ptr<PipelineJob> job = pipeline_workers.steal(pipe)) {
                compile(*job);
                count("pipeline_stolen");
            }
        const int64_t t0 = hos_now_ns();
        // (web: pipelines made asynchronously finish in callbacks this thread's event loop runs)
        while (pipe->state.load() == 0 && hos_now_ns() - t0 < 5000000000LL) {
            if (RX_WEB) wgpuDevicePoll(dev.device, true, nullptr);
            else std::this_thread::yield();
        }
        count("pipeline_waited");
        spike.pipeline_ns += hos_now_ns() - w0;
    }
    if (pipe->state.load() != 1) return skip("pipeline compiling");

    // ---- scissor: the viewport's rectangle, the scissor, the target ----
    float sx0 = std::max(0.0f, vt.x0), sy0 = std::max(0.0f, vt.y0);
    float sx1 = std::min((float)at.width, vt.x1), sy1 = std::min((float)at.height, vt.y1);
    const auto& sc = r.scissor_test[0];
    if (sc.enable) {
        const bool lower_left = r.window_origin.mode != Regs::WindowOrigin::Mode::UpperLeft;
        float y0 = (float)sc.min_y, y1 = (float)sc.max_y;
        if (lower_left) {
            const float h = (float)r.surface_clip.height;
            y0 = h - (float)sc.max_y;
            y1 = h - (float)sc.min_y;
        }
        sx0 = std::max(sx0, (float)sc.min_x);
        sx1 = std::min(sx1, (float)sc.max_x);
        sy0 = std::max(sy0, y0);
        sy1 = std::min(sy1, y1);
    }
    const uint32_t ix0 = (uint32_t)std::floor(sx0), iy0 = (uint32_t)std::floor(sy0);
    const uint32_t ix1 = (uint32_t)std::min<double>(std::ceil(sx1), at.width);
    const uint32_t iy1 = (uint32_t)std::min<double>(std::ceil(sy1), at.height);
    if (sx1 <= sx0 || sy1 <= sy0 || ix1 <= ix0 || iy1 <= iy0) return skip("empty scissor");

    count("t_state", clock_ns() - t_state);
    const int64_t t_bind = clock_ns();
    // ---- bind groups ----
    WGPUBindGroup g0 = bind_group(pipe->layouts[0], vs_res.entries);
    WGPUBindGroup g1 = prog->has_fs ? bind_group(pipe->layouts[1], fs_res.entries) : nullptr;

    count("t_bindgroups", clock_ns() - t_bind);
    const int64_t t_record = clock_ns();
    // ---- record ----
    // RX_SKIP_FS=<pixel program hex> (debugging): its draws are left out (what do they add to the picture?)
    if (static const uint64_t skip_fs = getenv("RX_SKIP_FS") ? strtoull(getenv("RX_SKIP_FS"), nullptr, 16) : 0;
        skip_fs && skip_fs == r.program_region.Address() + r.pipelines[5].offset)
        return;
    if (frame_dump_active()) dump_before_draw(r.program_region.Address() + r.pipelines[5].offset, at);
    // (RX_WRITEBACK_SMALL=<pixels> (experiment): every target up to that size is written back after each submission)
    static const uint64_t wb_small = getenv("RX_WRITEBACK_SMALL") ? strtoull(getenv("RX_WRITEBACK_SMALL"), nullptr, 10) : 0;
    for (Image* c : at.color)
        if (c && (draw_must_run || (uint64_t)c->width * c->height <= wb_small)) queue_target_writeback(c);
    WGPURenderPassEncoder pass = render_pass(at);
    // state the pass already has is not set again (every command costs at the encoder's finish)
    static struct PassState {
        uint64_t pass = ~0ull, trims = 0;
        WGPURenderPipeline pipeline;
        WGPUBindGroup groups[2];
        std::vector<uint32_t> offsets[2];
        std::array<VertexInput::Binding, 16> vertex;
        uint32_t vertex_count;
        uint32_t scissor[4];
        WGPUColor blend;
        uint32_t stencil;
    } ps;
    if (ps.pass != pass_count() || ps.trims != bind_group_trims) {   // a new pass (or groups released): nothing is set
        ps = PassState{};
        ps.pass = pass_count();
        ps.trims = bind_group_trims;
        rp_set_viewport(pass, 0, 0, (float)at.width, (float)at.height, 0.0f, 1.0f);
        ps.stencil = ~0u;
        ps.blend = {-1, -1, -1, -1};
    }
    if (ps.pipeline != pipe->pipeline) {
        rp_set_pipeline(pass, pipe->pipeline);
        ps.pipeline = pipe->pipeline;
    }
    if (ps.groups[0] != g0 || ps.offsets[0] != vs_res.dynamic_offsets) {
        rp_set_bind_group(pass, 0, g0, vs_res.dynamic_offsets.size(), vs_res.dynamic_offsets.data());
        ps.groups[0] = g0;
        ps.offsets[0] = vs_res.dynamic_offsets;
    }
    if (g1 && (ps.groups[1] != g1 || ps.offsets[1] != fs_res.dynamic_offsets)) {
        rp_set_bind_group(pass, 1, g1, fs_res.dynamic_offsets.size(), fs_res.dynamic_offsets.data());
        ps.groups[1] = g1;
        ps.offsets[1] = fs_res.dynamic_offsets;
    }
    for (uint32_t i = 0; i < vi.bindings.size() && i < 16; i++) {
        const auto& v = vi.bindings[i];
        if (i < ps.vertex_count && ps.vertex[i].buffer == v.buffer && ps.vertex[i].offset == v.offset &&
            ps.vertex[i].size == v.size)
            continue;
        rp_set_vertex_buffer(pass, i, v.buffer, v.offset, v.size);
        ps.vertex[i] = v;
    }
    ps.vertex_count = std::max<uint32_t>(ps.vertex_count, (uint32_t)vi.bindings.size());
    const uint32_t rect[4] = {ix0, iy0, ix1 - ix0, iy1 - iy0};
    if (std::memcmp(rect, ps.scissor, sizeof rect)) {
        rp_set_scissor(pass, rect[0], rect[1], rect[2], rect[3]);
        std::memcpy(ps.scissor, rect, sizeof rect);
    }
    const WGPUColor bc{r.blend_color.r, r.blend_color.g, r.blend_color.b, r.blend_color.a};
    if (std::memcmp(&bc, &ps.blend, sizeof bc)) {
        rp_set_blend_constant(pass, &bc);
        ps.blend = bc;
    }
    if (ps.stencil != (r.stencil_front_ref & 0xFF)) {
        rp_set_stencil_reference(pass, r.stencil_front_ref & 0xFF);
        ps.stencil = r.stencil_front_ref & 0xFF;
    }
    if (cached && indexed) {
        rp_set_index_buffer(pass, index_ref.buffer, index_format, index_ref.offset, index_ref.size);
        rp_draw_indexed(pass, p.count, p.instance_count, 0, p.base_vertex, p.base_instance);
    } else if (cached) {
        rp_draw(pass, p.count, p.instance_count, p.first, p.base_instance);
    } else if (indexed) {
        rp_set_index_buffer(pass, stream_ro.buffer, index_format, index_off, index_bytes);
        rp_draw_indexed(pass, (uint32_t)indices.size(), p.instance_count, 0, 0, 0);
    } else {
        rp_draw(pass, p.count, p.instance_count, 0, 0);
    }
    count("t_record", clock_ns() - t_record);
    if (frame_dump_active()) {
        std::string targets;
        for (uint32_t i = 0; i < ntargets; i++) {
            if (!at.color[i]) continue;
            char tb[160];
            const WGPUColorTargetState& t = targets_state_for_dump[i];
            if (t.blend)
                std::snprintf(tb, sizeof tb, " [%u->rt%u m%x b%x/%x/%x a%x/%x/%x]", i, r.rt_control.Map(i), (unsigned)t.writeMask,
                              (unsigned)t.blend->color.operation, (unsigned)t.blend->color.srcFactor,
                              (unsigned)t.blend->color.dstFactor, (unsigned)t.blend->alpha.operation,
                              (unsigned)t.blend->alpha.srcFactor, (unsigned)t.blend->alpha.dstFactor);
            else
                std::snprintf(tb, sizeof tb, " [%u->rt%u m%x]", i, r.rt_control.Map(i), (unsigned)t.writeMask);
            targets += tb;
        }
        char rs[160];
        std::snprintf(rs, sizeof rs, " | origin mode %u flip_y %u pixel_center %u front %#x cull %u/%#x vt %g,%g zclip %u",
                      (unsigned)r.window_origin.mode.Value(), (unsigned)r.window_origin.flip_y.Value(),
                      (unsigned)r.viewport_pixel_center, (unsigned)r.gl_front_face, r.gl_cull_test_enabled,
                      (unsigned)r.gl_cull_face, vt.ax, vt.ay, (unsigned)r.viewport_clip_control.geometry_clip.Value());
        dump_draw(("targets" + targets + rs).c_str());
        char b[1024];
        const uint32_t t0 = r.rt_control.Map(0);
        std::snprintf(b, sizeof b,
                      "draw vs %llx fs %llx %s%u x%u top %u | blend0 %u (%x,%x,%x) mask %x | depth %u/%u f%x stencil %u "
                      "| vs tex:%s | fs tex:%s",
                      (unsigned long long)(r.program_region.Address() + r.pipelines[1].offset),
                      (unsigned long long)(r.program_region.Address() + r.pipelines[5].offset), indexed ? "idx " : "",
                      p.count, p.instance_count, p.topology, (unsigned)r.blend.enable[t0], (unsigned)r.blend.color_op,
                      (unsigned)r.blend.color_source, (unsigned)r.blend.color_dest, r.color_mask[0].raw,
                      r.depth_test_enable, r.depth_write_enabled, (unsigned)r.depth_test_func, r.stencil_enable,
                      vs_textures.c_str(), dump_textures.c_str());
        dump_draw(b);
        {   // every target's blend state (blend0 above is target 0's)
            char bb[512];
            int n = std::snprintf(bb, sizeof bb, "  blend per target %d:", (int)r.blend_per_target_enabled);
            for (int t = 0; t < 8 && n < 480; t++) {
                if (!r.blend.enable[t]) continue;
                const auto& bt = r.blend_per_target[t];
                n += std::snprintf(bb + n, sizeof bb - n, " rt%d on (%x,%x,%x/%x,%x,%x)", t,
                                   r.blend_per_target_enabled ? (unsigned)bt.color_op : (unsigned)r.blend.color_op,
                                   r.blend_per_target_enabled ? (unsigned)bt.color_source : (unsigned)r.blend.color_source,
                                   r.blend_per_target_enabled ? (unsigned)bt.color_dest : (unsigned)r.blend.color_dest,
                                   r.blend_per_target_enabled ? (unsigned)bt.alpha_op : (unsigned)r.blend.alpha_op,
                                   r.blend_per_target_enabled ? (unsigned)bt.alpha_source : (unsigned)r.blend.alpha_source,
                                   r.blend_per_target_enabled ? (unsigned)bt.alpha_dest : (unsigned)r.blend.alpha_dest);
            }
            dump_draw(bb);
        }
        static const int after_rt = getenv("RX_DUMP_AFTER_RT") ? atoi(getenv("RX_DUMP_AFTER_RT")) & 7 : 0;   // (which target)
        dump_after_draw(r.program_region.Address() + r.pipelines[5].offset,
                        env_flag("RX_DUMP_AFTER_DEPTH") ? at.depth : at.color[after_rt]);
        char what[96];
        std::snprintf(what, sizeof what, "draw vs %llx fs %llx %ux%u",
                      (unsigned long long)(r.program_region.Address() + r.pipelines[1].offset),
                      (unsigned long long)(r.program_region.Address() + r.pipelines[5].offset), at.width, at.height);
        dump_pixel_after(at, what);
    }
    dump_draw(nullptr);
    st_draws++;
    count("draws");
    if (trace > 0 && st_draws < trace) {
        hos_log("rx: draw %lld: %s %u x%u topology %u target %ux%u vs %zu cbufs %zu tex, fs %zu cbufs %zu tex",
                (long long)st_draws, indexed ? "indexed" : "arrays", p.count, p.instance_count, p.topology,
                at.width, at.height, prog->vs.info.constant_buffer_descriptors.size(),
                prog->vs.info.texture_descriptors.size(), prog->fs.info.constant_buffer_descriptors.size(),
                prog->fs.info.texture_descriptors.size());
        uint32_t attachments = 0, outputs = 0;
        for (uint32_t i = 0; i < 8; i++) {
            if (at.color[i]) attachments |= 1u << i;
            if (prog->fs.info.stores_frag_color[i]) outputs |= 1u << i;
        }
        hos_log("rx: draw %lld attachments %#x fragment outputs %#x", (long long)st_draws, attachments, outputs);
    }
}

size_t pipeline_count() {
    return pipelines.size();
}

void trim_pipelines() {
    static const int64_t ttl = [] {
        const char* e = getenv("RX_PIPELINE_TTL_S");
        return (int64_t)(e ? atoi(e) : 600) * 60;
    }();
    static const size_t cap = [] {
        const char* e = getenv("RX_PIPELINE_CAP");
        return (size_t)(e && atoi(e) > 0 ? atoi(e) : 6000);
    }();
    if (ttl <= 0 || pipelines.size() <= cap) return;
    std::vector<std::pair<int64_t, uint64_t>> old;
    for (const auto& [key, p] : pipelines)
        if (p->state.load() == 1 && p->last_used < frame_number - ttl) old.emplace_back(p->last_used, key);
    std::sort(old.begin(), old.end());
    size_t dropped = 0;
    for (const auto& [used, key] : old) {
        if (pipelines.size() <= cap - cap / 8) break;
        auto it = pipelines.find(key);
        if (it == pipelines.end()) continue;
        release_pipeline_later(it->second->pipeline);
        pipelines.erase(it);
        dropped++;
    }
    if (dropped) hos_log("rx: %zu pipelines unused for %lld s released (%zu kept)", dropped, (long long)(ttl / 60), pipelines.size());
}

// ---- the native NVN layer: draws, from NVN state (rx_nvn.cpp's replay) ----
// Vertex attributes are described in the encoding the vertex fetch and the WGSL backend take (nvn_attribute), so the
// formats WebGPU cannot fetch are pulled or converted as for any draw. Each stage's bindings follow the NVN driver's
// layout: constant banks c0 / c2 (nvn_driver_bank), c1 (the shader package's constants), uniform buffer slot n as c3+n.

Regs::VertexAttribute nvn_attribute(uint32_t raw) {
    using S = Regs::VertexAttribute::Size;
    using T = Regs::VertexAttribute::Type;
    nvn_va_st v;
    std::memcpy(&v, &raw, 4);
    Regs::VertexAttribute a{};
    a.hex = 0;
    const int f = v.fmt;
    S size = S::Invalid;
    T type = T::Float;
    bool bgra = false;
    static const struct {
        int lo, hi;
        S size;
        T types[5];
    } ranges[] = {
        {0x01, 0x04, S::Size_R8, {T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x05, 0x09, S::Size_R16, {T::Float, T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x0A, 0x0C, S::Size_R32, {T::Float, T::UInt, T::SInt}},
        {0x0D, 0x10, S::Size_R8_G8, {T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x11, 0x15, S::Size_R16_G16, {T::Float, T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x16, 0x18, S::Size_R32_G32, {T::Float, T::UInt, T::SInt}},
        {0x19, 0x1C, S::Size_R8_G8_B8, {T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x1D, 0x21, S::Size_R16_G16_B16, {T::Float, T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x22, 0x24, S::Size_R32_G32_B32, {T::Float, T::UInt, T::SInt}},
        {0x25, 0x28, S::Size_R8_G8_B8_A8, {T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x29, 0x2D, S::Size_R16_G16_B16_A16, {T::Float, T::UNorm, T::SNorm, T::UInt, T::SInt}},
        {0x2E, 0x30, S::Size_R32_G32_B32_A32, {T::Float, T::UInt, T::SInt}},
    };
    // (*_UI2F / *_I2F: integers converted to floats, in the order R8 R16 R32 RG8 RG16 RG32 RGB8 ... RGBA32)
    static const S scaled_sizes[12] = {S::Size_R8, S::Size_R16, S::Size_R32, S::Size_R8_G8, S::Size_R16_G16,
                                       S::Size_R32_G32, S::Size_R8_G8_B8, S::Size_R16_G16_B16, S::Size_R32_G32_B32,
                                       S::Size_R8_G8_B8_A8, S::Size_R16_G16_B16_A16, S::Size_R32_G32_B32_A32};
    for (const auto& r : ranges)
        if (f >= r.lo && f <= r.hi) size = r.size, type = r.types[f - r.lo];
    if (f >= 0x51 && f <= 0x68) size = scaled_sizes[(f - 0x51) / 2], type = (f - 0x51) & 1 ? T::SScaled : T::UScaled;
    switch (f) {
    case NVN_FORMAT_RGB10A2: size = S::Size_A2_B10_G10_R10, type = T::UNorm; break;
    case NVN_FORMAT_RGB10A2UI: size = S::Size_A2_B10_G10_R10, type = T::UInt; break;
    case NVN_FORMAT_RGB10A2SN: size = S::Size_A2_B10_G10_R10, type = T::SNorm; break;
    case NVN_FORMAT_RGB10A2I: size = S::Size_A2_B10_G10_R10, type = T::SInt; break;
    case NVN_FORMAT_RGB10A2_UI2F: size = S::Size_A2_B10_G10_R10, type = T::UScaled; break;
    case NVN_FORMAT_RGB10A2_I2F: size = S::Size_A2_B10_G10_R10, type = T::SScaled; break;
    case NVN_FORMAT_R11G11B10F: size = S::Size_B10_G11_R11, type = T::Float; break;
    case NVN_FORMAT_BGRA8: case NVN_FORMAT_BGRX8: size = S::Size_R8_G8_B8_A8, type = T::UNorm, bgra = true; break;
    case NVN_FORMAT_RGBX8: size = S::Size_R8_G8_B8_A8, type = T::UNorm; break;
    default: break;
    }
    if (size == S::Invalid) {
        if (f) log_limited("nvn_vformat", 8, "rx: NVN vertex attribute format %#x: read as constant", f);
        a.constant.Assign(1);
        return a;
    }
    a.buffer.Assign(v.stream);
    a.offset.Assign(v.offset & 0x3FFF);
    a.size.Assign(size);
    a.type.Assign(type);
    a.bgra.Assign(bgra ? 1 : 0);
    return a;
}

namespace {

struct NvnStream {
    uint64_t addr = 0, size = 0;
    uint32_t stride = 0;
    bool instanced = false, enabled = false;
};
NvnStream nvn_stream(const NvnState& s, uint32_t i) {
    NvnStream st;
    if (i >= 16) return st;
    st.addr = s.vbuf[i].address;
    st.size = s.vbuf[i].size;
    st.stride = s.vstream[i].stride;
    st.instanced = s.vstream[i].divisor != 0;
    st.enabled = st.addr != 0;
    if (s.vstream[i].divisor > 1 && st.enabled)
        log_limited("nvn_vdivisor", 8, "rx: NVN vertex stream %u: instance divisor %u", i, s.vstream[i].divisor);
    return st;
}

bool nvn_fetchable(const Regs::VertexAttribute& a, uint32_t stride) {
    uint32_t size = 0;
    WGPUVertexFormat f = vertex_format(a, size);
    if (vertex_unpack(a)) {
        f = WGPUVertexFormat_Uint32;
        size = 4;
    }
    return f && !(stride & 3) && !(size && a.offset % std::min(4u, size)) && !(stride && a.offset + size > stride) &&
           !(!stride && a.offset + size > 2048);
}

}  // namespace

PullPlan nvn_pull_plan_uncached(const NvnState& s, const std::bitset<32>& loads);
/// (memoized: a plan depends on the attributes loaded, the vertex state (vtx_hash) and which streams have a buffer;
/// draws of a few materials take turns asking. RX_NO_PULL_PLAN_MEMO=1: computed each time)
PullPlan nvn_pull_plan(const NvnState& s, const std::bitset<32>& loads) {
    static const bool off = env_flag("RX_NO_PULL_PLAN_MEMO");
    if (off) return nvn_pull_plan_uncached(s, loads);
    uint32_t bufs = 0;
    for (int i = 0; i < 16; i++) bufs |= (s.vbuf[i].address != 0 ? 1u : 0u) << i;
    struct Memo {
        const NvnState* s = nullptr;
        uint64_t vtx = 0;
        uint32_t loads = 0, bufs = 0;
        PullPlan plan;
    };
    static Memo memo[256];
    const uint32_t l = (uint32_t)loads.to_ulong();
    Memo& m = memo[(s.vtx_hash ^ s.vtx_hash >> 29 ^ l * 0x9e3779b1u ^ bufs * 0x85ebca6bu) & 255];
    if (m.s == &s && m.vtx == s.vtx_hash && m.loads == l && m.bufs == bufs) return m.plan;
    m.s = &s, m.vtx = s.vtx_hash, m.loads = l, m.bufs = bufs;
    m.plan = nvn_pull_plan_uncached(s, loads);
    return m.plan;
}

PullPlan nvn_pull_plan_uncached(const NvnState& s, const std::bitset<32>& loads) {
    PullPlan plan;
    for (uint32_t i = 0; i < 16; i++) {
        if (!loads[i]) continue;
        const Regs::VertexAttribute a = nvn_attribute(s.vattr[i]);
        const NvnStream vs = nvn_stream(s, a.buffer % 16);
        if (a.constant || !vs.enabled || nvn_fetchable(a, vs.stride)) continue;
        uint32_t k = 0;
        while (k < plan.count && plan.streams[k] != a.buffer % 16) k++;
        if (k == plan.count) {
            if (plan.count == 3) continue;   // (more: the CPU converts them)
            plan.streams[plan.count++] = a.buffer % 16;
        }
        auto& p = plan.attr[i];
        p.enabled = true;
        p.stream = (uint8_t)k;
        p.instanced = vs.instanced;
        p.attribute = a.hex;
        p.stride = vs.stride;
    }
    const uint32_t max_buffers = dev.limits.maxVertexBuffers ? dev.limits.maxVertexBuffers : 16;
    uint32_t fetched = 0;
    bool constants = false;
    for (uint32_t i = 0; i < 16; i++) {
        if (!loads[i]) continue;
        const Regs::VertexAttribute a = nvn_attribute(s.vattr[i]);
        if (a.constant || !nvn_stream(s, a.buffer % 16).enabled) constants = true;
        else if (!plan.attr[i].enabled) fetched |= 1u << (a.buffer % 16);
    }
    for (uint32_t i = 16; i < 32; i++) constants |= loads[i];
    uint32_t buffers = (uint32_t)std::popcount(fetched) + (constants ? 1u : 0u);
    for (int st = 15; st >= 0 && buffers > max_buffers && plan.count < 3; st--) {
        if (!(fetched >> st & 1)) continue;
        const uint32_t k = plan.count++;
        plan.streams[k] = (uint32_t)st;
        const NvnStream vs = nvn_stream(s, (uint32_t)st);
        for (uint32_t i = 0; i < 16; i++) {
            const Regs::VertexAttribute a = nvn_attribute(s.vattr[i]);
            if (!loads[i] || a.constant || a.buffer % 16 != (uint32_t)st || plan.attr[i].enabled) continue;
            auto& p = plan.attr[i];
            p.enabled = true;
            p.stream = (uint8_t)k;
            p.instanced = vs.instanced;
            p.attribute = a.hex;
            p.stride = vs.stride;
        }
        buffers--;
    }
    return plan;
}

namespace {

/// The vertex buffers of an NVN draw (vertex_input's twin over NVN vertex state)
bool nvn_vertex_input(VertexInput& vi, const NvnState& s, const Shader::Info& info, uint32_t loads, uint32_t v0,
                      uint32_t v1, uint32_t base_instance, uint32_t instances, const CachedDraw& cached,
                      const PullPlan* pull) {
    struct Attr {
        uint32_t location, offset, size;
        WGPUVertexFormat format;
    };
    static std::array<std::vector<Attr>, 16> streams;
    for (auto& st : streams) st.clear();
    static std::vector<WGPUVertexAttribute> constants;
    constants.clear();
    static std::vector<uint32_t> converted;
    converted.clear();
    for (uint32_t i = 0; i < 32; i++) {
        if (!(loads >> i & 1)) continue;
        const Regs::VertexAttribute a = i < 16 ? nvn_attribute(s.vattr[i]) : [] {
            Regs::VertexAttribute c{};
            c.hex = 0;
            c.constant.Assign(1);
            return c;
        }();
        uint32_t size = 0;
        WGPUVertexFormat f = vertex_format(a, size);
        if (vertex_unpack(a)) {
            f = WGPUVertexFormat_Uint32;
            size = 4;
        }
        const NvnStream vs = nvn_stream(s, a.buffer % 16);
        const uint32_t stride = vs.stride;
        const bool constant = a.constant || !vs.enabled;
        if (!constant && pull && pull->attr[i].enabled) continue;
        if (!constant && (!f || (stride & 3) || (size && a.offset % std::min(4u, size)) ||
                          (stride && a.offset + size > stride) || (!stride && a.offset + size > 2048))) {
            converted.push_back(i);
            continue;
        }
        if (constant) {
            WGPUVertexAttribute c{};
            c.shaderLocation = i;
            c.offset = 0;
            using T = Regs::VertexAttribute::Type;
            c.format = a.constant ? WGPUVertexFormat_Float32x4
                       : a.type == T::SInt || a.type == T::SScaled   ? WGPUVertexFormat_Sint32x4
                       : a.type == T::UInt || a.type == T::UScaled ? WGPUVertexFormat_Uint32x4
                                                                   : WGPUVertexFormat_Float32x4;
            constants.push_back(c);
            vi.key.insert(vi.key.end(), {0xC0000000u | i, (uint32_t)c.format});
            continue;
        }
        streams[a.buffer % 16].push_back({i, a.offset, size, f});
    }
    if (cached.on && !converted.empty()) {
        count("nc_converted");
        return false;
    }
    for (uint32_t st = 0; st < 16; st++) {
        if (streams[st].empty()) continue;
        const NvnStream vs = nvn_stream(s, st);
        const uint32_t stride = vs.stride;
        const bool instanced = vs.instanced;
        uint32_t end = 0;
        for (const Attr& a : streams[st]) end = std::max(end, a.offset + a.size);
        if (cached.on) {
            uint64_t bytes;
            if (!stride) bytes = end;
            else if (instanced) bytes = (uint64_t)(base_instance + instances - 1) * stride + end;
            else if (!cached.indexed) bytes = (uint64_t)(cached.first + cached.count - 1) * stride + end;
            else bytes = std::max<uint64_t>(vs.size, end);
            bytes = std::min<uint64_t>((bytes + 3) & ~3ull, std::max<uint64_t>((vs.size + 3) & ~3ull, (end + 3) & ~3ull));
            if (vs.addr & 3) {
                count("nc_align");
                return false;
            }
            const BufferRef b = guest_buffer(vs.addr, bytes);
            if (!b.buffer) {
                count(bytes > (8u << 20) ? "nc_big" : "nc_unmapped");
                return false;
            }
            auto& attrs = vi.next_attributes();
            for (const Attr& a : streams[st]) {
                WGPUVertexAttribute w{};
                w.format = a.format;
                w.offset = a.offset;
                w.shaderLocation = a.location;
                attrs.push_back(w);
                vi.key.insert(vi.key.end(), {st << 16 | a.location, (uint32_t)a.format, a.offset});
            }
            vi.key.insert(vi.key.end(), {0x80000000u | st, stride, instanced ? 1u : 0u});
            WGPUVertexBufferLayout l{};
            l.stepMode = instanced ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
            l.arrayStride = stride;
            vi.layouts.push_back(l);
            vi.bindings.push_back({b.buffer, b.offset, b.size});
            continue;
        }
        const uint64_t first = instanced ? base_instance : v0;
        const uint64_t n_el = instanced ? instances : (uint64_t)v1 - v0 + 1;
        uint64_t bytes = stride ? (n_el - 1) * stride + end : end;
        bytes = (bytes + 3) & ~3ull;
        if (bytes > (64ull << 20)) return false;
        const uint64_t off = stream_ro.alloc(bytes, 16);
        if (off == UINT64_MAX) return false;
        const uint64_t src = vs.addr + first * stride;
        uint64_t n = bytes;
        if (first * stride >= vs.size) n = 0;
        else n = std::min<uint64_t>(n, vs.size - first * stride);
        if (n) gpu_read(src, stream_ro.at(off), n);
        if (n < bytes) std::memset(stream_ro.at(off + n), 0, bytes - n);
        count("up_vertex", (int64_t)bytes);
        auto& attrs = vi.next_attributes();
        for (const Attr& a : streams[st]) {
            WGPUVertexAttribute w{};
            w.format = a.format;
            w.offset = a.offset;
            w.shaderLocation = a.location;
            attrs.push_back(w);
            vi.key.insert(vi.key.end(), {st << 16 | a.location, (uint32_t)a.format, a.offset});
        }
        vi.key.insert(vi.key.end(), {0x80000000u | st, stride, instanced ? 1u : 0u});
        WGPUVertexBufferLayout l{};
        l.stepMode = instanced ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
        l.arrayStride = stride;
        vi.layouts.push_back(l);
        vi.bindings.push_back({stream_ro.buffer, off, bytes});
    }
    for (uint32_t i : converted) {
        const Regs::VertexAttribute a = nvn_attribute(s.vattr[i]);
        const NvnStream vs = nvn_stream(s, a.buffer % 16);
        const uint64_t first = vs.instanced ? base_instance : v0;
        const uint64_t n_el = vs.instanced ? instances : (uint64_t)v1 - v0 + 1;
        const uint64_t off = stream_ro.alloc(n_el * 16, 16);
        if (off == UINT64_MAX) return false;
        std::vector<uint8_t> src((size_t)((n_el - 1) * vs.stride + a.offset + 16));
        gpu_read(vs.addr + first * vs.stride, src.data(), src.size());
        uint8_t* dst = stream_ro.at(off);
        for (uint64_t e = 0; e < n_el; e++) {
            uint32_t v[4];
            decode_attribute(src.data() + e * vs.stride + a.offset, a, v);
            std::memcpy(dst + e * 16, v, 16);
        }
        using T = Regs::VertexAttribute::Type;
        WGPUVertexAttribute w{};
        w.shaderLocation = i;
        w.offset = 0;
        w.format = a.type == T::SInt || a.type == T::SScaled   ? WGPUVertexFormat_Sint32x4
                   : a.type == T::UInt || a.type == T::UScaled ? WGPUVertexFormat_Uint32x4
                                                               : WGPUVertexFormat_Float32x4;
        vi.next_attributes().push_back(w);
        WGPUVertexBufferLayout l{};
        l.stepMode = vs.instanced ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
        l.arrayStride = 16;
        vi.layouts.push_back(l);
        vi.bindings.push_back({stream_ro.buffer, off, n_el * 16});
        vi.key.insert(vi.key.end(), {0xA0000000u | i, (uint32_t)w.format, vs.instanced ? 1u : 0u});
    }
    if (!constants.empty()) {
        auto& ca = vi.next_attributes();
        ca.assign(constants.begin(), constants.end());
        WGPUVertexBufferLayout l{};
        l.stepMode = WGPUVertexStepMode_Vertex;
        l.arrayStride = 0;
        vi.layouts.push_back(l);
        vi.bindings.push_back({const_vertex_buffer, 0, 16});
    }
    for (size_t i = 0; i < vi.layouts.size(); i++) {
        vi.layouts[i].attributeCount = vi.attributes[i].size();
        vi.layouts[i].attributes = vi.attributes[i].data();
    }
    if (vi.layouts.size() > dev.limits.maxVertexBuffers) {
        log_limited("vbuffers", 4, "rx: %zu vertex buffers (WebGPU allows %u)", vi.layouts.size(),
                    dev.limits.maxVertexBuffers);
        return false;
    }
    return true;
}

/// The pulled vertex streams of an NVN draw (pulled_streams' twin)
bool nvn_pulled_streams(StageResources& out, const NvnState& s, const PullPlan& plan, const NvnDraw& d, bool cached) {
    if (!plan.count) return true;
    uint32_t pull[4] = {0, 0, 0, cached ? 0u : d.base_instance};
    struct Bind {
        BufferRef ref;
        uint64_t offset, size;
    } binds[3];
    const uint64_t align = dev.limits.minStorageBufferOffsetAlignment;
    for (uint32_t k = 0; k < plan.count; k++) {
        const NvnStream vs = nvn_stream(s, plan.streams[k]);
        uint64_t bytes = vs.size;
        if (!d.indexed && !vs.instanced) bytes = std::min<uint64_t>(bytes, (uint64_t)(d.first + d.count) * vs.stride + 64);
        if (vs.instanced) bytes = std::min<uint64_t>(bytes, (uint64_t)(d.base_instance + d.instances) * vs.stride + 64);
        uint64_t avail = 0;
        if (!hos_gpu_ptr(vs.addr, &avail)) return count("pull_unmapped"), false;
        if (!bytes || bytes > (8ull << 20)) bytes = 8ull << 20;
        bytes = std::min<uint64_t>(bytes, avail);
        if (!bytes) return count("pull_empty"), false;
        const BufferRef ref = guest_buffer(vs.addr, bytes);
        if (!ref.buffer) return count("pull_no_buffer"), false;
        const uint64_t off = ref.offset & ~(align - 1);
        pull[k] = (uint32_t)(ref.offset - off);
        uint64_t size = (ref.size + pull[k] + 3) & ~3ull;
        size = std::min<uint64_t>(size, wgpuBufferGetSize(ref.buffer) - off);
        binds[k] = {ref, off, size};
    }
    const uint64_t u = stream_ro.alloc(16, dev.limits.minUniformBufferOffsetAlignment);
    if (u == UINT64_MAX) return count("pull_space"), false;
    std::memcpy(stream_ro.at(u), pull, 16);
    uint32_t b = out.next_binding;
    WGPUBindGroupLayoutEntry l{};
    l.binding = b;
    l.visibility = WGPUShaderStage_Vertex;
    l.buffer.type = WGPUBufferBindingType_Uniform;
    l.buffer.hasDynamicOffset = true;
    out.layout.push_back(pack_layout(l));
    WGPUBindGroupEntry e{};
    e.binding = b++;
    e.buffer = stream_ro.buffer;
    e.size = 16;
    out.entries.push_back(e);
    out.dynamic_offsets.push_back((uint32_t)u);
    for (uint32_t k = 0; k < plan.count; k++) {
        WGPUBindGroupLayoutEntry sl{};
        sl.binding = b;
        sl.visibility = WGPUShaderStage_Vertex;
        sl.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        out.layout.push_back(pack_layout(sl));
        WGPUBindGroupEntry se{};
        se.binding = b++;
        se.buffer = binds[k].ref.buffer;
        se.offset = binds[k].offset;
        se.size = binds[k].size;
        out.entries.push_back(se);
    }
    count("pulled_draws");
    return true;
}


/// RX_DUMP_ATTRS (with a frame dump): the listed vertex programs' draws also log their vertex stage's uniform block
/// (vs ub[i]) and the constant buffers bound as storage (vs cbuf<n>[i]), for tools/wgsl_eval.py
std::vector<uint64_t> dump_attrs_programs() {
    std::vector<uint64_t> v;
    for (const char* p = getenv("RX_DUMP_ATTRS"); p && *p;) {
        char* e = nullptr;
        v.push_back(strtoull(p, &e, 16));
        if (!e || *e != ',') break;
        p = e + 1;
    }
    return v;
}
bool nvn_dump_vs_ub = false;
/// (debugging) a GPU buffer's bytes: copied to a staging buffer and mapped, waiting for the GPU
void read_buffer(WGPUBuffer src, uint64_t offset, uint8_t* dst, uint64_t size) {
    const uint64_t n = (size + 3) & ~3ull;
    WGPUBufferDescriptor d{};
    d.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    d.size = n;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(dev.device, &d);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(e, src, offset & ~3ull, staging, 0, n);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    struct Done { bool complete = false; WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; } done;
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        auto* dn = static_cast<Done*>(u1);
        dn->status = status;
        dn->complete = true;
    };
    mc.userdata1 = &done;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, n, mc);
    while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
    if (done.status == WGPUMapAsyncStatus_Success) {
        std::memcpy(dst, wgpuBufferGetConstMappedRange(staging, 0, n), (size_t)size);
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
}
void dump_words(const char* what, uint32_t index, const uint8_t* p, uint64_t bytes) {
    for (uint64_t i = 0; i < bytes / 16; i++) {
        uint32_t w[4];
        float f[4];
        std::memcpy(w, p + i * 16, 16);
        std::memcpy(f, p + i * 16, 16);
        hos_log("rx-frame:     vs %s%u[%llu] %08x %08x %08x %08x  (%g %g %g %g)", what, index, (unsigned long long)i, w[0],
                w[1], w[2], w[3], f[0], f[1], f[2], f[3]);
    }
}

/// One stage's uniform block, constant / storage buffers and textures for an NVN draw (stage_resources' twin; the
/// layout is the WGSL backend's)
bool nvn_vs_storage_writable;   // (nvn_vs_compute: the vertex stage's stores are made, by a compute program)
void nvn_stage_resources(StageResources& out, const NvnState& s, int stage, const Shader::Info& info,
                         WGPUShaderStage vis, const float rescale[16], const Attachments& at, uint32_t& dynamic_storage) {
    NvnBanks banks;
    banks.c1 = nvn_c1(s, stage);
    if (frame_dump_active()) dump_textures.clear();
    for (const auto& d : info.constant_buffer_descriptors)   // (reads past a uniform buffer's bound bytes)
        if (d.index >= 3 && d.index < 3 + 16 && s.ubo[stage][d.index - 3].addr &&
            info.constant_buffer_used_sizes[d.index] > s.ubo[stage][d.index - 3].size) {
            count("nvn_ubo_overread");
            log_limited("nvn_ubo_over", 8, "rx: an NVN %s program reads %#x bytes of uniform buffer %u, bound with %#llx",
                        stage ? "pixel" : "vertex", info.constant_buffer_used_sizes[d.index], d.index - 3,
                        (unsigned long long)s.ubo[stage][d.index - 3].size);
        }
    auto buffer_entry = [&](uint32_t b, WGPUBufferBindingType type, bool dynamic, WGPUBuffer buf, uint64_t off,
                            uint64_t size) {
        WGPUBindGroupLayoutEntry l{};
        l.binding = b;
        l.visibility = vis;
        l.buffer.type = type;
        l.buffer.hasDynamicOffset = dynamic;
        out.layout.push_back(pack_layout(l));
        WGPUBindGroupEntry e{};
        e.binding = b;
        e.buffer = buf;
        e.offset = dynamic ? 0 : off;
        e.size = size;
        out.entries.push_back(e);
        if (dynamic) out.dynamic_offsets.push_back((uint32_t)off);
    };
    const bool indirect = info.uses_global_memory || info.uses_cbuf_indirect;
    std::array<uint32_t, 18> base{};
    std::array<bool, 18> in_ub{};
    uint32_t ub_size = 4 + 2 * (uint32_t)info.texture_descriptors.size();
    if (!indirect)
        for (const auto& d : info.constant_buffer_descriptors) {
            const uint32_t size = std::max(Common::DivCeil(info.constant_buffer_used_sizes[d.index], 16U), 1U);
            // (as the translation lays the block out: big uniform buffers are bound in place, below)
            if (nvn_ub_limit() && d.index >= 3 && info.constant_buffer_used_sizes[d.index] > nvn_ub_limit()) continue;
            if (ub_size + size > 0x1000) continue;
            in_ub[d.index % 18] = true;
            base[d.index % 18] = ub_size;
            ub_size += size;
        }
    static std::vector<uint8_t> slot_bytes;
    slot_bytes.assign(info.texture_descriptors.size() * 32, 0);
    buffer_entry(0, WGPUBufferBindingType_Uniform, true, stream_ro.buffer, 0, (uint64_t)ub_size * 16);
    const size_t ub_dynamic = out.dynamic_offsets.size() - 1;
    uint32_t b = 1;
    for (const auto& d : info.constant_buffer_descriptors) {   // (the ones not in the block: read-only storage)
        if (in_ub[d.index % 18]) continue;
        uint8_t probe[4];
        const uint64_t bank = banks.read(s, stage, d.index, 0, probe, 4);
        const uint64_t size = std::clamp<uint64_t>(bank + 15 & ~15ull, 16, 0x10000);
        // a uniform buffer: bound where it is in the buffer cache, not copied for each draw (5-7 MB a frame at the
        // castle; a write to it while a batch uses it flushes the batch first, buffer_cache_written).
        // RX_NO_CBUF_IN_PLACE=1: copied, as before
        static const bool no_in_place = env_flag("RX_NO_CBUF_IN_PLACE");
        if (!no_in_place && d.index >= 3 && d.index < 3 + 16 && s.ubo[stage][d.index - 3].addr && buffer_cache_active()) {
            const BufferRef c = guest_buffer(s.ubo[stage][d.index - 3].addr, size);
            if (c.buffer && c.offset % dev.limits.minStorageBufferOffsetAlignment == 0) {
                if (nvn_dump_vs_ub && stage == 0) {
                    std::vector<uint8_t> tmp((size_t)std::min<uint64_t>(size, 0x1000));
                    banks.read(s, stage, d.index, 0, tmp.data(), tmp.size());
                    dump_words("cbuf", d.index, tmp.data(), tmp.size());
                }
                const bool dynamic = dynamic_storage > 0;
                dynamic_storage -= dynamic;
                buffer_entry(b, WGPUBufferBindingType_ReadOnlyStorage, dynamic, c.buffer, c.offset, size);
                count("cbuf_in_place", (int64_t)size);
                b += d.count;
                continue;
            }
        }
        const uint64_t off = stream_ro.alloc(size, dev.limits.minStorageBufferOffsetAlignment);
        if (off == UINT64_MAX) {
            out.ok = false;
            return;
        }
        banks.read(s, stage, d.index, 0, stream_ro.at(off), size);
        count("up_cbuf_storage", (int64_t)size);
        if (nvn_dump_vs_ub && stage == 0) dump_words("cbuf", d.index, stream_ro.at(off), std::min<uint64_t>(size, 0x1000));
        const bool dynamic = dynamic_storage > 0;
        dynamic_storage -= dynamic;
        buffer_entry(b, WGPUBufferBindingType_ReadOnlyStorage, dynamic, stream_ro.buffer, off, size);
        b += d.count;
    }
    for (const auto& d : info.storage_buffers_descriptors) {
        uint64_t raw_va = 0;
        uint32_t raw_size = 0;
        banks.read(s, stage, d.cbuf_index, d.cbuf_offset, &raw_va, 8);
        banks.read(s, stage, d.cbuf_index, d.cbuf_offset + 8, &raw_size, 4);
        const uint64_t va = raw_va & ~(uint64_t)(dev.limits.minStorageBufferOffsetAlignment - 1);
        uint64_t size = (uint64_t)raw_size + (raw_va - va);
        if (static const bool log_copies = env_flag("RX_LOG_SSBO_COPIES"); log_copies && size > (8u << 20))
            log_limited("ssbo_big_bind", 100, "rx: NVN %s program binds storage %llx+%x (c%u@%x)", stage ? "pixel" : "vertex",
                        (unsigned long long)raw_va, raw_size, d.cbuf_index, d.cbuf_offset);
        size = std::clamp<uint64_t>((size + 3) & ~3ull, 16, 16ull << 20);
        const bool writable = d.is_written && (stage == 1 || nvn_vs_storage_writable);
        if (frame_dump_active()) {
            char line[160];
            std::snprintf(line, sizeof line, "  %s ssbo %llx+%x%s gpu-written %d", stage ? "fs" : "vs",
                          (unsigned long long)raw_va, raw_size, writable ? " (written)" : "",
                          (int)(raw_va && gpu_written(raw_va, raw_size ? raw_size : 16)));
            dump_draw(line);
        }
        const StorageBinding sb = draw_storage(raw_va ? va : 0, size, writable);
        if (!sb.ok) {
            out.ok = false;
            return;
        }
        const bool dynamic = sb.in_stream && dynamic_storage > 0;
        dynamic_storage -= dynamic;
        buffer_entry(b, writable ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage, dynamic,
                     sb.buffer, sb.offset, sb.size);
        b += d.count;
    }
    if (!info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty() ||
        !info.image_descriptors.empty()) {
        log_limited("nvn_images", 8, "rx: an NVN graphics program uses images or texture buffers: not drawn yet");
        out.ok = false;
        return;
    }
    uint32_t t_index = 0;
    // (web: from the 17th texture on, a texture shares the sampler of the last earlier one of its kind that has its
    // own, as the WGSL backend declares them and stage_resources binds them: browsers allow 16 samplers a stage)
    uint32_t last_own[2] = {0, 0};
    for (const auto& d : info.texture_descriptors) {
        const bool shared_sampler = RX_WEB && t_index >= 16 && last_own[d.is_depth ? 1 : 0];
        if (!shared_sampler) last_own[d.is_depth ? 1 : 0] = b + 1;
        uint32_t primary = 0, secondary = 0;
        banks.read(s, stage, d.cbuf_index, d.cbuf_offset, &primary, 4);
        if (d.has_secondary) banks.read(s, stage, d.secondary_cbuf_index, d.secondary_cbuf_offset, &secondary, 4);
        const uint32_t handle = d.has_secondary ? (primary << d.shift_left) | (secondary << d.secondary_shift_left)
                                                : primary;
        BoundTexture t = nvn_texture(s, handle, 0, d);
        if (t.image) {   // sampling an image this pass draws into: WebGPU forbids it
            bool attached = at.depth == t.image;
            for (Image* c : at.color) attached |= c == t.image;
            if (attached) {
                Image* source = t.image;
                if (Image* snapshot = snapshot_for_feedback(*source)) {
                    t.view = snapshot->view(view_dimension(d.type), has_stencil(source->format));
                    t.image = nullptr;
                } else {
                    out.ok = false;
                    return;
                }
            }
        }
        std::memcpy(slot_bytes.data() + (size_t)t_index++ * 32, t.slots, sizeof t.slots);
        watch_tex_size(at, t);
        if (frame_dump_active()) {
            char tb[160];
            nvn_samp_par sp{};
            const bool has_sp = nvn_sampler_of((int)((handle >> 20) & 0xFFF), &sp);
            std::snprintf(tb, sizeof tb, " %c%s%08llx:%ux%u/%x h%x[%s f%u/%u w%u%u%u k%u]", t.source,
                          t.kind == TexKind::Depth ? "z" : "", (unsigned long long)t.guest, t.width, t.height, t.format,
                          handle, has_sp ? "" : "nosamp ", sp.minf, sp.magf, sp.ws, sp.wt, sp.wr, (unsigned)t.kind);
            dump_textures += tb;
        }
        WGPUBindGroupLayoutEntry tl{};
        tl.binding = b;
        tl.visibility = vis;
        tl.texture.sampleType = t.kind == TexKind::Float               ? WGPUTextureSampleType_Float
                                : t.kind == TexKind::UnfilterableFloat ? WGPUTextureSampleType_UnfilterableFloat
                                : t.kind == TexKind::Depth             ? WGPUTextureSampleType_Depth
                                : t.kind == TexKind::Uint              ? WGPUTextureSampleType_Uint
                                                                       : WGPUTextureSampleType_Sint;
        tl.texture.viewDimension = view_dimension(d.type);
        tl.texture.multisampled = d.is_multisample;
        out.layout.push_back(pack_layout(tl));
        WGPUBindGroupLayoutEntry sl{};
        sl.binding = b + 1;
        sl.visibility = vis;
        sl.sampler.type = t.kind == TexKind::Depth             ? WGPUSamplerBindingType_Comparison
                          : t.kind == TexKind::UnfilterableFloat ? WGPUSamplerBindingType_NonFiltering
                                                                 : WGPUSamplerBindingType_Filtering;
        if (!shared_sampler) out.layout.push_back(pack_layout(sl));
        WGPUBindGroupEntry te{};
        te.binding = b;
        te.textureView = t.view;
        out.entries.push_back(te);
        WGPUBindGroupEntry se{};
        se.binding = b + 1;
        se.sampler = t.sampler;
        if (!shared_sampler) out.entries.push_back(se);
        b += 2;
    }
    out.next_binding = b;
    // the uniform block: rescale, the texture slots, the constant banks that fit
    const uint64_t used_before = stream_ro.used;
    const uint64_t ub = stream_ro.alloc((uint64_t)ub_size * 16, dev.limits.minUniformBufferOffsetAlignment);
    if (ub == UINT64_MAX) {
        out.ok = false;
        return;
    }
    uint8_t* block = stream_ro.at(ub);
    std::memcpy(block, rescale, 64);
    for (const auto& d : info.constant_buffer_descriptors) {
        if (!in_ub[d.index % 18]) continue;
        const uint64_t size = (uint64_t)std::max(Common::DivCeil(info.constant_buffer_used_sizes[d.index], 16U), 1U) * 16;
        banks.read(s, stage, d.index, 0, block + (uint64_t)base[d.index % 18] * 16, size);
        count("up_cbuf", (int64_t)size);
    }
    if (!slot_bytes.empty()) std::memcpy(block + 64, slot_bytes.data(), slot_bytes.size());
    if (nvn_dump_vs_ub && stage == 0) dump_words("ub", 0, block, (uint64_t)ub_size * 16);
    // the same bytes as a block this batch already holds: that one is bound and this space given back (draws of a
    // material often carry equal blocks; then their bind group's offsets repeat and the group is not set again, and
    // less is uploaded: the browser's GPU process was the limit on the web. RX_NO_UB_DEDUPE=1 off)
    uint64_t bound = ub;
    static const bool no_dedupe = env_flag("RX_NO_UB_DEDUPE");
    if (!no_dedupe) {
        // (a direct-mapped table tagged with the batch: no allocation per block and nothing to clear; a slot taken
        // by another block only misses a dedupe. An unordered_map's node per block was ~3% of the replay thread)
        struct Same { uint64_t h = 0, ub = 0; int64_t batch = -1; };
        static std::vector<Same> same(1u << 14);
        const uint64_t bytes = (uint64_t)ub_size * 16;
        const uint64_t h = hash_bytes(block, bytes, bytes);
        Same& e = same[(h ^ h >> 29) & ((1u << 14) - 1)];
        if (e.batch == batch_number() && e.h == h) {
            if (!std::memcmp(stream_ro.at(e.ub), block, bytes) && stream_ro.used == ub + bytes) {
                bound = e.ub;
                stream_ro.used = used_before;
                count("ub_deduped", (int64_t)bytes);
            }
        } else {
            e = {h, ub, batch_number()};
        }
    }
    out.dynamic_offsets[ub_dynamic] = (uint32_t)bound;
}

WGPUCompareFunction nvn_compare(uint32_t f) {
    switch (f) {
    case 1: return WGPUCompareFunction_Never;
    case 2: return WGPUCompareFunction_Less;
    case 3: return WGPUCompareFunction_Equal;
    case 4: return WGPUCompareFunction_LessEqual;
    case 5: return WGPUCompareFunction_Greater;
    case 6: return WGPUCompareFunction_NotEqual;
    case 7: return WGPUCompareFunction_GreaterEqual;
    default: return WGPUCompareFunction_Always;
    }
}
WGPUStencilOperation nvn_stencil_op(uint32_t o) {
    switch (o) {
    case 2: return WGPUStencilOperation_Zero;
    case 3: return WGPUStencilOperation_Replace;
    case 4: return WGPUStencilOperation_IncrementClamp;
    case 5: return WGPUStencilOperation_DecrementClamp;
    case 6: return WGPUStencilOperation_Invert;
    case 7: return WGPUStencilOperation_IncrementWrap;
    case 8: return WGPUStencilOperation_DecrementWrap;
    default: return WGPUStencilOperation_Keep;
    }
}
WGPUBlendFactor nvn_blend_factor(uint32_t f) {
    switch (f) {
    case NVN_BLEND_FUNC_ZERO: return WGPUBlendFactor_Zero;
    case NVN_BLEND_FUNC_ONE: return WGPUBlendFactor_One;
    case NVN_BLEND_FUNC_SRC_COLOR: return WGPUBlendFactor_Src;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
    case NVN_BLEND_FUNC_SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return WGPUBlendFactor_OneMinusSrcAlpha;
    case NVN_BLEND_FUNC_DST_ALPHA: return WGPUBlendFactor_DstAlpha;
    case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return WGPUBlendFactor_OneMinusDstAlpha;
    case NVN_BLEND_FUNC_DST_COLOR: return WGPUBlendFactor_Dst;
    case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return WGPUBlendFactor_OneMinusDst;
    case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return WGPUBlendFactor_SrcAlphaSaturated;
    case NVN_BLEND_FUNC_CONSTANT_COLOR: case NVN_BLEND_FUNC_CONSTANT_ALPHA: return WGPUBlendFactor_Constant;
    case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR: case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA:
        return WGPUBlendFactor_OneMinusConstant;
    case NVN_BLEND_FUNC_SRC1_COLOR: case NVN_BLEND_FUNC_SRC1_ALPHA:
        log_limited("dual_source", 4, "rx: a blend uses the second source (dual-source blending): approximated");
        return WGPUBlendFactor_SrcAlpha;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA:
        log_limited("dual_source", 4, "rx: a blend uses the second source (dual-source blending): approximated");
        return WGPUBlendFactor_OneMinusSrcAlpha;
    default: return WGPUBlendFactor_One;
    }
}
WGPUBlendOperation nvn_blend_op(uint32_t e) {
    switch (e) {
    case NVN_BLEND_EQUATION_SUB: return WGPUBlendOperation_Subtract;
    case NVN_BLEND_EQUATION_REVERSE_SUB: return WGPUBlendOperation_ReverseSubtract;
    case NVN_BLEND_EQUATION_MIN: return WGPUBlendOperation_Min;
    case NVN_BLEND_EQUATION_MAX: return WGPUBlendOperation_Max;
    default: return WGPUBlendOperation_Add;
    }
}

/// Whether none of these targets was drawn in the last 64 frames (one_off_targets' twin, by the images' memory)
bool nvn_one_off(const Attachments& at) {
    static const bool off = env_flag("RX_SKIP_ONE_OFF");
    if (off) return false;
    uint64_t addr[9];
    int n = 0;
    for (Image* c : at.color)
        if (c) addr[n++] = c->guest;
    if (at.depth) addr[n++] = at.depth->guest;
    if (!n) return false;
    // the same targets again in the same frame (draw after draw of a pass): the same answer, and the marks below are
    // already this frame's
    static uint64_t last_addr[9];
    static int last_n = -1;
    static int64_t last_frame = -1;
    static bool last_answer;
    if (last_frame == frame_number && last_n == n && !std::memcmp(last_addr, addr, (size_t)n * sizeof addr[0]))
        return last_answer;
    struct Seen { int64_t last = -100, prev = -100; };
    static std::unordered_map<uint64_t, Seen> seen;
    bool recent = false;
    for (int i = 0; i < n; i++) {
        Seen& sn = seen[addr[i]];
        if (sn.last != frame_number) sn.prev = sn.last, sn.last = frame_number;
        recent |= sn.prev >= frame_number - 64;
    }
    std::memcpy(last_addr, addr, (size_t)n * sizeof addr[0]);
    last_n = n, last_frame = frame_number, last_answer = !recent;
    return !recent;
}

}  // namespace

/* A draw whose vertex program stores to memory: WebGPU's vertex stage cannot, so the program also runs as a compute
   program (ProgramPair::vs_compute_wgsl), one invocation per vertex and instance, with the vertex stage's resources
   and its storage writable (copies written back to guest memory after the batch, as pixel programs' are), before the
   draw. TOTK's Ultrahand picks the object under the reticle that way (a one-point draw reading the object-id target).
   Non-indexed draws from vertex 0 and instance 0 only (the compute program counts from 0); others are logged. */
void nvn_vs_compute(NvnState& s, const NvnDraw& d, ProgramPair& prog, const float rescale[16], const PullPlan& plan,
                    bool cached) {
    if (prog.vs_compute_failed) return;
    const uint32_t max = dev.limits.maxComputeWorkgroupsPerDimension;
    if (d.indexed || d.first || d.base_instance || !d.count || d.count > max || d.instances > max) {
        log_limited("nvn_vs_compute_draw", 8, "rx: a storing vertex program %016llx in a draw it cannot run as "
                    "compute (indexed %d, first %u, count %u, instance %u x%u)", (unsigned long long)prog.vs.key,
                    (int)d.indexed, d.first, d.count, d.base_instance, d.instances);
        return count("nvn_vs_compute_unsupported");
    }
    if (!prog.vs_compute_module) {
        WGPUShaderSourceWGSL src{};
        src.chain.sType = WGPUSType_ShaderSourceWGSL;
        src.code = WGPUStringView{prog.vs_compute_wgsl.data(), prog.vs_compute_wgsl.size()};
        WGPUShaderModuleDescriptor md{};
        md.nextInChain = &src.chain;
        scope_push();
        prog.vs_compute_module = program_module(&md);
        const std::string error = scope_pop();
        if (!error.empty() || !prog.vs_compute_module) {
            log_limited("nvn_vs_compute_module", 20, "rx: vertex program %016llx as compute: %s",
                        (unsigned long long)prog.vs.key, error.c_str());
            prog.vs_compute_failed = true;
            return;
        }
    }
    end_render_pass("pe_vs_compute");
    static StageResources cs;
    cs.layout.clear();
    cs.entries.clear();
    cs.dynamic_offsets.clear();
    cs.next_binding = 0;
    cs.ok = true;
    uint32_t dynamic_storage = dev.limits.maxDynamicStorageBuffersPerPipelineLayout;
    nvn_vs_storage_writable = true;
    nvn_stage_resources(cs, s, 0, prog.vs.info, WGPUShaderStage_Compute, rescale, Attachments{}, dynamic_storage);
    nvn_vs_storage_writable = false;
    if (!cs.ok || !nvn_pulled_streams(cs, s, plan, d, cached)) return count("nvn_vs_compute_skipped");
    std::vector<WGPUBindGroupLayoutEntry> layout;
    for (uint32_t v : cs.layout) {
        WGPUBindGroupLayoutEntry e = unpack_layout(v);
        e.visibility = WGPUShaderStage_Compute;
        layout.push_back(e);
    }
    struct Pipe { WGPUBindGroupLayout layout = nullptr; WGPUComputePipeline pipeline = nullptr; };
    static std::unordered_map<uint64_t, Pipe> pipes;
    const uint64_t h = hash_bytes(layout.data(), layout.size() * sizeof(WGPUBindGroupLayoutEntry)) ^
                       (uint64_t)(uintptr_t)prog.vs_compute_module * 0x9E3779B97F4A7C15ull;
    Pipe& p = pipes[h];
    if (!p.pipeline) {
        WGPUBindGroupLayoutDescriptor ld{};
        ld.entryCount = layout.size();
        ld.entries = layout.data();
        scope_push();
        p.layout = wgpuDeviceCreateBindGroupLayout(dev.device, &ld);
        WGPUPipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &p.layout;
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(dev.device, &pld);
        WGPUComputePipelineDescriptor cd{};
        cd.layout = pl;
        cd.compute.module = prog.vs_compute_module;
        cd.compute.entryPoint = sv("main");
        p.pipeline = wgpuDeviceCreateComputePipeline(dev.device, &cd);
        wgpuPipelineLayoutRelease(pl);
        const std::string error = scope_pop();
        if (!error.empty()) {
            log_limited("nvn_vs_compute_pipeline", 20, "rx: vertex program %016llx as compute, pipeline: %s",
                        (unsigned long long)prog.vs.key, error.c_str());
            prog.vs_compute_failed = true;
            return;
        }
    }
    WGPUBindGroupDescriptor bd{};
    bd.layout = p.layout;
    bd.entryCount = cs.entries.size();
    bd.entries = cs.entries.data();
    WGPUBindGroup group = RX_WEB_MT ? nullptr : wgpuDeviceCreateBindGroup(dev.device, &bd);
    WGPUComputePassEncoder pass = ce_begin_compute_pass(encoder());
    cp_set_pipeline(pass, p.pipeline);
    if (group) cp_set_bind_group(pass, 0, group, cs.dynamic_offsets.size(), cs.dynamic_offsets.data());
    else cp_set_bind_group_desc(pass, 0, bd.layout, cs.entries.data(), cs.entries.size());
    cp_dispatch(pass, d.count, d.instances, 1);
    cp_end(pass);
    if (group) wgpuBindGroupRelease(group);
    count("nvn_vs_compute");
    // RX_VS_COMPUTE_LOG=1 (debugging): the storage each dispatch writes, and what guest memory holds there (the
    // result of an earlier one, once written back)
    if (static const bool vlog = env_flag("RX_VS_COMPUTE_LOG"); vlog) {
        static int logged;
        NvnBanks banks;
        banks.c1 = nvn_c1(s, 0);
        for (const auto& sd : prog.vs.info.storage_buffers_descriptors) {
            if (!sd.is_written || logged >= 400) continue;
            uint64_t va = 0;
            banks.read(s, 0, sd.cbuf_index, sd.cbuf_offset, &va, 8);
            uint32_t w[8]{};
            gpu_read(va, w, sizeof w);
            logged++;
            hos_log("rx: vs %016llx as compute (%u x %u) writes %#llx; memory there: %08x %08x %08x %08x %08x %08x %08x %08x",
                    (unsigned long long)prog.vs.key, d.count, d.instances, (unsigned long long)va, w[0], w[1], w[2],
                    w[3], w[4], w[5], w[6], w[7]);
        }
    }
    if (frame_dump_active()) {
        char line[160];
        std::snprintf(line, sizeof line, "nvn vertex program %016llx as compute: %u x %u", (unsigned long long)prog.vs.key,
                      d.count, d.instances);
        dump_draw(line);
    }
}

void nvn_draw(NvnState& s, const NvnDraw& d) {
    // RX_NVN_NO_DRAWS=1 (measurement only: the screen is wrong): no draw is prepared, so the frame rate is what the
    // game's own threads allow (the web: how far preparing draws faster could go)
    if (static const bool no_draws = env_flag("RX_NVN_NO_DRAWS"); no_draws) return;
    // RX_DEBUG_DRAW_SPIN_NS=<ns> (measurement only): the replay thread spins that long per draw, to measure how much
    // the frame rate depends on the replay's time (what making it faster could give)
    if (static const int64_t spin = getenv("RX_DEBUG_DRAW_SPIN_NS") ? atoll(getenv("RX_DEBUG_DRAW_SPIN_NS")) : 0; spin) {
        const int64_t until = clock_ns() + spin;
        while (clock_ns() < until) _mm_pause();
    }
    struct DrawTimer {
        int64_t t0 = clock_ns();
        ~DrawTimer() { count("t_draw", clock_ns() - t0); spike.draws++; }
    } draw_timer;
    if (!const_vertex_buffer) {
        WGPUBufferDescriptor bd{};
        bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
        bd.size = 16;
        const_vertex_buffer = wgpuDeviceCreateBuffer(dev.device, &bd);
        const float v[4] = {0, 0, 0, 1};
        wgpuQueueWriteBuffer(dev.queue, const_vertex_buffer, 0, v, 16);
    }
    auto skip = [&](const char* why) {
        st_skipped++;
        count("skipped");
        count(why);
    };
    if (!d.count) return count("draws_empty");
    trim_bind_groups();
    const Attachments at = nvn_attachments(s);
    if (!at.width) return skip("no render target");
    if (frame_dump_enabled) dump_on_targets(at);
    if (static const bool log_res = env_flag("RX_LOG_SCENE_SIZE"); log_res) {   // (debugging: G-buffer size changes)
        int n = 0;
        for (Image* c : at.color) n += c != nullptr;
        static uint32_t lw, lh;
        if (n >= 4 && at.width >= 960 && (at.width != lw || at.height != lh)) {
            lw = at.width, lh = at.height;
            log_limited("scene_size", 400, "rx: scene %ux%u from frame %lld", lw, lh, (long long)frame_number);
        }
    }
    draw_must_run = nvn_one_off(at);
    struct MustRunReset {
        ~MustRunReset() { draw_must_run = false; }
    } must_run_reset;
    uint8_t out_types[8];
    for (int i = 0; i < 8; i++) {
        const Image* c = at.color[i];
        out_types[i] = (uint8_t)(!c || !VideoCore::Surface::IsPixelFormatInteger(c->pixel_format)
                                     ? Shader::AttributeType::Float
                                 : VideoCore::Surface::IsPixelFormatSignedInteger(c->pixel_format)
                                     ? Shader::AttributeType::SignedInt
                                     : Shader::AttributeType::UnsignedInt);
    }
    ProgramPair* prog = nvn_programs(s, out_types);
    if (!prog) return skip("programs");

    // ---- vertices and indices ----
    WGPUPrimitiveTopology topology{};
    const bool direct = direct_topology((uint32_t)d.prim, topology);
    const uint32_t index_size = d.index_type == NVN_INDEX_TYPE_UNSIGNED_BYTE ? 1 : d.index_type == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4;
    std::vector<uint32_t> indices;
    uint32_t vmin = 0, vmax = 0;
    bool indexed = d.indexed;
    const std::bitset<32> vs_loads(prog->vs.generic_loads());
    const PullPlan plan = nvn_pull_plan(s, vs_loads);
    static VertexInput vi;
    vi.clear();
    uint64_t index_off = 0, index_bytes = 0;
    WGPUIndexFormat index_format = WGPUIndexFormat_Undefined;
    bool cached = false;
    BufferRef index_ref;
    const bool restart_ok = !s.restart || !d.indexed || (index_size == 2 ? s.restart_index == 0xFFFF : s.restart_index == 0xFFFFFFFF);
    if (direct && restart_ok && buffer_cache_active() && (!d.indexed || index_size != 1)) {
        uint64_t estimate = 64 * 1024;
        for (const auto* info : {&prog->vs.info, prog->has_fs ? &prog->fs.info : nullptr}) {
            if (!info) continue;
            estimate += info->constant_buffer_descriptors.size() * 0x10100;
            estimate += info->storage_buffers_descriptors.size() * (1 << 20);
        }
        if (stream_ro.used + estimate > stream_ro.capacity || stream_rw.used + (8u << 20) > stream_rw.capacity)
            flush_batch();
        CachedDraw cd;
        cd.on = true;
        cd.indexed = d.indexed;
        cd.first = d.first;
        cd.count = d.count;
        cached = nvn_vertex_input(vi, s, prog->vs.info, prog->vs.generic_loads(), 0, 0, d.base_instance, d.instances, cd,
                                  &plan);
        if (cached && d.indexed) {
            index_ref = guest_buffer(d.index_gpu, (uint64_t)d.count * index_size);
            cached = index_ref.buffer && index_ref.offset % index_size == 0;
            index_format = index_size == 2 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
        }
        if (!cached) vi.clear();
        else count("draws_cached");
    }
    if (!cached) {
        if (d.indexed) {
            indices.resize(d.count);
            std::vector<uint8_t> raw((size_t)d.count * index_size);
            gpu_read(d.index_gpu, raw.data(), raw.size());
            for (uint32_t i = 0; i < d.count; i++) {
                const uint8_t* p = raw.data() + (size_t)i * index_size;
                indices[i] = index_size == 1 ? p[0] : index_size == 2 ? (uint32_t)(p[0] | p[1] << 8)
                                                                      : (uint32_t)(p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24);
            }
            vmin = UINT32_MAX;
            for (uint32_t& i : indices) {
                if (s.restart && i == s.restart_index) {
                    i = 0xFFFFFFFF;
                    continue;
                }
                const uint32_t v = (uint32_t)((int64_t)i + d.base_vertex);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
            if (vmin == UINT32_MAX) return skip("only restart indices");
        } else {
            vmin = d.first;
            vmax = d.first + d.count - 1;
        }
        if (!direct) {
            if (!d.indexed) {
                indices.resize(d.count);
                for (uint32_t i = 0; i < d.count; i++) indices[i] = d.first + i;
                indexed = true;
            } else {
                for (uint32_t& i : indices)
                    if (i != 0xFFFFFFFF) i = (uint32_t)((int64_t)i + d.base_vertex);
            }
            if (!expand_topology((uint32_t)d.prim, indices, topology)) return skip("topology");
        }
        if (vmax - vmin > (16u << 20)) return skip("vertex range");
        {
            uint64_t estimate = 64 * 1024 + (uint64_t)indices.size() * 4;
            for (const auto* info : {&prog->vs.info, prog->has_fs ? &prog->fs.info : nullptr}) {
                if (!info) continue;
                estimate += info->constant_buffer_descriptors.size() * 0x10100;
                estimate += info->storage_buffers_descriptors.size() * (1 << 20);
            }
            for (uint32_t st = 0; st < 16; st++)
                if (s.vbuf[st].address)
                    estimate += ((uint64_t)vmax - vmin + 1) * s.vstream[st].stride + 256 +
                                (uint64_t)d.instances * s.vstream[st].stride;
            if (stream_ro.used + estimate > stream_ro.capacity || stream_rw.used + (8u << 20) > stream_rw.capacity)
                flush_batch();
            if (estimate > stream_ro.capacity) return skip("too much data");
        }
        if (!nvn_vertex_input(vi, s, prog->vs.info, prog->vs.generic_loads(), vmin, vmax, d.base_instance, d.instances,
                              {}, &plan))
            return skip("vertex input");
        if (indexed) {
            const int64_t adjust = direct ? (int64_t)d.base_vertex - vmin : -(int64_t)vmin;
            const bool small = vmax - vmin < 0xFFFF;
            index_format = small ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
            index_bytes = ((uint64_t)indices.size() * (small ? 2 : 4) + 3) & ~3ull;
            index_off = stream_ro.alloc(index_bytes, 16);
            if (index_off == UINT64_MAX) return skip("index space");
            count("up_index", (int64_t)index_bytes);
            uint8_t* dst = stream_ro.at(index_off);
            for (size_t i = 0; i < indices.size(); i++) {
                const uint32_t v = indices[i] == 0xFFFFFFFF ? 0xFFFFFFFF : (uint32_t)((int64_t)indices[i] + adjust);
                if (small) {
                    const uint16_t sv = indices[i] == 0xFFFFFFFF ? 0xFFFF : (uint16_t)v;
                    std::memcpy(dst + i * 2, &sv, 2);
                } else {
                    std::memcpy(dst + i * 4, &v, 4);
                }
            }
            if (indices.size() & 1 && small) std::memset(dst + indices.size() * 2, 0, 2);
        }
    }

    // ---- the viewport transform (upper-left origin: NDC +y up, as WebGPU's) and per-draw uniforms ----
    float vx = s.viewport[0][0], vy = s.viewport[0][1], vw = s.viewport[0][2], vh = s.viewport[0][3];
    if (vw == 0.0f && vh == 0.0f) vx = 0, vy = 0, vw = (float)at.width, vh = (float)at.height;
    const float W = (float)at.width, H = (float)at.height;
    const bool minus_w = nvn_depth_minus_w() != 0;
    const float n = s.depth_near, f = s.depth_far;
    float rescale[16] = {0, 0, 1.0f, 0, W, H, 0, 0,
                         vw / W, (2.0f * vx + vw) / W - 1.0f, vh / H, 1.0f - (2.0f * vy + vh) / H,
                         minus_w ? (f - n) * 0.5f : f - n, minus_w ? (f + n) * 0.5f : n, 0, 0};
    {
        const int32_t vertex_offset = cached ? 0 : (int32_t)vmin;
        const int32_t instance_offset = cached ? -(int32_t)d.base_instance : 0;
        std::memcpy(&rescale[14], &vertex_offset, 4);
        std::memcpy(&rescale[15], &instance_offset, 4);
    }
    static const bool no_d16q = env_flag("RX_NO_D16Q");
    if (at.depth && at.depth->format == WGPUTextureFormat_Depth16Unorm && !no_d16q) rescale[6] = 1.0f;
    {
        uint32_t mask = 0;
        for (int i = 0; i < 8; i++) {
            const Image* c = at.color[i];
            if (!c || (c->format != WGPUTextureFormat_R16Float && c->format != WGPUTextureFormat_RG16Float &&
                       c->format != WGPUTextureFormat_RGBA16Float && c->format != WGPUTextureFormat_RGBA32Float))
                continue;
            switch (c->pixel_format) {
            case PixelFormat::R16_UNORM: case PixelFormat::R16G16_UNORM: case PixelFormat::R16G16B16A16_UNORM:
                mask |= 1u << i;
                break;
            case PixelFormat::R16_SNORM: case PixelFormat::R16G16_SNORM: case PixelFormat::R16G16B16A16_SNORM:
                mask |= 0x100u << i;
                break;
            default: break;
            }
        }
        rescale[7] = (float)mask;
    }

    static StageResources vs_res, fs_res;
    for (StageResources* sr : {&vs_res, &fs_res}) {
        sr->layout.clear();
        sr->entries.clear();
        sr->dynamic_offsets.clear();
        sr->next_binding = 0;
        sr->ok = true;
    }
    uint32_t dynamic_storage = dev.limits.maxDynamicStorageBuffersPerPipelineLayout;
    static const std::vector<uint64_t> dump_attrs_vs = dump_attrs_programs();
    nvn_dump_vs_ub = frame_dump_active() &&
                     std::find(dump_attrs_vs.begin(), dump_attrs_vs.end(), prog->vs.key) != dump_attrs_vs.end();
    if (nvn_dump_vs_ub) hos_log("rx-frame:   (uniforms of the next draw, vs %016llx)", (unsigned long long)prog->vs.key);
    nvn_stage_resources(vs_res, s, 0, prog->vs.info, WGPUShaderStage_Vertex, rescale, at, dynamic_storage);
    nvn_dump_vs_ub = false;
    if (!vs_res.ok) return skip("vertex resources");
    static std::string nvn_vs_textures;
    if (frame_dump_active()) nvn_vs_textures = dump_textures, dump_textures.clear();
    if (!nvn_pulled_streams(vs_res, s, plan, d, cached)) return skip("pulled vertices");
    if (prog->has_fs) {
        nvn_stage_resources(fs_res, s, 1, prog->fs.info, WGPUShaderStage_Fragment, rescale, at, dynamic_storage);
        if (!fs_res.ok) return skip("pixel resources");
    } else {
        for (Image* c : at.color)
            if (c) return skip("color targets without a pixel program");
    }
    if (!prog->vs_compute_wgsl.empty()) nvn_vs_compute(s, d, *prog, rescale, plan, cached);

    // RX_SKIP_FS=<pixel program key>[,...] (debugging): its draws are left out (what do they add to the picture?)
    static const std::vector<uint64_t> skip_fs = [] {
        std::vector<uint64_t> v;
        for (const char* p = getenv("RX_SKIP_FS"); p && *p;) {
            char* e = nullptr;
            v.push_back(strtoull(p, &e, 16));
            if (!e || *e != ',') break;
            p = e + 1;
        }
        return v;
    }();
    if (!skip_fs.empty() && std::find(skip_fs.begin(), skip_fs.end(), prog->fs.key) != skip_fs.end())
        return skip("RX_SKIP_FS");

    // ---- fixed-function state ----
    WGPUPrimitiveState prim{};
    prim.topology = topology;
    const bool strip = topology == WGPUPrimitiveTopology_TriangleStrip || topology == WGPUPrimitiveTopology_LineStrip;
    prim.stripIndexFormat = strip && indexed ? index_format : WGPUIndexFormat_Undefined;
    prim.frontFace = s.poly.front == NVN_FRONT_FACE_CCW ? WGPUFrontFace_CCW : WGPUFrontFace_CW;
    prim.unclippedDepth = dev.depth_clip_control && s.depth_clamp;
    prim.cullMode = WGPUCullMode_None;
    static const bool no_cull = env_flag("RX_NO_CULL");
    if (s.poly.cull && !no_cull) {
        if (s.poly.cull == NVN_FACE_FRONT_AND_BACK) return skip("culled");
        prim.cullMode = s.poly.cull == NVN_FACE_FRONT ? WGPUCullMode_Front : WGPUCullMode_Back;
    }
    WGPUDepthStencilState ds{};
    if (at.depth) {
        ds.format = at.depth->format;
        const bool test = s.ds.dt != 0;
        ds.depthWriteEnabled = test && s.ds.dw ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        ds.depthCompare = test ? nvn_compare(s.ds.df) : WGPUCompareFunction_Always;
        // RX_DEBUG_DEPTH_ALWAYS=<pixel program key> (debugging; with RX_NO_PIPELINE_CACHE=1): it passes the depth test
        if (static const uint64_t depth_always = getenv("RX_DEBUG_DEPTH_ALWAYS") ? strtoull(getenv("RX_DEBUG_DEPTH_ALWAYS"), nullptr, 16) : 0;
            depth_always && prog->fs.key == depth_always)
            ds.depthCompare = WGPUCompareFunction_Always;
        if (s.ds.st && has_stencil(ds.format)) {
            ds.stencilFront = {nvn_compare(s.ds.sf_f), nvn_stencil_op(s.ds.fail_f), nvn_stencil_op(s.ds.zfail_f),
                               nvn_stencil_op(s.ds.zpass_f)};
            ds.stencilBack = {nvn_compare(s.ds.sf_b), nvn_stencil_op(s.ds.fail_b), nvn_stencil_op(s.ds.zfail_b),
                              nvn_stencil_op(s.ds.zpass_b)};
            ds.stencilReadMask = s.stencil_value_mask[0] & 0xFF;
            ds.stencilWriteMask = s.stencil_mask[0] & 0xFF;
        } else {
            ds.stencilFront = ds.stencilBack = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep,
                                                WGPUStencilOperation_Keep, WGPUStencilOperation_Keep};
            ds.stencilReadMask = ds.stencilWriteMask = 0;
        }
        const bool tri = topology != WGPUPrimitiveTopology_PointList && topology != WGPUPrimitiveTopology_LineList &&
                         topology != WGPUPrimitiveTopology_LineStrip;
        if ((s.poly.offset_en & 4) && test && tri) {   // (NVN_POLYGON_OFFSET_ENABLE_FILL)
            ds.depthBias = (int32_t)std::lround(s.poly_offset[1]);
            ds.depthBiasSlopeScale = s.poly_offset[0];
            ds.depthBiasClamp = s.poly_offset[2];
        }
        // RX_DEBUG_DEPTH_BIAS=<pixel program key>:<units>[:<slope>] (debugging; with RX_NO_PIPELINE_CACHE=1)
        static uint64_t bias_fs = ~0ull;
        static int32_t bias_units;
        static float bias_slope;
        if (bias_fs == ~0ull) {
            bias_fs = 0;
            if (const char* e = getenv("RX_DEBUG_DEPTH_BIAS")) {
                char* p = nullptr;
                bias_fs = strtoull(e, &p, 16);
                if (p && *p == ':') bias_units = (int32_t)strtol(p + 1, &p, 10);
                if (p && *p == ':') bias_slope = strtof(p + 1, &p);
            }
        }
        if (bias_fs && prog->fs.key == bias_fs) ds.depthBias = bias_units, ds.depthBiasSlopeScale = bias_slope;
        for (WGPUStencilFaceState* fs : {&ds.stencilFront, &ds.stencilBack})
            if (!ds.stencilWriteMask) fs->failOp = fs->depthFailOp = fs->passOp = WGPUStencilOperation_Keep;
        auto inert = [](const WGPUStencilFaceState& fs) {
            return fs.compare == WGPUCompareFunction_Always && fs.failOp == WGPUStencilOperation_Keep &&
                   fs.depthFailOp == WGPUStencilOperation_Keep && fs.passOp == WGPUStencilOperation_Keep;
        };
        if (inert(ds.stencilFront) && inert(ds.stencilBack)) ds.stencilReadMask = ds.stencilWriteMask = 0;
    }
    if (prim.cullMode == WGPUCullMode_None &&
        (!prog->has_fs || (!prog->fs.info.loads.mask[(size_t)Shader::IR::Attribute::FrontFace] &&
                           !prog->fs.info.loads_indexed_attributes)) &&
        (!at.depth || !std::memcmp(&ds.stencilFront, &ds.stencilBack, sizeof ds.stencilFront)))
        prim.frontFace = WGPUFrontFace_CCW;
    static std::vector<WGPUColorTargetState> targets;
    targets.clear();
    std::array<WGPUBlendState, 8> blends{};
    uint32_t ntargets = 0;
    for (uint32_t i = 0; i < 8; i++)
        if (at.color[i]) ntargets = i + 1;
    static std::vector<uint32_t> key;
    key.clear();
    {
        const uint64_t vh_ = prog->vs.text_hash, fh = prog->has_fs ? prog->fs.text_hash : 0;
        key.insert(key.end(), {(uint32_t)vh_, (uint32_t)(vh_ >> 32), (uint32_t)fh, (uint32_t)(fh >> 32)});
    }
    for (uint32_t i = 0; i < ntargets; i++) {
        WGPUColorTargetState t{};
        Image* im = at.color[i];
        t.format = im ? im->format : WGPUTextureFormat_Undefined;
        if (im) {
            const uint32_t m = s.chmask.mask >> (4 * i) & 15;
            t.writeMask = (m & 1 ? WGPUColorWriteMask_Red : 0) | (m & 2 ? WGPUColorWriteMask_Green : 0) |
                          (m & 4 ? WGPUColorWriteMask_Blue : 0) | (m & 8 ? WGPUColorWriteMask_Alpha : 0);
            if (!prog->fs.info.stores_frag_color[i]) t.writeMask = WGPUColorWriteMask_None;
            const bool integer = VideoCore::Surface::IsPixelFormatInteger(im->pixel_format);
            const bool blend_on = (s.color.blend_en >> i & 1) && !integer && t.writeMask != WGPUColorWriteMask_None;
            if (blend_on) {
                const nvn_blend_st& bs = s.blend[i];
                WGPUBlendState& b = blends[i];
                b.color = {nvn_blend_op(bs.eq_rgb), nvn_blend_factor(bs.src_rgb), nvn_blend_factor(bs.dst_rgb)};
                b.alpha = {nvn_blend_op(bs.eq_a), nvn_blend_factor(bs.src_a), nvn_blend_factor(bs.dst_a)};
                for (WGPUBlendComponent* c : {&b.color, &b.alpha})
                    if (c->operation == WGPUBlendOperation_Min || c->operation == WGPUBlendOperation_Max)
                        c->srcFactor = c->dstFactor = WGPUBlendFactor_One;
                t.blend = &blends[i];
                key.insert(key.end(), {(uint32_t)b.color.operation, (uint32_t)b.color.srcFactor,
                                       (uint32_t)b.color.dstFactor, (uint32_t)b.alpha.operation,
                                       (uint32_t)b.alpha.srcFactor, (uint32_t)b.alpha.dstFactor});
            }
        }
        targets.push_back(t);
        key.insert(key.end(), {(uint32_t)t.format, (uint32_t)t.writeMask, t.blend ? 1u : 0u});
    }
    key.insert(key.end(), vi.key.begin(), vi.key.end());
    key.insert(key.end(), {0xDDDD, (uint32_t)prim.topology, (uint32_t)prim.stripIndexFormat, (uint32_t)prim.frontFace,
                           (uint32_t)prim.cullMode, (uint32_t)prim.unclippedDepth});
    if (at.depth)
        key.insert(key.end(),
                   {(uint32_t)ds.format, (uint32_t)ds.depthWriteEnabled, (uint32_t)ds.depthCompare,
                    (uint32_t)ds.stencilFront.compare, (uint32_t)ds.stencilFront.failOp,
                    (uint32_t)ds.stencilFront.depthFailOp, (uint32_t)ds.stencilFront.passOp,
                    (uint32_t)ds.stencilBack.compare, (uint32_t)ds.stencilBack.failOp,
                    (uint32_t)ds.stencilBack.depthFailOp, (uint32_t)ds.stencilBack.passOp, ds.stencilReadMask,
                    ds.stencilWriteMask, (uint32_t)ds.depthBias, std::bit_cast<uint32_t>(ds.depthBiasSlopeScale),
                    std::bit_cast<uint32_t>(ds.depthBiasClamp)});
    key.push_back(0xEEEE);
    for (const StageResources* res : {&vs_res, &fs_res}) {
        const uint64_t h = hash_bytes(res->layout.data(), res->layout.size() * sizeof(uint32_t));
        key.insert(key.end(), {(uint32_t)h, (uint32_t)(h >> 32), (uint32_t)res->layout.size()});
    }
    Pipeline* pipe = get_pipeline(key, *prog, vi, vs_res, prog->has_fs ? &fs_res : nullptr, prim,
                                  at.depth ? &ds : nullptr, targets, blends);
    if (!pipe) return skip("pipeline");
    if (pipe->state.load() == 0 && draw_must_run) {
        const int64_t w0 = hos_now_ns();
        if (!RX_WEB)
            if (std::unique_ptr<PipelineJob> job = pipeline_workers.steal(pipe)) compile(*job);
        const int64_t t0 = hos_now_ns();
        while (pipe->state.load() == 0 && hos_now_ns() - t0 < 5000000000LL) {
            if (RX_WEB) wgpuDevicePoll(dev.device, true, nullptr);
            else std::this_thread::yield();
        }
        spike.pipeline_ns += hos_now_ns() - w0;
    }
    if (pipe->state.load() != 1) return skip("pipeline compiling");

    // ---- scissor: the viewport's rectangle, the scissor, the target ----
    float sx0 = std::max(0.0f, std::min(vx, vx + vw)), sy0 = std::max(0.0f, std::min(vy, vy + vh));
    float sx1 = std::min(W, std::max(vx, vx + vw)), sy1 = std::min(H, std::max(vy, vy + vh));
    if (s.scissor[2] > 0 && s.scissor[3] > 0) {
        sx0 = std::max(sx0, (float)s.scissor[0]);
        sy0 = std::max(sy0, (float)s.scissor[1]);
        sx1 = std::min(sx1, (float)s.scissor[0] + s.scissor[2]);
        sy1 = std::min(sy1, (float)s.scissor[1] + s.scissor[3]);
    }
    const uint32_t ix0 = (uint32_t)std::floor(sx0), iy0 = (uint32_t)std::floor(sy0);
    const uint32_t ix1 = (uint32_t)std::min<double>(std::ceil(sx1), at.width);
    const uint32_t iy1 = (uint32_t)std::min<double>(std::ceil(sy1), at.height);
    if (sx1 <= sx0 || sy1 <= sy0 || ix1 <= ix0 || iy1 <= iy0) return skip("empty scissor");

    // ---- bind groups, record ----
    WGPUBindGroup g0 = bind_group(pipe->layouts[0], vs_res.entries);
    WGPUBindGroup g1 = prog->has_fs ? bind_group(pipe->layouts[1], fs_res.entries) : nullptr;
    // (RX_WRITEBACK_SMALL=<pixels> (experiment): every target up to that size is written back after each submission)
    static const uint64_t wb_small = getenv("RX_WRITEBACK_SMALL") ? strtoull(getenv("RX_WRITEBACK_SMALL"), nullptr, 10) : 0;
    // (a block view of a compressed texture: what it draws is that texture's data, read where it is decoded)
    for (Image* c : at.color)
        if (c && (draw_must_run || c->block_view || (uint64_t)c->width * c->height <= wb_small)) queue_target_writeback(c);
    // vertex or index data a pixel program of the open pass wrote as storage (its copy reaches the buffer cache when
    // the pass ends): the pass ends first, as for storage reads (draw_storage). RX_NO_VERTEX_HAZARD_BREAK=1: off
    if (static const bool no_vertex_hazard = env_flag("RX_NO_VERTEX_HAZARD_BREAK"); !no_vertex_hazard && pass_wrote_storage()) {
        bool hazard = false;
        for (uint32_t i = 0; i < 16 && !hazard; i++) {
            if (!(prog->vs.generic_loads() >> i & 1)) continue;
            const NvnStream st = nvn_stream(s, nvn_attribute(s.vattr[i]).buffer % 16);
            hazard = st.enabled && storage_written_this_pass(st.addr, st.size);
        }
        if (!hazard && d.indexed && d.index_gpu)
            hazard = storage_written_this_pass(d.index_gpu + (uint64_t)d.first * 4, (uint64_t)d.count * 4);
        if (hazard) end_render_pass("pe_vertex_hazard");
    }
    split_long_pass();
    WGPURenderPassEncoder pass = render_pass(at);
    static struct PassState {
        uint64_t pass = ~0ull, trims = 0;
        WGPURenderPipeline pipeline;
        WGPUBindGroup groups[2];
        std::vector<uint32_t> offsets[2];
        std::array<VertexInput::Binding, 16> vertex;
        uint32_t vertex_count;
        uint32_t scissor[4];
        WGPUColor blend;
        uint32_t stencil;
        WGPUBuffer index_buffer;
        WGPUIndexFormat index_format;
    } ps;
    if (ps.pass != pass_count() || ps.trims != bind_group_trims) {
        ps = PassState{};
        ps.pass = pass_count();
        ps.trims = bind_group_trims;
        rp_set_viewport(pass, 0, 0, W, H, 0.0f, 1.0f);
        ps.stencil = ~0u;
        ps.blend = {-1, -1, -1, -1};
    }
    if (ps.pipeline != pipe->pipeline) {
        rp_set_pipeline(pass, pipe->pipeline);
        ps.pipeline = pipe->pipeline;
    }
    if (ps.groups[0] != g0 || ps.offsets[0] != vs_res.dynamic_offsets) {
        rp_set_bind_group(pass, 0, g0, vs_res.dynamic_offsets.size(), vs_res.dynamic_offsets.data());
        ps.groups[0] = g0;
        ps.offsets[0] = vs_res.dynamic_offsets;
    }
    if (g1 && (ps.groups[1] != g1 || ps.offsets[1] != fs_res.dynamic_offsets)) {
        rp_set_bind_group(pass, 1, g1, fs_res.dynamic_offsets.size(), fs_res.dynamic_offsets.data());
        ps.groups[1] = g1;
        ps.offsets[1] = fs_res.dynamic_offsets;
    }
    for (uint32_t i = 0; i < vi.bindings.size() && i < 16; i++) {
        const auto& v = vi.bindings[i];
        if (i < ps.vertex_count && ps.vertex[i].buffer == v.buffer && ps.vertex[i].offset == v.offset &&
            ps.vertex[i].size == v.size)
            continue;
        rp_set_vertex_buffer(pass, i, v.buffer, v.offset, v.size);
        ps.vertex[i] = v;
    }
    ps.vertex_count = std::max<uint32_t>(ps.vertex_count, (uint32_t)vi.bindings.size());
    const uint32_t rect[4] = {ix0, iy0, ix1 - ix0, iy1 - iy0};
    if (std::memcmp(rect, ps.scissor, sizeof rect)) {
        rp_set_scissor(pass, rect[0], rect[1], rect[2], rect[3]);
        std::memcpy(ps.scissor, rect, sizeof rect);
    }
    const WGPUColor bc{s.blend_color[0], s.blend_color[1], s.blend_color[2], s.blend_color[3]};
    if (std::memcmp(&bc, &ps.blend, sizeof bc)) {
        rp_set_blend_constant(pass, &bc);
        ps.blend = bc;
    }
    if (ps.stencil != (s.stencil_ref[0] & 0xFF)) {
        rp_set_stencil_reference(pass, s.stencil_ref[0] & 0xFF);
        ps.stencil = s.stencil_ref[0] & 0xFF;
    }
    // an index buffer is bound whole, once a pass, and the draw starts at its indices (a draw's own range bound each
    // time was one more command a draw for the browser's GPU process)
    auto index_buffer = [&](WGPUBuffer b) {
        if (ps.index_buffer == b && ps.index_format == index_format) return;
        rp_set_index_buffer(pass, b, index_format, 0, WGPU_WHOLE_SIZE);
        ps.index_buffer = b;
        ps.index_format = index_format;
    };
    const uint32_t index_unit = index_format == WGPUIndexFormat_Uint16 ? 2 : 4;
    if (cached && indexed) {
        index_buffer(index_ref.buffer);
        rp_draw_indexed(pass, d.count, d.instances, (uint32_t)(index_ref.offset / index_unit), d.base_vertex,
                        d.base_instance);
    } else if (cached) {
        rp_draw(pass, d.count, d.instances, d.first, d.base_instance);
    } else if (indexed) {
        index_buffer(stream_ro.buffer);
        rp_draw_indexed(pass, (uint32_t)indices.size(), d.instances, (uint32_t)(index_off / index_unit), 0, 0);
    } else {
        rp_draw(pass, d.count, d.instances, 0, 0);
    }
    // RX_END_PASS_AFTER_FS=<pixel program key>[,...] (debugging): a draw of those ends its render pass, as RX_DUMP_AFTER's
    // snapshots do (what found the GPU particle rule below)
    static const std::vector<uint64_t> end_pass_after = [] {
        std::vector<uint64_t> v;
        for (const char* p = getenv("RX_END_PASS_AFTER_FS"); p && *p;) {
            char* e = nullptr;
            v.push_back(strtoull(p, &e, 16));
            if (!e || *e != ',') break;
            p = e + 1;
        }
        return v;
    }();
    if (!end_pass_after.empty() && std::find(end_pass_after.begin(), end_pass_after.end(), prog->fs.key) != end_pass_after.end())
        end_render_pass("pe_debug_after");
    // GPU particles (per-instance streams a dispatch wrote: TOTK's effect simulation) end their render pass. Drawn
    // several to a pass, some of their quads came out as long straight lines across the screen (the Zonai pot's vanish,
    // 2026-10-24); one to a pass they match the console. Why is open: not barriers between dispatch and draw (flushing
    // or waiting after each dispatch changed nothing), not rx's per-pass state (setting everything for every draw changed
    // nothing), not the first instance, the page merge or the dispatch's data. ~100 such draws a frame in the rainy
    // castle cost no measurable fps. RX_GPU_PARTICLE_PASS=0: off; =2: any vertex stream a dispatch wrote (skinned meshes
    // too, ~11% slower there)
    static const int particle_pass_set = getenv("RX_GPU_PARTICLE_PASS") ? atoi(getenv("RX_GPU_PARTICLE_PASS")) : 1;
    static const bool particle_ab = ab_named("RX_GPU_PARTICLE_PASS");   // (HOS_AB: what the rule costs)
    if (const int particle_pass = particle_ab && ab_flipped() ? !particle_pass_set : particle_pass_set; particle_pass) {
        bool gpu_fed = false;
        for (uint32_t i = 0; i < 16 && !gpu_fed; i++) {
            if (!(prog->vs.generic_loads() >> i & 1)) continue;
            const NvnStream st = nvn_stream(s, nvn_attribute(s.vattr[i]).buffer % 16);
            gpu_fed = st.enabled && (particle_pass == 2 || st.instanced) && gpu_written(st.addr, st.size);
        }
        if (gpu_fed) {
            // (RX_STATS: are they drawn from the buffer cache, or copied from guest memory, where the dispatch's
            // results arrive only after its batch ran)
            count(cached ? "gpu_fed_cached" : direct ? "gpu_fed_copied" : "gpu_fed_copied_topology");
            end_render_pass("pe_gpu_particles");
        }
    }
    if (frame_dump_active()) {
        char b[1600];
        std::snprintf(b, sizeof b,
                      "nvn draw vs %016llx fs %016llx %s%u x%u from %u prim %d | vp %g,%g %gx%g z %g..%g sc %u,%u %ux%u | "
                      "depth %d/%d f%d stencil %d ref %u | cull %d | blend %x | vs tex:%s | fs tex:%s",
                      (unsigned long long)prog->vs.key, (unsigned long long)prog->fs.key, indexed ? "idx " : "", d.count,
                      d.instances, d.base_instance, d.prim, s.viewport[0][0], s.viewport[0][1], s.viewport[0][2], s.viewport[0][3],
                      s.depth_near, s.depth_far, rect[0], rect[1], rect[2], rect[3], (int)s.ds.dt, (int)s.ds.dw,
                      (int)s.ds.df, (int)s.ds.st, s.stencil_ref[0] & 0xFF, (int)s.poly.cull, (unsigned)s.color.blend_en,
                      nvn_vs_textures.c_str(), dump_textures.c_str());
        dump_draw(b);
        // RX_DUMP_ATTRS=<vertex program key>[,...]: their vertex attributes (format, stream) and the first 2048
        // elements' values
        static const std::vector<uint64_t> attrs_vs = dump_attrs_programs();
        if (std::find(attrs_vs.begin(), attrs_vs.end(), prog->vs.key) != attrs_vs.end()) {
            for (uint32_t i = 0; i < 16; i++) {
                if (!(prog->vs.generic_loads() >> i & 1)) continue;
                const Regs::VertexAttribute a = nvn_attribute(s.vattr[i]);
                const NvnStream st = nvn_stream(s, a.buffer % 16);
                std::snprintf(b, sizeof b, "  attr %u: raw %08x stream %u offset %u size %u type %u constant %u | stream "
                              "%llx+%llx stride %u %s", i, s.vattr[i], (unsigned)a.buffer.Value(), (unsigned)a.offset.Value(),
                              (unsigned)a.size.Value(), (unsigned)a.type.Value(), (unsigned)a.constant.Value(),
                              (unsigned long long)st.addr, (unsigned long long)st.size, st.stride,
                              st.instanced ? "per instance" : "per vertex");
                dump_draw(b);
                if (a.constant || !st.enabled) continue;
                std::vector<uint8_t> gpu_vals;   // (RX_CHECK_ATTRS: the values printed are the GPU's)
                // RX_CHECK_ATTRS=1: the GPU's copy of the stream (the buffer cache's, as this draw read it) against
                // guest memory now: 16-byte records that differ
                if (static const bool check = env_flag("RX_CHECK_ATTRS"); check && st.stride) {
                    const uint64_t bytes = (uint64_t)(d.base_instance + d.instances) * st.stride;
                    const BufferRef ref = guest_buffer(st.addr, bytes);
                    if (ref.buffer) {
                        std::vector<uint8_t> gpu_copy(bytes), mem(bytes);
                        gpu_read(st.addr, mem.data(), bytes);   // (memory as the draw was recorded)
                        flush_batch();
                        wait_submitted();
                        read_buffer(ref.buffer, ref.offset, gpu_copy.data(), bytes);
                        gpu_vals = gpu_copy;
                        uint32_t diff = 0, first = ~0u;
                        for (uint64_t k = 0; k + 16 <= bytes; k += 16)
                            if (std::memcmp(gpu_copy.data() + k, mem.data() + k, 16)) {
                                diff++;
                                if (first == ~0u) first = (uint32_t)(k / 16);
                            }
                        std::snprintf(b, sizeof b, "  attr %u stream check: %u of %llu records differ (first %d)", i, diff,
                                      (unsigned long long)(bytes / 16), first == ~0u ? -1 : (int)first);
                        dump_draw(b);
                        {   // (the GPU copy's extremes: xyz magnitude, non-finite values)
                            float big = 0;
                            uint32_t nonfinite = 0, at = 0;
                            for (uint64_t k = 0; k + 16 <= bytes; k += 16) {
                                float g[4];
                                std::memcpy(g, gpu_copy.data() + k, 16);
                                for (int c = 0; c < 3; c++) {
                                    if (!std::isfinite(g[c])) nonfinite++;
                                    else if (std::fabs(g[c]) > big) big = std::fabs(g[c]), at = (uint32_t)(k / 16);
                                }
                            }
                            std::snprintf(b, sizeof b, "    gpu copy: largest |xyz| %g (record %u), %u non-finite", big, at,
                                          nonfinite);
                            dump_draw(b);
                        }
                        for (uint64_t k = 0, shown = 0; k + 16 <= bytes && shown < 3; k += 16)
                            if (std::memcmp(gpu_copy.data() + k, mem.data() + k, 16)) {
                                float g[4], m[4];
                                std::memcpy(g, gpu_copy.data() + k, 16);
                                std::memcpy(m, mem.data() + k, 16);
                                std::snprintf(b, sizeof b, "    record %llu: gpu %g %g %g %g | memory %g %g %g %g",
                                              (unsigned long long)(k / 16), g[0], g[1], g[2], g[3], m[0], m[1], m[2], m[3]);
                                dump_draw(b);
                                shown++;
                            }
                    }
                }
                const uint32_t n = std::min<uint32_t>(st.instanced ? d.instances : 4, 2048);
                for (uint32_t k = 0; k < n; k++) {
                    const uint32_t el = st.instanced ? d.base_instance + k : k;
                    float v[4] = {};
                    const uint64_t at = (uint64_t)el * st.stride + a.offset;
                    if (at + 16 <= gpu_vals.size()) std::memcpy(v, gpu_vals.data() + at, 16);
                    else gpu_read(st.addr + at, v, 16);
                    std::snprintf(b, sizeof b, "    [%u] %g %g %g %g", el, v[0], v[1], v[2], v[3]);
                    dump_draw(b);
                }
            }
        }
        if (s.poly.offset_en) {   // (polygon offset: enables, factor, units, clamp)
            std::snprintf(b, sizeof b, "  polygon offset enables %u factor %g units %g clamp %g", (unsigned)s.poly.offset_en,
                          s.poly_offset[0], s.poly_offset[1], s.poly_offset[2]);
            dump_draw(b);
        }
        char m[400];
        int n = std::snprintf(m, sizeof m, "  masks chmask %08x stores", s.chmask.mask);
        for (uint32_t i = 0; i < 8 && prog->has_fs; i++) n += std::snprintf(m + n, sizeof m - n, "%d", (int)prog->fs.info.stores_frag_color[i]);
        n += std::snprintf(m + n, sizeof m - n, " write");
        for (size_t i = 0; i < targets.size() && n < 360; i++)
            n += targets[i].blend
                     ? std::snprintf(m + n, sizeof m - n, " %xb%x/%x/%x/a%x/%x/%x", (unsigned)targets[i].writeMask,
                                     (unsigned)targets[i].blend->color.operation, (unsigned)targets[i].blend->color.srcFactor,
                                     (unsigned)targets[i].blend->color.dstFactor, (unsigned)targets[i].blend->alpha.operation,
                                     (unsigned)targets[i].blend->alpha.srcFactor, (unsigned)targets[i].blend->alpha.dstFactor)
                     : std::snprintf(m + n, sizeof m - n, " %x", (unsigned)targets[i].writeMask);
        dump_draw(m);
        std::string u = "  fs ubo:";
        for (int i = 0; i < 14; i++)
            if (s.ubo[1][i].addr) {
                char ub[64];
                std::snprintf(ub, sizeof ub, " %d=%llx+%llx", i, (unsigned long long)s.ubo[1][i].addr,
                              (unsigned long long)s.ubo[1][i].size);
                u += ub;
            }
        for (int st = 0; st < 2; st++) {   // (storage buffers: where they are and whether the program writes them)
            const Shader::Info& si = st ? prog->fs.info : prog->vs.info;
            if (st && !prog->has_fs) break;
            if (si.storage_buffers_descriptors.empty()) continue;
            NvnBanks sb;
            sb.c1 = nvn_c1(s, st);
            u += st ? " | fs ssbo:" : " | vs ssbo:";
            for (const auto& d : si.storage_buffers_descriptors) {
                uint64_t va = 0;
                uint32_t size = 0;
                sb.read(s, st, d.cbuf_index, d.cbuf_offset, &va, 8);
                sb.read(s, st, d.cbuf_index, d.cbuf_offset + 8, &size, 4);
                char ub[64];
                std::snprintf(ub, sizeof ub, " c%u@%x=%llx+%x%s", d.cbuf_index, d.cbuf_offset, (unsigned long long)va, size,
                              d.is_written ? "w" : "");
                u += ub;
            }
            u += " (bound:";
            for (int r = 0; r < 16; r++)
                if (s.ssbo[st][r].addr) {
                    char ub[48];
                    std::snprintf(ub, sizeof ub, " %d=%llx+%llx", r, (unsigned long long)s.ssbo[st][r].addr,
                                  (unsigned long long)s.ssbo[st][r].size);
                    u += ub;
                }
            u += ")";
        }
        // RX_NVN_PEEK=<bank>:<offset hex> (debugging): that word of the pixel stage's bank, as the draw reads it
        if (static const char* peek = getenv("RX_NVN_PEEK"); peek) {
            // (<bank>:<offset hex>[:<words>])
            const char* c1 = std::strchr(peek, ':');
            const char* c2 = c1 ? std::strchr(c1 + 1, ':') : nullptr;
            const uint32_t words = c2 ? std::clamp(atoi(c2 + 1), 1, 16) : 1;
            uint32_t v[16]{};
            NvnBanks pb;
            pb.read(s, 1, (uint32_t)atoi(peek), c1 ? strtoull(c1 + 1, nullptr, 16) : 0, v, 4 * words);
            u += " | peek";
            for (uint32_t i = 0; i < words; i++) {
                char ub[40];
                std::snprintf(ub, sizeof ub, " %g", std::bit_cast<float>(v[i]));
                u += ub;
            }
        }
        dump_draw(u.c_str());
        static const int after_rt = getenv("RX_DUMP_AFTER_RT") ? atoi(getenv("RX_DUMP_AFTER_RT")) & 7 : 0;
        dump_after_draw(prog->fs.key, env_flag("RX_DUMP_AFTER_DEPTH") ? at.depth : at.color[after_rt]);
        char what[96];
        std::snprintf(what, sizeof what, "nvn draw vs %016llx fs %016llx %ux%u", (unsigned long long)prog->vs.key,
                      (unsigned long long)prog->fs.key, at.width, at.height);
        dump_pixel_after(at, what);
        watch_buffer_after_draw(what);
    }
    dump_draw(nullptr);
    st_draws++;
    count("draws");
}

} // namespace rx
