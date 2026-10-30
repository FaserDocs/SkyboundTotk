// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WGSL backend (Rexcomp): derived from suyu's GLSL backend context. Module-scope declarations for one shader stage
// of a WebGPU pipeline.
//
// Layout:
//   - every resource of a stage is in one bind group: vertex 0, fragment 1, compute 0; binding numbers come from
//     Bindings::unified in declaration order (the uniform block, constant buffers indexed at run time, storage
//     buffers, textures (each a texture then its sampler), images)
//   - binding 0, the stage's uniform block <stage>_ub: array<vec4<u32>, N>, which the renderer binds at a dynamic
//     offset per draw: [0] resolution scaling, [1] render area, [2] (ax, bx, ay, by), [3] (az, bz, vertex id
//     offset, instance id offset): the viewport transform the vertex stage applies (WebGPU has no negative or
//     out-of-target viewports), [4 + t] the component swizzle of texture t (WebGPU views have none), then the
//     constant buffers (raw words; floats are bitcast), each at its base in declaration order
//   - constant buffers indexed at run time: var<storage, read> <stage>_cbuf<i>: array<vec4<u32>> of the bound size
//   - storage buffers: var<storage, read|read_write> <stage>_ssbo<i>: array<u32>
//   - GLSL's built-ins (gl_Position, gl_FragCoord, ...) and the stage's inputs/outputs (in_attr<i>, out_attr<i>,
//     frag_color<i>) are module-scope private variables with the GLSL names, so the instruction emitters read
//     like the GLSL backend's; the entry point (emit_wgsl.cpp) copies them from/to the WGSL built-ins and locations.

#include "common/div_ceil.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::WGSL {
namespace {
std::string_view ColorTextureType(TextureType type, bool is_multisample) {
    switch (type) {
    case TextureType::Color1D:
        return "texture_1d<f32>";
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return is_multisample ? "texture_multisampled_2d<f32>" : "texture_2d<f32>";
    case TextureType::ColorArray2D:
        return "texture_2d_array<f32>";
    case TextureType::Color3D:
        return "texture_3d<f32>";
    case TextureType::ColorCube:
        return "texture_cube<f32>";
    case TextureType::ColorArrayCube:
        return "texture_cube_array<f32>";
    default:
        throw NotImplementedException("WGSL texture type {}", type);
    }
}

std::string_view DepthTextureType(TextureType type) {
    switch (type) {
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return "texture_depth_2d";
    case TextureType::ColorArray2D:
        return "texture_depth_2d_array";
    case TextureType::ColorCube:
        return "texture_depth_cube";
    case TextureType::ColorArrayCube:
        return "texture_depth_cube_array";
    default:
        throw NotImplementedException("WGSL depth texture type {}", type);
    }
}

std::string_view StorageTextureType(TextureType type) {
    switch (type) {
    case TextureType::Color1D:
        return "texture_storage_1d";
    case TextureType::Color2D:
        return "texture_storage_2d";
    case TextureType::ColorArray2D:
        return "texture_storage_2d_array";
    case TextureType::Color3D:
        return "texture_storage_3d";
    default:
        throw NotImplementedException("WGSL storage texture type {}", type);
    }
}

std::string_view StorageFormat(ImageFormat format) {
    switch (format) {
    case ImageFormat::Typeless:
    case ImageFormat::R32_UINT:
        return "r32uint";
    case ImageFormat::R32G32_UINT:
        return "rg32uint";
    case ImageFormat::R32G32B32A32_UINT:
        return "rgba32uint";
    default:
        throw NotImplementedException("WGSL storage texture format {}", format);
    }
}
} // Anonymous namespace

thread_local std::array<u8, 64> ssbo_alias_next{};
thread_local std::array<u32, 64> ssbo_alias_words_next{};
thread_local bool vs_as_compute_next{};

EmitContext::EmitContext(IR::Program& program, Bindings& bindings, const Profile& profile_,
                         const RuntimeInfo& runtime_info_)
    : info{program.info}, profile{profile_}, runtime_info{runtime_info_}, stage{program.stage} {
    ssbo_alias = ssbo_alias_next;
    ssbo_alias_next = {};
    ssbo_alias_words = ssbo_alias_words_next;
    ssbo_alias_words_next = {};
    vs_compute = vs_as_compute_next && (stage == Stage::VertexA || stage == Stage::VertexB);
    vs_as_compute_next = false;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const std::string_view name{IR::NameOf(inst.GetOpcode())};
            if (name.starts_with("StorageAtomic") && inst.Arg(0).IsImmediate() && inst.Arg(0).U32() < 64) {
                atomic_ssbo.set(MapSsbo(inst.Arg(0).U32()));
            } else if (name.starts_with("SharedAtomic")) {
                atomic_smem = true;
            }
        }
    }
    switch (program.stage) {
    case Stage::VertexA:
    case Stage::VertexB:
        stage_name = "vs";
        group = 0;
        break;
    case Stage::Fragment:
        stage_name = "fs";
        position_name = "gl_FragCoord";
        group = 1;
        break;
    case Stage::Compute:
        stage_name = "cs";
        group = 0;
        break;
    default:
        throw NotImplementedException("WebGPU has no {} shaders", program.stage);
    }
    header += "diagnostic(off, derivative_uniformity);\n";
    DefineBuiltins();
    for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
        if (info.loads.Generic(index)) {
            header += fmt::format("var<private> in_attr{}: vec4<f32>;\n", index);
        }
        if (info.stores.Generic(index)) {
            header += fmt::format("var<private> out_attr{}: vec4<f32>;\n", index);
            const GenericElementInfo element_info{
                .name = fmt::format("out_attr{}", index),
                .first_element = 0,
                .num_components = 4,
            };
            output_generics[index].fill(element_info);
        }
    }
    if (stage == Stage::Fragment) {
        for (size_t index = 0; index < info.stores_frag_color.size(); ++index) {
            if (info.stores_frag_color[index]) {
                header += fmt::format("var<private> frag_color{}: vec4<f32>;\n", index);
            }
        }
    }
    // binding 0: the uniform block (see the layout above)
    ub_name = fmt::format("{}_ub", stage_name);
    ub_size = 4 + 2 * static_cast<u32>(info.texture_descriptors.size());
    if (!(info.uses_global_memory || info.uses_cbuf_indirect)) {
        // in declaration order, while they fit (the renderer lays the block out the same way)
        for (const auto& desc : info.constant_buffer_descriptors) {
            const u32 size{(std::max)(Common::DivCeil(info.constant_buffer_used_sizes[desc.index], 16U), 1U)};
            if (runtime_info.cbuf_ub_limit && desc.index >= 3 &&
                info.constant_buffer_used_sizes[desc.index] > runtime_info.cbuf_ub_limit) {
                continue;   // (bound in place: copying it for every draw costs more)
            }
            if (ub_size + size > 0x1000) {
                continue;
            }
            cbuf_in_ub[desc.index] = true;
            ub_cbuf_base[desc.index] = ub_size;
            ub_size += size;
        }
    }
    header += fmt::format("@group({}) @binding({}) var<uniform> {}: array<vec4<u32>, {}>;\n", group,
                          bindings.unified++, ub_name, ub_size);
    header += fmt::format("fn rescale_(i: u32) -> vec4<f32> {{ return bitcast<vec4<f32>>({}[i]); }}\n", ub_name);
    // Depth for a 16-bit target, on its step: the nearest step (halves up) plus a quarter, so that rounding and
    // truncating conversions agree and a value one draw writes passes another draw's EQUAL test (TOTK keeps
    // material numbers in a D16 target as 0.5 + id/510: even numbers fall halfway between two steps). On when
    // render_area().z is not 0 (the renderer sets it for draws into a 16-bit depth target).
    header += "fn d16q_(d: f32) -> f32 { return min((floor(clamp(d, 0.0, 1.0) * 65535.0 + 0.5) + 0.25) / 65535.0, 1.0); }\n";
    header += "fn scaling() -> vec4<f32> { return rescale_(0u); }\n"
              "fn render_area() -> vec4<f32> { return rescale_(1u); }\n";
    // opq_(x) == x, but the host compiler cannot see it (the uniform block's lane [0].w is always 0): it keeps an
    // addition of a tiny epsilon from being reassociated away, e.g. (1 - a) + 1e-10 into (1 + 1e-10) - a = 1 - a,
    // which then divides by zero where the guest's compiler kept the epsilon
    header += fmt::format("fn opq_(x: f32) -> f32 {{ return bitcast<f32>(bitcast<u32>(x) | {}[0u].w); }}\n", ub_name);
    DefineConstantBuffers(bindings);
    DefineConstantBufferIndirect();
    DefineStorageBuffers(bindings);
    SetupTextures(bindings);
    SetupImages(bindings);
    DefineHelperFunctions();
    DefineConstants();
}

void EmitContext::DefineBuiltins() {
    header += "var<private> gl_Position: vec4<f32>;\n"
              "var<private> gl_FragCoord: vec4<f32>;\n"
              "var<private> gl_VertexID: i32;\n"
              "var<private> gl_InstanceID: i32;\n"
              "var<private> gl_BaseVertex: i32;\n"
              "var<private> gl_BaseInstance: i32;\n"
              "var<private> gl_DrawID: i32;\n"
              "var<private> gl_FrontFacing: bool;\n"
              "var<private> gl_FragDepth: f32;\n"
              "var<private> gl_PointSize: f32;\n"
              "var<private> gl_PointCoord: vec2<f32>;\n"
              "var<private> gl_ClipDistance: array<f32, 8>;\n"
              "var<private> gl_Layer: i32;\n"
              "var<private> gl_ViewportIndex: i32;\n"
              "var<private> gl_PrimitiveID: i32;\n"
              "var<private> gl_SampleID: i32;\n"
              "var<private> gl_SampleMask: array<i32, 1>;\n"
              "var<private> gl_HelperInvocation: bool;\n"
              "var<private> gl_LocalInvocationID: vec3<u32>;\n"
              "var<private> gl_WorkGroupID: vec3<u32>;\n"
              "var<private> gl_SubGroupInvocationARB: u32;\n"
              "var<private> discarded_: bool;\n"
              "var<private> F32_INF_: u32 = 0x7f800000u;\n"
              "var<private> F32_NINF_: u32 = 0xff800000u;\n"
              "var<private> F32_NAN_: u32 = 0x7fc00000u;\n"
              // a call is never a const-expression: literal arguments of log2, sqrt, ... are not folded (and so
              // not rejected) at shader creation
              "fn rt_(x: f32) -> f32 { return x; }\n";
}

void EmitContext::DefineConstantBuffers(Bindings& bindings) {
    for (const auto& desc : info.constant_buffer_descriptors) {
        if (cbuf_in_ub[desc.index]) {
            continue;
        }
        // not in the uniform block (indexed at run time, or no room): a read-only storage buffer of the bound size
        header += fmt::format("@group({}) @binding({}) var<storage, read> {}_cbuf{}: array<vec4<u32>>;\n", group,
                              bindings.unified, stage_name, desc.index);
        // reads at a run-time offset (CbufAt): past the bound size the hardware reads 0, where WebGPU's bounds checks
        // would give the buffer's last vec4 (TOTK's particle programs index their emitter's tables so: long spikes)
        if (!CbufClamp()) {
            header += fmt::format("fn {0}_cbuf{1}_at(i: u32) -> vec4<u32> {{ if (i < arrayLength(&{0}_cbuf{1})) {{ "
                                  "return {0}_cbuf{1}[i]; }} return vec4<u32>(0u); }}\n",
                                  stage_name, desc.index);
        }
        cbuf_bindings.push_back({bindings.unified, desc.index});
        bindings.unified += desc.count;
    }
}

void EmitContext::DefineConstantBufferIndirect() {
    if (!info.uses_cbuf_indirect) {
        return;
    }
    header += "fn GetCbufIndirect(binding: u32, offset: u32) -> vec4<u32> {\n  switch (binding) {\n";
    for (const auto& desc : info.constant_buffer_descriptors) {
        header += fmt::format(fmt::runtime(CbufClamp() ? "    case {0}u: {{ return {1}_cbuf{0}[offset]; }}\n"
                                                       : "    case {0}u: {{ return {1}_cbuf{0}_at(offset); }}\n"),
                              desc.index, stage_name);
    }
    header += "    default: { return vec4<u32>(0u); }\n  }\n}\n";
}

void EmitContext::DefineStorageBuffers(Bindings& bindings) {
    u32 index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        if (index < 64 && ssbo_alias[index]) {   // (uses another descriptor's binding)
            bindings.unified += desc.count;
            index += desc.count;
            continue;
        }
        // WebGPU forbids writable storage buffers in vertex shaders
        // compute: every storage buffer read_write (they may share one GPU buffer, and WebGPU forbids a buffer in
        // one dispatch as both read-only and writable storage)
        const bool writable{stage == Stage::Compute ||
                            (desc.is_written && ((stage != Stage::VertexA && stage != Stage::VertexB) || vs_compute))};
        const bool atomic{index < 64 && atomic_ssbo[index]};
        header += fmt::format("@group({}) @binding({}) var<storage, {}> {}_ssbo{}: array<{}>;\n", group,
                              bindings.unified, writable || atomic ? "read_write" : "read", stage_name, index,
                              atomic ? "atomic<u32>" : "u32");
        ssbo_bindings.push_back({bindings.unified, index});
        bindings.unified += desc.count;
        index += desc.count;
    }
}

void EmitContext::SetupTextures(Bindings& bindings) {
    if (!info.texture_buffer_descriptors.empty()) {
        throw NotImplementedException("WGSL texture buffers");
    }
    textures.reserve(info.texture_descriptors.size());
    // Browsers allow 16 samplers a stage (wgpu more): from the 17th texture on, a texture samples with the sampler of
    // the last earlier texture of its kind that has one (gpu/render/rx_draw.cpp binds the same way)
    u32 last_own[2] = {0, 0};   // the last texture with a sampler of its own, by is_depth (binding + 1; 0: none)
    for (const auto& desc : info.texture_descriptors) {
        if (desc.count > 1) {
            throw NotImplementedException("WGSL texture arrays");
        }
        const u32 binding{bindings.unified};
        u32 shared{0};
#ifdef __EMSCRIPTEN__
        if (textures.size() >= 16) shared = last_own[desc.is_depth ? 1 : 0];   // (binding + 1)
#endif
        textures.push_back({binding, desc.count, shared});
        if (!shared) last_own[desc.is_depth ? 1 : 0] = binding + 1;
        const auto type{desc.is_depth ? DepthTextureType(desc.type)
                                      : ColorTextureType(desc.type, desc.is_multisample)};
        header += fmt::format("@group({}) @binding({}) var tex{}: {};\n", group, binding, binding, type);
        if (!shared)
            header += fmt::format("@group({}) @binding({}) var samp{}: {};\n", group, binding + 1, binding,
                                  desc.is_depth ? "sampler_comparison" : "sampler");
        // WebGPU texture views have no component swizzle and samplers no border color or LOD bias: per texture,
        // two slots of the uniform block: [0] the TIC's component mapping (0-7 each; x bits 3-5: the axes whose
        // wrap mode is border, y bits 3-5: the sampler's depth compare function, z bits 8-31: the sampler's LOD bias
        // in 1/256), [1] the border color
        const u32 swizzle_index{4 + 2 * static_cast<u32>(textures.size() - 1)};
        header += fmt::format("fn swizzle{}(v: vec4<f32>) -> vec4<f32> {{\n"
                              "  let s = {}[{}u] & vec4<u32>(7u);\n"
                              "  let c = array<f32,8>(0.0,0.0,v.x,v.y,v.z,v.w,1.0,1.0);\n"
                              "  return vec4<f32>(c[s.x],c[s.y],c[s.z],c[s.w]);\n}}\n",
                              binding, ub_name, swizzle_index);
        header += fmt::format("fn lodbias{}() -> f32 {{ return f32(bitcast<i32>({}[{}u].z) >> 8u) / 256.0; }}\n",
                              binding, ub_name, swizzle_index);
        header += fmt::format(
            "fn outside{0}(c: vec3<f32>) -> bool {{\n"
            "  let m = ({1}[{2}u].x >> 3u) & 7u;\n"
            "  return ((m & 1u) != 0u && (c.x < 0.0 || c.x > 1.0)) || ((m & 2u) != 0u && (c.y < 0.0 || c.y > 1.0)) ||"
            " ((m & 4u) != 0u && (c.z < 0.0 || c.z > 1.0));\n}}\n"
            "fn border{0}(c: vec3<f32>, v: vec4<f32>) -> vec4<f32> {{\n"
            "  return select(v, bitcast<vec4<f32>>({1}[{3}u]), outside{0}(c));\n}}\n"
            "fn bordercmp{0}(c: vec3<f32>, dref: f32, v: f32) -> f32 {{\n"
            "  if (!outside{0}(c)) {{ return v; }}\n"
            "  let b = bitcast<f32>({1}[{3}u].x);\n"
            "  switch (({1}[{2}u].y >> 3u) & 7u) {{\n"
            "    case 0u: {{ return 0.0; }}\n"
            "    case 1u: {{ return select(0.0, 1.0, dref < b); }}\n"
            "    case 2u: {{ return select(0.0, 1.0, dref == b); }}\n"
            "    case 3u: {{ return select(0.0, 1.0, dref <= b); }}\n"
            "    case 4u: {{ return select(0.0, 1.0, dref > b); }}\n"
            "    case 5u: {{ return select(0.0, 1.0, dref != b); }}\n"
            "    case 6u: {{ return select(0.0, 1.0, dref >= b); }}\n"
            "    default: {{ return 1.0; }}\n"
            "  }}\n}}\n",
            binding, ub_name, swizzle_index, swizzle_index + 1);
        texture_bindings.push_back({binding, static_cast<u32>(textures.size() - 1)});
        bindings.unified += 2;
    }
}

void EmitContext::SetupImages(Bindings& bindings) {
    if (!info.image_buffer_descriptors.empty()) {
        throw NotImplementedException("WGSL image buffers");
    }
    images.reserve(info.image_descriptors.size());
    for (const auto& desc : info.image_descriptors) {
        if (desc.count > 1) {
            throw NotImplementedException("WGSL image arrays");
        }
        images.push_back({bindings.unified, desc.count});
        header += fmt::format("@group({}) @binding({}) var img{}: {}<{}, {}>;\n", group, bindings.unified,
                              bindings.unified, StorageTextureType(desc.type), StorageFormat(desc.format),
                              desc.is_read ? "read_write" : "write");
        bindings.unified += 1;
    }
}

void EmitContext::DefineHelperFunctions() {
    header += "fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {"
              " let lo = a.x + b.x; return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x)); }\n"
              "fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {"
              " return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x)); }\n"
              "fn shl64(a: vec2<u32>, s: u32) -> vec2<u32> {"
              " if (s == 0u) { return a; } if (s >= 32u) { return vec2<u32>(0u, a.x << (s - 32u)); }"
              " return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s))); }\n"
              "fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {"
              " if (s == 0u) { return a; } if (s >= 32u) { return vec2<u32>(a.y >> (s - 32u), 0u); }"
              " return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s); }\n"
              "fn sar64(a: vec2<u32>, s: u32) -> vec2<u32> {"
              " let hi = i32(a.y); if (s == 0u) { return a; }"
              " if (s >= 32u) { return vec2<u32>(u32(hi >> (s - 32u)), u32(hi >> 31u)); }"
              " return vec2<u32>((a.x >> s) | (a.y << (32u - s)), u32(hi >> s)); }\n";
    header += "fn isnan_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }\n"
              "fn isinf_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }\n"
              "fn isnan2_(x: vec2<f32>) -> vec2<bool> { return vec2<bool>(isnan_(x.x), isnan_(x.y)); }\n"
              // a NaN level of detail or bias picks level 0 on Maxwell (its fixed-point conversion); on PC GPUs the
              // sample itself becomes NaN (TOTK: white blobs from an environment cube sampled at a NaN roughness LOD)
              "fn lodok_(x: f32) -> f32 { return select(x, 0.0, isnan_(x)); }\n";
    if (info.uses_global_increment || info.uses_shared_increment) {
        header += "fn CasIncrement(op_a: u32, op_b: u32) -> u32 { return select(op_a + 1u, 0u, op_a >= op_b); }\n";
    }
    if (info.uses_global_decrement || info.uses_shared_decrement) {
        header += "fn CasDecrement(op_a: u32, op_b: u32) -> u32 {"
                  " return select(op_a - 1u, op_b, op_a == 0u || op_a > op_b); }\n";
    }
    if (info.uses_atomic_f32_add) {
        header += "fn CasFloatAdd(op_a: u32, op_b: f32) -> u32 { return bitcast<u32>(bitcast<f32>(op_a) + op_b); }\n";
    }
    if (info.uses_atomic_s32_min) {
        header += "fn CasMinS32(op_a: u32, op_b: u32) -> u32 { return u32(min(i32(op_a), i32(op_b))); }\n";
    }
    if (info.uses_atomic_s32_max) {
        header += "fn CasMaxS32(op_a: u32, op_b: u32) -> u32 { return u32(max(i32(op_a), i32(op_b))); }\n";
    }
    if (info.uses_global_memory) {
        throw NotImplementedException("WGSL global memory (64-bit addresses)");
    }
    if (info.loads_indexed_attributes) {
        std::string func{"fn IndexedAttrLoad(offset: i32) -> f32 {\n"
                          "  let base_index = offset >> 2u;\n  let masked_index = u32(base_index) & 3u;\n"
                          "  switch (base_index >> 2u) {\n"};
        if (info.loads.AnyComponent(IR::Attribute::PositionX)) {
            func += fmt::format("    case {}: {{ return {}[masked_index]; }}\n",
                                static_cast<u32>(IR::Attribute::PositionX) >> 2, position_name);
        }
        const u32 base_attribute_value = static_cast<u32>(IR::Attribute::Generic0X) >> 2;
        for (u32 index = 0; index < IR::NUM_GENERICS; ++index) {
            if (info.loads.Generic(index)) {
                func += fmt::format("    case {}: {{ return in_attr{}[masked_index]; }}\n",
                                    base_attribute_value + index, index);
            }
        }
        func += "    default: { return 0.0f; }\n  }\n}\n";
        header += func;
    }
}

void EmitContext::DefineConstants() {
    if (info.uses_fswzadd) {
        header += "var<private> FSWZ_A: array<f32, 4> = array<f32, 4>(-1.0f, 1.0f, -1.0f, 0.0f);\n"
                  "var<private> FSWZ_B: array<f32, 4> = array<f32, 4>(-1.0f, -1.0f, 1.0f, -1.0f);\n";
    }
}

} // namespace Shader::Backend::WGSL

namespace Shader::Backend::WGSL {
std::string EmitContext::SsboPtr(u32 binding, std::string_view word) const {
    const std::string w{SsboWord(binding, word)};
    word = w;
    binding = MapSsbo(binding);
    return fmt::format("&{}_ssbo{}[{}]", stage_name, binding, word);
}

std::string EmitContext::SsboLoad(u32 binding, std::string_view word) const {
    const std::string w{SsboWord(binding, word)};
    word = w;
    binding = MapSsbo(binding);
    if (binding < 64 && atomic_ssbo[binding]) {
        return fmt::format("atomicLoad(&{}_ssbo{}[{}])", stage_name, binding, word);
    }
    return fmt::format("{}_ssbo{}[{}]", stage_name, binding, word);
}

std::string EmitContext::SsboStore(u32 binding, std::string_view word, std::string_view value) const {
    const std::string w{SsboWord(binding, word)};
    word = w;
    binding = MapSsbo(binding);
    if (binding < 64 && atomic_ssbo[binding]) {
        return fmt::format("atomicStore(&{}_ssbo{}[{}],{});", stage_name, binding, word, value);
    }
    return fmt::format("{}_ssbo{}[{}]={};", stage_name, binding, word, value);
}

std::string EmitContext::SmemLoad(std::string_view word) const {
    return atomic_smem ? fmt::format("atomicLoad(&smem[{}])", word) : fmt::format("smem[{}]", word);
}

std::string EmitContext::SmemStore(std::string_view word, std::string_view value) const {
    return atomic_smem ? fmt::format("atomicStore(&smem[{}],{});", word, value) : fmt::format("smem[{}]={};", word, value);
}
} // namespace Shader::Backend::WGSL
