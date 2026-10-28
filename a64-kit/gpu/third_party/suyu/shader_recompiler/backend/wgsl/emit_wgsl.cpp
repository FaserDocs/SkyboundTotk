// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>
#include <type_traits>

#include "common/div_ceil.h"
#include "common/settings.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"

namespace Shader::Backend::WGSL {
namespace {
template <class Func>
struct FuncTraits {};

template <class ReturnType_, class... Args>
struct FuncTraits<ReturnType_ (*)(Args...)> {
    using ReturnType = ReturnType_;

    static constexpr size_t NUM_ARGS = sizeof...(Args);

    template <size_t I>
    using ArgType = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <auto func, typename... Args>
void SetDefinition(EmitContext& ctx, IR::Inst* inst, Args... args) {
    inst->SetDefinition<Id>(func(ctx, std::forward<Args>(args)...));
}

template <typename ArgType>
auto Arg(EmitContext& ctx, const IR::Value& arg) {
    if constexpr (std::is_same_v<ArgType, std::string_view>) {
        return ctx.var_alloc.Consume(arg);
    } else if constexpr (std::is_same_v<ArgType, const IR::Value&>) {
        return arg;
    } else if constexpr (std::is_same_v<ArgType, u32>) {
        return arg.U32();
    } else if constexpr (std::is_same_v<ArgType, IR::Attribute>) {
        return arg.Attribute();
    } else if constexpr (std::is_same_v<ArgType, IR::Patch>) {
        return arg.Patch();
    } else if constexpr (std::is_same_v<ArgType, IR::Reg>) {
        return arg.Reg();
    }
}

template <auto func, bool is_first_arg_inst, size_t... I>
void Invoke(EmitContext& ctx, IR::Inst* inst, std::index_sequence<I...>) {
    using Traits = FuncTraits<decltype(func)>;
    constexpr size_t off = is_first_arg_inst ? 2 : 1;
    constexpr size_t N = sizeof...(I);
    // The operands are consumed last to first, in a defined order: Consume has side effects (a value's last use
    // releases its variable for reuse) and this backend was written and checked where function arguments are
    // evaluated right to left (the MSVC ABI). Compiled for wasm, which evaluates them left to right, comparisons
    // came out as "x > x" and constants changed. (A braced list is evaluated in order.)
    std::tuple<decltype(Arg<typename Traits::template ArgType<N - 1 - I + off>>(ctx, inst->Arg(N - 1 - I)))...> rev{
        Arg<typename Traits::template ArgType<N - 1 - I + off>>(ctx, inst->Arg(N - 1 - I))...};
    (void)rev;
    if constexpr (std::is_same_v<typename Traits::ReturnType, Id>) {
        if constexpr (is_first_arg_inst) {
            SetDefinition<func>(ctx, inst, *inst, std::get<N - 1 - I>(rev)...);
        } else {
            SetDefinition<func>(ctx, inst, std::get<N - 1 - I>(rev)...);
        }
    } else {
        if constexpr (is_first_arg_inst) {
            func(ctx, *inst, std::get<N - 1 - I>(rev)...);
        } else {
            func(ctx, std::get<N - 1 - I>(rev)...);
        }
    }
}

template <auto func>
void Invoke(EmitContext& ctx, IR::Inst* inst) {
    using Traits = FuncTraits<decltype(func)>;
    static_assert(Traits::NUM_ARGS >= 1, "Insufficient arguments");
    if constexpr (Traits::NUM_ARGS == 1) {
        Invoke<func, false>(ctx, inst, std::make_index_sequence<0>{});
    } else {
        using FirstArgType = typename Traits::template ArgType<1>;
        static constexpr bool is_first_arg_inst = std::is_same_v<FirstArgType, IR::Inst&>;
        using Indices = std::make_index_sequence<Traits::NUM_ARGS - (is_first_arg_inst ? 2 : 1)>;
        Invoke<func, is_first_arg_inst>(ctx, inst, Indices{});
    }
}

void EmitInst(EmitContext& ctx, IR::Inst* inst) {
    switch (inst->GetOpcode()) {
#define OPCODE(name, result_type, ...)                                                             \
    case IR::Opcode::name:                                                                         \
        return Invoke<&Emit##name>(ctx, inst);
#include "shader_recompiler/frontend/ir/opcodes.inc"
#undef OPCODE
    }
    throw LogicError("Invalid opcode {}", inst->GetOpcode());
}

bool IsReference(IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::Reference;
}

void PrecolorInst(IR::Inst& phi) {
    // Insert phi moves before references to avoid overwriting other phis
    const size_t num_args{phi.NumArgs()};
    for (size_t i = 0; i < num_args; ++i) {
        IR::Block& phi_block{*phi.PhiBlock(i)};
        auto it{std::find_if_not(phi_block.rbegin(), phi_block.rend(), IsReference).base()};
        IR::IREmitter ir{phi_block, it};
        const IR::Value arg{phi.Arg(i)};
        if (arg.IsImmediate()) {
            ir.PhiMove(phi, arg);
        } else {
            ir.PhiMove(phi, IR::Value{arg.InstRecursive()});
        }
    }
    for (size_t i = 0; i < num_args; ++i) {
        IR::IREmitter{*phi.PhiBlock(i)}.Reference(IR::Value{&phi});
    }
}

void Precolor(const IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& phi : block->Instructions()) {
            if (!IR::IsPhi(phi)) {
                break;
            }
            PrecolorInst(phi);
        }
    }
}

void EmitCode(EmitContext& ctx, const IR::Program& program) {
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Block:
            for (IR::Inst& inst : node.data.block->Instructions()) {
                EmitInst(ctx, &inst);
            }
            break;
        case IR::AbstractSyntaxNode::Type::If:
            ctx.Add("if ({}) {{", ctx.var_alloc.Consume(node.data.if_node.cond));
            break;
        case IR::AbstractSyntaxNode::Type::EndIf:
            ctx.Add("}}");
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (node.data.break_node.cond.IsImmediate()) {
                if (node.data.break_node.cond.U1()) {
                    ctx.Add("break;");
                }
            } else {
                ctx.Add("if ({}) {{ break; }}", ctx.var_alloc.Consume(node.data.break_node.cond));
            }
            break;
        case IR::AbstractSyntaxNode::Type::Return:
        case IR::AbstractSyntaxNode::Type::Unreachable:
            ctx.Add("return;");
            break;
        case IR::AbstractSyntaxNode::Type::Loop:
            ctx.Add("loop {{");
            break;
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (Settings::values.disable_shader_loop_safety_checks) {
                ctx.Add("if (!{}) {{ break; }} }}", ctx.var_alloc.Consume(node.data.repeat.cond));
            } else {
                const u32 n{ctx.num_safety_loop_vars++};
                ctx.Add("loop{} = loop{} - 1; if (loop{} < 0 || !{}) {{ break; }} }}", n, n, n,
                        ctx.var_alloc.Consume(node.data.repeat.cond));
            }
            break;
        default:
            throw NotImplementedException("AbstractSyntaxNode Type {}", node.type);
        }
    }
}

void DefineVariables(const EmitContext& ctx, std::string& header) {
    for (u32 i = 0; i < static_cast<u32>(WgslVarType::Void); ++i) {
        const auto type{static_cast<WgslVarType>(i)};
        const auto& tracker{ctx.var_alloc.GetUseTracker(type)};
        if (!tracker.uses_temp && tracker.num_used == 0) {
            continue;
        }
        const auto type_name{ctx.var_alloc.GetWgslType(type)};
        // Temps/return types that are never used are stored at index 0
        if (tracker.uses_temp) {
            header += fmt::format("var t{}: {};\n", ctx.var_alloc.Representation(0, type), type_name);
        }
        for (u32 index = 0; index < tracker.num_used; ++index) {
            header += fmt::format("var {}: {};\n", ctx.var_alloc.Representation(index, type), type_name);
        }
    }
    for (u32 i = 0; i < ctx.num_safety_loop_vars; ++i) {
        header += fmt::format("var loop{}: i32 = 0x2000;\n", i);
    }
}

std::string_view AttributeTypeName(AttributeType type) {
    switch (type) {
    case AttributeType::SignedInt:
    case AttributeType::SignedScaled:
        return "vec4<i32>";
    case AttributeType::UnsignedInt:
    case AttributeType::UnsignedScaled:
        return "vec4<u32>";
    default:
        return "vec4<f32>";
    }
}

bool IsIntegerAttribute(AttributeType type) {
    return type == AttributeType::SignedInt || type == AttributeType::UnsignedInt;
}

std::string_view InterpolationAttribute(Interpolation interp) {
    switch (interp) {
    case Interpolation::Flat:
        return "@interpolate(flat) ";
    case Interpolation::NoPerspective:
        return "@interpolate(linear) ";
    default:
        return "";
    }
}

/// A pulled vertex attribute (Options::VertexPull): its bytes read from the stream's storage binding and converted
/// as the vertex fetch would (Maxwell's attribute format word: offset 7-20, size 21-26, type 27-29, bgra 31).
std::string PulledAttribute(const EmitContext& ctx, size_t i, const Options::VertexPull& p) {
    const u32 a{p.attribute}, offset{(a >> 7) & 0x3fff}, size{(a >> 21) & 0x3f}, type{(a >> 27) & 7};
    const bool bgra{(a >> 31) != 0};
    int bits[4]{};
    switch (size) {
    case 0x01: bits[0] = bits[1] = bits[2] = bits[3] = 32; break;
    case 0x02: bits[0] = bits[1] = bits[2] = 32; break;
    case 0x03: bits[0] = bits[1] = bits[2] = bits[3] = 16; break;
    case 0x04: bits[0] = bits[1] = 32; break;
    case 0x05: bits[0] = bits[1] = bits[2] = 16; break;
    case 0x0A: case 0x33: bits[0] = bits[1] = bits[2] = bits[3] = 8; break;
    case 0x0F: bits[0] = bits[1] = 16; break;
    case 0x12: bits[0] = 32; break;
    case 0x13: bits[0] = bits[1] = bits[2] = 8; break;
    case 0x18: case 0x32: bits[0] = bits[1] = 8; break;
    case 0x1B: bits[0] = 16; break;
    case 0x1D: case 0x34: bits[0] = 8; break;
    default: break;   // packed (0x30, 0x31): below
    }
    const char k{"xyz"[p.stream % 3]};
    const auto element{p.instanced ? std::string{"(input.instance_index + vs_pull.w)"}
                                    : std::string{"u32(i32(input.vertex_index) + bitcast<i32>(rescale_(3u).z))"}};
    const auto base{fmt::format("(vs_pull.{} + {} * {}u + {}u)", k, element, p.stride, offset)};
    const bool integer{type == 3 || type == 4}, scaled{type == 5 || type == 6};
    const bool is_signed{type == 1 || type == 3 || type == 6};
    std::string out{fmt::format("  {{\n    let o = {};\n", base)};
    std::string comps[4];
    if (size == 0x30 || size == 0x31) {
        out += fmt::format("    let w = vb{}_32(o);\n", p.stream);
        if (size == 0x31) {
            out += fmt::format("    in_attr{} = vec4<f32>(uf11_(w & 0x7ffu), uf11_((w >> 11u) & 0x7ffu), uf10_(w >> 22u), 1.0);\n  }}\n", i);
            return out;
        }
        const char* u{"vec4<u32>(w & 0x3ffu, (w >> 10u) & 0x3ffu, (w >> 20u) & 0x3ffu, w >> 30u)"};
        const char* s{"vec4<i32>(i32(w << 22u) >> 22u, i32(w << 12u) >> 22u, i32(w << 2u) >> 22u, i32(w) >> 30u)"};
        std::string v;
        switch (type) {
        case 1: v = fmt::format("max(vec4<f32>({}) / vec4<f32>(511.0, 511.0, 511.0, 1.0), vec4<f32>(-1.0))", s); break;
        case 2: v = fmt::format("vec4<f32>({}) / vec4<f32>(1023.0, 1023.0, 1023.0, 3.0)", u); break;
        case 3: v = fmt::format("bitcast<vec4<f32>>({})", s); break;
        case 4: v = fmt::format("bitcast<vec4<f32>>({})", u); break;
        case 5: v = fmt::format("vec4<f32>({})", u); break;
        case 6: v = fmt::format("vec4<f32>({})", s); break;
        default: v = "vec4<f32>(0.0, 0.0, 0.0, 1.0)"; break;
        }
        out += fmt::format("    in_attr{} = {};\n  }}\n", i, v);
        return out;
    }
    u32 at{0};
    for (int c = 0; c < 4; ++c) {
        if (!bits[c]) {
            comps[c] = integer ? (c == 3 ? "1u" : "0u") : (c == 3 ? "1.0" : "0.0");
            continue;
        }
        const int n{bits[c]};
        const auto raw{fmt::format("vb{}_{}(o + {}u)", p.stream, n, at)};
        at += n / 8;
        const auto sext{n == 32 ? fmt::format("i32({})", raw)
                                : fmt::format("(i32({} << {}u) >> {}u)", raw, 32 - n, 32 - n)};
        if (integer) {
            comps[c] = is_signed ? fmt::format("u32({})", sext) : raw;
        } else if (scaled) {
            comps[c] = is_signed ? fmt::format("f32({})", sext) : fmt::format("f32({})", raw);
        } else if (type == 7) {   // float
            comps[c] = n == 32 ? fmt::format("bitcast<f32>({})", raw)
                     : n == 16 ? fmt::format("unpack2x16float({}).x", raw)
                               : fmt::format("f32({})", raw);
        } else if (type == 2) {   // unorm
            comps[c] = fmt::format("(f32({}) / {}.0)", raw, n == 32 ? 4294967295.0 : (double)((1u << n) - 1));
        } else {                  // snorm
            comps[c] = fmt::format("max(f32({}) / {}.0, -1.0)", sext, (1u << (n - 1)) - 1);
        }
    }
    if (bgra) std::swap(comps[0], comps[2]);
    if (size == 0x32) std::swap(comps[0], comps[1]);
    out += integer ? fmt::format("    in_attr{} = bitcast<vec4<f32>>(vec4<u32>({}, {}, {}, {}));\n  }}\n", i, comps[0],
                                 comps[1], comps[2], comps[3])
                   : fmt::format("    in_attr{} = vec4<f32>({}, {}, {}, {});\n  }}\n", i, comps[0], comps[1], comps[2],
                                 comps[3]);
    return out;
}

/// The WGSL entry point: copies the stage's inputs into the private variables the body reads, runs it, and returns
/// the private variables it wrote as the stage's outputs.
std::string EntryPoint(const EmitContext& ctx, const IR::Program& program, const Options& options) {
    const Info& info{ctx.info};
    std::string out;
    switch (ctx.stage) {
    case Stage::VertexA:
    case Stage::VertexB: {
        out += "struct VsIn {\n  @builtin(vertex_index) vertex_index: u32,\n"
               "  @builtin(instance_index) instance_index: u32,\n";
        for (size_t i = 0; i < IR::NUM_GENERICS; ++i) {
            if (info.loads.Generic(i) && !options.vertex_pull[i].enabled) {
                out += fmt::format("  @location({}) a{}: {},\n", i, i,
                                   options.vertex_unpack[i] != VertexUnpack::None
                                       ? std::string_view{"u32"}
                                       : AttributeTypeName(ctx.runtime_info.generic_input_types[i]));
            }
        }
        out += "};\nstruct VsOut {\n  @builtin(position) position: vec4<f32>,\n";
        for (size_t i = 0; i < IR::NUM_GENERICS; ++i) {
            if (info.stores.Generic(i)) {
                out += fmt::format("  @location({}) {}o{}: vec4<f32>,\n", i,
                                   InterpolationAttribute(options.output_interpolation[i]), i);
            }
        }
        // unsigned 11/10-bit floats (5-bit exponent, no sign)
        out += "};\nfn uf11_(v: u32) -> f32 { let e = v >> 6u; let m = f32(v & 63u);"
               " if (e == 0u) { return m * exp2(-20.0); } if (e == 31u) { return 65504.0; }"
               " return (1.0 + m / 64.0) * exp2(f32(e) - 15.0); }\n"
               "fn uf10_(v: u32) -> f32 { let e = v >> 5u; let m = f32(v & 31u);"
               " if (e == 0u) { return m * exp2(-19.0); } if (e == 31u) { return 65504.0; }"
               " return (1.0 + m / 32.0) * exp2(f32(e) - 15.0); }\n"
               "@vertex\nfn main(input: VsIn) -> VsOut {\n"
               "  gl_VertexID = i32(input.vertex_index) + bitcast<i32>(rescale_(3u).z);\n"
               "  gl_InstanceID = i32(input.instance_index) + bitcast<i32>(rescale_(3u).w);\n";
        for (size_t i = 0; i < IR::NUM_GENERICS; ++i) {
            if (info.loads.Generic(i) && options.vertex_pull[i].enabled) {
                out += PulledAttribute(ctx, i, options.vertex_pull[i]);
            } else if (info.loads.Generic(i) && options.vertex_unpack[i] != VertexUnpack::None) {
                // fields of a packed word: x = bits 0-9, y 10-19, z 20-29, w 30-31 (10/11-bit floats: 0-10,
                // 11-21, 22-31)
                const auto v{fmt::format("input.a{}", i)};
                const auto u{fmt::format("vec4<u32>({0}&0x3ffu,({0}>>10u)&0x3ffu,({0}>>20u)&0x3ffu,{0}>>30u)", v)};
                const auto s{fmt::format("vec4<i32>(i32({0}<<22u)>>22u,i32({0}<<12u)>>22u,i32({0}<<2u)>>22u,"
                                         "i32({0})>>30u)", v)};
                std::string value;
                switch (options.vertex_unpack[i]) {
                case VertexUnpack::A2B10G10R10_SNORM:
                    value = fmt::format("max(vec4<f32>({})/vec4<f32>(511.0,511.0,511.0,1.0),vec4<f32>(-1.0))", s);
                    break;
                case VertexUnpack::A2B10G10R10_UINT:
                    value = fmt::format("bitcast<vec4<f32>>({})", u);
                    break;
                case VertexUnpack::A2B10G10R10_SINT:
                    value = fmt::format("bitcast<vec4<f32>>({})", s);
                    break;
                case VertexUnpack::A2B10G10R10_USCALED:
                    value = fmt::format("vec4<f32>({})", u);
                    break;
                case VertexUnpack::A2B10G10R10_SSCALED:
                    value = fmt::format("vec4<f32>({})", s);
                    break;
                case VertexUnpack::B10G11R11_FLOAT:
                    value = fmt::format("vec4<f32>(uf11_({0}&0x7ffu),uf11_(({0}>>11u)&0x7ffu),uf10_({0}>>22u),1.0)", v);
                    break;
                default:
                    break;
                }
                out += fmt::format("  in_attr{} = {};\n", i, value);
            } else if (info.loads.Generic(i)) {
                const AttributeType type{ctx.runtime_info.generic_input_types[i]};
                out += IsIntegerAttribute(type) ? fmt::format("  in_attr{} = bitcast<vec4<f32>>(input.a{});\n", i, i)
                       : type == AttributeType::SignedScaled || type == AttributeType::UnsignedScaled
                           ? fmt::format("  in_attr{} = vec4<f32>(input.a{});\n", i, i)
                           : fmt::format("  in_attr{} = input.a{};\n", i, i);
            }
        }
        out += "  gl_Position = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);\n  shader_main();\n  var output: VsOut;\n"
               "  let p = gl_Position;\n  let t = rescale_(2u);\n  let z = rescale_(3u);\n"
               "  output.position = vec4<f32>(t.x * p.x + t.y * p.w, t.z * p.y + t.w * p.w, z.x * p.z + z.y * p.w,"
               " p.w);\n"
               // a 16-bit depth target: the depth on its step, as pixel shaders write it (d16q_); vertices outside
               // the depth range keep theirs (the visible part of the primitive keeps its slope)
               "  if (render_area().z != 0.0f && p.w > 0.0f) {\n"
               "    let d = output.position.z / p.w;\n"
               "    if (d >= 0.0f && d <= 1.0f) { output.position.z = d16q_(d) * p.w; }\n"
               "  }\n";
        // RX_DEBUG_VS_NONFINITE=1 (debugging): a position with an infinite or NaN component goes to the screen's
        // centre; =2: all of it NaN (a bit test: compilers may fold x != x)
        if (static const char* nonfinite = std::getenv("RX_DEBUG_VS_NONFINITE"); nonfinite)
            out += fmt::format("  if (any((bitcast<vec4<u32>>(p) & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u))) {{"
                               " output.position = {}; }}\n",
                               *nonfinite == '2' ? "bitcast<vec4<f32>>(vec4<u32>(0x7fc00000u))"
                                                 : "vec4<f32>(0.0f, 0.0f, 0.5f, 1.0f)");
        for (size_t i = 0; i < IR::NUM_GENERICS; ++i) {
            if (info.stores.Generic(i)) {
                out += fmt::format("  output.o{} = out_attr{};\n", i, i);
            }
        }
        out += "  return output;\n}\n";
        break;
    }
    case Stage::Fragment: {
        out += "struct FsIn {\n  @builtin(position) frag_coord: vec4<f32>,\n"
               "  @builtin(front_facing) front_facing: bool,\n";
        if (info.uses_sample_id) {
            out += "  @builtin(sample_index) sample_index: u32,\n";
        }
        for (size_t i = 0; i < IR::NUM_GENERICS; ++i) {
            if (info.loads.Generic(i) && ctx.runtime_info.previous_stage_stores.Generic(i)) {
                out += fmt::format("  @location({}) {}a{}: vec4<f32>,\n", i,
                                   InterpolationAttribute(info.interpolation[i]), i);
            }
        }
        out += "};\n";
        bool any_output{info.stores_frag_depth};
        std::string outs{"struct FsOut {\n"};
        for (size_t i = 0; i < info.stores_frag_color.size(); ++i) {
            if (info.stores_frag_color[i]) {
                outs += fmt::format("  @location({}) c{}: {},\n", i, i,
                                    AttributeTypeName(ctx.runtime_info.color_output_types[i]));
                any_output = true;
            }
        }
        if (info.stores_frag_depth) {
            outs += "  @builtin(frag_depth) depth: f32,\n";
        }
        outs += "};\n";
        if (any_output) {
            out += outs;
        }
        // (force_early_z: depth and stencil tests before the shader runs, so the fragments they reject make no stores:
        // the GPU's early-Z for a program that asks for it)
        const char* early = "";
        if (ctx.runtime_info.force_early_z && !info.stores_frag_depth) {
            static const char* attr = getenv("RX_EARLY_Z_ATTR") ? getenv("RX_EARLY_Z_ATTR") : "@early_depth_test(force)";
            early = attr;
        }
        out += fmt::format("@fragment {}\nfn main(input: FsIn){} {{\n", early, any_output ? " -> FsOut" : "");
        out += "  gl_FragCoord = input.frag_coord;\n  gl_FrontFacing = input.front_facing;\n";
        if (info.uses_sample_id) {
            out += "  gl_SampleID = i32(input.sample_index);\n";
        }
        for (size_t i = 0; i < IR::NUM_GENERICS; ++i) {
            if (info.loads.Generic(i) && ctx.runtime_info.previous_stage_stores.Generic(i)) {
                out += fmt::format("  in_attr{} = input.a{};\n", i, i);
            }
        }
        out += "  gl_FragDepth = input.frag_coord.z;\n  shader_main();\n";
        if (any_output) {
            out += "  var output: FsOut;\n";
            // render_area().w: bit i = colour target i has a UNORM format kept as float (16-bit UNORM targets
            // cannot be render attachments in WebGPU), bit 8 + i the same for SNORM: clamped as the hardware's
            // store would
            out += "  let ntc_ = u32(render_area().w);\n";
            for (size_t i = 0; i < info.stores_frag_color.size(); ++i) {
                if (info.stores_frag_color[i]) {
                    const auto type{ctx.runtime_info.color_output_types[i]};
                    out += IsIntegerAttribute(type)
                               ? fmt::format("  output.c{} = bitcast<{}>(frag_color{});\n", i, AttributeTypeName(type), i)
                               : fmt::format("  output.c{0} = frag_color{0};\n"
                                             "  if ((ntc_ & {1}u) != 0u) {{ output.c{0} = clamp(output.c{0}, vec4<f32>(0.0), vec4<f32>(1.0)); }}\n"
                                             "  if ((ntc_ & {2}u) != 0u) {{ output.c{0} = clamp(output.c{0}, vec4<f32>(-1.0), vec4<f32>(1.0)); }}\n",
                                             i, 1u << i, 0x100u << i);
                }
            }
            if (info.stores_frag_depth) {
                out += "  output.depth = select(gl_FragDepth, d16q_(gl_FragDepth), render_area().z != 0.0f);\n";
            }
            out += "  return output;\n";
        }
        out += "}\n";
        break;
    }
    case Stage::Compute:
        out += fmt::format("@compute @workgroup_size({}, {}, {})\n"
                           "fn main(@builtin(local_invocation_id) lid: vec3<u32>, "
                           "@builtin(workgroup_id) wid: vec3<u32>) {{\n"
                           "  gl_LocalInvocationID = lid;\n  gl_WorkGroupID = wid;\n  shader_main();\n}}\n",
                           (std::max)(program.workgroup_size[0], 1u), (std::max)(program.workgroup_size[1], 1u),
                           (std::max)(program.workgroup_size[2], 1u));
        break;
    default:
        throw NotImplementedException("WebGPU has no {} shaders", ctx.stage);
    }
    return out;
}
} // Anonymous namespace

std::string EmitWGSL(const Profile& profile, const RuntimeInfo& runtime_info, IR::Program& program,
                     Bindings& bindings, const Options& options) {
    EmitContext ctx{program, bindings, profile, runtime_info};
    u32 pulled_streams{0};
    for (const auto& p : options.vertex_pull) {
        if (p.enabled) {
            pulled_streams = (std::max)(pulled_streams, static_cast<u32>(p.stream) + 1);
        }
    }
    if (pulled_streams > 0 && (program.stage == Stage::VertexA || program.stage == Stage::VertexB)) {
        ctx.header += fmt::format("@group({}) @binding({}) var<uniform> vs_pull: vec4<u32>;\n", ctx.group,
                                  bindings.unified++);
        for (u32 k = 0; k < pulled_streams; ++k) {
            ctx.header += fmt::format(
                "@group({0}) @binding({1}) var<storage, read> vs_vb{2}: array<u32>;\n"
                "fn vb{2}_8(o: u32) -> u32 {{ return (vs_vb{2}[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }}\n"
                "fn vb{2}_16(o: u32) -> u32 {{ return vb{2}_8(o) | (vb{2}_8(o + 1u) << 8u); }}\n"
                "fn vb{2}_32(o: u32) -> u32 {{ return vb{2}_16(o) | (vb{2}_16(o + 2u) << 16u); }}\n",
                ctx.group, bindings.unified++, k);
        }
    }
    Precolor(program);
    EmitCode(ctx, program);
    if (program.shared_memory_size > 0) {
        ctx.header += fmt::format("var<workgroup> smem: array<{}, {}>;\n", ctx.atomic_smem ? "atomic<u32>" : "u32",
                                  Common::DivCeil(program.shared_memory_size, 4U));
    }
    std::string body{"fn shader_main() {\n"};
    if (program.local_memory_size > 0) {
        body += fmt::format("var lmem: array<u32, {}>;\n", Common::DivCeil(program.local_memory_size, 4U));
    }
    DefineVariables(ctx, body);
    if (ctx.uses_cc_carry) {
        body += "var carry: u32;\n";
    }
    if (program.info.uses_subgroup_shuffles) {
        body += "var shfl_in_bounds: bool;\nvar shfl_result: u32;\n";
    }
    return ctx.header + body + ctx.code + "}\n" + EntryPoint(ctx, program, options);
}

} // namespace Shader::Backend::WGSL
