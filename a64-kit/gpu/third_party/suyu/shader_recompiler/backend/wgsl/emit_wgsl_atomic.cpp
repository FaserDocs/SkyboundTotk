// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
namespace {
// WGSL atomics on atomic<u32> memory (EmitContext::atomic_ssbo / atomic_smem). The result is assigned after the
// operation, as it may share a variable with an operand.
void Native(EmitContext& ctx, IR::Inst& inst, std::string_view ptr, std::string_view function,
            std::string_view value) {
    const std::string op{fmt::format("{}({},{})", function, ptr, value)};
    const auto ret{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("{}={};", ret, op);
}

/// An operation WGSL has no atomic for: a compare-exchange loop over function(old, value) -> new.
void CasLoop(EmitContext& ctx, IR::Inst& inst, std::string_view ptr, std::string_view function,
             std::string_view value) {
    const std::string p{ptr}, v{value};
    const auto ret{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("loop {{let old_=atomicLoad({});let r_=atomicCompareExchangeWeak({},old_,{}(old_,{}));"
            "if (r_.exchanged) {{{}=old_;break;}}}}",
            p, p, function, v, ret);
}

std::string Smem(EmitContext&, std::string_view offset) {
    return fmt::format("&smem[{}>>2]", offset);
}

std::string Ssbo(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset) {
    return ctx.SsboPtr(binding.U32(), fmt::format("{}>>2", ctx.var_alloc.Consume(offset)));
}
} // Anonymous namespace

void EmitSharedAtomicIAdd32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                            std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicAdd", value);
}

void EmitSharedAtomicSMin32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                            std::string_view value) {
    CasLoop(ctx, inst, Smem(ctx, pointer_offset), "CasMinS32", fmt::format("u32({})", value));
}

void EmitSharedAtomicUMin32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                            std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicMin", value);
}

void EmitSharedAtomicSMax32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                            std::string_view value) {
    CasLoop(ctx, inst, Smem(ctx, pointer_offset), "CasMaxS32", fmt::format("u32({})", value));
}

void EmitSharedAtomicUMax32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                            std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicMax", value);
}

void EmitSharedAtomicInc32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                           std::string_view value) {
    CasLoop(ctx, inst, Smem(ctx, pointer_offset), "CasIncrement", value);
}

void EmitSharedAtomicDec32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                           std::string_view value) {
    CasLoop(ctx, inst, Smem(ctx, pointer_offset), "CasDecrement", value);
}

void EmitSharedAtomicAnd32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                           std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicAnd", value);
}

void EmitSharedAtomicOr32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                          std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicOr", value);
}

void EmitSharedAtomicXor32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                           std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicXor", value);
}

void EmitSharedAtomicExchange32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                                std::string_view value) {
    Native(ctx, inst, Smem(ctx, pointer_offset), "atomicExchange", value);
}

void EmitSharedAtomicExchange64(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                                std::string_view value) {
    NotImplemented();
}

void EmitSharedAtomicExchange32x2(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                                  std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicIAdd32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicAdd", value);
}

void EmitStorageAtomicSMin32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasMinS32", fmt::format("u32({})", value));
}

void EmitStorageAtomicUMin32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicMin", value);
}

void EmitStorageAtomicSMax32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasMaxS32", fmt::format("u32({})", value));
}

void EmitStorageAtomicUMax32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicMax", value);
}

void EmitStorageAtomicInc32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasIncrement", value);
}

void EmitStorageAtomicDec32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasDecrement", value);
}

void EmitStorageAtomicAnd32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicAnd", value);
}

void EmitStorageAtomicOr32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicOr", value);
}

void EmitStorageAtomicXor32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicXor", value);
}

void EmitStorageAtomicExchange32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, Ssbo(ctx, binding, offset), "atomicExchange", value);
}

void EmitStorageAtomicIAdd64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicSMin64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicUMin64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicSMax64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicUMax64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicAnd64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                            const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicOr64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                           const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicXor64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                            const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicExchange64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                                 const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicIAdd32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicSMin32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicUMin32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicSMax32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicUMax32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicAnd32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                              const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicOr32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicXor32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                              const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicExchange32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                                   const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicAddF32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    const std::string ptr{Ssbo(ctx, binding, offset)}, v{value};
    const auto ret{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("loop {{let old_=atomicLoad({});let r_=atomicCompareExchangeWeak({},old_,CasFloatAdd(old_,{}));"
            "if (r_.exchanged) {{{}=old_;break;}}}}",
            ptr, ptr, v, ret);
    ctx.AddF32("{}=bitcast<f32>({});", inst, ret);
}

void EmitStorageAtomicAddF16x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasFloatAdd16x2", value);
}

void EmitStorageAtomicAddF32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicMinF16x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasFloatMin16x2", value);
}

void EmitStorageAtomicMinF32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicMaxF16x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    CasLoop(ctx, inst, Ssbo(ctx, binding, offset), "CasFloatMax16x2", value);
}

void EmitStorageAtomicMaxF32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitGlobalAtomicIAdd32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMin32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMin32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMax32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMax32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicInc32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicDec32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAnd32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicOr32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicXor32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicExchange32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicIAdd64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicSMin64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicUMin64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicSMax64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicUMax64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicInc64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicDec64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicAnd64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicOr64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicXor64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicExchange64(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicIAdd32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicSMin32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicUMin32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicSMax32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicUMax32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicInc32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicDec32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicAnd32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicOr32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicXor32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicExchange32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicAddF32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAddF16x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAddF32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicMinF16x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicMinF32x2(EmitContext&) {
    NotImplemented();
}

void EmitGlobalAtomicMaxF16x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicMaxF32x2(EmitContext&) {
    NotImplemented();
}
} // namespace Shader::Backend::WGSL
