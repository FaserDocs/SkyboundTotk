// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
namespace {
void Alias(IR::Inst& inst, const IR::Value& value) {
    if (value.IsImmediate()) {
        return;
    }
    IR::Inst& value_inst{*value.InstRecursive()};
    value_inst.DestructiveAddUsage(inst.UseCount());
    value_inst.DestructiveRemoveUsage();
    inst.SetDefinition(value_inst.Definition<Id>());
}
} // Anonymous namespace

void EmitIdentity(EmitContext&, IR::Inst& inst, const IR::Value& value) {
    Alias(inst, value);
}

void EmitConditionRef(EmitContext& ctx, IR::Inst& inst, const IR::Value& value) {
    // Fake one usage to get a real variable out of the condition
    inst.DestructiveAddUsage(1);
    const auto ret{ctx.var_alloc.Define(inst, WgslVarType::U1)};
    const auto input{ctx.var_alloc.Consume(value)};
    const auto suffix{ctx.profile.has_gl_bool_ref_bug ? "?true:false" : ""};
    if (ret != input) {
        ctx.Add("{}={}{};", ret, input, suffix);
    }
}

void EmitBitCastU16F16([[maybe_unused]] EmitContext& ctx, [[maybe_unused]] IR::Inst& inst) {
    NotImplemented();
}

void EmitBitCastU32F32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=bitcast<u32>({});", inst, value);
}

void EmitBitCastU64F64(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    NotImplemented();
}

void EmitBitCastF16U16([[maybe_unused]] EmitContext& ctx, [[maybe_unused]] IR::Inst& inst) {
    NotImplemented();
}

void EmitBitCastF32U32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddF32("{}=bitcast<f32>({});", inst, value);
}

void EmitBitCastF64U64(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    NotImplemented();
}

void EmitPackUint2x32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU64("{}={};", inst, value);   // u64 is vec2<u32>
}

void EmitUnpackUint2x32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32x2("{}={};", inst, value);
}

void EmitPackFloat2x16(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=pack2x16float({});", inst, value);   // f16x2 is vec2<f32>
}

void EmitUnpackFloat2x16(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddF16x2("{}=unpack2x16float({});", inst, value);
}

void EmitPackHalf2x16(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=pack2x16float({});", inst, value);
}

void EmitUnpackHalf2x16(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddF32x2("{}=unpack2x16float({});", inst, value);
}

void EmitPackDouble2x32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    NotImplemented();
}

void EmitUnpackDouble2x32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    NotImplemented();
}

} // namespace Shader::Backend::WGSL
