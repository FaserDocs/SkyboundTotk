// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
void EmitSelectU1(EmitContext& ctx, IR::Inst& inst, std::string_view cond,
                  std::string_view true_value, std::string_view false_value) {
    ctx.AddU1("{}=select({},{},{});", inst, false_value, true_value, cond);
}

void EmitSelectU8([[maybe_unused]] EmitContext& ctx, [[maybe_unused]] std::string_view cond,
                  [[maybe_unused]] std::string_view true_value,
                  [[maybe_unused]] std::string_view false_value) {
    NotImplemented();
}

void EmitSelectU16([[maybe_unused]] EmitContext& ctx, [[maybe_unused]] std::string_view cond,
                   [[maybe_unused]] std::string_view true_value,
                   [[maybe_unused]] std::string_view false_value) {
    NotImplemented();
}

void EmitSelectU32(EmitContext& ctx, IR::Inst& inst, std::string_view cond,
                   std::string_view true_value, std::string_view false_value) {
    ctx.AddU32("{}=select({},{},{});", inst, false_value, true_value, cond);
}

void EmitSelectU64(EmitContext& ctx, IR::Inst& inst, std::string_view cond,
                   std::string_view true_value, std::string_view false_value) {
    ctx.AddU64("{}=select({},{},{});", inst, false_value, true_value, cond);
}

void EmitSelectF16([[maybe_unused]] EmitContext& ctx, [[maybe_unused]] std::string_view cond,
                   [[maybe_unused]] std::string_view true_value,
                   [[maybe_unused]] std::string_view false_value) {
    NotImplemented();
}

void EmitSelectF32(EmitContext& ctx, IR::Inst& inst, std::string_view cond,
                   std::string_view true_value, std::string_view false_value) {
    ctx.AddF32("{}=select({},{},{});", inst, false_value, true_value, cond);
}

void EmitSelectF64(EmitContext& ctx, IR::Inst& inst, std::string_view cond,
                   std::string_view true_value, std::string_view false_value) {
    ctx.AddF64("{}=select({},{},{});", inst, false_value, true_value, cond);
}

} // namespace Shader::Backend::WGSL
