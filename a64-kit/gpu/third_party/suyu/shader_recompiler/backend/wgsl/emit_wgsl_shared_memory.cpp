// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
namespace {
// a byte or a short: read-modify-write of its word
void SharedWriteCas(EmitContext& ctx, std::string_view offset, std::string_view value,
                    std::string_view bit_offset, u32 num_bits) {
    const auto word{fmt::format("{}>>2", offset)};
    ctx.Add("{}", ctx.SmemStore(word, fmt::format("insertBits({},{},u32({}),{}u)", ctx.SmemLoad(word), value,
                                                 bit_offset, num_bits)));
}

std::string Word(std::string_view offset, u32 plus = 0) {
    return plus ? fmt::format("({}+{})>>2", offset, plus) : fmt::format("{}>>2", offset);
}
} // Anonymous namespace

void EmitLoadSharedU8(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=extractBits({},u32({}%4)*8u,8u);", inst, ctx.SmemLoad(Word(offset)), offset);
}

void EmitLoadSharedS8(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=u32(extractBits(i32({}),u32({}%4)*8u,8u));", inst, ctx.SmemLoad(Word(offset)), offset);
}

void EmitLoadSharedU16(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=extractBits({},u32(({}>>1)%2)*16u,16u);", inst, ctx.SmemLoad(Word(offset)), offset);
}

void EmitLoadSharedS16(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=u32(extractBits(i32({}),u32(({}>>1)%2)*16u,16u));", inst, ctx.SmemLoad(Word(offset)), offset);
}

void EmitLoadSharedU32(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}={};", inst, ctx.SmemLoad(Word(offset)));
}

void EmitLoadSharedU64(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32x2("{}=vec2<u32>({},{});", inst, ctx.SmemLoad(Word(offset)), ctx.SmemLoad(Word(offset, 4)));
}

void EmitLoadSharedU128(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32x4("{}=vec4<u32>({},{},{},{});", inst, ctx.SmemLoad(Word(offset)), ctx.SmemLoad(Word(offset, 4)),
                 ctx.SmemLoad(Word(offset, 8)), ctx.SmemLoad(Word(offset, 12)));
}

void EmitWriteSharedU8(EmitContext& ctx, std::string_view offset, std::string_view value) {
    const auto bit_offset{fmt::format("i32({}%4)*8", offset)};
    SharedWriteCas(ctx, offset, value, bit_offset, 8);
}

void EmitWriteSharedU16(EmitContext& ctx, std::string_view offset, std::string_view value) {
    const auto bit_offset{fmt::format("i32(({}>>1)%2)*16", offset)};
    SharedWriteCas(ctx, offset, value, bit_offset, 16);
}

void EmitWriteSharedU32(EmitContext& ctx, std::string_view offset, std::string_view value) {
    ctx.Add("{}", ctx.SmemStore(Word(offset), value));
}

void EmitWriteSharedU64(EmitContext& ctx, std::string_view offset, std::string_view value) {
    ctx.Add("{}", ctx.SmemStore(Word(offset), fmt::format("{}.x", value)));
    ctx.Add("{}", ctx.SmemStore(Word(offset, 4), fmt::format("{}.y", value)));
}

void EmitWriteSharedU128(EmitContext& ctx, std::string_view offset, std::string_view value) {
    ctx.Add("{}", ctx.SmemStore(Word(offset), fmt::format("{}.x", value)));
    ctx.Add("{}", ctx.SmemStore(Word(offset, 4), fmt::format("{}.y", value)));
    ctx.Add("{}", ctx.SmemStore(Word(offset, 8), fmt::format("{}.z", value)));
    ctx.Add("{}", ctx.SmemStore(Word(offset, 12), fmt::format("{}.w", value)));
}

} // namespace Shader::Backend::WGSL
