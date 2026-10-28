// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
namespace {
// a byte or a short: read-modify-write of its word
void SsboWriteCas(EmitContext& ctx, const IR::Value& binding, std::string_view offset_var,
                  std::string_view value, std::string_view bit_offset, u32 num_bits) {
    const auto word{fmt::format("{}>>2", offset_var)};
    const auto merged{fmt::format("insertBits({},{},u32({}),{}u)", ctx.SsboLoad(binding.U32(), word), value,
                                  bit_offset, num_bits)};
    ctx.Add("{}", ctx.SsboStore(binding.U32(), word, merged));
}

std::string Word(std::string_view offset_var, u32 plus = 0) {
    return plus ? fmt::format("({}+{})>>2", offset_var, plus) : fmt::format("{}>>2", offset_var);
}
} // Anonymous namespace

void EmitLoadGlobalU8(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobalS8(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobalU16(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobalS16(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobal32(EmitContext& ctx, IR::Inst& inst, std::string_view address) {
    if (ctx.profile.support_int64) {
        return ctx.AddU32("{}=LoadGlobal32({});", inst, address);
    }
    LOG_WARNING(Shader_WGSL, "Int64 not supported, ignoring memory operation");
    ctx.AddU32("{}=0u;", inst);
}

void EmitLoadGlobal64(EmitContext& ctx, IR::Inst& inst, std::string_view address) {
    if (ctx.profile.support_int64) {
        return ctx.AddU32x2("{}=LoadGlobal64({});", inst, address);
    }
    LOG_WARNING(Shader_WGSL, "Int64 not supported, ignoring memory operation");
    ctx.AddU32x2("{}=vec2<u32>(0);", inst);
}

void EmitLoadGlobal128(EmitContext& ctx, IR::Inst& inst, std::string_view address) {
    if (ctx.profile.support_int64) {
        return ctx.AddU32x4("{}=LoadGlobal128({});", inst, address);
    }
    LOG_WARNING(Shader_WGSL, "Int64 not supported, ignoring memory operation");
    ctx.AddU32x4("{}=vec4<u32>(0);", inst);
}

void EmitWriteGlobalU8(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobalS8(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobalU16(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobalS16(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobal32(EmitContext& ctx, std::string_view address, std::string_view value) {
    if (ctx.profile.support_int64) {
        return ctx.Add("WriteGlobal32({},{});", address, value);
    }
    LOG_WARNING(Shader_WGSL, "Int64 not supported, ignoring memory operation");
}

void EmitWriteGlobal64(EmitContext& ctx, std::string_view address, std::string_view value) {
    if (ctx.profile.support_int64) {
        return ctx.Add("WriteGlobal64({},{});", address, value);
    }
    LOG_WARNING(Shader_WGSL, "Int64 not supported, ignoring memory operation");
}

void EmitWriteGlobal128(EmitContext& ctx, std::string_view address, std::string_view value) {
    if (ctx.profile.support_int64) {
        return ctx.Add("WriteGlobal128({},{});", address, value);
    }
    LOG_WARNING(Shader_WGSL, "Int64 not supported, ignoring memory operation");
}

void EmitLoadStorageU8(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=extractBits({},u32({}%4)*8u,8u);", inst, ctx.SsboLoad(binding.U32(), Word(offset_var)),
               offset_var);
}

void EmitLoadStorageS8(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=u32(extractBits(i32({}),u32({}%4)*8u,8u));", inst, ctx.SsboLoad(binding.U32(), Word(offset_var)),
               offset_var);
}

void EmitLoadStorageU16(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                        const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=extractBits({},u32(({}>>1)%2)*16u,16u);", inst, ctx.SsboLoad(binding.U32(), Word(offset_var)),
               offset_var);
}

void EmitLoadStorageS16(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                        const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=u32(extractBits(i32({}),u32(({}>>1)%2)*16u,16u));", inst,
               ctx.SsboLoad(binding.U32(), Word(offset_var)), offset_var);
}

void EmitLoadStorage32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}={};", inst, ctx.SsboLoad(binding.U32(), Word(offset_var)));
}

void EmitLoadStorage64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32x2("{}=vec2<u32>({},{});", inst, ctx.SsboLoad(binding.U32(), Word(offset_var)),
                 ctx.SsboLoad(binding.U32(), Word(offset_var, 4)));
}

void EmitLoadStorage128(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                        const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32x4("{}=vec4<u32>({},{},{},{});", inst, ctx.SsboLoad(binding.U32(), Word(offset_var)),
                 ctx.SsboLoad(binding.U32(), Word(offset_var, 4)), ctx.SsboLoad(binding.U32(), Word(offset_var, 8)),
                 ctx.SsboLoad(binding.U32(), Word(offset_var, 12)));
}

void EmitWriteStorageU8(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    const auto bit_offset{fmt::format("i32({}%4)*8", offset_var)};
    SsboWriteCas(ctx, binding, offset_var, value, bit_offset, 8);
}

void EmitWriteStorageS8(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    const auto bit_offset{fmt::format("i32({}%4)*8", offset_var)};
    SsboWriteCas(ctx, binding, offset_var, value, bit_offset, 8);
}

void EmitWriteStorageU16(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                         std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    const auto bit_offset{fmt::format("i32(({}>>1)%2)*16", offset_var)};
    SsboWriteCas(ctx, binding, offset_var, value, bit_offset, 16);
}

void EmitWriteStorageS16(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                         std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    const auto bit_offset{fmt::format("i32(({}>>1)%2)*16", offset_var)};
    SsboWriteCas(ctx, binding, offset_var, value, bit_offset, 16);
}

void EmitWriteStorage32(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var), value));
}

void EmitWriteStorage64(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var), fmt::format("{}.x", value)));
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var, 4), fmt::format("{}.y", value)));
}

void EmitWriteStorage128(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                         std::string_view value) {
    if (ctx.stage == Stage::VertexA || ctx.stage == Stage::VertexB) {
        return;   // WebGPU has no writable storage in vertex shaders
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var), fmt::format("{}.x", value)));
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var, 4), fmt::format("{}.y", value)));
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var, 8), fmt::format("{}.z", value)));
    ctx.Add("{}", ctx.SsboStore(binding.U32(), Word(offset_var, 12), fmt::format("{}.w", value)));
}
} // namespace Shader::Backend::WGSL
