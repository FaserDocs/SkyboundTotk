// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"

namespace Shader::Backend::WGSL {
void EmitBarrier(EmitContext& ctx) {
    ctx.Add("workgroupBarrier();");
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    ctx.Add("workgroupBarrier();");
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    ctx.Add("storageBarrier();");
}
} // namespace Shader::Backend::WGSL
