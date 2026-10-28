// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WGSL backend (Rexcomp): one stage of a WebGPU pipeline as a WGSL module with a `main` entry point.

#pragma once

#include <array>
#include <string>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Backend::WGSL {

/// Vertex attributes in formats WebGPU's vertex fetch lacks: read as one u32 and unpacked in the vertex stage.
enum class VertexUnpack : u8 {
    None,
    A2B10G10R10_SNORM,
    A2B10G10R10_UINT,     ///< (the generic is an unsigned integer)
    A2B10G10R10_SINT,     ///< (signed integer)
    A2B10G10R10_USCALED,  ///< (floats of the unsigned values)
    A2B10G10R10_SSCALED,  ///< (floats of the signed values)
    B10G11R11_FLOAT,
};

/// What one stage needs to know about its neighbours in the pipeline.
struct Options {
    /// Vertex stages: the interpolation the fragment stage declares for each generic (WebGPU requires both sides
    /// of a location to agree).
    std::array<Interpolation, 32> output_interpolation{};
    /// Vertex stages: attributes fetched packed (see VertexUnpack).
    std::array<VertexUnpack, 32> vertex_unpack{};
    /// Vertex stages: attributes the vertex stage reads from memory itself ("vertex pulling": layouts WebGPU's vertex
    /// fetch cannot describe, like strides that are not multiples of 4). Each names one of up to 3 pulled streams,
    /// bound after every other resource as `var<storage, read> vs_vb<k>: array<u32>`, preceded by
    /// `var<uniform> vs_pull: vec4<u32>` (x, y, z: where stream 0, 1, 2 starts in its binding, in bytes; w: added to
    /// the instance index of instanced streams).
    struct VertexPull {
        bool enabled{};
        u8 stream{};
        bool instanced{};
        u32 attribute{};   ///< the Maxwell attribute format word (offset, size, type, bgra)
        u32 stride{};
    };
    std::array<VertexPull, 32> vertex_pull{};
};

/// Storage buffers bound to one same range (the browser forbids two writable bindings of one range): for the next
/// EmitWGSL on this thread, ssbo_alias_next[j] = i + 1 makes descriptor j use descriptor i's binding (j is not
/// declared). Reset by the emission.
extern thread_local std::array<u8, 64> ssbo_alias_next;
/// ...and where descriptor j's data starts in descriptor i's binding, in words (overlapping ranges merged into one)
extern thread_local std::array<u32, 64> ssbo_alias_words_next;

[[nodiscard]] std::string EmitWGSL(const Profile& profile, const RuntimeInfo& runtime_info,
                                   IR::Program& program, Bindings& bindings, const Options& options = {});

[[nodiscard]] inline std::string EmitWGSL(const Profile& profile, IR::Program& program) {
    Bindings binding;
    return EmitWGSL(profile, {}, program, binding);
}

} // namespace Shader::Backend::WGSL
