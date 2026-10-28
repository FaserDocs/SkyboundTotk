// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>
#include <cstdlib>

#include <string>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#include "shader_recompiler/backend/wgsl/var_alloc.h"
#include "shader_recompiler/stage.h"

namespace Shader {
struct Info;
struct Profile;
struct RuntimeInfo;
} // namespace Shader

namespace Shader::Backend {
struct Bindings;
}

namespace Shader::IR {
class Inst;
struct Program;
} // namespace Shader::IR

namespace Shader::Backend::WGSL {

/// RX_CBUF_CLAMP=1: constant buffer reads at run-time offsets past the bound size get WebGPU's clamped read (the old
/// translation) instead of the hardware's 0 (<stage>_cbuf<n>_at).
inline bool CbufClamp() {
    static const bool clamp = std::getenv("RX_CBUF_CLAMP") != nullptr;
    return clamp;
}

struct GenericElementInfo {
    std::string name;
    u32 first_element{};
    u32 num_components{};
};

struct TextureImageDefinition {
    u32 binding;
    u32 count;
    u32 sampler_plus1 = 0;   // the texture whose sampler it samples with, binding + 1 (0: its own; shared: web)
    u32 textures_sampler_binding() const { return sampler_plus1 ? sampler_plus1 - 1 : binding; }
};

/// A resource of the stage's bind group: its binding number and the descriptor index it serves.
struct ResourceBinding {
    u32 binding;
    u32 index;
};

class EmitContext {
public:
    explicit EmitContext(IR::Program& program, Bindings& bindings, const Profile& profile_,
                         const RuntimeInfo& runtime_info_);

    template <WgslVarType type, typename... Args>
    void Add(const char* format_str, IR::Inst& inst, Args&&... args) {
        const auto var_def{var_alloc.AddDefine(inst, type)};
        if (var_def.empty()) {
            // the result is unused: WGSL needs a phony assignment to evaluate an expression for its effects
            code += "_=";
            code += fmt::format(fmt::runtime(format_str + 3), std::forward<Args>(args)...);
        } else {
            code += fmt::format(fmt::runtime(format_str), var_def, std::forward<Args>(args)...);
        }
        // TODO: Remove this
        code += '\n';
    }

    template <typename... Args>
    void AddU1(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U1>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF16x2(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F16x2>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU64(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U64>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF64(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F64>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32x2(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32x2>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32x2(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32x2>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32x3(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32x3>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32x3(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32x3>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32x4(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32x4>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32x4(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32x4>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddPrecF32(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::PrecF32>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddPrecF64(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::PrecF64>(format_str, inst, args...);
    }

    template <typename... Args>
    void Add(const char* format_str, Args&&... args) {
        code += fmt::format(fmt::runtime(format_str), std::forward<Args>(args)...);
        // TODO: Remove this
        code += '\n';
    }

    std::string header;
    std::string code;
    VarAlloc var_alloc;
    const Info& info;
    const Profile& profile;
    const RuntimeInfo& runtime_info;

    Stage stage{};
    std::string_view stage_name = "invalid";
    std::string_view position_name = "gl_Position";
    u32 group{};   ///< the bind group of this stage's resources

    /// The stage's uniform block (binding 0, bound at a dynamic offset per draw): its name, size in vec4s, and
    /// which constant buffers are in it and where (the others are storage buffers: indexed at run time, or no
    /// room left in the 64 KB a uniform binding may have).
    std::string ub_name;
    u32 ub_size{};
    std::array<bool, 18> cbuf_in_ub{};
    std::array<u32, 18> ub_cbuf_base{};

    std::vector<ResourceBinding> cbuf_bindings;
    std::vector<ResourceBinding> ssbo_bindings;
    // Storage buffers (by index) and shared memory an atomic operation touches: declared array<atomic<u32>> and
    // read and written through atomicLoad / atomicStore (WGSL atomics need atomic-typed memory)
    std::bitset<64> atomic_ssbo;
    std::array<u8, 64> ssbo_alias{};   // (see ssbo_alias_next)
    std::array<u32, 64> ssbo_alias_words{};
    std::string SsboWord(u32 binding, std::string_view word) const {
        return binding < 64 && ssbo_alias[binding] && ssbo_alias_words[binding]
                   ? fmt::format("(({}) + {}u)", word, ssbo_alias_words[binding]) : std::string(word);
    }
    u32 MapSsbo(u32 binding) const { return binding < 64 && ssbo_alias[binding] ? ssbo_alias[binding] - 1u : binding; }
    bool atomic_smem{};
    std::string SsboLoad(u32 binding, std::string_view word) const;
    std::string SsboStore(u32 binding, std::string_view word, std::string_view value) const;
    std::string SsboPtr(u32 binding, std::string_view word) const;
    std::string SmemLoad(std::string_view word) const;
    std::string SmemStore(std::string_view word, std::string_view value) const;
    std::vector<ResourceBinding> texture_bindings;

    std::vector<TextureImageDefinition> texture_buffers;
    std::vector<TextureImageDefinition> image_buffers;
    std::vector<TextureImageDefinition> textures;
    std::vector<TextureImageDefinition> images;
    std::array<std::array<GenericElementInfo, 4>, 32> output_generics{};

    u32 num_safety_loop_vars{};

    bool uses_y_direction{};
    bool uses_cc_carry{};
    bool uses_geometry_passthrough{};

private:
    void DefineBuiltins();
    void DefineConstantBuffers(Bindings& bindings);
    void DefineConstantBufferIndirect();
    void DefineStorageBuffers(Bindings& bindings);
    void DefineHelperFunctions();
    void DefineConstants();
    void SetupImages(Bindings& bindings);
    void SetupTextures(Bindings& bindings);
};

} // namespace Shader::Backend::WGSL
