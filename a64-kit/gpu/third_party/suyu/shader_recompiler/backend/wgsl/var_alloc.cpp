// SPDX-FileCopyrightText: Copyright 2025 suyu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>
#include <string_view>

#include <fmt/ranges.h>

#include "shader_recompiler/backend/wgsl/var_alloc.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
namespace {
std::string TypePrefix(WgslVarType type) {
    switch (type) {
    case WgslVarType::U1:
        return "b_";
    case WgslVarType::F16x2:
        return "f16x2_";
    case WgslVarType::U32:
        return "u_";
    case WgslVarType::F32:
        return "f_";
    case WgslVarType::U64:
        return "u64_";
    case WgslVarType::F64:
        return "d_";
    case WgslVarType::U32x2:
        return "u2_";
    case WgslVarType::F32x2:
        return "f2_";
    case WgslVarType::U32x3:
        return "u3_";
    case WgslVarType::F32x3:
        return "f3_";
    case WgslVarType::U32x4:
        return "u4_";
    case WgslVarType::F32x4:
        return "f4_";
    case WgslVarType::PrecF32:
        return "pf_";
    case WgslVarType::PrecF64:
        return "pd_";
    case WgslVarType::Void:
        return "";
    default:
        throw NotImplementedException("Type {}", type);
    }
}

std::string FormatFloat(std::string_view value, IR::Type type) {
    if (type != IR::Type::F32) {
        throw NotImplementedException("WGSL has no 64-bit floats");
    }
    // WGSL const-evaluates bitcasts of literals and rejects non-finite results: read them from private variables
    if (value == "nan") {
        return "bitcast<f32>(F32_NAN_)";
    }
    if (value == "inf") {
        return "bitcast<f32>(F32_INF_)";
    }
    if (value == "-inf") {
        return "bitcast<f32>(F32_NINF_)";
    }
    const bool scientific{value.find_first_of('e') != std::string_view::npos};
    const bool needs_dot{!scientific && value.find_first_of('.') == std::string_view::npos};
    return fmt::format("{}{}f", value, needs_dot ? ".0" : "");
}

std::string MakeImm(const IR::Value& value) {
    switch (value.Type()) {
    case IR::Type::U1:
        return value.U1() ? "true" : "false";
    case IR::Type::U32:
        return fmt::format("{}u", value.U32());
    case IR::Type::F32:
        return FormatFloat(fmt::format("{}", value.F32()), IR::Type::F32);
    case IR::Type::U64:
        return fmt::format("vec2<u32>({}u, {}u)", static_cast<u32>(value.U64()),
                           static_cast<u32>(value.U64() >> 32));
    case IR::Type::Void:
        return "";
    default:
        throw NotImplementedException("Immediate type {}", value.Type());
    }
}
} // Anonymous namespace

std::string VarAlloc::Representation(u32 index, WgslVarType type) const {
    const auto prefix{TypePrefix(type)};
    return fmt::format("{}{}", prefix, index);
}

std::string VarAlloc::Representation(Id id) const {
    return Representation(id.index, id.type);
}

std::string VarAlloc::Define(IR::Inst& inst, WgslVarType type) {
    if (inst.HasUses()) {
        inst.SetDefinition<Id>(Alloc(type));
        return Representation(inst.Definition<Id>());
    } else {
        Id id{};
        id.type.Assign(type);
        GetUseTracker(type).uses_temp = true;
        inst.SetDefinition<Id>(id);
        return 't' + Representation(inst.Definition<Id>());
    }
}

std::string VarAlloc::Define(IR::Inst& inst, IR::Type type) {
    return Define(inst, RegType(type));
}

std::string VarAlloc::PhiDefine(IR::Inst& inst, IR::Type type) {
    return AddDefine(inst, RegType(type));
}

std::string VarAlloc::AddDefine(IR::Inst& inst, WgslVarType type) {
    if (inst.HasUses()) {
        inst.SetDefinition<Id>(Alloc(type));
        return Representation(inst.Definition<Id>());
    } else {
        return "";
    }
}

std::string VarAlloc::Consume(const IR::Value& value) {
    return value.IsImmediate() ? MakeImm(value) : ConsumeInst(*value.InstRecursive());
}

std::string VarAlloc::ConsumeInst(IR::Inst& inst) {
    inst.DestructiveRemoveUsage();
    if (!inst.HasUses()) {
        Free(inst.Definition<Id>());
    }
    return Representation(inst.Definition<Id>());
}

std::string VarAlloc::GetWgslType(IR::Type type) const {
    return GetWgslType(RegType(type));
}

Id VarAlloc::Alloc(WgslVarType type) {
    auto& use_tracker{GetUseTracker(type)};
    const auto num_vars{use_tracker.var_use.size()};
    for (size_t var = 0; var < num_vars; ++var) {
        if (use_tracker.var_use[var]) {
            continue;
        }
        use_tracker.num_used = (std::max)(use_tracker.num_used, var + 1);
        use_tracker.var_use[var] = true;
        Id ret{};
        ret.is_valid.Assign(1);
        ret.type.Assign(type);
        ret.index.Assign(static_cast<u32>(var));
        return ret;
    }
    // Allocate a new variable
    use_tracker.var_use.push_back(true);
    Id ret{};
    ret.is_valid.Assign(1);
    ret.type.Assign(type);
    ret.index.Assign(static_cast<u32>(use_tracker.num_used));
    ++use_tracker.num_used;
    return ret;
}

void VarAlloc::Free(Id id) {
    if (id.is_valid == 0) {
        throw LogicError("Freeing invalid variable");
    }
    auto& use_tracker{GetUseTracker(id.type)};
    use_tracker.var_use[id.index] = false;
}

WgslVarType VarAlloc::RegType(IR::Type type) const {
    switch (type) {
    case IR::Type::U1:
        return WgslVarType::U1;
    case IR::Type::U32:
        return WgslVarType::U32;
    case IR::Type::F32:
        return WgslVarType::F32;
    case IR::Type::U64:
        return WgslVarType::U64;
    case IR::Type::F64:
        return WgslVarType::F64;
    default:
        throw NotImplementedException("IR type {}", type);
    }
}

std::string VarAlloc::GetWgslType(WgslVarType type) const {
    switch (type) {
    case WgslVarType::U1:
        return "bool";
    case WgslVarType::F16x2:   // no f16 in core WebGPU: half pairs are carried as f32 pairs
    case WgslVarType::F32x2:
        return "vec2<f32>";
    case WgslVarType::U32:
        return "u32";
    case WgslVarType::F32:
    case WgslVarType::PrecF32:
        return "f32";
    case WgslVarType::U64:     // no 64-bit integers either: low and high words
    case WgslVarType::U32x2:
        return "vec2<u32>";
    case WgslVarType::U32x3:
        return "vec3<u32>";
    case WgslVarType::F32x3:
        return "vec3<f32>";
    case WgslVarType::U32x4:
        return "vec4<u32>";
    case WgslVarType::F32x4:
        return "vec4<f32>";
    case WgslVarType::F64:
    case WgslVarType::PrecF64:
        throw NotImplementedException("WGSL has no 64-bit floats");
    case WgslVarType::Void:
        return "";
    default:
        throw NotImplementedException("Type {}", type);
    }
}

VarAlloc::UseTracker& VarAlloc::GetUseTracker(WgslVarType type) {
    switch (type) {
    case WgslVarType::U1:
        return var_bool;
    case WgslVarType::F16x2:
        return var_f16x2;
    case WgslVarType::U32:
        return var_u32;
    case WgslVarType::F32:
        return var_f32;
    case WgslVarType::U64:
        return var_u64;
    case WgslVarType::F64:
        return var_f64;
    case WgslVarType::U32x2:
        return var_u32x2;
    case WgslVarType::F32x2:
        return var_f32x2;
    case WgslVarType::U32x3:
        return var_u32x3;
    case WgslVarType::F32x3:
        return var_f32x3;
    case WgslVarType::U32x4:
        return var_u32x4;
    case WgslVarType::F32x4:
        return var_f32x4;
    case WgslVarType::PrecF32:
        return var_precf32;
    case WgslVarType::PrecF64:
        return var_precf64;
    default:
        throw NotImplementedException("Type {}", type);
    }
}

const VarAlloc::UseTracker& VarAlloc::GetUseTracker(WgslVarType type) const {
    switch (type) {
    case WgslVarType::U1:
        return var_bool;
    case WgslVarType::F16x2:
        return var_f16x2;
    case WgslVarType::U32:
        return var_u32;
    case WgslVarType::F32:
        return var_f32;
    case WgslVarType::U64:
        return var_u64;
    case WgslVarType::F64:
        return var_f64;
    case WgslVarType::U32x2:
        return var_u32x2;
    case WgslVarType::F32x2:
        return var_f32x2;
    case WgslVarType::U32x3:
        return var_u32x3;
    case WgslVarType::F32x3:
        return var_f32x3;
    case WgslVarType::U32x4:
        return var_u32x4;
    case WgslVarType::F32x4:
        return var_f32x4;
    case WgslVarType::PrecF32:
        return var_precf32;
    case WgslVarType::PrecF64:
        return var_precf64;
    default:
        throw NotImplementedException("Type {}", type);
    }
}

} // namespace Shader::Backend::WGSL
