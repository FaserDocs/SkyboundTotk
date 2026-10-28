// Rexcomp replacement for suyu's common/settings.h: only the values shader_recompiler reads, at suyu's defaults.
#pragma once
#include "common/common_types.h"

namespace Settings {
enum class AstcRecompression : u32 { Uncompressed = 0, Bc1 = 1, Bc3 = 2 };
template <typename T>
struct Setting {
    T value;
    T GetValue() const { return value; }
    operator T() const { return value; }
};
struct ResolutionScalingInfo {
    u32 up_scale{1};
    u32 down_shift{0};
    f32 up_factor{1.0f};
    f32 down_factor{1.0f};
    bool active{false};
};
struct Values {
    Setting<bool> disable_shader_loop_safety_checks{false};
    Setting<bool> renderer_debug{false};
    Setting<bool> rescale_hack{false};
    ResolutionScalingInfo resolution_info{};
    Setting<AstcRecompression> astc_recompression{AstcRecompression::Uncompressed};
};
inline Values values;
} // namespace Settings
