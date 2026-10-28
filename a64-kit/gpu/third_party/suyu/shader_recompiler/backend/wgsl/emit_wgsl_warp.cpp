// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
namespace {
constexpr char THREAD_ID[]{"gl_SubGroupInvocationARB"};

void SetInBoundsFlag(EmitContext& ctx, IR::Inst& inst) {
    IR::Inst* const in_bounds{inst.GetAssociatedPseudoOperation(IR::Opcode::GetInBoundsFromOp)};
    if (!in_bounds) {
        return;
    }
    ctx.AddU1("{}=shfl_in_bounds;", *in_bounds);
    in_bounds->Invalidate();
}

std::string ComputeMinThreadId(std::string_view thread_id, std::string_view segmentation_mask) {
    return fmt::format("({}&{})", thread_id, segmentation_mask);
}

std::string ComputeMaxThreadId(std::string_view min_thread_id, std::string_view clamp,
                               std::string_view not_seg_mask) {
    return fmt::format("({})|({}&{})", min_thread_id, clamp, not_seg_mask);
}

std::string GetMaxThreadId(std::string_view thread_id, std::string_view clamp,
                           std::string_view segmentation_mask) {
    const auto not_seg_mask{fmt::format("(~{})", segmentation_mask)};
    const auto min_thread_id{ComputeMinThreadId(thread_id, segmentation_mask)};
    return ComputeMaxThreadId(min_thread_id, clamp, not_seg_mask);
}

void UseShuffleNv(EmitContext& ctx, IR::Inst& inst, std::string_view shfl_op,
                  std::string_view value, std::string_view index,
                  [[maybe_unused]] std::string_view clamp, std::string_view segmentation_mask) {
    const auto width{fmt::format("32u>>(countOneBits({}&31u))", segmentation_mask)};
    ctx.AddU32("{}={}({},{},{},shfl_in_bounds);", inst, shfl_op, value, index, width);
    SetInBoundsFlag(ctx, inst);
}

std::string_view BallotIndex(EmitContext& ctx) {
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ".x";
    }
    return "[gl_SubGroupInvocationARB>>5]";
}

std::string GetMask(EmitContext& ctx, std::string_view mask) {
    const auto ballot_index{BallotIndex(ctx)};
    return fmt::format("u32(vec2<u32>({}){})", mask, ballot_index);
}
} // Anonymous namespace

void EmitLaneId(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}={}&31u;", inst, THREAD_ID);
}

void EmitVoteAll(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU1("{}={};", inst, pred);
}

void EmitVoteAny(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU1("{}={};", inst, pred);
}

void EmitVoteEqual(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU1("{}=true;", inst);
}

void EmitSubgroupBallot(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU32("{}=select(0u,1u,{});", inst, pred);
}

void EmitSubgroupEqMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=1u;", inst);
}

void EmitSubgroupLtMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitSubgroupLeMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=1u;", inst);
}

void EmitSubgroupGtMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitSubgroupGeMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=1u;", inst);
}

void EmitShuffleIndex(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                      std::string_view index, std::string_view clamp, std::string_view seg_mask) {
    if (ctx.profile.support_gl_warp_intrinsics) {
        UseShuffleNv(ctx, inst, "shuffleNV", value, index, clamp, seg_mask);
        return;
    }
    const bool big_warp{ctx.profile.warp_size_potentially_larger_than_guest};
    const auto is_upper_partition{"i32(gl_SubGroupInvocationARB)>=32"};
    const auto upper_index{fmt::format("select({},{}+32u,{})", index, index, is_upper_partition)};
    const auto upper_clamp{fmt::format("select({},{}+32u,{})", clamp, clamp, is_upper_partition)};

    const auto not_seg_mask{fmt::format("(~{})", seg_mask)};
    const auto min_thread_id{ComputeMinThreadId(THREAD_ID, seg_mask)};
    const auto max_thread_id{
        ComputeMaxThreadId(min_thread_id, big_warp ? upper_clamp : clamp, not_seg_mask)};

    const auto lhs{fmt::format("({}&{})", big_warp ? upper_index : index, not_seg_mask)};
    const auto src_thread_id{fmt::format("({})|({})", lhs, min_thread_id)};
    ctx.Add("shfl_in_bounds=i32({})<=i32({});", src_thread_id, max_thread_id);
    SetInBoundsFlag(ctx, inst);
    ctx.Add("shfl_result={};", value);
    ctx.AddU32("{}=select({},shfl_result,shfl_in_bounds);", inst, value);
}

void EmitShuffleUp(EmitContext& ctx, IR::Inst& inst, std::string_view value, std::string_view index,
                   std::string_view clamp, std::string_view seg_mask) {
    if (ctx.profile.support_gl_warp_intrinsics) {
        UseShuffleNv(ctx, inst, "shuffleUpNV", value, index, clamp, seg_mask);
        return;
    }
    const bool big_warp{ctx.profile.warp_size_potentially_larger_than_guest};
    const auto is_upper_partition{"i32(gl_SubGroupInvocationARB)>=32"};
    const auto upper_clamp{fmt::format("select({},{}+32u,{})", clamp, clamp, is_upper_partition)};

    const auto max_thread_id{GetMaxThreadId(THREAD_ID, big_warp ? upper_clamp : clamp, seg_mask)};
    const auto src_thread_id{fmt::format("({}-{})", THREAD_ID, index)};
    ctx.Add("shfl_in_bounds=i32({})>=i32({});", src_thread_id, max_thread_id);
    SetInBoundsFlag(ctx, inst);
    ctx.Add("shfl_result={};", value);
    ctx.AddU32("{}=select({},shfl_result,shfl_in_bounds);", inst, value);
}

void EmitShuffleDown(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                     std::string_view index, std::string_view clamp, std::string_view seg_mask) {
    if (ctx.profile.support_gl_warp_intrinsics) {
        UseShuffleNv(ctx, inst, "shuffleDownNV", value, index, clamp, seg_mask);
        return;
    }
    const bool big_warp{ctx.profile.warp_size_potentially_larger_than_guest};
    const auto is_upper_partition{"i32(gl_SubGroupInvocationARB)>=32"};
    const auto upper_clamp{fmt::format("select({},{}+32u,{})", clamp, clamp, is_upper_partition)};

    const auto max_thread_id{GetMaxThreadId(THREAD_ID, big_warp ? upper_clamp : clamp, seg_mask)};
    const auto src_thread_id{fmt::format("({}+{})", THREAD_ID, index)};
    ctx.Add("shfl_in_bounds=i32({})<=i32({});", src_thread_id, max_thread_id);
    SetInBoundsFlag(ctx, inst);
    ctx.Add("shfl_result={};", value);
    ctx.AddU32("{}=select({},shfl_result,shfl_in_bounds);", inst, value);
}

void EmitShuffleButterfly(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                          std::string_view index, std::string_view clamp,
                          std::string_view seg_mask) {
    if (ctx.profile.support_gl_warp_intrinsics) {
        UseShuffleNv(ctx, inst, "shuffleXorNV", value, index, clamp, seg_mask);
        return;
    }
    const bool big_warp{ctx.profile.warp_size_potentially_larger_than_guest};
    const auto is_upper_partition{"i32(gl_SubGroupInvocationARB)>=32"};
    const auto upper_clamp{fmt::format("select({},{}+32u,{})", clamp, clamp, is_upper_partition)};

    const auto max_thread_id{GetMaxThreadId(THREAD_ID, big_warp ? upper_clamp : clamp, seg_mask)};
    const auto src_thread_id{fmt::format("({}^{})", THREAD_ID, index)};
    ctx.Add("shfl_in_bounds=i32({})<=i32({});", src_thread_id, max_thread_id);
    SetInBoundsFlag(ctx, inst);
    ctx.Add("shfl_result={};", value);
    ctx.AddU32("{}=select({},shfl_result,shfl_in_bounds);", inst, value);
}

void EmitFSwizzleAdd(EmitContext& ctx, IR::Inst& inst, std::string_view op_a, std::string_view op_b,
                     std::string_view swizzle) {
    const auto mask{fmt::format("({}>>((gl_SubGroupInvocationARB&3)<<1))&3", swizzle)};
    const std::string modifier_a = fmt::format("FSWZ_A[{}]", mask);
    const std::string modifier_b = fmt::format("FSWZ_B[{}]", mask);
    ctx.AddF32("{}=({}*{})+({}*{});", inst, op_a, modifier_a, op_b, modifier_b);
}

void EmitDPdxFine(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdxFine({});", inst, op_a);
}

void EmitDPdyFine(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdyFine({});", inst, op_a);
}

void EmitDPdxCoarse(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdxCoarse({});", inst, op_a);
}

void EmitDPdyCoarse(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdyCoarse({});", inst, op_a);
}
} // namespace Shader::Backend::WGSL
