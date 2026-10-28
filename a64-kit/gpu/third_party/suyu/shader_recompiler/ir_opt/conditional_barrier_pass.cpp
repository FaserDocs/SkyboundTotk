// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {

// Barriers the host cannot run in conditional control flow are removed. One common case keeps its effect: a barrier
// after invocations returned early (the guest GPU counts exited invocations as arrived), e.g. every invocation
// stores to shared memory, all but the first return, the first waits and reads what the others stored. A barrier
// is put at the end of the last unconditional block before the construct holding the first early return: there all
// invocations are still running, and what they stored before returning is visible after it. (Rexcomp: without it,
// TOTK's lighting tile classification read shared memory before the other invocations had written it.)
void ConditionalBarrierPass(IR::Program& program) {
    s32 conditional_control_flow_count{0};
    s32 conditional_return_count{0};
    IR::Block* last_top_block{};       // the last block outside any construct
    IR::Block* before_construct{};     // the last block before the top-level construct being walked
    IR::Block* before_first_return{};  // the last block before the construct with the first early return
    bool hoisted{false};
    for (IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::If:
        case IR::AbstractSyntaxNode::Type::Loop:
            if (conditional_control_flow_count == 0) {
                before_construct = last_top_block;
            }
            conditional_control_flow_count++;
            break;
        case IR::AbstractSyntaxNode::Type::EndIf:
        case IR::AbstractSyntaxNode::Type::Repeat:
            conditional_control_flow_count--;
            break;
        case IR::AbstractSyntaxNode::Type::Unreachable:
        case IR::AbstractSyntaxNode::Type::Return:
            if (conditional_control_flow_count > 0) {
                if (conditional_return_count == 0) {
                    before_first_return = before_construct;
                }
                conditional_return_count++;
            }
            break;
        case IR::AbstractSyntaxNode::Type::Block:
            for (IR::Inst& inst : node.data.block->Instructions()) {
                if ((conditional_control_flow_count > 0 || conditional_return_count > 0) &&
                    inst.GetOpcode() == IR::Opcode::Barrier) {
                    LOG_WARNING(Shader, "Barrier within conditional control flow");
                    inst.ReplaceOpcode(IR::Opcode::Identity);
                    if (conditional_control_flow_count == 0 && before_first_return && !hoisted) {
                        before_first_return->PrependNewInst(before_first_return->end(), IR::Opcode::Barrier);
                        hoisted = true;
                    }
                }
            }
            if (conditional_control_flow_count == 0 && conditional_return_count == 0) {
                last_top_block = node.data.block;
            }
            break;
        default:
            break;
        }
    }
    ASSERT(conditional_control_flow_count == 0);
}

} // namespace Shader::Optimization
