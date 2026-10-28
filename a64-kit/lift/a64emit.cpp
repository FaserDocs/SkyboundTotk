// a64emit: the per-instruction AArch64 -> C translator of suyu's static recompiler (suyu/arm64_to_c.h,
// GPL-2.0-or-later), built as a DLL so the Python lifter (a64_lift.py) can call it through ctypes.
//
// The lifter owns everything around single instructions: functions, control flow, calls, jump tables, svc.
// It only asks this for the C of instructions that do not end a block, and treats a terminator answer as a
// lifter bug. Licensing: this file and every binary built from it are GPL-2.0-or-later (see suyu/).
//
// Build: python a64-kit/lift/build_emit.py   ->  a64-kit/lift/a64emit.dll

#include "suyu/arm64_to_c.h"

#if defined(_WIN32)
#define A64_EXPORT extern "C" __declspec(dllexport)
#else
#define A64_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using namespace suyu::recomp;

// translate_all: translate the forms suyu leaves to its JIT in Hybrid mode (we have no JIT).
// fpx: 0 = exact integer FP bodies only, 1 = FPX1 (native fast path, exact body when it cannot prove equality).
A64_EXPORT void a64_config(int translate_all, int fpx) {
    g_translate_all = translate_all != 0;
    g_emit_fpx = fpx != 0;
    g_emit_fpx_shadow = false;
    g_emit_fastmem = false;
    g_emit_guard_gen = false;
    g_chain_blocks = nullptr;
    g_chain_mod = nullptr;
}

// Translates n instructions. The C of instruction k is out[offs[k] .. offs[k+1]). flags[k]: bit 0 = unhandled
// (fell through to recomp_unhandled), bit 1 = terminator (Translate closed the block).
// Returns the number of bytes needed: if it is larger than cap, nothing past cap was written and the caller retries.
A64_EXPORT long long a64_translate_batch(const uint32_t* insns, const uint64_t* pcs, int n, char* out,
                                         long long cap, long long* offs, int* flags) {
    std::string s;
    s.reserve((size_t)n * 96);
    for (int k = 0; k < n; ++k) {
        offs[k] = (long long)s.size();
        bool unhandled = false;
        const bool open = Translate(insns[k], pcs[k], s, &unhandled);
        flags[k] = (unhandled ? 1 : 0) | (open ? 0 : 2);
    }
    offs[n] = (long long)s.size();
    if ((long long)s.size() <= cap) memcpy(out, s.data(), s.size());
    return (long long)s.size();
}

// Runtime texts the emitted C depends on, for the build to write next to the lifted code.
A64_EXPORT const char* a64_runtime_text(int which) {
    static std::string keep[8];
    switch (which) {
    case 0: return RuntimeH();
    case 1: return keep[1] = BuildRuntimeC(false, false), keep[1].c_str();
    case 2: return EstimateRuntimeC();
    case 3: return keep[3] = FpxExactC(), keep[3].c_str();
    default: return "";
    }
}
