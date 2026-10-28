/* TOTK's core count (experiment, HOS_GAME_CORES=<n>, 3-8): its job pools have one worker per core.

   f_0a9bcccc (v1.0.0) is the engine's core setup: it writes 3 (the Switch's application cores) to 0xC4E3610, builds the
   masks of all cores (bit 0 up, at 0xC63C000+288) and of the sub cores (bit 1 up, +308), and stores min(cores, 3) at
   0xC4E3614. The job pools size themselves from that last value: a pool's workers are popcount(its core mask & the
   cores below it), core 0 left out unless the pool asks for it (f_090db9f4), so 3 gives ModuleSystemWorker1-2 and
   BaseProcExecutor0-2. The frame waits on those pools (MainThread ~30% at the ModuleSystem barrier at the castle); a
   6-core PC could run more workers. This replaces the function with the same stores for <n> cores; the OS layer must
   report that many cores too (HOS_CORES, hos_thread.c), or threads on the new cores would share a core number (the
   engine keys per-core arenas on it). Off by default. RESULT (2026-10-20): the pools do grow (6 cores: ModuleSystem
   workers 1-5, BaseProc executors 0-5), but the rest of the engine is sized for 3 cores: with 4 the graphics layer's
   per-core command buffers broke ("command handle ... is not a live recording" at nvnCommandBufferCallCommands during
   boot), with 6 the boot deadlocked (MainThread waiting on a ModuleSystem job group). More workers would need the
   engine's per-core state found and widened; not pursued. */
#include "hos.h"
#include <stdlib.h>

void f_0c82bea0(Ctx *);   /* (sdk, called by the original with (0xC63C000+304, 0)) */

int hos_game_cores(void) {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("HOS_GAME_CORES");
        n = e && atoi(e) >= 3 && atoi(e) <= 8 ? atoi(e) : 0;
    }
    return n;
}

ENTRY_HOOK(0x0a9bcccc) {
    const int n = hos_game_cores();
    if (!n) return 0;
    const uint64_t info = 0xC63C000;
    G64(info + 292) = 0x100000000ull;
    G32(0xC4E3610) = (uint32_t)n;
    G32(info + 300) = 2;
    c->x[0] = info + 304;
    c->x[1] = 0;
    f_0c82bea0(c);
    uint32_t all = G32(info + 288) | 1u, sub = G32(info + 308);
    for (int i = 1; i < n; i++) all |= 1u << i, sub |= 1u << i;
    G32(info + 308) = sub;
    G32(info + 288) = all;
    G32(0xC4E3614) = (uint32_t)n;
    hos_log("cores: the game set up for %d cores (job pools: %d ModuleSystem workers, BaseProc executors by mask)", n, n - 1);
    return 1;
}

/* HOS_BASEPROC_EXECUTORS=<n> (experiment, 4-8): the BaseProc job pool alone gets n executors; the rest of the engine
   stays set up for 3 cores. At the castle the frame waits on that pool (ModuleSystemWorker2 ~3.9 ms a frame for the
   BaseProc job group, lr 0x874a588) while its 3 executors run actor jobs from shared lock-free queues; the 3 other
   game threads only wait then. f_09247560 makes the pool: one executor per core below its core count (read from
   0xC4E3614 into w26, here) whose bit is in the pool's mask (all bits), pinned to that core; its executor array is
   sized from the same count, and the setup after it (f_09247384) records only cores 0-2 in its per-core table.
   Executors past core 2 report core numbers wrapped (hos_thread_core), as the existing 3 executors already share
   core numbers with MainThread and the ModuleSystem workers. RESULT (2026-10-21, castle, 3 interleaved rounds): the
   5 executors start and run, but 67.0 -> 64.8 fps: each frame has ~10 BaseProc job phases whose length dependency
   chains set (the executors already wait on each other ~10% of the time, lr 0x883aa98). Off. */
SPLICE_AFTER(0x09247560, "c->x[26]=recomp_load32(c,c->x[8] + 0);") {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("HOS_BASEPROC_EXECUTORS");
        n = e && atoi(e) >= 4 && atoi(e) <= 8 ? atoi(e) : 0;
    }
    if (!n || hos_game_cores() || (uint32_t)c->x[26] >= (uint32_t)n) return;
    hos_log("cores: the BaseProc job pool gets %d executors (the game is set up for %u cores)", n, (unsigned)c->x[26]);
    c->x[26] = (uint64_t)n;
}
