/* TOTK's core count (experiment, HOS_GAME_CORES=<n>, 3-8): its job pools have one worker per core.

   f_0a9bcccc (v1.0.0, f_0aa2ee9c in 1.2.1) is the engine's core setup: it writes 3 (the Switch's application cores) to
   0xC4E3610 (1.2.1: 0xC5B8EC8), builds the masks of all cores (bit 0 up, at 0xC63C000+288; 1.2.1: 0xC716000+3872) and of
   the sub cores (bit 1 up, +308 / +3892), and stores min(cores, 3) at 0xC4E3614 (1.2.1: 0xC5B8ECC). The job pools size
   themselves from that last value: a pool's workers are popcount(its core mask & the cores below it), core 0 left out
   unless the pool asks for it (f_090db9f4), so 3 gives ModuleSystemWorker1-2 and BaseProcExecutor0-2. The frame waits on
   those pools (MainThread ~30% at the ModuleSystem barrier at the castle); a 6-core PC could run more workers. This
   replaces the function with the same stores for <n> cores; the OS layer must report that many cores too (HOS_CORES,
   hos_thread.c), or threads on the new cores would share a core number (the engine keys per-core arenas on it). Off by
   default. RESULT (2026-10-20): the pools do grow (6 cores: ModuleSystem workers 1-5, BaseProc executors 0-5), but the
   rest of the engine is sized for 3 cores: with 4 the graphics layer's per-core command buffers broke ("command handle
   ... is not a live recording" at nvnCommandBufferCallCommands during boot), with 6 the boot deadlocked (MainThread
   waiting on a ModuleSystem job group). More workers would need the engine's per-core state found and widened; not
   pursued. */
#include "hos.h"
#include <stdlib.h>

#if HOS_GAME_VERSION == 121
void f_0c92bea0(Ctx *);   /* (sdk, called by the original with (0xC716000+3888, 0)) */
#define CORES_MAGIC 0xC716F24ULL   /* (info + 3876) */
#define CORES_ALL 0xC716F20ULL     /* (info + 3872: the mask of all cores) */
#define CORES_FLAGS 0xC716F2CULL   /* (info + 3884) */
#define CORES_SDK 0xC716F30ULL     /* (info + 3888) */
#define CORES_SUB 0xC716F34ULL     /* (info + 3892: the sub cores' mask) */
#define CORES_POOL 0xC5B8EC8ULL    /* (the core count, 3) */
#define CORES_POOL2 0xC5B8ECCULL   /* (min(cores, 3)) */
#else
void f_0c82bea0(Ctx *);   /* (sdk, called by the original with (0xC63C000+304, 0)) */
#define CORES_MAGIC 0xC63C124ULL   /* (info + 292) */
#define CORES_ALL 0xC63C120ULL     /* (info + 288: the mask of all cores) */
#define CORES_FLAGS 0xC63C12CULL   /* (info + 300) */
#define CORES_SDK 0xC63C130ULL     /* (info + 304) */
#define CORES_SUB 0xC63C134ULL     /* (info + 308: the sub cores' mask) */
#define CORES_POOL 0xC4E3610ULL    /* (the core count, 3) */
#define CORES_POOL2 0xC4E3614ULL   /* (min(cores, 3)) */
#endif

int hos_game_cores(void) {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("HOS_GAME_CORES");
        n = e && atoi(e) >= 3 && atoi(e) <= 8 ? atoi(e) : 0;
    }
    return n;
}

static int cores_hook(Ctx *c) {
    const int n = hos_game_cores();
    if (!n) return 0;
    G64(CORES_MAGIC) = 0x100000000ull;
    G32(CORES_POOL) = (uint32_t)n;
    G32(CORES_FLAGS) = 2;
    c->x[0] = CORES_SDK;
    c->x[1] = 0;
#if HOS_GAME_VERSION == 121
    f_0c92bea0(c);
#else
    f_0c82bea0(c);
#endif
    uint32_t all = G32(CORES_ALL) | 1u, sub = G32(CORES_SUB);
    for (int i = 1; i < n; i++) all |= 1u << i, sub |= 1u << i;
    G32(CORES_SUB) = sub;
    G32(CORES_ALL) = all;
    G32(CORES_POOL2) = (uint32_t)n;
    hos_log("cores: the game set up for %d cores (job pools: %d ModuleSystem workers, BaseProc executors by mask)", n, n - 1);
    return 1;
}
#if HOS_GAME_VERSION == 121
ENTRY_HOOK(0x0aa2ee9c) { return cores_hook(c); }
#else
ENTRY_HOOK(0x0a9bcccc) { return cores_hook(c); }
#endif

/* HOS_BASEPROC_EXECUTORS=<n> (experiment, 4-8): the BaseProc job pool alone gets n executors; the rest of the engine
   stays set up for 3 cores. At the castle the frame waits on that pool (ModuleSystemWorker2 ~3.9 ms a frame for the
   BaseProc job group, lr 0x874a588) while its 3 executors run actor jobs from shared lock-free queues; the 3 other
   game threads only wait then. f_09247560 (f_09266494 in 1.2.1) makes the pool: one executor per core below its core
   count (read from 0xC4E3614 / 0xC5B8ECC into w26, here) whose bit is in the pool's mask (all bits), pinned to that
   core; its executor array is sized from the same count, and the setup after it (f_09247384) records only cores 0-2 in
   its per-core table. Executors past core 2 report core numbers wrapped (hos_thread_core), as the existing 3 executors
   already share core numbers with MainThread and the ModuleSystem workers. RESULT (2026-10-21, castle, 3 interleaved
   rounds): the 5 executors start and run, but 67.0 -> 64.8 fps: each frame has ~10 BaseProc job phases whose length
   dependency chains set (the executors already wait on each other ~10% of the time, lr 0x883aa98). Off. */
static void baseproc_executors(Ctx *c) {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("HOS_BASEPROC_EXECUTORS");
        n = e && atoi(e) >= 4 && atoi(e) <= 8 ? atoi(e) : 0;
    }
    if (!n || hos_game_cores() || (uint32_t)c->x[26] >= (uint32_t)n) return;
    hos_log("cores: the BaseProc job pool gets %d executors (the game is set up for %u cores)", n, (unsigned)c->x[26]);
    c->x[26] = (uint64_t)n;
}
#if HOS_GAME_VERSION == 121
SPLICE_AFTER(0x09266494, "c->x[26]=recomp_load32(c,c->x[8] + 0);") { baseproc_executors(c); }
#else
SPLICE_AFTER(0x09247560, "c->x[26]=recomp_load32(c,c->x[8] + 0);") { baseproc_executors(c); }
#endif
