/* The browser build's stand-ins for what hos_prof.c (the sampling profiler) and movie_mf.c (Media Foundation)
   provide natively, and the few Microsoft C runtime functions the OS layer calls. (Cutscenes: movie_web.c.) */
#define _GNU_SOURCE
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

void hos_host_thread_register_self(const char *name) { (void)name; }
#include <malloc.h>
void hos_memory_map_log(void) {}
void hos_process_memory(uint64_t *private_bytes, uint64_t *working_set) {
    struct mallinfo m = mallinfo();
    *private_bytes = (uint64_t)m.uordblks;
    *working_set = (uint64_t)m.arena;
}
void hos_backtrace(void) {}
/* HOS_DUMP_AFTER=<seconds> (web): the kernel's waiters and the threads' handles, then again every as many seconds */
static DWORD WINAPI dump_main(void *p) {
    const int s = (int)(intptr_t)p;
    for (;;) {
        Sleep((DWORD)s * 1000);
        void hos_kernel_dump(void);
        hos_kernel_dump();
    }
    return 0;
}
void web_leak_report(void);
/* RomFS fetches every 2 s, memory every 10 s: the host heap's top and use, and the game's lowest address region (the guest window's room).
   The heap's use (mallinfo) only with HOS_WEB_HEAP=1: it walks every chunk under the allocator's lock, ~0.3 s once the
   game runs, and every thread that allocated meanwhile waited (a stall every 2 s). */
#include <malloc.h>
#include <unistd.h>
static DWORD WINAPI mem_main(void *p) {
    (void)p;
    extern uint64_t hos_region_floor(void), hos_space_end;
    extern uint64_t web_fetch_count, web_fetch_bytes, web_fetch_us;
    uint64_t c0 = 0, b0 = 0, u0 = 0;
    const int heap_use = getenv("HOS_WEB_HEAP") != NULL;
    for (int tick = 0;; tick++) {
        Sleep(2000);
        const uint64_t c = web_fetch_count, b = web_fetch_bytes, u = web_fetch_us;
        if (c != c0)
            hos_log("web fetch: %u requests, %u MB, %u ms waited (threads summed) in 2 s", (unsigned)(c - c0),
                    (unsigned)((b - b0) >> 20), (unsigned)((u - u0) / 1000));
        c0 = c, b0 = b, u0 = u;
        {
            struct mallinfo mi = {0};
            if (heap_use) mi = mallinfo();
            int64_t waiting, queued, pending, rec, seed, left;
            void rx_texture_memory(int64_t *, int64_t *, int64_t *), rx_pipecache_memory(int64_t *, int64_t *, int64_t *);
            rx_texture_memory(&waiting, &queued, &pending);
            rx_pipecache_memory(&rec, &seed, &left);
            char use[64] = "";
            if (heap_use)
                snprintf(use, sizeof use, ", in use %u MB, free %u MB", (unsigned)(mi.uordblks >> 20),
                         (unsigned)(mi.fordblks >> 20));
            hos_log("web heap: top %u MB%s; textures waiting %u MB (%lld queued, %lld done); "
                    "pipeline cache %u + %u MB (%lld to make)", (unsigned)((uintptr_t)sbrk(0) >> 20), use,
                    (unsigned)(waiting >> 20),
                    (long long)queued, (long long)pending, (unsigned)(rec >> 20), (unsigned)(seed >> 20), (long long)left);
        }
        if (tick % 15 == 14) web_leak_report();   /* (RX_LEAK_SAMPLE) */
        if (tick % 5 != 4) continue;
        struct mallinfo m = {0};
        if (heap_use) m = mallinfo();
        const int stack_report = getenv("HOS_STACK_REPORT") != NULL;   /* (a scan of every stack) */
        if (stack_report) {
            size_t web_stack_high_water(int *threads, size_t *reserved);
            int nt = 0;
            size_t reserved = 0;
            const size_t worst = web_stack_high_water(&nt, &reserved);
            hos_log("web stacks: %d threads, %u MB reserved, deepest %u KB", nt, (unsigned)(reserved >> 20),
                    (unsigned)(worst >> 10));
        }
        hos_log("web memory: host heap top %u MB, in use %u MB%s; guest regions down to %u MB of %u",
                (unsigned)((uintptr_t)sbrk(0) >> 20), (unsigned)(m.uordblks >> 20), heap_use ? "" : " (HOS_WEB_HEAP=1)",
                (unsigned)(hos_region_floor() >> 20), (unsigned)(hos_space_end >> 20));
    }
    return 0;
}
void web_leak_init(void);
void hos_mem_checkpoint(const char *what) { (void)what; }   /* (hos_prof.c: HOS_MEM_TIMELINE, native only) */
int hos_sdl_pads(uint64_t *buttons, int32_t stick[4]) { (void)buttons; (void)stick; return 0; }   /* (pad_sdl.c) */
void hos_sdl_rumble(float low, float high) { (void)low; (void)high; }
void hos_prof_start(void) {
    web_leak_init();
    void web_movie_test(void);
    web_movie_test();
    CreateThread(NULL, 0, mem_main, NULL, 0, NULL);
    const char *s = getenv("HOS_DUMP_AFTER");
    if (s && atoi(s) > 0) CreateThread(NULL, 0, dump_main, (void *)(intptr_t)atoi(s), 0, NULL);
}

/* (the movie reader: movie_web.c) */

DWORD GetWindowThreadProcessId(HWND w, LPDWORD pid) {
    (void)w;
    if (pid) *pid = GetCurrentProcessId();
    return 1;
}
int localtime_s(struct tm *out, const time_t *t) { return localtime_r(t, out) ? 0 : 22; }
int gmtime_s(struct tm *out, const time_t *t) { return gmtime_r(t, out) ? 0 : 22; }
time_t _mkgmtime(struct tm *t) { return timegm(t); }
char *_strdup(const char *s) { return strdup(s); }

/* HOS_BIG_ALLOC_MB=<n> (debugging, web): every malloc of at least n MB is logged with the C stack that asked */
#include <emscripten.h>
static size_t big_alloc_floor;
void web_note_alloc(size_t n) {
    if (!big_alloc_floor) {
        const char *e = getenv("HOS_BIG_ALLOC_MB");
        big_alloc_floor = e && atoi(e) > 0 ? (size_t)atoi(e) << 20 : (size_t)-1;
    }
    if (n < big_alloc_floor) return;
    char stack[2048];
    emscripten_get_callstack(EM_LOG_C_STACK | EM_LOG_NO_PATHS, stack, sizeof stack);
    const char *rx_decoding_what(void);
    hos_log("web: %zu bytes allocation (decoding %s)\n%s", n, rx_decoding_what(), stack);
}

/* Allocation hooks (debugging, web): malloc itself, so C and C++ alike.
   - HOS_BIG_ALLOC_MB=<n>: allocations of at least n MB are logged with their stack (web_note_alloc above);
   - RX_LEAK_SAMPLE=1: about one allocation in 509 is kept with its stack until freed; every 30 s the stacks holding
     the most live sampled memory are logged (a leak's climbs). */
void *emscripten_builtin_malloc(size_t), emscripten_builtin_free(void *), *emscripten_builtin_calloc(size_t, size_t),
     *emscripten_builtin_realloc(void *, size_t);
#define LEAK_SLOTS 8192
typedef struct { void *p; size_t n; char site[200]; } LeakSample;
static LeakSample *leak_slots;
static volatile int leak_lock_word, leak_on = -1;
static volatile unsigned leak_counter;
static _Thread_local int leak_in_hook;
static void leak_acquire(void) { while (__atomic_exchange_n(&leak_lock_word, 1, __ATOMIC_ACQUIRE)) {} }
static void leak_release(void) { __atomic_store_n(&leak_lock_word, 0, __ATOMIC_RELEASE); }
static size_t leak_hash(void *p) { return ((uintptr_t)p >> 4) * 2654435761u % LEAK_SLOTS; }
static void leak_record(void *p, size_t n) {
    if (leak_in_hook || !p) return;
    leak_in_hook = 1;
    char stack[1024];
    emscripten_get_callstack(EM_LOG_C_STACK | EM_LOG_NO_PATHS, stack, sizeof stack);
    const char *q = stack;
    for (int skip = 0; skip < 3 && q; skip++) {   /* (past the hook's own frames) */
        q = strchr(q, '\n');
        if (q) q++;
    }
    leak_acquire();
    if (!leak_slots) leak_slots = (LeakSample *)emscripten_builtin_calloc(LEAK_SLOTS, sizeof(LeakSample));
    for (size_t i = leak_hash(p), k = 0; leak_slots && k < 64; k++, i = (i + 1) % LEAK_SLOTS)
        if (!leak_slots[i].p) {
            leak_slots[i].p = p;
            leak_slots[i].n = n;
            snprintf(leak_slots[i].site, sizeof leak_slots[i].site, "%s", q ? q : "?");
            for (char *c = leak_slots[i].site; *c; c++)
                if (*c == '\n') *c = '|';
            break;
        }
    leak_release();
    leak_in_hook = 0;
}
static void leak_forget(void *p) {
    if (!leak_slots || !p) return;
    leak_acquire();
    for (size_t i = leak_hash(p), k = 0; k < 64; k++, i = (i + 1) % LEAK_SLOTS)
        if (leak_slots[i].p == p) {
            leak_slots[i].p = NULL;
            break;
        }
    leak_release();
}
static inline int leak_sampling(void) { return leak_on == 1; }
void web_leak_init(void) { leak_on = getenv("RX_LEAK_SAMPLE") != NULL; }   /* (set by hos_prof_start: the environment is there) */
void *malloc(size_t n) {
    void *p = emscripten_builtin_malloc(n);
    if (!leak_in_hook && n >= (16u << 20)) web_note_alloc(n);
    if (leak_sampling() && __atomic_fetch_add(&leak_counter, 1, __ATOMIC_RELAXED) % 509 == 0) leak_record(p, n);
    return p;
}
void free(void *p) {
    if (leak_on == 1) leak_forget(p);
    emscripten_builtin_free(p);
}
void *calloc(size_t a, size_t b) {
    void *p = emscripten_builtin_calloc(a, b);
    if (leak_sampling() && __atomic_fetch_add(&leak_counter, 1, __ATOMIC_RELAXED) % 509 == 0) leak_record(p, a * b);
    return p;
}
void *realloc(void *p, size_t n) {
    if (leak_on == 1) leak_forget(p);
    void *q = emscripten_builtin_realloc(p, n);
    if (leak_sampling() && __atomic_fetch_add(&leak_counter, 1, __ATOMIC_RELAXED) % 509 == 0) leak_record(q, n);
    return q;
}
void web_leak_report(void) {
    if (leak_on != 1 || !leak_slots) return;
    typedef struct { char site[200]; size_t bytes, count; } Agg;
    static Agg agg[256];
    int nagg = 0;
    leak_acquire();
    for (int i = 0; i < LEAK_SLOTS; i++) {
        if (!leak_slots[i].p) continue;
        int j = 0;
        while (j < nagg && strcmp(agg[j].site, leak_slots[i].site)) j++;
        if (j == nagg && nagg < 256) {
            memcpy(agg[nagg].site, leak_slots[i].site, sizeof agg[nagg].site);
            agg[nagg].bytes = agg[nagg].count = 0;
            nagg++;
        }
        if (j < nagg) agg[j].bytes += leak_slots[i].n, agg[j].count++;
    }
    leak_release();
    for (int k = 0; k < 6 && k < nagg; k++) {   /* (the largest first; logged outside the lock: logging allocates) */
        int best = k;
        for (int j = k + 1; j < nagg; j++)
            if (agg[j].bytes > agg[best].bytes) best = j;
        Agg t = agg[k];
        agg[k] = agg[best];
        agg[best] = t;
        hos_log("leak sample #%d: %u KB in %u samples: %.190s", k + 1, (unsigned)(agg[k].bytes >> 10),
                (unsigned)agg[k].count, agg[k].site);
    }
}
