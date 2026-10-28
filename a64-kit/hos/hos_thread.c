/* Guest threads: one host thread each, with its own Ctx, TLS block and thread-local region.

   TPIDR_EL0 (TLS): a 4 KB block per thread, TP in its middle. TOTK's main reads its one thread-local at TP+0x10
   (local-exec, after AArch64's 16-byte TCB); the area below TP is zeroed room for whatever the SDK's musl keeps
   there. TPIDRRO_EL0 (thread-local region): 0x200 bytes; the SDK's nn::os keeps the current ThreadType* at
   +0x1F8, so lifted SDK code that reads it inline finds the same thread the HLE knows. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef __EMSCRIPTEN__
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>
#endif

struct HosThread {
    Ctx ctx;
    uint64_t guest_tt, entry, arg, stack, stack_size, id;
    uint64_t tls_block, tlr;
    int priority, ideal_core;
    uint64_t core_mask;
    uint64_t tls_slots[HOS_TLS_SLOTS];
    char name[48];
    HANDLE handle;
    int started;
    /* the kernel side (hos_svc.c): this thread's Horizon handle and the event it blocks on */
    uint32_t kernel_handle;
    HANDLE wake;
};

/* The SDK's own mutexes (InternalCriticalSection) tag a lock word with the owner's handle, read from
   ThreadType+0x1B0 through TLR+0x1F8 (sdk+0x13186c). Every thread gets a unique handle there. */
#define TT_HANDLE 0x1B0
static volatile LONG next_handle = 0x10000;
uint32_t hos_thread_handle(HosThread *t) { return t->kernel_handle; }

/* nnosInitialize gives the main thread the SDK's own ThreadType (it writes TLR+0x1F8): adopt it, so the HLE and the
   SDK agree on which object is the current thread. */
void hos_thread_adopt_guest(HosThread *t) {
    uint64_t tt = G64(t->tlr + 0x1F8);
    if (tt && tt != t->guest_tt) {
        t->guest_tt = tt;
        G32(tt + TT_HANDLE) = t->kernel_handle;
    }
}
int hos_thread_prio(HosThread *t) { return t->priority; }

int hos_trace;
extern const uint64_t hos_thread_start_hook, hos_thread_exit_hook;   /* gen/hos_image.c */
static _Thread_local HosThread *cur;
static SRWLOCK threads_lock = SRWLOCK_INIT;
static HosThread *threads[1024];
static int nthreads;
static volatile LONG64 next_id = 0x50;

HosThread *hos_self(void) { return cur; }
Ctx *hos_ctx(void) { return cur ? &cur->ctx : NULL; }
uint64_t hos_thread_guest(HosThread *t) { return t->guest_tt; }
uint64_t hos_thread_id(HosThread *t) { return t->id; }
const char *hos_thread_name(HosThread *t) { return t->name; }
uint64_t *hos_tls_slots(HosThread *t) { return t->tls_slots; }
/* The core a thread runs on. Most of TOTK's threads have no ideal core (-2, "use default") and a one-core affinity
   mask, so the mask decides; the engine keys per-core resources (its command memory arenas, for one) on
   GetCurrentCoreNumber, so threads pinned to different cores must report different ones. */
void hos_thread_affinity(HosThread *t, int ideal, uint64_t mask) {
    if (ideal >= 0) t->ideal_core = ideal;
    if (mask) t->core_mask = mask;
}
uint64_t hos_thread_mask(HosThread *t) { return t->core_mask; }
void hos_thread_boost(int level) {
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    static int off = -1;
    if (off < 0) off = getenv("HOS_NO_PRIORITIES") != NULL;
    if (!off) SetThreadPriority(GetCurrentThread(), level >= 2 ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_ABOVE_NORMAL);
#else
    (void)level;
#endif
}

int hos_cores(void) {
    static int n;
    if (!n) {
        const char *e = getenv("HOS_CORES");
        n = e && atoi(e) >= 1 && atoi(e) <= 8 ? atoi(e) : 3;
        if (!e && hos_game_cores()) n = hos_game_cores();   /* (totk_cores.c: the game set up for more cores) */
    }
    return n;
}
int hos_thread_core(HosThread *t) {
    if (!t) return 0;
    const int n = hos_cores();
    if (t->ideal_core >= 0) return t->ideal_core % n;
    /* the mask's lowest core, wrapped (a thread pinned past the cores there are shares a core number: the extra
       BaseProc executors, totk_cores.c) */
    return t->core_mask ? (int)(__builtin_ctzll(t->core_mask) % (unsigned)n) : 0;
}
void hos_thread_stack(HosThread *t, uint64_t *addr, uint64_t *size) { *addr = t->stack; *size = t->stack_size; }

int hos_thread_priority(HosThread *t, int set, int value) {
    int old = t->priority;
    if (set) t->priority = value;
    return old;
}

#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
static SRWLOCK log_lock = SRWLOCK_INIT, log_write_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE log_cv = CONDITION_VARIABLE_INIT;
static char *log_q[2048];
static int log_head, log_count;
static volatile LONG log_async = -1;

static int log_take_locked(char **out) {
    int n = 0;
    while (log_count) {
        out[n++] = log_q[log_head];
        log_head = (log_head + 1) % 2048;
        log_count--;
    }
    return n;
}

static void log_write(char **lines, int n) {
    for (int i = 0; i < n; i++) {
        fputs(lines[i], stderr);
        free(lines[i]);
    }
    fflush(stderr);
}

static DWORD WINAPI log_main(void *p) {
    (void)p;
    static char *batch[2048];
    for (;;) {
        AcquireSRWLockExclusive(&log_lock);
        while (!log_count) SleepConditionVariableSRW(&log_cv, &log_lock, INFINITE, 0);
        const int n = log_take_locked(batch);
        AcquireSRWLockExclusive(&log_write_lock);
        ReleaseSRWLockExclusive(&log_lock);
        log_write(batch, n);
        ReleaseSRWLockExclusive(&log_write_lock);
    }
}

static void log_flush(void) {
    if (log_async <= 0) return;
    static char *batch[2048];
    AcquireSRWLockExclusive(&log_write_lock);
    AcquireSRWLockExclusive(&log_lock);
    const int n = log_take_locked(batch);
    ReleaseSRWLockExclusive(&log_lock);
    log_write(batch, n);
    ReleaseSRWLockExclusive(&log_write_lock);
}

static LPTOP_LEVEL_EXCEPTION_FILTER log_prev_filter;

static LONG WINAPI log_crash_filter(EXCEPTION_POINTERS *e) {
    if (TryAcquireSRWLockExclusive(&log_write_lock)) {
        if (TryAcquireSRWLockExclusive(&log_lock)) {
            HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
            while (log_count) {
                const char *line = log_q[log_head];
                DWORD written;
                WriteFile(err, line, (DWORD)strlen(line), &written, NULL);
                log_head = (log_head + 1) % 2048;
                log_count--;
            }
            ReleaseSRWLockExclusive(&log_lock);
        }
        ReleaseSRWLockExclusive(&log_write_lock);
    }
    return log_prev_filter ? log_prev_filter(e) : EXCEPTION_CONTINUE_SEARCH;
}

static int log_queue(const char *line) {
    if (log_async < 0) {
        AcquireSRWLockExclusive(&log_lock);
        if (log_async < 0) {
            DWORD mode;
            int on = !getenv("HOS_LOG_SYNC") && GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode);
            if (on && !CreateThread(NULL, 0, log_main, NULL, 0, NULL)) on = 0;
            if (on) log_prev_filter = SetUnhandledExceptionFilter(log_crash_filter);
            log_async = on;
        }
        ReleaseSRWLockExclusive(&log_lock);
    }
    if (!log_async) return 0;
    char *s = _strdup(line);
    if (!s) return 0;
    AcquireSRWLockExclusive(&log_lock);
    if (log_count < 2048) {
        log_q[(log_head + log_count) % 2048] = s;
        log_count++;
        ReleaseSRWLockExclusive(&log_lock);
        WakeConditionVariable(&log_cv);
        return 1;
    }
    ReleaseSRWLockExclusive(&log_lock);
    free(s);
    log_flush();
    return 0;
}
#else
static int log_queue(const char *line) { (void)line; return 0; }
static void log_flush(void) {}
#endif

void hos_log_flush(void) { log_flush(); }

void hos_log(const char *fmt, ...) {
    char buf[1024], line[1100];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    snprintf(line, sizeof line, "[hos %s] %s\n", cur ? cur->name : "host", buf);
    if (log_queue(line)) return;
    fputs(line, stderr);
    fflush(stderr);
}

void hos_fatal(Ctx *c, const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!c) c = hos_ctx();
    log_flush();
    fprintf(stderr, "[hos %s] FATAL: %s", cur ? cur->name : "host", buf);
    if (c) fprintf(stderr, " (lr %#llx sp %#llx)", (unsigned long long)c->x[30], (unsigned long long)c->x[31]);
    fprintf(stderr, "\n");
    fflush(stderr);
    void hos_backtrace(void), hos_hostcall_stats(void);
    hos_backtrace();
    hos_hostcall_stats();
    ExitProcess(3);
}

/* Traps from the lifted code (rt/a64_rt.c): name the thread, then stop. */
void a64_on_trap(Ctx *c, uint64_t pc, uint32_t insn, const char *why) {
    hos_fatal(c, "%s at %#llx (insn %08x)", why, (unsigned long long)pc, insn);
}

static void setup_guest(HosThread *t) {
    t->tls_block = hos_sys_alloc(0x1000, 0x1000);
    t->tlr = hos_sys_alloc(0x200, 0x200);
    G64(t->tlr + 0x1F8) = t->guest_tt;
    t->kernel_handle = (uint32_t)InterlockedIncrement(&next_handle);
    G32(t->guest_tt + TT_HANDLE) = t->kernel_handle;
    t->wake = CreateEventA(NULL, FALSE, FALSE, NULL);   /* auto-reset: one signal, one wake-up */
    Ctx *c = &t->ctx;
    memset(c, 0, sizeof *c);
    c->tpidr_el0 = t->tls_block + 0x800;
    c->tpidrro_el0 = t->tlr;
    c->thread = t;
}

static void track(HosThread *t) {
    AcquireSRWLockExclusive(&threads_lock);
    if (nthreads < 1024) threads[nthreads++] = t;
    ReleaseSRWLockExclusive(&threads_lock);
}

/* Host handles of every running guest thread, for the profiler (hos_prof.c). */
int hos_thread_handles(HANDLE *out, const char **names, int max) {
    int n = 0;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads && n < max; i++)
        if (threads[i]->handle) { out[n] = threads[i]->handle; names[n] = threads[i]->name; n++; }
    ReleaseSRWLockShared(&threads_lock);
    return n;
}

/* HOS_FP_REPORT=1 (debugging): the FPCR / FPSR of the busiest game threads, with the thread stats */
int hos_thread_fp_report(char *out, int cap) {
    int len = 0;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads && len < cap - 64; i++) {
        const char *nm = threads[i]->name;
        if (strstr(nm, "Main") || strstr(nm, "Worker") || strstr(nm, "Executor"))
            len += snprintf(out + len, cap - len, " %s fpcr %llx fpsr %llx;", nm,
                            (unsigned long long)threads[i]->ctx.fpcr, (unsigned long long)threads[i]->ctx.fpsr);
    }
    ReleaseSRWLockShared(&threads_lock);
    return len;
}

HosThread *hos_thread_by_handle(uint32_t h) {
    HosThread *r = NULL;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads; i++)
        if (threads[i]->kernel_handle == h) { r = threads[i]; break; }
    ReleaseSRWLockShared(&threads_lock);
    return r;
}

/* Block on the thread's wake event (the kernel layer's lock must not be held). */
void hos_thread_wait(HosThread *t, int64_t timeout_ns, int *timed_out) {
    DWORD ms = timeout_ns < 0 ? INFINITE : (DWORD)((timeout_ns + 999999) / 1000000);
    *timed_out = WaitForSingleObject(t->wake, ms) == WAIT_TIMEOUT;
}
void hos_thread_wake(HosThread *t) { SetEvent(t->wake); }

HosThread *hos_thread_of(uint64_t guest_tt) {
    HosThread *r = NULL;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads; i++)
        if (threads[i]->guest_tt == guest_tt) { r = threads[i]; break; }
    ReleaseSRWLockShared(&threads_lock);
    return r;
}

Ctx *hos_boot_main_thread(uint64_t stack_size) {
    HosThread *t = (HosThread *)calloc(1, sizeof *t);
    t->guest_tt = hos_sys_alloc(0x400, 0x10);      /* a ThreadType for the main thread; the SDK never made one */
    t->stack_size = stack_size;
    t->stack = hos_sys_alloc(stack_size, 0x1000);
    t->id = InterlockedIncrement64(&next_id);
    t->priority = 44;                              /* main thread priority in TOTK's NPDM range */
    t->ideal_core = 0;
    t->started = 1;
    snprintf(t->name, sizeof t->name, "MainThread");
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &t->handle, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);   /* a real handle, for the profiler */
    hos_thread_pin(t->handle, t->name);
    setup_guest(t);
    t->ctx.x[31] = (t->stack + stack_size) & ~15ULL;
    cur = t;
    track(t);
    return &t->ctx;
}

static DWORD WINAPI thread_main(void *p) {
    HosThread *t = (HosThread *)p;
    cur = t;
    Ctx *c = &t->ctx;
    c->x[0] = t->arg;
    c->x[30] = 0;
    c->x[31] = (t->stack + t->stack_size) & ~15ULL;
    A64Fn f = a64_lookup(t->entry);
    if (!f) hos_fatal(c, "thread %s: entry %#llx is not a function", t->name, (unsigned long long)t->entry);
    if (hos_trace) hos_log("thread start: entry %#llx arg %#llx", (unsigned long long)t->entry, (unsigned long long)t->arg);
    /* What the SDK's own thread entry does before the thread function (see build.py thread_hooks): nnmusl's start
       hook gives the thread its TLS (thread pointer with the DTV at TP+8, every module's TLS block) and leaves the
       thread pointer at TLR+0x1F0. Without it every thread would share one DTV, and so every thread_local. */
    uint64_t hook = hos_thread_start_hook ? G64(hos_thread_start_hook) : 0;
    if (hook) {
        c->x[0] = t->guest_tt;
        a64_icall(c, hook);
        if (G64(t->tlr + 0x1F0)) c->tpidr_el0 = G64(t->tlr + 0x1F0);
        if (hos_trace || getenv("HOS_TLS_TRACE"))
            hos_log("thread %s: TLS hook gave TP %#llx, DTV %#llx; ideal core %d mask %#llx priority %d", t->name,
                    (unsigned long long)c->tpidr_el0, (unsigned long long)G64(c->tpidr_el0 + 8), t->ideal_core,
                    (unsigned long long)t->core_mask, t->priority);
        c->x[31] = (t->stack + t->stack_size) & ~15ULL;
    }
    c->x[0] = t->arg;
    c->x[30] = 0;
    f(c);
    hook = hos_thread_exit_hook ? G64(hos_thread_exit_hook) : 0;
    if (hook) {
        c->x[0] = t->guest_tt;
        a64_icall(c, hook);
    }
    if (hos_trace) hos_log("thread exit");
    return 0;
}

HosThread *hos_thread_create(uint64_t guest_tt, uint64_t entry, uint64_t arg, uint64_t stack, uint64_t stack_size,
                             int priority, int core) {
    HosThread *t = (HosThread *)calloc(1, sizeof *t);
    t->guest_tt = guest_tt;
    t->entry = entry;
    t->arg = arg;
    t->stack = stack;
    t->stack_size = stack_size;
    t->priority = priority;
    t->ideal_core = core;
    t->id = InterlockedIncrement64(&next_id);
    snprintf(t->name, sizeof t->name, "Thread_%llx", (unsigned long long)entry);
    setup_guest(t);   /* FPCR starts at 0, as the kernel does it; TOTK sets FZ itself (nnMain+0x474, thread entries) */
    track(t);
    return t;
}

void hos_thread_start(HosThread *t) {
    if (t->started) return;
    t->started = 1;
    /* Host stack: lifted code nests a C call per guest call, so give it plenty. */
    t->handle = CreateThread(NULL, 64 << 20, thread_main, t, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!t->handle) hos_fatal(NULL, "CreateThread failed for %s", t->name);
    hos_thread_pin(t->handle, t->name);
#ifdef __EMSCRIPTEN__
    { void web_thread_named(HANDLE t, const char *name); web_thread_named(t->handle, t->name); }
#endif
}

void hos_thread_join(HosThread *t) {
    if (t->handle) WaitForSingleObject(t->handle, INFINITE);
}

void hos_thread_destroy(HosThread *t) {
    AcquireSRWLockExclusive(&threads_lock);
    for (int i = 0; i < nthreads; i++)
        if (threads[i] == t) { threads[i] = threads[--nthreads]; break; }
    ReleaseSRWLockExclusive(&threads_lock);
    if (t->handle) CloseHandle(t->handle);
    free(t);
}

void hos_thread_set_name(HosThread *t, const char *name) {
    snprintf(t->name, sizeof t->name, "%s", name);
    if (t->handle) {
        wchar_t w[48];
        MultiByteToWideChar(CP_UTF8, 0, t->name, -1, w, 48);
        SetThreadDescription(t->handle, w);
        hos_thread_pin(t->handle, t->name);
    }
}

/* HOS_PIN=<name part>=<hex mask>,...[,*=<hex mask>] (experiment): host CPU affinity by thread name (game threads as
   they are named or started, host threads as they register); "*" for every other named thread. The first matching
   entry wins. (2026-10-20: the castle's MainThread and ModuleSystem workers each on a physical core of their own, the
   rest elsewhere, was slower than Windows' own scheduling: 63-68 against 69-71 fps.) */
/* HOS_PRIO=<name part>=<level>,... (experiment, same matching): host thread priority, -2 (lowest) to 2 (highest).
   (10-20: MainThread, ModuleSystem workers and BaseProc executors above normal: no change at the castle) */
typedef struct { char pat[40]; long long value; } NameSpec;
static int parse_name_specs(const char *e, NameSpec *out, int max, int base) {
    int n = 0;
    while (e && *e && n < max) {
        const char *eq = strchr(e, '='), *end = strchr(e, ',');
        if (!end) end = e + strlen(e);
        if (!eq || eq > end) break;
        snprintf(out[n].pat, sizeof out[n].pat, "%.*s", (int)(eq - e), e);
        out[n++].value = strtoll(eq + 1, NULL, base);
        e = *end ? end + 1 : end;
    }
    return n;
}
static const NameSpec *match_name_spec(const NameSpec *s, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(s[i].pat, "*") ? strstr(name, s[i].pat) != NULL : 1) return &s[i];
    return NULL;
}
void hos_thread_pin(void *h, const char *name) {
    static NameSpec pins[24], prios[24];
    static int npins = -1, nprios;
    static SRWLOCK lk = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lk);
    if (npins < 0) {
        npins = parse_name_specs(getenv("HOS_PIN"), pins, 24, 16);
        nprios = parse_name_specs(getenv("HOS_PRIO"), prios, 24, 10);
    }
    ReleaseSRWLockExclusive(&lk);
    const NameSpec *p = match_name_spec(pins, npins, name);
    if (p && p->value) SetThreadAffinityMask(h, (DWORD_PTR)p->value);
    static const int levels[5] = {THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL,
                                  THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_HIGHEST};
    if ((p = match_name_spec(prios, nprios, name)) && p->value >= -2 && p->value <= 2)
        SetThreadPriority(h, levels[p->value + 2]);
}

/* ---- time ------------------------------------------------------------------------------------------------ */
uint64_t hos_ticks(void) { return recomp_cntpct(NULL); }

int64_t hos_now_ns(void) {
    static LARGE_INTEGER freq;
    static int64_t ns_per_tick;
    LARGE_INTEGER now;
    if (!freq.QuadPart) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        ns_per_tick = 1000000000LL % f.QuadPart == 0 ? 1000000000LL / f.QuadPart : 0;
        freq = f;
    }
    QueryPerformanceCounter(&now);
    if (ns_per_tick) return (int64_t)now.QuadPart * ns_per_tick;
    return (int64_t)((now.QuadPart / freq.QuadPart) * 1000000000LL + (now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
}

void hos_sleep_ns(int64_t ns) {
    if (ns <= 0) { SwitchToThread(); return; }
    /* A high-resolution waitable timer (one per thread) sleeps to within ~0.5 ms without spinning; Sleep() rounds to
       the timer resolution, so without one the last millisecond is spun (threads that sleep in short steps, the
       vibration thread for one, used 15% of a core spinning). */
    static _Thread_local HANDLE timer;
    static _Thread_local int tried;
    int64_t until = hos_now_ns() + ns;
#ifdef __EMSCRIPTEN__
    /* The web has no high-resolution timer, and its clock is a call into JavaScript: spinning to the deadline below
       burned ~0.8 of a core across TOTK's sleeping threads (cores its game, render and WebGPU threads needed). A
       timed wait (a futex: Atomics.wait) to `slack` before the time, then the rest spun; slack follows how late the
       waits wake (HOS_WEB_SPIN_SLEEP=1: spun, as before). */
    static int web_spin = -1;
    if (web_spin < 0) web_spin = getenv("HOS_WEB_SPIN_SLEEP") != NULL;
    static _Atomic int64_t slack = 300000;
    const int64_t s = atomic_load_explicit(&slack, memory_order_relaxed);
    if (!web_spin && ns > s + 50000) {
        const int64_t want = until - s;
        usleep((useconds_t)((want - hos_now_ns()) / 1000));
        const int64_t late = hos_now_ns() - want;
        int64_t next = s + (late + 25000 - s) / 8;
        next = next < 50000 ? 50000 : next > 2000000 ? 2000000 : next;
        atomic_store_explicit(&slack, next, memory_order_relaxed);
    }
    while (hos_now_ns() < until) sched_yield();
    return;
#endif
    /* Under a millisecond: yielded away until the time (the timer's ~0.5 ms granularity made TOTK's 20 us waits for
       its job workers 25x longer, ~2000 a second on the frame's critical path). HOS_SHORT_SLEEP_TIMER=1: as before */
    static int short_timer = -1;
    if (short_timer < 0) short_timer = getenv("HOS_SHORT_SLEEP_TIMER") != NULL;
    if (ns < 1000000 && !short_timer) {
        while (hos_now_ns() < until)
            if (!SwitchToThread()) YieldProcessor();
        return;
    }
    if (!tried) {
        tried = 1;
        timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    }
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -(ns / 100 > 0 ? ns / 100 : 1);   /* relative, 100 ns units */
        if (SetWaitableTimerEx(timer, &due, 0, NULL, NULL, NULL, 0)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
    if (ns > 2000000) Sleep((DWORD)((ns - 1000000) / 1000000));
    while (hos_now_ns() < until) SwitchToThread();
}

/* ---- nn::os threads -------------------------------------------------------------------------------------- */
static HosThread *thread_arg(Ctx *c, uint64_t tt) {
    HosThread *t = hos_thread_of(tt);
    if (!t) hos_fatal(c, "unknown nn::os::ThreadType %#llx", (unsigned long long)tt);
    return t;
}

/* nn::os::CreateThread(ThreadType*, void (*)(void*), void* arg, void* stack, unsigned long size, int prio) */
HLE(_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi) {
    HOS_TRACE_CALL("CreateThread");
    hos_thread_create(ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), (int)ARG(5), -2);
    RET(0);
}

HLE(_ZN2nn2os11StartThreadEPNS0_10ThreadTypeE) {
    HOS_TRACE_CALL("StartThread");
    hos_thread_start(thread_arg(c, ARG(0)));
}

HLE(_ZN2nn2os10WaitThreadEPNS0_10ThreadTypeE) {
    HOS_TRACE_CALL("WaitThread");
    hos_thread_join(thread_arg(c, ARG(0)));
}

HLE(_ZN2nn2os13DestroyThreadEPNS0_10ThreadTypeE) {
    HOS_TRACE_CALL("DestroyThread");
    HosThread *t = thread_arg(c, ARG(0));
    hos_thread_join(t);
    hos_thread_destroy(t);
}

HLE(_ZN2nn2os13SetThreadNameEPNS0_10ThreadTypeEPKc) {
    HosThread *t = thread_arg(c, ARG(0));
    hos_thread_set_name(t, ARG(1) ? (const char *)GP(ARG(1)) : "");
    if (hos_trace) hos_log("SetThreadName %s", t->name);
}

HLE(_ZN2nn2os16GetCurrentThreadEv) { RET(cur->guest_tt); }
HLE(_ZN2nn2os11GetThreadIdEPKNS0_10ThreadTypeE) { RET(thread_arg(c, ARG(0))->id); }
HLE(_ZN2nn2os17GetThreadPriorityEPKNS0_10ThreadTypeE) { RET((uint32_t)hos_thread_priority(thread_arg(c, ARG(0)), 0, 0)); }
HLE(_ZN2nn2os20ChangeThreadPriorityEPNS0_10ThreadTypeEi) {
    RET((uint32_t)hos_thread_priority(thread_arg(c, ARG(0)), 1, (int)ARG(1)));
}

/* nn::os::SetThreadCoreMask(ThreadType*, int idealCore, unsigned long mask) */
HLE(_ZN2nn2os17SetThreadCoreMaskEPNS0_10ThreadTypeEim) {
    HosThread *t = thread_arg(c, ARG(0));
    hos_thread_affinity(t, (int)ARG(1), ARG(2));
}

HLE(_ZN2nn2os20GetCurrentCoreNumberEv) { RET((uint32_t)hos_thread_core(cur)); }
HLE(_ZN2nn2os26GetThreadAvailableCoreMaskEv) { RET((1u << hos_cores()) - 1); }   /* cores 0-2: what an application gets */

HLE(_ZN2nn2os19GetCurrentStackInfoEPmS1_) {
    if (ARG(0)) G64(ARG(0)) = cur->stack;
    if (ARG(1)) G64(ARG(1)) = cur->stack_size;
}

HLE(_ZN2nn2os11YieldThreadEv) { SwitchToThread(); }
/* HOS_SLEEP_LOG=1 (diagnostic): guest sleeps by caller (LR) every 5 s: count, asked, slept */
void hos_sleep_logged(int64_t ns, uint64_t lr) {
    /* HOS_POLL_SLEEP_NS=<ns> (experiment): a guest sleep of up to 100 us lasts at most that long. TOTK's job system
       waits for its workers by sleeping 20 us and looking again (~90 times a frame at the castle, on MainThread and
       the workers): a shorter step notices the job's end sooner (the loop looks again; nothing else changes).
       Measured 2026-10-21 (castle, 2000 ns, 3 rounds): no change. */
    static int64_t poll_cap = -1;
    if (poll_cap < 0) {
        const char *e = getenv("HOS_POLL_SLEEP_NS");
        poll_cap = e ? atoll(e) : 0;
    }
    if (poll_cap && ns > poll_cap && ns <= 100000) ns = poll_cap;
    static int on = -1;
    if (on < 0) on = getenv("HOS_SLEEP_LOG") != NULL;
    if (!on) { hos_sleep_ns(ns); return; }
    const int64_t t0 = hos_now_ns();
    hos_sleep_ns(ns);
    const int64_t took = hos_now_ns() - t0;
    static SRWLOCK lk = SRWLOCK_INIT;
    static struct { uint64_t lr; int64_t n, asked, slept; } tab[64];
    static int64_t last;
    AcquireSRWLockExclusive(&lk);
    int i = 0;
    while (i < 64 && tab[i].lr && tab[i].lr != lr) i++;
    if (i < 64) {
        tab[i].lr = lr;
        tab[i].n++;
        tab[i].asked += ns > 0 ? ns : 0;
        tab[i].slept += took;
    }
    const int64_t now = hos_now_ns();
    if (now - last > 5000000000LL) {
        last = now;
        for (int k = 0; k < 64 && tab[k].lr; k++)
            if (tab[k].n)
                hos_log("sleep: lr %#llx: %lld calls, asked %.2f ms, slept %.2f ms (%.1f us each)",
                        (unsigned long long)tab[k].lr, (long long)tab[k].n, tab[k].asked / 1e6, tab[k].slept / 1e6,
                        tab[k].slept / 1e3 / tab[k].n);
        memset(tab, 0, sizeof tab);
    }
    ReleaseSRWLockExclusive(&lk);
}

HLE(_ZN2nn2os11SleepThreadENS_8TimeSpanE) { hos_sleep_logged((int64_t)ARG(0), c->x[30]); }

/* ---- TLS slots --------------------------------------------------------------------------------------------- */
static volatile LONG slot_used[HOS_TLS_SLOTS];

/* nn::os::AllocateTlsSlot(TlsSlot*, void (*)(unsigned long)) -> Result */
HLE(_ZN2nn2os15AllocateTlsSlotEPNS0_7TlsSlotEPFvmE) {
    for (int i = 0; i < HOS_TLS_SLOTS; i++) {
        if (!InterlockedCompareExchange(&slot_used[i], 1, 0)) {
            G32(ARG(0)) = (uint32_t)i;
            RET(0);
            return;
        }
    }
    RET(0x2C01);   /* os::ResultOutOfResource */
}

HLE(_ZN2nn2os11FreeTlsSlotENS0_7TlsSlotE) { if ((uint32_t)ARG(0) < HOS_TLS_SLOTS) slot_used[(uint32_t)ARG(0)] = 0; }
HLE(_ZN2nn2os11GetTlsValueENS0_7TlsSlotE) { RET((uint32_t)ARG(0) < HOS_TLS_SLOTS ? cur->tls_slots[(uint32_t)ARG(0)] : 0); }
HLE(_ZN2nn2os11SetTlsValueENS0_7TlsSlotEm) { if ((uint32_t)ARG(0) < HOS_TLS_SLOTS) cur->tls_slots[(uint32_t)ARG(0)] = ARG(1); }

/* ---- ticks ------------------------------------------------------------------------------------------------- */
HLE(_ZN2nn2os13GetSystemTickEv) { RET(hos_ticks()); }
HLE(_ZN2nn2os22GetSystemTickFrequencyEv) { RET(HOS_TICK_HZ); }
/* 19.2 MHz: ticks = ns * 12 / 625 */
HLE(_ZN2nn2os13ConvertToTickENS_8TimeSpanE) {
    int64_t ns = (int64_t)ARG(0);
    RET((ns / 625) * 12 + (ns % 625) * 12 / 625);
}
HLE(_ZN2nn2os17ConvertToTimeSpanENS0_4TickE) {
    int64_t t = (int64_t)ARG(0);
    RET((t / 12) * 625 + (t % 12) * 625 / 12);
}

/* ---- fibers: imported, never created by the game so far ---------------------------------------------------- */
HLE(_ZN2nn2os15GetCurrentFiberEv) { RET(0); }
HLE(_ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE) { hos_fatal(c, "SwitchToFiber(%#llx): fibers are not implemented", (unsigned long long)ARG(0)); }

/* Every thread's kernel handle (the tag its SDK mutexes hold), for hos_kernel_dump. */
void hos_thread_log_handles(void) {
    AcquireSRWLockShared(&threads_lock);
    /* (lr: the return address of the thread's last call, x0 its first argument: in a wait, which game code waits on
       what; the web has no stack walk) */
    for (int i = 0; i < nthreads; i++)
        hos_log("kernel:   handle %#x = %s: lr %#llx x0 %#llx x19 %#llx x20 %#llx x21 %#llx x22 %#llx",
                threads[i]->kernel_handle, threads[i]->name, (unsigned long long)threads[i]->ctx.x[30],
                (unsigned long long)threads[i]->ctx.x[0], (unsigned long long)threads[i]->ctx.x[19],
                (unsigned long long)threads[i]->ctx.x[20], (unsigned long long)threads[i]->ctx.x[21],
                (unsigned long long)threads[i]->ctx.x[22]);
    ReleaseSRWLockShared(&threads_lock);
}
