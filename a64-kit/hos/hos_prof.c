/* Sampling profiler and hang finder (Windows): HOS_PROFILE=<seconds> samples every guest thread's host instruction
   pointer each millisecond for that long, then prints the hottest functions with their guest addresses (a lifted
   function f_XXXXXXXX is guest code at XXXXXXXX; anything else is the runtime or the OS layer). A spinning thread
   shows up at the top. Needs the PDB the native build writes next to the exe. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>

int hos_thread_handles(HANDLE *out, const char **names, int max);   /* hos_thread.c */
uint8_t *hos_guest_base(void);                                      /* hos_main.c */
uint64_t hos_guest_size(void);
int rx_guest_page_uploaded(uint64_t off);                           /* gpu/render/rx_buffer.cpp */

void hos_process_memory(uint64_t *private_bytes, uint64_t *working_set) {
    typedef struct {
        DWORD cb, PageFaultCount;
        SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
            QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage;
    } Counters;
    typedef BOOL(WINAPI * Fn)(HANDLE, Counters *, DWORD);
    static Fn fn;
    static int looked;
    if (!looked) {
        looked = 1;
        fn = (Fn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo");
    }
    Counters m;
    memset(&m, 0, sizeof m);
    m.cb = sizeof m;
    if (!fn || !fn(GetCurrentProcess(), &m, sizeof m)) memset(&m, 0, sizeof m);
    *private_bytes = m.PrivateUsage;
    *working_set = m.WorkingSetSize;
}

/* HOS_MEM_MAP=1: where the working set is, logged with each mem: line. Walks the address space and asks which pages
   are resident (QueryWorkingSetEx), by kind: the guest window (and its resident MB per 256 MB), the exe, DLLs, other
   private memory (host heaps, driver staging) and mapped views (GPU memory the driver maps in). */
void hos_memory_map_log(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("HOS_MEM_MAP"); on = e && *e && *e != '0'; }
    if (!on) return;
    typedef struct { void *va; ULONG_PTR attr; } WsEx;
    typedef BOOL(WINAPI * QFn)(HANDLE, void *, DWORD);
    static QFn q;
    if (!q) q = (QFn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32QueryWorkingSetEx");
    if (!q) return;
    enum { K_GUEST, K_EXE, K_DLL, K_PRIVATE, K_MAPPED, K_N };
    static const char *names[K_N] = {"guest", "exe", "dlls", "other private", "mapped"};
    uint64_t commit[K_N] = {0}, res[K_N] = {0}, shared[K_N] = {0};
    uint64_t guest_band[16] = {0}, guest_zero[16] = {0};
    uint64_t zero_uploaded = 0, uploaded = 0;
    struct { uint64_t base, size, res; } big[8] = {{0}};   /* the largest resident private allocations */
    uint64_t cur_alloc = 0, cur_size = 0, cur_res = 0;
    const uint64_t g0 = (uint64_t)(uintptr_t)hos_guest_base(), g1 = g0 + hos_guest_size();
    const HMODULE exe = GetModuleHandleA(NULL);
    enum { BATCH = 65536 };
    WsEx *ws = (WsEx *)malloc(BATCH * sizeof(WsEx));
    if (!ws) return;
    MEMORY_BASIC_INFORMATION mbi;
    for (uint8_t *p = NULL; VirtualQuery(p, &mbi, sizeof mbi) == sizeof mbi; p = (uint8_t *)mbi.BaseAddress + mbi.RegionSize) {
        if ((uint8_t *)mbi.BaseAddress + mbi.RegionSize <= p) break;
        if (mbi.State != MEM_COMMIT) continue;
        const uint64_t base = (uint64_t)(uintptr_t)mbi.BaseAddress;
        int k = base >= g0 && base < g1 ? K_GUEST
              : mbi.Type == MEM_IMAGE ? (mbi.AllocationBase == (void *)exe ? K_EXE : K_DLL)
              : mbi.Type == MEM_MAPPED ? K_MAPPED : K_PRIVATE;
        commit[k] += mbi.RegionSize;
        if (k == K_PRIVATE && (uint64_t)(uintptr_t)mbi.AllocationBase != cur_alloc) {
            /* a new allocation: keep the previous one if it is among the biggest */
            int m = 0;
            for (int i = 1; i < 8; i++) if (big[i].res < big[m].res) m = i;
            if (cur_res > big[m].res) { big[m].base = cur_alloc; big[m].size = cur_size; big[m].res = cur_res; }
            cur_alloc = (uint64_t)(uintptr_t)mbi.AllocationBase;
            cur_size = cur_res = 0;
        }
        if (k == K_PRIVATE) cur_size += mbi.RegionSize;
        for (uint64_t off = 0; off < mbi.RegionSize;) {
            DWORD n = 0;
            for (; n < BATCH && off + (uint64_t)n * 4096 < mbi.RegionSize; n++)
                ws[n].va = (void *)(uintptr_t)(base + off + (uint64_t)n * 4096), ws[n].attr = 0;
            if (q(GetCurrentProcess(), ws, n * (DWORD)sizeof(WsEx)))
                for (DWORD i = 0; i < n; i++)
                    if (ws[i].attr & 1) {
                        res[k] += 4096;
                        if (ws[i].attr & 0x8000) shared[k] += 4096;
                        if (k == K_PRIVATE) cur_res += 4096;
                        if (k == K_GUEST) {
                            const uint64_t band = ((uint64_t)(uintptr_t)ws[i].va - g0) >> 28;
                            if (band < 16) guest_band[band] += 4096;
                            const uint64_t *w = (const uint64_t *)ws[i].va;
                            int zero = 1;
                            for (int j = 0; j < 512 && zero; j++) zero = w[j] == 0;
                            if (zero && band < 16) guest_zero[band] += 4096;
                            if (rx_guest_page_uploaded((uint64_t)(uintptr_t)ws[i].va - g0)) {
                                uploaded += 4096;
                                if (zero) zero_uploaded += 4096;
                            }
                        }
                    }
            off += (uint64_t)n * 4096;
        }
    }
    free(ws);
    char line[512];
    int len = 0;
    for (int k = 0; k < K_N; k++)
        len += snprintf(line + len, sizeof line - len, "%s%s %llu MB resident (%llu shared) of %llu committed",
                        k ? ", " : "", names[k], (unsigned long long)(res[k] >> 20),
                        (unsigned long long)(shared[k] >> 20), (unsigned long long)(commit[k] >> 20));
    hos_log("mem map: %s", line);
    len = 0;
    for (int b = 0; b < 16; b++)
        if (guest_band[b] >> 20)
            len += snprintf(line + len, sizeof line - len, " %#llx:%llu", (unsigned long long)b << 28,
                            (unsigned long long)(guest_band[b] >> 20));
    hos_log("mem map: guest resident MB per 256 MB:%s", len ? line : " none");
    len = 0;
    uint64_t zero_total = 0;
    for (int b = 0; b < 16; b++) {
        zero_total += guest_zero[b];
        if (guest_zero[b] >> 20)
            len += snprintf(line + len, sizeof line - len, " %#llx:%llu", (unsigned long long)b << 28,
                            (unsigned long long)(guest_zero[b] >> 20));
    }
    hos_log("mem map: guest resident pages that are all zero: %llu MB (%llu MB of them uploaded to the GPU, of %llu MB "
            "uploaded):%s", (unsigned long long)(zero_total >> 20), (unsigned long long)(zero_uploaded >> 20),
            (unsigned long long)(uploaded >> 20), len ? line : "");
    len = 0;
    line[0] = 0;
    for (int i = 0; i < 8; i++)
        if (big[i].res >> 20)
            len += snprintf(line + len, sizeof line - len, " %#llx:%llu/%llu", (unsigned long long)big[i].base,
                            (unsigned long long)(big[i].res >> 20), (unsigned long long)(big[i].size >> 20));
    hos_log("mem map: biggest private allocations (resident/committed MB):%s", line);
    /* the heaps' busy blocks by size (power-of-two classes): who fills the host heap */
    HANDLE heaps[64];
    const DWORD nh = GetProcessHeaps(64, heaps);
    for (DWORD h = 0; h < nh && h < 64; h++) {
        uint64_t cnt[40] = {0}, bytes[40] = {0}, total = 0;
        PROCESS_HEAP_ENTRY e;
        memset(&e, 0, sizeof e);
        if (!HeapLock(heaps[h])) continue;
        while (HeapWalk(heaps[h], &e))
            if (e.wFlags & PROCESS_HEAP_ENTRY_BUSY) {
                int b = 0;
                while (b < 39 && (1ull << (b + 1)) <= e.cbData) b++;
                cnt[b]++;
                bytes[b] += e.cbData;
                total += e.cbData;
            }
        HeapUnlock(heaps[h]);
        if (total < (16ull << 20)) continue;
        len = 0;
        for (int b = 0; b < 40; b++)
            if (bytes[b] >= (4ull << 20))
                len += snprintf(line + len, sizeof line - len, " %lluB+:%llux%lluMB", 1ull << b, (unsigned long long)cnt[b],
                                (unsigned long long)(bytes[b] >> 20));
        hos_log("mem map: heap %p: %llu MB in busy blocks:%s", (void *)heaps[h], (unsigned long long)(total >> 20), line);
    }
}

/* HOS_MEM_TIMELINE=1: every second, how much of the guest window is resident (per 256 MB band): when memory is
   touched, to find who touches it (then HOS_PROFILE around that moment). */
static DWORD WINAPI mem_timeline_main(void *arg) {
    (void)arg;
    typedef struct { void *va; ULONG_PTR attr; } WsEx;
    typedef BOOL(WINAPI * QFn)(HANDLE, void *, DWORD);
    QFn q = (QFn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32QueryWorkingSetEx");
    if (!q) return 0;
    enum { BATCH = 65536 };
    WsEx *ws = (WsEx *)malloc(BATCH * sizeof(WsEx));
    const uint64_t g0 = (uint64_t)(uintptr_t)hos_guest_base(), pages = hos_guest_size() / 4096;
    const int64_t t0 = hos_now_ns();
    for (;;) {
        Sleep(1000);
        uint64_t band[16] = {0};
        for (uint64_t p = 0; p < pages; p += BATCH) {
            const DWORD n = (DWORD)(pages - p < BATCH ? pages - p : BATCH);
            for (DWORD i = 0; i < n; i++) ws[i].va = (void *)(uintptr_t)(g0 + (p + i) * 4096), ws[i].attr = 0;
            if (!q(GetCurrentProcess(), ws, n * (DWORD)sizeof(WsEx))) continue;
            for (DWORD i = 0; i < n; i++)
                if (ws[i].attr & 1) band[((p + i) * 4096) >> 28 & 15]++;
        }
        char line[400];
        int len = 0;
        uint64_t total = 0;
        for (int b = 0; b < 16; b++) {
            total += band[b];
            if (band[b] >> 8) len += snprintf(line + len, sizeof line - len, " %x:%llu", b, (unsigned long long)(band[b] >> 8));
        }
        hos_log("mem timeline %.0f s: guest %llu MB resident (MB per 256 MB band:%s)", (hos_now_ns() - t0) / 1e9,
                (unsigned long long)(total >> 8), line);
    }
}
/* (HOS_MEM_TIMELINE) how much of the heap is resident when a boot step starts */
void hos_mem_checkpoint(const char *what) {
    const char *e = getenv("HOS_MEM_TIMELINE");
    if (!e || *e != '1') return;
    typedef struct { void *va; ULONG_PTR attr; } WsEx;
    typedef BOOL(WINAPI * QFn)(HANDLE, void *, DWORD);
    QFn q = (QFn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32QueryWorkingSetEx");
    if (!q) return;
    enum { BATCH = 65536 };
    WsEx *ws = (WsEx *)malloc(BATCH * sizeof(WsEx));
    const uint64_t g0 = (uint64_t)(uintptr_t)hos_guest_base(), lo = 0x10000000ULL / 4096, hi = 0x93800000ULL / 4096;
    uint64_t res = 0;
    for (uint64_t p = lo; p < hi; p += BATCH) {
        const DWORD n = (DWORD)(hi - p < BATCH ? hi - p : BATCH);
        for (DWORD i = 0; i < n; i++) ws[i].va = (void *)(uintptr_t)(g0 + (p + i) * 4096), ws[i].attr = 0;
        if (q(GetCurrentProcess(), ws, n * (DWORD)sizeof(WsEx)))
            for (DWORD i = 0; i < n; i++) res += ws[i].attr & 1;
    }
    free(ws);
    hos_log("mem checkpoint before %s: heap %llu MB resident", what, (unsigned long long)(res >> 8));
}
void hos_mem_timeline_start(void) {
    const char *e = getenv("HOS_MEM_TIMELINE");
    if (e && *e == '1') CreateThread(NULL, 0, mem_timeline_main, NULL, 0, NULL);
}

/* Host threads of the runtime (the GPU thread, pipeline compilers, ...): named for HOS_THREAD_STATS. */
static struct { HANDLE h; char name[40]; } host_threads[32];
static volatile LONG nhost_threads;
void hos_host_thread_register_self(const char *name) {
    LONG i = InterlockedIncrement(&nhost_threads) - 1;
    if (i >= 32) return;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &host_threads[i].h, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);
    snprintf(host_threads[i].name, sizeof host_threads[i].name, "%s", name);
    hos_thread_pin(GetCurrentThread(), name);
}

typedef struct { DWORD64 addr; char name[96]; unsigned hits, self; } Hit;

static int cmp_hits(const void *a, const void *b) { return (int)((const Hit *)b)->hits - (int)((const Hit *)a)->hits; }
static int cmp_self(const void *a, const void *b) { return (int)((const Hit *)b)->self - (int)((const Hit *)a)->self; }

/* Function start and name of a code address (cached: symbol lookups are slow). */
typedef struct { DWORD64 pc, start; int hit; } PcCache;
static PcCache pc_cache[1 << 16];
static Hit *hits;
static int nhits;
enum { MAXH = 1 << 16 };
static int hit_slot[1 << 17];   /* function start -> hits index + 1 (open addressing: MAXH entries, half full) */
/* dbghelp, also searching the exe's folder (where wgpu_native.pdb goes, next to its DLL) */
static void sym_init(HANDLE proc) {
    char path[MAX_PATH + 8] = ".;";
    GetModuleFileNameA(NULL, path + 2, MAX_PATH);
    char *slash = strrchr(path, '\\');
    if (slash) *slash = 0;
    SymInitialize(proc, path, TRUE);
}

/* Rust's legacy mangling (wgpu): ZN<len><part>...17h<hash>E -> the last three parts, an impl block as {impl} */
static void rust_short(const char *m, char *out, size_t cap) {
    const char *p = m + (m[0] == '_') + 2;
    const char *part[32];
    int plen[32], n = 0;
    while (*p >= '0' && *p <= '9' && n < 32) {
        int len = 0;
        while (*p >= '0' && *p <= '9') len = len * 10 + (*p++ - '0');
        if ((int)strnlen(p, len) < len) break;
        if (!(len == 17 && p[0] == 'h')) { part[n] = p; plen[n] = len; n++; }
        p += len;
    }
    if (!n) { snprintf(out, cap, "%s", m); return; }
    size_t o = 0;
    for (int i = n > 3 ? n - 3 : 0; i < n && o + 1 < cap; i++) {
        if (o && o + 2 < cap) { out[o++] = ':'; out[o++] = ':'; }
        if (strstr(part[i], "impl") && strstr(part[i], "impl") < part[i] + plen[i]) {
            o += snprintf(out + o, cap - o, "{impl}");
            continue;
        }
        for (int k = 0; k < plen[i] && o + 1 < cap; k++) {
            const char *c = part[i] + k;
            if (*c == '$') {
                const char *e = memchr(c + 1, '$', plen[i] - k - 1);
                if (e) {
                    char r = !strncmp(c, "$LT$", 4) ? '<' : !strncmp(c, "$GT$", 4) ? '>' : !strncmp(c, "$u20$", 5) ? ' '
                             : !strncmp(c, "$RF$", 4) ? '&' : !strncmp(c, "$C$", 3) ? ',' : '?';
                    out[o++] = r;
                    k += (int)(e - c);
                    continue;
                }
            }
            if (c[0] == '.' && c[1] == '.' && k + 1 < plen[i]) { out[o++] = ':'; if (o + 1 < cap) out[o++] = ':'; k++; continue; }
            out[o++] = *c;
        }
    }
    out[o < cap ? o : cap - 1] = 0;
}

static int hit_index(HANDLE proc, DWORD64 pc, const char *thread) {
    PcCache *e = &pc_cache[(pc ^ pc >> 16) & 0xFFFF];
    if (e->pc == pc) return e->hit;
    char symbuf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *sym = (SYMBOL_INFO *)symbuf;
    memset(symbuf, 0, sizeof symbuf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0, start = pc;
    char name[96] = "?";
    if (SymFromAddr(proc, pc, &disp, sym)) {
        start = sym->Address;
        if (!strncmp(sym->Name, "ZN", 2) || !strncmp(sym->Name, "_ZN", 3)) rust_short(sym->Name, name, sizeof name);
        else snprintf(name, sizeof name, "%s", sym->Name);
    }
    unsigned h = (unsigned)((start >> 4) * 0x9E3779B1u) >> 15;
    while (hit_slot[h] && hits[hit_slot[h] - 1].addr != start) h = (h + 1) & ((1u << 17) - 1);
    int j;
    if (hit_slot[h]) {
        j = hit_slot[h] - 1;
    } else {
        if (nhits == MAXH) return -1;
        j = nhits++;
        hits[j].addr = start;
        snprintf(hits[j].name, sizeof hits[j].name, "%s [%s]", name, thread);
        hit_slot[h] = j + 1;
    }
    e->pc = pc;
    e->hit = j;
    return j;
}

static uint8_t stack_copy[512 << 10];
static DWORD64 copy_lo, copy_hi;
static SRWLOCK copy_lock = SRWLOCK_INIT;   /* the profiler and the exit dump share the copy */
/* StackWalk64's memory reads: the stack from the copy, anything else (code, unwind data) from the process */
static BOOL CALLBACK read_copy(HANDLE proc, DWORD64 addr, PVOID out, DWORD n, LPDWORD got) {
    if (addr >= copy_lo && addr + n <= copy_hi) {
        memcpy(out, stack_copy + (addr - copy_lo), n);
        *got = n;
        return TRUE;
    }
    SIZE_T r = 0;
    BOOL ok = ReadProcessMemory(proc, (void *)addr, out, n, &r);
    *got = (DWORD)r;
    return ok;
}

/* HOS_PROFILE=<seconds> (after HOS_PROFILE_DELAY=<seconds>). With HOS_PROFILE_THREAD=<name part>, only the threads
   whose name contains it are sampled, with their whole call stacks: inclusive time per function (what is under
   draw(), flush_batch(), ...) as well as the leaf ("self") time. HOS_PROFILE_CALLERS=<name part> (with
   HOS_PROFILE_THREAD) also counts the three callers above the first frame whose name contains it (who allocates,
   who reads guest memory, ...). HOS_PROFILE_EXCLUDE=<name part> leaves out the samples under such a frame (what
   the thread does besides that). HOS_PROFILE_LINES=<name part>: the self samples in such functions by source line. */
typedef struct { char chain[300]; unsigned n; } Caller;
typedef struct { char where[200]; unsigned n; } Line;   /* HOS_PROFILE_LINES: a source line and its self samples */
static Line lines[512];
static int nlines;
static int cmp_lines(const void *a, const void *b) { return (int)((const Line *)b)->n - (int)((const Line *)a)->n; }   /* HOS_PROFILE_CALLERS: a call chain and its samples */
static Caller callers[256];
static int ncallers;
#define MAXPCS (1u << 21)
static DWORD64 *pcs, *pcs_fn;   /* HOS_PROFILE_PCS: the self samples' addresses and their functions' */
static unsigned npcs;
static int cmp_callers(const void *a, const void *b) { return (int)((const Caller *)b)->n - (int)((const Caller *)a)->n; }
static DWORD WINAPI prof_main(void *p) {
    int seconds = (int)(intptr_t)p;
    const char *delay = getenv("HOS_PROFILE_DELAY"), *only = getenv("HOS_PROFILE_THREAD");
    const char *callers_of = only ? getenv("HOS_PROFILE_CALLERS") : NULL;   /* (with HOS_PROFILE_THREAD) */
    const char *exclude = only ? getenv("HOS_PROFILE_EXCLUDE") : NULL;      /* samples under that frame: skipped */
    const char *lines_of = only ? getenv("HOS_PROFILE_LINES") : NULL;       /* self samples there, by source line */
    if (lines_of) SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (delay) Sleep((DWORD)atoi(delay) * 1000);
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    sym_init(proc);
    hits = (Hit *)calloc(MAXH, sizeof *hits);
    /* HOS_PROFILE_PCS=<file>: every self sample's host instruction address and its function's start, as offsets in
       the exe (other modules': "abs <address>"), then "fn <start> <name>" lines: with llvm-objdump, which
       instructions of the hot lifted code cost */
    const char *pcs_file = getenv("HOS_PROFILE_PCS");
    if (pcs_file) pcs = (DWORD64 *)malloc(MAXPCS * sizeof *pcs), pcs_fn = (DWORD64 *)malloc(MAXPCS * sizeof *pcs_fn);
    unsigned total = 0;
    int64_t end = hos_now_ns() + (int64_t)seconds * 1000000000LL;
    hos_log("profile: sampling %s for %d s", only ? only : "every thread", seconds);
    extern volatile long hos_prof_gate;   /* (nn_os_sync.c HOS_PROFILE_GATE) */
    const int gated = getenv("HOS_PROFILE_GATE") != NULL;
    while (hos_now_ns() < end) {
        if (gated && !__atomic_load_n(&hos_prof_gate, __ATOMIC_ACQUIRE)) {   /* (waits for the gate, spinning) */
            SwitchToThread();
            continue;
        }
        HANDLE th[256];
        const char *names[256];
        int n = hos_thread_handles(th, names, 224);
        for (int i = 0; i < (nhost_threads < 32 ? (int)nhost_threads : 32) && n < 256; i++) {
            th[n] = host_threads[i].h;
            names[n++] = host_threads[i].name;
        }
        for (int i = 0; i < n; i++) {
            if (only && !strstr(names[i], only)) continue;
            AcquireSRWLockExclusive(&copy_lock);
            if (SuspendThread(th[i]) == (DWORD)-1) { ReleaseSRWLockExclusive(&copy_lock); continue; }
            CONTEXT ctx;
            ctx.ContextFlags = only ? CONTEXT_FULL : CONTEXT_CONTROL;
            int ok = GetThreadContext(th[i], &ctx);
            /* Only the registers and a copy of the stack are taken while the thread is stopped: walking it there
               (dbghelp allocates and takes locks the stopped thread may hold) deadlocked the profiler. */
            size_t copied = 0;
            if (ok && only) {
                MEMORY_BASIC_INFORMATION mbi;
                if (VirtualQuery((void *)ctx.Rsp, &mbi, sizeof mbi)) {
                    size_t avail = (size_t)((uint8_t *)mbi.BaseAddress + mbi.RegionSize - (uint8_t *)ctx.Rsp);
                    copied = avail < sizeof stack_copy ? avail : sizeof stack_copy;
                    memcpy(stack_copy, (void *)ctx.Rsp, copied);
                }
            }
            ResumeThread(th[i]);
            if (!ok) { ReleaseSRWLockExclusive(&copy_lock); continue; }
            DWORD64 stack[64];
            int depth = 0;
            stack[depth++] = ctx.Rip;
            if (only && copied) {
                copy_lo = ctx.Rsp;
                copy_hi = ctx.Rsp + copied;
                STACKFRAME64 f;
                memset(&f, 0, sizeof f);
                f.AddrPC.Offset = ctx.Rip; f.AddrPC.Mode = AddrModeFlat;
                f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
                f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
                for (int d = 0; d < 80 && depth < 64; d++) {
                    if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, th[i], &f, &ctx, read_copy, SymFunctionTableAccess64,
                                     SymGetModuleBase64, NULL) || !f.AddrPC.Offset)
                        break;
                    if (d > 0) stack[depth++] = f.AddrPC.Offset;
                }
            }
            ReleaseSRWLockExclusive(&copy_lock);
            if (exclude) {
                int skip = 0;
                for (int d = 0; d < depth && !skip; d++) {
                    const int j = hit_index(proc, stack[d], names[i]);
                    skip = j >= 0 && strstr(hits[j].name, exclude) != NULL;
                }
                if (skip) { total++; continue; }   /* (counted: percentages stay of all samples) */
            }
            if (callers_of && depth > 1) {   /* HOS_PROFILE_CALLERS: who calls the first frame matching it */
                for (int d = 0; d < depth - 1; d++) {
                    const int j = hit_index(proc, stack[d], names[i]);
                    if (j < 0 || !strstr(hits[j].name, callers_of)) continue;
                    char chain[300];
                    chain[0] = 0;
                    for (int u = d + 1; u < depth && u <= d + 3; u++) {
                        const int k = hit_index(proc, stack[u], names[i]);
                        if (k < 0) break;
                        size_t len = strlen(chain);
                        snprintf(chain + len, sizeof chain - len, "%s%.90s", len ? " < " : "", hits[k].name);
                    }
                    int c;
                    for (c = 0; c < ncallers && strcmp(callers[c].chain, chain); c++) {}
                    if (c == ncallers && ncallers < 256) { snprintf(callers[c].chain, sizeof callers[c].chain, "%s", chain); ncallers++; }
                    if (c < ncallers) callers[c].n++;
                    break;
                }
            }
            if (lines_of && depth) {
                const int j = hit_index(proc, stack[0], names[i]);
                if (j >= 0 && strstr(hits[j].name, lines_of)) {
                    IMAGEHLP_LINE64 li;
                    DWORD disp = 0;
                    memset(&li, 0, sizeof li);
                    li.SizeOfStruct = sizeof li;
                    char where[200];
                    if (SymGetLineFromAddr64(proc, stack[0], &disp, &li)) {
                        const char *f = strrchr(li.FileName, '\\');
                        snprintf(where, sizeof where, "%s:%lu", f ? f + 1 : li.FileName, (unsigned long)li.LineNumber);
                    } else {
                        snprintf(where, sizeof where, "(no line) %#llx", (unsigned long long)stack[0]);
                    }
                    int k;
                    for (k = 0; k < nlines && strcmp(lines[k].where, where); k++) {}
                    if (k == nlines && nlines < 512) { snprintf(lines[k].where, sizeof lines[k].where, "%s", where); nlines++; }
                    if (k < nlines) lines[k].n++;
                }
            }
            if (pcs && npcs < MAXPCS && depth) {
                const int j = hit_index(proc, stack[0], names[i]);
                pcs[npcs] = stack[0], pcs_fn[npcs++] = j >= 0 ? hits[j].addr : 0;
            }
            int seen[64], nseen = 0;
            for (int d = 0; d < depth; d++) {
                const int j = hit_index(proc, stack[d], names[i]);
                if (j < 0) continue;
                if (d == 0) hits[j].self++;
                int dup = 0;
                for (int k = 0; k < nseen; k++) dup |= seen[k] == j;
                if (dup) continue;
                seen[nseen++] = j;
                hits[j].hits++;
            }
            total++;
        }
        Sleep(1);
    }
    fprintf(stderr, "[hos profile] %u samples over %d s (guest address = the f_ name)\n", total, seconds);
    if (only) {
        qsort(hits, nhits, sizeof *hits, cmp_hits);
        fprintf(stderr, "[hos profile] inclusive:\n");
        for (int i = 0; i < nhits && i < 70; i++)
            fprintf(stderr, "  %6u  %5.1f%%  %s\n", hits[i].hits, 100.0 * hits[i].hits / (total ? total : 1), hits[i].name);
    }
    if (lines_of) {
        qsort(lines, nlines, sizeof *lines, cmp_lines);
        fprintf(stderr, "[hos profile] lines of %s (self samples):\n", lines_of);
        for (int i = 0; i < nlines && i < 40; i++)
            fprintf(stderr, "  %6u  %5.1f%%  %s\n", lines[i].n, 100.0 * lines[i].n / (total ? total : 1), lines[i].where);
    }
    if (callers_of) {
        qsort(callers, ncallers, sizeof *callers, cmp_callers);
        fprintf(stderr, "[hos profile] callers of %s:\n", callers_of);
        for (int i = 0; i < ncallers && i < 30; i++)
            fprintf(stderr, "  %6u  %5.1f%%  %s\n", callers[i].n, 100.0 * callers[i].n / (total ? total : 1), callers[i].chain);
    }
    qsort(hits, nhits, sizeof *hits, cmp_self);
    /* HOS_PROFILE_ORDER=<file>: every lifted function that was sampled, hottest first, "name samples" a line (samples: self + on-stack hits): sorted and stripped, a link order
       (build.py's RX_LINK_ORDER) that packs the hot code together (the exe is ~290 MB of lifted code) */
    if (getenv("HOS_PROFILE_ORDER")) {
        FILE *f = fopen(getenv("HOS_PROFILE_ORDER"), "w");
        int n = 0;
        for (int i = 0; f && i < nhits; i++) {
            if (strncmp(hits[i].name, "f_", 2)) continue;
            char fn[32];
            if (sscanf(hits[i].name, "%31[^ ]", fn) == 1) fprintf(f, "%s\n", fn), n++;
        }
        if (f) fclose(f);
        fprintf(stderr, "[hos profile] %d lifted functions written to %s\n", n, getenv("HOS_PROFILE_ORDER"));
    }
    if (pcs) {
        FILE *f = fopen(pcs_file, "w");
        const uint8_t *base = (const uint8_t *)GetModuleHandleW(NULL);
        const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)(base + ((const IMAGE_DOS_HEADER *)base)->e_lfanew);
        const DWORD64 lo = (DWORD64)(uintptr_t)base, hi = lo + nt->OptionalHeader.SizeOfImage;
        for (unsigned i = 0; f && i < npcs; i++) {
            if (pcs[i] >= lo && pcs[i] < hi && pcs_fn[i] >= lo)
                fprintf(f, "%llx %llx\n", (unsigned long long)(pcs[i] - lo), (unsigned long long)(pcs_fn[i] - lo));
            else fprintf(f, "abs %llx\n", (unsigned long long)pcs[i]);
        }
        for (int i = 0; f && i < nhits; i++)   /* the functions: "fn <start offset> <name [thread]>" */
            if (hits[i].addr >= lo && hits[i].addr < hi) fprintf(f, "fn %llx %s\n", (unsigned long long)(hits[i].addr - lo), hits[i].name);
        if (f) fclose(f);
        fprintf(stderr, "[hos profile] %u sample addresses written to %s\n", npcs, pcs_file);
    }
    fprintf(stderr, "[hos profile] self:\n");
    for (int i = 0; i < nhits && i < 50; i++)
        fprintf(stderr, "  %6u  %5.1f%%  %s\n", hits[i].self, 100.0 * hits[i].self / (total ? total : 1), hits[i].name);
    {   /* where the samples were, by module: the game's main, the SDK (NVN's driver is in it), the rest */
        unsigned m_main = 0, m_sdk = 0, m_wait = 0, m_host = 0;
        for (int i = 0; i < nhits; i++) {
            const char *nm = hits[i].name;
            unsigned long long a = 0;
            if (!strncmp(nm, "f_", 2) && sscanf(nm + 2, "%llx", &a) == 1) {
                if (a < 0xc700000) m_main += hits[i].self;
                else m_sdk += hits[i].self;
            } else if (!strncmp(nm, "Zw", 2) || !strncmp(nm, "Nt", 2) || strstr(nm, "Wait") || strstr(nm, "Sleep")) {
                m_wait += hits[i].self;
            } else {
                m_host += hits[i].self;
            }
        }
        const unsigned busy = m_main + m_sdk + m_host;
        fprintf(stderr, "[hos profile] busy samples: game code (main) %u (%.0f%%), SDK incl. NVN %u (%.0f%%), host %u (%.0f%%); "
                "waiting %u\n", m_main, 100.0 * m_main / (busy ? busy : 1), m_sdk, 100.0 * m_sdk / (busy ? busy : 1),
                m_host, 100.0 * m_host / (busy ? busy : 1), m_wait);
    }
    fflush(stderr);
    if (getenv("HOS_PROFILE_EXIT")) {
        hos_log_flush();
        ExitProcess(4);
    }
    return 0;
}

/* The guest call stack, from the host one: lifted code makes one C call per guest call, so every f_XXXXXXXX frame on
   the host stack is a guest function at XXXXXXXX. Printed by hos_fatal. */
void hos_backtrace(void) {
    static int ready;
    HANDLE proc = GetCurrentProcess();
    if (!ready) {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        sym_init(proc);
        ready = 1;
    }
    void *frames[64];
    USHORT n = RtlCaptureStackBackTrace(1, 64, frames, NULL);
    char buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *sym = (SYMBOL_INFO *)buf;
    fprintf(stderr, "[hos] backtrace (newest first; f_XXXXXXXX = guest code at XXXXXXXX):\n");
    for (USHORT i = 0; i < n; i++) {
        memset(buf, 0, sizeof buf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        IMAGEHLP_LINE64 line = {sizeof line};
        DWORD ldisp = 0;
        /* (the line of the call: a return address is the instruction after it) */
        const int has_line = SymGetLineFromAddr64(proc, (DWORD64)frames[i] - (i ? 1 : 0), &ldisp, &line);
        if (SymFromAddr(proc, (DWORD64)frames[i], &disp, sym))
            fprintf(stderr, "  %2u  %s +%#llx%s%s:%lu\n", i, sym->Name, (unsigned long long)disp, has_line ? "  " : "",
                    has_line ? line.FileName : "", has_line ? (unsigned long)line.LineNumber : 0ul);
        else fprintf(stderr, "  %2u  %p\n", i, frames[i]);
    }
    fflush(stderr);
}

/* Every guest thread's call stack, from its suspended host context: where each one runs or waits. A lifted guest
   function shows as f_XXXXXXXX; the frames above it are the runtime / OS layer (a wait shows as the HLE that
   blocks). */
void hos_dump_threads(void) {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    sym_init(proc);
    HANDLE th[256];
    const char *names[256];
    int n = hos_thread_handles(th, names, 256);
    char buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *sym = (SYMBOL_INFO *)buf;
    for (int i = 0; i < n; i++) {
        if (GetThreadId(th[i]) == GetCurrentThreadId()) continue;
        AcquireSRWLockExclusive(&copy_lock);
        if (SuspendThread(th[i]) == (DWORD)-1) { ReleaseSRWLockExclusive(&copy_lock); continue; }
        CONTEXT ctx;
        ctx.ContextFlags = CONTEXT_FULL;
        if (!GetThreadContext(th[i], &ctx)) { ResumeThread(th[i]); ReleaseSRWLockExclusive(&copy_lock); continue; }
        /* a copy of the stack, then the thread runs again before dbghelp walks it (a stopped thread holding the heap
           lock deadlocked the walk) */
        copy_lo = copy_hi = 0;
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery((void *)ctx.Rsp, &mbi, sizeof mbi)) {
            size_t avail = (size_t)((uint8_t *)mbi.BaseAddress + mbi.RegionSize - (uint8_t *)ctx.Rsp);
            size_t copied = avail < sizeof stack_copy ? avail : sizeof stack_copy;
            memcpy(stack_copy, (void *)ctx.Rsp, copied);
            copy_lo = ctx.Rsp;
            copy_hi = ctx.Rsp + copied;
        }
        ResumeThread(th[i]);
        STACKFRAME64 f;
        memset(&f, 0, sizeof f);
        f.AddrPC.Offset = ctx.Rip; f.AddrPC.Mode = AddrModeFlat;
        f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
        f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
        fprintf(stderr, "[hos] thread %s:", names[i]);
        int shown = 0;
        for (int d = 0; d < 48 && shown < 14; d++) {
            if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, th[i], &f, &ctx, read_copy, SymFunctionTableAccess64,
                             SymGetModuleBase64, NULL) || !f.AddrPC.Offset)
                break;
            memset(buf, 0, sizeof buf);
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            sym->MaxNameLen = 255;
            DWORD64 disp = 0;
            if (!SymFromAddr(proc, f.AddrPC.Offset, &disp, sym)) continue;
            if (!strncmp(sym->Name, "Nt", 2) || !strncmp(sym->Name, "Zw", 2) || strstr(sym->Name, "SleepConditionVariable")
                || strstr(sym->Name, "WaitForSingleObject") || strstr(sym->Name, "WaitForMultipleObjects")
                || strstr(sym->Name, "RtlSleep") || strstr(sym->Name, "SleepEx"))
                continue;   /* the OS wait itself: the frame below it says which */
            fprintf(stderr, " %s", sym->Name);
            shown++;
        }
        ReleaseSRWLockExclusive(&copy_lock);
        fprintf(stderr, "\n");
    }
    fflush(stderr);
}

/* HOS_EXIT_AFTER=<seconds>: end the run cleanly after that long (host-call stats and every thread's stack
   included), for scripted runs. */
static DWORD WINAPI exit_main(void *p) {
    Sleep((DWORD)(intptr_t)p * 1000);
    void hos_hostcall_stats(void);
    hos_log("HOS_EXIT_AFTER: %d s elapsed, exiting", (int)(intptr_t)p);
    hos_hostcall_stats();
    hos_dump_threads();
    void hos_kernel_dump(void);
    hos_kernel_dump();
    hos_log_flush();
    ExitProcess(0);
    return 0;
}

typedef struct { double pct; const char *name; } Busy;
static int cmp_busy(const void *a, const void *b) {
    const double d = ((const Busy *)b)->pct - ((const Busy *)a)->pct;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* HOS_THREAD_STATS=1: every 5 s, the busiest threads (guest and runtime) by CPU time: which one limits the frame
   rate. */
static DWORD WINAPI thread_stats_main(void *p) {
    (void)p;
    enum { MAXT = 320 };
    static struct { HANDLE h; uint64_t cpu; } prev[MAXT];
    int nprev = 0;
    for (;;) {
        Sleep(5000);
        HANDLE th[MAXT];
        const char *names[MAXT];
        int n = hos_thread_handles(th, names, 256);
        const int nh = nhost_threads < 32 ? (int)nhost_threads : 32;
        for (int i = 0; i < nh && n < MAXT; i++) { th[n] = host_threads[i].h; names[n] = host_threads[i].name; n++; }
        Busy busy[MAXT];
        static struct { HANDLE h; uint64_t cpu; } now[MAXT];
        double total = 0;
        for (int i = 0; i < n; i++) {
            FILETIME c, e, k, u;
            uint64_t cpu = 0;
            if (GetThreadTimes(th[i], &c, &e, &k, &u))
                cpu = ((uint64_t)k.dwHighDateTime << 32 | k.dwLowDateTime) + ((uint64_t)u.dwHighDateTime << 32 | u.dwLowDateTime);
            uint64_t before = cpu;
            for (int j = 0; j < nprev; j++) if (prev[j].h == th[i]) { before = prev[j].cpu; break; }
            now[i].h = th[i];
            now[i].cpu = cpu;
            busy[i].pct = (double)(cpu - before) / 5e7 * 100.0;   /* 100 ns units over 5 s */
            busy[i].name = names[i];
            total += busy[i].pct;
        }
        memcpy(prev, now, sizeof(now[0]) * n);
        nprev = n;
        qsort(busy, n, sizeof *busy, cmp_busy);
        char line[1024];
        int len = snprintf(line, sizeof line, "threads: %.0f%% of a core in all;", total);
        for (int i = 0; i < n && i < 12 && busy[i].pct >= 1.0; i++)
            len += snprintf(line + len, sizeof line - len, " %s %.0f%%,", busy[i].name, busy[i].pct);
        hos_log("%s", line);
        if (getenv("HOS_FP_REPORT")) {
            int hos_thread_fp_report(char *out, int cap);
            char fp[2048];
            hos_thread_fp_report(fp, sizeof fp);
            hos_log("fp state:%s", fp);
        }
    }
    return 0;
}

/* HOS_AB=<ms> (measurement on a busy machine): an A/B inside one run. The switch named by HOS_AB_SWITCH (rx:
   ab_named(), wgpu's TOTK patch switches) is flipped every <ms>, from HOS_AB_DELAY=<s> (default 95) on, once the
   frames draw like the world's and after a warm-up (below). Per period: the frames presented and the CPU time of the
   runtime's threads (renderer: NVN replay, GPU submit / finish, compilers, ...) and the game's, the first quarter of
   the period (up to 400 ms) left out for the frames in flight; every 20 s the per-phase medians and the median change
   between neighbouring periods are logged. Whatever else loads the machine loads both phases alike, where two runs in
   turn differed by 20 fps. Trust it after ~100 neighbours (the early medians moved by tens of percent). */
volatile LONG hos_ab_phase;
int hos_ab(void) { return (int)hos_ab_phase; }
extern volatile LONG64 hos_present_count;
typedef struct { HANDLE h; uint64_t cpu; int host; } AbThread;
static int ab_snapshot(AbThread *out, int max) {
    HANDLE th[256];
    const char *names[256];
    int n = hos_thread_handles(th, names, 224), k = 0;
    for (int i = 0; i < n && k < max; i++) out[k++] = (AbThread){th[i], 0, 0};
    for (int i = 0; i < (nhost_threads < 32 ? (int)nhost_threads : 32) && k < max; i++)
        out[k++] = (AbThread){host_threads[i].h, 0, 1};
    for (int i = 0; i < k; i++) {
        FILETIME c, e, kt, u;
        if (GetThreadTimes(out[i].h, &c, &e, &kt, &u))
            out[i].cpu = ((uint64_t)kt.dwHighDateTime << 32 | kt.dwLowDateTime) + ((uint64_t)u.dwHighDateTime << 32 | u.dwLowDateTime);
    }
    return k;
}
static int ab_cmp(const void *x, const void *y) {
    const double a = *(const double *)x, b = *(const double *)y;
    return a < b ? -1 : a > b;
}
static double ab_median(const double *v, int n) {
    static double tmp[4096];
    if (n <= 0) return 0;
    memcpy(tmp, v, sizeof *v * (size_t)n);
    qsort(tmp, (size_t)n, sizeof *tmp, ab_cmp);
    return n & 1 ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2;
}
static DWORD WINAPI ab_main(void *p) {
    const int period = (int)(intptr_t)p, settle = period / 4 < 400 ? period / 4 : 400;
    const char *sw = getenv("HOS_AB_SWITCH");
    Sleep((DWORD)(getenv("HOS_AB_DELAY") ? atoi(getenv("HOS_AB_DELAY")) : 95) * 1000);
    /* then until the frames have drawn HOS_AB_MIN_DRAWS (default 1500) each for 5 s: in the world (a busy machine
       loads it late; title and loading frames in the totals would hide the difference) */
    int64_t rx_stat_draws(void);
    const int64_t min_draws = getenv("HOS_AB_MIN_DRAWS") ? atoll(getenv("HOS_AB_MIN_DRAWS")) : 1500;
    for (int good = 0; good < 5;) {
        const int64_t d0 = rx_stat_draws(), f0 = hos_present_count;
        Sleep(1000);
        const int64_t df = hos_present_count - f0;
        good = df > 0 && (rx_stat_draws() - d0) / df >= min_draws ? good + 1 : 0;
    }
    /* and HOS_AB_WARMUP more seconds (default 30): the first half minute in the world translates programs and makes
       pipelines, which moved the early medians by tens of percent */
    Sleep((DWORD)(getenv("HOS_AB_WARMUP") ? atoi(getenv("HOS_AB_WARMUP")) : 30) * 1000);
    hos_log("ab: %s flips every %d ms from now on", sw ? sw : "(no HOS_AB_SWITCH)", period);
    /* per period: fps, runtime and game CPU ms a frame; reported as medians per phase and as the median change
       between neighbouring periods (a hitch or another job's burst moves one or two periods, not the medians) */
    enum { MAXP = 4096 };
    static double v[2][3][MAXP], pair[3][MAXP];
    int n[2] = {0, 0}, npair = 0, prev_ph = -1;
    double prev[3] = {0};
    static AbThread a[300], b[300];
    int64_t last_log = hos_now_ns();
    for (;;) {
        const int ph = (int)hos_ab_phase;
        Sleep((DWORD)settle);
        const int na = ab_snapshot(a, 300);
        const int64_t t0 = hos_now_ns(), f0 = hos_present_count;
        Sleep((DWORD)(period - settle));
        const int nb = ab_snapshot(b, 300);
        const int64_t t1 = hos_now_ns(), f1 = hos_present_count;
        InterlockedExchange(&hos_ab_phase, !ph);
        double host_ms = 0, game_ms = 0;
        for (int i = 0; i < nb; i++)
            for (int j = 0; j < na; j++)
                if (a[j].h == b[i].h) {
                    const double ms = (double)(b[i].cpu - a[j].cpu) / 1e4;   /* (100 ns units) */
                    if (b[i].host) host_ms += ms;
                    else game_ms += ms;
                    break;
                }
        const double frames = (double)(f1 - f0);
        if (frames < 1 || n[ph] >= MAXP || npair >= MAXP) continue;
        const double cur[3] = {frames / ((t1 - t0) / 1e9), host_ms / frames, game_ms / frames};
        for (int k = 0; k < 3; k++) v[ph][k][n[ph]] = cur[k];
        n[ph]++;
        if (prev_ph >= 0 && prev_ph != ph) {   /* flipped / as set - 1 */
            for (int k = 0; k < 3; k++) pair[k][npair] = ph ? cur[k] / prev[k] - 1 : prev[k] / cur[k] - 1;
            npair++;
        }
        prev_ph = ph;
        memcpy(prev, cur, sizeof prev);
        if (t1 - last_log >= 20000000000LL && n[0] && n[1] && npair) {
            last_log = t1;
            hos_log("ab: %s medians: as set %.1f fps, runtime %.2f ms/frame, game %.2f | flipped %.1f fps, runtime %.2f, "
                    "game %.2f | flipped vs as set, median of %d neighbours: fps %+.1f%%, runtime %+.1f%%, game %+.1f%%",
                    sw ? sw : "", ab_median(v[0][0], n[0]), ab_median(v[0][1], n[0]), ab_median(v[0][2], n[0]),
                    ab_median(v[1][0], n[1]), ab_median(v[1][1], n[1]), ab_median(v[1][2], n[1]), npair,
                    100 * ab_median(pair[0], npair), 100 * ab_median(pair[1], npair), 100 * ab_median(pair[2], npair));
        }
    }
    return 0;
}

/* HOS_MEMDUMP=<hex guest address>:<hex length>@<seconds>[,<seconds>...] (debugging): that guest memory is written to
   memdump_<seconds>.bin at those times after start. Diffing two dumps shows which fields of an object change when
   the game does something (a menu cursor moving, ...). The address may be a guest pointer read at dump time:
   *<hex address>+<hex offset> (e.g. a global holding the object). */
static DWORD WINAPI memdump_main(void *p) {
    (void)p;
    const char *spec = getenv("HOS_MEMDUMP");
    char addr_part[64] = {0};
    const char *at = strchr(spec, '@');
    if (!at || at - spec >= (ptrdiff_t)sizeof addr_part) return 0;
    memcpy(addr_part, spec, (size_t)(at - spec));
    char *colon = strchr(addr_part, ':');
    if (!colon) return 0;
    *colon = 0;
    const uint64_t len = strtoull(colon + 1, NULL, 16);
    const int64_t t0 = hos_now_ns();
    for (const char *t = at + 1; *t;) {
        char *end;
        const double secs = strtod(t, &end);
        if (end == t) break;
        t = *end == ',' ? end + 1 : end;
        const int64_t when = t0 + (int64_t)(secs * 1e9);
        while (hos_now_ns() < when) Sleep(5);
        uint64_t addr;
        if (addr_part[0] == '*') {   /* *<global>+<offset> */
            char *plus;
            const uint64_t g = strtoull(addr_part + 1, &plus, 16);
            addr = G64(g) + (*plus == '+' ? strtoull(plus + 1, NULL, 16) : 0);
        } else {
            addr = strtoull(addr_part, NULL, 16);
        }
        char path[64];
        snprintf(path, sizeof path, "memdump_%g.bin", secs);
        FILE *f = fopen(path, "wb");
        if (f && addr) {
            fwrite(GP(addr), 1, (size_t)len, f);
            hos_log("memdump: %#llx + %#llx -> %s", (unsigned long long)addr, (unsigned long long)len, path);
        }
        if (f) fclose(f);
    }
    return 0;
}

/* Function-entry tracing (debugging): lifted code instrumented to call hos_trace_fn(address, c->x) at entry
   (TOTK/ui_work/instrument_all.py) logs each function's first HOS_TRACE_MAX (6) calls between HOS_TRACE_FROM and
   HOS_TRACE_TO seconds after start, with x0-x3 and the return address. HOS_TRACE_NEW=1: only functions never called
   before that window (the code an action runs for the first time: a menu's decide handler, ...). */
void hos_trace_fn(uint32_t addr, const uint64_t *x) {
    static int64_t t0, from, to;
    static int max = -1, only_new;
    static volatile LONG counts[1 << 22];
    static volatile char before[1 << 22];
    if (max < 0) {
        t0 = hos_now_ns();
        from = getenv("HOS_TRACE_FROM") ? (int64_t)(atof(getenv("HOS_TRACE_FROM")) * 1e9) : 0;
        to = getenv("HOS_TRACE_TO") ? (int64_t)(atof(getenv("HOS_TRACE_TO")) * 1e9) : INT64_MAX;
        only_new = getenv("HOS_TRACE_NEW") != NULL;
        max = getenv("HOS_TRACE_MAX") ? atoi(getenv("HOS_TRACE_MAX")) : 6;
    }
    const int64_t t = hos_now_ns() - t0;
    const uint32_t slot = (addr >> 2) & ((1 << 22) - 1);
    if (t < from) {
        if (only_new) before[slot] = 1;
        return;
    }
    if (t > to || (only_new && before[slot])) return;
    if (InterlockedIncrement(&counts[slot]) > max) return;
    hos_log("T %.3f %08x x0=%llx x1=%llx x2=%llx x3=%llx lr=%llx", t / 1e9, addr, (unsigned long long)x[0],
            (unsigned long long)x[1], (unsigned long long)x[2], (unsigned long long)x[3], (unsigned long long)x[30]);
}

/* HOS_POKE=<hex address>=<hex byte>[,...]@<seconds> (debugging): writes those guest bytes at that time. */
static DWORD WINAPI poke_main(void *p) {
    (void)p;
    const char *spec = getenv("HOS_POKE"), *at = strchr(spec, '@');
    if (!at) return 0;
    const int64_t when = hos_now_ns() + (int64_t)(atof(at + 1) * 1e9);
    while (hos_now_ns() < when) Sleep(5);
    for (const char *s = spec; s < at;) {
        char *end;
        const uint64_t a = strtoull(s, &end, 16);
        if (*end != '=') break;
        const uint64_t v = strtoull(end + 1, &end, 16);
        *(uint8_t *)GP(a) = (uint8_t)v;
        hos_log("poke: %#llx = %#x", (unsigned long long)a, (unsigned)v);
        s = *end == ',' ? end + 1 : end;
    }
    return 0;
}

/* A crash in host code (the renderer, an HLE service): what faulted where, with the stack, before the process goes
   (a guest fault is reported by the guest's own handler; this one only sees what nothing handled) */
static LONG WINAPI crash_filter(EXCEPTION_POINTERS *e) {
    static volatile LONG once;
    if (InterlockedExchange(&once, 1)) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD *r = e->ExceptionRecord;
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    sym_init(proc);
    char buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *sym = (SYMBOL_INFO *)buf;
    memset(buf, 0, sizeof buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    const int named = SymFromAddr(proc, (DWORD64)r->ExceptionAddress, &disp, sym);
    fprintf(stderr, "[hos] crash: exception %#lx at %p (%s +%#llx) in host thread %lu", (unsigned long)r->ExceptionCode,
            r->ExceptionAddress, named ? sym->Name : "?", (unsigned long long)disp, (unsigned long)GetCurrentThreadId());
    if (r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r->NumberParameters >= 2)
        fprintf(stderr, ": %s %#llx", r->ExceptionInformation[0] == 1 ? "write to" : r->ExceptionInformation[0] == 8 ? "execute" : "read of",
                (unsigned long long)r->ExceptionInformation[1]);
    {
        uint8_t *hos_guest_base(void);
        uint64_t hos_guest_size(void);
        fprintf(stderr, " (guest memory: %p..%p)", (void *)hos_guest_base(), (void *)(hos_guest_base() + hos_guest_size()));
    }
    fputc('\n', stderr);
    hos_backtrace();
    return EXCEPTION_CONTINUE_SEARCH;
}

/* HOS_STALL_MS=<ms> (diagnostic): when no frame has been presented for that long (a hitch), the game threads (Main,
   ModuleSystem, BaseProc executors) and the NVN replay and GPU submit threads are stack-sampled every ~2 ms until
   frames come again; then one report per hitch: its length and each thread's most frequent functions over it
   (inclusive, % of that thread's samples). The first 30 hitches after HOS_STALL_AFTER=<s> (default 70). What a
   whole-run profile averages away. */
extern volatile LONG64 hos_present_count;
static DWORD WINAPI stall_main(void *p) {
    const int64_t limit = (int64_t)(intptr_t)p * 1000000LL;
    const char *after_s = getenv("HOS_STALL_AFTER");
    const int64_t start = hos_now_ns() + (after_s ? atoll(after_s) : 70) * 1000000000LL;
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    AcquireSRWLockExclusive(&copy_lock);
    if (!hits) {
        sym_init(proc);
        hits = (Hit *)calloc(MAXH, sizeof *hits);
    }
    ReleaseSRWLockExclusive(&copy_lock);
    enum { NT = 12 };
    static unsigned short counts[NT][MAXH];   /* per sampled thread, inclusive samples per function (hits index) */
    unsigned tsamples[NT];
    const char *tnames[NT];
    int reports = 0;
    LONG64 last = hos_present_count;
    int64_t last_t = hos_now_ns();
    while (reports < 30) {
        Sleep(2);
        const LONG64 now_count = hos_present_count;
        const int64_t now = hos_now_ns();
        if (now_count != last) { last = now_count, last_t = now; continue; }
        if (now < start || now - last_t < limit) continue;
        /* a hitch: sample until the next present */
        memset(counts, 0, sizeof counts);
        memset(tsamples, 0, sizeof tsamples);
        int nt = 0;
        const int64_t t0 = last_t;
        while (hos_present_count == last && hos_now_ns() - t0 < 5000000000LL) {
            HANDLE th[256];
            const char *names[256];
            int n = hos_thread_handles(th, names, 224);
            for (int i = 0; i < (nhost_threads < 32 ? (int)nhost_threads : 32) && n < 256; i++) {
                th[n] = host_threads[i].h;
                names[n++] = host_threads[i].name;
            }
            for (int i = 0; i < n; i++) {
                const char *nm = names[i];
                if (!strstr(nm, "Main") && !strstr(nm, "ModuleSystem") && !strstr(nm, "BaseProcExecutor") &&
                    !strstr(nm, "replay") && !strstr(nm, "GPU submit") && !strstr(nm, "write scan"))
                    continue;
                int k;
                for (k = 0; k < nt && strcmp(tnames[k], nm); k++) {}
                if (k == nt) {
                    if (nt == NT) continue;
                    tnames[nt++] = nm;
                }
                AcquireSRWLockExclusive(&copy_lock);
                if (SuspendThread(th[i]) == (DWORD)-1) { ReleaseSRWLockExclusive(&copy_lock); continue; }
                CONTEXT ctx;
                ctx.ContextFlags = CONTEXT_FULL;
                int ok = GetThreadContext(th[i], &ctx);
                size_t copied = 0;
                if (ok) {
                    MEMORY_BASIC_INFORMATION mbi;
                    if (VirtualQuery((void *)ctx.Rsp, &mbi, sizeof mbi)) {
                        size_t avail = (size_t)((uint8_t *)mbi.BaseAddress + mbi.RegionSize - (uint8_t *)ctx.Rsp);
                        copied = avail < sizeof stack_copy ? avail : sizeof stack_copy;
                        memcpy(stack_copy, (void *)ctx.Rsp, copied);
                    }
                }
                ResumeThread(th[i]);
                if (!ok) { ReleaseSRWLockExclusive(&copy_lock); continue; }
                DWORD64 stack[48];
                int depth = 0;
                stack[depth++] = ctx.Rip;
                copy_lo = ctx.Rsp, copy_hi = ctx.Rsp + copied;
                STACKFRAME64 f;
                memset(&f, 0, sizeof f);
                f.AddrPC.Offset = ctx.Rip; f.AddrPC.Mode = AddrModeFlat;
                f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
                f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
                for (int d = 0; d < 60 && depth < 48; d++) {
                    if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, th[i], &f, &ctx, read_copy, SymFunctionTableAccess64,
                                     SymGetModuleBase64, NULL) || !f.AddrPC.Offset)
                        break;
                    if (d > 0) stack[depth++] = f.AddrPC.Offset;
                }
                ReleaseSRWLockExclusive(&copy_lock);
                int seen[48], nseen = 0;
                for (int d = 0; d < depth; d++) {
                    const int j = hit_index(proc, stack[d], nm);
                    if (j < 0) continue;
                    int dup = 0;
                    for (int s = 0; s < nseen; s++) dup |= seen[s] == j;
                    if (dup) continue;
                    seen[nseen++] = j;
                    if (counts[k][j] < 65535) counts[k][j]++;
                }
                tsamples[k]++;
            }
            Sleep(1);
        }
        const int64_t len = hos_now_ns() - t0;
        last = hos_present_count, last_t = hos_now_ns();
        char line[4096];
        int len_out = snprintf(line, sizeof line, "stall: %.0f ms without a present at %.1f s;", len / 1e6,
                               (t0 - start) / 1e9 + (after_s ? atof(after_s) : 70));
        for (int k = 0; k < nt && len_out < (int)sizeof line - 200; k++) {
            if (!tsamples[k]) continue;
            len_out += snprintf(line + len_out, sizeof line - len_out, " | %s (%u):", tnames[k], tsamples[k]);
            for (int r = 0; r < 12; r++) {   /* the most frequent functions (roots and wait internals left out) */
                int best = -1;
                for (int j = 0; j < nhits; j++)
                    if (counts[k][j] && (best < 0 || counts[k][j] > counts[k][best])) best = j;
                if (best < 0) break;
                if (strstr(hits[best].name, "RtlUserThreadStart") || strstr(hits[best].name, "BaseThreadInitThunk") ||
                    strstr(hits[best].name, "thread_main") || strstr(hits[best].name, "main ") ||
                    strstr(hits[best].name, "ZwWaitForAlert") || strstr(hits[best].name, "SleepConditionVariableSRW") ||
                    strstr(hits[best].name, "__scrt_") || strstr(hits[best].name, "_Cnd_wait") || strstr(hits[best].name, "invoke")) {
                    counts[k][best] = 0;
                    r--;
                    continue;
                }
                char nm[40];
                snprintf(nm, sizeof nm, "%s", hits[best].name);
                char *sp = strchr(nm, ' ');
                if (sp) *sp = 0;   /* (without the thread tag) */
                len_out += snprintf(line + len_out, sizeof line - len_out, " %s %u%%", nm, counts[k][best] * 100 / tsamples[k]);
                counts[k][best] = 0;
            }
        }
        hos_log("%s", line);
        reports++;
    }
    return 0;
}

/* HOS_TRACE_DRIVER_ALLOC=<min KB> (debugging): the GPU driver's own memory (NVIDIA's nvoglv64.dll: its arenas are
   plain private memory, 1-2 GB reservations committed bit by bit) traced to what made it grow. Its VirtualAlloc
   import is patched once the driver loads; each commit of at least that size is counted under its call stack, and
   every 20 s the stacks that committed most are logged (frames outside the driver: wgpu's and ours). */
typedef LPVOID(WINAPI *VirtualAllocFn)(LPVOID, SIZE_T, DWORD, DWORD);
static VirtualAllocFn real_virtual_alloc;
static SIZE_T trace_min;
static uintptr_t drv_lo, drv_hi;
typedef struct { uint64_t key, bytes, count; void *frames[12]; int n; } AllocStack;
static AllocStack alloc_stacks[512];
static SRWLOCK alloc_lock = SRWLOCK_INIT;
static LPVOID WINAPI traced_virtual_alloc(LPVOID at, SIZE_T n, DWORD type, DWORD prot) {
    LPVOID r = real_virtual_alloc(at, n, type, prot);
    if (!r || !(type & MEM_COMMIT) || n < trace_min) return r;
    void *raw[40], *outside[12];
    const USHORT got = RtlCaptureStackBackTrace(1, 40, raw, NULL);
    int k = 0;
    uint64_t key = 0xcbf29ce484222325ull;
    for (USHORT i = 0; i < got && k < 12; i++) {
        const uintptr_t pc = (uintptr_t)raw[i];
        if (pc >= drv_lo && pc < drv_hi) continue;   /* (the driver's own frames: grouped together) */
        outside[k++] = raw[i];
        key = (key ^ pc) * 0x100000001B3ull;
    }
    AcquireSRWLockExclusive(&alloc_lock);
    for (int s = 0; s < 512; s++) {
        AllocStack *e = &alloc_stacks[(key + (uint64_t)s) % 512];
        if (e->count && e->key != key) continue;
        if (!e->count) e->key = key, e->n = k, memcpy(e->frames, outside, sizeof(void *) * (size_t)k);
        e->bytes += n, e->count++;
        break;
    }
    ReleaseSRWLockExclusive(&alloc_lock);
    return r;
}
static int patch_iat(HMODULE m, const char *fn, void *hook, void **orig) {
    const uint8_t *base = (const uint8_t *)m;
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)(base + ((const IMAGE_DOS_HEADER *)base)->e_lfanew);
    const IMAGE_DATA_DIRECTORY *dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    int patched = 0;
    for (const IMAGE_IMPORT_DESCRIPTOR *d = (const IMAGE_IMPORT_DESCRIPTOR *)(base + dir->VirtualAddress);
         dir->VirtualAddress && d->Name; d++) {
        IMAGE_THUNK_DATA *iat = (IMAGE_THUNK_DATA *)(base + d->FirstThunk);
        const IMAGE_THUNK_DATA *names = (const IMAGE_THUNK_DATA *)(base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
        for (; names->u1.AddressOfData; names++, iat++) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            const IMAGE_IMPORT_BY_NAME *ibn = (const IMAGE_IMPORT_BY_NAME *)(base + names->u1.AddressOfData);
            if (strcmp((const char *)ibn->Name, fn)) continue;
            DWORD old;
            VirtualProtect(&iat->u1.Function, sizeof(void *), PAGE_READWRITE, &old);
            if (!*orig) *orig = (void *)(uintptr_t)iat->u1.Function;
            iat->u1.Function = (ULONGLONG)(uintptr_t)hook;
            VirtualProtect(&iat->u1.Function, sizeof(void *), old, &old);
            patched++;
        }
    }
    return patched;
}
static DWORD WINAPI trace_driver_alloc_main(void *arg) {
    (void)arg;
    HMODULE m = NULL;
    while (!(m = GetModuleHandleA("nvoglv64.dll"))) Sleep(50);
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const uint8_t *)m + ((const IMAGE_DOS_HEADER *)m)->e_lfanew);
    drv_lo = (uintptr_t)m, drv_hi = drv_lo + nt->OptionalHeader.SizeOfImage;
    const int n = patch_iat(m, "VirtualAlloc", (void *)traced_virtual_alloc, (void **)&real_virtual_alloc);
    hos_log("mem: HOS_TRACE_DRIVER_ALLOC: nvoglv64.dll's VirtualAlloc patched (%d imports), commits >= %llu KB",
            n, (unsigned long long)(trace_min >> 10));
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    sym_init(proc);
    for (;;) {
        Sleep(20000);
        AllocStack top[8];
        int ntop = 0;
        uint64_t total = 0;
        AcquireSRWLockShared(&alloc_lock);
        for (int s = 0; s < 512; s++) {
            const AllocStack *e = &alloc_stacks[s];
            if (!e->count) continue;
            total += e->bytes;
            int at = ntop < 8 ? ntop++ : 7;
            if (at == 7 && ntop == 8 && top[7].bytes >= e->bytes) continue;
            top[at] = *e;
            for (; at > 0 && top[at - 1].bytes < top[at].bytes; at--) {
                AllocStack t = top[at - 1];
                top[at - 1] = top[at], top[at] = t;
            }
        }
        ReleaseSRWLockShared(&alloc_lock);
        hos_log("mem: driver commits so far %llu MB; by stack:", (unsigned long long)(total >> 20));
        char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO *sym = (SYMBOL_INFO *)buf;
        for (int i = 0; i < ntop; i++) {
            char line[1200];
            int len = snprintf(line, sizeof line, "  %llu MB in %llu:", (unsigned long long)(top[i].bytes >> 20),
                               (unsigned long long)top[i].count);
            for (int f = 0; f < top[i].n && len < (int)sizeof line - 120; f++) {
                memset(buf, 0, sizeof buf);
                sym->SizeOfStruct = sizeof(SYMBOL_INFO);
                sym->MaxNameLen = 255;
                DWORD64 disp = 0;
                char name[100] = "?";
                if (SymFromAddr(proc, (DWORD64)(uintptr_t)top[i].frames[f], &disp, sym)) {
                    if (!strncmp(sym->Name, "ZN", 2) || !strncmp(sym->Name, "_ZN", 3)) rust_short(sym->Name, name, sizeof name);
                    else snprintf(name, sizeof name, "%s", sym->Name);
                } else {
                    snprintf(name, sizeof name, "%p", top[i].frames[f]);
                }
                len += snprintf(line + len, sizeof line - (size_t)len, " < %s", name);
            }
            hos_log("%s", line);
        }
    }
    return 0;
}

void hos_prof_start(void) {
    SetUnhandledExceptionFilter(crash_filter);
    if (getenv("HOS_TRACE_DRIVER_ALLOC")) {
        trace_min = (SIZE_T)atoll(getenv("HOS_TRACE_DRIVER_ALLOC")) << 10;
        CreateThread(NULL, 0, trace_driver_alloc_main, NULL, 0, NULL);
    }
    if (getenv("HOS_STALL_MS") && atoi(getenv("HOS_STALL_MS")) > 0)
        CreateThread(NULL, 0, stall_main, (void *)(intptr_t)atoi(getenv("HOS_STALL_MS")), 0, NULL);
    if (getenv("HOS_POKE")) CreateThread(NULL, 0, poke_main, NULL, 0, NULL);
    if (getenv("HOS_MEMDUMP")) CreateThread(NULL, 0, memdump_main, NULL, 0, NULL);
    hos_mem_timeline_start();
    const char *s = getenv("HOS_PROFILE");
    if (s && atoi(s) > 0) CreateThread(NULL, 0, prof_main, (void *)(intptr_t)atoi(s), 0, NULL);
    s = getenv("HOS_EXIT_AFTER");
    if (s && atoi(s) > 0) CreateThread(NULL, 0, exit_main, (void *)(intptr_t)atoi(s), 0, NULL);
    if (getenv("HOS_THREAD_STATS")) CreateThread(NULL, 0, thread_stats_main, NULL, 0, NULL);
    s = getenv("HOS_AB");
    if (s && atoi(s) >= 200) CreateThread(NULL, 0, ab_main, (void *)(intptr_t)atoi(s), 0, NULL);
}
