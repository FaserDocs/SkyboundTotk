/* The browser build's Win32 (see windows.h): threads and events on pthreads, time from performance.now(), the
   game's files: the RomFS read from the page's server (synchronous XHR with a Range header, from the worker that
   asks: every guest thread is a worker) through a block cache, with a manifest for sizes and directory listings;
   everything else (saves, the user folder) in Emscripten's in-memory file system. */
#include <windows.h>
#include <mmsystem.h>
#include <dirent.h>
#include <emscripten.h>
#include <emscripten/threading.h>
#include <fcntl.h>
#include <stdio.h>
#include <strings.h>
#include <sys/stat.h>
#include <math.h>

volatile uint8_t web_keys[256];
EMSCRIPTEN_KEEPALIVE uintptr_t web_keys_address(void) { return (uintptr_t)web_keys; }

static _Thread_local DWORD last_error;
DWORD GetLastError(void) { return last_error; }
void SetLastError(DWORD e) { last_error = e; }

/* ---- handles ---- */
enum { H_THREAD = 1, H_EVENT, H_FILE, H_FIND };
typedef struct Handle {
    int kind;
    pthread_mutex_t m;
    pthread_cond_t c;
    /* thread */
    pthread_t t;
    LPTHREAD_START_ROUTINE fn;
    void *arg;
    int started, done;
    DWORD id;
    const char *name;    /* (the stack report: hos_thread.c names guest threads) */
    /* event */
    int manual, signaled;
    /* file */
    int fd;              /* local (Emscripten's FS), or -1 */
    int remote;          /* index in the RomFS manifest, or -1 */
    int64_t size, pos;
    /* find */
    char **names;
    uint32_t *attrs;
    int64_t *sizes;
    int count, next;
} Handle;

static Handle *new_handle(int kind) {
    Handle *h = (Handle *)calloc(1, sizeof *h);
    h->kind = kind;
    pthread_mutex_init(&h->m, NULL);
    pthread_cond_init(&h->c, NULL);
    h->fd = -1;
    h->remote = -1;
    return h;
}

void WakeConditionVariable(CONDITION_VARIABLE *c) {
    __atomic_add_fetch(&c->seq, 1, __ATOMIC_SEQ_CST);
    emscripten_futex_wake((void *)&c->seq, 1);
}
void WakeAllConditionVariable(CONDITION_VARIABLE *c) {
    __atomic_add_fetch(&c->seq, 1, __ATOMIC_SEQ_CST);
    emscripten_futex_wake((void *)&c->seq, INT32_MAX);
}
/* (the sequence is read before the lock is released: a wake between the release and the wait changes it, and the
   wait returns at once) */
BOOL SleepConditionVariableSRW(CONDITION_VARIABLE *c, SRWLOCK *l, DWORD ms, ULONG flags) {
    const uint32_t seq = __atomic_load_n(&c->seq, __ATOMIC_SEQ_CST);
    pthread_rwlock_unlock(&l->l);
    const int r = emscripten_futex_wait((void *)&c->seq, seq, ms == INFINITE ? INFINITY : (double)ms);
    if (flags & CONDITION_VARIABLE_LOCKMODE_SHARED) pthread_rwlock_rdlock(&l->l);
    else pthread_rwlock_wrlock(&l->l);
    if (r == -ETIMEDOUT) {
        last_error = ERROR_TIMEOUT;
        return FALSE;
    }
    return TRUE;
}

/* ---- threads ---- */
static _Thread_local Handle *self_handle;
static volatile LONG next_thread_id = 1;

/* Stack use (the memory log, stubs.c): each thread's unused stack is filled with a pattern at its start; the
   deepest byte changed since is its high-water mark. */
#include <emscripten/stack.h>
void hos_log(const char *fmt, ...);
#define STACK_SLOTS 256
static struct { uintptr_t end, base; DWORD id; Handle *h; } stack_slot[STACK_SLOTS];
static volatile LONG stack_slots;
static void stack_mark_start(Handle *h) {
    const DWORD id = h->id;
    const uintptr_t end = emscripten_stack_get_end(), cur = emscripten_stack_get_current();
    if (cur > end + 8192) memset((void *)(end + 256), 0xCD, cur - 4096 - end - 256);   /* (not the overflow cookie) */
    const LONG i = InterlockedIncrement(&stack_slots) - 1;
    if (i < STACK_SLOTS) {
        stack_slot[i].end = end;
        stack_slot[i].base = emscripten_stack_get_base();
        stack_slot[i].id = id;
        stack_slot[i].h = h;
    }
}
/* (a guest thread's name, hos_thread.c: kept on its handle) */
void web_thread_named(HANDLE t, const char *name) { ((Handle *)t)->name = name; }
/* the deepest any thread went (bytes) and the thread count measured; the three deepest are logged with their names */
size_t web_stack_high_water(int *threads, size_t *reserved) {
    size_t worst = 0, total = 0, top[3] = {0};
    int topi[3] = {-1, -1, -1};
    const int n = stack_slots < STACK_SLOTS ? (int)stack_slots : STACK_SLOTS;
    for (int i = 0; i < n; i++) {
        const uint8_t *p = (const uint8_t *)stack_slot[i].end + 256;
        while (p < (const uint8_t *)stack_slot[i].base && *p == 0xCD) p++;
        const size_t used = stack_slot[i].base - (uintptr_t)p;
        if (used > worst) worst = used;
        total += stack_slot[i].base - stack_slot[i].end;
        for (int k = 0; k < 3; k++)
            if (used > top[k]) {
                for (int m = 2; m > k; m--) top[m] = top[m - 1], topi[m] = topi[m - 1];
                top[k] = used, topi[k] = i;
                break;
            }
    }
    static size_t logged;
    if (worst > logged + (16 << 10)) {
        logged = worst;
        for (int k = 0; k < 3 && topi[k] >= 0; k++)
            hos_log("web stacks: #%d thread %u: %u KB", k + 1, (unsigned)stack_slot[topi[k]].id, (unsigned)(top[k] >> 10));
    }
    *threads = n;
    *reserved = total;
    return worst;
}

static void *thread_entry(void *p) {
    Handle *h = (Handle *)p;
    self_handle = h;
    stack_mark_start(h);
    pthread_mutex_lock(&h->m);
    while (!h->started) pthread_cond_wait(&h->c, &h->m);
    pthread_mutex_unlock(&h->m);
    h->fn(h->arg);
    pthread_mutex_lock(&h->m);
    h->done = 1;
    pthread_cond_broadcast(&h->c);
    pthread_mutex_unlock(&h->m);
    return NULL;
}

/* The next thread created gets the page's canvas (the renderer's thread: the browser shows what it draws) */
static const char *next_canvas;
void web_next_thread_takes_canvas(const char *selector) { next_canvas = selector; }

HANDLE CreateThread(LPSECURITY_ATTRIBUTES sa, SIZE_T stack, LPTHREAD_START_ROUTINE fn, LPVOID arg, DWORD flags,
                    LPDWORD id) {
    (void)sa;
    Handle *h = new_handle(H_THREAD);
    h->fn = fn;
    h->arg = arg;
    h->id = (DWORD)InterlockedIncrement(&next_thread_id);
    h->started = !(flags & CREATE_SUSPENDED);
    pthread_attr_t a;
    pthread_attr_init(&a);
    /* A reservation natively (64 MB a guest thread) is real memory here, and the wasm stack in memory holds only
       what has its address taken (locals live on the engine's own stack). */
    static size_t floor_size;
    if (!floor_size) {   /* HOS_STACK_KB: the stack of every thread made here (default 1 MB: most of the game's 60
                            use ~11 KB, one went past 512 KB (Opus, built with alloca?); 4 MB each held 240 MB of the
                            host's heap) */
        const char *e = getenv("HOS_STACK_KB");
        floor_size = (e && atoi(e) > 0 ? (size_t)atoi(e) : 1024) << 10;
    }
    if (flags & STACK_SIZE_PARAM_IS_A_RESERVATION) stack = floor_size;
    pthread_attr_setstacksize(&a, stack && stack > floor_size ? stack : floor_size);
    if (next_canvas) {
        emscripten_pthread_attr_settransferredcanvases(&a, next_canvas);
        next_canvas = NULL;
    }
    if (pthread_create(&h->t, &a, thread_entry, h)) {
        pthread_attr_destroy(&a);
        free(h);
        return NULL;
    }
    pthread_attr_destroy(&a);
    pthread_detach(h->t);
    if (id) *id = h->id;
    return h;
}

DWORD ResumeThread(HANDLE t) {
    Handle *h = (Handle *)t;
    if (!h || h->kind != H_THREAD) return (DWORD)-1;
    pthread_mutex_lock(&h->m);
    DWORD was = h->started ? 0 : 1;
    h->started = 1;
    pthread_cond_broadcast(&h->c);
    pthread_mutex_unlock(&h->m);
    return was;
}
DWORD SuspendThread(HANDLE t) { (void)t; return (DWORD)-1; }
void ExitThread(DWORD code) { (void)code; pthread_exit(NULL); }
static Handle main_handle = {H_THREAD};
HANDLE GetCurrentThread(void) { return self_handle ? self_handle : &main_handle; }
HANDLE GetCurrentProcess(void) { return (HANDLE)(intptr_t)-1; }
static _Thread_local DWORD other_id;   /* threads not made by CreateThread (std::thread, the main thread) */
DWORD GetCurrentThreadId(void) {
    if (self_handle) return self_handle->id;
    if (!other_id) other_id = (DWORD)InterlockedIncrement(&next_thread_id);
    return other_id;
}
DWORD GetCurrentProcessId(void) { return 1; }
DWORD GetThreadId(HANDLE t) { return t && ((Handle *)t)->kind == H_THREAD ? ((Handle *)t)->id : 0; }
BOOL SetThreadPriority(HANDLE t, int p) { (void)t; (void)p; return TRUE; }
int GetThreadPriority(HANDLE t) { (void)t; return 0; }
DWORD_PTR SetThreadAffinityMask(HANDLE t, DWORD_PTR m) { (void)t; return m; }
HRESULT SetThreadDescription(HANDLE t, LPCWSTR d) { (void)t; (void)d; return 0; }
BOOL DuplicateHandle(HANDLE sp, HANDLE s, HANDLE tp, HANDLE *t, DWORD a, BOOL i, DWORD o) {
    (void)sp; (void)tp; (void)a; (void)i; (void)o;
    *t = s == GetCurrentThread() ? (HANDLE)self_handle : s;
    if (!*t) *t = &main_handle;
    return TRUE;
}
BOOL CloseHandle(HANDLE h) {
    Handle *x = (Handle *)h;
    if (!x || x == &main_handle) return TRUE;
    if (x->kind == H_FILE && x->fd >= 0) close(x->fd);
    /* (threads and events stay: others may still wait on them) */
    if (x->kind == H_FILE) free(x);
    return TRUE;
}

/* ---- events and waits ---- */
HANDLE CreateEventA(LPSECURITY_ATTRIBUTES sa, BOOL manual, BOOL initial, LPCSTR name) {
    (void)sa; (void)name;
    Handle *h = new_handle(H_EVENT);
    h->manual = manual;
    h->signaled = initial;
    return h;
}
BOOL SetEvent(HANDLE e) {
    Handle *h = (Handle *)e;
    pthread_mutex_lock(&h->m);
    h->signaled = 1;
    pthread_cond_broadcast(&h->c);
    pthread_mutex_unlock(&h->m);
    return TRUE;
}
BOOL ResetEvent(HANDLE e) {
    Handle *h = (Handle *)e;
    pthread_mutex_lock(&h->m);
    h->signaled = 0;
    pthread_mutex_unlock(&h->m);
    return TRUE;
}

static int ready_locked(Handle *h) { return h->kind == H_THREAD ? h->done : h->signaled; }

DWORD WaitForSingleObject(HANDLE x, DWORD ms) {
    Handle *h = (Handle *)x;
    if (!h || (h->kind != H_THREAD && h->kind != H_EVENT)) return WAIT_FAILED;
    struct timespec ts;
    if (ms != INFINITE) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += ms / 1000;
        ts.tv_nsec += (long)(ms % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }
    }
    pthread_mutex_lock(&h->m);
    while (!ready_locked(h)) {
        if (ms == INFINITE) pthread_cond_wait(&h->c, &h->m);
        else if (pthread_cond_timedwait(&h->c, &h->m, &ts) == ETIMEDOUT) {
            pthread_mutex_unlock(&h->m);
            return WAIT_TIMEOUT;
        }
    }
    if (h->kind == H_EVENT && !h->manual) h->signaled = 0;
    pthread_mutex_unlock(&h->m);
    return WAIT_OBJECT_0;
}

DWORD WaitForMultipleObjects(DWORD n, const HANDLE *hs, BOOL all, DWORD ms) {
    /* (polled: rare here) */
    const double end = ms == INFINITE ? 1e300 : emscripten_get_now() + ms;
    for (;;) {
        int ready = 0, first = -1;
        for (DWORD i = 0; i < n; i++) {
            Handle *h = (Handle *)hs[i];
            pthread_mutex_lock(&h->m);
            if (ready_locked(h)) {
                ready++;
                if (first < 0) first = (int)i;
            }
            pthread_mutex_unlock(&h->m);
        }
        if (all ? ready == (int)n : ready > 0) {
            for (DWORD i = 0; i < n; i++) {
                Handle *h = (Handle *)hs[i];
                if (h->kind == H_EVENT && !h->manual && (all || (int)i == first)) h->signaled = 0;
            }
            return WAIT_OBJECT_0 + (all ? 0 : first);
        }
        if (emscripten_get_now() >= end) return WAIT_TIMEOUT;
        usleep(500);
    }
}

HANDLE CreateWaitableTimerExW(LPSECURITY_ATTRIBUTES sa, LPCWSTR name, DWORD flags, DWORD access) {
    (void)sa; (void)name; (void)flags; (void)access;
    return NULL;   /* (hos_sleep_ns sleeps and spins without one) */
}
BOOL SetWaitableTimerEx(HANDLE t, const LARGE_INTEGER *due, LONG period, void *fn, LPVOID arg, void *ctx, ULONG d) {
    (void)t; (void)due; (void)period; (void)fn; (void)arg; (void)ctx; (void)d;
    return FALSE;
}
void Sleep(DWORD ms) {
    if (!ms) sched_yield();
    else usleep((useconds_t)ms * 1000);
}
BOOL SwitchToThread(void) {
    sched_yield();
    return TRUE;
}

/* ---- time ---- */
BOOL QueryPerformanceCounter(LARGE_INTEGER *c) {
    c->QuadPart = (LONGLONG)(emscripten_get_now() * 1e6);
    return TRUE;
}
BOOL QueryPerformanceFrequency(LARGE_INTEGER *f) {
    f->QuadPart = 1000000000LL;
    return TRUE;
}
DWORD GetTickCount(void) { return (DWORD)emscripten_get_now(); }
ULONGLONG GetTickCount64(void) { return (ULONGLONG)emscripten_get_now(); }
DWORD timeGetTime(void) { return (DWORD)emscripten_get_now(); }

/* ---- the rest ---- */
void ExitProcess(UINT code) { emscripten_force_exit((int)code); }
void OutputDebugStringA(LPCSTR s) { fputs(s, stderr); }
DWORD GetModuleFileNameA(HMODULE m, LPSTR out, DWORD n) {
    (void)m;
    snprintf(out, n, "/totk");
    return (DWORD)strlen(out);
}
HMODULE GetModuleHandleA(LPCSTR n) { (void)n; return NULL; }
HMODULE LoadLibraryA(LPCSTR n) { (void)n; return NULL; }
void *GetProcAddress(HMODULE m, LPCSTR n) { (void)m; (void)n; return NULL; }
USHORT RtlCaptureStackBackTrace(ULONG skip, ULONG n, PVOID *out, PULONG hash) {
    (void)skip; (void)n; (void)out; (void)hash;
    return 0;
}
SHORT GetAsyncKeyState(int vk) { return web_keys[vk & 255] ? (SHORT)0x8000 : 0; }
HWND GetForegroundWindow(void) { return (HWND)1; }
int MultiByteToWideChar(UINT cp, DWORD flags, LPCSTR s, int n, LPWSTR out, int outn) {
    (void)cp; (void)flags;
    if (n < 0) n = (int)strlen(s) + 1;
    if (!outn) return n;
    int i = 0;
    for (; i < n && i < outn; i++) out[i] = (unsigned char)s[i];
    return i;
}

/* ---- memory ---- */
LPVOID VirtualAlloc(LPVOID at, SIZE_T n, DWORD type, DWORD prot) {
    (void)type; (void)prot;
    if (at) return at;   /* (inside the guest window: it is all there) */
    void *p = aligned_alloc(65536, (n + 65535) & ~(SIZE_T)65535);
    if (p) memset(p, 0, n);
    return p;
}
BOOL VirtualFree(LPVOID at, SIZE_T n, DWORD type) { (void)at; (void)n; (void)type; return TRUE; }
BOOL VirtualProtect(LPVOID at, SIZE_T n, DWORD prot, PDWORD old) {
    (void)at; (void)n; (void)prot;
    if (old) *old = PAGE_READWRITE;
    return TRUE;
}
SIZE_T VirtualQuery(LPCVOID at, MEMORY_BASIC_INFORMATION *out, SIZE_T n) {
    (void)n;
    memset(out, 0, sizeof *out);
    out->BaseAddress = (PVOID)at;
    out->RegionSize = 4096;
    out->State = MEM_COMMIT;
    out->Protect = PAGE_READWRITE;
    return sizeof *out;
}
/* Write watch from the guest stores' page marks (rt/a64.h A64_MARK; host code marks with web_mark_written): the
   marked pages of [base, base + n), cleared as they are reported when asked to reset. */
extern uint8_t a64_dirty[1u << 20];
UINT GetWriteWatch(DWORD flags, PVOID base, SIZE_T n, PVOID *addrs, ULONG_PTR *count, ULONG *gran) {
    const uint32_t p0 = (uint32_t)((uintptr_t)base >> 12), p1 = (uint32_t)(((uintptr_t)base + n + 4095) >> 12);
    ULONG_PTR k = 0;
    const ULONG_PTR cap = *count;
    for (uint32_t p = p0; p < p1 && p < (1u << 20) && k < cap;) {
        if (!(p & 7) && p + 8 <= p1) {   /* (eight pages a load while none is marked) */
            uint64_t w;
            memcpy(&w, &a64_dirty[p], 8);
            if (!w) {
                p += 8;
                continue;
            }
        }
        if (a64_dirty[p]) {
            if (flags & WRITE_WATCH_FLAG_RESET) __atomic_store_n(&a64_dirty[p], 0, __ATOMIC_RELAXED);
            addrs[k++] = (PVOID)((uintptr_t)p << 12);
        }
        p++;
    }
    *count = k;
    if (gran) *gran = 4096;
    return 0;
}
UINT ResetWriteWatch(LPVOID base, SIZE_T n) {
    const uint32_t p0 = (uint32_t)((uintptr_t)base >> 12), p1 = (uint32_t)(((uintptr_t)base + n + 4095) >> 12);
    if (p0 < (1u << 20)) memset(&a64_dirty[p0], 0, (p1 < (1u << 20) ? p1 : (1u << 20)) - p0);
    return 0;
}
void web_mark_written(const void *p, size_t n) {
    extern uint64_t hos_space_end;
    const uintptr_t a = (uintptr_t)p;
    if (!n || a >= hos_space_end) return;
    for (uintptr_t q = a >> 12; q <= (a + n - 1) >> 12 && q < (1u << 20); q++) a64_dirty[q] = 1;
}

/* ---- HTTP: synchronous ranged reads from the worker that asks ---- */
EM_JS(int, js_fetch_range, (const char *url, double offset, double n, void *dst, double cap), {
    try {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', UTF8ToString(url), false);
        xhr.responseType = 'arraybuffer';
        if (n > 0) xhr.setRequestHeader('Range', 'bytes=' + offset + '-' + (offset + n - 1));
        xhr.send(null);
        if (xhr.status != 200 && xhr.status != 206) return -1;
        var b = new Uint8Array(xhr.response);
        if (n <= 0 && b.length > cap) b = b.subarray(0, cap);
        if (n > 0 && xhr.status == 200 && b.length > n) b = b.subarray(offset, offset + n);   // (no Range support)
        HEAPU8.set(b, dst);
        return b.length;
    } catch (e) {
        console.error('fetch ' + UTF8ToString(url) + ': ' + e);
        return -1;
    }
});
EM_JS(char *, js_fetch_text, (const char *url), {
    try {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', UTF8ToString(url), false);
        xhr.send(null);
        if (xhr.status != 200) return 0;
        return stringToNewUTF8(xhr.responseText);
    } catch (e) {
        return 0;
    }
});

/* (counters for the memory log, stubs.c: requests, bytes and the time threads spent waiting on them) */
uint64_t web_fetch_count, web_fetch_bytes, web_fetch_us;
/* a whole file, at most cap bytes (no Range: the server may send it compressed) */
int web_fetch_all(const char *url, void *dst, uint64_t cap) { return js_fetch_range(url, 0, 0, dst, (double)cap); }

int web_fetch(const char *url, uint64_t offset, uint64_t n, void *dst) {
    const double t0 = emscripten_get_now();
    const int got = js_fetch_range(url, (double)offset, n == (uint64_t)-1 ? 0 : (double)n, dst, (double)n);
    __atomic_add_fetch(&web_fetch_us, (uint64_t)((emscripten_get_now() - t0) * 1000), __ATOMIC_RELAXED);
    __atomic_add_fetch(&web_fetch_count, 1, __ATOMIC_RELAXED);
    if (got > 0) __atomic_add_fetch(&web_fetch_bytes, (uint64_t)got, __ATOMIC_RELAXED);
    return got;
}

/* ---- the RomFS manifest: "romfs.json" from the server, {"path/in/romfs": size, ...}, sorted here ---- */
typedef struct { char *path; int64_t size; } Entry;
static Entry *entries;
static int nentries;
static char romfs_prefix[256];   /* the game's RomFS folder as nn_fs names it (hos_romfs_dir) */
static size_t romfs_prefix_len;

static int entry_cmp(const void *a, const void *b) { return strcasecmp(((const Entry *)a)->path, ((const Entry *)b)->path); }

void web_fs_init(const char *romfs_url, const char *romfs_dir) {
    (void)romfs_url;
    snprintf(romfs_prefix, sizeof romfs_prefix, "%s", romfs_dir ? romfs_dir : "romfs");
    romfs_prefix_len = strlen(romfs_prefix);
    char url[300];
    snprintf(url, sizeof url, "%s.json", romfs_prefix);
    char *text = js_fetch_text(url);
    if (!text) {
        fprintf(stderr, "web: no RomFS manifest at %s\n", url);
        return;
    }
    int cap = 1 << 16;
    entries = (Entry *)malloc(cap * sizeof(Entry));
    for (char *p = text; (p = strchr(p, '"'));) {
        char *q = strchr(p + 1, '"');
        if (!q) break;
        *q = 0;
        if (nentries == cap) entries = (Entry *)realloc(entries, (cap *= 2) * sizeof(Entry));
        entries[nentries].path = strdup(p + 1);
        char *colon = strchr(q + 1, ':');
        entries[nentries].size = colon ? strtoll(colon + 1, &p, 10) : 0;
        nentries++;
        if (!colon) break;
    }
    free(text);
    qsort(entries, nentries, sizeof(Entry), entry_cmp);
    fprintf(stderr, "web: RomFS manifest: %d files\n", nentries);
}

/* "romfs\\Actor\\x.bfres" -> "Actor/x.bfres" when it is under the RomFS folder; NULL otherwise */
static const char *romfs_rel(const char *path, char *buf, size_t cap) {
    snprintf(buf, cap, "%s", path);
    for (char *p = buf; *p; p++)
        if (*p == '\\') *p = '/';
    if (strncasecmp(buf, romfs_prefix, romfs_prefix_len) || (buf[romfs_prefix_len] && buf[romfs_prefix_len] != '/'))
        return NULL;
    const char *rel = buf + romfs_prefix_len;
    while (*rel == '/') rel++;
    return rel;
}
static int find_entry(const char *rel) {
    int lo = 0, hi = nentries;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        const int c = strcasecmp(entries[mid].path, rel);
        if (!c) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}
static int is_remote_dir(const char *rel) {   /* some entry starts with rel + "/" */
    if (!*rel) return 1;
    const size_t n = strlen(rel);
    int lo = 0, hi = nentries;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (strcasecmp(entries[mid].path, rel) < 0) lo = mid + 1;
        else hi = mid;
    }
    for (int i = lo; i < nentries && i < lo + 4; i++)
        if (!strncasecmp(entries[i].path, rel, n) && entries[i].path[n] == '/') return 1;
    return 0;
}

/* ---- the block cache (1 MB blocks of RomFS files) ---- */
#define BLOCK (1u << 20)
#define NBLOCKS 24   /* (24 MB: the host has ~570 MB above the guest window) */
static struct { int file; int64_t block; uint8_t *data; int len; uint64_t used; } cache[NBLOCKS];
static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t cache_clock;

static int remote_read(int file, int64_t offset, void *dst, int64_t n) {
    char url[600];
    snprintf(url, sizeof url, "%s/%s", romfs_prefix, entries[file].path);
    if (offset >= entries[file].size) return 0;
    if (offset + n > entries[file].size) n = entries[file].size - offset;
    if (n >= BLOCK) return web_fetch(url, (uint64_t)offset, (uint64_t)n, dst);   /* (large: straight in) */
    int64_t done = 0;
    while (done < n) {
        const int64_t at = offset + done, b = at / BLOCK, in = at % BLOCK;
        pthread_mutex_lock(&cache_lock);
        int slot = -1, oldest = 0;
        for (int i = 0; i < NBLOCKS; i++) {
            if (cache[i].data && cache[i].file == file && cache[i].block == b) { slot = i; break; }
            if (cache[i].used < cache[oldest].used) oldest = i;
        }
        if (slot < 0) {
            pthread_mutex_unlock(&cache_lock);
            uint8_t *data = (uint8_t *)malloc(BLOCK);
            const int64_t want = entries[file].size - b * BLOCK < BLOCK ? entries[file].size - b * BLOCK : BLOCK;
            const int got = web_fetch(url, (uint64_t)(b * BLOCK), (uint64_t)want, data);
            if (got <= 0) {
                free(data);
                return done ? (int)done : -1;
            }
            pthread_mutex_lock(&cache_lock);
            slot = oldest;
            free(cache[slot].data);
            cache[slot].file = file;
            cache[slot].block = b;
            cache[slot].data = data;
            cache[slot].len = got;
        }
        cache[slot].used = ++cache_clock;
        const int64_t take = (cache[slot].len - in) < (n - done) ? (cache[slot].len - in) : (n - done);
        if (take <= 0) {
            pthread_mutex_unlock(&cache_lock);
            break;
        }
        memcpy((uint8_t *)dst + done, cache[slot].data + in, (size_t)take);
        pthread_mutex_unlock(&cache_lock);
        done += take;
    }
    return (int)done;
}

/* ---- files ---- */
static void local_path(const char *path, char *out, size_t cap) {
    snprintf(out, cap, "%s%s", path[0] == '/' ? "" : "/", path);
    for (char *p = out; *p; p++)
        if (*p == '\\') *p = '/';
}

HANDLE CreateFileA(LPCSTR path, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa, DWORD disp, DWORD flags,
                   HANDLE tmpl) {
    (void)share; (void)sa; (void)flags; (void)tmpl;
    char buf[600];
    const char *rel = romfs_rel(path, buf, sizeof buf);
    if (rel) {
        const int e = find_entry(rel);
        if (e < 0 || (access & GENERIC_WRITE)) {
            last_error = e < 0 ? ERROR_FILE_NOT_FOUND : ERROR_ACCESS_DENIED;
            return INVALID_HANDLE_VALUE;
        }
        Handle *h = new_handle(H_FILE);
        h->remote = e;
        h->size = entries[e].size;
        return h;
    }
    char lp[600];
    local_path(path, lp, sizeof lp);
    int o = (access & GENERIC_WRITE) ? ((access & GENERIC_READ) ? O_RDWR : O_WRONLY) : O_RDONLY;
    switch (disp) {
    case CREATE_NEW: o |= O_CREAT | O_EXCL; break;
    case CREATE_ALWAYS: o |= O_CREAT | O_TRUNC; break;
    case OPEN_ALWAYS: o |= O_CREAT; break;
    case TRUNCATE_EXISTING: o |= O_TRUNC; break;
    default: break;
    }
    const int fd = open(lp, o, 0666);
    if (fd < 0) {
        last_error = errno == EEXIST ? ERROR_FILE_EXISTS : ERROR_FILE_NOT_FOUND;
        return INVALID_HANDLE_VALUE;
    }
    Handle *h = new_handle(H_FILE);
    h->fd = fd;
    struct stat st;
    h->size = fstat(fd, &st) ? 0 : st.st_size;
    return h;
}

BOOL ReadFile(HANDLE x, LPVOID buf, DWORD n, LPDWORD got, LPOVERLAPPED ov) {
    Handle *h = (Handle *)x;
    if (!h || h->kind != H_FILE) return FALSE;
    const int64_t at = ov ? ((int64_t)ov->OffsetHigh << 32 | ov->Offset) : h->pos;
    int r;
    if (h->remote >= 0) r = remote_read(h->remote, at, buf, n);
    else r = (int)pread(h->fd, buf, n, at);
    if (r > 0) web_mark_written(buf, (size_t)r);   /* (into guest memory: the GPU may read it) */
    if (r < 0) {
        if (got) *got = 0;
        last_error = ERROR_HANDLE_EOF;
        return FALSE;
    }
    if (!ov) h->pos = at + r;
    if (got) *got = (DWORD)r;
    return TRUE;
}

BOOL WriteFile(HANDLE x, LPCVOID buf, DWORD n, LPDWORD put, LPOVERLAPPED ov) {
    Handle *h = (Handle *)x;
    if (!h || h->kind != H_FILE || h->fd < 0) return FALSE;
    const int64_t at = ov ? ((int64_t)ov->OffsetHigh << 32 | ov->Offset) : h->pos;
    const int r = (int)pwrite(h->fd, buf, n, at);
    if (r < 0) return FALSE;
    if (!ov) h->pos = at + r;
    if (at + r > h->size) h->size = at + r;
    if (put) *put = (DWORD)r;
    return TRUE;
}

BOOL GetFileSizeEx(HANDLE x, PLARGE_INTEGER size) {
    Handle *h = (Handle *)x;
    if (h->fd >= 0) {
        struct stat st;
        if (!fstat(h->fd, &st)) h->size = st.st_size;
    }
    size->QuadPart = h->size;
    return TRUE;
}
DWORD GetFileSize(HANDLE x, LPDWORD high) {
    LARGE_INTEGER s;
    GetFileSizeEx(x, &s);
    if (high) *high = (DWORD)(s.QuadPart >> 32);
    return (DWORD)s.QuadPart;
}
BOOL SetFilePointerEx(HANDLE x, LARGE_INTEGER to, PLARGE_INTEGER now, DWORD how) {
    Handle *h = (Handle *)x;
    int64_t base = how == FILE_BEGIN ? 0 : how == FILE_CURRENT ? h->pos : h->size;
    h->pos = base + to.QuadPart;
    if (now) now->QuadPart = h->pos;
    return TRUE;
}
BOOL SetEndOfFile(HANDLE x) {
    Handle *h = (Handle *)x;
    if (h->fd < 0 || ftruncate(h->fd, h->pos)) return FALSE;
    h->size = h->pos;
    return TRUE;
}
BOOL FlushFileBuffers(HANDLE x) { (void)x; return TRUE; }

DWORD GetFileAttributesA(LPCSTR path) {
    char buf[600];
    const char *rel = romfs_rel(path, buf, sizeof buf);
    if (rel) {
        if (find_entry(rel) >= 0) return FILE_ATTRIBUTE_READONLY;
        return is_remote_dir(rel) ? FILE_ATTRIBUTE_DIRECTORY : INVALID_FILE_ATTRIBUTES;
    }
    char lp[600];
    local_path(path, lp, sizeof lp);
    struct stat st;
    if (stat(lp, &st)) return INVALID_FILE_ATTRIBUTES;
    return S_ISDIR(st.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL;
}

BOOL CreateDirectoryA(LPCSTR path, LPSECURITY_ATTRIBUTES sa) {
    (void)sa;
    char lp[600];
    local_path(path, lp, sizeof lp);
    if (mkdir(lp, 0777)) {
        last_error = errno == EEXIST ? ERROR_ALREADY_EXISTS : ERROR_PATH_NOT_FOUND;
        return FALSE;
    }
    return TRUE;
}
BOOL DeleteFileA(LPCSTR path) {
    char lp[600];
    local_path(path, lp, sizeof lp);
    return unlink(lp) == 0;
}
BOOL RemoveDirectoryA(LPCSTR path) {
    char lp[600];
    local_path(path, lp, sizeof lp);
    return rmdir(lp) == 0;
}
BOOL MoveFileExA(LPCSTR from, LPCSTR to, DWORD flags) {
    (void)flags;
    char a[600], b[600];
    local_path(from, a, sizeof a);
    local_path(to, b, sizeof b);
    return rename(a, b) == 0;
}

/* FindFirstFileA("dir\\*") */
static void add_found(Handle *f, const char *name, uint32_t attr, int64_t size) {
    for (int i = 0; i < f->count; i++)
        if (!strcasecmp(f->names[i], name)) return;
    f->names = (char **)realloc(f->names, (f->count + 1) * sizeof(char *));
    f->attrs = (uint32_t *)realloc(f->attrs, (f->count + 1) * sizeof(uint32_t));
    f->sizes = (int64_t *)realloc(f->sizes, (f->count + 1) * sizeof(int64_t));
    f->names[f->count] = strdup(name);
    f->attrs[f->count] = attr;
    f->sizes[f->count] = size;
    f->count++;
}
static void fill_find(Handle *f, LPWIN32_FIND_DATAA fd) {
    memset(fd, 0, sizeof *fd);
    snprintf(fd->cFileName, sizeof fd->cFileName, "%s", f->names[f->next]);
    fd->dwFileAttributes = f->attrs[f->next];
    fd->nFileSizeLow = (DWORD)f->sizes[f->next];
    fd->nFileSizeHigh = (DWORD)(f->sizes[f->next] >> 32);
    f->next++;
}
HANDLE FindFirstFileA(LPCSTR pattern, LPWIN32_FIND_DATAA fd) {
    char dir[600];
    snprintf(dir, sizeof dir, "%s", pattern);
    for (char *p = dir; *p; p++)
        if (*p == '\\') *p = '/';
    char *slash = strrchr(dir, '/');
    if (slash) *slash = 0;
    else dir[0] = 0;
    Handle *f = new_handle(H_FIND);
    char buf[600];
    const char *rel = romfs_rel(dir, buf, sizeof buf);
    if (rel) {
        const size_t n = strlen(rel);
        for (int i = 0; i < nentries; i++) {
            const char *p = entries[i].path;
            if (n && (strncasecmp(p, rel, n) || p[n] != '/')) continue;
            const char *rest = p + (n ? n + 1 : 0);
            const char *sep = strchr(rest, '/');
            if (sep) {
                char name[260];
                snprintf(name, sizeof name, "%.*s", (int)(sep - rest), rest);
                add_found(f, name, FILE_ATTRIBUTE_DIRECTORY, 0);
            } else {
                add_found(f, rest, FILE_ATTRIBUTE_READONLY, entries[i].size);
            }
        }
    } else {
        char lp[600];
        local_path(dir, lp, sizeof lp);
        DIR *d = opendir(lp);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d))) {
                if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
                char full[900];
                snprintf(full, sizeof full, "%s/%s", lp, e->d_name);
                struct stat st;
                if (stat(full, &st)) continue;
                add_found(f, e->d_name, S_ISDIR(st.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL,
                          st.st_size);
            }
            closedir(d);
        }
    }
    if (!f->count) {
        free(f);
        last_error = ERROR_FILE_NOT_FOUND;
        return INVALID_HANDLE_VALUE;
    }
    fill_find(f, fd);
    return f;
}
BOOL FindNextFileA(HANDLE x, LPWIN32_FIND_DATAA fd) {
    Handle *f = (Handle *)x;
    if (f->next >= f->count) {
        last_error = ERROR_NO_MORE_FILES;
        return FALSE;
    }
    fill_find(f, fd);
    return TRUE;
}
BOOL FindClose(HANDLE x) {
    Handle *f = (Handle *)x;
    for (int i = 0; i < f->count; i++) free(f->names[i]);
    free(f->names);
    free(f->attrs);
    free(f->sizes);
    free(f);
    return TRUE;
}

/* ---- audio: waveOut -> a ring of 16-bit stereo frames at 48 kHz that the page's AudioWorklet plays (index.html:
   it reads web_audio_address(): write and read counts, the ring's size in frames, then the samples) ---- */
#include <mmsystem.h>
#define WEB_AUDIO_FRAMES 16384   /* (~340 ms) */
static struct {
    volatile int32_t write, read, size, rate;
    int16_t data[WEB_AUDIO_FRAMES * 2];
} web_audio = {0, 0, WEB_AUDIO_FRAMES, 48000};
EMSCRIPTEN_KEEPALIVE uintptr_t web_audio_address(void) { return (uintptr_t)&web_audio; }

MMRESULT waveOutOpen(HWAVEOUT *w, UINT d, const WAVEFORMATEX *f, DWORD_PTR cb, DWORD_PTR u, DWORD fl) {
    (void)d; (void)cb; (void)u; (void)fl;
    if (!f || f->nChannels != 2 || f->wBitsPerSample != 16) return MMSYSERR_NODRIVER;
    web_audio.rate = (int32_t)f->nSamplesPerSec;
    *w = (HWAVEOUT)&web_audio;
    return MMSYSERR_NOERROR;
}

MMRESULT waveOutWrite(HWAVEOUT w, WAVEHDR *h, UINT n) {
    (void)w; (void)n;
    const int16_t *src = (const int16_t *)h->lpData;
    int32_t frames = (int32_t)(h->dwBufferLength / 4);
    const int32_t wr = __atomic_load_n(&web_audio.write, __ATOMIC_ACQUIRE);
    const int32_t rd = __atomic_load_n(&web_audio.read, __ATOMIC_ACQUIRE);
    const int32_t room = WEB_AUDIO_FRAMES - (wr - rd);
    if (frames > room) frames = room > 0 ? room : 0;   /* (nobody plays yet, or it fell behind: dropped) */
    for (int32_t i = 0; i < frames; i++) {
        const int32_t k = ((wr + i) % WEB_AUDIO_FRAMES) * 2;
        web_audio.data[k] = src[i * 2];
        web_audio.data[k + 1] = src[i * 2 + 1];
    }
    __atomic_store_n(&web_audio.write, wr + frames, __ATOMIC_RELEASE);
    h->dwFlags |= WHDR_DONE;   /* (copied: the block is free again) */
    return MMSYSERR_NOERROR;
}
