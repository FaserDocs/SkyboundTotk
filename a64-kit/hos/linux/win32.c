/* Linux Win32 stand-in (see windows.h): threads and events on pthreads, files on POSIX, audio on SDL3. */
#define _GNU_SOURCE
#include <windows.h>
#include <mmsystem.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <SDL3/SDL.h>

volatile uint8_t web_keys[256];

static _Thread_local DWORD last_error;
DWORD GetLastError(void) { return last_error; }
void SetLastError(DWORD e) { last_error = e; }

enum { H_THREAD = 1, H_EVENT, H_FILE, H_FIND };
typedef struct Handle {
    int kind;
    pthread_mutex_t m;
    pthread_cond_t c;
    pthread_t t;
    LPTHREAD_START_ROUTINE fn;
    void *arg;
    int started, done;
    DWORD id;
    int manual, signaled;
    int fd;
    int64_t size, pos;
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
    return h;
}

static int futex_wait(volatile uint32_t *addr, uint32_t val, const struct timespec *ts) {
    return (int)syscall(SYS_futex, addr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, val, ts, NULL, 0);
}
static void futex_wake(volatile uint32_t *addr, int n) {
    syscall(SYS_futex, addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, n, NULL, NULL, 0);
}

void WakeConditionVariable(CONDITION_VARIABLE *c) {
    __atomic_add_fetch(&c->seq, 1, __ATOMIC_SEQ_CST);
    futex_wake(&c->seq, 1);
}
void WakeAllConditionVariable(CONDITION_VARIABLE *c) {
    __atomic_add_fetch(&c->seq, 1, __ATOMIC_SEQ_CST);
    futex_wake(&c->seq, INT32_MAX);
}
BOOL SleepConditionVariableSRW(CONDITION_VARIABLE *c, SRWLOCK *l, DWORD ms, ULONG flags) {
    const uint32_t seq = __atomic_load_n(&c->seq, __ATOMIC_SEQ_CST);
    pthread_rwlock_unlock(&l->l);
    struct timespec ts, *tsp = NULL;
    if (ms != INFINITE) {
        ts.tv_sec = ms / 1000;
        ts.tv_nsec = (long)(ms % 1000) * 1000000L;
        tsp = &ts;
    }
    const int r = futex_wait(&c->seq, seq, tsp);
    const int timed_out = r < 0 && errno == ETIMEDOUT;
    if (flags & CONDITION_VARIABLE_LOCKMODE_SHARED) pthread_rwlock_rdlock(&l->l);
    else pthread_rwlock_wrlock(&l->l);
    if (timed_out) {
        last_error = ERROR_TIMEOUT;
        return FALSE;
    }
    return TRUE;
}

static _Thread_local Handle *self_handle;
static volatile LONG next_thread_id = 1;
static Handle main_handle = {H_THREAD};

static void *thread_entry(void *p) {
    Handle *h = (Handle *)p;
    self_handle = h;
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
    if (stack < (SIZE_T)PTHREAD_STACK_MIN) stack = 8u << 20;
    if (pthread_attr_setstacksize(&a, stack)) pthread_attr_setstacksize(&a, 8u << 20);
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
HANDLE GetCurrentThread(void) { return self_handle ? self_handle : &main_handle; }
HANDLE GetCurrentProcess(void) { return (HANDLE)(intptr_t)-1; }
static _Thread_local DWORD other_id;
DWORD GetCurrentThreadId(void) {
    if (self_handle) return self_handle->id;
    if (!other_id) other_id = (DWORD)InterlockedIncrement(&next_thread_id);
    return other_id;
}
DWORD GetCurrentProcessId(void) { return (DWORD)getpid(); }
DWORD GetThreadId(HANDLE t) { return t && ((Handle *)t)->kind == H_THREAD ? ((Handle *)t)->id : 0; }
BOOL SetThreadPriority(HANDLE t, int p) { (void)t; (void)p; return TRUE; }
int GetThreadPriority(HANDLE t) { (void)t; return 0; }
DWORD_PTR SetThreadAffinityMask(HANDLE t, DWORD_PTR m) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < CPU_SETSIZE && i < 64; i++)
        if (m & (1ULL << i)) CPU_SET(i, &set);
    Handle *h = (Handle *)t;
    pthread_t th = (h && h->kind == H_THREAD && h != &main_handle) ? h->t : pthread_self();
    pthread_setaffinity_np(th, sizeof set, &set);
    return m;
}
HRESULT SetThreadDescription(HANDLE t, LPCWSTR d) {
    (void)t;
    if (!d) return 0;
    char name[16];
    int i = 0;
    for (; d[i] && i < 15; i++) name[i] = d[i] < 128 ? (char)d[i] : '?';
    name[i] = 0;
    if (name[0]) pthread_setname_np(pthread_self(), name);
    return 0;
}
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
    if (x->kind == H_FILE) free(x);
    return TRUE;
}

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

static double mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

DWORD WaitForMultipleObjects(DWORD n, const HANDLE *hs, BOOL all, DWORD ms) {
    const double end = ms == INFINITE ? 1e300 : mono_ms() + ms;
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
        if (mono_ms() >= end) return WAIT_TIMEOUT;
        usleep(500);
    }
}

HANDLE CreateWaitableTimerExW(LPSECURITY_ATTRIBUTES sa, LPCWSTR name, DWORD flags, DWORD access) {
    (void)sa; (void)name; (void)flags; (void)access;
    return NULL;
}
BOOL SetWaitableTimerEx(HANDLE t, const LARGE_INTEGER *due, LONG period, void *fn, LPVOID arg, void *ctx, ULONG d) {
    (void)t; (void)due; (void)period; (void)fn; (void)arg; (void)ctx; (void)d;
    return FALSE;
}
void Sleep(DWORD ms) {
    if (!ms) sched_yield();
    else {
        struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000L};
        nanosleep(&ts, NULL);
    }
}
BOOL SwitchToThread(void) {
    sched_yield();
    return TRUE;
}

BOOL QueryPerformanceCounter(LARGE_INTEGER *c) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    c->QuadPart = (LONGLONG)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    return TRUE;
}
BOOL QueryPerformanceFrequency(LARGE_INTEGER *f) {
    f->QuadPart = 1000000000LL;
    return TRUE;
}
DWORD GetTickCount(void) { return (DWORD)mono_ms(); }
ULONGLONG GetTickCount64(void) { return (ULONGLONG)mono_ms(); }
DWORD timeGetTime(void) { return (DWORD)mono_ms(); }

void ExitProcess(UINT code) { _exit((int)code); }
void OutputDebugStringA(LPCSTR s) { fputs(s, stderr); }
DWORD GetModuleFileNameA(HMODULE m, LPSTR out, DWORD n) {
    (void)m;
    if (!out || n == 0) return 0;
    ssize_t got = readlink("/proc/self/exe", out, n - 1);
    if (got < 0) got = 0;
    out[got] = 0;
    return (DWORD)got;
}
HMODULE GetModuleHandleA(LPCSTR n) {
    if (n && !strcmp(n, "renderdoc.dll") && !dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI")) return NULL;
    return (HMODULE)1;
}
HMODULE LoadLibraryA(LPCSTR n) { (void)n; return NULL; }
void *GetProcAddress(HMODULE m, LPCSTR n) {
    (void)m;
    return n ? dlsym(RTLD_DEFAULT, n) : NULL;
}
USHORT RtlCaptureStackBackTrace(ULONG skip, ULONG n, PVOID *out, PULONG hash) {
    (void)skip; (void)n; (void)out; (void)hash;
    return 0;
}
SHORT GetAsyncKeyState(int vk) { return web_keys[vk & 255] ? (SHORT)0x8000 : 0; }
static volatile int linux_focused = 1;
void linux_set_focused(int on) { linux_focused = on; }
HWND GetForegroundWindow(void) { return linux_focused ? (HWND)1 : NULL; }
int MultiByteToWideChar(UINT cp, DWORD flags, LPCSTR s, int n, LPWSTR out, int outn) {
    (void)cp; (void)flags;
    if (n < 0) n = (int)strlen(s) + 1;
    if (!outn) return n;
    int i = 0;
    for (; i < n && i < outn; i++) out[i] = (unsigned char)s[i];
    return i;
}

/* Whole allocations, so MEM_RELEASE (size 0) can unmap what VirtualAlloc(NULL) created. Decommit does not go
   through here: it keeps the address and only drops the pages. */
enum { VA_BLOCKS = 128 };
static struct { void *p; size_t n; } va_blocks[VA_BLOCKS];
static pthread_mutex_t va_mu = PTHREAD_MUTEX_INITIALIZER;

static void va_track(void *p, size_t n) {
    pthread_mutex_lock(&va_mu);
    for (int i = 0; i < VA_BLOCKS; i++)
        if (!va_blocks[i].p) {
            va_blocks[i].p = p;
            va_blocks[i].n = n;
            pthread_mutex_unlock(&va_mu);
            return;
        }
    pthread_mutex_unlock(&va_mu);
}
static size_t va_forget(void *p) {
    size_t n = 0;
    pthread_mutex_lock(&va_mu);
    for (int i = 0; i < VA_BLOCKS; i++)
        if (va_blocks[i].p == p) {
            n = va_blocks[i].n;
            va_blocks[i].p = NULL;
            va_blocks[i].n = 0;
            break;
        }
    pthread_mutex_unlock(&va_mu);
    return n;
}

LPVOID VirtualAlloc(LPVOID at, SIZE_T n, DWORD type, DWORD prot) {
    (void)prot;
    if (!n) n = 1;
    /* Recommit of a range this process already reserved. Decommit only discarded the pages (madvise), so the
       address is still mapped; a fresh commit of decommitted pages reads as zeros, which DONTNEED already does. */
    if (at) {
        if ((type & MEM_COMMIT) && madvise(at, n, MADV_DONTNEED)) return NULL;
        return at;
    }
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    va_track(p, n);
    return p;
}
BOOL VirtualFree(LPVOID at, SIZE_T n, DWORD type) {
    if (!at) return FALSE;
    if (type & MEM_RELEASE) {
        size_t sz = va_forget(at);
        if (!sz) return FALSE;
        return munmap(at, sz) == 0;
    }
    /* MEM_DECOMMIT: the address stays reserved. munmap would let the next commit return the same pointer with
       nothing behind it, and the renderer's recycled batch buffers then fault on the first draw that fills them. */
    if (!n) return TRUE;
    return madvise(at, n, MADV_DONTNEED) == 0;
}
BOOL VirtualProtect(LPVOID at, SIZE_T n, DWORD prot, PDWORD old) {
    int p = PROT_READ | PROT_WRITE;
    if (prot == PAGE_NOACCESS) p = PROT_NONE;
    else if (prot == PAGE_READONLY) p = PROT_READ;
    if (old) *old = PAGE_READWRITE;
    if (!at || !n) return TRUE;
    return mprotect(at, n, p) == 0;
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
extern uint8_t a64_dirty[1u << 20];
UINT GetWriteWatch(DWORD flags, PVOID base, SIZE_T n, PVOID *addrs, ULONG_PTR *count, ULONG *gran) {
    const uint32_t p0 = (uint32_t)((uintptr_t)base >> 12), p1 = (uint32_t)(((uintptr_t)base + n + 4095) >> 12);
    ULONG_PTR k = 0;
    const ULONG_PTR cap = *count;
    for (uint32_t p = p0; p < p1 && p < (1u << 20) && k < cap;) {
        if (!(p & 7) && p + 8 <= p1) {
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

static void norm_slashes(const char *in, char *out, size_t cap) {
    size_t i = 0;
    if (!in) in = "";
    for (; in[i] && i + 1 < cap; i++) out[i] = in[i] == '\\' ? '/' : in[i];
    out[i] = 0;
}
static int access_ok(const char *p) {
    struct stat st;
    return p && p[0] && stat(p, &st) == 0;
}
static void append_comp(char *acc, size_t cap, const char *comp) {
    const size_t n = strlen(acc);
    const int slash = n > 0 && acc[n - 1] != '/';
    snprintf(acc + n, cap - n, "%s%s", slash ? "/" : "", comp);
}
static int resolve_ci(const char *in, char *out, size_t cap) {
    if (!in || !in[0]) return 0;
    if (access_ok(in)) {
        snprintf(out, cap, "%s", in);
        return 1;
    }
    char acc[4096] = "";
    const char *p = in;
    if (*p == '/') {
        snprintf(acc, sizeof acc, "/");
        p++;
    }
    while (*p) {
        while (*p == '/') p++;
        if (!*p) break;
        char comp[512];
        size_t c = 0;
        while (*p && *p != '/' && c + 1 < sizeof comp) comp[c++] = *p++;
        comp[c] = 0;
        char trial[4096];
        snprintf(trial, sizeof trial, "%s", acc);
        append_comp(trial, sizeof trial, comp);
        if (access_ok(trial)) {
            snprintf(acc, sizeof acc, "%s", trial);
            continue;
        }
        DIR *d = opendir(acc[0] ? acc : ".");
        if (!d) return 0;
        struct dirent *e;
        int found = 0;
        while ((e = readdir(d))) {
            if (!strcasecmp(e->d_name, comp)) {
                snprintf(trial, sizeof trial, "%s", acc);
                append_comp(trial, sizeof trial, e->d_name);
                snprintf(acc, sizeof acc, "%s", trial);
                found = 1;
                break;
            }
        }
        closedir(d);
        if (!found) return 0;
    }
    if (!acc[0]) return 0;
    snprintf(out, cap, "%s", acc);
    return 1;
}
static void path_for(const char *path, int creating, char *out, size_t cap);

int hos_linux_native_path(const char *in, char *out, size_t cap) {
    path_for(in, 0, out, cap);
    return out && out[0];
}

static void path_for(const char *path, int creating, char *out, size_t cap) {
    char norm[4096];
    norm_slashes(path, norm, sizeof norm);
    if (!creating && resolve_ci(norm, out, cap)) return;
    if (creating) {
        char *slash = strrchr(norm, '/');
        if (slash && slash != norm) {
            *slash = 0;
            char parent[4096];
            if (resolve_ci(norm, parent, sizeof parent)) {
                snprintf(out, cap, "%s/%s", parent, slash + 1);
                return;
            }
            *slash = '/';
        }
    }
    snprintf(out, cap, "%s", norm);
}

HANDLE CreateFileA(LPCSTR path, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa, DWORD disp, DWORD flags,
                   HANDLE tmpl) {
    (void)share; (void)sa; (void)flags; (void)tmpl;
    const int creating = disp == CREATE_NEW || disp == CREATE_ALWAYS || disp == OPEN_ALWAYS;
    char lp[4096];
    path_for(path, creating, lp, sizeof lp);
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
        last_error = errno == EEXIST ? ERROR_FILE_EXISTS : errno == EACCES ? ERROR_ACCESS_DENIED : ERROR_FILE_NOT_FOUND;
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
    const ssize_t r = pread(h->fd, buf, n, at);
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
    const ssize_t r = pwrite(h->fd, buf, n, at);
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
BOOL FlushFileBuffers(HANDLE x) {
    Handle *h = (Handle *)x;
    if (h && h->fd >= 0) fsync(h->fd);
    return TRUE;
}
DWORD GetFileAttributesA(LPCSTR path) {
    char lp[4096];
    path_for(path, 0, lp, sizeof lp);
    struct stat st;
    if (stat(lp, &st)) return INVALID_FILE_ATTRIBUTES;
    return S_ISDIR(st.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL;
}
BOOL CreateDirectoryA(LPCSTR path, LPSECURITY_ATTRIBUTES sa) {
    (void)sa;
    char lp[4096];
    path_for(path, 1, lp, sizeof lp);
    if (mkdir(lp, 0777)) {
        last_error = errno == EEXIST ? ERROR_ALREADY_EXISTS : ERROR_PATH_NOT_FOUND;
        return FALSE;
    }
    return TRUE;
}
BOOL DeleteFileA(LPCSTR path) {
    char lp[4096];
    path_for(path, 0, lp, sizeof lp);
    return unlink(lp) == 0;
}
BOOL RemoveDirectoryA(LPCSTR path) {
    char lp[4096];
    path_for(path, 0, lp, sizeof lp);
    return rmdir(lp) == 0;
}
BOOL MoveFileExA(LPCSTR from, LPCSTR to, DWORD flags) {
    (void)flags;
    char a[4096], b[4096];
    path_for(from, 0, a, sizeof a);
    path_for(to, 1, b, sizeof b);
    return rename(a, b) == 0;
}

static void add_found(Handle *f, const char *name, uint32_t attr, int64_t size) {
    for (int i = 0; i < f->count; i++)
        if (!strcasecmp(f->names[i], name)) return;
    f->names = (char **)realloc(f->names, (size_t)(f->count + 1) * sizeof(char *));
    f->attrs = (uint32_t *)realloc(f->attrs, (size_t)(f->count + 1) * sizeof(uint32_t));
    f->sizes = (int64_t *)realloc(f->sizes, (size_t)(f->count + 1) * sizeof(int64_t));
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
    char dir[4096];
    norm_slashes(pattern, dir, sizeof dir);
    char *slash = strrchr(dir, '/');
    if (slash) *slash = 0;
    else dir[0] = 0;
    char resolved[4096];
    const char *use = dir;
    if (dir[0] && resolve_ci(dir, resolved, sizeof resolved)) use = resolved;
    Handle *f = new_handle(H_FIND);
    DIR *d = opendir(use[0] ? use : ".");
    if (d) {
        struct dirent *e;
        while ((e = readdir(d))) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
            char full[8192];
            snprintf(full, sizeof full, "%s/%s", use[0] ? use : ".", e->d_name);
            struct stat st;
            if (stat(full, &st)) continue;
            add_found(f, e->d_name, S_ISDIR(st.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL, st.st_size);
        }
        closedir(d);
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

static SDL_AudioStream *audio_stream;
MMRESULT waveOutOpen(HWAVEOUT *w, UINT d, const WAVEFORMATEX *f, DWORD_PTR cb, DWORD_PTR u, DWORD fl) {
    (void)d; (void)cb; (void)u; (void)fl;
    if (!f || f->nChannels != 2 || f->wBitsPerSample != 16) return MMSYSERR_NODRIVER;
    SDL_InitSubSystem(SDL_INIT_AUDIO);
    SDL_AudioSpec spec;
    SDL_zero(spec);
    spec.format = SDL_AUDIO_S16LE;
    spec.channels = 2;
    spec.freq = (int)f->nSamplesPerSec;
    audio_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, NULL, NULL);
    if (!audio_stream) return MMSYSERR_NODRIVER;
    SDL_ResumeAudioStreamDevice(audio_stream);
    *w = (HWAVEOUT)audio_stream;
    return MMSYSERR_NOERROR;
}
MMRESULT waveOutWrite(HWAVEOUT w, WAVEHDR *h, UINT n) {
    (void)w; (void)n;
    if (audio_stream && h && h->lpData && h->dwBufferLength)
        SDL_PutAudioStreamData(audio_stream, h->lpData, (int)h->dwBufferLength);
    if (h) h->dwFlags |= WHDR_DONE;
    return MMSYSERR_NOERROR;
}
