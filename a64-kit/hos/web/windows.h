/* The browser build's stand-in for <windows.h>: the part of Win32 the OS layer and the renderer use, on pthreads and
   Emscripten. Locks, condition variables and atomics are inline; threads, events, files (the RomFS over HTTP, saves
   in memory), timers and the rest are in hos/web/win32.c. */
#pragma once
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WINAPI
#define CALLBACK
#define APIENTRY
#define WINBASEAPI
#define __stdcall
#define __cdecl
#define __declspec(x)
#define __forceinline inline __attribute__((always_inline))
#include <strings.h>
#define _stricmp strcasecmp
#define _strnicmp strncasecmp
#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
#endif

typedef int BOOL;
typedef unsigned char BYTE, UCHAR, BOOLEAN;
typedef unsigned short WORD, USHORT;
typedef short SHORT;
typedef unsigned int DWORD, UINT, ULONG;   /* (LP64 wasm: long is 32-bit too, but keep Windows' 32-bit DWORD) */
typedef int LONG, INT;
typedef int64_t LONG64, LONGLONG, INT64;
typedef uint64_t ULONG64, ULONGLONG, DWORD64, UINT64;
typedef char CHAR;
typedef uint16_t WCHAR;
typedef void *HANDLE, *HWND, *HINSTANCE, *HMODULE, *HDC, *HBRUSH, *HCURSOR, *HICON, *HMENU, *HGDIOBJ, *HRESULT_PTR;
typedef void *LPVOID, *PVOID;
typedef const void *LPCVOID;
typedef size_t SIZE_T;
typedef uintptr_t ULONG_PTR, DWORD_PTR, UINT_PTR;
typedef intptr_t LONG_PTR, INT_PTR;
typedef DWORD *LPDWORD, *PDWORD;
typedef ULONG *PULONG;
typedef LONG *PLONG;
typedef char *LPSTR;
typedef const char *LPCSTR;
typedef const WCHAR *LPCWSTR;
typedef WCHAR *LPWSTR;
typedef long HRESULT;
typedef BYTE *LPBYTE;
typedef ULONG_PTR *PULONG_PTR;
typedef struct { LONG left, top, right, bottom; } RECT;

typedef union {
    struct { DWORD LowPart; LONG HighPart; };
    struct { DWORD LowPart; LONG HighPart; } u;
    LONGLONG QuadPart;
} LARGE_INTEGER, *PLARGE_INTEGER;
typedef struct { DWORD dwLowDateTime, dwHighDateTime; } FILETIME;
typedef struct {
    ULONG_PTR Internal, InternalHigh;
    union {
        struct { DWORD Offset, OffsetHigh; };
        PVOID Pointer;
    };
    HANDLE hEvent;
} OVERLAPPED, *LPOVERLAPPED;
typedef struct { DWORD nLength; LPVOID lpSecurityDescriptor; BOOL bInheritHandle; } SECURITY_ATTRIBUTES, *LPSECURITY_ATTRIBUTES;
#define MAX_PATH 260
typedef struct {
    DWORD dwFileAttributes;
    FILETIME ftCreationTime, ftLastAccessTime, ftLastWriteTime;
    DWORD nFileSizeHigh, nFileSizeLow, dwReserved0, dwReserved1;
    CHAR cFileName[MAX_PATH];
    CHAR cAlternateFileName[14];
} WIN32_FIND_DATAA, *LPWIN32_FIND_DATAA;
typedef struct {
    PVOID BaseAddress, AllocationBase;
    DWORD AllocationProtect;
    SIZE_T RegionSize;
    DWORD State, Protect, Type;
} MEMORY_BASIC_INFORMATION;
typedef DWORD (*LPTHREAD_START_ROUTINE)(LPVOID);

#define TRUE 1
#define FALSE 0
#define INFINITE 0xFFFFFFFFu
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#define FILE_ATTRIBUTE_READONLY 0x1
#define FILE_ATTRIBUTE_DIRECTORY 0x10
#define FILE_ATTRIBUTE_NORMAL 0x80
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 1
#define FILE_SHARE_WRITE 2
#define FILE_SHARE_DELETE 4
#define CREATE_NEW 1
#define CREATE_ALWAYS 2
#define OPEN_EXISTING 3
#define OPEN_ALWAYS 4
#define TRUNCATE_EXISTING 5
#define FILE_BEGIN 0
#define FILE_CURRENT 1
#define FILE_END 2
#define FILE_FLAG_RANDOM_ACCESS 0x10000000
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000
#define MOVEFILE_REPLACE_EXISTING 1
#define ERROR_SUCCESS 0
#define ERROR_FILE_NOT_FOUND 2
#define ERROR_PATH_NOT_FOUND 3
#define ERROR_ACCESS_DENIED 5
#define ERROR_INVALID_HANDLE 6
#define ERROR_NO_MORE_FILES 18
#define ERROR_HANDLE_EOF 38
#define ERROR_FILE_EXISTS 80
#define ERROR_ALREADY_EXISTS 183
#define ERROR_TIMEOUT 1460
#define ERROR_DEVICE_NOT_CONNECTED 1167
#define WAIT_OBJECT_0 0
#define WAIT_TIMEOUT 258
#define WAIT_FAILED 0xFFFFFFFFu
#define MEM_COMMIT 0x1000
#define MEM_RESERVE 0x2000
#define MEM_RELEASE 0x8000
#define MEM_DECOMMIT 0x4000
#define MEM_WRITE_WATCH 0x200000
#define PAGE_NOACCESS 1
#define PAGE_READONLY 2
#define PAGE_READWRITE 4
#define PAGE_EXECUTE_READWRITE 0x40
#define WRITE_WATCH_FLAG_RESET 1
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 2
#define TIMER_ALL_ACCESS 0x1F0003
#define THREAD_PRIORITY_NORMAL 0
#define THREAD_PRIORITY_ABOVE_NORMAL 1
#define THREAD_PRIORITY_HIGHEST 2
#define THREAD_PRIORITY_TIME_CRITICAL 15
#define THREAD_PRIORITY_BELOW_NORMAL -1
#define THREAD_PRIORITY_LOWEST -2
#define THREAD_PRIORITY_IDLE -15
#define CREATE_SUSPENDED 4
#define STACK_SIZE_PARAM_IS_A_RESERVATION 0x10000
#define DUPLICATE_SAME_ACCESS 2
#define CONDITION_VARIABLE_LOCKMODE_SHARED 1
#define S_OK 0
#define SUCCEEDED(h) ((h) >= 0)
#define FAILED(h) ((h) < 0)
#define MAKELONG(a, b) ((LONG)(((WORD)(a)) | ((DWORD)((WORD)(b))) << 16))
#define LOWORD(l) ((WORD)((DWORD_PTR)(l) & 0xffff))
#define HIWORD(l) ((WORD)((DWORD_PTR)(l) >> 16))

/* ---- locks and condition variables: SRW locks are pthread reader/writer locks (shared acquires may nest, as
   Windows allows); condition variables are a futex word (they wait with either kind of SRW hold) ---- */
typedef struct { pthread_rwlock_t l; } SRWLOCK, *PSRWLOCK;
typedef struct { volatile uint32_t seq; } CONDITION_VARIABLE, *PCONDITION_VARIABLE;
typedef struct { pthread_mutex_t m; } CRITICAL_SECTION, *LPCRITICAL_SECTION;
#define SRWLOCK_INIT {PTHREAD_RWLOCK_INITIALIZER}
#define CONDITION_VARIABLE_INIT {0}
static inline void InitializeSRWLock(SRWLOCK *l) { pthread_rwlock_init(&l->l, NULL); }
static inline void AcquireSRWLockExclusive(SRWLOCK *l) { pthread_rwlock_wrlock(&l->l); }
static inline void ReleaseSRWLockExclusive(SRWLOCK *l) { pthread_rwlock_unlock(&l->l); }
static inline void AcquireSRWLockShared(SRWLOCK *l) { pthread_rwlock_rdlock(&l->l); }
static inline void ReleaseSRWLockShared(SRWLOCK *l) { pthread_rwlock_unlock(&l->l); }
static inline BOOLEAN TryAcquireSRWLockExclusive(SRWLOCK *l) { return pthread_rwlock_trywrlock(&l->l) == 0; }
static inline void InitializeConditionVariable(CONDITION_VARIABLE *c) { c->seq = 0; }
void WakeConditionVariable(CONDITION_VARIABLE *c);
void WakeAllConditionVariable(CONDITION_VARIABLE *c);
BOOL SleepConditionVariableSRW(CONDITION_VARIABLE *c, SRWLOCK *l, DWORD ms, ULONG flags);
static inline void InitializeCriticalSection(CRITICAL_SECTION *s) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&s->m, &a);
}
static inline void EnterCriticalSection(CRITICAL_SECTION *s) { pthread_mutex_lock(&s->m); }
static inline void LeaveCriticalSection(CRITICAL_SECTION *s) { pthread_mutex_unlock(&s->m); }
static inline void DeleteCriticalSection(CRITICAL_SECTION *s) { pthread_mutex_destroy(&s->m); }

/* ---- atomics (Win32's results: Increment/Add give the new value, Exchange/ExchangeAdd/CompareExchange the old) ---- */
#define InterlockedIncrement(p) __atomic_add_fetch((p), 1, __ATOMIC_SEQ_CST)
#define InterlockedDecrement(p) __atomic_sub_fetch((p), 1, __ATOMIC_SEQ_CST)
#define InterlockedIncrement64(p) __atomic_add_fetch((p), 1, __ATOMIC_SEQ_CST)
#define InterlockedDecrement64(p) __atomic_sub_fetch((p), 1, __ATOMIC_SEQ_CST)
#define InterlockedAdd(p, v) __atomic_add_fetch((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedAdd64(p, v) __atomic_add_fetch((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchangeAdd(p, v) __atomic_fetch_add((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchangeAdd64(p, v) __atomic_fetch_add((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchange(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchange64(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedOr(p, v) __atomic_fetch_or((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedAnd(p, v) __atomic_fetch_and((p), (v), __ATOMIC_SEQ_CST)
#define InterlockedExchangePointer(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
static inline LONG InterlockedCompareExchange(volatile LONG *p, LONG x, LONG cmp) {
    __atomic_compare_exchange_n(p, &cmp, x, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return cmp;
}
static inline LONG64 InterlockedCompareExchange64(volatile LONG64 *p, LONG64 x, LONG64 cmp) {
    __atomic_compare_exchange_n(p, &cmp, x, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return cmp;
}
static inline PVOID InterlockedCompareExchangePointer(PVOID volatile *p, PVOID x, PVOID cmp) {
    __atomic_compare_exchange_n(p, &cmp, x, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return cmp;
}
#define MemoryBarrier() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define YieldProcessor() ((void)0)
#define _ReadWriteBarrier() __atomic_signal_fence(__ATOMIC_SEQ_CST)

/* ---- threads, events, waits (win32.c) ---- */
HANDLE CreateThread(LPSECURITY_ATTRIBUTES sa, SIZE_T stack, LPTHREAD_START_ROUTINE fn, LPVOID arg, DWORD flags, LPDWORD id);
DWORD ResumeThread(HANDLE t);
DWORD SuspendThread(HANDLE t);
void ExitThread(DWORD code);
HANDLE GetCurrentThread(void);
HANDLE GetCurrentProcess(void);
DWORD GetCurrentThreadId(void);
DWORD GetCurrentProcessId(void);
DWORD GetThreadId(HANDLE t);
BOOL SetThreadPriority(HANDLE t, int p);
int GetThreadPriority(HANDLE t);
DWORD_PTR SetThreadAffinityMask(HANDLE t, DWORD_PTR m);
HRESULT SetThreadDescription(HANDLE t, LPCWSTR d);
BOOL DuplicateHandle(HANDLE sp, HANDLE s, HANDLE tp, HANDLE *t, DWORD a, BOOL i, DWORD o);
BOOL CloseHandle(HANDLE h);
HANDLE CreateEventA(LPSECURITY_ATTRIBUTES sa, BOOL manual, BOOL initial, LPCSTR name);
BOOL SetEvent(HANDLE e);
BOOL ResetEvent(HANDLE e);
DWORD WaitForSingleObject(HANDLE h, DWORD ms);
DWORD WaitForMultipleObjects(DWORD n, const HANDLE *hs, BOOL all, DWORD ms);
HANDLE CreateWaitableTimerExW(LPSECURITY_ATTRIBUTES sa, LPCWSTR name, DWORD flags, DWORD access);
BOOL SetWaitableTimerEx(HANDLE t, const LARGE_INTEGER *due, LONG period, void *fn, LPVOID arg, void *ctx, ULONG delay);
void Sleep(DWORD ms);
BOOL SwitchToThread(void);
DWORD GetLastError(void);
void SetLastError(DWORD e);
BOOL QueryPerformanceCounter(LARGE_INTEGER *c);
BOOL QueryPerformanceFrequency(LARGE_INTEGER *f);
DWORD GetTickCount(void);
ULONGLONG GetTickCount64(void);
void ExitProcess(UINT code);
void OutputDebugStringA(LPCSTR s);
DWORD GetModuleFileNameA(HMODULE m, LPSTR out, DWORD n);
HMODULE GetModuleHandleA(LPCSTR n);
HMODULE LoadLibraryA(LPCSTR n);
void *GetProcAddress(HMODULE m, LPCSTR n);
USHORT RtlCaptureStackBackTrace(ULONG skip, ULONG n, PVOID *out, PULONG hash);
SHORT GetAsyncKeyState(int vk);
HWND GetForegroundWindow(void);

/* ---- memory: wasm memory is all there and all writable (no write watch: the renderer uploads what draws read) ---- */
LPVOID VirtualAlloc(LPVOID at, SIZE_T n, DWORD type, DWORD prot);
BOOL VirtualFree(LPVOID at, SIZE_T n, DWORD type);
BOOL VirtualProtect(LPVOID at, SIZE_T n, DWORD prot, PDWORD old);
SIZE_T VirtualQuery(LPCVOID at, MEMORY_BASIC_INFORMATION *out, SIZE_T n);
UINT GetWriteWatch(DWORD flags, PVOID base, SIZE_T n, PVOID *addrs, ULONG_PTR *count, ULONG *gran);
UINT ResetWriteWatch(LPVOID base, SIZE_T n);

/* ---- files (win32.c: the RomFS from the page's server, everything else in Emscripten's file system) ---- */
HANDLE CreateFileA(LPCSTR path, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa, DWORD disp, DWORD flags, HANDLE tmpl);
BOOL ReadFile(HANDLE h, LPVOID buf, DWORD n, LPDWORD got, LPOVERLAPPED ov);
BOOL WriteFile(HANDLE h, LPCVOID buf, DWORD n, LPDWORD put, LPOVERLAPPED ov);
BOOL GetFileSizeEx(HANDLE h, PLARGE_INTEGER size);
DWORD GetFileSize(HANDLE h, LPDWORD high);
BOOL SetFilePointerEx(HANDLE h, LARGE_INTEGER to, PLARGE_INTEGER now, DWORD how);
BOOL SetEndOfFile(HANDLE h);
BOOL FlushFileBuffers(HANDLE h);
DWORD GetFileAttributesA(LPCSTR path);
BOOL CreateDirectoryA(LPCSTR path, LPSECURITY_ATTRIBUTES sa);
BOOL DeleteFileA(LPCSTR path);
BOOL RemoveDirectoryA(LPCSTR path);
BOOL MoveFileExA(LPCSTR from, LPCSTR to, DWORD flags);
HANDLE FindFirstFileA(LPCSTR pattern, LPWIN32_FIND_DATAA fd);
BOOL FindNextFileA(HANDLE f, LPWIN32_FIND_DATAA fd);
BOOL FindClose(HANDLE f);
int MultiByteToWideChar(UINT cp, DWORD flags, LPCSTR s, int n, LPWSTR out, int outn);
#define CP_UTF8 65001
#define CP_ACP 0

/* ---- keys (the page writes their state into web_keys[]) ---- */
#define VK_TAB 0x09
#define VK_RETURN 0x0D
#define VK_SHIFT 0x10
#define VK_CONTROL 0x11
#define VK_MENU 0x12
#define VK_ESCAPE 0x1B
#define VK_SPACE 0x20
#define VK_LEFT 0x25
#define VK_UP 0x26
#define VK_RIGHT 0x27
#define VK_DOWN 0x28
#define VK_ADD 0x6B
#define VK_SUBTRACT 0x6D
#define VK_F1 0x70
#define VK_OEM_PLUS 0xBB
#define VK_OEM_MINUS 0xBD
#define VK_LBUTTON 0x01
#define VK_RBUTTON 0x02
#define SW_SHOWNORMAL 1
extern volatile uint8_t web_keys[256];

/* ---- structured exceptions (the GPU write watch diagnostic): never raised here ---- */
typedef struct { DWORD ExceptionCode, ExceptionFlags; void *ExceptionRecord; PVOID ExceptionAddress; DWORD NumberParameters;
                 ULONG_PTR ExceptionInformation[15]; } EXCEPTION_RECORD;
typedef struct { uint64_t Rip, Rsp; } CONTEXT;
typedef struct { EXCEPTION_RECORD *ExceptionRecord; CONTEXT *ContextRecord; } EXCEPTION_POINTERS, *PEXCEPTION_POINTERS;
#define EXCEPTION_ACCESS_VIOLATION 0xC0000005u
#define EXCEPTION_CONTINUE_SEARCH 0
#define EXCEPTION_CONTINUE_EXECUTION (-1)
typedef LONG (*PVECTORED_EXCEPTION_HANDLER)(EXCEPTION_POINTERS *);
static inline PVOID AddVectoredExceptionHandler(ULONG first, PVECTORED_EXCEPTION_HANDLER h) { (void)first; (void)h; return NULL; }

/* the Microsoft C runtime's (stubs.c) */
struct tm;
int localtime_s(struct tm *out, const time_t *t);
int gmtime_s(struct tm *out, const time_t *t);
time_t _mkgmtime(struct tm *t);
char *_strdup(const char *s);

/* the web layer's own (win32.c) */
void web_fs_init(const char *romfs_url, const char *romfs_dir);
void web_next_thread_takes_canvas(const char *selector);
void web_mark_written(const void *p, size_t n);   /* host code wrote guest memory (the renderer's write watch) */
int web_fetch(const char *url, uint64_t offset, uint64_t n, void *dst);
int web_fetch_all(const char *url, void *dst, uint64_t cap);   /* bytes read, or -1 */
int64_t web_fetch_size(const char *url);

#ifdef __cplusplus
}
#endif
