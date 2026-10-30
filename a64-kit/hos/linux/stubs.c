/* Stand-ins for hos_prof.c on Linux, plus the Microsoft C runtime names the OS layer calls. */
#define _GNU_SOURCE
#include "hos.h"
#include <execinfo.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <windows.h>

void hos_host_thread_register_self(const char *name) { (void)name; }
void hos_memory_map_log(void) {}
void hos_process_memory(uint64_t *private_bytes, uint64_t *working_set) {
    unsigned long size = 0, resident = 0;
    FILE *f = fopen("/proc/self/statm", "r");
    *private_bytes = 0;
    *working_set = 0;
    if (f && fscanf(f, "%lu %lu", &size, &resident) == 2) {
        const long page = sysconf(_SC_PAGESIZE);
        *working_set = (uint64_t)resident * (uint64_t)(page > 0 ? page : 4096);
        *private_bytes = *working_set;
    }
    if (f) fclose(f);
}
void hos_backtrace(void) {
    void *frames[32];
    const int n = backtrace(frames, 32);
    backtrace_symbols_fd(frames, n, 2);
}
void hos_mem_checkpoint(const char *what) { (void)what; }
void hos_prof_start(void) {}
int hos_ab(void) { return 0; }

DWORD GetWindowThreadProcessId(HWND w, LPDWORD pid) {
    if (pid) *pid = w ? GetCurrentProcessId() : 0;
    return 1;
}
LONG_PTR GetWindowLongPtrA(HWND w, int index) {
    (void)w;
    (void)index;
    return 0;
}
int GetClassNameA(HWND w, LPSTR out, int n) {
    (void)w;
    if (!out || n <= 0) return 0;
    snprintf(out, (size_t)n, "Skybound");
    return (int)strlen(out);
}
int localtime_s(struct tm *out, const time_t *t) { return localtime_r(t, out) ? 0 : 22; }
int gmtime_s(struct tm *out, const time_t *t) { return gmtime_r(t, out) ? 0 : 22; }
time_t _mkgmtime(struct tm *t) { return timegm(t); }
char *_strdup(const char *s) { return strdup(s); }

static void crash_signal(int sig, siginfo_t *si, void *ctx) {
    (void)ctx;
    const char *name = "signal";
    if (sig == SIGSEGV) name = "SIGSEGV";
    else if (sig == SIGBUS) name = "SIGBUS";
    else if (sig == SIGFPE) name = "SIGFPE";
    else if (sig == SIGABRT) name = "SIGABRT";
    fprintf(stderr, "\n[hos] %s at %p\n", name, si ? si->si_addr : NULL);
    fflush(stderr);
    hos_backtrace();
    _exit(128 + sig);
}

__attribute__((constructor)) static void install_crash_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = crash_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}
