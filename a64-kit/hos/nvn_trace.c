/* NVN entry: the real NVN driver (lifted SDK code, over hos/nvdrv.c) serves every proc. nvnBootstrapLoader is hooked
   only to hand those procs out, through the driver's own resolver nvnBootstrapLoaderInternal (sdk+0x490110, no
   device state needed). The real nvnBootstrapLoader is skipped because it first builds a loader object we have no
   use for.

   HOS_NVN_TRACE=1: every proc becomes a host call that records the call (the host-call ring, printed by
   HOS_HOSTCALL_STATS) and forwards it untouched, so the NVN traffic of a run can be read. procs returned by
   nvnDeviceGetProcAddress are wrapped the same way.
   HOS_NVN_LOG=nvnQueueSubmitCommands,nvnSyncWait,...: those procs (wrapped as above) are also logged as they are
   called, with the thread, arguments and result. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

uint64_t hos_hostcall(const char *name, void (*fn)(Ctx *, void *), void *user);
uint64_t hos_call_guest(Ctx *c, uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);
extern const uint64_t hos_entry_nvnBootstrapLoaderInternal;
static int tracing = -1;
static const char *log_names;   /* HOS_NVN_LOG */

typedef struct { uint64_t real; const char *name; int log; } Proc;

/* HOS_NVN_INST_STATS=1 (debugging): nvnCommandBufferDraw{Elements,Arrays}Instanced summed by caller every 10 s
   (calls, instances, largest instance count, vertex count of the largest) */
static struct { uint64_t lr, calls, inst, max_inst, max_count; } inst_by[128];
static volatile LONG inst_lock;
static int64_t inst_last;
static void inst_note(uint64_t lr, uint64_t inst, uint64_t count) {
    while (InterlockedExchange(&inst_lock, 1)) {}
    int k;
    for (k = 0; k < 128 && inst_by[k].lr && inst_by[k].lr != lr; k++) {}
    if (k < 128) {
        inst_by[k].lr = lr;
        inst_by[k].calls++;
        inst_by[k].inst += inst;
        if (inst > inst_by[k].max_inst) inst_by[k].max_inst = inst, inst_by[k].max_count = count;
    }
    const int64_t now = hos_now_ns();
    if (now - inst_last > 10000000000LL) {
        inst_last = now;
        for (k = 0; k < 128 && inst_by[k].lr; k++)
            hos_log("nvn: instanced draws from %#llx: %llu calls, %llu instances, largest %llu (count %llu)",
                    (unsigned long long)inst_by[k].lr, (unsigned long long)inst_by[k].calls,
                    (unsigned long long)inst_by[k].inst, (unsigned long long)inst_by[k].max_inst,
                    (unsigned long long)inst_by[k].max_count);
    }
    InterlockedExchange(&inst_lock, 0);
}

/* The proxy: arguments are still in x0-x7, so just run the real proc. */
static void forward(Ctx *c, void *u) {
    Proc *p = (Proc *)u;
    uint64_t a[6] = {c->x[0], c->x[1], c->x[2], c->x[3], c->x[4], c->x[5]};
    const uint64_t lr = c->x[30];
    if (p->log == 2 && a[4]) {   /* (debugging) nvnCommandBufferSetRenderTargets: each new depth texture's object */
        static uint64_t seen[256];
        static int nseen;
        int k;
        for (k = 0; k < nseen && seen[k] != a[4]; k++) {}
        if (k == nseen && nseen < 256) {
            const uint32_t *w = (const uint32_t *)GP(a[4]);
            seen[nseen++] = a[4];
            hos_log("nvn: depth texture %#llx (from %#llx): %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                    "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x", (unsigned long long)a[4],
                    (unsigned long long)lr, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11],
                    w[12], w[13], w[14], w[15], w[16], w[17], w[18], w[19], w[20], w[21], w[22], w[23]);
        }
    }
    if (p->log == 4) {   /* HOS_NVN_RT_CALLERS: render target changes by caller (count: colors) */
        inst_note(lr, 1, a[1]);
        static uint64_t bt_lr = 1;   /* HOS_NVN_RT_BT=<caller hex>: the first calls from there with the host stack */
        static int bt_n;
        if (bt_lr == 1) bt_lr = getenv("HOS_NVN_RT_BT") ? strtoull(getenv("HOS_NVN_RT_BT"), NULL, 16) : 0;
        if (bt_lr && lr == bt_lr && bt_n++ < 3) {
            hos_log("nvn: SetRenderTargets from %#llx:", (unsigned long long)lr);
            void hos_backtrace(void);
            hos_backtrace();
        }
    }
    if (p->log == 3) {   /* HOS_NVN_INST_STATS: instanced draws summed by caller */
        const uint64_t inst = strstr(p->name, "Elements") ? c->x[7] : c->x[5];
        inst_note(lr, inst, a[3]);
    }
    a64_icall(c, p->real);
    if (p->log == 1) {
        HosThread *t = hos_self();
        hos_log("nvn: %s(%#llx, %#llx, %#llx, %#llx, %#llx) -> %#llx [%s] from %#llx", p->name, (unsigned long long)a[0],
                (unsigned long long)a[1], (unsigned long long)a[2], (unsigned long long)a[3], (unsigned long long)a[4],
                (unsigned long long)c->x[0], t ? hos_thread_name(t) : "?", (unsigned long long)lr);
    }
}

static uint64_t wrap(const char *name, uint64_t real);

/* nvnDeviceGetProcAddress through the proxy: wrap what it returns too. */
static void forward_get_proc(Ctx *c, void *u) {
    uint64_t name = ARG(1);
    forward(c, u);
    if (name && c->x[0]) c->x[0] = wrap((const char *)GP(name), c->x[0]);
}

static int listed(const char *name) {
    if (!log_names) return 0;
    size_t n = strlen(name);
    for (const char *s = log_names; (s = strstr(s, name)) != NULL; s += n)
        if ((s == log_names || s[-1] == ',') && (s[n] == ',' || s[n] == 0)) return 1;
    return 0;
}

static uint64_t wrap(const char *name, uint64_t real) {
    if (!real || !tracing) return real;
    Proc *p = (Proc *)calloc(1, sizeof *p);
    p->real = real;
    p->name = _strdup(name);
    p->log = listed(name);
    if (p->log) hos_log("nvn: %s is at %#llx", name, (unsigned long long)real);
    if (p->log && !strcmp(name, "nvnCommandBufferSetRenderTargets")) p->log = 2;
    if (getenv("HOS_NVN_INST_STATS") && (!strcmp(name, "nvnCommandBufferDrawElementsInstanced") ||
                                         !strcmp(name, "nvnCommandBufferDrawArraysInstanced")))
        p->log = 3;
    if (getenv("HOS_NVN_RT_CALLERS") && !strcmp(name, "nvnCommandBufferSetRenderTargets")) p->log = 4;
    return hos_hostcall(p->name, strcmp(name, "nvnDeviceGetProcAddress") ? forward : forward_get_proc, p);
}

/* nvnBootstrapLoader(char const* name) -> proc */
uint64_t nvn_native_proc(const char *name);
int nvn_native_on(void);
HLE(nvnBootstrapLoader) {
    if (nvn_native_on()) {   /* HOS_NVN_NATIVE=1: the native NVN layer answers every proc (hos/nvn_native.c) */
        RET(ARG(0) ? nvn_native_proc((const char *)GP(ARG(0))) : 0);
        return;
    }
    if (tracing < 0) {
        log_names = getenv("HOS_NVN_LOG");
        tracing = getenv("HOS_NVN_TRACE") != NULL || log_names != NULL || getenv("HOS_NVN_INST_STATS") != NULL ||
                  getenv("HOS_NVN_RT_CALLERS") != NULL;
    }
    uint64_t name = ARG(0);
    uint64_t real = hos_call_guest(c, hos_entry_nvnBootstrapLoaderInternal, name, 0, 0, 0);
    if (!real) hos_log("nvn: no proc named %s", name ? (const char *)GP(name) : "(null)");
    RET(name ? wrap((const char *)GP(name), real) : real);
}
