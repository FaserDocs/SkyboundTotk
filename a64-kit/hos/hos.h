/* Horizon OS layer (HLE of the Nintendo SDK at the nn:: boundary) for a64-kit games.

   An import or SDK function the game calls is a C function `void imp_<mangled name>(Ctx *c)` (HLE(...) below):
   arguments in c->x[0..7], the result in c->x[0]. A symbol with such a definition is hooked both where the game
   imports it and at the sdk's own definition, so lifted SDK code reaching it lands here too (nx_image.py).
   Everything without a definition gets a generated stub that logs its first call and returns 0.

   Guest address map (all below 4 GB, see PORT.md):
     0x01000000 - 0x07000000   system region: our own guest objects (main stack, TLS blocks, thread regions)
     0x07000000                trampolines of imports nothing defines (no memory behind them)
     0x08000000 - image end    main, sdk (nx_image.MAIN_BASE)
     0x10000000 - ...          heap (SetMemoryHeapSize), then address regions (AllocateAddressRegion) */
#ifndef HOS_H
#define HOS_H
#include "a64.h"
#include <stdarg.h>

#define HLE(sym) void imp_##sym(Ctx *c)
/* ENTRY_HOOK(0x0xxxxxxx) { ... }: runs at the entry of the lifted guest function at that address (8 hex digits),
   with its arguments in c; return 1 when it handled the call (the function then returns at once, x0 is the result),
   0 to run the function as usual. build.py splices the call into the lifted code (only that file recompiles). */
#define ENTRY_HOOK(addr) int entry_##addr(Ctx *c)
#define ENTRY_HOOK_SYM(sym) int entry_sym_##sym(Ctx *c)
/* SPLICE_AFTER(0x0xxxxxxx, "text") { ... }: runs inside the lifted guest function at that address, right after the
   first line of its lifted C that contains the text (a game patch in the middle of a function: c holds the guest
   registers there). build.py splices the call in and stops the build if the text is not found. One per function. */
#define SPLICE_AFTER(addr, anchor) void splice_##addr(Ctx *c)
/* SPLICE_AFTER_EACH(0x0xxxxxxx, "text") { ... }: the same after every line of that function with the text */
#define SPLICE_AFTER_EACH(addr, anchor) void splice_##addr(Ctx *c)
#define ARG(n) (c->x[n])
#define RET(v) (c->x[0] = (uint64_t)(v))
#define GP(a) ((void *)A64_PTR(a))           /* host pointer of a guest address */
#define G64(a) (*(uint64_t *)GP(a))
#define G32(a) (*(uint32_t *)GP(a))

#define HOS_SYSTEM_BASE 0x01000000ULL
#define HOS_SYSTEM_END 0x07000000ULL
#define HOS_HEAP_BASE 0x10000000ULL
#if defined(A64_WRITE_TRACK) && !defined(__EMSCRIPTEN__)
void hos_mark_written(const void *p, size_t n);   /* (hos_mem.c: guest memory written by host code, native soft watch) */
#define HOS_WRITTEN(p, n) hos_mark_written((p), (n))
#elif defined(A64_WRITE_TRACK)
void web_mark_written(const void *p, size_t n);   /* (hos/web/win32.c: guest memory written by host code) */
#define HOS_WRITTEN(p, n) web_mark_written((p), (n))
#else
#define HOS_WRITTEN(p, n) ((void)0)
#endif
#ifdef A64_MEM_ZERO
extern uint64_t hos_space_end;   /* web: the guest window ends where the host's own data starts (hos_main.c) */
#define HOS_SPACE_END hos_space_end
#else
#define HOS_SPACE_END 0xFFFF0000ULL
#endif
#define HOS_TICK_HZ 19200000ULL

/* ---- logging: HOS_TRACE=1 logs every HLE call that uses HOS_TRACE_CALL */
extern int hos_trace;
void hos_log(const char *fmt, ...);
void hos_log_flush(void);
#define HOS_TRACE_CALL(name) do { if (hos_trace) hos_log("%s(%#llx, %#llx, %#llx, %#llx) lr %#llx", name, \
    (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2], \
    (unsigned long long)c->x[3], (unsigned long long)c->x[30]); } while (0)
void hos_fatal(Ctx *c, const char *fmt, ...);

/* ---- guest memory (hos_mem.c) */
uint64_t hos_sys_alloc(uint64_t size, uint64_t align);     /* zeroed, from the system region, never freed */
uint64_t hos_region_alloc(uint64_t size, uint64_t align);  /* address space above the heap */
void hos_region_free(uint64_t addr);
void hos_mem_init(uint64_t image_end);
extern uint64_t hos_heap_size, hos_heap_used, hos_total_memory, hos_program_size;

/* ---- threads (hos_thread.c) */
typedef struct HosThread HosThread;
HosThread *hos_self(void);
Ctx *hos_ctx(void);
uint64_t hos_thread_guest(HosThread *t);                   /* its nn::os::ThreadType in guest memory */
uint64_t hos_thread_id(HosThread *t);
HosThread *hos_thread_of(uint64_t guest_thread_type);      /* NULL if unknown */
HosThread *hos_thread_create(uint64_t guest_tt, uint64_t entry, uint64_t arg, uint64_t stack, uint64_t stack_size,
                             int priority, int core);
void hos_thread_start(HosThread *t);
void hos_thread_join(HosThread *t);
void hos_thread_destroy(HosThread *t);
void hos_thread_set_name(HosThread *t, const char *name);
void hos_thread_pin(void *h, const char *name);   /* HOS_PIN: host CPU affinity by thread name */
int hos_game_cores(void);   /* totk_cores.c: HOS_GAME_CORES, or 0 */
const char *hos_thread_name(HosThread *t);
void hos_thread_affinity(HosThread *t, int ideal, uint64_t mask);
uint64_t hos_thread_mask(HosThread *t);
int hos_thread_priority(HosThread *t, int set, int value);
int hos_thread_core(HosThread *t);
int hos_cores(void);
void hos_thread_boost(int level);
void hos_thread_stack(HosThread *t, uint64_t *addr, uint64_t *size);
uint64_t *hos_tls_slots(HosThread *t);                     /* nn::os TLS slot values */
#define HOS_TLS_SLOTS 32
Ctx *hos_boot_main_thread(uint64_t stack_size);           /* the calling host thread becomes the guest main thread */
/* the kernel side: Horizon handles and blocking for hos_svc.c's arbitration */
uint32_t hos_thread_handle(HosThread *t);
HosThread *hos_thread_by_handle(uint32_t h);
int hos_thread_prio(HosThread *t);
void hos_thread_wait(HosThread *t, int64_t timeout_ns, int *timed_out);   /* < 0: forever */
void hos_thread_wake(HosThread *t);

/* ---- for host-side devices: game files by guest path (nn_fs.c), guest events (nn_os_sync.c) */
int hos_fs_host_path(const char *guest, char *out, size_t cap);
void hos_event_signal(uint64_t guest_event);

/* ---- videos (movie_mf.c): a video file read as compressed samples or as decoded NV12 frames */
typedef struct MovieReader MovieReader;
MovieReader *movie_reader_open(const char *host_path, int decoded, int *width, int *height, int64_t *duration_us);
/* compressed: the next sample into dst; its size, 0 at the end, -1 on an error or if it does not fit */
int movie_reader_sample(MovieReader *r, void *dst, int cap, int64_t *time_us);
/* decoded: the next frame as NV12 (width x height luma rows of `width` bytes, then the interleaved chroma rows);
   1, or 0 at the end */
int movie_reader_frame(MovieReader *r, uint8_t *dst, int64_t *time_us);
void movie_reader_close(MovieReader *r);
uint32_t movie_reader_codec(MovieReader *r);   /* the stream's FourCC: 'VP90', 'VP80', 'H264', ... */

/* ---- time */
uint64_t hos_ticks(void);
void hos_sleep_ns(int64_t ns);
int64_t hos_now_ns(void);
#endif
