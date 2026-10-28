/* The GPU: Tegra X1 GM20B ("Maxwell B"), as the NVN driver drives it through GPFIFO submissions (hos/nvdrv.c).

   Step 1 of milestone 5, the command processor. A submission is executed at once, in the submitting thread:
   nvdrv advances the channel's sync point after hos_gpu_submit returns, so a fence the driver gets back means
   "done". It decodes the pushbuffers, keeps each channel's engine state, and performs on the CPU everything the
   game can observe without pixels:
     - puller (every channel): object binding, semaphore acquire/release, sync point increments;
     - 3D (B197): macros (the MME: uploaded programs that re-enter the 3D methods), shadow RAM, report semaphores
       (fences and queries: payload, timestamp, counters read as 0), sync_info sync point increments, constant
       buffer updates (CB_DATA writes memory), inline-to-memory uploads;
     - compute (B1C0) and inline-to-memory (A140): uploads;
     - copy engine (B0B5): pitch and block-linear copies, remapped/constant fills, semaphore release.
   Draws, clears, compute launches and 2D blits are counted. The renderer (gpu/render/, rx.h) receives the 3D
   methods that draw and clear, and the end of each submission.

   GPU memory is the game's own: nvdrv maps GPU VAs to guest addresses (hos_gpu_ptr).

   Behavior follows suyu's video_core (dma_pusher, puller, engines/*, macro/macro_interpreter; GPL-2.0-or-later);
   method numbers are NVIDIA's (open-gpu-doc cl906f / clb197 / clb0b5 / cla140 / clb1c0). Register indices below
   are word indices (method numbers). */
#include "hos.h"
void hos_backtrace(void);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

uint8_t *hos_gpu_ptr(uint64_t va, uint64_t *avail);   /* hos/nvdrv.c */
uint8_t *hos_guest_base(void);
void hos_gpu_syncpt_incr(uint32_t id);
#include "../gpu/render/rx.h"

enum { CLS_3D = 0xB197, CLS_COMPUTE = 0xB1C0, CLS_I2M = 0xA140, CLS_2D = 0x902D, CLS_DMA = 0xB0B5 };
enum { NREGS = 0x1000, MME_CODE = 0x4000, MME_ENTRIES = 0x80 };

typedef struct {
    uint8_t *buf;                 /* inline-to-memory: the data of the upload in progress */
    uint32_t size, pos, linear;
} Upload;

typedef struct Channel_ {
    SRWLOCK lock;
    uint32_t cls[8];              /* class bound to each subchannel */
    uint32_t puller[0x40];
    uint32_t reg[8][NREGS];       /* each subchannel's engine registers */
    Upload up[8];
    /* 3D only (one 3D object per channel) */
    uint32_t shadow[NREGS];
    uint32_t mme_code[MME_CODE], mme_entry[MME_ENTRIES];
    uint32_t macro_method, *macro_params, nparams, capparams;
    /* the command being decoded: its arguments may continue in the next GPFIFO entry */
    uint32_t cmd_method, cmd_sub, cmd_left, cmd_mode, cmd_first;
    /* the last methods run, for the report on an anomaly: {GPU VA of the word, method | sub << 16, argument} */
    struct { uint64_t va; uint32_t msub, arg; } ring[64];
    uint32_t ring_next;
    uint64_t cur_va;     /* GPU VA of the word being decoded */
    uint64_t entry_va;   /* the command list being run */
    uint32_t entry_words;
    void *rx;            /* the renderer's state for this channel */
    uint64_t cb_lo, cb_hi;   /* CB_DATA words written since the renderer was last told (GPU VA range) */
    int front;               /* split mode's parsing side (see "split mode" below) */
} Channel;

static Channel *channels[256];
static Channel *anomaly_ch;   /* the channel being run, for anomaly reports */
static SRWLOCK channels_lock = SRWLOCK_INIT;

/* statistics, logged every 5 s with HOS_GPU_STATS=1 or on HOS_GPU_TRACE */
static volatile LONG64 st_submits, st_words, st_draws, st_clears, st_dispatches, st_dma, st_i2m, st_sem, st_macros,
    st_blits, st_unmapped;
static int gpu_trace = -1, gpu_stats = -1;
static volatile LONG64 traced;   /* methods logged so far under HOS_GPU_TRACE */
static int64_t st_last;

static void dump_ring(void) {
    Channel *ch = anomaly_ch;
    if (!ch) return;
    {   /* the command list as raw words, 8 per line */
        uint64_t avail;
        const uint32_t *w = (const uint32_t *)hos_gpu_ptr(ch->entry_va, &avail);
        hos_log("gpu: command list %#llx, %u words:", (unsigned long long)ch->entry_va, ch->entry_words);
        for (uint32_t i = 0; w && i < ch->entry_words && i < 512; i += 8) {
            char line[160];
            int k = snprintf(line, sizeof line, "gpu:   +%03x:", i * 4);
            for (uint32_t j = i; j < i + 8 && j < ch->entry_words; j++) k += snprintf(line + k, sizeof line - k, " %08x", w[j]);
            hos_log("%s", line);
        }
    }
    hos_log("gpu: the last methods on this channel (oldest first):");
    for (uint32_t i = 0; i < 64; i++) {
        uint32_t k = (ch->ring_next + i) % 64;
        if (!ch->ring[k].va && !ch->ring[k].msub) continue;
        hos_log("gpu:   @%#llx sub %u method %#05x = %#010x", (unsigned long long)ch->ring[k].va, ch->ring[k].msub >> 16,
                ch->ring[k].msub & 0xFFFF, ch->ring[k].arg);
    }
}

/* ---- GPU memory ------------------------------------------------------------------------------------------------- */
static void dump_ring(void);
static void unmapped(uint64_t va, const char *what) {
    if (InterlockedIncrement64(&st_unmapped) <= 16) {
        hos_log("gpu: %s at unmapped GPU address %#llx", what, (unsigned long long)va);
        if (st_unmapped == 1) dump_ring();
    }
}
static int gpu_read(uint64_t va, void *dst, uint64_t n) {
    uint8_t *d = (uint8_t *)dst;
    while (n) {
        uint64_t avail;
        uint8_t *p = hos_gpu_ptr(va, &avail);
        if (!p) { unmapped(va, "read"); memset(d, 0, n); return 0; }
        uint64_t k = n < avail ? n : avail;
        memcpy(d, p, k);
        d += k; va += k; n -= k;
    }
    return 1;
}
/* HOS_GPU_PBCHECK=1: report GPU writes that land on recently executed command memory (a wrong address) */
static int pbcheck = -1;
#define PB_RANGES 8192
static struct { uint8_t *lo, *hi; } pb_recent[PB_RANGES];
static unsigned pb_next;
static const char *write_what = "?";
static uint32_t write_method;
static const uint8_t *rx_methods;   /* the 3D methods the renderer takes (rx_3d_methods), when it runs */

static int gpu_write(uint64_t va, const void *src, uint64_t n) {
    const uint8_t *s = (const uint8_t *)src;
    if (rx_methods) rx_memory_written(va, n);   /* the renderer's copies of this memory are stale */
    while (n) {
        uint64_t avail;
        uint8_t *p = hos_gpu_ptr(va, &avail);
        if (!p) { unmapped(va, "write"); return 0; }
        uint64_t k = n < avail ? n : avail;
        if (pbcheck > 0)
            for (unsigned i = 0; i < (pb_next < PB_RANGES ? pb_next : PB_RANGES); i++)
                if (p < pb_recent[i].hi && p + k > pb_recent[i].lo) {
                    static volatile LONG reported;
                    if (InterlockedIncrement(&reported) <= 20)
                        hos_log("gpu: %s (method %#x) writes %llu bytes at GPU %#llx, inside command memory",
                                write_what, write_method, (unsigned long long)k, (unsigned long long)va);
                    break;
                }
        memcpy(p, s, k);
        s += k; va += k; n -= k;
    }
    return 1;
}
static uint32_t gpu_r32(uint64_t va) { uint32_t v; gpu_read(va, &v, 4); return v; }
static void gpu_w32(uint64_t va, uint32_t v) { gpu_write(va, &v, 4); }
static void gpu_w64(uint64_t va, uint64_t v) { gpu_write(va, &v, 8); }

/* GPU timestamps, in the GPU timer's units: ns * 384 / 625 (as suyu) */
static uint64_t gpu_ticks(void) {
    static double scale = -1;   /* HOS_GPU_TIME_SCALE (experiment): the GPU clock's rate against the host's */
    static int64_t t0;
    if (scale < 0) {
        const char *e = getenv("HOS_GPU_TIME_SCALE");
        scale = e ? atof(e) : 1.0;
        t0 = hos_now_ns();
    }
    const int64_t now = hos_now_ns();
    const int64_t t = scale == 1.0 ? now : t0 + (int64_t)((now - t0) * scale);
    return (uint64_t)t / 625 * 384;
}

/* a semaphore release: 4 bytes of payload, or 16 bytes {u64 payload, u64 timestamp} */
static void release(uint64_t va, uint64_t payload, int long_form) {
    InterlockedIncrement64(&st_sem);
    write_what = "semaphore release";
    if (long_form) {
        gpu_w64(va + 8, gpu_ticks());
        gpu_w64(va, payload);
    } else {
        gpu_w32(va, (uint32_t)payload);
    }
}

/* ---- block-linear layout ------------------------------------------------------------------------------------------
   A GOB is 64 bytes x 8 rows (512 bytes), swizzled inside; a block is 1 GOB wide, 2^bh GOBs tall and 2^bd deep;
   blocks run left to right, then top to bottom, then front to back. Coordinates are in bytes (x) and rows (y). */
static uint64_t bl_offset(uint32_t x, uint32_t y, uint32_t z, uint32_t width_bytes, uint32_t height, uint32_t bh,
                          uint32_t bd) {
    uint32_t gobs_x = (width_bytes + 63) >> 6;
    uint32_t blocks_y = (height + (8u << bh) - 1) >> (3 + bh);
    uint64_t block = 512ull << (bh + bd);
    uint64_t gob = ((x & 63) >> 5) << 8 | ((y & 7) >> 1) << 6 | ((x & 31) >> 4) << 5 | (y & 1) << 4 | (x & 15);
    return (uint64_t)(z >> bd) * gobs_x * blocks_y * block + (uint64_t)(y >> (3 + bh)) * gobs_x * block
         + (uint64_t)(x >> 6) * block + ((uint64_t)(z & ((1u << bd) - 1)) << (9 + bh))
         + ((uint64_t)((y >> 3) & ((1u << bh) - 1)) << 9) + gob;
}

typedef struct { uint32_t bh, bd, width, height, depth, layer, ox, oy; } Surf;
static Surf surf(const uint32_t *p) {
    return (Surf){(p[0] >> 4) & 15, (p[0] >> 8) & 15, p[1], p[2], p[3], p[4], p[5] & 0xFFFF, p[5] >> 16};
}
static uint64_t bl_size(uint32_t width_bytes, uint32_t height, uint32_t depth, uint32_t bh, uint32_t bd) {
    uint64_t gobs_x = (width_bytes + 63) >> 6, blocks_y = (height + (8u << bh) - 1) >> (3 + bh);
    uint64_t blocks_z = (depth + (1u << bd) - 1) >> bd;
    return gobs_x * blocks_y * (blocks_z ? blocks_z : 1) * (512ull << (bh + bd));
}
/* one row of a block-linear surface, to or from buf: 16-byte runs are contiguous inside a GOB */
static void bl_row(int write, uint64_t base, const Surf *s, uint32_t bpp, uint32_t row, uint32_t bytes, uint8_t *buf) {
    uint32_t wb = s->width * bpp, x0 = s->ox * bpp, y = s->oy + row;
    uint32_t depth = s->depth > s->layer ? s->depth : s->layer + 1;
    uint64_t avail;
    uint8_t *p = hos_gpu_ptr(base, &avail);
    if (p && avail < bl_size(wb, s->height, depth, s->bh, s->bd)) p = NULL;
    for (uint32_t x = 0; x < bytes;) {
        uint32_t bx = x0 + x, k = 16 - (bx & 15);
        if (k > bytes - x) k = bytes - x;
        uint64_t off = bl_offset(bx, y, s->layer, wb, s->height, s->bh, s->bd);
        if (p) { if (write) memcpy(p + off, buf + x, k); else memcpy(buf + x, p + off, k); }
        else if (write) gpu_write(base + off, buf + x, k);
        else gpu_read(base + off, buf + x, k);
        x += k;
    }
}
/* ---- inline-to-memory (3D, compute and A140 share the method layout at 0x60) -------------------------------------
   0x60 line_length_in  0x61 line_count  0x62/0x63 dst address  0x64 pitch  0x65 block dims (h 4-7, d 8-11)
   0x66 width  0x67 height  0x68 depth  0x69 layer (z)  0x6A x  0x6B y  0x6C launch (bit 0: pitch)  0x6D data */
static int sem_trace = -1;   /* HOS_SEM_TRACE=1: log semaphore operations and inline uploads */
static void upload_finish(Channel *ch, int sub) {
    Upload *u = &ch->up[sub];
    write_what = "inline upload";
    const uint32_t *r = ch->reg[sub];
    uint64_t dst = (uint64_t)r[0x62] << 32 | r[0x63];
    uint32_t len = r[0x60], lines = r[0x61] ? r[0x61] : 1;
    InterlockedIncrement64(&st_i2m);
    if (sem_trace < 0) sem_trace = getenv("HOS_SEM_TRACE") != NULL;
    if (sem_trace) {
        uint64_t head = 0;
        memcpy(&head, u->buf, len * lines < 8 ? len * lines : 8);
        hos_log("gpu: inline upload %#llx %u x %u %s first %#llx", (unsigned long long)dst, len, lines,
                u->linear ? "linear" : "block linear", (unsigned long long)head);
    }
    if (u->linear) {
        if (lines == 1) gpu_write(dst, u->buf, len);
        else for (uint32_t l = 0; l < lines; l++) gpu_write(dst + (uint64_t)l * r[0x64], u->buf + (uint64_t)l * len, len);
    } else {
        Surf bs = {(r[0x65] >> 4) & 15, (r[0x65] >> 8) & 15, r[0x66], r[0x67], r[0x68], r[0x69], r[0x6A], r[0x6B]};
        for (uint32_t l = 0; l < lines; l++) bl_row(1, dst, &bs, 1, l, len, u->buf + (uint64_t)l * len);
        if (rx_methods)   /* (bl_row writes memory directly) */
            rx_memory_written(dst, bl_size(bs.width, bs.height, bs.depth > bs.layer ? bs.depth : bs.layer + 1, bs.bh, bs.bd));
    }
    u->pos = u->size = 0;
}
static void upload_launch(Channel *ch, int sub, uint32_t arg) {
    Upload *u = &ch->up[sub];
    uint32_t lines = ch->reg[sub][0x61] ? ch->reg[sub][0x61] : 1;
    u->size = ch->reg[sub][0x60] * lines;
    u->pos = 0;
    u->linear = arg & 1;
    u->buf = (uint8_t *)realloc(u->buf, u->size ? u->size : 1);
    if (!u->size) upload_finish(ch, sub);
}
static void upload_data(Channel *ch, int sub, uint32_t w) {
    Upload *u = &ch->up[sub];
    if (u->pos >= u->size) return;
    uint32_t k = u->size - u->pos < 4 ? u->size - u->pos : 4;
    memcpy(u->buf + u->pos, &w, k);
    u->pos += k;
    if (u->pos == u->size) upload_finish(ch, sub);
}

/* ---- copy engine (B0B5) ------------------------------------------------------------------------------------------
   0x90/0x91 semaphore address  0x92 payload  0xC0 launch  0x100/0x101 src  0x102/0x103 dst  0x104 pitch in
   0x105 pitch out  0x106 line length  0x107 line count  0x1C0/0x1C1 const A/B  0x1C2 remap components
   0x1C3.. dst {block size, width, height, depth, layer, origin}  0x1CA.. src {same} */
static void dma_launch(Channel *ch, int sub, uint32_t launch) {
    write_what = "copy engine";
    const uint32_t *r = ch->reg[sub];
    uint64_t src = (uint64_t)r[0x100] << 32 | r[0x101], dst = (uint64_t)r[0x102] << 32 | r[0x103];
    int multi = (launch >> 9) & 1, remap = (launch >> 10) & 1;
    uint32_t len = r[0x106], lines = multi ? r[0x107] : 1;
    int src_pitch = !multi || ((launch >> 7) & 1), dst_pitch = !multi || ((launch >> 8) & 1);   /* 1D: bytes */
    uint32_t rc = r[0x1C2];
    uint32_t comp = remap ? ((rc >> 16) & 3) + 1 : 1;
    uint32_t nsrc = remap ? ((rc >> 20) & 3) + 1 : 1, ndst = remap ? ((rc >> 24) & 3) + 1 : 1;
    uint32_t sbpp = comp * nsrc, dbpp = comp * ndst;   /* bytes per element; len counts elements */
    Surf ss = surf(r + 0x1CA), ds = surf(r + 0x1C3);
    InterlockedIncrement64(&st_dma);
    {   /* HOS_DMA_WATCH=<GPU VA hex> (debugging): copies whose destination starts within 16 MB below it are logged */
        static uint64_t watch = 1;
        if (watch == 1) watch = getenv("HOS_DMA_WATCH") ? strtoull(getenv("HOS_DMA_WATCH"), NULL, 16) : 0;
        static int logged;
        static int want_w = -1;   /* HOS_DMA_LOG_W=<width>: copies into block-linear surfaces that wide (elements) */
        if (want_w < 0) want_w = getenv("HOS_DMA_LOG_W") ? atoi(getenv("HOS_DMA_LOG_W")) : 0;
        if (((watch && dst <= watch && watch - dst < (16u << 20)) || (want_w && (want_w == 1 ? !(((launch >> 8) & 1) && (launch >> 9) & 1) && ((launch >> 9) & 1) : ds.width == (uint32_t)want_w))) &&
            logged++ < 400)
            hos_log("gpu: copy %#llx -> %#llx launch %#x len %u lines %u pitch %u/%u src %ux%u bh%u layer %u dst %ux%u bh%u "
                    "layer %u remap %#x rendered-src %d", (unsigned long long)src, (unsigned long long)dst, launch, len,
                    lines, r[0x104], r[0x105], ss.width, ss.height, ss.bh, ss.layer, ds.width, ds.height, ds.bh,
                    ds.layer, remap ? rc : 0, rx_methods ? rx_gpu_writes_pending(src, (uint64_t)len * lines) : -1);
    }

    /* identity remap (or none) moves bytes; otherwise each destination component comes from its swizzle:
       a source component (0-3), constant A (4) or B (5), or nothing (6) */
    int identity = 1, need_src = !remap;
    for (uint32_t i = 0; remap && i < ndst; i++) {
        uint32_t sw = (rc >> (4 * i)) & 7;
        if (sw != i) identity = 0;
        if (sw < 4) need_src = 1;
    }
    if (identity) need_src = 1;
    uint32_t sbytes = len * sbpp, dbytes = len * dbpp;
    uint8_t *line = (uint8_t *)malloc((size_t)(sbytes > dbytes ? sbytes : dbytes) + 16);
    uint8_t *out = (uint8_t *)malloc((size_t)dbytes + 16);
    for (uint32_t l = 0; l < lines; l++) {
        if (need_src) {
            if (src_pitch) gpu_read(src + (uint64_t)l * r[0x104], line, sbytes);
            else bl_row(0, src, &ss, sbpp, l, sbytes, line);
        }
        uint8_t *data = line;
        if (remap && !identity) {
            if (!dst_pitch || ndst == 0) memset(out, 0, dbytes);
            for (uint32_t e = 0; e < len; e++)
                for (uint32_t i = 0; i < ndst; i++) {
                    uint32_t sw = (rc >> (4 * i)) & 7;
                    uint8_t *o = out + (size_t)e * dbpp + (size_t)i * comp;
                    if (sw < 4) memcpy(o, line + (size_t)e * sbpp + (size_t)sw * comp, comp);
                    else if (sw == 4) memcpy(o, &r[0x1C0], comp);
                    else if (sw == 5) memcpy(o, &r[0x1C1], comp);
                    else memset(o, 0, comp);   /* NO_WRITE: approximated as 0 */
                }
            data = out;
        }
        if (dst_pitch) gpu_write(dst + (uint64_t)l * r[0x105], data, dbytes);
        else bl_row(1, dst, &ds, dbpp, l, dbytes, data);
    }
    if (!dst_pitch && rx_methods)   /* (bl_row writes memory directly) */
        rx_memory_written(dst, bl_size(ds.width * dbpp, ds.height, ds.depth > ds.layer ? ds.depth : ds.layer + 1, ds.bh, ds.bd));
    free(line);
    free(out);
    uint32_t sem = (launch >> 3) & 3;
    if (sem) release((uint64_t)r[0x90] << 32 | r[0x91], r[0x92], sem == 2);
}

/* ---- 3D (B197) ---------------------------------------------------------------------------------------------------- */
static void m3d_method(Channel *ch, int sub, uint32_t method, uint32_t arg);
static void split_emit_method(int sub, uint32_t method, uint32_t arg);
static void split_emit(uint32_t kind, uint32_t sub);

/* The 3D methods with an effect besides setting their register (the renderer's, and m3d_method's own cases): split
   mode's two sides each keep a table (rebuilt if the renderer's table appears later). */
static void special_3d(uint8_t table[0x1000], const uint8_t **built_from) {
    if (*built_from == rx_methods && table[0x585]) return;
    *built_from = rx_methods;
    for (uint32_t m = 0; m < 0x1000; m++) table[m] = rx_methods ? rx_methods[m] : 0;
    static const uint32_t own[] = {0x45, 0x46, 0x47, 0x48, 0x49, 0x6C, 0x6D, 0xB2, 0x6C3, 0x8C4, 0x674, 0x585, 0x5F9,
                                   0x5FA, 0x5FB};
    for (size_t k = 0; k < sizeof own / sizeof *own; k++) table[own[k]] = 1;
    for (uint32_t m = 0x8E4; m < 0x8F4; m++) table[m] = 1;
}

/* The MME: macro code uploaded through 0x45/0x46, entry points through 0x47/0x48, called by writing method
   0xE00 + 2n (then its parameters to 0xE01 + 2n). */
static void mme_run(Channel *ch, int sub, uint32_t entry, const uint32_t *params, uint32_t nparams) {
    uint32_t R[8] = {0}, pc = ch->mme_entry[entry % MME_ENTRIES], next_param = 1, method_addr = 0, steps = 0;
    int carry = 0, delayed = -1, exiting = 0;
    R[1] = nparams ? params[0] : 0;
    InterlockedIncrement64(&st_macros);
#define FETCH() (next_param < nparams ? params[next_param++] : 0)
#define SETREG(d, v) do { if (d) R[d] = (v); } while (0)
#define SEND(v) do { if (gpu_trace && traced < 200000) \
                         hos_log("gpu:   macro %u sends method %#05x = %#010x", entry, method_addr & 0xFFF, (uint32_t)(v)); \
                     m3d_method(ch, sub, method_addr & 0xFFF, (v)); \
                     method_addr = (method_addr & ~0xFFFu) | ((method_addr + ((method_addr >> 12) & 63)) & 0xFFF); } while (0)
    for (;;) {
        if (++steps > 1000000) { hos_log("gpu: macro %u runs away", entry); break; }
        uint32_t op = ch->mme_code[pc % MME_CODE], here = pc;
        pc++;
        int in_delay = delayed >= 0;
        if (in_delay) { pc = (uint32_t)delayed; delayed = -1; }
        uint32_t kind = op & 7, rop = (op >> 4) & 7, dst = (op >> 8) & 7, a = R[(op >> 11) & 7], b = R[(op >> 14) & 7];
        int32_t imm = (int32_t)op >> 14;
        uint32_t bf_src = (op >> 17) & 31, bf_mask = (1u << ((op >> 22) & 31)) - 1, bf_dst = (op >> 27) & 31;
        uint32_t result = 0;
        int has_result = 1;
        switch (kind) {
        case 0: {   /* ALU */
            uint64_t t;
            switch ((op >> 17) & 31) {
            case 0: t = (uint64_t)a + b; carry = t >> 32; result = (uint32_t)t; break;
            case 1: t = (uint64_t)a + b + carry; carry = t >> 32; result = (uint32_t)t; break;
            case 2: t = (uint64_t)a - b; carry = t < 0x100000000ull; result = (uint32_t)t; break;
            case 3: t = (uint64_t)a - b - (carry ? 0 : 1); carry = t < 0x100000000ull; result = (uint32_t)t; break;
            case 8: result = a ^ b; break;
            case 9: result = a | b; break;
            case 10: result = a & b; break;
            case 11: result = a & ~b; break;
            case 12: result = ~(a & b); break;
            default: result = 0; break;
            }
            break;
        }
        case 1: result = a + (uint32_t)imm; break;                                       /* add immediate */
        case 2: result = (a & ~(bf_mask << bf_dst)) | (((b >> bf_src) & bf_mask) << bf_dst); break;   /* insert */
        case 3: result = ((b >> a) & bf_mask) << bf_dst; break;                             /* extract << imm */
        case 4: result = ((b >> bf_src) & bf_mask) << a; break;                             /* extract << reg */
        case 5: result = ch->reg[sub][(a + (uint32_t)imm) & (NREGS - 1)]; break;           /* read */
        case 7: {                                                                          /* branch */
            has_result = 0;
            int taken = ((op >> 4) & 1) ? a != 0 : a == 0;
            if (taken) {
                uint32_t target = here + (uint32_t)imm;
                if ((op >> 5) & 1) { pc = target; continue; }   /* annulled: no delay slot */
                delayed = (int)target;
                continue;   /* the delay slot runs next, then the jump */
            }
            break;
        }
        default: has_result = 0; break;
        }
        if (has_result) {
            switch (rop) {
            case 0: SETREG(dst, FETCH()); break;
            case 1: SETREG(dst, result); break;
            case 2: SETREG(dst, result); method_addr = result; break;
            case 3: SETREG(dst, FETCH()); SEND(result); break;
            case 4: SETREG(dst, result); SEND(result); break;
            case 5: SETREG(dst, FETCH()); method_addr = result; break;
            case 6: SETREG(dst, result); method_addr = result; SEND(FETCH()); break;
            case 7: SETREG(dst, result); method_addr = result; SEND((result >> 12) & 63); break;
            }
        }
        if (exiting) break;               /* that was the exit's delay slot */
        if ((op >> 7) & 1 && !in_delay) exiting = 1;
    }
#undef FETCH
#undef SETREG
#undef SEND
}

/* HOS_SHADER_DUMP=<dir>: every shader program a draw uses is written once to <dir>/<stage>_<gpu va>.bin (64 KB
   from its start: the 0x50-byte program header, then code), for the shader translator's tests. Program region
   0x582/0x583, pipelines at 0x800 (16 words each: enable/type, offset). */
static void dump_shaders(Channel *ch, int sub) {
    static const char *dir;
    static int init;
    if (!init) { init = 1; dir = getenv("HOS_SHADER_DUMP"); if (dir) CreateDirectoryA(dir, NULL); }
    if (!dir) return;
    static uint64_t seen[8192];
    static int nseen;
    const uint32_t *r = ch->reg[sub];
    uint64_t base = (uint64_t)r[0x582] << 32 | r[0x583];
    for (int st = 0; st < 6; st++) {
        uint32_t cfg = r[0x800 + st * 0x10];
        if (st != 1 && !(cfg & 1)) continue;   /* VertexB is always on */
        uint64_t va = base + r[0x800 + st * 0x10 + 1];
        int known = 0;
        for (int i = 0; i < nseen; i++) if (seen[i] == va) { known = 1; break; }
        if (known || nseen >= 8192) continue;
        seen[nseen++] = va;
        static uint8_t buf[0x10000];
        gpu_read(va, buf, sizeof buf);
        char path[512];
        snprintf(path, sizeof path, "%s/%d_%010llx.bin", dir, (cfg >> 4) & 15, (unsigned long long)va);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(buf, 1, sizeof buf, f); fclose(f); }
    }
}

static void report_semaphore(Channel *ch, int sub) {
    const uint32_t *r = ch->reg[sub];
    uint64_t va = (uint64_t)r[0x6C0] << 32 | r[0x6C1];
    uint32_t q = r[0x6C3], op = q & 3, report = (q >> 23) & 31, short_q = (q >> 28) & 1;
    if (sem_trace < 0) sem_trace = getenv("HOS_SEM_TRACE") != NULL;
    if (sem_trace) hos_log("gpu: report semaphore %#llx query %#x (op %u report %u short %u reduction %u/%u) payload %#x",
                           (unsigned long long)va, q, op, report, short_q, (q >> 3) & 1, (q >> 9) & 7, r[0x6C2]);
    if (op == 0 || op == 2) {   /* release, report only */
        /* payload; ZPASS_PIXEL_COUNT (2, 21): the renderer's estimate; other counters read 0 for now */
        uint64_t v = report == 0 ? r[0x6C2] : (report == 2 || report == 21) && ch->rx ? rx_channel_zpass(ch->rx) : 0;
        if (sem_trace && report) hos_log("gpu:   counter report %u = %llu", report, (unsigned long long)v);
        release(va, v, !short_q);
    }
    /* acquire (1) and trap (3) are not used by NVN's fences */
}

/* CB_DATA writes reach the renderer as one range (rx_memory_written per word was a lock and page bookkeeping each):
   reported before the renderer sees any other method, and at the end of each command. */
static void cb_flush(Channel *ch) {
    if (ch->cb_hi > ch->cb_lo && rx_methods) rx_memory_written(ch->cb_lo, ch->cb_hi - ch->cb_lo);
    ch->cb_lo = ch->cb_hi = 0;
}

static void m3d_method(Channel *ch, int sub, uint32_t method, uint32_t arg) {
    uint32_t *r = ch->reg[sub];
    if (method >= 0xE00) {   /* macro call: the even method starts it with its first parameter */
        if (!(method & 1) || !ch->macro_method) {
            ch->macro_method = method & ~1u;
            ch->nparams = 0;
        }
        if (ch->nparams == ch->capparams) {
            ch->capparams = ch->capparams ? ch->capparams * 2 : 64;
            ch->macro_params = (uint32_t *)realloc(ch->macro_params, ch->capparams * sizeof(uint32_t));
        }
        ch->macro_params[ch->nparams++] = arg;
        return;
    }
    if (ch->front) {   /* split mode's parsing side: the registers (what macros read), then on to the render side */
        const uint32_t orig = arg, smode = r[0x49];
        if (method != 0x49) {
            if (smode == 0 || smode == 1) ch->shadow[method] = arg;
            else if (smode == 3) arg = ch->shadow[method];
        }
        r[method] = arg;
        switch (method) {
        case 0x46: ch->mme_code[r[0x45]++ % MME_CODE] = arg; break;
        case 0x48: ch->mme_entry[r[0x47]++ % MME_ENTRIES] = arg; break;
        case 0x8C4: r[0xD00] = 1; break;
        default: if (method >= 0x8E4 && method < 0x8F4) r[0x8E3] += 4; break;
        }
        split_emit_method(sub, method, orig);   /* (the render side applies the shadow RAM itself, the same way) */
        return;
    }
    /* shadow RAM (0x49): track writes, or replay the tracked value */
    uint32_t mode = r[0x49];
    if (method != 0x49) {
        if (mode == 0 || mode == 1) ch->shadow[method] = arg;
        else if (mode == 3) arg = ch->shadow[method];
    }
    r[method] = arg;
    if (rx_methods && rx_methods[method]) {   /* draws, clears, constant buffer bindings: the renderer */
        cb_flush(ch);
        if (!ch->rx) ch->rx = rx_channel_new();
        rx_3d_method(ch->rx, r, method, arg);
    }
    switch (method) {
    case 0x45: break;                                                          /* MME instruction pointer */
    case 0x46: ch->mme_code[r[0x45]++ % MME_CODE] = arg; break;                 /* MME instruction */
    case 0x47: break;                                                          /* MME entry pointer */
    case 0x48: ch->mme_entry[r[0x47]++ % MME_ENTRIES] = arg; break;            /* MME entry */
    case 0x6C: upload_launch(ch, sub, arg); break;
    case 0x6D: upload_data(ch, sub, arg); break;
    case 0xB2: hos_gpu_syncpt_incr(arg & 0xFFFF); break;                       /* sync_info: sync point increment */
    case 0x6C3: report_semaphore(ch, sub); break;
    case 0x8C4: r[0xD00] = 1; break;                                           /* firmware call 4 */
    case 0x674: InterlockedIncrement64(&st_clears); break;                    /* clear surface */
    case 0x585: InterlockedIncrement64(&st_draws); dump_shaders(ch, sub); break;   /* draw end */
    case 0x5F9: case 0x5FA: case 0x5FB: InterlockedIncrement64(&st_draws); break;   /* inline index draws */
    default:
        if (method >= 0x8E4 && method < 0x8F4) {                              /* CB_DATA: write into the bound cb */
            uint64_t cb = (uint64_t)r[0x8E1] << 32 | r[0x8E2];
            const uint64_t va = cb + r[0x8E3];
            write_what = "CB_DATA";
            write_method = method;
            uint64_t avail;
            uint8_t *p = pbcheck > 0 ? NULL : hos_gpu_ptr(va, &avail);
            if (p && avail >= 4) {
                memcpy(p, &arg, 4);
                if (ch->cb_hi != va) { cb_flush(ch); ch->cb_lo = va; }
                ch->cb_hi = va + 4;
            } else {
                gpu_w32(va, arg);
            }
            r[0x8E3] += 4;
        }
        break;
    }
}

/* ---- dispatch ----------------------------------------------------------------------------------------------------- */
static int defer_release = -1;   /* HOS_DEFER_RELEASE (see the puller semaphore) */
static struct { uint64_t va; uint32_t payload; int long_form; } deferred[4096];
static int n_deferred;
static void flush_deferred(void) {
    for (int i = 0; i < n_deferred; i++) release(deferred[i].va, deferred[i].payload, deferred[i].long_form);
    n_deferred = 0;
}
static void method(Channel *ch, int sub, uint32_t m, uint32_t arg) {
    if (ch->front) {   /* split mode's parsing side: bindings and 3D registers kept, everything goes on */
        if (m < 0x40) {
            ch->puller[m] = arg;
            if (m == 0) ch->cls[sub] = arg & 0xFFFF;
            split_emit_method(sub, m, arg);
        } else if (ch->cls[sub] == CLS_3D) {
            m3d_method(ch, sub, m, arg);
        } else {
            if (m < NREGS) ch->reg[sub][m] = arg;
            split_emit_method(sub, m, arg);
        }
        return;
    }
    write_method = m;
    ch->ring[ch->ring_next % 64].va = ch->cur_va;
    ch->ring[ch->ring_next % 64].msub = m | (uint32_t)sub << 16;
    ch->ring[ch->ring_next % 64].arg = arg;
    ch->ring_next++;
    if (gpu_trace && InterlockedIncrement64(&traced) <= 200000)
        hos_log("gpu: [%p] sub %d cls %04x method %#05x = %#010x", (void *)ch, sub, m < 0x40 ? 0 : ch->cls[sub], m, arg);
    if (m < 0x40) {   /* the puller, shared by all subchannels */
        ch->puller[m] = arg;
        switch (m) {
        case 0x00: {                                                                         /* bind object */
            ch->cls[sub] = arg & 0xFFFF;
            static uint32_t seen_cls[16];
            static int nseen_cls;
            int k;
            for (k = 0; k < nseen_cls && seen_cls[k] != (arg & 0xFFFF); k++) {}
            if (k == nseen_cls && nseen_cls < 16) {
                seen_cls[nseen_cls++] = arg & 0xFFFF;
                hos_log("gpu: subchannel %d bound to class %#06x", sub, arg & 0xFFFF);
            }
            break;
        }
        case 0x07: {                                                                         /* semaphore op */
            uint64_t va = (uint64_t)(ch->puller[4] & 0xFF) << 32 | (ch->puller[5] & ~3u);
            uint32_t op = arg & 0x1F, seq = ch->puller[6];
            if (sem_trace < 0) sem_trace = getenv("HOS_SEM_TRACE") != NULL;
            if (sem_trace) hos_log("gpu: puller semaphore %#llx op %#x (arg %#x) sequence %#x", (unsigned long long)va, op, arg, seq);
            if (op == 2) {
                /* Releases that do not wait for idle (bit 20) land at the end of the GPFIFO entry, after the engine
                   methods around them. TOTK clears its pooled render targets under a render enable condition ("not
                   cleared yet": the CPU sets the word, a release after the clear zeroes it); with the release
                   visible at once, a target reused within one entry (a blur's ping-pong pass, the cloud shadows on
                   a reused 1/8 target) skipped its clears and the Room of Awakening went olive instead of teal; at
                   the end of the entry both that and the clears of the half-resolution depth (cloud dome over the
                   terrain when done) match the game. HOS_DEFER_RELEASE=0: at once; 2: at the end of the
                   submission. */
                if (defer_release < 0) defer_release = getenv("HOS_DEFER_RELEASE") ? atoi(getenv("HOS_DEFER_RELEASE")) : 1;
                if (defer_release && ((arg >> 20) & 1) && n_deferred < 4096) {
                    deferred[n_deferred].va = va;
                    deferred[n_deferred].payload = seq;
                    deferred[n_deferred].long_form = !((arg >> 24) & 1);
                    n_deferred++;
                    break;
                }
                release(va, seq, !((arg >> 24) & 1));   /* release size: bit 24 = 4 bytes */
            } else if (op == 1 || op == 4 || op == 8) {
                flush_deferred();   /* (whoever this waits for may be waiting for them) */
                int64_t until = hos_now_ns() + 2000000000LL;   /* another channel releases it; give up after 2 s */
                for (;;) {
                    uint32_t v = gpu_r32(va);
                    if (op == 1 ? v == seq : op == 4 ? (int32_t)(v - seq) >= 0 : (v & seq) != 0) break;
                    if (hos_now_ns() > until) { hos_log("gpu: semaphore acquire timed out at %#llx", (unsigned long long)va); break; }
                    hos_sleep_ns(50000);
                }
            }
            break;
        }
        case 0x1A: {                                                                         /* acquire == arg */
            uint64_t va = (uint64_t)(ch->puller[4] & 0xFF) << 32 | ch->puller[5];
            int64_t until = hos_now_ns() + 2000000000LL;
            while (gpu_r32(va) != arg && hos_now_ns() < until) hos_sleep_ns(50000);
            break;
        }
        case 0x1B: release((uint64_t)(ch->puller[4] & 0xFF) << 32 | ch->puller[5], arg, 0); break;   /* release */
        case 0x1D: if (arg & 1) hos_gpu_syncpt_incr((arg >> 8) & 0xFFF); break;                     /* syncpt incr */
        default: break;
        }
        return;
    }
    uint32_t cls = ch->cls[sub];
    if (cls == CLS_3D) { m3d_method(ch, sub, m, arg); return; }
    if (m >= NREGS) return;
    uint32_t *r = ch->reg[sub];
    r[m] = arg;
    switch (cls) {
    case CLS_COMPUTE:
        if (m == 0x6C) upload_launch(ch, sub, arg);
        else if (m == 0x6D) upload_data(ch, sub, arg);
        else if (m == 0xAF) {
            InterlockedIncrement64(&st_dispatches);
            if (rx_methods) {   /* the renderer runs the dispatch */
                if (!ch->rx) ch->rx = rx_channel_new();
                rx_compute_launch(ch->rx, r);
            }
        }
        break;
    case CLS_I2M:
        if (m == 0x6C) upload_launch(ch, sub, arg);
        else if (m == 0x6D) upload_data(ch, sub, arg);
        break;
    case CLS_DMA:
        if (m == 0xC0) dma_launch(ch, sub, arg);
        break;
    case CLS_2D:
        if (m == 0x237) InterlockedIncrement64(&st_blits);   /* pixels_from_memory src_y0: the blit trigger */
        break;
    }
}

/* the macro being called when a command's arguments run out: run it */
static void end_of_command(Channel *ch, int sub) {
    cb_flush(ch);
    if (ch->cls[sub] == CLS_3D && ch->macro_method) {
        uint32_t entry = (ch->macro_method - 0xE00) >> 1;
        ch->macro_method = 0;
        mme_run(ch, sub, entry, ch->macro_params, ch->nparams);
        cb_flush(ch);
        if (ch->front) split_emit(1, (uint32_t)sub);                     /* (the render side does the next line) */
        else if (ch->rx) rx_3d_method(ch->rx, ch->reg[sub], 0xFFFF, 0);   /* the macro's deferred instanced draws */
    }
}

/* One command list: method headers {method 0-12, subchannel 13-15, count 16-28, mode 29-31} and their arguments.
   Modes: 1 increasing, 3 non-increasing, 4 inline (the count field is the argument), 5 increase once. A command's
   arguments may run on into the channel's next command list, so the decoder state lives in the channel. */
static void run_pushbuffer(Channel *ch, const uint32_t *w, uint32_t n, uint64_t va) {
    uint32_t i = 0;
    while (i < n) {
        ch->cur_va = va + 4ull * i;
        if (ch->cmd_left && ch->cmd_mode == 3 && ch->cmd_method >= 0x8E4 && ch->cmd_method < 0x8F4 &&
            ch->cls[ch->cmd_sub] == CLS_3D && !gpu_trace && pbcheck <= 0 && !ch->front) {
            /* a run of CB_DATA words (NVN streams constant buffer updates as one non-incrementing command): one
               translation and one copy for the run instead of a method dispatch per word; the registers, the shadow
               RAM and the range the renderer is told about end as the per-word path leaves them */
            uint32_t *r = ch->reg[ch->cmd_sub];
            const uint32_t m = ch->cmd_method, shadow_mode = r[0x49];
            uint32_t k = n - i < ch->cmd_left ? n - i : ch->cmd_left;
            const uint64_t dst = ((uint64_t)r[0x8E1] << 32 | r[0x8E2]) + r[0x8E3];
            uint64_t avail;
            uint8_t *p = shadow_mode == 3 ? NULL : hos_gpu_ptr(dst, &avail);
            if (p && avail >= 4ull * k) {
                memcpy(p, w + i, 4ull * k);
                if (ch->cb_hi != dst) { cb_flush(ch); ch->cb_lo = dst; }
                ch->cb_hi = dst + 4ull * k;
                r[m] = w[i + k - 1];
                if (shadow_mode == 0 || shadow_mode == 1) ch->shadow[m] = w[i + k - 1];
                r[0x8E3] += 4 * k;
                write_method = m;
                ch->ring[ch->ring_next % 64].va = ch->cur_va + 4ull * (k - 1);
                ch->ring[ch->ring_next % 64].msub = m | (uint32_t)ch->cmd_sub << 16;
                ch->ring[ch->ring_next % 64].arg = w[i + k - 1];
                ch->ring_next++;
                i += k;
                ch->cmd_first = 0;
                if ((ch->cmd_left -= k) == 0) end_of_command(ch, (int)ch->cmd_sub);
                continue;
            }
        }
        if (ch->cmd_left) {
            method(ch, (int)ch->cmd_sub, ch->cmd_method, w[i++]);
            if (ch->cmd_mode == 1 || (ch->cmd_mode == 5 && ch->cmd_first)) ch->cmd_method++;
            ch->cmd_first = 0;
            if (--ch->cmd_left == 0) end_of_command(ch, (int)ch->cmd_sub);
            continue;
        }
        uint32_t h = w[i++], m = h & 0x1FFF, sub = (h >> 13) & 7, count = (h >> 16) & 0x1FFF, mode = h >> 29;
        if (gpu_trace && traced < 200000) hos_log("gpu: header %#010x at word %u of %u", h, i - 1, n);
        switch (mode) {
        case 4:
            method(ch, (int)sub, m, count);
            end_of_command(ch, (int)sub);
            break;
        case 1: case 3: case 5:
            ch->cmd_method = m;
            ch->cmd_sub = sub;
            ch->cmd_mode = mode;
            ch->cmd_left = count;
            ch->cmd_first = 1;
            if (!count) end_of_command(ch, (int)sub);
            break;
        default:   /* 0: the old formats / a zero word (nop); 2, 6, 7: not produced by NVN */
            if (h) {
                static volatile LONG bad;
                if (InterlockedIncrement(&bad) <= 3) {
                    hos_log("gpu: invalid pushbuffer header %#x (mode %u) at GPU %#llx", h, mode, (unsigned long long)ch->cur_va);
                    dump_ring();
                }
            }
            break;
        }
    }
    if (ch->front) split_emit(2, 0);   /* the render side's end of the entry (deferred releases) */
    else if (defer_release == 1) flush_deferred();
}

static Channel *channel(int fd) {
    if (fd < 0 || fd >= 256) return NULL;
    Channel *ch = channels[fd];
    if (ch) return ch;
    AcquireSRWLockExclusive(&channels_lock);
    if (!channels[fd]) {
        ch = (Channel *)calloc(1, sizeof *ch);
        InitializeSRWLock(&ch->lock);
        channels[fd] = ch;
    }
    ReleaseSRWLockExclusive(&channels_lock);
    return channels[fd];
}

int64_t hos_gpu_stat_draws(void) { return st_draws; }

void hos_gpu_submit(int channel_fd, const uint64_t *entries, uint32_t n);
static void submit(int channel_fd, const uint64_t *entries, uint32_t n, uint32_t *written, int64_t nwritten);

/* ---- the GPU thread (default; HOS_GPU_SYNC=1: in the submitting thread): submissions and presents run there in
   order; the submitting threads go on
   (their fences pass when the work is done: nvdrv.c) ------------------------------------------------------------ */
typedef struct Job {
    struct Job *next;
    int kind;              /* 0 submission, 1 present */
    int fd, incr_id;
    uint32_t n;
    uint64_t *entries;
    uint32_t *written;     /* a submission: the pages the CPU wrote before it (rx_collect_written) */
    int64_t nwritten;
    uint64_t address;
    uint32_t width, height;
    int64_t at_vsync;      /* a present: not before this vsync (0: now) */
} Job;
static Job *job_head, *job_tail;
static SRWLOCK job_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE job_cv = CONDITION_VARIABLE_INIT;
static int async_mode = -1;

int hos_gpu_async(void) {
    if (async_mode < 0) async_mode = getenv("HOS_GPU_SYNC") == NULL;   /* default: the GPU thread */
    return async_mode;
}

static int split_on(void);
static void front_submit(int fd, const uint64_t *entries, uint32_t n, uint32_t *written, int64_t nwritten, int incr_id);
static void front_present(uint64_t address, uint32_t width, uint32_t height, int64_t at_vsync);

static DWORD WINAPI gpu_thread(void *p) {
    (void)p;
    void hos_host_thread_register_self(const char *name);
    hos_host_thread_register_self("GPU commands");
    hos_thread_boost(1);
    for (;;) {
        AcquireSRWLockExclusive(&job_lock);
        while (!job_head) SleepConditionVariableSRW(&job_cv, &job_lock, INFINITE, 0);
        Job *j = job_head;
        job_head = j->next;
        if (!job_head) job_tail = NULL;
        ReleaseSRWLockExclusive(&job_lock);
        if (split_on()) {
            if (j->kind == 0) front_submit(j->fd, j->entries, j->n, j->written, j->nwritten, j->incr_id);
            else front_present(j->address, j->width, j->height, j->at_vsync);
        } else if (j->kind == 0) {
            submit(j->fd, j->entries, j->n, j->written, j->nwritten);
            if (j->incr_id >= 0) { hos_gpu_syncpt_incr((uint32_t)j->incr_id); hos_gpu_syncpt_incr((uint32_t)j->incr_id); }
        } else {
            if (j->at_vsync) { void hos_vsync_wait_until(int64_t); hos_vsync_wait_until(j->at_vsync); }
            rx_present(j->address, j->width, j->height);
        }
        free(j->entries);
        free(j);
    }
    return 0;
}

static void push_job(Job *j) {
    static HANDLE thread;
    AcquireSRWLockExclusive(&job_lock);
    if (!thread) thread = CreateThread(NULL, 1 << 20, gpu_thread, NULL, 0, NULL);
    if (job_tail) job_tail->next = j;
    else job_head = j;
    job_tail = j;
    ReleaseSRWLockExclusive(&job_lock);
    WakeConditionVariable(&job_cv);
}

void hos_gpu_queue_submit(int fd, const uint64_t *entries, uint32_t n, int incr_id) {
    Job *j = (Job *)calloc(1, sizeof *j);
    j->fd = fd;
    j->incr_id = incr_id;
    j->n = n;
    j->entries = (uint64_t *)malloc((size_t)(n ? n : 1) * 8);
    memcpy(j->entries, entries, (size_t)n * 8);
    /* The pages the CPU wrote are found here, on the submitting thread (GetWriteWatch over the guest arena was
       ~12% of the GPU thread), in submission order: the scan and the queueing are one step. Split into a parsing
       and a render thread, the parsing thread has room and scans as it starts the job (front_submit): the scan was
       ~18% of the game's presentation thread (3 ms a frame over 1.8 GB in the open world), on its way to queue the
       frame. (Pages written between the submit and the scan are seen early: harmless, the reset is still before
       the render thread reads them.) */
    static SRWLOCK scan_lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&scan_lock);
    static int scan_on_gpu = -1;   /* HOS_SCAN_ON_GPU=1 (debugging): the GPU thread scans when it starts the job */
    if (scan_on_gpu < 0) scan_on_gpu = getenv("HOS_SCAN_ON_GPU") != NULL;
    static int scan_here = -1;   /* HOS_SCAN_ON_SUBMIT=1: here when split, as before */
    if (scan_here < 0) scan_here = !split_on() || getenv("HOS_SCAN_ON_SUBMIT") != NULL;
    j->written = rx_methods && !scan_on_gpu && scan_here ? rx_collect_written(&j->nwritten) : NULL;
    if (!j->written) j->nwritten = -1;
    /* HOS_EARLY_FENCE=1 (experiment): the submission's fence passes as it is queued, not once the render thread has
       run it (the game then waits for nothing; what it reads back from the GPU may be a frame old) */
    static int early = -1;
    if (early < 0) early = getenv("HOS_EARLY_FENCE") != NULL;
    const int early_id = early ? j->incr_id : -1;
    if (early) j->incr_id = -1;
    push_job(j);
    ReleaseSRWLockExclusive(&scan_lock);
    if (early_id >= 0) { hos_gpu_syncpt_incr((uint32_t)early_id); hos_gpu_syncpt_incr((uint32_t)early_id); }
}

/* A window buffer is shown: after the work submitted before it (the GPU thread), or now. */
void hos_gpu_present(uint64_t address, uint32_t width, uint32_t height, int64_t at_vsync) {
    if (!hos_gpu_async()) {
        rx_present(address, width, height);
        return;
    }
    Job *j = (Job *)calloc(1, sizeof *j);
    j->kind = 1;
    j->address = address;
    j->width = width;
    j->height = height;
    j->at_vsync = at_vsync;
    push_job(j);
}

static void stats(void) {
    int64_t now = hos_now_ns();
    if (now - st_last < 5000000000LL) return;
    st_last = now;
    hos_log("gpu: %lld submissions, %lld words: %lld draws, %lld clears, %lld dispatches, %lld blits, %lld copies, "
            "%lld uploads, %lld macros, %lld semaphores", (long long)st_submits, (long long)st_words,
            (long long)st_draws, (long long)st_clears, (long long)st_dispatches, (long long)st_blits, (long long)st_dma,
            (long long)st_i2m, (long long)st_macros, (long long)st_sem);
}

/* HOS_GPU_WATCH=1 (diagnostic): after a command list runs, its pages are made read-only, and the first CPU writes
   into a list that already ran are reported with the writer's guest call stack (then the page is writable again). */
static int gpu_watch = -1;
static struct { uint8_t *lo, *hi; } watched[4096];
static volatile LONG nwatched, watch_reports;
static uint8_t *pool_lo, *pool_hi;   /* HOS_GPU_WATCHPOOL=<GPU VA>: 1 MB of command memory watched from the start */
static volatile LONG pool_reports;
static LONG CALLBACK watch_handler(EXCEPTION_POINTERS *e) {
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    uint8_t *a = (uint8_t *)e->ExceptionRecord->ExceptionInformation[1];
    if (a >= pool_lo && a < pool_hi) {
        LONG n = InterlockedIncrement(&pool_reports);
        if (n <= 4000) {   /* the first ones with the writer's call stack */
            HosThread *t = hos_self();
            Ctx *gc = t ? hos_ctx() : NULL;
            hos_log("gpu: CPU write at pool +%#llx by thread %s (x0 %#llx x1 %#llx x2 %#llx lr %#llx)%s",
                    (unsigned long long)(a - pool_lo), t ? hos_thread_name(t) : "?",
                    gc ? (unsigned long long)gc->x[0] : 0ull, gc ? (unsigned long long)gc->x[1] : 0ull,
                    gc ? (unsigned long long)gc->x[2] : 0ull, gc ? (unsigned long long)gc->x[30] : 0ull, n <= 20 ? ":" : "");
            uint64_t hos_guest_size(void);
            if (gc && gc->x[1] > 0x10000 && gc->x[1] < hos_guest_size()) {   /* (a copy's source: its first bytes) */
                const uint32_t *q = (const uint32_t *)(hos_guest_base() + gc->x[1]);
                char imgs[512];
                void rx_debug_images_at(uint64_t lo, uint64_t len, char *out, size_t cap);
                imgs[0] = 0;
                if (n <= 60 && rx_methods) rx_debug_images_at(gc->x[1] & ~0xFFFull, 0x1000, imgs, sizeof imgs);
                hos_log("gpu:   source %08x %08x %08x %08x %08x %08x %08x %08x;%s", q[0], q[1], q[2], q[3], q[4], q[5], q[6],
                        q[7], imgs);
            }
            if (n <= 20) hos_backtrace();
        }
        DWORD old;
        VirtualProtect((void *)((uintptr_t)a & ~(uintptr_t)4095), 4096, PAGE_READWRITE, &old);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    int hit = -1, n = nwatched < 4096 ? nwatched : 4096;
    for (int i = 0; i < n; i++)
        if (a >= watched[i].lo - 4096 && a < watched[i].hi + 4096) { hit = i; break; }
    if (hit < 0) return EXCEPTION_CONTINUE_SEARCH;
    int inside = a >= watched[hit].lo && a < watched[hit].hi && watched[hit].hi - watched[hit].lo >= 64;
    static uint8_t *reported_lists[64];
    for (int i = 0; inside && i < 64; i++)
        if (reported_lists[i] == watched[hit].lo) inside = 0;
    if (inside && InterlockedIncrement(&watch_reports) <= 12) {
        reported_lists[(watch_reports - 1) % 64] = watched[hit].lo;
        HosThread *t = hos_self();
        hos_log("gpu: CPU write at %p, inside command memory already run (list %p..%p), by thread %s:", (void *)a,
                (void *)watched[hit].lo, (void *)watched[hit].hi, t ? hos_thread_name(t) : "?");
        hos_backtrace();
    }
    DWORD old;
    VirtualProtect((void *)((uintptr_t)a & ~(uintptr_t)4095), 4096, PAGE_READWRITE, &old);
    return EXCEPTION_CONTINUE_EXECUTION;
}
static void watch_pool(void) {   /* (re)arm the pool watch: called at each submission */
    static int init, handler;
    static uint64_t pool_va;
    if (!init) {
        const char *v = getenv("HOS_GPU_WATCHPOOL");
        init = 1;
        if (v) pool_va = strtoull(v, NULL, 0);
        /* HOS_GPU_WATCHGUEST=<guest offset>[:<bytes>]: the same for guest memory (GPU addresses change between runs) */
        if ((v = getenv("HOS_GPU_WATCHGUEST")) != NULL) {
            char *end;
            uint64_t off = strtoull(v, &end, 0), len = *end == ':' ? strtoull(end + 1, NULL, 0) : 65536;
            pool_lo = hos_guest_base() + off;
            pool_hi = pool_lo + len;
            if (!handler++) AddVectoredExceptionHandler(1, watch_handler);
            hos_log("gpu: watching guest memory at %p..%p", (void *)pool_lo, (void *)pool_hi);
        }
    }
    if (pool_va) {   /* resolved again each time: the address can be mapped to other memory later */
        uint64_t avail;
        uint8_t *lo = hos_gpu_ptr(pool_va, &avail);
        if (lo && lo != pool_lo) {
            if (pool_lo) {   /* (the old range writable again) */
                DWORD old;
                VirtualProtect((void *)((uintptr_t)pool_lo & ~(uintptr_t)4095), (size_t)(pool_hi - pool_lo), PAGE_READWRITE, &old);
            }
            pool_lo = lo;
            pool_hi = pool_lo + (avail < (1 << 20) ? avail : (1 << 20));
            pool_reports = 0;
            if (!handler++) AddVectoredExceptionHandler(1, watch_handler);
            hos_log("gpu: watching memory at %p..%p (GPU %#llx)", (void *)pool_lo, (void *)pool_hi, (unsigned long long)pool_va);
        }
    }
    static int64_t arm_at = -1;   /* HOS_GPU_WATCH_AFTER=<s>: armed from then on */
    if (arm_at < 0) arm_at = hos_now_ns() + (int64_t)(atof(getenv("HOS_GPU_WATCH_AFTER") ? getenv("HOS_GPU_WATCH_AFTER") : "0") * 1e9);
    if (pool_lo && pool_reports < 4000 && hos_now_ns() >= arm_at) {
        DWORD old;
        VirtualProtect((void *)((uintptr_t)pool_lo & ~(uintptr_t)4095), (size_t)(pool_hi - pool_lo), PAGE_READONLY, &old);
    }
}
static void watch_list(uint8_t *p, uint32_t words) {
    if (gpu_watch < 0) {
        gpu_watch = getenv("HOS_GPU_WATCH") != NULL;
        if (gpu_watch) AddVectoredExceptionHandler(1, watch_handler);
    }
    if (!gpu_watch || nwatched >= 4096) return;
    LONG i = InterlockedIncrement(&nwatched) - 1;
    watched[i].lo = p;
    watched[i].hi = p + (uint64_t)words * 4;
    DWORD old;
    uintptr_t lo = (uintptr_t)p & ~(uintptr_t)4095, hi = ((uintptr_t)p + (uint64_t)words * 4 + 4095) & ~(uintptr_t)4095;
    VirtualProtect((void *)lo, hi - lo, PAGE_READONLY, &old);
}

/* GPFIFO entries: {GPU VA of a command list: bits 0-39; non-main: bit 41; length in words: bits 42-62}.
   written / nwritten: the pages the CPU wrote before this submission (rx_collect_written at the submit), or -1:
   scan for them now. */
static void submit(int channel_fd, const uint64_t *entries, uint32_t n, uint32_t *written, int64_t nwritten) {
    if (gpu_trace < 0) gpu_trace = getenv("HOS_GPU_TRACE") != NULL;
    if (gpu_stats < 0) gpu_stats = getenv("HOS_GPU_STATS") != NULL || gpu_trace;
    {   /* HOS_GPU_TRACE_AFTER=<s>: the trace starts then */
        static int64_t from = -1;
        if (from < 0) from = getenv("HOS_GPU_TRACE_AFTER") ? hos_now_ns() + (int64_t)(atof(getenv("HOS_GPU_TRACE_AFTER")) * 1e9) : 0;
        if (from) gpu_trace = getenv("HOS_GPU_TRACE") != NULL && hos_now_ns() >= from;
    }
    static int patchcheck = -1;
    if (patchcheck < 0) patchcheck = getenv("HOS_GPU_PATCHCHECK") != NULL;
    Channel *ch = channel(channel_fd);
    if (!ch) { free(written); return; }
    // Constant-buffer bindings are persistent channel state and NVN installs some before the window/device
    // exists. Track them from the first submission; rx_3d_method defers GPU work until rx_start succeeds.
    if (!rx_methods) rx_methods = rx_3d_methods();
    if (rx_methods) rx_begin_submission_written(written, nwritten);
    else free(written);
    AcquireSRWLockExclusive(&ch->lock);
    anomaly_ch = ch;
    watch_pool();
    if (patchcheck) {   /* experiment: does the CPU change command memory after submitting it? */
        static struct { uint64_t va; uint32_t words; uint32_t *copy; } snap[512];
        static int nsnap;
        for (int k = 0; k < nsnap; k++) {
            uint64_t avail;
            uint32_t *now = (uint32_t *)hos_gpu_ptr(snap[k].va, &avail);
            if (!now) continue;
            uint32_t changed = 0, first = 0;
            for (uint32_t j = 0; j < snap[k].words; j++)
                if (now[j] != snap[k].copy[j]) { if (!changed) first = j; changed++; }
            if (changed)
                hos_log("gpu: %u of %u words at %#llx changed after submit (word %u: %#x -> %#x)", changed, snap[k].words,
                        (unsigned long long)snap[k].va, first, snap[k].copy[first], now[first]);
            free(snap[k].copy);
        }
        nsnap = 0;
        for (uint32_t e = 0; e < n && nsnap < 512; e++) {
            uint64_t va = entries[e] & 0xFFFFFFFFFCULL, avail;
            uint32_t words = (uint32_t)((entries[e] >> 42) & 0x1FFFFF);
            uint8_t *p = hos_gpu_ptr(va, &avail);
            if (!p || avail < (uint64_t)words * 4 || !words) continue;
            snap[nsnap].va = va;
            snap[nsnap].words = words;
            snap[nsnap].copy = (uint32_t *)malloc((size_t)words * 4);
            memcpy(snap[nsnap].copy, p, (size_t)words * 4);
            nsnap++;
        }
    }
    InterlockedIncrement64(&st_submits);
    for (uint32_t e = 0; e < n; e++) {
        uint64_t va = entries[e] & 0xFFFFFFFFFCULL;
        uint32_t words = (uint32_t)((entries[e] >> 42) & 0x1FFFFF);
        if (gpu_trace && traced < 200000)
            hos_log("gpu: entry %u/%u raw %#018llx: %u words at %#llx", e, n, (unsigned long long)entries[e], words,
                    (unsigned long long)va);
        if (!words) continue;
        InterlockedAdd64(&st_words, words);
        {   /* HOS_GPU_TRACE_SPILL: entries that continue a command's arguments (indirect draw parameters) */
            static int spill = -1, spilled;
            if (spill < 0) spill = getenv("HOS_GPU_TRACE_SPILL") != NULL;
            if (spill && ch->cmd_left && spilled++ < 200) {
                uint32_t w0[6] = {0};
                gpu_read(va, w0, sizeof w0);
                hos_log("gpu: entry %#llx (%u words) continues method %#x (%u left, mode %u): %u %u %u %u %u %u; "
                        "shader-written %d", (unsigned long long)va, words, ch->cmd_method, ch->cmd_left, ch->cmd_mode,
                        w0[0], w0[1], w0[2], w0[3], w0[4], w0[5], rx_methods ? rx_gpu_writes_pending(va, (uint64_t)words * 4) : -1);
            }
        }
        /* commands a shader wrote (indirect draw arguments a compute dispatch produced): wait for the results */
        if (rx_methods && rx_gpu_writes_pending(va, (uint64_t)words * 4)) {
            static int syncs_logged;
            if (syncs_logged++ < 10) hos_log("gpu: commands at %#llx (%u words) were written by a shader: syncing",
                                             (unsigned long long)va, words);
            static int no_sync = -1;
            if (no_sync < 0) no_sync = getenv("RX_NO_WRITE_SYNC") != NULL;
            if (!no_sync) rx_sync_gpu_writes(va, (uint64_t)words * 4);
        }
        uint64_t avail;
        uint8_t *p = hos_gpu_ptr(va, &avail);
        ch->entry_va = va;
        ch->entry_words = words;
        if (pbcheck < 0) pbcheck = getenv("HOS_GPU_PBCHECK") != NULL;
        if (pbcheck > 0 && p) {   /* every command range run so far (deduplicated by start) */
            unsigned k, cnt = pb_next < PB_RANGES ? pb_next : PB_RANGES;
            for (k = 0; k < cnt && pb_recent[k].lo != p; k++) {}
            if (k == cnt) {
                pb_recent[pb_next % PB_RANGES].lo = p;
                pb_recent[pb_next % PB_RANGES].hi = p + (uint64_t)words * 4;
                pb_next++;
            } else if (p + (uint64_t)words * 4 > pb_recent[k].hi) {
                pb_recent[k].hi = p + (uint64_t)words * 4;
            }
        }
        if (p && avail >= (uint64_t)words * 4) {
            run_pushbuffer(ch, (const uint32_t *)p, words, va);
            watch_list(p, words);
        } else {   /* spans mappings (or is unmapped): gather it */
            uint32_t *tmp = (uint32_t *)malloc((size_t)words * 4);
            gpu_read(va, tmp, (uint64_t)words * 4);
            run_pushbuffer(ch, tmp, words, va);
            free(tmp);
        }
    }
    cb_flush(ch);
    if (defer_release == 2) flush_deferred();
    ReleaseSRWLockExclusive(&ch->lock);
    if (rx_methods) rx_flush();
    if (gpu_stats) stats();
}

void hos_gpu_submit(int channel_fd, const uint64_t *entries, uint32_t n) { submit(channel_fd, entries, n, NULL, -1); }

/* ---- split mode (the GPU thread is two; HOS_GPU_SPLIT=0: one, as before) -------------------------------------------
   "GPU commands" parses the command lists and runs the macros, with its own copy of each channel's bindings and 3D
   registers (what macros read), and hands every method that comes out, in order, to "GPU render", which runs them as
   the single thread did (registers, shadow RAM, the renderer, memory writes, semaphores, uploads, copies) and ends
   each submission (its sync point increments: the work is done when the render side is done). Parsing and macros
   were ~20% of the single thread's time, which bounded the frame rate. A command list that a shader wrote (indirect
   arguments) waits for the render side to catch up first. Records: {method | sub << 13, argument}; markers {kind << 16
   | sub, 0} followed by their payload. */
typedef struct { uint32_t a, b; } Rec;
static uint32_t split_ring_n = 1u << 21;
#define SPLIT_RING split_ring_n
enum { SPLIT_MACRO_END = 1, SPLIT_ENTRY_END = 2, SPLIT_SUBMIT = 3, SPLIT_SUBMIT_END = 4, SPLIT_PRESENT = 5 };
static Rec *split_ring;
static int64_t split_head_local, split_safe;          /* parsing side: records written; the end of the last group */
static volatile int64_t split_head, split_tail;       /* published to the render side; consumed by it */
static SRWLOCK split_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE split_more = CONDITION_VARIABLE_INIT, split_room = CONDITION_VARIABLE_INIT;
static Channel *front_channels[256];
static int split_mode = -1;

static void split_publish(void) {
    AcquireSRWLockExclusive(&split_lock);
    split_head = split_safe;
    ReleaseSRWLockExclusive(&split_lock);
    WakeConditionVariable(&split_more);
}
static void split_put(uint32_t a, uint32_t b) {
    if (split_head_local - split_tail >= SPLIT_RING) {   /* full: let the render side catch up */
        split_publish();
        AcquireSRWLockExclusive(&split_lock);
        while (split_head_local - split_tail >= SPLIT_RING) SleepConditionVariableSRW(&split_room, &split_lock, INFINITE, 0);
        ReleaseSRWLockExclusive(&split_lock);
    }
    split_ring[split_head_local & (SPLIT_RING - 1)] = (Rec){a, b};
    split_head_local++;
}
static void split_emit_method(int sub, uint32_t method, uint32_t arg) {
    split_put(method | (uint32_t)sub << 13, arg);
    split_safe = split_head_local;
    if ((split_safe & 4095) == 0) split_publish();
}
static void split_emit(uint32_t kind, uint32_t sub) {
    split_put(kind << 16 | sub, 0);
    split_safe = split_head_local;
}
static void split_put64(uint64_t v) { split_put((uint32_t)v, (uint32_t)(v >> 32)); }
static uint64_t rec64(Rec r) { return (uint64_t)r.a | (uint64_t)r.b << 32; }

/* Wait until the render side has run everything handed to it. */
static void split_drain(void) {
    split_publish();
    AcquireSRWLockExclusive(&split_lock);
    while (split_tail != split_safe) SleepConditionVariableSRW(&split_room, &split_lock, INFINITE, 0);
    ReleaseSRWLockExclusive(&split_lock);
}

/* HOS_FRAME_HIST=1: the render thread's busy time per frame (wall time between presents less its waits for
   records), as a histogram every 600 frames: which frames miss a vsync, and by how much */
static int64_t rt_idle_ns;
static void frame_hist(void) {
    static int on = -1;
    if (on < 0) on = getenv("HOS_FRAME_HIST") != NULL;
    static int64_t last, last_idle, hist[8], frames, max_busy, total_busy;
    const int64_t now = hos_now_ns();
    if (last && !on) {
        void rx_frame_spike(int64_t busy_ns, int64_t threshold_ns);
        if (rx_methods) rx_frame_spike((now - last) - (rt_idle_ns - last_idle), 45000000);
    } else if (last) {
        const int64_t busy = (now - last) - (rt_idle_ns - last_idle);
        static const int64_t edge[7] = {10, 12, 14, 16, 18, 22, 30};
        int b = 0;
        while (b < 7 && busy >= edge[b] * 1000000LL) b++;
        hist[b]++;
        if (busy > max_busy) max_busy = busy;
        void rx_frame_spike(int64_t busy_ns, int64_t threshold_ns);
        if (rx_methods) rx_frame_spike(busy, 18000000);
        total_busy += busy;
        if (++frames == 600) {
            hos_log("gpu: render busy per frame (ms) <10 %lld, 10-12 %lld, 12-14 %lld, 14-16 %lld, 16-18 %lld, 18-22 %lld, "
                    "22-30 %lld, 30+ %lld; mean %.2f, max %.1f", (long long)hist[0], (long long)hist[1], (long long)hist[2],
                    (long long)hist[3], (long long)hist[4], (long long)hist[5], (long long)hist[6], (long long)hist[7],
                    total_busy / 1e6 / frames, max_busy / 1e6);
            memset(hist, 0, sizeof hist);
            frames = max_busy = total_busy = 0;
        }
    }
    last = now;
    last_idle = rt_idle_ns;
}

static DWORD WINAPI render_thread(void *p) {
    (void)p;
    void hos_host_thread_register_self(const char *name);
    hos_host_thread_register_self("GPU render");
    hos_thread_boost(1);
#ifdef __EMSCRIPTEN__
    rx_start(NULL);   /* (web: the device is made here, on the thread that makes every WebGPU call) */
#endif
    Channel *ch = NULL;
    int64_t t = 0;
    for (;;) {
        AcquireSRWLockExclusive(&split_lock);
        if (split_head == t) {
            const int64_t w0 = hos_now_ns();
#ifdef __EMSCRIPTEN__
            /* (the web: no blocking wait here; the thread yields to its event loop, where WebGPU's callbacks run,
               and serves syncs other threads ask for) */
            void rx_web_idle(volatile uint32_t *wake, uint32_t seen);
            while (split_head == t) {
                const uint32_t seen = split_more.seq;   /* (read under the lock: a wake after it changes it) */
                ReleaseSRWLockExclusive(&split_lock);
                rx_web_idle(&split_more.seq, seen);
                AcquireSRWLockExclusive(&split_lock);
            }
#else
            while (split_head == t) SleepConditionVariableSRW(&split_more, &split_lock, INFINITE, 0);
#endif
            rt_idle_ns += hos_now_ns() - w0;
        }
        const int64_t h = split_head;
        ReleaseSRWLockExclusive(&split_lock);
        static uint8_t special[0x1000];
        static const uint8_t *special_from;
        special_3d(special, &special_from);
#define NEXT() split_ring[t++ & (SPLIT_RING - 1)]
        while (t < h) {
            const Rec r = NEXT();
            const uint32_t kind = r.a >> 16;
            if (kind == 0) {
                const uint32_t m = r.a & 0x1FFF, sub = (r.a >> 13) & 7;
                if (m >= 0x40 && m < 0xE00 && ch->cls[sub] == CLS_3D && !special[m]) {   /* a plain 3D register (most
                                                                                         records; below 0x40: puller) */
                    uint32_t *reg = ch->reg[sub], arg = r.b;
                    const uint32_t mode = reg[0x49];
                    if (mode == 0 || mode == 1) ch->shadow[m] = arg;
                    else if (mode == 3) arg = ch->shadow[m];
                    reg[m] = arg;
                    if (ch->cb_hi > ch->cb_lo) cb_flush(ch);
                    continue;
                }
                /* (the single thread told the renderer about CB_DATA writes at the end of each command) */
                if (ch->cb_hi > ch->cb_lo && !(ch->cls[sub] == CLS_3D && m >= 0x8E4 && m < 0x8F4)) cb_flush(ch);
                method(ch, (int)sub, m, r.b);
                continue;
            }
            switch (kind) {
            case SPLIT_MACRO_END:
                cb_flush(ch);
                if (ch->rx) rx_3d_method(ch->rx, ch->reg[r.a & 7], 0xFFFF, 0);   /* the macro's deferred instanced draws */
                break;
            case SPLIT_ENTRY_END:
                cb_flush(ch);
                if (defer_release == 1) flush_deferred();
                break;
            case SPLIT_SUBMIT: {
                const int fd = (int)NEXT().a;
                uint32_t *written = (uint32_t *)(uintptr_t)rec64(NEXT());
                const int64_t nwritten = (int64_t)rec64(NEXT());
                ch = channel(fd);
                if (rx_methods) rx_begin_submission_written(written, nwritten);
                else free(written);
                AcquireSRWLockExclusive(&ch->lock);
                anomaly_ch = ch;
                watch_pool();
                InterlockedIncrement64(&st_submits);
                break;
            }
            case SPLIT_SUBMIT_END: {
                const int incr_id = (int)NEXT().a;
                cb_flush(ch);
                if (defer_release == 2) flush_deferred();
                ReleaseSRWLockExclusive(&ch->lock);
                if (rx_methods) rx_flush();
                if (gpu_stats > 0) stats();
                if (incr_id >= 0) { hos_gpu_syncpt_incr((uint32_t)incr_id); hos_gpu_syncpt_incr((uint32_t)incr_id); }
                break;
            }
            case SPLIT_PRESENT: {
                const uint64_t address = rec64(NEXT());
                const Rec wh = NEXT();
                const int64_t at_vsync = (int64_t)rec64(NEXT());
                if (at_vsync) {
                    void hos_vsync_wait_until(int64_t);
                    const int64_t w0 = hos_now_ns();
                    hos_vsync_wait_until(at_vsync);
                    rt_idle_ns += hos_now_ns() - w0;
                }
                rx_present(address, wh.a, wh.b);
                frame_hist();
                break;
            }
            }
        }
#undef NEXT
        AcquireSRWLockExclusive(&split_lock);
        split_tail = t;
        ReleaseSRWLockExclusive(&split_lock);
        WakeAllConditionVariable(&split_room);
    }
    return 0;
}

static int split_on(void) {
    if (split_mode < 0) {
        const char *e = getenv("HOS_GPU_SPLIT");
        split_mode = e ? atoi(e) != 0 : 1;
        if (split_mode) {
            if (getenv("HOS_GPU_RING_MB")) {
                const unsigned mb = (unsigned)atoi(getenv("HOS_GPU_RING_MB"));
                uint32_t n = 1u << 17;
                while (n < (1u << 26) && (uint64_t)n * sizeof *split_ring < (uint64_t)mb << 20) n <<= 1;
                split_ring_n = n;
            }
            split_ring = (Rec *)malloc((size_t)SPLIT_RING * sizeof *split_ring);
#if defined(__EMSCRIPTEN__) && !RX_WEB_MT   /* (the threaded web build: the renderer's WebGPU thread takes it) */
            web_next_thread_takes_canvas("#screen");   /* (the browser shows what the render thread draws) */
#endif
            CreateThread(NULL, 1 << 20, render_thread, NULL, 0, NULL);
            hos_log("gpu: split into a parsing thread and a render thread");
        }
    }
    return split_mode;
}

/* A submission on the parsing side (split mode). */
static void front_submit(int fd, const uint64_t *entries, uint32_t n, uint32_t *written, int64_t nwritten, int incr_id) {
    if (gpu_trace < 0) gpu_trace = getenv("HOS_GPU_TRACE") != NULL;
    if (gpu_stats < 0) gpu_stats = getenv("HOS_GPU_STATS") != NULL || gpu_trace;
    if (fd < 0 || fd >= 256) { free(written); return; }
    Channel *ch = front_channels[fd];
    if (!ch) {
        ch = (Channel *)calloc(1, sizeof *ch);
        ch->front = 1;
        front_channels[fd] = ch;
    }
    if (!rx_methods) rx_methods = rx_3d_methods();
    static int scan_on_render = -1;   /* HOS_SCAN_ON_GPU=1: the render thread scans (buffer_cache_scan) */
    if (scan_on_render < 0) scan_on_render = getenv("HOS_SCAN_ON_GPU") != NULL;
    if (!written && nwritten < 0 && rx_methods && !scan_on_render) written = rx_collect_written(&nwritten);
    split_put(SPLIT_SUBMIT << 16, 0);
    split_put((uint32_t)fd, 0);
    split_put64((uint64_t)(uintptr_t)written);
    split_put64((uint64_t)nwritten);
    split_safe = split_head_local;
    for (uint32_t e = 0; e < n; e++) {
        uint64_t va = entries[e] & 0xFFFFFFFFFCULL;
        uint32_t words = (uint32_t)((entries[e] >> 42) & 0x1FFFFF);
        if (!words) continue;
        InterlockedAdd64(&st_words, words);
        /* commands a shader wrote (indirect draw arguments a compute dispatch produced): the render side first (the
           hint reads no lock: taking the renderer's for every list serialized the two threads) */
        if (rx_methods && rx_gpu_written_hint(va, (uint64_t)words * 4)) split_drain();
        if (rx_methods && split_tail == split_safe && rx_gpu_writes_pending(va, (uint64_t)words * 4)) {
            static int syncs_logged;
            if (syncs_logged++ < 10) hos_log("gpu: commands at %#llx (%u words) were written by a shader: syncing",
                                             (unsigned long long)va, words);
            if (!getenv("RX_NO_WRITE_SYNC")) rx_sync_gpu_writes(va, (uint64_t)words * 4);
        }
        uint64_t avail;
        uint8_t *p = hos_gpu_ptr(va, &avail);
        ch->entry_va = va;
        ch->entry_words = words;
        if (p && avail >= (uint64_t)words * 4) {
            run_pushbuffer(ch, (const uint32_t *)p, words, va);
        } else {   /* spans mappings (or is unmapped): gather it */
            uint32_t *tmp = (uint32_t *)malloc((size_t)words * 4);
            gpu_read(va, tmp, (uint64_t)words * 4);
            run_pushbuffer(ch, tmp, words, va);
            free(tmp);
        }
    }
    split_put(SPLIT_SUBMIT_END << 16, 0);
    split_put((uint32_t)incr_id, 0);
    split_safe = split_head_local;
    split_publish();
}

static void front_present(uint64_t address, uint32_t width, uint32_t height, int64_t at_vsync) {
    split_put(SPLIT_PRESENT << 16, 0);
    split_put64(address);
    split_put(width, height);
    split_put64((uint64_t)at_vsync);
    split_safe = split_head_local;
    split_publish();
}
