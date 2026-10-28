/* movie::: the SDK's video player library, modeled on Android's MediaExtractor / MediaCodec. It lives in subsdk0,
   which is not lifted; this is it over movie_mf.c.

   movie::Extractor (the container): SetDataSource(path), track count and formats, SelectTrack, then per sample
   Read(Buffer) / GetSampleTime / Advance. movie::Decoder (the codec): Configure(format), RegisterClientEvents(4
   nn::os events: input buffer available, output buffer available, format changed, error), Start; the game takes an
   input buffer index, has the extractor fill it, queues it (SendInputBufferForDecode); takes decoded output buffer
   indices with their time, copies a frame into its own memory (GetOutputBuffer) and releases the index.

   The game (TOTK) asks for NV12 (output format 4) and copies one frame per frame it renders into a buffer it reads
   as two linear textures (luma, interleaved chroma): the copy is all a frame needs, but it also means the video runs
   at the game's frame rate. So the decoder hands out a frame only when it is due by the video's clock: frames are
   dropped when the game falls behind, and the clock stops while the game takes none (a pause).

   The decoder decodes the file itself (a second reader of it, NV12) rather than the samples it is given: one frame
   per input sample, in the same order. Guest objects: Extractor and Decoder are {vtable, host object}; MediaData is
   {host object}; Buffer is {data, capacity, offset, size}, all ours. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

uint64_t hos_hostcall(const char *name, void (*fn)(Ctx *, void *), void *user);   /* hos_hostcall.c */

/* movie::Status: 0 is success; the rest are our own (the game only tests for non-zero) */
enum { ST_OK = 0, ST_FAILED = 0x1000, ST_END_OF_STREAM = 0x3002, ST_TRY_AGAIN = 0x3005 };
enum { FLAG_END_OF_STREAM = 4 };   /* buffer flags (SendInputBufferForDecode, AcquireOutputBufferIndex) */

#define TAG "movie: "
static int verbose = -1;
static int movie_verbose(void) {
    if (verbose < 0) { const char *e = getenv("HOS_MOVIE_LOG"); verbose = e && *e && *e != '0'; }
    return verbose;
}

/* ---- movie::Buffer: plain data, guest layout of our own --------------------------------------------------- */
#define BUF_DATA 0x00   /* u64 guest address */
#define BUF_CAP 0x08    /* u64 */
#define BUF_OFF 0x10    /* i32 */
#define BUF_SIZE 0x14   /* i32 */

static void buffer_set(uint64_t b, uint64_t data, uint64_t cap, int off, int size) {
    G64(b + BUF_DATA) = data;
    G64(b + BUF_CAP) = cap;
    G32(b + BUF_OFF) = (uint32_t)off;
    G32(b + BUF_SIZE) = (uint32_t)size;
}

HLE(_ZN5movie6BufferC1Ev) { buffer_set(ARG(0), 0, 0, 0, 0); RET(ARG(0)); }
HLE(_ZN5movie6BufferD1Ev) { RET(ARG(0)); }
HLE(_ZN5movie6Buffer18SetDataAndCapacityEPvm) { buffer_set(ARG(0), ARG(1), ARG(2), 0, 0); RET(0); }
HLE(_ZN5movie6Buffer8SetRangeEii) { G32(ARG(0) + BUF_OFF) = (uint32_t)ARG(1); G32(ARG(0) + BUF_SIZE) = (uint32_t)ARG(2); RET(0); }
HLE(_ZNK5movie6Buffer4SizeEv) { RET(G32(ARG(0) + BUF_SIZE)); }
HLE(_ZNK5movie6Buffer6OffsetEv) { RET(G32(ARG(0) + BUF_OFF)); }

/* ---- movie::MediaData: a track's or the decoder output's format ---------------------------------------- */
typedef struct {
    char mime[40];
    int width, height;
    int64_t duration_us;
    char path[MAX_PATH];   /* the file (host path): how the decoder finds what to decode */
} Format;

static Format *format_of(uint64_t md) {
    Format *f = (Format *)(uintptr_t)G64(md);
    if (!f) { f = (Format *)calloc(1, sizeof *f); G64(md) = (uint64_t)(uintptr_t)f; }
    return f;
}

/* Strings handed to the game (FindString): interned in guest memory, never freed. */
static uint64_t guest_string(const char *s) {
    static SRWLOCK lk = SRWLOCK_INIT;
    static struct { char s[40]; uint64_t g; } interned[32];
    static int n;
    AcquireSRWLockExclusive(&lk);
    uint64_t g = 0;
    for (int i = 0; i < n && !g; i++) if (!strcmp(interned[i].s, s)) g = interned[i].g;
    if (!g) {
        size_t len = strlen(s) + 1;
        g = hos_sys_alloc(len, 8);
        memcpy(GP(g), s, len);
        HOS_WRITTEN(GP(g), len);
        if (n < 32) { snprintf(interned[n].s, sizeof interned[n].s, "%s", s); interned[n++].g = g; }
    }
    ReleaseSRWLockExclusive(&lk);
    return g;
}

HLE(_ZN5movie9MediaDataC1Ev) { G64(ARG(0)) = 0; format_of(ARG(0)); RET(ARG(0)); }
HLE(_ZN5movie9MediaDataD1Ev) { free((void *)(uintptr_t)G64(ARG(0))); G64(ARG(0)) = 0; RET(ARG(0)); }
HLE(_ZN5movie9MediaData5ClearEv) { memset(format_of(ARG(0)), 0, sizeof(Format)); RET(0); }
/* FindString(name, const char **) / FindInt32(name, int *) / FindRect(name, &left, &top, &right, &bottom) -> found */
HLE(_ZNK5movie9MediaData10FindStringEPKcPS2_) {
    Format *f = format_of(ARG(0));
    const char *key = (const char *)GP(ARG(1));
    if (!strcmp(key, "mime") && f->mime[0]) { G64(ARG(2)) = guest_string(f->mime); RET(1); return; }
    RET(0);
}
HLE(_ZNK5movie9MediaData9FindInt32EPKcPi) {
    Format *f = format_of(ARG(0));
    const char *key = (const char *)GP(ARG(1));
    int v, found = 1;
    if (!strcmp(key, "width")) v = f->width;
    else if (!strcmp(key, "height")) v = f->height;
    else if (!strcmp(key, "stride")) v = f->width;
    else if (!strcmp(key, "slice-height")) v = f->height;
    else found = 0;
    if (found && f->width) G32(ARG(2)) = (uint32_t)v;
    RET(found && f->width);
}
HLE(_ZNK5movie9MediaData8FindRectEPKcPiS3_S3_S3_) {
    Format *f = format_of(ARG(0));
    if (strcmp((const char *)GP(ARG(1)), "crop") || !f->width) { RET(0); return; }
    G32(ARG(2)) = 0;
    G32(ARG(3)) = 0;
    G32(ARG(4)) = (uint32_t)(f->width - 1);
    G32(ARG(5)) = (uint32_t)(f->height - 1);
    RET(1);
}

/* ---- the library ----------------------------------------------------------------------------------------- */
HLE(_ZN5movie12SetAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_) { RET(0); }
HLE(_ZN5movie17GetThreadPriorityEv) { RET(16); }

#define VTABLE_SLOTS 32
typedef struct { int slot; const char *name; void (*fn)(Ctx *, void *); } Method;

static void unknown_method(Ctx *c, void *user) {
    static volatile LONG seen[2][VTABLE_SLOTS];
    uintptr_t u = (uintptr_t)user;
    int cls = (int)(u >> 8), slot = (int)(u & 0xFF);
    if (!InterlockedExchange(&seen[cls][slot], 1))
        hos_log(TAG "%s method %#x (%#llx, %#llx, %#llx) is not implemented", cls ? "Decoder" : "Extractor",
                slot * 8, (unsigned long long)ARG(1), (unsigned long long)ARG(2), (unsigned long long)ARG(3));
    RET(ST_FAILED);
}

static uint64_t make_vtable(int cls, const char *cls_name, const Method *m, int n) {
    uint64_t t = hos_sys_alloc(VTABLE_SLOTS * 8, 16);
    char name[64];
    for (int i = 0; i < VTABLE_SLOTS; i++) {
        snprintf(name, sizeof name, "movie.%s.%#x", cls_name, i * 8);
        G64(t + 8 * i) = hos_hostcall(name, unknown_method, (void *)(uintptr_t)(cls << 8 | i));
    }
    for (int i = 0; i < n; i++) {
        snprintf(name, sizeof name, "movie.%s.%s", cls_name, m[i].name);
        G64(t + 8 * m[i].slot) = hos_hostcall(name, m[i].fn, NULL);
    }
    return t;
}

/* ---- movie::Extractor ------------------------------------------------------------------------------------- */
#define SAMPLE_MAX (8 << 20)

typedef struct {
    SRWLOCK lk;
    MovieReader *r;
    Format format;
    int selected, have;
    uint8_t *sample;           /* the current sample (SAMPLE_MAX) */
    int len;
    int64_t time;
} Extractor;

static Extractor *extractor(Ctx *c) { return (Extractor *)(uintptr_t)G64(ARG(0) + 8); }

static void load_sample(Extractor *e) {   /* the next sample becomes the current one */
    int n = movie_reader_sample(e->r, e->sample, SAMPLE_MAX, &e->time);
    e->have = n > 0;
    e->len = n > 0 ? n : 0;
}

static void ex_destroy(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    if (e) {
        movie_reader_close(e->r);
        free(e->sample);
        free(e);
        G64(ARG(0) + 8) = 0;
    }
    RET(ARG(0));
}
/* SetDataSource(const char *path) */
static void ex_set_data_source(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    const char *guest = (const char *)GP(ARG(1));
    char path[MAX_PATH];
    if (!hos_fs_host_path(guest, path, sizeof path)) {
        hos_log(TAG "%s: not a path on a mount", guest);
        RET(ST_FAILED);
        return;
    }
    AcquireSRWLockExclusive(&e->lk);
    movie_reader_close(e->r);
    e->r = movie_reader_open(path, 0, &e->format.width, &e->format.height, &e->format.duration_us);
    int ok = e->r != NULL;
    if (ok) {
        uint32_t codec = movie_reader_codec(e->r);
        const char *mime = codec == 0x30395056 ? "video/x-vnd.on2.vp9"     /* 'VP90' */
                         : codec == 0x30385056 ? "video/x-vnd.on2.vp8"     /* 'VP80' */
                         : codec == 0x34363248 ? "video/avc"               /* 'H264' */
                                               : "video/unknown";
        snprintf(e->format.mime, sizeof e->format.mime, "%s", mime);
        snprintf(e->format.path, sizeof e->format.path, "%s", path);
        if (!e->sample) e->sample = (uint8_t *)malloc(SAMPLE_MAX);
        hos_log(TAG "playing %s (%s %dx%d, %.2f s)", guest, mime, e->format.width, e->format.height,
                e->format.duration_us / 1e6);
    }
    ReleaseSRWLockExclusive(&e->lk);
    RET(ok ? ST_OK : ST_FAILED);
}
/* GetTrackCount(int *) */
static void ex_track_count(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    G32(ARG(1)) = e->r ? 1 : 0;
    RET(ST_OK);
}
/* GetTrackConfiguration(int track, MediaData *) */
static void ex_track_format(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    if (!e->r || ARG(1) != 0) { RET(ST_FAILED); return; }
    *format_of(ARG(2)) = e->format;
    RET(ST_OK);
}
/* SelectTrack(int track) */
static void ex_select(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    if (!e->r || ARG(1) != 0) { RET(ST_FAILED); return; }
    AcquireSRWLockExclusive(&e->lk);
    if (!e->selected) { e->selected = 1; load_sample(e); }
    ReleaseSRWLockExclusive(&e->lk);
    RET(ST_OK);
}
/* GetTrackIndexForAvailableData(size_t *track): which track the current sample is of; an error at the end */
static void ex_available(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    AcquireSRWLockShared(&e->lk);
    int have = e->selected && e->have;
    ReleaseSRWLockShared(&e->lk);
    if (have) G64(ARG(1)) = 0;
    RET(have ? ST_OK : ST_END_OF_STREAM);
}
/* Read(Buffer *): the current sample into the buffer */
static void ex_read(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    uint64_t b = ARG(1);
    AcquireSRWLockShared(&e->lk);
    int st = ST_END_OF_STREAM;
    if (e->selected && e->have) {
        if ((uint64_t)e->len <= G64(b + BUF_CAP) && G64(b + BUF_DATA)) {
            memcpy(GP(G64(b + BUF_DATA)), e->sample, (size_t)e->len);
            HOS_WRITTEN(GP(G64(b + BUF_DATA)), (size_t)e->len);
            G32(b + BUF_OFF) = 0;
            G32(b + BUF_SIZE) = (uint32_t)e->len;
            st = ST_OK;
        } else {
            hos_log(TAG "a %d byte sample does not fit a %llu byte buffer", e->len, (unsigned long long)G64(b + BUF_CAP));
            st = ST_FAILED;
        }
    }
    ReleaseSRWLockShared(&e->lk);
    RET(st);
}
/* Advance() */
static void ex_advance(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    AcquireSRWLockExclusive(&e->lk);
    if (e->selected && e->have) load_sample(e);
    int have = e->have;
    ReleaseSRWLockExclusive(&e->lk);
    RET(have ? ST_OK : ST_END_OF_STREAM);
}
/* GetSampleTime(int64_t *us) */
static void ex_sample_time(Ctx *c, void *u) {
    (void)u;
    Extractor *e = extractor(c);
    AcquireSRWLockShared(&e->lk);
    int have = e->selected && e->have;
    G64(ARG(1)) = have ? (uint64_t)e->time : 0;
    ReleaseSRWLockShared(&e->lk);
    RET(have ? ST_OK : ST_END_OF_STREAM);
}

static const Method extractor_methods[] = {
    {0, "dtor", ex_destroy},         {1, "delete", ex_destroy},      {2, "SetDataSource", ex_set_data_source},
    {4, "GetTrackCount", ex_track_count}, {5, "GetTrackConfiguration", ex_track_format},
    {6, "SelectTrack", ex_select},   {9, "GetTrackIndexForAvailableData", ex_available},
    {10, "Read", ex_read},           {11, "Advance", ex_advance},    {12, "GetSampleTime", ex_sample_time},
};

/* movie::Extractor::Extractor(ContainerType) */
HLE(_ZN5movie9ExtractorC1ENS_13ContainerTypeE) {
    static uint64_t vt;
    if (!vt) vt = make_vtable(0, "Extractor", extractor_methods, sizeof extractor_methods / sizeof *extractor_methods);
    Extractor *e = (Extractor *)calloc(1, sizeof *e);
    InitializeSRWLock(&e->lk);
    G64(ARG(0)) = vt;
    G64(ARG(0) + 8) = (uint64_t)(uintptr_t)e;
    RET(ARG(0));
}

/* ---- movie::Decoder ------------------------------------------------------------------------------------------ */
#define NIN 4                   /* input buffers */
#define IN_CAP (4 << 20)        /* bytes each */
#define NOUT 6                  /* output buffers (decoded frames) */
#define STALL_NS 150000000LL    /* the game took no frame for this long while it had some: it is paused */

enum { EV_INPUT, EV_OUTPUT, EV_FORMAT, EV_ERROR };
enum { SLOT_FREE, SLOT_DECODING, SLOT_DECODED, SLOT_EXPOSED, SLOT_HELD };

typedef struct {
    SRWLOCK lk;
    CONDITION_VARIABLE cv;
    MovieReader *r;
    Format format;
    size_t frame_size;
    uint64_t ev[4];                       /* the game's events (RegisterClientEvents) */
    uint64_t in_mem;                      /* NIN input buffers of IN_CAP, guest memory */
    int in_free[NIN], n_in_free;          /* indices the game may take (AcquireInputBufferIndex), FIFO */
    struct { int idx, flags; int64_t time; } pending[NIN];
    int n_pending;                        /* queued for decode, FIFO */
    struct { uint8_t *data; int64_t time; int flags, state; } out[NOUT];
    int decoded[NOUT], n_decoded;         /* decoded, waiting to be due, FIFO */
    int exposed[NOUT], n_exposed;         /* the game may take them (AcquireOutputBufferIndex), FIFO */
    int held;                             /* taken by the game, not released */
    int64_t t0;                           /* host ns of the video clock's 0; 0: not running yet */
    int64_t last_progress;                /* host ns the game last released a frame (or the clock started) */
    int64_t last_time, frame_us;          /* the last frame's time and the frame period, for end of stream */
    int configured, started, quit, eos_queued, frames, dropped, consumed;
    HANDLE thread;
} Decoder;

static Decoder *decoder(Ctx *c) { return (Decoder *)(uintptr_t)G64(ARG(0) + 8); }

static void fifo_push(int *q, int *n, int v) { q[(*n)++] = v; }
static int fifo_pop(int *q, int *n) {
    int v = q[0];
    memmove(q, q + 1, (size_t)(--*n) * sizeof *q);
    return v;
}

static int free_slot(Decoder *d) {
    for (int i = 0; i < NOUT; i++) if (d->out[i].state == SLOT_FREE) return i;
    return -1;
}

/* The decode thread: decodes queued inputs into free output slots and hands decoded frames out when due. */
static DWORD WINAPI decode_thread(void *p) {
    Decoder *d = (Decoder *)p;
    AcquireSRWLockExclusive(&d->lk);
    while (!d->quit) {
        int signal_in = 0, signal_out = 0;
        DWORD wait_ms = INFINITE;
        const int64_t now = hos_now_ns();

        /* frames that are due */
        while (d->n_decoded) {
            const int s = d->decoded[0];
            const int eos = d->out[s].flags & FLAG_END_OF_STREAM;
            const int game_has = d->held + d->n_exposed;   /* frames the game has and has not shown */
            if (!d->t0) { d->t0 = now - d->out[s].time * 1000; d->last_progress = now; }
            const int64_t due = d->t0 + d->out[s].time * 1000;
            if (now < due) {
                wait_ms = (DWORD)((due - now + 999999) / 1000000);
                break;
            }
            if (eos ? game_has > 0 : game_has >= 2) {
                if (!d->consumed || now - d->last_progress > STALL_NS) {
                    /* the game is not taking frames (not started yet, paused, loading): the clock stops until it
                       does, with this frame due the moment it does */
                    d->t0 = now - d->out[s].time * 1000;
                    break;   /* a release wakes us */
                }
                if (!eos && d->n_decoded >= 2 && now >= d->t0 + d->out[d->decoded[1]].time * 1000 &&
                    !(d->out[d->decoded[1]].flags & FLAG_END_OF_STREAM)) {
                    /* behind: drop this one, the next is due too */
                    fifo_pop(d->decoded, &d->n_decoded);
                    d->out[s].state = SLOT_FREE;
                    d->dropped++;
                    continue;
                }
                wait_ms = 2;
                break;
            }
            fifo_pop(d->decoded, &d->n_decoded);
            d->out[s].state = SLOT_EXPOSED;
            fifo_push(d->exposed, &d->n_exposed, s);
            signal_out = 1;
        }

        /* decode the next input */
        int slot = d->n_pending ? free_slot(d) : -1;
        if (slot >= 0) {
            const int idx = d->pending[0].idx, flags = d->pending[0].flags;
            const int64_t time = d->pending[0].time;
            memmove(d->pending, d->pending + 1, (size_t)(--d->n_pending) * sizeof d->pending[0]);
            if (!(flags & FLAG_END_OF_STREAM) && !d->eos_queued) {
                /* the game may refill the input buffer while this decodes (after the end of stream it gets none
                   back: it has nothing more to queue, and would only queue the end again) */
                fifo_push(d->in_free, &d->n_in_free, idx);
                signal_in = 1;
                d->out[slot].state = SLOT_DECODING;
                ReleaseSRWLockExclusive(&d->lk);
                int64_t t = 0;
                const int got = movie_reader_frame(d->r, d->out[slot].data, &t);
                AcquireSRWLockExclusive(&d->lk);
                if (got) {
                    if (d->frames && time > d->last_time) d->frame_us = time - d->last_time;
                    d->last_time = time;
                    d->frames++;
                    d->out[slot].time = time;
                    d->out[slot].flags = flags;
                    d->out[slot].state = SLOT_DECODED;
                    fifo_push(d->decoded, &d->n_decoded, slot);
                } else {
                    d->out[slot].state = SLOT_FREE;   /* the file had fewer frames than samples */
                }
            } else if ((flags & FLAG_END_OF_STREAM) && !d->eos_queued) {
                /* the end: an output buffer flagged so, due when the last frame has had its time */
                d->eos_queued = 1;
                d->out[slot].time = d->last_time + (d->frame_us ? d->frame_us : 33333);
                d->out[slot].flags = FLAG_END_OF_STREAM;
                d->out[slot].state = SLOT_DECODED;
                fifo_push(d->decoded, &d->n_decoded, slot);
            }
            wait_ms = 0;
        }

        const uint64_t ev_in = d->ev[EV_INPUT], ev_out = d->ev[EV_OUTPUT];
        if (signal_in || signal_out) {
            ReleaseSRWLockExclusive(&d->lk);
            if (signal_in && ev_in) hos_event_signal(ev_in);
            if (signal_out && ev_out) hos_event_signal(ev_out);
            AcquireSRWLockExclusive(&d->lk);
        }
        if (wait_ms) SleepConditionVariableSRW(&d->cv, &d->lk, wait_ms, 0);
    }
    ReleaseSRWLockExclusive(&d->lk);
    return 0;
}

static void stop_thread(Decoder *d) {
    if (!d->thread) return;
    AcquireSRWLockExclusive(&d->lk);
    d->quit = 1;
    ReleaseSRWLockExclusive(&d->lk);
    WakeAllConditionVariable(&d->cv);
    WaitForSingleObject(d->thread, INFINITE);
    CloseHandle(d->thread);
    d->thread = NULL;
    d->started = 0;
    d->quit = 0;
}

static void de_destroy(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    if (d) {
        stop_thread(d);
        if (d->frames) hos_log(TAG "video done: %d frames decoded, %d dropped", d->frames, d->dropped);
        movie_reader_close(d->r);
        for (int i = 0; i < NOUT; i++) free(d->out[i].data);
        if (d->in_mem) hos_region_free(d->in_mem);
        free(d);
        G64(ARG(0) + 8) = 0;
    }
    RET(ARG(0));
}
/* Configure(MediaData *format) */
static void de_configure(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    const Format *f = format_of(ARG(1));
    if (d->configured) { RET(ST_OK); return; }
    d->format = *f;
    d->r = f->path[0] ? movie_reader_open(f->path, 1, &d->format.width, &d->format.height, NULL) : NULL;
    if (!d->r) { RET(ST_FAILED); return; }
    d->frame_size = (size_t)d->format.width * d->format.height * 3 / 2;
    for (int i = 0; i < NOUT; i++) d->out[i].data = (uint8_t *)malloc(d->frame_size);
    d->in_mem = hos_region_alloc((uint64_t)NIN * IN_CAP, 0x1000);
    d->configured = 1;
    RET(ST_OK);
}
/* RegisterClientEvents(events *): {input available, output available, format changed, error} */
static void de_events(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    for (int i = 0; i < 4; i++) d->ev[i] = G64(ARG(1) + 8 * i);
    RET(ST_OK);
}
/* GetOutputFormat(MediaData *) */
static void de_output_format(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    Format *f = format_of(ARG(1));
    *f = d->format;
    snprintf(f->mime, sizeof f->mime, "%s", "video/raw");
    RET(ST_OK);
}
/* Start() */
static void de_start(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    if (!d->configured) { RET(ST_FAILED); return; }
    if (d->started) { RET(ST_OK); return; }
    AcquireSRWLockExclusive(&d->lk);
    d->n_in_free = 0;
    d->eos_queued = 0;
    for (int i = 0; i < NIN; i++) d->in_free[d->n_in_free++] = i;
    d->started = 1;
    ReleaseSRWLockExclusive(&d->lk);
    d->thread = CreateThread(NULL, 0, decode_thread, d, 0, NULL);
    if (d->ev[EV_FORMAT]) hos_event_signal(d->ev[EV_FORMAT]);
    if (d->ev[EV_INPUT]) hos_event_signal(d->ev[EV_INPUT]);
    RET(ST_OK);
}
/* Stop() */
static void de_stop(Ctx *c, void *u) {
    (void)u;
    stop_thread(decoder(c));
    RET(ST_OK);
}
/* Flush(): queued inputs and frames not handed out are dropped */
static void de_flush(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    AcquireSRWLockExclusive(&d->lk);
    while (d->n_pending) {
        fifo_push(d->in_free, &d->n_in_free, d->pending[0].idx);
        memmove(d->pending, d->pending + 1, (size_t)(--d->n_pending) * sizeof d->pending[0]);
    }
    while (d->n_decoded) d->out[fifo_pop(d->decoded, &d->n_decoded)].state = SLOT_FREE;
    ReleaseSRWLockExclusive(&d->lk);
    WakeAllConditionVariable(&d->cv);
    RET(ST_OK);
}
/* AcquireInputBufferIndex(int *index, int *more): 0 and the index, with how many more there are */
static void de_acquire_input(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    AcquireSRWLockExclusive(&d->lk);
    int st = ST_TRY_AGAIN;
    if (d->n_in_free) {
        G32(ARG(1)) = (uint32_t)fifo_pop(d->in_free, &d->n_in_free);
        G32(ARG(2)) = (uint32_t)d->n_in_free;
        st = ST_OK;
    }
    ReleaseSRWLockExclusive(&d->lk);
    RET(st);
}
/* GetInputBuffer(int index, Buffer *) */
static void de_input_buffer(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    const int idx = (int)ARG(1);
    if (idx < 0 || idx >= NIN || !d->in_mem) { RET(ST_FAILED); return; }
    buffer_set(ARG(2), d->in_mem + (uint64_t)idx * IN_CAP, IN_CAP, 0, 0);
    RET(ST_OK);
}
/* SendInputBufferForDecode(int index, int offset, int size, int64_t time_us, int flags) */
static void de_queue_input(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    const int idx = (int)ARG(1);
    if (idx < 0 || idx >= NIN) { RET(ST_FAILED); return; }
    AcquireSRWLockExclusive(&d->lk);
    if (d->n_pending < NIN) {
        d->pending[d->n_pending].idx = idx;
        d->pending[d->n_pending].time = (int64_t)ARG(4);
        d->pending[d->n_pending].flags = (int)ARG(5);
        d->n_pending++;
    }
    ReleaseSRWLockExclusive(&d->lk);
    WakeAllConditionVariable(&d->cv);
    if (movie_verbose()) hos_log(TAG "input %d: %d bytes at %lld us, flags %d", idx, (int)ARG(3), (long long)ARG(4), (int)ARG(5));
    RET(ST_OK);
}
/* AcquireOutputBufferIndex(int *index, int64_t *time_us, uint32_t *flags, int *more) */
static void de_acquire_output(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    AcquireSRWLockExclusive(&d->lk);
    int st = ST_TRY_AGAIN;
    if (d->n_exposed) {
        const int s = fifo_pop(d->exposed, &d->n_exposed);
        d->out[s].state = SLOT_HELD;
        d->held++;
        G32(ARG(1)) = (uint32_t)s;
        G64(ARG(2)) = (uint64_t)d->out[s].time;
        G32(ARG(3)) = (uint32_t)d->out[s].flags;
        G32(ARG(4)) = (uint32_t)d->n_exposed;
        st = ST_OK;
    }
    ReleaseSRWLockExclusive(&d->lk);
    RET(st);
}
/* GetOutputBuffer(int index, Buffer *): the frame, NV12, into the game's memory */
static void de_output_buffer(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    const int s = (int)ARG(1);
    const uint64_t b = ARG(2);
    if (s < 0 || s >= NOUT || d->out[s].state != SLOT_HELD) { RET(ST_FAILED); return; }
    size_t n = d->frame_size;
    if (d->out[s].flags & FLAG_END_OF_STREAM) n = 0;
    const uint64_t data = G64(b + BUF_DATA), cap = G64(b + BUF_CAP);
    if (n > cap) n = (size_t)cap;
    if (data && n) {
        memcpy(GP(data + G32(b + BUF_OFF)), d->out[s].data, n);
        HOS_WRITTEN(GP(data + G32(b + BUF_OFF)), n);
    }
    G32(b + BUF_SIZE) = (uint32_t)n;
    RET(ST_OK);
}
/* ReleaseOutputBufferIndex(int index) */
static void de_release_output(Ctx *c, void *u) {
    (void)u;
    Decoder *d = decoder(c);
    const int s = (int)ARG(1);
    AcquireSRWLockExclusive(&d->lk);
    if (s >= 0 && s < NOUT && d->out[s].state == SLOT_HELD) {
        d->out[s].state = SLOT_FREE;
        d->held--;
        d->consumed++;
        d->last_progress = hos_now_ns();
    }
    ReleaseSRWLockExclusive(&d->lk);
    WakeAllConditionVariable(&d->cv);
    RET(ST_OK);
}
/* (0x70) asked once after Configure with two int outs: the buffer counts, as far as we can tell */
static void de_buffer_counts(Ctx *c, void *u) {
    (void)u;
    if (ARG(1)) G32(ARG(1)) = NIN;
    if (ARG(2)) G32(ARG(2)) = NOUT;
    RET(ST_OK);
}

static const Method decoder_methods[] = {
    {0, "dtor", de_destroy},             {1, "delete", de_destroy},               {2, "Configure", de_configure},
    {3, "RegisterClientEvents", de_events}, {4, "GetOutputFormat", de_output_format}, {5, "Start", de_start},
    {6, "Stop", de_stop},                {8, "Flush", de_flush},                  {9, "AcquireInputBufferIndex", de_acquire_input},
    {10, "GetInputBuffer", de_input_buffer}, {11, "SendInputBufferForDecode", de_queue_input},
    {12, "AcquireOutputBufferIndex", de_acquire_output}, {13, "GetOutputBuffer", de_output_buffer},
    {14, "BufferCounts", de_buffer_counts}, {15, "ReleaseOutputBufferIndex", de_release_output},
};

/* movie::Decoder::Decoder(DecoderType, DecoderOutputFormat, DecoderMode, int thread priority) */
HLE(_ZN5movie7DecoderC1ENS_11DecoderTypeENS_19DecoderOutputFormatENS_11DecoderModeEi) {
    static uint64_t vt;
    if (!vt) vt = make_vtable(1, "Decoder", decoder_methods, sizeof decoder_methods / sizeof *decoder_methods);
    if (ARG(2) != 4) hos_log(TAG "decoder output format %d asked for; frames are NV12 (4)", (int)ARG(2));
    Decoder *d = (Decoder *)calloc(1, sizeof *d);
    InitializeSRWLock(&d->lk);
    InitializeConditionVariable(&d->cv);
    G64(ARG(0)) = vt;
    G64(ARG(0) + 8) = (uint64_t)(uintptr_t)d;
    RET(ARG(0));
}
