/* The browser build's movie reader (what movie_mf.c does with Media Foundation natively): TOTK's cutscenes are
   VP9 in WebM. The container is read here (a small EBML reader over the file layer: RomFS files come over HTTP in
   blocks, never whole: the movies are ~55 MB and the host heap is ~570 MB); frames are decoded by the browser's
   WebCodecs VideoDecoder (JS, from this thread: JSPI suspends it while the decoder works) into packed NV12, as
   movie_mf.c gives them. decoded = 0: the compressed frames themselves (the movie extractor's samples). */
#include "hos.h"
#include <emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

struct MovieReader {
    HANDLE f;
    uint64_t size, pos;            /* the file, and where the next read starts */
    uint8_t buf[1 << 16];          /* (reads are buffered: EBML headers are a few bytes each) */
    uint64_t buf_at;               /* file offset of buf[0] */
    uint32_t buf_len;
    int decoded, width, height, id;
    uint32_t codec;                /* 'VP90' / 'VP80' as FourCC (Media Foundation's Data1) */
    uint64_t track;                /* the video track's number */
    uint64_t timescale;            /* ns per timecode tick */
    double duration;               /* in ticks */
    uint64_t cluster_time;         /* the current cluster's timecode */
    uint64_t cluster_end;          /* where it ends (or the segment's end) */
    uint64_t segment_end;
    uint8_t *frame;                /* the last compressed frame read */
    uint32_t frame_cap;
    int ended;
};

/* ---- reading ---- */
static int rd(MovieReader *r, uint64_t at, void *dst, uint32_t n) {
    uint8_t *d = (uint8_t *)dst;
    while (n) {
        if (at >= r->buf_at && at < r->buf_at + r->buf_len) {
            uint32_t k = (uint32_t)(r->buf_at + r->buf_len - at);
            if (k > n) k = n;
            memcpy(d, r->buf + (at - r->buf_at), k);
            d += k, at += k, n -= k;
            continue;
        }
        if (n >= sizeof r->buf) {   /* (big: straight in) */
            LARGE_INTEGER li;
            li.QuadPart = (LONGLONG)at;
            SetFilePointerEx(r->f, li, NULL, FILE_BEGIN);
            DWORD got = 0;
            if (!ReadFile(r->f, d, n, &got, NULL) || got != n) return 0;
            return 1;
        }
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG)at;
        SetFilePointerEx(r->f, li, NULL, FILE_BEGIN);
        DWORD got = 0;
        if (!ReadFile(r->f, r->buf, sizeof r->buf, &got, NULL) || got == 0) return 0;
        r->buf_at = at;
        r->buf_len = got;
    }
    return 1;
}

/* an EBML variable-length integer at r->pos: an element id keeps its marker bits, a size does not (all ones:
   unknown size, returned as ~0) */
static int vint(MovieReader *r, uint64_t *out, int is_id) {
    uint8_t b0;
    if (!rd(r, r->pos, &b0, 1) || !b0) return 0;
    int len = 1;
    while (!(b0 & (0x80 >> (len - 1)))) len++;
    uint8_t b[8];
    b[0] = b0;
    if (len > 1 && !rd(r, r->pos + 1, b + 1, (uint32_t)len - 1)) return 0;
    r->pos += (uint64_t)len;
    uint64_t v = is_id ? b0 : (uint64_t)(b0 & (0xFF >> len));
    int all_ones = (v == (uint64_t)(0xFF >> len));
    for (int i = 1; i < len; i++) {
        v = v << 8 | b[i];
        all_ones &= b[i] == 0xFF;
    }
    *out = !is_id && all_ones ? ~0ull : v;
    return 1;
}

static uint64_t read_uint(MovieReader *r, uint64_t size) {
    uint8_t b[8] = {0};
    if (size > 8 || !rd(r, r->pos, b, (uint32_t)size)) return 0;
    uint64_t v = 0;
    for (uint64_t i = 0; i < size; i++) v = v << 8 | b[i];
    return v;
}

static double read_float(MovieReader *r, uint64_t size) {
    uint64_t v = read_uint(r, size);
    if (size == 4) {
        uint32_t u = (uint32_t)v;
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    double d;
    memcpy(&d, &v, 8);
    return d;
}

/* ---- the container ---- */
#define ID_SEGMENT 0x18538067
#define ID_INFO 0x1549A966
#define ID_TIMESCALE 0x2AD7B1
#define ID_DURATION 0x4489
#define ID_TRACKS 0x1654AE6B
#define ID_TRACK 0xAE
#define ID_TRACKNUMBER 0xD7
#define ID_TRACKTYPE 0x83
#define ID_CODEC 0x86
#define ID_VIDEO 0xE0
#define ID_WIDTH 0xB0
#define ID_HEIGHT 0xBA
#define ID_CLUSTER 0x1F43B675
#define ID_TIMECODE 0xE7
#define ID_SIMPLEBLOCK 0xA3
#define ID_BLOCKGROUP 0xA0
#define ID_BLOCK 0xA1

static void read_track(MovieReader *r, uint64_t end) {
    uint64_t number = 0, type = 0;
    char codec[32] = {0};
    int w = 0, h = 0;
    while (r->pos < end) {
        uint64_t id, size;
        if (!vint(r, &id, 1) || !vint(r, &size, 0)) return;
        const uint64_t body = r->pos;
        if (id == ID_TRACKNUMBER) number = read_uint(r, size);
        else if (id == ID_TRACKTYPE) type = read_uint(r, size);
        else if (id == ID_CODEC) rd(r, r->pos, codec, size < sizeof codec - 1 ? (uint32_t)size : sizeof codec - 1);
        else if (id == ID_VIDEO) {
            const uint64_t vend = body + size;
            while (r->pos < vend) {
                uint64_t vid, vsize;
                if (!vint(r, &vid, 1) || !vint(r, &vsize, 0)) return;
                if (vid == ID_WIDTH) w = (int)read_uint(r, vsize);
                else if (vid == ID_HEIGHT) h = (int)read_uint(r, vsize);
                r->pos += vsize;
            }
        }
        r->pos = body + size;
    }
    if (type == 1 && !r->track) {   /* the first video track */
        r->track = number;
        r->width = w, r->height = h;
        r->codec = !strcmp(codec, "V_VP9") ? 0x30395056 : !strcmp(codec, "V_VP8") ? 0x30385056 : 0;   /* 'VP90'/'VP80' */
    }
}

/* Reads the headers up to the first cluster. */
static int read_headers(MovieReader *r) {
    uint64_t id, size;
    r->pos = 0;
    if (!vint(r, &id, 1) || id != 0x1A45DFA3 || !vint(r, &size, 0)) return 0;   /* EBML header */
    r->pos += size;
    if (!vint(r, &id, 1) || id != ID_SEGMENT || !vint(r, &size, 0)) return 0;
    r->segment_end = size == ~0ull ? r->size : r->pos + size;
    r->timescale = 1000000;
    while (r->pos < r->segment_end) {
        const uint64_t at = r->pos;
        if (!vint(r, &id, 1) || !vint(r, &size, 0)) return 0;
        const uint64_t body = r->pos;
        if (id == ID_CLUSTER) {
            r->pos = at;   /* (blocks are read from here) */
            return r->track != 0;
        }
        if (id == ID_INFO || id == ID_TRACKS) {
            const uint64_t end = body + size;
            while (r->pos < end) {
                uint64_t cid, csize;
                if (!vint(r, &cid, 1) || !vint(r, &csize, 0)) return 0;
                const uint64_t cbody = r->pos;
                if (cid == ID_TIMESCALE) r->timescale = read_uint(r, csize);
                else if (cid == ID_DURATION) r->duration = read_float(r, csize);
                else if (cid == ID_TRACK) read_track(r, cbody + csize);
                r->pos = cbody + csize;
            }
        }
        r->pos = body + (size == ~0ull ? 0 : size);
    }
    return 0;
}

/* The next block of the video track: its data in r->frame (returns its length; 0 at the end), its time. */
static uint32_t next_block(MovieReader *r, int64_t *time_us, int *key) {
    for (;;) {
        if (r->pos >= r->segment_end || r->pos >= r->size) return 0;
        uint64_t id, size;
        const uint64_t at = r->pos;
        if (!vint(r, &id, 1) || !vint(r, &size, 0)) return 0;
        const uint64_t body = r->pos;
        if (id == ID_CLUSTER) {   /* (its children are read in turn) */
            r->cluster_end = size == ~0ull ? r->segment_end : body + size;
            continue;
        }
        if (id == ID_TIMECODE) {
            r->cluster_time = read_uint(r, size);
            r->pos = body + size;
            continue;
        }
        if (id == ID_BLOCKGROUP) continue;   /* (its Block is read next) */
        if (id == ID_SIMPLEBLOCK || id == ID_BLOCK) {
            uint64_t track;
            if (!vint(r, &track, 0)) return 0;
            uint8_t h[3];
            if (!rd(r, r->pos, h, 3)) return 0;
            r->pos += 3;
            const int16_t rel = (int16_t)(h[0] << 8 | h[1]);
            const uint64_t end = body + size;
            if (track != r->track || (h[2] & 0x06)) {   /* another track, or laced (not used for video) */
                r->pos = end;
                continue;
            }
            const uint32_t n = (uint32_t)(end - r->pos);
            if (n > r->frame_cap) {
                free(r->frame);
                r->frame_cap = n + (n >> 2);
                r->frame = (uint8_t *)malloc(r->frame_cap);
            }
            if (!rd(r, r->pos, r->frame, n)) return 0;
            r->pos = end;
            const int64_t ticks = (int64_t)r->cluster_time + rel;
            *time_us = (int64_t)((double)ticks * (double)r->timescale / 1000.0);
            *key = id == ID_SIMPLEBLOCK ? (h[2] & 0x80) != 0 : 1;   /* (a Block without references: treated as key) */
            return n;
        }
        (void)at;
        r->pos = body + (size == ~0ull ? 0 : size);   /* Cues, Tags, ... */
    }
}

/* ---- decoding: WebCodecs (one decoder per reader id; frames out as packed NV12) ---- */
EM_ASYNC_JS(int, js_movie_decode, (int id, int vp9, int w, int h, const uint8_t *data, int len, double ts, int key,
                                   uint8_t *dst), {
    globalThis.rxMovies = globalThis.rxMovies || {};
    let m = rxMovies[id];
    if (!m) {
        m = rxMovies[id] = {frames: [], error: null};
        m.dec = new VideoDecoder({output: (f) => m.frames.push(f), error: (e) => { m.error = e; }});
        m.dec.configure({codec: vp9 ? 'vp09.00.10.08' : 'vp8', codedWidth: w, codedHeight: h});
    }
    if (m.error) return -1;
    if (len > 0) {
        m.dec.decode(new EncodedVideoChunk({type: key ? 'key' : 'delta', timestamp: ts,
                                            data: HEAPU8.slice(data >>> 0, (data >>> 0) + len)}));
    } else if (m.dec.state == 'configured') {
        await m.dec.flush();
    }
    // (a frame comes out a little after its chunk; hidden VP9 frames give none)
    for (let i = 0; i < 200 && !m.frames.length && m.dec.decodeQueueSize > 0; i++)
        await new Promise((r) => setTimeout(r, 1));
    if (!m.frames.length) {
        await new Promise((r) => setTimeout(r, 0));
        if (!m.frames.length) return 0;
    }
    const f = m.frames.shift();
    const size = f.allocationSize();
    const tmp = new Uint8Array(size);
    const layout = await f.copyTo(tmp);
    const out = dst >>> 0, fw = w, fh = h;
    const Y = layout[0];
    for (let y = 0; y < fh; y++) HEAPU8.set(tmp.subarray(Y.offset + y * Y.stride, Y.offset + y * Y.stride + fw), out + y * fw);
    const uvOut = out + fw * fh, cw = fw >> 1, ch = fh >> 1;
    if (f.format == 'NV12') {
        const UV = layout[1];
        for (let y = 0; y < ch; y++)
            HEAPU8.set(tmp.subarray(UV.offset + y * UV.stride, UV.offset + y * UV.stride + fw), uvOut + y * fw);
    } else {   // I420 (and I420A): interleave U and V
        const U = layout[1], V = layout[2];
        const row = new Uint8Array(fw);
        for (let y = 0; y < ch; y++) {
            const u = U.offset + y * U.stride, v = V.offset + y * V.stride;
            for (let x = 0; x < cw; x++) {
                row[2 * x] = tmp[u + x];
                row[2 * x + 1] = tmp[v + x];
            }
            HEAPU8.set(row, uvOut + y * fw);
        }
    }
    f.close();
    return 1;
});

EM_JS(void, js_movie_close, (int id), {
    if (globalThis.rxMovies && rxMovies[id]) {
        const m = rxMovies[id];
        for (const f of m.frames) f.close();
        try { m.dec.close(); } catch (e) {}
        delete rxMovies[id];
    }
});

/* ---- the interface (hos.h) ---- */
MovieReader *movie_reader_open(const char *host_path, int decoded, int *width, int *height, int64_t *duration_us) {
    HANDLE f = CreateFileA(host_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        hos_log("movie: %s: cannot open", host_path);
        return NULL;
    }
    MovieReader *r = (MovieReader *)calloc(1, sizeof *r);
    static int next_id = 1;
    r->f = f;
    r->id = __atomic_fetch_add(&next_id, 1, __ATOMIC_RELAXED);
    r->decoded = decoded;
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    r->size = (uint64_t)sz.QuadPart;
    if (!read_headers(r) || !r->codec) {
        hos_log("movie: %s: not a VP8/VP9 WebM this reader knows", host_path);
        movie_reader_close(r);
        return NULL;
    }
    if (width) *width = r->width;
    if (height) *height = r->height;
    if (duration_us) *duration_us = (int64_t)(r->duration * (double)r->timescale / 1000.0);
    hos_log("movie: %s: %dx%d %s, %.1f s (WebCodecs)", host_path, r->width, r->height,
            r->codec == 0x30395056 ? "VP9" : "VP8", r->duration * (double)r->timescale / 1e9);
    return r;
}

int movie_reader_sample(MovieReader *r, void *dst, int cap, int64_t *time_us) {
    int key;
    const uint32_t n = next_block(r, time_us, &key);
    if (!n) return 0;
    if ((int)n > cap) {
        hos_log("movie: a %u byte sample does not fit a %d byte buffer", n, cap);
        return -1;
    }
    memcpy(dst, r->frame, n);
    return (int)n;
}

int movie_reader_frame(MovieReader *r, uint8_t *dst, int64_t *time_us) {
    for (;;) {
        int key = 0;
        int64_t t = 0;
        const uint32_t n = r->ended ? 0 : next_block(r, &t, &key);
        if (!n) r->ended = 1;
        const int got = js_movie_decode(r->id, r->codec == 0x30395056, r->width, r->height, r->frame, (int)n,
                                        (double)t, key, dst);
        if (got < 0) {
            hos_log("movie: the browser's decoder failed");
            return 0;
        }
        if (got) {
            *time_us = t;   /* (frames come out in order: the last chunk's time is close enough) */
            return 1;
        }
        if (r->ended) return 0;
    }
}

void movie_reader_close(MovieReader *r) {
    if (!r) return;
    js_movie_close(r->id);
    if (r->f && r->f != INVALID_HANDLE_VALUE) CloseHandle(r->f);
    free(r->frame);
    free(r);
}

uint32_t movie_reader_codec(MovieReader *r) { return r->codec; }

/* HOS_MOVIE_TEST=<path> (debugging): decodes that movie through this reader at boot, logs the rate and saves frame
   120 as /movietest.nv12 (w*h*3/2 bytes) for a look */
static DWORD WINAPI movie_test_main(void *p) {
    const char *path = (const char *)p;
    int w = 0, h = 0;
    int64_t dur = 0;
    MovieReader *r = movie_reader_open(path, 1, &w, &h, &dur);
    if (!r) return 0;
    uint8_t *frame = (uint8_t *)malloc((size_t)w * h * 3 / 2);
    const int64_t t0 = hos_now_ns();
    int n = 0;
    int64_t t = 0;
    while (n < 300 && movie_reader_frame(r, frame, &t)) {
        if (++n == 120) {
            FILE *f = fopen("/movietest.nv12", "wb");
            if (f) { fwrite(frame, 1, (size_t)w * h * 3 / 2, f); fclose(f); }
        }
    }
    const double s = (hos_now_ns() - t0) / 1e9;
    hos_log("movie test: %d frames of %dx%d in %.2f s (%.0f fps), last at %.2f s", n, w, h, s, n / (s > 0 ? s : 1), t / 1e6);
    movie_reader_close(r);
    free(frame);
    return 0;
}
void web_movie_test(void) {
    const char *p = getenv("HOS_MOVIE_TEST");
    if (p && *p) CreateThread(NULL, 0, movie_test_main, (void *)p, 0, NULL);
}
