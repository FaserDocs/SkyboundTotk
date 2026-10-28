/* Videos through Media Foundation: a source reader over the file gives either its compressed samples (the native
   stream type: what movie::Extractor hands the game) or decoded frames as NV12 (what movie::Decoder outputs).
   Windows has the WebM source and the VP9 decoder (VP9 Video Extensions) built in. */
#define COBJMACROS
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

struct MovieReader {
    IMFSourceReader *reader;
    int decoded, width, height;
    uint32_t codec;
    LONG stride;   /* decoded: luma row pitch when a buffer is not a 2D buffer */
};

#define STREAM ((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM)

static INIT_ONCE started = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK start_mf(PINIT_ONCE once, PVOID p, PVOID *ctx) {
    (void)once; (void)p; (void)ctx;
    HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    if (FAILED(hr)) hos_log("movie: MFStartup failed (%08lx)", (unsigned long)hr);
    return TRUE;
}

static void com(void) { CoInitializeEx(NULL, COINIT_MULTITHREADED); }   /* per thread; an STA thread works too */

MovieReader *movie_reader_open(const char *host_path, int decoded, int *width, int *height, int64_t *duration_us) {
    com();
    InitOnceExecuteOnce(&started, start_mf, NULL, NULL);
    wchar_t wpath[MAX_PATH];
    if (!MultiByteToWideChar(CP_UTF8, 0, host_path, -1, wpath, MAX_PATH)) return NULL;
    IMFAttributes *attrs = NULL;
    MFCreateAttributes(&attrs, 2);
    if (decoded) IMFAttributes_SetUINT32(attrs, &MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    IMFSourceReader *reader = NULL;
    HRESULT hr = MFCreateSourceReaderFromURL(wpath, attrs, &reader);
    IMFAttributes_Release(attrs);
    if (FAILED(hr)) { hos_log("movie: %s: cannot open (%08lx)", host_path, (unsigned long)hr); return NULL; }
    IMFSourceReader_SetStreamSelection(reader, (DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
    IMFSourceReader_SetStreamSelection(reader, STREAM, TRUE);

    MovieReader *r = (MovieReader *)calloc(1, sizeof *r);
    r->reader = reader;
    r->decoded = decoded;
    IMFMediaType *native = NULL;
    if (SUCCEEDED(IMFSourceReader_GetNativeMediaType(reader, STREAM, 0, &native))) {
        GUID sub;
        if (SUCCEEDED(IMFMediaType_GetGUID(native, &MF_MT_SUBTYPE, &sub))) r->codec = sub.Data1;
        IMFMediaType_Release(native);
    }
    if (decoded) {
        IMFMediaType *out = NULL;
        MFCreateMediaType(&out);
        IMFMediaType_SetGUID(out, &MF_MT_MAJOR_TYPE, &MFMediaType_Video);
        IMFMediaType_SetGUID(out, &MF_MT_SUBTYPE, &MFVideoFormat_NV12);
        hr = IMFSourceReader_SetCurrentMediaType(reader, STREAM, NULL, out);
        IMFMediaType_Release(out);
        if (FAILED(hr)) {
            hos_log("movie: %s: no NV12 decoder (%08lx); is the VP9 Video Extensions package installed?", host_path,
                    (unsigned long)hr);
            movie_reader_close(r);
            return NULL;
        }
    }
    IMFMediaType *type = NULL;
    if (SUCCEEDED(IMFSourceReader_GetCurrentMediaType(reader, STREAM, &type))) {
        UINT32 w = 0, h = 0;
        UINT64 size = 0;
        if (SUCCEEDED(IMFMediaType_GetUINT64(type, &MF_MT_FRAME_SIZE, &size))) { w = (UINT32)(size >> 32); h = (UINT32)size; }
        r->width = (int)w;
        r->height = (int)h;
        UINT32 stride = 0;
        r->stride = SUCCEEDED(IMFMediaType_GetUINT32(type, &MF_MT_DEFAULT_STRIDE, &stride)) && (LONG)stride > 0
                        ? (LONG)stride : (LONG)w;
        IMFMediaType_Release(type);
    }
    if (width) *width = r->width;
    if (height) *height = r->height;
    if (duration_us) {
        PROPVARIANT v;
        PropVariantInit(&v);
        *duration_us = 0;
        if (SUCCEEDED(IMFSourceReader_GetPresentationAttribute(reader, (DWORD)MF_SOURCE_READER_MEDIASOURCE,
                                                               &MF_PD_DURATION, &v)) && v.vt == VT_UI8)
            *duration_us = (int64_t)(v.uhVal.QuadPart / 10);
        PropVariantClear(&v);
    }
    return r;
}

/* The next sample of the stream, or NULL at the end (or on an error). */
static IMFSample *next(MovieReader *r, int64_t *time_us) {
    com();
    for (;;) {
        DWORD stream = 0, flags = 0;
        LONGLONG ts = 0;
        IMFSample *s = NULL;
        HRESULT hr = IMFSourceReader_ReadSample(r->reader, STREAM, 0, &stream, &flags, &ts, &s);
        if (FAILED(hr)) { hos_log("movie: ReadSample failed (%08lx)", (unsigned long)hr); return NULL; }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { if (s) IMFSample_Release(s); return NULL; }
        if (!s) continue;   /* a stream tick or a format change: no data */
        if (time_us) *time_us = ts / 10;
        return s;
    }
}

int movie_reader_sample(MovieReader *r, void *dst, int cap, int64_t *time_us) {
    IMFSample *s = next(r, time_us);
    if (!s) return 0;
    IMFMediaBuffer *b = NULL;
    int n = -1;
    if (SUCCEEDED(IMFSample_ConvertToContiguousBuffer(s, &b))) {
        BYTE *p = NULL;
        DWORD len = 0;
        if (SUCCEEDED(IMFMediaBuffer_Lock(b, &p, NULL, &len))) {
            if ((int)len <= cap) { memcpy(dst, p, len); n = (int)len; }
            else hos_log("movie: a %lu byte sample does not fit a %d byte buffer", (unsigned long)len, cap);
            IMFMediaBuffer_Unlock(b);
        }
        IMFMediaBuffer_Release(b);
    }
    IMFSample_Release(s);
    return n;
}

/* Copies an NV12 picture with luma pitch `pitch` and `rows` luma rows allocated (>= height) to a packed one. */
static void copy_nv12(MovieReader *r, uint8_t *dst, const BYTE *src, LONG pitch, int rows) {
    const int w = r->width, h = r->height;
    for (int y = 0; y < h; y++) memcpy(dst + (size_t)y * w, src + (size_t)y * pitch, w);
    const BYTE *uv = src + (size_t)pitch * rows;
    uint8_t *duv = dst + (size_t)w * h;
    for (int y = 0; y < h / 2; y++) memcpy(duv + (size_t)y * w, uv + (size_t)y * pitch, w);
}

int movie_reader_frame(MovieReader *r, uint8_t *dst, int64_t *time_us) {
    IMFSample *s = next(r, time_us);
    if (!s) return 0;
    IMFMediaBuffer *b = NULL;
    if (SUCCEEDED(IMFSample_GetBufferByIndex(s, 0, &b))) {
        IMF2DBuffer2 *b2 = NULL;
        BYTE *p = NULL, *start = NULL;
        LONG pitch = 0;
        DWORD len = 0;
        if (SUCCEEDED(IMFMediaBuffer_QueryInterface(b, &IID_IMF2DBuffer2, (void **)&b2)) &&
            SUCCEEDED(IMF2DBuffer2_Lock2DSize(b2, MF2DBuffer_LockFlags_Read, &p, &pitch, &start, &len))) {
            /* the chroma plane follows the allocated luma rows, which a decoder may pad (1080 -> 1088) */
            int rows = pitch > 0 ? (int)((uint64_t)len * 2 / 3 / (uint64_t)pitch) : r->height;
            if (rows < r->height) rows = r->height;
            if (pitch > 0) copy_nv12(r, dst, p, pitch, rows);
            IMF2DBuffer2_Unlock2D(b2);
        } else if (SUCCEEDED(IMFMediaBuffer_Lock(b, &p, NULL, &len))) {
            int rows = r->stride > 0 ? (int)((uint64_t)len * 2 / 3 / (uint64_t)r->stride) : r->height;
            if (rows < r->height) rows = r->height;
            copy_nv12(r, dst, p, r->stride, rows);
            IMFMediaBuffer_Unlock(b);
        }
        if (b2) IMF2DBuffer2_Release(b2);
        IMFMediaBuffer_Release(b);
    }
    IMFSample_Release(s);
    return 1;
}

uint32_t movie_reader_codec(MovieReader *r) { return r->codec; }

void movie_reader_close(MovieReader *r) {
    if (!r) return;
    if (r->reader) IMFSourceReader_Release(r->reader);
    free(r);
}
