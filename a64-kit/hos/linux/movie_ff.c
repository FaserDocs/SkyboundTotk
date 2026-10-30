/* WebM / VP9 / H264 through FFmpeg. Same five movie_reader_* calls as movie_mf.c. */
#include "hos.h"
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct MovieReader {
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVPacket *pkt;
    AVFrame *frame;
    struct SwsContext *sws;
    int stream;
    int decoded, width, height;
    int pkt_ready;
    uint32_t codec;
};

static uint32_t fourcc_of(enum AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_VP9: return (uint32_t)'V' | ((uint32_t)'P' << 8) | ((uint32_t)'9' << 16) | ((uint32_t)'0' << 24);
    case AV_CODEC_ID_VP8: return (uint32_t)'V' | ((uint32_t)'P' << 8) | ((uint32_t)'8' << 16) | ((uint32_t)'0' << 24);
    case AV_CODEC_ID_H264: return (uint32_t)'H' | ((uint32_t)'2' << 8) | ((uint32_t)'6' << 16) | ((uint32_t)'4' << 24);
    default: return 0;
    }
}

static int64_t pts_us(const MovieReader *r, int64_t pts) {
    if (pts == AV_NOPTS_VALUE) return 0;
    const AVRational tb = r->fmt->streams[r->stream]->time_base;
    return av_rescale_q(pts, tb, (AVRational){1, 1000000});
}

int hos_linux_native_path(const char *in, char *out, size_t cap);

MovieReader *movie_reader_open(const char *host_path, int decoded, int *width, int *height, int64_t *duration_us) {
    char native[4096];
    /* nn_fs hands back a Windows path (backslashes). FFmpeg opens it itself, so it has to be a real Linux path. */
    if (host_path && hos_linux_native_path(host_path, native, sizeof native)) host_path = native;
    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, host_path, NULL, NULL) < 0 || avformat_find_stream_info(fmt, NULL) < 0) {
        hos_log("movie: %s: cannot open", host_path);
        if (fmt) avformat_close_input(&fmt);
        return NULL;
    }
    const int stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (stream < 0) {
        hos_log("movie: %s: no video stream", host_path);
        avformat_close_input(&fmt);
        return NULL;
    }
    MovieReader *r = (MovieReader *)calloc(1, sizeof *r);
    r->fmt = fmt;
    r->stream = stream;
    r->decoded = decoded;
    r->codec = fourcc_of(fmt->streams[stream]->codecpar->codec_id);
    r->width = fmt->streams[stream]->codecpar->width;
    r->height = fmt->streams[stream]->codecpar->height;
    r->pkt = av_packet_alloc();
    if (decoded) {
        const AVCodec *codec = avcodec_find_decoder(fmt->streams[stream]->codecpar->codec_id);
        r->dec = codec ? avcodec_alloc_context3(codec) : NULL;
        if (!r->dec || avcodec_parameters_to_context(r->dec, fmt->streams[stream]->codecpar) < 0 ||
            avcodec_open2(r->dec, codec, NULL) < 0) {
            hos_log("movie: %s: no decoder", host_path);
            movie_reader_close(r);
            return NULL;
        }
        r->frame = av_frame_alloc();
    }
    if (width) *width = r->width;
    if (height) *height = r->height;
    if (duration_us) *duration_us = fmt->duration > 0 ? fmt->duration : 0;
    return r;
}

static int next_packet(MovieReader *r) {
    if (r->pkt_ready) return 1;
    for (;;) {
        if (av_read_frame(r->fmt, r->pkt) < 0) return 0;
        if (r->pkt->stream_index == r->stream) {
            r->pkt_ready = 1;
            return 1;
        }
        av_packet_unref(r->pkt);
    }
}

int movie_reader_sample(MovieReader *r, void *dst, int cap, int64_t *time_us) {
    if (!next_packet(r)) return 0;
    if (r->pkt->size > cap) {
        hos_log("movie: a %d byte sample does not fit a %d byte buffer", r->pkt->size, cap);
        av_packet_unref(r->pkt);
        r->pkt_ready = 0;
        return -1;
    }
    memcpy(dst, r->pkt->data, (size_t)r->pkt->size);
    if (time_us) *time_us = pts_us(r, r->pkt->pts);
    const int n = r->pkt->size;
    av_packet_unref(r->pkt);
    r->pkt_ready = 0;
    return n;
}

static int copy_nv12(MovieReader *r, uint8_t *dst, const AVFrame *frame) {
    const int w = r->width, h = r->height;
    if (frame->format == AV_PIX_FMT_NV12 && frame->linesize[0] == w && frame->linesize[1] == w) {
        memcpy(dst, frame->data[0], (size_t)w * (size_t)h);
        memcpy(dst + (size_t)w * (size_t)h, frame->data[1], (size_t)w * (size_t)h / 2);
        return 1;
    }
    r->sws = sws_getCachedContext(r->sws, w, h, (enum AVPixelFormat)frame->format, w, h, AV_PIX_FMT_NV12, SWS_BILINEAR,
                                  NULL, NULL, NULL);
    if (!r->sws) return 0;
    uint8_t *planes[4] = {dst, dst + (size_t)w * (size_t)h, NULL, NULL};
    int lines[4] = {w, w, 0, 0};
    sws_scale(r->sws, (const uint8_t *const *)frame->data, frame->linesize, 0, h, planes, lines);
    return 1;
}

int movie_reader_frame(MovieReader *r, uint8_t *dst, int64_t *time_us) {
    if (!r->dec) return 0;
    for (;;) {
        int got = avcodec_receive_frame(r->dec, r->frame);
        if (got == 0) {
            if (time_us) *time_us = pts_us(r, r->frame->best_effort_timestamp);
            const int ok = copy_nv12(r, dst, r->frame);
            av_frame_unref(r->frame);
            return ok;
        }
        if (got != AVERROR(EAGAIN)) return 0;
        if (!next_packet(r)) {
            avcodec_send_packet(r->dec, NULL);
            got = avcodec_receive_frame(r->dec, r->frame);
            if (got != 0) return 0;
            if (time_us) *time_us = pts_us(r, r->frame->best_effort_timestamp);
            const int ok = copy_nv12(r, dst, r->frame);
            av_frame_unref(r->frame);
            return ok;
        }
        if (avcodec_send_packet(r->dec, r->pkt) < 0) {
            av_packet_unref(r->pkt);
            r->pkt_ready = 0;
            return 0;
        }
        av_packet_unref(r->pkt);
        r->pkt_ready = 0;
    }
}

uint32_t movie_reader_codec(MovieReader *r) { return r->codec; }

void movie_reader_close(MovieReader *r) {
    if (!r) return;
    sws_freeContext(r->sws);
    av_frame_free(&r->frame);
    av_packet_free(&r->pkt);
    avcodec_free_context(&r->dec);
    avformat_close_input(&r->fmt);
    free(r);
}
