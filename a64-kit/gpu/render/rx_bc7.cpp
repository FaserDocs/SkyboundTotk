#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "video_core/textures/workers.h"

namespace rx {
namespace {

constexpr int kWeights[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

struct Fit {
    int q[2][4];
    int p[2];
    uint8_t index[16];
    int64_t error;
};

void quantize(const float e[4], int q[4], int& p) {
    int64_t best = INT64_MAX;
    for (int pb = 0; pb < 2; pb++) {
        int64_t err = 0;
        int t[4];
        for (int c = 0; c < 4; c++) {
            t[c] = std::clamp((int)((e[c] - pb) * 0.5f + 0.5f), 0, 127);
            const float d = e[c] - (float)(t[c] * 2 + pb);
            err += (int64_t)(d * d * 16.0f);
        }
        if (err < best) {
            best = err;
            p = pb;
            std::memcpy(q, t, sizeof t);
        }
    }
}

void choose_indices(const uint8_t px[16][4], Fit& f) {
    int pal[16][4];
    for (int k = 0; k < 16; k++)
        for (int c = 0; c < 4; c++) {
            const int a = f.q[0][c] * 2 + f.p[0], b = f.q[1][c] * 2 + f.p[1];
            pal[k][c] = ((64 - kWeights[k]) * a + kWeights[k] * b + 32) >> 6;
        }
    f.error = 0;
    for (int i = 0; i < 16; i++) {
        int best = 0;
        int64_t best_err = INT64_MAX;
        for (int k = 0; k < 16; k++) {
            int64_t e = 0;
            for (int c = 0; c < 4; c++) {
                const int d = px[i][c] - pal[k][c];
                e += d * d;
            }
            if (e < best_err) {
                best_err = e;
                best = k;
            }
        }
        f.index[i] = (uint8_t)best;
        f.error += best_err;
    }
}

Fit fit_endpoints(const uint8_t px[16][4], const float e0[4], const float e1[4]) {
    Fit f;
    quantize(e0, f.q[0], f.p[0]);
    quantize(e1, f.q[1], f.p[1]);
    choose_indices(px, f);
    return f;
}

void put(uint64_t (&w)[2], int& pos, uint64_t v, int n) {
    if (pos < 64) {
        w[0] |= v << pos;
        if (pos + n > 64) w[1] |= v >> (64 - pos);
    } else {
        w[1] |= v << (pos - 64);
    }
    pos += n;
}

void encode_block(const uint8_t px[16][4], uint8_t out[16]) {
    float mean[4] = {0, 0, 0, 0};
    for (int i = 0; i < 16; i++)
        for (int c = 0; c < 4; c++) mean[c] += px[i][c];
    for (float& m : mean) m *= 1.0f / 16.0f;
    float cov[4][4] = {};
    for (int i = 0; i < 16; i++) {
        float d[4];
        for (int c = 0; c < 4; c++) d[c] = px[i][c] - mean[c];
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) cov[a][b] += d[a] * d[b];
    }
    float axis[4] = {cov[0][0], cov[1][1], cov[2][2], cov[3][3]};
    for (int it = 0; it < 8; it++) {
        float n[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) n[a] += cov[a][b] * axis[b];
        float m = 0;
        for (float v : n) m = std::max(m, std::fabs(v));
        if (m < 1e-6f) break;
        for (int a = 0; a < 4; a++) axis[a] = n[a] / m;
    }
    float len = 0;
    for (float v : axis) len += v * v;
    len = std::sqrt(len);
    float tmin = 0, tmax = 0;
    if (len > 1e-6f) {
        for (float& v : axis) v /= len;
        tmin = 1e30f;
        tmax = -1e30f;
        for (int i = 0; i < 16; i++) {
            float t = 0;
            for (int c = 0; c < 4; c++) t += (px[i][c] - mean[c]) * axis[c];
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
    }
    float e0[4], e1[4];
    for (int c = 0; c < 4; c++) {
        e0[c] = std::clamp(mean[c] + axis[c] * tmin, 0.0f, 255.0f);
        e1[c] = std::clamp(mean[c] + axis[c] * tmax, 0.0f, 255.0f);
    }
    Fit best = fit_endpoints(px, e0, e1);
    if (best.error > 64) {
        float aa = 0, ab = 0, bb = 0, ax[4] = {0, 0, 0, 0}, bx[4] = {0, 0, 0, 0};
        for (int i = 0; i < 16; i++) {
            const float b = kWeights[best.index[i]] / 64.0f, a = 1.0f - b;
            aa += a * a;
            ab += a * b;
            bb += b * b;
            for (int c = 0; c < 4; c++) {
                ax[c] += a * px[i][c];
                bx[c] += b * px[i][c];
            }
        }
        const float det = aa * bb - ab * ab;
        if (std::fabs(det) > 1e-6f) {
            float r0[4], r1[4];
            for (int c = 0; c < 4; c++) {
                r0[c] = std::clamp((bb * ax[c] - ab * bx[c]) / det, 0.0f, 255.0f);
                r1[c] = std::clamp((aa * bx[c] - ab * ax[c]) / det, 0.0f, 255.0f);
            }
            const Fit refit = fit_endpoints(px, r0, r1);
            if (refit.error < best.error) best = refit;
        }
    }
    if (best.index[0] & 8) {
        std::swap(best.q[0], best.q[1]);
        std::swap(best.p[0], best.p[1]);
        for (uint8_t& k : best.index) k = (uint8_t)(15 - k);
    }
    uint64_t w[2] = {0, 0};
    int pos = 0;
    put(w, pos, 1u << 6, 7);
    for (int c = 0; c < 4; c++) {
        put(w, pos, (uint64_t)best.q[0][c], 7);
        put(w, pos, (uint64_t)best.q[1][c], 7);
    }
    put(w, pos, (uint64_t)best.p[0], 1);
    put(w, pos, (uint64_t)best.p[1], 1);
    put(w, pos, best.index[0], 3);
    for (int i = 1; i < 16; i++) put(w, pos, best.index[i], 4);
    std::memcpy(out, w, 16);
}

void encode_rows(const uint8_t* rgba, uint32_t width, uint32_t height, uint32_t pitch, uint32_t by0, uint32_t by1,
                 uint8_t* out) {
    const uint32_t bw = (width + 3) / 4;
    uint8_t px[16][4];
    for (uint32_t by = by0; by < by1; by++)
        for (uint32_t bx = 0; bx < bw; bx++) {
            for (uint32_t y = 0; y < 4; y++)
                for (uint32_t x = 0; x < 4; x++) {
                    const uint32_t sx = std::min(bx * 4 + x, width - 1), sy = std::min(by * 4 + y, height - 1);
                    std::memcpy(px[y * 4 + x], rgba + (size_t)sy * pitch + (size_t)sx * 4, 4);
                }
            encode_block(px, out + ((size_t)by * bw + bx) * 16);
        }
}

} // namespace

void bc7_encode(const uint8_t* rgba, uint32_t width, uint32_t height, uint32_t pitch, uint8_t* out) {
    if (!width || !height) return;
    const uint32_t bh = (height + 3) / 4;
    const uint32_t per_job = std::max<uint32_t>(1, 16384 / std::max<uint32_t>(1, (width + 3) / 4));
    if (bh <= per_job) {
        encode_rows(rgba, width, height, pitch, 0, bh, out);
        return;
    }
    auto& workers = Tegra::Texture::GetThreadWorkers();
    for (uint32_t by = 0; by < bh; by += per_job) {
        const uint32_t end = std::min(bh, by + per_job);
        workers.QueueWork([=] { encode_rows(rgba, width, height, pitch, by, end, out); });
    }
    workers.WaitForRequests();
}

} // namespace rx
