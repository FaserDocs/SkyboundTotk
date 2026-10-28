/* TOTK's frustum test, natively. f_08704b7c (v1.0.0) is bool Frustum::isVisible(const AABB *box): the frustum in x0
   (its bounding box at +0x00 / +0x10, planes at +0x20 + 32 i as a normal (4 floats) and a distance at +0x10, the plane
   count at +0xE0, "test the bounding box first" at +0xE4), the box in x1 (min at +0x00, max at +0x10). It runs for
   every object the scene visits, on the ModuleSystem workers the frame waits for: ~5% of the castle's samples, the
   biggest single function, ~100 host instructions a plane as lifted code.

   The same operations in the same order as the guest's: with the flag set, the boxes are disjoint when one's min is
   above the other's max on an axis (FCMGT); then for each plane the corner furthest along the normal (each axis: max
   when the normal's component is < 0, FCMP + FCSEL), p = n * (corner, 0) lane by lane, (p2 + p0) + (p3 + p1), + d; the
   box is outside when that is > 0. Exact: NaNs compare false on both sides; FPCR must be round-to-nearest (FZ on or
   off) and FPSR's IOC, OFC and IXC already raised (what these operations raise; TOTK keeps FPSR = 0x9F), else the
   lifted code runs; under FZ the guest flushes subnormal inputs and tiny results, so any subnormal input, or any
   product or sum that is tiny (|x| <= FLT_MIN, not 0), sends the call to the lifted code too.
   HOS_NO_NATIVE_CULL=1: always the lifted code. HOS_CULL_CHECK=1 (testing): the lifted code runs after each native
   answer and the two are compared (counts and the first differences logged). */
#include "hos.h"
#include <stdlib.h>
#include <string.h>

#pragma clang fp contract(off)

void f_08704b7c(Ctx *);

static int cull_off(void) {
    static int off = -1;
    if (off < 0) off = getenv("HOS_NO_NATIVE_CULL") != NULL;
    return off;
}

static int cull_check(void) {
    static int on = -1;
    if (on < 0) on = getenv("HOS_CULL_CHECK") != NULL;
    return on;
}
static _Thread_local int in_check;   /* (the lifted code's own entry hook lets it run) */
static long check_calls, check_native, check_diff;

static int native_visible(Ctx *c, uint64_t *visible);

ENTRY_HOOK(0x08704b7c) {
    if (in_check || cull_off() || (c->fpcr != 0 && c->fpcr != 0x01000000) || (c->fpsr & 0x15) != 0x15) return 0;
    uint64_t v;
    if (!cull_check()) {
        if (!native_visible(c, &v)) return 0;
        c->x[0] = v;
        return 1;
    }
    const int done = native_visible(c, &v);
    in_check = 1;
    f_08704b7c(c);   /* (x0, x1 as the caller set them: the hook changed nothing yet) */
    in_check = 0;
    const long n = __atomic_add_fetch(&check_calls, 1, __ATOMIC_RELAXED);
    if (done) {
        __atomic_add_fetch(&check_native, 1, __ATOMIC_RELAXED);
        if (c->x[0] != v && __atomic_add_fetch(&check_diff, 1, __ATOMIC_RELAXED) <= 20)
            hos_log("cull check: native %llu, lifted %llu", (unsigned long long)v, (unsigned long long)c->x[0]);
    }
    if (!(n & 0xFFFFF))
        hos_log("cull check: %ld calls, %ld answered natively, %ld different", n, check_native, check_diff);
    return 1;
}

static float gf(uint64_t a) {
    float f;
    memcpy(&f, GP(a), 4);
    return f;
}

/* subnormal (an input the guest would flush) */
static int sub(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return !(b & 0x7f800000u) && (b & 0x7fffffu);
}

/* tiny (a result the guest might flush: up to FLT_MIN, rounding before or after) */
static int tiny(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return (b & 0x7fffffffu) - 1u < 0x00800000u;
}

/* 1 with *visible when the native answer is the guest's; 0 when the lifted code must run */
static int native_visible(Ctx *c, uint64_t *visible_out) {
    const uint64_t f = c->x[0], b = c->x[1];
    const float bmin[3] = {gf(b), gf(b + 4), gf(b + 8)}, bmax[3] = {gf(b + 16), gf(b + 20), gf(b + 24)};
    int bad = 0;
    for (int i = 0; i < 3; i++) bad |= sub(bmin[i]) | sub(bmax[i]);
    if (G32(f + 0xE4)) {
        const float fmin[3] = {gf(f), gf(f + 4), gf(f + 8)}, fmax[3] = {gf(f + 16), gf(f + 20), gf(f + 24)};
        int out = 0;
        for (int i = 0; i < 3; i++) {
            bad |= sub(fmin[i]) | sub(fmax[i]);
            out |= (fmin[i] > bmax[i]) | (bmin[i] > fmax[i]);
        }
        if (bad) return 0;
        if (out) {
            *visible_out = 0;
            return 1;
        }
    }
    const int32_t count = (int32_t)G32(f + 0xE0);
    int visible = 1;
    for (int32_t i = 0; i < count && visible; i++) {
        const uint64_t p = f + 0x20 + 32 * (uint64_t)i;
        const float n[4] = {gf(p), gf(p + 4), gf(p + 8), gf(p + 12)}, d = gf(p + 16);
        for (int k = 0; k < 4; k++) bad |= sub(n[k]);
        bad |= sub(d);
        const float x = n[0] < 0.0f ? bmax[0] : bmin[0];
        const float y = n[1] < 0.0f ? bmax[1] : bmin[1];
        const float z = n[2] < 0.0f ? bmax[2] : bmin[2];
        const float p0 = n[0] * x, p1 = n[1] * y, p2 = n[2] * z, p3 = n[3] * 0.0f;
        const float s0 = p2 + p0, s1 = p3 + p1;
        const float t = s0 + s1;
        const float dist = t + d;
        bad |= tiny(p0) | tiny(p1) | tiny(p2) | tiny(p3) | tiny(s0) | tiny(s1) | tiny(t) | tiny(dist);
        if (dist > 0.0f) visible = 0;
    }
    if (bad) return 0;
    *visible_out = (uint64_t)visible;
    return 1;
}
