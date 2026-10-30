/* TOTK ultrawide (F6 or RX_ULTRAWIDE=1, gpu/render/rx_present_fx.cpp): the game's cameras take the window's aspect
   instead of 16:9, and the present stretches the (still 16:9) frame over the whole window, so a 21:9 or 32:9 display
   shows more of the world to the sides at the right proportions (each pixel a little wider than it is tall; the HUD
   is stretched with it for now).
   TOTK v1.0.0 computes its gameplay cameras' aspect as width / height (an fdiv s0, s0, s1 stored at camera + 176:
   main+0x1968c2c and main+0x1e5eae8) and keeps one 16:9 constant (1.7777780, main+0x377ac54): the same three places
   the TOTK-Ultrawide tool (github.com/Fruithapje21/TOTK-Ultrawide) patches for 1.0.0. Here the division's result
   becomes hos_view_aspect when that is set (0: the game's own), and the constant follows it.
   v1.2.1's main has one such division left (0x09ecdea4), its 16:9 constant at main+0x3816ab4 and the HUD's loads
   at 0x092c77c4, 0x09adc630, 0x09add254 and 0x09ae177c (the addresses below, by HOS_GAME_VERSION). */
#include "hos.h"
#include <string.h>

float hos_view_aspect;

static uint32_t aspect_bits(void) {
    uint32_t b;
    memcpy(&b, &hos_view_aspect, 4);
    return b;
}

/* the camera's width / height, just computed into s0 */
static void camera_aspect(Ctx *c) {
    if (hos_view_aspect > 0) c->vreg[0][0] = aspect_bits(), c->vreg[0][1] = 0;
}
/* 1.0.0 has the division twice (main+0x1968c2c, main+0x1e5eae8); 1.2.1's main has one camera path left that
   computes it (0x09ecdea4). */
#if HOS_GAME_VERSION == 121
SPLICE_AFTER(0x09ecdea4, "a64_vset64(c->vreg[0],_v);}") { camera_aspect(c); }
#else
SPLICE_AFTER(0x09968b68, "a64_vset64(c->vreg[0],_v);}") { camera_aspect(c); }
SPLICE_AFTER(0x09e5ea24, "a64_vset64(c->vreg[0],_v);}") { camera_aspect(c); }
#endif

/* hos_view_aspect changed (the renderer: the window's size or the setting): the 16:9 constant follows it */
void hos_ultrawide_apply(void) {
#if HOS_GAME_VERSION == 121
    static const uint64_t kAspectConstant = 0x0B816AB4ULL;   /* main+0x3816ab4 */
#else
    static const uint64_t kAspectConstant = 0x0B77AC54ULL;   /* main+0x377ac54 */
#endif
    static uint32_t original;
    if (!original) original = G32(kAspectConstant);
    if (original != 0x3FE38E3Bu) return;   /* (not a main this was written for: left alone) */
    G32(kAspectConstant) = hos_view_aspect > 0 ? aspect_bits() : original;
}

/* The HUD: TOTK's 2D layout loads a pair of sizes / scales (ldp s0, s1) at five places (1.0.0: main+0x12ade68,
   +0x12ae24c, +0x1a8e69c, +0x1a8f18c, +0x1a93954: the places TOTK-Ultrawide hooks); its horizontal one times
   aspect / (16:9) lays the HUD out for the wider view, so after the present's stretch it keeps its proportions. */
static void ui_scale(Ctx *c) {
    if (hos_view_aspect <= 0) return;
    float f;
    memcpy(&f, &c->vreg[0][0], 4);
    f *= hos_view_aspect / (16.0f / 9.0f);
    uint32_t b;
    memcpy(&b, &f, 4);
    c->vreg[0][0] = b;
}
/* (1.2.1's main+0x1ade58 loads the pair the other way round: the horizontal value lands in s1) */
static void ui_scale_second(Ctx *c) {
    if (hos_view_aspect <= 0) return;
    float f;
    memcpy(&f, &c->vreg[1][0], 4);
    f *= hos_view_aspect / (16.0f / 9.0f);
    uint32_t b;
    memcpy(&b, &f, 4);
    c->vreg[1][0] = b;
}
#if HOS_GAME_VERSION == 121
SPLICE_AFTER_EACH(0x092c77c4, "_b=c->x[8]; int64_t _o=0; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09adc630, "_b=c->x[31]; int64_t _o=8; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[1],_p0); a64_vset64(c->vreg[0],_p1);") { ui_scale_second(c); }
SPLICE_AFTER(0x09add254, "_b=c->x[29]; int64_t _o=24; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09ae177c, "_b=c->x[20]; int64_t _o=32; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
#else
SPLICE_AFTER_EACH(0x092ada34, "_b=c->x[8]; int64_t _o=0; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09a8e5d4, "_b=c->x[31]; int64_t _o=8; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09a8f0e0, "_b=c->x[29]; int64_t _o=24; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09a935ec, "_b=c->x[20]; int64_t _o=32; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
#endif
