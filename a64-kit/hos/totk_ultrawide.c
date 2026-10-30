/* TOTK ultrawide (F6 or RX_ULTRAWIDE=1, gpu/render/rx_present_fx.cpp): the game's cameras take the window's aspect
   instead of 16:9, and the present stretches the (still 16:9) frame over the whole window, so a 21:9 or 32:9 display
   shows more of the world to the sides at the right proportions (each pixel a little wider than it is tall; the HUD
   is stretched with it for now).
   TOTK v1.0.0 computes its gameplay cameras' aspect as width / height (an fdiv s0, s0, s1 stored at camera + 176:
   main+0x1968c2c and main+0x1e5eae8) and keeps one 16:9 constant (1.7777780, main+0x377ac54): the same three places
   the TOTK-Ultrawide tool (github.com/Fruithapje21/TOTK-Ultrawide) patches for 1.0.0. Here the division's result
   becomes hos_view_aspect when that is set (0: the game's own), and the constant follows it.
   A window narrower than 16:9 (16:10, 4:3; SkyboundTotk issue 10) works the same way: the cameras take its aspect (the
   same height of view, less to the sides, as the game's vertical field of view is fixed) and the 16:9 frame is
   stretched taller; the HUD is squeezed vertically instead (totk_ultrawide_ui.c), the five UI-scale sites stay. */
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
SPLICE_AFTER(0x09968b68, "a64_vset64(c->vreg[0],_v);}") { camera_aspect(c); }
SPLICE_AFTER(0x09e5ea24, "a64_vset64(c->vreg[0],_v);}") { camera_aspect(c); }

/* hos_view_aspect changed (the renderer: the window's size or the setting): the 16:9 constant follows it */
void hos_ultrawide_apply(void) {
    static const uint64_t kAspectConstant = 0x0B77AC54ULL;   /* main+0x377ac54 */
    static uint32_t original;
    if (!original) original = G32(kAspectConstant);
    if (original != 0x3FE38E3Bu) return;   /* (not TOTK v1.0.0's main: left alone) */
    G32(kAspectConstant) = hos_view_aspect > 0 ? aspect_bits() : original;
}

/* The HUD: TOTK's 2D layout loads a pair of sizes / scales (ldp s0, s1) at five places (main+0x12ade68, +0x12ae24c,
   +0x1a8e69c, +0x1a8f18c, +0x1a93954: the places TOTK-Ultrawide hooks); its horizontal one times aspect / (16:9)
   lays the HUD out for the wider view, so after the present's stretch it keeps its proportions. */
static void ui_scale(Ctx *c) {
    if (hos_view_aspect <= 16.0f / 9.0f) return;   /* (wider than 16:9 only) */
    float f;
    memcpy(&f, &c->vreg[0][0], 4);
    f *= hos_view_aspect / (16.0f / 9.0f);
    uint32_t b;
    memcpy(&b, &f, 4);
    c->vreg[0][0] = b;
}
SPLICE_AFTER_EACH(0x092ada34, "_b=c->x[8]; int64_t _o=0; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09a8e5d4, "_b=c->x[31]; int64_t _o=8; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09a8f0e0, "_b=c->x[29]; int64_t _o=24; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
SPLICE_AFTER(0x09a935ec, "_b=c->x[20]; int64_t _o=32; { uint64_t _p0,_p1; recomp_ldp32(c,(_b+_o),&_p0,&_p1); a64_vset64(c->vreg[0],_p0); a64_vset64(c->vreg[1],_p1);") { ui_scale(c); }
