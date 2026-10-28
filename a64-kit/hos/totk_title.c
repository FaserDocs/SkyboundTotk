/* TOTK title menu credits: two entries under "amiibo".
     "Ported By NotRexed & slqnt": the game's own two-button system window, "NotRexed" opens
       https://github.com/NotRexed and "slqnt" https://slqnt.dev/
     "wasm.rip": opens https://wasm.rip
   Everything on screen is the game's own UI, driven from the outside:
   - Title_00.bflyt has six button parts (Pa_Button_00..05, in an auto-aligned list); the game shows four (Continue,
     New Game, Options, amiibo). The first two hidden ones are made visible (ui2d Pane flag bit 0 at pane+0x58, name
     at pane+0xB0) and given our text with the game's text setter; the menu's cursor then reaches them as usual.
   - Our entries never get "decided" by the game: deciding one puts the menu in its waiting-for-the-next-screen state,
     and the action a button starts comes from the layout's user data. So A on them is taken out of the input
     (nn_hid asks hos_title_input every read) and our action runs instead.
   - The window: there is no plain "show a message" call to reuse, and the title flow only opens it for the real New
     Game button (a decide of another button carrying New Game's message identity stalls it). So "Ported By" moves
     the cursor to New Game and presses A (injected input, a few frames), which asks "Overwrite previous save data?"
     in that window whenever there is a save. While that request is ours, the window's message and button texts are
     replaced (hooks on its text setters). The window answers by button POSITION and its first button is Start (a
     new game, tested): A there never reaches the game, the link opens and the window stays; the second button is
     its Cancel, so A goes through (closing it) and the cursor is moved back to "Ported By". Without a save (no
     Continue, so New Game would start at once) the entry opens the NotRexed link directly.
   Addresses and offsets are TOTK v1.0.0's. HOS_NO_CREDITS=1 leaves the menu alone; HOS_CREDITS_DRYRUN=1 logs the
   links instead of opening them (test runs). */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

void f_086f7164(Ctx *);   /* (ui2d layout of a UI object, +40; pane name) -> pane */
void f_08b90810(Ctx *);   /* (same layout; "Part/Pane" path; message view; 1; 0): set a text pane's string */
void f_08b906ec(Ctx *);   /* (message view): empty view { text 0, length 0, message -1 } */
void f_08b907a0(Ctx *);   /* (message view, &file name, &label, 1): the view of that message */
void f_09ad99dc(Ctx *);   /* system window: (window, button index, message index) -> that button's label */
void f_09ad9a98(Ctx *);   /* system window: (window, message) -> its text */

enum {
    PANE_FLAG = 0x58, PANE_NAME = 0xB0,                      /* nn::ui2d::Pane */
    UI_LAYOUT = 40,                                          /* UI objects (menu, window): their layout */
    MENU_BUTTON_COUNT = 0x1A0, MENU_BUTTONS = 0x1A8,         /* title menu */
    MENU_HOVER = 0x1B8, MENU_INPUT = 0xB0, MENU_STATE = 0x3C0, /* hovered button; 1 and 3 while taking input */
    WINDOW_BUTTON_COUNT = 424, WINDOW_BUTTONS = 432,         /* system window */
    BUTTON_STATE = 0x38,                                     /* button controller: 3 while hovered */
};
enum { PAD_A = 1 << 0, PAD_UP = 1 << 13, PAD_DOWN = 1 << 15 };   /* nn_hid's button bits */

static const char *const PORTED = "Ported By NotRexed & slqnt", *const WASM = "wasm.rip";

static int disabled = -1;
static uint64_t menu, menu_vtable, ported_btn, wasm_btn;   /* the title menu and our two button controllers */
static int has_save, ported_index, newgame_index;
static volatile int window_wanted;   /* "Ported By" asked for the window: the next one to set its texts is ours */
static volatile uint64_t window;     /* that window, while it shows our choice */
static volatile int64_t window_seen;
extern volatile LONG64 hos_present_count;
static volatile LONG in_call;        /* our own calls into hooked functions run them as the game would */

static int credits_off(void) {
    if (disabled < 0) disabled = getenv("HOS_NO_CREDITS") != NULL;
    return disabled;
}

static void open_url(const char *url) {
    static int dry = -1;
    if (dry < 0) dry = getenv("HOS_CREDITS_DRYRUN") != NULL;
    hos_log("title: opening %s%s", url, dry ? " (dry run)" : "");
    if (!dry) ShellExecuteA(NULL, "open", url, NULL, NULL, SW_SHOWNORMAL);
}

/* Calls lifted guest code from a hook: x0-x4 set, the guest stack below `sp`, every register restored after (the
   hooked function then runs with its own arguments). Returns x0. */
static uint64_t call(Ctx *c, void (*fn)(Ctx *), uint64_t sp, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                     uint64_t a4) {
    uint64_t x[32], v[32][2];
    const uint8_t f[4] = {c->n, c->z, c->c, c->v};
    memcpy(x, c->x, sizeof x);
    memcpy(v, c->vreg, sizeof v);
    c->x[0] = a0;
    c->x[1] = a1;
    c->x[2] = a2;
    c->x[3] = a3;
    c->x[4] = a4;
    c->x[31] = sp & ~15ULL;
    InterlockedIncrement(&in_call);
    fn(c);
    InterlockedDecrement(&in_call);
    const uint64_t r = c->x[0];
    memcpy(c->x, x, sizeof x);
    memcpy(c->vreg, v, sizeof v);
    c->n = f[0];
    c->z = f[1];
    c->c = f[2];
    c->v = f[3];
    return r;
}

/* strings for those calls, pushed below the guest stack pointer */
static uint64_t push_str(uint64_t *top, const char *s) {
    const size_t n = strlen(s) + 1;
    *top = (*top - n) & ~15ULL;
    memcpy(GP(*top), s, n);
    return *top;
}
static uint64_t push_wstr(uint64_t *top, const char *s) {
    const size_t n = strlen(s) + 1;
    *top = (*top - 2 * n) & ~15ULL;
    uint16_t *w = (uint16_t *)GP(*top);
    for (size_t i = 0; i < n; i++) w[i] = (uint8_t)s[i];
    return *top;
}

static uint64_t find_pane(Ctx *c, uint64_t layout, const char *name) {
    uint64_t top = c->x[31] - 256;
    const uint64_t n = push_str(&top, name);
    const uint64_t pane = call(c, f_086f7164, top - 64, layout, n, 0, 0, 0);
    if (pane && strcmp((const char *)GP(pane + PANE_NAME), name) != 0) {
        hos_log("title: pane %s at %#llx has another name: not the layout this was written for", name,
                (unsigned long long)pane);
        return 0;
    }
    return pane;
}

/* The text goes in as the game's message view: { const char16_t *text; u32 length; u64 tag info (-1: none, as
   f_08b906ec makes it) }. Returns 0 when the path names no text pane. */
static int set_text(Ctx *c, uint64_t layout, const char *path, const char *text) {
    uint64_t top = c->x[31] - 256;
    const uint64_t p = push_str(&top, path), t = push_wstr(&top, text);
    top = (top - 32) & ~15ULL;
    G64(top) = t;
    *(uint32_t *)GP(top + 8) = (uint32_t)strlen(text);
    G64(top + 16) = ~0ULL;
    return (int)(call(c, f_08b90810, top - 64, layout, p, top, 1, 0) & 1);
}

/* Our text, but carrying a game message's identity: a chosen button posts the message file and label behind its text
   to the title flow (murmur3 of "LayoutMsg/Title_00" and "0000"...), which picks the action from them. So a button
   labelled like this does what the button with that message does. */
static int set_text_as(Ctx *c, uint64_t layout, const char *path, const char *text, const char *file,
                       const char *label) {
    uint64_t top = c->x[31] - 512;
    const uint64_t f = push_str(&top, file), l = push_str(&top, label), t = push_wstr(&top, text);
    const uint64_t p = push_str(&top, path);
    top = (top - 96) & ~15ULL;
    const uint64_t view = top, fobj = top + 32, lobj = top + 64;
    G64(fobj) = f;
    G64(fobj + 8) = strlen(file);
    G64(lobj) = l;
    G64(lobj + 8) = strlen(label);
    call(c, f_08b906ec, top - 64, view, 0, 0, 0, 0);
    if (!(call(c, f_08b907a0, top - 64, view, fobj, lobj, 1, 0) & 1)) {
        hos_log("title: no message %s:%s", file, label);
        return set_text(c, layout, path, text);
    }
    G64(view) = t;
    *(uint32_t *)GP(view + 8) = (uint32_t)strlen(text);
    return (int)(call(c, f_08b90810, top - 64, layout, p, view, 1, 0) & 1);
}

/* The title menu reports each hover through f_09ada6c0 (menu, button): at the first one its layout is ready. Shows
   and labels our two entries, once per menu. */
static void setup_menu(Ctx *c, uint64_t m) {
    const uint64_t layout = G64(m + UI_LAYOUT), buttons = G64(m + MENU_BUTTONS);
    if (!layout || *(uint32_t *)GP(m + MENU_BUTTON_COUNT) != 6 || !buttons) return;
    uint64_t panes[6];
    int shown = 0;
    for (int i = 0; i < 6; i++) {
        char name[16];
        snprintf(name, sizeof name, "Pa_Button_%02d", i);
        if (!(panes[i] = find_pane(c, layout, name))) return;
        if (*(uint8_t *)GP(panes[i] + PANE_FLAG) & 1) shown = i + 1;
    }
    menu = m;
    menu_vtable = G64(m);
    if (shown > 4) return;   /* (ours are already showing) */
    if (shown < 3) {
        hos_log("title: %d menu entries, expected 3 or 4: credits not added", shown);
        return;
    }
    has_save = shown == 4;   /* Continue is there */
    ported_index = shown;
    newgame_index = has_save ? 1 : 0;
    ported_btn = G64(buttons + 8 * shown);
    wasm_btn = G64(buttons + 8 * (shown + 1));
    const char *texts[2] = {PORTED, WASM};
    for (int k = 0; k < 2; k++) {
        *(uint8_t *)GP(panes[shown + k] + PANE_FLAG) |= 1;
        char path[40];
        snprintf(path, sizeof path, "Pa_Button_%02d/T_Btn_00", shown + k);
        if (!set_text(c, layout, path, texts[k])) hos_log("title: no text pane %s", path);
    }
    hos_log("title: credits added as entries %d and %d%s", shown, shown + 1, has_save ? "" : " (no save)");
}

ENTRY_HOOK(0x09ada6c0) {
    if (credits_off() || in_call) return 0;
    if (c->x[0] != menu || !ported_btn) setup_menu(c, c->x[0]);
    return 0;
}

/* the system window's button labels: (window, index, message index) */
ENTRY_HOOK(0x09ad99dc) {
    if (credits_off() || in_call || !window_wanted) return 0;
    window = c->x[0];
    window_seen = hos_present_count;
    const uint32_t i = (uint32_t)c->x[1];
    call(c, f_09ad99dc, c->x[31] - 64, c->x[0], c->x[1], c->x[2], c->x[3], c->x[4]);
    char path[40];
    snprintf(path, sizeof path, "Pa_Button_%02u/T_Btn_00", i);
    set_text_as(c, G64(window + UI_LAYOUT), path, i == 0 ? "NotRexed" : "slqnt", "LayoutMsg/SystemWindow_00", "1001");
    return 1;
}

/* the system window's message: (window, message) */
ENTRY_HOOK(0x09ad9a98) {
    if (credits_off() || in_call || !window_wanted) return 0;
    window = c->x[0];
    window_seen = hos_present_count;
    call(c, f_09ad9a98, c->x[31] - 64, c->x[0], c->x[1], c->x[2], c->x[3], c->x[4]);
    set_text(c, G64(window + UI_LAYOUT), "T_Text_00", PORTED);
    return 1;
}

/* the system window's per-frame update: (window, ...). The game writes the window's message once more when it
   opens, so ours goes back in during its first moments. */
static int64_t window_opened;
ENTRY_HOOK(0x09ad9858) {
    if (credits_off() || in_call || !window || c->x[0] != window) return 0;
    const int64_t now = hos_now_ns();
    window_seen = hos_present_count;
    if (!window_opened) window_opened = now;
    if (now - window_opened < 1500000000LL) set_text(c, G64(window + UI_LAYOUT), "T_Text_00", PORTED);
    return 0;
}

static int menu_taking_input(void) {
    return menu && G64(menu) == menu_vtable && *(uint32_t *)GP(menu + MENU_STATE) == 3 &&
           *(uint8_t *)GP(menu + MENU_INPUT);
}

static uint64_t window_button(int i) {
    const uint64_t w = window;
    if (!w || *(uint32_t *)GP(w + WINDOW_BUTTON_COUNT) < 2) return 0;
    const uint64_t buttons = G64(w + WINDOW_BUTTONS);
    return buttons ? G64(buttons + 8 * (uint64_t)i) : 0;
}

/* injected input: presses the game reads as the player's, each held for a few frames */
static struct { uint64_t buttons; int ms; } steps[16];
static int nsteps, step;
static int64_t step_end;

static void inject(uint64_t buttons, int ms) {
    if (nsteps < (int)(sizeof steps / sizeof *steps)) {
        steps[nsteps].buttons = buttons;
        steps[nsteps].ms = ms;
        nsteps++;
    }
}

/* nn_hid, every input read: the buttons the game will see. */
uint64_t hos_title_input(uint64_t buttons) {
    static int swallowing, a_was_down;
    if (credits_off() || !menu) return buttons;
    if (G64(menu) != menu_vtable) {
        menu = menu_vtable = ported_btn = wasm_btn = 0;
        window = 0;
        window_wanted = 0;
        window_opened = 0;
        nsteps = step = 0;
        step_end = 0;
        swallowing = a_was_down = 0;
        return buttons;
    }
    if (step < nsteps) {   /* our presses replace the player's until they are done */
        const int64_t now = hos_now_ns();
        if (!step_end) step_end = now + steps[step].ms * 1000000LL;
        while (step < nsteps && now >= step_end) {
            step++;
            step_end = step < nsteps ? now + steps[step].ms * 1000000LL : 0;
        }
        if (step < nsteps) return steps[step].buttons;
        nsteps = step = 0;
        step_end = 0;
        swallowing = (buttons & PAD_A) != 0;
        a_was_down = (buttons & PAD_A) != 0;
        return buttons & ~(uint64_t)PAD_A;
    }
    if (window && hos_present_count - window_seen > 30) {
        window = 0;
        window_opened = 0;
        window_wanted = 0;
    }
    const int a = (buttons & PAD_A) != 0, pressed = a && !a_was_down;
    a_was_down = a;
    if (!a) swallowing = 0;
    if (swallowing) return buttons & ~(uint64_t)PAD_A;
    if (!pressed) return buttons;

    /* our window: both buttons carry Cancel's message, so the game closes it; their links open here */
    if (window && menu_taking_input()) {   /* (closed with B: the menu has the input again) */
        window = 0;
        window_opened = 0;
        window_wanted = 0;
    }
    if (window) {
        const uint64_t first = window_button(0), second = window_button(1);
        const int on_first = first && *(uint32_t *)GP(first + BUTTON_STATE) == 3;
        const int on_second = second && *(uint32_t *)GP(second + BUTTON_STATE) == 3;
        /* The window answers by button position, and its first button is Start (a new game): A there NEVER reaches
           the game. It opens the link and the window stays (B or "slqnt" closes it). The second button is the
           window's Cancel: the press goes through and closes it, then the cursor goes back to "Ported By". */
        if (on_first) {
            open_url("https://github.com/NotRexed");
            swallowing = 1;
            return buttons & ~(uint64_t)PAD_A;
        }
        if (on_second) {
            open_url("https://slqnt.dev/");
            window = 0;
            window_opened = 0;
            window_wanted = 0;
            inject(PAD_A, 80);
            inject(0, 1500);
            for (int i = newgame_index; i < ported_index; i++) {
                inject(PAD_DOWN, 40);
                inject(0, 40);
            }
            swallowing = 1;
            return buttons & ~(uint64_t)PAD_A;
        }
        return buttons & ~(uint64_t)PAD_A;   /* (no button hovered: nothing to decide) */
    }
    if (!menu_taking_input()) return buttons;
    const uint64_t hovered = G64(menu + MENU_HOVER);
    if (hovered == wasm_btn && wasm_btn) {   /* never decided by the game: the link opens here */
        open_url("https://wasm.rip");
        swallowing = 1;
        return buttons & ~(uint64_t)PAD_A;
    }
    if (hovered == ported_btn && ported_btn) {
        if (!has_save) {   /* (New Game would start a game: no window) */
            open_url("https://github.com/NotRexed");
            swallowing = 1;
            return buttons & ~(uint64_t)PAD_A;
        }
        /* New Game asks for the window: the cursor goes there (as fast as the menu takes it) and A is pressed */
        window_wanted = 1;
        swallowing = 1;
        for (int i = ported_index; i > newgame_index; i--) {
            inject(PAD_UP, 34);
            inject(0, 34);
        }
        inject(PAD_A, 50);
        inject(0, 34);
        return buttons & ~(uint64_t)PAD_A;
    }
    return buttons;
}
