/* SDL3 window, keyboard, mouse and gamepads. The guest runs on another thread; this thread owns the window. */
#include "hos.h"
#include "rx_window.h"
#include "../gpu/render/rx.h"
#include <SDL3/SDL.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

extern volatile uint8_t web_keys[256];
void linux_set_focused(int on);

static SDL_Window *window;
static RxLinuxWindow rxwin;
static int display_up;
static volatile int guest_done;
static volatile LONG mouse_dx, mouse_dy, mouse_captured;

int hos_mouse_take(int32_t *dx, int32_t *dy) {
    *dx = InterlockedExchange(&mouse_dx, 0);
    *dy = InterlockedExchange(&mouse_dy, 0);
    return mouse_captured;
}

static void sync_keys(void) {
    int n = 0;
    const bool *st = SDL_GetKeyboardState(&n);
    if (!st) return;
    memset((void *)web_keys, 0, sizeof web_keys);
    for (int c = 'A'; c <= 'Z'; c++) {
        const SDL_Scancode sc = SDL_GetScancodeFromKey((SDL_Keycode)(c - 'A' + 'a'), NULL);
        if (sc >= 0 && sc < n) web_keys[c] = st[sc] ? 1 : 0;
    }
    struct { int vk; SDL_Scancode sc; } map[] = {
        {VK_SPACE, SDL_SCANCODE_SPACE}, {VK_RETURN, SDL_SCANCODE_RETURN}, {VK_TAB, SDL_SCANCODE_TAB},
        {VK_ESCAPE, SDL_SCANCODE_ESCAPE}, {VK_LEFT, SDL_SCANCODE_LEFT}, {VK_RIGHT, SDL_SCANCODE_RIGHT},
        {VK_UP, SDL_SCANCODE_UP}, {VK_DOWN, SDL_SCANCODE_DOWN}, {VK_OEM_MINUS, SDL_SCANCODE_MINUS},
        {VK_OEM_PLUS, SDL_SCANCODE_EQUALS}, {VK_F1, SDL_SCANCODE_F1},
    };
    for (unsigned i = 0; i < sizeof map / sizeof *map; i++)
        if (map[i].sc < n) web_keys[map[i].vk] = st[map[i].sc] ? 1 : 0;
    web_keys[VK_SHIFT] = (st[SDL_SCANCODE_LSHIFT] || st[SDL_SCANCODE_RSHIFT]) ? 1 : 0;
    web_keys[VK_CONTROL] = (st[SDL_SCANCODE_LCTRL] || st[SDL_SCANCODE_RCTRL]) ? 1 : 0;
    web_keys[VK_MENU] = (st[SDL_SCANCODE_LALT] || st[SDL_SCANCODE_RALT]) ? 1 : 0;
    web_keys[VK_SUBTRACT] = st[SDL_SCANCODE_KP_MINUS] ? 1 : 0;
    web_keys[VK_ADD] = st[SDL_SCANCODE_KP_PLUS] ? 1 : 0;
    for (int f = 0; f < 10; f++)
        if (SDL_SCANCODE_F1 + f < n) web_keys[VK_F1 + f] = st[SDL_SCANCODE_F1 + f] ? 1 : 0;
    /* digits (1-4: the d-pad) and the mouse buttons (left Y, right ZL while captured: nn_hid.c) */
    for (int d = 1; d <= 9; d++)
        if (SDL_SCANCODE_1 + d - 1 < n) web_keys['0' + d] = st[SDL_SCANCODE_1 + d - 1] ? 1 : 0;
    if (SDL_SCANCODE_0 < n) web_keys['0'] = st[SDL_SCANCODE_0] ? 1 : 0;
    const SDL_MouseButtonFlags mb = SDL_GetMouseState(NULL, NULL);
    web_keys[VK_LBUTTON] = (mb & SDL_BUTTON_LMASK) ? 1 : 0;
    web_keys[VK_RBUTTON] = (mb & SDL_BUTTON_RMASK) ? 1 : 0;
}

static void refresh_size(void) {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    rxwin.width = w;
    rxwin.height = h;
}

static int fill_surface(void) {
    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    void *wl_display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);
    void *wl_surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
    void *xdisplay = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
    const uint64_t xwindow = (uint64_t)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
    refresh_size();
    if (wl_display && wl_surface) {
        rxwin.kind = 1;
        rxwin.display = wl_display;
        rxwin.wl_surface = wl_surface;
        return 1;
    }
    if (xdisplay && xwindow) {
        rxwin.kind = 2;
        rxwin.display = xdisplay;
        rxwin.xwindow = xwindow;
        return 1;
    }
    return 0;
}

const char *rx_present_hotkey(int key);

static void on_key(const SDL_KeyboardEvent *k) {
    if (k->down == false || k->repeat) return;
    if (k->key == SDLK_ESCAPE) {
        mouse_captured = 0;
        SDL_SetWindowRelativeMouseMode(window, false);
    } else if (k->key == SDLK_F11) {
        const bool full = (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
        SDL_SetWindowFullscreen(window, !full);
    } else if (k->key >= SDLK_F6 && k->key <= SDLK_F10) {
        const char *msg = rx_present_hotkey((int)(k->key - SDLK_F6) + 6);
        hos_log("present: %s", msg);
        if (k->key == SDLK_F6) {
            refresh_size();
            rx_resize(rxwin.width, rxwin.height);
        }
        SDL_SetWindowTitle(window, msg);
    }
}

static void pump(void) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            hos_log("display: window closed, exiting");
            hos_log_flush();
            _exit(0);
        case SDL_EVENT_KEY_DOWN:
            on_key(&e.key);
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (e.button.button == SDL_BUTTON_LEFT && !mouse_captured) {
                mouse_captured = 1;
                SDL_SetWindowRelativeMouseMode(window, true);
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (mouse_captured) {
                InterlockedExchangeAdd(&mouse_dx, (LONG)e.motion.xrel);
                InterlockedExchangeAdd(&mouse_dy, (LONG)e.motion.yrel);
            }
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED:
            refresh_size();
            rx_resize(rxwin.width, rxwin.height);
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
            linux_set_focused(1);
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            linux_set_focused(0);
            mouse_captured = 0;
            SDL_SetWindowRelativeMouseMode(window, false);
            break;
        default:
            break;
        }
    }
    sync_keys();
}

/* ---- gamepads (SDL3), same button positions as hos/pad_sdl.c ---- */
enum { H_A = 0, H_B, H_X, H_Y, H_LS, H_RS, H_L, H_R, H_ZL, H_ZR, H_PLUS, H_MINUS, H_LEFT, H_UP, H_RIGHT, H_DOWN };
#define MAX_PADS 8
static SDL_Gamepad *pads[MAX_PADS];
static SDL_JoystickID pad_ids[MAX_PADS];
static SRWLOCK state_lock = SRWLOCK_INIT;
static uint64_t state_buttons;
static int32_t state_stick[4];
static int state_count;
static volatile LONG rumble_low, rumble_high;

static int32_t stick_value(int16_t v, int flip) {
    int32_t a = v < -32767 ? -32767 : v;
    if (flip) a = -a;
    return (a > 7849 || a < -7849) ? a : 0;
}

static DWORD WINAPI pad_main(void *arg) {
    (void)arg;
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        hos_log("hid: SDL gamepad init failed: %s", SDL_GetError());
        return 0;
    }
    hos_log("hid: SDL3 controller support ready");
    int64_t next_scan = 0;
    uint16_t last_lo = 0, last_hi = 0;
    for (;;) {
        SDL_UpdateGamepads();
        const int64_t now = hos_now_ns();
        if (now >= next_scan) {
            next_scan = now + 1000000000LL;
            for (int i = 0; i < MAX_PADS; i++)
                if (pads[i] && !SDL_GamepadConnected(pads[i])) {
                    hos_log("hid: controller \"%s\" disconnected", SDL_GetGamepadName(pads[i]));
                    SDL_CloseGamepad(pads[i]);
                    pads[i] = NULL;
                }
            int count = 0;
            SDL_JoystickID *ids = SDL_GetGamepads(&count);
            for (int j = 0; ids && j < count; j++) {
                int have = 0, slot = -1;
                for (int i = 0; i < MAX_PADS; i++) {
                    if (pads[i] && pad_ids[i] == ids[j]) have = 1;
                    if (!pads[i] && slot < 0) slot = i;
                }
                if (have || slot < 0) continue;
                SDL_Gamepad *gc = SDL_OpenGamepad(ids[j]);
                if (!gc) continue;
                pads[slot] = gc;
                pad_ids[slot] = ids[j];
                hos_log("hid: controller \"%s\" connected", SDL_GetGamepadName(gc));
            }
            SDL_free(ids);
        }
        static const struct { SDL_GamepadButton sdl; int hid; } map[] = {
            {SDL_GAMEPAD_BUTTON_SOUTH, H_B}, {SDL_GAMEPAD_BUTTON_EAST, H_A},
            {SDL_GAMEPAD_BUTTON_WEST, H_Y}, {SDL_GAMEPAD_BUTTON_NORTH, H_X},
            {SDL_GAMEPAD_BUTTON_LEFT_STICK, H_LS}, {SDL_GAMEPAD_BUTTON_RIGHT_STICK, H_RS},
            {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, H_L}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, H_R},
            {SDL_GAMEPAD_BUTTON_START, H_PLUS}, {SDL_GAMEPAD_BUTTON_BACK, H_MINUS},
            {SDL_GAMEPAD_BUTTON_DPAD_LEFT, H_LEFT}, {SDL_GAMEPAD_BUTTON_DPAD_UP, H_UP},
            {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, H_RIGHT}, {SDL_GAMEPAD_BUTTON_DPAD_DOWN, H_DOWN}};
        uint64_t b = 0;
        int32_t s[4] = {0};
        int count = 0;
        const uint16_t lo = (uint16_t)rumble_low, hi = (uint16_t)rumble_high;
        for (int i = 0; i < MAX_PADS; i++) {
            SDL_Gamepad *gc = pads[i];
            if (!gc) continue;
            count++;
            for (unsigned k = 0; k < sizeof map / sizeof *map; k++)
                if (SDL_GetGamepadButton(gc, map[k].sdl)) b |= 1ULL << map[k].hid;
            if (SDL_GetGamepadAxis(gc, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 8000) b |= 1ULL << H_ZL;
            if (SDL_GetGamepadAxis(gc, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 8000) b |= 1ULL << H_ZR;
            const int32_t v[4] = {
                stick_value(SDL_GetGamepadAxis(gc, SDL_GAMEPAD_AXIS_LEFTX), 0),
                stick_value(SDL_GetGamepadAxis(gc, SDL_GAMEPAD_AXIS_LEFTY), 1),
                stick_value(SDL_GetGamepadAxis(gc, SDL_GAMEPAD_AXIS_RIGHTX), 0),
                stick_value(SDL_GetGamepadAxis(gc, SDL_GAMEPAD_AXIS_RIGHTY), 1)};
            for (int k = 0; k < 4; k++)
                if ((v[k] < 0 ? -v[k] : v[k]) > (s[k] < 0 ? -s[k] : s[k])) s[k] = v[k];
            if (lo != last_lo || hi != last_hi) SDL_RumbleGamepad(gc, lo, hi, lo || hi ? 1000 : 0);
        }
        last_lo = lo;
        last_hi = hi;
        AcquireSRWLockExclusive(&state_lock);
        state_buttons = b;
        memcpy(state_stick, s, sizeof s);
        state_count = count;
        ReleaseSRWLockExclusive(&state_lock);
        Sleep(4);
    }
}

int hos_sdl_pads(uint64_t *buttons, int32_t stick[4]) {
    static volatile LONG started;
    if (getenv("HOS_NO_SDL")) return 0;
    if (!started && !InterlockedExchange(&started, 1)) CreateThread(NULL, 0, pad_main, NULL, 0, NULL);
    AcquireSRWLockShared(&state_lock);
    const int n = state_count;
    *buttons |= state_buttons;
    for (int k = 0; k < 4; k++)
        if ((state_stick[k] < 0 ? -state_stick[k] : state_stick[k]) > (stick[k] < 0 ? -stick[k] : stick[k]))
            stick[k] = state_stick[k];
    ReleaseSRWLockShared(&state_lock);
    return n;
}
void hos_sdl_rumble(float low, float high) {
    if (low < 0) low = 0;
    if (low > 1) low = 1;
    if (high < 0) high = 0;
    if (high > 1) high = 1;
    InterlockedExchange(&rumble_low, (LONG)(low * 65535));
    InterlockedExchange(&rumble_high, (LONG)(high * 65535));
}

static void *boot_tramp(void *p) {
    void (*fn)(void) = p;
    fn();
    guest_done = 1;
    return NULL;
}

void linux_display_enter(void (*boot)(void)) {
    const int headless = getenv("HOS_HEADLESS") != NULL;
    if (!headless) {
        if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)) {
            fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
            exit(1);
        }
        int win_w = 1280, win_h = 720;
        if (getenv("HOS_WINDOW_SIZE")) {
            int w = 0, h = 0;
            if (sscanf(getenv("HOS_WINDOW_SIZE"), "%dx%d", &w, &h) == 2 && w >= 64 && h >= 64) win_w = w, win_h = h;
        }
        window = SDL_CreateWindow("Tears of the Kingdom", win_w, win_h,
                                  SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_VULKAN);
        if (!window) {
            fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
            exit(1);
        }
        if (getenv("HOS_FULLSCREEN") && atoi(getenv("HOS_FULLSCREEN"))) SDL_SetWindowFullscreen(window, true);
        if (!fill_surface()) {
            fprintf(stderr, "window has no Wayland or X11 surface\n");
            exit(1);
        }
        if (!rx_start(&rxwin)) hos_log("display: renderer did not start; the window will stay blank");
        display_up = 1;
        pump();
    }
    pthread_t th;
    pthread_attr_t a;
    pthread_attr_init(&a);
    if (pthread_attr_setstacksize(&a, 256u << 20))
        hos_log("display: 256 MB boot stack was refused; the guest runs on the default stack");
    if (pthread_create(&th, &a, boot_tramp, boot)) {
        fprintf(stderr, "cannot start the guest thread\n");
        exit(1);
    }
    pthread_attr_destroy(&a);
    if (headless) {
        pthread_join(th, NULL);
        return;
    }
    while (!guest_done) {
        pump();
        SDL_Delay(4);
    }
}

void hos_display_start(void) {
    if (getenv("HOS_HEADLESS")) {
        rx_start(NULL);
        return;
    }
    if (display_up) return;
    fprintf(stderr, "hos_display_start called before the SDL window existed\n");
}
