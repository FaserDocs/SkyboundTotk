/* The browser build's display: the page shows the renderer's frames (hos/web/library.js); no window of our own.
   The renderer starts on its own thread (gpu.c's render thread: WebGPU objects belong to the worker that made
   them). Mouse motion: the page adds pointer-lock movement to web_mouse[] (dx, dy, captured). Controller: the page
   polls the Gamepad API (standard mapping) into web_pad[]: connected, XInput button bits, triggers 0..255, sticks
   LX LY RX RY (XInput's signs: up positive), a packet count. */
#include "hos.h"
#include <emscripten.h>
#include <xinput.h>

volatile int32_t web_mouse[3];
EMSCRIPTEN_KEEPALIVE uintptr_t web_mouse_address(void) { return (uintptr_t)web_mouse; }

void hos_display_start(void) {}

/* The canvas's size on the page in device pixels (the page's ResizeObserver): the renderer sizes its surface, and the
   canvas's backing store with it, to that (the frame drawn into it with its aspect kept, upscaled with FSR 1) */
void rx_resize(int width, int height);
EMSCRIPTEN_KEEPALIVE void web_resize(int width, int height) { rx_resize(width, height); }
/* F7-F10 on the page: the present settings (rx_present_fx.cpp), logged */
const char *rx_present_hotkey(int key);
EMSCRIPTEN_KEEPALIVE void web_present_hotkey(int key) { hos_log("present: %s", rx_present_hotkey(key)); }

int hos_mouse_take(int32_t *dx, int32_t *dy) {
    *dx = __atomic_exchange_n(&web_mouse[0], 0, __ATOMIC_SEQ_CST);
    *dy = __atomic_exchange_n(&web_mouse[1], 0, __ATOMIC_SEQ_CST);
    return __atomic_load_n(&web_mouse[2], __ATOMIC_SEQ_CST);
}

volatile int32_t web_pad[9];
EMSCRIPTEN_KEEPALIVE uintptr_t web_pad_address(void) { return (uintptr_t)web_pad; }

DWORD XInputGetState(DWORD i, XINPUT_STATE *s) {
    if (i != 0 || !__atomic_load_n(&web_pad[0], __ATOMIC_ACQUIRE)) return ERROR_DEVICE_NOT_CONNECTED;
    s->dwPacketNumber = (DWORD)web_pad[8];
    s->Gamepad.wButtons = (WORD)web_pad[1];
    s->Gamepad.bLeftTrigger = (BYTE)web_pad[2];
    s->Gamepad.bRightTrigger = (BYTE)web_pad[3];
    s->Gamepad.sThumbLX = (SHORT)web_pad[4];
    s->Gamepad.sThumbLY = (SHORT)web_pad[5];
    s->Gamepad.sThumbRX = (SHORT)web_pad[6];
    s->Gamepad.sThumbRY = (SHORT)web_pad[7];
    return ERROR_SUCCESS;
}
