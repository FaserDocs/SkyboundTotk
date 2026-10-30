/* Opaque window handed to rx_start on Linux. kind 1 is Wayland, kind 2 is Xlib. */
#pragma once
#include <stdint.h>
typedef struct RxLinuxWindow {
    int kind;
    void *display;
    void *wl_surface;
    uint64_t xwindow;
    int width, height;
} RxLinuxWindow;
