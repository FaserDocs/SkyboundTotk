#pragma once
#include <emscripten.h>
#include <stdint.h>
static inline uint64_t __rdtsc(void) { return (uint64_t)(emscripten_get_now() * 1e6); }
