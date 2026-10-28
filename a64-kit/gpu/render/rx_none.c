/* The renderer's API when the build has no WebGPU (build.py native without --wgpu): nothing is drawn and the
   window shows the window buffers' memory (hos/display.c). */
#include "rx.h"

int rx_start(void *hwnd) { (void)hwnd; return 0; }
int rx_active(void) { return 0; }
void rx_resize(int width, int height) { (void)width; (void)height; }
void *rx_channel_new(void) { return 0; }
const uint8_t *rx_3d_methods(void) { return 0; }
void rx_3d_method(void *channel, const uint32_t *regs, uint32_t method, uint32_t arg) {
    (void)channel; (void)regs; (void)method; (void)arg;
}
void rx_flush(void) {}
void rx_compute_launch(void *channel, const uint32_t *regs) { (void)channel; (void)regs; }
void rx_begin_submission(void) {}
void rx_memory_written(uint64_t gpu_va, uint64_t size) { (void)gpu_va; (void)size; }
int rx_present(uint64_t guest_address, uint32_t width, uint32_t height) {
    (void)guest_address; (void)width; (void)height;
    return 0;
}
const char *rx_describe(void) { return "off"; }
void rx_stats(int64_t *draws, int64_t *skipped, int64_t *pipelines) { *draws = *skipped = *pipelines = 0; }
