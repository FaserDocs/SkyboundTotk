/* The renderer: the Maxwell 3D engine's work drawn with WebGPU (webgpu.h: wgpu-native on the PC, the browser's
   WebGPU on the web). hos/gpu.c runs the command streams and hands the renderer the 3D methods that draw; the
   renderer reads the engine's registers (suyu's Maxwell3D::Regs layout) and guest memory, and turns them into
   WebGPU passes, pipelines and draws. hos/display.c gives it the window; nn_vi.c tells it what to present.

   Threads: every entry point may be called from any thread; the renderer serializes them. */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The window to present into (a Windows HWND), from the window's thread once it exists; null: no window (the
   renderer draws offscreen). Returns 1 when the renderer runs, 0 when the window should show frames itself. */
int rx_start(void *hwnd);
int rx_active(void);
/* The window's client area changed size. */
void rx_resize(int width, int height);

/* One renderer state per GPU channel (its constant buffer bindings and the draw being assembled). */
void *rx_channel_new(void);
/* the channel's ZPASS_PIXEL_COUNT estimate (occlusion query reports) */
uint64_t rx_channel_zpass(void *channel);
/* The 3D methods the renderer acts on: a table of 0x1000 flags (hos/gpu.c calls rx_3d_method for those only).
   0xFFFF (not a method): a macro finished. */
const uint8_t *rx_3d_methods(void);
/* A 3D method was written: regs is the channel's 3D register file after the write (regs[method] == arg). */
void rx_3d_method(void *channel, const uint32_t *regs, uint32_t method, uint32_t arg);
/* The compute engine (B1C0) was launched: regs is its register file (the launch descriptor at regs[0xAD] << 8). */
void rx_compute_launch(void *channel, const uint32_t *regs);
/* A submission ended: what was recorded goes to the GPU. */
void rx_flush(void);
/* A submission starts: memory the CPU changed since the last one is uploaded again when used. */
void rx_begin_submission(void);
/* The same, split for a GPU thread: the pages the CPU wrote are collected by the thread that submits (at its submit,
   in submission order: a malloc'd array of guest page numbers, or null with *count -1 when nothing is tracked), and
   applied on the GPU thread when the submission starts (the array is freed there). */
uint32_t *rx_collect_written(int64_t *count);
void rx_begin_submission_written(uint32_t *pages, int64_t count);
/* The command processor is about to run guest memory as commands (an indirect draw's arguments, say): 1 when a
   shader wrote part of it on the GPU and the result is not in guest memory yet. rx_sync_gpu_writes puts it there
   (it submits what was recorded and waits for the GPU). */
int rx_gpu_writes_pending(uint64_t gpu_va, uint64_t size);
/* The same for dispatch-written memory only, without the renderer's lock (a hint: split mode's parsing thread). */
int rx_gpu_written_hint(uint64_t gpu_va, uint64_t size);
void rx_sync_gpu_writes(uint64_t gpu_va, uint64_t size);
/* The command processor wrote guest memory (inline uploads, copies) while running a submission. */
void rx_memory_written(uint64_t gpu_va, uint64_t size);

/* The game queued a window buffer for display: guest address, size in pixels. Returns 1 when the renderer drew
   that buffer and presented it. */
int rx_present(uint64_t guest_address, uint32_t width, uint32_t height);
/* What the window title shows: "<adapter> (<backend>)". */
const char *rx_describe(void);
/* Draws done / skipped (unsupported state) since the start. */
void rx_stats(int64_t *draws, int64_t *skipped, int64_t *pipelines);

#ifdef __cplusplus
}
#endif
