// The threaded web build (RX_WEB_MT, build.py RX_WEB_MT=1): in the browser a WebGPU object belongs to the worker
// that made it, so one thread, the WebGPU thread (rx_web_mt.cpp), makes every WebGPU call. The renderer's own thread
// prepares draws and records them (deferred encoding, as natively), and the WebGPU thread replays, submits and presents
// them, as the native submit thread does. Every other WebGPU call the renderer makes goes through these macros: on the
// WebGPU thread it is made at once, on any other it runs there while the caller waits. Made by
// scripts/web_mt_macros.py from the calls in gpu/render: a call not listed here would run on the wrong thread.
#pragma once
#if RX_WEB_MT
#include <atomic>
#include <type_traits>

namespace rx {
bool on_gpu_thread();
void run_on_gpu_thread(void (*fn)(void*), void* arg, const char* what);   // (and waits for it; RX_MT_LOG: counted)
// Creations (Now): ahead of the calls and batches queued before them, and run between a batch's commands too, as they
// depend on none of those and the caller would otherwise wait for the whole batch the WebGPU thread is running
void run_on_gpu_thread_now(void (*fn)(void*), void* arg, const char* what);
extern std::atomic<int> gpu_calls_now;   // (how many wait)
void serve_gpu_calls_now();
/// (the WebGPU thread, between the commands of a batch)
inline void serve_gpu_calls_now_if_any() {
    if (gpu_calls_now.load(std::memory_order_relaxed)) serve_gpu_calls_now();
}
template <bool Now = false, class F>
auto on_gpu(const char* what, F&& f) -> decltype(f()) {
    if (on_gpu_thread()) return f();
    using R = decltype(f());
    using Fn = std::remove_reference_t<F>;
    auto* const run = Now ? run_on_gpu_thread_now : run_on_gpu_thread;
    if constexpr (std::is_void_v<R>) {
        run([](void* p) { (*static_cast<Fn*>(p))(); }, &f, what);
    } else {
        struct Call {
            Fn* f;
            R r;
        } c{&f, R{}};
        run([](void* p) { auto* c = static_cast<Call*>(p); c->r = (*c->f)(); }, &c, what);
        return c.r;
    }
}
// Calls that need no answer go without waiting, in order with the others (they run before the batches flushed after
// them): reference counting, and queue writes (their data copied)
void post_to_gpu_thread(void (*fn)(void*), void* arg);
void mt_note_wait(const char* what, int64_t ns);   // (RX_MT_LOG: other waits counted with the calls)
template <auto Fn, class H>
void gpu_async_handle(H h) {
    if (on_gpu_thread()) return Fn(h);
    post_to_gpu_thread([](void* p) { Fn(static_cast<H>(p)); }, static_cast<void*>(h));
}
void mt_queue_write_buffer(WGPUQueue q, WGPUBuffer b, uint64_t offset, const void* data, size_t size);
void mt_queue_write_texture(WGPUQueue q, const WGPUTexelCopyTextureInfo* dst, const void* data, size_t size,
                            const WGPUTexelCopyBufferLayout* layout, const WGPUExtent3D* extent);
// Buffer sizes from their creation (the renderer asks ~1000 times a frame)
WGPUBuffer mt_create_buffer(WGPUDevice d, const WGPUBufferDescriptor* desc);
uint64_t mt_buffer_size(WGPUBuffer b);
WGPUTexture mt_create_texture(WGPUDevice d, const WGPUTextureDescriptor* desc);
// Bind groups are made without waiting: the handle the renderer gets is a stand-in (rx_web_mt.cpp BindGroupProxy)
// counting its own references, which the WebGPU thread fills before anything queued after it runs, and the calls that
// take one get the group itself (mt_bind_group) there
WGPUBindGroup mt_create_bind_group(WGPUDevice d, const WGPUBindGroupDescriptor* desc);
void mt_bind_group_add_ref(WGPUBindGroup g);
void mt_bind_group_release(WGPUBindGroup g);
WGPUBindGroup mt_bind_group(WGPUBindGroup g);   // (the WebGPU thread)
}  // namespace rx

#define wgpuAdapterGetInfo(...) ::rx::on_gpu("wgpuAdapterGetInfo", [&] { return wgpuAdapterGetInfo(__VA_ARGS__); })
#define wgpuAdapterGetLimits(...) ::rx::on_gpu("wgpuAdapterGetLimits", [&] { return wgpuAdapterGetLimits(__VA_ARGS__); })
#define wgpuAdapterHasFeature(...) ::rx::on_gpu("wgpuAdapterHasFeature", [&] { return wgpuAdapterHasFeature(__VA_ARGS__); })
#define wgpuAdapterInfoFreeMembers(...) ::rx::on_gpu("wgpuAdapterInfoFreeMembers", [&] { return wgpuAdapterInfoFreeMembers(__VA_ARGS__); })
#define wgpuAdapterRequestDevice(...) ::rx::on_gpu("wgpuAdapterRequestDevice", [&] { return wgpuAdapterRequestDevice(__VA_ARGS__); })
#define wgpuBindGroupAddRef(...) ::rx::mt_bind_group_add_ref(__VA_ARGS__)
#define wgpuBindGroupLayoutRelease(...) ::rx::gpu_async_handle<&wgpuBindGroupLayoutRelease>(__VA_ARGS__)
#define wgpuBindGroupRelease(...) ::rx::mt_bind_group_release(__VA_ARGS__)
#define wgpuBufferAddRef(...) ::rx::gpu_async_handle<&wgpuBufferAddRef>(__VA_ARGS__)
#define wgpuBufferGetConstMappedRange(...) ::rx::on_gpu("wgpuBufferGetConstMappedRange", [&] { return wgpuBufferGetConstMappedRange(__VA_ARGS__); })
#define wgpuBufferGetSize(...) ::rx::mt_buffer_size(__VA_ARGS__)
#define wgpuBufferMapAsync(...) ::rx::on_gpu("wgpuBufferMapAsync", [&] { return wgpuBufferMapAsync(__VA_ARGS__); })
#define wgpuBufferRelease(...) ::rx::gpu_async_handle<&wgpuBufferRelease>(__VA_ARGS__)
#define wgpuBufferUnmap(...) ::rx::on_gpu("wgpuBufferUnmap", [&] { return wgpuBufferUnmap(__VA_ARGS__); })
#define wgpuCommandBufferRelease(...) ::rx::gpu_async_handle<&wgpuCommandBufferRelease>(__VA_ARGS__)
#define wgpuCommandEncoderBeginComputePass(...) ::rx::on_gpu("wgpuCommandEncoderBeginComputePass", [&] { return wgpuCommandEncoderBeginComputePass(__VA_ARGS__); })
#define wgpuCommandEncoderBeginRenderPass(...) ::rx::on_gpu("wgpuCommandEncoderBeginRenderPass", [&] { return wgpuCommandEncoderBeginRenderPass(__VA_ARGS__); })
#define wgpuCommandEncoderCopyBufferToBuffer(...) ::rx::on_gpu("wgpuCommandEncoderCopyBufferToBuffer", [&] { return wgpuCommandEncoderCopyBufferToBuffer(__VA_ARGS__); })
#define wgpuCommandEncoderCopyBufferToTexture(...) ::rx::on_gpu("wgpuCommandEncoderCopyBufferToTexture", [&] { return wgpuCommandEncoderCopyBufferToTexture(__VA_ARGS__); })
#define wgpuCommandEncoderCopyTextureToBuffer(...) ::rx::on_gpu("wgpuCommandEncoderCopyTextureToBuffer", [&] { return wgpuCommandEncoderCopyTextureToBuffer(__VA_ARGS__); })
#define wgpuCommandEncoderCopyTextureToTexture(...) ::rx::on_gpu("wgpuCommandEncoderCopyTextureToTexture", [&] { return wgpuCommandEncoderCopyTextureToTexture(__VA_ARGS__); })
#define wgpuCommandEncoderFinish(...) ::rx::on_gpu("wgpuCommandEncoderFinish", [&] { return wgpuCommandEncoderFinish(__VA_ARGS__); })
#define wgpuCommandEncoderRelease(...) ::rx::gpu_async_handle<&wgpuCommandEncoderRelease>(__VA_ARGS__)
#define wgpuCommandEncoderResolveQuerySet(...) ::rx::on_gpu("wgpuCommandEncoderResolveQuerySet", [&] { return wgpuCommandEncoderResolveQuerySet(__VA_ARGS__); })
#define wgpuComputePassEncoderDispatchWorkgroups(...) ::rx::on_gpu("wgpuComputePassEncoderDispatchWorkgroups", [&] { return wgpuComputePassEncoderDispatchWorkgroups(__VA_ARGS__); })
#define wgpuComputePassEncoderEnd(...) ::rx::on_gpu("wgpuComputePassEncoderEnd", [&] { return wgpuComputePassEncoderEnd(__VA_ARGS__); })
#define wgpuComputePassEncoderRelease(...) ::rx::gpu_async_handle<&wgpuComputePassEncoderRelease>(__VA_ARGS__)
#define wgpuComputePassEncoderSetBindGroup(p, i, g, n, o) ::rx::on_gpu("wgpuComputePassEncoderSetBindGroup", [&] { return wgpuComputePassEncoderSetBindGroup(p, i, ::rx::mt_bind_group(g), n, o); })
#define wgpuComputePassEncoderSetPipeline(...) ::rx::on_gpu("wgpuComputePassEncoderSetPipeline", [&] { return wgpuComputePassEncoderSetPipeline(__VA_ARGS__); })
#define wgpuComputePipelineAddRef(...) ::rx::gpu_async_handle<&wgpuComputePipelineAddRef>(__VA_ARGS__)
#define wgpuComputePipelineGetBindGroupLayout(...) ::rx::on_gpu("wgpuComputePipelineGetBindGroupLayout", [&] { return wgpuComputePipelineGetBindGroupLayout(__VA_ARGS__); })
#define wgpuComputePipelineRelease(...) ::rx::gpu_async_handle<&wgpuComputePipelineRelease>(__VA_ARGS__)
#define wgpuCreateInstance(...) ::rx::on_gpu("wgpuCreateInstance", [&] { return wgpuCreateInstance(__VA_ARGS__); })
#define wgpuDeviceCreateBindGroup(...) ::rx::mt_create_bind_group(__VA_ARGS__)
#define wgpuDeviceCreateBindGroupLayout(...) ::rx::on_gpu<true>("wgpuDeviceCreateBindGroupLayout", [&] { return wgpuDeviceCreateBindGroupLayout(__VA_ARGS__); })
#define wgpuDeviceCreateBuffer(...) ::rx::mt_create_buffer(__VA_ARGS__)
#define wgpuDeviceCreateCommandEncoder(...) ::rx::on_gpu("wgpuDeviceCreateCommandEncoder", [&] { return wgpuDeviceCreateCommandEncoder(__VA_ARGS__); })
#define wgpuDeviceCreateComputePipeline(...) ::rx::on_gpu<true>("wgpuDeviceCreateComputePipeline", [&] { return wgpuDeviceCreateComputePipeline(__VA_ARGS__); })
#define wgpuDeviceCreateComputePipelineAsync(...) ::rx::on_gpu<true>("wgpuDeviceCreateComputePipelineAsync", [&] { return wgpuDeviceCreateComputePipelineAsync(__VA_ARGS__); })
#define wgpuDeviceCreatePipelineLayout(...) ::rx::on_gpu<true>("wgpuDeviceCreatePipelineLayout", [&] { return wgpuDeviceCreatePipelineLayout(__VA_ARGS__); })
#define wgpuDeviceCreateQuerySet(...) ::rx::on_gpu<true>("wgpuDeviceCreateQuerySet", [&] { return wgpuDeviceCreateQuerySet(__VA_ARGS__); })
#define wgpuDeviceCreateRenderPipeline(...) ::rx::on_gpu<true>("wgpuDeviceCreateRenderPipeline", [&] { return wgpuDeviceCreateRenderPipeline(__VA_ARGS__); })
#define wgpuDeviceCreateRenderPipelineAsync(...) ::rx::on_gpu<true>("wgpuDeviceCreateRenderPipelineAsync", [&] { return wgpuDeviceCreateRenderPipelineAsync(__VA_ARGS__); })
#define wgpuDeviceCreateSampler(...) ::rx::on_gpu<true>("wgpuDeviceCreateSampler", [&] { return wgpuDeviceCreateSampler(__VA_ARGS__); })
#define wgpuDeviceCreateShaderModule(...) ::rx::on_gpu<true>("wgpuDeviceCreateShaderModule", [&] { return wgpuDeviceCreateShaderModule(__VA_ARGS__); })
#define wgpuDeviceCreateShaderModuleTrusted(...) ::rx::on_gpu<true>("wgpuDeviceCreateShaderModuleTrusted", [&] { return wgpuDeviceCreateShaderModuleTrusted(__VA_ARGS__); })
#define wgpuDeviceCreateTexture(...) ::rx::mt_create_texture(__VA_ARGS__)
#define wgpuDeviceGetLimits(...) ::rx::on_gpu("wgpuDeviceGetLimits", [&] { return wgpuDeviceGetLimits(__VA_ARGS__); })
#define wgpuDeviceGetQueue(...) ::rx::on_gpu("wgpuDeviceGetQueue", [&] { return wgpuDeviceGetQueue(__VA_ARGS__); })
#define wgpuDevicePopErrorScope(...) ::rx::on_gpu("wgpuDevicePopErrorScope", [&] { return wgpuDevicePopErrorScope(__VA_ARGS__); })
#define wgpuDevicePushErrorScope(...) ::rx::on_gpu("wgpuDevicePushErrorScope", [&] { return wgpuDevicePushErrorScope(__VA_ARGS__); })
#define wgpuInstanceCreateSurface(...) ::rx::on_gpu("wgpuInstanceCreateSurface", [&] { return wgpuInstanceCreateSurface(__VA_ARGS__); })
#define wgpuInstanceProcessEvents(...) (::rx::on_gpu_thread() ? wgpuInstanceProcessEvents(__VA_ARGS__) : (void)0)
#define wgpuInstanceRequestAdapter(...) ::rx::on_gpu("wgpuInstanceRequestAdapter", [&] { return wgpuInstanceRequestAdapter(__VA_ARGS__); })
#define wgpuPipelineLayoutRelease(...) ::rx::gpu_async_handle<&wgpuPipelineLayoutRelease>(__VA_ARGS__)
#define wgpuQueueGetTimestampPeriod(...) ::rx::on_gpu("wgpuQueueGetTimestampPeriod", [&] { return wgpuQueueGetTimestampPeriod(__VA_ARGS__); })
#define wgpuQueueOnSubmittedWorkDone(...) ::rx::on_gpu("wgpuQueueOnSubmittedWorkDone", [&] { return wgpuQueueOnSubmittedWorkDone(__VA_ARGS__); })
#define wgpuQueueSubmit(...) ::rx::on_gpu("wgpuQueueSubmit", [&] { return wgpuQueueSubmit(__VA_ARGS__); })
#define wgpuQueueSubmitForIndex(...) ::rx::on_gpu("wgpuQueueSubmitForIndex", [&] { return wgpuQueueSubmitForIndex(__VA_ARGS__); })
#define wgpuQueueWriteBuffer(...) ::rx::mt_queue_write_buffer(__VA_ARGS__)
#define wgpuQueueWriteTexture(...) ::rx::mt_queue_write_texture(__VA_ARGS__)
#define wgpuRenderPassEncoderDraw(...) ::rx::on_gpu("wgpuRenderPassEncoderDraw", [&] { return wgpuRenderPassEncoderDraw(__VA_ARGS__); })
#define wgpuRenderPassEncoderDrawIndexed(...) ::rx::on_gpu("wgpuRenderPassEncoderDrawIndexed", [&] { return wgpuRenderPassEncoderDrawIndexed(__VA_ARGS__); })
#define wgpuRenderPassEncoderEnd(...) ::rx::on_gpu("wgpuRenderPassEncoderEnd", [&] { return wgpuRenderPassEncoderEnd(__VA_ARGS__); })
#define wgpuRenderPassEncoderRelease(...) ::rx::gpu_async_handle<&wgpuRenderPassEncoderRelease>(__VA_ARGS__)
#define wgpuRenderPassEncoderSetBindGroup(p, i, g, n, o) ::rx::on_gpu("wgpuRenderPassEncoderSetBindGroup", [&] { return wgpuRenderPassEncoderSetBindGroup(p, i, ::rx::mt_bind_group(g), n, o); })
#define wgpuRenderPassEncoderSetBlendConstant(...) ::rx::on_gpu("wgpuRenderPassEncoderSetBlendConstant", [&] { return wgpuRenderPassEncoderSetBlendConstant(__VA_ARGS__); })
#define wgpuRenderPassEncoderSetIndexBuffer(...) ::rx::on_gpu("wgpuRenderPassEncoderSetIndexBuffer", [&] { return wgpuRenderPassEncoderSetIndexBuffer(__VA_ARGS__); })
#define wgpuRenderPassEncoderSetPipeline(...) ::rx::on_gpu("wgpuRenderPassEncoderSetPipeline", [&] { return wgpuRenderPassEncoderSetPipeline(__VA_ARGS__); })
#define wgpuRenderPassEncoderSetScissorRect(...) ::rx::on_gpu("wgpuRenderPassEncoderSetScissorRect", [&] { return wgpuRenderPassEncoderSetScissorRect(__VA_ARGS__); })
#define wgpuRenderPassEncoderSetStencilReference(...) ::rx::on_gpu("wgpuRenderPassEncoderSetStencilReference", [&] { return wgpuRenderPassEncoderSetStencilReference(__VA_ARGS__); })
#define wgpuRenderPassEncoderSetVertexBuffer(...) ::rx::on_gpu("wgpuRenderPassEncoderSetVertexBuffer", [&] { return wgpuRenderPassEncoderSetVertexBuffer(__VA_ARGS__); })
#define wgpuRenderPassEncoderSetViewport(...) ::rx::on_gpu("wgpuRenderPassEncoderSetViewport", [&] { return wgpuRenderPassEncoderSetViewport(__VA_ARGS__); })
#define wgpuRenderPipelineAddRef(...) ::rx::gpu_async_handle<&wgpuRenderPipelineAddRef>(__VA_ARGS__)
#define wgpuRenderPipelineRelease(...) ::rx::gpu_async_handle<&wgpuRenderPipelineRelease>(__VA_ARGS__)
#define wgpuShaderModuleRelease(...) ::rx::gpu_async_handle<&wgpuShaderModuleRelease>(__VA_ARGS__)
#define wgpuSurfaceCapabilitiesFreeMembers(...) ::rx::on_gpu("wgpuSurfaceCapabilitiesFreeMembers", [&] { return wgpuSurfaceCapabilitiesFreeMembers(__VA_ARGS__); })
#define wgpuSurfaceConfigure(...) ::rx::on_gpu("wgpuSurfaceConfigure", [&] { return wgpuSurfaceConfigure(__VA_ARGS__); })
#define wgpuSurfaceGetCapabilities(...) ::rx::on_gpu("wgpuSurfaceGetCapabilities", [&] { return wgpuSurfaceGetCapabilities(__VA_ARGS__); })
#define wgpuSurfaceGetCurrentTexture(...) ::rx::on_gpu("wgpuSurfaceGetCurrentTexture", [&] { return wgpuSurfaceGetCurrentTexture(__VA_ARGS__); })
#define wgpuSurfacePresent(...) ::rx::on_gpu("wgpuSurfacePresent", [&] { return wgpuSurfacePresent(__VA_ARGS__); })
#define wgpuTextureAddRef(...) ::rx::gpu_async_handle<&wgpuTextureAddRef>(__VA_ARGS__)
#define wgpuTextureCreateView(...) ::rx::on_gpu<true>("wgpuTextureCreateView", [&] { return wgpuTextureCreateView(__VA_ARGS__); })
#define wgpuTextureRelease(...) ::rx::gpu_async_handle<&wgpuTextureRelease>(__VA_ARGS__)
#define wgpuTextureViewAddRef(...) ::rx::gpu_async_handle<&wgpuTextureViewAddRef>(__VA_ARGS__)
#define wgpuTextureViewRelease(...) ::rx::gpu_async_handle<&wgpuTextureViewRelease>(__VA_ARGS__)

#endif
