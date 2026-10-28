// Standalone native WebGPU regression for rx_core.cpp's pre-draw attachment snapshots.
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <condition_variable>

#include "../render/rx_internal.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

extern "C" {
void hos_log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
}
uint64_t hos_gpu_translate(uint64_t address) { return address; }
uint8_t guest_memory[0x10000]{};
uint8_t* hos_gpu_ptr(uint64_t address, uint64_t* available) {
    if (address >= sizeof guest_memory) { *available = 0; return nullptr; }
    *available = sizeof guest_memory - address;
    return guest_memory + address;
}
int64_t hos_now_ns() { return 0; }
uint8_t* hos_guest_base() { return guest_memory; }
uint64_t hos_guest_size() { return sizeof guest_memory; }
}
namespace rx {
void draw(Channel&, const Regs&, const DrawParams&) {}
}

namespace {
bool validation_failed = false;
void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void initialize() {
    using namespace rx;
    dev.instance = wgpuCreateInstance(nullptr);
    check(dev.instance != nullptr, "WebGPU instance");
    WGPURequestAdapterCallbackInfo adapter_callback{};
    adapter_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView message, void*, void*) {
        check(status == WGPURequestAdapterStatus_Success, str(message).c_str());
        dev.adapter = adapter;
    };
    wgpuInstanceRequestAdapter(dev.instance, nullptr, adapter_callback);
    for (unsigned i = 0; i < 100000 && !dev.adapter; ++i) wgpuInstanceProcessEvents(dev.instance);
    check(dev.adapter != nullptr, "WebGPU adapter");
    WGPUDeviceDescriptor descriptor{};
    descriptor.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType,
                                                          WGPUStringView message, void*, void*) {
        validation_failed = true;
        std::fprintf(stderr, "VALIDATION: %s\n", str(message).c_str());
    };
    WGPURequestDeviceCallbackInfo device_callback{};
    device_callback.mode = WGPUCallbackMode_AllowSpontaneous;
    device_callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                   WGPUStringView message, void*, void*) {
        check(status == WGPURequestDeviceStatus_Success, str(message).c_str());
        dev.device = device;
    };
    wgpuAdapterRequestDevice(dev.adapter, &descriptor, device_callback);
    for (unsigned i = 0; i < 100000 && !dev.device; ++i) wgpuInstanceProcessEvents(dev.instance);
    check(dev.device != nullptr, "WebGPU device");
    dev.queue = wgpuDeviceGetQueue(dev.device);
    wgpuDeviceGetLimits(dev.device, &dev.limits);
}

void expect_pixels(rx::Image& image, unsigned expected_red, unsigned expected_green = 64,
                   unsigned expected_blue = 96, unsigned expected_alpha = 255) {
    using namespace rx;
    WGPUBufferDescriptor descriptor{};
    descriptor.size = 512;
    descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev.device, &descriptor);
    WGPUTexelCopyTextureInfo source{};
    source.texture = image.tex;
    WGPUTexelCopyBufferInfo destination{};
    destination.buffer = buffer;
    destination.layout.bytesPerRow = 256;
    destination.layout.rowsPerImage = 2;
    const WGPUExtent3D extent{2, 2, 1};
    end_render_pass();
    wgpuCommandEncoderCopyTextureToBuffer(encoder(), &source, &destination, &extent);
    flush_batch();
    struct Result {
        std::mutex mutex;
        std::condition_variable ready;
        bool complete = false;
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    } result;
    WGPUBufferMapCallbackInfo callback{};
    callback.mode = WGPUCallbackMode_AllowSpontaneous;
    callback.userdata1 = &result;
    callback.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata, void*) {
        auto& result = *static_cast<Result*>(userdata);
        std::lock_guard lock(result.mutex);
        result.status = status;
        result.complete = true;
        result.ready.notify_one();
    };
    wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, 512, callback);
    wgpuDevicePoll(dev.device, true, nullptr);
    {
        std::unique_lock lock(result.mutex);
        result.ready.wait(lock, [&] { return result.complete; });
    }
    check(result.status == WGPUMapAsyncStatus_Success, "snapshot readback map");
    auto* pixels = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(buffer, 0, 512));
    check(pixels != nullptr, "snapshot readback pixels");
    for (unsigned y = 0; y < 2; ++y) {
        for (unsigned x = 0; x < 2; ++x) {
            const uint8_t* pixel = pixels + y * 256 + x * 4;
            const unsigned expected[4] = {expected_red, expected_green, expected_blue, expected_alpha};
            for (unsigned channel = 0; channel < 4; ++channel)
                check(std::abs(int(pixel[channel]) - int(expected[channel])) <= 1,
                      "GPU readback matches expected channel values");
        }
    }
    wgpuBufferUnmap(buffer);
    wgpuBufferRelease(buffer);
    check(!validation_failed, "no WebGPU validation errors");
}
} // namespace

int main() {
    using namespace rx;
    initialize();
    Image image;
    image.width = image.height = 2;
    image.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUTextureDescriptor texture{};
    texture.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment |
                    WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    texture.dimension = WGPUTextureDimension_2D;
    texture.size = {2, 2, 1};
    texture.format = image.format;
    texture.mipLevelCount = texture.sampleCount = 1;
    image.tex = wgpuDeviceCreateTexture(dev.device, &texture);
    image.attachment = wgpuTextureCreateView(image.tex, nullptr);
    const uint8_t initial[16] = {32,64,96,255, 32,64,96,255, 32,64,96,255, 32,64,96,255};
    WGPUTexelCopyTextureInfo upload{};
    upload.texture = image.tex;
    WGPUTexelCopyBufferLayout layout{};
    layout.bytesPerRow = 8;
    layout.rowsPerImage = 2;
    const WGPUExtent3D extent{2,2,1};
    wgpuQueueWriteTexture(dev.queue, &upload, initial, sizeof initial, &layout, &extent);

    const char* code = R"(
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(source, vec2<i32>(position.xy), 0) + vec4<f32>(32.0 / 255.0, 0.0, 0.0, 0.0);
}
)";
    WGPUShaderSourceWGSL wgsl{};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(code);
    WGPUShaderModuleDescriptor module_descriptor{};
    module_descriptor.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev.device, &module_descriptor);
    WGPUColorTargetState target{};
    target.format = image.format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fragment{};
    fragment.module = module;
    fragment.entryPoint = sv("fs");
    fragment.targetCount = 1;
    fragment.targets = &target;
    WGPURenderPipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.vertex.module = module;
    pipeline_descriptor.vertex.entryPoint = sv("vs");
    pipeline_descriptor.fragment = &fragment;
    pipeline_descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipeline_descriptor.multisample.count = 1;
    pipeline_descriptor.multisample.mask = 0xffffffff;
    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(dev.device, &pipeline_descriptor);
    WGPUBindGroupLayout group_layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    Attachments attachments;
    attachments.color[0] = &image;
    attachments.width = attachments.height = 2;
    Image* first_snapshot = nullptr;
    for (unsigned draw = 0; draw < 3; ++draw) {
        Image* snapshot = snapshot_for_feedback(image);
        check(snapshot != nullptr, "feedback snapshot created");
        check(snapshot->tex != image.tex, "feedback uses a distinct texture");
        if (!first_snapshot) first_snapshot = snapshot;
        check(snapshot == first_snapshot, "snapshot allocation reused");
        WGPUBindGroupEntry entry{};
        entry.textureView = snapshot->view(WGPUTextureViewDimension_2D, false);
        WGPUBindGroupDescriptor group_descriptor{};
        group_descriptor.layout = group_layout;
        group_descriptor.entryCount = 1;
        group_descriptor.entries = &entry;
        WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev.device, &group_descriptor);
        WGPURenderPassEncoder pass = render_pass(attachments);
        wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuBindGroupRelease(group);
        // First two draws share a batch/frame; third also checks reuse across submissions.
        if (draw == 1 || draw == 2) expect_pixels(image, 32 + 32 * (draw + 1));
    }
    // A cubemap and a one-layer render target can alias the same guest address. The 2D target must not
    // steal the cubemap binding and turn six valid guest layers into a dummy texture.
    {
        Image* alias = find_render_target(0x4000, 8, 8, PixelFormat::A8B8G8R8_UNORM);
        check(alias != nullptr, "alias render target created");
        Tegra::Texture::TICEntry tic{};
        tic.format.Assign(Tegra::Texture::TextureFormat::A8B8G8R8);
        tic.r_type.Assign(Tegra::Texture::ComponentType::UNORM);
        tic.g_type.Assign(Tegra::Texture::ComponentType::UNORM);
        tic.b_type.Assign(Tegra::Texture::ComponentType::UNORM);
        tic.a_type.Assign(Tegra::Texture::ComponentType::UNORM);
        tic.x_source.Assign(Tegra::Texture::SwizzleSource::R);
        tic.y_source.Assign(Tegra::Texture::SwizzleSource::G);
        tic.z_source.Assign(Tegra::Texture::SwizzleSource::B);
        tic.w_source.Assign(Tegra::Texture::SwizzleSource::A);
        tic.header_version.Assign(Tegra::Texture::TICHeaderVersion::BlockLinear);
        tic.address_low = 0x4000;
        tic.width_minus_one.Assign(7);
        tic.height_minus_1.Assign(7);
        tic.texture_type.Assign(Tegra::Texture::TextureType::TextureCubemap);
        tic.normalized_coords.Assign(1);
        std::memcpy(guest_memory + 0x1000, &tic, sizeof tic);
        Regs registers{};
        registers.tex_header.address_low = 0x1000;
        registers.tex_sampler.address_low = 0x2000;
        Shader::TextureDescriptor descriptor{};
        descriptor.type = Shader::TextureType::ColorCube;
        scope_push();
        BoundTexture bound = texture_for_handle(registers, 0, descriptor);
        BoundTexture dummy = dummy_texture(descriptor);
        check(scope_pop().empty(), "cubemap alias binding validates");
        check(bound.image == nullptr, "cubemap alias does not sample a 2D render target");
        check(bound.view != dummy.view, "cubemap alias preserves valid guest texture instead of dummy");
        check(!validation_failed, "cubemap alias produces no validation errors");
    }
    // Gather selects the guest's logical channel before gathering four neighboring texels. Its result
    // is not an RGBA vector: swizzling the four gathered values afterward would be equally incorrect.
    {
        std::fputs("gather: constructing IR\n", stderr);
        Shader::ObjectPool<Shader::IR::Inst> instructions;
        Shader::IR::Block block{instructions};
        Shader::IR::IREmitter ir{block, block.end()};
        Shader::IR::TextureInstInfo info{};
        info.type.Assign(Shader::TextureType::Color2D);
        const auto coords = ir.CompositeConstruct(ir.Imm32(0.5f), ir.Imm32(0.5f));
        const auto gathered = ir.ImageGather({}, coords, {}, {}, info);
        for (unsigned channel = 0; channel < 4; ++channel)
            ir.SetFragColor(0, channel, Shader::IR::F32{ir.CompositeExtract(gathered, channel)});
        Shader::IR::Program program;
        program.stage = Shader::Stage::Fragment;
        program.blocks.push_back(&block);
        program.syntax_list.push_back({{.block = &block}, Shader::IR::AbstractSyntaxNode::Type::Block});
        program.syntax_list.push_back({{}, Shader::IR::AbstractSyntaxNode::Type::Return});
        Shader::TextureDescriptor descriptor{};
        descriptor.type = Shader::TextureType::Color2D;
        descriptor.count = 1;
        program.info.texture_descriptors.push_back(descriptor);
        program.info.stores_frag_color[0] = true;
        Shader::Profile profile{};
        Shader::RuntimeInfo runtime{};
        Shader::Backend::Bindings bindings{};
        std::fputs("gather: emitting WGSL\n", stderr);
        std::string shader = Shader::Backend::WGSL::EmitWGSL(profile, runtime, program, bindings);
        std::fputs("gather: creating GPU pipeline\n", stderr);
        shader += R"(
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";
        wgsl.code = {shader.data(), shader.size()};
        scope_push();
        WGPUShaderModule gather_module = wgpuDeviceCreateShaderModule(dev.device, &module_descriptor);
        fragment.module = gather_module;
        fragment.entryPoint = sv("main");
        pipeline_descriptor.vertex.module = gather_module;
        pipeline_descriptor.fragment = &fragment;
        WGPURenderPipeline gather_pipeline = wgpuDeviceCreateRenderPipeline(dev.device, &pipeline_descriptor);
        check(scope_pop().empty(), "emitted gather WGSL and pipeline compile");
        WGPUBindGroupLayout empty_layout = wgpuRenderPipelineGetBindGroupLayout(gather_pipeline, 0);
        WGPUBindGroupDescriptor empty_descriptor{};
        empty_descriptor.layout = empty_layout;
        WGPUBindGroup empty_group = wgpuDeviceCreateBindGroup(dev.device, &empty_descriptor);
        WGPUBindGroupLayout gather_layout = wgpuRenderPipelineGetBindGroupLayout(gather_pipeline, 1);
        WGPUBufferDescriptor uniform_descriptor{};
        uniform_descriptor.size = 64;
        uniform_descriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer uniform = wgpuDeviceCreateBuffer(dev.device, &uniform_descriptor);
        WGPUSamplerDescriptor sampler_descriptor{};
        sampler_descriptor.maxAnisotropy = 1;
        WGPUSampler sampler = wgpuDeviceCreateSampler(dev.device, &sampler_descriptor);
        Image* snapshot = snapshot_for_feedback(image);
        check(snapshot != nullptr, "gather source snapshot");
        const unsigned selectors[] = {4, 0, 7, 3}; // B, Zero, OneFloat, G
        const unsigned values[] = {96, 0, 255, 64};
        for (unsigned test = 0; test < 4; ++test) {
            const uint32_t swizzle[4] = {selectors[test], 3, 4, 5};
            wgpuQueueWriteBuffer(dev.queue, uniform, 0, swizzle, sizeof swizzle);
            WGPUBindGroupEntry entries[3]{};
            entries[0].binding = 1;
            entries[0].textureView = snapshot->view(WGPUTextureViewDimension_2D, false);
            entries[1].binding = 2;
            entries[1].sampler = sampler;
            entries[2].binding = 3;
            entries[2].buffer = uniform;
            entries[2].size = 16;
            WGPUBindGroupDescriptor descriptor{};
            descriptor.layout = gather_layout;
            descriptor.entryCount = 3;
            descriptor.entries = entries;
            WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev.device, &descriptor);
            WGPURenderPassEncoder pass = render_pass(attachments);
            wgpuRenderPassEncoderSetPipeline(pass, gather_pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, empty_group, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(pass, 1, group, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            expect_pixels(image, values[test], values[test], values[test], values[test]);
            wgpuBindGroupRelease(group);
        }
        wgpuSamplerRelease(sampler);
        wgpuBufferRelease(uniform);
        wgpuBindGroupRelease(empty_group);
        wgpuBindGroupLayoutRelease(empty_layout);
        wgpuBindGroupLayoutRelease(gather_layout);
        wgpuRenderPipelineRelease(gather_pipeline);
        wgpuShaderModuleRelease(gather_module);
        std::puts("PASS: emitted gather shader honors channel remapping and constant zero/one");
    }
    wgpuBindGroupLayoutRelease(group_layout);
    wgpuRenderPipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    std::puts("PASS: feedback preserves channels, refreshes every draw, reuses storage, and survives submissions");
}
