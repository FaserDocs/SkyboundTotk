// wgsl_check: Maxwell programs dumped by the runtime (HOS_SHADER_DUMP) -> suyu's frontend -> the WGSL backend ->
// wgpu-native's shader compiler (the same one the renderer uses). One line per program that fails, then a summary.
// usage: wgsl_check [--out <dir>] <file.bin>...   (file names <program type>_<gpu va>.bin: 0/1 vertex, 5 pixel)
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <webgpu/webgpu.h>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace {
class FileEnvironment final : public Shader::Environment {
public:
    FileEnvironment(std::vector<unsigned char> data_, Shader::Stage stage_, u32 start_) : data{std::move(data_)} {
        stage = stage_;
        start_address = start_;
        std::memcpy(&sph, data.data(), sizeof(sph));
    }
    u64 ReadInstruction(u32 address) override {
        u64 v = 0;
        address -= start_address;
        if (address + 8 <= data.size()) std::memcpy(&v, data.data() + address, 8);
        return v;
    }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 2; }
    u32 LocalMemorySize() const override { return 0x1000; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}

private:
    std::vector<unsigned char> data;
};

WGPUDevice device;

std::string compile(const std::string& code) {
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = WGPUStringView{code.data(), code.size()};
    WGPUShaderModuleDescriptor d{};
    d.nextInChain = &src.chain;
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUShaderModule m = wgpuDeviceCreateShaderModule(device, &d);
    std::string error;
    WGPUPopErrorScopeCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* u1, void*) {
        if (type != WGPUErrorType_NoError)
            *static_cast<std::string*>(u1) = std::string(message.data, message.length == WGPU_STRLEN
                                                                           ? std::strlen(message.data)
                                                                           : message.length);
    };
    cb.userdata1 = &error;
    wgpuDevicePopErrorScope(device, cb);
    if (m) wgpuShaderModuleRelease(m);
    return error;
}
} // namespace

int main(int argc, char** argv) {
    WGPUInstance instance = wgpuCreateInstance(nullptr);
    WGPUAdapter adapter = nullptr;
    WGPURequestAdapterCallbackInfo acb{};
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = [](WGPURequestAdapterStatus, WGPUAdapter a, WGPUStringView, void* u1, void*) {
        *static_cast<WGPUAdapter*>(u1) = a;
    };
    acb.userdata1 = &adapter;
    wgpuInstanceRequestAdapter(instance, nullptr, acb);
    WGPURequestDeviceCallbackInfo dcb{};
    dcb.mode = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback = [](WGPURequestDeviceStatus, WGPUDevice d, WGPUStringView, void* u1, void*) {
        *static_cast<WGPUDevice*>(u1) = d;
    };
    dcb.userdata1 = &device;
    if (adapter) wgpuAdapterRequestDevice(adapter, nullptr, dcb);
    if (!device) {
        std::printf("no WebGPU device\n");
        return 2;
    }
    std::string out_dir;
    int ok = 0, failed = 0;
    for (int i = 1; i < argc; i++) {
        std::string path = argv[i];
        if (path == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
            continue;
        }
        std::ifstream f(path, std::ios::binary);
        std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), {});
        if (data.size() < 0x60) continue;
        const auto slash = path.find_last_of("/\\");
        const std::string base = path.substr(slash == std::string::npos ? 0 : slash + 1);
        const Shader::Stage stage = base[0] == '5' ? Shader::Stage::Fragment : Shader::Stage::VertexB;
        std::string code, error;
        try {
            const auto us = base.find('_');
            const u32 start = static_cast<u32>(std::stoull(base.substr(us + 1), nullptr, 16) & 0xff);
            FileEnvironment env{std::move(data), stage, start};
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
            Shader::ObjectPool<Shader::IR::Inst> insts;
            Shader::ObjectPool<Shader::IR::Block> blocks;
            Shader::Maxwell::Flow::CFG cfg{env, flow_blocks,
                                           env.StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader))};
            Shader::HostTranslateInfo host{};
            Shader::IR::Program program = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, host);
            Shader::Profile profile{};
            Shader::RuntimeInfo runtime{};
            runtime.previous_stage_stores.mask.set();
            Shader::Backend::Bindings bindings{};
            code = Shader::Backend::WGSL::EmitWGSL(profile, runtime, program, bindings);
            error = compile(code);
        } catch (const std::exception& e) {
            error = std::string("translation: ") + e.what();
        }
        if (!out_dir.empty() && !code.empty()) std::ofstream(out_dir + "/" + base + ".wgsl", std::ios::binary) << code;
        if (error.empty()) {
            ok++;
        } else {
            failed++;
            if (error.size() > 600) error.resize(600);
            std::printf("%s: %s\n", base.c_str(), error.c_str());
        }
    }
    std::printf("%d compile, %d fail\n", ok, failed);
    return failed ? 1 : 0;
}
