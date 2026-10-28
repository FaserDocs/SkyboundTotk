// shader_test: run Maxwell shader programs dumped by the runtime (HOS_SHADER_DUMP) through suyu's shader recompiler.
// usage: shader_test [--glsl | --wgsl] [--print] <file.bin>...
//   file names are <program type>_<gpu va>.bin: type 0/1 vertex, 5 pixel. Each file starts at the 0x50-byte program
//   header; code follows.
// Prints one line per program (ok / the exception), and the generated source with --print.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/glsl/emit_glsl.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

#ifdef HAVE_WGSL
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
static LONG CALLBACK crash_handler(EXCEPTION_POINTERS* e) {
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION &&
        e->ExceptionRecord->ExceptionCode != EXCEPTION_STACK_OVERFLOW)
        return EXCEPTION_CONTINUE_SEARCH;
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(proc, nullptr, TRUE);
    CONTEXT ctx = *e->ContextRecord;
    STACKFRAME64 f{};
    f.AddrPC.Offset = ctx.Rip; f.AddrPC.Mode = AddrModeFlat;
    f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
    f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
    std::fprintf(stderr, "crash %#lx at %p (address %p)\n", e->ExceptionRecord->ExceptionCode,
                 e->ExceptionRecord->ExceptionAddress, (void*)e->ExceptionRecord->ExceptionInformation[1]);
    char buf[sizeof(SYMBOL_INFO) + 256];
    for (int d = 0; d < 24 && StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(), &f, &ctx, nullptr,
                                            SymFunctionTableAccess64, SymGetModuleBase64, nullptr); d++) {
        auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
        std::memset(buf, 0, sizeof buf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        IMAGEHLP_LINE64 line{sizeof(line)};
        DWORD ldisp = 0;
        const bool has = SymFromAddr(proc, f.AddrPC.Offset, &disp, sym);
        const bool hl = SymGetLineFromAddr64(proc, f.AddrPC.Offset, &ldisp, &line);
        std::fprintf(stderr, "  %s:%lu  %s\n", hl ? line.FileName : "?", hl ? line.LineNumber : 0, has ? sym->Name : "?");
    }
    std::fflush(stderr);
    ExitProcess(3);
}
#endif

namespace {
class FileEnvironment final : public Shader::Environment {
public:
    // start_ keeps the program's low address bits: Maxwell's scheduling words sit at absolute 32-byte boundaries.
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
    u32 ReadViewportTransformState() override { return 0; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0x1000; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}

private:
    std::vector<unsigned char> data;
};
} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    AddVectoredExceptionHandler(1, crash_handler);
#endif
    bool print = false, wgsl = false;
    std::string out_dir;
    int ok = 0, failed = 0;
    for (int i = 1; i < argc; i++) {
        std::string path = argv[i];
        if (path == "--print") { print = true; continue; }
        if (path == "--wgsl") { wgsl = true; continue; }
        if (path == "--glsl") { wgsl = false; continue; }
        if (path == "--out" && i + 1 < argc) { out_dir = argv[++i]; continue; }
        std::ifstream f(path, std::ios::binary);
        std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), {});
        if (data.size() < 0x60) { std::printf("%s: too short\n", path.c_str()); continue; }
        const auto slash = path.find_last_of("/\\");
        const char type = path[slash == std::string::npos ? 0 : slash + 1];
        const Shader::Stage stage = type == '5' ? Shader::Stage::Fragment : Shader::Stage::VertexB;
        try {
            const auto us = path.find('_', slash == std::string::npos ? 0 : slash);
            const u32 start = static_cast<u32>(std::stoull(path.substr(us + 1), nullptr, 16) & 0xff);
            FileEnvironment env{std::move(data), stage, start};
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
            Shader::ObjectPool<Shader::IR::Inst> insts;
            Shader::ObjectPool<Shader::IR::Block> blocks;
            Shader::Maxwell::Flow::CFG cfg{env, flow_blocks, env.StartAddress() + static_cast<u32>(sizeof(Shader::ProgramHeader))};
            Shader::HostTranslateInfo host{};
            host.support_float64 = false;
            host.support_int64 = false;
            Shader::IR::Program program = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, host);
            Shader::Profile profile{};
            Shader::RuntimeInfo runtime{};
            runtime.previous_stage_stores.mask.set();   // every input is provided (vertex fetch / previous stage)
            Shader::Backend::Bindings bindings{};
            std::string code;
#ifdef HAVE_WGSL
            if (wgsl) code = Shader::Backend::WGSL::EmitWGSL(profile, runtime, program, bindings);
            else
#endif
                code = Shader::Backend::GLSL::EmitGLSL(profile, runtime, program, bindings);
            std::printf("%s: ok (%zu blocks, %zu bytes of %s)\n", path.c_str(), program.blocks.size(), code.size(),
                        wgsl ? "WGSL" : "GLSL");
            if (print) std::printf("%s\n", code.c_str());
            if (!out_dir.empty()) {
                const auto base = path.substr(slash == std::string::npos ? 0 : slash + 1);
                std::ofstream o(out_dir + "/" + base + (wgsl ? ".wgsl" : ".glsl"), std::ios::binary);
                o << code;
            }
            ok++;
        } catch (const std::exception& e) {
            std::printf("%s: FAILED: %s\n", path.c_str(), e.what());
            failed++;
        }
    }
    std::printf("%d ok, %d failed\n", ok, failed);
    return failed ? 1 : 0;
}
