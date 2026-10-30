// The native pipeline cache: every WGSL text the translator makes and every render pipeline made from them are
// appended to rxcache/pipelines.bin next to the exe; at the next start a thread reads the file and the render thread
// hands its pipelines to the pipeline compilers a few at a time (while the title screen shows), under the keys the
// draws will look them up by. Shader modules and pipelines (WGSL -> naga -> SPIR-V -> the driver, most of the
// pipeline compilers' time: 2.6 s of modules for 460 pipelines at one loading) are then ready before the game first
// draws with them; only the translation of the game's programs still happens on first use (on its workers).
// Compute programs ('C' records: their WGSL) get their shader modules made by the reading thread: the render thread
// made them in place at their first dispatch (WGSL parsing: ~270 ms of the boot's first frame).
// The file holds the game's shaders (made from its data): it stays on this machine, never in the repository.
// RX_NO_PIPELINE_CACHE=1: neither read nor written. RX_PIPELINE_CACHE_MAX=<n>: how many are made at a start (6000).
// Only pipelines some draw used in the last 4 sessions are made ('U' records: the keys each session used): each made
// pipeline holds ~390 KB of host memory (wgpu keeps its shader modules' IR, the driver its code), and the file holds
// every pipeline of every place ever played: all 3783 of them were 1.2 GB of the working set where a session at the
// title save used 690. RX_PIPELINE_CACHE_SESSIONS=<n> (4): how many sessions back; RX_PIPELINE_CACHE_ALL=1: all.
#if !RX_WEB
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rx_internal.h"

namespace rx {
namespace {

constexpr uint32_t kMagic = 0x4E505852;   // "RXPN"
constexpr uint32_t kVersion = 1;

uint32_t signature() {   // (another wgpu header layout or record format: the file is started again)
    return (uint32_t)(sizeof(WGPUVertexAttribute) * 3 + sizeof(WGPUColorTargetState) * 5 + sizeof(WGPUBlendState) * 7 +
                      sizeof(WGPUPrimitiveState) * 11 + sizeof(WGPUDepthStencilState) * 13 + kVersion * 17);
}

struct Writer {
    std::string& out;
    template <typename T> void put(const T& v) { out.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void bytes(const void* p, size_t n) { out.append(static_cast<const char*>(p), n); }
};
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <typename T> T get() {
        T v{};
        if (end - p < (ptrdiff_t)sizeof v) { ok = false; return v; }
        std::memcpy(&v, p, sizeof v);
        p += sizeof v;
        return v;
    }
};

std::string path_of_cache() {
    char exe[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, exe, sizeof exe);
    while (n && exe[n - 1] != '\\' && exe[n - 1] != '/') n--;
    exe[n] = 0;
    std::string dir = std::string(exe) + "rxcache";
    CreateDirectoryA(dir.c_str(), nullptr);
    return dir + "/pipelines.bin";
}

bool nc_on = true;
std::mutex nc_mu;   // (texts are recorded from the translation workers, pipelines from the render thread)
std::string nc_pending;
std::unordered_set<uint64_t> nc_known_texts, nc_known_pipelines;
std::string nc_path;
int64_t nc_last_save = 0;

// what the loader read (the render thread takes it once `loaded` is set)
std::atomic<bool> nc_loaded{false};
std::unordered_map<uint64_t, std::string> nc_texts;
std::vector<PipelineDesc> nc_records;   // in file order (the newest last)
size_t nc_next_record = 0;
std::unordered_map<uint64_t, std::string> nc_compute_texts;          // (the reading thread)
std::unordered_map<uint64_t, WGPUShaderModule> nc_compute_modules;   // made from them (nc_mu)
std::unordered_set<uint64_t> nc_known_compute;
uint32_t nc_session = 0;                 // this session's number (one more than the file's last; set by the reader)
std::vector<uint64_t> nc_used_pending;   // pipeline keys this session used, not yet written (nc_mu)
int64_t nc_made = 0, nc_t_start = 0;
bool nc_reported = false;

void put_desc(Writer& w, const PipelineDesc& d) {
    w.put(d.key);
    w.put(d.vs_text);
    w.put(d.fs_text);
    w.put((uint32_t)d.nlayouts);
    for (int i = 0; i < d.nlayouts && i < 2; i++) {
        w.put((uint32_t)d.layout[i].size());
        w.bytes(d.layout[i].data(), d.layout[i].size() * 4);
    }
    w.put((uint32_t)d.buffers.size());
    for (size_t i = 0; i < d.buffers.size(); i++) {
        w.put((uint32_t)d.buffers[i].stepMode);
        w.put(d.buffers[i].arrayStride);
        const auto& attrs = d.attributes[i];
        w.put((uint32_t)attrs.size());
        for (const WGPUVertexAttribute& a : attrs) {
            w.put((uint32_t)a.format);
            w.put(a.offset);
            w.put(a.shaderLocation);
        }
    }
    w.put((uint32_t)d.targets.size());
    for (size_t i = 0; i < d.targets.size(); i++) {
        w.put((uint32_t)d.targets[i].format);
        w.put((uint64_t)d.targets[i].writeMask);
        w.put((uint8_t)(i < d.has_blend.size() ? d.has_blend[i] : 0));
        const WGPUBlendState& b = d.blends[i < 8 ? i : 7];
        for (const WGPUBlendComponent& c : {b.color, b.alpha}) {
            w.put((uint32_t)c.operation);
            w.put((uint32_t)c.srcFactor);
            w.put((uint32_t)c.dstFactor);
        }
    }
    const WGPUPrimitiveState& p = d.primitive;
    w.put((uint32_t)p.topology);
    w.put((uint32_t)p.stripIndexFormat);
    w.put((uint32_t)p.frontFace);
    w.put((uint32_t)p.cullMode);
    w.put((uint32_t)p.unclippedDepth);
    w.put((uint8_t)d.has_depth);
    if (d.has_depth) {
        const WGPUDepthStencilState& s = d.depth;
        w.put((uint32_t)s.format);
        w.put((uint32_t)s.depthWriteEnabled);
        w.put((uint32_t)s.depthCompare);
        for (const WGPUStencilFaceState& f : {s.stencilFront, s.stencilBack}) {
            w.put((uint32_t)f.compare);
            w.put((uint32_t)f.failOp);
            w.put((uint32_t)f.depthFailOp);
            w.put((uint32_t)f.passOp);
        }
        w.put(s.stencilReadMask);
        w.put(s.stencilWriteMask);
        w.put(s.depthBias);
        w.put(s.depthBiasSlopeScale);
        w.put(s.depthBiasClamp);
    }
}

bool get_desc(Reader& r, PipelineDesc& d) {
    d.key = r.get<uint64_t>();
    d.vs_text = r.get<uint64_t>();
    d.fs_text = r.get<uint64_t>();
    d.nlayouts = (int)r.get<uint32_t>();
    if (d.nlayouts < 1 || d.nlayouts > 2) return false;
    for (int i = 0; i < d.nlayouts; i++) {
        const uint32_t n = r.get<uint32_t>();
        if (n > 256) return false;
        d.layout[i].resize(n);
        for (uint32_t k = 0; k < n; k++) d.layout[i][k] = r.get<uint32_t>();
    }
    const uint32_t nb = r.get<uint32_t>();
    if (nb > 32) return false;
    d.buffers.resize(nb);
    d.attributes.resize(nb);
    for (uint32_t i = 0; i < nb; i++) {
        WGPUVertexBufferLayout& b = d.buffers[i];
        b = WGPUVertexBufferLayout{};
        b.stepMode = (WGPUVertexStepMode)r.get<uint32_t>();
        b.arrayStride = r.get<uint64_t>();
        const uint32_t na = r.get<uint32_t>();
        if (na > 32) return false;
        d.attributes[i].resize(na);
        for (WGPUVertexAttribute& a : d.attributes[i]) {
            a = WGPUVertexAttribute{};
            a.format = (WGPUVertexFormat)r.get<uint32_t>();
            a.offset = r.get<uint64_t>();
            a.shaderLocation = r.get<uint32_t>();
        }
    }
    const uint32_t nt = r.get<uint32_t>();
    if (nt > 8) return false;
    d.targets.resize(nt);
    d.has_blend.resize(nt);
    for (uint32_t i = 0; i < nt; i++) {
        WGPUColorTargetState& t = d.targets[i];
        t = WGPUColorTargetState{};
        t.format = (WGPUTextureFormat)r.get<uint32_t>();
        t.writeMask = (WGPUColorWriteMask)r.get<uint64_t>();
        d.has_blend[i] = r.get<uint8_t>();
        WGPUBlendState& b = d.blends[i];
        for (WGPUBlendComponent* c : {&b.color, &b.alpha}) {
            c->operation = (WGPUBlendOperation)r.get<uint32_t>();
            c->srcFactor = (WGPUBlendFactor)r.get<uint32_t>();
            c->dstFactor = (WGPUBlendFactor)r.get<uint32_t>();
        }
    }
    WGPUPrimitiveState& p = d.primitive;
    p = WGPUPrimitiveState{};
    p.topology = (WGPUPrimitiveTopology)r.get<uint32_t>();
    p.stripIndexFormat = (WGPUIndexFormat)r.get<uint32_t>();
    p.frontFace = (WGPUFrontFace)r.get<uint32_t>();
    p.cullMode = (WGPUCullMode)r.get<uint32_t>();
    p.unclippedDepth = (WGPUBool)r.get<uint32_t>();
    d.has_depth = r.get<uint8_t>() != 0;
    if (d.has_depth) {
        WGPUDepthStencilState& s = d.depth;
        s = WGPUDepthStencilState{};
        s.format = (WGPUTextureFormat)r.get<uint32_t>();
        s.depthWriteEnabled = (WGPUOptionalBool)r.get<uint32_t>();
        s.depthCompare = (WGPUCompareFunction)r.get<uint32_t>();
        for (WGPUStencilFaceState* f : {&s.stencilFront, &s.stencilBack}) {
            f->compare = (WGPUCompareFunction)r.get<uint32_t>();
            f->failOp = (WGPUStencilOperation)r.get<uint32_t>();
            f->depthFailOp = (WGPUStencilOperation)r.get<uint32_t>();
            f->passOp = (WGPUStencilOperation)r.get<uint32_t>();
        }
        s.stencilReadMask = r.get<uint32_t>();
        s.stencilWriteMask = r.get<uint32_t>();
        s.depthBias = r.get<int32_t>();
        s.depthBiasSlopeScale = r.get<float>();
        s.depthBiasClamp = r.get<float>();
    }
    return r.ok;
}

void add_record(char type, const std::string& payload) {   // (nc_mu held)
    Writer w{nc_pending};
    w.put(type);
    w.put((uint32_t)payload.size());
    w.bytes(payload.data(), payload.size());
}

/// Reads the file (a thread of its own): texts by hash and the pipeline records, for the render thread.
void load() {
    FILE* f = std::fopen(nc_path.c_str(), "rb");
    if (!f) {
        nc_loaded = true;
        return;
    }
    uint32_t head[3] = {};
    std::string payload;
    size_t n_texts = 0;
    int64_t bytes = 0;
    if (std::fread(head, 1, sizeof head, f) == sizeof head && head[0] == kMagic && head[1] == kVersion &&
        head[2] == signature()) {
        std::unordered_map<uint64_t, size_t> where;   // pipeline key -> its record (a later one replaces it)
        std::vector<PipelineDesc> list;
        std::unordered_map<uint64_t, uint32_t> last_session;   // pipeline key -> the last session that used it
        uint32_t sessions = 0;
        for (;;) {
            char type;
            uint32_t len;
            if (std::fread(&type, 1, 1, f) != 1 || std::fread(&len, 1, 4, f) != 4 || len > (64u << 20)) break;
            payload.resize(len);
            if (std::fread(payload.data(), 1, len, f) != len) break;
            bytes += 5 + len;
            Reader r{reinterpret_cast<const uint8_t*>(payload.data()), reinterpret_cast<const uint8_t*>(payload.data()) + len};
            if (type == 'T' && len >= 8) {
                const uint64_t h = r.get<uint64_t>();
                std::lock_guard lk(nc_mu);
                if (nc_known_texts.insert(h).second) {
                    nc_texts.emplace(h, payload.substr(8));
                    n_texts++;
                }
            } else if (type == 'U' && len >= 4) {
                const uint32_t session = r.get<uint32_t>();
                sessions = std::max(sessions, session);
                while (r.ok && r.p < r.end) {
                    const uint64_t key = r.get<uint64_t>();
                    if (!r.ok) break;
                    uint32_t& last = last_session[key];
                    last = std::max(last, session);
                }
            } else if (type == 'C' && len >= 8) {
                const uint64_t h = r.get<uint64_t>();
                std::lock_guard lk(nc_mu);
                if (nc_known_compute.insert(h).second) nc_compute_texts.emplace(h, payload.substr(8));
            } else if (type == 'P') {
                PipelineDesc d;
                if (!get_desc(r, d)) continue;
                {
                    std::lock_guard lk(nc_mu);
                    nc_known_pipelines.insert(d.key);
                }
                auto it = where.find(d.key);
                if (it != where.end()) list[it->second] = std::move(d);
                else {
                    where.emplace(d.key, list.size());
                    list.push_back(std::move(d));
                }
            }
        }
        // (pipelines no recent session used are not made: their host memory is the cost, see the top)
        static const int window = getenv("RX_PIPELINE_CACHE_SESSIONS") ? atoi(getenv("RX_PIPELINE_CACHE_SESSIONS")) : 4;
        const bool all = getenv("RX_PIPELINE_CACHE_ALL") != nullptr;
        size_t stale = 0;
        for (PipelineDesc& d : list) {
            auto it = last_session.find(d.key);
            const uint32_t last = it == last_session.end() ? 0 : it->second;   // (0: recorded before 'U' records)
            if (all || (int64_t)last >= (int64_t)sessions + 1 - window) nc_records.push_back(std::move(d));
            else stale++;
        }
        nc_session = sessions + 1;
        if (stale)
            hos_log("rx: pipeline cache: %zu pipelines no draw used in the last %d sessions are not made", stale, window);
    } else {
        std::fclose(f);
        f = nullptr;
        std::remove(nc_path.c_str());
        hos_log("rx: pipeline cache %s is from another build or damaged: started again", nc_path.c_str());
    }
    if (f) std::fclose(f);
    if (bytes > (400ll << 20)) {   // (grown past use: the next session starts it again)
        std::remove(nc_path.c_str());
        std::lock_guard lk(nc_mu);
        nc_known_texts.clear();
        nc_known_pipelines.clear();
        hos_log("rx: pipeline cache %s: %lld MB, started again", nc_path.c_str(), (long long)(bytes >> 20));
    }
    hos_log("rx: pipeline cache: %zu shader texts, %zu pipelines to make in the background, %zu compute programs "
            "(%lld MB)", n_texts, nc_records.size(), nc_compute_texts.size(), (long long)(bytes >> 20));
    nc_loaded = true;
    // the compute programs' modules, here (WGSL parsing and validation; they compiled when they were recorded)
    const int64_t t0 = hos_now_ns();
    size_t made = 0;
    for (auto& [h, text] : nc_compute_texts) {
        WGPUShaderSourceWGSL src{};
        src.chain.sType = WGPUSType_ShaderSourceWGSL;
        src.code = WGPUStringView{text.data(), text.size()};
        WGPUShaderModuleDescriptor d{};
        d.nextInChain = &src.chain;
        if (WGPUShaderModule m = program_module(&d)) {
            std::lock_guard lk(nc_mu);
            nc_compute_modules.emplace(h, m);
            made++;
        }
    }
    nc_compute_texts.clear();
    if (made) hos_log("rx: pipeline cache: %zu compute modules made in %.0f ms", made, (hos_now_ns() - t0) / 1e6);
}

void save() {
    std::string out;
    {
        std::lock_guard lk(nc_mu);
        if (nc_loaded && nc_session && !nc_used_pending.empty()) {   // (the session's number is known once read)
            std::string payload;
            Writer w{payload};
            w.put(nc_session);
            w.bytes(nc_used_pending.data(), nc_used_pending.size() * sizeof(uint64_t));
            add_record('U', payload);
            nc_used_pending.clear();
        }
        if (nc_pending.empty()) return;
        out.swap(nc_pending);
    }
    FILE* f = std::fopen(nc_path.c_str(), "ab");
    if (!f) return;
    std::fseek(f, 0, SEEK_END);   // (in append mode the position reads 0 until the first write)
    if (std::ftell(f) == 0) {
        const uint32_t head[3] = {kMagic, kVersion, signature()};
        std::fwrite(head, 1, sizeof head, f);
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
}

} // namespace

bool native_cache_on() { return nc_on; }

void native_cache_start() {
    nc_on = getenv("RX_NO_PIPELINE_CACHE") == nullptr;
    if (!nc_on) return;
    nc_path = path_of_cache();
    nc_t_start = hos_now_ns();
    std::thread([] {
        background_thread();
        load();
    }).detach();
}

void native_cache_text(uint64_t h, const std::string& text) {
    if (!nc_on) return;
    std::lock_guard lk(nc_mu);
    if (!nc_known_texts.insert(h).second) return;
    std::string payload;
    Writer w{payload};
    w.put(h);
    w.bytes(text.data(), text.size());
    add_record('T', payload);
}

void native_cache_used(uint64_t key) {
    if (!nc_on) return;
    std::lock_guard lk(nc_mu);
    nc_used_pending.push_back(key);
}

void native_cache_compute_text(uint64_t h, const std::string& text) {
    if (!nc_on) return;
    std::lock_guard lk(nc_mu);
    if (!nc_known_compute.insert(h).second) return;
    std::string payload;
    Writer w{payload};
    w.put(h);
    w.bytes(text.data(), text.size());
    add_record('C', payload);
}

WGPUShaderModule native_cache_compute_module(uint64_t h) {
    if (!nc_on) return nullptr;
    std::lock_guard lk(nc_mu);
    auto it = nc_compute_modules.find(h);
    if (it == nc_compute_modules.end()) return nullptr;
    WGPUShaderModule m = it->second;   // (the caller's now)
    nc_compute_modules.erase(it);
    return m;
}

void native_cache_pipeline(const PipelineDesc& d) {
    if (!nc_on) return;
    std::lock_guard lk(nc_mu);
    if (!nc_known_pipelines.insert(d.key).second) return;
    std::string payload;
    Writer w{payload};
    put_desc(w, d);
    add_record('P', payload);
}

void native_cache_step() {
    if (!nc_on) return;
    const int64_t now = hos_now_ns();
    if (now - nc_last_save > 5000000000LL) {   // (new records every 5 s: an abrupt exit loses at most those)
        nc_last_save = now;
        save();
    }
    if (!nc_loaded.load(std::memory_order_acquire) || nc_next_record >= nc_records.size()) {
        if (nc_loaded && !nc_reported && nc_next_record && nc_next_record >= nc_records.size() && prewarm_in_flight() == 0) {
            nc_reported = true;
            hos_log("rx: pipeline cache: %lld pipelines made in the background in %.1f s", (long long)nc_made,
                    (hos_now_ns() - nc_t_start) / 1e9);
            nc_records.clear();
            nc_records.shrink_to_fit();
            std::lock_guard lk(nc_mu);
            nc_texts.clear();
        }
        return;
    }
    static const size_t cap = getenv("RX_PIPELINE_CACHE_MAX") ? (size_t)atoi(getenv("RX_PIPELINE_CACHE_MAX")) : 6000;
    if (nc_next_record == 0 && nc_records.size() > cap) nc_next_record = nc_records.size() - cap;   // (the newest)
    // a few at a time, while the compilers keep up: the game's own new pipelines go first
    for (int budget = 4; budget > 0 && nc_next_record < nc_records.size() && prewarm_in_flight() < 4; budget--) {
        const PipelineDesc& d = nc_records[nc_next_record++];
        std::string vs, fs;
        {
            std::lock_guard lk(nc_mu);
            auto v = nc_texts.find(d.vs_text);
            if (v == nc_texts.end()) continue;
            vs = v->second;
            if (d.fs_text) {
                auto f = nc_texts.find(d.fs_text);
                if (f == nc_texts.end()) continue;
                fs = f->second;
            }
        }
        ShaderModule* vm = prewarm_stage_module(d.vs_text, vs, "vertex");
        ShaderModule* fm = d.fs_text ? prewarm_stage_module(d.fs_text, fs, "pixel") : nullptr;
        if (!vm || (d.fs_text && !fm)) continue;
        if (prewarm_pipeline(d, vm, fm)) nc_made++;
    }
}

} // namespace rx

#else
#include "rx_internal.h"
namespace rx {
bool native_cache_on() { return false; }
void native_cache_start() {}
void native_cache_step() {}
void native_cache_text(uint64_t, const std::string&) {}
void native_cache_pipeline(const PipelineDesc&) {}
void native_cache_compute_text(uint64_t, const std::string&) {}
void native_cache_used(uint64_t) {}
WGPUShaderModule native_cache_compute_module(uint64_t) { return nullptr; }
} // namespace rx
#endif
