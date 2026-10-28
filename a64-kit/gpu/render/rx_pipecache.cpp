// The browser's pipeline cache: every render and compute pipeline the game makes is recorded (its WGSL and its whole
// descriptor) to /rxcache/pipelines.bin, which the page keeps in IndexedDB; the next time the page opens they are all
// made again in the background, a few at a time, and kept. The browser's WebGPU (Dawn) gives back the object it
// already has for an identical descriptor, so when the game first needs one of them it is ready at once instead of a
// Tint -> HLSL -> D3D compile in the middle of play (first jumps, new areas, camera cuts all stalled on these), and
// Chrome's own disk cache of compiled shaders makes the background compiles cheap after the first time.
// /rxcache/seed.bin (the page fetches pipelines.bin from the server when there is one) is made in the same way.
// The records hold the WebGPU structs as bytes: a cache from another build of the page may make pipelines the game
// never asks for, which costs only their compile.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <unordered_set>

#include "rx_internal.h"

namespace rx {
namespace {

#if RX_WEB
constexpr uint32_t kMagic = 0x43505852;   // "RXPC"
constexpr uint32_t kVersion = 1;
const char* const kCachePath = "/rxcache/pipelines.bin";
const char* const kSeedPath = "/rxcache/seed.bin";

uint32_t layout_signature() {
    return (uint32_t)(sizeof(WGPUBindGroupLayoutEntry) * 1 + sizeof(WGPUVertexBufferLayout) * 3 +
                      sizeof(WGPUVertexAttribute) * 7 + sizeof(WGPUColorTargetState) * 11 +
                      sizeof(WGPUBlendState) * 13 + sizeof(WGPUPrimitiveState) * 17 +
                      sizeof(WGPUDepthStencilState) * 19 + sizeof(WGPUMultisampleState) * 23);
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
    const uint8_t* take(size_t n) {
        if ((size_t)(end - p) < n) { ok = false; return nullptr; }
        const uint8_t* r = p;
        p += n;
        return r;
    }
};

std::mutex cache_mu;                                // (the recorders: the render thread and the compute path)
std::string pending;   // records made since the last save (appended to the file: the whole file is ~60-95 MB)
size_t saved_size = 0;
int64_t last_save_ns = 0;
std::unordered_set<uint64_t> known_modules, known_records;
// A loaded record, where it is in its file: read again when the prewarm makes it. (The files stay in the page's file
// system, JavaScript memory: held in the host heap, the cache and the seed were ~120 MB of its ~570 for minutes of
// prewarm, and the open world ran it out of memory.)
struct Item {
    uint8_t file;   // 0 the page's cache, 1 the seed
    uint32_t off, len;
    char type;
};
FILE* files[2];
std::unordered_map<uint64_t, Item> module_texts;   // (loaded, for the prewarm)

/// A record's bytes (into buf, reused)
bool read_item(const Item& item, std::string& buf) {
    FILE* f = files[item.file & 1];
    buf.resize(item.len);
    return f && !std::fseek(f, (long)item.off, SEEK_SET) && std::fread(buf.data(), 1, item.len, f) == item.len;
}

// the prewarm (render thread: everything here, recording included, runs on it)
std::deque<Item> prewarm;                       // render ('R') and compute ('C') records to make
std::unordered_map<uint64_t, WGPUShaderModule> warm_modules;
std::vector<WGPURenderPipeline> warm_render;
std::vector<WGPUComputePipeline> warm_compute;
std::atomic<int> in_flight{0};
int64_t warmed = 0, prewarm_total = 0, prewarm_t0 = 0;

uint64_t text_hash(const std::string& s) { return hash_bytes(s.data(), s.size(), 0x9e3779b97f4a7c15ull); }

void add_record(char type, const std::string& payload) {   // (mu held)
    const uint64_t h = hash_bytes(payload.data(), payload.size(), (uint64_t)type);
    if (!known_records.insert(h).second) return;
    Writer w{pending};
    w.put(type);
    w.put((uint32_t)payload.size());
    w.bytes(payload.data(), payload.size());
}

uint64_t add_module(const std::string& text) {   // (mu held)
    const uint64_t h = text_hash(text);
    if (known_modules.insert(h).second) {
        std::string payload;
        Writer w{payload};
        w.put(h);
        w.bytes(text.data(), text.size());
        add_record('M', payload);
    }
    return h;
}

uint64_t add_module_hashed(uint64_t h, const std::string& wgsl) {   // (mu held)
    if (!h) return add_module(web_shader_source(wgsl));
    if (known_modules.count(h) || wgsl.empty()) return h;
    return add_module(web_shader_source(wgsl));
}

void put_layouts(Writer& w, const std::vector<WGPUBindGroupLayoutEntry>* layouts, int n) {
    w.put((uint32_t)n);
    for (int i = 0; i < n; i++) {
        w.put((uint32_t)layouts[i].size());
        w.bytes(layouts[i].data(), layouts[i].size() * sizeof(WGPUBindGroupLayoutEntry));
    }
}

/// Reads a cache file's records (the page's own cache, or a seed): modules go to module_texts and the rest onto the
/// prewarm, as places in the file, which stays open until the prewarm is done.
void load(const char* path, bool keep) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return;
    uint32_t head[3] = {};
    if (std::fread(head, 1, sizeof head, f) != sizeof head || head[0] != kMagic || head[1] != kVersion ||
        head[2] != layout_signature()) {
        std::fclose(f);
        hos_log("rx: pipeline cache %s is from another build: not used", path);
        return;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f) - (long)sizeof head;
    std::fseek(f, sizeof head, SEEK_SET);
    if (keep && size > (96l << 20)) {   // (grown past use: started again)
        std::fclose(f);
        std::remove(path);
        hos_log("rx: pipeline cache %s: %ld MB, started again", path, size >> 20);
        return;
    }
    const uint8_t which = keep ? 0 : 1;
    files[which] = f;
    std::string payload;
    long pos = (long)sizeof head;
    int modules = 0, records = 0;
    for (;;) {
        char type;
        uint32_t n;
        if (std::fread(&type, 1, 1, f) != 1 || std::fread(&n, 1, 4, f) != 4) break;
        pos += 5;
        payload.resize(n);
        if (std::fread(payload.data(), 1, n, f) != n) break;
        const uint32_t off = (uint32_t)pos;
        pos += n;
        const uint64_t h = hash_bytes(payload.data(), n, (uint64_t)type);
        if (!keep && known_records.count(h)) continue;
        if (keep) known_records.insert(h);
        if (type == 'M' && n >= 8) {
            uint64_t mh;
            std::memcpy(&mh, payload.data(), 8);
            if (keep) known_modules.insert(mh);
            module_texts.emplace(mh, Item{which, off + 8, n - 8, 'M'});
            modules++;
        } else if (type == 'R' || type == 'C') {
            prewarm.push_back(Item{which, off, n, type});
            records++;
        }
    }
    hos_log("rx: pipeline cache %s: %d modules, %d pipelines to make in the background", path, modules, records);
}

WGPUShaderModule warm_module(uint64_t h) {
    auto it = warm_modules.find(h);
    if (it != warm_modules.end()) return it->second;
    auto t = module_texts.find(h);
    static std::string text;
    if (t == module_texts.end() || !read_item(t->second, text)) return warm_modules[h] = nullptr;
    WGPUShaderSourceWGSL src{};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = WGPUStringView{text.data(), text.size()};
    WGPUShaderModuleDescriptor d{};
    d.nextInChain = &src.chain;
    return warm_modules[h] = program_module(&d);
}

WGPUPipelineLayout warm_layout(Reader& r, WGPUShaderStage visibility_unused) {
    (void)visibility_unused;
    const uint32_t n = r.get<uint32_t>();
    WGPUBindGroupLayout bgl[4]{};
    if (n > 4) { r.ok = false; return nullptr; }
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t count = r.get<uint32_t>();
        const uint8_t* p = r.take((size_t)count * sizeof(WGPUBindGroupLayoutEntry));
        if (!r.ok) return nullptr;
        std::vector<WGPUBindGroupLayoutEntry> e(count);
        std::memcpy(e.data(), p, (size_t)count * sizeof(WGPUBindGroupLayoutEntry));
        WGPUBindGroupLayoutDescriptor ld{};
        ld.entryCount = count;
        ld.entries = e.data();
        bgl[i] = wgpuDeviceCreateBindGroupLayout(dev.device, &ld);
    }
    WGPUPipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = n;
    pld.bindGroupLayouts = bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(dev.device, &pld);
    for (uint32_t i = 0; i < n; i++) wgpuBindGroupLayoutRelease(bgl[i]);
    return pl;
}

void warm_one(const Item& item) {
    static std::string bytes;
    if (!read_item(item, bytes)) return;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(bytes.data());
    Reader r{p, p + item.len};
    const char type = item.type;
    if (type == 'C') {
        const uint64_t mh = r.get<uint64_t>();
        WGPUPipelineLayout pl = warm_layout(r, WGPUShaderStage_Compute);
        WGPUShaderModule m = warm_module(mh);
        if (!r.ok || !m || !pl) {
            if (pl) wgpuPipelineLayoutRelease(pl);
            return;
        }
        WGPUComputePipelineDescriptor cd{};
        cd.layout = pl;
        cd.compute.module = m;
        cd.compute.entryPoint = WGPUStringView{"main", 4};
        WGPUCreateComputePipelineAsyncCallbackInfo cb{};
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline p, WGPUStringView, void*, void*) {
            in_flight--;
            if (status == WGPUCreatePipelineAsyncStatus_Success && p) warm_compute.push_back(p);
        };
        in_flight++;
        wgpuDeviceCreateComputePipelineAsync(dev.device, &cd, cb);
        wgpuPipelineLayoutRelease(pl);
        return;
    }
    if (type != 'R') return;
    const uint64_t vh = r.get<uint64_t>(), fh = r.get<uint64_t>();
    WGPUPipelineLayout pl = warm_layout(r, WGPUShaderStage_Vertex);
    const uint32_t nbuf = r.get<uint32_t>();
    if (!r.ok || nbuf > 32) {
        if (pl) wgpuPipelineLayoutRelease(pl);
        return;
    }
    std::vector<WGPUVertexBufferLayout> buffers(nbuf);
    std::vector<std::vector<WGPUVertexAttribute>> attributes(nbuf);
    for (uint32_t i = 0; i < nbuf; i++) {
        buffers[i] = r.get<WGPUVertexBufferLayout>();
        const uint32_t na = r.get<uint32_t>();
        if (!r.ok || na > 64) { r.ok = false; break; }
        attributes[i].resize(na);
        for (uint32_t a = 0; a < na; a++) attributes[i][a] = r.get<WGPUVertexAttribute>();
        buffers[i].attributes = attributes[i].data();
        buffers[i].attributeCount = na;
    }
    const uint32_t nt = r.get<uint32_t>();
    if (!r.ok || nt > 8) {
        if (pl) wgpuPipelineLayoutRelease(pl);
        return;
    }
    std::vector<WGPUColorTargetState> targets(nt);
    std::array<WGPUBlendState, 8> blends{};
    for (uint32_t i = 0; i < nt; i++) {
        targets[i] = r.get<WGPUColorTargetState>();
        const uint32_t has_blend = r.get<uint32_t>();
        blends[i] = r.get<WGPUBlendState>();
        targets[i].blend = has_blend ? &blends[i] : nullptr;
        targets[i].nextInChain = nullptr;
    }
    WGPUPrimitiveState prim = r.get<WGPUPrimitiveState>();
    const uint32_t has_depth = r.get<uint32_t>();
    WGPUDepthStencilState depth = r.get<WGPUDepthStencilState>();
    WGPUMultisampleState ms = r.get<WGPUMultisampleState>();
    WGPUShaderModule vs = warm_module(vh), fs = fh ? warm_module(fh) : nullptr;
    if (!r.ok || !pl || !vs || (fh && !fs)) {
        if (pl) wgpuPipelineLayoutRelease(pl);
        return;
    }
    prim.nextInChain = nullptr;
    depth.nextInChain = nullptr;
    ms.nextInChain = nullptr;
    WGPUFragmentState fst{};
    if (fs) {
        fst.module = fs;
        fst.entryPoint = WGPUStringView{"main", 4};
        fst.targetCount = nt;
        fst.targets = targets.data();
    }
    WGPURenderPipelineDescriptor d{};
    d.layout = pl;
    d.vertex.module = vs;
    d.vertex.entryPoint = WGPUStringView{"main", 4};
    d.vertex.bufferCount = nbuf;
    d.vertex.buffers = buffers.data();
    d.primitive = prim;
    d.depthStencil = has_depth ? &depth : nullptr;
    d.multisample = ms;
    d.fragment = fs ? &fst : nullptr;
    WGPUCreateRenderPipelineAsyncCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUCreatePipelineAsyncStatus status, WGPURenderPipeline p, WGPUStringView, void*, void*) {
        in_flight--;
        if (status == WGPUCreatePipelineAsyncStatus_Success && p) warm_render.push_back(p);
    };
    in_flight++;
    wgpuDeviceCreateRenderPipelineAsync(dev.device, &d, cb);
    wgpuPipelineLayoutRelease(pl);
}

void save() {   // (render thread: the records made since the last save, appended)
    std::lock_guard lk(cache_mu);
    if (pending.empty()) return;
    FILE* f = std::fopen(kCachePath, "rb");
    const bool fresh = !f;
    if (f) std::fclose(f);
    f = std::fopen(kCachePath, "ab");
    if (!f) return;
    if (fresh) {
        const uint32_t head[3] = {kMagic, kVersion, layout_signature()};
        std::fwrite(head, 1, sizeof head, f);
    }
    std::fwrite(pending.data(), 1, pending.size(), f);
    std::fclose(f);
    std::string().swap(pending);
}
#endif
}   // namespace

extern "C" void rx_pipecache_memory(int64_t* recorded_bytes, int64_t* seed_bytes, int64_t* prewarm_left) {
#if RX_WEB
    *recorded_bytes = (int64_t)pending.capacity();
    *seed_bytes = 0;
    *prewarm_left = (int64_t)prewarm.size();
#else
    *recorded_bytes = *seed_bytes = *prewarm_left = 0;
#endif
}

std::string web_shader_source(const std::string& wgsl) {
    // the browser (Tint) enforces derivative uniformity, which wgpu (naga) does not: the translated programs sample
    // in control flow it cannot prove uniform
    return RX_WEB ? "diagnostic(off, derivative_uniformity);\n" + wgsl : wgsl;
}

void pipecache_start() {
#if RX_WEB
    static const bool off = env_flag("RX_NO_PIPELINE_CACHE");
    if (off) return;
    std::lock_guard lk(cache_mu);
    load(kCachePath, true);
    load(kSeedPath, false);
    prewarm_total = (int64_t)prewarm.size();
    prewarm_t0 = hos_now_ns();
    last_save_ns = hos_now_ns();
#endif
}

uint64_t pipecache_text_hash(const std::string& wgsl) {
#if RX_WEB
    return text_hash(web_shader_source(wgsl));
#else
    (void)wgsl;
    return 0;
#endif
}

bool pipecache_has_module(uint64_t h) {
#if RX_WEB
    std::lock_guard lk(cache_mu);
    return known_modules.count(h) != 0;
#else
    (void)h;
    return false;
#endif
}

void pipecache_record_render(const WGPURenderPipelineDescriptor& d, const ShaderModule& vs, const ShaderModule* fs,
                             const std::vector<WGPUBindGroupLayoutEntry>* layouts, int nlayouts) {
#if RX_WEB
    std::lock_guard lk(cache_mu);
    std::string payload;
    Writer w{payload};
    w.put(add_module_hashed(vs.cache_hash, vs.source()));
    w.put(fs ? add_module_hashed(fs->cache_hash, fs->source()) : (uint64_t)0);
    put_layouts(w, layouts, nlayouts);
    w.put((uint32_t)d.vertex.bufferCount);
    for (size_t i = 0; i < d.vertex.bufferCount; i++) {
        WGPUVertexBufferLayout b = d.vertex.buffers[i];
        b.attributes = nullptr;
        b.nextInChain = nullptr;
        w.put(b);
        w.put((uint32_t)d.vertex.buffers[i].attributeCount);
        for (size_t a = 0; a < d.vertex.buffers[i].attributeCount; a++) {
            WGPUVertexAttribute at = d.vertex.buffers[i].attributes[a];
            at.nextInChain = nullptr;
            w.put(at);
        }
    }
    const uint32_t nt = d.fragment ? (uint32_t)d.fragment->targetCount : 0;
    w.put(nt);
    for (uint32_t i = 0; i < nt; i++) {
        WGPUColorTargetState t = d.fragment->targets[i];
        const WGPUBlendState blend = t.blend ? *t.blend : WGPUBlendState{};
        const uint32_t has_blend = t.blend != nullptr;
        t.blend = nullptr;
        t.nextInChain = nullptr;
        w.put(t);
        w.put(has_blend);
        w.put(blend);
    }
    WGPUPrimitiveState prim = d.primitive;
    prim.nextInChain = nullptr;
    w.put(prim);
    w.put((uint32_t)(d.depthStencil != nullptr));
    WGPUDepthStencilState depth = d.depthStencil ? *d.depthStencil : WGPUDepthStencilState{};
    depth.nextInChain = nullptr;
    w.put(depth);
    WGPUMultisampleState ms = d.multisample;
    ms.nextInChain = nullptr;
    w.put(ms);
    add_record('R', payload);
#else
    (void)d; (void)vs; (void)fs; (void)layouts; (void)nlayouts;
#endif
}

void pipecache_record_compute(const std::string& wgsl, const std::vector<WGPUBindGroupLayoutEntry>& layout) {
#if RX_WEB
    std::lock_guard lk(cache_mu);
    std::string payload;
    Writer w{payload};
    w.put(add_module(web_shader_source(wgsl)));
    put_layouts(w, &layout, 1);
    add_record('C', payload);
#else
    (void)wgsl; (void)layout;
#endif
}

void pipecache_step() {
#if RX_WEB
    // as many in the browser's compilers as it keeps up with (their callbacks: at present), up to ~6 ms a frame here:
    // done fast, while the title screen and the first loads run (spread thin, 2026-10-15, it was still going when the
    // opening area loaded and took that to 11 fps; fast, it was done in ~30 s and the area came in at 55-60 fps)
    int made = 0;
    const int64_t t0 = hos_now_ns();
    while (!prewarm.empty() && in_flight.load() < 16 && made < 24 && hos_now_ns() - t0 < 6000000) {
        warm_one(prewarm.front());
        prewarm.pop_front();
        made++;
        warmed++;
    }
    if (prewarm_total && prewarm.empty() && in_flight.load() == 0) {
        hos_log("rx: pipeline cache: %lld pipelines made in the background in %.1f s", (long long)warmed,
                (hos_now_ns() - prewarm_t0) / 1e9);
        prewarm_total = 0;
        module_texts.clear();
        for (FILE*& f : files)
            if (f) std::fclose(f), f = nullptr;
    }
    const int64_t now = hos_now_ns();
    if (now - last_save_ns > 10000000000LL) {   // (every 10 s, when something new was made)
        last_save_ns = now;
        save();
    }
#endif
}

}   // namespace rx

#ifdef __EMSCRIPTEN__
// (debugging: HOS_BIG_ALLOC_MB and RX_LEAK_SAMPLE, hos/web/stubs.c: they hook malloc itself, which this goes to)
#endif
