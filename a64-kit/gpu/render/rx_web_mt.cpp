// The threaded web build's WebGPU thread (RX_WEB_MT; see rx_web_mt.h). It owns the device and the page's canvas. It
// runs, in the order they came, the calls other threads hand it (run_on_gpu_thread: they wait; post_to_gpu_thread:
// they do not) and the batches the renderer's thread flushed (rx_core.cpp push_batch posts gpu_thread_batch: replay,
// uploads, submit, present, readbacks, as the native submit thread), and with nothing to do waits in its event loop,
// where the browser runs WebGPU's callbacks.
#include "rx_internal.h"

#if RX_WEB_MT
#include <algorithm>
#include <atomic>
#include <climits>
#include <string>
#include <unordered_map>
#include <vector>
#include <deque>
#include <mutex>
#include <emscripten/threading.h>
#include <pthread.h>

extern "C" {
void rx_web_wait_word(uint32_t addr, uint32_t value, double ms);   // rx_core.cpp
void hos_host_thread_register_self(const char* name);
}

namespace rx {

namespace {
thread_local bool is_gpu_thread = false;
struct Job {
    void (*fn)(void*);
    void* arg;
    uint32_t done;
    bool owned;   // (posted: nobody waits, deleted once run)
};
std::mutex jobs_mu;
std::deque<Job*> jobs;
std::deque<Job*> jobs_now;   // (run_on_gpu_thread_now: ahead of jobs)
uint32_t gpu_word;   // bumped when a job or a batch comes: the idle WebGPU thread wakes on it
std::mutex start_mu;
bool started_thread = false;

void finish(Job* j) {
    if (j->owned) {
        delete j;
    } else {
        __atomic_store_n(&j->done, 1, __ATOMIC_RELEASE);
        emscripten_futex_wake(&j->done, INT_MAX);
    }
}

bool serve_jobs() {
    bool any = false;
    for (;;) {
        serve_gpu_calls_now_if_any();
        Job* j;
        {
            std::lock_guard lk(jobs_mu);
            if (jobs.empty()) return any;
            j = jobs.front();
            jobs.pop_front();
        }
        j->fn(j->arg);
        finish(j);
        any = true;
        if (framegen_web_due && hos_now_ns() >= framegen_web_due) framegen_web_tick();   // (jobs keep coming at times)
    }
}

void* gpu_main(void*) {
    is_gpu_thread = true;
    hos_host_thread_register_self("WebGPU");
    for (;;) {
        const uint32_t seen = __atomic_load_n(&gpu_word, __ATOMIC_SEQ_CST);
        const bool did = serve_jobs();
        // (the callbacks of pipelines made asynchronously and of readbacks: other threads' calls of this are no-ops)
        if (dev.instance) (wgpuInstanceProcessEvents)(dev.instance);
        // (frame interpolation: the made frames shown as they come due, rx_framegen.cpp)
        const int64_t next = framegen_web_tick();
        if (!did) rx_web_wait_word((uint32_t)(uintptr_t)&gpu_word, seen, next >= 0 ? std::min(2.0, next / 1e6) : 2.0);
    }
    return nullptr;
}
}  // namespace

bool on_gpu_thread() { return is_gpu_thread; }

std::atomic<int> gpu_calls_now{0};

void serve_gpu_calls_now() {
    for (;;) {
        Job* j;
        {
            std::lock_guard lk(jobs_mu);
            if (jobs_now.empty()) return;
            j = jobs_now.front();
            jobs_now.pop_front();
        }
        gpu_calls_now--;
        j->fn(j->arg);
        finish(j);
    }
}

void wake_gpu_thread() {
    __atomic_add_fetch(&gpu_word, 1, __ATOMIC_SEQ_CST);
    emscripten_futex_wake(&gpu_word, INT_MAX);
}

/// The WebGPU thread, started once: it takes the canvas the page shows (#screen).
void start_gpu_thread() {
    std::lock_guard lk(start_mu);
    if (started_thread) return;
    started_thread = true;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, 2 << 20);
    emscripten_pthread_attr_settransferredcanvases(&a, "#screen");
    pthread_t t;
    pthread_create(&t, &a, gpu_main, nullptr);
    pthread_attr_destroy(&a);
    pthread_detach(t);
}

/* RX_MT_LOG=1: every 5 s, the calls other threads handed the WebGPU thread, by function: how many and how long their
   callers waited (a call waits for the batch the thread is running) */
namespace {
struct CallStat {
    const char* what = nullptr;
    int64_t n = 0, wait_ns = 0;
};
std::mutex stat_mu;
CallStat stats[128];
int64_t stats_since;

void note_call(const char* what, int64_t wait_ns) {
    std::lock_guard lk(stat_mu);
    int i = 0;
    while (i < 128 && stats[i].what && stats[i].what != what) i++;
    if (i < 128) {
        stats[i].what = what;
        stats[i].n++;
        stats[i].wait_ns += wait_ns;
    }
    const int64_t now = hos_now_ns();
    if (!stats_since) stats_since = now;
    if (now - stats_since < 5000000000LL) return;
    std::sort(std::begin(stats), std::end(stats), [](const CallStat& a, const CallStat& b) { return a.wait_ns > b.wait_ns; });
    std::string line;
    int64_t total = 0, waited = 0;
    for (const CallStat& s : stats) {
        if (!s.what) break;
        total += s.n;
        waited += s.wait_ns;
        if (line.size() < 900) line += " " + std::string(s.what) + " " + std::to_string(s.n) + " (" + std::to_string(s.wait_ns / 1000000) + " ms)";
    }
    hos_log("rx: WebGPU thread calls from others in %.1f s: %lld, waited %.0f ms:%s", (now - stats_since) / 1e9,
            (long long)total, waited / 1e6, line.c_str());
    for (CallStat& s : stats) s = {};
    stats_since = now;
}
}  // namespace

void mt_note_wait(const char* what, int64_t ns) {
    static const bool log = env_flag("RX_MT_LOG");
    if (log) note_call(what, ns);
}

void run_on_gpu_thread(void (*fn)(void*), void* arg, const char* what) {
    start_gpu_thread();
    static const bool log = env_flag("RX_MT_LOG");
    const int64_t t0 = log ? hos_now_ns() : 0;
    Job j{fn, arg, 0, false};
    {
        std::lock_guard lk(jobs_mu);
        jobs.push_back(&j);
    }
    wake_gpu_thread();
    while (!__atomic_load_n(&j.done, __ATOMIC_ACQUIRE)) emscripten_futex_wait(&j.done, 0, 1000.0);
    if (log) note_call(what, hos_now_ns() - t0);
}
void run_on_gpu_thread_now(void (*fn)(void*), void* arg, const char* what) {
    start_gpu_thread();
    static const bool log = env_flag("RX_MT_LOG");
    const int64_t t0 = log ? hos_now_ns() : 0;
    Job j{fn, arg, 0, false};
    {
        std::lock_guard lk(jobs_mu);
        jobs_now.push_back(&j);
    }
    gpu_calls_now++;
    wake_gpu_thread();
    while (!__atomic_load_n(&j.done, __ATOMIC_ACQUIRE)) emscripten_futex_wait(&j.done, 0, 1000.0);
    if (log) note_call(what, hos_now_ns() - t0);
}
void post_to_gpu_thread(void (*fn)(void*), void* arg) {
    start_gpu_thread();
    {
        std::lock_guard lk(jobs_mu);
        jobs.push_back(new Job{fn, arg, 0, true});
    }
    wake_gpu_thread();
}

namespace {
struct BufferWrite {
    WGPUQueue q;
    WGPUBuffer b;
    uint64_t offset;
    std::vector<uint8_t> data;
};
struct TextureWrite {
    WGPUQueue q;
    WGPUTexelCopyTextureInfo dst;
    WGPUTexelCopyBufferLayout layout;
    WGPUExtent3D extent;
    std::vector<uint8_t> data;
};
// (the copies waiting in the queue: past 64 MB a write waits for them, as a load uploads faster than the WebGPU
// thread may take it and the web's host heap is ~570 MB)
std::atomic<int64_t> queued_bytes{0};
constexpr int64_t kQueuedBytesMax = 64ll << 20;
void before_write(size_t size) {
    queued_bytes += (int64_t)size;
    if (queued_bytes.load() > kQueuedBytesMax)
        on_gpu("(queued writes)", [] {});   // (everything queued before it has run when it returns)
}
std::mutex sizes_mu;
std::unordered_map<WGPUBuffer, uint64_t> sizes;   // (a reused handle is overwritten at its creation)
}  // namespace

void mt_queue_write_buffer(WGPUQueue q, WGPUBuffer b, uint64_t offset, const void* data, size_t size) {
    if (on_gpu_thread()) return (wgpuQueueWriteBuffer)(q, b, offset, data, size);
    before_write(size);
    auto* w = new BufferWrite{q, b, offset, std::vector<uint8_t>(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size)};
    post_to_gpu_thread([](void* p) {
        auto* w = static_cast<BufferWrite*>(p);
        (wgpuQueueWriteBuffer)(w->q, w->b, w->offset, w->data.data(), w->data.size());
        queued_bytes -= (int64_t)w->data.size();
        delete w;
    }, w);
}

void mt_queue_write_texture(WGPUQueue q, const WGPUTexelCopyTextureInfo* dst, const void* data, size_t size,
                            const WGPUTexelCopyBufferLayout* layout, const WGPUExtent3D* extent) {
    if (on_gpu_thread()) return (wgpuQueueWriteTexture)(q, dst, data, size, layout, extent);
    before_write(size);
    auto* w = new TextureWrite{q, *dst, *layout, *extent, std::vector<uint8_t>(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size)};
    post_to_gpu_thread([](void* p) {
        auto* w = static_cast<TextureWrite*>(p);
        (wgpuQueueWriteTexture)(w->q, &w->dst, w->data.data(), w->data.size(), &w->layout, &w->extent);
        queued_bytes -= (int64_t)w->data.size();
        delete w;
    }, w);
}

WGPUBuffer mt_create_buffer(WGPUDevice d, const WGPUBufferDescriptor* desc) {
    WGPUBuffer b = on_gpu<true>("wgpuDeviceCreateBuffer", [&] { return (wgpuDeviceCreateBuffer)(d, desc); });
    if (b) {
        std::lock_guard lk(sizes_mu);
        sizes[b] = desc->size;
    }
    return b;
}

WGPUTexture mt_create_texture(WGPUDevice d, const WGPUTextureDescriptor* desc) {
    // (named by its label in RX_MT_LOG's counts: a literal's address, so one entry each)
    const char* what = desc->label.data && desc->label.length ? desc->label.data : "wgpuDeviceCreateTexture (no label)";
    return on_gpu<true>(what, [&] { return (wgpuDeviceCreateTexture)(d, desc); });
}

namespace {
/// What the renderer holds for a bind group (as its WGPUBindGroup): the group, made by the WebGPU thread in the order
/// of the queue (so before the batches flushed after it, which use it), and the references the renderer counts
struct BindGroupProxy {
    WGPUBindGroup group = nullptr;
    std::atomic<int> refs{1};
};
struct BindGroupMake {
    BindGroupProxy* proxy;
    WGPUDevice device;
    WGPUBindGroupLayout layout;
    std::vector<WGPUBindGroupEntry> entries;
};
void free_bind_group(BindGroupProxy* g) {
    if (g->group) (wgpuBindGroupRelease)(g->group);
    delete g;
}
}  // namespace

WGPUBindGroup mt_create_bind_group(WGPUDevice d, const WGPUBindGroupDescriptor* desc) {
    auto* g = new BindGroupProxy;
    if (on_gpu_thread()) {
        g->group = (wgpuDeviceCreateBindGroup)(d, desc);
    } else {
        // (the entries' objects outlive this: their releases are queued after it)
        auto* m = new BindGroupMake{g, d, desc->layout, std::vector<WGPUBindGroupEntry>(desc->entries, desc->entries + desc->entryCount)};
        post_to_gpu_thread([](void* p) {
            auto* m = static_cast<BindGroupMake*>(p);
            WGPUBindGroupDescriptor bd{};
            bd.layout = m->layout;
            bd.entryCount = m->entries.size();
            bd.entries = m->entries.data();
            m->proxy->group = (wgpuDeviceCreateBindGroup)(m->device, &bd);
            delete m;
        }, m);
    }
    return reinterpret_cast<WGPUBindGroup>(g);
}
void mt_bind_group_add_ref(WGPUBindGroup g) {
    if (g) reinterpret_cast<BindGroupProxy*>(g)->refs++;
}
void mt_bind_group_release(WGPUBindGroup g) {
    auto* p = reinterpret_cast<BindGroupProxy*>(g);
    if (!p || --p->refs > 0) return;
    if (on_gpu_thread()) return free_bind_group(p);
    post_to_gpu_thread([](void* q) { free_bind_group(static_cast<BindGroupProxy*>(q)); }, p);   // (after its uses)
}
WGPUBindGroup mt_bind_group(WGPUBindGroup g) {
    return g ? reinterpret_cast<BindGroupProxy*>(g)->group : nullptr;
}

uint64_t mt_buffer_size(WGPUBuffer b) {
    {
        std::lock_guard lk(sizes_mu);
        if (auto it = sizes.find(b); it != sizes.end()) return it->second;
    }
    return on_gpu<true>("wgpuBufferGetSize", [&] { return (wgpuBufferGetSize)(b); });
}
}  // namespace rx
#endif
