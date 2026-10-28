// The buffer cache: guest memory mirrored in GPU buffers, so that vertex, index and read-only storage data stays on
// the GPU across draws and frames and only pages the game changed are uploaded again.
//
// Guest memory (4 GB at hos_guest_base) is covered by segments of 16 MB that start every 8 MB: any range of up to
// 8 MB lies inside one segment, so it binds as one buffer: an existing one that holds it, else an even one (they tile
// memory; an odd one only for a range across a 16 MB boundary; RX_SEGMENT_OVERLAP=1: always the one starting at or
// below it). A segment is created when a draw first needs it; its pages are uploaded when a draw needs them (one
// valid bit per 4 KB page; pages a dispatch wrote are copied from the neighbour that holds them).
//
// Changes: the guest arena is allocated with MEM_WRITE_WATCH (hos_main.c). At the start of each GPU submission the
// pages written since the last one are invalidated (GetWriteWatch), which catches every CPU write the submission's
// data needs. Writes the command processor makes while it runs a submission (inline uploads, copies: hos/gpu.c)
// are reported through rx_memory_written; if a page they change was read earlier in the open batch, the batch is
// submitted first, so earlier draws see the old contents.
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rx_internal.h"

extern "C" {
uint8_t* hos_guest_base(void);
uint64_t hos_guest_size(void);
int hos_gpu_guest_ranges(uint64_t (*out)[2], int max);
}

namespace rx {
namespace {

constexpr uint64_t kPage = 4096;
constexpr uint64_t kStep = 8ull << 20;      // a segment starts every kStep bytes...
constexpr uint64_t kSegment = 16ull << 20;  // ...and covers kSegment
constexpr uint64_t kPagesPerSegment = kSegment / kPage;

struct Segment {
    WGPUBuffer buffer = nullptr;
    uint64_t base = 0, size = 0;
    std::vector<uint64_t> valid;   // one bit per page
    int64_t batch = -1;            // the last batch that bound it
    int64_t last_frame = 0;
    bool page_valid(uint64_t p) const { return valid[p >> 6] >> (p & 63) & 1; }
    void set(uint64_t p) { valid[p >> 6] |= 1ull << (p & 63); }
    void clear(uint64_t p) { valid[p >> 6] &= ~(1ull << (p & 63)); }
};

std::vector<Segment*> segments;   // by base / kStep
int64_t batch_id;
bool watch_ok = true;
#if defined(A64_WRITE_TRACK) && !defined(__EMSCRIPTEN__)
extern "C" uint8_t a64_dirty[1u << 20];   // (rt/a64_rt.c)
#endif
// when each guest page was last written, in memory_seq() ticks (one per scan and per command-processor write)
std::vector<uint64_t> page_seq;
uint64_t mem_seq = 1;

std::vector<uint8_t> gpu_pages;   // pages a dispatch wrote (newer on the GPU than in guest memory)
std::atomic<const uint8_t*> gpu_pages_data{nullptr};   // (its storage, for rx_gpu_written_hint: allocated once)
// ...and what guest memory held in each when the dispatch wrote it: when the CPU writes such a page too, only the
// bytes it changed go to the GPU (the dispatch's results in the rest of the page stay). RX_NO_PAGE_MERGE=1: the
// whole page comes from guest memory again (the old behaviour: the GPU's results in it are lost).
std::unordered_map<uint64_t, std::unique_ptr<uint8_t[]>> gpu_page_shadow;
// (the render thread and the readbacks' map callbacks, which write the GPU's results into guest memory and the
// shadows together: readback_to_guest)
std::mutex shadow_mu;

bool page_merge_on() {
    static const bool off = getenv("RX_NO_PAGE_MERGE") != nullptr;
    return !off;
}

void page_written(uint64_t page) {
    if (page_seq.empty()) page_seq.assign(hos_guest_size() / kPage, 0);
    if (page < page_seq.size()) page_seq[page] = mem_seq;
    if (page < gpu_pages.size() && gpu_pages[page]) {
        count("cpu_wrote_gpu_page");
        std::lock_guard lk(shadow_mu);
        gpu_page_shadow.erase(page);
    }
    if (page < gpu_pages.size()) gpu_pages[page] = 0;
    textures_written(page * kPage, kPage);
}

Segment* segment(uint64_t index, bool create) {
    if (segments.empty()) segments.resize(hos_guest_size() / kStep + 1, nullptr);
    if (index >= segments.size()) return nullptr;
    Segment* s = segments[index];
    if (s || !create) return s;
    s = new Segment();
    s->base = index * kStep;
    s->size = std::min<uint64_t>(kSegment, hos_guest_size() - s->base);
    s->valid.assign((kPagesPerSegment + 63) / 64, 0);
    WGPUBufferDescriptor d{};
    d.label = sv("rx guest memory");
    d.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Index | WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
              WGPUBufferUsage_CopySrc;
    d.size = s->size;
    scope_push();
    s->buffer = wgpuDeviceCreateBuffer(dev.device, &d);
    if (const std::string e = scope_pop(); !e.empty()) {
        log_limited("segment", 4, "rx: guest memory segment not created: %s", e.c_str());
        delete s;
        return nullptr;
    }
    segments[index] = s;
    count("segments_new");
    return s;
}

/// The guest page was written: no segment holds it any more.
void invalidate_page(uint64_t page) {
    const uint64_t addr = page * kPage, index = addr / kStep;
    for (uint64_t i : {index, index - 1}) {
        if (i >= segments.size()) continue;
        if (Segment* s = segments[i]; s && addr >= s->base && addr < s->base + s->size) s->clear((addr - s->base) / kPage);
    }
}

/// The CPU (or the command processor) wrote a page a dispatch also wrote: the words it changed since the dispatch
/// go to every segment holding the page, the dispatch's results stay. False when the page is not such a page.
bool merge_cpu_write(uint64_t page) {
    if (!page_merge_on() || page >= gpu_pages.size() || !gpu_pages[page]) return false;
    // the words the CPU changed, taken under the shadows' lock (a readback writes guest memory and the shadow at once)
    static std::vector<std::pair<uint32_t, uint32_t>> runs;
    static uint32_t words[kPage / 4];
    runs.clear();
    {
        std::lock_guard lk(shadow_mu);
        auto it = gpu_page_shadow.find(page);
        if (it == gpu_page_shadow.end()) return false;
        const uint32_t* now = reinterpret_cast<const uint32_t*>(hos_guest_base() + page * kPage);
        uint32_t* old = reinterpret_cast<uint32_t*>(it->second.get());
        for (uint32_t w = 0; w < kPage / 4;) {
            const uint32_t v = now[w];
            if (v == old[w]) {
                w++;
                continue;
            }
            uint32_t e = w;
            for (uint32_t x; e < kPage / 4 && (x = now[e]) != old[e]; e++) words[e] = old[e] = x;
            runs.push_back({w, e});
            w = e;
        }
    }
    const uint64_t addr = page * kPage, index = addr / kStep;
    for (const auto& [w, e] : runs) {
        for (uint64_t i : {index, index - 1}) {
            if (i >= segments.size()) continue;
            Segment* sg = segments[i];
            if (!sg || addr < sg->base || addr >= sg->base + sg->size || !sg->page_valid((addr - sg->base) / kPage))
                continue;
            if (sg->batch == batch_id) {   // (a draw of the open batch may read the old words)
                count("flush_written");
                flush_batch();
            }
            queue_upload(sg->buffer, addr - sg->base + w * 4, words + w, (uint64_t)(e - w) * 4);
        }
        count("merged_words", (int64_t)(e - w));
    }
    count("merged_pages");
    return true;
}


bool holds_range(const Segment& s, uint64_t guest, uint64_t size) {
    const uint64_t first = (guest - s.base) / kPage, last = (guest + size - 1 - s.base) / kPage;
    for (uint64_t w = first >> 6; w <= last >> 6; w++) {
        uint64_t need = ~0ull;
        if (w == first >> 6) need &= ~0ull << (first & 63);
        if (w == last >> 6) need &= ~0ull >> (63 - (last & 63));
        if ((s.valid[w] & need) != need) return false;
    }
    return true;
}

bool range_gpu_written(uint64_t guest, uint64_t size) {
    if (gpu_pages.empty()) return false;
    for (uint64_t p = guest / kPage; p <= (guest + size - 1) / kPage && p < gpu_pages.size(); p++)
        if (gpu_pages[p]) return true;
    return false;
}

bool holds_gpu_pages(const Segment& s, uint64_t guest, uint64_t size) {
    for (uint64_t p = guest / kPage; p <= (guest + size - 1) / kPage && p < gpu_pages.size(); p++)
        if (gpu_pages[p] && !s.page_valid(p - s.base / kPage)) return false;
    return true;
}

Segment* gpu_page_holder(const Segment& s, uint64_t page) {
    const uint64_t a = page * kPage, index = s.base / kStep;
    for (uint64_t i : {index - 1, index + 1}) {
        if (i >= segments.size() || !segments[i]) continue;
        Segment* n = segments[i];
        if (a >= n->base && a < n->base + n->size && n->page_valid((a - n->base) / kPage)) return n;
    }
    return nullptr;
}

void fill_gpu_pages(Segment& s, uint64_t p0, uint64_t p1) {
    if (gpu_pages.empty()) return;
    const uint64_t g0 = s.base / kPage;
    for (uint64_t p = p0; p < p1;) {
        const uint64_t gp = g0 + p;
        Segment* from = gp < gpu_pages.size() && gpu_pages[gp] ? gpu_page_holder(s, gp) : nullptr;
        if (!from) {
            p++;
            continue;
        }
        uint64_t q = p + 1;
        while (q < p1 && g0 + q < gpu_pages.size() && gpu_pages[g0 + q] && gpu_page_holder(s, g0 + q) == from) q++;
        end_render_pass("pe_segment_fill");
        const uint64_t a = gp * kPage;
        ce_copy_buffer_to_buffer(encoder(), from->buffer, a - from->base, s.buffer, a - s.base, (q - p) * kPage);
        from->batch = batch_id;
        count("segment_gpu_fill", (int64_t)((q - p) * kPage));
        p = q;
    }
}

} // namespace

std::vector<WGPUBuffer> retired_segment_buffers;

void retire_unused_segments() {
    static const bool on = env_flag("RX_SEGMENT_RETIRE");
    if (!on) return;
    int retired = 0;
    for (size_t i = 0; i < segments.size(); i++) {
        Segment* s = segments[i];
        if (!s || s->last_frame >= frame_number - 1800 || s->batch >= batch_id) continue;
        bool gpu_data = false;
        const uint64_t p0 = s->base / kPage, p1 = std::min<uint64_t>((s->base + s->size) / kPage, gpu_pages.size());
        for (uint64_t p = p0; p < p1 && !gpu_data; p++) gpu_data = gpu_pages[p] != 0;
        if (gpu_data) continue;
        retired_segment_buffers.push_back(s->buffer);
        delete s;
        segments[i] = nullptr;
        retired++;
    }
    if (retired) count("segments_retired", retired);
}

bool segments_retired_pending() {
    return !retired_segment_buffers.empty();
}

std::vector<WGPUBuffer> take_retired_segment_buffers() {
    std::vector<WGPUBuffer> out;
    out.swap(retired_segment_buffers);
    return out;
}

size_t buffer_segments() {
    size_t n = 0;
    for (const Segment* s : segments) n += s != nullptr;
    return n;
}

}  // namespace rx

/// (HOS_MEM_MAP) Whether a segment holds the guest page at this offset (its contents were read for an upload).
extern "C" int rx_guest_page_uploaded(uint64_t off) {
    using namespace rx;
    const uint64_t index = off / kStep;
    for (uint64_t i : {index, index - 1}) {
        if (i >= segments.size()) continue;
        if (const Segment* s = segments[i]; s && off >= s->base && off < s->base + s->size &&
                                            s->page_valid((off - s->base) / kPage))
            return 1;
    }
    return 0;
}

namespace rx {

/// The open batch was submitted: segments bound from now on belong to the next one.
void buffer_cache_new_batch() {
    batch_id++;
}

BufferRef guest_buffer(uint64_t gpu_va, uint64_t size) {
    if (!size || size > kStep) return count("gb_size"), BufferRef{};
    uint64_t avail = 0;
    const uint8_t* host = hos_gpu_ptr(gpu_va, &avail);
    if (!host || avail < size) return count(!host ? "gb_unmapped" : "gb_avail"), BufferRef{};
    const uint64_t guest = (uint64_t)((uintptr_t)host - (uintptr_t)hos_guest_base());
    const uint64_t index = guest / kStep;
    Segment* s = nullptr;
    bool held = false;
    static const bool overlap = env_flag("RX_SEGMENT_OVERLAP");
    if (overlap) {
        s = segment(index, true);
        held = s && guest + size <= s->base + s->size && holds_range(*s, guest, size);
    } else {
        const bool lower_fits = index > 0 && guest + size <= index * kStep + kStep;
        const uint64_t none = UINT64_MAX;
        const uint64_t pick[2] = {(index & 1) ? (lower_fits ? index - 1 : none) : index,
                                  (index & 1) ? index : (lower_fits ? index - 1 : none)};
        Segment* have[2] = {};
        for (int k = 0; k < 2 && !s; k++) {
            if (pick[k] == none) continue;
            Segment* c = segment(pick[k], false);
            if (!c || guest + size > c->base + c->size) continue;
            have[k] = c;
            if (holds_range(*c, guest, size)) s = c, held = true;
        }
        if (!s && (have[0] || have[1]) && range_gpu_written(guest, size))
            for (int k = 0; k < 2 && !s; k++)
                if (have[k] && holds_gpu_pages(*have[k], guest, size)) s = have[k];
        if (!s) s = have[0] ? have[0] : have[1];
        if (!s) s = segment(pick[0] != none ? pick[0] : pick[1], true);
    }
    if (!s || guest + size > s->base + s->size) return count(!s ? "gb_no_segment" : "gb_bounds"), BufferRef{};
    // upload the pages the range needs that the segment does not hold (in runs)
    const uint64_t first = (guest - s->base) / kPage, last = (guest + size - 1 - s->base) / kPage;
    if (held) {
        s->batch = batch_id;
        s->last_frame = frame_number;
        return {s->buffer, guest - s->base, size};
    }
    for (uint64_t p = first; p <= last;) {
        if (s->page_valid(p)) {
            p++;
            continue;
        }
        uint64_t q = p;
        while (q <= last && !s->page_valid(q)) s->set(q++);
        const uint64_t off = p * kPage, n = std::min((q - p) * kPage, s->size - off);
        queue_upload(s->buffer, off, hos_guest_base() + s->base + off, n);
        count("up_cache", (int64_t)n);
        fill_gpu_pages(*s, p, q);
        p = q;
    }
    s->batch = batch_id;
    s->last_frame = frame_number;
    return {s->buffer, guest - s->base, size};
}

/// The guest pages the CPU wrote since the last call (GetWriteWatch, which also resets them): any thread. False when
/// write watching does not work (then nothing is tracked).
bool collect_written_pages(std::vector<uint32_t>& out) {
    out.clear();
#if defined(A64_WRITE_TRACK) && !defined(__EMSCRIPTEN__)
    {   // the soft write watch (rt/a64.h A64_MARK, hos_mark_written): the marks of the GPU-visible ranges, read and
        // cleared (an exchange: a mark set again meanwhile is the next scan's); ~0.1 ms where GetWriteWatch took 2-50
        std::vector<std::array<uint64_t, 2>> range_buf(4096);
        auto* ranges = reinterpret_cast<uint64_t(*)[2]>(range_buf.data());
        const int nranges = hos_gpu_guest_ranges(ranges, 4096);
        for (int r = 0; r < nranges; r++) {
            const uint64_t lo = ranges[r][0], hi = std::min<uint64_t>(ranges[r][1], hos_guest_size());
            const uint64_t p1 = std::min<uint64_t>((hi + kPage - 1) / kPage, 1u << 20);
            for (uint64_t p = lo / kPage; p < p1;) {
                if (!(p & 7) && p + 8 <= p1) {   // (eight pages a load while none is marked)
                    uint64_t w;
                    std::memcpy(&w, &a64_dirty[p], 8);
                    if (!w) {
                        p += 8;
                        continue;
                    }
                }
                if (__atomic_load_n(&a64_dirty[p], __ATOMIC_RELAXED) && __atomic_exchange_n(&a64_dirty[p], 0, __ATOMIC_ACQUIRE))
                    out.push_back((uint32_t)p);
                p++;
            }
        }
        return true;
    }
#endif
    if (!watch_ok) return false;
    // only the memory mapped into the GPU's address space (what the renderer can read); the cost of GetWriteWatch
    // grows with the range, not with the pages written
    std::vector<std::array<uint64_t, 2>> range_buf(4096);   // (any thread may scan)
    auto* ranges = reinterpret_cast<uint64_t(*)[2]>(range_buf.data());
    const int nranges = hos_gpu_guest_ranges(ranges, 4096);
    // (room for 1 GB of written pages a call: each extra call walks the whole range again)
    thread_local std::vector<void*> addrs(1 << 18);
    const int64_t t0 = hos_now_ns();
    int calls = 0;
    int64_t slowest = 0, slowest_range = 0;
    for (int r = 0; r < nranges; r++) {
        const uint64_t lo = ranges[r][0], hi = std::min<uint64_t>(ranges[r][1], hos_guest_size());
        if (lo >= hi) continue;
        // RX_SCAN_PROBE=1 (experiment): a range is first asked without a reset (no TLB flush) for one written page;
        // only a range that has one is read and reset (atomically, as before)
        static const bool probe = env_flag("RX_SCAN_PROBE");
        if (probe) {
            void* one[1];
            ULONG_PTR n1 = 1;
            DWORD gran1 = 0;
            const int64_t c0 = hos_now_ns();
            const UINT bad = GetWriteWatch(0, hos_guest_base() + lo, hi - lo, one, &n1, &gran1);
            calls++;
            if (hos_now_ns() - c0 > slowest) slowest = hos_now_ns() - c0, slowest_range = (int64_t)(hi - lo);
            if (!bad && n1 == 0) continue;
        }
        for (;;) {
            ULONG_PTR n = addrs.size();
            DWORD gran = 0;
            const int64_t c0 = hos_now_ns();
            if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, hos_guest_base() + lo, hi - lo, addrs.data(), &n, &gran)) {
                hos_log("rx: GetWriteWatch failed (%lu): every buffer is uploaded at each draw", GetLastError());
                watch_ok = false;
                return false;
            }
            calls++;
            if (hos_now_ns() - c0 > slowest) slowest = hos_now_ns() - c0, slowest_range = (int64_t)(hi - lo);
            for (ULONG_PTR i = 0; i < n; i++)
                out.push_back((uint32_t)(((uintptr_t)addrs[i] - (uintptr_t)hos_guest_base()) / kPage));
            if (n < addrs.size()) break;
        }
    }
    if (const int64_t took = hos_now_ns() - t0; took > 20000000)   // (a hitch: the game waits for this at its fences)
        log_limited("slow_scan", 60, "rx: write scan took %.1f ms: %zu written pages, %d calls over %d ranges, slowest "
                    "%.1f ms (%lld MB)", took / 1e6, out.size(), calls, nranges, slowest / 1e6,
                    (long long)(slowest_range >> 20));
    static int64_t logged_at;
    const int64_t now = hos_now_ns();
    if (now - logged_at > 30000000000LL) {   // (every 30 s: how much is watched)
        logged_at = now;
        uint64_t total = 0;
        for (int r = 0; r < nranges; r++) total += ranges[r][1] - ranges[r][0];
        hos_log("rx: change tracking watches %d ranges, %llu MB", nranges, (unsigned long long)(total >> 20));
    }
    return true;
}

/// Pages the CPU wrote (collect_written_pages) are invalidated (called with no batch open).
void buffer_cache_apply_written(const uint32_t* pages, size_t n) {
    mem_seq++;
    for (size_t i = 0; i < n; i++) {
        if (merge_cpu_write(pages[i])) {   // (the page stays valid and GPU-written)
            if (page_seq.empty()) page_seq.assign(hos_guest_size() / kPage, 0);
            if (pages[i] < page_seq.size()) page_seq[pages[i]] = mem_seq;
            textures_written((uint64_t)pages[i] * kPage, kPage);
            continue;
        }
        page_written(pages[i]);
        if (!segments.empty()) invalidate_page(pages[i]);
    }
    count("dirty_pages", (int64_t)n);
}

/// Pages the CPU wrote since the last call are invalidated (called with no batch open).
void buffer_cache_scan() {
    static std::vector<uint32_t> pages;
    const int64_t t0 = hos_now_ns();
    if (!collect_written_pages(pages)) return;
    count("t_scan", hos_now_ns() - t0);
    buffer_cache_apply_written(pages.data(), pages.size());
}

void buffer_cache_written(uint64_t gpu_va, uint64_t size) {
    if (!size) return;
    uint64_t avail = 0;
    const uint8_t* host = hos_gpu_ptr(gpu_va, &avail);
    if (!host) return;
    const uint64_t guest = (uint64_t)((uintptr_t)host - (uintptr_t)hos_guest_base());
    const uint64_t p0 = guest / kPage, p1 = (guest + std::min(size, avail) - 1) / kPage;
    mem_seq++;
    static std::vector<uint8_t> merged;   // (reused: called for every constant buffer upload and semaphore write)
    merged.assign(p1 - p0 + 1, 0);
    for (uint64_t p = p0; p <= p1; p++) {
        if (merge_cpu_write(p)) {
            merged[p - p0] = 1;
            if (page_seq.empty()) page_seq.assign(hos_guest_size() / kPage, 0);
            if (p < page_seq.size()) page_seq[p] = mem_seq;
            textures_written(p * kPage, kPage);
            continue;
        }
        page_written(p);
    }
    if (segments.empty()) return;
    for (uint64_t p = p0; p <= p1; p++) {
        if (merged[p - p0]) continue;
        const uint64_t addr = p * kPage, index = addr / kStep;
        for (uint64_t i : {index, index - 1}) {
            if (i >= segments.size()) continue;
            Segment* s = segments[i];
            if (!s || addr < s->base || addr >= s->base + s->size) continue;
            const uint64_t sp = (addr - s->base) / kPage;
            if (!s->page_valid(sp)) continue;
            // a draw of the open batch may read the old contents: they go to the GPU first
            if (s->batch == batch_id) {
                count("flush_written");
                flush_batch();
            }
            s->clear(sp);
        }
    }
}

/// A readback's results written into guest memory (a map callback: any thread), the words that changed in the batch
/// (now against was). They are the GPU's own results, not CPU writes: a page's shadow takes them too, so the next scan
/// does not upload them over what the GPU wrote since. (They were merged as CPU writes: TOTK's lighting tile flags,
/// rewritten every frame, came back a frame late over the newer ones and lit some tiles wrongly: light boxes at the
/// horizon that moved from frame to frame.)
int64_t readback_to_guest(uint8_t* guest, const uint8_t* now, const uint8_t* was, uint64_t n) {
    int64_t changed = 0;
    std::lock_guard lk(shadow_mu);
    uint64_t shadow_page = ~0ull;
    uint8_t* shadow = nullptr;
    for (uint64_t i = 0; i + 4 <= n; i += 4) {
        uint32_t v, before;
        std::memcpy(&v, now + i, 4);
        std::memcpy(&before, was + i, 4);
        if (v == before) continue;
        std::memcpy(guest + i, &v, 4);
        changed++;
        const uint64_t at = (uint64_t)(guest + i - hos_guest_base()), page = at / kPage;
        if (page != shadow_page) {
            shadow_page = page;
            auto it = gpu_page_shadow.find(page);
            shadow = it == gpu_page_shadow.end() ? nullptr : it->second.get();
        }
        if (shadow) std::memcpy(shadow + at % kPage, &v, 4);
    }
    return changed;
}

void buffer_cache_gpu_written(const BufferRef& ref) {
    Segment* s = nullptr;
    for (Segment* c : segments)   // (few segments: a lookup by buffer is not worth a map)
        if (c && c->buffer == ref.buffer) {
            s = c;
            break;
        }
    if (!s) return;
    const uint64_t g0 = s->base + ref.offset, g1 = g0 + ref.size;
    if (gpu_pages.empty()) {
        gpu_pages.assign(hos_guest_size() / kPage, 0);
        gpu_pages_data.store(gpu_pages.data(), std::memory_order_release);
    }
    for (uint64_t p = g0 / kPage; p <= (g1 - 1) / kPage && p < gpu_pages.size(); p++) {
        if (!gpu_pages[p] && page_merge_on()) {   // (what the CPU's side holds now: later CPU writes are diffed against it)
            std::lock_guard lk(shadow_mu);
            auto& sh = gpu_page_shadow[p];
            if (!sh) sh.reset(new uint8_t[kPage]);
            std::memcpy(sh.get(), hos_guest_base() + p * kPage, kPage);
        }
        gpu_pages[p] = 1;
    }
    const uint64_t index = s->base / kStep;
    for (uint64_t i : {index - 1, index + 1}) {
        if (i >= segments.size() || !segments[i]) continue;
        Segment* n = segments[i];
        const uint64_t a = std::max(g0, n->base), b = std::min(g1, n->base + n->size);
        if (a >= b) continue;
        // pages of the overlap the neighbour does not hold come from guest memory first (queue order: before the
        // batch), then the written bytes are copied over them in the batch, after the dispatch
        for (uint64_t p = (a - n->base) / kPage; p <= (b - 1 - n->base) / kPage; p++)
            if (!n->page_valid(p)) {
                const uint64_t off = p * kPage, len = std::min(kPage, n->size - off);
                queue_upload(n->buffer, off, hos_guest_base() + n->base + off, len);
                n->set(p);
            }
        ce_copy_buffer_to_buffer(encoder(), s->buffer, a - s->base, n->buffer, a - n->base, b - a);
        n->batch = batch_id;
    }
}

uint64_t memory_seq() {
    return watch_ok ? mem_seq : 0;
}

bool memory_written_since(uint64_t guest, uint64_t size, uint64_t seq) {
    if (page_seq.empty() || !size) return false;
    const uint64_t p1 = std::min<uint64_t>((guest + size - 1) / kPage, page_seq.size() - 1);
    for (uint64_t p = guest / kPage; p <= p1; p++)
        if (page_seq[p] > seq) return true;
    return false;
}

bool gpu_written(uint64_t gpu_va, uint64_t size) {
    if (gpu_pages.empty() || !size) return false;
    uint64_t avail = 0;
    const uint8_t* host = hos_gpu_ptr(gpu_va, &avail);
    if (!host) return false;
    const uint64_t guest = (uint64_t)((uintptr_t)host - (uintptr_t)hos_guest_base());
    for (uint64_t p = guest / kPage; p <= (guest + size - 1) / kPage && p < gpu_pages.size(); p++)
        if (gpu_pages[p]) return true;
    return false;
}

void buffer_cache_sync_to_guest(uint64_t gpu_va, uint64_t size) {
    if (gpu_pages.empty() || !size) return;
    uint64_t avail = 0;
    const uint8_t* host = hos_gpu_ptr(gpu_va, &avail);
    if (!host) return;
    const uint64_t guest = (uint64_t)((uintptr_t)host - (uintptr_t)hos_guest_base());
    uint64_t p0 = guest / kPage, p1 = (guest + std::min(size, avail) - 1) / kPage;
    // the segment holding the range (a dispatch's results are in every segment that holds its pages)
    Segment* s = nullptr;
    for (uint64_t i : {guest / kStep, guest / kStep - 1})
        if (i < segments.size() && segments[i] && guest >= segments[i]->base &&
            (p1 + 1) * kPage <= segments[i]->base + segments[i]->size && segments[i]->page_valid(p0 - segments[i]->base / kPage)) {
            s = segments[i];
            break;
        }
    if (!s) return;
    // with the GPU-written pages around it (up to 1 MB each way): one round trip for the neighbouring arguments
    const uint64_t s0 = s->base / kPage, s1 = (s->base + s->size) / kPage - 1;
    for (int k = 0; k < 256 && p0 > s0 && gpu_pages[p0 - 1] && s->page_valid(p0 - 1 - s0); k++) p0--;
    for (int k = 0; k < 256 && p1 < s1 && gpu_pages[p1 + 1] && s->page_valid(p1 + 1 - s0); k++) p1++;
    const uint64_t off = p0 * kPage - s->base, n = (p1 - p0 + 1) * kPage;
    WGPUBufferDescriptor d{};
    d.label = sv("rx guest memory readback");
    d.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    d.size = n;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(dev.device, &d);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(e, s->buffer, off, staging, 0, n);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(e);
    struct Done {
        bool complete = false;
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    } done;
    WGPUBufferMapCallbackInfo mc{};
    mc.mode = WGPUCallbackMode_AllowSpontaneous;
    mc.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        auto* dn = static_cast<Done*>(u1);
        dn->status = status;
        dn->complete = true;
    };
    mc.userdata1 = &done;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, n, mc);
    while (!done.complete) wgpuDevicePoll(dev.device, true, nullptr);
    if (done.status == WGPUMapAsyncStatus_Success) {
        uint8_t* dst = hos_guest_base() + p0 * kPage;
        // (pages the CPU wrote since the last scan keep their write-watch mark; the others are not reported as CPU
        // writes: the segment already holds this)
        std::vector<void*> dirty(p1 - p0 + 1);
        ULONG_PTR ndirty = dirty.size();
        DWORD gran = 0;
        const bool watched = !GetWriteWatch(0, dst, n, dirty.data(), &ndirty, &gran);
        const auto* src = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(staging, 0, n));
        std::lock_guard lk(shadow_mu);
        for (uint64_t p = p0; p <= p1; p++) {
            uint8_t* g = dst + (p - p0) * kPage;
            const uint8_t* q = src + (p - p0) * kPage;
            auto it = gpu_page_shadow.find(p);
            if (it == gpu_page_shadow.end()) {
                std::memcpy(g, q, kPage);
                continue;
            }
            // only the words the GPU changed: what the CPU wrote since (not scanned yet) stays
            const uint32_t* gw = reinterpret_cast<const uint32_t*>(q);
            const uint32_t* sw = reinterpret_cast<const uint32_t*>(it->second.get());
            uint32_t* dw = reinterpret_cast<uint32_t*>(g);
            for (uint32_t w = 0; w < kPage / 4; w++)
                if (gw[w] != sw[w]) dw[w] = gw[w];
        }
        if (watched && ndirty == 0) ResetWriteWatch(dst, n);
        wgpuBufferUnmap(staging);
        for (uint64_t p = p0; p <= p1; p++) {
            gpu_pages[p] = 0;
            gpu_page_shadow.erase(p);
        }
        count("gpu_sync_bytes", (int64_t)n);
    }
    wgpuBufferRelease(staging);
}

namespace {
std::vector<std::pair<uint64_t, uint64_t>> dispatch_writes;
}
void note_dispatch_write(uint64_t gpu_va, uint64_t size) {
    static const bool on = getenv("RX_SYNC_DISPATCH_WRITES") != nullptr;
    if (!on) return;
    for (auto& [v, n] : dispatch_writes)
        if (v == gpu_va) { n = std::max(n, size); return; }
    dispatch_writes.push_back({gpu_va, size});
}
void sync_dispatch_writes() {
    if (dispatch_writes.empty()) return;
    flush_batch();
    wait_submitted();
    for (const auto& [v, n] : dispatch_writes) buffer_cache_sync_to_guest(v, n);
    count("dispatch_writes_synced", (int64_t)dispatch_writes.size());
    dispatch_writes.clear();
}

int64_t batch_number() {
    return batch_id;
}

bool buffer_cache_active() {
    static const bool off = getenv("RX_NO_CACHE") != nullptr;   // (debugging: every draw copies its data)
    return watch_ok && !off;
}

} // namespace rx

/// Whether a dispatch may have written this memory, without the renderer's lock (the parsing thread asks for every
/// command list; a byte per page, read while the render thread may set it: a stale answer only delays a sync).
extern "C" int rx_gpu_written_hint(uint64_t gpu_va, uint64_t size) {
    const uint8_t* pages = rx::gpu_pages_data.load(std::memory_order_acquire);
    if (!pages || !size) return 0;
    uint64_t avail = 0;
    const uint8_t* host = hos_gpu_ptr(gpu_va, &avail);
    if (!host) return 0;
    const uint64_t guest = (uint64_t)((uintptr_t)host - (uintptr_t)hos_guest_base()), n = hos_guest_size() / rx::kPage;
    for (uint64_t p = guest / rx::kPage; p <= (guest + size - 1) / rx::kPage && p < n; p++)
        if (reinterpret_cast<const volatile uint8_t*>(pages)[p]) return 1;
    return 0;
}
