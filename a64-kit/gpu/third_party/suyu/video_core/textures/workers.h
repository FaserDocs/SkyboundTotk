// Rexcomp replacement for suyu's video_core/textures/workers.h: the ASTC decoder's work queue. A small pool of worker
// threads (started on first use, never joined), and the thread that waits runs queued jobs too. Decoding ASTC on the
// render thread alone was ~18 ms a frame for seconds after a load. RX_ASTC_THREADS=<n> sets the workers (0: inline).
#pragma once

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace Common {
inline void (*worker_thread_start)() = nullptr;

class ThreadWorker {
public:
    template <typename F>
    void QueueWork(F&& work) {
        start();
        if (!nthreads) {
            work();
            return;
        }
        {
            std::lock_guard lk(mu);
            jobs.emplace_back(std::forward<F>(work));
            pending++;
        }
        more.notify_one();
    }
    void WaitForRequests() {
        std::unique_lock lk(mu);
        for (;;) {
            if (!jobs.empty()) {   // (the waiting thread helps)
                std::function<void()> job = std::move(jobs.front());
                jobs.pop_front();
                lk.unlock();
                job();
                lk.lock();
                if (--pending == 0) done.notify_all();
                continue;
            }
            if (pending == 0) return;
            done.wait(lk);
        }
    }

private:
    void start() {
        std::call_once(once, [this] {
            const char* e = std::getenv("RX_ASTC_THREADS");
            const unsigned hw = std::thread::hardware_concurrency();
            nthreads = e ? (unsigned)std::atoi(e) : hw > 4 ? hw / 3 : 1;
            for (unsigned i = 0; i < nthreads; i++) std::thread([this] { if (worker_thread_start) worker_thread_start(); run(); }).detach();
        });
    }
    void run() {
        std::unique_lock lk(mu);
        for (;;) {
            more.wait(lk, [this] { return !jobs.empty(); });
            std::function<void()> job = std::move(jobs.front());
            jobs.pop_front();
            lk.unlock();
            job();
            lk.lock();
            if (--pending == 0) done.notify_all();
        }
    }
    std::once_flag once;
    unsigned nthreads = 0;
    std::mutex mu;
    std::condition_variable more, done;
    std::deque<std::function<void()>> jobs;
    unsigned pending = 0;
};
} // namespace Common

namespace Tegra::Texture {
inline Common::ThreadWorker& GetThreadWorkers() {
    static Common::ThreadWorker* workers = new Common::ThreadWorker();   // (never destroyed: workers outlive exit)
    return *workers;
}
} // namespace Tegra::Texture
