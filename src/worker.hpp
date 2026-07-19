#pragma once

#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace aldous_tsp::detail {

// Executes one indexed task per worker. Any exception raised by a task is
// captured inside the thread boundary, requests cancellation for workers that
// have not entered their task yet, and is rethrown only after every created
// thread has been joined. Thread-construction failures follow the same rule.
template <class Function>
void run_parallel_indexed(int count, Function&& function) {
    if (count <= 0) {
        return;
    }
    if (count == 1) {
        function(0);
        return;
    }

    std::atomic<bool> cancelled{false};
    std::mutex exception_mutex;
    std::exception_ptr first_exception;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(count));

    auto capture_exception = [&](std::exception_ptr exception) {
        {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (first_exception == nullptr) {
                first_exception = std::move(exception);
            }
        }
        cancelled.store(true, std::memory_order_release);
    };

    try {
        for (int index = 0; index < count; ++index) {
            workers.emplace_back([&, index]() {
                if (cancelled.load(std::memory_order_acquire)) {
                    return;
                }
                try {
                    function(index);
                } catch (...) {
                    capture_exception(std::current_exception());
                }
            });
        }
    } catch (...) {
        capture_exception(std::current_exception());
    }

    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

} // namespace aldous_tsp::detail
