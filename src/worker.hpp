#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
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

enum class WorkerState : unsigned char {
    NotStarted,
    Running,
    Success,
    Failure,
    Cancelled,
};

template <class Value>
struct WorkerOutcome {
    WorkerState state = WorkerState::NotStarted;
    std::optional<Value> value;
    std::exception_ptr exception;
};

struct WorkerSummary {
    int attempted = 0;
    int succeeded = 0;
    int failed = 0;
    int cancelled = 0;
};

// Runs a dynamically scheduled indexed work queue. Work executes on worker
// threads, while `completion` executes exclusively on the caller thread for
// each successful task. A work or completion exception requests cooperative
// cancellation, every worker is joined, all slots receive a terminal state,
// and only then is the first exception rethrown.
template <class Value, class Work, class Completion>
WorkerSummary run_parallel_work_queue(
    int task_count,
    int requested_threads,
    std::vector<WorkerOutcome<Value>>& outcomes,
    Work&& work,
    Completion&& completion) {
    outcomes.clear();
    if (task_count <= 0) {
        return {};
    }
    outcomes.resize(static_cast<std::size_t>(task_count));

    const int thread_count = std::max(1, std::min(requested_threads, task_count));
    std::atomic<int> next{0};
    std::atomic<int> live_workers{0};
    std::atomic<bool> cancelled{false};

    std::mutex exception_mutex;
    std::exception_ptr first_exception;
    auto capture_exception = [&](std::exception_ptr exception) {
        {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (first_exception == nullptr) {
                first_exception = std::move(exception);
            }
        }
        cancelled.store(true, std::memory_order_release);
    };

    // Pre-reserve one completion slot per task so worker-side publication is
    // allocation-free after the threads have started.
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    std::vector<int> completion_indices;
    completion_indices.reserve(static_cast<std::size_t>(task_count));
    std::size_t completions_consumed = 0;

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (int worker_index = 0; worker_index < thread_count; ++worker_index) {
        live_workers.fetch_add(1, std::memory_order_relaxed);
        try {
            workers.emplace_back([&]() {
                int current_index = -1;
                try {
                    for (;;) {
                        if (cancelled.load(std::memory_order_acquire)) {
                            break;
                        }
                        const int index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= task_count) {
                            break;
                        }
                        current_index = index;
                        WorkerOutcome<Value>& outcome =
                            outcomes[static_cast<std::size_t>(index)];
                        outcome.state = WorkerState::Running;
                        try {
                            outcome.value.emplace(work(index));
                            outcome.state = WorkerState::Success;
                            {
                                std::lock_guard<std::mutex> lock(completion_mutex);
                                completion_indices.push_back(index);
                            }
                            completion_cv.notify_one();
                            current_index = -1;
                        } catch (...) {
                            outcome.value.reset();
                            outcome.exception = std::current_exception();
                            outcome.state = WorkerState::Failure;
                            current_index = -1;
                            capture_exception(outcome.exception);
                            break;
                        }
                    }
                } catch (...) {
                    const std::exception_ptr exception = std::current_exception();
                    if (current_index >= 0) {
                        WorkerOutcome<Value>& outcome =
                            outcomes[static_cast<std::size_t>(current_index)];
                        outcome.value.reset();
                        outcome.exception = exception;
                        outcome.state = WorkerState::Failure;
                    }
                    capture_exception(exception);
                }
                live_workers.fetch_sub(1, std::memory_order_release);
                completion_cv.notify_all();
            });
        } catch (...) {
            live_workers.fetch_sub(1, std::memory_order_release);
            capture_exception(std::current_exception());
            completion_cv.notify_all();
            break;
        }
    }

    bool invoke_completion = true;
    for (;;) {
        int completed_index = -1;
        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            completion_cv.wait(lock, [&]() {
                return completions_consumed < completion_indices.size()
                    || live_workers.load(std::memory_order_acquire) == 0;
            });
            if (completions_consumed < completion_indices.size()) {
                completed_index = completion_indices[completions_consumed++];
            } else if (live_workers.load(std::memory_order_acquire) == 0) {
                break;
            }
        }

        if (invoke_completion && completed_index >= 0) {
            const WorkerOutcome<Value>& outcome =
                outcomes[static_cast<std::size_t>(completed_index)];
            try {
                completion(completed_index, *outcome.value);
            } catch (...) {
                invoke_completion = false;
                capture_exception(std::current_exception());
            }
        }
    }

    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    WorkerSummary summary;
    for (WorkerOutcome<Value>& outcome : outcomes) {
        if (outcome.state == WorkerState::NotStarted) {
            outcome.state = WorkerState::Cancelled;
        }
        switch (outcome.state) {
            case WorkerState::NotStarted:
            case WorkerState::Running:
                // Both are impossible after joining; treat them as cancelled
                // defensively rather than letting an incomplete slot leak into
                // aggregation.
                outcome.state = WorkerState::Cancelled;
                ++summary.cancelled;
                break;
            case WorkerState::Success:
                ++summary.attempted;
                ++summary.succeeded;
                break;
            case WorkerState::Failure:
                ++summary.attempted;
                ++summary.failed;
                break;
            case WorkerState::Cancelled:
                ++summary.cancelled;
                break;
        }
    }

    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
    return summary;
}

} // namespace aldous_tsp::detail
