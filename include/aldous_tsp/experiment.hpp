#pragma once

#include "aldous_tsp/results.hpp"

#include <functional>

namespace aldous_tsp {

struct ExperimentProgress {
    int completed = 0;
    int total = 0;
    int instance_index = -1;
    double instance_seconds = 0.0;
    double elapsed_seconds = 0.0;
    double eta_seconds = 0.0;
};

using ExperimentProgressCallback = std::function<void(const ExperimentProgress&)>;

class ExperimentRunner {
public:
    explicit ExperimentRunner(RunOptions options);

    [[nodiscard]] const RunOptions& options() const noexcept { return options_; }
    [[nodiscard]] ResultsDocument run(const ExperimentProgressCallback& progress = {}) const;

private:
    RunOptions options_;
};

} // namespace aldous_tsp
