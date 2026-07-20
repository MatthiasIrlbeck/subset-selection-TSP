#include "aldous_tsp/instance.hpp"
#include "aldous_tsp/validation.hpp"
#include "fuzz_common.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > 32768U) {
        return 0;
    }
    aldous_tsp::fuzz::Reader reader(data, size);
    const int n = reader.bounded_int(3, 48);
    const bool periodic = (reader.byte() & 1U) != 0U;
    const double side = 0.25 + 64.0 * reader.unit();
    std::vector<aldous_tsp::Point> points;
    points.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double scale = periodic ? 4.0 * side : 128.0;
        points.push_back({reader.finite(scale), reader.finite(scale)});
    }

    aldous_tsp::Instance instance;
    instance.periodic = periodic;
    if (periodic) {
        instance.explicit_side = side;
    }
    instance.set_points(std::move(points));
    const int k = reader.bounded_int(1, n - 1);
    const aldous_tsp::KnnBackend backend = (reader.byte() & 1U) != 0U
        ? aldous_tsp::KnnBackend::GridExact
        : aldous_tsp::KnnBackend::BruteForce;
    const double forced_cell = (reader.byte() & 3U) == 0U
        ? 0.0
        : std::max(1.0e-12, reader.unit() * std::max(instance.side, 1.0));
    instance.build_knn(k, backend, forced_cell);

    std::string error;
    if (!aldous_tsp::validate_instance(instance, error)) {
        __builtin_trap();
    }
    for (int sample = 0; sample < std::min(n, 8); ++sample) {
        const int a = reader.bounded_int(0, n - 1);
        const int b = reader.bounded_int(0, n - 1);
        const double ab = instance.dist(a, b);
        const double ba = instance.dist(b, a);
        if (!std::isfinite(ab) || std::fabs(ab - ba) > 1.0e-10) {
            __builtin_trap();
        }
    }
    aldous_tsp::Rng verify_rng(reader.u64());
    if (!instance.verify_knn(std::min(16, n), verify_rng)) {
        __builtin_trap();
    }
    return 0;
}
