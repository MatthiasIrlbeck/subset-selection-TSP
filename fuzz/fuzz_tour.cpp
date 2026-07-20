#include "aldous_tsp/tour.hpp"
#include "fuzz_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > 32768U) {
        return 0;
    }
    aldous_tsp::fuzz::Reader reader(data, size);
    const int n = reader.bounded_int(3, 64);
    aldous_tsp::Rng point_rng(reader.u64());
    aldous_tsp::Instance instance;
    instance.periodic = (reader.byte() & 1U) != 0U;
    instance.generate(n, point_rng);
    instance.build_knn(std::min(12, n - 1));

    const int k = reader.bounded_int(3, n);
    std::vector<int> nodes(static_cast<std::size_t>(n));
    std::iota(nodes.begin(), nodes.end(), 0);
    aldous_tsp::Rng shuffle_rng(reader.u64());
    shuffle_rng.partial_shuffle(nodes.begin(), nodes.end(), static_cast<std::size_t>(k));
    nodes.resize(static_cast<std::size_t>(k));

    aldous_tsp::Tour tour;
    tour.init(n);
    tour.set_tour(nodes, instance);
    const int operations = reader.bounded_int(0, 96);
    for (int operation = 0; operation < operations; ++operation) {
        if (!tour.check_invariants()) {
            __builtin_trap();
        }
        switch (reader.byte() % 4U) {
            case 0: {
                const int lo = reader.bounded_int(0, k - 1);
                const int hi = reader.bounded_int(lo, k - 1);
                tour.reverse_segment_nodes(lo, hi);
                tour.recompute_length(instance);
                break;
            }
            case 1: {
                const int start = reader.bounded_int(0, k - 1);
                const int length = reader.bounded_int(0, k);
                tour.reverse_cyclic_nodes(start, length);
                tour.recompute_length(instance);
                break;
            }
            case 2: {
                if (k >= 4) {
                    int first = reader.bounded_int(0, k - 1);
                    int second = reader.bounded_int(0, k - 1);
                    if (first > second) {
                        std::swap(first, second);
                    }
                    if (second - first >= 2 && !(first == 0 && second == k - 1)) {
                        tour.ensure_edges(instance);
                        const int a = tour.nodes[static_cast<std::size_t>(first)];
                        const int b = tour.nodes[static_cast<std::size_t>(first + 1)];
                        const int c = tour.nodes[static_cast<std::size_t>(second)];
                        const int d = tour.nodes[static_cast<std::size_t>(
                            second + 1 == k ? 0 : second + 1)];
                        const double delta = instance.dist(a, c) + instance.dist(b, d)
                            - tour.edge_len[static_cast<std::size_t>(first)]
                            - tour.edge_len[static_cast<std::size_t>(second)];
                        tour.apply_two_opt(first, second, instance, delta);
                    }
                }
                break;
            }
            default:
                tour.ensure_edges(instance);
                break;
        }
    }
    if (!tour.check_invariants()) {
        __builtin_trap();
    }
    const double exact = aldous_tsp::cycle_length(instance, tour.nodes);
    if (!std::isfinite(tour.length) || std::fabs(tour.length - exact) > 1.0e-8) {
        __builtin_trap();
    }
    return 0;
}
