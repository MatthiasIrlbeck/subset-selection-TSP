#pragma once

#include "aldous_tsp/config.hpp"
#include "aldous_tsp/geometry.hpp"
#include "aldous_tsp/rng.hpp"

#include <cstdint>
#include <vector>

namespace aldous_tsp {


struct KnnBuildInfo {
    KnnBackend requested_backend = KnnBackend::GridExact;
    KnnBackend effective_backend = KnnBackend::GridExact;
    bool brute_force_fallback = false;
    bool grid_cell_capped = false;
    double requested_cell_size = 0.0;
    double effective_cell_size = 0.0;
    int gx = 0;
    int gy = 0;
    std::uint64_t grid_cells = 0;
    double coordinate_span = 0.0;
};

class Instance {
public:
    int N = 0;
    // When true, the domain is a flat torus of side `side` (periodic boundary
    // conditions): distances wrap around each axis. This removes the O(1/sqrt N)
    // boundary correction to the mean optimal tour length, leaving only O(1/N)
    // corrections (Percus & Martin, PRL 76, 1188); the limit is unchanged
    // (Jaillet 1993), so estimates of f(p) converge much faster.
    bool periodic = false;
    // If > 0, recompute_bounds() pins the domain side to this value instead of
    // deriving it from the point spread. Used to build a KNN on a selected
    // subset that lives on the original (larger) torus, so the minimum-image
    // metric uses the true torus period rather than the subset's bounding box.
    double explicit_side = 0.0;
    double side = 0.0;
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    std::vector<Point> points;
    int knn_k = 0;
    KnnBackend knn_backend = KnnBackend::GridExact;
    KnnBuildInfo last_knn_build;
    std::vector<int> knn;
    std::vector<double> knn_d2;
    std::vector<double> knn_d;

    // Reverse exact KNN adjacency: for node v, nodes u such that v is in KNN(u).
    std::vector<int> rknn_begin;
    std::vector<int> rknn_nodes;

    // Uniform-grid metadata used by the grid-exact KNN backend.
    double cell_size = 1.0;
    double grid_min_x = 0.0;
    double grid_min_y = 0.0;
    double grid_max_x = 0.0;
    double grid_max_y = 0.0;
    int gx = 0;
    int gy = 0;
    std::vector<int> cell_x;
    std::vector<int> cell_y;
    std::vector<int> cell_begin;
    std::vector<int> cell_points;

    void generate(int n, Rng& rng);
    void set_points(std::vector<Point> pts);
    void recompute_bounds();
    void build_knn(int k, KnnBackend backend = KnnBackend::GridExact, double forced_cell_size = 0.0);
    bool verify_knn(int checks, Rng& rng) const;

    [[nodiscard]] double dist2(int a, int b) const noexcept;
    [[nodiscard]] double dist(int a, int b) const noexcept;
    [[nodiscard]] double dist2_to_point(int node, double x, double y) const noexcept;
    [[nodiscard]] double dist_to_point(int node, double x, double y) const noexcept;
    // Fast periodic query for a point already normalized into [0, side).
    // Open instances treat the coordinates as ordinary Euclidean values.
    [[nodiscard]] double dist2_to_canonical_point(int node, double x, double y) const noexcept;
    [[nodiscard]] int knn_at(int node, int rank) const noexcept;
    [[nodiscard]] double knn_d_at(int node, int rank) const noexcept;
    [[nodiscard]] double knn_d2_at(int node, int rank) const noexcept;

private:
    void clear_knn();
    void build_reverse_knn();
    void build_knn_bruteforce(int k);
    void build_grid(double forced_cell_size);
    void build_knn_grid(int k, double forced_cell_size);
    void insert_best(int* best_idx, double* best_d2, int& count, int idx, double d2) const;
    double cell_min_d2(double qx, double qy, int cell_id) const noexcept;
    double outside_bound_d2(int qi, int radius) const noexcept;
};

void dist_many_from(const Instance& inst, int src, const int* ids, int count, double* out);

} // namespace aldous_tsp
