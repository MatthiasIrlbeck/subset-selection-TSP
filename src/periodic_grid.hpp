#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace aldous_tsp::detail {

[[nodiscard]] inline int wrap_cell_index(int value, int count) noexcept {
    value %= count;
    return value < 0 ? value + count : value;
}

class GenerationMarks {
public:
    void begin(std::size_t size) {
        if (marks_.size() < size) {
            marks_.assign(size, 0U);
            epoch_ = 0U;
        }
        ++epoch_;
        if (epoch_ == 0U) {
            std::fill(marks_.begin(), marks_.end(), 0U);
            epoch_ = 1U;
        }
        visited_ = 0U;
    }

    [[nodiscard]] bool mark(std::size_t index) noexcept {
        if (marks_[index] == epoch_) {
            return false;
        }
        marks_[index] = epoch_;
        ++visited_;
        return true;
    }

    [[nodiscard]] std::size_t visited() const noexcept { return visited_; }

private:
    std::vector<std::uint32_t> marks_;
    std::uint32_t epoch_ = 0U;
    std::size_t visited_ = 0U;
};

[[nodiscard]] inline int periodic_max_ring(int cells_x, int cells_y) noexcept {
    return std::max(cells_x / 2, cells_y / 2);
}

template <typename Function>
int visit_periodic_ring_unique(int center_x,
                               int center_y,
                               int radius,
                               int cells_x,
                               int cells_y,
                               GenerationMarks& marks,
                               Function&& function) {
    int newly_visited = 0;
    auto visit = [&](int x, int y) {
        const int wrapped_x = wrap_cell_index(x, cells_x);
        const int wrapped_y = wrap_cell_index(y, cells_y);
        const int cell = wrapped_y * cells_x + wrapped_x;
        if (marks.mark(static_cast<std::size_t>(cell))) {
            function(cell);
            ++newly_visited;
        }
    };

    if (radius == 0) {
        visit(center_x, center_y);
        return newly_visited;
    }

    const int top = center_y - radius;
    const int bottom = center_y + radius;
    for (int dx = -radius; dx <= radius; ++dx) {
        visit(center_x + dx, top);
        visit(center_x + dx, bottom);
    }
    for (int dy = -radius + 1; dy <= radius - 1; ++dy) {
        visit(center_x - radius, center_y + dy);
        visit(center_x + radius, center_y + dy);
    }
    return newly_visited;
}

// Squared lower bound on the periodic distance from a query point to any cell
// not yet visited after all unique Chebyshev rings [0, scanned_radius] have
// been scanned. query_offset_{x,y} are the query's offsets inside its own cell.
// A strict comparison against this bound is required to preserve equal-distance
// node-id tie breaking.
[[nodiscard]] inline long double periodic_unvisited_distance2_lower_bound(
    double query_offset_x,
    double query_offset_y,
    double cell_width,
    double cell_height,
    int cells_x,
    int cells_y,
    int scanned_radius) noexcept {
    const long double infinity = std::numeric_limits<long double>::infinity();
    const long double width = static_cast<long double>(cell_width);
    const long double height = static_cast<long double>(cell_height);
    const long double offset_x = std::clamp(static_cast<long double>(query_offset_x), 0.0L, width);
    const long double offset_y = std::clamp(static_cast<long double>(query_offset_y), 0.0L, height);

    long double best_gap = infinity;
    if (scanned_radius < cells_x / 2) {
        const long double edge_gap = std::min(offset_x, width - offset_x);
        best_gap = std::min(best_gap,
                            static_cast<long double>(scanned_radius) * width + edge_gap);
    }
    if (scanned_radius < cells_y / 2) {
        const long double edge_gap = std::min(offset_y, height - offset_y);
        best_gap = std::min(best_gap,
                            static_cast<long double>(scanned_radius) * height + edge_gap);
    }
    if (!std::isfinite(best_gap)) {
        return infinity;
    }
    // Round the analytic lower bound toward zero before squaring. This keeps it
    // conservative even when the query/cell arithmetic lands on a rounded cell
    // boundary.
    if (best_gap > 0.0L) {
        best_gap = std::nextafter(best_gap, 0.0L);
    }
    return best_gap * best_gap;
}

} // namespace aldous_tsp::detail
