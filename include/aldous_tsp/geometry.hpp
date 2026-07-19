#pragma once

#include <cstddef>

namespace aldous_tsp {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

// Canonical geometry for a square flat torus with origin (0, 0).
// Instance validates the side length before invoking these operations.
struct PeriodicDomain {
    double side = 0.0;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] double normalize(double value) const noexcept;
    [[nodiscard]] Point normalize(Point point) const noexcept;
    [[nodiscard]] double signed_delta(double lhs, double rhs) const noexcept;
    [[nodiscard]] double delta(double lhs, double rhs) const noexcept;
    [[nodiscard]] double distance2(const Point& lhs, const Point& rhs) const noexcept;
    [[nodiscard]] double distance2(const Point& lhs, double rhs_x, double rhs_y) const noexcept;
};

[[nodiscard]] double euclidean_distance2(const Point& lhs, const Point& rhs) noexcept;
[[nodiscard]] double euclidean_distance2(const Point& lhs, double rhs_x, double rhs_y) noexcept;
[[nodiscard]] double metric_distance2(const Point& lhs,
                                      const Point& rhs,
                                      bool periodic,
                                      double side) noexcept;
[[nodiscard]] double metric_distance2(const Point& lhs,
                                      double rhs_x,
                                      double rhs_y,
                                      bool periodic,
                                      double side) noexcept;

// A periodic coordinate mean uses the circular mean when it is well-defined.
// For rotationally symmetric/antipodal samples, where the circular resultant
// vanishes, it deterministically falls back to averaging coordinates unwrapped
// around the first sample. This keeps seam-straddling clusters together while
// defining the otherwise ambiguous degenerate case.
class PeriodicMeanAccumulator {
public:
    explicit PeriodicMeanAccumulator(double side) noexcept;

    void add(double value) noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] double mean() const noexcept;

private:
    PeriodicDomain domain_;
    std::size_t count_ = 0;
    long double sin_sum_ = 0.0L;
    long double cos_sum_ = 0.0L;
    long double unwrapped_sum_ = 0.0L;
    double anchor_ = 0.0;
};

class PointMeanAccumulator {
public:
    PointMeanAccumulator(bool periodic, double side) noexcept;

    void add(const Point& point) noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] Point mean() const noexcept;

private:
    bool periodic_ = false;
    std::size_t count_ = 0;
    long double x_sum_ = 0.0L;
    long double y_sum_ = 0.0L;
    PeriodicMeanAccumulator periodic_x_;
    PeriodicMeanAccumulator periodic_y_;
};

} // namespace aldous_tsp
