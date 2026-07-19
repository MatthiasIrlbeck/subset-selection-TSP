#include "aldous_tsp/geometry.hpp"

#include "aldous_tsp/config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aldous_tsp {

bool PeriodicDomain::valid() const noexcept {
    return side > 0.0 && std::isfinite(side);
}

double PeriodicDomain::normalize(double value) const noexcept {
    if (!valid() || !std::isfinite(value)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double normalized = std::fmod(value, side);
    if (normalized < 0.0) {
        normalized += side;
    }
    // fmod should return a value strictly smaller than side, but rounding at
    // very large magnitudes can produce the endpoint. Keep the public domain
    // invariant half-open: [0, side).
    if (!(normalized < side)) {
        normalized = 0.0;
    }
    if (normalized == 0.0) {
        normalized = 0.0; // Canonicalize negative zero.
    }
    return normalized;
}

Point PeriodicDomain::normalize(Point point) const noexcept {
    point.x = normalize(point.x);
    point.y = normalize(point.y);
    return point;
}

double PeriodicDomain::signed_delta(double lhs, double rhs) const noexcept {
    const double lhs_normalized = normalize(lhs);
    const double rhs_normalized = normalize(rhs);
    double difference = lhs_normalized - rhs_normalized;
    const double half = 0.5 * side;
    if (difference > half) {
        difference -= side;
    } else if (difference < -half) {
        difference += side;
    }
    return difference;
}

double PeriodicDomain::delta(double lhs, double rhs) const noexcept {
    return std::fabs(signed_delta(lhs, rhs));
}

double PeriodicDomain::distance2(const Point& lhs, const Point& rhs) const noexcept {
    return distance2(lhs, rhs.x, rhs.y);
}

double PeriodicDomain::distance2(const Point& lhs, double rhs_x, double rhs_y) const noexcept {
    const double dx = signed_delta(lhs.x, rhs_x);
    const double dy = signed_delta(lhs.y, rhs_y);
    return dx * dx + dy * dy;
}

double euclidean_distance2(const Point& lhs, const Point& rhs) noexcept {
    return euclidean_distance2(lhs, rhs.x, rhs.y);
}

double euclidean_distance2(const Point& lhs, double rhs_x, double rhs_y) noexcept {
    const double dx = lhs.x - rhs_x;
    const double dy = lhs.y - rhs_y;
    return dx * dx + dy * dy;
}

double metric_distance2(const Point& lhs,
                        const Point& rhs,
                        bool periodic,
                        double side) noexcept {
    return metric_distance2(lhs, rhs.x, rhs.y, periodic, side);
}

double metric_distance2(const Point& lhs,
                        double rhs_x,
                        double rhs_y,
                        bool periodic,
                        double side) noexcept {
    if (periodic) {
        return PeriodicDomain{side}.distance2(lhs, rhs_x, rhs_y);
    }
    return euclidean_distance2(lhs, rhs_x, rhs_y);
}

PeriodicMeanAccumulator::PeriodicMeanAccumulator(double side) noexcept
    : domain_{side} {}

void PeriodicMeanAccumulator::add(double value) noexcept {
    const double normalized = domain_.normalize(value);
    if (count_ == 0U) {
        anchor_ = normalized;
    }
    const long double angle =
        2.0L * static_cast<long double>(kPi) * static_cast<long double>(normalized)
        / static_cast<long double>(domain_.side);
    sin_sum_ += std::sin(angle);
    cos_sum_ += std::cos(angle);
    unwrapped_sum_ += static_cast<long double>(anchor_)
        + static_cast<long double>(domain_.signed_delta(normalized, anchor_));
    ++count_;
}

double PeriodicMeanAccumulator::mean() const noexcept {
    if (count_ == 0U || !domain_.valid()) {
        return 0.0;
    }
    const long double resultant = std::hypot(sin_sum_, cos_sum_);
    const long double tolerance =
        64.0L * std::numeric_limits<long double>::epsilon()
        * static_cast<long double>(count_);
    if (resultant > tolerance) {
        long double angle = std::atan2(sin_sum_, cos_sum_);
        if (angle < 0.0L) {
            angle += 2.0L * static_cast<long double>(kPi);
        }
        const long double coordinate =
            angle * static_cast<long double>(domain_.side)
            / (2.0L * static_cast<long double>(kPi));
        return domain_.normalize(static_cast<double>(coordinate));
    }
    const long double unwrapped_mean = unwrapped_sum_ / static_cast<long double>(count_);
    return domain_.normalize(static_cast<double>(unwrapped_mean));
}

PointMeanAccumulator::PointMeanAccumulator(bool periodic, double side) noexcept
    : periodic_(periodic), periodic_x_(side), periodic_y_(side) {}

void PointMeanAccumulator::add(const Point& point) noexcept {
    if (periodic_) {
        periodic_x_.add(point.x);
        periodic_y_.add(point.y);
    } else {
        x_sum_ += static_cast<long double>(point.x);
        y_sum_ += static_cast<long double>(point.y);
    }
    ++count_;
}

Point PointMeanAccumulator::mean() const noexcept {
    if (count_ == 0U) {
        return {};
    }
    if (periodic_) {
        return {periodic_x_.mean(), periodic_y_.mean()};
    }
    const long double denominator = static_cast<long double>(count_);
    return {static_cast<double>(x_sum_ / denominator),
            static_cast<double>(y_sum_ / denominator)};
}

} // namespace aldous_tsp
