#include "aldous_tsp/instance.hpp"

#include "validation_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace aldous_tsp {

PreparedInstance PreparedInstance::from_instance(const Instance& instance) {
    std::string error;
    std::shared_ptr<const Instance> canonical =
        detail::validate_and_canonicalize_instance(instance, error);
    if (!canonical) {
        throw std::invalid_argument(error);
    }
    return PreparedInstance(std::move(canonical), TrustedTag{});
}

PreparedInstance PreparedInstance::from_instance(Instance&& instance) {
    // Public mutable state is never trusted merely because the caller supplied
    // an rvalue.  Canonical reconstruction is what makes the resulting object
    // immutable and independent of caller-owned KNN/grid buffers.
    return from_instance(static_cast<const Instance&>(instance));
}

InstanceBuilder& InstanceBuilder::periodic(const bool enabled) noexcept {
    instance_.periodic = enabled;
    return *this;
}

InstanceBuilder& InstanceBuilder::explicit_side(const double side) {
    if (side != 0.0 && (!std::isfinite(side) || !(side > 0.0))) {
        throw std::invalid_argument(
            "InstanceBuilder explicit side must be zero or finite and positive");
    }
    instance_.explicit_side = side;
    return *this;
}

InstanceBuilder& InstanceBuilder::generate(const int n, Rng& rng) {
    instance_.generate(n, rng);
    return *this;
}

InstanceBuilder& InstanceBuilder::set_points(std::vector<Point> points) {
    instance_.set_points(std::move(points));
    return *this;
}

PreparedInstance InstanceBuilder::build(const int knn_k,
                                        const KnnBackend backend,
                                        const double forced_cell_size) {
    instance_.build_knn(knn_k, backend, forced_cell_size);
    std::string error;
    std::shared_ptr<const Instance> adopted =
        detail::adopt_library_built_instance(std::move(instance_), error);
    instance_ = Instance{};
    if (!adopted) {
        throw std::invalid_argument(error);
    }
    return PreparedInstance(std::move(adopted), PreparedInstance::TrustedTag{});
}

void dist_many_from(const PreparedInstance& inst,
                    const int src,
                    const int* ids,
                    const int count,
                    double* out) {
    dist_many_from(inst.instance(), src, ids, count, out);
}

} // namespace aldous_tsp
