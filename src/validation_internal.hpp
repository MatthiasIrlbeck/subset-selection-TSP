#pragma once

#include <memory>
#include <string>

namespace aldous_tsp {
class Instance;

namespace detail {

// Returns an immutable canonical reconstruction when the public mutable state is
// structurally and numerically valid.  The reconstruction is built from points
// and configuration rather than reusing caller-owned KNN/grid arrays.
std::shared_ptr<const Instance> validate_and_canonicalize_instance(
    const Instance& instance,
    std::string& error);

// Used only by InstanceBuilder after all structures were created by the library
// itself.  It still performs the complete structural/numerical validation but
// avoids rebuilding a second KNN copy.
std::shared_ptr<const Instance> adopt_library_built_instance(
    Instance&& instance,
    std::string& error);

} // namespace detail
} // namespace aldous_tsp
