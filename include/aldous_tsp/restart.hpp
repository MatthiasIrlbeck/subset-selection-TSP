#pragma once

#include <array>
#include <cstddef>
#include <limits>

namespace aldous_tsp {

// Stable restart-kind codes used by the C++ API, result JSON, schema, and
// analysis tools. Additions must be made in restart_kinds.def; compile-time
// validation below rejects duplicate or non-contiguous codes.
enum class RestartKind : int {
#define ALDOUS_TSP_RESTART_KIND(name, code, label) name = code,
#include "aldous_tsp/restart_kinds.def"
#undef ALDOUS_TSP_RESTART_KIND
};

struct RestartKindInfo {
    RestartKind kind;
    int code;
    const char* name;
};

inline constexpr std::size_t kRestartKindCount = 0U
#define ALDOUS_TSP_RESTART_KIND(name, code, label) + 1U
#include "aldous_tsp/restart_kinds.def"
#undef ALDOUS_TSP_RESTART_KIND
;

inline constexpr std::array<RestartKindInfo, kRestartKindCount> kRestartKinds{{
#define ALDOUS_TSP_RESTART_KIND(name, code, label) {RestartKind::name, code, label},
#include "aldous_tsp/restart_kinds.def"
#undef ALDOUS_TSP_RESTART_KIND
}};

[[nodiscard]] constexpr bool restart_kind_metadata_is_valid() noexcept {
    for (std::size_t i = 0; i < kRestartKinds.size(); ++i) {
        if (kRestartKinds[i].code != static_cast<int>(i)
            || static_cast<int>(kRestartKinds[i].kind) != kRestartKinds[i].code
            || kRestartKinds[i].name == nullptr
            || kRestartKinds[i].name[0] == '\0') {
            return false;
        }
    }
    return true;
}

static_assert(restart_kind_metadata_is_valid(),
              "restart kind codes must be contiguous, unique, and named");

[[nodiscard]] constexpr int restart_kind_code(RestartKind kind) noexcept {
    return static_cast<int>(kind);
}

[[nodiscard]] constexpr bool is_valid_restart_kind_code(int code) noexcept {
    return code >= 0 && static_cast<std::size_t>(code) < kRestartKinds.size()
           && kRestartKinds[static_cast<std::size_t>(code)].code == code;
}

[[nodiscard]] constexpr const char* restart_kind_name(RestartKind kind) noexcept {
    const int code = restart_kind_code(kind);
    return is_valid_restart_kind_code(code)
        ? kRestartKinds[static_cast<std::size_t>(code)].name
        : "unknown";
}

[[nodiscard]] constexpr bool restart_kind_from_code(int code, RestartKind& out) noexcept {
    if (!is_valid_restart_kind_code(code)) {
        return false;
    }
    out = kRestartKinds[static_cast<std::size_t>(code)].kind;
    return true;
}

// A solve outside ExperimentRunner has one primary sweep. In experiment JSON,
// Primary is the descending sweep and Secondary is the optional ascending
// second sweep. Stable codes are serialized in restart_sweeps.
enum class RestartSweep : int {
    Primary = 0,
    Secondary = 1,
};

[[nodiscard]] constexpr int restart_sweep_code(RestartSweep sweep) noexcept {
    return static_cast<int>(sweep);
}

[[nodiscard]] constexpr const char* restart_sweep_name(RestartSweep sweep) noexcept {
    switch (sweep) {
        case RestartSweep::Primary: return "primary";
        case RestartSweep::Secondary: return "secondary";
    }
    return "unknown";
}

// One authoritative restart diagnostic. `length` is always in raw distance
// units; result JSON derives the backward-compatible restart_values (L/k)
// array from it. Geometry describes the selected node set at restart exit.
struct RestartRecord {
    double length = std::numeric_limits<double>::infinity();
    RestartKind kind = RestartKind::Random;
    RestartSweep sweep = RestartSweep::Primary;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    double radius = 0.0;
};

} // namespace aldous_tsp
