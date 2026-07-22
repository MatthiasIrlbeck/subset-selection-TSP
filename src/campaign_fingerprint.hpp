#pragma once

#include "aldous_tsp/config.hpp"

#include <string>

namespace aldous_tsp::detail {

struct ConfigurationFingerprints {
    std::string resolved;
    std::string method;
};

// Hashes canonical generated configuration rather than command-line spelling.
// `resolved` identifies one exact planned cell. `method` removes campaign,
// stream, output, and problem-grid identities so analysis can reject any
// quality-affecting solver/oracle mismatch across otherwise comparable cells.
ConfigurationFingerprints configuration_fingerprints(const RunOptions& options);

} // namespace aldous_tsp::detail
