#include "campaign_fingerprint.hpp"

#include "generated_options.hpp"
#include "sha256.hpp"

#include "aldous_tsp/version.hpp"

#include <sstream>
#include <utility>

namespace aldous_tsp::detail {
namespace {

std::string canonical_config(const RunOptions& options) {
    std::ostringstream out;
    write_generated_config(out, options, "");
    out << "\noracle_exec_path=" << options.solver.oracle.exec_path
        << "\noracle_exec_sha256=" << options.solver.oracle.exec_sha256
        << "\noracle_version=" << options.solver.oracle.version;
    return out.str();
}

RunOptions method_projection(const RunOptions& source) {
    RunOptions result = source;
    result.N = 0;
    result.instances = 0;
    if (result.solver.time_budget_per_p <= 0.0) {
        result.threads = 0;
        result.solver.restart_threads = 0;
    }
    result.memory_budget_mb = 0;
    result.verbose = false;
    result.include_instance_rows = false;
    result.campaign_id.clear();
    result.campaign_shard = 0;
    result.replicate_offset = 0;
    result.point_seed = 0;
    result.search_seed = 0;
    result.solver.seed = 0;
    result.output_path.clear();
    result.force_output = false;
    result.dry_run = false;
    result.dump_config = false;
    result.p_values.clear();
    return result;
}

} // namespace

ConfigurationFingerprints configuration_fingerprints(const RunOptions& options) {
    RunOptions resolved = options;
    // Probing uses --dry-run/--dump-config, but those execution controls do not
    // change the planned result. Normalize them so the probe and completed run
    // have the same exact-cell fingerprint.
    resolved.dry_run = false;
    resolved.dump_config = false;
    const std::string source_identity =
        std::string(kGitCommit) + ":" + kGitTree + ":"
        + (kSourceDirty ? "dirty" : "clean") + ":" + kRevisionSource;
    const std::string resolved_material =
        std::string("aldous-tsp-resolved-v1\nproject=") + kProjectVersion
        + "\nsource=" + source_identity + "\n" + canonical_config(resolved);
    const RunOptions method = method_projection(options);
    const std::string method_material =
        std::string("aldous-tsp-method-v1\nproject=") + kProjectVersion
        + "\nsource=" + source_identity + "\n" + canonical_config(method);
    return {
        sha256_hex(resolved_material),
        sha256_hex(method_material),
    };
}

} // namespace aldous_tsp::detail
