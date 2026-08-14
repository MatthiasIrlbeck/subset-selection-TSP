#include "cli_internal.hpp"

namespace aldous_tsp {

std::string config_summary(const RunOptions& opt) {
    std::ostringstream out;
    out << "N=" << opt.N
        << ", instances=" << opt.instances
        << ", threads=" << opt.threads
        << ", output_durability=" << output_durability_name(opt.output_durability)
        << ", seed=" << opt.solver.seed
        << ", point_seed=" << effective_point_seed(opt)
        << ", search_seed=" << effective_search_seed(opt)
        << ", campaign_id=" << opt.campaign_id
        << ", campaign_shard=" << opt.campaign_shard
        << ", replicate_offset=" << opt.replicate_offset
        << ", solver_policy_id=" << opt.solver_policy_id
        << ", fidelity_level=" << opt.fidelity_level
        << ", mode=" << solver_mode_name(opt.solver.mode)
        << ", knn=" << opt.solver.knn_k
        << ", knn_backend=" << knn_backend_name(opt.solver.knn_backend)
        << ", verify_knn_checks=" << opt.solver.verify_knn_checks
        << ", search_policy="
        << search_policy_preset_name(opt.solver.search_policy_preset)
        << ", subset_restarts="
        << (opt.solver.subset_restarts >= 1
                ? std::to_string(opt.solver.subset_restarts)
                : std::string("auto"))
        << ", continuation_restarts=" << opt.solver.continuation_restarts
        << ", continuation_policy="
        << continuation_policy_name(opt.solver.continuation_policy)
        << ", racing_candidates=" << opt.solver.racing_candidates
        << ", racing_survivors=" << opt.solver.racing_survivors
        << ", staged_search=" << (opt.solver.staged_search ? "on" : "off")
        << ", strong_polish_finalists=" << opt.solver.strong_polish_finalists
        << ", sa_iters=" << opt.solver.sa_iters
        << ", sa_iters_per_k=" << opt.solver.sa_iters_per_k
        << ", sa_iters_per_n=" << opt.solver.sa_iters_per_n
        << ", sa_t0=" << opt.solver.sa_t0
        << ", sa_t1=" << opt.solver.sa_t1
        << ", restart_threads=" << opt.solver.restart_threads
        << ", second_sweep=" << (opt.second_sweep ? "true" : "false")
        << ", periodic=" << (opt.periodic ? "true" : "false")
        << ", control_variate=" << (opt.control_variate ? "true" : "false")
        << ", held_karp=" << (opt.held_karp ? "true" : "false")
        << ", exact_subset_max_n=" << opt.solver.exact_subset_max_n
        << ", tsp_candidate_starts=" << opt.solver.tsp_candidate_starts
        << ", tsp_restarts=" << opt.solver.tsp_restarts
        << ", pair_exchange_max_k=" << opt.solver.pair_exchange_max_k
        << ", path_relink_top=" << opt.solver.path_relink_top
        << ", oracle=" << opt.solver.oracle.status
        << ", p_values=";
    for (std::size_t i = 0; i < opt.p_values.size(); ++i) {
        if (i != 0U) {
            out << ',';
        }
        out << std::setprecision(17) << opt.p_values[i];
    }
    return out.str();
}

} // namespace aldous_tsp
