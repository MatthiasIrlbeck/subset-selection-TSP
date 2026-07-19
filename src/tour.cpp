#include "aldous_tsp/tour.hpp"

#include "aldous_tsp/config.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace aldous_tsp {

void SearchPhaseTiming::add(const SearchPhaseTiming& other) noexcept {
    seed_construction_seconds += other.seed_construction_seconds;
    tsp_construction_seconds += other.tsp_construction_seconds;
    initial_polish_seconds += other.initial_polish_seconds;
    sa_seconds += other.sa_seconds;
    sa_checkpoint_polish_seconds += other.sa_checkpoint_polish_seconds;
    post_sa_polish_seconds += other.post_sa_polish_seconds;
    subset_swap_seconds += other.subset_swap_seconds;
    highp_exchange_seconds += other.highp_exchange_seconds;
    pair_exchange_seconds += other.pair_exchange_seconds;
    ruin_recreate_seconds += other.ruin_recreate_seconds;
    path_relink_seconds += other.path_relink_seconds;
    tsp_ils_seconds += other.tsp_ils_seconds;
    final_polish_seconds += other.final_polish_seconds;
    oracle_seconds += other.oracle_seconds;
    sa_proposal_samples += other.sa_proposal_samples;
    sa_insertion_samples += other.sa_insertion_samples;
    sa_proposal_sample_seconds += other.sa_proposal_sample_seconds;
    sa_insertion_sample_seconds += other.sa_insertion_sample_seconds;
}

void SearchStats::add(const SearchStats& other) {
    tsp_restarts += other.tsp_restarts;
    tsp_ils_iterations += other.tsp_ils_iterations;
    subset_restarts += other.subset_restarts;
    smallp_seed_restarts += other.smallp_seed_restarts;
    highp_delete_restarts += other.highp_delete_restarts;
    warm_restarts += other.warm_restarts;
    random_restarts += other.random_restarts;
    region_restarts += other.region_restarts;
    dense_restarts += other.dense_restarts;
    racing_pilot_restarts += other.racing_pilot_restarts;
    racing_promoted_restarts += other.racing_promoted_restarts;
    elite_restarts += other.elite_restarts;
    kick_restarts += other.kick_restarts;
    two_opt_scans += other.two_opt_scans;
    two_opt_improvements += other.two_opt_improvements;
    or_opt_scans += other.or_opt_scans;
    or_opt_improvements += other.or_opt_improvements;
    sa_moves += other.sa_moves;
    sa_accepted += other.sa_accepted;
    sa_improving += other.sa_improving;
    subset_swap_scans += other.subset_swap_scans;
    subset_swap_improvements += other.subset_swap_improvements;
    highp_exchange_scans += other.highp_exchange_scans;
    highp_exchange_improvements += other.highp_exchange_improvements;
    pair_exchange_scans += other.pair_exchange_scans;
    pair_exchange_improvements += other.pair_exchange_improvements;
    pair_exchange_skipped_large_k += other.pair_exchange_skipped_large_k;
    ruin_recreate_attempts += other.ruin_recreate_attempts;
    ruin_recreate_improvements += other.ruin_recreate_improvements;
    path_relink_attempts += other.path_relink_attempts;
    path_relink_feasible += other.path_relink_feasible;
    path_relink_elite_insertions += other.path_relink_elite_insertions;
    path_relink_best_improvements += other.path_relink_best_improvements;
    path_relink_improvements += other.path_relink_improvements;
    knn_build_seconds += other.knn_build_seconds;
    const std::uint64_t old_effective_grid = knn_effective_grid_instances;
    knn_requested_grid_instances += other.knn_requested_grid_instances;
    knn_requested_bruteforce_instances += other.knn_requested_bruteforce_instances;
    knn_effective_grid_instances += other.knn_effective_grid_instances;
    knn_effective_bruteforce_instances += other.knn_effective_bruteforce_instances;
    knn_bruteforce_fallback_instances += other.knn_bruteforce_fallback_instances;
    knn_grid_cell_capped_instances += other.knn_grid_cell_capped_instances;
    knn_grid_cell_samples += other.knn_grid_cell_samples;
    knn_grid_cells_sum += other.knn_grid_cells_sum;
    knn_grid_cells_max = std::max(knn_grid_cells_max, other.knn_grid_cells_max);
    if (other.knn_effective_grid_instances > 0) {
        if (old_effective_grid == 0) {
            grid_cell_effective_min = other.grid_cell_effective_min;
            grid_cell_effective_max = other.grid_cell_effective_max;
        } else {
            grid_cell_effective_min = std::min(grid_cell_effective_min, other.grid_cell_effective_min);
            grid_cell_effective_max = std::max(grid_cell_effective_max, other.grid_cell_effective_max);
        }
        grid_cell_effective_sum += other.grid_cell_effective_sum;
    }
    tsp_seconds += other.tsp_seconds;
    subset_seconds += other.subset_seconds;
    oracle_calls += other.oracle_calls;
    oracle_solved += other.oracle_solved;
    oracle_improved += other.oracle_improved;
    oracle_failed += other.oracle_failed;
    oracle_tsp_calls += other.oracle_tsp_calls;
    oracle_subset_calls += other.oracle_subset_calls;
    oracle_gain += other.oracle_gain;
    oracle_call_records.insert(oracle_call_records.end(), other.oracle_call_records.begin(), other.oracle_call_records.end());
    phases.add(other.phases);
}

const char* solver_mode_name(SolverMode mode) noexcept {
    switch (mode) {
        case SolverMode::Balanced: return "balanced";
        case SolverMode::SmallPRegion: return "smallp-region";
        case SolverMode::HighPDelete: return "highp-delete";
        case SolverMode::Hybrid: return "hybrid";
    }
    return "balanced";
}

bool parse_solver_mode(const std::string& text, SolverMode& out) noexcept {
    if (text == "balanced") { out = SolverMode::Balanced; return true; }
    if (text == "smallp-region") { out = SolverMode::SmallPRegion; return true; }
    if (text == "highp-delete") { out = SolverMode::HighPDelete; return true; }
    if (text == "hybrid") { out = SolverMode::Hybrid; return true; }
    return false;
}

const char* continuation_policy_name(ContinuationPolicy policy) noexcept {
    switch (policy) {
        case ContinuationPolicy::Supplemental: return "supplemental";
        case ContinuationPolicy::FixedBudget: return "fixed-budget";
    }
    return "supplemental";
}

bool parse_continuation_policy(const std::string& text, ContinuationPolicy& out) noexcept {
    if (text == "supplemental" || text == "append") {
        out = ContinuationPolicy::Supplemental;
        return true;
    }
    if (text == "fixed-budget" || text == "fixed" || text == "reserved") {
        out = ContinuationPolicy::FixedBudget;
        return true;
    }
    return false;
}

const char* knn_backend_name(KnnBackend backend) noexcept {
    switch (backend) {
        case KnnBackend::BruteForce: return "coords_exact_bruteforce_knn";
        case KnnBackend::GridExact: return "coords_exact_grid_knn";
    }
    return "coords_exact_grid_knn";
}

bool parse_knn_backend(const std::string& text, KnnBackend& out) noexcept {
    if (text == "bruteforce" || text == "brute-force") { out = KnnBackend::BruteForce; return true; }
    if (text == "grid" || text == "grid-exact") { out = KnnBackend::GridExact; return true; }
    return false;
}

const char* exhaustive_two_opt_policy_name(ExhaustiveTwoOptPolicy policy) noexcept {
    switch (policy) {
        case ExhaustiveTwoOptPolicy::Never: return "never";
        case ExhaustiveTwoOptPolicy::FinalOnly: return "final-only";
        case ExhaustiveTwoOptPolicy::AllPolish: return "all-polish";
    }
    return "final-only";
}

bool parse_exhaustive_two_opt_policy(const std::string& text, ExhaustiveTwoOptPolicy& out) noexcept {
    if (text == "never" || text == "off" || text == "none") { out = ExhaustiveTwoOptPolicy::Never; return true; }
    if (text == "final-only" || text == "final" || text == "reporting") { out = ExhaustiveTwoOptPolicy::FinalOnly; return true; }
    if (text == "all-polish" || text == "all" || text == "legacy") { out = ExhaustiveTwoOptPolicy::AllPolish; return true; }
    return false;
}

const char* external_oracle_mode_name(ExternalOracleMode mode) noexcept {
    switch (mode) {
        case ExternalOracleMode::None: return "none";
        case ExternalOracleMode::Auto: return "auto";
        case ExternalOracleMode::Lkh: return "lkh";
        case ExternalOracleMode::Concorde: return "concorde";
    }
    return "none";
}

bool parse_external_oracle_mode(const std::string& text, ExternalOracleMode& out) noexcept {
    if (text == "none") { out = ExternalOracleMode::None; return true; }
    if (text == "auto") { out = ExternalOracleMode::Auto; return true; }
    if (text == "lkh") { out = ExternalOracleMode::Lkh; return true; }
    if (text == "concorde") { out = ExternalOracleMode::Concorde; return true; }
    return false;
}

const char* oracle_problem_format_name(OracleProblemFormat format) noexcept {
    switch (format) {
        case OracleProblemFormat::Matrix: return "matrix";
        case OracleProblemFormat::Euc2d: return "euc2d";
    }
    return "matrix";
}

bool parse_oracle_problem_format(const std::string& text, OracleProblemFormat& out) noexcept {
    if (text == "matrix") { out = OracleProblemFormat::Matrix; return true; }
    if (text == "euc2d") { out = OracleProblemFormat::Euc2d; return true; }
    return false;
}

const char* resolved_oracle_mode_name(ResolvedOracleMode mode) noexcept {
    switch (mode) {
        case ResolvedOracleMode::None: return "none";
        case ResolvedOracleMode::Lkh: return "lkh";
        case ResolvedOracleMode::Concorde: return "concorde";
    }
    return "none";
}

std::vector<double> default_p_values() {
    return {0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00};
}

std::uint64_t subset_hash_nodes(const std::vector<int>& nodes) noexcept {
    std::uint64_t h = 0x243f6a8885a308d3ULL ^ (static_cast<std::uint64_t>(nodes.size()) * 0x9e3779b97f4a7c15ULL);
    for (int node : nodes) {
        h ^= mix_hash64(static_cast<std::uint64_t>(node) + 0x9e3779b97f4a7c15ULL) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    }
    return mix_hash64(h);
}

std::vector<int> canonical_set_key(std::vector<int> nodes) {
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

std::vector<int> canonical_cycle_key(const std::vector<int>& nodes) {
    const int n = static_cast<int>(nodes.size());
    if (n <= 1) {
        return nodes;
    }
    int min_idx = 0;
    for (int i = 1; i < n; ++i) {
        if (nodes[static_cast<std::size_t>(i)] < nodes[static_cast<std::size_t>(min_idx)]) {
            min_idx = i;
        }
    }
    auto fwd = [&](int t) { return nodes[static_cast<std::size_t>((min_idx + t) % n)]; };
    auto rev = [&](int t) {
        int idx = min_idx - t;
        while (idx < 0) { idx += n; }
        return nodes[static_cast<std::size_t>(idx)];
    };
    bool use_fwd = true;
    for (int t = 1; t < n; ++t) {
        if (fwd(t) < rev(t)) { use_fwd = true; break; }
        if (rev(t) < fwd(t)) { use_fwd = false; break; }
    }
    std::vector<int> out;
    out.reserve(nodes.size());
    for (int t = 0; t < n; ++t) {
        out.push_back(use_fwd ? fwd(t) : rev(t));
    }
    return out;
}

std::vector<int> ElitePool::make_key(const std::vector<int>& nodes) const {
    switch (mode_) {
        case EliteMode::Ordered: return nodes;
        case EliteMode::Cycle: return canonical_cycle_key(nodes);
        case EliteMode::Set: return canonical_set_key(nodes);
    }
    return nodes;
}

void ElitePool::try_add(const std::vector<int>& nodes, double length) {
    if (keep_ <= 0 || nodes.empty() || !std::isfinite(length)) {
        return;
    }
    std::vector<int> key = make_key(nodes);
    const std::uint64_t hash = subset_hash_nodes(key);
    for (EliteEntry& entry : entries_) {
        if (entry.hash == hash && entry.canonical_key == key) {
            if (length + kImprovementEps < entry.length) {
                entry.length = length;
                entry.nodes = nodes;
            }
            std::sort(entries_.begin(), entries_.end(), [](const EliteEntry& lhs, const EliteEntry& rhs) {
                return lhs.length < rhs.length;
            });
            return;
        }
    }
    EliteEntry entry;
    entry.length = length;
    entry.nodes = nodes;
    entry.canonical_key = std::move(key);
    entry.hash = hash;
    auto it = std::lower_bound(entries_.begin(), entries_.end(), length, [](const EliteEntry& lhs, double value) {
        return lhs.length < value;
    });
    entries_.insert(it, std::move(entry));
    if (static_cast<int>(entries_.size()) > keep_) {
        entries_.pop_back();
    }
}

std::vector<std::vector<int>> ElitePool::export_nodes() const {
    std::vector<std::vector<int>> out;
    out.reserve(entries_.size());
    for (const EliteEntry& entry : entries_) {
        out.push_back(entry.nodes);
    }
    return out;
}

void Tour::init(int n) {
    if (n < 0) {
        throw std::invalid_argument("Tour::init requires n >= 0");
    }
    N = n;
    k = 0;
    nodes.clear();
    pos.assign(static_cast<std::size_t>(N), -1);
    in_set.assign(static_cast<std::size_t>(N), 0U);
    edge_len.clear();
    length = 0.0;
    edge_valid = false;
}

void Tour::set_tour_only(const std::vector<int>& new_nodes) {
    nodes = new_nodes;
    k = static_cast<int>(nodes.size());
    rebuild_index();
    edge_len.assign(static_cast<std::size_t>(k), 0.0);
    length = 0.0;
    edge_valid = false;
}

void Tour::set_tour(const std::vector<int>& new_nodes, const Instance& inst) {
    set_tour_only(new_nodes);
    recompute_length(inst);
}

void Tour::rebuild_index() {
    if (N < 0) {
        throw std::logic_error("Tour has negative N");
    }
    pos.assign(static_cast<std::size_t>(N), -1);
    in_set.assign(static_cast<std::size_t>(N), 0U);
    for (int i = 0; i < k; ++i) {
        const int node = nodes[static_cast<std::size_t>(i)];
        if (node < 0 || node >= N) {
            throw std::out_of_range("tour node outside instance range");
        }
        if (pos[static_cast<std::size_t>(node)] >= 0) {
            throw std::logic_error("duplicate node in tour");
        }
        pos[static_cast<std::size_t>(node)] = i;
        in_set[static_cast<std::size_t>(node)] = 1U;
    }
}

void Tour::recompute_length(const Instance& inst) {
    edge_len.assign(static_cast<std::size_t>(k), 0.0);
    length = 0.0;
    if (k <= 1) {
        edge_valid = true;
        return;
    }
    for (int i = 0; i < k; ++i) {
        const int next = (i + 1 == k) ? 0 : (i + 1);
        const double d = inst.dist(nodes[static_cast<std::size_t>(i)], nodes[static_cast<std::size_t>(next)]);
        edge_len[static_cast<std::size_t>(i)] = d;
        length += d;
    }
    edge_valid = true;
}

void Tour::ensure_edges(const Instance& inst) {
    if (!edge_valid || static_cast<int>(edge_len.size()) != k) {
        recompute_length(inst);
    }
}

void Tour::reverse_segment_nodes(int lo, int hi) {
    while (lo < hi) {
        std::swap(nodes[static_cast<std::size_t>(lo)], nodes[static_cast<std::size_t>(hi)]);
        pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(lo)])] = lo;
        pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(hi)])] = hi;
        ++lo;
        --hi;
    }
    if (lo == hi) {
        pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(lo)])] = lo;
    }
}

void Tour::reverse_cyclic_nodes(int start, int len) {
    if (k <= 0 || len <= 1) {
        return;
    }
    int i = start % k;
    if (i < 0) { i += k; }
    int j = (start + len - 1) % k;
    if (j < 0) { j += k; }
    for (int s = 0; s < len / 2; ++s) {
        std::swap(nodes[static_cast<std::size_t>(i)], nodes[static_cast<std::size_t>(j)]);
        pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(i)])] = i;
        pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(j)])] = j;
        i = (i + 1 == k) ? 0 : (i + 1);
        j = (j == 0) ? (k - 1) : (j - 1);
    }
}

void Tour::recompute_edge_interval(const Instance& inst, int start, int count) {
    if (k <= 0) {
        edge_len.clear();
        length = 0.0;
        edge_valid = true;
        return;
    }
    if (static_cast<int>(edge_len.size()) != k) {
        recompute_length(inst);
        return;
    }
    start %= k;
    if (start < 0) {
        start += k;
    }
    count = std::max(0, std::min(count, k));
    int idx = start;
    for (int t = 0; t < count; ++t) {
        const int next = (idx + 1 == k) ? 0 : (idx + 1);
        edge_len[static_cast<std::size_t>(idx)] = inst.dist(nodes[static_cast<std::size_t>(idx)], nodes[static_cast<std::size_t>(next)]);
        idx = next;
    }
    edge_valid = true;
}

void Tour::affected_interval(int remove_pos, int insert_pos, int& start, int& count) const noexcept {
    if (k <= 0) {
        start = 0;
        count = 0;
        return;
    }
    if (insert_pos < remove_pos) {
        start = (insert_pos == 0) ? (k - 1) : (insert_pos - 1);
        count = remove_pos - insert_pos + 2;
    } else if (insert_pos > remove_pos) {
        start = (remove_pos == 0) ? (k - 1) : (remove_pos - 1);
        count = insert_pos - remove_pos + 2;
    } else {
        start = (remove_pos == 0) ? (k - 1) : (remove_pos - 1);
        count = 2;
    }
    if (count > k) {
        count = k;
    }
}

void Tour::apply_two_opt(int first_edge, int second_edge, const Instance& inst, double delta) {
    ensure_edges(inst);
    if (k < 4 || first_edge < 0 || second_edge <= first_edge || second_edge >= k) {
        return;
    }
    const int lo = first_edge + 1;
    const int hi = second_edge;
    const int inner = hi - lo + 1;   // nodes in the inner arc [lo, hi]
    const int outer = k - inner;     // nodes in the complementary (wrapping) arc
    // Reversing either arc of the cycle produces the same undirected tour, so
    // reverse whichever is shorter to bound the work at k/2 instead of always
    // paying the inner length (which can be ~k for far-apart edge pairs).
    if (inner <= outer) {
        reverse_segment_nodes(lo, hi);
        recompute_edge_interval(inst, first_edge, inner + 1);
    } else {
        reverse_cyclic_nodes((hi + 1 == k) ? 0 : (hi + 1), outer);
        recompute_edge_interval(inst, second_edge, outer + 1);
    }
    length += delta;
    edge_valid = true;
}

void Tour::swap_post_rem_nodes(int remove_pos, int post_remove_pred, int add_node) {
    const int removed = nodes[static_cast<std::size_t>(remove_pos)];
    const int insert_pos = post_remove_pred + 1;
    pos[static_cast<std::size_t>(removed)] = -1;
    in_set[static_cast<std::size_t>(removed)] = 0U;
    in_set[static_cast<std::size_t>(add_node)] = 1U;
    if (insert_pos < remove_pos) {
        for (int i = remove_pos; i > insert_pos; --i) {
            nodes[static_cast<std::size_t>(i)] = nodes[static_cast<std::size_t>(i - 1)];
            pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(i)])] = i;
        }
        nodes[static_cast<std::size_t>(insert_pos)] = add_node;
        pos[static_cast<std::size_t>(add_node)] = insert_pos;
    } else if (insert_pos > remove_pos) {
        for (int i = remove_pos; i < insert_pos; ++i) {
            nodes[static_cast<std::size_t>(i)] = nodes[static_cast<std::size_t>(i + 1)];
            pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(i)])] = i;
        }
        nodes[static_cast<std::size_t>(insert_pos)] = add_node;
        pos[static_cast<std::size_t>(add_node)] = insert_pos;
    } else {
        nodes[static_cast<std::size_t>(remove_pos)] = add_node;
        pos[static_cast<std::size_t>(add_node)] = remove_pos;
    }
}

void Tour::move_node_post_rem_nodes(int remove_pos, int post_remove_pred) {
    const int node = nodes[static_cast<std::size_t>(remove_pos)];
    const int insert_pos = post_remove_pred + 1;
    if (insert_pos < remove_pos) {
        for (int i = remove_pos; i > insert_pos; --i) {
            nodes[static_cast<std::size_t>(i)] = nodes[static_cast<std::size_t>(i - 1)];
            pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(i)])] = i;
        }
        nodes[static_cast<std::size_t>(insert_pos)] = node;
        pos[static_cast<std::size_t>(node)] = insert_pos;
    } else if (insert_pos > remove_pos) {
        for (int i = remove_pos; i < insert_pos; ++i) {
            nodes[static_cast<std::size_t>(i)] = nodes[static_cast<std::size_t>(i + 1)];
            pos[static_cast<std::size_t>(nodes[static_cast<std::size_t>(i)])] = i;
        }
        nodes[static_cast<std::size_t>(insert_pos)] = node;
        pos[static_cast<std::size_t>(node)] = insert_pos;
    }
}

void Tour::apply_swap_post_rem(int remove_pos, int post_remove_pred, int add_node, const Instance& inst, double delta) {
    ensure_edges(inst);
    const int insert_pos = post_remove_pred + 1;
    swap_post_rem_nodes(remove_pos, post_remove_pred, add_node);
    int start = 0;
    int count = 0;
    affected_interval(remove_pos, insert_pos, start, count);
    recompute_edge_interval(inst, start, count);
    length += delta;
    edge_valid = true;
}

void Tour::apply_move_post_rem(int remove_pos, int post_remove_pred, const Instance& inst, double delta) {
    ensure_edges(inst);
    const int insert_pos = post_remove_pred + 1;
    move_node_post_rem_nodes(remove_pos, post_remove_pred);
    int start = 0;
    int count = 0;
    affected_interval(remove_pos, insert_pos, start, count);
    recompute_edge_interval(inst, start, count);
    length += delta;
    edge_valid = true;
}


bool Tour::check_invariants() const {
    if (k != static_cast<int>(nodes.size())) {
        return false;
    }
    if (static_cast<int>(pos.size()) != N || static_cast<int>(in_set.size()) != N) {
        return false;
    }
    std::vector<int> seen(static_cast<std::size_t>(N), 0);
    for (int i = 0; i < k; ++i) {
        const int node = nodes[static_cast<std::size_t>(i)];
        if (node < 0 || node >= N) {
            return false;
        }
        if (++seen[static_cast<std::size_t>(node)] != 1) {
            return false;
        }
        if (pos[static_cast<std::size_t>(node)] != i || in_set[static_cast<std::size_t>(node)] == 0U) {
            return false;
        }
    }
    for (int node = 0; node < N; ++node) {
        const bool present = seen[static_cast<std::size_t>(node)] != 0;
        if (present != (in_set[static_cast<std::size_t>(node)] != 0U)) {
            return false;
        }
        if (present && pos[static_cast<std::size_t>(node)] < 0) {
            return false;
        }
        if (!present && pos[static_cast<std::size_t>(node)] != -1) {
            return false;
        }
    }
    if (edge_valid && static_cast<int>(edge_len.size()) != k) {
        return false;
    }
    return true;
}

int post_idx_from_cur(int current_index, int removed_index) noexcept {
    return (current_index < removed_index) ? current_index : (current_index - 1);
}

int next_live_idx(int index, int removed_index, int n) noexcept {
    int next = index + 1;
    if (next >= n) { next = 0; }
    if (next == removed_index) {
        ++next;
        if (next >= n) { next = 0; }
    }
    return next;
}

int prev_live_idx(int index, int removed_index, int n) noexcept {
    int prev = (index == 0) ? (n - 1) : (index - 1);
    if (prev == removed_index) {
        prev = (prev == 0) ? (n - 1) : (prev - 1);
    }
    return prev;
}

double cycle_length(const Instance& inst, const std::vector<int>& nodes) {
    const int k = static_cast<int>(nodes.size());
    if (k <= 1) {
        return 0.0;
    }
    double len = 0.0;
    for (int i = 0; i < k; ++i) {
        const int next = (i + 1 == k) ? 0 : (i + 1);
        len += inst.dist(nodes[static_cast<std::size_t>(i)], nodes[static_cast<std::size_t>(next)]);
    }
    return len;
}

} // namespace aldous_tsp
