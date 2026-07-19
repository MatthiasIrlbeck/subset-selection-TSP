#pragma once

#include "aldous_tsp/solver.hpp"
#include "aldous_tsp/oracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aldous_tsp {

using Clock = std::chrono::steady_clock;

std::vector<int> all_nodes(int n);
std::vector<int> random_subset(int n, int k, Rng& rng);
RestartRecord make_restart_record(const Instance& inst,
                                  const std::vector<int>& nodes,
                                  double length,
                                  RestartKind kind);
void push_unique(std::vector<int>& values,
                 int value,
                 const std::vector<unsigned char>* banned = nullptr,
                 int cap = std::numeric_limits<int>::max());

std::vector<int> nearest_to_point_seed(const Instance& inst, double cx, double cy, int k, Rng& rng);
std::vector<int> dense_seed(const Instance& inst, int k, Rng& rng, int rank_offset = 0);
std::vector<std::vector<int>> make_smallp_seed_pool(const Instance& inst, int k, Rng& rng, int max_budget);
std::vector<int> highp_delete_seed(const Instance& inst, const std::vector<int>& parent, int k, Rng& rng, int mode = 0);
std::vector<int> segment_delete_seed(const Instance& inst, const std::vector<int>& parent, int k, Rng& rng);
std::vector<int> resize_seed(const Instance& inst, const std::vector<int>& seed, int k, Rng& rng, int mode = 0);

int next_live_index(int idx, int removed, int n) noexcept;
int next_live_index_after_remove(int idx, int removed, int n) noexcept;
int post_index_from_current(int current_index, int removed) noexcept;

struct SwapInsertionMove {
    bool valid = false;
    int remove_pos = -1;
    int add_node = -1;
    int post_pred = -1;
    double delta = std::numeric_limits<double>::infinity();
    double new_length = std::numeric_limits<double>::infinity();
};

struct SwapMoveEval {
    bool valid = false;
    double delta = std::numeric_limits<double>::infinity();
    int post_remove_pred = 0;
};

SwapInsertionMove find_best_insert_after_remove(const Instance& inst, const Tour& tour, int remove_pos, int add_node);
SwapInsertionMove find_best_insert_after_remove_windowed(const Instance& inst, const Tour& tour, int remove_pos, int add_node, int window);

// Live spatial index over the CURRENT subset members.
//
// Why this exists. The SA move is "remove member ri, insert candidate `add`".
// Candidates come from the KNN rows of the removed node and its tour
// neighbours -- and from up to 24 UNIFORMLY RANDOM nodes. Those random
// candidates are the entire contraction mechanism: they are the only proposals
// that can pull a member out of a bad region into a good one. The windowed
// kernel cannot place them. Its slot set is (a) tour positions near the REMOVED
// node -- spatially irrelevant to `add` -- and (b) slots adjacent to `add`'s
// in-tour KNN, where the KNN row is over the FULL point set: at p = k/N = 0.01
// the expected number of a node's 40 nearest that are in the subset is 0.4, so
// (b) is empty almost always. Windowed insertion therefore windows around the
// wrong node, and the exact O(k) scan "works" only by brute force.
//
// This index answers the question the kernel actually needs -- "which CURRENT
// members are near `add`?" -- in O(1) per query, at any subset density. Cells
// are re-sized as the subset contracts so occupancy stays low in both regimes:
// a spread-out subset (~1 member/cell) and a tight cluster (which would
// otherwise pile 70+ members into one cell) both cost a few dozen distance
// evaluations per query, against 2000 for the exact scan at k=2000.
class SubsetIndex {
public:
    void build(const Instance& inst, const Tour& tour);
    void add_member(const Instance& inst, int node);
    void remove_member(const Instance& inst, int node);
    // The `m` nearest current members to `query_node`, excluding `exclude`
    // (the node about to be removed). Deterministic: sorted by (distance, id).
    // `max_rings < 0` searches until the m nearest are provably found (exact).
    // The SA passes a small cap: a candidate with no members within the capped
    // radius is one whose insertion costs ~2x its distance to the subset, so the
    // move is rejected whatever slot it gets -- paying an unbounded search to
    // place it precisely would be spending the budget on a foregone conclusion.
    void nearest(const Instance& inst, int query_node, int m, int exclude,
                 std::vector<int>& out, int max_rings = -1) const;
    int size() const noexcept { return count_; }

private:
    int cell_of(const Instance& inst, int node) const noexcept;
    void relink(const Instance& inst);

    int cells_side_ = 1;
    int cells_side_cap_ = 1;
    double cell_len_ = 1.0;
    int count_ = 0;
    int nonempty_ = 0;
    std::vector<int> head_;   // cells_side_^2 buckets, -1 = empty
    std::vector<int> next_;   // N
    std::vector<int> prev_;   // N
    std::vector<int> members_;  // scratch for rebuilds
};

// Insertion kernel whose candidate slots are the tour edges adjacent to the
// `neighbors` nearest CURRENT members of `add_node` (plus the vacated slot and
// a small local window). With neighbors >= k this gathers every slot and is
// therefore identical to the exact scan -- pinned by a unit test.
SwapInsertionMove find_best_insert_after_remove_spatial(const Instance& inst, const Tour& tour,
                                                        const SubsetIndex& index, int remove_pos,
                                                        int add_node, int neighbors, int window);
SwapMoveEval evaluate_swap_after_remove(const Instance& inst,
                                        const Tour& tour,
                                        int remove_pos,
                                        int add_node,
                                        const std::vector<int>* pred_positions = nullptr);
SwapMoveEval evaluate_move_after_remove(const Instance& inst,
                                        const Tour& tour,
                                        int remove_pos,
                                        const std::vector<int>* pred_positions = nullptr);
std::vector<int> collect_add_candidates(const Instance& inst, const Tour& tour, int remove_pos, Rng& rng, int cap);
void collect_add_candidates_into(const Instance& inst, const Tour& tour, int remove_pos, Rng& rng, int cap, std::vector<int>& candidates);
int choose_swap_candidate(const Instance& inst, const Tour& tour, int remove_pos, Rng& rng);

// Path relinking between elite solutions with symmetric difference above this
// cap is skipped: relink cost grows superlinearly in the difference while its
// marginal value over SA/restart search collapses for distant solution pairs.
inline constexpr int kPathRelinkMaxDiff = 64;

struct PathRelinkStep {
    bool valid = false;
    int remove_pos = -1;
    int add_node = -1;
    double delta = std::numeric_limits<double>::infinity();
};

// Exact best single relink step: over all (remove position, add node) pairs,
// the minimum-delta swap where the added node goes to its best insertion
// position. Costs O(|add| * k + |remove| * |add|) instead of the naive
// O(|remove| * |add| * k) full cross-product evaluation.
PathRelinkStep path_relink_best_step(const Instance& inst,
                                     const Tour& tour,
                                     const std::vector<int>& remove_positions,
                                     const std::vector<int>& add_nodes);

// Nearest-neighbor candidates restricted to the current subset members. The
// full-instance KNN list (knn_k entries) contains on average only knn_k * p
// subset members, so candidate local search degenerates at small/mid p. This
// table stores, for every subset member, its m nearest *subset* members,
// computed exactly via the instance grid (or brute force when no grid is
// available). It stays valid while membership is unchanged (2-opt and or-opt
// only reorder), and must be rebuilt after subset swaps.
struct SubsetCandidateTable {
    int m = 0;
    std::vector<int> row_of_node;  // size N, -1 for non-members
    std::vector<int> ids;          // rows * m node ids, -1 padded
    std::vector<double> dist;      // rows * m distances, +inf padded
};

void build_subset_candidates(const Instance& inst, const Tour& tour, int m, SubsetCandidateTable& table);
// Builds into thread-local storage and returns it, or nullptr when a subset
// table is not applicable (full tour, tiny k, or no grid with large k).
const SubsetCandidateTable* maybe_subset_candidates(const Instance& inst, const Tour& tour);

// Effective SA iteration budget for a size-k subset solve.
int effective_sa_iters(const SolverOptions& options, int k, int N) noexcept;

int two_opt_candidate_descent(Tour& tour, const Instance& inst, int max_passes, int cand_cap, SearchStats* stats, const SubsetCandidateTable* table = nullptr);
int or_opt_1_candidate_descent(Tour& tour, const Instance& inst, int max_passes, int cand_cap, SearchStats* stats, const SubsetCandidateTable* table = nullptr);
// Moves a segment of seg_len (2 or 3) consecutive nodes to its best candidate
// insertion edge, in either orientation. Best-improvement per pass; the tour
// is rebuilt on apply, so applies cost O(k) while evaluations stay O(1).
int or_opt_segment_candidate_descent(Tour& tour, const Instance& inst, int seg_len, int max_passes, int cand_cap, SearchStats* stats, const SubsetCandidateTable* table = nullptr);
void polish_tour(Tour& tour, const Instance& inst, const SolverOptions& options, SearchStats* stats, int strength = 1);
void final_polish_tour(Tour& tour, const Instance& inst, const SolverOptions& options, SearchStats* stats, int strength = 1);
bool use_all_polish_exhaustive_two_opt(const SolverOptions& options, int k) noexcept;
void polish_elite_with_oracle(ElitePool& elite,
                              const Instance& inst,
                              const SolverOptions& options,
                              bool full_tsp,
                              int top_keep,
                              SearchStats* stats);
void perturb_three_cut(Tour& tour, Rng& rng);

bool highp_delete_exchange_descent(Tour& tour,
                                   const Instance& inst,
                                   const std::vector<int>& reference,
                                   const SolverOptions& options,
                                   SearchStats* stats,
                                   int passes);
bool regret_repair_cycle(std::vector<int>& cycle,
                         const Instance& inst,
                         int target_k,
                         const std::vector<int>& pool,
                         const std::vector<unsigned char>& banned);
bool subset_ruin_recreate_lns(Tour& tour,
                              const Instance& inst,
                              Rng& rng,
                              const SolverOptions& options,
                              SearchStats* stats,
                              int rounds);
bool subset_pair_exchange_descent(Tour& tour,
                                  const Instance& inst,
                                  Rng& rng,
                                  const SolverOptions& options,
                                  SearchStats* stats,
                                  int passes);
bool subset_path_relink_bidirectional(const Instance& inst,
                                      const std::vector<int>& a,
                                      const std::vector<int>& b,
                                      Rng& rng,
                                      const SolverOptions& options,
                                      std::vector<int>& best_nodes,
                                      double& best_len,
                                      SearchStats* stats);

int subset_swap_descent_impl(Tour& tour, const Instance& inst, int max_passes, bool enable_two_opt, SearchStats* stats);

} // namespace aldous_tsp
