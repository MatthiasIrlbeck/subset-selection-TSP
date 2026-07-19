#pragma once

#include "aldous_tsp/instance.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace aldous_tsp {

std::uint64_t subset_hash_nodes(const std::vector<int>& nodes) noexcept;
std::vector<int> canonical_set_key(std::vector<int> nodes);
std::vector<int> canonical_cycle_key(const std::vector<int>& nodes);

struct EliteEntry {
    double length = std::numeric_limits<double>::infinity();
    std::vector<int> nodes;
    std::vector<int> canonical_key;
    std::uint64_t hash = 0;
};

enum class EliteMode {
    Ordered,
    Cycle,
    Set
};

class ElitePool {
public:
    explicit ElitePool(int keep = 0, EliteMode mode = EliteMode::Set) : keep_(keep), mode_(mode) {}

    void try_add(const std::vector<int>& nodes, double length);
    [[nodiscard]] const std::vector<EliteEntry>& entries() const noexcept { return entries_; }
    [[nodiscard]] std::vector<std::vector<int>> export_nodes() const;

private:
    int keep_ = 0;
    EliteMode mode_ = EliteMode::Set;
    std::vector<EliteEntry> entries_;

    [[nodiscard]] std::vector<int> make_key(const std::vector<int>& nodes) const;
};

class Tour {
public:
    int N = 0;
    int k = 0;
    std::vector<int> nodes;
    std::vector<int> pos;
    std::vector<unsigned char> in_set;
    std::vector<double> edge_len;
    double length = 0.0;
    bool edge_valid = false;

    void init(int n);
    void set_tour_only(const std::vector<int>& new_nodes);
    void set_tour(const std::vector<int>& new_nodes, const Instance& inst);
    void rebuild_index();
    void recompute_length(const Instance& inst);
    void ensure_edges(const Instance& inst);
    void reverse_segment_nodes(int lo, int hi);
    void reverse_cyclic_nodes(int start, int len);
    void recompute_edge_interval(const Instance& inst, int start, int count);
    void affected_interval(int remove_pos, int insert_pos, int& start, int& count) const noexcept;
    void apply_two_opt(int first_edge, int second_edge, const Instance& inst, double delta);
    void apply_swap_post_rem(int remove_pos, int post_remove_pred, int add_node, const Instance& inst, double delta);
    void apply_move_post_rem(int remove_pos, int post_remove_pred, const Instance& inst, double delta);
    [[nodiscard]] bool check_invariants() const;

private:
    void swap_post_rem_nodes(int remove_pos, int post_remove_pred, int add_node);
    void move_node_post_rem_nodes(int remove_pos, int post_remove_pred);
};

[[nodiscard]] double cycle_length(const Instance& inst, const std::vector<int>& nodes);
[[nodiscard]] int post_idx_from_cur(int current_index, int removed_index) noexcept;
[[nodiscard]] int next_live_idx(int index, int removed_index, int n) noexcept;
[[nodiscard]] int prev_live_idx(int index, int removed_index, int n) noexcept;

} // namespace aldous_tsp
