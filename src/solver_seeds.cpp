#include "solver_internal.hpp"

#include <queue>
#include <unordered_map>

namespace aldous_tsp {

std::vector<int> nearest_to_point_seed(const Instance& inst, double cx, double cy, int k, Rng& rng) {
    if (inst.periodic) {
        const PeriodicDomain domain{inst.side};
        cx = domain.normalize(cx);
        cy = domain.normalize(cy);
    }
    std::vector<int> order(static_cast<std::size_t>(inst.N));
    std::vector<double> scores(static_cast<std::size_t>(inst.N));
    std::iota(order.begin(), order.end(), 0);
    for (int node = 0; node < inst.N; ++node) {
        scores[static_cast<std::size_t>(node)] = inst.dist2_to_canonical_point(node, cx, cy);
    }
    auto cmp = [&](int a, int b) {
        const double da = scores[static_cast<std::size_t>(a)];
        const double db = scores[static_cast<std::size_t>(b)];
        if (da != db) {
            return da < db;
        }
        return a < b;
    };
    if (k < inst.N) {
        std::nth_element(order.begin(), order.begin() + k, order.end(), cmp);
    }
    order.resize(static_cast<std::size_t>(k));
    const int start = (k > 0) ? rng.randint(k) : 0;
    return nearest_neighbor_order(inst, order, start);
}

std::vector<int> dense_seed(const Instance& inst, int k, Rng& rng, int rank_offset) {
    if (inst.N <= 0 || k <= 0) {
        return {};
    }
    if (inst.knn_k <= 0) {
        return nearest_neighbor_order(inst, random_subset(inst.N, k, rng), 0);
    }
    const int rank_count = std::max(1, std::min(inst.knn_k, std::max(4, std::min(32 + rank_offset, std::max(4, k / 2)))));
    std::vector<std::pair<double, int>> scored;
    scored.reserve(static_cast<std::size_t>(inst.N));
    for (int node = 0; node < inst.N; ++node) {
        double score = 0.0;
        for (int r = 0; r < rank_count; ++r) {
            score += inst.knn_d_at(node, r);
        }
        score /= static_cast<double>(rank_count);
        score += 0.25 * inst.knn_d_at(node, rank_count - 1);
        scored.emplace_back(score, node);
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    const int choice = std::min(static_cast<int>(scored.size()) - 1, rng.randint(std::max(1, std::min(8, static_cast<int>(scored.size())))));
    const int center = scored[static_cast<std::size_t>(choice)].second;
    PointMeanAccumulator center_mean(inst.periodic, inst.side);
    center_mean.add(inst.points[static_cast<std::size_t>(center)]);
    const int lim = std::min(inst.knn_k, std::max(4, std::min(24, k / 2)));
    for (int r = 0; r < lim; ++r) {
        const int neighbor = inst.knn_at(center, r);
        center_mean.add(inst.points[static_cast<std::size_t>(neighbor)]);
    }
    const Point seed_center = center_mean.mean();
    return nearest_to_point_seed(inst, seed_center.x, seed_center.y, k, rng);
}

std::vector<std::vector<int>> make_smallp_seed_pool(const Instance& inst, int k, Rng& rng, int max_budget) {
    std::vector<std::vector<int>> out;
    const double p = static_cast<double>(k) / static_cast<double>(std::max(1, inst.N));
    if (p > 0.08 || max_budget <= 0) {
        return out;
    }
    const int budget = std::min(max_budget, (p <= 0.03 ? 8 : 5) + (inst.N >= 3000 ? 2 : 0));
    ElitePool dedup(budget, EliteMode::Set);
    auto add_seed = [&](std::vector<int> seed) {
        if (static_cast<int>(out.size()) >= budget || static_cast<int>(seed.size()) != k) {
            return;
        }
        const double len = cycle_length(inst, seed);
        const auto before = dedup.entries().size();
        dedup.try_add(seed, len);
        if (dedup.entries().size() != before) {
            out.push_back(std::move(seed));
        }
    };
    for (int i = 0; i < budget && static_cast<int>(out.size()) < budget; ++i) {
        add_seed(dense_seed(inst, k, rng, 4 * i));
    }
    if (inst.gx > 0 && inst.gy > 0 && static_cast<int>(out.size()) < budget) {
        std::vector<std::pair<int, int>> cells;
        cells.reserve(static_cast<std::size_t>(inst.gx * inst.gy));
        for (int c = 0; c < inst.gx * inst.gy; ++c) {
            const int cnt = inst.cell_begin[static_cast<std::size_t>(c + 1)] - inst.cell_begin[static_cast<std::size_t>(c)];
            if (cnt > 0) {
                cells.emplace_back(-cnt, c);
            }
        }
        std::sort(cells.begin(), cells.end());
        for (const auto& pr : cells) {
            const int c = pr.second;
            const int begin = inst.cell_begin[static_cast<std::size_t>(c)];
            const int end = inst.cell_begin[static_cast<std::size_t>(c + 1)];
            PointMeanAccumulator cell_mean(inst.periodic, inst.side);
            for (int pp = begin; pp < end; ++pp) {
                const int node = inst.cell_points[static_cast<std::size_t>(pp)];
                cell_mean.add(inst.points[static_cast<std::size_t>(node)]);
            }
            const Point center = cell_mean.mean();
            add_seed(nearest_to_point_seed(inst, center.x, center.y, k, rng));
            if (static_cast<int>(out.size()) >= budget) {
                break;
            }
        }
    }
    return out;
}

namespace {

void validate_resize_seed(const Instance& inst, const std::vector<int>& seed) {
    if (static_cast<int>(seed.size()) > inst.N) {
        throw std::invalid_argument("resize seed exceeds the instance cardinality");
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(inst.N), 0U);
    for (const int node : seed) {
        if (node < 0 || node >= inst.N) {
            throw std::invalid_argument("resize seed contains an out-of-range node");
        }
        if (seen[static_cast<std::size_t>(node)] != 0U) {
            throw std::invalid_argument("resize seed contains a duplicate node");
        }
        seen[static_cast<std::size_t>(node)] = 1U;
    }
}

struct DeleteHeapEntry {
    double score = -std::numeric_limits<double>::infinity();
    int node = -1;
    std::uint64_t version = 0;
};

struct DeleteHeapLess {
    bool operator()(const DeleteHeapEntry& lhs,
                    const DeleteHeapEntry& rhs) const noexcept {
        if (lhs.score != rhs.score) {
            return lhs.score < rhs.score;
        }
        return lhs.node > rhs.node;
    }
};

class ShrinkSeedState {
public:
    ShrinkSeedState(const Instance& inst,
                    const std::vector<int>& parent,
                    Rng& rng,
                    const int mode)
        : inst_(inst), rng_(rng), mode_(mode), original_(parent),
          prev_(static_cast<std::size_t>(inst.N), -1),
          next_(static_cast<std::size_t>(inst.N), -1),
          alive_(static_cast<std::size_t>(inst.N), 0U),
          version_(static_cast<std::size_t>(inst.N), 0U),
          count_(static_cast<int>(parent.size())) {
        validate_resize_seed(inst_, parent);
        if (parent.empty()) {
            return;
        }
        const int m = static_cast<int>(parent.size());
        for (int i = 0; i < m; ++i) {
            const int node = parent[static_cast<std::size_t>(i)];
            prev_[static_cast<std::size_t>(node)] =
                parent[static_cast<std::size_t>((i - 1 + m) % m)];
            next_[static_cast<std::size_t>(node)] =
                parent[static_cast<std::size_t>((i + 1) % m)];
            alive_[static_cast<std::size_t>(node)] = 1U;
        }
        for (const int node : parent) {
            refresh(node);
        }
    }

    int size() const noexcept { return count_; }

    std::vector<int> shrink_to(const int target) {
        if (target < 0 || target > count_) {
            throw std::invalid_argument("shrink target is outside the current seed size");
        }
        while (count_ > target) {
            std::vector<DeleteHeapEntry> top;
            const int width = mode_ == 1 ? std::min(count_, 8) : 1;
            top.reserve(static_cast<std::size_t>(width));
            while (static_cast<int>(top.size()) < width) {
                top.push_back(pop_valid());
            }
            const int choice = mode_ == 1 ? rng_.randint(width) : 0;
            const int removed = top[static_cast<std::size_t>(choice)].node;
            for (int i = 0; i < width; ++i) {
                if (i != choice) {
                    heap_.push(top[static_cast<std::size_t>(i)]);
                }
            }
            erase_node(removed);
        }
        return materialize();
    }

private:
    double score(const int node) const {
        if (count_ <= 1) {
            return 0.0;
        }
        const int before = prev_[static_cast<std::size_t>(node)];
        const int after = next_[static_cast<std::size_t>(node)];
        double value = inst_.dist(before, node) + inst_.dist(node, after)
                     - inst_.dist(before, after);
        if (mode_ == 2 && inst_.knn_k > 0) {
            value += 0.15 * inst_.knn_d_at(
                node, std::min(inst_.knn_k - 1, 10));
        }
        return value;
    }

    void refresh(const int node) {
        if (node < 0 || alive_[static_cast<std::size_t>(node)] == 0U) {
            return;
        }
        const std::uint64_t version = ++version_[static_cast<std::size_t>(node)];
        heap_.push(DeleteHeapEntry{score(node), node, version});
    }

    DeleteHeapEntry pop_valid() {
        while (!heap_.empty()) {
            const DeleteHeapEntry entry = heap_.top();
            heap_.pop();
            if (entry.node >= 0
                && alive_[static_cast<std::size_t>(entry.node)] != 0U
                && version_[static_cast<std::size_t>(entry.node)] == entry.version) {
                return entry;
            }
        }
        throw std::logic_error("shrink seed heap lost every live node");
    }

    void erase_node(const int node) {
        if (alive_[static_cast<std::size_t>(node)] == 0U) {
            throw std::logic_error("shrink seed selected an already deleted node");
        }
        const int before = prev_[static_cast<std::size_t>(node)];
        const int after = next_[static_cast<std::size_t>(node)];
        alive_[static_cast<std::size_t>(node)] = 0U;
        ++version_[static_cast<std::size_t>(node)];
        --count_;
        if (count_ == 0) {
            return;
        }
        next_[static_cast<std::size_t>(before)] = after;
        prev_[static_cast<std::size_t>(after)] = before;
        refresh(before);
        if (after != before) {
            refresh(after);
        }
    }

    std::vector<int> materialize() const {
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(count_));
        if (count_ == 0) {
            return out;
        }
        int anchor = -1;
        for (const int node : original_) {
            if (alive_[static_cast<std::size_t>(node)] != 0U) {
                anchor = node;
                break;
            }
        }
        if (anchor < 0) {
            throw std::logic_error("shrink seed has no live anchor");
        }
        int node = anchor;
        do {
            out.push_back(node);
            node = next_[static_cast<std::size_t>(node)];
        } while (node != anchor && static_cast<int>(out.size()) <= count_);
        if (static_cast<int>(out.size()) != count_) {
            throw std::logic_error("shrink seed cycle is inconsistent");
        }
        return out;
    }

    const Instance& inst_;
    Rng& rng_;
    int mode_ = 0;
    std::vector<int> original_;
    std::vector<int> prev_;
    std::vector<int> next_;
    std::vector<unsigned char> alive_;
    std::vector<std::uint64_t> version_;
    int count_ = 0;
    std::priority_queue<DeleteHeapEntry,
                        std::vector<DeleteHeapEntry>,
                        DeleteHeapLess> heap_;
};

struct InsertionProfile {
    bool valid = false;
    int predecessor = -1;
    double cost = std::numeric_limits<double>::infinity();
    int last_seen_step = -1;
};

class GrowthSeedState {
public:
    GrowthSeedState(const Instance& inst,
                    const std::vector<int>& seed,
                    const int mode)
        : inst_(inst), mode_(mode),
          next_(static_cast<std::size_t>(inst.N), -1),
          prev_(static_cast<std::size_t>(inst.N), -1),
          in_set_(static_cast<std::size_t>(inst.N), 0U),
          rank_(static_cast<std::size_t>(inst.N), -1),
          count_(static_cast<int>(seed.size())) {
        validate_resize_seed(inst_, seed);
        if (seed.empty()) {
            return;
        }
        anchor_ = seed.front();
        const int m = static_cast<int>(seed.size());
        for (int i = 0; i < m; ++i) {
            const int node = seed[static_cast<std::size_t>(i)];
            prev_[static_cast<std::size_t>(node)] =
                seed[static_cast<std::size_t>((i - 1 + m) % m)];
            next_[static_cast<std::size_t>(node)] =
                seed[static_cast<std::size_t>((i + 1) % m)];
            in_set_[static_cast<std::size_t>(node)] = 1U;
        }
        rebuild_order();
    }

    int size() const noexcept { return count_; }

    std::vector<int> grow_to(const int target) {
        if (target < count_ || target > inst_.N) {
            throw std::invalid_argument("growth target is outside the instance domain");
        }
        while (count_ < target) {
            const std::vector<int> pool = candidate_pool();
            if (pool.empty()) {
                throw std::logic_error("growth seed found no free candidate");
            }
            int best_node = -1;
            int best_predecessor = -1;
            double best_cost = std::numeric_limits<double>::infinity();
            for (const int candidate : pool) {
                InsertionProfile& profile = profiles_[candidate];
                profile.last_seen_step = step_;
                if (!profile.valid) {
                    recompute_profile(candidate, profile);
                }
                // Candidate order is the legacy global tie order: update only
                // on a strict improvement.
                if (profile.cost < best_cost) {
                    best_cost = profile.cost;
                    best_node = candidate;
                    best_predecessor = profile.predecessor;
                }
            }
            if (best_node < 0 || best_predecessor < 0) {
                throw std::logic_error("growth seed failed to select an insertion");
            }
            insert_after(best_predecessor, best_node);
            ++step_;
            prune_profiles();
        }
        return order_;
    }

private:
    std::vector<int> candidate_pool() const {
        std::vector<int> pool;
        pool.reserve(inst_.N <= 600 ? static_cast<std::size_t>(inst_.N) : 160U);
        const int limit = std::min(inst_.knn_k, 16 + 4 * mode_);
        for (const int seed_node : order_) {
            for (int r = 0; r < limit; ++r) {
                const int candidate = inst_.knn_at(seed_node, r);
                if (candidate >= 0 && candidate < inst_.N
                    && in_set_[static_cast<std::size_t>(candidate)] == 0U) {
                    push_unique(pool, candidate, nullptr, 160);
                }
            }
            if (static_cast<int>(pool.size()) >= 160) {
                break;
            }
        }
        if (pool.empty() || inst_.N <= 600) {
            for (int candidate = 0; candidate < inst_.N; ++candidate) {
                if (in_set_[static_cast<std::size_t>(candidate)] == 0U) {
                    push_unique(pool, candidate, nullptr, inst_.N);
                }
            }
        }
        return pool;
    }

    double insertion_cost(const int predecessor,
                          const int candidate) const {
        if (count_ <= 1) {
            return 0.0;
        }
        const int successor = next_[static_cast<std::size_t>(predecessor)];
        return inst_.dist(predecessor, candidate)
             + inst_.dist(candidate, successor)
             - inst_.dist(predecessor, successor);
    }

    void recompute_profile(const int candidate, InsertionProfile& profile) const {
        profile.valid = true;
        profile.predecessor = anchor_;
        profile.cost = std::numeric_limits<double>::infinity();
        if (count_ == 1) {
            profile.cost = 0.0;
            return;
        }
        for (const int predecessor : order_) {
            const double cost = insertion_cost(predecessor, candidate);
            if (cost < profile.cost) {
                profile.cost = cost;
                profile.predecessor = predecessor;
            }
        }
    }

    void insert_after(const int predecessor, const int node) {
        const int successor = next_[static_cast<std::size_t>(predecessor)];
        next_[static_cast<std::size_t>(predecessor)] = node;
        prev_[static_cast<std::size_t>(node)] = predecessor;
        next_[static_cast<std::size_t>(node)] = successor;
        prev_[static_cast<std::size_t>(successor)] = node;
        in_set_[static_cast<std::size_t>(node)] = 1U;
        ++count_;
        profiles_.erase(node);
        rebuild_order();

        for (auto& item : profiles_) {
            const int candidate = item.first;
            InsertionProfile& profile = item.second;
            if (!profile.valid) {
                continue;
            }
            if (profile.predecessor == predecessor) {
                recompute_profile(candidate, profile);
                continue;
            }
            const double first_cost = insertion_cost(predecessor, candidate);
            const double second_cost = insertion_cost(node, candidate);
            auto consider = [&](const int pred, const double cost) {
                if (cost < profile.cost
                    || (cost == profile.cost
                        && rank_[static_cast<std::size_t>(pred)]
                           < rank_[static_cast<std::size_t>(profile.predecessor)])) {
                    profile.cost = cost;
                    profile.predecessor = pred;
                }
            };
            consider(predecessor, first_cost);
            consider(node, second_cost);
        }
    }

    void rebuild_order() {
        order_.clear();
        order_.reserve(static_cast<std::size_t>(count_));
        std::fill(rank_.begin(), rank_.end(), -1);
        if (count_ == 0) {
            return;
        }
        int node = anchor_;
        do {
            rank_[static_cast<std::size_t>(node)] = static_cast<int>(order_.size());
            order_.push_back(node);
            node = next_[static_cast<std::size_t>(node)];
        } while (node != anchor_ && static_cast<int>(order_.size()) <= count_);
        if (static_cast<int>(order_.size()) != count_) {
            throw std::logic_error("growth seed cycle is inconsistent");
        }
    }

    void prune_profiles() {
        // Keep a bounded rolling cache. Re-entering candidates are recomputed
        // exactly; candidates active in recent pools retain O(1) edge updates.
        constexpr int kRetentionSteps = 8;
        if ((step_ & 7) != 0) {
            return;
        }
        for (auto it = profiles_.begin(); it != profiles_.end();) {
            if (it->second.last_seen_step + kRetentionSteps < step_) {
                it = profiles_.erase(it);
            } else {
                ++it;
            }
        }
    }

    const Instance& inst_;
    int mode_ = 0;
    int anchor_ = -1;
    std::vector<int> next_;
    std::vector<int> prev_;
    std::vector<unsigned char> in_set_;
    std::vector<int> order_;
    std::vector<int> rank_;
    int count_ = 0;
    int step_ = 0;
    std::unordered_map<int, InsertionProfile> profiles_;
};

std::vector<std::vector<int>> collect_shrink_snapshots(
    const Instance& inst,
    const std::vector<int>& parent,
    const std::vector<int>& targets,
    Rng& rng,
    const int mode) {
    ShrinkSeedState state(inst, parent, rng, mode);
    std::vector<std::vector<int>> out;
    out.reserve(targets.size());
    for (const int target : targets) {
        out.push_back(state.shrink_to(target));
    }
    return out;
}

std::vector<std::vector<int>> collect_growth_snapshots(
    const Instance& inst,
    const std::vector<int>& seed,
    const std::vector<int>& targets,
    const int mode) {
    GrowthSeedState state(inst, seed, mode);
    std::vector<std::vector<int>> out;
    out.reserve(targets.size());
    for (const int target : targets) {
        out.push_back(state.grow_to(target));
    }
    return out;
}

} // namespace

std::vector<std::vector<int>> shrink_seed_chain(
    const Instance& inst,
    const std::vector<int>& parent,
    const std::vector<int>& target_sizes,
    Rng& rng,
    const int mode) {
    int previous = static_cast<int>(parent.size());
    for (const int target : target_sizes) {
        if (target < 0 || target > previous) {
            throw std::invalid_argument(
                "shrink seed targets must be nonincreasing and in range");
        }
        previous = target;
    }
    return collect_shrink_snapshots(inst, parent, target_sizes, rng, mode);
}

std::vector<std::vector<int>> grow_seed_chain(
    const Instance& inst,
    const std::vector<int>& seed,
    const std::vector<int>& target_sizes,
    Rng& rng,
    const int mode) {
    (void)rng; // Growth is deterministic; retained for a symmetric public API.
    int previous = static_cast<int>(seed.size());
    for (const int target : target_sizes) {
        if (target < previous || target > inst.N) {
            throw std::invalid_argument(
                "growth seed targets must be nondecreasing and in range");
        }
        previous = target;
    }
    return collect_growth_snapshots(inst, seed, target_sizes, mode);
}

std::vector<int> highp_delete_seed(const Instance& inst,
                                   const std::vector<int>& parent,
                                   const int k,
                                   Rng& rng,
                                   const int mode) {
    if (static_cast<int>(parent.size()) <= k) {
        validate_resize_seed(inst, parent);
        return parent;
    }
    return shrink_seed_chain(inst, parent, {k}, rng, mode).front();
}

std::vector<int> segment_delete_seed(const Instance& inst,
                                     const std::vector<int>& parent,
                                     const int k,
                                     Rng& rng) {
    validate_resize_seed(inst, parent);
    if (static_cast<int>(parent.size()) <= k) {
        return parent;
    }
    const int q = static_cast<int>(parent.size()) - k;
    const int m = static_cast<int>(parent.size());
    const int segment = std::min(q, std::max(2, q / 2));
    std::vector<double> score(static_cast<std::size_t>(m), 0.0);
    std::vector<int> ord(static_cast<std::size_t>(m));
    std::iota(ord.begin(), ord.end(), 0);
    for (int i = 0; i < m; ++i) {
        const int before = parent[static_cast<std::size_t>((i - 1 + m) % m)];
        const int node = parent[static_cast<std::size_t>(i)];
        const int after = parent[static_cast<std::size_t>((i + 1) % m)];
        score[static_cast<std::size_t>(i)] =
            inst.dist(before, node) + inst.dist(node, after)
            - inst.dist(before, after);
    }
    std::sort(ord.begin(), ord.end(), [&](const int lhs, const int rhs) {
        if (score[static_cast<std::size_t>(lhs)]
            != score[static_cast<std::size_t>(rhs)]) {
            return score[static_cast<std::size_t>(lhs)]
                 > score[static_cast<std::size_t>(rhs)];
        }
        return parent[static_cast<std::size_t>(lhs)]
             < parent[static_cast<std::size_t>(rhs)];
    });
    const int start = ord[static_cast<std::size_t>(rng.randint(std::min(m, 3)))];
    std::vector<unsigned char> deleted(static_cast<std::size_t>(m), 0U);
    for (int i = 0; i < segment; ++i) {
        deleted[static_cast<std::size_t>((start + i) % m)] = 1U;
    }
    std::vector<int> cur;
    cur.reserve(static_cast<std::size_t>(m - segment));
    for (int i = 0; i < m; ++i) {
        if (deleted[static_cast<std::size_t>(i)] == 0U) {
            cur.push_back(parent[static_cast<std::size_t>(i)]);
        }
    }
    return highp_delete_seed(inst, cur, k, rng, 0);
}

std::vector<int> resize_seed(const Instance& inst,
                             const std::vector<int>& seed,
                             const int k,
                             Rng& rng,
                             const int mode) {
    if (k < 0 || k > inst.N) {
        throw std::invalid_argument("resize target is outside the instance domain");
    }
    if (static_cast<int>(seed.size()) == k) {
        validate_resize_seed(inst, seed);
        return seed;
    }
    if (seed.empty()) {
        return dense_seed(inst, k, rng);
    }
    if (static_cast<int>(seed.size()) > k) {
        return shrink_seed_chain(inst, seed, {k}, rng, mode).front();
    }
    return grow_seed_chain(inst, seed, {k}, rng, mode).front();
}

void apply_elite_kick(const Instance& inst, std::vector<int>& seed, Rng& rng, double fraction) {
    const int subset_size = static_cast<int>(seed.size());
    if (subset_size == 0) {
        return;
    }
    if (inst.N < 0 || subset_size > inst.N) {
        throw std::invalid_argument("elite kick seed size is outside the instance domain");
    }

    std::vector<unsigned char> in_set(static_cast<std::size_t>(inst.N), 0U);
    for (const int node : seed) {
        if (node < 0 || node >= inst.N) {
            throw std::invalid_argument("elite kick seed contains an out-of-range node");
        }
        unsigned char& present = in_set[static_cast<std::size_t>(node)];
        if (present != 0U) {
            throw std::invalid_argument("elite kick seed contains duplicate nodes");
        }
        present = 1U;
    }

    // Keep an exact free-node set with O(1) insertion/removal. The old bounded
    // random fallback could exhaust its attempts when k was close to N and
    // then reinsert an occupied node. This representation makes candidate
    // selection total: after removing one member there is always at least one
    // free node, even when k == N (in which case it is the removed node).
    std::vector<int> free_nodes;
    free_nodes.reserve(static_cast<std::size_t>(inst.N - subset_size + 1));
    std::vector<int> free_pos(static_cast<std::size_t>(inst.N), -1);
    for (int node = 0; node < inst.N; ++node) {
        if (in_set[static_cast<std::size_t>(node)] == 0U) {
            free_pos[static_cast<std::size_t>(node)] = static_cast<int>(free_nodes.size());
            free_nodes.push_back(node);
        }
    }

    auto add_free = [&](int node) {
        if (free_pos[static_cast<std::size_t>(node)] >= 0) {
            throw std::logic_error("elite kick free-node set is inconsistent");
        }
        free_pos[static_cast<std::size_t>(node)] = static_cast<int>(free_nodes.size());
        free_nodes.push_back(node);
    };
    auto remove_free = [&](int node) {
        const int position = free_pos[static_cast<std::size_t>(node)];
        if (position < 0) {
            throw std::logic_error("elite kick selected an occupied node");
        }
        const int last = free_nodes.back();
        free_nodes[static_cast<std::size_t>(position)] = last;
        free_pos[static_cast<std::size_t>(last)] = position;
        free_nodes.pop_back();
        free_pos[static_cast<std::size_t>(node)] = -1;
    };

    const double effective_fraction =
        (fraction > 0.0 && fraction < 1.0) ? fraction : 0.10;
    const int kick_count = std::min(
        subset_size,
        std::max(1, static_cast<int>(std::lround(
                        effective_fraction * static_cast<double>(subset_size)))));

    for (int step = 0; step < kick_count; ++step) {
        const int remove_pos = rng.randint(subset_size);
        const int removed = seed[static_cast<std::size_t>(remove_pos)];
        in_set[static_cast<std::size_t>(removed)] = 0U;
        add_free(removed);

        // Prefer a free KNN neighbor of a retained member. Do not select the
        // vacated position as the anchor when another retained member exists.
        int anchor = removed;
        if (subset_size > 1) {
            int anchor_pos = rng.randint(subset_size - 1);
            if (anchor_pos >= remove_pos) {
                ++anchor_pos;
            }
            anchor = seed[static_cast<std::size_t>(anchor_pos)];
        }

        int add = -1;
        if (inst.knn_k > 0) {
            const int start = rng.randint(inst.knn_k);
            for (int offset = 0; offset < inst.knn_k; ++offset) {
                const int candidate = inst.knn_at(anchor, (start + offset) % inst.knn_k);
                if (candidate >= 0 && candidate < inst.N
                    && free_pos[static_cast<std::size_t>(candidate)] >= 0) {
                    add = candidate;
                    break;
                }
            }
        }
        if (add < 0) {
            if (free_nodes.empty()) {
                throw std::logic_error("elite kick has no free replacement node");
            }
            add = free_nodes[static_cast<std::size_t>(
                rng.randint(static_cast<int>(free_nodes.size())))];
        }

        remove_free(add);
        in_set[static_cast<std::size_t>(add)] = 1U;
        seed[static_cast<std::size_t>(remove_pos)] = add;
    }

    // Keep the failure mode local and diagnosable if this routine is modified
    // later: no invalid seed may reach Tour::set_tour().
    std::fill(in_set.begin(), in_set.end(), 0U);
    for (const int node : seed) {
        if (node < 0 || node >= inst.N
            || in_set[static_cast<std::size_t>(node)] != 0U) {
            throw std::logic_error("elite kick failed to preserve subset uniqueness");
        }
        in_set[static_cast<std::size_t>(node)] = 1U;
    }
}

} // namespace aldous_tsp
