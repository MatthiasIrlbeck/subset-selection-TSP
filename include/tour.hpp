#pragma once

#include "problem.hpp"

static inline uint64_t mix_hash64(uint64_t z){
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t subset_hash_nodes(const std::vector<int>& nodes){
    uint64_t h1 = 0x243f6a8885a308d3ULL ^ ((uint64_t)nodes.size() * 0x9e3779b97f4a7c15ULL);
    uint64_t h2 = 0x13198a2e03707344ULL + ((uint64_t)nodes.size() * 0xbf58476d1ce4e5b9ULL);
    for(int v : nodes){
        uint64_t z = mix_hash64((uint64_t)v + 0x9e3779b97f4a7c15ULL);
        h1 += z;
        h2 ^= z + 0x9e3779b97f4a7c15ULL + (h2 << 6) + (h2 >> 2);
    }
    return h1 ^ (h2 * 0x94d049bb133111ebULL);
}

static inline uint64_t subset_set_hash_canon_copy(std::vector<int> nodes){
    std::sort(nodes.begin(), nodes.end());
    return subset_hash_nodes(nodes);
}

static inline uint64_t cycle_hash_canon_copy(const std::vector<int>& nodes){
    int n = (int)nodes.size();
    if(n <= 1) return subset_hash_nodes(nodes);
    int min_idx = 0;
    for(int i=1; i<n; ++i) if(nodes[i] < nodes[min_idx]) min_idx = i;
    auto fwd_at = [&](int t){ return nodes[(min_idx + t) % n]; };
    auto rev_at = [&](int t){ int idx = min_idx - t; while(idx < 0) idx += n; return nodes[idx]; };
    bool use_fwd = true;
    for(int t=1; t<n; ++t){
        int vf = fwd_at(t), vr = rev_at(t);
        if(vf < vr){ use_fwd = true; break; }
        if(vr < vf){ use_fwd = false; break; }
    }
    std::vector<int> canon;
    canon.reserve(n);
    for(int t=0; t<n; ++t) canon.push_back(use_fwd ? fwd_at(t) : rev_at(t));
    return subset_hash_nodes(canon);
}

enum class EliteHashMode { ORDERED, CYCLE, SET };

static inline uint64_t elite_hash_nodes(const std::vector<int>& nodes,EliteHashMode mode){
    switch(mode){
        case EliteHashMode::ORDERED: return subset_hash_nodes(nodes);
        case EliteHashMode::CYCLE: return cycle_hash_canon_copy(nodes);
        case EliteHashMode::SET: return subset_set_hash_canon_copy(nodes);
    }
    return subset_hash_nodes(nodes);
}

struct EliteEntry {
    double len = std::numeric_limits<double>::infinity();
    std::vector<int> nodes;
    uint64_t set_hash = 0;
};

struct ElitePool {
    int keep = 0;
    EliteHashMode mode = EliteHashMode::ORDERED;
    std::vector<EliteEntry> items;

    explicit ElitePool(int k=0,EliteHashMode m=EliteHashMode::ORDERED): keep(k), mode(m) {}

    void try_add(const std::vector<int>& nodes,double len){
        if(keep <= 0 || nodes.empty() || !std::isfinite(len)) return;
        uint64_t h = elite_hash_nodes(nodes, mode);
        // Approximate dedup: 64-bit hash + size check. Collisions are possible
        // but astronomically unlikely; full element-wise comparison not worthwhile.
        for(auto& e : items){
            if(e.set_hash == h && e.nodes.size() == nodes.size()){
                if(len + IMPROVEMENT_EPS < e.len){
                    e.len = len;
                    e.nodes = nodes;
                }
                std::sort(items.begin(), items.end(), [](const EliteEntry& a,const EliteEntry& b){ return a.len < b.len; });
                return;
            }
        }
        EliteEntry e;
        e.len = len;
        e.nodes = nodes;
        e.set_hash = h;
        auto it = std::lower_bound(items.begin(), items.end(), len,
            [](const EliteEntry& a,double v){ return a.len < v; });
        items.insert(it, std::move(e));
        if((int)items.size() > keep) items.pop_back();
    }

    std::vector<std::vector<int>> export_nodes() const {
        std::vector<std::vector<int>> out;
        out.reserve(items.size());
        for(const auto& e : items) out.push_back(e.nodes);
        return out;
    }
};


struct Tour {
    int N = 0;
    std::vector<int> nodes;
    int k = 0;
    std::vector<int> pos;
    std::vector<uint8_t> in_set;
    std::vector<double> edge_len;
    double length = 0.0;
    bool edge_valid = false;

    void init(int n){
        N = n;
        pos.assign(N, -1);
        in_set.assign(N, 0);
        nodes.clear();
        edge_len.clear();
        k = 0;
        length = 0.0;
        edge_valid = false;
    }
    void set_tour_only(const int* t,int size){
        nodes.assign(t, t + size);
        k = size;
        rebuild_index();
        edge_len.resize(k);
        length = 0.0;
        edge_valid = false;
    }
    void set_tour(const int* t,int size,const Instance& inst){
        set_tour_only(t, size);
        recompute_length(inst);
    }
    void rebuild_index(){
        std::fill(pos.begin(), pos.end(), -1);
        std::fill(in_set.begin(), in_set.end(), (uint8_t)0);
        for(int i=0;i<k;i++){
            pos[nodes[i]] = i;
            in_set[nodes[i]] = 1;
        }
    }
    void recompute_length(const Instance& inst){
        edge_len.resize(k);
        length = 0.0;
        if(k <= 0){
            edge_valid = true;
            return;
        }
        for(int i=0;i<k;i++){
            double d = inst.dist(nodes[i], nodes[(i+1)%k]);
            edge_len[i] = d;
            length += d;
        }
        edge_valid = true;
    }
    void ensure_edges(const Instance& inst){
        if(!edge_valid || (int)edge_len.size() != k) recompute_length(inst);
    }
    void remove_at(int ri){
        int rem = nodes[ri];
        pos[rem] = -1;
        in_set[rem] = 0;
        for(int i=ri;i<k-1;i++){
            nodes[i] = nodes[i+1];
            pos[nodes[i]] = i;
        }
        --k;
        nodes.resize(k);
        edge_len.resize(k);
        length = 0.0;
        edge_valid = false;
    }
    void insert_after(int ip,int node){
        int ins = ip + 1;
        nodes.push_back(0);
        ++k;
        for(int i=k-1;i>ins;i--){
            nodes[i] = nodes[i-1];
            pos[nodes[i]] = i;
        }
        nodes[ins] = node;
        pos[node] = ins;
        in_set[node] = 1;
        edge_len.resize(k);
        length = 0.0;
        edge_valid = false;
    }
    void swap_post_rem_nodes(int ri,int post_pred,int add){
        int rem = nodes[ri];
        int ins = post_pred + 1;
        pos[rem] = -1;
        in_set[rem] = 0;
        in_set[add] = 1;
        if(ins < ri){
            for(int i=ri;i>ins;i--){
                nodes[i] = nodes[i-1];
                pos[nodes[i]] = i;
            }
            nodes[ins] = add;
            pos[add] = ins;
        } else if(ins > ri){
            for(int i=ri;i<ins;i++){
                nodes[i] = nodes[i+1];
                pos[nodes[i]] = i;
            }
            nodes[ins] = add;
            pos[add] = ins;
        } else {
            nodes[ri] = add;
            pos[add] = ri;
        }
    }
    void move_node_post_rem_nodes(int ri,int post_pred){
        int node = nodes[ri];
        int ins = post_pred + 1;
        if(ins < ri){
            for(int i=ri;i>ins;i--){
                nodes[i] = nodes[i-1];
                pos[nodes[i]] = i;
            }
            nodes[ins] = node;
            pos[node] = ins;
        } else if(ins > ri){
            for(int i=ri;i<ins;i++){
                nodes[i] = nodes[i+1];
                pos[nodes[i]] = i;
            }
            nodes[ins] = node;
            pos[node] = ins;
        }
    }
    void reverse_segment_nodes(int lo,int hi){
        while(lo < hi){
            std::swap(nodes[lo], nodes[hi]);
            pos[nodes[lo]] = lo;
            pos[nodes[hi]] = hi;
            ++lo;
            --hi;
        }
        if(lo == hi) pos[nodes[lo]] = lo;
    }
    void affected_interval(int ri,int ins,int& start,int& count) const {
        if(k <= 0){
            start = 0;
            count = 0;
            return;
        }
        if(ins < ri){
            start = (ins == 0) ? (k - 1) : (ins - 1);
            count = ri - ins + 2;
        } else if(ins > ri){
            start = (ri == 0) ? (k - 1) : (ri - 1);
            count = ins - ri + 2;
        } else {
            start = (ri == 0) ? (k - 1) : (ri - 1);
            count = 2;
        }
        if(count > k) count = k;
    }
    void recompute_edge_interval(const Instance& inst,int start,int count){
        if(k <= 0){
            edge_len.clear();
            length = 0.0;
            edge_valid = true;
            return;
        }
        if((int)edge_len.size() != k) edge_len.resize(k);
        int idx = start;
        for(int t=0; t<count; ++t){
            int nxt = (idx + 1 == k) ? 0 : (idx + 1);
            edge_len[idx] = inst.dist(nodes[idx], nodes[nxt]);
            idx = nxt;
        }
        edge_valid = true;
    }
    void apply_swap_post_rem(int ri,int post_pred,int add,const Instance& inst,double delta){
        int ins = post_pred + 1;
        swap_post_rem_nodes(ri, post_pred, add);
        int start = 0, count = 0;
        affected_interval(ri, ins, start, count);
        recompute_edge_interval(inst, start, count);
        length += delta;
    }
    void apply_move_post_rem(int ri,int post_pred,const Instance& inst,double delta){
        int ins = post_pred + 1;
        move_node_post_rem_nodes(ri, post_pred);
        int start = 0, count = 0;
        affected_interval(ri, ins, start, count);
        recompute_edge_interval(inst, start, count);
        length += delta;
    }
    void apply_two_opt(int ii,int jj,const Instance& inst,double delta){
        int lo = ii + 1;
        int hi = jj;
        reverse_segment_nodes(lo, hi);
        if(hi - lo >= 1) std::reverse(edge_len.begin() + lo, edge_len.begin() + hi);
        edge_len[ii] = inst.dist(nodes[ii], nodes[(ii+1)%k]);
        edge_len[jj] = inst.dist(nodes[jj], nodes[(jj+1)%k]);
        length += delta;
        edge_valid = true;
    }
};

static inline int post_idx_from_cur(int cur_idx,int ri){ return (cur_idx < ri) ? cur_idx : (cur_idx - 1); }
static inline int next_live_idx(int idx,int ri,int n){ int s = idx + 1; if(s >= n) s = 0; if(s == ri){ ++s; if(s >= n) s = 0; } return s; }
static inline int prev_live_idx(int idx,int ri,int n){ int p = (idx == 0) ? (n - 1) : (idx - 1); if(p == ri) p = (p == 0) ? (n - 1) : (p - 1); return p; }
static inline double live_edge_len(const Tour& tour,const Instance& inst,int pred_idx,int succ_idx){
    int nxt = pred_idx + 1;
    if(nxt >= tour.k) nxt = 0;
    if(nxt == succ_idx) return tour.edge_len[pred_idx];
    return inst.dist(tour.nodes[pred_idx], tour.nodes[succ_idx]);
}


struct SmallNodeDistCache {
    // Tiny fixed-size cache; linear scan is fine at this scale.
    const Instance* inst = nullptr;
    int src = -1;
    int cnt = 0;
    static constexpr int CAP = 64;
    int node[CAP];
    double distv[CAP];

    SmallNodeDistCache(const Instance& in,int s): inst(&in), src(s), cnt(0) {}
    inline void seed(int v,double d){
        for(int i=0;i<cnt;i++) if(node[i] == v){ distv[i] = d; return; }
        if(cnt < CAP){ node[cnt] = v; distv[cnt] = d; ++cnt; }
    }
    inline double get(int v){
        if(v == src) return 0.0;
        for(int i=0;i<cnt;i++) if(node[i] == v) return distv[i];
        double d = inst->dist(src, v);
        if(cnt < CAP){ node[cnt] = v; distv[cnt] = d; ++cnt; }
        return d;
    }
};


