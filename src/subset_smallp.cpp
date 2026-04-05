#include "subset_solver_internal.hpp"

static int smallp_pocket_seed_budget_simple(int N,int k){
    if(N <= 0 || k <= 0) return 0;
    double p = (double)k / (double)N;
    if(p > SMALLP_SIMPLE_SEED_MAX_P) return 0;
    if(p <= 0.030) return 5;
    if(p <= 0.050) return 4;
    return 3;
}

static int smallp_pocket_seed_budget_multiscale(int N,int k){
    if(N <= 0 || k <= 0) return 0;
    double p = (double)k / (double)N;
    if(p > SMALLP_MULTISCALE_SEED_MAX_P) return 0;
    int base = (p <= 0.030) ? 5 : 4;
    if(N >= 3000){
        base += 2;
    }
    if(N >= 5000){
        if(p <= 0.030) base += 2;
        else base += 1;
    }
    return std::min(base, SMALLP_POCKET_MAX_BUDGET);
}

static void build_cycle_seed_from_xy(const Instance& inst,double cx,double cy,int k,Rng& rng,std::vector<int>& out){
    int N = inst.N;
    if(k <= 0 || N <= 0 || k > N){ out.clear(); return; }
    std::vector<double> score(N);
    std::vector<int> idx(N);
    for(int i=0; i<N; ++i){
        double dx = inst.x[i] - cx;
        double dy = inst.y[i] - cy;
        score[i] = dx*dx + dy*dy;
        idx[i] = i;
    }
    auto cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] < score[b];
        return a < b;
    };
    if(k < N) std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);
    idx.resize(k);
    int si = (k > 1) ? rng.randint(k) : 0;
    nn_tour(idx.data(), k, inst, si, out);
}



static inline void centroid_from_knn_prefix(const Instance& inst,int center,int lim,double& cx,double& cy){
    cx = inst.x[center];
    cy = inst.y[center];
    int cnt = 1;
    lim = std::min(lim, inst.knn_k);
    for(int ki=0; ki<lim; ++ki){
        int v = inst.knn_at(center, ki);
        cx += inst.x[v];
        cy += inst.y[v];
        ++cnt;
    }
    cx /= (double)cnt;
    cy /= (double)cnt;
}

static inline void rank_dense_nodes_scale(const Instance& inst,int rank_m,std::vector<std::pair<double,int>>& dense_rank){
    rank_m = std::max(1, std::min(rank_m, inst.knn_k));
    dense_rank.clear();
    dense_rank.reserve(inst.N);
    for(int v=0; v<inst.N; ++v){
        double s = 0.0;
        for(int ki=0; ki<rank_m; ++ki) s += inst.knn_d_at(v, ki);
        double edge = inst.knn_d_at(v, rank_m - 1);
        double score = s / (double)rank_m + SMALLP_EDGE_WEIGHT * edge;
        dense_rank.push_back({score, v});
    }
    std::sort(dense_rank.begin(), dense_rank.end(), [](const auto& a,const auto& b){
        if(a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });
}

static std::vector<std::vector<int>> make_smallp_spatial_seed_pool_simple(const Instance& inst,int k,Rng& rng){
    std::vector<std::vector<int>> out;
    int budget = smallp_pocket_seed_budget_simple(inst.N, k);
    if(budget <= 0) return out;

    std::vector<uint64_t> seen;
    auto push_seed = [&](std::vector<int> s){
        if((int)out.size() >= budget) return;
        if((int)s.size() != k) return;
        uint64_t h = subset_set_hash_canon_copy(s);
        for(uint64_t x : seen) if(x == h) return;
        seen.push_back(h);
        out.push_back(std::move(s));
    };

    int rank_m = std::min(inst.knn_k, std::max(6, std::min(SMALLP_POCKET_LOCAL_CENTROID_KNN, std::max(1, k / 3))));
    std::vector<std::pair<double,int>> dense_rank;
    dense_rank.reserve(inst.N);
    for(int v=0; v<inst.N; ++v){
        double s = 0.0;
        for(int ki=0; ki<rank_m; ++ki) s += inst.knn_d_at(v, ki);
        double edge = inst.knn_d_at(v, rank_m - 1);
        double score = s / std::max(rank_m, 1) + SMALLP_EDGE_WEIGHT * edge;
        dense_rank.push_back({score, v});
    }
    std::sort(dense_rank.begin(), dense_rank.end(), [](const auto& a,const auto& b){
        if(a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    std::vector<uint8_t> covered(inst.N, 0);
    std::vector<int> centers;
    int cover_lim = std::min(inst.knn_k, std::max(10, std::min(SMALLP_POCKET_COVER_KNN, std::max(1, k / 2))));
    for(const auto& pr : dense_rank){
        int v = pr.second;
        if(covered[v]) continue;
        centers.push_back(v);
        covered[v] = 1;
        for(int ki=0; ki<cover_lim; ++ki) covered[inst.knn_at(v, ki)] = 1;
        if((int)centers.size() >= SMALLP_POCKET_CENTER_CANDS) break;
    }

    for(size_t ci=0; ci<centers.size() && (int)out.size() < budget; ++ci){
        int c = centers[ci];
        std::vector<int> seed;
        build_cycle_seed_from_xy(inst, inst.x[c], inst.y[c], k, rng, seed);
        push_seed(std::move(seed));
        if((int)out.size() >= budget) break;
        int lm = std::min(inst.knn_k, std::max(4, std::min(SMALLP_POCKET_LOCAL_CENTROID_KNN, std::max(1, k / 4))));
        double cx = inst.x[c], cy = inst.y[c];
        int cnt = 1;
        for(int ki=0; ki<lm; ++ki){
            int v = inst.knn_at(c, ki);
            cx += inst.x[v];
            cy += inst.y[v];
            ++cnt;
        }
        cx /= (double)cnt;
        cy /= (double)cnt;
        build_cycle_seed_from_xy(inst, cx, cy, k, rng, seed);
        push_seed(std::move(seed));
        if(ci >= 1 && (int)out.size() >= budget) break;
    }

    if((int)out.size() < budget){
        int cells = inst.gx * inst.gy;
        std::vector<std::pair<int,int>> cell_rank;
        cell_rank.reserve(cells);
        for(int c=0; c<cells; ++c){
            int cnt = inst.cell_begin[c+1] - inst.cell_begin[c];
            if(cnt > 0) cell_rank.push_back({-cnt, c});
        }
        std::sort(cell_rank.begin(), cell_rank.end());
        std::vector<uint8_t> blocked(cells, 0);
        for(const auto& pr : cell_rank){
            int c = pr.second;
            if(blocked[c]) continue;
            int beg = inst.cell_begin[c], end = inst.cell_begin[c+1];
            if(beg >= end) continue;
            double cx = 0.0, cy = 0.0;
            int cnt = 0;
            for(int p=beg; p<end; ++p){
                int v = inst.cell_points[p];
                cx += inst.x[v];
                cy += inst.y[v];
                ++cnt;
            }
            if(cnt <= 0) continue;
            cx /= (double)cnt;
            cy /= (double)cnt;
            std::vector<int> seed;
            build_cycle_seed_from_xy(inst, cx, cy, k, rng, seed);
            push_seed(std::move(seed));
            int ccx = c % inst.gx, ccy = c / inst.gx;
            for(int dy=-1; dy<=1; ++dy) for(int dx=-1; dx<=1; ++dx){
                int nx = ccx + dx, ny = ccy + dy;
                if(nx >= 0 && nx < inst.gx && ny >= 0 && ny < inst.gy) blocked[ny * inst.gx + nx] = 1;
            }
            if((int)out.size() >= budget) break;
            if((int)out.size() >= SMALLP_POCKET_CELL_CANDS) break;
        }
    }

    return out;
}

static std::vector<std::vector<int>> make_smallp_spatial_seed_pool_multiscale(const Instance& inst,int k,Rng& rng){
    std::vector<std::vector<int>> out;
    int budget = smallp_pocket_seed_budget_multiscale(inst.N, k);
    if(budget <= 0) return out;

    std::vector<uint64_t> seen;
    auto push_seed = [&](std::vector<int> s){
        if((int)out.size() >= budget) return;
        if((int)s.size() != k) return;
        uint64_t h = subset_set_hash_canon_copy(s);
        for(uint64_t x : seen) if(x == h) return;
        seen.push_back(h);
        out.push_back(std::move(s));
    };

    int rank_m1 = std::min(inst.knn_k, std::max(6, std::min(SMALLP_POCKET_LOCAL_CENTROID_KNN, std::max(1, k / 4))));
    int rank_m2 = std::min(inst.knn_k, std::max(rank_m1 + 4, std::min(std::max(rank_m1 + 4, 18), std::max(1, k / 2))));
    int rank_m3 = std::min(inst.knn_k, std::max(rank_m2 + 4, std::min(std::max(rank_m2 + 4, 28), std::max(1, k))));
    std::vector<int> rank_ms = {rank_m1, rank_m2, rank_m3};
    if(inst.N < 3000) rank_ms.resize(2);

    std::vector<uint8_t> covered(inst.N, 0);
    std::vector<int> centers;
    centers.reserve(SMALLP_POCKET_CENTER_CANDS + 8);
    std::vector<std::pair<double,int>> dense_rank;
    int cover_lim = std::min(inst.knn_k, std::max(10, std::min(SMALLP_POCKET_COVER_KNN, std::max(1, k / 2))));
    for(size_t si=0; si<rank_ms.size() && (int)centers.size() < SMALLP_POCKET_CENTER_CANDS; ++si){
        rank_dense_nodes_scale(inst, rank_ms[si], dense_rank);
        int want = SMALLP_POCKET_CENTER_CANDS / std::max(1, SMALLP_POCKET_MULTI_RANKS);
        if((int)si == 0) want += 2;
        int added = 0;
        for(const auto& pr : dense_rank){
            int v = pr.second;
            if(covered[v]) continue;
            centers.push_back(v);
            covered[v] = 1;
            for(int ki=0; ki<cover_lim; ++ki) covered[inst.knn_at(v, ki)] = 1;
            ++added;
            if((int)centers.size() >= SMALLP_POCKET_CENTER_CANDS || added >= want) break;
        }
    }

    auto emit_center_family = [&](int c){
        std::vector<int> seed;
        build_cycle_seed_from_xy(inst, inst.x[c], inst.y[c], k, rng, seed);
        push_seed(std::move(seed));
        if((int)out.size() >= budget) return;

        double cx = 0.0, cy = 0.0;
        int lm_small = std::min(inst.knn_k, std::max(4, std::min(10, std::max(1, k / 6))));
        centroid_from_knn_prefix(inst, c, lm_small, cx, cy);
        build_cycle_seed_from_xy(inst, cx, cy, k, rng, seed);
        push_seed(std::move(seed));
        if((int)out.size() >= budget) return;

        int lm_mid = std::min(inst.knn_k, std::max(lm_small + 3, std::min(20, std::max(1, k / 3))));
        centroid_from_knn_prefix(inst, c, lm_mid, cx, cy);
        build_cycle_seed_from_xy(inst, cx, cy, k, rng, seed);
        push_seed(std::move(seed));
        if((int)out.size() >= budget) return;

        if(inst.N >= 3000){
            int lm_big = std::min(inst.knn_k, std::max(lm_mid + 4, std::min(32, std::max(1, (int)std::round(std::sqrt((double)k) * 2.0)))));
            centroid_from_knn_prefix(inst, c, lm_big, cx, cy);
            build_cycle_seed_from_xy(inst, cx, cy, k, rng, seed);
            push_seed(std::move(seed));
        }
    };

    for(size_t ci=0; ci<centers.size() && (int)out.size() < budget; ++ci){
        emit_center_family(centers[ci]);
    }

    auto emit_dense_blocks = [&](int bw,int max_take){
        if((int)out.size() >= budget || bw <= 0) return;
        struct BlockCand { double score; double cx; double cy; int ix; int iy; };
        std::vector<BlockCand> blocks;
        blocks.reserve((size_t)std::max(1, (inst.gx - bw + 1) * (inst.gy - bw + 1)));
        for(int iy=0; iy + bw <= inst.gy; ++iy){
            for(int ix=0; ix + bw <= inst.gx; ++ix){
                int cnt = 0;
                double cx = 0.0, cy = 0.0;
                for(int dy=0; dy<bw; ++dy){
                    for(int dx=0; dx<bw; ++dx){
                        int cell = (iy + dy) * inst.gx + (ix + dx);
                        for(int p=inst.cell_begin[cell]; p<inst.cell_begin[cell + 1]; ++p){
                            int v = inst.cell_points[p];
                            cx += inst.x[v];
                            cy += inst.y[v];
                            ++cnt;
                        }
                    }
                }
                if(cnt <= 0) continue;
                double area = (double)(bw * bw);
                blocks.push_back({-(double)cnt / area, cx / (double)cnt, cy / (double)cnt, ix, iy});
            }
        }
        std::sort(blocks.begin(), blocks.end(), [](const BlockCand& a,const BlockCand& b){
            if(a.score != b.score) return a.score < b.score;
            if(a.iy != b.iy) return a.iy < b.iy;
            return a.ix < b.ix;
        });
        std::vector<uint8_t> blocked((size_t)inst.gx * inst.gy, 0);
        int used = 0;
        for(const auto& bc : blocks){
            bool skip = false;
            for(int dy=-1; dy<=bw; ++dy){
                for(int dx=-1; dx<=bw; ++dx){
                    int nx = bc.ix + dx, ny = bc.iy + dy;
                    if(nx >= 0 && nx < inst.gx && ny >= 0 && ny < inst.gy && blocked[(size_t)ny * inst.gx + nx]){
                        skip = true;
                        break;
                    }
                }
                if(skip) break;
            }
            if(skip) continue;
            std::vector<int> seed;
            build_cycle_seed_from_xy(inst, bc.cx, bc.cy, k, rng, seed);
            push_seed(std::move(seed));
            ++used;
            for(int dy=-1; dy<=bw; ++dy){
                for(int dx=-1; dx<=bw; ++dx){
                    int nx = bc.ix + dx, ny = bc.iy + dy;
                    if(nx >= 0 && nx < inst.gx && ny >= 0 && ny < inst.gy) blocked[(size_t)ny * inst.gx + nx] = 1;
                }
            }
            if((int)out.size() >= budget || used >= max_take) break;
        }
    };

    emit_dense_blocks(1, SMALLP_POCKET_CELL_CANDS);
    if(inst.N >= 3000) emit_dense_blocks(2, SMALLP_POCKET_BLOCK2_CANDS);
    if(inst.N >= 5000) emit_dense_blocks(3, SMALLP_POCKET_BLOCK3_CANDS);

    return out;
}

std::vector<std::vector<int>> make_smallp_spatial_seed_pool(const Instance& inst,int k,Rng& rng){
    if(inst.N <= 0 || k <= 0) return {};
    double p = (double)k / (double)inst.N;
    if(p <= SMALLP_MULTISCALE_SEED_MAX_P) return make_smallp_spatial_seed_pool_multiscale(inst, k, rng);
    if(p <= SMALLP_SIMPLE_SEED_MAX_P) return make_smallp_spatial_seed_pool_simple(inst, k, rng);
    return {};
}


void choose_ruin_positions_segment(const Tour& tour,const Instance& inst,Rng& rng,int ruin_size,std::vector<int>& pos_out);
void choose_ruin_positions_spread(const Tour& tour,const Instance& inst,Rng& rng,int ruin_size,std::vector<int>& pos_out);
void choose_ruin_positions(const Tour& tour,const Instance& inst,Rng& rng,int ruin_size,std::vector<int>& pos_out);
bool subset_pair_exchange_descent(Tour& tour,const Instance& inst,Rng& rng,int passes);
bool subset_ruin_recreate_lns(Tour& tour,const Instance& inst,Rng& rng,int rounds);
void polish_subset_candidate(Tour& cand,const Instance& inst,int strength);

static int smallp_geom_master_budget(int N,int k){
    if(N <= 0 || k <= 0) return 0;
    double p = (double)k / (double)N;
    if(p > SMALLP_MULTISCALE_SEED_MAX_P) return 0;
    int b = (p <= 0.030) ? 4 : 3;
    if(N >= 5000 && p <= 0.030) ++b;
    return std::min(b, SMALLP_GEOM_MAX_BUDGET);
}

static int smallp_geom_master_slack(int N,int k){
    int slack = std::max(18, std::min(96, k / 4));
    if(N >= 5000) slack += 8;
    return std::min(slack, std::max(0, N - k));
}

static void centroid_of_nodes(const Instance& inst,const std::vector<int>& nodes,double& cx,double& cy){
    if(nodes.empty()){ cx = cy = 0.0; return; }
    cx = 0.0; cy = 0.0;
    for(int v : nodes){
        cx += inst.x[v];
        cy += inst.y[v];
    }
    cx /= (double)nodes.size();
    cy /= (double)nodes.size();
}

static void build_cycle_seed_from_xy_subset(const Instance& inst,const std::vector<int>& base,double cx,double cy,int k,Rng& rng,std::vector<int>& out){
    int m = (int)base.size();
    if(k <= 0 || m < k){ out.clear(); return; }
    std::vector<double> score(m);
    std::vector<int> ord(m);
    for(int i=0; i<m; ++i){
        int v = base[i];
        double dx = inst.x[v] - cx;
        double dy = inst.y[v] - cy;
        score[i] = dx*dx + dy*dy;
        ord[i] = i;
    }
    auto cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] < score[b];
        return base[a] < base[b];
    };
    if(k < m) std::nth_element(ord.begin(), ord.begin() + k, ord.end(), cmp);
    std::vector<int> sub(k);
    for(int i=0; i<k; ++i) sub[i] = base[ord[i]];
    int si = (k > 1) ? rng.randint(k) : 0;
    nn_tour(sub.data(), k, inst, si, out);
}

static void build_coord_order(const Instance& inst,bool use_x,std::vector<int>& ord,std::vector<int>& pos){
    int N = inst.N;
    ord.resize(N);
    std::iota(ord.begin(), ord.end(), 0);
    if(use_x){
        std::sort(ord.begin(), ord.end(), [&](int a,int b){
            if(inst.x[a] != inst.x[b]) return inst.x[a] < inst.x[b];
            if(inst.y[a] != inst.y[b]) return inst.y[a] < inst.y[b];
            return a < b;
        });
    } else {
        std::sort(ord.begin(), ord.end(), [&](int a,int b){
            if(inst.y[a] != inst.y[b]) return inst.y[a] < inst.y[b];
            if(inst.x[a] != inst.x[b]) return inst.x[a] < inst.x[b];
            return a < b;
        });
    }
    pos.assign(N, 0);
    for(int i=0; i<N; ++i) pos[ord[i]] = i;
}

static void build_universe_window(const std::vector<int>& ord,const std::vector<int>& pos,int center,int m,std::vector<int>& out){
    int N = (int)ord.size();
    if(m >= N){ out = ord; return; }
    int cp = pos[center];
    int lo = cp - m/2;
    if(lo < 0) lo = 0;
    if(lo + m > N) lo = N - m;
    out.assign(ord.begin() + lo, ord.begin() + lo + m);
}

static void build_universe_nearest_xy(const Instance& inst,double cx,double cy,int m,std::vector<int>& out){
    int N = inst.N;
    if(m >= N){
        out.resize(N);
        std::iota(out.begin(), out.end(), 0);
        return;
    }
    std::vector<double> score(N);
    std::vector<int> idx(N);
    for(int i=0; i<N; ++i){
        double dx = inst.x[i] - cx;
        double dy = inst.y[i] - cy;
        score[i] = dx*dx + dy*dy;
        idx[i] = i;
    }
    auto cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] < score[b];
        return a < b;
    };
    std::nth_element(idx.begin(), idx.begin() + m, idx.end(), cmp);
    idx.resize(m);
    out.swap(idx);
}


static void trim_or_augment_universe_by_centroid(const Instance& inst,const std::vector<int>& base,double cx,double cy,int target_m,std::vector<int>& out){
    int N = inst.N;
    if(target_m <= 0){ out.clear(); return; }
    std::vector<int> cur = base;
    std::sort(cur.begin(), cur.end());
    cur.erase(std::unique(cur.begin(), cur.end()), cur.end());
    if((int)cur.size() > target_m){
        std::vector<double> score(cur.size());
        std::vector<int> ord(cur.size());
        for(int i=0; i<(int)cur.size(); ++i){
            double dx = inst.x[cur[i]] - cx;
            double dy = inst.y[cur[i]] - cy;
            score[i] = dx*dx + dy*dy;
            ord[i] = i;
        }
        auto cmp = [&](int a,int b){
            if(score[a] != score[b]) return score[a] < score[b];
            return cur[a] < cur[b];
        };
        std::nth_element(ord.begin(), ord.begin() + target_m, ord.end(), cmp);
        out.resize(target_m);
        for(int i=0; i<target_m; ++i) out[i] = cur[ord[i]];
        return;
    }
    out = cur;
    if((int)out.size() >= target_m) return;
    std::vector<uint8_t> in_s(N, 0);
    for(int v : out) in_s[v] = 1;
    std::vector<double> score(N);
    std::vector<int> idx(N);
    for(int i=0; i<N; ++i){
        double dx = inst.x[i] - cx;
        double dy = inst.y[i] - cy;
        score[i] = dx*dx + dy*dy;
        idx[i] = i;
    }
    auto cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] < score[b];
        return a < b;
    };
    std::sort(idx.begin(), idx.end(), cmp);
    for(int v : idx){
        if(in_s[v]) continue;
        out.push_back(v);
        in_s[v] = 1;
        if((int)out.size() >= target_m) break;
    }
}

static void build_universe_box_xy(const Instance& inst,const std::vector<int>& xord,const std::vector<int>& xpos,
                                  const std::vector<int>& yord,const std::vector<int>& ypos,
                                  int center,int wx,int wy,int target_m,std::vector<int>& out){
    int N = inst.N;
    wx = std::min(std::max(wx, target_m), N);
    wy = std::min(std::max(wy, target_m), N);
    std::vector<uint8_t> in_x(N, 0);
    int cp = xpos[center];
    int lo = cp - wx/2;
    if(lo < 0) lo = 0;
    if(lo + wx > N) lo = N - wx;
    for(int i=lo; i<lo + wx; ++i) in_x[xord[i]] = 1;

    cp = ypos[center];
    lo = cp - wy/2;
    if(lo < 0) lo = 0;
    if(lo + wy > N) lo = N - wy;
    std::vector<int> inter;
    inter.reserve(std::min(wx, wy));
    for(int i=lo; i<lo + wy; ++i){
        int v = yord[i];
        if(in_x[v]) inter.push_back(v);
    }
    double cx = 0.0, cy = 0.0;
    if(!inter.empty()) centroid_of_nodes(inst, inter, cx, cy);
    else { cx = inst.x[center]; cy = inst.y[center]; }
    trim_or_augment_universe_by_centroid(inst, inter.empty() ? std::vector<int>{center} : inter, cx, cy, target_m, out);
}

static void build_universe_cell_block(const Instance& inst,int center,int bw,int target_m,std::vector<int>& out){
    int cx = inst.cell_x[center], cy = inst.cell_y[center];
    std::vector<int> nodes;
    nodes.reserve(target_m * 2);
    for(int dy=-bw; dy<=bw; ++dy){
        int yy = cy + dy;
        if(yy < 0 || yy >= inst.gy) continue;
        for(int dx=-bw; dx<=bw; ++dx){
            int xx = cx + dx;
            if(xx < 0 || xx >= inst.gx) continue;
            int cid = yy * inst.gx + xx;
            for(int p=inst.cell_begin[cid]; p<inst.cell_begin[cid + 1]; ++p) nodes.push_back(inst.cell_points[p]);
        }
    }
    double ux = 0.0, uy = 0.0;
    if(!nodes.empty()) centroid_of_nodes(inst, nodes, ux, uy);
    else { ux = inst.x[center]; uy = inst.y[center]; }
    trim_or_augment_universe_by_centroid(inst, nodes.empty() ? std::vector<int>{center} : nodes, ux, uy, target_m, out);
}

static int comb_limit_choose(int n,int r,int cap){
    if(r < 0 || r > n) return 0;
    r = std::min(r, n - r);
    long double v = 1.0;
    for(int i=1; i<=r; ++i){
        v = v * (long double)(n - r + i) / (long double)i;
        if(v > (long double)cap) return cap + 1;
    }
    return (int)std::llround(v);
}

struct SmallpRegionSubsetCand {
    double score = std::numeric_limits<double>::infinity();
    std::vector<int> nodes;
};

static void keep_best_region_cand(std::vector<SmallpRegionSubsetCand>& best,double score,const std::vector<int>& nodes,int cap){
    SmallpRegionSubsetCand c;
    c.score = score;
    c.nodes = nodes;
    auto it = std::lower_bound(best.begin(), best.end(), score, [](const SmallpRegionSubsetCand& a,double s){ return a.score < s; });
    best.insert(it, std::move(c));
    if((int)best.size() > cap) best.pop_back();
}

static void build_cycle_from_set_fast(const Instance& inst,const std::vector<int>& set_nodes,Rng& rng,std::vector<int>& out){
    int k = (int)set_nodes.size();
    if(k <= 3){ out = set_nodes; return; }
    if(k <= 48) farthest_ins(set_nodes.data(), k, inst, out);
    else {
        int si = (k > 1) ? rng.randint(k) : 0;
        nn_tour(set_nodes.data(), k, inst, si, out);
    }
}

static bool subset_swap_descent_allowed(Tour& tour,const Instance& inst,const std::vector<int>& universe,int passes=SMALLP_GEOM_RESTRICTED_SWAP_PASSES){
    if(tour.k < 3 || universe.size() <= (size_t)tour.k) return false;
    bool any = false;
    std::vector<int> pred_buf;
    pred_buf.reserve(SUBSET_MAX_EDGE_CAND);

    for(int pass=0; pass<passes; ++pass){
        std::vector<int> add_pool;
        add_pool.reserve(universe.size());
        for(int v : universe) if(!tour.in_set[v]) add_pool.push_back(v);
        if(add_pool.empty()) break;

        double global_best = -IMPROVEMENT_EPS;
        int best_ri = -1, best_add = -1, best_post_pred = 0;
        for(int ri=0; ri<tour.k; ++ri){
            int nt = tour.k;
            int prev_idx = (ri == 0) ? (nt - 1) : (ri - 1);
            int next_idx = (ri + 1 == nt) ? 0 : (ri + 1);
            int prv = tour.nodes[prev_idx], nxt = tour.nodes[next_idx];
            double gap_len = inst.dist(prv, nxt);
            double rs = tour.edge_len[prev_idx] + tour.edge_len[ri] - gap_len;
            bool force_full = (nt <= PR_FULL_EDGE_SCAN_K || (int)add_pool.size() <= 48);

            for(int add : add_pool){
                collect_pr_pred_candidates(tour, inst, ri, add, pred_buf, force_full);
                SmallNodeDistCache dc(inst, add);
                double best_ins = std::numeric_limits<double>::infinity();
                int best_post = (ri > 0) ? (ri - 1) : (nt - 2);
                for(int pred_idx : pred_buf){
                    int succ_idx = next_live_idx(pred_idx, ri, nt);
                    int a = tour.nodes[pred_idx];
                    int b = tour.nodes[succ_idx];
                    double base = (pred_idx == prev_idx) ? gap_len : tour.edge_len[pred_idx];
                    double c = dc.get(a) + dc.get(b) - base;
                    if(c < best_ins){
                        best_ins = c;
                        best_post = post_idx_from_cur(pred_idx, ri);
                    }
                }
                double delta = best_ins - rs;
                if(delta < global_best){
                    global_best = delta;
                    best_ri = ri;
                    best_add = add;
                    best_post_pred = best_post;
                }
            }
        }

        if(best_ri < 0) break;
        tour.apply_swap_post_rem(best_ri, best_post_pred, best_add, inst, global_best);
        two_opt_nn(tour, inst, 220, -1);
        or_opt_1(tour, inst, 3, TSP_OROPT_NN_CAND + 4, TSP_OROPT_LOCAL_WINDOW + 1);
        two_opt_nn(tour, inst, 100, -1);
        any = true;
    }
    return any;
}

static bool subset_ruin_recreate_allowed(Tour& tour,const Instance& inst,const std::vector<int>& universe,Rng& rng,int rounds=SMALLP_GEOM_RESTRICTED_LNS_ROUNDS){
    if(tour.k < 6 || universe.size() <= (size_t)tour.k) return false;
    bool any = false;
    int k = tour.k;
    std::vector<uint8_t> banned(inst.N, 1);
    for(int v : universe) banned[v] = 0;

    for(int round=0; round<rounds; ++round){
        int ruin_size = std::min(k - 3, 3 + round);
        if(ruin_size < 2) continue;
        std::vector<int> removed_pos;
        if((round % 3) == 0) choose_ruin_positions(tour, inst, rng, ruin_size, removed_pos);
        else if((round % 3) == 1) choose_ruin_positions_segment(tour, inst, rng, ruin_size, removed_pos);
        else choose_ruin_positions_spread(tour, inst, rng, ruin_size, removed_pos);
        if((int)removed_pos.size() != ruin_size) continue;

        std::vector<int> remain;
        remain.reserve(k);
        std::vector<uint8_t> remain_in(inst.N, 0);
        int rp = 0;
        for(int i=0; i<k; ++i){
            if(rp < ruin_size && removed_pos[rp] == i){ ++rp; }
            else {
                remain.push_back(tour.nodes[i]);
                remain_in[tour.nodes[i]] = 1;
            }
        }

        std::vector<int> pool;
        pool.reserve(universe.size());
        for(int v : universe) if(!remain_in[v]) pool.push_back(v);
        if((int)pool.size() < ruin_size) continue;
        if(!regret_repair_cycle(remain, inst, k, pool, banned)) continue;

        Tour cand;
        cand.init(inst.N);
        cand.set_tour_only(remain.data(), k);
        polish_fixed_subset_tour(cand, inst, ruin_size >= 5 ? 2 : 1);
        subset_swap_descent_allowed(cand, inst, universe, 1);
        polish_fixed_subset_tour(cand, inst, 1);
        if(cand.length < tour.length - IMPROVEMENT_EPS){
            tour = std::move(cand);
            any = true;
        }
    }
    return any;
}

static void rank_universe_by_center_score(const Instance& inst,const std::vector<int>& universe,double cx,double cy,
                                               std::vector<int>& ord,std::vector<double>& score){
    int m = (int)universe.size();
    ord.resize(m);
    score.resize(m);
    for(int i=0; i<m; ++i){
        int v = universe[i];
        double dx = inst.x[v] - cx;
        double dy = inst.y[v] - cy;
        score[i] = dx*dx + dy*dy;
        ord[i] = i;
    }
    std::sort(ord.begin(), ord.end(), [&](int a,int b){
        if(score[a] != score[b]) return score[a] < score[b];
        return universe[a] < universe[b];
    });
}

static void build_smallp_region_fringe_candidates(const Instance& inst,const std::vector<int>& universe,double cx,double cy,int k,
                                                  std::vector<SmallpRegionSubsetCand>& out){
    out.clear();
    int m = (int)universe.size();
    if(m < k) return;
    std::vector<int> ord;
    std::vector<double> score;
    rank_universe_by_center_score(inst, universe, cx, cy, ord, score);

    std::vector<int> nearest(k);
    double base_score = 0.0;
    for(int i=0; i<k; ++i){ nearest[i] = universe[ord[i]]; base_score += score[ord[i]]; }
    keep_best_region_cand(out, base_score, nearest, SMALLP_GEOM_TOP_SURVIVORS * 2);

    int pick_base = std::min(SMALLP_GEOM_FRINGE_PICK, std::max(4, std::min(k, m - k)));
    for(int tweak=0; tweak<3; ++tweak){
        int pick = pick_base + tweak - 1;
        if(pick < 4) continue;
        if(pick >= k) continue;
        int core_take = k - pick;
        int free_count = std::min(m - core_take, pick + SMALLP_GEOM_FRINGE_EXTRA);
        if(free_count < pick) continue;
        if(comb_limit_choose(free_count, pick, SMALLP_GEOM_ENUM_CAP) > SMALLP_GEOM_ENUM_CAP) continue;

        std::vector<int> core(core_take), fringe(free_count);
        double core_score = 0.0;
        for(int i=0; i<core_take; ++i){ core[i] = universe[ord[i]]; core_score += score[ord[i]]; }
        for(int i=0; i<free_count; ++i) fringe[i] = ord[core_take + i];

        std::vector<int> chosen;
        chosen.reserve(pick);
        auto rec = [&](auto&& self,int at,int need,double sumscore)->void{
            if(need == 0){
                std::vector<int> cand = core;
                cand.reserve(k);
                for(int idx : chosen) cand.push_back(universe[fringe[idx]]);
                keep_best_region_cand(out, core_score + sumscore, cand, SMALLP_GEOM_TOP_SURVIVORS * 2);
                return;
            }
            int remain = free_count - at;
            if(remain < need) return;
            for(int i=at; i<=free_count - need; ++i){
                chosen.push_back(i);
                self(self, i + 1, need - 1, sumscore + score[fringe[i]]);
                chosen.pop_back();
            }
        };
        rec(rec, 0, pick, 0.0);
    }
}

static bool build_smallp_geometric_master_candidate(const Instance& inst,const std::vector<int>& universe,
                                                    double cx,double cy,int k,Rng& rng,
                                                    std::vector<int>& out_nodes,double& out_len){
    if((int)universe.size() < k) return false;
    ElitePool local_best(SMALLP_GEOM_TOP_SURVIVORS, EliteHashMode::SET);

    std::vector<int> init;
    build_cycle_seed_from_xy_subset(inst, universe, cx, cy, k, rng, init);
    if((int)init.size() == k){
        Tour t;
        t.init(inst.N);
        t.set_tour_only(init.data(), k);
        polish_fixed_subset_tour(t, inst, 1);
        subset_swap_descent_allowed(t, inst, universe, 1);
        subset_ruin_recreate_allowed(t, inst, universe, rng, 1);
        polish_fixed_subset_tour(t, inst, 1);
        local_best.try_add(t.nodes, t.length);
    }

    std::vector<SmallpRegionSubsetCand> fringe_best;
    build_smallp_region_fringe_candidates(inst, universe, cx, cy, k, fringe_best);
    for(int ci=0; ci<(int)fringe_best.size(); ++ci){
        std::vector<int> cyc;
        build_cycle_from_set_fast(inst, fringe_best[ci].nodes, rng, cyc);
        Tour t;
        t.init(inst.N);
        t.set_tour_only(cyc.data(), k);
        polish_fixed_subset_tour(t, inst, 1);
        subset_swap_descent_allowed(t, inst, universe, ci == 0 ? 2 : 1);
        if(ci < 2) subset_ruin_recreate_allowed(t, inst, universe, rng, 1);
        polish_fixed_subset_tour(t, inst, 1);
        local_best.try_add(t.nodes, t.length);
    }

    if(local_best.items.empty()) return false;
    Tour fin;
    fin.init(inst.N);
    fin.set_tour_only(local_best.items[0].nodes.data(), k);
    subset_swap_descent_allowed(fin, inst, universe, 1);
    polish_fixed_subset_tour(fin, inst, 1);
    out_nodes = fin.nodes;
    out_len = fin.length;
    return true;
}

std::vector<std::vector<int>> make_smallp_geometric_master_pool(const Instance& inst,int k,Rng& rng){
    std::vector<std::vector<int>> out;
    int budget = smallp_geom_master_budget(inst.N, k);
    if(budget <= 0) return out;
    int m = std::min(inst.N, k + smallp_geom_master_slack(inst.N, k));
    if(m <= k) return out;

    std::vector<std::pair<double,int>> dense_rank;
    int rank_m = std::min(inst.knn_k, std::max(8, std::min(24, std::max(1, k / 4))));
    rank_dense_nodes_scale(inst, rank_m, dense_rank);

    std::vector<uint8_t> covered(inst.N, 0);
    std::vector<int> centers;
    int cover_lim = std::min(inst.knn_k, std::max(12, std::min(24, std::max(1, k / 2))));
    for(const auto& pr : dense_rank){
        int v = pr.second;
        if(covered[v]) continue;
        centers.push_back(v);
        covered[v] = 1;
        for(int ki=0; ki<cover_lim; ++ki) covered[inst.knn_at(v, ki)] = 1;
        if((int)centers.size() >= SMALLP_GEOM_CENTER_CANDS) break;
    }

    std::vector<int> xord, xpos, yord, ypos;
    build_coord_order(inst, true, xord, xpos);
    build_coord_order(inst, false, yord, ypos);

    ElitePool best_pool(budget, EliteHashMode::SET);
    std::vector<uint64_t> seen_universe;
    auto try_universe = [&](const std::vector<int>& universe,double ux,double uy){
        if((int)universe.size() < k) return;
        uint64_t uh = subset_set_hash_canon_copy(universe);
        for(uint64_t x : seen_universe) if(x == uh) return;
        seen_universe.push_back(uh);
        std::vector<int> nodes;
        double len = std::numeric_limits<double>::infinity();
        if(build_smallp_geometric_master_candidate(inst, universe, ux, uy, k, rng, nodes, len)) best_pool.try_add(nodes, len);
    };

    int evals = 0;
    for(int c : centers){
        if(evals >= SMALLP_GEOM_MAX_EVAL) break;
        std::vector<int> universe;

        build_universe_nearest_xy(inst, inst.x[c], inst.y[c], m, universe);
        try_universe(universe, inst.x[c], inst.y[c]);
        if(++evals >= SMALLP_GEOM_MAX_EVAL) break;

        double cx = 0.0, cy = 0.0;
        int lm = std::min(inst.knn_k, std::max(6, std::min(18, std::max(1, k / 5))));
        centroid_from_knn_prefix(inst, c, lm, cx, cy);
        build_universe_nearest_xy(inst, cx, cy, m, universe);
        try_universe(universe, cx, cy);
        if(++evals >= SMALLP_GEOM_MAX_EVAL) break;

        build_universe_window(xord, xpos, c, m, universe);
        centroid_of_nodes(inst, universe, cx, cy);
        try_universe(universe, cx, cy);
        if(++evals >= SMALLP_GEOM_MAX_EVAL) break;

        build_universe_window(yord, ypos, c, m, universe);
        centroid_of_nodes(inst, universe, cx, cy);
        try_universe(universe, cx, cy);
        if(++evals >= SMALLP_GEOM_MAX_EVAL) break;

        for(int sh=0; sh<SMALLP_GEOM_BOX_SHAPES && evals < SMALLP_GEOM_MAX_EVAL; ++sh){
            int wx = std::min(inst.N, m + sh * std::max(8, smallp_geom_master_slack(inst.N, k) / 3));
            int wy = std::min(inst.N, m + (SMALLP_GEOM_BOX_SHAPES - 1 - sh) * std::max(8, smallp_geom_master_slack(inst.N, k) / 3));
            build_universe_box_xy(inst, xord, xpos, yord, ypos, c, wx, wy, m, universe);
            centroid_of_nodes(inst, universe, cx, cy);
            try_universe(universe, cx, cy);
            ++evals;
        }
        if(evals >= SMALLP_GEOM_MAX_EVAL) break;

        build_universe_cell_block(inst, c, 1, m, universe);
        centroid_of_nodes(inst, universe, cx, cy);
        try_universe(universe, cx, cy);
        if(++evals >= SMALLP_GEOM_MAX_EVAL) break;
        if(inst.N >= 3000){
            build_universe_cell_block(inst, c, 2, m, universe);
            centroid_of_nodes(inst, universe, cx, cy);
            try_universe(universe, cx, cy);
            if(++evals >= SMALLP_GEOM_MAX_EVAL) break;
        }
        if(inst.N >= 5000){
            build_universe_cell_block(inst, c, 3, m, universe);
            centroid_of_nodes(inst, universe, cx, cy);
            try_universe(universe, cx, cy);
            if(++evals >= SMALLP_GEOM_MAX_EVAL) break;
        }
    }

    return best_pool.export_nodes();
}
