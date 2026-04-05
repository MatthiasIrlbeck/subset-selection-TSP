#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <string>
#include <limits>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <filesystem>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "config.hpp"

struct Rng {
    uint64_t s[4];
    void seed(uint64_t v){
        for(int i=0;i<4;i++){
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }
    static uint64_t rotl(uint64_t x,int k){ return (x << k) | (x >> (64 - k)); }
    uint64_t next_u64(){
        uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    double uniform(){ return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }
    int randint(int n){
        if(n <= 1) return 0;
        const uint64_t bound = (uint64_t)n;
        const uint64_t threshold = (uint64_t)(-bound) % bound;
        while(true){
            uint64_t x = next_u64();
            if(x >= threshold) return (int)(x % bound);
        }
    }
    void partial_shuffle(int* a,int n,int k){
        for(int i=0;i<k&&i<n-1;i++){
            int j = i + randint(n - i);
            std::swap(a[i], a[j]);
        }
    }
};

struct Instance {
    int N = 0;
    double side = 0.0;

    // Coordinates only; no full distance matrix.
    std::vector<double> x, y;

    // Exact KNN candidate sets.
    int knn_k = 0;
    std::vector<int> knn;
    std::vector<double> knn_d2;
    std::vector<double> knn_d;

    // Reverse exact KNN adjacency: for node v, all u such that v is in KNN(u).
    std::vector<int> rknn_begin, rknn_nodes;

    // Uniform grid for exact KNN construction (CSR-style cell buckets).
    double cell_size = 1.0;
    int gx = 0, gy = 0;
    std::vector<int> cell_x, cell_y;
    std::vector<int> cell_begin, cell_points;

    inline double dist2(int a,int b) const {
        double dx = x[a] - x[b];
        double dy = y[a] - y[b];
        return dx*dx + dy*dy;
    }
    inline double dist(int a,int b) const {
        double dx = x[a] - x[b];
        double dy = y[a] - y[b];
        return std::sqrt(dx*dx + dy*dy);
    }
    inline int knn_at(int node,int ki) const { return knn[(size_t)node * knn_k + ki]; }
    inline double knn_d2_at(int node,int ki) const { return knn_d2[(size_t)node * knn_k + ki]; }
    inline double knn_d_at(int node,int ki) const { return knn_d[(size_t)node * knn_k + ki]; }

    void generate(int n,Rng& rng){
        N = n;
        side = std::sqrt((double)N);
        x.resize(N);
        y.resize(N);
        for(int i=0;i<N;i++){
            x[i] = rng.uniform() * side;
            y[i] = rng.uniform() * side;
        }
    }

    void build_grid(double forced_cell_size = 0.0){
        // Density is 1 point per unit area in expectation, so a cell area in the
        // single-digit range keeps occupancy roughly constant as N grows.
        if(forced_cell_size > 0.0) cell_size = forced_cell_size;
        else {
            double target_occ = std::max(4.0, std::min(16.0, 0.35 * (double)std::max(knn_k, 1)));
            cell_size = std::sqrt(target_occ);
        }
        gx = std::max(1, (int)std::ceil(side / cell_size));
        gy = gx;
        int cells = gx * gy;
        cell_x.resize(N);
        cell_y.resize(N);
        std::vector<int> counts(cells, 0);
        for(int i=0;i<N;i++){
            int cx = std::min((int)(x[i] / cell_size), gx - 1);
            int cy = std::min((int)(y[i] / cell_size), gy - 1);
            cell_x[i] = cx;
            cell_y[i] = cy;
            counts[cy * gx + cx]++;
        }
        cell_begin.assign((size_t)cells + 1, 0);
        for(int c=0;c<cells;c++) cell_begin[c+1] = cell_begin[c] + counts[c];
        cell_points.assign(N, -1);
        std::vector<int> cur = cell_begin;
        for(int i=0;i<N;i++){
            int id = cell_y[i] * gx + cell_x[i];
            cell_points[cur[id]++] = i;
        }
    }

    // Small fixed candidate sets: linear insertion is cheapest here.
    inline void insert_best(int* best_idx,double* best_d2,int& cnt,int idx,double d2) const {
        int lim = knn_k;
        if(cnt < lim){
            int p = cnt;
            while(p > 0 && (d2 < best_d2[p-1] || (d2 == best_d2[p-1] && idx < best_idx[p-1]))){
                best_d2[p] = best_d2[p-1];
                best_idx[p] = best_idx[p-1];
                --p;
            }
            best_d2[p] = d2;
            best_idx[p] = idx;
            ++cnt;
            return;
        }
        if(d2 > best_d2[lim-1] || (d2 == best_d2[lim-1] && idx >= best_idx[lim-1])) return;
        int p = lim - 1;
        while(p > 0 && (d2 < best_d2[p-1] || (d2 == best_d2[p-1] && idx < best_idx[p-1]))){
            best_d2[p] = best_d2[p-1];
            best_idx[p] = best_idx[p-1];
            --p;
        }
        best_d2[p] = d2;
        best_idx[p] = idx;
    }

    template<class F>
    inline void visit_square(int cx,int cy,int r,F&& f) const {
        int xmin = std::max(0, cx - r), xmax = std::min(gx - 1, cx + r);
        int ymin = std::max(0, cy - r), ymax = std::min(gy - 1, cy + r);
        for(int iy=ymin; iy<=ymax; ++iy){
            int row = iy * gx;
            for(int ix=xmin; ix<=xmax; ++ix) f(row + ix);
        }
    }

    template<class F>
    inline void visit_ring(int cx,int cy,int r,F&& f) const {
        if(r == 0){
            f(cy * gx + cx);
            return;
        }
        int top = cy - r;
        int bottom = cy + r;
        int left = cx - r;
        int right = cx + r;
        int xmin = std::max(0, left), xmax = std::min(gx - 1, right);
        if(top >= 0){
            int row = top * gx;
            for(int ix=xmin; ix<=xmax; ++ix) f(row + ix);
        }
        if(bottom < gy && bottom != top){
            int row = bottom * gx;
            for(int ix=xmin; ix<=xmax; ++ix) f(row + ix);
        }
        int ymin = std::max(0, top + 1), ymax = std::min(gy - 1, bottom - 1);
        if(left >= 0){
            for(int iy=ymin; iy<=ymax; ++iy) f(iy * gx + left);
        }
        if(right < gx && right != left){
            for(int iy=ymin; iy<=ymax; ++iy) f(iy * gx + right);
        }
    }

    inline double cell_min_d2(double qx,double qy,int cell_id) const {
        int ix = cell_id % gx;
        int iy = cell_id / gx;
        double x0 = ix * cell_size;
        double x1 = std::min(side, x0 + cell_size);
        double y0 = iy * cell_size;
        double y1 = std::min(side, y0 + cell_size);
        double dx = 0.0;
        if(qx < x0) dx = x0 - qx;
        else if(qx > x1) dx = qx - x1;
        double dy = 0.0;
        if(qy < y0) dy = y0 - qy;
        else if(qy > y1) dy = qy - y1;
        return dx*dx + dy*dy;
    }

    inline double outside_bound_d2(int qi,int r) const {
        const double INF = std::numeric_limits<double>::infinity();
        int cx = cell_x[qi], cy = cell_y[qi];
        double qx = x[qi], qy = y[qi];
        double best_gap = INF;

        int xmin = cx - r;
        int xmax = cx + r;
        int ymin = cy - r;
        int ymax = cy + r;

        if(xmin > 0){
            double x_left = xmin * cell_size;
            best_gap = std::min(best_gap, qx - x_left);
        }
        if(xmax < gx - 1){
            double x_right = std::min(side, (xmax + 1) * cell_size);
            best_gap = std::min(best_gap, x_right - qx);
        }
        if(ymin > 0){
            double y_low = ymin * cell_size;
            best_gap = std::min(best_gap, qy - y_low);
        }
        if(ymax < gy - 1){
            double y_high = std::min(side, (ymax + 1) * cell_size);
            best_gap = std::min(best_gap, y_high - qy);
        }
        return best_gap * best_gap;
    }

    // Grid-accelerated exact KNN: expanding-ring scan over uniform grid cells,
    // with early termination once the k-th distance bounds any unseen cell.
    // Fallback to brute force if the ring doesn't cover enough points.
    void build_knn(int k,double forced_cell_size = 0.0){
        knn_k = std::min(k, N - 1);
        if(knn_k <= 0){
            knn.clear();
            knn_d2.clear();
            knn_d.clear();
            rknn_begin.clear();
            rknn_nodes.clear();
            return;
        }
        build_grid(forced_cell_size);
        knn.resize((size_t)N * knn_k);
        knn_d2.resize((size_t)N * knn_k);
        knn_d.resize((size_t)N * knn_k);

        std::vector<int> best_idx(knn_k);
        std::vector<double> best_d2(knn_k);

        int max_r = std::max(gx, gy);
        int guess_r = std::max(0, (int)std::ceil(std::sqrt((double)std::max(knn_k, 1) / PI) / std::max(cell_size, KNN_VERIFY_EPS)));
        guess_r = std::min(guess_r, max_r);
        for(int qi=0; qi<N; ++qi){
            int cnt = 0;
            std::fill(best_idx.begin(), best_idx.end(), -1);
            std::fill(best_d2.begin(), best_d2.end(), std::numeric_limits<double>::infinity());

            int cx = cell_x[qi], cy = cell_y[qi];
            double qx = x[qi], qy = y[qi];
            int r = 0;
            if(guess_r > 0){
                visit_square(cx, cy, guess_r, [&](int cell_id){
                    if(cnt >= knn_k){
                        double lb = cell_min_d2(qx, qy, cell_id);
                        if(lb > best_d2[knn_k - 1] + GEOM_BOUND_EPS) return;
                    }
                    for(int p=cell_begin[cell_id]; p<cell_begin[cell_id + 1]; ++p){
                        int j = cell_points[p];
                        if(j == qi) continue;
                        double d2 = dist2(qi, j);
                        insert_best(best_idx.data(), best_d2.data(), cnt, j, d2);
                    }
                });
                r = guess_r;
            } else {
                visit_ring(cx, cy, 0, [&](int cell_id){
                    for(int p=cell_begin[cell_id]; p<cell_begin[cell_id + 1]; ++p){
                        int j = cell_points[p];
                        if(j == qi) continue;
                        double d2 = dist2(qi, j);
                        insert_best(best_idx.data(), best_d2.data(), cnt, j, d2);
                    }
                });
                r = 1;
            }

            for(; r<=max_r; ++r){
                if(r > guess_r || guess_r == 0){
                    visit_ring(cx, cy, r, [&](int cell_id){
                        if(cnt >= knn_k){
                            double lb = cell_min_d2(qx, qy, cell_id);
                            if(lb > best_d2[knn_k - 1] + GEOM_BOUND_EPS) return;
                        }
                        for(int p=cell_begin[cell_id]; p<cell_begin[cell_id + 1]; ++p){
                            int j = cell_points[p];
                            if(j == qi) continue;
                            double d2 = dist2(qi, j);
                            insert_best(best_idx.data(), best_d2.data(), cnt, j, d2);
                        }
                    });
                }
                if(cnt >= knn_k){
                    double bound_d2 = outside_bound_d2(qi, r);
                    if(best_d2[knn_k - 1] <= bound_d2 + GEOM_BOUND_EPS) break;
                }
            }

            if(cnt < knn_k){
                for(int j=0; j<N; ++j){
                    if(j == qi) continue;
                    double d2 = dist2(qi, j);
                    insert_best(best_idx.data(), best_d2.data(), cnt, j, d2);
                }
            }

            size_t off = (size_t)qi * knn_k;
            for(int t=0; t<knn_k; ++t){
                knn[off + t] = best_idx[t];
                knn_d2[off + t] = best_d2[t];
                knn_d[off + t] = std::sqrt(best_d2[t]);
            }
        }

        rknn_begin.assign((size_t)N + 1, 0);
        for(int u=0; u<N; ++u){
            size_t off = (size_t)u * knn_k;
            for(int t=0; t<knn_k; ++t) ++rknn_begin[knn[off + t] + 1];
        }
        for(int i=0; i<N; ++i) rknn_begin[i + 1] += rknn_begin[i];
        rknn_nodes.assign((size_t)N * knn_k, -1);
        std::vector<int> rcur = rknn_begin;
        for(int u=0; u<N; ++u){
            size_t off = (size_t)u * knn_k;
            for(int t=0; t<knn_k; ++t){
                int v = knn[off + t];
                rknn_nodes[rcur[v]++] = u;
            }
        }
    }

    bool verify_knn(int checks,Rng& rng) const {
        if(knn_k <= 0 || N <= 1) return true;
        checks = std::min(checks, N);
        std::vector<int> idx(N-1), ref_idx(knn_k);
        std::vector<double> ref_d2(knn_k);
        for(int c=0; c<checks; ++c){
            int qi = rng.randint(N);
            int m = 0;
            for(int j=0; j<N; ++j) if(j != qi) idx[m++] = j;
            auto cmp = [&](int a,int b){
                double da = dist2(qi, a), db = dist2(qi, b);
                if(da != db) return da < db;
                return a < b;
            };
            if(knn_k < (int)idx.size()){
                std::nth_element(idx.begin(), idx.begin() + knn_k, idx.end(), cmp);
                std::sort(idx.begin(), idx.begin() + knn_k, cmp);
            } else {
                std::sort(idx.begin(), idx.end(), cmp);
            }
            for(int t=0; t<knn_k; ++t){
                ref_idx[t] = idx[t];
                ref_d2[t] = dist2(qi, idx[t]);
                int got = knn_at(qi, t);
                double gd2 = knn_d2_at(qi, t);
                if(got != ref_idx[t] || std::fabs(gd2 - ref_d2[t]) > KNN_VERIFY_EPS){
                    return false;
                }
            }
        }
        return true;
    }
};


static inline int bump_token(std::vector<int>& mark,int& token){
    ++token;
    if(token == std::numeric_limits<int>::max()){
        std::fill(mark.begin(), mark.end(), 0);
        token = 1;
    }
    return token;
}

static inline void dist_many_from(const Instance& inst,int src,const int* ids,int m,double* out){
    const double sx = inst.x[src];
    const double sy = inst.y[src];
#if defined(__AVX2__)
    if(m >= 8){
        const double* px = inst.x.data();
        const double* py = inst.y.data();
        __m256d vsx = _mm256_set1_pd(sx);
        __m256d vsy = _mm256_set1_pd(sy);
        int i = 0;
        for(; i + 4 <= m; i += 4){
            __m128i vid = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
            __m256d vx = _mm256_i32gather_pd(px, vid, 8);
            __m256d vy = _mm256_i32gather_pd(py, vid, 8);
            __m256d dx = _mm256_sub_pd(vsx, vx);
            __m256d dy = _mm256_sub_pd(vsy, vy);
        #if defined(__FMA__)
            __m256d d2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
        #else
            __m256d d2 = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
        #endif
            __m256d d = _mm256_sqrt_pd(d2);
            _mm256_storeu_pd(out + i, d);
        }
        for(; i < m; ++i){
            int v = ids[i];
            double dx = sx - px[v];
            double dy = sy - py[v];
            out[i] = std::sqrt(dx*dx + dy*dy);
        }
        return;
    }
#endif
    for(int i=0; i<m; ++i){
        int v = ids[i];
        double dx = sx - inst.x[v];
        double dy = sy - inst.y[v];
        out[i] = std::sqrt(dx*dx + dy*dy);
    }
}


