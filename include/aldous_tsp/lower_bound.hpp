#pragma once

#include <vector>

namespace aldous_tsp {

class Instance;

// Result of a Held-Karp (Lagrangian 1-tree) lower-bound computation.
struct HeldKarpBound {
    // Best Lagrangian lower bound found over the subgradient ascent. This is a
    // rigorous lower bound on the optimal tour length through the subset under
    // the instance metric (torus-aware when the instance is periodic).
    double bound = 0.0;
    // The plain minimum 1-tree bound (multipliers = 0), for reference.
    double one_tree = 0.0;
    // Subgradient iterations actually performed.
    int iterations = 0;
    // True if the ascent reached a degree-2 1-tree, i.e. a tour: then the bound
    // equals the exact optimum.
    bool closed_to_tour = false;
    // Non-negative gap upper_bound - bound (0 when no finite upper bound given).
    double gap_to_upper = 0.0;
    // False if the bound was not computed (e.g. subset too small or too large);
    // callers should treat the result as unavailable.
    bool computed = false;
};

// Held-Karp lower bound on the optimal tour visiting `subset` of `base`'s
// points, tightened by subgradient optimization of the 1-tree Lagrangian.
//
// The bound is the minimum 1-tree cost under modified edge weights
// c'(i,j) = d(i,j) + pi[i] + pi[j], minus 2*sum(pi); this is <= the optimal
// tour for every choice of pi (weak duality), and the ascent maximizes it.
// d(i,j) uses base.dist, so the bound is exact for the flat torus when
// base.periodic is set. `upper_bound` is a known tour length (>= optimal) used
// only for step sizing; pass the solver's tour length. `max_iters` bounds the
// ascent. The routine materializes an O(n^2) distance matrix for the subset, so
// it is intended for moderate subset sizes (a few thousand).
HeldKarpBound held_karp_bound(const Instance& base,
                              const std::vector<int>& subset,
                              double upper_bound,
                              int max_iters);

}  // namespace aldous_tsp
