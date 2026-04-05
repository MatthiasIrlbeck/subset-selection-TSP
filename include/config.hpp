#pragma once

// Numerical tolerances and mathematical constants.
inline constexpr double PI = 3.14159265358979323846;
inline constexpr double IMPROVEMENT_EPS = 1e-10;
inline constexpr double KNN_VERIFY_EPS = 1e-12;
inline constexpr double GEOM_BOUND_EPS = 1e-15;
inline constexpr double TEMP_FLOOR_EPS = 1e-15;

inline constexpr int SUBSET_ADD_SCAN_REM = 18;
inline constexpr int SUBSET_ADD_SCAN_GAP = 18;
inline constexpr int SUBSET_TOP_ADD = 10;
inline constexpr int SUBSET_EDGE_ADD_NN = 24;
inline constexpr int SUBSET_LOCAL_EDGE_WINDOW = 4;
inline constexpr int SUBSET_GLOBAL_PROBES = 6;
inline constexpr int SUBSET_MAX_ADD_CAND = 96;
inline constexpr int SUBSET_MAX_EDGE_CAND = 384;
inline constexpr int SUBSET_MAX_ENDPOINTS = 768;

inline constexpr int TSP_INIT_TWOOPT_K = -1;
inline constexpr int TSP_ILS_TWOOPT_K = 24;
inline constexpr int TSP_FINAL_TWOOPT_K = -1;  // full inst.knn_k
inline constexpr int TSP_OROPT_NN_CAND = 16;
inline constexpr int TSP_OROPT_LOCAL_WINDOW = 2;
inline constexpr int TSP_TOP_RESTART_CAND = 8;

inline constexpr int CROSSP_ELITE_KEEP = 6;
inline constexpr int CROSSP_WARM_LIMIT = 16;
inline constexpr int CROSSP_RECOMB_PAIR_LIMIT = 8;
inline constexpr int CROSSP_GUIDE_RELINK_LIMIT = 5;
inline constexpr int SUBSET_LOCAL_ELITE_KEEP = 5;
inline constexpr int SUBSET_FINAL_ELITE_REPOLISH = 2;
inline constexpr int SUBSET_GUIDE_RELINK_LOCAL = 3;
inline constexpr int SUBSET_GUIDE_RELINK_GUIDE = 3;
inline constexpr int SUBSET_LOCAL_RELINK_TOP = 3;
inline constexpr int RR_PAIR_TOP_REMOVE = 12;
inline constexpr int RR_POOL_BASE = 64;
inline constexpr int RR_POOL_PER_RUIN = 14;
inline constexpr int RR_MAX_POOL = 128;
inline constexpr int RR_ROUNDS_BASE = 7;
inline constexpr int PR_TOP_REMOVE = 12;
inline constexpr int PR_TOP_ADD = 18;
inline constexpr int PR_FULL_EDGE_SCAN_K = 320;
inline constexpr int PR_EDGE_ADD_NN = 40;
inline constexpr int PR_LOCAL_EDGE_WINDOW = 8;
inline constexpr int PR_POLISH_EVERY = 2;
inline constexpr int EXACT_SMALL_TSP_K = 16;
inline constexpr int CORE_RECOMB_KNN = 18;
inline constexpr int CORE_RECOMB_UNION_KEEP = 96;
inline constexpr int TWO2_TOP_REMOVE = 12;
inline constexpr int TWO2_TOP_POOL = 22;
inline constexpr int TWO2_POLISH_KEEP = 5;
inline constexpr int TWO2_GUIDE_PAIR_TRIES = 6;
inline constexpr double SMALLP_SIMPLE_SEED_MAX_P = 0.070;
inline constexpr double SMALLP_MULTISCALE_SEED_MAX_P = 0.050;
inline constexpr double REGIME_LOW_MAX_P = SMALLP_MULTISCALE_SEED_MAX_P;
inline constexpr int SMALLP_POCKET_CENTER_CANDS = 18;
inline constexpr int SMALLP_POCKET_CELL_CANDS = 8;
inline constexpr int SMALLP_POCKET_COVER_KNN = 18;
inline constexpr int SMALLP_POCKET_LOCAL_CENTROID_KNN = 12;
inline constexpr int SMALLP_POCKET_MULTI_RANKS = 3;
inline constexpr int SMALLP_POCKET_BLOCK2_CANDS = 6;
inline constexpr int SMALLP_POCKET_BLOCK3_CANDS = 4;
inline constexpr int SMALLP_POCKET_MAX_BUDGET = 10;
inline constexpr int SMALLP_GEOM_CENTER_CANDS = 14;
inline constexpr int SMALLP_GEOM_MAX_BUDGET = 7;
inline constexpr int SMALLP_GEOM_MAX_EVAL = 16;
inline constexpr int SMALLP_GEOM_RESTRICTED_SWAP_PASSES = 2;
inline constexpr int SMALLP_GEOM_RESTRICTED_LNS_ROUNDS = 3;
inline constexpr int SMALLP_GEOM_BLOCK2_CANDS = 5;
inline constexpr int SMALLP_GEOM_BLOCK3_CANDS = 3;
inline constexpr int SMALLP_GEOM_BOX_SHAPES = 3;
inline constexpr int SMALLP_GEOM_FRINGE_PICK = 8;
inline constexpr int SMALLP_GEOM_FRINGE_EXTRA = 10;
inline constexpr int SMALLP_GEOM_ENUM_CAP = 3000;
inline constexpr int SMALLP_GEOM_TOP_SURVIVORS = 4;

inline constexpr double REGIME_HIGH_MIN_P = 0.500;
inline constexpr int HIGHP_DELETE_SEED_BUDGET = 14;
inline constexpr int HIGHP_DELETE_PARENT_USE = 4;
inline constexpr int HIGHP_DELETE_REF_HOPS = 8;
inline constexpr int HIGHP_DELETE_REF_CAND = 80;
inline constexpr int HIGHP_DELETE_KNN_CAND = 12;
inline constexpr int HIGHP_DELETE_PASSES = 5;
inline constexpr int HIGHP_DELETE_SEG_SEEDS = 3;

inline constexpr double SA_T0 = 1.4;
inline constexpr double SA_T1 = 0.00005;
inline constexpr double SA_SWAP_PROB = 0.72;
inline constexpr int SA_RECOMPUTE_INTERVAL = 5000;
inline constexpr int SA_DESCENT_INTERVAL = 10000;
inline constexpr int SA_STAGNATION_LIMIT = 22000;
inline constexpr double REGRET_REPAIR_WEIGHT = 0.35;
inline constexpr double SMALLP_EDGE_WEIGHT = 0.35;
inline constexpr size_t TSPLIB_IO_BUFFER_BYTES = 1u << 16;

inline constexpr double BHH_REFERENCE = 0.7124;
