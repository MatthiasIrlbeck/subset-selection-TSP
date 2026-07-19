#!/usr/bin/env python3
"""A minimal stand-in for LKH used to validate the external-oracle path.

It speaks the same CLI protocol the solver uses for LKH:
  * probed once as `reference_lkh.py --version`
  * invoked as `reference_lkh.py run.par`, where run.par names PROBLEM_FILE
    (a TSPLIB EXPLICIT FULL_MATRIX) and TOUR_FILE (where to write the tour).

It reads the explicit distance matrix (which, under --periodic, already holds
the torus distances the solver serialized) and returns an optimal tour for
small instances (Held-Karp) or a 2-opt tour for larger ones. Because it solves
the matrix it is handed, comparing its result to the built-in exact torus
solver proves the oracle round-trip -- matrix serialization, tour parsing, and
torus length scoring -- is correct. It is NOT a substitute for real LKH on
large instances; it exists only for validation.
"""
import os
import re
import sys


def read_par(par_path):
    params = {}
    with open(par_path) as fh:
        for line in fh:
            if "=" in line:
                key, val = line.split("=", 1)
                params[key.strip()] = val.strip()
    return params


def read_matrix(tsp_path):
    text = open(tsp_path).read()
    dim = int(re.search(r"DIMENSION\s*:\s*(\d+)", text).group(1))
    marker = "EDGE_WEIGHT_SECTION"
    idx = text.index(marker)
    nums = re.findall(r"-?\d+", text[idx + len(marker):])
    vals = [int(x) for x in nums[: dim * dim]]
    matrix = [vals[i * dim:(i + 1) * dim] for i in range(dim)]
    return dim, matrix


def held_karp(matrix, n):
    """Exact TSP by dynamic programming over subsets. O(2^n n^2)."""
    INF = float("inf")
    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    dp[1][0] = 0
    for mask in range(size):
        if not (mask & 1):
            continue
        row = dp[mask]
        for j in range(n):
            base = row[j]
            if base == INF:
                continue
            mj = matrix[j]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nm = mask | (1 << nxt)
                c = base + mj[nxt]
                if c < dp[nm][nxt]:
                    dp[nm][nxt] = c
                    parent[nm][nxt] = j
    full = size - 1
    best, best_j = INF, -1
    for j in range(1, n):
        c = dp[full][j] + matrix[j][0]
        if c < best:
            best, best_j = c, j
    tour, mask, j = [], full, best_j
    while j != -1:
        tour.append(j)
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = pj
    tour.reverse()
    return tour


def two_opt(matrix, n):
    unvisited = set(range(1, n))
    tour, cur = [0], 0
    while unvisited:
        nxt = min(unvisited, key=lambda x: matrix[cur][x])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for k in range(i + 1, n):
                c, d = tour[k], tour[(k + 1) % n]
                if matrix[a][c] + matrix[b][d] < matrix[a][b] + matrix[c][d] - 1e-9:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    improved = True
                    b = tour[i]
    return tour


def write_tour(path, tour, n):
    with open(path, "w") as fh:
        fh.write(f"NAME : reference\nTYPE : TOUR\nDIMENSION : {n}\nTOUR_SECTION\n")
        for node in tour:
            fh.write(f"{node + 1}\n")
        fh.write("-1\nEOF\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print("reference_lkh 1.0 (validation stand-in)")
        return 0
    par_path = sys.argv[1]
    params = read_par(par_path)
    base = os.path.dirname(os.path.abspath(par_path))
    prob = os.path.join(base, params.get("PROBLEM_FILE", "problem.tsp"))
    out = os.path.join(base, params.get("TOUR_FILE", "out.tour"))
    n, matrix = read_matrix(prob)

    # Faithfully emulate the real LKH's fatal PRECISION guard: LKH stores costs
    # in C `int` and multiplies each by PRECISION (default 100), aborting via
    # eprintf (stderr + exit status 1) when that overflows. Reproducing it here
    # means our tests exercise the same constraint real LKH imposes -- this is
    # exactly the failure that silently invalidated a full campaign run.
    precision = int(params.get("PRECISION", 100))
    if precision > 1:
        int_max = 2**31 - 1
        worst = max((max(row) for row in matrix), default=0)
        if worst * precision > int_max:
            sys.stderr.write("\n*** Error ***\nPRECISION (= %d) is too large\n" % precision)
            return 1

    tour = held_karp(matrix, n) if n <= 17 else two_opt(matrix, n)
    write_tour(out, tour, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
