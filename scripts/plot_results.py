#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt


Row = Tuple[float, float, float, int]

BHH_REFERENCE = 0.7124


def load_results(path: Path) -> Tuple[dict, List[Row]]:
    doc = json.loads(path.read_text())

    summary = doc.get("summary", {})
    rows: List[Row] = []
    for p_str, entry in summary.items():
        p = float(p_str)
        mean = float(entry["mean"])
        stderr = float(entry.get("stderr", 0.0))
        n = int(entry.get("n", 0))
        rows.append((p, mean, stderr, n))

    if not rows:
        raise ValueError("results.json contains no summary rows")

    rows.sort(key=lambda row: row[0])
    return doc, rows



def annotation_layout(
    p_values: Sequence[float], means: Sequence[float], best_idx: int
) -> Tuple[Tuple[float, float], str, str]:
    x_value = p_values[best_idx]
    y_value = means[best_idx]

    x_mid = 0.5 * (min(p_values) + max(p_values))
    y_mid = 0.5 * (min(means) + max(means))

    x_offset = 16 if x_value <= x_mid else -16
    y_offset = 16 if y_value <= y_mid else -16

    horizontal_alignment = "left" if x_offset > 0 else "right"
    vertical_alignment = "bottom" if y_offset > 0 else "top"
    return (x_offset, y_offset), horizontal_alignment, vertical_alignment



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Aldous subset-selection TSP results from results.json."
    )
    parser.add_argument("input_json", help="Path to results.json")
    parser.add_argument(
        "-o", "--output", default="aldous_curve.png", help="Output PNG path"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom title. By default a compact title is generated.",
    )
    args = parser.parse_args()

    doc, rows = load_results(Path(args.input_json))

    p_values = [row[0] for row in rows]
    means = [row[1] for row in rows]
    stderrs = [row[2] for row in rows]

    best_idx = min(range(len(means)), key=lambda idx: means[idx])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)

    (line,) = ax.plot(
        p_values,
        means,
        marker="o",
        linewidth=2,
        label="mean edge length",
    )

    if any(stderr > 0.0 for stderr in stderrs):
        lower = [mean - stderr for mean, stderr in zip(means, stderrs)]
        upper = [mean + stderr for mean, stderr in zip(means, stderrs)]
        ax.fill_between(
            p_values,
            lower,
            upper,
            alpha=0.18,
            color=line.get_color(),
            label="±1 stderr",
        )

    ax.axhline(
        BHH_REFERENCE,
        linestyle="--",
        linewidth=1.5,
        color="grey",
        label=f"BHH asymptotic limit ({BHH_REFERENCE})",
    )

    ax.scatter([p_values[best_idx]], [means[best_idx]], zorder=3)
    xytext, ha, va = annotation_layout(p_values, means, best_idx)
    ax.annotate(
        f"lowest sampled mean\np = {p_values[best_idx]:.2f}, {means[best_idx]:.4f}",
        xy=(p_values[best_idx], means[best_idx]),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        arrowprops={"arrowstyle": "-", "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.95},
    )

    n_value = doc.get("N")
    mode = doc.get("mode")
    instances = doc.get("done")
    restarts = doc.get("restarts")

    title = args.title
    if title is None:
        title = (
            "Aldous subset-selection TSP\n"
            f"N = {n_value}, mode = {mode}, instances = {instances}, restarts = {restarts}"
        )

    ax.set_title(title)
    ax.set_xlabel("Subset fraction p = k / N")
    ax.set_ylabel("Estimated mean cycle edge length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(min(p_values), max(p_values))

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
