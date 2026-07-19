#!/usr/bin/env python3
"""Plot Aldous subset-selection TSP result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

BHH_REFERENCE = 0.7124


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    if doc.get("summary_rows"):
        for row in doc["summary_rows"]:
            rows.append(
                {
                    "p": float(row["p"]),
                    "k": int(row.get("k", 0)),
                    "mean": float(row["mean"]),
                    "stderr": float(row.get("stderr", 0.0)),
                    "std": float(row.get("std", 0.0)),
                    "n": int(row.get("n", 0)),
                    "min": float(row.get("min", row["mean"])),
                    "max": float(row.get("max", row["mean"])),
                    "exact_optimal_instances": int(
                        row.get("exact_optimal_instances", 0)
                    ),
                }
            )
    else:
        summary = doc.get("summary", {})
        for p_text, row in summary.items():
            rows.append(
                {
                    "p": float(p_text),
                    "k": int(row.get("k", 0)),
                    "mean": float(row["mean"]),
                    "stderr": float(row.get("stderr", 0.0)),
                    "std": float(row.get("std", 0.0)),
                    "n": int(row.get("n", 0)),
                    "min": float(row.get("min", row["mean"])),
                    "max": float(row.get("max", row["mean"])),
                    "exact_optimal_instances": int(
                        row.get("exact_optimal_instances", 0)
                    ),
                }
            )
    if not rows:
        raise ValueError(f"{path} contains no summary rows")
    rows.sort(key=lambda row: row["p"])
    return doc, rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "p", "k", "mean", "stderr", "std", "n", "min", "max",
                "exact_optimal_instances",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def default_title(doc: dict) -> str:
    return (
        "Aldous subset-selection TSP\n"
        f"N={doc.get('N')}, mode={doc.get('mode')}, instances={doc.get('done')}, "
        f"threads={doc.get('threads')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("aldous_curve.png"))
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-bhh", action="store_true")
    parser.add_argument("--stderr-multiplier", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None, help="Optional summary CSV output")
    args = parser.parse_args()

    doc, rows = load_rows(args.input_json)
    if args.csv is not None:
        write_csv(args.csv, rows)

    p = [row["p"] for row in rows]
    mean = [row["mean"] for row in rows]
    stderr = [row["stderr"] * args.stderr_multiplier for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=args.dpi)
    (line,) = ax.plot(p, mean, marker="o", linewidth=2, label="mean L(k)/k")

    if any(value > 0 for value in stderr):
        lower = [m - e for m, e in zip(mean, stderr)]
        upper = [m + e for m, e in zip(mean, stderr)]
        ax.fill_between(p, lower, upper, alpha=0.18, color=line.get_color(), label=f"±{args.stderr_multiplier:g} stderr")

    if not args.no_bhh:
        ax.axhline(BHH_REFERENCE, linestyle="--", linewidth=1.5, label=f"BHH ref ({BHH_REFERENCE})")

    best = min(range(len(rows)), key=lambda idx: mean[idx])
    ax.scatter([p[best]], [mean[best]], zorder=3)
    ax.annotate(
        f"lowest sampled mean\np={p[best]:.4g}, {mean[best]:.4f}",
        xy=(p[best], mean[best]),
        xytext=(16, 16),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.95},
    )

    ax.set_title(args.title or default_title(doc))
    ax.set_xlabel("subset fraction p = k/N")
    ax.set_ylabel("estimated mean cycle edge length L(k)/k")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(min(p), max(p))
    fig.tight_layout()

    save_kwargs = {"bbox_inches": "tight"}
    if args.format is not None:
        save_kwargs["format"] = args.format
    fig.savefig(args.output, **save_kwargs)


if __name__ == "__main__":
    main()
