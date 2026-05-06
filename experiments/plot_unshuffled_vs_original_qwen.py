"""
Side-by-side learning curves: unshuffled accuracy (shuffled demos, original mapping)
vs. original optimized baseline — both in plot_single style (one curve per n_relabel).

Unshuffled means are frozen from shuffled_experiment_summary.md (Section 2).
Original means loaded from learning_curves_relabel_demos_3classes_qwen/.../consolidated_metrics.csv
(collected on the fly if CSV is missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_BASE = (
    PROJECT_ROOT
    / "learning_curves_relabel_demos_3classes_qwen"
    / "claude_multitask"
    / "qwen2_7b_base"
)

# Rows: N = 0,10,...,100; cols: k = 10,20,...,100 (means from shuffled_experiment_summary.md §2)
UNSHUFFLED_MEAN = np.array(
    [
        [0.447, 0.460, 0.497, 0.540, 0.623, 0.627, 0.680, 0.590, 0.637, 0.637],
        [0.371, 0.372, 0.467, 0.457, 0.494, 0.443, 0.445, 0.403, 0.513, 0.513],
        [0.346, 0.353, 0.462, 0.430, 0.488, 0.392, 0.412, 0.386, 0.522, 0.522],
        [0.303, 0.329, 0.366, 0.359, 0.471, 0.334, 0.376, 0.312, 0.416, 0.416],
        [0.255, 0.282, 0.396, 0.280, 0.500, 0.338, 0.351, 0.253, 0.403, 0.403],
        [0.274, 0.261, 0.364, 0.290, 0.465, 0.319, 0.369, 0.247, 0.380, 0.380],
        [0.225, 0.233, 0.386, 0.255, 0.488, 0.336, 0.340, 0.250, 0.354, 0.354],
        [0.242, 0.247, 0.348, 0.269, 0.458, 0.328, 0.348, 0.220, 0.348, 0.348],
        [0.235, 0.249, 0.322, 0.276, 0.445, 0.337, 0.335, 0.226, 0.334, 0.334],
        [0.209, 0.247, 0.325, 0.260, 0.433, 0.321, 0.334, 0.173, 0.339, 0.339],
        [0.216, 0.247, 0.329, 0.238, 0.412, 0.318, 0.332, 0.150, 0.339, 0.339],
    ],
    dtype=float,
)


def load_original_pivot() -> pd.DataFrame:
    csv_path = ORIGINAL_BASE / "consolidated_metrics.csv"
    if not csv_path.exists():
        sys.path.insert(0, str(PROJECT_ROOT))
        from collect_metrics import collect_all_metrics

        if not ORIGINAL_BASE.exists():
            raise FileNotFoundError(f"Original results directory not found: {ORIGINAL_BASE}")
        rows = collect_all_metrics(str(ORIGINAL_BASE))
        if not rows:
            raise RuntimeError(f"No metrics collected from {ORIGINAL_BASE}")
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    if "accuracy_constrained" not in df.columns:
        raise ValueError("consolidated_metrics.csv missing accuracy_constrained")

    work = df.copy()
    if "num_classes" in work.columns:
        work = work[work["num_classes"] == 3]

    grouped = (
        work.groupby(["n_relabel", "demo_id"])["accuracy_constrained"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="demo_id", columns="n_relabel", values="accuracy_constrained")
    return pivot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "plots_shuffled"),
    )
    ap.add_argument(
        "--output_name",
        type=str,
        default="unshuffled_vs_original_qwen_3class.pdf",
    )
    args = ap.parse_args()

    n_demo = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int)
    n_relabel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int)

    if UNSHUFFLED_MEAN.shape != (len(n_demo), len(n_relabel)):
        raise ValueError(
            f"UNSHUFFLED_MEAN shape {UNSHUFFLED_MEAN.shape} != {(len(n_demo), len(n_relabel))}"
        )

    orig_pivot = load_original_pivot()
    orig_mat = np.full((len(n_demo), len(n_relabel)), np.nan, dtype=float)
    for i, nd in enumerate(n_demo):
        if nd not in orig_pivot.index:
            continue
        for j, k in enumerate(n_relabel):
            if k in orig_pivot.columns:
                orig_mat[i, j] = float(orig_pivot.loc[nd, k])

    if np.all(np.isnan(orig_mat)):
        raise RuntimeError(
            "Original pivot has no overlap with n_demo / n_relabel grid; check consolidated_metrics.csv"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    colors = plt.cm.viridis(np.linspace(0, 1, len(n_relabel)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    for j, k in enumerate(n_relabel):
        ax0.plot(
            n_demo,
            UNSHUFFLED_MEAN[:, j],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[j],
            label=str(int(k)),
        )
        ax1.plot(
            n_demo,
            orig_mat[:, j],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[j],
            label=str(int(k)),
        )

    ax0.set_xlabel("Number of Demonstrations (N)")
    ax0.set_ylabel("Accuracy")
    ax0.set_title("Unshuffled (shuffled demos, eval under original mapping)")
    ax0.set_ylim(0.1, 1.0)
    ax0.grid(True, alpha=0.3)

    ax1.set_xlabel("Number of Demonstrations (N)")
    ax1.set_title("Original / optimized (correct demos, original mapping)")
    ax1.grid(True, alpha=0.3)

    h, lab = ax0.get_legend_handles_labels()
    fig.legend(h, lab, title="n_relabel", ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Qwen2-7B-Base — 3-class sentiment: unshuffled vs. original baseline", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
