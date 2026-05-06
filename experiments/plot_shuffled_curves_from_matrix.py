"""
Plot shuffled-experiment accuracy curves from a provided n_demo x n_relabel matrix.

This is intentionally NOT a heatmap: it produces a plot_single-style multi-curve
line chart with x = n_demo (K) and one curve per n_relabel.

The matrix values should be accuracies in [0,1].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/data/oermannlab/users/im2178/class-representation-icl/plots_shuffled",
    )
    ap.add_argument("--output_name", type=str, default="shuffled_accuracy_all_curves_qwen.pdf")
    args = ap.parse_args()

    # x-axis: number of demonstrations N
    n_demo = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int)
    # one curve per n_relabel (k)
    n_relabel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int)

    # rows correspond to n_demo in the same order; columns correspond to n_relabel.
    # values are accuracies in [0,1].
    acc = np.array(
        [
            [0.267, 0.263, 0.163, 0.227, 0.183, 0.197, 0.177, 0.260, 0.233, 0.233],  # N=0
            [0.358, 0.354, 0.324, 0.335, 0.312, 0.398, 0.342, 0.345, 0.314, 0.314],  # N=10
            [0.367, 0.346, 0.396, 0.363, 0.324, 0.462, 0.396, 0.349, 0.298, 0.298],  # N=20
            [0.414, 0.384, 0.472, 0.386, 0.368, 0.528, 0.468, 0.421, 0.377, 0.377],  # N=30
            [0.463, 0.429, 0.464, 0.515, 0.338, 0.539, 0.416, 0.447, 0.488, 0.488],  # N=40
            [0.443, 0.439, 0.517, 0.505, 0.363, 0.585, 0.474, 0.447, 0.466, 0.466],  # N=50
            [0.509, 0.513, 0.486, 0.575, 0.381, 0.566, 0.429, 0.452, 0.543, 0.543],  # N=60
            [0.468, 0.503, 0.518, 0.562, 0.380, 0.563, 0.436, 0.485, 0.547, 0.547],  # N=70
            [0.506, 0.494, 0.529, 0.529, 0.396, 0.573, 0.452, 0.486, 0.553, 0.553],  # N=80
            [0.540, 0.516, 0.516, 0.527, 0.409, 0.572, 0.498, 0.511, 0.550, 0.550],  # N=90
            [0.493, 0.488, 0.505, 0.529, 0.401, 0.602, 0.486, 0.533, 0.530, 0.530],  # N=100
        ],
        dtype=float,
    )

    if acc.shape != (len(n_demo), len(n_relabel)):
        raise ValueError(f"Matrix shape mismatch: got {acc.shape}, expected {(len(n_demo), len(n_relabel))}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(n_relabel)))
    for j, k in enumerate(n_relabel):
        ax.plot(n_demo, acc[:, j], marker="o", linewidth=2.5, markersize=6, color=colors[j], label=str(int(k)))

    ax.set_xlabel("Number of Demonstrations (N)")
    ax.set_ylabel("Shuffled accuracy (eval under shuffled mapping)")
    ax.set_title("Shuffled Labels Experiment (Qwen2-7B-Base, 3-class)")
    ax.set_ylim(0.1, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="n_relabel", ncol=2)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

