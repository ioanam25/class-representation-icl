"""
Create a side-by-side plot for synonym-label results:
  - Left: 3-class (Optimized vs gold vs syn1..syn4)
  - Right: 5-class (Optimized vs gold vs syn1..syn3)

The script is intentionally data-driven from the values provided in
`synonym_experiment_summary.md` (mean over 10 runs).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Shared K grid
    K = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int)

    # 3-class: Joy(A) / Anger(C) / Fear(D)
    optimized_3 = np.array([0.623, 0.786, 0.790, 0.805, 0.805, 0.804, 0.812, 0.821, 0.820, 0.817, 0.823], dtype=float)
    gold_3 = np.array([0.540, 0.832, 0.848, 0.850, 0.849, 0.849, 0.851, 0.858, 0.856, 0.855, 0.861], dtype=float)
    syn1_3 = np.array([0.437, 0.788, 0.807, 0.819, 0.819, 0.817, 0.815, 0.814, 0.818, 0.825, 0.821], dtype=float)
    syn2_3 = np.array([0.347, 0.770, 0.770, 0.793, 0.786, 0.794, 0.776, 0.762, 0.762, 0.786, 0.781], dtype=float)
    syn3_3 = np.array([0.477, 0.786, 0.789, 0.804, 0.812, 0.808, 0.801, 0.807, 0.801, 0.816, 0.821], dtype=float)
    syn4_3 = np.array([0.490, 0.761, 0.794, 0.809, 0.796, 0.814, 0.801, 0.815, 0.811, 0.802, 0.812], dtype=float)

    # 5-class: Joy(A) / Sadness(B) / Anger(C) / Fear(D) / Surprise(E)
    optimized_5 = np.array([0.438, 0.585, 0.623, 0.619, 0.628, 0.648, 0.672, 0.656, 0.668, 0.690, 0.696], dtype=float)
    gold_5 = np.array([0.408, 0.736, 0.762, 0.765, 0.760, 0.762, 0.768, 0.773, 0.767, 0.783, 0.787], dtype=float)
    syn1_5 = np.array([0.276, 0.607, 0.674, 0.678, 0.695, 0.708, 0.715, 0.714, 0.723, 0.730, 0.727], dtype=float)
    syn2_5 = np.array([0.242, 0.639, 0.683, 0.679, 0.670, 0.686, 0.692, 0.696, 0.703, 0.717, 0.724], dtype=float)
    syn3_5 = np.array([0.252, 0.580, 0.625, 0.666, 0.609, 0.625, 0.628, 0.645, 0.674, 0.685, 0.634], dtype=float)

    out_dir = Path("/gpfs/data/oermannlab/users/im2178/class-representation-icl/plots_synonyms")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synonym_accuracy_side_by_side_qwen.pdf"
    # Figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Left: 3-class ---
    ln_optimized_3, = ax1.plot(K, optimized_3, marker="o", linewidth=2.8, label="Optimized")
    ln_gold_3, = ax1.plot(K, gold_3, marker="o", linewidth=2.8, label="gold")
    ln_syn1_3, = ax1.plot(K, syn1_3, marker="o", linewidth=2.0, label="syn1")
    ln_syn2_3, = ax1.plot(K, syn2_3, marker="o", linewidth=2.0, label="syn2")
    ln_syn3_3, = ax1.plot(K, syn3_3, marker="o", linewidth=2.0, label="syn3")
    ln_syn4_3, = ax1.plot(K, syn4_3, marker="o", linewidth=2.0, label="syn4")
    ax1.set_title("3-Class (Joy/Anger/Fear)")
    ax1.set_xlabel("K (n_examples)")
    ax1.set_ylabel("Accuracy (mean over 10 runs)")
    ax1.set_ylim(0.1, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend().remove()

    # --- Right: 5-class ---
    ln_optimized_5, = ax2.plot(K, optimized_5, marker="o", linewidth=2.8, label="Optimized")
    ln_gold_5, = ax2.plot(K, gold_5, marker="o", linewidth=2.8, label="gold")
    ln_syn1_5, = ax2.plot(K, syn1_5, marker="o", linewidth=2.0, label="syn1")
    ln_syn2_5, = ax2.plot(K, syn2_5, marker="o", linewidth=2.0, label="syn2")
    ln_syn3_5, = ax2.plot(K, syn3_5, marker="o", linewidth=2.0, label="syn3")
    ax2.set_title("5-Class (Joy/Sadness/Anger/Fear/Surprise)")
    ax2.set_xlabel("K (n_examples)")
    ax2.set_ylim(0.1, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend().remove()

    # Legend descriptions (from `synonym_experiment_summary.md`)
    # syn1/syn2/syn3/syn4 correspond to the token sets you provided for 3-class,
    # and syn1/syn2/syn3 correspond to the 5-class synonym sets shown above.
    legend_items = [
        (ln_gold_3, "gold: joy / anger / fear (A/C/D)"),
        (ln_syn1_3, "syn1: happiness / rage / anxiety"),
        (ln_syn2_3, "syn2: delight / fury / dread"),
        (ln_syn3_3, "syn3: cheerful / wrath / panic"),
        (ln_syn4_3, "syn4: pleased / irritation / terror"),
        (ln_optimized_5, "Optimized: arbitrary tokens (hill-climbing)"),
        (ln_gold_5, "gold (5-class): joy/sadness/anger/fear/surprise (A-E)"),
        (ln_syn1_5, "syn1 (5-class): happiness/grief/rage/anxiety/startled"),
        (ln_syn2_5, "syn2 (5-class): delight/sorrow/fury/dread/awe"),
        (ln_syn3_5, "syn3 (5-class): cheerful/misery/wrath/panic/shock"),
    ]

    # Deduplicate by label text in case handles differ between panels
    seen = set()
    handles, labels = [], []
    for h, lab in legend_items:
        if lab in seen:
            continue
        seen.add(lab)
        handles.append(h)
        labels.append(lab)

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)

    fig.suptitle("Synonym vs Gold Labels vs Optimized Arbitrary Tokens (Qwen2-7B-Base)", fontsize=15)
    # Leave room for legend at the bottom
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

