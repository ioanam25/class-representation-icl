"""
Plot prompt-format experiment learning curves (all n_relabel curves).

This mirrors the style of experiments/plot_accuracy_single.py, but reads metrics.pickle
directly (no need for consolidated CSV).

Example:
  python experiments/plot_prompt_format_curves.py \
    --base_dir /gpfs/data/oermannlab/users/im2178/class-representation-icl/learning_curves_prompt_arrow_3classes_qwen/claude_multitask/qwen2_7b_base \
    --output_dir /gpfs/data/oermannlab/users/im2178/class-representation-icl/plots_prompt_format \
    --title "Qwen2-7B (Arrow)"
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


RE_RELABEL_DEMO = re.compile(r"relabel(?P<n_relabel>\d+)_demo(?P<n_demo>\d+)")


def iter_metrics_pickles(base_dir: Path):
    yield from base_dir.glob("relabel*_demo*/run_*/metrics.pickle")


def parse_relabel_demo_from_path(p: Path) -> tuple[int, int]:
    for part in p.parts:
        m = RE_RELABEL_DEMO.match(part)
        if m:
            return int(m.group("n_relabel")), int(m.group("n_demo"))
    raise ValueError(f"Could not parse relabel/demo from path: {p}")


def parse_run_id_from_path(p: Path) -> int:
    for part in p.parts:
        if part.startswith("run_"):
            return int(part.split("_", 1)[1])
    raise ValueError(f"Could not parse run id from path: {p}")


def load_accuracy_constrained(p: Path) -> float:
    obj = pickle.load(open(p, "rb"))
    df = obj.get("metrics")
    if df is None:
        raise KeyError(f"metrics missing in {p}")
    acc = df.attrs.get("accuracy_constrained")
    if acc is None:
        # Backward compat: sometimes stored directly in dict
        maybe = obj.get("accuracy_constrained")
        if maybe is not None:
            acc = maybe
    if acc is None:
        raise KeyError(f"accuracy_constrained missing in attrs for {p}")
    return float(acc)


def load_results_df(base_dir: Path) -> pd.DataFrame:
    rows = []
    for mp in iter_metrics_pickles(base_dir):
        n_relabel, n_demo = parse_relabel_demo_from_path(mp)
        run_id = parse_run_id_from_path(mp)
        try:
            acc = load_accuracy_constrained(mp)
        except Exception as e:
            # Keep going; a single corrupt/incomplete run shouldn't kill plots.
            print(f"[warn] failed to read {mp}: {e}")
            continue
        rows.append(
            {
                "n_demo": n_demo,
                "n_relabel": n_relabel,
                "run_id": run_id,
                "accuracy": acc,
                "file": str(mp),
            }
        )
    if not rows:
        raise RuntimeError(f"No metrics.pickle files found/loaded under {base_dir}")
    return pd.DataFrame(rows)


def plot_curves(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    num_classes: int = 3,
):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    # Include 0 if present, and demos >= num_classes (same filter used elsewhere)
    filtered = results_df[(results_df["n_demo"] == 0) | (results_df["n_demo"] >= num_classes)].copy()

    summary = (
        filtered.groupby(["n_demo", "n_relabel"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["n_relabel", "n_demo"])
    )
    summary["ci"] = summary["std"] * stats.t.ppf((1 + 0.95) / 2, summary["count"] - 1) / np.sqrt(summary["count"])

    unique_relabels = sorted(summary["n_relabel"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    color_map = {r: colors[i] for i, r in enumerate(unique_relabels)}

    for r in unique_relabels:
        data = summary[summary["n_relabel"] == r]
        ax.errorbar(
            data["n_demo"],
            data["mean"],
            yerr=data["ci"],
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=4,
            alpha=0.9,
            color=color_map[r],
            label=str(r),
        )

    ax.set_xlabel("Number of Demonstrations (K)")
    ax.set_ylabel("Accuracy (constrained)")
    ax.set_title(title)
    ax.set_ylim(0.1, 1.0)
    ax.legend(title="n_relabel", ncol=2, fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot prompt-format learning curves from metrics.pickle folders.")
    ap.add_argument("--base_dir", type=str, required=True, help="Folder containing relabel*_demo*/run_*/metrics.pickle")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to write PDF plots")
    ap.add_argument("--title", type=str, required=True, help="Plot title")
    ap.add_argument("--num_classes", type=int, default=3, help="Number of classes (default: 3)")
    ap.add_argument(
        "--zero_shot_dir",
        type=str,
        default=None,
        help="Optional folder with relabel*_demo0/run_*/metrics.pickle to merge (adds K=0 points).",
    )
    ap.add_argument(
        "--output_name",
        type=str,
        default="accuracy_curves.pdf",
        help="Filename for output plot (default: accuracy_curves.pdf)",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    df = load_results_df(base_dir)
    # Optionally merge in zero-shot results from a different directory (to add K=0)
    if args.zero_shot_dir:
        zero_dir = Path(args.zero_shot_dir)
        if not zero_dir.exists():
            raise FileNotFoundError(zero_dir)
        zero_df = load_results_df(zero_dir)
        zero_df = zero_df[zero_df["n_demo"] == 0]
        if not zero_df.empty:
            df = pd.concat([df, zero_df], ignore_index=True)
            print(f"Merged {len(zero_df)} zero-shot rows from {zero_dir}")
        else:
            print(f"No K=0 rows found in {zero_dir}; skipping merge")
    out = Path(args.output_dir) / args.output_name
    plot_curves(df, out, title=args.title, num_classes=args.num_classes)


if __name__ == "__main__":
    main()

