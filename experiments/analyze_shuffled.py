#!/usr/bin/env python3
"""
Analyze shuffled-label experiment results and compare with original (optimized) results.

Computes THREE accuracy metrics:
  1. Shuffled accuracy:   demos shuffled, evaluated under shuffled mapping
  2. Unshuffled accuracy: demos shuffled, evaluated under ORIGINAL mapping
  3. Original accuracy:   demos original, evaluated under original mapping (baseline)

Metric (2) answers: when the model sees wrong demonstrations, does it still
predict the original optimized token (i.e., ignore the demos)?

Outputs:
  - Prints summary tables to stdout
  - Updates shuffled_experiment_summary.md with results
"""

import os
import sys
import pickle
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

# Add parent dir so we can import collect_metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect_metrics import collect_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
SHUFFLED_DIR = PROJECT_ROOT / "learning_curves/learning_curves_shuffled_3classes_qwen" / "claude_multitask" / "qwen2_7b_base"
ORIGINAL_DIR = "learning_curves/learning_curves_relabel_demos_3classes_qwen/claude_multitask/qwen2_7b_base"

ORIGINAL_RESULTS_DIR = PROJECT_ROOT / "learning_curves/learning_curves_relabel_demos_3classes_qwen" / "claude_multitask" / "qwen2_7b_base"
ORIG_RELABEL_DIR = PROJECT_ROOT / "relabelings/qwen2_7b_base_relabelings"
SHUF_RELABEL_DIR = PROJECT_ROOT / "relabelings/qwen2_7b_base_relabelings_shuffled"

N_RELABEL_GRID = list(range(10, 101, 10))
N_DEMOS_GRID = list(range(0, 101, 10))  # Include 0-shot
NUM_CLASSES = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_original_mapping(n_relabel):
    """Load the original (unshuffled) class→token_id mapping for a given n_relabel."""
    pkl_name = (
        f"qwen2_7b_base_relabelings_{NUM_CLASSES}classes_128256toptokens_"
        f"isensembledFalse_voting_{n_relabel}examples_1runs.pkl"
    )
    pkl_path = ORIG_RELABEL_DIR / pkl_name
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    labels = data["relabelings"][0]["labels"]
    # Returns {class_letter: token_id}  e.g. {'A': 12345, 'C': 67890, 'D': 11111}
    return {k: v[1] for k, v in labels.items()}


def load_shuffled_mapping(n_relabel):
    """Load the shuffled class→token_id mapping for a given n_relabel."""
    pkl_name = f"qwen2_7b_base_relabelings_{NUM_CLASSES}classes_{n_relabel}examples_shuffled.pkl"
    pkl_path = SHUF_RELABEL_DIR / pkl_name
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    labels = data["relabelings"][0]["labels"]
    return {k: v[1] for k, v in labels.items()}


def compute_0shot_from_original_experiment():
    """
    Load 0-shot results from the ORIGINAL experiment and re-evaluate under
    both the original and shuffled mappings.

    At 0-shot there are no demonstrations, so the model's predictions are
    identical regardless of mapping.  We just re-score the same predictions
    under each mapping.

    Returns a list of dicts with keys:
        n_relabel, n_demo(=0), run_id, acc_shuffled, acc_unshuffled
    """
    rows = []
    if not ORIGINAL_RESULTS_DIR.exists():
        print(f"  Original results dir not found: {ORIGINAL_RESULTS_DIR}")
        return rows

    # Pre-load both mappings
    orig_maps, shuf_maps = {}, {}
    for k in N_RELABEL_GRID:
        om = load_original_mapping(k)
        sm = load_shuffled_mapping(k)
        if om:
            orig_maps[k] = om
        if sm:
            shuf_maps[k] = sm

    for n_relabel in N_RELABEL_GRID:
        if n_relabel not in orig_maps or n_relabel not in shuf_maps:
            continue
        orig_map = orig_maps[n_relabel]
        shuf_map = shuf_maps[n_relabel]

        demo0_dir = ORIGINAL_RESULTS_DIR / f"relabel{n_relabel}_demo0"
        if not demo0_dir.exists():
            continue

        for run_dir in sorted(demo0_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            run_id = int(run_dir.name.split("_")[1])

            pkl_path = run_dir / "metrics.pickle"
            if not pkl_path.exists():
                continue

            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue

            if "metrics" not in data:
                continue
            df = data["metrics"]

            if "emotion_letter" not in df.columns or "highest_prob_token_constrained" not in df.columns:
                continue

            true_classes = df["emotion_letter"].apply(lambda x: str(x).strip())
            preds = df["highest_prob_token_constrained"].values

            valid = pd.notna(preds)
            if valid.sum() == 0:
                continue

            # Accuracy under original mapping (same as stored accuracy_constrained)
            original_targets = true_classes.map(orig_map).values
            acc_unshuffled = accuracy_score(original_targets[valid], preds[valid])

            # Accuracy under shuffled mapping
            shuffled_targets = true_classes.map(shuf_map).values
            acc_shuffled = accuracy_score(shuffled_targets[valid], preds[valid])

            rows.append({
                "n_relabel": n_relabel,
                "n_demo": 0,
                "run_id": run_id,
                "acc_shuffled": acc_shuffled,
                "acc_unshuffled": acc_unshuffled,
            })

    return rows


def compute_unshuffled_accuracy_from_pickles():
    """
    Walk the shuffled results directory tree, load each run's metrics.pickle,
    and compute accuracy against the ORIGINAL (unshuffled) token mapping.

    Returns a list of dicts: [{n_relabel, n_demo, run_id, acc_shuffled, acc_unshuffled}, ...]
    """
    rows = []
    if not SHUFFLED_DIR.exists():
        print(f"  Shuffled results dir not found: {SHUFFLED_DIR}")
        return rows

    # Pre-load all original mappings
    orig_maps = {}
    for k in N_RELABEL_GRID:
        m = load_original_mapping(k)
        if m:
            orig_maps[k] = m

    # Walk relabelK_demoN/run_R directories
    for subdir in sorted(SHUFFLED_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        match = re.match(r"relabel(\d+)_demo(\d+)", subdir.name)
        if not match:
            continue
        n_relabel = int(match.group(1))
        n_demo = int(match.group(2))

        if n_relabel not in orig_maps:
            continue
        orig_map = orig_maps[n_relabel]

        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            run_id = int(run_dir.name.split("_")[1])

            pkl_path = run_dir / "metrics.pickle"
            if not pkl_path.exists():
                continue

            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue

            if "metrics" not in data:
                continue
            df = data["metrics"]

            # Need emotion_letter (true class) and highest_prob_token_constrained (prediction)
            if "emotion_letter" not in df.columns or "highest_prob_token_constrained" not in df.columns:
                continue

            # True class letters (strip leading space for Qwen: ' A' → 'A')
            true_classes = df["emotion_letter"].apply(lambda x: str(x).strip())
            preds = df["highest_prob_token_constrained"].values

            # Compute unshuffled accuracy: compare prediction to ORIGINAL token
            original_targets = true_classes.map(orig_map).values
            valid = pd.notna(preds)
            if valid.sum() == 0:
                continue

            acc_unshuffled = accuracy_score(
                original_targets[valid], preds[valid]
            )

            # Also get shuffled accuracy (already computed, stored in attrs)
            acc_shuffled = df.attrs.get("accuracy_constrained", np.nan)

            rows.append({
                "n_relabel": n_relabel,
                "n_demo": n_demo,
                "run_id": run_id,
                "acc_shuffled": acc_shuffled,
                "acc_unshuffled": acc_unshuffled,
            })

    return rows


def collect_or_load_csv(base_dir):
    """Collect metrics into CSV if not already done, then load it."""
    base_path = PROJECT_ROOT / base_dir if not isinstance(base_dir, Path) else base_dir
    csv_path = base_path / "consolidated_metrics.csv"

    if csv_path.exists():
        print(f"  Loading existing CSV: {csv_path}")
        return pd.read_csv(csv_path)

    if not base_path.exists():
        print(f"  Directory not found: {base_path}")
        return None

    print(f"  Collecting metrics from: {base_path}")
    all_data = collect_all_metrics(str(base_path))
    if not all_data:
        print(f"  No data found in {base_path}")
        return None
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path} ({len(df)} rows)")
    return df


def pivot_accuracy(records_df, acc_col):
    """Pivot a records DataFrame into (n_demo × n_relabel) mean/std tables."""
    grouped = (records_df
               .groupby(["n_relabel", "n_demo"])[acc_col]
               .agg(["mean", "std", "count"])
               .reset_index())
    grouped.columns = ["n_relabel", "n_demo", "mean", "std", "count"]
    pivot_mean = grouped.pivot(index="n_demo", columns="n_relabel", values="mean")
    pivot_std = grouped.pivot(index="n_demo", columns="n_relabel", values="std")
    return pivot_mean, pivot_std


def get_accuracy_table(df, filter_3class=False):
    """From a consolidated CSV, get accuracy pivot tables."""
    if df is None or "accuracy_constrained" not in df.columns:
        return None, None
    work = df.copy()
    if filter_3class and "num_classes" in work.columns:
        work = work[work["num_classes"] == 3]
    grouped = (work
               .groupby(["n_relabel", "demo_id"])["accuracy_constrained"]
               .agg(["mean", "std", "count"])
               .reset_index())
    grouped.columns = ["n_relabel", "n_demo", "mean", "std", "count"]
    pivot_mean = grouped.pivot(index="n_demo", columns="n_relabel", values="mean")
    pivot_std = grouped.pivot(index="n_demo", columns="n_relabel", values="std")
    return pivot_mean, pivot_std


def format_md_table(pivot_mean, pivot_std, label="Accuracy"):
    """Format a pivot table as a markdown table with mean ± std."""
    if pivot_mean is None:
        return f"*No data available for {label}*\n"

    relabel_cols = sorted(pivot_mean.columns)
    header = "| N demos | " + " | ".join(f"k={k}" for k in relabel_cols) + " |"
    sep = "|---------|" + "|".join("--------:" for _ in relabel_cols) + "|"

    rows = [header, sep]
    for n_demo in sorted(pivot_mean.index):
        cells = []
        for k in relabel_cols:
            m = pivot_mean.loc[n_demo, k] if k in pivot_mean.columns else float("nan")
            s = pivot_std.loc[n_demo, k] if (pivot_std is not None and k in pivot_std.columns) else float("nan")
            if pd.isna(m):
                cells.append("—")
            else:
                cells.append(f"{m:.1%} ± {s:.1%}" if not pd.isna(s) else f"{m:.1%}")
        rows.append(f"| {int(n_demo)} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def format_comparison_summary(shuf_mean, unshuf_mean, orig_mean):
    """Head-to-head at N=100 demos: original vs shuffled vs unshuffled."""
    header = "| n_relabel | Original | Shuffled (eval shuffled) | Shuffled (eval original) |"
    sep = "|-----------|--------:|------------------------:|------------------------:|"
    rows = [header, sep]

    all_cols = set()
    for pm in [shuf_mean, unshuf_mean, orig_mean]:
        if pm is not None:
            all_cols |= set(pm.columns)

    n_demo = 100
    for k in sorted(all_cols):
        om = orig_mean.loc[n_demo, k] if (orig_mean is not None and n_demo in orig_mean.index and k in orig_mean.columns) else float("nan")
        sm = shuf_mean.loc[n_demo, k] if (shuf_mean is not None and n_demo in shuf_mean.index and k in shuf_mean.columns) else float("nan")
        um = unshuf_mean.loc[n_demo, k] if (unshuf_mean is not None and n_demo in unshuf_mean.index and k in unshuf_mean.columns) else float("nan")

        om_s = f"{om:.1%}" if not pd.isna(om) else "—"
        sm_s = f"{sm:.1%}" if not pd.isna(sm) else "—"
        um_s = f"{um:.1%}" if not pd.isna(um) else "—"
        rows.append(f"| {k} | {om_s} | {sm_s} | {um_s} |")

    return "\n".join(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Compute shuffled + unshuffled accuracy from raw pickles ────────────
    print("=" * 60)
    print("  Computing shuffled & unshuffled accuracy from raw pickles")
    print("=" * 60)
    records = compute_unshuffled_accuracy_from_pickles()

    # ── 1b. Add 0-shot data from original experiment ──────────────────────────
    print("\n" + "=" * 60)
    print("  Computing 0-shot accuracy from original experiment")
    print("=" * 60)
    zeroshot_records = compute_0shot_from_original_experiment()
    if zeroshot_records:
        print(f"  Loaded {len(zeroshot_records)} 0-shot run records")
        records.extend(zeroshot_records)
    else:
        print("  No 0-shot data found.")

    if records:
        rdf = pd.DataFrame(records)
        print(f"  Total: {len(rdf)} run records (including 0-shot)")
        shuf_mean, shuf_std = pivot_accuracy(rdf, "acc_shuffled")
        unshuf_mean, unshuf_std = pivot_accuracy(rdf, "acc_unshuffled")
    else:
        print("  No shuffled results found yet.")
        rdf = None
        shuf_mean = shuf_std = unshuf_mean = unshuf_std = None

    # ── 2. Collect original (optimized, correct-mapping) results ──────────────
    print("\n" + "=" * 60)
    print("  Collecting original (optimized) results")
    print("=" * 60)
    orig_df = collect_or_load_csv(ORIGINAL_DIR)
    orig_mean, orig_std = get_accuracy_table(orig_df, filter_3class=True)

    # ── Print to stdout ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  SHUFFLED ACCURACY  (demos shuffled, eval under shuffled mapping)")
    print("=" * 80)
    if shuf_mean is not None:
        print(shuf_mean.to_string(float_format="%.3f"))
    else:
        print("  (no data)")

    print("\n" + "=" * 80)
    print("  UNSHUFFLED ACCURACY  (demos shuffled, eval under ORIGINAL mapping)")
    print("=" * 80)
    if unshuf_mean is not None:
        print(unshuf_mean.to_string(float_format="%.3f"))
    else:
        print("  (no data)")

    print("\n" + "=" * 80)
    print("  ORIGINAL ACCURACY  (demos original, eval under original mapping)")
    print("=" * 80)
    if orig_mean is not None:
        print(orig_mean.to_string(float_format="%.3f"))
    else:
        print("  (no data)")

    # ── Update summary .md ────────────────────────────────────────────────────
    md_path = PROJECT_ROOT / "shuffled_experiment_summary.md"
    if md_path.exists():
        md = md_path.read_text()
    else:
        md = "# Shuffled Labels Experiment\n"

    shuf_table = format_md_table(shuf_mean, shuf_std, "Shuffled")
    unshuf_table = format_md_table(unshuf_mean, unshuf_std, "Unshuffled")
    orig_table = format_md_table(orig_mean, orig_std, "Original")
    comparison = format_comparison_summary(shuf_mean, unshuf_mean, orig_mean)

    results_section = f"""## Results

### 1. Shuffled Accuracy (demos shuffled, eval under shuffled mapping)

Does the model learn the wrong mapping from the shuffled demonstrations?

{shuf_table}

### 2. Unshuffled Accuracy (demos shuffled, eval under ORIGINAL mapping)

Despite seeing wrong demonstrations, does the model still predict the original optimized token?

{unshuf_table}

### 3. Original (Optimized) Accuracy — Baseline

Standard experiment: demos use the correct optimized mapping, eval under that mapping.

{orig_table}

### Head-to-Head at N=100 Demonstrations

{comparison}
"""

    marker = "## Results"
    if marker in md:
        md = md[:md.index(marker)] + results_section
    else:
        md += "\n" + results_section

    md_path.write_text(md)
    print(f"\n✓ Updated {md_path}")


if __name__ == "__main__":
    main()
