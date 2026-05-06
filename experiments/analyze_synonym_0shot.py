#!/usr/bin/env python3
import os
import pickle
import re
from pathlib import Path
from statistics import mean

ROOT = Path("/gpfs/data/oermannlab/users/im2178/class-representation-icl")
SUMMARY_MD = ROOT / "synonym_experiment_summary.md"

SET_NAMES_3C = ["syn1", "syn2", "syn3", "syn4"]
SET_NAMES_5C = ["syn1", "syn2", "syn3"]

def find_metrics_for(root_folder: str):
    """
    Return list of metrics.pickle paths for K=0 runs under a given root_folder.
    """
    base = ROOT / root_folder / "claude_multitask" / "qwen2_7b_base" / "relabel0_demo0"
    if not base.exists():
        return []
    return sorted(base.glob("run_*/metrics.pickle"))

def load_accuracy(metrics_pickle: Path) -> float:
    with open(metrics_pickle, "rb") as f:
        obj = pickle.load(f)
    # Accuracy is stored on the DataFrame as attrs (see src/LLMGeometry/evaluation.py).
    metrics_df = obj.get("metrics", None)
    if metrics_df is None:
        raise KeyError(f"Missing 'metrics' in: {metrics_pickle}")
    acc = metrics_df.attrs.get("accuracy_constrained", None)
    if acc is None:
        # Fallback to unconstrained accuracy if constrained is missing.
        acc = metrics_df.attrs.get("accuracy", None)
    if acc is None:
        raise KeyError(
            f"Missing accuracy_constrained/accuracy in metrics.attrs for: {metrics_pickle}"
        )
    return float(acc)

def aggregate_zero_shot():
    results = {}
    # 3-class
    for s in SET_NAMES_3C:
        folder = f"learning_curves_synonym_{s}_3classes_qwen"
        picks = find_metrics_for(folder)
        if picks:
            accs = [load_accuracy(p) for p in picks]
            results[("3c", s)] = mean(accs)
    # 5-class
    for s in SET_NAMES_5C:
        folder = f"learning_curves_synonym_{s}_5classes_qwen"
        picks = find_metrics_for(folder)
        if picks:
            accs = [load_accuracy(p) for p in picks]
            results[("5c", s)] = mean(accs)
    return results

def format_float(x: float) -> str:
    return f"{x:.3f}"

def update_summary_md(results: dict):
    text = SUMMARY_MD.read_text()
    # Update 3-class row
    # Current line format:
    # | 0   | —         | **0.540** | —     | —     | —     | —     |
    if any(k[0] == "3c" for k in results.keys()):
        syn1 = format_float(results.get(("3c", "syn1"), float("nan"))) if ("3c","syn1") in results else "—"
        syn2 = format_float(results.get(("3c", "syn2"), float("nan"))) if ("3c","syn2") in results else "—"
        syn3 = format_float(results.get(("3c", "syn3"), float("nan"))) if ("3c","syn3") in results else "—"
        syn4 = format_float(results.get(("3c", "syn4"), float("nan"))) if ("3c","syn4") in results else "—"
        # Keep optimized as em-dash and keep gold as is
        new_3c_row = f"| 0   | —         | **0.540** | {syn1 if syn1!='—' else '—'} | {syn2 if syn2!='—' else '—'} | {syn3 if syn3!='—' else '—'} | {syn4 if syn4!='—' else '—'} |"
        text = re.sub(r"\| 0\s+\|\s+—\s+\|\s+\*\*0\.540\*\*\s+\|\s+—\s+\|\s+—\s+\|\s+—\s+\|\s+—\s+\|", new_3c_row, text)
    # Update 5-class row
    if any(k[0] == "5c" for k in results.keys()):
        syn1 = format_float(results.get(("5c", "syn1"), float("nan"))) if ("5c","syn1") in results else "—"
        syn2 = format_float(results.get(("5c", "syn2"), float("nan"))) if ("5c","syn2") in results else "—"
        syn3 = format_float(results.get(("5c", "syn3"), float("nan"))) if ("5c","syn3") in results else "—"
        new_5c_row = f"| 0   | —         | **0.408** | {syn1 if syn1!='—' else '—'} | {syn2 if syn2!='—' else '—'} | {syn3 if syn3!='—' else '—'} |"
        text = re.sub(r"\| 0\s+\|\s+—\s+\|\s+\*\*0\.408\*\*\s+\|\s+—\s+\|\s+—\s+\|\s+—\s+\|", new_5c_row, text)
    SUMMARY_MD.write_text(text)

def main():
    results = aggregate_zero_shot()
    print("Computed 0-shot synonym accuracies:")
    for (k, s), v in sorted(results.items()):
        print(f"{k} {s}: {v:.4f}")
    update_summary_md(results)
    print(f"Updated {SUMMARY_MD}")

if __name__ == "__main__":
    main()

