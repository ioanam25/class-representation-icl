#!/usr/bin/env python3
"""
Analyze gold-init optimized vs gold labels (used directly).
1. Collect metrics from gold-init results (if CSVs don't exist yet)
2. Collect metrics from gold label (synonym_gold) results for comparison
3. Print summary tables and write gold_init_experiment_summary.md
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add parent dir so we can import collect_metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect_metrics import collect_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Experiment configurations ────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "label": "Sentiment 3-class",
        "gold_init_dir": "learning_curves/learning_curves_gold_init_3classes_qwen/claude_multitask/qwen2_7b_base",
        "gold_label_dir": "learning_curves/learning_curves_synonym_gold_3classes_qwen/claude_multitask/qwen2_7b_base",
        "gold_init_pickle_dir": "relabelings/qwen2_7b_base_claude_multitask_relabelings_gold_init",
        "num_classes": 3,
        "gold_words": {'A': 'joy', 'C': 'anger', 'D': 'fear'},
    },
    {
        "label": "Sentiment 5-class",
        "gold_init_dir": "learning_curves/learning_curves_gold_init_5classes_qwen/claude_multitask/qwen2_7b_base",
        "gold_label_dir": "learning_curves/learning_curves_synonym_gold_5classes_qwen/claude_multitask/qwen2_7b_base",
        "gold_init_pickle_dir": "relabelings/qwen2_7b_base_claude_multitask_relabelings_gold_init",
        "num_classes": 5,
        "gold_words": {'A': 'joy', 'B': 'sadness', 'C': 'anger', 'D': 'fear', 'E': 'surprise'},
    },
    {
        "label": "TREC 5-class",
        "gold_init_dir": "learning_curves/learning_curves_gold_init_5classes_TREC_qwen/TREC_coarse/qwen2_7b_base",
        "gold_label_dir": "learning_curves/learning_curves_synonym_gold_TREC_5classes_qwen/TREC_coarse/qwen2_7b_base",  # may not exist yet
        "gold_init_pickle_dir": "relabelings/qwen2_7b_base_TREC_coarse_relabelings_gold_init",
        "num_classes": 5,
        "gold_words": {'A': 'entity', 'B': 'description', 'C': 'human', 'D': 'location', 'E': 'numeric'},
    },
]

N_RELABEL_GRID = list(range(10, 101, 10))


def collect_or_load_csv(base_dir):
    """Collect metrics into CSV if not already done, then load it."""
    base_path = PROJECT_ROOT / base_dir
    csv_path = base_path / "consolidated_metrics.csv"
    
    if csv_path.exists():
        print(f"  Loading existing CSV: {csv_path}")
        return pd.read_csv(csv_path)
    
    print(f"  Collecting metrics from: {base_path}")
    all_data = collect_all_metrics(str(base_path))
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path} ({len(df)} rows)")
    return df


def load_relabeling_info(pickle_dir, num_classes):
    """Load the gold-init relabeling results (token assignments) from pickles."""
    info = {}
    for k in N_RELABEL_GRID:
        pkl_path = PROJECT_ROOT / pickle_dir / f"qwen2_7b_base_relabelings_{num_classes}classes_{k}examples_gold_init.pkl"
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            labels = data.get('new_labels', {})
            gold = data.get('gold_words', {})
            obj = data.get('objective', None)
            
            token_names = {}
            for cls_key, val in labels.items():
                if isinstance(val, tuple):
                    token_names[cls_key] = val[0].replace('Ġ', '')
                else:
                    token_names[cls_key] = str(val)
            
            info[k] = {
                'tokens': token_names,
                'objective': obj,
                'gold_words': gold,
            }
    return info


def get_gold_init_accuracy(df):
    """
    From gold-init metrics DataFrame, compute mean ± std accuracy per (n_relabel, n_demo).
    Returns pivot tables.
    """
    if 'accuracy_constrained' not in df.columns:
        return None, None
    
    grouped = df.groupby(['n_relabel', 'demo_id'])['accuracy_constrained'].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = ['n_relabel', 'n_demo', 'mean', 'std', 'count']
    
    pivot_mean = grouped.pivot(index='n_demo', columns='n_relabel', values='mean')
    pivot_std = grouped.pivot(index='n_demo', columns='n_relabel', values='std')
    
    return pivot_mean, pivot_std


def get_gold_label_accuracy(df):
    """
    From gold label metrics DataFrame, compute mean ± std accuracy per n_demo.
    Returns a DataFrame with columns: n_demo, mean, std.
    """
    if 'accuracy_constrained' not in df.columns:
        return None
    
    grouped = df.groupby('demo_id')['accuracy_constrained'].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = ['n_demo', 'mean', 'std', 'count']
    return grouped


def find_best_gold_init_curve(pivot_mean):
    """Find the n_relabel (k) with highest accuracy at max N demos."""
    if pivot_mean is None:
        return None, None
    max_n = pivot_mean.index.max()
    best_k = pivot_mean.loc[max_n].idxmax()
    return best_k, pivot_mean[best_k]


def format_comparison_table(gi_pivot_mean, gi_pivot_std, gold_df, demo_values=None):
    """Format markdown table comparing gold-init curves with gold label curve."""
    if demo_values is None:
        demo_values = sorted(gi_pivot_mean.index)
    
    relabel_values = sorted(gi_pivot_mean.columns)
    
    # Header
    header = "| N (demos) | **Gold labels** | " + " | ".join(f"k={k}" for k in relabel_values) + " |"
    separator = "|---" * (2 + len(relabel_values)) + "|"
    
    rows = [header, separator]
    for n_demo in demo_values:
        # Gold label value
        if gold_df is not None and n_demo in gold_df['n_demo'].values:
            gm = gold_df[gold_df['n_demo'] == n_demo]['mean'].values[0] * 100
            gs = gold_df[gold_df['n_demo'] == n_demo]['std'].values[0] * 100
            gold_cell = f"**{gm:.1f}±{gs:.1f}**"
        else:
            gold_cell = "—"
        
        # Gold-init values
        cells = []
        for k in relabel_values:
            if n_demo in gi_pivot_mean.index and k in gi_pivot_mean.columns and not pd.isna(gi_pivot_mean.loc[n_demo, k]):
                m = gi_pivot_mean.loc[n_demo, k] * 100
                s = gi_pivot_std.loc[n_demo, k] * 100 if not pd.isna(gi_pivot_std.loc[n_demo, k]) else 0
                cells.append(f"{m:.1f}±{s:.1f}")
            else:
                cells.append("—")
        
        rows.append(f"| {n_demo} | {gold_cell} | " + " | ".join(cells) + " |")
    
    return "\n".join(rows)


def main():
    all_results = {}
    all_relabel_info = {}
    
    for exp in EXPERIMENTS:
        label = exp['label']
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        
        # 1. Gold-init optimized results
        print(f"\n  --- Gold-init optimized ---")
        gi_df = collect_or_load_csv(exp['gold_init_dir'])
        gi_mean, gi_std = get_gold_init_accuracy(gi_df)
        
        # 2. Gold label results
        print(f"\n  --- Gold labels ---")
        gold_label_path = PROJECT_ROOT / exp['gold_label_dir']
        if gold_label_path.exists():
            gold_df = collect_or_load_csv(exp['gold_label_dir'])
            gold_acc = get_gold_label_accuracy(gold_df)
        else:
            print(f"  Gold label dir not found: {gold_label_path}")
            print(f"  (Gold label comparison will be skipped for {label})")
            gold_acc = None
        
        # 3. Relabeling info (what tokens the optimizer found)
        relabel_info = load_relabeling_info(exp['gold_init_pickle_dir'], exp['num_classes'])
        
        all_results[label] = {
            'gi_mean': gi_mean, 'gi_std': gi_std,
            'gold_acc': gold_acc,
            'gold_words': exp['gold_words'],
        }
        all_relabel_info[label] = relabel_info
    
    # ── Print summary to stdout ──────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  SUMMARY: Gold-init Optimized vs Gold Labels")
    print("="*80)
    
    for label, res in all_results.items():
        print(f"\n--- {label} ---")
        gi_mean = res['gi_mean']
        gold_acc = res['gold_acc']
        
        if gold_acc is not None:
            print(f"  Gold labels accuracy by N demos:")
            for _, row in gold_acc.iterrows():
                print(f"    N={int(row['n_demo']):3d}: {row['mean']*100:.1f}% ± {row['std']*100:.1f}%")
        
        if gi_mean is not None:
            best_k, best_curve = find_best_gold_init_curve(gi_mean)
            print(f"\n  Best gold-init curve: k={best_k}")
            for n_demo in sorted(best_curve.index):
                gi_val = best_curve[n_demo] * 100
                gold_val = gold_acc[gold_acc['n_demo'] == n_demo]['mean'].values[0] * 100 if (gold_acc is not None and n_demo in gold_acc['n_demo'].values) else None
                diff_str = f"  Δ={gi_val - gold_val:+.1f}pp" if gold_val is not None else ""
                print(f"    N={n_demo:3d}: {gi_val:.1f}%{diff_str}")
    
    # ── Write markdown summary ───────────────────────────────────────────────
    md = []
    md.append("# Gold-Init Optimized vs Gold Labels — Experiment Summary")
    md.append("")
    md.append("## Overview")
    md.append("")
    md.append("This experiment tests whether **optimizing from gold label initialization** can beat using **gold labels directly**.")
    md.append("")
    md.append("- **Gold labels**: Semantically meaningful tokens (e.g., *joy*, *anger*, *fear*) used as-is for classification")
    md.append("- **Gold-init optimized**: Hill-climbing optimization starts from gold labels and searches for better tokens")
    md.append("")
    md.append("If the optimizer finds tokens that outperform gold labels, it means the optimization landscape near gold labels")
    md.append("contains even better solutions — the model's internal representations don't perfectly align with human-chosen words.")
    md.append("")
    md.append("**Model**: Qwen2-7B (base)  ")
    md.append("**Dataset**: Sentiment classification (claude_multitask)  ")
    md.append("**k** = number of examples used for relabeling optimization (10–100)  ")
    md.append("**N** = number of in-context demonstrations at inference (0–100)  ")
    md.append("**Runs per config**: 10 (different random demo selections)")
    md.append("")
    
    for label, res in all_results.items():
        gi_mean = res['gi_mean']
        gi_std = res['gi_std']
        gold_acc = res['gold_acc']
        gold_words = res['gold_words']
        relabel_info = all_relabel_info[label]
        
        md.append(f"---")
        md.append(f"## {label}")
        md.append("")
        md.append(f"**Gold labels**: {', '.join(f'{c} → *{w}*' for c, w in sorted(gold_words.items()))}")
        md.append("")
        
        # What the optimizer found
        md.append("### Optimized tokens found (starting from gold)")
        md.append("")
        md.append("| k (relabel examples) | Optimized tokens | Objective |")
        md.append("|---|---|---|")
        for k in sorted(relabel_info.keys()):
            info = relabel_info[k]
            tokens_str = ", ".join(f"{c}→*{t}*" for c, t in sorted(info['tokens'].items()))
            obj = f"{info['objective']:.2f}" if info['objective'] is not None else "—"
            md.append(f"| {k} | {tokens_str} | {obj} |")
        md.append("")
        
        # Full comparison table
        if gi_mean is not None:
            md.append("### Accuracy (%) — Gold labels vs Gold-init optimized (by k)")
            md.append("")
            md.append(format_comparison_table(gi_mean, gi_std, gold_acc))
            md.append("")
        
        # Best curve comparison
        if gi_mean is not None and gold_acc is not None:
            best_k, best_curve = find_best_gold_init_curve(gi_mean)
            best_k_std = gi_std[best_k] if best_k in gi_std.columns else None
            
            md.append(f"### Best gold-init curve (k={best_k}) vs Gold labels")
            md.append("")
            md.append("| N (demos) | Gold labels | Gold-init optimized (k={}) | Δ (pp) |".format(best_k))
            md.append("|---|---|---|---|")
            
            for n_demo in sorted(best_curve.index):
                gi_val = best_curve[n_demo] * 100
                gi_s = best_k_std[n_demo] * 100 if best_k_std is not None and not pd.isna(best_k_std[n_demo]) else 0
                
                if n_demo in gold_acc['n_demo'].values:
                    g_val = gold_acc[gold_acc['n_demo'] == n_demo]['mean'].values[0] * 100
                    g_s = gold_acc[gold_acc['n_demo'] == n_demo]['std'].values[0] * 100
                    diff = gi_val - g_val
                    md.append(f"| {n_demo} | {g_val:.1f}±{g_s:.1f} | {gi_val:.1f}±{gi_s:.1f} | {diff:+.1f} |")
                else:
                    md.append(f"| {n_demo} | — | {gi_val:.1f}±{gi_s:.1f} | — |")
            md.append("")
            
            # Summary stats
            common_demos = [n for n in best_curve.index if n in gold_acc['n_demo'].values]
            if common_demos:
                diffs = []
                for n in common_demos:
                    gi_val = best_curve[n] * 100
                    g_val = gold_acc[gold_acc['n_demo'] == n]['mean'].values[0] * 100
                    diffs.append(gi_val - g_val)
                
                avg_diff = np.mean(diffs)
                max_demo = max(common_demos)
                max_diff = best_curve[max_demo] * 100 - gold_acc[gold_acc['n_demo'] == max_demo]['mean'].values[0] * 100
                
                winner_count = sum(1 for d in diffs if d > 0.5)  # >0.5pp margin
                
                md.append(f"**Average Δ across all N**: {avg_diff:+.1f}pp  ")
                md.append(f"**Δ at N={max_demo}**: {max_diff:+.1f}pp  ")
                md.append(f"**Gold-init wins at**: {winner_count}/{len(diffs)} N values (by >0.5pp)")
                md.append("")
                
                # What tokens were found for best k
                if best_k in relabel_info:
                    tokens = relabel_info[best_k]['tokens']
                    md.append(f"**Best gold-init tokens (k={best_k})**: {', '.join(f'{c}→*{t}*' for c, t in sorted(tokens.items()))}")
                    md.append("")
    
    # Key findings
    md.append("---")
    md.append("## Key Findings")
    md.append("")
    
    for label, res in all_results.items():
        gi_mean = res['gi_mean']
        gold_acc = res['gold_acc']
        
        if gi_mean is None or gold_acc is None:
            continue
        
        best_k, best_curve = find_best_gold_init_curve(gi_mean)
        max_n = best_curve.index.max()
        gi_at_max = best_curve[max_n] * 100
        gold_at_max = gold_acc[gold_acc['n_demo'] == max_n]['mean'].values[0] * 100 if max_n in gold_acc['n_demo'].values else None
        
        md.append(f"### {label}")
        md.append("")
        if gold_at_max is not None:
            diff = gi_at_max - gold_at_max
            if diff > 0.5:
                md.append(f"- Gold-init optimization **improves** over gold labels by **{diff:+.1f}pp** at N={max_n} (best k={best_k})")
            elif diff < -0.5:
                md.append(f"- Gold labels **outperform** gold-init optimization by **{abs(diff):.1f}pp** at N={max_n}")
            else:
                md.append(f"- Gold-init optimization performs **comparably** to gold labels (Δ={diff:+.1f}pp at N={max_n})")
        
        # Check across all k values at max N
        all_k_at_max = gi_mean.loc[max_n]
        any_beats_gold = (all_k_at_max * 100 > gold_at_max).any() if gold_at_max else False
        best_any_k = all_k_at_max.max() * 100
        best_any_k_id = all_k_at_max.idxmax()
        
        md.append(f"- Best gold-init accuracy at N={max_n}: **{best_any_k:.1f}%** (k={best_any_k_id})")
        if gold_at_max:
            md.append(f"- Gold label accuracy at N={max_n}: **{gold_at_max:.1f}%**")
        md.append("")
    
    # Write
    md_path = PROJECT_ROOT / "gold_init_experiment_summary.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md))
    print(f"\n✓ Summary written to: {md_path}")


if __name__ == "__main__":
    main()
