#!/usr/bin/env python3
"""
Analysis script for imbalanced ICL experiments.

Collects per-class F1 scores from pickle files and produces summary tables:
  - Rows   = K (number of in-context demonstrations)
  - Columns = per-class F1 + macro F1 + accuracy

Tables are printed, saved as CSV, and rendered as LaTeX.

Usage:
    python experiments/analysis_imbalanced.py                        # all experiments
    python experiments/analysis_imbalanced.py --model mistral        # mistral only
    python experiments/analysis_imbalanced.py --num_classes 3        # 3-class only
    python experiments/analysis_imbalanced.py --n_relabel 60         # single relabel value
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Experiment registry ──────────────────────────────────────────────────────
EXPERIMENTS = {
    'mistral_3class': {
        'base_dir': 'learning_curves_imbalanced_3classes_mistral/claude_multitask/mistral_7b_base',
        'model': 'mistral_7b_base',
        'num_classes': 3,
        'class_ratios': {'A': 0.6, 'C': 0.3, 'D': 0.1},
        'emotion_map': {'A': 'Joy', 'C': 'Anger', 'D': 'Fear'},
    },
    'mistral_5class': {
        'base_dir': 'learning_curves_imbalanced_5classes_mistral/claude_multitask/mistral_7b_base',
        'model': 'mistral_7b_base',
        'num_classes': 5,
        'class_ratios': {'A': 0.4, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.1},
        'emotion_map': {'A': 'Joy', 'B': 'Sadness', 'C': 'Anger', 'D': 'Fear', 'E': 'Surprise'},
    },
    'qwen_3class': {
        'base_dir': 'learning_curves_imbalanced_3classes_qwen/claude_multitask/qwen2_7b_base',
        'model': 'qwen2_7b_base',
        'num_classes': 3,
        'class_ratios': {'A': 0.6, 'C': 0.3, 'D': 0.1},
        'emotion_map': {'A': 'Joy', 'C': 'Anger', 'D': 'Fear'},
    },
    'qwen_5class': {
        'base_dir': 'learning_curves_imbalanced_5classes_qwen/claude_multitask/qwen2_7b_base',
        'model': 'qwen2_7b_base',
        'num_classes': 5,
        'class_ratios': {'A': 0.4, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.1},
        'emotion_map': {'A': 'Joy', 'B': 'Sadness', 'C': 'Anger', 'D': 'Fear', 'E': 'Surprise'},
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_per_class_f1(df, emotion_map):
    """
    Given a metrics DataFrame (from a single run pickle), compute per-class F1
    using constrained predictions.

    Returns dict:  {emotion_name: f1, ..., 'macro_f1': ..., 'accuracy': ...}
    """
    if 'target_token' not in df.columns or 'highest_prob_token_constrained' not in df.columns:
        return None

    y_true = df['target_token'].values
    y_pred = df['highest_prob_token_constrained'].values

    # Drop NaN predictions
    valid = pd.notna(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return None

    unique_labels = sorted(np.unique(y_true))

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', labels=unique_labels, zero_division=0)
    acc = np.mean(y_true == y_pred)

    # Build token_id → emotion_name mapping from the DataFrame
    token_to_emotion = {}
    if 'emotion_letter' in df.columns and 'token_relabel_id' in df.columns:
        for _, row in df[['emotion_letter', 'token_relabel_id']].drop_duplicates().iterrows():
            letter = str(row['emotion_letter']).strip()
            emo = emotion_map.get(letter, letter)
            token_to_emotion[int(row['token_relabel_id'])] = emo

    result = {}
    for i, tid in enumerate(unique_labels):
        name = token_to_emotion.get(int(tid), f'token_{int(tid)}')
        result[name] = f1_per[i]
    result['Macro F1'] = macro
    result['Accuracy'] = acc
    return result


def load_experiment_results(base_dir, emotion_map):
    """
    Walk the directory tree, load every metrics.pickle, and return a DataFrame
    with columns: n_relabel, n_demo, run_id, <per-class F1>, Macro F1, Accuracy.
    """
    base = Path(base_dir)
    if not base.exists():
        print(f"  WARNING: directory not found: {base_dir}")
        return pd.DataFrame()

    rows = []
    pickle_files = sorted(base.glob('**/metrics.pickle'))
    print(f"  Found {len(pickle_files)} pickle files in {base_dir}")

    for pf in pickle_files:
        # Parse folder: .../relabel{N}_demo{K}/run_{R}/metrics.pickle
        parts = pf.parts
        relabel_demo = run_part = None
        for p in parts:
            if p.startswith('relabel') and '_demo' in p:
                relabel_demo = p
            if p.startswith('run_'):
                run_part = p

        if relabel_demo is None or run_part is None:
            continue

        import re
        m = re.match(r'relabel(\d+)_demo(\d+)', relabel_demo)
        if not m:
            continue
        n_relabel = int(m.group(1))
        n_demo = int(m.group(2))
        run_id = int(run_part.split('_')[1])

        try:
            with open(pf, 'rb') as f:
                data = pickle.load(f)
            df = data['metrics']
            metrics = compute_per_class_f1(df, emotion_map)
            if metrics is None:
                continue
            metrics['n_relabel'] = n_relabel
            metrics['n_demo'] = n_demo
            metrics['run_id'] = run_id
            rows.append(metrics)
        except Exception as e:
            print(f"  Error loading {pf}: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_table(results_df, emotion_names, agg='mean'):
    """
    Build a summary table:
      rows    = K (n_demo), sorted
      columns = per-class F1 columns + Macro F1 + Accuracy

    Values are aggregated (mean ± std) across all n_relabel and run_id for each K.
    """
    if results_df.empty:
        return pd.DataFrame()

    metric_cols = list(emotion_names) + ['Macro F1', 'Accuracy']
    # Keep only columns that exist
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    grouped = results_df.groupby('n_demo')[metric_cols]

    if agg == 'mean':
        table = grouped.mean()
    elif agg == 'mean_std':
        means = grouped.mean()
        stds = grouped.std()
        # Format as "mean ± std"
        table = means.copy()
        for col in metric_cols:
            table[col] = means[col].apply(lambda x: f'{x:.3f}') + ' ± ' + stds[col].apply(lambda x: f'{x:.3f}')
    else:
        table = grouped.mean()

    table.index.name = 'K'
    return table


def build_table_by_relabel(results_df, emotion_names, n_relabel_value):
    """
    Build a summary table for a SINGLE n_relabel value:
      rows    = K (n_demo), sorted
      columns = per-class F1 + Macro F1 + Accuracy

    Values are mean ± std across runs.
    """
    if results_df.empty:
        return pd.DataFrame()

    sub = results_df[results_df['n_relabel'] == n_relabel_value]
    if sub.empty:
        print(f"  No data for n_relabel={n_relabel_value}")
        return pd.DataFrame()

    metric_cols = list(emotion_names) + ['Macro F1', 'Accuracy']
    metric_cols = [c for c in metric_cols if c in sub.columns]

    means = sub.groupby('n_demo')[metric_cols].mean()
    stds = sub.groupby('n_demo')[metric_cols].std()

    table = pd.DataFrame(index=means.index)
    table.index.name = 'K'
    for col in metric_cols:
        table[col] = means[col].apply(lambda x: f'{x:.3f}') + ' ± ' + stds[col].fillna(0).apply(lambda x: f'{x:.3f}')

    return table


def build_numeric_table_by_relabel(results_df, emotion_names, n_relabel_value):
    """
    Same as build_table_by_relabel but returns numeric (mean) values for plotting/CSV.
    """
    if results_df.empty:
        return pd.DataFrame()

    sub = results_df[results_df['n_relabel'] == n_relabel_value]
    if sub.empty:
        return pd.DataFrame()

    metric_cols = list(emotion_names) + ['Macro F1', 'Accuracy']
    metric_cols = [c for c in metric_cols if c in sub.columns]

    table = sub.groupby('n_demo')[metric_cols].mean()
    table.index.name = 'K'
    return table


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Analyse imbalanced ICL experiments')
    parser.add_argument('--model', type=str, default=None, choices=['mistral', 'qwen'],
                        help='Filter by model (default: both)')
    parser.add_argument('--num_classes', type=int, default=None, choices=[3, 5],
                        help='Filter by number of classes (default: both)')
    parser.add_argument('--n_relabel', type=int, default=None,
                        help='Show table for a single n_relabel value (default: aggregate all)')
    parser.add_argument('--output_dir', type=str, default='tables_imbalanced',
                        help='Directory to save output tables')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter experiments
    selected = {}
    for key, cfg in EXPERIMENTS.items():
        if args.model and args.model not in key:
            continue
        if args.num_classes and cfg['num_classes'] != args.num_classes:
            continue
        selected[key] = cfg

    if not selected:
        print("No experiments match the given filters.")
        return

    for exp_name, cfg in selected.items():
        print(f"\n{'='*80}")
        print(f"  Experiment: {exp_name}")
        print(f"  Model: {cfg['model']}  |  Classes: {cfg['num_classes']}")
        print(f"  Class ratios: {cfg['class_ratios']}")
        print(f"  Base dir: {cfg['base_dir']}")
        print(f"{'='*80}")

        emotion_names = list(cfg['emotion_map'].values())

        results_df = load_experiment_results(cfg['base_dir'], cfg['emotion_map'])
        if results_df.empty:
            print("  No results found. Skipping.\n")
            continue

        n_relabel_values = sorted(results_df['n_relabel'].unique())
        n_demo_values = sorted(results_df['n_demo'].unique())
        print(f"  n_relabel values: {n_relabel_values}")
        print(f"  n_demo (K) values: {n_demo_values}")
        print(f"  Total records: {len(results_df)}")

        # ── Print ratio legend ──
        print(f"\n  Class ratios in demonstrations:")
        for letter, emo in cfg['emotion_map'].items():
            ratio = cfg['class_ratios'].get(letter, '?')
            print(f"    {emo} ({letter}): {ratio}")

        if args.n_relabel is not None:
            # Single n_relabel table
            print(f"\n  ── Table for n_relabel = {args.n_relabel} ──")
            table = build_table_by_relabel(results_df, emotion_names, args.n_relabel)
            if not table.empty:
                print(table.to_string())

                # Save CSV (numeric version)
                num_table = build_numeric_table_by_relabel(results_df, emotion_names, args.n_relabel)
                csv_path = os.path.join(args.output_dir, f'{exp_name}_relabel{args.n_relabel}.csv')
                num_table.to_csv(csv_path, float_format='%.4f')
                print(f"\n  Saved: {csv_path}")

                # LaTeX
                latex_path = os.path.join(args.output_dir, f'{exp_name}_relabel{args.n_relabel}.tex')
                with open(latex_path, 'w') as f:
                    f.write(table.to_latex(escape=False))
                print(f"  Saved: {latex_path}")
        else:
            # Table for EACH n_relabel value
            for nr in n_relabel_values:
                print(f"\n  ── n_relabel = {nr} ──")
                table = build_table_by_relabel(results_df, emotion_names, nr)
                if not table.empty:
                    print(table.to_string())

                    num_table = build_numeric_table_by_relabel(results_df, emotion_names, nr)
                    csv_path = os.path.join(args.output_dir, f'{exp_name}_relabel{nr}.csv')
                    num_table.to_csv(csv_path, float_format='%.4f')
                    print(f"  Saved: {csv_path}")

            # Also produce an AGGREGATED table (mean across all n_relabel)
            print(f"\n  ── Aggregated across all n_relabel values ──")
            agg_table = build_table(results_df, emotion_names, agg='mean_std')
            if not agg_table.empty:
                print(agg_table.to_string())

                agg_num = build_table(results_df, emotion_names, agg='mean')
                csv_path = os.path.join(args.output_dir, f'{exp_name}_aggregated.csv')
                agg_num.to_csv(csv_path, float_format='%.4f')
                print(f"\n  Saved: {csv_path}")

                latex_path = os.path.join(args.output_dir, f'{exp_name}_aggregated.tex')
                with open(latex_path, 'w') as f:
                    f.write(agg_table.to_latex(escape=False))
                print(f"  Saved: {latex_path}")

    print(f"\nDone. All tables saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
