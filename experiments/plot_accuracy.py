import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import ast
import io
from tqdm import tqdm

def load_metrics_from_csv(csv_path):
    """
    Load metrics from consolidated CSV file and extract accuracy data.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique n_relabel values: {sorted(df['n_relabel'].unique())}")
    print(f"Demo ID range: {df['demo_id'].min()} to {df['demo_id'].max()}")
    print(f"Run ID range: {df['run_id'].min()} to {df['run_id'].max()}")
    
    # Check if accuracy_constrained column exists
    if 'accuracy_constrained' not in df.columns:
        raise ValueError("accuracy_constrained column not found in CSV. Please regenerate the CSV file.")
    
    # Create results DataFrame directly from the CSV
    results_df = pd.DataFrame({
        'n_demo': df['demo_id'],
        'n_relabel': df['n_relabel'], 
        'run_id': df['run_id'],
        'accuracy': df['accuracy_constrained'],
        'file': df['file_path']
    })
    
    # Remove any rows with missing accuracy values
    initial_len = len(results_df)
    results_df = results_df.dropna(subset=['accuracy'])
    final_len = len(results_df)
    
    if final_len < initial_len:
        print(f"Removed {initial_len - final_len} rows with missing accuracy values")
    
    # Print summary
    print(f"\nSuccessfully extracted {len(results_df)} valid results")
    print(f"n_demo range: {results_df['n_demo'].min()} to {results_df['n_demo'].max()}")
    print(f"n_relabel values: {sorted(results_df['n_relabel'].unique())}")
    
    # Verify we have consistent number of runs
    runs_per_config = results_df.groupby(['n_demo', 'n_relabel']).size()
    print("\nNumber of runs per configuration:")
    print(runs_per_config.value_counts().to_string())
    
    if runs_per_config.nunique() > 1:
        print("\nWARNING: Inconsistent number of runs across configurations!")
    
    return results_df

def extract_accuracy_alternative(csv_path):
    """
    Alternative method to extract accuracy - examine the actual structure first
    """
    print("Examining CSV structure...")
    df = pd.read_csv(csv_path, nrows=5)  # Just read first 5 rows for examination
    
    print("Sample row content:")
    for idx, row in df.iterrows():
        print(f"\nRow {idx}:")
        print(f"n_relabel: {row['n_relabel']}")
        print(f"demo_id: {row['demo_id']}")
        print(f"run_id: {row['run_id']}")
        print("Metrics content (first 200 chars):")
        print(repr(str(row['metrics'])[:200]))
        if idx == 0:  # Just show the first row in detail
            break
    
    return None

def plot_accuracy_curves(results_df, num_classes, smooth=False, limited_to_40_demos=False):
    """Create plot with accuracy curves for different relabeling schemes"""
    # Filter data to include 0 demos and demos >= num_classes
    if limited_to_40_demos:
        filtered_df = results_df[((results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)) & (results_df['n_demo'] <= 40)].copy()
    else:
        filtered_df = results_df[(results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)].copy()
    
    # Calculate mean and std of accuracy for each n_demo and n_relabel
    summary = filtered_df.groupby(['n_demo', 'n_relabel'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    # Apply smoothing if requested
    if smooth:
        from scipy.ndimage import uniform_filter1d
        for n_relabel in summary['n_relabel'].unique():
            mask = summary['n_relabel'] == n_relabel
            if mask.sum() >= 10:  # Only smooth if we have enough points
                summary.loc[mask, 'mean'] = uniform_filter1d(summary.loc[mask, 'mean'].values, size=10, mode='nearest')
                summary.loc[mask, 'std'] = uniform_filter1d(summary.loc[mask, 'std'].values, size=10, mode='nearest')
    
    # Create confidence intervals
    summary['ci'] = summary['std'] * stats.t.ppf((1 + 0.95) / 2, summary['count'] - 1) / np.sqrt(summary['count'])
    
    
    # Save plot
    suffix = "_smoothed" if smooth else ""
    suffix += "_limited40" if limited_to_40_demos else ""
    plt.savefig(f'plots_70b/icl_accuracy_curves_{num_classes}classes{suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as plots_70b/icl_accuracy_curves_{num_classes}classes{suffix}.pdf")


def main():
    # Path to consolidated CSV
    limited_to_40_demos = True
    num_classes = 5
    if num_classes == 3:
    csv_path_1b = "learning_curves_relabel_demos_3classes_1b/claude_multitask/llama3.1_1b_base/consolidated_metrics.csv"
        csv_path_7b = "learning_curves_relabel_demos_3classes_7b/claude_multitask/llama3.1_base/consolidated_metrics.csv"
        csv_path_70b = "learning_curves_relabel_demos_3classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    elif num_classes == 5:
        csv_path_1b = "learning_curves_relabel_demos_5classes_1b/claude_multitask/llama3.1_1b_base/consolidated_metrics.csv"
        csv_path_7b = "learning_curves_relabel_demos_5classes_7b/claude_multitask/llama3.1_base/consolidated_metrics.csv"
        csv_path_70b = "learning_curves_relabel_demos_5classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots_70b', exist_ok=True)
    
    try:
        results_df_1b = load_metrics_from_csv(csv_path_1b)
        results_df_7b = load_metrics_from_csv(csv_path_7b)
        results_df_70b = load_metrics_from_csv(csv_path_70b)
        
        plot_accuracy_curves(results_df, num_classes, smooth=False, limited_to_40_demos=limited_to_40_demos)
        
        # Create smoothed plots
        print("Creating smoothed plots...")
        plot_accuracy_curves(results_df, num_classes, smooth=True, limited_to_40_demos=limited_to_40_demos)
        plot_log_accuracy_curves(results_df, num_classes, smooth=True, limited_to_40_demos=limited_to_40_demos)
        
        print("\nAll plots generated successfully!")
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        print("Falling back to structure examination...")
        extract_accuracy_alternative(csv_path)

if __name__ == "__main__":
    main()
