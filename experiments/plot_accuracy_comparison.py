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

def plot_accuracy_curves_comparison(results_dfs, model_names, num_classes, smooth=False, limited_to_40_demos=False):
    """Create plot with 3 subplots side by side, one for each model size"""
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(108, 30))  # 3x wider for 3 subplots
    sns.set_style("whitegrid")
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Get all demo values for consistent x-axis across subplots
    all_demo_values = []
    for results_df in results_dfs:
        if limited_to_40_demos:
            filtered_df = results_df[((results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)) & (results_df['n_demo'] <= 40)]
        else:
            filtered_df = results_df[(results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)]
        all_demo_values.extend(filtered_df['n_demo'].unique())
    
    all_demo_values = sorted(set(all_demo_values))
    max_demo = max(all_demo_values)
    
    # Create evenly spaced ticks: 0, then every 10 from num_classes to max_demo
    ticks = []
    step = 10   # every 10 demos
    for i in range(0, max_demo + 1, step):
        ticks.append(i)
    if num_classes not in ticks:
        ticks.append(num_classes)
    
    # Create 10 colors for n_relabel values 10-100 (shared across all models)
    all_possible_relabels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_possible_relabels)))
    color_map = {relabel: colors[i] for i, relabel in enumerate(all_possible_relabels)}
    
    # Collect all legend information across all models
    all_legend_groups = {}
    
    # Plot data for each model in its own subplot
    for idx, (results_df, model_name) in enumerate(zip(results_dfs, model_names)):
        ax = axes[idx]
        
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
        
        # Get unique relabeling schemes for this model
        unique_relabels = sorted(summary['n_relabel'].unique())
        

        if num_classes == 3:
            # Define which curves are identical for each model
            identical_curves = {}
            if model_name == '1B':
                identical_curves = {
                    frozenset([80, 90, 100]): 100  # Use color of 100
                }
            elif model_name == '8B':
                identical_curves = {
                    frozenset([60, 70]): 70,    # Use color of 70
                    frozenset([80, 90]): 90     # Use color of 90
                }
            elif model_name == '70B':
                identical_curves = {
                    frozenset([40]): 50,        # Use color of 50
                    frozenset([70]): 100  # Use color of 100
                }
        else:
            identical_curves = {}
            if model_name == '70B':
                identical_curves = {
                    frozenset([70, 80]): 80,        # Use color of 50
                }
        
        # Identify duplicate curves and assign colors based on highest n_relabel
        curve_groups = {}
        group_colors = {}
        
        for n_relabel in unique_relabels:
            data = summary[summary['n_relabel'] == n_relabel]
            # Create a signature for this curve based on its y-values
            curve_signature = tuple(round(y, 6) for y in data['mean'].values)
            
            if curve_signature not in curve_groups:
                curve_groups[curve_signature] = []
                # Initially assign the color of the current n_relabel
                group_colors[curve_signature] = color_map[n_relabel]
            
            curve_groups[curve_signature].append(n_relabel)
        
        # Override colors for identical curves based on user specification
        for curve_signature, n_relabels in curve_groups.items():
            n_relabels_set = frozenset(n_relabels)
            
            # Check if this set of n_relabels matches any predefined identical curves
            for identical_set, highest_relabel in identical_curves.items():
                if n_relabels_set == identical_set:
                    group_colors[curve_signature] = color_map[highest_relabel]
                    break
            else:
                # If not a predefined identical set, use the color of the highest n_relabel in the group
                highest_in_group = max(n_relabels)
                group_colors[curve_signature] = color_map[highest_in_group]
        
        # Plot curves with grouped colors
        for curve_signature, n_relabels in curve_groups.items():
            color = group_colors[curve_signature]
            
            for n_relabel in n_relabels:
                data = summary[summary['n_relabel'] == n_relabel]
                ax.errorbar(data['n_demo'], data['mean'], 
                           yerr=data['ci'], 
                           marker='o',
                           color=color,
                           alpha=0.8,
                           markersize=12,
                           linewidth=5,
                           capsize=10,
                           capthick=4,
                           elinewidth=2)
            
            # Group labels for legend - add to global legend collection
            if len(n_relabels) > 1:
                legend_label = ', '.join(str(n) for n in sorted(n_relabels))
            else:
                legend_label = str(n_relabels[0])
            
            # Add to global legend groups (will automatically deduplicate)
            all_legend_groups[legend_label] = color
        
        # Set x-axis ticks for this subplot
        ax.set_xticks(ticks)
        
        # Set labels and title
        ax.set_xlabel('Number of Demonstrations', fontsize=128)
        if idx == 0:  # Only leftmost plot gets y-axis label
            ax.set_ylabel('Accuracy', fontsize=128)
        ax.set_title(f'{model_name}', fontsize=140)
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits for all subplots
        ax.set_ylim(0.1, 0.8)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=112)
        ax.tick_params(axis='both', which='minor', labelsize=96)
        
        # Hide y-axis tick labels for 7B and 70B models (idx 1 and 2)
        if idx > 0:
            ax.tick_params(axis='y', labelleft=False)
    
    # Create single legend below the entire figure showing all 10 colors
    from matplotlib.lines import Line2D
    # Create legend patches for all 10 possible n_relabel values
    legend_patches = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[relabel], markersize=60, label=str(relabel)) 
                     for relabel in all_possible_relabels]
    
    # Add single legend vertically on the right side, aligned with y-axis range
    fig.legend(handles=legend_patches, 
              bbox_to_anchor=(1.01, 0.53), 
              loc='center left', 
              ncol=1,  # All 10 items in one column
              fontsize=120,
              handletextpad=0.3,  # Space between dots and text
              handlelength=0.5,  # Make the handle line shorter
              borderaxespad=0.1)  # Reduce border padding
    
    plt.tight_layout()
    
    # Save plot
    suffix = "_smoothed" if smooth else ""
    suffix += "_limited40" if limited_to_40_demos else ""
    plt.savefig(f'plots/plots_comparison/accuracy_curves_subplots_{num_classes}classes{suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Subplot comparison plot saved as plots/plots_comparison/accuracy_curves_subplots_{num_classes}classes{suffix}.pdf")

def main():
    # Path to consolidated CSV files
    limited_to_40_demos = True
    num_classes = 5
    
    if num_classes == 3:
        csv_path_1b = "learning_curves/learning_curves_relabel_demos_3classes_1b/claude_multitask/llama3.1_1b_base/consolidated_metrics.csv"
        csv_path_7b = "learning_curves/learning_curves_relabel_demos_3classes_8b/claude_multitask/llama3.1_base/consolidated_metrics.csv"
        csv_path_70b = "learning_curves/learning_curves_relabel_demos_3classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    elif num_classes == 5:
        csv_path_1b = "learning_curves/learning_curves_relabel_demos_5classes_1b/claude_multitask/llama3.1_1b_base/consolidated_metrics.csv"
        csv_path_7b = "learning_curves/learning_curves_relabel_demos_5classes_8b/claude_multitask/llama3.1_base/consolidated_metrics.csv"
        csv_path_70b = "learning_curves/learning_curves_relabel_demos_5classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots/plots_comparison', exist_ok=True)
    
    try:
        # Load data for all models
        print("Loading 1B model data...")
        results_df_1b = load_metrics_from_csv(csv_path_1b)
        
        print("\nLoading 7B model data...")
        results_df_7b = load_metrics_from_csv(csv_path_7b)
        
        print("\nLoading 70B model data...")
        results_df_70b = load_metrics_from_csv(csv_path_70b)
        
        # Create comparison plots
        print("\nCreating comparison plots...")
        results_dfs = [results_df_1b, results_df_7b, results_df_70b]
        model_names = ['1B', '8B', '70B']
        
        plot_accuracy_curves_comparison(results_dfs, model_names, num_classes, smooth=False, limited_to_40_demos=limited_to_40_demos)
        
        # Create smoothed comparison plots
        print("Creating smoothed comparison plots...")
        plot_accuracy_curves_comparison(results_dfs, model_names, num_classes, smooth=True, limited_to_40_demos=limited_to_40_demos)
        
        print("\nAll comparison plots generated successfully!")
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
