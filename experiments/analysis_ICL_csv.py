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
    
    # Create plot
    plt.figure(figsize=(36, 30))
    sns.set_style("whitegrid")
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Plot a line for each relabeling scheme
    unique_relabels = sorted(summary['n_relabel'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    
    # Identify duplicate curves and assign same colors
    curve_groups = {}
    group_colors = {}
    next_color_idx = 0
    
    for n_relabel in unique_relabels:
        data = summary[summary['n_relabel'] == n_relabel]
        # Create a signature for this curve based on its y-values
        curve_signature = tuple(round(y, 6) for y in data['mean'].values)
        
        if curve_signature not in curve_groups:
            curve_groups[curve_signature] = []
            group_colors[curve_signature] = colors[next_color_idx]
            next_color_idx += 1
        
        curve_groups[curve_signature].append(n_relabel)
    
    # Plot curves with grouped colors
    legend_groups = {}
    
    for curve_signature, n_relabels in curve_groups.items():
        color = group_colors[curve_signature]
        
        for n_relabel in n_relabels:
            data = summary[summary['n_relabel'] == n_relabel]
            plt.errorbar(data['n_demo'], data['mean'], 
                        yerr=data['ci'], 
                        marker='o',
                        color=color,
                        alpha=0.8,
                        markersize=4,
                        capsize=3,
                        capthick=1)
        
        # Group labels for legend
        if len(n_relabels) > 1:
            legend_label = ', '.join(str(n) for n in sorted(n_relabels))
        else:
            legend_label = str(n_relabels[0])
        
        legend_groups[legend_label] = color

    # Set reasonable x-axis range and ticks
    all_demo_values = sorted(filtered_df['n_demo'].unique())
    max_demo = max(all_demo_values)
    
    # Create evenly spaced ticks: 0, then every 10 from num_classes to max_demo
    ticks = []
    print(max_demo)
    step = 10   # every 10 demos
    for i in range(0, max_demo + 1, step):
        ticks.append(i)
    print(ticks)
    if num_classes not in ticks:
        ticks.append(num_classes)
    
    print(ticks)
    plt.xticks(ticks)
    
    plt.xlabel('Number of Demonstrations', fontsize=128)
    plt.ylabel('Accuracy', fontsize=128)
    # Title removed per user request
    plt.grid(True, alpha=0.3)
    
    # Create custom legend below the plot using grouped colors
    from matplotlib.lines import Line2D
    legend_patches = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=50, label=label) for label, color in legend_groups.items()]
    
    # Add legend below the plot
    plt.legend(handles=legend_patches, 
              bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', 
              ncol=len(legend_patches),  # All in one row
              fontsize=100,
              columnspacing=0.2,  # Reduce spacing between legend items
              handletextpad=0.0001,  # Reduce spacing between dots and text
              handlelength=0.5,  # Make the handle line shorter
              borderaxespad=0.3)  # Reduce border padding
    
    # Increase tick label sizes much bigger
    plt.tick_params(axis='both', which='major', labelsize=112)
    plt.tick_params(axis='both', which='minor', labelsize=96)
    
    plt.tight_layout()
    
    # Save plot
    suffix = "_smoothed" if smooth else ""
    suffix += "_limited40" if limited_to_40_demos else ""
    plt.savefig(f'plots_70b/icl_accuracy_curves_{num_classes}classes{suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as plots_70b/icl_accuracy_curves_{num_classes}classes{suffix}.pdf")


def plot_log_accuracy_curves(results_df, num_classes, smooth=False, limited_to_40_demos=False):
    """Create plot with log accuracy curves for different relabeling schemes"""
    # Filter data to include 0 demos and demos >= num_classes
    if limited_to_40_demos:
        filtered_df = results_df[((results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)) & (results_df['n_demo'] <= 40)].copy()
    else:
        filtered_df = results_df[(results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)].copy()
    
    # Calculate mean accuracy for each n_demo and n_relabel
    summary = filtered_df.groupby(['n_demo', 'n_relabel'])['accuracy'].agg(['mean']).reset_index()
    
    # Apply smoothing if requested
    if smooth:
        from scipy.ndimage import uniform_filter1d
        for n_relabel in summary['n_relabel'].unique():
            mask = summary['n_relabel'] == n_relabel
            if mask.sum() >= 10:  # Only smooth if we have enough points
                summary.loc[mask, 'mean'] = uniform_filter1d(summary.loc[mask, 'mean'].values, size=10, mode='nearest')
    
    # Create plot
    plt.figure(figsize=(36, 30))
    sns.set_style("whitegrid")
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Plot a line for each relabeling scheme
    unique_relabels = sorted(summary['n_relabel'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    
    # Identify duplicate curves and assign same colors
    curve_groups = {}
    group_colors = {}
    next_color_idx = 0
    
    for n_relabel in unique_relabels:
        data = summary[summary['n_relabel'] == n_relabel]
        # Create a signature for this curve based on its y-values
        curve_signature = tuple(round(y, 6) for y in data['mean'].values)
        
        if curve_signature not in curve_groups:
            curve_groups[curve_signature] = []
            group_colors[curve_signature] = colors[next_color_idx]
            next_color_idx += 1
        
        curve_groups[curve_signature].append(n_relabel)
    
    # Plot curves with grouped colors
    legend_groups = {}
    
    for curve_signature, n_relabels in curve_groups.items():
        color = group_colors[curve_signature]
        
        for n_relabel in n_relabels:
            data = summary[summary['n_relabel'] == n_relabel]
            # Handle 0 demos case for log plot - add small epsilon to avoid log(0)
            x_values = data['n_demo'].copy()
            x_values[x_values == 0] = 0.1  # Use 0.1 instead of 0 for log scale
            
            plt.plot(x_values, np.log(data['mean']), 
                    marker='o',
                    color=color,
                    alpha=0.8,
                    markersize=4)
        
        # Group labels for legend
        if len(n_relabels) > 1:
            legend_label = ', '.join(str(n) for n in sorted(n_relabels))
        else:
            legend_label = str(n_relabels[0])
        
        legend_groups[legend_label] = color
    
    plt.xscale('log')
    
    # # Create labels (0.1 -> "0", others stay the same)
    # labels = [str(int(t)) if t != 0.1 else "0" for t in ticks]
    # plt.xticks(ticks, labels)
    
    plt.xlabel('Number of Demonstrations (log scale)', fontsize=128)
    plt.ylabel('Log Accuracy', fontsize=128)
    # Title removed per user request
    plt.grid(True, alpha=0.3)
    
    # Create custom legend below the plot using grouped colors
    from matplotlib.lines import Line2D
    legend_patches = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=15, label=label) for label, color in legend_groups.items()]
    
    # Add legend below the plot
    plt.legend(handles=legend_patches, 
              bbox_to_anchor=(0.5, -0.08), 
              loc='upper center', 
              ncol=len(legend_patches),  # All in one row
              fontsize=60,
              columnspacing=0.5)  # Reduce spacing between legend items
    
    # Increase tick label sizes much bigger
    plt.tick_params(axis='both', which='major', labelsize=112)
    plt.tick_params(axis='both', which='minor', labelsize=96)
    
    plt.tight_layout()
    
    # Save plot
    suffix = "_smoothed" if smooth else ""
    suffix += "_limited40" if limited_to_40_demos else ""
    plt.savefig(f'plots_70b/icl_log_accuracy_curves_{num_classes}classes{suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Log accuracy plot saved as plots_70b/icl_log_accuracy_curves_{num_classes}classes{suffix}.pdf")

def print_detailed_stats(results_df):
    """Print detailed statistics about the results"""
    print("\nDetailed Statistics:")
    
    # Overall stats
    print("\nOverall Statistics:")
    print(f"Total number of data points: {len(results_df)}")
    print(f"Number of unique relabeling schemes: {len(results_df['n_relabel'].unique())}")
    print(f"Number of demonstration sizes: {len(results_df['n_demo'].unique())}")
    print(f"Range of demonstrations: {min(results_df['n_demo'])} to {max(results_df['n_demo'])}")
    
    # Best performing configurations
    print("\nBest Performing Configurations:")
    best_configs = results_df.groupby(['n_relabel', 'n_demo'])['accuracy'].mean().reset_index()
    best_overall = best_configs.loc[best_configs['accuracy'].idxmax()]
    print(f"\nBest overall configuration:")
    print(f"  Relabeling scheme: {best_overall['n_relabel']} examples")
    print(f"  Number of demonstrations: {best_overall['n_demo']}")
    print(f"  Average accuracy: {best_overall['accuracy']:.3f}")
    
    # Best for each relabeling scheme
    print("\nBest number of demonstrations for each relabeling scheme:")
    for n_relabel in sorted(results_df['n_relabel'].unique()):
        scheme_data = best_configs[best_configs['n_relabel'] == n_relabel]
        best_scheme = scheme_data.loc[scheme_data['accuracy'].idxmax()]
        mean_acc = results_df[results_df['n_relabel'] == n_relabel]['accuracy'].mean()
        std_acc = results_df[results_df['n_relabel'] == n_relabel]['accuracy'].std()
        print(f"\nRelabeling with {n_relabel} examples:")
        print(f"  Best n_demo: {best_scheme['n_demo']}")
        print(f"  Best accuracy: {best_scheme['accuracy']:.3f}")
        print(f"  Average accuracy across all n_demo: {mean_acc:.3f} ± {std_acc:.3f}")

def main():
    # Path to consolidated CSV
    limited_to_40_demos = True
    num_classes = 3
    if num_classes == 3:
        csv_path = "learning_curves_relabel_demos_3classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    elif num_classes == 5:
        csv_path = "learning_curves_relabel_demos_5classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots_70b', exist_ok=True)
    
    try:
        # Try to load metrics from CSV
        results_df = load_metrics_from_csv(csv_path)
        
        # Print detailed statistics
        print_detailed_stats(results_df)
        
        # Create plots
        print("\nCreating plots...")
        plot_accuracy_curves(results_df, num_classes, smooth=False, limited_to_40_demos=limited_to_40_demos)
        plot_log_accuracy_curves(results_df, num_classes, smooth=False, limited_to_40_demos=limited_to_40_demos)
        
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
