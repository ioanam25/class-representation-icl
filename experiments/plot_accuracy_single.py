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

def plot_accuracy_curves_single(results_df, model_name, num_classes, smooth=False, output_dir="plots_single"):
    """Create plot for a single model using all available demos"""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(108, 50))
    sns.set_style("whitegrid")
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Filter data to include 0 demos and demos >= num_classes (no upper limit)
    filtered_df = results_df[(results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)].copy()
    
    # Get demo values for x-axis
    demo_values = sorted(filtered_df['n_demo'].unique())
    max_demo = max(demo_values)
    
    print(f"Demo values for {model_name}: {demo_values}")
    print(f"Max demo value: {max_demo}")
    
    # Create evenly spaced ticks: 0, then every 10 from num_classes to max_demo
    ticks = [0]
    step = 10   # every 10 demos
    for i in range(step, max_demo + 1, step):
        ticks.append(i)
    if num_classes not in ticks:
        ticks.append(num_classes)
    
    # Sort ticks
    ticks = sorted(set(ticks))
    
    # Create 10 colors for n_relabel values 10-100
    all_possible_relabels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_possible_relabels)))
    color_map = {relabel: colors[i] for i, relabel in enumerate(all_possible_relabels)}
    
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
    elif model_name == '70B' or model_name == 'llama3.1_70b_instruct':
        identical_curves = {
            frozenset([40]): 50,        # Use color of 50
            frozenset([70]): 100  # Use color of 100
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
    
    # Collect legend information
    legend_groups = {}
    
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
        
        # Group labels for legend
        if len(n_relabels) > 1:
            legend_label = ', '.join(str(n) for n in sorted(n_relabels))
        else:
            legend_label = str(n_relabels[0])
        
        legend_groups[legend_label] = color
    
    # Set x-axis ticks
    ax.set_xticks(ticks)
    
    # Set labels and title
    ax.set_xlabel('Number of Demonstrations', fontsize=128)
    ax.set_ylabel('Accuracy', fontsize=128)
    ax.set_title(f'{model_name} Model', fontsize=140)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(0.1, 1.0)  # Extended upper limit since we're not limited to 40 demos
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=112)
    ax.tick_params(axis='both', which='minor', labelsize=96)
    
    # Create legend showing all 10 colors
    from matplotlib.lines import Line2D
    # Create legend patches for all 10 possible n_relabel values
    legend_patches = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[relabel], markersize=60, label=str(relabel)) 
                     for relabel in all_possible_relabels]
    
    # Add legend on the right side
    ax.legend(handles=legend_patches, 
              bbox_to_anchor=(1.05, 0.5), 
              loc='center left', 
              ncol=1,  # All 10 items in one column
              fontsize=120,
              handletextpad=0.3,  # Space between dots and text
              handlelength=0.5,  # Make the handle line shorter
              borderaxespad=0.1)  # Reduce border padding
    
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(exist_ok=True)
    suffix = "_smoothed" if smooth else ""
    plot_filename = f'{output_dir}/accuracy_curves_{model_name.lower()}_{num_classes}classes{suffix}.pdf'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Single model plot saved as {plot_filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot accuracy curves from consolidated metrics.')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the consolidated_metrics.csv file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for plot title (e.g., "1B", "8B", "70B")')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes (3 or 5)')
    parser.add_argument('--output_dir', type=str, default='plots_single',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    model_name = args.model_name
    num_classes = args.num_classes
    
    print(f"Plotting {model_name} model with {num_classes} classes")
    print(f"Using CSV file: {csv_path}")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data for the specified model
        print(f"Loading {model_name} model data...")
        results_df = load_metrics_from_csv(csv_path)
        
        # Create single model plots
        print(f"\nCreating {model_name} accuracy curves...")
        plot_accuracy_curves_single(results_df, model_name, num_classes, smooth=False, output_dir=args.output_dir)
        
        # Create smoothed plots
        print(f"Creating smoothed {model_name} accuracy curves...")
        plot_accuracy_curves_single(results_df, model_name, num_classes, smooth=True, output_dir=args.output_dir)
        
        print(f"\n{model_name} plots generated successfully!")
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
