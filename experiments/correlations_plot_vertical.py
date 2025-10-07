import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_vertical_correlations(csv_path, model_name):
    """
    Load vertical correlations from CSV and compute average.
    """
    try:
        df = pd.read_csv(csv_path)
        
        print(f"{model_name} - Loaded {len(df)} vertical correlations")
        print(f"{model_name} - Correlation range: {df['correlation'].min():.4f} to {df['correlation'].max():.4f}")
        
        # filter df to only include n_demo values in n_demo_values
        # df = df[df['n_demo'].isin(n_demo_values)]
    
        # Compute average correlation 
        avg_correlation = df['correlation'].mean()
        print(f"{model_name} - Average correlation: {avg_correlation:.4f}")
        
        return avg_correlation, df['correlation'].values
        
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_bootstrap_vertical_data(summary_csv_path, raw_json_path, model_name):
    """
    Load bootstrap vertical correlation results from summary CSV and raw JSON files.
    
    Returns:
        dict with keys: n_demo, mean_correlation, std_correlation, ci_lower, ci_upper, raw_correlations
    """
    # Load summary statistics
    summary_df = pd.read_csv(summary_csv_path)
    
    # Load raw correlations for additional analysis if needed
    with open(raw_json_path, 'r') as f:
        raw_correlations = json.load(f)
    
    print(f"{model_name} - Loaded bootstrap vertical data with {len(summary_df)} demo points")
    print(f"{model_name} - Mean correlation range: {summary_df['mean_correlation'].min():.3f} to {summary_df['mean_correlation'].max():.3f}")
    print(f"{model_name} - Mean std: {summary_df['std_correlation'].mean():.3f}")
    
    return {
        'n_demo': summary_df['n_demo'].values,
        'mean_correlation': summary_df['mean_correlation'].values,
        'std_correlation': summary_df['std_correlation'].values,
        'ci_lower': summary_df['ci_2.5'].values,
        'ci_upper': summary_df['ci_97.5'].values,
        'median_correlation': summary_df['median_correlation'].values,
        'raw_correlations': raw_correlations
    }

def plot_bootstrap_vertical_correlations_comparison(bootstrap_paths, num_classes, output_dir="plots_correlations"):
    """
    Plot vertical correlations (n_demo vs correlation with 0-shot) using bootstrap results with error bars.
    
    Parameters:
    - bootstrap_paths: Dict with keys '1b', '8b', '70b', each containing 'summary' and 'raw' paths
    - num_classes: Number of classes
    - output_dir: Directory to save the plot
    """
    models = ['1b', '8b', '70b']
    model_names = ['1B', '8B', '70B']
    colors = ['lightseagreen', 'mediumslateblue', 'lightsalmon']
    
    # Load bootstrap data for all models
    bootstrap_data = {}
    for model in models:
        bootstrap_data[model] = load_bootstrap_vertical_data(
            bootstrap_paths[model]['summary'], 
            bootstrap_paths[model]['raw'], 
            model.upper() + " Model"
        )
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each model
    for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
        data = bootstrap_data[model]
        
        # For vertical correlations, x-axis is n_demo
        x = data['n_demo']
        y = data['mean_correlation']
        y_err_lower = data['mean_correlation'] - data['ci_lower']
        y_err_upper = data['ci_upper'] - data['mean_correlation']
        y_err = [y_err_lower, y_err_upper]
        
        # Sort by x-coordinate for smooth lines
        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]
        y_err_sorted = [y_err[0][sort_indices], y_err[1][sort_indices]]
        
        # Plot shaded confidence interval
        plt.fill_between(x_sorted, 
                        y_sorted - y_err_sorted[0], 
                        y_sorted + y_err_sorted[1], 
                        color=color, alpha=0.3, zorder=1)
        
        # Plot line connecting points
        plt.plot(x_sorted, y_sorted, color=color, alpha=0.8, linewidth=2, linestyle='-', zorder=3)
        
        # Plot points
        plt.scatter(x, y, color=color, label=model_name, 
                   s=80, alpha=0.9, zorder=5, edgecolors='white', linewidth=1)
    
    plt.xlabel('Number of Demonstrations', fontsize=28)
    plt.ylabel('Correlation with 0-shot (95% CI)', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(-0.3, 1.05)
    plt.legend(loc='lower right', fontsize=28)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_filename = f"bootstrap_vertical_correlations_{num_classes}classes.pdf"
    full_path = output_path / plot_filename
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Bootstrap vertical correlation plot saved to {full_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    for model, model_name in zip(models, model_names):
        data = bootstrap_data[model]
        print(f"\n{model_name} Model Bootstrap Summary:")
        print(f"  Number of demo points: {len(data['n_demo'])}")
        print(f"  Mean correlation: {np.mean(data['mean_correlation']):.4f}")
        print(f"  Mean bootstrap std: {np.mean(data['std_correlation']):.4f}")
        print(f"  Mean 95% CI width: {np.mean(data['ci_upper'] - data['ci_lower']):.4f}")

def plot_bootstrap_vertical_correlations_histogram(bootstrap_paths, num_classes, output_dir="plots_correlations"):
    """
    Create a histogram showing overall average vertical correlations with bootstrap confidence intervals.
    
    Parameters:
    - bootstrap_paths: Dict with keys '1b', '8b', '70b', each containing 'summary' and 'raw' paths
    - num_classes: Number of classes for labeling
    - output_dir: Directory to save the plot
    """
    models = ['1b', '8b', '70b']
    model_names = ['1B', '8B', '70B']
    colors = ['lightseagreen', 'mediumslateblue', 'lightsalmon']
    
    # Load bootstrap data for all models
    bootstrap_data = {}
    overall_averages = []
    overall_ci_lower = []
    overall_ci_upper = []
    
    for model in models:
        bootstrap_data[model] = load_bootstrap_vertical_data(
            bootstrap_paths[model]['summary'], 
            bootstrap_paths[model]['raw'], 
            model.upper() + " Model"
        )
        
        # Compute overall average across all n_demo values
        data = bootstrap_data[model]
        overall_avg = np.mean(data['mean_correlation'])
        overall_averages.append(overall_avg)
        
        # Compute overall confidence intervals (conservative approach)
        # Use the mean of the CI bounds across all n_demo values
        overall_ci_lower.append(overall_avg - np.mean(data['mean_correlation'] - data['ci_lower']))
        overall_ci_upper.append(overall_avg + np.mean(data['ci_upper'] - data['mean_correlation']))
    
    # Create the histogram
    plt.figure(figsize=(10, 8))
    
    # Create bar positions for x-axis
    x_pos = np.arange(len(model_names))
    bar_width = 0.6
    
    # Plot bars
    bars = plt.bar(x_pos, overall_averages, color=colors, alpha=0.7, width=bar_width)
    
    # Add shaded confidence intervals above each bar
    for i, (avg, ci_low, ci_up, color) in enumerate(zip(overall_averages, overall_ci_lower, overall_ci_upper, colors)):
        # Create a small rectangle for the confidence interval
        x_left = x_pos[i] - bar_width/4
        x_right = x_pos[i] + bar_width/4
        plt.fill_between([x_left, x_right], [ci_low, ci_low], [ci_up, ci_up], 
                        color=color, alpha=0.3, zorder=3)
    
    # Customize the plot
    plt.xlabel('Model Size', fontsize=28)
    plt.ylabel('Average Correlation (95% CI)', fontsize=28)
    plt.xticks(x_pos, model_names, fontsize=28)
    plt.yticks(fontsize=28)
    
    # Set y-axis limits to show the data clearly
    plt.ylim(0.3, 1.0)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_filename = f"bootstrap_vertical_correlations_histogram_{num_classes}classes.pdf"
    full_path = output_path / plot_filename
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Bootstrap vertical correlations histogram saved to {full_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print(f"\nBootstrap Summary Statistics:")
    for model, model_name, avg, ci_low, ci_up in zip(models, model_names, overall_averages, overall_ci_lower, overall_ci_upper):
        data = bootstrap_data[model]
        print(f"\n{model_name} Model:")
        print(f"  Overall average correlation: {avg:.4f}")
        print(f"  95% CI: [{ci_low:.4f}, {ci_up:.4f}]")
        print(f"  Number of demo points: {len(data['n_demo'])}")
        print(f"  Mean bootstrap std across demos: {np.mean(data['std_correlation']):.4f}")
    
    print(f"\nDifference (70B - 1B): {overall_averages[2] - overall_averages[0]:.4f}")

def plot_vertical_correlations_histogram(csv_path_1b, csv_path_7b, csv_path_70b, num_classes, output_dir="plots_correlations"):
    """
    Create a histogram comparing average vertical correlations between 1B and 7B models.
    
    Parameters:
    - csv_path_1b: Path to the 1B model vertical correlations CSV file
    - csv_path_7b: Path to the 7B model vertical correlations CSV file
    - num_classes: Number of classes for labeling
    - output_dir: Directory to save the plot
    """
    
    # Load data for both models
    avg_1b, correlations_1b = load_vertical_correlations(csv_path_1b, "1B Model")
    avg_7b, correlations_7b = load_vertical_correlations(csv_path_7b, "8B Model")
    avg_70b, correlations_70b = load_vertical_correlations(csv_path_70b, "70B Model")
    
    if avg_1b is None or avg_7b is None or avg_70b is None:
        print("Error: Could not load correlation data for one or more models")
        return
    
    # Prepare data for histogram
    models = ['1B', '8B', '70B']
    averages = [avg_1b, avg_7b, avg_70b]
    colors = ['lightseagreen', 'mediumslateblue', 'lightsalmon']
    
    # Create the histogram
    plt.figure(figsize=(10, 8))
    
    # Create bar positions for x-axis
    x_pos = np.arange(len(models))
    bar_width = 0.6
    
    # Plot bars
    bars = plt.bar(x_pos, averages, color=colors, alpha=0.7, width=bar_width)
    
    # Calculate error bars (standard deviation)
    std_1b = np.std(correlations_1b) if correlations_1b is not None else 0
    std_7b = np.std(correlations_7b) if correlations_7b is not None else 0
    std_70b = np.std(correlations_70b) if correlations_70b is not None else 0
    stds = [std_1b, std_7b, std_70b]
    
    # Add shaded confidence intervals above each bar
    for i, (avg, std, color) in enumerate(zip(averages, stds, colors)):
        # Create a small rectangle for the confidence interval
        x_left = x_pos[i] - bar_width/4
        x_right = x_pos[i] + bar_width/4
        ci_low = avg - std
        ci_up = avg + std
        plt.fill_between([x_left, x_right], [ci_low, ci_low], [ci_up, ci_up], 
                        color=color, alpha=0.3, zorder=3)
    
    # Customize the plot
    plt.xlabel('Model Size', fontsize=28)
    plt.ylabel('Correlation', fontsize=28)
    # plt.title(f'Average Vertical Correlation by Model Size\n({num_classes} classes, n_demo=[10,20,30,40])', fontsize=32)
    plt.xticks(x_pos, models, fontsize=28)
    plt.yticks(fontsize=28)
    # plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits to show the data clearly
    y_min = min(averages) - max(stds) - 0.05
    # y_max = max(averages) + max(stds) + 0.1
    plt.ylim(0.3, 1)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_filename = f"vertical_correlations_histogram_{num_classes}classes.pdf"
    full_path = output_path / plot_filename
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Vertical correlations histogram saved to {full_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"1B Model:")
    print(f"  Average vertical correlation: {avg_1b:.4f}")
    print(f"  Standard deviation: {std_1b:.4f}")
    print(f"  Number of correlations: {len(correlations_1b)}")
    
    print(f"\n8B Model:")
    print(f"  Average vertical correlation: {avg_7b:.4f}")
    print(f"  Standard deviation: {std_7b:.4f}")
    print(f"  Number of correlations: {len(correlations_7b)}")
    
    print(f"\n70B Model:")
    print(f"  Average vertical correlation: {avg_70b:.4f}")
    print(f"  Standard deviation: {std_70b:.4f}")
    print(f"  Number of correlations: {len(correlations_70b)}")
    
    print(f"\nDifference (70B - 1B): {avg_70b - avg_1b:.4f}")
    
    # Simple statistical test (difference relative to pooled standard error)
    pooled_se = np.sqrt((std_1b**2 / len(correlations_1b)) + (std_7b**2 / len(correlations_7b)) + (std_70b**2 / len(correlations_70b)))
    if pooled_se > 0:
        t_stat = (avg_70b - avg_1b) / pooled_se
        print(f"Approximate t-statistic: {t_stat:.2f}")

def main():
    """
    Main function to create vertical correlations plots.
    """
    num_classes = 5  # Changed to 3 to match your bootstrap results
    use_bootstrap = True  # Set to False to use original method
    
    if use_bootstrap:
        # Paths to bootstrap results
        bootstrap_paths = {
            '1b': {
                'summary': f"bootstrap_results/1b_vertical_bootstrap_{num_classes}classes_limited40_spearman_summary.csv",
                'raw': f"bootstrap_results/1b_vertical_bootstrap_{num_classes}classes_limited40_spearman_raw_correlations.json"
            },
            '8b': {
                'summary': f"bootstrap_results/8b_vertical_bootstrap_{num_classes}classes_limited40_spearman_summary.csv",
                'raw': f"bootstrap_results/8b_vertical_bootstrap_{num_classes}classes_limited40_spearman_raw_correlations.json"
            },
            '70b': {
                'summary': f"bootstrap_results/70b_vertical_bootstrap_{num_classes}classes_limited40_spearman_summary.csv",
                'raw': f"bootstrap_results/70b_vertical_bootstrap_{num_classes}classes_limited40_spearman_raw_correlations.json"
            }
        }
        
        print("Creating bootstrap vertical correlation plots...")
        print("Bootstrap files:")
        for model in ['1b', '8b', '70b']:
            print(f"  {model.upper()}: {bootstrap_paths[model]['summary']}")
        
        # Check if bootstrap files exist
        missing_files = []
        for model in ['1b', '8b', '70b']:
            if not Path(bootstrap_paths[model]['summary']).exists():
                missing_files.append(bootstrap_paths[model]['summary'])
            if not Path(bootstrap_paths[model]['raw']).exists():
                missing_files.append(bootstrap_paths[model]['raw'])
        
        if missing_files:
            print("Error: Missing bootstrap files:")
            for file in missing_files:
                print(f"  {file}")
            return
        
        # Create bootstrap correlation plot (line plot with error bars)
        try:
            print("\n" + "="*50)
            print("CREATING BOOTSTRAP VERTICAL CORRELATION LINE PLOT")
            print("="*50)
            plot_bootstrap_vertical_correlations_comparison(bootstrap_paths, num_classes, output_dir="plots_correlations")
            
        except Exception as e:
            print(f"Error creating bootstrap line plot: {e}")
            import traceback
            traceback.print_exc()
        
        # Create bootstrap histogram (bar plot with error bars)
        try:
            print("\n" + "="*50)
            print("CREATING BOOTSTRAP VERTICAL CORRELATION HISTOGRAM")
            print("="*50)
            plot_bootstrap_vertical_correlations_histogram(bootstrap_paths, num_classes, output_dir="plots_correlations")
            
        except Exception as e:
            print(f"Error creating bootstrap histogram: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Original method without bootstrap
        num_classes = 5  # You can change this back to 5 if you have those files
        csv_path_1b = f"plots_comparison/1b_vertical_correlations_{num_classes}classes_limited40_spearman.csv"
        csv_path_7b = f"plots_comparison/8b_vertical_correlations_{num_classes}classes_limited40_spearman.csv"
        csv_path_70b = f"plots_comparison/70b_vertical_correlations_{num_classes}classes_limited40_spearman.csv"
        
        print("Creating original vertical correlations histogram...")
        print(f"Using 1B CSV file: {csv_path_1b}")
        print(f"Using 7B CSV file: {csv_path_7b}")
        print(f"Using 70B CSV file: {csv_path_70b}")
        
        # Check if files exist
        if not Path(csv_path_1b).exists():
            print(f"Error: File {csv_path_1b} not found!")
            return
        
        if not Path(csv_path_7b).exists():
            print(f"Error: File {csv_path_7b} not found!")
            return
        
        if not Path(csv_path_70b).exists():
            print(f"Error: File {csv_path_70b} not found!")
            return
        
        # Create original histogram
        try:
            print("\n" + "="*50)
            print("CREATING ORIGINAL VERTICAL CORRELATIONS HISTOGRAM")
            print("="*50)
            plot_vertical_correlations_histogram(csv_path_1b, csv_path_7b, csv_path_70b, num_classes, output_dir="plots_correlations")
            
        except Exception as e:
            print(f"Error creating original histogram: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
