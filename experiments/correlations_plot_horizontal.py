import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import json
from pathlib import Path

def extract_zero_shot_accuracy(accuracy_string):
    """
    Extract the first element (0-shot accuracy) from the accuracy array string.
    """
    try:
        # Parse the string representation of the list
        accuracy_list = ast.literal_eval(accuracy_string)
        # Return the first element (0-shot accuracy)
        return accuracy_list[0]
    except (ValueError, SyntaxError, IndexError):
        print(f"Error parsing accuracy string: {accuracy_string}")
        return None

def load_bootstrap_data(summary_csv_path, raw_json_path, model_name):
    """
    Load bootstrap results from summary CSV and raw JSON files.
    
    Returns:
        dict with keys: n_relabel, zero_shot_accuracy, mean_correlation, 
                       std_correlation, ci_lower, ci_upper, raw_correlations
    """
    # Load summary statistics
    summary_df = pd.read_csv(summary_csv_path)
    
    # Load raw correlations for additional analysis if needed
    with open(raw_json_path, 'r') as f:
        raw_correlations = json.load(f)
    
    print(f"{model_name} - Loaded bootstrap data with {len(summary_df)} relabel schemes")
    print(f"{model_name} - Mean correlation range: {summary_df['mean_correlation'].min():.3f} to {summary_df['mean_correlation'].max():.3f}")
    print(f"{model_name} - Mean std: {summary_df['std_correlation'].mean():.3f}")
    
    # For horizontal correlations, we need to get the 0-shot accuracy for each n_relabel
    # This requires loading the original data or having it stored in the bootstrap results
    # For now, we'll use a placeholder approach - you may need to modify this based on your data structure
    
    return {
        'n_relabel': summary_df['n_relabel'].values,
        'mean_correlation': summary_df['mean_correlation'].values,
        'std_correlation': summary_df['std_correlation'].values,
        'ci_lower': summary_df['ci_2.5'].values,
        'ci_upper': summary_df['ci_97.5'].values,
        'median_correlation': summary_df['median_correlation'].values,
        'raw_correlations': raw_correlations
    }

def get_zero_shot_accuracies_for_relabels(original_csv_path, n_relabel_values):
    """
    Get 0-shot accuracies for each n_relabel from the original correlation CSV.
    This is needed to create the x-axis for the bootstrap plot.
    """
    # Load original correlation data
    df = pd.read_csv(original_csv_path)
    
    # Extract 0-shot accuracies
    df['zero_shot_accuracy'] = df['accuracies'].apply(extract_zero_shot_accuracy)
    df = df.dropna(subset=['zero_shot_accuracy'])
    
    # Create mapping from n_relabel to zero_shot_accuracy
    relabel_to_zero_shot = {}
    for _, row in df.iterrows():
        relabel_to_zero_shot[int(row['n_relabel'])] = row['zero_shot_accuracy']
    
    # Return zero-shot accuracies in the same order as n_relabel_values
    zero_shot_accs = []
    for n_rel in n_relabel_values:
        if int(n_rel) in relabel_to_zero_shot:
            zero_shot_accs.append(relabel_to_zero_shot[int(n_rel)])
        else:
            print(f"Warning: No 0-shot accuracy found for n_relabel={n_rel}")
            zero_shot_accs.append(None)
    
    return np.array(zero_shot_accs)

def process_data_for_plotting(csv_path, model_name):
    """
    Process data from CSV and return unique points with labels.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Extract 0-shot accuracies
    df['zero_shot_accuracy'] = df['accuracies'].apply(extract_zero_shot_accuracy)
    
    # Remove any rows where extraction failed
    df = df.dropna(subset=['zero_shot_accuracy'])
    
    print(f"{model_name} - Loaded {len(df)} data points")
    print(f"{model_name} - 0-shot accuracy range: {df['zero_shot_accuracy'].min():.3f} to {df['zero_shot_accuracy'].max():.3f}")
    print(f"{model_name} - Correlation range: {df['correlation'].min():.3f} to {df['correlation'].max():.3f}")
    
    # Group by unique (zero_shot_accuracy, correlation) pairs and combine n_relabel values
    unique_points = {}
    
    for _, row in df.iterrows():
        key = (row['zero_shot_accuracy'], row['correlation'])
        if key not in unique_points:
            unique_points[key] = []
        unique_points[key].append(int(row['n_relabel']))
    
    # Create lists for plotting
    x_coords = []
    y_coords = []
    labels = []
    
    for (x, y), n_relabels in unique_points.items():
        x_coords.append(x)
        y_coords.append(y)
        # Sort n_relabel values and join with equals sign
        sorted_relabels = sorted(n_relabels)
        labels.append(', '.join(map(str, sorted_relabels)))
    
    return x_coords, y_coords, labels, unique_points

def plot_bootstrap_correlations_comparison(bootstrap_paths, original_csv_paths, num_classes, output_dir="plots_correlations"):
    """
    Plot correlations vs 0-shot accuracies using bootstrap results with error bars.
    
    Parameters:
    - bootstrap_paths: Dict with keys '1b', '8b', '70b', each containing 'summary' and 'raw' paths
    - original_csv_paths: Dict with keys '1b', '8b', '70b' pointing to original CSV files for 0-shot accuracies
    - num_classes: Number of classes
    - output_dir: Directory to save the plot
    """
    models = ['1b', '8b', '70b']
    model_names = ['1B', '8B', '70B']
    colors = ['lightseagreen', 'mediumslateblue', 'lightsalmon']
    
    # Load bootstrap data for all models
    bootstrap_data = {}
    for model in models:
        bootstrap_data[model] = load_bootstrap_data(
            bootstrap_paths[model]['summary'], 
            bootstrap_paths[model]['raw'], 
            model.upper() + " Model"
        )
        
        # Get 0-shot accuracies for each n_relabel
        zero_shot_accs = get_zero_shot_accuracies_for_relabels(
            original_csv_paths[model], 
            bootstrap_data[model]['n_relabel']
        )
        bootstrap_data[model]['zero_shot_accuracy'] = zero_shot_accs
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each model
    for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
        data = bootstrap_data[model]
        
        # Filter out any None values in zero_shot_accuracy
        valid_mask = data['zero_shot_accuracy'] != None
        if not np.any(valid_mask):
            print(f"Warning: No valid 0-shot accuracies for {model_name}")
            continue
            
        x = data['zero_shot_accuracy'][valid_mask]
        y = data['mean_correlation'][valid_mask]
        y_err_lower = data['mean_correlation'][valid_mask] - data['ci_lower'][valid_mask]
        y_err_upper = data['ci_upper'][valid_mask] - data['mean_correlation'][valid_mask]
        y_err = [y_err_lower, y_err_upper]
        
        # Sort by x-coordinate for smooth lines
        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]
        y_err_sorted = [y_err[0][sort_indices], y_err[1][sort_indices]]
        
        # Plot line connecting points
        plt.plot(x_sorted, y_sorted, color=color, alpha=0.7, linewidth=2, linestyle='-')
        
        # Plot points with error bars
        plt.errorbar(x, y, yerr=y_err, fmt='o', color=color, label=model_name, 
                    capsize=5, capthick=2, markersize=8, alpha=0.9, zorder=5)
    
    plt.xlabel('0-shot accuracy', fontsize=28)
    plt.ylabel('Correlation (with 95% CI)', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(-0.7, 1.05)
    # plt.xlim(0.15, 0.9)
    plt.legend(loc='lower right', fontsize=28)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_filename = f"bootstrap_horizontal_correlations_{num_classes}classes.pdf"
    full_path = output_path / plot_filename
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Bootstrap correlation plot saved to {full_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    for model, model_name in zip(models, model_names):
        data = bootstrap_data[model]
        valid_mask = data['zero_shot_accuracy'] != None
        if np.any(valid_mask):
            print(f"\n{model_name} Model Bootstrap Summary:")
            print(f"  Number of relabel schemes: {np.sum(valid_mask)}")
            print(f"  Mean 0-shot accuracy: {np.mean(data['zero_shot_accuracy'][valid_mask]):.4f}")
            print(f"  Mean correlation: {np.mean(data['mean_correlation'][valid_mask]):.4f}")
            print(f"  Mean bootstrap std: {np.mean(data['std_correlation'][valid_mask]):.4f}")
            print(f"  Mean 95% CI width: {np.mean(data['ci_upper'][valid_mask] - data['ci_lower'][valid_mask]):.4f}")

def plot_correlations_comparison(csv_path_1b, csv_path_7b, csv_path_70b, num_classes, output_dir="plots_correlations"):
    """
    Plot correlations vs 0-shot accuracies for 1B, 8B, and 70B models on the same plot.
    Shows only unique points with combined labels for duplicates.
    
    Parameters:
    - csv_path_1b: Path to the 1B model horizontal correlations CSV file
    - csv_path_7b: Path to the 8B model horizontal correlations CSV file
    - csv_path_70b: Path to the 70B model horizontal correlations CSV file
    - output_dir: Directory to save the plot
    """
    # Process data for all three models
    x_1b, y_1b, labels_1b, unique_points_1b = process_data_for_plotting(csv_path_1b, "1B Model")
    x_7b, y_7b, labels_7b, unique_points_7b = process_data_for_plotting(csv_path_7b, "8B Model")
    x_70b, y_70b, labels_70b, unique_points_70b = process_data_for_plotting(csv_path_70b, "70B Model")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Sort data points by x-coordinate for smooth lines
    def sort_data_for_line(x_coords, y_coords, labels):
        """Sort data points by x-coordinate for smooth line plotting."""
        sorted_indices = np.argsort(x_coords)
        return (np.array(x_coords)[sorted_indices], 
                np.array(y_coords)[sorted_indices], 
                [labels[i] for i in sorted_indices])
    
    x_1b_sorted, y_1b_sorted, labels_1b_sorted = sort_data_for_line(x_1b, y_1b, labels_1b)
    x_7b_sorted, y_7b_sorted, labels_7b_sorted = sort_data_for_line(x_7b, y_7b, labels_7b)
    x_70b_sorted, y_70b_sorted, labels_70b_sorted = sort_data_for_line(x_70b, y_70b, labels_70b)
    
    # Plot lines first (so they appear behind the points)
    plt.plot(x_1b_sorted, y_1b_sorted, color='lightseagreen', alpha=0.7, linewidth=2, linestyle='-')
    plt.plot(x_7b_sorted, y_7b_sorted, color='mediumslateblue', alpha=0.7, linewidth=2, linestyle='-')
    plt.plot(x_70b_sorted, y_70b_sorted, color='lightsalmon', alpha=0.7, linewidth=2, linestyle='-')
    
    # Plot scatter points on top of lines
    scatter_1b = plt.scatter(x_1b, y_1b, alpha=0.9, s=80, color='lightseagreen', label='1B', zorder=5)
    scatter_7b = plt.scatter(x_7b, y_7b, alpha=0.9, s=80, color='mediumslateblue', label='8B', zorder=5)
    scatter_70b = plt.scatter(x_70b, y_70b, alpha=0.9, s=80, color='lightsalmon', label='70B', zorder=5)
    
    # Add labels for 1B model points with overlap detection
    def add_labels_with_overlap_detection(x_coords, y_coords, labels, color, threshold=0.02):
        """Add labels with automatic separation for overlapping points."""
        positions_used = []
        
        for i, (x, y, label) in enumerate(zip(x_coords, y_coords, labels)):
            # Check for nearby points
            offset_x = 0   # Default offset centered
            offset_y = 10  # Default offset above
            ha_align = 'left'  # Default horizontal alignment
            
            # Check if this position is too close to any previously used position
            for prev_x, prev_y, prev_offset_x, prev_offset_y in positions_used:
                if abs(x - prev_x) < threshold and abs(y - prev_y) < threshold:
                    # Points are close, alternate the label position left/right
                    if prev_offset_x <= 0:
                        offset_x = -10  # Position to the left
                        ha_align = 'left'
                    else:
                        offset_x = 15   # Position to the right
                        ha_align = 'center'
                    break
            
            # plt.annotate(label, (x, y), xytext=(offset_x, offset_y), textcoords='offset points', 
            #             fontsize=8, alpha=0.8, color=color, ha=ha_align)
            
            positions_used.append((x, y, offset_x, offset_y))
    
    # Add labels for all three models
    add_labels_with_overlap_detection(x_1b, y_1b, labels_1b, 'lightseagreen')
    add_labels_with_overlap_detection(x_7b, y_7b, labels_7b, 'mediumslateblue')
    add_labels_with_overlap_detection(x_70b, y_70b, labels_70b, 'lightsalmon')
    
    plt.xlabel('0-shot Accuracy', fontsize=28)
    plt.ylabel('Correlation', fontsize=28)
    # plt.title(f'Correlation vs 0-shot Accuracy Comparison\n1B: {len(unique_points_1b)} unique points, 8B: {len(unique_points_7b)} unique points, 70B: {len(unique_points_70b)} unique points', fontsize=14)
    # plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(-0.35, 1.05)
    plt.xlim(0.15, 0.9)
    plt.legend(loc='lower right', fontsize=28)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plot_filename = f"horizontal_correlations_{num_classes}classes.pdf"
    full_path = output_path / plot_filename
    
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {full_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print(f"\n1B Model Summary:")
    print(f"  Number of unique points: {len(unique_points_1b)}")
    print(f"  Mean 0-shot accuracy: {np.mean(x_1b):.4f}")
    print(f"  Mean correlation: {np.mean(y_1b):.4f}")
    
    print(f"\n8B Model Summary:")
    print(f"  Number of unique points: {len(unique_points_7b)}")
    print(f"  Mean 0-shot accuracy: {np.mean(x_7b):.4f}")
    print(f"  Mean correlation: {np.mean(y_7b):.4f}")
    
    print(f"\n70B Model Summary:")
    print(f"  Number of unique points: {len(unique_points_70b)}")
    print(f"  Mean 0-shot accuracy: {np.mean(x_70b):.4f}")
    print(f"  Mean correlation: {np.mean(y_70b):.4f}")
    
    # Print duplicate information for all models
    print(f"\n1B Model Duplicate information:")
    for (x, y), n_relabels in unique_points_1b.items():
        if len(n_relabels) > 1:
            print(f"  Point ({x:.3f}, {y:.3f}): n_relabel values {sorted(n_relabels)}")
    
    print(f"\n8B Model Duplicate information:")
    for (x, y), n_relabels in unique_points_7b.items():
        if len(n_relabels) > 1:
            print(f"  Point ({x:.3f}, {y:.3f}): n_relabel values {sorted(n_relabels)}")
    
    print(f"\n70B Model Duplicate information:")
    for (x, y), n_relabels in unique_points_70b.items():
        if len(n_relabels) > 1:
            print(f"  Point ({x:.3f}, {y:.3f}): n_relabel values {sorted(n_relabels)}")

def main():
    """
    Main function to create correlation comparison plot.
    """
    num_classes = 3  # Changed to 3 to match your bootstrap results
    use_bootstrap = True  # Set to False to use original method
    
    if use_bootstrap:
        # Paths to bootstrap results
        bootstrap_paths = {
            '1b': {
                'summary': f"bootstrap_results/1b_horizontal_bootstrap_{num_classes}classes_limited40_spearman_summary.csv",
                'raw': f"bootstrap_results/1b_horizontal_bootstrap_{num_classes}classes_limited40_spearman_raw_correlations.json"
            },
            '8b': {
                'summary': f"bootstrap_results/8b_horizontal_bootstrap_{num_classes}classes_limited40_spearman_summary.csv",
                'raw': f"bootstrap_results/8b_horizontal_bootstrap_{num_classes}classes_limited40_spearman_raw_correlations.json"
            },
            '70b': {
                'summary': f"bootstrap_results/70b_horizontal_bootstrap_{num_classes}classes_limited40_spearman_summary.csv",
                'raw': f"bootstrap_results/70b_horizontal_bootstrap_{num_classes}classes_limited40_spearman_raw_correlations.json"
            }
        }
        
        # Original CSV files (needed for 0-shot accuracies)
        original_csv_paths = {
            '1b': f"plots_comparison/1b_horizontal_correlations_{num_classes}classes_limited40_spearman.csv",
            '8b': f"plots_comparison/8b_horizontal_correlations_{num_classes}classes_limited40_spearman.csv",
            '70b': f"plots_comparison/70b_horizontal_correlations_{num_classes}classes_limited40_spearman.csv"
        }
        
        print("Creating bootstrap correlation comparison plot...")
        print("Bootstrap files:")
        for model in ['1b', '8b', '70b']:
            print(f"  {model.upper()}: {bootstrap_paths[model]['summary']}")
        
        print("Original CSV files (for 0-shot accuracies):")
        for model in ['1b', '8b', '70b']:
            print(f"  {model.upper()}: {original_csv_paths[model]}")
        
        # Check if bootstrap files exist
        missing_files = []
        for model in ['1b', '8b', '70b']:
            if not Path(bootstrap_paths[model]['summary']).exists():
                missing_files.append(bootstrap_paths[model]['summary'])
            if not Path(bootstrap_paths[model]['raw']).exists():
                missing_files.append(bootstrap_paths[model]['raw'])
            if not Path(original_csv_paths[model]).exists():
                missing_files.append(original_csv_paths[model])
        
        if missing_files:
            print("Error: Missing files:")
            for file in missing_files:
                print(f"  {file}")
            return
        
        # Create bootstrap comparison plot
        try:
            print("\n" + "="*50)
            print("CREATING BOOTSTRAP CORRELATION COMPARISON PLOT")
            print("="*50)
            plot_bootstrap_correlations_comparison(bootstrap_paths, original_csv_paths, num_classes, output_dir="plots_correlations")
            
        except Exception as e:
            print(f"Error creating bootstrap plot: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Original method without bootstrap
        num_classes = 5  # You can change this back to 5 if you have those files
        csv_path_1b = f"plots_comparison/1b_horizontal_correlations_{num_classes}classes_limited40_spearman.csv"
        csv_path_7b = f"plots_comparison/8b_horizontal_correlations_{num_classes}classes_limited40_spearman.csv"
        csv_path_70b = f"plots_comparison/70b_horizontal_correlations_{num_classes}classes_limited40_spearman.csv"
        
        print("Creating original correlation comparison plot...")
        print(f"Using 1B CSV file: {csv_path_1b}")
        print(f"Using 8B CSV file: {csv_path_7b}")
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
        
        # Create comparison plot
        try:
            print("\n" + "="*50)
            print("CREATING ORIGINAL CORRELATION COMPARISON PLOT")
            print("="*50)
            plot_correlations_comparison(csv_path_1b, csv_path_7b, csv_path_70b, num_classes, output_dir="plots_correlations")
            
        except Exception as e:
            print(f"Error creating original plot: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
