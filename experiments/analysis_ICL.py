import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import re
import pandas as pd
from scipy import stats
from tqdm import tqdm

def extract_config_from_folder(folder_name):
    """Extract number of relabel examples and demonstrations from folder name like 'relabelX_demoY'"""
    match = re.search(r'relabel(\d+)_demo(\d+)', folder_name)
    if match:
        n_relabel = int(match.group(1))
        n_demo = int(match.group(2))
        return n_relabel, n_demo
    return None, None

def load_metrics(base_path):
    """
    Load metrics from all runs and organize them by n_demo and n_relabel.
    """
    base_path = Path(base_path)
    results = []
    
    # Walk through all directories matching the pattern relabelX_demoY
    for config_dir in tqdm(list(base_path.glob("relabel*_demo*")), desc="Processing directories"):
        n_relabel, n_demo = extract_config_from_folder(config_dir.name)
        if n_relabel is None or n_demo is None:
            continue
            
        # Look for run directories
        run_dirs = list(config_dir.glob("run_*"))
        
        for run_dir in run_dirs:
            metrics_file = run_dir / "metrics.pickle"
            if not metrics_file.exists():
                continue
            print(metrics_file)
                
            try:
                with open(metrics_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # Extract accuracy from metrics
                # print(data['metrics'].attrs)
                accuracy = data['metrics'].attrs.get('accuracy_constrained', None)
                if accuracy is None:
                    print(f"Warning: No accuracy found in {metrics_file}")
                    continue
                
                # Get run ID from directory name
                run_id = int(run_dir.name.split('_')[1])
                
                results.append({
                    'n_demo': n_demo,
                    'n_relabel': n_relabel,
                    'run_id': run_id,
                    'accuracy': accuracy,
                    'file': str(metrics_file)
                })
            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")
        #break
    
    if not results:
        raise ValueError("No valid results found in the specified directory structure")
        
    results_df = pd.DataFrame(results)
    
    # Print summary
    print(f"\nFound {len(results_df)} valid results")
    print(f"n_demo range: {results_df['n_demo'].min()} to {results_df['n_demo'].max()}")
    print(f"n_relabel values: {sorted(results_df['n_relabel'].unique())}")
    
    # Verify we have consistent number of runs
    runs_per_config = results_df.groupby(['n_demo', 'n_relabel']).size()
    print("\nNumber of runs per configuration:")
    print(runs_per_config.value_counts().to_string())
    
    if runs_per_config.nunique() > 1:
        print("\nWARNING: Inconsistent number of runs across configurations!")
    
    return results_df

def plot_accuracy_curves(results_df):
    """Create plot with accuracy curves for different relabeling schemes"""
    # Calculate mean and std of accuracy for each n_demo and n_relabel
    print(results_df)
    summary = results_df.groupby(['n_demo', 'n_relabel'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create confidence intervals
    summary['ci'] = summary['std'] * stats.t.ppf((1 + 0.95) / 2, summary['count'] - 1) / np.sqrt(summary['count'])
    
    # Create plot
    plt.figure(figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # Plot a line for each relabeling scheme
    unique_relabels = sorted(summary['n_relabel'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    
    for n_relabel, color in zip(unique_relabels, colors):
        data = summary[summary['n_relabel'] == n_relabel]
        plt.errorbar(data['n_demo'], data['mean'], 
                    yerr=data['ci'], 
                    label=f'Relabeling with {n_relabel} examples',
                    marker='o',
                    color=color,
                    alpha=0.8,
                    markersize=4,
                    capsize=3,  # Add horizontal caps to error bars
                    capthick=1)  # Make the caps slightly thicker
    
    plt.xscale('log')
    plt.xlabel('Number of Demonstrations (log scale)')
    plt.ylabel('Accuracy')
    plt.title('ICL Performance with Different Relabeling Schemes\n(with 95% confidence intervals, log x-axis)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('icl_accuracy_curves_3classes_new_loglog.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_curves_limited(results_df, max_demos=10):
    """Create plot with accuracy curves for different relabeling schemes, limited to specified max demonstrations"""
    # Filter data for demos <= max_demos
    results_df_limited = results_df[results_df['n_demo'] <= max_demos].copy()
    
    # Calculate mean and std of accuracy for each n_demo and n_relabel
    summary = results_df_limited.groupby(['n_demo', 'n_relabel'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create confidence intervals
    summary['ci'] = summary['std'] * stats.t.ppf((1 + 0.95) / 2, summary['count'] - 1) / np.sqrt(summary['count'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Plot a line for each relabeling scheme
    unique_relabels = sorted(summary['n_relabel'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    
    for n_relabel, color in zip(unique_relabels, colors):
        data = summary[summary['n_relabel'] == n_relabel]
        plt.errorbar(data['n_demo'], data['mean'], 
                    yerr=data['ci'], 
                    label=f'Relabeling with {n_relabel} examples',
                    marker='o',
                    color=color,
                    alpha=0.8,
                    markersize=4,
                    capsize=3,
                    capthick=1)
    
    plt.xscale('log')
    plt.xlabel('Number of Demonstrations (log scale)')
    plt.ylabel('Accuracy')
    plt.title(f'ICL Performance with Different Relabeling Schemes\n(up to {max_demos} demonstrations, with 95% confidence intervals, log x-axis)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('icl_accuracy_curves_3classes_limited_new_loglog.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_log_accuracy_curves(results_df):
    """Create plot with log accuracy curves for different relabeling schemes"""
    # Calculate mean accuracy for each n_demo and n_relabel
    summary = results_df.groupby(['n_demo', 'n_relabel'])['accuracy'].agg(['mean']).reset_index()
    
    # Create plot
    plt.figure(figsize=(20, 8))
    sns.set_style("whitegrid")
    
    # Plot a line for each relabeling scheme
    unique_relabels = sorted(summary['n_relabel'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    
    for n_relabel, color in zip(unique_relabels, colors):
        data = summary[summary['n_relabel'] == n_relabel]
        plt.plot(data['n_demo'], np.log(data['mean']), 
                label=f'Relabeling with {n_relabel} examples',
                marker='o',
                color=color,
                alpha=0.8,
                markersize=4)
    
    plt.xscale('log')
    plt.xlabel('Number of Demonstrations (log scale)')
    plt.ylabel('Log Accuracy')
    plt.title('ICL Performance with Different Relabeling Schemes\n(Log-Log Scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('icl_log_accuracy_curves_3classes_new_loglog.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_log_accuracy_curves_limited(results_df, max_demos=10):
    """Create plot with log accuracy curves for different relabeling schemes, limited to specified max demonstrations"""
    # Filter data for demos <= max_demos
    results_df_limited = results_df[results_df['n_demo'] <= max_demos].copy()
    
    # Calculate mean accuracy for each n_demo and n_relabel
    summary = results_df_limited.groupby(['n_demo', 'n_relabel'])['accuracy'].agg(['mean']).reset_index()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Plot a line for each relabeling scheme
    unique_relabels = sorted(summary['n_relabel'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_relabels)))
    
    for n_relabel, color in zip(unique_relabels, colors):
        data = summary[summary['n_relabel'] == n_relabel]
        plt.plot(data['n_demo'], np.log(data['mean']), 
                label=f'Relabeling with {n_relabel} examples',
                marker='o',
                color=color,
                alpha=0.8,
                markersize=4)
    
    plt.xscale('log')
    plt.xlabel('Number of Demonstrations (log scale)')
    plt.ylabel('Log Accuracy')
    plt.title(f'ICL Performance with Different Relabeling Schemes\n(up to {max_demos} demonstrations, Log-Log Scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('icl_log_accuracy_curves_3classes_limited_new_loglog.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    # Base path to results
    base_path = Path("learning_curves_relabel_demos/claude_multitask/llama3.1_base")
    
    # Load all results
    print("Loading results...")
    results_df = load_metrics(base_path)
    
    # Print detailed statistics
    print_detailed_stats(results_df)
    
    # # Create plots
    # print("\nCreating plots...")
    # plot_accuracy_curves(results_df)
    # print("Full plot saved as icl_accuracy_curves2.png")
    
    # plot_accuracy_curves(results_df)
    # print("Full plot saved as icl_accuracy_curves_3classes_new.png")
    # plot_accuracy_curves_limited(results_df, max_demos=10)
    # print("Limited plot (up to 10 demonstrations) saved as icl_accuracy_curves_limited_new.png")
    
    plot_log_accuracy_curves(results_df)
    print("Full log accuracy plot saved as icl_log_accuracy_curves_3classes_new.png")
    plot_log_accuracy_curves_limited(results_df, max_demos=10)
    print("Limited log accuracy plot (up to 10 demonstrations) saved as icl_log_accuracy_curves_limited_new.png")

if __name__ == "__main__":
    main() 