import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

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
    
    return results_df

def identify_duplicate_relabel_schemes(results_df):
    """
    Identify duplicate n_relabel schemes by comparing their accuracy patterns.
    Returns a dictionary mapping each n_relabel to a unique group ID.
    """
    print("Identifying duplicate n_relabel schemes...")
    
    relabel_values = sorted(results_df['n_relabel'].unique())
    relabel_groups = {}
    group_representatives = {}
    next_group_id = 0
    
    for n_relabel in relabel_values:
        # Get accuracies for this relabel scheme across all n_demo values
        relabel_data = results_df[results_df['n_relabel'] == n_relabel]
        accuracy_pattern = relabel_data.groupby('n_demo')['accuracy'].mean().values
        
        # Check if this pattern matches any existing group
        found_match = False
        for group_id, representative_pattern in group_representatives.items():
            # Compare patterns with small tolerance for floating point differences
            if len(accuracy_pattern) == len(representative_pattern) and \
               np.allclose(accuracy_pattern, representative_pattern, rtol=1e-10):
                relabel_groups[n_relabel] = group_id
                found_match = True
                print(f"  n_relabel={n_relabel} is duplicate of group {group_id}")
                break
        
        if not found_match:
            # This is a unique pattern
            relabel_groups[n_relabel] = next_group_id
            group_representatives[next_group_id] = accuracy_pattern
            print(f"  n_relabel={n_relabel} is unique (group {next_group_id})")
            next_group_id += 1
    
    # Print summary
    unique_groups = len(group_representatives)
    total_relabels = len(relabel_values)
    print(f"\nFound {unique_groups} unique relabel schemes out of {total_relabels} total schemes")
    
    return relabel_groups

def bootstrap_sample_runs(results_df, n_bootstrap=1000, random_seed=42):
    """
    Generate bootstrap samples by randomly sampling one run for each (n_demo, n_relabel) combination.
    
    Args:
        results_df: DataFrame with columns ['n_demo', 'n_relabel', 'run_id', 'accuracy']
        n_bootstrap: Number of bootstrap samples to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        List of DataFrames, each representing one bootstrap sample
    """
    np.random.seed(random_seed)
    
    # Get all unique combinations of (n_demo, n_relabel)
    combinations = results_df[['n_demo', 'n_relabel']].drop_duplicates()
    
    bootstrap_samples = []
    
    for bootstrap_idx in range(n_bootstrap):
        bootstrap_data = []
        
        for _, row in combinations.iterrows():
            n_demo, n_relabel = row['n_demo'], row['n_relabel']
            
            # Get all runs for this combination
            subset = results_df[(results_df['n_demo'] == n_demo) & 
                               (results_df['n_relabel'] == n_relabel)]
            
            if len(subset) > 0:
                # Randomly sample one run
                sampled_run = subset.sample(n=1, random_state=bootstrap_idx*1000 + hash((n_demo, n_relabel)) % 1000)
                bootstrap_data.append(sampled_run)
        
        if bootstrap_data:
            bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)
            bootstrap_samples.append(bootstrap_df)
    
    return bootstrap_samples

def compute_bootstrap_vertical_correlations(results_df, num_classes, n_bootstrap=1000, 
                                          limited_to_40_demos=False, spearman=False, random_seed=42):
    """
    Compute vertical correlations using bootstrap resampling.
    
    Returns:
        dict with keys:
        - 'bootstrap_results': List of correlation DataFrames for each bootstrap sample
        - 'summary_stats': Summary statistics across bootstrap samples
        - 'raw_correlations': All individual correlation values for each n_demo
    """
    print(f"\nComputing bootstrap vertical correlations with {n_bootstrap} samples...")
    
    # Generate bootstrap samples
    bootstrap_samples = bootstrap_sample_runs(results_df, n_bootstrap, random_seed)
    
    # Store results for each bootstrap iteration
    bootstrap_results = []
    all_correlations_by_demo = defaultdict(list)  # n_demo -> list of correlations across bootstrap samples
    
    for i, bootstrap_df in enumerate(bootstrap_samples):
        if i % 100 == 0:
            print(f"Processing bootstrap sample {i+1}/{n_bootstrap}")
        
        # Compute correlations for this bootstrap sample (without averaging)
        correlations_df = compute_vertical_correlations_single_sample(
            bootstrap_df, num_classes, limited_to_40_demos, spearman
        )
        
        bootstrap_results.append(correlations_df)
        
        # Store correlations by n_demo for summary statistics
        for _, row in correlations_df.iterrows():
            all_correlations_by_demo[row['n_demo']].append(row['correlation'])
    
    # Compute summary statistics
    summary_stats = []
    for n_demo, correlations in all_correlations_by_demo.items():
        correlations = np.array(correlations)
        summary_stats.append({
            'n_demo': n_demo,
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'median_correlation': np.median(correlations),
            'ci_2.5': np.percentile(correlations, 2.5),
            'ci_97.5': np.percentile(correlations, 97.5),
            'ci_5': np.percentile(correlations, 5),
            'ci_95': np.percentile(correlations, 95),
            'n_bootstrap_samples': len(correlations)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    return {
        'bootstrap_results': bootstrap_results,
        'summary_stats': summary_df,
        'raw_correlations': dict(all_correlations_by_demo)
    }

def compute_vertical_correlations_single_sample(results_df, num_classes, limited_to_40_demos=False, spearman=False):
    """
    Compute vertical correlations for a single sample (used by bootstrap function).
    This is the original compute_vertical_correlations but without the averaging step.
    """
    # Filter data to include 0 demos and demos >= num_classes
    if limited_to_40_demos:
        filtered_df = results_df[((results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)) & (results_df['n_demo'] <= 40)].copy()
    else:
        filtered_df = results_df[(results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)].copy()
    
    # Get unique demo values (excluding 0)
    demo_values = sorted([d for d in filtered_df['n_demo'].unique() if d > 0])
    relabel_values = sorted(filtered_df['n_relabel'].unique())
    
    correlations = []
    
    for n_demo in demo_values:
        # Get 0-shot accuracies (n_demo = 0) for each relabel scheme
        zero_shot_data = filtered_df[filtered_df['n_demo'] == 0]
        zero_shot_accuracies = []
        
        # Get few-shot accuracies (current n_demo) for each relabel scheme
        few_shot_data = filtered_df[filtered_df['n_demo'] == n_demo]
        few_shot_accuracies = []
        
        # Ensure we have data for all relabel schemes
        for n_relabel in relabel_values:
            zero_shot_subset = zero_shot_data[zero_shot_data['n_relabel'] == n_relabel]
            few_shot_subset = few_shot_data[few_shot_data['n_relabel'] == n_relabel]
            
            if len(zero_shot_subset) > 0 and len(few_shot_subset) > 0:
                # Take the single sampled accuracy (no averaging)
                zero_shot_acc = zero_shot_subset['accuracy'].iloc[0]
                few_shot_acc = few_shot_subset['accuracy'].iloc[0]
                
                zero_shot_accuracies.append(zero_shot_acc)
                few_shot_accuracies.append(few_shot_acc)
        
        if len(zero_shot_accuracies) == len(few_shot_accuracies) and len(zero_shot_accuracies) > 1:
            # Compute correlation
            if spearman:
                correlation, p_value = spearmanr(zero_shot_accuracies, few_shot_accuracies)
            else:
                correlation, p_value = stats.pearsonr(zero_shot_accuracies, few_shot_accuracies)
            
            correlations.append({
                'n_demo': n_demo,
                'correlation': correlation,
                'p_value': p_value,
                'n_points': len(zero_shot_accuracies)
            })
    
    return pd.DataFrame(correlations)

def compute_vertical_correlations(results_df, num_classes, limited_to_40_demos=False, spearman=False):
    """
    Compute correlations between relabeling accuracies and 0-shot accuracy for each n_demo.
    """
    print(f"\nComputing correlations for {num_classes} classes...")
    if limited_to_40_demos:
        print("Limiting analysis to demos <= 40")
    if spearman:
        print("Using Spearman correlation")
    
    # Filter data to include 0 demos and demos >= num_classes
    if limited_to_40_demos:
        filtered_df = results_df[((results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)) & (results_df['n_demo'] <= 40)].copy()
    else:
        filtered_df = results_df[(results_df['n_demo'] == 0) | (results_df['n_demo'] >= num_classes)].copy()
    
    # Identify duplicate relabel schemes
    relabel_groups = identify_duplicate_relabel_schemes(filtered_df)
    
    # Get unique demo values (excluding 0)
    demo_values = sorted([d for d in filtered_df['n_demo'].unique() if d > 0])
    relabel_values = sorted(filtered_df['n_relabel'].unique())
    
    print(f"Demo values to analyze: {demo_values}")
    print(f"Relabel values: {relabel_values}")
    
    correlations = []
    
    for n_demo in demo_values:
        print(f"\nProcessing n_demo = {n_demo}...")
        
        # Get 0-shot accuracies (n_demo = 0) for each relabel scheme
        zero_shot_data = filtered_df[filtered_df['n_demo'] == 0]
        zero_shot_accuracies = []
        
        # Get few-shot accuracies (current n_demo) for each relabel scheme
        few_shot_data = filtered_df[filtered_df['n_demo'] == n_demo]
        few_shot_accuracies = []
        
        # Ensure we have data for all relabel schemes
        for n_relabel in relabel_values:
            zero_shot_subset = zero_shot_data[zero_shot_data['n_relabel'] == n_relabel]
            few_shot_subset = few_shot_data[few_shot_data['n_relabel'] == n_relabel]
            
            if len(zero_shot_subset) > 0 and len(few_shot_subset) > 0:
                # Take mean accuracy across runs for each relabel scheme
                zero_shot_mean = zero_shot_subset['accuracy'].mean()
                few_shot_mean = few_shot_subset['accuracy'].mean()
                
                zero_shot_accuracies.append(zero_shot_mean)
                few_shot_accuracies.append(few_shot_mean)
            else:
                print(f"Warning: Missing data for n_relabel={n_relabel} at n_demo={n_demo}")
        
        if len(zero_shot_accuracies) == len(few_shot_accuracies) and len(zero_shot_accuracies) > 1:
            # Compute correlation
            if spearman:
                correlation, p_value = spearmanr(zero_shot_accuracies, few_shot_accuracies)
            else:
                correlation, p_value = stats.pearsonr(zero_shot_accuracies, few_shot_accuracies)
            
            correlations.append({
                'n_demo': n_demo,
                'correlation': correlation,
                'p_value': p_value,
                'n_points': len(zero_shot_accuracies),
                'zero_shot_accuracies': zero_shot_accuracies,
                'few_shot_accuracies': few_shot_accuracies,
                'relabel_groups': relabel_groups
            })
            
            print(f"  Correlation: {correlation:.4f}, p-value: {p_value:.4f}, n_points: {len(zero_shot_accuracies)}")
        else:
            print(f"  Skipping n_demo={n_demo}: insufficient data points")
    
    return pd.DataFrame(correlations)

def compute_horizontal_correlations(results_df, num_classes, limited_to_40_demos=False, spearman=False):
    """
    Compute correlations between n_demos and accuracy for each n_relabel scheme.
    Only uses specific n_demo values [0, 10, 20, 30, 40] for correlation computation.
    """
    print(f"\nComputing horizontal correlations for {num_classes} classes...")
    if limited_to_40_demos:
        print("Limiting analysis to demos <= 40")
    if spearman:
        print("Using Spearman correlation")
    
    # Define specific demo values to use for horizontal correlations
    # target_demo_values = [0, 10, 20, 30, 40]
    # print(f"Computing horizontal correlations only for n_demo values: {target_demo_values}")
    
    if limited_to_40_demos:
        filtered_df = results_df[results_df['n_demo'] <= 40].copy()
    else:
        filtered_df = results_df.copy()
    
    # Identify duplicate relabel schemes
    relabel_groups = identify_duplicate_relabel_schemes(filtered_df)
    
    # Get unique relabel values
    relabel_values = sorted(filtered_df['n_relabel'].unique())
    
    print(f"Relabel values to analyze: {relabel_values}")
    
    horizontal_correlations = []
    
    for n_relabel in relabel_values:
        print(f"\nProcessing n_relabel = {n_relabel}...")
        
        # Get data for this relabeling scheme
        relabel_data = filtered_df[filtered_df['n_relabel'] == n_relabel]
        
        # Group by n_demo and compute mean accuracy across runs
        demo_accuracy = relabel_data.groupby('n_demo')['accuracy'].mean().reset_index()
        
        if len(demo_accuracy) > 1:
            # Compute correlation between n_demo and accuracy
            if spearman:
                correlation, p_value = spearmanr(demo_accuracy['n_demo'], demo_accuracy['accuracy'])
            else:
                correlation, p_value = stats.pearsonr(demo_accuracy['n_demo'], demo_accuracy['accuracy'])
            
            horizontal_correlations.append({
                'n_relabel': n_relabel,
                'correlation': correlation,
                'p_value': p_value,
                'n_points': len(demo_accuracy),
                'n_demos': demo_accuracy['n_demo'].tolist(),
                'accuracies': demo_accuracy['accuracy'].tolist(),
                'group_id': relabel_groups[n_relabel]
            })
            
            print(f"  Correlation: {correlation:.4f}, p-value: {p_value:.4f}, n_points: {len(demo_accuracy)}")
        else:
            print(f"  Skipping n_relabel={n_relabel}: insufficient data points")
    
    return pd.DataFrame(horizontal_correlations)

def compute_bootstrap_horizontal_correlations(results_df, num_classes, n_bootstrap=1000,
                                            limited_to_40_demos=False, spearman=False, random_seed=42):
    """
    Compute horizontal correlations using bootstrap resampling.
    
    Returns:
        dict with keys:
        - 'bootstrap_results': List of correlation DataFrames for each bootstrap sample  
        - 'summary_stats': Summary statistics across bootstrap samples
        - 'raw_correlations': All individual correlation values for each n_relabel
    """
    print(f"\nComputing bootstrap horizontal correlations with {n_bootstrap} samples...")
    
    # Generate bootstrap samples
    bootstrap_samples = bootstrap_sample_runs(results_df, n_bootstrap, random_seed)
    
    # Store results for each bootstrap iteration
    bootstrap_results = []
    all_correlations_by_relabel = defaultdict(list)  # n_relabel -> list of correlations across bootstrap samples
    
    for i, bootstrap_df in enumerate(bootstrap_samples):
        if i % 100 == 0:
            print(f"Processing bootstrap sample {i+1}/{n_bootstrap}")
        
        # Compute correlations for this bootstrap sample (without averaging)
        correlations_df = compute_horizontal_correlations_single_sample(
            bootstrap_df, num_classes, limited_to_40_demos, spearman
        )
        
        bootstrap_results.append(correlations_df)
        
        # Store correlations by n_relabel for summary statistics
        for _, row in correlations_df.iterrows():
            all_correlations_by_relabel[row['n_relabel']].append(row['correlation'])
    
    # Compute summary statistics
    summary_stats = []
    for n_relabel, correlations in all_correlations_by_relabel.items():
        correlations = np.array(correlations)
        summary_stats.append({
            'n_relabel': n_relabel,
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'median_correlation': np.median(correlations),
            'ci_2.5': np.percentile(correlations, 2.5),
            'ci_97.5': np.percentile(correlations, 97.5),
            'ci_5': np.percentile(correlations, 5),
            'ci_95': np.percentile(correlations, 95),
            'n_bootstrap_samples': len(correlations)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    return {
        'bootstrap_results': bootstrap_results,
        'summary_stats': summary_df,
        'raw_correlations': dict(all_correlations_by_relabel)
    }

def compute_horizontal_correlations_single_sample(results_df, num_classes, limited_to_40_demos=False, spearman=False):
    """
    Compute horizontal correlations for a single sample (used by bootstrap function).
    This is the original compute_horizontal_correlations but without the averaging step.
    """
    if limited_to_40_demos:
        filtered_df = results_df[results_df['n_demo'] <= 40].copy()
    else:
        filtered_df = results_df.copy()
    
    # Get unique relabel values
    relabel_values = sorted(filtered_df['n_relabel'].unique())
    
    horizontal_correlations = []
    
    for n_relabel in relabel_values:
        # Get data for this relabeling scheme
        relabel_data = filtered_df[filtered_df['n_relabel'] == n_relabel]
        
        # Since we have one run per (n_demo, n_relabel), no need to average
        demo_accuracy = relabel_data[['n_demo', 'accuracy']].copy()
        
        if len(demo_accuracy) > 1:
            # Compute correlation between n_demo and accuracy
            if spearman:
                correlation, p_value = spearmanr(demo_accuracy['n_demo'], demo_accuracy['accuracy'])
            else:
                correlation, p_value = stats.pearsonr(demo_accuracy['n_demo'], demo_accuracy['accuracy'])
            
            horizontal_correlations.append({
                'n_relabel': n_relabel,
                'correlation': correlation,
                'p_value': p_value,
                'n_points': len(demo_accuracy)
            })
    
    return pd.DataFrame(horizontal_correlations)

def save_bootstrap_results(bootstrap_results, model_name, correlation_type, num_classes, 
                          limited_to_40_demos=False, spearman=False):
    """
    Save bootstrap results to files.
    
    Args:
        bootstrap_results: Dict with 'bootstrap_results', 'summary_stats', 'raw_correlations'
        model_name: Model identifier (e.g., '8b', '70b')
        correlation_type: 'vertical' or 'horizontal'
        num_classes: Number of classes
        limited_to_40_demos: Whether analysis was limited to 40 demos
        spearman: Whether Spearman correlation was used
    """
    # Create suffix for filenames
    suffix = ""
    if limited_to_40_demos:
        suffix += "_limited40"
    if spearman:
        suffix += "_spearman"
    
    base_filename = f'bootstrap_results/{model_name}_{correlation_type}_bootstrap_{num_classes}classes{suffix}'
    
    # Save summary statistics (most important)
    summary_file = f'{base_filename}_summary.csv'
    bootstrap_results['summary_stats'].to_csv(summary_file, index=False)
    print(f"Bootstrap summary statistics saved to {summary_file}")
    
    # Save raw correlations as JSON for detailed analysis
    raw_file = f'{base_filename}_raw_correlations.json'
    with open(raw_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        raw_data = {}
        for key, values in bootstrap_results['raw_correlations'].items():
            raw_data[str(key)] = [float(v) for v in values]
        json.dump(raw_data, f, indent=2)
    print(f"Raw bootstrap correlations saved to {raw_file}")
    
    # Optionally save a sample of individual bootstrap results (first 10) for inspection
    sample_file = f'{base_filename}_sample_results.csv'
    if len(bootstrap_results['bootstrap_results']) > 0:
        sample_results = []
        for i, df in enumerate(bootstrap_results['bootstrap_results'][:10]):  # First 10 samples
            df_copy = df.copy()
            df_copy['bootstrap_sample'] = i
            sample_results.append(df_copy)
        
        if sample_results:
            sample_df = pd.concat(sample_results, ignore_index=True)
            sample_df.to_csv(sample_file, index=False)
            print(f"Sample bootstrap results (first 10) saved to {sample_file}")

def print_bootstrap_correlation_stats(bootstrap_results, correlation_type):
    """
    Print detailed statistics about the bootstrap correlations.
    """
    summary_df = bootstrap_results['summary_stats']
    
    print(f"\n{correlation_type.title()} Bootstrap Correlation Statistics:")
    print(f"Number of correlation points analyzed: {len(summary_df)}")
    
    if len(summary_df) > 0:
        print(f"Mean correlation (across all points): {summary_df['mean_correlation'].mean():.4f}")
        print(f"Std of mean correlations: {summary_df['mean_correlation'].std():.4f}")
        print(f"Mean bootstrap std: {summary_df['std_correlation'].mean():.4f}")
        
        print(f"\nConfidence Interval Coverage:")
        print(f"Mean 95% CI width: {(summary_df['ci_97.5'] - summary_df['ci_2.5']).mean():.4f}")
        print(f"Mean 90% CI width: {(summary_df['ci_95'] - summary_df['ci_5']).mean():.4f}")
        
        # Show detailed results
        print(f"\nDetailed Bootstrap Results:")
        display_cols = ['mean_correlation', 'std_correlation', 'median_correlation', 'ci_2.5', 'ci_97.5']
        if correlation_type == 'vertical':
            key_col = 'n_demo'
        else:
            key_col = 'n_relabel'
            
        if key_col in summary_df.columns:
            display_cols = [key_col] + display_cols
            
        print(summary_df[display_cols].round(4).to_string(index=False))

def compute_deduplicated_vertical_stats(correlations_df, spearman=False):
    """
    Compute statistics excluding duplicates by using only one representative from each group.
    """
    if 'relabel_groups' not in correlations_df.columns or len(correlations_df) == 0:
        return correlations_df['correlation'].mean(), correlations_df['correlation'].std(), len(correlations_df)
    
    # For each n_demo, compute average correlation excluding duplicates
    deduplicated_correlations = []
    
    for _, row in correlations_df.iterrows():
        relabel_groups = row['relabel_groups']
        zero_shot_accs = row['zero_shot_accuracies']
        few_shot_accs = row['few_shot_accuracies']
        
        # Group accuracies by group_id and take one representative from each group
        group_zero_shot = {}
        group_few_shot = {}
        
        # Assuming the order matches the relabel_values order used in computation
        relabel_values = sorted(relabel_groups.keys())
        for i, n_relabel in enumerate(relabel_values):
            if i < len(zero_shot_accs):  # Safety check
                group_id = relabel_groups[n_relabel]
                if group_id not in group_zero_shot:  # Take first representative of each group
                    group_zero_shot[group_id] = zero_shot_accs[i]
                    group_few_shot[group_id] = few_shot_accs[i]
        
        # Compute correlation with deduplicated data
        if len(group_zero_shot) > 1:
            unique_zero_shot = list(group_zero_shot.values())
            unique_few_shot = list(group_few_shot.values())
            if spearman:
                correlation, _ = spearmanr(unique_zero_shot, unique_few_shot)
            else:
                correlation, _ = stats.pearsonr(unique_zero_shot, unique_few_shot)
            deduplicated_correlations.append(correlation)
    
    if len(deduplicated_correlations) > 0:
        return np.mean(deduplicated_correlations), np.std(deduplicated_correlations), len(deduplicated_correlations)
    else:
        return np.nan, np.nan, 0

def print_vertical_correlation_stats(correlations_df, spearman=False):
    """
    Print detailed statistics about the correlations.
    """
    print(f"\nVertical Correlation Statistics:")
    print(f"Number of correlations computed: {len(correlations_df)}")
    print(f"Mean correlation: {correlations_df['correlation'].mean():.4f}")
    print(f"Std correlation: {correlations_df['correlation'].std():.4f}")
    print(f"Min correlation: {correlations_df['correlation'].min():.4f}")
    print(f"Max correlation: {correlations_df['correlation'].max():.4f}")
    
    # Compute deduplicated statistics
    dedup_mean, dedup_std, dedup_count = compute_deduplicated_vertical_stats(correlations_df, spearman)
    print(f"\nDeduplicated Statistics (excluding duplicate relabel schemes):")
    print(f"Mean correlation (deduplicated): {dedup_mean:.4f}")
    print(f"Std correlation (deduplicated): {dedup_std:.4f}")
    print(f"Number of correlations (deduplicated): {dedup_count}")
    
    # Count significant correlations (p < 0.05)
    significant = correlations_df[correlations_df['p_value'] < 0.05]
    print(f"\nSignificant correlations (p < 0.05): {len(significant)}/{len(correlations_df)}")
    
    # Strong correlations (|r| > 0.7)
    strong = correlations_df[abs(correlations_df['correlation']) > 0.7]
    print(f"Strong correlations (|r| > 0.7): {len(strong)}/{len(correlations_df)}")
    
    print(f"\nTop 5 highest correlations:")
    top_corr = correlations_df.nlargest(5, 'correlation')[['n_demo', 'correlation', 'p_value']]
    print(top_corr.to_string(index=False))
    
    print(f"\nTop 5 lowest correlations:")
    bottom_corr = correlations_df.nsmallest(5, 'correlation')[['n_demo', 'correlation', 'p_value']]
    print(bottom_corr.to_string(index=False))

def print_horizontal_correlation_stats(horizontal_correlations_df):
    """
    Print detailed statistics about the horizontal correlations.
    """
    print(f"\nHorizontal Correlation Statistics:")
    print(f"Number of correlations computed: {len(horizontal_correlations_df)}")
    print(f"Mean correlation: {horizontal_correlations_df['correlation'].mean():.4f}")
    print(f"Std correlation: {horizontal_correlations_df['correlation'].std():.4f}")
    print(f"Min correlation: {horizontal_correlations_df['correlation'].min():.4f}")
    print(f"Max correlation: {horizontal_correlations_df['correlation'].max():.4f}")
    
    # Compute deduplicated statistics
    if 'group_id' in horizontal_correlations_df.columns:
        # Get unique correlations (one per group)
        unique_correlations = horizontal_correlations_df.groupby('group_id')['correlation'].first()
        print(f"\nDeduplicated Statistics (excluding duplicate relabel schemes):")
        print(f"Number of unique relabel groups: {len(unique_correlations)}")
        print(f"Mean correlation (deduplicated): {unique_correlations.mean():.4f}")
        print(f"Std correlation (deduplicated): {unique_correlations.std():.4f}")
        print(f"Min correlation (deduplicated): {unique_correlations.min():.4f}")
        print(f"Max correlation (deduplicated): {unique_correlations.max():.4f}")
        
        # Count significant correlations among unique groups
        unique_df = horizontal_correlations_df.groupby('group_id').first()
        significant_unique = unique_df[unique_df['p_value'] < 0.05]
        print(f"Significant correlations (deduplicated, p < 0.05): {len(significant_unique)}/{len(unique_correlations)}")
        
        # Strong correlations among unique groups
        strong_unique = unique_df[abs(unique_df['correlation']) > 0.7]
        print(f"Strong correlations (deduplicated, |r| > 0.7): {len(strong_unique)}/{len(unique_correlations)}")
    
    # Count significant correlations (p < 0.05)
    significant = horizontal_correlations_df[horizontal_correlations_df['p_value'] < 0.05]
    print(f"\nAll correlations (including duplicates):")
    print(f"Significant correlations (p < 0.05): {len(significant)}/{len(horizontal_correlations_df)}")
    
    # Strong correlations (|r| > 0.7)
    strong = horizontal_correlations_df[abs(horizontal_correlations_df['correlation']) > 0.7]
    print(f"Strong correlations (|r| > 0.7): {len(strong)}/{len(horizontal_correlations_df)}")
    
    print(f"\nAll horizontal correlations:")
    if 'group_id' in horizontal_correlations_df.columns:
        all_corr = horizontal_correlations_df[['n_relabel', 'correlation', 'p_value', 'n_points', 'group_id']]
    else:
        all_corr = horizontal_correlations_df[['n_relabel', 'correlation', 'p_value', 'n_points']]
    print(all_corr.to_string(index=False))

def save_correlations_to_csv(correlations_df, num_classes, correlation_type, model_name, limited_to_40_demos=False, spearman=False):
    """
    Save correlations to CSV file.
    """
    suffix = ""
    if limited_to_40_demos:
        suffix += "_limited40"
    if spearman:
        suffix += "_spearman"
    
    output_file = f'plots_comparison/{model_name}_{correlation_type}_correlations_{num_classes}classes{suffix}.csv'
    correlations_df.to_csv(output_file, index=False)
    print(f"{correlation_type.title()} correlations saved to {output_file}")

def main():
    # Configuration
    limited_to_40_demos = True
    spearman = True
    num_classes = 5
    n_bootstrap = 1000  # Number of bootstrap samples
    use_bootstrap = True  # Set to False to use original averaging method
    
    if num_classes == 3:
        csv_path_1b = "learning_curves_relabel_demos_3classes_1b/claude_multitask/llama3.1_1b_base/consolidated_metrics.csv"
        csv_path_8b = "learning_curves_relabel_demos_3classes_8b/claude_multitask/llama3.1_base/consolidated_metrics.csv"
        csv_path_70b = "learning_curves_relabel_demos_3classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    elif num_classes == 5:
        csv_path_1b = "learning_curves_relabel_demos_5classes_1b/claude_multitask/llama3.1_1b_base/consolidated_metrics.csv"
        csv_path_8b = "learning_curves_relabel_demos_5classes_8b/claude_multitask/llama3.1_base/consolidated_metrics.csv"
        csv_path_70b = "learning_curves_relabel_demos_5classes_70b/claude_multitask/llama3.1_70b_instruct/consolidated_metrics.csv"
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")
    
    try:
        # Load data
        results_df_1b = load_metrics_from_csv(csv_path_1b)
        results_df_8b = load_metrics_from_csv(csv_path_8b)
        results_df_70b = load_metrics_from_csv(csv_path_70b)
        
        if use_bootstrap:
            # Bootstrap analysis
            print("="*60)
            print("BOOTSTRAP VERTICAL CORRELATIONS (n_demo vs correlation with 0-shot)")
            print("="*60)
            
            # Compute bootstrap vertical correlations
            bootstrap_vertical_1b = compute_bootstrap_vertical_correlations(
                results_df_1b, num_classes, n_bootstrap, limited_to_40_demos, spearman, random_seed=42)
            bootstrap_vertical_8b = compute_bootstrap_vertical_correlations(
                results_df_8b, num_classes, n_bootstrap, limited_to_40_demos, spearman, random_seed=43)
            bootstrap_vertical_70b = compute_bootstrap_vertical_correlations(
                results_df_70b, num_classes, n_bootstrap, limited_to_40_demos, spearman, random_seed=44)
            
            # Print and save results
            print("\n1B Model Results:")
            print_bootstrap_correlation_stats(bootstrap_vertical_1b, "vertical")
            save_bootstrap_results(bootstrap_vertical_1b, "1b", "vertical", num_classes, limited_to_40_demos, spearman)
            
            print("\n8B Model Results:")
            print_bootstrap_correlation_stats(bootstrap_vertical_8b, "vertical")
            save_bootstrap_results(bootstrap_vertical_8b, "8b", "vertical", num_classes, limited_to_40_demos, spearman)
            
            print("\n70B Model Results:")
            print_bootstrap_correlation_stats(bootstrap_vertical_70b, "vertical")
            save_bootstrap_results(bootstrap_vertical_70b, "70b", "vertical", num_classes, limited_to_40_demos, spearman)
            
            # Bootstrap horizontal correlations
            print("\n" + "="*60)
            print("BOOTSTRAP HORIZONTAL CORRELATIONS (n_relabel vs correlation with n_demos)")
            print("="*60)
            
            bootstrap_horizontal_1b = compute_bootstrap_horizontal_correlations(
                results_df_1b, num_classes, n_bootstrap, limited_to_40_demos, spearman, random_seed=45)
            bootstrap_horizontal_8b = compute_bootstrap_horizontal_correlations(
                results_df_8b, num_classes, n_bootstrap, limited_to_40_demos, spearman, random_seed=46)
            bootstrap_horizontal_70b = compute_bootstrap_horizontal_correlations(
                results_df_70b, num_classes, n_bootstrap, limited_to_40_demos, spearman, random_seed=47)
            
            # Print and save results
            print("\n1B Model Results:")
            print_bootstrap_correlation_stats(bootstrap_horizontal_1b, "horizontal")
            save_bootstrap_results(bootstrap_horizontal_1b, "1b", "horizontal", num_classes, limited_to_40_demos, spearman)
            
            print("\n8B Model Results:")
            print_bootstrap_correlation_stats(bootstrap_horizontal_8b, "horizontal")
            save_bootstrap_results(bootstrap_horizontal_8b, "8b", "horizontal", num_classes, limited_to_40_demos, spearman)
            
            print("\n70B Model Results:")
            print_bootstrap_correlation_stats(bootstrap_horizontal_70b, "horizontal")
            save_bootstrap_results(bootstrap_horizontal_70b, "70b", "horizontal", num_classes, limited_to_40_demos, spearman)
            
            print(f"\nBootstrap analysis complete with {n_bootstrap} samples!")
            
        else:
            # Original analysis (using averages)
            print("="*60)
            print("ORIGINAL VERTICAL CORRELATIONS (n_demo vs correlation with 0-shot)")
            print("="*60)
            correlations_df_1b = compute_vertical_correlations(results_df_1b, num_classes, limited_to_40_demos, spearman)
            correlations_df_8b = compute_vertical_correlations(results_df_8b, num_classes, limited_to_40_demos, spearman)
            correlations_df_70b = compute_vertical_correlations(results_df_70b, num_classes, limited_to_40_demos, spearman)
            print_vertical_correlation_stats(correlations_df_1b, spearman)
            print_vertical_correlation_stats(correlations_df_8b, spearman)
            print_vertical_correlation_stats(correlations_df_70b, spearman)
            save_correlations_to_csv(correlations_df_1b, num_classes, "vertical", "1b", limited_to_40_demos, spearman)
            save_correlations_to_csv(correlations_df_8b, num_classes, "vertical", "8b", limited_to_40_demos, spearman)
            save_correlations_to_csv(correlations_df_70b, num_classes, "vertical", "70b", limited_to_40_demos, spearman)
            
            # Compute horizontal correlations (n_relabel vs correlation with n_demos)
            print("\n" + "="*60)
            print("ORIGINAL HORIZONTAL CORRELATIONS (n_relabel vs correlation with n_demos)")
            print("="*60)
            horizontal_correlations_df_1b = compute_horizontal_correlations(results_df_1b, num_classes, limited_to_40_demos, spearman)
            horizontal_correlations_df_8b = compute_horizontal_correlations(results_df_8b, num_classes, limited_to_40_demos, spearman)
            horizontal_correlations_df_70b = compute_horizontal_correlations(results_df_70b, num_classes, limited_to_40_demos, spearman)
            print_horizontal_correlation_stats(horizontal_correlations_df_1b)
            print_horizontal_correlation_stats(horizontal_correlations_df_8b)
            print_horizontal_correlation_stats(horizontal_correlations_df_70b)
            save_correlations_to_csv(horizontal_correlations_df_1b, num_classes, "horizontal", "1b", limited_to_40_demos, spearman)
            save_correlations_to_csv(horizontal_correlations_df_8b, num_classes, "horizontal", "8b", limited_to_40_demos, spearman)
            save_correlations_to_csv(horizontal_correlations_df_70b, num_classes, "horizontal", "70b", limited_to_40_demos, spearman)
            
            print(f"\nOriginal analysis complete!")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
