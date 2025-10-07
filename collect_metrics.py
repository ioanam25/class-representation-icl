#!/usr/bin/env python3
"""
Script to collect all metrics from distributed pickle files and consolidate into a single CSV.
"""

import os
import pickle
import pandas as pd
import re
from pathlib import Path
import sys

def examine_pickle_structure(pickle_path):
    """Examine the structure of a single pickle file to understand the data format."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Examining pickle file: {pickle_path}")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)} - {value if not isinstance(value, (list, dict)) or len(str(value)) < 100 else f'{type(value)} with {len(value)} items'}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item: {data[0] if len(str(data[0])) < 100 else 'Large item'}")
        else:
            print(f"Data: {data if len(str(data)) < 200 else 'Large data structure'}")
        
        return data
        
    except Exception as e:
        print(f"Error reading {pickle_path}: {e}")
        return None

def parse_folder_info(folder_path):
    """Parse folder path to extract n_relabel, demo_id, and run_id."""
    path_parts = Path(folder_path).parts
    
    # Find the relabel folder (e.g., "relabel10_demo0")
    relabel_demo = None
    for part in path_parts:
        if part.startswith('relabel') and 'demo' in part:
            relabel_demo = part
            break
    
    if not relabel_demo:
        return None, None, None
    
    # Parse relabel number and demo number
    match = re.match(r'relabel(\d+)_demo(\d+)', relabel_demo)
    if not match:
        return None, None, None
    
    n_relabel = int(match.group(1))
    demo_id = int(match.group(2))
    
    # Find run folder (e.g., "run_1")
    run_id = None
    for part in path_parts:
        if part.startswith('run_'):
            run_id = int(part.split('_')[1])
            break
    
    return n_relabel, demo_id, run_id

def check_missing_pickles(base_dir, ignore_relabel40=False):
    """Check which folders don't have pickle files for ALL relabel configurations."""
    base_path = Path(base_dir)
    
    print("\n=== Checking for missing pickle files across ALL configurations ===")
    if ignore_relabel40:
        print("NOTE: Ignoring relabel40 configurations as requested")
    
    # Find all relabel folders, optionally excluding relabel40
    relabel_dirs = []
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith('relabel') and '_demo' in d.name:
            # Extract relabel number and optionally skip if it's 40
            relabel_num = int(d.name.split('_demo')[0].replace('relabel', ''))
            if not ignore_relabel40 or relabel_num != 40:
                relabel_dirs.append(d)
    
    relabel_dirs.sort(key=lambda x: (int(x.name.split('_demo')[0].replace('relabel', '')), int(x.name.split('_demo')[1])))
    
    print(f"Found {len(relabel_dirs)} relabel directories to check...")
    
    folders_without_pickles = []
    folders_with_pickles = []
    missing_files_details = []
    
    # Group by relabel number for better reporting
    relabel_groups = {}
    for demo_dir in relabel_dirs:
        parts = demo_dir.name.split('_demo')
        relabel_num = int(parts[0].replace('relabel', ''))
        demo_num = int(parts[1])
        
        if relabel_num not in relabel_groups:
            relabel_groups[relabel_num] = []
        relabel_groups[relabel_num].append((demo_num, demo_dir))
    
    total_expected_files = 0
    total_missing_files = 0
    
    for relabel_num in sorted(relabel_groups.keys()):
        print(f"\n--- Checking relabel{relabel_num} configurations ---")
        relabel_missing = 0
        relabel_present = 0
        
        for demo_num, demo_dir in sorted(relabel_groups[relabel_num]):
            # Check for run_* subdirectories and their metrics.pickle files
            run_dirs = [d for d in demo_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
            
            if not run_dirs:
                print(f"relabel{relabel_num}_demo{demo_num}: NO RUN DIRECTORIES")
                folders_without_pickles.append(f"relabel{relabel_num}_demo{demo_num}")
                missing_files_details.append(f"{demo_dir}: No run directories found")
                continue
            
            # Check if all run directories have metrics.pickle
            missing_pickles = []
            for run_dir in run_dirs:
                pickle_file = run_dir / "metrics.pickle"
                total_expected_files += 1
                if not pickle_file.exists():
                    missing_pickles.append(run_dir.name)
                    total_missing_files += 1
                    missing_files_details.append(f"{pickle_file}")
            
            if missing_pickles:
                print(f"relabel{relabel_num}_demo{demo_num}: MISSING PICKLES in {missing_pickles}")
                folders_without_pickles.append(f"relabel{relabel_num}_demo{demo_num}")
                relabel_missing += 1
            else:
                print(f"relabel{relabel_num}_demo{demo_num}: ✓ All pickles present ({len(run_dirs)} runs)")
                folders_with_pickles.append(f"relabel{relabel_num}_demo{demo_num}")
                relabel_present += 1
        
        print(f"relabel{relabel_num} summary: {relabel_present} complete, {relabel_missing} missing")
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total expected pickle files: {total_expected_files}")
    print(f"Total missing pickle files: {total_missing_files}")
    print(f"Total present pickle files: {total_expected_files - total_missing_files}")
    print(f"Folders with missing pickle files: {len(folders_without_pickles)}")
    print(f"Folders with complete pickle files: {len(folders_with_pickles)}")
    
    if missing_files_details:
        print(f"\n=== DETAILED MISSING FILES LIST ===")
        for missing_file in missing_files_details:
            print(f"MISSING: {missing_file}")
    
    return folders_without_pickles, missing_files_details

def collect_all_metrics(base_dir, ignore_relabel40=False):
    """Collect all metrics from all pickle files in the directory structure."""
    base_path = Path(base_dir)
    all_data = []
    
    # Find all metrics.pickle files
    pickle_files = list(base_path.glob("**/metrics.pickle"))
    
    print(f"Found {len(pickle_files)} pickle files to process...")
    if ignore_relabel40:
        print("NOTE: Skipping relabel40 configurations as requested")
    
    processed = 0
    errors = 0
    skipped_relabel40 = 0
    
    for pickle_file in pickle_files:
        try:
            # Parse folder information
            n_relabel, demo_id, run_id = parse_folder_info(str(pickle_file.parent))
            
            if n_relabel is None:
                print(f"Could not parse folder info from: {pickle_file}")
                errors += 1
                continue
            
            # Optionally skip relabel40 configurations
            if ignore_relabel40 and n_relabel == 40:
                skipped_relabel40 += 1
                continue
            
            # Load the pickle data
            with open(pickle_file, 'rb') as f:
                metrics = pickle.load(f)
            
            # Create a base record with folder information
            base_record = {
                'n_relabel': n_relabel,
                'demo_id': demo_id,
                'run_id': run_id,
                'file_path': str(pickle_file)
            }
            
            # Handle different data structures
            if isinstance(metrics, dict):
                # Check if it's the expected structure with 'metrics' and 'CONFIG' keys
                if 'metrics' in metrics and 'CONFIG' in metrics:
                    record = base_record.copy()
                    
                    # Extract the DataFrame and its attributes
                    df = metrics['metrics']
                    config = metrics['CONFIG']
                    
                    # Extract accuracy from DataFrame attrs if available
                    if hasattr(df, 'attrs'):
                        record.update(df.attrs)
                    
                    # Add config information
                    if isinstance(config, dict):
                        for key, value in config.items():
                            record[f'config_{key}'] = value
                    
                    # Store the DataFrame as string for reference (optional)
                    record['metrics_df_shape'] = str(df.shape)
                    record['metrics_df_columns'] = str(list(df.columns))
                    
                    all_data.append(record)
                else:
                    # If metrics is a general dictionary, flatten it
                    record = base_record.copy()
                    record.update(metrics)
                    all_data.append(record)
            elif isinstance(metrics, list):
                # If metrics is a list, create one record per item
                for i, item in enumerate(metrics):
                    record = base_record.copy()
                    if isinstance(item, dict):
                        record.update(item)
                    else:
                        record[f'metric_{i}'] = item
                    record['list_index'] = i
                    all_data.append(record)
            else:
                # If metrics is a simple value
                record = base_record.copy()
                record['metric_value'] = metrics
                all_data.append(record)
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} files...")
                
        except Exception as e:
            print(f"Error processing {pickle_file}: {e}")
            errors += 1
            continue
    
    if ignore_relabel40 and skipped_relabel40 > 0:
        print(f"Processing complete. Processed: {processed}, Errors: {errors}, Skipped relabel40: {skipped_relabel40}")
    else:
        print(f"Processing complete. Processed: {processed}, Errors: {errors}")
    return all_data

def main():
    ignore_relabel40 = False
    base_dir = "/gpfs/data/oermannlab/users/im2178/class-representation-icl/learning_curves_relabel_demos_5classes_70b/claude_multitask/llama3.1_70b_instruct"
    
    # Check for missing pickle files first
    missing_folders, missing_files_details = check_missing_pickles(base_dir, ignore_relabel40)
    
    # First, examine a sample pickle file
    sample_file = os.path.join(base_dir, "relabel10_demo0/run_1/metrics.pickle")
    print("\n=== Examining sample pickle file ===")
    sample_data = examine_pickle_structure(sample_file)
    print("\n" + "="*50 + "\n")
    
    if sample_data is None:
        print("Could not read sample file. Exiting.")
        return
    
    # Collect all metrics
    print("=== Collecting all metrics ===")
    all_data = collect_all_metrics(base_dir, ignore_relabel40)
    
    if not all_data:
        print("No data collected. Exiting.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Display basic info about the collected data
    print(f"\nCollected data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique n_relabel values: {sorted(df['n_relabel'].unique())}")
    print(f"Demo IDs range: {df['demo_id'].min()} to {df['demo_id'].max()}")
    print(f"Run IDs range: {df['run_id'].min()} to {df['run_id'].max()}")
    
    # Save to CSV
    output_file = os.path.join(base_dir, "consolidated_metrics.csv")
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    main()
