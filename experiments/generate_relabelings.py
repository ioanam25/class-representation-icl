from LLMGeometry.datasets import load_dataset_by_name
from LLMGeometry import load_model_and_tokenizer
import torch
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
from relabeling import optimize_tokens
import argparse

def main(MODEL_NAME='llama3.1_base', 
         DATASET_NAME='claude_multitask',
         num_classes=3,
         n_relabel_list=None,  # Now takes a list of n_relabel values
         n_runs=1,
         top_tokens=128256,
         whole_words_only=True,
         ensemble_assignment=True,
         ensemble_method='logit_averaging',
         ensemble_temperature=0,
         base_seed=42):
    
    if n_relabel_list is None:
        n_relabel_list = list(range(10, 101, 10))  # Default: 10, 20, ..., 100
    
    # Set random seed
    np.random.seed(base_seed)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Load dataset
    print("Loading dataset...")
    datasets = load_dataset_by_name(DATASET_NAME)
    train_df = datasets['train']
    
    # Split training data in half
    relabeling_df = train_df.sample(frac=0.5, random_state=base_seed)
    relabeling_df = relabeling_df.reset_index(drop=True)
    
    if num_classes == 3:
        # Keep only Joy, Anger, Fear
        print("Filtering for 3 emotions: Joy, Anger, Fear")
        relabeling_df = relabeling_df[relabeling_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        relabeling_df = relabeling_df.reset_index(drop=True)
        print(f"After filtering: {len(relabeling_df)} examples")
        print("Class distribution:")
        print(relabeling_df['emotion'].value_counts())
    
    # Load precomputed logits
    print("\nLoading precomputed logits...")
    file_suffix = '_whole_words' if whole_words_only else ''
    with open(f'sentence_info/template_sentence_probs_{top_tokens}{file_suffix}.pkl', 'rb') as f:
        sentence_probs = pickle.load(f)
    with open(f'sentence_info/template_sentence_logits_{top_tokens}{file_suffix}.pkl', 'rb') as f:
        sentence_logits = pickle.load(f)
    
    # Get tokens
    tokens = sorted_vocab[:top_tokens]
    all_tokens_str = [token[0] for token in tokens]
    
    # Initialize random seeds
    run_seeds = np.random.randint(0, 2**32-1, size=n_runs)
    
    # Generate multiple relabeling dictionaries
    for n_relabel in n_relabel_list:
        print(f"\n=== Generating relabeling for {n_relabel} examples ===")
        
        # Randomly select n_relabel examples
        run_examples = relabeling_df.sample(n=n_relabel, random_state=run_seeds[0])  # Use first seed for consistency
        sentences = run_examples['text']
        labels = run_examples['emotion_letter']
        
        print(f"Selected {n_relabel} examples")
        print("Label distribution:", labels.value_counts())
        
        # Generate relabeling
        new_labels, objective = optimize_tokens(
            list(all_tokens_str), 
            sentences, 
            labels, 
            sentence_logits,
            tokenizer=tokenizer,
            num_restarts=10,
            ensemble_assignment=ensemble_assignment,
            ensemble_method=ensemble_method,
            ensemble_temperature=ensemble_temperature,
            whole_words_only=whole_words_only
        )
        
        # Save results for this n_relabel
        save_dir = Path("relabelings")
        save_dir.mkdir(exist_ok=True)
        
        config = {
            'MODEL_NAME': MODEL_NAME,
            'DATASET_NAME': DATASET_NAME,
            'num_classes': num_classes,
            'n_relabel': n_relabel,
            'n_runs': n_runs,
            'top_tokens': top_tokens,
            'whole_words_only': whole_words_only,
            'ensemble_assignment': ensemble_assignment,
            'ensemble_method': ensemble_method,
            'ensemble_temperature': ensemble_temperature,
            'base_seed': base_seed
        }
        
        save_path = save_dir / f"relabelings_{top_tokens}toptokens_isensembled{ensemble_assignment}_{ensemble_method}_{n_relabel}examples_{n_runs}runs.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                'config': config,
                'relabelings': [{
                    'labels': new_labels,
                    'objective': objective,
                    'examples': {
                        'sentences': sentences.tolist(),
                        'labels': labels.tolist()
                    },
                    'run_seed': run_seeds[0]
                }]
            }, f)
        
        print(f"Generated relabeling with objective: {objective}")
        print("Label mappings:", new_labels)
        print(f"Saved to {save_path}")
        
        # Print statistics for this n_relabel
        print(f"\nStatistics for n_relabel = {n_relabel}:")
        print(f"Objective: {objective:.4f}")
    
    print("\nAll relabelings generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1_base", help="Model name")
    parser.add_argument("--dataset", default="claude_multitask", help="Dataset name")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--n_relabel_list", type=str, default="10,20,30,40,50,60,70,80,90,100", 
                      help="Comma-separated list of n_relabel values")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of relabelings to generate per n_relabel")
    parser.add_argument("--top_tokens", type=int, default=10000, help="Number of top tokens to use")
    parser.add_argument("--whole_words_only", type=bool, default=True, help="Whether to use only whole word tokens")
    parser.add_argument("--ensemble_assignment", type=bool, default=True, help="Whether to use ensemble assignment")
    parser.add_argument("--ensemble_method", default="voting", help="Ensemble method (voting or logit_averaging)")
    parser.add_argument("--ensemble_temperature", type=float, default=0, help="Temperature for logit averaging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Convert n_relabel_list from string to list of integers
    n_relabel_list = [int(x) for x in args.n_relabel_list.split(',')]
    
    main(
        MODEL_NAME=args.model,
        DATASET_NAME=args.dataset,
        num_classes=args.num_classes,
        n_relabel_list=n_relabel_list,
        n_runs=args.n_runs,
        top_tokens=args.top_tokens,
        whole_words_only=args.whole_words_only,
        ensemble_assignment=args.ensemble_assignment,
        ensemble_method=args.ensemble_method,
        ensemble_temperature=args.ensemble_temperature,
        base_seed=args.seed
    ) 