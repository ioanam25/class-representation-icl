"""
Generate relabelings with imbalanced class distributions.

For claude_multitask dataset:
  3-class (Joy/Anger/Fear = A/C/D): 60/30/10 ratio
  5-class (Joy/Sadness/Anger/Fear/Surprise = A/B/C/D/E): 40/20/20/10/10 ratio
"""
from LLMGeometry.datasets import load_dataset_by_name
from LLMGeometry import load_model_and_tokenizer
import torch
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import relabeling
from relabeling import optimize_tokens
import argparse


# Default imbalanced ratios
CLASS_RATIOS_3 = {'A': 0.6, 'C': 0.3, 'D': 0.1}  # Joy: 60%, Anger: 30%, Fear: 10%
CLASS_RATIOS_5 = {'A': 0.4, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.1}  # Joy/Sadness/Anger/Fear/Surprise


def sample_imbalanced(df, label_column, n_total, class_ratios, seed=None):
    """Sample n_total examples from df with class distribution according to class_ratios."""
    rng = np.random.RandomState(seed)
    
    # Calculate samples per class
    sorted_labels = sorted(class_ratios.keys(), key=lambda x: -class_ratios[x])
    n_per_class = {}
    total_assigned = 0
    
    for i, lab in enumerate(sorted_labels):
        if i == len(sorted_labels) - 1:
            n_per_class[lab] = n_total - total_assigned
        else:
            n_per_class[lab] = round(n_total * class_ratios[lab])
            total_assigned += n_per_class[lab]
    
    sampled_indices = []
    for lab, n_samples in n_per_class.items():
        lab_indices = df.index[df[label_column] == lab].tolist()
        n_samples = min(n_samples, len(lab_indices))
        if n_samples > 0:
            chosen = rng.choice(lab_indices, size=n_samples, replace=False)
            sampled_indices.extend(chosen.tolist())
    
    return df.loc[sampled_indices].reset_index(drop=True)


def main(MODEL_NAME='mistral_7b_base', 
         DATASET_NAME='claude_multitask',
         num_classes=3,
         n_relabel_list=None,
         n_runs=1,
         top_tokens=128256,
         whole_words_only=True,
         ensemble_assignment=False,
         ensemble_method='voting',
         ensemble_temperature=0,
         base_seed=42,
         class_ratios=None,
         logits_dir=None,
         label_column='emotion_letter'):
    
    if n_relabel_list is None:
        n_relabel_list = list(range(10, 101, 10))
    
    # Set default class ratios based on num_classes
    if class_ratios is None:
        if num_classes == 3:
            class_ratios = CLASS_RATIOS_3
        elif num_classes == 5:
            class_ratios = CLASS_RATIOS_5
        else:
            raise ValueError(f"No default class_ratios for {num_classes} classes")
    
    print(f"Imbalanced class ratios: {class_ratios}")
    
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
    
    # Split training data in half (same split as balanced experiments)
    relabeling_df = train_df.sample(frac=0.5, random_state=base_seed)
    relabeling_df = relabeling_df.reset_index(drop=True)
    
    if num_classes == 3:
        print("Filtering for 3 emotions: Joy, Anger, Fear")
        relabeling_df = relabeling_df[relabeling_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        relabeling_df = relabeling_df.reset_index(drop=True)
        print(f"After filtering: {len(relabeling_df)} examples")
        print("Class distribution:")
        print(relabeling_df[label_column].value_counts())
    
    # Determine logits directory
    if logits_dir is None:
        # Use the same convention as run_ICL_relabel.py
        if MODEL_NAME == 'mistral_7b_base':
            logits_dir = 'mistral_7b_base'
        elif MODEL_NAME == 'qwen2_7b_base':
            logits_dir = 'qwen2_7b_base'
        else:
            logits_dir = f'{MODEL_NAME}_{DATASET_NAME}'
    
    # Load precomputed logits
    print(f"\nLoading precomputed logits from {logits_dir}...")
    file_suffix = '_whole_words' if whole_words_only else ''
    with open(f'{logits_dir}/template_sentence_probs_{top_tokens}{file_suffix}.pkl', 'rb') as f:
        sentence_probs = pickle.load(f)
    with open(f'{logits_dir}/template_sentence_logits_{top_tokens}{file_suffix}.pkl', 'rb') as f:
        sentence_logits = pickle.load(f)
    
    # Get tokens
    tokens = sorted_vocab[:top_tokens]
    all_tokens_str = [token[0] for token in tokens]
    
    # Initialize random seeds
    run_seeds = np.random.randint(0, 2**32-1, size=n_runs)
    
    # Save directory - includes _imbalanced suffix
    save_dir = Path(f"{MODEL_NAME}_relabelings_imbalanced")
    save_dir.mkdir(exist_ok=True)
    
    # Set MODEL_NAME in relabeling module so whole-word filtering uses the correct prefix
    relabeling.MODEL_NAME = MODEL_NAME
    
    # Generate relabelings for each n_relabel
    for n_relabel in n_relabel_list:
        print(f"\n=== Generating IMBALANCED relabeling for {n_relabel} examples ===")
        
        # Sample with imbalanced ratios
        run_examples = sample_imbalanced(
            relabeling_df, label_column, n_relabel, class_ratios, seed=run_seeds[0]
        )
        sentences = run_examples['text']
        labels = run_examples[label_column]
        
        print(f"Selected {len(run_examples)} examples (target: {n_relabel})")
        print("Label distribution:")
        print(labels.value_counts())
        
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
        
        # Config for this relabeling
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
            'base_seed': base_seed,
            'class_ratios': class_ratios,
            'imbalanced': True
        }
        
        save_path = save_dir / f"{MODEL_NAME}_relabelings_{num_classes}classes_{top_tokens}toptokens_isensembled{ensemble_assignment}_{ensemble_method}_{n_relabel}examples_{n_runs}runs_imbalanced.pkl"
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
    
    print("\nAll imbalanced relabelings generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate imbalanced relabelings for ICL experiments")
    parser.add_argument("--model", default="mistral_7b_base", help="Model name")
    parser.add_argument("--dataset", default="claude_multitask", help="Dataset name")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (3 or 5)")
    parser.add_argument("--n_relabel_list", type=str, default="10,20,30,40,50,60,70,80,90,100", 
                      help="Comma-separated list of n_relabel values")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of relabelings to generate")
    parser.add_argument("--top_tokens", type=int, default=128256, help="Number of top tokens")
    parser.add_argument("--whole_words_only", action='store_true', default=True, help="Use only whole word tokens")
    parser.add_argument("--ensemble_assignment", action='store_true', default=False, help="Use ensemble assignment")
    parser.add_argument("--ensemble_method", default="voting", help="Ensemble method")
    parser.add_argument("--ensemble_temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label_column", default="emotion_letter", help="Column name for class labels")
    parser.add_argument("--logits_dir", default=None, help="Directory with precomputed logits (auto-detected if not set)")
    
    args = parser.parse_args()
    
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
        base_seed=args.seed,
        label_column=args.label_column,
        logits_dir=args.logits_dir
    )
