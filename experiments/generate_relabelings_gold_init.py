"""
Generate relabelings initialized from gold (semantically meaningful) labels
instead of random tokens. The hill-climbing optimization starts from the gold
labels and tries to find something better.

Usage:
  cd experiments/
  export PYTHONPATH=/gpfs/data/oermannlab/users/im2178/class-representation-icl/src:$PYTHONPATH

  # Sentiment 3-class
  python generate_relabelings_gold_init.py --model qwen2_7b_base --dataset claude_multitask \
      --num_classes 3 --logits_dir qwen2_7b_base --label_column emotion_letter

  # Sentiment 5-class
  python generate_relabelings_gold_init.py --model qwen2_7b_base --dataset claude_multitask \
      --num_classes 5 --logits_dir qwen2_7b_base --label_column emotion_letter

  # TREC 5-class
  python generate_relabelings_gold_init.py --model qwen2_7b_base --dataset TREC_coarse \
      --num_classes 5 --logits_dir qwen2_7b_base_TREC_coarse --label_column category_letter
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


# Gold label definitions
# Maps: class_letter -> gold word (lowercase, without tokenizer prefix)
GOLD_LABELS = {
    'claude_multitask': {
        3: {'A': 'joy', 'C': 'anger', 'D': 'fear'},
        5: {'A': 'joy', 'B': 'sadness', 'C': 'anger', 'D': 'fear', 'E': 'surprise'},
    },
    'TREC_coarse': {
        5: {'A': 'entity', 'B': 'description', 'C': 'human', 'D': 'location', 'E': 'numeric'},
    },
}


def find_gold_token(word, top_k_tokens, model_name):
    """
    Find the tokenizer-format version of a gold label word in the vocabulary.
    For Qwen (and Llama), whole word tokens start with 'Ġ'.
    For Mistral, they start with '▁'.
    """
    if model_name == 'mistral_7b_base':
        prefix = '▁'
    else:
        prefix = 'Ġ'
    
    # Try with prefix (standard whole-word token)
    candidate = prefix + word
    if candidate in top_k_tokens:
        return candidate
    
    # Try lowercase with prefix
    candidate = prefix + word.lower()
    if candidate in top_k_tokens:
        return candidate
    
    # Try without prefix (rare but possible)
    if word in top_k_tokens:
        return word
    if word.lower() in top_k_tokens:
        return word.lower()
    
    # Try capitalized
    candidate = prefix + word.capitalize()
    if candidate in top_k_tokens:
        return candidate
    
    return None


def main(MODEL_NAME='qwen2_7b_base',
         DATASET_NAME='claude_multitask',
         num_classes=3,
         n_relabel_list=None,
         top_tokens=128256,
         whole_words_only=True,
         ensemble_assignment=False,
         ensemble_method='voting',
         ensemble_temperature=0,
         base_seed=42,
         logits_dir=None,
         label_column='emotion_letter',
         num_restarts=10):

    if n_relabel_list is None:
        n_relabel_list = list(range(10, 101, 10))

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

    # Split training data in half (same split as other experiments)
    relabeling_df = train_df.sample(frac=0.5, random_state=base_seed)
    relabeling_df = relabeling_df.reset_index(drop=True)

    if DATASET_NAME == 'claude_multitask' and num_classes == 3:
        print("Filtering for 3 emotions: Joy, Anger, Fear")
        relabeling_df = relabeling_df[relabeling_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        relabeling_df = relabeling_df.reset_index(drop=True)
        print(f"After filtering: {len(relabeling_df)} examples")

    print("Class distribution:")
    print(relabeling_df[label_column].value_counts())

    # Load precomputed logits
    if logits_dir is None:
        logits_dir = f'{MODEL_NAME}_{DATASET_NAME}'
    print(f"\nLoading precomputed logits from {logits_dir}...")
    file_suffix = '_whole_words' if whole_words_only else ''
    with open(f'{logits_dir}/template_sentence_probs_{top_tokens}{file_suffix}.pkl', 'rb') as f:
        sentence_probs = pickle.load(f)
    with open(f'{logits_dir}/template_sentence_logits_{top_tokens}{file_suffix}.pkl', 'rb') as f:
        sentence_logits = pickle.load(f)

    # Get tokens
    tokens = sorted_vocab[:top_tokens]
    all_tokens_str = [token[0] for token in tokens]

    # Set MODEL_NAME in relabeling module for correct prefix filtering
    relabeling.MODEL_NAME = MODEL_NAME

    # --- Build gold label token mapping ---
    gold_words = GOLD_LABELS.get(DATASET_NAME, {}).get(num_classes, {})
    if not gold_words:
        raise ValueError(f"No gold labels defined for {DATASET_NAME} with {num_classes} classes")

    print("\n=== Gold Label Token Resolution ===")
    initial_labels = {}
    for class_letter, word in gold_words.items():
        token = find_gold_token(word, all_tokens_str, MODEL_NAME)
        if token is None:
            raise ValueError(f"Could not find token for gold label '{word}' (class {class_letter}) in vocabulary")
        initial_labels[class_letter] = token
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {class_letter} -> '{word}' -> token '{token}' (id={token_id})")

    # Initialize random seeds
    run_seeds = np.random.randint(0, 2**32-1, size=1)

    # Generate relabelings for each n_relabel
    save_dir = Path(f"{MODEL_NAME}_{DATASET_NAME}_relabelings_gold_init")
    save_dir.mkdir(exist_ok=True)

    for n_relabel in n_relabel_list:
        print(f"\n{'='*60}")
        print(f"=== Gold-init relabeling for {n_relabel} examples ===")
        print(f"{'='*60}")

        # Sample balanced examples from each class
        samples_per_class = n_relabel // num_classes
        sampled_indices = []
        for cls, group in relabeling_df.groupby(label_column):
            sampled_indices.extend(
                group.sample(n=min(samples_per_class, len(group)), random_state=run_seeds[0]).index.tolist()
            )
        run_examples = relabeling_df.loc[sampled_indices].reset_index(drop=True)
        sentences = run_examples['text']
        labels = run_examples[label_column]

        print(f"Selected {len(run_examples)} examples")
        print("Label distribution:", labels.value_counts().to_dict())

        # Generate relabeling with gold-label initialization
        new_labels, objective = optimize_tokens(
            list(all_tokens_str),
            sentences,
            labels,
            sentence_logits,
            tokenizer=tokenizer,
            num_restarts=num_restarts,
            ensemble_assignment=ensemble_assignment,
            ensemble_method=ensemble_method,
            ensemble_temperature=ensemble_temperature,
            whole_words_only=whole_words_only,
            initial_labels=initial_labels,  # <-- THE KEY CHANGE
        )

        # Check if gold labels were preserved or changed
        print("\n=== Comparison: Gold vs Optimized ===")
        for class_letter in sorted(initial_labels.keys()):
            gold_tok = initial_labels[class_letter]
            opt_tok = new_labels[class_letter][0] if class_letter in new_labels else '???'
            changed = "CHANGED" if gold_tok != opt_tok else "KEPT"
            print(f"  {class_letter}: gold='{gold_tok}' -> optimized='{opt_tok}' [{changed}]")

        # Save results
        config = {
            'MODEL_NAME': MODEL_NAME,
            'DATASET_NAME': DATASET_NAME,
            'num_classes': num_classes,
            'n_relabel': n_relabel,
            'top_tokens': top_tokens,
            'whole_words_only': whole_words_only,
            'ensemble_assignment': ensemble_assignment,
            'ensemble_method': ensemble_method,
            'ensemble_temperature': ensemble_temperature,
            'base_seed': base_seed,
            'gold_init': True,
            'gold_labels': {k: v for k, v in initial_labels.items()},
            'num_restarts': num_restarts,
        }

        save_path = save_dir / f"{MODEL_NAME}_relabelings_{num_classes}classes_{n_relabel}examples_gold_init.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                'config': config,
                'new_labels': new_labels,
                'objective': objective,
                'initial_labels': initial_labels,
                'gold_words': gold_words,
            }, f)

        print(f"Saved to {save_path}")

    print("\n\nAll gold-init relabelings generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate relabelings with gold-label initialization")
    parser.add_argument("--model", default="qwen2_7b_base", help="Model name")
    parser.add_argument("--dataset", default="claude_multitask", help="Dataset name")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--n_relabel_list", type=str, default="10,20,30,40,50,60,70,80,90,100",
                        help="Comma-separated list of n_relabel values")
    parser.add_argument("--top_tokens", type=int, default=128256)
    parser.add_argument("--whole_words_only", action='store_true', default=True)
    parser.add_argument("--ensemble_assignment", action='store_true', default=False)
    parser.add_argument("--ensemble_method", default="voting")
    parser.add_argument("--ensemble_temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logits_dir", type=str, default=None,
                        help="Directory containing precomputed logits (default: {model}_{dataset})")
    parser.add_argument("--label_column", type=str, default="emotion_letter",
                        help="Column name for class labels")
    parser.add_argument("--num_restarts", type=int, default=10,
                        help="Number of restarts for hill climbing")

    args = parser.parse_args()
    n_relabel_list = [int(x) for x in args.n_relabel_list.split(',')]

    main(
        MODEL_NAME=args.model,
        DATASET_NAME=args.dataset,
        num_classes=args.num_classes,
        n_relabel_list=n_relabel_list,
        top_tokens=args.top_tokens,
        whole_words_only=args.whole_words_only,
        ensemble_assignment=args.ensemble_assignment,
        ensemble_method=args.ensemble_method,
        ensemble_temperature=args.ensemble_temperature,
        base_seed=args.seed,
        logits_dir=args.logits_dir,
        label_column=args.label_column,
        num_restarts=args.num_restarts,
    )
