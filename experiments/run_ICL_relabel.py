from LLMGeometry.datasets import load_dataset_by_name
from LLMGeometry.in_context_learning import ICL_Template, load_ICL_template, list_ICL_templates_in_json
from LLMGeometry import load_model_and_tokenizer
from LLMGeometry.evaluation import run_on_dataframe
from LLMGeometry.utils import generate_random_samples, save_file_with_incremental_suffix
import torch
import pandas as pd
import pickle
from termcolor import colored
from pathlib import Path
import argparse
import json
import sys
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
from relabeling import optimize_tokens, template_new_labels
from sklearn.model_selection import train_test_split

def save_outputs(output, save_folder, CONFIG):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    with open(save_file_with_incremental_suffix(save_folder / "metrics.pickle"), "wb") as f:
        pickle.dump({
            'metrics' : output['metrics'],
            'CONFIG': CONFIG,
        }, f)

    with open(save_file_with_incremental_suffix(save_folder / "embeddings.pickle"), "wb") as f:
        pickle.dump({
            'metrics' : output['metrics'],
            # 'mean_pooled_embeddings' : output['mean_pooled_embeddings'],
            # 'last_token_embeddings' : output['last_token_embeddings'],
            'batches' : output['batches'],
            'CONFIG': CONFIG,
        }, f)


def create_template(train_df, prefix_type, n_examples, keyword, answer_field, dataset_name, shuffle_labels=False, seed=None):
    '''
        Create a template for the ICL prompt.
        
        Parameters:
        ----------
        prefix_type: str
            The type of the prefix. Can be 'raw', 'instruction', 'demos'.
        n_examples: int
            The number of examples to include in the prefix.
        keyword: str
            The keyword to be used in the template.
        answer_field: str
            The name of the answer field.
        dataset_name: str
            The name of the dataset.
        shuffle_labels: bool
            Whether to shuffle the labels.
        seed: int
            The seed for the random number generator.
        
        Returns:
        -------
        prefix: str
            The prefix of the ICL prompt.
        suffix: str
            The suffix of the ICL prompt
    '''
    # ds = load_dataset_by_name(dataset_name)
    # train_df = ds['train']
    category_list = train_df[answer_field].unique()
    # --- Creating the prefix 
    if prefix_type == 'raw':
        prefix = ''
    elif prefix_type=='instruction':
        prefix = 'This is a text classification task. Possible categories are: ' + ', '.join(category_list) + '.\n'
    elif prefix_type == 'demos':
        chosen_indices = generate_random_samples(train_df[answer_field], n_examples, seed=seed)
        chosen_sentences = train_df.loc[chosen_indices,'text']
        chosen_labels = train_df.loc[chosen_indices, answer_field]
        if shuffle_labels:
            chosen_labels = chosen_labels.sample(frac=1).reset_index(drop=True) # Shuffle the labels
        prefix = ''
        for sentence, label in zip(chosen_sentences, chosen_labels):
            prefix += 'Text: ' + sentence + f'\n{keyword}: ' + label + '\n'
    # --- Creating the suffix
    suffix = f'\n{keyword}:'
    return prefix, suffix, chosen_sentences, chosen_labels

def W_ICL(top_k_tokens, sentences, labels, sentence_probs):
    W = {}
    for class_label in labels.unique(): # C 
        W[class_label] = {}
        for i, token in enumerate(top_k_tokens): # V  
            W[class_label][token] = 0

    not_zero = 0
    total = 0
    for class_label in labels.unique(): # C
        for i, token in enumerate(top_k_tokens): # V
            sum = 0
            for sentence, label in zip(sentences, labels): # N
                prob = sentence_probs[sentence][i]
                log_prob = torch.log(prob)
                # one_minus_log_prob = torch.log(1 - prob)
                one_minus_log_prob = -log_prob
                if label == class_label:
                    sum += log_prob
                else:
                    sum += one_minus_log_prob
                    total += 1
                    if one_minus_log_prob != 0:
                        not_zero += 1
            W[class_label][token] = sum.item()
    print("not_zero: ", not_zero, "total: ", total)
    return W

def reassign_labels(all_tokens_str, labels, sentences, sentence_probs, tokenizer):
    num_to_label = {}
    num_to_class = {}
    W = W_ICL(all_tokens_str, sentences, labels, sentence_probs)
    weights = []
    
    for (i, v1) in enumerate(W):
        num_to_class[i] = v1
        for (j, v2) in enumerate(W[v1]):
            num_to_label[j] = v2

    weights = []
    for (i, v1) in enumerate(W):
        weights.append([])
        for (j, v2) in enumerate(W[v1]):
            weights[i].append(W[v1][v2])
    cost = np.array(weights)

    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

    new_labels = {}
    for i in range(len(row_ind)):
        token_str = num_to_label[col_ind[i]]
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        
        print(f"Class: {num_to_class[row_ind[i]]}")
        print(f"  Token string: '{token_str}'")
        print(f"  Token ID: {token_id}")
        print(f"  Back to string: '{tokenizer.convert_ids_to_tokens(token_id)}'")
        
        new_labels[num_to_class[row_ind[i]]] = (token_str, token_id)
    print("new_labels: ", new_labels)
    print("sum of cost: ", cost[row_ind, col_ind].sum())
    return new_labels

def main(MODEL_NAME, DATASET_NAME, num_classes, prefix_type, n_examples, n_relabel, keyword, answer_field, N_RUNS=50, 
         root_folder="DATA/ICL_stability/results", ensemble_assignment=False, ensemble_method='voting', 
         ensemble_temperature=1.0, top_tokens=-1, whole_words_only=False, base_seed=42):
    
    # Set a base seed that will be used to generate per-run seeds
    np.random.seed(base_seed)
    # Generate fixed seeds for each run
    run_seeds = np.random.randint(0, 2**32-1, size=N_RUNS)

    root_folder = Path(root_folder)

    # --- Loading the model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Define available token sizes
    token_sizes = [1000, 2000, 3000, 4000, 5000, 10000, 128256]
    if top_tokens not in token_sizes and top_tokens != -1:  # -1 for all tokens
        raise ValueError(f"top_tokens must be one of {token_sizes} or -1 for all tokens")
    
    top_tokens_dict = {size: sorted_vocab[:size] for size in token_sizes}
    
    if top_tokens in token_sizes:
        # Load precomputed data for specific token size
        file_suffix = '_whole_words' if whole_words_only else ''
        with open(f'sentence_info/template_sentence_probs_{top_tokens}{file_suffix}.pkl', 'rb') as f:
            sentence_probs = pickle.load(f)
        with open(f'sentence_info/template_sentence_logits_{top_tokens}{file_suffix}.pkl', 'rb') as f:
            sentence_logits = pickle.load(f)
        tokens = top_tokens_dict[top_tokens]
        all_tokens = [token[1] for token in tokens]  # Get token IDs
        all_tokens_str = [token[0] for token in tokens]  # Get token strings

    # --- Loading the dataset
    datasets = load_dataset_by_name(DATASET_NAME)
    train_df = datasets['train']
    test_df = datasets['test']
    
    # Split training data in half
    relabeling_df = train_df.sample(frac=0.5, random_state=base_seed)
    demonstrations_df = train_df.drop(relabeling_df.index)
    
    # Reset indices
    relabeling_df = relabeling_df.reset_index(drop=True)
    demonstrations_df = demonstrations_df.reset_index(drop=True)
    
    print(f"Split training data:")
    print(f"  Relabeling set: {len(relabeling_df)} examples")
    print(f"  Demonstrations set: {len(demonstrations_df)} examples")

    if num_classes == 3:
        # keep only the rows where the emotion is Joy, Anger, or Fear and reindex the dataframe
        print("\nFiltering for 3 emotions (Joy, Anger, Fear)...")
        relabeling_df = relabeling_df[relabeling_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        relabeling_df = relabeling_df.reset_index(drop=True)
        demonstrations_df = demonstrations_df[demonstrations_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        demonstrations_df = demonstrations_df.reset_index(drop=True)
        test_df = test_df[test_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        test_df = test_df.reset_index(drop=True)
        print(f"After filtering:")
        print(f"  Relabeling set: {len(relabeling_df)} examples")
        print(f"  Demonstrations set: {len(demonstrations_df)} examples")
        print(f"  Test set: {len(test_df)} examples")

    # --- Adding space to the answer field if needed
    if tokenizer.name_or_path in ['mistralai/Mistral-7B-v0.3']:
        pass # Mistral uses sentencepiece tokenizer, which does not require space before the answer
    else:
        test_df[answer_field] = " " + (test_df[answer_field].astype(str)) # Adding space before the answer
        if answer_field.endswith('_shuffled'):
            test_df[f'{answer_field[:-9]}'] = " " + (test_df[f'{answer_field[:-9]}'].astype(str)) # Adding space before the original category if we are running with category_shuffle
    
    # --- Creating the ICL template
    if prefix_type != 'demos':
        N_RUNS = 1 # If the prefix is raw or instruction, we run ICL only once with 0 examples in context

    # Create output folders with both n_relabel and n_examples in the path
    folder_with_runs = (root_folder / f"{DATASET_NAME}" / f"{MODEL_NAME}" /
                        f"relabel{n_relabel}_demo{n_examples}")
    # folder_with_runs_gold = (root_folder / f"{DATASET_NAME}" / f"{MODEL_NAME}" / f"{answer_field}" / 
    #                        f"{prefix_type}" / f"gold" / f"relabel{n_relabel}_demo{n_examples}")
    
    # Create directories if they don't exist
    folder_with_runs.mkdir(parents=True, exist_ok=True)
    # folder_with_runs_gold.mkdir(parents=True, exist_ok=True)
    
    n_existing_runs = len(list(folder_with_runs.glob("run_*"))) # Number of existing runs

    # Load precomputed relabeling
    # relabeling_path = Path(f"relabelings/relabelings_{n_relabel}examples_1runs.pkl")
    relabeling_path = Path(f"relabelings/relabelings_10000toptokens_isensembledTrue_voting_{n_relabel}examples_1runs.pkl")
    if not relabeling_path.exists():
        raise ValueError(f"No relabeling file found for n_relabel={n_relabel}. Please run generate_relabelings.py first.")
    
    print(f"\nLoading relabeling scheme from {relabeling_path}")
    with open(relabeling_path, 'rb') as f:
        relabeling_data = pickle.load(f)
        
    # Verify the relabeling configuration matches
    relabeling_config = relabeling_data['config']
    if relabeling_config['MODEL_NAME'] != MODEL_NAME:
        raise ValueError(f"Relabeling model ({relabeling_config['MODEL_NAME']}) doesn't match current model ({MODEL_NAME})")
    if relabeling_config['DATASET_NAME'] != DATASET_NAME:
        raise ValueError(f"Relabeling dataset ({relabeling_config['DATASET_NAME']}) doesn't match current dataset ({DATASET_NAME})")
    
    # Get the relabeling (using first run if multiple runs exist)
    new_labels = relabeling_data['relabelings'][0]['labels']
    print("\nUsing relabeling scheme:")
    for orig_label, (new_token, token_id) in new_labels.items():
        print(f"  {orig_label} -> {new_token} (ID: {token_id})")

    if num_classes == 3:
        gold_labels = {'A': ('Ġjoy', 16267), 'C': ('Ġanger', 19788), 'D': ('Ġfear', 8850)}
    elif num_classes == 5:
        gold_labels = {'A': ('Ġjoy', tokenizer.convert_tokens_to_ids('Ġjoy')), 
                       'B': ('Ġsadness', tokenizer.convert_tokens_to_ids('Ġsadness')), 
                       'C': ('Ġanger', tokenizer.convert_tokens_to_ids('Ġanger')), 
                       'D': ('Ġfear', tokenizer.convert_tokens_to_ids('Ġfear')), 
                       'E': ('Ġsurprise', tokenizer.convert_tokens_to_ids('Ġsurprise'))}

    for k in range(N_RUNS):
        # --- Running ICL
        print(colored(f"\nRunning ICL - Relabeling: {n_relabel} examples, Demonstrations: {n_examples}, Run: {k+1}/{N_RUNS}", 'magenta'))
        print(colored(f"Results will be saved to: {folder_with_runs}", 'cyan'))

        # --- Creating the ICL prompt, by selecting random examples from the demonstrations set
        prefix, suffix, sentences, labels = create_template(demonstrations_df, prefix_type, n_examples, keyword, answer_field, DATASET_NAME, seed=run_seeds[k])
        print("\nSelected demonstration examples:")
        for s, l in zip(sentences, labels):
            print(f"Text: {s[:50]}... | Label: {l}")

        prefix, suffix, prefix_tokens = template_new_labels(tokenizer, sentences, labels, new_labels, keyword)
        # prefix_gold, suffix_gold, prefix_tokens_gold = template_new_labels(tokenizer, sentences, labels, gold_labels, keyword)

        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        # suffix_tokens_gold = tokenizer.encode(suffix_gold, add_special_tokens=False)
        print('Suffix tokens:', [tokenizer.decode([t]) for t in suffix_tokens])
        suffix_crop = len(suffix_tokens) - 1 # Number of tokens in the suffix to crop (-1 because \n will get fused with "." token at the end of the sentence)

        test_df['text_with_suffix'] = 'Text: ' + test_df['text'] + suffix
        # test_df['text_with_suffix_gold'] = 'Text: ' + test_df['text'] + suffix_gold
        test_df['token_relabel_str'] = test_df[answer_field].apply(lambda x: new_labels[x[1:]][0])
        test_df['token_relabel_id'] = test_df[answer_field].apply(lambda x: new_labels[x[1:]][1])
        test_df['gold_label_str'] = test_df[answer_field].apply(lambda x: gold_labels[x[1:]][0])
        test_df['gold_label_id'] = test_df[answer_field].apply(lambda x: gold_labels[x[1:]][1])
        # print("test_df['token_relabel_str']: ", test_df['token_relabel_str'])
        # print("test_df['token_relabel_id']: ", test_df['token_relabel_id'])

        output = run_on_dataframe(test_df, model, tokenizer, 'text_with_suffix', answer_field='token_relabel_id', 
                         return_hidden_states=True, batch_size=10,
                         prefix_tokens=prefix_tokens,  # Use prefix_tokens instead of prefix_text
                         return_raw_batches=True, new_labels=new_labels)
        # output_gold = run_on_dataframe(test_df, model, tokenizer, 'text_with_suffix_gold', answer_field='gold_label_id', 
        #                          return_hidden_states=True, batch_size=10,
        #                          prefix_tokens=prefix_tokens_gold,  # Use prefix_tokens instead of prefix_text
        #                          return_raw_batches=True, new_labels=gold_labels)
        # --- Computing the metrics
        if answer_field.endswith('_shuffled'):
            # If we are running with _shuffled, also compute the accuracy for the original category
            original_field = answer_field[:-9]
            output['metrics'].attrs['accuracy_original'] = accuracy_score(
                output['metrics'][original_field].apply(lambda x: tokenizer.encode(x)[-1]),
                output['metrics'].highest_prob_token
            )
            output['metrics'].attrs['f1_original'] = f1_score(
                output['metrics'][original_field].apply(lambda x: tokenizer.encode(x)[-1]),
                output['metrics'].highest_prob_token,
                average='weighted'
            )

        # output['mean_pooled_embeddings'] = torch.stack([s[:, :-suffix_crop, :].mean(dim=1).float().cpu() for s in output['hidden_states']]) # Mean pooling over the tokens, excluding the suffix
        # output['last_token_embeddings'] = torch.stack([s[:, -1, :].float().cpu() for s in output['hidden_states']]) # Last token embeddings

        CONFIG = {
            'MODEL_NAME': MODEL_NAME,
            'DATASET_NAME': DATASET_NAME,
            'num_classes': num_classes,
            'prefix_type': prefix_type,
            'n_examples': n_examples,
            'n_relabel': n_relabel,  # Add n_relabel to CONFIG
            'keyword': keyword,
            'answer_field' : answer_field,
            'run' : k,
            'prefix_text' : prefix,
            'suffix' : suffix,
            'ensemble_assignment': ensemble_assignment,
            'ensemble_method': ensemble_method,
            'ensemble_temperature': ensemble_temperature,
            'whole_words_only': whole_words_only,
            'relabeling_path': str(relabeling_path)  # Also add the relabeling path for reference
        }

        total_run_id = n_existing_runs + k # Total number of runs
        save_folder_output  = folder_with_runs / f"run_{total_run_id}" 
        # save_folder_gold = folder_with_runs_gold / f"run_{total_run_id}"

        save_outputs(output, save_folder_output, CONFIG)
        print(colored(f"Results saved to {save_folder_output}", 'green'))
        # save_outputs(output_gold, save_folder_gold, CONFIG)
        # print(colored(f"Results saved to {save_folder_gold}", 'green'))



def main_SLURM():
    # Using this function to run the script with SLURM 
    parser = argparse.ArgumentParser()
    parser.add_argument("jobind", help="ID of the job (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("job_configs_file", help="Path to the JSON file containing job configurations")
    args = parser.parse_args()

    # Load job configurations
    with open(args.job_configs_file, 'r') as f:
        job_configs = json.load(f)
    
    # Get the configuration for this job
    job_idx = int(args.jobind)
    if job_idx >= len(job_configs):
        print(f"Job index {job_idx} is out of range (total configs: {len(job_configs)})")
        sys.exit(0)
    
    config = job_configs[job_idx]
    print(f"Running with config: {config}")  # Print config for debugging
    
    # Convert config keys to match main() parameter names
    config_mapping = {
        'model': 'MODEL_NAME',
        'dataset': 'DATASET_NAME',
        'n_runs': 'N_RUNS'
    }
    
    # Rename keys to match main() parameters
    for old_key, new_key in config_mapping.items():
        if old_key in config:
            config[new_key] = config.pop(old_key)
    
    main(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Check if running with SLURM arguments
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # SLURM mode
        parser.add_argument("jobind", help="ID of the job (SLURM_ARRAY_TASK_ID)")
        parser.add_argument("job_configs_file", help="Path to the JSON file containing job configurations")
        args = parser.parse_args()
        
        # Load job configurations
        with open(args.job_configs_file, 'r') as f:
            job_configs = json.load(f)
        
        # Get the configuration for this job
        job_idx = int(args.jobind)
        if job_idx >= len(job_configs):
            print(f"Job index {job_idx} is out of range (total configs: {len(job_configs)})")
            sys.exit(0)
        
        config = job_configs[job_idx]
        print(f"Running with config: {config}")  # Print config for debugging
        main(**config)
        
    else:
        # Command line mode
        parser.add_argument("--model", default="llama3.1_base", help="Model name")
        parser.add_argument("--dataset", default="claude_multitask", help="Dataset name")
        parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
        parser.add_argument("--prefix_type", default="demos", help="Prefix type")
        parser.add_argument("--n_examples", type=int, required=True, help="Number of examples in context")
        parser.add_argument("--n_relabel", type=int, required=True, help="Number of examples used for relabeling")
        parser.add_argument("--keyword", default="Category", help="Keyword")
        parser.add_argument("--answer_field", default="emotion_letter", help="Answer field")
        parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
        parser.add_argument("--root_folder", default="test_relabelings", help="Root folder for results")
        parser.add_argument("--ensemble_assignment", type=bool, default=False, help="Whether to use ensemble assignment")
        parser.add_argument("--ensemble_method", default="logit_averaging", help="Ensemble method")
        parser.add_argument("--ensemble_temperature", type=float, default=0, help="Ensemble temperature")
        parser.add_argument("--top_tokens", type=int, default=128256, help="Number of top tokens")
        parser.add_argument("--whole_words_only", type=bool, default=True, help="Whether to use only whole words")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        args = parser.parse_args()
        
        # Convert args to match main() parameter names
        config = {
            'MODEL_NAME': args.model,
            'DATASET_NAME': args.dataset,
            'num_classes': args.num_classes,
            'prefix_type': args.prefix_type,
            'n_examples': args.n_examples,
            'n_relabel': args.n_relabel,
            'keyword': args.keyword,
            'answer_field': args.answer_field,
            'N_RUNS': args.n_runs,
            'root_folder': args.root_folder,
            'ensemble_assignment': args.ensemble_assignment,
            'ensemble_method': args.ensemble_method,
            'ensemble_temperature': args.ensemble_temperature,
            'top_tokens': args.top_tokens,
            'whole_words_only': args.whole_words_only,
            'base_seed': args.seed
        }
        main(**config)