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

def template_new_labels(tokenizer, sentences, labels, new_labels, keyword):
    prefix = ''
    prefix_tokens = []
    for sentence, label in zip(sentences, labels):
        prefix += 'Text: ' + sentence + f'\n{keyword}: ' + new_labels[label][0] + '\n'
        prefix_tokens.extend(tokenizer.encode('Text: ' + sentence + f'\n{keyword}: ', add_special_tokens=False))
        prefix_tokens.append(new_labels[label][1])  # This is the token ID
        prefix_tokens.extend(tokenizer.encode('\n', add_special_tokens=False))
    suffix = f'\n{keyword}: '
    return prefix, suffix, prefix_tokens

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

# prev lambda_reg = 0.00001
def optimize_tokens(top_k_tokens, sentences, labels, sentence_logits, tokenizer, max_iterations=100, num_restarts=10, lambda_reg=0, ensemble_assignment=False, ensemble_method='voting', ensemble_temperature=1.0, whole_words_only=False):
    classes = labels.unique()
    
    # Convert to list if it's dict_keys
    if hasattr(top_k_tokens, 'keys'):
        top_k_tokens = list(top_k_tokens)
    
    # Filter for whole word tokens if requested
    if whole_words_only:
        top_k_tokens = [token for token in top_k_tokens if token.startswith('Ġ')]
        print(f"Using {len(top_k_tokens)} whole word tokens for optimization")
        
        if len(top_k_tokens) < len(classes):
            raise ValueError(f"Not enough whole word tokens ({len(top_k_tokens)}) for number of classes ({len(classes)})")
    else:
        print(f"Using {len(top_k_tokens)} tokens for optimization")
    
    # Helper function to calculate the objective (sum_support_ex)
    def calculate_objective(token_assignments):
        total_log_prob = 0
        for sentence, label in zip(sentences, labels):
            # Convert token string to token ID
            label_token_id = tokenizer.convert_tokens_to_ids(token_assignments[label])
            logit = sentence_logits[sentence][label_token_id]
            
            # Convert to float32 to avoid BFloat16 issues
            if hasattr(logit, 'float'):
                logit = logit.float()
            
            # Calculate denominator
            denominator = 0
            for c in classes:
                c_token_id = tokenizer.convert_tokens_to_ids(token_assignments[c])
                c_logit = sentence_logits[sentence][c_token_id]
                
                # Convert to float32 to avoid BFloat16 issues
                if hasattr(c_logit, 'float'):
                    c_logit = c_logit.float()
                
                denominator += torch.exp(c_logit)
            
            probs = torch.exp(logit) / denominator
            log_prob = torch.log(probs)
            total_log_prob += log_prob.item()  # Use .item() to extract scalar value
        return total_log_prob
    
    # Helper function for single hill climbing run
    def hill_climb_single_run(initial_assignments):
        token_assignments = initial_assignments.copy()
        current_objective = calculate_objective(token_assignments)
        
        # Hill climbing optimization
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for class_to_change in classes:
                current_token = token_assignments[class_to_change]
                
                # Get all candidate tokens (excluding current token)
                candidate_tokens = [token for token in top_k_tokens if token != current_token]
                
                if not candidate_tokens:
                    continue
                
                # Convert candidate tokens to token IDs
                candidate_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in candidate_tokens]
                candidate_token_ids = torch.tensor(candidate_token_ids, dtype=torch.long)
                
                # Get token IDs for other classes (unchanged)
                other_class_token_ids = []
                for c in classes:
                    if c != class_to_change:
                        other_class_token_ids.append(tokenizer.convert_tokens_to_ids(token_assignments[c]))
                other_class_token_ids = torch.tensor(other_class_token_ids, dtype=torch.long)
                
                # Vectorized calculation for all candidates
                total_log_probs = torch.zeros(len(candidate_tokens))
                
                for sentence, label in zip(sentences, labels):
                    sentence_logits_tensor = sentence_logits[sentence]
                    
                    if label == class_to_change:
                        # This sentence uses the class we're changing
                        label_logits = sentence_logits_tensor[candidate_token_ids].float()
                        other_logits = sentence_logits_tensor[other_class_token_ids].float()
                        
                        denominators = torch.exp(label_logits) + torch.exp(other_logits).sum()
                        probs = torch.exp(label_logits) / denominators
                        total_log_probs += torch.log(probs)
                        
                    else:
                        # This sentence uses a different class (unchanged)
                        label_token_id = tokenizer.convert_tokens_to_ids(token_assignments[label])
                        label_logit = sentence_logits_tensor[label_token_id].float()
                        
                        candidate_logits = sentence_logits_tensor[candidate_token_ids].float()
                        other_unchanged_logits = sentence_logits_tensor[other_class_token_ids].float()
                        
                        denominators = torch.exp(candidate_logits) + torch.exp(other_unchanged_logits).sum()
                        probs = torch.exp(label_logit) / denominators
                        total_log_probs += torch.log(probs)
                

                other_class_id_sum = sum(tokenizer.convert_tokens_to_ids(token_assignments[c]) 
                        for c in classes if c != class_to_change)

                # For each candidate, we penalize its token ID plus the sum of other class token IDs
                token_id_penalties = -lambda_reg * (candidate_token_ids.float() + other_class_id_sum)
                total_log_probs += token_id_penalties


                # Find the best candidate
                best_idx = torch.argmax(total_log_probs).item()
                best_objective = total_log_probs[best_idx].item()
                best_token = candidate_tokens[best_idx]
                
                # If improvement found, update
                if best_objective > current_objective:
                    token_assignments[class_to_change] = best_token
                    current_objective = best_objective
                    improved = True
                    break  # Move to next class after finding improvement
        
        return token_assignments, current_objective, iteration
    
    # Run multiple restarts
    best_overall_assignments = None
    best_overall_objective = float('-inf')
    best_restart = -1
    all_solutions = []  # Store all solutions for ensemble
    
    for restart in range(num_restarts):
        print(f"\n--- Restart {restart + 1}/{num_restarts} ---")
        
        # Initialize with random assignment for this restart
        initial_assignments = {c: random.choice(top_k_tokens) for c in classes}
        initial_objective = calculate_objective(initial_assignments)
        print(f"Initial objective: {initial_objective}")
        
        # Run hill climbing
        final_assignments, final_objective, iterations = hill_climb_single_run(initial_assignments)
        
        print(f"Final objective after {iterations} iterations: {final_objective}")
        print(f"Final token assignments: {final_assignments}")
        
        # Store solution for ensemble
        all_solutions.append((final_assignments.copy(), final_objective))
        
        # Update best overall solution
        if final_objective > best_overall_objective:
            best_overall_assignments = final_assignments
            best_overall_objective = final_objective
            best_restart = restart + 1
            print(f"*** New best solution found in restart {restart + 1}! ***")
    
    # Choose final assignments based on ensemble_assignment flag
    if ensemble_assignment:
        print(f"\n=== ENSEMBLE ASSIGNMENT ({ensemble_method.upper()}) ===")
        
        if ensemble_method == 'voting':
            # Use majority voting across all restarts
            ensemble_assignments = {}
            for class_label in classes:
                # Count votes for each token for this class
                token_votes = {}
                token_scores = {}  # Track best score for each token (for tie-breaking)
                
                for assignments, objective in all_solutions:
                    token = assignments[class_label]
                    if token not in token_votes:
                        token_votes[token] = 0
                        token_scores[token] = objective
                    token_votes[token] += 1
                    # Keep track of best objective score for this token (for tie-breaking)
                    if objective > token_scores[token]:
                        token_scores[token] = objective
                
                # Find most voted token, break ties by highest objective score
                max_votes = max(token_votes.values())
                tied_tokens = [token for token, votes in token_votes.items() if votes == max_votes]
                
                if len(tied_tokens) == 1:
                    chosen_token = tied_tokens[0]
                    print(f"Class {class_label}: '{chosen_token}' (votes: {max_votes}/{num_restarts})")
                else:
                    # Break tie by highest objective score
                    chosen_token = max(tied_tokens, key=lambda t: token_scores[t])
                    print(f"Class {class_label}: '{chosen_token}' (votes: {max_votes}/{num_restarts}, tie-broken by score: {token_scores[chosen_token]:.4f})")
                
                ensemble_assignments[class_label] = chosen_token
                
        elif ensemble_method == 'logit_averaging':
            # True ensemble averaging: create new token assignments from averaged logits
            print("Computing ensemble assignments via ensemble logit averaging over full vocabulary...")
            
            ensemble_assignments = {}
            
            for class_label in classes:
                print(f"\nProcessing class {class_label}:")
                
                # Collect tokens assigned to this class across restarts
                assigned_tokens = [assignments[class_label] for assignments, _ in all_solutions]
                print(f"  Tokens assigned across restarts: {assigned_tokens}")
                
                # For each sentence of this class, compute ensemble-averaged logits over ALL tokens
                ensemble_logit_sum = torch.zeros(len(top_k_tokens))
                sentence_count = 0
                
                for sentence, label in zip(sentences, labels):
                    if label == class_label:
                        # Get the full logit vector for this sentence
                        sentence_logits_tensor = sentence_logits[sentence].float()
                        
                        # Create ensemble logits by averaging across restart choices
                        restart_logit_sum = torch.zeros_like(sentence_logits_tensor)
                        for token in assigned_tokens:
                            token_id = tokenizer.convert_tokens_to_ids(token)
                            # Add this restart's logit contribution (all zeros except for chosen token)
                            restart_contribution = torch.zeros_like(sentence_logits_tensor)
                            restart_contribution[token_id] = sentence_logits_tensor[token_id]
                            restart_logit_sum += restart_contribution
                        
                        # Average across restarts
                        avg_restart_logits = restart_logit_sum / len(assigned_tokens)
                        
                        # Focus only on our candidate tokens (top_k_tokens)
                        candidate_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in top_k_tokens]
                        ensemble_logit_sum += avg_restart_logits[candidate_token_ids]
                        sentence_count += 1
                
                # Average across all sentences of this class
                if sentence_count > 0:
                    avg_ensemble_logits = ensemble_logit_sum / sentence_count
                else:
                    avg_ensemble_logits = torch.zeros(len(top_k_tokens))
                
                print(f"  Computed ensemble logits across {sentence_count} sentences")
                
                # Apply temperature scaling and softmax over ALL candidate tokens
                scaled_logits = avg_ensemble_logits / ensemble_temperature
                probs = torch.softmax(scaled_logits, dim=0)
                
                # Sample from the full distribution
                if ensemble_temperature == 0.0:  # Deterministic selection (argmax)
                    best_idx = torch.argmax(avg_ensemble_logits).item()
                    chosen_token = top_k_tokens[best_idx]
                    print(f"  → Chosen: '{chosen_token}' (deterministic, logit: {avg_ensemble_logits[best_idx]:.4f})")
                else:  # Probabilistic selection from full vocabulary
                    sampled_idx = torch.multinomial(probs, 1).item()
                    chosen_token = top_k_tokens[sampled_idx]
                    
                    # Show top candidates
                    top_k = 5
                    top_indices = torch.topk(probs, min(top_k, len(top_k_tokens))).indices
                    top_probs = [(top_k_tokens[i], probs[i].item()) for i in top_indices]
                    
                    print(f"  Top {len(top_probs)} candidates: {[(t, f'{p:.3f}') for t, p in top_probs]}")
                    print(f"  → Sampled: '{chosen_token}' (prob: {probs[sampled_idx]:.4f}, logit: {avg_ensemble_logits[sampled_idx]:.4f})")
                    
                    # Check if this is a new token not chosen by any restart
                    if chosen_token not in assigned_tokens:
                        print(f"  *** NEW TOKEN discovered by ensemble! ***")
                
                ensemble_assignments[class_label] = chosen_token
        
        else:
            raise ValueError(f"Unknown ensemble_method: {ensemble_method}")
        
        final_assignments = ensemble_assignments
        final_objective = calculate_objective(final_assignments)
        print(f"\nEnsemble objective: {final_objective}")
        print(f"Ensemble token assignments: {final_assignments}")
    else:
        print(f"\n=== BEST OVERALL SOLUTION (from restart {best_restart}) ===")
        print(f"Best objective: {best_overall_objective}")
        print(f"Best token assignments: {best_overall_assignments}")
        final_assignments = best_overall_assignments
        final_objective = best_overall_objective

    new_labels = {}
    for key, value in final_assignments.items():
        print(f"Class: {key}")
        print(f"  Token string: '{value}'")
        print(f"  Token ID: {tokenizer.convert_tokens_to_ids(value)}")
        print(f"  Back to string: '{tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(value))}'")
        new_labels[key] = (value, tokenizer.convert_tokens_to_ids(value))

    return new_labels, final_objective

def main(MODEL_NAME, DATASET_NAME, num_classes, prefix_type, n_examples, n_relabel, keyword, answer_field, N_RUNS=50, 
         root_folder="DATA/ICL_stability/results", ensemble_assignment=False, ensemble_method='voting', 
         ensemble_temperature=1.0, top_tokens=-1, whole_words_only=False, base_seed=42):
    
    if n_relabel > n_examples:
        raise ValueError(f"n_relabel ({n_relabel}) must be less than or equal to n_examples ({n_examples})")

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
    else:
        # Load all tokens
        file_suffix = '_whole_words' if whole_words_only else ''
        with open(f'sentence_info/template_sentence_probs_all{file_suffix}.pkl', 'rb') as f:
            sentence_probs = pickle.load(f)
        with open(f'sentence_info/template_sentence_logits_all{file_suffix}.pkl', 'rb') as f:
            sentence_logits = pickle.load(f)
        all_tokens = list(vocab.values())  # Get all token IDs
        all_tokens_str = list(vocab.keys())  # Get token strings

    # --- Loading the dataset
    datasets = load_dataset_by_name(DATASET_NAME)
    train_df = datasets['train']
    test_df = datasets['test']
    if num_classes == 3:
        # keep only the rows where the emotion is Joy, Anger, or Fear and reindex the dataframe
        train_df = train_df[train_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        train_df = train_df.reset_index(drop=True)
        test_df = test_df[test_df['emotion'].isin(['Joy', 'Anger', 'Fear'])]
        test_df = test_df.reset_index(drop=True)

    # # --- Adding space to the answer field if needed
    if tokenizer.name_or_path in ['mistralai/Mistral-7B-v0.3']:
        pass # Mistral uses sentencepiece tokenizer, which does not require space before the answer

    else:
        test_df[answer_field] = " " + (test_df[answer_field].astype(str)) # Adding space before the answer
        if answer_field.endswith('_shuffled'):
            test_df[f'{answer_field[:-9]}'] = " " + (test_df[f'{answer_field[:-9]}'].astype(str)) # Adding space before the original category if we are running with category_shuffle

    # --- Creating the ICL template
    if prefix_type != 'demos':
        N_RUNS = 1 # If the prefix is raw or instruction, we run ICL only once with 0 examples in context


    folder_with_runs = root_folder / f"{DATASET_NAME}" / f"{MODEL_NAME}" / f"{answer_field}" / f"{prefix_type}" / f"optimized" / f"{keyword}:{n_examples}_examples" # Folder with runs for a given number of examples in context
    folder_with_runs_gold  = root_folder / f"{DATASET_NAME}" / f"{MODEL_NAME}" / f"{answer_field}" / f"{prefix_type}" / f"gold" / f"{keyword}:{n_examples}_examples_gold" # Folder with runs for a given number of examples in context
    n_existing_runs = len(list(folder_with_runs.glob("run_*"))) # Number of existing runs


    for k in range(N_RUNS):
        # --- Running ICL
        print(colored(f"Running ICL with {n_examples} examples in context (using {n_relabel} for relabeling), run {k}", 'magenta'))

        # --- Creating the ICL prompt, by selecting random examples from the training data and applying the demonstrations template
        prefix, suffix, sentences, labels = create_template(train_df, prefix_type, n_examples, keyword, answer_field, DATASET_NAME, seed=run_seeds[k])
        print("sentences: ", sentences)
        print("labels: ", labels)

        # Use only the first n_relabel examples for optimization
        sentences_for_relabel = sentences[:n_relabel]
        labels_for_relabel = labels[:n_relabel]
        print(f"\nUsing {n_relabel} examples for relabeling optimization:")
        print("sentences for relabel: ", sentences_for_relabel)
        print("labels for relabel: ", labels_for_relabel)

        new_labels, _ = optimize_tokens(list(all_tokens_str), sentences_for_relabel, labels_for_relabel, sentence_logits, 
                                      tokenizer=tokenizer, num_restarts=100, ensemble_assignment=ensemble_assignment, 
                                      ensemble_method=ensemble_method, ensemble_temperature=ensemble_temperature, 
                                      whole_words_only=whole_words_only)
        print("new_labels: ", new_labels)
        if num_classes == 3:
            gold_labels = {'A': ('Ġjoy', 16267), 'C': ('Ġanger', 19788), 'D': ('Ġfear', 8850)}
        elif num_classes == 5:
            gold_labels = {'A': ('Ġjoy', tokenizer.convert_tokens_to_ids('Ġjoy')), 
                           'B': ('Ġsadness', tokenizer.convert_tokens_to_ids('Ġsadness')), 
                           'C': ('Ġanger', tokenizer.convert_tokens_to_ids('Ġanger')), 
                           'D': ('Ġfear', tokenizer.convert_tokens_to_ids('Ġfear')), 
                           'E': ('Ġsurprise', tokenizer.convert_tokens_to_ids('Ġsurprise'))}

        prefix, suffix, prefix_tokens = template_new_labels(tokenizer, sentences, labels, new_labels, keyword)
        prefix_gold, suffix_gold, prefix_tokens_gold = template_new_labels(tokenizer, sentences, labels, gold_labels, keyword)

        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        suffix_tokens_gold = tokenizer.encode(suffix_gold, add_special_tokens=False)
        print('Suffix tokens:', [tokenizer.decode([t]) for t in suffix_tokens])
        suffix_crop = len(suffix_tokens) - 1 # Number of tokens in the suffix to crop (-1 because \n will get fused with "." token at the end of the sentence)

        test_df['text_with_suffix'] = 'Text: ' + test_df['text'] + suffix
        test_df['text_with_suffix_gold'] = 'Text: ' + test_df['text'] + suffix_gold
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
        output_gold = run_on_dataframe(test_df, model, tokenizer, 'text_with_suffix_gold', answer_field='gold_label_id', 
                         return_hidden_states=True, batch_size=10,
                         prefix_tokens=prefix_tokens_gold,  # Use prefix_tokens instead of prefix_text
                         return_raw_batches=True, new_labels=gold_labels)
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
            "DATASET_NAME": DATASET_NAME,
            'num_classes': num_classes,
            'prefix_type': prefix_type,
            'n_examples': n_examples,
            'keyword': keyword,
            'answer_field' : answer_field,
            'run' : k,
            'prefix_text' : prefix,
            'suffix' : suffix,
            'ensemble_assignment': ensemble_assignment,
            'ensemble_method': ensemble_method,
            'ensemble_temperature': ensemble_temperature,
            'whole_words_only': whole_words_only,
        }

        total_run_id = n_existing_runs + k # Total number of runs
        save_folder_output  = folder_with_runs / f"run_{total_run_id}" 
        save_folder_gold = folder_with_runs_gold / f"run_{total_run_id}"

        save_outputs(output, save_folder_output, CONFIG)
        print(colored(f"Results saved to {save_folder_output}", 'green'))
        save_outputs(output_gold, save_folder_gold, CONFIG)
        print(colored(f"Results saved to {save_folder_gold}", 'green'))



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
    main(**config)


if __name__ == "__main__":
    # Check if running with SLURM arguments
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Running with SLURM
        main_SLURM()
    else:
        # Example of how to run the script manually
        main(
            MODEL_NAME = 'llama3.1_base',
            DATASET_NAME = 'claude_multitask',
            num_classes = 3,
            prefix_type = 'demos',
            n_examples = 40,
            n_relabel = 40,
            keyword = 'Category',
            answer_field = 'emotion_letter',
            N_RUNS = 1,
            root_folder = "temp_new/ICL_results_regularization",
            ensemble_assignment = True,
            ensemble_method = 'logit_averaging',
            ensemble_temperature = 0,  # Use 0 for deterministic selection
            top_tokens = 5000,  # Use top 5000 tokens
            whole_words_only = True  # Only use whole word tokens
        )