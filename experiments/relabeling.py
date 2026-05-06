import torch
import random
import numpy as np
from termcolor import colored

MODEL_NAME = 'llama3.1_base'

def optimize_tokens(top_k_tokens, sentences, labels, sentence_logits, tokenizer, max_iterations=100, num_restarts=10, lambda_reg=0, ensemble_assignment=False, ensemble_method='voting', ensemble_temperature=1.0, whole_words_only=False, initial_labels=None):
    '''
    Optimize token assignments for each class using hill climbing with multiple restarts.
    This function is used to find optimal token mappings for each class label.
    '''
    classes = labels.unique()
    
    # Convert to list if it's dict_keys
    if hasattr(top_k_tokens, 'keys'):
        top_k_tokens = list(top_k_tokens)
    
    # Filter for whole word tokens if requested
    if whole_words_only:
        if MODEL_NAME == 'mistral_7b_base':
            top_k_tokens = [token for token in top_k_tokens if token.startswith('▁')]
        else:
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
    
    # If initial_labels provided, run a single deterministic restart from gold,
    # then run remaining restarts from random for comparison.
    if initial_labels is not None:
        effective_restarts = 1  # Gold init is deterministic, only need 1
        print(f"\n*** Gold-label initialization (1 deterministic run) ***")
    else:
        effective_restarts = num_restarts
    
    for restart in range(effective_restarts):
        print(f"\n--- Restart {restart + 1}/{effective_restarts} ---")
        
        if initial_labels is not None:
            initial_assignments = {}
            for c in classes:
                if c in initial_labels:
                    gold_token = initial_labels[c]
                    if gold_token in top_k_tokens:
                        initial_assignments[c] = gold_token
                        print(f"  Class {c}: initialized with '{gold_token}'")
                    else:
                        raise ValueError(f"Gold token '{gold_token}' not in vocabulary for class {c}")
                else:
                    raise ValueError(f"No gold label provided for class {c}")
        else:
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

def template_new_labels(tokenizer, sentences, labels, new_labels, keyword, prompt_format=None):
    """
    Create template with new labels for the given sentences.
    
    Parameters:
    -----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use
    sentences : list
        List of sentences
    labels : list
        List of labels
    new_labels : dict
        Dictionary mapping original labels to (token_str, token_id) tuples
    keyword : str
        Keyword to use in the template
    prompt_format : str or None
        Format variant for the prompt. Options:
          - None / "default" : "Text: ... \n{keyword}: ..."
          - "sentence_label"  : "Sentence: ... \nLabel: ..."
          - "arrow"           : "Input: ... → ..."
        
    Returns:
    --------
    prefix : str
        The template prefix
    suffix : str
        The template suffix
    prefix_tokens : list
        List of token IDs for the prefix
    """
    prefix = ''
    prefix_tokens = []

    if prompt_format == 'sentence_label':
        for sentence, label in zip(sentences, labels):
            prefix += 'Sentence: ' + sentence + '\nLabel: ' + new_labels[label][0] + '\n'
            prefix_tokens.extend(tokenizer.encode('Sentence: ' + sentence + '\nLabel: ', add_special_tokens=False))
            prefix_tokens.append(new_labels[label][1])
            prefix_tokens.extend(tokenizer.encode('\n', add_special_tokens=False))
        if MODEL_NAME == 'mistral_7b_base':
            suffix = '\nLabel:'
        else:
            suffix = '\nLabel: '

    elif prompt_format == 'arrow':
        for sentence, label in zip(sentences, labels):
            prefix += 'Input: ' + sentence + ' → ' + new_labels[label][0] + '\n'
            prefix_tokens.extend(tokenizer.encode('Input: ' + sentence + ' → ', add_special_tokens=False))
            prefix_tokens.append(new_labels[label][1])
            prefix_tokens.extend(tokenizer.encode('\n', add_special_tokens=False))
        if MODEL_NAME == 'mistral_7b_base':
            suffix = ' →'
        else:
            suffix = ' → '

    else:  # default
        for sentence, label in zip(sentences, labels):
            prefix += 'Text: ' + sentence + f'\n{keyword}: ' + new_labels[label][0] + '\n'
            prefix_tokens.extend(tokenizer.encode('Text: ' + sentence + f'\n{keyword}: ', add_special_tokens=False))
            prefix_tokens.append(new_labels[label][1])
            prefix_tokens.extend(tokenizer.encode('\n', add_special_tokens=False))
        if MODEL_NAME == 'mistral_7b_base':
            suffix = f'\n{keyword}:'
        else:
            suffix = f'\n{keyword}: '

    return prefix, suffix, prefix_tokens 