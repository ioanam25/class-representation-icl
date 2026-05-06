import LLMGeometry.models.llama3_8b_base.evaluation
import LLMGeometry.models.llama3_1b_base.evaluation
import LLMGeometry.models.llama3_70b_instruct.evaluation
import LLMGeometry.models.gemma2_2b_base.evaluation
import LLMGeometry.models.mistral_7b_base.evaluation
import LLMGeometry.models.qwen2_7b_base.evaluation
from LLMGeometry.preprocessing import prepare_batch, DataFrameDataset
from LLMGeometry.postprocessing import get_token_probability, get_predictions_from_logits
from LLMGeometry.utils import take_last_from_attention_mask, create_causal_mask, expand_KV

try:
    from transformers.cache_utils import HybridCache, DynamicCache
except ImportError:
    # HybridCache might not be available in older transformers versions
    try:
        from transformers.cache_utils import DynamicCache
        HybridCache = None
    except ImportError:
        # Very old versions might not have either
        HybridCache = None
        DynamicCache = None
import math
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from termcolor import colored
import warnings
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.classification import CalibrationError
from collections import defaultdict

# ----------------------- Code for manual forward pass (legacy) ----------------------- 
# def manual_forward_pass(concatenated_input_embeddings, model, attention_mask, correct_labels=None, return_MLP_intermediate=False, return_attention_weights=False, return_QKV=False):
#     if model.name_or_path == 'meta-llama/Meta-Llama-3-8B-Instruct':
#         return LLMGeometry.models.llama3_instruct.evaluation.manual_forward_pass(concatenated_input_embeddings, model, attention_mask, correct_labels=correct_labels, return_MLP_intermediate=return_MLP_intermediate, return_attention_weights=return_attention_weights, return_QKV=return_QKV)
#     if model.name_or_path == 'meta-llama/Meta-Llama-3-8B':
#         return LLMGeometry.models.llama3_base.evaluation.manual_forward_pass(concatenated_input_embeddings, model, attention_mask, correct_labels=correct_labels, return_MLP_intermediate=return_MLP_intermediate, return_attention_weights=return_attention_weights, return_QKV=return_QKV)
#     if model.name_or_path == 'gpt2-xl':
#         return LLMGeometry.models.gpt2_xl.evaluation.manual_forward_pass(concatenated_input_embeddings, model, attention_mask, correct_labels=correct_labels, return_MLP_intermediate=return_MLP_intermediate, return_attention_weights=return_attention_weights, return_QKV=return_QKV)
    
#     raise ValueError(f"Model {model.name_or_path} not supported.")
# ----------------------- Code for manual forward pass (legacy) -----------------------


def wrap_in_dataloader(dataframe, batch_size=10, shuffle=False, **kwargs):
    '''
        Wrap a dataframe in a DataLoader.
    '''
    return DataLoader(DataFrameDataset(dataframe), batch_size=batch_size, collate_fn=lambda x: x, shuffle=shuffle, **kwargs)

def get_stacked_KV(batched_keys, batched_values, batched_attention_masks):

    acc_keys, acc_values = [], []
    for batch_idx, attention_mask in enumerate(batched_attention_masks):
        keys = batched_keys[batch_idx]
        values = batched_values[batch_idx]


        for i in range(attention_mask.shape[0]): # Iterate over invididual samples in the batch
            acc_keys.append(keys[:,i, attention_mask[i].bool().cpu()])
            acc_values.append(values[:,i, attention_mask[i].bool().cpu()])
    return acc_keys, acc_values


def hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=None, **kwargs):
    '''
        Forward pass of the model with hooks to extract intermediate states. 
    '''
    if model.name_or_path == 'meta-llama/Meta-Llama-3-8B' or model.name_or_path == 'meta-llama/Meta-Llama-3.1-8B':
        return LLMGeometry.models.llama3_8b_base.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == 'meta-llama/Llama-3.2-1B':
        return LLMGeometry.models.llama3_1b_base.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == '/gpfs/data/oermannlab/public_models/llama_models_hf/llama-3.1-70b-instruct':
        return LLMGeometry.models.llama3_70b_instruct.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == 'meta-llama/Meta-Llama-3-8B-Instruct':
        return LLMGeometry.models.llama3_8b_instruct.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == 'gpt2-xl':
        return LLMGeometry.models.gpt2_xl.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == 'google/gemma-2-2b':
        return LLMGeometry.models.gemma2_2b_base.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == 'mistralai/Mistral-7B-v0.3':
        return LLMGeometry.models.mistral_7b_base.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    if model.name_or_path == 'Qwen/Qwen2.5-7B':
        return LLMGeometry.models.qwen2_7b_base.evaluation.hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=correct_labels, **kwargs)
    else:
        raise ValueError(f"Model {model.name_or_path} not supported.")

def cache_prefix_text(model, tokenizer, prefix_text=None, prefix_tokens=None):
    '''
        Cache the prefix text or tokens in the model's past key values.
        Returns a DynamicCache object compatible with newer transformers versions.
    '''
    from transformers.cache_utils import DynamicCache
    
    # Ensure CUDA is properly initialized
    if torch.cuda.is_available():
        torch.cuda.init()
        device = model.device
        print(f"Using device: {device}")

    if isinstance(prefix_text, torch.Tensor):
        # If we are provided a tensor, assume is the input embeddings (e.g. a soft prompt from prompt tuning)
        assert ((prefix_text.ndim == 3) or (prefix_text.ndim == 2)), "The prefix text tensor should have shape (1, sequence_length, hidden_dim) or (sequence_length, hidden_dim)"
        if prefix_text.ndim == 2:
            prefix_text = prefix_text.unsqueeze(0)
            print(colored("Unsqueezed prefix text tensor", 'yellow'))

        batch_prefix = {'input_embeds': prefix_text, 'attention_mask': torch.ones(prefix_text.shape[:2], device=model.device, dtype=torch.long)}
        prefix_seq_len = prefix_text.shape[1]
        
    elif prefix_tokens is not None:
        # If we are provided with tokens, create embeddings from them
        print(colored("Using pre-tokenized prefix tokens", 'yellow'))
        
        # Flatten the prefix_tokens list and convert to tensor
        flat_tokens = []
        for item in prefix_tokens:
            if isinstance(item, list):
                flat_tokens.extend(item)
            else:
                flat_tokens.append(item)
        
        token_ids = torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0).to(model.device)
        
        # Create embeddings using the model's embedding layer
        if model.name_or_path.startswith('meta-llama'):
            from LLMGeometry.models.llama3_8b_base.preprocessing import embed_tokens
            input_embeds = embed_tokens(token_ids, model)
        else:
            # Add support for other models as needed
            input_embeds = model.model.embed_tokens(token_ids)
            
        batch_prefix = {
            'input_embeds': input_embeds, 
            'attention_mask': torch.ones(token_ids.shape, device=model.device, dtype=torch.long)
        }
        prefix_seq_len = token_ids.shape[1]
        
    else:
        # If we are provided with a string, tokenize it and prepare the batch
        batch_prefix = prepare_batch([{'text': prefix_text}], model, tokenizer)
        prefix_seq_len = batch_prefix['attention_mask'].shape[1]

    # Handle empty prefix case - return None for past_key_values and empty attention mask
    if prefix_seq_len == 0:
        print(colored("Empty prefix detected - returning None for past_key_values", 'yellow'))
        empty_attention_mask = torch.zeros((1, 0), device=model.device, dtype=torch.long)
        return None, empty_attention_mask

    if model.name_or_path == 'google/gemma-2-2b':
        # Since Gemma uses sliding window attention, caching is done with HybridCache, rather than tuple of tensors
        if HybridCache is None:
            raise ImportError("HybridCache is not available in this version of transformers. Please upgrade transformers to use Gemma models.")
        kv_cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=prefix_seq_len + 100, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            model(inputs_embeds = batch_prefix['input_embeds'], attention_mask = batch_prefix['attention_mask'], use_cache=True, past_key_values=kv_cache)
        print(colored("Cached prefix text", 'green'))
        kv_cache.seq_length = prefix_seq_len # Creating a custom attribute to store the sequence length (accessed in the forward pass)
        return kv_cache, batch_prefix['attention_mask']

    # For other models, caching is done with tuple of tensors
    with torch.no_grad():
        # Create a new dynamic cache
        cache = DynamicCache()
        
        # Ensure all tensors are on the correct device
        inputs_embeds = batch_prefix['input_embeds'].to(model.device)
        attention_mask = batch_prefix['attention_mask'].to(model.device)
        
        # Run model and update cache
        outputs = model(inputs_embeds=inputs_embeds, 
                       attention_mask=attention_mask, 
                       use_cache=True, 
                       past_key_values=cache)
        past_key_values = outputs.past_key_values
    print(colored("Cached prefix text", 'green'))
    return past_key_values, batch_prefix['attention_mask']


def run_on_dataframe(dataframe, model, tokenizer, prompt_text_field, answer_field=None, soft_embeds=None, soft_prompt_location='before', 
                     prefix_text=None, prefix_tokens=None, return_hidden_states=False, return_MLP_intermediate=False, return_queries=False, return_keys=False, return_values=False,
                     batch_size=10, 
                     target_token_offset_performance=0, target_token_offset_activations=0, return_raw_batches=True, keep_KV_batched=False, new_labels=None):
    '''
        Run the model on a text dataframe and return the performance metrics and embeddings.

        Parameters:
        ----------
        dataframe: pd.DataFrame
            The dataframe to run the model on. It should contain the prompt_text_field and answer_field.
        model: Callable
            The model to run (passed into the manual_forward_pass function).
        tokenizer: Callable
            The tokenizer to use. Used to tokenize the input text.
        prompt_text_field: str
            The name of the column in the dataframe containing the prompt text used as the input.
        answer_field: str
            The name of the column in the dataframe containing the correct answer tokens. If None, no correct answer is used and performance metrics are not computed.
        soft_embeds: torch.Tensor, optional
            The soft prompt embeddings to use. If None, no soft prompt is used.
        soft_prompt_location: str, optional
            The location of the soft prompt. Can be 'before' or 'after'. Ignored if soft_embeds is None.
        return_hidden_states: bool, optional
            If True, the hidden states are returned (residual stream embeddings). Tensor of shape (batch_size, sequence_length, hidden_dim). Extracted from the target token.
        return_MLP_intermediate: bool, optional
            If True, the intermediate MLP output is returned. Tensor of shape (batch_size, sequence_length, MLP_dim). Extracted from the target token.
        return_queries: bool, optional
            If True, the queries are returned. Tensor of shape (batch_size, sequence_length, hidden_dim). Extracted from the target token.
        batch_size: int, optional
            The batch size to use when running the model in batches. Default is 16.
        target_token_offset_performance: int, optional
            The offset to apply to the target token index when computing the performance metrics. Default is 0.
        target_token_offset_activations: int, optional
            The offset to apply to the target token index when extracting the hidden states, MLP intermediate, queries, keys and values. Default is 0.
    '''

    # --- First tokenize all the text in the dataframe (no padding) ---

    output_df = dataframe.copy()
    output_df[f'{prompt_text_field}_tokenized'] = output_df[prompt_text_field].apply(lambda x: tokenizer.encode(x, add_special_tokens=False, return_tensors='pt').squeeze(0))

    dataloader = wrap_in_dataloader(output_df, batch_size=batch_size, shuffle=False)
    acc_attention_masks = []
    acc_metrics = []
    acc_hidden_states = []
    acc_MLP_intermediate = []
    acc_attention_weights = []
    acc_keys = []
    acc_values = []
    acc_queries = []
    acc_batches = []

    if prefix_text is None and prefix_tokens is None:
        prefix_text = '' # If no prefix text or tokens are provided, we use an empty string
    
    # if prefix_tokens is not None:
    #     print("Using pre-tokenized prefix_tokens")
    #     print("prefix_tokens: ", prefix_tokens)
    # else:
    #     print("prefix_text: ", prefix_text)


    if not (answer_field is None):
        # --- Token IDs of acceptable answers (possible task categories)
        # if 'label' in dataframe.columns:
        #     groups = dataframe.sort_values('label')[answer_field].unique() # If the dataframe contains a 'label' column, we use it to sort the scope of acceptable answers
        # else:
        #     groups = dataframe[answer_field].unique() # Otherwise, we use the order in the dataframe
        # print('groups: ', groups)
        # # apply new_labels to the groups
        # groups = [' ' + new_labels[x[1:]] for x in groups]
        # print("new groups: ", groups)
        # outputs_scope = [tokenizer.encode(x, add_special_tokens=False)[0] for x in groups] # Token IDs of acceptable answers (possible task categories)
        # print("outputs_scope: ", outputs_scope)
        # # --- Also adding the correct token with space before as a possible answer
        # print(output_df[answer_field])
        # # apply new_labels to the answer field
        # output_df[answer_field] = output_df[answer_field].apply(lambda x: new_labels[x[1:]])
        # print(output_df[answer_field])
        # output_df['target_token_with_space'] = output_df[answer_field].apply(lambda x: tokenizer.encode(f' {x}')[0])

        outputs_scope = dataframe[answer_field].unique()
        print("outputs_scope: ", outputs_scope)

    print("prefix_text: ", prefix_text)
    # Cache the prefix text in the model's past key values
    # Use the new Cache format from transformers
    from transformers.cache_utils import DynamicCache
    
    # Initialize cache as None
    past_key_values = None
    prefix_attention_mask = torch.zeros((1, 0), device=model.device, dtype=torch.long)
    
    # Only cache if we have a prefix
    if prefix_text is not None or prefix_tokens is not None:
        past_key_values, prefix_attention_mask = cache_prefix_text(model, tokenizer, prefix_text=prefix_text, prefix_tokens=prefix_tokens)

    # --- For Gemma models, we need to .expand().clone() the past key values to the max batch size first
    if model.name_or_path == 'google/gemma-2-2b':
        if past_key_values is not None:  # Only expand if past_key_values is not None
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx].expand(batch_size, -1, -1,-1).clone()
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx].expand(batch_size, -1, -1,-1).clone()
            print(colored("Expanded past key values", 'green'))



    for batch in tqdm(dataloader):
        if return_raw_batches:
            acc_batches.append(batch)

        # Clear GPU memory before processing each batch to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            import gc
            gc.collect()
            # Print memory usage for debugging
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"DEBUG: GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        with torch.no_grad():
            tokenized_batch = prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_embeds, soft_prompt_location=soft_prompt_location, prompt_text_field=prompt_text_field,answer_field=answer_field, add_special_tokens=False)

            # Debug: Print batch info right after tokenization
            attention_mask = tokenized_batch['attention_mask']
            correct_labels = tokenized_batch['correct_labels']
            print(f"DEBUG: Batch size: {attention_mask.shape[0]}, Sequence length: {attention_mask.shape[1]}")
            print(f"DEBUG: Input embeds shape: {tokenized_batch['input_embeds'].shape}")
            print(f"DEBUG: About to call hooked_forward_pass...")
            
            # Handle concatenation when prefix_attention_mask might be empty (0 length)
            if prefix_attention_mask.shape[1] > 0:
                concatenated_attention_mask = torch.cat([prefix_attention_mask.expand(len(batch), -1), tokenized_batch['attention_mask']],1)
                print(f"DEBUG: Prefix length: {prefix_attention_mask.shape[1]}, Test input length: {tokenized_batch['attention_mask'].shape[1]}")
                print(f"DEBUG: TOTAL concatenated sequence length: {concatenated_attention_mask.shape[1]}")
            else:
                concatenated_attention_mask = tokenized_batch['attention_mask']
                print(f"DEBUG: No prefix, using test input length: {concatenated_attention_mask.shape[1]}")
            
            # Handle past_key_values that might be None (for empty prefix)
            past_kv_expanded = expand_KV(past_key_values, len(batch)) if past_key_values is not None else None
            
            output = hooked_forward_pass(model,tokenized_batch['input_embeds'],
                                        attention_mask=concatenated_attention_mask,
                                        correct_labels=correct_labels,
                                        past_key_values=past_kv_expanded
                                    )
            
            prediction = get_predictions_from_logits(output['output'].logits, attention_mask, N=3, target_token_offset=target_token_offset_performance)
            
            # Constrained predictions (only over valid class tokens)
            if answer_field is not None:
                prediction_constrained = get_predictions_from_logits(output['output'].logits, attention_mask, N=3, target_token_offset=target_token_offset_performance, constrained_token_ids=list(outputs_scope))
            
            # prediction 0 is ids, prediction 1 is probs
            #print("prediction shape: ", prediction[0].shape)
            # pred shape is batch x top N
            # print("prediction: ", prediction[0][0]) # top N predictions for the first batch example
            # print("prediction for full batch top 1: ", prediction[0][:,0])
            # print("prediction tokens for full batch top 1: ", tokenizer.convert_ids_to_tokens(prediction[0][:, 0]))
            # print("prediction probabilities for full batch top 1: ", prediction[1][:, 0])



            target_logits = take_last_from_attention_mask(output['output'].logits, tokenized_batch['attention_mask'])
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

            if answer_field is not None:
                target_token_ids = take_last_from_attention_mask(tokenized_batch['correct_labels'], tokenized_batch['attention_mask'])
                target_labels = [b[answer_field] for b in batch]
                # print("target_labels: ", target_labels)
                # print("target labels tokens", tokenizer.convert_ids_to_tokens(target_labels))
                # Probability of correct output
                correct_output_probs = get_token_probability(
                    torch.softmax(output['output'].logits, dim=2), target_labels , 
                    tokenizer=tokenizer, attention_mask=tokenized_batch['attention_mask'],
                    by_element_in_batch=True, target_token_offset=target_token_offset_performance)
                # print("correct_output_probs: ", correct_output_probs)
      
                acc_scope_logits, acc_scope_probs = [], []
                for b_idx in range(len(batch)):
                    acc_scope_logits.append({t: target_logits[b_idx, t].item() for t in outputs_scope})
                    acc_scope_probs.append({t: target_probs[b_idx, t].item() for t in outputs_scope})

            
        # --- Collecting embeddings and attention weights if needed ---
        if return_hidden_states:
            # Move all hidden states to the same device (model's device)
            hidden_states_list = [act['hidden_states'].to(model.device) for act in output['activations']]
            stacked_hidden_states = torch.stack(hidden_states_list)  # (layer, batch, sequence_length, hidden_dim)
        if return_MLP_intermediate:
            mlp_list = [act['MLP_intermediate'].to(model.device) for act in output['activations']]
            stacked_MLP_intermediate = torch.stack(mlp_list) # (layer, batch, sequence_length, MLP_dim)
        if return_queries:
            queries_list = [act['queries'].to(model.device) for act in output['activations']]
            stacked_queries = torch.stack(queries_list) # (layer, batch, sequence_length, hidden_dim)
        if return_keys:
            keys_list = [act['keys'].to(model.device) for act in output['activations']]
            stacked_keys = torch.stack(keys_list) # (layer, batch, sequence_length, hidden_dim)
        if return_values:
            values_list = [act['values'].to(model.device) for act in output['activations']]
            stacked_values = torch.stack(values_list) # (layer, batch, sequence_length, hidden_dim)
        
        # --- Un-batching the activations ---
        for i, b in enumerate(batch):
            acc_metrics.append({
                'target_token' : target_token_ids[i].item() if answer_field is not None else None,
                'correct_output_prob': correct_output_probs[i] if answer_field is not None else None,
                'scope_logits': acc_scope_logits[i] if answer_field is not None else None,
                'scope_probs': acc_scope_probs[i] if answer_field is not None else None,
                'predicted_output_tokens': prediction[0][i],
                'predicted_output_probs': prediction[1][i],
                'predicted_output_tokens_constrained': prediction_constrained[0][i] if answer_field is not None else None,
                'predicted_output_probs_constrained': prediction_constrained[1][i] if answer_field is not None else None,
            })
            if return_hidden_states:
                acc_hidden_states.append(stacked_hidden_states[:, i, tokenized_batch['attention_mask'][i].to(bool),  :]) # (layer, sequence_length, hidden_dim)

            if return_MLP_intermediate:
                acc_MLP_intermediate.append(stacked_MLP_intermediate[:, i, tokenized_batch['attention_mask'][i].to(bool),  :]) # (layer, sequence_length, hidden_dim)

            if return_queries:
                acc_queries.append(stacked_queries[:, i, tokenized_batch['attention_mask'][i].to(bool),  :]) # (layer, sequence_length, hidden_dim)

        # Not unbatching the keys and values if not specified
        if return_keys:
            acc_keys.append(stacked_keys)
        if return_values:
            acc_values.append(stacked_values)
        # Move attention_mask to CPU immediately to free GPU memory
        acc_attention_masks.append(attention_mask.cpu())
        
        # CRITICAL: Clear accumulated GPU tensors to prevent memory leak
        # Move any GPU tensors to CPU immediately after appending
        if acc_hidden_states:
            for i in range(len(acc_hidden_states)):
                if isinstance(acc_hidden_states[i], torch.Tensor) and acc_hidden_states[i].is_cuda:
                    acc_hidden_states[i] = acc_hidden_states[i].cpu()
        if acc_MLP_intermediate:
            for i in range(len(acc_MLP_intermediate)):
                if isinstance(acc_MLP_intermediate[i], torch.Tensor) and acc_MLP_intermediate[i].is_cuda:
                    acc_MLP_intermediate[i] = acc_MLP_intermediate[i].cpu()
        if acc_queries:
            for i in range(len(acc_queries)):
                if isinstance(acc_queries[i], torch.Tensor) and acc_queries[i].is_cuda:
                    acc_queries[i] = acc_queries[i].cpu()
        if acc_keys:
            for i in range(len(acc_keys)):
                if isinstance(acc_keys[i], torch.Tensor) and acc_keys[i].is_cuda:
                    acc_keys[i] = acc_keys[i].cpu()
        if acc_values:
            for i in range(len(acc_values)):
                if isinstance(acc_values[i], torch.Tensor) and acc_values[i].is_cuda:
                    acc_values[i] = acc_values[i].cpu()
        
        # Debug: Print accumulator sizes
        print(f"DEBUG: Accumulators - metrics: {len(acc_metrics)}, masks: {len(acc_attention_masks)}, hidden: {len(acc_hidden_states)}")
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()

        # # TODO: Remove this after testing
        # break
            
    p = pd.DataFrame(acc_metrics)
    metrics_df = pd.concat([output_df, p], axis=1)
    print("metrics_df: ", metrics_df)

    # Accuracy and F1 score
    metrics_df['highest_prob_token'] = metrics_df['predicted_output_tokens'].apply(lambda x: x[0])
    metrics_df['highest_prob_token_string'] = metrics_df.apply(lambda x: tokenizer.decode(x['predicted_output_tokens'][0]), axis=1)
    
    if answer_field is not None:
        # Add constrained prediction columns
        metrics_df['highest_prob_token_constrained'] = metrics_df['predicted_output_tokens_constrained'].apply(lambda x: x[0] if x is not None else None)
        
        # Original unconstrained accuracies
        metrics_df.attrs['accuracy'] = accuracy_score(metrics_df['target_token'], metrics_df['highest_prob_token'])
        metrics_df.attrs['f1_score'] = f1_score(metrics_df['target_token'], metrics_df['highest_prob_token'], average='weighted')
        
        # New constrained accuracies
        metrics_df.attrs['accuracy_constrained'] = accuracy_score(metrics_df['target_token'], metrics_df['highest_prob_token_constrained'])
        metrics_df.attrs['f1_score_constrained'] = f1_score(metrics_df['target_token'], metrics_df['highest_prob_token_constrained'], average='weighted')

        # --- Calibration error
        label_mapping = list(metrics_df.scope_probs.iloc[0].keys())
        print("label_mapping: ", label_mapping)
        stacked_scope_probs = torch.Tensor([list(s.values()) for s in metrics_df.scope_probs])
        stacked_scope_probs = torch.cat([stacked_scope_probs, 1-stacked_scope_probs.sum(axis=1, keepdims=True)], axis=1)
        target_class = torch.Tensor(metrics_df.target_token.apply(lambda x: label_mapping.index(x)))
        print(len(label_mapping))
        calibration_error = CalibrationError(task="multiclass", num_classes=len(label_mapping)+1, n_bins=15)

        metrics_df.attrs['expected_calibration_error'] = calibration_error(stacked_scope_probs, target_class).item()

        print(
            colored(f"Accuracy: {metrics_df.attrs['accuracy']:.2f}, F1 score: {metrics_df.attrs['f1_score']:.2f}, Expected calibration error: {metrics_df.attrs['expected_calibration_error']:.2f}", 'green'))
        print(
            colored(f"Accuracy (constrained): {metrics_df.attrs['accuracy_constrained']:.2f}, F1 score (constrained): {metrics_df.attrs['f1_score_constrained']:.2f}", 'blue'))

    if not keep_KV_batched:
        if return_keys and return_values:
            acc_keys, acc_values = get_stacked_KV(acc_keys, acc_values, acc_attention_masks)

    return {
        'metrics': metrics_df,
        'hidden_states': acc_hidden_states if return_hidden_states else None,
        'MLP_intermediate': acc_MLP_intermediate if return_MLP_intermediate else None,
        'queries': acc_queries if return_queries else None,
        'keys' : acc_keys if return_keys else None,
        'values' : acc_values if return_values else None,
        'attention_masks' : acc_attention_masks,
        'batches' : acc_batches if return_raw_batches else None,
    }

def compute_attention_output(query_vector, keys, values, attention_mask):
    '''
        Compute the output of the attention mechanism given a query vector, keys, values and an attention mask.

        Parameters:
        ----------
        query_vector: torch.Tensor
            The query vector. Shape: [batch_size, 1, hidden_dim] or [hidden_dim]
        keys: torch.Tensor
            The keys. Shape: [batch_size, sequence_length, num_heads, head_dim]
        values: torch.Tensor
            The values. Shape: [batch_size, sequence_length, num_heads, head_dim]
        attention_mask: torch.Tensor
            The attention mask. Shape: [batch_size, sequence_length]
    '''
    source_dtype = keys.dtype
    keys = keys.transpose(1, 2).float()
    values = values.transpose(1, 2).float()

    bsz, num_heads, q_len, head_dim = keys.shape
    if query_vector.ndim == 1:
        query_vector = query_vector.unsqueeze(0).unsqueeze(0)
    if query_vector.ndim == 2:
        query_vector = query_vector.unsqueeze(1)
    
    query_vector = query_vector.to(keys.device).expand(bsz, 1, -1) # Expand to the batch size
    query_vector = query_vector.view(bsz, 1, num_heads, head_dim).transpose(1, 2) # Reshape to match the keys and values 

    # Attention mask
    causal_mask_expanded = create_causal_mask(attention_mask)
    causal_mask_last_token = causal_mask_expanded[:,:,[-1],:]

    # Compute attention weights
    query_vector = query_vector.float() # upcast to float
    attn_weights = torch.matmul(query_vector, keys.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + causal_mask_last_token # Apply mask for padding tokens
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1) # Apply softmax

    # Compute output
    attn_output = torch.matmul(attn_weights, values)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, 1, -1)
    return attn_output.to(source_dtype), attn_weights