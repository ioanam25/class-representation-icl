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
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import glob

MODEL_NAME = 'mistral_7b_base'
DATASET_NAME = 'TREC_coarse'
prefix_type = 'demos'
n_examples = 5
keyword = 'Category'
answer_field = 'label'
N_RUNS = 50
root_folder = "temp/ICL_results"

datasets = load_dataset_by_name(DATASET_NAME)
train_df = datasets['train']
test_df = datasets['test']

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

top_128256_tokens = sorted_vocab[:128256]
top_10000_tokens = sorted_vocab[:10000]
top_5000_tokens = sorted_vocab[:5000]
top_4000_tokens = sorted_vocab[:4000]
top_3000_tokens = sorted_vocab[:3000]
top_2000_tokens = sorted_vocab[:2000]
top_1000_tokens = sorted_vocab[:1000]
top_100_tokens = sorted_vocab[:100]
top_10_tokens = sorted_vocab[:10]

def forward_pass(model, tokenizer, sentence):
    # print(sentence)
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    
    # Use with context to ensure memory is released
    with torch.no_grad(), torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        # Make a copy and move to CPU immediately
        last_token_logits = logits[0, -1, :].cpu()
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    
    # Clean up references explicitly
    del inputs, outputs, logits
    torch.cuda.empty_cache()
    
    return probs, last_token_logits

def get_filtered_tokens(vocab, k=None, whole_words_only=False):
    '''
    Filter and sort vocabulary tokens.
    
    Parameters:
    ----------
    vocab : dict
        The vocabulary dictionary from tokenizer
    k : int, optional
        Number of top tokens to return. If None, returns all filtered tokens
    whole_words_only : bool
        If True, only returns tokens starting with 'Ġ' (whole words)
    
    Returns:
    -------
    list
        List of (token, id) tuples sorted by id
    '''
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    if whole_words_only:
        # Filter for tokens starting with 'Ġ'
        if MODEL_NAME == 'mistral_7b_base':
            sorted_vocab = [(token, idx) for token, idx in sorted_vocab if token.startswith('▁')]
        else:
            sorted_vocab = [(token, idx) for token, idx in sorted_vocab if token.startswith('Ġ')]
    
    if k is not None:
        # Only slice if k is smaller than the list size
        k = min(k, len(sorted_vocab))
        sorted_vocab = sorted_vocab[:k]
    
    return sorted_vocab

# Get different token sets
whole_word_tokens = get_filtered_tokens(vocab, whole_words_only=True)
print(f"Total whole word tokens: {len(whole_word_tokens)}")

# Get top-k tokens (both filtered and unfiltered)
token_sizes = [1000, 2000, 3000, 4000, 5000, 10000, 128256]
top_k_tokens = {
    size: get_filtered_tokens(vocab, k=size, whole_words_only=False) for size in token_sizes
}
top_k_whole_words = {
    size: get_filtered_tokens(vocab, k=size, whole_words_only=True) for size in token_sizes
}

def precompute_W_forward_pass(sentences, top_k_tokens=None, whole_words_only=False):
    """
    Precompute probabilities and logits for sentences.
    
    Parameters:
    ----------
    sentences : list
        List of sentences to process
    top_k_tokens : list or None
        If list: either list of (token, id) tuples or list of token ids
        If None: use all tokens from vocabulary
    whole_words_only : bool
        If True, only consider tokens starting with 'Ġ'
    """
    sentence_probs = {}
    sentence_logits = {}
    
    # Handle token selection
    if top_k_tokens is not None:
        if isinstance(top_k_tokens[0], tuple):
            # Convert from (token, id) tuples to just ids
            token_indices = [token[1] for token in top_k_tokens]
            print("is instance of tuple: ", len(token_indices))
        else:
            # Already have token ids
            token_indices = top_k_tokens
            
        if whole_words_only:
            # Filter for whole words if needed
            token_strs = tokenizer.convert_ids_to_tokens(token_indices) #[tokenizer.decode([t]) for t in token_indices]
            # print("token strs: ", token_strs)
            # print("token strs starts with Ġ: ", [token_strs[i].startswith('Ġ') for i in range(len(token_strs))])
            if MODEL_NAME == 'mistral_7b_base':
                token_indices = [t for i, t in enumerate(token_indices) if token_strs[i].startswith('▁')]
            else:
                token_indices = [t for i, t in enumerate(token_indices) if token_strs[i].startswith('Ġ')]
            print("after filtering for whole words: ", len(token_indices))
    else:
        # Using all tokens
        if whole_words_only:
            # Get indices of all whole word tokens
            if MODEL_NAME == 'mistral_7b_base':
                token_indices = [idx for token, idx in vocab.items() if token.startswith('▁')]
            else:
                token_indices = [idx for token, idx in vocab.items() if token.startswith('Ġ')]
        else:
            # Use all token indices
            token_indices = list(range(len(vocab)))
    
    # print(token_indices)
    print(f"Using {len(token_indices)} tokens for computation")
    
    for i, sentence in enumerate(sentences):
        if MODEL_NAME == 'mistral_7b_base':
            template_sentence = 'Text: ' + sentence + '\n' + 'Category:'
        else:
            template_sentence = 'Text: ' + sentence + '\n' + 'Category: '
        print(f"Processing sentence {i+1} of {len(sentences)}: " + template_sentence)
        
        # Get probabilities (already on CPU from forward_pass)
        probs, logits = forward_pass(model, tokenizer, template_sentence)
        
        # Select the relevant token probabilities
        token_probs = probs[token_indices]
        
        sentence_probs[sentence] = token_probs
        sentence_logits[sentence] = logits
        
        # Clean up explicitly
        del probs, token_probs
        torch.cuda.empty_cache()
        
    return sentence_probs, sentence_logits


if __name__ == '__main__':
    all_tokens = vocab.values()
    all_tokens_str = vocab.keys()

    token_to_index = {token: i for i, token in enumerate(all_tokens)}
    index_to_token = {i: token for i, token in enumerate(all_tokens)}

    # print(top_128256_tokens)
    precomputed = False
    if not precomputed:
        sentences = train_df['text']
        
        # Compute for top 4000 whole words
        sentence_probs, sentence_logits = precompute_W_forward_pass(
            sentences, 
            top_k_tokens=top_k_tokens[128256],
            whole_words_only=True
        )
        
        Path(f'{MODEL_NAME}_{DATASET_NAME}').mkdir(parents=True, exist_ok=True)
        # Save results
        with open(f'{MODEL_NAME}_{DATASET_NAME}/template_sentence_probs_128256_whole_words.pkl', 'wb') as f:
            pickle.dump(sentence_probs, f)
        with open(f'{MODEL_NAME}_{DATASET_NAME}/template_sentence_logits_128256_whole_words.pkl', 'wb') as f:
            pickle.dump(sentence_logits, f)