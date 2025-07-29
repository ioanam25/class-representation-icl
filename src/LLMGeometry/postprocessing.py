import torch
import numpy as np
from LLMGeometry.utils import create_causal_mask, take_last_from_attention_mask
import pandas as pd
from collections import defaultdict
import warnings

def get_token_probability(logits_or_probs, token_string_or_id, attention_mask, tokenizer=None,  target_token_offset=None, by_element_in_batch=False):
    '''
        Returns logit/probability of a target token in a batched Tensor of logits/probabilities.

        Parameters:
        ----------
        logits_or_probs: torch.Tensor
            The logits/probabilities to extract elements from. Can be 1D, 2D or 3D tensor.
                If 1D, the tensor should have shape (vocab_size).
                If 2D, the tensor should have shape (batch_size, vocab_size).
                If 3D, the tensor should have shape (batch_size, sequence_length, vocab_size).
        token_string_or_id: str or int
            The token to compute the probability of. If a string, the tokenizer must be provided.
        attention_mask: torch.Tensor
            The attention mask to apply to the logits/probabilities. Should have shape (batch_size, sequence_length).
        tokenizer: transformers.PreTrainedTokenizer, optional
            The tokenizer to encode the token string. Ignored if token_string_or_id is an integer.
        target_token_offset: int, optional
            The offset of the target token in the sequence. If None, the probability of all tokens is returned, with NaNs for masked tokens.
        by_element_in_batch: bool, optional
            If True, the token_string_or_id should be a list of strings, with the length equal to the batch size.
    '''
    if logits_or_probs.requires_grad:
        logits_or_probs=logits_or_probs.detach().float()

    if logits_or_probs.ndim==1: 
        # If we are given a 1D tensor of logits (vocab_size)
        logits_or_probs = logits_or_probs.reshape(1,1,len(logits_or_probs))
    elif logits_or_probs.ndim==2:
        # 2D tensor of logits (sequence_length, vocab_size)
        logits_or_probs = logits_or_probs.unsqueeze(2)
    elif logits_or_probs.ndim>3:
        raise(ValueError("The number of dimensions is greater than 3"))
    batch_size = logits_or_probs.shape[0]
    sequence_len = logits_or_probs.shape[1]


    if by_element_in_batch: # If we are provides with (batch_len) unique tokens
        assert isinstance(token_string_or_id, list)
        if len(token_string_or_id)!=batch_size:
            return ValueError("Number of tokens should equal to the number of elements in a batch, when by_element_in_batch is True")
        targets = token_string_or_id
    else:
        targets = [token_string_or_id for _ in range(batch_size)] # Else, we are computing the probability of the same token for each element in a batch

    # --- Tokenizing target strings if necessary
    target_token_ids = []
    
    for t in targets:
        if isinstance(t, str):
            if tokenizer is None:
                raise ValueError(f"Tokenizer must be provided to convert token string {t}")
            token_id = tokenizer.encode(t, add_special_tokens=False)
            if len(token_id)>1:
                warnings.warn(f"Token {t} was split into multiple tokens")
                print(tokenizer.convert_ids_to_tokens(token_id))
                print(t, token_id, tokenizer.decode(token_id))
            token_id=token_id[0]
        else:
            token_id = t
        target_token_ids.append(token_id)

  # target_token_ids = torch.Tensor(target_token_ids)
    
    if attention_mask is None:
        attention_mask = torch.ones((batch_size,sequence_len)).to(logits_or_probs.device)


    # Selecting logits/probabilities for the target tokens
    selected_logits_or_probs = logits_or_probs[torch.arange(batch_size), :, target_token_ids]
    
    if target_token_offset is None:
        # In this case, we are computing the value of the target token across all elements in a sequence (attention_mask is used to set NaNs)
        return torch.where(attention_mask == 1, selected_logits_or_probs, float('nan')).float().cpu().numpy()

    return take_last_from_attention_mask(selected_logits_or_probs, attention_mask, target_token_offset).float().cpu().numpy()



def get_predictions_from_logits(logits, attention_mask=None, N=3, return_strings=False, tokenizer=None, target_token_offset=None, constrained_token_ids=None):
    '''
        Returns the top N predictions from the logits.

        Parameters:
        ----------
        logits: torch.Tensor
            The logits to compute the predictions from.
        constrained_token_ids: list or torch.Tensor, optional
            If provided, only compute softmax over these token IDs (constrained classification)
    '''
    if logits.requires_grad:
        logits=logits.detach().float()

    if logits.ndim==1: 
        # If we are given a 1D tensor of logits (vocab_size)
        logits = logits.reshape(1,1,len(logits))
    elif logits.ndim==2:
        # 2D tensor of logits (sequence_length, vocab_size)
        logits = logits.unsqueeze(2)
    elif logits.ndim>3:
        raise(ValueError("The number of dimensions is greater than 3"))

    batch_size = logits.shape[0]
    sequence_len = logits.shape[1]
    
    if attention_mask is None:
        attention_mask = torch.ones((batch_size,sequence_len)).to(logits.device)
        
    if constrained_token_ids is not None:
        # Constrained classification: only compute softmax over specified tokens
        if isinstance(constrained_token_ids, list):
            constrained_token_ids = torch.tensor(constrained_token_ids)
        
        # Extract logits for constrained tokens only
        constrained_logits = logits[:, :, constrained_token_ids]  # (batch, seq, constrained_vocab)
        probs = torch.softmax(constrained_logits, dim=2)
        
        # Get top N from constrained set
        selected_probs_indices = torch.argsort(probs, dim=2, descending=True)[:,:,:N].cpu().numpy()
        # Map back to original token IDs
        selected_ids = constrained_token_ids[selected_probs_indices].cpu().numpy()
        
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, sequence_len, N)
        seq_indices = torch.arange(sequence_len).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, N)
        selected_probs = probs[batch_indices, seq_indices, selected_probs_indices].float()
    else:
        # Original unconstrained classification
        probs = torch.softmax(logits, dim=2)
        selected_ids = torch.argsort(probs, dim=2,descending=True)[:,:,:N].cpu().numpy()
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, sequence_len, N)
        seq_indices = torch.arange(sequence_len).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, N)
        selected_probs = probs[batch_indices, seq_indices, selected_ids].float()

    # Applying attention mask
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, N)
    selected_probs = torch.where(attention_mask_expanded == 1, selected_probs, float('nan')).cpu().numpy()

    if target_token_offset is None:
        # Return all tokens
        if return_strings:
            token_strings = []
            for i in range(batch_size):
                token_strings.append([])
                for j in range(sequence_len):
                    token_strings[-1].append([tokenizer.decode(t) for t in selected_ids[i][j]])
            token_strings=np.array(token_strings)
            return token_strings,selected_probs
        return selected_ids, selected_probs
        
    # Return only predictions from the target token (index of last non-zero in each example in a batch - 1 - target_token_offset)
    target_token_indices = (torch.argmin(attention_mask, dim=1) - 1 - target_token_offset).cpu().numpy()
    if return_strings:
        token_strings = []
        for i in range(batch_size):
            token_strings.append([tokenizer.decode(t) for t in selected_ids[i][target_token_indices[i]]])
            
        token_strings=np.array(token_strings)
        return token_strings,selected_probs[torch.arange(batch_size), target_token_indices, :]
    return selected_ids[torch.arange(batch_size), target_token_indices, :], selected_probs[torch.arange(batch_size), target_token_indices, :]
