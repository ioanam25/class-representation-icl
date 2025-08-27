import torch
import torch
import numpy as np
from pathlib import Path
import h5py
import pickle
from transformers.cache_utils import HybridCache


def create_attention_mask_from_sequence_lengths(seq_lengths):
    '''
        Create an attention mask from a list of sequence lengths.

        Parameters:
        ----------
        seq_lengths: List[int] – A list of sequence lengths (batch_size elements).

        Returns:
        ----------
        torch.Tensor: A 2D tensor of shape (batch_size, max_length) with 1s in the positions corresponding to the sequence lengths.
    '''
    max_len = max(seq_lengths)
    attention_mask = torch.zeros(len(seq_lengths), max_len)
    for i, l in enumerate(seq_lengths):
        attention_mask[i, :l] = 1
    return attention_mask


def create_causal_mask(attention_mask):
    """
    Convert a batch 1D attention masks into a 4D causal attention mask.
    
        Args:
        attention_mask (torch.Tensor): 2D tensor of shape (batch_size, seq_length)
        
        Returns:
        torch.Tensor: 4D causal attention mask of shape (batch_size, 1, seq_length, seq_length)
    """
    batch_size, seq_length = attention_mask.size()
    
    # Create a causal mask
    causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=attention_mask.device))
    
    # Expand attention_mask to 3D
    expanded_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)
    
    # Combine causal mask with padding mask
    causal_attention_mask = causal_mask.unsqueeze(0) * expanded_mask
    
    # Convert to float and replace 0s with large negative value
    causal_attention_mask = causal_attention_mask.to(dtype=torch.float32)
    causal_attention_mask = causal_attention_mask.masked_fill(causal_attention_mask == 0, float('-inf'))
    
    return causal_attention_mask.unsqueeze(1)


def take_last_from_attention_mask(X, attention_mask, offset=0):
    '''
        Takes elements from X that correspond to the last non-zero element in each sequence in a batch, minus the offset, specified by the attention mask.

        Parameters:
        ----------
        X: torch.Tensor – The tensor to take elements from. Should be a tensor of shape (batch_size, sequence_length, ...)
        attention_mask: torch.Tensor – The attention mask to take the last elements from. Should be 2D tensor of shape (batch_size, sequence_length).
        offset: int, optional – The offset to subtract from the last non-zero element in each sequence.
    '''
    target_indices = (torch.argmin(attention_mask, dim=1) - 1 - offset)
    if X.ndim==2:
        return X[torch.arange(X.shape[0]), target_indices]
    if X.ndim==3:
        return X[torch.arange(X.shape[0]), target_indices, :]
    if X.ndim==4:
        return X[torch.arange(X.shape[0]), target_indices, :, :]
    if X.ndim==5:
        return X[torch.arange(X.shape[0]), target_indices, :, :, :]
    if X.ndim==6:
        return X[torch.arange(X.shape[0]), target_indices, :, :, :, :]


def expand_KV(past_key_values, batch_size):
    '''
        Expand the past key-values to the specified batch size (when passed a tuple of tensors).
    '''
    if past_key_values is None:  # Handle None (empty prefix) case
        return None
    if isinstance(past_key_values, HybridCache): # If past_key_values is a HybridCache object, we return it as is
        return past_key_values
    return [[x.expand(batch_size, -1, -1, -1) for x in past_key_values[n_layer]] for n_layer in range(len(past_key_values))]



def find_subtensor(tensor, subtensor,as_masks=True):
    '''
    Find all occurences of a subtensor in a tensor.

    Parameters:
        tensor (torch.Tensor): The tensor to search in.
        subtensor (torch.Tensor): The tensor to search for.
        as_masks (bool): If True, return masks of the occurences instead of indices.
    '''

    assert ((tensor.dim()==1) and (subtensor.dim()==1))
    N = len(subtensor)
    idx = []
    i=0
    while(i<len(tensor)-len(subtensor)+1):
        if torch.all(tensor[i:i+N]==subtensor):
            idx.append((i,i+N))
            i+=N
        else:
            i+=1

    if as_masks:
        masks = []
        for pair in idx:
            temp = torch.zeros_like(tensor,dtype=bool)
            temp[pair[0]:pair[1]] = True
            masks.append(temp)
        return masks
    return idx


def generate_token_variations(target):
    acceptable_variants = [target]
    acceptable_variants.append(' '+target)
    acceptable_variants.append(target.capitalize())
    acceptable_variants.append(' '+target.capitalize())
    
    return acceptable_variants


def find_tokenization_match(text_or_tensor, target, tokenizer, strict_match=False, device=None):
    '''
        Find all occurences of a target in a text using a tokenizer.

        Parameters:
            text (str or torch.Tensor): The text to search in.
            target (str): The target to search for.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            strict_match (bool): If True, only search for exact matches. If False, search for all possible tokenizations.

        Returns:
            List[torch.Tensor]: A list of masks of the occurences.
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acceptable_variants = [target]
    if not strict_match:
        acceptable_variants.append(' '+target) # When we are searching for a particular word, its tokenization often includes a leading whitespace
        acceptable_variants.append(target.capitalize())
        acceptable_variants.append(' '+target.capitalize())
        acceptable_variants.append(target+'-')
        acceptable_variants.append(' '+target+'-')
        acceptable_variants.append(target.capitalize()+'-')
        acceptable_variants.append(target+'s')
        acceptable_variants.append(' '+target+'s')
        acceptable_variants.append(target.capitalize()+'s')
        
    occurence_masks = []
    if isinstance(text_or_tensor, torch.Tensor):
        text_tokens = text_or_tensor
    else:
        text_tokens = tokenizer(text_or_tensor,return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        
        if text_tokens.dim() > 1:
            text_tokens = text_tokens.squeeze(dim=0)
        
    used_flag = torch.zeros_like(text_tokens,dtype=bool)


    for target_variant in acceptable_variants:
    
        target_tokens = tokenizer(target_variant,return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device).squeeze()
        if target_tokens.dim()==0:
            target_tokens = target_tokens.unsqueeze(0) # If the target is a single token, we need to unsqueeze it to make it a 1D tensor


        variant_masks = find_subtensor(
            text_tokens,
            target_tokens
        ) 
        if len(variant_masks)==0:
            continue # If a particular variant is not found, we continue to the next one

        for mask in variant_masks:
            # If the mask contains any tokens that have already been used, we skip it
            if torch.any(used_flag[mask]):
                continue
            occurence_masks.append(mask)
            used_flag[mask] = True

    return occurence_masks

def replace_from_dict(s, d):
    '''
        Replace substrings in a string according to a dictionary.

        Parameters:
            s (str): The string to replace substrings in.
            d (dict): The dictionary containing the replacements.
    '''
    for k, v in d.items():
        s = s.replace(k, v)
    return s

def get_instruction_type_dict(instruction_prompts):
    mapping = {}
    for key in instruction_prompts.keys():
        for prompt in instruction_prompts[key]:
            mapping[prompt] = key
    return mapping


def generate_random_samples(labels, N, seed=None):
    '''
        Randomly select a specified number of samples with as uniform coverage of labels as possible
    '''
    unique_labels= np.unique(labels)
    idx_by_label = {cat:np.arange(len(labels))[labels==cat] for cat in unique_labels}

    n_samples_per_category = N // len(unique_labels)
    n_remaining_samples = N - n_samples_per_category*len(unique_labels)

    acc_indices = []
    
    rng = np.random.default_rng(seed)
    for lab in unique_labels:
        acc_indices.extend(rng.choice(idx_by_label[lab],n_samples_per_category, replace=False))

    selected_categories = rng.choice(unique_labels, n_remaining_samples, replace=False)
    
    for lab in selected_categories:
        acc_indices.extend(rng.choice(idx_by_label[lab],1, replace=False))
        
    acc_indices = np.array(acc_indices)
    rng.shuffle(acc_indices) # Shuffle the indices to make sure they are not ordered by category
    return acc_indices
    
    
    
def save_file_with_incremental_suffix(file_path: Path):
    if not file_path.exists():
        return file_path

    file_name = file_path.stem
    file_ext = file_path.suffix
    parent_dir = file_path.parent

    # Find the highest existing suffix for the file name pattern
    suffix_num = 0
    for existing_file in parent_dir.glob(f"{file_name}_*{file_ext}"):
        try:
            suffix = existing_file.stem.split("_")[-1]
            if suffix.isdigit():
                suffix_num = max(suffix_num, int(suffix))
        except (IndexError, ValueError):
            pass

    # Increment the suffix and generate the new file name
    new_suffix = f"_{suffix_num + 1}"
    new_file_name = f"{file_name}{new_suffix}{file_ext}"
    new_file_path = parent_dir / new_file_name

    return new_file_path


def save_dict_to_h5(d, group):
    for key in d.keys():
        value = d[key]
        if type(value) in [str]:
            group.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            group.create_dataset(key, data=value)

def load_dict_from_h5(group):
    d = {}
    for key in group.keys():
        d[key] = group[key][()]
    return d

def dump_to_pickle(obj, filepath, verbose=True):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    if verbose:
        print(f"Saved to {filepath}")