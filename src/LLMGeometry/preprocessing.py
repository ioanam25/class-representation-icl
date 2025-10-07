import torch
from torch.utils.data import Dataset
import LLMGeometry.models.llama3_8b_base.preprocessing
import LLMGeometry.models.llama3_1b_base.preprocessing
import LLMGeometry.models.llama3_70b_instruct.preprocessing
import LLMGeometry.models.gemma2_2b_base.preprocessing
import LLMGeometry.models.mistral_7b_base.preprocessing

class DataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Return the row as a dictionary
        return self.df.iloc[idx].to_dict()


def prepare_batch(batch, model, tokenizer, soft_prompt_embeds=None, soft_prompt_location='before', prompt_text_field='text', answer_field=None, verbose=False, add_special_tokens=True):
    ''' 
        Prepare a batch for the model forward pass for the specified model.
    '''
    if model.name_or_path == 'meta-llama/Meta-Llama-3-8B' or model.name_or_path == 'meta-llama/Meta-Llama-3.1-8B':
        return LLMGeometry.models.llama3_8b_base.preprocessing.prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_prompt_embeds, soft_prompt_location=soft_prompt_location, prompt_text_field=prompt_text_field, answer_field=answer_field, verbose=verbose, add_special_tokens=add_special_tokens)

    if model.name_or_path == 'meta-llama/Llama-3.2-1B':
        return LLMGeometry.models.llama3_1b_base.preprocessing.prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_prompt_embeds, soft_prompt_location=soft_prompt_location, prompt_text_field=prompt_text_field, answer_field=answer_field, verbose=verbose, add_special_tokens=add_special_tokens)

    if model.name_or_path == '/gpfs/data/oermannlab/public_models/llama_models_hf/llama-3.1-70b-instruct':
        return LLMGeometry.models.llama3_70b_instruct.preprocessing.prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_prompt_embeds, soft_prompt_location=soft_prompt_location, prompt_text_field=prompt_text_field, answer_field=answer_field, verbose=verbose, add_special_tokens=add_special_tokens)

    if model.name_or_path == 'google/gemma-2-2b':
        return LLMGeometry.models.gemma2_2b_base.preprocessing.prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_prompt_embeds, soft_prompt_location=soft_prompt_location, prompt_text_field=prompt_text_field, answer_field=answer_field, verbose=verbose, add_special_tokens=add_special_tokens)
    
    if model.name_or_path == 'mistralai/Mistral-7B-v0.3':
        return LLMGeometry.models.mistral_7b_base.preprocessing.prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_prompt_embeds, soft_prompt_location=soft_prompt_location, prompt_text_field=prompt_text_field, answer_field=answer_field, verbose=verbose, add_special_tokens=add_special_tokens)
    
    raise ValueError(f"Model {model.name_or_path} not supported.")