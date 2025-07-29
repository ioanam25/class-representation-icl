import torch
import transformers
from termcolor import colored
import os

MODEL_KEY = "meta-llama/Meta-Llama-3.1-8B"

def load_llama3_pipeline(MODELS_PATH = "models", TOKENIZERS_PATH = "tokenizers"):
    '''
        Load the Meta-Llama-3.1-8B model and tokenizer.

        Parameters:
            MODELS_PATH (str): Path to the directory where the model is stored.
            TOKENIZERS_PATH (str): Path to the directory where the tokenizer is stored.
        
        Returns:
            model (transformers.PreTrainedModel): The Meta-Llama-3-8B model.
            tokenizer (transformers.PreTrainedTokenizer): The Meta-Llama-3-8B tokenizer.
            pipeline (transformers.Pipeline): The Meta-Llama-3-8B-Instruct pipeline for text generation.
    '''

    pipeline = transformers.pipeline(
        "text-generation", model=MODEL_KEY,
        model_kwargs={
            "cache_dir" : MODELS_PATH,
            "torch_dtype" : torch.bfloat16,
            "attn_implementation":"eager"
        }, 
        max_new_tokens=2048, device_map='auto',
        token= os.getenv("HUGGINGFACE_TOKEN"),
    )
    
    print(colored('Pipeline loaded', 'green'))
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    return model, tokenizer, pipeline

def load_model_and_tokenizer():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_KEY,
        cache_dir =  "models",
        device_map='auto',
        attn_implementation='eager',
        torch_dtype = torch.bfloat16,
        local_files_only = False,
        token= os.getenv("HUGGINGFACE_TOKEN")
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_KEY,
        cache_dir="tokenizers",
        token= os.getenv("HUGGINGFACE_TOKEN"),
        local_files_only = False
    )
    print(colored(f'{model.name_or_path} Model and tokenizer loaded', 'green'))
    return model, tokenizer

def load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_KEY,
        cache_dir="tokenizers",
        token= "your_token_here",
        local_files_only = True
    )
    return tokenizer