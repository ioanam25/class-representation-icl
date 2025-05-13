import torch
import transformers
from termcolor import colored

MODEL_KEY = "mistralai/Mistral-7B-v0.3"

def load_model_and_tokenizer():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_KEY,
        cache_dir =  "/mnt/ceph/users/akirsanov/LLM_Geometry/models",
        device_map='auto',
        attn_implementation='eager',
        torch_dtype = torch.bfloat16,
        local_files_only = True,
        token= "your_token_here"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_KEY,
        cache_dir="/mnt/ceph/users/akirsanov/LLM_Geometry/tokenizers",
        token= "your_token_here",
        local_files_only = True,
        use_fast=False,
        legacy=False
    )
    print(colored(f'{model.name_or_path} Model and tokenizer loaded', 'green'))
    return model, tokenizer

def load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_KEY,
        cache_dir="/mnt/ceph/users/akirsanov/LLM_Geometry/tokenizers",
        token= "your_token_here",
        local_files_only = True,
        use_fast=False,
        legacy=False
    )
    return tokenizer