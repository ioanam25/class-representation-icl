from LLMGeometry.datasets import load_dataset_by_name
from LLMGeometry import load_model_and_tokenizer
from LLMGeometry.evaluation import run_on_dataframe, wrap_in_dataloader, hooked_forward_pass
from LLMGeometry.preprocessing import prepare_batch
from LLMGeometry.prompt_tuning import initialize_soft_prompt, train_soft_prompt, TRAINING_CONFIG
from tqdm import tqdm
import numpy as np

import torch
import pandas as pd
import pickle
from termcolor import colored
from pathlib import Path
import argparse
import json
import sys


def gen_log_space(limit, n):
    '''
        Generate n numbers that fall within the range [0, limit-1] with a log distribution.
        First few numbers will be distributed linearly, then switch to log spacing.

        Taken from: https://stackoverflow.com/a/12421509
    '''
    result = [1]
    if n>1: 
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)



def main(MODEL_NAME, DATASET_NAME, answer_field, SOFT_PROMPT_LENGTH, keyword='Category', n_checkpoints=50, root_folder="/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/prompt_tuning/results"):
    # --- Loading the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    datasets = load_dataset_by_name(DATASET_NAME)

    train_df = datasets['train']
    test_df = datasets['test']
    train_df[answer_field] = train_df[answer_field].astype(str)
    test_df[answer_field] = test_df[answer_field].astype(str)

    # --- Adding space to the answer field if needed
    if tokenizer.name_or_path in ['mistralai/Mistral-7B-v0.3']:
        pass # Mistral uses sentencepiece tokenizer, which does not require space before the answer
    else:
        train_df[answer_field] = " " + train_df[answer_field] # Adding space before the answer
        test_df[answer_field] = " " + test_df[answer_field] # Adding space before the answer

    soft_embeds = initialize_soft_prompt(SOFT_PROMPT_LENGTH, model, tokenizer, soft_prompt_initial_text=keyword, random_init=False) # Initialize the soft prompt

    # --- Preparing the data
    suffix = f'\n{keyword}:'
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    suffix_crop = len(suffix_tokens) - 1 # Number of tokens in the suffix to crop (-1 because \n will get fused with "." token at the end of the sentence)

    train_df['text_with_suffix'] = 'Text: ' + train_df['text'] + suffix
    test_df['text_with_suffix'] = 'Text: ' + test_df['text'] + suffix

    train_dataloader = wrap_in_dataloader(train_df, batch_size=TRAINING_CONFIG['BATCH_SIZE'], shuffle=True)

    # --- Training the soft prompt
    train_metrics = train_soft_prompt(
        train_dataloader, model, tokenizer, 'text_with_suffix', answer_field, soft_embeds
    )
    # --- Extracting checkpoints
    checkpoint_idx = gen_log_space(len(train_metrics.train_batch), n_checkpoints)

    # --- Running the model on the test data and saving the results
    for idx in checkpoint_idx:
        print(colored(f"Running the model on the test data with checkpoint {idx}", 'magenta'))
        soft_embeds = train_metrics.soft_embeds_tensor[idx].to(model.device)
        output = run_on_dataframe(test_df, model, tokenizer, 'text_with_suffix', answer_field, return_hidden_states=True, batch_size=10, soft_embeds=soft_embeds, soft_prompt_location='before', return_raw_batches=True)

        output['mean_pooled_embeddings'] = torch.stack([s[:, SOFT_PROMPT_LENGTH+1:-suffix_crop, :].mean(dim=1).float().cpu() for s in output['hidden_states']]) # Mean pooling over the tokens, excluding the suffix and soft embeddings
        output['last_token_embeddings'] = torch.stack([s[:, -1, :].float().cpu() for s in output['hidden_states']]) # Last token embeddings

        CONFIG = {
            'MODEL_NAME': MODEL_NAME,
            "DATASET_NAME": DATASET_NAME,
            'keyword': keyword,
            'answer_field' : answer_field,
            'SOFT_PROMPT_LENGTH': SOFT_PROMPT_LENGTH,
            'checkpoint_idx': idx,
            'train_loss' : train_metrics.train_loss_total[idx],
            'TRAINING_CONFIG': TRAINING_CONFIG,
            'train_batch' : train_metrics.train_batch[idx],
            'soft_embeds' : soft_embeds.detach().clone().cpu(),
        }

        save_folder = Path(root_folder) / f"{DATASET_NAME}" / f"{MODEL_NAME}" / f"{answer_field}" / f"{keyword}" / f"{SOFT_PROMPT_LENGTH}_tokens" / f"checkpoint_{idx}"
        save_folder.mkdir(parents=True, exist_ok=True)
        with open(save_folder / "embeddings.pickle", "wb") as f:
            pickle.dump({
                'metrics' : output['metrics'],
                'mean_pooled_embeddings' : output['mean_pooled_embeddings'],
                'last_token_embeddings' : output['last_token_embeddings'],
                'batches' : output['batches'],
                'CONFIG': CONFIG,
            }, f)
        print(colored(f"Results saved to {save_folder}", 'green'))



def main_SLURM():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobind", help="ID of the job (DISBATCH_REPEAT_INDEX)")
    parser.add_argument("jobs_file", help="Path to the JSON file with job parameters")
    args = parser.parse_args()
    # --- Loading JSON with job parameters
    with open(args.jobs_file) as f:
        jobs = json.load(f)
    jobind = int(args.jobind)
    if jobind > len(jobs) - 1:
        print(f'Job index {jobind} is out of range, exiting...')
        sys.exit(0)
    job_config = jobs[jobind] # Parameters of a single job
    # --- Run the job
    main(**job_config)



if __name__ == "__main__":

   # main_SLURM()
    main(
        MODEL_NAME='llama3.1_base',
        DATASET_NAME='claude_multitask',
        answer_field='emotion',
        SOFT_PROMPT_LENGTH=10,
    )
