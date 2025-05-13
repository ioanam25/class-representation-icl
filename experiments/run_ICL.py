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
            'mean_pooled_embeddings' : output['mean_pooled_embeddings'],
            'last_token_embeddings' : output['last_token_embeddings'],
            'batches' : output['batches'],
            'CONFIG': CONFIG,
        }, f)


def create_template(prefix_type, n_examples, keyword, answer_field, dataset_name, shuffle_labels=False, seed=None):
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
    ds = load_dataset_by_name(dataset_name)
    train_df = ds['train']
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
    return prefix, suffix




def main(MODEL_NAME, DATASET_NAME, prefix_type, n_examples, keyword, answer_field, N_RUNS=50, root_folder="/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/ICL_stability/results"):
    
    root_folder = Path(root_folder)

    # --- Loading the model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # --- Loading the dataset
    datasets = load_dataset_by_name(DATASET_NAME)
    train_df = datasets['train']
    test_df = datasets['test']

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


    folder_with_runs = root_folder / f"{DATASET_NAME}" / f"{MODEL_NAME}" / f"{answer_field}" / f"{prefix_type}" / f"{keyword}:{n_examples}_examples" # Folder with runs for a given number of examples in context
    n_existing_runs = len(list(folder_with_runs.glob("run_*"))) # Number of existing runs


    for k in range(N_RUNS):
        # --- Running ICL
        print(colored(f"Running ICL with {n_examples} examples in context, run {k}", 'magenta'))

        # --- Creating the ICL prompt, by selecting random examples from the training data and applying the demonstrations template
        prefix, suffix = create_template(prefix_type, n_examples, keyword, answer_field, DATASET_NAME)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        print('Suffix tokens:', [tokenizer.decode([t]) for t in suffix_tokens])
        suffix_crop = len(suffix_tokens) - 1 # Number of tokens in the suffix to crop (-1 because \n will get fused with "." token at the end of the sentence)


        test_df['text_with_suffix'] = 'Text: ' + test_df['text'] + suffix

        output = run_on_dataframe(test_df, model, tokenizer, 'text_with_suffix', answer_field, return_hidden_states=True, batch_size=10,
                        prefix_text=prefix, return_raw_batches=True)
        
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

        output['mean_pooled_embeddings'] = torch.stack([s[:, :-suffix_crop, :].mean(dim=1).float().cpu() for s in output['hidden_states']]) # Mean pooling over the tokens, excluding the suffix
        output['last_token_embeddings'] = torch.stack([s[:, -1, :].float().cpu() for s in output['hidden_states']]) # Last token embeddings

        CONFIG = {
            'MODEL_NAME': MODEL_NAME,
            "DATASET_NAME": DATASET_NAME,
            'prefix_type': prefix_type,
            'n_examples': n_examples,
            'keyword': keyword,
            'answer_field' : answer_field,
            'run' : k,
            'prefix_text' : prefix,
            'suffix' : suffix,
        }

        total_run_id = n_existing_runs + k # Total number of runs
        save_folder = folder_with_runs / f"run_{total_run_id}" 

        save_outputs(output, save_folder, CONFIG)
        print(colored(f"Results saved to {save_folder}", 'green'))



def main_SLURM():
    # Using this function to run the script with SLURM 
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
    # Example of how to run the script
    main(
        MODEL_NAME = 'llama3.1_base',
        DATASET_NAME = 'claude_multitask',
        prefix_type = 'demos',
        n_examples = 5,
        keyword = 'Category',
        answer_field = 'emotion_letter',
        N_RUNS = 50,
        root_folder = "temp/ICL_results"
    )