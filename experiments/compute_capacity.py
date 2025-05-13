from LLMGeometry.gcmc import compute_capacity_from_labeled_array, compute_capacity_across_layers
from LLMGeometry.utils import dump_to_pickle, save_file_with_incremental_suffix
import pickle
from pathlib import Path
import argparse
import json
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

def participation_ratio(X):
    pca = PCA(n_components=X.shape[0])
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    p_ratio = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    return p_ratio

def gaussian_D_and_R(X, n_t=2000):
    '''
        Measures Gaussian dimension and radius of the data
    '''
    n, d = X.shape # n - number of samples, d - dimensionality
    X = X - np.mean(X, axis=0) # Center the data
    Ts = np.random.randn(n_t, d) # Random vectors
    anchor_ids = np.argmax(X @ Ts.T, axis=0) # Find the samples that maximize the dot product with random vectors
    anchor_vectors = X[anchor_ids]
    R = np.mean(np.linalg.norm(anchor_vectors, axis=1)) # Radius
    anchor_vectors_unit = anchor_vectors / np.linalg.norm(anchor_vectors, axis=1)[:, None]
    D = np.mean(np.diag(anchor_vectors_unit @ Ts.T) ** 2) # Dimension
    return R,D

def compute_additional_metrics_across_layers(metrics_df, embeddings, label_column):
    acc = []
    labels = metrics_df[label_column]
    
    for N_layer in tqdm(range(embeddings.shape[1])):
        max_dist,R,D,p_ratio = [],[],[],[]
        for l in labels.unique():
            mask = (labels == l)
            X = np.array(embeddings[mask, N_layer, :])
            max_dist.append(np.max(pdist(X)))
            R_,D_ = gaussian_D_and_R(X)
            R.append(R_)
            D.append(D_)
            p_ratio.append(participation_ratio(X))
        
        acc.append({
            'layer' : N_layer,
            'max_dist_R' : np.mean(max_dist), # Average maximum distance between samples of the same class
            'gaussian_R' : np.mean(R), # Average radius of the manifolds
            'gaussian_D' : np.mean(D), # Average Gaussian dimension of the manifolds
            'participation_ratio' : np.mean(p_ratio), # Average participation ratio of the class manifolds
        })
    
    return pd.DataFrame(acc)


def main(EMBEDDINGS_FILE_NAME, CAPACITY_TASK):
    '''
        Compute manifold capacity of the embeddings and save the results to a file.

        Parameters:
        -----------
        EMBEDDINGS_FILE_NAME : str
            Path to the file with embeddings. The file should contain a dictionary with the following keys:
            - 'metrics' : pd.DataFrame
                Dataframe with metrics of the model.
            - 'mean_pooled_embeddings' : np.array
                Array with mean-pooled embeddings.
            - 'last_token_embeddings' : np.array
                Array with last token embeddings.
            - 'CONFIG' : dict
                Dictionary with the configuration of the model.
        CAPACITY_TASK : str
            Task for which the capacity is computed. Should be one of the columns in the metrics dataframe.
    '''
    EMBEDDINGS_FILE_NAME = Path(EMBEDDINGS_FILE_NAME)
    with open(EMBEDDINGS_FILE_NAME, "rb") as f:
        embeddings = pickle.load(f)

    metrics = embeddings['metrics']
    mean_pooled_embeddings = embeddings['mean_pooled_embeddings']
    last_token_embeddings = embeddings['last_token_embeddings']

    capacity_mean_pooled = compute_capacity_across_layers(embeddings['metrics'], mean_pooled_embeddings, CAPACITY_TASK)
    additional_metrics_mean_pooled = compute_additional_metrics_across_layers(embeddings['metrics'], mean_pooled_embeddings, CAPACITY_TASK)
    capacity_mean_pooled = pd.merge(capacity_mean_pooled, additional_metrics_mean_pooled, on='layer')

    capacity_last_token = compute_capacity_across_layers(embeddings['metrics'], last_token_embeddings, CAPACITY_TASK)
    additional_metrics_last_token = compute_additional_metrics_across_layers(embeddings['metrics'], last_token_embeddings, CAPACITY_TASK)
    capacity_last_token = pd.merge(capacity_last_token, additional_metrics_last_token, on='layer')


    # --- Add token type column
    capacity_mean_pooled['token_type'] = 'mean_pooled'
    capacity_last_token['token_type'] = 'last_token'

    df = pd.concat([capacity_mean_pooled, capacity_last_token])

    # --- Copying attributes of metrics dataframe
    for key, value in metrics.attrs.items():
        df[key] = value

    # --- Copying selected CONFIG parameters to the dataframe (for both ICL and prompt tuning)
    for key, value in embeddings['CONFIG'].items():
        if key in ['MODEL_NAME', 'DATASET_NAME', 'prefix_type', 'n_examples', 'keyword', 'answer_field', 'SOFT_PROMPT_LENGTH', 'checkpoint_idx', 'train_loss', 'train_batch']:
            df[key] = value

    df['path'] = EMBEDDINGS_FILE_NAME

    # --- Save the results
    save_folder = EMBEDDINGS_FILE_NAME.parent
    save_path = save_folder / f"capacity_{CAPACITY_TASK}.pickle"
    dump_to_pickle(df, save_path)


def copy_config_main(EMBEDDINGS_FILE_NAME, CAPACITY_TASK):
    '''
        Opens the embeddings file and copies the CONFIG parameters to the dataframe with capacity results.
    '''
    # --- Load the embeddings file
    EMBEDDINGS_FILE_NAME = Path(EMBEDDINGS_FILE_NAME)
    with open(EMBEDDINGS_FILE_NAME, "rb") as f:
        embeddings = pickle.load(f)
    CONFIG = embeddings['CONFIG']

    # --- Load the capacity dataframe
    capacity_file = EMBEDDINGS_FILE_NAME.parent / f"capacity_{CAPACITY_TASK}.pickle"
    capacity_df = pickle.load(open(capacity_file, "rb"))

    # --- Copy the CONFIG parameters
    for key in ['MODEL_NAME', 'DATASET_NAME', 'keyword', 'answer_field', 'SOFT_PROMPT_LENGTH', 'checkpoint_idx', 'train_loss', 'train_batch']:
        capacity_df[key] = CONFIG[key]
    

    # --- Save the results
    new_filename = capacity_file.parent / f"capacity_{CAPACITY_TASK}_with_CONFIG.pickle"
    with open(new_filename, "wb") as f:
        pickle.dump(capacity_df, f)
    print(f"Results saved to {new_filename}")



def main_SLURM():
    # Use this function to run the script on SLURM
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
    main(job_config['EMBEDDINGS_FILE_NAME'], job_config['CAPACITY_TASK'])



if __name__ == "__main__":
   # main_SLURM()
   main(
        EMBEDDINGS_FILE_NAME="/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/ICL/results/claude_multitask/llama3.1_base/emotion/demos/Category:40_examples/run_1/embeddings.pickle",
        CAPACITY_TASK="emotion"
   )
  