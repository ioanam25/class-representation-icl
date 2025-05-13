import numpy as np
from gcmc import gcmc_analysis
from gcmc.types import DichotomyType
import pandas as pd
from tqdm import tqdm

CONFIG = {
    'num_random_hyperplanes' : 100, # Number of random hyperplanes
    'seed' : 42
}

def add_bias(XtoT, bias='global_center_norm'):
    '''
        Add bias term to the input arrays.a

        Parameters:
        ----------
            - XtoT : list of np.ndarray of shape (D, N), where D is the dimension of the embeddings, and N is the number of samples.
            - bias : str or int, if 'global_center_norm', the bias is the global center of the input arrays, otherwise it is the provided value.
    '''
    global_center = np.mean(np.concatenate(XtoT, axis=1), axis=1, keepdims=True)
    # Subtract the global center
    for i in range(len(XtoT)):
        XtoT[i] = XtoT[i] - global_center

    if bias == 'global_center_norm':
        bias = np.linalg.norm(global_center)

    # --- Append bias term
    XtoT_with_bias = [np.concatenate( [x, bias*np.ones((1,x.shape[1]))], axis=0) for x in XtoT]
    return XtoT_with_bias



def compute_capacity_from_labeled_array(arr, labels, bias='global_center_norm'):
    '''
        Compute manifold capacity of array with labels.

        Parameters:
        ----------
            - arr : np.ndarray of shape (N, D), where N is the number of samples, and D is the dimension of the embeddings.
            - labels : np.ndarray of shape (N,), where each element is the label of the corresponding sample in arr.

        Returns:
        --------
            - pd.DataFrame : a DataFrame containing the manifold capacity results
    
    '''
    acc_results = []
    XtoT = [arr[labels==l, :].T for l in np.unique(labels)]
    XtoT = add_bias(XtoT, bias=bias)
    
    manifold_output = gcmc_analysis(
        XtoT,
        dichotomies= DichotomyType.ONE_VERSUS_REST,
        n_hyperplanes=CONFIG['num_random_hyperplanes']
    )

    output = {
        'capacity' : manifold_output.capacity,
        'axis_alignment' : manifold_output.axis_alignment,
        'center_alignment' : manifold_output.center_alignment,
        'center_axis_alignment' : manifold_output.center_axis_alignment,
    }
    return pd.DataFrame([output])

def compute_capacity_across_layers(metrics, embeddings, capacity_task, CONFIG={}, **kwargs):
    acc_dfs = []
    for N_layer in tqdm(range(embeddings.shape[1])):
        df = compute_capacity_from_labeled_array(embeddings[:,N_layer,:], metrics[capacity_task], **kwargs)
        df['layer'] = N_layer
        df['capacity_task'] = capacity_task
        acc_dfs.append(df)
    master_df = pd.concat(acc_dfs)
    for key in CONFIG.keys():
        try:
            master_df[key] = CONFIG[key]
        except:
            print(f"Could not add {key} to the DataFrame")
    return master_df