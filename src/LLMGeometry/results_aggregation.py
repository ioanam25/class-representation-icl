from pathlib import Path
import pickle
from tqdm import tqdm
from termcolor import colored
import pandas as pd

def extract_capacity_results_from_SLURM_run(SLURM_JOB_ID, expand_result_keys=True):
    paths = list(Path(f'/mnt/home/akirsanov/ceph/LLM_Geometry/SLURM_OUTPUT/{SLURM_JOB_ID}/results').glob('*.pickle'))
    print(f'Found {colored(len(paths), "magenta")} paths')

    acc_results = []
    
    for path in tqdm(paths):
        data = pickle.load(open(path, 'rb'))
        acc_results.append({})
        
        for key in data['manifold_metadata'].keys():
            acc_results[-1][key] = data['manifold_metadata'][key]

        if expand_result_keys:
            for key in data['manifold_result'].keys(): # Store all the keys from manifold_result
                acc_results[-1][key] = data['manifold_result'][key]
        else:
            acc_results[-1]['manifold_result'] = data['manifold_result']
        
    df = pd.DataFrame(acc_results)
    return df