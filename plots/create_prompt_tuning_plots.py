import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy
import seaborn as sns
import cmasher
from termcolor import colored
import matplotlib.ticker as ticker
sns.set_style('whitegrid')

# Lower opacity of grid lines
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['svg.fonttype'] = 'none'

def get_palette_from_cmap(cmap, series):
    unique_values = sorted(series.unique())
    palette = {value: cmap(i/len(unique_values)) for i, value in enumerate(unique_values)}
    return palette



def load_capacity_df(dataset_name, model_name):
    aggregated_filename = f"/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/prompt_tuning/{dataset_name}_{model_name}_capacity_aggregated.pickle"
    if Path(aggregated_filename).exists():
        with open(aggregated_filename, "rb") as f:
            capacity_df = pickle.load(f)
            print(colored(f"Loaded aggregated data from {aggregated_filename}", "green"))
        return capacity_df
    else:
        root_path = "/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/prompt_tuning/results/"
        paths = list(Path(root_path).glob(f"{dataset_name}/{model_name}/*/*/*_tokens/checkpoint_*/capacity_*_with_CONFIG.pickle"))
        print(colored(f"Found {len(paths)} capacity files", "green"))
        acc_dfs = []
        for path in tqdm(paths):
            with open(path, "rb") as f:
                data = pickle.load(f)
            data['path'] = str(path)
            acc_dfs.append(data)

        capacity_df = pd.concat(acc_dfs)
        capacity_df.to_pickle(aggregated_filename)
        print(colored(f"Saved aggregated data to {aggregated_filename}", "green"))
        return capacity_df



def plot_cross_tasks_geometry(prompt_tuning_df, ICL_df, y_vars, token_type, prompt_length, layer=None):
    if not isinstance(y_vars, list):
        y_vars = [y_vars]


    prompt_tuning_df = prompt_tuning_df.query(f"token_type == '{token_type}' and SOFT_PROMPT_LENGTH == {prompt_length}").copy()
    prompt_tuning_df.train_batch = prompt_tuning_df.train_batch + 1 
    max_examples = ICL_df.query(f"token_type == '{token_type}' and prefix_type == 'demos'").n_examples.max()

    ICL_subset_df = pd.concat([
        ICL_df.query(f"token_type == '{token_type}' and prefix_type == 'instruction'"),
        ICL_df.query(f"token_type == '{token_type}' and prefix_type == 'demos' and n_examples == {max_examples}")
    ]) # Selecting only the last example for demos and instruction for comparison with prompt tuning


    fig, axs_grid = plt.subplots(len(y_vars),3, figsize=(14,3.5*len(y_vars)), dpi=300, sharey=False, sharex='col', squeeze=False)

    

    fig, axs_grid = plt.subplots(len(y_vars),3, figsize=(14,3.5*len(y_vars)), dpi=300, sharey=False, sharex='col', squeeze=False)

    kwargs_prompt_tuning = {
        'lw': 1.1,
        'errorbar': None,
        'marker' : 'o',
        'mew' : 0,
        'legend' : False,
        'alpha' : 1,
        'ms' : 0
    }

    kwargs_ICL = {
        'lw': 2.2,
       'errorbar': None,
        'legend' : False,
        'ls' : '--'
    }

    for y_var, axs in zip(y_vars, axs_grid):
        sns.lineplot( data=prompt_tuning_df.query('aligned == True'), x='layer', y=y_var, hue='checkpoint_idx',ax=axs[0],**kwargs_prompt_tuning,
            palette=get_palette_from_cmap(cmasher.get_sub_cmap(plt.cm.Greens, 0.4, 1), prompt_tuning_df.checkpoint_idx),
        ) 

        sns.lineplot(
            data=ICL_subset_df.query('aligned == True'), x='layer', y=y_var, ax=axs[0], hue = 'prefix_type', **kwargs_ICL,
            palette={
                'instruction' : 'black',
                'demos' : 'green'
            },
            style = 'prefix_type',
            dashes={
                'instruction': (1, 1),  # Dotted line
                'demos': (4, 2)         # dashed line
            }
            
        )

        sns.lineplot(data=prompt_tuning_df.query('aligned == False'),x='layer', y=y_var, hue='checkpoint_idx', palette=get_palette_from_cmap(cmasher.get_sub_cmap(plt.cm.Reds, 0.4, 1), prompt_tuning_df.checkpoint_idx), ax=axs[1], **kwargs_prompt_tuning
        )

        sns.lineplot(
            data=ICL_subset_df.query('aligned == False'), x='layer', y=y_var, ax=axs[1], hue = 'prefix_type', **kwargs_ICL,
            palette={
                'instruction' : 'black',
                'demos' : 'red'
            },
            style = 'prefix_type',
            dashes={
                'instruction': (1, 1),  # Dotted line
                'demos': (4, 2)         # dashed line
            }
            
        )

        sns.lineplot(
            data=prompt_tuning_df.query(f'layer == {layer}'),
            x='train_batch',
            y=y_var,
            hue='aligned',
            palette={True : 'green', False : 'red'},
            ax=axs[2],
            lw=3,
            legend=False,
            err_kws=dict(alpha=0.1),
        )

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('')


        for ax in axs[:2]:
            ax.axvline(layer, color='black', linestyle='--', lw=1)
            ax.set_xlabel('Layer', fontsize=22)
           # ax.tick_params(axis='both', rotation=0, labelsize=14)

        axs[2].set_xlabel('Train iteration', fontsize=22)
        axs[2].set_xscale('log')
        axs[2].axhline(ICL_df.query(f"token_type == '{token_type}' and prefix_type == 'instruction' and layer == {layer} and aligned == True")[y_var].mean(), color='green', lw = kwargs_ICL['lw'], ls='dotted')
        axs[2].axhline(ICL_df.query(f"token_type == '{token_type}' and prefix_type == 'instruction' and layer == {layer} and aligned == False")[y_var].mean(), color='red', lw = kwargs_ICL['lw'], ls='dotted')


    
        for ax in axs_grid.flatten():
            ax.tick_params(axis='both', rotation=0, labelsize=20)
    
        axs_grid[0,0].set_title('Coherent', fontsize=24, fontweight='bold')
        axs_grid[0,1].set_title('Incoherent', fontsize=24, fontweight='bold')
        axs_grid[0,2].set_title(f'Cross-section\nat layer {layer}', fontsize=20, fontweight='bold')
    
        for y_var, ax in zip(y_vars, axs_grid[:,0]):
            if y_var == 'capacity_norm':
                ax.set_ylabel('Capacity ( \ raw)', fontsize=20)
            elif y_var == 'participation_ratio_norm':
                ax.set_ylabel('Dimension ( \ raw)', fontsize=20)
            elif y_var == 'max_dist_R_norm':
                ax.set_ylabel('Radius ( \ raw)', fontsize=20)
            elif y_var == 'axes_alignment_norm':
                ax.set_ylabel('Axes alignment\n ( \ raw)', fontsize=20)
            elif y_var == 'center_axes_alignment_norm':
                ax.set_ylabel('Center-axes\nalignment ( \ raw)', fontsize=20)
        fig.tight_layout()
    return fig




def main():
    MODEL_NAME = 'llama3.1_base' 
    master_df = load_capacity_df('claude_multitask', MODEL_NAME)
    master_df.train_batch = master_df.train_batch + 1
    master_df.SOFT_PROMPT_LENGTH = master_df.SOFT_PROMPT_LENGTH.astype('category')
    performance_df = master_df.query(f'layer == 0 and capacity_task == "{master_df.capacity_task.iloc[0]}" and token_type == "{master_df.token_type.iloc[0]}"') # Selecting only a subset of capacity data for performance analysis

    ICL_df = pickle.load(open('/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/ICL/claude_multitask_llama3.1_base_capacity_aggregated.pickle', 'rb'))
    ICL_df = ICL_df.query('answer_field in ["emotion", "intent", "topic"] and keyword == "Category"')
    ICL_performance_df = ICL_df.query('layer == 0 and token_type == "last_token" and capacity_task == "emotion"')



    baseline_df = ICL_df.query('prefix_type == "raw" and keyword == "Category"')
    instruction_df = ICL_df.query('prefix_type == "instruction" and keyword == "Category"')
    
    # --- Aligning the tasks ---
    for task_icl in ['emotion', 'topic', 'intent']:
        for task_capacity in ['emotion', 'topic', 'intent']:
            keyword = 'Category' #TASK_keywords[task_icl]
            master_df.loc[ (master_df.capacity_task == task_capacity) & (master_df.answer_field == task_icl) & (master_df.keyword == keyword), 'aligned'] = (task_icl == task_capacity)
            ICL_df.loc[ (ICL_df.capacity_task == task_capacity) & (ICL_df.keyword == keyword) & (ICL_df.answer_field == task_icl), 'aligned'] = (task_icl == task_capacity)


        
    for token_type in ['mean_pooled', 'last_token']:
        for prompt_len in [5]:
            fig = plot_cross_tasks_geometry(master_df, ICL_df, ['capacity_norm', 'participation_ratio_norm', 'max_dist_R_norm', 'axes_alignment_norm', 'center_axes_alignment_norm'], token_type, prompt_len, layer=31)
    
    
            save_dir = Path("/mnt/home/akirsanov/LLMGeometry/naacl_2025/plots/prompt_tuning/")
            (save_dir / 'svg').mkdir(exist_ok=True, parents=True)
    
            fig.savefig(save_dir / f"svg/{MODEL_NAME}_{token_type}_prompt_tuning_{prompt_len}.svg")
            fig.savefig(save_dir / f"{MODEL_NAME}_{token_type}_prompt_tuning_{prompt_len}.png")


if __name__ == '__main__':
    main()
