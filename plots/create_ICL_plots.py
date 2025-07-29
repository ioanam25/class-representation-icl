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
sns.set_style('whitegrid')

# Lower opacity of grid lines
plt.rcParams['grid.alpha'] = 0.2

TASK_PALETTES = {
    'emotion' : cmasher.get_sub_cmap(plt.cm.Blues, 0.4, 1),
    'category' : cmasher.get_sub_cmap(plt.cm.Greens, 0.4, 1),
    'intent' : cmasher.get_sub_cmap(plt.cm.Oranges, 0.4, 1),
    'topic' : cmasher.get_sub_cmap(plt.cm.Purples, 0.4, 1)
}

TASK_keywords = {
    'emotion' : 'Emotion',
    'category' : 'Category',
    'intent' : 'Intent',
    'topic' : 'Topic'
}


def get_palette_from_cmap(cmap, series):
    unique_values = sorted(series.unique())
    palette = {value: cmap(i/len(unique_values)) for i, value in enumerate(unique_values)}
    return palette

def load_capacity_df(dataset_name, model_name):

    # Adjust the paths based on your setup

    aggregated_filename = f"/mnt/home/akirsanov/ceph/LLM_Geometry/DATA/ICL/{dataset_name}_{model_name}_capacity_aggregated.pickle"
    if Path(aggregated_filename).exists():
        with open(aggregated_filename, "rb") as f:
            capacity_df = pickle.load(f)
            print(colored(f"Loaded aggregated data from {aggregated_filename}", "green"))
        return capacity_df
    else:
        root_path = "LLM_Geometry/DATA/ICL/results/"
        paths = list(Path(root_path).glob(f"{dataset_name}/{model_name}/*/*/*:*_examples/run_*/capacity_*.pickle"))
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
    

def plot_geometry(df, task, keyword, token_type, y_vars=['capacity', 'participation_ratio', 'max_dist_R', 'axes_alignment', 'center_axes_alignment']):
    task_palette = get_palette_from_cmap(TASK_PALETTES[task], df.query('prefix_type == "demos"').n_examples)
    task_df = df.query(f"capacity_task == '{task}' and keyword == '{keyword}' and token_type == '{token_type}'")
    task_df = task_df[ task_df['answer_field'].apply(lambda x: x.startswith(task)) ]

    no_demos_df = task_df[ task_df['prefix_type'] != 'demos' ]
    demos_df = task_df[ task_df['prefix_type'] == 'demos' ]
    demos_df['n_examples'] = demos_df['n_examples'].astype('category')

    # --- Plotting ---
    fig, axs = plt.subplots(len(y_vars), 3, figsize=(10, 3*len(y_vars)), dpi=300, sharey='row', sharex='col')
    kwargs = {
        'lw': 2.25,
        'errorbar': None,
        # 'marker' : 'o',
        # 'mew' : 0,
        # 'ms' : 3
    }
    for y_var, ax_row in zip(y_vars, axs):
        sns.lineplot(data=demos_df[demos_df.answer_field == task], x='layer', y=y_var, hue='n_examples', ax=ax_row[0], palette=task_palette, **kwargs, legend=False)
        sns.lineplot(data=demos_df[demos_df.answer_field == task + '_letter'], x='layer', y=y_var, hue='n_examples', ax=ax_row[1], palette=task_palette, **kwargs, legend=False)
        sns.lineplot(data=demos_df[demos_df.answer_field == task + '_shuffled'], x='layer', y=y_var, hue='n_examples', ax=ax_row[2], palette=task_palette, **kwargs, legend=False)

        for ax in ax_row:
            sns.lineplot(data=no_demos_df, x='layer',  y=y_var, hue='prefix_type', ax=ax, palette={'raw': '#ababab', 'instruction': '#4a4a4a', 'instruction_detailed' : '#262626'}, ls='--', lw=1.25, errorbar=None, legend=(ax is axs[-1,1]))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel('Layer', fontsize=20)
            ax.tick_params(axis='both', rotation=0, labelsize=16)

        if y_var == 'capacity':
            ax_row[0].set_ylabel('Capacity', fontsize=20)
        elif y_var == 'participation_ratio':
            ax_row[0].set_ylabel('Dimension', fontsize=20)
        elif y_var == 'max_dist_R':
            ax_row[0].set_ylabel('Radius', fontsize=20)
        elif y_var == 'axes_alignment':
            ax_row[0].set_ylabel('Axes\nalignment', fontsize=20)
        elif y_var == 'center_axes_alignment':
            ax_row[0].set_ylabel('Center-axes\nalignment', fontsize=20)

    legend = axs[-1,1].legend(title='', fontsize=20, title_fontsize=20, loc='upper left', bbox_to_anchor=(0, -0.3))
  #  sns.move_legend(axs[-1,1], loc='upper left', bbox_to_anchor=(0, -0.1), fontsize=20)
    axs[0,0].set_title("Original labels", fontsize=20)
    axs[0,1].set_title("Letter codes", fontsize=20)
    axs[0,2].set_title("Shuffled labels", fontsize=20)
    return fig



def plot_performance(df, task, keyword):
    task_palette = get_palette_from_cmap(TASK_PALETTES[task], df.query('prefix_type == "demos"').n_examples)
    task_df = df[ df['answer_field'].apply(lambda x: x.startswith(task)) ].query(f'keyword == "{keyword}"')

    no_demos_df = task_df[ task_df['prefix_type'] != 'demos' ]
    demos_df = task_df[ task_df['prefix_type'] == 'demos' ]
    demos_df['n_examples'] = demos_df['n_examples'].astype('category')

    fig, axs_grid = plt.subplots(2, 2, figsize=(10, 8), sharey=True, dpi=300)

    axs = axs_grid.flatten()

    sns.barplot(data=demos_df[demos_df.answer_field == task], hue='n_examples', y='accuracy', ax=axs[0], palette=task_palette, legend=False, err_kws={'lw' : 1.2})
    axs[0].set_title("Gold labels", fontsize=20)

    # 3) Demonstration with letter codes
    sns.barplot(data=demos_df[demos_df.answer_field == task + '_letter'], hue='n_examples', y='accuracy', ax=axs[1], palette=task_palette, legend=False, err_kws={'lw' : 1.2})
    axs[1].set_title("Letter codes", fontsize=20)

    # 4) Demonstration with shuffled labels (target accuracy)
    sns.barplot(data=demos_df[demos_df.answer_field == task + '_shuffled'], hue='n_examples', y='accuracy', ax=axs[2], palette=task_palette, legend=False, err_kws={'lw' : 1.2})
    axs[2].set_title("Shuffled labels\n(target accuracy)", fontsize=20)

    # 5) Demonstration with shuffled labels (original accuracy)
    sns.barplot(data=demos_df[demos_df.answer_field == task + '_shuffled'], hue='n_examples', y='accuracy_original', ax=axs[3], palette=task_palette, legend=False, err_kws={'lw' : 1.2})
    axs[3].set_title("Shuffled labels\n(original accuracy)", fontsize=20)

    for ax in axs:
        ax.axhline(no_demos_df.query('prefix_type == "instruction"').accuracy.mean(), color='#4a4a4a', ls='--', lw=1.5, label='Instruction')
        ax.axhline(no_demos_df.query('prefix_type == "raw"').accuracy.mean(), color='#ababab', ls='--', lw=1.5, label='Raw')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
        ax.set_ylabel('Accuracy', fontsize=20)
        bars_xlim = (ax.patches[0].xy[0] + ax.patches[0].get_width()/2, ax.patches[-1].xy[0] + ax.patches[-1].get_width()/2)
        ax.set_xticks(np.linspace(*bars_xlim, len(demos_df.n_examples.unique())))
    
        tick_labels = sorted(demos_df.n_examples.unique())
    
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis='both', rotation=0, labelsize=12)

    
    axs_grid[1,0].set_xlabel('# demonstrations', fontsize=20)
    axs_grid[1,1].set_xlabel('# demonstrations', fontsize=20)
    axs_grid[-1,1].legend(title='', fontsize=20, title_fontsize=20, loc='upper left', bbox_to_anchor=(0, -0.3))

    # sns.move_legend(axs_grid[-1,1], loc='upper left', bbox_to_anchor=(-0.5, -0.3), fontsize=20)
   # fig.subplots_adjust(right=0.25)
   # fig.tight_layout()
    return fig



def main_ICL_single(MODEL_NAME, DATASET_NAME):
    master_df = load_capacity_df(DATASET_NAME, MODEL_NAME)
    master_df = master_df.query('prefix_type != "instruction_detailed"') # Removing detailed instructions from the analysis

    performance_df = master_df.query(f'layer == 0 and capacity_task == "{master_df.capacity_task.iloc[0]}" and token_type == "{master_df.token_type.iloc[0]}"') # Selecting only a subset of capacity data for performance analysis

    # --- Plotting performance ---
    save_dir = Path("plots/ICL_performance/")
    (save_dir / 'svg').mkdir(exist_ok=True, parents=True)

    if DATASET_NAME == 'claude_multitask':
        tasks = ['emotion',  'intent', 'topic']
    else:
        tasks = ['category']

    for task in tasks:
        fig = plot_performance(performance_df, task, 'Category')
        if task == 'emotion':
            fig.suptitle(f'Sentiment analysis (multitask dataset)', fontsize=25)
        elif task == 'intent':
            fig.suptitle(f'Intent classification (multitask dataset)', fontsize=25)
        elif task == 'topic':
            fig.suptitle(f'Topic classification (multitask dataset)', fontsize=25)
        elif task == 'category':
            fig.suptitle(f'{DATASET_NAME}', fontsize=25)

        fig.tight_layout()
        fig.savefig(save_dir / f"svg/{DATASET_NAME}_{MODEL_NAME}_{task}_performance.svg")
        fig.savefig(save_dir / f"{DATASET_NAME}_{MODEL_NAME}_{task}_performance.png", bbox_inches='tight')


    # --- Plotting geometry ---
    save_dir = Path("/mnt/home/akirsanov/LLMGeometry/naacl_2025/plots/ICL_geometry/")
    (save_dir / 'svg').mkdir(exist_ok=True, parents=True)


    if DATASET_NAME == 'claude_multitask':
        tasks = ['emotion',  'intent', 'topic']
    else:
        tasks = ['category']

    for task in tasks:
        for token_type in ['mean_pooled', 'last_token']:
            fig = plot_geometry(master_df, task, 'Category', token_type)
            
            if task == 'emotion':
                fig.suptitle(f'Sentiment analysis (multitask dataset)', fontsize=25)
            elif task == 'intent':
                fig.suptitle(f'Intent classification (multitask dataset)', fontsize=25)
            elif task == 'topic':
                fig.suptitle(f'Topic classification (multitask dataset)', fontsize=25)
            elif task == 'category':
                fig.suptitle(f'{DATASET_NAME}', fontsize=25)#, y=1.02)
            
            fig.tight_layout()
            fig.savefig(save_dir / f"svg/{DATASET_NAME}_{MODEL_NAME}_{task}_{token_type}_geometry.svg")
            fig.savefig(save_dir / f"{DATASET_NAME}_{MODEL_NAME}_{task}_{token_type}_geometry.png")

    print(colored(f"Saved plots for {DATASET_NAME} and {MODEL_NAME}", "green"))



def plot_cross_tasks_geometry(df, y_vars, token_type, layer=None):
    cross_task_df = df.query(f"token_type == '{token_type}'")
    if not isinstance(y_vars, list):
        y_vars = [y_vars]

    fig, axs_grid = plt.subplots(len(y_vars),3, figsize=(14,3.5*len(y_vars)), dpi=300, sharey=False, sharex='col', squeeze=False)
    kwargs_demos = {
        'lw': 2,
        'errorbar': None,
        'marker' : 'o',
        'mew' : 0,
        'legend' : False,
        'ms' : 4
    }

    kwargs_instruction = {
        'lw': 2.5,
       'errorbar': None,
        'legend' : False,
        'ls' : '--'
    }

    demos_df = cross_task_df.query('prefix_type == "demos"')
    instruction_df = cross_task_df.query('prefix_type == "instruction"')


    for y_var, axs in zip(y_vars, axs_grid):
        sns.lineplot( data=demos_df.query('aligned == True'), x='layer', y=y_var, hue='n_examples',ax=axs[0],**kwargs_demos,
            palette=get_palette_from_cmap(cmasher.get_sub_cmap(plt.cm.Greens, 0.4, 1), demos_df.query('aligned == True').n_examples),
        ) # Demos (coherent)

        sns.lineplot(data=instruction_df.query('aligned == True'), x='layer', y=y_var, color='gray', ax=axs[0], **kwargs_instruction) # Instruction (coherent)

        sns.lineplot( data=demos_df.query('aligned == False'), x='layer', y=y_var, hue='n_examples',ax=axs[1],**kwargs_demos,
            palette=get_palette_from_cmap(cmasher.get_sub_cmap(plt.cm.Reds, 0.4, 1), demos_df.query('aligned == False').n_examples),
        ) # Demos (incoherent)

        sns.lineplot(data=instruction_df.query('aligned == False'), x='layer', y=y_var, color='gray', ax=axs[1], **kwargs_instruction) # Instruction (incoherent)

        # Plotting normalized capacity as a function of the number of examples
        if layer is None:
            if token_type == 'last_token':
                layer = demos_df.layer.max()
            else:
                layer = 12

        for ax in axs[:2]:
            ax.axvline(layer, color='black', linestyle='--', lw=1)
            
        sns.lineplot( data=demos_df.query('layer == @layer'), x='n_examples', y=y_var, hue='aligned', palette={True : 'green', False : 'red'}, ax=axs[2], lw=3, marker='o', legend=False, err_kws=dict(alpha=0.1),mew=0,ms=6)

        axs[2].set_xscale('log')
        axs[2].axhline(instruction_df.query('layer == @layer and aligned==False')[y_var].mean(), color='red', linestyle='dotted', lw=3)
        axs[2].axhline(instruction_df.query('layer == @layer and aligned==True')[y_var].mean(), color='green', linestyle='dotted', lw=3)

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel('')
            # y_name = y_var[:-5].replace('_', ' ').title()
            # if ax is axs[0]:
            #     ax.set_ylabel(f'{y_name} (/ raw)')
            # else:
            #     ax.set_ylabel('')

            # ax.set_xlabel('Layer')



        for ax in axs[:2]:
            ax.set_xlabel('Layer', fontsize=22)
           # ax.tick_params(axis='both', rotation=0, labelsize=14)

        axs[2].set_xlabel('# demonstrations', fontsize=22)
        axs[2].set_xticks([1,2,3,5,10,20,40])
        axs[2].set_xticklabels([1,2,4,5,10,20,40])
            #axs[2].set_xticks(demos_df.n_examples.unique().astype(int))


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


def main_ICL_multitask(MODEL_NAME):
    DATASET_NAME = 'claude_multitask'
    master_df = load_capacity_df(DATASET_NAME, MODEL_NAME).query('prefix_type != "instruction_detailed"')


    acc_dfs = []
    for task_icl in ['emotion', 'topic', 'intent']:
        for task_capacity in ['emotion', 'topic', 'intent']:
            keyword = 'Category' #TASK_keywords[task_icl]
            d = master_df.query(f"capacity_task == '{task_capacity}' and answer_field == '{task_icl}' and keyword == '{keyword}'").copy()
            d['aligned'] = (task_icl == task_capacity)
            acc_dfs.append(d)

    cross_task_df = pd.concat(acc_dfs)
    cross_task_df.n_examples = cross_task_df.n_examples.astype('category')


    for token_type in ['mean_pooled', 'last_token']:
        if MODEL_NAME == 'gemma2_2b_base' and token_type == 'mean_pooled':
            layer = 14
        else:
            layer = None # Automatically select the layer

        fig = plot_cross_tasks_geometry(cross_task_df, ['capacity_norm', 'participation_ratio_norm', 'max_dist_R_norm', 'axes_alignment_norm', 'center_axes_alignment_norm'], token_type, layer=layer)
        
        save_dir = Path("/mnt/home/akirsanov/LLMGeometry/naacl_2025/plots/multitask_geometry/")
        (save_dir / 'svg').mkdir(exist_ok=True, parents=True)

        fig.savefig(save_dir / f"svg/{MODEL_NAME}_{token_type}_multitask_geometry.svg")
        fig.savefig(save_dir / f"{MODEL_NAME}_{token_type}_multitask_geometry.png")





if __name__ == '__main__':
  #  main_ICL_single('llama3.1_base', 'claude_multitask')
  #  main_ICL_single('llama3.1_base', 'ag_news')
    # main_ICL_single('llama3.1_base', 'TREC_coarse')

    # main_ICL_single('gemma2_2b_base', 'claude_multitask')
    # main_ICL_single('gemma2_2b_base', 'ag_news')
    # main_ICL_single('gemma2_2b_base', 'TREC_coarse')

    main_ICL_multitask('llama3.1_base')
    main_ICL_multitask('gemma2_2b_base')