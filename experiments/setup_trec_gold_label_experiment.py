#!/usr/bin/env python3
"""
Generate everything needed for the TREC gold label experiment:
1. Gold label relabeling pickle (with integer keys for TREC compatibility)
2. ICL config JSON
3. SLURM script

Usage:
    cd /gpfs/data/oermannlab/users/im2178/class-representation-icl
    python experiments/setup_trec_gold_label_experiment.py
"""

import os
import json
import pickle
import transformers

# ── Qwen tokenizer ──────────────────────────────────────────────────────────
def load_qwen_tokenizer():
    safe_name = 'Qwen/Qwen2.5-7B'.replace('/', '--')
    refs_main = os.path.join('tokenizers', f'models--{safe_name}', 'refs', 'main')
    with open(refs_main) as f:
        commit_hash = f.read().strip()
    snap = os.path.join('tokenizers', f'models--{safe_name}', 'snapshots', commit_hash)
    return transformers.AutoTokenizer.from_pretrained(snap, local_files_only=True)


# TREC gold labels: class index -> word
# run_ICL_relabel.py for TREC+Qwen does: new_labels[int(x[1:])]
# where x is like " 0", " 1", etc. So keys must be integers 0-4
TREC_GOLD = {0: 'entity', 1: 'description', 2: 'human', 3: 'location', 4: 'numeric'}


def main():
    print("Loading Qwen tokenizer...")
    tokenizer = load_qwen_tokenizer()

    # ── 1. Create gold label pickle ──────────────────────────────────────────
    output_dir = "qwen2_7b_base_relabelings_synonyms"
    os.makedirs(output_dir, exist_ok=True)

    new_labels = {}
    for cls_idx, word in TREC_GOLD.items():
        token_str = 'Ġ' + word
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        encoded = tokenizer.encode(' ' + word, add_special_tokens=False)
        
        if len(encoded) != 1 or token_id is None:
            print(f"  WARNING: '{word}' is NOT a single token! encoded={encoded}, trying alternatives...")
            # Try without leading space
            alt_encoded = tokenizer.encode(word, add_special_tokens=False)
            print(f"    Without space: {alt_encoded} = {[tokenizer.decode([t]) for t in alt_encoded]}")
            raise ValueError(f"'{word}' is multi-token. Pick a single-token alternative.")
        
        new_labels[cls_idx] = (token_str, token_id)
        print(f"  {cls_idx} -> '{word}' -> token '{token_str}' (id={token_id})")

    data = {
        'config': {
            'MODEL_NAME': 'qwen2_7b_base',
            'DATASET_NAME': 'TREC_coarse',
            'num_classes': 5,
            'top_tokens': 128256,
            'whole_words_only': True,
            'ensemble_assignment': False,
            'ensemble_method': 'voting',
            'synonym_set': 'gold',
        },
        'relabelings': [{
            'labels': new_labels,
            'objective': 0.0,
        }],
    }

    pkl_name = "qwen2_7b_base_relabelings_5classes_TREC_synonym_gold.pkl"
    pkl_path = os.path.join(output_dir, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n✓ Pickle: {pkl_path}")

    # ── 2. Create ICL config JSON ────────────────────────────────────────────
    configs = []
    for n_demos in range(10, 101, 10):
        configs.append({
            "MODEL_NAME": "qwen2_7b_base",
            "DATASET_NAME": "TREC_coarse",
            "num_classes": 5,
            "prefix_type": "demos",
            "n_examples": n_demos,
            "n_relabel": 0,
            "keyword": "Category",
            "answer_field": "label",
            "N_RUNS": 10,
            "root_folder": "learning_curves_synonym_gold_TREC_5classes_qwen",
            "ensemble_assignment": False,
            "ensemble_method": "voting",
            "ensemble_temperature": 1.0,
            "top_tokens": 128256,
            "whole_words_only": True,
            "base_seed": 42,
            "fixed_relabeling_path": pkl_path,
        })

    config_path = "experiments/configs/icl_configs_5classes_TREC_qwen_synonym_gold.json"
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"✓ Config: {config_path} ({len(configs)} configs)")

    # ── 3. Create SLURM script ───────────────────────────────────────────────
    slurm = """#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --exclude=sp-0001,sp-0009,sp-0010,sp-0013,sp-0003,sp-0006,sp-0007,sp-0008,sp-0016
#SBATCH --qos=qos_superpod
#SBATCH --job-name=icl_trec_gold
#SBATCH --array=0-{max_idx}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80GB

cd /gpfs/data/oermannlab/users/im2178/class-representation-icl
mkdir -p logs/logs_qwen_trec_gold_label

CONFIG_FILE="{config_file}"
JOB_IDX=${{SLURM_ARRAY_TASK_ID}}

CONFIG_VALUES=$(python -c "
import json
with open('$CONFIG_FILE') as f:
    configs = json.load(f)
c = configs[$JOB_IDX]
print(c['n_examples'])")

exec 1> "logs/logs_qwen_trec_gold_label/icl_trec_gold_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "logs/logs_qwen_trec_gold_label/icl_trec_gold_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

echo "Starting TREC gold label — n_examples=${{CONFIG_VALUES}}"
echo "Job ID: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"

source ~/.bashrc
conda activate arcenv
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

export PYTHONPATH="${{PYTHONPATH}}:/gpfs/data/oermannlab/users/im2178/class-representation-icl/src"
export HF_HOME="/gpfs/data/oermannlab/users/im2178/class-representation-icl"
export TRANSFORMERS_CACHE="${{HF_HOME}}/models"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets"

if [ ! -f "$SSL_CERT_FILE" ]; then unset SSL_CERT_FILE; fi

python experiments/run_ICL_relabel.py ${{SLURM_ARRAY_TASK_ID}} {config_file}
""".format(max_idx=len(configs) - 1, config_file=config_path)

    slurm_path = "experiments/run_scripts/run_icl_slurm_qwen_trec_gold_label.sh"
    with open(slurm_path, 'w') as f:
        f.write(slurm)
    print(f"✓ SLURM: {slurm_path} (array 0-{len(configs)-1})")

    print(f"\nTo run:\n  sbatch {slurm_path}")


if __name__ == '__main__':
    main()
