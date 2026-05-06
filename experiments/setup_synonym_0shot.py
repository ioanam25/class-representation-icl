#!/usr/bin/env python3
"""
Generate configs and a SLURM script for 0-shot runs for synonym label sets.
This complements the existing synonym learning curves (which start at K=10)
by adding K=0 entries under the same root folders.
"""

import json
import os

MODEL = "qwen2_7b_base"
DATASET = "claude_multitask"
RELABEL_DIR = "relabelings/qwen2_7b_base_relabelings_synonyms"

# Only synonym sets (exclude 'gold' since gold 0-shot is already done separately)
SYN_SETS_3C = ["syn1", "syn2", "syn3", "syn4"]
SYN_SETS_5C = ["syn1", "syn2", "syn3"]

N_RUNS = 10


def build_entry(num_classes: int, set_name: str) -> dict:
    pkl_name = f"qwen2_7b_base_relabelings_{num_classes}classes_synonym_{set_name}.pkl"
    pkl_path = os.path.join(RELABEL_DIR, pkl_name)
    return {
        "MODEL_NAME": MODEL,
        "DATASET_NAME": DATASET,
        "num_classes": num_classes,
        "prefix_type": "demos",
        "n_examples": 0,
        "n_relabel": 0,
        "keyword": "Category",
        "answer_field": "emotion_letter",
        "N_RUNS": N_RUNS,
        "root_folder": f"learning_curves/learning_curves_synonym_{set_name}_{num_classes}classes_qwen",
        "ensemble_assignment": False,
        "ensemble_method": "voting",
        "ensemble_temperature": 1.0,
        "top_tokens": 128256,
        "whole_words_only": True,
        "base_seed": 42,
        "fixed_relabeling_path": pkl_path,
    }


def main():
    configs = []
    # 3-class
    for s in SYN_SETS_3C:
        configs.append(build_entry(3, s))
    # 5-class
    for s in SYN_SETS_5C:
        configs.append(build_entry(5, s))

    config_path = "experiments/configs/icl_configs_synonym_0shot.json"
    with open(config_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"✓ Config: {config_path} ({len(configs)} configs)")

    slurm = f"""#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --exclude=sp-0001,sp-0009,sp-0010,sp-0013,sp-0003,sp-0006,sp-0007,sp-0008,sp-0016
#SBATCH --qos=qos_superpod
#SBATCH --job-name=syn_0shot
#SBATCH --array=0-{len(configs)-1}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80GB

cd /gpfs/data/oermannlab/users/im2178/class-representation-icl
mkdir -p logs/logs_synonym_0shot

CONFIG_FILE="{config_path}"
JOB_IDX=${{SLURM_ARRAY_TASK_ID}}

exec 1> "logs/logs_synonym_0shot/synonym_0shot_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "logs/logs_synonym_0shot/synonym_0shot_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

echo "Starting synonym 0-shot — task $JOB_IDX"

source ~/.bashrc
conda activate arcenv
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

export PYTHONPATH="${{PYTHONPATH}}:/gpfs/data/oermannlab/users/im2178/class-representation-icl/src"
export HF_HOME="/gpfs/data/oermannlab/users/im2178/class-representation-icl"
export TRANSFORMERS_CACHE="${{HF_HOME}}/models"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets"

if [ ! -f "$SSL_CERT_FILE" ]; then unset SSL_CERT_FILE; fi

python experiments/run_ICL_relabel.py ${{SLURM_ARRAY_TASK_ID}} {config_path}
"""
    slurm_path = "experiments/run_scripts/run_icl_slurm_synonym_0shot.sh"
    with open(slurm_path, "w") as f:
        f.write(slurm)
    print(f"✓ SLURM: {slurm_path} (array 0-{len(configs)-1})")
    print(f"\nTo run:\n  sbatch {slurm_path}")


if __name__ == "__main__":
    main()

