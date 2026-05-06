#!/usr/bin/env python3
"""
Generate configs and SLURM script for 0-shot gold label experiments
(sentiment 3-class, sentiment 5-class, TREC 5-class).
"""
import json
import os

RELABEL_DIR = "relabelings/qwen2_7b_base_relabelings_synonyms"

configs = [
    # 3-class sentiment, 0-shot
    {
        "MODEL_NAME": "qwen2_7b_base",
        "DATASET_NAME": "claude_multitask",
        "num_classes": 3,
        "prefix_type": "demos",
        "n_examples": 0,
        "n_relabel": 0,
        "keyword": "Category",
        "answer_field": "emotion_letter",
        "N_RUNS": 10,
        "root_folder": "learning_curves/learning_curves_synonym_gold_3classes_qwen",
        "ensemble_assignment": False,
        "ensemble_method": "voting",
        "ensemble_temperature": 1.0,
        "top_tokens": 128256,
        "whole_words_only": True,
        "base_seed": 42,
        "fixed_relabeling_path": os.path.join(RELABEL_DIR, "qwen2_7b_base_relabelings_3classes_synonym_gold.pkl"),
    },
    # 5-class sentiment, 0-shot
    {
        "MODEL_NAME": "qwen2_7b_base",
        "DATASET_NAME": "claude_multitask",
        "num_classes": 5,
        "prefix_type": "demos",
        "n_examples": 0,
        "n_relabel": 0,
        "keyword": "Category",
        "answer_field": "emotion_letter",
        "N_RUNS": 10,
        "root_folder": "learning_curves/learning_curves_synonym_gold_5classes_qwen",
        "ensemble_assignment": False,
        "ensemble_method": "voting",
        "ensemble_temperature": 1.0,
        "top_tokens": 128256,
        "whole_words_only": True,
        "base_seed": 42,
        "fixed_relabeling_path": os.path.join(RELABEL_DIR, "qwen2_7b_base_relabelings_5classes_synonym_gold.pkl"),
    },
    # TREC 5-class, 0-shot
    {
        "MODEL_NAME": "qwen2_7b_base",
        "DATASET_NAME": "TREC_coarse",
        "num_classes": 5,
        "prefix_type": "demos",
        "n_examples": 0,
        "n_relabel": 0,
        "keyword": "Category",
        "answer_field": "label",
        "N_RUNS": 10,
        "root_folder": "learning_curves/learning_curves_synonym_gold_TREC_5classes_qwen",
        "ensemble_assignment": False,
        "ensemble_method": "voting",
        "ensemble_temperature": 1.0,
        "top_tokens": 128256,
        "whole_words_only": True,
        "base_seed": 42,
        "fixed_relabeling_path": os.path.join(RELABEL_DIR, "qwen2_7b_base_relabelings_5classes_TREC_synonym_gold.pkl"),
    },
]

config_path = "experiments/configs/icl_configs_gold_label_0shot.json"
with open(config_path, 'w') as f:
    json.dump(configs, f, indent=2)
print(f"✓ Config: {config_path} ({len(configs)} configs)")

slurm = f"""#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --exclude=sp-0001,sp-0009,sp-0010,sp-0013,sp-0003,sp-0006,sp-0007,sp-0008,sp-0016
#SBATCH --qos=qos_superpod
#SBATCH --job-name=gold_0shot
#SBATCH --array=0-{len(configs)-1}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80GB

cd /gpfs/data/oermannlab/users/im2178/class-representation-icl
mkdir -p logs/logs_gold_0shot

CONFIG_FILE="{config_path}"
JOB_IDX=${{SLURM_ARRAY_TASK_ID}}

exec 1> "logs/logs_gold_0shot/gold_0shot_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "logs/logs_gold_0shot/gold_0shot_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

echo "Starting gold label 0-shot — task $JOB_IDX"

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

slurm_path = "experiments/run_scripts/run_icl_slurm_gold_label_0shot.sh"
with open(slurm_path, 'w') as f:
    f.write(slurm)
print(f"✓ SLURM: {slurm_path} (array 0-{len(configs)-1})")
print(f"\nTo run:\n  sbatch {slurm_path}")
