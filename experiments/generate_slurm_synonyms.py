#!/usr/bin/env python3
"""
Generate SLURM submission scripts for all synonym experiments (Qwen).
"""

import os

TEMPLATE = """#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --exclude=sp-0001,sp-0009,sp-0010,sp-0013,sp-0003,sp-0006,sp-0007,sp-0008,sp-0016
#SBATCH --qos=qos_superpod
#SBATCH --job-name={job_name}
#SBATCH --array=0-{max_idx}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80GB


# Go to project directory first
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

# Create logs directory if it doesn't exist
mkdir -p {logs_dir}

# Extract n_relabel and n_examples from the config JSON for this specific job
CONFIG_FILE="{config_file}"
JOB_IDX=${{SLURM_ARRAY_TASK_ID}}

# Use Python to extract the config values
CONFIG_VALUES=$(python -c "
import json, sys
with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)
if $JOB_IDX >= len(configs):
    print('ERROR: Job index out of range'); sys.exit(1)
c = configs[$JOB_IDX]
print(f'{{c[\"n_examples\"]}}')")

if [[ "$CONFIG_VALUES" == "ERROR:"* ]]; then echo "$CONFIG_VALUES"; exit 1; fi

exec 1> "{logs_dir}/{log_prefix}_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "{logs_dir}/{log_prefix}_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

echo "Starting job — n_examples=${{CONFIG_VALUES}}"
echo "Job ID: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Config file: $CONFIG_FILE"
echo "*** Synonym set: {syn_set} | {num_classes}-class | Qwen ***"

source ~/.bashrc
conda activate arcenv
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

export PYTHONPATH="${{PYTHONPATH}}:/gpfs/data/oermannlab/users/im2178/class-representation-icl/src"
export HF_HOME="/gpfs/data/oermannlab/users/im2178/class-representation-icl"
export TRANSFORMERS_CACHE="${{HF_HOME}}/models"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets"

if [ ! -f "$SSL_CERT_FILE" ]; then unset SSL_CERT_FILE; fi

python experiments/run_ICL_relabel.py ${{SLURM_ARRAY_TASK_ID}} {config_file}
"""

SYNONYM_SETS_3CLASS = ["gold", "syn1", "syn2", "syn3", "syn4"]
SYNONYM_SETS_5CLASS = ["gold", "syn1", "syn2", "syn3"]
N_CONFIGS = 10  # 10 n_examples values → indices 0-9


def main():
    os.makedirs("experiments", exist_ok=True)

    for num_classes, syn_sets in [(3, SYNONYM_SETS_3CLASS), (5, SYNONYM_SETS_5CLASS)]:
        for syn_set in syn_sets:
            job_name = f"icl_qw_{num_classes}c_{syn_set}"
            logs/logs_dir = f"logs/logs_qwen_{num_classes}class_synonym_{syn_set}"
            log_prefix = f"icl_qwen_{num_classes}c_{syn_set}"
            config_file = f"experiments/configs/icl_configs_{num_classes}classes_qwen_synonym_{syn_set}.json"
            max_idx = N_CONFIGS - 1

            script = TEMPLATE.format(
                job_name=job_name,
                max_idx=max_idx,
                logs/logs_dir=logs_dir,
                config_file=config_file,
                log_prefix=log_prefix,
                syn_set=syn_set,
                num_classes=num_classes,
            )

            out_path = f"experiments/run_scripts/run_icl_slurm_qwen_{num_classes}class_synonym_{syn_set}.sh"
            with open(out_path, "w") as f:
                f.write(script)
            print(f"  ✓ {out_path}  (array 0-{max_idx})")


if __name__ == "__main__":
    main()
