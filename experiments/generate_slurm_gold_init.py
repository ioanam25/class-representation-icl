#!/usr/bin/env python3
"""
Generate SLURM submission scripts for gold-init relabeling experiments (Qwen).
"""

import os
import json

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

# Use Python to extract the config values for log naming
CONFIG_VALUES=$(python -c "
import json, sys
with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)
if $JOB_IDX >= len(configs):
    print('ERROR: Job index out of range'); sys.exit(1)
c = configs[$JOB_IDX]
# Include the pickle path basename to identify which n_relabel
import os
pkl = os.path.basename(c.get('fixed_relabeling_path', 'none'))
print(f'{{c[\"n_examples\"]}}_{{pkl}}')")

if [[ "$CONFIG_VALUES" == "ERROR:"* ]]; then echo "$CONFIG_VALUES"; exit 1; fi

exec 1> "{logs_dir}/{log_prefix}_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "{logs_dir}/{log_prefix}_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

echo "Starting job — config=${{CONFIG_VALUES}}"
echo "Job ID: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Config file: $CONFIG_FILE"
echo "*** Gold-init experiment: {tag} ***"

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

EXPERIMENTS = [
    {"nc": 3, "ds": "claude_multitask", "tag": "3c_sentiment"},
    {"nc": 5, "ds": "claude_multitask", "tag": "5c_sentiment"},
    {"nc": 5, "ds": "TREC_coarse",      "tag": "5c_TREC"},
]


def main():
    os.makedirs("experiments", exist_ok=True)

    for exp in EXPERIMENTS:
        nc = exp['nc']
        ds = exp['ds']
        tag = exp['tag']

        config_file = f"experiments/configs/icl_configs_{nc}classes_{ds}_gold_init_qwen.json"

        # Read config to get array size
        config_path = config_file
        if os.path.exists(config_path):
            with open(config_path) as f:
                n_configs = len(json.load(f))
        else:
            print(f"  WARNING: {config_path} not found, assuming 110 configs")
            n_configs = 110

        job_name = f"gi_qw_{tag}"
        logs/logs_dir = f"logs/logs_qwen_gold_init_{tag}"
        log_prefix = f"icl_gold_init_{tag}"
        max_idx = n_configs - 1

        script = TEMPLATE.format(
            job_name=job_name,
            max_idx=max_idx,
            logs/logs_dir=logs_dir,
            config_file=config_file,
            log_prefix=log_prefix,
            tag=tag,
        )

        out_path = f"experiments/run_scripts/run_icl_slurm_qwen_gold_init_{tag}.sh"
        with open(out_path, "w") as f:
            f.write(script)
        print(f"  ✓ {out_path}  (array 0-{max_idx}, {n_configs} configs)")


if __name__ == "__main__":
    main()
