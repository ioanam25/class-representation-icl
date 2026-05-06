#!/usr/bin/env python3
"""
Setup shuffled-label experiment for 3-class sentiment (Qwen).

Takes existing optimized relabelings and permutes the class→token mapping
to break input-output correspondence. This tests whether the specific
optimized token assignments matter.

Creates:
  1. Shuffled relabeling pickles in qwen2_7b_base_relabelings_shuffled/
  2. ICL config JSON: experiments/configs/icl_configs_3classes_shuffled_qwen.json
  3. SLURM script:    experiments/run_scripts/run_icl_slurm_qwen_shuffled_3c.sh
"""

import json
import os
import pickle
from pathlib import Path
from itertools import permutations

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Experiment grid ───────────────────────────────────────────────────────────
MODEL = "qwen2_7b_base"
DATASET = "claude_multitask"
NUM_CLASSES = 3
N_DEMOS_GRID = list(range(10, 101, 10))      # 10, 20, …, 100
N_RELABEL_GRID = list(range(10, 101, 10))     # 10, 20, …, 100
N_RUNS = 10

SRC_DIR = PROJECT_ROOT / "relabelings/qwen2_7b_base_relabelings"
DST_DIR = PROJECT_ROOT / "relabelings/qwen2_7b_base_relabelings_shuffled"


def make_shuffled_pickle(src_path, dst_path):
    """Load an optimized relabeling pickle, apply a derangement, and save."""
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    orig_labels = data["relabelings"][0]["labels"]   # {key: (tok_str, tok_id)}
    keys = sorted(orig_labels.keys())                # e.g. ['A', 'C', 'D']

    # Build a cyclic derangement: shift values by 1 position
    # A gets C's token, C gets D's token, D gets A's token
    values = [orig_labels[k] for k in keys]
    shifted_values = values[1:] + values[:1]         # rotate left by 1

    shuffled_labels = dict(zip(keys, shifted_values))

    print(f"  Original:  { {k: v[0] for k, v in orig_labels.items()} }")
    print(f"  Shuffled:  { {k: v[0] for k, v in shuffled_labels.items()} }")

    # Build new pickle in the same format
    new_data = {
        "config": data["config"],
        "relabelings": [{
            "labels": shuffled_labels,
            "objective": data["relabelings"][0].get("objective", 0.0),
        }],
    }
    # Preserve any extra keys
    for extra in ("examples", "run_seed"):
        if extra in data["relabelings"][0]:
            new_data["relabelings"][0][extra] = data["relabelings"][0][extra]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(new_data, f)
    return shuffled_labels


def main():
    DST_DIR.mkdir(exist_ok=True)

    # ── Step 1: Create shuffled pickles ───────────────────────────────────────
    print("=" * 60)
    print("  Step 1: Creating shuffled relabeling pickles")
    print("=" * 60)

    configs = []

    for n_relabel in N_RELABEL_GRID:
        src_name = (
            f"qwen2_7b_base_relabelings_{NUM_CLASSES}classes_128256toptokens_"
            f"isensembledFalse_voting_{n_relabel}examples_1runs.pkl"
        )
        src_path = SRC_DIR / src_name
        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping n_relabel={n_relabel}")
            continue

        dst_name = (
            f"qwen2_7b_base_relabelings_{NUM_CLASSES}classes_"
            f"{n_relabel}examples_shuffled.pkl"
        )
        dst_path = DST_DIR / dst_name

        print(f"\nn_relabel={n_relabel}:")
        make_shuffled_pickle(str(src_path), str(dst_path))

        # ── Step 2: Generate configs for this n_relabel ───────────────────────
        for n_demos in N_DEMOS_GRID:
            configs.append({
                "MODEL_NAME": MODEL,
                "DATASET_NAME": DATASET,
                "num_classes": NUM_CLASSES,
                "prefix_type": "demos",
                "n_examples": n_demos,
                "n_relabel": n_relabel,
                "keyword": "Category",
                "answer_field": "emotion_letter",
                "N_RUNS": N_RUNS,
                "root_folder": "learning_curves/learning_curves_shuffled_3classes_qwen",
                "ensemble_assignment": False,
                "ensemble_method": "voting",
                "ensemble_temperature": 1.0,
                "top_tokens": 128256,
                "whole_words_only": True,
                "base_seed": 42,
                "fixed_relabeling_path": str(dst_path),
            })

    # ── Step 2b: Write config JSON ────────────────────────────────────────────
    config_file = "experiments/configs/icl_configs_3classes_shuffled_qwen.json"
    config_path = PROJECT_ROOT / config_file
    with open(config_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"  Step 2: Created {config_file} with {len(configs)} configs")
    print(f"{'=' * 60}")

    # ── Step 3: Generate SLURM script ─────────────────────────────────────────
    max_idx = len(configs) - 1

    slurm_script = f"""#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --exclude=sp-0001,sp-0009,sp-0010,sp-0013,sp-0003,sp-0006,sp-0007,sp-0008,sp-0016
#SBATCH --qos=qos_superpod
#SBATCH --job-name=shuf_qw_3c
#SBATCH --array=0-{max_idx}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80GB


# Go to project directory first
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

# Create logs directory if it doesn't exist
mkdir -p logs/logs_qwen_shuffled_3c

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
print(f'r{{c[\"n_relabel\"]}}_d{{c[\"n_examples\"]}}')")

if [[ "$CONFIG_VALUES" == "ERROR:"* ]]; then echo "$CONFIG_VALUES"; exit 1; fi

exec 1> "logs/logs_qwen_shuffled_3c/icl_shuffled_3c_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "logs/logs_qwen_shuffled_3c/icl_shuffled_3c_${{CONFIG_VALUES}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

echo "Starting job — config=${{CONFIG_VALUES}}"
echo "Job ID: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Config file: $CONFIG_FILE"
echo "*** Shuffled-label experiment: 3-class sentiment ***"

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

    slurm_path = PROJECT_ROOT / "experiments/run_scripts/run_icl_slurm_qwen_shuffled_3c.sh"
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    print(f"  Step 3: Created {slurm_path.name}  (array 0-{max_idx}, {len(configs)} configs)")
    print(f"\nTo submit:  sbatch experiments/run_scripts/run_icl_slurm_qwen_shuffled_3c.sh")


if __name__ == "__main__":
    main()
