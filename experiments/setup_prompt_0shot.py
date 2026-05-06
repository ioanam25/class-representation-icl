#!/usr/bin/env python3
"""
Generate 0-shot configs and SLURM scripts for prompt-format experiments (Qwen2-7B, 3-class).

Outputs:
  - experiments/configs/icl_configs_3classes_qwen_prompt_arrow_0shot.json
  - experiments/configs/icl_configs_3classes_qwen_prompt_sentence_label_0shot.json
  - experiments/slurm_prompt_arrow_0shot.sh
  - experiments/slurm_prompt_sentence_label_0shot.sh

Usage:
  python experiments/setup_prompt_0shot.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"

MODEL_NAME = "qwen2_7b_base"
DATASET_NAME = "claude_multitask"
NUM_CLASSES = 3
KEYWORD = "Category"
ANSWER_FIELD = "emotion_letter"
N_RUNS = 10
TOP_TOKENS = 128256
WHOLE_WORDS_ONLY = True
BASE_SEED = 42
N_RELABEL_VALUES = list(range(10, 101, 10))


def make_entry(prompt_format: str | None, root_folder: str, n_relabel: int) -> dict:
    entry = {
        "MODEL_NAME": MODEL_NAME,
        "DATASET_NAME": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "prefix_type": "demos",
        "keyword": KEYWORD,
        "answer_field": ANSWER_FIELD,
        "N_RUNS": N_RUNS,
        "ensemble_assignment": False,
        "ensemble_method": "logit_averaging",
        "ensemble_temperature": 0,
        "top_tokens": TOP_TOKENS,
        "whole_words_only": WHOLE_WORDS_ONLY,
        "base_seed": BASE_SEED,
        "root_folder": root_folder,
        "n_relabel": n_relabel,
        "n_examples": 0,
    }
    if prompt_format is not None:
        entry["prompt_format"] = prompt_format
    return entry


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def write_slurm(path: Path, config_json: Path, job_name: str) -> None:
    script = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o {job_name}.%A_%a.out
#SBATCH -e {job_name}.%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G

set -euo pipefail
echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

CONFIG="{config_json}"
INDEX="${{SLURM_ARRAY_TASK_ID:-0}}"

python experiments/run_ICL_relabel.py --config "$CONFIG" --index "$INDEX"
"""
    path.write_text(script)
    print(f"Wrote {path}")


def main():
    # Arrow
    arrow_entries = [
        make_entry("arrow", "learning_curves/learning_curves_prompt_arrow_3classes_qwen", nr) for nr in N_RELABEL_VALUES
    ]
    arrow_json = EXP_DIR / "icl_configs_3classes_qwen_prompt_arrow_0shot.json"
    write_json(arrow_json, arrow_entries)
    write_slurm(EXP_DIR / "slurm_prompt_arrow_0shot.sh", arrow_json, "pf_arrow_0shot")

    # Sentence/Label
    sent_entries = [
        make_entry("sentence_label", "learning_curves/learning_curves_prompt_sentence_label_3classes_qwen", nr)
        for nr in N_RELABEL_VALUES
    ]
    sent_json = EXP_DIR / "icl_configs_3classes_qwen_prompt_sentence_label_0shot.json"
    write_json(sent_json, sent_entries)
    write_slurm(EXP_DIR / "slurm_prompt_sentence_label_0shot.sh", sent_json, "pf_sentlabel_0shot")

    print("\nNext steps:")
    print(f"- Submit Arrow:    sbatch --array=0-{len(arrow_entries)-1} {EXP_DIR/'slurm_prompt_arrow_0shot.sh'}")
    print(f"- Submit Sent/Lbl: sbatch --array=0-{len(sent_entries)-1} {EXP_DIR/'slurm_prompt_sentence_label_0shot.sh'}")


if __name__ == "__main__":
    main()

