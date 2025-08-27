#!/bin/bash
#SBATCH --job-name=icl_relabel
#SBATCH --array=0-1009  # Total configs: 1010 (0-indexed)
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short

# Go to project directory first
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

# Create logs directory if it doesn't exist
mkdir -p logs

# Extract n_relabel and n_examples from the config JSON for this specific job
CONFIG_FILE="experiments/icl_configs.json"
JOB_IDX=${SLURM_ARRAY_TASK_ID}

# Use Python to extract the config values
CONFIG_VALUES=$(python -c "
import json
import sys
with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)
if $JOB_IDX >= len(configs):
    print('ERROR: Job index out of range')
    sys.exit(1)
config = configs[$JOB_IDX]
n_relabel = config.get('n_relabel', 'unknown')
n_examples = config.get('n_examples', 'unknown')
print(f'{n_relabel}_{n_examples}')
")

# Check if we got valid config values
if [[ "$CONFIG_VALUES" == "ERROR:"* ]]; then
    echo "$CONFIG_VALUES"
    exit 1
fi

# Set output and error log files with the extracted values
export SBATCH_OUTPUT="logs/icl_relabel_${CONFIG_VALUES}_%A_%a.out"
export SBATCH_ERROR="logs/icl_relabel_${CONFIG_VALUES}_%A_%a.err"

# Redirect stdout and stderr to our custom log files
exec 1> "logs/icl_relabel_${CONFIG_VALUES}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
exec 2> "logs/icl_relabel_${CONFIG_VALUES}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

echo "Starting job with config: n_relabel=${CONFIG_VALUES%_*}, n_examples=${CONFIG_VALUES#*_}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Config file: $CONFIG_FILE"
echo "Job index: $JOB_IDX"

# Activate conda environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

# Run the experiment - note that jobind is a positional argument, not a flag
python experiments/run_ICL_relabel.py ${SLURM_ARRAY_TASK_ID} experiments/icl_configs.json 