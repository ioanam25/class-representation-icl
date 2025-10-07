#!/bin/bash
#SBATCH --partition=oermannlab
#SBATCH --job-name=bootstrap_correlations
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=logs_bootstrap/bootstrap_correlations_%j.out
#SBATCH --error=logs_bootstrap/bootstrap_correlations_%j.err

# Go to project directory first
cd /gpfs/data/oermannlab/users/im2178/class-representation-icl

# Create logs directory if it doesn't exist
mkdir -p logs_bootstrap

# Create bootstrap_results directory if it doesn't exist
mkdir -p bootstrap_results

echo "Starting bootstrap correlation analysis"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Print system info
echo "CPU info:"
lscpu | head -20
echo "Memory info:"
free -h

# Activate conda environment if needed
# Uncomment and modify these lines if you need a specific conda environment
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

# Print Python environment info
echo "Python version:"
python --version
echo "Python path:"
which python

# Run the bootstrap analysis
echo "Starting bootstrap analysis..."
python experiments/stats_csv.py

echo "Bootstrap analysis completed at $(date)"
echo "Results saved to bootstrap_results/ directory"
