# On the Relationship Between the Choice of Representation and In-Context Learning

This repository contains the code and data for the paper "On the Relationship Between the Choice of Representation and In-Context Learning".

![Cover Image](fig1v2.pdf)

## Description

This project investigates how different sets of labels or "class representations" impact the performance of models during in-context learning (ICL) with large language models. 

The workflow is designed to:
- Pre-calculate token probabilities for a given dataset
- Generate multiple sets of labels (relabelings) for the classes  
- Run ICL experiments using these different label sets to evaluate performance

## Workflow and Usage

The main workflow consists of three steps. Ensure you run them in the specified order:

### 1. Precompute Weights

First, use `precompute_weights.py` to process your dataset and calculate the next token probabilities for each sentence. This step is necessary to prepare the data for the subsequent experiments.

```bash
python precompute_weights.py
```

### 2. Generate Relabelings

Next, run `generate_relabelings.py` to create different sets of class labels based on the original dataset. These new label sets will be used to test the model's ICL capabilities.

```bash
python experiments/generate_relabelings.py
```

### 3. Run In-Context Learning Experiments

Finally, use `run_icl_relabel.py` to execute the in-context learning experiments using the precomputed weights and the newly generated label sets. This script will evaluate and output the performance of the model under different labeling schemes.

```bash
python experiments/run_icl_relabel.py
```
