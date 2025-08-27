#!/bin/bash

# Run for n_relabel from 10 to 100 in steps of 10
for n in $(seq 10 10 100); do
    echo "Running with n_relabel = $n"
    python generate_relabelings.py --n_relabel $n
    echo "Completed n_relabel = $n"
    echo "-------------------"
done

echo "All runs completed!" 