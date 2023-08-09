#!/bin/bash

# Delete existing datasets
rm data/*.h5

tasks=("cleanHouse" "harvester" "fourCorners" "randomMaze" "stairClimber" "topOff")

for task in "${tasks[@]}"; do

    # Generate dataset
    python3 generate_random_dataset.py -g karel -n 7000 -b 65536 --seed 0 --karel_task "${task}" --overwrite

    # Train model
    python3 experiment.py --n_layer 2 --K 5 --embed_dim 16 --batch_size 32 --warmup_steps 10 --num_eval_episodes 3 \
        --num_steps_per_iter 10 --model_type dt --max_iters 10 --env karel --mode delayed --karel_task "${task}" \
        --log_to_wandb --use_seq_state_embedding
done