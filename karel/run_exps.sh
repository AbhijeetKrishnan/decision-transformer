#!/bin/bash

tasks=("cleanHouse" "harvester" "fourCorners" "randomMaze" "stairClimber" "topOff")

for task in "${tasks[@]}"; do

    # Generate dataset
    python3 generate_random_dataset.py -g karel -n 10000 -b 65536 --seed 0 --karel_task "${task}" --overwrite

    # Train model
    python3 experiment.py --n_layer 3 --n_head 1 --embed_dim 128 --activation_function "relu" --batch_size 64 --K 20 \
        --env_targets "2,1" --dropout 0.1 -lr 1e-4 -wd 1e-4 \
        --warmup_steps 10000 --num_eval_episodes 10 \
        --num_steps_per_iter 10000 --model_type dt --max_iters 10 --mode delayed --use_seq_state_embedding \
        --env karel --karel_task "${task}" --scale 1.0 \
        --log_to_wandb
done