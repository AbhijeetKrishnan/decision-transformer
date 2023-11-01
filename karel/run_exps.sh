#!/bin/bash

NUM_EPISODES=2
SEED=68997 # generated using https://www.random.org/
STEPS=100
DATASET="leaps"

tasks=("cleanHouse" "harvester" "fourCorners" "randomMaze" "stairClimber" "topOff")

for task in "${tasks[@]}"; do

    if [ "${DATASET}" == "random" ]; then
        # Generate dataset (random)
        python3 generate_dataset.py -g karel -n "${NUM_EPISODES}" -b 65536 --seed "${SEED}" --karel_task "${task}" --overwrite

        # Train model
        python3 experiment.py --n_layer 3 --n_head 1 --embed_dim 128 --activation_function "relu" --batch_size 64 --K 20 \
            --env_targets "1" --dropout 0.1 -lr 1e-4 -wd 1e-4 \
            --warmup_steps "${STEPS}" --num_eval_episodes 64 \
            --num_steps_per_iter "${STEPS}" --model_type dt --max_iters 10 --mode delayed --use_seq_state_embedding \
            --env karel --dataset random --karel_task "${task}" --scale 1.0 \
            --log_to_wandb

    elif [ "${DATASET}" == "leaps" ]; then
        # Generate dataset (LEAPS)
        python3 generate_dataset.py -g karel -b 65536 --seed "${SEED}" --agent playback --karel_task "${task}" # --overwrite

        # Train model
        python3 experiment.py --n_layer 3 --n_head 1 --embed_dim 128 --activation_function "relu" --batch_size 64 --K 20 \
            --env_targets "1" --dropout 0.1 -lr 1e-4 -wd 1e-4 \
            --warmup_steps "${STEPS}" --num_eval_episodes 64 \
            --num_steps_per_iter "${STEPS}" --model_type dt --max_iters 10 --mode delayed \
            --env karel --dataset playback --karel_task "${task}" --scale 1.0 \
            --log_to_wandb # --use_seq_state_embedding
    else
        echo "Invalid dataset: ${DATASET}"
        exit 1
    fi
done