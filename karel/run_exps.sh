#!/bin/bash

SEED=68997 # generated using https://www.random.org/
PCT_TRAJ=0.4
WARMUP_STEPS=1
NUM_STEPS_PER_ITER=1
BATCH_SIZE=64
K=50
MAX_ITERS=10
DATASET="leaps" # "random" or "leaps"

if [ $# -eq 0 ]; then
    tasks=("cleanHouse" "harvester" "fourCorners" "randomMaze" "stairClimber" "topOff")
else
    tasks="$@"
fi

for task in "${tasks[@]}"; do

    if [ "${DATASET}" == "random" ]; then
        NUM_EPISODES=2

        # Generate dataset (random)
        python3 generate_dataset.py -g karel -n "${NUM_EPISODES}" -b 65536 --seed "${SEED}" --karel_task "${task}" --overwrite

        # Train model
        python3 experiment.py --n_layer 3 --n_head 1 --embed_dim 128 --activation_function "relu" --batch_size "${BATCH_SIZE}" --K "${K}" \
            --env_targets "1" --dropout 0.1 -lr 1e-4 -wd 1e-4 --pct_traj "${PCT_TRAJ}" \
            --warmup_steps "${WARMUP_STEPS}" --num_eval_episodes 64 \
            --num_steps_per_iter "${NUM_STEPS_PER_ITER}" --model_type dt --max_iters "${MAX_ITERS}" --mode delayed \
            --env karel --dataset random --karel_task "${task}" --scale 1.0 --seed "${SEED}" \
            --log_to_wandb # --use_seq_state_embedding

    elif [ "${DATASET}" == "leaps" ]; then
        # Generate dataset (LEAPS)
        python3 generate_dataset.py -g karel -b 65536 --seed "${SEED}" --agent playback --karel_task "${task}" # --overwrite

        # Train model
        python3 experiment.py --n_layer 3 --n_head 1 --embed_dim 128 --activation_function "relu" --batch_size "${BATCH_SIZE}" --K "${K}" \
            --env_targets "1" --dropout 0.1 -lr 1e-4 -wd 1e-4 --pct_traj "${PCT_TRAJ}" \
            --warmup_steps "${WARMUP_STEPS}" --num_eval_episodes 64 \
            --num_steps_per_iter "${NUM_STEPS_PER_ITER}" --model_type dt --max_iters "${MAX_ITERS}" --mode delayed \
            --env karel --dataset playback --karel_task "${task}" --scale 1.0 --seed "${SEED}" \
            --log_to_wandb # --use_seq_state_embedding
    else
        echo "Invalid dataset: ${DATASET}"
        exit 1
    fi
done
