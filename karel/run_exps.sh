#!/bin/bash

DATASET="playback" # "random" or "playback" (leaps)
SEED=68997 # generated using https://www.random.org/
K=30 # max episode length in Karel LEAPS dataset
LOG=false

# Optimizable hyperparams
PCT_TRAJ=0.1
WARMUP_STEPS=1
NUM_STEPS_PER_ITER=1
BATCH_SIZE=64
MAX_ITERS=1
USE_SEQ_STATE_EMBD=true

if [ $# -eq 0 ]; then
    tasks=("cleanHouse" "harvester" "fourCorners" "randomMaze" "stairClimber" "topOff")
else
    tasks="$@"
fi

if [ "${LOG}" = true ]; then
    wandb_flag="--log_to_wandb"
else
    wandb_flag="--no-log_to_wandb"
fi

if [ "${USE_SEQ_STATE_EMBD}" = true ]; then
    seq_state_flag="--use_seq_state_embedding"
else
    seq_state_flag="--no-use_seq_state_embedding"
fi

for task in "${tasks[@]}"; do

    # Generate dataset
    NUM_EPISODES=2
    python3 generate_dataset.py -g karel -n "${NUM_EPISODES}" -b 65536 --seed "${SEED}" --agent "${DATASET}" --karel_task "${task}" # --overwrite

    # Train model
    python3 experiment.py --dataset "${DATASET}" --env karel --karel_task "${task}" --scale 1.0 --mode delayed --model_type dt --num_eval_episodes 64 \
        --n_layer 3 --n_head 1 --embed_dim 128 --activation_function "relu" \
        --dropout 0.1 -lr 1e-4 -wd 1e-4 \
        --batch_size "${BATCH_SIZE}" --K "${K}" --pct_traj "${PCT_TRAJ}" \
        --env_targets "1"   \
        --warmup_steps "${WARMUP_STEPS}" --num_steps_per_iter "${NUM_STEPS_PER_ITER}"  --max_iters "${MAX_ITERS}"  \
        --seed "${SEED}" "${seq_state_flag}" "${wandb_flag}"
    else
        echo "Invalid dataset: ${DATASET}"
        exit 1
    fi
done
