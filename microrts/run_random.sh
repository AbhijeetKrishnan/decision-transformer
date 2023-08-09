#!/bin/bash

# Delete existing datasets
rm data/*.pkl

tasks=("cleanHouse" "harvester" "fourCorners" "randomMaze" "stairClimber" "topOff")

for task in "${tasks[@]}"; do

    # Generate dataset
    python3 generate_random_dataset.py -g karel -n 3 -b 65536 --seed 0 --karel_task "${task}" --overwrite --format pkl

done