#! /bin/bash

## STEP A: Generate configuration files for the site and date range
python generate_config.py ws-site1 \
    --start-date 2025-11-01 --end-date 2025-11-07 \
    --nx1 150 --nx2 40 --nx3 40

## STEP B: Generate initial conditions for the site and date range
python prepare_initial_condition.py ws-site1

## STEP C: Run simulation pipeline for the site and date range
python run_frigate_prediction.py -c ws-site1.yaml -i input.part
