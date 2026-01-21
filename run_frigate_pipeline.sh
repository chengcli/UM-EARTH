#!/bin/bash

## STEP A: Generate configuration files for the site and date range
#python generate_config.py ws-site1 \
#    --start-date 2025-11-03 --end-date 2025-11-07 \
#    --x1-max 10000 --x2-extent 120000 --x3-extent 120000 \
#    --nx1 50 --nx2 60 --nx3 60

## STEP B: Generate initial conditions for the site and date range
#python prepare_initial_condition.py ws-site1
#python prepare_initial_condition.py ws-site1 --start-from 2

## STEP C: Run simulation pipeline for the site and date range
python run_frigate_prediction.py -c ws-site1.yaml -i input.part
