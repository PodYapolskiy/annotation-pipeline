#!/bin/bash

uv run main.py \
    --config configs/default.yaml \
    --input_dir examples \
    --output_dir examples \
    --min_confidence 0.05 \
