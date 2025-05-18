#!/bin/bash

uv run main.py \
    --config configs/default.yaml \
    --input_dir examples/raw \
    --output_dir examples/processed \
    --min_confidence 0.029 \
