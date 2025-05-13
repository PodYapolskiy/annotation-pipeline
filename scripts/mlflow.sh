#!/bin/bash

uvx mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri ./mlflow \
    --default-artifact-root ./mlflow \
