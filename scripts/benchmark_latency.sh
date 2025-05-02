# !/bin/bash

WORKSPACE=$(pwd)

python $WORKSPACE/benchmarks/benchmark_latency.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --input-len 16384 \
    --output-len 512 \
    --batch-size 8 \
    --num-iters 5 \
    --num-iters-warmup 5 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32K \
    --output-json $WORKSPACE/results/latency.json \
    --cpu-offload-gb 20 \
    # --enable-lmcache \
    # -v v1 \