# lmcache-bench
benchmark evaluation for lmcache &amp; vllm native offload strategy

# How to use
1. Install [`vLLM`](https://docs.vllm.ai/en/latest/getting_started/installation.html), [`LMCache`](https://docs.vllm.ai/en/latest/getting_started/examples/lmcache.html)
```bash
git clone git@github.com:Zongjy/lmcache-bench.git
cd lmcache-bench

uv venv --python=3.12
source .venv/bin/activate
uv pip install vllm lmcache 
```
Currently LMCache only supports *mistralai/Mistral-7B-Instruct-v0.2* for **vLLM-v0** &amp; *meta-llama/Meta-Llama-3.1-8B-Instruct* for **vLLM-v1**.

2. Modify run-scripts for testing. Here is the example of `benchmark_latency.sh`:
```bash
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
    # --cpu-offload-gb 20 \
    # --enable-lmcache \
    # -v v1 \
```
Last several args are related two offloading mode respectively:
- [cpu-offload-gb](https://docs.vllm.ai/en/latest/getting_started/examples/basic.html#cpu-offload): vLLM raw implementation for cpu offloading
- [enable-lmcache](https://docs.vllm.ai/en/latest/getting_started/examples/lmcache.html#example-materials): LMCache implementation for cpu offloading, `v1` is for lmcache's config