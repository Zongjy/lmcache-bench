# SPDX-License-Identifier: Apache-2.0
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import contextlib
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser
from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME
from vllm.config import KVTransferConfig


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)

def setup_lmcache(vllm_version: str):
    # LMCache-related environment variables
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = "512"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20.0"

    if vllm_version == "v0":
        return "LMCacheConnector"
    else:
        return "LMCacheConnectorV1"


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)
    
    if args.enable_lmcache:
        lmcache_connector = setup_lmcache(args.version)
        print(f"Using LMCache connector: {lmcache_connector}")
    
        ktc = KVTransferConfig(
            kv_connector=lmcache_connector,
            kv_role="kv_both", # LMCache 用于存储和加载 KV cache
        )
        engine_args.kv_transfer_config = ktc

        if args.version == "v0":
            engine_args.enable_chunked_prefill = True



    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = None
    try:
        llm = LLM(**dataclasses.asdict(engine_args))
        assert llm.llm_engine.model_config.max_model_len >= (
            args.input_len +
            args.output_len), ("Please ensure that max_model_len is greater than"
                            " the sum of input_len and output_len.")

        sampling_params = SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.output_len,
            detokenize=not args.disable_detokenize,
        )
        print(sampling_params)
        dummy_prompt_token_ids = np.random.randint(10000,
                                                size=(args.batch_size,
                                                        args.input_len))
        dummy_prompts: list[PromptType] = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]

        def llm_generate():
            if not args.use_beam_search:
                llm.generate(dummy_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False)
            else:
                llm.beam_search(
                    dummy_prompts,
                    BeamSearchParams(
                        beam_width=args.n,
                        max_tokens=args.output_len,
                        ignore_eos=True,
                    ),
                )

        def run_to_completion(profile_dir: Optional[str] = None):
            if profile_dir:
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(
                            str(profile_dir)),
                ) as p:
                    llm_generate()
                print(p.key_averages().table(sort_by="self_cuda_time_total"))
            else:
                start_time = time.perf_counter()
                llm_generate()
                end_time = time.perf_counter()
                latency = end_time - start_time
                return latency

        print("Warming up...")
        for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
            run_to_completion(profile_dir=None)

        if args.profile:
            profile_dir = args.profile_result_dir
            if not profile_dir:
                profile_dir = (Path(".") / "vllm_benchmark_result" /
                            f"latency_result_{time.time()}")
            print(f"Profiling (results will be saved to '{profile_dir}')...")
            run_to_completion(profile_dir=profile_dir)
            return
    finally:
        if args.enable_lmcache:
            print("Cleaning up LMCache backend...")
            LMCacheEngineBuilder.destroy(ENGINE_NAME)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=None,
        help=("path to save the pytorch profiler output. Can be visualized "
              "with ui.perfetto.dev or Tensorboard."),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )
    parser.add_argument(
        "--enable-lmcache",
        action="store_true",
        help="Enable LMCache integration (default: disabled)",
    )
    parser.add_argument(
        "-v",
        "--version",
        choices=["v0", "v1"],
        default="v1",
        help="Specify vLLM version for LMCache connector (default: v1)",
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
