#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from infer_eval import (
    available_combinations,
    available_countries,
    build_prompts,
    report_complete,
    save_model_results,
    save_reports,
)
from utils.distribution import normalized_emd_from_dicts


def option_token_id(tokenizer, option: int) -> int:
    token_ids = tokenizer.encode(str(option), add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Option {option!r} is not a single token for this tokenizer: {token_ids}")
    return int(token_ids[0])


def batch_option_distributions(
    *,
    llm: LLM,
    tokenizer,
    prompts: List[str],
    options_batch: List[List[int]],
    lora_request: LoRARequest | None,
) -> list[dict[str, float]]:
    allowed_token_ids = sorted({option_token_id(tokenizer, opt) for options in options_batch for opt in options})
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        allowed_token_ids=allowed_token_ids,
        logprobs=len(allowed_token_ids),
        detokenize=False,
    )
    outputs = llm.generate(prompts, params, lora_request=lora_request)

    distributions: list[dict[str, float]] = []
    for output, options in zip(outputs, options_batch):
        token_logprobs = output.outputs[0].logprobs
        if not token_logprobs:
            raise RuntimeError("vLLM did not return token logprobs; check SamplingParams.logprobs.")
        step_logprobs = token_logprobs[0]
        logits: list[float] = []
        for option in options:
            token_id = option_token_id(tokenizer, option)
            logprob = step_logprobs.get(token_id)
            logits.append(float(logprob.logprob) if logprob is not None else -1e30)

        max_logit = max(logits)
        exp_values = [math.exp(x - max_logit) for x in logits]
        total = sum(exp_values)
        distributions.append({str(option): value / total for option, value in zip(options, exp_values)})
    return distributions


def infer_items_vllm(
    *,
    llm: LLM,
    tokenizer,
    items: list[dict],
    batch_size: int,
    lora_request: LoRARequest | None,
) -> list[dict]:
    results = []
    for start in tqdm(range(0, len(items), batch_size), desc="vllm inference"):
        batch = items[start : start + batch_size]
        prompts = build_prompts(batch)
        options_batch = [[int(x) for x in item["options"]] for item in batch]
        p_model_dicts = batch_option_distributions(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            options_batch=options_batch,
            lora_request=lora_request,
        )
        for item, p_model, options in zip(batch, p_model_dicts, options_batch):
            row = dict(item)
            row["p_model"] = p_model
            row["emd"] = normalized_emd_from_dicts(row["p_real"], row["p_model"], options)
            results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM inference and EMD evaluation.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--model_name", default="causal")
    parser.add_argument("--eval_dir", default="data/eval")
    parser.add_argument("--result_dir", default="results/predictions")
    parser.add_argument("--report_dir", default="results/reports")
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--n_attributes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_complete", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    result_dir = Path(args.result_dir)
    report_dir = Path(args.report_dir)
    countries = args.countries or available_countries(eval_dir)

    print(f"model_name={args.model_name}")
    print(f"countries={countries}")
    print(f"n_attributes={args.n_attributes}")
    print("backend=vllm")

    llm = LLM(
        model=args.base_model,
        tokenizer=args.tokenizer_path or args.base_model,
        enable_lora=bool(args.adapter_path),
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    lora_request = LoRARequest(args.model_name, 1, args.adapter_path) if args.adapter_path else None

    config = {
        "backend": "vllm",
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    for country in countries:
        for n_attributes in args.n_attributes:
            for combo_dir in available_combinations(eval_dir, country, n_attributes):
                category_files = sorted(combo_dir.glob("*_real.json"))
                categories = [p.name.removesuffix("_real.json") for p in category_files]
                if args.skip_complete and report_complete(report_dir, country, n_attributes, combo_dir.name, categories):
                    print(f"[skip] {country} n={n_attributes} {combo_dir.name}")
                    continue
                print(f"[combo] {country} n={n_attributes} {combo_dir.name}")
                category_results: Dict[str, list[dict]] = {}
                for path in category_files:
                    category = path.name.removesuffix("_real.json")
                    with path.open("r", encoding="utf-8") as f:
                        items = json.load(f)
                    print(f"[infer] {country} n={n_attributes} {combo_dir.name} {category} items={len(items)}")
                    category_results[category] = infer_items_vllm(
                        llm=llm,
                        tokenizer=tokenizer,
                        items=items,
                        batch_size=args.batch_size,
                        lora_request=lora_request,
                    )
                save_model_results(
                    result_dir=result_dir,
                    model_name=args.model_name,
                    country=country,
                    n_attributes=n_attributes,
                    combination=combo_dir.name,
                    category_results=category_results,
                    config=config,
                )
                save_reports(
                    report_dir=report_dir,
                    model_name=args.model_name,
                    country=country,
                    n_attributes=n_attributes,
                    combination=combo_dir.name,
                    category_results=category_results,
                )
    print("[done]")


if __name__ == "__main__":
    main()
