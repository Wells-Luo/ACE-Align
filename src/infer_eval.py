#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from utils.distribution import (
    distributions_to_dict,
    load_model,
    normalized_emd_from_dicts,
    option_distribution,
)
from utils.prompt import build_survey_prompt


def country_display_name(slug: str) -> str:
    return slug.replace("_", " ").title().replace("United States", "United States").replace("Great Britain", "Great Britain")


def available_countries(eval_dir: Path) -> list[str]:
    return sorted(p.name for p in eval_dir.iterdir() if p.is_dir())


def available_combinations(eval_dir: Path, country: str, n_attributes: int) -> list[Path]:
    root = eval_dir / country / f"{n_attributes}_attributes"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def build_prompts(items: list[dict]) -> list[str]:
    prompts = []
    for item in items:
        demo = item["demographics"]
        prompts.append(
            build_survey_prompt(
                question_text=item["question_text"],
                options=item["options"],
                country=demo["country"],
                gender=demo.get("gender"),
                residence=demo.get("residence"),
                education=demo.get("education"),
                marital_status=demo.get("marital_status"),
            )
        )
    return prompts


def infer_items(model, tokenizer, items: list[dict], batch_size: int, softmax_method: str) -> list[dict]:
    results = []
    for start in tqdm(range(0, len(items), batch_size), desc="inference"):
        batch = items[start : start + batch_size]
        prompts = build_prompts(batch)
        options_batch = [[int(x) for x in item["options"]] for item in batch]
        p_model = option_distribution(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            options_batch=options_batch,
            softmax_method=softmax_method,
        )
        for item, p_model_dict, options in zip(batch, distributions_to_dict(p_model, options_batch), options_batch):
            row = dict(item)
            row["p_model"] = p_model_dict
            row["emd"] = normalized_emd_from_dicts(row["p_real"], row["p_model"], options)
            results.append(row)
    return results


def save_model_results(
    *,
    result_dir: Path,
    model_name: str,
    country: str,
    n_attributes: int,
    combination: str,
    category_results: Dict[str, list[dict]],
    config: dict,
) -> None:
    out_dir = result_dir / model_name / f"{n_attributes}_attributes" / combination
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "country": country,
        "n_attributes": n_attributes,
        "attribute_combination": combination,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "categories": {cat: {"items": rows} for cat, rows in category_results.items()},
    }
    with (out_dir / f"{country}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def summarize_category(rows: list[dict]) -> dict:
    values = [float(row["emd"]) for row in rows]
    by_question: dict[str, dict] = {}
    for row in rows:
        qid = str(row["question_id"])
        entry = by_question.setdefault(qid, {"question_text": row["question_text"], "demographic_groups": []})
        entry["demographic_groups"].append(
            {
                "demographics": row["demographics"],
                "sample_count": int(row.get("sample_count", 0)),
                "emd": float(row["emd"]),
            }
        )
    for entry in by_question.values():
        question_values = [g["emd"] for g in entry["demographic_groups"]]
        entry["overall"] = {"emd": {"mean": sum(question_values) / len(question_values)}}
    return {
        "total_items": len(rows),
        "overall": {"emd": {"mean": sum(values) / len(values) if values else None}},
        "by_question": by_question,
    }


def save_reports(
    *,
    report_dir: Path,
    model_name: str,
    country: str,
    n_attributes: int,
    combination: str,
    category_results: Dict[str, list[dict]],
) -> None:
    out_dir = report_dir / country / f"{n_attributes}_attributes" / combination
    out_dir.mkdir(parents=True, exist_ok=True)
    for category, rows in category_results.items():
        payload = {
            "timestamp": datetime.now().isoformat(),
            "metric": "emd",
            "models": {model_name: summarize_category(rows)},
        }
        with (out_dir / f"{category}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def report_complete(report_dir: Path, country: str, n_attributes: int, combination: str, categories: list[str]) -> bool:
    combo_dir = report_dir / country / f"{n_attributes}_attributes" / combination
    return all((combo_dir / f"{cat}.json").exists() for cat in categories)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and EMD evaluation.")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--softmax_method", default="selective", choices=["selective", "full"])
    parser.add_argument("--skip_complete", action="store_true")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    eval_dir = Path(args.eval_dir)
    result_dir = Path(args.result_dir)
    report_dir = Path(args.report_dir)
    countries = args.countries or available_countries(eval_dir)

    print(f"model_name={args.model_name}")
    print(f"countries={countries}")
    print(f"n_attributes={args.n_attributes}")
    model, tokenizer = load_model(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        dtype=dtype,
    )

    config = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "softmax_method": args.softmax_method,
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
                category_results: dict[str, list[dict]] = {}
                for path in category_files:
                    category = path.name.removesuffix("_real.json")
                    with path.open("r", encoding="utf-8") as f:
                        items = json.load(f)
                    print(f"[infer] {country} n={n_attributes} {combo_dir.name} {category} items={len(items)}")
                    category_results[category] = infer_items(
                        model=model,
                        tokenizer=tokenizer,
                        items=items,
                        batch_size=args.batch_size,
                        softmax_method=args.softmax_method,
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
