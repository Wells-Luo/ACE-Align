#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_COUNTRIES = [
    "chile",
    "mexico",
    "united_states",
    "germany",
    "great_britain",
    "russia",
    "india",
    "japan",
    "philippines",
    "egypt",
    "ethiopia",
    "nigeria",
    "australia",
    "new_zealand",
]


def report_value(path: Path, model_name: str, min_samples: int) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    models = payload.get("models", {})
    model_payload = models.get(model_name) or next(iter(models.values()), None)
    if model_payload is None:
        return None

    values: list[float] = []
    for question in model_payload.get("by_question", {}).values():
        for group in question.get("demographic_groups", []):
            if int(group.get("sample_count", 0)) >= min_samples and group.get("emd") is not None:
                values.append(float(group["emd"]))
    if values:
        return mean(values)
    overall = model_payload.get("overall", {}).get("emd", {}).get("mean")
    return float(overall) if overall is not None else None


def categories_for_combo(combo_dir: Path) -> list[str]:
    return sorted(p.stem for p in combo_dir.glob("*.json"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EMD reports.")
    parser.add_argument("--report_dir", default="results/reports")
    parser.add_argument("--output_dir", default="results/csv")
    parser.add_argument("--model_name", default="causal")
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--min_samples", type=int, default=10)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    countries = [c for c in args.countries if (report_dir / c).is_dir()]

    summary_rows: list[dict] = []
    long_rows: list[dict] = []
    for n_attributes in [1, 2, 3, 4]:
        row = {"n_attributes": n_attributes, "model": args.model_name}
        country_scores: list[float] = []
        for country in countries:
            values: list[float] = []
            attr_dir = report_dir / country / f"{n_attributes}_attributes"
            for combo_dir in sorted([p for p in attr_dir.glob("*") if p.is_dir()]):
                for category in categories_for_combo(combo_dir):
                    value = report_value(combo_dir / f"{category}.json", args.model_name, args.min_samples)
                    if value is None:
                        continue
                    values.append(value)
                    long_rows.append(
                        {
                            "n_attributes": n_attributes,
                            "country": country,
                            "category": category,
                            "combination": combo_dir.name,
                            "emd": value,
                            "score_1_minus_emd": 1.0 - value,
                        }
                    )
            if values:
                score = 1.0 - mean(values)
                row[country] = round(score * 100, 2)
                country_scores.append(score)
            else:
                row[country] = ""
        row["overall_mean"] = round(mean(country_scores) * 100, 2) if country_scores else ""
        summary_rows.append(row)

    write_csv(output_dir / "summary.csv", summary_rows, ["n_attributes", "model", *countries, "overall_mean"])
    write_csv(
        output_dir / "long.csv",
        long_rows,
        ["n_attributes", "country", "category", "combination", "emd", "score_1_minus_emd"],
    )
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'long.csv'}")


if __name__ == "__main__":
    main()
