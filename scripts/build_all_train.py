#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_COUNTRIES = [
    "australia",
    "chile",
    "egypt",
    "ethiopia",
    "germany",
    "great_britain",
    "india",
    "japan",
    "mexico",
    "new_zealand",
    "nigeria",
    "philippines",
    "russia",
    "united_states",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate per-country training jsonl files.")
    parser.add_argument("--train_dir", default="data/train")
    parser.add_argument("--output", default="data/train/all_countries.jsonl")
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as out:
        for country in args.countries:
            path = train_dir / f"{country}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        count += 1
    print(f"Wrote {count} examples to {output}")


if __name__ == "__main__":
    main()

