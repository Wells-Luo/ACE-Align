# Data Layout

This release uses short country slugs for all data paths.

```text
data/
  train/
    australia.jsonl
    chile.jsonl
    ...
  eval/
    australia/
    chile/
    ...
```

`data/train/*.jsonl` contains pairwise demographic examples. Each row has:

- `options`
- `group_A.prompt_text`
- `group_A.p_real`
- `group_B.prompt_text`
- `group_B.p_real`

The training script optimizes the EMD distance between the real and predicted
changes in the answer distributions from `group_A` to `group_B`.

`data/eval/<country>/` contains the evaluation targets grouped by the number of
demographic attributes:

```text
data/eval/australia/
  1_attributes/
  2_attributes/
  3_attributes/
  4_attributes/
```

`data/train/all_countries.jsonl` is intentionally not committed because it is a
large generated file. Build it locally with:

```bash
python scripts/build_all_train.py
```

