#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

EPS = 1e-8


class StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf: str) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self) -> None:
        pass


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


def distribution_tensor(p_dict: Dict[str, float], options: List[int]) -> torch.Tensor:
    values = torch.tensor([float(p_dict.get(str(opt), 0.0)) for opt in options], dtype=torch.float32)
    total = values.sum()
    if total <= EPS:
        return torch.full_like(values, 1.0 / len(options))
    return values / total


class PairDistributionDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.items: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                options = [int(x) for x in obj["options"]]
                group_a = obj["group_A"]
                group_b = obj["group_B"]
                self.items.append(
                    {
                        "prompt_a": group_a["prompt_text"],
                        "prompt_b": group_b["prompt_text"],
                        "options": options,
                        "p_real_a": distribution_tensor(group_a["p_real"], options),
                        "p_real_b": distribution_tensor(group_b["p_real"], options),
                        "mode_option_a": group_a["mode_option"],
                        "mode_option_b": group_b["mode_option"],
                    }
                )
        if not self.items:
            raise ValueError(f"No training examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


@dataclass
class Batch:
    prompts_a: list[str]
    prompts_b: list[str]
    options: list[list[int]]
    p_real_a: torch.Tensor
    p_real_b: torch.Tensor
    mode_option_a: torch.Tensor
    mode_option_b: torch.Tensor


def collate(batch: list[dict[str, Any]]) -> Batch:
    max_options = max(len(x["options"]) for x in batch)

    def pad(p: torch.Tensor) -> torch.Tensor:
        if p.numel() == max_options:
            return p
        return F.pad(p, (0, max_options - p.numel()), value=0.0)

    return Batch(
        prompts_a=[x["prompt_a"] for x in batch],
        prompts_b=[x["prompt_b"] for x in batch],
        options=[x["options"] for x in batch],
        p_real_a=torch.stack([pad(x["p_real_a"]) for x in batch]),
        p_real_b=torch.stack([pad(x["p_real_b"]) for x in batch]),
        mode_option_a=torch.tensor([int(x["mode_option_a"]) for x in batch], dtype=torch.long),
        mode_option_b=torch.tensor([int(x["mode_option_b"]) for x in batch], dtype=torch.long),
    )


class EMDTrainer:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        dataset: PairDistributionDataset,
        accelerator: Accelerator,
        output_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        gradient_accumulation_steps: int,
        logging_steps: int,
        objective_schedule: list[str],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.logging_steps = logging_steps
        self.objective_schedule = objective_schedule

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        self.model, self.optimizer, self.loader = accelerator.prepare(self.model, self.optimizer, self.loader)
        self.global_step = 0
        self._token_cache: dict[tuple[int, ...], list[int]] = {}

    def _token_ids(self, options: list[int]) -> list[int]:
        key = tuple(options)
        if key not in self._token_cache:
            ids = []
            for opt in options:
                tokenized = self.tokenizer.encode(str(opt), add_special_tokens=False)
                if len(tokenized) != 1:
                    raise ValueError(f"Option {opt!r} is not a single token for this tokenizer.")
                ids.append(tokenized[0])
            self._token_cache[key] = ids
        return self._token_cache[key]

    def _token_id(self, option: int) -> int:
        return self._token_ids([option])[0]

    def _answer_logits(self, prompts: list[str]) -> torch.Tensor:
        prompts = [p.rstrip() + "\n" for p in prompts]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        attention = inputs.get("attention_mask")
        if attention is None:
            return outputs.logits[:, -1, :]
        # Works for both left and right padding.  `attention.sum() - 1`
        # is only correct for right padding.
        positions = attention.size(1) - 1 - attention.flip(dims=[1]).argmax(dim=1)
        batch_idx = torch.arange(len(prompts), device=outputs.logits.device)
        return outputs.logits[batch_idx, positions, :]

    def _option_distribution(self, logits: torch.Tensor, options_batch: list[list[int]]) -> torch.Tensor:
        max_options = max(len(x) for x in options_batch)
        option_logits = torch.full(
            (len(options_batch), max_options),
            torch.finfo(logits.dtype).min,
            dtype=logits.dtype,
            device=logits.device,
        )
        for i, options in enumerate(options_batch):
            ids = torch.tensor(self._token_ids(options), device=logits.device)
            option_logits[i, : len(options)] = logits[i, ids]
        return F.softmax(option_logits.float(), dim=-1)

    def anchor_loss(self, batch: Batch, *, reverse_pair: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        prompts_a = batch.prompts_b if reverse_pair else batch.prompts_a
        prompts_b = batch.prompts_a if reverse_pair else batch.prompts_b
        mode_option_a = batch.mode_option_b if reverse_pair else batch.mode_option_a
        mode_option_b = batch.mode_option_a if reverse_pair else batch.mode_option_b

        logits_a = self._answer_logits(prompts_a)
        logits_b = self._answer_logits(prompts_b)
        labels_a = torch.tensor(
            [self._token_id(int(x)) for x in mode_option_a.tolist()],
            device=logits_a.device,
            dtype=torch.long,
        )
        labels_b = torch.tensor(
            [self._token_id(int(x)) for x in mode_option_b.tolist()],
            device=logits_b.device,
            dtype=torch.long,
        )
        loss_anchor_a = F.cross_entropy(logits_a.float(), labels_a)
        loss_anchor_b = F.cross_entropy(logits_b.float(), labels_b)
        loss_anchor = loss_anchor_a + loss_anchor_b
        return loss_anchor, {
            "loss": float(loss_anchor.detach().cpu()),
            "loss_anchor": float(loss_anchor.detach().cpu()),
            "loss_anchor_a": float(loss_anchor_a.detach().cpu()),
            "loss_anchor_b": float(loss_anchor_b.detach().cpu()),
        }

    def distribution_loss(self, batch: Batch, *, reverse_pair: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        prompts_a = batch.prompts_b if reverse_pair else batch.prompts_a
        prompts_b = batch.prompts_a if reverse_pair else batch.prompts_b
        p_real_a_batch = batch.p_real_b if reverse_pair else batch.p_real_a
        p_real_b_batch = batch.p_real_a if reverse_pair else batch.p_real_b

        logits_a = self._answer_logits(prompts_a)
        logits_b = self._answer_logits(prompts_b)

        p_model_a = self._option_distribution(logits_a, batch.options)
        p_model_b = self._option_distribution(logits_b, batch.options)

        p_real_a = p_real_a_batch.to(p_model_a.device)
        p_real_b = p_real_b_batch.to(p_model_b.device)

        lengths = torch.tensor([len(x) for x in batch.options], device=p_model_a.device)
        idx = torch.arange(p_model_a.shape[-1], device=p_model_a.device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).float()

        delta_model = torch.cumsum(p_model_b * mask, dim=-1) - torch.cumsum(p_model_a * mask, dim=-1)
        delta_real = torch.cumsum(p_real_b * mask, dim=-1) - torch.cumsum(p_real_a * mask, dim=-1)
        per_sample = ((delta_model - delta_real).abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        loss_emd = per_sample.mean()
        metrics = {
            "loss": float(loss_emd.detach().cpu()),
            "loss_emd": float(loss_emd.detach().cpu()),
        }
        return loss_emd, metrics

    def loss(self, batch: Batch, *, objective: str, reverse_pair: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        if objective == "anchor":
            return self.anchor_loss(batch, reverse_pair=reverse_pair)
        if objective == "distribution":
            return self.distribution_loss(batch, reverse_pair=reverse_pair)
        raise ValueError(f"Unknown objective: {objective}")

    def save(self, name: str) -> None:
        if not self.accelerator.is_main_process:
            return
        out = self.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        self.accelerator.unwrap_model(self.model).save_pretrained(out)
        self.tokenizer.save_pretrained(out)

    def train(self) -> None:
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print("Causal EMD training")
            print(f"train_examples={len(self.dataset)}")
            print(f"batch_size_per_device={self.batch_size}")
            print(f"num_processes={self.accelerator.num_processes}")
            print(f"gradient_accumulation={self.accelerator.gradient_accumulation_steps}")
            print(f"objective_schedule={','.join(self.objective_schedule)}")

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            objective = self.objective_schedule[min(epoch - 1, len(self.objective_schedule) - 1)]
            reverse_pair = objective == "distribution" and epoch % 2 == 0
            if self.accelerator.is_main_process:
                direction = "1->0" if reverse_pair else "0->1"
                print(f"epoch={epoch} objective={objective} pair_direction={direction}")
            progress = tqdm(self.loader, desc=f"epoch {epoch}/{self.epochs}", disable=not self.accelerator.is_main_process)
            running: dict[str, float] = {}
            logged = 0
            for batch in progress:
                with self.accelerator.accumulate(self.model):
                    loss, metrics = self.loss(batch, objective=objective, reverse_pair=reverse_pair)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        if not running:
                            running = {key: 0.0 for key in metrics}
                        for key in running:
                            running[key] += metrics[key]
                        logged += 1
                        if self.accelerator.is_main_process and self.global_step % self.logging_steps == 0:
                            denom = max(logged, 1)
                            avg = {key: value / denom for key, value in running.items()}
                            metric_text = " ".join(f"{key}={value:.6f}" for key, value in avg.items())
                            print(f"step={self.global_step} epoch={epoch} objective={objective} {metric_text}")
                            running = {key: 0.0 for key in metrics}
                            logged = 0
            self.save(f"epoch-{epoch}")
        self.save("final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with causal EMD loss.")
    parser.add_argument("--train_data", default="data/train/all_countries.jsonl")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output_dir", default="outputs/causal")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--objective_schedule", default="anchor,distribution")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--log_file", default="logs/train.log")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)
    if accelerator.is_main_process:
        setup_logging(Path(args.log_file))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = PairDistributionDataset(args.train_data, max_samples=args.max_samples)
    objective_schedule = [x.strip() for x in args.objective_schedule.split(",") if x.strip()]
    if not objective_schedule:
        raise ValueError("--objective_schedule must contain at least one objective")
    unknown = sorted(set(objective_schedule) - {"anchor", "distribution"})
    if unknown:
        raise ValueError(f"Unknown objectives in --objective_schedule: {unknown}")
    trainer = EMDTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        accelerator=accelerator,
        output_dir=Path(args.output_dir),
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        objective_schedule=objective_schedule,
    )
    trainer.train()


if __name__ == "__main__":
    main()
