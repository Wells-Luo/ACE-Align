from __future__ import annotations

import contextlib
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-8


def normalize_distribution(p_dict: Dict[str, float], options: Iterable[int]) -> torch.Tensor:
    values = torch.tensor([float(p_dict.get(str(opt), 0.0)) for opt in options], dtype=torch.float32)
    total = values.sum()
    if total <= EPS:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    return values / total


def emd_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (torch.cumsum(p, dim=-1) - torch.cumsum(q, dim=-1)).abs().sum(dim=-1)


def normalized_emd_from_dicts(p_real: Dict[str, float], p_model: Dict[str, float], options: List[int]) -> float:
    p = normalize_distribution(p_real, options)
    q = normalize_distribution(p_model, options)
    raw = emd_1d(p.unsqueeze(0), q.unsqueeze(0)).item()
    span = float(max(options) - min(options)) if options else 1.0
    return raw / (span if span > 0 else 1.0)


def load_model(
    *,
    base_model: str,
    adapter_path: str | None,
    tokenizer_path: str | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32, trust_remote_code=True)
        model = model.to("cpu")
    else:
        device_id = int(device.split(":")[1]) if ":" in device else 0
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map={"": device_id},
            trust_remote_code=True,
        )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def option_distribution(
    *,
    model,
    tokenizer,
    prompts: List[str],
    options_batch: List[List[int]],
    softmax_method: str = "selective",
) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    autocast_ctx = torch.cuda.amp.autocast if str(device).startswith("cuda") else contextlib.nullcontext

    with torch.no_grad():
        with autocast_ctx():
            outputs = model(**inputs)
            attention = inputs.get("attention_mask")
            if attention is None:
                logits = outputs.logits[:, -1, :]
            else:
                # Works for both left and right padding.  `attention.sum() - 1`
                # is only correct for right padding.
                positions = attention.size(1) - 1 - attention.flip(dims=[1]).argmax(dim=1)
                batch_idx = torch.arange(len(prompts), device=outputs.logits.device)
                logits = outputs.logits[batch_idx, positions, :]

    max_options = max(len(opts) for opts in options_batch)
    if softmax_method == "selective":
        option_logits = torch.full(
            (len(prompts), max_options),
            torch.finfo(logits.dtype).min,
            dtype=logits.dtype,
            device=logits.device,
        )
        for i, options in enumerate(options_batch):
            token_ids = [tokenizer.encode(str(opt), add_special_tokens=False)[0] for opt in options]
            option_logits[i, : len(options)] = logits[i, torch.tensor(token_ids, device=logits.device)]
        return F.softmax(option_logits.float(), dim=-1)

    if softmax_method == "full":
        vocab_probs = F.softmax(logits.float(), dim=-1)
        option_probs = torch.zeros((len(prompts), max_options), dtype=vocab_probs.dtype, device=vocab_probs.device)
        for i, options in enumerate(options_batch):
            token_ids = [tokenizer.encode(str(opt), add_special_tokens=False)[0] for opt in options]
            option_probs[i, : len(options)] = vocab_probs[i, torch.tensor(token_ids, device=vocab_probs.device)]
        return option_probs / option_probs.sum(dim=-1, keepdim=True).clamp_min(EPS)

    raise ValueError(f"Unknown softmax method: {softmax_method}")


def distributions_to_dict(distributions: torch.Tensor, options_batch: List[List[int]]) -> List[Dict[str, float]]:
    out: list[dict[str, float]] = []
    for row, options in zip(distributions, options_batch):
        out.append({str(opt): float(row[j].detach().cpu()) for j, opt in enumerate(options)})
    return out
