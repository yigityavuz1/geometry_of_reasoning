from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.extraction import (
    find_step_boundaries,
    split_steps,
    summarize_token_entropies,
)


@dataclass
class GenerationConfig:
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = False
    collect_token_embeddings: bool = False


def _build_step_prompt(user_prompt: str) -> str:
    return (
        "Solve the problem with explicit numbered reasoning steps.\n"
        "Use format: Step 1:, Step 2:, ... and finish with Final Answer:.\n\n"
        f"Problem:\n{user_prompt.strip()}"
    )


def _resolve_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    forced_device = os.getenv("GOR_DEVICE", "").strip().lower()
    if forced_device in {"cpu", "cuda", "mps"}:
        if forced_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GOR_DEVICE=cuda set, but CUDA is not available.")
        if forced_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("GOR_DEVICE=mps set, but MPS is not available.")
        if forced_device == "mps":
            return torch.device("mps"), torch.float16
        if forced_device == "cuda":
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cpu"), torch.float32

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    # Default to CPU on Apple Silicon for stability.
    return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device, dtype = _resolve_device_and_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_reasoning_trace(prompt: str, cfg: GenerationConfig) -> dict[str, Any]:
    model, tokenizer = load_model_and_tokenizer(cfg.model_name)
    device = next(model.parameters()).device
    text_prompt = _build_step_prompt(prompt)
    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )

    full_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    generated_text = full_text[len(text_prompt) :].strip() if full_text.startswith(text_prompt) else full_text
    boundaries = find_step_boundaries(generated_text)
    steps = split_steps(generated_text)
    entropy_summary = summarize_token_entropies(output.scores)

    trace: dict[str, Any] = {
        "prompt": prompt,
        "model_name": cfg.model_name,
        "generated_text": generated_text,
        "step_texts": steps,
        "step_boundaries": boundaries,
        "token_count": int(output.sequences.shape[-1]),
        "entropy_summary": entropy_summary,
    }

    if cfg.collect_token_embeddings:
        with torch.no_grad():
            sequence = output.sequences.to(device)
            attention_mask = torch.ones_like(sequence, device=device)
            forward = model(
                input_ids=sequence,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = forward.hidden_states[-1][0]
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        completion_hidden = last_hidden[prompt_token_count:]
        trace["token_embeddings"] = completion_hidden.detach().cpu().to(torch.float32).tolist()

    return trace
