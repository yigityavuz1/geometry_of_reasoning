from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.extraction import (
    entropy_from_logits,
    estimate_step_token_spans,
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
    collect_step_signals: bool = False


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


def generate_reasoning_trace(
    prompt: str,
    cfg: GenerationConfig,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if (model is None) != (tokenizer is None):
        raise ValueError("Pass both `model` and `tokenizer`, or neither.")
    if model is None or tokenizer is None:
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
    generated_text = full_text[len(text_prompt) :] if full_text.startswith(text_prompt) else full_text
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

    completion_hidden: torch.Tensor | None = None
    prompt_token_count = int(inputs["input_ids"].shape[-1])
    if cfg.collect_token_embeddings or cfg.collect_step_signals:
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
        completion_hidden = last_hidden[prompt_token_count:].detach().cpu().to(torch.float32)

    if cfg.collect_token_embeddings and completion_hidden is not None:
        trace["token_embeddings"] = completion_hidden.tolist()

    if cfg.collect_step_signals and completion_hidden is not None:
        token_entropies = [float(entropy_from_logits(step_logits).mean().item()) for step_logits in output.scores]
        token_spans = estimate_step_token_spans(
            generated_text=generated_text,
            step_boundaries=boundaries,
            tokenizer=tokenizer,
            n_completion_tokens=int(completion_hidden.shape[0]),
        )
        step_rows: list[dict[str, Any]] = []
        for span in token_spans:
            start = int(span["start_token"])
            end = int(span["end_token"])
            if end <= start:
                continue
            emb_chunk = completion_hidden[start:end]
            entropy_chunk = token_entropies[start:end]
            if emb_chunk.shape[0] == 0 or not entropy_chunk:
                continue
            step_rows.append(
                {
                    "step_index": int(span["step_index"]),
                    "start_token": start,
                    "end_token": end,
                    "entropy_mean": float(sum(entropy_chunk) / len(entropy_chunk)),
                    "embedding_mean": emb_chunk.mean(dim=0).tolist(),
                }
            )
        trace["step_signal_rows"] = step_rows
        trace["token_entropy"] = token_entropies

    return trace
