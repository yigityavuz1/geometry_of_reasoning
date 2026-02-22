from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.extraction import find_step_boundaries


@dataclass
class GenerationConfig:
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = False


def _build_step_prompt(user_prompt: str) -> str:
    return (
        "Solve the problem with explicit numbered reasoning steps.\n"
        "Use format: Step 1:, Step 2:, ... and finish with Final Answer:.\n\n"
        f"Problem:\n{user_prompt.strip()}"
    )


def load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_reasoning_trace(prompt: str, cfg: GenerationConfig) -> dict[str, Any]:
    model, tokenizer = load_model_and_tokenizer(cfg.model_name)
    text_prompt = _build_step_prompt(prompt)
    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

    return {
        "prompt": prompt,
        "model_name": cfg.model_name,
        "generated_text": generated_text,
        "step_boundaries": boundaries,
        "token_count": int(output.sequences.shape[-1]),
    }
