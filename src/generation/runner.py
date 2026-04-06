from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
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
    capture_layer_names: tuple[str, ...] = ("early", "middle", "late")
    retry_on_placeholder_output: bool = True
    max_format_retries: int = 1
    seed: int | None = None


_RUNTIME_CONFIGURED = False
LOGGER = logging.getLogger("gor.runner")
_PLACEHOLDER_OUTPUT_PATTERNS = (
    re.compile(r"<\s*equation\s*>", flags=re.IGNORECASE),
    re.compile(r"<\s*single\s+number\s*>", flags=re.IGNORECASE),
    re.compile(r"(?im)^\s*step\s*\d+\s*:\s*<[^>\n]+>\s*$"),
    re.compile(r"(?im)^\s*final\s+answer\s*:\s*<[^>\n]+>\s*$"),
)


def _build_step_prompt(user_prompt: str, *, strict_retry: bool = False) -> str:
    retry_notice = (
        "Retry note:\n"
        "- A previous attempt copied template placeholders or formatting instead of solving the problem.\n"
        "- Rewrite the full solution with real equations and one numeric final answer.\n"
        "- Do not use angle brackets anywhere in the response.\n\n"
        if strict_retry
        else ""
    )
    return (
        "Solve the arithmetic word problem.\n"
        "Use ONLY this output format:\n"
        "Step 1: 12 - 5 = 7\n"
        "Step 2: 7 + 9 = 16\n"
        "Step 3: 16 / 2 = 8\n"
        "Final Answer: 8\n"
        "Rules:\n"
        "- Start directly with 'Step 1:' (no preface text).\n"
        "- Every step must contain at least one explicit equation using '='.\n"
        "- Use numeric digits (example: 16, 3.5), not spelled-out numbers.\n"
        "- Keep steps short and calculation-focused.\n"
        "- Do not use bullet points, markdown, or commentary around the equation.\n"
        "- Do not include 'because', 'therefore', or explanation text after the equation.\n"
        "- Never output placeholder text or any angle-bracket template copied from the instructions.\n"
        "- 'Final Answer:' must appear once on its own line and must contain the actual numeric answer.\n\n"
        f"{retry_notice}"
        "Example 1\n"
        "Problem: A shelf has 12 books. 5 are borrowed and then 9 new books are added. How many books are on the shelf now?\n"
        "Answer:\n"
        "Step 1: 12 - 5 = 7\n"
        "Step 2: 7 + 9 = 16\n"
        "Final Answer: 16\n\n"
        "Example 2\n"
        "Problem: A runner completes 3 laps each day for 6 days. Each lap is 400 meters. What total distance does the runner cover in meters?\n"
        "Answer:\n"
        "Step 1: 3 * 6 = 18\n"
        "Step 2: 18 * 400 = 7200\n"
        "Final Answer: 7200\n\n"
        "Example 3\n"
        "Problem: A cafe sells 15 sandwiches at $4 each and 8 drinks at $3 each. What is the total revenue in dollars?\n"
        "Answer:\n"
        "Step 1: 15 * 4 = 60\n"
        "Step 2: 8 * 3 = 24\n"
        "Step 3: 60 + 24 = 84\n"
        "Final Answer: 84\n\n"
        "Now solve this problem.\n"
        f"Problem: {user_prompt.strip()}\n"
        "Answer:\n"
    )


def _placeholder_retry_reason(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    for pattern in _PLACEHOLDER_OUTPUT_PATTERNS:
        if pattern.search(stripped):
            return "placeholder_output"
    return None


def _resolve_capture_layers(total_layers: int, requested_names: tuple[str, ...]) -> list[tuple[str, int]]:
    if total_layers <= 0:
        return [("late", -1)]

    resolved: list[tuple[str, int]] = []
    quarter = max(1, int(round(total_layers * 0.25)))
    halfway = max(1, int(round(total_layers * 0.50)))
    default_map = {
        "early": quarter,
        "middle": halfway,
        "late": total_layers,
    }
    for raw_name in requested_names:
        name = str(raw_name).strip().lower()
        if not name:
            continue
        layer_index = default_map.get(name)
        if layer_index is None:
            try:
                parsed = int(name)
            except ValueError:
                raise ValueError(
                    f"Unsupported capture layer name '{raw_name}'. Use early, middle, late, or an integer layer index."
                ) from None
            layer_index = min(max(1, parsed), total_layers)
            name = f"layer_{layer_index}"
        resolved.append((name, int(layer_index)))

    if not resolved:
        return [("late", total_layers)]
    return resolved


def collect_model_metadata(model_name: str, model: Any, tokenizer: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}

    model_revision = (
        getattr(config, "_commit_hash", None)
        or getattr(generation_config, "_commit_hash", None)
        or getattr(model, "_commit_hash", None)
        or ""
    )
    tokenizer_revision = tokenizer_kwargs.get("_commit_hash") or ""

    device = ""
    dtype = ""
    try:
        first_param = next(model.parameters())
        device = str(first_param.device)
        dtype = str(first_param.dtype).replace("torch.", "")
    except (StopIteration, AttributeError):
        pass

    return {
        "requested_model_name": model_name,
        "resolved_model_name": (
            getattr(config, "_name_or_path", None)
            or getattr(model, "name_or_path", None)
            or model_name
        ),
        "model_revision": str(model_revision),
        "tokenizer_revision": str(tokenizer_revision),
        "device": device,
        "dtype": dtype,
        "quantization_mode": os.getenv("GOR_QUANTIZATION", "").strip().lower() or "none",
        "cpu_int8_enabled": _env_flag("GOR_CPU_INT8", default=False),
        "torch_compile_enabled": _env_flag("GOR_TORCH_COMPILE", default=False),
    }


def _run_generation_once(
    prompt: str,
    cfg: GenerationConfig,
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    strict_retry: bool,
) -> tuple[str, Any, dict[str, Any], int]:
    text_prompt = _build_step_prompt(prompt, strict_retry=strict_retry)
    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "output_scores": True,
        "return_dict_in_generate": True,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if cfg.do_sample:
        generate_kwargs["temperature"] = cfg.temperature
        generate_kwargs["top_p"] = cfg.top_p
        if cfg.seed is not None:
            generator = torch.Generator(device=device.type)
            generator.manual_seed(int(cfg.seed))
            generate_kwargs["generator"] = generator

    with torch.inference_mode():
        output = model.generate(**generate_kwargs)

    full_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    generated_text = full_text[len(text_prompt) :] if full_text.startswith(text_prompt) else full_text
    prompt_token_count = int(inputs["input_ids"].shape[-1])
    return generated_text, output, inputs, prompt_token_count


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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _configure_runtime() -> None:
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return

    num_threads = _env_int("GOR_NUM_THREADS")
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    num_interop_threads = _env_int("GOR_NUM_INTEROP_THREADS")
    if num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError:
            # PyTorch may reject runtime changes if an interop pool already exists.
            pass

    _RUNTIME_CONFIGURED = True


def _build_quantization_config(device: torch.device) -> Any | None:
    quantization_mode = os.getenv("GOR_QUANTIZATION", "").strip().lower()
    if not quantization_mode:
        return None

    if quantization_mode not in {"4bit", "8bit", "int4", "int8"}:
        raise ValueError(
            "GOR_QUANTIZATION must be one of: 4bit, 8bit, int4, int8."
        )
    if device.type != "cuda":
        raise RuntimeError("GOR_QUANTIZATION currently requires a CUDA device.")

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "bitsandbytes quantization requested but not available. "
            "Install optional dependency with: poetry install -E quantization"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=quantization_mode in {"4bit", "int4"},
        load_in_8bit=quantization_mode in {"8bit", "int8"},
    )


def _maybe_apply_cpu_int8_quantization(model: Any, device: torch.device) -> Any:
    if device.type != "cpu" or not _env_flag("GOR_CPU_INT8", default=False):
        return model

    try:
        supported = [
            str(engine).lower()
            for engine in getattr(torch.backends.quantized, "supported_engines", [])
            if str(engine).lower() != "none"
        ]
    except Exception:
        supported = []
    if not supported:
        LOGGER.warning(
            "GOR_CPU_INT8 requested but no CPU quantization engine is available; using default FP32 inference."
        )
        return model

    current_engine = str(getattr(torch.backends.quantized, "engine", "none")).lower()
    if current_engine not in supported:
        for preferred in ("qnnpack", "fbgemm", "onednn", "x86"):
            if preferred in supported:
                try:
                    torch.backends.quantized.engine = preferred
                    current_engine = preferred
                    break
                except Exception:
                    continue

    if current_engine not in supported:
        LOGGER.warning(
            "GOR_CPU_INT8 requested but no usable quantization engine found (supported=%s); using FP32.",
            ",".join(supported),
        )
        return model

    try:
        LOGGER.info("Applying CPU dynamic int8 quantization (engine=%s).", current_engine)
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as exc:
        if _env_flag("GOR_CPU_INT8_STRICT", default=False):
            raise RuntimeError("Failed to apply CPU dynamic int8 quantization.") from exc
        LOGGER.warning(
            "CPU int8 quantization failed (%s). Falling back to FP32. "
            "Set GOR_CPU_INT8_STRICT=1 to fail hard.",
            str(exc),
        )
        return model


def _maybe_compile_model(model: Any) -> Any:
    if not _env_flag("GOR_TORCH_COMPILE", default=False):
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    mode = os.getenv("GOR_TORCH_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    try:
        return compile_fn(model, mode=mode)
    except Exception:
        return model


def load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    _configure_runtime()
    LOGGER.info("Loading tokenizer for model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device, dtype = _resolve_device_and_dtype()
    quantization_config = _build_quantization_config(device)
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    LOGGER.info("Loading model weights: %s (device=%s, dtype=%s)", model_name, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if quantization_config is None:
        model.to(device)
    model = _maybe_apply_cpu_int8_quantization(model, device)
    model = _maybe_compile_model(model)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
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
    generated_text = ""
    output = None
    prompt_token_count = 0
    retry_count = 0
    format_issue = None
    for attempt_idx in range(max(0, int(cfg.max_format_retries)) + 1):
        generated_text, output, inputs, prompt_token_count = _run_generation_once(
            prompt,
            cfg,
            model=model,
            tokenizer=tokenizer,
            device=device,
            strict_retry=attempt_idx > 0,
        )
        format_issue = _placeholder_retry_reason(generated_text)
        retry_count = attempt_idx
        if not (cfg.retry_on_placeholder_output and format_issue and attempt_idx < int(cfg.max_format_retries)):
            break

    if output is None:
        raise RuntimeError("Model.generate returned no output.")

    boundaries = find_step_boundaries(generated_text)
    steps = split_steps(generated_text)
    entropy_summary = summarize_token_entropies(output.scores)

    trace: dict[str, Any] = {
        "prompt": prompt,
        "model_name": cfg.model_name,
        "model_metadata": collect_model_metadata(cfg.model_name, model, tokenizer),
        "generated_text": generated_text,
        "step_texts": steps,
        "step_boundaries": boundaries,
        "token_count": int(output.sequences.shape[-1]),
        "entropy_summary": entropy_summary,
        "format_retry_count": int(retry_count),
        "format_retry_issue": format_issue or "",
    }

    layer_completion_hidden: dict[str, dict[str, Any]] = {}
    if cfg.collect_token_embeddings or cfg.collect_step_signals:
        with torch.inference_mode():
            sequence = output.sequences.to(device)
            attention_mask = torch.ones_like(sequence, device=device)
            forward = model(
                input_ids=sequence,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = list(forward.hidden_states)
        total_layers = max(0, len(hidden_states) - 1)
        captured_layers = _resolve_capture_layers(total_layers, cfg.capture_layer_names)
        trace["captured_layers"] = [
            {"layer_name": layer_name, "layer_index": int(layer_index)}
            for layer_name, layer_index in captured_layers
        ]
        for layer_name, layer_index in captured_layers:
            hidden_idx = total_layers if layer_index < 0 else int(layer_index)
            hidden_idx = min(max(1, hidden_idx), total_layers)
            completion_hidden = hidden_states[hidden_idx][0][prompt_token_count:].detach().cpu().to(torch.float32)
            layer_completion_hidden[layer_name] = {
                "layer_index": int(hidden_idx),
                "embeddings": completion_hidden,
            }

    late_payload = layer_completion_hidden.get("late")
    if late_payload is None and layer_completion_hidden:
        late_payload = next(iter(layer_completion_hidden.values()))

    if cfg.collect_token_embeddings and late_payload is not None:
        trace["token_embeddings"] = late_payload["embeddings"].tolist()
        trace["token_embeddings_by_layer"] = {
            layer_name: {
                "layer_index": int(payload["layer_index"]),
                "embeddings": payload["embeddings"].tolist(),
            }
            for layer_name, payload in layer_completion_hidden.items()
        }

    if cfg.collect_step_signals and late_payload is not None:
        late_hidden = late_payload["embeddings"]
        token_entropies = [float(entropy_from_logits(step_logits).mean().item()) for step_logits in output.scores]
        token_spans = estimate_step_token_spans(
            generated_text=generated_text,
            step_boundaries=boundaries,
            tokenizer=tokenizer,
            n_completion_tokens=int(late_hidden.shape[0]),
        )
        step_rows_by_layer: dict[str, list[dict[str, Any]]] = {}
        for layer_name, payload in layer_completion_hidden.items():
            step_rows: list[dict[str, Any]] = []
            layer_hidden = payload["embeddings"]
            for span in token_spans:
                start = int(span["start_token"])
                end = int(span["end_token"])
                if end <= start:
                    continue
                emb_chunk = layer_hidden[start:end]
                entropy_chunk = token_entropies[start:end]
                if emb_chunk.shape[0] == 0 or not entropy_chunk:
                    continue
                step_rows.append(
                    {
                        "layer_name": layer_name,
                        "layer_index": int(payload["layer_index"]),
                        "step_index": int(span["step_index"]),
                        "start_token": start,
                        "end_token": end,
                        "entropy_mean": float(sum(entropy_chunk) / len(entropy_chunk)),
                        "embedding_mean": emb_chunk.mean(dim=0).tolist(),
                    }
                )
            step_rows_by_layer[layer_name] = step_rows
        trace["step_signal_rows_by_layer"] = step_rows_by_layer
        trace["step_signal_rows"] = step_rows_by_layer.get("late") or next(
            iter(step_rows_by_layer.values()),
            [],
        )
        trace["token_entropy"] = token_entropies

    return trace
