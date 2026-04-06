from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.runner import GenerationConfig, generate_reasoning_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one reasoning trace generation.")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--prompt", required=True, help="Input math problem")
    parser.add_argument("--out", default="results/generation_trace.json", help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--embeddings-out",
        default="",
        help="Optional .npy path to save completion token embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenerationConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        collect_token_embeddings=bool(args.embeddings_out),
    )
    trace = generate_reasoning_trace(args.prompt, cfg)
    token_embeddings = trace.pop("token_embeddings", None)
    token_embeddings_by_layer = trace.pop("token_embeddings_by_layer", None)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote generation trace to {output_path}")
    model_metadata = trace.get("model_metadata")
    if isinstance(model_metadata, dict):
        metadata_path = output_path.with_name("model_metadata.json")
        metadata_path.write_text(json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote model metadata to {metadata_path}")

    if args.embeddings_out and token_embeddings is not None:
        embeddings = np.asarray(token_embeddings, dtype=np.float32)
        emb_path = Path(args.embeddings_out)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        print(f"Wrote token embeddings to {emb_path}")
        if isinstance(token_embeddings_by_layer, dict):
            for layer_name, payload in token_embeddings_by_layer.items():
                layer_embeddings = np.asarray(payload.get("embeddings", []), dtype=np.float32)
                if layer_embeddings.ndim != 2:
                    continue
                layer_path = emb_path.with_name(f"{emb_path.stem}_{layer_name}{emb_path.suffix}")
                np.save(layer_path, layer_embeddings)
                print(f"Wrote {layer_name} token embeddings to {layer_path}")


if __name__ == "__main__":
    main()
