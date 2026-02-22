# Geometry of Reasoning

Training-free geometric diagnostics for early detection of reasoning derailment in LLM chain-of-thought trajectories.

## Scope

- Step-level reasoning traces on GSM8K
- Hidden-state geometry signals: LID (MLE, TwoNN, ABID), Participation Ratio
- Uncertainty signal: conditional entropy
- Offline symbolic checking with SymPy
- "Reasoning Seismograph" visualization

## Project Layout

```text
src/
  generation/
  metrics/
  evaluation/
  visualization/
configs/
scripts/
tests/
docs/
```

## Environment (Poetry)

```bash
poetry install
poetry shell
```

## First Commands

```bash
poetry run python scripts/run_generation.py --help
poetry run python scripts/run_judge.py --help
poetry run python scripts/run_metrics.py --help
poetry run python scripts/run_smoke_gsm8k_pipeline.py --help
```

## One-Sample End-to-End Smoke Run

```bash
poetry run python scripts/run_smoke_gsm8k_pipeline.py \
  --model sshleifer/tiny-gpt2 \
  --split test \
  --index 0 \
  --out-dir results/smoke_tiny \
  --max-new-tokens 96 \
  --k 10
```

For larger models on Apple Silicon, prefer CPU to avoid MPS temporary-buffer limits:

```bash
GOR_DEVICE=cpu poetry run python scripts/run_smoke_gsm8k_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct
```
