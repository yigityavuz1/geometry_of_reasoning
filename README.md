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
poetry run python scripts/run_phase2_labeling.py --help
poetry run python scripts/run_phase3_synthetic_validation.py --help
poetry run python scripts/run_ablation.py --help
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

## Phase 2 Labeling Run (Step-Level Table)

```bash
GOR_DEVICE=cpu poetry run python scripts/run_phase2_labeling.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split test \
  --start-index 0 \
  --num-samples 2 \
  --max-new-tokens 128 \
  --out-table-jsonl data/processed/gsm8k_step_labels.jsonl \
  --out-table-csv data/processed/gsm8k_step_labels.csv \
  --out-summary data/processed/gsm8k_step_labels_summary.json
```

## Phase 3 Metric Validation

```bash
poetry run python scripts/run_metrics.py \
  --embeddings results/smoke_qwen05_cpu/token_embeddings.npy \
  --k 10 \
  --k-values 5,10,20,40 \
  --out results/smoke_qwen05_cpu/metrics_summary_phase3.json
```

```bash
poetry run python scripts/run_phase3_synthetic_validation.py \
  --out results/phase3/synthetic_validation.json \
  --intrinsic-dims 4,8,12 \
  --k-values 5,10,20,40
```

## Phase 4 Ablation (A/B/C)

```bash
GOR_DEVICE=cpu poetry run python scripts/run_ablation.py \
  --experiment ALL \
  --models Qwen/Qwen2.5-0.5B-Instruct \
  --split test \
  --start-index 0 \
  --num-samples 3 \
  --max-new-tokens 192 \
  --k-values 5,10,20,40 \
  --primary-k 10 \
  --early-n 2 \
  --out results/ablation
```
