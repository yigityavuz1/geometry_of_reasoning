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
```
