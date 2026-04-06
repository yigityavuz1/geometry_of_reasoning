# Report Summary

- Primary analysis layer: `late`
- Primary k: `20`
- Experiment A AUROC/AUPRC: `0.816` / `0.630`
- Experiment A lead time mean: `0.670`
- Experiment A alarm-before-error rate: `0.345`

## Experiment B Common Index

- `Qwen/Qwen2.5-1.5B-Instruct`: AUROC `0.853`, AUPRC `0.694`, signal gap `1.870`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`: AUROC `0.785`, AUPRC `0.552`, signal gap `1.603`

## Demo Case

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Layer: `middle`
- Dataset index: `725`
- Lead time: `12.0`
- Selected policy: `trajectory_q68`
