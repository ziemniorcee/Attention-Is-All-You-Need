# Attention Is All You Need — tiny reproduction

Minimal project for reproducing a controlled Transformer experiment inspired by
Vaswani et al. (2017), including a comparison with an LSTM and an ablation over
the number of attention heads.

## Planned experiment

- Task: reversing synthetic token sequences.
- Main comparison: tiny Transformer vs LSTM.
- Ablation: 1, 2, 4, and 8 attention heads at fixed `d_model`.
- Metrics: token accuracy, exact-sequence accuracy, parameter count, and runtime.
- Repetitions: three fixed random seeds for the final configurations.

## Repository layout

```text
configs/              experiment configuration
src/tiny_transformer/ training and analysis code
tests/                small correctness and smoke tests
report/               scientific report source
slides/               presentation outline/source
data/                 downloaded/generated data (ignored by Git)
artifacts/             results, checkpoints, tables, figures (ignored by Git)
```

## Target command

Once implemented, the complete experiment should run with:

```bash
python -m tiny_transformer.train --config configs/base.yaml
```

The exact environment and reproduction instructions will be frozen before the
final experiment runs.

## Paper

Vaswani et al., *Attention Is All You Need*, NeurIPS 2017,
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

