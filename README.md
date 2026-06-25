# Attention Is All You Need - tiny reproduction

Minimal project for reproducing a controlled Transformer experiment inspired by
Vaswani et al. (2017). The project compares a tiny encoder-decoder Transformer
with an LSTM baseline and studies the effect of the number of attention heads.

## Experiment

- Task: reversing synthetic token sequences.
- Main comparison: tiny Transformer vs LSTM baseline.
- Ablation: 1, 2, 4, and 8 attention heads at fixed `d_model`.
- Metrics: token accuracy, exact-sequence accuracy, parameter count, and runtime.
- In-distribution lengths: 5-20 tokens.
- Generalization lengths: 21-40 tokens.
- Final seeds: 13, 37, and 71.

The final raw results are stored in:

```text
results/raw_runs.csv
```

This file contains 15 completed runs:

- LSTM: 3 seeds.
- Transformer with 1 head: 3 seeds.
- Transformer with 2 heads: 3 seeds.
- Transformer with 4 heads: 3 seeds.
- Transformer with 8 heads: 3 seeds.

## Repository layout

```text
configs/              experiment configuration
src/tiny_transformer/ training, evaluation, models, and analysis code
tests/                correctness and smoke tests
report/               scientific report source
slides/               presentation outline/source
results/              raw results and generated analysis outputs
artifacts/            checkpoints, histories, and metadata ignored by Git
```

## Setup

Create and activate a virtual environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

On Windows PowerShell, a typical local setup is:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run tests

```bash
python -m pytest
```

The test suite checks model shapes, masks, parameter counts, and a small training
smoke test.

## Reproduce analysis from existing results

The expensive training sweep has already been completed. To reproduce the final
tables and figures from `results/raw_runs.csv`, run:

```bash
python -m tiny_transformer.analysis
```

The command writes analysis artifacts to:

```text
results/analysis/
```

Generated outputs include:

```text
summary_by_config.csv
summary_by_config.md
summary_table_report.csv
key_findings.md
test_accuracy_by_config.png
heads_ablation_test_token_accuracy.png
heads_ablation_test_sequence_accuracy.png
generalization_accuracy_by_config.png
training_time_by_config.png
```

## Interactive terminal demo

To type your own token sequence and compare answers from the LSTM baseline and
Transformers with 1, 2, 4, and 8 attention heads, run:

```bash
python -m tiny_transformer.demo --config configs/base.yaml --seed 13
```

Then enter numbers from `3` to `31`, for example:

```text
3 7 9 12
```

The demo looks for trained checkpoints in the normal training layout, such as
`artifacts/lstm_heads-na_seed-13/best.pt` and
`artifacts/transformer_heads-8_seed-13/best.pt`. If checkpoints are missing, it
still runs and labels those model answers as untrained.

## Re-run the full experiment

To run or resume the full training sweep, use:

```bash
python -m tiny_transformer.train --config configs/base.yaml --resume
```

Compact final metrics are written to `results/raw_runs.csv`. Larger checkpoints,
per-epoch histories, configuration snapshots, software versions, and hardware
metadata are stored under the Git-ignored `artifacts/` directory.

Re-running the full sweep can take a long time on CPU. For report reproduction,
the recommended path is to use the existing `results/raw_runs.csv` file and run
the analysis script.

## Main result

The best Transformer configuration uses 8 attention heads. It improves clearly
over the 1-head Transformer, supporting the multi-head attention ablation.

However, in this small reverse-sequence experiment, the LSTM baseline achieves
the best overall test accuracy. All models fail to generalize exact full
sequences to longer lengths, with 0.0% exact-sequence accuracy on the
generalization split.

## Paper

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in
Neural Information Processing Systems, 30.

arXiv: https://arxiv.org/abs/1706.03762
