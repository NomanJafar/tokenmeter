# tokenmeter

Per-tenant, per-model token accounting for LLM serving: $ per million tokens, MFU tracking and a cost dashboard.

Project 12 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Per-tenant, per-model token accounting with $ per million tokens and MFU tracking on a dashboard.

## Why it matters

Inference is a unit-economics game. Whoever measures it wins.

## Status

Primer, proxy, cost math, dashboard and the Kaggle notebook are done; the GPU run is next.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/tokenmeter/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
