# CoinRun / Procgen Upgrade Plan

This is the recommended path for upgrading the current project from a toy mechanism validation into a stronger research portfolio artifact.

## Why CoinRun or Procgen

They are a good middle ground between bouncing square and Minecraft:

- visually richer than the toy environment
- action-conditioned from pixels
- easy to explain in interviews
- much cheaper than a full large-scale offline game dataset

## Design Goal

Preserve the existing 4-stage pipeline:

1. tokenizer pretraining
2. dynamics training
3. BC/reward finetuning
4. imagination RL

The environment upgrade should primarily affect:

- dataset generation / loading
- action encoding
- reward/task definitions
- evaluation scripts

## Recommended Implementation Order

### 1. Add a new data path without breaking the toy path

Keep `dreamer/data.py` working for bouncing square, and add a separate module for the stronger benchmark.

Suggested split:

- `dreamer/data.py`: toy environment stays untouched
- `dreamer/procgen_data.py`: new loader / wrappers for CoinRun or Procgen

### 2. Standardize the dataset output contract

Match the current training scripts as closely as possible:

- `frames`: `(B, T, H, W, C)`
- `actions`: `(B, T)` or compatible categorical format
- `rewards`: `(B, T)`
- optional `task_ids`: `(B,)`

Keep the Dreamer-style indexing convention where timestep `0` contains dummy action/reward entries.

### 3. Start with one task only

Do not start with multi-task Procgen.

Pick one:

- CoinRun
- a single Procgen game

The first success criterion is simply that the pipeline runs end to end and produces stable visual predictions.

### 4. Match image size to current model budget first

Before scaling model size, resize observations to a manageable resolution that works with the current tokenizer and patching setup.

Priorities:

- preserve enough structure for world modeling
- keep batch sizes feasible
- avoid architecture changes until the data path is stable

### 5. Define success stage by stage

Tokenizer:

- reconstructions preserve object positions and level layout

Dynamics:

- rollouts respect agent motion and scene continuity

BC / Reward:

- action prediction and reward prediction improve steadily

Policy:

- imagined training produces measurable improvement over BC-only behavior

## Minimal Interview-Ready Outcome

The upgrade is successful enough for internship applications once you can say:

- the same Dreamer4-style JAX stack works on both a toy domain and a richer visual-control benchmark
- you measured at least one transfer failure mode and one success case
- you can explain what broke first when moving beyond the toy environment

That story is already research-worthy even before chasing strong final returns.
