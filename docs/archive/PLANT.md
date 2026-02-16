# PLANT.md

This file captures the current implementation direction for Codex work in this repository.

## Current Build Focus

The repo is evolving from a toy-only Dreamer4 mechanism study into a dual-path research artifact:

- a stable `bouncing_square` baseline path
- a new `grasping_2p5d` robotics-inspired path

The grasping work should be treated as a serious but still in-progress benchmark extension. It supports a stronger MSc-level project narrative around world models for embodied visual control, but it is not a claim of full robot manipulation reproduction.

## What Exists Now

- The original 4-stage Dreamer4-style pipeline still exists.
- The toy `bouncing_square` environment remains the reference baseline.
- `dreamer/grasping_env.py` adds a deterministic top-down grasping-style environment.
- `dreamer/envs.py` adds an environment factory and batch-normalization layer.
- The codebase is moving toward environment-aware action spaces and task-conditioned training.

## Active Rules For Codex

### 1. Preserve the toy path

Do not break the `bouncing_square` workflow while extending grasping support.

That means:

- keep `dreamer/data.py` functioning
- keep existing script entrypoints working
- keep Dreamer indexing semantics unchanged

### 2. Treat grasping as additive and environment-aware

When editing training or evaluation scripts:

- prefer routing through `dreamer/envs.py`
- use `env_name` instead of hard-coding a single backend
- use environment specs for action vocabulary size and task conditioning where possible
- normalize batches through a helper instead of assuming a fixed tuple length

### 3. Respect the new grasping semantics

The grasping path uses:

- a discrete action space larger than the toy environment
- goal-conditioned `task_ids`
- a deterministic tabletop-style simulator rather than a physics engine

If you touch this path, preserve:

- dummy `a0` / `r0`
- batch output shape compatibility with the existing scripts
- honest research framing: robotics-inspired, not full manipulation benchmark

### 4. Prefer small interface wrappers over sweeping rewrites

This repo is still script-driven. Keep improvements incremental:

- environment factory helpers are good
- additive config fields like `env_name` are good
- lightweight metadata additions are good

Avoid:

- introducing a heavy new config system
- rewriting all scripts around a new framework
- removing old experimental knobs unless they are clearly dead or wrong

## Current Research Narrative

The intended public story is:

- Dreamer4-style world-model pipeline in JAX
- validated on a toy control baseline
- extended toward `2.5D` visual grasping as a robotics-inspired benchmark
- focused on mechanism understanding, reproducibility, and interpretable failure modes

Safe phrasing:

- "Dreamer4-style world model in JAX"
- "mechanism-level reproduction with imagination RL"
- "robotics-inspired 2.5D grasping extension"
- "visual manipulation benchmark path"

Avoid:

- "real robot system"
- "full manipulation benchmark reproduction"
- "robotics results" unless actual experiments support the claim

## High-Value Next Steps

The most useful contributions right now are:

1. Finish environment-aware training/eval wiring cleanly.
2. Validate grasping batch shapes, task IDs, and action vocab alignment.
3. Add grasp-specific evaluation metrics and visual diagnostics.
4. Update docs so the repo accurately reflects the new dual-path scope.
5. Add smoke tests that keep both environment paths trustworthy.

## Validation Expectations

For grasping-related code changes, prefer:

- reset/step smoke tests
- scripted rollout sanity checks
- one-batch shape checks through each training stage
- import/syntax checks on modified scripts

Do not claim training success on grasping unless the runs were actually executed.
