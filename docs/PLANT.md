# PLANT.md

This file captures the current implementation direction for Codex work.

## Current Build Focus

The repository has been refactored to a pipeline-first architecture:

- unified orchestration in `dreamer.pipeline`
- YAML-driven configuration in `configs/`
- standardized run artifacts under `runs/`

Main benchmark path:

- `grasping_2p5d`

Regression compatibility path:

- `bouncing_square`

## What Is Implemented

- Unified commands: `run`, `resume`, `stage-only`
- Stage chain: `tokenizer -> dynamics -> bc_rew -> policy -> eval -> report`
- Config overlays:
  - `configs/base.yaml`
  - `configs/env/grasping_2p5d.yaml`
  - `configs/profiles/quick_test.yaml`
  - `configs/profiles/production.yaml`
- Standard metrics/report surface for policy evaluation

## Operating Rules

- Prefer pipeline commands over direct script invocation.
- Keep algorithm behavior unchanged unless explicitly requested.
- Treat config and docs as part of the code contract.
- Keep old checkpoint paths usable via compatibility resolution.

## Immediate Next Priorities

1. Keep stage runners consistently parameterized by config.
2. Expand lightweight tests for pipeline resume and stage dependency checks.
3. Maintain clear and honest docs for quick vs production runs.
4. Avoid reintroducing script-local hardcoded environment paths.
5. Keep README Colab cell template aligned with actual CLI/config interface.
