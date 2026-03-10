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
  - `configs/profiles/production_policy_recover.yaml`
- Standard metrics/report surface for policy evaluation
- Policy eval media + diagnostics:
  - `policy/viz/real_env_eval_stepXXXXXX.mp4`
  - `policy/viz/real_env_eval_strip_stepXXXXXX_b*.png`
  - `policy/viz/real_env_eval_manifest_stepXXXXXX.json`
  - `policy/metrics.jsonl` with grasp diagnostics (`close/lower/lift`, `grasp_attempt`, `attached_ratio`, `goal_chase_while_unattached`)
- Video fallback path: if MP4 encoding fails, tiled PNG frames are written and marked in manifest.

## Operating Rules

- Prefer pipeline commands over direct script invocation.
- Keep algorithm behavior unchanged unless explicitly requested.
- Treat config and docs as part of the code contract.
- Keep old checkpoint paths usable via compatibility resolution.

## Immediate Next Priorities

1. Continue policy-stage recovery on existing production runs before restarting full-chain training.
2. Track acceptance metrics for recovery (`grasp_success_rate >= 0.2`, `place_success_rate >= 0.1`).
3. Keep eval media enabled in production/recovery profiles to support visual debugging.
4. Maintain clear docs for resume flow and profile selection (quick/production/recover).
5. Avoid reintroducing script-local hardcoded environment paths.
