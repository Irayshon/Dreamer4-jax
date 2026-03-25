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

- Unified commands: `run`, `resume`, `stage-only`, `visualize`
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
- Dynamics + BC/Reward eval media + diagnostics:
  - `dynamics/viz/step_XXXXXX/<tag>_grid.mp4`
  - `dynamics/viz/step_XXXXXX/eval_manifest.json`
  - `bc_rew/viz/step_XXXXXX/<tag>_grid.mp4`
  - `bc_rew/viz/step_XXXXXX/eval_manifest.json`
- Video fallback path: if MP4 encoding fails, tiled PNG frames are written and marked in manifest.
- Tokenizer foreground-weighted reconstruction path for stable object modeling:
  - `foreground_weight_enabled`
  - `foreground_weight_alpha`
  - `foreground_rgb_tolerance`
  - `foreground_min_patch_ratio`
  - logs include `loss_mse_fg_weighted` and `fg_patch_ratio`

## Operating Rules

- Prefer pipeline commands over direct script invocation.
- Keep algorithm behavior unchanged unless explicitly requested.
- Treat config and docs as part of the code contract.
- Keep old checkpoint paths usable via compatibility resolution.
- For posthoc visualization, always recover shape-critical settings from checkpoint metadata and avoid hardcoded legacy assumptions (`make_iterator` signatures, tuple arity, `action_dim`/`n_tasks`).

## Immediate Next Priorities

1. Continue policy-stage recovery on existing production runs before restarting full-chain training.
2. Track acceptance metrics for recovery (`grasp_success_rate >= 0.2`, `place_success_rate >= 0.1`).
3. Validate tokenizer foreground-weighted path on `grasping_2p5d` using `fg_patch_ratio > 0` and improved object visibility in recon grids.
4. Use `pipeline visualize` for `dynamics`/`bc_rew`/`policy` posthoc checks before deciding to retrain any stage.
5. Avoid reintroducing script-local hardcoded environment paths.
