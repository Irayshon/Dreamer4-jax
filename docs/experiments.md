# Experiments

This document defines the official experiment profiles for single-GPU work.

## Profile: `quick_test`

Goal: verify full pipeline wiring and produce sanity metrics with low cost.

- tokenizer: `B=8`, `T=32`, `max_steps=3_000`, `lpips_frac=0.25`
- dynamics: `B=8`, `T=32`, `max_steps=5_000`, `bootstrap_start=500`, `write_video_every=0`
- bc_rew: `B=8`, `T=32`, `max_steps=5_000`, `L=2`
- policy: `B=8`, `T=32`, `horizon=16`, `context_length=8`, `max_steps=10_000`, `eval_every=2_000`, `eval_episodes=2`

## Profile: `production`

Goal: stable, reproducible results on one 24GB GPU.

- tokenizer: `B=24`, `T=64`, `max_steps=120_000`, `lpips_frac=0.5`
- dynamics: `B=20`, `T=64`, `max_steps=180_000`, `bootstrap_start=10_000`, `self_fraction=0.25`
- bc_rew: `B=20`, `T=64`, `max_steps=140_000`, `L=2`, `loss_weight_shortcut=1.0`, `loss_weight_policy=0.3`, `loss_weight_reward=0.3`
- policy: `B=16`, `T=64`, `horizon=32`, `context_length=16`, `max_steps=220_000`, `eval_every=10_000`, `eval_episodes=8`

## Standard metrics

Final report tracks these metrics:

- `return_mean`
- `return_std`
- `grasp_success_rate`
- `place_success_rate`
- `attach_steps_mean`
- `final_goal_distance_mean`

## Minimal reproducibility checklist

- Use fixed config from `configs/profiles/*.yaml`
- Keep `config_resolved.yaml` with run artifacts
- Keep `manifest.json` and `summary.md`
- Avoid editing stage scripts mid-run

