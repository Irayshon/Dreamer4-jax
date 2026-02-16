# Research Playbook

This playbook turns the current repo into a strong research-intern portfolio artifact.

## Goal

Produce evidence that this project is:

- technically correct at the mechanism level
- reproducible on its current domain
- extensible to a stronger visual-control benchmark

## Phase A: Reproducibility Baseline

Run the full 4-stage pipeline with a fixed seed and record:

- script name
- seed
- `run_name`
- checkpoint directory
- wall-clock training time
- final metrics
- output figures/videos

Use a single experiment log table for all runs.

Suggested table columns:

| Stage | Script | Seed | Run name | Checkpoint path | Final metric(s) | Key artifact |
|---|---|---|---|---|---|---|
| Tokenizer | `scripts/train_tokenizer.py` | 0 |  |  | PSNR, LPIPS | reconstruction figure |
| Dynamics | `scripts/train_dynamics.py` | 0 |  |  | rollout PSNR | rollout figure/video |
| BC/Reward | `scripts/train_bc_rew_heads.py` | 0 |  |  | action CE, reward CE/MAE | curve image |
| Policy | `scripts/train_policy.py` | 0 |  |  | imagined return, real env return | RL curve/video |

## Phase B: Minimum Acceptance Metrics

Before scaling the project, confirm the current toy setup behaves sensibly.

Tokenizer:

- masked reconstructions look visually correct
- PSNR reaches a stable high range
- no obvious temporal inconsistency in short reconstructions

Dynamics:

- autoregressive rollout remains stable over long horizons
- shortcut sampling is close to finer-step sampling
- predicted motion respects action conditioning

BC / Reward:

- action CE decreases steadily
- reward predictions track the toy reward structure
- finetuning does not obviously collapse world model behavior

Policy:

- imagined return improves during training
- real toy-environment return improves
- policy remains close enough to the BC prior to avoid degenerate rollouts

## Phase C: Three Required Ablations

Run at least these three:

1. Remove shortcut bootstrap.
   - Keep only the finest-step flow loss.
   - Question answered: does shortcut supervision materially improve fast generation?

2. Remove ramp weighting.
   - Use uniform weighting across signal levels.
   - Question answered: does emphasizing higher-signal regions improve learning quality?

3. Remove PMPO-style imagination RL.
   - Keep BC policy only, or disable RL improvement over BC.
   - Question answered: does imagination RL improve beyond imitation on this task?

For each ablation, record:

- exact config difference
- training stability
- final quantitative metric
- one qualitative artifact
- a short interpretation

## Phase D: Stronger Environment Upgrade

Recommended next target: **CoinRun or Procgen**

Why:

- still visually rich but much smaller in scope than Minecraft
- easier to explain in an interview
- much better evidence than a bouncing-square toy problem

Keep the same 4-stage structure:

1. tokenizer on visual sequences
2. dynamics on latent sequences with actions
3. BC/reward finetuning
4. imagination RL

The goal is not instant state-of-the-art. The goal is to show that the current design transfers to a richer environment with only data/environment changes.

## Deliverables Checklist

- Updated README with honest positioning
- `docs/reproduction_audit.md`
- One experiment table with fixed-seed runs
- Four representative figures, one per stage
- Three ablation summaries
- One stronger environment roadmap or implementation branch
- One-page project summary
- Five-minute talk track

## Suggested Directory Convention

When you start rerunning experiments, keep the outputs structured:

- `logs/tokenizer_seed0_*`
- `logs/dynamics_seed0_*`
- `logs/bc_rew_seed0_*`
- `logs/policy_seed0_*`
- `docs/results/`
- `docs/results/ablations/`

This keeps the repo application-ready and makes it much easier to cite results accurately.
