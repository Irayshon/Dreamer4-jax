# Experiment Log Template

Use this file as the single running ledger for all reproducibility and ablation runs.

## Baseline Pipeline

| Date | Stage | Script | Seed | Run name | Checkpoint path | Main metrics | Artifacts | Notes |
|---|---|---|---|---|---|---|---|---|
|  | Tokenizer | `scripts/train_tokenizer.py` | 0 |  |  | `psnr=`, `lpips=` |  |  |
|  | Dynamics | `scripts/train_dynamics.py` | 0 |  |  | `rollout_psnr=`, `shortcut_psnr=` |  |  |
|  | BC/Reward | `scripts/train_bc_rew_heads.py` | 0 |  |  | `pi_ce=`, `rw_ce=`, `rw_mae=` |  |  |
|  | Policy | `scripts/train_policy.py` | 0 |  |  | `imag_return=`, `real_return=` |  |  |

## Ablations

| Date | Ablation | Config change | Seed | Run name | Main metrics | Comparison vs baseline | Interpretation |
|---|---|---|---|---|---|---|---|
|  | No bootstrap | disable shortcut bootstrap branch | 0 |  |  |  |  |
|  | No ramp | uniform signal weighting | 0 |  |  |  |  |
|  | No PMPO | BC-only or no imagination RL | 0 |  |  |  |  |

## Artifact Checklist

- Reconstruction figure saved
- Dynamics rollout figure or video saved
- BC/reward training curve saved
- Policy training curve saved
- Real environment evaluation media saved
- All checkpoint paths copied into this log
