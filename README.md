# Core Dreamer4 Mechanisms in JAX for Toy Control and 2.5D Grasping

This repository is an unofficial JAX implementation of the core training pipeline introduced in [Dreamer4](https://danijar.com/project/dreamer4/), focused on understanding and validating the method on a small synthetic control problem and then extending it toward a robotics-inspired 2.5D visual grasping benchmark.

It is best described as:

- a **core Dreamer4 mechanism reproduction in JAX**
- a **dual-path research artifact** with a stable toy baseline and an in-progress grasping path
- a **research portfolio project** for world models and imagination-based RL

It is not a claim of full paper reproduction on Minecraft-scale data, and it should not be described as a real robot manipulation system.

![dreamer4](docs/architecture.png)

## What This Repo Covers

The current codebase implements the main Dreamer4-style ingredients:

- **Causal tokenizer** trained with MAE-style masking and MSE/LPIPS reconstruction
- **Interactive dynamics model** with shortcut forcing, x-prediction, and ramp loss weighting
- **Agent/task tokens** plus behavior cloning and reward modeling heads
- **Imagination training** with TD-lambda returns, PMPO-style policy updates, and KL regularization to a BC prior
- **Environment-aware training/eval routing** for both the toy and grasping paths
- **Task-conditioned manipulation batches** for the new goal-conditioned grasping benchmark

The repository now has two environment paths:

- `bouncing_square`: the stable reference baseline used to validate the full 4-stage pipeline
- `grasping_2p5d`: a robotics-inspired top-down manipulation environment with task IDs and a larger discrete action space

The grasping path should be presented honestly as an in-progress benchmark extension for mechanism-level evaluation, not as a full manipulation benchmark reproduction.

## Current Status

What is already in place:

- End-to-end 4-stage training pipeline under `scripts/`
- Visualizations and example training curves under `docs/`
- World model, policy, value, and reward heads in pure JAX/Flax
- JIT-friendly imagination rollout utilities
- `dreamer/envs.py` environment factory helpers shared across toy and grasping setups
- `dreamer/grasping_env.py` for deterministic top-down 2.5D grasping demos and evaluation

What is still missing for a paper-level reproduction:

- Large-scale offline video datasets
- Minecraft-scale experiments
- Unlabeled-video plus small action-conditioned-data training regime
- Full paper engineering details such as RoPE, GQA, KV caching, and complete loss RMS normalization
- Fully validated stronger-environment results beyond the toy baseline

The practical takeaway is: this repo is already a strong starting point for studying Dreamer4-style world models, with a validated toy baseline and a promising grasping extension that is still under active validation.

## Repo Structure

- `dreamer/`
  - `models.py`: tokenizer, dynamics model, token routing, policy/value/reward heads
  - `data.py`: bouncing-square dataset and toy environment
  - `envs.py`: shared environment factory and batch-normalization helpers
  - `grasping_env.py`: robotics-inspired 2.5D top-down grasping environment
  - `imagination.py`: latent rollout and denoising schedule utilities
  - `sampler.py`: sampling helpers and visualization utilities
  - `utils.py`: checkpointing, state helpers, patchify helpers
- `scripts/`
  - `train_tokenizer.py`
  - `train_dynamics.py`
  - `train_bc_rew_heads.py`
  - `train_policy.py`
  - `eval_bc_rew_heads.py`
- `docs/`
  - figures, training curves, paper text extracts, and research-facing docs added in this pass

## Setup

We use `uv` for environment management:

```bash
uv sync
uv pip install -e .
```

By default this installs CPU JAX. For GPU, install the appropriate JAX build manually, for example:

```bash
uv pip install "jax[cuda12]"
```

## Run Pipeline

Use the unified pipeline entrypoint instead of manually editing script `__main__` blocks:

```bash
python -m dreamer.pipeline run --config configs/profiles/quick_test.yaml
python -m dreamer.pipeline run --config configs/profiles/production.yaml
python -m dreamer.pipeline run --config configs/profiles/production_policy_recover.yaml
```

Posthoc visualization (no retraining):

```bash
python -m dreamer.pipeline visualize --run-dir runs/<experiment>/<timestamp> --stage dynamics --ckpt latest
python -m dreamer.pipeline visualize --run-dir runs/<experiment>/<timestamp> --stage bc_rew --ckpt latest
python -m dreamer.pipeline visualize --run-dir runs/<experiment>/<timestamp> --stage policy --ckpt latest
```

Pipeline stages:

1. `tokenizer`
2. `dynamics`
3. `bc_rew`
4. `policy`
5. `eval`
6. `report`

All runs are saved under `runs/<experiment_name>/<timestamp>/` with:

- `config_resolved.yaml`
- `manifest.json`
- `summary.md`
- stage folders with checkpoints/media/`metrics.jsonl`/`curves/*.png`/`best_checkpoint.json`
- run-level `metrics/dashboard.png` and `metrics/kpi_table.png`

Policy evaluation media and diagnostics:

- `policy/viz/real_env_eval_stepXXXXXX.mp4`
- `policy/viz/real_env_eval_strip_stepXXXXXX_b*.png`
- `policy/viz/real_env_eval_manifest_stepXXXXXX.json`
- `policy/metrics.jsonl` includes grasp diagnostics (`close_count_mean`, `grasp_attempt_count_mean`, `attached_ratio_mean`, etc.)

Dynamics/BC-reward visualization artifacts:

- `dynamics/viz/step_XXXXXX/<tag>_grid.mp4` and `dynamics/viz/step_XXXXXX/eval_manifest.json`
- `bc_rew/viz/step_XXXXXX/<tag>_grid.mp4` and `bc_rew/viz/step_XXXXXX/eval_manifest.json`
- if MP4 codec is unavailable, fallback PNG frame folders are written and recorded in the manifest

Tokenizer foreground modeling knobs (`configs/base.yaml -> stages.tokenizer`):

- `foreground_weight_enabled`
- `foreground_weight_alpha`
- `foreground_rgb_tolerance`
- `foreground_min_patch_ratio`

Training logs now include:

- `tokenizer/loss_mse_fg_weighted`
- `tokenizer/fg_patch_ratio`

## Run on Colab

Recommended runtime:

- GPU runtime enabled
- High-RAM session when available
- Persistent Drive mount if you want robust resume across session resets

Minimal setup sequence:

```bash
git clone https://github.com/Irayshon/Dreamer4-jax.git
cd Dreamer4-jax
pip install uv
uv sync
uv pip install -e .
```

Colab dependency sync (recommended to avoid JAX plugin mismatch):

```bash
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
pip install -r requirements-colab.txt
```

Pipeline commands:

```bash
python -m dreamer.pipeline run --config configs/profiles/quick_test.yaml --output-root /content/drive/MyDrive/Dreamer4Runs
python -m dreamer.pipeline run --config configs/profiles/production.yaml --output-root /content/drive/MyDrive/Dreamer4Runs
python -m dreamer.pipeline run --config configs/profiles/production_policy_recover.yaml --output-root /content/drive/MyDrive/Dreamer4Runs
python -m dreamer.pipeline visualize --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp> --stage dynamics --ckpt latest
python -m dreamer.pipeline visualize --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp> --stage bc_rew --ckpt latest
python -m dreamer.pipeline visualize --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp> --stage policy --ckpt latest
```

With `--output-root` pointing to Drive, checkpoint saves are persisted immediately in the mounted directory.

If the session disconnects, resume from an existing run directory:

```bash
python -m dreamer.pipeline resume --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp>
```

To quickly validate environment/paths without launching full training:

```bash
python -m dreamer.pipeline stage-only --config configs/profiles/quick_test.yaml --stage report
```

Colab notebook template (cell-by-cell):

```python
# Cell 1: clone + setup
!git clone https://github.com/Irayshon/Dreamer4-jax.git
%cd Dreamer4-jax
from google.colab import drive
drive.mount("/content/drive")
!pip install uv
!uv sync
!uv pip install -e .
!pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
!pip install -r requirements-colab.txt
```

```python
# Cell 2: quick smoke check
!python -m dreamer.pipeline stage-only --config configs/profiles/quick_test.yaml --stage report
```

```python
# Cell 3: quick end-to-end run
!python -m dreamer.pipeline run --config configs/profiles/quick_test.yaml --output-root /content/drive/MyDrive/Dreamer4Runs
```

```python
# Cell 4: resume an interrupted run
!python -m dreamer.pipeline resume --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp>
```

```python
# Cell 5: production run
!python -m dreamer.pipeline run --config configs/profiles/production.yaml --output-root /content/drive/MyDrive/Dreamer4Runs
```

```python
# Cell 6: posthoc visualization from existing checkpoints
!python -m dreamer.pipeline visualize --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp> --stage dynamics --ckpt latest
!python -m dreamer.pipeline visualize --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp> --stage bc_rew --ckpt latest
!python -m dreamer.pipeline visualize --run-dir /content/drive/MyDrive/Dreamer4Runs/<experiment_name>/<timestamp> --stage policy --ckpt latest
```

## Run on Local Machine

Minimal local setup:

```bash
uv sync
uv pip install -e .
uv pip install "jax[cuda12]"
```

If you hit JAX/plugin mismatch locally, force the pinned set:

```bash
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
pip install -r requirements-local-gpu.txt
```

Recommended workflow:

```bash
python -m dreamer.pipeline run --config configs/profiles/quick_test.yaml
python -m dreamer.pipeline run --config configs/profiles/production.yaml
```

Common troubleshooting:

- CUDA/JAX mismatch: reinstall JAX with the CUDA build matching your driver/toolkit.
- GPU not detected: verify `jax.devices()` includes a CUDA device.
- Missing `jaxlpips`: run `pip install jaxlpips` or install from `requirements-colab.txt` / `requirements-local-gpu.txt`.
- OOM: reduce `B`, `T`, or `horizon` in profile YAML (prefer reducing `B` first).
- `make_iterator() missing ... channels` or tuple-unpack mismatches: use `python -m dreamer.pipeline visualize ...` instead of old ad-hoc eval scripts; pipeline path uses current `dreamer.envs.make_iterator` + `unpack_batch`.
- `Dynamics.__init__() got unexpected keyword argument 'action_dim'`: do not instantiate `Dynamics` manually with stale kwargs; current code uses `n_actions=action_dim+1`.
- Orbax shape mismatch (for example `Requested shape ... stored shape ...`): run posthoc visualization from the original run directory so stage metadata (`meta.cfg`) drives the restore shape.

## Estimated Runtime by GPU

Assumptions for estimates:

- single GPU
- default `quick_test` and `production` profiles
- no major I/O bottleneck
- no repeated restarts

Stage share guidance for debugging bottlenecks:

- tokenizer: 15-20%
- dynamics: 25-30%
- bc_rew: 20-25%
- policy: 30-40%

Estimated end-to-end wall-time (tokenizer + dynamics + bc_rew + policy + eval/report):

| GPU | quick_test | production |
|---|---:|---:|
| Colab T4 (16GB) | 6-12h | 10-18d |
| Colab L4 (24GB) | 3-6h | 5-9d |
| Colab A100 (40GB) | 1.5-3h | 2.5-4.5d |
| RTX 3060 (12GB) | 7-14h | Not recommended at default config (downscale needed) |
| RTX 3090 (24GB) | 2.5-5h | 4-7d |
| RTX 4090 (24GB) | 2-4h | 3.5-6d |

## Cost/Time Notes

- These are end-to-end wall-time ranges, not strict SLA numbers.
- Actual runtime can vary with driver/JAX version, logging/media I/O, and background load.
- `production` on Colab typically spans multiple sessions; use `resume` as the default workflow.
- 12GB-class GPUs may require reduced batch/time settings for `production`.

Primary docs:

- [Architecture](docs/architecture.md)
- [Pipeline](docs/pipeline.md)
- [Experiments](docs/experiments.md)
- [Reproduction Audit](docs/reproduction_audit.md)

## Expected Acceptance Signals

The current acceptance signals are split by path:

- Toy path:
  - Tokenizer: good masked reconstruction and strong PSNR
  - Dynamics: accurate autoregressive rollouts and stable shortcut sampling
  - BC/Reward stage: decreasing action/reward losses and sensible readouts
  - Policy stage: improved imagined return and real toy-environment return

- Grasping path:
  - End-to-end batch compatibility through all 4 training stages
  - Correct action-vocabulary and `task_ids` wiring
  - Coherent visual predictions across approach / grasp / transport / place phases
  - Interpretable grasp metrics such as grasp success and placement success

Representative outputs already checked into `docs/`:

- tokenizer reconstruction snapshot: `docs/step_75900.png`
- dynamics curves: `docs/dynamics_training.png`
- BC/reward curves: `docs/bc_rew_training.png`
- policy curves: `docs/rl_training.png`
- imagination/video demos: `docs/imagination-cropped.gif`, `docs/rl-cropped.gif`

## Honest Positioning

If you use this repo in applications, the safest framing is:

> Implemented and analyzed the core Dreamer4 training pipeline in JAX, validated on a toy visual-control domain and extended toward a robotics-inspired 2.5D grasping benchmark, including MAE tokenizer pretraining, shortcut-forcing dynamics, BC/reward finetuning, and imagination-based RL with PMPO-style updates.

That framing is both strong and accurate.

## References

- Dreamer4: [Training Agents Inside of Scalable World Models](https://danijar.com/project/dreamer4/)
- Jasmine: [A simple, performant and scalable JAX-based world modeling codebase](https://github.com/p-doom/jasmine)
