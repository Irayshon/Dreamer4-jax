# AGENTS.md

This file defines how coding agents should work in this repository after the pipeline refactor.

## Project Identity

Treat this repository as:

- a Dreamer4 mechanism-level implementation in JAX
- a pipeline-first research codebase for reproducible experiments
- primarily focused on `grasping_2p5d`, with `bouncing_square` kept for regression checks

Do not present it as:

- a full Dreamer4 paper reproduction
- a real robot manipulation stack

## Canonical Entrypoint

Use the unified pipeline entrypoint, not ad-hoc script execution:

```bash
python -m dreamer.pipeline run --config configs/profiles/quick_test.yaml
python -m dreamer.pipeline run --config configs/profiles/production.yaml
```

Supported commands:

- `run`
- `resume`
- `stage-only`

## Source of Truth

Primary architecture and usage docs:

- `docs/architecture.md`
- `docs/pipeline.md`
- `docs/experiments.md`

For notebook users, keep the Colab cell template in `README.md` up to date whenever pipeline commands or config paths change.

Historical planning docs were moved to `docs/archive/`.

## Configuration Rules

- All stage behavior should come from YAML config under `configs/`.
- Avoid introducing new hardcoded training paths in script `__main__` blocks.
- If adding config fields, update:
  - `configs/base.yaml`
  - profile overlays in `configs/profiles/`
  - docs that describe run profiles

## Artifact Contract

Every run must write to:

`runs/<experiment_name>/<timestamp>/`

with:

- `config_resolved.yaml`
- `manifest.json`
- `summary.md`
- stage outputs and checkpoints

Do not break this contract without updating pipeline and docs together.

## Change Priorities

When making changes, prioritize:

1. pipeline correctness and reproducibility
2. checkpoint compatibility (including legacy `logs/<name>/checkpoints` references)
3. stable stage interfaces
4. documentation accuracy

## Safe vs Risky Changes

Usually safe:

- docs updates
- config profile tuning
- pipeline metadata/reporting improvements
- additive tests

Risky (require extra validation):

- changes to Dreamer timestep semantics
- dynamics/imagination target alignment
- action vocabulary / task ID semantics
- checkpoint schema changes
