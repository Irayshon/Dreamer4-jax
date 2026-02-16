# Architecture

This repository now follows a pipeline-first structure for reproducible experiments.

## Layering

- `dreamer/core`: model and training-step logic (existing implementations in `dreamer/*.py` and stage scripts).
- `dreamer/env`: environment abstractions and adapters (`dreamer/envs.py`, `dreamer/grasping_env.py`, `dreamer/data.py`).
- `dreamer/pipeline`: orchestration, config loading, run manifests, report generation.
- `dreamer/io`: checkpoint and metrics artifact contracts (`runs/<exp>/<timestamp>/...`).

## Stage Contract

Each train stage is executed by the unified pipeline in this order:

1. `tokenizer`
2. `dynamics`
3. `bc_rew`
4. `policy`
5. `eval`
6. `report`

Every stage records status into `manifest.json` with start/end time and output checkpoint path.

## Artifact Layout

All runs are created under:

`runs/<experiment_name>/<timestamp>/`

with fixed files/folders:

- `config_resolved.yaml`
- `manifest.json`
- `summary.md`
- `checkpoints/`
- `metrics/`
- `media/`
- per-stage folders like `tokenizer/`, `dynamics/`, `bc_rew/`, `policy/`

## Backward Compatibility

The pipeline supports legacy checkpoint references:

- direct checkpoint path
- legacy shorthand resolved as `logs/<name>/checkpoints`

This allows older experiments to be resumed without rewriting all historical paths.

