# Pipeline

## One-command execution

Quick validation:

```bash
python -m dreamer.pipeline run --config configs/profiles/quick_test.yaml
```

Production run:

```bash
python -m dreamer.pipeline run --config configs/profiles/production.yaml
```

## Supported commands

- `run`: full pipeline from tokenizer to report
- `resume`: continue from an existing run directory
- `stage-only`: run one stage only in a new run

Examples:

```bash
python -m dreamer.pipeline stage-only --config configs/profiles/quick_test.yaml --stage dynamics
python -m dreamer.pipeline resume --run-dir runs/dreamer4_grasping_quick_test/20260411-220000 --stage-only report
```

## Stage outputs

- `tokenizer`: tokenizer checkpoints and reconstruction images
- `dynamics`: world-model checkpoints and optional rollout media
- `bc_rew`: BC/reward head checkpoints
- `policy`: policy/value checkpoints and `policy/metrics.jsonl`
- `eval`: normalized metrics into `metrics/metrics.jsonl`
- `report`: final `summary.md`

## Troubleshooting

- OOM: reduce `B`, `T`, or `horizon`, then rerun with `quick_test`.
- Missing checkpoint path: use `resume` or specify explicit `*_ckpt` in stage config.
- Stale script references: always use `python -m dreamer.pipeline ...` instead of legacy ad-hoc shell commands.

