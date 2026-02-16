# AGENTS.md

This file defines how Codex agents should work in this repository.

The project is a JAX/Flax implementation of the core Dreamer4 training pipeline, originally validated on a toy visual-control domain and now being extended toward a robotics-inspired **2.5D visual grasping** benchmark. Agents working here should optimize for correctness, reproducibility, and honest research framing.

## Project Identity

Treat this repository as:

- a **core-mechanism Dreamer4 reproduction**
- a **toy-to-grasping validation project**
- a **research portfolio artifact**, not a finished benchmark reproduction

Do not describe the project as:

- a full Dreamer4 paper reproduction
- a Minecraft-scale implementation
- a complete replica of all paper engineering details
- a real robot manipulation system

Preferred phrasing:

- "core Dreamer4 mechanisms in JAX"
- "Dreamer4-style world model on toy domains and robotics-inspired grasping"
- "mechanism-level reproduction with imagination RL"
- "robotics-inspired 2.5D visual grasping extension"

## Top Priorities

When making changes, prioritize in this order:

1. Preserve algorithmic correctness.
2. Preserve the Dreamer-style indexing conventions.
3. Preserve reproducibility and checkpoint compatibility where practical.
4. Improve clarity, experiment hygiene, and research usefulness.
5. Avoid overstating results or implementation completeness.

## Repository Map

- `dreamer/models.py`
  - Core model definitions.
  - Token layouts, routing, tokenizer, dynamics, task embeddings, and policy/value/reward heads.
- `dreamer/data.py`
  - Current toy environment and synthetic dataset generation.
  - This is the reference for transition indexing conventions.
- `dreamer/grasping_env.py`
  - Robotics-inspired 2.5D top-down grasping environment and demo generator.
  - Preserves the Dreamer batch contract while adding goal-conditioned manipulation tasks.
- `dreamer/envs.py`
  - Environment factory / batch-normalization layer used to keep toy and grasping paths aligned.
- `dreamer/imagination.py`
  - Latent rollout and denoising schedule logic.
  - Changes here can silently affect RL semantics; be conservative.
- `dreamer/sampler.py`
  - Sampling and visualization helpers.
- `dreamer/utils.py`
  - Patchify/unpatchify helpers, checkpointing, and train-state utilities.
- `scripts/train_tokenizer.py`
  - Phase 1: tokenizer pretraining.
- `scripts/train_dynamics.py`
  - Phase 2: shortcut-forcing dynamics training.
- `scripts/train_bc_rew_heads.py`
  - Phase 3: BC/reward finetuning on top of the world model.
- `scripts/train_policy.py`
  - Phase 4: imagination RL with TD-lambda and PMPO-style updates.
- `docs/`
  - Research-facing material, project framing, and result artifacts.

## Current Project Status

As of the current repo state:

- The 4-stage pipeline exists.
- Validation evidence is primarily on the bouncing-square toy setup.
- A new grasping environment path is being wired in, but should be treated as **in progress** until explicitly validated.
- Documentation positions the repo as a research artifact for mechanism-level understanding.
- Richer-environment work is in active development, not complete.

Agents should assume the repo is in an active research state and may contain partially completed experiments or in-progress refactors.

## Non-Negotiable Semantics

### Dreamer-style transition indexing

This repo uses the Dreamer convention:

- timestep `0` contains dummy `a0`, `r0`, `d0`, and initial state `s0`
- action `a_t` happens before state `s_t`
- initial action/reward/done entries must be treated as invalid

Do not "clean up" this convention into a standard RL indexing scheme unless the user explicitly requests a full migration and all affected code is updated together.

Any changes touching:

- reward prediction
- BC targets
- TD-lambda targets
- imagination rollouts
- policy/value alignment

must be checked against this convention.

### Research honesty

Do not add wording that claims:

- paper reproduction
- benchmark parity
- stronger environments already work
- real robot manipulation results

unless there is direct evidence in the repo.

### Toy-domain compatibility

The bouncing-square path should remain working unless the user explicitly asks to replace it.

If adding grasping or other richer benchmarks, do it alongside the toy path rather than by breaking the current data pipeline.

### Grasping-path honesty

The new manipulation benchmark should be described as:

- robotics-inspired
- top-down 2.5D grasping
- deterministic visual manipulation
- mechanism-level world-model evaluation

Do not describe it as:

- a real robot system
- a physics-accurate manipulation benchmark
- a full robot-learning reproduction

## How To Work On This Repo

### Before editing

Always inspect the relevant training stage first.

Examples:

- tokenizer work: inspect `scripts/train_tokenizer.py` and its calls into `dreamer/models.py`
- dynamics work: inspect `scripts/train_dynamics.py`, `dreamer/models.py`, and `dreamer/sampler.py`
- RL work: inspect `scripts/train_policy.py` and `dreamer/imagination.py`
- environment-routing work: inspect `dreamer/envs.py` plus the relevant script entrypoint
- grasping work: inspect `dreamer/grasping_env.py` and then the affected training/eval stage

Do not assume a standard Dreamer implementation shape. This repo has its own training pipeline and abstractions.

### When changing model logic

Be explicit about:

- tensor shapes
- timestep semantics
- whether gradients should or should not flow through imagined features
- whether a change affects checkpoints or training behavior

Prefer small, well-scoped edits with comments only where the logic is easy to misread.

### When changing experiments or docs

Prefer:

- honest scope statements
- explicit evidence tables
- run names, seed logging, and checkpoint paths
- side-by-side baseline vs ablation comparisons
- clear distinction between the stable toy baseline and the in-progress grasping path

## Code Change Guidelines

### Safe changes

Usually safe:

- documentation updates
- logging improvements
- adding experiment templates
- adding non-breaking helper functions
- adding stronger assertions around shapes or conventions
- additive environment-factory wrappers
- additive grasping-specific evaluation metrics

Higher-risk:

- changing the action/reward alignment
- changing imagination rollout semantics
- changing PMPO sign logic
- changing latent packing/unpacking
- changing tokenizer bottleneck shape assumptions
- changing action-vocabulary handling inside the dynamics model
- changing how `task_ids` are threaded through BC / reward / policy stages

For higher-risk changes, inspect nearby code and comments carefully before editing.

### Keep interfaces stable when possible

This repo currently relies on script-local configuration inside `__main__` blocks.

Unless requested otherwise:

- do not convert the whole project to a new config system in one sweep
- do not break existing checkpoint directory assumptions
- do not rename training scripts casually
- prefer lightweight fields like `env_name` and small factory helpers over a wholesale refactor

Incremental wrappers and additive improvements are preferred over large interface churn.

### Avoid speculative refactors

Do not introduce major architectural cleanup just because something could be cleaner.

This repository is a research codebase. Preserve working experimental wiring unless there is a clear user request or a correctness bug.

## Testing And Validation Expectations

For nontrivial changes, validate at the right level.

### Documentation-only changes

Check:

- file presence
- internal links
- consistency with current README positioning

### Lightweight code changes

Run at least one of:

- Python syntax parsing
- import smoke checks
- focused script execution if cheap

### Training-logic changes

Prefer targeted verification such as:

- shape checks
- one-step forward pass
- one train-step smoke test
- metric/logging sanity checks

Do not claim end-to-end training success unless you actually ran it.

If you could not run training, say so clearly.

## Documentation Expectations

When adding or updating docs:

- keep the project framed as a research artifact
- separate "implemented" from "planned"
- distinguish "validated on toy domain" from "grasping path under active validation"
- prefer concise, high-signal writing

Important docs already present:

- `README.md`
- `docs/reproduction_audit.md`
- `docs/research_playbook.md`
- `docs/experiment_log_template.md`
- `docs/coinrun_procgen_plan.md`
- `docs/project_summary.md`
- `docs/talk_track.md`
- `docs/resume_bullets.md`

Agents should update these rather than creating redundant overlapping documents unless a new file is clearly justified.

## Richer Environment Expansion

If the task is to extend the new grasping benchmark:

- keep the current toy path intact
- prefer `dreamer/grasping_env.py` and `dreamer/envs.py` over bloating `dreamer/data.py`
- preserve the current output contract used by training scripts
- keep dummy timestep-0 action/reward semantics
- keep `task_ids` aligned between offline data generation and policy evaluation
- favor deterministic transitions and interpretable failure modes over simulator complexity

Success for the first grasping iteration is:

- end-to-end pipeline compatibility
- coherent visual predictions through approach / grasp / transport / place phases
- grasp-aware evaluation metrics and interpretable failure cases

It is not required to achieve strong final returns immediately.

If the task is to add CoinRun or Procgen support:

- keep the current toy path intact
- prefer a new dataset/environment module rather than overloading `dreamer/data.py`
- preserve the current output contract used by training scripts
- keep dummy timestep-0 action/reward semantics
- get one environment working well before generalizing

Success for the first richer-environment iteration is:

- end-to-end pipeline compatibility
- stable visual predictions
- interpretable failure modes

It is not required to achieve strong final returns immediately.

## Git And Workspace Awareness

This workspace may already contain user changes.

Agents must:

- avoid reverting unrelated edits
- avoid overwriting existing experimental work
- read before editing if a file is already modified
- work with the current state rather than trying to "clean" the tree

If local modifications conflict directly with the requested task, stop and surface the conflict instead of guessing.

## Good Outcomes

A good contribution in this repo usually does one or more of the following:

- makes the implementation easier to trust
- makes an experiment easier to reproduce
- makes an algorithmic detail easier to understand
- preserves honesty while improving presentation
- moves the project one step closer to a stronger benchmark
- strengthens the bridge from toy control to robotics-inspired visual manipulation

## Bad Outcomes

Avoid changes that:

- blur the line between toy validation and paper reproduction
- blur the line between robotics-inspired grasping and real robot manipulation
- silently alter RL semantics
- replace working research code with cleaner but unvalidated abstractions
- break script-level workflows without adding a tested replacement
- describe planned work as completed work
