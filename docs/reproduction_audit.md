# Dreamer4 Reproduction Audit

This document is the source of truth for how this repository relates to the Dreamer4 paper.

## Verdict

This codebase is a **credible implementation of the core Dreamer4 method stack on a toy domain**, not a full paper reproduction.

Use this wording consistently:

- **Implemented**: core Dreamer4 mechanisms
- **Validated on**: bouncing-square toy environment
- **Not yet reproduced**: Minecraft-scale offline experiments and full paper engineering stack

## Mechanism-by-Mechanism Audit

| Paper component | Status in repo | Evidence | Notes |
|---|---|---|---|
| Causal tokenizer | Implemented | `dreamer/models.py`, `scripts/train_tokenizer.py` | Causal encoder/decoder path is present. |
| MAE-style patch masking | Implemented | `dreamer/models.py`, `scripts/train_tokenizer.py` | Random masking with learned replacement token is used. |
| MSE + LPIPS tokenizer loss | Implemented | `scripts/train_tokenizer.py` | Uses weighted combination with `lpips_weight=0.2`. |
| Efficient transformer backbone | Partially implemented | `dreamer/models.py` | Axial/space-time factorization is present; not all paper engineering details are included. |
| Interactive dynamics model | Implemented | `dreamer/models.py`, `scripts/train_dynamics.py` | Action-conditioned latent prediction is present. |
| Shortcut forcing | Implemented | `scripts/train_dynamics.py` | Finest-step flow loss plus coarser-step bootstrap consistency are both present. |
| x-prediction for dynamics | Implemented | `scripts/train_dynamics.py` | Dynamics predicts clean latents and converts to velocity-like quantities only for bootstrap targets. |
| Ramp weighting over signal level | Implemented | `scripts/train_dynamics.py` | Uses `0.9 * sigma + 0.1`. |
| Agent tokens / task embeddings | Implemented | `dreamer/models.py`, `scripts/train_bc_rew_heads.py` | Used in BC/reward stage and imagination stage. |
| BC head | Implemented | `dreamer/models.py`, `scripts/train_bc_rew_heads.py` | Multi-token prediction head exists. |
| Reward head | Implemented | `dreamer/models.py`, `scripts/train_bc_rew_heads.py`, `scripts/train_policy.py` | Symlog/two-hot reward modeling is present. |
| Value head | Implemented | `dreamer/models.py`, `scripts/train_policy.py` | Used for TD-lambda targets. |
| Imagination rollout in latent space | Implemented | `dreamer/imagination.py`, `scripts/train_policy.py` | JIT-friendly rollout path exists. |
| TD-lambda return targets | Implemented | `scripts/train_policy.py` | Reverse scan over imagined rewards/values is present. |
| PMPO-style policy update | Implemented | `scripts/train_policy.py` | Uses sign-only advantage grouping and KL to BC prior. |
| Reverse KL to policy prior | Implemented | `scripts/train_policy.py` | KL term matches the intended regularization direction described in the paper. |
| Running RMS normalization across loss terms | Missing / simplified | paper vs scripts | Current implementation mostly uses fixed weights rather than a full running-RMS loss normalization scheme. |
| Large-scale offline video learning | Missing | dataset layer | Repo uses only the toy environment today. |
| Unlabeled-video plus small action-labeled subset regime | Missing | dataset/training setup | Not yet wired into the current data pipeline. |
| RoPE | Missing | `docs/NOTES.md` roadmap | Mentioned as future work. |
| GQA | Missing | `docs/NOTES.md` roadmap | Mentioned as future work. |
| KV cache for fast generation | Missing | `docs/NOTES.md` roadmap | Mentioned as future work. |
| Minecraft-scale results | Missing | repo scope | Out of scope for the current codebase. |

## Static Concerns To Keep In Mind

These are not blockers to presenting the project, but they should be called out honestly:

- The repo has been **statically audited**, not fully re-run end-to-end during this pass.
- `scripts/train_policy.py` contains PMPO logic whose variable naming/comments are harder to read than the underlying math. The implemented objective appears directionally consistent with the paper, but this section deserves extra care during future verification.
- The strongest evidence today is still on a **toy environment**, so claims should stay at the level of mechanism validation.

## Recommended Public Framing

Good:

- "Core Dreamer4 mechanisms implemented in JAX and validated on a toy visual control task."
- "Educational Dreamer4-style reproduction with tokenizer pretraining, shortcut-forcing dynamics, BC/reward finetuning, and imagination RL."

Avoid:

- "Full Dreamer4 reproduction"
- "Reproduced the Dreamer4 paper"
- "Minecraft-capable Dreamer4 implementation"

## Promotion Criteria For The Next Version

The project can be upgraded from "core-mechanism reproduction" to a stronger research artifact once all of the following are true:

- The 4-stage pipeline has been re-run with fixed seeds and recorded checkpoints.
- Each stage has quantitative metrics and saved figures.
- At least 3 ablations are complete and interpreted.
- One stronger environment beyond bouncing square has been added.
- README and application materials reflect the stronger evidence.
