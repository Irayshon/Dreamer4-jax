# Five-Minute Talk Track

## 1. Project framing

This project is an educational but serious JAX implementation of the core Dreamer4 pipeline. I did not try to claim full paper reproduction on Minecraft. Instead, I focused on implementing the method stack correctly on a toy visual-control problem first, so I could validate the architecture and training loop end to end.

## 2. Why Dreamer4 is interesting

DreamerV3 uses an RSSM, which is lightweight and effective in narrow RL settings, but harder to scale to diverse video distributions. Dreamer4 moves toward a much more scalable world-model recipe by combining a stronger tokenizer, transformer-based dynamics, and shortcut-forcing so that generation can stay both accurate and fast enough for imagination training.

## 3. What I implemented

The repo has four stages.

1. A causal tokenizer trained with MAE-style masking and MSE/LPIPS reconstruction.
2. An action-conditioned dynamics model trained with shortcut forcing in latent space.
3. Agent/task tokens plus behavior-cloning and reward heads.
4. Imagination-based RL with TD-lambda returns, a value head, and a PMPO-style policy loss regularized toward a BC prior.

## 4. Why x-prediction and shortcut forcing matter

The dynamics model is trained to predict clean latent targets rather than directly emitting high-frequency velocity targets everywhere. The shortcut objective then teaches the model to take larger denoising steps consistently, which matters because imagination training needs fast rollout, not just high-quality diffusion at many tiny steps.

## 5. What PMPO is doing here

In the imagination stage, I use imagined rewards and value predictions to compute TD-lambda returns. The policy loss then uses the sign of the advantage rather than its full magnitude, which makes the update more robust. A KL term keeps the learned policy close to the BC prior so imagination training does not drift too far into unrealistic behaviors.

## 6. What is still missing

This is not yet a full Dreamer4 reproduction because it still runs on a toy domain and does not include the paper's full scale, data regime, or all engineering details like KV caching, GQA, or RoPE. The next step is to harden reproducibility, run ablations, and move to a stronger environment like CoinRun or Procgen.
