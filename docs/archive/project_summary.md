# Project Summary

## Title

Core Dreamer4 Mechanisms in JAX on Toy Visual Control

## Problem

Dreamer4 proposes a scalable world-model agent that learns video prediction, reward prediction, and policy optimization inside imagination. The full paper focuses on large-scale offline visual control, but reproducing that stack directly is expensive and difficult to debug.

This project tackles the problem by implementing the core Dreamer4 training pipeline in JAX on a controlled toy domain first, with the goal of validating the algorithmic structure before scaling to richer environments.

## What I Implemented

- A causal MAE-style tokenizer for video sequences
- A shortcut-forcing latent dynamics model with x-prediction
- Agent/task tokens with policy and reward heads
- Imagination rollouts in latent space
- TD-lambda value learning and PMPO-style policy improvement with KL regularization to a behavior prior

## Why It Matters

This project shows practical understanding of how modern world-model agents are built:

- representation learning from pixels
- action-conditioned generative dynamics
- training-time tradeoffs for fast sampling
- reward/value learning from imagined trajectories
- policy optimization inside a learned model

## Honest Scope

This is not a full Dreamer4 reproduction on Minecraft-scale data.

It is a strong mechanism-level reproduction on a toy domain, designed to answer:

- Is the method wiring correct?
- Do the core training stages work end to end?
- Can the pipeline be scaled to a richer benchmark next?

## Current Evidence

- End-to-end 4-stage training code is present
- Example reconstructions, rollouts, and RL curves are checked into `docs/`
- Static audit shows strong alignment with the Dreamer4 method stack

## Next Milestones

- Re-run the 4-stage pipeline with fixed seeds and recorded outputs
- Add ablations for shortcut bootstrap, ramp weighting, and PMPO
- Port the same pipeline to CoinRun or Procgen

## Interview One-Liner

Implemented the core Dreamer4 world-model pipeline in JAX on a toy visual-control domain, including MAE tokenizer pretraining, shortcut-forcing dynamics, BC/reward finetuning, and imagination-based RL with TD-lambda and PMPO-style updates.
