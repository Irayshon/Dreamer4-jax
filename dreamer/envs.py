"""Shared environment factory helpers for toy and grasping benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from dreamer import data as toy_data
from dreamer import grasping_env


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    action_dim: int
    null_action: int
    n_tasks: int
    supports_task_ids: bool


TOY_SPEC = EnvironmentSpec(
    name="bouncing_square",
    action_dim=4,
    null_action=int(toy_data.NULL_ACTION),
    n_tasks=1,
    supports_task_ids=False,
)

GRASP_SPEC = EnvironmentSpec(
    name="grasping_2p5d",
    action_dim=grasping_env.ACTION_DIM,
    null_action=int(grasping_env.NULL_ACTION),
    n_tasks=grasping_env.NUM_TASKS,
    supports_task_ids=True,
)


def get_env_spec(env_name: str) -> EnvironmentSpec:
    if env_name == "bouncing_square":
        return TOY_SPEC
    if env_name in {"grasping", "grasping_2p5d", "graspworld"}:
        return GRASP_SPEC
    raise ValueError(f"Unknown environment {env_name!r}")


def make_iterator(
    env_name: str,
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    **kwargs: Any,
):
    spec = get_env_spec(env_name)
    if spec.name == "bouncing_square":
        return toy_data.make_iterator(batch_size, time_steps, height, width, channels, **kwargs)
    return grasping_env.make_iterator(batch_size, time_steps, height, width, channels, **kwargs)


def make_env_reset_fn(env_name: str, **kwargs: Any):
    spec = get_env_spec(env_name)
    if spec.name == "bouncing_square":
        return toy_data.make_env_reset_fn(**kwargs)
    return grasping_env.make_env_reset_fn(**kwargs)


def make_env_step_fn(env_name: str, **kwargs: Any):
    spec = get_env_spec(env_name)
    if spec.name == "bouncing_square":
        return toy_data.make_env_step_fn(**kwargs)
    return grasping_env.make_env_step_fn(**kwargs)


def unpack_batch(batch: tuple[Any, ...], *, batch_size: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Normalize 3-tuple and 4-tuple environment batches."""
    if len(batch) == 4:
        frames, actions, rewards, task_ids = batch
        return frames, actions, rewards, task_ids
    if len(batch) == 3:
        frames, actions, rewards = batch
        if batch_size is None:
            batch_size = int(frames.shape[0])
        task_ids = jnp.zeros((batch_size,), dtype=jnp.int32)
        return frames, actions, rewards, task_ids
    raise ValueError(f"Expected a batch tuple of length 3 or 4, got {len(batch)}")
