"""2.5D top-down tabletop grasping environment and offline demo generator.

This module adds a robotics-inspired visual manipulation task while preserving
the existing Dreamer-style data contract:

- frames:  (B, T, H, W, C)
- actions: (B, T) with dummy a0
- rewards: (B, T) with dummy r0
- task_ids: (B,) goal/bin identities for task conditioning

The simulator is deliberately lightweight and deterministic. It models a
top-down gripper with planar motion, a discrete height mode, open/close
commands, one movable block, and one target bin.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

# Public action contract used by training scripts.
ACTION_UP = jnp.int32(0)
ACTION_DOWN = jnp.int32(1)
ACTION_LEFT = jnp.int32(2)
ACTION_RIGHT = jnp.int32(3)
ACTION_OPEN = jnp.int32(4)
ACTION_CLOSE = jnp.int32(5)
ACTION_LOWER = jnp.int32(6)
ACTION_LIFT = jnp.int32(7)

ACTION_DIM = 8
NULL_ACTION = jnp.int32(8)
NUM_ACTIONS_WITH_NULL = 9
NUM_TASKS = 4

HEIGHT_CONTACT = jnp.int32(0)
HEIGHT_HOVER = jnp.int32(1)
HEIGHT_LIFTED = jnp.int32(2)
MAX_HEIGHT = HEIGHT_LIFTED


def _goal_centers(height: int, width: int) -> jnp.ndarray:
    margin_y = max(5, height // 6)
    margin_x = max(5, width // 6)
    return jnp.asarray(
        [
            [margin_y, margin_x],
            [margin_y, width - margin_x - 1],
            [height - margin_y - 1, margin_x],
            [height - margin_y - 1, width - margin_x - 1],
        ],
        dtype=jnp.int32,
    )


def _goal_rects(height: int, width: int, radius: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    centers = _goal_centers(height, width)
    mins = jnp.maximum(centers - radius, 0)
    maxs = jnp.minimum(centers + radius + 1, jnp.asarray([height, width], dtype=jnp.int32))
    return mins, maxs


def _paint_rect(
    canvas: jnp.ndarray,
    top_left: jnp.ndarray,
    bottom_right: jnp.ndarray,
    color: jnp.ndarray,
) -> jnp.ndarray:
    """Paint axis-aligned rectangles on (B, H, W, C) canvases."""
    B, H, W, _ = canvas.shape
    ys = jnp.arange(H)[None, :, None]
    xs = jnp.arange(W)[None, None, :]
    y0 = top_left[:, 0][:, None, None]
    x0 = top_left[:, 1][:, None, None]
    y1 = bottom_right[:, 0][:, None, None]
    x1 = bottom_right[:, 1][:, None, None]
    mask = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
    return jnp.where(mask[..., None], color[:, None, None, :], canvas)


def _paint_disc(
    canvas: jnp.ndarray,
    centers: jnp.ndarray,
    radius: jnp.ndarray,
    color: jnp.ndarray,
) -> jnp.ndarray:
    B, H, W, _ = canvas.shape
    ys = jnp.arange(H)[None, :, None].astype(jnp.float32)
    xs = jnp.arange(W)[None, None, :].astype(jnp.float32)
    cy = centers[:, 0][:, None, None].astype(jnp.float32)
    cx = centers[:, 1][:, None, None].astype(jnp.float32)
    rr = radius[:, None, None].astype(jnp.float32)
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= rr**2
    return jnp.where(mask[..., None], color[:, None, None, :], canvas)


def _render_observation(
    *,
    gripper_pos: jnp.ndarray,
    gripper_height: jnp.ndarray,
    gripper_closed: jnp.ndarray,
    object_pos: jnp.ndarray,
    attached: jnp.ndarray,
    placed: jnp.ndarray,
    task_id: jnp.ndarray,
    object_color_id: jnp.ndarray,
    height: int,
    width: int,
    channels: int,
) -> jnp.ndarray:
    """Render top-down RGB observations."""
    B = gripper_pos.shape[0]
    del channels

    table_color = jnp.asarray([232, 228, 216], dtype=jnp.uint8)
    table = jnp.broadcast_to(table_color[None, None, None, :], (B, height, width, 3))

    bin_palette = jnp.asarray(
        [
            [220, 82, 82],
            [83, 163, 230],
            [92, 187, 126],
            [224, 170, 72],
        ],
        dtype=jnp.uint8,
    )
    object_palette = jnp.asarray(
        [
            [56, 87, 205],
            [214, 97, 74],
            [95, 168, 113],
            [150, 100, 198],
        ],
        dtype=jnp.uint8,
    )

    # Paint all bins as light context, then highlight the active target.
    goal_mins, goal_maxs = _goal_rects(height, width, radius=max(3, min(height, width) // 10))
    bin_bg = ((bin_palette.astype(jnp.float32) + 255.0) / 2.0).astype(jnp.uint8)
    for goal_idx in range(NUM_TASKS):
        table = _paint_rect(
            table,
            jnp.broadcast_to(goal_mins[goal_idx][None, :], (B, 2)),
            jnp.broadcast_to(goal_maxs[goal_idx][None, :], (B, 2)),
            jnp.broadcast_to(bin_bg[goal_idx][None, :], (B, 3)),
        )
    active_bin_color = bin_palette[task_id]
    table = _paint_rect(table, goal_mins[task_id], goal_maxs[task_id], active_bin_color)

    # Paint the object.
    object_size = jnp.full((B,), max(2, min(height, width) // 10), dtype=jnp.int32)
    object_half = object_size // 2
    obj_color = object_palette[object_color_id % object_palette.shape[0]]
    obj_top_left = jnp.maximum(object_pos - object_half[:, None], 0)
    obj_bottom_right = jnp.minimum(object_pos + object_half[:, None] + 1, jnp.asarray([height, width]))
    placed_color = jnp.broadcast_to(jnp.asarray([245, 245, 245], dtype=jnp.uint8)[None, :], (B, 3))
    object_draw_color = jnp.where(placed[:, None], placed_color, obj_color)
    table = _paint_rect(table, obj_top_left, obj_bottom_right, object_draw_color)

    # Subtle shadow indicates height and attachment.
    shadow_radius = 3 + gripper_height
    shadow_color = jnp.where(attached[:, None], jnp.asarray([120, 120, 120], dtype=jnp.uint8), jnp.asarray([170, 170, 170], dtype=jnp.uint8))
    table = _paint_disc(table, gripper_pos, shadow_radius, shadow_color)

    # Paint a cross-shaped gripper with open/closed jaws.
    gripper_color_open = jnp.asarray([38, 38, 42], dtype=jnp.uint8)
    gripper_color_closed = jnp.asarray([20, 20, 24], dtype=jnp.uint8)
    gripper_color = jnp.where(gripper_closed[:, None], gripper_color_closed[None, :], gripper_color_open[None, :])

    arm_half = jnp.full((B, 1), 1, dtype=jnp.int32)
    center_top = jnp.maximum(gripper_pos - arm_half, 0)
    center_bottom = jnp.minimum(gripper_pos + arm_half + 1, jnp.asarray([height, width]))
    table = _paint_rect(table, center_top, center_bottom, gripper_color)

    jaw_gap = jnp.where(gripper_closed, 1, 3).astype(jnp.int32)
    jaw_len = jnp.full((B,), 2, dtype=jnp.int32)
    left_top = jnp.stack([gripper_pos[:, 0] - jaw_len, gripper_pos[:, 1] - jaw_gap], axis=-1)
    left_bot = jnp.stack([gripper_pos[:, 0] + jaw_len + 1, gripper_pos[:, 1] - jaw_gap + 1], axis=-1)
    right_top = jnp.stack([gripper_pos[:, 0] - jaw_len, gripper_pos[:, 1] + jaw_gap], axis=-1)
    right_bot = jnp.stack([gripper_pos[:, 0] + jaw_len + 1, gripper_pos[:, 1] + jaw_gap + 1], axis=-1)
    left_top = jnp.maximum(left_top, 0)
    right_top = jnp.maximum(right_top, 0)
    left_bot = jnp.minimum(left_bot, jnp.asarray([height, width]))
    right_bot = jnp.minimum(right_bot, jnp.asarray([height, width]))
    table = _paint_rect(table, left_top, left_bot, gripper_color)
    table = _paint_rect(table, right_top, right_bot, gripper_color)

    return table.astype(jnp.float32) / 255.0


def _clip_pos(pos: jnp.ndarray, height: int, width: int, margin: int) -> jnp.ndarray:
    lo = jnp.asarray([margin, margin], dtype=jnp.int32)
    hi = jnp.asarray([height - margin - 1, width - margin - 1], dtype=jnp.int32)
    return jnp.clip(pos, lo, hi)


def _goal_center_for_id(task_id: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
    return _goal_centers(height, width)[task_id]


def _goal_reached(object_pos: jnp.ndarray, task_id: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
    goal_center = _goal_center_for_id(task_id, height, width).astype(jnp.float32)
    dist = jnp.linalg.norm(object_pos.astype(jnp.float32) - goal_center, axis=-1)
    return dist <= max(3.0, min(height, width) / 10.0)


def _scripted_policy(
    state: dict[str, jnp.ndarray],
    *,
    height: int,
    width: int,
) -> jnp.ndarray:
    """Simple goal-directed controller used for offline demonstrations."""
    del height, width
    gripper_pos = state["gripper_pos"]
    object_pos = state["object_pos"]
    gripper_height = state["gripper_height"]
    gripper_closed = state["gripper_closed"]
    attached = state["attached"]
    placed = state["placed"]
    task_id = state["task_id"]
    goal_center = _goal_center_for_id(task_id, state["height"], state["width"])

    target_pos = jnp.where(attached[:, None], goal_center, object_pos)
    delta = target_pos - gripper_pos
    abs_delta = jnp.abs(delta)
    aligned = jnp.max(abs_delta, axis=-1) <= 1

    move_axis_y = abs_delta[:, 0] >= abs_delta[:, 1]
    move_up = delta[:, 0] < 0
    move_left = delta[:, 1] < 0
    move_action = jnp.where(
        move_axis_y,
        jnp.where(move_up, ACTION_UP, ACTION_DOWN),
        jnp.where(move_left, ACTION_LEFT, ACTION_RIGHT),
    )

    lower_needed = (~attached) & (~placed) & aligned & (gripper_height > HEIGHT_CONTACT)
    close_needed = (~attached) & (~placed) & aligned & (gripper_height == HEIGHT_CONTACT) & (~gripper_closed)
    lift_needed = attached & (gripper_height < HEIGHT_LIFTED)
    place_aligned = attached & aligned
    lower_for_place = place_aligned & (gripper_height > HEIGHT_CONTACT)
    open_needed = attached & place_aligned & (gripper_height == HEIGHT_CONTACT) & gripper_closed
    move_needed = (~placed) & ~(lower_needed | close_needed | lift_needed | lower_for_place | open_needed)
    retreat_needed = placed & (gripper_height < HEIGHT_LIFTED)

    action = jnp.where(move_needed, move_action, ACTION_OPEN)
    action = jnp.where(lower_needed, ACTION_LOWER, action)
    action = jnp.where(close_needed, ACTION_CLOSE, action)
    action = jnp.where(lift_needed, ACTION_LIFT, action)
    action = jnp.where(lower_for_place, ACTION_LOWER, action)
    action = jnp.where(open_needed, ACTION_OPEN, action)
    action = jnp.where(retreat_needed, ACTION_LIFT, action)
    return action.astype(jnp.int32)


def _transition(
    state: dict[str, jnp.ndarray],
    actions: jnp.ndarray,
    *,
    height: int,
    width: int,
    pixels_per_step: int,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Pure transition function shared by env_step and offline data generation."""
    gripper_pos = state["gripper_pos"]
    object_pos = state["object_pos"]
    gripper_height = state["gripper_height"]
    gripper_closed = state["gripper_closed"]
    attached = state["attached"]
    placed = state["placed"]
    task_id = state["task_id"]

    deltas = jnp.asarray(
        [
            [-pixels_per_step, 0],
            [pixels_per_step, 0],
            [0, -pixels_per_step],
            [0, pixels_per_step],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],
        dtype=jnp.int32,
    )
    gripper_pos_after_move = _clip_pos(gripper_pos + deltas[actions], height, width, margin=max(4, pixels_per_step + 2))

    lower = actions == ACTION_LOWER
    lift = actions == ACTION_LIFT
    height_next = jnp.where(lower, jnp.maximum(gripper_height - 1, HEIGHT_CONTACT), gripper_height)
    height_next = jnp.where(lift, jnp.minimum(height_next + 1, MAX_HEIGHT), height_next)

    close = actions == ACTION_CLOSE
    open_ = actions == ACTION_OPEN
    dist_to_object_before = jnp.linalg.norm(
        gripper_pos.astype(jnp.float32) - object_pos.astype(jnp.float32),
        axis=-1,
    )
    near_object = dist_to_object_before <= max(2.0, pixels_per_step + 1.0)
    successful_grasp = close & (~gripper_closed) & (~attached) & (~placed) & (height_next == HEIGHT_CONTACT) & near_object

    attached_next = attached | successful_grasp
    released = open_ & gripper_closed & attached
    attached_next = attached_next & (~released)
    gripper_closed_next = jnp.where(open_, False, gripper_closed)
    gripper_closed_next = jnp.where(close, True, gripper_closed_next)

    object_pos_after_attach = jnp.where(attached_next[:, None], gripper_pos_after_move, object_pos)
    place_success = released & (height_next == HEIGHT_CONTACT) & _goal_reached(object_pos_after_attach, task_id, height, width)
    placed_next = placed | place_success

    goal_center = _goal_center_for_id(task_id, height, width)
    object_pos_next = jnp.where(place_success[:, None], goal_center, object_pos_after_attach)
    object_pos_next = _clip_pos(object_pos_next, height, width, margin=3)

    dist_to_object_after = jnp.linalg.norm(
        gripper_pos_after_move.astype(jnp.float32) - object_pos_next.astype(jnp.float32),
        axis=-1,
    )
    goal_dist_before = jnp.linalg.norm(
        object_pos.astype(jnp.float32) - goal_center.astype(jnp.float32),
        axis=-1,
    )
    goal_dist_after = jnp.linalg.norm(
        object_pos_next.astype(jnp.float32) - goal_center.astype(jnp.float32),
        axis=-1,
    )

    approach_reward = jnp.where(
        (~attached) & (~placed),
        0.03 * (dist_to_object_before - dist_to_object_after),
        0.0,
    )
    transport_reward = jnp.where(
        attached,
        0.04 * (goal_dist_before - goal_dist_after),
        0.0,
    )
    lift_bonus = jnp.where(attached_next & (gripper_height < HEIGHT_LIFTED) & (height_next == HEIGHT_LIFTED), 0.6, 0.0)
    grasp_bonus = jnp.where(successful_grasp, 1.2, 0.0)
    place_bonus = jnp.where(place_success, 2.5, 0.0)
    invalid_close = close & (gripper_closed | ~near_object | (height_next != HEIGHT_CONTACT))
    invalid_open = open_ & (~gripper_closed)
    invalid_penalty = -0.05 * invalid_close.astype(jnp.float32) - 0.03 * invalid_open.astype(jnp.float32)
    motion_penalty = -0.005 * ((actions <= ACTION_RIGHT) | lower | lift).astype(jnp.float32)
    rewards = approach_reward + transport_reward + lift_bonus + grasp_bonus + place_bonus + invalid_penalty + motion_penalty

    attach_steps_next = state["attach_steps"] + attached_next.astype(jnp.int32)
    grasp_seen_next = state["grasp_seen"] | successful_grasp
    placement_seen_next = state["placement_seen"] | place_success

    next_state = {
        **state,
        "gripper_pos": gripper_pos_after_move,
        "object_pos": object_pos_next,
        "gripper_height": height_next,
        "gripper_closed": gripper_closed_next,
        "attached": attached_next,
        "placed": placed_next,
        "attach_steps": attach_steps_next,
        "grasp_seen": grasp_seen_next,
        "placement_seen": placement_seen_next,
    }
    obs_next = _render_observation(
        gripper_pos=next_state["gripper_pos"],
        gripper_height=next_state["gripper_height"],
        gripper_closed=next_state["gripper_closed"],
        object_pos=next_state["object_pos"],
        attached=next_state["attached"],
        placed=next_state["placed"],
        task_id=next_state["task_id"],
        object_color_id=next_state["object_color_id"],
        height=height,
        width=width,
        channels=3,
    )
    return next_state, obs_next, rewards.astype(jnp.float32)


def env_reset(
    key: jax.Array,
    *,
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 2,
) -> tuple[dict[str, Any], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched environment reset."""
    del channels
    k_task, k_obj, k_grip, k_color = jax.random.split(key, 4)
    object_margin = max(8, min(height, width) // 4)
    low = jnp.asarray([object_margin, object_margin], dtype=jnp.int32)
    high = jnp.asarray([height - object_margin, width - object_margin], dtype=jnp.int32)

    task_id = jax.random.randint(k_task, (batch_size,), 0, NUM_TASKS, dtype=jnp.int32)
    object_pos = jax.random.randint(k_obj, (batch_size, 2), minval=low, maxval=high, dtype=jnp.int32)
    gripper_pos = jax.random.randint(
        k_grip,
        (batch_size, 2),
        minval=jnp.asarray([4, 4], dtype=jnp.int32),
        maxval=jnp.asarray([height - 4, width - 4], dtype=jnp.int32),
        dtype=jnp.int32,
    )
    object_color_id = jax.random.randint(k_color, (batch_size,), 0, 4, dtype=jnp.int32)
    state = {
        "gripper_pos": gripper_pos,
        "object_pos": object_pos,
        "gripper_height": jnp.full((batch_size,), HEIGHT_HOVER, dtype=jnp.int32),
        "gripper_closed": jnp.zeros((batch_size,), dtype=bool),
        "attached": jnp.zeros((batch_size,), dtype=bool),
        "placed": jnp.zeros((batch_size,), dtype=bool),
        "task_id": task_id,
        "goal_center": _goal_center_for_id(task_id, height, width),
        "object_color_id": object_color_id,
        "attach_steps": jnp.zeros((batch_size,), dtype=jnp.int32),
        "grasp_seen": jnp.zeros((batch_size,), dtype=bool),
        "placement_seen": jnp.zeros((batch_size,), dtype=bool),
        "height": height,
        "width": width,
        "channels": 3,
        "pixels_per_step": pixels_per_step,
    }
    obs0 = _render_observation(
        gripper_pos=state["gripper_pos"],
        gripper_height=state["gripper_height"],
        gripper_closed=state["gripper_closed"],
        object_pos=state["object_pos"],
        attached=state["attached"],
        placed=state["placed"],
        task_id=state["task_id"],
        object_color_id=state["object_color_id"],
        height=height,
        width=width,
        channels=3,
    )
    a0 = jnp.full((batch_size,), NULL_ACTION, dtype=jnp.int32)
    r0 = jnp.full((batch_size,), jnp.nan, dtype=jnp.float32)
    return state, obs0, a0, r0


def env_step(
    env_state: dict[str, Any],
    actions: jnp.ndarray,
    *,
    height: int,
    width: int,
    channels: int,
) -> tuple[dict[str, Any], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched environment step."""
    del channels
    next_state, obs_next, rewards = _transition(
        env_state,
        actions,
        height=height,
        width=width,
        pixels_per_step=int(env_state["pixels_per_step"]),
    )
    dones = next_state["placed"]
    return next_state, obs_next, rewards, dones


def make_iterator(
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 2,
):
    """Construct a JIT-friendly offline demo iterator."""
    del channels

    @jax.jit
    def next_batch(key: jax.Array):
        reset_key = key
        env_state, obs0, a0, r0 = env_reset(
            reset_key,
            batch_size=batch_size,
            height=height,
            width=width,
            channels=3,
            pixels_per_step=pixels_per_step,
        )

        def scan_body(state, _):
            action_t = _scripted_policy(state, height=height, width=width)
            state_next, obs_next, reward_t = _transition(
                state,
                action_t,
                height=height,
                width=width,
                pixels_per_step=pixels_per_step,
            )
            return state_next, (obs_next, action_t, reward_t)

        final_state, outputs = jax.lax.scan(
            scan_body,
            env_state,
            xs=jnp.arange(time_steps - 1),
        )
        obs_seq, action_seq, reward_seq = outputs

        frames = jnp.concatenate([obs0[:, None, ...], obs_seq.transpose(1, 0, 2, 3, 4)], axis=1)
        actions = jnp.concatenate([a0[:, None], action_seq.transpose(1, 0)], axis=1)
        rewards = jnp.concatenate([r0[:, None], reward_seq.transpose(1, 0)], axis=1)
        task_ids = final_state["task_id"]
        return key, (frames, actions, rewards, task_ids)

    return next_batch


def make_env_reset_fn(
    *,
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 2,
):
    """Create a partially applied reset function for the policy script."""
    return partial(
        env_reset,
        batch_size=batch_size,
        height=height,
        width=width,
        channels=channels,
        pixels_per_step=pixels_per_step,
    )


def make_env_step_fn(
    *,
    height: int,
    width: int,
    channels: int,
):
    """Create a partially applied step function for the policy script."""
    return partial(env_step, height=height, width=width, channels=channels)
