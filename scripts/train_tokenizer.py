"""Dreamer4 JAX tokenizer stage runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import time

import imageio
import jax
import jax.numpy as jnp
import optax
from jaxlpips import LPIPS

from dreamer.envs import get_env_spec, make_iterator, unpack_batch
from dreamer.models import Decoder, Encoder
from dreamer.utils import (
    make_manager,
    make_state,
    maybe_save,
    pack_mae_params,
    temporal_patchify,
    temporal_unpatchify,
    try_restore,
    unpack_mae_params,
)


@dataclass(frozen=True)
class TokenizerConfig:
    run_name: str = "tokenizer"
    log_dir: str = "./logs"
    env_name: str = "grasping_2p5d"
    use_wandb: bool = False

    B: int = 8
    T: int = 32
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2
    size_min: int = 6
    size_max: int = 14
    hold_min: int = 4
    hold_max: int = 9
    diversify_data: bool = True

    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dec: int = 64
    n_heads: int = 4
    enc_depth: int = 8
    dec_depth: int = 8

    max_steps: int = 3000
    log_every: int = 100
    write_image_every: int = 1000
    ckpt_save_every: int = 1000
    ckpt_max_to_keep: int = 5
    lr: float = 1e-4
    lpips_weight: float = 0.2
    lpips_frac: float = 0.25


def init_models(rng, encoder, decoder, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)
    enc_vars = encoder.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        patch_tokens,
        deterministic=True,
    )
    fake_z = jnp.zeros((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_z,
        deterministic=True,
    )
    return rng, enc_vars, dec_vars


def forward_apply(encoder, decoder, enc_vars, dec_vars, patches_btnd, *, mae_key, drop_key, train: bool):
    rngs_enc = {"mae": mae_key} if not train else {"mae": mae_key, "dropout": drop_key}
    z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd, rngs=rngs_enc, deterministic=not train)

    rngs_dec = {} if not train else {"dropout": drop_key}
    pred_btnd = decoder.apply(dec_vars, z_btLd, rngs=rngs_dec, deterministic=not train)
    return pred_btnd, mae_info


def recon_loss_from_mae(pred_btnd, patches_btnd, mae_mask):
    masked_pred = jnp.where(mae_mask, pred_btnd, 0.0)
    masked_target = jnp.where(mae_mask, patches_btnd, 0.0)
    num = jnp.maximum(mae_mask.sum(), 1)
    return jnp.sum((masked_pred - masked_target) ** 2) / (num * pred_btnd.shape[-1])


lpips_loss_fn = None


def lpips_on_mae_recon(pred, target, mae_mask, *, H, W, C, patch, subsample_frac: float = 1.0):
    recon_masked_btnd = jnp.where(mae_mask, pred, target)
    recon_imgs = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    target_imgs = temporal_unpatchify(target, H, W, C, patch)

    if subsample_frac < 1.0:
        T = recon_imgs.shape[1]
        step = max(1, int(1.0 / subsample_frac))
        idx = jnp.arange(T)[::step]
        recon_imgs = recon_imgs[:, idx]
        target_imgs = target_imgs[:, idx]

    recon_lp = jnp.clip(recon_imgs * 2.0 - 1.0, -1.0, 1.0)
    target_lp = jnp.clip(target_imgs * 2.0 - 1.0, -1.0, 1.0)
    BT = recon_lp.shape[0] * recon_lp.shape[1]
    H_, W_, C_ = recon_lp.shape[2], recon_lp.shape[3], recon_lp.shape[4]
    recon_lp = recon_lp.reshape((BT, H_, W_, C_))
    target_lp = target_lp.reshape((BT, H_, W_, C_))
    lp = lpips_loss_fn(recon_lp, target_lp)
    return jnp.mean(lp)


@partial(jax.jit, static_argnames=("encoder", "decoder", "patch"))
def viz_step(encoder, decoder, enc_vars, dec_vars, batch, *, patch, mae_key, drop_key):
    patches_btnd = temporal_patchify(batch, patch)
    pred_btnd, (mae_mask_btNp1, keep_prob_bt1) = forward_apply(
        encoder,
        decoder,
        enc_vars,
        dec_vars,
        patches_btnd,
        mae_key=mae_key,
        drop_key=drop_key,
        train=False,
    )

    masked_input_btnd = jnp.where(mae_mask_btNp1, 0.0, patches_btnd)
    recon_masked_btnd = jnp.where(mae_mask_btNp1, pred_btnd, patches_btnd)
    recon_full_btnd = pred_btnd
    return {
        "target": patches_btnd,
        "masked_input": masked_input_btnd,
        "recon_masked": recon_masked_btnd,
        "recon_full": recon_full_btnd,
        "keep_prob": keep_prob_bt1,
    }


@partial(jax.jit, static_argnames=("encoder", "decoder", "tx", "patch", "H", "W", "C", "lpips_weight", "lpips_frac"))
def train_step(
    encoder,
    decoder,
    tx,
    params,
    opt_state,
    enc_vars,
    dec_vars,
    batch,
    *,
    patch,
    H,
    W,
    C,
    master_key,
    step,
    lpips_weight=0.2,
    lpips_frac=1.0,
):
    patches_btnd = temporal_patchify(batch, patch)
    step_key = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    def loss_fn(packed_params):
        ev, dv = unpack_mae_params(packed_params, enc_vars, dec_vars)
        pred, mae_info = forward_apply(
            encoder,
            decoder,
            ev,
            dv,
            patches_btnd,
            mae_key=mae_key,
            drop_key=drop_key,
            train=True,
        )
        mae_mask, keep_prob = mae_info
        mse = recon_loss_from_mae(pred, patches_btnd, mae_mask)
        if lpips_weight > 0.0:
            lpips = lpips_on_mae_recon(pred, patches_btnd, mae_mask, H=H, W=W, C=C, patch=patch, subsample_frac=lpips_frac)
            total = mse + lpips_weight * lpips
        else:
            lpips = 0.0
            total = mse
        aux = {
            "loss_total": total,
            "loss_mse": mse,
            "loss_lpips": lpips,
            "keep_prob": keep_prob,
        }
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    del loss
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_enc_vars, new_dec_vars = unpack_mae_params(new_params, enc_vars, dec_vars)
    return new_params, opt_state, new_enc_vars, new_dec_vars, aux


def run(cfg: TokenizerConfig) -> dict[str, float]:
    global lpips_loss_fn

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir = log_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl_path = run_dir / "metrics.jsonl"

    env_spec = get_env_spec(cfg.env_name)
    if env_spec.name == "bouncing_square":
        iterator = make_iterator(
            cfg.env_name,
            cfg.B,
            cfg.T,
            cfg.H,
            cfg.W,
            cfg.C,
            pixels_per_step=cfg.pixels_per_step,
            size_min=cfg.size_min,
            size_max=cfg.size_max,
            hold_min=cfg.hold_min,
            hold_max=cfg.hold_max,
        )
    else:
        iterator = make_iterator(
            cfg.env_name,
            cfg.B,
            cfg.T,
            cfg.H,
            cfg.W,
            cfg.C,
            pixels_per_step=cfg.pixels_per_step,
        )

    def next_batch(rng):
        rng, batch = iterator(rng)
        videos, actions, rewards, task_ids = unpack_batch(batch, batch_size=cfg.B)
        del actions, rewards, task_ids
        return rng, videos

    rng = jax.random.PRNGKey(0)
    rng, first_batch = next_batch(rng)

    num_patches = (cfg.H // cfg.patch) * (cfg.W // cfg.patch)
    d_patch = cfg.patch * cfg.patch * cfg.C
    encoder = Encoder(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        depth=cfg.enc_depth,
        dropout=0.05,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0,
        mae_p_max=0.9,
        time_every=4,
    )
    decoder = Decoder(
        d_model=cfg.d_model_dec,
        n_heads=cfg.n_heads,
        n_patches=num_patches,
        n_latents=cfg.enc_n_latents,
        depth=cfg.dec_depth,
        d_patch=d_patch,
        dropout=0.05,
        time_every=4,
    )
    first_patches = temporal_patchify(first_batch, cfg.patch)
    rng, enc_vars, dec_vars = init_models(
        rng,
        encoder,
        decoder,
        first_patches,
        cfg.B,
        cfg.T,
        cfg.enc_n_latents,
        cfg.enc_d_bottleneck,
    )

    params = pack_mae_params(enc_vars, dec_vars)
    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(params)
    if cfg.lpips_weight > 0.0:
        lpips_loss_fn = LPIPS(pretrained_network="alexnet")

    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    state_example = make_state(params, opt_state, rng, step=0)
    meta_example = {
        "enc_kwargs": {
            "n_latents": cfg.enc_n_latents,
            "d_bottleneck": cfg.enc_d_bottleneck,
        },
        "dec_kwargs": {"n_latents": cfg.enc_n_latents},
        "H": cfg.H,
        "W": cfg.W,
        "C": cfg.C,
        "patch": cfg.patch,
        "env_name": cfg.env_name,
        "cfg": asdict(cfg),
    }
    restored = try_restore(mngr, state_example, meta_example)
    start_step = 0
    if restored is not None:
        latest_step, restored_items = restored
        params = restored_items.state["params"]
        opt_state = restored_items.state["opt_state"]
        rng = restored_items.state["rng"]
        start_step = int(restored_items.state["step"]) + 1
        enc_vars, dec_vars = unpack_mae_params(params, enc_vars, dec_vars)
        print(f"[restore] tokenizer resumed at step={latest_step}")

    final_metrics = {
        "tokenizer/loss_total": float("nan"),
        "tokenizer/loss_mse": float("nan"),
        "tokenizer/loss_lpips": float("nan"),
    }
    run_start = time()
    try:
        for step in range(start_step, cfg.max_steps + 1):
            data_start = time()
            rng, batch = next_batch(rng)
            data_time = time() - data_start
            train_start = time()
            rng, master_key = jax.random.split(rng)
            params, opt_state, enc_vars, dec_vars, aux = train_step(
                encoder,
                decoder,
                tx,
                params,
                opt_state,
                enc_vars,
                dec_vars,
                batch,
                patch=cfg.patch,
                H=cfg.H,
                W=cfg.W,
                C=cfg.C,
                master_key=master_key,
                step=step,
                lpips_weight=cfg.lpips_weight,
                lpips_frac=cfg.lpips_frac,
            )
            train_time = time() - train_start

            if (step % cfg.log_every == 0) or (step == cfg.max_steps):
                mse_loss = float(aux["loss_mse"])
                lpips_loss = float(aux["loss_lpips"])
                total_loss = float(aux["loss_total"])
                psnr = float(10 * jnp.log10(1.0 / jnp.maximum(mse_loss, 1e-10)))
                print(
                    f"[train] step={step:06d} | total={total_loss:.6f} | rmse={mse_loss**0.5:.6f} | "
                    f"lpips={lpips_loss:.5f} | psnr={psnr:.4f} | t={data_time + train_time:.3f}s"
                )
                with metrics_jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "stage": "tokenizer",
                                "step": int(step),
                                "elapsed_sec": float(time() - run_start),
                                "step_time_sec": float(data_time + train_time),
                                "loss_total": total_loss,
                                "loss_mse": mse_loss,
                                "loss_lpips": lpips_loss,
                                "rmse": float(mse_loss**0.5),
                                "psnr": psnr,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                final_metrics["tokenizer/loss_total"] = total_loss
                final_metrics["tokenizer/loss_mse"] = mse_loss
                final_metrics["tokenizer/loss_lpips"] = lpips_loss

            state = make_state(params, opt_state, rng, step)
            maybe_save(mngr, step, state, meta_example)

            if cfg.write_image_every and (step % cfg.write_image_every == 0):
                rng, viz_key = jax.random.split(rng)
                mae_key, drop_key, vis_batch_key = jax.random.split(viz_key, 3)
                _, viz_batch = next_batch(vis_batch_key)
                viz_batch = viz_batch[:8, :1]
                out = viz_step(
                    encoder,
                    decoder,
                    enc_vars,
                    dec_vars,
                    viz_batch,
                    patch=cfg.patch,
                    mae_key=mae_key,
                    drop_key=drop_key,
                )
                target = jnp.concatenate(temporal_unpatchify(out["target"], cfg.H, cfg.W, cfg.C, cfg.patch).squeeze(), axis=1)
                masked_input = jnp.concatenate(
                    temporal_unpatchify(out["masked_input"], cfg.H, cfg.W, cfg.C, cfg.patch).squeeze(),
                    axis=1,
                )
                recon_masked = jnp.concatenate(
                    temporal_unpatchify(out["recon_masked"], cfg.H, cfg.W, cfg.C, cfg.patch).squeeze(),
                    axis=1,
                )
                recon_full = jnp.concatenate(
                    temporal_unpatchify(out["recon_full"], cfg.H, cfg.W, cfg.C, cfg.patch).squeeze(),
                    axis=1,
                )
                grid = jnp.concatenate([target, masked_input, recon_masked, recon_full])
                imageio.imwrite(run_dir / f"step_{step:06d}.png", jnp.asarray(grid * 255.0, dtype=jnp.uint8))
    finally:
        mngr.wait_until_finished()

    return final_metrics


if __name__ == "__main__":
    cfg = TokenizerConfig()
    print("Running tokenizer config:\n  " + "\n  ".join([f"{k}={v}" for k, v in asdict(cfg).items()]))
    run(cfg)

