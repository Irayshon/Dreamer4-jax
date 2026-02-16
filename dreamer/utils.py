"""Dreamer4 JAX 训练与检查点工具函数。

本文件仅包含“胶水层”能力：
- 时序 patch/unpatch 的批处理封装；
- 潜变量空间与瓶颈空间之间的重排；
- Orbax 检查点保存/恢复的统一入口。

这些函数不承载算法创新，目标是让训练脚本保持简洁与可维护。
"""

import jax
import jax.numpy as jnp
from dreamer.data import patchify, unpatchify
import orbax.checkpoint as ocp
from pathlib import Path
from flax.core import freeze, unfreeze, FrozenDict
from einops import rearrange


# --- 基础张量变换工具 ---
temporal_patchify = jax.jit(
    jax.vmap(patchify, in_axes=(1, None), out_axes=1),  # (B,T,H,W,C) -> (B,T,Np,Dp)
    static_argnames=("patch",),
)

temporal_unpatchify = jax.jit(
    jax.vmap(unpatchify, in_axes=(1, None, None, None, None), out_axes=1),
    static_argnames=("H", "W", "C", "patch"),
)

def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
    Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
    """
    return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)

def unpack_spatial_to_bottleneck(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,S_z, D_z_pre) -> (B,T,N_b,D_b) by splitting D_z_pre into k channels along N_b.
    Requires: N_b == n_spatial * k  (e.g., 256 -> 512 with k=2).
    """
    return rearrange(z_btLd, 'b t n_spatial (k d) -> b t (n_spatial k) d', n_spatial=n_spatial, k=k)

# -------- 检查点工具 --------
def with_params(variables, new_params):
    """将变量树中的 params 替换为 new_params。

    兼容 FrozenDict 与普通 dict，返回类型始终为 FrozenDict。
    """
    d = unfreeze(variables) if isinstance(variables, FrozenDict) else dict(variables)
    d["params"] = new_params
    return freeze(d)

def pack_mae_params(enc_vars, dec_vars):
    """打包编码器/解码器参数，便于用单一优化器联合优化。"""
    return FrozenDict({
        "enc": enc_vars["params"],
        "dec": dec_vars["params"],
    })

def unpack_mae_params(packed_params, enc_vars, dec_vars):
    """从打包参数中解包并回填到 enc_vars/dec_vars。"""
    enc_vars = with_params(enc_vars, packed_params["enc"])
    dec_vars = with_params(dec_vars, packed_params["dec"])
    return enc_vars, dec_vars


def make_state(params, opt_state, rng, step):
    """构造可被 JAX/Orbax 安全序列化的训练状态树。"""
    return {
        "params": params,
        "opt_state": opt_state,
        "rng": rng,
        "step": jnp.int32(step),
    }

def make_manager(ckpt_dir: str, max_to_keep: int = 5, save_interval_steps: int = 1000, item_names=("state","meta")):
    """创建 Orbax CheckpointManager，并确保目录存在。"""
    path = Path(ckpt_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep,
                                           save_interval_steps=save_interval_steps)
    # item_names 用于恢复时的属性访问：restored.state / restored.meta
    mngr = ocp.CheckpointManager(path, options=options, item_names=item_names)
    return mngr

def try_restore(mngr: ocp.CheckpointManager, state_example: dict, meta_example: dict | None = None):
    """按当前 state 示例构建抽象树，并安全恢复最新检查点。"""
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)
    restore_args = ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),
        meta=ocp.args.JsonRestore() if meta_example is not None else None
    )
    latest = mngr.latest_step()
    if latest is None:
        return None
    restored = mngr.restore(latest, args=restore_args)
    return latest, restored

def maybe_save(mngr: ocp.CheckpointManager, step: int, state: dict, meta: dict | None = None):
    """按 manager 的保存策略触发异步保存。"""
    if not mngr.should_save(step):  # obey save interval policy
        return
    save_args = ocp.args.Composite(
        state=ocp.args.StandardSave(state),
        meta=ocp.args.JsonSave(meta) if meta is not None else None
    )
    mngr.save(step, args=save_args)  # 默认异步保存。
