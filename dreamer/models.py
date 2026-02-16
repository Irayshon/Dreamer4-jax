"""Dreamer4 JAX 核心模型定义。

包含：
- Token 布局与模态路由（图像/动作/寄存器/agent）；
- 编码器、解码器、动力学模型及策略/价值/奖励头；
- Block-causal Transformer 组件。

注：本文件是算法主干，注释重点放在张量形状与信息流方向。
"""

import jax.numpy as jnp
import flax.linen as nn
import jax
import time
from flax.core import FrozenDict
import flax
from enum import IntEnum
from typing import Optional, Tuple, Any
from einops import rearrange
import math

class Modality(IntEnum):
    LATENT   = -1
    IMAGE    = 0
    ACTION   = 1
    PROPRIO  = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7
    # add more as needed

@flax.struct.dataclass
class TokenLayout:
    """
    Describe the token ordering for a single timestep.

    This helper class defines how tokens are arranged inside one timestep.
    The convention is:

        [latent tokens] + [segments in the given order]

    where each segment is a pair `(modality, count)`.

    Attributes:
        n_latents:
            Number of latent tokens placed at the beginning of the sequence.

        segments:
            A tuple of `(modality, count)` pairs describing the remaining token
            groups in order.

    Example:
        >>> layout = TokenLayout(
        ...     n_latents=2,
        ...     segments=((Modality.IMAGE, 3), (Modality.ACTION, 2))
        ... )
        >>> layout.S()
        7
        >>> layout.modality_ids()
        # roughly: [-1, -1, 0, 0, 0, 1, 1]
        >>> layout.slices()[Modality.IMAGE]
        slice(2, 5)

        This means the token order is:
            [LATENT, LATENT, IMAGE, IMAGE, IMAGE, ACTION, ACTION]
    """

    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        """
        Return the total number of tokens in one timestep.

        Returns:
            Total sequence length:
                n_latents + sum(count for each segment)

        Example:
            >>> layout = TokenLayout(
            ...     n_latents=2,
            ...     segments=((Modality.IMAGE, 3), (Modality.ACTION, 2))
            ... )
            >>> layout.S()
            7
        """
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> jnp.ndarray:
        """
        Build a 1D modality-id array for the token sequence.

        The returned array has shape `(S,)`, where each entry stores the modality
        id of the corresponding token position.

        Layout rule:
            - latent tokens come first
            - then each segment contributes `count` repeated modality ids

        Returns:
            A JAX array of shape `(S,)` with dtype `int32`.

        Example:
            >>> layout = TokenLayout(
            ...     n_latents=2,
            ...     segments=((Modality.IMAGE, 3), (Modality.ACTION, 2))
            ... )
            >>> layout.modality_ids()
            # roughly: [-1, -1, 0, 0, 0, 1, 1]

            Meaning:
                positions 0~1 -> LATENT
                positions 2~4 -> IMAGE
                positions 5~6 -> ACTION
        """
        parts = (
            [jnp.full((self.n_latents,), Modality.LATENT, dtype=jnp.int32)]
            if self.n_latents > 0
            else []
        )

        for m, n in self.segments:
            if n > 0:
                parts.append(jnp.full((n,), int(m), dtype=jnp.int32))

        return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.int32)

    def slices(self) -> dict:
        """
        Return slice ranges for the first occurrence of each modality.

        This is a convenience method for indexing token blocks in the sequence.

        Notes:
            - Latent tokens are always placed first.
            - If the same modality appears multiple times in `segments`,
              only the first occurrence is recorded.

        Returns:
            A dictionary:
                {modality: slice(start, stop)}

        Example:
            >>> layout = TokenLayout(
            ...     n_latents=2,
            ...     segments=((Modality.IMAGE, 3), (Modality.ACTION, 2))
            ... )
            >>> layout.slices()
            # {
            #   Modality.LATENT: slice(0, 2),
            #   Modality.IMAGE:  slice(2, 5),
            #   Modality.ACTION: slice(5, 7),
            # }

            Then you can use:
                tokens[..., layout.slices()[Modality.IMAGE], :]
            to extract the IMAGE token block.
        """
        idx = 0
        out = {}

        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents

        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n

        return out

    
def sinusoid_table(
    n: int,
    d: int,
    base: float = 10000.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Create a standard sinusoidal positional-encoding table.

    This function generates the classic Transformer sinusoidal position encoding
    matrix of shape `(n, d)`:

        - even dimensions use sin(...)
        - odd dimensions use cos(...)

    The frequency for dimension pair `(2k, 2k+1)` is scaled by:

        base^(-2k / d)

    Args:
        n:
            Number of positions.

        d:
            Embedding dimension for each position.

        base:
            Frequency base used in the sinusoidal formula.
            Default is 10000.0, which matches the standard Transformer setting.

        dtype:
            Output dtype of the returned JAX array.

    Returns:
        A JAX array of shape `(n, d)` containing the positional encodings.

    Example:
        >>> table = sinusoid_table(n=3, d=4)
        >>> table.shape
        (3, 4)

        The rows correspond to positions 0, 1, 2.
        The columns correspond to embedding dimensions 0, 1, 2, 3.

        Rough structure:
            row 0 -> [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
            row 1 -> [sin(w0), cos(w0), sin(w1), cos(w1)]
            row 2 -> [sin(2*w0), cos(2*w0), sin(2*w1), cos(2*w1)]
    """
    # Position indices: shape (n, 1)
    # Example for n=3:
    #   [[0],
    #    [1],
    #    [2]]
    pos = jnp.arange(n, dtype=dtype)[:, None]

    # Dimension indices: shape (1, d)
    # Example for d=4:
    #   [[0, 1, 2, 3]]
    i = jnp.arange(d, dtype=dtype)[None, :]

    # Pair dimensions by frequency:
    #   dim 0,1 -> k=0
    #   dim 2,3 -> k=1
    #   dim 4,5 -> k=2
    k = jnp.floor(i / 2.0)

    # Frequency scaling term: base^{-2k/d}
    div = jnp.power(
        base,
        -(2.0 * k) / jnp.maximum(1.0, jnp.array(d, dtype))
    )

    # Broadcast multiplication:
    #   pos: (n, 1)
    #   div: (1, d)
    # -> angles: (n, d)
    angles = pos * div

    # Even dimensions use sin, odd dimensions use cos.
    table = jnp.where(
        (i % 2) == 0,
        jnp.sin(angles),
        jnp.cos(angles),
    )

    return table.astype(dtype)


def add_sinusoidal_positions(tokens_btSd: jnp.ndarray) -> jnp.ndarray:
    """
    Add sinusoidal positional encodings to both time and token dimensions.

    Input shape:
        (B, T, S, D)

    Output shape:
        (B, T, S, D)

    The same time-position table `(T, D)` is shared across all batches and
    token slots, and the same token-position table `(S, D)` is shared across
    all batches and timesteps.

    Example:
        If `tokens_btSd.shape == (2, 3, 4, 8)`, this function adds:
            - one time encoding for each of the 3 timesteps
            - one token encoding for each of the 4 token positions
        and returns a tensor of the same shape `(2, 3, 4, 8)`.
    """
    B, T, S, D = tokens_btSd.shape
    pos_t = sinusoid_table(T, D)  # (T, D)
    pos_s = sinusoid_table(S, D)  # (S, D)
    scale = 1.0 / jnp.sqrt(jnp.array(D, dtype=tokens_btSd.dtype))
    return tokens_btSd + scale * (
        pos_t[None, :, None, :] + pos_s[None, None, :, :]
    )

class MAEReplacer(nn.Module):
    """
    Randomly replace patch tokens with a learned mask token, following an
    MAE-style masking strategy.

    For each sample `(b, t)`, this module first draws a masking ratio `p` from
    `[p_min, p_max)`, then computes the keep probability:

        keep_prob = 1 - p

    Next, for each patch position `n`, it samples a Bernoulli keep/drop decision.
    Patch tokens that are dropped are replaced by a learned `mask_token`.

    Input shape:
        patches_btnd: (B, T, Np, D)

    Output:
        replaced:
            Masked patch tokens with the same shape `(B, T, Np, D)`.

        mae_mask:
            Boolean mask of shape `(B, T, Np, 1)`.
            `True` means the original patch was masked/replaced.

        keep_prob_bt1:
            Keep probability for each `(b, t)`, shape `(B, T, 1)`.

    Attributes:
        p_min:
            Minimum masking ratio.

        p_max:
            Maximum masking ratio.

    Example:
        >>> x = jnp.ones((2, 3, 4, 8))   # B=2, T=3, Np=4, D=8
        >>> replacer = MAEReplacer(p_min=0.2, p_max=0.6)
        >>> replaced, mae_mask, keep_prob = replacer(x)

        Possible behavior:
            - For one sample/time pair, the sampled masking ratio may be 0.3
            - Then keep_prob = 0.7
            - Each of the 4 patches is independently kept with probability 0.7
            - Dropped patches are replaced by the learned mask token

        Shapes:
            replaced.shape   == (2, 3, 4, 8)
            mae_mask.shape   == (2, 3, 4, 1)
            keep_prob.shape  == (2, 3, 1)
    """

    p_min: float = 0.0
    p_max: float = 0.9

    @nn.compact
    def __call__(self, patches_btnd: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply random MAE-style masking to patch tokens.

        Args:
            patches_btnd:
                Input patch-token tensor of shape `(B, T, Np, D)`.

        Returns:
            replaced:
                Tensor of shape `(B, T, Np, D)` where masked patches are replaced
                by the learned mask token.

            mae_mask:
                Boolean tensor of shape `(B, T, Np, 1)`.
                `True` indicates that the patch was masked.

            keep_prob_bt1:
                Keep probability tensor of shape `(B, T, 1)`.
        """
        # Input shape:
        #   B  = batch size
        #   T  = number of timesteps
        #   Np = number of patch tokens
        #   D  = patch embedding dimension
        B, T, Np, D = patches_btnd.shape

        # Learned token used to replace masked patches.
        # Shape: (D,)
        mask_token = self.param(
            "mask_token",
            nn.initializers.normal(0.02),
            (D,)
        )

        # Draw RNGs from the named "mae" stream.
        # One RNG is used to sample masking ratios p,
        # the other is used to sample keep/drop decisions.
        rng = self.make_rng("mae")
        p_rng, m_rng = jax.random.split(rng)

        # Sample one masking ratio p for each (batch, timestep).
        # Shape: (B, T)
        p_bt = jax.random.uniform(
            p_rng,
            (B, T),
            minval=self.p_min,
            maxval=self.p_max
        )

        # Convert masking ratio to keep probability.
        # Shape: (B, T, 1)
        keep_prob_bt1 = 1.0 - p_bt[..., None]

        # Sample which patches to keep.
        # True  -> keep original patch
        # False -> replace with mask token
        # Shape before expanding: (B, T, Np)
        keep = jax.random.bernoulli(
            m_rng,
            keep_prob_bt1,
            (B, T, Np)
        )

        # Expand to match patch tensor shape for broadcasting in jnp.where.
        # Shape: (B, T, Np, 1)
        keep = keep[..., None]

        # Replace dropped patches by the learned mask token.
        # mask_token is reshaped to (1, 1, 1, D) so it can broadcast over
        # batch, time, and patch dimensions.
        replaced = jnp.where(
            keep,
            patches_btnd,
            mask_token.reshape(1, 1, 1, D)
        )

        # Boolean mask indicating which patches were replaced.
        # True means "this patch was masked".
        # Shape: (B, T, Np, 1)
        mae_mask = (~keep).astype(jnp.bool_)

        return replaced, mae_mask, keep_prob_bt1


# ---------- small building blocks ----------

class RMSNorm(nn.Module):
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * (scale / jnp.sqrt(var + self.eps))

class MLP(nn.Module):
    """
    Transformer MLP with optional SwiGLU gating.

    Args:
      d_model:     input/output width of the block (last-dim of x).
      mlp_ratio:   expansion factor for hidden size if NOT using 2/3 parity.
      dropout:     dropout rate applied after activation and after output proj.
      swiglu:      if True, use SwiGLU; else standard GELU MLP.
      parity_2over3:
                   if True and swiglu=True, set hidden = (2/3)*mlp_ratio*d_model
                   to roughly match parameter count of a GELU MLP with mlp_ratio.
      dtype:       param/compute dtype.
    """
    d_model: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """
        Args:
          x:            (..., d_model) input activations.
          deterministic:
                        True disables dropout (eval); False enables dropout (train).

        Returns:
          y:            (..., d_model) output activations, same shape as input.
        """
        # Choose hidden size
        mult = self.mlp_ratio
        if self.swiglu and self.parity_2over3:
            mult = self.mlp_ratio * (2.0 / 3.0)  # param parity with GELU MLP

        hidden = int(self.d_model * mult)

        if self.swiglu:
            # SwiGLU: Dense -> split -> u * silu(v)
            pre = nn.Dense(
                2 * hidden, dtype=self.dtype, name="fc_in"
            )(x)  # (..., 2H)
            u, v = jnp.split(pre, 2, axis=-1)     # (..., H), (..., H)
            h = u * jax.nn.silu(v)                # (..., H)
        else:
            # Standard GELU MLP
            h = nn.Dense(hidden, dtype=self.dtype, name="fc_in")(x)
            h = nn.gelu(h)

        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        y = nn.Dense(self.d_model, dtype=self.dtype, name="fc_out")(h)
        y = nn.Dropout(self.dropout)(y, deterministic=deterministic)
        return y
# ---------- axial attention layers ----------
class SpaceSelfAttentionModality(nn.Module):
    """
    Apply self-attention over the spatial/token axis with modality-based routing.

    This module performs self-attention across the `S` tokens inside each
    timestep, while restricting which query tokens are allowed to attend to
    which key tokens.

    The routing rule is determined by:
        - `modality_ids`: modality type of each token position
        - `n_latents`: how many latent tokens are placed at the beginning
        - `mode`: which attention pattern to use

    Input convention:
        x has shape `(B, T, S, D)`, where attention is applied only across `S`
        for each `(B, T)` independently.

    Supported modes:
        "encoder":
            - latent queries attend to all tokens
            - non-latent queries attend only to tokens of the same modality

        "decoder":
            - latent queries attend only to latent tokens
            - non-latent queries attend to:
                (1) tokens of the same modality
                (2) latent tokens

        "wm_agent":
            - AGENT queries attend to all tokens
            - ACTION queries attend only to ACTION tokens
            - observation-type queries attend to observation tokens and ACTION tokens
            - no non-agent query may attend to AGENT keys

        "wm_agent_isolated":
            - AGENT queries attend to nobody
            - ACTION / observation routing is the same as in "wm_agent"
            - no non-agent query may attend to AGENT keys

    Attributes:
        d_model:
            Feature width of each token.

        n_heads:
            Number of attention heads.

        modality_ids:
            A JAX array of shape `(S,)` giving the modality id of each token
            position in one timestep.

        n_latents:
            Number of latent tokens placed at the beginning of the token sequence.

        mode:
            Routing mode used to build the attention mask.

        dropout:
            Dropout rate inside multi-head attention.

    Example:
        >>> layout = TokenLayout(
        ...     n_latents=2,
        ...     segments=((Modality.IMAGE, 3), (Modality.ACTION, 2))
        ... )
        >>> attn = SpaceSelfAttentionModality(
        ...     d_model=128,
        ...     n_heads=4,
        ...     modality_ids=layout.modality_ids(),
        ...     n_latents=layout.n_latents,
        ...     mode="decoder",
        ... )
        >>> x = jnp.zeros((8, 5, layout.S(), 128))
        >>> y = attn(x, deterministic=True)
        >>> y.shape
        (8, 5, 7, 128)

        This means:
            - batch size = 8
            - time length = 5
            - per timestep there are 7 tokens
            - attention happens only within each timestep over the 7 tokens
    """

    d_model: int
    n_heads: int
    modality_ids: jnp.ndarray
    n_latents: int
    mode: str = "encoder"
    dropout: float = 0.0

    def setup(self):
        """
        Precompute and store the modality routing mask.

        This method builds a boolean matrix of shape `(S, S)`:

            mask[q, k] = True   means query token q may attend to key token k
            mask[q, k] = False  means that attention edge is forbidden

        The mask is determined entirely by:
            - token position
            - token modality
            - latent prefix length
            - routing mode

        The result is stored as a constant variable with shape `(1, 1, S, S)`
        so it can later broadcast over:
            - batch/time dimension after flattening `(B*T)`
            - attention heads

        Notes:
            - The first `n_latents` token positions are treated as latent tokens.
            - Non-latent token behavior depends on `mode`.
            - In wm-agent modes, tokens are grouped into AGENT / ACTION /
              observation buckets.

        Stored variable:
            self.modality_mask.value:
                shape `(1, 1, S, S)`, boolean
        """
        # Total number of tokens per timestep
        S = int(self.modality_ids.shape[0])

        # ---------------------------------------------------------------------
        # Build query/key index grids
        #
        # q_idx: (S, 1)   -> query token index per row
        # k_idx: (1, S)   -> key token index per column
        #
        # After broadcasting, these define all (query, key) pairs.
        # ---------------------------------------------------------------------
        q_idx = jnp.arange(S)[:, None]
        k_idx = jnp.arange(S)[None, :]

        # ---------------------------------------------------------------------
        # Identify whether query/key positions belong to latent tokens.
        #
        # Convention:
        #   token positions [0, ..., n_latents-1] are latent tokens
        # ---------------------------------------------------------------------
        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        # ---------------------------------------------------------------------
        # Lookup modality id for each query row and key column.
        #
        # q_mod: (S, 1)
        # k_mod: (1, S)
        # same_mod: (S, S)
        # ---------------------------------------------------------------------
        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            """
            Encoder routing:

                latent query     -> all keys
                non-latent query -> same-modality keys only
            """
            allow_lat_q = jnp.ones((S, S), dtype=bool)
            allow_nonlat_q = same_mod
            mask = jnp.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        elif self.mode == "decoder":
            """
            Decoder routing:

                latent query     -> latent keys only
                non-latent query -> same-modality keys OR latent keys
            """
            allow_lat_q = is_k_lat
            allow_nonlat_q = jnp.logical_or(same_mod, is_k_lat)
            mask = jnp.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        elif self.mode in ["wm_agent", "wm_agent_isolated"]:
            """
            World-model agent routing.

            Token groups:
                - AGENT
                - ACTION
                - observation bucket:
                    SPATIAL, REGISTER, SHORTCUT_SIGNAL, SHORTCUT_STEP

            Rules:
                wm_agent:
                    - AGENT query attends to all keys
                    - ACTION query attends only to ACTION keys
                    - observation query attends to observation keys and ACTION keys
                    - non-agent queries never attend to AGENT keys

                wm_agent_isolated:
                    - AGENT query attends to nobody
                    - ACTION / observation routing same as above
                    - non-agent queries never attend to AGENT keys
            """
            q_idx = jnp.arange(S)[:, None]
            k_idx = jnp.arange(S)[None, :]
            q_mod = self.modality_ids[q_idx]
            k_mod = self.modality_ids[k_idx]

            # -----------------------------------------------------------------
            # Identify AGENT / ACTION token groups
            # -----------------------------------------------------------------
            is_agent_q = (q_mod == Modality.AGENT)
            is_agent_k = (k_mod == Modality.AGENT)
            is_action_q = (q_mod == Modality.ACTION)
            is_action_k = (k_mod == Modality.ACTION)

            # -----------------------------------------------------------------
            # Observation bucket:
            #   SPATIAL, REGISTER, SHORTCUT_SIGNAL, SHORTCUT_STEP
            # -----------------------------------------------------------------
            is_obs_k = (
                (k_mod == Modality.SPATIAL) |
                (k_mod == Modality.REGISTER) |
                (k_mod == Modality.SHORTCUT_SIGNAL) |
                (k_mod == Modality.SHORTCUT_STEP)
            )
            is_obs_q = (
                (q_mod == Modality.SPATIAL) |
                (q_mod == Modality.REGISTER) |
                (q_mod == Modality.SHORTCUT_SIGNAL) |
                (q_mod == Modality.SHORTCUT_STEP)
            )

            # -----------------------------------------------------------------
            # AGENT query permissions
            #
            # wm_agent:
            #   AGENT query -> all keys
            #
            # wm_agent_isolated:
            #   AGENT query -> no keys
            # -----------------------------------------------------------------
            allow_for_agent_q = jnp.where(
                self.mode == "wm_agent",
                jnp.ones((S, S), dtype=bool),
                jnp.zeros((S, S), dtype=bool)
            )

            # -----------------------------------------------------------------
            # Non-agent query permissions
            #
            # ACTION query      -> ACTION keys only
            # observation query -> observation keys OR ACTION keys
            # other query types -> no keys
            # -----------------------------------------------------------------
            allow_for_action_q = is_action_k
            allow_for_obs_q = (is_obs_k | is_action_k)

            allow_nonagent = jnp.where(
                is_action_q, allow_for_action_q,
                jnp.where(is_obs_q, allow_for_obs_q, jnp.zeros((S, S), dtype=bool))
            )

            # -----------------------------------------------------------------
            # Non-agent queries are not allowed to read AGENT keys
            # -----------------------------------------------------------------
            allow_nonagent = jnp.where(is_agent_k, False, allow_nonagent)

            # -----------------------------------------------------------------
            # Final routing:
            #   AGENT queries use AGENT rule
            #   all others use non-agent rule
            # -----------------------------------------------------------------
            mask = jnp.where(is_agent_q, allow_for_agent_q, allow_nonagent)

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # ---------------------------------------------------------------------
        # Store mask as shape (1, 1, S, S)
        #
        # This shape is chosen so it can later broadcast to:
        #   (B*T, 1, S, S)
        #
        # where:
        #   - B*T is the flattened batch-time dimension
        #   - the singleton head dimension can broadcast across all heads
        # ---------------------------------------------------------------------
        modality_mask = mask[None, None, :, :]
        self.modality_mask = self.variable(
            "constants",
            "modality_mask",
            lambda: modality_mask
        )

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        """
        Apply modality-routed self-attention over the token axis.

        Args:
            x:
                Input tensor of shape `(B, T, S, D)`.

                Meaning:
                    B = batch size
                    T = number of timesteps
                    S = number of tokens per timestep
                    D = token feature width

                Attention is applied over `S` only, independently for each
                `(B, T)` pair.

            deterministic:
                If True, disable dropout inside attention.
                If False, enable dropout for training.

        Returns:
            Output tensor of shape `(B, T, S, D)`.

        Example:
            >>> layout = TokenLayout(
            ...     n_latents=2,
            ...     segments=((Modality.IMAGE, 3), (Modality.ACTION, 2))
            ... )
            >>> attn = SpaceSelfAttentionModality(
            ...     d_model=128,
            ...     n_heads=4,
            ...     modality_ids=layout.modality_ids(),
            ...     n_latents=2,
            ...     mode="encoder",
            ... )
            >>> x = jnp.zeros((4, 6, 7, 128))
            >>> y = attn(x, deterministic=True)
            >>> y.shape
            (4, 6, 7, 128)

            This means:
                for each batch element and timestep, the 7 tokens attend to each
                other according to the precomputed modality mask.
        """
        # x: (B, T, S, D)
        B, T, S, D = x.shape

        # ---------------------------------------------------------------------
        # Flatten batch and time so each timestep becomes one attention sample:
        #
        #   (B, T, S, D) -> (B*T, S, D)
        #
        # Self-attention is then computed independently for each row in B*T.
        # ---------------------------------------------------------------------
        x_ = x.reshape(B * T, S, D)

        # ---------------------------------------------------------------------
        # Broadcast cached routing mask to every flattened sample.
        #
        # Cached:
        #   (1, 1, S, S)
        #
        # Broadcasted:
        #   (B*T, 1, S, S)
        #
        # Flax multi-head attention accepts masks shaped like:
        #   (batch, num_heads_or_1, q_len, k_len)
        # ---------------------------------------------------------------------
        mask = jnp.broadcast_to(self.modality_mask.value, (B * T, 1, S, S))

        # ---------------------------------------------------------------------
        # Standard self-attention:
        #   query = key = value = x_
        #
        # But the mask restricts which query-key pairs are allowed.
        # ---------------------------------------------------------------------
        y_ = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x_, x_, mask=mask)

        # Restore original batch-time layout
        y = y_.reshape(B, T, S, D)
        return y
    
class TimeSelfAttention(nn.Module):
    """
    Apply causal self-attention over the time axis.

    This module performs self-attention across the temporal dimension `T`,
    while keeping different token slots in `S` separated from each other.

    Input convention:
        x has shape `(B, T, S, D)` where:
            B = batch size
            T = number of timesteps
            S = number of token slots per timestep
            D = feature width

    Attention is applied along the time axis only, with a causal mask:
        - each timestep may attend to itself and past timesteps
        - future timesteps are not visible

    Two operating modes are supported:

        latents_only = True:
            Only the first `n_latents` token slots are updated by temporal
            attention. All other token slots remain unchanged.

        latents_only = False:
            All `S` token slots are updated independently by temporal attention.

    Attributes:
        d_model:
            Feature width of each token.

        n_heads:
            Number of attention heads.

        dropout:
            Dropout rate inside multi-head attention.

        latents_only:
            If True, perform temporal attention only on the first `n_latents`
            token slots. If False, perform temporal attention on all token slots.

        n_latents:
            Number of latent token slots at the beginning of the `S` dimension.
            This is required when `latents_only=True`.

    Example:
        >>> x.shape
        (B, T, S, D)

        If:
            B = 2, T = 5, S = 7, D = 128
            n_latents = 3

        Then:

        latents_only=True:
            temporal attention is applied only to:
                x[:, :, 0, :], x[:, :, 1, :], x[:, :, 2, :]
            and token slots 3~6 are left unchanged.

        latents_only=False:
            temporal attention is applied independently to all 7 token slots.
    """

    d_model: int
    n_heads: int
    dropout: float = 0.0
    latents_only: bool = True
    n_latents: int = 0   # required if latents_only

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        """
        Apply causal self-attention over time.

        Args:
            x:
                Input tensor of shape `(B, T, S, D)`.

                Meaning:
                    B = batch size
                    T = number of timesteps
                    S = number of token slots per timestep
                    D = feature width

            deterministic:
                If True, disable dropout inside attention.
                If False, enable dropout for training.

        Returns:
            Output tensor of shape `(B, T, S, D)`.

            - If `latents_only=True`, only the first `n_latents` token slots
              are updated by temporal attention.
            - If `latents_only=False`, all token slots are updated.

        Notes:
            - Attention is causal along time, so timestep `t` cannot attend to
              future timesteps `> t`.
            - Attention is applied independently for each token slot in `S`.
              In other words, token slot `s=0` attends only to its own history
              across time, not to other token slots.
        """
        # x: (B, T, S, D) -> attend across T, causal
        B, T, S, D = x.shape

        if self.latents_only:
            """
            Temporal attention only for the first n_latents token slots.

            Steps:
                1. Extract latent token slots:
                       (B, T, S, D) -> (B, T, L, D)
                   where L = n_latents

                2. Move latent-slot axis before time:
                       (B, T, L, D) -> (B, L, T, D)

                3. Flatten batch and latent-slot dimensions:
                       (B, L, T, D) -> (B*L, T, D)

                   This means each latent slot gets its own temporal sequence.

                4. Build a causal mask over T.

                5. Apply self-attention over time independently for each
                   flattened sequence.

                6. Reshape back and write the updated latent tokens into the
                   original tensor, leaving non-latent tokens unchanged.
            """
            assert 0 < self.n_latents <= S

            # Extract first L latent token slots
            lat = x[:, :, :self.n_latents, :]                # (B, T, L, D)

            # Rearrange so each latent slot becomes an independent temporal sequence
            lat_btld = lat.transpose(0, 2, 1, 3).reshape(B * self.n_latents, T, D)
            # shape: (B*L, T, D)

            # Build causal mask over time:
            # timestep t can attend only to positions <= t
            causal = nn.attention.make_causal_mask(
                jnp.ones((B * self.n_latents, T), dtype=bool)
            )

            # Apply temporal self-attention independently to each latent slot
            out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout,
                deterministic=deterministic,
            )(lat_btld, lat_btld, mask=causal)

            # Restore shape back to (B, T, L, D)
            out = out.reshape(B, self.n_latents, T, D).transpose(0, 2, 1, 3)

            # Replace only the latent token slots in the original tensor
            x = x.at[:, :, :self.n_latents, :].set(out)
            return x

        else:
            """
            Temporal attention for all token slots.

            Steps:
                1. Move token-slot axis before time:
                       (B, T, S, D) -> (B, S, T, D)

                2. Flatten batch and token-slot dimensions:
                       (B, S, T, D) -> (B*S, T, D)

                   This means each token slot gets its own temporal sequence.

                3. Build a causal mask over T.

                4. Apply self-attention over time independently for each token slot.

                5. Reshape back to the original `(B, T, S, D)` layout.
            """
            # Rearrange so each token slot becomes an independent temporal sequence
            x_bstd = x.transpose(0, 2, 1, 3).reshape(B * S, T, D)  # (B*S, T, D)

            # Build causal mask over time
            causal = nn.attention.make_causal_mask(
                jnp.ones((B * S, T), dtype=bool)
            )

            # Apply temporal self-attention independently to each token slot
            out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout,
                deterministic=deterministic,
            )(x_bstd, x_bstd, mask=causal)

            # Restore original layout
            out = out.reshape(B, S, T, D).transpose(0, 2, 1, 3)  # (B, T, S, D)
            return out

# ---------- a single block-causal layer ----------
class BlockCausalLayer(nn.Module):
    d_model: int
    n_heads: int
    n_latents: int
    modality_ids: jnp.ndarray     # (S,)
    space_mode: str               # "encoder", "decoder", "wm_agent", "wm_agent_isolated"
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    layer_index: int = 0
    time_every: int = 4
    latents_only_time: bool = True

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # --- Space attention (within timestep, modality-aware) ---
        y = RMSNorm()(x)
        y = SpaceSelfAttentionModality(
            d_model=self.d_model,
            n_heads=self.n_heads,
            modality_ids=self.modality_ids,
            n_latents=self.n_latents,
            mode=self.space_mode,
            dropout=self.dropout,
        )(y, deterministic=deterministic)
        x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)

        # --- Time attention (causal across timesteps), only on some layers ---
        if (self.layer_index + 1) % self.time_every == 0:
            y = RMSNorm()(x)
            y = TimeSelfAttention(
                self.d_model, self.n_heads, self.dropout,
                latents_only=self.latents_only_time, n_latents=self.n_latents
            )(y, deterministic=deterministic)
            x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)

        # --- MLP ---
        y = RMSNorm()(x)
        y = MLP(self.d_model, self.mlp_ratio, self.dropout)(y, deterministic=deterministic)
        x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)
        return x
# ---------- the transformer stack ----------

class BlockCausalTransformer(nn.Module):
    """由多个 BlockCausalLayer 堆叠而成的主干 Transformer。"""
    d_model: int
    n_heads: int
    depth: int
    n_latents: int
    modality_ids: jnp.ndarray   # (S,)
    space_mode: str             # "encoder" or "decoder"
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    latents_only_time: bool = True

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        for i in range(self.depth):
            x = BlockCausalLayer(
                self.d_model, self.n_heads, self.n_latents,
                modality_ids=self.modality_ids,
                space_mode=self.space_mode,
                dropout=self.dropout, mlp_ratio=self.mlp_ratio,
                layer_index=i, time_every=self.time_every,
                latents_only_time=self.latents_only_time,
            )(x, deterministic=deterministic)
        return x

class Encoder(nn.Module):
    """Dreamer 编码器：图像 patch + latent token 编码到瓶颈潜变量。"""
    d_model: int
    n_latents: int
    n_patches: int
    n_heads: int
    depth: int
    d_bottleneck: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    latents_only_time: bool = True
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    
    def setup(self):
        self.patch_proj = nn.Dense(self.d_model, name="patch_proj")
        self.bottleneck_proj = nn.Dense(self.d_bottleneck, name="bottleneck_proj")
        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        self.modality_ids = self.layout.modality_ids()            # (S,)
        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            n_latents=self.n_latents,
            modality_ids=self.modality_ids,
            space_mode="encoder",                 # << encoder routing
            dropout=self.dropout, mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=self.latents_only_time,
        )
        self.latents = self.param("latents_enc", nn.initializers.normal(0.02), (self.n_latents, self.d_model))

    @nn.compact
    def __call__(self, patch_tokens, *, deterministic: bool = True) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1) Project patches to D_model
        proj_patches = self.patch_proj(patch_tokens)  # (B,T,Np,D)

        # 2) MAE mask-and-replace on patch tokens (encoder input only)
        proj_patches_masked, patch_mask, keep_prob = MAEReplacer(name="mae", p_min=self.mae_p_min, p_max=self.mae_p_max)(proj_patches)
        # print(f"proj_patches_masked.shape: {proj_patches_masked.shape}")
        # print(f"patch_mask.shape: {patch_mask.shape}")

        # 3) Prepend learned latents (owned here)
        # print(f"latents.shape: {latents.shape}")
        B, T = proj_patches_masked.shape[:2]
        latents = jnp.broadcast_to(self.latents[None, None, ...], (B, T, *self.latents.shape))
        # print(f"lat_btld.shape: {lat_btld.shape}")
        tokens = jnp.concatenate([latents, proj_patches_masked], axis=2)  # (B,T,S=(Np+Nl),D)
        # print(f"tokens_btSd.shape: {tokens_btSd.shape}")

        # 4) Add sinusoidal positions (param-free)
        tokens = add_sinusoidal_positions(tokens)

        # 5) Feed tokens into transformer
        encoded_tokens = self.transformer(tokens, deterministic=deterministic)
        # print(f"encoded_tokens_btSd.shape: {encoded_tokens_btSd.shape}")

        # 6) Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, :self.n_latents, :]
        proj_tokens = nn.tanh(self.bottleneck_proj(latent_tokens))

        return proj_tokens, (patch_mask, keep_prob)  # keep mask if you want diagnostics

class Decoder(nn.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.

    Inputs:
      - z: (B, T, N_l, d_bottleneck)  -- encoder bottleneck output

    Config:
      - n_patches: number of patch query tokens to use in the decoder
      - d_patch:   dimensionality of each patch to reconstruct (D_patch)
      - d_model, n_heads, depth, dropout, mlp_ratio, time_every, latents_only_time
        typically mirror the encoder.
    """
    d_model: int
    n_heads: int
    depth: int
    n_latents: int
    n_patches: int
    d_patch: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    latents_only_time: bool = True

    def setup(self):
        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        self.modality_ids = self.layout.modality_ids()
        self.up_proj = nn.Dense(self.d_model, name="up_proj")
        self.patch_queries = self.param(
            "patch_queries",
            nn.initializers.normal(0.02),
            (self.n_patches, self.d_model),
        ) # (Np, D)
        self.patch_head = nn.Dense(self.d_patch, name="patch_head") # (Np, D_patch)
        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            n_latents=self.n_latents,
            modality_ids=self.modality_ids,
            space_mode="decoder",                 # << decoder routing
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=self.latents_only_time,
        )

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        B, T, N_l, d_bottleneck = z.shape

        # 1) Up-project latent bottleneck to d_model (per latent token)
        latents = nn.tanh(self.up_proj(z))  # (B, T, N_l, D)

        # 2) Learned per-patch query tokens (owned by the decoder)
        patches = jnp.broadcast_to(
            self.patch_queries[None, None, ...],
            (B, T, self.n_patches, self.d_model),
        )  # (B, T, Np, D)

        # 3) Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=2)

        # 4) Add sinusoidal positions
        tokens = add_sinusoidal_positions(tokens)

        # 5) Axial block-causal transformer
        #    - SpaceSelfAttention over all S tokens (latents + queries)
        #    - TimeSelfAttention only over the first N_l latent tokens
        x = self.transformer(tokens, deterministic=deterministic)
        # 6) Prediction head over the patch-query slice
        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = nn.sigmoid(self.patch_head(x_patches))  # (B,T,Np,D_patch)
        return pred_btnd

class ActionEncoder(nn.Module):
    d_model: int
    n_keyboard: int = 5  # up, down, left, right, null (categorical actions)

    @nn.compact
    def __call__(
        self,
        actions: Optional[jnp.ndarray],           # (B, T) int32 in [0, n_keyboard)
        batch_time_shape: Optional[Tuple[int,int]] = None,
        as_tokens: bool = True,
    ):
        # Base "action token" embedding (used always)
        base_emb = self.param(
            'base_action_emb', nn.initializers.normal(0.02), (self.d_model,)
        )

        if actions is None:
            # unlabeled videos: just broadcast base embedding
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = jnp.broadcast_to(base_emb, (B, T, self.d_model))
        else:
            # embed categorical actions
            emb_key = nn.Embed(self.n_keyboard, self.d_model, name="emb_key")(actions)
            out = emb_key + base_emb  # broadcast add

        if as_tokens:
            # expand a token axis (S_a = 1)
            out = out[:, :, None, :]

        return out

class Dynamics(nn.Module):
    d_model: int              # dimensionality of each token
    d_bottleneck: int         # dimensionality of the input bottleneck space
    d_spatial: int            # dimensionality of each spatial token input
    n_spatial: int            # number of spatial tokens
    n_register: int           # number of learned register tokens
    n_agent: int              # number of agent tokens
    n_heads: int
    depth: int
    k_max: int                 # maximum number of sampling steps (defines finest step 1/)
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    space_mode: str = "wm_agent_isolated" # or "wm_agent"
    n_actions: int = 5

    def setup(self):
        # Want to transform bottleneck inputs (B, T, N_b, D_b) to (B, T, N_b/packing_factor, D_b*packing_factor)
        assert self.d_spatial % self.d_bottleneck == 0
        self.spatial_proj = nn.Dense(self.d_model, name="proj_spatial") # converts spatial tokens, of dim d_spatial to d_model
        self.register_tokens = self.param(
            "register_tokens",
            nn.initializers.normal(0.02),
            (self.n_register, self.d_model),
        )
        self.action_encoder = ActionEncoder(d_model=self.d_model, n_keyboard=self.n_actions)

        # Two separate tokens for shortcut conditioning (your current layout):
        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),   # τ (signal level) token
            (Modality.SHORTCUT_STEP, 1),     # d (step size) token
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))
        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        self.spatial_slice = self.layout.slices()[Modality.SPATIAL]
        self.agent_slice  = self.layout.slices().get(Modality.AGENT, slice(0,0))  # safe if n_agent==0
        self.modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            n_latents=0,
            modality_ids=self.modality_ids,
            space_mode=self.space_mode,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=False,
        )

        # -------- Discrete embeddings for shortcut conditioning --------
        # Step size d ∈ {1, 1/2, 1/4, ..., 1/256}
        # We index steps by: step_idx = log2(1/d) ∈ {0, 1, 2, ...,7, 8}
        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embed(self.num_step_bins, self.d_model, name="step_embed")

        # Signal level τ ∈ {0, 1/d, 2/d, ..., 1 - 1/d} (grid length = 1/d)
        # We use a *shared* table with  bins and only use the first (1/d) entries for a given d.
        self.signal_embed = nn.Embed(self.k_max + 1, self.d_model, name="signal_embed")
        self.flow_x_head = nn.Dense(self.d_spatial, name="flow_x_head", kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros)  # zero-init

    @nn.compact
    def __call__(
        self,
        actions,             # (B,T)
        step_idxs,           # (B,T)
        signal_idxs,         # (B,T)
        packed_enc_tokens,   # (B,T,n_s,d_spatial)
        *,
        agent_tokens: Optional[jnp.ndarray] = None,  # (B,T,n_agent,D) or None
        deterministic: bool = True,
    ):
        """
        Pretrain script: instantiate with space_mode="wm_agent_isolated" and pass agent_tokens=None (dummy).
        Fine-tune script: instantiate with space_mode="wm_agent" and pass real agent_tokens from task embedding.
        Args:
          packed_enc_tokens:      (B, T, n_spatial, d_spatial) packed encoder tokens
          actions:    (B, T, N_a, D_a) raw action tokens
          steps:      (B, T) float32 — step sizes, 1/2^x
          signals:    (B, T) float32 - signal values, grid that is reachable by current step size

        Shapes produced:
          spatial_tokens: (B, T, n_spatial, d_model)
          action_tokens:  (B, T, 1, d_model)  # if your ActionEncoder emits one token
          signal_token:   (B, T, 1, d_model)
          step_token:     (B, T, 1, d_model)
        """
        # --- 1) Project spatial tokens to model dimension
        spatial_tokens = self.spatial_proj(packed_enc_tokens) # (B, T, n_spatial, d_model)

        # --- 2) Encode actions to d_model
        action_tokens = self.action_encoder(actions)  # (B, T, N_a, d_model)

        # --- 3) Prepare learned register tokens
        B, T = spatial_tokens.shape[:2]
        register_tokens = jnp.broadcast_to(
            self.register_tokens[None, None, ...],  # (1,1,n_register,d_model)
            (B, T, self.n_register, self.d_model),
        )

        # --- 4) Shortcut embeddings (discrete lookup)
        step_tok   = self.step_embed(step_idxs)[:, :, None, :]      # (B, T, 1, d_model)
        signal_tok = self.signal_embed(signal_idxs)[:, :, None, :]     # (B, T, 1, d_model)
        
        # --- 5) Concatenate in your declared layout order
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = jnp.zeros((B, T, self.n_agent, self.d_model), dtype=spatial_tokens.dtype)
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, agent_tokens]
        else:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens]
        tokens = jnp.concatenate(toks, axis=2)                    # (B,T,S,D)

        tokens = add_sinusoidal_positions(tokens)      # (B, T, N_total, d_model)
        x = self.transformer(tokens, deterministic=deterministic)
        spatial_tokens = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_tokens)
        h_t = x[:, :, self.agent_slice, :] if self.n_agent > 0 else None  # (B,T,n_agent,D) or None
        return x1_hat, h_t

class TaskEmbedder(nn.Module):
    d_model: int
    n_agent: int = 1
    use_ids: bool = True     # True: task is int ids; False: task is vector
    n_tasks: int = 128       # only used if use_ids=True
    d_task: int = 64         # only used if use_ids=False

    @nn.compact
    def __call__(self, task, B: int, T: int):
        """
        If use_ids=True:
            task: (B,) int32 ids in [0, n_tasks)
        Else:
            task: (B, d_task) float32 vector

        Returns agent tokens: (B, T, n_agent, d_model)
        """
        if self.use_ids:
            emb = nn.Embed(self.n_tasks, self.d_model, name="task_table")(task)  # (B, D)
        else:
            emb = nn.Dense(self.d_model, name="task_proj")(task)                 # (B, D)

        # Learned base + optional small MLP to decouple from raw table
        base = self.param("agent_base", nn.initializers.normal(0.02), (self.d_model,))
        x = emb + base[None, :]

        # Replicate across time and agent slots
        x = jnp.broadcast_to(x[:, None, None, :], (B, T, self.n_agent, self.d_model))
        return x

# === Phase B heads (use existing MLP) =========================================

class PolicyHeadMTP(nn.Module):
    """Multi-Token action prediction.
    Input:  h_t (B, T, D)  -- agent readouts (pool n_agent first if needed)
    Output: logits (B, T, L, A)
    """
    d_model: int
    action_dim: int
    L: int = 8
    kind: str = "categorical"  # or "vbinary"
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        # Feature projector (D -> D) using your MLP
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            swiglu=self.swiglu,
            parity_2over3=self.parity_2over3,
            dtype=self.dtype,
        )
        # Single matmul that produces all L offsets at once: (… , D) -> (…, L, A)
        self.out = nn.DenseGeneral(
            features=(self.L, self.action_dim),
            axis=-1,
            dtype=self.dtype,
            name="out",
        )

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.projector(h_t, deterministic=deterministic)  # (B, T, D)
        logits = self.out(x)                                  # (B, T, L, A)
        return logits  # softmax/sigmoid applied at loss-time based on `kind`


class RewardHeadMTP(nn.Module):
    """Multi-Token reward prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, L, K), centers (K,)
    """
    d_model: int
    L: int = 8
    num_bins: int = 101
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32
    # log-space bounds for symexp bins (tune per dataset)
    log_low: float = -8.0
    log_high: float = 8.0

    def setup(self):
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            swiglu=self.swiglu,
            parity_2over3=self.parity_2over3,
            dtype=self.dtype,
        )
        self.out = nn.DenseGeneral(
            features=(self.L, self.num_bins),
            axis=-1,
            dtype=self.dtype,
            name="out",
        )
        # Precompute bin centers as a constant (share across calls/checkpoints)
        # Simple choice: uniform in log-space, then exponentiate symmetrically.
        log_edges = jnp.linspace(self.log_low, self.log_high, self.num_bins)
        # centers ~ same length for convenience (pad to K if using edges-midpoints):
        centers = log_edges
        self.centers_var = self.variable("constants", "symexp_centers_log", lambda: centers)

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t, deterministic=deterministic)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, L, K)
        centers_log = self.centers_var.value                   # (K,)
        return logits, centers_log


class ValueHead(nn.Module):
    """Value prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, K), centers (K,)
    """
    d_model: int
    num_bins: int = 101
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32
    # log-space bounds for symexp bins (tune per dataset)
    log_low: float = -8.0
    log_high: float = 8.0

    def setup(self):
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            swiglu=self.swiglu,
            parity_2over3=self.parity_2over3,
            dtype=self.dtype,
        )
        self.out = nn.DenseGeneral(
            features=self.num_bins,
            axis=-1,
            dtype=self.dtype,
            name="out",
        )
        # Precompute bin centers as a constant (share across calls/checkpoints)
        # Simple choice: uniform in log-space, then exponentiate symmetrically.
        log_edges = jnp.linspace(self.log_low, self.log_high, self.num_bins)
        # centers ~ same length for convenience (pad to K if using edges-midpoints):
        centers = log_edges
        self.centers_var = self.variable("constants", "symexp_centers_log", lambda: centers)

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t, deterministic=deterministic)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, K)
        centers_log = self.centers_var.value                   # (K,)
        return logits, centers_log


def test_encoder_decoder():
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    n_patches = 4
    d_patch = 3
    enc_n_latents = 2
    enc_d_bottleneck = 3
    x = jnp.ones((B, T, n_patches, d_patch))  # (B,T,Np,D_patch)

    encoder = Encoder(d_model=8, n_latents=enc_n_latents, n_patches=n_patches, n_heads=2, depth=2, dropout=0.5, d_bottleneck=enc_d_bottleneck)
    decoder = Decoder(d_model=8, n_heads=2, depth=2, n_patches=n_patches, n_latents=enc_n_latents, d_patch=d_patch, dropout=0.5)
    # init: give both "mae" and "dropout" keys (dropout only needed if deterministic=False)
    enc_vars = encoder.init(
        {"params": rng, "mae": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        x,
        deterministic=True,
    )
    # Decode
    fake_z = jnp.ones((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_z,
        deterministic=True,
    )

    def forward_apply(enc_vars: FrozenDict, dec_vars: FrozenDict,
                    patches_btnd: jnp.ndarray,
                    *, mae_key=None, drop_key=None, train: bool):
        # Encoder
        rngs_enc = {}
        if train:
            rngs_enc = {"mae": mae_key, "dropout": drop_key}
        else:
            rngs_enc = {"mae": mae_key}  # if you still want masking during eval

        z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd,
                                        rngs=rngs_enc,
                                        deterministic=not train)
        # Decoder
        rngs_dec = {"dropout": drop_key} if train else {}
        pred_btnd = decoder.apply(dec_vars, z_btLd,
                                rngs=rngs_dec,
                                deterministic=not train)
        return pred_btnd, mae_info
    
    jit_forward = jax.jit(forward_apply, static_argnames=["train"])
    mae_key = jax.random.PRNGKey(0)
    drop_key = jax.random.PRNGKey(1)
    # Warm-up (compilation happens here)
    t0 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t1 = time.time()
    # Hot run (should be much faster)
    t2 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t3 = time.time()

    print(f"Warm-up (compile+run): {t1 - t0:.3f}s")
    print(f"Hot run (cached):      {t3 - t2:.3f}s")

def test_dynamics():
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    fake_enc_z = jnp.ones((B, T, 512, 16), dtype=jnp.float32)
    fake_actions = jnp.ones((B, T), dtype=jnp.int32)
    fake_steps = jnp.full((B, T), 1/256, dtype=jnp.float32)
    fake_signals = jnp.full((B, T), 0.0, dtype=jnp.float32)
    def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
        """
        (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
        Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
        """
        return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)
    fake_packed_enc_tokens = pack_bottleneck_to_spatial(fake_enc_z, n_spatial=256, k=2)


    # need some way to assert that 512 * 16 == 256 * 32
    dynamics_kwargs = {
        "d_model": 128,
        "n_spatial": 256,
        "d_spatial": 32,
        "d_bottleneck": 16,
        "k_max": 8,
        "n_register": 10,
        "n_heads": 4,
        "depth": 4,
        "dropout": 0.0
    }
    dynamics = Dynamics(**dynamics_kwargs)
    dynamics_vars = dynamics.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_actions,
        fake_steps,
        fake_signals,
        fake_packed_enc_tokens,
    )
    out = dynamics.apply(dynamics_vars, fake_actions, fake_steps, fake_signals, fake_packed_enc_tokens,
                        rngs={"dropout": jax.random.PRNGKey(2)},
                        deterministic=True)

def _build_modality_mask(modality_ids, mode: str, n_latents=0, d_model=16, n_heads=2):
    class _Peek(nn.Module):
        @nn.compact
        def __call__(self, x):
            att = SpaceSelfAttentionModality(
                d_model=d_model, n_heads=n_heads,
                modality_ids=modality_ids, n_latents=n_latents,
                mode=mode, dropout=0.0)
            y = att(x, deterministic=True)
            # expose stored mask
            mask = att.variables["constants"]["modality_mask"]  # (1,1,S,S)
            return y, mask

    B,T,S,D = 1,1,modality_ids.shape[0],d_model
    x = jnp.zeros((B,T,S,D))
    vars_ = _Peek().init(jax.random.PRNGKey(0), x)
    _, mask = _Peek().apply(vars_, x, mutable=False)
    return jnp.asarray(mask)  # (1,1,S,S)

def _pack_bottleneck_to_spatial(z_btLd, n_spatial, k):
    """将 (B,T,N_b,D_b) 重排为 (B,T,n_spatial,k*D_b)。"""
    return rearrange(z_btLd, 'b t (n k) d -> b t n (k d)', n=n_spatial, k=k)

def _abbr(m):
    """将模态 ID 转为短标签（调试打印用）。"""
    return {
        int(Modality.ACTION): "ACT",
        int(Modality.SHORTCUT_SIGNAL): "SIG",
        int(Modality.SHORTCUT_STEP): "STP",
        int(Modality.SPATIAL): "SPA",
        int(Modality.REGISTER): "REG",
        int(Modality.AGENT): "AGT",
        int(Modality.LATENT): "LAT",
    }.get(int(m), f"M{int(m)}")

def _print_mask_summary(name: str, modality_ids: jnp.ndarray, mask_2d: jnp.ndarray):
    """打印模态注意力掩码摘要。

    mask_2d 形状为 (S,S)，True 表示“该 query 行可读取该 key 列”。
    """
    S = modality_ids.shape[0]
    mods = [int(x) for x in list(modality_ids)]
    headers = "     " + " ".join(f"{_abbr(m):>3}" for m in mods)
    print(f"\n[{name}] modality order (Q rows / K cols): {mods}")
    print(headers)
    for q in range(S):
        row = "".join("  ✓" if bool(mask_2d[q, k]) else "  ·" for k in range(S))
        print(f"{_abbr(modality_ids[q]):>3}: {row}")
    # row-wise counts
    counts = jnp.sum(mask_2d, axis=1)
    print("Row read-counts:", counts.tolist())

def test_agent_firewall():
    # layout: [ACTION, SIG, STEP, SPATIALx3, REGISTERx2, AGENTx1]
    ACTION,SIGNAL,STEP,SPATIAL,REGISTER,AGENT = 1,5,6,4,3,7
    modality_ids = jnp.array([ACTION, SIGNAL, STEP, SPATIAL, SPATIAL, SPATIAL, REGISTER, REGISTER, AGENT], dtype=jnp.int32)
    S = modality_ids.shape[0]
    agent_col = (modality_ids == AGENT)  # keys that are agent
    agent_row = (modality_ids == AGENT)  # queries that are agent

    # ----- wm_agent -----
    mask = _build_modality_mask(modality_ids, "wm_agent")[0,0]  # (S,S)
    _print_mask_summary("wm_agent", modality_ids, mask)

    # Others never see agent: find any offending (q,k) where q!=agent and k is agent
    bad_q = []
    for q in range(S):
        if not bool(agent_row[q]):
            if bool(mask[q, agent_col].sum()):
                bad_q.append(q)
    if bad_q:
        print("Violations in wm_agent (non-agent reads agent) at query rows:", bad_q)

    # Agent reads all in wm_agent
    agent_q_idx = int(jnp.where(agent_row, size=1, fill_value=-1)[0][0])
    if agent_q_idx >= 0:
        agent_reads = mask[agent_q_idx, :]
        missing = [k for k in range(S) if not bool(agent_reads[k])]
        if missing:
            print("Violations in wm_agent (agent cannot read some keys). Missing cols:", missing)

    # Assertions
    for q in range(S):
        if not bool(agent_row[q]):
            assert mask[q, agent_col].sum() == 0, "Non-agent query can attend to agent!"
    if agent_q_idx >= 0:
        assert jnp.all(mask[agent_q_idx, :]), "Agent query cannot read some token in wm_agent"

    # ----- wm_agent_isolated -----
    mask_iso = _build_modality_mask(modality_ids, "wm_agent_isolated")[0,0]
    _print_mask_summary("wm_agent_isolated", modality_ids, mask_iso)

    # Others still never see agent
    bad_q_iso = []
    for q in range(S):
        if not bool(agent_row[q]):
            if bool(mask_iso[q, agent_col].sum()):
                bad_q_iso.append(q)
    if bad_q_iso:
        print("Violations in wm_agent_isolated (non-agent reads agent) at query rows:", bad_q_iso)

    # Agent reads nobody in isolated
    if agent_q_idx >= 0:
        agent_reads_iso = int(mask_iso[agent_q_idx, :].sum())
        print("Agent read-count in isolated mode:", agent_reads_iso)

    # Assertions
    for q in range(S):
        if not bool(agent_row[q]):
            assert mask_iso[q, agent_col].sum() == 0, "Non-agent query can attend to agent in isolated!"
    if agent_q_idx >= 0:
        assert mask_iso[agent_q_idx, :].sum() == 0, "Agent should read nobody in wm_agent_isolated"


def test_x1hat_invariant_to_agent_tokens():
    B,T = 2,5
    n_b, d_b = 8, 4      # encoder latents
    n_spatial, pack = 4, 2
    d_spatial = d_b * pack
    D = 32

    fake_enc_z = jnp.ones((B, T, n_b, d_b))
    packed = _pack_bottleneck_to_spatial(fake_enc_z, n_spatial=n_spatial, k=pack)
    actions = jnp.zeros((B,T), dtype=jnp.int32)
    step_idx = jnp.zeros((B,T), dtype=jnp.int32)
    sig_idx  = jnp.zeros((B,T), dtype=jnp.int32)

    dyn = Dynamics(
        d_model=D, d_bottleneck=d_b, d_spatial=d_spatial,
        n_spatial=n_spatial, n_register=2, n_agent=1,
        n_heads=2, depth=2, k_max=8, dropout=0.0, mlp_ratio=2.0,
        time_every=2, space_mode="wm_agent"  # try either mode
    )
    vars_ = dyn.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                     actions, step_idx, sig_idx, packed)

    # random agent vs zeros
    agent_rand = jax.random.normal(jax.random.PRNGKey(2), (B,T,1,D))
    x1_a, _ = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                        agent_tokens=agent_rand, rngs={"dropout": jax.random.PRNGKey(3)}, deterministic=True)
    x1_b, _ = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                        agent_tokens=jnp.zeros_like(agent_rand), rngs={"dropout": jax.random.PRNGKey(3)}, deterministic=True)

    diff = x1_a - x1_b
    max_abs = float(jnp.max(jnp.abs(diff)))
    l2 = float(jnp.sqrt(jnp.sum(diff * diff)))
    print("\n[x1_hat invariance] max|Δ| =", max_abs, " ||Δ||₂ =", l2)
    print("x1_a shape:", x1_a.shape, " x1_b shape:", x1_b.shape)

    # Must be exactly equal because agent cannot influence others
    assert jnp.allclose(x1_a, x1_b, atol=0, rtol=0), "x1_hat changed with agent tokens—firewall broken"


def test_shapes_and_h_t():
    B,T,D = 2,6,32
    n_b,d_b = 8,4
    n_spatial, pack = 4,2
    d_spatial = d_b*pack

    packed = _pack_bottleneck_to_spatial(jnp.ones((B,T,n_b,d_b)), n_spatial, pack)
    dyn = Dynamics(d_model=D, d_bottleneck=d_b, d_spatial=d_spatial,
                   n_spatial=n_spatial, n_register=3, n_agent=1,
                   n_heads=2, depth=2, k_max=8, space_mode="wm_agent")
    actions = jnp.zeros((B,T), dtype=jnp.int32)
    step_idx = jnp.zeros((B,T), dtype=jnp.int32)
    sig_idx  = jnp.zeros((B,T), dtype=jnp.int32)
    vars_ = dyn.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                     actions, step_idx, sig_idx, packed)

    x1_hat, h_t = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                            agent_tokens=jnp.zeros((B,T,1,D)))
    print("\n[shapes] x1_hat:", x1_hat.shape, " h_t:", (None if h_t is None else h_t.shape))
    print("Expect x1_hat =", (B,T,n_spatial,d_spatial), " h_t =", (B,T,1,D))
    assert x1_hat.shape == (B,T,n_spatial,d_spatial)
    assert h_t.shape     == (B,T,1,D)

def test_wm_routed():
    """
    Checks space-attention routing for Dreamer-4-style dynamics:
      - Action q -> {Action k}
      - Obs q    -> {Obs k ∪ Action k} and never Agent k
      - Agent q  -> {Obs k ∪ Action k ∪ Agent k}    (wm_agent)
                  -> {}                              (wm_agent_isolated)
      - For any non-agent q, Agent k is disallowed.
    """
    # Shorthand modality ints
    ACTION  = int(Modality.ACTION)
    SIGNAL  = int(Modality.SHORTCUT_SIGNAL)
    STEP    = int(Modality.SHORTCUT_STEP)
    SPATIAL = int(Modality.SPATIAL)
    REGISTER= int(Modality.REGISTER)
    AGENT   = int(Modality.AGENT)

    # Toy layout (Q rows / K cols share this order):
    # [ACT, SIG, STP, SPA, SPA, SPA, REG, REG, ACT, AGT]
    modality_ids = jnp.array(
        [ACTION, SIGNAL, STEP, SPATIAL, SPATIAL, SPATIAL, REGISTER, REGISTER, ACTION, AGENT],
        dtype=jnp.int32
    )
    S = modality_ids.shape[0]

    # Helper sets
    is_agent = (modality_ids == AGENT)
    is_action = (modality_ids == ACTION)
    is_obs = (
        (modality_ids == SPATIAL) |
        (modality_ids == REGISTER) |
        (modality_ids == SIGNAL)  |
        (modality_ids == STEP)
    )

    def assert_mask(mode: str):
        mask = _build_modality_mask(modality_ids, mode)[0, 0]  # (S,S) bool
        _print_mask_summary(mode, modality_ids, mask)

        # 1) Non-agent q must never see Agent k
        for q in range(S):
            if not bool(is_agent[q]):
                assert not bool(mask[q, is_agent].any()), f"[{mode}] non-agent q={q} can read Agent k!"

        # 2) Action q -> Action k only
        for q in range(S):
            if bool(is_action[q]):
                # Allowed: action keys only
                allowed = mask[q]
                assert bool(allowed[is_action].all()), f"[{mode}] action q={q} cannot read some action k!"
                assert not bool(allowed[~is_action].any()), f"[{mode}] action q={q} reads non-action keys!"

        # 3) Obs q -> Obs k ∪ Action k (and never Agent k, already checked)
        for q in range(S):
            if bool(is_obs[q]):
                allowed = mask[q]
                # Must allow all obs keys? We enforce "subset includes only obs∪action".
                # It's okay if some obs keys are masked by design, but we require no extra keys.
                extras = allowed & ~(is_obs | is_action)
                assert not bool(extras.any()), f"[{mode}] obs q={q} reads keys outside obs∪action!"

                # Should at least be able to read *some* obs or action key (nontrivial)
                assert bool((allowed & (is_obs | is_action)).any()), f"[{mode}] obs q={q} cannot read obs∪action at all!"

        # 4) Agent q behavior differs by mode
        agent_rows = [i for i in range(S) if bool(is_agent[i])]
        if agent_rows:
            q = agent_rows[0]
            if mode == "wm_agent":
                # Agent reads everyone (including agent)
                assert bool(mask[q].all()), "[wm_agent] agent q cannot read all keys!"
            else:
                # Isolated: agent reads nobody
                assert int(mask[q].sum()) == 0, "[wm_agent_isolated] agent q should read nobody!"

    # Run both modes
    assert_mask("wm_agent")
    assert_mask("wm_agent_isolated")
    print("\n[test_wm_routed] All routing assertions passed ✅")


if __name__ == "__main__":
    # test_agent_firewall()
    # test_x1hat_invariant_to_agent_tokens()
    # test_shapes_and_h_t()
    test_wm_routed()
    print("\nAll tests passed ✅")
