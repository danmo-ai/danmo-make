"""
Attention modules — shared across all models.

Reference implementations: Flux1/Flux2 Attention and spatiotemporal video attention patterns.
"""
from __future__ import annotations

import importlib
from typing import Any, Optional


def _int32_dtype(ctx: Any) -> Any:
    d = getattr(ctx, "int32", None)
    if callable(d):
        return d()
    if d is not None:
        return d
    raise RuntimeError("Context does not provide int32 dtype")


def _bool_dtype(ctx: Any) -> Any:
    for name in ("bool", "bool_"):
        d = getattr(ctx, name, None)
        if callable(d):
            return d()
        if d is not None:
            return d
    raise RuntimeError("Context does not provide bool dtype")


def build_key_padding_mask_from_lengths(
    ctx: Any,
    lengths: Any,
    seq_len: int,
    dtype: Any,
    *,
    neg_value: float = -1e9,
) -> Any:
    """Build key-padding mask from per-batch valid lengths."""
    b = int(lengths.shape[0])
    positions = ctx.arange(0, int(seq_len), dtype=_int32_dtype(ctx))
    valid_k = positions.reshape(1, 1, 1, int(seq_len)) < lengths.reshape(b, 1, 1, 1)
    shape = (b, 1, int(seq_len), int(seq_len))
    neg = ctx.full(shape, float(neg_value), dtype=dtype)
    return ctx.where(valid_k, ctx.zeros(shape, dtype=dtype), neg)


def build_causal_attention_mask(
    ctx: Any,
    seq_len: int,
    dtype: Any,
    *,
    neg_value: float = -1e9,
) -> Any:
    """Build lower-triangular causal mask ``[1, 1, L, L]`` for attention."""
    l = int(seq_len)
    positions = ctx.arange(0, l, dtype=_int32_dtype(ctx))
    q_pos = positions.reshape(1, 1, l, 1)
    k_pos = positions.reshape(1, 1, 1, l)
    keep = q_pos >= k_pos
    shape = (1, 1, l, l)
    neg = ctx.full(shape, float(neg_value), dtype=dtype)
    return ctx.where(keep, ctx.zeros(shape, dtype=dtype), neg)


def build_padding_attention_bias(
    ctx: Any,
    attention_mask: Any,
    seq_len: int,
    dtype: Any,
    *,
    valid_value: int = 1,
    neg_value: float = -1e9,
) -> Any:
    """Build ``[B,1,1,L]`` additive attention bias from binary token mask."""
    b = int(attention_mask.shape[0])
    l = int(seq_len)
    return ctx.where(
        attention_mask[:, None, None, :l] == valid_value,
        ctx.zeros((b, 1, 1, l), dtype=dtype),
        ctx.full((b, 1, 1, l), float(neg_value), dtype=dtype),
    )


def resolve_blhd_attention_mask(
    ctx: Any,
    q: Any,
    *,
    mask: Any | None = None,
    causal: bool = False,
    q_lens: Any | None = None,
    k_lens: Any | None = None,
    dtype: Any | None = None,
    neg_value: float = -1e9,
) -> Any | None:
    """Resolve additive mask for ``[B, L, H, D]`` attention."""
    if mask is not None:
        return mask
    target_dtype = dtype if dtype is not None else getattr(q, "dtype", None)
    if target_dtype is None:
        d = getattr(ctx, "float32", None)
        target_dtype = d() if callable(d) else d
    if target_dtype is None:
        raise RuntimeError("Cannot resolve attention mask dtype for BLHD attention.")
    seq_len = int(q.shape[1])
    if causal:
        return build_causal_attention_mask(ctx, seq_len, target_dtype, neg_value=neg_value)
    lens = k_lens if k_lens is not None else q_lens
    if lens is None:
        return None
    lens_i32 = (
        lens.astype(_int32_dtype(ctx))
        if hasattr(lens, "astype")
        else ctx.array(lens, dtype=_int32_dtype(ctx))
    )
    return build_key_padding_mask_from_lengths(
        ctx,
        lens_i32,
        seq_len,
        target_dtype,
        neg_value=neg_value,
    )


def build_bidirectional_bool_attention_mask(ctx: Any, token_mask: Any) -> Any:
    """Build symmetric bool mask ``[B,1,S,S]`` from token keep mask ``[B,S]``."""
    m = token_mask.astype(_bool_dtype(ctx))
    b = int(m.shape[0])
    s = int(m.shape[1])
    m1 = ctx.reshape(m, (b, 1, 1, s))
    m1 = ctx.broadcast_to(m1, (b, 1, s, s))
    m2 = ctx.transpose(m1, (0, 1, 3, 2))
    return m1 & m2


def left_pad_token_mask(ctx: Any, token_mask: Any, total_len: int) -> Any:
    """Left-pad token mask ``[B,S]`` to ``[B,total_len]`` with valid tokens."""
    mask = token_mask.astype(_bool_dtype(ctx))
    pad_len = int(total_len) - int(mask.shape[1])
    if pad_len <= 0:
        return mask
    pad = ctx.ones((int(mask.shape[0]), pad_len), dtype=_bool_dtype(ctx))
    return ctx.concat([pad, mask], axis=1)


def apply_binary_mask_bias(
    ctx: Any,
    attn_mask: Any,
    mask: Any,
    *,
    valid_value: int = 1,
    neg_value: float = float("-inf"),
) -> Any:
    """Apply binary keep-mask to attention bias tensor via ``where``."""
    m = mask
    while len(m.shape) < 4:
        m = ctx.expand_dims(m, axis=1)
    return ctx.where(m == valid_value, attn_mask, float(neg_value))


def build_causal_with_padding_bias(
    ctx: Any,
    attention_mask: Any | None,
    seq_len: int,
    dtype: Any,
    *,
    valid_value: int = 1,
    neg_value: float = -1e9,
    batch_size: int | None = None,
) -> Any:
    """Build additive 4D mask ``[B,1,S,S]`` = causal bias + optional padding bias."""
    s = int(seq_len)
    if batch_size is None:
        if attention_mask is not None:
            b = int(attention_mask.shape[0])
        else:
            b = 1
    else:
        b = int(batch_size)
    causal = ctx.broadcast_to(build_causal_attention_mask(ctx, s, dtype, neg_value=neg_value), (b, 1, s, s))
    if attention_mask is None:
        return causal
    pad = build_padding_attention_bias(
        ctx,
        attention_mask,
        s,
        dtype,
        valid_value=valid_value,
        neg_value=neg_value,
    )
    return causal + pad


def build_causal_with_offset_bias(
    ctx: Any,
    seq_len: int,
    offset: int = 0,
    *,
    dtype: Any | None = None,
    neg_value: float = float("-inf"),
) -> Any:
    """Build causal additive mask ``[1,1,S,T]`` where ``T = S + offset``."""
    s = int(seq_len)
    off = int(offset)
    total = s + off
    if dtype is None:
        dtype = getattr(ctx, "float32", None)
        dtype = dtype() if callable(dtype) else dtype
    row_indices = ctx.arange(s, dtype=_int32_dtype(ctx))[:, None]
    col_indices = ctx.arange(total, dtype=_int32_dtype(ctx))[None, :]
    causal = col_indices <= (row_indices + off)
    mask = ctx.where(causal, 0.0, float(neg_value))
    if dtype is not None and hasattr(mask, "astype"):
        mask = mask.astype(dtype)
    return mask[None, None, :, :]


def build_window_with_padding_bias(
    ctx: Any,
    seq_len: int,
    dtype: Any,
    *,
    attention_mask: Any | None = None,
    sliding_window: int | None = None,
    neg_value: float = -1e9,
    valid_value: int = 1,
) -> Any:
    """Build additive 4D mask ``[B,1,S,S]`` for full/sliding self-attention."""
    s = int(seq_len)
    idx = ctx.arange(s, dtype=_int32_dtype(ctx))
    diff = idx[:, None] - idx[None, :]
    valid = ctx.ones((s, s), dtype=_bool_dtype(ctx))
    if sliding_window is not None:
        valid = valid & (ctx.abs(diff) <= int(sliding_window))
    valid = ctx.expand_dims(ctx.expand_dims(valid, 0), 0)
    min_val = ctx.full((), float(neg_value), dtype=dtype)
    mask = ctx.where(valid, ctx.zeros(valid.shape, dtype=dtype), min_val)
    if attention_mask is not None:
        pad = build_padding_attention_bias(
            ctx,
            attention_mask,
            s,
            dtype,
            valid_value=valid_value,
            neg_value=neg_value,
        )
        mask = mask + pad
    return mask


def build_window_with_padding_bias_torch(
    seq_len: int,
    dtype: Any,
    device: Any,
    *,
    attention_mask: Any | None = None,
    sliding_window: int | None = None,
    neg_value: float = -1e9,
    valid_value: int = 1,
) -> Any:
    """Build additive 4D mask ``[B,1,S,S]`` for torch full/sliding self-attention."""
    torch = importlib.import_module("torch")
    s = int(seq_len)
    idx = torch.arange(s, device=device)
    diff = idx[:, None] - idx[None, :]
    valid = torch.ones((s, s), dtype=torch.bool, device=device)
    if sliding_window is not None:
        valid = valid & (diff.abs() <= int(sliding_window))
    zeros = torch.zeros((1, 1, s, s), dtype=dtype, device=device)
    neg = torch.full((1, 1, s, s), float(neg_value), dtype=dtype, device=device)
    mask = torch.where(valid.view(1, 1, s, s), zeros, neg)
    if attention_mask is not None:
        b = int(attention_mask.shape[0])
        keep = attention_mask[:, None, None, :s] == int(valid_value)
        pad_zeros = torch.zeros((b, 1, 1, s), dtype=dtype, device=device)
        pad_neg = torch.full((b, 1, 1, s), float(neg_value), dtype=dtype, device=device)
        mask = mask + torch.where(keep, pad_zeros, pad_neg)
    return mask


def build_frame_prefix_causal_bias(
    ctx: Any,
    n_frame: int,
    n_hw: int,
    batch_size: int,
    *,
    dtype: Any,
    neg_value: float = float("-inf"),
) -> Any:
    """Build frame-wise causal mask for flattened video tokens ``[B,1,S,S]``.

    Token ``i`` (in frame ``fi``) can attend to all tokens from frames ``<= fi``.
    """
    frames = int(n_frame)
    hw = int(n_hw)
    b = int(batch_size)
    seq_len = frames * hw
    token_idx = ctx.arange(seq_len, dtype=_int32_dtype(ctx))
    frame_idx = token_idx // hw
    j_pos = ctx.arange(seq_len, dtype=_int32_dtype(ctx))[None, :]
    i_frame = frame_idx[:, None]
    allowed = j_pos < (i_frame + 1) * hw
    mask2d = ctx.where(
        allowed,
        ctx.zeros((seq_len, seq_len), dtype=dtype),
        ctx.full((seq_len, seq_len), float(neg_value), dtype=dtype),
    )
    return ctx.broadcast_to(ctx.expand_dims(mask2d, (0, 1)), (b, 1, seq_len, seq_len))


def repeat_kv_heads_mx(ops: Any, x: Any, n_rep: int) -> Any:
    """Repeat KV heads for MLX-style tensors ``[B, H_kv, L, D]``."""
    rep = int(n_rep)
    if rep == 1:
        return x
    b, n_kv, l, d = (int(x.shape[i]) for i in range(4))
    x = ops.expand_dims(x, axis=2)
    x = ops.broadcast_to(x, (b, n_kv, rep, l, d))
    return x.reshape(b, n_kv * rep, l, d)


def repeat_kv_heads_torch(x: Any, n_rep: int) -> Any:
    """Repeat KV heads for torch tensors ``[B, H_kv, L, D]``."""
    rep = int(n_rep)
    if rep == 1:
        return x
    b, n_kv, l, d = (int(x.shape[i]) for i in range(4))
    x = x.unsqueeze(2)
    x = x.expand(b, n_kv, rep, l, d)
    return x.reshape(b, n_kv * rep, l, d)


def scaled_dot_product_attention_bhsd_mx(
    ops: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    mask: Any | None = None,
    compute_dtype: Any | None = None,
    out_dtype: Any | None = None,
    attention_backend: str = "auto",
) -> Any:
    """MLX SDPA for ``[B, H, S, D]`` with optional mlx-mfa dispatch."""
    q_in, k_in, v_in = q, k, v
    if compute_dtype is not None:
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

    if str(attention_backend or "auto").strip().lower() != "mlx":
        from backend.engine.common.ops.mfa_bridge_mlx import flash_attention

        out = flash_attention(q, k, v, backend=attention_backend, scale=scale, mask=mask)
    elif hasattr(ops, "fast") and hasattr(ops.fast, "scaled_dot_product_attention"):
        out = ops.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    else:
        from backend.engine.common.ops.mfa_bridge_mlx import flash_attention

        out = flash_attention(q, k, v, backend=attention_backend, scale=scale, mask=mask)
    if out_dtype is not None:
        return out.astype(out_dtype)
    if compute_dtype is not None and hasattr(q_in, "dtype"):
        return out.astype(q_in.dtype)
    return out


def scaled_dot_product_attention_bhsd_torch(
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float | None = None,
    mask: Any | None = None,
    out_dtype: Any | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Any:
    """Torch SDPA for ``[B, H, S, D]`` tensors with optional output dtype."""
    torch = importlib.import_module("torch")
    kwargs: dict[str, Any] = {
        "attn_mask": mask,
        "dropout_p": float(dropout_p),
        "is_causal": bool(is_causal),
    }
    if scale is not None:
        kwargs["scale"] = scale
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)
    if out_dtype is not None:
        return out.to(out_dtype)
    return out


def rotate_half_torch(x: Any) -> Any:
    """GPT-style split-half RoPE rotation for Torch tensors."""
    torch = importlib.import_module("torch")
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def attention_blhd(
    ctx: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    mask: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Scaled dot-product attention for ``[B, L, H, D]`` tensors."""
    if dtype is None:
        dtype = getattr(q, "dtype", None)
    if dtype is not None:
        q = q.astype(dtype)
        k = k.astype(dtype)
        v = v.astype(dtype)
    qh = ctx.permute(q, (0, 2, 1, 3))
    kh = ctx.permute(k, (0, 2, 1, 3))
    vh = ctx.permute(v, (0, 2, 1, 3))
    out = ctx.attention(qh, kh, vh, scale=scale, mask=mask)
    return ctx.permute(out, (0, 2, 1, 3))


def wan_blhd_attention(
    ctx: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    q_lens: Any | None = None,
    k_lens: Any | None = None,
    mask: Any | None = None,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dtype: Any | None = None,
) -> Any:
    """Wan layout attention on ``[B, L, num_heads, head_dim]`` tensors."""
    if window_size != (-1, -1):
        raise RuntimeError(
            f"Wan attention does not support sliding window_size={window_size!r}"
        )
    if q_scale is not None:
        q = q * q_scale
    target_dtype = dtype if dtype is not None else getattr(q, "dtype", ctx.float32())
    scale = softmax_scale
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5
    mask = resolve_blhd_attention_mask(
        ctx,
        q,
        mask=mask,
        causal=causal,
        q_lens=q_lens,
        k_lens=k_lens,
        dtype=target_dtype,
        neg_value=-1e9,
    )
    return attention_blhd(ctx, q, k, v, scale=scale, mask=mask, dtype=target_dtype)


wan_attention = wan_blhd_attention


def attention_bhsd(
    ctx: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    mask: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Scaled dot-product attention for ``[B, H, S, D]`` tensors."""
    if dtype is None:
        dtype = getattr(q, "dtype", None)
    if dtype is not None:
        q = q.astype(dtype)
        k = k.astype(dtype)
        v = v.astype(dtype)
    return ctx.attention(q, k, v, scale=scale, mask=mask)


def attention_bhsd_to_blhd(
    ctx: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    mask: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Scaled dot-product attention from ``[B,H,S,D]`` to ``[B,S,H,D]``."""
    out = attention_bhsd(ctx, q, k, v, scale=scale, mask=mask, dtype=dtype)
    return ctx.permute(out, (0, 2, 1, 3))


class SelfAttention:
    """标准自注意力 (QKV 投影 + RoPE + Scaled Dot-Product)。

    用于: Flux1 / Flux2 / Qwen / FIBO / Z-Image 图像模型的空间注意力。
    """

    def __init__(self, dim: int, num_heads: int, ctx: Any,
                 qk_norm: bool = True, qkv_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope_cos=None, rope_sin=None,
                mask=None) -> Any:
        ctx = self.ctx
        B, N, C = x.shape[0], x.shape[1], self.dim

        qkv = self.qkv(x)
        qkv = ctx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = ctx.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(ctx, q, rope_cos, rope_sin)
            k = _apply_rope(ctx, k, rope_cos, rope_sin)

        attn_out = ctx.attention(q, k, v, scale=self.scale, mask=mask)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B, N, C))
        return self.proj(attn_out)


class CrossAttention:
    """交叉注意力 (Q 来自 latents, KV 来自 text_embeds)。

    用于: Flux1 MM-DiT 的 cross-attn / LTX / Wan 的 text cross-attn。
    """

    def __init__(self, query_dim: int, ctx_dim: int, num_heads: int, ctx: Any,
                 out_dim: Optional[int] = None, qk_norm: bool = True,
                 qkv_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        out_dim = out_dim or query_dim

        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k = nn.Linear(ctx_dim, query_dim, bias=qkv_bias)
        self.v = nn.Linear(ctx_dim, query_dim, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = nn.Linear(query_dim, out_dim)

    def forward(self, x, context, mask=None) -> Any:
        ctx = self.ctx
        B, N, C = x.shape[0], x.shape[1], self.q.in_features

        q = ctx.reshape(self.q(x), (B, N, self.num_heads, self.head_dim))
        q = ctx.permute(q, (0, 2, 1, 3))

        ctx_len = context.shape[1]
        k = ctx.reshape(self.k(context), (B, ctx_len, self.num_heads, self.head_dim))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.reshape(self.v(context), (B, ctx_len, self.num_heads, self.head_dim))
        v = ctx.permute(v, (0, 2, 1, 3))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_out = ctx.attention(q, k, v, scale=self.scale, mask=mask)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B, N, self.q.in_features))
        return self.proj(attn_out)


class TemporalAttention:
    """时序自注意力 — 沿帧维度做 Self-Attention。

    用于: LTX / Wan 的时序混合层。输入 [B*F, HW, C]，reshape+permute 后做沿帧的 attn。
    """

    def __init__(self, dim: int, num_heads: int, num_frames: int, ctx: Any,
                 qk_norm: bool = True, qkv_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope_cos=None, rope_sin=None) -> Any:
        ctx = self.ctx
        # x: [B*F, H*W, C] → [B, F, H*W, C]
        BF, HW, C = x.shape[0], x.shape[1], self.dim
        F = self.num_frames
        B = BF // F
        H = W = int(HW ** 0.5)  # Square image assumption; flexible

        x_reshaped = ctx.reshape(x, (B, F, HW, C))
        x_t = ctx.permute(x_reshaped, (0, 2, 1, 3))  # [B, HW, F, C]
        x_t = ctx.reshape(x_t, (B * HW, F, C))

        qkv = self.qkv(x_t)
        qkv = ctx.reshape(qkv, (B * HW, F, 3, self.num_heads, self.head_dim))
        qkv = ctx.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(ctx, q, rope_cos, rope_sin)
            k = _apply_rope(ctx, k, rope_cos, rope_sin)

        attn_out = ctx.attention(q, k, v, scale=self.scale)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B * HW, F, C))
        proj_out = self.proj(attn_out)
        proj_out = ctx.reshape(proj_out, (B, HW, F, C))
        proj_out = ctx.permute(proj_out, (0, 2, 1, 3))  # [B, F, HW, C]
        proj_out = ctx.reshape(proj_out, (BF, HW, C))
        return proj_out


def _apply_rope(ctx, x, cos, sin):
    """将 RoPE 应用于 Q/K 张量。

    x: [B, num_heads, seq, head_dim]
    cos/sin: [1, 1, seq, rope_dim]
    """
    rope_dim = cos.shape[-1]
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rotated = x_rope * cos + rotate_half(ctx, x_rope) * sin
    return ctx.concat([x_rotated, x_pass], axis=-1)


def rotate_half(ctx: Any, x: Any) -> Any:
    """Rotate half the hidden dims of the input (GPT-style split-half RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ctx.concat([-x2, x1], axis=-1)


_rotate_half = rotate_half


def apply_rope_interleaved_real(ctx, x, cos, sin):
    """diffusers/Flux-style RoPE: ``out = x * cos + rotate_interleaved(x) * sin``.

    Matches ``apply_rotary_emb`` with ``use_real_unbind_dim=-1`` (adjacent dimension pairs),
    **not** GPT-style split-half ``rotate_half``.

    x: [B, num_heads, seq, head_dim]
    cos/sin: broadcastable to x (e.g. [1, 1, seq, rope_dim])
    """
    rope_dim = int(cos.shape[-1])
    x_tail = x[..., rope_dim:] if x.shape[-1] > rope_dim else None
    xr = x[..., :rope_dim]
    xf = xr.astype(ctx.float32())
    cf = cos.astype(ctx.float32())
    sf = sin.astype(ctx.float32())
    sh = xf.shape
    x2 = ctx.reshape(xf, (*sh[:-1], -1, 2))
    real = x2[..., 0]
    imag = x2[..., 1]
    rot = ctx.stack([-imag, real], axis=-1)
    rot = ctx.reshape(rot, (*sh[:-1], rope_dim))
    out = xf * cf + rot * sf
    out = out.astype(x.dtype)
    if x_tail is not None and x_tail.shape[-1] > 0:
        return ctx.concat([out, x_tail], axis=-1)
    return out


def apply_rope_real_unbind_dim2(ctx, x, cos, sin):
    """diffusers CogView4 / Stable Audio RoPE — ``use_real_unbind_dim=-2``."""
    rope_dim = int(cos.shape[-1])
    x_tail = x[..., rope_dim:] if x.shape[-1] > rope_dim else None
    xr = x[..., :rope_dim]
    xf = xr.astype(ctx.float32())
    cf = cos.astype(ctx.float32())
    sf = sin.astype(ctx.float32())
    sh = xf.shape
    half = rope_dim // 2
    x2 = ctx.reshape(xf, (*sh[:-1], 2, half))
    real = x2[..., 0, :]
    imag = x2[..., 1, :]
    rot = ctx.concat([-imag, real], axis=-1)
    out = xf * cf + rot * sf
    out = out.astype(x.dtype)
    if x_tail is not None and x_tail.shape[-1] > 0:
        return ctx.concat([out, x_tail], axis=-1)
    return out
