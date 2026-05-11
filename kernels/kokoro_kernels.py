"""
Triton kernels for the Kokoro TTS decoder.

Kokoro decoder architecture:
  - DecoderBlocks: Conv1d + AdaIN + LeakyReLU
  - Generator: ConvTranspose1d upsampling + Snake ResBlocks
  - Each ResBlock: 3 iterations of (AdaIN → Snake → dilated Conv1d) × 2

Key dimensions:
  Channels: 128, 256, 512, 1024
  Sequence length: ~30-100 (input) → ~18000-100000 (after upsample)
  Conv kernel sizes: 1, 3, 5, 7, 11, 12, 20
  Style dim: 128

All inputs/outputs fp16, compute in fp32.
"""
import triton
import triton.language as tl


# ─── Snake activation: x + sin²(αx)/α ──────────────────────────────────────

@triton.jit
def snake_activation(
    x_ptr, alpha_ptr, out_ptr,
    n_elements, n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Snake activation: out = x + sin²(α*x)/α

    x: [1, C, T] flattened = C*T elements
    alpha: [1, C, 1] — one alpha per channel, broadcast over T
    out: [1, C, T] same layout

    Each element at position i maps to channel c = i / seq_len.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # Determine channel index for each element: layout is [C, T] so channel = offset // seq_len
    ch_idx = offsets // seq_len
    alpha = tl.load(alpha_ptr + ch_idx, mask=mask).to(tl.float32)

    # snake(x) = x + sin²(α*x)/α
    ax = alpha * x
    sin_val = tl.sin(ax)
    sin_sq = sin_val * sin_val
    inv_alpha = 1.0 / (alpha + 1e-9)
    out = x + sin_sq * inv_alpha

    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


# ─── Fused AdaIN: instance norm + affine with style ─────────────────────────

@triton.jit
def adain_fused(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    n_channels, seq_len,
    BLOCK_T: tl.constexpr,
):
    """Fused AdaIN: instance normalization + (gamma+1)*norm + beta.

    x: [1, C, T] — one program per channel
    gamma: [1, C, 1] — style scale (added to 1.0)
    beta: [1, C, 1] — style bias
    out: [1, C, T]

    For each channel c:
      mean = mean(x[c, :])
      var = var(x[c, :])
      norm = (x[c, :] - mean) / sqrt(var + eps)
      out[c, :] = (gamma[c] + 1) * norm + beta[c]
    """
    ch_idx = tl.program_id(0)
    if ch_idx >= n_channels:
        return

    base = ch_idx * seq_len
    t_offsets = tl.arange(0, BLOCK_T)
    mask = t_offsets < seq_len

    # Load entire channel timeseries
    x = tl.load(x_ptr + base + t_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / seq_len
    x_centered = x - mean
    x_sq = tl.where(mask, x_centered * x_centered, 0.0)
    var = tl.sum(x_sq, axis=0) / seq_len
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    normed = x_centered * rstd

    # Apply style: (gamma + 1) * norm + beta
    gamma = tl.load(gamma_ptr + ch_idx).to(tl.float32)
    beta = tl.load(beta_ptr + ch_idx).to(tl.float32)
    out = (gamma + 1.0) * normed + beta

    tl.store(out_ptr + base + t_offsets, out.to(tl.float16), mask=mask)


# ─── Fused Snake + Conv1d (depthwise-style for common case) ─────────────────
# This is for the common pattern: snake → conv1d where channels match

@triton.jit
def leaky_relu_fp16(
    x_ptr, out_ptr,
    n_elements,
    SLOPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """LeakyReLU: out = max(x, 0) + slope * min(x, 0)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    out = tl.where(x > 0.0, x, x * SLOPE)
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


# ─── Conv1d: one threadgroup per output channel, BLOCK_T time steps ─────────
# Simple approach: each threadgroup handles one output channel across BLOCK_T outputs.
# Each thread computes one output element by accumulating over C_in * K.

@triton.jit
def conv1d_simple(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    """Conv1d: one threadgroup per (c_out, t_block).

    x: [C_in, T_in] (fp16)
    w: [C_out, C_in, K] (fp16) — standard layout
    bias: [C_out] (fp16)
    out: [C_out, T_out] (fp16)

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1)

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for c in range(C_in):
        for ki in range(K):
            # Input position for this kernel tap
            t_in_pos = t_offs * stride - padding + ki * dilation
            valid = t_mask & (t_in_pos >= 0) & (t_in_pos < T_in)

            # Load input: x[c, t_in_pos]
            x_idx = c * T_in + t_in_pos
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)

            # Load weight: w[c_out_idx, c, ki]
            w_idx = c_out_idx * C_in * K + c * K + ki
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += x_val * w_val

    # Add bias
    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    # Store
    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc.to(tl.float16), mask=t_mask)


# ─── Conv1d with K as constexpr: tiled over output channels ──────────────────
# Multiple output channels per threadgroup share input loads.

@triton.jit
def conv1d_k(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out,
    stride, padding, dilation,
    BLOCK_T: tl.constexpr,
    K: tl.constexpr,
):
    """Conv1d with compile-time K. Each threadgroup computes BLOCK_T outputs
    for one output channel, with K unrolled.

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1)

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for c in range(C_in):
        w_base = c_out_idx * C_in * K + c * K
        x_base = c * T_in
        for ki in tl.static_range(K):
            t_in_pos = t_offs * stride - padding + ki * dilation
            valid = t_mask & (t_in_pos >= 0) & (t_in_pos < T_in)

            x_val = tl.load(x_ptr + x_base + t_in_pos, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_base + ki).to(tl.float32)
            acc += x_val * w_val

    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc.to(tl.float16), mask=t_mask)


# ─── Fused Snake + AdaIN (most common pattern in resblocks) ─────────────────

@triton.jit
def adain_snake_fused(
    x_ptr, gamma_ptr, beta_ptr, alpha_ptr, out_ptr,
    n_channels, seq_len,
    BLOCK_T: tl.constexpr,
):
    """Fused AdaIN + Snake: first normalize, then apply snake activation.

    For each channel c:
      norm = AdaIN(x[c, :], gamma[c], beta[c])
      out[c, :] = snake(norm, alpha[c])

    This fuses two memory-bound operations into one pass.
    BLOCK_T must be <= max threads per group (1024 on D3D12).
    """
    ch_idx = tl.program_id(0)
    if ch_idx >= n_channels:
        return

    base = ch_idx * seq_len
    t_offsets = tl.arange(0, BLOCK_T)
    mask = t_offsets < seq_len

    # Load entire channel timeseries
    x = tl.load(x_ptr + base + t_offsets, mask=mask, other=0.0).to(tl.float32)

    # AdaIN: instance norm + style
    mean = tl.sum(x, axis=0) / seq_len
    x_centered = x - mean
    x_sq = tl.where(mask, x_centered * x_centered, 0.0)
    var = tl.sum(x_sq, axis=0) / seq_len
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    normed = x_centered * rstd

    gamma = tl.load(gamma_ptr + ch_idx).to(tl.float32)
    beta = tl.load(beta_ptr + ch_idx).to(tl.float32)
    styled = (gamma + 1.0) * normed + beta

    # Snake: x + sin²(α*x)/α
    alpha = tl.load(alpha_ptr + ch_idx).to(tl.float32)
    ax = alpha * styled
    sin_val = tl.sin(ax)
    sin_sq = sin_val * sin_val
    inv_alpha = 1.0 / (alpha + 1e-9)
    out = styled + sin_sq * inv_alpha

    tl.store(out_ptr + base + t_offsets, out.to(tl.float16), mask=mask)


@triton.jit
def adain_snake_looped(
    x_ptr, gamma_ptr, beta_ptr, alpha_ptr, out_ptr,
    n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ: tl.constexpr,
):
    """Looped AdaIN + Snake for seq_len > 1024 (D3D12 thread group limit).

    Two passes:
      Pass 1: compute mean and variance via loop over BLOCK_SIZE chunks.
      Pass 2: apply normalization + style + snake via same loop.

    Each thread group handles one channel. Grid: [n_channels, 1, 1].
    """
    ch_idx = tl.program_id(0)
    if ch_idx >= n_channels:
        return

    base = ch_idx * seq_len
    tid = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute mean
    acc_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offsets = start + tid
        mask = offsets < seq_len
        vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc_sum += tl.where(mask, vals, 0.0)
    total_sum = tl.sum(acc_sum, axis=0)
    mean = total_sum / seq_len

    # Pass 1b: compute variance
    acc_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offsets = start + tid
        mask = offsets < seq_len
        vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = vals - mean
        acc_var += tl.where(mask, diff * diff, 0.0)
    total_var = tl.sum(acc_var, axis=0)
    var = total_var / seq_len
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Load style params
    gamma = tl.load(gamma_ptr + ch_idx).to(tl.float32)
    beta = tl.load(beta_ptr + ch_idx).to(tl.float32)
    alpha = tl.load(alpha_ptr + ch_idx).to(tl.float32)
    inv_alpha = 1.0 / (alpha + 1e-9)
    scale = (gamma + 1.0) * rstd

    # Pass 2: normalize + style + snake
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offsets = start + tid
        mask = offsets < seq_len
        vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        normed = (vals - mean) * scale + beta
        ax = alpha * normed
        sin_val = tl.sin(ax)
        result = normed + sin_val * sin_val * inv_alpha
        tl.store(out_ptr + base + offsets, result.to(tl.float16), mask=mask)


@triton.jit
def instance_norm_stats(
    x_ptr, stats_ptr,
    n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ: tl.constexpr,
):
    """Compute per-channel mean and variance for instance normalization.

    Grid: [n_channels, 1, 1]. Each group handles one channel.
    Output: stats_ptr[ch*2] = mean, stats_ptr[ch*2+1] = rstd
    """
    ch_idx = tl.program_id(0)
    if ch_idx >= n_channels:
        return

    base = ch_idx * seq_len
    tid = tl.arange(0, BLOCK_SIZE)

    # Compute mean
    acc_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offsets = start + tid
        mask = offsets < seq_len
        vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc_sum += tl.where(mask, vals, 0.0)
    total_sum = tl.sum(acc_sum, axis=0)
    mean = total_sum / seq_len

    # Compute variance
    acc_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offsets = start + tid
        mask = offsets < seq_len
        vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = vals - mean
        acc_var += tl.where(mask, diff * diff, 0.0)
    total_var = tl.sum(acc_var, axis=0)
    var = total_var / seq_len
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Store stats
    tl.store(stats_ptr + ch_idx * 2, mean)
    tl.store(stats_ptr + ch_idx * 2 + 1, rstd)


@triton.jit
def norm_style_snake(
    x_ptr, stats_ptr, gamma_ptr, beta_ptr, alpha_ptr, out_ptr,
    n_elements, n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise: read per-channel stats, normalize, apply style, snake.

    Grid: [cdiv(n_elements, BLOCK_SIZE), 1, 1].
    stats_ptr[ch*2] = mean, stats_ptr[ch*2+1] = rstd (f32 buffer).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Determine channel index for each element
    ch_idx = offsets // seq_len

    # Load per-channel stats (mean, rstd)
    mean = tl.load(stats_ptr + ch_idx * 2, mask=mask, other=0.0)
    rstd = tl.load(stats_ptr + ch_idx * 2 + 1, mask=mask, other=0.0)

    # Load per-channel style params
    gamma = tl.load(gamma_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)

    # Normalize + style
    normed = (x - mean) * rstd
    styled = (gamma + 1.0) * normed + beta

    # Snake: x + sin²(αx)/α
    ax = alpha * styled
    sin_val = tl.sin(ax)
    sin_sq = sin_val * sin_val
    inv_alpha = 1.0 / (alpha + 1e-9)
    result = styled + sin_sq * inv_alpha

    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)


# ─── Im2col for conv1d: rearrange input for matmul-based convolution ──────────

@triton.jit
def im2col_conv1d(
    x_ptr, out_ptr,
    C_in, T_in, T_out, K,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    """Im2col for conv1d: reshape [C_in, T_in] → [C_in*K, T_out].

    For each output column t_out, gather K values from each input channel:
      out[c*K + ki, t_out] = x[c, t_out*stride - padding + ki*dilation]

    Grid: [cdiv(C_in * K * T_out, BLOCK_SIZE), 1, 1]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = C_in * K * T_out
    mask = offsets < n_elements

    # Map flat index → (row, col) in output [C_in*K, T_out]
    t_out_idx = offsets % T_out
    row = offsets // T_out  # row in [0, C_in*K)
    c = row // K
    ki = row % K

    # Input position
    t_in_pos = t_out_idx * stride - padding + ki * dilation
    valid = mask & (t_in_pos >= 0) & (t_in_pos < T_in)

    x_idx = c * T_in + t_in_pos
    val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


# ─── Im2col with fused leaky_relu on input ────────────────────────────────────

@triton.jit
def im2col_conv1d_act(
    x_ptr, out_ptr,
    C_in, T_in, T_out, K,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """Im2col for conv1d with optional activation on input values.

    Grid: [cdiv(C_in * K * T_out, BLOCK_SIZE), 1, 1]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = C_in * K * T_out
    mask = offsets < n_elements

    t_out_idx = offsets % T_out
    row = offsets // T_out
    c = row // K
    ki = row % K

    t_in_pos = t_out_idx * stride - padding + ki * dilation
    valid = mask & (t_in_pos >= 0) & (t_in_pos < T_in)

    x_idx = c * T_in + t_in_pos
    val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
    if ACTIVATION:
        val = ACTIVATION(val)
    tl.store(out_ptr + offsets, val.to(tl.float16), mask=mask)


# ─── Element-wise add: out = a + b ────────────────────────────────────────────

@triton.jit
def elementwise_add(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise add: out[i] = a[i] + b[i]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    out = a + b
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


# ─── Element-wise scale: out = x * scalar ─────────────────────────────────────

@triton.jit
def elementwise_scale_third(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise scale by 1/3: out[i] = x[i] / 3"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    out = x * 0.3333333333333333
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


# ─── Reflection pad 1D: pad_left=1, pad_right=0 ──────────────────────────────

@triton.jit
def reflection_pad1d_left1(
    x_ptr, out_ptr,
    n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Reflection pad1d with pad_left=1, pad_right=0.

    x: [C, T] flattened
    out: [C, T+1] flattened

    For each channel c: out[c, 0] = x[c, 1], out[c, 1:T+1] = x[c, 0:T]
    Grid: one program per output element block.
    """
    pid = tl.program_id(0)
    out_len = seq_len + 1
    n_elements = n_channels * out_len
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Map output flat index to (channel, time)
    ch = offsets // out_len
    t_out = offsets % out_len

    # Map output time to input time (reflection: out[0] = x[1], rest = x[t-1])
    t_in = tl.where(t_out == 0, 1, t_out - 1)

    # Load from input
    x_idx = ch * seq_len + t_in
    val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, val, mask=mask)


# ─── Inline activation helpers (passed as ACTIVATION constexpr) ───────────────

@triton.jit
def leaky_relu_01_act(x):
    return tl.where(x > 0.0, x, x * 0.1)

@triton.jit
def leaky_relu_001_act(x):
    return tl.where(x > 0.0, x, x * 0.01)


# ─── Conv1d with optional fused input activation ─────────────────────────────

@triton.jit
def conv1d_act(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding, dilation,
    BLOCK_T: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """Conv1d with optional activation applied to input on-the-fly.

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1)

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for c in range(C_in):
        for ki in range(K):
            t_in_pos = t_offs * stride - padding + ki * dilation
            valid = t_mask & (t_in_pos >= 0) & (t_in_pos < T_in)

            x_idx = c * T_in + t_in_pos
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
            if ACTIVATION:
                x_val = ACTIVATION(x_val)

            w_idx = c_out_idx * C_in * K + c * K + ki
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += x_val * w_val

    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc.to(tl.float16), mask=t_mask)


# ─── ConvTranspose1d with optional fused input activation ─────────────────────

@triton.jit
def conv_transpose1d_act(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding,
    BLOCK_T: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """ConvTranspose1d with optional activation applied to input on-the-fly.

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1)

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for c in range(C_in):
        for ki in range(K):
            numerator = t_offs + padding - ki
            valid_div = (numerator % stride) == 0
            t_in_pos = numerator // stride
            valid = t_mask & valid_div & (t_in_pos >= 0) & (t_in_pos < T_in)

            x_idx = c * T_in + t_in_pos
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
            if ACTIVATION:
                x_val = ACTIVATION(x_val)

            w_idx = c * C_out * K + c_out_idx * K + ki
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += x_val * w_val

    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc.to(tl.float16), mask=t_mask)


# ─── ConvTranspose1d: one threadgroup per output channel ─────────────────────

@triton.jit
def conv_transpose1d_simple(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding,
    BLOCK_T: tl.constexpr,
):
    """ConvTranspose1d: one threadgroup per (c_out, t_block).

    x: [C_in, T_in] (fp16)
    w: [C_in, C_out, K] (fp16)
    bias: [C_out] (fp16)
    out: [C_out, T_out] (fp16)

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1)

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # out[c_out, t_out] = sum_{c_in, k} x[c_in, t_in] * w[c_in, c_out, k]
    # where t_in = (t_out + padding - k) / stride, valid when divisible
    for c in range(C_in):
        for ki in range(K):
            numerator = t_offs + padding - ki
            valid_div = (numerator % stride) == 0
            t_in_pos = numerator // stride
            valid = t_mask & valid_div & (t_in_pos >= 0) & (t_in_pos < T_in)

            x_idx = c * T_in + t_in_pos
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)

            w_idx = c * C_out * K + c_out_idx * K + ki
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += x_val * w_val

    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc.to(tl.float16), mask=t_mask)


# ─── Row-broadcast bias add: out[i] = x[i] + bias[i / n_cols] ──────────────

@triton.jit
def row_bias_add(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i] = x[i] + bias[i // n_cols]. Broadcasts bias along rows (M dim)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    row_idx = offsets // n_cols
    b = tl.load(bias_ptr + row_idx, mask=mask)
    tl.store(out_ptr + offsets, x + b, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
# F32-intermediate variants for D3D12 precision
#
# On D3D12, f16 round-trips between ops compound through instance normalization.
# These kernels keep activations in f32 within resblocks to prevent amplification.
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def conv1d_f32io(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding, dilation,
    BLOCK_T: tl.constexpr,
):
    """Conv1d with f32 input/output, f16 weights.

    x: [C_in, T_in] (f32)
    w: [C_out, C_in, K] (fp16)
    bias: [C_out] (fp16)
    out: [C_out, T_out] (f32)

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1)

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    CK = C_in * K
    for ck in range(CK):
        c = ck // K
        ki = ck % K
        t_in_pos = t_offs * stride - padding + ki * dilation
        valid = t_mask & (t_in_pos >= 0) & (t_in_pos < T_in)

        x_idx = c * T_in + t_in_pos
        x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

        w_idx = c_out_idx * CK + ck
        w_val = tl.load(w_ptr + w_idx).to(tl.float32)

        acc += x_val * w_val

    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc, mask=t_mask)


@triton.jit
def instance_norm_stats_f32in(
    x_ptr, stats_ptr,
    n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ: tl.constexpr,
):
    """Compute per-channel mean and rstd from f32 input (single-pass, for short sequences).

    x: [n_channels, seq_len] (f32)
    stats: [n_channels * 2] (f32) — interleaved [mean0, rstd0, mean1, rstd1, ...]

    Grid: [n_channels, 1, 1]
    """
    ch = tl.program_id(0)
    base = ch * seq_len

    mean_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        mean_acc += x
        var_acc += x * x

    total = tl.sum(mean_acc, axis=0)
    sq_total = tl.sum(var_acc, axis=0)
    mean = total / seq_len
    var = sq_total / seq_len - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(stats_ptr + ch * 2, mean)
    tl.store(stats_ptr + ch * 2 + 1, rstd)


@triton.jit
def instance_norm_stats_f32in_twopass(
    x_ptr, stats_ptr,
    n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ: tl.constexpr,
):
    """Compute per-channel mean and rstd from f32 input (two-pass, for long sequences).

    x: [n_channels, seq_len] (f32)
    stats: [n_channels * 2] (f32) — interleaved [mean0, rstd0, mean1, rstd1, ...]

    Grid: [n_channels, 1, 1]
    """
    ch = tl.program_id(0)
    base = ch * seq_len

    # Pass 1: compute mean
    mean_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        mean_acc += x

    mean = tl.sum(mean_acc, axis=0) / seq_len

    # Pass 2: compute variance as E[(x - mean)^2]
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in tl.static_range(0, MAX_SEQ, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < seq_len
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        diff = tl.where(mask, x - mean, 0.0)
        var_acc += diff * diff

    var = tl.sum(var_acc, axis=0) / seq_len
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(stats_ptr + ch * 2, mean)
    tl.store(stats_ptr + ch * 2 + 1, rstd)


@triton.jit
def norm_style_snake_f32io(
    x_ptr, stats_ptr, gamma_ptr, beta_ptr, alpha_ptr, out_ptr,
    n_elements, n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize + style + snake with f32 input and f32 output.

    x: [n_channels, seq_len] (f32)
    stats: [n_channels * 2] (f32) — mean, rstd per channel
    gamma, beta, alpha: [n_channels] (fp16)
    out: [n_channels, seq_len] (f32)

    Grid: [cdiv(n_elements, BLOCK_SIZE), 1, 1]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    ch_idx = offsets // seq_len

    mean = tl.load(stats_ptr + ch_idx * 2, mask=mask, other=0.0)
    rstd = tl.load(stats_ptr + ch_idx * 2 + 1, mask=mask, other=0.0)

    gamma = tl.load(gamma_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + ch_idx, mask=mask, other=0.0).to(tl.float32)

    normed = (x - mean) * rstd
    styled = (gamma + 1.0) * normed + beta

    ax = alpha * styled
    sin_val = tl.sin(ax)
    sin_sq = sin_val * sin_val
    inv_alpha = 1.0 / (alpha + 1e-9)
    result = styled + sin_sq * inv_alpha

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def elementwise_add_f32(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise add of f32 buffers: out[i] = a[i] + b[i]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.jit
def elementwise_scale_third_f32(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise scale f32 by 1/3: out[i] = x[i] / 3"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 0.3333333333333333, mask=mask)


@triton.jit
def convert_f32_to_f16_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert f32 buffer to f16."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float16), mask=mask)


@triton.jit
def convert_f16_to_f32_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert f16 buffer to f32."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def leaky_relu_f32(
    x_ptr, out_ptr,
    n_elements,
    SLOPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """LeakyReLU on f32 buffers: out = x >= 0 ? x : x * slope."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x >= 0.0, x, x * SLOPE)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def conv_transpose1d_f32io(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding, T_offset,
    BLOCK_T: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """ConvTranspose1d with f32 input/output, f16 weights.

    x: [C_in, T_in] (f32)
    w: [C_in, C_out, K] (fp16)
    bias: [C_out] (fp16)
    out: [C_out, T_out] (f32)

    Grid: [C_out, cdiv(T_out, BLOCK_T), 1]
    T_offset: added to t_block for chunked dispatch (0 for non-chunked).
    """
    c_out_idx = tl.program_id(0)
    t_block = tl.program_id(1) + T_offset

    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T_out

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for c in range(C_in):
        for ki in range(K):
            numerator = t_offs + padding - ki
            valid_div = (numerator % stride) == 0
            t_in_pos = numerator // stride
            valid = t_mask & valid_div & (t_in_pos >= 0) & (t_in_pos < T_in)

            x_idx = c * T_in + t_in_pos
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            if ACTIVATION:
                x_val = ACTIVATION(x_val)

            w_idx = c * C_out * K + c_out_idx * K + ki
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += x_val * w_val

    b = tl.load(bias_ptr + c_out_idx).to(tl.float32)
    acc += b

    out_idx = c_out_idx * T_out + t_offs
    tl.store(out_ptr + out_idx, acc, mask=t_mask)


@triton.jit
def reflection_pad1d_f32(
    x_ptr, out_ptr,
    n_channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Reflection pad1d (pad_left=1, pad_right=0) for f32 buffers.

    x: [C, T] flattened (f32)
    out: [C, T+1] flattened (f32)
    """
    pid = tl.program_id(0)
    out_len = seq_len + 1
    n_elements = n_channels * out_len
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ch = offsets // out_len
    t_out = offsets % out_len

    t_in = tl.where(t_out == 0, 1, t_out - 1)

    x_idx = ch * seq_len + t_in
    val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, val, mask=mask)
