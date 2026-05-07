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


# ─── Conv1d as matmul: im2col approach ──────────────────────────────────────
# Express Conv1d [C_out, C_in, K] on [C_in, T] as a matmul:
#   W[C_out, C_in*K] × im2col[C_in*K, T_out] → out[C_out, T_out]
# Each thread block computes a [BLOCK_M, BLOCK_N] tile of the output.

@triton.jit
def conv1d_matmul(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding, dilation,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Conv1d expressed as tiled matmul with implicit im2col.

    x: [C_in, T_in] (fp16)
    w: [C_out, C_in*K] (fp16, pre-reshaped for matmul)
    bias: [C_out] (fp16)
    out: [C_out, T_out] (fp16)

    Grid: [cdiv(C_out, BLOCK_M) * cdiv(T_out, BLOCK_N), 1, 1]
    Each program computes a BLOCK_M × BLOCK_N tile of output channels × time steps.
    The K dimension of the matmul is C_in * kernel_size.
    """
    n_tiles_n = (T_out + BLOCK_N - 1) // BLOCK_N
    pid_m = tl.program_id(0) // n_tiles_n
    pid_n = tl.program_id(0) % n_tiles_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output channel indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output time indices

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-dim = C_in * kernel_size
    total_k = C_in * K
    for k_start in range(0, total_k, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < total_k

        # Load weight tile: W[offs_m, k_offs] where W is [C_out, C_in*K]
        w_ptrs = w_ptr + offs_m[:, None] * total_k + k_offs[None, :]
        w_mask = (offs_m[:, None] < C_out) & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Load im2col tile: for each (k, t_out), compute which (c_in, t_in) to load
        # k = c_in * K + kernel_pos → c_in = k // K, kernel_pos = k % K
        c_in_idx = k_offs // K
        kernel_pos = k_offs % K
        # t_in = t_out * stride - padding + kernel_pos * dilation
        t_in_base = offs_n * stride - padding  # [BLOCK_N]
        t_in = t_in_base[None, :] + (kernel_pos * dilation)[:, None]  # [BLOCK_K, BLOCK_N]

        # Gather from x: x[c_in_idx, t_in]
        x_idx = c_in_idx[:, None] * T_in + t_in  # [BLOCK_K, BLOCK_N]
        valid = k_mask[:, None] & (t_in >= 0) & (t_in < T_in) & (offs_n[None, :] < T_out)
        x_tile = tl.load(x_ptr + x_idx, mask=valid, other=0.0)  # [BLOCK_K, BLOCK_N]

        # matmul: [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N]
        acc += tl.dot(w, x_tile)

    # Add bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < C_out, other=0.0)
    acc += bias[:, None]

    # Store output tile
    out_ptrs = out_ptr + offs_m[:, None] * T_out + offs_n[None, :]
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < T_out)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


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


# ─── ConvTranspose1d as tiled matmul ─────────────────────────────────────────
# For upsample with stride >> 1 (stride=10, stride=6)
# Each output position gathers sparse contributions from input.
# We express it as: for each output tile, gather the valid (c_in, k) contributions.

@triton.jit
def conv_transpose1d(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    C_in, C_out, T_in, T_out, K,
    stride, padding,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """ConvTranspose1d as tiled output computation.

    x: [C_in, T_in] (fp16)
    w: [C_in, C_out, K] (fp16)
    bias: [C_out] (fp16)
    out: [C_out, T_out] (fp16)

    Grid: [cdiv(C_out, BLOCK_M) * cdiv(T_out, BLOCK_N), 1, 1]
    Each program computes a BLOCK_M × BLOCK_N tile of the output.
    """
    n_tiles_n = (T_out + BLOCK_N - 1) // BLOCK_N
    pid_m = tl.program_id(0) // n_tiles_n
    pid_n = tl.program_id(0) % n_tiles_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output channel indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output time indices

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For conv_transpose: out[c_out, t_out] = sum_{c_in, k} x[c_in, t_in] * w[c_in, c_out, k]
    # where t_in = (t_out + padding - k) / stride, only when divisible
    for c_start in range(0, C_in, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C_in

        for k in range(K):
            # t_in = (offs_n + padding - k) / stride, valid only when (offs_n + padding - k) % stride == 0
            numerator = offs_n + padding - k  # [BLOCK_N]
            valid_div = (numerator % stride) == 0
            t_in = numerator // stride  # [BLOCK_N]
            valid_t = valid_div & (t_in >= 0) & (t_in < T_in) & (offs_n < T_out)

            # Load x[c_in, t_in]: [BLOCK_C, BLOCK_N]
            x_idx = c_offs[:, None] * T_in + t_in[None, :]  # [BLOCK_C, BLOCK_N]
            x_valid = c_mask[:, None] & valid_t[None, :]
            x_val = tl.load(x_ptr + x_idx, mask=x_valid, other=0.0)  # [BLOCK_C, BLOCK_N]

            # Load w[c_in, c_out, k]: [BLOCK_C, BLOCK_M]
            w_idx = c_offs[:, None] * C_out * K + offs_m[None, :] * K + k  # [BLOCK_C, BLOCK_M]
            w_valid = c_mask[:, None] & (offs_m[None, :] < C_out)
            w_val = tl.load(w_ptr + w_idx, mask=w_valid, other=0.0)  # [BLOCK_C, BLOCK_M]

            # Accumulate: [BLOCK_M, BLOCK_N] += [BLOCK_M, BLOCK_C] @ [BLOCK_C, BLOCK_N]
            # = w_val^T @ x_val
            acc += tl.dot(tl.trans(w_val), x_val)

    # Add bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < C_out, other=0.0)
    acc += bias[:, None]

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * T_out + offs_n[None, :]
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < T_out)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
