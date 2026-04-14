"""
Triton kernels for the Moonshine V2 encoder.

Moonshine encoder architecture (per layer):
  - UnitOffset LayerNorm: LN(x) * (gamma + 1.0)
  - Self-attention: Q/K/V projections (768→640), O projection (640→768)
  - FFN: fc1 (768→3072) + bias + GELU, fc2 (3072→768) + bias
  - Residual connections

Key dimensions:
  encoder_dim=768, kv_dim=640, intermediate=3072, head_dim=64, n_heads=10
  Sequence length T = 50-500 (1-10s audio after 4x frontend reduction)
"""
import triton
import triton.language as tl


# ─── Matmul (no bias) — for Q/K/V/O projections ──────────────────────────────

@triton.jit
def matmul_fp16(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[K,N], all fp16, fp32 accumulate."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ─── Mixed-precision Matmul (f16 activation × f32 weight) ────────────────────

@triton.jit
def matmul_f16a_f32w(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = A_f16[M,K] @ B_f32[K,N], f32 accumulate, f16 output.
    A (activations) is fp16, B (weights) is fp32 for Q8_0-exact precision.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


# ─── Activation functions (standard Triton pattern) ──────────────────────────

@triton.jit
def gelu(x):
    return x * (0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)))

@triton.jit
def silu(x):
    return x / (1.0 + tl.exp(-x))


# ─── Matmul + bias + optional activation ─────────────────────────────────────
# ACTIVATION: pass a @triton.jit function (gelu, silu, ...) or None.

@triton.jit
def matmul_bias_fp16(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """C[M,N] = act(A[M,K] @ B[K,N] + bias[N]), all fp16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc + bias[None, :].to(tl.float32)
    if ACTIVATION:
        c = ACTIVATION(c)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float16), mask=mask)


@triton.jit
def matmul_bias_f16a_f32w(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """C_f16[M,N] = act(A_f16[M,K] @ B_f32[K,N] + bias_f32[N])."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc + bias[None, :]
    if ACTIVATION:
        c = ACTIVATION(c)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float16), mask=mask)


# ─── Q8 packed matmul: A(fp16) @ B(int8) * scales(fp16) ─────────────────────

@triton.jit
def matmul_q8(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scales_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K](fp16) @ B[K,N](int8) * scales[N](fp16), fp32 accum."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0).to(tl.float16)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * scales[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


@triton.jit
def matmul_q8_bias(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scales_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K](fp16) @ B[K,N](int8) * scales[N](fp16) + bias[N](f32), fp32 accum."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0).to(tl.float16)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * scales[None, :]

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float16), mask=mask)


@triton.jit
def matmul_q8_bias_gelu(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scales_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[M,N] = GELU(A[M,K](fp16) @ B[K,N](int8) * scales[N](fp16) + bias[N](f32)), fp32 accum."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0).to(tl.float16)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * scales[None, :]

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc + bias[None, :]
    c = gelu(c)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float16), mask=mask)


# ─── LayerNorm ────────────────────────────────────────────────────────────────

@triton.jit
def layernorm(
    x_ptr, weight_ptr, out_ptr,
    n_rows, n_cols,
    stride_x_row, stride_out_row,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNIT_OFFSET: tl.constexpr,
):
    """LayerNorm: out = LN(x) * scale.

    UNIT_OFFSET=1: scale = (weight + 1.0)  (Moonshine encoder style)
    UNIT_OFFSET=0: scale = weight           (standard style)

    Input x can be fp16 or fp32 (the .to(tl.float32) is a no-op for f32).
    One program per row.
    """
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptrs = x_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    x_sq = tl.where(mask, x_centered * x_centered, 0.0)
    var = tl.sum(x_sq, axis=0) / n_cols

    rstd = 1.0 / tl.sqrt(var + eps)
    normed = x_centered * rstd

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    if UNIT_OFFSET:
        out = normed + normed * weight
    else:
        out = normed * weight

    out_ptrs = out_ptr + row_idx * stride_out_row + col_offsets
    tl.store(out_ptrs, out.to(tl.float16), mask=mask)


# ─── Bare LayerNorm (gamma baked into weights) ────────────────────────────────

@triton.jit
def layernorm_bare(
    x_ptr, out_ptr,
    n_rows, n_cols,
    stride_x_row, stride_out_row,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bare LayerNorm (no gamma): out = (x - mean) / sqrt(var + eps)

    For use when gamma is baked into the downstream weight matrix.
    x and out are [n_rows, n_cols] in fp16, compute in fp32.
    """
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptrs = x_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    x_sq = tl.where(mask, x_centered * x_centered, 0.0)
    var = tl.sum(x_sq, axis=0) / n_cols

    rstd = 1.0 / tl.sqrt(var + eps)
    normed = x_centered * rstd

    out_ptrs = out_ptr + row_idx * stride_out_row + col_offsets
    tl.store(out_ptrs, normed.to(tl.float16), mask=mask)


# ─── GELU activation ─────────────────────────────────────────────────────────

@triton.jit
def gelu_forward(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    out = x * (0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)))
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


# ─── Residual add ─────────────────────────────────────────────────────────────

@triton.jit
def residual_add(
    x_ptr, residual_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x + residual, element-wise fp16."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    r = tl.load(residual_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + r, mask=mask)


# ─── Broadcast bias add ──────────────────────────────────────────────────────

@triton.jit
def bias_add(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i] = x[i] + bias[i % n_cols], broadcast bias across rows."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias_offsets = offsets % n_cols
    b = tl.load(bias_ptr + bias_offsets, mask=mask)
    tl.store(out_ptr + offsets, x + b, mask=mask)


# ─── Mixed-precision kernels (f32 residual stream) ───────────────────────────

@triton.jit
def residual_add_f32(
    x_ptr, residual_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out_f32 = x_f16 + residual_f32. Mixed-precision residual add.
    x is f16 (layer output), residual and output are f32 (hidden stream).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    r = tl.load(residual_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + r, mask=mask)


@triton.jit
def convert_f16_to_f32(
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


# ─── Decoder-specific kernels ────────────────────────────────────────────────

@triton.jit
def silu_forward(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU(x) = x * sigmoid(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


@triton.jit
def silu_mul_forward(
    gate_ptr, x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GLU: out = SiLU(gate) * x. For decoder GLU MLP."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    out = (gate * tl.sigmoid(gate)) * x
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


@triton.jit
def scale_inplace(
    x_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """x *= scale, element-wise. For attention score scaling."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(x_ptr + offsets, (x * scale).to(tl.float16), mask=mask)


@triton.jit
def add_bias_f32(
    x_ptr, bias_ptr, out_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i,j] = x[i,j] + bias[j]. For fc1/fc2 bias add (f32 bias, f16 in/out)."""
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_idx * stride_row + col_offsets, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask)
    out = x + bias
    tl.store(out_ptr + row_idx * stride_row + col_offsets, out.to(tl.float16), mask=mask)


@triton.jit
def convert_f32_to_f16(
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


# ─── GEMV (M=1 specialized, 1D grid over N) ─────────────────────────────────
# v2: scalar K-loop, 1 thread per output element.  No 2D tiles, no shared
# memory, no barriers.  Matches hand-written kernel design.

@triton.jit
def gemv_f16w(
    x_ptr, w_ptr, out_ptr,
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
):
    """GEMV: out_f16[N] = x_f16[K] @ W_f16[N,K], f32 accumulate.
    W stored as [N, K] row-major: stride_wn=K, stride_wk=1 for sequential K access.
    Each thread handles one output column, looping over K.
    Grid: (cdiv(N, BLOCK_N), 1, 1).
    Requires N % BLOCK_N == 0 (all decoder dims are multiples of 128).
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = col * stride_wn

    for k in range(0, K):
        x_k = tl.load(x_ptr + k)
        w_k = tl.load(w_ptr + w_base + k * stride_wk)
        acc += (x_k * w_k).to(tl.float32)

    tl.store(out_ptr + col, acc.to(tl.float16))


@triton.jit
def gemv_bias_f16w(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
):
    """GEMV + bias: out_f16[N] = x_f16[K] @ W_f16[N,K] + bias_f32[N].
    Requires N % BLOCK_N == 0.
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = col * stride_wn

    for k in range(0, K):
        x_k = tl.load(x_ptr + k)
        w_k = tl.load(w_ptr + w_base + k * stride_wk)
        acc += (x_k * w_k).to(tl.float32)

    bias = tl.load(bias_ptr + col)
    acc += bias

    tl.store(out_ptr + col, acc.to(tl.float16))


# ─── Q8 GEMV (M=1, packed int8 weights + f16 per-block scales) ──────────────
# Scalar K-loop: 1 thread per output, zero barriers, zero shared memory.
# BLOCK_K=32 matches Q8_0 block size — inner loop unrolled 32× by Triton frontend.
# qs_ptr:     i32[K, N/4]   — 4 packed int8 weights per uint32 (consecutive cols)
# scales_ptr: f16[K/32, N]  — per-K-block scale

@triton.jit
def gemv_q8(
    x_ptr, qs_ptr, scales_ptr, out_ptr,
    N, K, N_div4,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Q8 GEMV: out[N] = x[K] @ dequant(W_q8[K,N]), scalar K-loop, 32× unrolled."""
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = col < N
    col_div4 = col // 4
    col_shift = (col % 4) * 8

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    qs_off = col_div4
    scale_off = col

    for k_block in range(0, K, BLOCK_K):
        scale = tl.load(scales_ptr + scale_off, mask=n_mask, other=0.0).to(tl.float32)
        scale_off += N
        for ki in range(BLOCK_K):  # constexpr → unrolled 32×
            x_k = tl.load(x_ptr + k_block + ki).to(tl.float32)
            packed = tl.load(qs_ptr + qs_off, mask=n_mask, other=0)
            byte_val = (packed >> col_shift) & 0xFF
            ival = tl.where(byte_val > 127, byte_val - 256, byte_val)
            acc += x_k * ival.to(tl.float32) * scale
            qs_off += N_div4

    tl.store(out_ptr + col, acc.to(tl.float16), mask=n_mask)


@triton.jit
def gemv_q8_bias(
    x_ptr, qs_ptr, scales_ptr, bias_ptr, out_ptr,
    N, K, N_div4,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Q8 GEMV + bias: out[N] = x[K] @ dequant(W_q8[K,N]) + bias[N]."""
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = col < N
    col_div4 = col // 4
    col_shift = (col % 4) * 8

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    qs_off = col_div4
    scale_off = col

    for k_block in range(0, K, BLOCK_K):
        scale = tl.load(scales_ptr + scale_off, mask=n_mask, other=0.0).to(tl.float32)
        scale_off += N
        for ki in range(BLOCK_K):
            x_k = tl.load(x_ptr + k_block + ki).to(tl.float32)
            packed = tl.load(qs_ptr + qs_off, mask=n_mask, other=0)
            byte_val = (packed >> col_shift) & 0xFF
            ival = tl.where(byte_val > 127, byte_val - 256, byte_val)
            acc += x_k * ival.to(tl.float32) * scale
            qs_off += N_div4

    bias = tl.load(bias_ptr + col, mask=n_mask, other=0.0)
    acc += bias
    tl.store(out_ptr + col, acc.to(tl.float16), mask=n_mask)


# ─── Softmax (row-wise, with optional additive mask) ─────────────────────────

@triton.jit
def softmax_rows(
    input_ptr, output_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-wise softmax. One program per row.
    Input/output are [n_rows, n_cols].
    """
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * stride_row
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

    # Numerically stable softmax
    row_max = tl.max(x, axis=0)
    x = x - row_max
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(output_ptr + row_start + col_offsets, result.to(tl.float16), mask=mask)


# ─── Fused softmax with additive mask ────────────────────────────────────────

@triton.jit
def masked_softmax_rows(
    input_ptr, mask_ptr, output_ptr,
    n_rows, n_cols,
    stride_in_row, stride_mask_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-wise softmax with additive mask (0 or -inf).
    Used for sliding window attention.
    """
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    x = tl.load(input_ptr + row_idx * stride_in_row + col_offsets,
                mask=col_mask, other=-float('inf')).to(tl.float32)
    m = tl.load(mask_ptr + row_idx * stride_mask_row + col_offsets,
                mask=col_mask, other=-float('inf')).to(tl.float32)
    x = x + m

    row_max = tl.max(x, axis=0)
    x = x - row_max
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(output_ptr + row_idx * stride_in_row + col_offsets,
             result.to(tl.float16), mask=col_mask)


# ─── Flash Attention 2 (multi-head, sliding window) ────────────────────────

@triton.jit
def flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len,
    stride_h,    # stride between heads (= head_dim for interleaved [T, H, D])
    stride_qkv,  # stride between rows for Q/K/V (= n_heads*D for contiguous)
    stride_o,    # stride between rows for O output (may differ from stride_qkv)
    sm_scale,    # 1/sqrt(head_dim)
    window_left, window_right,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
):
    """Flash Attention 2 forward pass with sliding window masking.

    Q, K, V: [n_heads, seq_len, D] fp16, contiguous.
    O:       [n_heads, seq_len, D] fp16, contiguous.
    Grid:    (cdiv(seq_len, BM), n_heads, 1)

    Uses online softmax (Dao et al.) to avoid materializing the full
    [seq_len, seq_len] attention matrix. fp32 accumulator, fp16 output.
    """
    pid_m = tl.program_id(0)   # which BM-block of query rows
    pid_h = tl.program_id(1)   # which head

    off_m = pid_m * BM
    head_off = pid_h * stride_h

    # Load Q tile [BM, D] — persists across all K/V iterations
    offs_m = off_m + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + head_off + offs_m[:, None] * stride_qkv + offs_d[None, :],
                mask=offs_m[:, None] < seq_len, other=0.0)

    # Initialize online softmax accumulators
    m_i = tl.full([BM], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    # Compute tight loop bounds from sliding window.
    # Only iterate over K/V blocks that overlap the attention window.
    # Lower bound: align down to BN boundary.
    # Note: integer division truncates toward zero, so small negative values
    # (when off_m < window_left) correctly round up to 0.
    kv_lo = (off_m - window_left) // BN * BN
    # Upper bound: align up to BN boundary.
    # Extra iterations past seq_len are safe (loads are masked).
    kv_hi = (off_m + BM + window_right + BN - 1) // BN * BN

    # Loop over K/V blocks (only those in window range)
    for kv_start in range(kv_lo, kv_hi, BN):
        offs_n = kv_start + tl.arange(0, BN)

        # Load K tile [BN, D]
        k = tl.load(K_ptr + head_off + offs_n[:, None] * stride_qkv + offs_d[None, :],
                    mask=offs_n[:, None] < seq_len, other=0.0)

        # QK = Q @ K^T : [BM, D] x [D, BN] -> [BM, BN]
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale

        # Sliding window + bounds mask
        # A query at position i attends to key at position j iff:
        #   i - window_left <= j <= i + window_right  AND  j < seq_len
        qk = tl.where(
            (offs_n[None, :] >= (offs_m[:, None] - window_left)) &
            (offs_n[None, :] <= (offs_m[:, None] + window_right)) &
            (offs_n[None, :] < seq_len),
            qk, -1e9)

        # Online softmax update
        m_ij = tl.max(qk, axis=1)            # [BM]
        m_new = tl.maximum(m_i, m_ij)        # [BM]
        alpha = tl.exp(m_i - m_new)          # [BM]
        p = tl.exp(qk - m_new[:, None])      # [BM, BN]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V tile [BN, D]
        v = tl.load(V_ptr + head_off + offs_n[:, None] * stride_qkv + offs_d[None, :],
                    mask=offs_n[:, None] < seq_len, other=0.0)

        # acc += P @ V : [BM, BN] x [BN, D] -> [BM, D]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Final normalization: O = acc / l
    acc = acc / l_i[:, None]

    # Store O [BM, D] as fp16
    tl.store(O_ptr + pid_h * stride_h + offs_m[:, None] * stride_o + offs_d[None, :],
             acc.to(tl.float16), mask=offs_m[:, None] < seq_len)


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder fusion kernels — Triton source for hand-written HLSL equivalents.
# These replace hand-coded HLSL with honest Triton→TTIR→HLSL compilation.
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def glu_silu_fused(
    fc1_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GLU-SiLU from single buffer: out[i] = SiLU(fc1[N+i]) * fc1[i].
    fc1_ptr: [2*n_elements], first half is x, second half is gate.
    Replaces hand-written glu_silu.hlsl.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(fc1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(fc1_ptr + n_elements + offsets, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate * tl.sigmoid(gate)
    tl.store(out_ptr + offsets, (silu_gate * x).to(tl.float16), mask=mask)


@triton.jit
def kv_cache_append(
    new_kv_ptr, cache_ptr,
    total_elems, max_kv_len, head_dim, pos,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy new K or V into KV cache at position pos.
    Cache: [n_kv_heads, max_kv_len, head_dim]. Input: [n_kv_heads * head_dim].
    total_elems = n_kv_heads * head_dim.
    Replaces hand-written kv_cache_append.hlsl.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems
    head = offsets // head_dim
    d = offsets % head_dim
    cache_idx = head * max_kv_len * head_dim + pos * head_dim + d
    val = tl.load(new_kv_ptr + offsets, mask=mask)
    tl.store(cache_ptr + cache_idx, val, mask=mask)


@triton.jit
def rope_interleaved(
    x_ptr, rope_table_ptr,
    total_pairs, head_dim, half_rot, pos,
    BLOCK_SIZE: tl.constexpr,
):
    """Interleaved RoPE: rotate pairs (x[2i], x[2i+1]) in-place.
    rope_table: packed [cos..., sin...] per position.
      cos = rope_table[pos * half_rot * 2 + pair_idx]
      sin = rope_table[pos * half_rot * 2 + half_rot + pair_idx]
    total_pairs = n_heads * half_rot.
    Replaces hand-written rope_interleaved.hlsl.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_pairs
    pair_idx = offsets % half_rot
    head_idx = offsets // half_rot

    table_base = pos * half_rot * 2
    cos_val = tl.load(rope_table_ptr + table_base + pair_idx, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(rope_table_ptr + table_base + half_rot + pair_idx, mask=mask, other=0.0).to(tl.float32)

    base = head_idx * head_dim + pair_idx * 2
    x0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)

    tl.store(x_ptr + base, (x0 * cos_val - x1 * sin_val).to(tl.float16), mask=mask)
    tl.store(x_ptr + base + 1, (x1 * cos_val + x0 * sin_val).to(tl.float16), mask=mask)


@triton.jit
def rope_cache_fused(
    x_ptr, cache_ptr, rope_table_ptr,
    total_pairs, head_dim, half_rot, pos, max_kv_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE + KV cache append: rotate pairs in-place AND copy to cache.
    Same thread does both stores, eliminating cross-thread dependency barrier.
    Only works when 2*half_rot == head_dim (all elements rotated).
    total_pairs = n_heads * half_rot.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_pairs
    pair_idx = offsets % half_rot
    head_idx = offsets // half_rot

    table_base = pos * half_rot * 2
    cos_val = tl.load(rope_table_ptr + table_base + pair_idx, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(rope_table_ptr + table_base + half_rot + pair_idx, mask=mask, other=0.0).to(tl.float32)

    base = head_idx * head_dim + pair_idx * 2
    x0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)

    y0 = (x0 * cos_val - x1 * sin_val).to(tl.float16)
    y1 = (x1 * cos_val + x0 * sin_val).to(tl.float16)

    # Write rotated values back to x (in-place RoPE)
    tl.store(x_ptr + base, y0, mask=mask)
    tl.store(x_ptr + base + 1, y1, mask=mask)

    # Also write to KV cache at [head, pos, d]
    cache_base = head_idx * max_kv_len * head_dim + pos * head_dim + pair_idx * 2
    tl.store(cache_ptr + cache_base, y0, mask=mask)
    tl.store(cache_ptr + cache_base + 1, y1, mask=mask)


@triton.jit
def residual_add_layernorm_fused(
    proj_ptr, residual_ptr, out_f32_ptr,
    weight_ptr, norm_ptr,
    n_rows, dim, stride_in, stride_out,
    eps: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Fused residual-add + layernorm.
    out_f32 = f16_proj + f32_residual (new residual stream).
    norm_f16 = LN(out_f32) * weight.
    One program per row. Replaces hand-written residual_add_layernorm.hlsl.
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    in_idx = row * stride_in + offsets
    out_idx = row * stride_out + offsets

    # Residual add
    proj = tl.load(proj_ptr + in_idx, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + in_idx, mask=mask, other=0.0)
    val = proj + res
    tl.store(out_f32_ptr + in_idx, val, mask=mask)

    # Layernorm
    mean = tl.sum(val, axis=0) / dim
    centered = val - mean
    sq = tl.where(mask, centered * centered, 0.0)
    var = tl.sum(sq, axis=0) / dim
    inv_std = 1.0 / tl.sqrt(var + eps)
    normed = centered * inv_std

    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(norm_ptr + out_idx, (normed * w).to(tl.float16), mask=mask)


@triton.jit
def gemv_bias_glu_fused(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, K, stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
):
    """Fused GEMV + bias + GLU-SiLU for decoder MLP fc1.
    w: [2*N, K] row-major. x-half in rows [0,N), gate-half in rows [N,2N).
    bias: [2*N] (f32). stride_wn=K, stride_wk=1 for sequential access.
    out[n] = SiLU(gate_acc + bias[N+n]) * (x_acc + bias[n]).
    Grid: (cdiv(N, BLOCK_N), 1, 1).
    v2: scalar K-loop, 1 thread per output.
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_gate = col + N

    acc_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base_x = col * stride_wn
    w_base_g = col_gate * stride_wn

    for k in range(0, K):
        x_k = tl.load(x_ptr + k)
        wx_k = tl.load(w_ptr + w_base_x + k * stride_wk)
        wg_k = tl.load(w_ptr + w_base_g + k * stride_wk)
        acc_x += (x_k * wx_k).to(tl.float32)
        acc_gate += (x_k * wg_k).to(tl.float32)

    # Add bias
    bias_x = tl.load(bias_ptr + col)
    bias_gate = tl.load(bias_ptr + col + N)
    acc_x += bias_x
    acc_gate += bias_gate

    # GLU: SiLU(gate) * x
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    tl.store(out_ptr + col, (silu_gate * acc_x).to(tl.float16))


# ─── K-split GEMV (parallel reduction over K dimension) ─────────────────────
# Split the K-loop across multiple threadgroups for higher GPU occupancy.
# Phase 1: partial GEMV accumulates f32 partials in scratch buffer.
# Phase 2: reduce across K-splits + bias + optional GLU → f16 output.

@triton.jit
def gemv_splitk_partial(
    x_ptr, w_ptr, partial_ptr,
    N, K, stride_wk,
    k_per_split, stride_partial,
    BLOCK_N: tl.constexpr, N_N_BLOCKS: tl.constexpr,
):
    """K-split partial GEMV with f16 partials.
    Grid: (N_N_BLOCKS * n_splits, 1, 1).
    partial_ptr: f16[n_splits, N], stride_partial = N.
    Accumulates in f32, stores as f16.
    """
    pid = tl.program_id(0)
    pid_n = pid % N_N_BLOCKS
    pid_k = pid // N_N_BLOCKS
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * k_per_split

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for ki in range(0, k_per_split):
        k = k_start + ki
        x_k = tl.load(x_ptr + k)
        w_k = tl.load(w_ptr + col + k * stride_wk)
        acc += (x_k * w_k).to(tl.float32)

    tl.store(partial_ptr + pid_k * stride_partial + col, acc.to(tl.float16))


@triton.jit
def gemv_splitk_bias_reduce(
    partial_ptr, bias_ptr, out_ptr,
    N, n_splits, stride_partial,
    BLOCK_N: tl.constexpr,
):
    """Reduce f16 K-split partial sums + add bias -> f16 output.
    Grid: (cdiv(N, BLOCK_N), 1, 1).
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, n_splits):
        partial = tl.load(partial_ptr + s * stride_partial + col).to(tl.float32)
        acc += partial

    bias = tl.load(bias_ptr + col).to(tl.float32)
    acc += bias
    tl.store(out_ptr + col, acc.to(tl.float16))


@triton.jit
def gemv_splitk_reduce(
    partial_ptr, out_ptr,
    N, n_splits, stride_partial,
    BLOCK_N: tl.constexpr,
):
    """Reduce f16 K-split partial sums (no bias) -> f16 output.
    Grid: (cdiv(N, BLOCK_N), 1, 1).
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, n_splits):
        partial = tl.load(partial_ptr + s * stride_partial + col).to(tl.float32)
        acc += partial

    tl.store(out_ptr + col, acc.to(tl.float16))


@triton.jit
def gemv_splitk_reduce_resadd_ln(
    partial_ptr, residual_ptr, out_f32_ptr,
    ln_weight_ptr, norm_ptr,
    dim, n_splits, stride_partial,
    eps: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    """Fused split-K reduce + residual-add + layernorm. Replaces 2 dispatches with 1.
    Grid: (1, 1, 1). One threadgroup handles all dim outputs.
    Phase 1: Reduce f16 partial sums across splits -> f32 GEMV result
    Phase 2: Add f32 residual -> new f32 residual stream
    Phase 3: Layernorm -> f16 normalized output
    """
    offs = tl.arange(0, BLOCK_DIM)
    mask = offs < dim

    # Phase 1: Reduce partials
    acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    for s in range(0, n_splits):
        partial = tl.load(partial_ptr + s * stride_partial + offs, mask=mask, other=0.0).to(tl.float32)
        acc += partial

    # Phase 2: Residual add (cast GEMV result to f16 then back to f32 for precision matching)
    proj = acc.to(tl.float16).to(tl.float32)
    res = tl.load(residual_ptr + offs, mask=mask, other=0.0)
    val = proj + res
    tl.store(out_f32_ptr + offs, val, mask=mask)

    # Phase 3: Layernorm
    mean = tl.sum(val, axis=0) / dim
    centered = val - mean
    sq = tl.where(mask, centered * centered, 0.0)
    var = tl.sum(sq, axis=0) / dim
    inv_std = 1.0 / tl.sqrt(var + eps)
    normed = centered * inv_std

    w = tl.load(ln_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(norm_ptr + offs, (normed * w).to(tl.float16), mask=mask)


@triton.jit
def gemv_qkv_splitk_partial(
    x_ptr, wq_ptr, wk_ptr, wv_ptr,
    partial_ptr,
    N, K, stride_wk,
    k_per_split, stride_partial,
    BLOCK_N: tl.constexpr, N_N_BLOCKS: tl.constexpr,
):
    """Fused Q/K/V split-K partial GEMV. Single dispatch for all 3 projections.
    Grid: (N_N_BLOCKS * n_splits, 3, 1). program_id(1) selects Q/K/V.
    partial_ptr: f16[3, n_splits, N]. stride_partial = N.
    """
    pid = tl.program_id(0)
    proj = tl.program_id(1)   # 0=Q, 1=K, 2=V
    pid_n = pid % N_N_BLOCKS
    pid_k = pid // N_N_BLOCKS
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * k_per_split

    # Masks: only the selected projection loads from memory
    is_q = (proj == 0)
    is_k = (proj == 1)
    is_v = (proj == 2)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for ki in range(0, k_per_split):
        k = k_start + ki
        x_k = tl.load(x_ptr + k)
        addr = col + k * stride_wk
        # Masked loads: only 1 of 3 actually reads memory per threadgroup
        wq_val = tl.load(wq_ptr + addr, mask=is_q, other=0.0)
        wk_val = tl.load(wk_ptr + addr, mask=is_k, other=0.0)
        wv_val = tl.load(wv_ptr + addr, mask=is_v, other=0.0)
        w_k = wq_val + wk_val + wv_val
        acc += (x_k * w_k).to(tl.float32)

    # Store to proj-specific slice: partial[proj, split, col]
    n_splits = K // k_per_split
    proj_off = proj * n_splits * stride_partial
    tl.store(partial_ptr + proj_off + pid_k * stride_partial + col, acc.to(tl.float16))


@triton.jit
def gemv_qkv_splitk_reduce(
    partial_ptr, oq_ptr, ok_ptr, ov_ptr,
    N, n_splits, stride_partial,
    BLOCK_N: tl.constexpr,
):
    """Fused Q/K/V reduce. Grid: (cdiv(N, BLOCK_N), 3, 1).
    program_id(1) selects Q/K/V.
    """
    pid = tl.program_id(0)
    proj = tl.program_id(1)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    proj_off = proj * stride_partial * n_splits
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, n_splits):
        partial = tl.load(partial_ptr + proj_off + s * stride_partial + col).to(tl.float32)
        acc += partial

    result = acc.to(tl.float16)
    # Store to Q, K, or V output based on proj
    tl.store(oq_ptr + col, result, mask=proj == 0)
    tl.store(ok_ptr + col, result, mask=proj == 1)
    tl.store(ov_ptr + col, result, mask=proj == 2)


@triton.jit
def gemv_glu_splitk_partial(
    x_ptr, w_ptr, partial_ptr,
    N, K, stride_wk,
    k_per_split, stride_partial,
    BLOCK_N: tl.constexpr, N_N_BLOCKS: tl.constexpr,
):
    """K-split partial GEMV for GLU with f16 partials.
    partial_ptr: f16[n_splits, 2*N]. stride_partial = 2*N.
    """
    pid = tl.program_id(0)
    pid_n = pid % N_N_BLOCKS
    pid_k = pid // N_N_BLOCKS
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_gate = col + N
    k_start = pid_k * k_per_split

    acc_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for ki in range(0, k_per_split):
        k = k_start + ki
        x_k = tl.load(x_ptr + k)
        wx_k = tl.load(w_ptr + col + k * stride_wk)
        wg_k = tl.load(w_ptr + col_gate + k * stride_wk)
        acc_x += (x_k * wx_k).to(tl.float32)
        acc_gate += (x_k * wg_k).to(tl.float32)

    partial_off = pid_k * stride_partial
    tl.store(partial_ptr + partial_off + col, acc_x.to(tl.float16))
    tl.store(partial_ptr + partial_off + N + col, acc_gate.to(tl.float16))


@triton.jit
def gemv_glu_splitk_reduce(
    partial_ptr, bias_ptr, out_ptr,
    N, n_splits, stride_partial,
    BLOCK_N: tl.constexpr,
):
    """Reduce GLU f16 K-split partials + bias + SiLU gating -> f16.
    Grid: (cdiv(N, BLOCK_N), 1, 1).
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, n_splits):
        partial_off = s * stride_partial
        acc_x += tl.load(partial_ptr + partial_off + col).to(tl.float32)
        acc_gate += tl.load(partial_ptr + partial_off + N + col).to(tl.float32)

    # Add bias
    bias_x = tl.load(bias_ptr + col).to(tl.float32)
    bias_gate = tl.load(bias_ptr + col + N).to(tl.float32)
    acc_x += bias_x
    acc_gate += bias_gate

    # GLU: SiLU(gate) * x
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    tl.store(out_ptr + col, (silu_gate * acc_x).to(tl.float16))


@triton.jit
def gemv_resadd_ln_fused(
    x_ptr, w_ptr, residual_ptr, out_f32_ptr,
    ln_weight_ptr, norm_ptr,
    dim, K, stride_wn, stride_wk,
    eps: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    """Fused GEMV + residual-add + layernorm for decoder (M=1).
    One program (one threadgroup) handles all dim outputs.
    Grid: (1, 1, 1).
    v2: scalar K-loop, 1 thread per output element.

    Phase 1: GEMV — proj[i] = sum_k(x[k] * W[i, k]) with W in [N,K] layout
    Phase 2: val[i] = (half)proj[i] + f32_residual[i]
    Phase 3: norm[i] = LN(val) * ln_weight[i]
    """
    offs = tl.arange(0, BLOCK_DIM)
    d_mask = offs < dim

    # Phase 1: GEMV (scalar K-loop, 1 thread per output)
    acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    w_off = offs * stride_wn
    for k in range(0, K):
        x_k = tl.load(x_ptr + k).to(tl.float32)
        w_k = tl.load(w_ptr + w_off, mask=d_mask, other=0.0).to(tl.float32)
        acc += x_k * w_k
        w_off += stride_wk

    # Phase 2: Residual add (match HLSL: cast to f16 then back to f32)
    proj = acc.to(tl.float16).to(tl.float32)
    res = tl.load(residual_ptr + offs, mask=d_mask, other=0.0)
    val = proj + res
    tl.store(out_f32_ptr + offs, val, mask=d_mask)

    # Phase 3: Layernorm
    mean = tl.sum(val, axis=0) / dim
    centered = val - mean
    sq = tl.where(d_mask, centered * centered, 0.0)
    var = tl.sum(sq, axis=0) / dim
    inv_std = 1.0 / tl.sqrt(var + eps)
    normed = centered * inv_std

    w = tl.load(ln_weight_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(norm_ptr + offs, (normed * w).to(tl.float16), mask=d_mask)


@triton.jit
def gemv_qkv_fused(
    x_ptr, wq_ptr, wk_ptr, wv_ptr,
    oq_ptr, ok_ptr, ov_ptr,
    N, K, stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
):
    """Fused Q/K/V GEMV: 3 matrix-vector products in one dispatch.
    Grid: (cdiv(3*N, BLOCK_N), 1, 1). Each thread computes one output
    element from one of Q/K/V, selected by dividing global index by N.
    All projections have the same output dim N and input dim K.
    W stored as [N, K] row-major: stride_wn=K, stride_wk=1.
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    total_n = N * 3
    mask = col < total_n
    proj = col // N      # 0=Q, 1=K, 2=V
    n_idx = col % N      # output index within projection
    n_mask = n_idx < N   # always true when mask is true, but needed for loads

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # We compute all 3 projections' weight loads and select via proj index.
    # This avoids branching: load from all 3, mask by proj.
    wq_off = n_idx * stride_wn
    wk_off = n_idx * stride_wn
    wv_off = n_idx * stride_wn

    for k in range(0, K):
        x_k = tl.load(x_ptr + k).to(tl.float32)
        wq_k = tl.load(wq_ptr + wq_off, mask=mask, other=0.0).to(tl.float32)
        wk_k = tl.load(wk_ptr + wk_off, mask=mask, other=0.0).to(tl.float32)
        wv_k = tl.load(wv_ptr + wv_off, mask=mask, other=0.0).to(tl.float32)
        # Select the correct weight based on projection index
        w_k = tl.where(proj == 0, wq_k, tl.where(proj == 1, wk_k, wv_k))
        acc += x_k * w_k
        wq_off += stride_wk
        wk_off += stride_wk
        wv_off += stride_wk

    out_val = acc.to(tl.float16)
    # Store to appropriate output buffer
    tl.store(oq_ptr + n_idx, out_val, mask=(mask & (proj == 0)))
    tl.store(ok_ptr + n_idx, out_val, mask=(mask & (proj == 1)))
    tl.store(ov_ptr + n_idx, out_val, mask=(mask & (proj == 2)))


@triton.jit
def rope_qk_cache_fused(
    q_ptr, k_ptr, rope_table_ptr, cache_k_ptr,
    N_Q_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, HALF_ROT: tl.constexpr,
    pos, max_kv_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE(Q) + RoPE(K) + cache(K) in one dispatch (4 UAVs).
    V cache handled by separate kv_cache_append dispatch.
    Single threadgroup (grid 1,1,1). Each thread handles its own data:
      Phase 1: RoPE on Q (n_q_heads * half_rot pairs)
      Phase 2: RoPE on K + copy rotated K to cache (per-thread, no barrier)
      Phase 3: Copy K pass-through elements to cache (elements beyond rotary_dim)
    """
    tid = tl.arange(0, BLOCK_SIZE)

    # Table base for this position
    table_base = pos * HALF_ROT * 2
    rotary_dim = HALF_ROT * 2

    # Phase 1: RoPE on Q
    q_pairs = N_Q_HEADS * HALF_ROT
    q_mask = tid < q_pairs
    q_pair_idx = tid % HALF_ROT
    q_head_idx = tid // HALF_ROT

    cos_val = tl.load(rope_table_ptr + table_base + q_pair_idx, mask=q_mask, other=1.0).to(tl.float32)
    sin_val = tl.load(rope_table_ptr + table_base + HALF_ROT + q_pair_idx, mask=q_mask, other=0.0).to(tl.float32)

    q_base = q_head_idx * HEAD_DIM + q_pair_idx * 2
    q0 = tl.load(q_ptr + q_base, mask=q_mask, other=0.0).to(tl.float32)
    q1 = tl.load(q_ptr + q_base + 1, mask=q_mask, other=0.0).to(tl.float32)

    tl.store(q_ptr + q_base, (q0 * cos_val - q1 * sin_val).to(tl.float16), mask=q_mask)
    tl.store(q_ptr + q_base + 1, (q1 * cos_val + q0 * sin_val).to(tl.float16), mask=q_mask)

    # Phase 2: RoPE on K + direct cache copy (fused per-thread, no barrier needed)
    k_pairs = N_KV_HEADS * HALF_ROT
    k_mask = tid < k_pairs
    k_pair_idx = tid % HALF_ROT
    k_head_idx = tid // HALF_ROT

    cos_k = tl.load(rope_table_ptr + table_base + k_pair_idx, mask=k_mask, other=1.0).to(tl.float32)
    sin_k = tl.load(rope_table_ptr + table_base + HALF_ROT + k_pair_idx, mask=k_mask, other=0.0).to(tl.float32)

    k_base = k_head_idx * HEAD_DIM + k_pair_idx * 2
    k0 = tl.load(k_ptr + k_base, mask=k_mask, other=0.0).to(tl.float32)
    k1 = tl.load(k_ptr + k_base + 1, mask=k_mask, other=0.0).to(tl.float32)

    k0_rot = (k0 * cos_k - k1 * sin_k).to(tl.float16)
    k1_rot = (k1 * cos_k + k0 * sin_k).to(tl.float16)

    tl.store(k_ptr + k_base, k0_rot, mask=k_mask)
    tl.store(k_ptr + k_base + 1, k1_rot, mask=k_mask)

    # Copy rotated K to cache (same thread wrote it, no cross-thread dependency)
    cache_k_base = k_head_idx * max_kv_len * HEAD_DIM + pos * HEAD_DIM + k_pair_idx * 2
    tl.store(cache_k_ptr + cache_k_base, k0_rot, mask=k_mask)
    tl.store(cache_k_ptr + cache_k_base + 1, k1_rot, mask=k_mask)

    # Phase 3: Copy pass-through K elements to cache (elements rotary_dim..head_dim)
    pass_per_head = HEAD_DIM - rotary_dim
    k_pass_total = N_KV_HEADS * pass_per_head
    k_pass_mask = tid < k_pass_total
    k_pass_head = tid // pass_per_head
    k_pass_offset = tid % pass_per_head

    k_src = k_pass_head * HEAD_DIM + rotary_dim + k_pass_offset
    k_val = tl.load(k_ptr + k_src, mask=k_pass_mask, other=0.0)
    cache_dst = k_pass_head * max_kv_len * HEAD_DIM + pos * HEAD_DIM + rotary_dim + k_pass_offset
    tl.store(cache_k_ptr + cache_dst, k_val, mask=k_pass_mask)


@triton.jit
def attention_decode_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    kv_len,
    n_q_heads, n_kv_heads,
    sm_scale,
    is_causal, q_pos,
    stride_kv_head, stride_kv_seq,
    BLOCK_KV: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Single-query attention decode with online softmax.
    One program per query head. Supports GQA and strided KV layouts.

    Q: [n_q_heads * head_dim] contiguous.
    K, V: strided via stride_kv_head (between heads) and stride_kv_seq (between positions).
    O: [n_q_heads * head_dim] contiguous.
    Grid: (n_q_heads, 1, 1).
    Replaces hand-written attention_decode.hlsl.
    """
    q_head = tl.program_id(0)
    kv_head = q_head * n_kv_heads // n_q_heads

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + q_head * HEAD_DIM + offs_d).to(tl.float32)

    # Online softmax accumulators
    m_i = tl.zeros([1], dtype=tl.float32) - 1e30
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    kv_base = kv_head * stride_kv_head

    for kv_start in range(0, kv_len, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < kv_len

        # Load K [BLOCK_KV, HEAD_DIM]
        k = tl.load(K_ptr + kv_base + offs_kv[:, None] * stride_kv_seq + offs_d[None, :],
                    mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # scores[i] = dot(Q, K[i]) * sm_scale
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale  # [BLOCK_KV]

        # Causal + validity mask
        causal_ok = (is_causal == 0) | (offs_kv <= q_pos)
        scores = tl.where(kv_mask & causal_ok, scores, -1e30)

        # Online softmax
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)  # [BLOCK_KV]

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        # Load V and accumulate: acc += p @ V
        v = tl.load(V_ptr + kv_base + offs_kv[:, None] * stride_kv_seq + offs_d[None, :],
                    mask=kv_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)  # [HEAD_DIM]

        m_i = m_new

    # Normalize
    acc = acc / l_i
    tl.store(O_ptr + q_head * HEAD_DIM + offs_d, acc.to(tl.float16))


@triton.jit
def attention_decode_1d(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    kv_len,
    n_q_heads, n_kv_heads,
    sm_scale,
    stride_kv_head, stride_kv_seq,
    HEAD_DIM: tl.constexpr,
):
    """1D attention decode — one thread per head dimension.
    Each program handles one query head. Thread count = HEAD_DIM.
    Uses 1D operations + tl.sum for dot product → WaveActiveSum.
    No 2D tile operations, no causal mask (not needed for decode).
    Grid: (n_q_heads, 1, 1).
    """
    q_head = tl.program_id(0)
    kv_head = q_head * n_kv_heads // n_q_heads

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + q_head * HEAD_DIM + offs_d).to(tl.float32)

    # Online softmax accumulators
    m_i = tl.zeros([1], dtype=tl.float32) - 1e30
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    kv_base = kv_head * stride_kv_head

    for kv in range(0, kv_len):
        # 1D K load: one element per thread
        k = tl.load(K_ptr + kv_base + kv * stride_kv_seq + offs_d).to(tl.float32)

        # Dot product via 1D reduction → WaveActiveSum
        score = tl.sum(q * k, axis=0) * sm_scale

        # Online softmax update
        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)
        l_i = l_i * alpha + p
        acc = acc * alpha

        # 1D V load and accumulate
        v = tl.load(V_ptr + kv_base + kv * stride_kv_seq + offs_d).to(tl.float32)
        acc = acc + p * v

        m_i = m_new

    # Normalize
    acc = acc / l_i
    tl.store(O_ptr + q_head * HEAD_DIM + offs_d, acc.to(tl.float16))


@triton.jit
def attention_decode_1d_masked(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    kv_len,
    n_q_heads, n_kv_heads,
    sm_scale,
    stride_kv_head, stride_kv_seq,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    """1D attention decode with masked head_dim (for non-power-of-2 dims).
    BLOCK_D is the next power of 2 >= head_dim (compile-time).
    head_dim is the actual dimension (runtime).
    Grid: (n_q_heads, 1, 1). Thread count = BLOCK_D.
    """
    q_head = tl.program_id(0)
    kv_head = q_head * n_kv_heads // n_q_heads

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    q = tl.load(Q_ptr + q_head * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    # Use BLOCK_D-sized tiles for softmax state so they map to per-thread registers.
    # After tl.sum reduction, all threads have the same score, so each thread
    # independently computes the same m_i/l_i — no shared memory or barriers needed.
    m_i = tl.full([BLOCK_D], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    kv_base = kv_head * stride_kv_head

    for kv in range(0, kv_len):
        k = tl.load(K_ptr + kv_base + kv * stride_kv_seq + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * k, axis=0) * sm_scale

        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)
        l_i = l_i * alpha + p
        acc = acc * alpha

        v = tl.load(V_ptr + kv_base + kv * stride_kv_seq + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        acc = acc + p * v

        m_i = m_new

    acc = acc / l_i
    tl.store(O_ptr + q_head * head_dim + offs_d, acc.to(tl.float16), mask=d_mask)



# ─── Split-KV attention decode (flash decoding) ────────────────────────────

@triton.jit
def attention_decode_splitkv_partial(
    Q_ptr, K_ptr, V_ptr, partial_ptr,
    kv_len, n_q_heads, n_kv_heads, sm_scale,
    stride_kv_head, stride_kv_seq, head_dim,
    n_splits, kv_per_split, stride_partial,
    BLOCK_D: tl.constexpr,
):
    """Split-KV attention partial kernel.
    Grid: (n_q_heads * n_splits, 1, 1). Thread count = BLOCK_D.
    Each program computes partial attention for kv_per_split positions.
    Writes [m(BLOCK_D), l(BLOCK_D), acc(BLOCK_D)] f32 to partial_ptr.
    stride_partial = 3 * BLOCK_D.
    """
    pid = tl.program_id(0)
    q_head = pid // n_splits
    split_idx = pid % n_splits
    kv_head = q_head * n_kv_heads // n_q_heads

    kv_start = split_idx * kv_per_split

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    q = tl.load(Q_ptr + q_head * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_D], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    kv_base = kv_head * stride_kv_head

    for kv_offset in range(0, kv_per_split):
        kv_pos = kv_start + kv_offset
        if kv_pos < kv_len:
            k = tl.load(K_ptr + kv_base + kv_pos * stride_kv_seq + offs_d, mask=d_mask, other=0.0).to(tl.float32)
            score = tl.sum(q * k, axis=0) * sm_scale

            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(score - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha

            v = tl.load(V_ptr + kv_base + kv_pos * stride_kv_seq + offs_d, mask=d_mask, other=0.0).to(tl.float32)
            acc = acc + p * v
            m_i = m_new

    # Store partial: [m, l, acc] each BLOCK_D f32
    partial_off = pid * stride_partial
    tl.store(partial_ptr + partial_off + offs_d, m_i)
    tl.store(partial_ptr + partial_off + BLOCK_D + offs_d, l_i)
    tl.store(partial_ptr + partial_off + 2 * BLOCK_D + offs_d, acc, mask=d_mask)


@triton.jit
def attention_decode_splitkv_reduce(
    partial_ptr, O_ptr,
    n_q_heads, n_splits, head_dim, stride_partial,
    BLOCK_D: tl.constexpr,
):
    """Reduce split-KV partials to final output.
    Grid: (n_q_heads, 1, 1). Thread count = BLOCK_D.
    Reads [m, l, acc] per split from partial_ptr and combines online softmax.
    """
    q_head = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    m_i = tl.full([BLOCK_D], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(0, n_splits):
        partial_off = (q_head * n_splits + s) * stride_partial
        m_j = tl.load(partial_ptr + partial_off + offs_d)
        l_j = tl.load(partial_ptr + partial_off + BLOCK_D + offs_d)
        v_j = tl.load(partial_ptr + partial_off + 2 * BLOCK_D + offs_d, mask=d_mask, other=0.0)

        m_new = tl.maximum(m_i, m_j)
        alpha_i = tl.exp(m_i - m_new)
        alpha_j = tl.exp(m_j - m_new)
        l_i = l_i * alpha_i + l_j * alpha_j
        acc = acc * alpha_i + v_j * alpha_j
        m_i = m_new

    acc = acc / l_i
    tl.store(O_ptr + q_head * head_dim + offs_d, acc.to(tl.float16), mask=d_mask)


# ─── Q8 fused GEMV kernels ──────────────────────────────────────────────────

@triton.jit
def gemv_q8_bias_glu(
    x_ptr, qs_ptr, scales_ptr, bias_ptr, out_ptr,
    N, K, N_div4,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Q8 fused GEMV + bias + GLU-SiLU for decoder MLP fc1.
    W: [K, 2*N] packed Q8 — x-half in cols [0,N), gate-half in cols [N,2*N).
    scales: [K/32, 2*N]. bias: [2*N] f32.
    Grid: (cdiv(N, BLOCK_N), 1, 1).
    """
    pid = tl.program_id(0)
    col = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = col < N
    col_div4 = col // 4
    col_shift = (col % 4) * 8
    col_gate_div4 = (col + N) // 4  # N divisible by 4

    N2 = N * 2
    N2_div4 = N2 // 4

    acc_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    qs_off_x = col_div4
    qs_off_g = col_gate_div4
    scale_off_x = col
    scale_off_g = col + N

    for k_block in range(0, K, BLOCK_K):
        sx = tl.load(scales_ptr + scale_off_x, mask=n_mask, other=0.0).to(tl.float32)
        sg = tl.load(scales_ptr + scale_off_g, mask=n_mask, other=0.0).to(tl.float32)
        scale_off_x += N2
        scale_off_g += N2
        for ki in range(BLOCK_K):
            x_k = tl.load(x_ptr + k_block + ki).to(tl.float32)
            px = tl.load(qs_ptr + qs_off_x, mask=n_mask, other=0)
            bx = (px >> col_shift) & 0xFF
            ix = tl.where(bx > 127, bx - 256, bx)
            acc_x += x_k * ix.to(tl.float32) * sx
            pg = tl.load(qs_ptr + qs_off_g, mask=n_mask, other=0)
            bg = (pg >> col_shift) & 0xFF
            ig = tl.where(bg > 127, bg - 256, bg)
            acc_gate += x_k * ig.to(tl.float32) * sg
            qs_off_x += N2_div4
            qs_off_g += N2_div4

    bias_x = tl.load(bias_ptr + col, mask=n_mask, other=0.0)
    bias_gate = tl.load(bias_ptr + col + N, mask=n_mask, other=0.0)
    acc_x += bias_x
    acc_gate += bias_gate
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    tl.store(out_ptr + col, (silu_gate * acc_x).to(tl.float16), mask=n_mask)


@triton.jit
def gemv_q8_resadd_ln(
    x_ptr, qs_ptr, scales_ptr,
    residual_ptr, out_f32_ptr,
    ln_weight_ptr, norm_ptr,
    dim, K, N_div4,
    eps: tl.constexpr, BLOCK_DIM: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Q8 fused GEMV + residual-add + layernorm for decoder (M=1).
    Grid: (1, 1, 1). One threadgroup handles all dim outputs.
    """
    offs = tl.arange(0, BLOCK_DIM)
    d_mask = offs < dim
    col_div4 = offs // 4
    col_shift = (offs % 4) * 8

    acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    qs_off = col_div4
    scale_off = offs

    for k_block in range(0, K, BLOCK_K):
        scale = tl.load(scales_ptr + scale_off, mask=d_mask, other=0.0).to(tl.float32)
        scale_off += dim
        for ki in range(BLOCK_K):
            x_k = tl.load(x_ptr + k_block + ki).to(tl.float32)
            packed = tl.load(qs_ptr + qs_off, mask=d_mask, other=0)
            byte_val = (packed >> col_shift) & 0xFF
            ival = tl.where(byte_val > 127, byte_val - 256, byte_val)
            acc += x_k * ival.to(tl.float32) * scale
            qs_off += N_div4

    proj = acc.to(tl.float16).to(tl.float32)
    res = tl.load(residual_ptr + offs, mask=d_mask, other=0.0)
    val = proj + res
    tl.store(out_f32_ptr + offs, val, mask=d_mask)

    mean = tl.sum(val, axis=0) / dim
    centered = val - mean
    sq = tl.where(d_mask, centered * centered, 0.0)
    var = tl.sum(sq, axis=0) / dim
    inv_std = 1.0 / tl.sqrt(var + eps)
    normed = centered * inv_std

    w = tl.load(ln_weight_ptr + offs, mask=d_mask, other=0.0).to(tl.float32)
    tl.store(norm_ptr + offs, (normed * w).to(tl.float16), mask=d_mask)
