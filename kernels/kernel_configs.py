"""Consolidated kernel configurations for all platforms.

Each config: (name, function_name, signature, num_warps, grid, options)
  options dict keys:
    targets: list of "apple", "intel", "hlsl" (default: all three)
    force_acc_fp16: bool (HLSL only, default False)

The function_name refers to an @triton.jit function in moonshine_kernels.py.
"""

# ── Metal configs (Apple Silicon + Intel share same configs) ─────────────────

METAL_KERNELS = [
    # ── Encoder matmul (no bias) ──
    ("matmul_fp16_64x64x32", "matmul_fp16",
     "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_fp16_128x128x32", "matmul_fp16",
     "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32",
     8, ["cdiv(M, 128)", "cdiv(N, 128)", "1"]),

    # ── Matmul + bias ──
    ("matmul_bias_fp16_32x32x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),
    ("matmul_bias_fp16_64x64x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_bias_fp16_128x128x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32",
     8, ["cdiv(M, 128)", "cdiv(N, 128)", "1"]),

    # ── Fused matmul + bias + GELU ──
    ("matmul_bias_gelu_fp16_32x32x32", "matmul_bias_gelu_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),
    ("matmul_bias_gelu_fp16_64x64x32", "matmul_bias_gelu_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_bias_gelu_fp16_128x128x32", "matmul_bias_gelu_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32",
     8, ["cdiv(M, 128)", "cdiv(N, 128)", "1"]),

    # ── Flash Attention 2 ──
    ("flash_attention_fwd_32x32x64", "flash_attention_fwd",
     "*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, fp32, i32, i32, 32, 32, 64",
     4, ["cdiv(seq_len, 32)", "n_heads", "1"]),

    # ── LayerNorm ──
    ("layernorm_unit_offset_768", "layernorm",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024, 1",
     4, ["n_rows", "1", "1"]),
    ("layernorm_bare_768", "layernorm_bare",
     "*fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024",
     4, ["n_rows", "1", "1"]),
    ("layernorm_standard_768", "layernorm",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024, 0",
     4, ["n_rows", "1", "1"]),
    ("layernorm_standard_640", "layernorm",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024, 0",
     4, ["n_rows", "1", "1"]),
    ("layernorm_standard_f32in_640", "layernorm",
     "*fp32, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024, 0",
     4, ["n_rows", "1", "1"]),

    # ── Element-wise ──
    ("gelu_fp16", "gelu_forward",
     "*fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("residual_add_fp16", "residual_add",
     "*fp16, *fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("softmax_fp16", "softmax_rows",
     "*fp16, *fp16, i32, i32, i32, 1024",
     4, ["n_rows", "1", "1"]),
    ("masked_softmax_fp16", "masked_softmax_rows",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, 1024",
     4, ["n_rows", "1", "1"]),
    ("bias_add_fp16", "bias_add",
     "*fp16, *fp16, *fp16, i32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("residual_add_f32", "residual_add_f32",
     "*fp16, *fp32, *fp32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("convert_f32_to_f16", "convert_f32_to_f16",
     "*fp32, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── Decoder: GEMV ──
    ("gemv_f16w", "gemv_f16w",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_bias_f16w", "gemv_bias_f16w",
     "*fp16, *fp16, *fp32, *fp16, i32, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "1", "1"]),

    # ── Decoder: Attention decode ──
    ("attention_decode_1d_d80", "attention_decode_1d_masked",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, fp32, i32, i32, i32, 128",
     4, ["n_q_heads", "1", "1"]),
    ("attention_decode_splitkv_partial", "attention_decode_splitkv_partial",
     "*fp16, *fp16, *fp16, *fp32, i32, i32, i32, fp32, i32, i32, i32, i32, i32, i32, 128",
     4, ["n_q_heads * n_splits", "1", "1"]),
    ("attention_decode_splitkv_reduce", "attention_decode_splitkv_reduce",
     "*fp32, *fp16, i32, i32, i32, i32, 128",
     4, ["n_q_heads", "1", "1"]),

    # ── Decoder: RoPE + cache ──
    ("rope_qk_cache_fused", "rope_qk_cache_fused",
     "*fp16, *fp16, *fp32, *fp16, i32, i32, i32, i32, i32, i32, 512",
     4, ["1", "1", "1"]),
    ("kv_cache_append", "kv_cache_append",
     "*fp16, *fp16, i32, i32, i32, i32, 256",
     4, ["cdiv(total_elems, 256)", "1", "1"]),

    # ── Decoder: Fused ops ──
    ("glu_silu_fused", "glu_silu_fused",
     "*fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("residual_add_layernorm_fused", "residual_add_layernorm_fused",
     "*fp16, *fp32, *fp32, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024",
     4, ["n_rows", "1", "1"]),
    ("gemv_bias_glu_fused", "gemv_bias_glu_fused",
     "*fp16, *fp16, *fp32, *fp16, i32, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_resadd_ln_fused", "gemv_resadd_ln_fused",
     "*fp16, *fp16, *fp32, *fp32, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024",
     4, ["1", "1", "1"]),

    # ── Decoder: Split-K GEMV ──
    ("gemv_splitk_partial", "gemv_splitk_partial",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, 128",
     4, ["n_n_blocks * n_splits", "1", "1"]),
    ("gemv_splitk_bias_reduce", "gemv_splitk_bias_reduce",
     "*fp16, *fp32, *fp16, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_splitk_reduce", "gemv_splitk_reduce",
     "*fp16, *fp16, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_splitk_reduce_resadd_ln", "gemv_splitk_reduce_resadd_ln",
     "*fp16, *fp32, *fp32, *fp16, *fp16, i32, i32, i32, 1e-5, 1024",
     4, ["1", "1", "1"]),
    ("gemv_qkv_splitk_partial", "gemv_qkv_splitk_partial",
     "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, 128",
     4, ["n_n_blocks * n_splits", "3", "1"]),
    ("gemv_qkv_splitk_reduce", "gemv_qkv_splitk_reduce",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "3", "1"]),
    ("gemv_glu_splitk_partial", "gemv_glu_splitk_partial",
     "*fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, 128",
     4, ["n_n_blocks * n_splits", "1", "1"]),
    ("gemv_glu_splitk_reduce", "gemv_glu_splitk_reduce",
     "*fp16, *fp32, *fp16, i32, i32, i32, 128",
     4, ["cdiv(N, 128)", "1", "1"]),
]


# ── HLSL-only configs (D3D12, includes extra variants) ───────────────────────

HLSL_EXTRA_KERNELS = [
    # fp16 accumulate matmul variants (~2x throughput on native fp16 ALUs)
    ("matmul_fp16_acc16_64x64x32", "matmul_fp16",
     "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"],
     {"force_acc_fp16": True}),
    ("matmul_bias_fp16_acc16_32x32x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"],
     {"force_acc_fp16": True}),

    # Mixed-precision matmul: f16 activation x f32 weight
    ("matmul_f16a_f32w_64x64x32", "matmul_f16a_f32w",
     "*fp16:16, *fp32:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_bias_f16a_f32w_32x32x32", "matmul_bias_f16a_f32w",
     "*fp16:16, *fp32:16, *fp32, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),
    ("matmul_f16a_f32w_32x32x32", "matmul_f16a_f32w",
     "*fp16:16, *fp32:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),

    # Extra layernorm variants for HLSL
    ("layernorm_unit_offset_f32in_768", "layernorm",
     "*fp32, *fp16, *fp16, i32, i32, i32, i32, 1e-5, 1024, 1",
     4, ["n_rows", "1", "1"]),

    # Conversion
    ("convert_f16_to_f32", "convert_f16_to_f32",
     "*fp16, *fp32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # Additional decoder HLSL kernels
    ("silu_fp16", "silu_forward",
     "*fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("silu_mul_fp16", "silu_mul_forward",
     "*fp16, *fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("scale_inplace_fp16", "scale_inplace",
     "*fp16, i32, fp32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),
    ("add_bias_f32_fp16", "add_bias_f32",
     "*fp16, *fp32, *fp16, i32, i32, i32, 1024",
     4, ["n_rows", "1", "1"]),

    # HLSL GEMV variants
    ("gemv_qkv_fused", "gemv_qkv_fused",
     "*fp16, *fp16, *fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, 128",
     4, ["cdiv(N * 3, 128)", "1", "1"]),
    ("rope_interleaved", "rope_interleaved",
     "*fp16, *fp32, i32, i32, i32, i32, 256",
     4, ["cdiv(total_pairs, 256)", "1", "1"]),
    ("rope_cache_fused", "rope_cache_fused",
     "*fp16, *fp16, *fp32, i32, i32, i32, i32, i32, 256",
     4, ["cdiv(total_pairs, 256)", "1", "1"]),
    ("rope_qk_cache_fused_hlsl", "rope_qk_cache_fused",
     "*fp16, *fp16, *fp32, *fp16, i32, i32, i32, i32, i32, i32, 512",
     4, ["1", "1", "1"]),

    # HLSL attention decode variants
    ("flash_attention_d64", "flash_attention_fwd",
     "*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, fp32, i32, i32, 32, 32, 64",
     4, ["cdiv(seq_len, 32)", "n_heads", "1"]),
    ("attention_decode_fwd", "attention_decode_fwd",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, fp32, i32, i32, i32, i32, 8, 64",
     4, ["n_q_heads", "1", "1"]),
    ("attention_decode_1d", "attention_decode_1d",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, fp32, i32, i32, 64",
     4, ["n_q_heads", "1", "1"]),
    ("attention_decode_1d_d80", "attention_decode_1d_masked",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, fp32, i32, i32, i32, 128",
     4, ["n_q_heads", "1", "1"]),

    # Q8 GEMV variants (HLSL only)
    ("gemv_q8", "gemv_q8",
     "*fp16, *i32, *fp16, *fp16, i32, i32, i32, 64, 32",
     4, ["cdiv(N, 64)", "1", "1"]),
    ("gemv_q8_bias", "gemv_q8_bias",
     "*fp16, *i32, *fp16, *fp32, *fp16, i32, i32, i32, 64, 32",
     4, ["cdiv(N, 64)", "1", "1"]),
    ("gemv_q8_v2", "gemv_q8",
     "*fp16, *i32, *fp16, *fp16, i32, i32, i32, 128, 32",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_q8_bias_v2", "gemv_q8_bias",
     "*fp16, *i32, *fp16, *fp32, *fp16, i32, i32, i32, 128, 32",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_q8_bias_glu_v2", "gemv_q8_bias_glu",
     "*fp16, *i32, *fp16, *fp32, *fp16, i32, i32, i32, 128, 32",
     4, ["cdiv(N, 128)", "1", "1"]),
    ("gemv_q8_resadd_ln_v2", "gemv_q8_resadd_ln",
     "*fp16, *i32, *fp16, *fp32, *fp32, *fp16, *fp16, i32, i32, i32, 1e-5, 1024, 32",
     4, ["1", "1", "1"]),
]


def get_metal_kernels():
    """Return kernel configs for Metal (Apple Silicon + Intel)."""
    return METAL_KERNELS


def get_hlsl_kernels():
    """Return kernel configs for HLSL/D3D12 (shared Metal + HLSL-only)."""
    # HLSL uses the same base kernels as Metal, plus extras
    all_configs = []
    for cfg in METAL_KERNELS:
        all_configs.append(cfg if len(cfg) > 5 else (*cfg, {}))
    for cfg in HLSL_EXTRA_KERNELS:
        all_configs.append(cfg if len(cfg) > 5 else (*cfg, {}))
    return all_configs
