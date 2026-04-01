"""Consolidated kernel configurations for all platforms.

Each config: (name, function_name, signature, num_warps, grid, options)
  options dict keys:
    targets: list of "metal", "metal_nosimd", "hlsl" (default: all three)
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

    # ── Matmul + bias (ACTIVATION: None, gelu, silu — resolved from kernel module) ──
    ("matmul_bias_fp16_32x32x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, None",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),
    ("matmul_bias_fp16_64x64x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32, None",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_bias_fp16_128x128x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32, None",
     8, ["cdiv(M, 128)", "cdiv(N, 128)", "1"]),
    ("matmul_bias_gelu_fp16_32x32x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, gelu",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),
    ("matmul_bias_gelu_fp16_64x64x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32, gelu",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_bias_gelu_fp16_128x128x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32, gelu",
     8, ["cdiv(M, 128)", "cdiv(N, 128)", "1"]),
    ("matmul_bias_silu_fp16_32x32x32", "matmul_bias_fp16",
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, silu",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"]),
    # ── Mixed precision (f16 activations, f32 weights) ──
    ("matmul_bias_gelu_f16a_f32w_32x32x32", "matmul_bias_f16a_f32w",
     "*fp16:16, *fp32:16, *fp32, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, gelu",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"],
     {"targets": ["hlsl"]}),
    ("matmul_bias_silu_f16a_f32w_32x32x32", "matmul_bias_f16a_f32w",
     "*fp16:16, *fp32:16, *fp32, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, silu",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"],
     {"targets": ["hlsl"]}),

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
     "*fp16:16, *fp16:16, *fp16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, None",
     4, ["cdiv(M, 32)", "cdiv(N, 32)", "1"],
     {"force_acc_fp16": True}),

    # Mixed-precision matmul: f16 activation x f32 weight
    ("matmul_f16a_f32w_64x64x32", "matmul_f16a_f32w",
     "*fp16:16, *fp32:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32",
     4, ["cdiv(M, 64)", "cdiv(N, 64)", "1"]),
    ("matmul_bias_f16a_f32w_32x32x32", "matmul_bias_f16a_f32w",
     "*fp16:16, *fp32:16, *fp32, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 32, 32, 32, None",
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


# ── Kernel metadata for Rust codegen ────────────────────────────────────────
# Maps kernel name → metadata used by gen_rust.py to generate structs + load().
#   alias:    Rust field name in the pipeline struct
#   group:    "encoder" or "decoder" (which struct it belongs to)
#   optional: True → Option<Pipeline>, loads with .ok() instead of ?
#   tg_mem:   AIR threadgroup memory in bytes (aarch64 only, 0 = none)
#   d3d12:    True → include in D3D12 kernel set (generates DXIL const + PSO field)
#   d3d12_alias: D3D12 field name override (if different from alias)

KERNEL_METADATA = {
    # ── Encoder kernels ──
    "matmul_fp16_64x64x32": {
        "alias": "matmul_64x64", "group": "encoder", "d3d12": True,
    },
    "matmul_fp16_128x128x32": {
        "alias": "matmul_128x128", "group": "encoder", "optional": True, "d3d12": True,
    },
    "matmul_bias_fp16_32x32x32": {
        "alias": "matmul_bias_32x32", "group": "encoder", "d3d12": True,
    },
    "matmul_bias_fp16_64x64x32": {
        "alias": "matmul_bias_64x64", "group": "encoder", "optional": True,
    },
    "matmul_bias_fp16_128x128x32": {
        "alias": "matmul_bias_128x128", "group": "encoder", "optional": True,
    },
    "matmul_bias_gelu_fp16_32x32x32": {
        "alias": "matmul_bias_gelu_32x32", "group": "encoder",
    },
    "matmul_bias_gelu_fp16_64x64x32": {
        "alias": "matmul_bias_gelu_64x64", "group": "encoder", "optional": True,
    },
    "matmul_bias_gelu_fp16_128x128x32": {
        "alias": "matmul_bias_gelu_128x128", "group": "encoder", "optional": True,
    },
    "matmul_bias_gelu_f16a_f32w_32x32x32": {
        "alias": "matmul_bias_gelu_f32w_32x32", "group": "encoder", "d3d12": True,
    },
    "matmul_bias_silu_fp16_32x32x32": {
        "alias": "matmul_bias_silu_32x32", "group": "decoder",
    },
    "matmul_bias_silu_f16a_f32w_32x32x32": {
        "alias": "matmul_bias_silu_f32w_32x32", "group": "decoder", "d3d12": True,
    },
    "flash_attention_fwd_32x32x64": {
        "alias": "flash_attention", "group": "encoder", "tg_mem": 8192,
    },
    "layernorm_unit_offset_768": {
        "alias": "layernorm_unit_offset", "group": "encoder", "tg_mem": 4096,
        "d3d12": True,
    },
    "layernorm_bare_768": {
        "alias": "layernorm_bare", "group": "encoder", "optional": True, "tg_mem": 4096,
    },
    "layernorm_standard_768": {
        "alias": "layernorm_standard_768", "group": "encoder", "tg_mem": 4096,
    },
    "layernorm_standard_640": {
        "alias": "layernorm_standard_640", "group": "encoder", "tg_mem": 4096,
    },
    "layernorm_standard_f32in_640": {
        "alias": "layernorm_std_f32in", "group": "decoder", "tg_mem": 4096,
    },
    "gelu_fp16": {
        "alias": "gelu", "group": "encoder", "d3d12": True,
    },
    "residual_add_fp16": {
        "alias": "residual_add", "group": "encoder", "d3d12": True,
    },
    "softmax_fp16": {
        "alias": "softmax", "group": "encoder", "d3d12": True,
    },
    "masked_softmax_fp16": {
        "alias": "masked_softmax", "group": "encoder",
    },
    "bias_add_fp16": {
        "alias": "bias_add", "group": "encoder",
    },
    "residual_add_f32": {
        "alias": "residual_add_f32", "group": "encoder", "d3d12": True,
    },
    "convert_f32_to_f16": {
        "alias": "convert_f32_to_f16", "group": "decoder", "d3d12": True,
    },


    # ── Decoder kernels ──
    "gemv_f16w": {
        "alias": "gemv_f16w", "group": "decoder", "d3d12": True,
    },
    "gemv_bias_f16w": {
        "alias": "gemv_bias_f16w", "group": "decoder", "d3d12": True,
    },
    "attention_decode_1d_d80": {
        "alias": "attention_decode", "group": "decoder", "tg_mem": 512, "d3d12": True,
    },
    "attention_decode_splitkv_partial": {
        "alias": "attention_splitkv_partial", "group": "decoder",
    },
    "attention_decode_splitkv_reduce": {
        "alias": "attention_splitkv_reduce", "group": "decoder",
    },
    "rope_qk_cache_fused": {
        "alias": "rope_qk_cache_fused", "group": "decoder", "d3d12": True,
    },
    "kv_cache_append": {
        "alias": "kv_cache_append", "group": "decoder", "d3d12": True,
    },
    "glu_silu_fused": {
        "alias": "glu_silu", "group": "decoder", "d3d12": True,
    },
    "residual_add_layernorm_fused": {
        "alias": "residual_add_layernorm", "group": "decoder", "tg_mem": 4096,
        "d3d12": True,
    },
    "gemv_bias_glu_fused": {
        "alias": "gemv_bias_glu", "group": "decoder", "d3d12": True,
    },
    "gemv_resadd_ln_fused": {
        "alias": "gemv_resadd_ln", "group": "decoder", "tg_mem": 4096, "d3d12": True,
    },
    "gemv_splitk_partial": {
        "alias": "gemv_splitk_partial", "group": "decoder",
    },
    "gemv_splitk_bias_reduce": {
        "alias": "gemv_splitk_bias_reduce", "group": "decoder",
    },
    "gemv_splitk_reduce": {
        "alias": "gemv_splitk_reduce", "group": "decoder",
    },
    "gemv_splitk_reduce_resadd_ln": {
        "alias": "gemv_splitk_reduce_resadd_ln", "group": "decoder", "tg_mem": 4096,
    },
    "gemv_qkv_splitk_partial": {
        "alias": "gemv_qkv_splitk_partial", "group": "decoder",
    },
    "gemv_qkv_splitk_reduce": {
        "alias": "gemv_qkv_splitk_reduce", "group": "decoder",
    },
    "gemv_glu_splitk_partial": {
        "alias": "gemv_glu_splitk_partial", "group": "decoder",
    },
    "gemv_glu_splitk_reduce": {
        "alias": "gemv_glu_splitk_reduce", "group": "decoder",
    },

    # ── HLSL-only encoder kernels ──
    "matmul_fp16_acc16_64x64x32": {
        "alias": "matmul_acc16_64x64", "group": "encoder", "optional": True,
        "d3d12": True, "d3d12_only": True,
    },
    "matmul_bias_fp16_acc16_32x32x32": {
        "alias": "matmul_bias_acc16_32x32", "group": "encoder", "optional": True,
        "d3d12": True, "d3d12_only": True,
    },
    "matmul_f16a_f32w_64x64x32": {
        "alias": "matmul_f32w_64x64", "group": "encoder",
        "d3d12": True, "d3d12_only": True,
    },
    "matmul_bias_f16a_f32w_32x32x32": {
        "alias": "matmul_bias_f32w_32x32", "group": "encoder",
        "d3d12": True, "d3d12_only": True,
    },
    "matmul_f16a_f32w_32x32x32": {
        "alias": "matmul_f32w_32x32", "group": "encoder",
        "d3d12": True, "d3d12_only": True,
    },
    "layernorm_unit_offset_f32in_768": {
        "alias": "layernorm_f32in", "group": "encoder",
        "d3d12": True, "d3d12_only": True,
    },
    "convert_f16_to_f32": {
        "alias": "convert_f16_to_f32", "group": "encoder",
        "d3d12": True, "d3d12_only": True,
    },

    # ── HLSL-only decoder kernels ──
    "silu_fp16": {
        "alias": "silu", "group": "encoder", "d3d12": True, "d3d12_only": True,
    },
    "silu_mul_fp16": {
        "alias": "silu_mul", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "scale_inplace_fp16": {
        "alias": "scale_inplace", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "add_bias_f32_fp16": {
        "alias": "add_bias_f32", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "gemv_qkv_fused": {
        "alias": "gemv_qkv_fused", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "rope_interleaved": {
        "alias": "rope_interleaved", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "rope_cache_fused": {
        "alias": "rope_cache_fused", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "rope_qk_cache_fused_hlsl": {
        "alias": "rope_qk_cache_fused_hlsl", "group": "decoder",
        "d3d12": True, "d3d12_only": True,
    },
    "flash_attention_d64": {
        "alias": "flash_attn_d64", "group": "encoder",
        "d3d12": True, "d3d12_only": True,
    },
    "attention_decode_fwd": {
        "alias": "attention_decode_fwd", "group": "decoder",
        "d3d12": True, "d3d12_only": True,
    },
    "attention_decode_1d": {
        "alias": "attention_decode_1d", "group": "decoder",
        "d3d12": True, "d3d12_only": True,
    },
    "gemv_q8": {
        "alias": "gemv_q8", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "gemv_q8_bias": {
        "alias": "gemv_q8_bias", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "gemv_q8_v2": {
        "alias": "gemv_q8_v2", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "gemv_q8_bias_v2": {
        "alias": "gemv_q8_bias_v2", "group": "decoder", "d3d12": True, "d3d12_only": True,
    },
    "gemv_q8_bias_glu_v2": {
        "alias": "gemv_q8_bias_glu_v2", "group": "decoder",
        "d3d12": True, "d3d12_only": True,
    },
    "gemv_q8_resadd_ln_v2": {
        "alias": "gemv_q8_resadd_ln_v2", "group": "decoder",
        "d3d12": True, "d3d12_only": True,
    },
}


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
