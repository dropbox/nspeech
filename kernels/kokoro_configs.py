"""Kernel configurations for Kokoro TTS decoder.

Each config: (name, function_name, signature, num_warps, grid, options)

Kokoro decoder dimensions:
  - Channels: 128, 256, 512 (after upsampling: 256, 128)
  - Sequence length: variable, 30-100 input, up to ~20000 after upsample
  - Conv kernels: 3, 7, 11 (resblocks), 20/12 (upsample), 1 (projections)
  - Style dim: 128
"""

# ── Kokoro Metal kernel configs ──────────────────────────────────────────────

KOKORO_KERNELS = [
    # ── Snake activation: x + sin²(αx)/α ──
    # For 128-ch after upsample (up to 20000 timesteps): 128*20000 = 2.56M elements
    ("kokoro_snake_activation", "snake_activation",
     "*fp16, *fp16, *fp16, i32, i32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── Fused AdaIN + Snake (saves one full pass over data) ──
    # 1k: single-pass for seq_len <= 1024
    ("kokoro_adain_snake_1024", "adain_snake_fused",
     "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, 1024",
     4, ["n_channels", "1", "1"]),

    # 2k: looped for seq_len <= 2048 (stage0: 256ch x 1240t)
    ("kokoro_adain_snake_2048", "adain_snake_looped",
     "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, 1024, 2048",
     4, ["n_channels", "1", "1"]),

    # 8k: looped for seq_len <= 8192 (stage1: 128ch x 7441t)
    ("kokoro_adain_snake_8192", "adain_snake_looped",
     "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, 1024, 8192",
     4, ["n_channels", "1", "1"]),

    # ── Two-pass AdaIN+Snake (split reduction from element-wise) ──
    # Pass 1: instance norm stats (mean+rstd per channel)
    ("kokoro_instance_norm_stats_2k", "instance_norm_stats",
     "*fp16, *fp32, i32, i32, 1024, 2048",
     4, ["n_channels", "1", "1"]),

    ("kokoro_instance_norm_stats_8k", "instance_norm_stats",
     "*fp16, *fp32, i32, i32, 1024, 8192",
     4, ["n_channels", "1", "1"]),

    ("kokoro_instance_norm_stats_32k", "instance_norm_stats",
     "*fp16, *fp32, i32, i32, 1024, 32768",
     4, ["n_channels", "1", "1"]),

    # Pass 2: normalize + style + snake (element-wise)
    ("kokoro_norm_style_snake", "norm_style_snake",
     "*fp16, *fp32, *fp16, *fp16, *fp16, *fp16, i32, i32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),


    # ── LeakyReLU ──
    ("kokoro_leaky_relu_01", "leaky_relu_fp16",
     "*fp16, *fp16, i32, 0.1, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    ("kokoro_leaky_relu_02", "leaky_relu_fp16",
     "*fp16, *fp16, i32, 0.2, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    ("kokoro_leaky_relu_001", "leaky_relu_fp16",
     "*fp16, *fp16, i32, 0.01, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── Conv1d (simple, runtime K): one threadgroup per (C_out, t_block) ──
    ("kokoro_conv1d", "conv1d_simple",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, 256",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── Conv1d with constexpr K (unrolled inner loop) ──
    ("kokoro_conv1d_k3", "conv1d_k",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 256, 3",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    ("kokoro_conv1d_k7", "conv1d_k",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 256, 7",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    ("kokoro_conv1d_k11", "conv1d_k",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 256, 11",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── ConvTranspose1d (simple): one threadgroup per (C_out, t_block) ──
    ("kokoro_conv_transpose1d", "conv_transpose1d_simple",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 256",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── Fused LeakyReLU(0.1) + ConvTranspose1d (saves a full buffer pass) ──
    ("kokoro_conv_transpose1d_lrelu", "conv_transpose1d_act",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 256, leaky_relu_01_act",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── Fused LeakyReLU(0.01) + Conv1d (for final conv_post) ──
    ("kokoro_conv1d_lrelu001", "conv1d_act",
     "*fp16, *fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, 256, leaky_relu_001_act",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── Reflection pad1d (pad_left=1, pad_right=0) ──
    ("kokoro_reflection_pad1d", "reflection_pad1d_left1",
     "*fp16, *fp16, i32, i32, 1024",
     4, ["cdiv(n_channels * (seq_len + 1), 1024)", "1", "1"]),

    # ── Im2col for conv1d (enables matmul-based convolution) ──
    ("kokoro_im2col", "im2col_conv1d",
     "*fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 1024",
     4, ["cdiv(C_in * K * T_out, 1024)", "1", "1"]),

    # ── Im2col with fused LeakyReLU(0.1) on input ──
    ("kokoro_im2col_lrelu", "im2col_conv1d_act",
     "*fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, 1024, leaky_relu_01_act",
     4, ["cdiv(C_in * K * T_out, 1024)", "1", "1"]),

    # ── Element-wise add: out = a + b ──
    ("kokoro_add", "elementwise_add",
     "*fp16, *fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── Element-wise scale by 1/3 (for resblock averaging) ──
    ("kokoro_scale_third", "elementwise_scale_third",
     "*fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── Row-broadcast bias add: out[i] = x[i] + bias[i / n_cols] ──
    ("kokoro_row_bias_add", "row_bias_add",
     "*fp16, *fp16, *fp16, i32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── F32-intermediate variants (prevents precision loss through normalization) ──

    # Conv1d with f32 input/output, f16 weights
    ("kokoro_conv1d_f32io", "conv1d_f32io",
     "*fp32, *fp16, *fp16, *fp32, i32, i32, i32, i32, i32, i32, i32, i32, 256",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # Instance norm stats from f32 input (2k variant)
    ("kokoro_instance_norm_stats_f32in_2k", "instance_norm_stats_f32in",
     "*fp32, *fp32, i32, i32, 1024, 2048",
     4, ["n_channels", "1", "1"]),

    # Instance norm stats from f32 input (8k variant)
    ("kokoro_instance_norm_stats_f32in_8k", "instance_norm_stats_f32in",
     "*fp32, *fp32, i32, i32, 1024, 8192",
     4, ["n_channels", "1", "1"]),

    # Instance norm stats from f32 input (32k variant, two-pass for numerical stability)
    ("kokoro_instance_norm_stats_f32in_32k", "instance_norm_stats_f32in_twopass",
     "*fp32, *fp32, i32, i32, 1024, 32768",
     4, ["n_channels", "1", "1"]),

    # Instance norm stats from f32 input (64k variant, two-pass for numerical stability)
    ("kokoro_instance_norm_stats_f32in_64k", "instance_norm_stats_f32in_twopass",
     "*fp32, *fp32, i32, i32, 1024, 65536",
     4, ["n_channels", "1", "1"]),

    # Instance norm stats from f32 input (128k variant, two-pass for numerical stability)
    ("kokoro_instance_norm_stats_f32in_128k", "instance_norm_stats_f32in_twopass",
     "*fp32, *fp32, i32, i32, 1024, 131072",
     4, ["n_channels", "1", "1"]),

    # Normalize + style + snake: f32 in, f32 out
    ("kokoro_norm_style_snake_f32io", "norm_style_snake_f32io",
     "*fp32, *fp32, *fp16, *fp16, *fp16, *fp32, i32, i32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # Element-wise add of f32 buffers
    ("kokoro_add_f32", "elementwise_add_f32",
     "*fp32, *fp32, *fp32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # Scale f32 by 1/3
    ("kokoro_scale_third_f32", "elementwise_scale_third_f32",
     "*fp32, *fp32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # Convert f32 → f16
    ("kokoro_f32_to_f16", "convert_f32_to_f16_kernel",
     "*fp32, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # Convert f16 → f32
    ("kokoro_f16_to_f32", "convert_f16_to_f32_kernel",
     "*fp16, *fp32, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── LeakyReLU on f32 buffers ──
    ("kokoro_leaky_relu_f32_001", "leaky_relu_f32",
     "*fp32, *fp32, i32, 0.01, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    ("kokoro_leaky_relu_f32_01", "leaky_relu_f32",
     "*fp32, *fp32, i32, 0.1, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── ConvTranspose1d with f32 I/O (fused LeakyReLU(0.1) on input) ──
    ("kokoro_conv_transpose1d_f32io_lrelu", "conv_transpose1d_f32io",
     "*fp32, *fp16, *fp16, *fp32, i32, i32, i32, i32, i32, i32, i32, i32, 256, leaky_relu_01_act",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── ConvTranspose1d with f32 I/O (no activation) ──
    ("kokoro_conv_transpose1d_f32io", "conv_transpose1d_f32io",
     "*fp32, *fp16, *fp16, *fp32, i32, i32, i32, i32, i32, i32, i32, i32, 256, None",
     4, ["C_out", "cdiv(T_out, 256)", "1"]),

    # ── Im2col f32→f16: read f32 input, write f16 im2col output ──
    ("kokoro_im2col_f32_to_f16", "im2col_f32_to_f16",
     "*fp32, *fp16, i32, i32, i32, i32, i32, i32, i32, 1024",
     4, ["cdiv(C_in * K * T_out, 1024)", "1", "1"]),

    # ── Reflection pad1d (pad_left=1, pad_right=0) for f32 buffers ──
    ("kokoro_reflection_pad1d_f32", "reflection_pad1d_f32",
     "*fp32, *fp32, i32, i32, 1024",
     4, ["cdiv(n_channels * (seq_len + 1), 1024)", "1", "1"]),

]


# Metadata lives in kernel_configs.py KERNEL_METADATA (group="kokoro", d3d12=True)
