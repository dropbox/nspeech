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
    ("kokoro_adain_snake_1024", "adain_snake_fused",
     "*fp16, *fp16, *fp16, *fp16, *fp16, i32, i32, 1024",
     4, ["n_channels", "1", "1"]),

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

    # ── Element-wise add: out = a + b ──
    ("kokoro_add", "elementwise_add",
     "*fp16, *fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

    # ── Element-wise scale by 1/3 (for resblock averaging) ──
    ("kokoro_scale_third", "elementwise_scale_third",
     "*fp16, *fp16, i32, 1024",
     4, ["cdiv(n_elements, 1024)", "1", "1"]),

]


# Metadata lives in kernel_configs.py KERNEL_METADATA (group="kokoro", d3d12=True)
