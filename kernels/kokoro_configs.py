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

]


# ── Kernel metadata for Rust codegen ────────────────────────────────────────

KOKORO_METADATA = {
    "kokoro_snake_activation": {
        "alias": "snake", "group": "kokoro",
    },
    "kokoro_adain_snake_1024": {
        "alias": "adain_snake_1k", "group": "kokoro",
    },
    "kokoro_leaky_relu_01": {
        "alias": "leaky_relu_01", "group": "kokoro",
    },
    "kokoro_leaky_relu_02": {
        "alias": "leaky_relu_02", "group": "kokoro",
    },
    "kokoro_leaky_relu_001": {
        "alias": "leaky_relu_001", "group": "kokoro",
    },
}
