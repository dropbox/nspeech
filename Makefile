SHELL := /bin/bash

# Auto-detect platform and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Determine features based on platform
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    # Apple Silicon: Metal GPU + Accelerate BLAS + Triton Metal kernels
    FEATURES := triton-metal
    WIN_FEATURES := triton-d3d12
    RUSTFLAGS_EXTRA :=
    TARGET := aarch64-apple-darwin
  else
    # Intel Mac: Metal GPU (Intel UHD 630, no simdgroup_matrix)
    FEATURES := triton-metal
    WIN_FEATURES := triton-d3d12
    RUSTFLAGS_EXTRA :=
    TARGET := x86_64-apple-darwin
  endif
else ifeq ($(UNAME_S),Linux)
  FEATURES := fbgemm-bf16
  WIN_FEATURES := triton-d3d12
  RUSTFLAGS_EXTRA :=
  TARGET :=
endif

ifdef TARGET
  BIN_DIR := target/$(TARGET)/release/examples
  BUILD_TARGET := --target $(TARGET)
else
  BIN_DIR := target/release/examples
  BUILD_TARGET :=
endif

export RUSTFLAGS += $(RUSTFLAGS_EXTRA)

# === Build targets ===

build:
	cargo build --release $(BUILD_TARGET) --features $(FEATURES) --example synthesize_kokoro

kokoro: build
	ln -sf $(BIN_DIR)/synthesize_kokoro ./synthesize_kokoro

bench:
	cargo build --release $(BUILD_TARGET) --features $(FEATURES) --example bench_triton_encoder
	$(BIN_DIR)/bench_triton_encoder

# Cross-compile for Windows (D3D12 GPU backend)
win:
	cargo xwin build --release --target x86_64-pc-windows-msvc --features $(WIN_FEATURES) --example synthesize_kokoro
	cargo xwin build --release --target x86_64-pc-windows-msvc --features $(WIN_FEATURES) --example bench_triton_d3d12

# Deploy Windows binaries to remote machine
deploy-win: win
	ecp target/x86_64-pc-windows-msvc/release/examples/synthesize_kokoro.exe windows:speech/
	ecp target/x86_64-pc-windows-msvc/release/examples/bench_triton_d3d12.exe windows:speech/

# Build + deploy + run TTS on Windows
test-win: deploy-win
	ssh windows "cd speech && ./synthesize_kokoro.exe \"Hello from the GPU\" output_test.wav"
	ecp windows:speech/output_test.wav /tmp/windows_test.wav
	afplay /tmp/windows_test.wav

# Build + deploy + run benchmark on Windows
bench-win: deploy-win
	ssh windows "cd speech && ./bench_triton_d3d12.exe"

# Node.js native module (macOS)
module:
	cargo build --release $(BUILD_TARGET) --lib --features $(FEATURES)

# Kernel compilation (Triton → Metal/DXIL)
kernels:
	(cd kernels && python build.py)

.PHONY: build kokoro bench win deploy-win test-win bench-win module kernels
