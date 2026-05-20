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
else
  # Windows (MSYS2/Git Bash): D3D12 GPU backend
  FEATURES := triton-d3d12
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

.DEFAULT_GOAL := build

# === Python environment ===

env/pyvenv.cfg:
	uv venv env

env/.requirements.stamp: env/pyvenv.cfg requirements.txt
	(source env/*/activate && uv pip install -r requirements.txt)
	touch env/.requirements.stamp

# === Build targets ===

build: assets
	cargo build --release $(BUILD_TARGET) --features $(FEATURES) --examples

kokoro: build
	ln -sf $(BIN_DIR)/synthesize_kokoro ./synthesize_kokoro

speek:
	cargo build --release $(BUILD_TARGET) --features $(FEATURES) --example speek
	ln -sf $(BIN_DIR)/speek ./speek

listen:
	cargo build --release $(BUILD_TARGET) --features $(FEATURES) --example listen
	ln -sf $(BIN_DIR)/listen ./listen

speek-install: speek
	mkdir -p ~/bin ~/.claude/skills/speek ~/.codex/skills/speek
	ln -f $(realpath speek) ~/bin/speek
	rm -f ~/.claude/skills/speek/skill.md ~/.codex/skills/speek/skill.md
	cp skills/speek-claude/SKILL.md ~/.claude/skills/speek/.SKILL.md.tmp && mv ~/.claude/skills/speek/.SKILL.md.tmp ~/.claude/skills/speek/SKILL.md
	cp skills/speek-codex/SKILL.md ~/.codex/skills/speek/.SKILL.md.tmp && mv ~/.codex/skills/speek/.SKILL.md.tmp ~/.codex/skills/speek/SKILL.md

listen-install: listen
	mkdir -p ~/bin ~/.claude/skills/listen ~/.codex/skills/listen
	ln -f $(realpath listen) ~/bin/listen
	rm -f ~/.claude/skills/listen/skill.md ~/.codex/skills/listen/skill.md
	cp skills/listen-claude/SKILL.md ~/.claude/skills/listen/.SKILL.md.tmp && mv ~/.claude/skills/listen/.SKILL.md.tmp ~/.claude/skills/listen/SKILL.md
	cp skills/listen-codex/SKILL.md ~/.codex/skills/listen/.SKILL.md.tmp && mv ~/.codex/skills/listen/.SKILL.md.tmp ~/.codex/skills/listen/SKILL.md

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

# === Model assets ===

# Download and quantize Kokoro TTS model → assets/
assets/kokoro_q8_0.gguf: hf_kokoro/kokoro-v1_0.safetensors
	cargo run -p quantize-kokoro --release

hf_kokoro/kokoro-v1_0.safetensors: hf_kokoro/kokoro-v1_0.pth
	(source env/*/activate && python scripts/convert_kokoro_safetensors.py)

hf_kokoro/kokoro-v1_0.pth: env/.requirements.stamp
	(source env/*/activate && python scripts/download_kokoro.py)

assets/kokoro-config.json: hf_kokoro/config.json
	@mkdir -p assets
	cp $< $@

assets/kokoro-af_heart.safetensors: hf_kokoro/voices/af_heart.safetensors
	@mkdir -p assets
	cp $< $@

assets/us_gold.json: hf_kokoro/us_gold.json
	@mkdir -p assets
	cp $< $@

assets/us_silver.json: hf_kokoro/us_silver.json
	@mkdir -p assets
	cp $< $@

hf_kokoro/config.json hf_kokoro/voices/af_heart.safetensors hf_kokoro/us_gold.json hf_kokoro/us_silver.json: hf_kokoro/kokoro-v1_0.pth

# Download and quantize Moonshine V2 → assets/
assets/moonshine_q8_0.gguf: env/.requirements.stamp
	(source env/*/activate && python scripts/prepare_moonshine.py)

# Download and quantize Parakeet TDT → assets/
assets/parakeet-tdt-model_q8_0.gguf: env/.requirements.stamp
	(source env/*/activate && python scripts/download_parakeet_tdt.py)

# Download and quantize Silero VAD → assets/
assets/vad16_q8_0.gguf: env/.requirements.stamp
	(source env/*/activate && python scripts/download_vad.py)

assets: assets/kokoro_q8_0.gguf assets/kokoro-config.json assets/kokoro-af_heart.safetensors assets/us_gold.json assets/us_silver.json assets/moonshine_q8_0.gguf assets/parakeet-tdt-model_q8_0.gguf assets/vad16_q8_0.gguf

# Kernel compilation (Triton → Metal/DXIL)
kernels:
	cd kernels && python build.py

.PHONY: build kokoro speek listen speek-install listen-install bench win deploy-win test-win bench-win module kernels assets
