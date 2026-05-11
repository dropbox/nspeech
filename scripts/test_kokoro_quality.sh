#!/bin/bash
# Cross-platform Kokoro GPU/CPU quality verification.
# Builds test_long_sentence for all 3 targets, deploys, runs, and reports SNR.
#
# Platforms:
#   1. Apple Silicon (local) — triton-metal GPU + CPU
#   2. Intel Mac (ssh mac)  — CPU only (no Metal on Intel)
#   3. Windows (ssh windows) — triton-d3d12 GPU + CPU

set -e

cd "$(dirname "$0")/.."
PROJ="$(pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

pass=0
fail=0

report() {
    local platform="$1" output="$2"
    echo -e "${BOLD}=== $platform ===${NC}"
    while IFS= read -r line; do
        if echo "$line" | grep -q "FAIL"; then
            echo -e "  ${RED}$line${NC}"
            ((fail++))
        elif echo "$line" | grep -q "PASS"; then
            echo -e "  ${GREEN}$line${NC}"
            ((pass++))
        fi
    done <<< "$output"
}

# ── 1. Apple Silicon (local, Metal GPU) ──────────────────────────────────────
echo "Building for Apple Silicon (triton-metal)..."
cargo build --release --example test_long_sentence --features triton-metal 2>/dev/null

echo "Running on Apple Silicon..."
AARCH64_OUT=$(cargo run --release --example test_long_sentence --features triton-metal 2>&1 | grep "^rep=")
report "Apple Silicon (Metal GPU)" "$AARCH64_OUT"

# ── 2. Intel Mac (x86_64, Metal GPU) ────────────────────────────────────────
echo ""
echo "Cross-compiling for Intel Mac (triton-metal)..."
cargo build --release --example test_long_sentence --target x86_64-apple-darwin --features triton-metal 2>/dev/null

echo "Deploying to Intel Mac..."
ecp target/x86_64-apple-darwin/release/examples/test_long_sentence mac:speech/test_long_sentence

echo "Running on Intel Mac..."
INTELMAC_OUT=$(ssh mac "cd speech && ./test_long_sentence" 2>&1 | grep "^rep=")
report "Intel Mac (Metal GPU)" "$INTELMAC_OUT"

# ── 3. Windows (x86_64-pc-windows-msvc, D3D12 GPU) ──────────────────────────
echo ""
echo "Cross-compiling for Windows (triton-d3d12)..."
cargo xwin build --release --example test_long_sentence --target x86_64-pc-windows-msvc --features triton-d3d12 2>/dev/null

echo "Deploying to Windows..."
ecp target/x86_64-pc-windows-msvc/release/examples/test_long_sentence.exe windows:speech/test_long_sentence.exe

echo "Running on Windows..."
WIN_OUT=$(ssh windows "cd speech && ./test_long_sentence.exe" 2>&1 | grep "^rep=")
report "Windows (D3D12 GPU)" "$WIN_OUT"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Summary: ${GREEN}${pass} passed${NC}, ${RED}${fail} failed${NC}"
[ "$fail" -eq 0 ] && exit 0 || exit 1
