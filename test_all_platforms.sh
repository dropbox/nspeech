#!/usr/bin/env bash
# Build and test Moonshine transcription on all 3 platforms:
#   1. M2 Mac (local)        — triton-metal
#   2. Intel Mac (ssh mac)   — triton-metal
#   3. Windows (ssh windows) — triton-d3d12
#
# Usage: ./test_all_platforms.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXAMPLE=transcribe_moonshine
WAV=dots.wav
PASS=0
FAIL=0
RESULTS=""

header() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$1"; }
ok()     { PASS=$((PASS+1)); RESULTS+="  PASS  $1\n"; }
fail()   { FAIL=$((FAIL+1)); RESULTS+="  FAIL  $1\n"; }

# ── 1. Build kernels + Rust binaries ───────────────────────────────────────

#header "Building kernels"
## Generate TTIR + ninja rules (if sources changed)
#(cd kernels && python build.py) 2>&1 | tail -5

header "Building for M2 Mac (aarch64-apple-darwin)"
cargo build --release --features triton-metal --example $EXAMPLE 2>&1 \
  | grep -E 'Compiling speech|Finished|error' || true

header "Building for Intel Mac (x86_64-apple-darwin)"
cargo build --release --features triton-metal --example $EXAMPLE \
  --target x86_64-apple-darwin 2>&1 \
  | grep -E 'Compiling speech|Finished|error' || true

header "Building for Windows (x86_64-pc-windows-msvc)"
cargo xwin build --release --features triton-d3d12 --example $EXAMPLE \
  --target x86_64-pc-windows-msvc 2>&1 \
  | grep -E 'Compiling speech|Finished|error' || true

# ── 2. Deploy to remote hosts ──────────────────────────────────────────────

header "Deploying binaries"
echo "  Intel Mac..."
ecp target/x86_64-apple-darwin/release/examples/$EXAMPLE mac:speech/
echo "  Windows..."
ecp target/x86_64-pc-windows-msvc/release/examples/$EXAMPLE.exe windows:candle/

# ── 3. Run on M2 Mac (local) ──────────────────────────────────────────────

header "M2 Mac (local)"
OUTPUT=$(target/release/examples/$EXAMPLE "$WAV" assets 3 2>&1) || true
echo "$OUTPUT" | tail -5
if echo "$OUTPUT" | grep -q 'RTF'; then
  ok "M2 Mac"
else
  echo "$OUTPUT"
  fail "M2 Mac"
fi

# ── 4. Run on Intel Mac ───────────────────────────────────────────────────

header "Intel Mac (ssh mac)"
OUTPUT=$(ssh mac "cd speech && ./$EXAMPLE $WAV assets 3" 2>&1) || true
echo "$OUTPUT" | tail -5
if echo "$OUTPUT" | grep -q 'RTF'; then
  ok "Intel Mac"
else
  echo "$OUTPUT"
  fail "Intel Mac"
fi

# ── 5. Run on Windows ─────────────────────────────────────────────────────

header "Windows (ssh windows)"
OUTPUT=$(ssh windows "cd candle && ./$EXAMPLE.exe $WAV assets 3" 2>&1) || true
echo "$OUTPUT" | tail -5
if echo "$OUTPUT" | grep -q 'RTF'; then
  ok "Windows"
else
  echo "$OUTPUT"
  fail "Windows"
fi

# ── Summary ────────────────────────────────────────────────────────────────

header "Summary"
printf "$RESULTS"
echo ""
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
