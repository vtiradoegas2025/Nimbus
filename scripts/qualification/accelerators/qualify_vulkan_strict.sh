#!/usr/bin/env bash
# Fail-closed strict-validation Vulkan qualification on one usable device.
set -euo pipefail

cd "$(dirname "$0")/../../.."

BIN=${NIMBUS_BIN:-bin/nimbus_sim}
CFG=${NIMBUS_VULKAN_STRICT_CONFIG:-tests/fixtures/configs/qualification/gpu_residency.yaml}
DURATION=${NIMBUS_VULKAN_STRICT_DURATION:-12}
LOG=$(mktemp /tmp/nimbus_vulkan_strict_XXXXXX.log)
trap 'rm -f "$LOG"' EXIT

if [ ! -x "$BIN" ]; then
    echo "[vulkan-strict] missing executable: $BIN" >&2
    exit 1
fi

NIMBUS_GPU_KERNEL_TIMING=1 \
NIMBUS_GPU_RESIDENT=1 \
NIMBUS_GPU_LIN_BATCH=1 \
    ./scripts/run_with_vulkan_env.sh "$BIN" \
    --config "$CFG" \
    --compute-backend vulkan \
    --guard-mode strict \
    --headless \
    --duration="$DURATION" >"$LOG" 2>&1

if ! grep -q 'selected device' "$LOG"; then
    echo "[vulkan-strict] no Vulkan device selected" >&2
    tail -30 "$LOG" >&2
    exit 1
fi
if grep -q 'Falling back to CPU backend' "$LOG"; then
    echo "[vulkan-strict] Vulkan backend initialization fell back to CPU" >&2
    tail -30 "$LOG" >&2
    exit 1
fi
if ! grep -q 'guard_mode=strict' "$LOG"; then
    echo "[vulkan-strict] strict validation was not active" >&2
    tail -30 "$LOG" >&2
    exit 1
fi
if grep -q '\[PHYSICS GUARD FIELD\]' "$LOG"; then
    echo "[vulkan-strict] strict run observed a primary-state violation" >&2
    grep '\[PHYSICS GUARD FIELD\]' "$LOG" >&2
    exit 1
fi

TELEMETRY=$(grep '\[GPU TELEMETRY\]' "$LOG" | tail -1 || true)
if [ -z "$TELEMETRY" ]; then
    echo "[vulkan-strict] missing GPU telemetry summary" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

metric() {
    printf '%s\n' "$TELEMETRY" | sed -E "s/.* $1=([0-9]+).*/\\1/"
}

MODEL_STEPS=$(metric model_steps)
COMMAND_BUFFERS=$(metric command_buffers)
DISPATCHES=$(metric dispatches)
if [ "$MODEL_STEPS" -le 0 ] || [ "$COMMAND_BUFFERS" -le 0 ] \
    || [ "$DISPATCHES" -le 0 ]; then
    echo "[vulkan-strict] strict lane executed no counted device work" >&2
    echo "$TELEMETRY" >&2
    exit 1
fi

if ! grep -q \
    '\[GPU STAGE TELEMETRY\] dynamics/cartesian_cgrid_slow_tendencies.comp.spv' \
    "$LOG"; then
    echo "[vulkan-strict] slow-dynamics device tendency path did not execute" >&2
    exit 1
fi
if ! grep -q \
    '\[GPU STAGE TELEMETRY\] lin_pointwise_sedimentation_batch' "$LOG"; then
    echo "[vulkan-strict] Lin device tendency batch did not execute" >&2
    exit 1
fi

grep '\[PERF SUMMARY\]' "$LOG" | tail -1
grep '\[GPU STAGE TELEMETRY\]' "$LOG"
echo "$TELEMETRY"
echo "[vulkan-strict] RESULT: PASS"
