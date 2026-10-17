#!/usr/bin/env bash
# Fail-closed G-08 whole-runtime qualification. Requires a real Vulkan device.
set -euo pipefail

cd "$(dirname "$0")/../../.."

BIN=${NIMBUS_BIN:-bin/nimbus_sim}
CFG=${NIMBUS_GPU_RESIDENCY_CONFIG:-tests/fixtures/configs/qualification/gpu_residency.yaml}
LOG=$(mktemp /tmp/nimbus_gpu_residency_XXXXXX.log)
RESIDENT_MODE=${NIMBUS_GPU_RESIDENT_MODE:-1}
LIN_BATCH_MODE=${NIMBUS_GPU_LIN_BATCH_MODE:-1}
DURATION=${NIMBUS_GPU_RESIDENCY_DURATION:-}
trap 'rm -f "$LOG"' EXIT

if [ ! -x "$BIN" ]; then
    echo "[gpu-residency] missing executable: $BIN" >&2
    exit 1
fi

ARGS=(--config "$CFG" --headless)
if [ -n "$DURATION" ]; then
    ARGS+=(--duration="$DURATION")
fi
NIMBUS_GPU_KERNEL_TIMING=1 NIMBUS_GPU_RESIDENT="$RESIDENT_MODE" \
NIMBUS_GPU_LIN_BATCH="$LIN_BATCH_MODE" \
    ./scripts/run_with_vulkan_env.sh \
    "$BIN" "${ARGS[@]}" >"$LOG" 2>&1

if ! grep -q 'selected device' "$LOG"; then
    echo "[gpu-residency] no Vulkan device selected" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

PERF=$(grep '\[PERF SUMMARY\]' "$LOG" | tail -1 || true)
TELEMETRY=$(grep '\[GPU TELEMETRY\]' "$LOG" | tail -1 || true)
if [ -z "$PERF" ] || [ -z "$TELEMETRY" ]; then
    echo "[gpu-residency] missing performance or GPU telemetry summary" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

metric() {
    printf '%s\n' "$TELEMETRY" | sed -E "s/.* $1=([0-9]+).*/\\1/"
}

COMMAND_BUFFERS=$(metric command_buffers)
DISPATCHES=$(metric dispatches)
BATCHES=$(metric batched_command_buffers)
WAITS=$(metric synchronous_waits)
RESIDENT_HITS=$(metric resident_hits)
DESCRIPTOR_HITS=$(metric descriptor_cache_hits)

if [ "$COMMAND_BUFFERS" -le 0 ] || [ "$DISPATCHES" -le 0 ]; then
    echo "[gpu-residency] no counted GPU work" >&2
    exit 1
fi
if [ "$BATCHES" -le 0 ] || [ "$COMMAND_BUFFERS" -ge "$DISPATCHES" ]; then
    echo "[gpu-residency] dependency-aware batching did not execute" >&2
    exit 1
fi
if [ "$WAITS" -ne "$COMMAND_BUFFERS" ]; then
    echo "[gpu-residency] unexpected synchronization count: waits=$WAITS submissions=$COMMAND_BUFFERS" >&2
    exit 1
fi
if [ "$DESCRIPTOR_HITS" -le 0 ]; then
    echo "[gpu-residency] descriptor cache remained vacuous" >&2
    exit 1
fi
if [ "$RESIDENT_MODE" = "1" ] && [ "$RESIDENT_HITS" -le 0 ]; then
    echo "[gpu-residency] residency remained vacuous" >&2
    exit 1
fi

echo "$PERF"
echo "$TELEMETRY"
echo "[gpu-residency] PASS residency=$RESIDENT_MODE lin_batch=$LIN_BATCH_MODE"
