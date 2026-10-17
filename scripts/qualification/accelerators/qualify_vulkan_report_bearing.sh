#!/usr/bin/env bash
# Fail-closed physical Vulkan qualification with per-step science reports.
set -euo pipefail

cd "$(dirname "$0")/../../.."

BIN=${NIMBUS_BIN:-bin/nimbus_sim}
CFG=${NIMBUS_VULKAN_REPORT_CONFIG:-tests/fixtures/configs/qualification/gpu_report_bearing.yaml}
DURATION=${NIMBUS_VULKAN_REPORT_DURATION:-12}
LOG=$(mktemp /tmp/nimbus_vulkan_report_XXXXXX.log)
trap 'rm -f "$LOG"' EXIT

if [ ! -x "$BIN" ]; then
    echo "[vulkan-report] missing executable: $BIN" >&2
    exit 1
fi

NIMBUS_GPU_KERNEL_TIMING=1 \
NIMBUS_GPU_RESIDENT=1 \
NIMBUS_GPU_LIN_BATCH=1 \
    ./scripts/run_with_vulkan_env.sh \
    "$BIN" \
    --config "$CFG" \
    --compute-backend vulkan \
    --guard-mode off \
    --headless \
    --duration="$DURATION" \
    --no-timing >"$LOG" 2>&1

fail_log() {
    echo "[vulkan-report] $1" >&2
    tail -40 "$LOG" >&2
    exit 1
}

grep -q 'selected device' "$LOG" ||
    fail_log "no physical Vulkan device was selected"
if grep -q 'Falling back to CPU backend' "$LOG"; then
    fail_log "Vulkan initialization fell back to CPU"
fi
grep -q 'Scientific diagnostics: every 1 steps' "$LOG" ||
    fail_log "the report-bearing cadence was not active on every step"
if grep -Eq \
    'invalid Cartesian C-grid transport context|non-finite|\\[PHYSICS GUARD FIELD\\]' \
    "$LOG"; then
    fail_log "the report-bearing run observed invalid science state"
fi

TELEMETRY=$(grep '\[GPU TELEMETRY\]' "$LOG" | tail -1 || true)
if [ -z "$TELEMETRY" ]; then
    fail_log "missing final GPU telemetry"
fi

metric() {
    printf '%s\n' "$TELEMETRY" | sed -E "s/.* $1=([0-9]+).*/\\1/"
}

MODEL_STEPS=$(metric model_steps)
COMMAND_BUFFERS=$(metric command_buffers)
DISPATCHES=$(metric dispatches)
RESIDENT_HITS=$(metric resident_hits)
if [ "$MODEL_STEPS" -le 0 ] || [ "$COMMAND_BUFFERS" -le 0 ] \
    || [ "$DISPATCHES" -le 0 ] || [ "$RESIDENT_HITS" -le 0 ]; then
    fail_log "report-bearing device execution was vacuous"
fi

REQUIRED_STAGES=(
    'dynamics/cartesian_cgrid_slow_tendencies.comp.spv'
    'acoustic_fused_cartesian_cgrid'
    'lin_pointwise_sedimentation_batch'
    'smagorinsky_fused'
    'dynamics/slow_apply.comp.spv'
    'transport_batch_cartesian_cgrid'
)
for stage in "${REQUIRED_STAGES[@]}"; do
    if ! grep -Fq "[GPU STAGE TELEMETRY] $stage " "$LOG"; then
        fail_log "required Vulkan stage did not execute: $stage"
    fi
done

grep '\[GPU STAGE TELEMETRY\]' "$LOG"
echo "$TELEMETRY"
echo "[vulkan-report] RESULT: PASS"
