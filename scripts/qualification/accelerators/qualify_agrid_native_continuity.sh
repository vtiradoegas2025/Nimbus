#!/usr/bin/env bash
# Bounded physical proof for the mixed-execution A-grid continuity owner.
set -euo pipefail

cd "$(dirname "$0")/../../.."

BACKEND=${1:-}
BIN=${NIMBUS_BIN:-bin/nimbus_sim}
CFG=${NIMBUS_AGRID_NATIVE_CONFIG:-tests/fixtures/configs/qualification/agrid_sustained_periodic_none.yaml}
DURATION=${NIMBUS_AGRID_NATIVE_DURATION:-1}
LOG=$(mktemp /tmp/nimbus_agrid_native_XXXXXX.log)
trap 'rm -f "$LOG"' EXIT

case "$BACKEND" in
    vulkan|metal) ;;
    *)
        echo "usage: $0 vulkan|metal" >&2
        exit 2
        ;;
esac

if [ ! -x "$BIN" ]; then
    echo "[agrid-native] missing executable: $BIN" >&2
    exit 1
fi

COMMAND=("$BIN" --config "$CFG" --compute-backend "$BACKEND"
         --headless --duration "$DURATION" --write-every=0
         --checkpoint-every=0 --log-profile debug)
if [ "$BACKEND" = vulkan ]; then
    COMMAND=(./scripts/run_with_vulkan_env.sh "${COMMAND[@]}")
fi

"${COMMAND[@]}" >"$LOG" 2>&1

fail_log() {
    echo "[agrid-native] $1" >&2
    tail -40 "$LOG" >&2
    exit 1
}

grep -Fq "[COMPUTE] backend=$BACKEND, requested=$BACKEND" "$LOG" ||
    fail_log "requested backend was not active"
if grep -Fq 'Falling back to CPU backend' "$LOG"; then
    fail_log "accelerator initialization fell back to CPU"
fi
grep -Fq 'guard_mode=strict' "$LOG" ||
    fail_log "strict validation was not active"
if grep -Eq '\[PHYSICS GUARD FIELD\]|non-finite Cartesian.*mass flux' "$LOG"; then
    fail_log "strict execution observed invalid science state"
fi

TELEMETRY=$(grep -F '[COMPUTE DISPATCH]' "$LOG" | tail -1 || true)
if [ -z "$TELEMETRY" ]; then
    fail_log "missing final compute-dispatch telemetry"
fi
STAGE='acoustic/acoustic_continuity_massflux_cartesian_agrid.comp'
if [[ "$TELEMETRY" != *"backend=$BACKEND"* ]] ||
   [[ "$TELEMETRY" != *"$STAGE:"* ]]; then
    fail_log "native A-grid continuity stage was not attributed"
fi

SUBMISSIONS=$(printf '%s\n' "$TELEMETRY" |
    sed -E 's/.*completed_submissions=([0-9]+).*/\1/')
DISPATCHES=$(printf '%s\n' "$TELEMETRY" |
    sed -E 's/.*kernel_dispatches=([0-9]+).*/\1/')
STAGE_DISPATCHES=$(printf '%s\n' "$TELEMETRY" |
    sed -E "s/.*${STAGE//\//\\/}:([0-9]+).*/\\1/")
if [ "$SUBMISSIONS" -le 0 ] || [ "$DISPATCHES" -le 0 ] ||
   [ "$STAGE_DISPATCHES" -le 0 ] || [ "$STAGE_DISPATCHES" -ne "$DISPATCHES" ]; then
    fail_log "native dispatch counts were vacuous or internally inconsistent"
fi

grep -F '[COMPUTE] backend=' "$LOG" | tail -1
echo "$TELEMETRY"
echo "[agrid-native] backend=$BACKEND strict=yes submissions=$SUBMISSIONS dispatches=$DISPATCHES RESULT: PASS"
