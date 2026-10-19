#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim_agrid_qualification"
CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_warm_bubble.yaml"
CHECKER="$ROOT_DIR/tests/science/agrid/physics/check_agrid_warm_bubble.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-bubble.XXXXXX")"

cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-warm-bubble] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM

"$SIM_BIN" --headless --config="$CONFIG" \
    --duration=2 --write-every=2 --outdir="$TMP_ROOT/exports" \
    --compute-backend=cpu --guard-mode=strict --log-profile=quiet
python3 "$CHECKER" "$TMP_ROOT/exports"
printf '[agrid-warm-bubble] PASS: native thermal produced a conservative buoyant response\n'
