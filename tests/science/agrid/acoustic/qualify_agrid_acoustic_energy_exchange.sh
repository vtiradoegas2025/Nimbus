#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim_agrid_qualification"
CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_acoustic_wave.yaml"
CHECKER="$ROOT_DIR/tests/science/agrid/acoustic/check_agrid_acoustic_energy_exchange.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-standing.XXXXXX")"
cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-standing] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM
"$SIM_BIN" --headless --config="$CONFIG" \
    --qualification-acoustic-standing-wave --dt-large=0.3 \
    --duration=24 --write-every=24 --outdir="$TMP_ROOT/exports" \
    --compute-backend=cpu --guard-mode=strict --log-profile=quiet
python3 "$CHECKER" "$TMP_ROOT/exports"
printf '[agrid-standing] PASS: compression-to-kinetic acoustic energy exchange\n'
