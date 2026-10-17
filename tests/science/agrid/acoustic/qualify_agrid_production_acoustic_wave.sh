#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim_agrid_qualification"
CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_acoustic_wave.yaml"
CHECKER="$ROOT_DIR/tests/science/agrid/acoustic/check_agrid_production_acoustic_wave.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-wave.XXXXXX")"
cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-wave] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM
for step in 1.2 0.6 0.3; do
    "$SIM_BIN" --headless --config="$CONFIG" --qualification-acoustic-wave \
        --dt-large="$step" --duration=24 --write-every=24 \
        --outdir="$TMP_ROOT/run-$step" --compute-backend=cpu \
        --guard-mode=strict --log-profile=quiet
done
python3 "$CHECKER" "$TMP_ROOT/run-1.2" "$TMP_ROOT/run-0.6" "$TMP_ROOT/run-0.3"
printf '[agrid-wave] PASS: production acoustic phase/amplitude/energy timestep sweep\n'
