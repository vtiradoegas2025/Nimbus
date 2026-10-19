#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim_agrid_qualification"
CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_production_integrator_calm.yaml"
CHECKER="$ROOT_DIR/tests/science/agrid/acoustic/check_agrid_acoustic_sweep.py"
PROFILE="$ROOT_DIR/tests/fixtures/configs/qualification/soundings/moist_isothermal_profile.txt"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-acoustic.XXXXXX")"

cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-acoustic] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM

for substeps in 3 6 12; do
    output="$TMP_ROOT/run-$substeps"
    "$SIM_BIN" --headless --config="$CONFIG" --sounding="$PROFILE" \
        --duration=20 --write-every=20 --outdir="$output" \
        --acoustic-substeps="$substeps" --compute-backend=cpu \
        --guard-mode=strict --log-profile=quiet
done

python3 "$CHECKER" "$TMP_ROOT/run-3" "$TMP_ROOT/run-6" "$TMP_ROOT/run-12"
printf '[agrid-acoustic] PASS: bounded 3/6/12-substep production-integrator sensitivity\n'
