#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim_agrid_qualification"
CHECKER="$ROOT_DIR/tests/science/agrid/acoustic/check_agrid_production_acoustic_spatial.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-wave-spatial.XXXXXX")"
cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-wave-spatial] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM
for resolution in coarse medium fine; do
    config="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_acoustic_wave.yaml"
    if [[ "$resolution" == "coarse" ]]; then
        config="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_acoustic_wave_coarse.yaml"
    elif [[ "$resolution" == "fine" ]]; then
        config="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_acoustic_wave_fine.yaml"
    fi
    "$SIM_BIN" --headless --config="$config" --qualification-acoustic-wave \
        --dt-large=0.3 --duration=24 --write-every=24 \
        --outdir="$TMP_ROOT/$resolution" --compute-backend=cpu \
        --guard-mode=strict --log-profile=quiet
done
python3 "$CHECKER" "$TMP_ROOT/coarse" "$TMP_ROOT/medium" "$TMP_ROOT/fine"
printf '[agrid-wave-spatial] PASS: production acoustic 16/32/64-cell spatial refinement\n'
