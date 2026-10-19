#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim_agrid_qualification"
BASE_CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_acoustic_wave.yaml"
CHECKER="$ROOT_DIR/tests/science/agrid/boundaries/check_agrid_boundary_pulse.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-pulse.XXXXXX")"

cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-boundary-pulse] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM

sed -e 's/    west: periodic/    west: rigid/' \
    -e 's/    east: periodic/    east: rigid/' \
    "$BASE_CONFIG" > "$TMP_ROOT/rigid.yaml"
sed -e 's/    west: periodic/    west: zero_gradient/' \
    -e 's/    east: periodic/    east: zero_gradient/' \
    "$BASE_CONFIG" > "$TMP_ROOT/zero_gradient.yaml"

for boundary_name in rigid zero_gradient; do
    "$SIM_BIN" --headless --config="$TMP_ROOT/$boundary_name.yaml" \
        --qualification-acoustic-pulse --dt-large=0.3 \
        --duration=100 --write-every=1 \
        --outdir="$TMP_ROOT/$boundary_name-output" \
        --compute-backend=cpu --guard-mode=strict --log-profile=quiet
done

python3 "$CHECKER" "$TMP_ROOT/rigid-output" "$TMP_ROOT/zero_gradient-output"
printf '[agrid-boundary-pulse] PASS: rigid reflection and zero-gradient transmission\n'
