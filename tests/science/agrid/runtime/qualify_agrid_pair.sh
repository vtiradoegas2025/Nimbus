#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim"
CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/agrid_production_integrator_calm.yaml"
HOLD_CHECKER="$ROOT_DIR/tests/science/common/check_hydrostatic_hold.py"
PAIR_CHECKER="$ROOT_DIR/tests/science/hydrostatic/check_hydrostatic_pair.py"
PROFILE_DIR="$ROOT_DIR/tests/fixtures/configs/qualification/soundings"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-agrid-pair.XXXXXX")"

cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[agrid-pair] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM

for case_name in dry moist; do
    profile="$PROFILE_DIR/${case_name}_isothermal_profile.txt"
    output="$TMP_ROOT/${case_name}-exports"
    result="$TMP_ROOT/${case_name}-result.json"
    "$SIM_BIN" --headless --config="$CONFIG" --sounding="$profile" \
        --duration=20 --write-every=20 --outdir="$output" \
        --compute-backend=cpu --guard-mode=strict --log-profile=quiet
    python3 "$HOLD_CHECKER" "$output" --minimum-duration-s=20 \
        --output "$result"
done

python3 "$PAIR_CHECKER" "$TMP_ROOT/dry-result.json" \
    "$TMP_ROOT/moist-result.json"
printf '[agrid-pair] PASS: matched strict dry/moist 100-step production-integrator holds\n'
