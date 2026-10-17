#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim"
CONFIG="$ROOT_DIR/tests/fixtures/configs/qualification/hydrostatic_cgrid_calm_1h.yaml"
CHECKER="$ROOT_DIR/tests/science/common/check_hydrostatic_hold.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-hydrostatic-qualification.XXXXXX")"
OUT_DIR="$TMP_ROOT/exports"

cleanup() {
    if [[ "${NIMBUS_QUALIFICATION_KEEP:-0}" == "1" ]]; then
        printf '[hydrostatic-qualification] retained artifacts: %s\n' "$TMP_ROOT"
    elif [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM

fail() {
    printf '[hydrostatic-qualification] FAIL: %s\n' "$*" >&2
    exit 1
}

[[ -x "$SIM_BIN" ]] || fail "simulator binary is not executable: $SIM_BIN"
[[ -f "$CONFIG" ]] || fail "qualification configuration is missing: $CONFIG"
[[ -f "$CHECKER" ]] || fail "qualification checker is missing: $CHECKER"

"$SIM_BIN" \
    --headless \
    --config="$CONFIG" \
    --duration=3600 \
    --write-every=3600 \
    --outdir="$OUT_DIR" \
    --compute-backend=cpu \
    --guard-mode=strict \
    --log-profile=quiet

python3 "$CHECKER" "$OUT_DIR" --minimum-duration-s=3600
printf '[hydrostatic-qualification] PASS: strict 1-hour Cartesian C-grid hold\n'
