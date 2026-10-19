#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_BIN="$ROOT_DIR/bin/nimbus_sim"
VIEWER_BIN="$ROOT_DIR/bin/vulkan_viewer"
CONFIG="$ROOT_DIR/tests/fixtures/configs/test/smoke_e2e_presentation.yaml"
NPY_SIGNAL_CHECK="$ROOT_DIR/tests/science/analysis/check_npy_nonzero.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nimbus-smoke-e2e.XXXXXX")"
OUT_DIR="$TMP_ROOT/exports"

cleanup() {
    if [[ -n "${TMP_ROOT:-}" && -d "$TMP_ROOT" ]]; then
        rm -rf -- "$TMP_ROOT"
    fi
}
trap cleanup EXIT
trap 'exit 2' HUP INT TERM

fail() {
    printf '[smoke-test-e2e] FAIL: %s\n' "$*" >&2
    exit 1
}

require_nonempty_file() {
    local path="$1"
    [[ -s "$path" ]] || fail "missing or empty file: $path"
}

publication_generation() {
    local path="$1"
    sed -n 's/.*"publication_generation"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$path" |
        head -n 1 || true
}

[[ -x "$SIM_BIN" ]] || fail "simulator binary is not executable: $SIM_BIN"
[[ -x "$VIEWER_BIN" ]] || fail "viewer binary is not executable: $VIEWER_BIN"
[[ -f "$CONFIG" ]] || fail "smoke configuration is missing: $CONFIG"
[[ -f "$NPY_SIGNAL_CHECK" ]] || fail "NPY signal checker is missing: $NPY_SIGNAL_CHECK"

"$SIM_BIN" \
    --headless \
    --config="$CONFIG" \
    --duration=2 \
    --write-every=1 \
    --outdir="$OUT_DIR" \
    --log-profile=quiet

RUN_MARKER="$OUT_DIR/nimbus_output.json"
require_nonempty_file "$RUN_MARKER"
grep -Eq '"publication_state"[[:space:]]*:[[:space:]]*"ready"' "$RUN_MARKER" ||
    fail "run marker is not ready: $RUN_MARKER"

RUN_GENERATION="$(publication_generation "$RUN_MARKER")"
[[ -n "$RUN_GENERATION" ]] || fail "run marker has no publication generation"

STEP_DIRS=()
while IFS= read -r -d '' step_dir; do
    STEP_DIRS+=("$step_dir")
done < <(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'step_*' -print0)

(( ${#STEP_DIRS[@]} > 0 )) || fail "simulator published no step directories"

for step_dir in "${STEP_DIRS[@]}"; do
    manifest="$step_dir/manifest.json"
    require_nonempty_file "$manifest"
    require_nonempty_file "$step_dir/theta.npy"
    require_nonempty_file "$step_dir/w.npy"

    grep -Eq '"publication_state"[[:space:]]*:[[:space:]]*"complete"' "$manifest" ||
        fail "step manifest is not complete: $manifest"
    published_fields="$(grep -E '"published_fields"[[:space:]]*:' "$manifest" | head -n 1 || true)"
    [[ "$published_fields" == *'"theta"'* && "$published_fields" == *'"w"'* ]] ||
        fail "step manifest does not publish theta and w: $manifest"
    manifest_generation="$(publication_generation "$manifest")"
    [[ "$manifest_generation" == "$RUN_GENERATION" ]] ||
        fail "step generation does not match the current run: $manifest"
done

# The sheared fixture has nonzero horizontal vorticity at initialization. This
# specifically guards the publication order: step_000000 must contain computed
# diagnostics, not the diagnostic arrays' zero-filled allocation state.
require_nonempty_file "$OUT_DIR/step_000000/vorticity_magnitude.npy"
python3 "$NPY_SIGNAL_CHECK" \
    "$OUT_DIR/step_000000/vorticity_magnitude.npy" \
    --threshold 1e-6

LEFTOVER_TMP="$(find "$OUT_DIR" -name '*.tmp' -print -quit)"
[[ -z "$LEFTOVER_TMP" ]] || fail "temporary publication artifact remains: $LEFTOVER_TMP"

if ! VALIDATION_OUTPUT=$("$VIEWER_BIN" \
    --validate-input \
    --input "$OUT_DIR" \
    --fields theta,w 2>&1); then
    printf '%s\n' "$VALIDATION_OUTPUT" >&2
    fail "native reader rejected simulator output"
fi
printf '%s\n' "$VALIDATION_OUTPUT"

STEP_COUNT=${#STEP_DIRS[@]}
grep -Fq "field=theta resolved=theta frames=$STEP_COUNT " <<<"$VALIDATION_OUTPUT" ||
    fail "native reader did not accept every theta frame"
grep -Fq "field=w resolved=w frames=$STEP_COUNT " <<<"$VALIDATION_OUTPUT" ||
    fail "native reader did not accept every w frame"

# Trajectories are a run-scoped product and must work without periodic field
# snapshots. A fresh destination catches the prior missing-outdir failure.
TRAJECTORY_OUT_DIR="$TMP_ROOT/trajectory-only"
"$SIM_BIN" \
    --headless \
    --config="$CONFIG" \
    --duration=1 \
    --write-every=0 \
    --outdir="$TRAJECTORY_OUT_DIR" \
    --trajectories \
    --trajectory-count=1 \
    --compute-backend=cpu \
    --log-profile=quiet
require_nonempty_file "$TRAJECTORY_OUT_DIR/trajectories.npy"
TRAJECTORY_TMP="$(find "$TRAJECTORY_OUT_DIR" -name '*.tmp' -print -quit)"
[[ -z "$TRAJECTORY_TMP" ]] ||
    fail "temporary trajectory publication artifact remains: $TRAJECTORY_TMP"

# Reuse the same directory with a seed beyond the run horizon. The absence of
# new trajectory data must invalidate the prior run-scoped product.
"$SIM_BIN" \
    --headless \
    --config="$CONFIG" \
    --duration=1 \
    --write-every=0 \
    --outdir="$TRAJECTORY_OUT_DIR" \
    --trajectories \
    --trajectory-count=1 \
    --trajectory-seed-time=100 \
    --compute-backend=cpu \
    --log-profile=quiet
[[ ! -e "$TRAJECTORY_OUT_DIR/trajectories.npy" ]] ||
    fail "no-seed repeat retained stale trajectory data"

printf '[smoke-test-e2e] PASS: %d complete frame(s), generation=%s, trajectory-only=ok\n' \
    "$STEP_COUNT" "$RUN_GENERATION"
