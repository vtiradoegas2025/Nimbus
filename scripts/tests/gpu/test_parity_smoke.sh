#!/usr/bin/env bash
# Parity-harness CI smoke (GPU.md Phase 1).
#
# Runs a small Cartesian C-grid fixture under backend: gpu with the parity
# harness enabled. Fails on any [PARITY] tolerance breach (the simulator
# exits 2) and on a vacuous run (harness on, GPU selected, but zero parity
# samples taken). Skips cleanly when the binary is missing or no GPU is
# available, so CPU-only CI stays green. NIMBUS_GPU_REQUIRED=1 converts the
# no-device branch into a failure for promised accelerator lanes.
set -u
cd "$(dirname "$0")/../../.."

BIN=bin/nimbus_sim
LOG=$(mktemp /tmp/nimbus_parity_smoke_XXXXXX.log)
TMP=$(mktemp /tmp/nimbus_parity_smoke_XXXXXX.yaml)
trap 'rm -f "$TMP" "$LOG"' EXIT

if [ ! -x "$BIN" ]; then
    echo "[parity-smoke] $BIN not built - skipping"
    exit 0
fi

# Three fixtures: the calm hold (dynamics-only), the beast physics chain (lin
# microphysics + smagorinsky turbulence), and the full-physics anchor capped to
# a short active window. The first two have near-zero slow forcing at the open
# lateral boundary, so they do NOT exercise the KW78 radiative-face seed timing;
# the full-physics anchor (WK mean wind + f-plane Coriolis) drives strong
# boundary slow tendencies and is what catches an open-boundary acoustic
# parity regression (the w breach at the open corner). Each entry is
# "config:extra-args".
for ENTRY in "tests/fixtures/configs/test/cart_cgrid_calm.yaml:" \
             "tests/fixtures/configs/test/cart_cgrid_beastchain.yaml:" \
             "tests/fixtures/configs/test/cart_cgrid_full_physics.yaml:--duration=60"; do
    CFG="${ENTRY%%:*}"
    EXTRA="${ENTRY#*:}"
    # Flip the backend to GPU and opt the slow + SGS pointwise-apply kernels in
    # (they default OFF on perf grounds - GPU audit Phase 4.2). The opt-in
    # keeps live CPU-vs-GPU parity coverage of those kernels. modularity cleanup 6a
    # moved these from the NIMBUS_GPU_APPLY_* env vars to the config schema, so
    # inject extensions.compute.gpu_apply_* with the backend line's indent.
    awk '
        /^[[:space:]]*backend:[[:space:]]*cpu/ {
            match($0, /^[[:space:]]*/); ind = substr($0, 1, RLENGTH)
            sub(/cpu/, "gpu")
            print
            print ind "gpu_apply_slow: true"
            print ind "gpu_apply_sgs: true"
            next
        }
        { print }
    ' "$CFG" > "$TMP"

    # NIMBUS_VALIDATE_PARITY stays an env var (a diagnostic harness toggle).
    NIMBUS_VALIDATE_PARITY=1 \
        "$BIN" --config "$TMP" --headless $EXTRA > "$LOG" 2>&1
    rc=$?

    if ! grep -q 'selected device' "$LOG"; then
        if [ "${NIMBUS_GPU_REQUIRED:-0}" = "1" ]; then
            echo "[parity-smoke] FAIL ($CFG): required GPU was not selected"
            tail -20 "$LOG"
            exit 1
        fi
        echo "[parity-smoke] no GPU available - skipped"
        exit 0
    fi

    grep '\[PARITY\] RESULT' "$LOG" || true

    if [ $rc -ne 0 ]; then
        echo "[parity-smoke] FAIL ($CFG): simulator exited $rc"
        grep '\[PARITY\] BREACH' "$LOG" | head -3
        tail -5 "$LOG"
        exit 1
    fi

    if ! grep -q '\[PARITY\] RESULT: PASS' "$LOG"; then
        echo "[parity-smoke] FAIL ($CFG): harness enabled but zero parity samples taken (vacuous run)"
        exit 1
    fi
done

echo "[parity-smoke] PASS"
exit 0
