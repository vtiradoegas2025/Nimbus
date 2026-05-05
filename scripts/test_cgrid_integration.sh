#!/usr/bin/env bash
#
# Phase C.10 integration tests for the cylindrical Arakawa C-grid runtime
# path. Drives bin/tornado_sim against several c_grid YAML configs and
# checks that each completes without a crash, abort, or fatal validation
# event.
#
# This is the FIRST PASS scaffolding: it exercises the load_config ->
# orchestrator -> dynamics step -> output path end-to-end with c_grid
# enabled and confirms the runtime is wired up. It does NOT yet enforce
# the physical pass criteria from docs/CoordinateBackend_Plan.md C.10
# (mass-drift threshold, hydrostatic-velocity threshold, etc.); those
# require capturing the conservation budget after the run, which the
# harness does not yet parse. A second pass adds that.
#
# Exit code 0 if all scenarios run to completion. Non-zero if any
# scenario aborts, segfaults, or returns non-zero exit status.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO_ROOT/bin/tornado_sim"
LOG_DIR="${TMPDIR:-/tmp}/cgrid_integration_logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "[CGRID-INT] error: $BIN not found or not executable. Run 'make' first." >&2
    exit 2
fi

SCENARIOS=(
    "cgrid_hydrostatic       configs/test/cgrid_hydrostatic.yaml"
    "cgrid_uniform_wind      configs/test/cgrid_uniform_wind.yaml"
    "cgrid_mass_conservation configs/test/cgrid_mass_conservation.yaml"
)

# Patterns whose presence in the simulation log indicates a real failure
# (not just a warning). The PHYSICS BUDGET WARN messages are explicitly
# NOT in this list -- they are expected with a trigger bubble in the
# first 30 s and will be tightened in a second pass.
FAIL_PATTERNS=(
    "Aborted"
    "Segmentation fault"
    "Bus error"
    "FATAL"
    "what\\(\\):"             # uncaught std::exception
    "terminate called"
    "AddressSanitizer"
)

declare -i fail_count=0
declare -i pass_count=0
total=${#SCENARIOS[@]}

echo "[CGRID-INT] running $total c_grid scenarios via $BIN"
echo "[CGRID-INT] logs in $LOG_DIR"
echo

for entry in "${SCENARIOS[@]}"; do
    name="${entry%% *}"
    yaml="${entry##* }"
    log_path="$LOG_DIR/${name}.log"

    if [[ ! -f "$REPO_ROOT/$yaml" ]]; then
        echo "[CGRID-INT] FAIL $name (config $yaml missing)"
        fail_count+=1
        continue
    fi

    printf "[CGRID-INT] %-30s ... " "$name"

    # Run the simulation. Capture both stdout and stderr to the log file.
    # The YAML's duration_s controls model time; --headless ensures the
    # time loop actually runs (without it the binary returns 0 after init).
    "$BIN" --config "$yaml" --headless >"$log_path" 2>&1
    exit_code=$?

    fail_reason=""
    if [[ $exit_code -ne 0 ]]; then
        fail_reason="exit=$exit_code"
    else
        for pat in "${FAIL_PATTERNS[@]}"; do
            if grep -E -q "$pat" "$log_path"; then
                fail_reason="matched pattern '$pat'"
                break
            fi
        done
    fi

    if [[ -z "$fail_reason" ]]; then
        echo "PASS"
        pass_count+=1
    else
        echo "FAIL ($fail_reason)"
        echo "    log: $log_path"
        echo "    last 5 lines:"
        tail -5 "$log_path" | sed 's/^/      /'
        fail_count+=1
    fi
done

echo
if [[ $fail_count -eq 0 ]]; then
    echo "[CGRID-INT] all $pass_count/$total scenarios passed"
    exit 0
fi

echo "[CGRID-INT] $fail_count/$total scenarios failed"
exit 1
