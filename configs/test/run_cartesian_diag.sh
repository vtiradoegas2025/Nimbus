#!/bin/bash
# Cartesian-only physics diagnostic
# Detailed field evolution for each microphysics scheme

BIN="./bin/tornado_sim"
DURATION=300
TMPDIR=$(mktemp -d)

MICRO_SCHEMES="kessler lin thompson milbrandt"

run_diag() {
    local micro="$1"
    local label="cartesian+${micro}"
    local cfg="$TMPDIR/${label}.yaml"

    python3 -c "
lines = open('configs/test/base_cartesian.yaml').readlines()
in_micro = False
out = []
for line in lines:
    stripped = line.strip()
    if stripped == 'microphysics:':
        in_micro = True
    elif stripped and not stripped.startswith('#') and ':' in stripped and not stripped.startswith('scheme:') and not stripped.startswith('  '):
        in_micro = False
    if in_micro and stripped.startswith('scheme:'):
        out.append('  scheme: ${micro}\n')
    else:
        out.append(line)
open('${cfg}', 'w').writelines(out)
"

    echo "=========================================="
    echo "  cartesian + ${micro} (${DURATION}s)"
    echo "=========================================="

    local output
    output=$(TORNADO_DEBUG_EXPORTS=1 $BIN --headless --config="$cfg" --duration=$DURATION 2>&1)
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "  CRASHED (exit=$exit_code)"
        echo "$output" | tail -20
        echo ""
        return
    fi

    # Extract all per-step debug lines
    echo "$output" | grep -E "(TIME STEP DEBUG|Theta sample|Wind.*sample|Vertical velocity|NaN)" | head -60

    echo ""
    echo "  --- Summary ---"

    # Clamp/limit/nonfinite counts
    local clamp_count=$(echo "$output" | grep -c "clamped")
    local limited_count=$(echo "$output" | grep -c "limited")
    local nonfinite_count=$(echo "$output" | grep -c "non-finite")
    local guard_count=$(echo "$output" | grep -c "PHYSICS GUARD")
    local budget_count=$(echo "$output" | grep -c "PHYSICS BUDGET WARN")

    echo "  clamped=$clamp_count limited=$limited_count nonfinite=$nonfinite_count guards=$guard_count budget_warns=$budget_count"

    # Extract final state from the last INIT SUMMARY or step debug
    local init_theta=$(echo "$output" | grep "Theta:" | grep "INIT" | head -1)
    echo "  init: $init_theta"

    echo ""
}

for micro in $MICRO_SCHEMES; do
    run_diag "$micro"
done
