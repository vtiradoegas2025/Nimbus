#!/bin/bash
# Scheme combination matrix test
# Runs all dynamics x microphysics combinations on student grid
# Reports: pass/fail, NaN/Inf counts, field ranges, budget warnings

BIN="./bin/tornado_sim"
DURATION=30
TMPDIR=$(mktemp -d)
RESULTS="$TMPDIR/results.txt"

MICRO_SCHEMES="kessler lin thompson milbrandt"

# Cylindrical dynamics: supercell, tornado
CYL_DYN="supercell tornado"
# Cartesian dynamics
CART_DYN="cartesian"

run_test() {
    local base_cfg="$1"
    local dyn="$2"
    local micro="$3"
    local label="${dyn}+${micro}"

    local cfg="$TMPDIR/${label}.yaml"
    sed -e "s/scheme: supercell/scheme: ${dyn}/" \
        -e "s/scheme: cartesian/scheme: ${dyn}/" \
        -e "s/scheme: kessler/scheme: ${micro}/" \
        "$base_cfg" > "$cfg"

    # Handle dynamics line appearing before microphysics
    # The sed above is greedy -- fix by using section context
    python3 -c "
import sys
lines = open('${cfg}').readlines()
in_dynamics = False
in_micro = False
out = []
for line in lines:
    stripped = line.strip()
    if stripped == 'dynamics:':
        in_dynamics = True
        in_micro = False
    elif stripped == 'microphysics:':
        in_micro = True
        in_dynamics = False
    elif stripped and not stripped.startswith('#') and ':' in stripped and not stripped.startswith('scheme:'):
        if not stripped.startswith('  '):
            in_dynamics = False
            in_micro = False
    if in_dynamics and stripped.startswith('scheme:'):
        out.append('  scheme: ${dyn}\n')
    elif in_micro and stripped.startswith('scheme:'):
        out.append('  scheme: ${micro}\n')
    else:
        out.append(line)
open('${cfg}', 'w').writelines(out)
"

    echo -n "  ${label} ... "

    local output
    output=$($BIN --headless --config="$cfg" --duration=$DURATION 2>&1)
    local exit_code=$?

    # Extract key metrics
    local nan_count=$(echo "$output" | grep -o "NaN count: [0-9]*" | head -1 | grep -o "[0-9]*$")
    local inf_count=$(echo "$output" | grep -o "Inf count: [0-9]*" | head -1 | grep -o "[0-9]*$")
    local theta_range=$(echo "$output" | grep "Theta:" | head -1 | sed 's/.*Theta: //')
    local pressure_range=$(echo "$output" | grep "Pressure:" | head -1 | sed 's/.*Pressure: //')
    local sanitize_count=$(echo "$output" | grep -c "\[SANITIZE\]")
    local error_count=$(echo "$output" | grep -c "\[ERROR\]")
    local micro_init=$(echo "$output" | grep "Initialized microphysics scheme:" | head -1 | sed 's/.*scheme: //')
    local budget_warns=$(echo "$output" | grep -c "PHYSICS BUDGET WARN")

    # Check for field blowup or NaN in final state
    local field_nan=$(echo "$output" | grep -c "non-finite")
    local field_limited=$(echo "$output" | grep -c "theta limited")
    local field_clamped=$(echo "$output" | grep -c "clamped")

    local status="PASS"
    if [ $exit_code -ne 0 ]; then
        status="CRASH(exit=$exit_code)"
    elif [ "${nan_count:-0}" -gt 0 ] || [ "${inf_count:-0}" -gt 0 ]; then
        status="NaN/Inf"
    elif [ $error_count -gt 0 ]; then
        status="ERROR"
    fi

    printf "%-8s  micro=%s  nan=%s inf=%s sanitize=%d errors=%d budget_warns=%d\n" \
        "$status" "${micro_init:-?}" "${nan_count:-0}" "${inf_count:-0}" \
        "$sanitize_count" "$error_count" "$budget_warns"

    # Log details to results file
    echo "=== ${label} ===" >> "$RESULTS"
    echo "  status: $status" >> "$RESULTS"
    echo "  exit_code: $exit_code" >> "$RESULTS"
    echo "  micro_init: ${micro_init:-?}" >> "$RESULTS"
    echo "  nan_count: ${nan_count:-0}" >> "$RESULTS"
    echo "  inf_count: ${inf_count:-0}" >> "$RESULTS"
    echo "  theta: ${theta_range}" >> "$RESULTS"
    echo "  pressure: ${pressure_range}" >> "$RESULTS"
    echo "  sanitize_events: $sanitize_count" >> "$RESULTS"
    echo "  errors: $error_count" >> "$RESULTS"
    echo "  budget_warns: $budget_warns" >> "$RESULTS"
    echo "  field_nan: $field_nan" >> "$RESULTS"
    echo "  field_limited: $field_limited" >> "$RESULTS"
    echo "  field_clamped: $field_clamped" >> "$RESULTS"
    echo "" >> "$RESULTS"
}

echo "=============================================="
echo "  Scheme Matrix Test (${DURATION}s, student grid)"
echo "=============================================="
echo ""

echo "Cylindrical dynamics:"
for dyn in $CYL_DYN; do
    for micro in $MICRO_SCHEMES; do
        run_test configs/test/base_cylindrical.yaml "$dyn" "$micro"
    done
done

echo ""
echo "Cartesian dynamics:"
for micro in $MICRO_SCHEMES; do
    run_test configs/test/base_cartesian.yaml "cartesian" "$micro"
done

echo ""
echo "Detailed results: $RESULTS"
echo ""
cat "$RESULTS"
