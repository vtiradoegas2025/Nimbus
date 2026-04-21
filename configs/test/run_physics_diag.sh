#!/bin/bash
# Physics diagnostic runner
# Runs each dynamics+microphysics combo for 300s with field diagnostics enabled.
# Checks for physically meaningful convective behavior:
#   - Bubble rises: w_max should increase from ~0 to >1 m/s
#   - Condensation occurs: qc/qr fields become non-zero (from budget warns)
#   - Theta perturbation evolves: theta range should widen
#   - No blow-up: fields stay bounded
#   - Updraft develops: w_max should reach meaningful values

BIN="./bin/tornado_sim"
DURATION=300
TMPDIR=$(mktemp -d)

MICRO_SCHEMES="kessler lin thompson milbrandt"
CYL_DYN="supercell tornado"

run_diag() {
    local base_cfg="$1"
    local dyn="$2"
    local micro="$3"
    local label="${dyn}+${micro}"
    local cfg="$TMPDIR/${label}.yaml"

    # Generate config with correct scheme in correct section
    python3 -c "
lines = open('${base_cfg}').readlines()
in_dynamics = False
in_micro = False
out = []
for line in lines:
    stripped = line.strip()
    if stripped == 'dynamics:':
        in_dynamics = True; in_micro = False
    elif stripped == 'microphysics:':
        in_micro = True; in_dynamics = False
    elif stripped and not stripped.startswith('#') and ':' in stripped and not stripped.startswith('scheme:') and not stripped.startswith('  '):
        in_dynamics = False; in_micro = False
    if in_dynamics and stripped.startswith('scheme:'):
        out.append('  scheme: ${dyn}\n')
    elif in_micro and stripped.startswith('scheme:'):
        out.append('  scheme: ${micro}\n')
    else:
        out.append(line)
open('${cfg}', 'w').writelines(out)
"

    echo "--- ${label} ---"

    local output
    output=$(TORNADO_DEBUG_EXPORTS=1 $BIN --headless --config="$cfg" --duration=$DURATION 2>&1)
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "  CRASHED (exit=$exit_code)"
        echo ""
        return
    fi

    # Extract time-series of w (vertical velocity) from debug output
    local w_data=$(echo "$output" | grep "Vertical velocity" | \
        sed 's/.*min=\([-0-9.e+]*\).*max=\([-0-9.e+]*\).*/\1 \2/')

    # Extract time-series of theta
    local theta_data=$(echo "$output" | grep "Theta sample:" | grep "Step" -A1 | grep "Theta" | \
        sed 's/.*min=\([-0-9.e+]*\)K.*max=\([-0-9.e+]*\)K.*/\1 \2/')

    # Extract microphysics clamp events (indicates condensation activity)
    local clamp_count=$(echo "$output" | grep -c "clamped")
    local limited_count=$(echo "$output" | grep -c "limited")
    local nonfinite_count=$(echo "$output" | grep -c "non-finite")

    # Parse w evolution
    python3 -c "
import sys
lines = '''${w_data}'''.strip().split('\n')
if not lines or lines[0] == '':
    print('  w: no diagnostic data')
    sys.exit(0)
w_mins = []
w_maxs = []
for line in lines:
    parts = line.split()
    if len(parts) == 2:
        w_mins.append(float(parts[0]))
        w_maxs.append(float(parts[1]))
if not w_maxs:
    print('  w: no data parsed')
    sys.exit(0)
n = len(w_maxs)
# Early, mid, late snapshots
early = w_maxs[min(1, n-1)]
mid = w_maxs[n//2]
late = w_maxs[-1]
w_min_all = min(w_mins)
w_max_all = max(w_maxs)

print(f'  w_max evolution: early={early:.3f} mid={mid:.3f} late={late:.3f} m/s')
print(f'  w range: [{w_min_all:.3f}, {w_max_all:.3f}] m/s')

# Physical checks
checks = []
if w_max_all > 0.5:
    checks.append('updraft develops')
else:
    checks.append('NO UPDRAFT (w_max < 0.5)')
if late > early:
    checks.append('convection growing')
elif late < early * 0.5 and early > 0.1:
    checks.append('convection DECAYING')
else:
    checks.append('convection steady/slow')
if abs(w_min_all) > 100 or w_max_all > 100:
    checks.append('BLOW-UP RISK (|w| > 100)')
print(f'  physics: {\" | \".join(checks)}')
"

    # Parse theta evolution
    python3 -c "
import sys
lines = '''${theta_data}'''.strip().split('\n')
if not lines or lines[0] == '':
    print('  theta: no diagnostic data')
    sys.exit(0)
t_mins = []
t_maxs = []
for line in lines:
    parts = line.split()
    if len(parts) == 2:
        t_mins.append(float(parts[0]))
        t_maxs.append(float(parts[1]))
if not t_maxs:
    print('  theta: no data parsed')
    sys.exit(0)
n = len(t_maxs)
early_range = t_maxs[min(1, n-1)] - t_mins[min(1, n-1)]
late_range = t_maxs[-1] - t_mins[-1]
t_min_all = min(t_mins)
t_max_all = max(t_maxs)
print(f'  theta range: [{t_min_all:.1f}, {t_max_all:.1f}] K')
print(f'  theta spread: early={early_range:.1f}K late={late_range:.1f}K')
if late_range > early_range * 1.1:
    print(f'  theta: perturbation GROWING (bubble active)')
elif late_range < early_range * 0.9:
    print(f'  theta: perturbation shrinking')
else:
    print(f'  theta: perturbation stable')
if t_min_all < 150 or t_max_all > 500:
    print(f'  theta: OUT OF PHYSICAL BOUNDS')
"

    echo "  microphysics activity: clamped=$clamp_count limited=$limited_count nonfinite=$nonfinite_count"
    echo ""
}

echo "=============================================="
echo "  Physics Diagnostic (${DURATION}s, student grid)"
echo "=============================================="
echo ""

echo "=== Cylindrical dynamics ==="
echo ""
for dyn in $CYL_DYN; do
    for micro in $MICRO_SCHEMES; do
        run_diag configs/test/base_cylindrical.yaml "$dyn" "$micro"
    done
done

echo "=== Cartesian dynamics ==="
echo ""
for micro in $MICRO_SCHEMES; do
    run_diag configs/test/base_cartesian.yaml "cartesian" "$micro"
done
