#!/bin/bash
# Physics isolation test: run Cartesian+Kessler with different physics combos
# to verify each component contributes meaningfully.

BIN="./bin/tornado_sim"
DURATION=300
TMPDIR=$(mktemp -d)
BASE="configs/test/base_cartesian.yaml"

run_variant() {
    local label="$1"
    local extra_sed="$2"
    local cfg="$TMPDIR/${label}.yaml"

    cp "$BASE" "$cfg"
    if [ -n "$extra_sed" ]; then
        eval "$extra_sed"
    fi

    echo "=== ${label} ==="

    local output
    output=$(TORNADO_DEBUG_EXPORTS=1 $BIN --headless --config="$cfg" --duration=$DURATION 2>&1)
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "  CRASHED (exit=$exit_code)"
        echo ""
        return
    fi

    # Extract w and theta at key timesteps
    echo "  t(s)   theta_min  theta_max  w_min      w_max"
    echo "$output" | grep -E "Step [0-9]" | while read -r line; do
        local step_t=$(echo "$line" | sed 's/.*t=\([0-9.]*\)s.*/\1/')
        # Read next lines for theta and w
        read -r theta_line
        read -r nan_line
        read -r w_line
        local th_min=$(echo "$theta_line" | sed 's/.*min=\([^ ]*\)K.*/\1/')
        local th_max=$(echo "$theta_line" | sed 's/.*max=\([^ ]*\)K.*/\1/')
        local w_min=$(echo "$w_line" | sed 's/.*min=\([^ ]*\)m.*/\1/')
        local w_max=$(echo "$w_line" | sed 's/.*max=\([^ ]*\)m.*/\1/')
        printf "  %-6s  %-10s %-10s %-10s %s\n" "$step_t" "$th_min" "$th_max" "$w_min" "$w_max"
    done

    # Summary metrics
    local clamp_count=$(echo "$output" | grep -c "clamped")
    local guard_count=$(echo "$output" | grep -c "PHYSICS GUARD")
    local budget_count=$(echo "$output" | grep -c "PHYSICS BUDGET WARN")
    local micro_init=$(echo "$output" | grep "Initialized microphysics" | sed 's/.*scheme: //')
    local turb_init=$(echo "$output" | grep "Initialized turbulence" | sed 's/.*scheme: //')
    local rad_init=$(echo "$output" | grep "Initialized radiation" | sed 's/.*scheme: //')
    local bl_init=$(echo "$output" | grep "Initialized boundary layer" | sed 's/.*scheme: //')
    local adv_init=$(echo "$output" | grep "Initialized.*advection" | head -1 | sed 's/.*scheme.*: //')
    echo ""
    echo "  micro=${micro_init:-none} turb=${turb_init:-none} rad=${rad_init:-none} bl=${bl_init:-none}"
    echo "  clamps=$clamp_count guards=$guard_count budget_warns=$budget_count"
    echo ""
}

echo "=============================================="
echo "  Physics Isolation (Cartesian, Kessler, 300s)"
echo "=============================================="
echo ""

# 1. Full physics (baseline)
run_variant "full_physics" ""

# 2. No turbulence (Cs=0 effectively disables Smagorinsky)
run_variant "no_turbulence" \
    "sed -i.bak '/turbulence/,/Cs/{s/Cs: .*/Cs: 0.0/}' '$cfg' 2>/dev/null || true"

# 3. No radiation
run_variant "no_radiation" \
    "cat >> '$cfg' << 'YAML'
radiation:
  scheme: simple_grey
  lw: false
  sw: false
YAML"

# 4. No boundary layer
run_variant "no_boundary_layer" \
    "cat >> '$cfg' << 'YAML'
boundary_layer:
  scheme: slab
  surface_fluxes: false
YAML"

# 5. No diffusion
run_variant "no_diffusion" \
    "cat >> '$cfg' << 'YAML'
numerics:
  diffusion:
    scheme: explicit
    K_h: 0
    K_v: 0
YAML"
