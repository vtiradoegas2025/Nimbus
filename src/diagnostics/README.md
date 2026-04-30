# Diagnostics

Runtime field validation, conservation tracking, and contract enforcement.

## Why No Factory

Diagnostics are analysis tools, not configurable physics schemes. They observe the simulation state without modifying it. There is no user-facing scheme selection. All active diagnostics run unconditionally when enabled.

## Layout

```
src/diagnostics/
  field_contract.cpp         Contract definitions: valid ranges, required fields, export rules
  field_validation.cpp       Runtime contract checking: non-finite detection, bounds violations
  conservation_budget.cpp    Mass/energy/moisture conservation tracking per timestep

include/diagnostics/
  field_contract.hpp         Contract data structures and field registry
  field_validation.hpp       Validation API (strict mode, report generation)
  conservation_budget.hpp    Budget accumulator interface
```

## Field Contract System

The contract system defines valid ranges and export requirements for every 3D field in the simulation. It tracks:

- **99** total contract fields
- **87** exported with runtime validation
- **20** required-now fields that must be present and valid

In strict mode, any non-finite value or out-of-bounds violation in an exported field triggers an immediate failure. In report mode, violations are logged and summarized.

## Offline Validator

A standalone tool (`src/tools/field_validator.cpp`, built as `bin/field_validator`) can validate exported data after the fact:

```bash
./bin/field_validator \
  --input data/exports \
  --contract cm1 \
  --mode strict \
  --scope exported \
  --json validation_report.json
```

## Conservation Budget

Tracks per-timestep budgets for:
- Total mass (integrated density)
- Total energy (kinetic + potential + internal)
- Total moisture (all water species)

Budget drift is logged at configurable intervals. Target: <0.1% mass drift over 2-hour simulations.

## Validation Targets

```bash
make test-diagnostics    # Contract and validation unit tests
make test                # Full suite including diagnostics
```
