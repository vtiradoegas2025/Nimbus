#!/usr/bin/env python3
"""Keep module lifecycle implementations out of core orchestration."""

from pathlib import Path


MODULE_RUNTIMES = (
    "src/dynamics/runtime.cpp",
    "src/numerics/runtime.cpp",
    "src/radiation/runtime.cpp",
    "src/turbulence/runtime.cpp",
    "src/boundary_layer/runtime.cpp",
    "src/terrain/runtime.cpp",
)

PUBLIC_RUNTIME_HEADERS = (
    "include/boundary_conditions/runtime.hpp",
    "include/boundary_layer/runtime.hpp",
    "include/chaos/runtime.hpp",
    "include/dynamics/runtime.hpp",
    "include/microphysics/runtime.hpp",
    "include/numerics/runtime.hpp",
    "include/radar/runtime.hpp",
    "include/radiation/runtime.hpp",
    "include/soundings/runtime.hpp",
    "include/terrain/runtime.hpp",
    "include/turbulence/runtime.hpp",
)

REQUIRED_LIFECYCLE_SYMBOLS = {
    "src/dynamics/runtime.cpp": ("initialize_dynamics_scheme",),
    "src/numerics/runtime.cpp": (
        "initialize_advection",
        "initialize_diffusion",
        "initialize_time_stepping",
        "choose_runtime_timestep",
    ),
    "src/radiation/runtime.cpp": ("initialize_radiation", "step_radiation"),
    "src/turbulence/runtime.cpp": ("initialize_turbulence", "step_turbulence"),
    "src/boundary_layer/runtime.cpp": (
        "initialize_boundary_layer",
        "step_boundary_layer",
    ),
    "src/terrain/runtime.cpp": ("initialize_terrain", "build_terrain_fields"),
}

RETIRED_ORCHESTRATION_PATHS = (
    "src/core/orchestration/dynamics/numerics.cpp",
    "src/core/orchestration/physics/radiation.cpp",
    "src/core/orchestration/physics/turbulence.cpp",
    "src/core/orchestration/physics/boundary_layer.cpp",
    "src/core/orchestration/physics/terrain.cpp",
)

FORBIDDEN_DEFINITIONS = (
    "std::unique_ptr<tmv::DynamicsScheme>& tmv::dynamics_scheme",
    "global_compute_backend_runtime_state =",
    "global_parity_runtime_state =",
    "global_radiation_runtime_state =",
    "global_turbulence_runtime_state =",
    "global_boundary_layer_runtime_state =",
    "global_chaos_runtime_state =",
    "global_rayleigh_runtime_state =",
    "global_microphysics_runtime_state =",
    "global_terrain_runtime_state =",
    "global_radar_runtime_state =",
    "global_advection_runtime_state =",
    "global_diffusion_runtime_state =",
    "global_time_stepping_runtime_state =",
    "global_dynamics_runtime_state =",
)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    failures: list[str] = []

    for relative in MODULE_RUNTIMES:
        source = root / relative
        if not source.is_file():
            failures.append(f"missing module runtime: {relative}")
            continue
        text = source.read_text(encoding="utf-8")
        for symbol in REQUIRED_LIFECYCLE_SYMBOLS[relative]:
            if symbol not in text:
                failures.append(f"missing lifecycle symbol in {relative}: {symbol}")

    for relative in RETIRED_ORCHESTRATION_PATHS:
        if (root / relative).exists():
            failures.append(f"retired orchestration path returned: {relative}")

    for relative in PUBLIC_RUNTIME_HEADERS:
        if not (root / relative).is_file():
            failures.append(f"missing public runtime contract: {relative}")

    forbidden_umbrella = '#include "core/runtime/simulation.hpp"'
    for source_root in (root / "src", root / "vulkan" / "src"):
        for source in source_root.rglob("*.cpp"):
            if forbidden_umbrella in source.read_text(encoding="utf-8"):
                failures.append(
                    "production source includes compatibility umbrella: "
                    f"{source.relative_to(root)}"
                )

    orchestration = root / "src/core/orchestration"
    for source in orchestration.rglob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        for definition in FORBIDDEN_DEFINITIONS:
            if definition in text:
                failures.append(
                    f"runtime binding leaked into {source.relative_to(root)}: "
                    f"{definition[:-2]}"
                )

    binding_source = root / "src/core/runtime/global_runtime_bindings.cpp"
    if not binding_source.is_file():
        failures.append("missing compatibility adapter: src/core/runtime/global_runtime_bindings.cpp")
    else:
        binding_text = binding_source.read_text(encoding="utf-8")
        for definition in FORBIDDEN_DEFINITIONS[1:]:
            if definition not in binding_text:
                failures.append(
                    "compatibility adapter missing runtime binding: "
                    f"{definition[:-2]}"
                )

    if failures:
        for failure in failures:
            print(f"[module-ownership] FAIL: {failure}")
        return 1

    print(
        "[module-ownership] PASS: 11 public runtime contracts; no production "
        "simulation umbrella; core orchestration contains no module owners or "
        "process-runtime bindings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
