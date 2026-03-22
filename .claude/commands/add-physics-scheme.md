Create a new physics scheme following the existing factory pattern. The user will specify the module and scheme name.

Steps:
1. Identify the target module directory (e.g., `src/microphysics/`, `src/radiation/`, `src/turbulence/`)
2. Read an existing scheme in that module to understand the interface and patterns
3. Create `src/{module}/schemes/{scheme_name}/{scheme_name}.cpp` and `.hpp`
4. Implement the base interface from `include/{module}_base.hpp`
5. Register the new scheme in `src/{module}/factory.cpp`
6. Add config parsing support in `src/core/runtime_config.cpp`
7. Add a basic regression test in `tests/`
8. Update the module README if one exists

Follow existing naming conventions, error handling patterns, and Field3D usage throughout.