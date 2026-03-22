# vulkan

Experimental Vulkan rendering scaffold for Nimbus.

Legacy OpenGL/Python stacks are no longer part of the active Vulkan build path in this workspace.

## Current Scope

- Creates a Vulkan instance
- Optional validation-layer setup (`VK_LAYER_KHRONOS_validation`)
- Enumerates/scores physical devices
- Selects target device (automatic or `--device-index`)
- Creates logical device + graphics queue
- Uses pluggable render backend interface (`clear`, `volume`)
- `volume` backend now runs a Vulkan fullscreen raymarch shader path
- Supports `--dry-run` for CI/smoke testing

## Build

### Dependencies

- Vulkan headers and loader must be installed.
- macOS (Homebrew): `brew install vulkan-headers vulkan-loader molten-vk`
- Windowed Vulkan test path prefers GLFW when available.
- Install GLFW for the most reliable window/surface path:
  - macOS (Homebrew): `brew install glfw`
- Without GLFW, a SFML fallback path is used.
- SPIR-V shader compilation tool is required:
  - macOS (Homebrew): `brew install glslang`

Optional validation layers:
- If `--validation` reports missing layer, install: `brew install vulkan-validationlayers`

### Commands

From repository root:

```bash
make vulkan
```

Or directly:

```bash
make -C vulkan
```

Binary output:

```bash
bin/vulkan_viewer
```

## Run

```bash
# Basic bootstrap
./bin/vulkan_viewer

# Enable validation (if installed)
./bin/vulkan_viewer --validation

# List devices and selection score
./bin/vulkan_viewer --list-devices

# Force a specific GPU
./bin/vulkan_viewer --device-index 0 --dry-run

# CI/smoke mode
./bin/vulkan_viewer --dry-run

# Window + swapchain clear-pass smoke test (auto-exits after 300 frames)
./bin/vulkan_viewer --window-test --window-frames 300

# Explicit backend selection (currently: clear)
./bin/vulkan_viewer --window-test --render-backend clear

# Volume backend (single field)
./bin/vulkan_viewer --window-test --render-backend volume --input data/exports --field theta

# Supercell-style composite from multiple physical fields
./bin/vulkan_viewer --window-test --render-backend volume \
  --input data/exports \
  --fields qr,qg,qi,qc,w,theta \
  --volume-mode supercell \
  --texture-mode natural \
  --camera-mode orbit \
  --camera-orbit-fps 0.02 \
  --camera-distance 2.25 \
  --camera-height 0.85 \
  --camera-fov-deg 55 \
  --style cinematic-bw \
  --playback-fps 2.0 \
  --ray-steps 256 \
  --ray-threshold 0.28 \
  --ray-opacity 1.35 \
  --ray-brightness 1.2 \
  --ray-ambient 0.95 \
  --ray-anisotropy 0.62 \
  --ray-max-distance 5.5 \
  --sun-dir 0.70,0.32,0.64

# Interactive constrained free-fly camera
./bin/vulkan_viewer --window-test --render-backend volume \
  --input data/exports \
  --fields theta,w,qr,qi,vorticity_z \
  --volume-mode supercell \
  --texture-mode natural \
  --camera-mode freefly
```

Notes:
- If validation layers are missing at runtime, the app now falls back to non-validation mode and continues.
- If window test fails on SFML fallback, install GLFW and rebuild with `make vulkan`.
- `--window-test` now runs until you close the window by default (`--window-frames 0`).
- `volume` backend reuses the OpenGL export loader and streams frames into per-field Vulkan 3D textures (`R32_SFLOAT`).
- `--field` renders a single normalized field.
- `--fields` enables multi-field rendering.
- Use `--volume-mode supercell|composite|isolated|cycle` to switch whether fields render together or independently.
- `--texture-mode natural` adds stable world-space micro-detail to reduce synthetic smoothness.
- `--camera-mode orbit` keeps a physically coherent turntable view around the storm core.
- `--camera-mode freefly` enables constrained interactive motion with collision-like bounds:
  `WASD` move, `Q/E` descend/ascend, arrows or right-mouse look, `Shift` boost, `R` reset pose.
- Missing fields are skipped with a warning; known aliases are attempted automatically (e.g., `qi` can resolve to `qh` datasets).
- `--style cinematic-bw` enables a desaturated high-contrast storm palette.
- Playback is time-based through `--playback-fps`.
- Runtime shading controls are available via:
  - `--ray-steps`
  - `--ray-threshold`
  - `--ray-opacity`
  - `--ray-brightness`
  - `--ray-ambient`
  - `--ray-anisotropy`
  - `--ray-max-distance`
  - `--sun-dir`

## Mode Test Scripts

From repository root:

```bash
./vulkan/scripts/test_supercell_mode.sh [input_dir]
./vulkan/scripts/test_composite_mode.sh [input_dir]
./vulkan/scripts/test_isolated_mode.sh [input_dir]
./vulkan/scripts/test_cycle_mode.sh [input_dir]
./vulkan/scripts/test_freefly_controls.sh [input_dir]
```

## Next Incremental Milestones

1. Interactive camera and live controls (keyboard/mouse + UI overlay)
2. Physical transfer-function tuning per field (hail/rain/ice separation)
3. Temporal smoothing/interpolation between frames for smoother playback
4. Backend switch flag (`--backend opengl|vulkan`) in native viewer
