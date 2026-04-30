# Vulkan Viewer

Native GPU-accelerated 3D visualization for Nimbus simulations. Volume ray marching through atmospheric fields with real-time interactive camera controls.

## Current Scope

- Vulkan instance with optional validation layers (`VK_LAYER_KHRONOS_validation`)
- Automatic GPU selection with scoring (or manual `--device-index`)
- Pluggable render backends (`clear`, `volume`)
- Volume ray marching with per-field 3D textures (`R32_SFLOAT`)
- Live shared-memory ingest from running simulations
- NPY field ingestion from simulation exports
- `--dry-run` mode for CI/smoke testing

## Build

### Dependencies

| Dependency | Required | Install |
|-----------|----------|---------|
| Vulkan headers + loader | Yes | macOS: `brew install vulkan-headers vulkan-loader molten-vk` | 
| | | Ubuntu/Debian: `sudo apt install libvulkan-dev` |
| | | Fedora: `sudo dnf install vulkan-headers vulkan-loader-devel` |
| | | Arch: `sudo pacman -S vulkan-headers vulkan-icd-loader` |
| glslangValidator | Yes | macOS: `brew install glslang` |
| | | Ubuntu/Debian: `sudo apt install glslang-tools` |
| | | Fedora: `sudo dnf install glslang` |
| | | Arch: `sudo pacman -S glslang` |
| GLFW | Recommended | macOS: `brew install glfw` |
| | | Ubuntu/Debian: `sudo apt install libglfw3-dev` |
| | | Fedora: `sudo dnf install glfw-devel` |
| | | Arch: `sudo pacman -S glfw-x11` |

Without GLFW, a SFML fallback path is used.

Optional validation layers (for debugging):
- macOS: `brew install vulkan-validationlayers`
- Ubuntu/Debian: `sudo apt install vulkan-validationlayers`

**LunarG Vulkan SDK:** If using the LunarG SDK instead of system packages, set `VULKAN_SDK` before building. See [BUILDING.md](../BUILDING.md) for details.

**Automated setup:** Run `./scripts/setup.sh` from the repository root to install all dependencies for your platform automatically.

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

### Notes

- If validation layers are missing at runtime, the viewer falls back to non-validation mode automatically.
- If the SFML window path fails, install GLFW and rebuild with `make vulkan`.
- `--window-test` runs until you close the window by default (`--window-frames 0`).
- `--field` renders a single normalized field; `--fields` enables multi-field rendering.
- `--volume-mode supercell|composite|isolated|cycle` controls whether fields render together or independently.
- `--texture-mode natural` adds world-space micro-detail to reduce synthetic smoothness.
- `--camera-mode orbit` provides a turntable view around the storm core.
- `--camera-mode freefly` enables constrained interactive motion:
  `WASD` move, `Q/E` descend/ascend, arrows or right-mouse look, `Shift` boost, `R` reset pose.
- Missing fields are skipped with a warning; known aliases are attempted automatically (e.g., `qi` resolves to `qh`).
- `--style cinematic-bw` enables a desaturated high-contrast storm palette.

### Ray Marching Controls

| Flag | Purpose |
|------|---------|
| `--ray-steps` | Number of ray march steps |
| `--ray-threshold` | Density threshold for visibility |
| `--ray-opacity` | Global opacity multiplier |
| `--ray-brightness` | Brightness multiplier |
| `--ray-ambient` | Ambient light level |
| `--ray-anisotropy` | Phase function anisotropy |
| `--ray-max-distance` | Maximum ray travel distance |
| `--sun-dir` | Sun direction vector (x,y,z) |
| `--playback-fps` | Time-based frame playback rate |

## Mode Test Scripts

From repository root:

```bash
./vulkan/scripts/test_supercell_mode.sh [input_dir]
./vulkan/scripts/test_composite_mode.sh [input_dir]
./vulkan/scripts/test_isolated_mode.sh [input_dir]
./vulkan/scripts/test_cycle_mode.sh [input_dir]
./vulkan/scripts/test_freefly_controls.sh [input_dir]
```

## GPU Compute

The Vulkan backend also provides GPU-accelerated compute for the simulation engine (advection, acoustic substeps, microphysics, diffusion). This is separate from the viewer and configured via the simulation YAML:

```yaml
numerics:
  compute:
    backend: vulkan    # or "cpu"
```

Compute shaders live in `shaders/compute/` and are pre-compiled as SPIR-V (`.spv`) in the repository. Running `make` from the project root automatically recompiles any shaders whose source has changed (requires `glslangValidator`).
