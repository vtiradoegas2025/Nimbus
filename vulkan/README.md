# Vulkan compute and standalone viewer

This directory contains the solver compute backend and a separate native
viewer. The viewer draws saved/live fields through Vulkan graphics shaders;
the compute backend advances supported atmospheric operators. Neither selects
Studio's WebGL/WebGPU rendering backend. See the GPU and field-to-pixel chapters
in the [architecture guide](../docs/ARCHITECTURE.pdf).

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

**LunarG Vulkan SDK:** If using the LunarG SDK instead of system packages, set `VULKAN_SDK` before building. See [BUILDING.md](../docs/BUILDING.md) for details.

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

# Fail-closed resize qualification (forces and verifies swapchain recreation)
./bin/vulkan_viewer --validation --window-test --window-frames 180 \
  --window-resize-frame 60 --window-resize-width 960 --window-resize-height 540

# Explicit clear-pass backend
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
  --vertical-exaggeration 1.0 \
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

- `--validation` is fail-closed: an unavailable layer or any validation error
  produces a nonzero exit after orderly renderer/context shutdown.
- If the SFML window path fails, install GLFW and rebuild with `make vulkan`.
- `--window-test` runs until you close the window by default (`--window-frames 0`).
- `--window-seconds` adds a graceful wall-clock deadline; unlike a frame count,
  it remains stable across different display refresh rates.
- `--capture-limit` caps periodic `--capture-interval` output while leaving
  manual F12 capture available.
- `--field` renders a single normalized field; `--fields` enables multi-field rendering.
- `--volume-mode supercell|composite|isolated|cycle` controls whether fields render together or independently.
- `--texture-mode natural` adds world-space micro-detail to reduce synthetic smoothness.
- `--camera-mode orbit` provides a turntable view around the storm core.
- `--camera-mode freefly` enables constrained interactive motion:
  `WASD` move, `Q/E` descend/ascend, arrows or right-mouse look, `Shift` boost, `R` reset pose.
- Missing fields are skipped with a warning; known aliases are attempted automatically (e.g., `qi` resolves to `qh`).
- Cartesian export manifests automatically set physical volume proportions. Use
  `--voxel-spacing-m x,y,z` for SHM-only or manifest-less data; vertical
  exaggeration is explicit and defaults to `1.0`.
- `--style cinematic-bw` enables a desaturated high-contrast storm palette.

### Ray Marching Controls

| Flag | Purpose |
|------|---------|
| `--ray-steps` | Maximum adaptive ray march steps |
| `--ray-threshold` | Density threshold for visibility |
| `--ray-opacity` | Global opacity multiplier |
| `--ray-brightness` | Brightness multiplier |
| `--ray-ambient` | Ambient light level |
| `--ray-anisotropy` | Phase function anisotropy |
| `--ray-max-distance` | Maximum ray travel distance |
| `--voxel-spacing-m` | Physical cell spacing override in metres (x,y,z) |
| `--vertical-exaggeration` | Explicit vertical display scale |
| `--sun-dir` | Sun direction vector (x,y,z) |
| `--playback-fps` | Time-based frame playback rate |

## Headless Input Validation

From repository root:

```bash
make smoke-test-e2e

# Validate an existing export without creating a Vulkan device or window.
./bin/vulkan_viewer --validate-input --input data/exports --fields theta,w
```

The E2E target runs a small CPU simulation, verifies atomic run/step manifests
and payloads, and loads every `theta` and `w` frame through the native reader.

## Physical hardware checks

Use the maintained compute-dispatch, parity and residency targets with a Vulkan
device that supports the selected profile. Software Vulkan can exercise API and
numerical contracts; it does not establish discrete-memory transfer behavior or
physical GPU throughput. Inspect the device and memory-policy diagnostics and
retain the exact executable/configuration identity with the result.

A sustained presentation or performance campaign needs an explicitly chosen
workload, duration, thermal state and reviewed thresholds for that hardware.
The repository does not publish a machine-specific baseline or infer one GPU's
acceptance limits for every Vulkan device. See the
[architecture guide](../docs/ARCHITECTURE.pdf) for ownership, synchronization and
measurement constraints. Long campaigns and scientific review remain separate
from bounded dispatch/parity checks.

## GPU Compute

The Vulkan backend also provides GPU compute sources for advection, acoustic
and dynamics work, microphysics, turbulence, and diffusion. This is separate
from the viewer and configured via the simulation YAML; source presence does
not mean every path is active or physically qualified:

```yaml
extensions:
  compute:
    backend: gpu       # cpu | gpu; vulkan/vk are accepted GPU aliases
```

Exploratory execution may allow CPU fallback when explicitly configured.
Qualified `nimbus.v1` execution forces fallback off, and
`extensions.compute.allow_fallback` exposes the policy in ordinary strict-v1
configuration. Simulator device-index selection is intentionally not exposed.

Compute shaders live in `shaders/compute/` and ship as SPIR-V (`.spv`) in the
repository. Ordinary root builds verify the source, GLSL include and artifact
identities, generated division variants and compiled kernel contract before
using those bytes. They do not regenerate shaders with the host's compiler.
Changed or missing assets fail verification. After an intentional shader change,
use `make vulkan-compute-shaders` and `make verify-shaders` with compatible
GLSL/SPIR-V tools, including `SPV_KHR_fma` support, and review the generated diff.

The maintained kernel manifest, generated shader identities and parity tests
are the current operation inventory; counts change as implementations evolve.
On a suitable physical device, `make qualify-vulkan-evidence` runs the selected
native qualification tools and writes generated evidence under `reports/`.
Hardware campaigns are separate from a clean source build, and their artifacts
must be reviewed against the declared scope before making qualification claims.
