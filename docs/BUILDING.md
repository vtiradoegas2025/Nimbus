# Building Nimbus

---

## Quick Start (all platforms)

```bash
git clone https://github.com/vtiradoegas2025/Nimbus.git
cd Nimbus
./scripts/setup.sh        # detects OS, installs dependencies
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
make test
```

`setup.sh` handles macOS (Homebrew), Ubuntu/Debian/Pop!_OS (apt), Fedora (dnf), and Arch (pacman). Use `--check` to see what's missing without installing anything. Use `--headless` to skip Vulkan/viewer dependencies if you only need the simulation engine.

---

## Dependencies

| Dependency | Required | Purpose | Notes |
|-----------|----------|---------|-------|
| C++17 compiler | Yes | Core build | g++ >= 7.0 or clang++ >= 9.0 |
| GNU Make | Yes | Build system | |
| OpenMP | Recommended | Parallel computation | 4-8x speedup; simulation runs single-threaded without it |
| Vulkan SDK | Optional | GPU compute + 3D viewer | Headers, loader, and platform runtime |
| glslangValidator | Optional | Compile GLSL shaders to SPIR-V | Pre-compiled binaries included; only needed if modifying shader source |
| GLFW | Optional | Windowed viewer surface | Required for the Vulkan viewer's window mode |
| pkg-config | Optional | Library detection on Linux | Helps find Vulkan/GLFW paths |
| ZFP | Optional | Scientific data compression | Enable with `make ZFP=1` |
| ffmpeg | Optional | Demo video recording | Only for `scripts/demo.sh --record` |

---

## Platform-Specific Instructions

### macOS

Requires [Homebrew](https://brew.sh).

```bash
# Core (simulation only)
# Xcode CLT provides clang++ and make
xcode-select --install
brew install libomp

# Viewer (optional)
brew install vulkan-headers vulkan-loader molten-vk glslang glfw

# Build
make -j$(sysctl -n hw.ncpu)
make test

# Build viewer
make vulkan
```

**Notes:**
- The system `g++` on macOS is actually Apple Clang. This is fine -- the build auto-detects it.
- If OpenMP is not found, the simulation still builds but runs single-threaded. Install `libomp` for parallel performance.

### Ubuntu / Debian / Pop!_OS

```bash
# Core
sudo apt update
sudo apt install g++ make libomp-dev

# Viewer (optional)
sudo apt install libvulkan-dev vulkan-tools glslang-tools libglfw3-dev pkg-config

# Build
make -j$(nproc)
make test

# Build viewer
make vulkan
```

**Pop!_OS note:** Pop!_OS includes the System76 driver manager which often pre-installs Vulkan drivers for NVIDIA/AMD GPUs. The `libvulkan-dev` package provides the headers and loader needed for compilation.

### Fedora / RHEL

```bash
# Core
sudo dnf install gcc-c++ make libomp-devel

# Viewer (optional)
sudo dnf install vulkan-headers vulkan-loader-devel glslang glfw-devel pkgconf-pkg-config

# Build
make -j$(nproc)
make test

# Build viewer
make vulkan
```

### Arch Linux

```bash
# Core
sudo pacman -S gcc make openmp

# Viewer (optional)
sudo pacman -S vulkan-headers vulkan-icd-loader glslang glfw-x11 pkgconf

# Build
make -j$(nproc)
make test

# Build viewer
make vulkan
```

### Windows (WSL 2)

Native Windows builds are not supported. Use Windows Subsystem for Linux:

1. Install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu
2. Follow the Ubuntu instructions above inside your WSL terminal
3. For the Vulkan viewer, you need WSLg (included in Windows 11) for GPU passthrough

```powershell
# In PowerShell (admin)
wsl --install -d Ubuntu
```

Then open the Ubuntu terminal and follow the Ubuntu section above.

---

## Build Targets

```bash
make                    # Build simulation + auto-compile GPU shaders
make vulkan             # Build Vulkan viewer (bin/vulkan_viewer)
make test               # Run full test suite
make check-deps         # Check which dependencies are installed
make clean              # Remove build artifacts

# Headless simulation
./bin/tornado_sim --headless --config=configs/student/student.yaml --duration=60

# One-command demo (builds + launches sim + viewer)
./scripts/demo.sh --preset quick
```

`make` builds the simulation binary (`bin/tornado_sim`) and automatically recompiles any GPU compute shaders whose source has changed (requires `glslangValidator`). Pre-compiled SPIR-V binaries are included in the repository, so GPU compute works on a fresh clone even without the shader compiler installed.

### Build Options

```bash
make CXX=clang++        # Override compiler
make ZFP=1              # Enable ZFP compression
make -j8                # Parallel build (set to your core count)
```

---

## LunarG Vulkan SDK

If you installed the Vulkan SDK from [LunarG](https://vulkan.lunarg.com/sdk/home) instead of your package manager, set the `VULKAN_SDK` environment variable before building:

```bash
source ~/vulkansdk/setup-env.sh   # or wherever you installed it
make
make vulkan
```

Both Makefiles (root and vulkan/) respect `VULKAN_SDK` for include and library paths.

---

## Verifying Your Setup

```bash
# Check all dependencies at once
make check-deps

# Or use the setup script in check-only mode
./scripts/setup.sh --check
```

Example output when everything is installed:

```
=== Nimbus dependency check ===
C++ compiler:   Apple clang version 16.0.0
GNU Make:       GNU Make 4.4.1
OpenMP:         available (Homebrew libomp)
Vulkan headers: found
glslangValidator: glslang 14.0.0
GLFW:           found (Homebrew)
=== Done ===
```

---

## Troubleshooting

**"OpenMP not found" warning during build**

The simulation builds without OpenMP but runs single-threaded. Install the OpenMP runtime for your platform (see dependency table above). On macOS, this is `brew install libomp`.

**"Vulkan headers/loader not found" when building viewer**

The viewer requires Vulkan development packages. On Linux, make sure you have both headers and the loader:
- Ubuntu: `sudo apt install libvulkan-dev`
- Fedora: `sudo dnf install vulkan-headers vulkan-loader-devel`

If using the LunarG SDK, ensure `VULKAN_SDK` is set (see above).

**"glslangValidator not found" when building viewer**

The SPIR-V shader compiler is required to build the Vulkan viewer (rendering shaders must be compiled). For the simulation-only build (`make`), pre-compiled compute shaders are included in the repository, so `glslangValidator` is only needed if you modify `.comp` shader source files. Install from your package manager or the Vulkan SDK.

**Build fails with "c++17" errors**

Your compiler is too old. Nimbus requires g++ >= 7.0 or clang++ >= 9.0. Update your compiler or set `CXX` to a newer version:
```bash
make CXX=g++-12
```

**Linker errors about `-lvulkan`**

The Vulkan loader library is not in your linker search path. Install `vulkan-loader` (macOS), `libvulkan-dev` (Debian), or set `VULKAN_SDK`.

**Tests pass but simulation is slow**

Check that OpenMP is working: `make check-deps` should show "OpenMP: available". Without it, the simulation uses a single CPU core.
