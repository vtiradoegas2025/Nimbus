# Platforms and compatibility

Nimbus maintains build and packaging routes for the following 64-bit desktop
families. A row describes a target and its tools; it is not a blanket claim
that every hardware model, OS release or scientific combination has been tested.

| Platform | Native CPU targets | Compute routes | Package route |
| --- | --- | --- | --- |
| macOS | Apple Silicon, Intel x86-64 | CPU, Metal, Vulkan through MoltenVK | `.app` |
| Linux | x86-64, ARM64 | CPU, Vulkan with a suitable driver | AppImage |
| Windows | Windows 10/11 x64; Windows 11 ARM64 | CPU, Vulkan with a suitable driver | NSIS installer |

The source setup scripts provide Homebrew, Debian/Ubuntu, Fedora and Arch
package-manager routes. Windows setup selects an architecture-native MSYS2 and
Rust stack. The project does not target 32-bit CPUs or mobile products.

The macOS packaging minimum is 12.0, matching the pinned MoltenVK 1.4.2 runtime's
[upstream deployment target](https://github.com/KhronosGroup/MoltenVK/blob/v1.4.2/MoltenVK/MoltenVK.xcodeproj/project.pbxproj).
That setting does not prove runtime or WebView acceptance on macOS 12. The
package verifier checks the deployment metadata of bundled Mach-O files;
actual installed-application tests are still required. A standalone solver's
lower source-build target does not lower the desktop package's dependency floor.
Intel, ARM64 and x86-64 builds
must each be tested as the architecture they claim; emulation is separate
information. Hardware-specific performance claims require measurements on that
hardware and configuration.

CPU/OpenMP and GPU memory behavior also vary within an architecture. A discrete
GPU's host-visible memory window is not a unified host/device pool. Nimbus uses
capability-driven admission and memory policy, while preserving explicit user
controls. Passing a generic x86-64 build on one AMD workstation does not prove
that every Intel/AMD CPU or every Vulkan driver has been exercised.

Use the executable's `--version` and `--scheme-catalog=json` output to establish
its compiled features and admitted scientific profiles. Consult current CI and
package test results for the specific artifact; long campaigns and independent
scientific review remain separate from build compatibility.

See [BUILDING.md](BUILDING.md) and
[the architecture guide](ARCHITECTURE.pdf).
