# Building and running Nimbus

Nimbus has a C++17 simulator and an optional Tauri desktop application. The
normal setup route builds both. A terminal-only route builds the same solver
without the desktop toolchain. Run repository commands from the checkout root
unless a command explicitly changes directory.

## Build and open only

Use this path when checking a new Windows, Linux or macOS machine. The goal is
to compile Nimbus and open its native desktop window. No simulation, sample
dataset, scientific table download or qualification campaign is required.
The build and Studio may invoke the simulator's version and catalog commands;
these describe the executable and do not advance an atmosphere.

Clone the repository with Git and enter the checkout. Use a short local path
on Windows, such as `C:\src\Nimbus`, to avoid toolchain path-length limits.
Keep the terminal output if a step fails. Dependency downloads need an internet
connection and enough disk space for the C++, npm and Cargo build trees.

### macOS

Check and provision the desktop dependencies, then build a bundle without
installing it over another copy:

```sh
export NIMBUS_BUILD_JOBS=2 CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
./scripts/setup.sh --check
./scripts/setup.sh --dependencies-only
npm --prefix ui ci
bash ui/scripts/build_native_app.sh --bundle-only
open "ui/src-tauri/target/release/bundle/macos/Nimbus Studio.app"
```

The package script builds the simulator and its private runtime libraries.
The initial dependency build can take substantially longer than later builds.
The regular `./scripts/setup.sh` route also installs the app in `/Applications`;
the commands above leave it in the build directory.

### Linux

Use a graphical desktop session on a distribution covered by the setup script:

```sh
export NIMBUS_BUILD_JOBS=2 CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
./scripts/setup.sh --check
./scripts/setup.sh --dependencies-only
make -j2 REQUIRE_FULL_STACK=1
npm --prefix ui ci
bash ui/scripts/build_native_app.sh --bundle-only
```

Launch the `.AppImage` path printed by the build. The script marks it executable.
If the host reports that FUSE is unavailable, try the same AppImage with
`--appimage-extract-and-run` and record that requirement in the test report;
see the [AppImage FUSE guidance](https://docs.appimage.org/user-guide/troubleshooting/fuse.html).
An SSH terminal without a graphical display is not a desktop launch check.

### Windows

Use a native 64-bit PowerShell terminal, with Git available:

```powershell
$env:NIMBUS_BUILD_JOBS = "2"
$env:CARGO_BUILD_JOBS = "2"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "2"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Check
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The script provisions the architecture-native C++ and Rust toolchains, builds
the solver, stages its DLLs and produces an NSIS installer. The execution-policy
setting applies only to that PowerShell process. The build selects native CMake
and libclang for Rust's readers and checks that their architecture matches the
Rust MSVC toolchain. Run the installer
path printed at completion, then launch **Nimbus Studio** from the Start menu.
Follow the setup output if a tool installation requires a new terminal or a
restart. Use this Windows route instead of the Bash packaging command. An
unsigned local installer may show an operating-system trust prompt; record
whether it prevented installation or launch.

These commands limit C++ and Rust compilation to two workers to reduce memory
pressure on unfamiliar machines. Increase the limits if memory permits;
`NIMBUS_BUILD_JOBS` controls Nimbus's build scripts, while `CARGO_BUILD_JOBS`
controls Rust compilation. `CMAKE_BUILD_PARALLEL_LEVEL` also bounds dependency
builds that use CMake. They do not configure simulation threads.

For all three platforms, a nonzero dependency-check result identifies missing
prerequisites; it is not itself a compiler failure. Complete the setup prompts
and repeat the check before treating a dependency problem as a source defect.

### What a successful first opening looks like

On a fresh profile, Studio opens **Configure** with configuration controls and
navigation for **Projects**, **Runs**, **Plots** and **Settings**. Configuration
templates are inputs, not completed simulations. No run starts automatically.
An existing local profile may restore a previously selected page or preference.

- **Runs** has no local runs and no selected run.
- **Plots** shows **Open data**; there are no atmospheric fields, frames or
  animations until a real dataset is opened.
- **Projects** and recent-data lists can be empty. Creating or importing a
  project is optional for this check.
- **Settings** should open and the window should resize, close and reopen
  normally. The packaged simulator should be discoverable; a persistent
  missing-binary, failed-catalog or missing-DLL error is a problem to report.

The first desktop session may briefly measure loading workers using generated
temporary data. You can stop it and inspect or change its settings under
**Settings → Performance**; this does not run the simulator.

Visit the empty pages and Settings, then close and reopen the app. **Do not
press Run, start a guided simulation, provision Thompson/RRTMG tables or create
fake output to fill the views.** Missing simulation results are expected;
a blank window, crash, endless loading indicator or failed-view message is not.
No pre-run storm dataset is shipped in this repository.

Send the commit (`git rev-parse HEAD`), OS/version, CPU architecture, RAM,
GPU/driver if known, and the exact build command. Report build, installation
and first opening separately. Include the first useful error and its surrounding
terminal output, plus a screenshot if the window fails. Note whether the app
closed and reopened successfully and whether you used an existing Studio profile.
This establishes build/startup behavior on that machine; it does not validate
simulation correctness, GPU compute, output formats or performance.

## Guided setup

On macOS and Linux:

```sh
./scripts/setup.sh --check
./scripts/setup.sh
```

`--check` reports dependencies without installing or building. Normal setup
explains the platform route and asks before installing packages. Use
`--dependencies-only` to stop after dependencies, or `--headless-only` to omit
Studio. `--yes` accepts the script's installation prompts for automation;
it does not make unsupported platforms or scientific configurations supported.
On macOS the default desktop build installs Nimbus Studio in `/Applications`.

On Windows, use native PowerShell and the architecture-native toolchain:

```powershell
.\scripts\setup.ps1 -Check
.\scripts\setup.ps1
```

The script selects the maintained MSYS2/Rust route for x64 or ARM64. See
[platform scope](SUPPORTED_PLATFORMS.md). Build routes,
deployment metadata and tests on a particular machine are distinct evidence.

## Manual dependencies

The solver needs a C++17 compiler, Make, Python 3 and OpenMP. The ordinary
native build also needs Vulkan headers and a loader; GLSL/SPIR-V tools are used
when generating or verifying shaders. macOS additionally uses its system Metal
frameworks. ZFP and netCDF provide the optional compressed/scientific output
formats; full product packaging requires those features.

For a typical Homebrew development environment:

```sh
xcode-select --install
brew install libomp vulkan-headers vulkan-loader molten-vk glslang glfw
brew install zfp netcdf python3 node cmake ffmpeg
```

For an Ubuntu/Debian development environment:

```sh
sudo apt update
sudo apt install build-essential python3 libomp-dev libvulkan-dev glslang-tools spirv-tools libglfw3-dev pkg-config
sudo apt install libzfp-dev libnetcdf-dev netcdf-bin
```

Studio also needs Node/npm, Rust/Cargo and the operating system's Tauri build
prerequisites. Use Node.js 22.13 or later within the 22.x series, or Node.js 24
or later; CI selects Node.js 22. The crate declares Rust 1.77.2 as its minimum; the locked
native dependency set and release checks determine the tested build toolchain.
On Linux the setup route installs the required WebKitGTK and application
integration libraries. FFmpeg is needed for movie export.

The setup scripts are the maintained package-manager mapping. Package names,
GPU drivers and operating-system libraries vary; inspect `--check` rather than
assuming another computer's installed stack is present.

## Choose a solver build

```sh
make -j4
make help
```

The ordinary build produces `bin/nimbus_sim` and `bin/field_validator`. Limit
`-j` to suit available CPU and memory. Common options are:

| Option | Purpose |
| --- | --- |
| `CPU_ONLY=1` | Build without the Vulkan implementation and shader compilation |
| `PORTABLE=1` | Avoid host-native CPU ISA tuning |
| `DISTRIBUTION=1 PORTABLE=1` | Enforce the portable distribution build contract |
| `ZFP=0` | Disable optional ZFP output explicitly |
| `NETCDF=0` | Disable optional netCDF output explicitly |
| `CHECKED=1` | Retain assertion checks in the selected profile |
| `SANITIZE=address-undefined` | Instrument supported builds for memory/undefined-behavior checks |

For example, a separate portable CPU build is:

```sh
make -j4 CPU_ONLY=1 PORTABLE=1 all
./bin/nimbus_sim_cpu_portable --version
```

Build profiles have separate object directories and artifact names. Inspect
`--version` to confirm which features were compiled. CPU-only or reduced-output
binaries are useful development tools; they are not substitutes for the full
native simulator when a test or package requires those capabilities.

An ordinary native build verifies the committed SPIR-V shader artifacts,
their GLSL sources/includes and the compiled kernel manifest. It uses those
reviewed bytes instead of regenerating them with the host's shader compiler.
This keeps a clean build independent of shader-optimizer differences between
platform packages. Changed or missing shader assets fail the build.

Shader developers use `make vulkan-compute-shaders` to rebuild and refresh the
artifacts deliberately, then `make verify-shaders` to validate reproducibility
and the kernel contract. That workflow needs compatible GLSL/SPIR-V tools,
including support for `SPV_KHR_fma`; older distribution tools may not provide
it. Rebuilding shaders is separate from the build-and-open check above.

To reproduce the declared toolchain, run
`bash scripts/dev/shaders/build_spirv_tools.sh build/shader-toolchain`, then use
`PATH="$PWD/build/shader-toolchain/bin:$PATH" make verify-shaders`. This builds
the pinned GLSL compiler together with its matching SPIR-V optimizer. Updating
only the standalone optimizer leaves an older compiler's embedded optimizer
unchanged.

Portable describes CPU instruction selection within one architecture. It does
not remove dynamic-library requirements. In particular, a local macOS build can
link to Homebrew libraries that require a newer OS than the application's
minimum deployment setting. The macOS packaging path builds a separate pinned
runtime stack and targets macOS 12.0, matching its MoltenVK dependency floor.
Bundled Mach-O deployment metadata is checked; the setting alone does not prove
runtime acceptance on macOS 12. An ordinary development binary is current-host
evidence. See [platform compatibility](SUPPORTED_PLATFORMS.md).

## Provision scientific inputs

Thompson and RRTMG require external table files even when the executable builds
successfully. Provision only the tables required by the selected physics:

```sh
make -j4 thompson-data
bash scripts/dependencies/fetch_rrtmg_coefficients.sh
```

`make thompson-data` downloads the [pinned runtime-table release](https://github.com/vtiradoegas2025/Nimbus/releases/tag/runtime-tables-v1)
and verifies the archive and every member before publishing any input. It installs
`data/thompson/ccn_activation.ntact`, `data/thompson/collection_v1.ntctbl` and
`data/thompson/cold_process_v1.ntcold`, with upstream source, license and provenance.
The download is about 296 MiB and the installed set about 357 MiB. Existing verified
files are reused. Different compiler/math-library arithmetic can generate different
floating-point bytes, so ordinary provisioning distributes the unchanged canonical
inputs instead of weakening their identities. Runtime checks still reject missing
or incompatible tables.

For offline installation, download the same release archive beforehand and run:

```sh
python3 scripts/dependencies/fetch_thompson_tables.py --archive /path/to/thompson-wrf471-v1.tar.gz
```

`make thompson-generate-data` retains the explicit C++ generator workflow for
maintainers. Its output must satisfy the existing pinned identities; an alternate
compiler's output is not automatically a replacement scientific input.

The RRTMG downloader obtains longwave and shortwave coefficients from pinned
AER revisions, verifies their SHA-256 identities and creates
`data/rrtmg/rrtmg_lw.nc` and `data/rrtmg/rrtmg_sw.nc`. Set
`NIMBUS_RRTMG_LW_COEFFICIENTS` and `NIMBUS_RRTMG_SW_COEFFICIENTS` for files stored
elsewhere. netCDF support must also be present in the build/runtime.

No runtime data directory is tracked in Git. Provisioning creates ignored local
data; deploying an executable alone does not supply its physics tables. Keep
the selected inputs available on every execution machine, and start source-tree
runs at the repository root so default relative paths resolve consistently.

## Validate and run an example

Start with a coarse configuration and a short duration:

```sh
./bin/nimbus_sim --validate-only --config=configs/1000meterResolution/kessler/baseline.yaml
./bin/nimbus_sim --headless --config=configs/1000meterResolution/kessler/baseline.yaml --duration=10 --threads=4 --outdir=runs/quickstart
```

The public presets declare their execution and validation modes. Most catalog
presets are exploratory starting points; a valid file is not evidence of a
qualified storm. Read [the configuration guide](../configs/README.md), retain
validation diagnostics and choose an output directory with sufficient space.
Use `--help`, `--version` and `--scheme-catalog=json` on the actual executable
rather than relying on an old binary's options or capabilities.

## Build Studio

```sh
cd ui
npm ci
npm run typecheck
npm run test:unit
npm run build
npm run tauri:dev
```

`npm run build` produces the frontend bundle; it does not create a desktop
installer. `tauri:dev` uses the Rust shell and the locally built simulator.
`npm run dev` starts only the browser development frontend, whose filesystem
and process behavior differs from the native shell.

For a desktop package without installing it, return to the repository root:

```sh
bash ui/scripts/build_native_app.sh --bundle-only
```

On macOS, `--bundle-dir /absolute/output/directory` also copies and signs the
assembled `.app` there. That route builds compatible private runtime libraries
and a portable simulator before packaging. Its dependency cache defaults to
`build/macos-runtime`; `NIMBUS_MACOS_RUNTIME_WORK` and `NIMBUS_BUILD_JOBS` select
another cache location and build parallelism. Linux creates an AppImage. Native
Windows uses `ui/scripts/build_native_app.ps1` and NSIS.

A package must be tested on its target OS and architecture, including scientific
input availability, native dispatch, output reopening and restart. A successful
link, frontend build or installer creation does not complete those checks.

For a Windows release that requires Authenticode signing, set
`NIMBUS_WINDOWS_SIGNING_REQUIRED=1` and provide
`NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT` for a certificate available in the
appropriate certificate store. `NIMBUS_WINDOWS_TIMESTAMP_URL` selects the
timestamp service; the repository does not pin one. Inspect the produced
installer with `Get-AuthenticodeSignature` and complete the installed-product
checks. No certificate or password belongs in the repository.

## Checks and generated files

```sh
make check-docs
make test-script-suite
make test
```

The full suite has data and backend prerequisites. Use the relevant named
CPU/native targets for a focused change and record unavailable hardware honestly.
Long physical campaigns and signing/install acceptance are separate release
work, not an automatic consequence of a source build.

`build/`, `bin/`, `ui/build/`, Cargo `target/`, downloaded data and run outputs
are regenerable local files. Do not commit them. `make compile-commands` creates
an ignored editor compilation database. `make clean` affects the selected build
products; inspect its target before deleting unrelated working data.

## Rebuild the technical documents

The mathematics, architecture and Studio guides are LaTeX PDFs. Their maintained
sources live with the [document build scripts](../scripts/dev/docs/latex/README.md);
`docs/` holds the published PDFs and reader-facing Markdown guides.
Install Tectonic and run:

```sh
make docs-pdf
python3 scripts/dev/docs/build_guides.py --check
```

The first build may download TeX packages and fonts. Intermediate files remain
under the ignored `build/docs/` directory. Commit the `.tex` sources, PDF and
source-identity manifest together.
