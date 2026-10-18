# Nimbus

Nimbus is an atmospheric simulation engine with a desktop workspace for configuring
runs, inspecting scientific output and comparing datasets. The C++ solver runs
from the terminal or through Nimbus Studio, a Tauri/React application. CPU,
Vulkan and Metal execution are available for the combinations admitted by the
runtime catalog.

Nimbus has been an independent codebase from the beginning; it is not a port
or fork of CM1. Its goal is to make high-fidelity atmospheric experiments
practical on ordinary workstations and MacBooks through efficient algorithms,
GPU acceleration and a configurable, maintainable solver. The technical guides
explain how to evaluate resolution, accuracy and resource cost, and when an
experiment needs resources beyond one computer.

Nimbus is under active scientific and engineering development. A successful
build or an admitted configuration is not full scientific qualification. Some
couplings remain incomplete and are deliberately rejected. Read the
[mathematics and assumptions](docs/MATHEMATICS.pdf) before interpreting results.

## Build and run

For a first check on another computer, follow
[build and open only](docs/BUILDING.md#build-and-open-only). It covers Windows,
Linux and macOS without running a simulation. Studio starts with configuration
controls and empty run/data views; this repository ships no pre-run simulations.

On macOS or Linux, start by checking the required tools:

```sh
./scripts/setup.sh --check
```

The normal setup builds the solver and Studio. It explains the platform route
and asks before installing dependencies. On macOS the normal desktop build also
installs the application in `/Applications`.

```sh
./scripts/setup.sh
```

### Secondary terminal-only setup and launch

For terminal use without building Studio:

```sh
./scripts/setup.sh --headless-only
./bin/nimbus_sim --validate-only --config=configs/1000meterResolution/kessler/baseline.yaml
./bin/nimbus_sim --headless --config=configs/1000meterResolution/kessler/baseline.yaml --duration=10 --outdir=runs/quickstart
```

These public presets are starting experiments, not proof of a qualified storm
simulation. Their execution and validation settings are explained in the
[configuration guide](configs/README.md). Thompson microphysics and RRTMG also
need the [runtime tables](docs/BUILDING.md#provision-scientific-inputs); generated tables and downloaded
coefficients are provisioned separately from Git source.

On native Windows, use PowerShell:

```powershell
.\scripts\setup.ps1 -Check
.\scripts\setup.ps1
```

See [BUILDING.md](docs/BUILDING.md) for manual dependencies, CPU-only builds,
portable binaries and packaging without installation. The
[platform guide](docs/SUPPORTED_PLATFORMS.md) distinguishes build targets from
verified runtime compatibility.

## Using Nimbus

Studio opens configurations, launches local or remote runs and reads completed
output for plotting. [The Studio guide](ui/README.md) covers development,
remote compute and its checks. The simulator remains usable independently:

```sh
./bin/nimbus_sim --help
./bin/nimbus_sim --version
./bin/nimbus_sim --scheme-catalog=json
```

The catalog is the authoritative description of selectable schemes and their
admission constraints. Do not infer support for a combination merely because
its individual schemes appear in the catalog. Output and checkpoint settings
are part of that combination.

## Understanding and changing the model

- [Mathematics and assumptions](docs/MATHEMATICS.pdf): governing variables,
  numerical approximations, closure choices and limits.
- [Architecture](docs/ARCHITECTURE.pdf): configuration, state ownership,
  backend memory, publication and Studio boundaries.
- [Scientific references](docs/foundationalScience.md): primary literature
  and its relationship to the implementation.
- [Namelist and configuration reference](docs/NAMELIST.md): every strict-v1
  setting, including defaults, dependencies and guarded or unfinished options.
- [Contributing](CONTRIBUTING.md): code layout, tests and change discipline.

Source, maintained tests, public configurations and these guides belong in Git.
Build products, downloaded dependencies, run output and historical investigation
records do not. See [the documentation index](docs/README.md) for the maintained
guides and [LICENSE](LICENSE) for the project license.
