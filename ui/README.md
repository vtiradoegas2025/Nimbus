# Nimbus Studio

Nimbus Studio is the desktop workspace for configuring Nimbus, launching runs
and inspecting scientific datasets. React/TypeScript provides the interface;
Tauri's Rust shell owns process control, filesystem access and native readers.
Plotly and VTK provide plotting, and Pyodide runs the Python workspace. The
simulator is a separate C++ executable.

Read [the Studio guide](../docs/STUDIO.pdf) for detailed workflows, every plot
family, analysis, rendering, resource ownership and publication limits.
Read [the build guide](../docs/BUILDING.md) for toolchains and packaging, and
[the architecture guide](../docs/ARCHITECTURE.pdf) for ownership and publication
boundaries. Scientific assumptions are documented in
[the mathematics guide](../docs/MATHEMATICS.pdf).

For a new-machine check, use [build and open only](../docs/BUILDING.md#build-and-open-only).
With no saved runs, **Runs** is empty and **Plots** offers **Open data**. This is
the expected first-run state. Opening Studio does not start a simulation.
The [namelist reference](../docs/NAMELIST.md) also covers solver settings that
the form does not expose, including their current restrictions.

## Loading workers and I/O memory

On the first desktop launch on a new computer, Studio runs a short automatic
measurement after the interface becomes idle. It creates a small temporary NPY
file, compares cached reads and scanning at several worker counts, and removes
the file afterward. You can stop it or rerun it in **Settings → Performance**.
The saved recommendation applies when **Loading workers** is **Automatic**;
an explicit count remains your choice. The profile is reused only for matching
CPU, core-count and RAM information. It measures Studio loading, not simulator
threads, GPU performance or sustained disk bandwidth. Different file formats
and working datasets can favor different counts.

**I/O memory limit (MiB)** controls temporary native field-reading memory. Use
`0` for an automatic limit based on this computer's RAM, or enter a manual value
within the range shown. Save applies the limit without leaving the plot. A
temporary shortage queues reads; a single request that exceeds the limit reports
the required size. Increase the limit or choose a smaller region. Failed
budget-related plot reads retry after a changed limit is applied. Lowering the
limit lets existing reads finish while new reads wait for capacity.

The I/O limit is separate from the field cache, Python memory and GPU resources;
raising it does not increase video memory or remove a plot's other capacity
checks. Deselecting a field or leaving its plot cancels that plot's unused
prefetches. A shared read stays active while another visible consumer needs it.

## Python coordinates and display detail

The Python workspace retains the complete numeric result for export. Large
array plots use a bounded preview; zooming refines the visible samples from
that retained result. Heatmap colors keep the full result's finite range while
zooming. Native cell boundaries can remain visible: interpolation does not add
scientific resolution. Matplotlib figures follow the rendering choices in the
script rather than the array-preview policy.

Use `coordinate_units` with `x_coordinates`, `y_coordinates` and `z_coordinates`
to inspect the bound grid. `x_m`, `y_m` and `z_m` contain only physical metre
coordinates; an unavailable physical axis is empty. Cylindrical angles are
available as `theta_rad`. Index-only geometry stays in index units, and terrain
levels stay model-level indices unless the loading path explicitly remaps them
to physical height. `dataset()` preserves these units and the supplied scalar
coordinates, including translated origins and nonuniform spacing.

## Develop locally

Build the simulator at the repository root first. Studio can discover it in the
checkout; `NIMBUS_SIM_BIN` or the binary-path preference selects another build.
Use Node.js 22.13+ within 22.x or Node.js 24+, Rust/Cargo, and the
Tauri system dependencies installed by the setup route.

```sh
cd ui
npm ci
npm run typecheck
npm run test:unit
npm run tauri:dev
```

`npm run dev` starts the browser development frontend, whose mocked/stubbed
native operations are not a complete local-run workflow. `npm run build` creates
the frontend bundle. `npm run tauri:build` invokes the native packaging script;
on macOS its default route also installs the application. To build without
installation, use `bash ui/scripts/build_native_app.sh --bundle-only` from the
repository root.

## Remote compute

Studio can launch simulations on named Linux SSH connections while plots and
playback run on the viewing computer. Connections are user preferences; no
workstation name, credentials, or personal directory is built into the feature.
Each configuration opens with **This computer** selected.

The viewing computer needs OpenSSH. The compute node needs Python 3.8+, an
installed Nimbus simulator with the current catalog contract, and an existing
writable work directory. Set up key authentication and the known host using
your SSH client first. Studio uses your SSH configuration and agent, verifies
the host key, and does not show password or host-trust prompts.

1. In **Settings → Compute nodes**, save a connection name, SSH alias or
   `user@host`, absolute simulator path, and absolute work directory. Use
   **Check connection** to verify that node's executable and catalog.
2. Optional input mappings stage local files at relative paths in each job's
   private directory. Use those relative paths in the YAML. Absolute mappings
   under the configured work directory refer to existing remote files whose
   contents must already match the selected local file; Studio verifies them
   without overwriting them. The staged YAML bytes are preserved.
3. In **Configure → Compute target**, select the connection and choose the
   simulation's compute backend. Catalog discovery, checkpoint inspection,
   validation, preflight, migration, and launch use the selected node. The
   executable, catalog, target, inputs, and YAML must still match at launch.
4. Follow progress and logs in Runs. Confirmed completed, failed, and stopped
   histories are copied into the local Runs directory for plotting and
   continuation. **Receiving results** identifies this phase; **Cancel transfer**
   stops the copy. A frame directory is published only after its files pass
   byte-count and SHA-256 checks. Remote files remain available after transfer
   failure. Live remote frame streaming is not provided.

An unreachable node produces an explicit error. **Remote termination is
unconfirmed** requires restoring the connection and choosing **Retry stop**;
Studio does not report the remote process as stopped without confirmation.
Changing or removing a selected connection cannot silently launch locally.
Runs belong to the active Studio session. Keep the viewing computer awake and
connected while a run is active: quitting or losing the connection retires the
remote job through its ownership lease. Detached scheduler jobs are not provided
by this direct SSH workflow.

Rendering budgets use the viewing computer's RAM and CPU facts. A compute
node's GPU or memory does not raise the viewing computer's resource limits.
Requested playback rates through 60 fps wait for each visible plot's completed
saved frame; achievable cadence depends on the viewer, data, and plot.

## Verification

```sh
npm run typecheck
npm run test:unit
npm run build
npm run typecheck:browser
npm run test:browser
```

The unit runner discovers the maintained TypeScript contracts under
`tests/ui/`. Tests construct their arrays and temporary files during execution;
recorded personal datasets and fixtures are not part of the application tree.
Browser checks need the Playwright browser dependencies. WebKit and native
packaged checks are separate evidence from Chromium. The Python test runtime
is downloaded into ignored caches and verified by
[../tests/ui/vendor_pyodide.mjs](../tests/ui/vendor_pyodide.mjs). A first run needs
network access; subsequent runs can reuse the verified cache.

Native-shell checks use Cargo in `src-tauri/`. Some integration cases require
a simulator or a real packaged application and are explicitly marked. A skipped
native test is not a passing installed-application test. See the script names
in [package.json](package.json) and the maintained CI workflows.

## Data and process boundaries

A completed-frame record, not the mere presence of an array file, makes an
output frame eligible for display. Dataset reads, derived products and plot
preparation retain frame identity so that cancellation or a late read does not
publish a mixture of model times. Caches and worker queues have explicit owners
and resource bounds.

CM1 is a read-only data and configuration-interchange source. Studio does not
install or run the CM1 model. Translation distinguishes mechanical mappings
from choices that need scientific interpretation. The
[CM1 readers and translation code](src/lib/cm1/) implement this boundary.

Local projects, remote connection preferences, run outputs, Node dependencies,
frontend bundles and Cargo products are not repository source. Keep generated
files in their ignored directories and use lockfiles for reproducible installs.
