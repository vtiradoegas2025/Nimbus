# Nimbus documentation

Start with [building and running Nimbus](BUILDING.md). The
[configuration guide](../configs/README.md) explains how to choose an example
and validate it, and the [Studio guide](STUDIO.pdf) explains the desktop and
remote-compute workflows, plots, analysis and publication.
The [complete namelist reference](NAMELIST.md) lists configuration keys,
omission defaults, units, aliases and current restrictions, including unfinished
options and legacy-reader compatibility.

The [mathematics guide](MATHEMATICS.pdf) explains the equations and assumptions
behind the model, including physical cell ownership, staggering, terrain metrics,
transport, initial conditions, radiation, microphysics, mixing, diagnostics and
a complete coupled timestep. It also connects effective resolution, timestep
cost, refinement studies and ensemble design to high-fidelity local experiments.
The [architecture guide](ARCHITECTURE.pdf) explains how configuration, state,
compute backends, CPU instruction sets, memory topology, compression, native
ownership and GPU rendering fit together. It explains how to decide between
local and larger-scale execution, and what a future multi-GPU solver would
require; distributed solver execution is a design proposal, not a shipped feature.
It also distinguishes the existing Vulkan PTX retained-flux assist from a
proposed Vulkan/CUDA runtime hybrid, including memory sharing, synchronization,
scientific parity and performance gates.
The [Studio guide](STUDIO.pdf) follows data from native readers through workers,
Plotly, VTK and graphics shaders to the displayed and captured frame. It covers
all plot families, the research and Python workspaces, sampling and coordinate
assumptions, worker measurement, I/O budgets, memory ownership and recovery.
The mathematics guide also covers
moving frames, terrain and nesting. Read [scientific references](foundationalScience.md)
for the supporting literature.

[Supported platforms](SUPPORTED_PLATFORMS.md) describes the maintained build
routes and evidence limits.

Contributors should read [CONTRIBUTING.md](../CONTRIBUTING.md), use the
[script index](../scripts/README.md), and keep the relevant code, tests and
operating documentation consistent.
Historical task handoffs, development
journals, campaign reports and one-off build records are maintained outside
this source repository. The executable catalog and retained tests describe the
current implementation; old reports are not a substitute for current evidence.

The published PDF manuals are available here. The maintainer keeps their
editable sources and build tooling locally; those files are not distributed
in the GitHub repository. Report corrections against the relevant manual and
section so the published explanation can stay aligned with the implementation.


Both Architecture and Mathematics PDF's were orginal source material from my 
journals and notes as well as lessons learned over the years working on Nimbus. 
The docs here are for repository documenting purposes. 
Original Nimbus engineering start date: March 2024
