# Working on Nimbus

This file is the repository-wide contract for contributors and automated coding
agents. It describes the public project, its invariants and how to make a
reviewable change. It must remain useful on a fresh checkout: do not add a
maintainer's machine names, private paths, account settings, personal prompts,
local campaign status or session handoff notes.

## Project and source map

Nimbus is a C++17 atmospheric simulator with CPU, Vulkan and Metal execution
paths. Nimbus Studio is a separate Tauri application: React/TypeScript owns the
interface and rendering preparation; Rust owns native readers, filesystem and
process operations. The simulator and Studio communicate across explicit
process/output interfaces. Studio rendering is separate from solver compute.

- `src/` contains numerical implementations, physical schemes, orchestration,
  initialization, diagnostics and output. `include/` contains their public and
  shared contracts. Keep module ownership explicit.
- `metal/` and `vulkan/` contain native compute implementations and shader
  assets. Some headers and SPIR-V payloads are generated and have exact source
  or artifact checks; edit their authoritative generator/source first.
- `ui/src/` and `ui/src-tauri/` contain Studio application code. UI tests,
  browser harnesses, synthetic data generators and analysis helpers live in
  `tests/ui/`. Keep test artifacts out of the application package.
- `tests/` contains maintained bounded tests and small essential reference
  inputs. `tests/science/` contains reusable scientific test drivers. A stored
  historical reference is not a receipt qualifying the current executable.
- `configs/` contains public configuration. `scripts/` and `mk/` contain build,
  provisioning, validation and release automation.
- `docs/` contains reader-facing Markdown guides and the mathematics and
  architecture PDFs. Their editable LaTeX sources, figures and source/PDF hash
  manifest live in `scripts/dev/docs/latex/`.

Start with [README.md](README.md), [docs/BUILDING.md](docs/BUILDING.md),
[CONTRIBUTING.md](CONTRIBUTING.md), [the mathematics guide](docs/MATHEMATICS.pdf)
and [the architecture guide](docs/ARCHITECTURE.pdf). Consult
[the namelist](docs/NAMELIST.md) for the complete strict-v1 configuration and
legacy-reader inventory. Read the nearest module
README and the actual implementation before changing a scientific assumption.
If the maintainer provides a local DEVELOPMENT.md, use it for the active
priority and acceptance gates. Keep private work queues untracked and update
the supplied plan in place; do not create competing roadmaps.

## Agent coordination

Use at most two agents in total, including the lead. Do not create nested
sub-agents or a development swarm. One agent works sequentially on the active
target. With two, assign disjoint file ownership and one bounded secondary task;
the lead owns the contract and final integration. "Continue" resumes the active
target unless a confirmed blocker or explicit maintainer decision changes it.

NPY remains supported until its netCDF replacement satisfies Ready and Fully
Qualified for the required replacement scope, including campaigns and independent
scientific review. Bounded netCDF support alone does not authorize NPY removal.

## Working procedure

1. Inspect the working tree, current branch and relevant code before editing.
   Preserve unrelated changes. Never reset, clean, overwrite or rewrite history
   merely to obtain a convenient baseline.
2. Identify the public selector, state owner, consumers and relevant tests for
   the behavior being changed. Follow the ordinary runtime path, not just a
   similarly named helper or a convenient isolated kernel.
3. State consequential assumptions: units, grid locations, physical domain,
   source timing, boundary exchange, precision, backend and failure behavior.
   If an assumption changes, update its mathematical explanation and tests.
4. Make the smallest coherent implementation change. Update all affected
   configuration, catalog, UI, fixture and documentation consumers together.
   A guard is a temporary protection for incomplete work, not its completion.
5. Run bounded checks that exercise the changed behavior and its failure path.
   Test from source without relying on old local binaries, private run data or
   reports. Record skipped requirements and unavailable hardware explicitly.
6. Review the resulting diff and report the behavior, checks and material
   limitations. Distinguish implementation, bounded verification, campaigns and
   independent scientific review using the contract below.

Use a task branch for a reviewable change. Do not include build directories,
compiled executables, runtime table downloads, simulation fields, personal
benchmark baselines, device inventories or dated engineering diaries in Git.
Keep local experiments outside the maintained source tree or in ignored output
locations. Do not invent universal performance defaults from one computer.

## Mathematical and state invariants

Dry density, mixing ratios and particle number units are distinct currencies.
A mixing ratio per kg dry air has inventory `rho_d * physical_volume * q`.
A number per unit volume has a different carrier. Preserve that distinction in
transport, remapping, sources, diagnostics and restart.

Allocated array bounds are not physical atmospheric bounds. Use the physical
cell-domain and metric owners. Halos and boundary-closure planes do not acquire
volume because they contain finite values. C-grid face velocities, collocated
velocities, terrain-normal fluxes and Earth-frame winds are not interchangeable.
Use physical layer thicknesses and interface heights for stretched or mapped
columns; do not reconstruct them from a nominal spacing.

Conservative scalar transport must use the mass carrier and time interval that
its continuity owner declares. Preserve uniform-tracer consistency and shared
interior flux cancellation. Treat limiting, clipping, remapping and storage
rounding as observable operations with appropriate inventory accounting.

Physics coupling is part of the algorithm. Preserve source ordering, shared
reservoir availability, private moments, held source cadences, boundary history
and the equation-of-state projection. A source heat law and an energy ledger
are different contracts; a zero accounting residual alone does not validate a
physical transfer. Document where Nimbus intentionally differs from a paper.

Multi-output operations need a clear validation/commit boundary. A rejected
transaction must not leave primary fields, private state, diagnostics, reports
or device ownership describing different endpoints. Restart must retain the
complete continuation state, not merely the fields needed to draw a frame.

## Backend, architecture and performance discipline

ARM64 and x86-64 are instruction-set targets. Unified and discrete memory are
memory topologies. Do not infer one from the other or assume host-visible GPU
memory has unified-memory behavior. Query device capabilities and preserve the
allocation, transfer, synchronization and ownership contracts they require.

A pointer match is not proof that a buffer contains current data. Respect field
generations, scope lifetimes, dirty state, completion fences and host publication.
An asynchronous copy or submitted kernel is not necessarily complete. Host
consumers, diagnostics, rollback and output must see the required endpoint.

Native-required profiles must reach the intended native implementation without
hidden CPU substitution. Report dispatch/fallback evidence alongside parity
results. A software Vulkan test is useful bounded coverage and is not evidence
of physical discrete-GPU throughput or a universal device capability.

Preserve the declared floating-point contract. Storage precision, intermediate
precision, fused operations, denormal handling and reduction order can differ.
Do not relax tolerances, enable unsafe arithmetic or remove a synchronization
step to improve timing without justified scientific and correctness evidence.
Portable builds avoid host-specific ISA assumptions but do not become universal
binaries or prove compatibility with every operating-system library version.

A performance comparison must identify the executable, source/configuration,
compiler flags, runtime libraries, CPU/GPU, memory topology, worker/affinity
settings, warmup and measured interval. Compare matching scientific endpoints
and use repeated trials. An isolated kernel improvement is not automatically a
solver-step or end-to-end improvement. Profiling and long hardware campaigns
are deliberate work, not a side effect of a documentation or cleanup change.

## Configuration, catalogs and generated identities

Public YAML selection, typed loading, configuration compilation, runtime
admission, scheme descriptors, readiness records and Studio must agree on the
same supported scope. Browser snapshots are generated from the executable;
never hand-edit them to advertise an implementation the runtime does not admit.

Some fixtures bind exact reviewed bytes and some generated manifests bind
source hashes. Determine which kind of identity is involved before changing a
reference. Moving a historical fixture does not authorize changing its payload
or re-signing it as fresh evidence. Prefer an explicit bounded location mapping
where a reviewed historical path must remain part of the original bytes.

After an intentional source-metadata change, inspect the entire dependency
chain: generated manifests/headers, catalog snapshot/reference and TypeScript
and Rust consumers can all pin a digest. Verify that only intended metadata
changed and that configuration membership, thresholds and readiness states did
not silently change. Regenerating a hash is not a scientific review.

Do not mass-format generated headers or raw-source-bound scientific files.
`make format-check` reports changed-line formatting; regeneration and source
identity checks are separate requirements. A formatter/tool failure must fail
the check rather than being mistaken for an empty successful diff.

## Build and bounded verification

Run commands from the repository root unless a command changes directory.
Use [BUILDING.md](docs/BUILDING.md) for platform prerequisites and runtime-table
provisioning; do not encode a contributor's dependency installation paths.
Typical entry points are:

```sh
make -j4 CPU_ONLY=1 PORTABLE=1 all
make test-script-suite
make test-qualification-tools
make check-docs
python3 scripts/dev/docs/build_guides.py --check
npm --prefix ui ci
npm --prefix ui run typecheck
npm --prefix ui run test:unit
npm --prefix ui run typecheck:browser
npm --prefix ui run build
```

These commands are entry points, not a sufficient validation program for every
change. Select the applicable numerical, physics, restart, invalid-input,
threading and native-backend targets from `make help` and `mk/tests.mk`. A
CPU-only build cannot establish native parity. `make test` is broader and may
require provisioned coefficients or device-specific prerequisites. Report what
actually ran, not what a target name suggests should have run.

UI tests create their arrays and temporary artifacts during execution. Keep
maintained test definitions and generators in `tests/ui/`, not recorded personal
datasets in `ui/`. A test that needs an external integration dataset must make
that prerequisite explicit and report an unavailable requirement as such. Test
discovery must fail when an expected suite is accidentally empty.

Rebuild technical documents with `make docs-pdf`. Keep LaTeX inputs, their hash
manifest and published PDFs in agreement, inspect changed equations/figures,
and run the link check. A paper supplies scientific context; describe the
implemented approximation, parameter choices and remaining limitations in
natural language alongside the equations.

## Scope and reporting

Do not claim support for an arbitrary scheme combination merely because each
component exists. Use the ordinary public admission path and maintained
readiness registry. When reporting a change, identify incomplete engineering,
bounded tests, campaign evidence and independent review separately. User-facing
documentation must not convert historical evidence into current qualification.

## Ready

A tracked Nimbus capability or intended capability combination is **ready**
only when all of the following are true:

- The engineering is complete. There are no placeholders, hidden CPU
  substitutions, unfinished coupling paths, deferred state owners, or known
  implementation gaps in the declared scope.
- The capability is covered by bounded tests appropriate to its risks,
  including scientific references, conservation/closure, invalid-input and
  failure-atomicity behavior, restart/continuation, threading determinism, and
  CPU/Vulkan/Metal parity where those backends apply.
- It is selectable through the ordinary public configuration and runtime path.
  Selection must reach the intended implementation without fallback when the
  profile requires native execution.
- The ordinary runtime admission path accepts the supported profile without an
  invalid-science, unavailable, maturity, or development-only flag.
- Documentation, configuration fixtures, capability descriptors, manifests,
  and runtime behavior agree on the exact supported scope.

Fail-closed guards are required while work is incomplete, but a guarded or
runtime-non-selectable implementation is not ready. A source implementation,
focused unit test, or exploratory selector alone is also not ready.

The repository-wide objective is to bring every tracked capability and
intended combination in the maintained readiness matrix to this ready state
before beginning its long-running qualification campaign.

## Fully Qualified

A capability is **fully qualified** only after it is ready and all required
long-running campaigns have completed successfully, their artifacts and
thresholds have been validated, and the applicable scientific review has been
completed. Independent review by qualified domain experts is part of the final validation
program and must be reported separately from automated test success.

Do not describe bounded tests, implementation completion, historical evidence,
or fail-closed behavior as full qualification. Do not describe a capability as
fully qualified while a required long campaign or external scientific review
is still pending.

## Required Reporting Discipline

When reporting status, distinguish these states explicitly:

1. **Engineering incomplete** — implementation or coupling work remains.
2. **Ready** — engineering, bounded testing, public selection, and ordinary
   science admission are complete; long campaigns may begin.
3. **Campaign validated** — required long campaigns passed and evidence was
   checked.
4. **Externally reviewed** — the stated independent scientific review passed.
5. **Fully qualified** — every required ready, campaign, evidence, and review
   condition is satisfied.

Never use “outside scope,” “experimental,” or “fail-closed” to imply that a
tracked readiness item may be ignored. If an item is intentionally removed
from the product rather than completed, that requires an explicit product
decision and corresponding removal from public configuration, catalogs,
documentation, fixtures, and qualification claims.

## Code Comments

Keep comments rare and useful.

Do not add comments that only describe what the code does. Add comments only
when the code cannot fully express the logic, the logic is unusually complex,
or there are historical reasons, invariants, constraints, compatibility
concerns, or known pitfalls that future maintainers need to understand.

Do not add task-specific comments that will be meaningless after review.
