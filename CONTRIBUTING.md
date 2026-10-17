# Contributing to Nimbus

Build from the current source and keep a change focused on one identifiable
behavior. Agree on priorities and bounded work assignments with the maintainer.
[Architecture](docs/ARCHITECTURE.pdf) explains ownership and code
boundaries; [mathematics](docs/MATHEMATICS.pdf) explains the model assumptions.
The [build guide](docs/BUILDING.md) lists toolchains and runtime data.

## Work on a branch

```sh
git switch -c my-change
make -j4
```

Use `make help` to find maintained targets. The implementation is grouped under
`src/`, public interfaces under `include/`, native backends under `metal/` and
`vulkan/`, and Studio under `ui/`. Source and build recipes must not depend on
objects or results from a previous checkout.

## Verify the behavior you change

```sh
make test-script-suite
make test
cd ui
npm ci
npm run typecheck
npm run test:unit
npm run build
```

The full simulator suite includes physics, configuration, output and backend
checks. Some checks require provisioned scientific data or a physical device;
read their reported requirements and select the relevant backend targets.
Skipping a device-dependent test does not establish native parity. The Studio
unit runner needs Node.js 22 or newer; CI selects Node.js 22. UI tests and their
data generators live under `tests/ui/`; `ui/` contains the application and its
build configuration.

For a numerical change, explain the reference, conservation/closure assumptions,
invalid-input behavior, rollback boundary, restart implications and relevant
CPU/native comparisons. Preserve existing tolerances unless a scientific reason
and independent evidence justify changing them. A performance claim needs
matching scientific endpoints and repeatable measurements of the affected
workload, not just a faster isolated kernel.

The meanings of status labels are deliberate. **Engineering incomplete** means
implementation or coupling remains. **Ready** requires complete engineering,
bounded verification, public selection and ordinary admission for the stated
scope. **Campaign validated** and **externally reviewed** report separate later
steps. **Fully qualified** requires all applicable conditions. A fail-closed guard
is necessary during unfinished work; it does not make that capability ready.

## Keep contracts and documentation together

Changes to runtime admission must preserve agreement between the typed loader,
configuration compiler, catalog, Studio snapshot and maintained fixtures. The
catalog tests reject accidental fingerprint drift. Regenerate a snapshot only
after reviewing the intended semantic change; do not change a reference merely
to silence a failing test.

Update operating instructions and the relevant mathematical or architectural
explanation when behavior changes. Run `make check-docs` before submitting.
Do not add dated development diaries, run receipts or generated output to Git.
Keep essential small test fixtures with their tests and explain their purpose.

## Style and submission

`.clang-format`, `.clang-tidy` and `.editorconfig` contain the maintained style
settings. `make format-check` checks changed C++ lines; `make format` applies that
formatting. `make compile-commands` regenerates the ignored compilation database
for editor tooling.

For an initial commit with no parent, CI checks the whole source with
`make format-check FORMAT_BASE=empty-tree`. Unrelated or missing history must
not silently turn a formatting check into a comparison of the tree with itself.

Open a pull request that describes the problem, resulting behavior and checks
performed. Identify material gaps honestly. Follow the
[Code of Conduct](CODE_OF_CONDUCT.md) and use [SECURITY.md](SECURITY.md) for
vulnerability reporting.
