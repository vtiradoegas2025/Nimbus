# Studio verification

This directory contains Studio's maintained tests, browser harnesses, synthetic
data generators and validation helpers. The application and its build scripts
remain under `ui/`. Do not commit recorded personal datasets, generated field
arrays, browser captures or test-result directories here.

From the repository root:

```sh
npm --prefix ui ci
npm --prefix ui run test:unit
npm --prefix ui run typecheck:browser
npm --prefix ui run build:browser-test
npm --prefix ui run test:browser
```

CI distributes adjacent rendering cases across twelve partitions per browser,
with one browser worker per job. The Linux Chromium jobs use ANGLE over Mesa's
lavapipe software Vulkan driver. The profile verifies the actual WebGL2 renderer
and a rendered pixel before running tests, and retains that identity in the test
results. Install `mesa-vulkan-drivers`, select its `lvp_icd*.json` file through
`VK_ICD_FILENAMES`, set `LP_NUM_THREADS=2` and `NIMBUS_BROWSER_LAVAPIPE=1`, then run
`node tests/ui/run_browser_partition.mjs chromium-lavapipe 1/12` from the root.
Use `webkit-standard` or `webkit-retina` for the other CI profiles. Append
`--list-only` to inspect its inventory without running a browser. The runner
verifies the selected tests through Playwright before execution and writes the
inventory under `ui/build/browser-partitions/` alongside the usual test reports.

The unit runner discovers `unit/**/test_*.ts` and `release/test_*.ts`; it fails
if discovery is empty. Node.js 22 or newer is required. Browser tests also need
the applicable Playwright browsers. Local Chromium commands default to
`chromium-swiftshader` when `NIMBUS_BROWSER_LAVAPIPE` is unset. It remains useful
for reproducing driver-specific behavior, but full-resolution, shadowed volume
scenes can exceed its functional time limits even when the same scene completes
with Mesa or physical GPUs. Switching CI drivers does not change the test data,
viewport, sampling, assertions or interaction limits. Software-renderer results
do not establish physical GPU throughput. WebKit and native packaged application
checks provide additional coverage.

`prepare.mjs` connects this test workspace to the dependency installation in
`ui/node_modules` using an ignored directory link (a junction on Windows).
It does not create a second dependency installation or overwrite an unrelated
existing directory. The npm test entry points and browser configuration prepare
that workspace automatically.

Browser support modules construct analytic fields and fake native responses
in memory. The shared NPY generator under `data/` constructs the same validation
cases for TypeScript and Rust; serialized corpus files are not checked in.
Browser Python runtime files and verified package wheels are generated under
`ui/build/test-public` and `ui/build/pyodide-cache`.

Optional preserved-run comparisons require `NIMBUS_BEAST_XL_REFERENCE_RUN` and,
for reference-image comparison, `NIMBUS_BEAST_XL_REFERENCE_MANIFEST`. These are
explicit external inputs for that test case. The ordinary synthetic suite does
not require a maintainer's run directory or an archived reference image.
Missing external integrations are reported separately from passing tests.

Packaged visualization checks use `qualify_packaged_visualization.sh` or its
PowerShell counterpart through the npm scripts in `ui/package.json`. Their
outputs belong under ignored build/report directories. A test result or stable
image does not establish scientific qualification of the underlying simulation.
