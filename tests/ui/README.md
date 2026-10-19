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

The unit runner discovers `unit/**/test_*.ts` and `release/test_*.ts`; it fails
if discovery is empty. Node.js 22 or newer is required. Browser tests also need
the applicable Playwright browsers. Chromium/SwiftShader, WebKit and native
packaged application checks provide different coverage.

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
