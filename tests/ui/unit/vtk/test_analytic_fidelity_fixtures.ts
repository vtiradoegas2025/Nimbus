import {
  ANALYTIC_FIDELITY_FIELD_NAMES,
  createAnalyticFidelityFixture,
  isAnalyticFidelityFieldName,
} from "../../../../ui/src/lib/vtk/analyticFidelityFixtures.ts";
import {
  defaultVolumeColormap,
  volumeOpacityClass,
} from "../../../../ui/src/lib/plot/volumeTransferPolicy.ts";

let passed = 0;

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
  passed += 1;
}

const shape: [number, number, number] = [33, 29, 21];
const voxels = shape[0] * shape[1] * shape[2];
const fixtures = Object.fromEntries(ANALYTIC_FIDELITY_FIELD_NAMES.map((name) => [
  name,
  createAnalyticFidelityFixture(name, shape),
]));

for (const name of ANALYTIC_FIDELITY_FIELD_NAMES) {
  const fixture = fixtures[name];
  check(fixture.field.data.length === voxels, `${name} fills the requested lattice exactly`);
  check(fixture.expected.min <= fixture.expected.max, `${name} has a finite ordered range`);
  check(isAnalyticFidelityFieldName(name), `${name} is recognized by the fixture loader`);
}

const gradient = fixtures.analytic_smooth_gradient;
check(gradient.expected.min === 0 && gradient.expected.max === 1, "smooth gradient spans its exact unit range");
check(gradient.expected.horizontallyUniform, "smooth gradient has no horizontal structure to masquerade as banding");

const sheet = fixtures.analytic_thin_sheet;
check(sheet.expected.nonzeroCount === shape[0] * shape[1], "thin sheet is exactly one cell deep");
check(sheet.expected.max === 1, "thin sheet preserves its source maximum");

const tube = fixtures.analytic_helical_tube;
check(tube.expected.max > 0.9, "helical tube has a resolvable continuous core");
check(!tube.expected.horizontallyUniform, "helical tube carries curved horizontal structure");

const couplet = fixtures.analytic_signed_couplet;
check(couplet.expected.signChanges, "signed couplet contains both signs");
check(Math.abs(couplet.expected.max + couplet.expected.min) < 1e-5, "signed couplet is symmetric about zero");

const stratified = fixtures.analytic_stratified_anomaly;
check(stratified.expected.min >= 0.34999, "stratified fixture retains its physical background");
check(stratified.expected.max > 0.9, "stratified fixture contains a feature above the background profile");

check(volumeOpacityClass("analytic_thin_sheet") === "sparse-positive", "sheet uses feature-preserving opacity");
check(volumeOpacityClass("analytic_helical_tube") === "sparse-positive", "tube uses feature-preserving opacity");
check(volumeOpacityClass("analytic_signed_couplet") === "signed-perturbation", "couplet uses signed opacity");
check(volumeOpacityClass("analytic_smooth_gradient") === "background-stratified", "gradient uses background-field opacity");
check(volumeOpacityClass("analytic_stratified_anomaly") === "background-stratified", "stratified anomaly uses background-field opacity");
check(defaultVolumeColormap("analytic_signed_couplet") === "Cool to Warm", "couplet defaults to a diverging palette");

console.log(`analytic fidelity fixtures: ${passed} checks passed`);
