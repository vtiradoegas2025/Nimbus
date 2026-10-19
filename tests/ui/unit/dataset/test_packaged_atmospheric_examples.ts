import { listDatasetFrames } from "../../../../ui/src/lib/dataset/datasetRuntime.ts";
import {
  PACKAGED_ATMOSPHERIC_EXAMPLES,
  createPackagedAtmosphericExampleSource,
  isPackagedAtmosphericExampleUri,
  listPackagedAtmosphericExampleSteps,
  packagedAtmosphericExample,
  readPackagedAtmosphericExampleField,
  reducePackagedAtmosphericExampleField,
  reducePackagedAtmosphericExampleFieldAxis,
} from "../../../../ui/src/lib/dataset/packagedAtmosphericExamples.ts";
import { readFieldAny, readFieldWindows, reduceField, reduceFieldAxis, type GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";
import { aggregateDerivedWindow } from "../../../../ui/src/lib/run/windowedField.ts";
import { gridForWindowedField } from "../../../../ui/src/lib/science/windowedGrid.ts";
import { axisCoords } from "../../../../ui/src/lib/science/gridCoords.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const throws = (action: () => unknown, label: string) => {
  try { action(); failures.push(label); } catch { passed += 1; }
};

const manifest = PACKAGED_ATMOSPHERIC_EXAMPLES[0];
check(manifest.schemaVersion === 1 && manifest.provenance.externalData === false, "example publishes versioned provenance");
check(manifest.provenance.license === "MIT" && manifest.provenance.scientificUseNotice.includes("not a numerical weather simulation"), "license and scientific limitation are explicit");
check(packagedAtmosphericExample(manifest.uri)?.id === manifest.id, "manifest resolves from its stable package URI");
check(isPackagedAtmosphericExampleUri(manifest.uri), "package URI is distinguishable from user filesystem paths");

const source = createPackagedAtmosphericExampleSource(manifest.id);
check(source.kind === "nimbus_run" && source.readOnly && source.locator.path === manifest.uri, "example creates an immutable analysis source without a fake local path");

const steps = listPackagedAtmosphericExampleSteps(manifest.uri);
check(steps.length === 8 && steps.every(({ publicationState }) => publicationState === "complete"), "all packaged frames publish complete immutable identities");
check(steps[0].grid?.geometryKind === "physical" && steps[0].grid?.physicalBoundsM?.upper[2] === 10_000, "example declares a physical Cartesian grid");
check(steps[0].fields.includes("reflectivity_dbz") && steps[0].fieldUnits.vorticity_z === "s-1", "recognizable atmospheric fields carry units");

const generation = steps[3].publicationGeneration;
const reflectivity = readPackagedAtmosphericExampleField(manifest.uri, 3, "reflectivity_dbz", generation);
check(reflectivity.shape.join("×") === "36×32×20" && reflectivity.data.length === 23_040, "full manufactured volume has its declared shape");
check(reflectivity.data.every(Number.isFinite), "manufactured field contains only finite values");
check(Math.max(...reflectivity.data) > 55 && Math.min(...reflectivity.data) >= -10, "reflectivity values occupy a recognizable meteorological range");

const window = readPackagedAtmosphericExampleField(manifest.uri, 3, "reflectivity_dbz", generation, {
  origin: [2, 3, 4], extent: [7, 9, 7], stride: [2, 2, 3], maxOutputBytes: 4 * 4 * 5 * 3,
});
check(window.shape.join("×") === "4×5×3" && window.data.length === 60, "bounded stride window generates only the requested output lattice");
let firstBlockSum = 0;
for (let x = 2; x < 4; x += 1) for (let y = 3; y < 5; y += 1) for (let z = 4; z < 7; z += 1) {
  firstBlockSum += reflectivity.data[((x * 32) + y) * 20 + z]!;
}
check(Math.abs(window.data[0]! - firstBlockSum / 12) < 1e-5, "window values are uniform-cell block means of the full C-order volume");
throws(() => readPackagedAtmosphericExampleField(manifest.uri, 3, "w", generation, {
  origin: [0, 0, 0], extent: [10, 10, 10], stride: [1, 1, 1], maxOutputBytes: 100,
}), "field windows fail closed when the output exceeds the caller budget");
throws(() => readPackagedAtmosphericExampleField(manifest.uri, 3, "w", "stale-generation"), "stale project generation is rejected");
throws(() => readPackagedAtmosphericExampleField(manifest.uri, 3, "not_a_field", generation), "unknown fields are rejected");

const reduction = reducePackagedAtmosphericExampleField(manifest.uri, 3, "w", generation, 10);
check(reduction.finite === reduction.total && reduction.max > 30 && reduction.min < -5 && reduction.countAbove > 0, "full-field reductions work without native IPC");
const verticalMax = reducePackagedAtmosphericExampleFieldAxis(manifest.uri, 3, "w", 2, "max", generation);
check(verticalMax.length === 20 && verticalMax.every(Number.isFinite), "axis reductions support packaged time-height workflows");

const runtimeSteps = await listDatasetFrames(source);
check(runtimeSteps.length === 8 && runtimeSteps[0].fields.length >= steps[0].fields.length, "normal dataset runtime opens the example without desktop or simulator APIs");
const runtimeField = await readFieldAny(manifest.uri, 3, "theta_prime", generation);
check(runtimeField.shape.join("×") === "36×32×20" && runtimeField.data.some((value) => value < -5), "normal field bridge reads packaged data outside Tauri");
const runtimeReduction = await reduceField(manifest.uri, 3, "vorticity_z", generation, 0.01);
check(runtimeReduction.max > 0.015 && runtimeReduction.countAbove > 0, "normal reduction bridge stays simulator-free for packaged fields");
const runtimeProfile = await reduceFieldAxis(manifest.uri, 3, "reflectivity_dbz", 2, "max", generation);
check(runtimeProfile.length === 20 && runtimeProfile[2]! > 40, "normal axis-reduction bridge supports packaged vertical profiles");

const selection = { origin: [2, 3, 4], extent: [7, 9, 7], stride: [2, 2, 3] };
const rawTile = readPackagedAtmosphericExampleField(manifest.uri, 3, "reflectivity_dbz", generation,
  { ...selection, stride: [1, 1, 1] });
check(rawTile.windowGeometry?.sourceShape?.join() === "36,32,20", "bounded reads retain the source shape for derived diagnostics");
check(rawTile.windowGeometry?.origin?.join() === "2,3,4", "bounded reads retain their nonzero source origin");
for (const aggregation of ["cell_mean", "linear_reflectivity_mean", "block_max", "block_min", "magnitude_extreme", "point_sample"] as const) {
  const request = { ...selection, aggregation };
  for (const fieldName of ["reflectivity_dbz", "w"]) {
    const raw = fieldName === "reflectivity_dbz" ? rawTile
      : readPackagedAtmosphericExampleField(manifest.uri, 3, fieldName, generation,
          { ...selection, stride: [1, 1, 1] });
    const expected = aggregateDerivedWindow(raw, request, [36, 32, 20]);
    const actual = readPackagedAtmosphericExampleField(manifest.uri, 3, fieldName, generation, request);
    check(actual.data.every((value, index) => value === expected.data[index]),
      `${fieldName} ${aggregation} matches the independent tile reducer including partial final blocks`);
    check(actual.windowGeometry?.aggregation === aggregation
      && actual.windowGeometry.sampleLocation === (aggregation === "point_sample" ? "source_cell_center" : "block_center"),
      `${fieldName} ${aggregation} declares its sampling geometry`);
  }
}
const reducedGrid = gridForWindowedField<GridInfo>(window, manifest.grid)!;
check(reducedGrid.scalarCoordinatesM?.[0].join() === "1500,2500,3500,4250",
  "cell-mean coordinates include the shortened final x block");
check(reducedGrid.scalarCoordinatesM?.[2].join() === "2750,4250,5250",
  "cell-mean vertical coordinates use actual block centres");
const sampled = readPackagedAtmosphericExampleField(manifest.uri, 3, "w", generation,
  { ...selection, aggregation: "point_sample" });
const sampledGrid = gridForWindowedField<GridInfo>(sampled, manifest.grid)!;
check(sampledGrid.scalarCoordinatesM?.[0].join() === "1250,2250,3250,4250",
  "point-sampled x coordinates remain at source cell centres");
check(sampledGrid.scalarCoordinatesM?.[2].join() === "2250,3750,5250",
  "point-sampled z coordinates do not shift to block centres");
const stretchedGrid = gridForWindowedField(sampled, {
  ...manifest.grid,
  scalarCoordinatesM: [
    Array.from({ length: 36 }, (_, index) => index * index * 100),
    Array.from({ length: 32 }, (_, index) => 250 + index * 500),
    Array.from({ length: 20 }, (_, index) => 250 + index * 500),
  ] as [number[], number[], number[]],
})!;
check(stretchedGrid.scalarCoordinatesM[0].join() === "400,1600,3600,6400",
  "point sampling retains exact nonuniform source coordinates");
throws(() => readPackagedAtmosphericExampleField(manifest.uri, 3, "w", generation,
  { ...selection, aggregation: "unsupported" as never }), "unknown aggregation modes are rejected");
throws(() => readPackagedAtmosphericExampleField(manifest.uri, 3, "w", generation,
  { ...selection, maxSourceBytes: 4 }), "source budgets remain enforced for bounded reads");

const columnWindow = [{ origin: [10, 5, 0], extent: [1, 1, 20], stride: [1, 1, 1], aggregation: "cell_mean" as const }];
const column = await readFieldWindows(manifest.uri, 3, "u", generation, columnWindow);
const columnGrid = gridForWindowedField(column, manifest.grid)!;
check(axisCoords(0, 1, columnGrid).values?.[0] === 5.25
  && axisCoords(1, 1, columnGrid).values?.[0] === 2.75,
  "ordinary column reads report their selected physical location");
const temperatureColumn = await readFieldWindows(manifest.uri, 3, "temperature", generation, columnWindow);
const temperature = await readFieldAny(manifest.uri, 3, "temperature", generation);
check(temperatureColumn.data.every((value, index) => value === temperature.data[(10 * 32 + 5) * 20 + index]),
  "Skew-T's bounded derived temperature matches its full-field column exactly");
const derivedTile = await readFieldWindows(manifest.uri, 3, "temperature", generation,
  [{ ...selection, stride: [1, 1, 1], aggregation: "cell_mean" }]);
const reducedTemperature = await readFieldWindows(manifest.uri, 3, "temperature", generation,
  [{ ...selection, aggregation: "cell_mean" }]);
const expectedTemperature = aggregateDerivedWindow(derivedTile, { ...selection, aggregation: "cell_mean" }, [36, 32, 20]);
check(reducedTemperature.data.every((value, index) => value === expectedTemperature.data[index]),
  "nonlinear temperature is derived at source resolution before reduction");

console.log(`packaged atmospheric examples: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
