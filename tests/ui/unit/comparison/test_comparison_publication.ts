import { alignDatasets } from "../../../../ui/src/lib/dataset/datasetAlignment.ts";
import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import { createComparisonPublicationWorkspace, parseComparisonPublication, serializeComparisonPublication } from "../../../../ui/src/lib/comparison/comparisonPublication.ts";
import type { GridInfo, StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const grid: GridInfo = { nr: 2, nth: 2, nz: 2, coordinateSystem: "cartesian", geometryKind: "physical", scalarCoordinatesM: [[0, 1], [0, 1], [0, 1]], scalarOriginM: [0, 0, 0] };
const step: StepInfo = { index: 0, simulationTimeS: 0, fields: ["w"], fieldUnits: { w: "m/s" }, grid, publicationState: "complete", publicationGeneration: "fixture" };
const left = createLocalDatasetSource("cm1_netcdf", "/case/cm1out.nc");
const right = createLocalDatasetSource("nimbus_run", "/case/nimbus");
const alignment = alignDatasets(left, [step], right, [step]);
const workspace = createComparisonPublicationWorkspace({ left, right, alignment, recipe: null });
const parsed = parseComparisonPublication(serializeComparisonPublication(workspace));
check(parsed.left.locator.path === "/case/cm1out.nc", "publication round trips source A");
check(parsed.alignment.differenceAllowed, "publication retains alignment evidence");
check(parsed.recipe === null, "publication permits export before a view recipe exists");
let rejected = false;
try { parseComparisonPublication(JSON.stringify({ ...workspace, schemaVersion: 99 })); } catch { rejected = true; }
check(rejected, "unknown publication version fails closed");
rejected = false;
try { parseComparisonPublication("not json"); } catch { rejected = true; }
check(rejected, "malformed publication fails closed");
rejected = false;
try { parseComparisonPublication(JSON.stringify({ ...workspace, alignment: { ...alignment, leftDatasetIdentity: "foreign" } })); } catch { rejected = true; }
check(rejected, "alignment evidence must identify the exported sources");
rejected = false;
try { parseComparisonPublication(JSON.stringify({ ...workspace, recipe: { schema: "nimbus.studio.comparison-workspace", schemaVersion: 4 } })); } catch { rejected = true; }
check(rejected, "malformed view recipe fails closed");
console.log(`comparison publication: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) { failures.forEach((failure) => console.error(`FAIL ${failure}`)); process.exit(1); }
