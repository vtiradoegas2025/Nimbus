import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  createInvestigationArchive,
  createInvestigationPublication,
  parseInvestigationArchive,
  parseInvestigationPublication,
  serializeInvestigationPublication,
  sha256Hex,
} from "../../../../ui/src/lib/studio/investigationPublication.ts";
import type { InvestigationWorkspaceRecipe } from "../../../../ui/src/lib/studio/investigationWorkspace.ts";
import { createStoreZip, readStoreZip } from "../../../../ui/src/lib/platform/storeZip.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0; const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const rejects = async (action: () => unknown | Promise<unknown>, label: string) => { try { await action(); failures.push(label); } catch { passed += 1; } };

const text = new TextEncoder();
const smallZip = createStoreZip([{ name: "a.txt", data: text.encode("alpha") }, { name: "nested/b.txt", data: text.encode("beta") }]);
const unpacked = readStoreZip(smallZip);
check(new TextDecoder().decode(unpacked.get("a.txt")) === "alpha" && unpacked.size === 2, "store ZIP round trip");
await rejects(() => createStoreZip([{ name: "../escape", data: text.encode("no") }]), "unsafe ZIP path rejected");

const dataset = createLocalDatasetSource("cm1_netcdf", "/case/cm1out.nc");
const identity = JSON.stringify([dataset.schema, dataset.schemaVersion, dataset.kind, dataset.locator.kind, [dataset.locator.path]]);
const recipe: InvestigationWorkspaceRecipe = {
  schema: "nimbus.studio.investigation-workspace", schemaVersion: 1, datasetIdentity: identity,
  stepIndex: 4, fieldName: "w", zIndex: 2, start: { xKm: 1, yKm: 2 }, end: { xKm: 5, yKm: 6 },
  probeFraction: 0.4, annotations: [], savedAt: new Date().toISOString(),
};
const step: StepInfo = {
  index: 4, simulationTimeS: 120, fields: ["w"], fieldUnits: { w: "m/s" },
  publicationState: "complete", publicationGeneration: "netcdf:/case/cm1out.nc",
  grid: { nr: 10, nth: 10, nz: 5, coordinateSystem: "cartesian", geometryKind: "physical", scalarCoordinatesM: [[0, 1000], [0, 1000], [0, 500]] },
};
const png = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3, 4]);
const publication = createInvestigationPublication({
  title: "WK supercell transect", dataset, recipe, step, unit: "m/s",
  figure: { filename: "figure.png", mediaType: "image/png", width: 1520, height: 1004, byteLength: png.length, sha256: await sha256Hex(png), panels: ["plan", "vertical_transect", "vertical_profile"] },
});
check(parseInvestigationPublication(serializeInvestigationPublication(publication)).recipe.stepIndex === 4, "publication round trip");
check(publication.provenance.replay.headlessBatchRendering === false, "batch boundary disclosed");
const archive = createInvestigationArchive(publication, png);
const restored = await parseInvestigationArchive(archive);
check(restored.publication.datasetIdentity === identity && restored.png.length === png.length, "archive validates pixels and source");
check(readStoreZip(archive).size === 3, "archive has exact bounded contents");
const tampered = archive.slice(); tampered[35] ^= 1;
await rejects(() => parseInvestigationArchive(tampered), "tampered archive rejected");
await rejects(() => parseInvestigationPublication(serializeInvestigationPublication({ ...publication, datasetIdentity: "foreign" })), "foreign dataset identity rejected");
await rejects(() => createInvestigationPublication({ title: "bad", dataset, recipe: { ...recipe, stepIndex: 5 }, step, unit: "m/s", figure: publication.figure }), "frame mismatch rejected");

console.log(`investigation publication: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) { failures.forEach((failure) => console.error(`FAIL ${failure}`)); process.exit(1); }
