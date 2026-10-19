import { loadInvestigationWorkspace, saveInvestigationWorkspace, type InvestigationWorkspaceRecipe } from "../../../../ui/src/lib/studio/investigationWorkspace.ts";
import { buildVerticalTransect, sampleHorizontalPoint } from "../../../../ui/src/lib/plot/transectSampling.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0; const failures: string[] = []; const check = (value: boolean, label: string) => value ? passed += 1 : failures.push(label);
const field: Npy3D = { shape: [3, 3, 2], data: Float32Array.from(Array.from({ length: 18 }, (_, index) => index)) };
const grid: GridInfo = { nr: 3, nth: 3, nz: 2, coordinateSystem: "cartesian", geometryKind: "physical", scalarCoordinatesM: [[0, 1000, 2000], [0, 1000, 2000], [0, 500]], scalarOriginM: [0, 0, 0] };
const transect = buildVerticalTransect(field, grid, { xKm: 0, yKm: 0 }, { xKm: 2, yKm: 2 }, 3);
check(transect.sampleCount === 3 && Math.abs(transect.distanceKm[2] - Math.sqrt(8)) < 1e-6, "physical distance");
check(transect.values[0][1] === 8 && transect.values[1][1] === 9, "bilinear diagonal sampling");
check(sampleHorizontalPoint(field, grid, { xKm: 0.5, yKm: 0.5 }, 0) === 4, "physical point sampling");

class MemoryStorage {
  value: string | null = null;
  failWrites = false;
  getItem() { return this.value; }
  setItem(_key: string, value: string) { if (this.failWrites) throw new Error("quota"); this.value = value; }
}
const storage = new MemoryStorage();
const recipe: InvestigationWorkspaceRecipe = { schema: "nimbus.studio.investigation-workspace", schemaVersion: 1, datasetIdentity: "dataset", stepIndex: 2, fieldName: "w", zIndex: 1, start: { xKm: 0, yKm: 0 }, end: { xKm: 2, yKm: 2 }, probeFraction: .5, annotations: [], savedAt: new Date().toISOString() };
check(saveInvestigationWorkspace(storage, recipe), "successful workspace persistence is reported");
check(loadInvestigationWorkspace(storage, "dataset")?.fieldName === "w", "workspace round trip");
check(loadInvestigationWorkspace(storage, "foreign") === null, "foreign dataset fails closed");
storage.value = JSON.stringify({ ...recipe, probeFraction: 2 });
check(loadInvestigationWorkspace(storage, "dataset") === null, "invalid probe fails closed");
storage.failWrites = true;
check(!saveInvestigationWorkspace(storage, recipe), "blocked workspace persistence is reported without throwing");
console.log(`investigation workspace: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) { failures.forEach((failure) => console.error(`FAIL ${failure}`)); process.exit(1); }
