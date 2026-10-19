import {
  COMPARISON_WORKSPACE_SCHEMA,
  COMPARISON_WORKSPACE_VERSION,
  comparisonWorkspaceKey,
  loadComparisonWorkspace,
  saveComparisonWorkspace,
  type ComparisonWorkspaceRecipe,
} from "../../../../ui/src/lib/comparison/comparisonWorkspace.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

class MemoryStorage {
  values = new Map<string, string>();
  failWrites = false;
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void { if (this.failWrites) throw new Error("quota"); this.values.set(key, value); }
}

const storage = new MemoryStorage();
const recipe: ComparisonWorkspaceRecipe = {
  schema: COMPARISON_WORKSPACE_SCHEMA,
  schemaVersion: COMPARISON_WORKSPACE_VERSION,
  leftIdentity: "cm1:/case/cm1out.nc",
  rightIdentity: "nimbus:/case/control",
  fieldName: "reflectivity_dbz",
  sliceAxis: 2,
  sliceIndex: 4,
  rangeMode: "dBZ",
  colormap: "dBZ",
  metricScope: "volume",
  showMetricSeries: true,
  showVectorComparison: true,
  showVolumeComparison: true,
  showCompositeComparison: true,
  viewportBounds: { xMin: 1, xMax: 5, yMin: 2, yMax: 6 },
  savedAt: "2026-07-19T12:00:00.000Z",
};

check(saveComparisonWorkspace(storage, recipe), "successful workspace persistence is reported");
check(loadComparisonWorkspace(storage, recipe.leftIdentity, recipe.rightIdentity)?.sliceIndex === 4, "versioned recipe round trips");
check(loadComparisonWorkspace(storage, recipe.leftIdentity, recipe.rightIdentity)?.viewportBounds?.xMax === 5, "physical viewport round trips");
check(loadComparisonWorkspace(storage, recipe.rightIdentity, recipe.leftIdentity) == null, "A/B order is part of workspace identity");
check(comparisonWorkspaceKey(recipe.leftIdentity, recipe.rightIdentity) === comparisonWorkspaceKey(recipe.leftIdentity, recipe.rightIdentity), "workspace key is deterministic");

storage.setItem(comparisonWorkspaceKey(recipe.leftIdentity, recipe.rightIdentity), JSON.stringify({ ...recipe, schemaVersion: 99 }));
check(loadComparisonWorkspace(storage, recipe.leftIdentity, recipe.rightIdentity) == null, "unknown workspace version fails closed");
storage.setItem(comparisonWorkspaceKey(recipe.leftIdentity, recipe.rightIdentity), "not json");
check(loadComparisonWorkspace(storage, recipe.leftIdentity, recipe.rightIdentity) == null, "malformed workspace fails closed");
storage.setItem(comparisonWorkspaceKey(recipe.leftIdentity, recipe.rightIdentity), JSON.stringify({ ...recipe, viewportBounds: { xMin: 2, xMax: 1, yMin: 0, yMax: 1 } }));
check(loadComparisonWorkspace(storage, recipe.leftIdentity, recipe.rightIdentity) == null, "reversed viewport bounds fail closed");
storage.failWrites = true;
check(!saveComparisonWorkspace(storage, recipe), "blocked workspace persistence is reported without throwing");

console.log(`comparison workspace: ${passed} checks passed, ${failures.length} failed`);
if (failures.length > 0) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
