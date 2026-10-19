import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  clearComparisonSession,
  createComparisonSession,
  loadComparisonSession,
  parseComparisonSession,
  saveComparisonSession,
} from "../../../../ui/src/lib/comparison/comparisonSession.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

class MemoryStorage {
  private values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

const left = createLocalDatasetSource("cm1_netcdf", "/case/cm1out.nc");
const right = createLocalDatasetSource("nimbus_run", "/case/nimbus");
const session = createComparisonSession(left, right);
const storage = new MemoryStorage();

saveComparisonSession(storage, session);
const restored = loadComparisonSession(storage);
check(restored?.left.locator.path === "/case/cm1out.nc", "saved CM1 source reopens");
check(restored?.right.locator.path === "/case/nimbus", "saved Nimbus source reopens");
check(parseComparisonSession({ ...session, schemaVersion: 99 }) === null, "unknown session version fails closed");
check(parseComparisonSession({ ...session, left: { ...left, readOnly: false } }) === null, "mutable source record fails closed");
check(parseComparisonSession({ ...session, left: { ...left, sourceModel: "nimbus" } }) === null, "inconsistent source identity fails closed");

storage.setItem("nimbus:comparison-session:v1", "not json");
check(loadComparisonSession(storage) === null, "malformed storage fails closed");
saveComparisonSession(storage, session);
clearComparisonSession(storage);
check(loadComparisonSession(storage) === null, "forget removes the saved pair");

console.log(`comparison session: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
