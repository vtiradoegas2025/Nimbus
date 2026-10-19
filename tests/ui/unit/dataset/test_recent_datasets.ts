import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  RECENT_DATASETS_LIMIT,
  RECENT_DATASETS_STORAGE_KEY,
  clearRecentDatasets,
  createRecentDatasetEntry,
  forgetRecentDataset,
  loadRecentDatasets,
  parseRecentDatasetStore,
  rememberRecentDataset,
  saveRecentDatasets,
  setRecentDatasetPinned,
} from "../../../../ui/src/lib/dataset/recentDatasets.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const at = (minute: number) => `2026-07-19T12:${String(minute).padStart(2, "0")}:00.000Z`;

class MemoryStorage {
  values = new Map<string, string>();
  failWrites = false;
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { if (this.failWrites) throw new Error("quota"); this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

const cm1 = createLocalDatasetSource("cm1_netcdf", ["/cases/alpha/cm1out_000001.nc", "/cases/alpha/cm1out_000002.nc"]);
const nimbus = createLocalDatasetSource("nimbus_run", "/runs/nimbus-alpha");
const cm1Case = {
  ...cm1,
  displayName: "…/cases/alpha · 2 history files",
  caseInventory: {
    schema: "nimbus.studio.cm1-case-inventory" as const,
    schemaVersion: 2 as const,
    directory: "/cases/alpha",
    historyFiles: [...cm1.locator.paths!],
    namelistInput: "/cases/alpha/namelist.input",
    inputSounding: null,
    restartFiles: [],
    statisticsFiles: [],
    otherFileCount: 0,
    nestedDirectoryCount: 0,
  },
};

const fileEntry = createRecentDatasetEntry(cm1, { openedAt: at(1) });
const caseEntry = createRecentDatasetEntry(cm1Case, { openedAt: at(2) });
const nimbusEntry = createRecentDatasetEntry(nimbus, { openedAt: at(3), pinned: true });
check(fileEntry.selection.kind === "cm1_files" && fileEntry.selection.paths.length === 2, "CM1 multi-file selection retains every local reference");
check(caseEntry.selection.kind === "cm1_case_folder" && caseEntry.selection.paths[0] === "/cases/alpha", "CM1 case selection reopens from its folder");
check(caseEntry.source.caseInventory == null, "runtime case inventories are not persisted as recent data");
check(nimbusEntry.selection.kind === "nimbus_folder", "Nimbus runs retain a folder reference");
check(caseEntry.datasetIdentity === fileEntry.datasetIdentity, "recent identity uses the stable dataset source identity");

let entries = rememberRecentDataset([], cm1, { openedAt: at(1) });
entries = rememberRecentDataset(entries, nimbus, { openedAt: at(2), pinned: true });
entries = rememberRecentDataset(entries, cm1, { openedAt: at(3) });
check(entries.length === 2, "reopening a stable identity deduplicates the recent list");
check(entries.find((entry) => entry.source.kind === "cm1_netcdf")?.lastOpenedAt === at(3), "reopen refreshes last-opened time");
check(entries[0].pinned && entries[0].source.kind === "nimbus_run", "pins sort before unpinned recency");

entries = setRecentDatasetPinned(entries, fileEntry.datasetIdentity, true);
check(entries.every((entry) => entry.pinned), "pin state can be changed without rebuilding a source");
entries = rememberRecentDataset(entries, cm1, { openedAt: at(4) });
check(entries.find((entry) => entry.datasetIdentity === fileEntry.datasetIdentity)?.pinned === true, "normal reopen preserves an existing pin");
entries = forgetRecentDataset(entries, nimbusEntry.datasetIdentity);
check(entries.length === 1 && entries[0].source.kind === "cm1_netcdf", "forget removes only the requested identity");

let boundedEntries = [] as typeof entries;
for (let index = 0; index < RECENT_DATASETS_LIMIT + 5; index += 1) {
  boundedEntries = rememberRecentDataset(
    boundedEntries,
    createLocalDatasetSource("nimbus_run", `/runs/${index}`),
    { openedAt: new Date(Date.UTC(2026, 6, 19, 13, index)).toISOString() },
  );
}
check(boundedEntries.length === RECENT_DATASETS_LIMIT, "recent datasets are hard-bounded");
check(boundedEntries[0].source.locator.path === "/runs/16", "newest dataset sorts first");
check(!boundedEntries.some((entry) => entry.source.locator.path === "/runs/0"), "bound evicts the oldest unpinned reference");

const storage = new MemoryStorage();
check(saveRecentDatasets(storage, [caseEntry, nimbusEntry], at(5)), "valid recent references persist");
const restored = loadRecentDatasets(storage);
check(restored.length === 2 && restored[0].pinned, "load restores and orders valid references");
check(restored[1].selection.kind === "cm1_case_folder", "case-folder reopen intent survives storage");

const saved = JSON.parse(storage.values.get(RECENT_DATASETS_STORAGE_KEY)!);
check(saved.entries[1].source.caseInventory == null, "storage contains no CM1 case inspection payload");
check(parseRecentDatasetStore({ ...saved, schemaVersion: 99 }) === null, "unsupported store versions fail closed");
check(parseRecentDatasetStore({ ...saved, entries: [{ ...saved.entries[0], pinned: "yes" }] }) === null, "malformed entry fields fail closed");
check(parseRecentDatasetStore({ ...saved, entries: [{ ...saved.entries[0], datasetIdentity: "forged" }] }) === null, "identity mismatch fails closed");
check(parseRecentDatasetStore({ ...saved, entries: [{ ...saved.entries[0], selection: { kind: "remote_url", paths: ["https://example.test"] } }] }) === null, "unsupported reopen kinds fail closed");

storage.values.set(RECENT_DATASETS_STORAGE_KEY, "not json");
check(loadRecentDatasets(storage).length === 0, "malformed JSON loads as an empty fail-closed list");
storage.failWrites = true;
check(!saveRecentDatasets(storage, [fileEntry], at(6)), "blocked or quota-limited storage is reported without throwing");
storage.failWrites = false;
check(clearRecentDatasets(storage) && loadRecentDatasets(storage).length === 0, "clear forgets all stored references");

let invalidTimestampRejected = false;
try { createRecentDatasetEntry(cm1, { openedAt: "yesterday" }); } catch { invalidTimestampRejected = true; }
check(invalidTimestampRejected, "non-contract timestamps are rejected");

console.log(`recent datasets: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
