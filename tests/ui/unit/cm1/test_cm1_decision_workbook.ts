import type { Cm1CompatibilityReport } from "../../../../ui/src/lib/cm1/cm1Compatibility.ts";
import {
  CM1_DECISION_WORKBOOK_MAX_JSON_BYTES,
  CM1_DECISION_NOTE_MAX_CHARS,
  clearCm1MappingDecision,
  cm1DecisionOptions,
  cm1DecisionProgress,
  createCm1DecisionWorkbook,
  decideCm1MappingItem,
  importCm1DecisionWorkbook,
  loadCm1DecisionWorkbook,
  parseCm1DecisionWorkbook,
  saveCm1DecisionWorkbook,
  serializeCm1DecisionWorkbook,
  validateCm1ManualValue,
} from "../../../../ui/src/lib/cm1/cm1DecisionWorkbook.ts";
import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const report: Cm1CompatibilityReport = {
  schema: "nimbus.studio.cm1-compatibility-report",
  schemaVersion: 1,
  entries: [
    { id: "param1.nx#1", group: "param1", key: "nx", rawValue: "64", status: "ready", target: "grid.nx", reason: "same scalar" },
    { id: "param1.ztop#1", group: "param1", key: "ztop", rawValue: "18000", status: "needs_decision", target: "grid.ztop", reason: "stretching requires review" },
    { id: "param1.cm1setup#1", group: "param1", key: "cm1setup", rawValue: "4", status: "needs_decision", target: null, reason: "compound control" },
    { id: "param2.irdamp#1", group: "param2", key: "irdamp", rawValue: "1", status: "different", target: "damping.enabled", reason: "different formulation" },
    { id: "param0.irst#1", group: "param0", key: "irst", rawValue: "1", status: "unsupported", target: null, reason: "CM1 restarts are unsupported" },
  ],
  assetChecks: [],
  counts: { ready: 1, needs_decision: 2, different: 1, unsupported: 1 },
  draftEligible: false,
  draftBlockers: ["review required"],
};
const sourceA = createLocalDatasetSource("cm1_netcdf", "/case-a/cm1out.nc");
const sourceB = createLocalDatasetSource("cm1_netcdf", "/case-b/cm1out.nc");
const workbook = createCm1DecisionWorkbook(sourceA, report, "2026-07-20T00:00:00.000Z");
check(workbook.items.length === 4, "Ready mappings are excluded from the decision workbook");
check(workbook.runnable === false && workbook.scientificEquivalenceClaimed === false, "workbook cannot claim runnable or equivalent state");
check(cm1DecisionProgress(workbook).remaining === 4, "initial remaining count covers every non-Ready mapping");

const ztop = workbook.items.find(({ key }) => key === "ztop")!;
check(cm1DecisionOptions(ztop).includes("keep_nimbus_default"), "needs-decision item can keep the Nimbus default");
check(cm1DecisionOptions(ztop).includes("manual_resolution"), "descriptor-backed target permits manual resolution");
check(ztop.targetDescriptor?.defaultValue === 16_000, "applicable target records its exact current Nimbus default");
check(workbook.source.descriptorFingerprint.startsWith("fnv1a32:"), "workbook source binds the complete descriptor snapshot");
check(validateCm1ManualValue(ztop, "18000").valid, "manual scalar passes the current Nimbus descriptor");
check(!validateCm1ManualValue(ztop, "50").valid, "manual scalar outside descriptor bounds fails closed");
const compound = workbook.items.find(({ key }) => key === "cm1setup")!;
check(!cm1DecisionOptions(compound).includes("manual_resolution"), "mapping without a target cannot claim manual resolution");
check(cm1DecisionOptions(workbook.items.find(({ key }) => key === "irdamp")!).join() === "acknowledge_difference", "different formulation has an acknowledgement decision only");
check(cm1DecisionOptions(workbook.items.find(({ key }) => key === "irst")!).join() === "omit_unsupported", "unsupported mapping has an omission decision only");

const decided = decideCm1MappingItem(workbook, ztop.id, { kind: "manual_resolution", manualValue: "18000", note: "Reviewed against the stretched grid." }, "2026-07-20T01:00:00.000Z");
check(cm1DecisionProgress(decided).complete === 1, "recording a decision updates complete count");
check(decided.items.find(({ id }) => id === ztop.id)?.decision?.manualValue === "18000", "validated canonical manual value is retained");
check(clearCm1MappingDecision(decided, ztop.id).items.find(({ id }) => id === ztop.id)?.decision === null, "decision can be cleared without changing source evidence");
const keptDefault = decideCm1MappingItem(workbook, ztop.id, { kind: "keep_nimbus_default" });
check(keptDefault.items.find(({ id }) => id === ztop.id)?.decision?.recordedDefault === 16_000, "keep-default decision freezes the exact reviewed default value");
let wrongDispositionRejected = false;
try { decideCm1MappingItem(workbook, ztop.id, { kind: "omit_unsupported" }); } catch { wrongDispositionRejected = true; }
check(wrongDispositionRejected, "decision inappropriate to disposition is rejected");
let longNoteRejected = false;
try { decideCm1MappingItem(workbook, ztop.id, { kind: "keep_nimbus_default", note: "x".repeat(CM1_DECISION_NOTE_MAX_CHARS + 1) }); } catch { longNoteRejected = true; }
check(longNoteRejected, "oversize optional note is rejected");

const json = serializeCm1DecisionWorkbook(decided);
check(importCm1DecisionWorkbook(json, workbook).items[0].decision != null, "portable JSON round trips against the same source");
let sourceMismatchRejected = false;
try { importCm1DecisionWorkbook(json, createCm1DecisionWorkbook(sourceB, report)); } catch { sourceMismatchRejected = true; }
check(sourceMismatchRejected, "portable JSON import rejects a different source identity");
check(parseCm1DecisionWorkbook({ ...decided, schemaVersion: 99 }) === null, "unknown schema version fails closed");
check(parseCm1DecisionWorkbook({ ...decided, runnable: true }) === null, "runnable claim fails closed");
check(parseCm1DecisionWorkbook({ ...decided, source: { ...decided.source, mappingFingerprint: "fnv1a32:00000000:4" } }) === null, "mapping snapshot fingerprint mismatch fails closed");
let oversizeImportRejected = false;
try { importCm1DecisionWorkbook(" ".repeat(CM1_DECISION_WORKBOOK_MAX_JSON_BYTES + 1), workbook); } catch { oversizeImportRejected = true; }
check(oversizeImportRejected, "oversize portable JSON fails before parsing");
const descriptorDrift = {
  ...workbook,
  source: { ...workbook.source, descriptorFingerprint: "fnv1a32:descriptor-drift" },
};
let descriptorDriftRejected = false;
try { importCm1DecisionWorkbook(serializeCm1DecisionWorkbook(keptDefault), descriptorDrift); } catch { descriptorDriftRejected = true; }
check(descriptorDriftRejected, "descriptor drift rejects a previously completed review");
check(parseCm1DecisionWorkbook({
  ...keptDefault,
  items: keptDefault.items.map((item) => item.id === ztop.id
    ? { ...item, decision: { ...item.decision!, recordedDefault: 18_000 } }
    : item),
}) === null, "tampered recorded default fails closed");

const storage = new MemoryStorage();
const workbookB = createCm1DecisionWorkbook(sourceB, report);
saveCm1DecisionWorkbook(storage, decided);
saveCm1DecisionWorkbook(storage, workbookB);
check(storage.values.size === 2, "identical mappings from two source datasets use isolated storage keys");
check(loadCm1DecisionWorkbook(storage, workbook)?.items[0].decision != null, "source A decisions survive source B persistence");
check(loadCm1DecisionWorkbook(storage, workbookB)?.items.every(({ decision }) => decision == null) ?? false, "source B loads only its own decisions");

const readyOnly = createCm1DecisionWorkbook(sourceA, { ...report, entries: report.entries.slice(0, 1) });
check(cm1DecisionProgress(readyOnly).total === 0 && cm1DecisionProgress(readyOnly).remaining === 0, "zero-item workbook is complete with no decisions required");

console.log(`CM1 decision workbook: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
