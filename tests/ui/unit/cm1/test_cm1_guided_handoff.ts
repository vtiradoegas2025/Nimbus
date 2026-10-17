import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  cm1HandoffComparisonPair,
  cm1StagedLaunchMismatch,
  consumeCm1SmokeConfig,
  createCm1GuidedHandoff,
  fnv1a64Text,
  loadCm1GuidedHandoff,
  markCm1ComparisonOpened,
  markCm1SmokeRunning,
  parseCm1GuidedHandoff,
  reconcileCm1SmokeRun,
  saveCm1GuidedHandoff,
} from "../../../../ui/src/lib/cm1/cm1GuidedHandoff.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const cm1 = {
  ...createLocalDatasetSource("cm1_netcdf", ["/case/cm1out_000001.nc", "/case/cm1out_000002.nc"]),
  caseInventory: { historyFiles: ["a", "b"] },
} as ReturnType<typeof createLocalDatasetSource>;
const baseInput = {
  cm1Source: cm1,
  soundingSourcePath: "/case/input_sounding",
  projectedLevelCount: 72,
  mappingCounts: { ready: 9, needsDecision: 3, different: 2, unsupported: 4 },
  mappingBlockers: ["3 decisions remain"],
  soundingPath: "/work/cm1-sounding.txt",
  configPath: "/work/cm1-smoke.yaml",
  yaml: "schema: nimbus.v1\ntime:\n  run_time: 30\n  output_interval: 10\n",
  catalogFingerprint: "sha256:test",
  executablePath: "/bin/nimbus_sim",
  initializationExecutablePath: "/bin/nimbus_sim",
  now: "2026-07-20T00:00:00.000Z",
} as const;
const base = createCm1GuidedHandoff(baseInput);
check(base.evidence.historyFileCount === 2, "history evidence is retained");
check(base.smoke.durationS === 30 && base.smoke.outputIntervalS === 10, "smoke bounds are explicit");
check(base.smoke.configurationDigest === fnv1a64Text(baseInput.yaml), "exact staged YAML digest is deterministic");
check(cm1StagedLaunchMismatch(base, { yaml: base.smoke.yaml, catalogFingerprint: base.smoke.catalogFingerprint, executablePath: base.smoke.executablePath }) === null, "exact staged launch evidence is accepted");
check(cm1StagedLaunchMismatch(base, { yaml: base.smoke.yaml.replace("run_time: 30", "run_time: 31"), catalogFingerprint: base.smoke.catalogFingerprint, executablePath: base.smoke.executablePath })?.includes("YAML") ?? false, "edited YAML is rejected before guided launch");
check(cm1StagedLaunchMismatch(base, { yaml: base.smoke.yaml, catalogFingerprint: "other", executablePath: base.smoke.executablePath })?.includes("catalog") ?? false, "changed catalog is rejected before guided launch");
check(base.cm1Source.caseInventory == null, "persisted source is compact and relocatable");
check(base.evidence.mappingBlockerTotal === 1 && !base.evidence.mappingBlockersTruncated, "blocker accounting is explicit");
check(base.evidence.soundingSourceMode === "reviewed_projection", "existing handoffs retain reviewed-projection provenance by default");
check(cm1HandoffComparisonPair(base) === null, "comparison fails closed before a run");

const consumed = consumeCm1SmokeConfig(base);
const running = markCm1SmokeRunning(consumed, "run-1", "/runs/run-1");
const durable = (overrides: Record<string, unknown> = {}) => ({
  outputDir: "/runs/run-1",
  health: "ready" as const,
  issue: null,
  frameCount: 3,
  provenance: {
    health: "verified" as const,
    runtimeHealth: "verified" as const,
    runId: "run-1",
    executablePath: base.smoke.executablePath,
    configurationDigest: base.smoke.configurationDigest,
    catalogFingerprint: base.smoke.catalogFingerprint,
    finalStatus: "completed",
    exitCode: 0,
    completeFrameCount: 3,
  },
  ...overrides,
});
const complete = reconcileCm1SmokeRun(running, durable());
check(complete.run.status === "complete", "only exact durable output evidence completes the handoff");
const pair = cm1HandoffComparisonPair(complete);
check(pair?.left.kind === "cm1_netcdf" && pair.right.locator.path === "/runs/run-1", "qualified pair preserves CM1 and Nimbus identities");
check(markCm1ComparisonOpened(complete).comparisonOpenedAt != null, "comparison opening is recorded");

const storage = new MemoryStorage();
saveCm1GuidedHandoff(storage, complete);
check(loadCm1GuidedHandoff(storage)?.run.status === "complete", "versioned handoff round trips");
check(parseCm1GuidedHandoff({ ...complete, schemaVersion: 99 }) === null, "unknown version fails closed");
check(parseCm1GuidedHandoff({ ...complete, run: { status: "complete", runId: "run-1", outputDir: null, pendingReason: null } }) === null, "invalid complete evidence fails closed");
check(parseCm1GuidedHandoff({ ...base, smoke: { ...base.smoke, yaml: `${base.smoke.yaml}# edit\n` } }) === null, "edited staged YAML fails its persisted digest");
const native = createCm1GuidedHandoff({
  ...baseInput,
  soundingPath: "/case/input_sounding",
  soundingSourceMode: "native_cm1",
});
check(native.evidence.soundingSourceMode === "native_cm1" && parseCm1GuidedHandoff(native) !== null, "native CM1 source provenance round trips in the additive v2 record");
const legacyWithoutMode = { ...base, evidence: { ...base.evidence } };
delete legacyWithoutMode.evidence.soundingSourceMode;
check(parseCm1GuidedHandoff(legacyWithoutMode) !== null, "handoffs saved before native source provenance remain readable");

const manyBlockers = createCm1GuidedHandoff({
  ...baseInput,
  mappingCounts: { ready: 9, needsDecision: 13, different: 2, unsupported: 4 },
  yaml: "schema: nimbus.v1",
  mappingBlockers: Array.from({ length: 13 }, (_, index) => `blocker ${index + 1}`),
});
check(manyBlockers.evidence.mappingBlockers.length === 12, "stored blocker detail is bounded");
check(manyBlockers.evidence.mappingBlockerTotal === 13 && manyBlockers.evidence.mappingBlockersTruncated, "bounded blocker detail discloses truncation");
let rejectedOversizeYaml = false;
try { createCm1GuidedHandoff({ ...baseInput, yaml: "x".repeat(512_001) }); } catch { rejectedOversizeYaml = true; }
check(rejectedOversizeYaml, "oversize staged YAML is rejected");
check(reconcileCm1SmokeRun(running, null).run.status === "pending", "missing durable scan evidence remains pending");
check(reconcileCm1SmokeRun(running, durable({ outputDir: "/runs/wrong" })).run.status === "pending", "different output directory remains pending");
check(reconcileCm1SmokeRun(running, durable({ provenance: { ...durable().provenance, runId: "other" } })).run.status === "pending", "different provenance run identity remains pending");
check(reconcileCm1SmokeRun(running, durable({ health: "invalid", issue: "corrupt manifests", frameCount: 0 })).run.status === "pending", "corrupt output remains pending");
check(reconcileCm1SmokeRun(running, durable({ frameCount: 0, provenance: { ...durable().provenance, completeFrameCount: 0 } })).run.status === "pending", "zero-frame output remains pending");
check(reconcileCm1SmokeRun(running, durable({ provenance: { ...durable().provenance, configurationDigest: fnv1a64Text(`${base.smoke.yaml}edited`) } })).run.status === "pending", "edited YAML provenance remains pending");
check(reconcileCm1SmokeRun(running, durable({ provenance: { ...durable().provenance, catalogFingerprint: "other" } })).run.status === "pending", "different catalog fingerprint remains pending");
check(reconcileCm1SmokeRun(running, durable({ provenance: { ...durable().provenance, executablePath: "/bin/other" } })).run.status === "pending", "different executable identity remains pending");
check(reconcileCm1SmokeRun(running, durable({ provenance: { ...durable().provenance, finalStatus: "failed", exitCode: 1 } })).run.status === "failed", "durable terminal failure remains fail closed");
const restarted = parseCm1GuidedHandoff(JSON.parse(JSON.stringify(running)))!;
check(reconcileCm1SmokeRun(restarted, durable()).run.status === "complete", "post-restart durable scan recovers the exact matching output");
check(reconcileCm1SmokeRun(complete, null).run.status === "pending", "persisted complete state is not evidence when durable output is missing");
check(reconcileCm1SmokeRun(complete, durable({ health: "invalid", issue: "manifest changed" })).run.status === "pending", "persisted complete state is revoked when durable output becomes invalid");
check(reconcileCm1SmokeRun(complete, durable()).run.status === "complete", "persisted complete state survives fresh matching durable evidence");
const restartedWithoutLivePath = parseCm1GuidedHandoff(JSON.parse(JSON.stringify(
  markCm1SmokeRunning(consumed, "run-1", null),
)))!;
const rebound = reconcileCm1SmokeRun(restartedWithoutLivePath, durable());
check(rebound.run.status === "complete" && rebound.run.outputDir === "/runs/run-1", "post-restart durable provenance uniquely binds an output path lost from the live registry");

console.log(`CM1 guided handoff: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
