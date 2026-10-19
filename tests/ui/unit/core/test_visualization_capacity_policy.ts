import {
  BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES,
  resolveVisualizationCapacity,
  parseVisualizationCapacityQualification,
  type VisualizationCapacityQualification,
} from "../../../../ui/src/lib/core/visualizationCapacityPolicy.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const hardware = {
  source: "tauri" as const,
  systemRamBytes: 64 * 1024 * 1024 * 1024,
  logicalCores: 16,
};
const current = resolveVisualizationCapacity(hardware);
check(current.mode === "observe-only" && current.capacityBytes === null, "missing evidence cannot enable capacity enforcement");
check(
  current.observedFloorBytes === 422_363_312
    && current.blockers.some((blocker) => blocker.includes("GPU residency")),
  "observe-only policy discloses the measured floor and opaque blockers",
);

const qualified: VisualizationCapacityQualification = {
  schema: "nimbus.studio.visualization-capacity-qualification",
  schemaVersion: 2,
  qualifiedCapacityBytes: 1024 * 1024 * 1024,
  minimumSystemRamBytes: 16 * 1024 * 1024 * 1024,
  packagedMemoryCoverage: "whole-application",
  gpuResidencyValidated: true,
  opaqueWorkerHeadroomValidated: true,
  profiles: ["macos-wkwebview", "windows-webview2", "linux-webkitgtk"],
  targetOs: "macos",
  targetArch: "aarch64",
  appVersion: "0.1.0",
  campaignDurationSeconds: 30 * 60,
  evidenceSha256: "a".repeat(64),
};
check(
  parseVisualizationCapacityQualification(qualified)?.qualifiedCapacityBytes
    === qualified.qualifiedCapacityBytes,
  "a complete target-bound campaign artifact parses",
);
let shortCampaignRejected = false;
try {
  parseVisualizationCapacityQualification({ ...qualified, campaignDurationSeconds: 1799 });
} catch {
  shortCampaignRejected = true;
}
check(shortCampaignRejected, "a sub-30-minute artifact cannot authorize enforcement");
const admitted = resolveVisualizationCapacity(hardware, qualified);
check(
  admitted.mode === "enforced" && admitted.capacityBytes === qualified.qualifiedCapacityBytes,
  "complete evidence enables only its explicit qualified capacity",
);

const belowObserved = resolveVisualizationCapacity(hardware, {
  ...qualified,
  qualifiedCapacityBytes: BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES - 1,
});
check(
  belowObserved.mode === "observe-only"
    && belowObserved.blockers.some((blocker) => blocker.includes("below the observed")),
  "a nominal qualification cannot undercut the measured Beast commitment",
);

const parentOnly = resolveVisualizationCapacity(hardware, {
  ...qualified,
  packagedMemoryCoverage: "parent-descendants",
});
check(
  parentOnly.mode === "observe-only"
    && parentOnly.blockers.some((blocker) => blocker.includes("whole-application RSS")),
  "a parent/descendant sampler cannot authorize aggregate capacity enforcement",
);

const unsupportedSchema = resolveVisualizationCapacity(hardware, {
  ...qualified,
  schemaVersion: 1,
} as unknown as VisualizationCapacityQualification);
check(
  unsupportedSchema.mode === "observe-only"
    && unsupportedSchema.blockers.some((blocker) => blocker.includes("schema is unsupported")),
  "an obsolete capacity evidence schema fails closed",
);

const insufficientHost = resolveVisualizationCapacity({
  ...hardware,
  systemRamBytes: qualified.minimumSystemRamBytes - 1,
}, qualified);
check(
  insufficientHost.mode === "observe-only"
    && insufficientHost.blockers.some((blocker) => blocker.includes("below the qualified hardware")),
  "qualified capacity remains bounded to its measured hardware envelope",
);

const unknownRam = resolveVisualizationCapacity({
  source: "browser",
  systemRamBytes: null,
  logicalCores: 8,
}, qualified);
check(
  unknownRam.mode === "observe-only"
    && unknownRam.blockers.some((blocker) => blocker.includes("system RAM is unavailable")),
  "unknown browser RAM cannot inherit a desktop capacity claim",
);

console.log(`\nvisualization capacity policy tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
