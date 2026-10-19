import {
  validatePackagedVisualizationEvidence,
} from "./validate_packaged_visualization_evidence.ts";
import { PACKAGED_VISUALIZATION_MATRIX } from "./aggregate_packaged_visualization_evidence.ts";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { aggregatePackagedVisualizationEvidence } from "./aggregate_packaged_visualization_evidence.ts";

const checkNames = [
  "module-worker-api",
  "profile-byte-contract",
  "ui-thread-watchdog",
  "process-memory-coverage",
  "process-memory-monotonic",
  "packaged-appearance-scenes",
  "packaged-appearance-repeatability",
  "packaged-native-brick-volume",
  "packaged-native-brick-streamlines",
  "packaged-terrain-brick-diagnostics",
  "physical-gpu-residency",
  ...[1, 2].flatMap((cycle) => [
    `cycle-${cycle}-scalar-worker-handle`,
    `cycle-${cycle}-scalar-main-resolution`,
    `cycle-${cycle}-contour-geometry`,
    `cycle-${cycle}-vector-worker-handle`,
    `cycle-${cycle}-vector-main-resolution`,
    `cycle-${cycle}-streamline-geometry`,
    `cycle-${cycle}-registry-retired`,
    `cycle-${cycle}-worker-residency-retired`,
  ]),
];

function report() {
  const scalarBytes = 40 * 1024 * 1024;
  const vectorBytes = 120 * 1024 * 1024;
  const residentHighWaterBytes = 160 * 1024 * 1024;
  const highWaterBytes = 280 * 1024 * 1024;
  const labels = ["startup", "fixture-allocated", "gpu-baseline",
    "gpu-cycle-1-resident", "gpu-cycle-1-retired", "gpu-cycle-2-resident", "gpu-cycle-2-retired",
    ...[1, 2].flatMap(cycle => ["scalar-resident", "contour-published", "scalar-vector-resident", "streamline-published", "shutdown"]
      .map(stage => `cycle-${cycle}-${stage}`)), "packaged-native-bricks-resident", "final"];
  const memorySamples = labels.map((label, index) => ({
    label,
    elapsedMs: index,
    samplingMs: 1,
    memory: {
      schema: "nimbus.studio.process-memory",
      schemaVersion: 3,
      currentRssBytes: 200_000_000 + index,
      peakRssBytes: 400_000_000,
      processTreeRssBytes: index === 5 ? 329_000_000 : index === 6 ? 328_000_000 : 300_000_000 + index * 10_000_000,
      processTreePeakRssBytes: 300_000_000 + index * 10_000_000,
      processTreeProcessCount: 5,
      processTreeCoverage: "whole_application",
      measurement: "fixture",
    },
  }));
  const baseline = memorySamples[0].memory.processTreeRssBytes;
  const currentHigh = memorySamples.at(-1)!.memory.processTreeRssBytes;
  const delta = currentHigh - baseline;
  const accounted = scalarBytes + vectorBytes + residentHighWaterBytes;
  return {
    schema: "nimbus.studio.packaged-visualization-qualification",
    schemaVersion: 2,
    scope: "bounded-packaged-lifecycle",
    memoryCampaign: "pending",
    status: "passed",
    profile: "production",
    inputHandoffYieldMode: "scheduler-yield",
    packagedRuntime: { appVersion: "1", targetOs: "macos", targetArch: "aarch64", buildMode: "release" },
    checks: checkNames.map((name) => ({ name, passed: true, detail: name })),
    shape: [256, 256, 160],
    scalarBytes,
    vectorBytes,
    maximumWatchdogDelayMs: 10,
    watchdog: {
      intervalMs: 50,
      eventThresholdMs: 25,
      maximumDelayMs: 10,
      maximumByStageMs: Object.fromEntries([
        "initialization",
        ...[1, 2].flatMap((cycle) => [
          `cycle-${cycle}-scalar-preparation`,
          `cycle-${cycle}-contour-extraction`,
          `cycle-${cycle}-vector-preparation`,
          `cycle-${cycle}-streamline-extraction`,
          `cycle-${cycle}-worker-shutdown`,
          `cycle-${cycle}-retirement-checks`,
        ]),
        "qualification-checks",
        "packaged-appearance",
        "packaged-native-bricks",
        "physical-gpu-residency",
      ].map((stage, index) => [stage, index === 0 ? 10 : 0])),
      events: [] as Array<{
        stage: string;
        elapsedMs: number;
        delayMs: number;
        visibilityState: "hidden" | "visible";
        documentHasFocus: boolean;
      }>,
    },
    elapsedMs: 100,
    cycles: [1, 2].map(() => ({
      contourTriangles: 10,
      integratedPoints: 10,
      timings: {
        scalarDispatchMs: 1, scalarPreparationMs: 1, contourExtractionMs: 1,
        vectorDispatchMs: 1, vectorPreparationMs: 1, streamlineExtractionMs: 1,
      },
    })),
    packagedAppearance: {
      schema: "nimbus.studio.packaged-vtk-appearance",
      schemaVersion: 1,
      scenes: ["volume", "isosurface", "streamlines", "supercell", "trajectories"].map(
        (plot, index) => ({
          plot,
          pngSha256: String(index + 1).repeat(64),
          repeatPngSha256: String(index + 1).repeat(64),
          metadataFingerprint: `fnv1a64:${String(index + 1).repeat(16)}`,
          repeatMetadataFingerprint: `fnv1a64:${String(index + 1).repeat(16)}`,
          meanLuma: 0.2,
          lumaDeviation: 0.1,
          viewport: {
            cssWidth: 640,
            cssHeight: 420,
            backingWidth: 1280,
            backingHeight: 840,
            pixelRatio: 2,
          },
          visibleLayers: 1,
        }),
      ),
    },
    nativeBrickEvidence: {
      volumeBricks: 8,
      volumeSamples: 1_568,
      anomalyBricks: 8,
      excludedParentSamples: 256,
      vectorBricks: 8,
      integratedPoints: 100,
      terrainFollowing: true,
      terrainGeometryWindows: 24,
      terrainDiagnosticTriangles: 120,
      terrainDiagnosticRange: [-0.01, 0.02],
    },
    physicalGpuEvidence: {
      measurementProtocol: "isolated-prepared-pair-post-scope-v2",
      renderingPath: "nimbus-coherent-volume",
      cycles: 2,
      scalarTextureBytesPerCycle: 80 * 1024 * 1024,
      preparedCpuBufferBytes: 80 * 1024 * 1024,
      preparedCpuBuffers: 2,
      retiredContexts: 2,
      retiredOwnedObjects: [15, 15],
      zeroedCanvasBackingStores: 2,
      postScopeRetirementMs: [30, 32],
      maximumLogicalGpuResidentBytes: 80 * 1024 * 1024,
      rss: {
        currency: "process-tree-rss",
        baselineBytes: 320_000_000,
        residentBytes: [330_000_000, 329_000_000],
        retiredBytes: [340_000_000, 328_000_000],
        residentDeltaBytes: [10_000_000, 9_000_000],
        restartGrowthBytes: -1_000_000,
        limitation: "Process-tree RSS is not physical VRAM.",
      },
      viewport: { width: 640, height: 420 },
      actualRenderedTexturesPerCycle: 2,
      contextLost: false,
      glErrors: [0, 0],
      maximum3DTextureSize: 2_048,
      renderPngSha256: ["a".repeat(64), "b".repeat(64)],
    },
    finalRegistry: { bytes: 0, handles: 0, references: 0, mainBytes: 0, mainHandles: 0, workerBytes: 0, workerHandles: 0 },
    finalResources: {
      allocations: 0, reservations: 0, residentBytes: 0, reservedBytes: 0, committedBytes: 0,
      residentHighWaterBytes,
      highWaterBytes,
      prediction: { underpredictions: 0 },
    },
    memorySamples,
    memoryReconciliation: {
      coverage: "whole_application",
      baselineTreeRssBytes: baseline,
      processTreeCurrentHighBytes: currentHigh,
      processTreePeakBytes: currentHigh,
      processTreeDeltaBytes: delta,
      maximumSamplingMs: 1,
      syntheticInputBytes: scalarBytes + vectorBytes,
      trackedResidentHighWaterBytes: residentHighWaterBytes,
      trackedCommittedHighWaterBytes: highWaterBytes,
      accountedInputAndResidentHighWaterBytes: accounted,
      observedMinusAccountedBytes: delta - accounted,
      limitation: "fixture",
    },
  };
}

let passed = 0;
const failures: string[] = [];
function accepts(label: string, mutate?: (value: ReturnType<typeof report>) => void): void {
  const value = report();
  mutate?.(value);
  try {
    validatePackagedVisualizationEvidence(value, {
      profile: "production", mode: "release", targetOs: "macos", targetArch: "aarch64",
    });
    passed += 1;
  } catch (error) {
    failures.push(`FAIL ${label}: ${error instanceof Error ? error.message : String(error)}`);
  }
}
function rejects(label: string, mutate: (value: ReturnType<typeof report>) => void, expectedError?: string): void {
  const value = report();
  mutate(value);
  try {
    validatePackagedVisualizationEvidence(value, {
      profile: "production", mode: "release", targetOs: "macos", targetArch: "aarch64",
    });
    failures.push(`FAIL ${label}: invalid evidence was accepted`);
  } catch (error) {
    if (expectedError && !(error instanceof Error && error.message.includes(expectedError))) {
      failures.push(`FAIL ${label}: unexpected rejection ${String(error)}`);
    } else passed += 1;
  }
}
function rejectsForTarget(
  label: string,
  targetOs: "linux" | "windows",
  mutate: (value: ReturnType<typeof report>) => void,
): void {
  const value = report();
  value.packagedRuntime.targetOs = targetOs;
  mutate(value);
  try {
    validatePackagedVisualizationEvidence(value, {
      profile: "production", mode: "release", targetOs, targetArch: "aarch64",
    });
    failures.push(`FAIL ${label}: invalid evidence was accepted`);
  } catch {
    passed += 1;
  }
}

accepts("complete production evidence");
rejects("platform without host-current RSS", (value) => {
  value.memorySamples.forEach((sample) => { sample.memory.currentRssBytes = null as unknown as number; });
});
rejects("wrong scalar bytes", (value) => { value.scalarBytes -= 4; });
rejects("wrong build mode", (value) => { value.packagedRuntime.buildMode = "debug"; });
rejects("missing input handoff yield mode", (value) => {
  value.inputHandoffYieldMode = "";
});
rejects("missing check", (value) => { value.checks.pop(); });
rejects("duplicate check", (value) => { value.checks[1].name = value.checks[0].name; });
rejects("failed check", (value) => { value.checks[0].passed = false; });
rejects("watchdog breach", (value) => { value.maximumWatchdogDelayMs = 501; });
rejects("watchdog maximum mismatch", (value) => { value.watchdog.maximumDelayMs = 11; });
rejects("missing watchdog stage", (value) => {
  delete value.watchdog.maximumByStageMs["cycle-2-vector-preparation"];
});
rejects("watchdog event below threshold", (value) => {
  value.watchdog.events.push({
    stage: "initialization",
    elapsedMs: 1,
    delayMs: 24,
    visibilityState: "visible",
    documentHasFocus: true,
  });
});
rejects("one cycle", (value) => { value.cycles.pop(); });
rejects("empty geometry", (value) => { value.cycles[0].contourTriangles = 0; });
rejects("missing packaged appearance scene", (value) => { value.packagedAppearance.scenes.pop(); });
rejects("nonrepeatable packaged appearance", (value) => {
  value.packagedAppearance.scenes[0].repeatPngSha256 = "f".repeat(64);
});
rejects("blank packaged appearance", (value) => {
  value.packagedAppearance.scenes[0].meanLuma = 0;
  value.packagedAppearance.scenes[0].lumaDeviation = 0;
});
rejects("invalid packaged backing store", (value) => {
  value.packagedAppearance.scenes[0].viewport.backingWidth = 0;
});
rejects("retained handle", (value) => { value.finalRegistry.handles = 1; });
rejects("retained worker bytes", (value) => { value.finalResources.residentBytes = 1; });
rejects("underprediction", (value) => { value.finalResources.prediction.underpredictions = 1; });
function gpuSamples(value: ReturnType<typeof report>, bytes: [number, number, number, number, number]): void {
  bytes.forEach((rss, index) => { value.memorySamples[index + 2].memory.processTreeRssBytes = rss; });
  const [baselineBytes, first, retiredFirst, second, retiredSecond] = bytes;
  Object.assign(value.physicalGpuEvidence.rss, { baselineBytes,
    residentBytes: [first, second], retiredBytes: [retiredFirst, retiredSecond],
    residentDeltaBytes: [first - baselineBytes, second - baselineBytes], restartGrowthBytes: second - first });
  let peak = 0;
  value.memorySamples.forEach(sample => {
    peak = Math.max(peak, sample.memory.processTreeRssBytes);
    sample.memory.processTreePeakRssBytes = peak;
  });
  const reconciliation = value.memoryReconciliation;
  reconciliation.processTreeCurrentHighBytes = reconciliation.processTreePeakBytes = peak;
  reconciliation.processTreeDeltaBytes = peak - reconciliation.baselineTreeRssBytes;
  reconciliation.observedMinusAccountedBytes = reconciliation.processTreeDeltaBytes - reconciliation.accountedInputAndResidentHighWaterBytes;
}
rejects("obsolete v1 report", value => { value.schemaVersion = 1; });
rejects("memory campaign promotion", value => { value.memoryCampaign = "validated"; });
rejects("unbounded scope claim", value => { value.scope = "capacity-qualified"; });
rejects("legacy clipped GPU protocol", value => { value.physicalGpuEvidence.measurementProtocol = "legacy-per-cycle-clipped"; });
rejects("upstream-only GPU path", value => { value.physicalGpuEvidence.renderingPath = "upstream-volume"; });
rejects("missing prepared input ownership", value => { value.physicalGpuEvidence.preparedCpuBufferBytes -= 4; });
rejects("retained core owner", value => { value.physicalGpuEvidence.retiredOwnedObjects[1] = 14; });
rejects("retained canvas backing", value => { value.physicalGpuEvidence.zeroedCanvasBackingStores = 1; });
rejects("unconfirmed context retirement", value => { value.physicalGpuEvidence.retiredContexts = 1; });
rejects("scope-retained GPU protocol", value => { value.physicalGpuEvidence.measurementProtocol = "isolated-prepared-pair-common-baseline-v1"; });
rejects("missing post-scope retirement timing", value => { value.physicalGpuEvidence.postScopeRetirementMs = []; });
rejects("invalid post-scope retirement timing", value => { value.physicalGpuEvidence.postScopeRetirementMs[0] = -1; });
accepts("signed negative common-baseline deltas", value => {
  gpuSamples(value, [400_000_000, 350_000_000, 340_000_000, 349_000_000, 339_000_000]);
});
accepts("unclipped growth below the common baseline remains an observation", value => {
  gpuSamples(value, [400_000_000, 350_000_000, 340_000_000, 360_000_000, 339_000_000]);
});
accepts("positive growth is retained without claiming a plateau", value => {
  gpuSamples(value, [300_000_000, 400_000_000, 390_000_000, 450_000_000, 390_000_000]);
});
accepts("first-cycle decline does not hide subsequent growth", value => {
  gpuSamples(value, [400_000_000, 350_000_000, 351_000_000, 450_000_000, 449_000_000]);
});
rejects("clipped signed RSS summary", value => {
  gpuSamples(value, [400_000_000, 350_000_000, 340_000_000, 349_000_000, 339_000_000]);
  value.physicalGpuEvidence.rss.residentDeltaBytes[0] = 0;
}, "without clipping");
rejects("forged resident RSS summary", value => { value.physicalGpuEvidence.rss.residentBytes[1] -= 1; });
rejects("forged retired RSS summary", value => { value.physicalGpuEvidence.rss.retiredBytes[1] -= 1; });
rejects("forged GPU baseline", value => { value.physicalGpuEvidence.rss.baselineBytes += 1; });
rejects("GPU phase after worker phases", value => { value.memorySamples[2].label = "cycle-1-scalar-resident"; });
rejects("missing memory sample", (value) => { value.memorySamples.pop(); });
rejects("host-only memory", (value) => { value.memorySamples[0].memory.processTreeCoverage = "host_only"; });
rejects("incomplete macOS coalition", (value) => {
  value.memorySamples[0].memory.processTreeProcessCount = 3;
});
rejectsForTarget("Linux descendant-only coverage", "linux", (value) => {
  value.memorySamples.forEach((sample) => {
    sample.memory.processTreeCoverage = "host_plus_descendants";
  });
});
rejectsForTarget("Windows descendant-only coverage", "windows", (value) => {
  value.memorySamples.forEach((sample) => {
    sample.memory.processTreeCoverage = "host_plus_descendants";
  });
});
rejects("non-monotonic memory peak", (value) => {
  value.memorySamples[1].memory.processTreePeakRssBytes = 1;
});
rejects("inconsistent memory delta", (value) => { value.memoryReconciliation.processTreeDeltaBytes += 1; });

const matrixKeys = new Set(PACKAGED_VISUALIZATION_MATRIX.map(
  (entry) => `${entry.targetOs}/${entry.targetArch}`,
));
if (PACKAGED_VISUALIZATION_MATRIX.length === 6 && matrixKeys.size === 6
    && ["macos", "linux", "windows"].every((os) =>
      matrixKeys.has(`${os}/aarch64`) && matrixKeys.has(`${os}/x86_64`))) {
  passed += 1;
} else {
  failures.push("FAIL packaged matrix must contain unique arm64/x64 profiles for macOS/Linux/Windows");
}

const matrixRoot = mkdtempSync(join(tmpdir(), "nimbus-packaged-visualization-matrix-"));
try {
  for (const profile of PACKAGED_VISUALIZATION_MATRIX) {
    const value = report();
    value.packagedRuntime.targetOs = profile.targetOs;
    value.packagedRuntime.targetArch = profile.targetArch;
    writeFileSync(join(matrixRoot, profile.file), JSON.stringify(value));
  }
  const aggregate = aggregatePackagedVisualizationEvidence(matrixRoot);
  if (aggregate.status === "passed" && aggregate.entries.length === 6
      && new Set(aggregate.entries.map((entry) => entry.sha256)).size === 6) {
    passed += 1;
  } else {
    failures.push("FAIL complete six-profile evidence does not aggregate deterministically");
  }
  rmSync(join(matrixRoot, PACKAGED_VISUALIZATION_MATRIX[0].file));
  try {
    aggregatePackagedVisualizationEvidence(matrixRoot);
    failures.push("FAIL matrix aggregation accepted a missing platform report");
  } catch {
    passed += 1;
  }
} finally {
  rmSync(matrixRoot, { recursive: true, force: true });
}

function pythonReport() {
  const base = report();
  const checks = ["script-file", "run-field", "figure", "produced-file", "exact-export", "stop", "restart"]
    .map((name) => ({ name: `python-${name}`, passed: true, detail: name }));
  return { ...base, checks: [...base.checks, ...checks], python: { checks, elapsedMs: 10, crossOriginIsolated: false } };
}
try {
  validatePackagedVisualizationEvidence(pythonReport(), { profile: "production", mode: "release", requirePython: true });
  passed++;
} catch (error) { failures.push(`valid native Python evidence: ${error}`); }
for (const mutate of [
  (value: ReturnType<typeof pythonReport>) => { delete (value as { python?: unknown }).python; },
  (value: ReturnType<typeof pythonReport>) => { value.python.checks.pop(); },
  (value: ReturnType<typeof pythonReport>) => { value.python.checks[0].passed = false; },
  (value: ReturnType<typeof pythonReport>) => { value.python.checks[1] = value.python.checks[0]; },
]) {
  const value = pythonReport(); mutate(value);
  try {
    validatePackagedVisualizationEvidence(value, { profile: "production", mode: "release", requirePython: true });
    failures.push("invalid native Python evidence accepted");
  } catch { passed++; }
}
for (const [input, batch] of [
  [report(), { schema: "nimbus.studio.packaged-config-qualification.v1", binarySha256: "a".repeat(64), cases: [] }],
  [{ ...report(), configPreflight: { reports: [] } }, undefined],
]) {
  try {
    validatePackagedVisualizationEvidence(input, { profile: "production", mode: "release", configPreflightBatch: batch });
    failures.push("missing or unrequested configuration evidence accepted");
  } catch { passed++; }
}
console.log(`packaged visualization evidence tests: ${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
