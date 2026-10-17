import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { validateConfigPreflightReport, validateConfigPreflightEvidence, type ConfigPreflightRequest } from "../../../ui/src/lib/config/configPreflight.ts";
import { pathToFileURL } from "node:url";

const SCHEMA = "nimbus.studio.packaged-visualization-qualification";
const REQUIRED_CHECKS = new Set([
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
  "physical-gpu-restart-plateau",
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
]);

const PYTHON_CHECKS = ["script-file", "run-field", "figure", "produced-file", "exact-export", "stop", "restart"].map((name) => `python-${name}`);

type UnknownRecord = Record<string, unknown>;

export interface EvidenceExpectation {
  profile: "smoke" | "production";
  mode: "debug" | "release";
  targetOs?: string;
  targetArch?: string;
  requirePython?: boolean;
  configPreflightBatch?: unknown;
}

function record(value: unknown, label: string): UnknownRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as UnknownRecord;
}

function number(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function zero(value: unknown, label: string): void {
  if (number(value, label) !== 0) throw new Error(`${label} must be zero`);
}

export function validatePackagedVisualizationEvidence(
  input: unknown,
  expected: EvidenceExpectation,
): UnknownRecord {
  const report = record(input, "report");
  if (report.schema !== SCHEMA || report.schemaVersion !== 1 || report.status !== "passed") {
    throw new Error("report schema/version/status is not a passed v1 qualification");
  }
  if (report.profile !== expected.profile) {
    throw new Error(`report profile ${String(report.profile)} does not match ${expected.profile}`);
  }
  const runtime = record(report.packagedRuntime, "packagedRuntime");
  if (runtime.buildMode !== expected.mode) {
    throw new Error(`packaged build mode ${String(runtime.buildMode)} does not match ${expected.mode}`);
  }
  if (expected.targetOs && runtime.targetOs !== expected.targetOs) {
    throw new Error(`packaged target OS ${String(runtime.targetOs)} does not match ${expected.targetOs}`);
  }
  if (expected.targetArch && runtime.targetArch !== expected.targetArch) {
    throw new Error(`packaged target arch ${String(runtime.targetArch)} does not match ${expected.targetArch}`);
  }
  if (typeof runtime.appVersion !== "string" || runtime.appVersion.length === 0) {
    throw new Error("packaged app version is missing");
  }
  if (!["scheduler-yield", "message-channel", "microtask"].includes(
    String(report.inputHandoffYieldMode),
  )) {
    throw new Error("packaged input-handoff yield mode is missing or invalid");
  }

  const python = report.python == null ? null : record(report.python, "python");
  if (expected.requirePython && !python) throw new Error("native Python evidence was required but is missing");
  const requiredChecks = new Set([...REQUIRED_CHECKS, ...(python ? PYTHON_CHECKS : [])]);
  if (python) {
    if (!Array.isArray(python.checks) || python.checks.length !== PYTHON_CHECKS.length
        || number(python.elapsedMs, "python.elapsedMs") < 0 || typeof python.crossOriginIsolated !== "boolean") {
      throw new Error("native Python evidence has invalid checks or runtime metadata");
    }
    const names = new Set<string>();
    for (const value of python.checks) {
      const check = record(value, "python.check");
      if (typeof check.name !== "string" || check.passed !== true || names.has(check.name)
          || typeof check.detail !== "string" || !check.detail) throw new Error("native Python check is invalid, repeated, or failed");
      names.add(check.name);
    }
    if (PYTHON_CHECKS.some((name) => !names.has(name))) throw new Error("native Python evidence is incomplete");
  }
  if (expected.configPreflightBatch !== undefined || report.configPreflight != null) {
    const batch = record(expected.configPreflightBatch, "required configuration batch");
    const evidence = record(report.configPreflight, "configuration evidence");
    if (batch.schema !== "nimbus.studio.packaged-config-qualification.v1" ||
        typeof batch.binarySha256 !== "string" || !/^[a-f0-9]{64}$/.test(batch.binarySha256) ||
        evidence.schema !== batch.schema || evidence.binarySha256 !== batch.binarySha256 ||
        !Array.isArray(batch.cases) || batch.cases.length === 0 || batch.cases.length > 128 ||
        !Array.isArray(evidence.reports) || evidence.reports.length !== batch.cases.length ||
        !Array.isArray(evidence.checks) || evidence.checks.length !== batch.cases.length) {
      throw new Error("configuration qualification batch/evidence mismatch");
    }
    const seen = new Set<string>();
    for (const [index, value] of batch.cases.entries()) {
      const test = record(value, "configuration case");
      const row = record(evidence.reports[index], "configuration report");
      const request = record(test.request, "configuration request") as unknown as ConfigPreflightRequest;
      if (typeof test.name !== "string" || !test.name.trim() || seen.has(test.name) ||
          typeof test.expectedAcceptance !== "boolean" || row.name !== test.name ||
          row.expectedAcceptance !== test.expectedAcceptance || request.source?.kind !== "exact_document") {
        throw new Error("invalid or duplicate configuration qualification case");
      }
      seen.add(test.name);
      const preflight = validateConfigPreflightEvidence(validateConfigPreflightReport(row.report), request);
      const sourceSha = createHash("sha256").update(request.source.yamlText).digest("hex");
      const fingerprint = `sha256:${batch.binarySha256}`;
      if (preflight.compatible !== test.expectedAcceptance ||
          preflight.nativeAdmission.state !== (test.expectedAcceptance ? "accepted" : "rejected") ||
          preflight.nativeAdmission.sourceSha256 !== sourceSha ||
          preflight.executable.binaryFingerprint !== fingerprint ||
          preflight.nativeAdmission.validatedBinarySha256 !== fingerprint ||
          preflight.executable.canonicalPath !== evidence.canonicalPath ||
          preflight.executable.catalogFingerprint !== evidence.catalogFingerprint ||
          (batch.executionTarget != null && !preflight.executionTargetFingerprint)) {
        throw new Error(`configuration qualification identity/admission mismatch: ${test.name}`);
      }
      const name = `packaged-config-${test.name}`;
      const check = record(evidence.checks[index], "configuration check");
      if (check.name !== name || check.passed !== true) throw new Error("configuration check mismatch");
      requiredChecks.add(name);
    }
  }
  const checks = report.checks;
  if (!Array.isArray(checks) || checks.length !== requiredChecks.size) {
    throw new Error(`report must contain exactly ${requiredChecks.size} checks`);
  }
  const names = new Set<string>();
  for (const [index, value] of checks.entries()) {
    const check = record(value, `checks[${index}]`);
    if (typeof check.name !== "string" || check.passed !== true) {
      throw new Error(`checks[${index}] is unnamed or failed`);
    }
    if (names.has(check.name)) throw new Error(`duplicate check '${check.name}'`);
    names.add(check.name);
  }
  for (const required of requiredChecks) {
    if (!names.has(required)) throw new Error(`missing required check '${required}'`);
  }

  const shape = report.shape;
  if (!Array.isArray(shape) || shape.length !== 3 || shape.some((value) => !Number.isSafeInteger(value))) {
    throw new Error("report shape must contain three integer dimensions");
  }
  if (expected.profile === "production") {
    if (shape.join("x") !== "256x256x160") throw new Error("production shape must be 256x256x160");
    if (report.scalarBytes !== 40 * 1024 * 1024) throw new Error("production scalar must be exactly 40 MiB");
    if (report.vectorBytes !== 120 * 1024 * 1024) throw new Error("production vector must be exactly 120 MiB");
  }
  if (number(report.maximumWatchdogDelayMs, "maximumWatchdogDelayMs") > 500) {
    throw new Error("UI watchdog delay exceeds 500 ms");
  }
  const watchdog = record(report.watchdog, "watchdog");
  const watchdogInterval = number(watchdog.intervalMs, "watchdog.intervalMs");
  const watchdogEventThreshold = number(watchdog.eventThresholdMs, "watchdog.eventThresholdMs");
  const watchdogMaximum = number(watchdog.maximumDelayMs, "watchdog.maximumDelayMs");
  if (watchdogInterval !== 50 || watchdogEventThreshold !== 25
      || watchdogMaximum !== report.maximumWatchdogDelayMs) {
    throw new Error("watchdog configuration or maximum is inconsistent");
  }
  const maximumByStage = record(watchdog.maximumByStageMs, "watchdog.maximumByStageMs");
  const requiredWatchdogStages = [
    "initialization",
    ...[1, 2].flatMap((cycle) => [
      `cycle-${cycle}-scalar-preparation`,
      `cycle-${cycle}-contour-extraction`,
      `cycle-${cycle}-vector-preparation`,
      `cycle-${cycle}-streamline-extraction`,
      `cycle-${cycle}-worker-shutdown`,
      `cycle-${cycle}-retirement-checks`,
    ]),
    "packaged-appearance",
    "packaged-native-bricks",
    "physical-gpu-residency",
    "qualification-checks",
  ];
  const stageEntries = Object.entries(maximumByStage);
  for (const stage of requiredWatchdogStages) {
    if (!(stage in maximumByStage)) throw new Error(`watchdog is missing stage '${stage}'`);
  }
  for (const [stage, delay] of stageEntries) {
    if (!requiredWatchdogStages.includes(stage)
        || number(delay, `watchdog.maximumByStageMs.${stage}`) < 0) {
      throw new Error(`watchdog stage '${stage}' is invalid`);
    }
  }
  const stagedMaximum = Math.max(...stageEntries.map(([stage, delay]) =>
    number(delay, `watchdog.maximumByStageMs.${stage}`)));
  if (stagedMaximum !== watchdogMaximum) {
    throw new Error("watchdog staged maximum is inconsistent");
  }
  if (!Array.isArray(watchdog.events)) throw new Error("watchdog.events must be an array");
  let previousWatchdogEventElapsed = 0;
  for (const [index, value] of watchdog.events.entries()) {
    const event = record(value, `watchdog.events[${index}]`);
    if (typeof event.stage !== "string" || !(event.stage in maximumByStage)) {
      throw new Error(`watchdog.events[${index}] has an unknown stage`);
    }
    if (!["hidden", "visible"].includes(String(event.visibilityState))
        || typeof event.documentHasFocus !== "boolean") {
      throw new Error(`watchdog.events[${index}] has no document scheduling context`);
    }
    const elapsed = number(event.elapsedMs, `watchdog.events[${index}].elapsedMs`);
    const delay = number(event.delayMs, `watchdog.events[${index}].delayMs`);
    if (elapsed < previousWatchdogEventElapsed || delay < watchdogEventThreshold
        || delay > number(maximumByStage[event.stage],
          `watchdog.maximumByStageMs.${event.stage}`)) {
      throw new Error(`watchdog.events[${index}] is inconsistent`);
    }
    previousWatchdogEventElapsed = elapsed;
  }
  if (!(number(report.elapsedMs, "elapsedMs") > 0)) throw new Error("elapsedMs must be positive");

  if (!Array.isArray(report.cycles) || report.cycles.length !== 2) {
    throw new Error("report must contain exactly two restart cycles");
  }
  report.cycles.forEach((value, cycle) => {
    const result = record(value, `cycles[${cycle}]`);
    if (!(number(result.contourTriangles, `cycles[${cycle}].contourTriangles`) > 0)
        || !(number(result.integratedPoints, `cycles[${cycle}].integratedPoints`) > 0)) {
      throw new Error(`cycle ${cycle + 1} published no geometry`);
    }
    const timings = record(result.timings, `cycles[${cycle}].timings`);
    for (const stage of [
      "scalarDispatchMs", "scalarPreparationMs", "contourExtractionMs",
      "vectorDispatchMs", "vectorPreparationMs", "streamlineExtractionMs",
    ]) {
      if (number(timings[stage], `cycles[${cycle}].timings.${stage}`) < 0) {
        throw new Error(`cycles[${cycle}].timings.${stage} must be nonnegative`);
      }
    }
  });

  const appearance = record(report.packagedAppearance, "packagedAppearance");
  if (appearance.schema !== "nimbus.studio.packaged-vtk-appearance"
      || appearance.schemaVersion !== 1 || !Array.isArray(appearance.scenes)
      || appearance.scenes.length !== 5) {
    throw new Error("packaged appearance schema or scene count is invalid");
  }
  const appearancePlots = new Set<string>();
  for (const [index, value] of appearance.scenes.entries()) {
    const scene = record(value, `packagedAppearance.scenes[${index}]`);
    if (typeof scene.plot !== "string" || appearancePlots.has(scene.plot)) {
      throw new Error(`packagedAppearance.scenes[${index}] has a duplicate or missing plot`);
    }
    appearancePlots.add(scene.plot);
    for (const field of ["pngSha256", "repeatPngSha256"]) {
      if (!/^[a-f0-9]{64}$/.test(String(scene[field]))) {
        throw new Error(`packagedAppearance.scenes[${index}].${field} is invalid`);
      }
    }
    for (const field of ["metadataFingerprint", "repeatMetadataFingerprint"]) {
      if (!/^fnv1a64:[a-f0-9]{16}$/.test(String(scene[field]))) {
        throw new Error(`packagedAppearance.scenes[${index}].${field} is invalid`);
      }
    }
    if (scene.pngSha256 !== scene.repeatPngSha256
        || scene.metadataFingerprint !== scene.repeatMetadataFingerprint) {
      throw new Error(`packagedAppearance.scenes[${index}] is not repeatable`);
    }
    const mean = number(scene.meanLuma, `packagedAppearance.scenes[${index}].meanLuma`);
    const deviation = number(scene.lumaDeviation,
      `packagedAppearance.scenes[${index}].lumaDeviation`);
    if (mean < 0 || deviation < 0 || (mean <= 0.07 && deviation <= 0.025)) {
      throw new Error(`packagedAppearance.scenes[${index}] has no rendered pixel signal`);
    }
    if (!Number.isSafeInteger(scene.visibleLayers) || (scene.visibleLayers as number) < 1) {
      throw new Error(`packagedAppearance.scenes[${index}] has no visible layer`);
    }
    const viewport = record(scene.viewport, `packagedAppearance.scenes[${index}].viewport`);
    for (const field of ["cssWidth", "cssHeight", "backingWidth", "backingHeight", "pixelRatio"]) {
      if (!(number(viewport[field], `packagedAppearance.scenes[${index}].viewport.${field}`) > 0)) {
        throw new Error(`packagedAppearance.scenes[${index}] has an invalid viewport`);
      }
    }
  }
  const expectedAppearancePlots = ["volume", "isosurface", "streamlines", "supercell", "trajectories"];
  if (expectedAppearancePlots.some((plot) => !appearancePlots.has(plot))) {
    throw new Error("packaged appearance is missing a required renderer");
  }

  const registry = record(report.finalRegistry, "finalRegistry");
  for (const field of ["bytes", "handles", "references", "mainBytes", "mainHandles", "workerBytes", "workerHandles"]) {
    zero(registry[field], `finalRegistry.${field}`);
  }
  const resources = record(report.finalResources, "finalResources");
  for (const field of ["allocations", "reservations", "residentBytes", "reservedBytes", "committedBytes"]) {
    zero(resources[field], `finalResources.${field}`);
  }
  const prediction = record(resources.prediction, "finalResources.prediction");
  zero(prediction.underpredictions, "finalResources.prediction.underpredictions");

  const nativeBrickEvidence = record(report.nativeBrickEvidence, "nativeBrickEvidence");
  if (nativeBrickEvidence.volumeBricks !== 8 || nativeBrickEvidence.volumeSamples !== 1_568
      || nativeBrickEvidence.anomalyBricks !== 8
      || !(number(nativeBrickEvidence.excludedParentSamples,
        "nativeBrickEvidence.excludedParentSamples") > 0)
      || nativeBrickEvidence.vectorBricks !== 8
      || !(number(nativeBrickEvidence.integratedPoints, "nativeBrickEvidence.integratedPoints") > 0)
      || nativeBrickEvidence.terrainFollowing !== true
      || nativeBrickEvidence.terrainGeometryWindows !== 24
      || !(number(nativeBrickEvidence.terrainDiagnosticTriangles,
        "nativeBrickEvidence.terrainDiagnosticTriangles") > 0)
      || !Array.isArray(nativeBrickEvidence.terrainDiagnosticRange)
      || nativeBrickEvidence.terrainDiagnosticRange.length !== 2
      || !(number(nativeBrickEvidence.terrainDiagnosticRange[1],
        "nativeBrickEvidence.terrainDiagnosticRange[1]")
          > number(nativeBrickEvidence.terrainDiagnosticRange[0],
            "nativeBrickEvidence.terrainDiagnosticRange[0]"))) {
    throw new Error("packaged native-brick evidence is incomplete");
  }
  const physicalGpu = record(report.physicalGpuEvidence, "physicalGpuEvidence");
  const expectedGpuTextureBytes = number(report.scalarBytes, "scalarBytes") * 2;
  if (physicalGpu.cycles !== 2
      || number(physicalGpu.scalarTextureBytesPerCycle,
        "physicalGpuEvidence.scalarTextureBytesPerCycle") !== expectedGpuTextureBytes
      || number(physicalGpu.maximumLogicalGpuResidentBytes,
        "physicalGpuEvidence.maximumLogicalGpuResidentBytes") < expectedGpuTextureBytes
      || physicalGpu.actualRenderedTexturesPerCycle !== 2
      || physicalGpu.contextLost !== false
      || !Array.isArray(physicalGpu.glErrors)
      || physicalGpu.glErrors.length !== 2
      || physicalGpu.glErrors.some((value) => value !== 0)
      || number(physicalGpu.maximum3DTextureSize,
        "physicalGpuEvidence.maximum3DTextureSize") < Math.max(...(report.shape as number[]))
      || !Array.isArray(physicalGpu.renderPngSha256)
      || physicalGpu.renderPngSha256.length !== 2
      || physicalGpu.renderPngSha256.some((value) => !/^[a-f0-9]{64}$/.test(String(value)))
      || number(physicalGpu.maximumPhysicalResidentDeltaBytes,
        "physicalGpuEvidence.maximumPhysicalResidentDeltaBytes") < 0
      || number(physicalGpu.secondResidentDeltaBytes,
        "physicalGpuEvidence.secondResidentDeltaBytes")
          > number(physicalGpu.firstResidentDeltaBytes,
            "physicalGpuEvidence.firstResidentDeltaBytes")) {
    throw new Error("physical GPU-residency evidence is incomplete");
  }

  if (!Array.isArray(report.memorySamples) || report.memorySamples.length !== 19) {
    throw new Error("report must contain exactly 19 lifecycle memory samples");
  }
  let priorPeak = 0;
  for (const [index, value] of report.memorySamples.entries()) {
    const sample = record(value, `memorySamples[${index}]`);
    if (typeof sample.label !== "string" || sample.label.length === 0) {
      throw new Error(`memorySamples[${index}] has no label`);
    }
    if (number(sample.samplingMs, `memorySamples[${index}].samplingMs`) < 0) {
      throw new Error(`memorySamples[${index}].samplingMs must be nonnegative`);
    }
    const memory = record(sample.memory, `memorySamples[${index}].memory`);
    if (memory.schema !== "nimbus.studio.process-memory" || memory.schemaVersion !== 3
        || !["whole_application", "host_plus_descendants"].includes(
          String(memory.processTreeCoverage))) {
      throw new Error(`memorySamples[${index}] has insufficient process-tree coverage`);
    }
    if (!(number(memory.currentRssBytes,
      `memorySamples[${index}].memory.currentRssBytes`) > 0)) {
      throw new Error(`memorySamples[${index}].memory.currentRssBytes must be positive`);
    }
    for (const field of [
      "peakRssBytes", "processTreeRssBytes", "processTreePeakRssBytes", "processTreeProcessCount",
    ]) {
      if (!(number(memory[field], `memorySamples[${index}].memory.${field}`) > 0)) {
        throw new Error(`memorySamples[${index}].memory.${field} must be positive`);
      }
    }
    const peak = number(memory.processTreePeakRssBytes,
      `memorySamples[${index}].memory.processTreePeakRssBytes`);
    if (peak < priorPeak) throw new Error("process-tree peak is not monotonic");
    priorPeak = peak;
  }
  if (expected.targetOs === "macos" && report.memorySamples.some((value, index) =>
    record(value, `memorySamples[${index}]`).memory == null
      || record(record(value, `memorySamples[${index}]`).memory,
        `memorySamples[${index}].memory`).processTreeCoverage !== "whole_application"
      || number(record(record(value, `memorySamples[${index}]`).memory,
        `memorySamples[${index}].memory`).processTreeProcessCount,
      `memorySamples[${index}].memory.processTreeProcessCount`) < 4)) {
    throw new Error("macOS packaged evidence requires whole-application resource-coalition coverage");
  }
  if (expected.targetOs === "linux" && report.memorySamples.some((value, index) => {
    const memory = record(record(value, `memorySamples[${index}]`).memory,
      `memorySamples[${index}].memory`);
    return memory.processTreeCoverage !== "whole_application"
      || number(memory.processTreeProcessCount,
        `memorySamples[${index}].memory.processTreeProcessCount`) < 2;
  })) {
    throw new Error("Linux packaged evidence requires isolated whole-application session coverage");
  }
  if (expected.targetOs === "windows" && report.memorySamples.some((value, index) => {
    const memory = record(record(value, `memorySamples[${index}]`).memory,
      `memorySamples[${index}].memory`);
    return memory.processTreeCoverage !== "whole_application"
      || number(memory.processTreeProcessCount,
        `memorySamples[${index}].memory.processTreeProcessCount`) < 2;
  })) {
    throw new Error("Windows packaged evidence requires whole-application Job Object coverage");
  }
  const reconciliation = record(report.memoryReconciliation, "memoryReconciliation");
  if (!["whole_application", "host_plus_descendants"].includes(String(reconciliation.coverage))) {
    throw new Error("memory reconciliation coverage is insufficient");
  }
  const maximumSampling = Math.max(...report.memorySamples.map((value, index) =>
    number(record(value, `memorySamples[${index}]`).samplingMs,
      `memorySamples[${index}].samplingMs`)));
  if (number(reconciliation.maximumSamplingMs, "memoryReconciliation.maximumSamplingMs")
      !== maximumSampling) {
    throw new Error("memory reconciliation maximum sampling duration is inconsistent");
  }
  const baseline = number(reconciliation.baselineTreeRssBytes,
    "memoryReconciliation.baselineTreeRssBytes");
  const currentHigh = number(reconciliation.processTreeCurrentHighBytes,
    "memoryReconciliation.processTreeCurrentHighBytes");
  const delta = number(reconciliation.processTreeDeltaBytes,
    "memoryReconciliation.processTreeDeltaBytes");
  if (currentHigh < baseline || delta !== currentHigh - baseline) {
    throw new Error("memory reconciliation process-tree delta is inconsistent");
  }
  if (number(reconciliation.processTreePeakBytes, "memoryReconciliation.processTreePeakBytes")
      < currentHigh) {
    throw new Error("memory reconciliation peak is below sampled current high");
  }
  const synthetic = number(reconciliation.syntheticInputBytes,
    "memoryReconciliation.syntheticInputBytes");
  if (synthetic !== number(report.scalarBytes, "scalarBytes") + number(report.vectorBytes, "vectorBytes")) {
    throw new Error("memory reconciliation input bytes are inconsistent");
  }
  const residentHigh = number(reconciliation.trackedResidentHighWaterBytes,
    "memoryReconciliation.trackedResidentHighWaterBytes");
  const committedHigh = number(reconciliation.trackedCommittedHighWaterBytes,
    "memoryReconciliation.trackedCommittedHighWaterBytes");
  if (residentHigh !== number(resources.residentHighWaterBytes, "finalResources.residentHighWaterBytes")
      || committedHigh !== number(resources.highWaterBytes, "finalResources.highWaterBytes")) {
    throw new Error("memory reconciliation does not match the visualization ledger");
  }
  const accounted = number(reconciliation.accountedInputAndResidentHighWaterBytes,
    "memoryReconciliation.accountedInputAndResidentHighWaterBytes");
  if (accounted !== synthetic + residentHigh
      || number(reconciliation.observedMinusAccountedBytes,
        "memoryReconciliation.observedMinusAccountedBytes") !== delta - accounted) {
    throw new Error("memory reconciliation accounted/unaccounted arithmetic is inconsistent");
  }
  return report;
}

function cliArguments(args: string[]): { path: string; expected: EvidenceExpectation } {
  const path = args[0];
  if (!path) throw new Error("usage: validate_packaged_visualization_evidence.ts REPORT --profile P --mode M [--target-os OS] [--target-arch ARCH]");
  const values = new Map<string, string>();
  for (let index = 1; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || value == null) throw new Error(`invalid argument '${key ?? ""}'`);
    values.set(key.slice(2), value);
  }
  if (values.has("require-python") && !["0", "1"].includes(values.get("require-python")!)) throw new Error("--require-python must be 0 or 1");
  const profile = values.get("profile");
  const mode = values.get("mode");
  if (profile !== "smoke" && profile !== "production") throw new Error("--profile must be smoke or production");
  if (mode !== "debug" && mode !== "release") throw new Error("--mode must be debug or release");
  return {
    path,
    expected: {
      profile,
      mode,
      targetOs: values.get("target-os"),
      targetArch: values.get("target-arch"),
      requirePython: values.get("require-python") === "1",
      configPreflightBatch: values.get("config-preflight")
        ? JSON.parse(readFileSync(values.get("config-preflight")!, "utf8")) : undefined,
    },
  };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  try {
    const { path, expected } = cliArguments(process.argv.slice(2));
    const report = validatePackagedVisualizationEvidence(
      JSON.parse(readFileSync(path, "utf8")), expected,
    );
    console.log(JSON.stringify({
      status: report.status,
      profile: report.profile,
      packagedRuntime: report.packagedRuntime,
      shape: report.shape,
      checks: (report.checks as unknown[]).length,
      ...(report.python ? { pythonElapsedMs: record(report.python, "python").elapsedMs } : {}),
      elapsedMs: report.elapsedMs,
      maximumWatchdogDelayMs: report.maximumWatchdogDelayMs,
    }, null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}
