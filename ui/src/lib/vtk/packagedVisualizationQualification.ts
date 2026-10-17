import { invokeCommand } from "@/lib/platform/appInvoke";
import { getVisualizationResourceSnapshot } from "@/lib/core/visualizationResourceGovernor";
import { preparedDatasetRegistrySnapshot, resolvePreparedDataset } from "./vtkPreparedDatasetRegistry";
import {
  extractContoursFromWorkerOwnedHandles,
  prepareScalarInWorkerOwnedService,
  shutdownWorkerOwnedContourServices,
} from "./contourWorkerOwnedPreparation";
import {
  extractStreamlinesFromWorkerOwnedHandle,
  prepareVectorInWorkerOwnedService,
  shutdownWorkerOwnedStreamlineService,
} from "./streamlineWorkerOwnedPreparation";
import type { Npy3D } from "@/lib/platform/npy";
import type { GridSpacing } from "@/lib/science/cartesianInterp";
import { getProcessMemoryInfo } from "@/lib/platform/tauri";
import type { ProcessMemoryInfo } from "@/lib/core/types";
import { vtkPreparationInputYieldMode } from "./vtkPreparationInputHandoff";
import {
  capturePackagedAppearanceEvidence,
  qualifyPackagedPhysicalGpuResidency,
} from "./packagedAppearanceQualification";
import {
  listSteps,
  clearFieldCache,
  readField,
  readFieldBrickHierarchy,
  readFieldWindows,
} from "@/lib/run/runOutputs";
import {
  planAllContourBricks,
  planVectorBrickNeighborhood,
  planVolumeBricks,
} from "@/lib/plot/fieldBrickHierarchy";
import { VolumeBrickPreparationClient } from "./volumeBrickClient";
import { BrickedContourClient } from "./brickedContourClient";
import { MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES } from "@/lib/dataset/nativeFieldWindow";

const REPORT_SCHEMA = "nimbus.studio.packaged-visualization-qualification" as const;

export interface PackagedVisualizationQualificationRequest {
  schema: "nimbus.studio.packaged-visualization-qualification-request";
  schemaVersion: 1;
  profile: string;
  python?: boolean;
  configPreflight?: unknown;
  visualLoad?: { runDir?: string; specification?: Record<string, unknown>; specificationSha256?: string; error?: string };
}

interface Check {
  name: string;
  passed: boolean;
  detail: string;
}

interface MemorySample {
  label: string;
  elapsedMs: number;
  samplingMs: number;
  memory: ProcessMemoryInfo;
}

interface WatchdogEvent {
  stage: string;
  elapsedMs: number;
  delayMs: number;
  visibilityState: DocumentVisibilityState;
  documentHasFocus: boolean;
}

interface WatchdogDiagnostics {
  intervalMs: number;
  eventThresholdMs: number;
  maximumDelayMs: number;
  maximumByStageMs: Record<string, number>;
  events: WatchdogEvent[];
}

interface NativeBrickFixture {
  runDir: string;
  stepIndex: number;
  publicationGeneration: string;
  scalarField: string;
  vectorFields: [string, string, string];
  terrainFollowing: boolean;
}

async function qualifyPackagedNativeBricks(
  sampleMemory: (label: string) => Promise<void>,
): Promise<{
  volumeBricks: number;
  volumeSamples: number;
  anomalyBricks: number;
  excludedParentSamples: number;
  vectorBricks: number;
  integratedPoints: number;
  terrainFollowing: boolean;
  terrainGeometryWindows: number;
  terrainDiagnosticTriangles: number;
  terrainDiagnosticRange: [number, number];
}> {
  const fixture = await invokeCommand<NativeBrickFixture>(
    "create_visualization_brick_qualification_fixture",
  );
  const client = new VolumeBrickPreparationClient();
  const controller = new AbortController();
  const contourClient = new BrickedContourClient();
  let vectorLease: Awaited<ReturnType<typeof prepareVectorInWorkerOwnedService>> | null = null;
  try {
    const steps = await listSteps(fixture.runDir);
    const step = steps.find((candidate) => candidate.index === fixture.stepIndex);
    if (!step?.grid || step.publicationGeneration !== fixture.publicationGeneration) {
      throw new Error(
        `Packaged native-brick fixture did not reopen through ordinary step discovery: ${JSON.stringify(steps)}`,
      );
    }
    const fixtureGrid = step.grid;
    if (!fixture.terrainFollowing || !fixtureGrid.terrainFollowing) {
      throw new Error("Packaged native-brick fixture did not reopen as terrain-following");
    }
    const scalarHierarchy = await readFieldBrickHierarchy(
      fixture.runDir,
      step,
      fixture.scalarField,
    );
    const volumePlan = planVolumeBricks(scalarHierarchy, [[-1, 1], [1, 1]], {
      maximumResidentBytes: 64 * 1024 * 1024,
      terrainFollowing: true,
    });
    const volumeFields = [];
    for (const item of volumePlan.items) {
      const field = await readField(
        fixture.runDir,
        step.index,
        fixture.scalarField,
        step.publicationGeneration,
        {
          window: item.window,
          ownership: "transferable",
          cacheRetention: "transient",
        },
      );
      volumeFields.push(field);
    }
    const preparedVolume = await client.prepareAnomalyBatch(volumeFields.map((field) => ({
        kind: "scalar",
        field,
        grid: fixtureGrid,
        refineForDisplay: false,
        scalarTransform: "horizontal-level-anomaly",
        excludePhysicalBoundsM: {
          lower: [400, 400, 200],
          upper: [1_200, 1_200, 600],
        },
        datasetHandoff: "render-only",
      })), controller.signal);
    const vectorHierarchies = await Promise.all(fixture.vectorFields.map((field) =>
      readFieldBrickHierarchy(fixture.runDir, step, field)));
    const diagnosticPlan = planAllContourBricks(vectorHierarchies[0], {
      level: 0,
      terrainFollowing: true,
    });
    await contourClient.reset(`packaged-terrain-diagnostic:${fixture.publicationGeneration}`, controller.signal);
    await contourClient.admitBricks(diagnosticPlan.items.map((item) => {
      const outputBytes = item.readWindow.maxOutputBytes!;
      const inputBytes = Math.max(outputBytes, item.readWindow.maxSourceBytes ?? 0);
      return {
        brickId: item.brickId,
        inputBytes: (inputBytes + MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES) * 3,
        outputBytes: outputBytes + item.readWindow.extent[2] * 3 * Float32Array.BYTES_PER_ELEMENT,
      };
    }), controller.signal);
    let terrainGeometryWindows = 0;
    let terrainDiagnosticMinimum = Number.POSITIVE_INFINITY;
    let terrainDiagnosticMaximum = Number.NEGATIVE_INFINITY;
    for (const item of diagnosticPlan.items) {
      const fields = await Promise.all(fixture.vectorFields.map((field) => readField(
        fixture.runDir, step.index, field, step.publicationGeneration, {
          window: item.readWindow,
          ownership: "transferable",
          cacheRetention: "transient",
        },
      ))) as [Npy3D, Npy3D, Npy3D];
      terrainGeometryWindows += fields.filter((field) =>
        field.windowGeometry?.weighting === "physical_cell_volume"
          && field.windowGeometry.centroidsM != null
          && field.windowGeometry.volumesM3 != null).length;
      const readOrigin = item.readWindow.origin.map((value, axis) =>
        value - vectorHierarchies[0].sourceIndexOffset[axis]);
      const lower = item.ownedCellOrigin.map((value, axis) => value - readOrigin[axis]);
      const upper = lower.map((value, axis) => value + item.ownedCellExtent[axis]);
      await contourClient.loadBrick(item.brickId, {
        kind: "velocity-diagnostic",
        diagnostic: "q-criterion",
        fields,
        grid: fixtureGrid,
        refineForDisplay: false,
        datasetHandoff: "shared-geometry",
      }, controller.signal, [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]);
      const stats = contourClient.brickStats(item.brickId);
      if (stats) {
        terrainDiagnosticMinimum = Math.min(terrainDiagnosticMinimum, stats.min);
        terrainDiagnosticMaximum = Math.max(terrainDiagnosticMaximum, stats.max);
      }
    }
    if (!(terrainDiagnosticMaximum > terrainDiagnosticMinimum)) {
      throw new Error(`Packaged terrain diagnostic has no finite span (${terrainDiagnosticMinimum}, ${terrainDiagnosticMaximum})`);
    }
    const terrainDiagnosticThreshold = terrainDiagnosticMinimum
      + 0.5 * (terrainDiagnosticMaximum - terrainDiagnosticMinimum);
    const terrainDiagnosticChunks = await contourClient.extract(
      diagnosticPlan.items.map((item) => item.brickId), terrainDiagnosticThreshold, controller.signal,
    );
    const terrainDiagnosticTriangles = terrainDiagnosticChunks.reduce(
      (sum, chunk) => sum + (chunk.geometry?.triangles ?? 0), 0,
    );
    const vectorPlan = planVectorBrickNeighborhood(
      vectorHierarchies as [typeof vectorHierarchies[0], typeof vectorHierarchies[0], typeof vectorHierarchies[0]],
      { maximumPreparedCells: 1_000_000, brickRadius: 1, terrainFollowing: true },
    );
    const windows = vectorPlan.items.map((item) => item.window);
    const vectors = await Promise.all(fixture.vectorFields.map((field, component) =>
      readFieldWindows(
        fixture.runDir,
        step.index,
        field,
        step.publicationGeneration,
        windows,
        { latestScope: `packaged-native-vector:${component}` },
      ))) as [Npy3D, Npy3D, Npy3D];
    vectorLease = await prepareVectorInWorkerOwnedService({
      kind: "vector",
      fields: vectors,
      grid: fixtureGrid,
      maxVoxels: vectors[0].data.length,
      datasetHandoff: "shared-geometry",
    });
    const streamlines = await extractStreamlinesFromWorkerOwnedHandle(vectorLease.handle, {
      seeds: new Float32Array([800, 800, 400, 650, 800, 300, 950, 800, 500]),
      integrationStep: 1,
      maxSteps: 80,
      tubeRadius: 25,
      scale: { xy: 1, z: 1 },
    });
    await sampleMemory("packaged-native-bricks-resident");
    return {
      volumeBricks: preparedVolume.length,
      volumeSamples: preparedVolume.reduce((sum, prepared) => sum + prepared.values.length, 0),
      anomalyBricks: preparedVolume.filter((prepared) =>
        prepared.scalarTransform === "horizontal-level-anomaly").length,
      excludedParentSamples: preparedVolume.reduce((sum, prepared) =>
        sum + prepared.values.reduce((count, value) => count + (value === 0 ? 1 : 0), 0), 0),
      vectorBricks: vectorPlan.items.length,
      integratedPoints: streamlines.integratedPoints,
      terrainFollowing: fixtureGrid.terrainFollowing,
      terrainGeometryWindows,
      terrainDiagnosticTriangles,
      terrainDiagnosticRange: [terrainDiagnosticMinimum, terrainDiagnosticMaximum],
    };
  } finally {
    controller.abort();
    vectorLease?.release();
    client.shutdown();
    contourClient.shutdown();
    shutdownWorkerOwnedStreamlineService();
    clearFieldCache();
    await invokeCommand("remove_visualization_brick_qualification_fixture").catch(() => undefined);
  }
}

let activeFailureContext: {
  checks: Check[];
  memorySamples: MemorySample[];
  watchdog: WatchdogDiagnostics;
} | null = null;

function failureContext() {
  return activeFailureContext;
}

function shapeFor(profile: string): [number, number, number] {
  if (profile === "production") return [256, 256, 160];
  if (profile === "smoke") return [48, 48, 32];
  throw new Error(`Unknown packaged visualization qualification profile '${profile}'`);
}

function scalarField(shape: [number, number, number]): Npy3D {
  const [nx, ny, nz] = shape;
  const data = new Float32Array(nx * ny * nz);
  let target = 0;
  for (let x = 0; x < nx; x += 1) {
    const dx = (x - (nx - 1) * 0.5) / Math.max(1, nx * 0.24);
    for (let y = 0; y < ny; y += 1) {
      const dy = (y - (ny - 1) * 0.5) / Math.max(1, ny * 0.24);
      for (let z = 0; z < nz; z += 1) {
        const dz = (z - (nz - 1) * 0.45) / Math.max(1, nz * 0.3);
        data[target++] = Math.exp(-(dx * dx + dy * dy + dz * dz) * 2.8) - 0.2;
      }
    }
  }
  return { shape, data };
}

function vectorFields(shape: [number, number, number]): [Npy3D, Npy3D, Npy3D] {
  const [nx, ny, nz] = shape;
  const elements = nx * ny * nz;
  const u = new Float32Array(elements);
  const v = new Float32Array(elements);
  const w = new Float32Array(elements);
  let target = 0;
  for (let x = 0; x < nx; x += 1) {
    const dx = (x - (nx - 1) * 0.5) / Math.max(1, nx);
    for (let y = 0; y < ny; y += 1) {
      const dy = (y - (ny - 1) * 0.5) / Math.max(1, ny);
      for (let z = 0; z < nz; z += 1) {
        const envelope = Math.max(0.15, 1 - z / Math.max(1, nz - 1));
        u[target] = -dy * 18 * envelope;
        v[target] = dx * 18 * envelope;
        w[target] = 0.25 * envelope;
        target += 1;
      }
    }
  }
  return [{ shape, data: u }, { shape, data: v }, { shape, data: w }];
}

function cannotResolveOnMain(handle: Parameters<typeof resolvePreparedDataset>[0]): boolean {
  try {
    resolvePreparedDataset(handle);
    return false;
  } catch {
    return true;
  }
}

function requireCheck(checks: Check[], name: string, condition: boolean, detail: string): void {
  checks.push({ name, passed: condition, detail });
  if (!condition) throw new Error(`${name}: ${detail}`);
}

async function runCycle(
  cycle: number,
  scalar: Npy3D,
  vectors: [Npy3D, Npy3D, Npy3D],
  grid: GridSpacing,
  checks: Check[],
  sampleMemory: (label: string) => Promise<void>,
  beginWatchdogStage: (stage: string) => Promise<void>,
): Promise<{
  contourTriangles: number;
  integratedPoints: number;
  timings: Record<string, number>;
}> {
  const timings: Record<string, number> = {};
  await beginWatchdogStage(`cycle-${cycle}-scalar-preparation`);
  let stageStarted = performance.now();
  const scalarPreparation = prepareScalarInWorkerOwnedService({
    kind: "scalar",
    field: scalar,
    grid,
    maxVoxels: scalar.data.length,
    featureSign: 1,
    datasetHandoff: "shared-geometry",
  }, undefined, `packaged-qualification-scalar:${cycle}`);
  timings.scalarDispatchMs = performance.now() - stageStarted;
  const scalarLease = await scalarPreparation;
  timings.scalarPreparationMs = performance.now() - stageStarted;
  await sampleMemory(`cycle-${cycle}-scalar-resident`);
  requireCheck(checks, `cycle-${cycle}-scalar-worker-handle`,
    scalarLease.handle.residency === "worker", "scalar preparation must remain worker-owned");
  requireCheck(checks, `cycle-${cycle}-scalar-main-resolution`,
    cannotResolveOnMain(scalarLease.handle), "worker scalar must fail closed on main");
  await beginWatchdogStage(`cycle-${cycle}-contour-extraction`);
  stageStarted = performance.now();
  const contour = await extractContoursFromWorkerOwnedHandles(scalarLease.handle, null, {
    thresholds: { core: 0.25, opposite: null, mesocyclone: null },
    anchor: null,
    condensateFloor: null,
  });
  timings.contourExtractionMs = performance.now() - stageStarted;
  await sampleMemory(`cycle-${cycle}-contour-published`);
  requireCheck(checks, `cycle-${cycle}-contour-geometry`,
    contour.core != null && contour.core.triangles > 0,
    `worker contour must publish triangles (got ${contour.core?.triangles ?? 0})`);
  const contourTriangles = contour.core?.triangles ?? 0;
  scalarLease.release();

  await beginWatchdogStage(`cycle-${cycle}-vector-preparation`);
  stageStarted = performance.now();
  const vectorPreparation = prepareVectorInWorkerOwnedService({
    kind: "vector",
    fields: vectors,
    grid,
    maxVoxels: vectors[0].data.length,
    datasetHandoff: "shared-geometry",
  });
  timings.vectorDispatchMs = performance.now() - stageStarted;
  const vectorLease = await vectorPreparation;
  timings.vectorPreparationMs = performance.now() - stageStarted;
  await sampleMemory(`cycle-${cycle}-scalar-vector-resident`);
  requireCheck(checks, `cycle-${cycle}-vector-worker-handle`,
    vectorLease.handle.residency === "worker", "vector preparation must remain worker-owned");
  requireCheck(checks, `cycle-${cycle}-vector-main-resolution`,
    cannotResolveOnMain(vectorLease.handle), "worker vector must fail closed on main");
  const [nx, ny, nz] = vectors[0].shape;
  const seeds = new Float32Array([
    nx * 50, ny * 50, nz * 50,
    nx * 45, ny * 50, nz * 35,
    nx * 55, ny * 50, nz * 65,
  ]);
  await beginWatchdogStage(`cycle-${cycle}-streamline-extraction`);
  stageStarted = performance.now();
  const streamlines = await extractStreamlinesFromWorkerOwnedHandle(vectorLease.handle, {
    seeds,
    integrationStep: 0.5,
    maxSteps: 80,
    tubeRadius: 25,
    scale: { xy: 1, z: 1 },
  });
  timings.streamlineExtractionMs = performance.now() - stageStarted;
  await sampleMemory(`cycle-${cycle}-streamline-published`);
  requireCheck(checks, `cycle-${cycle}-streamline-geometry`,
    streamlines.integratedPoints > 0,
    `worker streamline extraction must integrate points (got ${streamlines.integratedPoints})`);
  vectorLease.release();
  return { contourTriangles, integratedPoints: streamlines.integratedPoints, timings };
}

async function execute(profile: string) {
  const checks: Check[] = [];
  const started = performance.now();
  const memorySamples: MemorySample[] = [];
  const watchdogDiagnostics: WatchdogDiagnostics = {
    intervalMs: 50,
    eventThresholdMs: 25,
    maximumDelayMs: 0,
    maximumByStageMs: {},
    events: [],
  };
  activeFailureContext = { checks, memorySamples, watchdog: watchdogDiagnostics };
  let previousWatchdog = performance.now();
  let watchdogEnabled = false;
  let watchdogStage = "initialization";
  const observeWatchdog = (now: number) => {
    if (!watchdogEnabled) {
      previousWatchdog = now;
      return;
    }
    const delayMs = Math.max(0, now - previousWatchdog - watchdogDiagnostics.intervalMs);
    watchdogDiagnostics.maximumDelayMs = Math.max(
      watchdogDiagnostics.maximumDelayMs,
      delayMs,
    );
    watchdogDiagnostics.maximumByStageMs[watchdogStage] = Math.max(
      watchdogDiagnostics.maximumByStageMs[watchdogStage] ?? 0,
      delayMs,
    );
    if (delayMs >= watchdogDiagnostics.eventThresholdMs) {
      watchdogDiagnostics.events.push({
        stage: watchdogStage,
        elapsedMs: now - started,
        delayMs,
        visibilityState: document.visibilityState,
        documentHasFocus: document.hasFocus(),
      });
    }
    previousWatchdog = now;
  };
  const setWatchdogStage = (stage: string) => {
    observeWatchdog(performance.now());
    watchdogStage = stage;
  };
  const recordProgress = async (stage: string) => {
    observeWatchdog(performance.now());
    const resumeWatchdog = watchdogEnabled;
    watchdogEnabled = false;
    try {
      await invokeCommand("record_visualization_qualification_progress", { stage })
        .catch(() => undefined);
    } finally {
      previousWatchdog = performance.now();
      watchdogEnabled = resumeWatchdog;
    }
  };
  const beginWatchdogStage = async (stage: string) => {
    setWatchdogStage(stage);
    await recordProgress(stage);
  };
  const sampleMemory = async (label: string) => {
    const priorStage = watchdogStage;
    setWatchdogStage(`memory:${label}`);
    const resumeWatchdog = watchdogEnabled;
    watchdogEnabled = false;
    try {
      await invokeCommand("record_visualization_qualification_progress", {
        stage: `memory:${label}:start`,
      }).catch(() => undefined);
      const sampleStarted = performance.now();
      const memory = await getProcessMemoryInfo();
      if (!memory) throw new Error(`process memory unavailable at ${label}`);
      const now = performance.now();
      memorySamples.push({
        label,
        elapsedMs: now - started,
        samplingMs: now - sampleStarted,
        memory,
      });
      await invokeCommand("record_visualization_qualification_progress", {
        stage: `memory:${label}:complete`,
      }).catch(() => undefined);
    } finally {
      previousWatchdog = performance.now();
      watchdogStage = priorStage;
      watchdogEnabled = resumeWatchdog;
    }
  };
  await recordProgress("qualification-execute");
  const shape = shapeFor(profile);
  requireCheck(checks, "module-worker-api", "Worker" in globalThis,
    "the packaged WebView must expose module workers");
  await sampleMemory("startup");
  const setupStarted = performance.now();
  const scalar = scalarField(shape);
  const vectors = vectorFields(shape);
  const setupMs = performance.now() - setupStarted;
  const scalarBytes = scalar.data.byteLength;
  const vectorBytes = vectors.reduce((bytes, field) => bytes + field.data.byteLength, 0);
  requireCheck(checks, "profile-byte-contract",
    profile !== "production"
      || (scalarBytes === 40 * 1024 * 1024 && vectorBytes === 120 * 1024 * 1024),
    `production profile requires exact 40/120 MiB lattices (got ${scalarBytes}/${vectorBytes} bytes)`);
  await sampleMemory("fixture-allocated");
  const grid: GridSpacing = {
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    drM: 100,
    dyM: 100,
    dzM: 100,
    scalarOriginM: [0, 0, 0],
  };
  previousWatchdog = performance.now();
  watchdogEnabled = true;
  const watchdog = window.setInterval(() => {
    observeWatchdog(performance.now());
  }, watchdogDiagnostics.intervalMs);
  const cycles = [];
  let nativeBrickEvidence: Awaited<ReturnType<typeof qualifyPackagedNativeBricks>> | null = null;
  let physicalGpuEvidence: Awaited<ReturnType<typeof qualifyPackagedPhysicalGpuResidency>> | null = null;
  let packagedAppearance: Awaited<ReturnType<typeof capturePackagedAppearanceEvidence>> | null = null;
  try {
    for (let cycle = 1; cycle <= 2; cycle += 1) {
      cycles.push(await runCycle(
        cycle,
        scalar,
        vectors,
        grid,
        checks,
        sampleMemory,
        beginWatchdogStage,
      ));
      await beginWatchdogStage(`cycle-${cycle}-worker-shutdown`);
      shutdownWorkerOwnedContourServices();
      shutdownWorkerOwnedStreamlineService();
      await Promise.resolve();
      await sampleMemory(`cycle-${cycle}-shutdown`);
      await beginWatchdogStage(`cycle-${cycle}-retirement-checks`);
      const registry = preparedDatasetRegistrySnapshot();
      const resources = getVisualizationResourceSnapshot();
      requireCheck(checks, `cycle-${cycle}-registry-retired`,
        registry.handles === 0 && registry.references === 0,
        `retirement must clear handles/references (got ${registry.handles}/${registry.references})`);
      requireCheck(checks, `cycle-${cycle}-worker-residency-retired`,
        resources.byStage["worker-resident"].residentBytes === 0
          && resources.byStage["worker-resident"].reservedBytes === 0,
        "retirement must release worker residency and reservations");
    }
    await beginWatchdogStage("packaged-native-bricks");
    nativeBrickEvidence = await qualifyPackagedNativeBricks(sampleMemory);
    requireCheck(checks, "packaged-native-brick-volume",
      nativeBrickEvidence.volumeBricks === 8
        && nativeBrickEvidence.volumeSamples === 1_568
        && nativeBrickEvidence.anomalyBricks === 8
        && nativeBrickEvidence.excludedParentSamples > 0,
      `ordinary native windows must prepare the complete 14×14×8 valid volume (got ${JSON.stringify(nativeBrickEvidence)})`);
    requireCheck(checks, "packaged-native-brick-streamlines",
      nativeBrickEvidence.vectorBricks === 8 && nativeBrickEvidence.integratedPoints > 0,
      `ordinary native u/v/w windows must publish trajectory geometry (got ${JSON.stringify(nativeBrickEvidence)})`);
    requireCheck(checks, "packaged-terrain-brick-diagnostics",
      nativeBrickEvidence.terrainFollowing
        && nativeBrickEvidence.terrainGeometryWindows === nativeBrickEvidence.vectorBricks * 3
        && nativeBrickEvidence.terrainDiagnosticTriangles > 0
        && nativeBrickEvidence.terrainDiagnosticRange[1] > nativeBrickEvidence.terrainDiagnosticRange[0],
      `manifest-v5 terrain windows must publish physical geometry and Q contours (got ${JSON.stringify(nativeBrickEvidence)})`);
    await beginWatchdogStage("physical-gpu-residency");
    physicalGpuEvidence = await qualifyPackagedPhysicalGpuResidency(scalar, sampleMemory);
    const gpuBaseline = memorySamples.find((sample) => sample.label === "gpu-baseline")!;
    const gpuResidentOne = memorySamples.find((sample) => sample.label === "gpu-cycle-1-resident")!;
    const gpuRetiredOne = memorySamples.find((sample) => sample.label === "gpu-cycle-1-retired")!;
    const gpuResidentTwo = memorySamples.find((sample) => sample.label === "gpu-cycle-2-resident")!;
    const baselineRss = gpuBaseline.memory.processTreeRssBytes!;
    const firstResidentDeltaBytes = Math.max(
      0,
      gpuResidentOne.memory.processTreeRssBytes! - baselineRss,
    );
    const secondResidentDeltaBytes = Math.max(
      0,
      gpuResidentTwo.memory.processTreeRssBytes! - gpuRetiredOne.memory.processTreeRssBytes!,
    );
    const maximumPhysicalResidentDeltaBytes = Math.max(
      firstResidentDeltaBytes,
      secondResidentDeltaBytes,
    );
    requireCheck(checks, "physical-gpu-residency",
      physicalGpuEvidence.maximumLogicalGpuResidentBytes
        >= physicalGpuEvidence.scalarTextureBytesPerCycle
        && physicalGpuEvidence.actualRenderedTexturesPerCycle === 2
        && !physicalGpuEvidence.contextLost
        && physicalGpuEvidence.glErrors.every((error) => error === 0)
        && physicalGpuEvidence.maximum3DTextureSize >= Math.max(...shape)
        && physicalGpuEvidence.renderPngSha256.length === 2
        && maximumPhysicalResidentDeltaBytes >= 0,
      `two rendered 3D textures must allocate, render, and survive restart (logical ${physicalGpuEvidence.maximumLogicalGpuResidentBytes}, RSS-observed delta ${maximumPhysicalResidentDeltaBytes})`);
    requireCheck(checks, "physical-gpu-restart-plateau",
      secondResidentDeltaBytes <= firstResidentDeltaBytes,
      `the second fresh GPU-cycle RSS delta (${secondResidentDeltaBytes}) must not exceed the first observed allocation delta (${firstResidentDeltaBytes})`);
    physicalGpuEvidence = {
      ...physicalGpuEvidence,
      firstResidentDeltaBytes,
      secondResidentDeltaBytes,
      maximumPhysicalResidentDeltaBytes,
    };
    await beginWatchdogStage("packaged-appearance");
    packagedAppearance = await capturePackagedAppearanceEvidence();
    requireCheck(checks, "packaged-appearance-scenes",
      packagedAppearance.scenes.length === 5
        && packagedAppearance.scenes.every((scene) =>
          scene.visibleLayers > 0
          && scene.viewport.backingWidth > 0
          && scene.viewport.backingHeight > 0
          && (scene.meanLuma > 0.07 || scene.lumaDeviation > 0.025)
        ),
      `packaged WKWebView must render five nonempty VTK scenes with backing stores: ${JSON.stringify(
        packagedAppearance.scenes.map((scene) => ({
          plot: scene.plot,
          meanLuma: scene.meanLuma,
          lumaDeviation: scene.lumaDeviation,
          visibleLayers: scene.visibleLayers,
          viewport: scene.viewport,
        })),
      )}`);
    requireCheck(checks, "packaged-appearance-repeatability",
      packagedAppearance.scenes.every((scene) =>
        scene.pngSha256 === scene.repeatPngSha256
        && scene.metadataFingerprint === scene.repeatMetadataFingerprint
      ),
      "packaged WKWebView scene PNG and metadata captures must repeat exactly");
    await beginWatchdogStage("qualification-checks");
    requireCheck(checks, "ui-thread-watchdog",
      watchdogDiagnostics.maximumDelayMs <= 500,
      `worker execution must keep watchdog delay <= 500 ms (got ${watchdogDiagnostics.maximumDelayMs.toFixed(1)} ms)`);
  } finally {
    observeWatchdog(performance.now());
    watchdogEnabled = false;
    window.clearInterval(watchdog);
  }
  const finalRegistry = preparedDatasetRegistrySnapshot();
  const finalResources = getVisualizationResourceSnapshot();
  await sampleMemory("final");
  const memoryCoverage = memorySamples[0].memory.processTreeCoverage;
  const processTreeCoverage = memorySamples.every(
    (sample) => sample.memory.processTreeCoverage === memoryCoverage
      && sample.memory.processTreeCoverage !== "host_only"
      && sample.memory.processTreeRssBytes != null
      && sample.memory.processTreePeakRssBytes != null,
  );
  const wholeApplicationMembership = memoryCoverage !== "whole_application"
    || memorySamples.every((sample) => (sample.memory.processTreeProcessCount ?? 0) >= 4);
  requireCheck(checks, "process-memory-coverage", processTreeCoverage,
    "every packaged sample must include one stable application RSS coverage contract");
  if (!wholeApplicationMembership) {
    throw new Error("whole-application packaged evidence is missing required UI, WebContent, GPU, or Networking coalition members");
  }
  const sampledPeaks = memorySamples.map((sample) => sample.memory.processTreePeakRssBytes!);
  const monotonicPeaks = sampledPeaks.every(
    (value, index) => index === 0 || value >= sampledPeaks[index - 1],
  );
  requireCheck(checks, "process-memory-monotonic", monotonicPeaks,
    "sampled process-tree peak must remain monotonic across the lifecycle");
  const baselineTreeRssBytes = memorySamples[0].memory.processTreeRssBytes!;
  const processTreeCurrentHighBytes = Math.max(...memorySamples.map(
    (sample) => sample.memory.processTreeRssBytes!,
  ));
  const processTreePeakBytes = Math.max(...sampledPeaks);
  const processTreeDeltaBytes = Math.max(0, processTreeCurrentHighBytes - baselineTreeRssBytes);
  const accountedInputAndResidentHighWaterBytes = scalarBytes + vectorBytes
    + finalResources.residentHighWaterBytes;
  await recordProgress("qualification-complete");
  return {
    schema: REPORT_SCHEMA,
    schemaVersion: 1,
    status: "passed" as const,
    profile,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    inputHandoffYieldMode: vtkPreparationInputYieldMode(),
    shape,
    scalarBytes,
    vectorBytes,
    setupMs,
    elapsedMs: performance.now() - started,
    maximumWatchdogDelayMs: watchdogDiagnostics.maximumDelayMs,
    watchdog: watchdogDiagnostics,
    cycles,
    nativeBrickEvidence,
    physicalGpuEvidence,
    packagedAppearance,
    checks,
    memorySamples,
    memoryReconciliation: {
      coverage: memorySamples[0].memory.processTreeCoverage,
      baselineTreeRssBytes,
      processTreeCurrentHighBytes,
      processTreePeakBytes,
      processTreeDeltaBytes,
      maximumSamplingMs: Math.max(...memorySamples.map((sample) => sample.samplingMs)),
      syntheticInputBytes: scalarBytes + vectorBytes,
      trackedResidentHighWaterBytes: finalResources.residentHighWaterBytes,
      trackedCommittedHighWaterBytes: finalResources.highWaterBytes,
      accountedInputAndResidentHighWaterBytes,
      observedMinusAccountedBytes: processTreeDeltaBytes - accountedInputAndResidentHighWaterBytes,
      limitation: memoryCoverage === "whole_application"
        ? "macOS resource-coalition membership covers the UI process and its associated WebKit/XPC services; hardware capacity still requires campaign-derived opaque scratch headroom."
        : "Parent/descendant sampling may exclude detached WKWebView/WebView2/XPC renderer helpers; this is reconciliation evidence, not an enforceable universal capacity.",
    },
    finalRegistry,
    finalResources,
  };
}

export async function runPackagedVisualizationQualification(
  request: PackagedVisualizationQualificationRequest,
): Promise<void> {
  let report: Awaited<ReturnType<typeof execute>> | {
    schema: typeof REPORT_SCHEMA;
    schemaVersion: 1;
    status: "failed";
    profile: string;
    userAgent: string;
    platform: string;
    error: string;
    errorStack: string | null;
    checks: Check[];
    memorySamples: MemorySample[];
    watchdog: WatchdogDiagnostics;
  };
  activeFailureContext = null;
  try {
    report = await execute(request.profile);
    if (request.configPreflight != null) {
      const { qualifyPackagedConfigPreflight } = await import("../config/packagedConfigPreflightQualification");
      const configPreflight = await qualifyPackagedConfigPreflight(request.configPreflight);
      report.checks.push(...configPreflight.checks);
      Object.assign(report, { configPreflight });
    }
    if (request.python) {
      const { checkPackagedPython } = await import("../platform/packagedPythonCheck");
      const python = await checkPackagedPython();
      report.checks.push(...python.checks);
      Object.assign(report, { python });
    }
  } catch (error) {
    const context = failureContext();
    report = {
      schema: REPORT_SCHEMA,
      schemaVersion: 1,
      status: "failed",
      profile: request.profile,
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      error: error instanceof Error ? error.message : String(error),
      errorStack: error instanceof Error ? error.stack ?? null : null,
      checks: context?.checks ?? [],
      memorySamples: context?.memorySamples ?? [],
      watchdog: context?.watchdog ?? {
        intervalMs: 50,
        eventThresholdMs: 25,
        maximumDelayMs: 0,
        maximumByStageMs: {},
        events: [],
      },
    };
  }
  await invokeCommand("finish_visualization_qualification", { report });
}
