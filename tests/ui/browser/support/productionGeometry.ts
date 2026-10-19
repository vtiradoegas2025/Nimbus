import streamlineExtractionWorkerUrl from "@nimbus/lib/vtk/streamlineExtraction.worker.ts?worker&url";
import vtkPreparationWorkerUrl from "@nimbus/lib/vtk/vtkPreparation.worker.ts?worker&url";
import type {
  PreparedVtkImage,
  VtkPreparationRequest,
  VtkPreparationResult,
} from "@nimbus/lib/vtk/vtkPreparation";
import { SharedVortexContourDatasetService } from "@nimbus/lib/vtk/vortexContourClient";
import type {
  StreamlineWorkerReply,
  StreamlineWorkerRequest,
} from "@nimbus/lib/vtk/streamlineExtractionProtocol";
import { StreamlineWorkerOwnedPreparationService } from "@nimbus/lib/vtk/streamlineWorkerOwnedPreparation";
import { ContourWorkerOwnedPreparationService } from "@nimbus/lib/vtk/contourWorkerOwnedPreparation";

export interface ProductionGeometryQualification {
  scalarInputBytes: number;
  preparedScalarBytes: number;
  workerOwnedPreparedScalarBytes: number;
  workerOwnedContourPreparationElapsedMs: number;
  combinedContourPairElapsedMs: number;
  combinedContourPairBytes: number;
  vectorInputBytes: number;
  workerOwnedPreparedVectorBytes: number;
  workerOwnedPreparationElapsedMs: number;
  preparationElapsedMs: number;
  contourTriangles: number;
  contourBytes: number;
  contourKnownScratchBytes: number;
  contourElapsedMs: number;
  streamlineIntegratedPoints: number;
  streamlineKnownScratchBytes: number;
  streamlineElapsedMs: number;
  maximumWatchdogDelayMs: number;
  totalElapsedMs: number;
}

type PreparationReply =
  | { kind: "ready" }
  | { kind: "result"; ok: true; result: VtkPreparationResult }
  | { kind: "result"; ok: false; error: string };

let qualificationWorkerGeneration = 0;

function workerUrl(url: string, kind: string): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}qualification=${kind}&generation=${++qualificationWorkerGeneration}`;
}

function runWorker<
  Request extends { kind: string },
  Reply extends { kind: string },
>(
  url: string,
  name: string,
  request: Request,
  transfer: Transferable[],
  timeoutMs = 180_000,
): Promise<Reply> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(workerUrl(url, name), { type: "module", name });
    let posted = false;
    const timeout = window.setTimeout(() => {
      worker.terminate();
      reject(new Error(`${name} exceeded ${timeoutMs} ms`));
    }, timeoutMs);
    const finish = (action: () => void) => {
      window.clearTimeout(timeout);
      worker.onerror = null;
      worker.onmessageerror = null;
      action();
    };
    worker.onerror = (event) => {
      event.preventDefault();
      finish(() => {
        worker.terminate();
        reject(new Error(event.message || `${name} failed`));
      });
    };
    worker.onmessageerror = (event) => {
      event.preventDefault();
      finish(() => {
        worker.terminate();
        reject(new Error(`${name} returned an unreadable result`));
      });
    };
    worker.onmessage = (event: MessageEvent<Reply>) => {
      const probe = event.data as Reply & { kind: string; buffer?: SharedArrayBuffer };
      if (probe.kind === "shared-buffer-probe" && probe.buffer) {
        worker.postMessage({ kind: "shared-buffer-probe-ack", buffer: probe.buffer });
        return;
      }
      if (event.data.kind === "ready") {
        if (posted) return;
        posted = true;
        worker.postMessage(request, transfer);
        return;
      }
      finish(() => {
        worker.postMessage({ kind: "close" });
        resolve(event.data);
      });
    };
  });
}

async function withWatchdog<T>(task: Promise<T>): Promise<{ result: T; maximumDelayMs: number }> {
  let maximumDelayMs = 0;
  let previous = performance.now();
  const timer = window.setInterval(() => {
    const now = performance.now();
    maximumDelayMs = Math.max(maximumDelayMs, now - previous);
    previous = now;
  }, 16);
  try {
    const result = await task;
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    return { result, maximumDelayMs };
  } finally {
    window.clearInterval(timer);
  }
}

export async function qualifyProductionGeometryWorkers(): Promise<ProductionGeometryQualification> {
  const shape: [number, number, number] = [256, 256, 160];
  const spacing: [number, number, number] = [100, 100, 100];
  const origin: [number, number, number] = [0, 0, 0];
  const voxels = shape[0] * shape[1] * shape[2];
  const qualificationStarted = performance.now();

  const scalarSource = new Float32Array(voxels);
  for (let index = 0; index < scalarSource.length; index += 1) {
    scalarSource[index] = index % shape[2] - 79.5;
  }
  const scalarInputBytes = scalarSource.byteLength;
  const preparationRequest: VtkPreparationRequest = {
    kind: "scalar",
    field: { shape, data: scalarSource },
    grid: {
      coordinateSystem: "cartesian",
      drM: spacing[0],
      dyM: spacing[1],
      dzM: spacing[2],
      scalarOriginM: origin,
    },
    maxVoxels: voxels,
    datasetHandoff: "render-only",
  };
  const preparationStarted = performance.now();
  const preparation = await withWatchdog(runWorker<VtkPreparationRequest, PreparationReply>(
    vtkPreparationWorkerUrl,
    "nimbus-production-preparation-qualification",
    preparationRequest,
    [],
  ));
  const preparationElapsedMs = performance.now() - preparationStarted;
  if (preparation.result.kind !== "result" || !preparation.result.ok) {
    throw new Error(preparation.result.kind === "result"
      ? preparation.result.error
      : "preparation qualification returned no result");
  }
  if (!("values" in preparation.result.result) || "volume" in preparation.result.result) {
    throw new Error("preparation qualification returned a non-scalar result");
  }
  const prepared = preparation.result.result as PreparedVtkImage;
  const preparedScalarBytes = prepared.values.byteLength;
  const contourService = new SharedVortexContourDatasetService(prepared, null);
  const contour = await withWatchdog(contourService.enqueue({
    thresholds: { core: 0, opposite: null, mesocyclone: null },
    anchor: [12_800, 12_800, 7_950],
    condensateFloor: null,
  })).finally(() => contourService.shutdown());

  const workerOwnedContourService = new ContourWorkerOwnedPreparationService();
  const workerOwnedContourStarted = performance.now();
  const workerOwnedContourPreparation = await withWatchdog(
    workerOwnedContourService.prepare({
      kind: "scalar",
      field: { shape, data: scalarSource },
      grid: {
        coordinateSystem: "cartesian",
        drM: spacing[0],
        dyM: spacing[1],
        dzM: spacing[2],
        scalarOriginM: origin,
      },
      maxVoxels: voxels,
      datasetHandoff: "shared-geometry",
    }),
  );
  const workerOwnedContourPreparationElapsedMs = performance.now() - workerOwnedContourStarted;
  const workerOwnedPreparedScalarBytes = workerOwnedContourPreparation.result.handle.bytes;
  if (workerOwnedContourPreparation.result.handle.residency !== "worker"
      || "values" in workerOwnedContourPreparation.result.handle.metadata) {
    workerOwnedContourPreparation.result.release();
    workerOwnedContourService.shutdown();
    throw new Error("combined scalar preparation did not retain an opaque worker dataset");
  }
  workerOwnedContourPreparation.result.release();
  workerOwnedContourService.shutdown();

  const pairShape: [number, number, number] = [140, 140, 100];
  const pairVoxels = pairShape[0] * pairShape[1] * pairShape[2];
  const pairPrimary = new Float32Array(pairVoxels);
  const pairCondensate = new Float32Array(pairVoxels);
  for (let i = 0; i < pairShape[0]; i += 1) {
    for (let j = 0; j < pairShape[1]; j += 1) {
      const radius2 = ((i - 70) / 14) ** 2 + ((j - 70) / 14) ** 2;
      if (radius2 > 16) continue;
      for (let k = 0; k < pairShape[2]; k += 1) {
        const index = (i * pairShape[1] + j) * pairShape[2] + k;
        const vertical = Math.max(0, 1 - k / 120);
        pairPrimary[index] = 0.08 * Math.exp(-radius2) * vertical;
        pairCondensate[index] = 0.001 * Math.exp(-radius2 * 0.7) * vertical;
      }
    }
  }
  const pairGrid = {
    coordinateSystem: "cartesian" as const,
    drM: 100,
    dyM: 100,
    dzM: 100,
    scalarOriginM: [0, 0, 0] as [number, number, number],
  };
  const combinedPairStarted = performance.now();
  const pairService = new ContourWorkerOwnedPreparationService();
  const combinedPair = await withWatchdog((async () => {
    const primary = await pairService.prepare({ kind: "scalar", field: { shape: pairShape, data: pairPrimary },
      grid: pairGrid, maxVoxels: pairVoxels, featureSign: 1, datasetHandoff: "shared-geometry" });
    try {
      const condensate = await pairService.prepare({ kind: "scalar", field: { shape: pairShape, data: pairCondensate },
        grid: pairGrid, maxVoxels: pairVoxels, datasetHandoff: "shared-geometry" });
      try {
        return await pairService.extract(primary.handle, condensate.handle, {
          thresholds: { core: 0.02, opposite: null, mesocyclone: 0.005 },
          anchor: [7_000, 7_000, 100], condensateFloor: 0.00001,
        });
      } finally { condensate.release(); }
    } finally { primary.release(); }
  })()).finally(() => pairService.shutdown());
  const combinedContourPairElapsedMs = performance.now() - combinedPairStarted;
  if (combinedPair.result.usage.triangles <= 0) throw new Error("combined contour pair returned no crossing surface");
  const combinedContourPairBytes = pairPrimary.byteLength + pairCondensate.byteLength;

  const vectorValues = new Float32Array(voxels * 3);
  for (let tuple = 0; tuple < voxels; tuple += 1) {
    vectorValues[tuple * 3] = 10;
    vectorValues[tuple * 3 + 2] = 1;
  }
  const vectorInputBytes = vectorValues.byteLength;
  const workerOwnedU = new Float32Array(voxels).fill(10);
  const workerOwnedV = new Float32Array(voxels);
  const workerOwnedW = new Float32Array(voxels).fill(1);
  const workerOwnedService = new StreamlineWorkerOwnedPreparationService();
  const workerOwnedStarted = performance.now();
  const workerOwnedPreparation = await withWatchdog(workerOwnedService.prepare({
    kind: "vector",
    fields: [
      { shape, data: workerOwnedU },
      { shape, data: workerOwnedV },
      { shape, data: workerOwnedW },
    ],
    grid: {
      coordinateSystem: "cartesian",
      drM: spacing[0],
      dyM: spacing[1],
      dzM: spacing[2],
      scalarOriginM: origin,
    },
    maxVoxels: voxels,
    datasetHandoff: "shared-geometry",
  }));
  const workerOwnedPreparationElapsedMs = performance.now() - workerOwnedStarted;
  const workerOwnedPreparedVectorBytes = workerOwnedPreparation.result.handle.bytes;
  if (workerOwnedPreparation.result.handle.residency !== "worker"
      || "values" in workerOwnedPreparation.result.handle.metadata) {
    workerOwnedPreparation.result.release();
    workerOwnedService.shutdown();
    throw new Error("combined vector preparation did not retain an opaque worker dataset");
  }
  workerOwnedPreparation.result.release();
  workerOwnedService.shutdown();
  const seeds = Float32Array.from([
    6_400, 6_400, 4_000,
    6_400, 19_200, 4_000,
    12_800, 12_800, 8_000,
    19_200, 6_400, 8_000,
    19_200, 19_200, 8_000,
    6_400, 12_800, 12_000,
    12_800, 6_400, 12_000,
    12_800, 19_200, 12_000,
  ]);
  const streamlineRequest: StreamlineWorkerRequest = {
    kind: "extract",
    requestId: 1,
    datasetId: 1,
    image: { shape, spacing, origin, values: vectorValues },
    seeds,
    integrationStep: 5,
    maxSteps: 40,
    tubeRadius: 50,
    scale: { xy: 1, z: 1 },
  };
  const streamline = await withWatchdog(runWorker<
    StreamlineWorkerRequest,
    StreamlineWorkerReply
  >(
    streamlineExtractionWorkerUrl,
    "nimbus-production-streamline-qualification",
    streamlineRequest,
    [vectorValues.buffer, seeds.buffer],
  ));
  if (streamline.result.kind !== "result" || !streamline.result.ok) {
    throw new Error(streamline.result.kind === "result"
      ? streamline.result.error
      : "streamline qualification returned no result");
  }

  return {
    scalarInputBytes,
    preparedScalarBytes,
    workerOwnedPreparedScalarBytes,
    workerOwnedContourPreparationElapsedMs,
    combinedContourPairElapsedMs,
    combinedContourPairBytes,
    vectorInputBytes,
    workerOwnedPreparedVectorBytes,
    workerOwnedPreparationElapsedMs,
    preparationElapsedMs,
    contourTriangles: contour.result.usage.triangles,
    contourBytes: contour.result.usage.bytes,
    contourKnownScratchBytes: contour.result.knownScratchBytes,
    contourElapsedMs: contour.result.elapsedMs,
    streamlineIntegratedPoints: streamline.result.result.integratedPoints,
    streamlineKnownScratchBytes: streamline.result.result.knownScratchBytes,
    streamlineElapsedMs: streamline.result.result.elapsedMs,
    maximumWatchdogDelayMs: Math.max(
      preparation.maximumDelayMs,
      workerOwnedContourPreparation.maximumDelayMs,
      combinedPair.maximumDelayMs,
      contour.maximumDelayMs,
      streamline.maximumDelayMs,
      workerOwnedPreparation.maximumDelayMs,
    ),
    totalElapsedMs: performance.now() - qualificationStarted,
  };
}
