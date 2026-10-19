import { CARTESIAN_VOXEL_CAP_VOLUME } from "@/lib/science/cartesianInterp";
import {
  vtkPreparationTransferables,
  type VtkPreparationRequest,
  type VtkPreparationResult,
} from "./vtkPreparation";
import type { VisualizationResourcePlan } from "@/lib/core/visualizationResourceGovernor";

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const VECTOR_COMPONENTS = 3;
const MAX_FEATURE_SEED_BYTES = 512 * 3 * FLOAT32_BYTES;

function safeProduct(shape: readonly number[]): number {
  let product = 1;
  for (const extent of shape) {
    if (!Number.isSafeInteger(extent) || extent <= 0) return 0;
    product *= extent;
    if (!Number.isSafeInteger(product)) return Number.MAX_SAFE_INTEGER;
  }
  return product;
}

function boundedCells(sourceCells: number, maxVoxels: number | undefined): number {
  const limit = maxVoxels == null
    ? CARTESIAN_VOXEL_CAP_VOLUME
    : Math.max(1, Math.floor(maxVoxels));
  return Math.min(sourceCells, limit);
}

function refinementUpperBound(maxVoxels: number | undefined): number {
  return maxVoxels == null
    ? CARTESIAN_VOXEL_CAP_VOLUME
    : Math.max(1, Math.floor(maxVoxels));
}

export function vtkPreparationInputBytes(request: VtkPreparationRequest): number {
  const buffers = new Set<ArrayBufferLike>();
  const add = (field: { data: Float32Array } | undefined) => {
    if (!field) return;
    buffers.add(field.data.buffer);
    const geometry = (field as { windowGeometry?: {
      centroidsM?: { x: Float64Array; y: Float64Array; z: Float64Array } | null;
      volumesM3?: Float64Array | null;
    } }).windowGeometry;
    if (geometry?.centroidsM) {
      buffers.add(geometry.centroidsM.x.buffer);
      buffers.add(geometry.centroidsM.y.buffer);
      buffers.add(geometry.centroidsM.z.buffer);
    }
    if (geometry?.volumesM3) buffers.add(geometry.volumesM3.buffer);
  };
  if (request.kind === "scalar") {
    add(request.field);
    add(request.subtractField);
  } else if (request.kind === "cloud") {
    add(request.field);
    add(request.subtractField);
    request.flowFields?.forEach(add);
  } else {
    request.fields.forEach(add);
  }
  let bytes = 0;
  for (const buffer of buffers) bytes += buffer.byteLength;
  return bytes;
}

export function vtkPreparationOutputUpperBoundBytes(request: VtkPreparationRequest): number {
  if (request.kind === "cloud") {
    return (request.volumeMaxVoxels + request.surfaceMaxVoxels) * (FLOAT32_BYTES + 1);
  }
  const sourceCells = request.kind === "scalar"
    ? safeProduct(request.field.shape)
    : Math.min(...request.fields.map((field) => safeProduct(field.shape)));
  if (request.kind === "vector") {
    return boundedCells(sourceCells, request.maxVoxels) * VECTOR_COMPONENTS * FLOAT32_BYTES
      + MAX_FEATURE_SEED_BYTES;
  }
  const outputCells = request.refineForDisplay
    ? refinementUpperBound(request.maxVoxels)
    : boundedCells(sourceCells, request.maxVoxels);
  const centerlineBytes = request.kind === "velocity-diagnostic"
    ? Math.max(...request.fields.map((field) => field.shape[2] ?? 0)) * 3 * FLOAT32_BYTES
    : 0;
  const terrainCoordinateBytes = request.kind === "scalar" && request.grid.terrainFollowing
    ? outputCells * VECTOR_COMPONENTS * FLOAT32_BYTES
    : 0;
  const validityBytes = request.includeValidity === false ? 0 : outputCells;
  return outputCells * FLOAT32_BYTES + validityBytes + terrainCoordinateBytes + centerlineBytes;
}

export function vtkPreparationWorkerScratchUpperBoundBytes(request: VtkPreparationRequest): number {
  if (request.kind !== "scalar") return 0;
  const sourceCells = safeProduct(request.field.shape);
  return (request.scalarTransform === "horizontal-level-anomaly" ? boundedCells(sourceCells, request.maxVoxels) * FLOAT32_BYTES : 0)
    + (request.background && request.cropBackground ? vtkPreparationOutputUpperBoundBytes(request) : 0);
}

export function vtkPreparationResultBytes(result: VtkPreparationResult): number {
  const buffers = new Set<ArrayBufferLike>(vtkPreparationTransferables(result));
  let bytes = 0;
  for (const buffer of buffers) bytes += buffer.byteLength;
  return bytes;
}

export function vtkPreparationResourcePlan(
  request: VtkPreparationRequest,
  identity: string,
  representation: VisualizationResourcePlan["representation"] = "full-domain-lod",
): VisualizationResourcePlan {
  const inputBytes = vtkPreparationInputBytes(request);
  const outputBytes = vtkPreparationOutputUpperBoundBytes(request);
  const workerScratchBytes = vtkPreparationWorkerScratchUpperBoundBytes(request);
  const sampledCells = request.kind === "cloud"
    ? Math.max(request.volumeMaxVoxels, request.surfaceMaxVoxels)
    : Math.max(1, Math.floor(outputBytes / FLOAT32_BYTES
      / (request.kind === "vector" ? VECTOR_COMPONENTS : 1)));
  return {
    identity,
    execution: "worker-cpu",
    representation,
    sampledShape: null,
    sampledCellUpperBound: sampledCells,
    expectedPeakBytes: {
      nativeRead: 0,
      decoded: inputBytes,
      preparedMain: outputBytes,
      workerResident: inputBytes,
      workerScratch: workerScratchBytes,
      handoff: outputBytes,
      retainedGeometry: 0,
      gpu: 0,
    },
    outputBudget: { bytes: 0, primitives: 0 },
    provenance: `${request.kind} preparation upper bound; worker scratch not yet instrumented`,
  };
}
