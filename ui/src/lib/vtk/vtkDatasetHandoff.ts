import { vtkPreparationTransferables, type VtkPreparationResult } from "./vtkPreparation";

export type VtkDatasetHandoffMode =
  | "render-only-transfer"
  | "shared-zero-copy"
  | "transferable-one-copy"
  | "warm-reuse"
  | "retry-transfer";

export interface VtkDatasetHandoffReport {
  mode: VtkDatasetHandoffMode;
  logicalBytes: number;
  copiedBytes: number;
  datasets: number;
  stage: "preparation" | "geometry" | "upload";
  dataset: string;
  retry?: boolean;
  sourceRetained?: boolean;
  reason?: string;
}

export interface VtkDatasetHandoffMetrics {
  sharedDatasets: number;
  transferredDatasets: number;
  logicalBytes: number;
  copiedBytes: number;
  retries: number;
  warmReuses: number;
  maxInflightBytes: number;
  sourceRetainedDatasets: number;
  downgradeReasons: string[];
  byStage: Record<"preparation" | "geometry" | "upload", {
    datasets: number;
    logicalBytes: number;
    copiedBytes: number;
  }>;
}

const metrics: VtkDatasetHandoffMetrics = {
  sharedDatasets: 0,
  transferredDatasets: 0,
  logicalBytes: 0,
  copiedBytes: 0,
  retries: 0,
  warmReuses: 0,
  maxInflightBytes: 0,
  sourceRetainedDatasets: 0,
  downgradeReasons: [],
  byStage: {
    preparation: { datasets: 0, logicalBytes: 0, copiedBytes: 0 },
    geometry: { datasets: 0, logicalBytes: 0, copiedBytes: 0 },
    upload: { datasets: 0, logicalBytes: 0, copiedBytes: 0 },
  },
};

declare global {
  interface Window {
    __NIMBUS_VTK_HANDOFF_METRICS__?: VtkDatasetHandoffMetrics;
  }
}

function publishMetrics(): void {
  if (typeof window !== "undefined") {
    window.__NIMBUS_VTK_HANDOFF_METRICS__ = {
      ...metrics,
      byStage: {
        preparation: { ...metrics.byStage.preparation },
        geometry: { ...metrics.byStage.geometry },
        upload: { ...metrics.byStage.upload },
      },
      downgradeReasons: [...metrics.downgradeReasons],
    };
  }
}

export function recordWarmVtkDatasetReuse(dataset: string): void {
  recordVtkDatasetHandoff({
    mode: "warm-reuse",
    logicalBytes: 0,
    copiedBytes: 0,
    datasets: 1,
    stage: "geometry",
    dataset,
  });
}

export function recordVtkDatasetHandoff(report: VtkDatasetHandoffReport): void {
  if (report.mode === "shared-zero-copy") metrics.sharedDatasets += report.datasets;
  else metrics.transferredDatasets += report.datasets;
  metrics.logicalBytes += report.logicalBytes;
  metrics.copiedBytes += report.copiedBytes;
  metrics.retries += report.retry ? 1 : 0;
  metrics.warmReuses += report.mode === "warm-reuse" ? report.datasets : 0;
  metrics.maxInflightBytes = Math.max(metrics.maxInflightBytes, report.logicalBytes);
  metrics.sourceRetainedDatasets += report.sourceRetained ? report.datasets : 0;
  if (report.reason && !metrics.downgradeReasons.includes(report.reason)) {
    metrics.downgradeReasons.push(report.reason);
  }
  const stage = metrics.byStage[report.stage];
  stage.datasets += report.datasets;
  stage.logicalBytes += report.logicalBytes;
  stage.copiedBytes += report.copiedBytes;
  publishMetrics();
}

export function vtkDatasetHandoffMetrics(): VtkDatasetHandoffMetrics {
  return {
    ...metrics,
    byStage: {
      preparation: { ...metrics.byStage.preparation },
      geometry: { ...metrics.byStage.geometry },
      upload: { ...metrics.byStage.upload },
    },
    downgradeReasons: [...metrics.downgradeReasons],
  };
}

export function resetVtkDatasetHandoffMetrics(): void {
  metrics.sharedDatasets = 0;
  metrics.transferredDatasets = 0;
  metrics.logicalBytes = 0;
  metrics.copiedBytes = 0;
  metrics.retries = 0;
  metrics.warmReuses = 0;
  metrics.maxInflightBytes = 0;
  metrics.sourceRetainedDatasets = 0;
  metrics.downgradeReasons.length = 0;
  for (const stage of Object.values(metrics.byStage)) {
    stage.datasets = 0;
    stage.logicalBytes = 0;
    stage.copiedBytes = 0;
  }
  publishMetrics();
}

function sharedArrayBuffersAvailable(): boolean {
  return typeof SharedArrayBuffer !== "undefined"
    && (typeof crossOriginIsolated === "undefined" || crossOriginIsolated === true);
}

let sharedGeometryQualification: Promise<boolean> | null = null;

/** Main-thread upload smoke before a request opts into shared prepared grids.
 * The preparation worker separately performs a real SAB echo round trip. VTK
 * itself remains guarded at setPreparedImageScalars, where a rejected shared
 * upload is retried from an intact AB copy. */
export function qualifySharedVtkGeometryHandoff(): Promise<boolean> {
  if (sharedGeometryQualification) return sharedGeometryQualification;
  sharedGeometryQualification = (async () => {
    if (!sharedArrayBuffersAvailable() || typeof document === "undefined") return false;
    try {
      const probe = new Float32Array(new SharedArrayBuffer(16));
      probe.set([1, 2, 3, 4]);
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
      if (!gl) return false;
      const buffer = gl.createBuffer();
      if (!buffer) return false;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, probe, gl.STATIC_DRAW);
      const webGlAccepted = gl.getError() === gl.NO_ERROR;
      gl.deleteBuffer(buffer);
      // The probe must not leave a second ANGLE context competing with the
      // VTK render window while Dawn initializes its Metal device. Brave on
      // macOS can terminate the renderer process several seconds later when
      // both transient contexts survive until GC.
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      if (!webGlAccepted) return false;
      return true;
    } catch {
      return false;
    }
  })();
  return sharedGeometryQualification;
}

export function isSharedVtkDataset(values: Float32Array): boolean {
  return typeof SharedArrayBuffer !== "undefined" && values.buffer instanceof SharedArrayBuffer;
}

/**
 * Preparation-worker boundary. In a cross-origin-isolated browser, promote
 * full prepared grids to SharedArrayBuffer once while they are still worker
 * owned. The UI and any extraction workers can then retain the exact same
 * immutable bytes through context/backend rebuilds without cloning them.
 * This deliberately covers the prepared P-byte geometry fan-out only: raw
 * cached field inputs still cross the preparation boundary by structured
 * clone (R/3R/etc.), which is a separate source-cache transport concern.
 */
export function sharePreparedVtkResult(
  result: VtkPreparationResult,
  allowSharedGeometry: boolean,
): VtkDatasetHandoffReport {
  const images = "volume" in result
    ? [result.volume, result.surface]
    : [result];
  const uniqueBuffers = new Set<ArrayBufferLike>();
  for (const image of images) {
    uniqueBuffers.add(image.values.buffer);
    if (image.validity) uniqueBuffers.add(image.validity.buffer);
  }
  const logicalBytes = [...uniqueBuffers].reduce((sum, buffer) => sum + buffer.byteLength, 0);
  if (!allowSharedGeometry) {
    return {
      mode: "render-only-transfer",
      logicalBytes,
      copiedBytes: 0,
      datasets: uniqueBuffers.size,
      stage: "preparation",
      dataset: "prepared-grid",
    };
  }
  if (!sharedArrayBuffersAvailable()) {
    return {
      mode: "transferable-one-copy",
      logicalBytes,
      copiedBytes: 0,
      datasets: uniqueBuffers.size,
      stage: "preparation",
      dataset: "prepared-grid",
    };
  }

  const replacements = new Map<ArrayBufferLike, SharedArrayBuffer>();
  let copiedBytes = 0;
  for (const buffer of uniqueBuffers) {
    if (buffer instanceof SharedArrayBuffer) {
      replacements.set(buffer, buffer);
      continue;
    }
    const shared = new SharedArrayBuffer(buffer.byteLength);
    new Uint8Array(shared).set(new Uint8Array(buffer));
    replacements.set(buffer, shared);
    copiedBytes += buffer.byteLength;
  }
  for (const image of images) {
    const values = image.values;
    image.values = new Float32Array(
      replacements.get(values.buffer)!,
      values.byteOffset,
      values.length,
    );
    if (image.validity) image.validity = new Uint8Array(
      replacements.get(image.validity.buffer)!, image.validity.byteOffset, image.validity.length,
    );
  }
  return {
    mode: "shared-zero-copy",
    logicalBytes,
    copiedBytes,
    datasets: uniqueBuffers.size,
    stage: "preparation",
    dataset: "prepared-grid",
  };
}

export interface ExtractionDatasetHandoff {
  values: Float32Array;
  transfer: ArrayBuffer[];
  report: VtkDatasetHandoffReport;
}

/** Extraction-worker boundary: shared preparation is a zero-copy structured
 * clone. Non-isolated runtimes make exactly one retained-preserving copy and
 * transfer that copy, avoiding a second structured-clone allocation. */
export function handoffVtkExtractionDataset(
  values: Float32Array,
  dataset = "prepared-grid",
  validity?: Uint8Array,
): ExtractionDatasetHandoff {
  if (validity && validity.length !== values.length) throw new Error("Scalar geometry validity does not match its values");
  if (isSharedVtkDataset(values) && !validity) {
    return {
      values,
      transfer: [],
      report: {
        mode: "shared-zero-copy",
        logicalBytes: values.byteLength,
        copiedBytes: 0,
        datasets: 1,
        stage: "geometry",
        dataset,
        sourceRetained: true,
      },
    };
  }
  const copy = values.slice();
  if (validity) for (let index = 0; index < copy.length; index++) if (!validity[index]) copy[index] = Number.NaN;
  return {
    values: copy,
    transfer: [copy.buffer as ArrayBuffer],
    report: {
      mode: "transferable-one-copy",
      logicalBytes: values.byteLength,
      copiedBytes: values.byteLength,
      datasets: 1,
      stage: "geometry",
      dataset,
      sourceRetained: values.byteLength === copy.byteLength,
    },
  };
}

export function geometryScalarValues(image: { values: Float32Array; validity?: Uint8Array }): Float32Array {
  return image.validity ? handoffVtkExtractionDataset(image.values, "masked-geometry", image.validity).values : image.values;
}

const uploadFallbacks = new WeakMap<object, Map<string, Float32Array>>();

/** Guard the vtkDataArray attachment boundary. Qualified browsers take the
 * shared fast path; if vtk.js rejects that view synchronously, cache one AB
 * copy for this immutable dataset and retry without detaching the SAB used by
 * geometry workers or later backend rebuilds. */
export function uploadPreparedVtkValues(
  values: Float32Array,
  upload: (compatibleValues: Float32Array) => void,
  dataset = "prepared-grid",
): void {
  if (!isSharedVtkDataset(values)) {
    upload(values);
    return;
  }
  try {
    upload(values);
  } catch (error) {
    let spans = uploadFallbacks.get(values.buffer);
    if (!spans) {
      spans = new Map();
      uploadFallbacks.set(values.buffer, spans);
    }
    const span = `${values.byteOffset}:${values.byteLength}`;
    let fallback = spans.get(span);
    let copiedBytes = 0;
    if (!fallback) {
      fallback = Float32Array.from(values);
      spans.set(span, fallback);
      copiedBytes = fallback.byteLength;
    }
    recordVtkDatasetHandoff({
      mode: "retry-transfer",
      logicalBytes: values.byteLength,
      copiedBytes,
      datasets: 1,
      stage: "upload",
      dataset,
      retry: true,
    });
    try {
      upload(fallback);
    } catch {
      throw error;
    }
  }
}

/** SharedArrayBuffer is cloneable but not transferable. */
export function transferablePreparedBuffers(result: VtkPreparationResult): ArrayBuffer[] {
  const candidates = vtkPreparationTransferables(result);
  return [...new Set(candidates)].filter(
    (buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer,
  );
}

export function downgradeSharedPreparedResult(result: VtkPreparationResult): VtkDatasetHandoffReport {
  const images = "volume" in result ? [result.volume, result.surface] : [result];
  const replacements = new Map<SharedArrayBuffer, ArrayBuffer>();
  let copiedBytes = 0;
  for (const image of images) {
    if (image.validity?.buffer instanceof SharedArrayBuffer) {
      const validity = image.validity;
      const buffer = validity.buffer as SharedArrayBuffer;
      let replacement = replacements.get(buffer);
      if (!replacement) {
        replacement = new ArrayBuffer(buffer.byteLength);
        new Uint8Array(replacement).set(new Uint8Array(buffer));
        replacements.set(buffer, replacement);
        copiedBytes += replacement.byteLength;
      }
      image.validity = new Uint8Array(replacement, validity.byteOffset, validity.length);
    }
    const values = image.values;
    if (!(values.buffer instanceof SharedArrayBuffer)) continue;
    let replacement = replacements.get(values.buffer);
    if (!replacement) {
      replacement = new ArrayBuffer(values.buffer.byteLength);
      new Uint8Array(replacement).set(new Uint8Array(values.buffer));
      replacements.set(values.buffer, replacement);
      copiedBytes += replacement.byteLength;
    }
    image.values = new Float32Array(replacement, values.byteOffset, values.length);
  }
  return {
    mode: "retry-transfer",
    logicalBytes: copiedBytes,
    copiedBytes,
    datasets: replacements.size,
    stage: "preparation",
    dataset: "prepared-grid",
    retry: true,
  };
}
