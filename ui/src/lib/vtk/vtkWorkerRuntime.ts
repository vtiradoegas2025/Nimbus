/** UI-thread fallbacks are a last-resort compatibility path. Keep their input and
 * algorithmic work small enough that a failed worker cannot turn into a long
 * uninterruptible main-thread task. */
export const LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP = 8 * 1024 * 1024;
export const LOCAL_VTK_CONTOUR_VOXEL_CAP = 500_000;
export const LOCAL_VTK_STREAMLINE_SEED_CAP = 512;
export const LOCAL_VTK_STREAMLINE_STEP_CAP = 2_000;
export const LOCAL_VTK_STREAMLINE_WORK_CAP = 1_000_000;

export function localVtkContourBudgetError(
  voxelCount: number,
  inputBytes: number,
): string | null {
  if (inputBytes > LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP) {
    return `Local contour extraction requires ${(inputBytes / 1024 / 1024).toFixed(1)} MiB; `
      + `the UI-thread limit is ${LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP / 1024 / 1024} MiB`;
  }
  if (voxelCount > LOCAL_VTK_CONTOUR_VOXEL_CAP) {
    return `Local contour extraction requires ${voxelCount.toLocaleString()} voxels; `
      + `the UI-thread limit is ${LOCAL_VTK_CONTOUR_VOXEL_CAP.toLocaleString()}`;
  }
  return null;
}

export function localVtkStreamlineBudgetError(
  inputBytes: number,
  seedCount: number,
  maxSteps: number,
): string | null {
  if (inputBytes > LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP) {
    return `Local streamline extraction requires ${(inputBytes / 1024 / 1024).toFixed(1)} MiB; `
      + `the UI-thread limit is ${LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP / 1024 / 1024} MiB`;
  }
  if (seedCount > LOCAL_VTK_STREAMLINE_SEED_CAP) {
    return `Local streamline extraction requires ${seedCount.toLocaleString()} seeds; `
      + `the UI-thread limit is ${LOCAL_VTK_STREAMLINE_SEED_CAP.toLocaleString()}`;
  }
  if (maxSteps > LOCAL_VTK_STREAMLINE_STEP_CAP
      || seedCount * maxSteps * 2 > LOCAL_VTK_STREAMLINE_WORK_CAP) {
    return "Local streamline extraction exceeds the UI-thread integration budget";
  }
  return null;
}

export function throwIfLocalVtkTaskAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException("Local VTK geometry task aborted", "AbortError");
}

export type VtkGeometryExecutionMode = "worker" | "local";

/** Worker availability, rather than browser identity, owns the execution
 * decision. Worker startup is separately retried and fails closed, so a
 * Safari/WebKit user is not pre-emptively confined to the small UI-thread
 * safety budget merely because of their user agent. */
export function vtkGeometryExecutionMode(
  workerAvailable = typeof Worker === "function",
): VtkGeometryExecutionMode {
  return workerAvailable ? "worker" : "local";
}

export function requiresLocalVtkGeometryFallback(): boolean {
  return vtkGeometryExecutionMode() === "local";
}
