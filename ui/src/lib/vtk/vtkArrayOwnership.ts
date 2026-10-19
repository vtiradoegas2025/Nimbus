import { MAX_FIELD_DECODED_BYTES } from "@/lib/platform/npy";

/** Internal slot names must not depend on the user-selected field name. */
export const VTK_SCALAR_SLOT = "__nimbus_active_scalars";
export const VTK_VECTOR_SLOT = "__nimbus_active_vectors";

export interface OwnedVtkArray {
  getName: () => string;
  delete?: () => void;
}

export interface VtkPointDataArrays {
  getArrays: () => OwnedVtkArray[];
  removeAllArrays: () => void;
}

export interface VtkPointDataOwner extends VtkPointDataArrays {
  setScalars: (array: OwnedVtkArray) => unknown;
  setVectors: (array: OwnedVtkArray) => unknown;
}

export type VtkAttributeKind = "scalars" | "vectors";

/** Attach next and delete stale arrays. setScalars/setVectors replaces the ref without
 *  deleting the displaced array, so explicit cleanup via stable slot names is required. */
export function replaceVtkPointDataArray(
  pointData: VtkPointDataOwner,
  kind: VtkAttributeKind,
  next: OwnedVtkArray,
): void {
  const before = pointData.getArrays().slice();
  try {
    if (kind === "scalars") pointData.setScalars(next);
    else pointData.setVectors(next);
  } catch (error) {
    next.delete?.();
    throw error;
  }

  // Rebuild to one owned slot; also clears arrays from older field-name-based builds.
  const stale = new Set<OwnedVtkArray>();
  for (const array of before) if (array !== next) stale.add(array);
  for (const array of pointData.getArrays()) if (array !== next) stale.add(array);
  pointData.removeAllArrays();
  if (kind === "scalars") pointData.setScalars(next);
  else pointData.setVectors(next);

  for (const array of stale) array.delete?.();
}

/** Detach and delete every owned point-data array before its data set dies. */
export function releaseVtkPointDataArrays(pointData: VtkPointDataArrays): void {
  const arrays = [...new Set(pointData.getArrays())];
  pointData.removeAllArrays();
  for (const array of arrays) array.delete?.();
}

/**
 * Reject an individual derived VTK array before allocating it. This matches
 * the validated decoded-field ceiling and turns an oversize render into a
 * deterministic error instead of a browser allocation failure.
 */
export function assertVtkArrayAllocation(
  elementCount: number,
  numberOfComponents: number,
): void {
  if (
    !Number.isSafeInteger(elementCount) ||
    elementCount < 0 ||
    !Number.isSafeInteger(numberOfComponents) ||
    numberOfComponents <= 0
  ) {
    throw new Error("Invalid VTK array dimensions");
  }
  const bytes = elementCount * numberOfComponents * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(bytes) || bytes > MAX_FIELD_DECODED_BYTES) {
    throw new Error(
      `Derived VTK array is ${bytes} bytes; limit is ${MAX_FIELD_DECODED_BYTES}`,
    );
  }
}
