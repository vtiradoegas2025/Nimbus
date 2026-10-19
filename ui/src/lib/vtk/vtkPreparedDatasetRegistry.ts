import {
  type PreparedCloudImages,
  type PreparedVtkImage,
  vtkPreparationTransferables,
  type VtkPreparationResult,
} from "./vtkPreparation";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";

export type VtkPreparedDatasetKind = "scalar" | "vector" | "velocity-diagnostic" | "cloud";

type ImageMetadata = Omit<PreparedVtkImage, "values">;
export type VtkPreparedDatasetMetadata<T extends VtkPreparationResult> =
  T extends PreparedCloudImages
    ? Omit<T, "volume" | "surface"> & { volume: ImageMetadata; surface: ImageMetadata }
    : Omit<T, "values">;

export interface VtkPreparedDatasetHandle<T extends VtkPreparationResult = VtkPreparationResult> {
  readonly schema: "nimbus.studio.vtk-prepared-dataset-handle";
  readonly schemaVersion: 2;
  readonly id: number;
  readonly kind: VtkPreparedDatasetKind;
  readonly bytes: number;
  readonly residency: "main" | "worker";
  readonly metadata: VtkPreparedDatasetMetadata<T>;
  readonly _type?: T;
}

export interface VtkPreparedDatasetLease<T extends VtkPreparationResult> {
  readonly handle: VtkPreparedDatasetHandle<T>;
  readonly active: boolean;
  release: () => void;
}

interface Entry<T extends VtkPreparationResult = VtkPreparationResult> {
  handle: VtkPreparedDatasetHandle<T>;
  prepared: T | null;
  remoteBacking: object | null;
  onLastRelease: (() => void) | null;
  references: number;
  reservationId: number;
}

function imageMetadata(image: PreparedVtkImage): ImageMetadata {
  const { values: _values, ...metadata } = image;
  return Object.freeze(metadata);
}

export function vtkPreparedDatasetMetadata<T extends VtkPreparationResult>(
  prepared: T,
): VtkPreparedDatasetMetadata<T> {
  if ("volume" in prepared) {
    const { volume, surface, ...metadata } = prepared;
    return Object.freeze({
      ...metadata,
      volume: imageMetadata(volume),
      surface: imageMetadata(surface),
    }) as VtkPreparedDatasetMetadata<T>;
  }
  return imageMetadata(prepared) as VtkPreparedDatasetMetadata<T>;
}

let handleSequence = 0;
const entries = new Map<number, Entry>();
const entriesByPrepared = new WeakMap<object, Entry>();
const entriesByRemoteBacking = new WeakMap<object, Entry>();
type RegistryListener = () => void;
const listeners = new Set<RegistryListener>();

function publish(): void {
  for (const listener of listeners) listener();
}

export function retainPreparedDataset<T extends VtkPreparationResult>(
  prepared: T,
  kind: VtkPreparedDatasetKind,
): VtkPreparedDatasetLease<T> {
  let entry = entriesByPrepared.get(prepared) as Entry<T> | undefined;
  if (!entry) {
    const id = ++handleSequence;
    const buffers = vtkPreparationTransferables(prepared);
    const bytes = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
    const resources = visualizationResourceGovernor.reserve({
      owner: "vtk-prepared-dataset-registry",
      generation: `handle:${id}`,
      priority: "visible",
      claims: buffers.map((buffer) => visualizationBufferClaim(buffer, "prepared-main")),
    });
    if (!resources.ok) throw new Error(resources.reason);
    entry = {
      handle: Object.freeze({
        schema: "nimbus.studio.vtk-prepared-dataset-handle",
        schemaVersion: 2,
        id,
        kind,
        bytes,
        residency: "main",
        metadata: vtkPreparedDatasetMetadata(prepared),
      }) as VtkPreparedDatasetHandle<T>,
      prepared,
      remoteBacking: null,
      onLastRelease: null,
      references: 0,
      reservationId: resources.reservation.id,
    };
    entries.set(id, entry);
    entriesByPrepared.set(prepared, entry);
  } else if (entry.handle.kind !== kind) {
    throw new Error(`Prepared dataset handle kind changed from ${entry.handle.kind} to ${kind}`);
  }
  return leaseEntry(entry);
}

export function retainRemotePreparedDataset<T extends VtkPreparationResult>(
  remoteBacking: object,
  options: {
    kind: VtkPreparedDatasetKind;
    bytes: number;
    metadata: VtkPreparedDatasetMetadata<T>;
    onLastRelease?: () => void;
  },
): VtkPreparedDatasetLease<T> {
  if (!Number.isSafeInteger(options.bytes) || options.bytes < 0) {
    throw new Error("Remote prepared dataset bytes must be a non-negative safe integer");
  }
  let entry = entriesByRemoteBacking.get(remoteBacking) as Entry<T> | undefined;
  if (!entry) {
    const id = ++handleSequence;
    entry = {
      handle: Object.freeze({
        schema: "nimbus.studio.vtk-prepared-dataset-handle",
        schemaVersion: 2,
        id,
        kind: options.kind,
        bytes: options.bytes,
        residency: "worker",
        metadata: options.metadata,
      }) as VtkPreparedDatasetHandle<T>,
      prepared: null,
      remoteBacking,
      onLastRelease: options.onLastRelease ?? null,
      references: 0,
      reservationId: 0,
    };
    entries.set(id, entry);
    entriesByRemoteBacking.set(remoteBacking, entry);
  } else if (entry.handle.kind !== options.kind
      || entry.handle.bytes !== options.bytes
      || entry.handle.metadata !== options.metadata) {
    throw new Error("Remote prepared dataset identity changed while retained");
  }
  return leaseEntry(entry);
}

function leaseEntry<T extends VtkPreparationResult>(entry: Entry<T>): VtkPreparedDatasetLease<T> {
  entry.references += 1;
  publish();
  let released = false;
  return {
    handle: entry.handle,
    get active() { return !released; },
    release: () => {
      if (released) return;
      released = true;
      entry.references -= 1;
      if (entry.references > 0) {
        publish();
        return;
      }
      if (entry.reservationId !== 0) visualizationResourceGovernor.release(entry.reservationId);
      entries.delete(entry.handle.id);
      if (entry.prepared) entriesByPrepared.delete(entry.prepared);
      if (entry.remoteBacking) entriesByRemoteBacking.delete(entry.remoteBacking);
      const onLastRelease = entry.onLastRelease;
      publish();
      onLastRelease?.();
    },
  };
}

export function subscribePreparedDatasetRegistry(listener: RegistryListener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function resolvePreparedDataset<T extends VtkPreparationResult>(
  handle: VtkPreparedDatasetHandle<T>,
): T {
  const entry = entries.get(handle.id);
  if (!entry || entry.handle !== handle) {
    throw new Error(`Prepared dataset handle ${handle.id} is no longer retained`);
  }
  if (!entry.prepared) {
    throw new Error(`Prepared dataset handle ${handle.id} is worker-resident and cannot be resolved on main`);
  }
  return entry.prepared as T;
}

export function preparedDatasetRegistrySnapshot(): {
  handles: number;
  references: number;
  bytes: number;
  mainHandles: number;
  workerHandles: number;
  mainBytes: number;
  workerBytes: number;
} {
  let references = 0;
  let bytes = 0;
  let mainHandles = 0;
  let workerHandles = 0;
  let mainBytes = 0;
  let workerBytes = 0;
  for (const entry of entries.values()) {
    references += entry.references;
    bytes += entry.handle.bytes;
    if (entry.handle.residency === "main") {
      mainHandles += 1;
      mainBytes += entry.handle.bytes;
    } else {
      workerHandles += 1;
      workerBytes += entry.handle.bytes;
    }
  }
  return Object.freeze({
    handles: entries.size,
    references,
    bytes,
    mainHandles,
    workerHandles,
    mainBytes,
    workerBytes,
  });
}
