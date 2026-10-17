import type { FieldWindowGeometry, Npy3D } from "@/lib/platform/npy";
import type { VtkPreparationRequest } from "./vtkPreparation";
import { vtkPreparationInputBytes } from "./vtkPreparationResourcePlanning";
import { recordVtkDatasetHandoff } from "./vtkDatasetHandoff";

const COPY_CHUNK_BYTES = 2 * 1024 * 1024;

export interface VtkPreparationInputHandoff<T extends VtkPreparationRequest> {
  request: T;
  transfer: ArrayBuffer[];
  copiedBytes: number;
}

export type VtkPreparationInputYieldMode = "scheduler-yield" | "message-channel" | "microtask";

export function vtkPreparationInputYieldMode(): VtkPreparationInputYieldMode {
  const scheduler = (globalThis as typeof globalThis & {
    scheduler?: { yield?: () => Promise<void> };
  }).scheduler;
  if (scheduler?.yield) return "scheduler-yield";
  if (typeof MessageChannel !== "undefined") return "message-channel";
  return "microtask";
}

function abortError(): DOMException {
  return new DOMException("VTK input handoff cancelled", "AbortError");
}

function yieldToEventLoop(): Promise<void> {
  const scheduler = (globalThis as typeof globalThis & {
    scheduler?: { yield?: () => Promise<void> };
  }).scheduler;
  if (scheduler?.yield) return scheduler.yield();
  if (typeof MessageChannel === "undefined") {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      channel.port2.close();
      resolve();
    };
    channel.port2.postMessage(null);
  });
}

async function cloneBuffer(
  source: ArrayBufferLike,
  copies: Map<ArrayBufferLike, Promise<ArrayBufferLike>>,
  transfers: Set<ArrayBuffer>,
  cancelled: () => boolean,
): Promise<ArrayBufferLike> {
  if (typeof SharedArrayBuffer !== "undefined" && source instanceof SharedArrayBuffer) return source;
  const existing = copies.get(source);
  if (existing) return existing;
  const pending = (async () => {
    if (source.byteLength === 0) throw new Error("Cannot hand off a detached VTK input buffer");
    const target = new ArrayBuffer(source.byteLength);
    const input = new Uint8Array(source);
    const output = new Uint8Array(target);
    for (let offset = 0; offset < input.length; offset += COPY_CHUNK_BYTES) {
      if (cancelled()) throw abortError();
      output.set(input.subarray(offset, Math.min(input.length, offset + COPY_CHUNK_BYTES)), offset);
      if (offset + COPY_CHUNK_BYTES < input.length) await yieldToEventLoop();
    }
    transfers.add(target);
    return target;
  })();
  copies.set(source, pending);
  return pending;
}

async function cloneView<T extends Float32Array | Float64Array>(
  source: T,
  copies: Map<ArrayBufferLike, Promise<ArrayBufferLike>>,
  transfers: Set<ArrayBuffer>,
  cancelled: () => boolean,
): Promise<T> {
  const buffer = await cloneBuffer(source.buffer, copies, transfers, cancelled);
  const Constructor = source.constructor as {
    new(buffer: ArrayBufferLike, byteOffset: number, length: number): T;
  };
  return new Constructor(buffer, source.byteOffset, source.length);
}

async function cloneWindowGeometry(
  source: FieldWindowGeometry | undefined,
  copies: Map<ArrayBufferLike, Promise<ArrayBufferLike>>,
  transfers: Set<ArrayBuffer>,
  cancelled: () => boolean,
): Promise<FieldWindowGeometry | undefined> {
  if (!source) return undefined;
  const centroidsM = source.centroidsM ? {
    x: await cloneView(source.centroidsM.x, copies, transfers, cancelled),
    y: await cloneView(source.centroidsM.y, copies, transfers, cancelled),
    z: await cloneView(source.centroidsM.z, copies, transfers, cancelled),
  } : null;
  return {
    ...source,
    centroidsM,
    volumesM3: source.volumesM3
      ? await cloneView(source.volumesM3, copies, transfers, cancelled)
      : source.volumesM3,
  };
}

async function cloneField(
  source: Npy3D,
  copies: Map<ArrayBufferLike, Promise<ArrayBufferLike>>,
  transfers: Set<ArrayBuffer>,
  cancelled: () => boolean,
): Promise<Npy3D> {
  return {
    ...source,
    data: await cloneView(source.data, copies, transfers, cancelled),
    windowGeometry: await cloneWindowGeometry(source.windowGeometry, copies, transfers, cancelled),
  };
}

export async function handoffVtkPreparationInput<T extends VtkPreparationRequest>(
  request: T,
  cancelled: () => boolean = () => false,
): Promise<VtkPreparationInputHandoff<T>> {
  const copies = new Map<ArrayBufferLike, Promise<ArrayBufferLike>>();
  const transfers = new Set<ArrayBuffer>();
  let handedOff: VtkPreparationRequest;
  if (request.kind === "scalar") {
    handedOff = {
      ...request,
      field: await cloneField(request.field, copies, transfers, cancelled),
      subtractField: request.subtractField
        ? await cloneField(request.subtractField, copies, transfers, cancelled)
        : undefined,
    };
  } else if (request.kind === "cloud") {
    handedOff = {
      ...request,
      field: await cloneField(request.field, copies, transfers, cancelled),
      subtractField: request.subtractField
        ? await cloneField(request.subtractField, copies, transfers, cancelled)
        : undefined,
      flowFields: request.flowFields ? [
        await cloneField(request.flowFields[0], copies, transfers, cancelled),
        await cloneField(request.flowFields[1], copies, transfers, cancelled),
        await cloneField(request.flowFields[2], copies, transfers, cancelled),
      ] : undefined,
    };
  } else {
    handedOff = {
      ...request,
      fields: [
        await cloneField(request.fields[0], copies, transfers, cancelled),
        await cloneField(request.fields[1], copies, transfers, cancelled),
        await cloneField(request.fields[2], copies, transfers, cancelled),
      ],
    };
  }
  if (cancelled()) throw abortError();
  const copiedBytes = [...transfers].reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
  const logicalBytes = vtkPreparationInputBytes(request);
  recordVtkDatasetHandoff({
    mode: copiedBytes === 0 ? "shared-zero-copy" : "transferable-one-copy",
    logicalBytes,
    copiedBytes,
    datasets: 1,
    stage: "preparation",
    dataset: `worker-owned-${request.kind}-input`,
    sourceRetained: true,
    reason: copiedBytes > 0
      ? "Worker-owned preparation stages cancellable input copies before transfer"
      : undefined,
  });
  return {
    request: handedOff as T,
    transfer: [...transfers],
    copiedBytes,
  };
}
