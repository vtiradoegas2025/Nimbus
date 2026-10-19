import type { NpyN } from "@/lib/platform/npy";
import { derivedFieldDefinition } from "./derivedFieldRegistry";
import {
  evaluateDerivedWorkerRequest,
  type DerivedFieldWorkerReply,
  type DerivedFieldWorkerRequest,
} from "./derivedFieldWorkerProtocol";
import { DERIVED_SLAB_CELLS, planSlabs, sliceInputs } from "./derivedFieldSlabs";
import derivedFieldWorkerUrl from "./derivedField.worker.ts?worker&url";

export interface DerivedFieldWorker {
  onmessage: ((event: { data: DerivedFieldWorkerReply }) => void) | null;
  onerror: ((event: { message: string; preventDefault?: () => void }) => void) | null;
  onmessageerror?: ((event: unknown) => void) | null;
  postMessage: (message: DerivedFieldWorkerRequest) => void;
  terminate: () => void;
  close?: () => void;
}

export type DerivedFieldWorkerFactory = () => DerivedFieldWorker;

export interface DerivedFieldWorkerOptions {
  signal?: AbortSignal;
  slabCells?: number;
  workerFactory?: DerivedFieldWorkerFactory;
  timeoutMs?: number;
}

let workerGeneration = 0;
function defaultWorkerFactory(): DerivedFieldWorker {
  const separator = derivedFieldWorkerUrl.includes("?") ? "&" : "?";
  const worker = new Worker(`${derivedFieldWorkerUrl}${separator}generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-derived-field",
  });
  const owned = worker as unknown as DerivedFieldWorker;
  owned.close = () => {
    const watchdog = setTimeout(() => worker.terminate(), 5_000);
    worker.onmessage = ({ data }) => {
      if (data?.closed !== true) return;
      clearTimeout(watchdog);
      worker.onmessage = null;
    };
    try {
      worker.postMessage({ kind: "close" });
    } catch {
      clearTimeout(watchdog);
      worker.terminate();
    }
  };
  return owned;
}

function abortError(): DOMException {
  return new DOMException("Derived field evaluation cancelled", "AbortError");
}

export function isDerivedEvaluationAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function referenceInput(
  fieldName: string,
  inputs: Readonly<Record<string, NpyN>>,
): NpyN {
  const definition = derivedFieldDefinition(fieldName);
  if (!definition) throw new Error(`Unknown derived field '${fieldName}'`);
  const reference = inputs[definition.dependencies[0].field];
  if (!reference) {
    throw new Error(`Derived field '${fieldName}' is missing '${definition.dependencies[0].field}'`);
  }
  return reference;
}

/**
 * Evaluate a built-in derived field off the UI thread, one slab at a time.
 *
 * Every built-in derived field is elementwise, so slabs are independent and reassembly is a
 * copy at an offset. Slabbing is what makes the worker usable at all: whole-field inputs for
 * theta_e are ~504 MB on a 512x512x160 run, far past any sane message size, and evaluating
 * them inline blocked the main thread for about three seconds per frame.
 */
export async function evaluateDerivedFieldInWorker(
  fieldName: string,
  inputs: Readonly<Record<string, NpyN>>,
  options: DerivedFieldWorkerOptions = {},
): Promise<NpyN> {
  const reference = referenceInput(fieldName, inputs);
  const shape = reference.shape;
  const length = shape.reduce((product, extent) => product * extent, 1);
  /*
   * Carried from the inputs: on a windowed read this records where the aggregated cells
   * physically sit. Dropping it would place a windowed derived field on the wrong geometry,
   * while its dependencies were drawn on the right one.
   */
  const windowGeometry = reference.windowGeometry;
  const output = new Float32Array(length);
  const slabs = planSlabs(length, options.slabCells ?? DERIVED_SLAB_CELLS);
  if (options.signal?.aborted) throw abortError();

  const factory = options.workerFactory ?? defaultWorkerFactory;
  const timeoutMs = options.timeoutMs ?? 30_000;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("Derived field timeout must be positive and finite");
  // Non-browser (test) fallback: evaluate in place when Worker is absent.
  if (!("Worker" in globalThis) && factory === defaultWorkerFactory) {
    for (const slab of slabs) {
      if (options.signal?.aborted) throw abortError();
      const result = evaluateDerivedWorkerRequest({
        fieldName,
        inputs: sliceInputs(inputs, slab),
      });
      output.set(result.data as Float32Array, slab.start);
    }
    return { shape: [...shape], data: output, ...(windowGeometry ? { windowGeometry } : {}) };
  }

  // One worker for the whole field: spawning per slab would cost more than the slab evaluates.
  const worker = factory();
  let completed = false;
  try {
    for (const slab of slabs) {
      if (options.signal?.aborted) throw abortError();
      const result = await new Promise<NpyN>((resolve, reject) => {
        let settled = false;
        const finish = (action: () => void) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          options.signal?.removeEventListener("abort", onAbort);
          worker.onmessage = null;
          worker.onerror = null;
          worker.onmessageerror = null;
          action();
        };
        const onAbort = () => finish(() => reject(abortError()));
        const timer = setTimeout(() => finish(() => reject(new Error("Derived field worker did not finish its slab before the timeout"))), timeoutMs);
        options.signal?.addEventListener("abort", onAbort, { once: true });
        worker.onmessage = ({ data }) => {
          finish(() => {
            if (!data?.ok || !data.result) { reject(new Error(data?.error || "Derived field evaluation failed")); return; }
            if (!(data.result.data instanceof Float32Array) || data.result.data.length !== slab.end - slab.start) {
              reject(new Error("Derived field worker returned an incomplete slab"));
              return;
            }
            resolve(data.result);
          });
        };
        worker.onerror = (event) => {
          event.preventDefault?.();
          finish(() => reject(new Error(event.message)));
        };
        worker.onmessageerror = () => finish(() => reject(new Error("Derived field worker returned an unreadable result")));
        try {
          worker.postMessage({ fieldName, inputs: sliceInputs(inputs, slab) });
        } catch (error) {
          finish(() => reject(error));
        }
      });
      output.set(result.data as Float32Array, slab.start);
    }
    if (options.signal?.aborted) throw abortError();
    completed = true;
  } finally {
    worker.onmessage = null;
    worker.onerror = (event) => event.preventDefault?.();
    worker.onmessageerror = null;
    // Closing after receipt avoids racing WebKit's transferable-result delivery.
    if (completed && worker.close) worker.close();
    else worker.terminate();
  }
  return { shape: [...shape], data: output, ...(windowGeometry ? { windowGeometry } : {}) };
}
