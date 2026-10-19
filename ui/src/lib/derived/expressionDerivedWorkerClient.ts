import type {
  CompiledExpressionDerivedField,
  ExpressionDerivedFieldDefinition,
} from "./expressionDerivedFields";
import type { NpyN } from "@/lib/platform/npy";
import expressionDerivedWorkerUrl from "./expressionDerived.worker.ts?worker&url";
import {
  evaluateExpressionWorkerRequest,
  type ExpressionDerivedWorkerReply,
  type ExpressionDerivedWorkerRequest,
} from "./expressionDerivedWorkerProtocol";

const MAX_WORKER_INPUT_BYTES = 256 * 1024 * 1024;

export interface ExpressionDerivedWorker {
  onmessage: ((event: { data: ExpressionDerivedWorkerReply }) => void) | null;
  onerror: ((event: { message: string }) => void) | null;
  onmessageerror?: ((event: unknown) => void) | null;
  postMessage: (message: ExpressionDerivedWorkerRequest) => void;
  terminate: () => void;
}

export type ExpressionDerivedWorkerFactory = () => ExpressionDerivedWorker;

export interface ExpressionWorkerOptions {
  signal?: AbortSignal;
  maxElements?: number;
  workerFactory?: ExpressionDerivedWorkerFactory;
  timeoutMs?: number;
}

const workerSession = Date.now().toString(36);
let workerGeneration = 0;

function defaultWorkerFactory(): ExpressionDerivedWorker {
  const separator = expressionDerivedWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${expressionDerivedWorkerUrl}${separator}session=${workerSession}&generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-expression-derived-field",
  }) as unknown as ExpressionDerivedWorker;
}

function abortError(): DOMException {
  return new DOMException("Expression evaluation cancelled", "AbortError");
}

export function isExpressionEvaluationAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function requestBytes(inputs: Readonly<Record<string, NpyN>>): number {
  let total = 0;
  for (const input of Object.values(inputs)) {
    if (!(input.data instanceof Float32Array)) throw new Error("Expression worker inputs must be Float32 arrays");
    total += input.data.byteLength;
    if (!Number.isSafeInteger(total) || total > MAX_WORKER_INPUT_BYTES) {
      throw new Error(
        `expression worker inputs exceed the ${MAX_WORKER_INPUT_BYTES / (1024 * 1024)} MiB admission limit; use a slice or lower-resolution source`,
      );
    }
  }
  return total;
}

/** Evaluate off the UI thread. Inputs are cloned (not transferred) to keep
 *  cached buffers intact. Abort terminates the worker immediately. */
export function evaluateExpressionDerivedInWorker(
  definitionOrCompiled: ExpressionDerivedFieldDefinition | CompiledExpressionDerivedField,
  inputs: Readonly<Record<string, NpyN>>,
  options: ExpressionWorkerOptions = {},
): Promise<NpyN> {
  const definition = "compiled" in definitionOrCompiled
    ? definitionOrCompiled.definition
    : definitionOrCompiled;
  requestBytes(inputs);
  if (options.signal?.aborted) return Promise.reject(abortError());
  const timeoutMs = options.timeoutMs ?? 30_000;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("Expression worker timeout must be positive and finite");
  const request: ExpressionDerivedWorkerRequest = {
    definition,
    inputs: Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, {
      shape: [...input.shape],
      data: input.data.buffer instanceof ArrayBuffer && input.data.byteOffset === 0
        && input.data.byteLength === input.data.buffer.byteLength ? input.data
        : new Float32Array(new Uint8Array(input.data.buffer, input.data.byteOffset, input.data.byteLength).slice().buffer),
    }])),
    maxElements: options.maxElements,
  };

  const factory = options.workerFactory ?? defaultWorkerFactory;
  // Non-browser (test) fallback: run synchronously when Worker is absent.
  if (!("Worker" in globalThis) && factory === defaultWorkerFactory) {
    return Promise.resolve().then(() => {
      if (options.signal?.aborted) throw abortError();
      return evaluateExpressionWorkerRequest(request);
    });
  }

  return new Promise((resolve, reject) => {
    const worker = factory();
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const finish = (action: () => void) => {
      if (settled) return;
      settled = true;
      if (timer != null) clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
      worker.terminate();
      action();
    };
    const onAbort = () => finish(() => reject(abortError()));
    options.signal?.addEventListener("abort", onAbort, { once: true });
    worker.onmessage = ({ data }) => {
      if (data.ok && data.result) finish(() => resolve(data.result!));
      else finish(() => reject(new Error(data.error || "Expression evaluation failed")));
    };
    worker.onerror = ({ message }) => finish(() => reject(new Error(message || "Expression worker failed to start or execute")));
    worker.onmessageerror = () => finish(() => reject(new Error("Expression worker returned an unreadable result")));
    timer = setTimeout(() => finish(() => reject(new Error("Expression worker startup or evaluation exceeded its timeout"))), timeoutMs);
    try { worker.postMessage(request); }
    catch (error) { finish(() => reject(error)); }
  });
}
