/*
 * Interpreter worker: dispatch table only. All Python lives in pyodideEngine.
 * Ops are serialised through `queue` (interpreter is single-threaded).
 * callClient is the reverse channel for load_field (only the main thread can read frames).
 */

import {
  bindContext,
  bindStagedFiles,
  completeNames,
  discardInterpreter,
  getPyodide,
  inspectNamespace,
  interpreterIsResponsive,
  installPackage,
  interpreterHeapBytes,
  loadedPackageNames,
  loadPyodidePackage,
  micropipInstall,
  readProducedFile,
  removeStagedFile,
  runPythonCell,
  writeStagedFile,
} from "./pyodideEngine";
import {
  CLIENT_OPS,
  WORKER_OPS,
  type FieldWindow,
  type NpyN,
  type PyodideStatus,
  type RpcMessage,
  type RpcRequest,
  type SerializableInjectionContext,
  type SerializableRunOptions,
  type SerializedError,
} from "./pyodideProtocol";

/* eslint-disable @typescript-eslint/no-explicit-any */

// Typed structurally (not DedicatedWorkerGlobalScope): adding the webworker lib
// alongside dom redeclares hundreds of globals with conflicting signatures.
const scope = self as unknown as {
  postMessage(message: RpcMessage, transfer?: Transferable[]): void;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<RpcMessage>) => void,
  ): void;
};

function post(message: RpcMessage, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) scope.postMessage(message, transfer);
  else scope.postMessage(message);
}

/** Worker-owned response buffers can move to the UI without another full clone.
 * Field inputs are intentionally absent: those can alias the main-thread frame cache. */
function responseTransferables(op: string, value: unknown): Transferable[] {
  const buffers = new Set<ArrayBuffer>();
  if (op === WORKER_OPS.readProducedFile && value instanceof Uint8Array
      && value.buffer instanceof ArrayBuffer) {
    buffers.add(value.buffer);
  }
  if (op === WORKER_OPS.runCell && value && typeof value === "object") {
    const outputs = (value as { outputs?: unknown }).outputs;
    if (Array.isArray(outputs)) {
      for (const output of outputs) {
        if (!output || typeof output !== "object") continue;
        const data = (output as { kind?: unknown; data?: unknown }).data;
        if ((output as { kind?: unknown }).kind === "ndarray"
            && ArrayBuffer.isView(data) && data.buffer instanceof ArrayBuffer) {
          buffers.add(data.buffer);
        }
      }
    }
  }
  return [...buffers];
}

function emitStatus(status: PyodideStatus): void {
  post({ kind: "event", event: "status", status });
}

// Flatten for postMessage: structured clone drops subclasses (instanceof checks fail on the far side).
function serializeError(error: unknown, fatal: boolean): SerializedError {
  if (error instanceof Error) {
    const extra = error as Error & {
      pkg?: string; underlying?: string; hostPath?: string; byteLength?: number;
    };
    return {
      name: error.name,
      message: error.message,
      pkg: extra.pkg,
      underlying: extra.underlying,
      hostPath: extra.hostPath,
      byteLength: extra.byteLength,
      fatal,
    };
  }
  return { name: "Error", message: String(error), fatal };
}

// Probe after every failure: a wasm heap abort raises unrelated errors; only the probe is reliable.
// Drops the dead interpreter but does NOT replace it (held memory; client must restart the worker).
function survived(): boolean {
  if (interpreterIsResponsive()) return true;
  discardInterpreter();
  return false;
}

/* ---------- Reverse channel: asking the main thread ---------- */

let nextClientRequestId = 1;
const pendingClientCalls = new Map<
  number,
  { resolve: (value: unknown) => void; reject: (error: Error) => void }
>();

function callClient(op: string, args: readonly unknown[]): Promise<unknown> {
  const id = nextClientRequestId++;
  return new Promise((resolve, reject) => {
    pendingClientCalls.set(id, { resolve, reject });
    try {
      post({ kind: "request", id, op, args });
    } catch (error) {
      pendingClientCalls.delete(id);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

/* ---------- Serialised op queue ---------- */

let queue: Promise<unknown> = Promise.resolve();

function enqueue<T>(work: () => Promise<T>): Promise<T> {
  const next = queue.then(work, work);
  // Swallow on the chain only so one failed op doesn't poison all subsequent ones.
  queue = next.then(() => undefined, () => undefined);
  return next;
}

/* ---------- Ops ---------- */

async function handle(request: RpcRequest): Promise<unknown> {
  const args = request.args;
  switch (request.op) {
    case WORKER_OPS.boot: {
      const [interruptBuffer] = args as [Uint8Array | undefined];
      await getPyodide(emitStatus, interruptBuffer);
      return null;
    }
    case WORKER_OPS.bindContext: {
      const [context] = args as [SerializableInjectionContext];
      const handleRef = await getPyodide(emitStatus);
      await bindContext(handleRef, {
        ...context,
        // Rebuild the callback here (can't clone); keep undefined when client won't serve frames.
        loadFrameField: context.canLoadFrames
          ? (fieldName: string, frameIndex: number, source: string, window?: FieldWindow) =>
            callClient(
              CLIENT_OPS.loadFrameField,
              [fieldName, frameIndex, source, window],
            ) as Promise<NpyN>
          : undefined,
      });
      return null;
    }
    case WORKER_OPS.runCell: {
      const [source, options] = args as [string, SerializableRunOptions];
      const handleRef = await getPyodide(emitStatus);
      return runPythonCell(
        handleRef,
        source,
        {
          resetProducedFiles: options.resetProducedFiles,
          resultOnly: options.resultOnly,
          onStream: options.streaming
            ? (stream: "stdout" | "stderr", text: string) =>
              post({ kind: "event", event: "stream", stream, text })
            : undefined,
        },
        // Forward status so "Loading scipy…" reaches the badge during cell pre-load.
        emitStatus,
      );
    }
    case WORKER_OPS.loadPackage: {
      const [importName] = args as [string];
      await loadPyodidePackage(importName, emitStatus);
      return null;
    }
    case WORKER_OPS.micropipInstall: {
      const [importName] = args as [string];
      await micropipInstall(importName, emitStatus);
      return null;
    }
    case WORKER_OPS.installPackage: {
      const [importName] = args as [string];
      await installPackage(importName, emitStatus);
      return null;
    }
    // Set doesn't survive structured clone in all engines; client rebuilds from the array.
    case WORKER_OPS.loadedPackages:
      return [...loadedPackageNames()];
    case WORKER_OPS.complete: {
      const [path, partial] = args as [readonly string[], string];
      return completeNames(path, partial);
    }
    case WORKER_OPS.inspectNamespace:
      return inspectNamespace();
    case WORKER_OPS.heapBytes:
      return interpreterHeapBytes();
    case WORKER_OPS.readProducedFile: {
      const [file] = args as [any];
      const handleRef = await getPyodide(emitStatus);
      return readProducedFile(handleRef, file);
    }
    case WORKER_OPS.writeStagedFile: {
      const [directory, virtualPath, bytes] = args as [string, string, Uint8Array];
      const handleRef = await getPyodide(emitStatus);
      writeStagedFile(handleRef, directory, virtualPath, bytes);
      return null;
    }
    case WORKER_OPS.removeStagedFile: {
      const [virtualPath] = args as [string];
      const handleRef = await getPyodide(emitStatus);
      removeStagedFile(handleRef, virtualPath);
      return null;
    }
    case WORKER_OPS.bindStagedFiles: {
      const [staged] = args as [ReadonlyArray<{ name: string; virtualPath: string }>];
      const handleRef = await getPyodide(emitStatus);
      bindStagedFiles(handleRef, staged);
      return null;
    }
    default:
      throw new Error(`Unknown interpreter op: ${request.op}`);
  }
}

scope.addEventListener("message", (event: MessageEvent<RpcMessage>) => {
  const message = event.data;
  if (message.kind === "response") {
    // A reply to something WE asked the main thread for.
    const pending = pendingClientCalls.get(message.id);
    if (!pending) return;
    pendingClientCalls.delete(message.id);
    if (message.ok) pending.resolve(message.value);
    else pending.reject(Object.assign(new Error(message.error.message), { name: message.error.name }));
    return;
  }
  if (message.kind !== "request") return;

  void enqueue(async () => {
    try {
      const value = await handle(message);
      post(
        { kind: "response", id: message.id, ok: true, value },
        responseTransferables(message.op, value),
      );
    } catch (error) {
      post({
        kind: "response",
        id: message.id,
        ok: false,
        error: serializeError(error, !survived()),
      });
    }
  });
});
