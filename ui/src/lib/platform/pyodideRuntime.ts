/*
 * Main-thread client for the interpreter worker.
 * Allocates the SharedArrayBuffer interrupt sentinel here so interruptPython()
 * can write it synchronously while Python blocks the worker thread.
 */

import pyodideWorkerUrl from "./pyodideWorker.ts?worker&url";
import {
  CLIENT_OPS,
  WORKER_OPS,
  type CompletionCandidate,
  type FieldWindow,
  type InjectionContext,
  type NamespaceEntry,
  type NpyN,
  type ProducedFile,
  type PyodideStatusListener,
  type RpcMessage,
  type RpcRequest,
  type RunOptions,
  type RunResult,
  type SerializedError,
} from "./pyodideProtocol";

export {
  MAX_ANIMATION_FRAMES,
  MAX_COMPLETIONS,
  MAX_OUTPUTS_PER_RUN,
  PYODIDE_OUTPUT_DIR,
  WASM32_ADDRESS_CEILING_BYTES,
  type FieldBinding,
  type InjectionContext,
  type NamespaceEntry,
  type ProducedFile,
  type PyodideStatus,
  type PyodideStatusListener,
  type PythonOutput,
  type RunOptions,
  type RunResult,
} from "./pyodideProtocol";

// Re-exported for backward-compatible imports; pure table reads, no interpreter needed.
export {
  detectMissingModule,
  isBundledPyodideImport,
  packageNameForImport,
} from "./pyodideImports";

/**
 * Thrown when loadPackage rejects a package that isn't in the curated
 * Pyodide index. The UI catches this specifically to offer the
 * micropip fallback path.
 */
export class MissingPackageError extends Error {
  readonly pkg: string;
  readonly underlying: string;

  constructor(pkg: string, underlying: string) {
    super(
      `Pyodide doesn't ship a '${pkg}' package. Try the micropip fallback for pure-Python wheels on PyPI.`,
    );
    this.name = "MissingPackageError";
    this.pkg = pkg;
    this.underlying = underlying;
  }
}

/** Interpreter is dead; only a worker restart recovers it. */
export class PyodideFatalError extends Error {
  constructor(detail: string) {
    super(
      `The Python interpreter stopped and cannot continue (${detail}). `
      + `Restart it to keep working; the namespace does not survive.`,
    );
    this.name = "PyodideFatalError";
  }
}

/** The user explicitly replaced the interpreter; not a cell error. */
export class PythonStoppedError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = "PythonStoppedError";
  }
}

export class PythonStartupError extends Error {
  constructor(detail: string) {
    super(`Python could not finish starting: ${detail}. This startup did not run your script. Try Run again or Restart.`);
    this.name = "PythonStartupError";
  }
}

export const PYTHON_BOOT_ATTEMPT_TIMEOUT_MS = 30_000;
export const PYTHON_BOOT_TOTAL_TIMEOUT_MS = 75_000;

/* ---------- Transport ---------- */

// Interrupt sentinel. crossOriginIsolated is checked explicitly (not just typeof SharedArrayBuffer):
// on Chromium the constructor exists without isolation but the worker gets a private copy,
// silently breaking Stop. Isolation is what determines whether the worker sees the write.
const interruptShared = typeof SharedArrayBuffer !== "undefined"
  && typeof crossOriginIsolated !== "undefined"
  && crossOriginIsolated;

// WebKit can reject the second worker when two large CPython/WASM runtimes
// compile at exactly the same time. Serialize cold boot only; initialized
// clients remain separate workers and execute concurrently afterwards.
let workerBootTail: Promise<void> = Promise.resolve();
function bootAbortReason(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new PythonStoppedError("Python startup was cancelled");
}
function inWorkerBootSlot<T>(start: () => Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(bootAbortReason(signal));
  const result = workerBootTail.catch(() => undefined).then(() => {
    if (signal.aborted) throw bootAbortReason(signal);
    return start();
  });
  workerBootTail = result.then(() => undefined, () => undefined);
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(bootAbortReason(signal));
    signal.addEventListener("abort", abort, { once: true });
    void result.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}

const workerPageToken = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
  ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
let workerInstanceSequence = 0;

function reviveError(error: SerializedError): Error {
  // fatal checked first: a dead-interpreter error must surface as PyodideFatalError,
  // not as whatever unrelated error the corpse happened to raise.
  if (error.fatal) {
    return new PyodideFatalError(error.message);
  }
  if (error.name === "MissingPackageError") {
    return new MissingPackageError(error.pkg ?? "", error.underlying ?? error.message);
  }
  const revived = new Error(error.message);
  revived.name = error.name;
  if (error.hostPath !== undefined) {
    Object.assign(revived, { hostPath: error.hostPath, byteLength: error.byteLength });
  }
  return revived;
}

/** Opaque session token; the interpreter is accessed only through its owning client. */
export interface PyodideHandle {
  readonly ready: true;
  readonly runtimeKey: string;
}

type PendingCall = { resolve: (value: unknown) => void; reject: (error: Error) => void };

export interface PythonRuntimeClient {
  readonly key: string;
  canInterrupt(): boolean;
  pythonNeedsRestart(): boolean;
  restartPython(): void;
  dispose(): void;
  interruptPython(): boolean;
  isCellInFlight(): boolean;
  isBusy(): boolean;
  tryAcquireExecution(): boolean;
  releaseExecution(): void;
  getPyodide(onStatus?: PyodideStatusListener): Promise<PyodideHandle>;
  loadPyodidePackage(importName: string, onStatus?: PyodideStatusListener): Promise<void>;
  micropipInstall(importName: string, onStatus?: PyodideStatusListener): Promise<void>;
  installPackage(importName: string, onStatus?: PyodideStatusListener): Promise<void>;
  loadedPackageNames(): Promise<ReadonlySet<string>>;
  bindContext(handle: PyodideHandle, ctx: InjectionContext): Promise<void>;
  runPythonCell(handle: PyodideHandle, source: string, options?: RunOptions): Promise<RunResult>;
  completeNames(path: readonly string[], partial: string): Promise<readonly CompletionCandidate[]>;
  inspectNamespace(): Promise<readonly NamespaceEntry[]>;
  interpreterHeapBytes(): Promise<number | null>;
  readProducedFile(handle: PyodideHandle, file: ProducedFile): Promise<Uint8Array>;
  writeStagedFile(directory: string, virtualPath: string, bytes: Uint8Array): Promise<void>;
  removeStagedFile(virtualPath: string): Promise<void>;
  bindStagedFilesInPython(staged: ReadonlyArray<{ name: string; virtualPath: string }>): Promise<void>;
}

class WorkerRuntimeClient implements PythonRuntimeClient {
  readonly key: string;
  private worker: Worker | null = null;
  private nextRequestId = 1;
  private readonly pending = new Map<number, PendingCall>();
  private readonly interruptBuffer = new Uint8Array(
    interruptShared ? new SharedArrayBuffer(1) : new ArrayBuffer(1),
  );
  private cellInFlight = false;
  private executionClaimed = false;
  private dead = false;
  private disposed = false;
  private booted = false;
  private workerGeneration = 0;
  private frameLoader: InjectionContext["loadFrameField"] = undefined;
  private statusListener: PyodideStatusListener | undefined;
  private streamListener: RunOptions["onStream"];
  private booting: Promise<unknown> | null = null;
  private bootController: AbortController | null = null;
  private readonly handle: PyodideHandle;
  private readonly onIdle: (() => void) | undefined;

  constructor(key: string, onIdle?: () => void) {
    this.key = key;
    this.onIdle = onIdle;
    this.handle = Object.freeze({ ready: true as const, runtimeKey: key });
  }

  canInterrupt(): boolean { return interruptShared; }
  pythonNeedsRestart(): boolean { return this.dead; }
  isCellInFlight(): boolean { return this.cellInFlight; }
  isBusy(): boolean {
    return this.executionClaimed || this.cellInFlight || this.pending.size > 0 || this.bootController !== null;
  }
  tryAcquireExecution(): boolean {
    if (this.disposed || this.isBusy()) return false;
    this.executionClaimed = true;
    return true;
  }
  releaseExecution(): void {
    this.executionClaimed = false;
    this.onIdle?.();
  }

  private ensureWorker(): Worker {
    if (this.disposed) throw new PythonStoppedError("The Python runtime was disposed");
    if (this.worker) return this.worker;
    // A distinct module-worker URL per runtime is required on WebKit: reusing
    // one URL for two retained Pyodide workers can make the second fail during
    // module startup. It also keeps restart generations unambiguously fresh.
    const separator = pyodideWorkerUrl.includes("?") ? "&" : "?";
    const suffix = `${separator}runtime=${encodeURIComponent(this.key)}`
      + `&generation=${this.workerGeneration}&instance=${workerPageToken}-${++workerInstanceSequence}`;
    const created = new Worker(`${pyodideWorkerUrl}${suffix}`, {
      type: "module",
      name: `nimbus-pyodide-${this.key}`,
    });
    created.addEventListener("message", (event: MessageEvent<RpcMessage>) => {
      if (this.worker !== created) return;
      void this.onWorkerMessage(created, event.data);
    });
    created.addEventListener("error", (event) => {
      if (this.worker !== created) return;
      const message = event.message || "Interpreter worker failed to start";
      this.dead = this.booted;
      if (this.booted) this.statusListener?.({ kind: "error", message });
      for (const [, waiter] of this.pending) waiter.reject(new PyodideFatalError(message));
      this.pending.clear();
    });
    this.worker = created;
    return created;
  }

  private async onWorkerMessage(target: Worker, message: RpcMessage): Promise<void> {
    if (this.worker !== target) return;
    if (message.kind === "event") {
      if (message.event === "status") this.statusListener?.(message.status);
      else this.streamListener?.(message.stream, message.text);
      return;
    }
    if (message.kind === "response") {
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      this.pending.delete(message.id);
      if (message.ok) waiter.resolve(message.value);
      else {
        const error = reviveError(message.error);
        if (error instanceof PyodideFatalError && this.booted) this.dead = true;
        waiter.reject(error);
      }
      // Promise continuations run first; only then may an overflow LRU dispose
      // a client whose last RPC has become idle.
      if (!this.isBusy()) queueMicrotask(() => {
        if (!this.isBusy()) this.onIdle?.();
      });
      return;
    }
    await this.serveWorkerRequest(target, message);
  }

  private async serveWorkerRequest(target: Worker, request: RpcRequest): Promise<void> {
    try {
      if (request.op !== CLIENT_OPS.loadFrameField) throw new Error(`Unknown client op: ${request.op}`);
      if (!this.frameLoader) throw new Error("No frame loader is bound");
      const [fieldName, frameIndex, source, window] = request.args as [
        string, number, string, FieldWindow | undefined,
      ];
      const field = await this.frameLoader(fieldName, frameIndex, source, window);
      // Deliberately clone: the returned field may alias the shared frame cache.
      if (this.worker === target) {
        target.postMessage({ kind: "response", id: request.id, ok: true, value: field });
      }
    } catch (error) {
      if (this.worker !== target) return;
      target.postMessage({
        kind: "response",
        id: request.id,
        ok: false,
        error: {
          name: error instanceof Error ? error.name : "Error",
          message: error instanceof Error ? error.message : String(error),
        },
      });
    }
  }

  private send(op: string, args: readonly unknown[], transfer: Transferable[] = []): Promise<unknown> {
    const target = this.ensureWorker();
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        target.postMessage({ kind: "request", id, op, args } satisfies RpcRequest, transfer);
      } catch (error) {
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private retireWorker(target: Worker, error: Error): void {
    if (this.worker !== target) return;
    this.worker = null;
    target.terminate();
    for (const [, waiter] of this.pending) waiter.reject(error);
    this.pending.clear();
  }

  private async bootAttempt(signal: AbortSignal): Promise<unknown> {
    if (signal.aborted) throw bootAbortReason(signal);
    const target = this.ensureWorker();
    const abort = () => this.retireWorker(target, bootAbortReason(signal));
    const timer = setTimeout(() => this.retireWorker(target,
      new Error("an initialization attempt exceeded 30 seconds")), PYTHON_BOOT_ATTEMPT_TIMEOUT_MS);
    signal.addEventListener("abort", abort, { once: true });
    try {
      return await this.send(WORKER_OPS.boot, [this.interruptBuffer]);
    } catch (error) {
      this.retireWorker(target, error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
    }
  }

  private ensureBooted(): Promise<unknown> {
    if (this.disposed) return Promise.reject(new PythonStoppedError("The Python runtime was disposed"));
    if (this.dead) return Promise.reject(new PyodideFatalError("it was not restarted after the last failure"));
    if (this.booting) return this.booting;
    const generation = this.workerGeneration;
    const controller = new AbortController();
    this.bootController = controller;
    const timer = setTimeout(() => controller.abort(new PythonStartupError(
      "initialization, including its queue wait, exceeded 75 seconds")), PYTHON_BOOT_TOTAL_TIMEOUT_MS);
    const booting = (async () => {
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          await inWorkerBootSlot(() => this.bootAttempt(controller.signal), controller.signal);
          if (controller.signal.aborted) throw bootAbortReason(controller.signal);
          if (this.workerGeneration !== generation) throw new PythonStoppedError("The interpreter was replaced while loading");
          this.booted = true;
          return;
        } catch (error) {
          if (controller.signal.aborted) throw bootAbortReason(controller.signal);
          if (this.workerGeneration !== generation || this.disposed) {
            throw new PythonStoppedError("The interpreter was replaced while loading");
          }
          if (attempt === 1) throw new PythonStartupError(`both initialization attempts failed (${error instanceof Error ? error.message : String(error)})`);
          this.statusListener?.({ kind: "loading-runtime" });
        }
      }
    })();
    this.booting = booting;
    const finish = (error?: unknown) => {
      clearTimeout(timer);
      if (this.bootController === controller) this.bootController = null;
      if (error !== undefined && this.booting === booting) {
        this.booting = null;
        if (!(error instanceof PythonStoppedError)) this.statusListener?.({
          kind: "error", message: error instanceof Error ? error.message : String(error),
        });
      }
      if (!this.isBusy()) queueMicrotask(() => { if (!this.isBusy()) this.onIdle?.(); });
    };
    void booting.then(() => finish(), finish);
    return booting;
  }

  private async call(
    op: string,
    args: readonly unknown[] = [],
    transfer: Transferable[] = [],
  ): Promise<unknown> {
    if (this.dead) throw new PyodideFatalError("it was not restarted after the last failure");
    const generation = this.workerGeneration;
    if (op !== WORKER_OPS.boot) await this.ensureBooted();
    if (this.workerGeneration !== generation) throw new PythonStoppedError("The interpreter was replaced before this operation started");
    const value = await this.send(op, args, transfer);
    if (this.workerGeneration !== generation) throw new PythonStoppedError("The interpreter was replaced while this operation finished");
    return value;
  }

  restartPython(): void {
    this.workerGeneration += 1;
    this.bootController?.abort(new PythonStoppedError("The interpreter was replaced before startup finished"));
    this.bootController = null;
    if (this.worker) this.retireWorker(this.worker, new PythonStoppedError("The interpreter was replaced before this finished"));
    this.booting = null;
    this.booted = false;
    this.dead = false;
    this.cellInFlight = false;
    this.frameLoader = undefined;
    this.streamListener = undefined;
    this.interruptBuffer[0] = 0;
    for (const [, waiter] of this.pending) {
      waiter.reject(new PythonStoppedError("The interpreter was replaced before this finished"));
    }
    this.pending.clear();
  }

  dispose(): void {
    this.restartPython();
    this.disposed = true;
    this.statusListener = undefined;
    if (scopedClients.get(this.key) === this) scopedClients.delete(this.key);
  }

  interruptPython(): boolean {
    if (this.bootController) {
      this.bootController.abort(new PythonStoppedError("Python startup was stopped"));
      return true;
    }
    if (!interruptShared || !this.worker) return false;
    this.interruptBuffer[0] = 2;
    return true;
  }

  async getPyodide(onStatus?: PyodideStatusListener): Promise<PyodideHandle> {
    if (onStatus) this.statusListener = onStatus;
    const generation = this.workerGeneration;
    await this.ensureBooted();
    if (this.workerGeneration !== generation) throw new PythonStoppedError("The interpreter was replaced while loading");
    onStatus?.({ kind: "ready" });
    return this.handle;
  }

  private assertHandle(handle: PyodideHandle): void {
    if (handle.runtimeKey !== this.key) throw new Error("Pyodide handle belongs to a different runtime session");
  }

  async loadPyodidePackage(importName: string, onStatus?: PyodideStatusListener): Promise<void> {
    if (onStatus) this.statusListener = onStatus;
    await this.call(WORKER_OPS.loadPackage, [importName]);
  }

  async micropipInstall(importName: string, onStatus?: PyodideStatusListener): Promise<void> {
    if (onStatus) this.statusListener = onStatus;
    await this.call(WORKER_OPS.micropipInstall, [importName]);
  }

  async installPackage(importName: string, onStatus?: PyodideStatusListener): Promise<void> {
    if (onStatus) this.statusListener = onStatus;
    await this.call(WORKER_OPS.installPackage, [importName]);
  }

  async loadedPackageNames(): Promise<ReadonlySet<string>> {
    if (!this.worker) return new Set();
    return new Set(await this.call(WORKER_OPS.loadedPackages) as string[]);
  }

  async bindContext(handle: PyodideHandle, ctx: InjectionContext): Promise<void> {
    this.assertHandle(handle);
    const generation = this.workerGeneration;
    const { loadFrameField: _omitted, ...serializable } = ctx;
    await this.call(WORKER_OPS.bindContext, [
      {
        ...serializable,
        fields: ctx.fields.map(({ name, field }) => ({ name, field: { shape: field.shape, data: field.data } })),
        canLoadFrames: Boolean(ctx.loadFrameField),
      },
    ]);
    if (this.workerGeneration !== generation) throw new PythonStoppedError("The interpreter was replaced while binding fields");
    this.frameLoader = ctx.loadFrameField;
  }

  async runPythonCell(
    handle: PyodideHandle,
    source: string,
    options: RunOptions = {},
  ): Promise<RunResult> {
    this.assertHandle(handle);
    if (this.cellInFlight) throw new Error("This Python runtime is already executing a cell");
    this.streamListener = options.onStream;
    this.cellInFlight = true;
    const generation = this.workerGeneration;
    try {
      return await this.call(WORKER_OPS.runCell, [
        source,
        { streaming: Boolean(options.onStream), resetProducedFiles: options.resetProducedFiles, resultOnly: options.resultOnly },
      ]) as RunResult;
    } finally {
      if (this.workerGeneration === generation) {
        this.cellInFlight = false;
        this.streamListener = undefined;
        this.interruptBuffer[0] = 0;
        this.onIdle?.();
      }
    }
  }

  async completeNames(path: readonly string[], partial: string): Promise<readonly CompletionCandidate[]> {
    if (!this.worker || this.cellInFlight) return [];
    try {
      return await this.call(WORKER_OPS.complete, [path, partial]) as CompletionCandidate[];
    } catch {
      return [];
    }
  }

  async inspectNamespace(): Promise<readonly NamespaceEntry[]> {
    if (!this.worker || this.cellInFlight) return [];
    return await this.call(WORKER_OPS.inspectNamespace) as NamespaceEntry[];
  }

  async interpreterHeapBytes(): Promise<number | null> {
    if (!this.worker) return null;
    return await this.call(WORKER_OPS.heapBytes) as number | null;
  }

  async readProducedFile(handle: PyodideHandle, file: ProducedFile): Promise<Uint8Array> {
    this.assertHandle(handle);
    return await this.call(WORKER_OPS.readProducedFile, [file]) as Uint8Array;
  }

  async writeStagedFile(directory: string, virtualPath: string, bytes: Uint8Array): Promise<void> {
    const transferable = bytes.buffer instanceof ArrayBuffer ? [bytes.buffer] : [];
    await this.call(WORKER_OPS.writeStagedFile, [directory, virtualPath, bytes], transferable);
  }

  async removeStagedFile(virtualPath: string): Promise<void> {
    await this.call(WORKER_OPS.removeStagedFile, [virtualPath]);
  }

  async bindStagedFilesInPython(
    staged: ReadonlyArray<{ name: string; virtualPath: string }>,
  ): Promise<void> {
    await this.call(WORKER_OPS.bindStagedFiles, [staged.map((file) => ({ ...file }))]);
  }
}

const DEFAULT_RUNTIME = new WorkerRuntimeClient("default");
const scopedClients = new Map<string, WorkerRuntimeClient>();
const MAX_SCOPED_CLIENTS = 12;

/** Evict least-recently-used idle clients until the retained set is bounded.
 * Busy clients are never terminated, but their idle transition retries the
 * eviction so a temporary all-busy overflow cannot become permanent. */
function evictIdleScopedClients(preserveKey?: string): void {
  while (scopedClients.size > MAX_SCOPED_CLIENTS) {
    let evicted = false;
    for (const [candidateKey, candidate] of scopedClients) {
      if (candidateKey === preserveKey || candidate.isBusy()) continue;
      scopedClients.delete(candidateKey);
      candidate.dispose();
      evicted = true;
      break;
    }
    if (!evicted) return;
  }
}

/** Dedicated compatibility/runtime-internal client; never participates in scoped LRU. */
export function getDefaultPythonRuntimeClient(): PythonRuntimeClient {
  return DEFAULT_RUNTIME;
}

/** Stable interpreter client for one project/pane run key. */
export function getPythonRuntimeClient(key: string): PythonRuntimeClient {
  const existing = scopedClients.get(key);
  if (existing) {
    scopedClients.delete(key);
    scopedClients.set(key, existing);
    return existing;
  }
  const client = new WorkerRuntimeClient(key, evictIdleScopedClients);
  scopedClients.set(key, client);
  evictIdleScopedClients(key);
  return client;
}

export function disposePythonRuntimeClient(key: string): void {
  const client = scopedClients.get(key);
  if (!client) return;
  scopedClients.delete(key);
  client.dispose();
}

/** Backward-compatible default-runtime wrappers. */
export const canInterrupt = () => DEFAULT_RUNTIME.canInterrupt();
export const pythonNeedsRestart = () => DEFAULT_RUNTIME.pythonNeedsRestart();
export const restartPython = () => DEFAULT_RUNTIME.restartPython();
export const interruptPython = () => DEFAULT_RUNTIME.interruptPython();
export const isCellInFlight = () => DEFAULT_RUNTIME.isCellInFlight();
export const getPyodide = (onStatus?: PyodideStatusListener) => DEFAULT_RUNTIME.getPyodide(onStatus);
export const loadPyodidePackage = (name: string, status?: PyodideStatusListener) =>
  DEFAULT_RUNTIME.loadPyodidePackage(name, status);
export const micropipInstall = (name: string, status?: PyodideStatusListener) =>
  DEFAULT_RUNTIME.micropipInstall(name, status);
export const installPackage = (name: string, status?: PyodideStatusListener) =>
  DEFAULT_RUNTIME.installPackage(name, status);
export const loadedPackageNames = () => DEFAULT_RUNTIME.loadedPackageNames();
export const bindContext = (handle: PyodideHandle, ctx: InjectionContext) =>
  DEFAULT_RUNTIME.bindContext(handle, ctx);
export const runPythonCell = (handle: PyodideHandle, source: string, options?: RunOptions) =>
  DEFAULT_RUNTIME.runPythonCell(handle, source, options);
export const completeNames = (path: readonly string[], partial: string) =>
  DEFAULT_RUNTIME.completeNames(path, partial);
export const inspectNamespace = () => DEFAULT_RUNTIME.inspectNamespace();
export const interpreterHeapBytes = () => DEFAULT_RUNTIME.interpreterHeapBytes();
export const readProducedFile = (handle: PyodideHandle, file: ProducedFile) =>
  DEFAULT_RUNTIME.readProducedFile(handle, file);
export const writeStagedFile = (directory: string, path: string, bytes: Uint8Array) =>
  DEFAULT_RUNTIME.writeStagedFile(directory, path, bytes);
export const removeStagedFile = (path: string) => DEFAULT_RUNTIME.removeStagedFile(path);
export const bindStagedFilesInPython = (
  staged: ReadonlyArray<{ name: string; virtualPath: string }>,
) => DEFAULT_RUNTIME.bindStagedFilesInPython(staged);

export type { CompletionCandidate, NpyN };
