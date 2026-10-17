import { visualizationStagingQueue } from "./visualizationStagingQueue";
import { nativeWindowReservationBytes } from "./nativeWindowBudget";
import {
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "./visualizationResourceGovernor";

/** Error used when latest-wins scheduling prevents obsolete work from starting. */
export class SupersededLoadError extends Error {
  constructor() {
    super("field load superseded by a newer request");
    this.name = "SupersededLoadError";
  }
}

export interface LoadScheduleOptions {
  /** Stable owner for latest-wins requests. Omit for batch/history work. */
  latestScope?: string;
  /** Conservative reservation for decoded/uncompressed bytes. */
  estimatedBytes: number;
  /** Interactive work always owns at least one concurrency slot. */
  priority?: "interactive" | "background";
  signal?: AbortSignal;
}

export interface BoundedFieldWindowReservation {
  extent: readonly number[];
  stride: readonly number[];
  maxOutputBytes?: number;
  maxSourceBytes?: number;
}

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const TERRAIN_OUTPUT_BYTES_PER_CELL = FLOAT32_BYTES + 4 * Float64Array.BYTES_PER_ELEMENT;

function safeProduct(values: readonly number[]): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) return Number.MAX_SAFE_INTEGER;
    product *= value;
    if (!Number.isSafeInteger(product)) return Number.MAX_SAFE_INTEGER;
  }
  return product;
}

/**
 * Conservative memory reservation for one bounded native field transaction.
 *
 * A window must not inherit the full source field's reservation: large-domain
 * plots deliberately tile/reduce before IPC, and charging every tile as the
 * complete field serializes the u/v/w reads. Uniform aggregation holds several
 * output-sized native work arrays plus response/cache copies, while terrain's
 * explicit source budget already includes geometry and aggregation work.
 */
export function estimateBoundedFieldWindowBytes(
  window: BoundedFieldWindowReservation,
  terrainFollowing: boolean,
): number {
  const sourceCells = safeProduct(window.extent);
  const outputShape = window.extent.map((extent, axis) => {
    const stride = window.stride[axis];
    if (!Number.isSafeInteger(stride) || stride <= 0) return Number.MAX_SAFE_INTEGER;
    return Math.floor((extent - 1) / stride) + 1;
  });
  const outputCells = safeProduct(outputShape);
  const sourceBytes = Math.min(
    Number.MAX_SAFE_INTEGER,
    sourceCells * FLOAT32_BYTES,
  );
  const outputBytesPerCell = terrainFollowing
    ? TERRAIN_OUTPUT_BYTES_PER_CELL
    : FLOAT32_BYTES;
  const outputBytes = Math.min(
    Number.MAX_SAFE_INTEGER,
    outputCells * outputBytesPerCell,
  );
  const admittedSourceBytes = window.maxSourceBytes ?? sourceBytes;
  // Uniform NPY aggregation uses f64 sums, usize counts, invalid flags, the
  // float body, encoded response, IPC buffer, and cached JS view. Terrain's
  // source budget already accounts for its native aggregation work arrays.
  const outputCopies = terrainFollowing ? 3 : 7;
  const estimate = Math.max(admittedSourceBytes + outputBytes * outputCopies, nativeWindowReservationBytes(window));
  return Number.isFinite(estimate)
    ? Math.max(1, Math.min(Number.MAX_SAFE_INTEGER, Math.ceil(estimate)))
    : Number.MAX_SAFE_INTEGER;
}

export interface LoadToken {
  latestScope: string | null;
  generation: number;
}

interface QueuedLoad<T> {
  token: LoadToken;
  bytes: number;
  priority: "interactive" | "background";
  task: (token: LoadToken) => Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
  signal?: AbortSignal;
  detachAbort: () => void;
}

export interface LoadSchedulerStats {
  readonly active: number;
  readonly activeBytes: number;
  readonly activeBackground: number;
  readonly queued: number;
  readonly queuedBytes: number;
  readonly maxConcurrent: number;
  readonly maxBytes: number;
  readonly completed: number;
  readonly cancelled: number;
  readonly superseded: number;
  readonly failures: number;
  readonly latestScopes: number;
}

export type LoadSchedulerListener = () => void;

/** Byte/concurrency scheduler. Latest-scoped requests supersede older queued work for
 *  the same lane; in-flight I/O completes but its result is rejected as superseded. */
export class LoadScheduler {
  private readonly waitingForStaging = new Map<LoadToken, () => void>();
  private readonly queue: Array<QueuedLoad<unknown>> = [];
  private readonly latest = new Map<string, number>();
  private readonly activeLatestScopes = new Map<string, number>();
  private nextGeneration = 0;
  private active = 0;
  private activeBytes = 0;
  private activeBackground = 0;
  private maxConcurrent: number;
  private maxBytes: number;
  private readonly visualizationOwner: string | null;
  private completed = 0;
  private cancelled = 0;
  private supersededCount = 0;
  private failures = 0;
  private readonly listeners = new Set<LoadSchedulerListener>();
  private snapshot: LoadSchedulerStats;

  constructor(maxConcurrent: number, maxBytes: number, visualizationOwner: string | null = null) {
    this.maxConcurrent = Number.isFinite(maxConcurrent)
      ? Math.max(1, Math.floor(maxConcurrent))
      : 1;
    this.maxBytes = Number.isFinite(maxBytes)
      ? Math.max(1, Math.floor(maxBytes))
      : 1;
    this.visualizationOwner = visualizationOwner?.trim() || null;
    this.snapshot = this.createSnapshot();
  }

  schedule<T>(
    options: LoadScheduleOptions,
    task: (token: LoadToken) => Promise<T>,
  ): Promise<T> {
    if (options.signal?.aborted) {
      return Promise.reject(new DOMException("field load aborted", "AbortError"));
    }
    const generation = ++this.nextGeneration;
    const latestScope = options.latestScope ?? null;
    const token = { latestScope, generation };
    const bytes = Number.isFinite(options.estimatedBytes)
      ? Math.max(1, Math.floor(options.estimatedBytes))
      : 1;

    if (latestScope) this.advanceLatestScope(latestScope, generation);

    const promise = new Promise<T>((resolve, reject) => {
      const load: QueuedLoad<T> = {
        token,
        bytes,
        priority: options.priority ?? "interactive",
        task,
        resolve,
        reject,
        signal: options.signal,
        detachAbort: () => undefined,
      };
      if (options.signal) {
        const abort = () => {
          const index = this.queue.indexOf(load as QueuedLoad<unknown>);
          if (index >= 0) {
            this.queue.splice(index, 1);
            load.detachAbort();
            this.cancelled += 1;
            reject(new DOMException("field load aborted", "AbortError"));
            if (latestScope) this.retireLatestScope(latestScope);
            this.drain();
            this.publish();
          } else {
            this.waitingForStaging.get(token)?.();
          }
        };
        options.signal.addEventListener("abort", abort, { once: true });
        load.detachAbort = () => options.signal!.removeEventListener("abort", abort);
      }
      // Latest-scoped work is front-queued; unscoped history loads go to the back.
      if (latestScope) this.queue.unshift(load as QueuedLoad<unknown>);
      else this.queue.push(load as QueuedLoad<unknown>);
    });
    this.drain();
    this.publish();
    return promise;
  }

  cancelQueued(): void {
    for (const cancel of this.waitingForStaging.values()) cancel();
    const queued = this.queue.splice(0);
    const latestScopes = new Set<string>();
    for (const load of queued) {
      load.detachAbort();
      this.cancelled += 1;
      if (load.token.latestScope) latestScopes.add(load.token.latestScope);
      load.reject(new SupersededLoadError());
    }
    for (const latestScope of latestScopes) this.retireLatestScope(latestScope);
    this.publish();
  }

  setLimits(maxConcurrent: number, maxBytes: number): void {
    if (!Number.isSafeInteger(maxConcurrent) || maxConcurrent < 1
        || !Number.isSafeInteger(maxBytes) || maxBytes < 1) {
      throw new Error("Field loader limits must be positive safe integers");
    }
    this.maxConcurrent = maxConcurrent;
    this.maxBytes = maxBytes;
    this.drain();
    this.publish();
  }

  /** Mark a cached request as newest without reserving or starting work. */
  supersede(latestScope: string): void {
    const generation = ++this.nextGeneration;
    this.advanceLatestScope(latestScope, generation);
    this.drain();
    this.retireLatestScope(latestScope);
    this.publish();
  }

  stats(): LoadSchedulerStats {
    return this.snapshot;
  }

  /** Subscribe to immutable state-transition snapshots; no polling required. */
  subscribe(listener: LoadSchedulerListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private createSnapshot(): LoadSchedulerStats {
    return Object.freeze({
      active: this.active,
      activeBytes: this.activeBytes,
      activeBackground: this.activeBackground,
      queued: this.queue.length,
      queuedBytes: this.queue.reduce((sum, load) => sum + load.bytes, 0),
      maxConcurrent: this.maxConcurrent,
      maxBytes: this.maxBytes,
      completed: this.completed,
      cancelled: this.cancelled,
      superseded: this.supersededCount,
      failures: this.failures,
      latestScopes: this.latest.size,
    });
  }

  private publish(): void {
    this.snapshot = this.createSnapshot();
    for (const listener of this.listeners) listener();
  }

  private isObsolete(token: LoadToken): boolean {
    return token.latestScope != null &&
      this.latest.get(token.latestScope) !== token.generation;
  }

  private advanceLatestScope(latestScope: string, generation: number): void {
    this.latest.set(latestScope, generation);
    for (const [token, cancel] of this.waitingForStaging) {
      if (token.latestScope === latestScope) cancel();
    }
    for (let index = this.queue.length - 1; index >= 0; --index) {
      const queued = this.queue[index];
      if (queued.token.latestScope === latestScope) {
        this.queue.splice(index, 1);
        this.supersededCount += 1;
        queued.detachAbort();
        queued.reject(new SupersededLoadError());
      }
    }
  }

  private retireLatestScope(latestScope: string): void {
    if ((this.activeLatestScopes.get(latestScope) ?? 0) > 0) return;
    if (this.queue.some((load) => load.token.latestScope === latestScope)) return;
    this.latest.delete(latestScope);
  }

  private drain(): void {
    const concurrentLimit = Math.max(1, Math.floor(this.maxConcurrent));
    const backgroundLimit = concurrentLimit === 1 ? 1 : concurrentLimit - 1;
    while (this.active < concurrentLimit && this.queue.length > 0) {
      const fits = (load: QueuedLoad<unknown>) =>
        this.activeBytes + load.bytes <= this.maxBytes;
      const hasInteractive = this.queue.some((load) => load.priority === "interactive");
      let nextIndex = this.queue.findIndex((load) =>
        load.priority === "interactive" && fits(load),
      );
      if (nextIndex < 0 && !hasInteractive && this.activeBackground < backgroundLimit) {
        nextIndex = this.queue.findIndex((load) =>
          load.priority === "background" && fits(load),
        );
      }
      // A single oversized field must still be able to make progress, but it
      // runs alone rather than defeating the byte cap alongside other loads.
      if (nextIndex < 0 && this.active === 0) {
        nextIndex = this.queue.findIndex((load) => load.priority === "interactive");
        if (nextIndex < 0 && this.activeBackground < backgroundLimit) nextIndex = 0;
      }
      if (nextIndex < 0) return;

      const load = this.queue.splice(nextIndex, 1)[0];
      if (this.isObsolete(load.token)) {
        this.supersededCount += 1;
        load.detachAbort();
        load.reject(new SupersededLoadError());
        continue;
      }
      this.active += 1;
      this.activeBytes += load.bytes;
      if (load.priority === "background") this.activeBackground += 1;
      if (load.token.latestScope) {
        this.activeLatestScopes.set(
          load.token.latestScope,
          (this.activeLatestScopes.get(load.token.latestScope) ?? 0) + 1,
        );
      }
      let visualizationReservationId: number | null = null;
      const release = () => {
        load.detachAbort();
        if (visualizationReservationId != null) {
          visualizationResourceGovernor.release(visualizationReservationId);
          visualizationReservationId = null;
        }
        this.active -= 1;
        this.activeBytes -= load.bytes;
        if (load.priority === "background") this.activeBackground -= 1;
        if (load.token.latestScope) {
          const remaining = (this.activeLatestScopes.get(load.token.latestScope) ?? 1) - 1;
          if (remaining <= 0) this.activeLatestScopes.delete(load.token.latestScope);
          else this.activeLatestScopes.set(load.token.latestScope, remaining);
        }
      };
      const task = () => load.signal?.aborted
        ? Promise.reject(new DOMException("field load aborted", "AbortError"))
        : this.isObsolete(load.token)
          ? Promise.reject(new SupersededLoadError()) : load.task(load.token);
      const result = this.visualizationOwner
        ? new Promise<void>((resolve, reject) => {
            let granted = false;
            const cancel = visualizationStagingQueue.schedule({
              owner: this.visualizationOwner!,
              generation: `load:${load.token.generation}`,
              priority: load.priority,
              claims: [visualizationReservationClaim(
                `native-read:${this.visualizationOwner}:${load.token.generation}`,
                "native-read", load.bytes,
              )],
            }, (resources) => {
              granted = true;
              this.waitingForStaging.delete(load.token);
              if (!resources.ok) { reject(new Error(resources.reason)); return; }
              visualizationReservationId = resources.reservation.id;
              resolve();
            });
            if (!granted) this.waitingForStaging.set(load.token, () => {
              this.waitingForStaging.delete(load.token);
              cancel();
              reject(load.signal?.aborted
                ? new DOMException("field load aborted", "AbortError")
                : new SupersededLoadError());
            });
          }).then(task)
        : task();
      void result.then(
        (value) => {
          release();
          if (load.signal?.aborted) {
            this.cancelled += 1;
            load.reject(new DOMException("field load aborted", "AbortError"));
          } else if (this.isObsolete(load.token)) {
            this.supersededCount += 1;
            load.reject(new SupersededLoadError());
          } else {
            this.completed += 1;
            load.resolve(value);
          }
          if (load.token.latestScope) this.retireLatestScope(load.token.latestScope);
          this.drain();
          this.publish();
        },
        (error) => {
          release();
          if (load.signal?.aborted || error instanceof DOMException && error.name === "AbortError") this.cancelled += 1;
          else if (error instanceof SupersededLoadError) this.supersededCount += 1;
          else this.failures += 1;
          load.reject(load.signal?.aborted ? new DOMException("field load aborted", "AbortError") : error);
          if (load.token.latestScope) this.retireLatestScope(load.token.latestScope);
          this.drain();
          this.publish();
        },
      );
    }
  }
}

const MIB = 1024 * 1024;
export const fieldLoadScheduler = new LoadScheduler(3, 384 * MIB, "field-loader");

export function fieldLoadSchedulerStats(): LoadSchedulerStats {
  return fieldLoadScheduler.stats();
}

export function subscribeFieldLoadScheduler(
  listener: LoadSchedulerListener,
): () => void {
  return fieldLoadScheduler.subscribe(listener);
}
