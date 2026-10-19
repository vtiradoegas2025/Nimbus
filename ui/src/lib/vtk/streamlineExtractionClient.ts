import type { PreparedVectorImage } from "./vtkPreparation";
import type { StreamlineExtractionParameters, StreamlineExtractionResult, StreamlineWorkerReply, StreamlineWorkerRequest } from "./streamlineExtractionProtocol";
import { handoffVtkExtractionDataset, isSharedVtkDataset, recordWarmVtkDatasetReuse, recordVtkDatasetHandoff } from "./vtkDatasetHandoff";
import streamlineExtractionWorkerUrl from "./streamlineExtraction.worker.ts?worker&url";
import { recordVisualizationTransientReservation } from "@/lib/core/visualizationResourceGovernor";
import { VtkWorkerDatasetOwnership } from "./vtkWorkerDatasetOwnership";
import { vtkExtractionServicePool } from "./vtkExtractionServicePool";

export interface StreamlineExtractionWorker {
  onmessage: ((event: MessageEvent<StreamlineWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: StreamlineWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}
export type StreamlineExtractionWorkerFactory = () => StreamlineExtractionWorker;

interface Job {
  identity: string;
  prepared: PreparedVectorImage;
  parameters: StreamlineExtractionParameters;
  datasetId: number;
  requestId: number;
  attempts: number;
  consumers: number;
  settled: boolean;
  promise: Promise<StreamlineExtractionResult>;
  resolve: (result: StreamlineExtractionResult) => void;
  reject: (error: unknown) => void;
}

let workerGeneration = 0;
let bufferSequence = 0;
const bufferIds = new WeakMap<object, number>();
function bufferId(buffer: ArrayBufferLike): number {
  let id = bufferIds.get(buffer);
  if (id == null) bufferIds.set(buffer, id = ++bufferSequence);
  return id;
}
function defaultWorkerFactory(): StreamlineExtractionWorker {
  const separator = streamlineExtractionWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${streamlineExtractionWorkerUrl}${separator}generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-streamline-extraction",
  }) as unknown as StreamlineExtractionWorker;
}
function abortError(): DOMException {
  return new DOMException("Streamline extraction cancelled", "AbortError");
}
function requestIdentity(prepared: PreparedVectorImage, parameters: StreamlineExtractionParameters): string {
  return JSON.stringify([
    bufferId(prepared.values.buffer), prepared.values.byteOffset, prepared.values.byteLength,
    prepared.shape, prepared.spacing, prepared.origin,
    parameters.integrationStep, parameters.maxSteps, parameters.tubeRadius,
    parameters.scale.xy, parameters.scale.z, [...parameters.seeds],
  ]);
}

export class SharedStreamlineExtractionService {
  private readonly workerFactory: StreamlineExtractionWorkerFactory;
  private readonly ownership = new VtkWorkerDatasetOwnership("vtk:streamline-service:dataset");
  private readonly shared = new Map<string, Job>();
  private readonly datasetIds = new WeakMap<object, number>();
  private readonly queue: Job[] = [];
  private worker: StreamlineExtractionWorker | null = null;
  private ready = false;
  private active: Job | null = null;
  private loadedDatasetId = -1;
  private datasetSequence = 0;
  private requestSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(workerFactory: StreamlineExtractionWorkerFactory = defaultWorkerFactory) {
    this.workerFactory = workerFactory;
  }

  enqueue(prepared: PreparedVectorImage, parameters: StreamlineExtractionParameters, signal?: AbortSignal): Promise<StreamlineExtractionResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    const identity = requestIdentity(prepared, parameters);
    let job = this.shared.get(identity);
    if (!job) {
      let resolve!: (result: StreamlineExtractionResult) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<StreamlineExtractionResult>((yes, no) => { resolve = yes; reject = no; });
      let datasetId = this.datasetIds.get(prepared);
      if (datasetId == null) this.datasetIds.set(prepared, datasetId = ++this.datasetSequence);
      job = { identity, prepared, parameters, datasetId, requestId: 0, attempts: 0, consumers: 0, settled: false, promise, resolve, reject };
      this.shared.set(identity, job);
      this.queue.push(job);
    }
    job.consumers += 1;
    this.ensureWorker();
    this.startNext();
    const sharedJob = job;
    return new Promise((resolve, reject) => {
      let released = false;
      const release = () => {
        if (released) return;
        released = true;
        signal?.removeEventListener("abort", onAbort);
        sharedJob.consumers = Math.max(0, sharedJob.consumers - 1);
        if (!sharedJob.settled && sharedJob.consumers === 0) this.cancel(sharedJob);
      };
      const onAbort = () => { release(); reject(abortError()); };
      signal?.addEventListener("abort", onAbort, { once: true });
      sharedJob.promise.then(
        (result) => { if (!released) { release(); resolve(result); } },
        (error) => { if (!released) { release(); reject(error); } },
      );
    });
  }

  shutdown(): void {
    if (this.active) this.finish(this.active, abortError());
    for (const job of this.queue.splice(0)) this.finish(job, abortError());
    this.disposeWorker();
  }

  private cancel(job: Job): void {
    const index = this.queue.indexOf(job);
    if (index >= 0) { this.queue.splice(index, 1); this.finish(job, abortError()); return; }
    if (this.active === job) {
      this.active = null;
      this.finish(job, abortError());
      this.disposeWorker();
      this.restartForQueue();
    }
  }
  private finish(job: Job, outcome: StreamlineExtractionResult | Error | DOMException): void {
    if (job.settled) return;
    job.settled = true;
    if (this.shared.get(job.identity) === job) this.shared.delete(job.identity);
    if ("forward" in outcome) job.resolve(outcome);
    else job.reject(outcome);
  }
  private ensureWorker(): void {
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    if (this.worker) return;
    const worker = this.workerFactory();
    this.worker = worker;
    this.ready = false;
    worker.onmessage = (event) => this.handleMessage(worker, event.data);
    worker.onerror = (event) => { event.preventDefault?.(); this.workerFailed(worker, event.message || "Streamline extraction worker failed"); };
    worker.onmessageerror = (event) => { event.preventDefault?.(); this.workerFailed(worker, "Streamline extraction worker returned an unreadable result"); };
    this.startupTimer = setTimeout(() => {
      if (this.worker === worker && !this.ready) this.workerFailed(worker, "Streamline extraction worker did not start");
    }, 3_000);
  }
  private handleMessage(worker: StreamlineExtractionWorker, reply: StreamlineWorkerReply): void {
    if (this.worker !== worker) return;
    if (reply.kind === "ready") {
      if (this.startupTimer != null) clearTimeout(this.startupTimer);
      this.startupTimer = null;
      this.ready = true;
      this.startNext();
      return;
    }
    if (reply.kind === "prepared") return;
    const job = this.active;
    if (!job || reply.requestId !== job.requestId) return;
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    this.executionTimer = null;
    this.active = null;
    this.ownership.commit(reply.requestId);
    if (reply.ok) {
      this.loadedDatasetId = reply.datasetId;
      recordVisualizationTransientReservation("vtk:streamline-service:scratch", `request:${reply.requestId}`, "worker-scratch", reply.result.knownScratchBytes);
      this.finish(job, reply.result);
    } else this.finish(job, new Error(reply.error));
    this.startNext();
  }
  private startNext(): void {
    if (!this.worker || !this.ready || this.active) return;
    let job: Job | undefined;
    while ((job = this.queue.shift())) {
      if (!job.settled && job.consumers > 0) break;
      job = undefined;
    }
    if (!job) { this.scheduleRetirement(); return; }
    this.active = job;
    job.attempts += 1;
    job.requestId = ++this.requestSequence;
    const needsHandoff = this.loadedDatasetId !== job.datasetId;
    let handoff = null as ReturnType<typeof handoffVtkExtractionDataset> | null;
    if (needsHandoff) {
      try {
        this.ownership.reserveCopy(
          job.requestId,
          isSharedVtkDataset(job.prepared.values) ? 0 : job.prepared.values.byteLength,
        );
        handoff = handoffVtkExtractionDataset(job.prepared.values, "streamline-vector-grid");
        this.ownership.materialize(job.requestId, [{
          buffer: handoff.values.buffer,
          sharedWithMain: handoff.transfer.length === 0,
        }]);
      } catch (error) {
        this.ownership.discard(job.requestId);
        this.active = null;
        this.finish(job, error instanceof Error ? error : new Error(String(error)));
        this.startNext();
        return;
      }
    }
    const seeds = job.parameters.seeds.slice();
    const message: StreamlineWorkerRequest = {
      kind: "extract", requestId: job.requestId, datasetId: job.datasetId,
      ...job.parameters, seeds,
      ...(handoff ? { image: { shape: job.prepared.shape, spacing: job.prepared.spacing, origin: job.prepared.origin, values: handoff.values } } : {}),
    };
    const transfer: Transferable[] = [seeds.buffer];
    if (handoff) {
      transfer.push(...handoff.transfer);
    }
    try {
      this.worker.postMessage(message, transfer);
      if (handoff) recordVtkDatasetHandoff(handoff.report); else recordWarmVtkDatasetReuse("streamline-vector-grid");
    } catch (error) {
      this.ownership.discard(job.requestId);
      this.workerFailed(this.worker, error instanceof Error ? error.message : String(error));
      return;
    }
    const deadline = Math.min(180_000, Math.max(30_000, 15_000 + job.prepared.values.byteLength / (2 * 1024 * 1024) * 1_000));
    this.executionTimer = setTimeout(() => {
      if (this.active === job && this.worker) this.workerFailed(this.worker, `Streamline extraction exceeded its ${Math.ceil(deadline / 1_000)} s memory-safe deadline`);
    }, deadline);
  }
  private workerFailed(worker: StreamlineExtractionWorker, detail: string): void {
    if (this.worker !== worker) return;
    const active = this.active;
    this.active = null;
    this.disposeWorker();
    if (active && !active.settled && active.consumers > 0 && active.attempts < 2) this.queue.unshift(active);
    else if (active && !active.settled) this.finish(active, new Error(detail));
    this.restartForQueue();
  }
  private restartForQueue(): void {
    if (this.queue.some((job) => !job.settled && job.consumers > 0)) {
      setTimeout(() => {
        if (!this.queue.some((job) => !job.settled && job.consumers > 0)) return;
        this.ensureWorker();
        this.startNext();
      }, 25);
    }
  }
  private scheduleRetirement(): void {
    if (!this.worker || this.active || this.queue.length || this.retireTimer != null) return;
    vtkExtractionServicePool.park(this, () => {
      if (!this.active && !this.queue.length) this.disposeWorker();
    });
    this.retireTimer = setTimeout(() => { this.retireTimer = null; if (!this.active && !this.queue.length) this.disposeWorker(); }, 15_000);
  }
  private disposeWorker(): void {
    vtkExtractionServicePool.remove(this);
    const worker = this.worker;
    this.worker = null;
    this.ready = false;
    this.loadedDatasetId = -1;
    for (const timer of [this.startupTimer, this.executionTimer, this.retireTimer]) if (timer != null) clearTimeout(timer);
    this.startupTimer = this.executionTimer = this.retireTimer = null;
    this.ownership.reset();
    if (!worker) return;
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    worker.terminate();
  }
}

let defaultService: SharedStreamlineExtractionService | null = null;
export function extractStreamlinesInSharedWorker(prepared: PreparedVectorImage, parameters: StreamlineExtractionParameters, signal?: AbortSignal): Promise<StreamlineExtractionResult> {
  defaultService ??= new SharedStreamlineExtractionService();
  return defaultService.enqueue(prepared, parameters, signal);
}
