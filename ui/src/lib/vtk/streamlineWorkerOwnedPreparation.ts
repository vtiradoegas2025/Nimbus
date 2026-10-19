import type { PreparedVectorImage, VectorPreparationRequest } from "./vtkPreparation";
import {
  type PreparedVectorMetadata,
  type StreamlineExtractionParameters,
  type StreamlineExtractionResult,
  type StreamlineWorkerReply,
  type StreamlineWorkerRequest,
} from "./streamlineExtractionProtocol";
import streamlineExtractionWorkerUrl from "./streamlineExtraction.worker.ts?worker&url";
import { vtkPreparationIdentity } from "./vtkPreparationClient";
import {
  retainRemotePreparedDataset,
  type VtkPreparedDatasetHandle,
  type VtkPreparedDatasetLease,
} from "./vtkPreparedDatasetRegistry";
import {
  recordVisualizationTransientReservation,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import {
  vtkPreparationInputBytes,
  vtkPreparationOutputUpperBoundBytes,
} from "./vtkPreparationResourcePlanning";
import { handoffVtkPreparationInput } from "./vtkPreparationInputHandoff";
import { vtkExtractionServicePool } from "./vtkExtractionServicePool";

export interface WorkerOwnedPreparationWorker {
  onmessage: ((event: MessageEvent<StreamlineWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: StreamlineWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

export type WorkerOwnedPreparationWorkerFactory = () => WorkerOwnedPreparationWorker;

interface Backing {
  identity: string;
  request: VectorPreparationRequest;
  datasetId: number;
  metadata: PreparedVectorMetadata;
  bytes: number;
  reservationId: number;
}

interface ExtractionJob {
  identity: string;
  backing: Backing;
  parameters: StreamlineExtractionParameters;
  requestId: number;
  attempts: number;
  consumers: number;
  settled: boolean;
  rebuildReservationId: number | null;
  promise: Promise<StreamlineExtractionResult>;
  resolve: (result: StreamlineExtractionResult) => void;
  reject: (error: unknown) => void;
}

interface Job {
  identity: string;
  request: VectorPreparationRequest;
  datasetId: number;
  requestId: number;
  attempts: number;
  consumers: number;
  settled: boolean;
  reservationId: number | null;
  promise: Promise<Backing>;
  resolve: (backing: Backing) => void;
  reject: (error: unknown) => void;
}

let workerGeneration = 0;

function defaultWorkerFactory(): WorkerOwnedPreparationWorker {
  const separator = streamlineExtractionWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${streamlineExtractionWorkerUrl}${separator}owned=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-worker-owned-vector-preparation",
  }) as unknown as WorkerOwnedPreparationWorker;
}

function abortError(): DOMException {
  return new DOMException("Worker-owned vector preparation cancelled", "AbortError");
}

export class StreamlineWorkerOwnedPreparationService {
  private readonly workerFactory: WorkerOwnedPreparationWorkerFactory;
  private readonly shared = new Map<string, Job>();
  private readonly backings = new Map<string, Backing>();
  private readonly queue: Job[] = [];
  private readonly handleBackings = new Map<number, Backing>();
  private readonly sharedExtractions = new Map<string, ExtractionJob>();
  private readonly extractionQueue: ExtractionJob[] = [];
  private worker: WorkerOwnedPreparationWorker | null = null;
  private ready = false;
  private active: Job | null = null;
  private activeExtraction: ExtractionJob | null = null;
  private loaded: Backing | null = null;
  private datasetSequence = 0;
  private requestSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(workerFactory: WorkerOwnedPreparationWorkerFactory = defaultWorkerFactory) {
    this.workerFactory = workerFactory;
  }

  prepare(
    request: VectorPreparationRequest,
    signal?: AbortSignal,
  ): Promise<VtkPreparedDatasetLease<PreparedVectorImage>> {
    if (signal?.aborted) return Promise.reject(abortError());
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    const identity = vtkPreparationIdentity(request);
    const resident = this.backings.get(identity);
    if (resident) return Promise.resolve(this.lease(resident));
    let job = this.shared.get(identity);
    if (!job) {
      let resolve!: (backing: Backing) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<Backing>((yes, no) => { resolve = yes; reject = no; });
      job = {
        identity,
        request,
        datasetId: ++this.datasetSequence,
        requestId: 0,
        attempts: 0,
        consumers: 0,
        settled: false,
        reservationId: null,
        promise,
        resolve,
        reject,
      };
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
        (backing) => { if (!released) { release(); resolve(this.lease(backing)); } },
        (error) => { if (!released) { release(); reject(error); } },
      );
    });
  }

  extract(
    handle: VtkPreparedDatasetHandle<PreparedVectorImage>,
    parameters: StreamlineExtractionParameters,
    signal?: AbortSignal,
  ): Promise<StreamlineExtractionResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    const backing = this.handleBackings.get(handle.id);
    if (!backing || handle.residency !== "worker") {
      return Promise.reject(new Error(`Worker-owned vector handle ${handle.id} is unavailable`));
    }
    const identity = JSON.stringify([
      handle.id,
      parameters.integrationStep,
      parameters.maxSteps,
      parameters.tubeRadius,
      parameters.scale.xy,
      parameters.scale.z,
      [...parameters.seeds],
    ]);
    let job = this.sharedExtractions.get(identity);
    if (!job) {
      let resolve!: (result: StreamlineExtractionResult) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<StreamlineExtractionResult>((yes, no) => {
        resolve = yes;
        reject = no;
      });
      job = {
        identity,
        backing,
        parameters,
        requestId: 0,
        attempts: 0,
        consumers: 0,
        settled: false,
        rebuildReservationId: null,
        promise,
        resolve,
        reject,
      };
      this.sharedExtractions.set(identity, job);
      this.extractionQueue.push(job);
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
        if (!sharedJob.settled && sharedJob.consumers === 0) this.cancelExtraction(sharedJob);
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
    if (this.active) this.fail(this.active, abortError());
    for (const job of this.queue.splice(0)) this.fail(job, abortError());
    if (this.activeExtraction) this.failExtraction(this.activeExtraction, abortError());
    for (const job of this.extractionQueue.splice(0)) this.failExtraction(job, abortError());
    this.disposeWorker();
  }

  private lease(backing: Backing): VtkPreparedDatasetLease<PreparedVectorImage> {
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    let handleId = 0;
    const lease = retainRemotePreparedDataset<PreparedVectorImage>(backing, {
      kind: "vector",
      bytes: backing.bytes,
      metadata: backing.metadata,
      onLastRelease: () => {
        this.handleBackings.delete(handleId);
        if (!this.retireIfUnused()) this.scheduleRetirement();
      },
    });
    handleId = lease.handle.id;
    this.handleBackings.set(handleId, backing);
    return lease;
  }

  private cancel(job: Job): void {
    const queued = this.queue.indexOf(job);
    if (queued >= 0) {
      this.queue.splice(queued, 1);
      this.fail(job, abortError());
      this.retireIfUnused();
    } else if (this.active === job) {
      this.active = null;
      this.fail(job, abortError());
      this.disposeWorker();
      this.restartForQueue();
    }
  }

  private cancelExtraction(job: ExtractionJob): void {
    const queued = this.extractionQueue.indexOf(job);
    if (queued >= 0) {
      this.extractionQueue.splice(queued, 1);
      this.failExtraction(job, abortError());
      this.retireIfUnused();
    } else if (this.activeExtraction === job) {
      this.activeExtraction = null;
      this.failExtraction(job, abortError());
      this.disposeWorker();
      this.restartForQueue();
    }
  }

  private ensureWorker(): void {
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    if (this.worker) return;
    const worker = this.workerFactory();
    this.worker = worker;
    this.ready = false;
    worker.onmessage = (event) => this.message(worker, event.data);
    worker.onerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, new Error(event.message || "Worker-owned preparation failed"));
    };
    worker.onmessageerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, new Error("Worker-owned preparation returned unreadable metadata"));
    };
    this.startupTimer = setTimeout(() => {
      if (this.worker === worker && !this.ready) {
        this.workerFailed(worker, new Error("Worker-owned preparation worker did not start"));
      }
    }, 3_000);
  }

  private message(worker: WorkerOwnedPreparationWorker, reply: StreamlineWorkerReply): void {
    if (this.worker !== worker) return;
    if (reply.kind === "ready") {
      if (this.startupTimer != null) clearTimeout(this.startupTimer);
      this.startupTimer = null;
      this.ready = true;
      this.startNext();
      return;
    }
    if (reply.kind === "result") {
      const extraction = this.activeExtraction;
      if (!extraction || reply.requestId !== extraction.requestId) return;
      if (this.executionTimer != null) clearTimeout(this.executionTimer);
      this.executionTimer = null;
      this.activeExtraction = null;
      if (!reply.ok) {
        this.failExtraction(extraction, new Error(reply.error));
        this.startNext();
        return;
      }
      if (extraction.rebuildReservationId != null) {
        const transitioned = visualizationResourceGovernor.transition(
          extraction.rebuildReservationId,
          [{
            allocationId: `worker-owned-vector:${extraction.backing.datasetId}`,
            stage: "worker-resident",
            bytes: extraction.backing.bytes,
          }],
        );
        if (!transitioned.ok) {
          this.failExtraction(extraction, new Error(transitioned.reason));
          this.startNext();
          return;
        }
        if (this.loaded?.reservationId) {
          visualizationResourceGovernor.release(this.loaded.reservationId);
          this.loaded.reservationId = 0;
        }
        extraction.backing.reservationId = extraction.rebuildReservationId;
        extraction.rebuildReservationId = null;
        this.loaded = extraction.backing;
      }
      recordVisualizationTransientReservation(
        "vtk-worker-owned-streamline:scratch",
        `request:${reply.requestId}`,
        "worker-scratch",
        reply.result.knownScratchBytes,
      );
      this.sharedExtractions.delete(extraction.identity);
      extraction.settled = true;
      extraction.resolve(reply.result);
      this.startNext();
      return;
    }
    if (reply.kind !== "prepared") return;
    const job = this.active;
    if (!job || reply.requestId !== job.requestId) return;
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    this.executionTimer = null;
    this.active = null;
    if (!reply.ok) {
      this.fail(job, new Error(reply.error));
      this.startNext();
      return;
    }
    const transitioned = visualizationResourceGovernor.transition(job.reservationId!, [{
      allocationId: `worker-owned-vector:${job.datasetId}`,
      stage: "worker-resident",
      bytes: reply.bytes,
    }]);
    if (!transitioned.ok) {
      this.fail(job, new Error(transitioned.reason));
      this.startNext();
      return;
    }
    visualizationResourceGovernor.recordPrediction(
      vtkPreparationOutputUpperBoundBytes(job.request),
      reply.bytes,
    );
    if (this.loaded?.reservationId) {
      visualizationResourceGovernor.release(this.loaded.reservationId);
      this.loaded.reservationId = 0;
    }
    const backing: Backing = {
      identity: job.identity,
      request: job.request,
      datasetId: job.datasetId,
      metadata: reply.metadata,
      bytes: reply.bytes,
      reservationId: job.reservationId!,
    };
    job.reservationId = null;
    this.loaded = backing;
    this.backings.clear();
    this.backings.set(job.identity, backing);
    this.shared.delete(job.identity);
      job.settled = true;
      job.resolve(backing);
    this.startNext();
    this.scheduleRetirement();
  }

  private startNext(): void {
    if (!this.worker || !this.ready || this.active || this.activeExtraction) return;
    let job: Job | undefined;
    while ((job = this.queue.shift())) {
      if (!job.settled && job.consumers > 0) break;
      job = undefined;
    }
    if (!job) {
      this.startNextExtraction();
      return;
    }
    this.active = job;
    job.requestId = ++this.requestSequence;
    job.attempts += 1;
    const input = vtkPreparationInputBytes(job.request);
    const output = vtkPreparationOutputUpperBoundBytes(job.request);
    const resources = visualizationResourceGovernor.reserve({
      owner: "vtk-worker-owned-vector-preparation",
      generation: `request:${job.requestId}:${job.attempts}`,
      priority: "visible",
      claims: [
        visualizationReservationClaim(`worker-owned-vector-input:${job.requestId}`, "worker-resident", input),
        visualizationReservationClaim(`worker-owned-vector-output:${job.requestId}`, "worker-resident", output),
      ],
    });
    if (!resources.ok) {
      this.active = null;
      this.fail(job, new Error(resources.reason));
      this.startNext();
      return;
    }
    job.reservationId = resources.reservation.id;
    void this.dispatchPreparation(job);
  }

  private async dispatchPreparation(job: Job): Promise<void> {
    try {
      const handoff = await handoffVtkPreparationInput(
        job.request,
        () => this.active !== job || job.settled,
      );
      const worker = this.worker;
      if (!worker || this.active !== job || job.settled) return;
      worker.postMessage({
        kind: "prepare",
        requestId: job.requestId,
        datasetId: job.datasetId,
        request: handoff.request,
      }, handoff.transfer);
    } catch (error) {
      const worker = this.worker;
      if (worker && this.active === job && !job.settled) {
        this.workerFailed(worker, error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    this.executionTimer = setTimeout(() => {
      if (this.active === job && this.worker) {
        this.workerFailed(this.worker, new Error("Worker-owned vector preparation timed out"));
      }
    }, 180_000);
  }

  private startNextExtraction(): void {
    let job: ExtractionJob | undefined;
    while ((job = this.extractionQueue.shift())) {
      if (!job.settled && job.consumers > 0) break;
      job = undefined;
    }
    if (!job || !this.worker || !this.ready) {
      if (!job) this.scheduleRetirement();
      return;
    }
    this.activeExtraction = job;
    job.requestId = ++this.requestSequence;
    job.attempts += 1;
    const needsRebuild = this.loaded?.datasetId !== job.backing.datasetId;
    if (needsRebuild) {
      const input = vtkPreparationInputBytes(job.backing.request);
      const output = vtkPreparationOutputUpperBoundBytes(job.backing.request);
      const resources = visualizationResourceGovernor.reserve({
        owner: "vtk-worker-owned-vector-rebuild",
        generation: `request:${job.requestId}:${job.attempts}`,
        priority: "visible",
        claims: [
          visualizationReservationClaim(`worker-owned-rebuild-input:${job.requestId}`, "worker-resident", input),
          visualizationReservationClaim(`worker-owned-rebuild-output:${job.requestId}`, "worker-resident", output),
        ],
      });
      if (!resources.ok) {
        this.activeExtraction = null;
        this.failExtraction(job, new Error(resources.reason));
        this.startNext();
        return;
      }
      job.rebuildReservationId = resources.reservation.id;
    }
    void this.dispatchExtraction(job, needsRebuild);
  }

  private async dispatchExtraction(job: ExtractionJob, needsRebuild: boolean): Promise<void> {
    try {
      const handoff = needsRebuild
        ? await handoffVtkPreparationInput(
          job.backing.request,
          () => this.activeExtraction !== job || job.settled,
        )
        : null;
      const worker = this.worker;
      if (!worker || this.activeExtraction !== job || job.settled) return;
      const seeds = job.parameters.seeds.slice();
      worker.postMessage({
        kind: "extract",
        requestId: job.requestId,
        datasetId: job.backing.datasetId,
        ...job.parameters,
        seeds,
        ...(handoff ? { preparation: handoff.request } : {}),
      }, [seeds.buffer, ...(handoff?.transfer ?? [])]);
    } catch (error) {
      const worker = this.worker;
      if (worker && this.activeExtraction === job && !job.settled) {
        this.workerFailed(worker, error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    this.executionTimer = setTimeout(() => {
      if (this.activeExtraction === job && this.worker) {
        this.workerFailed(this.worker, new Error("Worker-owned streamline extraction timed out"));
      }
    }, 180_000);
  }

  private workerFailed(worker: WorkerOwnedPreparationWorker, error: Error): void {
    if (this.worker !== worker) return;
    const active = this.active;
    const activeExtraction = this.activeExtraction;
    this.active = null;
    this.activeExtraction = null;
    this.disposeWorker();
    if (active?.reservationId != null) {
      visualizationResourceGovernor.release(active.reservationId);
      active.reservationId = null;
    }
    if (active && !active.settled && active.consumers > 0 && active.attempts < 2) {
      this.queue.unshift(active);
    } else if (active && !active.settled) {
      this.fail(active, error);
    }
    if (activeExtraction?.rebuildReservationId != null) {
      visualizationResourceGovernor.release(activeExtraction.rebuildReservationId);
      activeExtraction.rebuildReservationId = null;
    }
    if (activeExtraction && !activeExtraction.settled
        && activeExtraction.consumers > 0 && activeExtraction.attempts < 2) {
      this.extractionQueue.unshift(activeExtraction);
    } else if (activeExtraction && !activeExtraction.settled) {
      this.failExtraction(activeExtraction, error);
    }
    this.restartForQueue();
  }

  private fail(job: Job, error: unknown): void {
    if (job.settled) return;
    if (job.reservationId != null) visualizationResourceGovernor.release(job.reservationId);
    job.reservationId = null;
    job.settled = true;
    this.shared.delete(job.identity);
    job.reject(error);
  }

  private failExtraction(job: ExtractionJob, error: unknown): void {
    if (job.settled) return;
    if (job.rebuildReservationId != null) {
      visualizationResourceGovernor.release(job.rebuildReservationId);
      job.rebuildReservationId = null;
    }
    job.settled = true;
    this.sharedExtractions.delete(job.identity);
    job.reject(error);
  }

  private restartForQueue(): void {
    if (this.hasQueuedWork()) {
      setTimeout(() => {
        if (!this.hasQueuedWork()) return;
        this.ensureWorker();
        this.startNext();
      }, 25);
    }
  }

  private scheduleRetirement(): void {
    if (!this.worker || this.handleBackings.size > 0 || this.active || this.activeExtraction
        || this.hasQueuedWork() || this.retireTimer != null) {
      return;
    }
    vtkExtractionServicePool.park(this, () => {
      this.retireIfUnused();
    });
    if (!this.worker) return;
    this.retireTimer = setTimeout(() => {
      this.retireTimer = null;
      this.retireIfUnused();
    }, 15_000);
  }
  private retireIfUnused(): boolean {
    if (this.handleBackings.size > 0 || this.active || this.activeExtraction
        || this.hasQueuedWork()) return false;
    this.disposeWorker();
    return true;
  }
  private hasQueuedWork(): boolean {
    return this.queue.some((job) => !job.settled && job.consumers > 0)
      || this.extractionQueue.some((job) => !job.settled && job.consumers > 0);
  }

  private disposeWorker(): void {
    vtkExtractionServicePool.remove(this);
    const worker = this.worker;
    this.worker = null;
    this.ready = false;
    if (this.startupTimer != null) clearTimeout(this.startupTimer);
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.startupTimer = this.executionTimer = this.retireTimer = null;
    if (this.loaded?.reservationId) {
      visualizationResourceGovernor.release(this.loaded.reservationId);
      this.loaded.reservationId = 0;
    }
    this.loaded = null;
    this.backings.clear();
    if (!worker) return;
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    worker.terminate();
  }
}

let defaultService: StreamlineWorkerOwnedPreparationService | null = null;

function sharedService(): StreamlineWorkerOwnedPreparationService {
  defaultService ??= new StreamlineWorkerOwnedPreparationService();
  return defaultService;
}

export function prepareVectorInWorkerOwnedService(
  request: VectorPreparationRequest,
  signal?: AbortSignal,
): Promise<VtkPreparedDatasetLease<PreparedVectorImage>> {
  return sharedService().prepare(request, signal);
}

export function extractStreamlinesFromWorkerOwnedHandle(
  handle: VtkPreparedDatasetHandle<PreparedVectorImage>,
  parameters: StreamlineExtractionParameters,
  signal?: AbortSignal,
): Promise<StreamlineExtractionResult> {
  return sharedService().extract(handle, parameters, signal);
}

export function shutdownWorkerOwnedStreamlineService(): void {
  defaultService?.shutdown();
  defaultService = null;
}
