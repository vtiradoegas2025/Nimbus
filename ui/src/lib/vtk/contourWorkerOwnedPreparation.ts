import { visualizationStagingQueue } from "@/lib/core/visualizationStagingQueue";
import type { PreparedVtkImage, ScalarPreparationRequest } from "./vtkPreparation";
import type {
  PreparedScalarMetadata,
  VortexContourResult,
  VortexContourThresholds,
  VortexContourWorkerReply,
  VortexContourWorkerRequest,
} from "./vortexContourProtocol";
import { vortexContourTransferables } from "./vortexContourProtocol";
import vortexContourWorkerUrl from "./vortexContour.worker.ts?worker&url";
import { vtkPreparationIdentity } from "./vtkPreparationClient";
import {
  retainRemotePreparedDataset,
  type VtkPreparedDatasetHandle,
  type VtkPreparedDatasetLease,
} from "./vtkPreparedDatasetRegistry";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import {
  vtkPreparationInputBytes,
  vtkPreparationOutputUpperBoundBytes,
} from "./vtkPreparationResourcePlanning";
import { handoffVtkPreparationInput } from "./vtkPreparationInputHandoff";
import { vtkExtractionServicePool } from "./vtkExtractionServicePool";
import { contourUsageViolation } from "./vtkContourBudget";

export interface ContourOwnedPreparationWorker {
  onmessage: ((event: MessageEvent<VortexContourWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: VortexContourWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}
export type ContourOwnedPreparationWorkerFactory = () => ContourOwnedPreparationWorker;

interface Backing {
  namespace: string;
  identity: string;
  request: ScalarPreparationRequest;
  metadata: PreparedScalarMetadata;
  bytes: number;
  datasetId: number;
  reservationId: number;
}
export interface WorkerOwnedContourParameters {
  thresholds: VortexContourThresholds;
  anchor: [number, number, number] | null;
  condensateFloor: number | null;
}
interface ExtractionJob {
  identity: string;
  pairIdentity: string;
  primary: Backing;
  condensate: Backing | null;
  parameters: WorkerOwnedContourParameters;
  pairDatasetId: number;
  requestId: number;
  attempts: number;
  consumers: number;
  settled: boolean;
  cancelStaging?: () => void;
  rebuildReservationId: number | null;
  outputReservationId: number | null;
  maximumOutputBytes: number;
  maximumTriangles: number;
  maximumScratchBytes: number;
  promise: Promise<VortexContourResult>;
  resolve: (result: VortexContourResult) => void;
  reject: (error: unknown) => void;
}
interface CompletedExtraction {
  identity: string;
  result: VortexContourResult;
  reservationId: number;
}
interface Job {
  identity: string;
  request: ScalarPreparationRequest;
  requestId: number;
  datasetId: number;
  attempts: number;
  consumers: number;
  settled: boolean;
  cancelStaging?: () => void;
  reservationId: number | null;
  promise: Promise<Backing>;
  resolve: (backing: Backing) => void;
  reject: (error: unknown) => void;
}

let workerGeneration = 0;
let serviceSequence = 0;
function defaultWorkerFactory(): ContourOwnedPreparationWorker {
  const separator = vortexContourWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${vortexContourWorkerUrl}${separator}owned=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-worker-owned-contour-preparation",
  }) as unknown as ContourOwnedPreparationWorker;
}
function abortError(): DOMException {
  return new DOMException("Worker-owned contour preparation cancelled", "AbortError");
}

const remoteBackings = new Map<number, Backing>();
const handleServices = new Map<number, ContourWorkerOwnedPreparationService>();
const progressByDataset = new Map<string, string>();

export function workerOwnedContourProgressSnapshot(): Readonly<Record<string, string>> {
  return Object.freeze(Object.fromEntries(progressByDataset));
}

export class ContourWorkerOwnedPreparationService {
  private readonly workerFactory: ContourOwnedPreparationWorkerFactory;
  private readonly namespace: string;
  private readonly onRetired: (() => void) | null;
  private readonly shared = new Map<string, Job>();
  private readonly backings = new Map<string, Backing>();
  private readonly queue: Job[] = [];
  private readonly handleBackings = new Map<number, Backing>();
  private readonly sharedExtractions = new Map<string, ExtractionJob>();
  private readonly extractionQueue: ExtractionJob[] = [];
  private readonly pairDatasetIds = new Map<string, number>();
  private readonly progressKeys = new Set<string>();
  private worker: ContourOwnedPreparationWorker | null = null;
  private ready = false;
  private active: Job | null = null;
  private activeExtraction: ExtractionJob | null = null;
  private completedExtraction: CompletedExtraction | null = null;
  private loaded: Backing | null = null;
  private loadedPairIdentity: string | null = null;
  private pendingRelease: { requestId: number; proceed: () => void } | null = null;
  private requestSequence = 0;
  private datasetSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    workerFactory: ContourOwnedPreparationWorkerFactory = defaultWorkerFactory,
    namespace = `contour-service:${++serviceSequence}`,
    onRetired: (() => void) | null = null,
  ) {
    this.workerFactory = workerFactory;
    this.namespace = namespace;
    this.onRetired = onRetired;
  }

  prepare(
    request: ScalarPreparationRequest,
    signal?: AbortSignal,
    identityOverride?: string,
  ): Promise<VtkPreparedDatasetLease<PreparedVtkImage>> {
    if (signal?.aborted) return Promise.reject(abortError());
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    const identity = identityOverride ?? vtkPreparationIdentity(request);
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
        requestId: 0,
        datasetId: ++this.datasetSequence,
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
    primaryHandle: VtkPreparedDatasetHandle<PreparedVtkImage>,
    condensateHandle: VtkPreparedDatasetHandle<PreparedVtkImage> | null,
    parameters: WorkerOwnedContourParameters,
    signal?: AbortSignal,
  ): Promise<VortexContourResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    const primary = remoteBackings.get(primaryHandle.id);
    const condensate = condensateHandle ? remoteBackings.get(condensateHandle.id) ?? null : null;
    if (!primary || primaryHandle.residency !== "worker"
        || (condensateHandle && (!condensate || condensateHandle.residency !== "worker"))) {
      return Promise.reject(new Error("Worker-owned contour handle pair is unavailable"));
    }
    const pairIdentity = JSON.stringify([
      primary.namespace,
      primary.identity,
      condensate?.namespace ?? null,
      condensate?.identity ?? null,
      primary.metadata.shape,
      primary.metadata.spacing,
      primary.metadata.origin,
      primary.request.featureSign ?? null,
      primary.request.featureZRangeM ?? null,
      condensate ? [
        condensate.metadata.shape,
        condensate.metadata.spacing,
        condensate.metadata.origin,
      ] : null,
    ]);
    const identity = JSON.stringify([
      pairIdentity,
      parameters.thresholds.core,
      parameters.thresholds.opposite,
      parameters.thresholds.mesocyclone,
      parameters.anchor,
      parameters.condensateFloor,
    ]);
    if (this.completedExtraction?.identity === identity) {
      return Promise.resolve(this.completedExtraction.result);
    }
    let job = this.sharedExtractions.get(identity);
    if (!job) {
      let resolve!: (result: VortexContourResult) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<VortexContourResult>((yes, no) => { resolve = yes; reject = no; });
      job = {
        identity,
        pairIdentity,
        primary,
        condensate,
        parameters,
        pairDatasetId: this.pairDatasetIds.get(pairIdentity)
          ?? (!condensate && primary.namespace === this.namespace
            ? primary.datasetId : ++this.datasetSequence),
        requestId: 0,
        attempts: 0,
        consumers: 0,
        settled: false,
        rebuildReservationId: null,
        outputReservationId: null,
        maximumOutputBytes: 0,
        maximumTriangles: 0,
        maximumScratchBytes: 0,
        promise,
        resolve,
        reject,
      };
      this.sharedExtractions.set(identity, job);
      this.pairDatasetIds.set(pairIdentity, job.pairDatasetId);
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
    this.clearRetiredState();
  }

  private lease(backing: Backing): VtkPreparedDatasetLease<PreparedVtkImage> {
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    let handleId = 0;
    const lease = retainRemotePreparedDataset<PreparedVtkImage>(backing, {
      kind: "scalar",
      bytes: backing.bytes,
      metadata: backing.metadata,
      onLastRelease: () => {
        this.handleBackings.delete(handleId);
        remoteBackings.delete(handleId);
        handleServices.delete(handleId);
        if (!this.retireIfUnused()) this.scheduleRetirement();
      },
    });
    handleId = lease.handle.id;
    this.handleBackings.set(handleId, backing);
    remoteBackings.set(handleId, backing);
    handleServices.set(handleId, this);
    return lease;
  }
  private cancel(job: Job): void {
    const index = this.queue.indexOf(job);
    if (index >= 0) {
      this.queue.splice(index, 1);
      this.fail(job, abortError());
      this.retireIfUnused();
    } else if (this.active === job) {
      this.active = null;
      this.fail(job, abortError());
      this.disposeWorker();
      this.restart();
    }
  }
  private cancelExtraction(job: ExtractionJob): void {
    const index = this.extractionQueue.indexOf(job);
    if (index >= 0) {
      this.extractionQueue.splice(index, 1);
      this.failExtraction(job, abortError());
      this.retireIfUnused();
    } else if (this.activeExtraction === job) {
      this.activeExtraction = null;
      this.failExtraction(job, abortError());
      this.disposeWorker();
      this.restart();
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
      this.workerFailed(worker, new Error(event.message || "Worker-owned contour preparation failed"));
    };
    worker.onmessageerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, new Error("Worker-owned contour preparation returned unreadable metadata"));
    };
    this.startupTimer = setTimeout(() => {
      if (this.worker === worker && !this.ready) {
        this.workerFailed(worker, new Error("Worker-owned contour preparation worker did not start"));
      }
    }, 3_000);
  }
  private message(worker: ContourOwnedPreparationWorker, reply: VortexContourWorkerReply): void {
    if (this.worker !== worker) return;
    if (reply.kind === "ready") {
      if (this.startupTimer != null) clearTimeout(this.startupTimer);
      this.startupTimer = null;
      this.ready = true;
      this.startNext();
      return;
    }
    if (reply.kind === "released") {
      const pending = this.pendingRelease;
      if (!pending || reply.requestId !== pending.requestId) return;
      this.pendingRelease = null;
      if (this.executionTimer != null) clearTimeout(this.executionTimer);
      this.executionTimer = null;
      if (this.loaded?.reservationId) {
        visualizationResourceGovernor.release(this.loaded.reservationId);
        this.loaded.reservationId = 0;
      }
      this.loaded = null;
      this.loadedPairIdentity = null;
      this.backings.clear();
      pending.proceed();
      return;
    }
    if (reply.kind === "progress") {
      this.recordProgress(reply.datasetId, reply.stage);
      return;
    }
    if (reply.kind === "admission") {
      const extraction = this.activeExtraction;
      if (!extraction || reply.requestId !== extraction.requestId) return;
      try {
        if (extraction.rebuildReservationId != null) {
          const bytes = extraction.primary.bytes + (extraction.condensate?.bytes ?? 0);
          const transitioned = visualizationResourceGovernor.transition(
            extraction.rebuildReservationId,
            [{
              allocationId: `${this.namespace}:worker-owned-contour-pair:${extraction.pairDatasetId}`,
              stage: "worker-resident",
              bytes,
            }],
          );
          if (!transitioned.ok) {
            throw new Error(transitioned.reason);
          }
          if (this.loaded?.reservationId) {
            visualizationResourceGovernor.release(this.loaded.reservationId);
            this.loaded.reservationId = 0;
          }
          extraction.primary.reservationId = extraction.rebuildReservationId;
          extraction.rebuildReservationId = null;
          this.loaded = extraction.primary;
          this.loadedPairIdentity = extraction.pairIdentity;
          this.backings.clear();
          this.backings.set(extraction.primary.identity, extraction.primary);
        }
      } catch (error) {
        this.workerFailed(worker, error instanceof Error ? error : new Error(String(error)));
        return;
      }
      extraction.cancelStaging = visualizationStagingQueue.schedule({
        owner: `vtk-worker-owned-contour-output:${this.namespace}`,
        generation: `request:${reply.requestId}:${extraction.attempts}`,
        priority: "visible",
        claims: [
          visualizationReservationClaim(
            `${this.namespace}:worker-owned-contour-geometry:${reply.requestId}`,
            "retained-geometry",
            reply.admission.maximumOutputBytes,
          ),
          visualizationReservationClaim(
            `${this.namespace}:worker-owned-contour-scratch:${reply.requestId}`,
            "worker-scratch",
            reply.admission.maximumScratchBytes,
          ),
        ],
      }, (resources) => {
        extraction.cancelStaging = undefined;
        if (!resources.ok) {
          this.activeExtraction = null;
          this.failExtraction(extraction, new Error(resources.reason));
          this.startNext();
          return;
        }
        extraction.outputReservationId = resources.reservation.id;
        extraction.maximumOutputBytes = reply.admission.maximumOutputBytes;
        extraction.maximumTriangles = reply.admission.maximumTriangles;
        extraction.maximumScratchBytes = reply.admission.maximumScratchBytes;
        worker.postMessage({ kind: "continue", requestId: reply.requestId,
          datasetId: reply.datasetId, admission: reply.admission });
      });
      return;
    }
    if (reply.kind === "result") {
      const extraction = this.activeExtraction;
      if (!extraction || reply.requestId !== extraction.requestId) return;
      this.recordProgress(reply.datasetId, "service-result-received");
      if (this.executionTimer != null) clearTimeout(this.executionTimer);
      this.executionTimer = null;
      this.activeExtraction = null;
      if (!reply.ok) {
        this.failExtraction(extraction, new Error(reply.error));
        this.startNext();
        return;
      }
      try {
        const outputBuffers = [...new Set(vortexContourTransferables(reply.result))];
        const outputBytes = outputBuffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
        const violation = contourUsageViolation(
          reply.result.usage,
          "Worker-owned contour output",
          extraction.maximumOutputBytes,
          extraction.maximumTriangles,
        );
        if (violation || outputBytes > extraction.maximumOutputBytes
            || reply.result.knownScratchBytes > extraction.maximumScratchBytes
            || extraction.outputReservationId == null) {
          throw new Error(violation
            ?? "Worker-owned contour output exceeded its topology-derived admission");
        }
        const retained = visualizationResourceGovernor.transition(
          extraction.outputReservationId,
          outputBuffers.map((buffer) => visualizationBufferClaim(buffer, "retained-geometry")),
        );
        if (!retained.ok) throw new Error(retained.reason);
        visualizationResourceGovernor.recordPrediction(
          extraction.maximumOutputBytes + extraction.maximumScratchBytes,
          outputBytes + reply.result.knownScratchBytes,
        );
        this.cacheCompletedExtraction(
          extraction.identity,
          reply.result,
          extraction.outputReservationId,
        );
        extraction.outputReservationId = null;
        this.sharedExtractions.delete(extraction.identity);
        extraction.settled = true;
        extraction.resolve(reply.result);
        this.recordProgress(reply.datasetId, "service-resolved");
      } catch (error) {
        this.recordProgress(reply.datasetId, "service-failed");
        this.failExtraction(extraction, error);
      }
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
    try {
      const transitioned = visualizationResourceGovernor.transition(job.reservationId!, [{
        allocationId: `${this.namespace}:worker-owned-contour:${job.datasetId}`,
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
        namespace: this.namespace,
        identity: job.identity,
        request: job.request,
        metadata: reply.metadata,
        bytes: reply.bytes,
        datasetId: job.datasetId,
        reservationId: job.reservationId!,
      };
      job.reservationId = null;
      this.loaded = backing;
      this.loadedPairIdentity = null;
      this.backings.clear();
      this.backings.set(job.identity, backing);
      this.shared.delete(job.identity);
      job.settled = true;
      job.resolve(backing);
    } catch (error) {
      this.fail(job, error);
    }
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
    this.releaseForReplacement(job.requestId, () => {
      job.cancelStaging = visualizationStagingQueue.schedule({
        owner: `vtk-worker-owned-contour-preparation:${this.namespace}`,
        generation: `request:${job.requestId}:${job.attempts}`,
        priority: "visible",
        claims: [
          visualizationReservationClaim(`${this.namespace}:worker-owned-contour-input:${job.requestId}`, "worker-resident", vtkPreparationInputBytes(job.request)),
          visualizationReservationClaim(`${this.namespace}:worker-owned-contour-output:${job.requestId}`, "worker-resident", vtkPreparationOutputUpperBoundBytes(job.request)),
        ],
      }, (resources) => {
        job.cancelStaging = undefined;
        if (!resources.ok) {
          this.active = null;
          this.fail(job, new Error(resources.reason));
          this.startNext();
          return;
        }
        job.reservationId = resources.reservation.id;
        void this.dispatchPreparation(job);
      });
    });
  }
  private releaseForReplacement(requestId: number, proceed: () => void): void {
    const worker = this.worker;
    if (!worker || !this.loaded) { proceed(); return; }
    this.pendingRelease = { requestId, proceed };
    this.executionTimer = setTimeout(() => {
      if (this.pendingRelease?.requestId === requestId && this.worker === worker) {
        this.workerFailed(worker, new Error("Contour worker did not release its displaced dataset"));
      }
    }, 3_000);
    worker.postMessage({ kind: "release", requestId });
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
        this.workerFailed(this.worker, new Error("Worker-owned contour preparation timed out"));
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
    if (!job.condensate && this.loaded === job.primary && this.loadedPairIdentity == null) {
      this.loadedPairIdentity = job.pairIdentity;
    }
    const needsRebuild = this.loadedPairIdentity !== job.pairIdentity;
    if (needsRebuild) {
      const input = vtkPreparationInputBytes(job.primary.request)
        + (job.condensate ? vtkPreparationInputBytes(job.condensate.request) : 0);
      const output = vtkPreparationOutputUpperBoundBytes(job.primary.request)
        + (job.condensate ? vtkPreparationOutputUpperBoundBytes(job.condensate.request) : 0);
      this.releaseForReplacement(job.requestId, () => {
        job.cancelStaging = visualizationStagingQueue.schedule({
          owner: `vtk-worker-owned-contour-rebuild:${this.namespace}`,
          generation: `request:${job.requestId}:${job.attempts}`,
          priority: "visible",
          claims: [
            visualizationReservationClaim(`${this.namespace}:worker-owned-contour-rebuild-input:${job.requestId}`, "worker-resident", input),
            visualizationReservationClaim(`${this.namespace}:worker-owned-contour-rebuild-output:${job.requestId}`, "worker-resident", output),
          ],
        }, (resources) => {
          job.cancelStaging = undefined;
          if (!resources.ok) {
            this.activeExtraction = null;
            this.failExtraction(job, new Error(resources.reason));
            this.startNext();
            return;
          }
          job.rebuildReservationId = resources.reservation.id;
          void this.dispatchExtraction(job, true);
        });
      });
    } else {
      void this.dispatchExtraction(job, false);
    }
  }
  private async dispatchExtraction(job: ExtractionJob, needsRebuild: boolean): Promise<void> {
    try {
      const primaryHandoff = needsRebuild
        ? await handoffVtkPreparationInput(
          job.primary.request,
          () => this.activeExtraction !== job || job.settled,
        )
        : null;
      const condensateHandoff = needsRebuild && job.condensate
        ? await handoffVtkPreparationInput(
          job.condensate.request,
          () => this.activeExtraction !== job || job.settled,
        )
        : null;
      const worker = this.worker;
      if (!worker || this.activeExtraction !== job || job.settled) return;
      worker.postMessage({
        kind: "extract",
        requestId: job.requestId,
        datasetId: job.pairDatasetId,
        thresholds: job.parameters.thresholds,
        anchor: job.parameters.anchor,
        condensateFloor: job.parameters.condensateFloor,
        ...(primaryHandoff ? {
          preparation: primaryHandoff.request,
          ...(condensateHandoff ? { condensatePreparation: condensateHandoff.request } : {}),
        } : {}),
        reportProgress: job.condensate != null,
      }, [...(primaryHandoff?.transfer ?? []), ...(condensateHandoff?.transfer ?? [])]);
    } catch (error) {
      const worker = this.worker;
      if (worker && this.activeExtraction === job && !job.settled) {
        this.workerFailed(worker, error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    this.executionTimer = setTimeout(() => {
      if (this.activeExtraction === job && this.worker) {
        this.workerFailed(this.worker, new Error("Worker-owned contour extraction timed out"));
      }
    }, 180_000);
  }
  private workerFailed(worker: ContourOwnedPreparationWorker, error: Error): void {
    if (this.worker !== worker) return;
    const active = this.active;
    const activeExtraction = this.activeExtraction;
    this.active = null;
    this.activeExtraction = null;
    this.disposeWorker();
    active?.cancelStaging?.();
    if (active) active.cancelStaging = undefined;
    activeExtraction?.cancelStaging?.();
    if (activeExtraction) activeExtraction.cancelStaging = undefined;
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
    if (activeExtraction?.outputReservationId != null) {
      visualizationResourceGovernor.release(activeExtraction.outputReservationId);
      activeExtraction.outputReservationId = null;
    }
    if (activeExtraction && !activeExtraction.settled
        && activeExtraction.consumers > 0 && activeExtraction.attempts < 2) {
      this.extractionQueue.unshift(activeExtraction);
    } else if (activeExtraction && !activeExtraction.settled) {
      this.failExtraction(activeExtraction, error);
    }
    this.restart();
  }
  private fail(job: Job, error: unknown): void {
    if (job.settled) return;
    job.cancelStaging?.();
    job.cancelStaging = undefined;
    if (job.reservationId != null) visualizationResourceGovernor.release(job.reservationId);
    job.reservationId = null;
    job.settled = true;
    this.shared.delete(job.identity);
    job.reject(error);
  }
  private failExtraction(job: ExtractionJob, error: unknown): void {
    if (job.settled) return;
    job.cancelStaging?.();
    job.cancelStaging = undefined;
    if (job.rebuildReservationId != null) {
      visualizationResourceGovernor.release(job.rebuildReservationId);
      job.rebuildReservationId = null;
    }
    if (job.outputReservationId != null) {
      visualizationResourceGovernor.release(job.outputReservationId);
      job.outputReservationId = null;
    }
    job.settled = true;
    this.sharedExtractions.delete(job.identity);
    job.reject(error);
  }
  private cacheCompletedExtraction(
    identity: string,
    result: VortexContourResult,
    reservationId: number,
  ): void {
    if (this.completedExtraction) {
      visualizationResourceGovernor.release(this.completedExtraction.reservationId);
    }
    this.completedExtraction = {
      identity,
      result,
      reservationId,
    };
  }
  private restart(): void {
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
        || this.hasQueuedWork() || this.retireTimer != null) return;
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
    this.clearRetiredState();
    this.onRetired?.();
    return true;
  }
  private hasQueuedWork(): boolean {
    return this.queue.some((job) => !job.settled && job.consumers > 0)
      || this.extractionQueue.some((job) => !job.settled && job.consumers > 0);
  }
  private disposeWorker(): void {
    this.pendingRelease = null;
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
    this.loadedPairIdentity = null;
    this.backings.clear();
    if (this.completedExtraction) {
      visualizationResourceGovernor.release(this.completedExtraction.reservationId);
      this.completedExtraction = null;
    }
    if (!worker) return;
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    worker.terminate();
  }

  private clearRetiredState(): void {
    this.pairDatasetIds.clear();
    for (const key of this.progressKeys) progressByDataset.delete(key);
    this.progressKeys.clear();
  }

  private recordProgress(datasetId: number, stage: string): void {
    const key = `${this.namespace}:${datasetId}`;
    this.progressKeys.add(key);
    progressByDataset.set(key, stage);
  }
}

const servicesByPreparation = new Map<string, ContourWorkerOwnedPreparationService>();

function semanticScalarPreparationIdentity(request: ScalarPreparationRequest): string {
  const geometry = (field: ScalarPreparationRequest["field"] | undefined) => field ? {
    shape: field.shape,
    window: field.windowGeometry ? {
      aggregation: field.windowGeometry.aggregation,
      weighting: field.windowGeometry.weighting,
      sampleLocation: field.windowGeometry.sampleLocation,
      sourceShape: field.windowGeometry.sourceShape,
      origin: field.windowGeometry.origin,
      extent: field.windowGeometry.extent,
      stride: field.windowGeometry.stride,
      sourceFormat: field.windowGeometry.sourceFormat,
    } : null,
  } : null;
  return JSON.stringify([
    geometry(request.field),
    geometry(request.subtractField),
    request.maxVoxels ?? null,
    request.refineForDisplay ?? false,
    request.featureSign ?? null,
    request.featureZRangeM ?? null,
    request.scalarTransform ?? "none",
    request.structuralSupportRangeFraction ?? null,
    request.grid.scalarOriginM ?? null,
    request.grid.physicalBoundsM ?? null,
  ]);
}

export function prepareScalarInWorkerOwnedService(
  request: ScalarPreparationRequest,
  signal?: AbortSignal,
  serviceKey?: string,
): Promise<VtkPreparedDatasetLease<PreparedVtkImage>> {
  const identity = vtkPreparationIdentity(request);
  const registryKey = serviceKey ?? identity;
  let service = servicesByPreparation.get(registryKey);
  if (!service) {
    let created!: ContourWorkerOwnedPreparationService;
    created = new ContourWorkerOwnedPreparationService(defaultWorkerFactory, registryKey, () => {
      if (servicesByPreparation.get(registryKey) === created) {
        servicesByPreparation.delete(registryKey);
      }
    });
    service = created;
    servicesByPreparation.set(registryKey, service);
  }
  return service.prepare(
    request,
    signal,
    serviceKey ? semanticScalarPreparationIdentity(request) : undefined,
  );
}

export function extractContoursFromWorkerOwnedHandles(
  primaryHandle: VtkPreparedDatasetHandle<PreparedVtkImage>,
  condensateHandle: VtkPreparedDatasetHandle<PreparedVtkImage> | null,
  parameters: WorkerOwnedContourParameters,
  signal?: AbortSignal,
): Promise<VortexContourResult> {
  const primary = remoteBackings.get(primaryHandle.id);
  const service = primary
    ? servicesByPreparation.get(primary.namespace) ?? handleServices.get(primaryHandle.id)
    : handleServices.get(primaryHandle.id);
  if (!service) {
    return Promise.reject(new Error(`Worker-owned contour handle ${primaryHandle.id} has no service`));
  }
  return service.extract(primaryHandle, condensateHandle, parameters, signal);
}

export function shutdownWorkerOwnedContourServices(): void {
  for (const service of new Set(servicesByPreparation.values())) service.shutdown();
  servicesByPreparation.clear();
}
