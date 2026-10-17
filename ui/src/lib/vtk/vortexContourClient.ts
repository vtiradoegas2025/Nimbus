import type { PreparedVtkImage } from "./vtkPreparation";
import type {
  VortexContourResult,
  VortexContourThresholds,
  VortexContourWorkerReply,
  VortexContourWorkerRequest,
} from "./vortexContourProtocol";
import { vortexContourTransferables } from "./vortexContourProtocol";
import {
  handoffVtkExtractionDataset,
  isSharedVtkDataset,
  recordWarmVtkDatasetReuse,
  recordVtkDatasetHandoff,
} from "./vtkDatasetHandoff";
import vortexContourWorkerUrl from "./vortexContour.worker.ts?worker&url";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import { contourUsageViolation } from "./vtkContourBudget";
import { VtkWorkerDatasetOwnership } from "./vtkWorkerDatasetOwnership";
import { vtkExtractionServicePool } from "./vtkExtractionServicePool";

export interface VortexContourRequestParameters {
  thresholds: VortexContourThresholds;
  anchor: [number, number, number] | null;
  condensateFloor: number | null;
}

export interface VortexContourExtractionWorker {
  onmessage: ((event: MessageEvent<VortexContourWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: VortexContourWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

export type VortexContourWorkerFactory = () => VortexContourExtractionWorker;

interface Job {
  identity: string;
  parameters: VortexContourRequestParameters;
  requestId: number;
  attempts: number;
  consumers: number;
  settled: boolean;
  outputReservationId: number | null;
  maximumOutputBytes: number;
  maximumTriangles: number;
  maximumScratchBytes: number;
  promise: Promise<VortexContourResult>;
  resolve: (result: VortexContourResult) => void;
  reject: (error: unknown) => void;
}

export const CONTOUR_WORKER_IDLE_RETIRE_MS = 15_000;
let workerGeneration = 0;
let serviceSequence = 0;

function defaultWorkerFactory(): VortexContourExtractionWorker {
  const separator = vortexContourWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${vortexContourWorkerUrl}${separator}generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-vortex-contours",
  }) as unknown as VortexContourExtractionWorker;
}

function abortError(): DOMException {
  return new DOMException("Vortex contour extraction cancelled", "AbortError");
}

function requestIdentity(parameters: VortexContourRequestParameters): string {
  return JSON.stringify([
    parameters.thresholds.core,
    parameters.thresholds.opposite,
    parameters.thresholds.mesocyclone,
    parameters.anchor,
    parameters.condensateFloor,
  ]);
}

export class SharedVortexContourDatasetService {
  private readonly workerFactory: VortexContourWorkerFactory;
  private readonly prepared: PreparedVtkImage;
  private readonly condensatePrepared: PreparedVtkImage | null;
  private readonly ownership: VtkWorkerDatasetOwnership;
  private readonly shared = new Map<string, Job>();
  private readonly queue: Job[] = [];
  private readonly datasetId = ++serviceSequence;
  private worker: VortexContourExtractionWorker | null = null;
  private ready = false;
  private datasetLoaded = false;
  private active: Job | null = null;
  private requestSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;
  private completedOutputReservationId: number | null = null;

  constructor(
    prepared: PreparedVtkImage,
    condensatePrepared: PreparedVtkImage | null,
    workerFactory: VortexContourWorkerFactory = defaultWorkerFactory,
  ) {
    this.prepared = prepared;
    this.condensatePrepared = condensatePrepared;
    this.workerFactory = workerFactory;
    this.ownership = new VtkWorkerDatasetOwnership(`vtk:contour-service:${this.datasetId}:dataset`);
  }

  enqueue(parameters: VortexContourRequestParameters, signal?: AbortSignal): Promise<VortexContourResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    const identity = requestIdentity(parameters);
    let job = this.shared.get(identity);
    if (!job) {
      let resolve!: (result: VortexContourResult) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<VortexContourResult>((yes, no) => { resolve = yes; reject = no; });
      job = { identity, parameters, requestId: 0, attempts: 0, consumers: 0,
        settled: false, outputReservationId: null, maximumOutputBytes: 0,
        maximumTriangles: 0, maximumScratchBytes: 0, promise, resolve, reject };
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
    if (this.completedOutputReservationId != null) {
      visualizationResourceGovernor.release(this.completedOutputReservationId);
      this.completedOutputReservationId = null;
    }
    this.disposeWorker();
  }

  private cancel(job: Job): void {
    const index = this.queue.indexOf(job);
    if (index >= 0) {
      this.queue.splice(index, 1);
      this.finish(job, abortError());
      return;
    }
    if (this.active === job) {
      this.active = null;
      this.finish(job, abortError());
      this.disposeWorker();
      this.restartForQueue();
    }
  }

  private finish(job: Job, outcome: VortexContourResult | Error | DOMException): void {
    if (job.settled) return;
    job.settled = true;
    if (!("usage" in outcome) && job.outputReservationId != null) {
      visualizationResourceGovernor.release(job.outputReservationId);
      job.outputReservationId = null;
    }
    if (this.shared.get(job.identity) === job) this.shared.delete(job.identity);
    if ("usage" in outcome) job.resolve(outcome);
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
    this.datasetLoaded = false;
    worker.onmessage = (event) => this.handleMessage(worker, event.data);
    worker.onerror = (event) => {
      event.preventDefault?.();
      const detail = event.error instanceof Error ? event.error.stack || event.error.message : event.message;
      this.workerFailed(worker, detail || "Vortex contour worker failed");
    };
    worker.onmessageerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, "Vortex contour worker returned an unreadable result");
    };
    this.startupTimer = setTimeout(() => {
      if (this.worker === worker && !this.ready) this.workerFailed(worker, "Vortex contour worker did not start");
    }, 3_000);
  }

  private handleMessage(worker: VortexContourExtractionWorker, reply: VortexContourWorkerReply): void {
    if (this.worker !== worker) return;
    if (reply.kind === "ready") {
      if (this.startupTimer != null) clearTimeout(this.startupTimer);
      this.startupTimer = null;
      this.ready = true;
      this.startNext();
      return;
    }
    if (reply.kind === "prepared") return;
    if (reply.kind === "released") return;
    if (reply.kind === "progress") return;
    const job = this.active;
    if (!job || reply.requestId !== job.requestId) return;
    if (reply.kind === "admission") {
      this.ownership.commit(reply.requestId);
      this.datasetLoaded = true;
      const { admission } = reply;
      const resources = visualizationResourceGovernor.reserve({
        owner: `vtk:contour-service:${this.datasetId}:output`,
        generation: `request:${reply.requestId}`,
        priority: "visible",
        claims: [
          visualizationReservationClaim(
            `vtk:contour-service:${this.datasetId}:${reply.requestId}:geometry`,
            "retained-geometry",
            admission.maximumOutputBytes,
          ),
          visualizationReservationClaim(
            `vtk:contour-service:${this.datasetId}:${reply.requestId}:scratch`,
            "worker-scratch",
            admission.maximumScratchBytes,
          ),
        ],
      });
      if (!resources.ok) {
        this.active = null;
        this.finish(job, new Error(resources.reason));
        this.startNext();
        return;
      }
      job.outputReservationId = resources.reservation.id;
      job.maximumOutputBytes = admission.maximumOutputBytes;
      job.maximumTriangles = admission.maximumTriangles;
      job.maximumScratchBytes = admission.maximumScratchBytes;
      this.worker.postMessage({ kind: "continue", requestId: reply.requestId,
        datasetId: reply.datasetId, admission });
      return;
    }
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    this.executionTimer = null;
    this.active = null;
    this.ownership.commit(reply.requestId);
    if (reply.ok) {
      this.datasetLoaded = true;
      const buffers = [...new Set(vortexContourTransferables(reply.result))];
      const bytes = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
      const violation = contourUsageViolation(reply.result.usage, "Contour worker output",
        job.maximumOutputBytes, job.maximumTriangles);
      if (violation || bytes > job.maximumOutputBytes
          || reply.result.knownScratchBytes > job.maximumScratchBytes
          || job.outputReservationId == null) {
        this.finish(job, new Error(violation
          ?? "Contour worker exceeded its preflight admission"));
        this.startNext();
        return;
      }
      const retained = visualizationResourceGovernor.transition(
        job.outputReservationId,
        buffers.map((buffer) => visualizationBufferClaim(buffer, "retained-geometry")),
      );
      if (!retained.ok) {
        this.finish(job, new Error(retained.reason));
        this.startNext();
        return;
      }
      visualizationResourceGovernor.recordPrediction(
        job.maximumOutputBytes + job.maximumScratchBytes,
        bytes + reply.result.knownScratchBytes,
      );
      if (this.completedOutputReservationId != null) {
        visualizationResourceGovernor.release(this.completedOutputReservationId);
      }
      this.completedOutputReservationId = job.outputReservationId;
      job.outputReservationId = null;
      this.finish(job, reply.result);
    } else {
      this.finish(job, new Error(reply.error));
    }
    this.startNext();
  }

  private startNext(): void {
    if (!this.worker || !this.ready || this.active) return;
    let job: Job | undefined;
    while ((job = this.queue.shift())) {
      if (!job.settled && job.consumers > 0) break;
      job = undefined;
    }
    if (!job) {
      this.scheduleRetirement();
      return;
    }
    this.active = job;
    job.attempts += 1;
    job.requestId = ++this.requestSequence;
    let imageHandoff = null as ReturnType<typeof handoffVtkExtractionDataset> | null;
    let condensateHandoff = null as ReturnType<typeof handoffVtkExtractionDataset> | null;
    if (!this.datasetLoaded) {
      try {
        const copyBytes = (isSharedVtkDataset(this.prepared.values) && !this.prepared.validity
          ? 0
          : this.prepared.values.byteLength) + (
          !this.condensatePrepared || (isSharedVtkDataset(this.condensatePrepared.values) && !this.condensatePrepared.validity)
            ? 0
            : this.condensatePrepared.values.byteLength
        );
        this.ownership.reserveCopy(job.requestId, copyBytes);
        imageHandoff = handoffVtkExtractionDataset(this.prepared.values, "contour-scalar-grid", this.prepared.validity);
        condensateHandoff = this.condensatePrepared
          ? handoffVtkExtractionDataset(this.condensatePrepared.values, "contour-condensate-grid", this.condensatePrepared.validity)
          : null;
        this.ownership.materialize(job.requestId, [
          { buffer: imageHandoff.values.buffer, sharedWithMain: imageHandoff.transfer.length === 0 },
          ...(condensateHandoff ? [{
            buffer: condensateHandoff.values.buffer,
            sharedWithMain: condensateHandoff.transfer.length === 0,
          }] : []),
        ]);
      } catch (error) {
        this.ownership.discard(job.requestId);
        this.active = null;
        this.finish(job, error instanceof Error ? error : new Error(String(error)));
        this.startNext();
        return;
      }
    }
    const message: VortexContourWorkerRequest = {
      kind: "extract",
      requestId: job.requestId,
      datasetId: this.datasetId,
      thresholds: job.parameters.thresholds,
      anchor: job.parameters.anchor,
      condensateFloor: job.parameters.condensateFloor,
      ...(imageHandoff ? {
        image: {
          shape: this.prepared.shape,
          spacing: this.prepared.spacing,
          origin: this.prepared.origin,
          values: imageHandoff.values,
        },
        ...(this.condensatePrepared && condensateHandoff ? {
          condensateImage: {
            shape: this.condensatePrepared.shape,
            spacing: this.condensatePrepared.spacing,
            origin: this.condensatePrepared.origin,
            values: condensateHandoff.values,
          },
        } : {}),
      } : {}),
    };
    const transfer: Transferable[] = [
      ...(imageHandoff?.transfer ?? []),
      ...(condensateHandoff?.transfer ?? []),
    ];
    try {
      this.worker.postMessage(message, transfer);
      if (imageHandoff) recordVtkDatasetHandoff(imageHandoff.report);
      if (condensateHandoff) recordVtkDatasetHandoff(condensateHandoff.report);
      if (!imageHandoff) recordWarmVtkDatasetReuse("contour-scalar-grid");
    } catch (error) {
      this.ownership.discard(job.requestId);
      this.workerFailed(this.worker, error instanceof Error ? error.message : String(error));
      return;
    }
    const bytes = this.prepared.values.byteLength + (this.condensatePrepared?.values.byteLength ?? 0);
    const deadline = Math.min(180_000, Math.max(30_000, 15_000 + bytes / (2 * 1024 * 1024) * 1_000));
    this.executionTimer = setTimeout(() => {
      if (this.active === job && this.worker) {
        this.workerFailed(this.worker, `Vortex extraction exceeded its ${Math.ceil(deadline / 1_000)} s memory-safe deadline`);
      }
    }, deadline);
  }

  private workerFailed(worker: VortexContourExtractionWorker, detail: string): void {
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
    this.retireTimer = setTimeout(() => {
      this.retireTimer = null;
      if (!this.active && !this.queue.length) this.disposeWorker();
    }, CONTOUR_WORKER_IDLE_RETIRE_MS);
  }

  private disposeWorker(): void {
    vtkExtractionServicePool.remove(this);
    const worker = this.worker;
    this.worker = null;
    this.ready = false;
    this.datasetLoaded = false;
    for (const timer of [this.startupTimer, this.executionTimer, this.retireTimer]) {
      if (timer != null) clearTimeout(timer);
    }
    this.startupTimer = this.executionTimer = this.retireTimer = null;
    this.ownership.reset();
    if (!worker) return;
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    worker.terminate();
  }
}

const noCondensate = {};
const services = new WeakMap<PreparedVtkImage, WeakMap<object, SharedVortexContourDatasetService>>();

function serviceFor(
  prepared: PreparedVtkImage,
  condensatePrepared: PreparedVtkImage | null,
): SharedVortexContourDatasetService {
  let byCondensate = services.get(prepared);
  if (!byCondensate) {
    byCondensate = new WeakMap();
    services.set(prepared, byCondensate);
  }
  const condensateKey = condensatePrepared ?? noCondensate;
  let service = byCondensate.get(condensateKey);
  if (!service) {
    service = new SharedVortexContourDatasetService(prepared, condensatePrepared);
    byCondensate.set(condensateKey, service);
  }
  return service;
}

export function extractVortexContoursInSharedWorker(
  prepared: PreparedVtkImage,
  condensatePrepared: PreparedVtkImage | null,
  parameters: VortexContourRequestParameters,
  signal?: AbortSignal,
): Promise<VortexContourResult> {
  return serviceFor(prepared, condensatePrepared).enqueue(parameters, signal);
}
