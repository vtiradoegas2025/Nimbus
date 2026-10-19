import type { NpyN } from "@/lib/platform/npy";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import {
  handoffVtkExtractionDataset,
  isSharedVtkDataset,
  recordWarmVtkDatasetReuse,
  recordVtkDatasetHandoff,
} from "./vtkDatasetHandoff";
import { vtkExtractionServicePool } from "./vtkExtractionServicePool";
import { VtkWorkerDatasetOwnership } from "./vtkWorkerDatasetOwnership";
import trajectoryGeometryWorkerUrl from "./trajectoryGeometry.worker.ts?worker&url";
import {
  trajectoryGeometryBuffers,
  type TrajectoryGeometryParameters,
  type TrajectoryGeometryResult,
  type TrajectoryGeometryWorkerReply,
  type TrajectoryGeometryWorkerRequest,
} from "./trajectoryGeometryProtocol";

export interface TrajectoryGeometryWorker {
  onmessage: ((event: MessageEvent<TrajectoryGeometryWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: TrajectoryGeometryWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

export type TrajectoryGeometryWorkerFactory = () => TrajectoryGeometryWorker;

export interface RetainedTrajectoryGeometry extends TrajectoryGeometryResult {
  release: () => void;
}

interface Job {
  trajectory: NpyN;
  parameters: TrajectoryGeometryParameters;
  datasetId: number;
  requestId: number;
  attempts: number;
  maximumOutputBytes: number;
  maximumScratchBytes: number;
  outputReservationId: number;
  settled: boolean;
  signal: AbortSignal | undefined;
  onAbort: () => void;
  resolve: (result: RetainedTrajectoryGeometry) => void;
  reject: (error: unknown) => void;
}

const IDLE_RETIRE_MS = 15_000;
let workerGeneration = 0;
let serviceSequence = 0;

function defaultWorkerFactory(): TrajectoryGeometryWorker {
  const separator = trajectoryGeometryWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${trajectoryGeometryWorkerUrl}${separator}generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-trajectory-geometry",
  }) as unknown as TrajectoryGeometryWorker;
}

function abortError(): DOMException {
  return new DOMException("Trajectory geometry preparation cancelled", "AbortError");
}

function shapeOf(trajectory: NpyN): [number, number, number] {
  if (trajectory.shape.length !== 3) throw new Error("Trajectory data must have rank three");
  return [trajectory.shape[0]!, trajectory.shape[1]!, trajectory.shape[2]!];
}

export class TrajectoryGeometryService {
  private readonly serviceId = ++serviceSequence;
  private readonly workerFactory: TrajectoryGeometryWorkerFactory;
  private readonly ownership: VtkWorkerDatasetOwnership;
  private readonly datasetIds = new WeakMap<object, Map<string, number>>();
  private readonly queue: Job[] = [];
  private worker: TrajectoryGeometryWorker | null = null;
  private ready = false;
  private active: Job | null = null;
  private loadedDatasetId = -1;
  private datasetSequence = 0;
  private requestSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(workerFactory: TrajectoryGeometryWorkerFactory = defaultWorkerFactory) {
    this.workerFactory = workerFactory;
    this.ownership = new VtkWorkerDatasetOwnership(
      `vtk:trajectory-service:${this.serviceId}:dataset`,
    );
  }

  extract(
    trajectory: NpyN,
    parameters: TrajectoryGeometryParameters,
    maximumOutputBytes: number,
    maximumScratchBytes: number,
    signal?: AbortSignal,
  ): Promise<RetainedTrajectoryGeometry> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (!Number.isSafeInteger(maximumOutputBytes) || maximumOutputBytes < 1
        || !Number.isSafeInteger(maximumScratchBytes) || maximumScratchBytes < 1) {
      return Promise.reject(new Error("Trajectory output and scratch admission must be positive"));
    }
    const requestId = ++this.requestSequence;
    const output = visualizationResourceGovernor.reserve({
      owner: `vtk:trajectory-service:${this.serviceId}:output`,
      generation: `request:${requestId}`,
      priority: "visible",
      claims: [visualizationReservationClaim(
        `vtk:trajectory-service:${this.serviceId}:request:${requestId}:output`,
        "retained-geometry",
        maximumOutputBytes,
      ), visualizationReservationClaim(
        `vtk:trajectory-service:${this.serviceId}:request:${requestId}:scratch`,
        "worker-scratch",
        maximumScratchBytes,
      )],
    });
    if (!output.ok) return Promise.reject(new Error(output.reason));

    let datasets = this.datasetIds.get(trajectory.data.buffer);
    if (!datasets) {
      datasets = new Map();
      this.datasetIds.set(trajectory.data.buffer, datasets);
    }
    const datasetKey = `${trajectory.data.byteOffset}:${trajectory.data.byteLength}:${trajectory.shape.join(":")}`;
    let datasetId = datasets.get(datasetKey);
    if (datasetId == null) {
      datasetId = ++this.datasetSequence;
      datasets.set(datasetKey, datasetId);
    }
    return new Promise<RetainedTrajectoryGeometry>((resolve, reject) => {
      const job: Job = {
        trajectory,
        parameters,
        datasetId,
        requestId,
        attempts: 0,
        maximumOutputBytes,
        maximumScratchBytes,
        outputReservationId: output.reservation.id,
        settled: false,
        signal,
        onAbort: () => undefined,
        resolve,
        reject,
      };
      job.onAbort = () => this.cancel(job);
      signal?.addEventListener("abort", job.onAbort, { once: true });
      this.queue.push(job);
      this.ensureWorker();
      this.startNext();
    });
  }

  shutdown(): void {
    if (this.active) this.finishError(this.active, abortError());
    this.active = null;
    for (const job of this.queue.splice(0)) this.finishError(job, abortError());
    this.disposeWorker();
  }

  private cancel(job: Job): void {
    if (job.settled) return;
    const queued = this.queue.indexOf(job);
    if (queued >= 0) {
      this.queue.splice(queued, 1);
      this.finishError(job, abortError());
      return;
    }
    if (this.active === job) {
      this.active = null;
      this.ownership.discard(job.requestId);
      this.finishError(job, abortError());
      this.disposeWorker();
      this.restartForQueue();
    }
  }

  private finishError(job: Job, error: Error | DOMException): void {
    if (job.settled) return;
    job.settled = true;
    job.signal?.removeEventListener("abort", job.onAbort);
    visualizationResourceGovernor.release(job.outputReservationId);
    job.reject(error);
  }

  private finishResult(job: Job, result: TrajectoryGeometryResult): void {
    if (job.settled) return;
    const buffers = trajectoryGeometryBuffers(result.geometry);
    const actualBytes = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
    if (actualBytes !== result.retainedBytes || actualBytes > job.maximumOutputBytes) {
      this.finishError(job, new Error(
        `Trajectory geometry exceeded its admitted output (${actualBytes}/${job.maximumOutputBytes} bytes)`,
      ));
      return;
    }
    const transitioned = visualizationResourceGovernor.transition(
      job.outputReservationId,
      buffers.map((buffer) => visualizationBufferClaim(buffer, "retained-geometry")),
    );
    if (!transitioned.ok) {
      this.finishError(job, new Error(transitioned.reason));
      return;
    }
    if (result.knownScratchBytes > job.maximumScratchBytes) {
      this.finishError(job, new Error(
        `Trajectory geometry exceeded its admitted scratch (${result.knownScratchBytes}/${job.maximumScratchBytes} bytes)`,
      ));
      return;
    }
    visualizationResourceGovernor.recordPrediction(
      job.maximumOutputBytes + job.maximumScratchBytes,
      actualBytes + result.knownScratchBytes,
    );
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      visualizationResourceGovernor.release(job.outputReservationId);
    };
    job.settled = true;
    job.signal?.removeEventListener("abort", job.onAbort);
    job.resolve({ ...result, release });
  }

  private ensureWorker(): void {
    vtkExtractionServicePool.activate(this);
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    if (this.worker) return;
    const worker = this.workerFactory();
    this.worker = worker;
    this.ready = false;
    this.loadedDatasetId = -1;
    worker.onmessage = (event) => this.handleMessage(worker, event.data);
    worker.onerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, event.message || "Trajectory geometry worker failed");
    };
    worker.onmessageerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, "Trajectory geometry worker returned an unreadable result");
    };
    this.startupTimer = setTimeout(() => {
      if (this.worker === worker && !this.ready) {
        this.workerFailed(worker, "Trajectory geometry worker did not start");
      }
    }, 3_000);
  }

  private handleMessage(
    worker: TrajectoryGeometryWorker,
    reply: TrajectoryGeometryWorkerReply,
  ): void {
    if (worker !== this.worker) return;
    if (reply.kind === "ready") {
      if (this.startupTimer != null) clearTimeout(this.startupTimer);
      this.startupTimer = null;
      this.ready = true;
      this.startNext();
      return;
    }
    const job = this.active;
    if (!job || reply.requestId !== job.requestId) return;
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    this.executionTimer = null;
    this.active = null;
    this.ownership.commit(job.requestId);
    this.loadedDatasetId = reply.datasetId;
    if (reply.ok) {
      this.finishResult(job, reply.result);
    } else {
      this.finishError(job, new Error(reply.error));
    }
    this.startNext();
  }

  private startNext(): void {
    if (!this.worker || !this.ready || this.active) return;
    let job: Job | undefined;
    while ((job = this.queue.shift())) {
      if (!job.settled && !job.signal?.aborted) break;
      if (!job.settled) this.finishError(job, abortError());
      job = undefined;
    }
    if (!job) {
      this.scheduleRetirement();
      return;
    }
    this.active = job;
    job.attempts += 1;
    const needsHandoff = this.loadedDatasetId !== job.datasetId;
    let handoff: ReturnType<typeof handoffVtkExtractionDataset> | null = null;
    if (needsHandoff) {
      try {
        this.ownership.reserveCopy(
          job.requestId,
          isSharedVtkDataset(job.trajectory.data) ? 0 : job.trajectory.data.byteLength,
        );
        handoff = handoffVtkExtractionDataset(job.trajectory.data, "trajectory-samples");
        this.ownership.materialize(job.requestId, [{
          buffer: handoff.values.buffer,
          sharedWithMain: handoff.transfer.length === 0,
        }]);
      } catch (error) {
        this.ownership.discard(job.requestId);
        this.active = null;
        this.finishError(job, error instanceof Error ? error : new Error(String(error)));
        this.startNext();
        return;
      }
    }
    const message: TrajectoryGeometryWorkerRequest = {
      kind: "extract",
      requestId: job.requestId,
      datasetId: job.datasetId,
      ...job.parameters,
      ...(handoff ? {
        dataset: { shape: shapeOf(job.trajectory), values: handoff.values },
      } : {}),
    };
    try {
      this.worker.postMessage(message, handoff?.transfer ?? []);
      if (handoff) recordVtkDatasetHandoff(handoff.report);
      else recordWarmVtkDatasetReuse("trajectory-samples");
    } catch (error) {
      this.ownership.discard(job.requestId);
      this.workerFailed(this.worker, error instanceof Error ? error.message : String(error));
      return;
    }
    const deadline = Math.min(
      180_000,
      Math.max(30_000, 15_000 + job.trajectory.data.byteLength / (2 * 1024 * 1024) * 1_000),
    );
    this.executionTimer = setTimeout(() => {
      if (this.active === job && this.worker) {
        this.workerFailed(
          this.worker,
          `Trajectory geometry preparation exceeded its ${Math.ceil(deadline / 1_000)} s deadline`,
        );
      }
    }, deadline);
  }

  private workerFailed(worker: TrajectoryGeometryWorker, detail: string): void {
    if (worker !== this.worker) return;
    const active = this.active;
    this.active = null;
    this.disposeWorker();
    if (active && !active.settled && !active.signal?.aborted && active.attempts < 2) {
      this.queue.unshift(active);
    } else if (active && !active.settled) {
      this.finishError(active, new Error(detail));
    }
    this.restartForQueue();
  }

  private restartForQueue(): void {
    if (!this.queue.some((job) => !job.settled && !job.signal?.aborted)) return;
    setTimeout(() => {
      if (!this.queue.some((job) => !job.settled && !job.signal?.aborted)) return;
      this.ensureWorker();
      this.startNext();
    }, 25);
  }

  private scheduleRetirement(): void {
    if (!this.worker || this.active || this.queue.length || this.retireTimer != null) return;
    vtkExtractionServicePool.park(this, () => {
      if (!this.active && !this.queue.length) this.disposeWorker();
    });
    this.retireTimer = setTimeout(() => {
      this.retireTimer = null;
      if (!this.active && !this.queue.length) this.disposeWorker();
    }, IDLE_RETIRE_MS);
  }

  private disposeWorker(): void {
    vtkExtractionServicePool.remove(this);
    const worker = this.worker;
    this.worker = null;
    this.ready = false;
    this.loadedDatasetId = -1;
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
