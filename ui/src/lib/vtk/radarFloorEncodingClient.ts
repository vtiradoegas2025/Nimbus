import type { ScalarPlanePolicy } from "@/lib/plot/scalarPlanePolicy";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import radarFloorEncodingWorkerUrl from "./radarFloorEncoding.worker.ts?worker&url";
import type {
  RadarFloorEncodingResult,
  RadarFloorEncodingWorkerReply,
  RadarFloorEncodingWorkerRequest,
} from "./radarFloorEncodingProtocol";

export interface RadarFloorEncodingWorker {
  onmessage: ((event: MessageEvent<RadarFloorEncodingWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: RadarFloorEncodingWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

export interface RetainedRadarFloorEncoding extends RadarFloorEncodingResult {
  release: () => void;
}

export type RadarFloorEncodingWorkerFactory = () => RadarFloorEncodingWorker;

interface Job {
  requestId: number;
  values: Float32Array;
  shape: [number, number];
  policy: ScalarPlanePolicy;
  reservationId: number;
  attempts: number;
  settled: boolean;
  signal: AbortSignal | undefined;
  onAbort: () => void;
  resolve: (result: RetainedRadarFloorEncoding) => void;
  reject: (error: unknown) => void;
}

const IDLE_RETIRE_MS = 15_000;
let workerGeneration = 0;
let serviceSequence = 0;

function defaultWorkerFactory(): RadarFloorEncodingWorker {
  const separator = radarFloorEncodingWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${radarFloorEncodingWorkerUrl}${separator}generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-radar-floor-encoding",
  }) as unknown as RadarFloorEncodingWorker;
}

function abortError(): DOMException {
  return new DOMException("Radar floor encoding cancelled", "AbortError");
}

export class RadarFloorEncodingService {
  private readonly serviceId = ++serviceSequence;
  private readonly workerFactory: RadarFloorEncodingWorkerFactory;
  private worker: RadarFloorEncodingWorker | null = null;
  private ready = false;
  private active: Job | null = null;
  private requestSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(workerFactory: RadarFloorEncodingWorkerFactory = defaultWorkerFactory) {
    this.workerFactory = workerFactory;
  }

  encode(
    values: Float32Array,
    shape: [number, number],
    policy: ScalarPlanePolicy,
    signal?: AbortSignal,
  ): Promise<RetainedRadarFloorEncoding> {
    if (signal?.aborted) return Promise.reject(abortError());
    const cells = shape[0] * shape[1];
    const outputBytes = cells * 4;
    if (!Number.isSafeInteger(cells) || cells < 1 || values.length !== cells
        || !Number.isSafeInteger(outputBytes)) {
      return Promise.reject(new Error("Radar floor encoding shape is invalid"));
    }
    if (this.active) this.cancel(this.active);
    const requestId = ++this.requestSequence;
    const reserved = visualizationResourceGovernor.reserve({
      owner: `radar-floor-encoding:${this.serviceId}`,
      generation: `request:${requestId}`,
      priority: "visible",
      claims: [
        visualizationReservationClaim(
          `radar-floor-encoding:${this.serviceId}:${requestId}:input`,
          "worker-resident",
          values.byteLength,
        ),
        visualizationReservationClaim(
          `radar-floor-encoding:${this.serviceId}:${requestId}:output`,
          "prepared-main",
          outputBytes,
        ),
      ],
    });
    if (!reserved.ok) return Promise.reject(new Error(reserved.reason));
    return new Promise<RetainedRadarFloorEncoding>((resolve, reject) => {
      const job: Job = {
        requestId,
        values,
        shape,
        policy,
        reservationId: reserved.reservation.id,
        attempts: 0,
        settled: false,
        signal,
        onAbort: () => undefined,
        resolve,
        reject,
      };
      job.onAbort = () => this.cancel(job);
      signal?.addEventListener("abort", job.onAbort, { once: true });
      this.active = job;
      this.ensureWorker();
      this.dispatch();
    });
  }

  shutdown(): void {
    if (this.active) this.finishError(this.active, abortError());
    this.active = null;
    this.disposeWorker();
  }

  private cancel(job: Job): void {
    if (job.settled || this.active !== job) return;
    this.active = null;
    this.finishError(job, abortError());
    this.disposeWorker();
  }

  private finishError(job: Job, error: Error | DOMException): void {
    if (job.settled) return;
    job.settled = true;
    job.signal?.removeEventListener("abort", job.onAbort);
    visualizationResourceGovernor.release(job.reservationId);
    job.reject(error);
  }

  private ensureWorker(): void {
    if (this.retireTimer != null) clearTimeout(this.retireTimer);
    this.retireTimer = null;
    if (this.worker) return;
    const worker = this.workerFactory();
    this.worker = worker;
    this.ready = false;
    worker.onmessage = (event) => this.handleMessage(worker, event.data);
    worker.onerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, event.message || "Radar floor encoding worker failed");
    };
    worker.onmessageerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, "Radar floor encoding worker returned an unreadable result");
    };
    this.startupTimer = setTimeout(() => {
      if (worker === this.worker && !this.ready) {
        this.workerFailed(worker, "Radar floor encoding worker did not start");
      }
    }, 3_000);
  }

  private dispatch(): void {
    const job = this.active;
    if (!this.worker || !this.ready || !job || job.settled) return;
    job.attempts += 1;
    const values = job.values.slice();
    try {
      this.worker.postMessage({
        kind: "encode",
        requestId: job.requestId,
        values,
        shape: job.shape,
        policy: job.policy,
      }, [values.buffer]);
    } catch (error) {
      this.workerFailed(this.worker, error instanceof Error ? error.message : String(error));
      return;
    }
    this.executionTimer = setTimeout(() => {
      if (this.worker && this.active === job) {
        this.workerFailed(this.worker, "Radar floor encoding exceeded its 30 s deadline");
      }
    }, 30_000);
  }

  private handleMessage(
    worker: RadarFloorEncodingWorker,
    reply: RadarFloorEncodingWorkerReply,
  ): void {
    if (worker !== this.worker) return;
    if (reply.kind === "ready") {
      if (this.startupTimer != null) clearTimeout(this.startupTimer);
      this.startupTimer = null;
      this.ready = true;
      this.dispatch();
      return;
    }
    const job = this.active;
    if (!job || reply.requestId !== job.requestId) return;
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    this.executionTimer = null;
    this.active = null;
    if (!reply.ok) {
      this.finishError(job, new Error(reply.error));
      this.scheduleRetirement();
      return;
    }
    const expectedBytes = job.shape[0] * job.shape[1] * 4;
    if (reply.result.rgba.byteLength !== expectedBytes
        || reply.result.paintedCells < 0
        || reply.result.paintedCells > job.shape[0] * job.shape[1]) {
      this.finishError(job, new Error("Radar floor worker returned invalid encoded output"));
      this.scheduleRetirement();
      return;
    }
    const transitioned = visualizationResourceGovernor.transition(job.reservationId, [
      visualizationBufferClaim(reply.result.rgba.buffer, "prepared-main"),
    ]);
    if (!transitioned.ok) {
      this.finishError(job, new Error(transitioned.reason));
      this.scheduleRetirement();
      return;
    }
    visualizationResourceGovernor.recordPrediction(
      job.values.byteLength + expectedBytes,
      job.values.byteLength + reply.result.rgba.byteLength,
    );
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      visualizationResourceGovernor.release(job.reservationId);
    };
    job.settled = true;
    job.signal?.removeEventListener("abort", job.onAbort);
    job.resolve({ ...reply.result, release });
    this.scheduleRetirement();
  }

  private workerFailed(worker: RadarFloorEncodingWorker, detail: string): void {
    if (worker !== this.worker) return;
    const active = this.active;
    this.disposeWorker();
    if (!active || active.settled) return;
    if (!active.signal?.aborted && active.attempts < 2) {
      this.active = active;
      setTimeout(() => {
        if (this.active !== active || active.settled || active.signal?.aborted) return;
        this.ensureWorker();
        this.dispatch();
      }, 25);
    } else {
      this.active = null;
      this.finishError(active, new Error(detail));
    }
  }

  private scheduleRetirement(): void {
    if (!this.worker || this.active || this.retireTimer != null) return;
    this.retireTimer = setTimeout(() => {
      this.retireTimer = null;
      if (!this.active) this.disposeWorker();
    }, IDLE_RETIRE_MS);
  }

  private disposeWorker(): void {
    const worker = this.worker;
    this.worker = null;
    this.ready = false;
    for (const timer of [this.startupTimer, this.executionTimer, this.retireTimer]) {
      if (timer != null) clearTimeout(timer);
    }
    this.startupTimer = this.executionTimer = this.retireTimer = null;
    if (!worker) return;
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    worker.terminate();
  }
}
