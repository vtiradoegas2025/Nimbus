import type { FloorReduction } from "@/lib/plot/radarFloor";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import radarFloorPreparationWorkerUrl from "./radarFloorPreparation.worker.ts?worker&url";
import {
  radarFloorPreparationBuffers,
  type RadarFloorPreparationResult,
  type RadarFloorPreparationWorkerReply,
  type RadarFloorPreparationWorkerRequest,
  type RadarFloorTerrainInput,
} from "./radarFloorPreparationProtocol";

export interface RadarFloorPreparationWorker {
  onmessage: ((event: MessageEvent<RadarFloorPreparationWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: RadarFloorPreparationWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

export interface RetainedRadarFloorPreparation extends RadarFloorPreparationResult {
  release: () => void;
}

export type RadarFloorPreparationWorkerFactory = () => RadarFloorPreparationWorker;

interface Job {
  requestId: number;
  values: Float32Array;
  shape: [number, number, number];
  reduction: FloorReduction;
  terrain: RadarFloorTerrainInput | null;
  inputBytes: number;
  outputBytes: number;
  scratchBytes: number;
  reservationId: number;
  attempts: number;
  settled: boolean;
  signal: AbortSignal | undefined;
  onAbort: () => void;
  resolve: (result: RetainedRadarFloorPreparation) => void;
  reject: (error: unknown) => void;
}

const IDLE_RETIRE_MS = 15_000;
let workerGeneration = 0;
let serviceSequence = 0;

function defaultWorkerFactory(): RadarFloorPreparationWorker {
  const separator = radarFloorPreparationWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${radarFloorPreparationWorkerUrl}${separator}generation=${++workerGeneration}`, {
    type: "module",
    name: "nimbus-radar-floor-preparation",
  }) as unknown as RadarFloorPreparationWorker;
}

function abortError(): DOMException {
  return new DOMException("Radar floor preparation cancelled", "AbortError");
}

function terrainInputBytes(terrain: RadarFloorTerrainInput | null): number {
  if (!terrain) return 0;
  return terrain.centroids.x.byteLength + terrain.centroids.y.byteLength
    + terrain.centroids.z.byteLength + terrain.volumesM3.byteLength;
}

function copiedTerrain(terrain: RadarFloorTerrainInput | null): RadarFloorTerrainInput | null {
  if (!terrain) return null;
  return {
    ...terrain,
    centroids: {
      x: terrain.centroids.x.slice(),
      y: terrain.centroids.y.slice(),
      z: terrain.centroids.z.slice(),
    },
    volumesM3: terrain.volumesM3.slice(),
  };
}

export class RadarFloorPreparationService {
  private readonly serviceId = ++serviceSequence;
  private readonly workerFactory: RadarFloorPreparationWorkerFactory;
  private worker: RadarFloorPreparationWorker | null = null;
  private ready = false;
  private active: Job | null = null;
  private requestSequence = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private retireTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(workerFactory: RadarFloorPreparationWorkerFactory = defaultWorkerFactory) {
    this.workerFactory = workerFactory;
  }

  prepare(
    values: Float32Array,
    shape: [number, number, number],
    reduction: FloorReduction,
    terrain: RadarFloorTerrainInput | null,
    signal?: AbortSignal,
  ): Promise<RetainedRadarFloorPreparation> {
    if (signal?.aborted) return Promise.reject(abortError());
    const cells = shape[0] * shape[1];
    const sourceElements = cells * shape[2];
    if (!Number.isSafeInteger(sourceElements) || sourceElements < 1
        || sourceElements !== values.length) {
      return Promise.reject(new Error("Radar floor preparation shape is invalid"));
    }
    if (this.active) this.cancel(this.active);
    const requestId = ++this.requestSequence;
    const inputBytes = values.byteLength + terrainInputBytes(terrain);
    const meshBytes = terrain
      ? cells * (3 * Float32Array.BYTES_PER_ELEMENT + 2 * Float32Array.BYTES_PER_ELEMENT)
        + (shape[0] - 1) * (shape[1] - 1) * 8 * Uint32Array.BYTES_PER_ELEMENT
      : 0;
    const outputBytes = cells * Float32Array.BYTES_PER_ELEMENT + meshBytes;
    const scratchBytes = cells * (Float32Array.BYTES_PER_ELEMENT
      + (terrain ? Float64Array.BYTES_PER_ELEMENT : 0));
    const reserved = visualizationResourceGovernor.reserve({
      owner: `radar-floor-preparation:${this.serviceId}`,
      generation: `request:${requestId}`,
      priority: "visible",
      claims: [
        visualizationReservationClaim(
          `radar-floor-preparation:${this.serviceId}:${requestId}:input`,
          "worker-resident",
          inputBytes,
        ),
        visualizationReservationClaim(
          `radar-floor-preparation:${this.serviceId}:${requestId}:output`,
          "prepared-main",
          outputBytes,
        ),
        visualizationReservationClaim(
          `radar-floor-preparation:${this.serviceId}:${requestId}:scratch`,
          "worker-scratch",
          scratchBytes,
        ),
      ],
    });
    if (!reserved.ok) return Promise.reject(new Error(reserved.reason));
    return new Promise<RetainedRadarFloorPreparation>((resolve, reject) => {
      const job: Job = {
        requestId,
        values,
        shape,
        reduction,
        terrain,
        inputBytes,
        outputBytes,
        scratchBytes,
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
      this.workerFailed(worker, event.message || "Radar floor preparation worker failed");
    };
    worker.onmessageerror = (event) => {
      event.preventDefault?.();
      this.workerFailed(worker, "Radar floor preparation worker returned an unreadable result");
    };
    this.startupTimer = setTimeout(() => {
      if (worker === this.worker && !this.ready) {
        this.workerFailed(worker, "Radar floor preparation worker did not start");
      }
    }, 3_000);
  }

  private dispatch(): void {
    const job = this.active;
    if (!this.worker || !this.ready || !job || job.settled) return;
    job.attempts += 1;
    const values = job.values.slice();
    const terrain = copiedTerrain(job.terrain);
    const transfer = [
      values.buffer,
      ...(terrain ? [
        terrain.centroids.x.buffer,
        terrain.centroids.y.buffer,
        terrain.centroids.z.buffer,
        terrain.volumesM3.buffer,
      ] : []),
    ] as ArrayBuffer[];
    try {
      this.worker.postMessage({
        kind: "prepare",
        requestId: job.requestId,
        values,
        shape: job.shape,
        reduction: job.reduction,
        terrain,
      }, transfer);
    } catch (error) {
      this.workerFailed(this.worker, error instanceof Error ? error.message : String(error));
      return;
    }
    const deadline = Math.min(
      180_000,
      Math.max(30_000, 15_000 + job.inputBytes / (2 * 1024 * 1024) * 1_000),
    );
    this.executionTimer = setTimeout(() => {
      if (this.worker && this.active === job) {
        this.workerFailed(this.worker, `Radar floor preparation exceeded its ${Math.ceil(deadline / 1_000)} s deadline`);
      }
    }, deadline);
  }

  private handleMessage(
    worker: RadarFloorPreparationWorker,
    reply: RadarFloorPreparationWorkerReply,
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
    const cells = job.shape[0] * job.shape[1];
    const terrain = reply.result.terrainMesh;
    const valid = (reply.result.values == null && reply.result.stats == null && terrain == null)
      || (reply.result.values?.length === cells && reply.result.stats != null
        && ((!job.terrain && terrain == null) || (job.terrain != null && terrain != null
          && terrain.points.length === cells * 3 && terrain.tcoords.length === cells * 2
          && terrain.polys.length === (job.shape[0] - 1) * (job.shape[1] - 1) * 8)));
    if (!valid || reply.result.knownScratchBytes > job.scratchBytes) {
      this.finishError(job, new Error("Radar floor worker returned output outside its admitted contract"));
      this.scheduleRetirement();
      return;
    }
    const buffers = radarFloorPreparationBuffers(reply.result);
    const transitioned = visualizationResourceGovernor.transition(
      job.reservationId,
      buffers.map((buffer) => visualizationBufferClaim(buffer, "prepared-main")),
    );
    if (!transitioned.ok) {
      this.finishError(job, new Error(transitioned.reason));
      this.scheduleRetirement();
      return;
    }
    const actualOutputBytes = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
    visualizationResourceGovernor.recordPrediction(
      job.inputBytes + job.outputBytes + job.scratchBytes,
      job.inputBytes + actualOutputBytes + reply.result.knownScratchBytes,
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

  private workerFailed(worker: RadarFloorPreparationWorker, detail: string): void {
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
