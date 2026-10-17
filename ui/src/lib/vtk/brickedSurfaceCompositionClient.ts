import workerUrl from "./brickedSurfaceComposition.worker.ts?worker&url";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import type { OwnedBrickSurface } from "./vtkBrickedContour";
import type {
  BrickedSurfaceCompositionWorkerReply,
  BrickedSurfaceCompositionWorkerRequest,
} from "./brickedSurfaceCompositionProtocol";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

interface WorkerLike {
  onmessage: ((event: MessageEvent<BrickedSurfaceCompositionWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage: (request: BrickedSurfaceCompositionWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

interface Pending {
  requestId: number;
  identity: string;
  retentionKey: string;
  reservationId: number;
  maximumOutputBytes: number;
  resolve: (geometry: VortexSurfaceGeometry | null) => void;
  reject: (error: Error) => void;
}

let sequence = 0;

export class BrickedSurfaceCompositionClient {
  private readonly owner = `bricked-surface-composition:${++sequence}`;
  private readonly worker: WorkerLike;
  private readonly ready: Promise<void>;
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private pending: Pending | null = null;
  private readonly retained = new Map<string, number>();
  private requestSequence = 0;
  private closed = false;

  constructor(worker?: WorkerLike) {
    const url = `${workerUrl}${workerUrl.includes("?") ? "&" : "?"}generation=${sequence}`;
    this.worker = worker ?? new Worker(url, {
      type: "module",
      name: "nimbus-bricked-surface-composition",
    }) as unknown as WorkerLike;
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.worker.onmessage = (event) => this.message(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault?.();
      this.fail(new Error(event.message || "Bricked surface composition worker failed"));
    };
  }

  isClosed(): boolean {
    return this.closed;
  }

  async compose(
    identity: string,
    retentionKey: string,
    surfaces: OwnedBrickSurface[],
    coordinateTolerance: number,
    signal: AbortSignal,
  ): Promise<VortexSurfaceGeometry | null> {
    await this.ready;
    if (this.closed) throw new Error("Bricked surface composition client is closed");
    if (this.pending) throw new Error("Bricked surface composition already has an active request");
    if (signal.aborted) throw new DOMException("Bricked surface composition cancelled", "AbortError");
    const inputBytes = surfaces.reduce((sum, surface) => sum + surface.geometry.bytes, 0);
    if (inputBytes === 0) {
      this.release(retentionKey);
      return null;
    }
    const requestId = ++this.requestSequence;
    const admitted = visualizationResourceGovernor.reserve({
      owner: this.owner,
      generation: identity,
      priority: "visible",
      claims: [
        visualizationReservationClaim(
          `${this.owner}:${requestId}:scratch`,
          "worker-scratch",
          inputBytes * 2,
        ),
        visualizationReservationClaim(
          `${this.owner}:${requestId}:output`,
          "retained-geometry",
          inputBytes,
        ),
      ],
    });
    if (!admitted.ok) throw new Error(admitted.reason);
    const promise = new Promise<VortexSurfaceGeometry | null>((resolve, reject) => {
      this.pending = {
        requestId,
        identity,
        retentionKey,
        reservationId: admitted.reservation.id,
        maximumOutputBytes: inputBytes,
        resolve,
        reject,
      };
    });
    const abort = () => {
      if (this.pending?.requestId !== requestId) return;
      const pending = this.pending;
      this.pending = null;
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new DOMException("Bricked surface composition cancelled", "AbortError"));
      this.shutdown();
    };
    signal.addEventListener("abort", abort, { once: true });
    const transfer = surfaces.flatMap((surface) => [
      surface.geometry.points.buffer,
      surface.geometry.polys.buffer,
      surface.geometry.normals.buffer,
    ]);
    this.worker.postMessage({
      kind: "compose",
      requestId,
      identity,
      surfaces,
      coordinateTolerance,
    }, [...new Set(transfer)] as Transferable[]);
    return promise.finally(() => signal.removeEventListener("abort", abort));
  }

  release(retentionKey: string): void {
    const reservation = this.retained.get(retentionKey);
    if (reservation == null) return;
    visualizationResourceGovernor.release(reservation);
    this.retained.delete(retentionKey);
  }

  shutdown(): void {
    if (this.closed) return;
    this.closed = true;
    this.fail(new DOMException("Bricked surface composition client closed", "AbortError"));
    for (const reservation of this.retained.values()) {
      visualizationResourceGovernor.release(reservation);
    }
    this.retained.clear();
    this.worker.postMessage({ kind: "close" });
    this.worker.onmessage = this.worker.onerror = null;
    this.worker.terminate();
  }

  private message(reply: BrickedSurfaceCompositionWorkerReply): void {
    if (reply.kind === "ready") {
      this.resolveReady();
      return;
    }
    const pending = this.pending;
    if (!pending || reply.requestId !== pending.requestId || reply.identity !== pending.identity) return;
    this.pending = null;
    if (!reply.ok) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(reply.error));
      return;
    }
    const geometry = reply.geometry;
    if ((geometry?.bytes ?? 0) > pending.maximumOutputBytes) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error("Bricked surface composition exceeded its input-topology bound"));
      return;
    }
    const transitioned = visualizationResourceGovernor.transition(
      pending.reservationId,
      geometry ? [geometry.points, geometry.polys, geometry.normals]
        .map((array) => visualizationBufferClaim(array.buffer, "retained-geometry")) : [],
    );
    if (!transitioned.ok) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(transitioned.reason));
      return;
    }
    const previous = this.retained.get(pending.retentionKey);
    if (previous != null) visualizationResourceGovernor.release(previous);
    if (geometry) this.retained.set(pending.retentionKey, pending.reservationId);
    else visualizationResourceGovernor.release(pending.reservationId);
    visualizationResourceGovernor.recordPrediction(
      pending.maximumOutputBytes,
      geometry?.bytes ?? 0,
    );
    pending.resolve(geometry);
  }

  private fail(error: Error): void {
    if (!this.pending) {
      this.rejectReady(error);
      return;
    }
    const pending = this.pending;
    this.pending = null;
    visualizationResourceGovernor.release(pending.reservationId);
    pending.reject(error);
  }
}
