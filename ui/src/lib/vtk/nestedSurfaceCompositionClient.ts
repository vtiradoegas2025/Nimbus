import workerUrl from "./nestedSurfaceComposition.worker.ts?worker&url";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import type {
  NestedSurfaceCompositionInput,
  NestedSurfaceCompositionWorkerReply,
  NestedSurfaceCompositionWorkerRequest,
} from "./nestedSurfaceCompositionProtocol";
import { nestedSurfaceCompositionUpperBound } from "./nestedSurfaceCompositionProtocol";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

interface WorkerLike {
  onmessage: ((event: MessageEvent<NestedSurfaceCompositionWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage: (request: NestedSurfaceCompositionWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

export interface NestedSurfaceCompositionResult {
  primary: { geometry: VortexSurfaceGeometry | null; transitionTriangles: number };
  opposite: { geometry: VortexSurfaceGeometry | null; transitionTriangles: number };
}

let sequence = 0;

export class NestedSurfaceCompositionClient {
  private readonly owner = `nested-surface-composition:${++sequence}`;
  private readonly worker: WorkerLike;
  private readonly ready: Promise<void>;
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private pending: {
    requestId: number;
    identity: string;
    reservationId: number;
    primary: NestedSurfaceCompositionInput;
    opposite: NestedSurfaceCompositionInput;
    maximumScratchBytes: number;
    maximumOutputBytes: number;
    resolve: (result: NestedSurfaceCompositionResult) => void;
    reject: (error: Error) => void;
  } | null = null;
  private retainedReservationId: number | null = null;
  private requestId = 0;
  private closed = false;

  constructor(worker?: WorkerLike) {
    const generationUrl = `${workerUrl}${workerUrl.includes("?") ? "&" : "?"}generation=${sequence}`;
    this.worker = worker ?? new Worker(generationUrl, { type: "module", name: "nimbus-nested-surface-composition" }) as unknown as WorkerLike;
    this.ready = new Promise((resolve, reject) => { this.resolveReady = resolve; this.rejectReady = reject; });
    this.worker.onmessage = (event) => this.message(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault?.();
      const location = event.filename
        ? ` (${event.filename}:${event.lineno}:${event.colno})` : "";
      const cause = event.error instanceof Error ? `: ${event.error.message}` : "";
      this.fail(new Error(`${event.message || "Nested surface composition worker failed"}${cause}${location}`));
    };
  }

  isClosed(): boolean { return this.closed; }

  async compose(identity: string, primary: NestedSurfaceCompositionInput, opposite: NestedSurfaceCompositionInput, signal: AbortSignal): Promise<NestedSurfaceCompositionResult> {
    await this.ready;
    if (this.closed) throw new Error("Nested surface composition client is closed");
    if (signal.aborted) throw new DOMException("Nested surface composition cancelled", "AbortError");
    if (this.pending) throw new Error("Nested surface composition already has an active request");
    if (primary.geometries.length < 2 || primary.geometries.length !== opposite.geometries.length
        || primary.childBounds.length !== primary.geometries.length
        || opposite.childBounds.length !== primary.geometries.length
        || JSON.stringify(primary.childBounds) !== JSON.stringify(opposite.childBounds)) {
      throw new Error("Nested surface composition inputs do not share one hierarchy");
    }
    const topology = nestedSurfaceCompositionUpperBound(primary, opposite);
    const nextRequestId = ++this.requestId;
    const admission = visualizationResourceGovernor.reserve({
      owner: this.owner, generation: identity, priority: "visible", claims: [
        visualizationReservationClaim(`${this.owner}:${nextRequestId}:scratch`, "worker-scratch", topology.scratchBytes),
        visualizationReservationClaim(`${this.owner}:${nextRequestId}:output`, "retained-geometry", topology.outputBytes),
      ],
    });
    if (!admission.ok) throw new Error(admission.reason);
    const promise = new Promise<NestedSurfaceCompositionResult>((resolve, reject) => {
      this.pending = { requestId: nextRequestId, identity, reservationId: admission.reservation.id,
        primary, opposite, maximumScratchBytes: topology.scratchBytes,
        maximumOutputBytes: topology.outputBytes, resolve, reject };
    });
    const abort = () => {
      if (this.pending?.requestId !== nextRequestId) return;
      const pending = this.pending; this.pending = null;
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new DOMException("Nested surface composition cancelled", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    this.worker.postMessage({
      kind: "begin", requestId: nextRequestId, identity,
      childBounds: primary.childBounds, layerCount: primary.geometries.length,
      maximumOutputBytes: topology.outputBytes, maximumTriangles: topology.outputTriangles,
    });
    return promise.finally(() => signal.removeEventListener("abort", abort));
  }

  shutdown(): void {
    if (this.closed) return;
    this.closed = true;
    this.fail(new DOMException("Nested surface composition client closed", "AbortError"));
    if (this.retainedReservationId != null) visualizationResourceGovernor.release(this.retainedReservationId);
    this.retainedReservationId = null;
    this.worker.postMessage({ kind: "close" });
    this.worker.onmessage = this.worker.onerror = null;
    this.worker.terminate();
  }

  private message(reply: NestedSurfaceCompositionWorkerReply): void {
    if (reply.kind === "ready") { this.resolveReady(); return; }
    const pending = this.pending;
    if (!pending || pending.requestId !== reply.requestId || pending.identity !== reply.identity) return;
    if (reply.kind === "ack") {
      if (reply.nextLayerIndex < pending.primary.geometries.length) {
        const primary = pending.primary.geometries[reply.nextLayerIndex];
        const opposite = pending.opposite.geometries[reply.nextLayerIndex];
        const transfer = [primary, opposite].flatMap((geometry) => geometry
          ? [geometry.points.buffer, geometry.polys.buffer, geometry.normals.buffer] : []);
        this.worker.postMessage({ kind: "layer", requestId: pending.requestId,
          identity: pending.identity, layerIndex: reply.nextLayerIndex, primary, opposite },
        [...new Set(transfer)] as Transferable[]);
      } else {
        this.worker.postMessage({ kind: "finish", requestId: pending.requestId, identity: pending.identity });
      }
      return;
    }
    this.pending = null;
    if (!reply.ok) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(reply.error));
      return;
    }
    const geometries = [reply.primary.geometry, reply.opposite.geometry]
      .filter((geometry): geometry is VortexSurfaceGeometry => geometry != null);
    const actualBytes = geometries.reduce((sum, geometry) => sum + geometry.bytes, 0);
    if (actualBytes > pending.maximumOutputBytes
        || reply.knownScratchBytes > pending.maximumScratchBytes) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error("Nested surface composition exceeded its topology-derived admission"));
      return;
    }
    const transition = visualizationResourceGovernor.transition(
      pending.reservationId,
      geometries.flatMap((geometry) => [geometry.points, geometry.polys, geometry.normals])
        .map((array) => visualizationBufferClaim(array.buffer, "retained-geometry")),
    );
    if (!transition.ok) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(transition.reason));
      return;
    }
    if (this.retainedReservationId != null) visualizationResourceGovernor.release(this.retainedReservationId);
    this.retainedReservationId = pending.reservationId;
    visualizationResourceGovernor.recordPrediction(
      pending.maximumOutputBytes + pending.maximumScratchBytes,
      actualBytes + reply.knownScratchBytes,
    );
    pending.resolve({ primary: reply.primary, opposite: reply.opposite });
  }

  private fail(error: Error): void {
    if (!this.pending) { this.rejectReady(error); return; }
    const pending = this.pending; this.pending = null;
    visualizationResourceGovernor.release(pending.reservationId);
    pending.reject(error);
  }
}
