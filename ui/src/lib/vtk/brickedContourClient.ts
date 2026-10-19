import brickedContourWorkerUrl from "./brickedContour.worker.ts?worker&url";
import type {
  BrickedContourWorkerReply,
  BrickedContourWorkerRequest,
  BrickedPreparedStats,
  BrickedScalarPreparationRequest,
} from "./brickedContourProtocol";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";
import { handoffVtkPreparationInput } from "./vtkPreparationInputHandoff";
import {
  vtkPreparationInputBytes,
  vtkPreparationOutputUpperBoundBytes,
} from "./vtkPreparationResourcePlanning";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import { contourTopologyUpperBound, contourUsageViolation } from "./vtkContourBudget";

interface WorkerLike {
  onmessage: ((event: MessageEvent<BrickedContourWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: BrickedContourWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

interface PendingLoad {
  brickId: string;
  resolve: (stats: BrickedPreparedStats) => void;
  reject: (error: unknown) => void;
}

interface BrickAdmission {
  inputBytes: number;
  outputBytes: number;
  actualBytes: number | null;
  contourOutputBytes: number;
  contourTriangles: number;
}

interface PendingExtraction {
  reservationId: number;
  retentionKey: string;
  maximumOutputBytes: number;
  maximumTriangles: number;
  maximumScratchBytes: number;
  resolve: (chunks: Array<{ brickId: string; geometry: VortexSurfaceGeometry | null }>) => void;
  reject: (error: unknown) => void;
}

let clientSequence = 0;

export class BrickedContourClient {
  private readonly namespace = `bricked-contour:${++clientSequence}`;
  private readonly worker: WorkerLike;
  private readonly loaded = new Set<string>();
  private readonly admissions = new Map<string, BrickAdmission>();
  private readonly stats = new Map<string, BrickedPreparedStats>();
  private readonly pendingLoads = new Map<number, PendingLoad>();
  private readonly pendingExtractions = new Map<number, PendingExtraction>();
  private readonly extractionReservations = new Map<string, number>();
  private readonly extractionGenerations = new Map<string, Set<string>>();
  private extractionGenerationSequence = 0;
  private requestSequence = 0;
  private datasetIdentity = "";
  private generationReservationId: number | null = null;
  private ready = false;
  private readyResolve!: () => void;
  private readyReject!: (error: unknown) => void;
  private readonly readyPromise: Promise<void>;
  private closed = false;
  private readonly externallyReserved: boolean;

  constructor(worker?: WorkerLike, options?: { externallyReserved?: boolean }) {
    this.externallyReserved = options?.externallyReserved ?? false;
    const separator = brickedContourWorkerUrl.includes("?") ? "&" : "?";
    const url = `${brickedContourWorkerUrl}${separator}generation=${clientSequence}`;
    this.worker = worker ?? new Worker(url, {
      type: "module",
      name: "nimbus-bricked-contour-set",
    }) as unknown as WorkerLike;
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.worker.onmessage = (event) => this.message(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault?.();
      this.fail(new Error(event.message || "Bricked contour worker failed"));
    };
    this.worker.onmessageerror = () => this.fail(new Error("Bricked contour worker returned unreadable data"));
  }

  loadedBrickCount(): number {
    return this.loaded.size;
  }

  residentBytes(): number {
    return [...this.admissions.values()].reduce(
      (sum, admission) => sum + (admission.actualBytes ?? 0), 0,
    );
  }

  isClosed(): boolean {
    return this.closed;
  }

  hasBrick(brickId: string): boolean {
    return this.loaded.has(brickId);
  }

  brickStats(brickId: string): BrickedPreparedStats | null {
    return this.stats.get(brickId) ?? null;
  }

  admittedBrickCount(): number {
    return this.admissions.size;
  }

  async admitBricks(
    bricks: readonly {
      brickId: string;
      inputBytes: number;
      outputBytes: number;
      sampleUpperBound?: number;
      contourMethod?: "marching-cubes" | "terrain-tetrahedra";
    }[],
    signal?: AbortSignal,
  ): Promise<void> {
    await this.readyPromise;
    if (signal?.aborted) throw new DOMException("Bricked contour admission cancelled", "AbortError");
    if (!this.datasetIdentity) throw new Error("Bricked contour dataset is not initialized");
    const additions = bricks.filter((brick) => !this.admissions.has(brick.brickId));
    if (additions.length === 0) return;
    const candidate = new Map(this.admissions);
    for (const brick of additions) {
      if (!Number.isSafeInteger(brick.inputBytes) || brick.inputBytes <= 0
          || !Number.isSafeInteger(brick.outputBytes) || brick.outputBytes <= 0
          || !Number.isSafeInteger(brick.sampleUpperBound ?? brick.outputBytes / 4)
          || (brick.sampleUpperBound ?? brick.outputBytes / 4) < 1) {
        throw new Error("Bricked contour admission bytes are invalid");
      }
      const contour = contourTopologyUpperBound(
        brick.sampleUpperBound ?? brick.outputBytes / 4,
        brick.contourMethod,
      );
      candidate.set(brick.brickId, {
        inputBytes: brick.inputBytes,
        outputBytes: brick.outputBytes,
        actualBytes: null,
        contourOutputBytes: contour.bytes,
        contourTriangles: contour.triangles,
      });
    }
    if (!this.externallyReserved) {
      const claims = this.claims(candidate);
      const outcome = this.generationReservationId == null
        ? visualizationResourceGovernor.reserve({
            owner: this.namespace,
            generation: this.datasetIdentity,
            priority: "visible",
            claims,
          })
        : visualizationResourceGovernor.transition(this.generationReservationId, claims);
      if (!outcome.ok) throw new Error(outcome.reason);
      this.generationReservationId = outcome.reservation.id;
    }
    this.admissions.clear();
    this.stats.clear();
    for (const [brickId, admission] of candidate) this.admissions.set(brickId, admission);
  }

  async reset(datasetIdentity: string, signal?: AbortSignal): Promise<void> {
    await this.readyPromise;
    if (signal?.aborted) throw new DOMException("Bricked contour reset cancelled", "AbortError");
    if (this.datasetIdentity === datasetIdentity) return;
    this.rejectPending(new DOMException("Bricked contour generation replaced", "AbortError"));
    if (this.generationReservationId != null) {
      visualizationResourceGovernor.release(this.generationReservationId);
      this.generationReservationId = null;
    }
    this.admissions.clear();
    this.loaded.clear();
    this.releaseExtractionReservations();
    this.datasetIdentity = datasetIdentity;
    this.worker.postMessage({ kind: "reset", datasetIdentity });
  }

  async loadBrick(
    brickId: string,
    request: BrickedScalarPreparationRequest,
    signal: AbortSignal,
    terrainCellBounds?: [number, number, number, number, number, number],
  ): Promise<BrickedPreparedStats> {
    await this.readyPromise;
    if (this.closed) throw new Error("Bricked contour client is closed");
    if (!this.datasetIdentity) throw new Error("Bricked contour dataset is not initialized");
    if (this.loaded.has(brickId)) return this.stats.get(brickId)!;
    if (!this.admissions.has(brickId)) {
      throw new Error("Bricked contour brick was not admitted before its field read");
    }
    if (signal.aborted) throw new DOMException("Bricked contour load cancelled", "AbortError");
    const requestId = ++this.requestSequence;
    const admission = this.admissions.get(brickId)!;
    request = { ...request, includeValidity: false };
    const actualInput = vtkPreparationInputBytes(request);
    const actualOutput = vtkPreparationOutputUpperBoundBytes(request);
    if (actualInput > admission.inputBytes || actualOutput > admission.outputBytes) {
      throw new Error(
        `Bricked contour preparation exceeded its admitted upper bound `
          + `(input ${actualInput}/${admission.inputBytes} bytes, `
          + `output ${actualOutput}/${admission.outputBytes} bytes)`,
      );
    }
    const promise = new Promise<BrickedPreparedStats>((resolve, reject) => {
      this.pendingLoads.set(requestId, {
        brickId,
        resolve,
        reject,
      });
    });
    try {
      const handoff = await handoffVtkPreparationInput(request, () => signal.aborted);
      if (signal.aborted) throw new DOMException("Bricked contour load cancelled", "AbortError");
      this.worker.postMessage({
        kind: "load",
        requestId,
        datasetIdentity: this.datasetIdentity,
        brickId,
        request: handoff.request,
        terrainCellBounds,
      }, handoff.transfer);
    } catch (error) {
      const pending = this.pendingLoads.get(requestId);
      if (pending) {
        this.pendingLoads.delete(requestId);
        this.removeUnloadedAdmission(pending.brickId);
        pending.reject(error);
      }
    }
    return promise;
  }

  async extract(
    brickIds: readonly string[],
    threshold: number,
    signal: AbortSignal,
    options: { retentionKey?: string; maximumOutputBytes?: number } = {},
  ): Promise<Array<{ brickId: string; geometry: VortexSurfaceGeometry | null }>> {
    await this.readyPromise;
    if (!Number.isFinite(threshold)) throw new Error("Bricked contour threshold must be finite");
    if (brickIds.some((brickId) => !this.loaded.has(brickId))) {
      throw new Error("Bricked contour extraction requested an unloaded brick");
    }
    if (signal.aborted) throw new DOMException("Bricked contour extraction cancelled", "AbortError");
    const retentionKey = options.retentionKey?.trim() || "default";
    if (brickIds.length === 0) {
      this.releaseExtraction(retentionKey);
      return [];
    }
    if (options.maximumOutputBytes != null
        && (!Number.isSafeInteger(options.maximumOutputBytes) || options.maximumOutputBytes <= 0)) {
      throw new Error("Bricked contour output admission is invalid");
    }
    const generation = ++this.extractionGenerationSequence;
    const nextKeys = new Set<string>();
    const chunks: Array<{ brickId: string; geometry: VortexSurfaceGeometry | null }> = [];
    let actualBytes = 0;
    try {
      for (const [index, brickId] of brickIds.entries()) {
        if (signal.aborted) throw new DOMException("Bricked contour extraction cancelled", "AbortError");
        const admission = this.admissions.get(brickId)!;
        const key = `${retentionKey}:generation:${generation}:brick:${index}:${brickId}`;
        const chunk = await this.extractOne(
          brickId,
          threshold,
          signal,
          key,
          admission.contourOutputBytes,
          admission.contourTriangles,
        );
        chunks.push(chunk);
        if (chunk.geometry) actualBytes += chunk.geometry.bytes;
        if (options.maximumOutputBytes != null && actualBytes > options.maximumOutputBytes) {
          throw new Error(
            `Progressive contour output is ${actualBytes} bytes; limit is ${options.maximumOutputBytes}`,
          );
        }
        if (this.extractionReservations.has(key)) nextKeys.add(key);
      }
      const previous = this.extractionGenerations.get(retentionKey);
      if (previous) for (const key of previous) this.releaseExtractionKey(key);
      this.extractionGenerations.set(retentionKey, nextKeys);
      return chunks;
    } catch (error) {
      for (const key of nextKeys) this.releaseExtractionKey(key);
      throw error;
    }
  }

  private async extractOne(
    brickId: string,
    threshold: number,
    signal: AbortSignal,
    retentionKey: string,
    maximumOutputBytes: number,
    maximumTriangles: number,
  ): Promise<{ brickId: string; geometry: VortexSurfaceGeometry | null }> {
    if (signal.aborted) throw new DOMException("Bricked contour extraction cancelled", "AbortError");
    const requestId = ++this.requestSequence;
    const admission = visualizationResourceGovernor.reserve({
      owner: this.namespace,
      generation: `${this.datasetIdentity}:${retentionKey}:${requestId}`,
      priority: "visible",
      claims: [
        visualizationReservationClaim(
          `${this.namespace}:${retentionKey}:${requestId}:geometry`,
          "retained-geometry",
          maximumOutputBytes,
        ),
        visualizationReservationClaim(
          `${this.namespace}:${retentionKey}:${requestId}:scratch`,
          "worker-scratch",
          maximumOutputBytes,
        ),
      ],
    });
    if (!admission.ok) throw new Error(admission.reason);
    const promise = new Promise<Array<{ brickId: string; geometry: VortexSurfaceGeometry | null }>>(
      (resolve, reject) => this.pendingExtractions.set(requestId, {
        reservationId: admission.reservation.id,
        retentionKey,
        maximumOutputBytes,
        maximumTriangles,
        maximumScratchBytes: maximumOutputBytes,
        resolve,
        reject,
      }),
    );
    const abort = () => {
      const pending = this.pendingExtractions.get(requestId);
      if (!pending) return;
      this.pendingExtractions.delete(requestId);
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new DOMException("Bricked contour extraction cancelled", "AbortError"));
      this.shutdown();
    };
    signal.addEventListener("abort", abort, { once: true });
    this.worker.postMessage({
      kind: "extract",
      requestId,
      datasetIdentity: this.datasetIdentity,
      brickIds: [brickId],
      threshold,
    });
    return promise.then((chunks) => {
      if (signal.aborted) {
        this.releaseExtraction(retentionKey);
        throw new DOMException("Bricked contour extraction cancelled", "AbortError");
      }
      const chunk = chunks[0];
      if (!chunk || chunk.brickId !== brickId || chunks.length !== 1) {
        this.releaseExtraction(retentionKey);
        throw new Error("Bricked contour worker returned the wrong progressive brick");
      }
      return chunk;
    }).finally(() => signal.removeEventListener("abort", abort));
  }

  shutdown(): void {
    if (this.closed) return;
    this.closed = true;
    this.rejectPending(new DOMException("Bricked contour client closed", "AbortError"));
    if (this.generationReservationId != null) {
      visualizationResourceGovernor.release(this.generationReservationId);
      this.generationReservationId = null;
    }
    this.admissions.clear();
    this.stats.clear();
    this.loaded.clear();
    this.releaseExtractionReservations();
    this.worker.postMessage({ kind: "close" });
    this.worker.onmessage = this.worker.onerror = this.worker.onmessageerror = null;
    this.worker.terminate();
  }

  private message(reply: BrickedContourWorkerReply): void {
    if (reply.kind === "ready") {
      this.ready = true;
      this.readyResolve();
      return;
    }
    if (reply.kind === "loaded") {
      const pending = this.pendingLoads.get(reply.requestId);
      if (!pending) return;
      this.pendingLoads.delete(reply.requestId);
      if (!reply.ok) {
        this.removeUnloadedAdmission(pending.brickId);
        pending.reject(new Error(reply.error));
        return;
      }
      const admission = this.admissions.get(pending.brickId);
      if (!admission || reply.bytes > admission.outputBytes
          || (!this.externallyReserved && this.generationReservationId == null)) {
        pending.reject(new Error("Bricked contour worker exceeded its admitted residency"));
        this.fail(new Error("Bricked contour worker exceeded its admitted residency"));
        return;
      }
      admission.actualBytes = reply.bytes;
      if (!this.externallyReserved) {
        const transition = visualizationResourceGovernor.transition(
          this.generationReservationId!,
          this.claims(this.admissions),
        );
        if (!transition.ok) {
          pending.reject(new Error(transition.reason));
          this.fail(new Error(transition.reason));
          return;
        }
      }
      visualizationResourceGovernor.recordPrediction(admission.outputBytes, reply.bytes);
      this.loaded.add(pending.brickId);
      this.stats.set(pending.brickId, reply.stats);
      pending.resolve(reply.stats);
      return;
    }
    const pending = this.pendingExtractions.get(reply.requestId);
    if (!pending) return;
    this.pendingExtractions.delete(reply.requestId);
    if (!reply.ok) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(reply.error));
      return;
    }
    const geometries = reply.chunks.map((chunk) => chunk.geometry)
      .filter((geometry): geometry is VortexSurfaceGeometry => geometry != null);
    const usage = geometries.reduce((total, geometry) => ({
      bytes: total.bytes + geometry.bytes,
      triangles: total.triangles + geometry.triangles,
    }), { bytes: 0, triangles: 0 });
    const violation = contourUsageViolation(
      usage,
      "Bricked contour worker output",
      pending.maximumOutputBytes,
      pending.maximumTriangles,
    );
    if (violation || reply.knownScratchBytes > pending.maximumScratchBytes) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(violation
        ?? `Bricked contour scratch exceeded its admitted ${pending.maximumScratchBytes} bytes`));
      return;
    }
    const retained = visualizationResourceGovernor.transition(
      pending.reservationId,
      geometries.flatMap((geometry) => [geometry.points, geometry.polys, geometry.normals])
        .map((array) => visualizationBufferClaim(array.buffer, "retained-geometry")),
    );
    if (!retained.ok) {
      visualizationResourceGovernor.release(pending.reservationId);
      pending.reject(new Error(retained.reason));
      return;
    }
    const previous = this.extractionReservations.get(pending.retentionKey);
    if (previous != null) visualizationResourceGovernor.release(previous);
    if (geometries.length > 0) this.extractionReservations.set(
      pending.retentionKey, pending.reservationId,
    ); else visualizationResourceGovernor.release(pending.reservationId);
    visualizationResourceGovernor.recordPrediction(
      pending.maximumOutputBytes + pending.maximumScratchBytes,
      usage.bytes + reply.knownScratchBytes,
    );
    pending.resolve(reply.chunks);
  }

  private rejectPending(error: Error): void {
    if (!this.ready) this.readyReject(error);
    for (const pending of this.pendingLoads.values()) {
      pending.reject(error);
    }
    for (const pending of this.pendingExtractions.values()) pending.reject(error);
    for (const pending of this.pendingExtractions.values()) {
      visualizationResourceGovernor.release(pending.reservationId);
    }
    this.pendingLoads.clear();
    this.pendingExtractions.clear();
  }

  private fail(error: Error): void {
    this.rejectPending(error);
    this.shutdown();
  }

  private claims(admissions: ReadonlyMap<string, BrickAdmission>) {
    return [...admissions].flatMap(([brickId, admission]) => admission.actualBytes == null
      ? [
          visualizationReservationClaim(
            `${this.namespace}:${brickId}:input`,
            "worker-resident",
            admission.inputBytes,
          ),
          visualizationReservationClaim(
            `${this.namespace}:${brickId}:output`,
            "worker-resident",
            admission.outputBytes,
          ),
        ]
      : [{
          allocationId: `${this.namespace}:${brickId}:resident`,
          stage: "worker-resident" as const,
          bytes: admission.actualBytes,
        }]);
  }

  private removeUnloadedAdmission(brickId: string): void {
    const admission = this.admissions.get(brickId);
    if (!admission || admission.actualBytes != null) return;
    this.admissions.delete(brickId);
    if (this.externallyReserved || this.generationReservationId == null) return;
    if (this.admissions.size === 0) {
      visualizationResourceGovernor.release(this.generationReservationId);
      this.generationReservationId = null;
      return;
    }
    const transition = visualizationResourceGovernor.transition(
      this.generationReservationId,
      this.claims(this.admissions),
    );
    if (!transition.ok) this.fail(new Error(transition.reason));
  }

  releaseExtraction(retentionKey: string): void {
    const keys = this.extractionGenerations.get(retentionKey);
    if (keys) for (const key of keys) this.releaseExtractionKey(key);
    this.extractionGenerations.delete(retentionKey);
    this.releaseExtractionKey(retentionKey);
  }

  private releaseExtractionKey(retentionKey: string): void {
    const reservation = this.extractionReservations.get(retentionKey);
    if (reservation == null) return;
    visualizationResourceGovernor.release(reservation);
    this.extractionReservations.delete(retentionKey);
  }

  private releaseExtractionReservations(): void {
    for (const reservation of this.extractionReservations.values()) {
      visualizationResourceGovernor.release(reservation);
    }
    this.extractionReservations.clear();
    this.extractionGenerations.clear();
  }
}
