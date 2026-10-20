import {
  prepareVtkRequest,
  numericalCloudPreparationRequest,
  type ScalarPreparationRequest,
  type VectorPreparationRequest,
  type VelocityDiagnosticPreparationRequest,
  type CloudPreparationRequest,
  type PreparedVtkImage,
  type PreparedVectorImage,
  type PreparedCloudImages,
  type VtkPreparationRequest,
  type VtkPreparationResult,
} from "./vtkPreparation";
import {
  recordVtkDatasetHandoff,
  qualifySharedVtkGeometryHandoff,
  type VtkDatasetHandoffReport,
} from "./vtkDatasetHandoff";
import { requiresLocalVtkGeometryFallback } from "./vtkWorkerRuntime";
import vtkPreparationWorkerUrl from "./vtkPreparation.worker.ts?worker&url";
import {
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import {
  vtkPreparationInputBytes,
  vtkPreparationOutputUpperBoundBytes,
  vtkPreparationWorkerScratchUpperBoundBytes,
  vtkPreparationResultBytes,
} from "./vtkPreparationResourcePlanning";
import type { GridSpacing } from "@/lib/science/cartesianInterp";
import type { Npy3D } from "@/lib/platform/npy";

interface WorkerResultReply {
  kind?: "result";
  requestId?: number;
  ok: boolean;
  result?: VtkPreparationResult;
  error?: string;
  handoff?: VtkDatasetHandoffReport;
}

interface WorkerReadyReply {
  kind: "ready";
  sharedGeometrySupported?: boolean;
}

interface WorkerSharedBufferProbeReply {
  kind: "shared-buffer-probe";
  buffer: SharedArrayBuffer;
}

type WorkerReply = WorkerResultReply | WorkerReadyReply;

export interface PreparationWorker {
  onmessage: ((event: { data: WorkerReply }) => void) | null;
  onerror: ((event: { message: string; preventDefault?: () => void }) => void) | null;
  onmessageerror?: ((event: { preventDefault?: () => void }) => void) | null;
  postMessage: (message:
    | VtkPreparationRequest
    | { kind: "prepare"; requestId: number; request: VtkPreparationRequest }
    | { kind: "close" }
  ) => void;
  terminate: () => void;
}

export type PreparationWorkerFactory = () => PreparationWorker;

export interface PreparationScheduleOptions {
  /** Stable owner lane. Only its newest queued request is allowed to start. */
  scope?: string;
  /** Explicit execution deadline for tests or unusually slow platforms. */
  timeoutMs?: number;
}

let preparationWorkerGeneration = 0;

function defaultWorkerFactory(): PreparationWorker {
  const separator = vtkPreparationWorkerUrl.includes("?") ? "&" : "?";
  const url = `${vtkPreparationWorkerUrl}${separator}generation=${++preparationWorkerGeneration}`;
  return new Worker(url, {
    type: "module",
    name: "nimbus-vtk-preparation",
  }) as unknown as PreparationWorker;
}

function abortError(): DOMException {
  return new DOMException("VTK preparation cancelled", "AbortError");
}

function transferableGeometryHandoffReason(): string | null {
  if (typeof navigator === "undefined") return null;
  // Brave 1.84/Dawn on macOS terminates the renderer process when vtk.js
  // marching-cubes workers consume a large SAB-backed scalar grid. The
  // process dies without a catchable DataCloneError, so its safe capability
  // result is transferable AB even though the small SAB echo probe succeeds.
  if ("brave" in navigator) {
    return "Brave disabled SAB geometry after a qualified large-grid VTK worker crash";
  }
  const ua = navigator.userAgent;
  if (/AppleWebKit/i.test(ua) && !/(Chrome|Chromium|CriOS|Edg|OPR)/i.test(ua)) {
    return "WebKit uses transferable geometry because SAB-backed vtk.js arrays can stall rendering";
  }
  return null;
}

function prepareVtkLocally(
  request: VtkPreparationRequest,
  signal?: AbortSignal,
): Promise<VtkPreparationResult> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const onAbort = () => {
      clearTimeout(task);
      reject(abortError());
    };
    const task = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      try {
        if (signal?.aborted) throw abortError();
        const result = prepareVtkRequest(request);
        if (signal?.aborted) throw abortError();
        visualizationResourceGovernor.recordPrediction(
          vtkPreparationOutputUpperBoundBytes(request),
          vtkPreparationResultBytes(result),
        );
        resolve(result);
      } catch (error) {
        reject(error);
      }
    }, 0);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function isPreparationAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

interface QueuedPreparation {
  id: number;
  request: VtkPreparationRequest;
  scope: string | null;
  signal?: AbortSignal;
  timeoutMs: number;
  executionAttempts: number;
  resourceReservationId: number | null;
  predictedPreparedBytes: number;
  predictedWorkerScratchBytes: number;
  settled: boolean;
  resolve: (result: VtkPreparationResult) => void;
  reject: (error: unknown) => void;
  onAbort: () => void;
}

/* Reuse the preparation worker across a rapid playback/slider burst without
 * holding its module graph for seconds after the UI becomes idle. This also
 * keeps `workers === 0` useful as a settled-state signal for lifecycle and
 * resource-budget instrumentation. */
const PREPARATION_WORKER_IDLE_MS = 250;
// Cold module loading competes with software rendering before the ready handshake.
const PREPARATION_WORKER_STARTUP_MS = 30_000;

function preparationDeadlineMs(request: VtkPreparationRequest): number {
  // Structured cloning, resampling and packing are all proportional to input
  // bytes. This is deliberately generous for low-end machines while still
  // converting a dead worker into a recoverable error instead of an infinite spinner.
  return Math.min(180_000, Math.max(30_000, 15_000 + vtkPreparationInputBytes(request) / (2 * 1024 * 1024) * 1_000));
}

/**
 * A single persistent worker is the memory boundary for all VTK preparation.
 * Each plot lane retains at most one queued generation. An aborted active
 * generation terminates its worker: the numerical kernel is synchronous and
 * cannot observe a message, so worker termination is the only genuine
 * latest-wins primitive during fast playback.
 */
export class PreparationCoordinator {
  private readonly workerFactory: PreparationWorkerFactory;
  private worker: PreparationWorker | null = null;
  private ready = false;
  private active: QueuedPreparation | null = null;
  private readonly queue: QueuedPreparation[] = [];
  private sequence = 0;
  private startupAttempts = 0;
  private startupTimer: ReturnType<typeof setTimeout> | null = null;
  private executionTimer: ReturnType<typeof setTimeout> | null = null;
  private idleTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(workerFactory: PreparationWorkerFactory = defaultWorkerFactory) {
    this.workerFactory = workerFactory;
  }

  enqueue(
    request: VtkPreparationRequest,
    signal?: AbortSignal,
    options: PreparationScheduleOptions = {},
  ): Promise<VtkPreparationResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (request.kind === "cloud") request = numericalCloudPreparationRequest(request);
    if (!this.worker && !this.active && this.queue.length === 0) this.startupAttempts = 0;
    const scope = options.scope ?? null;
    if (scope) {
      for (let index = this.queue.length - 1; index >= 0; --index) {
        const obsolete = this.queue[index];
        if (obsolete.scope !== scope) continue;
        this.queue.splice(index, 1);
        this.reject(obsolete, abortError());
      }
    }
    return new Promise((resolve, reject) => {
      const item: QueuedPreparation = {
        id: ++this.sequence,
        request,
        scope,
        signal,
        timeoutMs: options.timeoutMs ?? preparationDeadlineMs(request),
        executionAttempts: 0,
        resourceReservationId: null,
        predictedPreparedBytes: vtkPreparationOutputUpperBoundBytes(request),
        predictedWorkerScratchBytes: vtkPreparationWorkerScratchUpperBoundBytes(request),
        settled: false,
        resolve,
        reject,
        onAbort: () => this.abort(item),
      };
      signal?.addEventListener("abort", item.onAbort, { once: true });
      this.queue.push(item);
      this.ensureWorker();
      this.startNext();
    });
  }

  /** Deterministic teardown for app shutdown and scheduler tests. */
  shutdown(): void {
    const error = abortError();
    if (this.active) {
      this.releaseResources(this.active);
      this.reject(this.active, error);
    }
    this.active = null;
    for (const item of this.queue.splice(0)) this.reject(item, error);
    this.clearExecutionTimer();
    if (this.idleTimer != null) clearTimeout(this.idleTimer);
    this.idleTimer = null;
    this.disposeWorker();
  }

  private abort(item: QueuedPreparation): void {
    if (item.settled) return;
    const queuedIndex = this.queue.indexOf(item);
    if (queuedIndex >= 0) this.queue.splice(queuedIndex, 1);
    if (this.active === item) {
      this.active = null;
      this.clearExecutionTimer();
      this.releaseResources(item);
      this.reject(item, abortError());
      this.disposeWorker();
      if (this.queue.length > 0) {
        this.ensureWorker();
        this.startNext();
      }
      return;
    }
    this.reject(item, abortError());
    if (!this.active && this.queue.length === 0) this.disposeWorker();
  }

  private reject(item: QueuedPreparation, error: unknown): void {
    if (item.settled) return;
    item.settled = true;
    item.signal?.removeEventListener("abort", item.onAbort);
    item.reject(error);
  }

  private resolve(item: QueuedPreparation, result: VtkPreparationResult): void {
    if (item.settled) return;
    item.settled = true;
    item.signal?.removeEventListener("abort", item.onAbort);
    item.resolve(result);
  }

  private ensureWorker(): void {
    if (this.worker) return;
    if (this.idleTimer != null) clearTimeout(this.idleTimer);
    this.idleTimer = null;
    const current = this.workerFactory();
    this.worker = current;
    this.ready = false;
    current.onmessage = (event) => {
      if (this.worker !== current) return;
      const reply = event.data as WorkerReply | WorkerSharedBufferProbeReply;
      if (reply.kind === "shared-buffer-probe") {
        try {
          (current.postMessage as (message: unknown) => void)({
            kind: "shared-buffer-probe-ack",
            buffer: reply.buffer,
          });
        } catch {
          // The worker's probe timeout will advertise transferable-only mode.
        }
        return;
      }
      if (reply.kind === "ready") {
        this.ready = true;
        this.startupAttempts = 0;
        if (this.startupTimer != null) clearTimeout(this.startupTimer);
        this.startupTimer = null;
        this.startNext();
        return;
      }
      const active = this.active;
      if (!active || (reply.requestId != null && reply.requestId !== active.id)) return;
      this.clearExecutionTimer();
      this.active = null;
      if (reply.ok && reply.result) {
        if (reply.handoff) recordVtkDatasetHandoff(reply.handoff);
        visualizationResourceGovernor.recordPrediction(
          active.predictedPreparedBytes,
          vtkPreparationResultBytes(reply.result),
        );
        this.releaseResources(active);
        this.resolve(active, reply.result);
      }
      else {
        this.releaseResources(active);
        this.reject(active, new Error(reply.error || "VTK preparation failed"));
      }
      this.startNext();
      this.retireWhenIdle();
    };
    current.onerror = (event) => {
      event.preventDefault?.();
      if (this.worker !== current) return;
      this.workerFailed(new Error(event.message || "VTK preparation worker failed"));
    };
    current.onmessageerror = (event) => {
      event.preventDefault?.();
      if (this.worker !== current) return;
      this.workerFailed(new Error("VTK preparation worker returned an unreadable result"));
    };
    this.startupTimer = setTimeout(() => {
      if (this.worker === current && !this.ready) {
        this.workerFailed(new Error("VTK preparation worker did not start"));
      }
    }, PREPARATION_WORKER_STARTUP_MS);
  }

  private startNext(): void {
    if (!this.worker || !this.ready || this.active) return;
    while (this.queue.length > 0) {
      const item = this.queue.shift()!;
      if (item.settled || item.signal?.aborted) {
        this.reject(item, abortError());
        continue;
      }
      this.active = item;
      item.executionAttempts += 1;
      const resources = visualizationResourceGovernor.reserve({
        owner: item.scope ?? "vtk-preparation",
        generation: `${item.id}:${item.executionAttempts}`,
        priority: "interactive",
        claims: [
          {
            allocationId: `vtk-preparation-input:${item.id}:${item.executionAttempts}`,
            stage: "worker-resident",
            bytes: vtkPreparationInputBytes(item.request),
          },
          visualizationReservationClaim(
            `vtk-preparation-output:${item.id}:${item.executionAttempts}`,
            "handoff",
            item.predictedPreparedBytes,
          ),
          visualizationReservationClaim(
            `vtk-preparation-scratch:${item.id}:${item.executionAttempts}`,
            "worker-scratch",
            item.predictedWorkerScratchBytes,
          ),
        ],
      });
      if (!resources.ok) {
        this.active = null;
        this.reject(item, new Error(resources.reason));
        continue;
      }
      item.resourceReservationId = resources.reservation.id;
      try {
        this.worker.postMessage({ kind: "prepare", requestId: item.id, request: item.request });
      } catch (error) {
        this.workerFailed(error instanceof Error ? error : new Error(String(error)));
        return;
      }
      this.executionTimer = setTimeout(() => {
        if (this.active !== item) return;
        this.workerFailed(new Error(
          `VTK preparation exceeded its ${Math.ceil(item.timeoutMs / 1_000)} s memory-safe deadline`,
        ));
      }, item.timeoutMs);
      return;
    }
    this.retireWhenIdle();
  }

  private workerFailed(error: Error): void {
    const active = this.active;
    this.active = null;
    this.clearExecutionTimer();
    this.disposeWorker();
    if (active) {
      this.releaseResources(active);
      if (!active.settled && !active.signal?.aborted && active.executionAttempts < 2) {
        this.queue.unshift(active);
      } else {
        this.reject(active, error);
      }
      if (this.queue.length > 0) {
        setTimeout(() => {
          if (this.queue.length === 0) return;
          this.ensureWorker();
          this.startNext();
        }, 25);
      }
      return;
    }
    this.startupAttempts += 1;
    if (this.queue.length === 0) return;
    if (this.startupAttempts < 2) {
      setTimeout(() => {
        if (this.queue.length === 0) return;
        this.ensureWorker();
        this.startNext();
      }, 25);
      return;
    }
    const queued = this.queue.splice(0);
    for (const item of queued) this.reject(item, error);
  }

  private clearExecutionTimer(): void {
    if (this.executionTimer != null) clearTimeout(this.executionTimer);
    this.executionTimer = null;
  }

  private releaseResources(item: QueuedPreparation): void {
    if (item.resourceReservationId == null) return;
    visualizationResourceGovernor.release(item.resourceReservationId);
    item.resourceReservationId = null;
  }

  private retireWhenIdle(): void {
    if (this.active || this.queue.length > 0 || this.idleTimer != null) return;
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (!this.active && this.queue.length === 0) this.disposeWorker();
    }, PREPARATION_WORKER_IDLE_MS);
  }

  private disposeWorker(): void {
    const current = this.worker;
    this.worker = null;
    this.ready = false;
    if (this.startupTimer != null) clearTimeout(this.startupTimer);
    this.startupTimer = null;
    if (!current) return;
    current.onmessage = null;
    current.onerror = (event) => event.preventDefault?.();
    if (current.onmessageerror !== undefined) current.onmessageerror = null;
    current.terminate();
  }
}

let defaultCoordinator: PreparationCoordinator | null = null;

let preparationObjectSequence = 0;
const preparationObjectIds = new WeakMap<object, number>();

function preparationObjectId(value: object | null | undefined): number {
  if (!value) return 0;
  let id = preparationObjectIds.get(value);
  if (id == null) {
    id = ++preparationObjectSequence;
    preparationObjectIds.set(value, id);
  }
  return id;
}

function identityNumber(value: number | null | undefined): string {
  if (value == null) return value === null ? "null" : "undefined";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function identityNumbers(values: readonly number[] | null | undefined): readonly string[] | null {
  return values?.map(identityNumber) ?? null;
}

function vtkGridIdentity(grid: GridSpacing): string {
  return JSON.stringify([
    grid.coordinateSystem ?? null,
    grid.geometryKind ?? null,
    grid.terrainFollowing ?? false,
    identityNumber(grid.drM),
    identityNumber(grid.dyM),
    identityNumber(grid.dthetaRad),
    identityNumber(grid.dzM),
    grid.scalarCoordinatesM?.map(identityNumbers) ?? null,
    identityNumbers(grid.scalarOriginM),
    grid.physicalBoundsM
      ? [identityNumbers(grid.physicalBoundsM.lower), identityNumbers(grid.physicalBoundsM.upper)]
      : null,
    grid.validIndexRange
      ? [identityNumbers(grid.validIndexRange.lower), identityNumbers(grid.validIndexRange.upper)]
      : null,
  ]);
}

function typedArrayIdentity(view: ArrayBufferView | null | undefined): string | null {
  if (!view) return null;
  return [
    preparationObjectId(view.buffer),
    view.byteOffset,
    view.byteLength,
  ].join(":");
}

function vtkFieldIdentity(field: Npy3D | undefined): string {
  if (!field) return "none";
  const geometry = field.windowGeometry;
  return JSON.stringify([
    typedArrayIdentity(field.data),
    field.shape,
    geometry ? [
      geometry.aggregation ?? null,
      geometry.weighting,
      geometry.sampleLocation,
      geometry.centroidsM
        ? [
            typedArrayIdentity(geometry.centroidsM.x),
            typedArrayIdentity(geometry.centroidsM.y),
            typedArrayIdentity(geometry.centroidsM.z),
          ]
        : null,
      typedArrayIdentity(geometry.volumesM3),
      identityNumbers(geometry.sourceShape),
      identityNumbers(geometry.origin),
      identityNumbers(geometry.extent),
      identityNumbers(geometry.stride),
      geometry.sourceFormat ?? null,
    ] : null,
  ]);
}

/**
 * Exact identity for an immutable preparation input. Field identity avoids
 * hashing volume-sized arrays. Grid metadata is structural because each pane
 * builds its own small wrapper around the same coordinates.
 */
export function vtkPreparationIdentity(request: VtkPreparationRequest): string {
  const fieldId = vtkFieldIdentity;
  const grid = vtkGridIdentity(request.grid);
  if (request.kind === "scalar") {
    return [
      "scalar", fieldId(request.field), grid, request.maxVoxels ?? "",
      request.includeValidity !== false,
      request.datasetHandoff ?? "render-only",
      fieldId(request.subtractField), request.refineForDisplay ? 1 : 0,
      request.featureSign ?? "", request.featureZRangeM?.[0] ?? "",
      request.featureZRangeM?.[1] ?? "",
      request.scalarTransform ?? "none",
      JSON.stringify(request.background ?? null), request.cropBackground === true,
      request.structuralSupportRangeFraction ?? "",
      ...(request.excludePhysicalBoundsM
        ? [...request.excludePhysicalBoundsM.lower, ...request.excludePhysicalBoundsM.upper]
        : ["", "", "", "", "", ""]),
    ].join(":");
  }
  if (request.kind === "cloud") {
    return [
      "cloud", fieldId(request.field), grid, request.volumeMaxVoxels,
      request.datasetHandoff ?? "render-only",
      request.surfaceMaxVoxels, fieldId(request.subtractField),
      ...(request.excludePhysicalBoundsM
        ? [...request.excludePhysicalBoundsM.lower, ...request.excludePhysicalBoundsM.upper]
        : ["", "", "", "", "", ""]),
    ].join(":");
  }
  return [
    request.kind,
    request.kind === "velocity-diagnostic" ? request.includeValidity !== false : "",
    ...(request.fields.map(fieldId)),
    grid,
    request.maxVoxels ?? "",
    request.datasetHandoff ?? "render-only",
    request.kind === "vector"
      ? request.detectVerticalVortexCore ? 1 : 0
      : `${request.diagnostic}:${request.refineForDisplay ? 1 : 0}`,
  ].join(":");
}

interface SharedPreparation {
  controller: AbortController;
  promise: Promise<VtkPreparationResult>;
  consumers: number;
  complete: boolean;
}

/**
 * Coalesces identical, concurrent preparations across panes. Each consumer
 * retains independent cancellation; the synchronous worker is terminated only
 * after the final consumer releases it. Results are not retained here—the
 * owning React state and VTK scene already hold them, so this cannot become a
 * second multi-frame volume cache.
 */
export class SharedPreparationCoordinator {
  private readonly shared = new Map<string, SharedPreparation>();
  private readonly coordinator: PreparationCoordinator;

  constructor(coordinator: PreparationCoordinator) {
    this.coordinator = coordinator;
  }

  enqueue(
    request: VtkPreparationRequest,
    signal?: AbortSignal,
    options: PreparationScheduleOptions = {},
  ): Promise<VtkPreparationResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    const key = vtkPreparationIdentity(request);
    let load = this.shared.get(key);
    if (!load) {
      const controller = new AbortController();
      load = {
        controller,
        consumers: 0,
        complete: false,
        // A shared job cannot belong to one caller's latest-wins scope. The
        // reference-counted consumer signals below are its ownership boundary.
        promise: this.coordinator.enqueue(request, controller.signal, {
          timeoutMs: options.timeoutMs,
        }),
      };
      this.shared.set(key, load);
      const current = load;
      current.promise.finally(() => {
        current.complete = true;
        if (this.shared.get(key) === current) this.shared.delete(key);
      }).catch(() => undefined);
    }
    load.consumers += 1;

    return new Promise((resolve, reject) => {
      let settled = false;
      const release = () => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener("abort", onAbort);
        load!.consumers = Math.max(0, load!.consumers - 1);
        if (load!.consumers === 0 && !load!.complete) {
          // The worker rejects asynchronously after termination. Remove this
          // abandoned generation before aborting it so an immediate recovery
          // generation with the same immutable inputs cannot join a promise
          // that is already destined to reject.
          if (this.shared.get(key) === load) this.shared.delete(key);
          load!.controller.abort();
        }
      };
      const onAbort = () => {
        release();
        reject(abortError());
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      load!.promise.then(
        (result) => {
          if (settled) return;
          release();
          resolve(result);
        },
        (error) => {
          if (settled) return;
          release();
          reject(error);
        },
      );
    });
  }
}

let defaultSharedCoordinator: SharedPreparationCoordinator | null = null;

function coordinatedPreparation(
  request: VtkPreparationRequest,
  signal?: AbortSignal,
  options?: PreparationScheduleOptions,
): Promise<VtkPreparationResult> {
  defaultCoordinator ??= new PreparationCoordinator();
  defaultSharedCoordinator ??= new SharedPreparationCoordinator(defaultCoordinator);
  return defaultSharedCoordinator.enqueue(request, signal, options);
}

export function prepareVtkInWorker(
  request: ScalarPreparationRequest,
  signal?: AbortSignal,
  workerFactory?: PreparationWorkerFactory,
  options?: PreparationScheduleOptions,
): Promise<PreparedVtkImage>;
export function prepareVtkInWorker(
  request: VectorPreparationRequest,
  signal?: AbortSignal,
  workerFactory?: PreparationWorkerFactory,
  options?: PreparationScheduleOptions,
): Promise<PreparedVectorImage>;
export function prepareVtkInWorker(
  request: VelocityDiagnosticPreparationRequest,
  signal?: AbortSignal,
  workerFactory?: PreparationWorkerFactory,
  options?: PreparationScheduleOptions,
): Promise<PreparedVtkImage>;
export function prepareVtkInWorker(
  request: CloudPreparationRequest,
  signal?: AbortSignal,
  workerFactory?: PreparationWorkerFactory,
  options?: PreparationScheduleOptions,
): Promise<PreparedCloudImages>;
export function prepareVtkInWorker(
  request: VtkPreparationRequest,
  signal?: AbortSignal,
  workerFactory?: PreparationWorkerFactory,
  options?: PreparationScheduleOptions,
): Promise<VtkPreparationResult>;
export function prepareVtkInWorker(
  request: VtkPreparationRequest,
  signal?: AbortSignal,
  workerFactory: PreparationWorkerFactory = defaultWorkerFactory,
  options: PreparationScheduleOptions = {},
): Promise<VtkPreparationResult> {
  if (signal?.aborted) return Promise.reject(abortError());
  if (request.datasetHandoff === "shared-geometry"
      && workerFactory === defaultWorkerFactory) {
    if (requiresLocalVtkGeometryFallback()) {
      return prepareVtkInWorker(
        { ...request, datasetHandoff: "render-only" },
        signal,
        workerFactory,
        options,
      );
    }
    const transferableReason = transferableGeometryHandoffReason();
    if (transferableReason) {
      recordVtkDatasetHandoff({
        mode: "render-only-transfer",
        logicalBytes: 0,
        copiedBytes: 0,
        datasets: 0,
        stage: "preparation",
        dataset: "prepared-grid",
        reason: transferableReason,
      });
      return prepareVtkInWorker(
        { ...request, datasetHandoff: "render-only" },
        signal,
        workerFactory,
        options,
      );
    }
    return qualifySharedVtkGeometryHandoff().then((qualified) => coordinatedPreparation(
      { ...request, datasetHandoff: qualified ? "shared-geometry" : "render-only" },
      signal,
      options,
    ));
  }
  // Node-only deterministic path for bounded library tests. A browser without
  // module workers must fail closed rather than execute preparation in React.
  if (!("Worker" in globalThis) && workerFactory === defaultWorkerFactory) {
    return typeof window === "undefined"
      ? prepareVtkLocally(request, signal)
      : Promise.reject(new Error("3D field preparation requires module-worker support"));
  }
  if (workerFactory === defaultWorkerFactory) {
    return coordinatedPreparation(request, signal, options);
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    let worker: PreparationWorker | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let startupTimer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    const releaseWorker = (terminate = true) => {
      const current = worker;
      worker = null;
      if (startupTimer != null) clearTimeout(startupTimer);
      startupTimer = null;
      if (!current) return;
      // WebKit can dispatch a late module-worker error while terminate() tears
      // down a superseded preparation. Detach the handlers first so a handled
      // cancellation never becomes a window-level page error.
      current.onmessage = null;
      current.onerror = (event) => event.preventDefault?.();
      if (terminate) {
        current.terminate();
      }
    };
    const finish = (action: () => void, terminate = true) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", onAbort);
      if (retryTimer != null) clearTimeout(retryTimer);
      retryTimer = null;
      releaseWorker(terminate);
      action();
    };
    const onAbort = () => finish(() => reject(abortError()));
    signal?.addEventListener("abort", onAbort, { once: true });

    const start = () => {
      retryTimer = null;
      if (settled) return;
      if (signal?.aborted) {
        finish(() => reject(abortError()));
        return;
      }
      attempts += 1;
      const current = workerFactory();
      worker = current;
      current.onmessage = (event) => {
        if (worker !== current) return;
        const reply = event.data as WorkerReply | WorkerSharedBufferProbeReply;
        if (reply.kind === "shared-buffer-probe") {
          try {
            (current.postMessage as (message: unknown) => void)({
              kind: "shared-buffer-probe-ack",
              buffer: reply.buffer,
            });
          } catch {
            // The worker's probe timeout will advertise transferable-only mode.
          }
          return;
        }
        if (reply.kind === "ready") {
          if (startupTimer != null) clearTimeout(startupTimer);
          startupTimer = null;
          current.postMessage(request);
          return;
        }
        // Acknowledging receipt before the worker closes makes transferable
        // delivery deterministic in WebKit. The close command is processed
        // only after this message has crossed to the UI thread.
        current.postMessage({ kind: "close" });
        // The one-shot worker closes itself after receiving the acknowledgment;
        // forcing termination here would race that delivery in WebKit.
        if (reply.ok && reply.result) {
          if (reply.handoff) recordVtkDatasetHandoff(reply.handoff);
          visualizationResourceGovernor.recordPrediction(
            vtkPreparationOutputUpperBoundBytes(request),
            vtkPreparationResultBytes(reply.result),
          );
          finish(() => resolve(reply.result!), false);
        }
        else finish(() => reject(new Error(reply.error || "VTK preparation failed")), false);
      };
      const handleStartupFailure = (message: string, preventDefault?: () => void) => {
        if (worker !== current || settled) return;
        preventDefault?.();
        releaseWorker();
        // WKWebView/WebKit may briefly reject a same-origin module worker when
        // a prior instance of the same module was just terminated. One delayed
        // restart crosses that teardown boundary without changing the request.
        if (attempts < 2) {
          retryTimer = setTimeout(start, 25);
          return;
        }
        // Correctness fallback for WebKit/WKWebView environments that refuse
        // the worker even after a clean restart. The request is already voxel
        // bounded; running the same pure preparation locally is preferable to
        // presenting an empty scene as if it were valid data.
        retryTimer = setTimeout(() => {
          retryTimer = null;
          if (settled) return;
          if (signal?.aborted) {
            finish(() => reject(abortError()), false);
            return;
          }
          try {
            const result = prepareVtkRequest(request);
            visualizationResourceGovernor.recordPrediction(
              vtkPreparationOutputUpperBoundBytes(request),
              vtkPreparationResultBytes(result),
            );
            if (signal?.aborted) {
              finish(() => reject(abortError()), false);
              return;
            }
            finish(() => resolve(result), false);
          } catch (fallbackError) {
            finish(() => reject(new Error(
              `${message}; local fallback failed: ${fallbackError instanceof Error
                ? fallbackError.message
                : String(fallbackError)}`,
            )), false);
          }
        }, 0);
      };
      current.onerror = (event) => handleStartupFailure(
        event.message || "VTK preparation worker failed",
        event.preventDefault?.bind(event),
      );
      // Injected workers predate the browser ready handshake and remain
      // intentionally minimal for deterministic unit tests.
      current.postMessage(request);
    };
    start();
  });
}
