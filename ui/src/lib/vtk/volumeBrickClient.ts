import volumeBrickWorkerUrl from "./volumeBrick.worker.ts?worker&url";
import type { PreparedVtkImage, ScalarPreparationRequest } from "./vtkPreparation";
import { handoffVtkPreparationInput } from "./vtkPreparationInputHandoff";
import type { VolumeBrickWorkerReply, VolumeBrickWorkerRequest } from "./volumeBrickProtocol";

interface WorkerLike {
  onmessage: ((event: MessageEvent<VolumeBrickWorkerReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage: (message: VolumeBrickWorkerRequest, transfer?: Transferable[]) => void;
  terminate: () => void;
}

interface Pending {
  resolve: (prepared: PreparedVtkImage) => void;
  reject: (error: unknown) => void;
}

interface PendingBatch {
  resolve: (prepared: PreparedVtkImage[]) => void;
  reject: (error: unknown) => void;
}

let workerGeneration = 0;

export class VolumeBrickPreparationClient {
  private readonly worker: WorkerLike;
  private readonly handoff: typeof handoffVtkPreparationInput;
  private readonly pending = new Map<number, Pending>();
  private readonly pendingBatches = new Map<number, PendingBatch>();
  private sequence = 0;
  private readyResolve!: () => void;
  private readyReject!: (error: unknown) => void;
  private readonly ready: Promise<void>;
  private closed = false;
  private terminated = false;
  private startupCloseTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(worker?: WorkerLike, handoff = handoffVtkPreparationInput) {
    this.handoff = handoff;
    const separator = volumeBrickWorkerUrl.includes("?") ? "&" : "?";
    const url = `${volumeBrickWorkerUrl}${separator}generation=${++workerGeneration}`;
    this.worker = worker ?? new Worker(url, {
      type: "module",
      name: "nimbus-volume-brick-preparation",
    }) as unknown as WorkerLike;
    this.ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    void this.ready.catch(() => undefined);
    this.worker.onmessage = (event) => this.message(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault?.();
      this.fail(new Error(event.message || "Volume brick worker failed"));
      this.terminateWorker();
    };
    this.worker.onmessageerror = () => {
      this.fail(new Error("Volume brick worker returned unreadable data"));
      this.terminateWorker();
    };
  }

  async prepare(request: ScalarPreparationRequest, signal: AbortSignal): Promise<PreparedVtkImage> {
    await this.ready;
    if (this.closed) throw new Error("Volume brick worker is closed");
    if (signal.aborted) throw new DOMException("Volume brick preparation cancelled", "AbortError");
    const requestId = ++this.sequence;
    const promise = new Promise<PreparedVtkImage>((resolve, reject) =>
      this.pending.set(requestId, { resolve, reject }));
    // Shutdown can reject while input handoff is still awaiting. The original
    // promise below still propagates that rejection to the consumer.
    void promise.catch(() => undefined);
    try {
      const handoff = await this.handoff(request, () => signal.aborted || this.closed);
      if (signal.aborted) throw new DOMException("Volume brick preparation cancelled", "AbortError");
      if (this.closed) throw new DOMException("Volume brick worker closed", "AbortError");
      this.worker.postMessage({ kind: "prepare", requestId, request: handoff.request }, handoff.transfer);
    } catch (error) {
      this.pending.get(requestId)?.reject(error);
      this.pending.delete(requestId);
    }
    return promise.then((prepared) => {
      if (signal.aborted) throw new DOMException("Volume brick preparation cancelled", "AbortError");
      return prepared;
    });
  }

  async prepareAnomalyBatch(
    requests: readonly ScalarPreparationRequest[],
    signal: AbortSignal,
  ): Promise<PreparedVtkImage[]> {
    await this.ready;
    if (this.closed) throw new Error("Volume brick worker is closed");
    if (signal.aborted) throw new DOMException("Volume brick batch cancelled", "AbortError");
    if (requests.length === 0) return [];
    const requestId = ++this.sequence;
    const promise = new Promise<PreparedVtkImage[]>((resolve, reject) =>
      this.pendingBatches.set(requestId, { resolve, reject }));
    void promise.catch(() => undefined);
    try {
      const handoffs = [];
      for (const request of requests) {
        handoffs.push(await this.handoff(
          { ...request, scalarTransform: "none" },
          () => signal.aborted || this.closed,
        ));
      }
      if (signal.aborted) throw new DOMException("Volume brick batch cancelled", "AbortError");
      if (this.closed) throw new DOMException("Volume brick worker closed", "AbortError");
      this.worker.postMessage({
        kind: "prepare-anomaly-batch",
        requestId,
        requests: handoffs.map((handoff) => handoff.request as ScalarPreparationRequest),
      }, handoffs.flatMap((handoff) => handoff.transfer));
    } catch (error) {
      this.pendingBatches.get(requestId)?.reject(error);
      this.pendingBatches.delete(requestId);
    }
    return promise.then((prepared) => {
      if (signal.aborted) throw new DOMException("Volume brick batch cancelled", "AbortError");
      return prepared;
    });
  }

  shutdown(): void {
    if (this.closed) return;
    const busy = this.pending.size > 0 || this.pendingBatches.size > 0;
    this.closed = true;
    this.fail(new DOMException("Volume brick worker closed", "AbortError"), false);
    if (busy) { this.terminateWorker(); return; }
    // Terminating after requesting self-close races WebKit's module teardown.
    // Idle workers acknowledge their own close; busy or unresponsive workers
    // are terminated directly so cancellation remains bounded.
    this.startupCloseTimer = setTimeout(() => this.terminateWorker(), 5000);
    this.worker.postMessage({ kind: "close" });
  }

  private terminateWorker(force = true): void {
    if (this.terminated) return;
    this.terminated = true;
    if (this.startupCloseTimer != null) clearTimeout(this.startupCloseTimer);
    this.startupCloseTimer = null;
    this.worker.onmessage = null;
    this.worker.onerror = (event) => event.preventDefault?.();
    this.worker.onmessageerror = null;
    if (force) this.worker.terminate();
  }

  private message(reply: VolumeBrickWorkerReply): void {
    if (reply.kind === "closed") {
      if (this.closed) this.terminateWorker(false);
      return;
    }
    if (reply.kind === "ready") {
      if (this.closed) return;
      this.readyResolve();
      return;
    }
    const pending = this.pending.get(reply.requestId);
    if (reply.kind === "prepared-batch") {
      const batch = this.pendingBatches.get(reply.requestId);
      if (!batch) return;
      this.pendingBatches.delete(reply.requestId);
      if (reply.ok) batch.resolve(reply.prepared);
      else batch.reject(new Error(reply.error));
      return;
    }
    if (!pending) return;
    this.pending.delete(reply.requestId);
    if (reply.ok) pending.resolve(reply.prepared);
    else pending.reject(new Error(reply.error));
  }

  private fail(error: Error, close = true): void {
    this.readyReject(error);
    for (const pending of this.pending.values()) pending.reject(error);
    for (const pending of this.pendingBatches.values()) pending.reject(error);
    this.pending.clear();
    this.pendingBatches.clear();
    if (close) this.shutdown();
  }
}
