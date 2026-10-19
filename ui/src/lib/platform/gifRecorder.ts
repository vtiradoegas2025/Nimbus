import { captureFrameDataUri, dataUriToImageData, type CaptureSize } from "@/lib/run/frameCapture";
import { assertCaptureCurrent, type CaptureGuard } from "@/lib/run/captureGuard";
import { validateGifSize } from "./gifLimits";
import type { GifReply, GifRequest } from "./gifEncoding.worker";
import workerUrl from "./gifEncoding.worker.ts?worker&url";

export interface GifRecorderOptions { fps: number }

export class GifRecorder {
  private worker: Worker | null = null;
  private pending: ((error: Error) => void) | null = null;
  private count = 0;
  private disposed = false;
  private readonly fps: number;
  private readonly size: CaptureSize;

  constructor(opts: GifRecorderOptions & CaptureSize) {
    validateGifSize(opts.width, opts.height);
    if (!Number.isFinite(opts.fps)) throw new Error("Invalid GIF frame rate");
    this.fps = Math.max(1, Math.min(60, opts.fps));
    this.size = { width: opts.width, height: opts.height };
  }

  async captureFrame(el: HTMLElement, guard: CaptureGuard = {}): Promise<void> {
    const uri = await captureFrameDataUri(el, this.size, guard);
    await this.writeDataUriFrame(uri, guard);
  }

  async writeDataUriFrame(uri: string, guard: CaptureGuard = {}): Promise<void> {
    assertCaptureCurrent(guard);
    const frame = await dataUriToImageData(uri, this.size, guard);
    assertCaptureCurrent(guard);
    await this.request({ kind: "frame", ...this.size, fps: this.fps, rgba: frame.data }, [frame.data.buffer], guard.signal);
    assertCaptureCurrent(guard);
    this.count++;
  }

  frames(): number { return this.count; }

  async finish(signal?: AbortSignal): Promise<Uint8Array> {
    try {
      const result = await this.request({ kind: "finish" }, [], signal);
      if (!result.bytes) throw new Error("GIF encoder returned no bytes");
      return result.bytes;
    } finally { this.dispose(); }
  }

  dispose(): void {
    this.disposed = true;
    this.pending?.(new DOMException("GIF encoding cancelled", "AbortError"));
    this.worker?.terminate();
    this.worker = null;
  }

  private request(message: GifRequest, transfer: Transferable[], signal?: AbortSignal): Promise<Extract<GifReply, { ok: true }>> {
    if (this.disposed || signal?.aborted) return Promise.reject(new DOMException("GIF encoding cancelled", "AbortError"));
    if (this.pending) return Promise.reject(new Error("Wait for the previous GIF frame to finish"));
    const worker = this.worker ??= new Worker(workerUrl, { type: "module", name: "nimbus-gif-encoder" });
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timeout);
        signal?.removeEventListener("abort", abort);
        worker.onmessage = worker.onerror = worker.onmessageerror = null;
        this.pending = null;
      };
      const fail = (error: Error) => { cleanup(); worker.terminate(); this.worker = null; this.disposed = true; reject(error); };
      const abort = () => fail(new DOMException("GIF encoding cancelled", "AbortError"));
      const timeout = setTimeout(() => fail(new Error("GIF encoding timed out; choose a smaller export")), 120_000);
      this.pending = fail;
      signal?.addEventListener("abort", abort, { once: true });
      worker.onerror = event => { event.preventDefault(); fail(new Error("GIF encoding worker failed")); };
      worker.onmessageerror = () => fail(new Error("GIF encoding reply could not be read"));
      worker.onmessage = ({ data }: MessageEvent<GifReply>) => {
        if (!data.ok) { fail(new Error(data.error)); return; }
        cleanup(); resolve(data);
      };
      try { worker.postMessage(message, transfer); } catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
    });
  }
}
