import assert from "node:assert/strict";
import { PythonArrayExportService, type PythonArrayExportWorker } from "../../../../ui/src/lib/platform/pythonArrayExportClient.ts";
import { MAX_PYTHON_ARRAY_EXPORT_BYTES, type PythonArrayExportRequest, type PythonArrayExportReply } from "../../../../ui/src/lib/platform/pythonArrayExportProtocol.ts";
import { encodeCsv, encodeNpy } from "../../../../ui/src/lib/platform/pythonExport.ts";

class FakeWorker implements PythonArrayExportWorker {
  static urls: string[] = [];
  static defaults: FakeWorker[] = [];
  onmessage: ((event: MessageEvent<PythonArrayExportReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  request: PythonArrayExportRequest | null = null;
  terminated = false;
  failPost = false;
  constructor(url?: string) {
    if (url != null) { FakeWorker.urls.push(url); FakeWorker.defaults.push(this); }
  }
  postMessage(request: PythonArrayExportRequest): void {
    if (this.failPost) throw new DOMException("clone failed", "DataCloneError");
    this.request = structuredClone(request);
  }
  terminate(): void { this.terminated = true; }
  reply(id = this.request!.id): void {
    const request = this.request!;
    const bytes = request.format === "npy" ? encodeNpy(request.data, request.shape, request.dtype)
      : new TextEncoder().encode(encodeCsv(request.data, request.shape, request.dtype));
    this.onmessage?.({ data: { id, bytes } } as MessageEvent<PythonArrayExportReply>);
  }
}
const workers: FakeWorker[] = [];
const service = new PythonArrayExportService(() => { const worker = new FakeWorker(); workers.push(worker); return worker; });
const source = Float64Array.of(1, 1 + 1e-12);
const exported = service.encode(source, [2], "csv");
assert.equal(source.byteLength, 16, "retained output storage stays attached");
await assert.rejects(service.encode(source, [2], "csv"), /still encoding/, "concurrent exports cannot clone an unbounded queue of inputs");
const first = workers.at(-1)!;
assert.notEqual(first.request!.data.buffer, source.buffer);
first.reply();
assert.equal(new TextDecoder().decode(await exported), "1\n1.000000000001\n");
assert(first.terminated);
assert.equal(source[1], 1 + 1e-12);

const parent = Float32Array.of(99, 2.5, NaN, 88);
const cropped = service.encode(parent.subarray(1, 3), [2], "npy");
const cropWorker = workers.at(-1)!;
assert.equal(cropWorker.request!.data.buffer.byteLength, 8, "a view cannot clone its entire unrelated backing allocation");
assert.deepEqual(cropWorker.request!.data, Float32Array.of(2.5, NaN));
cropWorker.reply();
await cropped;
assert.equal(parent.byteLength, 16);

const controller = new AbortController();
const cancelled = service.encode(source, [2], "npy", undefined, controller.signal);
const cancelledWorker = workers.at(-1)!;
controller.abort();
await assert.rejects(cancelled, error => error instanceof DOMException && error.name === "AbortError");
assert(cancelledWorker.terminated);
const restarted = service.encode(source, [2], "npy");
cancelledWorker.reply();
workers.at(-1)!.reply();
await restarted;

const wrongReply = service.encode(source, [2], "npy");
workers.at(-1)!.reply(99_999);
await assert.rejects(wrongReply, /identity mismatch/);
assert(workers.at(-1)!.terminated);
const failingWorker = new FakeWorker(); failingWorker.failPost = true;
const failingService = new PythonArrayExportService(() => failingWorker);
await assert.rejects(failingService.encode(source, [2], "npy"), /clone failed/);
assert(failingWorker.terminated);
const count = workers.length;
await assert.rejects(service.encode(new Uint8Array(MAX_PYTHON_ARRAY_EXPORT_BYTES + 1), [MAX_PYTHON_ARRAY_EXPORT_BYTES + 1], "npy"), /64 MiB/);
await assert.rejects(service.encode(Float64Array.of(1), [1], "npy", "<i8"), /dtype/);
assert.equal(workers.length, count, "invalid or oversized arrays are rejected before a worker receives a copy");
Object.assign(globalThis, { Worker: FakeWorker });
const defaultService = new PythonArrayExportService();
for (const format of ["npy", "csv", "npy"] as const) {
  const pending = defaultService.encode(source, [2], format);
  FakeWorker.defaults.at(-1)!.reply();
  await pending;
  assert(FakeWorker.defaults.at(-1)!.terminated);
}
assert.equal(new Set(FakeWorker.urls).size, 3, "worker retirement cannot reuse a conditional-cache URL on the next export");
assert(FakeWorker.urls.every(url => /session=.*&generation=\d+/.test(url)));
console.log("Python array export ownership, budgets, cancellation, stale replies and failure cleanup: passed");
