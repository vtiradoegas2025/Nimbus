import assert from "node:assert/strict";
import { getPythonRuntimeClient, PythonStoppedError } from "../../../../ui/src/lib/platform/pyodideRuntime.ts";
import { WORKER_OPS, type RpcMessage, type RpcRequest } from "../../../../ui/src/lib/platform/pyodideProtocol.ts";

class FakeWorker {
  static all: FakeWorker[] = [];
  static holdBoot = false;
  readonly messages: RpcRequest[] = [];
  cloneFailure = false;
  private listener: ((event: { data: RpcMessage }) => void) | null = null;
  constructor() { FakeWorker.all.push(this); }
  addEventListener(name: string, listener: (event: { data: RpcMessage }) => void): void {
    if (name === "message") this.listener = listener;
  }
  postMessage(message: RpcRequest): void {
    if (this.cloneFailure) { this.cloneFailure = false; throw new DOMException("uncloneable argument", "DataCloneError"); }
    this.messages.push(message);
    if (message.op !== WORKER_OPS.runCell && !(message.op === WORKER_OPS.boot && FakeWorker.holdBoot)) queueMicrotask(() => this.reply(message));
  }
  reply(message: RpcRequest): void { this.emit({ kind: "response", id: message.id, ok: true, value: null }); }
  emit(message: RpcMessage): void { this.listener?.({ data: message }); }
  terminate(): void {}
}
Object.assign(globalThis, { Worker: FakeWorker });
const runtime = getPythonRuntimeClient("rpc-lifecycle-test");
const statuses: string[] = [];
const handle = await runtime.getPyodide(status => { statuses.push(status.kind); });
const firstWorker = FakeWorker.all.at(-1)!;
firstWorker.cloneFailure = true;
await assert.rejects(runtime.bindContext(handle, { fields: [], grid: null, simulationTimeS: null }), /uncloneable/);
assert.equal(runtime.isBusy(), false, "failed postMessage cannot retain a phantom pending RPC");
await runtime.bindContext(handle, { fields: [], grid: null, simulationTimeS: null });
const data = Float32Array.of(1, 2);
await runtime.bindContext(handle, { fields: [{ name: "w", field: { shape: [1, 1, 2], data,
  windowGeometry: { sourceShape: [1, 1, 2], origin: [0, 0, 0], extent: [1, 1, 2], stride: [1, 1, 1],
    weighting: "physical_cell_volume", sampleLocation: "physical_volume_centroid",
    centroidsM: { x: Float64Array.of(1, 1), y: Float64Array.of(3, 3), z: Float64Array.of(5, 6) } } } }], grid: null, simulationTimeS: null });
const binding = firstWorker.messages.filter(message => message.op === WORKER_OPS.bindContext).at(-1)!.args[0] as { fields: { field: { data: Float32Array; windowGeometry?: unknown } }[] };
assert.equal(binding.fields[0].field.windowGeometry, undefined, "unused dense geometry cannot be cloned into the interpreter context");
assert.equal(binding.fields[0].field.data, data);
assert.equal(data.byteLength, 8, "borrowed field buffers remain attached");

const old = runtime.runPythonCell(handle, "old cell").catch(error => error);
await new Promise<void>(resolve => setTimeout(resolve, 0));
runtime.restartPython();
const streams: string[] = [];
const fresh = runtime.runPythonCell(handle, "new cell", { onStream: (_, text) => streams.push(text), resultOnly: true });
assert(await old instanceof PythonStoppedError);
assert.equal(runtime.isCellInFlight(), true, "an old cell's completion cannot clear a new generation's execution flag");
await new Promise<void>(resolve => setTimeout(resolve, 0));
const newWorker = FakeWorker.all.at(-1)!;
assert.notEqual(newWorker, firstWorker);
firstWorker.emit({ kind: "event", event: "stream", stream: "stdout", text: "obsolete" });
firstWorker.emit({ kind: "event", event: "status", status: { kind: "error", message: "obsolete" } });
newWorker.emit({ kind: "event", event: "stream", stream: "stdout", text: "current" });
assert.deepEqual(streams, ["current"]);
assert.deepEqual(statuses, ["ready"]);
const cell = newWorker.messages.find(message => message.op === WORKER_OPS.runCell)!;
assert.equal((cell.args[1] as { resultOnly: boolean }).resultOnly, true);
newWorker.reply(cell);
await fresh;
assert.equal(runtime.isCellInFlight(), false);
assert.equal(runtime.isBusy(), false);
runtime.dispose();
FakeWorker.holdBoot = true;
const blocker = getPythonRuntimeClient("boot-blocker-test");
const heldBoot = blocker.getPyodide();
await new Promise<void>(resolve => setTimeout(resolve, 0));
const blockerWorker = FakeWorker.all.at(-1)!;
const queued = getPythonRuntimeClient("queued-boot-test");
const retiredBoot = queued.getPyodide().catch(error => error);
queued.restartPython();
const replacementBoot = queued.getPyodide();
const workerCount = FakeWorker.all.length;
FakeWorker.holdBoot = false;
blockerWorker.reply(blockerWorker.messages.find(message => message.op === WORKER_OPS.boot)!);
await heldBoot;
assert(await retiredBoot instanceof PythonStoppedError);
await replacementBoot;
assert.equal(FakeWorker.all.length, workerCount + 1, "a cancelled cold-boot slot cannot start a retired generation's worker");
assert.equal(queued.isBusy(), false);
blocker.dispose();
queued.dispose();
console.log("Pyodide RPC clone failure, worker replacement and result-only protocol: passed");
