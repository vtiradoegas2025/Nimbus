import assert from "node:assert/strict";
import {
  disposePythonRuntimeClient, getPythonRuntimeClient, PyodideFatalError,
  PythonStartupError, PythonStoppedError, PYTHON_BOOT_ATTEMPT_TIMEOUT_MS,
  PYTHON_BOOT_TOTAL_TIMEOUT_MS,
} from "../../../../ui/src/lib/platform/pyodideRuntime.ts";
import { WORKER_OPS, type RpcMessage, type RpcRequest } from "../../../../ui/src/lib/platform/pyodideProtocol.ts";

const originalSetTimeout = globalThis.setTimeout;
const originalClearTimeout = globalThis.clearTimeout;
const originalWorker = globalThis.Worker;
let now = 0, nextTimer = 0;
const timers = new Map<number, { at: number; call: () => void }>();
globalThis.setTimeout = ((call: () => void, delay = 0) => {
  const id = ++nextTimer;
  timers.set(id, { at: now + delay, call });
  return id;
}) as unknown as typeof setTimeout;
globalThis.clearTimeout = ((id: number) => { timers.delete(id); }) as unknown as typeof clearTimeout;
const flush = async () => { for (let index = 0; index < 30; index++) await Promise.resolve(); };
async function advance(milliseconds: number): Promise<void> {
  const until = now + milliseconds;
  for (;;) {
    const next = [...timers].filter(([, timer]) => timer.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
    if (!next) break;
    now = next[1].at; timers.delete(next[0]); next[1].call();
    await flush();
  }
  now = until;
  await flush();
}

type BootPlan = "ok" | "hang" | "error" | "reject";
class FakeWorker {
  static all: FakeWorker[] = [];
  static plans = new Map<string, BootPlan[]>();
  readonly key: string;
  readonly url: string;
  readonly messages: RpcRequest[] = [];
  readonly bootPlan: BootPlan;
  terminated = false;
  failUserCell = false;
  private messageListener: ((event: { data: RpcMessage }) => void) | null = null;
  private errorListener: ((event: { message: string }) => void) | null = null;
  constructor(url: string) {
    this.url = url;
    this.key = new URL(url, "https://nimbus.test").searchParams.get("runtime")!;
    this.bootPlan = FakeWorker.plans.get(this.key)?.shift() ?? "ok";
    FakeWorker.all.push(this);
  }
  addEventListener(type: string, listener: unknown): void {
    if (type === "message") this.messageListener = listener as typeof this.messageListener;
    if (type === "error") this.errorListener = listener as typeof this.errorListener;
  }
  postMessage(message: RpcRequest): void {
    this.messages.push(message);
    if (message.op === WORKER_OPS.boot && this.bootPlan === "hang") return;
    queueMicrotask(() => {
      if ((message.op === WORKER_OPS.boot && this.bootPlan === "error")
          || (message.op === WORKER_OPS.runCell && this.failUserCell)) {
        this.errorListener?.({ message: "worker failed" }); return;
      }
      if (message.op === WORKER_OPS.boot && this.bootPlan === "reject") {
        this.emit({ kind: "response", id: message.id, ok: false,
          error: { name: "Error", message: "package initialization failed", fatal: true } }); return;
      }
      this.reply(message);
    });
  }
  reply(message: RpcRequest): void { this.emit({ kind: "response", id: message.id, ok: true, value: null }); }
  emit(message: RpcMessage): void { this.messageListener?.({ data: message }); }
  terminate(): void { this.terminated = true; }
}
Object.assign(globalThis, { Worker: FakeWorker });
const keys = new Set<string>();
function client(key: string, plans: BootPlan[] = []) {
  keys.add(key); FakeWorker.plans.set(key, [...plans]);
  return getPythonRuntimeClient(key);
}
const workers = (key: string) => FakeWorker.all.filter(worker => worker.key === key);
function dispose(key: string): void { disposePythonRuntimeClient(key); keys.delete(key); }

try {
  const retry = client("boot-timeout-retry", ["hang", "ok"]);
  const first = retry.getPyodide();
  const concurrent = retry.getPyodide();
  await flush();
  assert.equal(workers(retry.key).length, 1, "concurrent callers share one startup transaction");
  await advance(PYTHON_BOOT_ATTEMPT_TIMEOUT_MS);
  const handle = await first;
  assert.equal(await concurrent, handle);
  const [retired, replacement] = workers(retry.key);
  assert(retired.terminated);
  assert.notEqual(retired.url, replacement.url);
  retired.emit({ kind: "event", event: "status", status: { kind: "error", message: "stale" } });
  retired.reply(retired.messages[0]);
  await retry.runPythonCell(handle, "side_effect_once()");
  assert.deepEqual(retired.messages.map(message => message.op), [WORKER_OPS.boot]);
  assert.equal(replacement.messages.filter(message => message.op === WORKER_OPS.runCell).length, 1);
  assert.equal(timers.size, 0, "successful boot retires both deadline timers");
  dispose(retry.key);

  for (const failure of ["error", "reject"] as const) {
    const recovering = client(`boot-${failure}-retry`, [failure, "ok"]);
    await recovering.getPyodide();
    assert.equal(workers(recovering.key).length, 2);
    assert(workers(recovering.key)[0].terminated);
    assert.equal(recovering.pythonNeedsRestart(), false);
    dispose(recovering.key);
  }

  const exhausted = client("boot-final-error", ["hang", "hang"]);
  const failure = exhausted.getPyodide().catch(error => error);
  await flush();
  await advance(2 * PYTHON_BOOT_ATTEMPT_TIMEOUT_MS);
  assert(await failure instanceof PythonStartupError);
  assert.equal(workers(exhausted.key).length, 2, "automatic retries stop after two attempts");
  assert(workers(exhausted.key).every(worker => worker.terminated));
  assert.equal(exhausted.isBusy(), false);
  assert.equal(exhausted.pythonNeedsRestart(), false, "failed cold startup has no live namespace to discard");
  await exhausted.getPyodide();
  assert.equal(workers(exhausted.key).length, 3, "a later explicit request can recover");
  dispose(exhausted.key);

  const blocker = client("boot-fair-blocker", ["hang", "ok"]);
  const held = blocker.getPyodide();
  const queued = client("boot-cancelled-queue");
  const cancelled = queued.getPyodide().catch(error => error);
  await flush();
  assert.equal(queued.isBusy(), true, "queued startup cannot be evicted as idle");
  queued.restartPython();
  await flush();
  assert(await cancelled instanceof PythonStoppedError, "queued cancellation does not await the previous worker");
  assert.equal(workers(queued.key).length, 0);
  const freshQueued = queued.getPyodide();
  const start = FakeWorker.all.length;
  await advance(PYTHON_BOOT_ATTEMPT_TIMEOUT_MS);
  await Promise.all([held, freshQueued]);
  assert.deepEqual(FakeWorker.all.slice(start).map(worker => worker.key), [queued.key, blocker.key],
    "a failed attempt yields the boot slot before retrying");
  dispose(blocker.key); dispose(queued.key);

  const stopped = client("boot-explicit-stop", ["hang"]);
  const stopResult = stopped.getPyodide().catch(error => error);
  await flush();
  assert(stopped.interruptPython());
  assert(await stopResult instanceof PythonStoppedError);
  await advance(PYTHON_BOOT_TOTAL_TIMEOUT_MS);
  assert.equal(workers(stopped.key).length, 1, "Stop cannot trigger an automatic replacement");
  const oldUrl = workers(stopped.key)[0].url;
  stopped.dispose();
  keys.delete(stopped.key);
  await assert.rejects(stopped.getPyodide(), PythonStoppedError);
  const recreated = client(stopped.key);
  await recreated.getPyodide();
  assert.notEqual(workers(stopped.key).at(-1)!.url, oldUrl, "recreating the same key never reuses a module-worker URL");
  dispose(recreated.key);

  const live = client("boot-live-namespace");
  const liveHandle = await live.getPyodide();
  const liveWorker = workers(live.key)[0];
  liveWorker.failUserCell = true;
  await assert.rejects(live.runPythonCell(liveHandle, "persist_side_effect()"), PyodideFatalError);
  await assert.rejects(live.getPyodide(), PyodideFatalError);
  await advance(PYTHON_BOOT_TOTAL_TIMEOUT_MS);
  assert.equal(workers(live.key).length, 1, "a live interpreter is never automatically replaced");
  assert.equal(liveWorker.messages.filter(message => message.op === WORKER_OPS.runCell).length, 1);
  dispose(live.key);

  const slowA = client("boot-total-a", ["hang", "hang"]);
  const slowB = client("boot-total-b", ["hang", "hang"]);
  const results = [slowA.getPyodide().catch(error => error), slowB.getPyodide().catch(error => error)];
  await flush();
  await advance(PYTHON_BOOT_TOTAL_TIMEOUT_MS);
  for (const result of await Promise.all(results)) assert(result instanceof PythonStartupError);
  assert.equal(slowA.isBusy(), false);
  assert.equal(slowB.isBusy(), false);
  const after = client("boot-after-total-deadline");
  await after.getPyodide();
  assert.equal(after.isBusy(), false, "the global boot queue recovers after total deadlines");
  dispose(slowA.key); dispose(slowB.key); dispose(after.key);
  assert.equal(timers.size, 0);
  console.log("Python startup deadlines, retry fairness, cancellation, URL isolation and no user-code replay: passed");
} finally {
  for (const key of keys) disposePythonRuntimeClient(key);
  globalThis.setTimeout = originalSetTimeout;
  globalThis.clearTimeout = originalClearTimeout;
  Object.assign(globalThis, { Worker: originalWorker });
}
