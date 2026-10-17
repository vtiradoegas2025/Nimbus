import assert from "node:assert/strict";
import { assertCaptureCurrent, guardedCapture, gifFrameDelayMs } from "../../../../ui/src/lib/run/captureGuard.ts";
import { waitForFrameReady } from "../../../../ui/src/lib/run/frameReady.ts";

let current = 1;
let finish: (value: string) => void = () => {};
const captured = guardedCapture(() => new Promise<string>(resolve => { finish = resolve; }), {
  assertCurrent: () => assert.equal(current, 1, "captured source remains the selected source"),
});
current = 2;
finish("old pixels");
await assert.rejects(captured, /captured source remains/);

const cancelled = new AbortController();
cancelled.abort();
assert.throws(() => assertCaptureCurrent({ signal: cancelled.signal }), { name: "AbortError" });
let invoked = false;
await assert.rejects(guardedCapture(async () => { invoked = true; return "pixels"; }, { signal: cancelled.signal }), { name: "AbortError" });
assert.equal(invoked, false, "cancelled export starts no graphics operation");

for (const fps of [2, 5, 10, 15, 24, 30, 60]) {
  let duration = 0;
  for (let frame = 0; frame < fps * 10; frame++) {
    const delay = gifFrameDelayMs(frame, fps);
    assert.equal(delay % 10, 0, "GIF stores whole centiseconds");
    assert.ok(delay > 0);
    duration += delay;
    assert.ok(Math.abs(duration - (frame + 1) * 1000 / fps) <= 5.00000001, "quantized timing does not accumulate drift");
  }
  assert.equal(duration, 10_000, `${fps} fps retains exact total clip duration`);
}
assert.deepEqual(Array.from({ length: 6 }, (_, i) => gifFrameDelayMs(i, 30)), [30, 40, 30, 30, 40, 30]);
assert.throws(() => gifFrameDelayMs(-1, 30));
assert.throws(() => gifFrameDelayMs(0, 0));

const originalRequest = globalThis.requestAnimationFrame;
const originalCancel = globalThis.cancelAnimationFrame;
const scheduled = new Map<number, ReturnType<typeof setTimeout>>();
let sequence = 0;
let ready = true;
globalThis.requestAnimationFrame = callback => {
  const index = ++sequence;
  scheduled.set(index, setTimeout(() => {
    scheduled.delete(index);
    if (index === 3) ready = false;
    callback(performance.now());
  }, 1));
  return index;
};
globalThis.cancelAnimationFrame = index => { clearTimeout(scheduled.get(index)); scheduled.delete(index); };
try {
  assert.equal(await waitForFrameReady(0, 12, () => ready), false, "readiness is checked again after the final paint");
  const signal = new AbortController();
  const pending = waitForFrameReady(100, 1000, () => true, signal.signal);
  signal.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(scheduled.size, 0, "cancelled readiness retires its scheduled frame");
} finally {
  globalThis.requestAnimationFrame = originalRequest;
  globalThis.cancelAnimationFrame = originalCancel;
}
console.log("Capture guards, paint currentness, cancellation and deterministic GIF cadence passed.");
