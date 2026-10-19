import assert from "node:assert/strict";
import { radarFloorPresentationOwnership, selectRadarFloorRequestState } from "../../../../ui/src/lib/vtk/radarFloorOwnership.ts";

let passed = 0;
const failures: string[] = [];
function test(label: string, run: () => void): void {
  try { run(); passed++; }
  catch (error) { failures.push(`${label}: ${error instanceof Error ? error.message : String(error)}`); }
}
const pixels = new Float32Array([12, 24, 36, 48]);
const completed = { requestKey: "frame-23/source-a", loadedRequestIdentity: "frame-23/source-a",
  loading: false, error: null as string | null, values: pixels, stats: { min: 12, max: 48 } };
const idle = { requestKey: "idle", loadedRequestIdentity: null as string | null,
  loading: false, error: null as string | null, values: null as Float32Array | null,
  stats: null as { min: number; max: number } | null };
const attached = { enabled: true, requestedRequestIdentity: completed.requestKey,
  loadedRequestIdentity: completed.loadedRequestIdentity, requestLoading: false, encodingLoading: false,
  encodingRequired: true, encodingReady: true, gpuRequired: true, gpuAdmitted: true,
  attachedLifecycleGeneration: 7 as number | null,
  lifecycleGeneration: 7, error: null as string | null };

test("complete source and attached cached pixels are eligible for one coherent commit", () => {
  assert.deepEqual(radarFloorPresentationOwnership(attached), {
    currentRequest: true, attached: true, loading: false, action: "commit",
  });
});

test("rapid disable synchronously retires values and forbids cached-pixel recommits before effects", () => {
  let state: typeof idle = completed;
  for (let toggle = 0; toggle < 12; toggle++) {
    const disabled = selectRadarFloorRequestState(state, `disabled-${toggle}`, false, idle);
    assert.equal(disabled.values, null);
    assert.equal(disabled.loadedRequestIdentity, null);
    assert.equal(disabled.loading, false);
    const ownership = radarFloorPresentationOwnership({ ...attached, enabled: false,
      requestedRequestIdentity: disabled.requestedRequestIdentity, loadedRequestIdentity: disabled.loadedRequestIdentity });
    assert.equal(ownership.action, "hide");
    assert.equal(ownership.loading, false);
    state = completed;
  }
});

test("disabling an errored request reports idle instead of a retired source error", () => {
  const disabled = selectRadarFloorRequestState({ ...completed, error: "retired failure" }, "off", false, idle);
  assert.equal(disabled.error, null);
  assert.equal(disabled.values, null);
  assert.equal(radarFloorPresentationOwnership({ ...attached, enabled: false, error: "retired failure" }).action, "hide");
});

for (const requested of ["frame-24/source-a", "frame-23/source-b", "frame-23/revised-publication", "frame-23/column-max"]) {
  test(`${requested}: a pending enabled request retains the complete prior sheet with its old identity`, () => {
    const next = selectRadarFloorRequestState(completed, requested, true, idle);
    assert.strictEqual(next.values, pixels);
    assert.equal(next.loadedRequestIdentity, completed.loadedRequestIdentity);
    assert.equal(next.requestedRequestIdentity, requested);
    assert.equal(next.loading, true);
    assert.equal(next.error, null);
    const ownership = radarFloorPresentationOwnership({ ...attached,
      requestedRequestIdentity: next.requestedRequestIdentity, loadedRequestIdentity: next.loadedRequestIdentity,
      requestLoading: next.loading });
    assert.equal(ownership.currentRequest, false);
    assert.equal(ownership.loading, true);
    assert.equal(ownership.action, "retain");
  });
}

test("current-key loading cannot borrow a retained previous identity", () => {
  const next = selectRadarFloorRequestState({ ...completed, requestKey: "frame-24/source-a", loading: true },
    "frame-24/source-a", true, idle);
  assert.strictEqual(next.values, pixels);
  assert.equal(radarFloorPresentationOwnership({ ...attached,
    requestedRequestIdentity: next.requestedRequestIdentity, loadedRequestIdentity: next.loadedRequestIdentity,
    requestLoading: next.loading }).action, "retain");
});

test("obsolete source errors cannot poison an enabled replacement request", () => {
  const next = selectRadarFloorRequestState({ ...completed, error: "source-a failed" }, "frame-23/source-b", true, idle);
  assert.equal(next.error, null);
  assert.equal(next.loading, true);
  assert.strictEqual(next.values, pixels);
});

test("cached rgba cannot settle a lifecycle rebuild before the new actor attaches", () => {
  for (const attachedLifecycleGeneration of [7, null]) {
    const rebuilding = radarFloorPresentationOwnership({ ...attached, lifecycleGeneration: 8, attachedLifecycleGeneration });
    assert.equal(rebuilding.currentRequest, true);
    assert.equal(rebuilding.attached, false);
    assert.equal(rebuilding.loading, true);
    assert.equal(rebuilding.action, "retain");
  }
  const reattached = radarFloorPresentationOwnership({ ...attached, lifecycleGeneration: 8, attachedLifecycleGeneration: 8 });
  assert.equal(reattached.action, "commit");
  assert.equal(reattached.loading, false);
});

test("an older attachment cannot complete a second rebuild or a newer source", () => {
  const state = radarFloorPresentationOwnership({ ...attached, requestedRequestIdentity: "frame-24/source-b",
    loadedRequestIdentity: "frame-24/source-b", lifecycleGeneration: 9, attachedLifecycleGeneration: 8 });
  assert.equal(state.action, "retain");
  assert.equal(state.loading, true);
});

test("cached pixels and a rebuilt actor still wait for their current GPU reservation", () => {
  const waiting = radarFloorPresentationOwnership({ ...attached, lifecycleGeneration: 8,
    attachedLifecycleGeneration: 8, gpuAdmitted: false });
  assert.equal(waiting.action, "retain");
  assert.equal(waiting.loading, true);
});

test("current values wait for current encoding even when an actor is already attached", () => {
  for (const state of [{ encodingReady: false, encodingLoading: false }, { encodingReady: true, encodingLoading: true }]) {
    const ownership = radarFloorPresentationOwnership({ ...attached, ...state });
    assert.equal(ownership.action, "retain");
    assert.equal(ownership.loading, true);
  }
});

test("an all-missing current plane can settle transparently without inventing encoded pixels", () => {
  const empty = radarFloorPresentationOwnership({ ...attached, encodingRequired: false, encodingReady: false,
    gpuRequired: false, gpuAdmitted: false });
  assert.equal(empty.action, "commit");
  assert.equal(empty.loading, false);
  assert.equal(radarFloorPresentationOwnership({ ...attached, encodingRequired: false,
    encodingReady: false, attachedLifecycleGeneration: null }).action, "retain");
});

test("a correctly encoded transparent plane does not require a GPU allocation to settle", () => {
  const transparent = radarFloorPresentationOwnership({ ...attached, gpuRequired: false, gpuAdmitted: false });
  assert.equal(transparent.action, "commit");
  assert.equal(transparent.loading, false);
});

for (const error of ["current read failed", "current encoding failed", "current attachment failed"]) {
  test(`${error}: failure retains the previous complete scene and reports a settled error`, () => {
    const failed = radarFloorPresentationOwnership({ ...attached, error });
    assert.equal(failed.action, "retain");
    assert.equal(failed.loading, false);
  });
}

test("an unowned initial request cannot be presented as completed", () => {
  assert.equal(radarFloorPresentationOwnership({ ...attached, requestedRequestIdentity: null,
    loadedRequestIdentity: null }).action, "retain");
});

console.log(`Radar floor ownership: ${passed} behavioral cases passed, ${failures.length} failed.`);
for (const failure of failures) console.error(failure);
if (failures.length) process.exitCode = 1;
