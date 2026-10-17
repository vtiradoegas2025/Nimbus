import assert from "node:assert/strict";
import { fullPlaybackScreenResolution, observePlaybackGpuDuration } from "../../../../ui/src/lib/vtk/playbackScreenResolutionPolicy.ts";
import { attachVtkPlaybackScreenResolution, holdFullVtkPlaybackResolution, lookupVtkPlaybackScreenResolution } from "../../../../ui/src/lib/vtk/vtkPlaybackScreenResolution.ts";
import { createVtkGpuRenderTimer, type GpuRenderTimer } from "../../../../ui/src/lib/vtk/vtkGpuRenderTimer.ts";
import { beginVtkRenderTraversal } from "../../../../ui/src/lib/vtk/vtkRenderTraversal.ts";

let state = fullPlaybackScreenResolution();
for (let index = 0; index < 2; index++) state = observePlaybackGpuDuration(state, 40, 16.67);
assert.equal(state.distance, 1);
state = observePlaybackGpuDuration(state, 40, 16.67);
assert.equal(state.distance, 1.25);
for (let index = 0; index < 5; index++) state = observePlaybackGpuDuration(state, 40, 16.67);
assert.equal(state.distance, 1.25);
for (let index = 0; index < 20; index++) state = observePlaybackGpuDuration(state, 40, 16.67);
assert.equal(state.distance, 2);
assert.ok(state.recentMs.length <= 5);
for (let index = 0; index < 40; index++) state = observePlaybackGpuDuration(state, index % 2 ? 18 : 17, 16.67);
assert.equal(state.distance, 2);
for (let index = 0; index < 45; index++) state = observePlaybackGpuDuration(state, 1, 16.67);
assert.equal(state.distance, 1);
for (const invalid of [NaN, Infinity, -1, 0, 10_001]) assert.equal(observePlaybackGpuDuration(state, invalid, 16.67), state);
assert.equal(observePlaybackGpuDuration(state, 40, NaN), state);

let callback: ((value: number) => void) | null = null;
let timing: ReturnType<GpuRenderTimer["status"]> = "available";
let cancels = 0, disposals = 0, renders = 0;
const timer: GpuRenderTimer = { begin: () => timing === "available", end: next => { callback = next; },
  cancel: () => { cancels++; callback = null; }, dispose: () => { disposals++; callback = null; }, status: () => timing };
let pixels = 1;
const source = new Float32Array([0, 1, -3, 7]);
const mapper = { getImageSampleDistance: () => pixels, setImageSampleDistance: (next: number) => { pixels = next; },
  getSampleDistance: () => 0.5, source };
const originalSetter = mapper.setImageSampleDistance;
const view = {};
const core = Object.freeze({ render: () => { const end = beginVtkRenderTraversal(view); renders++; end(); } });
const originalRender = core.render;
const window = { getRenderWindow: () => core, getApiSpecificRenderWindow: () => view };
let activeMappers = [mapper];
const controller = attachVtkPlaybackScreenResolution(window, { getMappers: () => activeMappers, timer });
assert.equal(lookupVtkPlaybackScreenResolution(window), controller);
controller.setPlayback({ playing: true, recording: false, intervalMs: 16.67 });
for (let index = 0; index < 9; index++) { core.render(); callback!(40); }
assert.equal(pixels, 1.5);
assert.equal(mapper.getSampleDistance(), 0.5);
assert.equal(mapper.source, source);
assert.deepEqual([...source], [0, 1, -3, 7]);
controller.setPlayback({ playing: true, recording: false, intervalMs: 33.33 });
assert.equal(pixels, 1.5);
controller.setPlayback({ playing: true, recording: false, intervalMs: 16.67 });
mapper.setImageSampleDistance(2);
core.render();
assert.equal(pixels, 2);
mapper.setImageSampleDistance(1);
core.render();
assert.equal(pixels, 1.5);
const release = holdFullVtkPlaybackResolution(window), releaseNested = holdFullVtkPlaybackResolution(window);
assert.equal(pixels, 1);
core.render(); assert.equal(callback, null);
release(); assert.equal(pixels, 1);
releaseNested(); assert.equal(pixels, 1.5);
releaseNested(); assert.equal(pixels, 1.5);
timing = "disjoint"; core.render(); assert.equal(pixels, 1);
timing = "available";
for (let index = 0; index < 3; index++) { core.render(); callback!(40); }
controller.setPlayback({ playing: true, recording: true, intervalMs: 16.67 });
assert.equal(pixels, 1);
assert.equal(controller.describe().playing, false);
controller.setPlayback({ playing: true, recording: false, intervalMs: 16.67 });
for (let index = 0; index < 3; index++) { core.render(); callback!(40); }
controller.setPlayback({ playing: false, recording: false, intervalMs: 16.67 });
assert.equal(pixels, 1);
activeMappers = []; core.render(); assert.equal(mapper.setImageSampleDistance, originalSetter);
controller.dispose(); controller.dispose();
assert.equal(disposals, 1);
assert.equal(core.render, originalRender);
assert.equal(lookupVtkPlaybackScreenResolution(window), null);
assert.ok(cancels > 0 && renders > 0);

const callbacks = new Map<number, FrameRequestCallback>();
let nextFrame = 0;
globalThis.requestAnimationFrame = next => { callbacks.set(++nextFrame, next); return nextFrame; };
globalThis.cancelAnimationFrame = id => { callbacks.delete(id); };
let available = false, disjoint = false, lost = false, deleted = 0, ended = 0;
const fakeGl = { CURRENT_QUERY: 1, QUERY_RESULT_AVAILABLE: 2, QUERY_RESULT: 3,
  isContextLost: () => lost, getExtension: () => ({ TIME_ELAPSED_EXT: 4, GPU_DISJOINT_EXT: 5 }),
  getParameter: () => disjoint, getQuery: () => null, createQuery: () => ({}), beginQuery: () => {},
  endQuery: () => { ended++; }, deleteQuery: () => { deleted++; },
  getQueryParameter: (_query: unknown, parameter: number) => parameter === 2 ? available : 42_000_000 } as unknown as WebGL2RenderingContext;
const gpuTimer = createVtkGpuRenderTimer(() => fakeGl);
let duration = 0;
assert.equal(gpuTimer.begin(), true); gpuTimer.end(value => { duration = value; });
assert.equal(gpuTimer.begin(), false);
const poll = () => { const value = callbacks.entries().next().value!; callbacks.delete(value[0]); value[1](0); };
poll(); assert.equal(duration, 0); assert.equal(callbacks.size, 1);
available = true; poll();
assert.equal(duration, 42); assert.equal(deleted, 1); assert.equal(ended, 1); assert.equal(callbacks.size, 0);
assert.equal(gpuTimer.status(), "available");
assert.equal(gpuTimer.begin(), true); gpuTimer.end(() => { throw new Error("Disjoint samples must be discarded"); });
disjoint = true; poll(); assert.equal(gpuTimer.status(), "disjoint"); assert.equal(callbacks.size, 0);
disjoint = false; assert.equal(gpuTimer.begin(), true); gpuTimer.end(() => { throw new Error("Lost contexts cannot provide timing"); });
lost = true; poll(); assert.equal(callbacks.size, 0);
gpuTimer.dispose(); assert.equal(gpuTimer.begin(), false);
const unavailable = createVtkGpuRenderTimer(() => null);
assert.equal(unavailable.begin(), false); assert.equal(unavailable.status(), "unavailable"); unavailable.dispose();
console.log("Playback screen resolution: bounded hysteresis, source/ray invariants, capture/restoration, mapper retirement and asynchronous GPU timer tests passed.");
