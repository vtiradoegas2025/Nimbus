import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { volumeTextureCapabilities } from "../../../../ui/src/lib/vtk/volumeTextureCapabilities.ts";
import { VolumeSamplingShaderMemo, volumeSamplingShader } from "../../../../ui/src/lib/vtk/volumeSamplingShader.ts";
import { planVolumeTextureLayout } from "../../../../ui/src/lib/vtk/volumeTexturePacking.ts";

function viewer(maximum = 8, linear = false) {
  const canvas = new EventTarget();
  const state = { maximum, linear, lost: false, queries: 0, extensions: 0 };
  const gl = { canvas, MAX_3D_TEXTURE_SIZE: 32883, isContextLost: () => state.lost,
    getParameter(parameter: number) { assert.equal(parameter, 32883); state.queries++; return state.maximum; },
    getExtension(name: string) { assert.equal(name, "OES_texture_float_linear"); state.extensions++; return state.linear ? {} : null; },
  } as unknown as WebGL2RenderingContext;
  return { canvas, state, gl };
}

for (const panes of [1, 2, 4]) {
  const viewers = Array.from({ length: panes }, (_, index) => viewer(index % 2 ? 16 : 8, Boolean(index % 2)));
  const initial = viewers.map(({ gl }) => volumeTextureCapabilities(gl));
  for (let frame = 0; frame < 100; frame++) for (let input = 0; input < 4; input++) {
    viewers.forEach(({ gl }, index) => assert.equal(volumeTextureCapabilities(gl), initial[index]));
  }
  viewers.forEach(({ state }) => assert.deepEqual([state.queries, state.extensions], [1, 1]));
  const first = viewers[0];
  first.canvas.dispatchEvent(new Event("webglcontextlost")); first.state.lost = true;
  assert.throws(() => volumeTextureCapabilities(first.gl), /available WebGL2/);
  first.state.maximum = 4; first.state.linear = true; first.state.lost = false;
  first.canvas.dispatchEvent(new Event("webglcontextrestored"));
  const restored = volumeTextureCapabilities(first.gl);
  assert.deepEqual(restored, { maximumDimension: 4, maximumCells: 64, floatLinear: true });
  assert.notEqual(restored, initial[0]);
  assert.deepEqual([first.state.queries, first.state.extensions], [2, 2]);
}
const invalid = viewer(1);
assert.throws(() => volumeTextureCapabilities(invalid.gl), /valid viewer/);
assert.equal(invalid.state.extensions, 0);
invalid.state.maximum = 16;
assert.equal(volumeTextureCapabilities(invalid.gl).maximumDimension, 16);

const layouts = [planVolumeTextureLayout([2, 3, 41], 8), planVolumeTextureLayout([4, 4, 4], 8),
  planVolumeTextureLayout([3, 2, 41], 8), planVolumeTextureLayout([2, 2, 2], 8)];
const filters = [true, true, true, false];
const options = { validity: [true, false, true, false], nearest: [false, false, false, true] };
const memo = new VolumeSamplingShaderMemo();
const reference = volumeSamplingShader(layouts, filters, options);
const first = memo.get(layouts, filters, options);
assert.deepEqual(first, reference);
const started = performance.now();
for (let frame = 0; frame < 5000; frame++) {
  assert.equal(memo.get([...layouts], [...filters], { validity: [...options.validity], nearest: [...options.nearest] }), first);
}
const memoMs = performance.now() - started;
const baselineStarted = performance.now();
for (let frame = 0; frame < 5000; frame++) volumeSamplingShader(layouts, filters, options);
const regeneratedMs = performance.now() - baselineStarted;
for (const changes of [
  { validity: [false, false, true, false], nearest: options.nearest },
  { validity: options.validity, nearest: [true, false, false, true] },
]) {
  const changed = memo.get(layouts, filters, changes);
  assert.notEqual(changed, first);
  assert.deepEqual(changed, volumeSamplingShader(layouts, filters, changes));
}
const reshaped = [planVolumeTextureLayout([3, 2, 41], 8), ...layouts.slice(1)];
assert.deepEqual(memo.get(reshaped, filters, options), volumeSamplingShader(reshaped, filters, options));
assert.notEqual(memo.get(reshaped, filters, options).key, first.key);
const hardware = memo.get(layouts, [false, false, false, false], options);
assert.deepEqual(hardware, volumeSamplingShader(layouts, [false, false, false, false], options));
assert.notEqual(hardware.key, first.key);
const nativeMemo = new VolumeSamplingShaderMemo();
const nativeOptions = { validity: [false], nearest: [false] };
const native = nativeMemo.get([planVolumeTextureLayout([2, 3, 4], 16)], [false], nativeOptions);
assert.equal(nativeMemo.get([planVolumeTextureLayout([7, 6, 5], 16)], [false], nativeOptions), native,
  "native ROI shape changes remain uniform-only and reuse exact shader source");
console.log(JSON.stringify({ panes: [1, 2, 4], inputsPerPane: 4, renderedFrames: 100,
  capabilityQueriesPerContext: 1, shaderChecks: 5000, generatedWarmSourceObjects: 1, memoMs, regeneratedMs }));
