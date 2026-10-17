import assert from "node:assert/strict";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume.js";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper.js";
import { createCloudMotionController } from "../../../../ui/src/lib/vtk/cloudMotionController.ts";
import { bindJointVolumes } from "../../../../ui/src/lib/vtk/jointVolumeRendering.ts";
import { visualizationResourceGovernor } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
Object.assign(globalThis, { cancelAnimationFrame: () => {} });
const image = prepareVtkRequest({ kind: "scalar", field: { shape: [12, 12, 12], data: new Float32Array(1728).fill(0.001) }, grid: { drM: 100, dyM: 100, dzM: 100 } });
const cloud = vtkVolume.newInstance(), context = vtkVolume.newInstance();
const cloudMapper = vtkVolumeMapper.newInstance(), contextMapper = vtkVolumeMapper.newInstance();
cloud.setMapper(cloudMapper); context.setMapper(contextMapper);
let draws = 0, starts = 0;
const motion = createCloudMotionController({ addVolume() {}, removeVolume() {} }, () => { draws++; },
  () => ({ fenceSync() {} }) as unknown as WebGL2RenderingContext, () => {});
const binding = bindJointVolumes([{ actor: cloud, role: "cloud" }, { actor: context, role: "context" }, ...motion.sources]);
motion.attach(binding);
const input = {
  frame: { key: "first", domainKey: "same", step: 0, timeS: 0, center: [550, 550, 550] as [number, number, number], cloud: image, context: null },
  enabled: true, durationMs: 500, scale: { xy: 1, z: 1 }, current: [context, cloud] as [typeof cloud, typeof cloud],
  onStart() { starts++; }, onFinish() {}, onProgress() {},
};
for (const previous of [false, true]) {
  if (previous) assert.equal(motion.commit(input), false);
  const before = image.values.slice();
  const failed = { ...input, frame: { ...input.frame, key: "failed-wind", step: 1, timeS: 30,
    wind: { ok: false as const, reason: "Wind point sampling unavailable: read failed" } } };
  assert.equal(motion.commit(failed), false);
  assert.equal(motion.describe().reason, failed.frame.wind.reason);
  assert.equal(motion.describe().active, false);
  assert.equal(motion.describe().toStep, 1);
  assert.equal(motion.describe().timeS, 30);
  assert.equal(motion.describe().alignment, null);
  assert.deepEqual(image.values, before);
  assert.equal(starts, 0);
  assert.equal(draws, 0);
  motion.reset();
}
assert.equal(motion.commit(input), false);
assert.equal(motion.describe().reason, "Waiting for the next saved cloud frame");
visualizationResourceGovernor.setCapacity({ totalBytes: null, stageBytes: { gpu: 0 } });
assert.equal(motion.commit({ ...input, frame: { ...input.frame, key: "next", step: 1, timeS: 30, cloud: { ...image, values: image.values.slice() } } }), false);
assert.equal(motion.describe().active, false);
assert.ok(motion.describe().reason?.toLowerCase().includes("gpu"));
assert.equal(starts, 0);
assert.equal(draws, 0);
motion.dispose(); binding.dispose(); cloud.delete(); context.delete(); cloudMapper.delete(); contextMapper.delete();
assert.equal(visualizationResourceGovernor.stats().allocations, 0);
console.log("Cloud interpolation admission: memory rejection preserves exact-frame fallback and releases retained buffers.");

visualizationResourceGovernor.setCapacity({ totalBytes: null, stageBytes: { gpu: null } });
let nextAnimation = 0;
const animations = new Map<number, FrameRequestCallback>();
Object.assign(globalThis, {
  requestAnimationFrame: (callback: FrameRequestCallback) => { animations.set(++nextAnimation, callback); return nextAnimation; },
  cancelAnimationFrame: (id: number) => { animations.delete(id); },
});
const tick = (now: number) => {
  const pending = [...animations.values()]; animations.clear();
  for (const callback of pending) callback(now);
};
const currentCloud = vtkVolume.newInstance(), currentContext = vtkVolume.newInstance();
const currentCloudMapper = vtkVolumeMapper.newInstance(), currentContextMapper = vtkVolumeMapper.newInstance();
currentCloud.setMapper(currentCloudMapper); currentContext.setMapper(currentContextMapper);
let completions = 0, liveFences = 0;
const preview = createCloudMotionController({ addVolume() {}, removeVolume() {} }, () => {}, () => ({
  SYNC_GPU_COMMANDS_COMPLETE: 1, TIMEOUT_EXPIRED: 2, WAIT_FAILED: 3,
  fenceSync() { liveFences++; return {}; }, deleteSync() { liveFences--; }, clientWaitSync() { return 1; }, flush() {},
}) as unknown as WebGL2RenderingContext, () => {});
const previewBinding = bindJointVolumes([{ actor: currentCloud, role: "cloud" }, { actor: currentContext, role: "context" }, ...preview.sources]);
preview.attach(previewBinding);
const transitionInput = { ...input, current: [currentContext, currentCloud] as [typeof currentCloud, typeof currentCloud], onFinish() { completions++; } };
assert.equal(preview.commit(transitionInput), false);
assert.equal(preview.commit({ ...transitionInput, frame: { ...input.frame, key: "second", step: 1, timeS: 30 } }), true);
const now = performance.now(); tick(now + 100); tick(now + 250);
assert.ok(preview.describe().timeS! > 0 && preview.describe().timeS! < 30);
preview.stop(true);
assert.equal(preview.describe().timeS, 30, "interruption must report the endpoint that is now displayed");
assert.equal(preview.describe().toStep, 1);
assert.equal(preview.describe().fraction, 1);
assert.equal(completions, 1);
assert.equal(liveFences, 0);
assert.equal(animations.size, 0);
assert.equal(preview.commit({ ...transitionInput, enabled: false, frame: { ...input.frame, key: "third", step: 2, timeS: 60 } }), false);
assert.equal(preview.describe().timeS, 60, "saved-frame playback must advance inactive preview metadata too");
assert.equal(preview.describe().toStep, 2);
preview.dispose(); previewBinding.dispose(); currentCloud.delete(); currentContext.delete(); currentCloudMapper.delete(); currentContextMapper.delete();
assert.equal(visualizationResourceGovernor.stats().allocations, 0);
console.log("Cloud interruption: exact endpoint timestamp, one completion, canceled draw, and released fences/buffers passed.");
