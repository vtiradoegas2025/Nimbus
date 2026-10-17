import assert from "node:assert/strict";
import { attachVtkInteractionQuality, createVtkVolumeQualityController, holdVtkFullQuality } from "../../../../ui/src/lib/vtk/vtkInteraction.ts";

const timers = new Map<number, () => void>();
let timer = 0, renders = 0;
const changes: boolean[] = [];
const first = { distance: 0, maximum: 0, screen: 0,
  setSampleDistance(value: number) { this.distance = value; },
  setMaximumSamplesPerRay(value: number) { this.maximum = value; },
  setImageSampleDistance(value: number) { this.screen = value; },
};
const second = { ...first };
const quality = createVtkVolumeQualityController({ mapper: first, render() { renders++; },
  onPreviewChange(value) { changes.push(value); }, fullMaximumSamples: 2400, previewMaximumSamples: 600,
  previewDistanceMultiplier: 3, previewImageSampleDistance: 2,
  setIdle(callback) { timers.set(++timer, callback); return timer; },
  clearIdle(handle) { timers.delete(handle as number); },
});
quality.setFullSampleDistance(120);
const renderWindow = { getInteractor: () => ({ getInteractorStyle: () => null }) };
const detach = attachVtkInteractionQuality(renderWindow, quality);
quality.begin("interaction");
assert.equal(quality.isPreview(), true);
assert.deepEqual([first.distance, first.maximum, first.screen], [360, 600, 2]);
const releaseFirst = holdVtkFullQuality(renderWindow);
const releaseSecond = quality.holdFullQuality();
assert.equal(quality.isPreview(), false);
assert.deepEqual([first.distance, first.maximum, first.screen], [120, 2400, 1]);
const unregister = quality.registerMapper(second, 75);
assert.deepEqual([second.distance, second.maximum, second.screen], [75, 2400, 1]);
quality.begin("resize"); quality.end("interaction");
releaseFirst(); releaseFirst();
assert.equal(quality.isPreview(), false);
assert.equal(timers.size, 0);
releaseSecond();
assert.equal(quality.isPreview(), true);
assert.deepEqual([second.distance, second.maximum, second.screen], [225, 600, 2]);
assert.equal(renders, 0);

const releaseSettling = quality.holdFullQuality();
quality.end("resize");
assert.equal(timers.size, 1);
for (const callback of timers.values()) callback(); timers.clear();
releaseSettling();
assert.equal(quality.isPreview(), false);
assert.deepEqual([first.distance, first.maximum, first.screen], [120, 2400, 1]);
assert.equal(renders, 0);

quality.begin("synchronization");
const releaseChanged = quality.holdFullQuality();
quality.setFullSampleDistanceFor(second, 45);
quality.begin("interaction"); quality.end("synchronization");
assert.equal(second.distance, 45);
releaseChanged();
assert.equal(second.distance, 135);
assert.equal(quality.isPreview(), true);
const releaseDisposed = quality.holdFullQuality();
detach(); releaseDisposed(); releaseDisposed();
assert.equal(quality.isPreview(), false);
assert.equal(timers.size, 0);
assert.equal(renders, 0);
const before = [...changes];
holdVtkFullQuality(renderWindow)();
assert.deepEqual(changes, before);
unregister();
console.log("Balanced capture holds preserve active gestures, nested holds, idle expiry, changed per-mapper sampling, dynamic mappers, and disposal without redundant renders");
