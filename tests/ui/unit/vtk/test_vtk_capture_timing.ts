import assert from "node:assert/strict";
import { registerVtkAppearanceCapture, lookupVtkAppearanceCapture, lookupVtkCapture, unregisterVtkCapture } from "../../../../ui/src/lib/vtk/vtkCaptureRegistry.ts";
import { attachVtkInteractionQuality } from "../../../../ui/src/lib/vtk/vtkInteraction.ts";
Object.assign(globalThis, { HTMLCanvasElement: class {}, HTMLElement: class {} });
let frame = 1, exportHooks = 0;
let qualityHolds = 0;
let resolveImage: (value: string) => void = () => {};
const window = {
  getInteractor: () => ({ getInteractorStyle: () => null }),
  getRenderer: () => ({ getActors: () => [], getVolumes: () => [] }),
  getApiSpecificRenderWindow: () => ({ captureNextImage: () => new Promise<string>(resolve => { resolveImage = resolve; }) }),
  getRenderWindow: () => ({ render: () => { frame++; } }),
};
const element = {} as HTMLElement;
const releaseQuality = attachVtkInteractionQuality(window, {
  begin() {}, end() {}, setFullSampleDistance() {}, setFullSampleDistanceFor() {},
  registerMapper: () => () => {}, dispose() {}, isPreview: () => qualityHolds === 0,
  holdFullQuality: () => { qualityHolds++; return () => { qualityHolds--; }; },
});
registerVtkAppearanceCapture(element, window, () => ({ plot: "supercell", backend: "webgl", frameIndex: frame }), () => { exportHooks++; });
const appearance = lookupVtkAppearanceCapture(element)!();
assert.equal(qualityHolds, 1, "evidence capture holds settled ray sampling until the image resolves");
frame = 10;
resolveImage("data:image/png;base64,fixture");
assert.equal((await appearance).evidence.descriptor.frameIndex, 1, "metadata belongs to the captured draw, not later animation state");
assert.equal(qualityHolds, 0, "evidence capture releases its quality hold");
assert.equal(exportHooks, 0, "inspection capture retains its intermediate state");
const saved = lookupVtkCapture(element)!();
assert.equal(exportHooks, 1, "ordinary export invokes the exact-frame hook before drawing");
resolveImage("data:image/png;base64,fixture");
await saved;
unregisterVtkCapture(element);
releaseQuality();
assert.equal(lookupVtkCapture(element), null);
console.log("VTK capture timing: frozen evidence and exact-frame export hooks passed.");
