import assert from "node:assert/strict";
import vtkRenderWindow from "@kitware/vtk.js/Rendering/Core/RenderWindow.js";
import { installOwnedRenderWindow } from "../../../../ui/src/lib/vtk/ownedRenderWindow.ts";
import { captureVtkPng } from "../../../../ui/src/lib/vtk/vtkCaptureRegistry.ts";
import { attachVtkPlaybackScreenResolution } from "../../../../ui/src/lib/vtk/vtkPlaybackScreenResolution.ts";
import type { OwnedCanvasCaptureRequest } from "../../../../ui/src/lib/vtk/ownedCanvasCapture.ts";

interface TestView {
  getContext(): WebGL2RenderingContext | null;
  setContext(context: object | null): void;
  setRenderable(core: object): void;
  setRenderPasses(passes: object[]): void;
  traverseAllPasses(): void;
  requestImageCapture(format: string): OwnedCanvasCaptureRequest;
  delete(): void;
}
const originalImage = Object.getOwnPropertyDescriptor(globalThis, "Image");
const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
let renders = 0, encodes = 0, temporaryStoresRetired = 0;
Object.assign(globalThis, {
  Image: class { style = {}; },
  document: { createElement: () => {
    const canvas = { width: 0, height: 0,
      getContext: () => ({ drawImage() {} }),
      toDataURL: () => { encodes++; return `data:image/png;base64,frame-${renders}`; },
      remove: () => { assert.equal(canvas.width, 0); assert.equal(canvas.height, 0); temporaryStoresRetired++; },
    };
    return canvas;
  } },
});
try {
  installOwnedRenderWindow();
  const canvas = { width: 256, height: 160, style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {},
    getBoundingClientRect: () => ({ x: 0, y: 0 }) };
  const view = vtkRenderWindow.newAPISpecificView("WebGL", { canvas }) as unknown as TestView;
  view.setRenderable(vtkRenderWindow.newInstance());
  let failRender = false, deferRender = true;
  view.setRenderPasses([{ traverse: () => {
    if (failRender) throw new Error("render pass failed");
    renders++;
  }, delete() {} }]);
  let distance = 1.5;
  const mapper = { getImageSampleDistance: () => distance, setImageSampleDistance: (value: number) => { distance = value; } };
  const window = { getApiSpecificRenderWindow: () => view,
    getRenderWindow: () => ({ render: () => { if (!deferRender) view.traverseAllPasses(); } }) };
  const controller = attachVtkPlaybackScreenResolution(window, { getMappers: () => [mapper], timer: {
    begin: () => false, end() {}, cancel() {}, dispose() {}, status: () => "available",
  } });
  const pending = captureVtkPng(window);
  assert.equal(distance, 1, "capture holds full resolution when an active interactor defers rendering");
  assert.equal(encodes, 0, "a deferred render must not capture the preceding preview frame");
  view.traverseAllPasses();
  assert.equal(await pending, "data:image/png;base64,frame-1");
  assert.equal(distance, 1.5, "completed capture restores the prior playback resolution");

  deferRender = false;
  const immediate = captureVtkPng(window);
  assert.equal(distance, 1, "an immediate traversal still keeps its hold until the capture promise settles");
  assert.equal(await immediate, "data:image/png;base64,frame-2");
  assert.equal(distance, 1.5);

  deferRender = true;
  await assert.rejects(captureVtkPng(window, 1), /capture timed out/);
  assert.equal(distance, 1.5, "timeout releases the resolution hold");
  view.traverseAllPasses();
  assert.equal(encodes, 2, "a timed-out request cannot encode a later unrelated frame");

  deferRender = false; failRender = true;
  await assert.rejects(captureVtkPng(window), /render pass failed/);
  assert.equal(encodes, 2, "a failed traversal cannot publish partial pixels");
  assert.equal(distance, 1.5);

  deferRender = true; failRender = false;
  const deleted = assert.rejects(captureVtkPng(window), /scene was deleted before capture/);
  controller.dispose();
  view.delete();
  await deleted;
  assert.equal(encodes, temporaryStoresRetired, "every completed snapshot retires its temporary backing store");
} finally {
  if (originalImage) Object.defineProperty(globalThis, "Image", originalImage);
  else Reflect.deleteProperty(globalThis, "Image");
  if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
  else Reflect.deleteProperty(globalThis, "document");
}
console.log("Owned captures await actual draws, preserve quality holds and retire cancelled or failed requests: passed");
