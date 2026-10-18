import assert from "node:assert/strict";
import vtkRenderWindow from "@kitware/vtk.js/Rendering/Core/RenderWindow.js";
import { installOwnedRenderWindow } from "../../../../ui/src/lib/vtk/ownedRenderWindow.ts";
import { guardVtkContextRendering, holdVtkSceneRendering, installVtkContextRenderGuard } from "../../../../ui/src/lib/vtk/vtkContextRenderGuard.ts";

const originalImage = globalThis.Image;
Object.assign(globalThis, { Image: class { style = {}; } });
try {
  installVtkContextRenderGuard();
  installOwnedRenderWindow();
  installVtkContextRenderGuard();
  let lost = false, traversals = 0, disposals = 0;
  const canvas = { style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {} };
  const view = vtkRenderWindow.newAPISpecificView("WebGL", { canvas }) as unknown as {
    setContext(context: object | null): void;
    setRenderable(core: object): void;
    setRenderPasses(passes: object[]): void;
    traverseAllPasses(): void;
    delete(): void;
  };
  view.setContext({ isContextLost: () => lost });
  view.setRenderable(vtkRenderWindow.newInstance());
  view.setRenderPasses([{ traverse: () => { traversals++; }, delete: () => { disposals++; } }]);
  const window = { getApiSpecificRenderWindow: () => view };
  view.traverseAllPasses();
  assert.equal(traversals, 1);
  const release = holdVtkSceneRendering(window);
  view.traverseAllPasses();
  assert.equal(traversals, 1, "the installed owned constructor retains scene holding");
  release();
  lost = true;
  view.traverseAllPasses();
  assert.equal(traversals, 1, "the installed constructor blocks traversal before the loss event arrives");
  const retire = guardVtkContextRendering(window);
  retire();
  lost = false;
  view.traverseAllPasses();
  assert.equal(traversals, 1, "a restored obsolete view stays retired until rebuilt");
  view.setContext(null);
  view.delete();
  view.delete();
  assert.equal(disposals, 1, "traversal protection retains owned render-pass disposal");
} finally {
  Object.assign(globalThis, { Image: originalImage });
}
console.log("Owned WebGL construction preserves traversal, loss and disposal guards: passed");
