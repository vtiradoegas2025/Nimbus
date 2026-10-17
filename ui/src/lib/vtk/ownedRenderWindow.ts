import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkOpenGLRenderWindow from "@kitware/vtk.js/Rendering/OpenGL/RenderWindow";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass.js";
import { registerViewConstructor } from "@kitware/vtk.js/Rendering/Core/RenderWindow";
import { createFramebuffer } from "./sharedVolumeFramebuffer";

function extend(api: object, state: object, initialValues: object = {}): void {
  vtkOpenGLRenderWindow.extend(api, state, { ...initialValues, renderPasses: [] });
  const view = api as { delete(): void };
  const model = state as { renderPasses: Array<{ delete(): void }> };
  model.renderPasses.forEach(pass => pass.delete());
  const framebuffer = createFramebuffer();
  // The upstream depth prepass replaces attachments on every viewport resize
  // without retiring them. Use the same owned attachment lifecycle as volumes.
  model.renderPasses = [vtkForwardPass.newInstance({ framebuffer })];
  const dispose = view.delete;
  let deleted = false;
  view.delete = () => {
    if (deleted) return;
    deleted = true;
    framebuffer.releaseGraphicsResources(); framebuffer.delete();
    model.renderPasses.forEach(pass => pass.delete());
    dispose();
  };
}

const create = vtkMacro.newInstance(extend, "vtkOpenGLRenderWindow");
let installed = false;
export function installOwnedRenderWindow(): void {
  if (installed) return;
  installed = true;
  registerViewConstructor("WebGL", create);
}
