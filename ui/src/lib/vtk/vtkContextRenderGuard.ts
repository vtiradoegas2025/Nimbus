import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkRenderWindow from "@kitware/vtk.js/Rendering/Core/RenderWindow.js";
import vtkOpenGLRenderWindow from "@kitware/vtk.js/Rendering/OpenGL/RenderWindow.js";
import { beginVtkRenderTraversal } from "./vtkRenderTraversal";

interface GuardedView {
  traverseAllPasses: (...args: unknown[]) => unknown;
  getContext?: () => { isContextLost?: () => boolean } | null;
}
const retirements = new WeakMap<object, () => void>();
const holds = new WeakMap<object, number>();
let installed = false;

export function protectVtkTraversal(view: GuardedView): void {
  const traverse = view.traverseAllPasses;
  let retired = false;
  view.traverseAllPasses = (...args) => {
    if (retired || (holds.get(view) ?? 0) > 0 || view.getContext?.()?.isContextLost?.()) return;
    const end = beginVtkRenderTraversal(view);
    try { return traverse.apply(view, args); }
    finally { end(); }
  };
  retirements.set(view, () => { retired = true; });
}

export function holdVtkSceneRendering(window: { getApiSpecificRenderWindow: () => object }): () => void {
  const view = window.getApiSpecificRenderWindow();
  holds.set(view, (holds.get(view) ?? 0) + 1);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    const remaining = (holds.get(view) ?? 1) - 1;
    if (remaining <= 0) holds.delete(view);
    else holds.set(view, remaining);
  };
}

export function installVtkContextRenderGuard(): void {
  if (installed) return;
  installed = true;
  vtkRenderWindow.registerViewConstructor("WebGL", vtkMacro.newInstance((api, model, initialValues) => {
    vtkOpenGLRenderWindow.extend(api, model, initialValues);
    // VTK freezes public APIs after construction; guard every traversal before that freeze.
    protectVtkTraversal(api as unknown as GuardedView);
  }, "vtkNimbusOpenGLRenderWindow"));
}

export function guardVtkContextRendering(window: { getApiSpecificRenderWindow: () => object }): () => void {
  return retirements.get(window.getApiSpecificRenderWindow()) ?? (() => {});
}
