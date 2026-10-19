import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkOpenGLRenderer from "@kitware/vtk.js/Rendering/OpenGL/Renderer.js";
import { registerOverride } from "@kitware/vtk.js/Rendering/OpenGL/ViewNodeFactory.js";

interface OrderedNode {
  getRenderable(): object;
}

export function orderRenderNodes<T extends OrderedNode>(children: T[], renderables: readonly object[]): void {
  const ranks = new Map(renderables.map((value, index) => [value, index]));
  const slots: number[] = [];
  const ordered: T[] = [];
  children.forEach((child, index) => {
    if (ranks.has(child.getRenderable())) { slots.push(index); ordered.push(child); }
  });
  ordered.sort((a, b) => ranks.get(a.getRenderable())! - ranks.get(b.getRenderable())!);
  slots.forEach((slot, index) => { children[slot] = ordered[index]; });
}

function extend(api: object, state: object, initialValues: object = {}): void {
  vtkOpenGLRenderer.extend(api, state, initialValues);
  const renderer = api as { addMissingNodes(values: object[], enforceOrder?: boolean): void };
  const model = state as { children: OrderedNode[] };
  const add = renderer.addMissingNodes;
  renderer.addMissingNodes = (values, enforceOrder = false) => {
    add(values, false);
    // vtk.js 36 passes three arguments to its two-argument moveChild helper,
    // leaving cached view nodes in creation order after a camera-side change.
    if (enforceOrder && values?.length) orderRenderNodes(model.children, values);
  };
}

const createOrderedRenderer = vtkMacro.newInstance(extend, "vtkNimbusOrderedRenderer");
let installed = false;
export function installOrderedRenderer(): void {
  if (installed) return;
  installed = true;
  registerOverride("vtkRenderer", createOrderedRenderer);
}
