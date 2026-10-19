import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkOpenGLRenderWindow from "@kitware/vtk.js/Rendering/OpenGL/RenderWindow.js";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass.js";
import { registerViewConstructor } from "@kitware/vtk.js/Rendering/Core/RenderWindow.js";
import { createFramebuffer } from "./sharedVolumeFramebuffer";
import { protectVtkTraversal } from "./vtkTraversalGuard";
import { captureOwnedCanvasImage, type OwnedCanvasCaptureRequest } from "./ownedCanvasCapture";

function extend(api: object, state: object, initialValues: object = {}): void {
  vtkOpenGLRenderWindow.extend(api, state, { ...initialValues, renderPasses: [] });
  const view = api as { delete(): void; traverseAllPasses(...args: unknown[]): unknown;
    captureCurrentImage(format?: string): string;
    requestImageCapture(format?: string, onCaptured?: () => void): OwnedCanvasCaptureRequest };
  const model = state as { renderPasses: Array<{ delete(): void }>;
    context?: { isContextLost(): boolean };
    canvas: HTMLCanvasElement;
    renderable: { getRenderers(): Parameters<typeof captureOwnedCanvasImage>[1] } };
  let deleted = false;
  const captures = new Set<{ format: string; onCaptured?: () => void; resolve(value: string): void; reject(error: unknown): void }>();
  const requireCaptureContext = () => {
    const reason = deleted ? "scene deleted" : !model.canvas ? "canvas missing"
      : model.context?.isContextLost() ? "context lost" : null;
    if (reason) throw new Error(`Cannot capture an unavailable WebGL scene (${reason})`);
  };
  view.captureCurrentImage = (format) => {
    requireCaptureContext();
    return captureOwnedCanvasImage(model.canvas, model.renderable.getRenderers(), format);
  };
  view.requestImageCapture = (format = "image/png", onCaptured) => {
    requireCaptureContext();
    let request!: { format: string; onCaptured?: () => void; resolve(value: string): void; reject(error: unknown): void };
    const promise = new Promise<string>((resolve, reject) => {
      request = { format, onCaptured, resolve, reject };
      captures.add(request);
    });
    return { promise, cancel: () => {
      if (captures.delete(request)) request.reject(new Error("VTK frame capture cancelled"));
    } };
  };
  const traverse = view.traverseAllPasses;
  view.traverseAllPasses = (...args) => {
    const pending = [...captures];
    try {
      const result = traverse.apply(view, args);
      for (const request of pending) {
        if (!captures.delete(request)) continue;
        try {
          const png = view.captureCurrentImage(request.format);
          request.onCaptured?.();
          request.resolve(png);
        }
        catch (error) { request.reject(error); }
      }
      return result;
    } catch (error) {
      for (const request of pending) if (captures.delete(request)) request.reject(error);
      throw error;
    }
  };
  model.renderPasses.forEach(pass => pass.delete());
  const framebuffer = createFramebuffer();
  // The upstream depth prepass replaces attachments on every viewport resize
  // without retiring them. Use the same owned attachment lifecycle as volumes.
  model.renderPasses = [vtkForwardPass.newInstance({ framebuffer })];
  const dispose = view.delete;
  view.delete = () => {
    if (deleted) return;
    deleted = true;
    for (const request of captures) request.reject(new Error("VTK scene was deleted before capture"));
    captures.clear();
    framebuffer.releaseGraphicsResources(); framebuffer.delete();
    model.renderPasses.forEach(pass => pass.delete());
    dispose();
  };
  // VTK freezes views after construction; one factory must install both ownership and traversal guards.
  protectVtkTraversal(view);
}

const create = vtkMacro.newInstance(extend, "vtkOpenGLRenderWindow");
let installed = false;
export function installOwnedRenderWindow(): void {
  if (installed) return;
  installed = true;
  registerViewConstructor("WebGL", create);
}
