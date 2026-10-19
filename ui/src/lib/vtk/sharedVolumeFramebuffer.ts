import vtkFramebuffer from "@kitware/vtk.js/Rendering/OpenGL/Framebuffer.js";

interface Texture {
  releaseGraphicsResources(window: object): void;
  delete(): void;
}

interface Framebuffer {
  create(width: number, height: number): void;
  releaseGraphicsResources(): void;
  setOpenGLRenderWindow(window: object): void;
  delete(): void;
}

const targets = new WeakMap<object, { framebuffer: Framebuffer; users: number }>();

export function createFramebuffer(): Framebuffer {
  const api = {} as Framebuffer;
  const model = {} as {
    context: WebGLRenderingContext;
    _openGLRenderWindow: object;
    glFramebuffer: WebGLFramebuffer | null;
    depthTexture: WebGLRenderbuffer | null;
    colorBuffers: Texture[];
  };
  vtkFramebuffer.extend(api, model);
  const create = api.create;
  api.releaseGraphicsResources = () => {
    for (const texture of model.colorBuffers) {
      texture.releaseGraphicsResources(model._openGLRenderWindow);
      texture.delete();
    }
    model.colorBuffers = [];
    if (model.depthTexture) model.context.deleteRenderbuffer(model.depthTexture);
    if (model.glFramebuffer) model.context.deleteFramebuffer(model.glFramebuffer);
    model.depthTexture = null;
    model.glFramebuffer = null;
  };
  api.create = (width, height) => {
    api.releaseGraphicsResources();
    create(width, height);
  };
  return api;
}

export function acquireVolumeFramebuffer(window: object): { framebuffer: Framebuffer; release(): void } {
  let target = targets.get(window);
  if (!target) {
    const framebuffer = createFramebuffer();
    framebuffer.setOpenGLRenderWindow(window);
    target = { framebuffer, users: 0 };
    targets.set(window, target);
  }
  target.users++;
  let released = false;
  return { framebuffer: target.framebuffer, release: () => {
    if (released) return;
    released = true;
    target.users--;
    if (target.users === 0) {
      target.framebuffer.releaseGraphicsResources();
      target.framebuffer.delete();
      targets.delete(window);
    }
  } };
}
