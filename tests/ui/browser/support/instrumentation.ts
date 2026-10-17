export interface BrowserResourceSnapshot {
  canvases: number;
  webglContexts: number;
  textures: number;
  buffers: number;
  gpuLogicalBufferBytes: number;
  gpuLogicalTextureBytes: number;
  gpuLogicalBytes: number;
  texture3DUploads: number;
  workers: number;
  workerKinds: Record<string, number>;
  workerPosts: number;
  workerTerminations: number;
  contextLosses: number;
  plannedContextReleases: number;
  contextRestores: number;
  longTasks: number;
  maxLongTaskMs: number;
  frameSamples: number;
  p50FrameMs: number | null;
  p95FrameMs: number | null;
  maxFrameMs: number | null;
  jsHeapBytes: number | null;
}

interface PerformanceWithMemory extends Performance {
  memory?: { usedJSHeapSize?: number };
}

type Gl = WebGLRenderingContext | WebGL2RenderingContext;

interface GlResourceCounts {
  textures: number;
  buffers: number;
  textureBytes: number;
  bufferBytes: number;
}
const contextRefs: Array<WeakRef<Gl>> = [];
const knownContexts = new WeakSet<Gl>();
const resourceCounts = new WeakMap<Gl, GlResourceCounts>();
const bufferSizes = new WeakMap<WebGLBuffer, number>();
const textureLevels = new WeakMap<WebGLTexture, Map<string, number>>();
const boundBuffers = new WeakMap<Gl, Map<number, WebGLBuffer | null>>();
const boundTextures = new WeakMap<Gl, Map<number, WebGLTexture | null>>();
const observedCanvases = new WeakSet<HTMLCanvasElement>();
const frameDeltas: number[] = [];
let lastFrame = 0;
let workers = 0;
const workerKinds = new Map<string, number>();
let workerPosts = 0;
let workerTerminations = 0;
let contextLosses = 0;
let plannedContextReleases = 0;
let contextRestores = 0;
let longTasks = 0;
let maxLongTaskMs = 0;
let texture3DUploads = 0;

function percentile(values: readonly number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
}

function recordContext(context: RenderingContext | null, canvas: HTMLCanvasElement): void {
  if (!(context instanceof WebGLRenderingContext) &&
      !(typeof WebGL2RenderingContext !== "undefined" && context instanceof WebGL2RenderingContext)) {
    return;
  }
  if (!knownContexts.has(context)) {
    knownContexts.add(context);
    contextRefs.push(new WeakRef(context));
    resourceCounts.set(context, { textures: 0, buffers: 0, textureBytes: 0, bufferBytes: 0 });
    boundBuffers.set(context, new Map());
    boundTextures.set(context, new Map());
  }
  if (observedCanvases.has(canvas)) return;
  observedCanvases.add(canvas);
  canvas.addEventListener("webglcontextlost", () => {
    // WebKit may deliver the loss asynchronously after React has detached a
    // deliberately retired canvas. That is teardown evidence, not an active
    // scene failure, even if the element's dataset was already discarded.
    if (canvas.dataset.nimbusVtkRelease === "planned" || !canvas.isConnected) {
      plannedContextReleases += 1;
    } else {
      contextLosses += 1;
    }
  });
  canvas.addEventListener("webglcontextrestored", () => {
    contextRestores += 1;
  });
}

function patchContextPrototype(prototype: WebGLRenderingContext): void {
  const originalCreateTexture = prototype.createTexture;
  const originalDeleteTexture = prototype.deleteTexture;
  const originalCreateBuffer = prototype.createBuffer;
  const originalDeleteBuffer = prototype.deleteBuffer;
  const originalBindBuffer = prototype.bindBuffer;
  const originalBufferData = prototype.bufferData;
  const originalBindTexture = prototype.bindTexture;
  const originalTexImage2D = prototype.texImage2D;
  const originalTexImage3D = (prototype as WebGL2RenderingContext).texImage3D;
  const originalTexStorage2D = (prototype as WebGL2RenderingContext).texStorage2D;
  const originalTexStorage3D = (prototype as WebGL2RenderingContext).texStorage3D;
  const originalTexSubImage2D = prototype.texSubImage2D;
  const originalTexSubImage3D = (prototype as WebGL2RenderingContext).texSubImage3D;
  if ((prototype as unknown as { __nimbusPatched?: boolean }).__nimbusPatched) return;
  (prototype as unknown as { __nimbusPatched?: boolean }).__nimbusPatched = true;

  prototype.createTexture = function createTexture(this: Gl) {
    const texture = originalCreateTexture.call(this);
    const counts = resourceCounts.get(this);
    if (texture && counts) counts.textures += 1;
    return texture;
  };
  prototype.deleteTexture = function deleteTexture(this: Gl, texture: WebGLTexture | null) {
    const counts = resourceCounts.get(this);
    if (texture && counts) {
      counts.textures = Math.max(0, counts.textures - 1);
      const bytes = [...(textureLevels.get(texture)?.values() ?? [])]
        .reduce((sum, value) => sum + value, 0);
      counts.textureBytes = Math.max(0, counts.textureBytes - bytes);
      textureLevels.delete(texture);
    }
    return originalDeleteTexture.call(this, texture);
  };
  prototype.createBuffer = function createBuffer(this: Gl) {
    const buffer = originalCreateBuffer.call(this);
    const counts = resourceCounts.get(this);
    if (buffer && counts) counts.buffers += 1;
    return buffer;
  };
  prototype.deleteBuffer = function deleteBuffer(this: Gl, buffer: WebGLBuffer | null) {
    const counts = resourceCounts.get(this);
    if (buffer && counts) {
      counts.buffers = Math.max(0, counts.buffers - 1);
      counts.bufferBytes = Math.max(0, counts.bufferBytes - (bufferSizes.get(buffer) ?? 0));
      bufferSizes.delete(buffer);
    }
    return originalDeleteBuffer.call(this, buffer);
  };
  prototype.bindBuffer = function bindBuffer(this: Gl, target: number, buffer: WebGLBuffer | null) {
    boundBuffers.get(this)?.set(target, buffer);
    return originalBindBuffer.call(this, target, buffer);
  };
  prototype.bufferData = function bufferData(this: Gl, ...args: unknown[]) {
    const target = Number(args[0]);
    const source = args[1];
    const bytes = typeof source === "number"
      ? Math.max(0, source)
      : source instanceof ArrayBuffer || ArrayBuffer.isView(source)
        ? source.byteLength
        : 0;
    const buffer = boundBuffers.get(this)?.get(target) ?? null;
    const counts = resourceCounts.get(this);
    if (buffer && counts) {
      const previous = bufferSizes.get(buffer) ?? 0;
      bufferSizes.set(buffer, bytes);
      counts.bufferBytes = Math.max(0, counts.bufferBytes - previous + bytes);
    }
    return (originalBufferData as (...callArgs: unknown[]) => void).apply(this, args);
  } as typeof prototype.bufferData;
  prototype.bindTexture = function bindTexture(this: Gl, target: number, texture: WebGLTexture | null) {
    boundTextures.get(this)?.set(target, texture);
    return originalBindTexture.call(this, target, texture);
  };
  const textureTarget = (context: Gl, target: number): number => {
    if (target >= context.TEXTURE_CUBE_MAP_POSITIVE_X
        && target <= context.TEXTURE_CUBE_MAP_NEGATIVE_Z) return context.TEXTURE_CUBE_MAP;
    return target;
  };
  const componentBytes = (context: Gl, type: number): number => {
    if (type === context.FLOAT || type === context.UNSIGNED_INT || type === context.INT) return 4;
    if (type === context.UNSIGNED_SHORT || type === context.SHORT
        || type === (context as WebGL2RenderingContext).HALF_FLOAT) return 2;
    return 1;
  };
  const components = (context: Gl, format: number): number => {
    if (format === context.RGBA) return 4;
    if (format === context.RGB) return 3;
    if (format === context.LUMINANCE_ALPHA || format === (context as WebGL2RenderingContext).RG) return 2;
    return 1;
  };
  const recordTexture = (
    context: Gl,
    target: number,
    level: number,
    width: number,
    height: number,
    depth: number,
    format: number,
    type: number,
    pixels: unknown,
    explicitBytes?: number,
    preserveLargerAllocation = false,
  ) => {
    const bound = boundTextures.get(context)?.get(textureTarget(context, target)) ?? null;
    const counts = resourceCounts.get(context);
    if (!bound || !counts) return;
    const bytes = explicitBytes ?? (
      pixels instanceof ArrayBuffer || ArrayBuffer.isView(pixels)
        ? pixels.byteLength
        : Math.max(0, width) * Math.max(0, height) * Math.max(0, depth)
          * components(context, format) * componentBytes(context, type)
    );
    let levels = textureLevels.get(bound);
    if (!levels) {
      levels = new Map();
      textureLevels.set(bound, levels);
    }
    const key = `${target}:${level}`;
    const previous = levels.get(key) ?? 0;
    const retainedBytes = preserveLargerAllocation ? Math.max(previous, bytes) : bytes;
    levels.set(key, retainedBytes);
    counts.textureBytes = Math.max(0, counts.textureBytes - previous + retainedBytes);
  };
  const internalFormatBytes = (format: number): number => {
    const known = new Map<number, number>([
      [0x8229, 1], [0x822d, 2], [0x822e, 4],
      [0x822b, 2], [0x822f, 4], [0x8230, 8],
      [0x8051, 3], [0x881b, 6], [0x8815, 12],
      [0x8058, 4], [0x881a, 8], [0x8814, 16],
    ]);
    return known.get(format) ?? 0;
  };
  const recordImmutableTexture = (
    context: Gl,
    target: number,
    levels: number,
    internalFormat: number,
    width: number,
    height: number,
    depth: number,
  ) => {
    const bytesPerTexel = internalFormatBytes(internalFormat);
    if (bytesPerTexel === 0) return;
    for (let level = 0; level < levels; level += 1) {
      recordTexture(
        context,
        target,
        level,
        Math.max(1, width >> level),
        Math.max(1, height >> level),
        Math.max(1, depth >> level),
        0,
        0,
        null,
        Math.max(1, width >> level)
          * Math.max(1, height >> level)
          * Math.max(1, depth >> level)
          * bytesPerTexel,
      );
    }
  };
  prototype.texImage2D = function texImage2D(this: Gl, ...args: unknown[]) {
    if (args.length >= 9) {
      recordTexture(
        this,
        Number(args[0]), Number(args[1]), Number(args[3]), Number(args[4]), 1,
        Number(args[6]), Number(args[7]), args[8],
      );
    }
    return (originalTexImage2D as (...callArgs: unknown[]) => void).apply(this, args);
  } as typeof prototype.texImage2D;
  prototype.texSubImage2D = function texSubImage2D(this: Gl, ...args: unknown[]) {
    if (args.length >= 9) {
      recordTexture(
        this,
        Number(args[0]), Number(args[1]), Number(args[4]), Number(args[5]), 1,
        Number(args[6]), Number(args[7]), args[8], undefined, true,
      );
    }
    return (originalTexSubImage2D as (...callArgs: unknown[]) => void).apply(this, args);
  } as typeof prototype.texSubImage2D;
  if (originalTexImage3D) {
    (prototype as WebGL2RenderingContext).texImage3D = function texImage3D(
      this: WebGL2RenderingContext,
      ...args: unknown[]
    ) {
      if (args.length >= 10) {
        if (args[9] != null) texture3DUploads++;
        recordTexture(
          this,
          Number(args[0]), Number(args[1]), Number(args[3]), Number(args[4]), Number(args[5]),
          Number(args[7]), Number(args[8]), args[9],
        );
      }
      return (originalTexImage3D as (...callArgs: unknown[]) => void).apply(this, args);
    } as typeof WebGL2RenderingContext.prototype.texImage3D;
  }
  if (originalTexSubImage3D) {
    (prototype as WebGL2RenderingContext).texSubImage3D = function texSubImage3D(
      this: WebGL2RenderingContext,
      ...args: unknown[]
    ) {
      if (args.length >= 11) {
        texture3DUploads++;
        recordTexture(
          this,
          Number(args[0]), Number(args[1]), Number(args[5]), Number(args[6]), Number(args[7]),
          Number(args[8]), Number(args[9]), args[10], undefined, true,
        );
      }
      return (originalTexSubImage3D as (...callArgs: unknown[]) => void).apply(this, args);
    } as typeof WebGL2RenderingContext.prototype.texSubImage3D;
  }
  if (originalTexStorage2D) {
    (prototype as WebGL2RenderingContext).texStorage2D = function texStorage2D(
      this: WebGL2RenderingContext,
      target: number,
      levels: number,
      internalFormat: number,
      width: number,
      height: number,
    ) {
      recordImmutableTexture(this, target, levels, internalFormat, width, height, 1);
      return originalTexStorage2D.call(this, target, levels, internalFormat, width, height);
    };
  }
  if (originalTexStorage3D) {
    (prototype as WebGL2RenderingContext).texStorage3D = function texStorage3D(
      this: WebGL2RenderingContext,
      target: number,
      levels: number,
      internalFormat: number,
      width: number,
      height: number,
      depth: number,
    ) {
      recordImmutableTexture(this, target, levels, internalFormat, width, height, depth);
      return originalTexStorage3D.call(this, target, levels, internalFormat, width, height, depth);
    };
  }
}

export function installBrowserInstrumentation(): void {
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function getContext(
    this: HTMLCanvasElement,
    contextId: string,
    options?: unknown,
  ): RenderingContext | null {
    const context = originalGetContext.call(this, contextId, options as never);
    recordContext(context, this);
    return context;
  } as typeof HTMLCanvasElement.prototype.getContext;

  patchContextPrototype(WebGLRenderingContext.prototype);
  if (typeof WebGL2RenderingContext !== "undefined") {
    patchContextPrototype(WebGL2RenderingContext.prototype);
  }

  const NativeWorker = window.Worker;
  class InstrumentedWorker extends NativeWorker {
    private counted = true;
    private readonly workerKind: string;

    constructor(scriptURL: string | URL, options?: WorkerOptions) {
      super(scriptURL, options);
      this.workerKind = options?.name || String(scriptURL).split("/").at(-1)?.split("?")[0] || "unnamed";
      workers += 1;
      workerKinds.set(this.workerKind, (workerKinds.get(this.workerKind) ?? 0) + 1);
    }

    private retireCount(): void {
      if (!this.counted) return;
      this.counted = false;
      workers -= 1;
      workerTerminations += 1;
      const remaining = (workerKinds.get(this.workerKind) ?? 1) - 1;
      if (remaining > 0) workerKinds.set(this.workerKind, remaining);
      else workerKinds.delete(this.workerKind);
    }

    override postMessage(message: unknown, transfer: Transferable[]): void;
    override postMessage(message: unknown, options?: StructuredSerializeOptions): void;
    override postMessage(
      message: unknown,
      second?: Transferable[] | StructuredSerializeOptions,
    ): void {
      workerPosts += 1;
      // VTK preparation workers close themselves only after the UI has
      // acknowledged their transferable result. Count that explicit protocol
      // as retirement; forcing native terminate afterward creates a false
      // access-control page error in WebKit.
      if (
        this.counted
        && typeof message === "object"
        && message != null
        && (message as { kind?: unknown }).kind === "close"
      ) {
        this.retireCount();
      }
      if (Array.isArray(second)) super.postMessage(message, second);
      else super.postMessage(message, second);
    }

    override terminate(): void {
      if (this.counted) {
        this.retireCount();
      }
      super.terminate();
    }
  }
  window.Worker = InstrumentedWorker as typeof Worker;

  if (typeof PerformanceObserver !== "undefined") {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          longTasks += 1;
          maxLongTaskMs = Math.max(maxLongTaskMs, entry.duration);
        }
      });
      observer.observe({ type: "longtask", buffered: true });
    } catch {
      // WebKit may not implement the Long Tasks API; null support is reported.
    }
  }

  const sampleFrame = (now: number) => {
    if (lastFrame > 0) {
      frameDeltas.push(now - lastFrame);
      if (frameDeltas.length > 20_000) frameDeltas.splice(0, 10_000);
    }
    lastFrame = now;
    requestAnimationFrame(sampleFrame);
  };
  requestAnimationFrame(sampleFrame);
}

export function snapshotBrowserResources(): BrowserResourceSnapshot {
  /*
   * Compacted in place, not just filtered.
   *
   * One WeakRef is appended per WebGL context ever created and nothing ever
   * removed them, so the array grew for the whole run — a thousand-mount soak
   * carried a thousand entries and walked all of them on every sample. Tiny
   * next to what it is measuring, but it is unbounded growth inside the
   * instrument, which is the last place it should be.
   */
  for (let i = contextRefs.length - 1; i >= 0; i -= 1) {
    if (contextRefs[i].deref() === undefined) contextRefs.splice(i, 1);
  }
  const liveContexts = contextRefs
    .map((reference) => reference.deref())
    .filter((context): context is Gl => Boolean(
      context && context.canvas instanceof HTMLCanvasElement && context.canvas.isConnected,
    ));
  const liveTextures = liveContexts.reduce(
    (sum, context) => sum + (resourceCounts.get(context)?.textures ?? 0),
    0,
  );
  const liveBuffers = liveContexts.reduce(
    (sum, context) => sum + (resourceCounts.get(context)?.buffers ?? 0),
    0,
  );
  const gpuLogicalBufferBytes = liveContexts.reduce(
    (sum, context) => sum + (resourceCounts.get(context)?.bufferBytes ?? 0),
    0,
  );
  const gpuLogicalTextureBytes = liveContexts.reduce(
    (sum, context) => sum + (resourceCounts.get(context)?.textureBytes ?? 0),
    0,
  );
  const heap = (performance as PerformanceWithMemory).memory?.usedJSHeapSize;
  return {
    canvases: document.querySelectorAll("canvas").length,
    webglContexts: liveContexts.length,
    textures: liveTextures,
    buffers: liveBuffers,
    gpuLogicalBufferBytes,
    gpuLogicalTextureBytes,
    gpuLogicalBytes: gpuLogicalBufferBytes + gpuLogicalTextureBytes,
    texture3DUploads,
    workers,
    workerKinds: Object.fromEntries([...workerKinds].sort(([left], [right]) => left.localeCompare(right))),
    workerPosts,
    workerTerminations,
    contextLosses,
    plannedContextReleases,
    contextRestores,
    longTasks,
    maxLongTaskMs,
    frameSamples: frameDeltas.length,
    p50FrameMs: percentile(frameDeltas, 0.5),
    p95FrameMs: percentile(frameDeltas, 0.95),
    maxFrameMs: frameDeltas.length > 0 ? Math.max(...frameDeltas) : null,
    jsHeapBytes: Number.isFinite(heap) ? heap! : null,
  };
}
