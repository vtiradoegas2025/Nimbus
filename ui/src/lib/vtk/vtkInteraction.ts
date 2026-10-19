import { setSettledVolumeDistance, setFlatVolumePreview } from "./volumeShaderBudget";
import {
  backingStoreSize,
  MIN_PIXEL_RATIO,
  type BackingStoreSize,
  type ResolutionPolicy,
} from "./vtkResolution";

type TimerHandle = number | ReturnType<typeof setTimeout>;
type AnimationFrameCallback = (timestamp: number) => void;

export interface ResizeSchedulerOptions {
  resize: (width: number, height: number, pixelRatio: number) => void;
  onPreviewChange?: (preview: boolean) => void;
  requestFrame?: (callback: AnimationFrameCallback) => number;
  cancelFrame?: (handle: number) => void;
  setIdle?: (callback: () => void, milliseconds: number) => TimerHandle;
  clearIdle?: (handle: TimerHandle) => void;
  idleMs?: number;
}

export interface VtkResizeScheduler {
  /** `variant` distinguishes states that resolve the same CSS size to a
   *  different backing store — currently at-rest vs mid-gesture resolution.
   *  Without it a gesture that does not move the container would coalesce
   *  away the very resize it is supposed to trigger. */
  notify: (width: number, height: number, pixelRatio?: number, variant?: number) => void;
  dispose: () => void;
}

type ResizeIdentity = [number, number, number, number];

function sameIdentity(a: ResizeIdentity | null, b: ResizeIdentity): boolean {
  return a !== null && a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

/** Coalesce a resize storm to the newest non-zero CSS size and display scale. */
export function createVtkResizeScheduler(options: ResizeSchedulerOptions): VtkResizeScheduler {
  const animation = globalThis as unknown as {
    requestAnimationFrame: (callback: AnimationFrameCallback) => number;
    cancelAnimationFrame: (handle: number) => void;
  };
  const requestFrame = options.requestFrame ?? animation.requestAnimationFrame.bind(globalThis);
  const cancelFrame = options.cancelFrame ?? animation.cancelAnimationFrame.bind(globalThis);
  const setIdle = options.setIdle ?? setTimeout;
  const clearIdle = options.clearIdle ?? clearTimeout;
  const idleMs = options.idleMs ?? 160;
  let frame: number | null = null;
  let idle: TimerHandle | null = null;
  let pending: { identity: ResizeIdentity; measurement: [number, number, number] } | null = null;
  let applied: ResizeIdentity | null = null;
  let disposed = false;
  let preview = false;

  const setPreview = (next: boolean) => {
    if (preview === next) return;
    preview = next;
    options.onPreviewChange?.(next);
  };
  const scheduleIdle = () => {
    if (idle !== null) clearIdle(idle);
    idle = setIdle(() => {
      idle = null;
      setPreview(false);
    }, idleMs);
  };
  const notify = (width: number, height: number, pixelRatio = 1, variant = 0) => {
    if (disposed || !(width > 0) || !(height > 0)) return;
    const finitePixelRatio = Number.isFinite(pixelRatio) && pixelRatio > 0
      ? pixelRatio
      : 1;
    const next: ResizeIdentity = [
      Math.round(width),
      Math.round(height),
      Math.round(finitePixelRatio * 1000) / 1000,
      variant,
    ];
    if (frame === null && sameIdentity(applied, next)) return;
    pending = { identity: next, measurement: [width, height, finitePixelRatio] };
    setPreview(true);
    if (idle !== null) { clearIdle(idle); idle = null; }
    if (frame !== null) return;
    frame = requestFrame(() => {
      frame = null;
      if (disposed || !pending) return;
      const size = pending;
      pending = null;
      if (!sameIdentity(applied, size.identity)) {
        applied = size.identity;
        // Keeping DPR in this scheduler's identity is what makes a same-size
        // move between a standard and a Retina display update the backing
        // store; keeping the variant in it is what makes entering and leaving
        // a gesture do the same without the container ever changing size.
        options.resize(...size.measurement);
      }
      scheduleIdle();
    });
  };
  return {
    notify,
    dispose: () => {
      disposed = true;
      if (frame !== null) cancelFrame(frame);
      if (idle !== null) clearIdle(idle);
      frame = null;
      idle = null;
      pending = null;
      setPreview(false);
    },
  };
}

interface SizableRenderWindow {
  getApiSpecificRenderWindow: () => { setSize: (width: number, height: number) => void };
  getRenderWindow: () => { render: () => void };
  getInteractor: () => { getInteractorStyle: () => InteractionEvents | null };
}

export interface VtkResizeOptions {
  /** Pixel budget for this class of scene — see lib/vtk/vtkResolution.ts. */
  policy: ResolutionPolicy;
  onPreviewChange?: (preview: boolean) => void;
}

interface ResolutionMediaQuery {
  addEventListener?: (type: "change", listener: () => void) => void;
  removeEventListener?: (type: "change", listener: () => void) => void;
  addListener?: (listener: () => void) => void;
  removeListener?: (listener: () => void) => void;
}

const resolutionScales = new WeakMap<object, number>();
const resolutionUpdates = new WeakMap<object, (draw: boolean) => void>();

export function setVtkResolutionScale(renderWindow: object, scale: number): void {
  if (!Number.isFinite(scale) || scale < 0.5 || scale > 1) throw new Error("Invalid VTK preview resolution scale");
  if (scale === 1) resolutionScales.delete(renderWindow); else resolutionScales.set(renderWindow, scale);
  resolutionUpdates.get(renderWindow)?.(false);
}

export function attachVtkResize(
  container: { getBoundingClientRect: () => { width: number; height: number } },
  renderWindow: SizableRenderWindow,
  options: VtkResizeOptions,
): () => void {
  const { policy, onPreviewChange } = options;
  let appliedSize: BackingStoreSize | null = null;

  const ResizeObserverConstructor = (globalThis as unknown as {
    ResizeObserver: new (
      callback: (entries: Array<{ contentRect: { width: number; height: number } }>) => void,
    ) => { observe: (target: unknown) => void; disconnect: () => void };
  }).ResizeObserver;
  const display = globalThis as unknown as {
    devicePixelRatio?: number;
    matchMedia?: (query: string) => ResolutionMediaQuery;
    addEventListener?: (type: "resize", listener: () => void) => void;
    removeEventListener?: (type: "resize", listener: () => void) => void;
  };
  const currentPixelRatio = () => {
    const ratio = display.devicePixelRatio ?? 1;
    return Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
  };
  /** Size the buffer under the policy; replaces GenericRenderWindow.resize() (no pixel ceiling). */
  const applySize = (draw = true, measurement?: { width: number; height: number; pixelRatio: number }) => {
    const rect = measurement ?? container.getBoundingClientRect();
    let size = backingStoreSize(
      rect.width,
      rect.height,
      measurement?.pixelRatio ?? currentPixelRatio(),
      policy,
    );
    if (!size) return;
    const scale = resolutionScales.get(renderWindow) ?? 1;
    if (scale < 1) {
      const ratio = Math.max(MIN_PIXEL_RATIO, size.pixelRatio * scale);
      size = { width: Math.max(1, Math.floor(rect.width * ratio)), height: Math.max(1, Math.floor(rect.height * ratio)), pixelRatio: ratio };
    }
    if (appliedSize
      && appliedSize.width === size.width
      && appliedSize.height === size.height) return;
    appliedSize = size;
    renderWindow.getApiSpecificRenderWindow().setSize(size.width, size.height);
    if (draw) renderWindow.getRenderWindow().render();
  };
  resolutionUpdates.set(renderWindow, applySize);

  const scheduler = createVtkResizeScheduler({
    // A later layout read must not allocate a different size from the identity the scheduler records.
    resize: (width, height, pixelRatio) => applySize(true, { width, height, pixelRatio }),
    onPreviewChange,
  });

  const notifyContainer = () => {
    const rect = container.getBoundingClientRect();
    scheduler.notify(rect.width, rect.height, currentPixelRatio());
  };
  const observer = new ResizeObserverConstructor((entries) => {
    const entry = entries.at(-1);
    if (entry) {
      scheduler.notify(
        entry.contentRect.width,
        entry.contentRect.height,
        currentPixelRatio(),
      );
    }
  });
  observer.observe(container);

  let resolutionQuery: ResolutionMediaQuery | null = null;
  const removeResolutionListener = () => {
    resolutionQuery?.removeEventListener?.("change", resolutionListener);
    resolutionQuery?.removeListener?.(resolutionListener);
    resolutionQuery = null;
  };
  const armResolutionListener = () => {
    removeResolutionListener();
    resolutionQuery = display.matchMedia?.(
      `(resolution: ${currentPixelRatio()}dppx)`,
    ) ?? null;
    resolutionQuery?.addEventListener?.("change", resolutionListener);
    if (!resolutionQuery?.addEventListener) {
      resolutionQuery?.addListener?.(resolutionListener);
    }
  };
  function resolutionListener(): void {
    notifyContainer();
    // The old query no longer describes the active display. Re-arm against
    // the new DPR so subsequent monitor moves and browser zoom changes work.
    armResolutionListener();
  }
  const windowResizeListener = () => notifyContainer();
  display.addEventListener?.("resize", windowResizeListener);
  armResolutionListener();

  // Do NOT resize the backing store mid-gesture: vtk.js measures pointer deltas
  // in backing-store pixels against a private previousPosition, and rescaling
  // mid-drag produces wildly incorrect camera rotation angles.
  notifyContainer();
  return () => {
    resolutionUpdates.delete(renderWindow); resolutionScales.delete(renderWindow);
    observer.disconnect();
    removeResolutionListener();
    display.removeEventListener?.("resize", windowResizeListener);
    scheduler.dispose();
  };
}

export interface VolumeMapperQuality {
  setSampleDistance: (distance: number) => unknown;
  setMaximumSamplesPerRay: (samples: number) => unknown;
  /** Optional screen-space ray density control. vtk.js renders to a smaller
   * intermediate viewport when this is greater than one, then composites it
   * at the canvas size. Unlike increasing sampleDistance, this does not move
   * the samples through the scientific volume or change thin-feature opacity. */
  setImageSampleDistance?: (distance: number) => unknown;
}

export interface VtkVolumeQualityController {
  begin: (reason: "interaction" | "resize" | "synchronization") => void;
  end: (reason: "interaction" | "resize" | "synchronization") => void;
  setFullSampleDistance: (distance: number) => void;
  /** Set one volume's settled spacing without overwriting the sampling policy
   * of sibling volumes governed by the same interaction lifecycle. */
  setFullSampleDistanceFor: (mapper: VolumeMapperQuality, distance: number) => void;
  registerMapper: (mapper: VolumeMapperQuality, fullSampleDistance?: number) => () => void;
  holdFullQuality: () => () => void;
  dispose: () => void;
  isPreview: () => boolean;
}

export function createVtkVolumeQualityController(options: {
  mapper: VolumeMapperQuality | readonly VolumeMapperQuality[];
  render: () => void;
  onPreviewChange?: (preview: boolean) => void;
  fullMaximumSamples?: number;
  previewMaximumSamples?: number;
  previewDistanceMultiplier?: number;
  flatPreview?: boolean;
  fullImageSampleDistance?: number;
  previewImageSampleDistance?: number;
  idleMs?: number;
  setIdle?: (callback: () => void, milliseconds: number) => TimerHandle;
  clearIdle?: (handle: TimerHandle) => void;
}): VtkVolumeQualityController {
  const mappers = new Set(Array.isArray(options.mapper)
    ? options.mapper
    : [options.mapper]);
  const fullMaximum = options.fullMaximumSamples ?? 3000;
  const previewMaximum = options.previewMaximumSamples ?? 1200;
  const multiplier = options.previewDistanceMultiplier ?? 2;
  const fullImageSampleDistance = options.fullImageSampleDistance ?? 1;
  const previewImageSampleDistance = options.previewImageSampleDistance ?? 1;
  const setIdle = options.setIdle ?? setTimeout;
  const clearIdle = options.clearIdle ?? clearTimeout;
  const reasons = new Set<"interaction" | "resize" | "synchronization">();
  const fullDistances = new Map<VolumeMapperQuality, number>(
    [...mappers].map((mapper) => [mapper, 5000]),
  );
  for (const mapper of mappers) { setSettledVolumeDistance(mapper, 5000); setFlatVolumePreview(mapper, false); }
  let preview = false;
  let desiredPreview = false;
  let fullQualityHolds = 0;
  let disposed = false;
  let idle: TimerHandle | null = null;

  const applyActual = (nextPreview: boolean, shouldRender = true) => {
    if (disposed || preview === nextPreview) return;
    preview = nextPreview;
    for (const mapper of mappers) {
      setFlatVolumePreview(mapper, preview && options.flatPreview === true);
      mapper.setSampleDistance(
        (fullDistances.get(mapper) ?? 5000) * (preview ? multiplier : 1),
      );
      mapper.setMaximumSamplesPerRay(preview ? previewMaximum : fullMaximum);
      mapper.setImageSampleDistance?.(
        preview ? previewImageSampleDistance : fullImageSampleDistance,
      );
    }
    options.onPreviewChange?.(preview);
    if (shouldRender) options.render();
  };
  const apply = (nextPreview: boolean, shouldRender = true) => {
    desiredPreview = nextPreview;
    applyActual(fullQualityHolds > 0 ? false : nextPreview, shouldRender);
  };
  return {
    begin: (reason) => {
      if (disposed) return;
      reasons.add(reason);
      if (idle !== null) { clearIdle(idle); idle = null; }
      // Resize and camera interaction each perform their own imminent render;
      // only change mapper policy here so entering preview adds no extra frame.
      apply(true, false);
    },
    end: (reason) => {
      if (disposed) return;
      reasons.delete(reason);
      if (reasons.size > 0) return;
      if (idle !== null) clearIdle(idle);
      idle = setIdle(() => {
        idle = null;
        if (reasons.size === 0) apply(false);
      }, options.idleMs ?? 180);
    },
    setFullSampleDistance: (distance) => {
      if (!(distance > 0) || !Number.isFinite(distance)) return;
      for (const mapper of mappers) {
        setSettledVolumeDistance(mapper, distance);
        fullDistances.set(mapper, distance);
        mapper.setSampleDistance(distance * (preview ? multiplier : 1));
      }
    },
    setFullSampleDistanceFor: (mapper, distance) => {
      if (!(distance > 0) || !Number.isFinite(distance) || !fullDistances.has(mapper)) return;
      setSettledVolumeDistance(mapper, distance);
      fullDistances.set(mapper, distance);
      mapper.setSampleDistance(distance * (preview ? multiplier : 1));
    },
    registerMapper: (mapper, fullSampleDistance = 5000) => {
      if (disposed) return () => undefined;
      mappers.add(mapper);
      setFlatVolumePreview(mapper, preview && options.flatPreview === true);
      setSettledVolumeDistance(mapper, fullSampleDistance);
      fullDistances.set(mapper, fullSampleDistance);
      mapper.setSampleDistance(fullSampleDistance * (preview ? multiplier : 1));
      mapper.setMaximumSamplesPerRay(preview ? previewMaximum : fullMaximum);
      mapper.setImageSampleDistance?.(
        preview ? previewImageSampleDistance : fullImageSampleDistance,
      );
      return () => {
        mappers.delete(mapper);
        fullDistances.delete(mapper);
      };
    },
    holdFullQuality: () => {
      if (disposed) return () => {};
      fullQualityHolds++;
      applyActual(false, false);
      let released = false;
      return () => {
        if (released) return;
        released = true;
        fullQualityHolds--;
        if (!disposed && fullQualityHolds === 0) applyActual(desiredPreview, false);
      };
    },
    dispose: () => {
      const wasPreview = preview;
      disposed = true;
      if (idle !== null) clearIdle(idle);
      idle = null;
      reasons.clear();
      preview = false;
      if (wasPreview) options.onPreviewChange?.(false);
    },
    isPreview: () => preview,
  };
}

interface Subscription { unsubscribe: () => void; }
interface InteractionEvents {
  onStartInteractionEvent: (callback: () => void) => Subscription;
  onInteractionEvent?: (callback: () => void) => Subscription;
  onEndInteractionEvent: (callback: () => void) => Subscription;
}
/** The subset of a vtk.js render-window interactor this module subscribes to. */
interface DragInteractor {
  getInteractorStyle: () => DragStyle | null;
  /** Generated by the event macro for every entry in `handledEvents`. */
  onStartMouseMove?: (callback: (callData: unknown) => void) => Subscription;
}

interface InteractionWindow {
  getInteractor: () => {
    getInteractorStyle: () => (InteractionEvents & DragStyle) | null;
    onStartMouseMove?: (callback: (callData: unknown) => void) => Subscription;
  };
}

const volumeQualityControllers = new WeakMap<object, VtkVolumeQualityController>();
export function holdVtkFullQuality(renderWindow: object): () => void {
  return volumeQualityControllers.get(renderWindow)?.holdFullQuality() ?? (() => {});
}

export function attachVtkInteractionQuality(
  renderWindow: InteractionWindow,
  quality: VtkVolumeQualityController,
  options: { onStartInteraction?: () => void; onEndInteraction?: () => void } = {},
): () => void {
  // vtk.js declares similarly named events on vtkRenderWindowInteractor, but
  // camera Start/EndInteractionEvent is emitted by vtkInteractorStyle. Using
  // the render-window interactor here throws at runtime in vtk.js 36 even
  // though its declaration file suggests those methods exist.
  const style = renderWindow.getInteractor().getInteractorStyle();
  volumeQualityControllers.set(renderWindow, quality);
  const dispose = () => {
    if (volumeQualityControllers.get(renderWindow) === quality) volumeQualityControllers.delete(renderWindow);
    quality.dispose();
  };
  if (!style) return dispose;
  const start = style.onStartInteractionEvent(() => {
    options.onStartInteraction?.();
    quality.begin("interaction");
  });
  const end = style.onEndInteractionEvent(() => {
    quality.end("interaction");
    options.onEndInteraction?.();
  });
  return () => {
    start.unsubscribe();
    end.unsubscribe();
    dispose();
  };
}

interface StabilizedRenderer {
  getActiveCamera: () => {
    getPosition?: () => ArrayLike<number>;
    getFocalPoint?: () => ArrayLike<number>;
    getDirectionOfProjection?: () => ArrayLike<number>;
    getParallelProjection?: () => boolean;
    setPosition?: (x: number, y: number, z: number) => boolean;
    setViewUp: (x: number, y: number, z: number) => boolean;
    orthogonalizeViewUp?: () => void;
  };
  computeVisiblePropBounds: () => [number, number, number, number, number, number];
  resetCameraClippingRange: (bounds?: [number, number, number, number, number, number]) => boolean;
  setNearClippingPlaneTolerance?: (tolerance: number) => boolean;
}

export interface VtkCameraStabilityOptions {
  /**
   * vtk.js defaults the near plane to at least 1% of the far plane. That is a
   * useful depth-precision tradeoff for compact geometry, but a volume scene
   * with a domain-sized context box can have a far plane hundreds of kilometres
   * away while the camera is inspecting cloud structure a few hundred metres
   * away. In that case the default near plane clips the entire subject.
   */
  nearClippingPlaneTolerance?: number;
  /** Scientific-data bounds to use instead of domain boxes, axes, and other
   * reference props when calculating the close-inspection clipping range. */
  getClippingBounds?: () => [number, number, number, number, number, number] | null;
  /** Perspective cameras become numerically singular when dolly collapses the
   * eye onto the focal point. This fraction of the clipping-bounds diagonal is
   * the closest permitted inspection distance. */
  minimumViewDistanceRatio?: number;
  /** A volume quality controller may already own the settled idle render. */
  renderOnEnd?: boolean;
}

type CameraBounds = [number, number, number, number, number, number];

function validCameraBounds(bounds: ArrayLike<number> | null | undefined): bounds is CameraBounds {
  return bounds != null
    && bounds.length === 6
    && Array.from(bounds).every(Number.isFinite)
    && bounds[1]! >= bounds[0]!
    && bounds[3]! >= bounds[2]!
    && bounds[5]! >= bounds[4]!;
}

/** Keep perspective zoom finite without preventing close or inside-volume
 * inspection. vtk.js's volume shader explicitly clips a ray's entry distance
 * to zero when its origin is inside the volume. Treating the image bounds as
 * a collision body therefore rejects valid views and, for sparse full-domain
 * fields, can place the collision face tens of kilometres from visible data.
 * Returns true only when the eye has actually collapsed onto the focal point. */
export function constrainPerspectiveCameraDistance(
  camera: {
    getPosition?: () => ArrayLike<number>;
    getFocalPoint?: () => ArrayLike<number>;
    getDirectionOfProjection?: () => ArrayLike<number>;
    getParallelProjection?: () => boolean;
    setPosition?: (x: number, y: number, z: number) => boolean;
  },
  bounds: CameraBounds,
  minimumRatio = 1e-4,
): boolean {
  if (camera.getParallelProjection?.()) return false;
  const position = camera.getPosition?.();
  const focal = camera.getFocalPoint?.();
  if (!position || !focal || !camera.setPosition) return false;
  const diagonal = Math.hypot(
    bounds[1] - bounds[0],
    bounds[3] - bounds[2],
    bounds[5] - bounds[4],
  );
  if (!(diagonal > 0) || !Number.isFinite(diagonal)) return false;
  const minimumDistance = Math.max(1e-3, diagonal * Math.max(1e-8, minimumRatio));
  const dx = focal[0]! - position[0]!;
  const dy = focal[1]! - position[1]!;
  const dz = focal[2]! - position[2]!;
  const distance = Math.hypot(dx, dy, dz);
  const fallback = camera.getDirectionOfProjection?.();
  const rawDirection: [number, number, number] = distance > 1e-12
    // `dx` points from the eye to the focus. The collision calculation below
    // needs the opposite vector: from the focus out toward the eye.
    ? [-dx / distance, -dy / distance, -dz / distance]
    : fallback && fallback.length >= 3
      ? [-fallback[0]!, -fallback[1]!, -fallback[2]!]
      : [0, 0, 1];
  const directionLength = Math.hypot(...rawDirection);
  const direction: [number, number, number] = directionLength > 1e-12
    ? rawDirection.map((value) => value / directionLength) as [number, number, number]
    : [0, 0, 1];

  if (Number.isFinite(distance) && distance >= minimumDistance) return false;

  camera.setPosition(
    focal[0]! + direction[0] * minimumDistance,
    focal[1]! + direction[1] * minimumDistance,
    focal[2]! + direction[2] * minimumDistance,
  );
  return true;
}

/**
 * Refresh clipping range and correct roll on each gesture-end.
 * Also installs `enableContinuousCameraDrag` so drags work on slow frames.
 */
export function attachVtkCameraStability(
  renderWindow: InteractionWindow,
  renderer: StabilizedRenderer,
  render: () => void,
  options: VtkCameraStabilityOptions = {},
): () => void {
  const style = renderWindow.getInteractor().getInteractorStyle();
  if (!style) return () => {};
  const nearTolerance = options.nearClippingPlaneTolerance;
  if (nearTolerance != null && Number.isFinite(nearTolerance) && nearTolerance > 0) {
    renderer.setNearClippingPlaneTolerance?.(nearTolerance);
  }
  const detachDrag = enableContinuousCameraDrag(renderWindow);
  const stabilize = (shouldRender: boolean) => {
    const camera = renderer.getActiveCamera();
    const position = camera.getPosition?.();
    const focalPoint = camera.getFocalPoint?.();
    const dx = position && focalPoint ? focalPoint[0]! - position[0]! : 1;
    const dy = position && focalPoint ? focalPoint[1]! - position[1]! : 0;
    const dz = position && focalPoint ? focalPoint[2]! - position[2]! : 0;
    const length = Math.hypot(dx, dy, dz);
    // At the vertical pole, atmospheric +z is parallel to the viewing
    // direction and vtk.js collapses the camera basis to a zero vector. Use
    // geographic north as screen-up there; away from the pole keep z upright.
    const polar = length > 0 && Math.abs(dz / length) >= 0.985;
    camera.setViewUp(0, polar ? 1 : 0, polar ? 0 : 1);
    camera.orthogonalizeViewUp?.();
    const requestedBounds = options.getClippingBounds?.();
    const visibleBounds = renderer.computeVisiblePropBounds();
    const bounds = validCameraBounds(requestedBounds) ? requestedBounds : visibleBounds;
    if (validCameraBounds(bounds)) {
      constrainPerspectiveCameraDistance(
        camera,
        bounds,
        options.minimumViewDistanceRatio,
      );
    }
    // An empty scene reports inverted sentinel bounds; handing those to
    // resetCameraClippingRange would install a degenerate range.
    if (validCameraBounds(bounds)) {
      renderer.resetCameraClippingRange(bounds);
    } else {
      renderer.resetCameraClippingRange();
    }
    if (shouldRender) render();
  };
  // vtk.js recalculates clipping against every visible prop during wheel and
  // rotate events. Override that immediately, before the interactor presents
  // the frame, so a domain box cannot create one black close-zoom frame.
  const interaction = style.onInteractionEvent?.(() => stabilize(false));
  const end = style.onEndInteractionEvent(() => stabilize(options.renderOnEnd !== false));
  return () => {
    interaction?.unsubscribe();
    end.unsubscribe();
    detachDrag();
  };
}

/** The subset of a vtk.js interactor style this module drives. */
interface DragStyle {
  handleMouseMove?: (callData: unknown) => unknown;
  /** Absent in vtk.js 36; present if a later release handles the event itself. */
  handleStartMouseMove?: (callData: unknown) => unknown;
}

/**
 * Route vtk.js StartMouseMove into handleMouseMove to fix drag freezing on slow
 * frames (vtk.js 36: StartMouseMove is dropped by the interactor style).
 */
export function enableContinuousCameraDrag(
  renderWindow: { getInteractor: () => DragInteractor },
): () => void {
  const interactor = renderWindow.getInteractor();
  const style = interactor.getInteractorStyle();
  if (!style || typeof style.handleMouseMove !== "function") return () => {};
  // A later vtk.js handling the event itself would deliver every move twice.
  if (typeof style.handleStartMouseMove === "function") return () => {};
  if (typeof interactor.onStartMouseMove !== "function") return () => {};

  const subscription = interactor.onStartMouseMove((callData) => {
    style.handleMouseMove?.(callData);
  });
  return () => subscription.unsubscribe();
}

/**
 * Configure the explicit transparency controls exposed by the active backend.
 * vtk.js 36's OpenGL path has no renderer depth-peeling API; it selects its
 * weighted blended OIT pass automatically when floating-point targets exist.
 * The optional calls below are therefore a forward/alternate-backend hook, not
 * a claim that depth peeling is active in the current renderer.
 */
export function configureTranslucentRendering(
  renderer: {
    setUseDepthPeeling?: (value: boolean) => void;
    setMaximumNumberOfPeels?: (value: number) => void;
    setOcclusionRatio?: (value: number) => void;
  },
  maximumPeels = 4,
): void {
  renderer.setUseDepthPeeling?.(true);
  renderer.setMaximumNumberOfPeels?.(maximumPeels);
  // Stop peeling once the remaining contribution is negligible; without a
  // non-zero ratio every scene pays all N passes whether or not they change
  // a pixel.
  renderer.setOcclusionRatio?.(0.1);
}
