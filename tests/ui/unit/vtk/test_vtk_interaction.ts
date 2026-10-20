import { readFileSync } from "node:fs";
import { setCameraAdjustedVolumeRayDistances } from "../../../../ui/src/lib/vtk/volumeOpticalDepth.ts";
import {
  attachVtkCameraStability,
  attachVtkInteractionQuality,
  attachVtkResize,
  setVtkResolutionScale,
  constrainPerspectiveCameraDistance,
  createVtkResizeScheduler,
  createVtkVolumeQualityController,
  enableContinuousCameraDrag,
} from "../../../../ui/src/lib/vtk/vtkInteraction.ts";
import {
  SURFACE_RESOLUTION_POLICY,
  VOLUME_RESOLUTION_POLICY,
} from "../../../../ui/src/lib/vtk/vtkResolution.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else { failed++; failures.push(`FAIL ${label}`); }
}

{
  const firstDistances: number[] = [];
  const secondDistances: number[] = [];
  const firstImageDistances: number[] = [];
  const secondImageDistances: number[] = [];
  let viewProperties: Record<string, unknown> = { OpenGL: { ShaderReplacements: ["preserved"] } };
  let propertyWrites = 0;
  const firstMapper = {
    setSampleDistance: (distance: number) => firstDistances.push(distance),
    setMaximumSamplesPerRay: () => {},
    setImageSampleDistance: (distance: number) => firstImageDistances.push(distance),
    getViewSpecificProperties: () => viewProperties,
    setViewSpecificProperties: (properties: Record<string, unknown>) => { viewProperties = properties; propertyWrites++; },
  };
  const secondMapper = {
    setSampleDistance: (distance: number) => secondDistances.push(distance),
    setMaximumSamplesPerRay: () => {},
    setImageSampleDistance: (distance: number) => secondImageDistances.push(distance),
  };
  const quality = createVtkVolumeQualityController({
    mapper: [firstMapper, secondMapper],
    render: () => {},
    previewImageSampleDistance: 2,
  });
  quality.setFullSampleDistance(200);
  quality.setFullSampleDistanceFor(firstMapper, 100);
  const dynamicDistances: number[] = [];
  const unregisterDynamic = quality.registerMapper({
    setSampleDistance: (distance: number) => dynamicDistances.push(distance),
    setMaximumSamplesPerRay: () => {},
  }, 75);
  setCameraAdjustedVolumeRayDistances(
    { setScalarOpacityUnitDistance: () => {} },
    { getDirectionOfProjection: () => [0, 0, 1] },
    quality, 100, 50, { xy: 1, z: 2 }, firstMapper,
  );
  check(firstDistances.at(-1) === 100 && secondDistances.at(-1) === 200
      && dynamicDistances.at(-1) === 75,
    "physical-ray correction changes only the owning mapper under vertical exaggeration");
  const optical = viewProperties.OpenGL as Record<string, unknown>;
  check(optical.nimbusPhysicalReference === 100 && JSON.stringify(optical.nimbusInverseSceneScale) === "[1,1,0.5]",
    "fragment optical correction receives physical reference and inverse display scale");
  setCameraAdjustedVolumeRayDistances(
    { setScalarOpacityUnitDistance: () => {} }, { getDirectionOfProjection: () => [1, 0, 0] },
    quality, 100, 50, { xy: 1, z: 2 }, firstMapper,
  );
  check(propertyWrites === 1 && JSON.stringify(optical.ShaderReplacements) === '["preserved"]',
    "camera motion preserves shader identity and does not rewrite optical metadata");
  setCameraAdjustedVolumeRayDistances(
    { setScalarOpacityUnitDistance: () => {} }, { getDirectionOfProjection: () => [0, 0, 1] },
    quality, 100, 50, { xy: 1, z: 2 }, firstMapper,
  );
  quality.begin("interaction");
  check(
    firstDistances.at(-1) === 200 && secondDistances.at(-1) === 400,
    "multi-volume quality preserves each mapper's settled ray spacing in preview",
  );
  check(
    firstImageDistances.at(-1) === 2 && secondImageDistances.at(-1) === 2,
    "volume preview reduces screen-space ray density without dropping a mapper",
  );
  check(dynamicDistances.at(-1) === 150,
    "a dynamically registered volume brick joins the active preview policy");
  unregisterDynamic();
  quality.end("interaction");
  quality.dispose();
}

class FakeClock {
  next = 1;
  frames = new Map<number, (timestamp: number) => void>();
  timers = new Map<number, () => void>();
  requestFrame = (callback: (timestamp: number) => void): number => {
    const id = this.next++;
    this.frames.set(id, callback);
    return id;
  };
  cancelFrame = (id: number): void => { this.frames.delete(id); };
  setIdle = (callback: () => void): number => {
    const id = this.next++;
    this.timers.set(id, callback);
    return id;
  };
  clearIdle = (id: number | ReturnType<typeof setTimeout>): void => {
    if (typeof id === "number") this.timers.delete(id);
  };
  flushFrame(): void {
    const entries = [...this.frames.entries()];
    this.frames.clear();
    for (const [, callback] of entries) callback(0);
  }
  flushIdle(): void {
    const callbacks = [...this.timers.values()];
    this.timers.clear();
    for (const callback of callbacks) callback();
  }
}

{
  const clock = new FakeClock();
  let resizes = 0;
  const previews: boolean[] = [];
  const scheduler = createVtkResizeScheduler({
    resize: () => { resizes++; },
    onPreviewChange: (preview) => previews.push(preview),
    requestFrame: clock.requestFrame,
    cancelFrame: clock.cancelFrame,
    setIdle: clock.setIdle,
    clearIdle: clock.clearIdle,
  });
  scheduler.notify(0, 100);
  scheduler.notify(100, 0);
  check(clock.frames.size === 0, "zero-sized transitions never schedule VTK work");
  scheduler.notify(100, 100);
  scheduler.notify(110, 100);
  scheduler.notify(120, 100);
  check(clock.frames.size === 1, "a resize storm coalesces to one animation frame");
  check(previews.join(",") === "true", "resize activity enters preview once");
  clock.flushFrame();
  check(resizes === 1, "one coalesced resize performs one GenericRenderWindow render");
  clock.flushIdle();
  check(previews.join(",") === "true,false", "resize preview clears only after idle");
  scheduler.notify(120, 100);
  check(clock.frames.size === 0, "an unchanged size is ignored");
  scheduler.notify(120, 100, 2);
  check(clock.frames.size === 1, "a DPR-only display change schedules a backing-store resize");
  clock.flushFrame();
  check(resizes === 2, "a same-size move from 1x to 2x resizes the backing store");
  scheduler.notify(120, 100, 2);
  check(clock.frames.size === 0, "an unchanged CSS size and DPR are both ignored");

  // A camera gesture resolves the SAME css size and DPR to a smaller backing
  // store. Without the variant in the identity, the resize that makes a drag
  // affordable would be coalesced away as a no-op.
  scheduler.notify(120, 100, 2, 1);
  check(clock.frames.size === 1, "entering gesture resolution schedules a resize at an unchanged size");
  clock.flushFrame();
  check(resizes === 3, "gesture resolution is actually applied");
  scheduler.notify(120, 100, 2, 1);
  check(clock.frames.size === 0, "staying in a gesture does not re-resize every event");
  scheduler.notify(120, 100, 2, 0);
  check(clock.frames.size === 1, "leaving a gesture schedules the restoring resize");
  clock.flushFrame();
  check(resizes === 4, "at-rest resolution is restored");

  scheduler.notify(121, 100);
  scheduler.dispose();
  check(clock.frames.size === 0, "dispose cancels a pending resize frame");
}

{
  const clock = new FakeClock();
  const distances: number[] = [];
  const maxima: number[] = [];
  const previews: boolean[] = [];
  let renders = 0;
  const quality = createVtkVolumeQualityController({
    mapper: {
      setSampleDistance: (distance) => distances.push(distance),
      setMaximumSamplesPerRay: (maximum) => maxima.push(maximum),
    },
    render: () => { renders++; },
    onPreviewChange: (preview) => previews.push(preview),
    setIdle: clock.setIdle,
    clearIdle: clock.clearIdle,
  });
  quality.begin("interaction");
  quality.begin("resize");
  check(quality.isPreview(), "resize/camera activity enters explicit preview quality");
  check(distances.at(-1) === 10_000 && maxima.at(-1) === 1200, "preview doubles ray spacing and caps samples");
  check(renders === 0, "entering preview relies on the imminent interaction/resize render");
  quality.setFullSampleDistance(250);
  check(distances.at(-1) === 500, "new field spacing preserves the active preview multiplier");
  quality.end("resize");
  check(clock.timers.size === 0, "quality cannot restore while camera interaction remains active");
  quality.end("interaction");
  check(clock.timers.size === 1, "last activity schedules one idle restoration");
  clock.flushIdle();
  check(!quality.isPreview(), "idle restores full scientific quality");
  check(distances.at(-1) === 250 && maxima.at(-1) === 3000, "restoration exactly reapplies full ray settings");
  check(previews.join(",") === "true,false", "preview policy is externally observable for labeling");
  quality.dispose();
}

{
  let startCallback: (() => void) | null = null;
  let endCallback: (() => void) | null = null;
  let unsubscribes = 0;
  const begins: string[] = [];
  const ends: string[] = [];
  let disposals = 0;
  let userStarts = 0;
  const detach = attachVtkInteractionQuality(
    {
      getInteractor: () => ({
        getInteractorStyle: () => ({
          onStartInteractionEvent: (callback) => {
            startCallback = callback;
            return { unsubscribe: () => { unsubscribes++; } };
          },
          onEndInteractionEvent: (callback) => {
            endCallback = callback;
            return { unsubscribe: () => { unsubscribes++; } };
          },
        }),
      }),
    },
    {
      begin: (reason) => begins.push(reason),
      end: (reason) => ends.push(reason),
      setFullSampleDistance: () => {},
      setFullSampleDistanceFor: () => {},
      holdFullQuality: () => () => {},
      registerMapper: () => () => {},
      dispose: () => { disposals++; },
      isPreview: () => false,
    },
    { onStartInteraction: () => { userStarts++; } },
  );
  check(startCallback !== null && endCallback !== null, "camera quality subscribes on vtk interactor style events");
  (startCallback as (() => void) | null)?.();
  (endCallback as (() => void) | null)?.();
  check(begins.join(",") === "interaction", "style start event enters interaction preview");
  check(userStarts === 1, "style start event reports an explicit user camera touch");
  check(ends.join(",") === "interaction", "style end event settles interaction preview");
  detach();
  check(unsubscribes === 2, "interaction detach releases both style subscriptions");
  check(disposals === 1, "interaction detach disposes the quality controller");
}

{
  const scenes: Array<[string, number]> = [
    ["VolumeRender3D.tsx", 1],
    ["Isosurface3D.tsx", 1],
    ["Streamlines3D.tsx", 1],
    ["Trajectories3D.tsx", 1],
    ["SupercellRender3D.tsx", 1],
  ];
  for (const [file, expected] of scenes) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    check(!source.includes("new ResizeObserver"), `${file} has no direct ResizeObserver`);
    check(!source.includes("renderWindow.resize()"), `${file} has no direct uncoalesced resize`);
    check(
      (source.match(/vtkRenderWindowOptions\(/g)?.length ?? 0) === expected,
      `${file} routes every render-window construction through the resize-safe backend options`,
    );
    check(
      (source.match(/attachVtkResize\(/g)?.length ?? 0) === expected,
      `${file} attaches the shared resize owner to every scene`,
    );
    check(
      (source.match(/exposeVtkBackendEvidence\(/g)?.length ?? 0) === expected,
      `${file} confirms the initialized API-specific renderer before reporting it active`,
    );
  }
  const backendOptionsSource = readFileSync(
    new URL("../../../../ui/src/lib/vtk/vtkBackend.ts", import.meta.url),
    "utf8",
  );
  check(
    backendOptionsSource.includes("listenWindowResize: false"),
    "the shared render-window options disable vtk.js's implicit window resize listener",
  );
  check(
    backendOptionsSource.includes("workload === \"volume\"")
      && backendOptionsSource.includes("VTK_WEBGPU_VOLUME_LIMITATION"),
    "backend qualification keeps feature-incomplete WebGPU volume rendering off the correctness path",
  );
  check(
    backendOptionsSource.includes("onInitialized?.(confirm)")
      && backendOptionsSource.includes("onDeviceLost?.")
      && backendOptionsSource.includes("did not initialize within 15 seconds"),
    "backend status waits for vtk.js initialization and falls back on timeout or device loss",
  );
  for (const file of ["VolumeRender3D.tsx", "SupercellRender3D.tsx"]) {
    const source = readFileSync(new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url), "utf8");
    check(source.includes("attachVtkInteractionQuality"), `${file} observes camera interaction quality`);
    check(source.includes("Preview while moving · settles to full quality"), `${file} labels reduced quality in control chrome`);
    check(!source.includes("volume-render__quality-badge"), `${file} keeps preview status off the scientific canvas`);
  }
}

/* ---------- Camera stability during and after a gesture ----------
 * vtk.js's trackball moves the camera without revisiting the clipping range, so
 * dollying in slices geometry against a stale near plane and the scene appears
 * to dissolve. It also lets roll accumulate, which leaves an atmospheric domain
 * with no usable "up". Both are corrected once on interaction end. */
{
  type EndCallback = () => void;
  const makeStyle = () => {
    const interactions: EndCallback[] = [];
    const ends: EndCallback[] = [];
    return {
      style: {
        onStartInteractionEvent: () => ({ unsubscribe: () => {} }),
        onInteractionEvent: (cb: EndCallback) => {
          interactions.push(cb);
          return { unsubscribe: () => { interactions.splice(interactions.indexOf(cb), 1); } };
        },
        onEndInteractionEvent: (cb: EndCallback) => {
          ends.push(cb);
          return { unsubscribe: () => { ends.splice(ends.indexOf(cb), 1); } };
        },
      },
      fireInteraction: () => { for (const cb of [...interactions]) cb(); },
      fire: () => { for (const cb of [...ends]) cb(); },
      count: () => interactions.length + ends.length,
    };
  };

  const makeRenderer = (
    bounds: [number, number, number, number, number, number],
    position: [number, number, number] = [100, -100, 80],
    focalPoint: [number, number, number] = [50, 50, 10],
  ) => {
    const calls = {
      viewUp: [] as number[][],
      orthogonalized: 0,
      clipBounds: [] as unknown[],
      nearTolerances: [] as number[],
      positions: [] as number[][],
    };
    return {
      calls,
      renderer: {
        getActiveCamera: () => ({
          getPosition: () => position,
          getFocalPoint: () => focalPoint,
          getDirectionOfProjection: () => [0, 0, -1],
          getParallelProjection: () => false,
          setPosition: (x: number, y: number, z: number) => {
            position[0] = x;
            position[1] = y;
            position[2] = z;
            calls.positions.push([x, y, z]);
            return true;
          },
          setViewUp: (x: number, y: number, z: number) => { calls.viewUp.push([x, y, z]); return true; },
          orthogonalizeViewUp: () => { calls.orthogonalized += 1; },
        }),
        computeVisiblePropBounds: () => bounds,
        resetCameraClippingRange: (b?: unknown) => { calls.clipBounds.push(b ?? null); return true; },
        setNearClippingPlaneTolerance: (value: number) => {
          calls.nearTolerances.push(value);
          return true;
        },
      },
    };
  };

  const finite: [number, number, number, number, number, number] = [0, 100, 0, 100, 0, 20];
  {
    const s = makeStyle();
    const { renderer, calls } = makeRenderer(finite);
    let renders = 0;
    const detach = attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => { renders += 1; },
    );
    check(calls.clipBounds.length === 0, "camera stability does nothing before a gesture ends");
    check(calls.nearTolerances.length === 0, "surface scenes retain vtk's default near-plane tolerance");
    s.fire();
    check(calls.viewUp.length === 1, "one gesture re-pins the view-up vector exactly once");
    check(
      calls.viewUp[0][0] === 0 && calls.viewUp[0][1] === 0 && calls.viewUp[0][2] === 1,
      "the view-up vector is re-pinned to +z, the atmospheric vertical",
    );
    check(calls.orthogonalized === 1, "accumulated roll is orthogonalized away on gesture end");
    check(calls.clipBounds.length === 1, "the clipping range is recomputed once per gesture");
    check(
      Array.isArray(calls.clipBounds[0]) && (calls.clipBounds[0] as number[])[1] === 100,
      "the clipping range is recomputed over the visible prop bounds",
    );
    check(renders === 1, "the corrected camera is rendered once, not per property change");

    s.fire();
    s.fire();
    check(calls.clipBounds.length === 3, "each subsequent gesture corrects again");

    detach();
    check(s.count() === 0, "detaching removes the interaction and end subscriptions");
    s.fire();
    check(calls.clipBounds.length === 3, "a detached owner does not act on later gestures");
  }

  {
    const s = makeStyle();
    const { renderer, calls } = makeRenderer(finite);
    let renders = 0;
    attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => { renders += 1; },
      { renderOnEnd: false },
    );
    s.fire();
    check(calls.clipBounds.length === 1, "deferred volume settle still commits corrected clipping");
    check(renders === 0, "volume quality ownership suppresses the redundant end render");
  }

  {
    const cameraPosition: [number, number, number] = [50, 50, 1e-12];
    const focalPoint: [number, number, number] = [50, 50, 0];
    const setPositions: number[][] = [];
    const corrected = constrainPerspectiveCameraDistance({
      getPosition: () => cameraPosition,
      getFocalPoint: () => focalPoint,
      getDirectionOfProjection: () => [0, 0, -1],
      getParallelProjection: () => false,
      setPosition: (x, y, z) => {
        setPositions.push([x, y, z]);
        return true;
      },
    }, finite, 1e-4);
    check(corrected && setPositions.length === 1, "a focal-point collision is corrected before it can black the view");
    check(
      Math.hypot(
        setPositions[0][0] - focalPoint[0],
        setPositions[0][1] - focalPoint[1],
        setPositions[0][2] - focalPoint[2],
      ) >= Math.hypot(100, 100, 20) * 1e-4,
      "the corrected eye keeps a scene-relative finite inspection distance",
    );
  }

  {
    const cameraPosition: [number, number, number] = [50, 50, 15];
    const focalPoint: [number, number, number] = [50, 50, 10];
    const setPositions: number[][] = [];
    const corrected = constrainPerspectiveCameraDistance({
      getPosition: () => cameraPosition,
      getFocalPoint: () => focalPoint,
      getDirectionOfProjection: () => [0, 0, -1],
      getParallelProjection: () => false,
      setPosition: (x, y, z) => {
        setPositions.push([x, y, z]);
        return true;
      },
    }, finite, 1e-4);
    check(!corrected && setPositions.length === 0, "an eye inside sparse volume bounds remains available for close inspection");
  }

  {
    const fullDomain: [number, number, number, number, number, number] =
      [0, 51_200, 0, 51_200, 4_200, 16_000];
    const cameraPosition: [number, number, number] = [25_600, 25_600, 8_000];
    const focalPoint: [number, number, number] = [26_000, 25_600, 7_500];
    const setPositions: number[][] = [];
    const corrected = constrainPerspectiveCameraDistance({
      getPosition: () => cameraPosition,
      getFocalPoint: () => focalPoint,
      getDirectionOfProjection: () => [0.625, 0, -0.78125],
      getParallelProjection: () => false,
      setPosition: (x, y, z) => {
        setPositions.push([x, y, z]);
        return true;
      },
    }, fullDomain, 1e-4);
    check(
      !corrected && setPositions.length === 0,
      "a beast-scale sparse support envelope does not become a domain-wide camera collision body",
    );
  }

  {
    const s = makeStyle();
    const fullDomain: [number, number, number, number, number, number] =
      [-100_000, 100_000, -100_000, 100_000, 0, 30_000];
    const stormFocus: [number, number, number, number, number, number] =
      [-10_000, 10_000, -8_000, 8_000, 0, 18_000];
    const { renderer, calls } = makeRenderer(fullDomain, [0, 0, 1e-10], [0, 0, 0]);
    let renders = 0;
    attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => { renders += 1; },
      {
        getClippingBounds: () => stormFocus,
        minimumViewDistanceRatio: 1e-4,
      },
    );
    s.fireInteraction();
    check(
      calls.clipBounds.length === 1 && calls.clipBounds[0] === stormFocus,
      "mid-gesture clipping uses scientific focus bounds instead of the domain box",
    );
    check(calls.positions.length === 1, "mid-gesture zoom is constrained before presentation");
    check(renders === 0, "mid-gesture stabilization relies on the interactor's imminent render");
    s.fire();
    check(renders === 1, "gesture end presents one settled camera frame");
  }

  {
    const s = makeStyle();
    const { renderer, calls } = makeRenderer(finite);
    attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => {},
      { nearClippingPlaneTolerance: 1e-4 },
    );
    check(
      calls.nearTolerances.length === 1 && calls.nearTolerances[0] === 1e-4,
      "large-domain volume scenes can keep close cloud geometry ahead of the near plane",
    );
    s.fire();
    check(calls.clipBounds.length === 1, "volume clipping still refreshes after tolerance configuration");
  }

  {
    const s = makeStyle();
    const { renderer, calls } = makeRenderer(finite, [50, 50, 100], [50, 50, 0]);
    attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => {},
    );
    s.fire();
    check(
      calls.viewUp[0][0] === 0 && calls.viewUp[0][1] === 1 && calls.viewUp[0][2] === 0,
      "a vertical view uses +y as screen-up instead of a degenerate +z vector",
    );
    check(calls.orthogonalized === 1, "the pole-safe view-up is orthogonalized once");
  }

  {
    // An empty scene reports inverted sentinel bounds; passing those through
    // would install a degenerate clip range and blank the next real frame.
    const s = makeStyle();
    const empty: [number, number, number, number, number, number] = [1, -1, 1, -1, 1, -1];
    const { renderer, calls } = makeRenderer(empty);
    attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => {},
    );
    s.fire();
    check(calls.clipBounds[0] === null, "inverted empty-scene bounds fall back to an unbounded reset");
  }

  {
    const s = makeStyle();
    const nonFinite: [number, number, number, number, number, number] =
      [0, Number.NaN, 0, 100, 0, 20];
    const { renderer, calls } = makeRenderer(nonFinite);
    attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => s.style }) },
      renderer,
      () => {},
    );
    s.fire();
    check(calls.clipBounds[0] === null, "non-finite bounds fall back to an unbounded reset");
  }

  {
    // Before the interactor style exists the attach must be inert, not throw:
    // the scenes attach during mount, ahead of the first render.
    const { renderer, calls } = makeRenderer(finite);
    const detach = attachVtkCameraStability(
      { getInteractor: () => ({ getInteractorStyle: () => null }) },
      renderer,
      () => {},
    );
    detach();
    check(calls.clipBounds.length === 0, "a missing interactor style attaches inertly and detaches safely");
  }

  // Every 3D scene must carry it — the defect was scene-wide, and
  // attachVtkInteractionQuality only ever covered the two volume scenes.
  for (const file of [
    "VolumeRender3D.tsx",
    "SupercellRender3D.tsx",
    "Isosurface3D.tsx",
    "Streamlines3D.tsx",
    "Trajectories3D.tsx",
  ]) {
    const source = readFileSync(new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url), "utf8");
    check(source.includes("attachVtkCameraStability("), `${file} stabilizes the camera after a gesture`);
    check(source.includes("detachCameraStability"), `${file} releases the camera-stability subscription`);
  }
}

/* ---------- attachVtkResize owns the backing store ----------
 *
 * End to end, because the defect this replaced was a composition failure
 * rather than a arithmetic one: every individual piece worked, and vtk.js's
 * GenericRenderWindow.resize() still multiplied the container by the raw
 * devicePixelRatio with no ceiling. Testing the policy alone would not have
 * caught that the policy was never consulted.
 *
 * Runs last, and restores the globals it stubs, because it is the only block
 * here that needs a browser surface to exist. */
{
  const browser = globalThis as unknown as Record<string, unknown>;
  const saved = {
    requestAnimationFrame: browser.requestAnimationFrame,
    cancelAnimationFrame: browser.cancelAnimationFrame,
    devicePixelRatio: browser.devicePixelRatio,
    matchMedia: browser.matchMedia,
    ResizeObserver: browser.ResizeObserver,
  };

  const queued: Array<() => void> = [];
  const flushFrames = () => {
    const pending = queued.splice(0, queued.length);
    for (const run of pending) run();
  };
  browser.requestAnimationFrame = (callback: (timestamp: number) => void) => {
    queued.push(() => callback(0));
    return queued.length;
  };
  browser.cancelAnimationFrame = () => {};
  browser.devicePixelRatio = 2;
  browser.matchMedia = () => ({ addEventListener: () => {}, removeEventListener: () => {} });
  // Captures the callback so a test can deliver a container resize, which is
  // the one thing that legitimately re-sizes the backing store.
  type ResizeEntries = Array<{ contentRect: { width: number; height: number } }>;
  // A no-op rather than null: initialising to null lets TypeScript narrow the
  // binding to `never` at the call site, since it cannot see the constructor run.
  let lastResizeObserver: (entries: ResizeEntries) => void = () => {};
  browser.ResizeObserver = class {
    constructor(callback: (entries: ResizeEntries) => void) {
      lastResizeObserver = callback;
    }
    observe(): void {}
    disconnect(): void {}
  };

  const CSS_WIDTH = 1438;
  const CSS_HEIGHT = 941;
  const sizes: Array<[number, number]> = [];
  let renders = 0;
  let unsubscribes = 0;
  const gestureStarts: Array<() => void> = [];
  const gestureEnds: Array<() => void> = [];
  const renderWindow = {
    getApiSpecificRenderWindow: () => ({
      setSize: (width: number, height: number) => { sizes.push([width, height]); },
    }),
    getRenderWindow: () => ({ render: () => { renders++; } }),
    getInteractor: () => ({
      getInteractorStyle: () => ({
        onStartInteractionEvent: (callback: () => void) => {
          gestureStarts.push(callback);
          return { unsubscribe: () => { unsubscribes++; } };
        },
        onEndInteractionEvent: (callback: () => void) => {
          gestureEnds.push(callback);
          return { unsubscribe: () => { unsubscribes++; } };
        },
      }),
    }),
  };
  const container = {
    getBoundingClientRect: () => ({ width: CSS_WIDTH, height: CSS_HEIGHT }),
  };

  const detach = attachVtkResize(container, renderWindow, {
    policy: VOLUME_RESOLUTION_POLICY,
  });
  flushFrames();

  const uncapped = CSS_WIDTH * 2 * CSS_HEIGHT * 2;
  check(sizes.length === 1, "attaching sizes the backing store exactly once");
  const [restWidth, restHeight] = sizes[0];
  check(
    restWidth * restHeight <= VOLUME_RESOLUTION_POLICY.maxPixels,
    `an attached volume scene respects its pixel budget (was ${restWidth}x${restHeight})`,
  );
  check(
    restWidth * restHeight < uncapped,
    "the backing store is smaller than devicePixelRatio alone would make it",
  );
  check(renders === 1, "sizing the backing store renders it once, not zero times");

  /*
   * A CAMERA GESTURE MUST NOT RESIZE THE BACKING STORE.
   *
   * This used to drop resolution for the duration of a drag. vtk.js reports
   * pointer positions in backing-store pixels and rotates by the difference
   * against a `previousPosition` it recorded earlier and keeps private, so
   * resizing the buffer mid-drag rescales the new position and not the stored
   * one. Measured on Retina: the buffer went 2876x899 -> 1438x449 on mouse-down
   * and the next 20 px move threw the camera from 42 degrees to 192.
   *
   * The gesture hooks are still handed to us, so the guard is that we ignore
   * them rather than that they are absent.
   */
  check(
    gestureStarts.length === 0 && gestureEnds.length === 0,
    "nothing subscribes to gestures in order to resize",
  );

  const sizesBeforeGesture = sizes.length;
  // Empty by the assertion above; driven anyway so that a future subscription
  // fails this test rather than silently reinstating the defect.
  for (const start of gestureStarts) start();
  flushFrames();
  await new Promise((resolve) => setTimeout(resolve, 260));
  flushFrames();
  check(
    sizes.length === sizesBeforeGesture,
    "a camera gesture leaves the backing store — and so the pointer scale — alone",
  );

  for (const end of gestureEnds) end();
  flushFrames();
  const rendersBeforePreview = renders;
  setVtkResolutionScale(renderWindow, 0.5);
  check(sizes.at(-1)![0] < restWidth && sizes.at(-1)![1] < restHeight, "explicit playback preview lowers the bounded backing store");
  check(renders === rendersBeforePreview, "preview sizing leaves the coherent draw to its owner");
  setVtkResolutionScale(renderWindow, 1);
  check(sizes.at(-1)![0] === restWidth && sizes.at(-1)![1] === restHeight, "saved-frame resolution is restored exactly");
  detach();
  check(unsubscribes === 0, "no gesture subscription is taken, so none needs releasing");

  // A genuine container resize must still work; it is the reason this exists.
  const resizeSizes: Array<[number, number]> = [];
  let resizeWidth = CSS_WIDTH;
  const resizeWindow = {
    ...renderWindow,
    getApiSpecificRenderWindow: () => ({
      setSize: (width: number, height: number) => { resizeSizes.push([width, height]); },
    }),
  };
  const detachResizable = attachVtkResize(
    { getBoundingClientRect: () => ({ width: resizeWidth, height: CSS_HEIGHT }) },
    resizeWindow,
    { policy: VOLUME_RESOLUTION_POLICY },
  );
  flushFrames();
  const afterAttach = resizeSizes.length;
  resizeWidth = CSS_WIDTH / 2;
  lastResizeObserver([{ contentRect: { width: resizeWidth, height: CSS_HEIGHT } }]);
  flushFrames();
  check(
    resizeSizes.length === afterAttach + 1,
    "a container that actually changes size still re-sizes the backing store",
  );
  detachResizable();

  const observedSizes: Array<[number, number]> = [];
  let panelHeight = 570;
  const detachObserved = attachVtkResize(
    { getBoundingClientRect: () => ({ width: 958, height: panelHeight }) },
    {
      ...renderWindow,
      getApiSpecificRenderWindow: () => ({
        setSize: (width: number, height: number) => { observedSizes.push([width, height]); },
      }),
    },
    { policy: VOLUME_RESOLUTION_POLICY },
  );
  panelHeight = 428.5;
  flushFrames();
  panelHeight = 570;
  lastResizeObserver([{ contentRect: { width: 958, height: panelHeight } }]);
  flushFrames();
  check(observedSizes.at(-1)?.[0] === 1916 && observedSizes.at(-1)?.[1] === 1140,
    "a layout change between observation and RAF cannot strand the old backing size after restoring the panel");
  panelHeight = 428.5;
  lastResizeObserver([{ contentRect: { width: 958, height: panelHeight } }]);
  flushFrames();
  check(observedSizes.at(-1)?.[1] === 857,
    "the observed fractional CSS height is preserved when allocating device pixels");
  detachObserved();

  // A rasterized scene pays nothing for pixels and must keep every one of them.
  const surfaceSizes: Array<[number, number]> = [];
  const surfaceWindow = {
    ...renderWindow,
    getApiSpecificRenderWindow: () => ({
      setSize: (width: number, height: number) => { surfaceSizes.push([width, height]); },
    }),
  };
  const detachSurface = attachVtkResize(container, surfaceWindow, {
    policy: SURFACE_RESOLUTION_POLICY,
  });
  flushFrames();
  check(
    surfaceSizes[0][0] === CSS_WIDTH * 2 && surfaceSizes[0][1] === CSS_HEIGHT * 2,
    "a rasterized scene keeps the full Retina backing store",
  );
  detachSurface();

  browser.requestAnimationFrame = saved.requestAnimationFrame;
  browser.cancelAnimationFrame = saved.cancelAnimationFrame;
  browser.devicePixelRatio = saved.devicePixelRatio;
  browser.matchMedia = saved.matchMedia;
  browser.ResizeObserver = saved.ResizeObserver;
}

/* Every 3D scene must route its backing store through the shared policy —
 * a single direct GenericRenderWindow.resize() reintroduces the uncapped
 * DPR multiply for that scene alone, which is exactly how this started. */
for (const file of [
  "VolumeRender3D.tsx",
  "SupercellRender3D.tsx",
  "Isosurface3D.tsx",
  "Streamlines3D.tsx",
  "Trajectories3D.tsx",
]) {
  const source = readFileSync(new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url), "utf8");
  check(source.includes("RESOLUTION_POLICY"), `${file} declares a backing-store pixel budget`);
}

/*
 * Continuous camera drag.
 *
 * vtk.js delivers the first pointer move of a burst as StartMouseMove and only
 * the rest as MouseMove, re-arming that split after every 200 ms gap. The
 * trackball style implements handleMouseMove alone, so once a frame takes over
 * 200 ms — routine for a ray-marched volume — every move arrives as a dropped
 * StartMouseMove and the camera stops responding mid-drag.
 */
{
  /*
   * Stand-in for vtkInteractorStyleTrackballCamera behind a vtk interactor.
   *
   * FROZEN, because every vtk.js instance is: `newInstance` returns
   * `Object.freeze(publicAPI)`. The first version of this fix installed a
   * `handleStartMouseMove` method on the style, which throws "object is not
   * extensible" under a strict-mode ES module and took the whole plot down
   * behind its error boundary. A mock that allowed the assignment passed while
   * the application was broken, so the mock is frozen too.
   */
  function fakeInteractor() {
    const moves: string[] = [];
    const listeners: Array<(callData: unknown) => void> = [];
    const style = Object.freeze({
      handleMouseMove(callData: unknown) { moves.push(String(callData)); },
    });
    const interactor = Object.freeze({
      getInteractorStyle: () => style as never,
      onStartMouseMove: (callback: (callData: unknown) => void) => {
        listeners.push(callback);
        return Object.freeze({
          unsubscribe: () => {
            const at = listeners.indexOf(callback);
            if (at >= 0) listeners.splice(at, 1);
          },
        });
      },
    });
    /** What the interactor dispatches for the first move after a >200 ms stall. */
    const dispatchStartMouseMove = (callData: unknown) => {
      for (const listener of listeners.slice()) listener(callData);
    };
    return { interactor, moves, listeners, dispatchStartMouseMove };
  }

  const { interactor, moves, listeners, dispatchStartMouseMove } = fakeInteractor();

  dispatchStartMouseMove("dropped-before-the-fix");
  check(
    moves.length === 0,
    "baseline: a move delivered as StartMouseMove reaches nothing, which is the defect",
  );

  const detach = enableContinuousCameraDrag({ getInteractor: () => interactor });
  check(listeners.length === 1, "the fix subscribes without mutating the frozen style");

  dispatchStartMouseMove("move-after-a-stall");
  check(
    moves.at(-1) === "move-after-a-stall",
    "a move delivered as StartMouseMove now drives the camera",
  );

  // Every move of a slow drag arrives this way, so all of them must land.
  for (let i = 0; i < 5; i++) dispatchStartMouseMove(`drag-${i}`);
  check(moves.length === 6, "every move of a slow drag reaches the camera, not just the first");

  detach();
  dispatchStartMouseMove("after-detach");
  check(
    listeners.length === 0 && moves.length === 6,
    "detaching unsubscribes, so a disposed scene stops driving a stale style",
  );
}

{
  // A later vtk.js handling the event itself would otherwise see every stalled
  // move twice — once from its own handler and once from ours.
  let subscribed = 0;
  const style = Object.freeze({
    handleMouseMove: () => {},
    handleStartMouseMove: () => {},
  });
  const detach = enableContinuousCameraDrag({
    getInteractor: () => Object.freeze({
      getInteractorStyle: () => style as never,
      onStartMouseMove: () => { subscribed++; return Object.freeze({ unsubscribe: () => {} }); },
    }),
  });
  detach();
  check(subscribed === 0, "a style that already handles StartMouseMove is left alone");
}

{
  // vtk.js 36 exposes the event; a build that does not must degrade to stock
  // behaviour rather than throw on mount.
  const detach = enableContinuousCameraDrag({
    getInteractor: () => Object.freeze({
      getInteractorStyle: () => Object.freeze({ handleMouseMove: () => {} }) as never,
    }),
  });
  detach();
  check(true, "an interactor without the StartMouseMove event attaches without throwing");
}

{
  const detach = enableContinuousCameraDrag({
    getInteractor: () => Object.freeze({ getInteractorStyle: () => null }),
  });
  detach();
  check(true, "a scene with no interactor style attaches and detaches without throwing");
}

{
  /* Three facts about vtk.js carry this fix, and none of them is visible in a
   * mock: the event exists, the burst timer still re-arms, and instances are
   * frozen. If any changes upstream, this must be revisited rather than
   * silently keep passing. */
  const interactorSource = readFileSync(
    new URL("../../../../ui/node_modules/@kitware/vtk.js/Rendering/Core/RenderWindowInteractor.js", import.meta.url),
    "utf8",
  );
  check(
    interactorSource.includes('"StartMouseMove"'),
    "vtk.js still dispatches StartMouseMove, so the subscription still has an event",
  );
  check(
    /moveTimeoutID = setTimeout/.test(interactorSource),
    "vtk.js still re-arms the move burst on a timer, so the fix is still needed",
  );
  const macroSource = readFileSync(
    new URL("../../../../ui/node_modules/@kitware/vtk.js/macros.js", import.meta.url),
    "utf8",
  );
  check(
    macroSource.includes("Object.freeze(publicAPI)"),
    "vtk.js instances are still frozen, so the style must not be mutated",
  );
}

/* The fix is only in effect because every 3D scene routes through the shared
 * stability attachment; a scene that mounts its own interactor would silently
 * keep the frozen drag. */
for (const file of [
  "VolumeRender3D.tsx",
  "SupercellRender3D.tsx",
  "Isosurface3D.tsx",
  "Streamlines3D.tsx",
  "Trajectories3D.tsx",
]) {
  const source = readFileSync(new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url), "utf8");
  check(
    source.includes("attachVtkCameraStability"),
    `${file} routes its camera through the shared drag/stability attachment`,
  );
}

{
  const commonSource = readFileSync(
    new URL("../../../../ui/src/components/plots/common.tsx", import.meta.url),
    "utf8",
  );
  check(
    commonSource.includes('"aria-labelledby": labelId'),
    "ControlGroup gives otherwise unnamed native controls the visible group label",
  );
  check(
    commonSource.includes('role="group" aria-labelledby={labelId}'),
    "ControlGroup exposes its visible label as the accessible group name",
  );
  check(
    !commonSource.includes('<label className="control-group__label">'),
    "ControlGroup no longer emits an unassociated label element",
  );
}

console.log(`\nvtkInteraction tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
