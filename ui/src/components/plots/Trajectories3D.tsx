import { useEffect, useRef, useState } from "react";
import { readTrajectories } from "@/lib/run/runOutputs";
import type { NpyN } from "@/lib/platform/npy";
import { disposeVtkScene } from "@/lib/vtk/vtkLifecycle";
import { VTK_SCALAR_SLOT } from "@/lib/vtk/vtkArrayOwnership";
import { useVtkGpuResourceOwnership } from "@/lib/vtk/useVtkGpuResourceOwnership";
import {
  TrajectoryGeometryService,
  type RetainedTrajectoryGeometry,
} from "@/lib/vtk/trajectoryGeometryClient";
import {
  trajectoryGeometryUpperBounds,
  type TrajectoryTubeGeometry,
} from "@/lib/vtk/trajectoryGeometryProtocol";
import { SURFACE_RESOLUTION_POLICY } from "@/lib/vtk/vtkResolution";
import { attachVtkCameraStability, attachVtkResize } from "@/lib/vtk/vtkInteraction";
import {
  registerVtkAppearanceCapture,
  unregisterVtkCapture,
} from "@/lib/vtk/vtkCaptureRegistry";
import type { VtkAppearanceDescriptor } from "@/lib/vtk/vtkAppearanceEvidence";
import { Icon } from "@/components/shell/Icon";
import { Plot3DAxisLabels, Plot3DScaleBadge } from "./Plot3DOverlay";
import { Plot3DCompass } from "./Plot3DCompass";
import { useCameraOrientation } from "@/lib/vtk/useCameraOrientation";
import { useCameraSync } from "@/lib/vtk/useCameraSync";
import { usePaneContext } from "./paneContext";
import { RootHistoryScopeNotice } from "./PlotSourceDisclosure";
import {
  ControlGroup,
  PillGroup,
  refitVtkCameraToBounds,
  resetVtkCameraToStormView,
  useVtkContextLoss,
  useFieldResourceOwnership,
  useSceneScale,
  SceneScaleControl,
  type PlotComponentProps,
} from "./common";
import {
  CameraElevationControls,
  useCameraElevation,
  useCameraElevationPlacement,
} from "./CameraElevationControls";
import { SceneAppearanceControls, useSceneAppearance } from "./SceneAppearanceControls";
import {
  exposeVtkBackendEvidence,
  useVtkBackend,
  vtkRenderWindowOptions,
} from "@/lib/vtk/vtkBackend";
import { Grouped3DControls } from "./Grouped3DControls";
import { useThreeDControlScope } from "@/lib/plot/threeDControlScope";
import {
  applySurfaceLighting,
  createVtkSceneLighting,
  type VtkSceneLighting,
} from "@/lib/vtk/sceneLighting";
import { useDomainBox } from "./domainBox";
import { useDomainAxes } from "./useDomainAxes";
import "./common.css";
import "./VolumeRender3D.css";

/* Lagrangian parcel trajectories (from --trajectories run): one tube per parcel, coloured by z/t/w/ζ.
 * Precomputed online every model step — unlike Streamlines3D (instantaneous wind). */

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import { applyImmutablePolyDataGeometry } from "@/lib/vtk/vtkImmutablePolyData";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkColorMaps from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps";
import { ScalarMode } from "@kitware/vtk.js/Rendering/Core/Mapper/Constants";

type ColorBy = "z" | "t" | "w" | "zeta";
const COLOR_OPTIONS: ReadonlyArray<{ value: ColorBy; label: string }> = [
  { value: "z", label: "Height" },
  { value: "t", label: "Time" },
  { value: "w", label: "Updraft" },
  { value: "zeta", label: "Vorticity" },
];
// First six columns in the trajectory record [t, x, y, z, w, zeta, ...].
const COLOR_COL: Record<ColorBy, number> = { z: 3, t: 0, w: 4, zeta: 5 };
const COLOR_UNIT: Record<ColorBy, string> = {
  z: "m",
  t: "s",
  w: "m/s",
  zeta: "s⁻¹",
};
const COLOR_PRESET: Record<ColorBy, string> = {
  z: "Viridis (matplotlib)",
  t: "Viridis (matplotlib)",
  w: "Cool to Warm",
  zeta: "Cool to Warm",
};
// Signed variables get a symmetric, zero-centred range (diverging palette).
const COLOR_SYMMETRIC: Record<ColorBy, boolean> = { z: false, t: false, w: true, zeta: true };

function applyTrajectoryGeometry(
  target: ReturnType<typeof vtkPolyData.newInstance>,
  geometry: TrajectoryTubeGeometry | null,
): void {
  applyImmutablePolyDataGeometry(target, geometry);
}

export function Trajectories3D({
  runDir,
  steps,
  stepIdx,
  renderFidelityMode = "scientific",
}: PlotComponentProps) {
  const controlScope = useThreeDControlScope();
  const sceneAppearance = useSceneAppearance(controlScope.scope, "trajectories");
  const vtkBackend = useVtkBackend(controlScope.scope, "trajectories");
  const sceneAppearanceRef = useRef(sceneAppearance.resolved);
  sceneAppearanceRef.current = sceneAppearance.resolved;
  // Grid is only for the scale badge; trajectories are in physical metres independent of frame.
  const grid = steps[0]?.grid ?? null;
  const publicationGeneration = steps[0]?.publicationGeneration ?? null;
  const [colorBy, setColorBy] = useState<ColorBy>("z");
  const [tubeRadius, setTubeRadius] = useState(120);
  const [sceneScale, setSceneScale] = useSceneScale(grid, undefined, controlScope.scope, "trajectories");
  const [resetTick, setResetTick] = useState(0);
  const ctxLoss = useVtkContextLoss();
  const {
    generation: contextGeneration,
    markFrameRendered,
  } = ctxLoss;
  const vtkLifecycleGeneration = contextGeneration * 2
    + (vtkBackend.backend === "webgpu" ? 1 : 0);
  const attachContextLossRef = useRef(ctxLoss.attach);
  attachContextLossRef.current = ctxLoss.attach;

  const [traj, setTraj] = useState<NpyN | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "absent" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);
  const [geometryError, setGeometryError] = useState<string | null>(null);
  const [preparedGeometry, setPreparedGeometry] = useState<{
    generation: string;
    result: RetainedTrajectoryGeometry;
  } | null>(null);
  const [parcelCount, setParcelCount] = useState(0);
  const [colorRange, setColorRange] = useState<[number, number] | null>(null);
  const [historyTimeRangeS, setHistoryTimeRangeS] = useState<[number, number] | null>(null);
  const geometryGeneration = JSON.stringify([
    runDir, publicationGeneration, colorBy, tubeRadius,
    sceneScale.xy, sceneScale.z,
  ]);
  const gpuGeneration = JSON.stringify([geometryGeneration, vtkLifecycleGeneration]);
  const trajectoryParcelCount = Math.max(1, traj?.shape[0] ?? 1);
  const geometryBounds = traj
    ? trajectoryGeometryUpperBounds(traj.shape, trajectoryParcelCount)
    : null;
  const maximumGeometryBytes = geometryBounds
    ? Math.max(1, geometryBounds.retainedBytes) : 0;
  const maximumGeometryScratchBytes = geometryBounds
    ? Math.max(1, geometryBounds.scratchBytes) : 0;
  const [measuredGpu, setMeasuredGpu] = useState<{ generation: string; bytes: number } | null>(null);
  const gpuUploadBytes = measuredGpu?.generation === gpuGeneration
    ? measuredGpu.bytes : maximumGeometryBytes;
  const gpuUpload = useVtkGpuResourceOwnership({
    enabled: status === "ready" && traj != null && gpuUploadBytes > 0,
    owner: "trajectories",
    generation: gpuGeneration,
    bytes: gpuUploadBytes,
  });
  const appearanceEvidenceRef = useRef<VtkAppearanceDescriptor>({
    plot: "trajectories",
    backend: vtkBackend.backend,
  });
  appearanceEvidenceRef.current = {
    plot: "trajectories",
    backend: vtkBackend.backend,
    datasetId: runDir,
    frameIndex: null,
    field: colorBy,
    fidelityMode: renderFidelityMode,
    parameters: {
      tubeRadius,
      parcelCount,
      colorRange: colorRange ?? null,
      temporalScope: "whole-recorded-history",
      historyTimeRangeS,
      lighting: sceneAppearance.resolved.lighting,
      lightingControls: sceneAppearance.resolved.lightingOptions,
      background: sceneAppearance.resolved.backgroundHex,
      logicalGpuUploadBytes: gpuUpload.admitted ? gpuUploadBytes : 0,
    },
  };
  useFieldResourceOwnership(
    `${controlScope.scope}:trajectories-source`,
    traj,
    `${publicationGeneration ?? "none"}:${runDir}`,
  );
  const geometryServiceRef = useRef<TrajectoryGeometryService>();
  if (!geometryServiceRef.current) geometryServiceRef.current = new TrajectoryGeometryService();
  const currentGeometryRef = useRef<RetainedTrajectoryGeometry | null>(null);
  const pendingGeometryRef = useRef<RetainedTrajectoryGeometry | null>(null);

  // ---------- Load the run's trajectory file ----------
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setTraj(null);
    setError(null);
    readTrajectories(runDir, publicationGeneration)
      .then((t) => {
        if (cancelled) return;
        if (!t) {
          setStatus("absent");
          return;
        }
        setTraj(t);
        setStatus("ready");
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [runDir, publicationGeneration]);

  // ---------- vtk.js scene ----------
  const containerRef = useRef<HTMLDivElement | null>(null);
  type SceneRefs = {
    renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    data: ReturnType<typeof vtkPolyData.newInstance>;
    mapper: ReturnType<typeof vtkMapper.newInstance>;
    actor: ReturnType<typeof vtkActor.newInstance>;
    ctf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    lighting: VtkSceneLighting;
  };
  const sceneRef = useRef<SceneRefs | null>(null);
  const cameraInitializedRef = useRef(false);
  // Domain bounding box: keeps camera framing stable as storms advect.
  useDomainBox(
    () => sceneRef.current?.renderWindow ?? null,
    grid,
    sceneScale,
    vtkLifecycleGeneration,
    sceneAppearance.resolved.reference,
  );
  const axisLabels = useDomainAxes(
    () => sceneRef.current?.renderWindow ?? null,
    containerRef,
    grid,
    sceneScale,
    vtkLifecycleGeneration,
    sceneAppearance.resolved.reference,
  );
  /* Compass: viewer direction relative to north. No storm track available for trajectories. */
  const cameraOrientation = useCameraOrientation(
    () => sceneRef.current?.renderWindow ?? null,
    sceneScale,
    undefined,
    vtkLifecycleGeneration,
  );

  /* Pane camera lock: orbiting one pane orbits all, keeping side-by-side comparisons aligned. */
  const pane = usePaneContext();
  useCameraSync(
    () => sceneRef.current?.renderWindow ?? null,
    pane.bus,
    {
      paneId: pane.paneId,
      enabled: pane.cameraLocked,
      scale: sceneScale,
      authority: pane.cameraAuthority,
      isReady: () => cameraInitializedRef.current,
      lifecycleGeneration: vtkLifecycleGeneration,
      onCameraApplied: () => { cameraInitializedRef.current = true; },
    },
  );
  const cameraElevation = useCameraElevation(grid);
  useCameraElevationPlacement(() => sceneRef.current?.renderWindow ?? null, {
    grid,
    scale: sceneScale,
    selection: cameraElevation,
    lifecycleGeneration: vtkLifecycleGeneration,
    onCameraApplied: () => { cameraInitializedRef.current = true; },
  });

  useEffect(() => {
    cameraInitializedRef.current = false;
  }, [runDir, publicationGeneration]);

  // ---------- Mount / unmount ----------
  useEffect(() => {
    if (!ctxLoss.available) return;
    const container = containerRef.current;
    if (!container) return;

    const renderWindow = vtkGenericRenderWindow.newInstance(vtkRenderWindowOptions(
      [...sceneAppearanceRef.current.background] as [number, number, number],
      vtkBackend.backend,
    ));
    cameraInitializedRef.current = false;
    renderWindow.setContainer(container);
    const clearBackendEvidence = exposeVtkBackendEvidence(
      container,
      renderWindow,
      vtkBackend.backend,
      vtkBackend.confirmActiveApi,
    );

    const data = vtkPolyData.newInstance();
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(data);
    const ctf = vtkColorTransferFunction.newInstance();
    mapper.setLookupTable(ctf);
    mapper.setScalarVisibility(true);
    mapper.setUseLookupTableScalarRange(true);
    // Use named array: vtkTubeFilter doesn't mark output scalars active, so default mode finds nothing.
    mapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    mapper.setColorByArrayName(VTK_SCALAR_SLOT);

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);

    const renderer = renderWindow.getRenderer();
    const lighting = createVtkSceneLighting(renderer);
    lighting.apply(
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );
    applySurfaceLighting(actor.getProperty(), sceneAppearanceRef.current.lightingOptions);
    renderer.addActor(actor);
    const scene: SceneRefs = {
      renderWindow, data, mapper, actor, ctf, lighting,
    };
    sceneRef.current = scene;

    registerVtkAppearanceCapture(container, renderWindow, () => appearanceEvidenceRef.current);

    const detachContextLoss = attachContextLossRef.current(renderWindow);
    const detachResize = attachVtkResize(container, renderWindow, {
      policy: SURFACE_RESOLUTION_POLICY,
    });
    const detachCameraStability = attachVtkCameraStability(
      renderWindow,
      renderWindow.getRenderer(),
      () => renderWindow.getRenderWindow().render(),
    );

    return () => {
      disposeVtkScene({
        clearScene: () => {
          if (sceneRef.current?.renderWindow === renderWindow) sceneRef.current = null;
        },
        cleanupSteps: [
          detachResize,
          clearBackendEvidence,
          () => lighting.dispose(),
          detachCameraStability,
          detachContextLoss,
          () => unregisterVtkCapture(container),
          () => renderWindow.getRenderer().removeActor(actor),
        ],
        resources: [actor, mapper, ctf, data],
        renderWindow,
      });
    };
  }, [ctxLoss.available, ctxLoss.generation, vtkBackend.backend, vtkBackend.confirmActiveApi]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const appearance = sceneAppearance.resolved;
    scene.renderWindow.getRenderer().setBackground(...appearance.background);
    scene.lighting.apply(appearance.lightingOptions, appearance.isLight);
    applySurfaceLighting(scene.actor.getProperty(), appearance.lightingOptions);
    scene.renderWindow.getRenderWindow().render();
  }, [sceneAppearance.resolved, vtkLifecycleGeneration]);

  // ---------- Worker-owned path scan, scaling, and tube expansion ----------
  useEffect(() => {
    if (!traj || status !== "ready") {
      pendingGeometryRef.current?.release();
      pendingGeometryRef.current = null;
      setPreparedGeometry(null);
      setGeometryLoading(false);
      return;
    }
    if (gpuUpload.error || !gpuUpload.admitted) return;
    const controller = new AbortController();
    setGeometryLoading(true);
    setGeometryError(null);
    geometryServiceRef.current!.extract(traj, {
      colorColumn: COLOR_COL[colorBy],
      tubeRadius,
      scale: { xy: sceneScale.xy, z: sceneScale.z },
      maximumParcels: trajectoryParcelCount,
    }, maximumGeometryBytes, maximumGeometryScratchBytes, controller.signal).then((result) => {
      if (controller.signal.aborted) {
        result.release();
        return;
      }
      if (pendingGeometryRef.current
          && pendingGeometryRef.current !== currentGeometryRef.current) {
        pendingGeometryRef.current.release();
      }
      pendingGeometryRef.current = result;
      setPreparedGeometry({ generation: geometryGeneration, result });
      setGeometryLoading(false);
    }).catch((reason: unknown) => {
      if (controller.signal.aborted) return;
      setGeometryLoading(false);
      setGeometryError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => controller.abort();
  }, [
    traj,
    status,
    colorBy,
    tubeRadius,
    sceneScale.xy,
    sceneScale.z,
    geometryGeneration,
    maximumGeometryBytes,
    maximumGeometryScratchBytes,
    trajectoryParcelCount,
    gpuUpload.admitted,
    gpuUpload.error,
  ]);

  // ---------- Coherent main-thread commit after exact GPU admission ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const prepared = preparedGeometry;
    if (!prepared || prepared.generation !== geometryGeneration) {
      if (!traj || status !== "ready") {
        applyTrajectoryGeometry(scene.data, null);
        scene.actor.setVisibility(false);
        currentGeometryRef.current?.release();
        currentGeometryRef.current = null;
        pendingGeometryRef.current?.release();
        pendingGeometryRef.current = null;
        setParcelCount(0);
        setColorRange(null);
        scene.renderWindow.getRenderWindow().render();
      }
      return;
    }
    if (!prepared.result.geometry) {
      setHistoryTimeRangeS(prepared.result.timeRangeS ?? null);
      applyTrajectoryGeometry(scene.data, null);
      scene.actor.setVisibility(false);
      currentGeometryRef.current?.release();
      currentGeometryRef.current = prepared.result;
      pendingGeometryRef.current = null;
      setParcelCount(0);
      setColorRange(null);
      scene.renderWindow.getRenderWindow().render();
      return;
    }
    const exactGpuBytes = prepared.result.retainedBytes;
    if (measuredGpu?.generation !== gpuGeneration || measuredGpu.bytes !== exactGpuBytes) {
      setMeasuredGpu({ generation: gpuGeneration, bytes: exactGpuBytes });
      return;
    }
    if (gpuUpload.error || !gpuUpload.admitted) return;
    setHistoryTimeRangeS(prepared.result.timeRangeS ?? null);
    applyTrajectoryGeometry(scene.data, prepared.result.geometry);
    const previous = currentGeometryRef.current;
    currentGeometryRef.current = prepared.result;
    if (pendingGeometryRef.current === prepared.result) pendingGeometryRef.current = null;
    if (previous && previous !== prepared.result) previous.release();
    scene.actor.setVisibility(true);
    setParcelCount(prepared.result.parcels);
    let lo = prepared.result.min!;
    let hi = prepared.result.max!;
    if (COLOR_SYMMETRIC[colorBy]) {
      const extent = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
      lo = -extent;
      hi = extent;
    }
    if (!(hi > lo)) hi = lo + 1;
    setColorRange([lo, hi]);
    const presetData = vtkColorMaps.getPresetByName(COLOR_PRESET[colorBy]);
    if (presetData) scene.ctf.applyColorMap(presetData);
    scene.ctf.setMappingRange(lo, hi);
    scene.ctf.modified();
    if (!cameraInitializedRef.current) {
      resetVtkCameraToStormView(scene.renderWindow.getRenderer(), scene.actor.getBounds());
      cameraInitializedRef.current = true;
    } else {
      refitVtkCameraToBounds(scene.renderWindow.getRenderer(), scene.actor.getBounds());
    }
    scene.renderWindow.getRenderWindow().render();
    markFrameRendered(contextGeneration);
  }, [
    colorBy,
    contextGeneration,
    geometryGeneration,
    gpuGeneration,
    gpuUpload.admitted,
    gpuUpload.error,
    markFrameRendered,
    measuredGpu,
    preparedGeometry,
    status,
    traj,
    vtkLifecycleGeneration,
  ]);

  useEffect(() => () => {
    geometryServiceRef.current?.shutdown();
    pendingGeometryRef.current?.release();
    if (currentGeometryRef.current !== pendingGeometryRef.current) {
      currentGeometryRef.current?.release();
    }
    pendingGeometryRef.current = null;
    currentGeometryRef.current = null;
  }, []);

  // ---------- Reset camera ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    resetVtkCameraToStormView(scene.renderWindow.getRenderer(), scene.actor.getBounds());
    scene.renderWindow.getRenderWindow().render();
  }, [resetTick]);

  return (
    <div className="volume-render">
      <RootHistoryScopeNotice
        runDir={runDir}
        step={steps[stepIdx] ?? steps[0] ?? null}
        label="Root-owned Earth-coordinate trajectories"
      />
      <Grouped3DControls
        ariaLabel="3D path controls"
        scope={controlScope}
        groups={[
          {
            id: "paths",
            label: "Paths",
            status: status === "ready"
              ? geometryLoading ? "Preparing" : `${parcelCount} parcels`
              : status,
            controls: <>
        <ControlGroup label="Colour by">
          <PillGroup options={COLOR_OPTIONS} value={colorBy} onChange={setColorBy} />
        </ControlGroup>
        <ControlGroup label="Tube radius">
          <input
            type="range"
            min={30}
            max={500}
            step={10}
            value={tubeRadius}
            onChange={(e) => setTubeRadius(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
          />
          <span className="volume-render__slider-value">{tubeRadius} m</span>
        </ControlGroup>
            </>,
          },
          {
            id: "scene",
            label: "Scene",
            status: sceneAppearance.selection.backgroundMode === "theme"
              ? "Theme"
              : sceneAppearance.selection.backgroundMode,
            controls: <>
        <SceneAppearanceControls state={sceneAppearance} backend={vtkBackend} />
        <SceneScaleControl value={sceneScale} onChange={setSceneScale} />
        {(!pane.cameraLocked || pane.cameraAuthority) && (
          <>
            <CameraElevationControls selection={cameraElevation} />
            <button
              type="button"
              className="btn btn--secondary volume-render__reset"
              onClick={() => setResetTick((t) => t + 1)}
            >
              <Icon name="settings" size={12} />
              <span>Reset camera</span>
            </button>
          </>
        )}
            </>,
          },
        ]}
      />

      <div className="volume-render__canvas-wrap plot3d-overlay-host" style={sceneAppearance.hostStyle}
        data-logical-gpu-upload-bytes={gpuUpload.admitted ? gpuUploadBytes : 0}>
        <div
          ref={containerRef}
          className="volume-render__canvas"
          role="img"
          aria-label="Interactive three-dimensional parcel trajectories; drag to orbit, use camera controls for keyboard adjustment"
        />
        <Plot3DAxisLabels labels={axisLabels} />
        <Plot3DScaleBadge
          grid={grid}
          zExag={sceneScale.z}
          fidelityMode={renderFidelityMode}
        />
        <Plot3DCompass orientation={cameraOrientation} />
        {ctxLoss.lost && (
          <div className="volume-render__overlay volume-render__overlay--error">
            3D graphics context lost
            <div className="volume-render__overlay-hint">
              Rebuilding after the GPU dropped WebGL. This clears after the
              trajectory frame renders successfully.
            </div>
          </div>
        )}
        {status === "loading" && (
          <div className="volume-render__overlay">Loading trajectories…</div>
        )}
        {status === "ready" && geometryLoading && currentGeometryRef.current == null && (
          <div className="volume-render__overlay">Preparing trajectory geometry…</div>
        )}
        {status === "error" && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Trajectory load failed: {error}
          </div>
        )}
        {gpuUpload.error && status !== "error" && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Trajectory GPU admission failed: {gpuUpload.error}
          </div>
        )}
        {geometryError && status === "ready" && !gpuUpload.error && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Trajectory geometry preparation failed: {geometryError}
          </div>
        )}
        {status === "absent" && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            No trajectory data for this run.
            <div className="volume-render__overlay-hint">
              Re-run the simulator with <code>--trajectories</code> to write
              parcel paths.
            </div>
          </div>
        )}
        {status === "ready" && parcelCount > 0 && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            {parcelCount} parcel paths · whole recorded history{historyTimeRangeS
              ? ` · ${historyTimeRangeS[0]}–${historyTimeRangeS[1]} s` : ""} · colour = {COLOR_OPTIONS.find((o) => o.value === colorBy)?.label}
            {colorRange
              ? ` · ${colorRange[0].toPrecision(4)} to ${colorRange[1].toPrecision(4)} ${COLOR_UNIT[colorBy]}`
              : ""}
          </div>
        )}
      </div>
    </div>
  );
}
