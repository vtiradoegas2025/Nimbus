import { streamlineMagnitudeMaximum } from "@/lib/vtk/streamlineDirection";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fieldDefinitionIdentity,
  readFieldBrickHierarchy,
  readFieldWindows,
  type FieldLoadWindow,
} from "@/lib/run/runOutputs";
import { beginFieldLoad, endFieldLoad } from "@/lib/run/frameReady";
import {
  cancelInflightPrefetches,
  createPrefetchController,
  endPrefetch,
  frameIdentitiesToPrefetch,
  releasePrefetchController,
  tryBeginPrefetch,
} from "@/lib/run/framePrefetch";
import { describeFrameTransition, findPublishedFrame } from "@/lib/run/frameIdentity";
import type { Npy3D } from "@/lib/platform/npy";
import { Plot3DAxisLabels, Plot3DColorbar, Plot3DOverlayStack, Plot3DScaleBadge } from "./Plot3DOverlay";
import { Plot3DCompass } from "./Plot3DCompass";
import { unitFor } from "@/lib/science/fieldUnits";
import { useCameraOrientation } from "@/lib/vtk/useCameraOrientation";
import { useCameraSync } from "@/lib/vtk/useCameraSync";
import { usePaneContext } from "./paneContext";
import { fieldBackingStores } from "@/lib/core/fieldCache";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import { MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES } from "@/lib/dataset/nativeFieldWindow";
import { stormTrackCentre, useStormTrack } from "./useStormTrack";
import { scaleVoxelBudget } from "@/lib/plot/paneLayout";
import {
  planInteractiveVolumeWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { useRadarFloorLayer } from "./radarFloorLayer";
import { buildVortexRingSeeds } from "@/lib/plot/vortexSeeding";
import { RadarFloorControls, useRadarFloorSelection } from "./RadarFloorControls";
import {
  CameraElevationControls,
  useCameraElevation,
  useCameraElevationPlacement,
} from "./CameraElevationControls";
import {
  CARTESIAN_VOXEL_CAP_STREAMLINES,
} from "@/lib/science/cartesianInterp";
import { releaseVtkPointDataArrays, VTK_SCALAR_SLOT } from "@/lib/vtk/vtkArrayOwnership";
import { usePreparedVectorImage } from "@/lib/vtk/useVtkPreparation";
import { disposeVtkScene } from "@/lib/vtk/vtkLifecycle";
import { useStreamlineExtraction } from "@/lib/vtk/useStreamlineExtraction";
import type { StreamlineGeometry } from "@/lib/vtk/streamlineExtractionProtocol";
import { lazyAppearanceEvidence } from "@/lib/vtk/lazyAppearanceEvidence";
import { useVtkGpuResourceOwnership } from "@/lib/vtk/useVtkGpuResourceOwnership";
import {
  streamlineGeometryFingerprint,
  streamlineSeedFingerprint,
} from "@/lib/vtk/streamlineGeometryParity";
import { carriedMagnitude, latchMagnitude, type LatchedMagnitude } from "@/lib/plot/viewSync";
import { SURFACE_RESOLUTION_POLICY } from "@/lib/vtk/vtkResolution";
import { attachVtkCameraStability, attachVtkResize } from "@/lib/vtk/vtkInteraction";
import {
  registerVtkAppearanceCapture,
  unregisterVtkCapture,
} from "@/lib/vtk/vtkCaptureRegistry";
import type { VtkAppearanceDescriptor } from "@/lib/vtk/vtkAppearanceEvidence";
import { Icon } from "@/components/shell/Icon";
import {
  ControlGroup,
  PillGroup,
  refitVtkCameraToBounds,
  resetVtkCameraToStormView,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useSceneScale,
  useVtkContextLoss,
  useVtkRunReset,
  SceneScaleControl,
  type PlotComponentProps,
} from "./common";
import { SceneAppearanceControls, useSceneAppearance } from "./SceneAppearanceControls";
import {
  exposeVtkBackendEvidence,
  useVtkBackend,
  vtkRenderWindowOptions,
} from "@/lib/vtk/vtkBackend";
import { Grouped3DControls } from "./Grouped3DControls";
import { useThreeDControlScope } from "@/lib/plot/threeDControlScope";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsSharePlotRank } from "@/lib/plot/fieldPlotEligibility";
import {
  contrastAdjustedColor,
  hexToSceneRgb,
  sceneRgbToHex,
} from "@/lib/plot/sceneAppearance";
import {
  applySurfaceLighting,
  createVtkSceneLighting,
  type VtkSceneLighting,
} from "@/lib/vtk/sceneLighting";
import { useDomainBox } from "./domainBox";
import { useDomainAxes } from "./useDomainAxes";
import { usePanePresentationBarrier } from "./panePresentationContext";
import { defaultStreamlineTubeRadiusM } from "@/lib/plot/streamlinePresentation";
import { streamlineBoxSeedDimensions } from "@/lib/plot/streamlineSeedDomain";
import { detectSeedBoundaryContacts } from "@/lib/vtk/streamlineBoundary";
import {
  expandVectorBrickNeighborhood,
  planVectorBrickNeighborhood,
  selectVectorBrickFeatureFocus,
  type VectorBrickExpansion,
  type VectorBrickNeighborhoodPlan,
} from "@/lib/plot/fieldBrickHierarchy";
import "./common.css";
import "./VolumeRender3D.css";

/* VTK.js RK2 streamlines through (u,v,w) or 3D vorticity, rendered as tubes.
 * Seeds center on the strongest third-component column (updraft or vortex core). */

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import { applyImmutablePolyDataGeometry } from "@/lib/vtk/vtkImmutablePolyData";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkColorMaps from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps";
import { ScalarMode } from "@kitware/vtk.js/Rendering/Core/Mapper/Constants";

interface VectorMode {
  id: "velocity" | "vorticity";
  /** Pill label. */
  short: string;
  /** Descriptive label for the missing-field overlay. */
  label: string;
  /** The three component field names (x, y, z components). The vorticity
   *  slots keep the legacy vorticity_r/theta/z manifest names pending the
   *  deferred grid-metadata field rename. */
  components: readonly [string, string, string];
  /** Tube colour [r, g, b] in 0..1. */
  color: [number, number, number];
}

/* Velocity (default) and vorticity share the same Cartesian packing and storm-centered seeding. */
const VECTOR_MODES: ReadonlyArray<VectorMode> = [
  {
    id: "velocity",
    short: "Velocity",
    label: "Velocity streamlines",
    components: ["u", "v", "w"],
    color: [0.36, 0.65, 1.0], // light blue
  },
  {
    id: "vorticity",
    short: "Vorticity",
    label: "Vortex lines (3D vorticity)",
    components: ["vorticity_r", "vorticity_theta", "vorticity_z"],
    color: [1.0, 0.6, 0.25], // amber
  },
];

/** `vortex` seeds a ring at the near-surface rotation maximum radius (produces the Orf arching-line look). */
type SeedMode = "box" | "feature" | "vortex";
type StreamlineColorMode = "magnitude" | "uniform";
const STREAMLINE_COLOR_OPTIONS: ReadonlyArray<{ value: StreamlineColorMode; label: string }> = [
  { value: "magnitude", label: "Magnitude" },
  { value: "uniform", label: "Uniform" },
];
let streamlineVectorResourceSequence = 0;
const INITIAL_VECTOR_BRICK_EXPANSION: VectorBrickExpansion = {
  lower: [0, 0, 0],
  upper: [0, 0, 0],
};

export function Streamlines3D({
  runDir,
  steps,
  stepIdx,
  renderFidelityMode = "scientific",
  onFrameDisplayChange,
}: PlotComponentProps) {
  const controlScope = useThreeDControlScope();
  const sceneAppearance = useSceneAppearance(controlScope.scope, "streamlines");
  const vtkBackend = useVtkBackend(controlScope.scope, "streamlines");
  const sceneAppearanceRef = useRef(sceneAppearance.resolved);
  sceneAppearanceRef.current = sceneAppearance.resolved;
  const rootCurrentStep = steps[stepIdx] ?? null;
  const [vectorModeId, setVectorModeId] = useState<VectorMode["id"]>("velocity");
  const activeMode = VECTOR_MODES.find((mode) => mode.id === vectorModeId) ?? VECTOR_MODES[0];
  const rootLoadFrame = useDebouncedFrameIdentity(rootCurrentStep, SCRUB_DEBOUNCE_MS);
  const rootPlanningTrackStep = findPublishedFrame(
    steps,
    rootLoadFrame.stepIndex,
    rootLoadFrame.publicationGeneration,
  );
  const stormTrack = useStormTrack(
    runDir,
    rootPlanningTrackStep?.index ?? null,
    rootPlanningTrackStep?.simulationTimeS ?? null,
    rootPlanningTrackStep?.publicationGeneration ?? null,
  );
  const stormCentre = stormTrackCentre(stormTrack);
  const outputHierarchySource = useOutputHierarchySource({
    runDir,
    steps,
    stepIdx,
    scope: controlScope.scope,
    requiredFields: activeMode.components,
    focusPointM: stormCentre ? [stormCentre[0], stormCentre[1]] : null,
  });
  const currentStep = outputHierarchySource.currentStep;
  const sourceRunDir = outputHierarchySource.sourceRunDir;
  const sourceSteps = outputHierarchySource.sourceSteps;
  const fieldsForStep = useMemo(() => currentStep?.fields ?? [], [currentStep?.fields]);
  const availableModes = VECTOR_MODES.filter((mode) =>
    fieldsSharePlotRank(sourceRunDir, currentStep, mode.components, [3]),
  );

  // Box mode uses NxNxN storm-centered seeds. Feature mode uses 4N ranked
  // column maxima from the worker so coherent structures are not over-seeded.
  // Vortex mode rings the near-surface rotation maximum — see `vortexSeeding`.
  const [seedCount, setSeedCount] = useState(4); // 4×4×4 = 64 seeds
  const [seedMode, setSeedMode] = useState<SeedMode>("box");
  const [maxSteps, setMaxSteps] = useState(300);
  // Tube radius in metres — the streamlines render as 3D tubes of this radius.
  const [tubeRadius, setTubeRadius] = useState(250);
  const [colorMode, setColorMode] = useState<StreamlineColorMode>("magnitude");
  const [uniformColors, setUniformColors] = useState<Record<VectorMode["id"], string>>(
    () => Object.fromEntries(VECTOR_MODES.map((mode) => [
      mode.id,
      sceneRgbToHex(mode.color),
    ])) as Record<VectorMode["id"], string>,
  );
  const uniformColor = uniformColors[activeMode.id];
  const [colorLatchEpoch, setColorLatchEpoch] = useState(0);
  const appearanceEvidenceRef = useRef<VtkAppearanceDescriptor>({
    plot: "streamlines",
    backend: vtkBackend.backend,
  });
  const committedAppearanceRef = useRef<VtkAppearanceDescriptor | null>(null);
  const committedGeometryEvidenceRef = useRef<typeof geometryEvidence | null>(null);
  appearanceEvidenceRef.current = {
    plot: "streamlines",
    backend: vtkBackend.backend,
    datasetId: sourceRunDir,
    frameIndex: currentStep?.index ?? null,
    field: vectorModeId,
    fidelityMode: renderFidelityMode,
    parameters: {
      seedCount,
      seedMode,
      maxSteps,
      tubeRadius,
      colorMode,
      uniformColor,
      lighting: sceneAppearance.resolved.lighting,
      lightingControls: sceneAppearance.resolved.lightingOptions,
      background: sceneAppearance.resolved.backgroundHex,
    },
  };
  const tubeRadiusGridRef = useRef<string | null>(null);
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
  const pane = usePaneContext();
  /* Split voxel budget across panes; four full-budget volumes exhausts VRAM in a 2x2 grid. */
  const paneVoxelCap = scaleVoxelBudget(CARTESIAN_VOXEL_CAP_STREAMLINES, pane.voxelBudgetScale);
  const brickedVectorEligible = Boolean(
    activeMode.components.every((field) => currentStep?.fieldBrickHierarchies?.[field]),
  );
  const [vectorBrickPlan, setVectorBrickPlan] = useState<VectorBrickNeighborhoodPlan | null>(null);
  const vectorBrickIdentity = [
    sourceRunDir,
    currentStep?.publicationGeneration ?? "none",
    currentStep?.index ?? "none",
    activeMode.id,
    seedMode,
  ].join(":");
  const [vectorBrickRequest, setVectorBrickRequest] = useState<{
    identity: string;
    expansion: VectorBrickExpansion;
    iteration: number;
  } | null>(null);
  const vectorBrickExpansion = vectorBrickRequest?.identity === vectorBrickIdentity
    ? vectorBrickRequest.expansion
    : INITIAL_VECTOR_BRICK_EXPANSION;
  const vectorBrickIteration = vectorBrickRequest?.identity === vectorBrickIdentity
    ? vectorBrickRequest.iteration
    : 0;
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      representation: brickedVectorEligible ? "trajectory-native-bricks" : "bounded-monolithic",
      brickHierarchyLevel: vectorBrickPlan?.level ?? null,
      selectedBrickCount: vectorBrickPlan?.items.length ?? null,
      totalBrickCount: vectorBrickPlan?.totalBrickCount ?? null,
      brickExpansionIteration: vectorBrickIteration,
    },
  };
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const sourceDefinitionIdentity = JSON.stringify(activeMode.components.map(field =>
    fieldDefinitionIdentity(sourceRunDir, currentStep?.index ?? null, field, currentStep?.publicationGeneration ?? null)));
  const stormCenterX = stormCentre?.[0] ?? null;
  const stormCenterY = stormCentre?.[1] ?? null;
  const boundedWindowPlan = useMemo(
    () => planInteractiveVolumeWindows(
      currentStep?.grid,
      paneVoxelCap,
      stormCenterX != null && stormCenterY != null ? [stormCenterX, stormCenterY] : null,
    ),
    [currentStep?.grid, paneVoxelCap, stormCenterX, stormCenterY],
  );
  const [loadedVectors, setLoadedVectors] = useState<{
    vectors: [Npy3D, Npy3D, Npy3D];
    runDir: string;
    modeId: VectorMode["id"];
    stepIndex: number;
    publicationGeneration: string | null;
    definitionIdentity: string;
  } | null>(null);
  const loadedVectorFrame = loadedVectors?.modeId === vectorModeId && loadedVectors.runDir === sourceRunDir
      && loadedVectors.definitionIdentity === sourceDefinitionIdentity
    ? loadedVectors
    : null;
  const vectors = loadedVectorFrame?.vectors ?? null;
  const vectorResourceOwnerRef = useRef<string>();
  if (!vectorResourceOwnerRef.current) {
    vectorResourceOwnerRef.current = `streamline-source:${++streamlineVectorResourceSequence}`;
  }
  useEffect(() => {
    if (!vectors || !loadedVectorFrame) return;
    const resources = visualizationResourceGovernor.reserve({
      owner: vectorResourceOwnerRef.current!,
      generation: `${loadedVectorFrame.publicationGeneration ?? "none"}:${loadedVectorFrame.stepIndex}`,
      priority: "visible",
      claims: vectors.flatMap((field) => fieldBackingStores(field).map((buffer) =>
        visualizationBufferClaim(buffer, "decoded-field")
      )),
    });
    if (!resources.ok) return;
    return () => { visualizationResourceGovernor.release(resources.reservation.id); };
  }, [loadedVectorFrame, vectors]);
  const loadedStep = findPublishedFrame(
    sourceSteps,
    loadedVectorFrame?.stepIndex ?? null,
    loadedVectorFrame?.publicationGeneration ?? null,
  );
  const renderGrid = loadedStep?.grid ?? null;
  const [sceneScale, setSceneScale] = useSceneScale(renderGrid, undefined, controlScope.scope, "streamlines");
  // Ref so the per-field effect always reads the current scale without taking it as a dependency.
  const sceneScaleRef = useRef(sceneScale);
  sceneScaleRef.current = sceneScale;
  const preparation = usePreparedVectorImage(
    vectors,
    renderGrid,
    paneVoxelCap,
    ctxLoss.generation,
    {
      detectVerticalVortexCore: vectorModeId === "vorticity",
      geometryHandoff: true,
    },
  );
  const vectorBrickSeedAnchorRef = useRef<{
    identity: string | null;
    seeds: Float32Array;
  }>({ identity: null, seeds: new Float32Array() });

  const haveVectors = fieldsSharePlotRank(
    sourceRunDir,
    currentStep,
    activeMode.components,
    [3],
  );
  const [renderedStepIndex, setRenderedStepIndex] = useState<number | null>(null);
  const [renderedPublicationGeneration, setRenderedPublicationGeneration] =
    useState<string | null>(null);
  appearanceEvidenceRef.current.frameIndex = renderedStepIndex;
  appearanceEvidenceRef.current.parameters = { ...appearanceEvidenceRef.current.parameters,
    requestedFrameIndex: outputHierarchySource.requestedStep?.index ?? null, renderedPublicationGeneration,
  };
  const requestedStepIndex = outputHierarchySource.requestedStep?.index ?? null;
  const requestedPublicationGeneration = outputHierarchySource.requestedStep?.publicationGeneration ?? null;
  const frameTransition = describeFrameTransition(
    requestedStepIndex,
    renderedStepIndex,
    requestedPublicationGeneration,
    renderedPublicationGeneration,
  );

  // Fall back to an available vector field when the selected one isn't exported
  // by the loaded run (e.g. vorticity selected, then a velocity-only run loads).
  useEffect(() => {
    const avail = VECTOR_MODES.filter((mode) =>
      fieldsSharePlotRank(sourceRunDir, currentStep, mode.components, [3]),
    );
    if (avail.length > 0 && !avail.some((m) => m.id === vectorModeId)) {
      setVectorModeId(avail[0].id);
    }
  }, [currentStep, sourceRunDir, vectorModeId]);

  // Vorticity defaults to "feature" (ranked columns) not "vortex" (ring at core).
  useEffect(() => {
    setSeedMode(vectorModeId === "vorticity" ? "feature" : "box");
  }, [vectorModeId]);

  useEffect(() => {
    setRenderedStepIndex(null);
    setRenderedPublicationGeneration(null);
  }, [sourceRunDir, vectorModeId, sourceDefinitionIdentity]);

  // Near-surface rotation maximum for vortex seeding; restricted to lowest km to avoid mid-level meso.
  const vortexCore = useMemo(() => {
    const prepared = preparation.metadata;
    if (!prepared || vectorModeId !== "vorticity") return null;
    const core = prepared.vortexCore;
    if (!core) return null;
    // The detector works in grid-relative metres (index × spacing); seeds have
    // to be world-space, so carry the prepared lattice's origin.
    return {
      ...core,
      xM: core.xM + prepared.origin[0],
      yM: core.yM + prepared.origin[1],
      zM: core.zM + prepared.origin[2],
    };
  }, [preparation.metadata, vectorModeId]);

  const streamlineParameters = useMemo(() => {
    const prepared = preparation.metadata;
    if (!prepared) return null;
    const storm = prepared.stormCenter;
    const seedDomain = streamlineBoxSeedDimensions(
      renderGrid,
      prepared.shape,
      prepared.spacing,
      prepared.origin,
    );
    const domainH = seedDomain.domainSpan[2];
    const desiredSeedCount = seedMode === "feature"
      ? seedCount * 4
      : seedCount * seedCount * seedCount;
    const ringSeeds = seedMode === "vortex" && vortexCore
      ? buildVortexRingSeeds(vortexCore, {
          xMinM: prepared.origin[0],
          xMaxM: prepared.origin[0] + (prepared.shape[0] - 1) * prepared.spacing[0],
          yMinM: prepared.origin[1],
          yMaxM: prepared.origin[1] + (prepared.shape[1] - 1) * prepared.spacing[1],
          zMinM: prepared.origin[2],
          zMaxM: prepared.origin[2] + domainH,
        }, { azimuthCount: seedCount * 4 })
      : null;
    let seeds = ringSeeds
      ? ringSeeds.points
      : seedMode === "feature" && prepared.featureSeeds.length > 0
        ? prepared.featureSeeds.slice(0, Math.min(prepared.featureSeeds.length, desiredSeedCount * 3))
        : buildSeedBox(
            [storm[0], storm[1], seedDomain.centerZ],
            seedDomain.halfSize,
            seedCount,
          );
    if (brickedVectorEligible && seedMode === "box") {
      const seedIdentity = `${vectorBrickIdentity}:${seedCount}`;
      if (vectorBrickSeedAnchorRef.current.identity === seedIdentity) {
        seeds = vectorBrickSeedAnchorRef.current.seeds;
      } else {
        vectorBrickSeedAnchorRef.current = { identity: seedIdentity, seeds };
      }
    }
    const minSpacing = Math.min(...prepared.spacing);
    const speedReference = prepared.seedColumnSpeedMaximum ?? prepared.speed95;
    const integrationStep = Number.isFinite(minSpacing) && minSpacing > 0 && speedReference > 0
      ? (minSpacing * 0.5) / speedReference
      : 0.5;
    return {
      seeds,
      integrationStep,
      maxSteps,
      tubeRadius,
      scale: { xy: sceneScale.xy, z: sceneScale.z },
    };
  }, [preparation.metadata, renderGrid, seedMode, seedCount, vortexCore, maxSteps, tubeRadius, sceneScale.xy, sceneScale.z, brickedVectorEligible, vectorBrickIdentity]);
  const streamlineExtractionKey = streamlineParameters
    ? [seedMode, seedCount, maxSteps, tubeRadius, sceneScale.xy, sceneScale.z].join(":")
    : "none";
  const pendingSeedBrickExpansion = brickedVectorEligible
    && vectorBrickPlan
    && streamlineParameters
    && preparation.metadata
    ? expandVectorBrickNeighborhood(
        vectorBrickPlan,
        detectSeedBoundaryContacts(
          streamlineParameters.seeds,
          preparation.metadata.shape,
          preparation.metadata.spacing,
          preparation.metadata.origin,
        ),
      )
    : null;
  const streamlineExtraction = useStreamlineExtraction({
    enabled: preparation.handle != null && vectors != null && pendingSeedBrickExpansion == null,
    preparedHandle: preparation.handle,
    parameters: streamlineParameters,
    key: streamlineExtractionKey,
  });
  const pendingTrajectoryBrickExpansion = brickedVectorEligible
    && vectorBrickPlan
    && streamlineExtraction.result
    ? expandVectorBrickNeighborhood(
        vectorBrickPlan,
        streamlineExtraction.result.boundaryContacts,
      )
    : null;
  const pendingVectorBrickExpansion = pendingSeedBrickExpansion
    ?? pendingTrajectoryBrickExpansion;
  const colorScope = `${sourceRunDir}\u0000${activeMode.id}\u0000epoch=${colorLatchEpoch}`;
  const colorMagnitudeRef = useRef<LatchedMagnitude>({ scope: colorScope, reference: 0 });
  const carriedColorMaximum = carriedMagnitude(colorMagnitudeRef.current, colorScope);
  const frameColorMaximum = useMemo(() => pendingVectorBrickExpansion ? 0 : streamlineMagnitudeMaximum([
    streamlineExtraction.result?.forward.path?.scalars,
    streamlineExtraction.result?.backward.path?.scalars,
  ]), [streamlineExtraction.result, pendingVectorBrickExpansion]);
  const colorMaximum = Math.max(carriedColorMaximum, frameColorMaximum);
  useEffect(() => {
    if (!(frameColorMaximum > 0)) return;
    colorMagnitudeRef.current = latchMagnitude(
      colorMagnitudeRef.current,
      colorScope,
      frameColorMaximum,
    );
  }, [colorScope, frameColorMaximum]);

  const gpuUploadBytes = streamlineExtraction.result
    ? [
        streamlineExtraction.result.forward.path,
        streamlineExtraction.result.forward.arrows,
        streamlineExtraction.result.backward.path,
        streamlineExtraction.result.backward.arrows,
      ].reduce((sum, geometry) => sum + streamlineGeometryBytes(geometry), 0)
    : 0;
  const gpuUpload = useVtkGpuResourceOwnership({
    enabled: streamlineExtraction.result != null && !streamlineExtraction.loading
      && streamlineExtraction.error == null && pendingVectorBrickExpansion == null
      && gpuUploadBytes > 0,
    owner: "streamlines",
    generation: JSON.stringify([
      sourceRunDir, sourceDefinitionIdentity, loadedVectorFrame?.stepIndex, loadedVectorFrame?.publicationGeneration,
      vectorModeId, streamlineExtractionKey, vtkLifecycleGeneration,
    ]),
    bytes: gpuUploadBytes,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const geometryEvidence = useMemo(() => lazyAppearanceEvidence(() => ({
    geometryFingerprint: streamlineGeometryFingerprint(streamlineExtraction.result),
    seedFingerprint: streamlineSeedFingerprint(streamlineParameters?.seeds ?? null),
  })), [streamlineExtraction.result, streamlineParameters?.seeds]);
  const geometryEvidenceRef = useRef(geometryEvidence);
  geometryEvidenceRef.current = geometryEvidence;
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      integratedPoints: streamlineExtraction.result?.integratedPoints ?? null,
      colorMaximum, colorReference: "maximum-integrated-path-magnitude",
      missingSupport: (preparation.metadata?.sourceNonFiniteCount ?? 0) > 0,
      missingDataPolicy: "stop-at-incomplete-support",
      preparedShape: preparation.metadata?.shape ?? null,
      preparedOrigin: preparation.metadata?.origin ?? null,
      preparedStormCenter: preparation.metadata?.stormCenter ?? null,
      integrationStep: streamlineParameters?.integrationStep ?? null,
      logicalGpuUploadBytes: gpuUpload.admitted ? gpuUploadBytes : 0,
    },
  };
  useEffect(() => {
    if (!pendingVectorBrickExpansion) return;
    setVectorBrickRequest((current) => {
      if (current?.identity === vectorBrickIdentity
          && current.expansion.lower.join(":") === pendingVectorBrickExpansion.lower.join(":")
          && current.expansion.upper.join(":") === pendingVectorBrickExpansion.upper.join(":")) {
        return current;
      }
      return {
        identity: vectorBrickIdentity,
        expansion: pendingVectorBrickExpansion,
        iteration: current?.identity === vectorBrickIdentity ? current.iteration + 1 : 1,
      };
    });
  }, [pendingVectorBrickExpansion, vectorBrickIdentity]);

  // ---------- vtk.js scene refs ----------
  const containerRef = useRef<HTMLDivElement | null>(null);
  type SceneRefs = {
    renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    forwardData: ReturnType<typeof vtkPolyData.newInstance>;
    mapper: ReturnType<typeof vtkMapper.newInstance>;
    actor: ReturnType<typeof vtkActor.newInstance>;
    forwardArrows: ReturnType<typeof vtkPolyData.newInstance>;
    forwardArrowMapper: ReturnType<typeof vtkMapper.newInstance>;
    forwardArrowActor: ReturnType<typeof vtkActor.newInstance>;
    forwardHasMagnitude: boolean;
    backwardData: ReturnType<typeof vtkPolyData.newInstance>;
    backwardMapper: ReturnType<typeof vtkMapper.newInstance>;
    backwardActor: ReturnType<typeof vtkActor.newInstance>;
    backwardArrows: ReturnType<typeof vtkPolyData.newInstance>;
    backwardArrowMapper: ReturnType<typeof vtkMapper.newInstance>;
    backwardArrowActor: ReturnType<typeof vtkActor.newInstance>;
    backwardHasMagnitude: boolean;
    ctf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    lighting: VtkSceneLighting;
  };
  const sceneRef = useRef<SceneRefs | null>(null);
  const cameraInitializedRef = useRef(false);
  const streamlineColorStateRef = useRef({
    activeMode,
    colorMaximum,
    colorMode,
    uniformColor,
    contextGeneration: vtkLifecycleGeneration,
  });
  streamlineColorStateRef.current = {
    activeMode,
    colorMaximum,
    colorMode,
    uniformColor,
    contextGeneration: vtkLifecycleGeneration,
  };
  const appliedStreamlineColorSignatureRef = useRef<string | null>(null);
  const applyStreamlineColor = useCallback((scene: SceneRefs): boolean => {
    const state = streamlineColorStateRef.current;
    const background = sceneAppearanceRef.current.background;
    const signature = [
      state.contextGeneration,
      state.activeMode.id,
      state.colorMaximum,
      state.colorMode,
      state.uniformColor,
      scene.forwardHasMagnitude,
      scene.backwardHasMagnitude,
      ...background,
    ].join(":");
    if (appliedStreamlineColorSignatureRef.current === signature) return false;

    const upper = state.colorMaximum > 0 ? state.colorMaximum : 1;
    scene.ctf.setMappingRange(0, upper);
    scene.ctf.modified();
    const useMagnitude = state.colorMode === "magnitude";
    scene.mapper.setScalarVisibility(useMagnitude && scene.forwardHasMagnitude);
    scene.backwardMapper.setScalarVisibility(useMagnitude && scene.backwardHasMagnitude);
    scene.forwardArrowMapper.setScalarVisibility(useMagnitude && scene.forwardHasMagnitude);
    scene.backwardArrowMapper.setScalarVisibility(useMagnitude && scene.backwardHasMagnitude);
    const uniform = contrastAdjustedColor(hexToSceneRgb(state.uniformColor), background);
    scene.actor.getProperty().setColor(...uniform);
    scene.backwardActor.getProperty().setColor(...uniform);
    const markerColor = contrastAdjustedColor(
      hexToSceneRgb(state.uniformColor).map((channel) => Math.min(1, channel * 1.25)) as
        [number, number, number],
      background,
    );
    scene.forwardArrowActor.getProperty().setColor(...markerColor);
    scene.backwardArrowActor.getProperty().setColor(...markerColor);
    appliedStreamlineColorSignatureRef.current = signature;
    return true;
  }, []);
  const applyStreamlineColorRef = useRef(applyStreamlineColor);
  applyStreamlineColorRef.current = applyStreamlineColor;

  /* Compass points at the tracked mesocyclone; undefined when tracking is off or no feature found. */
  const cameraOrientation = useCameraOrientation(
    () => sceneRef.current?.renderWindow ?? null,
    sceneScale,
    stormCentre,
    vtkLifecycleGeneration,
  );

  /* Pane camera lock: orbiting one pane orbits all, keeping side-by-side comparisons aligned. */
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

  /* Radar floor: own 2D hyperslab path, resolution independent of volume voxel cap. */
  const floorSelection = useRadarFloorSelection(
    fieldsForStep,
    controlScope.scope,
    `streamlines:${sourceRunDir}`,
  );
  const floor = useRadarFloorLayer(() => sceneRef.current?.renderWindow ?? null, {
    runDir: sourceRunDir,
    stepIndex: requestedStepIndex,
    publicationGeneration: requestedPublicationGeneration,
    fieldName: floorSelection.fieldName,
    reduction: floorSelection.reduction,
    opacity: floorSelection.opacity,
    availableFields: fieldsForStep,
    fieldOrigins: currentStep?.fieldOrigins,
    enabled: floorSelection.enabled,
    deferDataRender: true,
    grid: currentStep?.grid ?? renderGrid,
    scale: sceneScale,
    lifecycleGeneration: vtkLifecycleGeneration,
  });
  const panePreparationError = floor.error ?? error ?? preparation.error ?? streamlineExtraction.error ?? gpuUpload.error;
  const panePreparationReady = haveVectors && boundedWindowPlan.ok && !panePreparationError && !loading
    && !preparation.loading && !streamlineExtraction.loading && pendingVectorBrickExpansion == null
    && preparation.metadata != null && streamlineParameters != null && streamlineExtraction.result != null
    && loadedVectorFrame?.stepIndex === requestedStepIndex
    && loadedVectorFrame?.publicationGeneration === requestedPublicationGeneration
    && (gpuUploadBytes === 0 || gpuUpload.admitted)
    && (!floorSelection.enabled || (!floor.loading && floor.values != null));
  const panePresentation = usePanePresentationBarrier({ ready: panePreparationReady, error: panePreparationError });
  const committedGridRef = useRef(renderGrid);
  const displayGrid = panePresentation.pending || !panePreparationReady ? committedGridRef.current : renderGrid;
  // Domain bounding box: keeps camera framing stable as storms advect.
  useDomainBox(
    () => sceneRef.current?.renderWindow ?? null,
    displayGrid,
    sceneScale,
    vtkLifecycleGeneration,
    sceneAppearance.resolved.reference,
  );
  const axisLabels = useDomainAxes(
    () => sceneRef.current?.renderWindow ?? null,
    containerRef,
    displayGrid,
    sceneScale,
    vtkLifecycleGeneration,
    sceneAppearance.resolved.reference,
  );
  useEffect(() => {
    onFrameDisplayChange?.({
      requestedStepIndex,
      requestedPublicationGeneration,
      renderedStepIndex,
      renderedPublicationGeneration,
      loading: haveVectors && boundedWindowPlan.ok && !error && !preparation.error
        && !streamlineExtraction.error
        && (loading || preparation.loading || streamlineExtraction.loading
          || pendingVectorBrickExpansion != null
          || frameTransition.kind !== "ready"
          || !panePreparationReady || panePresentation.pending || floor.loading || floor.error != null
          || floorSelection.enabled && floor.committedPresentationIdentity !== floor.presentationIdentity),
    });
  }, [
    panePreparationReady, panePresentation.pending, floor.loading, floor.error, floorSelection.enabled,
    floor.committedPresentationIdentity, floor.presentationIdentity,
    boundedWindowPlan,
    error,
    frameTransition.kind,
    haveVectors,
    loading,
    onFrameDisplayChange,
    preparation.error,
    preparation.loading,
    streamlineExtraction.error,
    streamlineExtraction.loading,
    pendingVectorBrickExpansion,
    renderedPublicationGeneration,
    renderedStepIndex,
    requestedPublicationGeneration,
    requestedStepIndex,
  ]);

  const cameraElevation = useCameraElevation(displayGrid);
  useCameraElevationPlacement(() => sceneRef.current?.renderWindow ?? null, {
    grid: displayGrid,
    scale: sceneScale,
    selection: cameraElevation,
    targetXYM: stormCentre ? [stormCentre[0], stormCentre[1]] : null,
    lifecycleGeneration: vtkLifecycleGeneration,
    onCameraApplied: () => { cameraInitializedRef.current = true; },
  });

  // Re-frame on run change; a different run can have a very different domain extent.
  useVtkRunReset(sourceRunDir, cameraInitializedRef, () => {
    // Hide stale tubes on run switch; re-shown once new vectors load.
    const s = sceneRef.current;
    if (!s) return;
    s.actor.setVisibility(false);
    s.backwardActor.setVisibility(false);
    s.forwardArrowActor.setVisibility(false);
    s.backwardArrowActor.setVisibility(false);
    s.renderWindow.getRenderWindow().render();
  });

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

    const forwardData = vtkPolyData.newInstance();

    const ctf = vtkColorTransferFunction.newInstance();
    const magnitudePreset = vtkColorMaps.getPresetByName("Viridis (matplotlib)");
    if (magnitudePreset) ctf.applyColorMap(magnitudePreset);
    ctf.setMappingRange(0, 1);
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(forwardData);
    mapper.setLookupTable(ctf);
    mapper.setUseLookupTableScalarRange(true);
    mapper.setInterpolateScalarsBeforeMapping(true);
    mapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    mapper.setColorByArrayName(VTK_SCALAR_SLOT);
    mapper.setScalarVisibility(false);

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(0.36, 0.65, 1.0); // light blue
    const forwardArrows = vtkPolyData.newInstance();
    const forwardArrowMapper = vtkMapper.newInstance();
    forwardArrowMapper.setInputData(forwardArrows);
    forwardArrowMapper.setLookupTable(ctf);
    forwardArrowMapper.setUseLookupTableScalarRange(true);
    forwardArrowMapper.setInterpolateScalarsBeforeMapping(true);
    forwardArrowMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    forwardArrowMapper.setColorByArrayName(VTK_SCALAR_SLOT);
    forwardArrowMapper.setScalarVisibility(false);
    const forwardArrowActor = vtkActor.newInstance();
    forwardArrowActor.setMapper(forwardArrowMapper);
    forwardArrowActor.getProperty().setColor(0.82, 0.9, 1.0);

    const backwardData = vtkPolyData.newInstance();
    const backwardMapper = vtkMapper.newInstance();
    backwardMapper.setInputData(backwardData);
    backwardMapper.setLookupTable(ctf);
    backwardMapper.setUseLookupTableScalarRange(true);
    backwardMapper.setInterpolateScalarsBeforeMapping(true);
    backwardMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    backwardMapper.setColorByArrayName(VTK_SCALAR_SLOT);
    backwardMapper.setScalarVisibility(false);
    const backwardActor = vtkActor.newInstance();
    backwardActor.setMapper(backwardMapper);
    backwardActor.getProperty().setColor(0.36, 0.65, 1.0);
    const backwardArrows = vtkPolyData.newInstance();
    const backwardArrowMapper = vtkMapper.newInstance();
    backwardArrowMapper.setInputData(backwardArrows);
    backwardArrowMapper.setLookupTable(ctf);
    backwardArrowMapper.setUseLookupTableScalarRange(true);
    backwardArrowMapper.setInterpolateScalarsBeforeMapping(true);
    backwardArrowMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    backwardArrowMapper.setColorByArrayName(VTK_SCALAR_SLOT);
    backwardArrowMapper.setScalarVisibility(false);
    const backwardArrowActor = vtkActor.newInstance();
    backwardArrowActor.setMapper(backwardArrowMapper);
    backwardArrowActor.getProperty().setColor(0.82, 0.9, 1.0);

    const renderer = renderWindow.getRenderer();
    const lighting = createVtkSceneLighting(renderer);
    lighting.apply(
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );
    for (const surfaceActor of [actor, backwardActor, forwardArrowActor, backwardArrowActor]) {
      applySurfaceLighting(surfaceActor.getProperty(), sceneAppearanceRef.current.lightingOptions);
    }
    renderer.addActor(actor);
    renderer.addActor(backwardActor);
    renderer.addActor(forwardArrowActor);
    renderer.addActor(backwardArrowActor);
    // Defer resetCamera until first valid integration finishes.

    sceneRef.current = {
      renderWindow,
      forwardData,
      mapper,
      actor,
      forwardArrows,
      forwardArrowMapper,
      forwardArrowActor,
      forwardHasMagnitude: false,
      backwardData,
      backwardMapper,
      backwardActor,
      backwardArrows,
      backwardArrowMapper,
      backwardArrowActor,
      backwardHasMagnitude: false,
      ctf,
      lighting,
    };

    registerVtkAppearanceCapture(container, renderWindow, () => ({
      ...(committedAppearanceRef.current ?? appearanceEvidenceRef.current),
      parameters: {
        ...(committedAppearanceRef.current?.parameters ?? appearanceEvidenceRef.current.parameters),
        ...(committedGeometryEvidenceRef.current ?? geometryEvidenceRef.current)(),
        requestedFrameIndex: appearanceEvidenceRef.current.parameters?.requestedFrameIndex ?? null,
      },
    }));

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
          () => renderer.removeActor(actor),
          () => renderer.removeActor(backwardActor),
          () => renderer.removeActor(forwardArrowActor),
          () => renderer.removeActor(backwardArrowActor),
          () => releaseVtkPointDataArrays(forwardData.getPointData()),
          () => releaseVtkPointDataArrays(backwardData.getPointData()),
          () => releaseVtkPointDataArrays(forwardArrows.getPointData()),
          () => releaseVtkPointDataArrays(backwardArrows.getPointData()),
        ],
        resources: [
          forwardArrowActor,
          forwardArrowMapper,
          forwardArrows,
          backwardArrowActor,
          backwardArrowMapper,
          backwardArrows,
          actor,
          mapper,
          forwardData,
          backwardActor,
          backwardMapper,
          backwardData,
          ctf,
        ],
        renderWindow,
      });
    };
  }, [ctxLoss.available, ctxLoss.generation, vtkBackend.backend, vtkBackend.confirmActiveApi]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const appearance = sceneAppearance.resolved;
    if (panePresentation.pending) return;
    scene.renderWindow.getRenderer().setBackground(...appearance.background);
    scene.lighting.apply(appearance.lightingOptions, appearance.isLight);
    for (const actor of [
      scene.actor,
      scene.backwardActor,
      scene.forwardArrowActor,
      scene.backwardArrowActor,
    ]) {
      applySurfaceLighting(actor.getProperty(), appearance.lightingOptions);
    }
    const uniform = contrastAdjustedColor(hexToSceneRgb(uniformColor), appearance.background);
    const markerColor = contrastAdjustedColor(
      hexToSceneRgb(uniformColor).map((channel) => Math.min(1, channel * 1.25)) as [number, number, number],
      appearance.background,
    );
    scene.actor.getProperty().setColor(...uniform);
    scene.backwardActor.getProperty().setColor(...uniform);
    scene.forwardArrowActor.getProperty().setColor(...markerColor);
    scene.backwardArrowActor.getProperty().setColor(...markerColor);
    scene.renderWindow.getRenderWindow().render();
    const committed = committedAppearanceRef.current;
    if (committed) committedAppearanceRef.current = { ...committed, parameters: {
      ...committed.parameters,
      background: appearance.backgroundHex,
      lighting: appearance.lighting,
      lightingControls: appearance.lightingOptions,
    } };
  }, [sceneAppearance.resolved, uniformColor, vtkLifecycleGeneration, panePresentation.pending]);

  // ---------- Field loading: (u, v, w) ----------
  // Debounce the scrubbed step so a fast drag loads only the settled frame —
  // three ~21 MB fields plus RK2 integration is the heaviest per-frame cost.
  const loadScopeRef = useRef<string | null>(null);
  if (loadScopeRef.current == null) {
    loadScopeRef.current = `streamlines:${Math.random().toString(36).slice(2)}`;
  }

  useEffect(() => {
    if (outputHierarchySource.loading) return;
    if (loadFrame.stepIndex == null || !haveVectors) {
      setLoadedVectors(null);
      setVectorBrickPlan(null);
      return;
    }
    const [c0, c1, c2] = activeMode.components;
    const components = [c0, c1, c2] as const;
    let cancelled = false;
    let completed = false;
    let admissionReservationId: number | null = null;
    const controller = new AbortController();
    if (!brickedVectorEligible && !boundedWindowPlan.ok) {
      setLoadedVectors(null);
      setLoading(false);
      setError(boundedWindowPlan.reason);
      return;
    }
    setLoading(true);
    setError(null);
    cancelInflightPrefetches();
    beginFieldLoad();
    let componentWindows: readonly (readonly FieldLoadWindow[])[] = [];
    let brickHierarchyLevel: number | undefined;
    let plannedVectorBrickPlan: VectorBrickNeighborhoodPlan | null = null;
    const loadComponents = async () => {
      if (brickedVectorEligible) {
        const hierarchies = await Promise.all(components.map((component) =>
          readFieldBrickHierarchy(sourceRunDir, currentStep!, component)));
        const plan = planVectorBrickNeighborhood(
          hierarchies as [typeof hierarchies[0], typeof hierarchies[0], typeof hierarchies[0]],
          {
            maximumPreparedCells: paneVoxelCap,
            expansion: vectorBrickExpansion,
            terrainFollowing: currentStep?.grid?.terrainFollowing,
            focusSourceIndex: seedMode === "box"
              ? undefined
              : selectVectorBrickFeatureFocus(
                  hierarchies as [typeof hierarchies[0], typeof hierarchies[0], typeof hierarchies[0]],
                  seedMode === "vortex" ? { maximumSourceZFraction: 0.25 } : undefined,
                ),
          },
        );
        plannedVectorBrickPlan = plan;
        componentWindows = components.map(() => plan.items.map((item) => item.window));
        brickHierarchyLevel = plan.level > 0 ? plan.level : undefined;
      } else {
        setVectorBrickPlan(null);
        const windows = boundedWindowPlan.ok ? boundedWindowPlan.plan.windows : [];
        componentWindows = components.map((component) => windowsForFieldAggregation(
          windows,
          component,
          {
            sourceKind: currentStep?.fieldOrigins?.[component],
            terrainFollowing: currentStep?.grid?.terrainFollowing,
          },
        ));
      }
      const admission = visualizationResourceGovernor.reserve({
        owner: vectorResourceOwnerRef.current!,
        generation: `${loadFrame.publicationGeneration ?? "none"}:${loadFrame.stepIndex}:planned`,
        priority: "visible",
        claims: components.flatMap((component, componentIndex) =>
          componentWindows[componentIndex].map((window, windowIndex) =>
            visualizationReservationClaim(
              `${vectorResourceOwnerRef.current}:${loadFrame.publicationGeneration ?? "none"}:${loadFrame.stepIndex}:${component}:${windowIndex}:input`,
              "native-read",
              (window.maxOutputBytes ?? 0) + MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES,
            ))),
      });
      if (!admission.ok) throw new Error(admission.reason);
      admissionReservationId = admission.reservation.id;
      return Promise.all(components.map((component, componentIndex) => readFieldWindows(
        sourceRunDir,
        loadFrame.stepIndex!,
        component,
        loadFrame.publicationGeneration,
        componentWindows[componentIndex],
        {
          latestScope: `${loadScopeRef.current}:${componentIndex}`,
          signal: controller.signal,
          brickHierarchyLevel,
        },
      ))) as Promise<[Npy3D, Npy3D, Npy3D]>;
    };
    loadComponents()
      .then(([a, b, c]) => {
        if (cancelled) return;
        completed = true;
        setLoadedVectors({
          vectors: [a, b, c],
          runDir: sourceRunDir,
          modeId: activeMode.id,
          stepIndex: loadFrame.stepIndex!,
          publicationGeneration: loadFrame.publicationGeneration,
          definitionIdentity: sourceDefinitionIdentity,
        });
        setVectorBrickPlan(plannedVectorBrickPlan);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        // Cooperative supersession is not a field failure. Ordinarily the
        // effect cleanup marks this request cancelled first; keep the boundary
        // defensive for platform cancellation acknowledgement races.
        if (e instanceof DOMException && e.name === "AbortError") {
          setLoading(false);
          return;
        }
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      })
      .finally(() => {
        if (admissionReservationId != null) {
          visualizationResourceGovernor.release(admissionReservationId);
        }
        endFieldLoad();
        if (cancelled || !completed) return;
        if (brickedVectorEligible) return;
        for (const nextFrame of frameIdentitiesToPrefetch(sourceRunDir, loadFrame.stepIndex!, loadFrame.publicationGeneration)) {
          const nextPublication = nextFrame.publicationGeneration;
          for (let componentIndex = 0; componentIndex < components.length; componentIndex += 1) {
            if (!tryBeginPrefetch()) return;
            const prefetchController = createPrefetchController();
            void readFieldWindows(
              sourceRunDir,
              nextFrame.stepIndex,
              components[componentIndex],
              nextPublication,
              componentWindows[componentIndex],
              { signal: prefetchController.signal, priority: "background" },
            ).catch(() => undefined).finally(() => {
              releasePrefetchController(prefetchController);
              endPrefetch();
            });
          }
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [
    sourceRunDir,
    loadFrame,
    sourceDefinitionIdentity,
    haveVectors,
    activeMode,
    boundedWindowPlan,
    brickedVectorEligible,
    paneVoxelCap,
    vectorBrickExpansion,
    currentStep,
    currentStep?.fieldOrigins,
    currentStep?.grid?.terrainFollowing,
    outputHierarchySource.loading,
    seedMode,
  ]);


  // ---------- Off-thread integration + tube/arrow geometry ----------
  useEffect(() => {
    const scene = sceneRef.current;
    const result = streamlineExtraction.result;
    if (!scene || panePresentation.pending || !panePreparationReady || floor.loading || floor.error || !result || !preparation.metadata || !streamlineParameters
        || pendingVectorBrickExpansion) return;
    if (gpuUpload.error) {
      scene.actor.setVisibility(false);
      scene.backwardActor.setVisibility(false);
      scene.forwardArrowActor.setVisibility(false);
      scene.backwardArrowActor.setVisibility(false);
      scene.renderWindow.getRenderWindow().render();
      return;
    }
    if (gpuUploadBytes > 0 && !gpuUpload.admitted) return;
    applyStreamlineGeometry(scene.forwardData, result.forward.path);
    applyStreamlineGeometry(scene.backwardData, result.backward.path);
    applyStreamlineGeometry(scene.forwardArrows, result.forward.arrows);
    applyStreamlineGeometry(scene.backwardArrows, result.backward.arrows);
    scene.forwardHasMagnitude = result.forward.path?.scalars != null;
    scene.backwardHasMagnitude = result.backward.path?.scalars != null;
    scene.actor.setVisibility(result.forward.path != null);
    scene.backwardActor.setVisibility(result.backward.path != null);
    scene.forwardArrowActor.setVisibility(result.forward.arrows != null);
    scene.backwardArrowActor.setVisibility(result.backward.arrows != null);
    const lineWidth = result.representation === "gpu-lines" ? 2.5 : 1;
    scene.actor.getProperty().setLineWidth(lineWidth);
    scene.backwardActor.getProperty().setLineWidth(lineWidth);
    applyStreamlineColorRef.current(scene);
    if (!cameraInitializedRef.current) {
      resetVtkCameraToStormView(
        scene.renderWindow.getRenderer(),
        seedMode === "feature"
          ? featureSeedBounds(
              streamlineParameters.seeds,
              preparation.metadata.spacing,
              sceneScaleRef.current,
            )
          : null,
      );
      cameraInitializedRef.current = true;
    }
    floor.commitData();
    scene.renderWindow.getRenderWindow().render();
    committedGridRef.current = renderGrid;
    committedAppearanceRef.current = { ...appearanceEvidenceRef.current,
      frameIndex: loadedVectorFrame?.stepIndex ?? null, parameters: { ...appearanceEvidenceRef.current.parameters,
        renderedPublicationGeneration: loadedVectorFrame?.publicationGeneration ?? null } };
    committedGeometryEvidenceRef.current = geometryEvidenceRef.current;
    markFrameRendered(contextGeneration);
    setRenderedStepIndex(loadedVectorFrame?.stepIndex ?? null);
    setRenderedPublicationGeneration(loadedVectorFrame?.publicationGeneration ?? null);
  }, [
    streamlineExtraction.result,
    panePresentation.pending,
    panePreparationReady,
    floor.loading, floor.error, floor.commitData,
    renderGrid,
    preparation.metadata,
    streamlineParameters,
    seedMode,
    contextGeneration,
    markFrameRendered,
    loadedVectorFrame?.publicationGeneration,
    loadedVectorFrame?.stepIndex,
    vtkLifecycleGeneration,
    pendingVectorBrickExpansion,
    gpuUpload.admitted,
    gpuUpload.error,
    gpuUploadBytes,
  ]);

  // Preserve manual changes while scrubbing, but choose a physical default
  // again when the prepared lattice spacing changes (for example coarse/fine).
  useEffect(() => {
    const prepared = preparation.metadata;
    if (!prepared) return;
    const key = `${prepared.shape.join(":")}|${prepared.spacing.join(":")}`;
    if (tubeRadiusGridRef.current === key) return;
    tubeRadiusGridRef.current = key;
    setTubeRadius(defaultStreamlineTubeRadiusM(prepared.shape, prepared.spacing));
  }, [preparation.metadata]);

  // ---------- Vector-field colour ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || panePresentation.pending) return;
    if (applyStreamlineColorRef.current(scene)) {
      scene.renderWindow.getRenderWindow().render();
    }
  }, [activeMode, colorMaximum, colorMode, uniformColor, vtkLifecycleGeneration, panePresentation.pending]);
  const streamlineGeometryCommitted = !streamlineExtraction.loading
    && streamlineExtraction.error == null
    && pendingVectorBrickExpansion == null
    && (streamlineExtraction.result?.integratedPoints ?? 0) > 0
    && gpuUploadBytes > 0
    && renderedStepIndex === loadedVectorFrame?.stepIndex
    && renderedPublicationGeneration === loadedVectorFrame?.publicationGeneration;

  // ---------- Vertical exaggeration ----------
  // Scale applied to points, not actor (avoids distorting tube cross-sections); re-frame after first bounds.
  const focusInputsRef = useRef({ prepared: preparation.metadata, seedMode, seedCount });
  focusInputsRef.current = { prepared: preparation.metadata, seedMode, seedCount };
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const currentScale = { xy: sceneScale.xy, z: sceneScale.z };
    if (cameraInitializedRef.current) {
      // Reframe in place — keep the user's orbit; re-applying the oblique storm
      // view here would yank the camera back to the preset on every z-scale tick.
      const { prepared, seedMode: mode, seedCount: count } = focusInputsRef.current;
      const focus = prepared && mode === "feature"
        ? featureSeedBounds(
            prepared.featureSeeds.subarray(0, Math.min(
              prepared.featureSeeds.length,
              count * 4 * 3,
            )),
            prepared.spacing,
            currentScale,
          )
        : null;
      refitVtkCameraToBounds(scene.renderWindow.getRenderer(), focus ?? undefined);
    }
    scene.renderWindow.getRenderWindow().render();
    // Focus inputs through a ref: depending on preparation.metadata would re-fit camera every frame.
  }, [sceneScale.xy, sceneScale.z, vtkLifecycleGeneration]);

  // ---------- Reset camera ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    resetVtkCameraToStormView(
      scene.renderWindow.getRenderer(),
      focusInputsRef.current.prepared && focusInputsRef.current.seedMode === "feature"
        ? featureSeedBounds(
            focusInputsRef.current.prepared.featureSeeds.subarray(0, Math.min(
              focusInputsRef.current.prepared.featureSeeds.length,
              focusInputsRef.current.seedCount * 4 * 3,
            )),
            focusInputsRef.current.prepared.spacing,
            sceneScaleRef.current,
          )
        : null,
    );
    scene.renderWindow.getRenderWindow().render();
  }, [resetTick]);

  const seedTotal = seedCount * seedCount * seedCount;
  const desiredFeatureSeeds = seedCount * 4;
  const displayedSeedTotal = seedMode === "feature"
    ? Math.min(desiredFeatureSeeds, (preparation.metadata?.featureSeeds.length ?? 0) / 3)
    : seedTotal;
  return (
    <div className="volume-render">
      <Grouped3DControls
        ariaLabel="Streamline controls"
        scope={controlScope}
        groups={[
          {
            id: "field",
            label: "Field",
            status: activeMode.short,
            controls: <>
        <OutputHierarchyControls source={outputHierarchySource} />
        <PlotCompressionControls step={loadedStep} fields={activeMode.components} />
        <ControlGroup label="Vector field">
          <PillGroup
            options={VECTOR_MODES.map((mode) => ({
              value: mode.id,
              label: mode.short,
              unavailable: availableModes.some((available) => available.id === mode.id)
                ? undefined
                : `Requires ${mode.components.join(", ")} as rank-3 fields in this frame.`,
            }))}
            value={vectorModeId}
            onChange={setVectorModeId}
          />
        </ControlGroup>
        <ControlGroup label="Color">
          <PillGroup
            options={STREAMLINE_COLOR_OPTIONS}
            value={colorMode}
            onChange={setColorMode}
          />
              <button
                type="button"
                className="plot-reanchor-btn"
                disabled={colorMode !== "magnitude" || !(carriedColorMaximum > frameColorMaximum && frameColorMaximum > 0)}
                onClick={() => setColorLatchEpoch((epoch) => epoch + 1)}
                title={colorMode !== "magnitude" ? "Re-anchoring applies to magnitude colors"
                  : carriedColorMaximum > frameColorMaximum && frameColorMaximum > 0
                    ? `Colour is normalized to an earlier path maximum of ${carriedColorMaximum.toPrecision(3)}. Re-anchor to this frame.`
                    : "Color range already matches this frame"}
              >
                Re-anchor
              </button>
        </ControlGroup>
        {colorMode === "uniform" && (
          <ControlGroup label="Uniform color">
            <input
              type="color"
              className="scene-color-input"
              aria-label="Uniform streamline color"
              value={uniformColor}
              onChange={(event) => setUniformColors((current) => ({
                ...current,
                [activeMode.id]: event.target.value,
              }))}
            />
          </ControlGroup>
        )}
            </>,
          },
          {
            id: "tracing",
            label: "Tracing",
            status: seedMode === "feature" ? "Features" : seedMode === "vortex" ? "Vortex" : "Box",
            controls: <>
        <ControlGroup label="Seed placement">
          <PillGroup
            options={vectorModeId === "vorticity"
              ? [
                { value: "vortex" as const, label: "Vortex ring" },
                { value: "feature" as const, label: "Resolved features" },
                { value: "box" as const, label: "Storm box" },
              ]
              : [
                { value: "feature" as const, label: "Resolved features" },
                { value: "box" as const, label: "Storm box" },
              ]}
            value={seedMode}
            onChange={setSeedMode}
          />
        </ControlGroup>
        <ControlGroup label="Seeds">
          <input
            type="range"
            min={2}
            max={8}
            step={1}
            value={seedCount}
            onChange={(e) => setSeedCount(Number.parseInt(e.target.value, 10))}
            className="volume-render__slider"
          />
          <span className="volume-render__slider-value">
            {seedMode === "vortex"
              ? (vortexCore
                ? `ring · r≈${Math.round(vortexCore.estimatedCoreRadiusM)} m`
                : "no near-surface vortex")
              : seedMode === "feature"
              ? `${Math.floor(displayedSeedTotal)} ranked`
              : `${seedCount}³ = ${seedTotal}`}
          </span>
        </ControlGroup>
        <ControlGroup label="Max steps">
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={maxSteps}
            onChange={(e) => setMaxSteps(Number.parseInt(e.target.value, 10))}
            className="volume-render__slider"
          />
          <span className="volume-render__slider-value">{maxSteps}</span>
        </ControlGroup>
        <ControlGroup label="Tube radius">
          <input
            type="range"
            min={25}
            max={1000}
            step={25}
            value={tubeRadius}
            onChange={(e) => setTubeRadius(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
          />
          <span className="volume-render__slider-value">{tubeRadius} m</span>
        </ControlGroup>
            </>,
          },
          {
            id: "radar",
            label: "Radar",
            status: floorSelection.planeFields.length === 0
              ? "Unavailable"
              : floorSelection.enabled ? "On" : "Off",
            controls: <>
        <RadarFloorControls
          selection={floorSelection}
          lattice={floor.lattice}
          paintedCells={floor.paintedCells}
          error={floor.error}
          source={floor.source}
          loading={floor.loading}
        />
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

      <div
        className="volume-render__canvas-wrap plot3d-overlay-host"
        style={sceneAppearance.hostStyle}
        data-streamline-loading={loading || preparation.loading
          || streamlineExtraction.loading || pendingVectorBrickExpansion != null}
        data-streamline-representation={brickedVectorEligible
          ? "trajectory-native-bricks"
          : "bounded-monolithic"}
        data-vector-brick-level={vectorBrickPlan?.level ?? ""}
        data-vector-brick-count={vectorBrickPlan?.items.length ?? 0}
        data-vector-brick-total={vectorBrickPlan?.totalBrickCount ?? 0}
        data-vector-brick-iteration={vectorBrickIteration}
        data-streamline-geometry-committed={streamlineGeometryCommitted}
        data-streamline-integrated-points={streamlineExtraction.result?.integratedPoints ?? 0}
        data-logical-gpu-upload-bytes={gpuUpload.admitted ? gpuUploadBytes : 0}
        data-output-level={outputHierarchySource.activeLevel?.levelId ?? "root"}
        data-output-hierarchy-valid={outputHierarchySource.valid}
      >
        <div
          ref={containerRef}
          className="volume-render__canvas"
          role="img"
          aria-label="Interactive three-dimensional streamlines; drag to orbit, use camera controls for keyboard adjustment"
        />
        <Plot3DAxisLabels labels={axisLabels} />
        <Plot3DScaleBadge
          missingSupport={(preparation.metadata?.sourceNonFiniteCount ?? 0) > 0}
          grid={renderGrid}
          zExag={sceneScale.z}
          fidelityMode={renderFidelityMode}
          samplingProvenance={preparation.metadata?.samplingProvenance}
          detail={(seedMode === "vortex"
            ? (vortexCore
              ? `vortex-ring seeds · |zeta| ${vortexCore.peakVorticityPerS.toFixed(3)} s^-1`
                + ` · ${vortexCore.rotationSign > 0 ? "cyclonic" : "anticyclonic"}`
                + " · bidirectional · cones point with vector"
              : "no near-surface vortex resolved · storm-box seeds")
            : seedMode === "feature"
            ? `${Math.floor(displayedSeedTotal)} resolved-feature seeds · bidirectional · cones point with vector`
            : `${seedTotal} storm-box seeds · bidirectional · cones point with vector`)
            + (streamlineExtraction.result?.representation === "gpu-lines"
              ? " · exact-path GPU line fallback"
              : "")
            + (brickedVectorEligible && vectorBrickPlan
              ? ` · trajectory-loaded native bricks L${vectorBrickPlan.level}`
                + ` · ${vectorBrickPlan.items.length}/${vectorBrickPlan.totalBrickCount}`
                + ` · ${vectorBrickIteration} expansion${vectorBrickIteration === 1 ? "" : "s"}`
              : "")
            + (boundedWindowPlan.ok && boundedWindowPlan.mode === "native-detail"
              ? ` · extreme source scan blocked · ${stormCenterX != null
                ? "tracker-centred" : "metadata-centred"} native ROI`
              : "")
            + (outputHierarchySource.activeLevel?.depth
              ? ` · ${outputHierarchySource.selection?.disclosure ?? outputHierarchySource.activeLevel.levelId}`
              : outputHierarchySource.selection?.fallbackReason
                ? ` · ${outputHierarchySource.selection.fallbackReason}`
                : "")}
        />
        <Plot3DOverlayStack>
          {streamlineGeometryCommitted && colorMode === "magnitude" && colorMaximum > 0 && (
            <Plot3DColorbar
              colormap="Viridis"
              min={0}
              max={colorMaximum}
              label={activeMode.id === "velocity" ? "Speed" : "|ω|"}
              unit={unitFor(
                activeMode.components[0],
                loadedStep?.fieldUnits ?? currentStep?.fieldUnits,
              )}
            />
          )}
          {floor.visible && floorSelection.fieldName && floor.policy && (
            <Plot3DColorbar
              colormap={floor.policy.colormap}
              min={floor.policy.min}
              max={floor.policy.max}
              label={`${floorSelection.fieldName} floor`}
              unit={unitFor(floorSelection.fieldName, loadedStep?.fieldUnits ?? currentStep?.fieldUnits)}
            />
          )}
        </Plot3DOverlayStack>
        <Plot3DCompass orientation={cameraOrientation} />
        {ctxLoss.lost && (
          <div className="volume-render__overlay volume-render__overlay--error">
            3D graphics context lost
            <div className="volume-render__overlay-hint">
              Rebuilding after the GPU dropped WebGL. This clears after the
              selected vector field renders successfully.
            </div>
          </div>
        )}
        {(error || preparation.error || streamlineExtraction.error || gpuUpload.error) && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Vector rendering failed: {error || preparation.error || streamlineExtraction.error || gpuUpload.error}
          </div>
        )}
        {!haveVectors && !loading && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            {activeMode.label} needs{" "}
            <code>{activeMode.components.join(", ")}</code> in this run&apos;s export.
            <div className="volume-render__overlay-hint">
              Missing:{" "}
              {activeMode.components
                .filter((c) => !fieldsForStep.includes(c))
                .join(", ") || "none"}
              .
            </div>
          </div>
        )}
        {streamlineExtraction.result != null && !loading && !preparation.loading
          && !streamlineExtraction.loading && !error && !preparation.error && !streamlineExtraction.error
          && gpuUploadBytes === 0 && renderedStepIndex === loadedVectorFrame?.stepIndex && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            No paths from the current seeds.
            <div className="volume-render__overlay-hint">
              {(preparation.metadata?.sourceNonFiniteCount ?? 0) > 0
                ? "Seeds in missing data are skipped, and paths stop at gaps."
                : "Adjust the seeds or choose a frame with nonzero flow."}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- helpers ---------- */

function streamlineGeometryBytes(geometry: StreamlineGeometry | null): number {
  if (!geometry) return 0;
  return geometry.points.byteLength + geometry.lines.byteLength + geometry.polys.byteLength + geometry.strips.byteLength
    + geometry.normals.byteLength + (geometry.scalars?.byteLength ?? 0);
}

function applyStreamlineGeometry(
  target: ReturnType<typeof vtkPolyData.newInstance>,
  geometry: StreamlineGeometry | null,
): void {
  applyImmutablePolyDataGeometry(target, geometry);
}

/**
 * Build a regular NxNxN grid of seed points inside an axis-aligned box
 * centered at `center` with half-sizes `half`. Returns an interleaved
 * Float32Array (x0,y0,z0, x1,y1,z1, ...) for vtkPoints.setData().
 */
function buildSeedBox(
  center: [number, number, number],
  half: [number, number, number],
  n: number,
): Float32Array {
  const out = new Float32Array(n * n * n * 3);
  let idx = 0;
  for (let i = 0; i < n; i++) {
    const fx = n === 1 ? 0 : (i / (n - 1)) * 2 - 1; // [-1, 1]
    const x = center[0] + fx * half[0];
    for (let j = 0; j < n; j++) {
      const fy = n === 1 ? 0 : (j / (n - 1)) * 2 - 1;
      const y = center[1] + fy * half[1];
      for (let k = 0; k < n; k++) {
        const fz = n === 1 ? 0 : (k / (n - 1)) * 2 - 1;
        const z = center[2] + fz * half[2];
        out[idx++] = x;
        out[idx++] = y;
        out[idx++] = z;
      }
    }
  }
  return out;
}

type FocusBounds = [number, number, number, number, number, number];

function featureSeedBounds(
  points: ArrayLike<number>,
  spacing: [number, number, number],
  scale: { xy: number; z: number },
): FocusBounds | null {
  if (points.length < 3) return null;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (let index = 0; index + 2 < points.length; index += 3) {
    const x = points[index] * scale.xy;
    const y = points[index + 1] * scale.xy;
    const z = points[index + 2] * scale.z;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }
  if (!Number.isFinite(minX)) return null;
  // Seed points sit inside the resulting lines, not at their endpoints. Leave
  // several physical cells around the seed envelope so bidirectional paths do
  // not open cropped while still avoiding a full-domain fit.
  const padX = Math.max((maxX - minX) * 0.7, spacing[0] * scale.xy * 6);
  const padY = Math.max((maxY - minY) * 0.7, spacing[1] * scale.xy * 6);
  const padZ = Math.max((maxZ - minZ) * 0.8, spacing[2] * scale.z * 8);
  return [
    minX - padX,
    maxX + padX,
    minY - padY,
    maxY + padY,
    minZ - padZ,
    maxZ + padZ,
  ];
}
