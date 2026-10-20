import { cloudCoordinateError } from "@/lib/plot/cloudCoordinates";
import { useCommittedLegend } from "./useCommittedLegend";
import { CLOUD_WIND_MAX_COMPRESSED_SOURCE_BYTES, CLOUD_WIND_MAX_SOURCE_BYTES, cloudDensityAvailable, cloudWindUnavailable, planCloudWindWindow, estimateCloudWind } from "@/lib/plot/cloudWind";
import { setVtkResolutionScale } from "@/lib/vtk/vtkInteraction";
import { createCloudMotionController } from "@/lib/vtk/cloudMotionController";
import { volumeImageBytes } from "@/lib/vtk/volumeValidityBudget";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import { configureCoherentVolumeSampling } from "@/lib/vtk/coherentVolumeSampling";
import { bindJointVolumes } from "@/lib/vtk/jointVolumeRendering";
import { coherentSceneGeneration } from "@/lib/vtk/coherentSceneGeneration";
import { holdVtkSceneRendering } from "@/lib/vtk/vtkContextRenderGuard";
import {
  CARTESIAN_VOXEL_CAP_SUPERCELL,
  isCartesianGrid,
} from "@/lib/science/cartesianInterp";
import {
  coalesceNativeFieldWindows,
  fieldSupportsBoundedWindows,
  gridHorizontalCenterM,
  clampGridHorizontalPointM,
  nativeDetailCoversDomain,
  interactiveFullDomainSourceSafe,
  isNativeStormDetailWindowPlan,
  minimumNominalGridSpacingM,
  nativeDetailRecenteringThresholdM,
  planBoundedVolumeWindows,
  planNativeStormDetailWindows,
  planNativeStructuralSupportWindows,
  planBoundedStructuralSupportWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import {
  setPreparedImageDisplayGeometry,
  setPreparedImageScalars,
} from "@/lib/vtk/vtkImageData";
import {
  usePreparedCloudImages,
  usePreparedScalarImage,
} from "@/lib/vtk/useVtkPreparation";
import { unitFor } from "@/lib/science/fieldUnits";
import { describeScalarPlane } from "@/lib/plot/scalarPlanePolicy";
import {
  availablePrecipitatingSpecies,
  sumFields,
} from "@/lib/plot/precipitatingSpecies";
import {
  formatCoreThreshold,
  assessMesocycloneContext,
  mesocycloneEnvelopeThreshold,
  VORTEX_CORE_FIELDS,
  vortexCoreContours,
  vortexCoreDescription,
  vortexCoreLabel,
  vortexCoreReferenceMagnitude,
  vortexCoreSign,
  stableVortexReferenceMagnitude,
  type MesocycloneEnvelopeAssessment,
  type MesocycloneContextAssessment,
  type TornadoVortexAssessment,
} from "@/lib/plot/vortexCore";
import { findPublishedFrame, loadedFramesAreCoherent } from "@/lib/run/frameIdentity";
import { fieldDefinitionIdentity, type FieldLoadWindow } from "@/lib/run/runOutputs";
import type { FieldWindowAggregation } from "@/lib/dataset/nativeFieldWindow";
import { disposeVtkScene } from "@/lib/vtk/vtkLifecycle";
import { useVortexContourExtraction } from "@/lib/vtk/useVortexContourExtraction";
import { useSupercellWorkerOwnedContour } from "@/lib/vtk/useSupercellWorkerOwnedContour";
import { useVtkGpuResourceOwnership } from "@/lib/vtk/useVtkGpuResourceOwnership";
import type {
  VortexContourThresholds,
  VortexSurfaceGeometry,
} from "@/lib/vtk/vortexContourProtocol";
import type { TornadicCondensationAssessment } from "@/lib/plot/tornadicCondensation";
import { releaseVtkPointDataArrays } from "@/lib/vtk/vtkArrayOwnership";
import {
  exposeVtkBackendEvidence,
  useVtkBackend,
  vtkRenderWindowOptions,
} from "@/lib/vtk/vtkBackend";
import { MIN_CONTOUR_VOXEL_BUDGET } from "@/lib/vtk/contourLodBackoff";
import {
  CLOUD_OPACITY_REFERENCE_M,
  cloudDensityForDiagnostics,
  cloudOpacityPoints,
  scaledCloudPeakOpacity,
  scaledPrecipPeakOpacity,
  supercellAuxiliaryVoxelBudget,
  type CloudEdgeMode,
} from "@/lib/plot/supercellRenderQuality";
import { setCameraAdjustedVolumeRayDistances } from "@/lib/vtk/volumeOpticalDepth";
import { coarsestVtkSampling, describeLayerVtkSampling, stableRenderSeed } from "@/lib/plot/renderFidelity";
import { VOLUME_RESOLUTION_POLICY } from "@/lib/vtk/vtkResolution";
import {
  attachVtkInteractionQuality,
  attachVtkCameraStability,
  configureTranslucentRendering,
  attachVtkResize,
  createVtkVolumeQualityController,
  type VtkVolumeQualityController,
} from "@/lib/vtk/vtkInteraction";
import {
  registerVtkAppearanceCapture,
  unregisterVtkCapture,
} from "@/lib/vtk/vtkCaptureRegistry";
import type { VtkAppearanceDescriptor } from "@/lib/vtk/vtkAppearanceEvidence";
import { Icon } from "@/components/shell/Icon";
import {
  ControlGroup,
  PillGroup,
  focusVtkCameraOnBounds,
  refitVtkCameraToBounds,
  unionVtkBounds,
  resetVtkCameraToStormView,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useDebouncedValue,
  useField,
  useFieldResourceOwnership,
  useSceneScale,
  applySceneScale,
  useVtkContextLoss,
  useVtkRunReset,
  SceneScaleControl,
  type PlotComponentProps,
} from "./common";
import { SceneAppearanceControls, useSceneAppearance } from "./SceneAppearanceControls";
import { ThreeDControlScopeToggle } from "./Grouped3DControls";
import { useThreeDControlScope } from "@/lib/plot/threeDControlScope";
import { contrastAdjustedColor } from "@/lib/plot/sceneAppearance";
import {
  cloudSceneColors,
  precipitationSceneColor,
  vortexSceneColors,
  type CloudTone,
  type PrecipitationMode,
  type VortexColorMode,
} from "@/lib/plot/supercellSceneAppearance";
import {
  applySurfaceLighting,
  applyVolumeLighting,
  createVtkSceneLighting,
  type VtkSceneLighting,
} from "@/lib/vtk/sceneLighting";
import { useDomainBox } from "./domainBox";
import { useDomainAxes } from "./useDomainAxes";
import { usePanePresentationBarrier, usePanePresentationSynchronized } from "./panePresentationContext";
import {
  Plot3DAxisLabels,
  Plot3DColorbar,
  Plot3DDiscreteLegend,
  Plot3DOverlayStack,
  Plot3DScaleBadge,
} from "./Plot3DOverlay";
import { Plot3DCompass } from "./Plot3DCompass";
import { useRadarFloorLayer } from "./radarFloorLayer";
import { RadarFloorControls, useRadarFloorSelection } from "./RadarFloorControls";
import { attachVtkPlaybackScreenResolution } from "@/lib/vtk/vtkPlaybackScreenResolution";
import { useVtkPlaybackScreenResolution } from "@/lib/vtk/useVtkPlaybackScreenResolution";
import { useVolumeTextureCapacity } from "@/lib/vtk/useVolumeTextureCapacity";
import {
  CameraElevationControls,
  useCameraElevation,
  useCameraElevationPlacement,
} from "./CameraElevationControls";
import { useCameraOrientation } from "@/lib/vtk/useCameraOrientation";
import { useCameraSync } from "@/lib/vtk/useCameraSync";
import { usePaneContext } from "./paneContext";
import { stormTrackCentre, useStormTrack } from "./useStormTrack";
import { stormTrackMatchesFrame } from "@/lib/run/stormTrack";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { scaleVoxelBudget } from "@/lib/plot/paneLayout";
import { planSupercellSceneBudget, type SceneFieldBudgetInput } from "@/lib/plot/supercellSceneBudget";
import { getVisualizationResourceSnapshot } from "@/lib/core/visualizationResourceGovernor";
import { usePublishedVolumeOverview } from "./usePublishedVolumeOverview";
import { usePlaybackContext } from "./playbackContext";
import "./common.css";
import "./VolumeRender3D.css";

/* Composite Leigh-Orf-style render: aggregate cloud volume, radar floor (textured quad),
 * updraft/downdraft w-isosurfaces, and signed vortex-core isosurfaces — all in one vtk.js renderer. */

import "@kitware/vtk.js/Rendering/Profiles/Volume";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";

// Prefer cloud_condensate (suspended condensate); fallback is total_condensate minus every falling species.
const CLOUD_FIELDS = ["cloud_condensate", "qc", "total_condensate"];
const UPDRAFT_FIELDS = ["w"];
const SUPERCELL_HIERARCHY_FIELDS = ["u", "v", "w"] as const;

const UPDRAFT_GOLD: [number, number, number] = [0.85, 0.64, 0.25];
const TORNADIC_CONDENSATION_SILVER: [number, number, number] = [0.78, 0.84, 0.90];
// Outer surface (downdraft, opposite-sign core) gets 0.7× opacity so the inner surface remains visible.
const SIGNED_PAIR_SECONDARY_OPACITY_RATIO = 0.7;
const DOWNDRAFT_BLUE: [number, number, number] = [0.36, 0.61, 0.92];
/** Volume-native boundary emphasis; no polygon shell participates in depth. */
type CloudEdge = CloudEdgeMode;
type StormVolumeResolution = "storm" | "domain";
type DiagnosticVisibility = "auto" | "unchanged";
type SupercellControlGroup = "domain" | "cloud" | "precip" | "radar" | "flow" | "vortex" | "scene";
const CLOUD_EDGE_OPTIONS: ReadonlyArray<{ value: CloudEdge; label: string }> = [
  { value: "off", label: "Off" },
  { value: "subtle", label: "Subtle" },
  { value: "full", label: "Full" },
];
const STORM_VOLUME_RESOLUTION_OPTIONS: ReadonlyArray<{
  value: StormVolumeResolution;
  label: string;
}> = [
  { value: "storm", label: "Native storm" },
  { value: "domain", label: "Full domain" },
];
const DIAGNOSTIC_VISIBILITY_OPTIONS: ReadonlyArray<{
  value: DiagnosticVisibility;
  label: string;
}> = [
  { value: "auto", label: "Auto reveal" },
  { value: "unchanged", label: "Cloud unchanged" },
];
const CLOUD_TONE_OPTIONS: ReadonlyArray<{ value: CloudTone; label: string }> = [
  { value: "auto", label: "Auto contrast" },
  { value: "white", label: "White" },
  { value: "neutral", label: "Neutral" },
  { value: "storm", label: "Storm gray" },
  { value: "custom", label: "Custom" },
];
const VORTEX_COLOR_OPTIONS: ReadonlyArray<{ value: VortexColorMode; label: string }> = [
  { value: "neutral", label: "Neutral gray" },
  { value: "cloud", label: "Match cloud" },
  { value: "diagnostic", label: "Signed diagnostic" },
];
const PRECIP_OPACITY_REFERENCE_M = 1_000;
const PRECIP_DISPLAY_MAX = 0.005;
/** Fine enough to avoid stair-stepped self-shadow bands at rest. Interaction
 * deliberately widens this spacing because the settled frame replaces it. */
const CLOUD_SHADOW_SAMPLE_FACTOR_FULL = 4;
const CLOUD_SHADOW_SAMPLE_FACTOR_PREVIEW = 8;
/** The full-domain box can put the far clip plane hundreds of kilometres away.
 * vtk.js's 1% default then clips close cloud inspection out of existence. */
const VOLUME_NEAR_CLIPPING_TOLERANCE = 1e-4;
/** Composite surfaces must remain cheap enough to orbit and animate beside a
 * ray-marched cloud. Oversized output backs off its sampling lattice without
 * changing the user's physical contour threshold. */
const TORNADO_LOCATOR_Z_RANGE_M: readonly [number, number] = [0, 1_500];
/** 0.05 g/kg: visible solved condensate without promoting numerical haze. */
const TORNADIC_CONDENSATE_FLOOR_KGKG = 0.05e-3;

function rgbCss(color: readonly [number, number, number]): string {
  const channel = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return `rgb(${channel(color[0])}, ${channel(color[1])}, ${channel(color[2])})`;
}

function scalarContourIsInRange(
  contour: number,
  stats: { min: number; max: number } | null | undefined,
): boolean {
  return !stats || (Number.isFinite(contour) && contour >= stats.min && contour <= stats.max);
}

function pickField(present: string[], prefs: string[]): string | null {
  for (const f of prefs) if (present.includes(f)) return f;
  return null;
}

export function SupercellRender3D({
  runDir,
  steps,
  stepIdx,
  onFrameDisplayChange,
  renderFidelityMode = "scientific",
}: PlotComponentProps) {
  const controlScope = useThreeDControlScope();
  const sceneAppearance = useSceneAppearance(controlScope.scope, "supercell");
  const vtkBackend = useVtkBackend(controlScope.scope, "supercell", "volume");
  const sceneAppearanceRef = useRef(sceneAppearance.resolved);
  sceneAppearanceRef.current = sceneAppearance.resolved;
  const [detailTargetMode, setDetailTargetMode] = useState<"storm" | "view">("storm");
  const [viewDetailTarget, setViewDetailTarget] = useState<{ runDir: string; center: [number, number] } | null>(null);
  const updateViewDetailRef = useRef<() => void>(() => {});
  const viewDetailInteractionRef = useRef({ local: false, remote: false, pointerHeld: false });
  const scheduleViewDetailRef = useRef<(() => void) | null>(null);
  const rootCurrentStep = steps[stepIdx] ?? null;
  const viewDetailSupported = Boolean(rootCurrentStep?.grid) && isCartesianGrid(rootCurrentStep?.grid?.coordinateSystem);
  const activeViewTarget = viewDetailSupported && detailTargetMode === "view" && viewDetailTarget?.runDir === runDir ? viewDetailTarget.center : null;
  useEffect(() => { setDetailTargetMode("storm"); setViewDetailTarget(null); }, [runDir]);
  useEffect(() => { if (!viewDetailSupported) setDetailTargetMode("storm"); }, [viewDetailSupported]);
  const hierarchyRootFrame = useDebouncedFrameIdentity(rootCurrentStep, SCRUB_DEBOUNCE_MS);
  const hierarchyRootPlanningStep = findPublishedFrame(
    steps,
    hierarchyRootFrame.stepIndex,
    hierarchyRootFrame.publicationGeneration,
  ) ?? rootCurrentStep;
  const hierarchyStormTrack = useStormTrack(
    runDir,
    hierarchyRootFrame.stepIndex,
    hierarchyRootPlanningStep?.simulationTimeS ?? null,
    hierarchyRootFrame.publicationGeneration,
  );
  const hierarchyMatchingTrack = stormTrackMatchesFrame(
    hierarchyStormTrack,
    hierarchyRootPlanningStep?.simulationTimeS,
  ) ? hierarchyStormTrack : null;
  const hierarchyStormCenter = stormTrackCentre(hierarchyMatchingTrack);
  const outputHierarchySource = useOutputHierarchySource({
    runDir,
    steps,
    stepIdx,
    scope: controlScope.scope,
    requiredFields: SUPERCELL_HIERARCHY_FIELDS,
    focusPointM: activeViewTarget ?? (hierarchyStormCenter
      ? [hierarchyStormCenter[0], hierarchyStormCenter[1]]
      : null),
  });
  const activeOutputLevel = outputHierarchySource.activeLevel;
  const sourceRunDir = outputHierarchySource.sourceRunDir;
  const sourceSteps = outputHierarchySource.sourceSteps;
  const currentStep = outputHierarchySource.currentStep;
  const hierarchyDiscoveryPending = outputHierarchySource.loading;
  // Stable identity: several memos key off this list, and a fresh [] each render
  // would re-run every one of them on every render.
  const fieldsForStep = useMemo(() => currentStep?.fields ?? [], [currentStep?.fields]);
  const volumeFields = useMemo(
    () => fieldsForPlotRanks(sourceRunDir, currentStep, [3]),
    [currentStep, sourceRunDir],
  );

  const cloudFieldName = useMemo(() => pickField(volumeFields, CLOUD_FIELDS), [volumeFields]);
  /* Radar floor: own 2D hyperslab at native resolution, independent of this scene's voxel cap. */
  const floorSelection = useRadarFloorSelection(
    fieldsForStep,
    controlScope.scope,
    `supercell:${sourceRunDir}`,
  );
  const updraftFieldName = useMemo(() => pickField(volumeFields, UPDRAFT_FIELDS), [volumeFields]);
  const vortexCoreFields = useMemo(
    () => VORTEX_CORE_FIELDS.filter((f) => volumeFields.includes(f)),
    [volumeFields],
  );
  const tornadicCondensationFields = useMemo(() => {
    if (volumeFields.includes("qc")) {
      return ["qc", ...(volumeFields.includes("qi") ? ["qi"] : [])];
    }
    return volumeFields.includes("cloud_condensate") ? ["cloud_condensate"] : [];
  }, [volumeFields]);
  const tornadicCondensationSourceLabel = tornadicCondensationFields[0] === "cloud_condensate"
    ? "solved cloud condensate"
    : "solved qc+qi";

  // Bridge for runs that only export the lumped total_condensate: subtract every
  // falling species so the precip shaft below cloud base isn't rendered as
  // cloud. A run with the dedicated cloud_condensate field needs none of this.
  const availablePrecipSpecies = useMemo(
    () => availablePrecipitatingSpecies(volumeFields),
    [volumeFields],
  );
  const cloudSubtractSpecies = cloudFieldName === "total_condensate"
    ? availablePrecipSpecies
    : [];
  const subtractPrecip = cloudSubtractSpecies.length > 0;

  // Layer toggles + tunables.
  const [showCloud, setShowCloud] = useState(true);
  const [showUpdraft, setShowUpdraft] = useState(false);
  const cloudMotionTimeRef = useRef<HTMLSpanElement>(null);
  const [cloudMotionEnabled, setCloudMotionEnabled] = useState(false);
  const [cloudWindEnabled, setCloudWindEnabled] = useState(false);
  const [cloudMotionActive, setCloudMotionActive] = useState(false);
  const [interpolatedDwellMs, setInterpolatedDwellMs] = useState(0);
  const [cloudMotionReason, setCloudMotionReason] = useState<string | null>(null);
  const [cloudOpacity, setCloudOpacity] = useState(1.0);
  // Cloud visibility floor in g/kg: condensate below this is fully transparent.
  // Default 0.1 g/kg shows the full storm envelope (anvil, fresh cloud), not
  // just the dense core. Stored in g/kg for the UI; converted to kg/kg below.
  const [cloudFloorGkg, setCloudFloorGkg] = useState(0.1);
  // Boundary emphasis is encoded in the volume transfer function. A separate
  // translucent shell clips volume rays in vtk.js and produces a grey skin.
  const [cloudEdge, setCloudEdge] = useState<CloudEdge>("subtle");
  const [cloudTone, setCloudTone] = useState<CloudTone>("auto");
  const [cloudCustomColor, setCloudCustomColor] = useState("#dbe5f2");
  const [cloudBrightness, setCloudBrightness] = useState(0.85);
  const [diagnosticVisibility, setDiagnosticVisibility] =
    useState<DiagnosticVisibility>("auto");
  // Rendered Supercell is storm-focused: spend the volume budget on native
  // cells around the tracked circulation unless the user asks for domain context.
  const [stormVolumeResolution, setStormVolumeResolution] =
    useState<StormVolumeResolution>("storm");
  const [showPrecip, setShowPrecip] = useState(false);
  const [precipMode, setPrecipMode] = useState<PrecipitationMode>("total");
  const [precipOpacity, setPrecipOpacity] = useState(0.75);
  const [precipFloorGkg, setPrecipFloorGkg] = useState(0.05);
  const [precipCustomColor, setPrecipCustomColor] = useState<string | null>(null);
  const [updraftThreshold, setUpdraftThreshold] = useState(6);
  const [updraftOpacity, setUpdraftOpacity] = useState(0.85);
  const [vortexCoreOpacity, setVortexCoreOpacity] = useState(0.8);
  const [vortexColorMode, setVortexColorMode] = useState<VortexColorMode>("neutral");
  const [vortexStructure, setVortexStructure] = useState<"core" | "hierarchy">("hierarchy");
  const [mesocycloneAssessment, setMesocycloneAssessment] =
    useState<MesocycloneEnvelopeAssessment | null>(null);
  const [vortexAssessment, setVortexAssessment] =
    useState<TornadoVortexAssessment | null>(null);
  const [showVortexCore, setShowVortexCore] = useState(false);
  const [showTornadicCondensation, setShowTornadicCondensation] = useState(false);
  const appearanceEvidenceRef = useRef<VtkAppearanceDescriptor>({
    plot: "supercell",
    backend: vtkBackend.backend,
  });
  appearanceEvidenceRef.current = {
    plot: "supercell",
    backend: vtkBackend.backend,
    datasetId: sourceRunDir,
    frameIndex: currentStep?.index ?? null,
    field: cloudFieldName,
    fidelityMode: renderFidelityMode,
    parameters: {
      showCloud,
      cloudInterpolationEnabled: cloudMotionEnabled,
      cloudOpacity,
      cloudFloorGkg,
      cloudEdge,
      stormVolumeResolution,
      detailTargetMode,
      viewDetailTargetM: activeViewTarget,
      showPrecip,
      precipMode,
      precipOpacity,
      showUpdraft,
      updraftThreshold,
      showVortexCore,
      showTornadicCondensation,
      lighting: sceneAppearance.resolved.lighting,
      lightingControls: sceneAppearance.resolved.lightingOptions,
      background: sceneAppearance.resolved.backgroundHex,
    },
  };
  const [tornadicCondensationAssessment, setTornadicCondensationAssessment] =
    useState<TornadicCondensationAssessment | null>(null);
  const [openControlGroup, setOpenControlGroup] = useState<SupercellControlGroup | null>(null);
  const [vortexCoreFieldName, setVortexCoreFieldName] = useState<string | null>(null);
  const vortexSelectionRunRef = useRef(sourceRunDir);
  /* Fraction of the frame's robust signed magnitude to contour at. Expressed
   * as a fraction rather than an absolute value because zeta and p' differ by
   * orders of magnitude and a shared absolute slider would be meaningless. */
  const [vortexCoreFraction, setVortexCoreFraction] = useState(0.45);
  useEffect(() => {
    if (vortexSelectionRunRef.current !== sourceRunDir) {
      vortexSelectionRunRef.current = sourceRunDir;
      setVortexCoreFieldName(vortexCoreFields[0] ?? null);
    } else if (vortexCoreFieldName == null && vortexCoreFields.length > 0) {
      setVortexCoreFieldName(vortexCoreFields[0]);
    }
  }, [sourceRunDir, vortexCoreFieldName, vortexCoreFields]);
  const vortexCoreSourceAvailable = vortexCoreFieldName != null && vortexCoreFields.includes(vortexCoreFieldName);
  useEffect(() => {
    if (showVortexCore && vortexCoreFieldName === "vorticity_z") return;
    setShowTornadicCondensation(false);
  }, [showVortexCore, vortexCoreFieldName]);

  /* Let React paint layer toggles before dispatching preparation/extraction.
   * Disabling remains immediate through the show* guards below. */
  const activateUpdraftRender = useAfterPaintActivation(showUpdraft);
  const activateVortexCoreRender = useAfterPaintActivation(showVortexCore);

  const [resetTick, setResetTick] = useState(0);
  const [qualityPreview, setQualityPreview] = useState(false);
  const [playbackScreenDistance, setPlaybackScreenDistance] = useState(1);
  const [flowContourError, setFlowContourError] = useState<string | null>(null);
  const [vortexContourError, setVortexContourError] = useState<string | null>(null);
  const ctxLoss = useVtkContextLoss();
  const {
    generation: contextGeneration,
    markFrameRendered,
  } = ctxLoss;
  const vtkLifecycleGeneration = contextGeneration * 2
    + (vtkBackend.backend === "webgpu" ? 1 : 0);
  const textureCapacity = useVolumeTextureCapacity(vtkLifecycleGeneration);
  const attachContextLossRef = useRef(ctxLoss.attach);
  attachContextLossRef.current = ctxLoss.attach;
  const pane = usePaneContext();
  const panesSynchronized = usePanePresentationSynchronized();
  const { playing, recording, interpolationDurationMs = 500 } = usePlaybackContext();
  const synchronizedPresentation = playing || recording;
  const cloudMotionAvailable = !panesSynchronized && renderFidelityMode === "presentation" && ctxLoss.available
    && pane.bus == null && pane.voxelBudgetScale >= 1 && showCloud
    && !showPrecip && !showUpdraft && !showVortexCore && !floorSelection.enabled;
  const cloudMotionAllowed = cloudMotionEnabled && cloudMotionAvailable && playing && !recording;
  useEffect(() => {
    if (renderFidelityMode === "presentation" && !panesSynchronized) return;
    setCloudMotionEnabled(false);
    setCloudWindEnabled(false);
    setCloudMotionReason(null);
  }, [renderFidelityMode, panesSynchronized]);
  useEffect(() => {
    if (ctxLoss.lost && cloudMotionEnabled) {
      setCloudMotionEnabled(false);
      setCloudMotionReason("Cloud interpolation unavailable: the graphics context was lost");
    }
  }, [ctxLoss.lost, cloudMotionEnabled]);
  // Debounce the complete publication identity before planning any source windows.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const [renderedStepIndex, setRenderedStepIndex] = useState<number | null>(null);
  const [renderedPublicationGeneration, setRenderedPublicationGeneration] =
    useState<string | null>(null);
  const [committedSceneIdentity, setCommittedSceneIdentity] = useState<string | null>(null);
  const [committedContextGrid, setCommittedContextGrid] = useState<NonNullable<typeof currentStep>["grid"] | null>(null);
  const committedAppearanceRef = useRef<VtkAppearanceDescriptor | null>(null);
  const [committedScaleDisclosure, setCommittedScaleDisclosure] = useState<Omit<ComponentProps<typeof Plot3DScaleBadge>, "zExag"> | null>(null);
  const scaleDisclosureRef = useRef<{ generation: string; value: Omit<ComponentProps<typeof Plot3DScaleBadge>, "zExag"> } | null>(null);
  const committedCameraTargetRef = useRef<[number, number] | null>(null);
  const releaseSceneCommitRef = useRef<(() => void) | null>(null);
  const prepareSceneAxesRef = useRef<(() => void) | null>(null);
  appearanceEvidenceRef.current.frameIndex = renderedStepIndex;
  appearanceEvidenceRef.current.parameters = { ...appearanceEvidenceRef.current.parameters,
    requestedFrameIndex: outputHierarchySource.requestedStep?.index ?? null, renderedPublicationGeneration,
  };
  const requestedStepIndex = outputHierarchySource.requestedStep?.index ?? null;
  const requestedPublicationGeneration = outputHierarchySource.requestedStep?.publicationGeneration ?? null;
  useEffect(() => {
    setRenderedStepIndex(null);
    setRenderedPublicationGeneration(null);
  }, [sourceRunDir]);
  const planningStep = findPublishedFrame(
    sourceSteps,
    loadFrame.stepIndex,
    loadFrame.publicationGeneration,
  ) ?? currentStep;
  /* Split voxel budget across panes; four full-budget volumes exhausts VRAM in a 2x2 grid. */
  const paneVoxelCap = Math.min(textureCapacity.maximumCells,
    scaleVoxelBudget(CARTESIAN_VOXEL_CAP_SUPERCELL, pane.voxelBudgetScale));
  const [viewerResources] = useState(() => getVisualizationResourceSnapshot());
  const hierarchyContextActive = (activeOutputLevel?.depth ?? 0) > 0;
  const contextStep = hierarchyContextActive ? hierarchyRootPlanningStep : planningStep;
  const contextRunDir = hierarchyContextActive ? runDir : sourceRunDir;
  const rootContextVolumeFields = useMemo(
    () => fieldsForPlotRanks(contextRunDir, contextStep, [3]),
    [contextStep, contextRunDir],
  );
  const contextCloudCandidate = pickField(rootContextVolumeFields, ["cloud_condensate", "qc"]);
  const sceneCondensationRequested = showTornadicCondensation && showVortexCore
    && vortexCoreFieldName === "vorticity_z" && tornadicCondensationFields.length > 0;
  const scenePrecipFields = availablePrecipSpecies.filter(field =>
    showCloud && subtractPrecip || showPrecip && (precipMode === "total" || precipMode === field));
  const sceneBudgetFields: SceneFieldBudgetInput[] = [];
  const budgetField = (field: string | null, role: SceneFieldBudgetInput["role"], domain: string = role) => {
    if (!field) return;
    sceneBudgetFields.push({ role, bytesPerCell: planningStep?.grid?.terrainFollowing ? 36 : 4,
      identity: JSON.stringify([sourceRunDir, domain, field,
        fieldDefinitionIdentity(sourceRunDir, planningStep?.index ?? null, field, planningStep?.publicationGeneration ?? null)]),
    });
  };
  if (showCloud) budgetField(cloudFieldName, "detail");
  scenePrecipFields.forEach(field => budgetField(field, "detail"));
  if (showUpdraft) budgetField(updraftFieldName, "auxiliary", "flow");
  if (showVortexCore) budgetField(vortexCoreFieldName, "auxiliary", "vortex");
  if (sceneCondensationRequested) tornadicCondensationFields.forEach(field => budgetField(field, "auxiliary", "vortex"));
  const budgetInput = {
    baseCells: paneVoxelCap, paneScale: pane.voxelBudgetScale,
    systemRamBytes: viewerResources.hardware.systemRamBytes,
    qualifiedCapacityBytes: viewerResources.capacityBytes,
    fields: sceneBudgetFields,
    preparedChannels: {
      detail: Number(showCloud && cloudFieldName != null)
        + Number(showPrecip && scenePrecipFields.length > 0),
      auxiliary: Number(showUpdraft && updraftFieldName != null) + Number(showVortexCore && vortexCoreFieldName != null)
        + 2 * Number(sceneCondensationRequested),
    },
    scratchChannels: { detail: Number(showCloud && subtractPrecip) + Number(scenePrecipFields.length > 1),
      auxiliary: Number(sceneCondensationRequested && tornadicCondensationFields.length > 1) },
  };
  const preliminaryBudget = planSupercellSceneBudget(budgetInput);
  const wholeCloudFitsDetail = useMemo(() => nativeDetailCoversDomain(
    contextStep?.grid, Math.floor(preliminaryBudget.detailCells * 0.75), [28_000, 28_000, 10_000], true,
  ), [contextStep?.grid, preliminaryBudget.detailCells]);
  const publishedContextDeclared = contextCloudCandidate != null && contextStep?.publicationGeneration != null
    && (contextStep?.fieldBrickHierarchies?.[contextCloudCandidate]?.schemaVersion ?? 0) >= 2;
  const cloudContextActive = showCloud && ((cloudMotionEnabled && outputHierarchySource.availableLevels.length > 1) || hierarchyContextActive
      || (stormVolumeResolution === "storm" && !wholeCloudFitsDetail))
    && contextCloudCandidate != null
    && (publishedContextDeclared || interactiveFullDomainSourceSafe(contextStep?.grid)
      && fieldSupportsBoundedWindows(contextStep?.fieldOrigins?.[contextCloudCandidate]));
  const publishedContextActive = cloudContextActive && publishedContextDeclared;
  const rootContextCloudFieldName = cloudContextActive ? contextCloudCandidate : null;
  const sceneBudget = planSupercellSceneBudget({ ...budgetInput,
    fields: cloudContextActive ? [...sceneBudgetFields, {
      role: "context", bytesPerCell: contextStep?.grid?.terrainFollowing ? 36 : 4,
      identity: JSON.stringify([contextRunDir, rootContextCloudFieldName, contextStep?.publicationGeneration, "context"]),
    }] : sceneBudgetFields,
    preparedChannels: { ...budgetInput.preparedChannels, context: Number(cloudContextActive) },
  });
  const contextVoxelCap = sceneBudget.contextCells;
  const detailVoxelCap = sceneBudget.detailCells;
  const rootContextOutcome = useMemo(
    () => cloudContextActive && !publishedContextActive ? planBoundedVolumeWindows(contextStep?.grid, contextVoxelCap) : null,
    [contextVoxelCap, cloudContextActive, publishedContextActive, contextStep?.grid],
  );
  const rootContextWindows = rootContextCloudFieldName && rootContextOutcome?.ok
    ? windowsForFieldAggregation(rootContextOutcome.plan.windows, rootContextCloudFieldName, {
        sourceKind: contextStep?.fieldOrigins?.[rootContextCloudFieldName],
        terrainFollowing: contextStep?.grid?.terrainFollowing,
      })
    : null;
  const terrainBlocked = false;
  const boundedWindowOutcome = useMemo(
    () => planBoundedVolumeWindows(planningStep?.grid, detailVoxelCap),
    [planningStep?.grid, detailVoxelCap],
  );
  const boundedWindows = boundedWindowOutcome.ok ? boundedWindowOutcome.plan.windows : null;
  const fullDomainInteractiveSafe = interactiveFullDomainSourceSafe(planningStep?.grid);
  const forceNativeStormDetail = planningStep != null && !fullDomainInteractiveSafe;
  useEffect(() => {
    if (forceNativeStormDetail && stormVolumeResolution !== "storm") {
      setStormVolumeResolution("storm");
    }
  }, [forceNativeStormDetail, stormVolumeResolution]);
  const boundedWindowsFor = (
    field: string | null,
    windows: readonly FieldLoadWindow[] | null = boundedWindows,
  ) => field && windows
    && fieldSupportsBoundedWindows(planningStep?.fieldOrigins?.[field])
    ? windowsForFieldAggregation(windows, field, {
        sourceKind: planningStep?.fieldOrigins?.[field],
        terrainFollowing: planningStep?.grid?.terrainFollowing,
      })
    : null;

  const matchingStormTrack = stormTrackMatchesFrame(
    hierarchyStormTrack,
    planningStep?.simulationTimeS,
  ) ? hierarchyStormTrack : null;
  const trackedStormCenter = stormTrackCentre(matchingStormTrack);
  const [locatedStormCenter, setLocatedStormCenter] = useState<{
    runDir: string;
    center: [number, number];
  } | null>(null);
  const [locatedFlowCenter, setLocatedFlowCenter] = useState<{
    runDir: string;
    fieldName: string;
    frameIdentity: string;
    center: [number, number];
    supportBounds: [number, number, number, number, number, number] | null;
  } | null>(null);
  const [locatedVortexCenter, setLocatedVortexCenter] = useState<{
    runDir: string;
    fieldName: string;
    frameIdentity: string;
    center: [number, number];
  } | null>(null);
  const diagnosticFrameIdentity = `${planningStep?.publicationGeneration ?? "none"}:${planningStep?.index ?? "none"}`;
  const fallbackStormCenter = locatedStormCenter?.runDir === sourceRunDir
    ? locatedStormCenter.center
    : null;
  const stormDetailCenter: [number, number] | null = activeViewTarget ?? (trackedStormCenter
    ? [trackedStormCenter[0], trackedStormCenter[1]]
    : fallbackStormCenter
      ?? (forceNativeStormDetail ? gridHorizontalCenterM(planningStep?.grid) : null));
  const stormDetailCenterX = stormDetailCenter?.[0] ?? null;
  const stormDetailCenterY = stormDetailCenter?.[1] ?? null;
  const stormDetailOutcome = useMemo(
    () => planNativeStormDetailWindows(
      planningStep?.grid,
      stormDetailCenterX != null && stormDetailCenterY != null
        ? [stormDetailCenterX, stormDetailCenterY]
        : null,
      detailVoxelCap,
      [28_000, 28_000, 10_000],
      undefined,
      true,
    ),
    [planningStep?.grid, stormDetailCenterX, stormDetailCenterY, detailVoxelCap],
  );
  const stormDetailWindows = stormDetailOutcome.ok ? stormDetailOutcome.plan.windows : null;
  const stormRecenteringThresholdM = nativeDetailRecenteringThresholdM(
    stormDetailOutcome.ok ? stormDetailOutcome.plan : null,
  );
  const nativeStormVolumes = (stormVolumeResolution === "storm" || forceNativeStormDetail)
    && stormDetailWindows != null;
  const compositeVolumeWindows = nativeStormVolumes
    ? stormDetailWindows
    : forceNativeStormDetail
      ? null
      : boundedWindows;
  const auxiliaryContourVoxelCap = supercellAuxiliaryVoxelBudget(
    sceneBudget.auxiliaryCells,
    minimumNominalGridSpacingM(planningStep?.grid),
  );
  const supportContourVoxelCap = auxiliaryContourVoxelCap;
  const flowDiagnosticCenter = locatedFlowCenter?.runDir === sourceRunDir
      && locatedFlowCenter.fieldName === updraftFieldName
      && locatedFlowCenter.frameIdentity === diagnosticFrameIdentity
    ? locatedFlowCenter.center
    : null;
  const flowSupportBounds = locatedFlowCenter?.runDir === sourceRunDir
      && locatedFlowCenter.fieldName === updraftFieldName
      && locatedFlowCenter.frameIdentity === diagnosticFrameIdentity
    ? locatedFlowCenter.supportBounds
    : null;
  const vortexDiagnosticCenter = locatedVortexCenter?.runDir === sourceRunDir
      && locatedVortexCenter.fieldName === vortexCoreFieldName
      && locatedVortexCenter.frameIdentity === diagnosticFrameIdentity
    ? locatedVortexCenter.center
    : null;
  const flowPlanningCenter = flowDiagnosticCenter
    ?? (forceNativeStormDetail ? stormDetailCenter : null);
  const vortexPlanningCenter = vortexDiagnosticCenter
    ?? (forceNativeStormDetail ? stormDetailCenter : null);
  /* Each solved-field overlay first performs one inexpensive extrema-preserving
   * domain locator, then reads a native storm-scale lattice around its own
   * dynamic core. A cloud-mass centroid is not a valid proxy: anvils can be
   * kilometres away from the updraft or mesocyclone. */
  const flowDiagnosticOutcome = useMemo(
    () => flowPlanningCenter
      ? flowSupportBounds
        ? (() => {
          const bounds = {
            lower: [flowSupportBounds[0], flowSupportBounds[2], flowSupportBounds[4]] as [number, number, number],
            upper: [flowSupportBounds[1], flowSupportBounds[3], flowSupportBounds[5]] as [number, number, number],
          };
          const native = planNativeStructuralSupportWindows(
            planningStep?.grid,
            bounds,
            supportContourVoxelCap,
          );
          return native.ok ? native : planBoundedStructuralSupportWindows(planningStep?.grid, bounds, supportContourVoxelCap);
        })()
        : planNativeStormDetailWindows(
            planningStep?.grid,
            flowPlanningCenter,
            auxiliaryContourVoxelCap,
          )
      : planBoundedVolumeWindows(planningStep?.grid, MIN_CONTOUR_VOXEL_BUDGET),
    [
      planningStep?.grid,
      flowPlanningCenter,
      flowSupportBounds,
      auxiliaryContourVoxelCap,
      supportContourVoxelCap,
    ],
  );
  const vortexDiagnosticOutcome = useMemo(
    () => vortexPlanningCenter
      ? planNativeStormDetailWindows(
          planningStep?.grid,
          vortexPlanningCenter,
          auxiliaryContourVoxelCap,
        )
      : planBoundedVolumeWindows(planningStep?.grid, MIN_CONTOUR_VOXEL_BUDGET),
    [planningStep?.grid, vortexPlanningCenter, auxiliaryContourVoxelCap],
  );
  const flowRecenteringThresholdM = nativeDetailRecenteringThresholdM(
    flowDiagnosticOutcome.ok && isNativeStormDetailWindowPlan(flowDiagnosticOutcome.plan)
      ? flowDiagnosticOutcome.plan
      : null,
  );
  const vortexRecenteringThresholdM = nativeDetailRecenteringThresholdM(
    vortexDiagnosticOutcome.ok && isNativeStormDetailWindowPlan(vortexDiagnosticOutcome.plan)
      ? vortexDiagnosticOutcome.plan
      : null,
  );
  const diagnosticWindowsFor = (
    outcome: ReturnType<typeof planBoundedVolumeWindows>
      | ReturnType<typeof planNativeStormDetailWindows>,
    field: string | null,
    aggregation: FieldWindowAggregation,
  ) => {
    if (!field || !fieldSupportsBoundedWindows(planningStep?.fieldOrigins?.[field])) return null;
    if (!outcome.ok) return null;
    return coalesceNativeFieldWindows(outcome.plan.windows.map((window) => ({
      ...window,
      aggregation,
    })), {
      sourceKind: planningStep?.fieldOrigins?.[field],
      terrainFollowing: planningStep?.grid?.terrainFollowing,
    });
  };
  const flowUsesExactLattice = flowDiagnosticOutcome.ok
    && flowDiagnosticOutcome.plan.stride.every((stride) => stride === 1)
    && flowDiagnosticCenter != null;
  const flowRenderLatticeAvailable = flowUsesExactLattice || flowSupportBounds != null && flowDiagnosticOutcome.ok;
  const vortexUsesExactLattice = vortexDiagnosticOutcome.ok
    && vortexDiagnosticOutcome.plan.stride.every((stride) => stride === 1)
    && vortexDiagnosticCenter != null;

  const loadStepIndex = !textureCapacity.known || !ctxLoss.available || terrainBlocked || !boundedWindows || hierarchyDiscoveryPending
    ? null
    : loadFrame.stepIndex;
  const cloudWindows = boundedWindowsFor(
    showCloud ? cloudFieldName : null,
    compositeVolumeWindows,
  );
  const cloud = useField(
    sourceRunDir,
    showCloud && cloudWindows ? loadStepIndex : null,
    showCloud ? cloudFieldName : null,
    loadFrame.publicationGeneration,
    cloudWindows,
    { cacheRetention: stormVolumeResolution === "storm" && !nativeStormVolumes ? "transient" : "resident" },
  );
  const rawRootContextCloud = useField(
    contextRunDir,
    textureCapacity.known && ctxLoss.available && showCloud && rootContextWindows ? contextStep?.index ?? null : null,
    showCloud ? rootContextCloudFieldName : null,
    contextStep?.publicationGeneration ?? null,
    rootContextWindows,
  );
  const publishedRootContextCloud = usePublishedVolumeOverview({
    enabled: textureCapacity.known && ctxLoss.available && publishedContextActive, runDir: contextRunDir, step: contextStep,
    fieldName: rootContextCloudFieldName, maxCells: contextVoxelCap,
  });
  const rootContextCloud = publishedContextActive ? publishedRootContextCloud : rawRootContextCloud;
  const windUnavailable = useMemo(() => cloudWindUnavailable(planningStep)
    ?? (!["u", "v", "w"].every(field => volumeFields.includes(field)
      && fieldSupportsBoundedWindows(planningStep?.fieldOrigins?.[field])) ? "Wind fields do not support bounded reads" : null),
  [planningStep, volumeFields]);
  const useWindDensity = cloudDensityAvailable(planningStep) && volumeFields.includes("rho")
    && fieldSupportsBoundedWindows(planningStep?.fieldOrigins?.rho);
  const allWindsCompressed = ["u", "v", "w", ...(useWindDensity ? ["rho"] : [])]
    .every(field => planningStep?.fieldCompression?.[field]?.codec === "zfp");
  const windWindow = useMemo(() => planCloudWindWindow(planningStep?.grid, allWindsCompressed), [planningStep?.grid, allWindsCompressed]);
  const windWindowsFor = (field: string) => windWindow ? [{ ...windWindow,
    maxSourceBytes: planningStep?.grid?.terrainFollowing || planningStep?.fieldCompression?.[field]?.codec === "zfp" ? CLOUD_WIND_MAX_COMPRESSED_SOURCE_BYTES : CLOUD_WIND_MAX_SOURCE_BYTES }] : null;
  const requestCloudWind = renderFidelityMode === "presentation" && cloudMotionEnabled
    && cloudWindEnabled && cloudMotionAvailable && !recording
    && !hierarchyDiscoveryPending && windUnavailable == null && windWindow != null;
  const windU = useField(sourceRunDir, requestCloudWind ? planningStep?.index ?? null : null, requestCloudWind ? "u" : null,
    planningStep?.publicationGeneration ?? null, windWindowsFor("u"), { prefetch: false, cacheRetention: "transient" });
  const windV = useField(sourceRunDir, requestCloudWind ? planningStep?.index ?? null : null, requestCloudWind ? "v" : null,
    planningStep?.publicationGeneration ?? null, windWindowsFor("v"), { prefetch: false, cacheRetention: "transient" });
  const windW = useField(sourceRunDir, requestCloudWind ? planningStep?.index ?? null : null, requestCloudWind ? "w" : null,
    planningStep?.publicationGeneration ?? null, windWindowsFor("w"), { prefetch: false, cacheRetention: "transient" });
  const windRho = useField(sourceRunDir, requestCloudWind && useWindDensity ? planningStep?.index ?? null : null,
    requestCloudWind && useWindDensity ? "rho" : null, planningStep?.publicationGeneration ?? null,
    windWindowsFor("rho"), { prefetch: false, cacheRetention: "transient" });
  const windPending = requestCloudWind && cloudMotionAllowed && (windU.loading || windV.loading || windW.loading || useWindDensity && windRho.loading);
  /* One hook per precip-shaft species: hook count cannot vary. The cloud bridge
   * may need all species, while the visible precipitation layer loads only the
   * selected species (or all species for Total). */
  const loadPrecip = (name: string) => availablePrecipSpecies.includes(name)
    && ((showCloud && subtractPrecip)
      || (showPrecip && (precipMode === "total" || precipMode === name)));
  const rainWindows = boundedWindowsFor(
    loadPrecip("qr") ? "qr" : null,
    compositeVolumeWindows,
  );
  const rain = useField(
    sourceRunDir,
    loadPrecip("qr") && rainWindows ? loadStepIndex : null,
    loadPrecip("qr") ? "qr" : null,
    loadFrame.publicationGeneration,
    rainWindows,
  );
  const graupelWindows = boundedWindowsFor(
    loadPrecip("qg") ? "qg" : null,
    compositeVolumeWindows,
  );
  const graupel = useField(
    sourceRunDir,
    loadPrecip("qg") && graupelWindows ? loadStepIndex : null,
    loadPrecip("qg") ? "qg" : null,
    loadFrame.publicationGeneration,
    graupelWindows,
  );
  const hailWindows = boundedWindowsFor(
    loadPrecip("qh") ? "qh" : null,
    compositeVolumeWindows,
  );
  const hail = useField(
    sourceRunDir,
    loadPrecip("qh") && hailWindows ? loadStepIndex : null,
    loadPrecip("qh") ? "qh" : null,
    loadFrame.publicationGeneration,
    hailWindows,
  );
  const selectedVortexSign = vortexCoreSign(vortexCoreFieldName);
  const vortexCoreWindows = diagnosticWindowsFor(
    vortexDiagnosticOutcome,
    activateVortexCoreRender && vortexCoreSourceAvailable ? vortexCoreFieldName : null,
    vortexUsesExactLattice
      ? "cell_mean"
      : selectedVortexSign > 0 ? "block_max" : "block_min",
  );
  const vortexCoreField = useField(
    sourceRunDir,
    activateVortexCoreRender && vortexCoreWindows ? loadStepIndex : null,
    activateVortexCoreRender && vortexCoreSourceAvailable ? vortexCoreFieldName : null,
    loadFrame.publicationGeneration,
    vortexCoreWindows,
    { prefetch: false },
  );
  const tornadicCondensationEnabled = showTornadicCondensation
    && activateVortexCoreRender
    && vortexCoreSourceAvailable
    && vortexCoreFieldName === "vorticity_z"
    && tornadicCondensationFields.length > 0;
  const primaryCondensateName = tornadicCondensationFields[0] ?? null;
  const frozenCondensateName = tornadicCondensationFields[1] ?? null;
  const primaryCondensateWindows = diagnosticWindowsFor(
    vortexDiagnosticOutcome,
    tornadicCondensationEnabled ? primaryCondensateName : null,
    "cell_mean",
  );
  const primaryCondensate = useField(
    sourceRunDir,
    tornadicCondensationEnabled && primaryCondensateWindows ? loadStepIndex : null,
    tornadicCondensationEnabled ? primaryCondensateName : null,
    loadFrame.publicationGeneration,
    primaryCondensateWindows,
    { prefetch: false },
  );
  const frozenCondensateWindows = diagnosticWindowsFor(
    vortexDiagnosticOutcome,
    tornadicCondensationEnabled ? frozenCondensateName : null,
    "cell_mean",
  );
  const frozenCondensate = useField(
    sourceRunDir,
    tornadicCondensationEnabled && frozenCondensateWindows ? loadStepIndex : null,
    tornadicCondensationEnabled ? frozenCondensateName : null,
    loadFrame.publicationGeneration,
    frozenCondensateWindows,
    { prefetch: false },
  );
  const updraftWindows = diagnosticWindowsFor(
    flowDiagnosticOutcome,
    activateUpdraftRender ? updraftFieldName : null,
    flowUsesExactLattice || flowSupportBounds != null ? "cell_mean" : "block_max",
  );
  const updraft = useField(
    sourceRunDir,
    activateUpdraftRender && updraftWindows ? loadStepIndex : null,
    activateUpdraftRender ? updraftFieldName : null,
    loadFrame.publicationGeneration,
    updraftWindows,
    { prefetch: false },
  );
  const sourceDefinitionIdentity = JSON.stringify([
    ...[cloudFieldName, updraftFieldName, vortexCoreFieldName, primaryCondensateName, frozenCondensateName,
      "qr", "qg", "qh", "u", "v", "w", "rho"].map(field => fieldDefinitionIdentity(
      sourceRunDir, currentStep?.index ?? null, field, currentStep?.publicationGeneration ?? null)),
    fieldDefinitionIdentity(contextRunDir, contextStep?.index ?? null, rootContextCloudFieldName,
      contextStep?.publicationGeneration ?? null),
    fieldDefinitionIdentity(sourceRunDir, currentStep?.index ?? null, floorSelection.fieldName,
      currentStep?.publicationGeneration ?? null),
  ]);
  // Base-volume coherence is deliberately independent from auxiliary surfaces.
  // Toggling Flow or Vortex must not null the render grid and re-run a cloud
  // preparation that already completed for this exact frame.
  const activeLoads = [
    ...(showCloud && cloudFieldName && cloudWindows ? [cloud] : []),
    ...(showCloud && rootContextCloudFieldName && (rootContextWindows || publishedContextActive) ? [rootContextCloud] : []),
    ...([["qr", rain, rainWindows], ["qg", graupel, graupelWindows], ["qh", hail, hailWindows]] as const)
      .filter(([name, , windows]) => loadPrecip(name) && windows)
      .map(([, load]) => load),
  ];
  const boundedPlanError = planningStep && !terrainBlocked && !boundedWindowOutcome.ok
    ? boundedWindowOutcome.reason
    : null;
  const flowDiagnosticPlanError = planningStep && !terrainBlocked && !flowDiagnosticOutcome.ok
    ? flowDiagnosticOutcome.reason
    : null;
  const vortexDiagnosticPlanError = planningStep && !terrainBlocked && !vortexDiagnosticOutcome.ok
    ? vortexDiagnosticOutcome.reason
    : null;
  const requestedWindowFields = [
    showCloud ? cloudFieldName : null,
    loadPrecip("qr") ? "qr" : null,
    loadPrecip("qg") ? "qg" : null,
    loadPrecip("qh") ? "qh" : null,
    activateVortexCoreRender ? vortexCoreFieldName : null,
    tornadicCondensationEnabled ? primaryCondensateName : null,
    tornadicCondensationEnabled ? frozenCondensateName : null,
    activateUpdraftRender ? updraftFieldName : null,
  ].filter((field): field is string => field != null);
  const unsupportedWindowFields = [...new Set(requestedWindowFields.filter((field) =>
    !fieldSupportsBoundedWindows(planningStep?.fieldOrigins?.[field])
  ))];
  const sourceLoads = activeLoads.filter(load => load !== rootContextCloud);
  const coherentFrame = loadedFramesAreCoherent(...sourceLoads);
  const baseLoadedStep = coherentFrame && sourceLoads.length > 0
    ? findPublishedFrame(
        sourceSteps,
        sourceLoads[0].loadedStepIndex,
        sourceLoads[0].loadedPublicationGeneration,
      )
    : null;
  const diagnosticLoadedStep = findPublishedFrame(
    sourceSteps,
    vortexCoreField.loadedStepIndex ?? updraft.loadedStepIndex,
    vortexCoreField.loadedStepIndex != null
      ? vortexCoreField.loadedPublicationGeneration
      : updraft.loadedPublicationGeneration,
  );
  const loadedStep = baseLoadedStep ?? diagnosticLoadedStep ?? planningStep;
  const renderGrid = loadedStep?.grid ?? planningStep?.grid ?? null;
  // Child scalars are already Earth-registered. Keep the root box/axes as
  // environmental context while the fine child owns the storm data.
  const contextGrid = activeOutputLevel?.depth
    ? rootCurrentStep?.grid ?? renderGrid
    : renderGrid;
  const loadMatchesRenderedStep = (load: typeof updraft): boolean => Boolean(
    loadedStep
    && load.loadedStepIndex === loadedStep.index
    && load.loadedPublicationGeneration === loadedStep.publicationGeneration,
  );
  const loadMatchesRequestedStep = (load: typeof updraft): boolean => Boolean(
    planningStep && !load.loading && load.error == null && load.field != null
    && load.loadedStepIndex === planningStep.index
    && load.loadedPublicationGeneration === planningStep.publicationGeneration,
  );
  const flowFieldReady = flowRenderLatticeAvailable && loadMatchesRenderedStep(updraft);
  const vortexFieldReady = vortexUsesExactLattice && loadMatchesRenderedStep(vortexCoreField);
  const condensateLoadsReady = !tornadicCondensationEnabled || (
    loadMatchesRenderedStep(primaryCondensate)
    && (!frozenCondensateName || loadMatchesRenderedStep(frozenCondensate))
  );
  // True aspect ratio (no vertical exaggeration by default).
  const [sceneScale, setSceneScale] = useSceneScale(contextGrid, undefined, controlScope.scope, "supercell");
  const sceneScaleRef = useRef(sceneScale);
  sceneScaleRef.current = sceneScale;
  const diagnosticRevealActive = diagnosticVisibility === "auto"
    && showCloud && (showUpdraft || showVortexCore);
  const effectiveCloudOpacity = cloudDensityForDiagnostics(
    cloudOpacity,
    diagnosticRevealActive,
  );
  const cloudOpacityRef = useRef(effectiveCloudOpacity);
  cloudOpacityRef.current = effectiveCloudOpacity;
  const cloudEdgeRef = useRef(cloudEdge);
  cloudEdgeRef.current = cloudEdge;
  const cloudVisualRef = useRef({
    floorGkg: cloudFloorGkg,
    tone: cloudTone,
    customColor: cloudCustomColor,
    brightness: cloudBrightness,
  });
  cloudVisualRef.current = {
    floorGkg: cloudFloorGkg,
    tone: cloudTone,
    customColor: cloudCustomColor,
    brightness: cloudBrightness,
  };
  const precipVisualRef = useRef({
    opacity: precipOpacity,
    floorGkg: precipFloorGkg,
    mode: precipMode,
    customColor: precipCustomColor,
  });
  precipVisualRef.current = {
    opacity: precipOpacity,
    floorGkg: precipFloorGkg,
    mode: precipMode,
    customColor: precipCustomColor,
  };
  /* All falling species as one subtrahend; refuses on a shape disagreement
   * rather than subtracting one cell's precip from another cell's cloud. */
  const precipTotal = useMemo(
    () => ((subtractPrecip || (showPrecip && precipMode === "total")) && coherentFrame
      ? sumFields([rain.field, graupel.field, hail.field])
      : null),
    [
      subtractPrecip,
      showPrecip,
      precipMode,
      coherentFrame,
      rain.field,
      graupel.field,
      hail.field,
    ],
  );
  const precipRenderField = useMemo(() => {
    if (!showPrecip || !coherentFrame) return null;
    if (precipMode === "total") return precipTotal;
    return precipMode === "qr"
      ? rain.field
      : precipMode === "qg"
        ? graupel.field
        : hail.field;
  }, [
    showPrecip,
    coherentFrame,
    precipMode,
    rain.field,
    graupel.field,
    hail.field,
    precipTotal,
  ]);
  const tornadicCondensateField = useMemo(() => {
    if (!tornadicCondensationEnabled || !condensateLoadsReady) return null;
    return sumFields([
      primaryCondensate.field,
      frozenCondensateName ? frozenCondensate.field : null,
    ]);
  }, [
    tornadicCondensationEnabled,
    condensateLoadsReady,
    primaryCondensate.field,
    frozenCondensateName,
    frozenCondensate.field,
  ]);
  const derivedResourceGeneration = `${loadedStep?.publicationGeneration ?? "none"}:${loadedStep?.index ?? "none"}`;
  useFieldResourceOwnership(
    `${controlScope.scope}:supercell-precip-total`,
    precipTotal,
    derivedResourceGeneration,
  );
  useFieldResourceOwnership(
    `${controlScope.scope}:supercell-tornadic-condensate`,
    tornadicCondensateField,
    derivedResourceGeneration,
  );
  const cloudPreparation = usePreparedCloudImages(
    showCloud && coherentFrame ? cloud.field : null,
    subtractPrecip ? precipTotal : null,
    renderGrid,
    detailVoxelCap,
    detailVoxelCap,
    ctxLoss.generation,
    {
      fidelityMode: renderFidelityMode,
      detailSeed: stableRenderSeed(runDir),
      simulationTimeS: loadedStep?.simulationTimeS ?? 0,
    },
  );
  const rootContextPreparation = usePreparedCloudImages(
    showCloud && cloudContextActive ? rootContextCloud.field : null,
    null,
    contextStep?.grid ?? null,
    contextVoxelCap,
    contextVoxelCap,
    ctxLoss.generation,
    {
      fidelityMode: "scientific",
      detailSeed: stableRenderSeed(runDir),
      simulationTimeS: contextStep?.simulationTimeS ?? 0,
    },
  );
  const activeRootContextPreparedRef = useRef(rootContextPreparation.prepared?.volume ?? null);
  const precipPreparation = usePreparedScalarImage(
    precipRenderField,
    renderGrid,
    detailVoxelCap,
    ctxLoss.generation,
  );
  const activePrecipPreparedRef = useRef(precipPreparation.prepared);
  const activeCloudVolumePrepared = renderFidelityMode === "presentation"
    ? cloudPreparation.prepared?.surface ?? null
    : cloudPreparation.prepared?.volume ?? null;
  const activeCloudVolumePreparedRef = useRef(activeCloudVolumePrepared);
  const vortexCorePreparation = usePreparedScalarImage(
    activateVortexCoreRender && loadMatchesRenderedStep(vortexCoreField)
      ? vortexCoreField.field
      : null,
    renderGrid,
    auxiliaryContourVoxelCap,
    ctxLoss.generation,
    {
      featureSign: selectedVortexSign,
      featureZRangeM: TORNADO_LOCATOR_Z_RANGE_M,
      geometryHandoff: true,
      stableWorkerLane: `supercell-vortex:${sourceRunDir}:${loadFrame.publicationGeneration ?? "none"}:${loadFrame.stepIndex ?? "none"}:${vortexCoreFieldName ?? "none"}`,
    },
  );
  const tornadicCondensatePreparation = usePreparedScalarImage(
    tornadicCondensationEnabled && condensateLoadsReady
      ? tornadicCondensateField
      : null,
    renderGrid,
    auxiliaryContourVoxelCap,
    ctxLoss.generation,
    {
      geometryHandoff: true,
      stableWorkerLane: `supercell-condensate:${sourceRunDir}:${loadFrame.publicationGeneration ?? "none"}:${loadFrame.stepIndex ?? "none"}:${primaryCondensateName ?? "none"}:${frozenCondensateName ?? "none"}`,
    },
  );
  const updraftPreparation = usePreparedScalarImage(
    activateUpdraftRender && loadMatchesRenderedStep(updraft) ? updraft.field : null,
    renderGrid,
    supportContourVoxelCap,
    ctxLoss.generation,
    {
      featureSign: 1,
      geometryHandoff: true,
      stableWorkerLane: `supercell-flow:${sourceRunDir}:${loadFrame.publicationGeneration ?? "none"}:${loadFrame.stepIndex ?? "none"}:${updraftFieldName ?? "none"}`,
    },
  );
  const flowPreparationHandleRef = useRef<number | null>(null);
  const flowPreparationCommitsRef = useRef(0);
  const flowRecentersRef = useRef(0);
  const lastFlowPeakRef = useRef<[number, number, number] | null>(null);
  if (updraftPreparation.handle?.id !== flowPreparationHandleRef.current) {
    flowPreparationHandleRef.current = updraftPreparation.handle?.id ?? null;
    if (updraftPreparation.handle) flowPreparationCommitsRef.current += 1;
  }
  /* Cloud presentation follows the tracker when available and otherwise uses
   * suspended condensate mass. It deliberately does not borrow a diagnostic
   * extremum: cloud framing and solved-field feature location are separate
   * scientific questions. */
  useEffect(() => {
    if (trackedStormCenter || loadFrame.stepIndex == null) return;
    const locatorLoad = !hierarchyContextActive && cloudContextActive ? rootContextCloud : cloud;
    const locatorStep = !hierarchyContextActive && cloudContextActive ? contextStep : planningStep;
    if (!locatorStep || locatorLoad.loading || locatorLoad.loadedStepIndex !== locatorStep.index
        || locatorLoad.loadedPublicationGeneration !== locatorStep.publicationGeneration) return;
    const locator = !hierarchyContextActive && cloudContextActive
      ? rootContextPreparation.prepared
      : !nativeStormVolumes ? cloudPreparation.prepared : null;
    const peak = showCloud ? locator?.cloudCenter : null;
    if (!peak || !Number.isFinite(peak[0]) || !Number.isFinite(peak[1])) return;
    const next: [number, number] = [peak[0], peak[1]];
    setLocatedStormCenter((previous) => {
      if (previous?.runDir === sourceRunDir
          && Math.hypot(
            previous.center[0] - next[0],
            previous.center[1] - next[1],
          ) < stormRecenteringThresholdM) {
        return previous;
      }
      return { runDir: sourceRunDir, center: next };
    });
  }, [
    trackedStormCenter,
    loadFrame.stepIndex,
    sourceRunDir,
    showCloud,
    cloudPreparation.prepared?.cloudCenter,
    rootContextPreparation.prepared,
    hierarchyContextActive,
    cloudContextActive,
    nativeStormVolumes,
    stormRecenteringThresholdM,
    cloud.loading, cloud.loadedStepIndex, cloud.loadedPublicationGeneration,
    rootContextCloud.loading, rootContextCloud.loadedStepIndex, rootContextCloud.loadedPublicationGeneration,
  ]);
  /* Flow and vortex own independent extrema locators. Their first coarse read
   * preserves the selected sign across the full domain; subsequent reads are
   * native and recenter only after meaningful physical drift. */
  useEffect(() => {
    if (!activateUpdraftRender || !updraftFieldName || loadFrame.stepIndex == null) return;
    if (!loadMatchesRequestedStep(updraft)) return;
    const peak = updraftPreparation.metadata?.featurePeakPosition;
    if (!peak || !Number.isFinite(peak[0]) || !Number.isFinite(peak[1])) return;
    lastFlowPeakRef.current = peak;
    const next: [number, number] = [peak[0], peak[1]];
    const supportBounds = updraftPreparation.metadata?.absoluteFocusBounds ?? null;
    setLocatedFlowCenter((previous) => {
      if (previous?.runDir === sourceRunDir && previous.fieldName === updraftFieldName
          && previous.frameIdentity === diagnosticFrameIdentity
          && Math.hypot(
            previous.center[0] - next[0],
            previous.center[1] - next[1],
          ) < flowRecenteringThresholdM) {
        return previous;
      }
      flowRecentersRef.current += 1;
      return {
        runDir: sourceRunDir,
        fieldName: updraftFieldName,
        frameIdentity: diagnosticFrameIdentity,
        center: next,
        supportBounds,
      };
    });
  }, [
    activateUpdraftRender,
    updraftFieldName,
    loadFrame.stepIndex,
    sourceRunDir,
    diagnosticFrameIdentity,
    updraftPreparation.metadata?.featurePeakPosition,
    updraftPreparation.metadata?.absoluteFocusBounds,
    updraft.loading, updraft.loadedStepIndex, updraft.loadedPublicationGeneration,
    flowRecenteringThresholdM,
  ]);
  useEffect(() => {
    if (!activateVortexCoreRender || !vortexCoreFieldName || loadFrame.stepIndex == null) return;
    if (!loadMatchesRequestedStep(vortexCoreField)) return;
    const peak = vortexCorePreparation.metadata?.featurePeakPosition;
    if (!peak || !Number.isFinite(peak[0]) || !Number.isFinite(peak[1])) return;
    const next: [number, number] = [peak[0], peak[1]];
    setLocatedVortexCenter((previous) => {
      if (previous?.runDir === sourceRunDir && previous.fieldName === vortexCoreFieldName
          && previous.frameIdentity === diagnosticFrameIdentity
          && Math.hypot(
            previous.center[0] - next[0],
            previous.center[1] - next[1],
          ) < vortexRecenteringThresholdM) {
        return previous;
      }
      return {
        runDir: sourceRunDir,
        fieldName: vortexCoreFieldName,
        frameIdentity: diagnosticFrameIdentity,
        center: next,
      };
    });
  }, [
    activateVortexCoreRender,
    vortexCoreFieldName,
    loadFrame.stepIndex,
    sourceRunDir,
    diagnosticFrameIdentity,
    vortexCorePreparation.metadata?.featurePeakPosition,
    vortexRecenteringThresholdM,
    vortexCoreField.loading, vortexCoreField.loadedStepIndex, vortexCoreField.loadedPublicationGeneration,
  ]);
  const committedCloudFloorGkg = useDebouncedValue(cloudFloorGkg, 180);
  const committedUpdraftThreshold = useDebouncedValue(updraftThreshold, 160);
  const committedVortexCoreFraction = useDebouncedValue(vortexCoreFraction, 160);
  const flowPrepared = updraftPreparation.metadata;
  const flowThreshold = Math.max(0.5, committedUpdraftThreshold);
  const updraftSurfaceAvailable = scalarContourIsInRange(flowThreshold, flowPrepared?.stats);
  const downdraftSurfaceAvailable = scalarContourIsInRange(
    -flowThreshold,
    flowPrepared?.stats,
  );
  const flowExtraction = useVortexContourExtraction({
    // `enabled` expresses layer intent. A null prepared image is the normal
    // handoff between frames and must put the warm worker on hold, not stop it.
    enabled: showUpdraft && flowRenderLatticeAvailable,
    preparedHandle: updraftPreparation.handle,
    thresholds: {
      core: updraftSurfaceAvailable ? flowThreshold : null,
      opposite: downdraftSurfaceAvailable ? -flowThreshold : null,
      mesocyclone: null,
    },
  });
  const flowExtractionLoading = flowExtraction.loading;
  useEffect(() => {
    if (flowExtractionLoading) setFlowContourError(null);
  }, [flowExtractionLoading]);
  // Fraction of the selected sign's resolved peak: compact vortices can occupy
  // far less than 2% of a mesoscale ROI, making a global p98 an unusably low ceiling.
  const instantaneousVortexReference = useMemo(() => {
    const featurePeak = vortexCorePreparation.metadata?.featurePeakValue;
    if (featurePeak != null && Number.isFinite(featurePeak)) {
      const magnitude = Math.abs(featurePeak);
      if (magnitude > 0) return magnitude;
    }
    const stats = vortexCorePreparation.metadata?.stats;
    if (!stats) return null;
    const magnitude = vortexCoreReferenceMagnitude(vortexCoreFieldName, stats);
    if (!Number.isFinite(magnitude) || magnitude <= 0) return null;
    return magnitude;
  }, [
    vortexCorePreparation.metadata?.featurePeakValue,
    vortexCorePreparation.metadata?.stats,
    vortexCoreFieldName,
  ]);
  const vortexReferenceRef = useRef<{
    key: string;
    magnitude: number;
  } | null>(null);
  const vortexReferenceKey = `${sourceRunDir}\u0000${vortexCoreFieldName ?? ""}`;
  const retainedVortexReference = vortexReferenceRef.current?.key === vortexReferenceKey
    ? vortexReferenceRef.current.magnitude
    : null;
  const vortexReferenceMagnitude = stableVortexReferenceMagnitude(
    instantaneousVortexReference,
    retainedVortexReference,
    synchronizedPresentation,
  );
  if (
    vortexReferenceMagnitude != null
    && (!synchronizedPresentation || retainedVortexReference == null)
  ) {
    vortexReferenceRef.current = {
      key: vortexReferenceKey,
      magnitude: vortexReferenceMagnitude,
    };
  }
  const vortexCoreThreshold = vortexReferenceMagnitude == null
    ? null
    : vortexReferenceMagnitude * committedVortexCoreFraction;
  const mesocycloneThreshold = vortexStructure === "hierarchy" && vortexCoreThreshold != null
    ? mesocycloneEnvelopeThreshold(vortexCoreFieldName, vortexCoreThreshold)
    : null;
  const requestedVortexContours = useMemo(
    () => vortexCoreThreshold == null
      ? null
      : vortexCoreContours(vortexCoreFieldName, vortexCoreThreshold),
    [vortexCoreFieldName, vortexCoreThreshold],
  );
  const vortexPrepared = vortexCorePreparation.metadata;
  const mesocycloneContext = assessMesocycloneContext(
    vortexPrepared?.shape,
    vortexPrepared?.spacing,
  );
  const vortexCoreAvailable = requestedVortexContours != null && scalarContourIsInRange(
    requestedVortexContours.core,
    vortexPrepared?.stats,
  );
  const vortexOppositeAvailable = requestedVortexContours != null && scalarContourIsInRange(
    requestedVortexContours.opposite,
    vortexPrepared?.stats,
  );
  const vortexMesocycloneAvailable = vortexCoreAvailable
    && mesocycloneContext.qualified
    && mesocycloneThreshold != null
    && scalarContourIsInRange(mesocycloneThreshold, vortexPrepared?.stats);
  const vortexContourExpected = vortexCoreAvailable
    || vortexOppositeAvailable
    || vortexMesocycloneAvailable;
  const vortexExtractionGeneration = JSON.stringify([
    sourceRunDir,
    sourceDefinitionIdentity,
    loadFrame.publicationGeneration,
    loadFrame.stepIndex,
    vortexCoreFieldName,
    vortexCorePreparation.metadata?.shape ?? null,
    vortexCorePreparation.metadata?.spacing ?? null,
    vortexCorePreparation.metadata?.origin ?? null,
    tornadicCondensationEnabled ? tornadicCondensatePreparation.metadata?.shape ?? null : null,
    tornadicCondensationEnabled ? tornadicCondensatePreparation.metadata?.spacing ?? null : null,
    tornadicCondensationEnabled ? tornadicCondensatePreparation.metadata?.origin ?? null : null,
    vortexCoreAvailable ? requestedVortexContours?.core ?? null : null,
    vortexOppositeAvailable ? requestedVortexContours?.opposite ?? null : null,
    vortexMesocycloneAvailable ? mesocycloneThreshold : null,
    vortexPrepared?.featurePeakPosition ?? null,
    tornadicCondensationEnabled ? TORNADIC_CONDENSATE_FLOOR_KGKG : null,
  ]);
  const committedVortexExtractionGeneration = useDebouncedValue(
    vortexExtractionGeneration,
    200,
  );
  const vortexExtractionGenerationReady =
    committedVortexExtractionGeneration === vortexExtractionGeneration;
  const vortexExtractionPublicationKey = JSON.stringify([
    sourceRunDir,
    sourceDefinitionIdentity,
    loadFrame.publicationGeneration,
    loadFrame.stepIndex,
    vortexCoreFieldName,
    committedVortexCoreFraction,
    vortexStructure,
    tornadicCondensationEnabled,
    vortexCoreWindows,
    primaryCondensateWindows,
    frozenCondensateWindows,
  ]);
  const [committedVortexExtraction, setCommittedVortexExtraction] = useState<{
    generation: string;
    publicationKey: string;
    preparedHandle: typeof vortexCorePreparation.handle;
    condensateHandle: typeof tornadicCondensatePreparation.handle;
    thresholds: VortexContourThresholds;
    anchor: readonly [number, number, number] | null;
    condensateFloor: number | null;
  } | null>(null);
  useEffect(() => {
    setCommittedVortexExtraction(null);
  }, [
    sourceRunDir,
    loadFrame.publicationGeneration,
    loadFrame.stepIndex,
    vortexCoreFieldName,
    sourceDefinitionIdentity,
  ]);
  useEffect(() => {
    if (!showVortexCore || !vortexExtractionGenerationReady
        || !vortexCorePreparation.handle
        || !vortexUsesExactLattice
        || !vortexContourExpected
        || (tornadicCondensationEnabled && !tornadicCondensatePreparation.handle)) {
      return;
    }
    setCommittedVortexExtraction((previous) =>
      previous?.publicationKey === vortexExtractionPublicationKey ? previous : {
        generation: vortexExtractionGeneration,
        publicationKey: vortexExtractionPublicationKey,
        preparedHandle: vortexCorePreparation.handle,
        condensateHandle: tornadicCondensationEnabled
          ? tornadicCondensatePreparation.handle
          : null,
        thresholds: {
          core: vortexCoreAvailable ? requestedVortexContours?.core ?? null : null,
          opposite: vortexOppositeAvailable ? requestedVortexContours?.opposite ?? null : null,
          mesocyclone: vortexMesocycloneAvailable ? mesocycloneThreshold : null,
        },
        anchor: vortexPrepared?.featurePeakPosition ?? null,
        condensateFloor: tornadicCondensationEnabled
          ? TORNADIC_CONDENSATE_FLOOR_KGKG
          : null,
      }
    );
  }, [
    showVortexCore,
    vortexExtractionGenerationReady,
    vortexExtractionGeneration,
    vortexExtractionPublicationKey,
    vortexCorePreparation.handle,
    vortexUsesExactLattice,
    vortexContourExpected,
    tornadicCondensationEnabled,
    tornadicCondensatePreparation.handle,
    vortexCoreAvailable,
    vortexOppositeAvailable,
    vortexMesocycloneAvailable,
    requestedVortexContours,
    mesocycloneThreshold,
    vortexPrepared,
    vortexPrepared?.featurePeakPosition,
  ]);
  const vortexExtractionEnabled = showVortexCore && committedVortexExtraction != null
    && committedVortexExtraction.publicationKey === vortexExtractionPublicationKey;
  const workerOwnedVortexRequest = useMemo(
    () => committedVortexExtraction?.preparedHandle?.residency === "worker"
      && (!committedVortexExtraction.condensateHandle
        || committedVortexExtraction.condensateHandle.residency === "worker")
      ? {
          ...committedVortexExtraction,
          preparedHandle: committedVortexExtraction.preparedHandle,
        }
      : null,
    [committedVortexExtraction],
  );
  const workerOwnedVortexExtraction = useSupercellWorkerOwnedContour(
    vortexExtractionEnabled && workerOwnedVortexRequest != null,
    workerOwnedVortexRequest,
  );
  const mainVortexExtraction = useVortexContourExtraction({
    enabled: vortexExtractionEnabled && workerOwnedVortexRequest == null,
    preparedHandle: committedVortexExtraction?.preparedHandle ?? null,
    thresholds: committedVortexExtraction?.thresholds ?? {
      core: null,
      opposite: null,
      mesocyclone: null,
    },
    anchor: committedVortexExtraction?.anchor ?? null,
    condensateHandle: committedVortexExtraction?.condensateHandle ?? null,
    condensateFloor: committedVortexExtraction?.condensateFloor ?? null,
  });
  const vortexExtraction = workerOwnedVortexRequest
    ? workerOwnedVortexExtraction
    : mainVortexExtraction;
  const vortexExtractionLoading = vortexExtraction.loading;
  const activeCondensationPreparedRef = useRef(vortexExtraction.result?.condensation ?? null);
  useEffect(() => {
    if (vortexExtractionLoading) setVortexContourError(null);
  }, [vortexExtractionLoading]);
  const flowLayerLoading = showUpdraft && updraftFieldName != null && flowDiagnosticPlanError == null
    && updraftPreparation.error == null && flowExtraction.error == null && (
      !activateUpdraftRender
      || updraft.loading
      || !loadMatchesRenderedStep(updraft)
      || ((updraftSurfaceAvailable || downdraftSurfaceAvailable) && !flowRenderLatticeAvailable)
      || updraftPreparation.loading
      || flowPrepared == null
      || flowExtractionLoading
      || ((updraftSurfaceAvailable || downdraftSurfaceAvailable)
        && flowExtraction.result == null)
    );
  const vortexLayerLoading = showVortexCore && vortexCoreFieldName != null
    && vortexCoreFields.includes(vortexCoreFieldName) && vortexDiagnosticPlanError == null
    && vortexCorePreparation.error == null && vortexExtraction.error == null && (
      !activateVortexCoreRender
      || vortexCoreField.loading
      || !loadMatchesRenderedStep(vortexCoreField)
      || (vortexContourExpected && !vortexUsesExactLattice)
      || vortexCorePreparation.loading
      || (tornadicCondensationEnabled && (
        primaryCondensate.loading
        || (frozenCondensateName != null && frozenCondensate.loading)
        || tornadicCondensatePreparation.loading
      ))
      || vortexPrepared == null
      || vortexExtractionLoading
      || (vortexContourExpected && !vortexExtractionEnabled)
      || (vortexContourExpected && vortexExtraction.result == null)
    );

  // Same 4-order gap as above: without the unit, a bare threshold cannot be read as either.
  const vortexCoreUnit = vortexCoreFieldName
    ? unitFor(vortexCoreFieldName, loadedStep?.fieldUnits)
    : null;
  const withVortexCoreUnit = (text: string): string =>
    (vortexCoreUnit ? `${text} ${vortexCoreUnit}` : text);
  const tornadoScaleResolved = vortexCoreFieldName === "vorticity_z"
    && vortexAssessment?.present === true;
  const vortexSurfaceColors = useMemo(() => vortexSceneColors(
    vortexColorMode,
    cloudSceneColors(
      cloudTone,
      cloudCustomColor,
      cloudBrightness,
      cloudEdge,
      sceneAppearance.resolved,
    ),
    sceneAppearance.resolved,
  ), [
    vortexColorMode,
    cloudTone,
    cloudCustomColor,
    cloudBrightness,
    cloudEdge,
    sceneAppearance.resolved,
  ]);
  const vortexSurfaceColorsRef = useRef(vortexSurfaceColors);
  vortexSurfaceColorsRef.current = vortexSurfaceColors;

  /* Legend follows the contours and selected material actually drawn. Naming
   * the sign as well as the inequality keeps neutral/cloud-matched geometry
   * scientifically readable without forcing diagnostic color. */
  const vortexCoreLegendItems = useMemo(() => {
    const contours = vortexCoreContours(vortexCoreFieldName, vortexCoreThreshold ?? Number.NaN);
    const result = vortexExtraction.result;
    if (!contours || !result) return null;
    return [
      (result.core?.triangles ?? 0) > 0 ? {
        color: rgbCss(contrastAdjustedColor(
          vortexSurfaceColors.core,
          sceneAppearance.resolved.background,
        )),
        label: `${withVortexCoreUnit(vortexCoreLabel(contours.core))}`
          + ` · ${tornadoScaleResolved
            ? "connected tornado-scale vortex"
            : vortexCoreFieldName === "vorticity_z"
              ? "connected low-level cyclonic core"
              : vortexCoreDescription(vortexCoreFieldName, "core")}`
          + `${(vortexExtraction.result?.coreComponentCount ?? 0) > 1
            ? ` · ${vortexExtraction.result!.coreComponentCount} resolved components`
            : ""}`,
      } : null,
      (result.opposite?.triangles ?? 0) > 0 ? {
        color: rgbCss(contrastAdjustedColor(
          vortexSurfaceColors.opposite,
          sceneAppearance.resolved.background,
        )),
        label: `${withVortexCoreUnit(vortexCoreLabel(contours.opposite))}`
          + ` · ${vortexCoreDescription(vortexCoreFieldName, "opposite")}`,
      } : null,
      ...(mesocycloneAssessment?.present && mesocycloneThreshold != null
          && (result.mesocyclone?.triangles ?? 0) > 0
        ? [{
            color: rgbCss(contrastAdjustedColor(
              vortexSurfaceColors.mesocyclone,
              sceneAppearance.resolved.background,
            )),
            label: `>= +${formatCoreThreshold(mesocycloneThreshold)} s⁻¹`
              + " · connected mesocyclone-scale envelope",
          }]
        : []),
      ...(showTornadicCondensation && tornadicCondensationAssessment?.visible
          && result.condensation != null
        ? [{
            color: rgbCss(contrastAdjustedColor(
              TORNADIC_CONDENSATION_SILVER,
              sceneAppearance.resolved.background,
            )),
            label: tornadicCondensationAssessment.present
              ? `${tornadicCondensationSourceLabel} · resolved condensation funnel`
              : `${tornadicCondensationSourceLabel} · rotating condensation evidence`,
          }]
        : []),
    ].filter((item): item is { color: string; label: string } => item != null);
    // withVortexCoreUnit closes over vortexCoreUnit, which is the only value it reads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    vortexCoreFieldName,
    vortexCoreThreshold,
    vortexCoreUnit,
    tornadoScaleResolved,
    mesocycloneAssessment,
    mesocycloneThreshold,
    showTornadicCondensation,
    tornadicCondensationAssessment,
    tornadicCondensationSourceLabel,
    vortexSurfaceColors,
    sceneAppearance.resolved.background,
    vortexExtraction.result?.coreComponentCount,
    vortexExtraction.result?.core?.triangles,
    vortexExtraction.result?.opposite?.triangles,
    vortexExtraction.result?.mesocyclone?.triangles,
    vortexExtraction.result?.condensation,
  ]);

  // ---------- vtk.js scene refs ----------
  const containerRef = useRef<HTMLDivElement | null>(null);
  type SceneRefs = {
    motion: ReturnType<typeof createCloudMotionController>;
    renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    contextCloudImage: ReturnType<typeof vtkImageData.newInstance>;
    contextCloudVolume: ReturnType<typeof vtkVolume.newInstance>;
    contextCloudMapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
    contextCloudCtf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    contextCloudPwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>;
    // Cloud volume
    cloudImage: ReturnType<typeof vtkImageData.newInstance>;
    cloudVolume: ReturnType<typeof vtkVolume.newInstance>;
    cloudMapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
    cloudCtf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    cloudPwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>;
    condensationImage: ReturnType<typeof vtkImageData.newInstance>;
    condensationVolume: ReturnType<typeof vtkVolume.newInstance>;
    condensationMapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
    condensationCtf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    condensationPwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>;
    precipImage: ReturnType<typeof vtkImageData.newInstance>;
    precipVolume: ReturnType<typeof vtkVolume.newInstance>;
    precipMapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
    precipCtf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    precipPwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>;
    quality: VtkVolumeQualityController;
    physicalSampleDistance: number;
    contextPhysicalSampleDistance: number;
    condensationPhysicalSampleDistance: number;
    precipPhysicalSampleDistance: number;
    lighting: VtkSceneLighting;
    // dBZ ground plane
    // Updraft / downdraft geometry arrives from the contour worker.
    upData: ReturnType<typeof vtkPolyData.newInstance>;
    coreData: ReturnType<typeof vtkPolyData.newInstance>;
    coreActor: ReturnType<typeof vtkActor.newInstance>;
    oppositeData: ReturnType<typeof vtkPolyData.newInstance>;
    oppositeActor: ReturnType<typeof vtkActor.newInstance>;
    mesocycloneData: ReturnType<typeof vtkPolyData.newInstance>;
    mesocycloneActor: ReturnType<typeof vtkActor.newInstance>;
    upActor: ReturnType<typeof vtkActor.newInstance>;
    downData: ReturnType<typeof vtkPolyData.newInstance>;
    downActor: ReturnType<typeof vtkActor.newInstance>;
  };
  const sceneRef = useRef<SceneRefs | null>(null);
  updateViewDetailRef.current = () => {
    if (viewDetailInteractionRef.current.local || viewDetailInteractionRef.current.remote || viewDetailInteractionRef.current.pointerHeld) return;
    if (!viewDetailSupported || detailTargetMode !== "view" || playing || recording || !ctxLoss.available
        || (!showCloud && !showPrecip) || (!forceNativeStormDetail && stormVolumeResolution !== "storm")) return;
    const focal = sceneRef.current?.renderWindow.getRenderer().getActiveCamera().getFocalPoint();
    const scale = sceneScaleRef.current.xy;
    if (!focal || !Number.isFinite(scale) || scale <= 0) return;
    const center = clampGridHorizontalPointM(rootCurrentStep?.grid, [focal[0] / scale, focal[1] / scale]);
    if (!center) return;
    setViewDetailTarget(previous => {
      const validPrevious = previous?.runDir === runDir ? clampGridHorizontalPointM(rootCurrentStep?.grid, previous.center) : null;
      if (previous?.runDir === runDir && validPrevious
          && validPrevious.every((value, axis) => value === previous.center[axis])
          && outputHierarchySource.levelAtFocus(previous.center) === outputHierarchySource.levelAtFocus(center)
          && Math.hypot(center[0] - previous.center[0], center[1] - previous.center[1]) < stormRecenteringThresholdM) return previous;
      return { runDir, center };
    });
  };
  useEffect(() => { scheduleViewDetailRef.current?.(); }, [detailTargetMode, playing, recording, runDir,
    rootCurrentStep?.grid, showCloud, showPrecip, stormVolumeResolution, vtkLifecycleGeneration]);

  useEffect(() => {
    if (cloudMotionAllowed) return;
    sceneRef.current?.motion.reset();
    setCloudMotionActive(false);
    setInterpolatedDwellMs(0);
  }, [cloudMotionAllowed, vtkLifecycleGeneration]);
  const cameraInitializedRef = useRef(false);
  const resetFocusBoundsRef = useRef<BoundsTuple | null>(null);
  const clippingSupportBoundsRef = useRef<BoundsTuple | null>(null);
  const resetFocusRunRef = useRef(sourceRunDir);
  if (resetFocusRunRef.current !== sourceRunDir) {
    resetFocusRunRef.current = sourceRunDir;
    resetFocusBoundsRef.current = null;
    clippingSupportBoundsRef.current = null;
  }
  const cameraOrientation = useCameraOrientation(
    () => sceneRef.current?.renderWindow ?? null,
    sceneScale,
    stormTrackCentre(matchingStormTrack),
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
      onCameraApplied: () => {
        cameraInitializedRef.current = true;
        const scene = sceneRef.current;
        if (!scene) return;
        setCameraAdjustedVolumeRayDistances(
          scene.contextCloudVolume.getProperty(),
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          CLOUD_OPACITY_REFERENCE_M,
          scene.contextPhysicalSampleDistance,
          sceneScaleRef.current,
          scene.contextCloudMapper,
        );
        setCameraAdjustedVolumeRayDistances(
          scene.cloudVolume.getProperty(),
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          CLOUD_OPACITY_REFERENCE_M,
          scene.physicalSampleDistance,
          sceneScaleRef.current,
          scene.cloudMapper,
        );
        setCameraAdjustedVolumeRayDistances(
          scene.condensationVolume.getProperty(),
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          CLOUD_OPACITY_REFERENCE_M,
          scene.condensationPhysicalSampleDistance,
          sceneScaleRef.current,
          scene.condensationMapper,
        );
        setPrecipRayDistances(
          scene.precipVolume,
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          scene.precipPhysicalSampleDistance,
          sceneScaleRef.current,
          scene.precipMapper,
        );
      },
      onRemoteInteractionChange: (active) => {
        viewDetailInteractionRef.current.remote = active;
        if (!active) scheduleViewDetailRef.current?.();
        const quality = sceneRef.current?.quality;
        if (active) quality?.begin("synchronization");
        else quality?.end("synchronization");
      },
    },
  );
  const radarFloor = useRadarFloorLayer(() => sceneRef.current?.renderWindow ?? null, {
    runDir: sourceRunDir,
    // The floor is a small 2D hyperslab and resolves well before the 3D fields,
    // so requesting the debounced step would paint a later frame under a storm
    // still showing an earlier one. Follow the step the volumes settled on.
    stepIndex: loadedStep?.index ?? loadFrame.stepIndex,
    publicationGeneration:
      loadedStep?.publicationGeneration ?? loadFrame.publicationGeneration,
    fieldName: floorSelection.fieldName,
    reduction: floorSelection.reduction,
    opacity: floorSelection.opacity,
    availableFields: fieldsForStep,
    fieldOrigins: loadedStep?.fieldOrigins ?? currentStep?.fieldOrigins,
    enabled: floorSelection.enabled && !terrainBlocked,
    grid: renderGrid,
    scale: sceneScale,
    lifecycleGeneration: vtkLifecycleGeneration,
    deferDataRender: true,
  });
  const supercellGpuUploadBytes =
    (showCloud ? volumeImageBytes(activeCloudVolumePrepared, textureCapacity.known ? textureCapacity.maximumDimension : undefined) : 0)
    + (showCloud && cloudContextActive
      ? volumeImageBytes(rootContextPreparation.prepared?.volume, textureCapacity.known ? textureCapacity.maximumDimension : undefined) : 0)
    + (showPrecip ? volumeImageBytes(precipPreparation.prepared, textureCapacity.known ? textureCapacity.maximumDimension : undefined) : 0)
    + (showUpdraft
      ? surfaceGeometryBytes(flowExtraction.result?.core ?? null)
        + surfaceGeometryBytes(flowExtraction.result?.opposite ?? null)
      : 0)
    + (showVortexCore
      ? surfaceGeometryBytes(vortexExtraction.result?.core ?? null)
        + surfaceGeometryBytes(vortexExtraction.result?.opposite ?? null)
        + surfaceGeometryBytes(vortexExtraction.result?.mesocyclone ?? null)
        + (showTornadicCondensation
            && vortexExtraction.result?.condensationAssessment?.visible
          ? volumeImageBytes(vortexExtraction.result?.condensation, textureCapacity.known ? textureCapacity.maximumDimension : undefined) : 0)
      : 0);
  const supercellGpuUpload = useVtkGpuResourceOwnership({
    enabled: supercellGpuUploadBytes > 0,
    owner: "supercell",
    generation: JSON.stringify([
      sourceRunDir, sourceDefinitionIdentity, loadedStep?.index, loadedStep?.publicationGeneration,
      showCloud, cloudContextActive, showPrecip, precipMode,
      showUpdraft, flowThreshold, showVortexCore, vortexCoreFieldName,
      vortexCoreThreshold, vortexStructure, showTornadicCondensation,
      vtkLifecycleGeneration,
    ]),
    bytes: supercellGpuUploadBytes,
  });
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      volumeRaySampling: "pixel-anchored-physical-distance",
      shadowIntegration: "physical-midpoint-intervals",
      shadowSampleFactor: qualityPreview ? CLOUD_SHADOW_SAMPLE_FACTOR_PREVIEW : CLOUD_SHADOW_SAMPLE_FACTOR_FULL,
      logicalGpuUploadBytes: supercellGpuUpload.admitted ? supercellGpuUploadBytes : 0,
    },
  };

  const cloudFrameExpected = showCloud && cloudFieldName != null && cloudWindows != null;
  const contextCloudFrameExpected = showCloud && cloudContextActive
    && rootContextCloudFieldName != null && (rootContextWindows != null || publishedContextActive);
  const flowSourceError = showUpdraft && currentStep && !updraftFieldName
    ? "Flow unavailable: this frame does not publish the solved w field. Turn off Updraft / downdraft to present its available layers."
    : null;
  const vortexSourceError = showVortexCore && currentStep
    && (!vortexCoreFieldName || !vortexCoreFields.includes(vortexCoreFieldName))
    ? `Vortex core unavailable: this frame does not publish selected ${vortexCoreFieldName ?? "vortex diagnostic"}. Select an available core field or turn off Vortex core.`
    : null;
  const precipSourceError = showPrecip && currentStep
    && (precipMode === "total" ? availablePrecipSpecies.length === 0 : !availablePrecipSpecies.includes(precipMode))
    ? precipMode === "total"
      ? "Precipitation unavailable: this frame does not publish any falling hydrometeor species. Turn off Precipitation to present its available layers."
      : `Precipitation unavailable: this frame does not publish selected ${precipMode}. Select an available species or turn off Precipitation.`
    : null;
  const precipFrameExpected = showPrecip && precipSourceError == null;
  const radarFrameExpected = floorSelection.enabled && floorSelection.fieldName != null;
  const sceneGenerationIdentity = JSON.stringify([
    sourceRunDir, planningStep?.index, planningStep?.publicationGeneration, sourceDefinitionIdentity,
    contextRunDir, contextStep?.index, contextStep?.publicationGeneration, vtkLifecycleGeneration,
    cloudWindows, rootContextWindows, publishedContextActive ? publishedRootContextCloud.sourceLevel : null,
    rainWindows, graupelWindows, hailWindows, updraftWindows, vortexCoreWindows,
    primaryCondensateWindows, frozenCondensateWindows,
    showCloud, showPrecip, precipMode, showUpdraft, showVortexCore, showTornadicCondensation,
    flowThreshold, committedVortexCoreFraction, vortexStructure, floorSelection.fieldName,
    floorSelection.reduction, floorSelection.enabled, renderFidelityMode,
  ]);
  const contextCenter = !hierarchyContextActive && cloudContextActive
    ? rootContextPreparation.prepared?.cloudCenter
    : !nativeStormVolumes ? cloudPreparation.prepared?.cloudCenter : null;
  const cloudLocatorPending = detailTargetMode === "storm" && showCloud && !trackedStormCenter
    && contextCenter != null && (!fallbackStormCenter || Math.hypot(contextCenter[0] - fallbackStormCenter[0], contextCenter[1] - fallbackStormCenter[1]) >= stormRecenteringThresholdM);
  const baseFramePending = (
    windPending || cloudLocatorPending || (activeLoads.length > 0 && !coherentFrame)
    || (cloudFrameExpected && (
      cloud.loading
      || cloudPreparation.loading
      || (cloudPreparation.prepared == null && cloudPreparation.error == null)
    ))
    || (contextCloudFrameExpected && (
      rootContextCloud.loading
      || publishedContextActive && publishedRootContextCloud.planning
      || rootContextPreparation.loading
      || (rootContextPreparation.prepared == null && rootContextPreparation.error == null)
    ))
    || (precipFrameExpected && (
      precipPreparation.loading
      || (precipPreparation.prepared == null && precipPreparation.error == null)
    ))
    || (radarFrameExpected && radarFloor.loading)
    || (supercellGpuUploadBytes > 0 && !supercellGpuUpload.admitted)
  );
  const baseReadError = activeLoads.find(load => load.error != null)?.error ?? null;
  const requiredLayers = [
    { layer: "base", identity: sceneGenerationIdentity,
      ready: !baseFramePending && activeLoads.every(load => load === rootContextCloud
        ? Boolean(contextStep && !load.loading && load.loadedStepIndex === contextStep.index
          && load.loadedPublicationGeneration === contextStep.publicationGeneration)
        : loadMatchesRequestedStep(load)),
      error: textureCapacity.error ?? precipSourceError ?? baseReadError ?? (forceNativeStormDetail && (showCloud || showPrecip) && !stormDetailOutcome.ok ? stormDetailOutcome.reason : null)
        ?? boundedPlanError ?? cloudPreparation.error ?? precipPreparation.error
        ?? (contextCloudFrameExpected ? rootContextPreparation.error ?? rootContextCloud.error : null)
        ?? supercellGpuUpload.error },
    ...(showUpdraft ? [{ layer: "flow", identity: sceneGenerationIdentity,
      ready: flowSourceError == null && !flowLayerLoading && loadMatchesRequestedStep(updraft),
      error: flowSourceError ?? updraft.error ?? flowDiagnosticPlanError ?? updraftPreparation.error ?? flowExtraction.error }] : []),
    ...(showVortexCore ? [{ layer: "vortex", identity: sceneGenerationIdentity,
      ready: vortexSourceError == null && !vortexLayerLoading && loadMatchesRequestedStep(vortexCoreField) && condensateLoadsReady,
      error: vortexSourceError ?? vortexCoreField.error ?? vortexDiagnosticPlanError ?? vortexCorePreparation.error ?? vortexExtraction.error
        ?? (tornadicCondensationEnabled ? primaryCondensate.error ?? frozenCondensate.error ?? tornadicCondensatePreparation.error : null) }] : []),
    ...(radarFrameExpected ? [{ layer: "radar", identity: sceneGenerationIdentity,
      ready: !radarFloor.loading && radarFloor.values != null, error: radarFloor.error }] : []),
  ];
  const sceneGeneration = coherentSceneGeneration(sceneGenerationIdentity, requiredLayers);
  const sceneFrameCommitPending = !sceneGeneration.ready;
  const panePresentation = usePanePresentationBarrier({ ready: sceneGeneration.ready, error: sceneGeneration.error });
  const scenePresentationPending = sceneFrameCommitPending || panePresentation.pending;
  // Hold before layout effects rescale geometry; presentation releases after axes and materials agree.
  useLayoutEffect(() => {
    const scene = sceneRef.current;
    if (!scene || scenePresentationPending) return;
    const release = holdVtkSceneRendering(scene.renderWindow);
    releaseSceneCommitRef.current = release;
    return () => { release(); if (releaseSceneCommitRef.current === release) releaseSceneCommitRef.current = null; };
  }, [sceneGenerationIdentity, scenePresentationPending, sceneScale.xy, sceneScale.z,
    sceneAppearance.resolved, vortexSurfaceColors.core, vortexSurfaceColors.mesocyclone,
    vortexSurfaceColors.opposite, vtkLifecycleGeneration]);
  const displayContextGrid = scenePresentationPending ? committedContextGrid ?? contextGrid : contextGrid;
  useDomainBox(() => sceneRef.current?.renderWindow ?? null, displayContextGrid, sceneScale,
    vtkLifecycleGeneration, sceneAppearance.resolved.reference);
  const axisLabels = useDomainAxes(() => sceneRef.current?.renderWindow ?? null, containerRef,
    displayContextGrid, sceneScale, vtkLifecycleGeneration, sceneAppearance.resolved.reference, prepareSceneAxesRef);
  useEffect(() => {
    if (scenePresentationPending) return;
    activeRootContextPreparedRef.current = rootContextPreparation.prepared?.volume ?? null;
    activeCloudVolumePreparedRef.current = activeCloudVolumePrepared;
    activePrecipPreparedRef.current = precipPreparation.prepared;
    activeCondensationPreparedRef.current = vortexExtraction.result?.condensation ?? null;
    resetFocusBoundsRef.current = showCloud ? cloudPreparation.prepared?.surfaceFocusBounds ?? null : null;
    clippingSupportBoundsRef.current = showCloud ? cloudPreparation.prepared?.surfaceSupportBounds ?? null : null;
  }, [
    scenePresentationPending, rootContextPreparation.prepared, activeCloudVolumePrepared,
    precipPreparation.prepared, vortexExtraction.result, showCloud, cloudPreparation.prepared,
  ]);
  const cameraElevation = useCameraElevation(displayContextGrid);
  useCameraElevationPlacement(() => sceneRef.current?.renderWindow ?? null, {
    grid: displayContextGrid,
    scale: sceneScale,
    selection: cameraElevation,
    targetXYM: scenePresentationPending ? committedCameraTargetRef.current
      : trackedStormCenter ? [trackedStormCenter[0], trackedStormCenter[1]] : null,
    lifecycleGeneration: vtkLifecycleGeneration,
    onCameraApplied: () => {
      cameraInitializedRef.current = true;
      const scene = sceneRef.current;
      if (!scene) return;
      setCameraAdjustedVolumeRayDistances(
        scene.contextCloudVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.contextPhysicalSampleDistance,
        sceneScaleRef.current,
        scene.contextCloudMapper,
      );
      setCameraAdjustedVolumeRayDistances(
        scene.cloudVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.physicalSampleDistance,
        sceneScaleRef.current,
        scene.cloudMapper,
      );
      setCameraAdjustedVolumeRayDistances(
        scene.condensationVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.condensationPhysicalSampleDistance,
        sceneScaleRef.current,
        scene.condensationMapper,
      );
      setPrecipRayDistances(
        scene.precipVolume,
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        scene.precipPhysicalSampleDistance,
        sceneScaleRef.current,
        scene.precipMapper,
      );
    },
  });

  // Re-frame on run change; a different run can have a very different domain extent.
  useVtkRunReset(runDir, cameraInitializedRef, () => {
    const s = sceneRef.current;
    if (!s) return;
    s.contextCloudVolume.setVisibility(false);
    s.cloudVolume.setVisibility(false);
    s.condensationVolume.setVisibility(false);
    s.precipVolume.setVisibility(false);
    s.coreActor.setVisibility(false);
    s.oppositeActor.setVisibility(false);
    s.mesocycloneActor.setVisibility(false);
    s.upActor.setVisibility(false);
    s.downActor.setVisibility(false);
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
    textureCapacity.probe(renderWindow);
    const renderer = renderWindow.getRenderer();
    configureTranslucentRendering(renderer);
    const lighting = createVtkSceneLighting(renderer);
    lighting.apply(
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );

    // --- Coarse parent context volume (masked wherever the fine child owns data) ---
    const contextCloudImage = vtkImageData.newInstance();
    const contextCloudMapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(contextCloudMapper);
    contextCloudMapper.setSampleDistance(5000);
    contextCloudMapper.setMaximumSamplesPerRay(1800);
    contextCloudMapper.setAutoAdjustSampleDistances(false);
    contextCloudMapper.setVolumeShadowSamplingDistFactor(CLOUD_SHADOW_SAMPLE_FACTOR_FULL);
    contextCloudMapper.setInputData(contextCloudImage);
    const contextCloudCtf = vtkColorTransferFunction.newInstance();
    const contextCloudPwf = vtkPiecewiseFunction.newInstance();
    const contextCloudVolume = vtkVolume.newInstance();
    contextCloudVolume.setMapper(contextCloudMapper);
    contextCloudVolume.setVisibility(false);
    {
      const property = contextCloudVolume.getProperty();
      property.setRGBTransferFunction(0, contextCloudCtf);
      property.setScalarOpacity(0, contextCloudPwf);
      property.setInterpolationTypeToLinear();
      property.setComputeNormalFromOpacity(true);
      property.setShade(false);
      property.setAmbient(0.3);
      property.setDiffuse(0.85);
      property.setSpecular(0);
      property.setVolumetricScatteringBlending(1);
      property.setGlobalIlluminationReach(0.05);
      property.setAnisotropy(0.24);
    }
    renderer.addVolume(contextCloudVolume);

    // --- Fine/root primary cloud volume ---
    const cloudImage = vtkImageData.newInstance();
    const cloudMapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(cloudMapper);
    cloudMapper.setSampleDistance(5000);
    cloudMapper.setMaximumSamplesPerRay(3000);
    // Disable vtk.js's hidden heuristic; Nimbus applies its own labeled,
    // deterministic interaction preview and restores full settings at idle.
    cloudMapper.setAutoAdjustSampleDistances(false);
    // Self-shadow rays are still cheaper than primary rays, but not so coarse
    // that their intervals become visible as parallel ridges in dense cloud.
    cloudMapper.setVolumeShadowSamplingDistFactor(CLOUD_SHADOW_SAMPLE_FACTOR_FULL);
    cloudMapper.setInputData(cloudImage);
    const cloudCtf = vtkColorTransferFunction.newInstance();
    const cloudPwf = vtkPiecewiseFunction.newInstance();
    const cloudVolume = vtkVolume.newInstance();
    cloudVolume.setMapper(cloudMapper);
    cloudVolume.setVisibility(false);
    {
      const p = cloudVolume.getProperty();
      p.setRGBTransferFunction(0, cloudCtf);
      p.setScalarOpacity(0, cloudPwf);
      p.setInterpolationTypeToLinear();
      p.setComputeNormalFromOpacity(true);
      // Gradients on empty image data are unsafe on some WebGL drivers. The
      // first prepared frame enables the selected lighting preset.
      p.setShade(false);
      p.setAmbient(0.3);
      p.setDiffuse(0.85);
      p.setSpecular(0.0);
      p.setVolumetricScatteringBlending(1);
      p.setGlobalIlluminationReach(0.05);      // shadow-ray length, fraction of diagonal
      p.setAnisotropy(0.24);                   // mild forward scatter
    }
    renderer.addVolume(cloudVolume);

    // --- Evidence-gated tornadic condensation volume ---
    // Scalars contain only solved condensate retained around the connected
    // vorticity centreline. The worker never extrapolates a funnel into dry air.
    const condensationImage = vtkImageData.newInstance();
    const condensationMapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(condensationMapper);
    condensationMapper.setSampleDistance(5000);
    condensationMapper.setMaximumSamplesPerRay(1800);
    condensationMapper.setAutoAdjustSampleDistances(false);
    condensationMapper.setVolumeShadowSamplingDistFactor(CLOUD_SHADOW_SAMPLE_FACTOR_FULL);
    condensationMapper.setInputData(condensationImage);
    const condensationCtf = vtkColorTransferFunction.newInstance();
    const condensationPwf = vtkPiecewiseFunction.newInstance();
    const condensationVolume = vtkVolume.newInstance();
    condensationVolume.setMapper(condensationMapper);
    condensationVolume.setVisibility(false);
    {
      const property = condensationVolume.getProperty();
      property.setRGBTransferFunction(0, condensationCtf);
      property.setScalarOpacity(0, condensationPwf);
      property.setInterpolationTypeToLinear();
      property.setComputeNormalFromOpacity(true);
      property.setShade(false);
      property.setAmbient(0.45);
      property.setDiffuse(0.75);
      property.setSpecular(0);
      property.setVolumetricScatteringBlending(1);
      property.setGlobalIlluminationReach(0.06);
      property.setAnisotropy(0.25);
    }
    renderer.addVolume(condensationVolume);

    // --- Falling-hydrometeor volume ---
    // Exactly one active precip volume is retained. The user can view the sum
    // or one species without paying for three concurrent ray-marched textures.
    const precipImage = vtkImageData.newInstance();
    const precipMapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(precipMapper);
    precipMapper.setSampleDistance(5000);
    precipMapper.setMaximumSamplesPerRay(3000);
    precipMapper.setAutoAdjustSampleDistances(false);
    precipMapper.setInputData(precipImage);
    const precipCtf = vtkColorTransferFunction.newInstance();
    const precipPwf = vtkPiecewiseFunction.newInstance();
    const precipVolume = vtkVolume.newInstance();
    precipVolume.setMapper(precipMapper);
    precipVolume.setVisibility(false);
    {
      const property = precipVolume.getProperty();
      property.setRGBTransferFunction(0, precipCtf);
      property.setScalarOpacity(0, precipPwf);
      property.setInterpolationTypeToLinear();
      property.setShade(false);
      property.setAmbient(1);
      property.setDiffuse(0);
      property.setSpecular(0);
    }
    renderer.addVolume(precipVolume);

    // --- Updraft / downdraft isosurfaces (off-thread extraction) ---
    const upData = vtkPolyData.newInstance();
    const upMapper = vtkMapper.newInstance({ scalarVisibility: false });
    upMapper.setInputData(upData);
    const upActor = vtkActor.newInstance();
    upActor.setMapper(upMapper);
    upActor.setVisibility(false);
    upActor.getProperty().setColor(...UPDRAFT_GOLD);
    upActor.getProperty().setOpacity(0.85);
    renderer.addActor(upActor);

    const downData = vtkPolyData.newInstance();
    const downMapper = vtkMapper.newInstance({ scalarVisibility: false });
    downMapper.setInputData(downData);
    const downActor = vtkActor.newInstance();
    downActor.setMapper(downMapper);
    downActor.setVisibility(false);
    downActor.getProperty().setColor(...DOWNDRAFT_BLUE);
    downActor.getProperty().setOpacity(0.6);
    renderer.addActor(downActor);

    // --- Vortex core: geometry arrives from the latest-wins contour worker. ---
    const coreData = vtkPolyData.newInstance();
    const coreMapper = vtkMapper.newInstance({ scalarVisibility: false });
    coreMapper.setInputData(coreData);
    const coreActor = vtkActor.newInstance();
    coreActor.setMapper(coreMapper);
    coreActor.setVisibility(false);
    coreActor.getProperty().setColor(...vortexSurfaceColorsRef.current.core);
    coreActor.getProperty().setOpacity(0.8);
    renderer.addActor(coreActor);

    const oppositeData = vtkPolyData.newInstance();
    const oppositeMapper = vtkMapper.newInstance({ scalarVisibility: false });
    oppositeMapper.setInputData(oppositeData);
    const oppositeActor = vtkActor.newInstance();
    oppositeActor.setMapper(oppositeMapper);
    oppositeActor.setVisibility(false);
    oppositeActor.getProperty().setColor(...vortexSurfaceColorsRef.current.opposite);
    oppositeActor.getProperty().setOpacity(0.55);
    renderer.addActor(oppositeActor);

    const mesocycloneData = vtkPolyData.newInstance();
    const mesocycloneMapper = vtkMapper.newInstance({ scalarVisibility: false });
    mesocycloneMapper.setInputData(mesocycloneData);
    const mesocycloneActor = vtkActor.newInstance();
    mesocycloneActor.setMapper(mesocycloneMapper);
    mesocycloneActor.setVisibility(false);
    mesocycloneActor.getProperty().setColor(...vortexSurfaceColorsRef.current.mesocyclone);
    mesocycloneActor.getProperty().setOpacity(0.22);
    renderer.addActor(mesocycloneActor);

    for (const surfaceActor of [
      upActor,
      downActor,
      coreActor,
      oppositeActor,
      mesocycloneActor,
    ]) {
      applySurfaceLighting(
        surfaceActor.getProperty(),
        sceneAppearanceRef.current.lightingOptions,
        "translucent",
      );
    }

    const quality = createVtkVolumeQualityController({
      mapper: [contextCloudMapper, cloudMapper, condensationMapper, precipMapper],
      render: () => renderWindow.getRenderWindow().render(),
      // Preserve depth sampling closely enough that thin cloud does not blink
      // or change optical character during a gesture. Most preview savings now
      // come from one ray per 2x2 screen pixels, which leaves the samples along
      // every surviving ray scientifically stable and is cheaper than the old
      // 4x depth stride.
      previewDistanceMultiplier: 2,
      previewImageSampleDistance: 2,
      previewMaximumSamples: 768,
      onPreviewChange: (preview) => {
        if (sceneRef.current?.renderWindow !== renderWindow) return;
        cloudMapper.setVolumeShadowSamplingDistFactor(
          preview
            ? CLOUD_SHADOW_SAMPLE_FACTOR_PREVIEW
            : CLOUD_SHADOW_SAMPLE_FACTOR_FULL,
        );
        contextCloudMapper.setVolumeShadowSamplingDistFactor(
          preview
            ? CLOUD_SHADOW_SAMPLE_FACTOR_PREVIEW
            : CLOUD_SHADOW_SAMPLE_FACTOR_FULL,
        );
        condensationMapper.setVolumeShadowSamplingDistFactor(
          preview ? CLOUD_SHADOW_SAMPLE_FACTOR_PREVIEW : CLOUD_SHADOW_SAMPLE_FACTOR_FULL,
        );
        setQualityPreview(preview);
      },
    });
    const motion = createCloudMotionController(renderer, () => renderWindow.getRenderWindow().render(),
      () => renderWindow.getApiSpecificRenderWindow().getContext() as WebGL2RenderingContext | null,
      scale => setVtkResolutionScale(renderWindow, scale));
    sceneRef.current = {
      motion, renderWindow,
      contextCloudImage, contextCloudVolume, contextCloudMapper, contextCloudCtf, contextCloudPwf,
      coreData, coreActor,
      oppositeData, oppositeActor,
      mesocycloneData, mesocycloneActor,
      cloudImage, cloudVolume, cloudMapper, cloudCtf, cloudPwf,
      condensationImage, condensationVolume, condensationMapper,
      condensationCtf, condensationPwf,
      precipImage, precipVolume, precipMapper, precipCtf, precipPwf,
      quality,
      physicalSampleDistance: 5000,
      contextPhysicalSampleDistance: 5000,
      condensationPhysicalSampleDistance: 5000,
      precipPhysicalSampleDistance: 5000,
      lighting,
      upData, upActor, downData, downActor,
    };
    const playbackResolution = attachVtkPlaybackScreenResolution(renderWindow, {
      getMappers: () => [contextCloudMapper, cloudMapper, condensationMapper, precipMapper],
      shouldMeasure: () => !quality.isPreview(),
      onChange: snapshot => {
        setPlaybackScreenDistance(snapshot.distance);
        container.dataset.playbackScreenDistance = String(snapshot.distance);
        container.dataset.playbackScreenTiming = snapshot.timing;
      },
    });

    let viewDetailTimer: ReturnType<typeof setTimeout> | null = null;
    const scheduleViewDetail = () => {
      if (viewDetailTimer != null) clearTimeout(viewDetailTimer);
      viewDetailTimer = setTimeout(() => { viewDetailTimer = null; updateViewDetailRef.current(); }, 400);
    };
    scheduleViewDetailRef.current = scheduleViewDetail;
    const viewDetailCamera = renderer.getActiveCamera().onModified(scheduleViewDetail);
    scheduleViewDetail();
    const jointVolumes = bindJointVolumes([
      { actor: contextCloudVolume, role: "context", shadowReachM: 3000 }, { actor: cloudVolume, role: "cloud", shadowReachM: 3000 },
      { actor: condensationVolume, role: "condensation", shadowReachM: 3000 }, { actor: precipVolume, role: "independent" },
      ...motion.sources,
    ]);
    motion.attach(jointVolumes);
    const heldPointers = new Set<number>();
    const holdViewDetail = (event: PointerEvent) => {
      heldPointers.add(event.pointerId);
      viewDetailInteractionRef.current.pointerHeld = true;
    };
    const releaseViewDetail = (event: PointerEvent) => {
      heldPointers.delete(event.pointerId);
      viewDetailInteractionRef.current.pointerHeld = heldPointers.size > 0;
      if (!heldPointers.size) scheduleViewDetail();
    };
    const clearHeldPointers = () => {
      heldPointers.clear();
      viewDetailInteractionRef.current.pointerHeld = false;
      viewDetailInteractionRef.current.local = false;
      scheduleViewDetail();
    };
    container.addEventListener("pointerdown", holdViewDetail, { capture: true });
    document.addEventListener("pointerup", releaseViewDetail, { capture: true });
    document.addEventListener("pointercancel", releaseViewDetail, { capture: true });
    window.addEventListener("blur", clearHeldPointers);
    const stopMotionBeforeInput = () => motion.stop(true);
    for (const event of ["pointerdown", "mousedown", "wheel"]) container.addEventListener(event, stopMotionBeforeInput, { capture: true });
    registerVtkAppearanceCapture(container, renderWindow, () => ({
      ...(committedAppearanceRef.current ?? appearanceEvidenceRef.current),
      frameIndex: motion.describe().active ? null : committedAppearanceRef.current?.frameIndex ?? null,
      fidelityMode: motion.describe().active ? "presentation" : committedAppearanceRef.current?.fidelityMode ?? appearanceEvidenceRef.current.fidelityMode,
      parameters: { ...(committedAppearanceRef.current?.parameters ?? appearanceEvidenceRef.current.parameters),
        shadowSampleFactor: cloudMapper.getVolumeShadowSamplingDistFactor(),
        interactionQuality: quality.isPreview() ? "preview" : "settled",
        cloudMotion: motion.describe(),
        volumeComposition: jointVolumes.describe().mode, jointVolumeLayers: jointVolumes.describe().layers,
      },
    }), () => motion.stop(true));

    const detachContextLoss = attachContextLossRef.current(renderWindow);

    const detachInteractionQuality = attachVtkInteractionQuality(renderWindow, quality, {
      onStartInteraction: () => { viewDetailInteractionRef.current.local = true; motion.stop(true); },
      onEndInteraction: () => { viewDetailInteractionRef.current.local = false; scheduleViewDetail(); },
    });
    const detachCameraStability = attachVtkCameraStability(renderWindow, renderer, () => {
      setCameraAdjustedVolumeRayDistances(
        contextCloudVolume.getProperty(),
        renderer.getActiveCamera(),
        quality,
        CLOUD_OPACITY_REFERENCE_M,
        sceneRef.current?.renderWindow === renderWindow
          ? sceneRef.current.contextPhysicalSampleDistance
          : 5000,
        sceneScaleRef.current,
        contextCloudMapper,
      );
      setCameraAdjustedVolumeRayDistances(
        cloudVolume.getProperty(),
        renderer.getActiveCamera(),
        quality,
        CLOUD_OPACITY_REFERENCE_M,
        sceneRef.current?.renderWindow === renderWindow
          ? sceneRef.current.physicalSampleDistance
          : 5000,
        sceneScaleRef.current,
        cloudMapper,
      );
      setCameraAdjustedVolumeRayDistances(
        condensationVolume.getProperty(),
        renderer.getActiveCamera(),
        quality,
        CLOUD_OPACITY_REFERENCE_M,
        sceneRef.current?.renderWindow === renderWindow
          ? sceneRef.current.condensationPhysicalSampleDistance
          : 5000,
        sceneScaleRef.current,
        condensationMapper,
      );
      setPrecipRayDistances(
        precipVolume,
        renderer.getActiveCamera(),
        quality,
        sceneRef.current?.renderWindow === renderWindow
          ? sceneRef.current.precipPhysicalSampleDistance
          : 5000,
        sceneScaleRef.current,
        precipMapper,
      );
      renderWindow.getRenderWindow().render();
    }, {
      nearClippingPlaneTolerance: VOLUME_NEAR_CLIPPING_TOLERANCE,
      minimumViewDistanceRatio: VOLUME_NEAR_CLIPPING_TOLERANCE,
      getClippingBounds: () => {
        const activeScene = sceneRef.current;
        if (!activeScene || activeScene.renderWindow !== renderWindow) return null;
        return supercellFocusBounds(
          activeScene,
          clippingSupportBoundsRef.current ?? resetFocusBoundsRef.current,
          sceneScaleRef.current,
        );
      },
    });
    const detachResize = attachVtkResize(container, renderWindow, {
      policy: VOLUME_RESOLUTION_POLICY,
      onPreviewChange: (preview) => {
        if (preview) quality.begin("resize");
        else quality.end("resize");
      },
    });

    return () => {
      disposeVtkScene({
        clearScene: () => {
          if (sceneRef.current?.renderWindow === renderWindow) sceneRef.current = null;
        },
        cleanupSteps: [
          () => {
            viewDetailCamera.unsubscribe();
            container.removeEventListener("pointerdown", holdViewDetail, { capture: true });
            document.removeEventListener("pointerup", releaseViewDetail, { capture: true });
            document.removeEventListener("pointercancel", releaseViewDetail, { capture: true });
            window.removeEventListener("blur", clearHeldPointers);
            viewDetailInteractionRef.current = { local: false, remote: false, pointerHeld: false };
            if (viewDetailTimer != null) clearTimeout(viewDetailTimer);
            if (scheduleViewDetailRef.current === scheduleViewDetail) scheduleViewDetailRef.current = null;
          },
          () => { for (const event of ["pointerdown", "mousedown", "wheel"]) container.removeEventListener(event, stopMotionBeforeInput, { capture: true }); },
          motion.dispose,
          () => playbackResolution.dispose(),
          jointVolumes.dispose,
          detachResize,
          clearBackendEvidence,
          detachInteractionQuality,
          () => lighting.dispose(),
          detachCameraStability,
          detachContextLoss,
          () => unregisterVtkCapture(container),
          () => renderer.removeVolume(contextCloudVolume),
          () => renderer.removeVolume(cloudVolume),
          () => renderer.removeVolume(condensationVolume),
          () => renderer.removeVolume(precipVolume),
          () => renderer.removeActor(coreActor),
          () => renderer.removeActor(oppositeActor),
          () => renderer.removeActor(mesocycloneActor),
          () => renderer.removeActor(upActor),
          () => renderer.removeActor(downActor),
        ],
        resources: [
          contextCloudVolume,
          contextCloudMapper,
          contextCloudCtf,
          contextCloudPwf,
          contextCloudImage,
          cloudVolume,
          cloudMapper,
          cloudCtf,
          cloudPwf,
          cloudImage,
          condensationVolume,
          condensationMapper,
          condensationCtf,
          condensationPwf,
          condensationImage,
          precipVolume,
          precipMapper,
          precipCtf,
          precipPwf,
          precipImage,
          coreActor,
          coreMapper,
          coreData,
          oppositeActor,
          oppositeMapper,
          oppositeData,
          mesocycloneActor,
          mesocycloneMapper,
          mesocycloneData,
          upActor,
          upMapper,
          upData,
          downActor,
          downMapper,
          downData,
        ],
        renderWindow,
      });
    };
  }, [ctxLoss.available, ctxLoss.generation, textureCapacity.probe, vtkBackend.backend, vtkBackend.confirmActiveApi]);

  useVtkPlaybackScreenResolution(useCallback(() => sceneRef.current?.renderWindow ?? null, []),
    vtkLifecycleGeneration, !cloudMotionAllowed);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const appearance = sceneAppearance.resolved;
    const cloudVisual = cloudVisualRef.current;
    const precipVisual = precipVisualRef.current;
    scene.renderWindow.getRenderer().setBackground(...appearance.background);
    scene.lighting.apply(appearance.lightingOptions, appearance.isLight);
    if (scene.contextCloudVolume.getVisibility()) {
      applyVolumeLighting(scene.contextCloudVolume.getProperty(), appearance.lightingOptions, "cloud");
    }
    if (scene.cloudVolume.getVisibility()) {
      applyVolumeLighting(scene.cloudVolume.getProperty(), appearance.lightingOptions, "cloud");
    }
    if (scene.condensationVolume.getVisibility()) {
      applyVolumeLighting(
        scene.condensationVolume.getProperty(),
        appearance.lightingOptions,
        "cloud",
      );
    }
    if (scene.precipVolume.getVisibility()) {
      applyVolumeLighting(scene.precipVolume.getProperty(), appearance.lightingOptions);
    }
    const surfaceColors = [
      [scene.upActor, UPDRAFT_GOLD],
      [scene.downActor, DOWNDRAFT_BLUE],
      [scene.coreActor, vortexSurfaceColors.core],
      [scene.oppositeActor, vortexSurfaceColors.opposite],
      [scene.mesocycloneActor, vortexSurfaceColors.mesocyclone],
    ] as const;
    for (const [actor, color] of surfaceColors) {
      applySurfaceLighting(actor.getProperty(), appearance.lightingOptions, "translucent");
      actor.getProperty().setColor(...contrastAdjustedColor(color, appearance.background));
    }
    applyCloudTransfer(
      scene.contextCloudCtf,
      scene.contextCloudPwf,
      cloudOpacityRef.current,
      clampCloudFloor(cloudVisual.floorGkg / 1000),
      cloudEdgeRef.current,
      cloudSceneColors(
        cloudVisual.tone,
        cloudVisual.customColor,
        cloudVisual.brightness,
        cloudEdgeRef.current,
        appearance,
      ),
    );
    applyCloudTransfer(
      scene.cloudCtf,
      scene.cloudPwf,
      cloudOpacityRef.current,
      clampCloudFloor(cloudVisual.floorGkg / 1000),
      cloudEdgeRef.current,
      cloudSceneColors(
        cloudVisual.tone,
        cloudVisual.customColor,
        cloudVisual.brightness,
        cloudEdgeRef.current,
        appearance,
      ),
    );
    applyCloudTransfer(
      scene.condensationCtf,
      scene.condensationPwf,
      1.35,
      TORNADIC_CONDENSATE_FLOOR_KGKG,
      "subtle",
      cloudSceneColors("custom", "#c7d6e6", 1.08, "subtle", appearance),
    );
    applyPrecipTransfer(
      scene.precipCtf,
      scene.precipPwf,
      precipVisual.opacity,
      precipVisual.floorGkg / 1000,
      precipitationSceneColor(precipVisual.mode, precipVisual.customColor, appearance),
    );
  }, [
    sceneAppearance.resolved,
    vortexSurfaceColors.core,
    vortexSurfaceColors.mesocyclone,
    vortexSurfaceColors.opposite,
    vtkLifecycleGeneration,
  ]);

  // ---------- Precipitation layer update ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const prepared = precipPreparation.prepared;
    if (scenePresentationPending) return;
    if (!showPrecip || !precipRenderField || !prepared || !coherentFrame) {
      scene.precipVolume.setVisibility(false);
      return;
    }
    setPreparedImageScalars(scene.precipImage, prepared);
    setPreparedImageDisplayGeometry(scene.precipImage, prepared, sceneScaleRef.current);
    const minSpacing = Math.min(...prepared.spacing);
    if (Number.isFinite(minSpacing) && minSpacing > 0) {
      scene.precipPhysicalSampleDistance = minSpacing * 0.5;
      scene.quality.setFullSampleDistanceFor(
        scene.precipMapper,
        scene.precipPhysicalSampleDistance,
      );
    }
    const precipVisual = precipVisualRef.current;
    applyPrecipTransfer(
      scene.precipCtf,
      scene.precipPwf,
      precipVisual.opacity,
      precipVisual.floorGkg / 1000,
      precipitationSceneColor(
        precipVisual.mode,
        precipVisual.customColor,
        sceneAppearanceRef.current,
      ),
    );
    applyVolumeLighting(
      scene.precipVolume.getProperty(),
      sceneAppearanceRef.current.lightingOptions,
    );
    setPrecipRayDistances(
      scene.precipVolume,
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      scene.precipPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.precipMapper,
    );
    scene.precipVolume.setVisibility(true);
    scene.precipMapper.modified();
    maybeResetCamera(scene, cameraInitializedRef);
  }, [
    showPrecip,
    precipRenderField,
    precipPreparation.prepared,
    coherentFrame,
    scenePresentationPending,
    vtkLifecycleGeneration,
  ]);

  // ---------- Cloud layer update ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || scenePresentationPending) return;
    const prepared = rootContextPreparation.prepared?.volume;
    if (!showCloud || !cloudContextActive || !rootContextCloud.field || !prepared || !coherentFrame) {
      scene.contextCloudVolume.setVisibility(false);
      return;
    }
    setPreparedImageScalars(scene.contextCloudImage, prepared);
    setPreparedImageDisplayGeometry(scene.contextCloudImage, prepared, sceneScaleRef.current);
    const minSpacing = Math.min(...prepared.spacing);
    if (Number.isFinite(minSpacing) && minSpacing > 0) {
      scene.contextPhysicalSampleDistance = minSpacing * 0.5;
      scene.quality.setFullSampleDistanceFor(
        scene.contextCloudMapper,
        scene.contextPhysicalSampleDistance,
      );
    }
    const visual = cloudVisualRef.current;
    applyCloudTransfer(
      scene.contextCloudCtf,
      scene.contextCloudPwf,
      cloudOpacityRef.current,
      clampCloudFloor(committedCloudFloorGkg / 1000),
      cloudEdgeRef.current,
      cloudSceneColors(
        visual.tone,
        visual.customColor,
        visual.brightness,
        cloudEdgeRef.current,
        sceneAppearanceRef.current,
      ),
    );
    applyVolumeLighting(
      scene.contextCloudVolume.getProperty(),
      sceneAppearanceRef.current.lightingOptions,
      "cloud",
    );
    setCameraAdjustedVolumeRayDistances(
      scene.contextCloudVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.contextPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.contextCloudMapper,
    );
    scene.contextCloudVolume.setVisibility(true);
    scene.contextCloudMapper.modified();
  }, [
    coherentFrame,
    committedCloudFloorGkg,
    cloudContextActive,
    rootContextCloud.field,
    rootContextPreparation.prepared,
    scenePresentationPending,
    radarFloor.commitData,
    showCloud,
    vtkLifecycleGeneration,
  ]);

  // ---------- Fine/root primary cloud layer update ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (scenePresentationPending) return;
    if (!showCloud || !cloud.field || !coherentFrame) {
      scene.cloudVolume.setVisibility(false);
      return;
    }
    const prepared = cloudPreparation.prepared;
    if (!prepared) return;
    // Aggregate suspended condensate owns both visibility and opacity. Splitting
    // liquid/frozen mass into separately thresholded actors creates empty gaps
    // where their sum is a valid cloud and also makes the result draw-order
    // dependent. Presentation detail is mass-preserving and lives on the
    // prepared display lattice; scientific mode aliases the aggregate volume.
    const volumePrepared = renderFidelityMode === "presentation"
      ? prepared.surface
      : prepared.volume;
    const cloudVisual = cloudVisualRef.current;
    const floor = clampCloudFloor(committedCloudFloorGkg / 1000);

    setPreparedImageScalars(scene.cloudImage, volumePrepared);
    setPreparedImageDisplayGeometry(scene.cloudImage, volumePrepared, sceneScaleRef.current);
    const minSpacing = Math.min(...volumePrepared.spacing);
    if (Number.isFinite(minSpacing) && minSpacing > 0) {
      scene.physicalSampleDistance = minSpacing * 0.5;
      setCameraAdjustedVolumeRayDistances(
        scene.cloudVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.physicalSampleDistance,
        sceneScaleRef.current,
        scene.cloudMapper,
      );
    }
    applyCloudTransfer(
      scene.contextCloudCtf,
      scene.contextCloudPwf,
      cloudOpacityRef.current,
      floor,
      cloudEdgeRef.current,
      cloudSceneColors(
        cloudVisual.tone,
        cloudVisual.customColor,
        cloudVisual.brightness,
        cloudEdgeRef.current,
        sceneAppearanceRef.current,
      ),
    );
    applyCloudTransfer(
      scene.cloudCtf,
      scene.cloudPwf,
      cloudOpacityRef.current,
      floor,
      cloudEdgeRef.current,
      cloudSceneColors(
        cloudVisual.tone,
        cloudVisual.customColor,
        cloudVisual.brightness,
        cloudEdgeRef.current,
        sceneAppearanceRef.current,
      ),
    );
    applyVolumeLighting(
      scene.cloudVolume.getProperty(),
      sceneAppearanceRef.current.lightingOptions,
      "cloud",
    );
    scene.cloudVolume.setVisibility(true);
    scene.contextCloudMapper.modified();
    scene.cloudMapper.modified();

    maybeResetCamera(
      scene,
      cameraInitializedRef,
      prepared.surfaceFocusBounds,
      sceneScaleRef.current,
    );
    setCameraAdjustedVolumeRayDistances(
      scene.cloudVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.physicalSampleDistance,
      sceneScaleRef.current,
      scene.cloudMapper,
    );
  }, [
    cloud.field,
    precipTotal,
    coherentFrame,
    subtractPrecip,
    cloudPreparation.prepared,
    committedCloudFloorGkg,
    showCloud,
    renderFidelityMode,
    scenePresentationPending,
    vtkLifecycleGeneration,
  ]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (scenePresentationPending) return;
    applyPrecipTransfer(
      scene.precipCtf,
      scene.precipPwf,
      precipOpacity,
      precipFloorGkg / 1000,
      precipitationSceneColor(
        precipMode,
        precipCustomColor,
        sceneAppearanceRef.current,
      ),
    );
    scene.precipMapper.modified();
  }, [
    precipOpacity,
    precipFloorGkg,
    precipMode,
    precipCustomColor,
    showPrecip,
    scenePresentationPending,
    synchronizedPresentation,
    vtkLifecycleGeneration,
  ]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (scenePresentationPending) return;
    const floor = clampCloudFloor(cloudFloorGkg / 1000);
    applyCloudTransfer(
      scene.contextCloudCtf,
      scene.contextCloudPwf,
      effectiveCloudOpacity,
      floor,
      cloudEdge,
      cloudSceneColors(
        cloudTone,
        cloudCustomColor,
        cloudBrightness,
        cloudEdge,
        sceneAppearanceRef.current,
      ),
    );
    applyCloudTransfer(
      scene.cloudCtf,
      scene.cloudPwf,
      effectiveCloudOpacity,
      floor,
      cloudEdge,
      cloudSceneColors(
        cloudTone,
        cloudCustomColor,
        cloudBrightness,
        cloudEdge,
        sceneAppearanceRef.current,
      ),
    );
    scene.contextCloudMapper.modified();
    scene.cloudMapper.modified();
  }, [
    effectiveCloudOpacity,
    cloudFloorGkg,
    cloudEdge,
    cloudTone,
    cloudCustomColor,
    cloudBrightness,
    showCloud,
    scenePresentationPending,
    synchronizedPresentation,
    vtkLifecycleGeneration,
  ]);


  // ---------- Updraft / downdraft isosurface update ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (scenePresentationPending) return;
    if (!showUpdraft) {
      scene.upActor.setVisibility(false);
      scene.downActor.setVisibility(false);
      setFlowContourError(null);
      return;
    }
    // Keep the last coherent geometry visible while the next field and its
    // prepared lattice settle. Clearing here creates a blank flash between
    // every pair of playback frames.
    if (!updraft.field || !flowPrepared || !loadMatchesRenderedStep(updraft)) return;
    if (!updraftSurfaceAvailable && !downdraftSurfaceAvailable) {
      applySurfaceGeometry(scene.upData, null);
      applySurfaceGeometry(scene.downData, null);
      scene.upActor.setVisibility(false);
      scene.downActor.setVisibility(false);
      setFlowContourError(
        `No vertical velocity reaches ±${flowThreshold} m/s in this frame`
          + ` — lower the |w| threshold to draw a surface.`,
      );
      return;
    }
    if (!flowFieldReady) return;
    if (flowExtraction.error) {
      scene.upActor.setVisibility(false);
      scene.downActor.setVisibility(false);
      setFlowContourError(
        `Vertical-velocity contour extraction failed: ${flowExtraction.error}`,
      );
      return;
    }
    const result = flowExtraction.result;
    if ((updraftSurfaceAvailable || downdraftSurfaceAvailable) && !result) return;
    applySurfaceGeometry(scene.upData, result?.core ?? null);
    applySurfaceGeometry(scene.downData, result?.opposite ?? null);
    scene.upActor.setVisibility((result?.core?.triangles ?? 0) > 0);
    scene.downActor.setVisibility((result?.opposite?.triangles ?? 0) > 0);
    // A threshold above the frame's peak w is the common case early in a run,
    // and produces the same empty canvas as a broken layer.
    setFlowContourError(
      (result?.core?.triangles ?? 0) + (result?.opposite?.triangles ?? 0) === 0
        ? `No vertical velocity reaches ±${flowThreshold} m/s in this frame`
          + ` — lower the |w| threshold to draw a surface.`
        : null,
    );
    maybeResetCamera(scene, cameraInitializedRef);
  }, [
    updraft.field,
    flowFieldReady,
    flowPrepared,
    showUpdraft,
    flowThreshold,
    updraftSurfaceAvailable,
    downdraftSurfaceAvailable,
    flowExtraction.result,
    flowExtraction.error,
    scenePresentationPending,
    vtkLifecycleGeneration,
  ]);

  // ---------- Vortex core isosurface update ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (scenePresentationPending) return;
    const hideCondensation = () => {
      scene.condensationVolume.setVisibility(false);
      setTornadicCondensationAssessment(null);
    };
    if (!showVortexCore) {
      scene.coreActor.setVisibility(false);
      scene.oppositeActor.setVisibility(false);
      scene.mesocycloneActor.setVisibility(false);
      setMesocycloneAssessment(null);
      setVortexAssessment(null);
      hideCondensation();
      setVortexContourError(null);
      return;
    }
    // The field hooks intentionally retain only coherent frames. Preserve the
    // last completed core during the brief read/preparation gap instead of
    // hiding all three actors and flashing the scene on every playback tick.
    if (!vortexCoreField.field || !vortexPrepared || !loadMatchesRenderedStep(vortexCoreField)) return;
    if (vortexCoreThreshold == null || !requestedVortexContours) {
      applySurfaceGeometry(scene.coreData, null);
      applySurfaceGeometry(scene.oppositeData, null);
      applySurfaceGeometry(scene.mesocycloneData, null);
      scene.coreActor.setVisibility(false);
      scene.oppositeActor.setVisibility(false);
      scene.mesocycloneActor.setVisibility(false);
      setMesocycloneAssessment(null);
      setVortexAssessment(null);
      hideCondensation();
      setVortexContourError(
        `No resolved ${vortexCoreFieldName ?? "vortex diagnostic"} extremum is available in this frame.`,
      );
      return;
    }
    if (!vortexContourExpected) {
      applySurfaceGeometry(scene.coreData, null);
      applySurfaceGeometry(scene.oppositeData, null);
      applySurfaceGeometry(scene.mesocycloneData, null);
      scene.coreActor.setVisibility(false);
      scene.oppositeActor.setVisibility(false);
      scene.mesocycloneActor.setVisibility(false);
      setMesocycloneAssessment(null);
      setVortexAssessment(null);
      hideCondensation();
      setVortexContourError(
        `No ${vortexCoreFieldName} reaches ±${formatCoreThreshold(vortexCoreThreshold)}`
          + " in this frame — lower the core threshold to draw a surface.",
      );
      return;
    }
    if (!vortexFieldReady) return;
    if (vortexExtraction.error) {
      scene.coreActor.setVisibility(false);
      scene.oppositeActor.setVisibility(false);
      scene.mesocycloneActor.setVisibility(false);
      setMesocycloneAssessment(null);
      setVortexAssessment(null);
      hideCondensation();
      setVortexContourError(
        `Vortex contour extraction failed: ${vortexExtraction.error}`,
      );
      return;
    }
    const result = vortexExtraction.result;
    if ((vortexCoreAvailable || vortexMesocycloneAvailable) && !result) return;
    if (vortexOppositeAvailable && !result) return;

    applySurfaceGeometry(scene.coreData, result?.core ?? null);
    applySurfaceGeometry(scene.oppositeData, result?.opposite ?? null);
    applySurfaceGeometry(scene.mesocycloneData, result?.mesocyclone ?? null);
    scene.coreActor.setVisibility((result?.core?.triangles ?? 0) > 0);
    scene.oppositeActor.setVisibility((result?.opposite?.triangles ?? 0) > 0);
    scene.mesocycloneActor.setVisibility(
      result?.mesocyclone != null && result.mesocyclone.triangles > 0
        && Boolean(result.mesocycloneAssessment?.present),
    );
    setMesocycloneAssessment(result?.mesocycloneAssessment ?? null);
    setVortexAssessment(result?.vortexAssessment ?? null);
    setTornadicCondensationAssessment(result?.condensationAssessment ?? null);
    const condensation = result?.condensation;
    const showResolvedCondensation = showTornadicCondensation
      && condensation != null
      && result?.condensationAssessment?.visible === true;
    if (showResolvedCondensation && condensation) {
      setPreparedImageScalars(scene.condensationImage, condensation);
      setPreparedImageDisplayGeometry(
        scene.condensationImage,
        condensation,
        sceneScaleRef.current,
      );
      const minSpacing = Math.min(...condensation.spacing);
      if (Number.isFinite(minSpacing) && minSpacing > 0) {
        scene.condensationPhysicalSampleDistance = minSpacing * 0.5;
        scene.quality.setFullSampleDistanceFor(
          scene.condensationMapper,
          scene.condensationPhysicalSampleDistance,
        );
      }
      applyCloudTransfer(
        scene.condensationCtf,
        scene.condensationPwf,
        1.35,
        TORNADIC_CONDENSATE_FLOOR_KGKG,
        "subtle",
        cloudSceneColors(
          "custom",
          "#c7d6e6",
          1.08,
          "subtle",
          sceneAppearanceRef.current,
        ),
      );
      applyVolumeLighting(
        scene.condensationVolume.getProperty(),
        sceneAppearanceRef.current.lightingOptions,
        "cloud",
      );
      setCameraAdjustedVolumeRayDistances(
        scene.condensationVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.condensationPhysicalSampleDistance,
        sceneScaleRef.current,
        scene.condensationMapper,
      );
      scene.condensationVolume.setVisibility(true);
      scene.condensationMapper.modified();
    } else {
      scene.condensationVolume.setVisibility(false);
    }
    const surfaceCount = (result?.core?.triangles ?? 0)
      + (result?.opposite?.triangles ?? 0)
      + (result?.mesocyclone?.triangles ?? 0);
    setVortexContourError(
      surfaceCount === 0
        ? `No ${vortexCoreFieldName} reaches ±${formatCoreThreshold(vortexCoreThreshold)}`
          + ` in this frame — lower the core threshold to draw a surface.`
        : null,
    );
    maybeResetCamera(scene, cameraInitializedRef);
  }, [
    vortexCoreField.field,
    vortexFieldReady,
    vortexPrepared,
    showVortexCore,
    showTornadicCondensation,
    vortexCoreFieldName,
    vortexCoreThreshold,
    requestedVortexContours,
    vortexContourExpected,
    vortexExtraction.result,
    vortexExtraction.error,
    vortexCoreAvailable,
    vortexOppositeAvailable,
    vortexMesocycloneAvailable,
    scenePresentationPending,
    cloudPreparation.prepared,
    precipPreparation.prepared,
    flowExtraction.result,
    vtkLifecycleGeneration,
  ]);



  // ---------- Coherent scene presentation ----------
  useEffect(() => {
    if (scenePresentationPending) return;
    const scene = sceneRef.current;
    if (!scene) return;
    const releaseAnnotations = holdVtkSceneRendering(scene.renderWindow);
    try {
      radarFloor.commitData();
      prepareSceneAxesRef.current?.();
    } finally { releaseAnnotations(); }
    const renderedStep = loadedStep ?? (radarFrameExpected && !radarFloor.loading ? planningStep : null);
    const commitFrame = (elapsedMs?: number) => {
      if (renderedStep) {
        setCommittedSceneIdentity(sceneGenerationIdentity);
        setCommittedContextGrid(contextGrid);
        if (scaleDisclosureRef.current?.generation === sceneGenerationIdentity) {
          setCommittedScaleDisclosure(scaleDisclosureRef.current.value);
        }
        committedCameraTargetRef.current = trackedStormCenter ? [trackedStormCenter[0], trackedStormCenter[1]] : null;
        committedAppearanceRef.current = {
          ...appearanceEvidenceRef.current,
          frameIndex: renderedStep.index,
          parameters: { ...appearanceEvidenceRef.current.parameters,
            renderedPublicationGeneration: renderedStep.publicationGeneration,
            sceneGenerationIdentity,
            floorRequestIdentity: floorSelection.enabled ? radarFloor.loadedRequestIdentity : null,
            floorSourceField: floorSelection.enabled ? radarFloor.source?.fieldName ?? null : null,
            floorScalarRange: floorSelection.enabled && radarFloor.stats
              ? [radarFloor.stats.min, radarFloor.stats.max] : null,
            floorDisplayRange: floorSelection.enabled && radarFloor.policy
              ? [radarFloor.policy.min, radarFloor.policy.max] : null,
          },
        };
      }
      setCloudMotionActive(false);
      const motionReason = scene.motion.describe().reason;
      if (cloudMotionEnabled || motionReason) setCloudMotionReason(motionReason);
      if (motionReason?.startsWith("Cloud interpolation unavailable")) setCloudMotionEnabled(false);
      setInterpolatedDwellMs(previous => elapsedMs ?? (renderedStep?.index === renderedStepIndex
        && renderedStep?.publicationGeneration === renderedPublicationGeneration ? previous : 0));
      markFrameRendered(contextGeneration);
      if (renderedStep) {
        setRenderedStepIndex(renderedStep.index);
        setRenderedPublicationGeneration(renderedStep.publicationGeneration);
      }
    };
    const prepared = activeCloudVolumePrepared;
    releaseSceneCommitRef.current?.();
    releaseSceneCommitRef.current = null;
    const center = rootContextPreparation.prepared?.cloudCenter ?? cloudPreparation.prepared?.cloudCenter;
    if (renderedStep && prepared && center) {
      const started = scene.motion.commit({
        frame: {
          key: JSON.stringify([sourceRunDir, sourceDefinitionIdentity, renderedStep.index, renderedStep.publicationGeneration]),
          domainKey: JSON.stringify([runDir, cloudFieldName, cloudOpacity, committedCloudFloorGkg,
            cloudEdge, cloudTone, cloudCustomColor, cloudBrightness, sceneAppearance.resolved, sceneScale, cloudWindEnabled, useWindDensity]),
          step: hierarchyRootPlanningStep?.index ?? renderedStep.index, timeS: renderedStep.simulationTimeS ?? NaN, center,
          coordinateError: cloudCoordinateError(renderedStep) ?? cloudCoordinateError(contextStep),
          cloud: prepared, context: cloudContextActive ? rootContextPreparation.prepared?.volume ?? null : null,
          sources: [
            { role: "cloud", runDir: sourceRunDir, step: renderedStep.index, publicationGeneration: renderedStep.publicationGeneration,
              shape: prepared.shape, spacingM: prepared.spacing, originM: prepared.origin },
            ...(cloudContextActive && rootContextPreparation.prepared && contextStep ? [{ role: "context" as const,
              runDir: contextRunDir, step: contextStep.index, publicationGeneration: contextStep.publicationGeneration,
              shape: rootContextPreparation.prepared.volume.shape, spacingM: rootContextPreparation.prepared.volume.spacing,
              originM: rootContextPreparation.prepared.volume.origin }] : []),
          ],
          ...(cloudWindEnabled ? { wind: planningStep && windWindow && windU.field && windV.field && windW.field && (!useWindDensity || windRho.field)
            && [windU, windV, windW, ...(useWindDensity ? [windRho] : [])].every(load => load.loadedStepIndex === planningStep.index
              && load.loadedPublicationGeneration === planningStep.publicationGeneration)
            ? estimateCloudWind(planningStep, windWindow, { u: windU.field, v: windV.field, w: windW.field, rho: useWindDensity ? windRho.field : null }, cloudPreparation.prepared?.volume ?? prepared,
                cloudContextActive ? rootContextPreparation.prepared?.volume ?? null : null)
            : { ok: false as const, reason: windUnavailable ?? windU.error ?? windV.error ?? windW.error ?? windRho.error ?? "Wind samples are unavailable within the read budget" } } : {}),
        },
        enabled: cloudMotionAllowed, durationMs: interpolationDurationMs, scale: sceneScale,
        current: [scene.contextCloudVolume, scene.cloudVolume],
        onStart: () => { setCloudMotionActive(true); setRenderedStepIndex(null); setInterpolatedDwellMs(0); },
        onFinish: commitFrame,
        onProgress: (fraction, timeS) => {
          const host = containerRef.current?.closest<HTMLElement>(".plot3d-overlay-host");
          if (host) { host.dataset.cloudMotionFraction = String(fraction); host.dataset.cloudMotionTimeS = String(timeS); }
          if (cloudMotionTimeRef.current) cloudMotionTimeRef.current.textContent = timeS.toFixed(2);
        },
      });
      setCloudMotionReason(scene.motion.describe().reason);
      if (started) return;
    }
    scene.renderWindow.getRenderWindow().render();
    commitFrame();
  }, [
    sceneGenerationIdentity,
    contextGrid,
    scenePresentationPending,
    cloudMotionAllowed, interpolationDurationMs, activeCloudVolumePrepared, rootContextPreparation.prepared,
    cloudWindEnabled, contextStep, windWindow, windU.field, windV.field, windU.error, windV.error, windW.field, windW.error, windRho.field, windRho.error, useWindDensity, windUnavailable,
    sourceRunDir, sourceDefinitionIdentity, cloudFieldName, cloudOpacity, committedCloudFloorGkg, cloudEdge, cloudTone,
    cloudCustomColor, cloudBrightness, sceneAppearance.resolved, sceneScale.xy, sceneScale.z,
    vortexSurfaceColors.core, vortexSurfaceColors.mesocyclone, vortexSurfaceColors.opposite,
    radarFloor.commitData,
    showCloud,
    cloudPreparation.prepared,
    cloudPreparation.error,
    showPrecip,
    precipPreparation.prepared,
    precipPreparation.error,
    showUpdraft,
    flowExtraction.result,
    flowExtraction.error,
    updraftSurfaceAvailable,
    downdraftSurfaceAvailable,
    showVortexCore,
    vortexExtraction.result,
    vortexExtraction.error,
    vortexContourExpected,
    radarFrameExpected,
    floorSelection.enabled,
    radarFloor.values,
    radarFloor.loadedRequestIdentity,
    radarFloor.source,
    radarFloor.stats,
    radarFloor.policy,
    radarFloor.error,
    radarFloor.loading,
    loadedStep,
    planningStep,
    contextGeneration,
    markFrameRendered,
    vtkLifecycleGeneration,
  ]);

  /* ---------- Layer opacity (separate effect: opacity changes must not re-run marching cubes) ---------- */
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const secondary = SIGNED_PAIR_SECONDARY_OPACITY_RATIO;
    scene.upActor.getProperty().setOpacity(updraftOpacity);
    scene.downActor.getProperty().setOpacity(updraftOpacity * secondary);
    scene.coreActor.getProperty().setOpacity(vortexCoreOpacity);
    scene.oppositeActor.getProperty().setOpacity(vortexCoreOpacity * secondary);
    scene.mesocycloneActor.getProperty().setOpacity(vortexCoreOpacity * 0.28);
    scene.renderWindow.getRenderWindow().render();
  }, [updraftOpacity, vortexCoreOpacity, vtkLifecycleGeneration]);

  // ---------- Vertical exaggeration ----------
  // Polygon layers use actor scale. The volume's ImageData geometry is scaled
  // directly because vtk's shader-loop bound does not include actor scale.
  useLayoutEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    // Commit every layer before passive material/camera effects can redraw it.
    const release = holdVtkSceneRendering(scene.renderWindow);
    try {
      scene.motion.reset();
      setCloudMotionActive(false);
      setInterpolatedDwellMs(0);
      const currentScale = { xy: sceneScale.xy, z: sceneScale.z };
      scene.contextCloudVolume.setScale(1, 1, 1);
      scene.cloudVolume.setScale(1, 1, 1);
      scene.condensationVolume.setScale(1, 1, 1);
      scene.precipVolume.setScale(1, 1, 1);
      applySceneScale(
        currentScale,
        scene.upActor,
        scene.downActor,
        scene.coreActor,
        scene.oppositeActor,
        scene.mesocycloneActor,
      );
      const volumePrepared = activeCloudVolumePreparedRef.current;
      if (volumePrepared) {
        setPreparedImageDisplayGeometry(scene.cloudImage, volumePrepared, currentScale);
        scene.cloudMapper.modified();
      }
      const contextPrepared = activeRootContextPreparedRef.current;
      if (contextPrepared) {
        setPreparedImageDisplayGeometry(scene.contextCloudImage, contextPrepared, currentScale);
        scene.contextCloudMapper.modified();
      }
      const precipPrepared = activePrecipPreparedRef.current;
      if (precipPrepared) {
        setPreparedImageDisplayGeometry(scene.precipImage, precipPrepared, currentScale);
        scene.precipMapper.modified();
      }
      const condensationPrepared = activeCondensationPreparedRef.current;
      if (condensationPrepared) {
        setPreparedImageDisplayGeometry(
          scene.condensationImage,
          condensationPrepared,
          currentScale,
        );
        scene.condensationMapper.modified();
      }
      setCameraAdjustedVolumeRayDistances(
        scene.contextCloudVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.contextPhysicalSampleDistance,
        currentScale,
        scene.contextCloudMapper,
      );
      setCameraAdjustedVolumeRayDistances(
        scene.cloudVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.physicalSampleDistance,
        currentScale,
        scene.cloudMapper,
      );
      setPrecipRayDistances(
        scene.precipVolume,
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        scene.precipPhysicalSampleDistance,
        currentScale,
        scene.precipMapper,
      );
      setCameraAdjustedVolumeRayDistances(
        scene.condensationVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        CLOUD_OPACITY_REFERENCE_M,
        scene.condensationPhysicalSampleDistance,
        currentScale,
        scene.condensationMapper,
      );
      if (cameraInitializedRef.current) {
        const renderer = scene.renderWindow.getRenderer();
        const camera = renderer.getActiveCamera();
        const previousFocus = detailTargetMode === "view" ? [...camera.getFocalPoint()] : null;
        refitVtkCameraToBounds(
          scene.renderWindow.getRenderer(),
          supercellFocusBounds(
            scene,
            resetFocusBoundsRef.current,
            currentScale,
          ),
        );
        if (previousFocus) {
          const focal = camera.getFocalPoint(), position = camera.getPosition();
          camera.setPosition(position[0] + previousFocus[0] - focal[0], position[1] + previousFocus[1] - focal[1], position[2]);
          camera.setFocalPoint(previousFocus[0], previousFocus[1], focal[2]);
          renderer.resetCameraClippingRange();
        }
      }
    } finally { release(); }
    scene.renderWindow.getRenderWindow().render();
  }, [sceneScale.xy, sceneScale.z, vtkLifecycleGeneration]);

  // ---------- Reset camera ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    resetVtkCameraToStormView(
      scene.renderWindow.getRenderer(),
      supercellFocusBounds(
        scene,
        resetFocusBoundsRef.current,
        sceneScaleRef.current,
      ),
    );
    setCameraAdjustedVolumeRayDistances(
      scene.contextCloudVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.contextPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.contextCloudMapper,
    );
    setCameraAdjustedVolumeRayDistances(
      scene.cloudVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.physicalSampleDistance,
      sceneScaleRef.current,
      scene.cloudMapper,
    );
    setCameraAdjustedVolumeRayDistances(
      scene.condensationVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.condensationPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.condensationMapper,
    );
    setPrecipRayDistances(
      scene.precipVolume,
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      scene.precipPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.precipMapper,
    );
    scene.renderWindow.getRenderWindow().render();
  }, [resetTick]);

  const anyLayerLoading =
    cloud.loading || rain.loading || graupel.loading || hail.loading ||
    updraft.loading || vortexCoreField.loading ||
    primaryCondensate.loading || frozenCondensate.loading ||
    vortexCorePreparation.loading ||
    tornadicCondensatePreparation.loading ||
    flowLayerLoading || vortexLayerLoading ||
    cloudPreparation.loading || precipPreparation.loading ||
    radarFloor.loading || updraftPreparation.loading;
  const preparationError = sceneGeneration.error || flowContourError || vortexContourError ||
    supercellGpuUpload.error ||
    (showUpdraft ? flowDiagnosticPlanError : null) ||
    (showVortexCore ? vortexDiagnosticPlanError : null) ||
    cloudPreparation.error ||
    precipPreparation.error || radarFloor.error || updraftPreparation.error ||
    vortexCorePreparation.error || tornadicCondensatePreparation.error ||
    primaryCondensate.error || frozenCondensate.error;
  const missingAll = !cloudFieldName && floorSelection.planeFields.length === 0
    && availablePrecipSpecies.length === 0
    && !updraftFieldName && vortexCoreFields.length === 0;
  const exactSceneCommitted = renderedStepIndex === requestedStepIndex
    && renderedPublicationGeneration === requestedPublicationGeneration
    && committedSceneIdentity === sceneGenerationIdentity
    && !scenePresentationPending;
  const vortexLegendCommitted = exactSceneCommitted && showVortexCore
    && !vortexLayerLoading && vortexContourError == null
    && vortexExtraction.error == null && (vortexCoreLegendItems?.length ?? 0) > 0;
  const precipLegendCommitted = exactSceneCommitted && showPrecip
    && !precipPreparation.loading && precipPreparation.error == null
    && precipPreparation.prepared != null;
  const vortexLegend = useCommittedLegend(
    runDir,
    exactSceneCommitted,
    vortexLegendCommitted && vortexCoreFieldName && vortexCoreLegendItems
      ? { title: `${vortexCoreFieldName} vortex hierarchy`, items: vortexCoreLegendItems } : null,
  );
  const precipLegend = useCommittedLegend(
    runDir,
    exactSceneCommitted,
    precipLegendCommitted ? { title: "Precipitation volume", items: [{
      color: rgbCss(precipitationSceneColor(precipMode, precipCustomColor, sceneAppearance.resolved)),
      label: precipMode === "total" ? availablePrecipSpecies.join(" + ") : precipMode === "qr" ? "rain mixing ratio (qr)"
        : precipMode === "qg" ? "graupel mixing ratio (qg)" : "hail mixing ratio (qh)",
    }] } : null,
  );
  const floorLegend = useCommittedLegend(
    runDir,
    exactSceneCommitted && !radarFloor.loading && !radarFloor.error
      && radarFloor.committedPresentationIdentity === radarFloor.presentationIdentity,
    radarFloor.visible && floorSelection.fieldName && radarFloor.policy ? {
      colormap: radarFloor.policy.colormap, min: radarFloor.policy.min, max: radarFloor.policy.max,
      label: `${floorSelection.fieldName} floor`, unit: unitFor(floorSelection.fieldName, loadedStep?.fieldUnits),
    } : null,
  );
  useEffect(() => {
    const exact = renderedStepIndex === requestedStepIndex
      && renderedPublicationGeneration === requestedPublicationGeneration;
    onFrameDisplayChange?.({
      requestedStepIndex,
      requestedPublicationGeneration,
      renderedStepIndex,
      renderedPublicationGeneration,
      loading: !preparationError && !missingAll && (scenePresentationPending || cloudMotionActive || anyLayerLoading || !exact),
      interpolatedDwellMs: cloudMotionActive ? 0 : interpolatedDwellMs,
    });
  }, [
    onFrameDisplayChange,
    requestedStepIndex,
    requestedPublicationGeneration,
    renderedStepIndex,
    renderedPublicationGeneration,
    preparationError,
    missingAll,
    anyLayerLoading, cloudMotionActive, interpolatedDwellMs,
    scenePresentationPending,
  ]);
  const cloudDetail = cloudPreparation.prepared
    ? [
        "aggregate suspended condensate",
        `single-pass volume · physical opacity per ${CLOUD_OPACITY_REFERENCE_M / 1_000} km · ${
          cloudEdge === "off" ? "no edge emphasis" : `${cloudEdge} volume edge`
        }`,
      ].filter(Boolean).join(" · ")
    : null;
  const stormDetail = ((showCloud && stormVolumeResolution === "storm") || showPrecip)
    ? stormDetailOutcome.ok
      ? `storm-detail source ${stormDetailOutcome.plan.sourceShape.join("×")}`
        + ` native cells · ${stormDetailOutcome.plan.detailSpanM
          .map((span) => `${Number((span / 1_000).toPrecision(3))}`)
          .join("×")} km ROI · ${activeViewTarget ? "view-centred" : trackedStormCenter ? "tracker-centred" : "field-located"}`
        + `${nativeStormVolumes ? " · cloud/precip native ROI" : " · cloud/precip full-domain LOD"}`
        + `${cloudContextActive ? ` · full-domain cloud context${publishedRootContextCloud.disclosure ? ` · ${publishedRootContextCloud.disclosure}` : ""}` : ""}`
        + `${forceNativeStormDetail ? " · full-domain raw scan blocked" : ""}`
      : stormDetailCenter
        ? `bounded domain fallback · native storm detail unavailable: ${stormDetailOutcome.reason}`
        : "bounded domain locator · native storm detail follows"
    : null;
  const flowDetail = showUpdraft
    ? flowUsesExactLattice && flowDiagnosticOutcome.ok
      ? `flow native source ${flowDiagnosticOutcome.plan.sourceShape.join("×")}`
        + " cells · independent updraft-extremum ROI"
      : flowSupportBounds && flowDiagnosticOutcome.ok
        ? `flow complete-support cell-mean ${flowDiagnosticOutcome.plan.stride.join("×")}`
          + ` · display ${flowDiagnosticOutcome.plan.sampledShape.join("×")} · native detail exceeds the scene allocation`
        : "flow extrema-preserving domain locator · native ROI follows"
    : null;
  const vortexDetail = showVortexCore
    ? vortexUsesExactLattice && vortexDiagnosticOutcome.ok
      ? `vortex native source ${vortexDiagnosticOutcome.plan.sourceShape.join("×")}`
        + " cells · independent signed-extremum ROI"
      : "vortex extrema-preserving domain locator · native ROI follows"
    : null;
  const scaleDetail = [
    cloudMotionActive ? "Interpolated cloud preview · reduced screen sampling · not a saved model state" : null,
    activeOutputLevel?.depth
      ? `${outputHierarchySource.selection?.disclosure ?? activeOutputLevel.levelId}`
        + " · selected branch owns storm fields · root retained as Earth-frame context"
      : outputHierarchySource.error
        ? `nested history withheld: ${outputHierarchySource.error}`
        : outputHierarchySource.selection?.fallbackReason
          ? outputHierarchySource.selection.fallbackReason
          : null,
    rootContextPreparation.error
      ? `cloud context withheld: ${rootContextPreparation.error}`
      : null,
    cloudDetail,
    showPrecip && precipPreparation.prepared
      ? `${precipMode === "total" ? "total falling hydrometeors" : precipMode} precipitation volume`
        + ` · physical opacity per ${PRECIP_OPACITY_REFERENCE_M / 1_000} km`
      : null,
    stormDetail,
    flowDetail,
    vortexDetail,
    diagnosticRevealActive ? "cloud opacity auto-revealed for internal diagnostics" : null,
    showVortexCore && vortexStructure === "hierarchy" && vortexCoreFieldName === "vorticity_z"
      ? vortexExtraction.result?.mesocycloneAssessment?.present
        ? `connected mesocyclone-scale envelope ≥ ${formatCoreThreshold(mesocycloneThreshold ?? 0)} s⁻¹`
          + ` · ${(vortexExtraction.result.mesocycloneAssessment.depthM / 1_000).toFixed(1)} km deep`
        : formatMesocycloneAssessment(
            vortexExtraction.result?.mesocycloneAssessment ?? null,
            mesocycloneThreshold,
            mesocycloneContext,
          )
      : null,
    showVortexCore && vortexCoreFieldName === "vorticity_z"
      ? formatTornadoVortexAssessment(vortexExtraction.result?.vortexAssessment ?? null)
      : null,
    showTornadicCondensation
      ? `${formatTornadicCondensationAssessment(vortexExtraction.result?.condensationAssessment ?? null)}`
        + " · solved condensate only · vorticity-proximity opacity mask"
      : null,
    `renderer ${vtkBackend.backend === "webgpu" ? "WebGPU" : "WebGL"}`,
    showVortexCore && vortexExtraction.result
      ? `vortex geometry ${Math.round(vortexExtraction.result.elapsedMs)} ms off-thread`
        + ` · ${vortexExtraction.result.coreComponentCount} same-sign component${
          vortexExtraction.result.coreComponentCount === 1 ? "" : "s"
        } retained`
      : null,
    floorSelection.enabled && radarFloor.policy ? describeScalarPlane(radarFloor.policy) : null,
  ].filter(Boolean).join(" · ") || null;
  const layerSampling = [
    showCloud && renderGrid?.geometryKind !== "index"
      ? describeLayerVtkSampling("Cloud detail", activeCloudVolumePrepared?.samplingProvenance) : null,
    showCloud && cloudContextActive && contextStep?.grid?.geometryKind !== "index"
      ? describeLayerVtkSampling("Cloud context", rootContextPreparation.prepared?.volume.samplingProvenance) : null,
    showPrecip && renderGrid?.geometryKind !== "index"
      ? describeLayerVtkSampling("Precipitation", precipPreparation.prepared?.samplingProvenance) : null,
    showUpdraft && renderGrid?.geometryKind !== "index"
      ? describeLayerVtkSampling("Flow", updraftPreparation.metadata?.samplingProvenance) : null,
    showVortexCore && renderGrid?.geometryKind !== "index"
      ? describeLayerVtkSampling("Vortex", vortexCorePreparation.metadata?.samplingProvenance) : null,
    showTornadicCondensation && renderGrid?.geometryKind !== "index"
      ? describeLayerVtkSampling("Condensation", tornadicCondensatePreparation.metadata?.samplingProvenance) : null,
  ].filter((value): value is string => value != null);
  const disclosureGrid = renderGrid ? {
    nr: renderGrid.nr, nth: renderGrid.nth, nz: renderGrid.nz,
    drM: renderGrid.drM, dyM: renderGrid.dyM, dthetaRad: renderGrid.dthetaRad, dzM: renderGrid.dzM,
    geometryKind: renderGrid.geometryKind, coordinateSystem: renderGrid.coordinateSystem,
    terrainFollowing: renderGrid.terrainFollowing,
    physicalBoundsM: renderGrid.physicalBoundsM ? {
      lower: [...renderGrid.physicalBoundsM.lower] as [number, number, number],
      upper: [...renderGrid.physicalBoundsM.upper] as [number, number, number],
    } : undefined,
  } : null;
  scaleDisclosureRef.current = { generation: sceneGenerationIdentity, value: {
    grid: disclosureGrid,
    fidelityMode: cloudMotionActive ? "presentation" : renderFidelityMode,
    missingSupport: Boolean(showCloud && (activeCloudVolumePrepared?.validity || rootContextPreparation.prepared?.volume.validity)
      || showPrecip && precipPreparation.prepared?.validity
      || showUpdraft && (flowPrepared?.sourceNonFiniteCount ?? 0) > 0
      || showVortexCore && (vortexPrepared?.sourceNonFiniteCount ?? 0) > 0
      || showTornadicCondensation && vortexExtraction.result?.condensation?.validity),
    samplingProvenance: coarsestVtkSampling([
      showCloud && cloudContextActive ? rootContextPreparation.prepared?.volume.samplingProvenance : null,
      showCloud ? activeCloudVolumePrepared?.samplingProvenance : null,
      showPrecip ? precipPreparation.prepared?.samplingProvenance : null,
      showUpdraft ? updraftPreparation.metadata?.samplingProvenance : null,
      showVortexCore ? vortexCorePreparation.metadata?.samplingProvenance : null,
      showTornadicCondensation ? tornadicCondensatePreparation.metadata?.samplingProvenance : null,
    ]) ?? undefined,
    detail: [...layerSampling, scaleDetail].filter(Boolean).join(" · ") || null,
  } };
  const beginVolumeControlPreview = () => sceneRef.current?.quality.begin("interaction");
  const endVolumeControlPreview = () => sceneRef.current?.quality.end("interaction");
  const focusVortex = () => {
    const scene = sceneRef.current;
    const focus = unionVtkBounds(
      vortexExtraction.result?.core?.bounds,
      showTornadicCondensation ? vortexExtraction.result?.condensationBounds : null,
    );
    if (!scene || !focus) return;
    focusVtkCameraOnBounds(
      scene.renderWindow.getRenderer(),
      scaleBounds(focus as BoundsTuple, sceneScaleRef.current),
    );
    cameraInitializedRef.current = true;
    setCameraAdjustedVolumeRayDistances(
      scene.contextCloudVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.contextPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.contextCloudMapper,
    );
    setCameraAdjustedVolumeRayDistances(
      scene.cloudVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.physicalSampleDistance,
      sceneScaleRef.current,
      scene.cloudMapper,
    );
    setCameraAdjustedVolumeRayDistances(
      scene.condensationVolume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      CLOUD_OPACITY_REFERENCE_M,
      scene.condensationPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.condensationMapper,
    );
    setPrecipRayDistances(
      scene.precipVolume,
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      scene.precipPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.precipMapper,
    );
    scene.renderWindow.getRenderWindow().render();
  };
  const volumeSliderInteractionProps = {
    onPointerDown: beginVolumeControlPreview,
    onPointerUp: endVolumeControlPreview,
    onPointerCancel: endVolumeControlPreview,
    onBlur: endVolumeControlPreview,
    onKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]
        .includes(event.key)) beginVolumeControlPreview();
    },
    onKeyUp: endVolumeControlPreview,
  };
  const sceneStatus = sceneAppearance.selection.backgroundMode === "theme"
    ? "Theme"
    : `${sceneAppearance.selection.backgroundMode[0].toUpperCase()}${
        sceneAppearance.selection.backgroundMode.slice(1)
      }`;

  return (
    <div className="volume-render">
      <div className="volume-render__controls volume-render__controls--grouped">
        <div className="plot3d-control-toolbar" role="toolbar" aria-label="Rendered supercell controls">
          {(outputHierarchySource.availableLevels.length > 1
            || outputHierarchySource.loading
            || outputHierarchySource.error) && <SupercellControlTab
            group="domain"
            label="Domain"
            status={activeOutputLevel?.depth
              ? activeOutputLevel.step.grid?.drM != null
                ? `${activeOutputLevel.step.grid.drM.toLocaleString()} m`
                : "Fine"
              : "Root"}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />}
          <SupercellControlTab
            group="cloud"
            label="Cloud"
            status={!cloudFieldName
              ? "Unavailable"
              : showCloud
                ? nativeStormVolumes ? "Native storm" : "Domain"
                : "Off"}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />
          <SupercellControlTab
            group="precip"
            label="Precip"
            status={availablePrecipSpecies.length === 0
              ? "Unavailable"
              : showPrecip ? precipMode === "total" ? "Total" : precipMode.toUpperCase() : "Off"}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />
          <SupercellControlTab
            group="radar"
            label="Radar"
            status={floorSelection.planeFields.length === 0
              ? "Unavailable"
              : floorSelection.enabled ? "On" : "Off"}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />
          <SupercellControlTab
            group="flow"
            label="Flow"
            status={!updraftFieldName ? "Unavailable" : showUpdraft ? "On" : "Off"}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />
          <SupercellControlTab
            group="vortex"
            label="Vortex"
            status={!vortexCoreSourceAvailable
              ? "Unavailable"
              : showVortexCore
                ? synchronizedPresentation
                  ? showTornadicCondensation
                    ? vortexStructure === "hierarchy" ? "Funnel + meso" : "Funnel"
                    : vortexStructure === "hierarchy" ? "Core + meso" : "Core"
                : tornadicCondensationAssessment?.present
                  ? mesocycloneAssessment?.present ? "Funnel + meso" : "Funnel"
                  : tornadoScaleResolved
                  ? mesocycloneAssessment?.present ? "Tornado + meso" : "Tornado core"
                  : mesocycloneAssessment?.present ? "Vortex + meso" : "Vortex core"
                : "Off"}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />
          <SupercellControlTab
            group="scene"
            label="Scene"
            status={sceneStatus}
            openGroup={openControlGroup}
            onToggle={setOpenControlGroup}
          />
          <span
            className={`plot3d-control-notice${qualityPreview || playbackScreenDistance > 1 ? " is-visible" : ""}`}
            role="status"
            aria-hidden={qualityPreview || playbackScreenDistance > 1 ? undefined : true}
          >
            {qualityPreview ? "Preview while moving · settles to full quality"
              : "Playback screen preview · full resolution returns when paused"}
          </span>
          <ThreeDControlScopeToggle state={controlScope} />
        </div>
        {openControlGroup === "domain" && (
          <div
            id="supercell-controls-domain"
            className="plot3d-control-panel"
            role="region"
            aria-label="Domain detail controls"
          >
            <OutputHierarchyControls source={outputHierarchySource} />
            <PlotCompressionControls
              step={loadedStep}
              fields={[
                showCloud ? cloudFieldName : null,
                ...(showPrecip
                  ? precipMode === "total"
                    ? availablePrecipSpecies
                    : [precipMode]
                  : []),
                showUpdraft ? updraftFieldName : null,
                showVortexCore ? vortexCoreFieldName : null,
                floorSelection.enabled ? floorSelection.fieldName : null,
              ]}
            />
          </div>
        )}
        {openControlGroup === "cloud" && (
          <div
            id="supercell-controls-cloud"
            className="plot3d-control-panel"
            role="region"
            aria-label="Cloud controls"
          >
        <ControlGroup label="Cloud">
          <LayerToggle on={showCloud} onClick={() => setShowCloud((v) => !v)} disabled={!cloudFieldName}
            label={cloudFieldName ?? "n/a"} />
        </ControlGroup>
        <ControlGroup label="Cloud motion">
          <PillGroup
            options={[{ value: "saved", label: "Saved frames" }, { value: "interpolated", label: "Interpolate cloud", unavailable: renderFidelityMode !== "presentation"
              ? "Temporal preview generates inferred states and requires explicit Presentation mode. Scientific mode presents exact saved frames."
              : !cloudMotionAvailable || recording ? "Use a single view with cloud alone; split views, diagnostics, and recording use saved frames." : undefined },
              { value: "wind", label: "Wind-informed", unavailable: renderFidelityMode !== "presentation"
                ? "Temporal preview generates inferred states and requires explicit Presentation mode. Scientific mode presents exact saved frames."
                : !cloudMotionAvailable || recording
                ? "Use a single view with cloud alone; split views, diagnostics, and recording use saved frames."
                : windUnavailable ?? (windWindow ? undefined : "Wind samples exceed the bounded read budget") }]}
            value={cloudMotionEnabled ? cloudWindEnabled ? "wind" : "interpolated" : "saved"}
            onChange={value => {
              if (value !== "saved" && renderFidelityMode !== "presentation") return;
              setCloudMotionEnabled(value !== "saved"); setCloudWindEnabled(value === "wind"); setCloudMotionReason(null);
            }}
          />
          <span className="volume-render__slider-value">
            {renderFidelityMode !== "presentation" ? "Exact saved model states · temporal previews infer data and require Presentation mode"
              : cloudMotionActive ? "Interpolated display · not a saved model state"
              : cloudMotionEnabled ? !cloudMotionAvailable ? "Saved frames while diagnostics or split views are enabled"
                : recording ? "Recording saved frames" : !playing ? "Starts during playback"
                  : cloudMotionReason ?? "Waiting for the next cloud frame" : cloudMotionReason?.startsWith("Cloud interpolation unavailable") ? "Saved frames · interpolation unavailable" : "Exact saved model states"}
          </span>
        </ControlGroup>
        {cloudMotionEnabled && <p className="plot3d-control-notice">{cloudWindEnabled
          ? "Display preview fits three-dimensional transport, rotation, and shear from saved winds, with air-density weighting when available. Positions use Earth coordinates and saved terrain heights. It rejects unsupported deformation or cloud changes."
          : "Display preview aligns saved cloud centroids horizontally and blends density."} It retains native volume samples, uses simplified preview lighting, reduced screen sampling, and two cloud frames, and skips large changes. This is not resolved fluid motion.</p>}
        <ControlGroup label="Volume resolution">
          <PillGroup
            options={STORM_VOLUME_RESOLUTION_OPTIONS}
            value={stormVolumeResolution}
            onChange={setStormVolumeResolution}
            disabled={forceNativeStormDetail || (!showCloud && !showPrecip)}
            disabledReason={forceNativeStormDetail
              ? "This source is too large for an interactive raw full-domain scan; native storm detail is enforced."
              : "Enable cloud or precipitation to choose its three-dimensional volume extent."}
          />
          <span className="volume-render__slider-value">
            {stormVolumeResolution === "storm"
              ? nativeStormVolumes && stormDetailOutcome.ok
                ? `${stormDetailOutcome.plan.detailSpanM
                    .map((span) => Number((span / 1_000).toPrecision(3)))
                    .join("×")} km at native cells${forceNativeStormDetail
                      ? " · full-domain source scan blocked"
                      : ""}`
                : "Locating storm; full-domain LOD until a centre is available"
              : "Full-domain bounded LOD"}
          </span>
        </ControlGroup>
        <ControlGroup label="Native detail target">
          <PillGroup
            options={[{ value: "storm", label: "Storm", unavailable: playing || recording ? "Pause playback to change the detail target." : undefined },
              { value: "view", label: "View focus", unavailable: !viewDetailSupported ? "View targeting requires Cartesian coordinates." : playing || recording ? "Pause playback to change the detail target." : undefined }]}
            value={detailTargetMode}
            onChange={setDetailTargetMode}
            disabled={(!showCloud && !showPrecip) || (!forceNativeStormDetail && stormVolumeResolution !== "storm")}
            disabledReason="Use native storm resolution to position its detail region."
          />
          <span className="volume-render__slider-value">
            {detailTargetMode === "view" ? playing || recording ? "View target held during playback"
              : "Pan or focus the camera; detail updates after movement stops" : "Native cloud and precipitation follow the storm"}
          </span>
        </ControlGroup>
        <ControlGroup label="Cloud density">
          <input type="range" min={0.2} max={2.5} step={0.05} value={cloudOpacity}
            onChange={(e) => setCloudOpacity(Number.parseFloat(e.target.value))}
            {...volumeSliderInteractionProps}
            className="volume-render__slider" disabled={!showCloud || !cloudFieldName}
            title={disabledLayerHint(showCloud, cloudFieldName, "Cloud")
              ?? "Strengthen the cloud body without changing its condensate threshold"} />
          <span className="volume-render__slider-value">{cloudOpacity.toFixed(2)}×</span>
        </ControlGroup>
        <ControlGroup label="Internal diagnostics">
          <PillGroup
            options={DIAGNOSTIC_VISIBILITY_OPTIONS}
            value={diagnosticVisibility}
            onChange={setDiagnosticVisibility}
            disabled={!showCloud || !cloudFieldName}
            disabledReason={disabledLayerHint(showCloud, cloudFieldName, "Cloud")}
          />
          <span className="volume-render__slider-value">
            {diagnosticRevealActive
              ? `cloud display density ${effectiveCloudOpacity.toFixed(2)}× while Flow/Vortex is visible`
              : "Auto reveal changes display opacity only; solved fields and thresholds stay exact"}
          </span>
        </ControlGroup>
        <ControlGroup label="Cloud edge">
          <PillGroup
            options={CLOUD_EDGE_OPTIONS}
            value={cloudEdge}
            onChange={(value) => {
              beginVolumeControlPreview();
              setCloudEdge(value);
              endVolumeControlPreview();
            }}
            disabled={!showCloud || !cloudFieldName}
            disabledReason={disabledLayerHint(showCloud, cloudFieldName, "Cloud")}
          />
        </ControlGroup>
        <ControlGroup label="Cloud tone">
          <select
            className="volume-render__select"
            value={cloudTone}
            disabled={!showCloud || !cloudFieldName}
            title={disabledLayerHint(showCloud, cloudFieldName, "Cloud")}
            onChange={(event) => setCloudTone(event.target.value as CloudTone)}
          >
            {CLOUD_TONE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          {cloudTone === "custom" && (
            <input
              type="color"
              className="scene-color-input"
              value={cloudCustomColor}
              disabled={!showCloud || !cloudFieldName}
              aria-label="Custom cloud color"
              title="Custom cloud color"
              onChange={(event) => setCloudCustomColor(event.target.value)}
            />
          )}
        </ControlGroup>
        <ControlGroup label="Cloud brightness">
          <input
            type="range"
            min={0.3}
            max={1.2}
            step={0.05}
            value={cloudBrightness}
            onChange={(event) => setCloudBrightness(Number.parseFloat(event.target.value))}
            {...volumeSliderInteractionProps}
            className="volume-render__slider"
            disabled={!showCloud || !cloudFieldName}
            title={disabledLayerHint(showCloud, cloudFieldName, "Cloud")}
          />
          <span className="volume-render__slider-value">{cloudBrightness.toFixed(2)}×</span>
        </ControlGroup>
        <ControlGroup label="Cloud floor">
          <input type="range" min={0.05} max={1.2} step={0.05} value={cloudFloorGkg}
            onChange={(e) => setCloudFloorGkg(Number.parseFloat(e.target.value))}
            {...volumeSliderInteractionProps}
            className="volume-render__slider" disabled={!showCloud || !cloudFieldName}
            title={disabledLayerHint(showCloud, cloudFieldName, "Cloud")} />
          <span className="volume-render__slider-value">
            {cloudFloorGkg.toFixed(2)} g/kg
            {cloudFloorGkg !== committedCloudFloorGkg ? " · volume preview" : ""}
          </span>
        </ControlGroup>
          </div>
        )}
        {openControlGroup === "precip" && (
          <div
            id="supercell-controls-precip"
            className="plot3d-control-panel"
            role="region"
            aria-label="Precipitation controls"
          >
        <ControlGroup label="Precipitation">
          <LayerToggle
            on={showPrecip}
            onClick={() => setShowPrecip((value) => !value)}
            disabled={!showPrecip && availablePrecipSpecies.length === 0}
            label={availablePrecipSpecies.length > 0 ? "falling hydrometeors" : showPrecip ? "unavailable precipitation" : "n/a"}
          />
        </ControlGroup>
        <ControlGroup label="Precip field">
          <select
            className="volume-render__select"
            value={precipMode}
            disabled={!showPrecip || availablePrecipSpecies.length === 0}
            title={disabledLayerHint(
              showPrecip,
              availablePrecipSpecies[0] ?? null,
              "Precipitation",
            )}
            onChange={(event) => setPrecipMode(event.target.value as PrecipitationMode)}
          >
            <option value="total">Total</option>
            {precipMode !== "total" && !availablePrecipSpecies.includes(precipMode)
              && <option value={precipMode} disabled>{precipMode.toUpperCase()} · unavailable in this frame</option>}
            {availablePrecipSpecies.includes("qr") && <option value="qr">Rain</option>}
            {availablePrecipSpecies.includes("qg") && <option value="qg">Graupel</option>}
            {availablePrecipSpecies.includes("qh") && <option value="qh">Hail</option>}
          </select>
        </ControlGroup>
        <ControlGroup label="Precip color">
          <PillGroup
            options={[
              { value: "auto", label: "Species" },
              { value: "custom", label: "Custom" },
            ]}
            value={precipCustomColor == null ? "auto" : "custom"}
            onChange={(value) => setPrecipCustomColor(value === "auto" ? null : "#35a8d4")}
            disabled={!showPrecip || availablePrecipSpecies.length === 0}
            disabledReason={disabledLayerHint(
              showPrecip,
              availablePrecipSpecies[0] ?? null,
              "Precipitation",
            )}
          />
          {precipCustomColor != null && (
            <input
              type="color"
              className="scene-color-input"
              value={precipCustomColor}
              disabled={!showPrecip}
              aria-label="Custom precipitation color"
              title="Custom precipitation color"
              onChange={(event) => setPrecipCustomColor(event.target.value)}
            />
          )}
        </ControlGroup>
        <ControlGroup label="Precip opacity">
          <input
            type="range"
            min={0.1}
            max={1.5}
            step={0.05}
            value={precipOpacity}
            onChange={(event) => setPrecipOpacity(Number.parseFloat(event.target.value))}
            {...volumeSliderInteractionProps}
            className="volume-render__slider"
            disabled={!showPrecip || availablePrecipSpecies.length === 0}
            title={disabledLayerHint(
              showPrecip,
              availablePrecipSpecies[0] ?? null,
              "Precipitation",
            )}
          />
          <span className="volume-render__slider-value">{precipOpacity.toFixed(2)}×</span>
        </ControlGroup>
        <ControlGroup label="Precip floor">
          <input
            type="range"
            min={0.01}
            max={1.5}
            step={0.01}
            value={precipFloorGkg}
            onChange={(event) => setPrecipFloorGkg(Number.parseFloat(event.target.value))}
            {...volumeSliderInteractionProps}
            className="volume-render__slider"
            disabled={!showPrecip || availablePrecipSpecies.length === 0}
            title={disabledLayerHint(
              showPrecip,
              availablePrecipSpecies[0] ?? null,
              "Precipitation",
            )}
          />
          <span className="volume-render__slider-value">{precipFloorGkg.toFixed(2)} g/kg</span>
        </ControlGroup>
          </div>
        )}
        {openControlGroup === "radar" && (
          <div
            id="supercell-controls-radar"
            className="plot3d-control-panel"
            role="region"
            aria-label="Radar floor controls"
          >
        <RadarFloorControls
          selection={floorSelection}
          lattice={radarFloor.lattice}
          paintedCells={radarFloor.paintedCells}
          error={radarFloor.error}
          source={radarFloor.source}
          loading={radarFloor.loading}
        />
          </div>
        )}
        {openControlGroup === "flow" && (
          <div
            id="supercell-controls-flow"
            className="plot3d-control-panel"
            role="region"
            aria-label="Flow controls"
          >
        <ControlGroup label="Updraft / downdraft">
          <LayerToggle on={showUpdraft} onClick={() => setShowUpdraft((v) => !v)} disabled={!showUpdraft && !updraftFieldName}
            label={updraftFieldName ? "w isosurface" : showUpdraft ? "unavailable w isosurface" : "n/a"} />
        </ControlGroup>
        <ControlGroup label="|w| threshold">
          <input type="range" min={2} max={40} step={1} value={updraftThreshold}
            onChange={(e) => setUpdraftThreshold(Number.parseInt(e.target.value, 10))}
            className="volume-render__slider" disabled={!showUpdraft || !updraftFieldName}
            title={disabledLayerHint(showUpdraft, updraftFieldName, "Updraft / downdraft")} />
          <span className="volume-render__slider-value">
            {updraftThreshold} m/s
            {updraftThreshold !== committedUpdraftThreshold ? " · preview" : ""}
          </span>
        </ControlGroup>
        <ControlGroup label="Updraft opacity">
          <input type="range" min={0.1} max={1} step={0.05} value={updraftOpacity}
            onChange={(e) => setUpdraftOpacity(Number.parseFloat(e.target.value))}
            className="volume-render__slider" disabled={!showUpdraft || !updraftFieldName}
            title={disabledLayerHint(showUpdraft, updraftFieldName, "Updraft / downdraft")} />
          <span className="volume-render__slider-value">
            {Math.round(updraftOpacity * 100)}%
          </span>
        </ControlGroup>
          </div>
        )}
        {openControlGroup === "vortex" && (
          <div
            id="supercell-controls-vortex"
            className="plot3d-control-panel"
            role="region"
            aria-label="Vortex controls"
          >
        <ControlGroup label="Vortex core">
          <LayerToggle
            on={showVortexCore}
            onClick={() => setShowVortexCore((v) => !v)}
            disabled={!showVortexCore && !vortexCoreSourceAvailable}
            label={vortexCoreFieldName === "p_prime"
              ? "pressure well"
              : "rotation core"}
          />
        </ControlGroup>
        <ControlGroup label="Core field">
          <select
            className="volume-render__select"
            value={vortexCoreFieldName ?? ""}
            disabled={vortexCoreFields.length === 0}
            title={vortexCoreFields.length === 0
              ? "This run exported neither vorticity_z nor p_prime, so there is no field to build a core from."
              : "Field the rotation core is thresholded on"}
            onChange={(event) => setVortexCoreFieldName(event.target.value || null)}
          >
            {vortexCoreFields.length === 0 && <option value="">n/a</option>}
            {vortexCoreFieldName != null && !vortexCoreSourceAvailable
              && <option value={vortexCoreFieldName} disabled>{vortexCoreFieldName} · unavailable in this frame</option>}
            {vortexCoreFields.map((name) => (
              <option value={name} key={name}>{name}</option>
            ))}
          </select>
        </ControlGroup>
        <ControlGroup label="Vortex structure">
          <PillGroup
            options={[
              { value: "core" as const, label: "Low-level core" },
              { value: "hierarchy" as const, label: "Core + meso" },
            ]}
            value={vortexStructure}
            onChange={setVortexStructure}
            disabled={!showVortexCore || vortexCoreFieldName !== "vorticity_z"}
            disabledReason={vortexCoreFieldName !== "vorticity_z"
              ? "A mesocyclone requires the solved vertical-vorticity field; pressure deficit alone does not establish rotation."
              : disabledLayerHint(showVortexCore, vortexCoreFieldName, "Vortex core")}
          />
        </ControlGroup>
        <ControlGroup label="Vortex color">
          <PillGroup
            options={VORTEX_COLOR_OPTIONS}
            value={vortexColorMode}
            onChange={setVortexColorMode}
            disabled={!showVortexCore || vortexCoreFields.length === 0}
            disabledReason={disabledLayerHint(showVortexCore, vortexCoreFieldName, "Vortex core")}
          />
          <span className="volume-render__slider-value">
            {vortexColorMode === "diagnostic"
              ? "cyclonic, anticyclonic, and mesocyclone colors remain distinct"
              : vortexColorMode === "cloud"
                ? "geometry follows the selected cloud tone"
                : "signed meaning remains in the legend"}
          </span>
        </ControlGroup>
        <ControlGroup label="Tornadic condensation">
          <LayerToggle
            on={showTornadicCondensation}
            onClick={() => setShowTornadicCondensation((value) => !value)}
            disabled={!showVortexCore
              || vortexCoreFieldName !== "vorticity_z"
              || tornadicCondensationFields.length === 0}
            disabledReason={tornadicCondensationFields.length === 0
              ? "This frame exports no solved cloud-water or cloud-ice condensate."
              : vortexCoreFieldName !== "vorticity_z"
                ? "Condensation evidence must be associated with solved vertical vorticity; pressure alone cannot establish rotation."
                : "Enable the vortex core before requesting its condensation evidence."}
            label="solved condensation evidence"
          />
        </ControlGroup>
        <ControlGroup label="Core threshold">
          <input type="range" min={0.1} max={0.9} step={0.05} value={vortexCoreFraction}
            onChange={(e) => setVortexCoreFraction(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
            disabled={!showVortexCore || vortexCoreFields.length === 0}
            title={disabledLayerHint(showVortexCore, vortexCoreFieldName, "Vortex core")} />
          <span className="volume-render__slider-value">
            {vortexCoreThreshold != null
              ? withVortexCoreUnit(`±${formatCoreThreshold(vortexCoreThreshold)}`)
              : `${Math.round(vortexCoreFraction * 100)}% of core peak`}
            {vortexCoreFraction !== committedVortexCoreFraction ? " · preview" : ""}
            {synchronizedPresentation && vortexCoreThreshold != null
              ? ` · fixed across ${recording ? "export" : "playback"}`
              : ""}
            {!synchronizedPresentation && vortexExtractionLoading ? " · extracting off-thread" : ""}
            {!synchronizedPresentation && vortexExtraction.result && vortexExtraction.result.elapsedMs >= 100
              ? ` · ${Math.round(vortexExtraction.result.elapsedMs)} ms geometry`
              : ""}
          </span>
        </ControlGroup>
        <ControlGroup label="Core opacity">
          <input type="range" min={0.1} max={1} step={0.05} value={vortexCoreOpacity}
            onChange={(e) => setVortexCoreOpacity(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
            disabled={!showVortexCore || vortexCoreFields.length === 0}
            title={disabledLayerHint(showVortexCore, vortexCoreFieldName, "Vortex core")} />
          <span className="volume-render__slider-value">
            {Math.round(vortexCoreOpacity * 100)}%
          </span>
        </ControlGroup>
        {vortexCoreFieldName === "vorticity_z" && (
          <ControlGroup label="Vortex classification" stretch>
            <span className="volume-render__slider-value">
              {synchronizedPresentation
                ? "Updates with each fully presented frame"
                : formatTornadoVortexAssessment(vortexAssessment)}
            </span>
          </ControlGroup>
        )}
        {showTornadicCondensation && vortexCoreFieldName === "vorticity_z" && (
          <ControlGroup label="Condensation classification" stretch>
            <span className="volume-render__slider-value">
              {synchronizedPresentation
                ? "Updates with each fully presented frame"
                : formatTornadicCondensationAssessment(tornadicCondensationAssessment)}
            </span>
          </ControlGroup>
        )}
        {vortexStructure === "hierarchy" && vortexCoreFieldName === "vorticity_z" && (
          <ControlGroup label="Meso envelope" stretch>
            <span className="volume-render__slider-value">
              {synchronizedPresentation
                ? "Updates with each fully presented frame"
                : formatMesocycloneAssessment(
                    mesocycloneAssessment,
                    mesocycloneThreshold,
                    mesocycloneContext,
                  )}
            </span>
          </ControlGroup>
        )}
        <ControlGroup label="Camera">
          <button
            type="button"
            className="btn btn--secondary volume-render__reset"
            onClick={focusVortex}
            disabled={!vortexExtraction.result?.core?.bounds
              || (pane.cameraLocked && !pane.cameraAuthority)}
            title={pane.cameraLocked && !pane.cameraAuthority
              ? "This pane follows the shared camera authority."
              : "Fill the viewport with the connected vortex and any solved condensation evidence."}
          >
            <Icon name="plot" size={12} />
            <span>Focus vortex</span>
          </button>
        </ControlGroup>
          </div>
        )}
        {openControlGroup === "scene" && (
          <div
            id="supercell-controls-scene"
            className="plot3d-control-panel"
            role="region"
            aria-label="Scene and camera controls"
          >
        <SceneAppearanceControls state={sceneAppearance} backend={vtkBackend} />
        <SceneScaleControl value={sceneScale} onChange={(value) => {
          beginVolumeControlPreview();
          setSceneScale(value);
          endVolumeControlPreview();
        }} />
        {(!pane.cameraLocked || pane.cameraAuthority) && (
          <>
            <CameraElevationControls selection={cameraElevation} />
            <button type="button" className="btn btn--secondary volume-render__reset"
              onClick={() => setResetTick((t) => t + 1)}>
              <Icon name="settings" size={12} />
              <span>Reset camera</span>
            </button>
          </>
        )}
          </div>
        )}
      </div>

      <div
        className="volume-render__canvas-wrap plot3d-overlay-host"
        style={sceneAppearance.hostStyle}
        data-vortex-core-enabled={showVortexCore}
        data-tornadic-condensation-enabled={showTornadicCondensation}
        data-diagnostic-reveal-active={diagnosticRevealActive}
        data-flow-contour-loading={flowLayerLoading}
        data-flow-preparation-loading={updraftPreparation.loading}
        data-flow-extraction-loading={flowExtractionLoading}
        data-flow-preparation-commits={flowPreparationCommitsRef.current}
        data-flow-recenters={flowRecentersRef.current}
        data-flow-center={flowDiagnosticCenter?.join(",") ?? ""}
        data-flow-peak={lastFlowPeakRef.current?.join(",") ?? ""}
        data-vortex-contour-loading={vortexLayerLoading}
        data-vortex-preparation-loading={vortexCorePreparation.loading}
        data-condensate-preparation-loading={tornadicCondensatePreparation.loading}
        data-vortex-extraction-loading={vortexExtractionLoading}
        data-detail-target={detailTargetMode}
        data-detail-center-m={stormDetailCenter ? stormDetailCenter.join(",") : ""}
        data-cloud-motion-active={cloudMotionActive}
        data-cloud-load-state={JSON.stringify([cloud.loadedStepIndex, cloud.loadedPublicationGeneration, cloud.loading])}
        data-texture-capacity-state={JSON.stringify([vtkLifecycleGeneration, ctxLoss.available, ctxLoss.lost,
          textureCapacity.known, textureCapacity.maximumDimension, textureCapacity.error])}
        data-cloud-preparation-state={JSON.stringify([coherentFrame, cloudPreparation.loading,
          cloudPreparation.prepared != null, cloudPreparation.error])}
        data-scene-gpu-admission={JSON.stringify([supercellGpuUploadBytes, supercellGpuUpload.admitted, supercellGpuUpload.error])}
        data-cloud-motion-enabled={cloudMotionEnabled}
        data-cloud-wind-preview-enabled={cloudWindEnabled}
        data-cloud-wind-requested={requestCloudWind}
        data-scene-frame-commit-pending={sceneFrameCommitPending}
        data-scene-presentation-pending={scenePresentationPending}
        data-requested-scene-generation={sceneGenerationIdentity}
        data-committed-scene-generation={committedSceneIdentity ?? undefined}
        data-scene-planning-peak-bytes={sceneBudget.estimatedPeakBytes}
        data-scene-planning-envelope-bytes={sceneBudget.planningEnvelopeBytes}
        data-scene-unique-fields={sceneBudget.uniqueFields}
        data-scene-detail-cell-cap={sceneBudget.detailCells}
        data-scene-context-cell-cap={sceneBudget.contextCells}
        data-scene-auxiliary-cell-cap={sceneBudget.auxiliaryCells}
        data-synchronized-presentation={synchronizedPresentation}
        data-output-level={activeOutputLevel?.levelId ?? "root"}
        data-output-hierarchy-valid={outputHierarchySource.valid}
        data-logical-gpu-upload-bytes={supercellGpuUpload.admitted ? supercellGpuUploadBytes : 0}
      >
        <div
          ref={containerRef}
          className="volume-render__canvas"
          role="img"
          aria-label="Interactive three-dimensional rendered supercell; drag to orbit, use camera controls for keyboard adjustment"
        />
        <Plot3DAxisLabels labels={axisLabels} />
        <div className="plot3d-motion-readout" hidden={!cloudMotionActive} role="status" aria-live="off">
          Interpolated cloud · t=<span ref={cloudMotionTimeRef} /> s
        </div>
        <Plot3DScaleBadge
          {...committedScaleDisclosure}
          grid={committedScaleDisclosure?.grid ?? null}
          zExag={sceneScale.z}
        />
        <Plot3DOverlayStack>
          {floorLegend && <Plot3DColorbar {...floorLegend} />}
          {vortexLegend && <Plot3DDiscreteLegend {...vortexLegend} />}
          {precipLegend && <Plot3DDiscreteLegend {...precipLegend} />}
        </Plot3DOverlayStack>
        <Plot3DCompass orientation={cameraOrientation} />
        {ctxLoss.lost && (
          <div className="volume-render__overlay volume-render__overlay--error">
            3D graphics context lost
            <div className="volume-render__overlay-hint">
              Rebuilding on the compatibility backend after the GPU dropped the active device. This clears after a
              selected-frame layer renders successfully.
            </div>
          </div>
        )}
        {preparationError && (
          <div className="volume-render__overlay volume-render__overlay--error">
            3D layer failed: {preparationError}
          </div>
        )}
        {boundedPlanError && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Storm-field loading is unavailable within the bounded-read policy.
            <div className="volume-render__overlay-hint">
              {boundedPlanError} Nimbus will not fall back to full-volume reads.
            </div>
          </div>
        )}
        {unsupportedWindowFields.length > 0 && !terrainBlocked && !boundedPlanError && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Some layers were skipped because their providers cannot serve bounded windows.
            <div className="volume-render__overlay-hint">
              {unsupportedWindowFields.join(", ")}. Full-volume fallback is disabled.
            </div>
          </div>
        )}
        {missingAll && !terrainBlocked && (
          <div className="volume-render__overlay volume-render__overlay--error">
            This run has none of the supercell fields.
            <div className="volume-render__overlay-hint">
              Re-run with <code>output.preset: presentation</code> (or higher) so
              <code> total_condensate</code>, <code>reflectivity_dbz</code>, and
              <code> w</code> are exported.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- small UI helpers ---------- */

function SupercellControlTab({
  group,
  label,
  status,
  openGroup,
  onToggle,
}: {
  group: SupercellControlGroup;
  label: string;
  status: string;
  openGroup: SupercellControlGroup | null;
  onToggle: React.Dispatch<React.SetStateAction<SupercellControlGroup | null>>;
}) {
  const open = openGroup === group;
  const unavailable = status === "Unavailable";
  return (
    <button
      type="button"
      className={`plot3d-control-tab${open ? " plot3d-control-tab--active" : ""}`}
      data-state={unavailable ? "unavailable" : status.toLowerCase()}
      aria-expanded={open}
      title={`${label}: ${status}`}
      aria-controls={`supercell-controls-${group}`}
      onClick={() => onToggle((current) => current === group ? null : group)}
    >
      <span>{label}</span>
      <span className="plot3d-control-tab__status">{status}</span>
      <Icon name={open ? "chevron-up" : "chevron-down"} size={12} />
    </button>
  );
}

/**
 * Activate a heavy render path only after the requested UI state has
 * reached the screen. Two frames are intentional: the first lets React commit,
 * while the second leaves one complete painted frame in which the toggle can
 * acknowledge the click before vtk.js begins lazy contour evaluation.
 */
function useAfterPaintActivation(requested: boolean): boolean {
  const [active, setActive] = useState(false);

  useEffect(() => {
    if (!requested) {
      setActive(false);
      return;
    }

    let secondFrame = 0;
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => setActive(true));
    });

    return () => {
      window.cancelAnimationFrame(firstFrame);
      if (secondFrame !== 0) window.cancelAnimationFrame(secondFrame);
    };
  }, [requested]);

  return requested && active;
}

/**
 * Why a layer's settings are inert, phrased as the action that revives them.
 *
 * Each layer here owns a toggle and several sliders, and the sliders are
 * disabled while the toggle is off — correct, because moving them would change
 * nothing visible. But a greyed slider with no tooltip is indistinguishable
 * from a broken one: the user aims at it, nothing happens, and the screen never
 * connects it to the toggle three controls to the left. This is the sentence
 * that makes the connection.
 *
 * Returns undefined when the control is live, so it never overwrites a title
 * the control wanted to set for itself.
 */
function disabledLayerHint(
  layerOn: boolean,
  fieldName: string | null,
  layerLabel: string,
): string | undefined {
  if (!fieldName) return `This run exported no field for the ${layerLabel} layer.`;
  if (!layerOn) return `Turn the ${layerLabel} layer on to use this.`;
  return undefined;
}

function formatMesocycloneAssessment(
  assessment: MesocycloneEnvelopeAssessment | null,
  threshold: number | null,
  context?: MesocycloneContextAssessment,
): string {
  if (threshold == null) return "No distinct lower vorticity contour at this core threshold";
  const contour = `≥ ${formatCoreThreshold(threshold)} s⁻¹`;
  if (context && !context.qualified) {
    const span = context.spanM.map((value) => value >= 1_000
      ? `${Number((value / 1_000).toPrecision(2))} km`
      : `${Math.round(value)} m`).join(" × ");
    return `${contour} · mesoscale context withheld: ${span} native ROI is too small`
      + " · requires a published multiscale field product";
  }
  if (!assessment) return `${contour} · evaluating solved vorticity`;
  if (assessment.present) {
    return `${contour} · connected · ${(assessment.diameterM / 1_000).toFixed(1)} km wide`
      + ` · ${(assessment.depthM / 1_000).toFixed(1)} km deep`;
  }
  const reason = assessment.reason === "detached"
    ? "not connected to the resolved core"
    : assessment.reason === "too-shallow"
      ? "vertical extent below 2 km"
      : assessment.reason === "too-narrow"
        ? "horizontal extent below 1.5 km"
        : assessment.reason === "boundary-clipped"
          ? "surface reaches the retained ROI boundary"
        : "no connected surface";
  return `${contour} · withheld: ${reason}`;
}

function formatTornadoVortexAssessment(
  assessment: TornadoVortexAssessment | null,
): string {
  if (!assessment) return "Evaluating the connected low-level solved-vorticity core";
  if (assessment.present) {
    return `Tornado-scale vortex resolved · ${(assessment.diameterM / 1_000).toFixed(2)} km wide`
      + ` · ${(assessment.depthM / 1_000).toFixed(2)} km deep`;
  }
  const reason = assessment.reason === "under-resolved"
    ? "grid spacing is too coarse for a tornado-scale label"
    : assessment.reason === "aloft"
      ? "connected core does not reach the lowest model levels"
      : assessment.reason === "too-shallow"
        ? "connected core lacks 0.5 km vertical continuity"
        : assessment.reason === "too-broad"
          ? "connected circulation is mesocyclone-scale rather than tornado-scale"
          : "no connected low-level core at this threshold";
  return `Low-level vortex · ${reason}`;
}

function formatTornadicCondensationAssessment(
  assessment: TornadicCondensationAssessment | null,
): string {
  if (!assessment) {
    return "Evaluating solved condensate around the connected low-level vortex";
  }
  if (assessment.reason === "invalid-input") return "Condensation withheld: invalid field geometry or threshold";
  const tip = assessment.lowestCondensateM == null
    ? null
    : `${(assessment.lowestCondensateM / 1_000).toFixed(2)} km`;
  const cloudBase = assessment.ambientCloudBaseM == null
    ? null
    : `${(assessment.ambientCloudBaseM / 1_000).toFixed(2)} km`;
  if (assessment.present) {
    return `Resolved condensation funnel · tip ${tip}`
      + ` · ${(assessment.downwardExtensionM / 1_000).toFixed(2)} km below local cloud base`;
  }
  if (assessment.reason === "rotating-cloud") {
    return `Rotating wall cloud only · lowest condensate ${tip}`
      + `${cloudBase ? ` · local cloud base ${cloudBase}` : ""}`
      + " · no resolved downward funnel extension";
  }
  if (assessment.reason === "no-qualified-vortex") {
    return assessment.visible
      ? `Rotating condensation evidence at ${tip} · circulation is not tornado-qualified`
      : "Dry or non-tornado-scale low-level vortex · no qualified condensation funnel";
  }
  if (assessment.reason === "too-small") {
    return `Condensation evidence is below the resolved continuity minimum`
      + `${tip ? ` · lowest sample ${tip}` : ""}`;
  }
  if (assessment.reason === "misaligned-fields") {
    return "Condensation withheld because condensate and vorticity lattices do not align";
  }
  return "Dry vortex · no solved condensate is connected to this circulation";
}

function LayerToggle({
  on, onClick, disabled, disabledReason, label,
}: {
  on: boolean;
  onClick: () => void;
  disabled?: boolean;
  disabledReason?: string;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`pill ${on && !disabled ? "pill--active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      title={disabled ? disabledReason ?? "Field not exported in this run" : label}
    >
      {disabled ? "n/a" : on ? `on · ${label}` : "off"}
    </button>
  );
}

function maybeResetCamera(
  scene: {
    renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    upActor: ReturnType<typeof vtkActor.newInstance>;
    downActor: ReturnType<typeof vtkActor.newInstance>;
    coreActor: ReturnType<typeof vtkActor.newInstance>;
    oppositeActor: ReturnType<typeof vtkActor.newInstance>;
    mesocycloneActor: ReturnType<typeof vtkActor.newInstance>;
  },
  initialized: React.MutableRefObject<boolean>,
  cloudFocusBounds: BoundsTuple | null = null,
  scale: { xy: number; z: number } = { xy: 1, z: 1 },
) {
  if (initialized.current) return;
  resetVtkCameraToStormView(
    scene.renderWindow.getRenderer(),
    supercellFocusBounds(scene, cloudFocusBounds, scale),
  );
  initialized.current = true;
}

type BoundsTuple = [number, number, number, number, number, number];

function applySurfaceGeometry(
  target: ReturnType<typeof vtkPolyData.newInstance>,
  geometry: VortexSurfaceGeometry | null,
): void {
  releaseVtkPointDataArrays(target.getPointData());
  target.getPoints().setData(geometry?.points ?? new Float32Array(), 3);
  target.getPolys().setData(geometry?.polys ?? new Uint32Array());
  if (geometry && geometry.normals.length === geometry.points.length) {
    target.getPointData().setNormals(vtkDataArray.newInstance({
      name: "Normals",
      values: geometry.normals,
      numberOfComponents: 3,
    }));
  }
  target.modified();
}

function surfaceGeometryBytes(geometry: VortexSurfaceGeometry | null): number {
  return geometry?.bytes ?? 0;
}

type BoundedActor = {
  getVisibility: () => boolean;
  getBounds: () => BoundsTuple;
};

/** Fit the rendered-supercell preset to meteorological geometry, not the much
 * larger context box. The box remains visible through the camera clip range. */
function supercellFocusBounds(scene: {
  upActor: BoundedActor;
  downActor: BoundedActor;
  coreActor: BoundedActor;
  oppositeActor: BoundedActor;
  mesocycloneActor: BoundedActor;
}, cloudFocusBounds: BoundsTuple | null = null, scale = { xy: 1, z: 1 }): BoundsTuple | null {
  let combined = cloudFocusBounds ? scaleBounds(cloudFocusBounds, scale) : null;
  const actors = [
    scene.upActor,
    scene.downActor,
    scene.coreActor,
    scene.oppositeActor,
    scene.mesocycloneActor,
  ];
  for (const actor of actors) {
    if (!actor.getVisibility()) continue;
    const bounds = actor.getBounds();
    if (!bounds.every(Number.isFinite)) continue;
    if (!combined) {
      combined = [...bounds] as BoundsTuple;
      continue;
    }
    combined[0] = Math.min(combined[0], bounds[0]);
    combined[1] = Math.max(combined[1], bounds[1]);
    combined[2] = Math.min(combined[2], bounds[2]);
    combined[3] = Math.max(combined[3], bounds[3]);
    combined[4] = Math.min(combined[4], bounds[4]);
    combined[5] = Math.max(combined[5], bounds[5]);
  }
  return combined;
}

function scaleBounds(
  bounds: BoundsTuple,
  scale: { xy: number; z: number },
): BoundsTuple {
  const x0 = bounds[0] * scale.xy;
  const x1 = bounds[1] * scale.xy;
  const y0 = bounds[2] * scale.xy;
  const y1 = bounds[3] * scale.xy;
  const z0 = bounds[4] * scale.z;
  const z1 = bounds[5] * scale.z;
  return [
    Math.min(x0, x1), Math.max(x0, x1),
    Math.min(y0, y1), Math.max(y0, y1),
    Math.min(z0, z1), Math.max(z0, z1),
  ];
}

// Condensate at which the cloud becomes optically THICK (full white / peak
// opacity). This is an ABSOLUTE physical value, not the storm's max: a cloud is
// opaque at ~1-2 g/kg of condensate regardless of how intense the storm is, so
// normalising to the field max would wrongly make a weak cloud as opaque as a
// strong one (and the old 20 g/kg anchor left every realistic storm ~95%
// transparent). Fixed, so the render also doesn't flicker as the storm grows.
const CLOUD_DISPLAY_MAX = 0.0015; // kg/kg = 1.5 g/kg
function clampCloudFloor(floor: number): number {
  return Math.min(Math.max(floor, 1e-6), CLOUD_DISPLAY_MAX * 0.8);
}

/**
 * Cloud transfer function. The cloud exists ONLY where condensate exceeds
 * `floor` (the condensation threshold — the point the parcel condenses), giving
 * a sharp cloud boundary; above it the opacity rises to an optically thick,
 * white body without a long low-opacity tail. `scale` (the
 * "Cloud density" slider) scales the peak opacity; `floor` (the "Cloud floor"
 * slider, kg/kg) sets the condensation cutoff.
 */
function applyCloudTransfer(
  ctf: ReturnType<typeof vtkColorTransferFunction.newInstance>,
  pwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>,
  scale: number,
  floor: number,
  edge: CloudEdge,
  colors: { edge: readonly [number, number, number]; body: readonly [number, number, number] },
) {
  // The floor MUST stay strictly below CLOUD_DISPLAY_MAX (with headroom for a
  // usable span) or the colour/opacity nodes below — which vtk requires to be
  // strictly increasing — invert, collapsing the transfer function so the cloud
  // renders blank/garbled. The lower bound is a small strictly-positive epsilon
  // (never 0) so the opacity ramp's addPoint(0, 0) and addPoint(floor, 0) remain
  // distinct x nodes — floor === 0 would put two nodes at x=0 and break the same
  // strictly-increasing rule. The "Cloud floor" slider is capped to match, but
  // clamp here too as the owner of the invariant (defence in depth).
  floor = clampCloudFloor(floor);

  ctf.removeAllPoints();
  ctf.addRGBPoint(floor, colors.edge[0], colors.edge[1], colors.edge[2]);
  ctf.addRGBPoint(CLOUD_DISPLAY_MAX, colors.body[0], colors.body[1], colors.body[2]);
  ctf.setMappingRange(floor, CLOUD_DISPLAY_MAX);
  ctf.updateRange();

  const peak = scaledCloudPeakOpacity(scale);
  pwf.removeAllPoints();
  for (const point of cloudOpacityPoints(floor, CLOUD_DISPLAY_MAX, peak, edge)) {
    pwf.addPoint(point.scalar, point.opacity);
  }
}

function applyPrecipTransfer(
  ctf: ReturnType<typeof vtkColorTransferFunction.newInstance>,
  pwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>,
  opacity: number,
  requestedFloor: number,
  color: readonly [number, number, number],
): void {
  const floor = Math.min(
    Math.max(requestedFloor, 1e-6),
    PRECIP_DISPLAY_MAX * 0.8,
  );
  ctf.removeAllPoints();
  ctf.addRGBPoint(floor, color[0], color[1], color[2]);
  ctf.addRGBPoint(PRECIP_DISPLAY_MAX, color[0], color[1], color[2]);
  ctf.setMappingRange(floor, PRECIP_DISPLAY_MAX);
  ctf.updateRange();

  const peak = scaledPrecipPeakOpacity(opacity);
  pwf.removeAllPoints();
  pwf.addPoint(0, 0);
  pwf.addPoint(floor, 0);
  pwf.addPoint(Math.min(PRECIP_DISPLAY_MAX, floor * 1.15), peak * 0.08);
  pwf.addPoint(Math.max(floor * 1.16, PRECIP_DISPLAY_MAX * 0.35), peak * 0.45);
  pwf.addPoint(PRECIP_DISPLAY_MAX, peak);
}

function setPrecipRayDistances(
  volume: ReturnType<typeof vtkVolume.newInstance>,
  camera: { getDirectionOfProjection: () => number[] },
  quality: VtkVolumeQualityController,
  physicalSampleDistance: number,
  scale: { xy: number; z: number },
  mapper: ReturnType<typeof vtkVolumeMapper.newInstance>,
): void {
  setCameraAdjustedVolumeRayDistances(
    volume.getProperty(),
    camera,
    quality,
    PRECIP_OPACITY_REFERENCE_M,
    physicalSampleDistance,
    scale,
    mapper,
  );
}
