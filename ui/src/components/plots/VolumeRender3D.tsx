import { volumeZoomCellBudget } from "@/lib/plot/volumeZoomBudget";
import { volumeImageBytes } from "@/lib/vtk/volumeValidityBudget";
import { useVolumeTextureCapacity } from "@/lib/vtk/useVolumeTextureCapacity";
import { usesVolumeBackground, backgroundAffectsOpacity } from "@/lib/vtk/volumeBackground";
import { useVolumeBackground } from "@/lib/vtk/useVolumeBackground";
import { setVolumeBackground } from "@/lib/vtk/volumeBackgroundTexture";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fieldDefinitionIdentity } from "@/lib/run/runOutputs";
import vtkPlane from "@kitware/vtk.js/Common/DataModel/Plane";
import { configureCoherentVolumeSampling } from "@/lib/vtk/coherentVolumeSampling";
import { buildVolumeOrder, orderedVolumeIndices, uniqueVolumeImages, volumeOwnedBounds, volumeOwnedScalarRows, volumeOpenUpperFaces, type VolumeOrder } from "@/lib/vtk/volumeBrickGeometry";
import { unitFor } from "@/lib/science/fieldUnits";
import { Plot3DAxisLabels, Plot3DColorbar, Plot3DOverlayStack, Plot3DScaleBadge } from "./Plot3DOverlay";
import { useCommittedLegend } from "./useCommittedLegend";
import { Plot3DCompass } from "./Plot3DCompass";
import { useCameraOrientation } from "@/lib/vtk/useCameraOrientation";
import { useCameraSync } from "@/lib/vtk/useCameraSync";
import { usePaneContext } from "./paneContext";
import { stormTrackCentre, useStormTrack } from "./useStormTrack";
import { scaleVoxelBudget } from "@/lib/plot/paneLayout";
import {
  CARTESIAN_VOXEL_CAP_VOLUME,
} from "@/lib/science/cartesianInterp";
import {
  fieldSupportsBoundedWindows,
  isNativeStormDetailWindowPlan,
  coalesceNativeFieldWindows,
  MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS,
  planBoundedVolumeWindows,
  planInteractiveVolumeWindows,
  validGridCellCount,
  minimumNominalGridSpacingM,
} from "@/lib/plot/fieldLoadPlanning";
import { useRadarFloorLayer } from "./radarFloorLayer";
import { RadarFloorControls, useRadarFloorSelection } from "./RadarFloorControls";
import {
  CameraElevationControls,
  useCameraElevation,
  useCameraElevationPlacement,
} from "./CameraElevationControls";
import {
  setPreparedImageDisplayGeometry,
  setPreparedImageScalars,
} from "@/lib/vtk/vtkImageData";
import { usePreparedScalarImage } from "@/lib/vtk/useVtkPreparation";
import { DBZ_VTK_STOPS_VOLUME } from "@/lib/plot/plotColormaps";
import {
  defaultVolumeColormap,
  isReflectivityField,
  preferredVolumetricReflectivityField,
  resolveVolumeTransferPolicy,
  supportsVolumeStructure,
  VOLUME_OPACITY_REFERENCE_M,
  VOLUME_CONTEXT_OPACITY_FACTOR,
  volumeFieldWindowAggregation,
  volumeOpacityClass,
  volumeOpacityReferenceM,
  volumeOpacityPoints,
  volumeUsesOpacityNormals,
  volumeUsesStructuralLighting,
  type VolumeOpacityClass,
  type VolumeTransferKind,
} from "@/lib/plot/volumeTransferPolicy";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { describeFrameTransition, findPublishedFrame } from "@/lib/run/frameIdentity";
import { disposeVtkScene } from "@/lib/vtk/vtkLifecycle";
import { VOLUME_RESOLUTION_POLICY } from "@/lib/vtk/vtkResolution";
import { attachVtkPlaybackScreenResolution, lookupVtkPlaybackScreenResolution } from "@/lib/vtk/vtkPlaybackScreenResolution";
import { useVtkPlaybackScreenResolution } from "@/lib/vtk/useVtkPlaybackScreenResolution";
import { setCameraAdjustedVolumeRayDistances } from "@/lib/vtk/volumeOpticalDepth";
import {
  attachVtkInteractionQuality,
  attachVtkCameraStability,
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
  focusVtkCameraOnBounds,
  focusVtkCameraOnPoint,
  PillGroup,
  refitVtkCameraToBounds,
  useSceneScaleFraming,
  resetVtkCameraToStormView,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useField,
  useSceneScale,
  useFieldLink,
  useVtkContextLoss,
  useVtkRunReset,
  SceneScaleControl,
  type PlotComponentProps,
} from "./common";
import { describeDeclinedFieldLink, SINGLE_LEVEL_LINK_REASON } from "@/lib/plot/fieldLink";
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
import {
  applyVolumeLighting,
  createVtkSceneLighting,
  type VtkSceneLighting,
} from "@/lib/vtk/sceneLighting";
import { useDomainBox } from "./domainBox";
import { useDomainAxes } from "./useDomainAxes";
import { usePanePresentationBarrier } from "./panePresentationContext";
import { holdVtkSceneRendering } from "@/lib/vtk/vtkContextRenderGuard";
import { coarsestVtkSampling } from "@/lib/plot/renderFidelity";
import { domainBoundsM } from "@/lib/plot/domainAxes";
import { useBrickedVolumePreparation } from "@/lib/vtk/useBrickedVolumePreparation";
import {
  scalarMultisetFingerprint,
  scalarQuantizedMultisetFingerprint,
} from "@/lib/vtk/scalarCompositionParity";
import { selectCameraVolumeLevel } from "@/lib/plot/volumeLodPolicy";
import { lazyAppearanceEvidence } from "@/lib/vtk/lazyAppearanceEvidence";
import { visualizationResourceGovernor } from "@/lib/core/visualizationResourceGovernor";
import { useVtkGpuResourceOwnership } from "@/lib/vtk/useVtkGpuResourceOwnership";
import { planNestedVolumeOwnership } from "@/lib/run/outputHierarchy";
import { useNestedBrickedVolumePreparation } from "@/lib/vtk/useNestedBrickedVolumePreparation";
import "./common.css";
import "./VolumeRender3D.css";

/* VTK.js scalar volume renderer. Cartesian field -> vtkImageData -> volume mapper.
 * Lazy-imported so VTK only ships for users who open 3D views. */

// Profile import registers OpenGL volume rendering side effects.
import "@kitware/vtk.js/Rendering/Profiles/Volume";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkColorMaps from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps";

type ColormapPreset =
  | "Inferno"
  | "Viridis"
  | "Cool to Warm"
  | "Black-Body Radiation"
  | "Greys"
  | "dBZ";

const COLORMAPS: ReadonlyArray<{ value: ColormapPreset; label: string; vtkName: string | null }> = [
  { value: "Inferno", label: "Inferno", vtkName: "Inferno (matplotlib)" },
  { value: "Viridis", label: "Viridis", vtkName: "Viridis (matplotlib)" },
  { value: "Cool to Warm", label: "Cool→Warm", vtkName: "Cool to Warm" },
  { value: "Black-Body Radiation", label: "Black-body", vtkName: "Black-Body Radiation" },
  // Greyscale + dBZ are custom anchor lists (no vtk preset by that name); we
  // apply them manually via addRGBPoint instead of applyColorMap.
  { value: "Greys", label: "Greyscale", vtkName: null },
  { value: "dBZ", label: "NWS dBZ", vtkName: null },
];

/** Preferred field defaults; pick the first present in the step's list, else first available. */
const PREFERRED_DEFAULTS = [
  "vorticity_magnitude",
  "vorticity_z",
  "w",
  "reflectivity_dbz",
  "theta",
  "qc",
];

export function VolumeRender3D({
  runDir,
  steps,
  stepIdx,
  preset,
  onPresetCleared,
  onFrameDisplayChange,
  renderFidelityMode = "scientific",
  linkedField,
  onFieldChange,
  volumeAppearanceQualification,
}: PlotComponentProps) {
  const controlScope = useThreeDControlScope();
  const sceneAppearance = useSceneAppearance(controlScope.scope, "volume");
  const vtkBackend = useVtkBackend(controlScope.scope, "volume", "volume");
  const sceneAppearanceRef = useRef(sceneAppearance.resolved);
  sceneAppearanceRef.current = sceneAppearance.resolved;
  const opacityClassRef = useRef<VolumeOpacityClass>("generic");
  const rootCurrentStep = steps[stepIdx] ?? null;
  const [fieldName, setFieldName] = useState<string | null>(preset?.field ?? null);
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
  const handoffFocusCenter = preset?.volumeFocus?.centerM;
  const handoffFocusXY = useMemo(() => handoffFocusCenter
      && Number.isFinite(handoffFocusCenter[0])
      && Number.isFinite(handoffFocusCenter[1])
    ? [handoffFocusCenter[0]!, handoffFocusCenter[1]!] as [number, number]
    : null, [handoffFocusCenter]);
  const hierarchyRequiredFields = useMemo(() => fieldName ? [fieldName] : [], [fieldName]);
  const outputHierarchySource = useOutputHierarchySource({
    runDir,
    steps,
    stepIdx,
    scope: controlScope.scope,
    requiredFields: hierarchyRequiredFields,
    focusPointM: handoffFocusXY ?? (stormCentre ? [stormCentre[0], stormCentre[1]] : null),
  });
  const currentStep = outputHierarchySource.currentStep;
  const sourceRunDir = outputHierarchySource.sourceRunDir;
  const sourceDefinitionIdentity = fieldDefinitionIdentity(sourceRunDir, currentStep?.index ?? null,
    fieldName, currentStep?.publicationGeneration ?? null);
  const sourceSteps = outputHierarchySource.sourceSteps;
  const fieldsForStep = useMemo(() => currentStep?.fields ?? [], [currentStep?.fields]);

  // Restricted to fields with a vertical dimension: the fallback is otherwise whichever
  // field sorts first, which on a run leading with cape opens the widget on a shape error.
  const renderableFields = useMemo(
    () => fieldsForPlotRanks(sourceRunDir, currentStep, [3]),
    [currentStep, sourceRunDir],
  );
  const defaultField = useMemo(() => {
    for (const name of PREFERRED_DEFAULTS) {
      if (renderableFields.includes(name)) return name;
    }
    return renderableFields[0] ?? null;
  }, [renderableFields]);

  const [colormap, setColormap] = useState<ColormapPreset>(
    isVtkColormap(preset?.colormap)
      ? (preset!.colormap as ColormapPreset)
      : defaultVolumeColormap(preset?.field ?? defaultField) as ColormapPreset,
  );
  // Opacity scale stretches/compresses the transfer function around its midpoint.
  const [opacityScale, setOpacityScale] = useState<number>(1.0);
  const [backgroundFieldMode, setBackgroundFieldMode] = useState<"raw" | "structure">("structure");
  const backgroundRoot = useMemo(() => steps.filter(step => step.fields.includes(fieldName ?? ""))
    .sort((a, b) => Math.abs((a.simulationTimeS ?? 0) - (currentStep?.simulationTimeS ?? 0))
      - Math.abs((b.simulationTimeS ?? 0) - (currentStep?.simulationTimeS ?? 0)))[0], [steps, fieldName, currentStep?.simulationTimeS]);
  const ctxLoss = useVtkContextLoss();
  const { generation: contextGeneration, markFrameRendered } = ctxLoss;
  const vtkLifecycleGeneration = contextGeneration * 2 + (vtkBackend.backend === "webgpu" ? 1 : 0);
  const textureCapacity = useVolumeTextureCapacity(vtkLifecycleGeneration);
  const background = useVolumeBackground(textureCapacity.known && ctxLoss.available
    && backgroundFieldMode === "structure" && usesVolumeBackground(fieldName),
    backgroundRoot ? runDir : sourceRunDir, backgroundRoot ?? currentStep, fieldName);
  const backgroundRef = useRef(background.profile);
  backgroundRef.current = background.profile;
  // [min, max] of the colour transfer function, surfaced for the on-canvas colorbar.
  const [colorRange, setColorRange] = useState<{ min: number; max: number } | null>(null);
  const [opacityMaximum, setOpacityMaximum] = useState<number | null>(null);
  const [transferKind, setTransferKind] = useState<VolumeTransferKind>("sequential");
  const [opacityClass, setOpacityClass] = useState<VolumeOpacityClass>("generic");
  const [opacityReferenceM, setOpacityReferenceM] = useState(VOLUME_OPACITY_REFERENCE_M);
  const appearanceEvidenceRef = useRef<VtkAppearanceDescriptor>({
    plot: "volume",
    backend: vtkBackend.backend,
  });
  appearanceEvidenceRef.current = {
    plot: "volume",
    backend: vtkBackend.backend,
    datasetId: sourceRunDir,
    frameIndex: currentStep?.index ?? null,
    field: fieldName,
    fidelityMode: renderFidelityMode,
    parameters: {
      colormap,
      opacityScale,
      opacityClass,
      opacityReferenceM,
      backgroundFieldMode,
      stormStructure: backgroundFieldMode === "structure",
      backgroundProfile: background.profile ? { method: "sampled horizontal median", heightsM: background.profile.heights,
        values: background.profile.values, threshold: background.profile.threshold, positiveOnly: background.profile.positiveOnly,
        referenceStep: (backgroundRoot ?? currentStep)?.index, referenceTimeS: (backgroundRoot ?? currentStep)?.simulationTimeS,
        interiorBoundsM: background.profile.interior ?? null, boundaryFeatherM: background.profile.featherM ?? null } : null,
      colorRange: colorRange ? [colorRange.min, colorRange.max] : null,
      opacityMaximum,
      transferKind,
      normalBasis: volumeUsesStructuralLighting(opacityClass) ? "scalar-gradient" : "unlit",
      opacityPoints: colorRange
        ? volumeOpacityPoints({
            kind: transferKind,
            opacityClass,
            min: colorRange.min,
            max: colorRange.max,
            ...(opacityMaximum == null ? {} : { opacityMax: opacityMaximum }),
          }, opacityScale * (volumeAppearanceQualification?.detailOpacityFactor ?? 1))
          .map((point) => [point.scalar, point.opacity])
        : [],
      lighting: sceneAppearance.resolved.lighting,
      lightingControls: sceneAppearance.resolved.lightingOptions,
      background: sceneAppearance.resolved.backgroundHex,
    },
  };
  const [resetTick, setResetTick] = useState(0);
  const resetRequestIdentity = JSON.stringify([
    resetTick,
    preset?.id ?? null,
    preset?.volumeFocus?.centerM ?? null,
  ]);
  const appliedResetRequestIdentityRef = useRef(resetRequestIdentity);
  const [qualityPreview, setQualityPreview] = useState(false);
  const [playbackScreenDistance, setPlaybackScreenDistance] = useState(1);
  // Context restoration and an explicit backend switch both replace every VTK
  // object. Auxiliary hooks use this scene identity so they repopulate the new
  // renderer even when the scientific frame itself did not change.
  const attachContextLossRef = useRef(ctxLoss.attach);
  attachContextLossRef.current = ctxLoss.attach;
  const pane = usePaneContext();
  const fieldOpacityClass = volumeOpacityClass(fieldName);
  const [cameraLodSignal, setCameraLodSignal] = useState<{
    rangeM: number | null;
    projectedRangeM?: number;
    interacting: boolean;
    viewportHeightPx: number;
  }>({ rangeM: null, interacting: false, viewportHeightPx: 620 });
  const [hardwareRam] = useState(() => visualizationResourceGovernor.stats().hardware.systemRamBytes);
  const zoomKey = `${sourceRunDir}|${fieldName}|${pane.voxelBudgetScale}`;
  const [zoomBudget, setZoomBudget] = useState({ key: "", cells: 0 });
  const baseVoxelCap = scaleVoxelBudget(
    fieldOpacityClass === "sparse-positive"
      ? MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS
      : CARTESIAN_VOXEL_CAP_VOLUME,
    pane.voxelBudgetScale,
  );
  // Resolve the small tracker sidecar only after the same scrub debounce as
  // field loading. Extreme domains can then enter directly through the storm
  // ROI instead of first reading a metadata-centred volume and relocating it.
  const paneVoxelCap = Math.min(textureCapacity.maximumCells,
    zoomBudget.key === zoomKey ? Math.max(baseVoxelCap, zoomBudget.cells) : baseVoxelCap);
  useEffect(() => {
    if (cameraLodSignal.interacting) return;
    const timer = window.setTimeout(() => {
      const cells = volumeZoomCellBudget({ baseCells: baseVoxelCap,
        sourceCells: currentStep?.grid?.terrainFollowing ? null : validGridCellCount(currentStep?.grid),
        systemRamBytes: hardwareRam, nativeSpacingM: minimumNominalGridSpacingM(currentStep?.grid) ?? NaN,
        rangeM: cameraLodSignal.projectedRangeM ?? cameraLodSignal.rangeM,
        viewportHeightPx: cameraLodSignal.viewportHeightPx, paneScale: pane.voxelBudgetScale,
        active: zoomBudget.key === zoomKey && zoomBudget.cells > baseVoxelCap });
      setZoomBudget(previous => previous.key === zoomKey && previous.cells === cells ? previous : { key: zoomKey, cells });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [baseVoxelCap, cameraLodSignal, currentStep?.grid, hardwareRam, pane.voxelBudgetScale, zoomKey, zoomBudget.key, zoomBudget.cells]);

  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const stormCenterX = stormCentre?.[0] ?? null;
  const stormCenterY = stormCentre?.[1] ?? null;
  const terrainBlocked = false;
  const boundedProvider = fieldName
    ? fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[fieldName])
    : false;
  const scalarTransform = supportsVolumeStructure(fieldName)
      && backgroundFieldMode === "structure"
    ? "horizontal-level-anomaly" as const
    : "none" as const;
  const effectiveOpacityClass = scalarTransform === "horizontal-level-anomaly"
    ? "signed-perturbation" as const
    : fieldOpacityClass;
  const nestedOwnership = useMemo(() => {
    if (!fieldName || !outputHierarchySource.selection
        || outputHierarchySource.selection.chain.length < 2) return null;
    try {
      return planNestedVolumeOwnership(outputHierarchySource.selection, fieldName);
    } catch {
      return null;
    }
  }, [fieldName, outputHierarchySource.selection]);
  const parentOwnership = nestedOwnership?.at(-2) ?? null;
  const childOwnership = nestedOwnership?.at(-1) ?? null;
  const publishedBrickedVolumeAvailable = Boolean(
    fieldName
      && currentStep?.fieldBrickHierarchies?.[fieldName]?.cullingSafe
      && (scalarTransform === "none"
        || scalarTransform === "horizontal-level-anomaly"),
  );
  const transientBrickedVolumeAvailable = Boolean(
    fieldName
      && !publishedBrickedVolumeAvailable
      && boundedProvider
      && !currentStep?.grid?.terrainFollowing
      && (fieldOpacityClass === "sparse-positive" || fieldOpacityClass === "reflectivity")
      && (validGridCellCount(currentStep?.grid) ?? 0) > CARTESIAN_VOXEL_CAP_VOLUME
      && scalarTransform === "none",
  );
  const transientBudgetKey = JSON.stringify([sourceRunDir, currentStep?.index, currentStep?.publicationGeneration, fieldName, paneVoxelCap]);
  const [overviewFallbackKey, setOverviewFallbackKey] = useState<string | null>(null);
  const boundedOverviewFallback = overviewFallbackKey === transientBudgetKey;
  const brickedVolumeAvailable = publishedBrickedVolumeAvailable
    || (transientBrickedVolumeAvailable && !boundedOverviewFallback);
  const brickedVolumeEligible = brickedVolumeAvailable
    && volumeAppearanceQualification?.progressiveBricks !== false;
  const nestedBrickedEligible = Boolean(
    publishedBrickedVolumeAvailable
      && brickedVolumeEligible
      && parentOwnership
      && childOwnership
      && fieldName
      && parentOwnership.level.step.fieldBrickHierarchies?.[fieldName]?.cullingSafe,
  );
  const brickedMetadataOnly = brickedVolumeAvailable
    && publishedBrickedVolumeAvailable
    && volumeAppearanceQualification?.progressiveBricks === false;
  const brickedTransfer = useMemo(() => ({
    colormap,
    opacityScale: opacityScale
      * (volumeAppearanceQualification?.detailOpacityFactor ?? 1),
    scalarTransform, stormStructure: backgroundFieldMode === "structure",
  }), [backgroundFieldMode, colormap, opacityScale, scalarTransform, volumeAppearanceQualification?.detailOpacityFactor]);
  const [cameraRequestedLevel, setCameraRequestedLevel] = useState<number | null>(null);
  useEffect(() => {
    setCameraRequestedLevel(null);
  }, [sourceRunDir, fieldName, currentStep?.publicationGeneration]);
  const perLayerVoxelCap = nestedBrickedEligible
    ? Math.max(1, Math.floor(paneVoxelCap / 2))
    : paneVoxelCap;
  const brickedPreparation = useBrickedVolumePreparation({
    enabled: textureCapacity.known && ctxLoss.available && (brickedVolumeEligible || brickedMetadataOnly) && !nestedBrickedEligible && !background.loading,
    background: background.profile,
    runDir: sourceRunDir,
    step: currentStep,
    fieldName,
    transfer: brickedTransfer,
    maximumPreparedCells: perLayerVoxelCap,
    maximumTextureDimension: textureCapacity.known ? textureCapacity.maximumDimension : undefined,
    metadataOnly: brickedMetadataOnly,
    compactBricks: volumeAppearanceQualification?.compactBricks,
    requestedLevel: cameraRequestedLevel,
    allowTransientHierarchy: transientBrickedVolumeAvailable,
    opacityClass: fieldOpacityClass,
  });
  useEffect(() => {
    if (brickedVolumeEligible && brickedPreparation.current && brickedPreparation.requiresBoundedOverview) {
      setOverviewFallbackKey(transientBudgetKey);
    }
  }, [brickedVolumeEligible, brickedPreparation.current, brickedPreparation.requiresBoundedOverview, transientBudgetKey]);
  const parentBrickedPreparation = useBrickedVolumePreparation({
    enabled: false,
    runDir: parentOwnership?.level.runDir ?? sourceRunDir,
    step: parentOwnership?.level.step ?? null,
    fieldName,
    transfer: brickedTransfer,
    maximumPreparedCells: perLayerVoxelCap,
    requestedLevel: cameraRequestedLevel,
    opacityClass: fieldOpacityClass,
    excludePhysicalBoundsM: parentOwnership?.excludeChildBoundsM ?? undefined,
  });
  const nestedBrickedPreparation = useNestedBrickedVolumePreparation({
    enabled: textureCapacity.known && ctxLoss.available && nestedBrickedEligible && !background.loading,
    background: background.profile,
    layers: nestedOwnership ?? [],
    fieldName,
    transfer: brickedTransfer,
    maximumPreparedCells: paneVoxelCap,
    maximumTextureDimension: textureCapacity.known ? textureCapacity.maximumDimension : undefined,
    requestedLevel: cameraRequestedLevel,
  });
  const brickedVolumeCurrent = brickedPreparation.current;
  const activeBrickedCurrent = nestedBrickedEligible
    ? nestedBrickedPreparation.current
    : brickedVolumeCurrent;
  const nestedGenerationCurrent = !nestedBrickedEligible || nestedBrickedPreparation.current;
  const composedPreparedBricks = useMemo(() => !brickedVolumeEligible ? [] : nestedBrickedEligible
    ? nestedBrickedPreparation.bricks
    : brickedPreparation.bricks, [
      brickedPreparation.bricks,
      nestedBrickedEligible, brickedVolumeEligible,
      nestedBrickedPreparation.bricks,
    ]);
  const activeBrickedPlan = !(brickedVolumeEligible || brickedMetadataOnly) ? null : nestedBrickedEligible
    ? nestedBrickedPreparation.plans.at(-1)?.plan ?? null
    : brickedPreparation.plan;
  const composedScalarBytes = useMemo(() => uniqueVolumeImages(composedPreparedBricks)
    .reduce((sum, brick) => sum + volumeImageBytes(brick.prepared,
      textureCapacity.known ? textureCapacity.maximumDimension : undefined), 0),
    [composedPreparedBricks, textureCapacity.known, textureCapacity.maximumDimension]);
  const activeBrickedStats = !(brickedVolumeEligible || brickedMetadataOnly) ? null : nestedBrickedEligible
    ? nestedBrickedPreparation.stats
    : brickedPreparation.stats;
  const activeBrickedLoading = !(brickedVolumeEligible || brickedMetadataOnly) ? false : nestedBrickedEligible
    ? nestedBrickedPreparation.loading
    : brickedPreparation.loading;
  const activeBrickedError = !(brickedVolumeEligible || brickedMetadataOnly) || brickedPreparation.requiresBoundedOverview ? null : nestedBrickedEligible
    ? nestedBrickedPreparation.error
    : brickedPreparation.error;
  const nestedParentBrickCount = nestedBrickedEligible && nestedOwnership
    ? composedPreparedBricks.filter((brick) =>
        "levelId" in brick && brick.levelId !== nestedOwnership.at(-1)!.level.levelId).length
    : 0;
  const lodTransitionSignatureRef = useRef<string | null>(null);
  const [lodTransitionEvidence, setLodTransitionEvidence] = useState({
    count: 0,
    retainedGpuBytes: 0,
  });
  useEffect(() => {
    if (activeBrickedCurrent || composedPreparedBricks.length === 0
        || cameraRequestedLevel == null) return;
    const signature = `${currentStep?.publicationGeneration ?? "none"}:${fieldName}:${cameraRequestedLevel}`;
    if (lodTransitionSignatureRef.current === signature) return;
    lodTransitionSignatureRef.current = signature;
    setLodTransitionEvidence((current) => ({
      count: current.count + 1,
      retainedGpuBytes: visualizationResourceGovernor.stats().byStage.gpu.residentBytes,
    }));
  }, [
    activeBrickedCurrent,
    composedPreparedBricks.length,
    cameraRequestedLevel,
    currentStep?.publicationGeneration,
    fieldName,
  ]);
  useEffect(() => {
    const hierarchy = brickedPreparation.hierarchy;
    const grid = currentStep?.grid;
    if (!brickedVolumeEligible || !hierarchy || !grid || cameraLodSignal.rangeM == null) {
      if (!brickedVolumeEligible && cameraRequestedLevel != null) setCameraRequestedLevel(null);
      return;
    }
    const nativeSpacingM = Math.max(
      grid.drM ?? 1,
      grid.dyM ?? grid.drM ?? 1,
      grid.dzM ?? 1,
    );
    const next = selectCameraVolumeLevel(hierarchy.levels, {
      nativeSpacingM,
      cameraRangeM: cameraLodSignal.projectedRangeM ?? cameraLodSignal.rangeM,
      viewportHeightPx: cameraLodSignal.viewportHeightPx,
      currentLevel: cameraRequestedLevel ?? brickedPreparation.plan?.level ?? 0,
      interacting: cameraLodSignal.interacting,
    });
    if (cameraRequestedLevel == null && brickedPreparation.plan?.level === next) return;
    if (next !== cameraRequestedLevel) setCameraRequestedLevel(next);
  }, [
    brickedPreparation.hierarchy,
    brickedPreparation.plan?.level,
    brickedVolumeEligible,
    cameraLodSignal,
    cameraRequestedLevel,
    currentStep?.grid,
  ]);
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      scalarTransform,
      effectiveOpacityClass,
      volumeRaySampling: "pixel-anchored-physical-distance",
      sourcePlaneFocus: preset?.volumeFocus?.centerM ?? null,
      qualificationOverrides: volumeAppearanceQualification ? {
        contextOpacityFactor: volumeAppearanceQualification.contextOpacityFactor ?? null,
        detailOpacityFactor: volumeAppearanceQualification.detailOpacityFactor ?? null,
        opacityReferenceM: volumeAppearanceQualification.opacityReferenceM ?? null,
        sampleDistanceFactor: volumeAppearanceQualification.sampleDistanceFactor ?? null,
        previewDistanceMultiplier: volumeAppearanceQualification.previewDistanceMultiplier ?? null,
        lighting: volumeAppearanceQualification.lighting ?? null,
        overviewAggregation: volumeAppearanceQualification.overviewAggregation ?? null,
        nativeDetail: volumeAppearanceQualification.nativeDetail ?? null,
        frameNativeDetail: volumeAppearanceQualification.frameNativeDetail ?? null,
        progressiveBricks: volumeAppearanceQualification.progressiveBricks ?? null,
      } : null,
    },
  };
  const featureIdentity = `${sourceRunDir}|${fieldName ?? ""}|${currentStep?.publicationGeneration ?? currentStep?.index ?? "none"}`;
  const [locatedFeature, setLocatedFeature] = useState<{
    identity: string;
    center: [number, number];
    supportBounds: [number, number, number, number, number, number] | null;
  } | null>(null);
  const locatedFeatureCenter = locatedFeature?.identity === featureIdentity
    ? locatedFeature.center
    : null;
  const locatedSupportBounds = locatedFeature?.identity === featureIdentity
    ? locatedFeature.supportBounds
    : null;
  const preferredDetailCenter = useMemo(
    () => handoffFocusXY
      ?? (stormCenterX != null && stormCenterY != null
      ? [stormCenterX, stormCenterY] as [number, number]
      : locatedFeatureCenter),
    [handoffFocusXY, locatedFeatureCenter, stormCenterX, stormCenterY],
  );
  const boundedWindowOutcome = useMemo(
    () => planInteractiveVolumeWindows(
      currentStep?.grid,
      paneVoxelCap,
      preferredDetailCenter,
      {
        preferNativeDetail: fieldOpacityClass === "sparse-positive"
          && preferredDetailCenter != null
          && volumeAppearanceQualification?.nativeDetail !== false,
        structuralSupportBoundsM: locatedSupportBounds ? {
          lower: [locatedSupportBounds[0], locatedSupportBounds[2], locatedSupportBounds[4]],
          upper: [locatedSupportBounds[1], locatedSupportBounds[3], locatedSupportBounds[5]],
        } : null,
      },
    ),
    [
      currentStep?.grid,
      fieldOpacityClass,
      paneVoxelCap,
      preferredDetailCenter,
      volumeAppearanceQualification?.nativeDetail,
      locatedSupportBounds,
    ],
  );
  const windowAggregation = volumeAppearanceQualification?.overviewAggregation
    ?? volumeFieldWindowAggregation(fieldName, scalarTransform);
  const boundedWindows = fieldName && boundedProvider && boundedWindowOutcome.ok
    ? coalesceNativeFieldWindows(boundedWindowOutcome.plan.windows.map((window) => ({
        ...window,
        aggregation: windowAggregation,
      })), {
        sourceKind: currentStep?.fieldOrigins?.[fieldName],
        terrainFollowing: currentStep?.grid?.terrainFollowing,
      })
    : null;
  const hybridDetailActive = !brickedVolumeEligible && boundedWindowOutcome.ok
    && boundedWindowOutcome.mode === "native-detail";
  const fullNativeInteractionPreview = !brickedVolumeEligible
    && boundedWindowOutcome.ok
    && boundedWindowOutcome.mode === "full-domain"
    && boundedWindowOutcome.plan.stride.every((stride) => stride === 1)
    && boundedWindowOutcome.plan.sampledCells > CARTESIAN_VOXEL_CAP_VOLUME;
  const contextVolumeRequired = hybridDetailActive || fullNativeInteractionPreview;
  const fullNativePreviewRef = useRef(fullNativeInteractionPreview);
  fullNativePreviewRef.current = fullNativeInteractionPreview;
  const nativePreviewDetachedRef = useRef(false);
  const sparseReducedOverviewActive = !brickedVolumeEligible
    && fieldOpacityClass === "sparse-positive"
    && boundedWindowOutcome.ok
    && boundedWindowOutcome.mode === "full-domain"
    && boundedWindowOutcome.plan.stride.some((stride) => stride > 1);
  const sparseOverviewRef = useRef(sparseReducedOverviewActive);
  sparseOverviewRef.current = sparseReducedOverviewActive;
  const mainOpacityFactor = volumeAppearanceQualification?.detailOpacityFactor ?? 1;
  const contextOpacityFactor = volumeAppearanceQualification?.contextOpacityFactor
    ?? VOLUME_CONTEXT_OPACITY_FACTOR;
  const effectiveMainOpacityScale = opacityScale * mainOpacityFactor
    * (sparseReducedOverviewActive ? contextOpacityFactor : 1);
  const overviewWindowOutcome = useMemo(
    () => planBoundedVolumeWindows(
      currentStep?.grid,
      scaleVoxelBudget(CARTESIAN_VOXEL_CAP_VOLUME, pane.voxelBudgetScale),
    ),
    [currentStep?.grid, pane.voxelBudgetScale],
  );
  const overviewWindows = fieldName && boundedProvider && contextVolumeRequired
      && overviewWindowOutcome.ok
    ? coalesceNativeFieldWindows(overviewWindowOutcome.plan.windows.map((window) => ({
        ...window,
        aggregation: windowAggregation,
      })), {
        sourceKind: currentStep?.fieldOrigins?.[fieldName],
        terrainFollowing: currentStep?.grid?.terrainFollowing,
      })
    : null;
  const contextWindowError = contextVolumeRequired && !overviewWindowOutcome.ok
    ? overviewWindowOutcome.reason : null;
  const detailPlan = boundedWindowOutcome.ok
      && isNativeStormDetailWindowPlan(boundedWindowOutcome.plan)
    ? boundedWindowOutcome.plan
    : null;
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      representation: brickedVolumeEligible
        ? "progressive-native-bricks"
        : hybridDetailActive ? "hybrid-overview-native-detail" : "full-domain",
      brickHierarchySource: brickedVolumeEligible
        ? publishedBrickedVolumeAvailable ? "published" : "transient"
        : null,
      reducedContextOnly: sparseReducedOverviewActive,
      boundedOverviewFallback,
      effectiveOpacityScale: effectiveMainOpacityScale,
      opacityPoints: colorRange
        ? volumeOpacityPoints({
            kind: transferKind,
            opacityClass,
            min: colorRange.min,
            max: colorRange.max,
            ...(opacityMaximum == null ? {} : { opacityMax: opacityMaximum }),
          }, effectiveMainOpacityScale).map((point) => [point.scalar, point.opacity])
        : [],
      windowAggregation,
      sampledShape: boundedWindowOutcome.ok ? boundedWindowOutcome.plan.sampledShape : null,
      sourceShape: boundedWindowOutcome.ok ? boundedWindowOutcome.plan.sourceShape : null,
      fullSourceShape: overviewWindowOutcome.ok
        ? overviewWindowOutcome.plan.fullSourceShape ?? overviewWindowOutcome.plan.sourceShape
        : boundedWindowOutcome.ok
          ? boundedWindowOutcome.plan.fullSourceShape ?? boundedWindowOutcome.plan.sourceShape
          : null,
      stride: boundedWindowOutcome.ok ? boundedWindowOutcome.plan.stride : null,
      detailCenterM: detailPlan?.detailCenterM ?? null,
      detailIndexLower: detailPlan?.detailIndexRange.lower ?? null,
      detailIndexUpper: detailPlan?.detailIndexRange.upper ?? null,
      detailSpanM: detailPlan?.detailSpanM ?? null,
      structuralSupportBounds: locatedSupportBounds,
      overviewSampledShape: overviewWindowOutcome.ok
        ? overviewWindowOutcome.plan.sampledShape
        : null,
      overviewStride: overviewWindowOutcome.ok ? overviewWindowOutcome.plan.stride : null,
      brickHierarchyLevel: activeBrickedPlan?.level ?? null,
      volumeHaloSamples: activeBrickedPlan?.haloSamples ?? 0,
      selectedBrickCount: composedPreparedBricks.length || null,
      backgroundCulledBricks: background.profile ? Math.max(0, (activeBrickedPlan?.items.length ?? 0) - composedPreparedBricks.length) : 0,
      transparentBrickCount: brickedPreparation?.plan?.transparentBrickCount ?? null,
      boundaryOnlyBrickCount: activeBrickedPlan?.boundaryOnlyBrickCount ?? 0,
      nestedVolumeLayers: nestedBrickedEligible ? nestedOwnership?.length ?? 1 : 1,
      parentExcludedBrickCount: parentBrickedPreparation.plan?.excludedBrickCount ?? 0,
      cameraRequestedLevel,
      cameraLodInteracting: cameraLodSignal.interacting,
      zoomNativeData: paneVoxelCap > baseVoxelCap,
      volumeCellBudget: paneVoxelCap,
      cameraRangeM: cameraLodSignal.rangeM,
      lodRetainedTransitionCount: lodTransitionEvidence.count,
      lodRetainedGpuBytes: lodTransitionEvidence.retainedGpuBytes,
    },
  };
  const boundedLoadError = currentStep && fieldName && !terrainBlocked && !brickedVolumeEligible
    ? !boundedProvider
      ? "This field provider cannot serve bounded volume windows."
      : !boundedWindowOutcome.ok
        ? boundedWindowOutcome.reason
        : null
    : null;

  // Re-apply preset config when the preset id changes.
  useEffect(() => {
    if (!preset) return;
    if (preset.field) setFieldName(preset.field);
    if (isVtkColormap(preset.colormap)) {
      setColormap(preset.colormap as ColormapPreset);
    }
  }, [preset]);

  const colorInitedFieldRef = useRef<string | null>(null);
  useEffect(() => {
    if (!fieldName || colorInitedFieldRef.current === fieldName) return;
    colorInitedFieldRef.current = fieldName;
    const requested = preset?.field === fieldName && isVtkColormap(preset.colormap)
      ? preset.colormap!
      : defaultVolumeColormap(fieldName);
    setColormap(requested as ColormapPreset);
  }, [fieldName, preset]);

  // The control cannot offer dBZ on a non-reflectivity field, but a restored
  // session or preset can still arrive holding that pairing. Fall back rather
  // than render the field against a ramp that cannot describe it.
  useEffect(() => {
    if (colormap === "dBZ" && !isReflectivityField(fieldName)) {
      setColormap(defaultVolumeColormap(fieldName) as ColormapPreset);
    }
  }, [colormap, fieldName]);

  const clearPreset = () => onPresetCleared?.();

  // Follows a linked pane only into a field with a volume; otherwise it keeps its own and
  // `declinedLink` explains the refusal on the canvas.
  const declinedLink = useFieldLink(
    linkedField, fieldName, (f) => renderableFields.includes(f), setFieldName,
  );

  // Re-pick a default when the field list changes (e.g. user switches to a different run with a
  // different schema), or when a preset carried in a field this renderer cannot draw.
  useEffect(() => {
    if (fieldName == null || !renderableFields.includes(fieldName)) {
      setFieldName(defaultField);
    }
  }, [defaultField, fieldName, renderableFields]);

  // Debounce the scrubbed step so a fast slider drag loads only the settled
  // bounded volume and does not prepare every intermediate frame.
  const fieldResult = useField(
    sourceRunDir,
    !textureCapacity.known || !ctxLoss.available || terrainBlocked || boundedLoadError || outputHierarchySource.loading || brickedVolumeEligible
      ? null : loadFrame.stepIndex,
    !textureCapacity.known || !ctxLoss.available || terrainBlocked || boundedLoadError || outputHierarchySource.loading || brickedVolumeEligible
      ? null : fieldName,
    loadFrame.publicationGeneration,
    boundedWindows,
    { prefetch: !fullNativeInteractionPreview },
  );
  const overviewFieldResult = useField(
    sourceRunDir,
    textureCapacity.known && ctxLoss.available && contextVolumeRequired && !contextWindowError && !boundedLoadError && !outputHierarchySource.loading
      ? loadFrame.stepIndex
      : null,
    textureCapacity.known && ctxLoss.available && contextVolumeRequired && !contextWindowError && !boundedLoadError && !outputHierarchySource.loading
      ? fieldName
      : null,
    loadFrame.publicationGeneration,
    overviewWindows,
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
  const loadedStep = brickedVolumeEligible ? currentStep : findPublishedFrame(
      sourceSteps,
      fieldResult.loadedStepIndex,
      fieldResult.loadedPublicationGeneration,
    );
  const renderGrid = loadedStep?.grid ?? null;
  const [sceneScale, setSceneScale] = useSceneScale(renderGrid, undefined, controlScope.scope, "volume");
  const sceneScaleRef = useRef(sceneScale);
  sceneScaleRef.current = sceneScale;
  const latestScaleFraming = useSceneScaleFraming(sceneScale);
  const scaleFramingRef = useRef(latestScaleFraming);
  scaleFramingRef.current = latestScaleFraming;
  const logicalFrameBounds = useMemo(() => {
    const bounds = domainBoundsM(renderGrid);
    return bounds ? scaledFeatureBounds([
      bounds.lower[0], bounds.upper[0],
      bounds.lower[1], bounds.upper[1],
      bounds.lower[2], bounds.upper[2],
    ], sceneScale) : null;
  }, [renderGrid, sceneScale]);
  const initialFramingRef = useRef({ bounds: logicalFrameBounds, focus: handoffFocusCenter });
  initialFramingRef.current = { bounds: logicalFrameBounds, focus: handoffFocusCenter };
  const preparation = usePreparedScalarImage(
    brickedVolumeEligible || background.loading ? null : fieldResult.field,
    renderGrid,
    /* In pane mode: explicit divided budget; four full-budget volumes exhausts VRAM. */
    paneVoxelCap,
    ctxLoss.generation,
    {
      refineForDisplay: renderFidelityMode === "presentation",
      featureSign: fieldOpacityClass === "sparse-positive" ? 1 : undefined,
      scalarTransform,
      background: background.profile ?? undefined, cropBackground: true,
      structuralSupportRangeFraction: fieldOpacityClass === "sparse-positive" ? 0.9 : undefined,
    },
  );
  const overviewPreparation = usePreparedScalarImage(
    background.loading ? null : overviewFieldResult.field,
    renderGrid,
    paneVoxelCap,
    ctxLoss.generation,
    {
      refineForDisplay: false,
      scalarTransform,
      background: background.profile ?? undefined, cropBackground: true,
      structuralSupportRangeFraction: fieldOpacityClass === "sparse-positive" ? 0.9 : undefined,
    },
  );
  const monolithicGpuUploadBytes = brickedVolumeEligible ? 0
    : (volumeImageBytes(preparation.prepared, textureCapacity.known ? textureCapacity.maximumDimension : undefined))
      + (contextVolumeRequired ? volumeImageBytes(overviewPreparation.prepared, textureCapacity.known ? textureCapacity.maximumDimension : undefined) : 0);
  const monolithicGpuUpload = useVtkGpuResourceOwnership({
    enabled: !brickedVolumeEligible && preparation.prepared != null
      && (!contextVolumeRequired || overviewPreparation.prepared != null)
      && monolithicGpuUploadBytes > 0,
    owner: "volume",
    generation: JSON.stringify([
      sourceRunDir, fieldResult.loadedStepIndex, fieldResult.loadedPublicationGeneration,
      fieldName, sourceDefinitionIdentity, contextVolumeRequired, vtkLifecycleGeneration,
    ]),
    bytes: monolithicGpuUploadBytes,
  });
  const backgroundGpuBytes = useMemo(() => {
    if (!background.profile) return 0;
    const images = brickedVolumeEligible ? composedPreparedBricks.map(brick => brick.prepared)
      : [preparation.prepared, contextVolumeRequired ? overviewPreparation.prepared : null].filter((image): image is NonNullable<typeof image> => image != null);
    const spans = new Map(images.map(image => [[image.origin[2], image.spacing[2], image.shape[2]].join(":"), image.shape[2] * 4]));
    return [...spans.values()].reduce((sum, bytes) => sum + bytes, 0);
  }, [background.profile, brickedVolumeEligible, composedPreparedBricks, preparation.prepared, contextVolumeRequired, overviewPreparation.prepared]);
  const backgroundGpu = useVtkGpuResourceOwnership({ enabled: backgroundGpuBytes > 0, owner: "volume-background",
    generation: JSON.stringify([sourceRunDir, currentStep?.index, currentStep?.publicationGeneration, fieldName, background.profile]), bytes: backgroundGpuBytes });
  const backgroundAdmitted = backgroundGpuBytes === 0 || backgroundGpu.admitted;
  const scalarEvidence = useMemo(() => lazyAppearanceEvidence(() => {
    const arrays = brickedVolumeEligible
      ? composedPreparedBricks.flatMap((brick) => volumeOwnedScalarRows(brick.prepared))
      : preparation.prepared ? [preparation.prepared.values] : [];
    return {
      scalarCompositionFingerprint: arrays.length ? scalarMultisetFingerprint(arrays) : null,
      scalarCompositionQuantizedFingerprint: arrays.length ? scalarQuantizedMultisetFingerprint(arrays) : null,
    };
  }), [brickedVolumeEligible, composedPreparedBricks, preparation.prepared]);
  const scalarEvidenceRef = useRef(scalarEvidence);
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      cameraFocusBounds: preparation.prepared?.structuralSupportBounds
        ?? preparation.prepared?.absoluteFocusBounds
        ?? null,
      logicalGpuUploadBytes: brickedVolumeEligible
        ? composedScalarBytes
        : monolithicGpuUpload.admitted ? monolithicGpuUploadBytes : 0,
    },
  };
  const volumeFocusRef = useRef({
    absoluteFocusBounds: preparation.prepared?.absoluteFocusBounds,
    structuralSupportBounds: preparation.prepared?.structuralSupportBounds,
    structural: effectiveOpacityClass === "sparse-positive"
      || effectiveOpacityClass === "signed-perturbation",
  });
  volumeFocusRef.current = {
    absoluteFocusBounds: preparation.prepared?.absoluteFocusBounds,
    structuralSupportBounds: preparation.prepared?.structuralSupportBounds,
    structural: effectiveOpacityClass === "sparse-positive"
      || effectiveOpacityClass === "signed-perturbation",
  };
  const preparedRef = useRef(preparation.prepared);
  appearanceEvidenceRef.current.parameters = { ...appearanceEvidenceRef.current.parameters,
    backgroundCropped: preparation.prepared?.focusCropped ?? false, backgroundProfileGpuBytes: backgroundGpuBytes };
  preparedRef.current = preparation.prepared;
  const floorSelection = useRadarFloorSelection(
    fieldsForStep,
    controlScope.scope,
    `volume:${sourceRunDir}`,
  );
  const expectedMaterialIdentity = fieldName && !background.loading && backgroundAdmitted && (
    brickedVolumeEligible
      ? activeBrickedCurrent && nestedGenerationCurrent && activeBrickedStats != null
      : fieldResult.field && preparation.prepared
  )
    ? JSON.stringify([
        sourceRunDir,
        brickedVolumeEligible ? currentStep?.index ?? null : fieldResult.loadedStepIndex,
        brickedVolumeEligible
          ? currentStep?.publicationGeneration ?? null
          : fieldResult.loadedPublicationGeneration,
        fieldName,
        sourceDefinitionIdentity,
        colormap,
        opacityScale,
        scalarTransform,
        backgroundFieldMode, background.profile,
        sceneAppearance.resolved.lightingOptions,
        volumeAppearanceQualification ?? null,
        floorSelection.enabled, floorSelection.fieldName, floorSelection.reduction,
        fieldDefinitionIdentity(sourceRunDir, requestedStepIndex, floorSelection.fieldName, requestedPublicationGeneration),
        sparseReducedOverviewActive,
        activeBrickedPlan?.level ?? null,
      ])
    : null;
  const [committedMaterialIdentity, setCommittedMaterialIdentity] = useState<string | null>(null);
  const [committedContextMaterialIdentity, setCommittedContextMaterialIdentity] =
    useState<string | null>(null);
  const pendingMaterialIdentityRef = useRef<string | null>(null);
  const pendingContextMaterialIdentityRef = useRef<string | null>(null);
  const materialCommitted = expectedMaterialIdentity != null
    && committedMaterialIdentity === expectedMaterialIdentity
    && (!contextVolumeRequired || committedContextMaterialIdentity === expectedMaterialIdentity);

  useEffect(() => {
    if (fieldOpacityClass !== "sparse-positive"
        || !boundedWindowOutcome.ok
        || boundedWindowOutcome.mode !== "full-domain") return;
    const peak = preparation.prepared?.featurePeakPosition;
    if (!peak || !peak.slice(0, 2).every(Number.isFinite)) return;
    const center: [number, number] = [peak[0], peak[1]];
    const supportBounds = preparation.prepared?.structuralSupportBounds ?? null;
    setLocatedFeature((current) => current?.identity === featureIdentity
      && current.center[0] === center[0] && current.center[1] === center[1]
      && JSON.stringify(current.supportBounds) === JSON.stringify(supportBounds)
      ? current
      : { identity: featureIdentity, center, supportBounds });
  }, [boundedWindowOutcome, featureIdentity, fieldOpacityClass, preparation.prepared]);

  useEffect(() => {
    setRenderedStepIndex(null);
    setRenderedPublicationGeneration(null);
    setCommittedMaterialIdentity(null);
    setCommittedContextMaterialIdentity(null);
  }, [sourceRunDir, fieldName, sourceDefinitionIdentity]);


  // ---------- vtk.js scene refs ----------
  const containerRef = useRef<HTMLDivElement | null>(null);
  type SceneRefs = {
    renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    imageData: ReturnType<typeof vtkImageData.newInstance>;
    volume: ReturnType<typeof vtkVolume.newInstance>;
    mapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
    ctf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    pwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>;
    contextImageData: ReturnType<typeof vtkImageData.newInstance>;
    contextVolume: ReturnType<typeof vtkVolume.newInstance>;
    contextMapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
    contextCtf: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    contextPwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>;
    quality: VtkVolumeQualityController;
    physicalOpacityDistance: number;
    physicalSampleDistance: number;
    contextPhysicalSampleDistance: number;
    lighting: VtkSceneLighting;
    brickOrder: VolumeOrder;
    appliedBrickOrder: string;
    brickVolumes: Array<{
      imageData: ReturnType<typeof vtkImageData.newInstance>;
      mapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
      volume: ReturnType<typeof vtkVolume.newInstance>;
      unregisterQuality: () => void;
      physicalSampleDistance: number;
      ownedBounds: VolumeBounds;
      clipPlanes: ReturnType<typeof vtkPlane.newInstance>[];
    }>;
  };
  const sceneRef = useRef<SceneRefs | null>(null);
  const updateRayDistances = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const camera = scene.renderWindow.getRenderer().getActiveCamera();
    const update = (volume: SceneRefs["volume"], mapper: SceneRefs["mapper"], sample: number, opacity: number) => {
      setCameraAdjustedVolumeRayDistances(
        volume.getProperty(), camera, scene.quality, opacity, sample, sceneScaleRef.current, mapper,
      );
    };
    update(scene.volume, scene.mapper, scene.physicalSampleDistance, scene.physicalOpacityDistance);
    update(scene.contextVolume, scene.contextMapper, scene.contextPhysicalSampleDistance, VOLUME_OPACITY_REFERENCE_M);
    for (const brick of scene.brickVolumes) {
      update(brick.volume, brick.mapper, brick.physicalSampleDistance, scene.physicalOpacityDistance);
    }
    const scale = sceneScaleRef.current;
    const eye = camera.getPosition().map((value, axis) => value / (axis === 2 ? scale.z : scale.xy));
    const focal = camera.getFocalPoint().map((value, axis) => value / (axis === 2 ? scale.z : scale.xy));
    const rangeM = Math.hypot(...eye.map((value, axis) => value - focal[axis]));
    const projectedRangeM = camera.getParallelProjection()
      ? camera.getParallelScale() / scale.xy / Math.tan(Math.PI / 12)
      : rangeM * Math.tan(camera.getViewAngle() * Math.PI / 360) / Math.tan(Math.PI / 12);
    const viewportHeightPx = containerRef.current?.querySelector("canvas")?.height ?? 620;
    setCameraLodSignal(current => current.rangeM === rangeM && current.projectedRangeM === projectedRangeM
      && current.viewportHeightPx === viewportHeightPx ? current : { ...current, rangeM, projectedRangeM, viewportHeightPx });
    const order = orderedVolumeIndices(scene.brickOrder, eye,
      camera.getParallelProjection() ? camera.getDirectionOfProjection() : undefined);
    const key = order.join(":");
    if (key !== scene.appliedBrickOrder) {
      const renderer = scene.renderWindow.getRenderer();
      for (const brick of scene.brickVolumes) renderer.removeVolume(brick.volume);
      for (const index of order) renderer.addVolume(scene.brickVolumes[index].volume);
      scene.appliedBrickOrder = key;
    }
  }, []);
  const cameraInitializedRef = useRef(false);
  const cameraTouchedRef = useRef(false);
  const framedDetailIdentityRef = useRef<string | null>(null);
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
      onCameraApplied: () => {
        cameraInitializedRef.current = true;
        const scene = sceneRef.current;
        if (!scene) return;
        setCameraAdjustedVolumeRayDistances(
          scene.volume.getProperty(),
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          scene.physicalOpacityDistance,
          scene.physicalSampleDistance,
          sceneScaleRef.current,
          scene.mapper,
        );
        if (scene.contextVolume.getVisibility()) {
          setCameraAdjustedVolumeRayDistances(
            scene.contextVolume.getProperty(),
            scene.renderWindow.getRenderer().getActiveCamera(),
            scene.quality,
            VOLUME_OPACITY_REFERENCE_M,
            scene.contextPhysicalSampleDistance,
            sceneScaleRef.current,
            scene.contextMapper,
          );
        }
      },
      onRemoteInteractionChange: (active) => {
        const quality = sceneRef.current?.quality;
        if (active) quality?.begin("synchronization");
        else quality?.end("synchronization");
      },
    },
  );

  /* Radar floor: own 2D hyperslab path, resolution independent of volume voxel cap. */
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
  const contextPreparationError = contextVolumeRequired
    ? contextWindowError ?? overviewFieldResult.error ?? overviewPreparation.error : null;
  const fieldPreparationError = textureCapacity.error ?? background.error ?? backgroundGpu.error
    ?? fieldResult.error ?? preparation.error ?? contextPreparationError ?? activeBrickedError ?? monolithicGpuUpload.error;
  const panePreparationError = boundedLoadError ?? fieldPreparationError ?? floor.error;
  const panePreparationReady = !panePreparationError && !background.loading && backgroundAdmitted
    && loadedStep != null && loadedStep.index === requestedStepIndex
    && loadedStep.publicationGeneration === requestedPublicationGeneration
    && (brickedVolumeEligible
      ? activeBrickedCurrent && nestedGenerationCurrent && !activeBrickedLoading
      : !fieldResult.loading && !preparation.loading && preparation.prepared != null && monolithicGpuUpload.admitted
        && (!contextVolumeRequired || (!overviewPreparation.loading && overviewPreparation.prepared != null
          && !overviewFieldResult.loading && overviewFieldResult.loadedStepIndex === requestedStepIndex
          && overviewFieldResult.loadedPublicationGeneration === requestedPublicationGeneration)))
    && (!floorSelection.enabled || (!floor.loading && floor.values != null));
  const panePresentation = usePanePresentationBarrier({ ready: panePreparationReady, error: panePreparationError });
  const panePresentationPending = panePresentation.pending || !panePreparationReady;
  const committedGridRef = useRef(renderGrid);
  const displayGrid = panePresentationPending ? committedGridRef.current : renderGrid;
  const releasePresentationRef = useRef<(() => void) | null>(null);
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || panePresentationPending) return;
    const release = holdVtkSceneRendering(scene.renderWindow);
    releasePresentationRef.current = release;
    return () => { release(); if (releasePresentationRef.current === release) releasePresentationRef.current = null; };
  }, [panePresentationPending, expectedMaterialIdentity, vtkLifecycleGeneration]);
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
      renderedStepIndex: panePreparationError ? null : renderedStepIndex,
      renderedPublicationGeneration: panePreparationError ? null : renderedPublicationGeneration,
      loading: !panePreparationError && !terrainBlocked
        && (
          background.loading || (brickedVolumeEligible
            ? activeBrickedLoading
            : fieldResult.loading || preparation.loading)
          || frameTransition.kind !== "ready"
          || !materialCommitted
          || panePresentationPending || floor.loading || floor.error != null
          || floorSelection.enabled && floor.committedPresentationIdentity !== floor.presentationIdentity
        ),
    });
  }, [
    panePresentationPending, floor.loading, floor.error, floorSelection.enabled,
    floor.committedPresentationIdentity, floor.presentationIdentity,
    onFrameDisplayChange,
    requestedStepIndex,
    requestedPublicationGeneration,
    renderedStepIndex,
    renderedPublicationGeneration,
    fieldResult.loading,
    preparation.loading,
    brickedVolumeEligible,
    activeBrickedLoading,
    parentBrickedPreparation.loading,
    frameTransition.kind,
    materialCommitted,
    background.loading,
    terrainBlocked,
    boundedLoadError,
    panePreparationError,
  ]);

  const cameraElevation = useCameraElevation(displayGrid);
  useCameraElevationPlacement(() => sceneRef.current?.renderWindow ?? null, {
    grid: displayGrid,
    scale: sceneScale,
    selection: cameraElevation,
    targetXYM: stormCentre ? [stormCentre[0], stormCentre[1]] : null,
    lifecycleGeneration: vtkLifecycleGeneration,
    onCameraApplied: () => {
      cameraInitializedRef.current = true;
      const scene = sceneRef.current;
      if (!scene) return;
      setCameraAdjustedVolumeRayDistances(
        scene.volume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        scene.physicalOpacityDistance,
        scene.physicalSampleDistance,
        sceneScaleRef.current,
        scene.mapper,
      );
      if (scene.contextVolume.getVisibility()) {
        setCameraAdjustedVolumeRayDistances(
          scene.contextVolume.getProperty(),
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          VOLUME_OPACITY_REFERENCE_M,
          scene.contextPhysicalSampleDistance,
          sceneScaleRef.current,
          scene.contextMapper,
        );
      }
    },
  });
  // Reset once after first data load (empty bounds NaN the camera); subsequent field changes preserve user orbit.

  // Re-frame on run change; a different run can have a very different domain extent.
  useVtkRunReset(sourceRunDir, cameraInitializedRef, () => {
    cameraTouchedRef.current = false;
    framedDetailIdentityRef.current = null;
    // Hide stale volume on run switch; re-shown once new run's frame loads.
    const s = sceneRef.current;
    if (!s) return;
    s.volume.setVisibility(false);
    s.contextVolume.setVisibility(false);
    for (const brick of s.brickVolumes) brick.volume.setVisibility(false);
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
    cameraTouchedRef.current = false;
    framedDetailIdentityRef.current = null;
    renderWindow.setContainer(container);
    const clearBackendEvidence = exposeVtkBackendEvidence(
      container,
      renderWindow,
      vtkBackend.backend,
      vtkBackend.confirmActiveApi,
    );
    textureCapacity.probe(renderWindow);

    const imageData = vtkImageData.newInstance();
    const mapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(mapper);
    // Configure sample params BEFORE setInputData; mapper caches its texture allocator on first frame.
    // Conservative 5000 m start; tightened per-field once spacing is known.
    mapper.setSampleDistance(5000);
    // Headroom for a tall domain under vertical exaggeration.
    mapper.setMaximumSamplesPerRay(3000);
    // Nimbus owns an explicit quality controller; disable vtk.js implicit heuristic.
    mapper.setAutoAdjustSampleDistances(false);
    mapper.setInputData(imageData);

    const ctf = vtkColorTransferFunction.newInstance();
    const pwf = vtkPiecewiseFunction.newInstance();

    const volume = vtkVolume.newInstance();
    volume.setMapper(mapper);
    // Hidden until per-field effect sets real data + safe sample distance (empty+shade=true can hang GPU).
    volume.setVisibility(false);
    const property = volume.getProperty();
    property.setRGBTransferFunction(0, ctf);
    property.setScalarOpacity(0, pwf);
    property.setInterpolationTypeToLinear();
    // Opacity gradients turn transfer-function knots into artificial shaded
    // surfaces and add six opacity lookups per lit sample. Scalar gradients
    // preserve resolved structure without coupling geometry to UI opacity.
    property.setComputeNormalFromOpacity(false);
    // No shading initially; gradient on empty/just-loaded data is unstable. Enabled after first frame.
    property.setShade(false);
    property.setAmbient(0.4);
    property.setDiffuse(0.6);
    property.setSpecular(0.0);

    const contextImageData = vtkImageData.newInstance();
    const contextMapper = vtkVolumeMapper.newInstance();
    configureCoherentVolumeSampling(contextMapper);
    contextMapper.setSampleDistance(5000);
    contextMapper.setMaximumSamplesPerRay(1800);
    contextMapper.setAutoAdjustSampleDistances(false);
    contextMapper.setInputData(contextImageData);
    const contextCtf = vtkColorTransferFunction.newInstance();
    const contextPwf = vtkPiecewiseFunction.newInstance();
    const contextVolume = vtkVolume.newInstance();
    contextVolume.setMapper(contextMapper);
    contextVolume.setVisibility(false);
    const contextProperty = contextVolume.getProperty();
    contextProperty.setRGBTransferFunction(0, contextCtf);
    contextProperty.setScalarOpacity(0, contextPwf);
    contextProperty.setInterpolationTypeToLinear();
    contextProperty.setComputeNormalFromOpacity(false);
    contextProperty.setShade(false);
    contextProperty.setAmbient(1);
    contextProperty.setDiffuse(0);
    contextProperty.setSpecular(0);

    const renderer = renderWindow.getRenderer();
    const lighting = createVtkSceneLighting(renderer);
    lighting.apply(
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );
    renderer.addVolume(contextVolume);
    renderer.addVolume(volume);
    // No resetCamera() yet: empty ImageData has no bounds; defer to per-field effect.

    const quality = createVtkVolumeQualityController({
      mapper: [contextMapper, mapper],
      render: () => renderWindow.getRenderWindow().render(),
      previewDistanceMultiplier: volumeAppearanceQualification?.previewDistanceMultiplier ?? 2,
      previewImageSampleDistance: 2,
      flatPreview: false,
      previewMaximumSamples: 900,
      idleMs: 140,
      onPreviewChange: (preview) => {
        const current = sceneRef.current;
        if (current?.renderWindow !== renderWindow) return;
        if (current?.renderWindow === renderWindow) {
          applyVolumeInteractionLighting(
            current.volume.getProperty(),
            preview,
            sceneAppearanceRef.current.lightingOptions,
            sparseOverviewRef.current ? "signed-perturbation" : opacityClassRef.current,
          );
          applyVolumeInteractionLighting(
            current.contextVolume.getProperty(),
            preview,
            sceneAppearanceRef.current.lightingOptions,
            "signed-perturbation",
          );
          for (const brick of current.brickVolumes) {
            applyVolumeInteractionLighting(
              brick.volume.getProperty(),
              preview,
              sceneAppearanceRef.current.lightingOptions,
              opacityClassRef.current,
            );
          }
        }
        if (current?.renderWindow === renderWindow
            && fullNativePreviewRef.current) {
          if (preview && !nativePreviewDetachedRef.current) {
            renderer.removeVolume(current.volume);
            nativePreviewDetachedRef.current = true;
          } else if (!preview && nativePreviewDetachedRef.current) {
            renderer.addVolume(current.volume);
            nativePreviewDetachedRef.current = false;
          }
          current.volume.setVisibility(true);
          current.contextVolume.setVisibility(preview);
        }
        setQualityPreview(preview);
      },
    });
    const brickVolumes: SceneRefs["brickVolumes"] = [];
    sceneRef.current = {
      renderWindow, imageData, volume, mapper, ctf, pwf,
      contextImageData, contextVolume, contextMapper, contextCtf, contextPwf,
      quality,
      physicalSampleDistance: 5000, lighting,
      physicalOpacityDistance: VOLUME_OPACITY_REFERENCE_M,
      contextPhysicalSampleDistance: 5000,
      brickVolumes,
      brickOrder: buildVolumeOrder([]),
      appliedBrickOrder: "",
    };

    const playbackResolution = attachVtkPlaybackScreenResolution(renderWindow, {
      shouldMeasure: () => !quality.isPreview(),
      getMappers: () => {
        const current = sceneRef.current;
        return current?.renderWindow === renderWindow
          ? [current.mapper, current.contextMapper, ...current.brickVolumes.map(brick => brick.mapper)] : [];
      },
      onChange: snapshot => {
        setPlaybackScreenDistance(snapshot.distance);
        container.dataset.playbackScreenDistance = String(snapshot.distance);
        container.dataset.playbackScreenTiming = snapshot.timing;
      },
    });
    const detachNativeBeforeGesture = () => {
      const current = sceneRef.current;
      if (current?.renderWindow !== renderWindow
          || !fullNativePreviewRef.current
          || nativePreviewDetachedRef.current) return;
      renderer.removeVolume(current.volume);
      nativePreviewDetachedRef.current = true;
      current.contextVolume.setVisibility(true);
      quality.begin("interaction");
    };
    const finishNativeGesture = () => {
      const current = sceneRef.current;
      if (current?.renderWindow === renderWindow && nativePreviewDetachedRef.current) {
        renderer.addVolume(current.volume);
        nativePreviewDetachedRef.current = false;
        current.volume.setVisibility(true);
        current.contextVolume.setVisibility(false);
        current.renderWindow.getRenderWindow().render();
      }
      quality.end("interaction");
    };
    container.addEventListener("pointerdown", detachNativeBeforeGesture, true);
    window.addEventListener("pointerup", finishNativeGesture, true);
    window.addEventListener("pointercancel", finishNativeGesture, true);

    // captureNextImage triggers a fresh render, sidestepping preserveDrawingBuffer: false.
    registerVtkAppearanceCapture(container, renderWindow, () => ({
      ...appearanceEvidenceRef.current,
      parameters: {
        ...appearanceEvidenceRef.current.parameters,
        interactionQuality: sceneRef.current?.quality.isPreview() ? "preview" : "settled",
        playbackScreenResolution: { ...lookupVtkPlaybackScreenResolution(renderWindow)?.describe() },
        volumeOrdering: sceneRef.current?.brickOrder.unresolved ? "overlapping-layers" : "spatial-partition",
        ...scalarEvidenceRef.current(),
      },
    }));

    const detachContextLoss = attachContextLossRef.current(renderWindow);
    const cameraSampling = renderer.getActiveCamera().onModified(updateRayDistances);

    // Handle container resizes.
    const detachInteractionQuality = attachVtkInteractionQuality(renderWindow, quality, {
      onStartInteraction: () => {
        cameraTouchedRef.current = true;
        setCameraLodSignal((current) => ({
          ...current,
          interacting: true,
          viewportHeightPx: container.getBoundingClientRect().height || current.viewportHeightPx,
        }));
      },
      onEndInteraction: () => {
        setCameraLodSignal((current) => ({ ...current, interacting: false }));
      },
    });
    const detachCameraStability = attachVtkCameraStability(renderWindow, renderer, () => {
      setCameraAdjustedVolumeRayDistances(
        volume.getProperty(),
        renderer.getActiveCamera(),
        quality,
          sceneRef.current?.renderWindow === renderWindow
            ? sceneRef.current.physicalOpacityDistance
            : VOLUME_OPACITY_REFERENCE_M,
        sceneRef.current?.renderWindow === renderWindow
          ? sceneRef.current.physicalSampleDistance
          : 5000,
        sceneScaleRef.current,
        mapper,
      );
      const scene = sceneRef.current;
      if (scene?.renderWindow === renderWindow) {
        setCameraAdjustedVolumeRayDistances(
          scene.contextVolume.getProperty(),
          renderer.getActiveCamera(),
          quality,
          VOLUME_OPACITY_REFERENCE_M,
          scene.contextPhysicalSampleDistance,
          sceneScaleRef.current,
          scene.contextMapper,
        );
        for (const brick of scene.brickVolumes) {
          setCameraAdjustedVolumeRayDistances(
            brick.volume.getProperty(),
            renderer.getActiveCamera(),
            quality,
            scene.physicalOpacityDistance,
            brick.physicalSampleDistance,
            sceneScaleRef.current,
            brick.mapper,
          );
        }
      }
      renderWindow.getRenderWindow().render();
    }, { renderOnEnd: false });
    const detachResize = attachVtkResize(container, renderWindow, {
      policy: VOLUME_RESOLUTION_POLICY,
      onPreviewChange: (preview) => {
        if (preview) quality.begin("resize");
        else { quality.end("resize"); updateRayDistances(); }
      },
    });

    return () => {
      disposeVtkScene({
        clearScene: () => {
          if (sceneRef.current?.renderWindow === renderWindow) sceneRef.current = null;
          nativePreviewDetachedRef.current = false;
        },
        cleanupSteps: [
          () => playbackResolution.dispose(),
          detachResize,
          () => cameraSampling.unsubscribe(),
          () => container.removeEventListener("pointerdown", detachNativeBeforeGesture, true),
          () => window.removeEventListener("pointerup", finishNativeGesture, true),
          () => window.removeEventListener("pointercancel", finishNativeGesture, true),
          clearBackendEvidence,
          detachInteractionQuality,
          () => lighting.dispose(),
          detachCameraStability,
          detachContextLoss,
          () => unregisterVtkCapture(container),
          () => renderer.removeVolume(volume),
          () => renderer.removeVolume(contextVolume),
          () => {
            const images = new Set(brickVolumes.map((brick) => brick.imageData));
            for (const brick of brickVolumes) {
              renderer.removeVolume(brick.volume);
              brick.unregisterQuality();
              brick.volume.delete();
              brick.mapper.delete();
              brick.clipPlanes.forEach((plane) => plane.delete());
            }
            images.forEach((image) => image.delete());
          },
        ],
        resources: [
          volume, mapper, ctf, pwf, imageData,
          contextVolume, contextMapper, contextCtf, contextPwf, contextImageData,
        ],
        renderWindow,
      });
    };
  }, [
    ctxLoss.available,
    ctxLoss.generation,
    textureCapacity.probe,
    volumeAppearanceQualification?.previewDistanceMultiplier,
    updateRayDistances,
    vtkBackend.backend,
    vtkBackend.confirmActiveApi,
  ]);
  useVtkPlaybackScreenResolution(useCallback(() => sceneRef.current?.renderWindow ?? null, []), vtkLifecycleGeneration);

  useEffect(() => {
    const scene = sceneRef.current;
    if (panePresentationPending) return;
    if (!scene) return;
    scene.renderWindow.getRenderer().setBackground(...sceneAppearance.resolved.background);
    scene.lighting.apply(
      sceneAppearance.resolved.lightingOptions,
      sceneAppearance.resolved.isLight,
    );
    if (scene.volume.getVisibility()) {
      applyFieldVolumeLighting(
        scene.volume.getProperty(),
        sceneAppearance.resolved.lightingOptions,
        sparseOverviewRef.current ? "signed-perturbation" : opacityClassRef.current,
      );
    }
    for (const brick of scene.brickVolumes) {
      applyFieldVolumeLighting(
        brick.volume.getProperty(),
        sceneAppearance.resolved.lightingOptions,
        opacityClassRef.current,
      );
    }
    scene.renderWindow.getRenderWindow().render();
  }, [
    panePresentationPending,sceneAppearance.resolved, vtkLifecycleGeneration]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (panePresentationPending) return;
    if (!scene) return;
    const disposeBricks = (bricks: SceneRefs["brickVolumes"]) => {
      const renderer = scene.renderWindow.getRenderer();
      const images = new Set(bricks.map((brick) => brick.imageData));
      for (const brick of bricks) {
        renderer.removeVolume(brick.volume);
        brick.unregisterQuality();
        brick.volume.delete();
        brick.mapper.delete();
        brick.clipPlanes.forEach((plane) => plane.delete());
      }
      images.forEach((image) => image.delete());
    };
    if (!brickedVolumeEligible) {
      const previous = scene.brickVolumes.splice(0);
      scene.brickOrder = buildVolumeOrder([]);
      scene.appliedBrickOrder = "";
      disposeBricks(previous);
      return;
    }
    if (!activeBrickedCurrent || !nestedGenerationCurrent || !renderGrid || !backgroundAdmitted || background.loading) return;
    const next: SceneRefs["brickVolumes"] = [];
    const images = new Map<Float32Array, ReturnType<typeof vtkImageData.newInstance>>();
    try {
      for (const { prepared } of composedPreparedBricks) {
        let imageData = images.get(prepared.values);
        if (!imageData) {
          imageData = vtkImageData.newInstance();
          images.set(prepared.values, imageData);
          setPreparedImageScalars(imageData, prepared);
          setPreparedImageDisplayGeometry(imageData, prepared, sceneScaleRef.current);
        }
        const mapper = vtkVolumeMapper.newInstance();
        configureCoherentVolumeSampling(mapper, true, volumeOpenUpperFaces(prepared));
        const ownedBounds = volumeOwnedBounds(prepared);
        const clipPlanes = [0, 1, 2, 3, 4, 5].map((face) => {
          const axis = Math.floor(face / 2);
          const normal: [number, number, number] = [0, 0, 0];
          normal[axis] = face % 2 ? -1 : 1;
          const origin: [number, number, number] = [0, 0, 0];
          origin[axis] = ownedBounds[face] * (axis === 2 ? sceneScaleRef.current.z : sceneScaleRef.current.xy);
          const plane = vtkPlane.newInstance({ normal, origin });
          mapper.addClippingPlane(plane);
          return plane;
        });
        const minimumSpacing = Math.min(...prepared.spacing);
        const sampleDistance = Number.isFinite(minimumSpacing) && minimumSpacing > 0
          ? minimumSpacing * (volumeAppearanceQualification?.sampleDistanceFactor ?? 0.5)
          : 5000;
        mapper.setSampleDistance(sampleDistance);
        mapper.setMaximumSamplesPerRay(3000);
        mapper.setAutoAdjustSampleDistances(false);
        mapper.setInputData(imageData);
        setVolumeBackground(mapper, backgroundRef.current, prepared);
        const volume = vtkVolume.newInstance();
        volume.setMapper(mapper);
        const property = volume.getProperty();
        property.setRGBTransferFunction(0, scene.ctf);
        property.setScalarOpacity(0, scene.pwf);
        property.setInterpolationTypeToLinear();
        property.setComputeNormalFromOpacity(volumeUsesOpacityNormals(opacityClassRef.current));
        applyVolumeInteractionLighting(
          property,
          scene.quality.isPreview(),
          sceneAppearanceRef.current.lightingOptions,
          opacityClassRef.current,
        );
        const unregisterQuality = scene.quality.registerMapper(mapper, sampleDistance);
        setCameraAdjustedVolumeRayDistances(
          property,
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          scene.physicalOpacityDistance,
          sampleDistance,
          sceneScaleRef.current,
          mapper,
        );
        volume.setVisibility(false);
        scene.renderWindow.getRenderer().addVolume(volume);
        next.push({
          imageData,
          mapper,
          volume,
          unregisterQuality,
          physicalSampleDistance: sampleDistance,
          ownedBounds,
          clipPlanes,
        });
      }
    } catch (error) {
      disposeBricks(next);
      images.forEach((image) => { if (!image.isDeleted()) image.delete(); });
      throw error;
    }
    for (const brick of next) brick.volume.setVisibility(true);
    scene.volume.setVisibility(false);
    scene.contextVolume.setVisibility(false);
    const previous = scene.brickVolumes.splice(0, scene.brickVolumes.length, ...next);
    scene.brickOrder = buildVolumeOrder(next.map((brick) => brick.ownedBounds));
    scene.appliedBrickOrder = "";
    disposeBricks(previous);
    if (!cameraInitializedRef.current && composedPreparedBricks.length > 0) {
      const brickFocusBounds = unionPreparedVolumeBounds(
        composedPreparedBricks.map((brick) => brick.prepared),
        sceneScaleRef.current,
      );
      resetVtkCameraToStormView(
        scene.renderWindow.getRenderer(),
        brickFocusBounds,
        initialFramingRef.current.bounds,
      );
      focusVtkCameraOnPoint(
        scene.renderWindow.getRenderer(),
        scaledHandoffFocus(initialFramingRef.current.focus, sceneScaleRef.current),
      );
      cameraInitializedRef.current = true;
    }
    updateRayDistances();
    scene.renderWindow.getRenderWindow().render();
    scalarEvidenceRef.current = scalarEvidence;
    markFrameRendered(contextGeneration);
    setRenderedStepIndex(currentStep?.index ?? null);
    setRenderedPublicationGeneration(currentStep?.publicationGeneration ?? null);
  }, [
    panePresentationPending,
    composedPreparedBricks,
    scalarEvidence,
    backgroundAdmitted,
    background.loading,
    activeBrickedCurrent,
    nestedGenerationCurrent,
    brickedMetadataOnly,
    brickedVolumeEligible,
    contextGeneration,
    currentStep?.index,
    currentStep?.publicationGeneration,
    markFrameRendered,
    renderGrid,
    vtkLifecycleGeneration,
    volumeAppearanceQualification?.sampleDistanceFactor,
    updateRayDistances,
  ]);

  // ---------- Immutable voxel / geometry update ----------
  // Keep this effect independent from material controls. Replacing the scalar
  // array dirties the volume mapper and can re-upload millions of voxels, so a
  // colormap or opacity edit must never enter this path.
  useEffect(() => {
    const scene = sceneRef.current;
    if (panePresentationPending) return;
    if (brickedVolumeEligible
        || !scene || !fieldResult.field || !renderGrid || !preparation.prepared) return;
    if (monolithicGpuUpload.error || !monolithicGpuUpload.admitted || !backgroundAdmitted) return;
    const prepared = preparation.prepared;

    // Push voxels into the existing imageData (re-creating forces a fresh mapper allocation).
    setPreparedImageScalars(scene.imageData, prepared);
    setPreparedImageDisplayGeometry(scene.imageData, prepared, sceneScaleRef.current);

    // Half-voxel sampling is the product default. Qualification may vary only
    // this factor to distinguish ray aliasing from resolved source structure.
    const minSpacing = Math.min(...prepared.spacing);
    if (Number.isFinite(minSpacing) && minSpacing > 0) {
      scene.physicalSampleDistance = minSpacing
        * (volumeAppearanceQualification?.sampleDistanceFactor ?? 0.5);
      setCameraAdjustedVolumeRayDistances(
        scene.volume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        scene.physicalOpacityDistance,
        scene.physicalSampleDistance,
        sceneScaleRef.current,
        scene.mapper,
      );
    }

    // Show volume now that bounds are real; hidden before to skip expensive renders on empty data.
    scene.volume.setVisibility(true);

    const detailIdentity = hybridDetailActive
      ? `${featureIdentity}|${fieldResult.loadedStepIndex ?? "none"}|${fieldResult.loadedPublicationGeneration ?? "none"}`
      : null;
    const shouldFramePromotedDetail = volumeAppearanceQualification?.frameNativeDetail !== false
      && detailIdentity != null
      && framedDetailIdentityRef.current !== detailIdentity
      && !cameraTouchedRef.current;
    if (!cameraInitializedRef.current || shouldFramePromotedDetail) {
      const promotedFocus = scaledFeatureBounds(
        prepared.structuralSupportBounds ?? prepared.absoluteFocusBounds,
        sceneScaleRef.current,
      );
      resetVtkCameraToStormView(
        scene.renderWindow.getRenderer(),
        volumeFocusRef.current.structural ? promotedFocus : null,
        logicalFrameBounds,
      );
      if (shouldFramePromotedDetail) {
        focusVtkCameraOnBounds(scene.renderWindow.getRenderer(), promotedFocus);
      } else if (!cameraInitializedRef.current
          && !hybridDetailActive
          && volumeFocusRef.current.structural) {
        // Full-native fields no longer need an ROI to preserve fidelity. Frame
        // the actual structural support on first open while retaining the
        // complete domain only for clipping, matching the detail view's useful
        // scale without cropping any voxels.
        focusVtkCameraOnBounds(scene.renderWindow.getRenderer(), promotedFocus);
      }
      focusVtkCameraOnPoint(
        scene.renderWindow.getRenderer(),
        scaledHandoffFocus(handoffFocusCenter, sceneScaleRef.current),
      );
      cameraInitializedRef.current = true;
      if (detailIdentity) framedDetailIdentityRef.current = detailIdentity;
    }

    setCameraAdjustedVolumeRayDistances(
      scene.volume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      scene.physicalOpacityDistance,
      scene.physicalSampleDistance,
      sceneScaleRef.current,
      scene.mapper,
    );

    scene.mapper.modified();
    scene.renderWindow.getRenderWindow().render();
    scalarEvidenceRef.current = scalarEvidence;
    markFrameRendered(contextGeneration);
    setRenderedStepIndex(fieldResult.loadedStepIndex);
    setRenderedPublicationGeneration(fieldResult.loadedPublicationGeneration);
  }, [
    panePresentationPending,
    fieldResult.field,
    fieldResult.loadedStepIndex,
    fieldResult.loadedPublicationGeneration,
    sourceDefinitionIdentity,
    renderGrid,
    preparation.prepared,
    scalarEvidence,
    contextGeneration,
    markFrameRendered,
    vtkLifecycleGeneration,
    handoffFocusCenter,
    hybridDetailActive,
    featureIdentity,
    volumeAppearanceQualification?.frameNativeDetail,
    volumeAppearanceQualification?.sampleDistanceFactor,
    brickedVolumeEligible,
    logicalFrameBounds,
    monolithicGpuUpload.admitted,
    monolithicGpuUpload.error,
    backgroundAdmitted,
  ]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (panePresentationPending) return;
    const prepared = overviewPreparation.prepared;
    if (!scene) return;
    if (!contextVolumeRequired || !overviewFieldResult.field || !prepared) {
      scene.contextVolume.setVisibility(false);
      scene.renderWindow.getRenderWindow().render();
      return;
    }
    if (monolithicGpuUpload.error || !monolithicGpuUpload.admitted) return;
    setPreparedImageScalars(scene.contextImageData, prepared);
    setPreparedImageDisplayGeometry(scene.contextImageData, prepared, sceneScaleRef.current);
    const minSpacing = Math.min(...prepared.spacing);
    if (Number.isFinite(minSpacing) && minSpacing > 0) {
      scene.contextPhysicalSampleDistance = minSpacing;
      setCameraAdjustedVolumeRayDistances(
        scene.contextVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        VOLUME_OPACITY_REFERENCE_M,
        scene.contextPhysicalSampleDistance,
        sceneScaleRef.current,
        scene.contextMapper,
      );
    }
    scene.contextMapper.modified();
    scene.contextVolume.setVisibility(!fullNativeInteractionPreview || scene.quality.isPreview());
    scene.renderWindow.getRenderWindow().render();
  }, [
    panePresentationPending,
    fullNativeInteractionPreview,
    contextVolumeRequired,
    hybridDetailActive,
    overviewFieldResult.field,
    overviewPreparation.prepared,
    vtkLifecycleGeneration,
    monolithicGpuUpload.admitted,
    monolithicGpuUpload.error,
  ]);

  // ---------- Transfer-function / material update ----------
  // Transfer functions use robust ranges covering the visible opacity tail; dBZ uses its fixed
  // meteorological range. These property-only edits reuse the existing scalar
  // texture and deliberately do not call setPreparedImageScalars/mapper.modified.
  useEffect(() => {
    const scene = sceneRef.current;
    if (panePresentationPending) return;
    const prepared = preparation.prepared;
    const brickStats = activeBrickedCurrent
        && (!brickedMetadataOnly || scalarTransform === "none")
      ? activeBrickedStats
      : null;
    if (!scene || (!prepared && !brickStats) || !backgroundAdmitted) return;
    const stats = brickStats ?? prepared?.stats ?? {
      p2: Number.NaN,
      p98: Number.NaN,
      min: Number.NaN,
      max: Number.NaN,
    };
    const transfer = resolveVolumeTransferPolicy(fieldName, colormap, stats, scalarTransform, backgroundFieldMode === "structure");
    const focusValues = backgroundAffectsOpacity(background.profile,
      volumeOpacityPoints(transfer, opacityScale).map(point => [point.scalar, point.opacity] as const));
    if (prepared) setVolumeBackground(scene.mapper, background.profile, prepared, focusValues);
    if (overviewPreparation.prepared) setVolumeBackground(scene.contextMapper, background.profile, overviewPreparation.prepared, focusValues);
    for (const [index, brick] of scene.brickVolumes.entries()) {
      const image = composedPreparedBricks[index]?.prepared;
      if (image) setVolumeBackground(brick.mapper, background.profile, image, focusValues);
    }
    const referenceM = volumeAppearanceQualification?.opacityReferenceM
      ?? volumeOpacityReferenceM(transfer.opacityClass);
    applyColormap(scene.ctf, colormap, transfer.min, transfer.max);
    applyOpacity(
      scene.pwf,
      transfer,
      effectiveMainOpacityScale,
    );
    const mainReferenceM = sparseReducedOverviewActive
      ? VOLUME_OPACITY_REFERENCE_M
      : referenceM;
    scene.physicalOpacityDistance = mainReferenceM;
    setColorRange({ min: transfer.min, max: transfer.max });
    setOpacityMaximum(transfer.opacityMax ?? null);
    setTransferKind(transfer.kind);
    setOpacityClass(transfer.opacityClass);
    opacityClassRef.current = transfer.opacityClass;
    setOpacityReferenceM(mainReferenceM);
    setCameraAdjustedVolumeRayDistances(
      scene.volume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      scene.physicalOpacityDistance,
      scene.physicalSampleDistance,
      sceneScaleRef.current,
      scene.mapper,
    );
    applyFieldVolumeLighting(
      scene.volume.getProperty(),
      sceneAppearanceRef.current.lightingOptions,
      sparseReducedOverviewActive || volumeAppearanceQualification?.lighting === "flat"
        ? "signed-perturbation"
        : transfer.opacityClass,
    );
    for (const brick of scene.brickVolumes) {
      applyFieldVolumeLighting(
        brick.volume.getProperty(),
        sceneAppearanceRef.current.lightingOptions,
        volumeAppearanceQualification?.lighting === "flat"
          ? "signed-perturbation"
          : transfer.opacityClass,
      );
      setCameraAdjustedVolumeRayDistances(
        brick.volume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        referenceM,
        brick.physicalSampleDistance,
        sceneScaleRef.current,
        brick.mapper,
      );
    }
    scene.renderWindow.getRenderWindow().render();
    pendingMaterialIdentityRef.current = expectedMaterialIdentity;
  }, [
    panePresentationPending,
    preparation.prepared,
    activeBrickedStats,
    activeBrickedCurrent,
    brickedMetadataOnly,
    fieldName,
    colormap,
    opacityScale,
    scalarTransform,
    sparseReducedOverviewActive,
    effectiveMainOpacityScale,
    vtkLifecycleGeneration,
    volumeAppearanceQualification,
    expectedMaterialIdentity,
    background.profile, backgroundFieldMode,
  ]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (panePresentationPending) return;
    const contextPrepared = overviewPreparation.prepared;
    const detailStats = activeBrickedStats ?? preparation.prepared?.stats;
    if (!scene || !contextVolumeRequired || !contextPrepared || !detailStats) return;
    const stats = detailStats ?? {
      p2: Number.NaN,
      p98: Number.NaN,
      min: Number.NaN,
      max: Number.NaN,
    };
    const transfer = resolveVolumeTransferPolicy(fieldName, colormap, stats, scalarTransform, backgroundFieldMode === "structure");
    applyColormap(scene.contextCtf, colormap, transfer.min, transfer.max);
    applyOpacity(
      scene.contextPwf,
      transfer,
      hybridDetailActive
        ? opacityScale * contextOpacityFactor
        : effectiveMainOpacityScale,
    );
    const property = scene.contextVolume.getProperty();
    applyFieldVolumeLighting(
      property,
      sceneAppearanceRef.current.lightingOptions,
      hybridDetailActive ? "signed-perturbation" : transfer.opacityClass,
    );
    setCameraAdjustedVolumeRayDistances(
      property,
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      hybridDetailActive ? VOLUME_OPACITY_REFERENCE_M : scene.physicalOpacityDistance,
      scene.contextPhysicalSampleDistance,
      sceneScaleRef.current,
      scene.contextMapper,
    );
    scene.renderWindow.getRenderWindow().render();
    pendingContextMaterialIdentityRef.current = expectedMaterialIdentity;
  }, [
    panePresentationPending,
    colormap,
    fieldName,
    contextVolumeRequired,
    hybridDetailActive,
    contextOpacityFactor,
    effectiveMainOpacityScale,
    opacityScale,
    scalarTransform,
    volumeAppearanceQualification,
    activeBrickedStats,
    overviewPreparation.prepared,
    preparation.prepared,
    vtkLifecycleGeneration,
    expectedMaterialIdentity,
  ]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || panePresentationPending || expectedMaterialIdentity == null
        || pendingMaterialIdentityRef.current !== expectedMaterialIdentity
        || contextVolumeRequired && pendingContextMaterialIdentityRef.current !== expectedMaterialIdentity) return;
    floor.commitData();
    releasePresentationRef.current?.();
    releasePresentationRef.current = null;
    scene.renderWindow.getRenderWindow().render();
    setCommittedMaterialIdentity(expectedMaterialIdentity);
    setCommittedContextMaterialIdentity(contextVolumeRequired ? expectedMaterialIdentity : null);
    committedGridRef.current = renderGrid;
    markFrameRendered(contextGeneration);
    setRenderedStepIndex(loadedStep?.index ?? null);
    setRenderedPublicationGeneration(loadedStep?.publicationGeneration ?? null);
  }, [panePresentationPending, expectedMaterialIdentity, contextVolumeRequired, floor.commitData, renderGrid, loadedStep,
    contextGeneration, markFrameRendered, vtkLifecycleGeneration]);

  // ---------- Vertical exaggeration ----------
  // Scale ImageData geometry so vtk's compiled ray-loop bound sees the same
  // display extent as the camera; actor scale is invisible to that calculation.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const currentScale = { xy: sceneScale.xy, z: sceneScale.z };
    scene.volume.setScale(1, 1, 1);
    const scaleFraming = scaleFramingRef.current;
    const prepared = preparedRef.current;
    if (prepared) {
      setPreparedImageDisplayGeometry(scene.imageData, prepared, currentScale);
      scene.mapper.modified();
    }
    const contextPrepared = overviewPreparation.prepared;
    if (contextPrepared && contextVolumeRequired) {
      setPreparedImageDisplayGeometry(scene.contextImageData, contextPrepared, currentScale);
      scene.contextMapper.modified();
    }
    scene.brickVolumes.forEach((brick, index) => {
      const preparedBrick = composedPreparedBricks[index]?.prepared;
      if (!preparedBrick) return;
      setPreparedImageDisplayGeometry(brick.imageData, preparedBrick, currentScale);
      brick.mapper.modified();
      brick.clipPlanes.forEach((plane, face) => {
        const axis = Math.floor(face / 2);
        const origin: [number, number, number] = [0, 0, 0];
        origin[axis] = brick.ownedBounds[face] * (axis === 2 ? currentScale.z : currentScale.xy);
        plane.setOrigin(...origin);
      });
      const minimumSpacing = Math.min(...preparedBrick.spacing);
      if (Number.isFinite(minimumSpacing) && minimumSpacing > 0) {
        setCameraAdjustedVolumeRayDistances(
          brick.volume.getProperty(),
          scene.renderWindow.getRenderer().getActiveCamera(),
          scene.quality,
          scene.physicalOpacityDistance,
          brick.physicalSampleDistance,
          currentScale,
          brick.mapper,
        );
      }
    });
    setCameraAdjustedVolumeRayDistances(
      scene.volume.getProperty(),
      scene.renderWindow.getRenderer().getActiveCamera(),
      scene.quality,
      scene.physicalOpacityDistance,
      scene.physicalSampleDistance,
      currentScale,
      scene.mapper,
    );
    if (contextPrepared && contextVolumeRequired) {
      setCameraAdjustedVolumeRayDistances(
        scene.contextVolume.getProperty(),
        scene.renderWindow.getRenderer().getActiveCamera(),
        scene.quality,
        VOLUME_OPACITY_REFERENCE_M,
        scene.contextPhysicalSampleDistance,
        currentScale,
        scene.contextMapper,
      );
    }
    if (cameraInitializedRef.current && scaleFraming.consume()) {
      // Reframe in place — keep the user's orbit; re-applying the oblique storm
      // view here would yank the camera back to the preset on every z-scale tick.
      const focus = volumeFocusRef.current.structural
        ? scaledFeatureBounds(
            prepared?.structuralSupportBounds ?? prepared?.absoluteFocusBounds,
            currentScale,
          )
        : null;
      if (focus) focusVtkCameraOnBounds(scene.renderWindow.getRenderer(), focus);
      else refitVtkCameraToBounds(scene.renderWindow.getRenderer());
    }
    scene.renderWindow.getRenderWindow().render();
  }, [
    sceneScale.xy,
    sceneScale.z,
    contextVolumeRequired,
    overviewPreparation.prepared,
    composedPreparedBricks,
    vtkLifecycleGeneration,
  ]);

  // ---------- Reset camera ----------
  useEffect(() => {
    if (appliedResetRequestIdentityRef.current === resetRequestIdentity) return;
    appliedResetRequestIdentityRef.current = resetRequestIdentity;
    const scene = sceneRef.current;
    if (!scene) return;
    const renderer = scene.renderWindow.getRenderer();
    const focus = volumeFocusRef.current.structural
      ? scaledFeatureBounds(
          volumeFocusRef.current.structuralSupportBounds
            ?? volumeFocusRef.current.absoluteFocusBounds,
          sceneScaleRef.current,
        )
      : null;
    resetVtkCameraToStormView(
      renderer,
      focus,
    );
    focusVtkCameraOnPoint(
      renderer,
      scaledHandoffFocus(handoffFocusCenter, sceneScaleRef.current),
    );
    setCameraAdjustedVolumeRayDistances(
      scene.volume.getProperty(),
      renderer.getActiveCamera(),
      scene.quality,
      scene.physicalOpacityDistance,
      scene.physicalSampleDistance,
      sceneScaleRef.current,
      scene.mapper,
    );
    scene.renderWindow.getRenderWindow().render();
  }, [resetRequestIdentity, handoffFocusCenter]);

  const hasField = renderGrid != null && (brickedVolumeEligible
    ? activeBrickedCurrent && activeBrickedStats != null
    : fieldResult.field != null);
  const displayedStats = brickedVolumeEligible
    ? activeBrickedCurrent ? activeBrickedStats : null
    : preparation.prepared?.stats;
  const volumeLegend = useCommittedLegend(
    runDir,
    materialCommitted,
    hasField && colorRange && fieldName ? {
      colormap: colormap === "dBZ" ? "dBZ volume" : colormap,
      min: colorRange.min, max: colorRange.max,
      label: scalarTransform === "horizontal-level-anomaly" ? `${fieldName} level anomaly` : fieldName,
      unit: unitFor(fieldName, loadedStep?.fieldUnits ?? currentStep?.fieldUnits),
    } : null,
  );
  const floorLegend = useCommittedLegend(
    runDir,
    materialCommitted && !floor.loading && !floor.error
      && floor.committedPresentationIdentity === floor.presentationIdentity,
    floorSelection.enabled && floorSelection.opacity > 0 && floor.visible && floorSelection.fieldName && floor.policy ? {
      colormap: floor.policy.colormap, min: floor.policy.min, max: floor.policy.max,
      label: `${floorSelection.fieldName} floor`,
      unit: unitFor(floorSelection.fieldName, loadedStep?.fieldUnits ?? currentStep?.fieldUnits),
    } : null,
  );
  const uniformField = hasField && materialCommitted && displayedStats != null
    && Number.isFinite(displayedStats.min) && Number.isFinite(displayedStats.max)
    && displayedStats.min === displayedStats.max;
  /* Single-level products (cape, cin, vil, composite_reflectivity, ...) have no volume to
   * ray-march, so offering them here only produces a shape error after the load. */
  const fieldOptions = useMemo(
    () => renderableFields.map((f) => ({ value: f, label: f })),
    [renderableFields],
  );
  /* The field the NWS ramp can describe, if this run exported one. Picking that
   * ramp switches to it — see the Colormap group below. */
  const reflectivityFieldName = useMemo(
    () => preferredVolumetricReflectivityField(fieldsForStep),
    [fieldsForStep],
  );
  const beginVolumeControlPreview = () => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (fullNativeInteractionPreview && !nativePreviewDetachedRef.current) {
      scene.renderWindow.getRenderer().removeVolume(scene.volume);
      nativePreviewDetachedRef.current = true;
    }
    scene.quality.begin("interaction");
  };
  const endVolumeControlPreview = () => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (nativePreviewDetachedRef.current) {
      scene.renderWindow.getRenderer().addVolume(scene.volume);
      nativePreviewDetachedRef.current = false;
      scene.volume.setVisibility(true);
      scene.contextVolume.setVisibility(false);
      scene.renderWindow.getRenderWindow().render();
    }
    scene.quality.end("interaction");
  };
  const volumeControlPreviewProps = {
    onPointerDown: beginVolumeControlPreview,
    onPointerUp: endVolumeControlPreview,
    onPointerCancel: endVolumeControlPreview,
    onBlur: endVolumeControlPreview,
  };

  return (
    <div className="volume-render">
      <Grouped3DControls
        ariaLabel="Volume controls"
        scope={controlScope}
        notice={qualityPreview
          ? fullNativeInteractionPreview
            ? "Reduced volume while moving · native detail restores at idle"
            : "Preview while moving · settles to full quality"
          : playbackScreenDistance > 1 ? "Playback screen preview · full resolution returns when paused" : null}
        groups={[
          {
            id: "volume",
            label: "Volume",
            status: fieldName ?? "No field",
            controls: <>
        <OutputHierarchyControls source={outputHierarchySource} />
        <PlotCompressionControls step={loadedStep} fields={[fieldName]} />
        <ControlGroup label="Field">
          <select
            aria-label="Field"
            className="volume-render__select"
            value={fieldName ?? ""}
            onChange={(e) => {
              const next = e.target.value || null;
              setFieldName(next);
              onFieldChange?.(next);
              clearPreset();
            }}
          >
            <option value="">— pick a field —</option>
            {fieldOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </ControlGroup>
          <ControlGroup label="Volume view">
            <PillGroup
              options={[
                { value: "structure" as const, label: "Structure", hint: usesVolumeBackground(fieldName)
                  ? "Hide the widespread horizontal background in opacity. Colors retain the original field values."
                  : supportsVolumeStructure(fieldName) ? "Show signed departures from each horizontal level's mean."
                  : "Emphasize storm structure without magnifying weak background values." },
                { value: "raw" as const, label: "Raw field", hint: "Show the full field with its original definition and no background selection." },
              ]}
              value={backgroundFieldMode}
              onChange={setBackgroundFieldMode}
            />
          </ControlGroup>
        <ControlGroup label="Colormap">
          <PillGroup
            /* NWS dBZ ramp is anchored to physical dBZ values; offered only for reflectivity fields. */
            options={COLORMAPS.map((c) => ({
              value: c.value,
              label: c.label,
              unavailable: c.value !== "dBZ" || reflectivityFieldName
                ? undefined
                : "This run exported no reflectivity field, and the NWS ramp is anchored to dBZ values so it cannot describe anything else.",
            }))}
            value={colormap}
            onChange={(v) => {
              /*
               * Asking for the NWS ramp on a 3D scene is unambiguous: it means "show me
               * reflectivity", since the ramp's stops are anchored to physical dBZ and
               * describe no other quantity. Disabling it instead left the control dead in
               * exactly the situation a user reaches for it.
               */
              if (v === "dBZ" && !isReflectivityField(fieldName) && reflectivityFieldName) {
                setFieldName(reflectivityFieldName);
              }
              setColormap(v);
              clearPreset();
            }}
          />
        </ControlGroup>
        <ControlGroup label="Opacity">
          <input
            type="range"
            min={0.1}
            max={2.5}
            step={0.05}
            value={opacityScale}
            onChange={(e) => setOpacityScale(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
            title="Strengthen feature-bearing values without making the low-value background opaque"
            {...volumeControlPreviewProps}
          />
          <span className="volume-render__slider-value">{opacityScale.toFixed(2)}×</span>
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
        <SceneAppearanceControls state={sceneAppearance} backend={vtkBackend} lightingEnabled={volumeUsesStructuralLighting(sparseReducedOverviewActive ? "signed-perturbation" : effectiveOpacityClass)} />
        <SceneScaleControl value={sceneScale} onChange={setSceneScale} />
        {(!pane.cameraLocked || pane.cameraAuthority) && (
          <>
            <CameraElevationControls selection={cameraElevation} />
            <button
              type="button"
              className="btn btn--secondary volume-render__reset"
              onClick={() => {
                cameraTouchedRef.current = false;
                framedDetailIdentityRef.current = null;
                setResetTick((t) => t + 1);
              }}
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
        data-output-level={outputHierarchySource.activeLevel?.levelId ?? "root"}
        data-output-hierarchy-valid={outputHierarchySource.valid}
        data-volume-representation={brickedVolumeEligible
          ? "progressive-bricks"
          : hybridDetailActive ? "hybrid" : "full-domain"}
        data-volume-hierarchy-source={brickedVolumeEligible
          ? publishedBrickedVolumeAvailable ? "published" : "transient"
          : "none"}
        data-volume-reduced-context-only={sparseReducedOverviewActive}
        data-volume-context-ready={contextVolumeRequired && overviewPreparation.prepared != null}
        data-scalar-transform={brickedVolumeEligible
          ? "none"
          : preparation.prepared?.scalarTransform ?? "loading"}
        data-source-plane-axis={preset?.volumeFocus?.axis ?? ""}
        data-source-plane-index={preset?.volumeFocus?.sourceIndex ?? ""}
        data-volume-material-committed={materialCommitted}
        data-volume-quality-preview={qualityPreview}
        data-volume-zoom-native={paneVoxelCap > baseVoxelCap}
        data-volume-brick-level={activeBrickedPlan?.level ?? ""}
        data-volume-brick-count={composedPreparedBricks.length}
        data-volume-nested-layers={nestedBrickedEligible ? nestedOwnership?.length ?? 1 : 1}
        data-volume-parent-brick-count={nestedParentBrickCount}
        data-volume-parent-excluded-bricks={nestedBrickedPreparation.plans[0]?.plan.excludedBrickCount ?? 0}
        data-volume-child-bounds-lower={childOwnership?.level.step.grid?.physicalBoundsM?.lower.join(",") ?? ""}
        data-volume-transparent-bricks={activeBrickedPlan?.transparentBrickCount ?? 0}
        data-volume-boundary-only-bricks={activeBrickedPlan?.boundaryOnlyBrickCount ?? 0}
        data-volume-brick-error={activeBrickedError ?? monolithicGpuUpload.error ?? ""}
        data-logical-gpu-upload-bytes={brickedVolumeEligible
          ? composedScalarBytes
          : monolithicGpuUpload.admitted ? monolithicGpuUploadBytes : 0}
        data-volume-lod-retained-transitions={lodTransitionEvidence.count}
        data-volume-lod-retained-gpu-bytes={lodTransitionEvidence.retainedGpuBytes}
      >
        <div
          ref={containerRef}
          className="volume-render__canvas"
          role="img"
          aria-label="Interactive three-dimensional scalar volume; drag to orbit, use camera controls for keyboard adjustment"
        />
        <Plot3DAxisLabels labels={axisLabels} />
        <Plot3DScaleBadge
          grid={renderGrid}
          missingSupport={Boolean((!preparation.prepared?.focusEmpty && preparation.prepared?.validity) || (!overviewPreparation.prepared?.focusEmpty && overviewPreparation.prepared?.validity)
            || composedPreparedBricks.some(({ prepared }) => prepared.validity))}
          zExag={sceneScale.z}
          fidelityMode={renderFidelityMode}
          samplingProvenance={coarsestVtkSampling([
            composedPreparedBricks[0]?.prepared.samplingProvenance,
            preparation.prepared?.samplingProvenance,
            hybridDetailActive ? overviewPreparation.prepared?.samplingProvenance : null,
          ]) ?? undefined}
          detail={(background.profile ? `storm structure · background suppressed in opacity; colors retain ${fieldName} values · ` : "") + (scalarTransform === "horizontal-level-anomaly"
            ? `horizontal-level anomaly · signed structural volume · opacity per ${formatDistance(opacityReferenceM)}`
            : transferKind === "signed"
            ? `signed diverging volume · structural deadband · opacity per ${formatDistance(opacityReferenceM)}`
            : `${opacityClass.replaceAll("-", " ")} opacity · per ${formatDistance(opacityReferenceM)} · ${volumeUsesStructuralLighting(opacityClass) ? "scalar-gradient" : "flat"} lit`
              + (brickedVolumeEligible
                ? ` · ${publishedBrickedVolumeAvailable ? "published" : "transient"} progressive native bricks`
                : boundedWindowOutcome.ok && boundedWindowOutcome.mode === "native-detail"
                ? ` · hybrid overview + ${stormCenterX != null
                  ? "tracker-centred" : "feature-centred"} stride-1 ROI`
                : sparseReducedOverviewActive
                  ? " · complete reduced context; native ROI withheld until visible support fits"
                  : boundedWindowOutcome.ok
                      && boundedWindowOutcome.mode === "full-domain"
                      && boundedWindowOutcome.plan.stride.every((stride) => stride === 1)
                    ? " · full-domain native stride-1"
                  : ""))
            + (boundedOverviewFallback ? " · full-domain overview fitted to memory budget" : "")
            + (preset?.volumeFocus
              ? ` · focused from source axis ${preset.volumeFocus.axis} index ${preset.volumeFocus.sourceIndex}`
              : "")
            + (outputHierarchySource.activeLevel?.depth
              ? ` · ${outputHierarchySource.selection?.disclosure ?? outputHierarchySource.activeLevel.levelId}`
              : outputHierarchySource.selection?.fallbackReason
                ? ` · ${outputHierarchySource.selection.fallbackReason}`
                : "")}
        />
        <Plot3DOverlayStack>
          {volumeLegend && <Plot3DColorbar {...volumeLegend} />}
          {floorLegend && <Plot3DColorbar {...floorLegend} />}
        </Plot3DOverlayStack>
        <Plot3DCompass orientation={cameraOrientation} />
        {ctxLoss.lost && (
          <div className="volume-render__overlay volume-render__overlay--error">
            3D graphics context lost
            <div className="volume-render__overlay-hint">
              Rebuilding after the GPU dropped WebGL. This clears after the
              selected frame renders successfully.
            </div>
          </div>
        )}
        {boundedLoadError && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Volume loading is unavailable within the bounded-read policy.
            <div className="volume-render__overlay-hint">
              {boundedLoadError} Nimbus will not fall back to reading the full source volume.
            </div>
          </div>
        )}
        {fieldPreparationError && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Field failed: {fieldPreparationError}
            <div className="volume-render__overlay-hint">
              The selected field could not be prepared. Check the reported source, read-path, or memory error.
            </div>
          </div>
        )}
        {!terrainBlocked && !boundedLoadError && !hasField
          && !fieldResult.loading && !preparation.loading && !activeBrickedLoading
          && !fieldPreparationError && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            Pick a 3D scalar field above.
          </div>
        )}
        {declinedLink && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            {describeDeclinedFieldLink(declinedLink, SINGLE_LEVEL_LINK_REASON)}
          </div>
        )}
        {!uniformField && materialCommitted && background.profile
          && (brickedVolumeEligible ? composedPreparedBricks.length === 0 : preparation.prepared?.focusEmpty) && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            No localized structure above the horizontal background in this frame. Raw field shows the complete field.
          </div>
        )}
        {uniformField && !fieldResult.loading && !preparation.loading && !activeBrickedLoading
          && !fieldPreparationError && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            {fieldName}{scalarTransform === "horizontal-level-anomaly" ? " level anomaly" : ""} is uniform in this frame
            {displayedStats && ` (${displayedStats.min.toPrecision(3)} everywhere)`}
            {" "}— nothing to render. Scrub forward, or pick another field.
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- VTK transfer-function helpers ---------- */

function applyColormap(
  ctf: ReturnType<typeof vtkColorTransferFunction.newInstance>,
  preset: ColormapPreset,
  min: number,
  max: number,
) {
  ctf.removeAllPoints();
  if (preset === "dBZ") {
    // Physical dBZ anchors (not normalized), so a given dBZ reads the same colour in every run.
    for (const stop of DBZ_VTK_STOPS_VOLUME) {
      ctf.addRGBPoint(stop.scalar, stop.r, stop.g, stop.b);
    }
    ctf.updateRange();
    return;
  }
  if (preset === "Greys") {
    // Black (low) -> white (high) across data range.
    if (max <= min) {
      ctf.addRGBPoint(min - 1, 0, 0, 0);
      ctf.addRGBPoint(min + 1, 1, 1, 1);
    } else {
      ctf.addRGBPoint(min, 0, 0, 0);
      ctf.addRGBPoint(max, 1, 1, 1);
    }
    ctf.updateRange();
    return;
  }
  const entry = COLORMAPS.find((c) => c.value === preset);
  const vtkName = entry?.vtkName ?? "Inferno (matplotlib)";
  if (vtkName) {
    const presetData = vtkColorMaps.getPresetByName(vtkName);
    if (presetData) {
      ctf.applyColorMap(presetData);
    }
  }
  if (max <= min) {
    // Degenerate: flat field. Avoid divide-by-zero in setMappingRange.
    ctf.setMappingRange(min - 1, min + 1);
  } else {
    ctf.setMappingRange(min, max);
  }
  ctf.updateRange();
}

function isVtkColormap(c: string | undefined): boolean {
  if (!c) return false;
  return COLORMAPS.some((m) => m.value === c);
}

function applyOpacity(
  pwf: ReturnType<typeof vtkPiecewiseFunction.newInstance>,
  transfer: ReturnType<typeof resolveVolumeTransferPolicy>,
  scale: number,
) {
  pwf.removeAllPoints();
  for (const point of volumeOpacityPoints(transfer, scale)) {
    pwf.addPoint(point.scalar, point.opacity);
  }
}

function formatDistance(metres: number): string {
  return metres >= 1_000 ? `${metres / 1_000} km` : `${metres} m`;
}

function applyFieldVolumeLighting(
  property: Parameters<typeof applyVolumeLighting>[0] & {
    setComputeNormalFromOpacity: (enabled: boolean) => unknown;
  },
  preset: Parameters<typeof applyVolumeLighting>[1],
  opacityClass: VolumeOpacityClass,
): void {
  property.setComputeNormalFromOpacity(volumeUsesOpacityNormals(opacityClass));
  if (volumeUsesStructuralLighting(opacityClass)) {
    applyVolumeLighting(
      property,
      preset,
      opacityClass === "sparse-positive" ? "structure" : "scalar",
    );
    return;
  }
  property.setShade(false);
  property.setAmbient(1);
  property.setDiffuse(0);
  property.setSpecular(0);
}

function applyVolumeInteractionLighting(
  property: Parameters<typeof applyFieldVolumeLighting>[0],
  _preview: boolean,
  preset: Parameters<typeof applyVolumeLighting>[1],
  opacityClass: VolumeOpacityClass,
): void {
  // Preview flattening is a uniform; material flags must stay stable for shader reuse.
  applyFieldVolumeLighting(property, preset, opacityClass);
}

type VolumeBounds = [number, number, number, number, number, number];

function unionPreparedVolumeBounds(
  prepared: readonly { shape: readonly number[]; spacing: readonly number[]; origin: readonly number[] }[],
  scale: { xy: number; z: number },
): VolumeBounds | null {
  if (prepared.length === 0) return null;
  const bounds: VolumeBounds = [
    Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
    Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
    Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
  ];
  for (const image of prepared) {
    for (let axis = 0; axis < 3; axis += 1) {
      const factor = axis === 2 ? scale.z : scale.xy;
      const lower = image.origin[axis] * factor;
      const upper = (image.origin[axis]
        + Math.max(0, image.shape[axis] - 1) * image.spacing[axis]) * factor;
      bounds[axis * 2] = Math.min(bounds[axis * 2], lower, upper);
      bounds[axis * 2 + 1] = Math.max(bounds[axis * 2 + 1], lower, upper);
    }
  }
  return bounds.every(Number.isFinite) ? bounds : null;
}

function scaledFeatureBounds(
  bounds: VolumeBounds | null | undefined,
  scale: { xy: number; z: number },
): VolumeBounds | null {
  if (!bounds || !bounds.every(Number.isFinite)) return null;
  const scaled: VolumeBounds = [
    bounds[0] * scale.xy,
    bounds[1] * scale.xy,
    bounds[2] * scale.xy,
    bounds[3] * scale.xy,
    bounds[4] * scale.z,
    bounds[5] * scale.z,
  ];
  return scaled;
}

function scaledHandoffFocus(
  focus: readonly [number | null, number | null, number | null] | null | undefined,
  scale: { xy: number; z: number },
): [number | null, number | null, number | null] | null {
  if (!focus) return null;
  return [
    focus[0] == null ? null : focus[0] * scale.xy,
    focus[1] == null ? null : focus[1] * scale.xy,
    focus[2] == null ? null : focus[2] * scale.z,
  ];
}
