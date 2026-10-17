import { lightingSettings, shadowMaterial } from "@/lib/plot/sceneAppearance";
import { useEffect, useMemo, useRef, useState } from "react";
import { fieldDefinitionIdentity } from "@/lib/run/runOutputs";
import { Plot3DAxisLabels, Plot3DColorbar, Plot3DDiscreteLegend, Plot3DOverlayStack, Plot3DScaleBadge } from "./Plot3DOverlay";
import { Plot3DCompass } from "./Plot3DCompass";
import { unitFor } from "@/lib/science/fieldUnits";
import { useCameraOrientation } from "@/lib/vtk/useCameraOrientation";
import { useCameraSync } from "@/lib/vtk/useCameraSync";
import { usePaneContext } from "./paneContext";
import { stormTrackCentre, useStormTrack } from "./useStormTrack";
import { scaleVoxelBudget } from "@/lib/plot/paneLayout";
import {
  coalesceNativeFieldWindows,
  fieldSupportsBoundedWindows,
  planInteractiveVolumeWindows,
} from "@/lib/plot/fieldLoadPlanning";
import { useRadarFloorLayer } from "./radarFloorLayer";
import { RadarFloorControls, useRadarFloorSelection } from "./RadarFloorControls";
import {
  CameraElevationControls,
  useCameraElevation,
  useCameraElevationPlacement,
} from "./CameraElevationControls";
import {
  CARTESIAN_VOXEL_CAP_ISOSURFACE,
} from "@/lib/science/cartesianInterp";
import {
  usePreparedScalarImage,
  usePreparedVelocityDiagnostic,
} from "@/lib/vtk/useVtkPreparation";
import {
  describeFrameTransition,
  findPublishedFrame,
  loadedFramesAreCoherent,
} from "@/lib/run/frameIdentity";
import { disposeVtkScene } from "@/lib/vtk/vtkLifecycle";
import { useVortexContourExtraction } from "@/lib/vtk/useVortexContourExtraction";
import { useBrickedIsosurfaceExtraction } from "@/lib/vtk/useBrickedIsosurfaceExtraction";
import { useNestedBrickedIsosurfaceExtraction } from "@/lib/vtk/useNestedBrickedIsosurfaceExtraction";
import { useVtkGpuResourceOwnership } from "@/lib/vtk/useVtkGpuResourceOwnership";
import type { VortexSurfaceGeometry } from "@/lib/vtk/vortexContourProtocol";
import {
  defaultsToRotationDominatedSurface,
  defaultsToPositiveQSurface,
  defaultsToSignedPair,
  initialPositiveQThreshold,
  initialRotationDominatedThreshold,
  isosurfaceWindowAggregations,
  physicalIsosurfaceThreshold,
  positiveQThresholdRange,
  rotationDominatedThresholdRange,
  signedMagnitudeScale,
  signedThresholdRange,
  supportsSignedPair,
} from "@/lib/plot/coherentStructures";
import { fieldsForPlotRanks, fieldsSharePlotRank } from "@/lib/plot/fieldPlotEligibility";
import { SURFACE_RESOLUTION_POLICY } from "@/lib/vtk/vtkResolution";
import {
  attachVtkCameraStability,
  attachVtkResize,
  configureTranslucentRendering,
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
  refitVtkCameraToBounds,
  unionVtkBounds,
  useSceneScaleFraming,
  resetVtkCameraToStormView,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useDebouncedValue,
  useField,
  useSceneScale,
  applySceneScale,
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
import { planNestedVolumeOwnership } from "@/lib/run/outputHierarchy";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { contrastAdjustedColor, sceneRgbToHex } from "@/lib/plot/sceneAppearance";
import {
  applySurfaceLighting,
  createVtkSceneLighting,
  type VtkSceneLighting,
} from "@/lib/vtk/sceneLighting";
import { useDomainBox } from "./domainBox";
import { useDomainAxes } from "./useDomainAxes";
import { usePanePresentationBarrier } from "./panePresentationContext";
import "./common.css";
import "./VolumeRender3D.css";

/* VTK.js marching-cubes isosurface. Same Cartesian pipeline as VolumeRender3D;
 * field-specific default thresholds (w≥10, qc≥0.1 g/kg, |ω|≥0.1 s⁻¹). */

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkTubeFilter from "@kitware/vtk.js/Filters/General/TubeFilter";
import { applyImmutablePolyDataGeometry } from "@/lib/vtk/vtkImmutablePolyData";

// Prefer vorticity_z over vorticity_magnitude; magnitude includes horizontal shear slabs absent in vorticity_z.
const PREFERRED_DEFAULTS = [
  "vorticity_z",
  "w",
  "qc",
  "vorticity_magnitude",
  "reflectivity_dbz",
  "theta",
];

const PRESET_COLORS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "#5b9bff", label: "Blue" },
  { value: "#5dbf81", label: "Green" },
  { value: "#d9a441", label: "Amber" },
  { value: "#e26370", label: "Red" },
  { value: "#a3a8b3", label: "Grey" },
];

const SIGNED_POSITIVE_COLOR = "#f26b4c";
const SIGNED_NEGATIVE_COLOR = "#4e9eff";
const Q_CRITERION_FIELD = "q_criterion";
const LAMBDA2_FIELD = "lambda2_criterion";
const SWIRLING_STRENGTH_FIELD = "swirling_strength";
const VELOCITY_COMPONENT_FIELDS = ["u", "v", "w"] as const;
const VELOCITY_DIAGNOSTICS = {
  [Q_CRITERION_FIELD]: "q-criterion",
  [LAMBDA2_FIELD]: "lambda-2",
  [SWIRLING_STRENGTH_FIELD]: "swirling-strength",
} as const;

export function Isosurface3D({
  runDir,
  steps,
  stepIdx,
  preset,
  onPresetCleared,
  onFrameDisplayChange,
  renderFidelityMode = "scientific",
  linkedField,
  onFieldChange,
}: PlotComponentProps) {
  const controlScope = useThreeDControlScope();
  const sceneAppearance = useSceneAppearance(controlScope.scope, "isosurface");
  const vtkBackend = useVtkBackend(controlScope.scope, "isosurface");
  const sceneAppearanceRef = useRef(sceneAppearance.resolved);
  sceneAppearanceRef.current = sceneAppearance.resolved;
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
  const hierarchyRequiredFields = useMemo(() =>
    fieldName && Object.hasOwn(VELOCITY_DIAGNOSTICS, fieldName)
      ? ["u", "v", "w"]
      : fieldName ? [fieldName] : [], [fieldName]);
  const outputHierarchySource = useOutputHierarchySource({
    runDir,
    steps,
    stepIdx,
    scope: controlScope.scope,
    requiredFields: hierarchyRequiredFields,
    focusPointM: stormCentre ? [stormCentre[0], stormCentre[1]] : null,
  });
  const currentStep = outputHierarchySource.currentStep;
  const sourceRunDir = outputHierarchySource.sourceRunDir;
  const sourceSteps = outputHierarchySource.sourceSteps;
  const fieldsForStep = useMemo(() => currentStep?.fields ?? [], [currentStep?.fields]);
  const qCriterionAvailable = fieldsSharePlotRank(
    sourceRunDir,
    currentStep,
    ["u", "v", "w"],
    [3],
  );
  /* Marching cubes needs a volume, so single-level products (cape, vil, composite_reflectivity,
   * ...) are excluded rather than offered and then rejected on load. */
  const availableFields = useMemo(() => {
    const renderable = fieldsForPlotRanks(sourceRunDir, currentStep, [3]);
    if (!qCriterionAvailable) {
      return renderable;
    }
    return [...renderable, ...Object.keys(VELOCITY_DIAGNOSTICS).filter((field) => !renderable.includes(field))];
  }, [currentStep, qCriterionAvailable, sourceRunDir]);

  const defaultField = useMemo(() => {
    for (const name of PREFERRED_DEFAULTS) {
      if (availableFields.includes(name)) return name;
    }
    return availableFields[0] ?? null;
  }, [availableFields]);

  const [threshold, setThreshold] = useState<number | null>(null);
  const [thresholdFieldName, setThresholdFieldName] = useState<string | null>(null);
  const committedThreshold = useDebouncedValue(threshold, 160);
  const [color, setColor] = useState<string>("#5b9bff");
  const [signedPair, setSignedPair] = useState(false);
  const [opacity, setOpacity] = useState<number>(0.7);
  const [showCoreLine, setShowCoreLine] = useState(true);
  const appearanceEvidenceRef = useRef<VtkAppearanceDescriptor>({
    plot: "isosurface",
    backend: vtkBackend.backend,
  });
  const committedAppearanceRef = useRef<VtkAppearanceDescriptor | null>(null);
  appearanceEvidenceRef.current = {
    plot: "isosurface",
    backend: vtkBackend.backend,
    datasetId: sourceRunDir,
    frameIndex: currentStep?.index ?? null,
    field: fieldName,
    fidelityMode: renderFidelityMode,
    parameters: {
      threshold,
      signedPair,
      opacity,
      showCoreLine,
      lighting: sceneAppearance.resolved.lighting,
      lightingControls: sceneAppearance.resolved.lightingOptions,
      background: sceneAppearance.resolved.backgroundHex,
    },
  };
  const [contourError, setContourError] = useState<string | null>(null);
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
  const paneVoxelCap = scaleVoxelBudget(CARTESIAN_VOXEL_CAP_ISOSURFACE, pane.voxelBudgetScale);
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const stormCenterX = stormCentre?.[0] ?? null;
  const stormCenterY = stormCentre?.[1] ?? null;
  const volumeWindowOutcome = useMemo(
    () => planInteractiveVolumeWindows(
      currentStep?.grid,
      paneVoxelCap,
      stormCenterX != null && stormCenterY != null ? [stormCenterX, stormCenterY] : null,
    ),
    [currentStep?.grid, paneVoxelCap, stormCenterX, stormCenterY],
  );
  const terrainBlocked = false;
  const volumeReductionRequired = Boolean(
    volumeWindowOutcome.ok
      && volumeWindowOutcome.plan.stride.some((stride) => stride > 1),
  );
  const windowAggregations = isosurfaceWindowAggregations(fieldName, volumeReductionRequired);
  const positiveWindowAggregation = windowAggregations[0];
  const negativeWindowAggregation = windowAggregations[1] ?? null;
  const usesSignedExtremaEnvelopes = negativeWindowAggregation != null;
  const positiveScalarWindows = useMemo(() => {
    if (!fieldName || !fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[fieldName])) {
      return null;
    }
    if (!volumeWindowOutcome.ok) return null;
    return coalesceNativeFieldWindows(volumeWindowOutcome.plan.windows.map((window) => ({
      ...window,
      aggregation: positiveWindowAggregation,
    })), {
      sourceKind: currentStep?.fieldOrigins?.[fieldName],
      terrainFollowing: currentStep?.grid?.terrainFollowing,
    });
  }, [currentStep?.fieldOrigins, currentStep?.grid?.terrainFollowing, fieldName, positiveWindowAggregation, volumeWindowOutcome]);
  const negativeScalarWindows = useMemo(() => {
    if (negativeWindowAggregation == null || !volumeWindowOutcome.ok) return null;
    return coalesceNativeFieldWindows(volumeWindowOutcome.plan.windows.map((window) => ({
      ...window,
      aggregation: negativeWindowAggregation,
    })), {
      sourceKind: fieldName ? currentStep?.fieldOrigins?.[fieldName] : undefined,
      terrainFollowing: currentStep?.grid?.terrainFollowing,
    });
  }, [currentStep?.fieldOrigins, currentStep?.grid?.terrainFollowing, fieldName, negativeWindowAggregation, volumeWindowOutcome]);
  const selectedVelocityDiagnostic = fieldName && Object.hasOwn(VELOCITY_DIAGNOSTICS, fieldName)
    ? VELOCITY_DIAGNOSTICS[fieldName as keyof typeof VELOCITY_DIAGNOSTICS]
    : null;
  const qCriterionSelected = selectedVelocityDiagnostic != null;
  const sourceDefinitionIdentity = JSON.stringify((qCriterionSelected ? VELOCITY_COMPONENT_FIELDS : [fieldName])
    .map(field => fieldDefinitionIdentity(sourceRunDir, currentStep?.index ?? null, field, currentStep?.publicationGeneration ?? null)));
  const scalarBrickEligible = Boolean(
    fieldName
      && currentStep?.fieldBrickHierarchies?.[fieldName]?.cullingSafe
      && physicalIsosurfaceThreshold(fieldName) != null
      && !qCriterionSelected
  );
  const diagnosticBrickEligible = Boolean(
    qCriterionSelected
      && VELOCITY_COMPONENT_FIELDS.every((name) => currentStep?.fieldBrickHierarchies?.[name])
  );
  const brickedContourEligible = scalarBrickEligible || diagnosticBrickEligible;
  const boundedProvider = qCriterionSelected
    ? ["u", "v", "w"].every((name) =>
        fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[name]))
    : Boolean(fieldName
        && fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[fieldName]));
  const qReductionBlocked = qCriterionSelected && !diagnosticBrickEligible && volumeWindowOutcome.ok
    && volumeWindowOutcome.plan.stride.some((stride) => stride > 1);
  const velocityWindows = useMemo(() => {
    if (!volumeWindowOutcome.ok || qReductionBlocked) return null;
    return volumeWindowOutcome.plan.windows;
  }, [qReductionBlocked, volumeWindowOutcome]);

  // Re-apply preset when its id changes.
  useEffect(() => {
    if (!preset) return;
    if (preset.field) setFieldName(preset.field);
  }, [preset]);

  useEffect(() => {
    if (!brickedContourEligible || !fieldName) return;
    const physical = physicalIsosurfaceThreshold(fieldName);
    if (physical == null || thresholdFieldName === fieldName) return;
    setThresholdFieldName(fieldName);
    setSignedPair(defaultsToSignedPair(fieldName));
    setThreshold(physical);
  }, [brickedContourEligible, fieldName, thresholdFieldName]);

  const clearPreset = () => onPresetCleared?.();

  // Marching cubes needs a volume, so a single-level field is refused rather than followed;
  // `declinedLink` explains that on the canvas.
  const declinedLink = useFieldLink(
    linkedField, fieldName, (f) => availableFields.includes(f), setFieldName,
  );

  useEffect(() => {
    if (fieldName == null || !availableFields.includes(fieldName)) {
      setFieldName(defaultField);
    }
  }, [availableFields, defaultField, fieldName]);

  // Debounce the scrubbed step so a fast slider drag loads only the settled
  // frame (load + marching cubes), not every frame passed through.
  const boundedLoadError = currentStep && fieldName && !terrainBlocked && !brickedContourEligible
    ? !boundedProvider
      ? "The selected field provider cannot serve bounded 3D windows."
      : !volumeWindowOutcome.ok
        ? volumeWindowOutcome.reason
        : null
    : null;
  const fieldLoadBlocked = terrainBlocked || Boolean(boundedLoadError)
    || qReductionBlocked || outputHierarchySource.loading;
  const positiveScalarFieldResult = useField(
    sourceRunDir,
    fieldLoadBlocked ? null : loadFrame.stepIndex,
    fieldLoadBlocked || qCriterionSelected || brickedContourEligible ? null : fieldName,
    loadFrame.publicationGeneration,
    positiveScalarWindows,
    { prefetch: !usesSignedExtremaEnvelopes },
  );
  const negativeScalarFieldResult = useField(
    sourceRunDir,
    fieldLoadBlocked || !usesSignedExtremaEnvelopes || brickedContourEligible ? null : loadFrame.stepIndex,
    fieldLoadBlocked || qCriterionSelected || !usesSignedExtremaEnvelopes || brickedContourEligible ? null : fieldName,
    loadFrame.publicationGeneration,
    negativeScalarWindows,
    { prefetch: false },
  );
  const qU = useField(
    sourceRunDir,
    loadFrame.stepIndex,
    qCriterionSelected && !fieldLoadBlocked && !diagnosticBrickEligible ? "u" : null,
    loadFrame.publicationGeneration,
    velocityWindows,
  );
  const qV = useField(
    sourceRunDir,
    loadFrame.stepIndex,
    qCriterionSelected && !fieldLoadBlocked && !diagnosticBrickEligible ? "v" : null,
    loadFrame.publicationGeneration,
    velocityWindows,
  );
  const qW = useField(
    sourceRunDir,
    loadFrame.stepIndex,
    qCriterionSelected && !fieldLoadBlocked && !diagnosticBrickEligible ? "w" : null,
    loadFrame.publicationGeneration,
    velocityWindows,
  );
  // Keep the tuple stable while its three coherent field objects are unchanged.
  // A fresh tuple every render restarts and aborts the preparation worker in a
  // loop, so a derived Q surface can otherwise remain permanently "loading".
  const qFieldsCoherent = loadedFramesAreCoherent(qU, qV, qW);
  const qFields = useMemo<[
    NonNullable<typeof qU.field>,
    NonNullable<typeof qV.field>,
    NonNullable<typeof qW.field>,
  ] | null>(() => {
    if (!qU.field || !qV.field || !qW.field || !qFieldsCoherent) {
      return null;
    }
    return [qU.field, qV.field, qW.field];
  }, [
    qU.field,
    qFieldsCoherent,
    qV.field,
    qW.field,
  ]);
  const signedExtremaCoherent = !usesSignedExtremaEnvelopes || loadedFramesAreCoherent(
    positiveScalarFieldResult,
    negativeScalarFieldResult,
  );
  const scalarFieldResult = usesSignedExtremaEnvelopes
    ? {
      ...positiveScalarFieldResult,
      field: signedExtremaCoherent && positiveScalarFieldResult.field
          && negativeScalarFieldResult.field
        ? positiveScalarFieldResult.field
        : null,
      loading: positiveScalarFieldResult.loading || negativeScalarFieldResult.loading,
      error: positiveScalarFieldResult.error || negativeScalarFieldResult.error,
    }
    : positiveScalarFieldResult;
  const fieldResult = qCriterionSelected
    ? {
      ...qU,
      field: qFields?.[0] ?? null,
      loading: qU.loading || qV.loading || qW.loading,
      error: qU.error || qV.error || qW.error,
    }
    : scalarFieldResult;
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
  const loadedStep = brickedContourEligible
    ? currentStep
    : findPublishedFrame(
        sourceSteps,
        fieldResult.loadedStepIndex,
        fieldResult.loadedPublicationGeneration,
      );
  const renderGrid = loadedStep?.grid ?? null;
  const [sceneScale, setSceneScale] = useSceneScale(renderGrid, undefined, controlScope.scope, "isosurface");
  const latestScaleFraming = useSceneScaleFraming(sceneScale);
  const scaleFramingRef = useRef(latestScaleFraming);
  scaleFramingRef.current = latestScaleFraming;
  const positiveScalarPreparation = usePreparedScalarImage(
    qCriterionSelected || brickedContourEligible ? null : positiveScalarFieldResult.field,
    renderGrid,
    paneVoxelCap,
    vtkLifecycleGeneration,
    {
      refineForDisplay: true,
      featureSign: 1,
      geometryHandoff: true,
    },
  );
  const negativeScalarPreparation = usePreparedScalarImage(
    qCriterionSelected || !usesSignedExtremaEnvelopes || brickedContourEligible
      ? null
      : negativeScalarFieldResult.field,
    renderGrid,
    paneVoxelCap,
    vtkLifecycleGeneration,
    {
      refineForDisplay: true,
      featureSign: -1,
      geometryHandoff: true,
    },
  );
  const qPreparation = usePreparedVelocityDiagnostic(
    qCriterionSelected ? qFields : null,
    renderGrid,
    paneVoxelCap,
    vtkLifecycleGeneration,
    selectedVelocityDiagnostic ?? "q-criterion",
    // Velocity-gradient diagnostics are already derived at the resolved model
    // lattice. Trilinearly refining them adds no scientific information and a
    // fragmented Q boundary can multiply marching-cubes output enough to kill
    // WebKit's worker before the retained-geometry guard can inspect it.
    { refineForDisplay: false, geometryHandoff: true },
  );
  const scalarPreparation = usesSignedExtremaEnvelopes
    ? {
      metadata: signedExtremaCoherent && positiveScalarPreparation.metadata
          && negativeScalarPreparation.metadata
        ? positiveScalarPreparation.metadata
        : null,
      handle: signedExtremaCoherent && positiveScalarPreparation.handle
          && negativeScalarPreparation.handle
        ? positiveScalarPreparation.handle
        : null,
      loading: positiveScalarPreparation.loading || negativeScalarPreparation.loading,
      error: positiveScalarPreparation.error || negativeScalarPreparation.error,
    }
    : positiveScalarPreparation;
  const preparation = qCriterionSelected ? qPreparation : scalarPreparation;
  const negativePrepared = usesSignedExtremaEnvelopes
    ? signedExtremaCoherent
      ? negativeScalarPreparation.metadata
      : null
    : preparation.metadata;
  const negativePreparedHandle = usesSignedExtremaEnvelopes
    ? signedExtremaCoherent
      ? negativeScalarPreparation.handle
      : null
    : preparation.handle;
  const brickedThresholdReady = thresholdFieldName === fieldName
    && committedThreshold != null
    && threshold === committedThreshold;
  const nestedContourLayers = useMemo(() => {
    if (!outputHierarchySource.selection || outputHierarchySource.selection.chain.length < 2) return [];
    const ownershipField = qCriterionSelected ? VELOCITY_COMPONENT_FIELDS[0] : fieldName;
    if (!ownershipField) return [];
    try {
      return planNestedVolumeOwnership(outputHierarchySource.selection, ownershipField);
    } catch {
      return [];
    }
  }, [fieldName, outputHierarchySource.selection, qCriterionSelected]);
  const nestedContourEnabled = brickedContourEligible && nestedContourLayers.length > 1
    && (diagnosticBrickEligible || brickedThresholdReady);
  const singleBrickedContourExtraction = useBrickedIsosurfaceExtraction({
    enabled: brickedContourEligible && !nestedContourEnabled,
    runDir: sourceRunDir,
    step: currentStep,
    fieldName,
    threshold: brickedThresholdReady ? committedThreshold : null,
    oppositeThreshold: !diagnosticBrickEligible && signedPair && defaultsToSignedPair(fieldName)
      && committedThreshold != null
      ? -Math.abs(committedThreshold)
      : null,
    diagnostic: diagnosticBrickEligible ? selectedVelocityDiagnostic : null,
    componentFields: diagnosticBrickEligible ? VELOCITY_COMPONENT_FIELDS : null,
    maximumPreparedCells: paneVoxelCap,
  });
  const nestedBrickedContourExtraction = useNestedBrickedIsosurfaceExtraction({
    enabled: nestedContourEnabled,
    layers: nestedContourLayers,
    fieldName,
    threshold: diagnosticBrickEligible && !brickedThresholdReady ? null : committedThreshold,
    oppositeThreshold: !diagnosticBrickEligible && signedPair && defaultsToSignedPair(fieldName)
      && committedThreshold != null ? -Math.abs(committedThreshold) : null,
    diagnostic: diagnosticBrickEligible ? selectedVelocityDiagnostic : null,
    componentFields: diagnosticBrickEligible ? VELOCITY_COMPONENT_FIELDS : null,
    maximumPreparedCells: paneVoxelCap,
  });
  const brickedContourExtraction = nestedContourEnabled
    ? nestedBrickedContourExtraction : singleBrickedContourExtraction;
  const nonFiniteSourceCount = brickedContourEligible
    ? brickedContourExtraction.nonFiniteCount
    : Math.max(
        preparation.metadata?.sourceNonFiniteCount ?? 0,
        negativePrepared?.sourceNonFiniteCount ?? 0,
      );

  useEffect(() => {
    setRenderedStepIndex(null);
    setRenderedPublicationGeneration(null);
  }, [sourceRunDir, fieldName, sourceDefinitionIdentity]);

  // Auto-derive an initial threshold when the field changes: the documented physical
  // default where one exists, otherwise a fraction of the value range (see below).
  const fieldStats = useMemo(() => {
    if (brickedContourEligible) return brickedContourExtraction.stats;
    const positive = preparation.metadata?.stats;
    if (!usesSignedExtremaEnvelopes) return positive ?? null;
    const negative = negativePrepared?.stats;
    if (!positive || !negative) return null;
    return {
      min: negative.min,
      max: positive.max,
      p2: negative.p2,
      p98: positive.p98,
      ...(positive.positiveP98 == null ? {} : { positiveP98: positive.positiveP98 }),
    };
  }, [brickedContourEligible, brickedContourExtraction.stats, negativePrepared?.stats, preparation.metadata?.stats, usesSignedExtremaEnvelopes]);
  const controlStatsRef = useRef<{ runDir: string; field: string | null; stats: NonNullable<typeof fieldStats> } | null>(null);
  if (fieldStats) controlStatsRef.current = { runDir: sourceRunDir, field: fieldName, stats: fieldStats };
  const controlStats = fieldStats ?? (controlStatsRef.current?.runDir === sourceRunDir
    && controlStatsRef.current.field === fieldName ? controlStatsRef.current.stats : null);
  const rotationDominatedSurface = Boolean(
    controlStats && defaultsToRotationDominatedSurface(fieldName) && controlStats.min < 0,
  );
  const positiveQSurface = Boolean(
    controlStats && defaultsToPositiveQSurface(fieldName) && controlStats.max > 0,
  );
  const signedPairAvailable = !rotationDominatedSurface &&
    !positiveQSurface &&
    supportsSignedPair(controlStats);
  const pairedContours = signedPair && signedPairAvailable;
  // Read by the scale effect without becoming a trigger for it (see below).
  const pairedContoursRef = useRef(pairedContours);
  pairedContoursRef.current = pairedContours;

  const hasField = renderGrid != null && (brickedContourEligible
    ? brickedContourExtraction.plan != null
    : fieldResult.field != null);
  const uniform = hasField && fieldStats != null && !(fieldStats.max > fieldStats.min);

  // A threshold is a value OF the field, so it carries the field's unit. Bare, "10" reads the same
  // for w (m/s) as for vorticity_z (s^-1), where it is orders past anything physical.
  const fieldUnit = fieldName === SWIRLING_STRENGTH_FIELD
    ? "s⁻¹"
    : fieldName === Q_CRITERION_FIELD || fieldName === LAMBDA2_FIELD
      ? "s⁻²"
      : fieldName
        ? unitFor(fieldName, loadedStep?.fieldUnits ?? currentStep?.fieldUnits)
        : null;
  const withFieldUnit = (text: string): string => (fieldUnit ? `${text} ${fieldUnit}` : text);

  // Threshold is latched per field; detect when it falls outside the current frame's range
  // (e.g. t=0 where max(w)=0.1 but threshold=10). Paired surface: vanishes only when
  // NEITHER signed contour crosses.
  const thresholdOutOfRange = useMemo(() => {
    if (!hasField || uniform || !fieldStats || committedThreshold == null) return false;
    if (pairedContours) {
      const magnitude = Math.abs(committedThreshold);
      return !(fieldStats.max > magnitude) && !(fieldStats.min < -magnitude);
    }
    return !(committedThreshold > fieldStats.min && committedThreshold < fieldStats.max);
  }, [hasField, uniform, fieldStats, committedThreshold, pairedContours]);

  const contourReady = thresholdFieldName === fieldName
    && committedThreshold != null
    && threshold === committedThreshold
    && !thresholdOutOfRange
    && (brickedContourEligible || fieldResult.field != null);
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      missingSupport: nonFiniteSourceCount > 0,
      missingDataPolicy: "omit-incomplete-cells",
      representation: brickedContourEligible ? "progressive-native-bricks" : "bounded-monolithic",
      brickHierarchyLevel: brickedContourExtraction.plan?.level ?? null,
      brickGeometry: brickedContourEligible && renderGrid?.terrainFollowing
        ? "terrain-physical-centroids"
        : "separable-uniform",
      selectedBrickCount: brickedContourExtraction.plan?.items.length ?? null,
      culledBrickCount: brickedContourExtraction.plan?.culledBrickCount ?? null,
      brickSourceBytes: brickedContourExtraction.plan?.sourceBytes ?? null,
      residentBrickCount: brickedContourExtraction.residentBrickCount,
      rawBrickTriangles: brickedContourExtraction.rawTriangleCounts,
      mergedBrickTriangles: {
        primary: brickedContourExtraction.geometry?.triangles ?? 0,
        opposite: brickedContourExtraction.oppositeGeometry?.triangles ?? 0,
      },
      nestedLayerCount: nestedContourEnabled ? nestedBrickedContourExtraction.layerCount : 1,
      nestedTransitionTriangles: nestedContourEnabled
        ? nestedBrickedContourExtraction.transitionTriangles : 0,
      nestedAtomicAdmission: nestedContourEnabled
        ? nestedBrickedContourExtraction.atomicAdmission : null,
      nestedComposition: nestedContourEnabled ? "worker-streamed" : null,
    },
  };
  const contourMagnitude = Math.abs(committedThreshold ?? 0);
  const primaryContourPreparedHandle = !pairedContours && (committedThreshold ?? 0) < 0
    ? negativePreparedHandle
    : preparation.handle;
  const primaryContourExtraction = useVortexContourExtraction({
    // Keep the worker and last completed surface warm through the normal
    // prepared-image gap between playback frames.
    enabled: contourReady && !brickedContourEligible,
    preparedHandle: primaryContourPreparedHandle,
    thresholds: {
      core: contourReady
        ? pairedContours ? contourMagnitude : committedThreshold
        : null,
      opposite: null,
      mesocyclone: null,
    },
  });
  const negativeContourExtraction = useVortexContourExtraction({
    enabled: contourReady && pairedContours && !brickedContourEligible,
    preparedHandle: negativePreparedHandle,
    thresholds: {
      core: contourReady && pairedContours ? -contourMagnitude : null,
      opposite: null,
      mesocyclone: null,
    },
  });
  const contourExtractionLoading = (brickedContourEligible
    ? brickedContourExtraction.loading
    : primaryContourExtraction.loading)
    || negativeContourExtraction.loading;
  const gpuPrimaryGeometry = brickedContourEligible
    ? brickedContourExtraction.geometry
    : primaryContourExtraction.result?.core ?? null;
  const gpuNegativeGeometry = brickedContourEligible
    ? brickedContourExtraction.oppositeGeometry
    : pairedContours ? negativeContourExtraction.result?.core ?? null : null;
  const gpuUploadBytes = (gpuPrimaryGeometry?.bytes ?? 0) + (gpuNegativeGeometry?.bytes ?? 0);
  const gpuUploadGeneration = JSON.stringify([
    sourceRunDir, requestedStepIndex, requestedPublicationGeneration, fieldName, sourceDefinitionIdentity,
    committedThreshold, pairedContours, vtkLifecycleGeneration,
  ]);
  const gpuUpload = useVtkGpuResourceOwnership({
    enabled: contourReady && !contourExtractionLoading && !thresholdOutOfRange
      && !brickedContourExtraction.error && !primaryContourExtraction.error
      && !negativeContourExtraction.error && gpuUploadBytes > 0,
    owner: "isosurface",
    generation: gpuUploadGeneration,
    bytes: gpuUploadBytes,
  });
  appearanceEvidenceRef.current = {
    ...appearanceEvidenceRef.current,
    parameters: {
      ...appearanceEvidenceRef.current.parameters,
      logicalGpuUploadBytes: gpuUpload.admitted ? gpuUploadBytes : 0,
    },
  };
  useEffect(() => {
    if (contourExtractionLoading) {
      setContourError(null);
    }
  }, [contourExtractionLoading]);


  // Re-pick auto-threshold only when the field changes, not every frame (preserves user-adjusted threshold).
  const threshInitedFieldRef = useRef<string | null>(null);
  useEffect(() => {
    if (!fieldStats || fieldName == null) return;
    if (threshInitedFieldRef.current === fieldName) return;
    // Thresholds are values of a specific field. Keep that identity alongside
    // the value so a field switch cannot briefly contour the new lattice with
    // the previous field's (possibly opposite-signed) threshold.
    setThresholdFieldName(fieldName);
    const initializePair = defaultsToSignedPair(fieldName) && supportsSignedPair(fieldStats);
    setSignedPair(initializePair);
    const span = fieldStats.max - fieldStats.min;
    const physical = physicalIsosurfaceThreshold(fieldName);
    if (!(span > 0)) {
      // Clear-air startup is not a usable automatic level. Initialize when
      // structure first appears, then hold that level across later frames.
      setThreshold(physical ?? fieldStats.min);
      return;
    }
    threshInitedFieldRef.current = fieldName;
    // Use physical default as-is; clamping to a [2%, 98%] band would silently draw a different surface.
    if (defaultsToPositiveQSurface(fieldName) && fieldStats.max > 0) {
      setSignedPair(false);
      setThreshold(initialPositiveQThreshold(fieldStats));
      return;
    }
    if (defaultsToRotationDominatedSurface(fieldName) && fieldStats.min < 0) {
      setSignedPair(false);
      setThreshold(initialRotationDominatedThreshold(fieldStats));
      return;
    }
    if (initializePair) {
      const magnitudeScale = signedMagnitudeScale(fieldStats);
      const fullMagnitude = Math.max(Math.abs(fieldStats.min), Math.abs(fieldStats.max));
      if (physical != null && physical > 0 && physical <= fullMagnitude) {
        setThreshold(physical);
      } else {
        setThreshold(magnitudeScale * 0.35);
      }
      return;
    }
    if (physical != null && physical >= fieldStats.min && physical <= fieldStats.max) {
      setThreshold(physical);
      return;
    }
    // No physical default: start at 60% of range, clamped inside slider's [2%, 98%] usable band.
    const initial = fieldStats.min + 0.6 * span;
    const lo = fieldStats.min + 0.02 * span;
    const hi = fieldStats.max - 0.02 * span;
    setThreshold(Math.min(Math.max(initial, lo), hi));
  }, [fieldName, fieldStats]);

  // ---------- vtk.js scene refs ----------
  const containerRef = useRef<HTMLDivElement | null>(null);
  type SceneRefs = {
    renderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    surfaceData: ReturnType<typeof vtkPolyData.newInstance>;
    actor: ReturnType<typeof vtkActor.newInstance>;
    mapper: ReturnType<typeof vtkMapper.newInstance>;
    negativeSurfaceData: ReturnType<typeof vtkPolyData.newInstance>;
    negativeActor: ReturnType<typeof vtkActor.newInstance>;
    negativeMapper: ReturnType<typeof vtkMapper.newInstance>;
    centerlineData: ReturnType<typeof vtkPolyData.newInstance>;
    centerlineTube: ReturnType<typeof vtkTubeFilter.newInstance>;
    centerlineActor: ReturnType<typeof vtkActor.newInstance>;
    centerlineMapper: ReturnType<typeof vtkMapper.newInstance>;
    lighting: VtkSceneLighting;
  };
  const sceneRef = useRef<SceneRefs | null>(null);
  const cameraInitializedRef = useRef(false);
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
    `isosurface:${sourceRunDir}`,
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
  const panePreparationError = floor.error ?? fieldResult.error ?? preparation.error ?? brickedContourExtraction.error
    ?? primaryContourExtraction.error ?? negativeContourExtraction.error ?? gpuUpload.error ?? boundedLoadError;
  const panePreparationReady = !fieldLoadBlocked && !panePreparationError && !fieldResult.loading
    && !preparation.loading && !contourExtractionLoading && loadedStep != null
    && loadedStep.index === requestedStepIndex && loadedStep.publicationGeneration === requestedPublicationGeneration
    && thresholdFieldName === fieldName && committedThreshold != null && threshold === committedThreshold
    && (thresholdOutOfRange || uniform || (contourReady
      && (brickedContourEligible ? brickedContourExtraction.plan != null : primaryContourExtraction.result != null)
      && (!pairedContours || brickedContourEligible || negativeContourExtraction.result != null)))
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
      loading: !fieldLoadBlocked
        && (fieldResult.loading || preparation.loading
          || contourExtractionLoading
          || frameTransition.kind !== "ready"
          || !panePreparationReady || panePresentation.pending || floor.loading || floor.error != null
          || floorSelection.enabled && floor.committedPresentationIdentity !== floor.presentationIdentity),
    });
  }, [
    panePreparationReady, panePresentation.pending, floor.loading, floor.error, floorSelection.enabled,
    floor.committedPresentationIdentity, floor.presentationIdentity,
    onFrameDisplayChange,
    requestedStepIndex,
    requestedPublicationGeneration,
    renderedStepIndex,
    renderedPublicationGeneration,
    fieldResult.loading,
    preparation.loading,
    contourExtractionLoading,
    frameTransition.kind,
    fieldLoadBlocked,
    nonFiniteSourceCount,
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
  // Reset camera once after first data load (empty marching-cubes output has no bounds).

  // Re-frame on run change; also re-pick threshold so run B doesn't inherit run A's value.
  useVtkRunReset(sourceRunDir, cameraInitializedRef, () => {
    threshInitedFieldRef.current = null;
    const physicalThreshold = physicalIsosurfaceThreshold(fieldName);
    if (brickedContourEligible && fieldName && physicalThreshold != null) {
      setThresholdFieldName(fieldName);
      setSignedPair(defaultsToSignedPair(fieldName));
      setThreshold(physicalThreshold);
    } else {
      setThresholdFieldName(null);
    }
    const s = sceneRef.current;
    if (!s) return;
    s.actor.setVisibility(false);
    s.negativeActor.setVisibility(false);
    s.centerlineActor.setVisibility(false);
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

    const surfaceData = vtkPolyData.newInstance();
    const mapper = vtkMapper.newInstance({ scalarVisibility: false });
    mapper.setInputData(surfaceData);

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    // Hidden until per-field effect has real threshold + data; avoids running marching cubes on empty pipeline.
    actor.setVisibility(false);

    const negativeSurfaceData = vtkPolyData.newInstance();
    const negativeMapper = vtkMapper.newInstance({ scalarVisibility: false });
    negativeMapper.setInputData(negativeSurfaceData);
    const negativeActor = vtkActor.newInstance();
    negativeActor.setMapper(negativeMapper);
    negativeActor.setVisibility(false);

    const centerlineData = vtkPolyData.newInstance();
    const centerlineTube = vtkTubeFilter.newInstance({
      radius: 100,
      numberOfSides: 8,
      capping: true,
    });
    centerlineTube.setInputData(centerlineData);
    const centerlineMapper = vtkMapper.newInstance({ scalarVisibility: false });
    centerlineMapper.setInputConnection(centerlineTube.getOutputPort());
    const centerlineActor = vtkActor.newInstance();
    centerlineActor.setMapper(centerlineMapper);
    centerlineActor.getProperty().setColor(1, 0.88, 0.25);
    centerlineActor.getProperty().setOpacity(0.95);
    centerlineActor.setVisibility(false);

    const renderer = renderWindow.getRenderer();
    configureTranslucentRendering(renderer);
    const lighting = createVtkSceneLighting(renderer);
    lighting.apply(
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );
    applyIsosurfaceLighting(
      actor.getProperty(),
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );
    applyIsosurfaceLighting(
      negativeActor.getProperty(),
      sceneAppearanceRef.current.lightingOptions,
      sceneAppearanceRef.current.isLight,
    );
    applySurfaceLighting(centerlineActor.getProperty(), sceneAppearanceRef.current.lightingOptions);
    renderer.addActor(actor);
    renderer.addActor(negativeActor);
    renderer.addActor(centerlineActor);
    // No resetCamera() yet: marching cubes has no output until first field+threshold. Deferred to per-field effect.

    sceneRef.current = {
      renderWindow,
      surfaceData,
      actor,
      mapper,
      negativeSurfaceData,
      negativeActor,
      negativeMapper,
      centerlineData,
      centerlineTube,
      centerlineActor,
      centerlineMapper,
      lighting,
    };

    registerVtkAppearanceCapture(container, renderWindow, () => ({
      ...(committedAppearanceRef.current ?? appearanceEvidenceRef.current),
      parameters: { ...(committedAppearanceRef.current?.parameters ?? appearanceEvidenceRef.current.parameters),
        requestedFrameIndex: appearanceEvidenceRef.current.parameters?.requestedFrameIndex ?? null },
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
          () => renderer.removeActor(negativeActor),
          () => renderer.removeActor(centerlineActor),
        ],
        resources: [
          actor,
          mapper,
          surfaceData,
          negativeActor,
          negativeMapper,
          negativeSurfaceData,
          centerlineActor,
          centerlineMapper,
          centerlineTube,
          centerlineData,
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
    applyIsosurfaceLighting(scene.actor.getProperty(), appearance.lightingOptions, appearance.isLight);
    applyIsosurfaceLighting(
      scene.negativeActor.getProperty(),
      appearance.lightingOptions,
      appearance.isLight,
    );
    applySurfaceLighting(scene.centerlineActor.getProperty(), appearance.lightingOptions);
    const positive = contrastAdjustedColor(
      hexToRgb(pairedContours ? SIGNED_POSITIVE_COLOR : color),
      appearance.background,
    );
    scene.actor.getProperty().setColor(...positive);
    scene.negativeActor.getProperty().setColor(...contrastAdjustedColor(
      hexToRgb(SIGNED_NEGATIVE_COLOR),
      appearance.background,
    ));
    scene.centerlineActor.getProperty().setColor(...contrastAdjustedColor(
      [1, 0.88, 0.25],
      appearance.background,
    ));
    scene.renderWindow.getRenderWindow().render();
  }, [sceneAppearance.resolved, pairedContours, color, vtkLifecycleGeneration, panePresentation.pending]);

  // ---------- Per-field centerline update ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || panePresentation.pending || !fieldResult.field || !renderGrid || !preparation.metadata) return;
    const centerline = preparation.metadata.vortexCenterline;
    if (centerline && centerline.length >= 6) {
      const pointCount = centerline.length / 3;
      scene.centerlineData.getPoints().setData(centerline, 3);
      scene.centerlineData.getLines().setData(Uint32Array.from([
        pointCount,
        ...Array.from({ length: pointCount }, (_, index) => index),
      ]));
      scene.centerlineTube.setRadius(Math.max(40, Math.min(...preparation.metadata.spacing) * 0.3));
      scene.centerlineData.modified();
      scene.centerlineTube.modified();
    } else {
      scene.centerlineData.getPoints().setData(new Float32Array(), 3);
      scene.centerlineData.getLines().setData(new Uint32Array());
      scene.centerlineData.modified();
    }
  }, [
    fieldResult.field,
    renderGrid,
    preparation.metadata,
    nonFiniteSourceCount,
    fieldName,
    vtkLifecycleGeneration,
    panePresentation.pending,
  ]);

  // ---------- Threshold / geometry ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || panePresentation.pending || !panePreparationReady || floor.loading || floor.error || (!brickedContourEligible && !preparation.metadata)) return;
    if (thresholdFieldName !== fieldName) {
      scene.actor.setVisibility(false);
      scene.negativeActor.setVisibility(false);
      scene.centerlineActor.setVisibility(false);
      scene.renderWindow.getRenderWindow().render();
      return;
    }
    // Keep the last valid surface while a slider settles, but never feed its
    // preview value into marching cubes. This also gives an auto-threshold one
    // render to replace the prior field's committed value.
    if (committedThreshold == null || threshold !== committedThreshold) return;
    if (thresholdOutOfRange || uniform) {
      applySurfaceGeometry(scene.surfaceData, null);
      applySurfaceGeometry(scene.negativeSurfaceData, null);
      scene.centerlineData.getPoints().setData(new Float32Array(), 3);
      scene.centerlineData.getLines().setData(new Uint32Array());
      scene.centerlineData.modified();
      scene.actor.setVisibility(false);
      scene.negativeActor.setVisibility(false);
      scene.centerlineActor.setVisibility(false);
      setContourError(null);
      floor.commitData();
      scene.renderWindow.getRenderWindow().render();
      committedGridRef.current = renderGrid;
      committedAppearanceRef.current = { ...appearanceEvidenceRef.current,
        frameIndex: (brickedContourEligible ? currentStep : loadedStep)?.index ?? null, parameters: { ...appearanceEvidenceRef.current.parameters,
          renderedPublicationGeneration: (brickedContourEligible ? currentStep : loadedStep)?.publicationGeneration ?? null } };
      if (brickedContourEligible && currentStep) {
        markFrameRendered(contextGeneration);
        setRenderedStepIndex(currentStep.index);
        setRenderedPublicationGeneration(currentStep.publicationGeneration);
      } else if (fieldResult.field) {
        markFrameRendered(contextGeneration);
        setRenderedStepIndex(fieldResult.loadedStepIndex);
        setRenderedPublicationGeneration(fieldResult.loadedPublicationGeneration);
      }
      return;
    }
    const primaryError = gpuUpload.error ?? (brickedContourEligible
      ? brickedContourExtraction.error
      : primaryContourExtraction.error);
    if (primaryError || negativeContourExtraction.error) {
      scene.actor.setVisibility(false);
      scene.negativeActor.setVisibility(false);
      scene.centerlineActor.setVisibility(false);
      setContourError(
        `Isosurface extraction failed: ${primaryError ?? negativeContourExtraction.error}`,
      );
      scene.renderWindow.getRenderWindow().render();
      return;
    }
    const primaryResult = primaryContourExtraction.result;
    const negativeResult = pairedContours ? negativeContourExtraction.result : null;
    const brickedComplete = brickedContourEligible
      && brickedContourExtraction.plan != null
      && !brickedContourExtraction.loading;
    if (brickedContourEligible ? !brickedComplete : !primaryResult) return;
    if (pairedContours && !brickedContourEligible && !negativeResult) return;
    const primaryGeometry = brickedContourEligible
      ? brickedContourExtraction.geometry
      : primaryResult?.core ?? null;
    const negativeGeometry = brickedContourEligible
      ? brickedContourExtraction.oppositeGeometry
      : negativeResult?.core ?? null;
    if (((primaryGeometry?.bytes ?? 0) + (negativeGeometry?.bytes ?? 0)) > 0
        && !gpuUpload.admitted) return;

    setContourError(null);
    applySurfaceGeometry(scene.surfaceData, primaryGeometry);
    applySurfaceGeometry(scene.negativeSurfaceData, negativeGeometry);
    scene.actor.setVisibility((primaryGeometry?.triangles ?? 0) > 0);
    scene.negativeActor.setVisibility(
      pairedContours && (negativeGeometry?.triangles ?? 0) > 0,
    );
    scene.centerlineActor.setVisibility(
      showCoreLine && qCriterionSelected
        && (preparation.metadata?.vortexCenterline?.length ?? 0) >= 6,
    );
    if (!cameraInitializedRef.current) {
      resetVtkCameraToStormView(
        scene.renderWindow.getRenderer(),
        unionVtkBounds(
          scene.actor.getBounds(),
          pairedContours ? scene.negativeActor.getBounds() : null,
        ),
      );
      cameraInitializedRef.current = true;
    }
    floor.commitData();
    scene.renderWindow.getRenderWindow().render();
    committedGridRef.current = renderGrid;
    committedAppearanceRef.current = { ...appearanceEvidenceRef.current,
      frameIndex: (brickedContourEligible ? currentStep : loadedStep)?.index ?? null, parameters: { ...appearanceEvidenceRef.current.parameters,
        renderedPublicationGeneration: (brickedContourEligible ? currentStep : loadedStep)?.publicationGeneration ?? null } };
    if (brickedContourEligible && currentStep) {
      markFrameRendered(contextGeneration);
      setRenderedStepIndex(currentStep.index);
      setRenderedPublicationGeneration(currentStep.publicationGeneration);
    } else if (fieldResult.field) {
      markFrameRendered(contextGeneration);
      setRenderedStepIndex(fieldResult.loadedStepIndex);
      setRenderedPublicationGeneration(fieldResult.loadedPublicationGeneration);
    }
  }, [
    committedThreshold,
    panePresentation.pending,
    panePreparationReady,
    floor.loading, floor.error, floor.commitData,
    uniform,
    renderGrid,
    threshold,
    thresholdFieldName,
    fieldName,
    pairedContours,
    preparation.metadata,
    nonFiniteSourceCount,
    fieldResult.field,
    fieldResult.loadedStepIndex,
    fieldResult.loadedPublicationGeneration,
    sourceDefinitionIdentity,
    contextGeneration,
    markFrameRendered,
    thresholdOutOfRange,
    showCoreLine,
    qCriterionSelected,
    primaryContourExtraction.result,
    primaryContourExtraction.error,
    brickedContourEligible,
    brickedContourExtraction.geometry,
    brickedContourExtraction.oppositeGeometry,
    brickedContourExtraction.plan,
    brickedContourExtraction.loading,
    brickedContourExtraction.error,
    gpuUpload.admitted,
    gpuUpload.error,
    currentStep,
    negativeContourExtraction.result,
    negativeContourExtraction.error,
    vtkLifecycleGeneration,
  ]);

  // ---------- Surface opacity ----------
  // Actor-only changes must not invalidate or re-measure marching-cubes output.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    scene.actor.getProperty().setOpacity(opacity);
    scene.negativeActor.getProperty().setOpacity(opacity);
    scene.renderWindow.getRenderWindow().render();
  }, [opacity, vtkLifecycleGeneration]);

  // ---------- Vertical exaggeration ----------
  // Display-only z-scale; re-frame only after data exists (empty bounds NaN the camera).
  // Fit the isosurface bounds only, not the domain box, to avoid zooming out on every scale tick.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const currentScale = { xy: sceneScale.xy, z: sceneScale.z };
    const scaleFraming = scaleFramingRef.current;
    applySceneScale(currentScale, scene.actor);
    applySceneScale(currentScale, scene.negativeActor);
    applySceneScale(currentScale, scene.centerlineActor);
    if (cameraInitializedRef.current && scaleFraming.consume()) {
      const focus = unionVtkBounds(
        scene.actor.getBounds(),
        pairedContoursRef.current ? scene.negativeActor.getBounds() : null,
      );
      refitVtkCameraToBounds(scene.renderWindow.getRenderer(), focus ?? undefined);
    }
    scene.renderWindow.getRenderWindow().render();
    // pairedContours read through a ref to avoid firing camera re-fit on every fieldStats change.
  }, [sceneScale.xy, sceneScale.z, vtkLifecycleGeneration]);

  // ---------- Reset camera ----------
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    resetVtkCameraToStormView(
      scene.renderWindow.getRenderer(),
      unionVtkBounds(
        scene.actor.getBounds(),
        pairedContoursRef.current ? scene.negativeActor.getBounds() : null,
      ),
    );
    scene.renderWindow.getRenderWindow().render();
  }, [resetTick]);

  const fieldOptions = useMemo(
    () => availableFields.map((f) => ({
      value: f,
      label: f === Q_CRITERION_FIELD
        ? "Q criterion (derived from u/v/w)"
        : f === LAMBDA2_FIELD
          ? "Lambda-2 criterion (derived from u/v/w)"
          : f === SWIRLING_STRENGTH_FIELD
            ? "Swirling strength λci (derived from u/v/w)"
            : f,
    })),
    [availableFields],
  );
  const contourFrameCommitted = renderedStepIndex === (brickedContourEligible
    ? currentStep?.index ?? null
    : fieldResult.loadedStepIndex)
    && renderedPublicationGeneration === (brickedContourEligible
      ? currentStep?.publicationGeneration ?? null
      : fieldResult.loadedPublicationGeneration)
    && !contourExtractionLoading
    && contourError == null;
  const primarySurfaceVisible = contourFrameCommitted
    && (brickedContourEligible
      ? brickedContourExtraction.geometry?.triangles ?? 0
      : primaryContourExtraction.result?.core?.triangles ?? 0) > 0;
  const negativeSurfaceVisible = contourFrameCommitted && pairedContours
    && (brickedContourEligible
      ? brickedContourExtraction.oppositeGeometry?.triangles ?? 0
      : negativeContourExtraction.result?.core?.triangles ?? 0) > 0;

  // Slider range: span 5% to 95% of the actual field range so the user
  // doesn't waste precision on the extremes.
  const sliderRange = useMemo(() => {
    if (!controlStats) return { min: 0, max: 1, step: 0.01 };
    if (positiveQSurface) return positiveQThresholdRange(controlStats);
    if (rotationDominatedSurface) return rotationDominatedThresholdRange(controlStats);
    if (pairedContours) return signedThresholdRange(controlStats);
    const span = controlStats.max - controlStats.min;
    const lo = controlStats.min + 0.02 * span;
    const hi = controlStats.max - 0.02 * span;
    // ~200 steps across the range — fine-grained enough for any plausible field.
    const step = span > 0 ? span / 200 : 0.001;
    return { min: lo, max: hi, step };
  }, [controlStats, pairedContours, positiveQSurface, rotationDominatedSurface]);

  return (
    <div className="volume-render">
      <Grouped3DControls
        ariaLabel="Isosurface controls"
        scope={controlScope}
        groups={[
          {
            id: "data",
            label: "Data",
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
              setThresholdFieldName(null);
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
          <ControlGroup label="Structure">
            <PillGroup
              options={[
                { value: "single", label: "Single level" },
                { value: "signed", label: "Signed ± pair", unavailable: signedPairAvailable ? undefined : "Requires a field with positive and negative values" },
              ]}
              value={pairedContours ? "signed" : "single"}
              onChange={(value) => {
                setSignedPair(value === "signed");
                if (value === "signed" && fieldStats) {
                  const range = signedThresholdRange(fieldStats);
                  setThreshold((current) => Math.min(
                    range.max,
                    Math.max(range.min, Math.abs(current ?? range.min)),
                  ));
                }
                clearPreset();
              }}
            />
          </ControlGroup>
        {qCriterionSelected && (
          <ControlGroup label="Core ridge">
            <PillGroup
              options={[
                { value: "on", label: "Show" },
                { value: "off", label: "Hide" },
              ]}
              value={showCoreLine ? "on" : "off"}
              onChange={(value) => setShowCoreLine(value === "on")}
            />
          </ControlGroup>
        )}
            </>,
          },
          {
            id: "surface",
            label: "Surface",
            status: pairedContours ? "Signed ± pair" : "Single",
            controls: <>
        <ControlGroup label="Threshold">
          <input
            type="range"
            min={sliderRange.min}
            max={sliderRange.max}
            step={sliderRange.step}
            value={threshold ?? sliderRange.min}
            title={uniform ? "A constant field has no crossing level" : "Isovalue in the field's physical units"}
            onChange={(e) => setThreshold(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
            disabled={!hasField || uniform}
          />
          <span className="volume-render__slider-value">
            {threshold == null
              ? "—"
              : pairedContours
              ? withFieldUnit(`±${formatThreshold(Math.abs(threshold))}`)
              : withFieldUnit(formatThreshold(threshold))}
            {threshold !== committedThreshold ? " · preview" : ""}
          </span>
        </ControlGroup>
        <button type="button" className="btn btn--secondary volume-render__reset"
          disabled={!fieldStats || !(fieldStats.max > fieldStats.min)}
          title="Choose an isovalue inside this frame's finite range; constant fields have no surface"
          onClick={() => {
            if (!fieldStats || !(fieldStats.max > fieldStats.min)) return;
            setThresholdFieldName(fieldName);
            setThreshold(positiveQSurface ? initialPositiveQThreshold(fieldStats)
              : rotationDominatedSurface ? initialRotationDominatedThreshold(fieldStats)
              : pairedContours ? signedMagnitudeScale(fieldStats) * 0.35
              : fieldStats.min + 0.6 * (fieldStats.max - fieldStats.min));
            clearPreset();
          }}>Fit threshold</button>
        <ControlGroup label="Color">
          <div className="iso-color-row">
            {PRESET_COLORS.map((c) => (
              <button
                key={c.value}
                type="button"
                aria-label={c.label}
                title={pairedContours ? "Signed pairs use distinct positive and negative colors" : c.label}
                disabled={pairedContours}
                onClick={() => setColor(c.value)}
                className={`iso-swatch ${color === c.value ? "iso-swatch--active" : ""}`}
                style={{ background: c.value }}
              />
            ))}
          </div>
        </ControlGroup>
        <ControlGroup label="Opacity">
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number.parseFloat(e.target.value))}
            className="volume-render__slider"
          />
          <span className="volume-render__slider-value">{opacity.toFixed(2)}</span>
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
        data-contour-loading={contourExtractionLoading}
        data-contour-ready={contourReady}
        data-contour-threshold={threshold ?? ""}
        data-contour-committed-threshold={committedThreshold ?? ""}
        data-contour-threshold-field={thresholdFieldName ?? ""}
        data-contour-field={fieldName ?? ""}
        data-contour-paired={pairedContours}
        data-contour-representation={brickedContourEligible ? "progressive-bricks" : "bounded-monolithic"}
        data-scalar-brick-eligible={scalarBrickEligible}
        data-diagnostic-brick-eligible={diagnosticBrickEligible}
        data-declared-brick-fields={Object.keys(currentStep?.fieldBrickHierarchies ?? {}).sort().join(",")}
        data-brick-resident-count={brickedContourExtraction.residentBrickCount}
        data-brick-raw-primary-triangles={brickedContourExtraction.rawTriangleCounts.primary}
        data-brick-raw-opposite-triangles={brickedContourExtraction.rawTriangleCounts.opposite}
        data-brick-merged-primary-triangles={brickedContourExtraction.geometry?.triangles ?? 0}
        data-brick-merged-opposite-triangles={brickedContourExtraction.oppositeGeometry?.triangles ?? 0}
        data-brick-error={brickedContourExtraction.error ?? ""}
        data-brick-loading={brickedContourExtraction.loading}
        data-brick-plan-count={brickedContourExtraction.plan?.items.length ?? 0}
        data-brick-hierarchy-level={brickedContourExtraction.plan?.level ?? ""}
        data-nested-contour-planned-layers={nestedContourLayers.length}
        data-nested-contour-enabled={nestedContourEnabled}
        data-nested-contour-composition={nestedContourEnabled ? "worker-streamed" : "none"}
        data-nested-contour-layers={nestedContourEnabled ? nestedBrickedContourExtraction.layerCount : 1}
        data-nested-contour-transition-triangles={nestedContourEnabled
          ? nestedBrickedContourExtraction.transitionTriangles : 0}
        data-nested-contour-atomic-admission={nestedContourEnabled
          ? nestedBrickedContourExtraction.atomicAdmission : false}
        data-nested-contour-child-bounds-lower={nestedContourLayers[0]?.excludeChildBoundsM?.lower.join(",") ?? ""}
        data-brick-geometry={brickedContourEligible && renderGrid?.terrainFollowing
          ? "terrain-physical-centroids"
          : "separable-uniform"}
        data-logical-gpu-upload-bytes={gpuUpload.admitted ? gpuUploadBytes : 0}
        data-output-level={outputHierarchySource.activeLevel?.levelId ?? "root"}
        data-output-hierarchy-valid={outputHierarchySource.valid}
      >
        <div
          ref={containerRef}
          className="volume-render__canvas"
          role="img"
          aria-label="Interactive three-dimensional isosurface; drag to orbit, use camera controls for keyboard adjustment"
        />
        <Plot3DAxisLabels labels={axisLabels} />
        <Plot3DScaleBadge
          missingSupport={nonFiniteSourceCount > 0}
          grid={renderGrid}
          zExag={sceneScale.z}
          fidelityMode={renderFidelityMode}
          samplingProvenance={brickedContourEligible
            ? null
            : preparation.metadata?.samplingProvenance}
          detail={(brickedContourEligible && brickedContourExtraction.plan
            ? `progressive native bricks · L${brickedContourExtraction.plan.level} · ${brickedContourExtraction.plan.items.length} selected · ${brickedContourExtraction.plan.culledBrickCount} extrema-culled`
            : ((pairedContours
            ? "signed ± resolved surfaces"
            : positiveQSurface
            ? fieldName === SWIRLING_STRENGTH_FIELD
              ? "3D swirling-strength surface · λci > 0 · worker-derived from u/v/w"
              : "full 3D Q-criterion surface · Q > 0 · worker-derived from u/v/w"
            : rotationDominatedSurface
            ? fieldName === LAMBDA2_FIELD
              ? "3D lambda-2 vortex surface · λ2 < 0 · worker-derived from u/v/w"
              : "rotation-dominated Okubo-Weiss surface · W < 0"
            : "")
            + (volumeWindowOutcome.ok && volumeWindowOutcome.mode === "native-detail"
              ? ` · extreme source scan blocked · ${stormCenterX != null
                ? "tracker-centred" : "metadata-centred"} native ROI`
              : "")
            + (outputHierarchySource.activeLevel?.depth
              ? ` · ${outputHierarchySource.selection?.disclosure ?? outputHierarchySource.activeLevel.levelId}`
              : outputHierarchySource.selection?.fallbackReason
                ? ` · ${outputHierarchySource.selection.fallbackReason}`
                : ""))) || null}
        />
        <Plot3DOverlayStack>
        {pairedContours && fieldName && committedThreshold != null
          && (primarySurfaceVisible || negativeSurfaceVisible) && (
          <Plot3DDiscreteLegend
            title={`${fieldName} signed structure`}
            items={[
              primarySurfaceVisible ? {
                color: sceneRgbToHex(contrastAdjustedColor(
                  hexToRgb(SIGNED_POSITIVE_COLOR),
                  sceneAppearance.resolved.background,
                )),
                label: withFieldUnit(`≥ +${formatThreshold(Math.abs(committedThreshold))}`),
              } : null,
              negativeSurfaceVisible ? {
                color: sceneRgbToHex(contrastAdjustedColor(
                  hexToRgb(SIGNED_NEGATIVE_COLOR),
                  sceneAppearance.resolved.background,
                )),
                label: withFieldUnit(`≤ −${formatThreshold(Math.abs(committedThreshold))}`),
              } : null,
            ].filter((item): item is { color: string; label: string } => item != null)}
          />
        )}
        {rotationDominatedSurface && committedThreshold != null && primarySurfaceVisible && (
          <Plot3DDiscreteLegend
            title={fieldName === LAMBDA2_FIELD ? "Resolved 3D lambda-2 vortex core" : "Resolved horizontal rotation core"}
            items={[{
              color: sceneRgbToHex(contrastAdjustedColor(
                hexToRgb(color),
                sceneAppearance.resolved.background,
              )),
              label: fieldName === LAMBDA2_FIELD
                ? `λ2 ≤ ${formatThreshold(committedThreshold)} s⁻²`
                : `W ≤ ${formatThreshold(committedThreshold)} s⁻²`,
            }]}
          />
        )}
        {positiveQSurface && committedThreshold != null && primarySurfaceVisible && (
          <Plot3DDiscreteLegend
            title={fieldName === SWIRLING_STRENGTH_FIELD ? "Resolved 3D swirling core" : "Resolved 3D rotation-dominated core"}
            items={[{
              color: sceneRgbToHex(contrastAdjustedColor(
                hexToRgb(color),
                sceneAppearance.resolved.background,
              )),
              label: fieldName === SWIRLING_STRENGTH_FIELD
                ? `λci ≥ ${formatThreshold(committedThreshold)} s⁻¹`
                : `Q ≥ ${formatThreshold(committedThreshold)} s⁻²`,
            }]}
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
              selected surface renders successfully.
            </div>
          </div>
        )}
        {boundedLoadError && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Isosurface loading is unavailable within the bounded-read policy.
            <div className="volume-render__overlay-hint">
              {boundedLoadError} Nimbus will not fall back to reading the full source volume.
            </div>
          </div>
        )}
        {qReductionBlocked && !terrainBlocked && (
          <div className="volume-render__overlay volume-render__overlay--error">
            {fieldName ?? "Velocity diagnostic"} needs resolved velocity gradients at this domain size.
            <div className="volume-render__overlay-hint">
              The pane budget would require velocity LOD. Nimbus refuses to compute Q from block-mean u/v/w because derive-after-reduce changes the diagnostic.
            </div>
          </div>
        )}
        {(fieldResult.error || preparation.error || contourError) && (
          <div className="volume-render__overlay volume-render__overlay--error">
            Field failed: {fieldResult.error || preparation.error || contourError}
          </div>
        )}
        {!fieldLoadBlocked && !hasField && !fieldResult.loading && !preparation.loading && !fieldResult.error && !preparation.error && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            Pick a 3D scalar field above.
          </div>
        )}
        {declinedLink && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            {describeDeclinedFieldLink(declinedLink, SINGLE_LEVEL_LINK_REASON)}
          </div>
        )}
        {uniform && !fieldResult.loading && !preparation.loading && !fieldResult.error && !preparation.error && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            {nonFiniteSourceCount > 0
              ? "No isosurface crosses the finite data at this threshold."
              : "Field is uniform (min = max) — no isosurface to extract."}
          </div>
        )}
        {thresholdOutOfRange && !fieldResult.loading && !preparation.loading && !fieldResult.error && !preparation.error && (
          <div className="volume-render__overlay volume-render__overlay--muted">
            Threshold {withFieldUnit(committedThreshold?.toPrecision(3) ?? "—")} is outside this frame&rsquo;s range
            {fieldStats && ` ${withFieldUnit(`[${fieldStats.min.toPrecision(3)}, ${fieldStats.max.toPrecision(3)}]`)}`}
            {" "}— no surface to extract here.
            <div className="volume-render__overlay-hint">
              The threshold is held across frames so a scrub stays comparable.
              Lower it, or scrub to a frame that reaches it.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- helpers ---------- */

function applyIsosurfaceLighting(
  property: Parameters<typeof applySurfaceLighting>[0],
  preset: Parameters<typeof applySurfaceLighting>[1],
  lightBackground: boolean,
): void {
  applySurfaceLighting(property, preset, "translucent");
  if (!lightBackground || lightingSettings(preset).preset === "flat") return;
  // On a light canvas, back-facing translucent tubes were driven almost to
  // black by diffuse-only falloff. Preserve shape from the key light while
  // keeping enough ambient data colour to distinguish the signed pair.
  const material = shadowMaterial(0.72, 0.46, preset);
  property.setAmbient(material.ambient);
  property.setDiffuse(material.diffuse);
  property.setSpecular(0);
}

function applySurfaceGeometry(
  target: ReturnType<typeof vtkPolyData.newInstance>,
  geometry: VortexSurfaceGeometry | null,
): void {
  applyImmutablePolyDataGeometry(target, geometry);
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) return [1, 1, 1];
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  return [r, g, b];
}

/**
 * Format the threshold for the readout. Small magnitudes (qc ~ 1e-4) get
 * scientific notation; larger values get fixed precision. Avoids
 * showing "0.00" for cloud-water thresholds.
 */
function formatThreshold(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs < 0.01 || abs >= 1e4) return v.toExponential(2);
  return v.toFixed(abs < 1 ? 3 : 2);
}
