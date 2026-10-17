import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { slice2D, sliceDisplayAxes, type SliceAxis, type Npy3D, type NpyN } from "@/lib/platform/npy";
import { planeFocusFromView } from "@/lib/plot/volumetricHandoff";
import {
  axisCoords,
  axisLabel,
  axisOptions,
  hasPhysicalAxisCoordinates,
  heightKmToZIndex,
  planeSupportsLockedAspect,
  scalarCoordinateKm,
} from "@/lib/science/gridCoords";
import { fieldDefinitionIdentity, registeredFieldRank, type GridInfo } from "@/lib/run/runOutputs";
import { findPublishedFrame, loadedFramesAreCoherent } from "@/lib/run/frameIdentity";
import { labelWithUnit, unitFor, type FieldUnits } from "@/lib/science/fieldUnits";
import { cropToValidPlotDomain, validAxisLength } from "@/lib/science/validDomain";
import {
  DBZ_RANGE,
  usesFixedAnchorRange,
  isDbzField,
  isDivergingColorscale,
  padDegenerateRange,
  resolveColorscale,
  type ExtendedColorscale,
} from "@/lib/plot/plotColormaps";
import { describeContourSkips, type ContourSkip } from "@/lib/plot/plotLabels";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  ControlGroup,
  FIT_OPTIONS,
  fitAxisFragments,
  PillGroup,
  type PlotFitMode,
  type PlotChrome,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useFieldAny,
  useLatchedRange,
  usePlaneViewport,
  useRetainedPlot,
  useFieldLink,
  usePlotChrome,
  useScalarSelectionReport,
  scalarColorRangeAnnotation,
  VolumeHandoffButton,
  type PlotComponentProps,
} from "./common";
import { displayedRange, displayedRangeScope } from "@/lib/plot/displayedRange";
import { defaultScalarDisplay } from "@/lib/plot/scalarDisplayDefaults";
import {
  fieldSupportsBoundedWindows,
  fieldWindowAggregation,
  PLOTLY_HEATMAP_CELL_CAP,
  planBoundedPlaneWindows,
  planBoundedSliceWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { loadedSliceWindowIndex, sourceSliceControlsAvailable } from "@/lib/plot/sliceWindowSelection";
import { remapTerrainVerticalSlice } from "@/lib/science/terrainSliceRemap";
import { parseContourLevels } from "@/lib/plot/contourLevels";
import { SlicePlaneContext } from "./SlicePlaneContext";
import { SliceIndexInput } from "./SliceIndexInput";

/* Filled heatmap + up to two contour overlays on shared axes.
 * Each layer sliced independently; layers self-degrade when a field is absent. */

const FILL_COLORMAPS: ReadonlyArray<{ value: ExtendedColorscale; label: string }> = [
  { value: "Viridis", label: "Viridis" },
  { value: "Plasma", label: "Plasma" },
  { value: "Inferno", label: "Inferno" },
  { value: "RdBu", label: "RdBu" },
  { value: "Greys", label: "Greys" },
  { value: "dBZ", label: "NWS dBZ" },
  { value: "ColdPool", label: "Cold pool (BuRd)" },
];

type FillRangeMode = "auto" | "symmetric" | "dBZ";
const FILL_RANGE_MODES: ReadonlyArray<{ value: FillRangeMode; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "symmetric", label: "± auto" },
  { value: "dBZ", label: "dBZ" },
];

const CONTOUR_COLORS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "#0b0e14", label: "Black" },
  { value: "#ffffff", label: "White" },
  { value: "#1e6feb", label: "Blue" },
  { value: "#e23b3b", label: "Red" },
  { value: "#19b562", label: "Green" },
  { value: "#d000d0", label: "Magenta" },
];

type DragMode = "pan" | "zoom";
const DRAG_OPTIONS: ReadonlyArray<{ value: DragMode; label: string }> = [
  { value: "pan", label: "Pan" },
  { value: "zoom", label: "Box zoom" },
];

interface ContourLayer {
  enabled: boolean;
  field: string | null;
  /** Levels: comma list ("40" or "0.01,0.02") or "start:end:step". */
  levels: string;
  color: string;
}

/** First field in `available` matching any name in `prefer` (case-insensitive). */
function pickField(available: string[], prefer: string[]): string | null {
  const lower = available.map((f) => f.toLowerCase());
  for (const p of prefer) {
    const i = lower.indexOf(p.toLowerCase());
    if (i >= 0) return available[i];
  }
  return null;
}

/** Sensible fill colormap + range for a field. dBZ routing is driven by the
 *  field's unit (so the linear `radar` field is NOT painted on the dBZ scale). */

/** Sensible default contour levels string for a field. The 40-dBZ default
 *  applies only to true dBZ fields, not the linear `radar` field. */
function defaultLevels(field: string | null, fieldUnits: FieldUnits): string {
  if (!field) return "";
  if (isDbzField(field, fieldUnits)) return "40";
  if (field.toLowerCase().includes("vorticity")) return "0.01,0.02";
  return "";
}

export function CompositePlot({
  runDir: rootRunDir,
  steps: rootSteps,
  stepIdx: rootStepIdx,
  preset,
  onPresetCleared,
  defaultSliceAxis,
  onFrameDisplayChange,
  onScalarSelectionChange,
  volumeHandoff,
  linkedField,
  onFieldChange,
}: PlotComponentProps) {
  const chrome = usePlotChrome();
  const rootCurrentStep = rootSteps[rootStepIdx] ?? null;
  const rootFields = useMemo(() => rootCurrentStep?.fields ?? [], [rootCurrentStep?.fields]);

  const [sliceAxis, setSliceAxis] = useState<SliceAxis>(
    preset?.sliceAxis ?? defaultSliceAxis ?? 2,
  );
  const [sliceIndex, setSliceIndex] = useState(0);
  const [displayedSliceIndex, setDisplayedSliceIndex] = useState(0);
  const appliedSlicePresetRef = useRef<string | null>(null);
  const sliceTouchedRef = useRef(false);

  const [fillField, setFillField] = useState<string | null>(
    preset?.field ?? pickField(rootFields, ["w"]) ?? rootFields[0] ?? null,
  );
  const fillDefaults = defaultScalarDisplay(preset?.field ?? fillField, rootCurrentStep?.fieldUnits);
  const [fillColormap, setFillColormap] = useState<ExtendedColorscale>(
    (preset?.colormap as ExtendedColorscale | undefined) ?? fillDefaults.colormap,
  );
  const [fillRange, setFillRange] = useState<FillRangeMode>(
    (preset?.rangeMode as FillRangeMode | undefined) ?? fillDefaults.rangeMode,
  );

  const [contourA, setContourA] = useState<ContourLayer>(() => {
    const f = pickField(rootFields, ["reflectivity_dbz", "composite_reflectivity", "radar"]);
    return { enabled: f != null, field: f, levels: defaultLevels(f, rootCurrentStep?.fieldUnits), color: "#0b0e14" };
  });
  const [contourB, setContourB] = useState<ContourLayer>(() => {
    const f = pickField(rootFields, ["vorticity_z", "vorticity_magnitude"]);
    return { enabled: f != null, field: f, levels: defaultLevels(f, rootCurrentStep?.fieldUnits), color: "#d000d0" };
  });
  const hierarchyRequiredFields = useMemo(
    () => [
      fillField,
      contourA.enabled ? contourA.field : null,
      contourB.enabled ? contourB.field : null,
    ].filter((field): field is string => field != null),
    [contourA.enabled, contourA.field, contourB.enabled, contourB.field, fillField],
  );
  const outputHierarchySource = useOutputHierarchySource({
    runDir: rootRunDir,
    steps: rootSteps,
    stepIdx: rootStepIdx,
    scope: "shared",
    requiredFields: hierarchyRequiredFields,
  });
  const runDir = outputHierarchySource.sourceRunDir;
  const steps = outputHierarchySource.sourceSteps;
  const currentStep = outputHierarchySource.currentStep;
  const renderableFields = useMemo(
    () => fieldsForPlotRanks(runDir, currentStep, defaultSliceAxis === 1 ? [3] : [2, 3]),
    [currentStep, defaultSliceAxis, runDir],
  );
  const grid = currentStep?.grid ?? null;

  const [dragMode, setDragMode] = useState<DragMode>("zoom");
  // See PlotFitMode: true shape by default, Fill opt-in per plot.
  const [fitMode, setFitMode] = useState<PlotFitMode>("true-shape");
  const [resetTick, setResetTick] = useState(0);
  const [colorResetTick, setColorResetTick] = useState(0);
  const planeViewIdentity = JSON.stringify([runDir, fillField, sliceAxis]);
  const { viewport, onRelayout, resetViewport } = usePlaneViewport(planeViewIdentity);

  const clearPreset = () => onPresetCleared?.();

  /* Colour follows the quantity, for the picker and for a linked field alike. Only the fill
   * layer participates: the contour overlays are the second and third variables of this plot,
   * and a link that overwrote them would discard the comparison the user built. */
  const applyFillField = useCallback((next: string | null) => {
    setFillField(next);
    const display = defaultScalarDisplay(next, currentStep?.fieldUnits);
    setFillColormap(display.colormap);
    setFillRange(display.rangeMode);
  }, [currentStep?.fieldUnits]);

  useFieldLink(linkedField, fillField, (f) => renderableFields.includes(f), applyFillField);

  // Apply a preset (fill field/colormap/range + axis) when its id changes.
  useEffect(() => {
    if (!preset) return;
    sliceTouchedRef.current = false;
    appliedSlicePresetRef.current = null;
    if (preset.field) setFillField(preset.field);
    if (preset.colormap) setFillColormap(preset.colormap as ExtendedColorscale);
    if (preset.rangeMode && preset.rangeMode !== "fixed") {
      setFillRange(preset.rangeMode as FillRangeMode);
    }
    if (preset.sliceAxis != null) setSliceAxis(preset.sliceAxis);
  }, [preset]);

  // Reset to the category's default slice orientation on category change.
  useEffect(() => {
    if (defaultSliceAxis == null) return;
    setSliceAxis(defaultSliceAxis);
  }, [defaultSliceAxis]);

  // Repick layer fields when the current one is absent from the step.
  useEffect(() => {
    if (renderableFields.length === 0) return;
    if (!fillField || !renderableFields.includes(fillField)) {
      applyFillField(pickField(renderableFields, ["w"]) ?? renderableFields[0] ?? null);
    }
    setContourA((prev) =>
      prev.field && renderableFields.includes(prev.field) ? prev : { ...prev, enabled: false },
    );
    setContourB((prev) =>
      prev.field && renderableFields.includes(prev.field) ? prev : { ...prev, enabled: false },
    );
  }, [renderableFields, fillField, applyFillField]);

  // Debounce the bounded plane reads so a scrub/autoplay does not request every
  // intermediate frame. Bypassed while recording.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const planLayer = (field: string | null) => {
    if (!currentStep || !field) return { windows: null, outcome: null, error: null };
    if (!fieldSupportsBoundedWindows(currentStep.fieldOrigins?.[field])) {
      return {
        windows: null,
        outcome: null,
        error: `${field}: this provider cannot serve bounded plot windows.`,
      };
    }
    const rank = registeredFieldRank(
      runDir, currentStep.index, field, currentStep.publicationGeneration,
    );
    const outcome = rank === 3
      ? planBoundedSliceWindows(
          grid, sliceAxis, sliceIndex, PLOTLY_HEATMAP_CELL_CAP, viewport,
        )
      : rank === 2
        ? planBoundedPlaneWindows(grid, PLOTLY_HEATMAP_CELL_CAP, viewport)
        : null;
    if (!outcome) {
      return { windows: null, outcome: null, error: `${field}: expected a rank-2 or rank-3 field.` };
    }
    if (!outcome.ok) return { windows: null, outcome, error: `${field}: ${outcome.reason}` };
    return {
      windows: windowsForFieldAggregation(outcome.plan.windows, field),
      outcome,
      error: null,
    };
  };
  const fillPlan = planLayer(fillField);
  const contourAPlan = planLayer(contourA.enabled ? contourA.field : null);
  const contourBPlan = planLayer(contourB.enabled ? contourB.field : null);
  const sampledLayerDescriptions = [
    [fillField, fillPlan] as const,
    [contourA.enabled ? contourA.field : null, contourAPlan] as const,
    [contourB.enabled ? contourB.field : null, contourBPlan] as const,
  ].flatMap(([field, plan]) => field && plan.outcome?.ok
      && plan.outcome.plan.stride.some((stride) => stride > 1)
    ? [`${field}: ${fieldWindowAggregation(field) === "linear_reflectivity_mean" ? "linear-Z" : "cell"} mean ${plan.outcome.plan.stride.join("×")}`]
    : []);
  const fill = useFieldAny(
    runDir, fillPlan.windows ? loadFrame.stepIndex : null, fillField,
    loadFrame.publicationGeneration, fillPlan.windows,
    JSON.stringify([runDir, fillField, sliceAxis]),
  );
  const caLoad = useFieldAny(
    runDir,
    contourAPlan.windows ? loadFrame.stepIndex : null,
    contourA.enabled ? contourA.field : null,
    loadFrame.publicationGeneration,
    contourAPlan.windows,
    JSON.stringify([runDir, contourA.field, sliceAxis]),
  );
  const cbLoad = useFieldAny(
    runDir,
    contourBPlan.windows ? loadFrame.stepIndex : null,
    contourB.enabled ? contourB.field : null,
    loadFrame.publicationGeneration,
    contourBPlan.windows,
    JSON.stringify([runDir, contourB.field, sliceAxis]),
  );
  const activeLoads = [
    fill,
    ...(contourA.enabled && contourA.field && contourAPlan.windows ? [caLoad] : []),
    ...(contourB.enabled && contourB.field && contourBPlan.windows ? [cbLoad] : []),
  ];
  const coherentFrame = activeLoads.every(load => !load.loading && load.field != null && !load.error)
    && loadedFramesAreCoherent(...activeLoads);
  const loadedStep = coherentFrame
    ? findPublishedFrame(
        steps,
        fill.loadedStepIndex,
        fill.loadedPublicationGeneration,
      )
    : null;
  const fillDomain = useMemo(
    () => coherentFrame && fill.field
      ? cropToValidPlotDomain(fill.field, loadedStep?.grid)
      : null,
    [coherentFrame, fill.field, loadedStep?.grid],
  );
  const contourADomain = useMemo(
    () => coherentFrame && caLoad.field
      ? cropToValidPlotDomain(caLoad.field, loadedStep?.grid)
      : null,
    [coherentFrame, caLoad.field, loadedStep?.grid],
  );
  const contourBDomain = useMemo(
    () => coherentFrame && cbLoad.field
      ? cropToValidPlotDomain(cbLoad.field, loadedStep?.grid)
      : null,
    [coherentFrame, cbLoad.field, loadedStep?.grid],
  );
  const fillPhysical = useMemo(() => remapLayer(fillDomain, sliceAxis), [fillDomain, sliceAxis]);
  const contourAPhysical = useMemo(() => remapLayer(contourADomain, sliceAxis), [contourADomain, sliceAxis]);
  const contourBPhysical = useMemo(() => remapLayer(contourBDomain, sliceAxis), [contourBDomain, sliceAxis]);
  const displayGrid = fillPhysical.grid;
  const displayFieldUnits = loadedStep?.fieldUnits;

  const fillSourceRank = currentStep && fillField
    ? registeredFieldRank(
        runDir, currentStep.index, fillField, currentStep.publicationGeneration,
      )
    : null;
  const fillRank = fillDomain?.field.shape.length ?? fillSourceRank;
  const fillIs3D = fillRank === 3;
  const sourceSliceDim = validAxisLength(sliceAxis, grid)
    ?? ([grid?.nr, grid?.nth, grid?.nz][sliceAxis] ?? 0);
  useEffect(() => {
    if (fillSourceRank !== 3 || !Number.isSafeInteger(sourceSliceDim) || sourceSliceDim < 1) return;
    setSliceIndex(previous => Math.max(0, Math.min(previous, sourceSliceDim - 1)));
  }, [fillSourceRank, sourceSliceDim]);
  const sliceControlsAvailable = sourceSliceControlsAvailable(fillSourceRank, sourceSliceDim, fillPlan.error);
  const loadedSliceIndex = loadedSliceWindowIndex(fill.field, sliceAxis, loadedStep?.grid?.validIndexRange?.lower,
    fill.loading ? displayedSliceIndex : sliceIndex);

  // Resolve the preset from source dimensions before requesting its plane.
  useEffect(() => {
    if (!preset) {
      appliedSlicePresetRef.current = null;
      return;
    }
    if (fillSourceRank !== 3 || !Number.isSafeInteger(sourceSliceDim) || sourceSliceDim < 1 || sliceTouchedRef.current) return;
    if (preset.sliceAxis != null && sliceAxis !== preset.sliceAxis) return;
    const dim = sourceSliceDim;
    if (appliedSlicePresetRef.current === preset.id) return;
    appliedSlicePresetRef.current = preset.id;
    if (preset?.sliceHeightKm != null && sliceAxis === 2
        && grid?.nz && hasPhysicalAxisCoordinates(2, grid.nz, grid)) {
      const storageIndex = heightKmToZIndex(preset.sliceHeightKm, grid, grid.nz);
      setSliceIndex(Math.max(0, Math.min(
        dim - 1,
        storageIndex - (grid.validIndexRange?.lower[2] ?? 0),
      )));
      return;
    }
    const hint = preset?.sliceIndexHint;
    if (hint == null) return;
    if (hint === "surface") setSliceIndex(0);
    else if (hint === "mid") setSliceIndex(Math.floor((dim - 1) / 2));
    else if (hint === "top") setSliceIndex(dim - 1);
    else if (typeof hint === "number") setSliceIndex(Math.min(hint, dim - 1));
  }, [fillSourceRank, sourceSliceDim, sliceAxis, preset, grid]);

  /*
   * Fitted to the plane on screen, then latched across frames (see SlicePlot): a per-frame
   * re-fit makes the same physical value paint a different colour each step, so the animation
   * flickers and no two frames can be compared. Fitting to the whole volume instead washed the
   * plane out whenever the column's extremes sat at other heights.
   */
  const fillPlane = useMemo(() => fillPhysical.field
    ? layerTo2D(fillPhysical.field, sliceAxis, loadedSliceIndex) : null, [fillPhysical.field, sliceAxis, loadedSliceIndex]);
  const contourAPlane = useMemo(() => contourAPhysical.field
    ? prepareContourPlane(contourAPhysical.field, sliceAxis, loadedSliceIndex) : null, [contourAPhysical.field, sliceAxis, loadedSliceIndex]);
  const contourBPlane = useMemo(() => contourBPhysical.field
    ? prepareContourPlane(contourBPhysical.field, sliceAxis, loadedSliceIndex) : null, [contourBPhysical.field, sliceAxis, loadedSliceIndex]);
  const frameRange = useMemo(() => displayedRange(fillPlane?.data), [fillPlane]);
  const range = useLatchedRange(
    `${displayedRangeScope(
      runDir, fillField, fillIs3D ? sliceAxis : null, fillIs3D ? loadedSliceIndex : null,
    )}|colors:${colorResetTick}`,
    frameRange,
  );

  const zCoordinateKm =
    fillIs3D && sliceAxis === 2 && sliceControlsAvailable
      && grid?.nz && hasPhysicalAxisCoordinates(2, grid.nz, grid)
      ? scalarCoordinateKm(
          2,
          sliceIndex + (grid.validIndexRange?.lower[2] ?? 0),
          grid.nz,
          grid,
        )
      : null;
  const zHeightKm = zCoordinateKm == null ? null : Number(zCoordinateKm.toFixed(2));
  const planeFocus = useMemo(() => {
    if (!fillIs3D || !fillDomain?.field) return null;
    const sourceCounts = [grid?.nr, grid?.nth, grid?.nz];
    const cellCount = sourceCounts[sliceAxis] ?? fillDomain.field.shape[sliceAxis];
    const sourceIndex = sliceIndex + (grid?.validIndexRange?.lower[sliceAxis] ?? 0);
    const coordinateKm = scalarCoordinateKm(sliceAxis, sourceIndex, cellCount, grid);
    const { rowAxis, colAxis } = sliceDisplayAxes(sliceAxis);
    const physicalViewport = grid?.coordinateSystem == null || grid.coordinateSystem === "cartesian"
      ? viewport
      : null;
    return planeFocusFromView(
      sliceAxis,
      sourceIndex,
      coordinateKm == null ? null : coordinateKm * 1_000,
      rowAxis,
      colAxis,
      physicalViewport,
    );
  }, [fillDomain, fillIs3D, grid, sliceAxis, sliceIndex, viewport]);
  useScalarSelectionReport(
    onScalarSelectionChange,
    fillField,
    fillColormap,
    fillRange,
    fillRank,
    "composite",
    planeFocus,
  );

  const plot = useMemo(
    () =>
      buildComposite({
        fill: fillPlane,
        fillRank,
        fillName: fillField,
        fillColormap,
        fillRange,
        range,
        frameRange,
        contourA: { ...contourA, plane: contourAPlane, grid: contourAPhysical.grid },
        contourB: { ...contourB, plane: contourBPlane, grid: contourBPhysical.grid },
        sliceAxis,
        grid: displayGrid,
        fieldUnits: displayFieldUnits,
        dragMode,
        fitMode,
        chrome,
      }),
    [
      fillPlane, fillRank, fillField, fillColormap, fillRange, range, frameRange,
      contourA, contourAPlane, contourAPhysical.grid, contourB, contourBPlane, contourBPhysical.grid,
      sliceAxis, displayGrid, displayFieldUnits, dragMode, fitMode, chrome,
    ],
  );

  const figureScope = JSON.stringify([
      runDir,
      fillField ?? "",
      contourA.enabled ? contourA.field ?? "" : "",
      contourB.enabled ? contourB.field ?? "" : "",
      sliceAxis,
      [fillField, contourA.enabled ? contourA.field : null, contourB.enabled ? contourB.field : null]
        .map(field => field && currentStep ? [fieldDefinitionIdentity(runDir, currentStep.index, field,
          currentStep.publicationGeneration), currentStep.fieldUnits?.[field]] : null),
    ]);
  const layerError = [fillPlan.error, contourAPlan.error, contourBPlan.error,
    ...activeLoads.map(load => load.error), fillPhysical.error,
    contourA.enabled ? contourAPhysical.error ?? parseContourLevels(contourA.levels).error : null,
    contourB.enabled ? contourBPhysical.error ?? parseContourLevels(contourB.levels).error : null].find(Boolean);
  const samplingDescription = sampledLayerDescriptions.join(" · ");
  const candidate = useMemo(() => plot && loadedStep && !layerError ? { plot, step: loadedStep,
    sliceIndex: loadedSliceIndex, samplingDescription,
    terrainRemapped: fillPhysical.remapped || contourAPhysical.remapped || contourBPhysical.remapped,
  } : null, [plot, loadedStep, loadedSliceIndex, samplingDescription, layerError,
    fillPhysical.remapped, contourAPhysical.remapped, contourBPhysical.remapped]);
  const { shown } = useRetainedPlot(candidate, figureScope);
  const shownPlot = shown?.plot;
  const [rendered, setRendered] = useState<{ scope: string; figure: NonNullable<typeof candidate> } | null>(null);
  const committed = rendered?.scope === figureScope ? rendered.figure : null;
  const pending = activeLoads.some(load => load.loading) || candidate == null || committed?.plot !== candidate.plot;
  useEffect(() => {
    onFrameDisplayChange?.({ requestedStepIndex: currentStep?.index ?? null,
      requestedPublicationGeneration: currentStep?.publicationGeneration ?? null,
      renderedStepIndex: layerError ? null : committed?.step.index ?? null,
      renderedPublicationGeneration: layerError ? null : committed?.step.publicationGeneration ?? null,
      loading: !layerError && (pending || committed?.step.index !== currentStep?.index
        || committed?.step.publicationGeneration !== currentStep?.publicationGeneration) });
  }, [onFrameDisplayChange, currentStep?.index, currentStep?.publicationGeneration, committed, layerError, pending]);
  const dim = fillIs3D ? sourceSliceDim : 0;

  return (
    <>
      <div className="plot-controls plot-controls--composite">
        <div className="composite-controls__row composite-controls__row--primary">
        <OutputHierarchyControls source={outputHierarchySource} />
        <ControlGroup label="Fill field">
          <select
            aria-label="Fill field"
            value={fillField ?? ""}
            onChange={(e) => {
              const f = e.target.value || null;
              applyFillField(f);
              onFieldChange?.(f);
              clearPreset();
            }}
          >
            {renderableFields.length === 0 && (
              <option value="">{defaultSliceAxis === 1 ? "(no rank-3 fields)" : "(no rank-2/3 fields)"}</option>
            )}
            {renderableFields.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </ControlGroup>

        <PlotCompressionControls
          step={loadedStep}
          fields={[
            fillField,
            contourA.enabled ? contourA.field : null,
            contourB.enabled ? contourB.field : null,
          ]}
        />

        <ControlGroup label="Fill colormap">
          <select
            aria-label="Fill colormap"
            value={fillColormap}
            onChange={(e) => {
              setFillColormap(e.target.value as ExtendedColorscale);
              clearPreset();
            }}
          >
            {FILL_COLORMAPS.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </ControlGroup>

        <ControlGroup label="Fill range">
          <PillGroup
            options={FILL_RANGE_MODES}
            value={fillRange}
            onChange={(v) => {
              setFillRange(v);
              clearPreset();
            }}
          />
        </ControlGroup>

        {fillIs3D && (
          <>
            <ControlGroup label="Slice axis">
              <PillGroup
                options={axisOptions(grid)}
                value={sliceAxis}
                onChange={(v) => {
                  sliceTouchedRef.current = true;
                  setSliceAxis(v);
                  clearPreset();
                }}
              />
            </ControlGroup>
            <ControlGroup
              label={`Index (${axisLabel(sliceAxis, grid)} = ${sliceIndex})`}
              stretch
              className="composite-index-control"
            >
              <input
                type="range"
                aria-label={`Selected ${axisLabel(sliceAxis, grid)} index`}
                min={0}
                max={dim > 0 ? dim - 1 : 0}
                value={sliceIndex}
                onChange={(e) => {
                  sliceTouchedRef.current = true;
                  setSliceIndex(Number(e.target.value));
                  clearPreset();
                }}
                disabled={!sliceControlsAvailable}
              />
              <SliceIndexInput value={sliceIndex} max={Math.max(0, dim - 1)} disabled={!sliceControlsAvailable}
                label={`Go to ${axisLabel(sliceAxis, grid)} index`}
                onCommit={index => { sliceTouchedRef.current = true; setSliceIndex(index); clearPreset(); }} />
            </ControlGroup>
            <div className="composite-coordinate-slot">
              {sliceAxis === 2 && sliceControlsAvailable && grid?.nz
                && hasPhysicalAxisCoordinates(2, grid.nz, grid) && (
              <ControlGroup
                label={`Height${zHeightKm != null ? ` (${zHeightKm} km)` : " (km)"}`}
                className="composite-height-control"
              >
                <input
                  type="number"
                  min={0}
                  step={Number(axisCoords(2, grid.nz, grid).spacing.toFixed(3))}
                  value={zHeightKm ?? 0}
                  onChange={(e) => {
                    const km = Number(e.target.value);
                    if (Number.isFinite(km) && sliceControlsAvailable) {
                      sliceTouchedRef.current = true;
                      setSliceIndex(
                        Math.max(0, Math.min(
                          dim - 1,
                          heightKmToZIndex(km, grid, grid.nz!)
                            - (grid.validIndexRange?.lower[2] ?? 0),
                        )),
                      );
                      clearPreset();
                    }
                  }}
                  title="Jump to a height in km AGL (snaps to the nearest model level)"
                />
              </ControlGroup>
              )}
            </div>
          </>
        )}
        </div>

        <div className="composite-controls__row composite-controls__row--secondary">
        <ContourControls
          label="Contour A"
          layer={contourA}
          fields={renderableFields}
          fieldUnits={currentStep?.fieldUnits}
          warning={shownPlot?.contourWarnings.find(
            (warning) => warning.field === contourA.field,
          )?.text ?? null}
          onChange={(next) => {
            setContourA(next);
            clearPreset();
          }}
        />
        <ContourControls
          label="Contour B"
          layer={contourB}
          fields={renderableFields}
          fieldUnits={currentStep?.fieldUnits}
          warning={shownPlot?.contourWarnings.find(
            (warning) => warning.field === contourB.field,
          )?.text ?? null}
          onChange={(next) => {
            setContourB(next);
            clearPreset();
          }}
        />

        {shownPlot?.aspectLockable && (
          <ControlGroup label="Fit">
            <PillGroup options={FIT_OPTIONS} value={fitMode} onChange={setFitMode} />
          </ControlGroup>
        )}

        <ControlGroup label="Drag">
          <PillGroup options={DRAG_OPTIONS} value={dragMode} onChange={setDragMode} />
        </ControlGroup>

        <div className="composite-controls__actions">
          <button
            type="button"
            className="plot-reset-btn"
            onClick={() => {
              resetViewport();
              setResetTick((t) => t + 1);
            }}
            title="Reset pan/zoom to fit"
          >
            Reset view
          </button>

          <button
            type="button"
            className="plot-reset-btn"
            disabled={!frameRange}
            onClick={() => {
              applyFillField(fillField);
              setColorResetTick((tick) => tick + 1);
              clearPreset();
            }}
            title="Restore the field's palette and fit colors to this slice; reflectivity keeps its fixed dBZ scale"
          >
            Reset colors
          </button>

          <VolumeHandoffButton offer={volumeHandoff} />
        </div>
        </div>
      </div>

      <div className="plot-body">
        {fillIs3D && <SlicePlaneContext grid={committed?.step.grid} axis={sliceAxis}
          index={committed?.sliceIndex ?? null} physicalHeight={committed?.terrainRemapped} />}
        {pending && !layerError && (
          <div className="plot-overlay" role="status">
            <Icon name="dot" size={12} />
            <span>
              {committed && fillIs3D
                ? `Displaying index ${displayedSliceIndex} · loading selected ${axisLabel(sliceAxis, grid)} index ${sliceIndex}…`
                : "Loading composite layers…"}
            </span>
          </div>
        )}
        {layerError && (
          <div className="plot-overlay plot-overlay--error" role="alert">
            <Icon name="warning" size={12} />
            <span>{layerError}{committed ? " · Previous completed composite retained." : ""}</span>
          </div>
        )}
        {committed?.samplingDescription ? (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>
              bounded · {committed.samplingDescription}
            </span>
          </div>
        ) : null}
        {committed?.terrainRemapped && <div className="plot-overlay">Physical-volume terrain → Cartesian vertical remap</div>}
        {viewport && fillPlan.outcome?.ok && !pending && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>
              viewport {fillPlan.outcome.plan.stride.every((stride) => stride === 1)
                ? "native cells"
                : `${fieldWindowAggregation(fillField) === "linear_reflectivity_mean" ? "linear-Z" : "cell"} mean ${fillPlan.outcome.plan.stride.join("×")}`}
              {fillPlan.outcome.plan.fullSourceShape
                ? ` · full plane ${fillPlan.outcome.plan.fullSourceShape.join("×")}`
                : ""}
            </span>
          </div>
        )}
        {shownPlot ? (
          <PlotCanvas
            key={resetTick}
            viewIdentity={planeViewIdentity}
            data={shownPlot.data}
            layout={shownPlot.layout}
            onRelayout={onRelayout}
            onRendered={() => {
              if (!shown) return;
              setRendered(previous => previous?.scope === figureScope && previous.figure === shown
                ? previous : { scope: figureScope, figure: shown });
              setDisplayedSliceIndex(shown.sliceIndex);
            }}
          />
        ) : (
          !fill.loading &&
          !fill.error &&
          !fillPlan.error && (
            <div className="plot-overlay">
              <span>Pick a fill field to build the composite.</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

/* ---------- contour layer controls ---------- */

interface ContourControlsProps {
  label: string;
  layer: ContourLayer;
  fields: string[];
  fieldUnits: FieldUnits;
  warning: string | null;
  onChange: (next: ContourLayer) => void;
}

function ContourControls({ label, layer, fields, fieldUnits, warning, onChange }: ContourControlsProps) {
  const levelError = layer.enabled ? parseContourLevels(layer.levels).error : null;
  return (
    <ControlGroup label={label} className="composite-contour-control">
      <div className="composite-contour-stack">
        <div className="composite-contour">
        <input
          type="checkbox"
          aria-label={`Enable ${label}`}
          checked={layer.enabled}
          onChange={(e) => onChange({ ...layer, enabled: e.target.checked })}
          title={`Toggle ${label}`}
        />
        <select
          aria-label={`${label} field`}
          value={layer.field ?? ""}
          disabled={!layer.enabled}
          onChange={(e) =>
            onChange({ ...layer, field: e.target.value || null, levels: e.target.value && !layer.levels ? defaultLevels(e.target.value, fieldUnits) : layer.levels })
          }
        >
          <option value="">(field)</option>
          {fields.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
        <input
          type="text"
          aria-label={`${label} levels`}
          aria-invalid={levelError != null}
          className="composite-levels"
          value={layer.levels}
          disabled={!layer.enabled}
          placeholder="levels"
          title="Levels: a comma list (40 or 0.01,0.02) or start:end:step"
          onChange={(e) => onChange({ ...layer, levels: e.target.value })}
        />
        <select
          aria-label={`${label} color`}
          value={layer.color}
          disabled={!layer.enabled}
          onChange={(e) => onChange({ ...layer, color: e.target.value })}
        >
          {CONTOUR_COLORS.map((c) => (
            <option key={c.value} value={c.value}>{c.label}</option>
          ))}
        </select>
        </div>
        {(levelError || warning) && (
          <span className="composite-contour__status" role="status">{levelError ?? warning}</span>
        )}
      </div>
    </ControlGroup>
  );
}

/* ---------- trace construction ---------- */

interface Grid2D {
  data: number[][];
  rowAxis: SliceAxis;
  colAxis: SliceAxis;
  rows: number;
  cols: number;
}

function remapLayer(domain: ReturnType<typeof cropToValidPlotDomain> | null, axis: SliceAxis) {
  if (!domain) return { field: null, grid: null, error: null, remapped: false };
  try { return { ...remapTerrainVerticalSlice(domain.field, domain.grid, axis), error: null }; }
  catch (error) { return { field: null, grid: null, remapped: false,
    error: error instanceof Error ? error.message : String(error) }; }
}

/** Slice a 3D field at (axis, index), or pass a 2D (nr, nθ) surface field
 *  through directly, into a uniform 2D grid (rows = first axis, cols = second). */
function layerTo2D(field: NpyN, axis: SliceAxis, index: number): Grid2D | null {
  if (field.shape.length === 3) {
    const f = field as Npy3D;
    const d = f.shape[axis];
    const s = slice2D(f, axis, Math.min(index, d - 1));
    return {
      data: s.data,
      rowAxis: s.rowAxis as SliceAxis,
      colAxis: s.colAxis as SliceAxis,
      rows: s.rows,
      cols: s.cols,
    };
  }
  if (field.shape.length === 2) {
    const [nr, nth] = field.shape;
    const data: number[][] = new Array(nr);
    for (let i = 0; i < nr; i++) {
      const row = new Array<number>(nth);
      const base = i * nth;
      for (let j = 0; j < nth; j++) row[j] = field.data[base + j];
      data[i] = row;
    }
    return { data, rowAxis: 0, colAxis: 1, rows: nr, cols: nth };
  }
  return null;
}

interface ContourArg extends ContourLayer {
  plane: ReturnType<typeof prepareContourPlane>;
  grid: GridInfo | null;
}

function prepareContourPlane(field: NpyN, axis: SliceAxis, index: number) {
  const plane = layerTo2D(field, axis, index);
  return plane ? { ...plane, range: displayedRange(plane.data) } : null;
}

interface BuildArgs {
  fill: Grid2D | null;
  fillRank: number | null;
  fillName: string | null;
  fillColormap: ExtendedColorscale;
  fillRange: FillRangeMode;
  range: { min: number; max: number } | null;
  frameRange: { min: number; max: number } | null;
  contourA: ContourArg;
  contourB: ContourArg;
  sliceAxis: SliceAxis;
  grid: GridInfo | null;
  fieldUnits: FieldUnits;
  dragMode: DragMode;
  fitMode: PlotFitMode;
  chrome: PlotChrome;
}

function buildComposite({
  fill,
  fillRank,
  fillName,
  fillColormap,
  fillRange,
  range,
  frameRange,
  contourA,
  contourB,
  sliceAxis,
  grid,
  fieldUnits,
  dragMode,
  fitMode,
  chrome,
}: BuildArgs): {
  data: Data[];
  layout: Partial<Layout>;
  /** Whether Fit applies to this plane; see SlicePlot's BuiltPlot. */
  aspectLockable: boolean;
  contourWarnings: Array<{ field: string; text: string }>;
} | null {
  if (!fill || !fillName) return null;
  const fillGrid = fill;

  const xCoords = axisCoords(fillGrid.colAxis, fillGrid.cols, grid);
  const yCoords = axisCoords(fillGrid.rowAxis, fillGrid.rows, grid);
  const { zmin, zmax } = resolveFillZ(
    fillRange, padDegenerateRange(range), isDivergingColorscale(fillColormap),
    usesFixedAnchorRange(fillColormap),
  );

  const data: Data[] = [
    {
      type: "heatmap",
      // Keep the fill and its contour source on the same resolved lattice.
      zsmooth: false,
      z: fillGrid.data,
      x: xCoords.values,
      y: yCoords.values,
      colorscale: resolveColorscale(fillColormap) as never,
      zmin,
      zmax,
      colorbar: { title: { text: labelWithUnit(fillName, fieldUnits), side: "right" } },
      hovertemplate:
        `${axisLabel(fillGrid.rowAxis, grid)}=%{y${yCoords.hoverFmt}}` +
        `<br>${axisLabel(fillGrid.colAxis, grid)}=%{x${xCoords.hoverFmt}}` +
        `<br>${fillName}=%{z:.4g}<extra></extra>`,
    },
  ];

  const fillPlane = { rowAxis: fillGrid.rowAxis, colAxis: fillGrid.colAxis };
  const skipped: ContourSkip[] = [];
  for (const layer of [contourA, contourB]) {
    const result = contourTraces(layer, layer.grid, fieldUnits, fillPlane);
    data.push(...result.traces);
    if (result.skipped) skipped.push(result.skipped);
  }

  const titleSuffix = sliceAxis != null && fillRank === 3
    ? ` — ${axisLabel(sliceAxis, grid)}-slice`
    : "";
  // Keep each diagnostic with the layer control that owns the remedy.
  const contourWarnings = skipped.flatMap((skip) => {
    const text = describeContourSkips([skip],
      `${axisLabel(fillPlane.rowAxis, grid)}–${axisLabel(fillPlane.colAxis, grid)}`)[0];
    return text ? [{ field: skip.field, text }] : [];
  });
  // Lock 1:1 on a horizontal plan view (see planeSupportsLockedAspect) so the
  // fill and its contour overlays keep true shape; sections stay exaggerated.
  const lockAspect = planeSupportsLockedAspect(
    fillGrid.rowAxis, fillGrid.colAxis, yCoords, xCoords,
  );
  const fit = fitAxisFragments(lockAspect, fitMode);
  const colorNotice = scalarColorRangeAnnotation(
    frameRange,
    { min: zmin, max: zmax },
    labelWithUnit(fillName, fieldUnits),
    chrome,
  );
  const hasContourLegend = data.some((trace) => trace.type === "contour" && trace.showlegend === true);
  const layout: Partial<Layout> = {
    xaxis: { title: { text: xCoords.title }, ...fit.x },
    yaxis: { title: { text: yCoords.title }, ...fit.y },
    dragmode: dragMode,
    showlegend: hasContourLegend,
    legend: hasContourLegend ? {
      x: 0.99,
      y: 0.99,
      xanchor: "right",
      yanchor: "top",
      bgcolor: chrome.overlayBackground,
      bordercolor: chrome.zeroline,
      borderwidth: 1,
      font: { size: 10 },
    } : undefined,
    title: { text: `${fillName}${titleSuffix} + contours`, font: { size: 13 } },
    annotations: colorNotice ? [colorNotice] : [],
  };

  return { data, layout, aspectLockable: lockAspect, contourWarnings };
}

/** Result of building one contour layer: its traces, plus why it drew nothing
 *  when an enabled layer produced no output. See lib/plot/plotLabels.ts. */
interface ContourResult {
  traces: Data[];
  skipped: ContourSkip | null;
}

/** One Plotly contour trace per requested level (constant-colour lines). A
 *  layer is only drawn when it shares the fill's plane: a 2D column product
 *  (always r-θ / x-y) layered on a vertical fill slice would otherwise be drawn
 *  against the wrong coordinate, so it is skipped and reported instead. */
function contourTraces(
  layer: ContourArg,
  grid: GridInfo | null,
  fieldUnits: FieldUnits,
  fillPlane: { rowAxis: SliceAxis; colAxis: SliceAxis },
): ContourResult {
  if (!layer.enabled || !layer.field || !layer.plane) return { traces: [], skipped: null };
  const g = layer.plane;
  if (g.rowAxis !== fillPlane.rowAxis || g.colAxis !== fillPlane.colAxis) {
    return { traces: [], skipped: { field: layer.field, reason: "plane" } };
  }
  const levels = parseContourLevels(layer.levels).levels;
  if (levels.length === 0) return { traces: [], skipped: { field: layer.field, reason: "levels" } };

  const finiteRange = g.range;
  const drawableLevels = finiteRange == null
    ? []
    : levels.filter((level) => level >= finiteRange.min && level <= finiteRange.max);
  const skipped = drawableLevels.length < levels.length
    ? { field: layer.field, reason: "range" as const }
    : null;

  const xCoords = axisCoords(g.colAxis, g.cols, grid);
  const yCoords = axisCoords(g.rowAxis, g.rows, grid);
  // Contour level reads "field = L unit" (e.g. "w = 10 m/s"); the unit trails
  // the level rather than the field name so the equality stays readable.
  const unit = unitFor(layer.field, fieldUnits);
  const unitSuffix = unit ? ` ${unit}` : "";

  const traces = drawableLevels.map(
    (L) =>
      ({
        type: "contour",
        z: g.data,
        x: xCoords.values,
        y: yCoords.values,
        // start == end draws the single level L; coloring "none" keeps the
        // constant line colour rather than mapping it through a colorscale.
        // `size` only has to be positive — over a zero-width range it steps once
        // and stops — so it is a constant. Deriving it from |L| made the spec
        // depend on the level's magnitude for no reason, and fell back to an
        // arbitrary 1 for the most commonly requested level of all, L = 0.
        contours: { coloring: "none", start: L, end: L, size: 1 },
        // Draw Plotly's contour geometry as computed. Spline smoothing moves
        // the isoline away from the field = L crossing it represents.
        line: { color: layer.color, width: 1.6, smoothing: 0 },
        connectgaps: false,
        showscale: false,
        showlegend: true,
        hoverinfo: "skip",
        name: `${layer.field} = ${L}${unitSuffix}`,
      }) as Data,
  );
  return { traces, skipped };
}

function resolveFillZ(
  mode: FillRangeMode,
  range: { min: number; max: number },
  /** True when the fill colormap diverges, so its neutral colour is only honest
   *  sitting on zero. See isDivergingColorscale. */
  diverging: boolean,
  /** True when the colormap's stops are anchored to physical values, so the
   *  range is not the user's to fit. See usesFixedAnchorRange. */
  fixedAnchors: boolean,
): { zmin: number; zmax: number } {
  // Ahead of the mode: stretching the NWS ramp over a fitted range moves every
  // colour to the wrong reflectivity. See SlicePlot's resolveRange.
  if (mode === "dBZ" || fixedAnchors) return { zmin: DBZ_RANGE[0], zmax: DBZ_RANGE[1] };
  if (mode === "symmetric" || diverging) {
    // "auto" with a diverging palette still centres on zero: otherwise the
    // neutral colour lands at (min + max) / 2 and the zero contour is drawn in
    // a signed colour. See SlicePlot's resolveRange.
    // padIfDegenerate already guarantees a non-zero span, so no `|| 1` fallback
    // — matching SlicePlot's resolveRange, which computes this identically.
    const m = Math.max(Math.abs(range.min), Math.abs(range.max));
    return { zmin: -m, zmax: m };
  }
  return { zmin: range.min, zmax: range.max };
}
