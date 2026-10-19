import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { slice2D, sliceDisplayAxes, type SliceAxis, type Npy3D, type NpyN } from "@/lib/platform/npy";
import { planeFocusFromView } from "@/lib/plot/volumetricHandoff";
import type { AppErrorCode } from "@/lib/platform/appInvoke";
import {
  axisCoords,
  axisLabel,
  axisOptions,
  hasPhysicalAxisCoordinates,
  heightKmToZIndex,
  planeSupportsLockedAspect,
  scalarCoordinateKm,
} from "@/lib/science/gridCoords";
import { registeredFieldRank, type GridInfo } from "@/lib/run/runOutputs";
import { describeFrameTransition, findPublishedFrame } from "@/lib/run/frameIdentity";
import { labelWithUnit, type FieldUnits } from "@/lib/science/fieldUnits";
import { cropToValidPlotDomain, validAxisLength } from "@/lib/science/validDomain";
import { axisViewBounds } from "@/lib/plot/viewSync";
import { WIDEST_OUTPUT_PRESET, fieldOrigins, presetPublishing } from "@/lib/plot/fieldProvenance";
import {
  DBZ_RANGE,
  usesFixedAnchorRange,
  isDivergingColorscale,
  padDegenerateRange,
  resolveColorscale,
  type ExtendedColorscale,
} from "@/lib/plot/plotColormaps";
import { defaultScalarDisplay } from "@/lib/plot/scalarDisplayDefaults";
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
  useFieldLink,
  usePlaneViewport,
  usePlotChrome,
  useScalarSelectionReport,
  scalarColorRangeAnnotation,
  VolumeHandoffButton,
  type PlotComponentProps,
} from "./common";
import { displayedRange, displayedRangeScope } from "@/lib/plot/displayedRange";
import {
  fieldSupportsBoundedWindows,
  fieldWindowAggregation,
  PLOTLY_HEATMAP_CELL_CAP,
  planBoundedPlaneWindows,
  planBoundedSliceWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { remapTerrainVerticalSlice } from "@/lib/science/terrainSliceRemap";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { loadedSliceWindowIndex, sourceSliceControlsAvailable } from "@/lib/plot/sliceWindowSelection";
import { SlicePlaneContext } from "./SlicePlaneContext";
import { SliceIndexInput } from "./SliceIndexInput";

/*
 * 2D slice heatmap. Handles three input shapes:
 *
 *   (nr, nth, nz)  prognostic + 3D diagnostic fields. User picks a
 *                   slice axis + index; heatmap shows the remaining 2D.
 *   (nr, nth)      surface / column diagnostic fields (composite_
 *                   reflectivity, cape, t2, ...). Renders directly,
 *                   slice picker hidden.
 *   (nz,) or (nr,) other shapes — not supported, shows a hint.
 */

type RangeMode = "auto" | "dBZ" | "symmetric" | "fixed";

const COLORMAPS: ReadonlyArray<{ value: ExtendedColorscale; label: string }> = [
  { value: "Viridis", label: "Viridis" },
  { value: "Plasma", label: "Plasma" },
  { value: "Inferno", label: "Inferno" },
  { value: "RdBu", label: "RdBu" },
  { value: "Greys", label: "Greys" },
  { value: "dBZ", label: "NWS dBZ" },
  { value: "ColdPool", label: "Cold pool (BuRd)" },
];

const RANGE_MODES: ReadonlyArray<{ value: RangeMode; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "symmetric", label: "± auto" },
  { value: "dBZ", label: "dBZ" },
  { value: "fixed", label: "Fixed" },
];

type DragMode = "pan" | "zoom";

const DRAG_OPTIONS: ReadonlyArray<{ value: DragMode; label: string }> = [
  { value: "pan", label: "Pan" },
  { value: "zoom", label: "Box zoom" },
];

export function SlicePlot({
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

  const [fieldName, setFieldName] = useState<string | null>(
    preset?.field ?? rootFields[0] ?? null,
  );
  const hierarchyRequiredFields = useMemo(() => fieldName ? [fieldName] : [], [fieldName]);
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
  const fieldsForStep = useMemo(() => currentStep?.fields ?? [], [currentStep?.fields]);
  const renderableFields = useMemo(
    () => fieldsForPlotRanks(runDir, currentStep, defaultSliceAxis === 1 ? [3] : [2, 3]),
    [currentStep, defaultSliceAxis, runDir],
  );
  const grid = currentStep?.grid ?? null;
  const [sliceAxis, setSliceAxis] = useState<SliceAxis>(
    preset?.sliceAxis ?? defaultSliceAxis ?? 2,
  );
  const [sliceIndex, setSliceIndex] = useState(0);
  const [displayedSliceIndex, setDisplayedSliceIndex] = useState(0);
  const appliedSlicePresetRef = useRef<string | null>(null);
  const sliceTouchedRef = useRef(false);
  const initialDisplay = defaultScalarDisplay(
    preset?.field ?? rootFields[0] ?? null,
    currentStep?.fieldUnits,
  );
  const [colormap, setColormap] = useState<ExtendedColorscale>(
    (preset?.colormap as ExtendedColorscale | undefined) ?? initialDisplay.colormap,
  );
  const [rangeMode, setRangeMode] = useState<RangeMode>(
    preset?.rangeMode ?? initialDisplay.rangeMode,
  );
  // Box-zoom default: natural "drag to inspect" gesture for a radar slice.
  const [dragMode, setDragMode] = useState<DragMode>("zoom");
  // True shape default: plan views are maps; stretched maps misreport distances.
  const [fitMode, setFitMode] = useState<PlotFitMode>("true-shape");
  // Bumping forces a fresh Plotly newPlot and re-applies autorange.
  const [resetTick, setResetTick] = useState(0);
  const [colorResetTick, setColorResetTick] = useState(0);
  const [usePresetRange, setUsePresetRange] = useState(true);
  const planeViewIdentity = JSON.stringify([runDir, fieldName, sliceAxis]);
  const { viewport, onRelayout, resetViewport } = usePlaneViewport(planeViewIdentity);

  // Apply a new preset when its id changes.
  useEffect(() => {
    if (!preset) return;
    setUsePresetRange(true);
    sliceTouchedRef.current = false;
    appliedSlicePresetRef.current = null;
    if (preset.field) setFieldName(preset.field);
    if (preset.colormap) setColormap(preset.colormap as ExtendedColorscale);
    if (preset.rangeMode) setRangeMode(preset.rangeMode);
    if (preset.sliceAxis != null) setSliceAxis(preset.sliceAxis);
  }, [preset]);

  // Reset slice orientation on category change; keyed on prop so manual changes within a category persist.
  useEffect(() => {
    if (defaultSliceAxis == null) return;
    setSliceAxis(defaultSliceAxis);
  }, [defaultSliceAxis]);

  /* Colour follows the quantity. A dBZ palette carried over from a radar preset anchors the
   * colorbar to [-30, 75] dBZ, so leaving it in place relabels the bar with the new field's
   * unit while the numbers on it still mean reflectivity. Shared by the picker and by
   * adopting a linked field, which must land in the same state. */
  const applyField = useCallback((next: string | null) => {
    setFieldName(next);
    setUsePresetRange(false);
    const display = defaultScalarDisplay(next, currentStep?.fieldUnits);
    setColormap(display.colormap);
    setRangeMode(display.rangeMode);
  }, [currentStep?.fieldUnits]);

  // Any rank draws here: a 3D field is sliced, a surface field is drawn directly.
  useFieldLink(linkedField, fieldName, (f) => renderableFields.includes(f), applyField);

  // Helper: wrap a state setter so each manual change clears the preset.
  const clearPreset = () => onPresetCleared?.();
  const update = <T,>(setter: (v: T) => void) => (v: T) => {
    setter(v);
    clearPreset();
  };

  // Repick the field when the step changes and the current field isn't
  // present in the new step.
  useEffect(() => {
    if (renderableFields.length === 0) return;
    if (fieldName && renderableFields.includes(fieldName)) return;
    applyField(renderableFields[0] ?? null);
  }, [renderableFields, fieldName, applyField]);

  // Debounce so a fast scrub loads only the settled frame, not every frame swept through.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const sourceRank = currentStep && fieldName
    ? registeredFieldRank(
        runDir,
        currentStep.index,
        fieldName,
        currentStep.publicationGeneration,
      )
    : null;
  const sourceSliceDim = validAxisLength(sliceAxis, grid)
    ?? ([grid?.nr, grid?.nth, grid?.nz][sliceAxis] ?? 0);
  useEffect(() => {
    if (sourceRank !== 3 || !Number.isSafeInteger(sourceSliceDim) || sourceSliceDim < 1) return;
    setSliceIndex(previous => Math.max(0, Math.min(previous, sourceSliceDim - 1)));
  }, [sourceRank, sourceSliceDim]);
  const boundedProvider = fieldName
    ? fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[fieldName])
    : false;
  const sliceWindowOutcome = useMemo(() => {
    if (!fieldName || !boundedProvider) return null;
    if (sourceRank === 3) return planBoundedSliceWindows(
      grid, sliceAxis, sliceIndex, PLOTLY_HEATMAP_CELL_CAP, viewport,
    );
    if (sourceRank === 2) return planBoundedPlaneWindows(
      grid, PLOTLY_HEATMAP_CELL_CAP, viewport,
    );
    return null;
  }, [sourceRank, fieldName, boundedProvider, grid, sliceAxis, sliceIndex, viewport]);
  const boundedLoadError = fieldName && currentStep
    ? !boundedProvider
      ? "This field provider cannot serve bounded plot windows; the full-field fallback is disabled."
      : sourceRank !== 2 && sourceRank !== 3
        ? "This plot requires a rank-2 plane or rank-3 volume with declared grid metadata."
        : sliceWindowOutcome && !sliceWindowOutcome.ok
          ? sliceWindowOutcome.reason
          : null
    : null;
  const sliceWindows = sliceWindowOutcome?.ok
    ? windowsForFieldAggregation(sliceWindowOutcome.plan.windows, fieldName)
    : null;
  const fieldResult = useFieldAny(
    runDir,
    boundedLoadError ? null : loadFrame.stepIndex,
    fieldName,
    loadFrame.publicationGeneration,
    sliceWindows,
    JSON.stringify([runDir, fieldName, sourceRank, sliceAxis]),
  );
  const { error, errorCode } = fieldResult;
  const loading = fieldResult.loading || sourceRank === 3 && displayedSliceIndex !== sliceIndex;
  const sliceControlsAvailable = sourceSliceControlsAvailable(sourceRank, sourceSliceDim, boundedLoadError);
  const [renderedStepIndex, setRenderedStepIndex] = useState<number | null>(null);
  const [renderedSampling, setRenderedSampling] = useState<{
    runDir: string; fieldName: string | null; viewport: boolean; stride: number[]; sourceShape: number[]; fullShape: number[];
  } | null>(null);
  const [renderedPlane, setRenderedPlane] = useState<{
    runDir: string; fieldName: string | null; grid: GridInfo | null | undefined; axis: SliceAxis; index: number; physicalHeight: boolean;
  } | null>(null);
  const [renderedPublicationGeneration, setRenderedPublicationGeneration] =
    useState<string | null>(null);

  useEffect(() => {
    setRenderedStepIndex(null);
    setRenderedPublicationGeneration(null);
  }, [runDir, fieldName]);

  const requestedStepIndex = currentStep?.index ?? null;
  const requestedPublicationGeneration = currentStep?.publicationGeneration ?? null;
  const frameTransition = describeFrameTransition(
    requestedStepIndex,
    renderedStepIndex,
    requestedPublicationGeneration,
    renderedPublicationGeneration,
  );
  useEffect(() => {
    onFrameDisplayChange?.({
      requestedStepIndex,
      requestedPublicationGeneration,
      renderedStepIndex,
      renderedPublicationGeneration,
      loading: !boundedLoadError && (loading || frameTransition.kind !== "ready"),
    });
  }, [
    onFrameDisplayChange,
    requestedStepIndex,
    requestedPublicationGeneration,
    renderedStepIndex,
    renderedPublicationGeneration,
    loading,
    frameTransition.kind,
    boundedLoadError,
  ]);

  const loadedStep = findPublishedFrame(
    steps,
    fieldResult.loadedStepIndex,
    fieldResult.loadedPublicationGeneration,
  );
  const validDomain = useMemo(
    () => fieldResult.field
      ? cropToValidPlotDomain(fieldResult.field, loadedStep?.grid)
      : null,
    [fieldResult.field, loadedStep?.grid],
  );
  const field = validDomain?.field ?? null;
  const loadedSliceIndex = loadedSliceWindowIndex(fieldResult.field, sliceAxis, loadedStep?.grid?.validIndexRange?.lower,
    fieldResult.loading ? displayedSliceIndex : sliceIndex);
  const displayGrid = validDomain?.grid ?? null;
  const displayFieldUnits = loadedStep?.fieldUnits;

  const rank = field?.shape.length ?? null;
  const is3D = sourceRank === 3 || rank === 3;
  const is2D = rank === 2;
  const terrainSlice = useMemo(() => {
    if (!field) return { value: null, error: null as string | null };
    try {
      return {
        value: remapTerrainVerticalSlice(field, displayGrid, sliceAxis),
        error: null as string | null,
      };
    } catch (remapError) {
      return {
        value: null,
        error: remapError instanceof Error ? remapError.message : String(remapError),
      };
    }
  }, [displayGrid, field, sliceAxis]);
  const plotField = terrainSlice.value?.field ?? field;
  const plotGrid = terrainSlice.value?.grid ?? displayGrid;

  // Resolve the preset from source dimensions before requesting its plane.
  useEffect(() => {
    if (!preset) {
      appliedSlicePresetRef.current = null;
      return;
    }
    if (sourceRank !== 3 || !Number.isSafeInteger(sourceSliceDim) || sourceSliceDim < 1 || sliceTouchedRef.current) return;
    if (preset.sliceAxis != null && sliceAxis !== preset.sliceAxis) return;
    const dim = sourceSliceDim;
    if (appliedSlicePresetRef.current === preset.id) return;
    appliedSlicePresetRef.current = preset.id;
    // A height-in-km preset wins for z-slices when the grid spacing is known
    // (lets a preset say "w at 5 km" rather than a bare cell index).
    if (preset?.sliceHeightKm != null && sliceAxis === 2
        && grid?.nz && hasPhysicalAxisCoordinates(2, grid.nz, grid)) {
      const storageIndex = heightKmToZIndex(preset.sliceHeightKm, grid, grid.nz);
      const validLower = grid.validIndexRange?.lower[2] ?? 0;
      setSliceIndex(Math.max(0, Math.min(dim - 1, storageIndex - validLower)));
      return;
    }
    const hint = preset?.sliceIndexHint;
    if (hint == null) return;
    if (hint === "surface") setSliceIndex(0);
    else if (hint === "mid") setSliceIndex(Math.floor((dim - 1) / 2));
    else if (hint === "top") setSliceIndex(dim - 1);
    else if (typeof hint === "number") setSliceIndex(Math.min(hint, dim - 1));
  }, [sourceRank, sliceAxis, preset, grid, sourceSliceDim]);

  // Physical height (km AGL) of the current z-slice — drives the Height control
  // and the level readout. Null when the slice isn't a horizontal (z) plane or
  // the grid's vertical spacing is unknown.
  const zCoordinateKm =
    is3D && sliceAxis === 2 && sliceControlsAvailable
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
    if (!is3D || !field) return null;
    const sourceCounts = [grid?.nr, grid?.nth, grid?.nz];
    const cellCount = sourceCounts[sliceAxis] ?? field.shape[sliceAxis];
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
  }, [field, grid, is3D, sliceAxis, sliceIndex, viewport]);
  useScalarSelectionReport(
    onScalarSelectionChange,
    fieldName,
    colormap,
    rangeMode,
    rank,
    "slice",
    planeFocus,
  );

  // Build the 2D heatmap data from whichever rank the field has.
  const heatmap = useMemo(() => {
    if (!plotField) return null;
    if (is3D) {
      const f = plotField as Npy3D;
      const dim = f.shape[sliceAxis];
      const safeIndex = f.windowGeometry ? 0 : Math.min(loadedSliceIndex, dim - 1);
      const s = slice2D(f, sliceAxis, safeIndex);
      return {
        data: s.data,
        rowAxis: s.rowAxis as SliceAxis,
        colAxis: s.colAxis as SliceAxis,
        rows: s.rows,
        cols: s.cols,
      };
    }
    if (is2D) {
      // Surface field: shape (nr, nth). Render directly as (rows=r, cols=θ).
      return inflate2D(plotField);
    }
    return null;
  }, [plotField, sliceAxis, loadedSliceIndex, is3D, is2D]);

  /*
   * Colour scale fitted to the plane on screen, then latched so one physical value keeps one
   * colour across frames. Fitting to the whole volume washed the plane out whenever the
   * column's extremes sat at other heights: on a beast_xl frame w spans 6% of the column
   * range at the ground, and vorticity_z there is identically zero.
   */
  const frameRange = useMemo(() => displayedRange(heatmap?.data), [heatmap]);
  const range = useLatchedRange(
    `${displayedRangeScope(runDir, fieldName, is2D ? null : sliceAxis, is2D ? null : loadedSliceIndex)}|colors:${colorResetTick}`,
    frameRange,
  );

  const plot = useMemo(
    () => buildPlot({
      heatmap,
      sliceAxis: is2D ? null : sliceAxis,
      fieldName,
      colormap,
      rangeMode,
      symmetricMax: usePresetRange && preset?.field === fieldName ? preset.symmetricMax : undefined,
      fixedRange: usePresetRange && preset?.field === fieldName ? preset.fixedRange : undefined,
      range,
      frameRange,
      grid: plotGrid,
      fieldUnits: displayFieldUnits,
      dragMode,
      fitMode,
      chrome,
    }),
    [
      heatmap,
      sliceAxis,
      fieldName,
      colormap,
      rangeMode,
      preset,
      usePresetRange,
      range,
      frameRange,
      plotGrid,
      displayFieldUnits,
      dragMode,
      fitMode,
      chrome,
      is2D,
    ],
  );

  const fieldMissing = currentStep != null && !!fieldName && !fieldsForStep.includes(fieldName);

  return (
    <>
      <div className="plot-controls">
        <OutputHierarchyControls source={outputHierarchySource} />
        <ControlGroup label="Field">
          <select
            aria-label="Field"
            value={fieldName ?? ""}
            onChange={(e) => {
              const next = e.target.value || null;
              applyField(next);
              onFieldChange?.(next);
              clearPreset();
            }}
          >
            {renderableFields.length === 0 && (
              <option value="">{defaultSliceAxis === 1 ? "(no rank-3 fields)" : "(no rank-2/3 fields)"}</option>
            )}
            {renderableFields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
            {fieldMissing && fieldName && (
              <option value={fieldName}>
                {fieldName} (not exported)
              </option>
            )}
          </select>
        </ControlGroup>

        <PlotCompressionControls step={loadedStep} fields={[fieldName]} />

        {is3D && (
          <>
            <ControlGroup label="Slice axis">
              <PillGroup
                options={axisOptions(grid)}
                value={sliceAxis}
                onChange={(value) => {
                  sliceTouchedRef.current = true;
                  update(setSliceAxis)(value);
                }}
              />
            </ControlGroup>

            <ControlGroup
              label={`Index (${axisLabel(sliceAxis, grid)} = ${sliceIndex})`}
              stretch
            >
              <input
                type="range"
                aria-label={`Selected ${axisLabel(sliceAxis, grid)} index`}
                min={0}
                max={Math.max(0, sourceSliceDim - 1)}
                value={sliceIndex}
                onChange={(e) => {
                  sliceTouchedRef.current = true;
                  setSliceIndex(Number(e.target.value));
                  clearPreset();
                }}
                disabled={!sliceControlsAvailable}
              />
              <SliceIndexInput value={sliceIndex} max={Math.max(0, sourceSliceDim - 1)}
                disabled={!sliceControlsAvailable} label={`Go to ${axisLabel(sliceAxis, grid)} index`}
                onCommit={value => { sliceTouchedRef.current = true; setSliceIndex(value); clearPreset(); }} />
            </ControlGroup>

            {sliceAxis === 2 && sliceControlsAvailable && grid?.nz
              && hasPhysicalAxisCoordinates(2, grid.nz, grid) && (
              <ControlGroup
                label={`Height${zHeightKm != null ? ` (${zHeightKm} km)` : " (km)"}`}
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
                      const storageIndex = heightKmToZIndex(km, grid, grid.nz!);
                      const validLower = grid.validIndexRange?.lower[2] ?? 0;
                      setSliceIndex(Math.max(
                        0,
                        Math.min(sourceSliceDim - 1, storageIndex - validLower),
                      ));
                      clearPreset();
                    }
                  }}
                  title="Jump to a height in km AGL (snaps to the nearest model level)"
                />
              </ControlGroup>
            )}
          </>
        )}

        <ControlGroup label="Colormap">
          <select
            value={colormap}
            onChange={(e) => {
              setColormap(e.target.value as ExtendedColorscale);
              clearPreset();
            }}
          >
            {COLORMAPS.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </ControlGroup>

        <ControlGroup label="Range">
          <PillGroup
            options={RANGE_MODES}
            value={rangeMode}
            onChange={update(setRangeMode)}
          />
        </ControlGroup>

        {plot?.aspectLockable && (
          <ControlGroup label="Fit">
            <PillGroup options={FIT_OPTIONS} value={fitMode} onChange={setFitMode} />
          </ControlGroup>
        )}

        <ControlGroup label="Drag">
          <PillGroup options={DRAG_OPTIONS} value={dragMode} onChange={setDragMode} />
        </ControlGroup>

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
            applyField(fieldName);
            setColorResetTick((tick) => tick + 1);
            clearPreset();
          }}
          title="Restore the field's palette and fit colors to this slice; reflectivity keeps its fixed dBZ scale"
        >
          Reset colors
        </button>

        <VolumeHandoffButton offer={volumeHandoff} />
      </div>

      <div className="plot-body">
        {plot && is3D && (
          <SlicePlaneContext grid={renderedPlane?.runDir === runDir && renderedPlane.fieldName === fieldName ? renderedPlane.grid : null}
            axis={renderedPlane?.runDir === runDir && renderedPlane.fieldName === fieldName ? renderedPlane.axis : sliceAxis}
            index={renderedPlane?.runDir === runDir && renderedPlane.fieldName === fieldName ? renderedPlane.index : null}
            physicalHeight={renderedPlane?.physicalHeight} />
        )}
        {loading && field && is3D && (
          <div className="plot-overlay" role="status">
            <Icon name="dot" size={12} />
            <span>
              Displaying index {displayedSliceIndex} · loading selected {axisLabel(sliceAxis, grid)} index {sliceIndex}…
            </span>
          </div>
        )}
        {error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{describeFieldError(error, errorCode, fieldName)}</span>
          </div>
        )}
        {boundedLoadError && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{boundedLoadError}</span>
          </div>
        )}
        {terrainSlice.error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{terrainSlice.error}</span>
          </div>
        )}
        {terrainSlice.value?.remapped && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>physical-volume terrain→Cartesian vertical remap</span>
          </div>
        )}
        {field && renderedSampling?.runDir === runDir && renderedSampling.fieldName === fieldName && (
          <>
            {renderedSampling.stride.some(stride => stride > 1) && (
              <div className="plot-overlay">
                <Icon name="dot" size={12} />
                <span>{fieldWindowAggregation(fieldName) === "linear_reflectivity_mean" ? "bounded linear-Z mean" : "bounded cell-mean"} {renderedSampling.stride.join("×")} · source {renderedSampling.sourceShape.join("×")}</span>
              </div>
            )}
            {renderedSampling.viewport && (
              <div className="plot-overlay">
                <Icon name="dot" size={12} />
                <span>viewport {renderedSampling.stride.every(stride => stride === 1) ? "native cells"
                  : `${fieldWindowAggregation(fieldName) === "linear_reflectivity_mean" ? "linear-Z" : "cell"} mean ${renderedSampling.stride.join("×")}`}
                  {` · full plane ${renderedSampling.fullShape.join("×")}`}</span>
              </div>
            )}
          </>
        )}
        {!field && !loading && !error && fieldMissing && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{describeUnexportedField(fieldName)}</span>
          </div>
        )}
        {plot ? (
          <PlotCanvas
            key={resetTick}
            viewIdentity={planeViewIdentity}
            data={plot.data}
            layout={plot.layout}
            onRelayout={onRelayout}
            onRendered={() => {
              if (is3D) {
                setDisplayedSliceIndex(loadedSliceIndex);
                const next = { runDir, fieldName, grid: loadedStep?.grid, axis: sliceAxis,
                  index: loadedSliceIndex, physicalHeight: terrainSlice.value?.remapped === true };
                setRenderedPlane(previous => previous?.runDir === next.runDir && previous.fieldName === next.fieldName &&
                  previous.grid === next.grid && previous.axis === next.axis && previous.index === next.index &&
                  previous.physicalHeight === next.physicalHeight ? previous : next);
              }
              if (fieldResult.field) {
                const geometry = fieldResult.field.windowGeometry;
                const triple = (values: readonly number[]) => [values[0] ?? 1, values[1] ?? 1, values[2] ?? 1];
                const next = { runDir, fieldName, viewport: viewport != null,
                  stride: triple(geometry?.stride ?? [1,1,1]),
                  sourceShape: triple(geometry?.extent ?? fieldResult.field.shape),
                  fullShape: triple(geometry?.sourceShape ?? fieldResult.field.shape) };
                setRenderedSampling(previous => JSON.stringify(previous) === JSON.stringify(next) ? previous : next);
              }
              if (fieldResult.loadedStepIndex != null) {
                setRenderedStepIndex(fieldResult.loadedStepIndex);
                setRenderedPublicationGeneration(
                  fieldResult.loadedPublicationGeneration,
                );
              }
            }}
          />
        ) : (
          !loading &&
          !error &&
          !boundedLoadError && (
            <div className="plot-overlay">
              <span>No data to plot.</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

interface Heatmap2D {
  data: number[][];
  rowAxis: SliceAxis;
  colAxis: SliceAxis;
  rows: number;
  cols: number;
}

/** Inflate a 2D (nr, nth) field into the same Heatmap2D shape SlicePlot
 *  uses for 3D-slice output, so the rendering path is unified. Row
 *  axis = r, column axis = θ. */
function inflate2D(field: NpyN): Heatmap2D {
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

interface BuildPlotArgs {
  heatmap: Heatmap2D | null;
  sliceAxis: SliceAxis | null;
  fieldName: string | null;
  colormap: ExtendedColorscale;
  rangeMode: RangeMode;
  symmetricMax?: number;
  fixedRange?: readonly [number, number];
  range: { min: number; max: number } | null;
  frameRange: { min: number; max: number } | null;
  grid: GridInfo | null;
  fieldUnits: FieldUnits;
  dragMode: DragMode;
  fitMode: PlotFitMode;
  chrome: PlotChrome;
}

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
  /**
   * Whether this plane's axes are commensurable, i.e. whether Fit does
   * anything. Reported so the control can be hidden on a vertical section
   * rather than offered as a toggle that silently has no effect.
   */
  aspectLockable: boolean;
}

function buildPlot({
  heatmap,
  sliceAxis,
  fieldName,
  colormap,
  rangeMode,
  symmetricMax,
  fixedRange,
  range,
  frameRange,
  grid,
  fieldUnits,
  dragMode,
  fitMode,
  chrome,
}: BuildPlotArgs): BuiltPlot | null {
  if (!heatmap || !fieldName) return null;

  const padded = padDegenerateRange(range);
  const { zmin, zmax } = resolveRange(
    rangeMode, padded, symmetricMax, fixedRange, isDivergingColorscale(colormap),
    usesFixedAnchorRange(colormap),
  );
  const colorscale = resolveColorscale(colormap);

  const xCoords = axisCoords(heatmap.colAxis, heatmap.data[0]?.length ?? 0, grid);
  const yCoords = axisCoords(heatmap.rowAxis, heatmap.data.length, grid);

  const data: Data[] = [
    {
      type: "heatmap",
      // Preserve the resolved cells exactly. Plotly's interpolated modes are a
      // presentation reconstruction and can visually bridge masked cells.
      zsmooth: false,
      z: heatmap.data,
      x: xCoords.values,
      y: yCoords.values,
      colorscale: colorscale as never,
      zmin,
      zmax,
      opacity: 1.0,
      colorbar: { title: { text: labelWithUnit(fieldName, fieldUnits), side: "right" } },
      hovertemplate:
        `${axisLabel(heatmap.rowAxis, grid)}=%{y${yCoords.hoverFmt}}` +
        `<br>${axisLabel(heatmap.colAxis, grid)}=%{x${xCoords.hoverFmt}}` +
        `<br>${fieldName}=%{z:.4g}<extra></extra>`,
    },
  ];

  // Rank-2 fields are plan views (surface fields OR column products); "plan view" is unambiguous.
  const titleSuffix = sliceAxis != null ? ` — ${axisLabel(sliceAxis, grid)}-slice` : " — plan view";
  // Lock 1:1 on a horizontal plan view so a square domain stays square (see
  // planeSupportsLockedAspect); vertical sections stay free to exaggerate.
  const lockAspect = planeSupportsLockedAspect(
    heatmap.rowAxis, heatmap.colAxis, yCoords, xCoords,
  );
  const fit = fitAxisFragments(lockAspect, fitMode);
  const colorNotice = scalarColorRangeAnnotation(
    frameRange,
    { min: zmin, max: zmax },
    labelWithUnit(fieldName, fieldUnits),
    chrome,
  );
  const layout: Partial<Layout> = {
    // Bound zoom-out to the simulated domain so the storm doesn't become a speck.
    xaxis: {
      title: { text: xCoords.title },
      ...axisViewBounds(xCoords.values),
      ...fit.x,
    },
    yaxis: {
      title: { text: yCoords.title },
      ...axisViewBounds(yCoords.values),
      ...fit.y,
    },
    dragmode: dragMode,
    title: {
      text: `${fieldName}${titleSuffix}`,
      font: { size: 13 },
    },
    annotations: colorNotice ? [colorNotice] : [],
  };

  return { data, layout, aspectLockable: lockAspect };
}

function resolveRange(
  mode: RangeMode,
  range: { min: number; max: number },
  symmetricMax: number | undefined,
  fixedRange: readonly [number, number] | undefined,
  /** True when the chosen colormap diverges, so its neutral colour is only
   *  honest sitting on zero. See isDivergingColorscale. */
  diverging: boolean,
  /** True when the colormap's stops are anchored to physical values and the
   *  range is therefore not the user's to fit. See usesFixedAnchorRange. */
  fixedAnchors: boolean,
): { zmin: number; zmax: number } {
  // Before the mode is consulted: the NWS ramp's colours are anchored AT
  // reflectivity values, so it is only meaningful over DBZ_RANGE. Selecting it
  // while the range mode sat on its "auto" default stretched the ramp across
  // the fitted data range and moved every colour to the wrong dBZ.
  if (mode === "dBZ" || fixedAnchors) {
    return { zmin: DBZ_RANGE[0], zmax: DBZ_RANGE[1] };
  }
  if (mode === "symmetric") {
    const m =
      symmetricMax != null
        ? symmetricMax
        : Math.max(Math.abs(range.min), Math.abs(range.max));
    return { zmin: -m, zmax: m };
  }
  if (mode === "fixed" && fixedRange) {
    return { zmin: fixedRange[0], zmax: fixedRange[1] };
  }
  // "auto" + diverging colormap: centre on zero so the neutral colour stays honest.
  if (diverging) {
    const m = Math.max(Math.abs(range.min), Math.abs(range.max));
    return { zmin: -m, zmax: m };
  }
  return { zmin: range.min, zmax: range.max };
}

/** Map a field-load error to user-facing text; translates "not_found" to actionable advice. */
function describeFieldError(
  err: string,
  code: AppErrorCode | null,
  fieldName: string | null,
): string {
  if (code === "not_found") return describeUnexportedField(fieldName);
  return err;
}

/** Actionable explanation for a field absent from this run's output.
 *  Recommends only the minimum output preset that actually produces it. */
function describeUnexportedField(fieldName: string | null): string {
  if (!fieldName) return "Field not exported for this run.";
  const origins = fieldOrigins(fieldName);
  if (!origins.includes("nimbus_output")) {
    return origins.includes("cm1_history")
      ? `${fieldName} is a CM1 history product; the Nimbus engine has no such field, so no output preset produces it.`
      : `${fieldName} is not available from this dataset.`;
  }
  const preset = presetPublishing(fieldName) ?? WIDEST_OUTPUT_PRESET;
  return `${fieldName} not exported for this run — set output.preset to ${preset} in your config and rerun.`;
}
