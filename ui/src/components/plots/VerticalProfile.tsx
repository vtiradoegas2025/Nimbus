import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { profile1D } from "@/lib/platform/npy";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  COMMON_LINE_LAYOUT,
  ControlGroup,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useField,
  type PlotComponentProps,
} from "./common";
import { axisCoords, axisLabel, type AxisCoords } from "@/lib/science/gridCoords";
import { labelWithUnit, type FieldUnits } from "@/lib/science/fieldUnits";
import { findPublishedFrame } from "@/lib/run/frameIdentity";
import { cropToValidPlotDomain, validAxisLength } from "@/lib/science/validDomain";
import {
  fieldSupportsBoundedWindows,
  planExactLineWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { finiteProfileLine } from "@/lib/plot/finiteProfileLine";

/* Field-vs-z at fixed (r, θ): z on y-axis, field on x-axis — sounding convention. */

export function VerticalProfile({
  runDir: rootRunDir,
  steps: rootSteps,
  stepIdx: rootStepIdx,
}: PlotComponentProps) {
  const rootCurrentStep = rootSteps[rootStepIdx] ?? null;
  const rootFields = useMemo(() => rootCurrentStep?.fields ?? [], [rootCurrentStep?.fields]);
  const [fieldName, setFieldName] = useState<string | null>(rootFields[0] ?? null);
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
  const profileFields = useMemo(
    () => fieldsForPlotRanks(runDir, currentStep, [3]),
    [currentStep, runDir],
  );
  const grid = currentStep?.grid ?? null;
  const nr = grid?.nr ?? 0;
  const nth = grid?.nth ?? 0;
  // Index-picker axis letters for the Cartesian grid: x/y/z.
  const axis0Label = axisLabel(0, grid);
  const axis1Label = axisLabel(1, grid);

  const [rIdx, setRIdx] = useState(0);
  const [thIdx, setThIdx] = useState(0);

  useEffect(() => {
    if (fieldName && profileFields.includes(fieldName)) return;
    setFieldName(profileFields[0] ?? null);
  }, [profileFields, fieldName]);

  // Debounce the exact-column read so a scrub/autoplay does not request every
  // intermediate frame. Recording bypasses the debounce.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const linePlan = useMemo(() => {
    if (!fieldName || !fieldSupportsBoundedWindows(currentStep?.fieldOrigins?.[fieldName])) {
      return null;
    }
    const outcome = planExactLineWindows(grid, 2, [rIdx, thIdx, 0]);
    return outcome.ok ? windowsForFieldAggregation(outcome.plan.windows, fieldName) : null;
  }, [currentStep?.fieldOrigins, fieldName, grid, rIdx, thIdx]);
  const fieldResult = useField(
    runDir,
    linePlan ? loadFrame.stepIndex : null,
    fieldName,
    loadFrame.publicationGeneration,
    linePlan,
  );
  const { loading, error } = fieldResult;
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
  const field = validDomain?.field.shape.length === 3
    ? validDomain.field as typeof fieldResult.field
    : null;
  const displayGrid = validDomain?.grid ?? null;
  const displayFieldUnits = loadedStep?.fieldUnits;

  const validNr = validAxisLength(0, grid) ?? nr;
  const validNth = validAxisLength(1, grid) ?? nth;
  // Clamp index pickers when shape changes.
  useEffect(() => {
    if (validNr > 0) setRIdx((prev) => Math.min(prev, validNr - 1));
    if (validNth > 0) setThIdx((prev) => Math.min(prev, validNth - 1));
  }, [validNr, validNth]);

  const plot = useMemo(() => {
    if (!field || !fieldName) return null;
    const values = profile1D(field, 2, 0, 0);
    const zAxis = axisCoords(2, values.length, displayGrid);
    const zCoords = zAxis.values ?? Array.from(values, (_, k) => k);
    return buildPlot(
      values, zCoords, fieldName, zAxis, rIdx, thIdx, axis0Label, axis1Label,
      displayFieldUnits,
    );
  }, [field, fieldName, rIdx, thIdx, displayGrid, axis0Label, axis1Label, displayFieldUnits]);

  const maxR = Math.max(0, validNr - 1);
  const maxTh = Math.max(0, validNth - 1);

  return (
    <>
      <div className="plot-controls">
        <OutputHierarchyControls source={outputHierarchySource} />
        <ControlGroup label="Field">
          <select
            aria-label="Field"
            value={fieldName ?? ""}
            onChange={(e) => setFieldName(e.target.value || null)}
          >
            {profileFields.length === 0 && <option value="">(no rank-3 fields)</option>}
            {profileFields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </ControlGroup>

        <PlotCompressionControls step={loadedStep} fields={[fieldName]} />

        <ControlGroup label={`${axis0Label} index (${rIdx} / ${maxR})`} stretch>
          <input
            type="range"
            min={0}
            max={maxR}
            value={rIdx}
            onChange={(e) => setRIdx(Number(e.target.value))}
            disabled={!field}
          />
        </ControlGroup>

        <ControlGroup label={`${axis1Label} index (${thIdx} / ${maxTh})`} stretch>
          <input
            type="range"
            min={0}
            max={maxTh}
            value={thIdx}
            onChange={(e) => setThIdx(Number(e.target.value))}
            disabled={!field}
          />
        </ControlGroup>
      </div>

      <div className="plot-body">
        {loading && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>Loading…</span>
          </div>
        )}
        {error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{error}</span>
          </div>
        )}
        {plot ? (
          <PlotCanvas data={plot.data} layout={plot.layout} />
        ) : (
          !loading &&
          !error && (
            <div className="plot-overlay">
              <span>No data to plot.</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
}

function buildPlot(
  values: Float32Array,
  zCoords: number[],
  fieldName: string,
  /* The axis descriptor, not a "physical?" flag. `values != null` does NOT mean
   * kilometres: on a terrain-following grid axisCoords returns bare model-level
   * INDICES (with title "model level k"), so a boolean derived from it labels
   * level 30 as "30 km". The descriptor already carries the right title and
   * hover format — use them, as SlicePlot and TimeHeightPlot do. */
  zAxis: AxisCoords,
  rIdx: number,
  thIdx: number,
  axis0Label: string,
  axis1Label: string,
  fieldUnits: FieldUnits,
): BuiltPlot {
  const xTitle = labelWithUnit(fieldName, fieldUnits);
  const data: Data[] = [
    {
      type: "scatter",
      mode: "lines+markers",
      connectgaps: false,
      x: finiteProfileLine(values),
      y: zCoords,
      name: fieldName,
      line: { color: "#4a8cff", width: 1.8 },
      marker: { size: 3, color: "#4a8cff" },
      hovertemplate: `${fieldName}=%{x:.4g}<br>${zAxis.title}=%{y${zAxis.hoverFmt}}<extra></extra>`,
    },
  ];

  const layout: Partial<Layout> = {
    ...COMMON_LINE_LAYOUT,
    showlegend: false,
    xaxis: { title: { text: xTitle } },
    yaxis: { title: { text: zAxis.title } },
    title: {
      text: `${fieldName} vs z @ (${axis0Label}=${rIdx}, ${axis1Label}=${thIdx})`,
      font: { size: 13 },
    },
  };

  return { data, layout };
}
