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
  usePlotChrome,
  useRetainedPlot,
  type PlotChrome,
  type PlotComponentProps,
} from "./common";
import { axisCoords, axisLabel, type AxisCoords } from "@/lib/science/gridCoords";
import { findPublishedFrame, loadedFramesAreCoherent } from "@/lib/run/frameIdentity";
import { cropToValidPlotDomain, validAxisLength } from "@/lib/science/validDomain";
import type { Npy3D } from "@/lib/platform/npy";
import { unitFor } from "@/lib/science/fieldUnits";
import { planExactLineWindows, windowsForFieldAggregation } from "@/lib/plot/fieldLoadPlanning";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { finiteHodographLine } from "@/lib/plot/finiteProfileLine";
import { canonicalHistoryUnit } from "@/lib/plot/historyReductionCache";

/* Wind hodograph: u(z) vs v(z), height encoded as color. NWS/WK1982 convention. Fields: u, v per field_contract. */

export function Hodograph({
  runDir: rootRunDir,
  steps: rootSteps,
  stepIdx: rootStepIdx,
}: PlotComponentProps) {
  const outputHierarchySource = useOutputHierarchySource({
    runDir: rootRunDir,
    steps: rootSteps,
    stepIdx: rootStepIdx,
    scope: "shared",
    requiredFields: ["u", "v"],
  });
  const runDir = outputHierarchySource.sourceRunDir;
  const steps = outputHierarchySource.sourceSteps;
  const currentStep = outputHierarchySource.currentStep;
  const fieldsForStep = currentStep?.fields ?? [];
  const volumeFields = fieldsForPlotRanks(runDir, currentStep, [3]);
  const grid = currentStep?.grid ?? null;
  // Index-picker axis letters for the Cartesian grid: x/y/z.
  const axis0Label = axisLabel(0, grid);
  const axis1Label = axisLabel(1, grid);
  const hasU = volumeFields.includes("u");
  const hasV = volumeFields.includes("v");

  const [rIdx, setRIdx] = useState(0);
  const [thIdx, setThIdx] = useState(0);
  const chrome = usePlotChrome();

  // Debounce the paired exact-column reads so a scrub/autoplay does not request
  // every intermediate frame. Recording bypasses the debounce.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const linePlan = useMemo(() => {
    const outcome = planExactLineWindows(grid, 2, [rIdx, thIdx, 0]);
    return outcome.ok ? outcome.plan.windows : null;
  }, [grid, rIdx, thIdx]);
  const u = useField(
    runDir,
    linePlan && hasU ? loadFrame.stepIndex : null,
    hasU ? "u" : null,
    loadFrame.publicationGeneration,
    linePlan ? windowsForFieldAggregation(linePlan, "u") : null,
  );
  const v = useField(
    runDir,
    linePlan && hasV ? loadFrame.stepIndex : null,
    hasV ? "v" : null,
    loadFrame.publicationGeneration,
    linePlan ? windowsForFieldAggregation(linePlan, "v") : null,
  );
  const coherentFrame = loadedFramesAreCoherent(u, v);
  const loadedStep = coherentFrame
    ? findPublishedFrame(steps, u.loadedStepIndex, u.loadedPublicationGeneration)
    : null;
  const uDomain = useMemo(
    () => coherentFrame && loadedStep && u.field
      ? cropToValidPlotDomain(u.field, loadedStep?.grid)
      : null,
    [coherentFrame, u.field, loadedStep],
  );
  const vDomain = useMemo(
    () => coherentFrame && loadedStep && v.field
      ? cropToValidPlotDomain(v.field, loadedStep?.grid)
      : null,
    [coherentFrame, v.field, loadedStep],
  );
  const uField = uDomain?.field as Npy3D | undefined;
  const vField = vDomain?.field as Npy3D | undefined;
  const displayGrid = uDomain?.grid ?? null;
  const uUnit = unitFor("u", loadedStep?.fieldUnits);
  const vUnit = unitFor("v", loadedStep?.fieldUnits);
  const incompatibleUnits = uUnit != null && vUnit != null
    && canonicalHistoryUnit(uUnit) !== canonicalHistoryUnit(vUnit);

  const validNr = validAxisLength(0, grid) ?? grid?.nr ?? 0;
  const validNth = validAxisLength(1, grid) ?? grid?.nth ?? 0;
  // Clamp index pickers when shape changes.
  useEffect(() => {
    if (validNr > 0) setRIdx((prev) => Math.min(prev, validNr - 1));
    if (validNth > 0) setThIdx((prev) => Math.min(prev, validNth - 1));
  }, [validNr, validNth]);

  const plot = useMemo(() => {
    if (!uField || !vField || incompatibleUnits) return null;
    if (
      uField.shape[0] !== vField.shape[0] ||
      uField.shape[1] !== vField.shape[1] ||
      uField.shape[2] !== vField.shape[2]
    ) {
      return null;
    }
    const uVec = profile1D(uField, 2, 0, 0);
    const vVec = profile1D(vField, 2, 0, 0);
    const zAxis = axisCoords(2, uVec.length, displayGrid);
    const zCoords = zAxis.values ?? Array.from(uVec, (_, k) => k);
    const xAxis = axisCoords(0, uField.shape[0], displayGrid);
    const yAxis = axisCoords(1, uField.shape[1], displayGrid);
    const xPosition = xAxis.values?.[0];
    const yPosition = yAxis.values?.[0];
    return buildPlot(
      uVec,
      vVec,
      zCoords,
      zAxis,
      rIdx,
      thIdx,
      axis0Label,
      axis1Label,
      uUnit,
      vUnit,
      xPosition,
      yPosition,
      chrome,
    );
  }, [
    uField,
    vField,
    rIdx,
    thIdx,
    displayGrid,
    axis0Label,
    axis1Label,
    uUnit,
    vUnit,
    incompatibleUnits,
    chrome,
  ]);

  // u and v resolve independently, so the coherence gate above nulls the figure on every frame swap.
  const { shown: shownPlot } = useRetainedPlot(plot, `${runDir}|${rIdx}|${thIdx}|${canonicalHistoryUnit(unitFor("u", currentStep?.fieldUnits))}|${canonicalHistoryUnit(unitFor("v", currentStep?.fieldUnits))}`);

  const loading = u.loading || v.loading;
  const error = u.error ?? v.error;
  const missing = !hasU || !hasV;
  const maxR = Math.max(1, validNr);
  const maxTh = Math.max(1, validNth);

  return (
    <>
      <div className="plot-controls">
        <OutputHierarchyControls source={outputHierarchySource} />
        <PlotCompressionControls step={loadedStep} fields={["u", "v"]} />
        <ControlGroup label={`${axis0Label} index (${rIdx} / ${maxR - 1})`} stretch>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxR - 1)}
            value={rIdx}
            onChange={(e) => setRIdx(Number(e.target.value))}
            disabled={!uField || !vField}
          />
        </ControlGroup>

        <ControlGroup label={`${axis1Label} index (${thIdx} / ${maxTh - 1})`} stretch>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxTh - 1)}
            value={thIdx}
            onChange={(e) => setThIdx(Number(e.target.value))}
            disabled={!uField || !vField}
          />
        </ControlGroup>
      </div>

      <div className="plot-body">
        {missing && (
          <div className="plot-overlay plot-overlay--warn">
            <Icon name="warning" size={12} />
            <span>
              Need both <code>u</code> and <code>v</code> fields. Available:{" "}
              {fieldsForStep.join(", ") || "(none)"}
            </span>
          </div>
        )}
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
        {incompatibleUnits && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>Hodograph components have incompatible units: u ({uUnit}), v ({vUnit})</span>
          </div>
        )}
        {shownPlot ? (
          <PlotCanvas data={shownPlot.data} layout={shownPlot.layout} />
        ) : (
          !loading &&
          !error &&
          !incompatibleUnits &&
          !missing && (
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
  uVec: Float32Array,
  vVec: Float32Array,
  zCoords: number[],
  /* The axis descriptor, not a "physical?" flag: on a terrain-following grid
   * axisCoords returns bare model-level INDICES, so a `values != null` boolean
   * would colour-bar level 30 as "30 km". See VerticalProfile. */
  zAxis: AxisCoords,
  rIdx: number,
  thIdx: number,
  axis0Label: string,
  axis1Label: string,
  uUnit: string | null,
  vUnit: string | null,
  xPositionKm: number | undefined,
  yPositionKm: number | undefined,
  chrome: PlotChrome,
): BuiltPlot {
  const { x: u, y: v, magnitude: absMax } = finiteHodographLine(uVec, vVec);
  const uSuffix = uUnit ? ` ${uUnit}` : "";
  const vSuffix = vUnit ? ` ${vUnit}` : "";
  const heightLabel = zAxis.title;
  // u/v are the horizontal (x/y) winds — the usual earth-relative hodograph.
  const uName = "u";
  const vName = "v";

  const data: Data[] = [
    {
      type: "scatter",
      mode: "lines+markers",
      connectgaps: false,
      x: u,
      y: v,
      name: "hodograph",
      line: { color: "#4a8cff", width: 1.5 },
      marker: {
        size: 6,
        color: zCoords,
        colorscale: "Viridis",
        showscale: true,
        colorbar: { title: { text: heightLabel, side: "right" } },
      },
      hovertemplate: `${uName}=%{x:.2f}${uSuffix}<br>${vName}=%{y:.2f}${vSuffix}<br>${heightLabel}=%{marker.color${zAxis.hoverFmt}}<extra></extra>`,
    },
  ];

  // Equal-aspect axes so vector magnitudes read correctly.
  const pad = absMax * 0.1;
  const range: [number, number] = [-(absMax + pad), absMax + pad];

  const layout: Partial<Layout> = {
    ...COMMON_LINE_LAYOUT,
    showlegend: false,
    xaxis: {
      title: { text: uUnit ? `${uName} (${uUnit})` : uName },
      zeroline: true,
      zerolinecolor: chrome.zeroline,
      range,
      // Equal-aspect: compress the plot area (domain) into a square rather than
      // extending the range, so a wide panel renders a true square hodograph
      // (wind vectors read correctly) instead of a horizontally-stretched one.
      constrain: "domain",
    },
    yaxis: {
      title: { text: vUnit ? `${vName} (${vUnit})` : vName },
      zeroline: true,
      zerolinecolor: chrome.zeroline,
      range,
      scaleanchor: "x",
      scaleratio: 1,
      constrain: "domain",
    },
    title: {
      text: xPositionKm != null && yPositionKm != null
        ? `Hodograph @ (${axis0Label}=${xPositionKm.toFixed(2)} km, ${axis1Label}=${yPositionKm.toFixed(2)} km)`
        : `Hodograph @ (${axis0Label} index=${rIdx}, ${axis1Label} index=${thIdx})`,
      font: { size: 13 },
    },
  };

  return { data, layout };
}
