import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import type { EnsembleAlignmentReport, EnsembleMemberInput } from "@/lib/comparison/ensembleAlignment";
import { computeEnsembleCellStatistics, withEnsembleThreshold, type EnsembleCellStatistics } from "@/lib/comparison/ensembleStatistics";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import { matchedPhysicalVolumes } from "@/lib/comparison/comparisonMetrics";
import { axisCoords } from "@/lib/science/gridCoords";
import { unitFor } from "@/lib/science/fieldUnits";
import { statisticLabel, thresholdLabel } from "@/lib/plot/plotLabels";
import { fieldsWithSharedPlotRank } from "@/lib/plot/fieldPlotEligibility";
import { slice2D, type Npy3D, type NpyN, type SliceAxis } from "@/lib/platform/npy";
import { DBZ_RANGE, isDbzField, resolveColorscale } from "@/lib/plot/plotColormaps";
import {
  readFieldAnyWindows,
  fieldDefinitionIdentity,
  registeredFieldRank,
} from "@/lib/run/runOutputs";
import { cropToValidPlotDomain, validAxisLength, type PlotGrid } from "@/lib/science/validDomain";
import {
  fieldSupportsBoundedWindows,
  describeBoundedSampling,
  PLOTLY_HEATMAP_CELL_CAP,
  planBoundedPlaneWindows,
  planBoundedSliceWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import "./EnsembleAnalysisPlot.css";

interface MemberPlane {
  flat: Float32Array;
  matrix: number[][];
  rows: number;
  cols: number;
  rowAxis: SliceAxis;
  colAxis: SliceAxis;
}

interface LoadedEnsemble {
  identity: string;
  scope: string;
  fieldName: string;
  fieldUnits: Record<string, string>;
  memberNames: string[];
  simulationTimeS: number;
  threshold: number;
  planes: MemberPlane[];
  statistics: EnsembleCellStatistics;
  /** Reference member's grid after halo crop — the one axisCoords must use. */
  grid: PlotGrid | null;
  samplingDisclosure: string | null;
}

const MEMBER_COLORS = ["#c56f49", "#5f8da6", "#7f9b72", "#bd9a49", "#96729b", "#638e87", "#b36e75", "#7482aa", "#a87a58", "#5884a0", "#8a9060", "#8f6d86"];

function defaultThreshold(field: string): number {
  if (field.includes("reflectivity")) return 30;
  if (field === "w") return 5;
  if (field.includes("vorticity")) return 0.01;
  return 0;
}

function memberPlane(field: NpyN, axis: SliceAxis, index: number): MemberPlane | null {
  if (field.shape.length === 3) {
    const volume = field as Npy3D;
    const sliced = slice2D(volume, axis, Math.max(0, Math.min(volume.shape[axis] - 1, index)));
    return {
      flat: Float32Array.from(sliced.data.flat()),
      matrix: sliced.data,
      rows: sliced.rows,
      cols: sliced.cols,
      rowAxis: sliced.rowAxis as SliceAxis,
      colAxis: sliced.colAxis as SliceAxis,
    };
  }
  if (field.shape.length !== 2) return null;
  const [rows, cols] = field.shape;
  return {
    flat: field.data,
    matrix: Array.from({ length: rows }, (_, row) => Array.from(field.data.subarray(row * cols, (row + 1) * cols))),
    rows,
    cols,
    rowAxis: 0,
    colAxis: 1,
  };
}

function matrix(values: Float32Array, rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, row) => Array.from(values.subarray(row * cols, (row + 1) * cols)));
}

function finiteRange(values: Float32Array, fallback: [number, number] = [0, 1]): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  if (!Number.isFinite(min)) return fallback;
  if (min === max) return [min, min + Math.max(Math.abs(min) * 1e-6, 1e-12)];
  return [min, max];
}

function format(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 0.001 || Math.abs(value) >= 10_000 ? value.toExponential(3) : value.toPrecision(4);
}

export function EnsembleAnalysisPlot({
  members,
  alignment,
  referenceFrameIndex,
}: {
  members: readonly EnsembleMemberInput[];
  alignment: EnsembleAlignmentReport;
  referenceFrameIndex: number;
}) {
  const timeMatch = alignment.timeMatches.find((match) => match.referenceFrameIndex === referenceFrameIndex) ?? null;
  const referenceStep = timeMatch ? members[0]?.steps[timeMatch.memberFrameIndices[0]] ?? null : null;
  const plotEligibleFields = useMemo(() => fieldsWithSharedPlotRank(
    alignment.eligibleFields,
    !timeMatch ? [] : members.map((member, memberIndex) => ({
      runDir: member.source.locator.path,
      step: member.steps[timeMatch.memberFrameIndices[memberIndex]],
    })),
    [2, 3],
  ), [alignment.eligibleFields, members, timeMatch]);
  const [fieldName, setFieldName] = useState(() => plotEligibleFields.includes("reflectivity_dbz") ? "reflectivity_dbz" : plotEligibleFields[0] ?? "");
  const [sliceAxis, setSliceAxis] = useState<SliceAxis>(2);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [threshold, setThreshold] = useState(() => defaultThreshold(fieldName));
  const [thresholdInput, setThresholdInput] = useState(() => String(defaultThreshold(fieldName)));
  const thresholdRef = useRef(threshold);
  thresholdRef.current = threshold;
  const [loaded, setLoaded] = useState<LoadedEnsemble | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedRank = fieldName && referenceStep
    ? registeredFieldRank(
        members[0]?.source.locator.path ?? "",
        referenceStep.index,
        fieldName,
        referenceStep.publicationGeneration,
      )
    : null;
  // Bound the slider by the VALID extent. The planes below are cropped, so a
  // slider sized off the manifest's nr/nth/nz ranges over halo indices the
  // cropped volume does not have — memberPlane clamps, and the top haloWidth
  // positions silently redisplay the last valid slice under a changing label.
  const axisLength = selectedRank === 3 && referenceStep?.grid
    ? validAxisLength(sliceAxis, referenceStep.grid)
      ?? [referenceStep.grid.nr, referenceStep.grid.nth, referenceStep.grid.nz][sliceAxis]
      ?? 1
    : 1;
  // The selected field's unit, for every readout that reports a value in it.
  // Spread and mean carry the field's units; only the exceedance percentage is
  // dimensionless.
  const fieldUnit = referenceStep ? unitFor(fieldName, referenceStep.fieldUnits) : null;
  const scope = useMemo(() => JSON.stringify([fieldName, fieldUnit, sliceAxis, sliceIndex,
    members.map(member => datasetSourceIdentity(member.source)),
    timeMatch ? members.map((member, index) => {
      const step = member.steps[timeMatch.memberFrameIndices[index]];
      return fieldDefinitionIdentity(member.source.locator.path, step?.index ?? null, fieldName, step?.publicationGeneration ?? null);
    }) : null]), [fieldName, fieldUnit, members, sliceAxis, sliceIndex, timeMatch]);
  const requestIdentity = useMemo(() => JSON.stringify([scope, timeMatch?.simulationTimeS,
    timeMatch ? members.map((member, index) => {
      const step = member.steps[timeMatch.memberFrameIndices[index]];
      return [step?.index, step?.publicationGeneration, step?.grid, step?.fieldUnits?.[fieldName]];
    }) : null]), [scope, timeMatch, members, fieldName]);

  useEffect(() => {
    if (fieldName && plotEligibleFields.includes(fieldName)) return;
    setFieldName(plotEligibleFields.includes("reflectivity_dbz") ? "reflectivity_dbz" : plotEligibleFields[0] ?? "");
  }, [fieldName, plotEligibleFields]);
  useEffect(() => setSliceIndex((current) => Math.min(current, Math.max(0, axisLength - 1))), [axisLength]);
  useEffect(() => {
    setThreshold(defaultThreshold(fieldName));
    setThresholdInput(String(defaultThreshold(fieldName)));
  }, [fieldName]);
  useEffect(() => {
    if (!timeMatch || !fieldName) { setLoaded(null); setLoading(false); setError(null); return; }
    let cancelled = false;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    const memberNames = members.map(member => member.source.displayName);
    void Promise.all(members.map(async (member, memberIndex) => {
      const step = member.steps[timeMatch.memberFrameIndices[memberIndex]];
      if (!step) throw new Error(`Member ${memberIndex + 1} has no frame at ${timeMatch.simulationTimeS} s.`);
      const rank = registeredFieldRank(
        member.source.locator.path,
        step.index,
        fieldName,
        step.publicationGeneration,
      );
      const perMemberCells = Math.max(
        65_536,
        Math.floor(PLOTLY_HEATMAP_CELL_CAP / Math.max(1, members.length)),
      );
      if ((rank !== 2 && rank !== 3)
          || !fieldSupportsBoundedWindows(step.fieldOrigins?.[fieldName])) {
        throw new Error(`${fieldName} has no bounded rank-2/3 ensemble reader.`);
      }
      const outcome = rank === 3
        ? planBoundedSliceWindows(step.grid, sliceAxis, sliceIndex, perMemberCells)
        : planBoundedPlaneWindows(step.grid, perMemberCells);
      if (!outcome.ok) throw new Error(outcome.reason);
      const windows = windowsForFieldAggregation(outcome.plan.windows, fieldName);
      const raw = await readFieldAnyWindows(
        member.source.locator.path,
        step.index,
        fieldName,
        step.publicationGeneration,
        windows,
        { signal: controller.signal },
      );
      const domain = cropToValidPlotDomain(raw, step.grid);
      return {
        field: domain.field,
        plane: memberPlane(domain.field, sliceAxis, 0),
        grid: domain.grid,
        sampled: outcome.plan,
      };
    })).then((members) => {
      if (cancelled) return;
      const planes = members.map((member) => member.plane);
      if (planes.some((plane) => plane == null)) throw new Error(`${fieldName} is not a qualified 2-D or 3-D ensemble field.`);
      const qualified = planes as MemberPlane[];
      const first = qualified[0];
      if (qualified.some((plane) => plane.rows !== first.rows || plane.cols !== first.cols || plane.rowAxis !== first.rowAxis || plane.colAxis !== first.colAxis)) {
        throw new Error("Ensemble member slices changed geometry after alignment.");
      }
      // Terrain-following array indices are not a physical registration.
      // Verify exact cell centroids and volumes before aggregating by cell.
      for (let index = 1; index < members.length; index += 1) {
        matchedPhysicalVolumes(members[0].field, members[index].field);
      }
      setLoaded({
        identity: requestIdentity, scope, fieldName, fieldUnits: referenceStep?.fieldUnits ?? {},
        memberNames,
        simulationTimeS: timeMatch.simulationTimeS,
        threshold: thresholdRef.current,
        planes: qualified,
        statistics: computeEnsembleCellStatistics(qualified.map((plane) => plane.flat), thresholdRef.current),
        grid: members[0].grid,
        samplingDisclosure: describeBoundedSampling(members[0].sampled, fieldName),
      });
    }).catch((reason: unknown) => {
      if (!cancelled) { setLoaded(null); setError(reason instanceof Error ? reason.message : String(reason)); }
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; controller.abort(); };
  }, [fieldName, members, sliceAxis, sliceIndex, timeMatch, requestIdentity, scope, referenceStep?.fieldUnits]);

  const displayed = loaded?.scope === scope && timeMatch ? loaded : null;
  const displayedStatistics = useMemo(() => !displayed ? null : displayed.threshold === threshold ? displayed.statistics
    : withEnsembleThreshold(displayed.planes.map(plane => plane.flat), displayed.statistics, threshold), [displayed, threshold]);
  const requestPending = Boolean(timeMatch && fieldName) && (loading || loaded?.identity !== requestIdentity);
  const thresholdInvalid = thresholdInput.trim() === "" || !Number.isFinite(Number(thresholdInput));

  const plot = useMemo((): { data: Data[]; layout: Partial<Layout> } | null => {
    if (!displayed || !displayedStatistics) return null;
    const { planes } = displayed;
    const statistics = displayedStatistics;
    const first = planes[0];
    const x = axisCoords(first.colAxis, first.cols, displayed.grid);
    const y = axisCoords(first.rowAxis, first.rows, displayed.grid);
    // One decision, used for both the ramp and its range: the NWS colours are anchored AT dBZ
    // values, so a fitted range moves every colour to the wrong reflectivity. Unit-driven via
    // isDbzField like the rest of the studio, rather than a second independent name test.
    const anchoredDbz = isDbzField(displayed.fieldName, displayed.fieldUnits);
    const meanRange = anchoredDbz ? [DBZ_RANGE[0], DBZ_RANGE[1]] as [number, number] : finiteRange(statistics.mean);
    const spreadRange = finiteRange(statistics.spread);
      // Mean/spread carry the field's units; probability is dimensionless (%).
    const units = displayed.fieldUnits;
    const hoverUnit = unitFor(fieldName, units) ? ` ${unitFor(fieldName, units)}` : "";
    const data: Data[] = [
      { type: "heatmap", zsmooth: false, z: matrix(statistics.mean, first.rows, first.cols), x: x.values, y: y.values, coloraxis: "coloraxis", name: "Ensemble mean", hovertemplate: `Mean=%{z:.4g}${hoverUnit}<extra></extra>` } as Data,
      { type: "heatmap", zsmooth: false, z: matrix(statistics.spread, first.rows, first.cols), x: x.values, y: y.values, xaxis: "x2", yaxis: "y2", coloraxis: "coloraxis2", name: "Sample spread", hovertemplate: `Spread=%{z:.4g}${hoverUnit}<extra></extra>` } as Data,
      { type: "heatmap", zsmooth: false, z: matrix(statistics.probabilityPercent, first.rows, first.cols), x: x.values, y: y.values, xaxis: "x3", yaxis: "y3", coloraxis: "coloraxis3", name: "Exceedance probability", hovertemplate: `P(>${threshold}${hoverUnit})=%{z:.1f}%<extra></extra>` } as Data,
    ];
    planes.forEach((plane, index) => data.push({
      type: "contour", z: plane.matrix, x: x.values, y: y.values, xaxis: "x4", yaxis: "y4",
      name: displayed.memberNames[index], showscale: false, hoverinfo: "skip",
      contours: { coloring: "none", start: threshold, end: threshold, size: Math.max(Math.abs(threshold) * 1e-6, 1e-9), showlabels: false },
      line: { color: MEMBER_COLORS[index % MEMBER_COLORS.length], width: index === 0 ? 2.2 : 1.25, smoothing: 0 },
    } as Data));
    const axis = (domain: [number, number], title: string, matches?: string) => ({ domain, title: { text: title }, ...(matches ? { matches } : {}) });
    return {
      data,
      layout: ({
        // Both rows 0.43 tall so matches:"y" enforces the same data range and visual scale.
        xaxis: axis([0, 0.43], x.title), yaxis: axis([0.57, 1], y.title),
        xaxis2: axis([0.57, 1], x.title, "x"), yaxis2: axis([0.57, 1], y.title, "y"),
        xaxis3: axis([0, 0.43], x.title, "x"), yaxis3: axis([0, 0.43], y.title, "y"),
        xaxis4: axis([0.57, 1], x.title, "x"), yaxis4: axis([0, 0.43], y.title, "y"),
        coloraxis: { colorscale: resolveColorscale(anchoredDbz ? "dBZ" : "Viridis") as never, cmin: meanRange[0], cmax: meanRange[1], colorbar: { title: { text: statisticLabel("Mean", fieldName, units), side: "right" }, x: 0.445, y: 0.79, len: 0.38 } },
        coloraxis2: { colorscale: resolveColorscale("Viridis") as never, cmin: Math.max(0, spreadRange[0]), cmax: spreadRange[1], colorbar: { title: { text: statisticLabel("Spread", fieldName, units), side: "right" }, x: 1.01, y: 0.79, len: 0.38 } },
        coloraxis3: { colorscale: resolveColorscale("Viridis") as never, cmin: 0, cmax: 100, colorbar: { title: { text: "Exceedance (%)", side: "right" }, x: 0.445, y: 0.215, len: 0.38 } },
        annotations: [
          { text: "Mean", x: 0.215, y: 1.055, xref: "paper", yref: "paper", showarrow: false },
          { text: "Sample spread · N−1", x: 0.785, y: 1.055, xref: "paper", yref: "paper", showarrow: false },
          { text: thresholdLabel(threshold, fieldName, units), x: 0.215, y: 0.485, xref: "paper", yref: "paper", showarrow: false },
          { text: `${members.length} member contours`, x: 0.785, y: 0.485, xref: "paper", yref: "paper", showarrow: false },
        ],
        legend: { orientation: "h", x: 0.56, y: -0.12, font: { size: 10 } },
        margin: { l: 58, r: 72, t: 50, b: 64 }, dragmode: "zoom",
        uirevision: `ensemble:${fieldName}:${sliceAxis}`,
      } as unknown as Partial<Layout>),
    };
  }, [fieldName, displayed, displayedStatistics, members.length, sliceAxis, threshold]);

  return (
    <section className="ensemble-analysis" aria-label="Ensemble mean spread probability and spaghetti analysis">
      <header className="ensemble-analysis__header">
        <div><span>Qualified ensemble</span><strong>Distribution board</strong><p>Mean · sample spread · exceedance probability · member contours</p></div>
        <div className="ensemble-analysis__identity"><b>{members.length}</b><span>members</span><b>{(displayed?.simulationTimeS ?? timeMatch?.simulationTimeS)?.toFixed(3) ?? "—"}</b><span>seconds</span></div>
      </header>
      <div className="ensemble-analysis__controls">
        <label><span>Canonical field</span><select value={fieldName} onChange={(event) => setFieldName(event.target.value)}>{plotEligibleFields.map((field) => <option key={field} value={field}>{field}</option>)}</select></label>
        {selectedRank === 3 && <div role="group" aria-label="Ensemble slice axis"><span>Plane normal</span>{([0, 1, 2] as SliceAxis[]).map((axis) => <button key={axis} type="button" aria-pressed={sliceAxis === axis} onClick={() => setSliceAxis(axis)}>{["x", "y", "z"][axis]}</button>)}</div>}
        {selectedRank === 3 && <label><span>Slice index</span><input type="range" min={0} max={Math.max(0, axisLength - 1)} value={sliceIndex} onChange={(event) => setSliceIndex(Number(event.target.value))} /><b>{sliceIndex}</b></label>}
        <label><span>Exceedance / contour{fieldUnit ? ` (${fieldUnit})` : ""}</span><input aria-label={fieldUnit ? `Ensemble exceedance threshold in ${fieldUnit}` : "Ensemble exceedance threshold"}
          type="text" inputMode="decimal" value={thresholdInput} aria-invalid={thresholdInvalid || undefined}
          onChange={(event) => {
            const input = event.target.value;
            setThresholdInput(input);
            if (input.trim() !== "" && Number.isFinite(Number(input))) setThreshold(Number(input));
          }} /></label>
      </div>
      <div className="ensemble-analysis__contract"><Icon name="info" size={12} /><span>Strict common finite sample at every cell · sample standard deviation (N−1) · probability counts members strictly above threshold · no interpolation or unit conversion</span></div>
      {thresholdInvalid && <div className="ensemble-analysis__notice" role="status"><span>Enter a finite threshold. The displayed results retain the previous valid threshold of {threshold}{fieldUnit ? ` ${fieldUnit}` : ""}.</span></div>}
      {displayedStatistics && <div className="ensemble-analysis__metrics" aria-label="Ensemble aggregate statistics"><span>Complete cells <b>{displayedStatistics.completeCellCount.toLocaleString()}</b> / {displayedStatistics.totalCellCount.toLocaleString()}</span><span>Mean spread <b>{format(displayedStatistics.domainMeanSpread)}</b>{fieldUnit && ` ${fieldUnit}`}</span><span>Maximum spread <b>{format(displayedStatistics.maximumSpread)}</b>{fieldUnit && ` ${fieldUnit}`}</span><span>Domain exceedance <b>{displayedStatistics.domainExceedancePercent.toFixed(1)}%</b></span></div>}
      {!timeMatch && <div className="ensemble-analysis__notice"><Icon name="warning" size={12} /><span>The selected reference frame is not present across every member. Choose one of the {alignment.timeMatches.length} common physical times.</span></div>}
      <div className="ensemble-analysis__plot plot-notice-layout">
        {requestPending && !error && <div className="plot-overlay"><Icon name="dot" size={12} /><span>{displayed
          ? `Showing previous completed ${displayed.fieldName} ensemble at ${displayed.simulationTimeS} s while loading ${members.length} aligned member slices…`
          : `Loading ${members.length} aligned member slices…`}</span></div>}
        {error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{error}</span></div>}
        {displayed?.samplingDisclosure && <div className="plot-overlay"><Icon name="dot" size={12} /><span>{displayed.samplingDisclosure} identically across all members</span></div>}
        {plot && !error && <PlotCanvas data={plot.data} layout={plot.layout} className="ensemble-analysis__canvas" />}
      </div>
    </section>
  );
}
