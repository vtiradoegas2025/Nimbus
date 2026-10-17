import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import { axisCoords } from "@/lib/science/gridCoords";
import { slice2D, type Npy3D, type NpyN, type SliceAxis } from "@/lib/platform/npy";
import { DBZ_RANGE, resolveColorscale } from "@/lib/plot/plotColormaps";
import type { StepInfo } from "@/lib/run/runOutputs";
import {
  fieldSupportsBoundedWindows,
  describeBoundedSampling,
  planBoundedSliceWindows,
  windowsForFieldAggregation,
} from "@/lib/plot/fieldLoadPlanning";
import { gridForWindowedField } from "@/lib/science/windowedGrid";
import { remapTerrainVerticalSlice } from "@/lib/science/terrainSliceRemap";
import { matchedPhysicalVolumes } from "@/lib/comparison/comparisonMetrics";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import { loadedFramesAreCoherent } from "@/lib/run/frameIdentity";
import { labelWithUnit, unitFor } from "@/lib/science/fieldUnits";
import { useFieldAny, useRetainedPlot } from "./common";

interface Plane { data: number[][]; rows: number; cols: number; rowAxis: SliceAxis; colAxis: SliceAxis }

function plane(field: NpyN | null, axis: SliceAxis, index: number): Plane | null {
  if (!field || field.shape.length !== 3) return null;
  const volume = field as Npy3D;
  const sliced = slice2D(volume, axis, Math.min(index, volume.shape[axis] - 1));
  return { data: sliced.data, rows: sliced.rows, cols: sliced.cols, rowAxis: sliced.rowAxis as SliceAxis, colAxis: sliced.colAxis as SliceAxis };
}

function difference(left: Plane, right: Plane): Plane | null {
  if (left.rows !== right.rows || left.cols !== right.cols) return null;
  return {
    ...left,
    data: left.data.map((row, i) => row.map((value, j) => Number.isFinite(value) && Number.isFinite(right.data[i][j]) ? value - right.data[i][j] : Number.NaN)),
  };
}

function magnitude(...planes: Plane[]): number {
  let maximum = 0;
  for (const item of planes) for (const row of item.data) for (const value of row) if (Number.isFinite(value)) maximum = Math.max(maximum, Math.abs(value));
  return Math.max(maximum, 1e-12);
}

/** Caption for a contour layer drawn as `start: -span, end: span, size: span/4`
 *  — i.e. five levels one quarter-span apart. */
function contourStep(span: number): string {
  return `±${span.toPrecision(2)} / ${(span / 4).toPrecision(2)}`;
}

export function ComparisonCompositePlot({
  leftRunDir,
  leftLabel,
  leftStep,
  rightRunDir,
  rightLabel,
  rightStep,
  sliceAxis,
  sliceIndex,
}: {
  leftRunDir: string;
  leftLabel: string;
  leftStep: StepInfo | null;
  rightRunDir: string;
  rightLabel: string;
  rightStep: StepInfo | null;
  sliceAxis: SliceAxis;
  sliceIndex: number;
}) {
  const terrainFollowing = leftStep?.grid?.terrainFollowing === true
    || rightStep?.grid?.terrainFollowing === true;
  const terrainBlocked = false;
  const compositeFields = ["reflectivity_dbz", "w", "vorticity_z"] as const;
  const boundedSupported = compositeFields.every((field) =>
    fieldSupportsBoundedWindows(leftStep?.fieldOrigins?.[field])
    && fieldSupportsBoundedWindows(rightStep?.fieldOrigins?.[field])
  );
  const plans = useMemo(() => {
    if (terrainBlocked || !boundedSupported) return null;
    const left = planBoundedSliceWindows(leftStep?.grid, sliceAxis, sliceIndex);
    const right = planBoundedSliceWindows(rightStep?.grid, sliceAxis, sliceIndex);
    if (!left.ok || !right.ok || left.plan.sampledShape.join("x") !== right.plan.sampledShape.join("x")) {
      return null;
    }
    return {
      left: left.plan.windows,
      right: right.plan.windows,
      samplingDisclosure: describeBoundedSampling(left.plan, "reflectivity_dbz"),
    };
  }, [boundedSupported, leftStep?.grid, rightStep?.grid, sliceAxis, sliceIndex, terrainBlocked]);
  const enabled = plans != null;
  const useCompositeField = (side: "left" | "right", field: typeof compositeFields[number]) => {
    const step = side === "left" ? leftStep : rightStep;
    const runDir = side === "left" ? leftRunDir : rightRunDir;
    const windows = plans ? windowsForFieldAggregation(plans[side], field) : null;
    return useFieldAny(runDir, enabled ? step?.index ?? null : null, field, step?.publicationGeneration ?? null, windows);
  };
  const leftReflectivity = useCompositeField("left", "reflectivity_dbz");
  const leftW = useCompositeField("left", "w");
  const leftVorticity = useCompositeField("left", "vorticity_z");
  const rightReflectivity = useCompositeField("right", "reflectivity_dbz");
  const rightW = useCompositeField("right", "w");
  const rightVorticity = useCompositeField("right", "vorticity_z");
  // Each side's three fields resolve independently and useFieldAny holds the previous frame while the
  // next loads, so without this gate a panel could pair one run's step N reflectivity with its step
  // N+1 vertical velocity and label the result a single frame. Sides are checked apart: they are
  // different runs, so their step indices and publication generations are unrelated.
  const leftCoherent = loadedFramesAreCoherent(leftReflectivity, leftW, leftVorticity);
  const rightCoherent = loadedFramesAreCoherent(rightReflectivity, rightW, rightVorticity);
  const coherentFrames = leftCoherent && rightCoherent;

  // The fill is always reflectivity here; take its unit from the manifest rather than assuming dBZ,
  // and name the quantity so the bar reads like every other colorbar in the studio.
  const reflectivityLabel = labelWithUnit("reflectivity_dbz", leftStep?.fieldUnits);
  const reflectivityUnit = unitFor("reflectivity_dbz", leftStep?.fieldUnits);
  const differenceLabel = reflectivityUnit ? `A−B (${reflectivityUnit})` : "A−B";

  const fields = useMemo(() => {
    if (!coherentFrames) {
      return { lr: null, lw: null, lv: null, rr: null, rw: null, rv: null, grid: null, remapped: false, error: null as string | null };
    }
    const raw = {
      lr: leftReflectivity.field,
      lw: leftW.field,
      lv: leftVorticity.field,
      rr: rightReflectivity.field,
      rw: rightW.field,
      rv: rightVorticity.field,
    };
    const leftGrid = raw.lr ? gridForWindowedField(raw.lr, leftStep?.grid) : null;
    const rightGrid = raw.rr ? gridForWindowedField(raw.rr, rightStep?.grid) : null;
    if (!raw.lr || !raw.lw || !raw.lv || !raw.rr || !raw.rw || !raw.rv) {
      return { ...raw, grid: leftGrid, remapped: false, error: null as string | null };
    }
    try {
      if (terrainFollowing) {
        matchedPhysicalVolumes(raw.lr, raw.rr);
        matchedPhysicalVolumes(raw.lw, raw.rw);
        matchedPhysicalVolumes(raw.lv, raw.rv);
      }
      const lr = remapTerrainVerticalSlice(raw.lr, leftGrid, sliceAxis);
      const lw = remapTerrainVerticalSlice(raw.lw, leftGrid, sliceAxis);
      const lv = remapTerrainVerticalSlice(raw.lv, leftGrid, sliceAxis);
      const rr = remapTerrainVerticalSlice(raw.rr, rightGrid, sliceAxis);
      const rw = remapTerrainVerticalSlice(raw.rw, rightGrid, sliceAxis);
      const rv = remapTerrainVerticalSlice(raw.rv, rightGrid, sliceAxis);
      return {
        lr: lr.field, lw: lw.field, lv: lv.field,
        rr: rr.field, rw: rw.field, rv: rv.field,
        grid: lr.grid,
        remapped: lr.remapped || rr.remapped,
        error: null as string | null,
      };
    } catch (remapError) {
      return {
        lr: null, lw: null, lv: null, rr: null, rw: null, rv: null,
        grid: leftGrid,
        remapped: false,
        error: remapError instanceof Error ? remapError.message : String(remapError),
      };
    }
  }, [coherentFrames, leftReflectivity.field, leftStep?.grid, leftVorticity.field, leftW.field, rightReflectivity.field, rightStep?.grid, rightVorticity.field, rightW.field, sliceAxis, terrainFollowing]);

  const plot = useMemo((): { data: Data[]; layout: Partial<Layout> } | null => {
    const planes = {
      lr: plane(fields.lr, sliceAxis, sliceIndex), lw: plane(fields.lw, sliceAxis, sliceIndex), lv: plane(fields.lv, sliceAxis, sliceIndex),
      rr: plane(fields.rr, sliceAxis, sliceIndex), rw: plane(fields.rw, sliceAxis, sliceIndex), rv: plane(fields.rv, sliceAxis, sliceIndex),
    };
    if (Object.values(planes).some((value) => !value)) return null;
    const p = planes as Record<keyof typeof planes, Plane>;
    const dr = difference(p.lr, p.rr); const dw = difference(p.lw, p.rw); const dv = difference(p.lv, p.rv);
    if (!dr || !dw || !dv) return null;
    const x = axisCoords(p.lr.colAxis, p.lr.cols, fields.grid);
    const y = axisCoords(p.lr.rowAxis, p.lr.rows, fields.grid);
    const wMagnitude = magnitude(p.lw, p.rw);
    const vorticityMagnitude = magnitude(p.lv, p.rv);
    const differenceReflectivity = magnitude(dr);
    const axes = [
      { x: "x", y: "y" }, { x: "x2", y: "y2" }, { x: "x3", y: "y3" },
    ] as const;
    const panels = [
      { reflectivity: p.lr, w: p.lw, vorticity: p.lv, label: `A · ${leftLabel}`, diff: false },
      { reflectivity: p.rr, w: p.rw, vorticity: p.rv, label: `B · ${rightLabel}`, diff: false },
      { reflectivity: dr, w: dw, vorticity: dv, label: "A−B composite", diff: true },
    ];
    // A and B share one contour span; the difference panel needs its own, which
    // is typically an order of magnitude smaller. Since the contours are drawn
    // unlabelled and carry no colorbar, nothing on screen would otherwise
    // disclose that an identical white line means a different value in panel 3 —
    // and a reader naturally reads "same line = same w". Keep the spans so the
    // per-panel interval can be stated in the captions below.
    const panelSpans = panels.map((panel) => ({
      w: panel.diff ? magnitude(dw) : wMagnitude,
      vorticity: panel.diff ? magnitude(dv) : vorticityMagnitude,
    }));
    const data: Data[] = [];
    panels.forEach((panel, index) => {
      const axis = axes[index];
      data.push({
        type: "heatmap", zsmooth: false, z: panel.reflectivity.data, x: x.values, y: y.values, xaxis: axis.x, yaxis: axis.y,
        coloraxis: panel.diff ? "coloraxis2" : "coloraxis", name: `${panel.label} reflectivity`,
        hovertemplate: `${panel.label}<br>reflectivity=%{z:.4g} dBZ<extra></extra>`,
      } as Data);
      const contour = (z: number[][], name: string, color: string, span: number): Data => ({
        type: "contour", z, x: x.values, y: y.values, xaxis: axis.x, yaxis: axis.y, name,
        contours: { coloring: "none", start: -span, end: span, size: span / 4, showlabels: false },
        line: { color, width: 1, smoothing: 0 }, showscale: false, hoverinfo: "skip",
      } as Data);
      data.push(contour(panel.w.data, `${panel.label} w`, "#edf0f2", panelSpans[index].w));
      data.push(contour(panel.vorticity.data, `${panel.label} ζz`, "#f0a35e", panelSpans[index].vorticity));
    });
    return {
      data,
      layout: ({
        grid: { rows: 1, columns: 3, pattern: "independent", xgap: 0.09 },
        xaxis: { title: { text: x.title } }, yaxis: { title: { text: y.title } },
        xaxis2: { title: { text: x.title }, matches: "x" }, yaxis2: { title: { text: y.title }, matches: "y" },
        xaxis3: { title: { text: x.title }, matches: "x" }, yaxis3: { title: { text: y.title }, matches: "y" },
        coloraxis: { colorscale: resolveColorscale("dBZ") as never, cmin: DBZ_RANGE[0], cmax: DBZ_RANGE[1], colorbar: { title: { text: reflectivityLabel, side: "right" }, x: 0.646, len: 0.8 } },
        coloraxis2: { colorscale: resolveColorscale("RdBu") as never, cmin: -differenceReflectivity, cmax: differenceReflectivity, cmid: 0, colorbar: { title: { text: differenceLabel, side: "right" }, x: 1.015, len: 0.8 } },
        annotations: [
          ...panels.map((panel, index) => ({ text: panel.label, x: [0.145, 0.5, 0.855][index], y: 1.07, xref: "paper", yref: "paper", showarrow: false })),
          // State each panel's contour interval: the lines are unlabelled and
          // panel 3 is drawn on its own, much smaller span.
          ...panels.map((_panel, index) => ({
            text: `w ${contourStep(panelSpans[index].w)} · ζz ${contourStep(panelSpans[index].vorticity)}`,
            x: [0.145, 0.5, 0.855][index],
            y: 1.02,
            xref: "paper",
            yref: "paper",
            showarrow: false,
            font: { size: 9, color: "#8a9099" },
          })),
        ],
        margin: { l: 55, r: 80, t: 45, b: 45 }, dragmode: "zoom", uirevision: `comparison-composite:${sliceAxis}`,
      } as unknown as Partial<Layout>),
    };
  }, [differenceLabel, fields, leftLabel, reflectivityLabel, rightLabel, sliceAxis, sliceIndex]);
  // Six loads means six chances per frame for the coherence gate to blank the figure; hold the last
  // coherent composite so the canvas updates in place rather than tearing down once per field.
  const { shown: shownPlot } = useRetainedPlot(plot, `${leftRunDir}|${rightRunDir}|${sliceAxis}`);

  const loads = [leftReflectivity, leftW, leftVorticity, rightReflectivity, rightW, rightVorticity];
  const loading = loads.some((item) => item.loading);
  const error = loads.map((item) => item.error).find(Boolean) ?? null;
  return (
    <section className="comparison-composite" aria-label="Synchronized storm composite comparison">
      <header><strong>Storm composite</strong><span>Reflectivity shading · vertical-velocity contours · vertical-vorticity contours · matched slice/time</span></header>
      <div className="comparison-composite__body plot-notice-layout">
        {fields.error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{fields.error}</span></div>}
        {fields.remapped && <div className="plot-overlay"><Icon name="dot" size={12} /><span>physical-volume terrain→Cartesian composite remap</span></div>}
        {!terrainBlocked && !boundedSupported && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>A composite comparison field cannot serve bounded windows.</span></div>}
        {plans?.samplingDisclosure && <div className="plot-overlay"><Icon name="dot" size={12} /><span>{plans.samplingDisclosure} · contour layers use bounded cell means</span></div>}
        {loading && <div className="plot-overlay"><Icon name="dot" size={12} /><span>Loading matched composite fields…</span></div>}
        {error && <div className="plot-overlay plot-overlay--error"><Icon name="warning" size={12} /><span>{error}</span></div>}
        {shownPlot && !error && <PlotCanvas data={shownPlot.data} layout={shownPlot.layout} className="comparison-composite__canvas" />}
      </div>
    </section>
  );
}
