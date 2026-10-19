import { trajectoryBudgetSeries, TRAJECTORY_BUDGET_COLUMNS } from "@/lib/plot/trajectoryBudgetSeries";
import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import { readTrajectories } from "@/lib/run/runOutputs";
import type { NpyN } from "@/lib/platform/npy";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  COMMON_LINE_LAYOUT,
  ControlGroup,
  PillGroup,
  type PlotComponentProps,
} from "./common";
import { RootHistoryScopeNotice } from "./PlotSourceDisclosure";

/* Vorticity budget (tilting/stretching/baroclinic) vs height or time for one selected parcel.
 * Budget computed online every model step — finer cadence than gridded output fields. */

// Column indices in TRAJECTORY_VARS.
const COL = TRAJECTORY_BUDGET_COLUMNS;
/** trajectories.npy must carry the 9-column (budget-bearing) layout. */
const BUDGET_COLS = 9;

type XAxis = "z" | "t";
const X_OPTIONS: ReadonlyArray<{ value: XAxis; label: string }> = [
  { value: "z", label: "Height" },
  { value: "t", label: "Time" },
];

const TERMS = [
  { col: COL.tilting, color: "#e0567a", label: "Tilting" },
  { col: COL.stretching, color: "#4a8cff", label: "Stretching" },
  { col: COL.baroclinic, color: "#3fb46b", label: "Baroclinic" },
] as const;

interface ParcelInfo {
  idx: number;
  z0Km: number;
  zMaxKm: number;
  nSamples: number;
  hasBudget: boolean;
}

/** Per-parcel summary for the picker: ascent extent + whether the parcel has
 *  any finite budget sample (false on grids where the online budget is NaN). */
function buildParcelInfo(traj: NpyN | null): ParcelInfo[] {
  if (!traj || traj.shape.length !== 3) return [];
  const [P, S, V] = traj.shape;
  const d = traj.data;
  const out: ParcelInfo[] = [];
  for (let p = 0; p < P; p++) {
    let z0 = NaN;
    let zMax = -Infinity;
    let n = 0;
    let hasBudget = false;
    for (let s = 0; s < S; s++) {
      const base = (p * S + s) * V;
      const z = d[base + COL.z];
      if (!Number.isFinite(z)) continue;
      if (Number.isNaN(z0)) z0 = z;
      if (z > zMax) zMax = z;
      n++;
      if (V >= BUDGET_COLS && !hasBudget && Number.isFinite(d[base + COL.tilting])) {
        hasBudget = true;
      }
    }
    if (n === 0) continue;
    out.push({
      idx: p,
      z0Km: z0 / 1000,
      zMaxKm: zMax / 1000,
      nSamples: n,
      hasBudget,
    });
  }
  return out;
}

/** Most interesting default: the parcel that ascends the most among those with
 *  a finite budget (the tornadogenesis story lives in a strongly lofted parcel). */
function defaultParcel(info: ParcelInfo[]): number {
  const withBudget = info.filter((i) => i.hasBudget);
  const pool = withBudget.length > 0 ? withBudget : info;
  if (pool.length === 0) return 0;
  let best = pool[0];
  for (const i of pool) {
    if (i.zMaxKm - i.z0Km > best.zMaxKm - best.z0Km) best = i;
  }
  return best.idx;
}

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
}

function buildPlot(traj: NpyN | null, parcelIdx: number, xAxis: XAxis): BuiltPlot | null {
  const samples = trajectoryBudgetSeries(traj, parcelIdx, xAxis);
  if (!samples) return null;
  const { xs, term, sum } = samples;

  const xTitle = xAxis === "z" ? "Height (km)" : "t (s)";
  const xHover = xAxis === "z" ? "z=%{x:.2f} km" : "t=%{x:.1f} s";

  const data: Data[] = TERMS.map((tm, i) => ({
    type: "scatter",
      connectgaps: false,
    mode: "lines+markers",
    x: xs,
    y: term[i],
    name: tm.label,
    line: { color: tm.color, width: 1.8 },
    marker: { size: 4, color: tm.color },
    hovertemplate: `${xHover}<br>${tm.label}=%{y:.3e} s⁻²<extra></extra>`,
  }));
  data.push({
    type: "scatter",
      connectgaps: false,
    mode: "lines",
    x: xs,
    y: sum,
    name: "Sum",
    line: { color: "#9aa4b2", width: 1.2, dash: "dash" },
    hovertemplate: `${xHover}<br>Sum=%{y:.3e} s⁻²<extra></extra>`,
  });

  const layout: Partial<Layout> = {
    ...COMMON_LINE_LAYOUT,
    xaxis: { title: { text: xTitle } },
    yaxis: { title: { text: "vorticity tendency (s⁻²)" }, zeroline: true },
    title: { text: `Vorticity budget — parcel ${parcelIdx}`, font: { size: 13 } },
  };
  return { data, layout };
}

export function TrajectoryBudgetPlot({ runDir, steps, stepIdx }: PlotComponentProps) {
  const publicationGeneration = steps[0]?.publicationGeneration ?? null;
  const [traj, setTraj] = useState<NpyN | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "absent" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [parcelIdx, setParcelIdx] = useState(0);
  const [xAxis, setXAxis] = useState<XAxis>("z");

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setTraj(null);
    setError(null);
    readTrajectories(runDir, publicationGeneration)
      .then((t) => {
        if (cancelled) return;
        if (!t) {
          setStatus("absent");
          return;
        }
        setTraj(t);
        setStatus("ready");
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [runDir, publicationGeneration]);

  const parcelInfo = useMemo(() => buildParcelInfo(traj), [traj]);

  // Land on the most-lofted parcel that has a budget when a run loads.
  useEffect(() => {
    if (parcelInfo.length > 0) setParcelIdx(defaultParcel(parcelInfo));
  }, [parcelInfo]);

  const anyBudget = useMemo(() => parcelInfo.some((i) => i.hasBudget), [parcelInfo]);
  const plot = useMemo(() => buildPlot(traj, parcelIdx, xAxis), [traj, parcelIdx, xAxis]);

  return (
    <>
      <RootHistoryScopeNotice
        runDir={runDir}
        step={steps[stepIdx] ?? steps[0] ?? null}
        label="Root-owned Earth-coordinate trajectories"
      />
      <div className="plot-controls">
        <ControlGroup label="Parcel">
          <select
            value={parcelIdx}
            onChange={(e) => setParcelIdx(Number.parseInt(e.target.value, 10))}
            disabled={parcelInfo.length === 0}
          >
            {parcelInfo.length === 0 && <option value={0}>(no parcels)</option>}
            {parcelInfo.map((i) => (
              <option key={i.idx} value={i.idx}>
                #{i.idx} · z {i.z0Km.toFixed(1)}→{i.zMaxKm.toFixed(1)} km
                {i.hasBudget ? "" : " (no budget)"}
              </option>
            ))}
          </select>
        </ControlGroup>

        <ControlGroup label="x-axis">
          <PillGroup options={X_OPTIONS} value={xAxis} onChange={setXAxis} />
        </ControlGroup>

        <ControlGroup label="Parcels" stretch>
          <span className="plot-controls__status">
            {parcelInfo.length} tracked
          </span>
        </ControlGroup>
      </div>

      <div className="plot-body">
        {status === "loading" && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>Loading trajectories…</span>
          </div>
        )}
        {status === "error" && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>Trajectory load failed: {error}</span>
          </div>
        )}
        {status === "absent" && (
          <div className="plot-overlay">
            <span>No trajectory data — re-run the simulator with --trajectories.</span>
          </div>
        )}
        {status === "ready" && !anyBudget && (
          <div className="plot-overlay">
            <Icon name="info" size={12} />
            <span>
              The vorticity budget is computed only on the Cartesian C-grid; this
              run's parcels carry positions but no budget.
            </span>
          </div>
        )}
        {status === "ready" && anyBudget && plot && (
          <PlotCanvas data={plot.data} layout={plot.layout} />
        )}
        {status === "ready" && anyBudget && !plot && (
          <div className="plot-overlay">
            <span>Selected parcel has no finite budget samples.</span>
          </div>
        )}
      </div>
    </>
  );
}
