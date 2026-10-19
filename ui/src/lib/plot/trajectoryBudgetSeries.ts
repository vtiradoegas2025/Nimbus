import type { NpyN } from "../platform/npy";

export const TRAJECTORY_BUDGET_COLUMNS = { t: 0, z: 3, tilting: 6, stretching: 7, baroclinic: 8 } as const;
const COL = TRAJECTORY_BUDGET_COLUMNS;
const BUDGET_COLS = 9;

export function trajectoryBudgetSeries(traj: NpyN | null, parcelIdx: number, xAxis: "z" | "t") {
  if (!traj || traj.shape.length !== 3) return null;
  const [P, S, V] = traj.shape;
  if (V < BUDGET_COLS || parcelIdx < 0 || parcelIdx >= P) return null;
  const d = traj.data;

  const xs: (number | null)[] = [];
  const term: (number | null)[][] = [[], [], []];
  const sum: (number | null)[] = [];
  for (let s = 0; s < S; s++) {
    const base = (parcelIdx * S + s) * V;
    const t = d[base + COL.t];
    const z = d[base + COL.z];
    const ti = d[base + COL.tilting];
    const st = d[base + COL.stretching];
    const ba = d[base + COL.baroclinic];
    // Require a finite x and all three budget terms (NaN tail / unsupported grid).
    if (![z, t, ti, st, ba].every(Number.isFinite)) {
      xs.push(Number.isFinite(xAxis === "z" ? z : t) ? (xAxis === "z" ? z / 1000 : t) : null);
      term.forEach((values) => values.push(null));
      sum.push(null);
      continue;
    }
    xs.push(xAxis === "z" ? z / 1000 : t);
    term[0].push(ti);
    term[1].push(st);
    term[2].push(ba);
    sum.push(ti + st + ba);
  }
  if (!sum.some((value) => value != null)) return null;

  return { xs, term, sum };
}
