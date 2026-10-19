/* Multi-pane layout arithmetic for the Plot Studio canvas. Pure — no React, no vtk.js, no DOM. */

export type PaneLayout = "single" | "1x2" | "2x1" | "2x2";

export const PANE_LAYOUTS: readonly PaneLayout[] = ["single", "1x2", "2x1", "2x2"];

/** Max simultaneous 3D panes; leaves headroom for remount context overlap under browser ~8-16 cap. */
export const MAX_PANES = 4;

export function paneCount(layout: PaneLayout): number {
  switch (layout) {
    case "single":
      return 1;
    case "1x2":
    case "2x1":
      return 2;
    case "2x2":
      return 4;
  }
}

/** CSS `grid-template-columns` / `-rows` for the pane container. */
export function paneGridTemplate(layout: PaneLayout): { columns: string; rows: string } {
  switch (layout) {
    case "single":
      return { columns: "minmax(0, 1fr)", rows: "minmax(0, 1fr)" };
    case "1x2":
      return { columns: "minmax(0, 1fr) minmax(0, 1fr)", rows: "minmax(0, 1fr)" };
    case "2x1":
      return { columns: "minmax(0, 1fr)", rows: "minmax(0, 1fr) minmax(0, 1fr)" };
    case "2x2":
      return {
        columns: "minmax(0, 1fr) minmax(0, 1fr)",
        rows: "minmax(0, 1fr) minmax(0, 1fr)",
      };
  }
}

export function paneLayoutLabel(layout: PaneLayout): string {
  switch (layout) {
    case "single":
      return "Single";
    case "1x2":
      return "Side by side";
    case "2x1":
      return "Stacked";
    case "2x2":
      return "Quad";
  }
}

/** Floor for per-pane voxel budget; budget is divided by pane count to bound total resident GPU volume. */
export const MIN_PANE_VOXEL_BUDGET = 250_000;

export function scaleVoxelBudget(fullBudget: number, scale: number): number {
  if (!Number.isFinite(fullBudget) || fullBudget <= 0) return MIN_PANE_VOXEL_BUDGET;
  if (!Number.isFinite(scale) || scale <= 0) return MIN_PANE_VOXEL_BUDGET;
  const scaled = Math.floor(fullBudget * Math.min(1, scale));
  return Math.max(MIN_PANE_VOXEL_BUDGET, Math.min(fullBudget, scaled));
}

export function paneVoxelBudget(fullBudget: number, layout: PaneLayout): number {
  // Delegates to scaleVoxelBudget so floor/clamp logic lives in one place.
  return scaleVoxelBudget(fullBudget, 1 / paneCount(layout));
}

export function activeThreeDPaneBudgetScale(activeThreeD: readonly boolean[]): number {
  const count = activeThreeD.filter(Boolean).length;
  return 1 / Math.max(1, count);
}

export function isPaneLayout(value: unknown): value is PaneLayout {
  return typeof value === "string" && (PANE_LAYOUTS as readonly string[]).includes(value);
}
