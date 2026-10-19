import type { PlotCategory, PlotType } from "./plotRegistry";

export const MAX_PLOT_VIEW_HISTORY = 32;

export interface PlotViewSelection {
  category: PlotCategory;
  plotType: PlotType;
  presetId: string | null;
}

export interface PlotViewHistory {
  past: readonly PlotViewSelection[];
  future: readonly PlotViewSelection[];
}

export interface PlotViewHistoryMove {
  history: PlotViewHistory;
  selection: PlotViewSelection | null;
}

export const EMPTY_PLOT_VIEW_HISTORY: PlotViewHistory = Object.freeze({
  past: Object.freeze([]),
  future: Object.freeze([]),
});

export function samePlotView(
  left: PlotViewSelection,
  right: PlotViewSelection,
): boolean {
  return left.category === right.category &&
    left.plotType === right.plotType &&
    left.presetId === right.presetId;
}

export function recordPlotView(
  history: PlotViewHistory,
  current: PlotViewSelection,
  next: PlotViewSelection,
): PlotViewHistory {
  if (samePlotView(current, next)) return history;
  return {
    past: [...history.past, current].slice(-MAX_PLOT_VIEW_HISTORY),
    future: [],
  };
}

export function undoPlotView(
  history: PlotViewHistory,
  current: PlotViewSelection,
): PlotViewHistoryMove {
  const selection = history.past.at(-1) ?? null;
  if (!selection) return { history, selection: null };
  return {
    history: {
      past: history.past.slice(0, -1),
      future: [current, ...history.future].slice(0, MAX_PLOT_VIEW_HISTORY),
    },
    selection,
  };
}

export function redoPlotView(
  history: PlotViewHistory,
  current: PlotViewSelection,
): PlotViewHistoryMove {
  const selection = history.future[0] ?? null;
  if (!selection) return { history, selection: null };
  return {
    history: {
      past: [...history.past, current].slice(-MAX_PLOT_VIEW_HISTORY),
      future: history.future.slice(1),
    },
    selection,
  };
}
