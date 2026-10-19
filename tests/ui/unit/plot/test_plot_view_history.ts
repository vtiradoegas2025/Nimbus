import {
  EMPTY_PLOT_VIEW_HISTORY,
  MAX_PLOT_VIEW_HISTORY,
  recordPlotView,
  redoPlotView,
  samePlotView,
  undoPlotView,
  type PlotViewSelection,
} from "../../../../ui/src/lib/plot/plotViewHistory.ts";

let passed = 0;
function check(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
  passed++;
}

const slice: PlotViewSelection = { category: "plan-views", plotType: "slice", presetId: null };
const coldPool: PlotViewSelection = { category: "plan-views", plotType: "slice", presetId: "cold-pool" };
const skewT: PlotViewSelection = { category: "soundings", plotType: "skewt", presetId: null };

check(samePlotView(slice, { ...slice }), "equivalent selections must compare equal");
check(!samePlotView(slice, coldPool), "preset identity must participate in equality");
check(recordPlotView(EMPTY_PLOT_VIEW_HISTORY, slice, slice) === EMPTY_PLOT_VIEW_HISTORY, "no-op selections must not create history");

const afterColdPool = recordPlotView(EMPTY_PLOT_VIEW_HISTORY, slice, coldPool);
const afterSkewT = recordPlotView(afterColdPool, coldPool, skewT);
check(afterSkewT.past.length === 2 && afterSkewT.future.length === 0, "user selections must append bounded undo state and clear redo");

const firstUndo = undoPlotView(afterSkewT, skewT);
check(samePlotView(firstUndo.selection!, coldPool), "undo must restore the immediately previous view");
check(firstUndo.history.past.length === 1 && firstUndo.history.future.length === 1, "undo must move the current view to redo");
const secondUndo = undoPlotView(firstUndo.history, coldPool);
check(samePlotView(secondUndo.selection!, slice), "repeated undo must preserve selection order");
check(undoPlotView(secondUndo.history, slice).selection === null, "undo must stop cleanly at the dataset-local boundary");

const firstRedo = redoPlotView(secondUndo.history, slice);
check(samePlotView(firstRedo.selection!, coldPool), "redo must restore the next view");
const divergent = recordPlotView(firstRedo.history, coldPool, skewT);
check(divergent.future.length === 0, "a divergent selection must invalidate redo state");

let bounded = EMPTY_PLOT_VIEW_HISTORY;
let current = slice;
for (let index = 0; index < MAX_PLOT_VIEW_HISTORY + 5; index++) {
  const next: PlotViewSelection = {
    category: index % 2 === 0 ? "soundings" : "plan-views",
    plotType: index % 2 === 0 ? "skewt" : "slice",
    presetId: String(index),
  };
  bounded = recordPlotView(bounded, current, next);
  current = next;
}
check(bounded.past.length === MAX_PLOT_VIEW_HISTORY, "history must retain only the latest bounded selections");

console.log(`plot view history: ${passed} checks passed`);
