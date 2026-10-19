/* Decides when frame playback may advance. Pure - no React, no DOM, no timers. */

import { captureFrameIsReady, type CaptureFrameIdentity } from "@/lib/run/frameIdentity";
import { paneCount, type PaneLayout } from "./paneLayout";
import { reportsFrameIdentity, type PlotType } from "./plotRegistry";

/** The run and open the studio is currently showing; a report tagged otherwise is stale.
 *  `runDir` is nullable because the studio holds this identity before a run is opened. */
export interface PlaybackView {
  runDir: string | null;
  runPublicationGeneration: string | null;
  openEpoch: number;
}

/** What a pane last drew, tagged with the view it drew it for. */
export interface PaneFrameReport extends PlaybackView {
  interpolatedDwellMs?: number;
  runDir: string;
  plotType: PlotType;
  requestedStepIndex: number | null;
  requestedPublicationGeneration: string | null;
  renderedStepIndex: number | null;
  renderedPublicationGeneration: string | null;
  loading: boolean;
}

/** One visible pane: its current type and its latest report, if it has filed one. */
export interface GatedPane {
  plotType: PlotType;
  report: PaneFrameReport | null | undefined;
}

/**
 * The panes on screen, primary first. Companion types beyond the layout's capacity are dropped:
 * they keep their selection for when the user grows the grid back, but they are not being drawn
 * and must not hold playback.
 */
export function visiblePaneTypes(
  layout: PaneLayout,
  primaryType: PlotType,
  companionTypes: readonly PlotType[],
): PlotType[] {
  if (layout === "single") return [primaryType];
  return [primaryType, ...companionTypes.slice(0, paneCount(layout) - 1)];
}

/**
 * Whether one pane has drawn `selected`.
 *
 * A type that never reports its frame identity cannot be waited on, so it counts as ready rather
 * than blocking forever. Everything else must have filed a report for THIS view: a report carrying
 * another run, generation, open epoch, or plot type describes a picture that is no longer on screen.
 */
export function paneHasDrawnFrame(
  pane: GatedPane,
  view: PlaybackView,
  selected: CaptureFrameIdentity | null,
): boolean {
  if (!reportsFrameIdentity(pane.plotType) || selected == null) return true;
  const report = pane.report;
  return report != null
    && report.runDir === view.runDir
    && report.plotType === pane.plotType
    && report.runPublicationGeneration === view.runPublicationGeneration
    && report.openEpoch === view.openEpoch
    && captureFrameIsReady(selected, report);
}

/**
 * Whether every visible pane has drawn the selected frame.
 *
 * Playback advances on a timer, so without this a split view runs each pane at its own speed: the
 * cheap pane races ahead while the expensive one is still resolving, and the grid never shows one
 * coherent moment. Gating on the slowest pane trades frame rate for a picture that agrees with
 * itself. Callers must bound how long they honour a false result - see PLAYBACK_GATE_MAX_HOLD_MS.
 */
export function panesReadyForAdvance(
  panes: readonly GatedPane[],
  view: PlaybackView,
  selected: CaptureFrameIdentity | null,
): boolean {
  return panes.every((pane) => paneHasDrawnFrame(pane, view, selected));
}

/** True while at least one current pane is genuinely working on `selected`.
 * Playback may wait much longer for this case than for an absent/error report;
 * timing out an active high-resolution read just starts the same expensive
 * cancellation cycle on the next frame. */
export function panesAreLoadingFrame(
  panes: readonly GatedPane[],
  view: PlaybackView,
  selected: CaptureFrameIdentity | null,
): boolean {
  if (selected == null) return false;
  return panes.some((pane) => {
    if (!reportsFrameIdentity(pane.plotType)) return false;
    const report = pane.report;
    return report != null
      && report.runDir === view.runDir
      && report.plotType === pane.plotType
      && report.runPublicationGeneration === view.runPublicationGeneration
      && report.openEpoch === view.openEpoch
      && report.loading
      && report.requestedStepIndex === selected.stepIndex
      && report.requestedPublicationGeneration === selected.publicationGeneration;
  });
}

/** Playback dwell begins when a coherent frame is actually presented, not when
 * its field reads were requested. This prevents a slow frame that finishes just
 * before the cadence timer fires from flashing for only a few milliseconds. */
export interface PresentationDwellState {
  readonly frameKey: string | null;
  readonly readySinceMs: number | null;
}

export const EMPTY_PRESENTATION_DWELL: PresentationDwellState = {
  frameKey: null,
  readySinceMs: null,
};

export function updatePresentationDwell(
  state: PresentationDwellState,
  selected: CaptureFrameIdentity | null,
  ready: boolean,
  nowMs: number,
  minimumDwellMs: number,
): { state: PresentationDwellState; advance: boolean } {
  if (!ready || selected == null) {
    return { state: EMPTY_PRESENTATION_DWELL, advance: false };
  }
  const frameKey = JSON.stringify([
    selected.stepIndex,
    selected.publicationGeneration,
  ]);
  if (state.frameKey !== frameKey || state.readySinceMs == null) {
    return {
      state: { frameKey, readySinceMs: nowMs },
      advance: false,
    };
  }
  if (nowMs - state.readySinceMs < Math.max(0, minimumDwellMs)) {
    return { state, advance: false };
  }
  return { state: EMPTY_PRESENTATION_DWELL, advance: true };
}
