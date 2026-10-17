import { useIoBudgetRetry } from "@/lib/core/useIoBudgetRetry";
import { guardVtkContextRendering, installVtkContextRenderGuard } from "@/lib/vtk/vtkContextRenderGuard";
import {
  Children,
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useReducer,
  useRef,
  useState,
  type ReactElement,
} from "react";
import type { Annotations } from "plotly.js";
import {
  latchRange,
  planeViewportFromRelayout,
  planeViewportForScope,
  type ScopedPlaneViewport,
  type LatchedRange,
  type PlaneViewportBounds,
  type ValueRange,
} from "@/lib/plot/viewSync";
import { scalarColorRangeLabel, uniformFieldLabel } from "@/lib/plot/plotLabels";
import { appErrorCode, type AppErrorCode } from "@/lib/platform/appInvoke";
import type { MutableRefObject } from "react";
import {
  readField,
  readFieldAny,
  readFieldAnyWindows,
  readFieldWindows,
  plotFieldWindows,
  fieldDefinitionIdentity,
  type FieldLoadWindow,
  type StepInfo,
} from "@/lib/run/runOutputs";
import { beginFieldLoad, endFieldLoad, isRecording } from "@/lib/run/frameReady";
import { reportVtkBackendFailure } from "@/lib/vtk/vtkBackend";
import { usePlaybackContext } from "@/components/plots/playbackContext";
import {
  createPrefetchController,
  endPrefetch,
  frameIdentitiesToPrefetch,
  releasePrefetchController,
  tryBeginPrefetch,
} from "@/lib/run/framePrefetch";
import type { Npy3D, NpyN, SliceAxis } from "@/lib/platform/npy";
import type { PlotPreset } from "@/lib/plot/plotPresets";
import type { ScalarSelection, VolumeHandoffOffer } from "@/lib/plot/volumetricHandoff";
import { translatedVolumeCameraFocus } from "@/lib/plot/volumetricHandoff";
import type { VolumeAppearanceQualificationOverrides } from "@/lib/vtk/volumeAppearanceQualification";
import { resolveFieldLink } from "@/lib/plot/fieldLink";
import { cartesianDyMeters } from "@/lib/science/cartesianInterp";
import {
  beginFrameRequest,
  completeFrameRequest,
  type RequestedFrameIdentity,
} from "@/lib/run/frameIdentity";
import { reduceVtkContextState } from "@/lib/vtk/vtkLifecycle";
import type { RenderFidelityMode } from "@/lib/plot/renderFidelity";
import {
  publishShared3DSetting,
  readLocal3DSetting,
  readShared3DSetting,
  storeLocal3DSetting,
  subscribeShared3DSetting,
  type ThreeDControlScope,
} from "@/lib/plot/threeDControlScope";
import {
  currentStudioAppearance,
  STUDIO_APPEARANCE_APPLIED_EVENT,
} from "@/lib/studio/studioAppearance";
import { fieldBackingStores } from "@/lib/core/fieldCache";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import "./common.css";

/* Shared primitives for all PlotStudio sub-components: props, hooks, camera utilities. */

/**
 * One dataset Studio currently has open.
 *
 * `runDir`/`steps` on PlotComponentProps are the primary dataset and remain the
 * default for every plot. This list exists for the few surfaces that can
 * legitimately work with more than one at a time — scripting, comparison —
 * which previously had no way to reach the comparison dataset even though
 * PlotStudio already had it loaded and frame-aligned.
 */
export interface PlotDatasetHandle {
  /** Stable key: the dataset's own identity, not its display name. */
  readonly id: string;
  /** How Studio is using it. Also the key user code selects it by. */
  readonly role: "primary" | "comparison";
  /** Human label for menus. */
  readonly displayName: string;
  /** The locator `readFieldAny` takes. */
  readonly runDir: string;
  readonly steps: readonly StepInfo[];
}

export interface PlotComponentProps {
  runDir: string;
  steps: StepInfo[];
  stepIdx: number;
  /**
   * Every dataset Studio has open, primary first. Optional because most plots
   * are single-dataset by nature and should keep using runDir/steps.
   */
  datasets?: readonly PlotDatasetHandle[];
  /**
   * Optional one-shot preset applied on mount or when the preset id
   * changes. Widgets use it to pre-configure field / colormap / range /
   * axis. Widgets MUST switch back to "custom" (clear the preset) as
   * soon as the user manually touches a control — the parent owns
   * the preset state and handles that signal externally.
   */
  preset?: PlotPreset | null;
  /**
   * Called by the widget when the user manually changes a control
   * that should drop the active preset. PlotStudio uses this to clear
   * the "active preset" highlight in the pill bar.
   */
  onPresetCleared?: () => void;
  /**
   * Default slice axis for the category the user is viewing (Plan views → z,
   * Cross-sections → θ). SlicePlot / VectorField open at this orientation and
   * reset to it when the category changes; other plots ignore it. The user can
   * still change the axis manually inside the widget (allow selection).
   */
  defaultSliceAxis?: SliceAxis;
  /** Per-frame widgets report the absolute step whose pixels have actually
   * rendered. PlotStudio keeps selected and displayed labels separate. */
  onFrameDisplayChange?: (report: FrameDisplayReport) => void;
  /**
   * Scalar-on-a-plane widgets report the field and display settings currently on
   * screen, so PlotStudio can hand them to a volume render. Reporting the loaded
   * array's rank keeps the offer off single-level fields. Other plots ignore it.
   */
  onScalarSelectionChange?: (selection: ScalarSelection | null) => void;
  /**
   * Present when the host can open this widget's scalar as a volume render. The widget
   * puts the control in its own row, beside the field it applies to.
   */
  volumeHandoff?: VolumeHandoffOffer;
  /**
   * Field another pane moved to, when pane fields are linked. A widget adopts it only if it
   * can draw it. Null when linking is off.
   */
  linkedField?: string | null;
  /**
   * The user picked a field in this widget. Fires on user action only — never when adopting
   * `linkedField`, or two linked panes would drive each other.
   */
  onFieldChange?: (field: string | null) => void;
  volumeAppearanceQualification?: VolumeAppearanceQualificationOverrides;
  /** One Plot Studio-wide contract shared by all 3D artifacts. */
  renderFidelityMode?: RenderFidelityMode;
  /**
   * The open project, or null when Studio is browsing a dataset directly.
   *
   * Only the scripting tab reads it, to scope its editor buffers: hand-written
   * analysis belongs to the analysis it was written for, so opening a different
   * project must not present the previous project's scripts. Plots that hold no
   * per-project state ignore it.
   */
  projectId?: string | null;
}

export interface FrameDisplayReport {
  /** Display interpolation already spent on this incoming frame. */
  interpolatedDwellMs?: number;
  requestedStepIndex: number | null;
  requestedPublicationGeneration: string | null;
  renderedStepIndex: number | null;
  renderedPublicationGeneration: string | null;
  loading: boolean;
}

// Re-exported from pure modules so widget imports have a single entry point.
export type { PlotType } from "@/lib/plot/plotRegistry";
export { PER_FRAME_PLOTS } from "@/lib/plot/plotRegistry";
export type { PlotFitMode } from "@/lib/plot/plotFit";
export { FIT_OPTIONS, fitAxisFragments } from "@/lib/plot/plotFit";

interface ControlGroupProps {
  label: string;
  children: React.ReactNode;
  stretch?: boolean;
  className?: string;
}

function controlReadoutText(node: React.ReactNode): string {
  return Children.toArray(node).map(child => {
    if (typeof child === "string" || typeof child === "number") return String(child);
    return isValidElement<{ children?: React.ReactNode }>(child) ? controlReadoutText(child.props.children) : "";
  }).join("");
}

export function ControlGroup({ label, children, stretch, className }: ControlGroupProps) {
  const labelId = useId();
  const labelledChildren = Children.map(children, (child) => {
    if (!isValidElement(child) || typeof child.type !== "string") return child;
    if (child.type === "span") {
      const readout = child as ReactElement<{ className?: string; title?: string; children?: React.ReactNode }>;
      if (!readout.props.title && /volume-render__slider-value|plot-controls__status/.test(readout.props.className ?? "")) {
        return cloneElement(readout, { title: controlReadoutText(readout.props.children) });
      }
    }
    if (child.type !== "input" && child.type !== "select" && child.type !== "textarea") {
      return child;
    }
    const control = child as ReactElement<{
      "aria-label"?: string;
      "aria-labelledby"?: string;
    }>;
    if (control.props["aria-label"] || control.props["aria-labelledby"]) return child;
    return cloneElement(control, { "aria-labelledby": labelId });
  });
  const classes = [
    "control-group",
    stretch ? "control-group--stretch" : null,
    className,
  ].filter(Boolean).join(" ");
  return (
    <div className={classes}>
      <span className="control-group__label" id={labelId}>{label}</span>
      <div className="control-group__body" role="group" aria-labelledby={labelId}>
        {labelledChildren}
      </div>
    </div>
  );
}

interface PillGroupProps<T extends string | number> {
  options: ReadonlyArray<{
    value: T;
    label: string;
    /** Why this one option cannot be chosen right now. Present means disabled,
     *  and the text becomes its tooltip — an option that is greyed out without
     *  saying why is a dead end. */
    unavailable?: string;
    /** What this option is for, when the label alone is too terse to say. */
    hint?: string;
  }>;
  value: T;
  onChange: (next: T) => void;
  /** Matches the `disabled` every other control in a plot toolbar accepts, so a
   *  pill group whose subject layer is switched off greys out with the rest of
   *  the row instead of staying live and doing nothing. */
  disabled?: boolean;
  /** Why the whole group is disabled, phrased as the action that revives it.
   *  Required in practice for the same reason `unavailable` is: a greyed
   *  control that will not say what turned it off is indistinguishable from a
   *  broken one. */
  disabledReason?: string;
}

export function PillGroup<T extends string | number>({
  options,
  value,
  onChange,
  disabled = false,
  disabledReason,
}: PillGroupProps<T>) {
  return (
    <div className="pill-group">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          className={`pill ${value === o.value ? "pill--active" : ""}`}
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          disabled={disabled || o.unavailable != null}
          title={o.unavailable ?? (disabled ? disabledReason : o.hint)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Common Plotly line/scatter behavior. Surface and text colors are resolved
 * by PlotCanvas from the active Studio appearance. */
export const COMMON_LINE_LAYOUT = {
  showlegend: true,
  legend: { x: 0, y: 1, bgcolor: "transparent" },
  hovermode: "closest" as const,
};

/* ---------- usePlotChrome — theme-resolved Plotly chrome colors ---------- */

/** Non-data colors a plot paints itself: gridlines, zero lines, and the fill
 * behind legends/annotations that float over the plot area. Data colormaps are
 * NOT in here — those stay scientifically stable across themes. */
export interface PlotChrome {
  grid: string;
  zeroline: string;
  overlayBackground: string;
  /** High-contrast neutral for marks drawn ON the plot background: overlay
   * text/borders, and neutral reference traces such as the Skew-T lifted
   * parcel. Must invert with the theme or it vanishes into the paper. */
  ink: string;
}

/** Fallbacks matching theme.css, used only before the document is available
 * (SSR/test harnesses) or if a token is somehow missing. */
const PLOT_CHROME_FALLBACK: Record<"dark" | "light", PlotChrome> = {
  dark: {
    grid: "rgba(255,255,255,0.04)",
    zeroline: "#2e3340",
    overlayBackground: "rgba(20,23,29,0.7)",
    ink: "#eef1f7",
  },
  light: {
    grid: "rgba(24,26,30,0.1)",
    zeroline: "rgba(24,26,30,0.22)",
    overlayBackground: "rgba(253,251,247,0.82)",
    ink: "rgba(24,26,30,0.92)",
  },
};

/** Resolve `--plot-*` CSS tokens to literals Plotly can consume; re-resolves on theme change. */
export function usePlotChrome(): PlotChrome {
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const onAppearance = () => setRevision((previous) => previous + 1);
    window.addEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
    return () => window.removeEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
  }, []);
  return useMemo(() => {
    void revision;
    const fallback = PLOT_CHROME_FALLBACK[currentStudioAppearance()];
    if (typeof document === "undefined") return fallback;
    const root = getComputedStyle(document.documentElement);
    const token = (name: string, fallbackValue: string) =>
      root.getPropertyValue(name).trim() || fallbackValue;
    return {
      grid: token("--plot-gridline", fallback.grid),
      zeroline: token("--plot-zeroline", fallback.zeroline),
      overlayBackground: token("--plot-overlay-bg", fallback.overlayBackground),
      ink: token("--plot-ink", fallback.ink),
    };
  }, [revision]);
}

/** Compact on-canvas disclosure for a scientifically valid constant field.
 * Without this, an honest min=max heatmap looks indistinguishable from a
 * renderer that failed and painted one solid colour. */
export function uniformFieldAnnotation(
  range: ValueRange | null,
  label: string,
  chrome: PlotChrome,
): Partial<Annotations> | null {
  const text = uniformFieldLabel(range, label);
  return plotDataAnnotation(text, chrome);
}

export function scalarColorRangeAnnotation(
  range: ValueRange | null,
  colorRange: ValueRange,
  label: string,
  chrome: PlotChrome,
): Partial<Annotations> | null {
  return plotDataAnnotation(uniformFieldLabel(range, label)
    ?? scalarColorRangeLabel(range, colorRange), chrome);
}

function plotDataAnnotation(text: string | null, chrome: PlotChrome): Partial<Annotations> | null {
  if (text == null) return null;
  return {
    text,
    xref: "paper",
    yref: "paper",
    x: 0.01,
    y: 0.99,
    xanchor: "left",
    yanchor: "top",
    showarrow: false,
    font: { size: 10, color: chrome.ink },
    bgcolor: chrome.overlayBackground,
    bordercolor: chrome.zeroline,
    borderwidth: 1,
    borderpad: 4,
  };
}

/* ---------- Scale latching ---------- */

export type { ValueRange };

export function usePlaneViewport(scope: string, delayMs = 180): {
  viewport: PlaneViewportBounds | null;
  onRelayout: (event: Readonly<Record<string, unknown>>) => void;
  resetViewport: () => void;
} {
  const [state, setViewport] = useState<ScopedPlaneViewport>({ scope, viewport: null });
  const viewport = planeViewportForScope(state, scope);
  const scopeRef = useRef(scope);
  scopeRef.current = scope;
  const pendingRef = useRef<ScopedPlaneViewport>({ scope, viewport: null });
  const timerRef = useRef<number | null>(null);
  const commit = useCallback((next: PlaneViewportBounds | null) => {
    pendingRef.current = { scope, viewport: next };
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      if (scopeRef.current !== scope) return;
      setViewport((previous) => {
        const pending = planeViewportForScope(pendingRef.current, scope);
        return previous.scope === scope
          && previous.viewport?.xMin === pending?.xMin
          && previous.viewport?.xMax === pending?.xMax
          && previous.viewport?.yMin === pending?.yMin
          && previous.viewport?.yMax === pending?.yMax
          ? previous
          : { scope, viewport: pending };
      });
    }, delayMs);
  }, [delayMs, scope]);
  const onRelayout = useCallback((event: Readonly<Record<string, unknown>>) => {
    if (scopeRef.current !== scope) return;
    const previous = pendingRef.current.scope === scope ? pendingRef.current.viewport : viewport;
    const next = planeViewportFromRelayout(event, previous);
    if (next !== undefined) commit(next);
  }, [commit, scope, viewport]);
  const resetViewport = useCallback(() => {
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    pendingRef.current = { scope, viewport: null };
    setViewport(previous => previous.scope === scope && previous.viewport === null
      ? previous : { scope, viewport: null });
  }, [scope]);
  useEffect(() => () => {
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
  }, []);
  return { viewport, onRelayout, resetViewport };
}

/**
 * Widen a colour scale's range monotonically across frames so identical values
 * paint the same colour throughout a run. `scope` resets the latch when the
 * run or field changes; resolved during render, not in an effect.
 */
export function useLatchedRange(
  scope: string,
  frameRange: ValueRange | null,
): ValueRange | null {
  const [latched, setLatched] = useState<LatchedRange>({ scope, range: null });
  const next = latchRange(latched, scope, frameRange);

  useEffect(() => {
    setLatched((previous) => latchRange(previous, scope, frameRange));
    // latchRange returns `previous` unchanged when nothing moved, so setLatched
    // bails on its own and this cannot loop. Keyed on the resolved bounds rather
    // than on frameRange's identity, which is a fresh object every frame.
  }, [scope, next.range?.min, next.range?.max]); // eslint-disable-line react-hooks/exhaustive-deps

  return next.range;
}

/**
 * Opens the field on screen as a volume render. Lives in the control row rather than a
 * menu: it acts on the field selected two controls to its left, and it is the step a
 * reader takes right after choosing that field.
 */
export function VolumeHandoffButton({ offer }: { offer?: VolumeHandoffOffer }) {
  if (!offer) return null;
  return (
    <button
      type="button"
      className="plot-handoff-btn"
      disabled={!offer.available}
      onClick={offer.onOpen}
      title={offer.blockedReason
        ?? `Open ${offer.field} as a volume render beside this view`}
    >
      {offer.label}
    </button>
  );
}

/* ---------- useFieldLink — follow another pane's field ---------- */

/**
 * Adopts a linked field when this widget can draw it, and returns the field it refused.
 *
 * `canShow` is held in a ref: it closes over the step's field list and so is a new function
 * every render, and depending on its identity would re-run the effect continuously.
 */
export function useFieldLink(
  linkedField: string | null | undefined,
  currentField: string | null,
  canShow: (field: string) => boolean,
  adopt: (field: string) => void,
): string | null {
  const canShowRef = useRef(canShow);
  canShowRef.current = canShow;
  const adoptRef = useRef(adopt);
  adoptRef.current = adopt;
  const [declined, setDeclined] = useState<string | null>(null);

  useEffect(() => {
    const decision = resolveFieldLink(linkedField, currentField, (f) => canShowRef.current(f));
    if (decision.adopt != null) adoptRef.current(decision.adopt);
    setDeclined(decision.declined);
  }, [linkedField, currentField]);

  return declined;
}

/* ---------- useScalarSelectionReport — publish what is on screen ---------- */

/**
 * Publishes the widget's current scalar selection upward and withdraws it on unmount,
 * so an offer to render it in 3D cannot outlive the view that produced it.
 *
 * The callback is held in a ref: parents pass inline functions, and depending on the
 * callback's identity would re-publish on every parent render.
 */
export function useScalarSelectionReport(
  report: ((selection: ScalarSelection | null) => void) | undefined,
  field: string | null,
  colormap: ScalarSelection["colormap"],
  rangeMode: ScalarSelection["rangeMode"],
  rank: number | null,
  source: ScalarSelection["source"],
  planeFocus: ScalarSelection["planeFocus"] = null,
): void {
  const reportRef = useRef(report);
  reportRef.current = report;

  useEffect(() => {
    reportRef.current?.(field ? { field, colormap, rangeMode, rank, source, planeFocus } : null);
  }, [field, colormap, rangeMode, rank, source, planeFocus]);

  useEffect(() => () => reportRef.current?.(null), []);
}

/* ---------- useDebouncedValue — settle a fast-changing value ---------- */

/** Default debounce for feeding the step index into the heavy 3D widgets. */
export const SCRUB_DEBOUNCE_MS = 150;

/** Grace before a frame-transition overlay is shown; below this a swap reads as a flash, not a status. */
export const TRANSITION_OVERLAY_GRACE_MS = 400;

/** Debounce a value so the 3D widgets load only the settled frame, not every scrubbed frame.
 *  Bypassed while the recorder drives frames. */
export function useDebouncedValue<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  const lastAppliedAt = useRef(0);
  useEffect(() => {
    if (isRecording()) {
      setDebounced(value);
      return;
    }
    // Leading + trailing: first idle frame loads immediately; only rapid bursts are throttled.
    const now = Date.now();
    const sinceLast = now - lastAppliedAt.current;
    if (sinceLast >= ms) {
      lastAppliedAt.current = now;
      setDebounced(value);
      return;
    }
    const id = setTimeout(() => {
      lastAppliedAt.current = Date.now();
      setDebounced(value);
    }, ms - sinceLast);
    return () => clearTimeout(id);
  }, [value, ms]);
  return debounced;
}

/* Declared in lib/plot/retainedFigure.ts; the hook below is the React binding for it. */
import {
  selectRetainedFigure,
  type RetainedFigure,
  type RetainedSelection,
} from "@/lib/plot/retainedFigure";

/** Debounce the full frame identity together; debouncing only the index can mismatch the generation. */
export function useDebouncedFrameIdentity(
  step: Pick<StepInfo, "index" | "publicationGeneration"> | null | undefined,
  ms: number,
): RequestedFrameIdentity {
  const { playing, recording } = usePlaybackContext();
  const request = useMemo<RequestedFrameIdentity>(
    () => ({
      stepIndex: step?.index ?? null,
      publicationGeneration: step?.publicationGeneration ?? null,
    }),
    [step?.index, step?.publicationGeneration],
  );
  const synchronizedSequence = playing || recording;
  const debounced = useDebouncedValue(request, synchronizedSequence ? 0 : ms);
  // Returning the selected identity directly matters even with a zero delay:
  // useEffect-driven debounce state otherwise adds one full React commit before
  // every playback read. The internal value still follows for a clean pause.
  return synchronizedSequence ? request : debounced;
}

/**
 * `active`, but only after it has stayed true for `ms` — false the moment it clears.
 *
 * Keeps a transition indicator off screen for reads that resolve faster than a person can read it,
 * so a fast frame swap doesn't flash an overlay on and off over the controls.
 */
export function useDelayedTrue(active: boolean, ms: number): boolean {
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    if (!active) {
      setSettled(false);
      return;
    }
    const id = setTimeout(() => setSettled(true), ms);
    return () => clearTimeout(id);
  }, [active, ms]);
  return active && settled;
}

/**
 * Keep the last built figure on screen while the next one resolves, so the canvas updates in place
 * instead of unmounting. See `selectRetainedFigure` for the rule and what `scope` must cover.
 */
export function useRetainedPlot<T>(
  plot: T | null | undefined,
  scope: string,
): Pick<RetainedSelection<T>, "shown" | "stale"> {
  // Cache-only ref: it holds nothing this render did not already produce.
  const retainedRef = useRef<RetainedFigure<T> | null>(null);
  const selection = selectRetainedFigure(plot, retainedRef.current, scope);
  retainedRef.current = selection.retain;
  return { shown: selection.shown, stale: selection.stale };
}

/* ---------- Vertical exaggeration (3D z-scale) ---------- */

/* Declared in lib/plot/sceneScale.ts; re-exported here for the single-import convention. */
import {
  autoVerticalExag,
  HORIZONTAL_SCALE_DEFAULT,
  HORIZONTAL_SCALE_RANGE,
  VERTICAL_EXAG_DEFAULT,
  VERTICAL_EXAG_RANGE,
  type SceneScale,
} from "@/lib/plot/sceneScale";

export {
  autoVerticalExag,
  HORIZONTAL_SCALE_DEFAULT,
  HORIZONTAL_SCALE_RANGE,
  VERTICAL_EXAG_DEFAULT,
  VERTICAL_EXAG_RANGE,
};
export type { SceneScale };

/** Physical face-to-face domain extent in km from grid metadata. */
export function gridExtentKm(
  grid: StepInfo["grid"] | null | undefined,
): { xKm: number; yKm: number; zKm: number } | null {
  if (!grid || grid.geometryKind === "index") return null;
  const { nr, nth, nz, drM, dzM } = grid;
  if (!nr || !nz || !drM || !dzM) return null;
  const bounds = grid.physicalBoundsM;
  const xExtM = bounds ? bounds.upper[0] - bounds.lower[0] : (nr - 1) * drM;
  const yExtM = bounds
    ? bounds.upper[1] - bounds.lower[1]
    : nth ? (nth - 1) * cartesianDyMeters(grid, drM) : xExtM;
  const zKm = (bounds ? bounds.upper[2] - bounds.lower[2] : (nz - 1) * dzM) / 1000;
  return { xKm: xExtM / 1000, yKm: yExtM / 1000, zKm };
}


/** Stable identity for a run's grid: shape + spacing. Used as the dependency
 *  key for camera re-frame / auto-exaggeration resets so two runs with
 *  different shapes or spacings do not collide on the same key. */
export function gridKeyOf(grid: StepInfo["grid"] | null | undefined): string {
  if (!grid) return "";
  return `${grid.geometryKind ?? "legacy"}:${grid.nr}x${grid.nth}x${grid.nz}:${grid.drM}:${grid.dyM ?? grid.dthetaRad}:${grid.dzM}:${grid.scalarOriginM}:${grid.physicalBoundsM?.lower}:${grid.physicalBoundsM?.upper}`;
}

/** Vertical-exaggeration state for a 3D widget: seeded from the domain aspect
 *  and re-picked when the grid first loads or the run changes (so a new run with
 *  a different shape gets an appropriate default), while preserving a manual
 *  slider change made within a run. */
export function useVerticalExag(
  grid: StepInfo["grid"] | null | undefined,
  initial?: number,
): [number, (v: number) => void] {
  const initialValue = () => initial ?? autoVerticalExag(grid);
  const [zExag, setZExag] = useState(initialValue);
  const gridKey = gridKeyOf(grid);
  useEffect(() => {
    if (grid) setZExag(initial ?? autoVerticalExag(grid));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey, initial]);
  return [zExag, setZExag];
}



export function useSceneScale(
  grid: StepInfo["grid"] | null | undefined,
  initialVerticalExag?: number,
  scope: ThreeDControlScope = "independent",
  viewKey = "default",
): [SceneScale, (next: SceneScale) => void] {
  const [z, setZ] = useVerticalExag(grid, initialVerticalExag);
  const [xy, setXy] = useState(HORIZONTAL_SCALE_DEFAULT);
  const gridKey = gridKeyOf(grid);
  useEffect(() => {
    setXy(HORIZONTAL_SCALE_DEFAULT);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey]);
  useEffect(() => {
    const isScale = (value: unknown): value is SceneScale => {
      if (!value || typeof value !== "object") return false;
      const candidate = value as Partial<SceneScale>;
      return Number.isFinite(candidate.xy) && Number.isFinite(candidate.z)
        && (candidate.xy ?? 0) > 0 && (candidate.z ?? 0) > 0;
    };
    const apply = (next: SceneScale) => {
      setXy(next.xy);
      setZ(next.z);
    };
    if (scope === "independent") {
      const local = readLocal3DSetting(viewKey, "scene-scale", isScale);
      if (local) apply(local);
      return;
    }
    const stored = readShared3DSetting("scene-scale", isScale);
    if (stored) apply(stored);
    return subscribeShared3DSetting("scene-scale", isScale, apply);
  }, [gridKey, scope, setZ, viewKey]);
  const set = useCallback(
    (next: SceneScale) => {
      setXy(next.xy);
      setZ(next.z);
      if (scope === "shared") publishShared3DSetting("scene-scale", next);
      else storeLocal3DSetting(viewKey, "scene-scale", next);
    },
    [scope, setZ, viewKey],
  );
  const scale = useMemo(() => ({ xy, z }), [xy, z]);
  return [scale, set];
}

/** Apply a SceneScale to one or more vtk props (display-only). Null/undefined
 *  props are skipped, so optional actors (e.g. a domain box) can be passed. */
type ScalableProp = { setScale: (x: number, y: number, z: number) => void };
export function applySceneScale(
  scale: SceneScale,
  ...props: Array<ScalableProp | null | undefined>
): void {
  for (const p of props) p?.setScale(scale.xy, scale.xy, scale.z);
}

/** On `resetKey` change: clear camera-init flag and call `hide` (must hide actors + render)
 *  so the prior run's geometry is not left painted while new data loads. */
export function useVtkRunReset(
  resetKey: string,
  cameraInitializedRef: MutableRefObject<boolean>,
  hide: () => void,
): void {
  const hideRef = useRef(hide);
  hideRef.current = hide;
  useEffect(() => {
    cameraInitializedRef.current = false;
    hideRef.current();
    // resetKey is the only real dependency; hide is read fresh via the ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey]);
}

/* ---------- Shared 3D camera framing ---------- */

import {
  cameraAzimuthRadians,
  CAMERA_VIEW_ANGLE_DEGREES,
  DEFAULT_AZIMUTH_RADIANS,
  type CameraPlacement,
} from "@/lib/plot/cameraElevation";

type VtkCameraLike = {
  setFocalPoint: (x: number, y: number, z: number) => boolean;
  setPosition: (x: number, y: number, z: number) => boolean;
  setViewUp: (x: number, y: number, z: number) => boolean;
  getPosition?: () => ArrayLike<number>;
  getFocalPoint?: () => ArrayLike<number>;
  setParallelProjection?: (parallel: boolean) => boolean;
  setParallelScale?: (scale: number) => boolean;
  setViewAngle?: (angle: number) => boolean;
  orthogonalizeViewUp?: () => void;
  /** Divides the camera-to-focal-point distance, preserving the view angle. */
  dolly?: (amount: number) => void;
};

type VtkBounds = [number, number, number, number, number, number];

type VtkRendererLike = {
  computeVisiblePropBounds: () => VtkBounds;
  getActiveCamera: () => VtkCameraLike;
  resetCamera: (bounds?: VtkBounds) => boolean;
  resetCameraClippingRange: (bounds?: VtkBounds) => boolean;
  /** vtk.js types this as number[]; at runtime it is a Float64Array triple. */
  worldToNormalizedDisplay?: (
    x: number,
    y: number,
    z: number,
    aspect: number,
  ) => ArrayLike<number>;
  getRenderWindow?: () => {
    getViews?: () => Array<{ getSize?: () => ArrayLike<number> | null }>;
  } | null;
};

/** Viewport aspect (width / height) for a renderer's view, or null if unknown. */
function rendererAspect(renderer: VtkRendererLike): number | null {
  const size = renderer.getRenderWindow?.()?.getViews?.()?.[0]?.getSize?.();
  if (!size || size.length < 2) return null;
  const width = size[0];
  const height = size[1];
  if (!(width > 0) || !(height > 0)) return null;
  return width / height;
}

/** Share of the viewport left empty around the framed bounds, per side. */
const FRAME_MARGIN = 0.04;

/** Dolly the camera until the bounds fill the viewport, correcting for the bounding-sphere
 *  over-distance of resetCamera. Iterates to convergence; no-op if projection is unavailable. */
function dollyToFillFrame(
  renderer: VtkRendererLike,
  camera: VtkCameraLike,
  bounds: VtkBounds,
): void {
  const aspect = rendererAspect(renderer);
  const project = renderer.worldToNormalizedDisplay;
  if (aspect == null || !project || !camera.dolly) return;

  const projectedReach = (box: VtkBounds): { x: number; y: number } | null => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let corner = 0; corner < 8; corner++) {
      const projected = project.call(
        renderer,
        box[corner & 1 ? 1 : 0],
        box[corner & 2 ? 3 : 2],
        box[corner & 4 ? 5 : 4],
        aspect,
      );
      const nx = projected[0];
      const ny = projected[1];
      const nz = projected[2];
      // A corner behind the near plane makes the projected silhouette
      // meaningless; keep the conservative fit rather than guess.
      if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) return null;
      if (nz < 0 || nz > 1) return null;
      if (nx < minX) minX = nx;
      if (nx > maxX) maxX = nx;
      if (ny < minY) minY = ny;
      if (ny > maxY) maxY = ny;
    }
    const x = Math.max(Math.abs(maxX - 0.5), Math.abs(0.5 - minX));
    const y = Math.max(Math.abs(maxY - 0.5), Math.abs(0.5 - minY));
    return x > 0 && y > 0 ? { x, y } : null;
  };

  // Half the viewport, less the margin: the distance from centre to the edge
  // the furthest corner is allowed to reach.
  const target = 0.5 - FRAME_MARGIN;
  for (let pass = 0; pass < 4; pass++) {
    const reach = projectedReach(bounds);
    if (!reach) return;
    const factor = Math.min(target / reach.x, target / reach.y);
    if (!Number.isFinite(factor) || factor <= 0) return;
    // Converged: a sub-percent correction is not worth another pass.
    if (Math.abs(factor - 1) < 0.01) return;
    // Never leap more than 2x in one pass; a wild first estimate on a very
    // oblique box would otherwise dolly straight through the scene.
    camera.dolly(Math.min(Math.max(factor, 0.5), 2));
    // Moving in brings the near plane forward with it. Without this the next
    // pass sees its own corners clipped and abandons the fit half-done.
    renderer.resetCameraClippingRange(padBounds(bounds, 0.01));
  }
}

/** Re-fit the camera in place, preserving view direction; used for scale changes and new frames. */
export function refitVtkCameraToBounds(
  renderer: VtkRendererLike,
  focusBounds?: VtkBounds | null,
): void {
  const visibleBounds = renderer.computeVisiblePropBounds();
  const bounds = focusBounds && isFiniteBounds(focusBounds)
    ? focusBounds
    : visibleBounds;
  if (!isFiniteBounds(bounds)) {
    renderer.resetCamera();
    return;
  }
  renderer.resetCamera(padBounds(bounds, 0.08));
  // Centre on focus bounds but size to all visible props so the domain box stays unclipped.
  dollyToFillFrame(
    renderer,
    renderer.getActiveCamera(),
    isFiniteBounds(visibleBounds) ? visibleBounds : bounds,
  );
  renderer.resetCameraClippingRange(
    isFiniteBounds(visibleBounds) ? padBounds(visibleBounds, 0.01) : bounds,
  );
}

/**
 * Fill the viewport with one explicit feature while retaining every visible
 * prop in the clipping range. Unlike the general scene refit, the domain box
 * does not participate in zoom sizing, so a sub-kilometre vortex inside a
 * mesoscale domain can actually be inspected.
 */
export function focusVtkCameraOnBounds(
  renderer: VtkRendererLike,
  focusBounds: VtkBounds | null | undefined,
): void {
  if (!isFiniteBounds(focusBounds)) return;
  const visibleBounds = renderer.computeVisiblePropBounds();
  const framed = padBounds(focusBounds, 0.18);
  renderer.resetCamera(framed);
  dollyToFillFrame(renderer, renderer.getActiveCamera(), focusBounds);
  renderer.resetCameraClippingRange(
    isFiniteBounds(visibleBounds) ? padBounds(visibleBounds, 0.01) : framed,
  );
}

/** The bearing the camera currently orbits from, so a pitch change can keep it. */
export function currentVtkCameraAzimuth(renderer: VtkRendererLike): number {
  const camera = renderer.getActiveCamera();
  const position = camera.getPosition?.();
  const focalPoint = camera.getFocalPoint?.();
  if (!position || !focalPoint || position.length < 2 || focalPoint.length < 2) {
    return DEFAULT_AZIMUTH_RADIANS;
  }
  return cameraAzimuthRadians(
    [position[0]!, position[1]!, position[2] ?? 0],
    [focalPoint[0]!, focalPoint[1]!, focalPoint[2] ?? 0],
  );
}

/** How far back the eye currently stands, so a height change can keep the zoom. */
export function currentVtkCameraDistance(renderer: VtkRendererLike): number | null {
  const camera = renderer.getActiveCamera();
  const position = camera.getPosition?.();
  const focalPoint = camera.getFocalPoint?.();
  if (!position || !focalPoint || position.length < 3 || focalPoint.length < 3) return null;
  const distance = Math.hypot(
    position[0]! - focalPoint[0]!,
    position[1]! - focalPoint[1]!,
    position[2]! - focalPoint[2]!,
  );
  return Number.isFinite(distance) && distance > 0 ? distance : null;
}

/** Viewport width/height, for callers that must fit a scene to the frame themselves. */
export function vtkRendererAspect(renderer: VtkRendererLike): number | null {
  return rendererAspect(renderer);
}

/**
 * Move the camera to an explicit placement, keeping the scene's own framing
 * rules. Unlike the reset presets this does NOT re-fit to bounds: the placement
 * already states where to stand, and a re-fit would pull the eye back out of
 * the low levels it was asked to enter.
 */
export function applyVtkCameraPlacement(
  renderer: VtkRendererLike,
  placement: CameraPlacement,
  projection: "perspective" | "orthographic" = "perspective",
): void {
  const camera = renderer.getActiveCamera();
  const distance = Math.hypot(
    placement.position[0] - placement.focalPoint[0],
    placement.position[1] - placement.focalPoint[1],
    placement.position[2] - placement.focalPoint[2],
  );
  camera.setParallelProjection?.(projection === "orthographic");
  camera.setViewAngle?.(CAMERA_VIEW_ANGLE_DEGREES);
  if (projection === "orthographic" && Number.isFinite(distance) && distance > 0) {
    camera.setParallelScale?.(
      distance * Math.tan(0.5 * CAMERA_VIEW_ANGLE_DEGREES * Math.PI / 180),
    );
  }
  camera.setFocalPoint(...placement.focalPoint);
  camera.setPosition(...placement.position);
  camera.setViewUp(...placement.viewUp);
  camera.orthogonalizeViewUp?.();
  // Standing inside the domain puts geometry behind the eye; without a fresh
  // range the near plane keeps the previous, far-away setting and slices the
  // scene open.
  const visibleBounds = renderer.computeVisiblePropBounds();
  renderer.resetCameraClippingRange(
    isFiniteBounds(visibleBounds) ? padBounds(visibleBounds, 0.01) : undefined,
  );
}

/** Set an oblique storm-chaser view direction, then fit to bounds.
 *  Without seeding the view, resetCamera would open top-down. */
export function resetVtkCameraToStormView(
  renderer: VtkRendererLike,
  focusBounds?: VtkBounds | null,
  framingBounds?: VtkBounds | null,
): void {
  const visibleBounds = renderer.computeVisiblePropBounds();
  const fitBounds = framingBounds && isFiniteBounds(framingBounds)
    ? framingBounds
    : visibleBounds;
  const bounds = focusBounds && isFiniteBounds(focusBounds)
    ? focusBounds
    : visibleBounds;
  if (!isFiniteBounds(bounds)) {
    renderer.resetCamera();
    return;
  }

  const cx = 0.5 * (bounds[0] + bounds[1]);
  const cy = 0.5 * (bounds[2] + bounds[3]);
  const cz = 0.5 * (bounds[4] + bounds[5]);
  const sx = Math.max(bounds[1] - bounds[0], 1);
  const sy = Math.max(bounds[3] - bounds[2], 1);
  const sz = Math.max(bounds[5] - bounds[4], 1);
  const horizontal = Math.max(sx, sy);

  const camera = renderer.getActiveCamera();
  camera.setParallelProjection?.(false);
  camera.setViewAngle?.(30);
  camera.setFocalPoint(cx, cy, cz);
  camera.setPosition(
    cx - horizontal * 1.1,
    cy - horizontal * 1.35,
    cz + Math.max(horizontal * 0.55, sz * 1.8),
  );
  camera.setViewUp(0, 0, 1);
  camera.orthogonalizeViewUp?.();

  const framed = padBounds(bounds, 0.08);
  renderer.resetCamera(framed);
  camera.setViewUp(0, 0, 1);
  camera.orthogonalizeViewUp?.();
  // Close in until bounds fill the frame; size to all visible props (keeps domain box unclipped).
  dollyToFillFrame(
    renderer,
    camera,
    isFiniteBounds(fitBounds) ? fitBounds : bounds,
  );
  renderer.resetCameraClippingRange(
    isFiniteBounds(fitBounds) ? padBounds(fitBounds, 0.01) : framed,
  );
}

export function focusVtkCameraOnPoint(
  renderer: VtkRendererLike,
  point: readonly [number | null, number | null, number | null] | null | undefined,
): void {
  if (!point) return;
  const camera = renderer.getActiveCamera();
  const currentFocal = camera.getFocalPoint?.();
  const currentPosition = camera.getPosition?.();
  if (!currentFocal || !currentPosition || currentFocal.length < 3 || currentPosition.length < 3) return;
  const visibleBounds = renderer.computeVisiblePropBounds();
  const focal: [number, number, number] = [
    currentFocal[0]!,
    currentFocal[1]!,
    currentFocal[2]!,
  ];
  const position: [number, number, number] = [
    currentPosition[0]!,
    currentPosition[1]!,
    currentPosition[2]!,
  ];
  const translated = translatedVolumeCameraFocus(
    focal,
    position,
    point,
    isFiniteBounds(visibleBounds) ? visibleBounds : null,
  );
  if (translated.focalPoint.every((value, axis) => value === focal[axis])) return;
  camera.setFocalPoint(...translated.focalPoint);
  camera.setPosition(...translated.position);
  renderer.resetCameraClippingRange(
    isFiniteBounds(visibleBounds) ? padBounds(visibleBounds, 0.01) : undefined,
  );
}

export type { VtkBounds, VtkRendererLike };


/** Smallest bounds enclosing every finite input; null when none is usable. */
export function unionVtkBounds(
  ...candidates: ReadonlyArray<unknown>
): VtkBounds | null {
  let union: VtkBounds | null = null;
  for (const candidate of candidates) {
    if (!isFiniteBounds(candidate)) continue;
    union = union == null ? [...candidate] as VtkBounds : [
      Math.min(union[0], candidate[0]), Math.max(union[1], candidate[1]),
      Math.min(union[2], candidate[2]), Math.max(union[3], candidate[3]),
      Math.min(union[4], candidate[4]), Math.max(union[5], candidate[5]),
    ];
  }
  return union;
}

function isFiniteBounds(bounds: unknown): bounds is VtkBounds {
  return (
    Array.isArray(bounds) &&
    bounds.length >= 6 &&
    bounds.every(Number.isFinite) &&
    bounds[1] >= bounds[0] &&
    bounds[3] >= bounds[2] &&
    bounds[5] >= bounds[4]
  );
}

/** Expand bounds by a fractional margin on each axis so camera framing leaves
 *  breathing room around the scene instead of cropping it at the viewport edge. */
function padBounds(b: VtkBounds, frac: number): VtkBounds {
  const dx = (b[1] - b[0]) * frac;
  const dy = (b[3] - b[2]) * frac;
  const dz = (b[5] - b[4]) * frac;
  return [b[0] - dx, b[1] + dx, b[2] - dy, b[3] + dy, b[4] - dz, b[5] + dz];
}

/** Horizontal (x=y) and vertical (z) display-scale sliders; props only — data stays in true coords. */
export function SceneScaleControl({
  value,
  onChange,
}: {
  value: SceneScale;
  onChange: (next: SceneScale) => void;
}) {
  return (
    <>
      <ControlGroup label="Horizontal scale">
        <input
          aria-label="Horizontal scene scale"
          type="range"
          min={HORIZONTAL_SCALE_RANGE.min}
          max={HORIZONTAL_SCALE_RANGE.max}
          step={HORIZONTAL_SCALE_RANGE.step}
          value={value.xy}
          onChange={(e) => onChange({ ...value, xy: Number.parseFloat(e.target.value) })}
          className="volume-render__slider"
        />
        <span className="volume-render__slider-value">{value.xy.toFixed(1)}×</span>
      </ControlGroup>
      <ControlGroup label="Vertical scale">
        <input
          aria-label="Vertical scene scale"
          type="range"
          min={VERTICAL_EXAG_RANGE.min}
          max={VERTICAL_EXAG_RANGE.max}
          step={VERTICAL_EXAG_RANGE.step}
          value={value.z}
          onChange={(e) => onChange({ ...value, z: Number.parseFloat(e.target.value) })}
          className="volume-render__slider"
        />
        <span className="volume-render__slider-value">{value.z.toFixed(1)}×</span>
      </ControlGroup>
    </>
  );
}

/* ---------- useField — load one field at one step ---------- */

export interface UseFieldResult {
  field: Npy3D | null;
  loading: boolean;
  error: string | null;
  /** Rust variant behind `error`, so callers branch on the code instead of
   *  matching message text. Null for non-IPC failures. */
  errorCode: AppErrorCode | null;
  requestedStepIndex: number | null;
  requestedPublicationGeneration: string | null;
  loadedStepIndex: number | null;
  loadedPublicationGeneration: string | null;
}

let nextFieldLoadScope = 0;

function allocateFieldLoadScope(kind: string): string {
  nextFieldLoadScope += 1;
  return `${kind}:${nextFieldLoadScope}`;
}

function isFieldLoadAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function useFieldResourceOwnership(
  owner: string,
  field: NpyN | null,
  generation: string,
): void {
  useEffect(() => {
    if (!field) return;
    const resources = visualizationResourceGovernor.reserve({
      owner,
      generation,
      priority: "visible",
      claims: fieldBackingStores(field).map((buffer) =>
        visualizationBufferClaim(buffer, "decoded-field")
      ),
    });
    if (!resources.ok) return;
    return () => { visualizationResourceGovernor.release(resources.reservation.id); };
  }, [field, generation, owner]);
}

function useOwnedFieldPrefetches(identity: string | null): Set<AbortController> {
  const owned = useRef(new Set<AbortController>());
  useEffect(() => {
    const controllers = owned.current;
    return () => {
      for (const controller of controllers) controller.abort();
      controllers.clear();
    };
  }, [identity]);
  return owned.current;
}

/** Fetch a single 3D field; cancellation-safe. Pass null to clear. */
export function useField(
  runDir: string,
  stepIndex: number | null,
  fieldName: string | null,
  publicationGeneration: string | null,
  /**
   * Optional read-time window: the reader aggregates on the Rust side and returns the reduced
   * lattice, so a capped renderer moves a fraction of the bytes. Held in a ref because it is
   * rebuilt each render; the load is keyed on the frame identity, not on this object.
   */
  windows?: readonly FieldLoadWindow[] | null,
  options: { prefetch?: boolean; cacheRetention?: "resident" | "transient"; brickHierarchyLevel?: number } = {},
): UseFieldResult {
  const cacheRetention = options.cacheRetention ?? "resident";
  const brickHierarchyLevel = options.brickHierarchyLevel;
  const prefetchEnabled = options.prefetch !== false && cacheRetention === "resident" && brickHierarchyLevel == null;
  const budgetedWindows = plotFieldWindows(runDir, stepIndex, fieldName, publicationGeneration, windows);
  const windowsRef = useRef(budgetedWindows);
  windowsRef.current = budgetedWindows;
  const windowsKey = JSON.stringify([brickHierarchyLevel ?? 0, budgetedWindows]);
  const definitionIdentity = fieldDefinitionIdentity(runDir, stepIndex, fieldName, publicationGeneration);
  // Speculation belongs to the selected series, so advancing onto a prefetched
  // frame can join its read without the previous frame's cleanup aborting it.
  const ownedPrefetches = useOwnedFieldPrefetches(stepIndex == null || !fieldName || !prefetchEnabled
    ? null : JSON.stringify([runDir, fieldName, windowsKey, definitionIdentity]));
  const [state, setState] = useState<UseFieldResult & { sourceSeries: string }>({
    field: null,
    loading: false,
    error: null,
    errorCode: null,
    requestedStepIndex: null,
    requestedPublicationGeneration: null,
    loadedStepIndex: null,
    loadedPublicationGeneration: null,
    sourceSeries: "",
  });
  const seriesRef = useRef<string>("");
  const budgetRetry = useIoBudgetRetry(state.error);
  const loadScopeRef = useRef<string | null>(null);
  if (loadScopeRef.current == null) {
    loadScopeRef.current = allocateFieldLoadScope("field3d");
  }

  useEffect(() => {
    if (stepIndex == null || !fieldName) {
      setState({
        field: null,
        loading: false,
        error: null,
        errorCode: null,
        requestedStepIndex: null,
        requestedPublicationGeneration: null,
        loadedStepIndex: null,
        loadedPublicationGeneration: null,
        sourceSeries: "",
      });
      seriesRef.current = "";
      return;
    }
    let cancelled = false;
    let loadedSuccessfully = false;
    const controller = new AbortController();
    // Same series (run + field): keep previous frame for smooth scrubbing. On change: clear.
    const series = JSON.stringify([runDir, fieldName, windowsKey, definitionIdentity]);
    const sameSeries = seriesRef.current === series;
    seriesRef.current = series;
    setState((prev) => {
      const identity = beginFrameRequest(
        prev.loadedStepIndex,
        prev.loadedPublicationGeneration,
        stepIndex,
        publicationGeneration,
        sameSeries,
      );
      return {
        field: sameSeries ? prev.field : null,
        loading: true,
        error: null,
        errorCode: null,
        ...identity,
        sourceSeries: series,
      };
    });
    beginFieldLoad();
    const activeWindows = windowsRef.current;
    const load = activeWindows
      ? readFieldWindows(
          runDir,
          stepIndex,
          fieldName,
          publicationGeneration,
          activeWindows,
          {
            latestScope: loadScopeRef.current ?? undefined,
            signal: controller.signal,
            cacheRetention,
            brickHierarchyLevel,
          },
        )
      : readField(runDir, stepIndex, fieldName, publicationGeneration, {
          latestScope: loadScopeRef.current ?? undefined,
          cacheRetention,
          signal: controller.signal,
        });
    load
      .then(
        (f) => {
          if (!cancelled) {
            loadedSuccessfully = true;
            setState({
              field: f,
              loading: false,
              error: null,
              errorCode: null,
              ...completeFrameRequest(stepIndex, publicationGeneration),
              sourceSeries: series,
            });
          }
        },
        (e) => {
          if (cancelled) return;
          if (isFieldLoadAbort(e)) {
            // Superseding a frame/window is routine during playback and plot
            // changes. Preserve the last complete frame from this series and
            // never present cooperative cancellation as a failed data read.
            setState((previous) => previous.sourceSeries === series
              ? { ...previous, loading: false, error: null, errorCode: null }
              : previous);
            return;
          }
          setState({
              field: null,
              loading: false,
              error: e instanceof Error ? e.message : String(e),
              errorCode: appErrorCode(e),
              requestedStepIndex: stepIndex,
              requestedPublicationGeneration: publicationGeneration,
              loadedStepIndex: null,
              loadedPublicationGeneration: null,
              sourceSeries: series,
            });
        },
      )
      .finally(() => {
        endFieldLoad();
        // Prefetch upcoming playback frames; unawaited; respects global ceiling.
        if (cancelled || !loadedSuccessfully || !prefetchEnabled) return;
        for (const nextFrame of frameIdentitiesToPrefetch(runDir, stepIndex, publicationGeneration)) {
          if (!tryBeginPrefetch()) break;
          const prefetchController = createPrefetchController();
          ownedPrefetches.add(prefetchController);
          const prefetch = activeWindows
            ? readFieldWindows(
                runDir,
                nextFrame.stepIndex,
                fieldName,
                nextFrame.publicationGeneration,
                activeWindows,
                { signal: prefetchController.signal, priority: "background" },
              )
            : readFieldAny(runDir, nextFrame.stepIndex, fieldName, nextFrame.publicationGeneration, {
                signal: prefetchController.signal,
                priority: "background",
              });
          void prefetch
            .catch(() => {})
            .finally(() => {
              ownedPrefetches.delete(prefetchController);
              releasePrefetchController(prefetchController);
              endPrefetch();
            });
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [budgetRetry, runDir, stepIndex, fieldName, publicationGeneration, windowsKey, prefetchEnabled, cacheRetention, definitionIdentity, brickHierarchyLevel, ownedPrefetches]);

  const desiredSeries = stepIndex == null || !fieldName
    ? ""
    : JSON.stringify([runDir, fieldName, windowsKey, definitionIdentity]);
  const currentField = state.sourceSeries === desiredSeries ? state.field : null;
  useFieldResourceOwnership(
    loadScopeRef.current,
    currentField,
    `${state.loadedPublicationGeneration ?? "none"}:${state.loadedStepIndex ?? "none"}`,
  );
  if (state.sourceSeries !== desiredSeries) {
    return {
      ...state,
      field: null,
      loading: desiredSeries !== "",
      error: null,
      errorCode: null,
      requestedStepIndex: stepIndex,
      requestedPublicationGeneration: publicationGeneration,
      loadedStepIndex: null,
      loadedPublicationGeneration: null,
    };
  }
  if (state.requestedStepIndex !== stepIndex
      || state.requestedPublicationGeneration !== publicationGeneration) {
    return {
      ...state,
      loading: desiredSeries !== "",
      error: null,
      errorCode: null,
      requestedStepIndex: stepIndex,
      requestedPublicationGeneration: publicationGeneration,
    };
  }
  return state;
}

/* ---------- useVtkContextLoss — recover from a lost WebGL context ---------- */

interface VtkRenderWindowLike {
  getRenderWindow?: () => { render: (...args: unknown[]) => unknown };
  getApiSpecificRenderWindow: () => {
    getContext?: () => { isContextLost?: () => boolean } | null;
    getCanvas?: () => HTMLCanvasElement | null;
    onDeviceLost?: (callback: (event: {
      reason?: string;
      message?: string;
      recoverable?: boolean;
    }) => void) => { unsubscribe: () => void };
  };
}

export interface VtkContextLoss {
  /** True after the GPU dropped this widget's WebGL context. */
  lost: boolean;
  /** A restored context increments this and forces the owning scene to rebuild. */
  generation: number;
  /** False while the old context is lost. Scene mount effects must wait for
   * restoration rather than rendering through vtk.js's stale shader objects. */
  available: boolean;
  /**
   * Attach the context-loss listeners to a freshly-created render window inside
   * the mount effect; returns a detach function for the effect cleanup. Safe
   * no-op when the canvas isn't available yet.
   */
  attach: (renderWindow: VtkRenderWindowLike) => () => void;
  /** Clear the blocking recovery state only after this rebuilt generation has
   * rendered real data with the selected frame identity. */
  markFrameRendered: (generation: number) => void;
}

/** Track WebGL context loss; surfaces a recoverable error when the GPU drops the context.
 *  preventDefault arms restoration; `lost` clears only after the rebuilt generation renders real data. */
export function useVtkContextLoss(): VtkContextLoss {
  installVtkContextRenderGuard();
  const [state, dispatch] = useReducer(reduceVtkContextState, {
    lost: false,
    available: true,
    generation: 0,
  });
  const attach = useCallback((renderWindow: VtkRenderWindowLike) => {
    const apiWindow = renderWindow.getApiSpecificRenderWindow();
    const canvas = apiWindow.getCanvas?.() ?? null;
    if (!canvas) return () => {};
    const retireRendering = guardVtkContextRendering(renderWindow);
    let restoreFrameOne: number | null = null;
    let restoreFrameTwo: number | null = null;
    let restoreExpiry: number | null = null;
    let contextLost = false;
    const onLost = (e: Event) => {
      e.preventDefault();
      retireRendering();
      contextLost = true;
      dispatch({ type: "lost" });
    };
    const onRestored = () => {
      canvas.removeEventListener("webglcontextrestored", onRestored, false);
      if (restoreExpiry != null) {
        window.clearTimeout(restoreExpiry);
        restoreExpiry = null;
      }
      // WebKit announces restoration before vtk.js can reliably compile a new
      // shader program on the restored drawing surface. Rebuilding in the same
      // event turn can therefore reach a mapper whose getProgram() is still
      // null. Two paint boundaries let the browser finish restoring the canvas;
      // the generation change then disposes the stale pipeline and builds a
      // completely fresh one. The recovery overlay remains visible meanwhile.
      restoreFrameOne = requestAnimationFrame(() => {
        restoreFrameOne = null;
        restoreFrameTwo = requestAnimationFrame(() => {
          restoreFrameTwo = null;
          dispatch({ type: "restored" });
        });
      });
    };
    const deviceLossSubscription = apiWindow.onDeviceLost?.((event) => {
      if (canvas.dataset.nimbusVtkRelease === "planned") return;
      retireRendering();
      contextLost = true;
      const reason = [event.reason, event.message].filter(Boolean).join(": ")
        || "WebGPU device lost";
      // VTK can recreate a WebGPU device internally, but rebuilding this scene
      // on the qualified WebGL fallback is deterministic and avoids submitting
      // stale mapper resources during recovery.
      reportVtkBackendFailure(reason);
      dispatch({ type: "lost" });
      onRestored();
    });
    canvas.addEventListener("webglcontextlost", onLost, false);
    canvas.addEventListener("webglcontextrestored", onRestored, false);
    return () => {
      retireRendering();
      canvas.removeEventListener("webglcontextlost", onLost, false);
      deviceLossSubscription?.unsubscribe();
      if (!contextLost) {
        canvas.removeEventListener("webglcontextrestored", onRestored, false);
      } else if (restoreExpiry == null) {
        // Generation-driven cleanup leaves the old canvas's restoration
        // listener alive for the native event. Some software/browser drivers
        // stop delivering that event after sustained context churn; rebuild on
        // a fresh generation if it does not arrive promptly. onRestored is
        // idempotent because it removes its own listener before scheduling the
        // two-paint-boundary transition.
        restoreExpiry = window.setTimeout(() => {
          restoreExpiry = null;
          onRestored();
        }, 2_000);
      }
      // A loss-driven generation cleanup must not cancel the paint-boundary
      // transition that re-enables scene mounting on the fallback backend.
      if (!contextLost && restoreFrameOne != null) cancelAnimationFrame(restoreFrameOne);
      if (!contextLost && restoreFrameTwo != null) cancelAnimationFrame(restoreFrameTwo);
    };
  }, []);
  const markFrameRendered = useCallback((generation: number) => {
    dispatch({ type: "frame_rendered", generation });
  }, []);
  return {
    lost: state.lost,
    available: state.available,
    generation: state.generation,
    attach,
    markFrameRendered,
  };
}

/* ---------- useFieldAny — load any-rank field ---------- */

export interface UseFieldAnyResult {
  field: NpyN | null;
  loading: boolean;
  error: string | null;
  /** Rust variant behind `error`, so callers branch on the code instead of
   *  matching message text. Null for non-IPC failures. */
  errorCode: AppErrorCode | null;
  requestedStepIndex: number | null;
  requestedPublicationGeneration: string | null;
  loadedStepIndex: number | null;
  loadedPublicationGeneration: string | null;
}

/** Rank-agnostic field loader: returns 2D for surface/column diagnostics, 3D for prognostic fields. */
export function useFieldAny(
  runDir: string,
  stepIndex: number | null,
  fieldName: string | null,
  publicationGeneration: string | null,
  windows?: readonly FieldLoadWindow[] | null,
  continuityKey?: string | null,
): UseFieldAnyResult {
  const budgetedWindows = plotFieldWindows(runDir, stepIndex, fieldName, publicationGeneration, windows);
  const windowsRef = useRef(budgetedWindows);
  windowsRef.current = budgetedWindows;
  const windowsKey = JSON.stringify(budgetedWindows);
  const definitionIdentity = fieldDefinitionIdentity(runDir, stepIndex, fieldName, publicationGeneration);
  const ownedPrefetches = useOwnedFieldPrefetches(stepIndex == null || !fieldName
    ? null : JSON.stringify([runDir, fieldName, windowsKey, definitionIdentity, continuityKey]));
  const [state, setState] = useState<UseFieldAnyResult & {
    sourceSeries: string;
    continuitySeries: string;
  }>({
    field: null,
    loading: false,
    error: null,
    errorCode: null,
    requestedStepIndex: null,
    requestedPublicationGeneration: null,
    loadedStepIndex: null,
    loadedPublicationGeneration: null,
    sourceSeries: "",
    continuitySeries: "",
  });
  const continuityRef = useRef<string>("");
  const budgetRetry = useIoBudgetRetry(state.error);
  const loadScopeRef = useRef<string | null>(null);
  if (loadScopeRef.current == null) {
    loadScopeRef.current = allocateFieldLoadScope("field-any");
  }

  useEffect(() => {
    if (stepIndex == null || !fieldName) {
      setState({
        field: null,
        loading: false,
        error: null,
        errorCode: null,
        requestedStepIndex: null,
        requestedPublicationGeneration: null,
        loadedStepIndex: null,
        loadedPublicationGeneration: null,
        sourceSeries: "",
        continuitySeries: "",
      });
      continuityRef.current = "";
      return;
    }
    let cancelled = false;
    let loadedSuccessfully = false;
    const controller = new AbortController();
    // Same series: keep previous frame for smooth scrubbing; otherwise clear.
    const requestIdentity = JSON.stringify([runDir, publicationGeneration, fieldName, windowsKey, definitionIdentity]);
    const continuity = JSON.stringify([continuityKey ?? JSON.stringify([runDir, fieldName, windowsKey]), definitionIdentity]);
    const sameSeries = continuityRef.current === continuity;
    continuityRef.current = continuity;
    setState((prev) => {
      const identity = beginFrameRequest(
        prev.loadedStepIndex,
        prev.loadedPublicationGeneration,
        stepIndex,
        publicationGeneration,
        sameSeries,
      );
      return {
        field: sameSeries ? prev.field : null,
        loading: true,
        error: null,
        errorCode: null,
        ...identity,
        sourceSeries: requestIdentity,
        continuitySeries: continuity,
      };
    });
    beginFieldLoad();
    const activeWindows = windowsRef.current;
    const load = activeWindows
      ? readFieldAnyWindows(
          runDir,
          stepIndex,
          fieldName,
          publicationGeneration,
          activeWindows,
          {
            latestScope: loadScopeRef.current ?? undefined,
            signal: controller.signal,
          },
        )
      : readFieldAny(runDir, stepIndex, fieldName, publicationGeneration, {
          latestScope: loadScopeRef.current ?? undefined,
          signal: controller.signal,
        });
    load
      .then(
        (f) => {
          if (!cancelled) {
            loadedSuccessfully = true;
            setState({
              field: f,
              loading: false,
              error: null,
              errorCode: null,
              ...completeFrameRequest(stepIndex, publicationGeneration),
              sourceSeries: requestIdentity,
              continuitySeries: continuity,
            });
          }
        },
        (e) => {
          if (cancelled) return;
          if (isFieldLoadAbort(e)) {
            setState((previous) => previous.sourceSeries === requestIdentity
              ? { ...previous, loading: false, error: null, errorCode: null }
              : previous);
            return;
          }
          setState({
              field: null,
              loading: false,
              error: e instanceof Error ? e.message : String(e),
              errorCode: appErrorCode(e),
              requestedStepIndex: stepIndex,
              requestedPublicationGeneration: publicationGeneration,
              loadedStepIndex: null,
              loadedPublicationGeneration: null,
              sourceSeries: requestIdentity,
              continuitySeries: continuity,
            });
        },
      )
      .finally(() => {
        endFieldLoad();
        // Warm what playback is about to ask for, behind this render rather than
        // after it. Unscoped and unawaited: see lib/run/framePrefetch.ts.
        if (cancelled || !loadedSuccessfully) return;
        for (const nextFrame of frameIdentitiesToPrefetch(runDir, stepIndex, publicationGeneration)) {
          // Refuse rather than queue when the global ceiling is reached: a
          // multi-field scene mounts many hooks and speculation must never sit
          // in front of the frame the user is waiting for.
          if (!tryBeginPrefetch()) break;
          const prefetchController = createPrefetchController();
          ownedPrefetches.add(prefetchController);
          const prefetch = activeWindows
            ? readFieldAnyWindows(
                runDir,
                nextFrame.stepIndex,
                fieldName,
                nextFrame.publicationGeneration,
                activeWindows,
                { signal: prefetchController.signal, priority: "background" },
              )
            : readFieldAny(runDir, nextFrame.stepIndex, fieldName, nextFrame.publicationGeneration, {
                signal: prefetchController.signal,
                priority: "background",
              });
          void prefetch
            .catch(() => {})
            .finally(() => {
              ownedPrefetches.delete(prefetchController);
              releasePrefetchController(prefetchController);
              endPrefetch();
            });
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [budgetRetry, runDir, stepIndex, fieldName, publicationGeneration, windowsKey, continuityKey, definitionIdentity, ownedPrefetches]);

  const desiredSeries = stepIndex == null || !fieldName
    ? ""
    : JSON.stringify([runDir, publicationGeneration, fieldName, windowsKey, definitionIdentity]);
  const desiredContinuity = stepIndex == null || !fieldName
    ? ""
    : JSON.stringify([continuityKey ?? JSON.stringify([runDir, fieldName, windowsKey]), definitionIdentity]);
  const continuityMatches = state.continuitySeries === desiredContinuity;
  const currentField = state.sourceSeries === desiredSeries || continuityMatches
    ? state.field
    : null;
  useFieldResourceOwnership(
    loadScopeRef.current,
    currentField,
    `${state.loadedPublicationGeneration ?? "none"}:${state.loadedStepIndex ?? "none"}`,
  );
  if (state.sourceSeries !== desiredSeries) {
    return {
      ...state,
      field: continuityMatches ? state.field : null,
      loading: desiredSeries !== "",
      error: null,
      errorCode: null,
      requestedStepIndex: stepIndex,
      requestedPublicationGeneration: publicationGeneration,
      loadedStepIndex: continuityMatches ? state.loadedStepIndex : null,
      loadedPublicationGeneration: continuityMatches ? state.loadedPublicationGeneration : null,
    };
  }
  return state;
}

/** Gate that fires only when the display scale actually changed, not when other effect deps move. */
export function useSceneScaleFraming(scale: SceneScale): { consume: () => boolean } {
  const framed = useRef<SceneScale | null>(null);
  const latest = useRef(scale);
  latest.current = scale;
  return useMemo(() => ({
    consume: () => {
      const current = latest.current;
      const previous = framed.current;
      if (previous != null && previous.xy === current.xy && previous.z === current.z) {
        return false;
      }
      framed.current = { xy: current.xy, z: current.z };
      return true;
    },
  }), []);
}
