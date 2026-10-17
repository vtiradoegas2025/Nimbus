/*
 * Handing a 2-D scalar view to a volume render, carrying the field and its display settings.
 * Pure — no React. Unit-tested in scripts/plot/test_volumetric_handoff.ts.
 */
import { isPlotAvailable, type PlotType } from "./plotRegistry";
import type { PlotPreset } from "./plotPresets";

/**
 * Renderers showing a scalar on a plane through a live 3-D array, so the same data
 * also has a volume rendering. The quiver plot is excluded on purpose: it carries a
 * vector pair rather than a scalar, and its 3-D counterpart is streamlines.
 */
export const VOLUMETRIC_SOURCE_TYPES: readonly PlotType[] = ["slice", "composite"];

/** Rank of an array that carries a vertical dimension. Anything else has no volume to render. */
const VOLUMETRIC_RANK = 3;

/** What a 2-D scalar renderer currently shows. Reported upward so the handoff can reuse it. */
export interface ScalarSelection {
  field: string;
  colormap: PlotPreset["colormap"];
  rangeMode: PlotPreset["rangeMode"];
  /**
   * Rank of the array actually loaded, not a guess from the field name. A column
   * product such as composite_reflectivity is rank 2 and has no volume.
   */
  rank: number | null;
  source: "slice" | "composite";
  planeFocus?: PlotPreset["volumeFocus"] | null;
}

export function planeFocusFromView(
  axis: 0 | 1 | 2,
  sourceIndex: number,
  coordinateM: number | null,
  rowAxis: 0 | 1 | 2,
  colAxis: 0 | 1 | 2,
  viewportKm: { xMin: number; xMax: number; yMin: number; yMax: number } | null,
): NonNullable<ScalarSelection["planeFocus"]> {
  const centerM: [number | null, number | null, number | null] = [null, null, null];
  centerM[axis] = Number.isFinite(coordinateM) ? coordinateM : null;
  if (viewportKm) {
    centerM[colAxis] = 500 * (viewportKm.xMin + viewportKm.xMax);
    centerM[rowAxis] = 500 * (viewportKm.yMin + viewportKm.yMax);
  }
  return { axis, sourceIndex, coordinateM, centerM };
}

export function translatedVolumeCameraFocus(
  focalPoint: readonly [number, number, number],
  position: readonly [number, number, number],
  requested: readonly [number | null, number | null, number | null],
  bounds: readonly [number, number, number, number, number, number] | null,
): { focalPoint: [number, number, number]; position: [number, number, number] } {
  const nextFocal: [number, number, number] = [...focalPoint];
  const nextPosition: [number, number, number] = [...position];
  for (let axis = 0; axis < 3; axis += 1) {
    const value = requested[axis];
    if (value == null || !Number.isFinite(value)) continue;
    const target = bounds
      ? Math.min(bounds[axis * 2 + 1], Math.max(bounds[axis * 2], value))
      : value;
    const delta = target - nextFocal[axis];
    nextFocal[axis] = target;
    nextPosition[axis] += delta;
  }
  return { focalPoint: nextFocal, position: nextPosition };
}

/**
 * The offer a scalar widget renders in its own control row. Present only when the host
 * can act on it, so the widget decides nothing about availability itself.
 */
export interface VolumeHandoffOffer {
  available: boolean;
  /** Why it is unavailable, for the disabled control's tooltip; null when available. */
  blockedReason: string | null;
  field: string | null;
  label: string;
  onOpen: () => void;
}

const VOLUME_HANDOFF_ID_PREFIX = "handoff-volume";

/** True for any id minted by `volumeHandoffPreset`; these are ephemeral and never in PLOT_PRESETS. */
export function isVolumeHandoffPresetId(id: string): boolean {
  return id.startsWith(`${VOLUME_HANDOFF_ID_PREFIX}:`);
}

/** True when this view can be reproduced as a volume render of the same field. */
export function canHandOffToVolume(
  plotType: PlotType,
  selection: ScalarSelection | null,
  available: ReadonlySet<string>,
): boolean {
  if (!VOLUMETRIC_SOURCE_TYPES.includes(plotType)) return false;
  if (!selection || !selection.field) return false;
  if (selection.rank !== VOLUMETRIC_RANK) return false;
  return isPlotAvailable("volume", available);
}

/**
 * Why the handoff is unavailable, for a disabled control's tooltip. Returns null
 * when it is available.
 */
export function describeVolumeHandoffBlock(
  plotType: PlotType,
  selection: ScalarSelection | null,
  available: ReadonlySet<string>,
): string | null {
  if (plotType === "vector") {
    return "A quiver shows a vector pair. Use the streamlines renderer for its 3D form.";
  }
  if (!VOLUMETRIC_SOURCE_TYPES.includes(plotType)) {
    return "This plot reduces the data rather than slicing it, so it has no volume to show.";
  }
  if (!selection || !selection.field) return "Choose a field first.";
  if (selection.rank !== VOLUMETRIC_RANK) {
    return `${selection.field} is stored as a single level, so there is no volume to render.`;
  }
  if (!isPlotAvailable("volume", available)) {
    return "This run exports no field the volume renderer can read.";
  }
  return null;
}

/**
 * An ephemeral preset carrying the 2-D view's field and display settings into the
 * volume renderer. `requires` names the field so the pane gates on it like any preset.
 *
 * `sequence` must differ per invocation: the volume renderer re-applies a preset only
 * when its id changes, so a fixed id would leave a second handoff showing the first field.
 */
export function volumeHandoffPreset(selection: ScalarSelection, sequence: number): PlotPreset {
  const sourceDescription = selection.source === "composite"
    ? "Composite fill only; 2-D contour overlays remain in their source pane."
    : "Scalar volume corresponding to the selected plane field.";
  return {
    id: `${VOLUME_HANDOFF_ID_PREFIX}:${sequence}`,
    label: "3D volume",
    description: `Volume render of ${selection.field}, carried over from the 2-D view. ${sourceDescription}`,
    category: "volumetric",
    plotType: "volume",
    field: selection.field,
    colormap: selection.colormap,
    rangeMode: selection.rangeMode,
    volumeFocus: selection.planeFocus ?? undefined,
    requires: [[selection.field]],
  };
}
