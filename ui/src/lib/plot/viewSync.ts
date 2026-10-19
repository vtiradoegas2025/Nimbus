/* Pure state transitions for stable cross-frame/panel view. Latches hold scale across animation steps;
 * camera-signature dedup breaks echo loops between synced panels. All functions are identity-stable. */

export interface ValueRange {
  min: number;
  max: number;
}

export interface PlaneViewportBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface ScopedPlaneViewport {
  scope: string;
  viewport: PlaneViewportBounds | null;
}

export function planeViewportForScope(state: ScopedPlaneViewport, scope: string): PlaneViewportBounds | null {
  return state.scope === scope ? state.viewport : null;
}

function relayoutAxisRange(
  event: Readonly<Record<string, unknown>>,
  axis: "xaxis" | "yaxis",
): [number, number] | null | undefined {
  if (event[`${axis}.autorange`] === true) return null;
  const combined = event[`${axis}.range`];
  if (Array.isArray(combined) && combined.length === 2) {
    const values = combined.map(Number);
    if (values.every(Number.isFinite)) return values[0] <= values[1]
      ? [values[0], values[1]] : [values[1], values[0]];
  }
  const first = Number(event[`${axis}.range[0]`]);
  const second = Number(event[`${axis}.range[1]`]);
  if (!Number.isFinite(first) || !Number.isFinite(second)) return undefined;
  return first <= second ? [first, second] : [second, first];
}

export function planeViewportFromRelayout(
  event: Readonly<Record<string, unknown>>,
  previous: PlaneViewportBounds | null,
): PlaneViewportBounds | null | undefined {
  const x = relayoutAxisRange(event, "xaxis");
  const y = relayoutAxisRange(event, "yaxis");
  if (x === null || y === null) return null;
  const resolvedX = x ?? (previous ? [previous.xMin, previous.xMax] as const : null);
  const resolvedY = y ?? (previous ? [previous.yMin, previous.yMax] as const : null);
  if (!resolvedX || !resolvedY) return undefined;
  return {
    xMin: resolvedX[0], xMax: resolvedX[1],
    yMin: resolvedY[0], yMax: resolvedY[1],
  };
}

/** A range held across frames, tagged with the identity it belongs to. */
export interface LatchedRange {
  scope: string;
  range: ValueRange | null;
}

/** A magnitude held across frames, tagged with the identity it belongs to. */
export interface LatchedMagnitude {
  scope: string;
  reference: number;
}

/** Range carried into `scope`; null when latch belongs to a different scope. Resolve during render, not in effects. */
export function carriedRange(previous: LatchedRange, scope: string): ValueRange | null {
  return previous.scope === scope ? previous.range : null;
}

/** Widen the latch to cover `frameRange`; union only grows within a scope. Scope change resets. */
export function latchRange(
  previous: LatchedRange,
  scope: string,
  frameRange: ValueRange | null,
): LatchedRange {
  const carried = carriedRange(previous, scope);
  const merged: ValueRange | null = carried && frameRange
    ? {
        min: Math.min(carried.min, frameRange.min),
        max: Math.max(carried.max, frameRange.max),
      }
    : frameRange ?? carried;
  const unchanged = previous.scope === scope
    && previous.range?.min === merged?.min
    && previous.range?.max === merged?.max;
  return unchanged ? previous : { scope, range: merged };
}

/** The magnitude carried into `scope` — zero when the latch belongs to a
 *  different one. Resolved during render; see carriedRange. */
export function carriedMagnitude(previous: LatchedMagnitude, scope: string): number {
  return previous.scope === scope ? previous.reference : 0;
}

/** Raise the latch to cover `frameMax`. `frameMax` must be data-space magnitude, not display-stride-dependent. */
export function latchMagnitude(
  previous: LatchedMagnitude,
  scope: string,
  frameMax: number,
): LatchedMagnitude {
  const carried = carriedMagnitude(previous, scope);
  if (frameMax > carried) return { scope, reference: frameMax };
  // Re-anchor a carried-over entry to the current scope so the stale one is not
  // read again; otherwise nothing moved and the previous state stands.
  return previous.scope === scope ? previous : { scope, reference: carried };
}

/** Stable string identity for a Plotly 3-D camera; compare to detect echoes in synced panels.
 * Returns "" for non-camera objects. */
export function cameraSignature(camera: unknown): string {
  if (!camera || typeof camera !== "object" || Array.isArray(camera)) return "";
  const parts = camera as Record<string, { x?: number; y?: number; z?: number } | undefined>;
  const components = (["eye", "center", "up"] as const).map((part) => {
    const vector = parts[part];
    return vector && typeof vector === "object"
      ? `${vector.x ?? ""},${vector.y ?? ""},${vector.z ?? ""}`
      : "";
  });
  // An object carrying none of the three vectors is not a camera. Signing it
  // anyway would yield a constant non-empty key, so the first such event would
  // be broadcast and every later one silently swallowed as its own echo.
  return components.some((component) => component !== "") ? components.join("|") : "";
}

/** Axis bounds with VIEW_BOUND_MARGIN margin; prevents zoom-out past the simulated domain.
 * Returns undefined for degenerate extents so callers can spread unconditionally. */
export const VIEW_BOUND_MARGIN = 0.15;

export function axisViewBounds(
  values: readonly number[] | null | undefined,
): { minallowed: number; maxallowed: number } | undefined {
  if (!values || values.length === 0) return undefined;
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) return undefined;
  const margin = (max - min) * VIEW_BOUND_MARGIN;
  return { minallowed: min - margin, maxallowed: max + margin };
}
