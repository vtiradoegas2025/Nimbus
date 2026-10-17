import { cartesianAxisSpacing as gridAxisSpacing, cartesianDyMeters, cartesianScalarAxisOrigin as gridAxisOrigin } from "@/lib/science/cartesianCoordinates";
import type { FieldLoadWindow, GridInfo } from "@/lib/run/runOutputs";
import { sliceDisplayAxes, type SliceAxis } from "@/lib/platform/npy";
import {
  MAX_NATIVE_FIELD_WINDOW_BYTES,
  MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
  nativeFieldWindowBodyBytes,
  type FieldWindowAggregation,
} from "@/lib/dataset/nativeFieldWindow";

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;

export interface BoundedVolumeWindowPlan {
  windows: FieldLoadWindow[];
  sourceShape: [number, number, number];
  sampledShape: [number, number, number];
  stride: [number, number, number];
  sourceCells: number;
  sampledCells: number;
  fullSourceShape?: [number, number, number];
  viewportBounds?: PlaneViewportBounds;
}

export interface PlaneViewportBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/** User-facing provenance for any renderer consuming a bounded reduced lattice. */
export function describeBoundedSampling(
  plan: Pick<BoundedVolumeWindowPlan, "sourceShape" | "sampledShape" | "stride"> | null | undefined,
  fieldName: string | null | undefined,
): string | null {
  if (!plan || plan.stride.every((stride) => stride === 1)) return null;
  const method = fieldWindowAggregation(fieldName) === "linear_reflectivity_mean"
    ? "bounded linear-Z mean"
    : "bounded cell-mean";
  return `${method} ${plan.stride.join("×")} · source ${plan.sourceShape.join("×")} → display ${plan.sampledShape.join("×")}`;
}

export interface NativeStormDetailWindowPlan extends BoundedVolumeWindowPlan {
  /** Inclusive storage indices selected before the ordinary bounded tiler runs. */
  detailIndexRange: {
    lower: [number, number, number];
    upper: [number, number, number];
  };
  /** Requested feature centre after clamping it to the valid physical domain. */
  detailCenterM: [number, number];
  /** Physical cell-centre span actually retained by the native detail lattice. */
  detailSpanM: [number, number, number];
}

export function isNativeStormDetailWindowPlan(
  plan: BoundedVolumeWindowPlan | NativeStormDetailWindowPlan,
): plan is NativeStormDetailWindowPlan {
  return "detailSpanM" in plan
    && "detailIndexRange" in plan
    && "detailCenterM" in plan;
}

/** Move a native ROI only after its feature traverses a meaningful fraction of
 * the retained horizontal span. The cap preserves the established mesoscale
 * behavior; tiny high-resolution bricks receive correspondingly small, but
 * nonzero, hysteresis instead of the former fixed 1.4 km dead zone. */
export function nativeDetailRecenteringThresholdM(
  plan: NativeStormDetailWindowPlan | null | undefined,
  maximumM = 1_400,
): number {
  if (!plan) return Math.max(1, maximumM);
  const horizontalSpan = Math.min(plan.detailSpanM[0], plan.detailSpanM[1]);
  const nativeCell = Math.max(
    1,
    plan.detailSpanM[0] / Math.max(1, plan.sourceShape[0]),
    plan.detailSpanM[1] / Math.max(1, plan.sourceShape[1]),
  );
  return Math.max(nativeCell * 4, Math.min(maximumM, horizontalSpan * 0.2));
}

export interface ExactWindowPlan {
  windows: FieldLoadWindow[];
  shape: number[];
  cells: number;
}

export type ExactWindowOutcome =
  | { ok: true; plan: ExactWindowPlan }
  | { ok: false; reason: string };

/** Plotly heatmaps above this are reduced before IPC; roughly a 2048² plane. */
export const PLOTLY_HEATMAP_CELL_CAP = 4_194_304;

/** Above 512 MiB of float32 source per field, an interactive "reduced" full-
 * domain view is still a pathological raw scan. Extreme runs must enter through
 * a tracker/domain-centred native ROI or a published multiresolution product. */
export const MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS = 134_217_728;

export type BoundedVolumeWindowOutcome =
  | { ok: true; plan: BoundedVolumeWindowPlan }
  | { ok: false; reason: string };

export type NativeStormDetailWindowOutcome =
  | { ok: true; plan: NativeStormDetailWindowPlan }
  | { ok: false; reason: string };

export type InteractiveVolumeWindowOutcome =
  | {
      ok: true;
      plan: BoundedVolumeWindowPlan | NativeStormDetailWindowPlan;
      mode: "full-domain" | "native-detail";
    }
  | { ok: false; reason: string };

/** Mesocyclone-scale native region. At 50 m this is 280×280×200 = 15.68M cells. */
export const STORM_DETAIL_TARGET_SPAN_M: readonly [number, number, number] = [
  14_000,
  14_000,
  10_000,
];

/** A native extreme-detail brick must be physically capable of resolving the
 * 500 m connected-depth tornado criterion. Extra vertical margin prevents a
 * qualifying surface from being clipped at the top of the ROI. */
export const EXTREME_DETAIL_MIN_VERTICAL_SPAN_M = 600;
const EXTREME_DETAIL_MIN_HORIZONTAL_CELLS = 16;

/**
 * Full-domain reduction is acceptable only while its raw source scan remains
 * bounded. Beyond that point every general 3D plot enters through the same
 * native, metadata/tracker-centred ROI as the supercell renderer. This makes
 * the source-cell ceiling global rather than a view-specific convention.
 */
export function planInteractiveVolumeWindows(
  grid: GridInfo | null | undefined,
  maxVoxels: number,
  preferredCenterM?: readonly [number, number] | null,
  options: {
    preferNativeDetail?: boolean;
    structuralSupportBoundsM?: {
      lower: readonly [number, number, number];
      upper: readonly [number, number, number];
    } | null;
  } = {},
): InteractiveVolumeWindowOutcome {
  if (interactiveFullDomainSourceSafe(grid)) {
    const native = planBoundedVolumeWindows(grid, maxVoxels);
    if (native.ok && native.plan.stride.every((stride) => stride === 1)) {
      return { ok: true, plan: native.plan, mode: "full-domain" };
    }
  }
  if (options.preferNativeDetail
      && preferredCenterM?.length === 2
      && preferredCenterM.every(Number.isFinite)) {
    if (options.structuralSupportBoundsM) {
      const support = planNativeStructuralSupportWindows(
        grid,
        options.structuralSupportBoundsM,
        maxVoxels,
      );
      if (support.ok) return { ok: true, plan: support.plan, mode: "native-detail" };
      // A known support envelope that does not fit is not permission to crop
      // it to an unrelated fixed box. Keep the complete reduced overview when
      // its source scan is bounded; extreme sources still require the native
      // ROI because scanning the whole domain is itself inadmissible.
      if (interactiveFullDomainSourceSafe(grid)) {
        const overview = planBoundedVolumeWindows(grid, maxVoxels);
        return overview.ok
          ? { ok: true, plan: overview.plan, mode: "full-domain" }
          : overview;
      }
    }
    const detail = planNativeStormDetailWindows(grid, preferredCenterM, maxVoxels);
    if (detail.ok) return { ok: true, plan: detail.plan, mode: "native-detail" };
  }
  if (interactiveFullDomainSourceSafe(grid)) {
    const outcome = planBoundedVolumeWindows(grid, maxVoxels);
    return outcome.ok
      ? { ok: true, plan: outcome.plan, mode: "full-domain" }
      : outcome;
  }
  const center = preferredCenterM && preferredCenterM.every(Number.isFinite)
    ? preferredCenterM
    : gridHorizontalCenterM(grid);
  const detail = planNativeStormDetailWindows(grid, center, maxVoxels);
  return detail.ok
    ? { ok: true, plan: detail.plan, mode: "native-detail" }
    : detail;
}

/** Published fields aggregate natively. Built-in derived fields use the same
 * geometry plan but evaluate exact raw tiles before applying the reduction. */
export function fieldSupportsBoundedWindows(
  origin: "native" | "studio_derived" | "user_expression" | "external_provider" | undefined,
): boolean {
  return origin == null || origin === "native" || origin === "studio_derived"
    || origin === "user_expression" || origin === "external_provider";
}

/** dBZ is logarithmic: arithmetic means in dBZ space are not reflectivity means. */
export function fieldWindowAggregation(fieldName: string | null | undefined): FieldWindowAggregation {
  const normalized = fieldName?.trim().toLowerCase() ?? "";
  return normalized.includes("dbz") || normalized.includes("reflectiv")
    ? "linear_reflectivity_mean"
    : "cell_mean";
}

/** Reuse one geometry plan across composite layers while retaining field-specific physics. */
export function windowsForFieldAggregation(
  windows: readonly FieldLoadWindow[],
  fieldName: string | null | undefined,
  options: {
    sourceKind?: "native" | "studio_derived" | "user_expression" | "external_provider";
    terrainFollowing?: boolean;
  } = {},
): FieldLoadWindow[] {
  const aggregation = fieldWindowAggregation(fieldName);
  const mapped = windows.map((window) => ({ ...window, aggregation }));
  return coalesceNativeFieldWindows(mapped, options);
}

/** Preserve a caller's extrema/point aggregation while removing avoidable native tiling. */
export function coalesceNativeFieldWindows(
  windows: readonly FieldLoadWindow[],
  options: {
    sourceKind?: "native" | "studio_derived" | "user_expression" | "external_provider";
    terrainFollowing?: boolean;
  } = {},
): FieldLoadWindow[] {
  const mapped = windows.map((window) => ({ ...window }));
  if (mapped.length <= 1 || options.sourceKind !== "native" || options.terrainFollowing) return mapped;
  const rank = mapped[0].origin.length;
  const stride = mapped[0].stride;
  if (rank !== 3 || mapped.some((window) =>
    window.origin.length !== rank
      || window.extent.length !== rank
      || window.stride.length !== rank
      || window.aggregation !== mapped[0].aggregation
      || window.stride.some((value, axis) => value !== stride[axis])
  )) return mapped;
  const origin = Array.from({ length: rank }, (_, axis) =>
    Math.min(...mapped.map((window) => window.origin[axis]))
  );
  const upper = Array.from({ length: rank }, (_, axis) =>
    Math.max(...mapped.map((window) => window.origin[axis] + window.extent[axis]))
  );
  const extent = upper.map((value, axis) => value - origin[axis]);
  const sourceBytes = product(extent) * FLOAT32_BYTES;
  const outputBytes = product(extent.map((count, axis) => outputCount(count, stride[axis])))
    * FLOAT32_BYTES;
  if (sourceBytes > MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
      || outputBytes > MAX_NATIVE_FIELD_WINDOW_BYTES) return mapped;
  if (mapped.some(window => window.origin.some((value, axis) =>
    (value - origin[axis]) % stride[axis] !== 0
      || window.origin[axis] + window.extent[axis] < upper[axis] && window.extent[axis] % stride[axis] !== 0))) return mapped;
  if (mapped.reduce((cells, window) => cells + product(window.extent), 0) !== product(extent)) return mapped;
  for (let i = 0; i < mapped.length; i++) {
    for (let j = i + 1; j < mapped.length; j++) {
      if (mapped[i].origin.every((value, axis) => value < mapped[j].origin[axis] + mapped[j].extent[axis]
          && mapped[j].origin[axis] < value + mapped[i].extent[axis])) return mapped;
    }
  }
  // Native aggregation streams the source and emits the final reduced lattice,
  // so one admitted request avoids N IPC envelopes, N tile arrays and a JS assembly pass.
  return [{
    aggregation: mapped[0].aggregation,
    origin,
    extent,
    stride: [...stride],
    maxOutputBytes: outputBytes,
    maxSourceBytes: sourceBytes,
  }];
}

function finitePositiveInteger(value: number | undefined): value is number {
  return value != null && Number.isSafeInteger(value) && value > 0;
}

function product(values: readonly number[]): number {
  return values.reduce((total, value) => total * value, 1);
}

function outputCount(extent: number, stride: number): number {
  return Math.floor((extent - 1) / stride) + 1;
}

/** Smallest declared/coordinate-derived model spacing used only for renderer
 * fidelity policy. Returns null rather than inventing a physical resolution. */
export function minimumNominalGridSpacingM(
  grid: GridInfo | null | undefined,
): number | null {
  if (!grid) return null;
  const counts = [grid.nr, grid.nth, grid.nz];
  const declared = [grid.drM, cartesianDyMeters(grid, grid.drM ?? Number.NaN), grid.dzM];
  const spacing = [0, 1, 2].map((axis) => {
    const direct = declared[axis];
    if (direct != null && Number.isFinite(direct) && direct > 0) return direct;
    const coordinates = grid.scalarCoordinatesM?.[axis];
    const count = counts[axis];
    if (coordinates && count === coordinates.length && coordinates.length > 1) {
      const span = coordinates.at(-1)! - coordinates[0]!;
      if (Number.isFinite(span) && span > 0) return span / (coordinates.length - 1);
    }
    return Number.NaN;
  });
  return spacing.every((value) => Number.isFinite(value) && value > 0)
    ? Math.min(...spacing)
    : null;
}

export function validGridCellCount(grid: GridInfo | null | undefined): number | null {
  if (!grid || !finitePositiveInteger(grid.nr)
      || !finitePositiveInteger(grid.nth) || !finitePositiveInteger(grid.nz)) return null;
  const shape = [grid.nr, grid.nth, grid.nz];
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? shape.map((count) => count - 1);
  const counts = upper.map((value, axis) => Math.trunc(value) - Math.trunc(lower[axis]) + 1);
  if (counts.some((count, axis) => !finitePositiveInteger(count)
      || lower[axis] < 0 || upper[axis] >= shape[axis])) return null;
  const cells = product(counts);
  return Number.isSafeInteger(cells) ? cells : null;
}

export function interactiveFullDomainSourceSafe(
  grid: GridInfo | null | undefined,
  ceiling = MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS,
): boolean {
  const cells = validGridCellCount(grid);
  return cells != null && cells <= ceiling;
}

function gridAxisCoordinate(grid: GridInfo, axis: number, index: number): number {
  const count = axis === 0 ? grid.nr! : axis === 1 ? grid.nth! : grid.nz!;
  const exact = grid.scalarCoordinatesM?.[axis];
  if (exact?.length === count && Number.isFinite(exact[index])) return exact[index];
  const spacing = gridAxisSpacing(grid, axis);
  return gridAxisOrigin(grid, axis, count, spacing) + index * spacing;
}

/** Metadata-only bootstrap used when scanning an extreme full domain would be
 * more expensive than the entire retained renderer budget. */
export function gridHorizontalCenterM(
  grid: GridInfo | null | undefined,
): [number, number] | null {
  if (!grid || !finitePositiveInteger(grid.nr) || !finitePositiveInteger(grid.nth)) return null;
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? [grid.nr - 1, grid.nth - 1, (grid.nz ?? 1) - 1];
  const indices = [0, 1].map((axis) => Math.round(
    (Math.trunc(lower[axis]) + Math.trunc(upper[axis])) * 0.5,
  ));
  if (indices.some((index, axis) => index < 0 || index >= (axis === 0 ? grid.nr! : grid.nth!))) {
    return null;
  }
  return [
    gridAxisCoordinate(grid, 0, indices[0]),
    gridAxisCoordinate(grid, 1, indices[1]),
  ];
}

export function clampGridHorizontalPointM(
  grid: GridInfo | null | undefined,
  point: readonly number[],
): [number, number] | null {
  if (!grid || !finitePositiveInteger(grid.nr) || !finitePositiveInteger(grid.nth)
      || point.length < 2 || !point.slice(0, 2).every(Number.isFinite)) return null;
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? [grid.nr - 1, grid.nth - 1, 0];
  if ([0, 1].some(axis => !Number.isSafeInteger(lower[axis]) || !Number.isSafeInteger(upper[axis])
      || lower[axis] < 0 || upper[axis] < lower[axis] || upper[axis] >= (axis === 0 ? grid.nr! : grid.nth!))) return null;
  const center = [0, 1].map(axis => {
    const count = axis === 0 ? grid.nr! : grid.nth!;
    const first = gridAxisCoordinate(grid, axis, lower[axis]);
    const last = gridAxisCoordinate(grid, axis, upper[axis]);
    const before = lower[axis] > 0 ? gridAxisCoordinate(grid, axis, lower[axis] - 1)
      : first - (count > 1 ? gridAxisCoordinate(grid, axis, 1) - first : gridAxisSpacing(grid, axis));
    const after = upper[axis] + 1 < count ? gridAxisCoordinate(grid, axis, upper[axis] + 1)
      : last + (count > 1 ? last - gridAxisCoordinate(grid, axis, count - 2) : gridAxisSpacing(grid, axis));
    const minimum = Math.max((first + before) * 0.5, grid.physicalBoundsM?.lower[axis] ?? -Infinity);
    const maximum = Math.min((last + after) * 0.5, grid.physicalBoundsM?.upper[axis] ?? Infinity);
    return minimum <= maximum ? Math.max(minimum, Math.min(maximum, point[axis])) : NaN;
  }) as [number, number];
  return center.every(Number.isFinite) ? center : null;
}

function nearestGridIndex(
  grid: GridInfo,
  axis: 0 | 1 | 2,
  coordinateM: number,
  lower: number,
  upper: number,
): number {
  const exact = grid.scalarCoordinatesM?.[axis];
  const count = axis === 0 ? grid.nr! : axis === 1 ? grid.nth! : grid.nz!;
  if (exact?.length === count) {
    let lo = lower;
    let hi = upper;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (exact[mid] < coordinateM) lo = mid + 1;
      else hi = mid;
    }
    if (lo > lower && Math.abs(exact[lo - 1] - coordinateM) <= Math.abs(exact[lo] - coordinateM)) {
      return lo - 1;
    }
    return lo;
  }
  const spacing = gridAxisSpacing(grid, axis);
  const origin = gridAxisOrigin(grid, axis, count, spacing);
  return Math.max(lower, Math.min(upper, Math.round((coordinateM - origin) / spacing)));
}

/**
 * Select a storm-following region that remains stride-1 under the renderer budget.
 *
 * The full-domain ceiling remains a memory safety boundary. This planner removes
 * it as a feature-resolution boundary by spending those cells only around the
 * tracked/located circulation, before any field bytes cross IPC.
 */
export function planNativeStormDetailWindows(
  grid: GridInfo | null | undefined,
  centerM: readonly [number, number] | null | undefined,
  maxVoxels: number,
  targetSpanM: readonly [number, number, number] = STORM_DETAIL_TARGET_SPAN_M,
  centerZM?: number,
  preserveFullHeight = false,
): NativeStormDetailWindowOutcome {
  if (!grid || !finitePositiveInteger(grid.nr)
      || !finitePositiveInteger(grid.nth) || !finitePositiveInteger(grid.nz)) {
    return { ok: false, reason: "The frame does not declare a complete 3D grid." };
  }
  if (!centerM || !centerM.every(Number.isFinite)) {
    return { ok: false, reason: "No finite storm centre is available for native detail." };
  }
  if (!Number.isSafeInteger(maxVoxels) || maxVoxels < 1) {
    return { ok: false, reason: "The renderer supplied an invalid native-detail budget." };
  }
  if (targetSpanM.some((span) => !Number.isFinite(span) || span <= 0)) {
    return { ok: false, reason: "The native-detail physical span is invalid." };
  }

  const fullShape: [number, number, number] = [grid.nr, grid.nth, grid.nz];
  const validLower = (grid.validIndexRange?.lower ?? [0, 0, 0]).map(Math.trunc) as
    [number, number, number];
  const validUpper = (grid.validIndexRange?.upper
    ?? fullShape.map((count) => count - 1)).map(Math.trunc) as [number, number, number];
  const validCounts = validUpper.map((upper, axis) => upper - validLower[axis] + 1) as
    [number, number, number];
  if (validCounts.some((count, axis) => !finitePositiveInteger(count)
      || validLower[axis] < 0 || validUpper[axis] >= fullShape[axis])) {
    return { ok: false, reason: "The frame's valid-domain indices are inconsistent with its grid." };
  }
  if (validCounts.some(count => count < 2) || maxVoxels < 8) {
    return { ok: false, reason: "Native 3D detail requires at least two samples per axis." };
  }

  const nominalSpacing = [0, 1, 2].map((axis) => {
    const coordinates = grid.scalarCoordinatesM?.[axis];
    if (coordinates?.length === fullShape[axis] && validCounts[axis] > 1) {
      const first = coordinates[validLower[axis]];
      const last = coordinates[validUpper[axis]];
      if (Number.isFinite(first) && Number.isFinite(last) && last > first) {
        return (last - first) / (validCounts[axis] - 1);
      }
    }
    return gridAxisSpacing(grid, axis);
  }) as [number, number, number];
  if (nominalSpacing.some((spacing) => !Number.isFinite(spacing) || spacing <= 0)) {
    return { ok: false, reason: "The native-detail grid spacing is invalid." };
  }
  let counts = targetSpanM.map((span, axis) => Math.max(
    2,
    Math.min(validCounts[axis], Math.round(span / nominalSpacing[axis])),
  )) as [number, number, number];
  if (preserveFullHeight) {
    if (validCounts[2] * 4 > maxVoxels) {
      return { ok: false, reason: "The native cloud budget cannot retain a full 3D vertical column." };
    }
    counts[2] = validCounts[2];
  }
  const desiredCells = product(counts);
  if (desiredCells > maxVoxels) {
    const scale = preserveFullHeight
      ? Math.sqrt(maxVoxels / desiredCells) : Math.cbrt(maxVoxels / desiredCells);
    counts = counts.map((count, axis) => preserveFullHeight && axis === 2
      ? count : Math.max(2, Math.floor(count * scale))) as
      [number, number, number];
  }
  // At sub-30 m spacing, isotropic shrinking eventually creates a shallow
  // cube that cannot contain the physical vertical continuity it is meant to
  // diagnose. Reserve the minimum low-level column when the budget can also
  // retain a nontrivial horizontal stencil; the ordinary 30–50 m storm box is
  // already deeper and is unchanged by this branch.
  const extremeVerticalCount = Math.min(
    validCounts[2],
    Math.max(1, Math.ceil(EXTREME_DETAIL_MIN_VERTICAL_SPAN_M / nominalSpacing[2])),
  );
  const minimumHorizontalCounts = [0, 1].map((axis) => Math.min(
    validCounts[axis],
    EXTREME_DETAIL_MIN_HORIZONTAL_CELLS,
  ));
  const extremeMinimumCells = minimumHorizontalCounts[0]
    * minimumHorizontalCounts[1]
    * extremeVerticalCount;
  const reserveExtremeVertical = desiredCells > maxVoxels
    && Math.min(...nominalSpacing) <= 30
    && extremeVerticalCount > counts[2]
    && extremeMinimumCells <= maxVoxels;
  if (reserveExtremeVertical) counts[2] = extremeVerticalCount;
  while (product(counts) > maxVoxels) {
    let axis = -1;
    for (let candidate = 0; candidate < 3; candidate += 1) {
      const minimum = preserveFullHeight && candidate === 2 ? validCounts[2] : reserveExtremeVertical
        ? candidate === 2 ? extremeVerticalCount : minimumHorizontalCounts[candidate]
        : 2;
      if (counts[candidate] <= minimum) continue;
      if (axis < 0) {
        axis = candidate;
        continue;
      }
      if (counts[candidate] * nominalSpacing[candidate]
          > counts[axis] * nominalSpacing[axis]) axis = candidate;
    }
    if (axis < 0) {
      return { ok: false, reason: "The native-detail budget cannot retain one 3D cell." };
    }
    counts[axis] -= 1;
  }

  const centreIndices: [number, number, number] = [
    nearestGridIndex(grid, 0, centerM[0], validLower[0], validUpper[0]),
    nearestGridIndex(grid, 1, centerM[1], validLower[1], validUpper[1]),
    centerZM == null
      ? validLower[2] + Math.floor((counts[2] - 1) / 2)
      : nearestGridIndex(grid, 2, centerZM, validLower[2], validUpper[2]),
  ];
  const detailLower: [number, number, number] = [0, 0, validLower[2]];
  for (const axis of [0, 1, 2] as const) {
    if (axis === 2 && centerZM == null) continue;
    const centered = centreIndices[axis] - Math.floor((counts[axis] - 1) / 2);
    detailLower[axis] = Math.max(
      validLower[axis],
      Math.min(validUpper[axis] - counts[axis] + 1, centered),
    );
  }
  const detailUpper = detailLower.map((lower, axis) => lower + counts[axis] - 1) as
    [number, number, number];
  const bounded = planBoundedVolumeWindows({
    ...grid,
    validIndexRange: { lower: detailLower, upper: detailUpper },
  }, maxVoxels);
  if (!bounded.ok) return bounded;
  if (bounded.plan.stride.some((stride) => stride !== 1)) {
    return { ok: false, reason: "The selected storm-detail region did not remain native." };
  }

  const clampedCenter: [number, number] = [
    gridAxisCoordinate(grid, 0, centreIndices[0]),
    gridAxisCoordinate(grid, 1, centreIndices[1]),
  ];
  const detailSpan = detailLower.map((lower, axis) => {
    const first = gridAxisCoordinate(grid, axis, lower);
    const last = gridAxisCoordinate(grid, axis, detailUpper[axis]);
    return Math.max(nominalSpacing[axis], last - first + nominalSpacing[axis]);
  }) as [number, number, number];
  return {
    ok: true,
    plan: {
      ...bounded.plan,
      detailIndexRange: { lower: detailLower, upper: detailUpper },
      detailCenterM: clampedCenter,
      detailSpanM: detailSpan,
    },
  };
}

export function nativeDetailCoversDomain(
  grid: GridInfo | null | undefined,
  maxVoxels: number,
  targetSpanM: readonly [number, number, number],
  preserveFullHeight = false,
): boolean {
  const result = planNativeStormDetailWindows(grid, gridHorizontalCenterM(grid), maxVoxels, targetSpanM, undefined, preserveFullHeight);
  if (!grid || !result.ok) return false;
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? [grid.nr! - 1, grid.nth! - 1, grid.nz! - 1];
  return result.plan.detailIndexRange.lower.every((value, axis) => value === lower[axis])
    && result.plan.detailIndexRange.upper.every((value, axis) => value === upper[axis]);
}

export function planNativeStructuralSupportWindows(
  grid: GridInfo | null | undefined,
  boundsM: {
    lower: readonly [number, number, number];
    upper: readonly [number, number, number];
  } | null | undefined,
  maxVoxels: number,
): NativeStormDetailWindowOutcome {
  if (!grid || !boundsM
      || ![...boundsM.lower, ...boundsM.upper].every(Number.isFinite)
      || boundsM.lower.some((value, axis) => value > boundsM.upper[axis])) {
    return { ok: false, reason: "No finite structural-support bounds are available." };
  }
  const spacing = [0, 1, 2].map((axis) => gridAxisSpacing(grid, axis)) as
    [number, number, number];
  const targetSpan = boundsM.lower.map((lower, axis) =>
    boundsM.upper[axis] - lower + 3 * spacing[axis]) as [number, number, number];
  const center = boundsM.lower.map((lower, axis) =>
    0.5 * (lower + boundsM.upper[axis])) as [number, number, number];
  const planned = planNativeStormDetailWindows(
    grid,
    [center[0], center[1]],
    maxVoxels,
    targetSpan,
    center[2],
  );
  if (!planned.ok) return planned;
  const range = planned.plan.detailIndexRange;
  for (const axis of [0, 1, 2] as const) {
    const lower = gridAxisCoordinate(grid, axis, range.lower[axis]);
    const upper = gridAxisCoordinate(grid, axis, range.upper[axis]);
    if (lower > boundsM.lower[axis] || upper < boundsM.upper[axis]) {
      return { ok: false, reason: "The native-detail budget cannot cover structural-support bounds." };
    }
  }
  return planned;
}

export function planBoundedStructuralSupportWindows(
  grid: GridInfo | null | undefined,
  boundsM: { lower: readonly [number, number, number]; upper: readonly [number, number, number] },
  maxVoxels: number,
): BoundedVolumeWindowOutcome {
  if (!grid || validGridCellCount(grid) == null
      || [0, 1, 2].some(axis => !Number.isFinite(gridAxisSpacing(grid, axis)) || gridAxisSpacing(grid, axis) <= 0)
      || ![...boundsM.lower, ...boundsM.upper].every(Number.isFinite)
      || boundsM.lower.some((value, axis) => value > boundsM.upper[axis])) {
    return { ok: false, reason: "The structural-support grid or bounds are invalid." };
  }
  const sourceShape = [grid.nr!, grid.nth!, grid.nz!];
  const validLower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const validUpper = grid.validIndexRange?.upper ?? sourceShape.map(count => count - 1);
  const lower = ([0, 1, 2] as const).map(axis => Math.max(validLower[axis],
    nearestGridIndex(grid, axis, boundsM.lower[axis], validLower[axis], validUpper[axis]) - 1)) as [number, number, number];
  const upper = ([0, 1, 2] as const).map(axis => Math.min(validUpper[axis],
    nearestGridIndex(grid, axis, boundsM.upper[axis], validLower[axis], validUpper[axis]) + 1)) as [number, number, number];
  const cells = lower.reduce((count, value, axis) => count * (upper[axis] - value + 1), 1);
  if (cells > MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS) {
    return { ok: false, reason: "Complete structural support requires a published overview; its raw scan is too large." };
  }
  return planBoundedVolumeWindows({ ...grid, validIndexRange: { lower, upper } }, maxVoxels);
}

/** Exact stride-1 valid-domain tiles. Intended for reductions/comparisons that
 * must inspect every model cell but never need the complete field resident. */
export function planExactValidDomainWindows(
  grid: GridInfo | null | undefined,
  rank: 2 | 3,
  localRegion?: { lower: readonly number[]; upper: readonly number[] },
  maxTileBytes = 16 * 1024 * 1024,
): ExactWindowOutcome {
  if (!grid || !finitePositiveInteger(grid.nr) || !finitePositiveInteger(grid.nth)
      || (rank === 3 && !finitePositiveInteger(grid.nz))) {
    return { ok: false, reason: `The frame does not declare a complete rank-${rank} grid.` };
  }
  const full = (rank === 3 ? [grid.nr, grid.nth, grid.nz!] : [grid.nr, grid.nth]) as number[];
  const validLower = (grid.validIndexRange?.lower ?? [0, 0, 0]).slice(0, rank).map(Math.trunc);
  const validUpper = (grid.validIndexRange?.upper
    ?? full.map((count) => count - 1)).slice(0, rank).map(Math.trunc);
  const validShape = validUpper.map((upper, axis) => upper - validLower[axis] + 1);
  if (validShape.some((count, axis) => !finitePositiveInteger(count)
      || validLower[axis] < 0 || validUpper[axis] >= full[axis])) {
    return { ok: false, reason: "The frame's valid-domain indices are inconsistent with its grid." };
  }
  const localLower = localRegion?.lower.slice(0, rank).map(Math.trunc)
    ?? Array(rank).fill(0) as number[];
  const localUpper = localRegion?.upper.slice(0, rank).map(Math.trunc)
    ?? validShape.map((count) => count - 1);
  if (localLower.length !== rank || localUpper.length !== rank
      || localLower.some((lower, axis) => lower < 0 || localUpper[axis] < lower
        || localUpper[axis] >= validShape[axis])) {
    return { ok: false, reason: "The exact comparison region lies outside the valid domain." };
  }
  const origin = localLower.map((lower, axis) => validLower[axis] + lower);
  const extent = localUpper.map((upper, axis) => upper - localLower[axis] + 1);
  const terrainGeometryBytesPerCell = rank === 3 && grid.terrainFollowing
    ? 4 * Float64Array.BYTES_PER_ELEMENT
    : 0;
  const outputBytesPerCell = FLOAT32_BYTES + terrainGeometryBytesPerCell;
  const byteCeiling = Math.max(outputBytesPerCell, Math.min(
    Math.floor(maxTileBytes),
    MAX_NATIVE_FIELD_WINDOW_BYTES,
    MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
  ));
  const tile = [...extent];
  while (nativeFieldWindowBodyBytes(product(tile), terrainGeometryBytesPerCell) > byteCeiling) {
    const axis = tile.indexOf(Math.max(...tile));
    if (tile[axis] <= 1) {
      return { ok: false, reason: "One exact field tile exceeds the native read ceiling." };
    }
    tile[axis] = Math.max(1, Math.ceil(tile[axis] / 2));
  }
  const windows: FieldLoadWindow[] = [];
  const push = (tileOrigin: number[]): void => {
    const tileExtent = tileOrigin.map((start, axis) => Math.min(
      tile[axis],
      origin[axis] + extent[axis] - start,
    ));
    const outputBytes = nativeFieldWindowBodyBytes(
      product(tileExtent), terrainGeometryBytesPerCell,
    );
    const sourceBytes = product(tileExtent) * FLOAT32_BYTES;
    windows.push({
      aggregation: "cell_mean",
      origin: tileOrigin,
      extent: tileExtent,
      stride: Array(rank).fill(1),
      maxOutputBytes: outputBytes,
      maxSourceBytes: rank === 3 && grid.terrainFollowing
        ? MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
        : sourceBytes,
    });
  };
  if (rank === 2) {
    for (let i = origin[0]; i < origin[0] + extent[0]; i += tile[0]) {
      for (let j = origin[1]; j < origin[1] + extent[1]; j += tile[1]) push([i, j]);
    }
  } else {
    for (let i = origin[0]; i < origin[0] + extent[0]; i += tile[0]) {
      for (let j = origin[1]; j < origin[1] + extent[1]; j += tile[1]) {
        for (let k = origin[2]; k < origin[2] + extent[2]; k += tile[2]) push([i, j, k]);
      }
    }
  }
  return { ok: true, plan: { windows, shape: extent, cells: product(extent) } };
}

/** Exact rank-3 line through the valid domain. `fixedLocalIndices` are local
 * valid-domain indices; the entry on `axis` is ignored. */
export function planExactLineWindows(
  grid: GridInfo | null | undefined,
  axis: 0 | 1 | 2,
  fixedLocalIndices: readonly [number, number, number],
  maxTileBytes = 16 * 1024 * 1024,
): ExactWindowOutcome {
  const full = planExactValidDomainWindows(grid, 3, undefined, maxTileBytes);
  if (!full.ok) return full;
  const lower: [number, number, number] = [0, 0, 0];
  const upper = full.plan.shape.map((count) => count - 1) as [number, number, number];
  for (const fixedAxis of [0, 1, 2] as const) {
    if (fixedAxis === axis) continue;
    lower[fixedAxis] = Math.trunc(fixedLocalIndices[fixedAxis]);
    upper[fixedAxis] = lower[fixedAxis];
  }
  return planExactValidDomainWindows(grid, 3, { lower, upper }, maxTileBytes);
}

/**
 * Plan physically balanced cell-mean windows under both renderer and native-I/O
 * ceilings. Large sources are tiled on block boundaries, so concatenating the
 * responses is identical to one full-domain aggregation request.
 */
export function planBoundedVolumeWindows(
  grid: GridInfo | null | undefined,
  maxVoxels: number,
): BoundedVolumeWindowOutcome {
  if (!grid || !finitePositiveInteger(grid.nr)
      || !finitePositiveInteger(grid.nth) || !finitePositiveInteger(grid.nz)) {
    return { ok: false, reason: "The frame does not declare a complete 3D grid." };
  }
  const fullShape: [number, number, number] = [grid.nr, grid.nth, grid.nz];
  const range = grid.validIndexRange;
  const origin: [number, number, number] = range
    ? range.lower.map(Math.trunc) as [number, number, number]
    : [0, 0, 0];
  const upper: [number, number, number] = range
    ? range.upper.map(Math.trunc) as [number, number, number]
    : fullShape.map((count) => count - 1) as [number, number, number];
  const extent = upper.map((value, axis) => value - origin[axis] + 1) as
    [number, number, number];
  if (extent.some((count, axis) =>
    !finitePositiveInteger(count) || origin[axis] < 0 || upper[axis] >= fullShape[axis]
  )) {
    return { ok: false, reason: "The frame's valid-domain indices are inconsistent with its grid." };
  }

  // A terrain block returns float32 data plus float64 x/y/z centroids and its
  // exact float64 physical volume. Geometry is part of the renderer budget;
  // counting only the field body made even small terrain slices request four
  // bytes of budget for a 36-byte response and fail in the native bridge.
  const terrainGeometryBytesPerCell = grid.terrainFollowing
    ? 4 * Float64Array.BYTES_PER_ELEMENT
    : 0;
  const outputBytesPerCell = grid.terrainFollowing
    ? Float32Array.BYTES_PER_ELEMENT + 4 * Float64Array.BYTES_PER_ELEMENT
    : Float32Array.BYTES_PER_ELEMENT;
  // `maxVoxels` governs the complete assembled renderer lattice. The native
  // envelope is a per-request limit and is enforced by the tiler below; using
  // it here silently capped a safely tiled 160 MiB Beast volume at one 64 MiB
  // IPC response and forced an artificial cropped ROI.
  const outputCellCeiling = Math.max(1, Math.floor(maxVoxels));
  if (!Number.isFinite(outputCellCeiling) || outputCellCeiling < 1) {
    return { ok: false, reason: "The renderer supplied an invalid voxel budget." };
  }

  const spacing: [number, number, number] = [
    grid.drM ?? 1,
    grid.dyM ?? grid.drM ?? 1,
    grid.dzM ?? 1,
  ];
  const stride: [number, number, number] = [1, 1, 1];
  const sampledShape = () => extent.map(
    (count, axis) => outputCount(count, stride[axis]),
  ) as [number, number, number];
  let sampled = sampledShape();
  while (product(sampled) > outputCellCeiling) {
    let axis = 0;
    for (let candidate = 1; candidate < 3; candidate += 1) {
      const currentResolution = spacing[axis] * stride[axis];
      const candidateResolution = spacing[candidate] * stride[candidate];
      if (candidateResolution < currentResolution
          || (candidateResolution === currentResolution
            && sampled[candidate] > sampled[axis])) {
        axis = candidate;
      }
    }
    if (stride[axis] >= extent[axis]) {
      const available = [0, 1, 2].filter((candidate) => stride[candidate] < extent[candidate]);
      if (available.length === 0) break;
      axis = available[0];
    }
    stride[axis] += 1;
    sampled = sampledShape();
  }
  if (product(sampled) > outputCellCeiling) {
    return { ok: false, reason: "The grid cannot be reduced beneath the renderer's hard voxel ceiling." };
  }

  // Tile in OUTPUT-block space. Source extents stay aligned to aggregation
  // blocks, making separate reads numerically identical to one larger read.
  const tileBlocks = [...sampled] as [number, number, number];
  const sourceBytesForTile = () => product(tileBlocks.map((blocks, axis) =>
    Math.min(extent[axis], blocks * stride[axis])
  )) * FLOAT32_BYTES;
  // A built-in/expression/chunk-aware provider must first receive this tile at
  // stride 1. On terrain that raw response carries 32 geometry bytes per cell,
  // so keep every planned tile admissible for both native and derived fields.
  const sourceTileCeiling = grid.terrainFollowing
    ? Math.floor(MAX_NATIVE_FIELD_WINDOW_BYTES * FLOAT32_BYTES / outputBytesPerCell)
    : Math.min(MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES, 16 * 1024 * 1024);
  while (sourceBytesForTile() > sourceTileCeiling) {
    let axis = tileBlocks.indexOf(Math.max(...tileBlocks));
    if (tileBlocks[axis] <= 1) {
      const available = [0, 1, 2].filter((candidate) => tileBlocks[candidate] > 1);
      if (available.length === 0) {
        return { ok: false, reason: "One aggregation block exceeds the native source-read ceiling." };
      }
      axis = available[0];
    }
    tileBlocks[axis] = Math.max(1, Math.ceil(tileBlocks[axis] / 2));
  }

  const windows: FieldLoadWindow[] = [];
  for (let bx = 0; bx < sampled[0]; bx += tileBlocks[0]) {
    const blocksX = Math.min(tileBlocks[0], sampled[0] - bx);
    for (let by = 0; by < sampled[1]; by += tileBlocks[1]) {
      const blocksY = Math.min(tileBlocks[1], sampled[1] - by);
      for (let bz = 0; bz < sampled[2]; bz += tileBlocks[2]) {
        const blocksZ = Math.min(tileBlocks[2], sampled[2] - bz);
        const blockOrigin: [number, number, number] = [bx, by, bz];
        const blockCounts: [number, number, number] = [blocksX, blocksY, blocksZ];
        const sourceOrigin = blockOrigin.map(
          (block, axis) => origin[axis] + block * stride[axis],
        ) as [number, number, number];
        const sourceExtent = blockCounts.map((blocks, axis) => Math.min(
          blocks * stride[axis],
          origin[axis] + extent[axis] - sourceOrigin[axis],
        )) as [number, number, number];
        const outputBytes = nativeFieldWindowBodyBytes(
          product(blockCounts), terrainGeometryBytesPerCell,
        );
        const sourceBytes = product(sourceExtent) * FLOAT32_BYTES;
        windows.push({
          aggregation: "cell_mean",
          origin: sourceOrigin,
          extent: sourceExtent,
          stride: [...stride],
          maxOutputBytes: Math.min(MAX_NATIVE_FIELD_WINDOW_BYTES, outputBytes),
          // Terrain admission includes float64 center/interface geometry and
          // aggregation workspaces in addition to field bytes. The native
          // reader computes and enforces the exact amount; grant the protocol
          // ceiling while the tile planner keeps the source field bounded.
          maxSourceBytes: grid.terrainFollowing
            ? MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
            : Math.min(MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES, sourceBytes),
        });
      }
    }
  }

  return {
    ok: true,
    plan: {
      windows,
      sourceShape: extent,
      sampledShape: sampled,
      stride,
      sourceCells: product(extent),
      sampledCells: product(sampled),
    },
  };
}

/** Plan a one-cell-thick source slice, with bounded in-plane aggregation only when necessary. */
export function planBoundedSliceWindows(
  grid: GridInfo | null | undefined,
  axis: SliceAxis,
  validIndex: number,
  maxCells = PLOTLY_HEATMAP_CELL_CAP,
  viewport: PlaneViewportBounds | null = null,
): BoundedVolumeWindowOutcome {
  if (!grid) return { ok: false, reason: "The frame does not declare a grid." };
  const counts = [grid.nr, grid.nth, grid.nz];
  const sourceCount = counts[axis];
  if (!finitePositiveInteger(sourceCount)) {
    return { ok: false, reason: "The selected slice axis has no declared source length." };
  }
  const validLower = Math.trunc(grid.validIndexRange?.lower[axis] ?? 0);
  const validUpper = Math.trunc(grid.validIndexRange?.upper[axis] ?? sourceCount - 1);
  const validCount = validUpper - validLower + 1;
  if (!Number.isSafeInteger(validIndex) || validIndex < 0 || validIndex >= validCount) {
    return { ok: false, reason: "The selected slice index lies outside the valid domain." };
  }
  const lower: [number, number, number] = grid.validIndexRange
    ? [...grid.validIndexRange.lower] as [number, number, number]
    : [0, 0, 0];
  const upper: [number, number, number] = grid.validIndexRange
    ? [...grid.validIndexRange.upper] as [number, number, number]
    : [grid.nr! - 1, grid.nth! - 1, grid.nz! - 1];
  lower[axis] = validLower + validIndex;
  upper[axis] = lower[axis];
  const fullSourceShape = upper.map((value, currentAxis) =>
    currentAxis === axis ? 1 : value - lower[currentAxis] + 1,
  ) as [number, number, number];
  applyPlaneViewport(grid, axis, lower, upper, viewport);
  const planned = planBoundedVolumeWindows({
    ...grid,
    validIndexRange: { lower, upper },
  }, maxCells);
  return planned.ok ? {
    ok: true,
    plan: {
      ...planned.plan,
      fullSourceShape,
      ...(viewport ? { viewportBounds: viewport } : {}),
    },
  } : planned;
}

/** Rank-2 companion to `planBoundedSliceWindows`. It reuses the same physical
 * x/y balancing and tile budgets on a one-level planning grid, then removes
 * the synthetic z axis from each transport request. */
export function planBoundedPlaneWindows(
  grid: GridInfo | null | undefined,
  maxCells = PLOTLY_HEATMAP_CELL_CAP,
  viewport: PlaneViewportBounds | null = null,
): BoundedVolumeWindowOutcome {
  if (!grid) return { ok: false, reason: "The frame does not declare a grid." };
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? [(grid.nr ?? 1) - 1, (grid.nth ?? 1) - 1, 0];
  const viewportLower: [number, number, number] = [lower[0], lower[1], 0];
  const viewportUpper: [number, number, number] = [upper[0], upper[1], 0];
  const fullSourceShape: [number, number, number] = [
    viewportUpper[0] - viewportLower[0] + 1,
    viewportUpper[1] - viewportLower[1] + 1,
    1,
  ];
  applyPlaneViewport(grid, 2, viewportLower, viewportUpper, viewport);
  const planned = planBoundedVolumeWindows({
    ...grid,
    nz: 1,
    terrainFollowing: false,
    validIndexRange: {
      lower: viewportLower,
      upper: viewportUpper,
    },
  }, maxCells);
  if (!planned.ok) return planned;
  return {
    ok: true,
    plan: {
      ...planned.plan,
      fullSourceShape,
      ...(viewport ? { viewportBounds: viewport } : {}),
      windows: planned.plan.windows.map((window) => ({
        ...window,
        origin: window.origin.slice(0, 2),
        extent: window.extent.slice(0, 2),
        stride: window.stride.slice(0, 2),
      })),
    },
  };
}

function applyPlaneViewport(
  grid: GridInfo,
  sliceAxis: SliceAxis,
  lower: [number, number, number],
  upper: [number, number, number],
  viewport: PlaneViewportBounds | null,
): void {
  if (!viewport) return;
  const { rowAxis, colAxis } = sliceDisplayAxes(sliceAxis);
  const colRange = viewportIndexRange(grid, colAxis, viewport.xMin, viewport.xMax);
  const rowRange = viewportIndexRange(grid, rowAxis, viewport.yMin, viewport.yMax);
  if (colRange) {
    lower[colAxis] = Math.max(lower[colAxis], colRange[0]);
    upper[colAxis] = Math.min(upper[colAxis], colRange[1]);
  }
  if (rowRange) {
    lower[rowAxis] = Math.max(lower[rowAxis], rowRange[0]);
    upper[rowAxis] = Math.min(upper[rowAxis], rowRange[1]);
  }
}

function viewportIndexRange(
  grid: GridInfo,
  axis: SliceAxis,
  first: number,
  second: number,
): [number, number] | null {
  if (!Number.isFinite(first) || !Number.isFinite(second)) return null;
  const count = axis === 0 ? grid.nr : axis === 1 ? grid.nth : grid.nz;
  if (!finitePositiveInteger(count)) return null;
  const validLower = Math.max(0, Math.trunc(grid.validIndexRange?.lower[axis] ?? 0));
  const validUpper = Math.min(count - 1, Math.trunc(grid.validIndexRange?.upper[axis] ?? count - 1));
  if (validUpper < validLower) return null;
  const exact = grid.scalarCoordinatesM?.[axis];
  const spacing = gridAxisSpacing(grid, axis);
  const exactPhysical = exact?.length === count
    && exact.every(Number.isFinite)
    && exact.every((value, index) => index === 0 || value > exact[index - 1]);
  const physical = !(axis === 2 && grid.terrainFollowing)
    && (exactPhysical || spacing > 0);
  const uniformOrigin = physical && !exactPhysical
    ? gridAxisOrigin(grid, axis, count, spacing)
    : 0;
  const coordinate = (index: number) => exactPhysical
    ? exact![index] / 1000
    : physical
      ? (uniformOrigin + index * spacing) / 1000
      : index;
  const minimum = Math.min(first, second);
  const maximum = Math.max(first, second);
  let lo = validLower;
  let hi = validUpper + 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (coordinate(mid) < minimum) lo = mid + 1;
    else hi = mid;
  }
  const start = Math.max(validLower, Math.min(validUpper, lo - 1));
  lo = validLower;
  hi = validUpper + 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (coordinate(mid) <= maximum) lo = mid + 1;
    else hi = mid;
  }
  const end = Math.max(start, Math.max(validLower, Math.min(validUpper, lo)));
  return [start, end];
}
