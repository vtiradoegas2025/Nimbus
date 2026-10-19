import type { Npy3D } from "@/lib/platform/npy";
import { scalarVolumeStats, type ScalarVolumeStats } from "./scalarVolumeStats";
import type { FieldBrickHierarchy, Triple } from "@/lib/plot/fieldBrickHierarchy";
import type { VolumeOpacityClass } from "@/lib/plot/volumeTransferPolicy";
import {
  readField,
  type FieldLoadWindow,
  type GridInfo,
  type StepInfo,
} from "@/lib/run/runOutputs";
import { MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES } from "@/lib/dataset/nativeFieldWindow";
import { assembleWindowedField } from "@/lib/run/windowedField";

export const TRANSIENT_VOLUME_BRICK_SHAPE: Triple = [128, 128, 40];
export const TRANSIENT_VOLUME_PROXY_STRIDE: Triple = [8, 8, 4];
const SAMPLE_CAP = 200_000;
const ZERO_SHA256 = "0".repeat(64);

export interface TransientVolumeHierarchyResult {
  hierarchy: FieldBrickHierarchy;
  stats: ScalarVolumeStats;
}

type WindowReader = (window: FieldLoadWindow, lane: string) => Promise<Npy3D>;

interface WindowGroup {
  brickOrigin: Triple;
  brickCounts: Triple;
  window: FieldLoadWindow;
}

function checkedProduct(values: readonly number[]): number {
  const value = values.reduce((product, item) => product * item, 1);
  if (!Number.isSafeInteger(value) || value < 1) throw new Error("Transient brick extent overflow");
  return value;
}

function validDomain(grid: GridInfo): { lower: Triple; shape: Triple } {
  if (![grid.nr, grid.nth, grid.nz].every((value) =>
    Number.isSafeInteger(value) && (value ?? 0) > 0)) {
    throw new Error("Transient volume bricks require a complete rank-3 grid");
  }
  const full = [grid.nr!, grid.nth!, grid.nz!] as Triple;
  const lower = (grid.validIndexRange?.lower ?? [0, 0, 0]).map(Math.trunc) as Triple;
  const upper = (grid.validIndexRange?.upper
    ?? full.map((count) => count - 1)).map(Math.trunc) as Triple;
  const shape = upper.map((value, axis) => value - lower[axis] + 1) as Triple;
  if (shape.some((count, axis) => count < 1 || lower[axis] < 0 || upper[axis] >= full[axis])) {
    throw new Error("Transient volume brick domain is inconsistent with the grid");
  }
  return { lower, shape };
}

function groupedWindows(
  grid: GridInfo,
  aggregation: FieldLoadWindow["aggregation"],
  stride: Triple = TRANSIENT_VOLUME_BRICK_SHAPE,
): WindowGroup[] {
  const { lower, shape } = validDomain(grid);
  const brickGrid = shape.map((count, axis) =>
    Math.ceil(count / TRANSIENT_VOLUME_BRICK_SHAPE[axis])) as Triple;
  const maximumCells = Math.floor(
    MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES / Float32Array.BYTES_PER_ELEMENT,
  );
  const group = [1, 1, 1] as Triple;
  for (const axis of [0, 1, 2] as const) {
    const otherCells = group.reduce((product, count, otherAxis) =>
      product * (otherAxis === axis ? 1 : count * TRANSIENT_VOLUME_BRICK_SHAPE[otherAxis]), 1);
    group[axis] = Math.max(1, Math.min(
      brickGrid[axis],
      Math.floor(maximumCells / (otherCells * TRANSIENT_VOLUME_BRICK_SHAPE[axis])),
    ));
  }
  const groups: WindowGroup[] = [];
  for (let bx = 0; bx < brickGrid[0]; bx += group[0]) {
    for (let by = 0; by < brickGrid[1]; by += group[1]) {
      for (let bz = 0; bz < brickGrid[2]; bz += group[2]) {
        const brickOrigin = [bx, by, bz] as Triple;
        const brickCounts = group.map((count, axis) =>
          Math.min(count, brickGrid[axis] - brickOrigin[axis])) as Triple;
        const localOrigin = brickOrigin.map((value, axis) =>
          value * TRANSIENT_VOLUME_BRICK_SHAPE[axis]) as Triple;
        const extent = brickCounts.map((count, axis) => Math.min(
          count * TRANSIENT_VOLUME_BRICK_SHAPE[axis],
          shape[axis] - localOrigin[axis],
        )) as Triple;
        const outputShape = extent.map((count, axis) =>
          Math.floor((count - 1) / stride[axis]) + 1) as Triple;
        const sourceBytes = checkedProduct(extent) * Float32Array.BYTES_PER_ELEMENT;
        groups.push({
          brickOrigin,
          brickCounts,
          window: {
            aggregation,
            origin: localOrigin.map((value, axis) => value + lower[axis]),
            extent,
            stride: [...stride],
            maxOutputBytes: checkedProduct(outputShape) * Float32Array.BYTES_PER_ELEMENT,
            maxSourceBytes: sourceBytes,
          },
        });
      }
    }
  }
  return groups;
}

function axisCenter(grid: GridInfo, axis: number, index: number): number {
  const coordinates = grid.scalarCoordinatesM?.[axis];
  if (coordinates?.length && Number.isFinite(coordinates[index])) return coordinates[index];
  const spacing = axis === 0 ? grid.drM : axis === 1 ? grid.dyM ?? grid.drM : grid.dzM;
  if (!(spacing != null && spacing > 0)) throw new Error("Transient bricks require physical spacing");
  const origin = grid.scalarOriginM?.[axis]
    ?? (grid.physicalBoundsM?.lower[axis] != null
      ? grid.physicalBoundsM.lower[axis] + 0.5 * spacing
      : axis === 2 ? 0.5 * spacing : 0);
  return origin + index * spacing;
}

function axisEdge(grid: GridInfo, axis: number, index: number, upper: boolean): number {
  const count = axis === 0 ? grid.nr! : axis === 1 ? grid.nth! : grid.nz!;
  const coordinates = grid.scalarCoordinatesM?.[axis];
  if (coordinates?.length === count && count > 1) {
    if (upper) {
      return index + 1 < count
        ? 0.5 * (coordinates[index] + coordinates[index + 1])
        : coordinates[index] + 0.5 * (coordinates[index] - coordinates[index - 1]);
    }
    return index > 0
      ? 0.5 * (coordinates[index - 1] + coordinates[index])
      : coordinates[index] - 0.5 * (coordinates[1] - coordinates[0]);
  }
  const spacing = axis === 0 ? grid.drM! : axis === 1 ? grid.dyM ?? grid.drM! : grid.dzM!;
  return axisCenter(grid, axis, index) + (upper ? 0.5 : -0.5) * spacing;
}

async function sampledStats(
  grid: GridInfo,
  groups: readonly WindowGroup[],
  read: WindowReader,
): Promise<TransientVolumeHierarchyResult["stats"]> {
  const { shape } = validDomain(grid);
  const stride = Math.max(1, Math.ceil(Math.cbrt(checkedProduct(shape) / SAMPLE_CAP)));
  const samples: number[] = [];
  for (let index = 0; index < groups.length; index += 1) {
    const source = groups[index].window;
    const outputShape = source.extent.map((count) => Math.floor((count - 1) / stride) + 1);
    const field = await read({
      ...source,
      aggregation: "point_sample",
      stride: [stride, stride, stride],
      maxOutputBytes: checkedProduct(outputShape) * Float32Array.BYTES_PER_ELEMENT,
    }, `stats:${index}`);
    for (const value of field.data) if (Number.isFinite(value)) samples.push(value);
  }
  if (samples.length === 0) throw new Error("Transient volume statistics found no finite samples");
  return scalarVolumeStats(samples);
}

export async function sampleVolumeFieldStats(input: {
  runDir: string;
  step: StepInfo;
  fieldName: string;
  signal?: AbortSignal;
  reader?: WindowReader;
}): Promise<TransientVolumeHierarchyResult["stats"]> {
  const { runDir, step, fieldName, signal } = input;
  const grid = step.grid;
  if (!grid || grid.terrainFollowing) {
    throw new Error("Sampled volume statistics require a separable physical grid");
  }
  const read = input.reader ?? ((window) => readField(
    runDir,
    step.index,
    fieldName,
    step.publicationGeneration,
    {
      signal,
      window,
      cacheRetention: "transient",
    },
  ));
  return sampledStats(grid, groupedWindows(grid, "block_max"), read);
}

export async function buildTransientVolumeHierarchy(input: {
  runDir: string;
  step: StepInfo;
  fieldName: string;
  opacityClass: VolumeOpacityClass;
  signal?: AbortSignal;
  reader?: WindowReader;
}): Promise<TransientVolumeHierarchyResult> {
  const { runDir, step, fieldName, opacityClass, signal } = input;
  const grid = step.grid;
  if (!grid || grid.terrainFollowing) {
    throw new Error("Transient volume bricks require a separable physical grid");
  }
  // Indexing never transfers these buffers. Shared reads prevent separate
  // panes from superseding each other's same-field metadata scans.
  const read = input.reader ?? ((window) => readField(
    runDir,
    step.index,
    fieldName,
    step.publicationGeneration,
    {
      signal,
      window,
      cacheRetention: "transient",
    },
  ));
  const signed = opacityClass === "signed-perturbation";
  const groups = groupedWindows(
    grid,
    signed ? "magnitude_extreme" : "block_max",
    TRANSIENT_VOLUME_PROXY_STRIDE,
  );
  const summaries: Array<{ group: WindowGroup; field: Npy3D }> = [];
  for (let index = 0; index < groups.length; index += 1) {
    if (signal?.aborted) throw new DOMException("Transient volume indexing cancelled", "AbortError");
    summaries.push({ group: groups[index], field: await read(groups[index].window, `range:${index}`) });
  }
  const stats = await sampledStats(grid, groups, read);
  const { lower, shape } = validDomain(grid);
  const assembled = assembleWindowedField(
    summaries.map(({ field }) => field),
    groups.map(({ window }) => window),
  );
  const previewField: Npy3D = {
    shape: assembled.shape as Triple,
    data: assembled.data,
    ...(assembled.windowGeometry ? { windowGeometry: assembled.windowGeometry } : {}),
  };
  const brickGrid = shape.map((count, axis) =>
    Math.ceil(count / TRANSIENT_VOLUME_BRICK_SHAPE[axis])) as Triple;
  const bricks = [];
  for (let x = 0; x < brickGrid[0]; x += 1) {
    for (let y = 0; y < brickGrid[1]; y += 1) {
      for (let z = 0; z < brickGrid[2]; z += 1) {
          const brickIndex = [x, y, z] as Triple;
          const origin = brickIndex.map((value, axis) =>
            value * TRANSIENT_VOLUME_BRICK_SHAPE[axis]) as Triple;
          const extent = origin.map((value, axis) => Math.min(
            TRANSIENT_VOLUME_BRICK_SHAPE[axis], shape[axis] - value,
          )) as Triple;
          const proxyLower = origin.map((value, axis) =>
            Math.floor(value / TRANSIENT_VOLUME_PROXY_STRIDE[axis])) as Triple;
          const proxyUpper = origin.map((value, axis) => Math.min(
            previewField.shape[axis],
            Math.ceil((value + extent[axis]) / TRANSIENT_VOLUME_PROXY_STRIDE[axis]),
          )) as Triple;
          let summary = signed ? 0 : Number.NEGATIVE_INFINITY;
          for (let px = proxyLower[0]; px < proxyUpper[0]; px += 1) {
            for (let py = proxyLower[1]; py < proxyUpper[1]; py += 1) {
              const base = (px * previewField.shape[1] + py) * previewField.shape[2];
              for (let pz = proxyLower[2]; pz < proxyUpper[2]; pz += 1) {
                const value = previewField.data[base + pz];
                if (signed ? Math.abs(value) > Math.abs(summary) : value > summary) summary = value;
              }
            }
          }
          const finite = Number.isFinite(summary);
          const magnitude = finite ? Math.abs(summary) : Number.MAX_VALUE;
          const minimum = signed ? -magnitude : finite ? summary : -Number.MAX_VALUE;
          const maximum = signed ? magnitude : finite ? summary : Number.MAX_VALUE;
          const storageLower = origin.map((value, axis) => value + lower[axis]) as Triple;
          const storageUpper = storageLower.map((value, axis) => value + extent[axis] - 1) as Triple;
          bricks.push({
            id: `l0:${brickIndex.join(":")}`,
            index: brickIndex,
            origin,
            extent,
            scalarRange: {
              minimum,
              maximum,
              finiteCount: finite ? checkedProduct(extent) : 1,
              nonFiniteCount: finite ? 0 : checkedProduct(extent) - 1,
            },
            physicalBoundsM: {
              lower: storageLower.map((value, axis) => axisEdge(grid, axis, value, false)) as Triple,
              upper: storageUpper.map((value, axis) => axisEdge(grid, axis, value, true)) as Triple,
            },
            contentSha256: ZERO_SHA256,
          });
      }
    }
  }
  return {
    stats,
    hierarchy: {
      schema: "nimbus.field-brick-hierarchy",
      schemaVersion: 1,
      field: fieldName,
      publicationGeneration: step.publicationGeneration ?? `legacy-step-${step.index}`,
      sourceShape: shape,
      sourceIndexOffset: lower,
      sourceDtype: "float32",
      sampleLocation: "cell_center",
      extremaSemantics: "stored_values",
      extremaAbsolutePadding: 0,
      cullingSafe: true,
      levels: [{
        level: 0,
        stride: [1, 1, 1],
        shape,
        brickShape: [...TRANSIENT_VOLUME_BRICK_SHAPE],
        source: { kind: "primary_field" },
        bricks,
      }],
    },
  };
}
