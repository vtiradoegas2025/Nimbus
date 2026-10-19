import type { FieldLoadWindow } from "@/lib/run/runOutputs";

export const FIELD_BRICK_HIERARCHY_SCHEMA = "nimbus.field-brick-hierarchy" as const;
export const FIELD_BRICK_HIERARCHY_VERSION = 3 as const;

export type Triple = [number, number, number];

export interface BrickBoundaryContacts {
  lower: [boolean, boolean, boolean];
  upper: [boolean, boolean, boolean];
}

export interface VectorBrickExpansion {
  lower: Triple;
  upper: Triple;
}

export interface FieldBrickScalarRange {
  minimum: number | null;
  maximum: number | null;
  finiteCount: number;
  nonFiniteCount: number;
}

export interface FieldBrickDescriptor {
  id: string;
  index: Triple;
  origin: Triple;
  extent: Triple;
  scalarRange: FieldBrickScalarRange;
  physicalBoundsM: { lower: Triple; upper: Triple };
  contentSha256: string;
}

export interface FieldBrickLevel {
  level: number;
  stride: Triple;
  shape: Triple;
  brickShape: Triple;
  source:
    | { kind: "primary_field" }
    | {
        kind: "npy_lod";
        path: string;
        contentSha256: string;
        aggregation: "recursive_cell_mean_2x2x2";
        geometry?: {
          kind: "physical_volume_centroids";
          xM: { path: string; contentSha256: string };
          yM: { path: string; contentSha256: string };
          zM: { path: string; contentSha256: string };
          volumeM3: { path: string; contentSha256: string };
        };
      };
  bricks: FieldBrickDescriptor[];
}

export interface FieldBrickHierarchy {
  schema: typeof FIELD_BRICK_HIERARCHY_SCHEMA;
  schemaVersion: 1 | 2 | typeof FIELD_BRICK_HIERARCHY_VERSION;
  field: string;
  publicationGeneration: string;
  sourceShape: Triple;
  sourceIndexOffset: Triple;
  sourceDtype: "float32";
  sampleLocation: "cell_center";
  extremaSemantics: "stored_values" | "decoded_value_bounds" | "pre_storage_source";
  extremaAbsolutePadding: number;
  cullingSafe: boolean;
  levels: FieldBrickLevel[];
}

export interface ContourBrickPlanItem {
  brickId: string;
  brickIndex: Triple;
  readWindow: FieldLoadWindow;
  ownedCellOrigin: Triple;
  ownedCellExtent: Triple;
  scalarRange: FieldBrickScalarRange;
}

export interface ContourBrickPlan {
  field: string;
  publicationGeneration: string;
  level: number;
  threshold: number;
  items: ContourBrickPlanItem[];
  culledBrickCount: number;
  sourceBytes: number;
}

export interface VolumeBrickPlanItem {
  brickId: string;
  brickIndex: Triple;
  window: FieldLoadWindow;
  scalarRange: FieldBrickScalarRange;
  priority: number;
  preparedScalarBytes: number;
  reconstruction?: { shape: Triple; origin: Triple; spacing: Triple };
  ownedOffset?: Triple;
  ownedShape?: Triple;
  boundaryOnly?: boolean;
}

export interface VolumeBrickPlan {
  field: string;
  publicationGeneration: string;
  level: number;
  items: VolumeBrickPlanItem[];
  transparentBrickCount: number;
  deferredBrickCount: number;
  residentBytes: number;
  excludedBrickCount: number;
  haloSamples?: number;
  boundaryOnlyBrickCount?: number;
}

export interface VectorBrickNeighborhoodPlan {
  publicationGeneration: string;
  level: number;
  componentFields: readonly [string, string, string];
  items: Array<{
    brickId: string;
    brickIndex: Triple;
    window: FieldLoadWindow;
  }>;
  preparedCells: number;
  totalBrickCount: number;
  focusBrick: Triple;
  lowerBrick: Triple;
  upperBrick: Triple;
  brickCounts: Triple;
  expansion: VectorBrickExpansion;
}

export type BrickHierarchyValidation =
  | { ok: true; hierarchy: FieldBrickHierarchy }
  | { ok: false; reason: string };

const SHA256 = /^[0-9a-f]{64}$/;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;

function triple(value: unknown, label: string, allowZero: boolean): Triple {
  if (!Array.isArray(value) || value.length !== 3) throw new Error(`${label} must be a triple`);
  const result = value.map(Number) as Triple;
  if (!result.every((item) => Number.isSafeInteger(item) && (allowZero ? item >= 0 : item > 0))) {
    throw new Error(`${label} contains invalid extents`);
  }
  return result;
}

function finiteTriple(value: unknown, label: string): Triple {
  if (!Array.isArray(value) || value.length !== 3) throw new Error(`${label} must be a triple`);
  const result = value.map(Number) as Triple;
  if (!result.every(Number.isFinite)) throw new Error(`${label} must be finite`);
  return result;
}

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product)) throw new Error(`${label} exceeds safe integer range`);
  }
  return product;
}

function expectedBrickCount(shape: Triple, brickShape: Triple): Triple {
  return shape.map((count, axis) => Math.ceil(count / brickShape[axis])) as Triple;
}

function canonicalBrickId(level: number, index: Triple): string {
  return `l${level}:${index[0]}:${index[1]}:${index[2]}`;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function parseScalarRange(value: unknown, cells: number): FieldBrickScalarRange {
  const source = record(value, "brick scalar range");
  const finiteCount = Number(source.finiteCount);
  const nonFiniteCount = Number(source.nonFiniteCount);
  if (!Number.isSafeInteger(finiteCount) || finiteCount < 0
      || !Number.isSafeInteger(nonFiniteCount) || nonFiniteCount < 0
      || finiteCount + nonFiniteCount !== cells) {
    throw new Error("brick scalar counts do not cover its samples");
  }
  const minimum = source.minimum == null ? null : Number(source.minimum);
  const maximum = source.maximum == null ? null : Number(source.maximum);
  if (finiteCount === 0) {
    if (minimum != null || maximum != null) throw new Error("empty finite range must use null bounds");
  } else if (minimum == null || maximum == null
      || !Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum < minimum) {
    throw new Error("finite brick range is invalid");
  }
  return { minimum, maximum, finiteCount, nonFiniteCount };
}

export function validateFieldBrickHierarchy(
  input: unknown,
  expected?: {
    field?: string;
    publicationGeneration?: string;
    sourceShape?: readonly number[];
    sourceIndexOffset?: readonly number[];
  },
): BrickHierarchyValidation {
  try {
    const source = record(input, "field brick hierarchy");
    if (source.schema !== FIELD_BRICK_HIERARCHY_SCHEMA
        || (source.schemaVersion !== 1 && source.schemaVersion !== 2
          && source.schemaVersion !== FIELD_BRICK_HIERARCHY_VERSION)) {
      throw new Error("field brick hierarchy schema/version mismatch");
    }
    const field = typeof source.field === "string" ? source.field : "";
    const publicationGeneration = typeof source.publicationGeneration === "string"
      ? source.publicationGeneration
      : "";
    if (!field || !publicationGeneration) throw new Error("field brick hierarchy identity is incomplete");
    if (expected?.field != null && field !== expected.field) throw new Error("field brick hierarchy field changed");
    if (expected?.publicationGeneration != null
        && publicationGeneration !== expected.publicationGeneration) {
      throw new Error("field brick hierarchy generation changed");
    }
    const sourceShape = triple(source.sourceShape, "sourceShape", false);
    const sourceIndexOffset = triple(source.sourceIndexOffset, "sourceIndexOffset", true);
    if (expected?.sourceShape != null
        && sourceShape.join("x") !== triple(expected.sourceShape, "expected sourceShape", false).join("x")) {
      throw new Error("field brick hierarchy source shape changed");
    }
    if (expected?.sourceIndexOffset != null
        && sourceIndexOffset.join("x")
          !== triple(expected.sourceIndexOffset, "expected sourceIndexOffset", true).join("x")) {
      throw new Error("field brick hierarchy source offset changed");
    }
    if (source.sourceDtype !== "float32" || source.sampleLocation !== "cell_center") {
      throw new Error("field brick hierarchy storage semantics are unsupported");
    }
    const extremaAbsolutePadding = Number(source.extremaAbsolutePadding);
    if ((source.extremaSemantics !== "stored_values"
        && source.extremaSemantics !== "decoded_value_bounds"
        && source.extremaSemantics !== "pre_storage_source")
        || typeof source.cullingSafe !== "boolean"
        || !Number.isFinite(extremaAbsolutePadding) || extremaAbsolutePadding < 0
        || (source.cullingSafe && source.extremaSemantics === "pre_storage_source")
        || (!source.cullingSafe && source.extremaSemantics !== "pre_storage_source")
        || (source.extremaSemantics === "stored_values" && extremaAbsolutePadding !== 0)
        || (source.extremaSemantics === "pre_storage_source" && extremaAbsolutePadding !== 0)) {
      throw new Error("field brick hierarchy extrema semantics are invalid");
    }
    if (!Array.isArray(source.levels) || source.levels.length === 0) {
      throw new Error("field brick hierarchy has no levels");
    }
    const rawLevels = source.levels;
    const schemaVersion = source.schemaVersion as 1 | 2 | typeof FIELD_BRICK_HIERARCHY_VERSION;
    const levels = rawLevels.map((rawLevel, levelPosition): FieldBrickLevel => {
      const levelSource = record(rawLevel, `level ${levelPosition}`);
      const level = Number(levelSource.level);
      if (level !== levelPosition) throw new Error("brick levels must be dense and ordered");
      const stride = triple(levelSource.stride, `level ${level} stride`, false);
      const shape = triple(levelSource.shape, `level ${level} shape`, false);
      const brickShape = triple(levelSource.brickShape, `level ${level} brickShape`, false);
      if (level === 0 && (stride.some((item) => item !== 1)
          || shape.join("x") !== sourceShape.join("x"))) {
        throw new Error("level zero must represent every native source sample");
      }
      const expectedShape = sourceShape.map((count, axis) => Math.ceil(count / stride[axis]));
      if (shape.join("x") !== expectedShape.join("x")) throw new Error("brick level shape disagrees with stride");
      if (schemaVersion >= 2 && level > 0) {
        const previous = record(rawLevels[level - 1], `level ${level - 1}`);
        const previousStride = triple(previous.stride, `level ${level - 1} stride`, false);
        if (stride.some((value, axis) => value !== previousStride[axis] * 2)) {
          throw new Error("durable brick LOD strides must double on every axis");
        }
      }
      const rawSource = levelSource.source;
      let levelSourceDescriptor: FieldBrickLevel["source"];
      if (schemaVersion === 1 && rawSource == null) {
        levelSourceDescriptor = { kind: "primary_field" };
      } else {
        const sourceDescriptor = record(rawSource, `level ${level} source`);
        if (level === 0) {
          if (sourceDescriptor.kind !== "primary_field"
              || Object.keys(sourceDescriptor).some((key) => key !== "kind")) {
            throw new Error("level zero must use the primary field payload");
          }
          levelSourceDescriptor = { kind: "primary_field" };
        } else {
          const path = typeof sourceDescriptor.path === "string" ? sourceDescriptor.path : "";
          const contentSha256 = typeof sourceDescriptor.contentSha256 === "string"
            ? sourceDescriptor.contentSha256
            : "";
          if (sourceDescriptor.kind !== "npy_lod"
              || path !== `${field}.lod${level}.npy`
              || path.includes("/") || path.includes("\\") || path.includes("..")
              || !SHA256.test(contentSha256)
              || sourceDescriptor.aggregation !== "recursive_cell_mean_2x2x2") {
            throw new Error("durable brick LOD source identity is invalid");
          }
          let geometry: Extract<FieldBrickLevel["source"], { kind: "npy_lod" }>["geometry"];
          if (schemaVersion === FIELD_BRICK_HIERARCHY_VERSION) {
            const rawGeometry = record(sourceDescriptor.geometry, `level ${level} terrain geometry`);
            if (rawGeometry.kind !== "physical_volume_centroids") {
              throw new Error("durable terrain LOD geometry kind is invalid");
            }
            const components = ["xM", "yM", "zM", "volumeM3"] as const;
            const prefixes = [
              "physical_x_center_m", "physical_y_center_m",
              "physical_z_center_m", "physical_volume_m3",
            ] as const;
            const parsed = Object.fromEntries(components.map((component, index) => {
              const descriptor = record(rawGeometry[component], `level ${level} ${component}`);
              const componentPath = typeof descriptor.path === "string" ? descriptor.path : "";
              const componentSha = typeof descriptor.contentSha256 === "string"
                ? descriptor.contentSha256
                : "";
              if (componentPath !== `${prefixes[index]}.lod${level}.npy`
                  || componentPath.includes("/") || componentPath.includes("\\")
                  || componentPath.includes("..") || !SHA256.test(componentSha)) {
                throw new Error("durable terrain LOD geometry identity is invalid");
              }
              return [component, { path: componentPath, contentSha256: componentSha }];
            })) as Record<typeof components[number], { path: string; contentSha256: string }>;
            geometry = {
              kind: "physical_volume_centroids",
              xM: parsed.xM,
              yM: parsed.yM,
              zM: parsed.zM,
              volumeM3: parsed.volumeM3,
            };
          } else if (sourceDescriptor.geometry != null) {
            throw new Error("legacy durable LOD cannot advertise unversioned terrain geometry");
          }
          levelSourceDescriptor = {
            kind: "npy_lod",
            path,
            contentSha256,
            aggregation: "recursive_cell_mean_2x2x2",
            ...(geometry ? { geometry } : {}),
          };
        }
      }
      const counts = expectedBrickCount(shape, brickShape);
      const expectedCount = checkedProduct(counts, "brick count");
      if (!Array.isArray(levelSource.bricks) || levelSource.bricks.length !== expectedCount) {
        throw new Error(`level ${level} brick directory is incomplete`);
      }
      const seen = new Set<string>();
      const bricks = levelSource.bricks.map((rawBrick): FieldBrickDescriptor => {
        const brickSource = record(rawBrick, `level ${level} brick`);
        const index = triple(brickSource.index, "brick index", true);
        if (index.some((item, axis) => item >= counts[axis])) throw new Error("brick index is outside its level");
        const id = canonicalBrickId(level, index);
        if (brickSource.id !== id || seen.has(id)) throw new Error("brick identity is noncanonical or duplicated");
        seen.add(id);
        const origin = triple(brickSource.origin, "brick origin", true);
        const expectedOrigin = index.map((item, axis) => item * brickShape[axis]) as Triple;
        if (origin.join("x") !== expectedOrigin.join("x")) throw new Error("brick origin is not topology-aligned");
        const extent = triple(brickSource.extent, "brick extent", false);
        const expectedExtent = origin.map((item, axis) => Math.min(brickShape[axis], shape[axis] - item));
        if (extent.join("x") !== expectedExtent.join("x")) throw new Error("brick extent leaves a gap or overlap");
        const cells = checkedProduct(extent, "brick sample count");
        const contentSha256 = typeof brickSource.contentSha256 === "string"
          ? brickSource.contentSha256
          : "";
        if (!SHA256.test(contentSha256)) throw new Error("brick checksum is invalid");
        const boundsSource = record(brickSource.physicalBoundsM, "brick physical bounds");
        const lower = finiteTriple(boundsSource.lower, "brick physical lower bound");
        const upper = finiteTriple(boundsSource.upper, "brick physical upper bound");
        if (upper.some((value, axis) => value <= lower[axis])) {
          throw new Error("brick physical bounds are not increasing");
        }
        return {
          id,
          index,
          origin,
          extent,
          scalarRange: parseScalarRange(brickSource.scalarRange, cells),
          physicalBoundsM: { lower, upper },
          contentSha256,
        };
      });
      return { level, stride, shape, brickShape, source: levelSourceDescriptor, bricks };
    });
    return {
      ok: true,
      hierarchy: {
        schema: FIELD_BRICK_HIERARCHY_SCHEMA,
        schemaVersion,
        field,
        publicationGeneration,
        sourceShape,
        sourceIndexOffset,
        sourceDtype: "float32",
        sampleLocation: "cell_center",
        extremaSemantics: source.extremaSemantics,
        extremaAbsolutePadding,
        cullingSafe: source.cullingSafe,
        levels,
      },
    };
  } catch (error) {
    return { ok: false, reason: error instanceof Error ? error.message : String(error) };
  }
}

function levelOf(hierarchy: FieldBrickHierarchy, level: number): FieldBrickLevel {
  const selected = hierarchy.levels[level];
  if (!selected || selected.level !== level) throw new Error(`Brick level ${level} is unavailable`);
  return selected;
}

function rangeCrosses(range: FieldBrickScalarRange, threshold: number): boolean {
  return range.finiteCount > 0 && range.minimum != null && range.maximum != null
    && threshold >= range.minimum && threshold <= range.maximum;
}

export function levelReadWindow(
  hierarchy: FieldBrickHierarchy,
  level: FieldBrickLevel,
  levelOrigin: Triple,
  levelExtent: Triple,
  terrainFollowing = false,
): FieldLoadWindow {
  const localSourceOrigin = levelOrigin.map((value, axis) =>
    value * level.stride[axis]) as Triple;
  const sourceExtent = levelExtent.map((value, axis) => Math.min(
    hierarchy.sourceShape[axis] - localSourceOrigin[axis],
    value * level.stride[axis],
  )) as Triple;
  const outputCells = checkedProduct(levelExtent, "brick output");
  if (terrainFollowing && level.source.kind === "npy_lod" && !level.source.geometry) {
    throw new Error("Durable terrain LOD geometry is not yet published");
  }
  const outputBytes = outputCells * (terrainFollowing ? 36 : FLOAT32_BYTES)
    + (terrainFollowing && outputCells % 2 === 1 ? 4 : 0);
  const sourceBytes = level.source.kind === "npy_lod"
    ? outputBytes
    : terrainFollowing
      ? (() => {
          const sourceCells = checkedProduct(sourceExtent, "terrain brick source");
          const interfaceCells = checkedProduct(
            [sourceExtent[0], sourceExtent[1], sourceExtent[2] + 1],
            "terrain brick interfaces",
          );
          return sourceCells * FLOAT32_BYTES
            + (sourceCells + interfaceCells) * Float64Array.BYTES_PER_ELEMENT
            + outputCells * 69;
        })()
      : checkedProduct(sourceExtent, "brick source") * FLOAT32_BYTES;
  return {
    aggregation: "cell_mean",
    origin: localSourceOrigin.map((value, axis) =>
      value + hierarchy.sourceIndexOffset[axis]),
    extent: sourceExtent,
    stride: [...level.stride],
    maxOutputBytes: outputBytes,
    maxSourceBytes: sourceBytes,
  };
}

export function planContourBricks(
  hierarchy: FieldBrickHierarchy,
  threshold: number,
  options: {
    level?: number;
    haloSamples?: number;
    maximumSourceBytes?: number;
    terrainFollowing?: boolean;
    excludePhysicalBoundsM?: {
      lower: readonly [number, number, number];
      upper: readonly [number, number, number];
    };
  } = {},
): ContourBrickPlan {
  if (!hierarchy.cullingSafe) {
    throw new Error("Contour brick culling requires safe decoded-value bounds");
  }
  if (!Number.isFinite(threshold)) throw new Error("Contour threshold must be finite");
  const level = levelOf(hierarchy, options.level ?? 0);
  const halo = options.haloSamples ?? 1;
  if (!Number.isSafeInteger(halo) || halo < 1) throw new Error("Contour bricks require a positive sample halo");
  const maximumSourceBytes = options.maximumSourceBytes ?? Number.MAX_SAFE_INTEGER;
  if (!Number.isSafeInteger(maximumSourceBytes) || maximumSourceBytes <= 0) {
    throw new Error("Contour brick source budget is invalid");
  }
  const items: ContourBrickPlanItem[] = [];
  let sourceBytes = 0;
  for (const brick of level.bricks) {
    if (!rangeCrosses(brick.scalarRange, threshold)) continue;
    const localReadOrigin = brick.origin.map((value) => Math.max(0, value - halo)) as Triple;
    const readUpper = brick.origin.map((value, axis) =>
      Math.min(level.shape[axis], value + brick.extent[axis] + halo)) as Triple;
    const readExtent = readUpper.map((value, axis) => value - localReadOrigin[axis]) as Triple;
    const ownedCellExtent = brick.origin.map((value, axis) =>
      Math.max(0, Math.min(brick.extent[axis], level.shape[axis] - 1 - value))) as Triple;
    if (ownedCellExtent.some((value) => value === 0)) continue;
    const readWindow = levelReadWindow(
      hierarchy, level, localReadOrigin, readExtent, options.terrainFollowing);
    const bytes = readWindow.maxSourceBytes!;
    if (sourceBytes + bytes > maximumSourceBytes) {
      throw new Error(
        `Contour brick plan requires ${sourceBytes + bytes} bytes; the admitted source budget is ${maximumSourceBytes}.`,
      );
    }
    sourceBytes += bytes;
    items.push({
      brickId: brick.id,
      brickIndex: brick.index,
      readWindow,
      ownedCellOrigin: brick.origin,
      ownedCellExtent,
      scalarRange: brick.scalarRange,
    });
  }
  return {
    field: hierarchy.field,
    publicationGeneration: hierarchy.publicationGeneration,
    level: level.level,
    threshold,
    items,
    culledBrickCount: level.bricks.length - items.length,
    sourceBytes,
  };
}

export function unionContourBrickPlans(
  primary: ContourBrickPlan,
  opposite: ContourBrickPlan | null,
): ContourBrickPlan {
  if (!opposite) return primary;
  if (primary.field !== opposite.field
      || primary.publicationGeneration !== opposite.publicationGeneration
      || primary.level !== opposite.level) {
    throw new Error("Contour brick plans do not share one field generation");
  }
  const items = new Map(primary.items.map((item) => [item.brickId, item]));
  for (const item of opposite.items) items.set(item.brickId, item);
  const merged = [...items.values()].sort((left, right) => left.brickId.localeCompare(right.brickId));
  return {
    ...primary,
    items: merged,
    culledBrickCount: primary.items.length + primary.culledBrickCount - merged.length,
    sourceBytes: merged.reduce((sum, item) => sum + (item.readWindow.maxSourceBytes ?? 0), 0),
  };
}

export function assertCompatibleBrickHierarchies(
  hierarchies: readonly FieldBrickHierarchy[],
): void {
  if (hierarchies.length < 2) throw new Error("Compatible brick planning requires multiple fields");
  const reference = hierarchies[0];
  for (const hierarchy of hierarchies.slice(1)) {
    if (hierarchy.publicationGeneration !== reference.publicationGeneration
        || hierarchy.sourceShape.join("x") !== reference.sourceShape.join("x")
        || hierarchy.sourceIndexOffset.join("x") !== reference.sourceIndexOffset.join("x")
        || hierarchy.levels.length !== reference.levels.length) {
      throw new Error("Brick hierarchies do not share one field generation and source lattice");
    }
    for (let levelIndex = 0; levelIndex < reference.levels.length; levelIndex += 1) {
      const left = reference.levels[levelIndex];
      const right = hierarchy.levels[levelIndex];
      if (left.stride.join("x") !== right.stride.join("x")
          || left.shape.join("x") !== right.shape.join("x")
          || left.brickShape.join("x") !== right.brickShape.join("x")
          || left.bricks.length !== right.bricks.length) {
        throw new Error("Brick hierarchy levels are not topologically compatible");
      }
      for (let brickIndex = 0; brickIndex < left.bricks.length; brickIndex += 1) {
        const a = left.bricks[brickIndex];
        const b = right.bricks[brickIndex];
        if (a.id !== b.id || a.origin.join("x") !== b.origin.join("x")
            || a.extent.join("x") !== b.extent.join("x")
            || JSON.stringify(a.physicalBoundsM) !== JSON.stringify(b.physicalBoundsM)) {
          throw new Error("Brick hierarchy directories disagree on ownership geometry");
        }
      }
    }
  }
}

export function planAllContourBricks(
  hierarchy: FieldBrickHierarchy,
  options: {
    level?: number;
    haloSamples?: number;
    maximumSourceBytes?: number;
    terrainFollowing?: boolean;
  } = {},
): ContourBrickPlan {
  const level = levelOf(hierarchy, options.level ?? 0);
  const halo = options.haloSamples ?? 1;
  if (!Number.isSafeInteger(halo) || halo < 1) throw new Error("Contour bricks require a positive sample halo");
  const maximumSourceBytes = options.maximumSourceBytes ?? Number.MAX_SAFE_INTEGER;
  const items: ContourBrickPlanItem[] = [];
  let sourceBytes = 0;
  for (const brick of level.bricks) {
    const localReadOrigin = brick.origin.map((value) => Math.max(0, value - halo)) as Triple;
    const readUpper = brick.origin.map((value, axis) =>
      Math.min(level.shape[axis], value + brick.extent[axis] + halo)) as Triple;
    const readExtent = readUpper.map((value, axis) => value - localReadOrigin[axis]) as Triple;
    const ownedCellExtent = brick.origin.map((value, axis) =>
      Math.max(0, Math.min(brick.extent[axis], level.shape[axis] - 1 - value))) as Triple;
    if (ownedCellExtent.some((value) => value === 0)) continue;
    const readWindow = levelReadWindow(
      hierarchy, level, localReadOrigin, readExtent, options.terrainFollowing);
    const bytes = readWindow.maxSourceBytes!;
    if (sourceBytes + bytes > maximumSourceBytes) {
      throw new Error(
        `Contour brick plan requires ${sourceBytes + bytes} bytes; the admitted source budget is ${maximumSourceBytes}.`,
      );
    }
    sourceBytes += bytes;
    items.push({
      brickId: brick.id,
      brickIndex: brick.index,
      readWindow,
      ownedCellOrigin: brick.origin,
      ownedCellExtent,
      scalarRange: brick.scalarRange,
    });
  }
  return {
    field: hierarchy.field,
    publicationGeneration: hierarchy.publicationGeneration,
    level: level.level,
    threshold: 0,
    items,
    culledBrickCount: 0,
    sourceBytes,
  };
}

export function selectContourHierarchyLevel(
  hierarchy: FieldBrickHierarchy,
  options: {
    maximumPreparedCells: number;
    threshold?: number;
    oppositeThreshold?: number | null;
    allBricks?: boolean;
    terrainFollowing?: boolean;
  },
): number {
  if (!Number.isSafeInteger(options.maximumPreparedCells)
      || options.maximumPreparedCells <= 0) {
    throw new Error("Contour hierarchy cell budget is invalid");
  }
  if (!options.allBricks && !Number.isFinite(options.threshold)) {
    throw new Error("Contour hierarchy selection requires a finite threshold");
  }
  for (const level of hierarchy.levels) {
    if (options.terrainFollowing && level.source.kind === "npy_lod" && !level.source.geometry) continue;
    const primary = options.allBricks
      ? planAllContourBricks(hierarchy, {
          level: level.level,
          terrainFollowing: options.terrainFollowing,
        })
      : planContourBricks(hierarchy, options.threshold!, {
          level: level.level,
          terrainFollowing: options.terrainFollowing,
        });
    const opposite = !options.allBricks && options.oppositeThreshold != null
      ? planContourBricks(hierarchy, options.oppositeThreshold, {
          level: level.level,
          terrainFollowing: options.terrainFollowing,
        })
      : null;
    const union = unionContourBrickPlans(primary, opposite);
    const preparedCells = union.items.reduce((sum, item) =>
      sum + (item.readWindow.maxOutputBytes ?? 0)
        / (options.terrainFollowing ? 36 : FLOAT32_BYTES), 0);
    if (preparedCells <= options.maximumPreparedCells) return level.level;
  }
  throw new Error(
    `The coarsest durable brick level exceeds the admitted ${options.maximumPreparedCells.toLocaleString()}-cell working set.`,
  );
}

function opacityAt(points: readonly (readonly [number, number])[], scalar: number): number {
  if (points.length === 0) return 0;
  if (scalar <= points[0][0]) return points[0][1];
  for (let index = 1; index < points.length; index += 1) {
    const right = points[index];
    const left = points[index - 1];
    if (scalar > right[0]) continue;
    const span = right[0] - left[0];
    const fraction = span > 0 ? (scalar - left[0]) / span : 0;
    return left[1] + fraction * (right[1] - left[1]);
  }
  return points[points.length - 1][1];
}

export function scalarIntervalHasOpacity(
  minimum: number,
  maximum: number,
  opacityPoints: readonly (readonly [number, number])[],
): boolean {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum < minimum) return false;
  const sorted = [...opacityPoints]
    .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]))
    .sort((left, right) => left[0] - right[0]);
  if (opacityAt(sorted, minimum) > 0 || opacityAt(sorted, maximum) > 0) return true;
  return sorted.some(([scalar, opacity]) => scalar > minimum && scalar < maximum && opacity > 0);
}

function squaredDistanceToBrick(brick: FieldBrickDescriptor, focus: Triple): number {
  let squared = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const lower = brick.origin[axis];
    const upper = lower + brick.extent[axis];
    const delta = focus[axis] < lower ? lower - focus[axis] : focus[axis] > upper ? focus[axis] - upper : 0;
    squared += delta * delta;
  }
  return squared;
}

export function planVolumeBricks(
  hierarchy: FieldBrickHierarchy,
  opacityPoints: readonly (readonly [number, number])[],
  options: {
    level?: number;
    focusIndex?: readonly number[];
    maximumResidentBytes: number;
    terrainFollowing?: boolean;
    haloSamples?: number;
    excludePhysicalBoundsM?: {
      lower: readonly [number, number, number];
      upper: readonly [number, number, number];
    };
  },
): VolumeBrickPlan {
  if (!hierarchy.cullingSafe) {
    throw new Error("Volume brick skipping requires safe decoded-value bounds");
  }
  const level = levelOf(hierarchy, options.level ?? 0);
  const halo = options.haloSamples ?? 0;
  if (!Number.isSafeInteger(halo) || halo < 0 || halo > 2) throw new Error("Volume halo must contain zero, one, or two samples");
  if (!Number.isSafeInteger(options.maximumResidentBytes) || options.maximumResidentBytes <= 0) {
    throw new Error("Volume brick residency budget is invalid");
  }
  const focus = options.focusIndex == null
    ? level.shape.map((count) => (count - 1) * 0.5) as Triple
    : options.focusIndex.map(Number) as Triple;
  if (focus.length !== 3 || !focus.every(Number.isFinite)) throw new Error("Volume brick focus is invalid");
  const excludedBrickCount = level.bricks.filter((brick) => {
    const bounds = options.excludePhysicalBoundsM;
    return bounds != null && brick.physicalBoundsM.lower.every((value, axis) =>
      value >= bounds.lower[axis] && brick.physicalBoundsM.upper[axis] <= bounds.upper[axis]);
  }).length;
  const byIndex = new Map(level.bricks.map((brick) => [brick.index.join(":"), brick]));
  let retainedSourceBricks = 0;
  let boundaryOnlyBrickCount = 0;
  const candidates = level.bricks.flatMap((source) => {
    const range = source.scalarRange;
    if (range.finiteCount === 0 || range.minimum == null || range.maximum == null) return [];
    const bounds = options.excludePhysicalBoundsM;
    if (bounds && source.physicalBoundsM.lower.every((value, axis) =>
      value >= bounds.lower[axis] && source.physicalBoundsM.upper[axis] <= bounds.upper[axis])) return [];
    const candidate = (brick: FieldBrickDescriptor, fringe: boolean) => {
      const support = volumeBrickSupport(level, brick.origin, brick.extent, halo);
      return { brick, fringe, support, bytes: checkedProduct(support.readExtent, "volume brick") * FLOAT32_BYTES,
        distance: squaredDistanceToBrick(brick, focus) };
    };
    if (scalarIntervalHasOpacity(range.minimum, range.maximum, opacityPoints)) {
      retainedSourceBricks++;
      return [candidate(source, false)];
    }
    if (halo === 0) return [];
    const neighbors: Triple[] = [];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      if (dx === 0 && dy === 0 && dz === 0) continue;
      const other = byIndex.get([source.index[0] + dx, source.index[1] + dy, source.index[2] + dz].join(":"))?.scalarRange;
      if (other && other.finiteCount > 0 && other.minimum != null && other.maximum != null
          && scalarIntervalHasOpacity(Math.min(range.minimum, other.minimum), Math.max(range.maximum, other.maximum), opacityPoints)) {
        neighbors.push([dx, dy, dz]);
      }
    }
    const regions = volumeBoundaryRegions(source.origin, source.extent, neighbors);
    if (regions.length) { retainedSourceBricks++; boundaryOnlyBrickCount++; }
    return regions.map((region, index) => candidate({ ...source,
      id: `${source.id}:boundary:${index}`, origin: region.origin, extent: region.extent,
    }, true));
  }).sort((left, right) => left.distance - right.distance || left.brick.id.localeCompare(right.brick.id));
  const items: VolumeBrickPlanItem[] = [];
  let residentBytes = 0;
  for (const candidate of candidates) {
    if (residentBytes + candidate.bytes > options.maximumResidentBytes) continue;
    residentBytes += candidate.bytes;
    items.push({
      brickId: candidate.brick.id,
      brickIndex: candidate.brick.index,
      window: levelReadWindow(
        hierarchy,
        level,
        candidate.support.readOrigin,
        candidate.support.readExtent,
        options.terrainFollowing,
      ),
      scalarRange: candidate.brick.scalarRange,
      priority: candidate.distance,
      preparedScalarBytes: candidate.bytes,
      ...(candidate.fringe ? { boundaryOnly: true } : {}),
      ...(halo ? { ownedOffset: candidate.support.ownedOffset, ownedShape: candidate.brick.extent } : {}),
    });
  }
  return {
    field: hierarchy.field,
    publicationGeneration: hierarchy.publicationGeneration,
    level: level.level,
    items,
    transparentBrickCount: level.bricks.length - retainedSourceBricks,
    ...(halo ? { boundaryOnlyBrickCount } : {}),
    deferredBrickCount: candidates.length - items.length,
    residentBytes,
    excludedBrickCount,
    ...(halo ? { haloSamples: halo } : {}),
  };
}

export class VolumeResidencyError extends Error {
  constructor(maximumResidentBytes: number) {
    super(`The coarsest volume hierarchy exceeds the admitted ${maximumResidentBytes.toLocaleString()}-byte residency.`);
    this.name = "VolumeResidencyError";
  }
}

export function selectVolumeHierarchyPlan(
  hierarchy: FieldBrickHierarchy,
  opacityPoints: readonly (readonly [number, number])[],
  options: {
    focusIndex?: readonly number[];
    maximumResidentBytes: number;
    terrainFollowing?: boolean;
    haloSamples?: number;
    requestedLevel?: number | null;
    excludePhysicalBoundsM?: {
      lower: readonly [number, number, number];
      upper: readonly [number, number, number];
    };
  },
): VolumeBrickPlan {
  const requestedLevel = options.requestedLevel == null
    ? 0
    : Math.max(0, Math.min(hierarchy.levels.length - 1, options.requestedLevel));
  for (const level of hierarchy.levels.slice(requestedLevel)) {
    if (options.terrainFollowing
        && level.source.kind === "npy_lod" && !level.source.geometry) continue;
    const plan = planVolumeBricks(hierarchy, opacityPoints, {
      level: level.level,
      focusIndex: options.focusIndex?.map((value, axis) =>
        value / level.stride[axis]),
      maximumResidentBytes: options.maximumResidentBytes,
      terrainFollowing: options.terrainFollowing,
      haloSamples: options.haloSamples,
      excludePhysicalBoundsM: options.excludePhysicalBoundsM,
    });
    if (plan.deferredBrickCount === 0) return plan;
  }
  throw new VolumeResidencyError(options.maximumResidentBytes);
}

export function compactVolumeBrickPlan(
  hierarchy: FieldBrickHierarchy,
  plan: VolumeBrickPlan,
  options: {
    maximumResidentBytes: number;
    terrainFollowing?: boolean;
    targetShape?: Triple;
  },
): VolumeBrickPlan {
  if (plan.items.length < 2 || plan.deferredBrickCount > 0) return plan;
  const level = levelOf(hierarchy, plan.level);
  const target = options.targetShape ?? [256, 128, 80];
  const groupShape = target.map((count, axis) => Math.max(
    1,
    Math.floor(count / level.brickShape[axis]),
  )) as Triple;
  const hasBoundaries = plan.items.some((item) => item.boundaryOnly);
  const groups = new Map<string, VolumeBrickPlanItem[]>();
  for (const item of plan.items) {
    const index = item.brickIndex.map((value, axis) =>
      Math.floor(value / groupShape[axis])) as Triple;
    const key = index.join(":");
    const group = groups.get(key) ?? [];
    group.push(item);
    groups.set(key, group);
  }
  const compacted: VolumeBrickPlanItem[] = [];
  let residentBytes = 0;
  for (const [key, items] of [...groups].sort(([left], [right]) => left.localeCompare(right))) {
    if (hasBoundaries) {
      const regions = compactOwnedVolumeRegions(items.map((item) => ({
        origin: item.window.origin.map((value, axis) =>
          (value - hierarchy.sourceIndexOffset[axis]) / level.stride[axis] + item.ownedOffset![axis]) as Triple,
        extent: item.ownedShape!, items: [item],
      })));
      for (const [index, region] of regions.entries()) {
        const support = volumeBrickSupport(level, region.origin, region.extent, plan.haloSamples ?? 0);
        const bytes = checkedProduct(support.readExtent, "compacted volume support") * FLOAT32_BYTES;
        residentBytes += bytes;
        compacted.push({
          brickId: `m${level.level}:${key}:${index}`,
          brickIndex: region.origin.map((value, axis) => Math.floor(value / level.brickShape[axis])) as Triple,
          window: levelReadWindow(hierarchy, level, support.readOrigin, support.readExtent, options.terrainFollowing),
          scalarRange: {
            minimum: Math.min(...region.items.map((item) => item.scalarRange.minimum!)),
            maximum: Math.max(...region.items.map((item) => item.scalarRange.maximum!)),
            finiteCount: region.items.reduce((sum, item) => sum + item.scalarRange.finiteCount, 0),
            nonFiniteCount: region.items.reduce((sum, item) => sum + item.scalarRange.nonFiniteCount, 0),
          },
          priority: Math.min(...region.items.map((item) => item.priority)),
          preparedScalarBytes: bytes, ownedOffset: support.ownedOffset, ownedShape: region.extent,
          ...(region.items.every((item) => item.boundaryOnly) ? { boundaryOnly: true } : {}),
        });
      }
      continue;
    }
    const groupIndex = key.split(":").map(Number) as Triple;
    const origin = groupIndex.map((value, axis) =>
      value * groupShape[axis] * level.brickShape[axis]) as Triple;
    const extent = origin.map((value, axis) => Math.min(
      groupShape[axis] * level.brickShape[axis],
      level.shape[axis] - value,
    )) as Triple;
    const support = volumeBrickSupport(level, origin, extent, plan.haloSamples ?? 0);
    const bytes = checkedProduct(support.readExtent, "compacted volume brick") * FLOAT32_BYTES;
    const groupRanges = level.bricks.filter((brick) => brick.origin.every((value, axis) =>
      value >= origin[axis] && value < origin[axis] + extent[axis]))
      .map((brick) => brick.scalarRange);
    if (plan.haloSamples && groupRanges.length !== items.length) {
      compacted.push(...items);
      residentBytes += items.reduce((sum, item) => sum + item.preparedScalarBytes, 0);
      continue;
    }
    residentBytes += bytes;
    const finiteRanges = groupRanges.filter((range) =>
      range.finiteCount > 0 && range.minimum != null && range.maximum != null);
    compacted.push({
      brickId: `m${level.level}:${key}`,
      brickIndex: groupIndex,
      window: levelReadWindow(
        hierarchy,
        level,
        support.readOrigin,
        support.readExtent,
        options.terrainFollowing,
      ),
      scalarRange: finiteRanges.length > 0 ? {
        minimum: Math.min(...finiteRanges.map((range) => range.minimum!)),
        maximum: Math.max(...finiteRanges.map((range) => range.maximum!)),
        finiteCount: groupRanges.reduce((sum, range) => sum + range.finiteCount, 0),
        nonFiniteCount: groupRanges.reduce((sum, range) => sum + range.nonFiniteCount, 0),
      } : { minimum: null, maximum: null, finiteCount: 0, nonFiniteCount: checkedProduct(extent, "empty compacted volume brick") },
      priority: Math.min(...items.map((item) => item.priority)),
      preparedScalarBytes: bytes,
      ...(plan.haloSamples ? { ownedOffset: support.ownedOffset, ownedShape: extent } : {}),
    });
  }
  if (compacted.length >= plan.items.length
      || residentBytes > options.maximumResidentBytes) return plan;
  return { ...plan, items: compacted, residentBytes };
}

interface OwnedVolumeRegion {
  origin: Triple;
  extent: Triple;
  items: VolumeBrickPlanItem[];
}

function compactOwnedVolumeRegions(regions: OwnedVolumeRegion[]): OwnedVolumeRegion[] {
  if (regions.length < 2) return regions;
  const bounds = (parts: OwnedVolumeRegion[]) => {
    const origin = [0, 1, 2].map((axis) => Math.min(...parts.map((part) => part.origin[axis]))) as Triple;
    const extent = origin.map((value, axis) =>
      Math.max(...parts.map((part) => part.origin[axis] + part.extent[axis])) - value) as Triple;
    return { origin, extent, volume: checkedProduct(extent, "owned volume region") };
  };
  const whole = bounds(regions);
  const occupied = regions.reduce((sum, region) => sum + checkedProduct(region.extent, "owned region"), 0);
  // Input owners are disjoint. Equal volume proves a rectangular union without
  // filling a culled hole; partition planes preserve camera-sortable output.
  if (whole.volume === occupied) return [{ origin: whole.origin, extent: whole.extent,
    items: regions.flatMap((region) => region.items) }];
  let best: { lower: OwnedVolumeRegion[]; upper: OwnedVolumeRegion[]; volume: number; balance: number } | null = null;
  for (let axis = 0; axis < 3; axis++) {
    const cuts = new Set(regions.flatMap((region) => [region.origin[axis], region.origin[axis] + region.extent[axis]]));
    for (const cut of cuts) {
      const lower = regions.filter((region) => region.origin[axis] + region.extent[axis] <= cut);
      const upper = regions.filter((region) => region.origin[axis] >= cut);
      if (!lower.length || !upper.length || lower.length + upper.length !== regions.length) continue;
      const volume = bounds(lower).volume + bounds(upper).volume;
      const balance = Math.abs(lower.length - upper.length);
      if (!best || volume < best.volume || (volume === best.volume && balance < best.balance)) {
        best = { lower, upper, volume, balance };
      }
    }
  }
  return best ? [...compactOwnedVolumeRegions(best.lower), ...compactOwnedVolumeRegions(best.upper)] : regions;
}

function volumeBrickSupport(level: FieldBrickLevel, origin: Triple, extent: Triple, halo: number) {
  const readOrigin = origin.map((value) => Math.max(0, value - halo)) as Triple;
  const readExtent = origin.map((value, axis) =>
    Math.min(level.shape[axis], value + extent[axis] + halo) - readOrigin[axis]) as Triple;
  return { readOrigin, readExtent, ownedOffset: origin.map((value, axis) => value - readOrigin[axis]) as Triple };
}

function volumeBoundaryRegions(origin: Triple, extent: Triple, neighbors: readonly Triple[]): Array<{ origin: Triple; extent: Triple }> {
  if (!neighbors.length) return [];
  const cuts = extent.map((count) => [...new Set([0, 1, Math.max(1, count - 1), count])].sort((a, b) => a - b));
  const selected = (index: Triple) => neighbors.some((neighbor) => neighbor.every((direction, axis) =>
    direction === 0 || (direction < 0 ? cuts[axis][index[axis]] === 0 : cuts[axis][index[axis] + 1] === extent[axis])));
  const regions: Array<{ origin: Triple; extent: Triple }> = [];
  const visit = (lower: Triple, upper: Triple): void => {
    let occupied = 0;
    let cells = 0;
    for (let x = lower[0]; x < upper[0]; x++) for (let y = lower[1]; y < upper[1]; y++) for (let z = lower[2]; z < upper[2]; z++) {
      cells++;
      if (selected([x, y, z])) occupied++;
    }
    if (occupied === 0) return;
    if (occupied === cells) {
      regions.push({
        origin: lower.map((value, axis) => origin[axis] + cuts[axis][value]) as Triple,
        extent: upper.map((value, axis) => cuts[axis][value] - cuts[axis][lower[axis]]) as Triple,
      });
      return;
    }
    const axis = upper.findIndex((value, axis) => value - lower[axis] > 1);
    const split = lower[axis] + 1;
    const firstUpper = [...upper] as Triple;
    firstUpper[axis] = split;
    const secondLower = [...lower] as Triple;
    secondLower[axis] = split;
    visit(lower, firstUpper);
    visit(secondLower, upper);
  };
  visit([0, 0, 0], cuts.map((values) => values.length - 1) as Triple);
  return regions;
}

export function adjacentBrickIds(level: FieldBrickLevel, brickId: string): string[] {
  const brick = level.bricks.find((candidate) => candidate.id === brickId);
  if (!brick) throw new Error(`Brick ${brickId} is unavailable`);
  const counts = expectedBrickCount(level.shape, level.brickShape);
  const result: string[] = [];
  for (let axis = 0; axis < 3; axis += 1) {
    for (const direction of [-1, 1]) {
      const index = [...brick.index] as Triple;
      index[axis] += direction;
      if (index[axis] >= 0 && index[axis] < counts[axis]) {
        result.push(canonicalBrickId(level.level, index));
      }
    }
  }
  return result.sort();
}

export function selectVectorBrickFeatureFocus(
  hierarchies: readonly [FieldBrickHierarchy, FieldBrickHierarchy, FieldBrickHierarchy],
  options: { maximumSourceZFraction?: number } = {},
): Triple {
  assertCompatibleBrickHierarchies(hierarchies);
  const maximumSourceZFraction = options.maximumSourceZFraction ?? 1;
  if (!Number.isFinite(maximumSourceZFraction)
      || maximumSourceZFraction <= 0 || maximumSourceZFraction > 1) {
    throw new Error("Vector brick focus height fraction is invalid");
  }
  const level = hierarchies[0].levels[0];
  const maximumZ = hierarchies[0].sourceShape[2] * maximumSourceZFraction;
  const candidates = level.bricks.filter((brick) => brick.origin[2] < maximumZ);
  const pool = candidates.length > 0 ? candidates : level.bricks;
  let selected: FieldBrickDescriptor | null = null;
  let selectedScore = Number.NEGATIVE_INFINITY;
  for (let brickIndex = 0; brickIndex < pool.length; brickIndex += 1) {
    const reference = pool[brickIndex];
    let score = 0;
    let valid = false;
    for (const hierarchy of hierarchies) {
      const range = hierarchy.levels[0].bricks.find((brick) => brick.id === reference.id)?.scalarRange;
      if (!range || range.finiteCount === 0) continue;
      const bound = Math.max(Math.abs(range.minimum ?? 0), Math.abs(range.maximum ?? 0));
      if (!Number.isFinite(bound)) continue;
      score += bound * bound;
      valid = true;
    }
    if (valid && score > selectedScore) {
      selected = reference;
      selectedScore = score;
    }
  }
  if (!selected) return hierarchies[0].sourceShape.map((extent) => (extent - 1) * 0.5) as Triple;
  return selected.origin.map((origin, axis) =>
    origin + Math.max(0, selected!.extent[axis] - 1) * 0.5) as Triple;
}

export function planVectorBrickNeighborhood(
  hierarchies: readonly [FieldBrickHierarchy, FieldBrickHierarchy, FieldBrickHierarchy],
  options: {
    focusSourceIndex?: readonly number[];
    brickRadius?: number;
    expansion?: VectorBrickExpansion;
    maximumPreparedCells: number;
    terrainFollowing?: boolean;
  },
): VectorBrickNeighborhoodPlan {
  assertCompatibleBrickHierarchies(hierarchies);
  if (!Number.isSafeInteger(options.maximumPreparedCells) || options.maximumPreparedCells <= 0) {
    throw new Error("Vector brick working-set budget is invalid");
  }
  const radius = options.brickRadius ?? 1;
  if (!Number.isSafeInteger(radius) || radius < 0 || radius > 4) {
    throw new Error("Vector brick adjacency radius is invalid");
  }
  const expansion = options.expansion ?? {
    lower: [radius, radius, radius],
    upper: [radius, radius, radius],
  };
  for (const side of [expansion.lower, expansion.upper]) {
    if (side.length !== 3 || !side.every((value) =>
      Number.isSafeInteger(value) && value >= 0)) {
      throw new Error("Vector brick expansion is invalid");
    }
  }
  const hierarchy = hierarchies[0];
  const sourceFocus = options.focusSourceIndex == null
    ? hierarchy.sourceShape.map((extent) => (extent - 1) * 0.5)
    : options.focusSourceIndex.map(Number);
  if (sourceFocus.length !== 3 || !sourceFocus.every(Number.isFinite)) {
    throw new Error("Vector brick focus is invalid");
  }
  for (const level of hierarchy.levels) {
    if (options.terrainFollowing
        && level.source.kind === "npy_lod" && hierarchies.some((candidate) => {
          const source = candidate.levels[level.level].source;
          return source.kind !== "npy_lod" || !source.geometry;
        })) continue;
    if (level.source.kind === "npy_lod"
        && hierarchies.some((candidate) => candidate.levels[level.level].source.kind !== "npy_lod")) {
      continue;
    }
    const focus = sourceFocus.map((value, axis) => value / level.stride[axis]);
    const brickCounts = level.shape.map((extent, axis) =>
      Math.ceil(extent / level.brickShape[axis])) as Triple;
    const focusBrick = focus.map((value, axis) => Math.max(0, Math.min(
      brickCounts[axis] - 1,
      Math.floor(value / level.brickShape[axis]),
    ))) as Triple;
    const lower = focusBrick.map((value, axis) =>
      Math.max(0, value - expansion.lower[axis])) as Triple;
    const upper = focusBrick.map((value, axis) => Math.min(
      brickCounts[axis] - 1,
      value + expansion.upper[axis],
    )) as Triple;
    const items = level.bricks.filter((brick) => brick.index.every((value, axis) =>
      value >= lower[axis] && value <= upper[axis])).map((brick) => ({
        brickId: brick.id,
        brickIndex: brick.index,
        window: levelReadWindow(
          hierarchy,
          level,
          brick.origin,
          brick.extent,
          options.terrainFollowing,
        ),
      }));
    const preparedCells = items.reduce((sum, item) =>
      sum + checkedProduct(
        item.window.extent.map((extent, axis) =>
          Math.floor((extent - 1) / item.window.stride[axis]) + 1) as Triple,
        "vector brick prepared output",
      ), 0);
    if (preparedCells <= options.maximumPreparedCells) {
      return {
        publicationGeneration: hierarchy.publicationGeneration,
        level: level.level,
        componentFields: [hierarchies[0].field, hierarchies[1].field, hierarchies[2].field],
        items,
        preparedCells,
        totalBrickCount: level.bricks.length,
        focusBrick,
        lowerBrick: lower,
        upperBrick: upper,
        brickCounts,
        expansion: {
          lower: [...expansion.lower],
          upper: [...expansion.upper],
        },
      };
    }
  }
  throw new Error(
    `The coarsest vector brick neighborhood exceeds the admitted ${options.maximumPreparedCells.toLocaleString()}-cell working set.`,
  );
}

export function expandVectorBrickNeighborhood(
  plan: VectorBrickNeighborhoodPlan,
  contacts: BrickBoundaryContacts,
): VectorBrickExpansion | null {
  const lower = [...plan.expansion.lower] as Triple;
  const upper = [...plan.expansion.upper] as Triple;
  let changed = false;
  for (let axis = 0; axis < 3; axis += 1) {
    if (contacts.lower[axis] && plan.lowerBrick[axis] > 0) {
      lower[axis] += 1;
      changed = true;
    }
    if (contacts.upper[axis] && plan.upperBrick[axis] + 1 < plan.brickCounts[axis]) {
      upper[axis] += 1;
      changed = true;
    }
  }
  return changed ? { lower, upper } : null;
}
