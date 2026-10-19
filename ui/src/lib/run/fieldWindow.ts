/*
 * Subset a cached field before it crosses the postMessage boundary into wasm.
 * Reduction happens on the main thread, so interpreter memory is bounded by the request.
 */

import type { NpyN } from "@/lib/platform/npy";
import type { FieldWindow } from "@/lib/platform/pyodideProtocol";
import {
  MAX_NATIVE_FIELD_WINDOW_BYTES,
  MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
  nativeFieldWindowBodyBytes,
  type FieldWindowAggregation,
} from "@/lib/dataset/nativeFieldWindow";
import type { FieldLoadWindow } from "./runOutputs";

/** Every cell of every axis: what a request with no window means. */
export const WHOLE_FIELD: FieldWindow = Object.freeze({
  stride: Object.freeze([]) as readonly number[],
  box: null,
});

const MAX_PYTHON_FIELD_TILES = 4096;
const MAX_DENSE_PYTHON_SOURCE_BYTES = 1024 * 1024 * 1024;

function checkedProduct(values: readonly number[], context: string): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product)) throw new Error(`${context} exceeds safe integer range`);
  }
  return product;
}

function enforceTileCount(shape: readonly number[], tile: readonly number[]): void {
  const count = checkedProduct(
    shape.map((extent, axis) => Math.ceil(extent / tile[axis])),
    "Python field tile count",
  );
  if (count > MAX_PYTHON_FIELD_TILES) {
    throw new Error(
      `Python field request needs ${count.toLocaleString()} native tiles; `
        + `the safety limit is ${MAX_PYTHON_FIELD_TILES.toLocaleString()}. `
        + "Use a tighter box, or a storage format with native sparse sampling.",
    );
  }
}

/** Normalise a window against a real shape. Missing entries mean all of that axis; bounds are clamped, not rejected. */
export function resolveWindow(
  shape: readonly number[],
  window: FieldWindow | undefined,
): { readonly starts: number[]; readonly stops: number[]; readonly steps: number[] } {
  const starts: number[] = [];
  const stops: number[] = [];
  const steps: number[] = [];
  for (let axis = 0; axis < shape.length; axis += 1) {
    const extent = Math.max(0, shape[axis]);
    const bounds = window?.box?.[axis];
    const rawStart = bounds ? Math.trunc(bounds[0]) : 0;
    const rawStop = bounds ? Math.trunc(bounds[1]) : extent;
    const start = Math.min(Math.max(0, rawStart), extent);
    starts.push(start);
    stops.push(Math.min(Math.max(start, rawStop), extent));
    steps.push(Math.max(1, Math.trunc(window?.stride?.[axis] ?? 1)));
  }
  return { starts, stops, steps };
}

/** The shape a window produces, without producing it. */
export function windowedShape(
  shape: readonly number[],
  window: FieldWindow | undefined,
): number[] {
  const { starts, stops, steps } = resolveWindow(shape, window);
  return shape.map((_, axis) =>
    Math.max(0, Math.ceil((stops[axis] - starts[axis]) / steps[axis])));
}

/**
 * Copy out the requested window of a field. Returns the field itself when the
 * window covers all of it (no copy). Ranks 1-3 are unrolled; rank 4+ throws.
 */
export function windowField(field: NpyN, window: FieldWindow | undefined): NpyN {
  const shape = field.shape;
  const { starts, stops, steps } = resolveWindow(shape, window);
  const outShape = shape.map((_, axis) =>
    Math.max(0, Math.ceil((stops[axis] - starts[axis]) / steps[axis])));

  const untouched = shape.every((extent, axis) =>
    starts[axis] === 0 && stops[axis] === extent && steps[axis] === 1);
  if (untouched) return field;

  const total = outShape.reduce((product, extent) => product * extent, 1);
  const data = new Float32Array(total);
  const source = field.data;

  if (shape.length === 3) {
    const [, n1, n2] = shape;
    let out = 0;
    for (let i = starts[0]; i < stops[0]; i += steps[0]) {
      const planeBase = i * n1 * n2;
      for (let j = starts[1]; j < stops[1]; j += steps[1]) {
        const rowBase = planeBase + j * n2;
        for (let k = starts[2]; k < stops[2]; k += steps[2]) {
          data[out] = source[rowBase + k];
          out += 1;
        }
      }
    }
  } else if (shape.length === 2) {
    const [, n1] = shape;
    let out = 0;
    for (let i = starts[0]; i < stops[0]; i += steps[0]) {
      const rowBase = i * n1;
      for (let j = starts[1]; j < stops[1]; j += steps[1]) {
        data[out] = source[rowBase + j];
        out += 1;
      }
    }
  } else if (shape.length === 1) {
    let out = 0;
    for (let i = starts[0]; i < stops[0]; i += steps[0]) {
      data[out] = source[i];
      out += 1;
    }
  } else {
    // Nothing here produces rank 0 or rank 4+. Refusing loudly beats returning
    // a silently unwindowed field that the caller will treat as reduced.
    throw new Error(`Cannot window a rank-${shape.length} field`);
  }

  return { shape: outShape, data, descriptor: field.descriptor };
}

export interface SubsampledWindowAssembly {
  field: NpyN;
  addTile: (tile: NpyN, request: FieldLoadWindow) => void;
}

/** Plan native point-sample tiles for Python's crop-then-stride semantics.
 * Tile dimensions are measured in returned samples. The NPY reader gathers
 * one bounded contiguous fastest-axis span per selected prefix; netCDF uses a
 * native strided hyperslab. Both output and actual NPY row traffic are budgeted. */
export function planPythonFieldTiles(
  shape: readonly number[],
  window: FieldWindow | undefined,
  maxTileBytes = 16 * 1024 * 1024,
  outputBytesPerCell = Float32Array.BYTES_PER_ELEMENT,
  aggregation: FieldWindowAggregation = "point_sample",
): { windows: FieldLoadWindow[]; outputShape: number[]; outputBytes: number } {
  if (shape.length < 1 || shape.length > 3
      || shape.some((count) => !Number.isSafeInteger(count) || count <= 0)) {
    throw new Error(`Cannot plan a rank-${shape.length} Python field window`);
  }
  const resolved = resolveWindow(shape, window);
  const extent = resolved.stops.map((stop, axis) => stop - resolved.starts[axis]);
  if (extent.some((count) => count <= 0)) throw new Error("Python field window is empty");
  const outputShape = windowedShape(shape, window);
  const pointSample = aggregation === "point_sample"
    && resolved.steps.some((step) => step > 1);
  const blockAggregate = aggregation !== "point_sample";
  const geometryBytesPerCell = Math.max(
    0, outputBytesPerCell - Float32Array.BYTES_PER_ELEMENT,
  );
  const outputBytes = nativeFieldWindowBodyBytes(
    checkedProduct(outputShape, "Python field output shape"), geometryBytesPerCell,
  );
  const outputBudget = Math.min(maxTileBytes, MAX_NATIVE_FIELD_WINDOW_BYTES);
  const sourceBudget = Math.min(maxTileBytes, MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES);
  if (!Number.isSafeInteger(outputBytesPerCell) || outputBytesPerCell < Float32Array.BYTES_PER_ELEMENT) {
    throw new Error("Python field output bytes per cell must hold at least one float32 sample");
  }
  if (!Number.isSafeInteger(outputBudget)
      || outputBudget < nativeFieldWindowBodyBytes(1, geometryBytesPerCell)) {
    throw new Error("Python field tile budget must hold at least one output cell");
  }
  const cellCeiling = Math.max(
    1,
    Math.floor((outputBudget - (geometryBytesPerCell > 0 ? 4 : 0)) / outputBytesPerCell),
  );
  const tile = [...outputShape];
  const pointSourceBytes = (counts: readonly number[]): number => {
    const lastAxis = counts.length - 1;
    const prefixCount = checkedProduct(
      counts.slice(0, lastAxis),
      "Python point-sample prefix count",
    );
    const rowSpan = (counts[lastAxis] - 1) * resolved.steps[lastAxis] + 1;
    return checkedProduct(
      [prefixCount, rowSpan, Float32Array.BYTES_PER_ELEMENT],
      "Python point-sample source span",
    );
  };
  while (
    tile.reduce((product, count) => product * count, 1) > cellCeiling
    || (pointSample && pointSourceBytes(tile) > sourceBudget)
  ) {
    const contributions = pointSample
      ? tile.map((count, axis) => axis === tile.length - 1
        ? (count - 1) * resolved.steps[axis] + 1
        : count)
      : tile;
    const axis = contributions.indexOf(Math.max(...contributions));
    if (tile[axis] <= 1) {
      throw new Error("Python field tile source span cannot fit the native read budget");
    }
    tile[axis] = Math.max(1, Math.ceil(tile[axis] / 2));
  }
  enforceTileCount(outputShape, tile);
  const windows: FieldLoadWindow[] = [];
  const push = (outputOrigin: number[]): void => {
    const outputExtent = outputOrigin.map((start, axis) => Math.min(
      tile[axis],
      outputShape[axis] - start,
    ));
    const sourceOrigin = outputOrigin.map((start, axis) =>
      resolved.starts[axis] + start * resolved.steps[axis]
    );
    const sourceExtent = outputExtent.map((count, axis) => blockAggregate
      ? Math.min(
        count * resolved.steps[axis],
        resolved.stops[axis] - sourceOrigin[axis],
      )
      : (count - 1) * resolved.steps[axis] + 1);
    const bytes = nativeFieldWindowBodyBytes(
      checkedProduct(outputExtent, "Python field tile shape"), geometryBytesPerCell,
    );
    const sourceBytes = outputBytesPerCell > Float32Array.BYTES_PER_ELEMENT
      ? MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
      : pointSample
      // NPY is float32. Classic netCDF may store the same values in an
      // eight-byte numeric type, so reserve the raw-file worst case too.
      ? Math.min(
        pointSourceBytes(outputExtent) * 2,
        MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
      )
      : bytes;
    windows.push({
      // Unit-stride point requests are one-cell means. That form works on all
      // storage backends while remaining numerically identical to native data.
      aggregation: aggregation === "point_sample" && !pointSample ? "cell_mean" : aggregation,
      origin: sourceOrigin,
      extent: sourceExtent,
      stride: [...resolved.steps],
      maxOutputBytes: bytes,
      maxSourceBytes: sourceBytes,
    });
  };
  if (shape.length === 1) {
    for (let i = 0; i < outputShape[0]; i += tile[0]) push([i]);
  } else if (shape.length === 2) {
    for (let i = 0; i < outputShape[0]; i += tile[0]) {
      for (let j = 0; j < outputShape[1]; j += tile[1]) push([i, j]);
    }
  } else {
    for (let i = 0; i < outputShape[0]; i += tile[0]) {
      for (let j = 0; j < outputShape[1]; j += tile[1]) {
        for (let k = 0; k < outputShape[2]; k += tile[2]) push([i, j, k]);
      }
    }
  }
  return { windows, outputShape, outputBytes };
}

/** Compatibility path for storage backends that cannot seek individual cells
 * (compressed ZFP bricks and terrain geometry). Tiles stay bounded in dense
 * source space; assembly then keeps only the caller's requested sample points. */
export function planPythonDenseFieldTiles(
  shape: readonly number[],
  window: FieldWindow | undefined,
  maxTileBytes = 16 * 1024 * 1024,
  outputBytesPerCell = Float32Array.BYTES_PER_ELEMENT,
): { windows: FieldLoadWindow[]; outputShape: number[]; outputBytes: number } {
  if (shape.length < 1 || shape.length > 3
      || shape.some((count) => !Number.isSafeInteger(count) || count <= 0)) {
    throw new Error(`Cannot plan a rank-${shape.length} dense Python field window`);
  }
  const resolved = resolveWindow(shape, window);
  const denseShape = resolved.stops.map((stop, axis) => stop - resolved.starts[axis]);
  if (denseShape.some((count) => count <= 0)) throw new Error("Python field window is empty");
  const denseSourceBytes = checkedProduct(
    [...denseShape, outputBytesPerCell],
    "dense Python source window",
  );
  if (denseSourceBytes > MAX_DENSE_PYTHON_SOURCE_BYTES) {
    throw new Error(
      `This storage format would need to scan ${(denseSourceBytes / (1024 ** 3)).toFixed(1)} GiB `
        + "to serve the requested sparse Python field. Use a tighter box, or an uncompressed NPY/netCDF source.",
    );
  }
  const outputShape = windowedShape(shape, window);
  const geometryBytesPerCell = Math.max(
    0, outputBytesPerCell - Float32Array.BYTES_PER_ELEMENT,
  );
  const outputBytes = nativeFieldWindowBodyBytes(
    checkedProduct(outputShape, "dense Python field output shape"), geometryBytesPerCell,
  );
  const tileBudget = Math.min(maxTileBytes, MAX_NATIVE_FIELD_WINDOW_BYTES);
  if (!Number.isSafeInteger(outputBytesPerCell) || outputBytesPerCell < Float32Array.BYTES_PER_ELEMENT) {
    throw new Error("Dense Python field output bytes per cell must hold at least one float32 sample");
  }
  if (!Number.isSafeInteger(tileBudget)
      || tileBudget < nativeFieldWindowBodyBytes(1, geometryBytesPerCell)) {
    throw new Error("Dense Python field tile budget must hold at least one output cell");
  }
  const cellCeiling = Math.max(
    1,
    Math.floor((tileBudget - (geometryBytesPerCell > 0 ? 4 : 0)) / outputBytesPerCell),
  );
  const tile = [...denseShape];
  while (tile.reduce((product, count) => product * count, 1) > cellCeiling) {
    const axis = tile.indexOf(Math.max(...tile));
    tile[axis] = Math.max(1, Math.ceil(tile[axis] / 2));
  }
  enforceTileCount(denseShape, tile);
  const windows: FieldLoadWindow[] = [];
  const push = (relativeOrigin: number[]): void => {
    const extent = relativeOrigin.map((start, axis) =>
      Math.min(tile[axis], denseShape[axis] - start));
    const bytes = nativeFieldWindowBodyBytes(
      checkedProduct(extent, "dense Python field tile shape"), geometryBytesPerCell,
    );
    windows.push({
      aggregation: "cell_mean",
      origin: relativeOrigin.map((start, axis) => resolved.starts[axis] + start),
      extent,
      stride: shape.map(() => 1),
      maxOutputBytes: bytes,
      // A compressed brick or terrain geometry tile has bounded working data
      // beyond its float output. The native reader still enforces this ceiling.
      maxSourceBytes: MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
    });
  };
  if (shape.length === 1) {
    for (let i = 0; i < denseShape[0]; i += tile[0]) push([i]);
  } else if (shape.length === 2) {
    for (let i = 0; i < denseShape[0]; i += tile[0]) {
      for (let j = 0; j < denseShape[1]; j += tile[1]) push([i, j]);
    }
  } else {
    for (let i = 0; i < denseShape[0]; i += tile[0]) {
      for (let j = 0; j < denseShape[1]; j += tile[1]) {
        for (let k = 0; k < denseShape[2]; k += tile[2]) push([i, j, k]);
      }
    }
  }
  return { windows, outputShape, outputBytes };
}

/** Scatter exact transport tiles into Python's requested point-subsample. */
export function createSubsampledWindowAssembly(
  sourceShape: readonly number[],
  window: FieldWindow | undefined,
): SubsampledWindowAssembly {
  const resolved = resolveWindow(sourceShape, window);
  const outputShape = windowedShape(sourceShape, window);
  const outputElements = checkedProduct(outputShape, "Python field assembly shape");
  const field: NpyN = { shape: outputShape, data: new Float32Array(outputElements) };
  let geometryMode: boolean | null = null;
  let provenanceMode: boolean | null = null;
  let centroids: NonNullable<NonNullable<NpyN["windowGeometry"]>["centroidsM"]> | null = null;
  let volumes: Float64Array | null = null;
  const addTile = (chunk: NpyN, request: FieldLoadWindow): void => {
    const expectedShape = request.extent.map((count, axis) =>
      Math.floor((count - 1) / request.stride[axis]) + 1
    );
    if (chunk.shape.join("x") !== expectedShape.join("x")) {
      throw new Error("Python transport tile shape changed during read");
    }
    const chunkGeometry = chunk.windowGeometry;
    const carriesGeometry = chunkGeometry?.centroidsM != null
      && chunkGeometry.volumesM3 != null
      && chunkGeometry.centroidsM.x.length === chunk.data.length
      && chunkGeometry.centroidsM.y.length === chunk.data.length
      && chunkGeometry.centroidsM.z.length === chunk.data.length
      && chunkGeometry.volumesM3.length === chunk.data.length;
    const carriesProvenance = chunkGeometry != null;
    if (provenanceMode == null) {
      provenanceMode = carriesProvenance;
      if (carriesProvenance) {
        field.windowGeometry = {
          aggregation: chunkGeometry.aggregation ?? request.aggregation,
          weighting: chunkGeometry.weighting,
          sampleLocation: chunkGeometry.sampleLocation,
          centroidsM: null,
          volumesM3: null,
          sourceShape: [...sourceShape],
          origin: [...resolved.starts],
          extent: resolved.stops.map((stop, axis) => stop - resolved.starts[axis]),
          stride: [...resolved.steps],
          sourceFormat: chunkGeometry.sourceFormat,
        };
      }
    } else if (provenanceMode !== carriesProvenance) {
      throw new Error("Python field tiles disagree on read provenance");
    } else if (carriesProvenance && field.windowGeometry
        && (field.windowGeometry.aggregation !== (chunkGeometry.aggregation ?? request.aggregation)
          || field.windowGeometry.weighting !== chunkGeometry.weighting
          || field.windowGeometry.sampleLocation !== chunkGeometry.sampleLocation
          || field.windowGeometry.sourceFormat !== chunkGeometry.sourceFormat)) {
      throw new Error("Python field tiles disagree on sampling or storage format");
    }
    if (geometryMode == null) {
      geometryMode = carriesGeometry;
      if (carriesGeometry) {
        centroids = {
          x: new Float64Array(outputElements),
          y: new Float64Array(outputElements),
          z: new Float64Array(outputElements),
        };
        volumes = new Float64Array(outputElements);
        field.windowGeometry = {
          aggregation: request.aggregation,
          weighting: "physical_cell_volume",
          sampleLocation: request.aggregation === "point_sample"
            ? "source_cell_center"
            : "physical_volume_centroid",
          centroidsM: centroids,
          volumesM3: volumes,
          sourceShape: [...sourceShape],
          origin: [...resolved.starts],
          extent: resolved.stops.map((stop, axis) => stop - resolved.starts[axis]),
          stride: [...resolved.steps],
          sourceFormat: chunkGeometry.sourceFormat,
        };
      }
    } else if (geometryMode !== carriesGeometry) {
      throw new Error("Python terrain tiles disagree on physical geometry");
    }
    for (let linear = 0; linear < chunk.data.length; linear += 1) {
      let remainder = linear;
      const local = new Array<number>(sourceShape.length);
      for (let axis = sourceShape.length - 1; axis >= 0; axis -= 1) {
        local[axis] = remainder % chunk.shape[axis];
        remainder = Math.floor(remainder / chunk.shape[axis]);
      }
      const global = local.map((index, axis) =>
        request.origin[axis] + index * request.stride[axis]
      );
      if (global.some((index, axis) =>
        index < resolved.starts[axis] || index >= resolved.stops[axis]
        || (index - resolved.starts[axis]) % resolved.steps[axis] !== 0
      )) continue;
      const target = global.map((index, axis) =>
        (index - resolved.starts[axis]) / resolved.steps[axis]
      ).reduce((index, value, axis) => index * outputShape[axis] + value, 0);
      field.data[target] = chunk.data[linear];
      if (centroids && volumes && chunkGeometry?.centroidsM && chunkGeometry.volumesM3) {
        centroids.x[target] = chunkGeometry.centroidsM.x[linear];
        centroids.y[target] = chunkGeometry.centroidsM.y[linear];
        centroids.z[target] = chunkGeometry.centroidsM.z[linear];
        volumes[target] = chunkGeometry.volumesM3[linear];
      }
    }
  };
  return { field, addTile };
}

/** Uniform stride for horizontal axes that brings the field under `budgetBytes`. Vertical axis is not thinned. */
export function strideForBudget(
  shape: readonly number[],
  budgetBytes: number,
  bytesPerCell = 4,
): number {
  const cells = shape.reduce((product, extent) => product * Math.max(1, extent), 1);
  if (cells * bytesPerCell <= budgetBytes || budgetBytes <= 0) return 1;
  const horizontal = shape.length >= 2 ? 2 : 1;
  // Each step of the stride divides the count by stride^horizontal.
  const needed = (cells * bytesPerCell) / budgetBytes;
  return Math.max(1, Math.ceil(needed ** (1 / horizontal)));
}
