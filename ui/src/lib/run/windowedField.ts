import type { NpyN, FieldWindowGeometry } from "@/lib/platform/npy";
import type { FieldLoadWindow } from "./runOutputs";

function product(values: readonly number[]): number {
  return values.reduce((total, value) => total * value, 1);
}

function outputShape(window: FieldLoadWindow): number[] {
  return window.extent.map((extent, axis) =>
    Math.floor((extent - 1) / window.stride[axis]) + 1
  );
}

/** Reduce one exact derived-field tile after evaluation. This ordering is
 * essential for nonlinear diagnostics: derive(raw) then mean, never
 * derive(mean(raw dependencies)). */
export function aggregateDerivedWindow(
  field: NpyN,
  window: FieldLoadWindow,
  sourceShape: readonly number[],
): NpyN {
  if (field.shape.join("x") !== window.extent.join("x")) {
    throw new Error("Derived raw tile shape disagrees with its aggregation window");
  }
  const shape = outputShape(window);
  const elements = product(shape);
  const sums = new Float64Array(elements);
  const weights = new Float64Array(elements);
  const counts = new Uint32Array(elements);
  const invalid = new Uint8Array(elements);
  const sourceCentroids = field.windowGeometry?.centroidsM ?? null;
  const sourceVolumes = field.windowGeometry?.volumesM3 ?? null;
  const terrainGeometry = sourceCentroids != null
    && sourceVolumes != null
    && sourceVolumes.length === field.data.length;
  const xMoments = terrainGeometry ? new Float64Array(elements) : null;
  const yMoments = terrainGeometry ? new Float64Array(elements) : null;
  const zMoments = terrainGeometry ? new Float64Array(elements) : null;
  for (let source = 0; source < field.data.length; source += 1) {
    let remainder = source;
    const targetIndices = new Array<number>(field.shape.length);
    let selectedPoint = true;
    for (let axis = field.shape.length - 1; axis >= 0; axis -= 1) {
      const index = remainder % field.shape[axis];
      remainder = Math.floor(remainder / field.shape[axis]);
      if (window.aggregation === "point_sample" && index % window.stride[axis] !== 0) {
        selectedPoint = false;
      }
      targetIndices[axis] = Math.floor(index / window.stride[axis]);
    }
    if (!selectedPoint) continue;
    const target = targetIndices.reduce(
      (linear, index, axis) => linear * shape[axis] + index,
      0,
    );
    const value = field.data[source];
    const physicalVolume = terrainGeometry ? sourceVolumes[source] : 1;
    if (!Number.isFinite(physicalVolume) || physicalVolume <= 0) {
      invalid[target] = 1;
      continue;
    }
    weights[target] += physicalVolume;
    if (terrainGeometry) {
      xMoments![target] += physicalVolume * sourceCentroids.x[source];
      yMoments![target] += physicalVolume * sourceCentroids.y[source];
      zMoments![target] += physicalVolume * sourceCentroids.z[source];
    }
    if (!Number.isFinite(value)) {
      invalid[target] = 1;
    } else {
      switch (window.aggregation) {
        case "block_max":
          if (counts[target] === 0 || value > sums[target]) sums[target] = value;
          break;
        case "block_min":
          if (counts[target] === 0 || value < sums[target]) sums[target] = value;
          break;
        case "magnitude_extreme":
          if (counts[target] === 0 || Math.abs(value) > Math.abs(sums[target])) {
            sums[target] = value;
          }
          break;
        case "linear_reflectivity_mean":
          sums[target] += physicalVolume * 10 ** (value / 10);
          break;
        case "cell_mean":
          sums[target] += physicalVolume * value;
          break;
        case "point_sample":
          sums[target] = value;
          break;
      }
      counts[target] += 1;
    }
  }
  const data = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) {
    if (invalid[index] || counts[index] === 0) {
      data[index] = Number.NaN;
      continue;
    }
    if (["block_max", "block_min", "magnitude_extreme", "point_sample"].includes(window.aggregation)) {
      data[index] = sums[index];
      continue;
    }
    const mean = sums[index] / weights[index];
    data[index] = window.aggregation === "linear_reflectivity_mean"
      ? 10 * Math.log10(mean)
      : mean;
  }
  const centroidsM = terrainGeometry ? {
    x: Float64Array.from(weights, (weight, index) => xMoments![index] / weight),
    y: Float64Array.from(weights, (weight, index) => yMoments![index] / weight),
    z: Float64Array.from(weights, (weight, index) => zMoments![index] / weight),
  } : null;
  return {
    shape,
    data,
    windowGeometry: {
      aggregation: window.aggregation,
      weighting: terrainGeometry ? "physical_cell_volume" : "uniform_cell_volume",
      sampleLocation: window.aggregation === "point_sample"
        ? "source_cell_center"
        : terrainGeometry ? "physical_volume_centroid" : "block_center",
      centroidsM,
      volumesM3: terrainGeometry ? weights : null,
      sourceShape: [...sourceShape],
      origin: [...window.origin],
    extent: [...window.extent],
    stride: [...window.stride],
    sourceFormat: field.windowGeometry?.sourceFormat,
    },
  };
}

/** Assemble block-aligned field-window tiles into one C-order lattice. */
export function assembleWindowedField(
  chunks: readonly NpyN[],
  windows: readonly FieldLoadWindow[],
): NpyN {
  if (chunks.length === 0 || chunks.length !== windows.length) {
    throw new Error("Window assembly requires one response per planned tile");
  }
  if (chunks.length === 1) return chunks[0];
  const rank = chunks[0].shape.length;
  if (rank < 1 || rank > 3 || windows.some((window) => window.origin.length !== rank)) {
    throw new Error("Window tiles disagree on rank");
  }
  const stride = windows[0].stride;
  if (windows.some((window) => window.stride.some((value, axis) => value !== stride[axis]))) {
    throw new Error("Window tiles disagree on aggregation stride");
  }
  const aggregation = windows[0].aggregation;
  if (windows.some((window) => window.aggregation !== aggregation)) {
    throw new Error("Window tiles disagree on aggregation method");
  }
  const overallOrigin = Array.from({ length: rank }, (_, axis) =>
    Math.min(...windows.map((window) => window.origin[axis]))
  );
  const overallEnd = Array.from({ length: rank }, (_, axis) =>
    Math.max(...windows.map((window) => window.origin[axis] + window.extent[axis]))
  );
  const overallExtent = overallEnd.map((end, axis) => end - overallOrigin[axis]);
  const shape = overallExtent.map((extent, axis) =>
    Math.floor((extent - 1) / stride[axis]) + 1
  );
  const data = new Float32Array(product(shape));
  const carriesTerrainGeometry = chunks.every((chunk) =>
    chunk.windowGeometry?.centroidsM != null
      && chunk.windowGeometry.volumesM3 != null
      && chunk.windowGeometry.volumesM3.length === chunk.data.length
  );
  const assembledCentroids = carriesTerrainGeometry ? {
    x: new Float64Array(data.length),
    y: new Float64Array(data.length),
    z: new Float64Array(data.length),
  } : null;
  const assembledVolumes = carriesTerrainGeometry ? new Float64Array(data.length) : null;

  const copyChunk = (chunk: NpyN, window: FieldLoadWindow): void => {
    const expected = outputShape(window);
    if (chunk.shape.some((count, axis) => count !== expected[axis])) {
      throw new Error("Window response shape disagrees with its planned tile");
    }
    const offset = window.origin.map((value, axis) => {
      const delta = value - overallOrigin[axis];
      if (delta % stride[axis] !== 0) throw new Error("Window tile is not block-aligned");
      return delta / stride[axis];
    });
    const n0 = chunk.shape[0];
    const n1 = chunk.shape[1] ?? 1;
    const n2 = chunk.shape[2] ?? 1;
    const outN1 = shape[1] ?? 1;
    const outN2 = shape[2] ?? 1;
    for (let i = 0; i < n0; i += 1) {
      for (let j = 0; j < n1; j += 1) {
        const source = (i * n1 + j) * n2;
        const target = ((i + offset[0]) * outN1 + (j + (offset[1] ?? 0))) * outN2
          + (offset[2] ?? 0);
        data.set(chunk.data.subarray(source, source + n2), target);
        if (assembledCentroids && assembledVolumes) {
          const geometry = chunk.windowGeometry!;
          assembledCentroids.x.set(geometry.centroidsM!.x.subarray(source, source + n2), target);
          assembledCentroids.y.set(geometry.centroidsM!.y.subarray(source, source + n2), target);
          assembledCentroids.z.set(geometry.centroidsM!.z.subarray(source, source + n2), target);
          assembledVolumes.set(geometry.volumesM3!.subarray(source, source + n2), target);
        }
      }
    }
  };
  chunks.forEach((chunk, index) => copyChunk(chunk, windows[index]));

  const firstGeometry = chunks[0].windowGeometry;
  const geometry: FieldWindowGeometry | undefined = firstGeometry ? {
    aggregation: firstGeometry.aggregation ?? aggregation,
    weighting: firstGeometry.weighting,
    sampleLocation: firstGeometry.sampleLocation,
    centroidsM: assembledCentroids,
    volumesM3: assembledVolumes,
    sourceShape: firstGeometry.sourceShape ? [...firstGeometry.sourceShape] : undefined,
    origin: overallOrigin,
    extent: overallExtent,
    stride: [...stride],
    sourceFormat: firstGeometry.sourceFormat,
  } : undefined;
  return { shape, data, ...(geometry ? { windowGeometry: geometry } : {}) };
}
