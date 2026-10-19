import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import vtkTubeFilter from "@kitware/vtk.js/Filters/General/TubeFilter.js";
import { releaseVtkPointDataArrays, VTK_SCALAR_SLOT } from "./vtkArrayOwnership";
import {
  TRAJECTORY_TUBE_SIDES,
  trajectoryGeometryBuffers,
  type TrajectoryDataset,
  type TrajectoryGeometryParameters,
  type TrajectoryGeometryResult,
  type TrajectoryTubeGeometry,
} from "./trajectoryGeometryProtocol";

function validatedShape(dataset: TrajectoryDataset): [number, number, number] {
  const [parcels, samples, variables] = dataset.shape;
  if (dataset.shape.length !== 3
      || !Number.isSafeInteger(parcels) || parcels < 0
      || !Number.isSafeInteger(samples) || samples < 0
      || !Number.isSafeInteger(variables) || variables < 6
      || parcels * samples * variables !== dataset.values.length) {
    throw new Error("Trajectory data has an invalid shape");
  }
  return [parcels, samples, variables];
}

function validatedParameters(
  parameters: TrajectoryGeometryParameters,
  variables: number,
): void {
  if (!Number.isSafeInteger(parameters.colorColumn)
      || parameters.colorColumn < 0 || parameters.colorColumn >= variables) {
    throw new Error("Trajectory colour column is outside the sample record");
  }
  if (!(Number.isFinite(parameters.tubeRadius) && parameters.tubeRadius > 0)) {
    throw new Error("Trajectory tube radius must be positive");
  }
  if (!(Number.isFinite(parameters.scale.xy) && parameters.scale.xy > 0
      && Number.isFinite(parameters.scale.z) && parameters.scale.z > 0)) {
    throw new Error("Trajectory display scale must be positive");
  }
  if (!Number.isSafeInteger(parameters.maximumParcels) || parameters.maximumParcels < 1) {
    throw new Error("Trajectory parcel limit must be a positive safe integer");
  }
}

function ownedTubeGeometry(
  output: ReturnType<typeof vtkPolyData.newInstance>,
): TrajectoryTubeGeometry | null {
  const floats = (values: ArrayLike<number>): Float32Array => values instanceof Float32Array
      && values.byteOffset === 0 && values.byteLength === values.buffer.byteLength
    ? values : Float32Array.from(values);
  const cells = (values: ArrayLike<number>): Uint32Array => values instanceof Uint32Array
      && values.byteOffset === 0 && values.byteLength === values.buffer.byteLength
    ? values : Uint32Array.from(values);
  const points = floats(output.getPoints().getData() ?? []);
  const lines = cells(output.getLines().getData() ?? []);
  const polys = cells(output.getPolys().getData() ?? []);
  const strips = cells(output.getStrips().getData() ?? []);
  if (points.length === 0
      || (lines.length === 0 && polys.length === 0 && strips.length === 0)) return null;
  const normals = floats(output.getPointData().getNormals()?.getData() ?? []);
  const scalarArray = output.getPointData().getArrayByName(VTK_SCALAR_SLOT);
  const scalars = floats(scalarArray?.getData() ?? []);
  if (scalars.length !== points.length / 3) {
    throw new Error("Trajectory tube output lost its colour scalars");
  }
  return { points, lines, polys, strips, normals, scalars };
}

export function buildTrajectoryTubeGeometry(
  dataset: TrajectoryDataset,
  parameters: TrajectoryGeometryParameters,
): TrajectoryGeometryResult {
  const startedAt = performance.now();
  const [parcelTotal, samplesPerParcel, variables] = validatedShape(dataset);
  validatedParameters(parameters, variables);
  const stride = parcelTotal > parameters.maximumParcels
    ? Math.ceil(parcelTotal / parameters.maximumParcels)
    : 1;
  const values = dataset.values;
  let minimumTime = Infinity, maximumTime = -Infinity;
  const visitSegments = (parcel: number, visit: (start: number, length: number, minimum: number, maximum: number) => void) => {
    let start = 0;
    let length = 0;
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let sample = 0; sample <= samplesPerParcel; sample++) {
      const base = (parcel * samplesPerParcel + sample) * variables;
      if (sample < samplesPerParcel && Number.isFinite(values[base])) {
        minimumTime = Math.min(minimumTime, values[base]);
        maximumTime = Math.max(maximumTime, values[base]);
      }
      const color = values[base + parameters.colorColumn];
      if (sample < samplesPerParcel && Number.isFinite(values[base + 1])
          && Number.isFinite(values[base + 2]) && Number.isFinite(values[base + 3]) && Number.isFinite(color)) {
        if (length === 0) start = sample;
        length++;
        minimum = Math.min(minimum, color);
        maximum = Math.max(maximum, color);
      } else {
        if (length >= 2) visit(start, length, minimum, maximum);
        length = 0;
        minimum = Infinity;
        maximum = -Infinity;
      }
    }
  };
  const selectedParcelCount = parcelTotal === 0 ? 0 : Math.ceil(parcelTotal / stride);
  const counts = new Uint32Array(selectedParcelCount);
  let pointCount = 0;
  let lineValueCount = 0;
  let parcels = 0;
  let min = Infinity;
  let max = -Infinity;

  let selectedParcelIndex = 0;
  for (let parcel = 0; parcel < parcelTotal; parcel += stride, selectedParcelIndex++) {
    let valid = 0;
    visitSegments(parcel, (_start, length, minimum, maximum) => {
      valid += length;
      pointCount += length;
      lineValueCount += length + 1;
      min = Math.min(min, minimum);
      max = Math.max(max, maximum);
    });
    counts[selectedParcelIndex] = valid;
    if (valid >= 2) {
      parcels += 1;
    }
  }

  if (parcels === 0) {
    return {
      geometry: null,
      min: null,
      max: null,
      parcels: 0,
      sourceSamples: 0,
      retainedBytes: 0,
      knownScratchBytes: counts.byteLength,
      elapsedMs: performance.now() - startedAt,
      timeRangeS: minimumTime <= maximumTime ? [minimumTime, maximumTime] : null,
    };
  }

  const points = new Float32Array(pointCount * 3);
  const scalars = new Float32Array(pointCount);
  const lines = new Uint32Array(lineValueCount);
  let pointCursor = 0;
  let lineCursor = 0;
  let selectedParcel = 0;
  for (let parcel = 0; parcel < parcelTotal; parcel += stride, selectedParcel++) {
    const valid = counts[selectedParcel]!;
    if (valid < 2) continue;
    visitSegments(parcel, (start, length) => {
      lines[lineCursor++] = length;
      for (let sample = start; sample < start + length; sample++) {
      const base = (parcel * samplesPerParcel + sample) * variables;
      const x = values[base + 1];
      const y = values[base + 2];
      const z = values[base + 3];
      const color = values[base + parameters.colorColumn];
      const pointOffset = pointCursor * 3;
      points[pointOffset] = x * parameters.scale.xy;
      points[pointOffset + 1] = y * parameters.scale.xy;
      points[pointOffset + 2] = z * parameters.scale.z;
      scalars[pointCursor] = color;
      lines[lineCursor++] = pointCursor++;
      }
    });
  }

  const paths = vtkPolyData.newInstance();
  paths.getPoints().setData(points, 3);
  paths.getLines().setData(lines);
  paths.getPointData().setScalars(vtkDataArray.newInstance({
    name: VTK_SCALAR_SLOT,
    values: scalars,
    numberOfComponents: 1,
  }));
  paths.modified();
  const tube = vtkTubeFilter.newInstance({
    radius: parameters.tubeRadius,
    numberOfSides: TRAJECTORY_TUBE_SIDES,
    capping: true,
  });
  tube.setInputData(paths);
  const output = tube.getOutputData();
  const geometry = ownedTubeGeometry(output);
  const retainedBuffers = new Set(trajectoryGeometryBuffers(geometry));
  const retainedBytes = [...retainedBuffers]
    .reduce((sum, buffer) => sum + buffer.byteLength, 0);
  const internalOutputBuffers = new Set<ArrayBufferLike>();
  const add = (view: unknown) => {
    if (ArrayBuffer.isView(view)) internalOutputBuffers.add(view.buffer);
  };
  add(output.getPoints().getData());
  add(output.getLines().getData());
  add(output.getPolys().getData());
  add(output.getStrips().getData());
  for (const array of output.getPointData().getArrays()) add(array.getData());
  for (const array of output.getCellData().getArrays()) add(array.getData());
  const knownScratchBytes = points.byteLength + lines.byteLength + scalars.byteLength
    + [...internalOutputBuffers].filter(buffer => !retainedBuffers.has(buffer)).reduce((sum, buffer) => sum + buffer.byteLength, 0)
    + counts.byteLength;
  releaseVtkPointDataArrays(output.getPointData());
  releaseVtkPointDataArrays(paths.getPointData());
  tube.delete();
  paths.delete();
  return {
    geometry,
    min,
    max,
    parcels,
    sourceSamples: pointCount,
    retainedBytes,
    knownScratchBytes,
    elapsedMs: performance.now() - startedAt,
    timeRangeS: minimumTime <= maximumTime ? [minimumTime, maximumTime] : null,
  };
}
