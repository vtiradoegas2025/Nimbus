import { finiteImageMarchingCubes } from "./finiteImageMarchingCubes";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
import { prepareVtkRequest } from "./vtkPreparation";
import { contourTerrainBrick } from "./vtkTerrainContour";
import type { BrickedContourWorkerReply, BrickedContourWorkerRequest } from "./brickedContourProtocol";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";
import { terrainVelocityGradientDiagnostic } from "./vtkTerrainVelocityDiagnostic";

const scope = self as unknown as {
  onmessage: ((event: { data: BrickedContourWorkerRequest }) => void) | null;
  postMessage: (reply: BrickedContourWorkerReply, transfer?: ArrayBuffer[]) => void;
  close: () => void;
};

let datasetIdentity = "";
type ResidentBrick =
  | { kind: "image"; image: ReturnType<typeof vtkImageData.newInstance> }
  | {
      kind: "terrain";
      shape: [number, number, number];
      values: Float32Array;
      centroidsM: { x: Float32Array; y: Float32Array; z: Float32Array };
      cellBounds: [number, number, number, number, number, number];
    };
const bricks = new Map<string, ResidentBrick>();

function clear(): void {
  for (const brick of bricks.values()) {
    if (brick.kind === "image") brick.image.delete();
  }
  bricks.clear();
}

function finiteBounds(value: readonly number[]): [number, number, number, number, number, number] | null {
  return value.length === 6 && value.every(Number.isFinite)
    ? [value[0], value[1], value[2], value[3], value[4], value[5]]
    : null;
}

function geometryFrom(output: ReturnType<ReturnType<typeof vtkImageMarchingCubes.newInstance>["getOutputData"]>): VortexSurfaceGeometry | null {
  if (output.getNumberOfPolys() === 0) return null;
  const points = Float32Array.from(output.getPoints().getData() ?? []);
  const polys = Uint32Array.from(output.getPolys().getData() ?? []);
  const normals = Float32Array.from(output.getPointData().getNormals()?.getData() ?? []);
  return {
    points,
    polys,
    normals,
    bounds: finiteBounds(output.getBounds()),
    triangles: output.getNumberOfPolys(),
    bytes: points.byteLength + polys.byteLength + normals.byteLength,
  };
}

function scalarStats(values: Float32Array) {
  const finite = Array.from(values).filter(Number.isFinite).sort((left, right) => left - right);
  const nonFiniteCount = values.length - finite.length;
  if (finite.length === 0) {
    return { min: 0, max: 0, p2: 0, p98: 0, positiveP98: 0, sourceNonFiniteCount: nonFiniteCount };
  }
  const quantile = (source: readonly number[], fraction: number) =>
    source[Math.min(source.length - 1, Math.floor(fraction * (source.length - 1)))];
  const positive = finite.filter((value) => value > 0);
  return {
    min: finite[0],
    max: finite[finite.length - 1],
    p2: quantile(finite, 0.02),
    p98: quantile(finite, 0.98),
    positiveP98: positive.length > 0 ? quantile(positive, 0.98) : 0,
    sourceNonFiniteCount: nonFiniteCount,
  };
}

function load(request: Extract<BrickedContourWorkerRequest, { kind: "load" }>): void {
  if (datasetIdentity !== request.datasetIdentity) {
    clear();
    datasetIdentity = request.datasetIdentity;
  }
  if ((request.request.kind === "scalar" || request.request.kind === "velocity-diagnostic")
      && request.request.grid.terrainFollowing) {
    const field = request.request.kind === "velocity-diagnostic"
      ? terrainVelocityGradientDiagnostic(request.request.fields, request.request.diagnostic)
      : request.request.field;
    const geometry = field.windowGeometry;
    const centroids = geometry?.centroidsM;
    const volumes = geometry?.volumesM3;
    if (request.request.refineForDisplay
        || geometry?.weighting !== "physical_cell_volume"
        || geometry.sampleLocation !== "physical_volume_centroid"
        || !centroids || !volumes
        || centroids.x.length !== field.data.length
        || centroids.y.length !== field.data.length
        || centroids.z.length !== field.data.length
        || volumes.length !== field.data.length) {
      throw new Error("Terrain brick contour requires exact native physical-volume geometry");
    }
    const resident: ResidentBrick = {
      kind: "terrain",
      shape: [...field.shape],
      values: Float32Array.from(field.data),
      centroidsM: {
        x: Float32Array.from(centroids.x),
        y: Float32Array.from(centroids.y),
        z: Float32Array.from(centroids.z),
      },
      cellBounds: request.terrainCellBounds ?? [
        0, field.shape[0] - 1,
        0, field.shape[1] - 1,
        0, field.shape[2] - 1,
      ],
    };
    const previous = bricks.get(request.brickId);
    if (previous?.kind === "image") previous.image.delete();
    bricks.set(request.brickId, resident);
    scope.postMessage({
      kind: "loaded",
      requestId: request.requestId,
      datasetIdentity,
      brickId: request.brickId,
      ok: true,
      bytes: resident.values.byteLength + resident.centroidsM.x.byteLength
        + resident.centroidsM.y.byteLength + resident.centroidsM.z.byteLength,
      stats: scalarStats(resident.values),
    });
    return;
  }
  const prepared = prepareVtkRequest({ ...request.request, includeValidity: false });
  if (!("values" in prepared) || !prepared.stats) {
    throw new Error("Bricked contour preparation did not produce scalar diagnostic values");
  }
  const image = vtkImageData.newInstance();
  image.setDimensions(...prepared.shape);
  image.setSpacing(prepared.spacing);
  image.setOrigin(prepared.origin);
  image.getPointData().setScalars(vtkDataArray.newInstance({
    name: `__nimbus_brick_${request.brickId}`,
    values: prepared.values,
    numberOfComponents: 1,
  }));
  image.modified();
  const previous = bricks.get(request.brickId);
  if (previous?.kind === "image") previous.image.delete();
  bricks.set(request.brickId, { kind: "image", image });
  scope.postMessage({
    kind: "loaded",
    requestId: request.requestId,
    datasetIdentity,
    brickId: request.brickId,
    ok: true,
    bytes: prepared.values.byteLength,
    stats: { ...prepared.stats, sourceNonFiniteCount: prepared.sourceNonFiniteCount },
  });
}

function extract(request: Extract<BrickedContourWorkerRequest, { kind: "extract" }>): void {
  if (request.datasetIdentity !== datasetIdentity) throw new Error("Bricked contour dataset identity changed");
  const chunks: Array<{ brickId: string; geometry: VortexSurfaceGeometry | null }> = [];
  const transfer: ArrayBuffer[] = [];
  let knownScratchBytes = 0;
  for (const brickId of request.brickIds) {
    const brick = bricks.get(brickId);
    if (!brick) throw new Error(`Bricked contour worker is missing ${brickId}`);
    if (brick.kind === "terrain") {
      const geometry = contourTerrainBrick(
        brick.shape,
        brick.values,
        brick.centroidsM,
        request.threshold,
        { cellBounds: brick.cellBounds },
      );
      if (geometry) transfer.push(
        geometry.points.buffer as ArrayBuffer,
        geometry.polys.buffer as ArrayBuffer,
        geometry.normals.buffer as ArrayBuffer,
      );
      chunks.push({ brickId, geometry });
      continue;
    }
    const image = brick.image;
    const marching = finiteImageMarchingCubes({
      contourValue: request.threshold,
      computeNormals: true,
      mergePoints: true,
    });
    marching.setInputData(image);
    const output = marching.getOutputData();
    const geometry = geometryFrom(output);
    if (geometry) transfer.push(
      geometry.points.buffer as ArrayBuffer,
      geometry.polys.buffer as ArrayBuffer,
      geometry.normals.buffer as ArrayBuffer,
    );
    for (const array of output.getPointData().getArrays()) {
      const data = array.getData();
      if (ArrayBuffer.isView(data)
          && (!geometry || !transfer.includes(data.buffer as ArrayBuffer))) {
        knownScratchBytes += data.byteLength;
      }
    }
    chunks.push({ brickId, geometry });
    marching.delete();
  }
  scope.postMessage({
    kind: "result",
    requestId: request.requestId,
    datasetIdentity,
    ok: true,
    chunks,
    knownScratchBytes,
  }, transfer);
}

scope.onmessage = (event) => {
  const request = event.data;
  if (request.kind === "close") {
    clear();
    scope.close();
    return;
  }
  if (request.kind === "reset") {
    clear();
    datasetIdentity = request.datasetIdentity;
    return;
  }
  try {
    if (request.kind === "load") load(request);
    else extract(request);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (request.kind === "load") {
      scope.postMessage({
        kind: "loaded",
        requestId: request.requestId,
        datasetIdentity: request.datasetIdentity,
        brickId: request.brickId,
        ok: false,
        error: message,
      });
    } else {
      scope.postMessage({
        kind: "result",
        requestId: request.requestId,
        datasetIdentity: request.datasetIdentity,
        ok: false,
        error: message,
      });
    }
  }
};

scope.postMessage({ kind: "ready" });
