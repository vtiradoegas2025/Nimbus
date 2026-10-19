import { finiteImageStreamline } from "./finiteImageStreamline";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkTubeFilter from "@kitware/vtk.js/Filters/General/TubeFilter";
import { VTK_SCALAR_SLOT, VTK_VECTOR_SLOT } from "./vtkArrayOwnership";
import { scalePointTriples } from "./vtkScaledPolyData";
import { buildStreamlineArrowGeometry, sampleStreamlineMagnitudes } from "./streamlineDirection";
import {
  streamlineResultTransferables,
  type StreamlineDirectionGeometry,
  type StreamlineExtractionResult,
  type StreamlineGeometry,
  type StreamlineImage,
  type StreamlineWorkerReply,
  type StreamlineWorkerRequest,
} from "./streamlineExtractionProtocol";
import { prepareVtkRequest } from "./vtkPreparation";
import { transferablePreparedBuffers } from "./vtkDatasetHandoff";
import {
  detectStreamlineBoundaryContacts,
  emptyBrickBoundaryContacts,
  mergeBrickBoundaryContacts,
} from "./streamlineBoundary";
import type { BrickBoundaryContacts, Triple } from "@/lib/plot/fieldBrickHierarchy";

/** Above this, retain every integrated point but draw GPU line primitives. */
const TUBE_INPUT_POINT_CAP = 250_000;

const scope = self as unknown as {
  onmessage: ((event: { data: StreamlineWorkerRequest }) => void) | null;
  postMessage: (message: StreamlineWorkerReply, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

let imageData: ReturnType<typeof vtkImageData.newInstance> | null = null;
let activeDatasetId = -1;

function loadImage(image: StreamlineImage): void {
  imageData?.delete();
  imageData = vtkImageData.newInstance();
  imageData.setDimensions(...image.shape);
  imageData.setSpacing(image.spacing);
  imageData.setOrigin(image.origin);
  imageData.getPointData().setVectors(vtkDataArray.newInstance({
    name: VTK_VECTOR_SLOT,
    values: image.values,
    numberOfComponents: 3,
  }));
  imageData.modified();
}

function geometryFrom(polyData: ReturnType<typeof vtkPolyData.newInstance>): StreamlineGeometry | null {
  const points = Float32Array.from(polyData.getPoints().getData() ?? []);
  const lines = Uint32Array.from(polyData.getLines().getData() ?? []);
  const polys = Uint32Array.from(polyData.getPolys().getData() ?? []);
  const strips = Uint32Array.from(polyData.getStrips().getData() ?? []);
  if (points.length === 0 || (lines.length === 0 && polys.length === 0 && strips.length === 0)) return null;
  const normals = Float32Array.from(polyData.getPointData().getNormals()?.getData() ?? []);
  const sourceScalars = polyData.getPointData().getArrayByName(VTK_SCALAR_SLOT)?.getData();
  return {
    points,
    lines,
    polys,
    strips,
    normals,
    scalars: sourceScalars ? Float32Array.from(sourceScalars) : null,
  };
}

function geometryBuffers(geometry: StreamlineGeometry | null): Set<ArrayBufferLike> {
  const buffers = new Set<ArrayBufferLike>();
  if (!geometry) return buffers;
  for (const view of [
    geometry.points,
    geometry.lines,
    geometry.polys,
    geometry.strips,
    geometry.normals,
    geometry.scalars,
  ]) {
    if (view) buffers.add(view.buffer);
  }
  return buffers;
}

function addPolyDataBuffers(
  buffers: Set<ArrayBufferLike>,
  polyData: ReturnType<typeof vtkPolyData.newInstance> | null,
  retained: ReadonlySet<ArrayBufferLike>,
): void {
  if (!polyData) return;
  const add = (view: unknown) => {
    if (ArrayBuffer.isView(view) && !retained.has(view.buffer)) buffers.add(view.buffer);
  };
  add(polyData.getPoints().getData());
  add(polyData.getLines().getData());
  add(polyData.getPolys().getData());
  add(polyData.getStrips().getData());
  for (const array of polyData.getPointData().getArrays()) add(array.getData());
  for (const array of polyData.getCellData().getArrays()) add(array.getData());
}

function direction(
  request: Extract<StreamlineWorkerRequest, { kind: "extract" }>,
  sign: 1 | -1,
): {
  geometry: StreamlineDirectionGeometry;
  integratedPoints: number;
  usedTubes: boolean;
  knownScratchBytes: number;
  boundaryContacts: BrickBoundaryContacts;
} {
  const seeds = vtkPolyData.newInstance();
  seeds.getPoints().setData(request.seeds, 3);
  seeds.modified();
  const streamline = finiteImageStreamline({
    integrationStep: sign * request.integrationStep,
    maximumNumberOfSteps: request.maxSteps,
  }, imageData!);
  streamline.setInputData(imageData!);
  streamline.setInputData(seeds, 1);
  const integrated = streamline.getOutputData();
  const rawPoints = integrated.getPoints().getData() ?? [];
  const rawLines = integrated.getLines().getData() ?? [];
  const pointCount = rawPoints.length / 3;
  if (pointCount === 0 || rawLines.length === 0) {
    streamline.delete();
    seeds.delete();
    return {
      geometry: { path: null, arrows: null },
      integratedPoints: 0,
      usedTubes: true,
      knownScratchBytes: 0,
      boundaryContacts: emptyBrickBoundaryContacts(),
    };
  }
  const points = scalePointTriples(rawPoints, request.scale);
  const boundaryContacts = detectStreamlineBoundaryContacts(
    rawPoints,
    rawLines,
    imageData!.getDimensions() as Triple,
    imageData!.getSpacing() as Triple,
    imageData!.getOrigin() as Triple,
  );
  const lines = Uint32Array.from(rawLines);
  // vtkImageStreamline emits geometry without vector attributes. Sample its
  // own interpolant at unscaled path points before computing the magnitude.
  const sourceVectors = imageData!.getPointData().getVectors();
  const magnitudes = sampleStreamlineMagnitudes(rawPoints, (position, vector) =>
    streamline.vectorAt(position, sourceVectors, imageData!, vector));
  const scaled = vtkPolyData.newInstance();
  scaled.getPoints().setData(points, 3);
  scaled.getLines().setData(lines);
  if (magnitudes) scaled.getPointData().setScalars(vtkDataArray.newInstance({
    name: VTK_SCALAR_SLOT,
    values: magnitudes,
    numberOfComponents: 1,
  }));
  scaled.modified();

  let tube: ReturnType<typeof vtkTubeFilter.newInstance> | null = null;
  let path: StreamlineGeometry | null;
  const useTubes = pointCount <= TUBE_INPUT_POINT_CAP;
  if (useTubes) {
    tube = vtkTubeFilter.newInstance({ radius: request.tubeRadius, numberOfSides: 6 });
    tube.setInputData(scaled);
    path = geometryFrom(tube.getOutputData());
  } else {
    path = geometryFrom(scaled);
  }
  const arrow = buildStreamlineArrowGeometry(
    points,
    lines,
    sign > 0 ? "forward" : "backward",
    Math.max(75, request.tubeRadius * 3.2),
    magnitudes,
  );
  const arrowData = vtkPolyData.newInstance();
  if (arrow) {
    arrowData.getPoints().setData(arrow.points, 3);
    arrowData.getPolys().setData(arrow.polys);
    if (arrow.scalars) arrowData.getPointData().setScalars(vtkDataArray.newInstance({
      name: VTK_SCALAR_SLOT,
      values: arrow.scalars,
      numberOfComponents: 1,
    }));
    arrowData.modified();
  }
  const arrows = arrow ? geometryFrom(arrowData) : null;

  const retained = new Set<ArrayBufferLike>([
    ...geometryBuffers(path),
    ...geometryBuffers(arrows),
  ]);
  const scratch = new Set<ArrayBufferLike>();
  const add = (view: unknown) => {
    if (ArrayBuffer.isView(view) && !retained.has(view.buffer)) scratch.add(view.buffer);
  };
  for (const view of [rawPoints, rawLines, points, lines, magnitudes]) add(view);
  if (arrow) {
    add(arrow.points);
    add(arrow.polys);
    add(arrow.scalars);
  }
  addPolyDataBuffers(scratch, integrated, retained);
  addPolyDataBuffers(scratch, scaled, retained);
  addPolyDataBuffers(scratch, tube?.getOutputData() ?? null, retained);
  addPolyDataBuffers(scratch, arrowData, retained);
  let knownScratchBytes = 0;
  for (const buffer of scratch) knownScratchBytes += buffer.byteLength;

  tube?.delete();
  arrowData.delete();
  scaled.delete();
  streamline.delete();
  seeds.delete();
  return {
    geometry: { path, arrows },
    integratedPoints: pointCount,
    usedTubes: useTubes,
    knownScratchBytes,
    boundaryContacts,
  };
}

function extract(request: Extract<StreamlineWorkerRequest, { kind: "extract" }>): StreamlineExtractionResult {
  if (request.preparation) {
    const prepared = prepareVtkRequest(request.preparation);
    loadImage({
      shape: prepared.shape,
      spacing: prepared.spacing,
      origin: prepared.origin,
      values: prepared.values,
    });
    activeDatasetId = request.datasetId;
  }
  if (request.image) {
    loadImage(request.image);
    activeDatasetId = request.datasetId;
  }
  if (!imageData || activeDatasetId !== request.datasetId) {
    throw new Error("Streamline worker does not have the requested prepared frame");
  }
  const startedAt = performance.now();
  const forward = direction(request, 1);
  const backward = direction(request, -1);
  return {
    forward: forward.geometry,
    backward: backward.geometry,
    representation: forward.usedTubes && backward.usedTubes ? "tubes" : "gpu-lines",
    integratedPoints: forward.integratedPoints + backward.integratedPoints,
    boundaryContacts: mergeBrickBoundaryContacts(
      forward.boundaryContacts,
      backward.boundaryContacts,
    ),
    knownScratchBytes: Math.max(forward.knownScratchBytes, backward.knownScratchBytes),
    elapsedMs: performance.now() - startedAt,
  };
}

scope.onmessage = (event) => {
  if (event.data.kind === "close") {
    imageData?.delete();
    imageData = null;
    scope.close();
    return;
  }
  if (event.data.kind === "prepare") {
    const { requestId, datasetId } = event.data;
    try {
      const prepared = prepareVtkRequest(event.data.request);
      loadImage({
        shape: prepared.shape,
        spacing: prepared.spacing,
        origin: prepared.origin,
        values: prepared.values,
      });
      activeDatasetId = datasetId;
      const { values: _values, ...metadata } = prepared;
      const transfer = transferablePreparedBuffers(prepared).filter(
        (buffer) => buffer !== prepared.values.buffer,
      );
      scope.postMessage({
        kind: "prepared",
        requestId,
        datasetId,
        ok: true,
        metadata,
        bytes: prepared.values.byteLength,
      }, transfer);
    } catch (error) {
      scope.postMessage({
        kind: "prepared",
        requestId,
        datasetId,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      }, []);
    }
    return;
  }
  const { requestId, datasetId } = event.data;
  try {
    const result = extract(event.data);
    scope.postMessage(
      { kind: "result", requestId, datasetId, ok: true, result },
      streamlineResultTransferables(result),
    );
  } catch (error) {
    scope.postMessage({
      kind: "result",
      requestId,
      datasetId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

scope.postMessage({ kind: "ready" }, []);

export {};
