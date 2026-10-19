import { finiteImageStreamline } from "./finiteImageStreamline";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import vtkTubeFilter from "@kitware/vtk.js/Filters/General/TubeFilter.js";
import { VTK_SCALAR_SLOT, VTK_VECTOR_SLOT } from "./vtkArrayOwnership";
import { scalePointTriples } from "./vtkScaledPolyData";
import { buildStreamlineArrowGeometry, sampleStreamlineMagnitudes } from "./streamlineDirection";
import type { PreparedVectorImage } from "./vtkPreparation";
import type {
  StreamlineDirectionGeometry,
  StreamlineExtractionParameters,
  StreamlineExtractionResult,
  StreamlineGeometry,
} from "./streamlineExtractionProtocol";
import {
  localVtkStreamlineBudgetError,
  throwIfLocalVtkTaskAborted,
} from "./vtkWorkerRuntime";
import {
  detectStreamlineBoundaryContacts,
  emptyBrickBoundaryContacts,
  mergeBrickBoundaryContacts,
} from "./streamlineBoundary";
import type { BrickBoundaryContacts, Triple } from "@/lib/plot/fieldBrickHierarchy";

const TUBE_INPUT_POINT_CAP = 250_000;

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

function extractDirection(
  image: ReturnType<typeof vtkImageData.newInstance>,
  parameters: StreamlineExtractionParameters,
  sign: 1 | -1,
): {
  geometry: StreamlineDirectionGeometry;
  integratedPoints: number;
  usedTubes: boolean;
  knownScratchBytes: number;
  boundaryContacts: BrickBoundaryContacts;
} {
  const seeds = vtkPolyData.newInstance();
  seeds.getPoints().setData(parameters.seeds, 3);
  seeds.modified();
  const streamline = finiteImageStreamline({
    integrationStep: sign * parameters.integrationStep,
    maximumNumberOfSteps: parameters.maxSteps,
  }, image);
  streamline.setInputData(image);
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
  const points = scalePointTriples(rawPoints, parameters.scale);
  const boundaryContacts = detectStreamlineBoundaryContacts(
    rawPoints,
    rawLines,
    image.getDimensions() as Triple,
    image.getSpacing() as Triple,
    image.getOrigin() as Triple,
  );
  const lines = Uint32Array.from(rawLines);
  const sourceVectors = image.getPointData().getVectors();
  const magnitudes = sampleStreamlineMagnitudes(rawPoints, (position, vector) =>
    streamline.vectorAt(position, sourceVectors, image, vector));
  const scaled = vtkPolyData.newInstance();
  scaled.getPoints().setData(points, 3);
  scaled.getLines().setData(lines);
  if (magnitudes) scaled.getPointData().setScalars(vtkDataArray.newInstance({
    name: VTK_SCALAR_SLOT,
    values: magnitudes,
    numberOfComponents: 1,
  }));
  scaled.modified();
  const useTubes = pointCount <= TUBE_INPUT_POINT_CAP;
  let tube: ReturnType<typeof vtkTubeFilter.newInstance> | null = null;
  let path: StreamlineGeometry | null;
  if (useTubes) {
    tube = vtkTubeFilter.newInstance({ radius: parameters.tubeRadius, numberOfSides: 6 });
    tube.setInputData(scaled);
    path = geometryFrom(tube.getOutputData());
  } else {
    path = geometryFrom(scaled);
  }
  const arrow = buildStreamlineArrowGeometry(
    points,
    lines,
    sign > 0 ? "forward" : "backward",
    Math.max(75, parameters.tubeRadius * 3.2),
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
  tube?.delete();
  arrowData.delete();
  scaled.delete();
  streamline.delete();
  seeds.delete();
  return {
    geometry: { path, arrows },
    integratedPoints: pointCount,
    usedTubes: useTubes,
    knownScratchBytes: 0,
    boundaryContacts,
  };
}

/** Correctness fallback for WebKit module-worker startup failure. */
export function extractStreamlinesLocally(
  prepared: PreparedVectorImage,
  parameters: StreamlineExtractionParameters,
  signal?: AbortSignal,
): StreamlineExtractionResult {
  const seedCount = parameters.seeds.length / 3;
  const budgetError = localVtkStreamlineBudgetError(
    prepared.values.byteLength,
    seedCount,
    parameters.maxSteps,
  );
  if (budgetError) throw new Error(budgetError);
  throwIfLocalVtkTaskAborted(signal);
  const startedAt = performance.now();
  const image = vtkImageData.newInstance();
  try {
    image.setDimensions(...prepared.shape);
    image.setSpacing(prepared.spacing);
    image.setOrigin(prepared.origin);
    image.getPointData().setVectors(vtkDataArray.newInstance({
      name: VTK_VECTOR_SLOT,
      values: prepared.values,
      numberOfComponents: 3,
    }));
    image.modified();
    const forward = extractDirection(image, parameters, 1);
    throwIfLocalVtkTaskAborted(signal);
    const backward = extractDirection(image, parameters, -1);
    throwIfLocalVtkTaskAborted(signal);
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
  } finally {
    image.delete();
  }
}
