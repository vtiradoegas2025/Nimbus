import { finiteImageMarchingCubes } from "./finiteImageMarchingCubes";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
import {
  assessMesocycloneEnvelope,
  assessTornadoScaleVortex,
} from "@/lib/plot/vortexCore";
import { deriveTornadicCondensation } from "@/lib/plot/tornadicCondensation";
import {
  connectedSurfaceHierarchyNearestAnchor,
  connectedSurfaceNearestAnchor,
} from "./vortexSurfaceConnectivity";
import {
  vortexContourTransferables,
  type VortexContourImage,
  type VortexContourAdmission,
  type VortexContourResult,
  type VortexContourWorkerReply,
  type VortexContourWorkerRequest,
  type VortexSurfaceGeometry,
} from "./vortexContourProtocol";
import { prepareVtkRequest } from "./vtkPreparation";
import { transferablePreparedBuffers } from "./vtkDatasetHandoff";

const workerScope = self as unknown as {
  onmessage: ((event: { data: VortexContourWorkerRequest }) => void) | null;
  postMessage: (message: VortexContourWorkerReply, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

let activeDatasetId = -1;
let imageData: ReturnType<typeof vtkImageData.newInstance> | null = null;
let condensateImage: VortexContourImage | null = null;
let pendingExtraction: Extract<VortexContourWorkerRequest, { kind: "extract" }> | null = null;

function loadImage(image: VortexContourImage, condensate: VortexContourImage | null): void {
  imageData?.delete();
  imageData = vtkImageData.newInstance();
  imageData.setDimensions(...image.shape);
  imageData.setSpacing(image.spacing);
  imageData.setOrigin(image.origin);
  imageData.getPointData().setScalars(vtkDataArray.newInstance({
    name: "__nimbus_vortex_scalars",
    values: image.values,
    numberOfComponents: 1,
  }));
  imageData.modified();
  condensateImage = condensate;
}

function finiteBounds(
  bounds: readonly number[],
): [number, number, number, number, number, number] | null {
  if (bounds.length !== 6 || !bounds.every(Number.isFinite)) return null;
  return [bounds[0]!, bounds[1]!, bounds[2]!, bounds[3]!, bounds[4]!, bounds[5]!];
}

function geometryFrom(
  polyData: ReturnType<ReturnType<typeof vtkImageMarchingCubes.newInstance>["getOutputData"]>,
): VortexSurfaceGeometry {
  const points = Float32Array.from(polyData.getPoints().getData() ?? []);
  const polys = Uint32Array.from(polyData.getPolys().getData() ?? []);
  const sourceNormals = polyData.getPointData().getNormals()?.getData() ?? [];
  const normals = Float32Array.from(sourceNormals);
  return {
    points,
    polys,
    normals,
    bounds: points.length > 0 ? finiteBounds(polyData.getBounds()) : null,
    triangles: polyData.getNumberOfPolys(),
    bytes: points.byteLength + polys.byteLength + normals.byteLength,
  };
}

function contour(value: number) {
  const marching = finiteImageMarchingCubes({
    contourValue: value,
    computeNormals: true,
    mergePoints: true,
  });
  marching.setInputData(imageData!);
  const output = marching.getOutputData();
  return { marching, output, geometry: geometryFrom(output) };
}

function contourAdmission(
  request: Extract<VortexContourWorkerRequest, { kind: "extract" }>,
): VortexContourAdmission {
  const dimensions = imageData!.getDimensions() as [number, number, number];
  const [nx, ny, nz] = dimensions;
  const values = imageData!.getPointData().getScalars().getData() as Float32Array;
  const thresholds = [
    request.thresholds.core,
    request.thresholds.opposite,
    request.thresholds.core == null ? null : request.thresholds.mesocyclone,
  ].filter((value): value is number => value != null);
  const crossings = new Array<number>(thresholds.length).fill(0);
  const at = (i: number, j: number, k: number) => values[i + nx * (j + ny * k)]!;
  for (let k = 0; k + 1 < nz; k += 1) {
    for (let j = 0; j + 1 < ny; j += 1) {
      for (let i = 0; i + 1 < nx; i += 1) {
        const a = at(i, j, k); const b = at(i + 1, j, k);
        const c = at(i, j + 1, k); const d = at(i + 1, j + 1, k);
        const e = at(i, j, k + 1); const f = at(i + 1, j, k + 1);
        const g = at(i, j + 1, k + 1); const h = at(i + 1, j + 1, k + 1);
        if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)
            || !Number.isFinite(d) || !Number.isFinite(e) || !Number.isFinite(f)
            || !Number.isFinite(g) || !Number.isFinite(h)) continue;
        const minimum = Math.min(a, b, c, d, e, f, g, h);
        const maximum = Math.max(a, b, c, d, e, f, g, h);
        for (let index = 0; index < thresholds.length; index += 1) {
          if (minimum <= thresholds[index]! && maximum >= thresholds[index]!
              && minimum !== maximum) crossings[index]! += 1;
        }
      }
    }
  }
  const crossingCells = crossings.reduce((sum, value) => sum + value, 0);
  const maximumTriangles = crossingCells * 5;
  const surfaceBytes = maximumTriangles * 88;
  const condensationBytes = request.condensateFloor != null && condensateImage
    ? values.length * (Float32Array.BYTES_PER_ELEMENT + 1)
    : 0;
  const maximumOutputBytes = surfaceBytes + condensationBytes;
  const maximumScratchBytes = surfaceBytes * 3 + condensationBytes;
  if (![maximumTriangles, maximumOutputBytes, maximumScratchBytes].every(Number.isSafeInteger)) {
    throw new Error("Contour preflight topology exceeds safe integer arithmetic");
  }
  return { maximumTriangles, maximumOutputBytes, maximumScratchBytes, crossingCells };
}

function knownContourScratchBytes(
  extracted: ReadonlyArray<ReturnType<typeof contour> | null>,
  result: VortexContourResult,
): number {
  const retained = new Set<ArrayBufferLike>(vortexContourTransferables(result));
  const scratch = new Set<ArrayBufferLike>();
  const add = (view: unknown) => {
    if (ArrayBuffer.isView(view) && !retained.has(view.buffer)) scratch.add(view.buffer);
  };
  for (const item of extracted) {
    if (!item) continue;
    add(item.geometry.points);
    add(item.geometry.polys);
    add(item.geometry.normals);
    add(item.output.getPoints().getData());
    add(item.output.getPolys().getData());
    for (const array of item.output.getPointData().getArrays()) add(array.getData());
    for (const array of item.output.getCellData().getArrays()) add(array.getData());
  }
  let bytes = 0;
  for (const buffer of scratch) bytes += buffer.byteLength;
  return bytes;
}

function extract(request: Extract<VortexContourWorkerRequest, { kind: "extract" }>): VortexContourResult {
  const progress = (stage: Extract<VortexContourWorkerReply, { kind: "progress" }>["stage"]) => {
    if (request.reportProgress) workerScope.postMessage({
      kind: "progress",
      requestId: request.requestId,
      datasetId: request.datasetId,
      stage,
    }, []);
  };
  if (request.preparation) {
    const prepared = prepareVtkRequest({ ...request.preparation, includeValidity: false });
    progress("primary-prepared");
    const preparedCondensate = request.condensatePreparation
      ? prepareVtkRequest({ ...request.condensatePreparation, includeValidity: false })
      : null;
    if (preparedCondensate) progress("condensate-prepared");
    loadImage({
      shape: prepared.shape,
      spacing: prepared.spacing,
      origin: prepared.origin,
      values: prepared.values,
    }, preparedCondensate ? {
      shape: preparedCondensate.shape,
      spacing: preparedCondensate.spacing,
      origin: preparedCondensate.origin,
      values: preparedCondensate.values,
      validity: preparedCondensate.validity,
    } : request.condensateImage ?? null);
    activeDatasetId = request.datasetId;
  } else if (request.condensatePreparation) {
    const preparedCondensate = prepareVtkRequest({ ...request.condensatePreparation, includeValidity: false });
    condensateImage = {
      shape: preparedCondensate.shape,
      spacing: preparedCondensate.spacing,
      origin: preparedCondensate.origin,
      values: preparedCondensate.values,
      validity: preparedCondensate.validity,
    };
  }
  if (request.image) {
    loadImage(request.image, request.condensateImage ?? null);
    activeDatasetId = request.datasetId;
  }
  if (!imageData || activeDatasetId !== request.datasetId) {
    throw new Error("Contour worker does not have the requested prepared frame");
  }
  progress("extracting");

  const startedAt = performance.now();
  const core = request.thresholds.core == null ? null : contour(request.thresholds.core);
  progress("core-extracted");
  const opposite = request.thresholds.opposite == null
    ? null
    : contour(request.thresholds.opposite);
  progress("opposite-extracted");
  const mesocyclone = request.thresholds.mesocyclone == null || !core
    ? null
    : contour(request.thresholds.mesocyclone);
  progress("mesocyclone-extracted");

  const spacing = imageData.getSpacing() as [number, number, number];
  const origin = imageData.getOrigin() as [number, number, number];
  const dimensions = imageData.getDimensions() as [number, number, number];
  const contextBounds: [number, number, number, number, number, number] = [
    origin[0], origin[0] + Math.max(0, dimensions[0] - 1) * spacing[0],
    origin[1], origin[1] + Math.max(0, dimensions[1] - 1) * spacing[1],
    origin[2], origin[2] + Math.max(0, dimensions[2] - 1) * spacing[2],
  ];
  const coreHierarchy = connectedSurfaceHierarchyNearestAnchor(
    core?.geometry ?? null,
    request.anchor,
    3_000,
    { maxComponents: 24, minimumTriangles: 8 },
  );
  const oppositeHierarchy = connectedSurfaceHierarchyNearestAnchor(
    opposite?.geometry ?? null,
    request.anchor,
    5_000,
    { maxComponents: 16, minimumTriangles: 8 },
  );
  const coreGeometry = coreHierarchy.combined;
  const primaryCoreGeometry = coreHierarchy.primary;
  const oppositeGeometry = oppositeHierarchy.combined;
  progress("connectivity-complete");
  let mesocycloneGeometry = connectedSurfaceNearestAnchor(
    mesocyclone?.geometry ?? null,
    request.anchor,
    8_000,
  );
  const mesocycloneAssessment = mesocycloneGeometry && primaryCoreGeometry
    ? assessMesocycloneEnvelope(
        mesocycloneGeometry.bounds,
        primaryCoreGeometry.bounds,
        spacing,
        contextBounds,
      )
    : null;
  if (!mesocycloneAssessment?.present) mesocycloneGeometry = null;
  const vortexAssessment = primaryCoreGeometry
    ? assessTornadoScaleVortex(primaryCoreGeometry.bounds, spacing, origin[2])
    : null;
  const condensation = request.condensateFloor != null && condensateImage && primaryCoreGeometry
    ? deriveTornadicCondensation(
        condensateImage,
        {
          shape: dimensions,
          spacing,
          origin,
          values: imageData.getPointData().getScalars().getData() as Float32Array,
        },
        request.anchor,
        primaryCoreGeometry.bounds,
        request.thresholds.core ?? Number.NaN,
        request.condensateFloor,
        vortexAssessment?.present === true,
      )
    : null;
  progress("condensation-complete");

  const retained = [coreGeometry, oppositeGeometry, mesocycloneGeometry]
    .filter((surface): surface is VortexSurfaceGeometry => surface != null);
  const result: VortexContourResult = {
    core: coreGeometry,
    opposite: oppositeGeometry,
    coreComponentCount: coreHierarchy.componentCount,
    oppositeComponentCount: oppositeHierarchy.componentCount,
    mesocyclone: mesocycloneGeometry,
    mesocycloneAssessment,
    vortexAssessment,
    condensation: condensation?.image ?? null,
    condensationBounds: condensation?.bounds ?? null,
    condensationAssessment: condensation?.assessment ?? null,
    usage: retained.reduce(
      (total, surface) => ({
        triangles: total.triangles + surface.triangles,
        bytes: total.bytes + surface.bytes,
      }),
      { triangles: 0, bytes: 0 },
    ),
    knownScratchBytes: 0,
    elapsedMs: performance.now() - startedAt,
  };
  result.usage.bytes += (result.condensation?.values.byteLength ?? 0)
    + (result.condensation?.validity?.byteLength ?? 0);
  progress("result-built");
  result.knownScratchBytes = knownContourScratchBytes([core, opposite, mesocyclone], result);
  progress("scratch-complete");

  mesocyclone?.marching.delete();
  opposite?.marching.delete();
  core?.marching.delete();
  return result;
}

workerScope.onmessage = (event) => {
  if (event.data.kind === "close" || event.data.kind === "release") {
    imageData?.delete();
    imageData = null;
    condensateImage = null;
    pendingExtraction = null;
    activeDatasetId = -1;
    if (event.data.kind === "release") {
      workerScope.postMessage({ kind: "released", requestId: event.data.requestId }, []);
      return;
    }
    workerScope.close();
    return;
  }
  if (event.data.kind === "prepare") {
    const { requestId, datasetId } = event.data;
    try {
      const prepared = prepareVtkRequest({ ...event.data.request, includeValidity: false });
      loadImage({
        shape: prepared.shape,
        spacing: prepared.spacing,
        origin: prepared.origin,
        values: prepared.values,
      }, null);
      activeDatasetId = datasetId;
      const { values: _values, ...metadata } = prepared;
      const transfer = transferablePreparedBuffers(prepared).filter(
        (buffer) => buffer !== prepared.values.buffer,
      );
      workerScope.postMessage({
        kind: "prepared",
        requestId,
        datasetId,
        ok: true,
        metadata,
        bytes: prepared.values.byteLength,
      }, transfer);
    } catch (error) {
      workerScope.postMessage({
        kind: "prepared",
        requestId,
        datasetId,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      }, []);
    }
    return;
  }
  if (event.data.kind === "continue") {
    const { requestId, datasetId, admission } = event.data;
    const request = pendingExtraction;
    pendingExtraction = null;
    if (!request || request.requestId !== requestId || request.datasetId !== datasetId) {
      workerScope.postMessage({ kind: "result", requestId, datasetId, ok: false,
        error: "Contour admission continuation has no matching preflight" }, []);
      return;
    }
    try {
      const expected = contourAdmission(request);
      if (JSON.stringify(expected) !== JSON.stringify(admission)) {
        throw new Error("Contour admission continuation changed its preflight bound");
      }
      const result = extract(request);
      workerScope.postMessage(
        { kind: "result", requestId, datasetId, ok: true, result },
        vortexContourTransferables(result),
      );
    } catch (error) {
      workerScope.postMessage({ kind: "result", requestId, datasetId, ok: false,
        error: error instanceof Error ? error.message : String(error) }, []);
    }
    return;
  }
  const { requestId, datasetId } = event.data;
  if (event.data.kind === "extract" && event.data.reportProgress) workerScope.postMessage({
    kind: "progress",
    requestId,
    datasetId,
    stage: "received",
  }, []);
  try {
    // Load/rebuild the immutable dataset, then stop before marching cubes.
    // `extract` owns loading today, so invoke a preflight-only clone by loading
    // once and retaining the request for the authorized continuation.
    if (event.data.preparation || event.data.image || event.data.condensatePreparation) {
      const request = event.data;
      const progress = (stage: Extract<VortexContourWorkerReply, { kind: "progress" }>["stage"]) => {
        if (request.reportProgress) workerScope.postMessage({ kind: "progress",
          requestId, datasetId, stage }, []);
      };
      if (request.preparation) {
        const prepared = prepareVtkRequest({ ...request.preparation, includeValidity: false });
        progress("primary-prepared");
        const preparedCondensate = request.condensatePreparation
          ? prepareVtkRequest({ ...request.condensatePreparation, includeValidity: false }) : null;
        if (preparedCondensate) progress("condensate-prepared");
        loadImage({ shape: prepared.shape, spacing: prepared.spacing,
          origin: prepared.origin, values: prepared.values }, preparedCondensate ? {
          shape: preparedCondensate.shape, spacing: preparedCondensate.spacing,
          origin: preparedCondensate.origin, values: preparedCondensate.values, validity: preparedCondensate.validity,
        } : request.condensateImage ?? null);
        activeDatasetId = request.datasetId;
      } else if (request.image) {
        loadImage(request.image, request.condensateImage ?? null);
        activeDatasetId = request.datasetId;
      } else if (request.condensatePreparation) {
        const prepared = prepareVtkRequest({ ...request.condensatePreparation, includeValidity: false });
        progress("condensate-prepared");
        condensateImage = { shape: prepared.shape, spacing: prepared.spacing,
          origin: prepared.origin, values: prepared.values, validity: prepared.validity };
      }
    }
    if (!imageData || activeDatasetId !== datasetId) {
      throw new Error("Contour worker does not have the requested prepared frame");
    }
    pendingExtraction = { ...event.data, image: undefined, preparation: undefined,
      condensateImage: undefined, condensatePreparation: undefined };
    workerScope.postMessage({ kind: "admission", requestId, datasetId,
      admission: contourAdmission(pendingExtraction) }, []);
  } catch (error) {
    workerScope.postMessage({
      kind: "result",
      requestId,
      datasetId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

workerScope.postMessage({ kind: "ready" }, []);

export {};
