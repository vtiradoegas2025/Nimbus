import { volumeImageBytes } from "./volumeValidityBudget";
import { finiteImageMarchingCubes } from "./finiteImageMarchingCubes";
import { geometryScalarValues } from "./vtkDatasetHandoff";
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
import type { PreparedVtkImage } from "./vtkPreparation";
import type {
  VortexContourResult,
  VortexContourThresholds,
  VortexSurfaceGeometry,
} from "./vortexContourProtocol";
import {
  localVtkContourBudgetError,
  throwIfLocalVtkTaskAborted,
} from "./vtkWorkerRuntime";

function geometryFrom(polyData: ReturnType<ReturnType<typeof vtkImageMarchingCubes.newInstance>["getOutputData"]>): VortexSurfaceGeometry {
  const points = Float32Array.from(polyData.getPoints().getData() ?? []);
  const polys = Uint32Array.from(polyData.getPolys().getData() ?? []);
  const normals = Float32Array.from(polyData.getPointData().getNormals()?.getData() ?? []);
  const bounds = points.length > 0 && polyData.getBounds().every(Number.isFinite)
    ? polyData.getBounds() as [number, number, number, number, number, number]
    : null;
  return {
    points,
    polys,
    normals,
    bounds,
    triangles: polyData.getNumberOfPolys(),
    bytes: points.byteLength + polys.byteLength + normals.byteLength,
  };
}

/** Correctness fallback for WebKit module-worker startup failure. */
export function extractVtkContoursLocally(
  prepared: PreparedVtkImage,
  thresholds: VortexContourThresholds,
  anchor?: readonly [number, number, number] | null,
  condensatePrepared?: PreparedVtkImage | null,
  condensateFloor?: number | null,
  signal?: AbortSignal,
): VortexContourResult {
  const voxelCount = prepared.shape[0] * prepared.shape[1] * prepared.shape[2];
  const budgetError = localVtkContourBudgetError(
    voxelCount,
    volumeImageBytes(prepared) + volumeImageBytes(condensatePrepared),
  );
  if (budgetError) throw new Error(budgetError);
  throwIfLocalVtkTaskAborted(signal);
  const startedAt = performance.now();
  const image = vtkImageData.newInstance();
  image.setDimensions(...prepared.shape);
  image.setSpacing(prepared.spacing);
  image.setOrigin(prepared.origin);
  image.getPointData().setScalars(vtkDataArray.newInstance({
    name: "__nimbus_contour_fallback_scalars",
    values: geometryScalarValues(prepared),
    numberOfComponents: 1,
  }));
  image.modified();
  const contour = (value: number) => {
    const filter = finiteImageMarchingCubes({
      contourValue: value,
      computeNormals: true,
      mergePoints: true,
    });
    filter.setInputData(image);
    const output = filter.getOutputData();
    return { filter, output, geometry: geometryFrom(output) };
  };
  const core = thresholds.core == null ? null : contour(thresholds.core);
  const opposite = thresholds.opposite == null ? null : contour(thresholds.opposite);
  const mesocyclone = thresholds.mesocyclone == null || !core
    ? null
    : contour(thresholds.mesocyclone);
  const coreHierarchy = connectedSurfaceHierarchyNearestAnchor(
    core?.geometry ?? null,
    anchor,
    3_000,
    { maxComponents: 24, minimumTriangles: 8 },
  );
  const oppositeHierarchy = connectedSurfaceHierarchyNearestAnchor(
    opposite?.geometry ?? null,
    anchor,
    5_000,
    { maxComponents: 16, minimumTriangles: 8 },
  );
  const coreGeometry = coreHierarchy.combined;
  const primaryCoreGeometry = coreHierarchy.primary;
  const oppositeGeometry = oppositeHierarchy.combined;
  let mesocycloneGeometry = connectedSurfaceNearestAnchor(
    mesocyclone?.geometry ?? null,
    anchor,
    8_000,
  );
  const assessment = mesocycloneGeometry && primaryCoreGeometry
    ? assessMesocycloneEnvelope(
        mesocycloneGeometry.bounds,
        primaryCoreGeometry.bounds,
        prepared.spacing,
        [
          prepared.origin[0], prepared.origin[0] + Math.max(0, prepared.shape[0] - 1) * prepared.spacing[0],
          prepared.origin[1], prepared.origin[1] + Math.max(0, prepared.shape[1] - 1) * prepared.spacing[1],
          prepared.origin[2], prepared.origin[2] + Math.max(0, prepared.shape[2] - 1) * prepared.spacing[2],
        ],
      )
    : null;
  if (!assessment?.present) mesocycloneGeometry = null;
  const vortexAssessment = primaryCoreGeometry
    ? assessTornadoScaleVortex(primaryCoreGeometry.bounds, prepared.spacing, prepared.origin[2])
    : null;
  const condensation = condensatePrepared && condensateFloor != null && primaryCoreGeometry
    ? deriveTornadicCondensation(
        condensatePrepared,
        prepared,
        anchor,
        primaryCoreGeometry.bounds,
        thresholds.core ?? Number.NaN,
        condensateFloor,
        vortexAssessment?.present === true,
      )
    : null;
  const retained = [coreGeometry, oppositeGeometry, mesocycloneGeometry]
    .filter((surface): surface is VortexSurfaceGeometry => surface != null);
  const result: VortexContourResult = {
    core: coreGeometry,
    opposite: oppositeGeometry,
    coreComponentCount: coreHierarchy.componentCount,
    oppositeComponentCount: oppositeHierarchy.componentCount,
    mesocyclone: mesocycloneGeometry,
    mesocycloneAssessment: assessment,
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
  result.usage.bytes += volumeImageBytes(result.condensation);
  mesocyclone?.filter.delete();
  opposite?.filter.delete();
  core?.filter.delete();
  image.delete();
  return result;
}
