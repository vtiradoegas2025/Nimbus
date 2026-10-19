import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import { releaseVtkPointDataArrays } from "./vtkArrayOwnership";
import { setPointScalars } from "./vtkImageData";

export interface ImmutablePolyDataGeometry {
  points: Float32Array;
  lines?: Uint32Array;
  polys?: Uint32Array;
  strips?: Uint32Array;
  normals?: Float32Array;
  scalars?: Float32Array | null;
}

const emptyPoints = new Float32Array();
const emptyCells = new Uint32Array();

function sameSpan(actual: ArrayLike<number>, expected: ArrayBufferView): boolean {
  return ArrayBuffer.isView(actual) && actual.constructor === expected.constructor
    && actual.buffer === expected.buffer && actual.byteOffset === expected.byteOffset
    && actual.byteLength === expected.byteLength;
}

function sameCells(actual: ArrayLike<number>, expected: Uint32Array | undefined): boolean {
  return expected ? sameSpan(actual, expected) : actual.length === 0;
}

export function applyImmutablePolyDataGeometry(
  target: ReturnType<typeof vtkPolyData.newInstance>,
  geometry: ImmutablePolyDataGeometry | null,
): boolean {
  const count = (geometry?.points.length ?? 0) / 3;
  if (!Number.isSafeInteger(count)
      || (geometry?.scalars && geometry.scalars.length !== count)
      || (geometry?.normals?.length && geometry.normals.length !== count * 3)) {
    throw new Error("Immutable VTK geometry attributes do not match its points");
  }
  const pointData = target.getPointData();
  const scalars = pointData.getScalars();
  const normals = pointData.getNormals();
  const hasScalars = geometry?.scalars != null;
  const hasNormals = Boolean(geometry?.normals?.length);
  // vtkDataArray.setData changes mtime even for an unchanged buffer, invalidating cached GPU geometry.
  const unchanged = geometry
    ? sameSpan(target.getPoints().getData(), geometry.points)
      && target.getPoints().getNumberOfComponents() === 3
      && sameCells(target.getLines().getData(), geometry.lines)
      && sameCells(target.getPolys().getData(), geometry.polys)
      && sameCells(target.getStrips().getData(), geometry.strips)
      && target.getVerts().getData().length === 0
      && (hasScalars ? scalars && scalars.getNumberOfComponents() === 1 && sameSpan(scalars.getData(), geometry.scalars!) : !scalars)
      && (hasNormals ? normals && normals.getNumberOfComponents() === 3 && sameSpan(normals.getData(), geometry.normals!) : !normals)
      && pointData.getArrays().length === Number(hasScalars) + Number(hasNormals)
    : target.getPoints().getData().length === 0 && target.getLines().getData().length === 0
      && target.getPolys().getData().length === 0 && target.getStrips().getData().length === 0
      && target.getVerts().getData().length === 0 && pointData.getArrays().length === 0;
  if (unchanged) return false;
  releaseVtkPointDataArrays(pointData);
  target.getPoints().setData(geometry?.points ?? emptyPoints, 3);
  target.getLines().setData(geometry?.lines ?? emptyCells);
  target.getPolys().setData(geometry?.polys ?? emptyCells);
  target.getStrips().setData(geometry?.strips ?? emptyCells);
  target.getVerts().setData(emptyCells);
  if (hasScalars) setPointScalars(pointData, geometry!.scalars!);
  if (hasNormals) pointData.setNormals(vtkDataArray.newInstance({
    name: "Normals", values: geometry!.normals, numberOfComponents: 3,
  }));
  target.modified();
  return true;
}
