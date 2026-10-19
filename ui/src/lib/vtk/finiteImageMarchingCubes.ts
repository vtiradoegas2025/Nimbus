import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes.js";
import { hasMissingValues } from "./finiteVectorSupport";

type Marching = ReturnType<typeof vtkImageMarchingCubes.newInstance>;
type Options = Parameters<typeof vtkImageMarchingCubes.newInstance>[0];
type Tuple = number[];
interface ExtractionApi {
  requestData(input: unknown[], output: unknown[]): void;
  produceTriangles(value: number, i: number, j: number, k: number, extent: Tuple, slice: number,
    dimensions: Tuple, origin: Tuple, spacing: Tuple, scalars: Float32Array,
    points: number[], triangles: number[], normals: number[]): void;
  getPointGradient(i: number, j: number, k: number, dimensions: Tuple, slice: number,
    spacing: Tuple, scalars: Float32Array, gradient: number[]): void;
}

export function finiteImageMarchingCubes(options: Options): Marching {
  return vtkMacro.newInstance((api: object, model: object) => {
    vtkImageMarchingCubes.extend(api, model, options);
    const extraction = api as ExtractionApi;
    const produce = extraction.produceTriangles;
    const pointGradient = extraction.getPointGradient;
    let hasMissing = false;
    let source: Float32Array | null = null;
    const request = extraction.requestData;
    extraction.requestData = (input, output) => { source = null; request(input, output); };
    extraction.produceTriangles = (value, i, j, k, extent, slice, dims, origin, spacing, scalars, points, triangles, normals) => {
      if (source !== scalars) { source = scalars; hasMissing = hasMissingValues(scalars); }
      if (!hasMissing) { produce(value, i, j, k, extent, slice, dims, origin, spacing, scalars, points, triangles, normals); return; }
      const a = i + j * dims[0] + k * slice;
      if (!Number.isFinite(scalars[a]) || !Number.isFinite(scalars[a + 1])
          || !Number.isFinite(scalars[a + dims[0]]) || !Number.isFinite(scalars[a + dims[0] + 1])
          || !Number.isFinite(scalars[a + slice]) || !Number.isFinite(scalars[a + slice + 1])
          || !Number.isFinite(scalars[a + slice + dims[0]]) || !Number.isFinite(scalars[a + slice + dims[0] + 1])) return;
      produce(value, i, j, k, extent, slice, dims, origin, spacing, scalars, points, triangles, normals);
    };
    // A missing neighbor is not a zero-valued scalar. Use a finite one-sided
    // derivative at gap boundaries; preserve VTK's negative-gradient convention.
    extraction.getPointGradient = (i, j, k, dims, slice, spacing, scalars, gradient) => {
      if (!hasMissing) { pointGradient(i, j, k, dims, slice, spacing, scalars, gradient); return; }
      const center = i + j * dims[0] + k * slice;
      for (let axis = 0; axis < 3; axis++) {
        const index = axis === 0 ? i : axis === 1 ? j : k;
        const stride = axis === 0 ? 1 : axis === 1 ? dims[0] : slice;
        const below = index > 0 && Number.isFinite(scalars[center - stride]) ? center - stride : center;
        const above = index + 1 < dims[axis] && Number.isFinite(scalars[center + stride]) ? center + stride : center;
        const distance = Number(below !== center) + Number(above !== center);
        gradient[axis] = distance > 0 ? (scalars[below] - scalars[above]) / (distance * spacing[axis]) : 0;
      }
    };
  }, "vtkNimbusFiniteImageMarchingCubes")() as Marching;
}
