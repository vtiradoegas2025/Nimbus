import vtkMacro from "@kitware/vtk.js/macros.js";
import vtkImageStreamline from "@kitware/vtk.js/Filters/General/ImageStreamline.js";
import { finiteVectorSupport, hasMissingValues } from "./finiteVectorSupport";

type Image = Parameters<ReturnType<typeof vtkImageStreamline.newInstance>["vectorAt"]>[2];
export function finiteImageStreamline(options: Parameters<typeof vtkImageStreamline.newInstance>[0], image: Image) {
  const vectors = image.getPointData().getVectors();
  const values = vectors.getData() as Float32Array;
  const missing = hasMissingValues(values);
  const dimensions = image.getDimensions(), origin = image.getOrigin(), spacing = image.getSpacing();
  const extent = image.getExtent(), bounds = image.getBounds();
  const support = missing ? finiteVectorSupport(values, dimensions, origin, spacing) : null;
  return vtkMacro.newInstance((api: object, model: object) => {
    vtkImageStreamline.extend(api, model, options);
    const integration = api as ReturnType<typeof vtkImageStreamline.newInstance>;
    if (!support) {
      const indices = new Int32Array(3), coordinates = new Float32Array(3);
      const weights = new Float32Array(8), ids = new Uint32Array(8);
      const interpolation = integration as unknown as {
        computeStructuredCoordinates: (...arrays: ArrayLike<number>[]) => boolean;
        interpolationFunctions: (...arrays: ArrayLike<number>[]) => void;
        getVoxelIndices: (...arrays: ArrayLike<number>[]) => void;
      };
      // VTK rebuilds image bounds and allocates tuples at every RK2 sample.
      // Retain its coordinate/weight arithmetic, including half-cell edges.
      integration.vectorAt = (position, _vectors, _image, vector) => {
        if (!interpolation.computeStructuredCoordinates(position, indices, coordinates, extent, spacing, origin, bounds)) return false;
        interpolation.interpolationFunctions(coordinates, weights);
        interpolation.getVoxelIndices(indices, dimensions, ids);
        vector[0] = 0; vector[1] = 0; vector[2] = 0;
        for (let i = 0; i < 8; i++) {
          const offset = ids[i]! * 3;
          for (let component = 0; component < 3; component++) vector[component] += weights[i]! * values[offset + component]!;
        }
        return true;
      };
      return;
    }
    integration.vectorAt = (position, _vectors, _image, vector) => support.sample(position, vector);
    const advance = integration.computeNextStep;
    const start = [0, 0, 0];
    integration.computeNextStep = (field, source, duration, position) => {
      for (let axis = 0; axis < 3; axis++) start[axis] = position[axis];
      if (advance(field, source, duration, position) && support.connects(start, position)) return true;
      for (let axis = 0; axis < 3; axis++) position[axis] = start[axis];
      return false;
    };
  }, "vtkNimbusFiniteImageStreamline")() as ReturnType<typeof vtkImageStreamline.newInstance>;
}
