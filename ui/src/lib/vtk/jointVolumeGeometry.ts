import { identityAffine, inverseAffine, transformPoint } from "@/lib/plot/affineWind";
export interface JointImage {
  getDimensions(): number[];
  getOrigin(): number[];
  getSpacing(): number[];
  getDirection(): number[];
  getMotionTransform?(): number[];
}

export function jointVolumeGeometry(images: readonly JointImage[]) {
  if (images.length < 1 || images.length > 4) throw new Error("Joint rendering requires one to four scalar volumes");
  const geometry = images.map((image) => {
    const dimensions = image.getDimensions(), spacing = image.getSpacing(), origin = image.getOrigin();
    if (dimensions.length !== 3 || spacing.length !== 3 || origin.length !== 3
        || dimensions.some((value) => !Number.isSafeInteger(value) || value < 2)
        || spacing.some((value) => !Number.isFinite(value) || value <= 0)
        || origin.some((value) => !Number.isFinite(value))
        || image.getDirection().length !== 9
        || image.getDirection().some((value, index) => value !== Number(index % 4 === 0))) {
      throw new Error("Joint volume inputs require finite axis-aligned three-dimensional images");
    }
    const lower = origin.map((value, axis) => value - 0.5 * spacing[axis]);
    const length = dimensions.map((value, axis) => value * spacing[axis]);
    if (![...lower, ...length, ...lower.map((value, axis) => value + length[axis])].every(Number.isFinite)
        || lower.some((value, axis) => value + length[axis] <= value)) {
      throw new Error("Joint volume physical bounds overflow");
    }
    const transform = image.getMotionTransform?.() ?? identityAffine();
    const inverse = inverseAffine(transform);
    const corners = Array.from({ length: 8 }, (_, corner) => transformPoint(transform, lower.map((value, axis) => value + ((corner >> axis) & 1) * length[axis])));
    const worldLower = [0, 1, 2].map(axis => Math.min(...corners.map(point => point[axis])));
    const worldUpper = [0, 1, 2].map(axis => Math.max(...corners.map(point => point[axis])));
    const inverseNorm = Math.sqrt(Math.max(...[0, 1, 2].map(row => [0, 1, 2].reduce((sum, col) => sum + Math.abs(inverse[col * 4 + row]), 0)))
      * Math.max(...[0, 1, 2].map(col => [0, 1, 2].reduce((sum, row) => sum + Math.abs(inverse[col * 4 + row]), 0))));
    return { lower, length, spacing, dimensions, transform, inverse, worldLower, worldUpper, sampleScale: Math.min(1, 1 / inverseNorm) };
  });
  const reference = geometry[0];
  const lower = [0, 1, 2].map((axis) => Math.min(...geometry.map((item) => item.worldLower[axis])));
  const upper = [0, 1, 2].map((axis) => Math.max(...geometry.map((item) => item.worldUpper[axis])));
  return {
    bounds: [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]],
    lower: lower.map((value, axis) => (value - reference.lower[axis]) / reference.length[axis]),
    upper: upper.map((value, axis) => (value - reference.lower[axis]) / reference.length[axis]),
    layers: geometry.map((item) => ({
      lower: item.lower, transform: item.transform, inverse: item.inverse, sampleScale: item.sampleScale,
      worldLength: item.worldUpper.map((value, axis) => value - item.worldLower[axis]),
      length: item.length,
      scale: reference.length.map((value, axis) => value / item.length[axis]),
      shift: reference.lower.map((value, axis) => (value - item.lower[axis]) / item.length[axis]),
      step: item.spacing.map((value, axis) => value / reference.length[axis]),
      spacing: item.spacing,
    })),
  };
}

export function jointVolumeViewGeometry(geometry: ReturnType<typeof jointVolumeGeometry>, view: ArrayLike<number>) {
  if (view.length !== 16 || Array.from(view).some((value) => !Number.isFinite(value))) throw new Error("Joint volume camera matrix is invalid");
  return geometry.layers.map((layer) => {
    const moved = transformPoint(layer.transform, layer.lower);
    return {
      origin: [0, 1, 2].map(row => view[12 + row] + moved.reduce((sum, value, col) => sum + view[col * 4 + row] * value, 0)),
      fixedOrigin: [0, 1, 2].map(row => view[12 + row] + layer.lower.reduce((sum, value, col) => sum + view[col * 4 + row] * value, 0)),
      toIndex: Array.from({ length: 9 }, (_, index) => {
        const row = index % 3, col = Math.floor(index / 3);
        return [0, 1, 2].reduce((sum, k) => sum + layer.inverse[k * 4 + row] * view[k * 4 + col], 0) / layer.length[row];
      }),
      fixedToIndex: Array.from({ length: 9 }, (_, index) => view[(index % 3) * 4 + Math.floor(index / 3)] / layer.length[index % 3]),
      normal: Array.from({ length: 9 }, (_, index) => {
        const row = index % 3, col = Math.floor(index / 3);
        return [0, 1, 2].reduce((sum, k) => sum + view[k * 4 + row] * layer.inverse[k * 4 + col], 0);
      }),
      step: Array.from({ length: 9 }, (_, index) => {
        const row = index % 3, col = Math.floor(index / 3);
        return [0, 1, 2].reduce((sum, k) => sum + view[k * 4 + row] * layer.transform[col * 4 + k], 0) * layer.spacing[col];
      }),
    };
  });
}
