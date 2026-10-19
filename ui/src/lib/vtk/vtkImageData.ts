import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import {
  cartesianGridToVtkScalars,
  cartesianGridToVtkVectors,
  type CartesianVoxelGrid,
} from "@/lib/science/cartesianInterp";
import {
  assertVtkArrayAllocation,
  replaceVtkPointDataArray,
  VTK_SCALAR_SLOT,
  VTK_VECTOR_SLOT,
  type VtkPointDataOwner,
} from "./vtkArrayOwnership";
import type { PreparedVtkImage } from "./vtkPreparation";
import { uploadPreparedVtkValues } from "./vtkDatasetHandoff";
import { getVolumeValidity, setVolumeValidity } from "./volumeValidity";

type VtkImageData = ReturnType<typeof vtkImageData.newInstance>;
type VtkPointData = ReturnType<VtkImageData["getPointData"]>;
type VtkArray = ReturnType<typeof vtkDataArray.newInstance>;

const preparedArrays = new WeakMap<VtkImageData, {
  source: Float32Array;
  uploaded: Float32Array;
  array: VtkArray;
  shape: readonly number[];
  kind: "scalars" | "vectors";
}>();

function assertImageGeometry(spacing: readonly number[], origin: readonly number[]): void {
  if (spacing.length !== 3 || spacing.some(value => !Number.isFinite(value) || value <= 0)
      || origin.length !== 3 || origin.some(value => !Number.isFinite(value))) {
    throw new Error("Prepared VTK image geometry requires finite origins and positive finite spacing");
  }
}

/**
 * Give a new marching-cubes input a structurally valid empty lattice. Hidden
 * actors can still be queried by resize/render passes before worker data lands,
 * and vtk.js assumes the input already owns a scalar array.
 */
export function initializeEmptyImageScalars(image: VtkImageData): void {
  image.setDimensions(2, 2, 2);
  image.setSpacing([1, 1, 1]);
  image.setOrigin([0, 0, 0]);
  replaceVtkPointDataArray(
    image.getPointData() as unknown as VtkPointDataOwner,
    "scalars",
    vtkDataArray.newInstance({
      name: VTK_SCALAR_SLOT,
      values: new Float32Array(8),
      numberOfComponents: 1,
    }),
  );
  image.modified();
}

/**
 * Configure a vtkImageData from a CartesianVoxelGrid SCALAR field: set
 * dimensions/spacing/origin and attach the scalars in VTK's x-fastest order
 * (via cartesianGridToVtkScalars). Centralizes the near-identical scalar-push
 * sites so a new 3D consumer cannot set the dimensions but forget the repack
 * (which silently transposes/scrambles non-cubic grids).
 */
export function setImageScalars(image: VtkImageData, cart: CartesianVoxelGrid): void {
  image.setDimensions(cart.shape[0], cart.shape[1], cart.shape[2]);
  image.setSpacing(cart.spacing);
  image.setOrigin(cart.origin);
  const elementCount = cart.shape[0] * cart.shape[1] * cart.shape[2];
  assertVtkArrayAllocation(elementCount, 1);
  replaceVtkPointDataArray(
    image.getPointData() as unknown as VtkPointDataOwner,
    "scalars",
    vtkDataArray.newInstance({
      name: VTK_SCALAR_SLOT,
      values: cartesianGridToVtkScalars(cart),
      numberOfComponents: 1,
    }),
  );
  image.modified();
}

/** Vector counterpart of setImageScalars: attach 3-component velocity in VTK's
 *  x-fastest order (via cartesianGridToVtkVectors) for streamline integration. */
export function setImageVectors(image: VtkImageData, cart: CartesianVoxelGrid): void {
  image.setDimensions(cart.shape[0], cart.shape[1], cart.shape[2]);
  image.setSpacing(cart.spacing);
  image.setOrigin(cart.origin);
  const elementCount = cart.shape[0] * cart.shape[1] * cart.shape[2];
  assertVtkArrayAllocation(elementCount, 3);
  replaceVtkPointDataArray(
    image.getPointData() as unknown as VtkPointDataOwner,
    "vectors",
    vtkDataArray.newInstance({
      name: VTK_VECTOR_SLOT,
      values: cartesianGridToVtkVectors(cart),
      numberOfComponents: 3,
    }),
  );
  image.modified();
}

/** Attach a scalar array to arbitrary vtkPolyData with the same ownership rule. */
export function setPointScalars(
  pointData: VtkPointData,
  values: Float32Array,
): void {
  assertVtkArrayAllocation(values.length, 1);
  replaceVtkPointDataArray(
    pointData as unknown as VtkPointDataOwner,
    "scalars",
    vtkDataArray.newInstance({
      name: VTK_SCALAR_SLOT,
      values,
      numberOfComponents: 1,
    }),
  );
}

export function setPreparedImageScalars(
  image: VtkImageData,
  prepared: Pick<PreparedVtkImage, "shape" | "spacing" | "origin" | "values" | "validity">,
): void {
  if (prepared.validity && prepared.validity.length !== prepared.values.length) {
    throw new Error("Volume validity length does not match its scalar image");
  }
  setPreparedImageArray(image, prepared, "scalars", 1, VTK_SCALAR_SLOT);
  const previousValidity = getVolumeValidity(image);
  setVolumeValidity(image, prepared.validity);
  if (previousValidity !== prepared.validity) image.modified();
}

export function setPreparedImageVectors(image: VtkImageData, prepared: Pick<PreparedVtkImage, "shape" | "spacing" | "origin" | "values">): void {
  setPreparedImageArray(image, prepared, "vectors", 3, VTK_VECTOR_SLOT);
}

/**
 * Put a prepared Cartesian image directly into display coordinates. Volume
 * mapper shader-loop bounds are derived from ImageData bounds and ignore an
 * actor's scale, so scaling the volume actor can truncate rays under vertical
 * exaggeration. Scaling spacing/origin keeps those bounds honest while leaving
 * scalar values and model fidelity untouched.
 */
export function setPreparedImageDisplayGeometry(
  image: VtkImageData,
  prepared: Pick<PreparedVtkImage, "spacing" | "origin">,
  scale: { xy: number; z: number },
): void {
  if (!Number.isFinite(scale.xy) || scale.xy <= 0 || !Number.isFinite(scale.z) || scale.z <= 0) {
    throw new Error("VTK display scale must be positive and finite");
  }
  assertImageGeometry(prepared.spacing, prepared.origin);
  const spacing: [number, number, number] = [
    prepared.spacing[0] * scale.xy,
    prepared.spacing[1] * scale.xy,
    prepared.spacing[2] * scale.z,
  ];
  const origin: [number, number, number] = [
    prepared.origin[0] * scale.xy,
    prepared.origin[1] * scale.xy,
    prepared.origin[2] * scale.z,
  ];
  assertImageGeometry(spacing, origin);
  image.setSpacing(spacing);
  image.setOrigin(origin);
}

function setPreparedImageArray(
  image: VtkImageData,
  prepared: Pick<PreparedVtkImage, "shape" | "spacing" | "origin" | "values">,
  kind: "scalars" | "vectors",
  numberOfComponents: 1 | 3,
  name: string,
): void {
  if (prepared.shape.length !== 3 || prepared.shape.some(value => !Number.isSafeInteger(value) || value < 1)) {
    throw new Error("Prepared VTK image dimensions must be positive safe integers");
  }
  const [nx, ny, nz] = prepared.shape;
  assertImageGeometry(prepared.spacing, prepared.origin);
  assertVtkArrayAllocation(nx * ny * nz, numberOfComponents);
  if (prepared.values.length !== nx * ny * nz * numberOfComponents) {
    throw new Error("Prepared VTK array length does not match image dimensions");
  }
  const pointData = image.getPointData();
  const previous = preparedArrays.get(image);
  const active = kind === "scalars" ? pointData.getScalars() : pointData.getVectors();
  // vtk.js keys scalar textures by array mtime; immutable display geometry must not replace that array.
  const reusable = previous?.kind === kind && previous.array === active
    && active.getData() === previous.uploaded && active.getNumberOfComponents() === numberOfComponents
    && previous.source.buffer === prepared.values.buffer
    && previous.source.byteOffset === prepared.values.byteOffset
    && previous.source.byteLength === prepared.values.byteLength
    && previous.shape.every((value, axis) => value === prepared.shape[axis])
    && image.getDimensions().every((value, axis) => value === prepared.shape[axis]);
  if (!reusable) {
    uploadPreparedVtkValues(prepared.values, (values) => {
      const array = vtkDataArray.newInstance({ name, values, numberOfComponents });
      replaceVtkPointDataArray(pointData as unknown as VtkPointDataOwner, kind, array);
      preparedArrays.set(image, { source: prepared.values, uploaded: values, array, kind, shape: [...prepared.shape] });
    }, kind === "vectors" ? "vtk-vector-upload" : "vtk-scalar-upload");
  }
  image.setDimensions(nx, ny, nz);
  image.setSpacing(prepared.spacing);
  image.setOrigin(prepared.origin);
}
