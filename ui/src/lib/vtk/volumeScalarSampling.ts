import vtkTexture from "@kitware/vtk.js/Rendering/OpenGL/Texture.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import { Filter, Wrap } from "@kitware/vtk.js/Rendering/OpenGL/Texture/Constants.js";
import { planVolumeTextureLayout, uploadVolumeTexturePrefix, VOLUME_TEXTURE_ADDRESS_GLSL,
  type VolumeTextureLayout } from "./volumeTexturePacking";
import { volumeTextureCapabilities } from "./volumeTextureCapabilities";

type Texture = ReturnType<typeof vtkTexture.newInstance>;
export interface VolumeScalarTexture extends Texture {
  setMinificationFilter(value: number): void;
  setMagnificationFilter(value: number): void;
  getHandle(): unknown;
  getVolumeInfo(): { scale: number[]; offset: number[]; width: number; height: number; depth: number };
}
export interface VolumeTextureWindow {
  getContext(): WebGL2RenderingContext;
}
export function assertVolumeTextureDimensions(dimensions: readonly number[], maximumDimension: number): void {
  if (dimensions.length !== 3 || dimensions.some(value => !Number.isSafeInteger(value) || value < 1)) {
    throw new Error("Volume texture dimensions must be positive safe integers");
  }
  if (!Number.isSafeInteger(maximumDimension) || maximumDimension < 1 || dimensions.some(value => value > maximumDimension)) {
    throw new Error(`Scientific volume dimensions ${dimensions.join("×")} exceed the viewer 3D texture limit ${maximumDimension}`);
  }
}
type Scalars = Parameters<Texture["create3DFilterableFromDataArray"]>[0]["dataArray"];
const scientificTextures = new WeakSet<object>();
const textureLayouts = new WeakMap<object, VolumeTextureLayout>();
export function volumeScalarTextureLayout(texture: object | null | undefined): VolumeTextureLayout | null {
  return texture ? textureLayouts.get(texture) ?? null : null;
}
export function isFloat32VolumeTexture(value: object | null | undefined): boolean { return value != null && scientificTextures.has(value); }

export function createFloat32VolumeTexture(window: VolumeTextureWindow, dimensions: readonly number[], scalars: Scalars): VolumeScalarTexture {
  if (dimensions.length !== 3 || dimensions.some(value => !Number.isSafeInteger(value) || value < 1)
      || scalars.getNumberOfComponents() !== 1 || scalars.getDataType() !== "Float32Array") {
    throw new Error("Scientific volume textures require a Float32 scalar lattice");
  }
  const gl = window.getContext();
  const { maximumDimension, floatLinear } = volumeTextureCapabilities(gl);
  const layout = planVolumeTextureLayout(dimensions, maximumDimension);
  if (scalars.getData().length !== layout.sourceCells) throw new Error("Scientific scalar length does not match its logical volume");
  const filter = !layout.tiled && floatLinear ? Filter.LINEAR : Filter.NEAREST;
  const texture = vtkTexture.newInstance({ minificationFilter: filter, magnificationFilter: filter,
    wrapS: Wrap.CLAMP_TO_EDGE, wrapT: Wrap.CLAMP_TO_EDGE, wrapR: Wrap.CLAMP_TO_EDGE }) as VolumeScalarTexture;
  const vtkWindow = window as unknown as Parameters<Texture["setOpenGLRenderWindow"]>[0];
  texture.setOpenGLRenderWindow(vtkWindow);
  texture.enableUseHalfFloat(false);
  try {
    const [width, height, depth] = layout.textureShape;
    if (layout.tiled) {
      if (!texture.create3DFromRaw({ width, height, depth, numComps: 1,
        dataType: vtkDataArray.VtkDataTypes.FLOAT, data: null as unknown as Float32Array, updatedExtents: [] })) {
        throw new Error("Float32 scalar volume texture allocation failed");
      }
      texture.activate(); texture.bind();
      uploadVolumeTexturePrefix(gl, scalars.getData() as Float32Array, layout);
      texture.deactivate();
      const [minimum, maximum] = scalars.getRange(0);
      Object.assign(texture.getVolumeInfo(), { width, height, depth,
        dataComputedOffset: [minimum], dataComputedScale: [maximum - minimum || 1] });
    } else if (!texture.create3DFilterableFromDataArray({ width, height, depth,
      dataArray: scalars, preferSizeOverAccuracy: false })) throw new Error("Float32 scalar volume texture upload failed");
  } catch (error) {
    texture.releaseGraphicsResources(vtkWindow);
    texture.delete();
    throw error;
  }
  scientificTextures.add(texture);
  textureLayouts.set(texture, layout);
  return texture;
}

export const VOLUME_SCALAR_SAMPLING_GLSL = `
${VOLUME_TEXTURE_ADDRESS_GLSL}
uniform vec4 nimbusManualScalarFilter;
uniform vec4 nimbusNearestScalarFilter;
vec4 nimbusScalarTexture(highp sampler3D image, vec3 coordinate, int component) {
  if (nimbusManualScalarFilter[component] < 0.5 && nimbusTiledTexture[component] < 0.5) return texture(image, coordinate);
  ivec3 dimensions = nimbusTiledTexture[component] > 0.5
    ? ivec3(nimbusSourceDimensions[component]) : textureSize(image, 0);
  if (nimbusNearestScalarFilter[component] > 0.5) return nimbusLogicalTexel(image,
    ivec3(clamp(floor(coordinate * vec3(dimensions)), vec3(0.0), vec3(dimensions - ivec3(1)))), dimensions, component);
  vec3 index = clamp(coordinate * vec3(dimensions) - vec3(0.5), vec3(0.0), vec3(dimensions - ivec3(1)));
  ivec3 lower = ivec3(floor(index)), upper = min(lower + ivec3(1), dimensions - ivec3(1));
  vec3 fraction = index - vec3(lower);
  vec4 a = mix(nimbusLogicalTexel(image, lower, dimensions, component), nimbusLogicalTexel(image, ivec3(upper.x, lower.yz), dimensions, component), fraction.x);
  vec4 b = mix(nimbusLogicalTexel(image, ivec3(lower.x, upper.y, lower.z), dimensions, component), nimbusLogicalTexel(image, ivec3(upper.xy, lower.z), dimensions, component), fraction.x);
  vec4 c = mix(nimbusLogicalTexel(image, ivec3(lower.xy, upper.z), dimensions, component), nimbusLogicalTexel(image, ivec3(upper.x, lower.y, upper.z), dimensions, component), fraction.x);
  vec4 d = mix(nimbusLogicalTexel(image, ivec3(lower.x, upper.yz), dimensions, component), nimbusLogicalTexel(image, upper, dimensions, component), fraction.x);
  return mix(mix(a, b, fraction.y), mix(c, d, fraction.y), fraction.z);
}
`;
