import vtkTexture from "@kitware/vtk.js/Rendering/OpenGL/Texture.js";
import { Filter, Wrap } from "@kitware/vtk.js/Rendering/OpenGL/Texture/Constants.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import { planVolumeTextureLayout, uploadVolumeTexturePrefix } from "./volumeTexturePacking";
import { volumeTextureCapabilities } from "./volumeTextureCapabilities";

const masks = new WeakMap<object, Uint8Array>();
export function getVolumeValidity(image: object): Uint8Array | undefined { return masks.get(image); }
export function setVolumeValidity(image: object, validity?: Uint8Array): void {
  if (validity) masks.set(image, validity);
  else masks.delete(image);
}

type Texture = ReturnType<typeof vtkTexture.newInstance>;
type RenderWindow = Parameters<Texture["setOpenGLRenderWindow"]>[0];
const textures = new WeakMap<object, WeakMap<Uint8Array, Map<string, { texture: Texture; users: number }>>>();

export function acquireVolumeValidity(window: RenderWindow, image: object, dimensions: number[]) {
  const mask = masks.get(image);
  if (!mask) return null;
  if (dimensions.length !== 3 || dimensions.some(value => !Number.isSafeInteger(value) || value < 1)
      || dimensions.reduce((count, value) => count * value, 1) !== mask.length) {
    throw new Error("Volume missing-data mask dimensions do not match its image");
  }
  let cache = textures.get(window);
  if (!cache) { cache = new WeakMap(); textures.set(window, cache); }
  let shapes = cache.get(mask);
  if (!shapes) { shapes = new Map(); cache.set(mask, shapes); }
  const shape = dimensions.join(":");
  let entry = shapes.get(shape);
  if (!entry) {
    const gl = window.getContext() as WebGL2RenderingContext;
    const layout = planVolumeTextureLayout(dimensions, volumeTextureCapabilities(gl).maximumDimension);
    const texture = vtkTexture.newInstance({
      minificationFilter: Filter.LINEAR, magnificationFilter: Filter.LINEAR,
      wrapS: Wrap.CLAMP_TO_EDGE, wrapT: Wrap.CLAMP_TO_EDGE, wrapR: Wrap.CLAMP_TO_EDGE,
    });
    texture.setOpenGLRenderWindow(window);
    try {
      const [width, height, depth] = layout.textureShape;
      if (!texture.create3DFromRaw({ width, height, depth,
        numComps: 1, dataType: vtkDataArray.VtkDataTypes.UNSIGNED_CHAR,
        data: layout.tiled ? null as unknown as Uint8Array : mask, updatedExtents: [] })) {
        throw new Error("The volume missing-data mask could not be uploaded");
      }
      if (layout.tiled) {
        texture.activate(); texture.bind();
        uploadVolumeTexturePrefix(gl, mask, layout);
        texture.deactivate();
      }
    } catch (error) {
      texture.releaseGraphicsResources(window); texture.delete();
      throw error;
    }
    entry = { texture, users: 0 };
    shapes.set(shape, entry);
  }
  entry.users++;
  let released = false;
  return { mask, dimensions: [...dimensions], texture: entry.texture, release: () => {
    if (released) return;
    released = true;
    if (--entry!.users === 0) {
      entry!.texture.releaseGraphicsResources(window);
      entry!.texture.delete();
      shapes!.delete(shape);
      if (shapes!.size === 0) cache!.delete(mask);
    }
  } };
}

export const VOLUME_VALIDITY_GLSL = `
uniform vec4 nimbusValidityEnabled;
uniform highp sampler3D nimbusValidityTexture[4];
float nimbusValidityValue(highp sampler3D image, vec3 pos, int component) {
  if (nimbusTiledTexture[component] < 0.5) return texture(image, pos).r;
  ivec3 dimensions = ivec3(nimbusSourceDimensions[component]);
  vec3 index = clamp(pos * vec3(dimensions) - vec3(0.5), vec3(0.0), vec3(dimensions - ivec3(1)));
  ivec3 lower = ivec3(floor(index)), upper = min(lower + ivec3(1), dimensions - ivec3(1));
  vec3 fraction = index - vec3(lower);
  float a = mix(nimbusLogicalTexel(image, lower, dimensions, component).r, nimbusLogicalTexel(image, ivec3(upper.x, lower.yz), dimensions, component).r, fraction.x);
  float b = mix(nimbusLogicalTexel(image, ivec3(lower.x, upper.y, lower.z), dimensions, component).r, nimbusLogicalTexel(image, ivec3(upper.xy, lower.z), dimensions, component).r, fraction.x);
  float c = mix(nimbusLogicalTexel(image, ivec3(lower.xy, upper.z), dimensions, component).r, nimbusLogicalTexel(image, ivec3(upper.x, lower.y, upper.z), dimensions, component).r, fraction.x);
  float d = mix(nimbusLogicalTexel(image, ivec3(lower.x, upper.yz), dimensions, component).r, nimbusLogicalTexel(image, upper, dimensions, component).r, fraction.x);
  return mix(mix(a, b, fraction.y), mix(c, d, fraction.y), fraction.z);
}
${Array.from({ length: 4 }, (_, index) => `
bool nimbusValid${index}(vec3 pos) {
  return nimbusValidityEnabled[${index}] < 0.5 || nimbusValidityValue(nimbusValidityTexture[${index}], pos, ${index}) >= 0.999999;
}`).join("\n")}
bool nimbusValidNormal(vec3 pos) {
  if (nimbusValidityEnabled[0] < 0.5) return true;
  for (int axis = 0; axis < 3; ++axis) {
    vec3 delta = vec3(0.0);
    delta[axis] = volume.inverseDimensions[axis];
    if (!nimbusValid0(pos + delta) || !nimbusValid0(pos - delta)) return false;
  }
  return true;
}
`;
