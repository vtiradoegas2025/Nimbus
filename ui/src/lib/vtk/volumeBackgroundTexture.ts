import vtkTexture from "@kitware/vtk.js/Rendering/OpenGL/Texture.js";
import { Filter, Wrap } from "@kitware/vtk.js/Rendering/OpenGL/Texture/Constants.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import { backgroundAt, backgroundInterior, type VolumeBackground } from "./volumeBackground";
import type { PreparedVtkImage } from "./vtkPreparation";

type Texture = ReturnType<typeof vtkTexture.newInstance>;
type RenderWindow = Parameters<Texture["setOpenGLRenderWindow"]>[0];
const bindings = new WeakMap<object, { data: Float32Array; profile: VolumeBackground; filterValues: boolean; interior: number[]; feather: number[] }>();
const profiles = new WeakMap<VolumeBackground, Map<string, Float32Array>>();
const empty = new Float32Array([0]);
const textures = new WeakMap<object, WeakMap<Float32Array, { texture: Texture; users: number }>>();

export function setVolumeBackground(mapper: object, profile: VolumeBackground | null | undefined, image: PreparedVtkImage, filterValues = true): void {
  if (!profile) { bindings.delete(mapper); return; }
  let cache = profiles.get(profile);
  if (!cache) { cache = new Map(); profiles.set(profile, cache); }
  const key = [image.origin[2], image.spacing[2], image.shape[2]].join(":");
  let data = cache.get(key);
  if (!data) {
    data = Float32Array.from({ length: image.shape[2] }, (_, z) => backgroundAt(profile, image.origin[2] + z * image.spacing[2]));
    cache.set(key, data);
  }
  const interior = backgroundInterior(profile, image)?.map((value, face) => {
    const axis = Math.floor(face / 2);
    return ((value - image.origin[axis]) / image.spacing[axis] + 0.5) / image.shape[axis];
  }) ?? [-1e20, 1e20, -1e20, 1e20];
  const feather = [0, 1].map(axis => (profile.featherM?.[axis] ?? image.spacing[axis]) / (image.shape[axis] * image.spacing[axis]));
  bindings.set(mapper, { data, profile, filterValues, interior, feather });
}

export function volumeBackgroundBinding(mapper: object) { return bindings.get(mapper); }

export function acquireVolumeBackground(window: RenderWindow, mapper: object) {
  const data = bindings.get(mapper)?.data ?? empty;
  let cache = textures.get(window);
  if (!cache) { cache = new WeakMap(); textures.set(window, cache); }
  let entry = cache.get(data);
  if (!entry) {
    const texture = vtkTexture.newInstance({ minificationFilter: Filter.NEAREST, magnificationFilter: Filter.NEAREST,
      wrapS: Wrap.CLAMP_TO_EDGE, wrapT: Wrap.CLAMP_TO_EDGE });
    texture.setOpenGLRenderWindow(window);
    if (!texture.create2DFromRaw({ width: data.length, height: 1, numComps: 1,
      dataType: vtkDataArray.VtkDataTypes.FLOAT, data })) {
      texture.releaseGraphicsResources(window); texture.delete();
      throw new Error("The volume background profile could not be uploaded");
    }
    entry = { texture, users: 0 }; cache.set(data, entry);
  }
  entry.users++;
  let released = false;
  return { data, texture: entry.texture, release: () => {
    if (released) return; released = true;
    if (--entry!.users === 0) { entry!.texture.releaseGraphicsResources(window); entry!.texture.delete(); cache!.delete(data); }
  } };
}

export const VOLUME_BACKGROUND_GLSL = `
uniform highp sampler2D nimbusBackgroundTexture;
uniform vec4 nimbusBackgroundSettings;
uniform vec4 nimbusBackgroundScalar;
uniform vec4 nimbusBackgroundInterior;
uniform vec4 nimbusBackgroundFeather;
float nimbusBackgroundWeight(float scalar, vec3 pos) {
  if (nimbusBackgroundSettings.x < 0.5) return 1.0;
  float edge = smoothstep(0.0, nimbusBackgroundFeather.x, min(pos.x - nimbusBackgroundInterior.x, nimbusBackgroundInterior.y - pos.x))
    * smoothstep(0.0, nimbusBackgroundFeather.y, min(pos.y - nimbusBackgroundInterior.z, nimbusBackgroundInterior.w - pos.y));
  if (edge <= 0.0 || nimbusBackgroundSettings.x > 1.5) return edge;
  int count = textureSize(nimbusBackgroundTexture, 0).x;
  float index = clamp(pos.z * float(count) - 0.5, 0.0, float(count - 1));
  int lower = int(floor(index));
  float background = mix(texelFetch(nimbusBackgroundTexture, ivec2(lower, 0), 0).r,
    texelFetch(nimbusBackgroundTexture, ivec2(min(lower + 1, count - 1), 0), 0).r, fract(index));
  float delta = scalar * nimbusBackgroundScalar.x + nimbusBackgroundScalar.y - background;
  if (nimbusBackgroundSettings.w < 0.5) delta = abs(delta);
  return edge * smoothstep(nimbusBackgroundSettings.y, nimbusBackgroundSettings.z, delta);
}
`;
