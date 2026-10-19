import type { VolumeTextureLayout } from "./volumeTexturePacking";

function interpolation(fetch: (index: string) => string): string {
  return `
  vec4 a = mix(${fetch("lower")}, ${fetch("ivec3(upper.x, lower.yz)")}, fraction.x);
  vec4 b = mix(${fetch("ivec3(lower.x, upper.y, lower.z)")}, ${fetch("ivec3(upper.xy, lower.z)")}, fraction.x);
  vec4 c = mix(${fetch("ivec3(lower.xy, upper.z)")}, ${fetch("ivec3(upper.x, lower.y, upper.z)")}, fraction.x);
  vec4 d = mix(${fetch("ivec3(lower.x, upper.yz)")}, ${fetch("upper")}, fraction.x);
  return mix(mix(a, b, fraction.y), mix(c, d, fraction.y), fraction.z);`;
}

interface VolumeSamplingOptions { validity?: readonly boolean[]; nearest?: readonly boolean[] }

export function volumeSamplingShaderKey(layouts: readonly (VolumeTextureLayout | null)[],
  manualFilter: readonly boolean[], options: VolumeSamplingOptions = {}): string {
  return JSON.stringify({ layouts: layouts.map((layout, index) => layout?.tiled
    ? [layout.sourceShape, layout.textureShape] : Boolean(manualFilter[index])),
    validity: options.validity ?? null, nearest: options.nearest ?? null });
}

export class VolumeSamplingShaderMemo {
  private previous: ReturnType<typeof volumeSamplingShader> | null = null;

  get(layouts: readonly (VolumeTextureLayout | null)[], manualFilter: readonly boolean[], options: VolumeSamplingOptions) {
    const key = volumeSamplingShaderKey(layouts, manualFilter, options);
    if (this.previous?.key !== key) this.previous = volumeSamplingShader(layouts, manualFilter, options);
    return this.previous;
  }
}

export function volumeSamplingShader(layouts: readonly (VolumeTextureLayout | null)[],
  manualFilter: readonly boolean[] = layouts.map(() => true),
  options: VolumeSamplingOptions = {}) {
  const tiled = layouts.some(layout => layout?.tiled);
  const runtimeNearest = tiled && options.nearest == null;
  const needsNativeManual = layouts.some((layout, index) => !layout?.tiled && manualFilter[index]);
  const scalar = `
${runtimeNearest ? "uniform vec4 nimbusNearestScalarFilter;" : ""}
${needsNativeManual ? `vec4 nimbusNativeScalar(highp sampler3D image, vec3 coordinate) {
  ivec3 dimensions = textureSize(image, 0);
  vec3 index = clamp(coordinate * vec3(dimensions) - vec3(0.5), vec3(0.0), vec3(dimensions - ivec3(1)));
  ivec3 lower = ivec3(floor(index)), upper = min(lower + ivec3(1), dimensions - ivec3(1));
  vec3 fraction = index - vec3(lower);
  ${interpolation(index => `texelFetch(image, ${index}, 0)`)}
}` : ""}
${layouts.map((layout, component) => {
    if (!layout?.tiled) return "";
    const [width, height] = layout.textureShape;
    const fetch = (index: string) => `nimbusPackedTexel${component}(image, ${index})`;
    const nearest = options.nearest?.[component];
    const sampleNeeded = nearest !== true || options.validity?.[component] !== false;
    const nearestFetch = fetch(`ivec3(clamp(floor(coordinate * vec3(${layout.sourceShape.join(",")})), vec3(0.0), vec3(${layout.sourceShape.map(v => v - 1).join(",")})))`);
    return `
vec4 nimbusPackedTexel${component}(highp sampler3D image, ivec3 index) {
  int ordinal = index.x + ${layout.sourceShape[0]} * (index.y + ${layout.sourceShape[1]} * index.z);
  return texelFetch(image, ivec3(ordinal % ${width}, (ordinal / ${width}) % ${height}, ordinal / ${width * height}), 0);
}
${sampleNeeded ? `vec4 nimbusPackedSample${component}(highp sampler3D image, vec3 coordinate) {
  ivec3 dimensions = ivec3(${layout.sourceShape.join(",")});
  vec3 index = clamp(coordinate * vec3(dimensions) - vec3(0.5), vec3(0.0), vec3(dimensions - ivec3(1)));
  ivec3 lower = ivec3(floor(index)), upper = min(lower + ivec3(1), dimensions - ivec3(1));
  vec3 fraction = index - vec3(lower);
  ${interpolation(fetch)}
}` : ""}
vec4 nimbusPackedScalar${component}(highp sampler3D image, vec3 coordinate) {
  ${nearest === true ? `return ${nearestFetch};` : nearest === false
    ? `return nimbusPackedSample${component}(image, coordinate);`
    : `if (nimbusNearestScalarFilter[${component}] > 0.5) return ${nearestFetch};\n  return nimbusPackedSample${component}(image, coordinate);`}
}`;
  }).join("\n")}
vec4 nimbusScalarTexture(highp sampler3D image, vec3 coordinate, int component) {
  ${layouts.map((layout, component) => layout?.tiled
    ? `if (component == ${component}) return nimbusPackedScalar${component}(image, coordinate);`
    : manualFilter[component] ? `if (component == ${component}) return nimbusNativeScalar(image, coordinate);` : "").join("\n")}
  return texture(image, coordinate);
}
`;
  const validity = `${options.validity == null || options.validity.some(Boolean) ? `uniform vec4 nimbusValidityEnabled;
uniform highp sampler3D nimbusValidityTexture[4];` : ""}
${Array.from({ length: 4 }, (_, component) => {
    if (options.validity != null && !options.validity[component]) return `bool nimbusValid${component}(vec3 pos) { return true; }`;
    const layout = layouts[component]?.tiled ? layouts[component]! : null;
    return `
bool nimbusValid${component}(vec3 pos) {
  if (nimbusValidityEnabled[${component}] < 0.5) return true;
  ${layout ? `return nimbusPackedSample${component}(nimbusValidityTexture[${component}], pos).r >= 0.999999;`
    : `return texture(nimbusValidityTexture[${component}], pos).r >= 0.999999;`}
}`;
  }).join("\n")}
bool nimbusValidNormal(vec3 pos) {
  ${options.validity != null && !options.validity[0] ? "return true;" : `
  if (nimbusValidityEnabled[0] < 0.5) return true;
  for (int axis = 0; axis < 3; ++axis) {
    vec3 delta = vec3(0.0); delta[axis] = volume.inverseDimensions[axis];
    if (!nimbusValid0(pos + delta) || !nimbusValid0(pos - delta)) return false;
  }
  return true;`}
}
`;
  return { scalar, validity, key: volumeSamplingShaderKey(layouts, manualFilter, options) };
}
