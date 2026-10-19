import { installPhysicalVolumeOpacity } from "./vtkPhysicalVolumeOpacity";
import { VOLUME_OPTICAL_RATIO_GLSL, VOLUME_SHADOW_INTEGRATION_GLSL } from "./volumeShadowSampling";

export const VOLUME_RAY_PHASE = `
    float phase;
    #if __VERSION__ >= 300
      highp uvec2 pixel = uvec2(gl_FragCoord.xy);
      highp uint hash = pixel.x * 1664525u + pixel.y * 1013904223u;
      hash = (hash ^ (hash >> 16u)) * 2246822519u;
      hash = (hash ^ (hash >> 13u)) * 3266489917u;
      hash = hash ^ (hash >> 16u);
      phase = (float(hash >> 8u) + 0.5) / 16777216.0;
    #else
      vec3 hash = fract(vec3(gl_FragCoord.xyx) * vec3(0.1031, 0.1030, 0.0973));
      hash += dot(hash, hash.yxz + 33.33);
      phase = fract((hash.x + hash.y) * hash.z);
    #endif
`;

// Shared bins make clipping a volume into bricks an optical partition: both
// sides sample the same reconstructed field and weight only their owned length.
const COMPOSITE = `
  #if vtkBlendMode == COMPOSITE_BLEND
    ${VOLUME_RAY_PHASE}
    float originDistance = dot(rayOriginVC, rayDirVC);
    float binStart = (floor((minDistance + originDistance) / sampleDistance - phase) + phase) * sampleDistance - originDistance;
    vec4 accumulated = vec4(0.0);
    for (int index = 0; index < vtkMaximumNumberOfSamples + 2; ++index) {
      float begin = max(minDistance, binStart);
      float end = min(maxDistance, binStart + sampleDistance);
      if (end > begin) {
        vec3 position = rayOriginVC + (binStart + 0.5 * sampleDistance) * rayDirVC;
        vec4 value = getColorAtPos(position);
        float fraction = (end - begin) / sampleDistance;
        float alpha = fraction == 1.0 ? value.a : 1.0 - pow(max(0.0, 1.0 - value.a), fraction);
        accumulated += (1.0 - accumulated.a) * vec4(value.rgb * alpha, alpha);
      }
      if (end >= maxDistance || accumulated.a >= 0.9999) break;
      binStart += sampleDistance;
    }
    gl_FragData[0] = accumulated.a > 0.0
      ? vec4(accumulated.rgb / accumulated.a, accumulated.a) : vec4(0.0);
    return;
  #endif
  #if vtkBlendMode == COMPOSITE_BLEND && 0
`;

const NORMALS = ["result.xyz", "opacityG.xyz", "normal"].map((normal) => ({
  shaderType: "Fragment",
  originalValue: `${normal} = normalize(vecISToVC(${normal}));`.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  replacementValue: `${normal} = normalize(vecISToVC(${normal} * volume.inverseDimensions * volume.inverseSpacing));`,
  replaceFirst: true,
  replaceAll: true,
}));

function appendReplacements(value: object, replacements: object[]): void {
  installPhysicalVolumeOpacity();
  const mapper = value as {
    getViewSpecificProperties(): Record<string, unknown>;
    setViewSpecificProperties(value: object): unknown;
  };
  const properties = mapper.getViewSpecificProperties();
  const openGL = (properties.OpenGL ?? {}) as { ShaderReplacements?: object[] };
  // Interpolated origins on different brick quads can disagree at shared
  // pixels. Reconstruct from the common camera matrix before ownership tests.
  mapper.setViewSpecificProperties({ ...properties, OpenGL: {
    ...openGL, nimbusPhysicalOpacity: true,
    ShaderReplacements: [...(openGL.ShaderReplacements ?? []), {
      shaderType: "Fragment", originalValue: "vertexVCVSOutput", replacementValue: "nimbusRayOriginVC",
      replaceFirst: true, replaceAll: true,
    }, {
      shaderType: "Fragment", originalValue: "in vec3 nimbusRayOriginVC;",
      replacementValue: "vec3 nimbusRayOriginVC;\nuniform mat4 PCVCMatrix;",
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "void main() {",
      replacementValue: `void main() {
        vec4 nimbusOrigin = PCVCMatrix * fragCoordToPCPos(gl_FragCoord);
        nimbusRayOriginVC = nimbusOrigin.xyz / nimbusOrigin.w;`,
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "//VTK::System::Dec",
      replacementValue: "//VTK::System::Dec\nuniform float nimbusOpacityRatio;\nuniform float nimbusPhysicalReference;\nuniform vec3 nimbusInverseSceneScale;\nfloat nimbusPrimaryRatio;",
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment",
      originalValue: "vec2 rayIntersectVolumeDistances\\(vec3 rayOriginVC, vec3 rayDirVC\\) \\{[\\s\\S]*?\\n\\}",
      replacementValue: `vec2 rayIntersectVolumeDistances(vec3 rayOriginVC, vec3 rayDirVC) {
        vec3 origin = posVCtoIS(rayOriginVC);
        vec3 direction = vecVCToIS(rayDirVC);
        vec2 interval = vec2(-infinity, infinity);
        for (int axis = 0; axis < 3; ++axis) {
          if (direction[axis] == 0.0) {
            if (origin[axis] < 0.0 || origin[axis] > 1.0) return vec2(1.0, -1.0);
          } else {
            float first = -origin[axis] / direction[axis];
            float last = (1.0 - origin[axis]) / direction[axis];
            interval.x = max(interval.x, min(first, last));
            interval.y = min(interval.y, max(first, last));
          }
        }
        return interval;
      }`,
      replaceFirst: true, replaceAll: true,
    }, {
      shaderType: "Fragment",
      originalValue: "return texture2D(opacityTexture, vec2(scaledScalar, height)).r;",
      replacementValue: "float opacity = texture2D(opacityTexture, vec2(scaledScalar, height)).r;\n  return 1.0 - pow(max(0.0, 1.0 - opacity), nimbusPrimaryRatio);",
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "float getOpacityFromTexture(float scalar, int component, float height) {",
      replacementValue: `${VOLUME_OPTICAL_RATIO_GLSL}\nfloat getOpacityFromTexture(float scalar, int component, float height) {`,
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "vec3 rayOriginVC = nimbusRayOriginVC;",
      replacementValue: "nimbusPrimaryRatio = nimbusRayOpticalRatio(rayDirVC, sampleDistance);\n  vec3 rayOriginVC = nimbusRayOriginVC;",
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "vec4 opacityG;", replacementValue: "vec4 opacityG = vec4(0.0);",
      replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment",
      originalValue: "float computeVolumeShadowWithoutCache\\(vec3 posVC, vec3 lightDirNormVC\\) \\{[\\s\\S]*?\\n  \\}",
      replacementValue: VOLUME_SHADOW_INTEGRATION_GLSL,
      replaceFirst: true, replaceAll: true,
    }, {
      shaderType: "Fragment", originalValue: "if (alpha < EPSILON) {",
      replacementValue: "if (alpha <= 0.0) {", replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "vec4 normalLight;",
      replacementValue: "vec4 normalLight = normal0;", replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "gl_FragData[0] = vec4(color.rgb / color.a, color.a);",
      replacementValue: "gl_FragData[0] = color.a > 0.0 ? vec4(color.rgb / color.a, color.a) : vec4(0.0);",
      replaceFirst: true, replaceAll: false,
    }, ...replacements],
  } });
}

export function configurePhysicalVolumeNormals(value: object): void {
  // VTK has already divided gradients by physical spacing, but its
  // image-to-view vector matrix still includes the full texture extent.
  appendReplacements(value, NORMALS);
}

export function configureCoherentVolumeSampling(value: object, ownershipClipping = false,
  openUpperFaces: [number, number, number] = [0, 0, 0]): void {
  appendReplacements(value, [...NORMALS, {
    shaderType: "Fragment",
    originalValue: "#if vtkBlendMode == COMPOSITE_BLEND",
    replacementValue: COMPOSITE,
    replaceFirst: true,
    replaceAll: false,
  }, ...(ownershipClipping ? [{
    shaderType: "Fragment",
    originalValue: "isPointClipped\\(posIStoVC\\(offsetedPosIS\\)\\)",
    replacementValue: "false",
    replaceFirst: true,
    replaceAll: true,
  }, {
    shaderType: "Fragment", originalValue: "//VTK::System::Dec",
    replacementValue: "//VTK::System::Dec\nuniform vec3 nimbusOpenUpperFaces;",
    replaceFirst: true, replaceAll: false,
  }, {
    shaderType: "Fragment", originalValue: "//VTK::ClipPlane::Impl",
    replacementValue: `
      for (int plane = 0; plane < 6; ++plane) {
        float direction = dot(rayDirVC, vClipPlaneNormals[plane]);
        float distance = dot(nimbusRayOriginVC - vClipPlaneOrigins[plane], vClipPlaneNormals[plane]);
        if (direction == 0.0) {
          float tolerance = 8.0 * 1.1920929e-7 * max(length(nimbusRayOriginVC), 1.0);
          if (abs(distance) <= tolerance) {
            if (plane % 2 == 1 && nimbusOpenUpperFaces[plane / 2] > 0.5) dists.x = dists.y;
          } else if (distance < 0.0) dists.x = dists.y;
        } else {
          float intersection = -distance / direction;
          if (direction < 0.0) dists.y = min(dists.y, intersection);
          else dists.x = max(dists.x, intersection);
        }
      }`,
    replaceFirst: true, replaceAll: false,
  }] : [])]);
  if (ownershipClipping) {
    const mapper = value as { getViewSpecificProperties(): Record<string, unknown>; setViewSpecificProperties(value: object): void };
    const properties = mapper.getViewSpecificProperties();
    mapper.setViewSpecificProperties({ ...properties, OpenGL: {
      ...(properties.OpenGL as object), nimbusOpenUpperFaces: openUpperFaces,
    } });
  }
}
