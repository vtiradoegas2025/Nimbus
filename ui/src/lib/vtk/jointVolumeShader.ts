import { VOLUME_RAY_PHASE } from "./coherentVolumeSampling";

const rawLayerCode = Array.from({ length: 4 }, (_, index) => `
#if vtkNumberOfComponents > ${index}
float nimbusRawScalar${index}(vec3 pos) {
  return nimbusScalarTexture(volumeTexture[${index}], nimbusLayerToIS[${index}] * (pos - nimbusLayerOriginVC[${index}]), ${index}).r;
}
#endif`).join("\n");
const layerCode = `${rawLayerCode}
bool nimbusRawContains(vec3 pos, int component) {
  vec3 local = nimbusLayerToIS[component] * (pos - nimbusLayerOriginVC[component]);
  return all(greaterThanEqual(local, vec3(0.0))) && all(lessThanEqual(local, vec3(1.0)));
}
float nimbusPhysicalScalar(vec3 pos, int component) {
  ${Array.from({ length: 4 }, (_, i) => `
#if vtkNumberOfComponents > ${i}
  if (component == ${i}) return nimbusRawScalar${i}(pos) * nimbusScalarScales[${i}] + nimbusScalarOffsets[${i}];
#endif`).join("\n")}
  return 0.0;
}
bool nimbusRawValid(vec3 pos, int component) {
  if (!nimbusRawContains(pos, component)) return false;
  vec3 local = nimbusLayerToIS[component] * (pos - nimbusLayerOriginVC[component]);
  ${Array.from({ length: 4 }, (_, i) => `
#if vtkNumberOfComponents > ${i}
  if (component == ${i}) return nimbusValid${i}(local);
#endif`).join("\n")}
  return false;
}
bool nimbusFixedValid(vec3 pos, int component) {
  vec3 local = nimbusFixedToIS[component] * (pos - nimbusFixedOriginVC[component]);
  if (any(lessThan(local, vec3(0.0))) || any(greaterThan(local, vec3(1.0)))) return false;
  ${Array.from({ length: 4 }, (_, i) => `
#if vtkNumberOfComponents > ${i}
  if (component == ${i}) return nimbusValid${i}(local);
#endif`).join("\n")}
  return false;
}
bool nimbusSampleContains(vec3 pos, int component) {
  int peer = int(nimbusTemporalPeers[component]);
  if (peer < 0 || nimbusTemporalFraction >= 1.0) return nimbusRawContains(pos, component);
  if (nimbusTemporalFraction <= 0.0) return nimbusRawContains(pos, peer);
  return nimbusRawContains(pos, component) && nimbusRawContains(pos, peer);
}
bool nimbusSampleValid(vec3 pos, int component) {
  int peer = int(nimbusTemporalPeers[component]);
  if (peer < 0 || nimbusTemporalFraction >= 1.0) return nimbusRawValid(pos, component);
  if (nimbusTemporalFraction <= 0.0) return nimbusRawValid(pos, peer);
  return nimbusRawValid(pos, component) && nimbusRawValid(pos, peer)
    && (nimbusTemporalMasked[component] < 0.5 || nimbusFixedValid(pos, component))
    && (nimbusTemporalMasked[peer] < 0.5 || nimbusFixedValid(pos, peer));
}
${Array.from({ length: 4 }, (_, i) => `
#if vtkNumberOfComponents > ${i}
float nimbusScalar${i}(vec3 pos) {
  int peer = int(nimbusTemporalPeers[${i}]);
  if (peer < 0 || nimbusTemporalFraction >= 1.0) return nimbusRawScalar${i}(pos);
  float previous = nimbusPhysicalScalar(pos, peer);
  float current = nimbusPhysicalScalar(pos, ${i});
  return (mix(previous, current, nimbusTemporalFraction) - nimbusScalarOffsets[${i}]) / max(nimbusScalarScales[${i}], 1e-30);
}
#endif`).join("\n")}`;

const coefficients = Array.from({ length: 4 }, (_, index) => `
#if vtkNumberOfComponents > ${index}
  if (nimbusRoles[${index}] != 4.0 && nimbusSampleValid(pos, ${index})) {
    float role = nimbusRoles[${index}];
    if (!(role == 0.0 && primaryOwns)) {
      float opacity = nimbusReferenceOpacity(nimbusScalar${index}(pos), ${index}, volume.transferFunctionsSampleHeight[${index}]);
      coefficients[${index}] = (opacity >= 1.0 ? 1e30 : -log(max(1.0 - opacity, 1e-30))) / nimbusReferences[${index}];
      if (role == 0.0 || role == 1.0) cloud = ${index};
      if (role == 2.0) condensation = ${index};
    }
  }
#endif`).join("\n");

const normalAndColor = Array.from({ length: 4 }, (_, index) => `
#if vtkNumberOfComponents > ${index}
  if (coefficients[${index}] > 0.0) {
    vec3 color = getColorFromTexture(nimbusScalar${index}(posVC), ${index}, volume.transferFunctionsSampleHeight[${index}]);
    vec4 normal = vec4(0.0);
    if (nimbusLighting[${index}].x > 0.5) {
      if (nimbusLighting[${index}].z < 1.0) {
        vec3 gradient;
        float centerScalar = nimbusScalar${index}(posVC);
        if (nimbusLighting[${index}].y > 0.5) centerScalar = getOpacityFromTexture(centerScalar, ${index}, volume.transferFunctionsSampleHeight[${index}]);
        bool gradientSupported = true;
        for (int axis = 0; axis < 3; ++axis) {
          vec3 offset = nimbusLayerStepVC[${index}][axis];
          bool aboveValid = nimbusSampleValid(posVC + offset, ${index});
          bool belowValid = nimbusSampleValid(posVC - offset, ${index});
          if (!aboveValid && !belowValid) { gradient[axis] = 0.0; gradientSupported = false; continue; }
          float above = aboveValid ? nimbusScalar${index}(posVC + offset) : centerScalar;
          float below = belowValid ? nimbusScalar${index}(posVC - offset) : centerScalar;
          if (nimbusLighting[${index}].y > 0.5) {
            if (aboveValid) above = getOpacityFromTexture(above, ${index}, volume.transferFunctionsSampleHeight[${index}]);
            if (belowValid) below = getOpacityFromTexture(below, ${index}, volume.transferFunctionsSampleHeight[${index}]);
          }
          gradient[axis] = (above - below) / (nimbusLayerSpacing[${index}][axis] * (aboveValid && belowValid ? 2.0 : 1.0));
        }
        if (!gradientSupported) gradient = vec3(0.0);
        normal.w = length(gradient);
        if (normal.w > 0.0) normal.xyz = normalize(nimbusLayerNormalVC[${index}] * gradient);
      }
      color = nimbusLight(color, normal, posVC, ${index}, 1.0 - exp(-coefficients[${index}] * nimbusPrimaryRatio));
    }
    colors[${index}] = color;
  }
#endif`).join("\n");

export const JOINT_VOLUME_DECLARATIONS = `
uniform vec3 nimbusLayerOriginVC[4], nimbusLayerSpacing[4];
uniform mat3 nimbusLayerToIS[4], nimbusLayerStepVC[4];
vec2 nimbusInterval(vec3 origin, vec3 direction, int component);
uniform vec4 nimbusReferences, nimbusRoles, nimbusReach, nimbusShadowSteps, nimbusSampleSteps;
uniform vec4 nimbusSurface[4], nimbusLighting[4];
uniform vec4 nimbusTemporalPeers, nimbusScalarScales, nimbusScalarOffsets, nimbusTemporalMasked;
uniform float nimbusTemporalFraction;
uniform vec3 nimbusFixedOriginVC[4];
uniform mat3 nimbusFixedToIS[4], nimbusLayerNormalVC[4];
bool nimbusTemporalLit[4];
vec3 nimbusTemporalIncident[4];
`;

export const JOINT_VOLUME_SAMPLERS = `${layerCode}
vec4 rawSampleTexture(vec3 pos) {
  vec4 value = vec4(0.0);
  ${Array.from({ length: 4 }, (_, i) => `\n#if vtkNumberOfComponents > ${i}\n  value[${i}] = nimbusScalar${i}(pos);\n#endif`).join("")}
  return value;
}`;

export const JOINT_VOLUME_FUNCTIONS = `
bool nimbusContains(vec3 pos, int component) {
  vec3 local = nimbusLayerToIS[component] * (pos - nimbusLayerOriginVC[component]);
  return all(greaterThanEqual(local, vec3(0.0))) && all(lessThanEqual(local, vec3(1.0)));
}
vec2 nimbusInterval(vec3 origin, vec3 direction, int component) {
  origin = nimbusLayerToIS[component] * (origin - nimbusLayerOriginVC[component]);
  direction = nimbusLayerToIS[component] * direction;
  vec2 interval = vec2(-infinity, infinity);
  for (int axis = 0; axis < 3; ++axis) {
    if (direction[axis] == 0.0) {
      if (origin[axis] < 0.0 || origin[axis] > 1.0) return vec2(1.0, -1.0);
    } else {
      float a = -origin[axis] / direction[axis], b = (1.0 - origin[axis]) / direction[axis];
      interval.x = max(interval.x, min(a,b)); interval.y = min(interval.y, max(a,b));
    }
  }
  return interval;
}
vec4 nimbusCoefficients(vec3 pos, out int cloud, out int condensation) {
  vec4 coefficients = vec4(0.0);
  cloud = -1; condensation = -1;
  bool primaryOwns = false;
  for (int i = 0; i < vtkNumberOfComponents; ++i) {
    if (nimbusRoles[i] == 1.0 && nimbusSampleContains(pos, i)) primaryOwns = true;
  }
  ${coefficients}
  return coefficients;
}
float nimbusExtinction(vec3 pos) {
  int cloud, condensation;
  vec4 coefficients = nimbusCoefficients(pos, cloud, condensation);
  if (cloud >= 0 && condensation >= 0) {
    coefficients[cloud] = max(coefficients[cloud], coefficients[condensation]);
    coefficients[condensation] = 0.0;
  }
  return dot(coefficients, vec4(1.0));
}
float nimbusNextBoundary(float current, float end, vec2 intervals[4]) {
  for (int i = 0; i < vtkNumberOfComponents; ++i) {
    if (intervals[i].y <= intervals[i].x) continue;
    if (intervals[i].x > current) end = min(end, intervals[i].x);
    if (intervals[i].y > current) end = min(end, intervals[i].y);
  }
  return end;
}
float nimbusShadow(vec3 posVC, vec3 lightDirection, int component) {
  float physicalPerDisplay = nimbusRayOpticalRatio(lightDirection, 1.0);
  vec2 unionInterval = rayIntersectVolumeDistances(posVC, lightDirection);
  float current = max(0.0, unionInterval.x);
  float end = min(unionInterval.y, nimbusReach[component] / physicalPerDisplay);
  if (end <= current) return 1.0;
  vec3 origin = posVC, direction = lightDirection;
  vec2 intervals[4];
  for (int i = 0; i < vtkNumberOfComponents; ++i) intervals[i] = nimbusInterval(origin, direction, i);
  float depth = 0.0;
  for (int region = 0; region < 9; ++region) {
    if (current >= end) break;
    float regionEnd = nimbusNextBoundary(current, end, intervals);
    float midpoint = 0.5 * (current + regionEnd);
    float physicalStep = nimbusShadowSteps[component];
    for (int i = 0; i < vtkNumberOfComponents; ++i) {
      if (intervals[i].x <= midpoint && intervals[i].y >= midpoint) physicalStep = min(physicalStep, nimbusShadowSteps[i]);
    }
    float stride = physicalStep / physicalPerDisplay;
    for (int step = 0; step < vtkMaximumNumberOfSamples + 2; ++step) {
      if (current >= regionEnd) break;
      float next = min(regionEnd, current + stride);
      depth += nimbusExtinction(origin + 0.5 * (current + next) * direction) * (next - current) * physicalPerDisplay;
      if (depth >= 9.210341) return 0.0;
      current = next;
    }
  }
  return exp(-depth);
}
vec3 nimbusLight(vec3 color, vec4 normal, vec3 posVC, int component, float alpha) {
  #if vtkNumberOfLights > 0
    float blend = nimbusLighting[component].z;
    vec3 shaded = color;
    if (blend < 1.0) {
      blend *= (1.0 - alpha / 2.0) * (1.0 - atan(normal.w) * INV4PI);
      vec3 surface = applySurfaceShadowLighting(vec3(1.0), alpha, posVC, normal);
      shaded = color * (surface * nimbusSurface[component].y + nimbusSurface[component].x);
    }
    if (blend > 0.0) {
      vec3 incident = vec3(0.0);
      bool cacheLighting = nimbusTemporalPeers[component] >= 0.0;
      int lightSlot = nimbusRoles[component] <= 1.0 ? 0 : component;
      if (cacheLighting && nimbusTemporalLit[lightSlot]) {
        incident = nimbusTemporalIncident[lightSlot];
      } else {
      float g = nimbusLighting[component].w;
      for (int light = 0; light < vtkNumberOfLights; ++light) {
        vec3 direction = lights[light].isPositional == 1 ? normalize(lights[light].positionVC - posVC) : -lights[light].directionVC;
        float cosine = dot(rayDirVC, direction);
        float phase = (1.0 - g*g) / (2.0 * pow(max(1.0 + g*g - 2.0*g*cosine, 1e-6), 1.5));
        incident += phase * nimbusShadow(posVC, direction, component) * lights[light].color;
      }
      if (cacheLighting) { nimbusTemporalLit[lightSlot] = true; nimbusTemporalIncident[lightSlot] = incident; }
      }
      shaded = mix(shaded, color * (incident * nimbusSurface[component].y + nimbusSurface[component].x), blend);
    }
    return shaded;
  #else
    return color;
  #endif
}
vec4 getColorAtPos(vec3 posVC) {
  int cloud, condensation;
  vec4 coefficients = nimbusCoefficients(posVC, cloud, condensation);
  vec3 colors[4];
  for (int i = 0; i < 4; ++i) colors[i] = vec3(0.0);
  ${normalAndColor}
  if (cloud >= 0 && condensation >= 0) {
    float strongest = max(coefficients[cloud], coefficients[condensation]);
    if (strongest > 0.0) colors[cloud] = mix(colors[cloud], colors[condensation], coefficients[condensation] / strongest);
    coefficients[cloud] = strongest; coefficients[condensation] = 0.0;
  }
  float total = dot(coefficients, vec4(1.0));
  if (total <= 0.0) return vec4(0.0);
  vec3 color = vec3(0.0);
  for (int i = 0; i < vtkNumberOfComponents; ++i) color += colors[i] * (coefficients[i] / total);
  return vec4(color, 1.0 - exp(-min(total * nimbusPrimaryRatio, 80.0)));
}
`;

export function configureJointVolumeShader(value: object): void {
  const mapper = value as { getViewSpecificProperties(): Record<string, unknown>; setViewSpecificProperties(value: object): void };
  const properties = mapper.getViewSpecificProperties();
  const gl = properties.OpenGL as { ShaderReplacements: object[] };
  mapper.setViewSpecificProperties({ ...properties, OpenGL: { ...gl,
    ShaderReplacements: [...gl.ShaderReplacements, {
      shaderType: "Fragment", originalValue: "//VTK::System::Dec", replacementValue: `//VTK::System::Dec\n${JOINT_VOLUME_DECLARATIONS}`, replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "vec4 rawSampleTexture\\(vec3 pos\\) \\{[\\s\\S]*?\\n\\}", replacementValue: JOINT_VOLUME_SAMPLERS, replaceFirst: true, replaceAll: true,
    }, {
      shaderType: "Fragment", originalValue: "vec4 getColorAtPos\\(vec3 posVC\\) \\{[\\s\\S]*?\\n\\}", replacementValue: JOINT_VOLUME_FUNCTIONS, replaceFirst: true, replaceAll: true,
    }, {
      shaderType: "Fragment", originalValue: "nimbusPrimaryRatio);", replacementValue: "nimbusPrimaryRatio / nimbusReferences[component]);", replaceFirst: true, replaceAll: false,
    }, {
      shaderType: "Fragment", originalValue: "vec2 rayIntersectVolumeDistances\\(vec3 rayOriginVC, vec3 rayDirVC\\) \\{[\\s\\S]*?return interval;\\n      \\}",
      replacementValue: `vec2 rayIntersectVolumeDistances(vec3 rayOriginVC, vec3 rayDirVC) {
        vec2 result = vec2(infinity, -infinity);
        for (int i = 0; i < vtkNumberOfComponents; ++i) {
          vec2 interval = nimbusInterval(rayOriginVC, rayDirVC, i);
          if (interval.y > interval.x) { result.x = min(result.x, interval.x); result.y = max(result.y, interval.y); }
        }
        return result;
      }`, replaceFirst: true, replaceAll: true,
    }, {
      shaderType: "Fragment", originalValue: "#if vtkBlendMode == COMPOSITE_BLEND[\\s\\S]*?#if vtkBlendMode == COMPOSITE_BLEND && 0",
      replacementValue: `#if vtkBlendMode == COMPOSITE_BLEND
        ${VOLUME_RAY_PHASE}
        vec2 intervals[4];
        for (int i = 0; i < vtkNumberOfComponents; ++i) intervals[i] = nimbusInterval(rayOriginVC, rayDirVC, i);
        for (int i = 0; i < 4; ++i) { nimbusTemporalLit[i] = false; nimbusTemporalIncident[i] = vec3(0.0); }
        vec4 accumulated = vec4(0.0);
        float physicalPerDisplay = nimbusRayOpticalRatio(rayDirVC, 1.0);
        float regionStart = minDistance;
        float originDistance = dot(rayOriginVC, rayDirVC);
        for (int region = 0; region < 9; ++region) {
          if (regionStart >= maxDistance || accumulated.a >= 0.9999) break;
          float regionEnd = nimbusNextBoundary(regionStart, maxDistance, intervals);
          float midpoint = 0.5 * (regionStart + regionEnd);
          float stride = infinity;
          for (int i = 0; i < vtkNumberOfComponents; ++i) {
            if (intervals[i].x <= midpoint && intervals[i].y >= midpoint) stride = min(stride, nimbusSampleSteps[i]);
          }
          if (stride < infinity) {
            float binStart = (floor((regionStart + originDistance) / stride - phase) + phase) * stride - originDistance;
            for (int sampleIndex = 0; sampleIndex < vtkMaximumNumberOfSamples + 2; ++sampleIndex) {
              float begin = max(regionStart, binStart), end = min(regionEnd, binStart + stride);
              if (end > begin) {
                nimbusPrimaryRatio = (end - begin) * physicalPerDisplay;
                vec4 value = getColorAtPos(rayOriginVC + 0.5 * (begin + end) * rayDirVC);
                float alpha = value.a;
                accumulated += (1.0 - accumulated.a) * vec4(value.rgb * alpha, alpha);
              }
              if (end >= regionEnd || accumulated.a >= 0.9999) break;
              binStart += stride;
            }
          }
          regionStart = regionEnd;
        }
        gl_FragData[0] = accumulated.a > 0.0 ? vec4(accumulated.rgb / accumulated.a, accumulated.a) : vec4(0.0);
        return;
      #endif
      #if vtkBlendMode == COMPOSITE_BLEND && 0`, replaceFirst: true, replaceAll: true,
    }],
  } });
}
