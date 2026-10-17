export const VOLUME_OPTICAL_RATIO_GLSL = `
float nimbusRayOpticalRatio(vec3 directionVC, float distance) {
  if (nimbusPhysicalReference <= 0.0) return distance / sampleDistance * nimbusOpacityRatio;
  if (nimbusInverseSceneScale.x == nimbusInverseSceneScale.y && nimbusInverseSceneScale.y == nimbusInverseSceneScale.z) {
    return distance * nimbusInverseSceneScale.x / nimbusPhysicalReference;
  }
  vec3 physicalDirection = (volume.vecVCToISMatrix * normalize(directionVC))
    * vec3(volume.dimensions) * volume.spacing * nimbusInverseSceneScale;
  return distance * length(physicalDirection) / nimbusPhysicalReference;
}
float nimbusReferenceOpacity(float scalar, int component, float height) {
  float scaledScalar = scalar * volume.opacityTextureScale[component] + volume.opacityTextureShift[component];
  return texture2D(opacityTexture, vec2(scaledScalar, height)).r;
}
`;

// Integrate optical depth over the actual light-ray intervals. Shadow stride
// chooses quadrature resolution; it must not change the density of the medium.
export const VOLUME_SHADOW_INTEGRATION_GLSL = `
float computeVolumeShadowWithoutCache(vec3 posVC, vec3 lightDirNormVC) {
  float rayStepLength = max(volumeShadowSampleDistance, sampleDistance);
  float depthPerDistance = nimbusRayOpticalRatio(lightDirNormVC, 1.0);
  float reach = volume.diagonalLength * volume.globalIlluminationReach;
  if (nimbusPhysicalReference > 0.0 && (nimbusInverseSceneScale.x != nimbusInverseSceneScale.y
      || nimbusInverseSceneScale.y != nimbusInverseSceneScale.z)) {
    float physicalDiagonal = length(vec3(volume.dimensions) * volume.spacing * nimbusInverseSceneScale);
    reach = physicalDiagonal * volume.globalIlluminationReach / (depthPerDistance * nimbusPhysicalReference);
  }
  vec2 interval = rayIntersectVolumeDistances(posVC, lightDirNormVC);
  float startDistance = max(interval.x, 0.0);
  float endDistance = min(interval.y, startDistance + reach);
  #ifdef vtkClippingPlanesOn
    for (int plane = 0; plane < clip_numPlanes; ++plane) {
      float offset = dot(posVC - vClipPlaneOrigins[plane], vClipPlaneNormals[plane]);
      float direction = dot(lightDirNormVC, vClipPlaneNormals[plane]);
      if (direction == 0.0) {
        if (offset < 0.0) return 1.0;
      } else {
        float crossing = -offset / direction;
        if (direction < 0.0) endDistance = min(endDistance, crossing);
        else startDistance = max(startDistance, crossing);
      }
    }
  #endif
  if (endDistance <= startDistance) return 1.0;
  vec3 originIS = posVCtoIS(posVC);
  vec3 directionIS = vecVCToIS(lightDirNormVC);
  float opticalDepth = 0.0;
  float current = startDistance;
  for (int step = 0; step < vtkMaximumNumberOfSamples + 2; ++step) {
    if (current >= endDistance) break;
    float length = min(rayStepLength, endDistance - current);
    vec3 posIS = originIS + (current + 0.5 * length) * directionIS;
    if (!nimbusValid0(posIS)) { current += length; continue; }
    vec4 scalar = getTextureValue(posIS);
    float opacity = nimbusReferenceOpacity(scalar.r, 0, 0.5);
    #if defined(EnabledGradientOpacity) && !defined(EnabledIndependentComponents)
      vec3 scalarInterp[2];
      vec4 normal = computeNormalForDensity(posIS, scalarInterp, 3);
      opacity *= computeGradientOpacityFactor(normal.w, 0);
    #endif
    if (opacity >= 1.0) return 0.0;
    opticalDepth -= log(max(1.0 - opacity, 1e-30)) * length * depthPerDistance;
    if (opticalDepth >= 9.210341) return 0.0;
    current += length;
  }
  return exp(-opticalDepth);
}
`;
