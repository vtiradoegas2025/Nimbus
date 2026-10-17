/**
 * Convert a physical opacity reference distance into the display-space
 * distance expected by vtk.js after an anisotropic actor scale is applied.
 *
 * vtk.js measures ray length after the actor transform. Without this
 * correction, vertical exaggeration makes the same atmospheric column more
 * opaque merely because it occupies more display-space metres.
 */
export function displayDistanceForPhysicalRay(
  physicalDistance: number,
  scale: { xy: number; z: number },
  displayRayDirection: readonly number[],
): number {
  if (!(physicalDistance > 0) || !Number.isFinite(physicalDistance)) return 1;
  return physicalDistance * displayScaleAlongRay(scale, displayRayDirection);
}

/** Display-space stretch of a physical ray under the actor's xyz scale. */
export function displayScaleAlongRay(
  scale: { xy: number; z: number },
  displayRayDirection: readonly number[],
): number {
  const xy = Number.isFinite(scale.xy) && scale.xy > 0 ? scale.xy : 1;
  const z = Number.isFinite(scale.z) && scale.z > 0 ? scale.z : 1;
  if (xy === z) return xy;
  const dx = displayRayDirection[0] ?? 0;
  const dy = displayRayDirection[1] ?? 0;
  const dz = displayRayDirection[2] ?? 0;
  const norm = Math.hypot(dx, dy, dz);
  if (!(norm > 0) || !Number.isFinite(norm)) return 1;

  // A unit display-space ray maps back through the inverse actor scale. The
  // reciprocal of that physical length is the display stretch along the ray.
  const physicalPerDisplay = Math.hypot(
    dx / norm / xy,
    dy / norm / xy,
    dz / norm / z,
  );
  if (!(physicalPerDisplay > 0) || !Number.isFinite(physicalPerDisplay)) {
    return 1;
  }
  return 1 / physicalPerDisplay;
}

interface SampleDistanceControllerLike {
  setFullSampleDistanceFor: (mapper: VolumeSampleMapperLike, distance: number) => unknown;
}

interface VolumeSampleMapperLike {
  setSampleDistance: (distance: number) => unknown;
  setMaximumSamplesPerRay: (samples: number) => unknown;
  getViewSpecificProperties?: () => Record<string, unknown>;
  setViewSpecificProperties?: (properties: Record<string, unknown>) => unknown;
}

/**
 * Keep both ray sampling and opacity correction in physical units. Adjusting
 * opacity alone fixes density but leaves vtk's shader loop too short after an
 * actor stretch; adjusting both preserves physical samples per cell as well.
 */
export function setCameraAdjustedVolumeRayDistances(
  property: OpacityPropertyLike,
  camera: DirectionCameraLike,
  quality: SampleDistanceControllerLike,
  physicalOpacityDistance: number,
  physicalSampleDistance: number,
  scale: { xy: number; z: number },
  mapper: VolumeSampleMapperLike,
): { opacityDistance: number; sampleDistance: number } {
  const direction = camera.getDirectionOfProjection();
  const opacityDistance = displayDistanceForPhysicalRay(
    physicalOpacityDistance,
    scale,
    direction,
  );
  const sampleDistance = displayDistanceForPhysicalRay(
    physicalSampleDistance,
    scale,
    direction,
  );
  property.setScalarOpacityUnitDistance(0, opacityDistance);
  if (mapper.getViewSpecificProperties && mapper.setViewSpecificProperties) {
    const properties = mapper.getViewSpecificProperties();
    const openGL = (properties.OpenGL ?? {}) as Record<string, unknown>;
    const inverseScale = [scale.xy, scale.xy, scale.z].map((value) =>
      Number.isFinite(value) && value > 0 ? 1 / value : 1);
    const priorScale = openGL.nimbusInverseSceneScale as number[] | undefined;
    if (openGL.nimbusPhysicalReference !== physicalOpacityDistance
        || inverseScale.some((value, axis) => value !== priorScale?.[axis])) {
      mapper.setViewSpecificProperties({ ...properties, OpenGL: { ...openGL,
        nimbusPhysicalReference: physicalOpacityDistance, nimbusInverseSceneScale: inverseScale,
      } });
    }
  }
  quality.setFullSampleDistanceFor(mapper, sampleDistance);
  return { opacityDistance, sampleDistance };
}

interface OpacityPropertyLike {
  setScalarOpacityUnitDistance: (component: number, distance: number) => unknown;
}

interface DirectionCameraLike {
  getDirectionOfProjection: () => number[];
}
