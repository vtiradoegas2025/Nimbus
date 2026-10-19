/**
 * Evidence-gated condensation associated with one connected low-level vortex.
 *
 * This module never creates condensate. It retains solved condensate near the
 * connected vorticity centreline and uses the surrounding solved cloud base to
 * decide whether that condensate actually descends as a funnel. The returned
 * scalar is a display-opacity field: values are only attenuated near the edge
 * of the diagnostic tube, never increased or extrapolated into dry cells.
 */

export interface TornadicCondensationImage {
  shape: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  /** Scalars in vtkImageData x-fastest tuple order. */
  values: Float32Array;
  validity?: Uint8Array;
}

export type TornadicCondensationReason =
  | "resolved-funnel"
  | "rotating-cloud"
  | "no-condensate"
  | "too-small"
  | "no-qualified-vortex"
  | "misaligned-fields"
  | "invalid-input";

export interface TornadicCondensationAssessment {
  /** True only for a resolved downward condensation extension. */
  present: boolean;
  /** True when solved rotating condensation can be displayed at all. */
  visible: boolean;
  reason: TornadicCondensationReason;
  lowestCondensateM: number | null;
  ambientCloudBaseM: number | null;
  downwardExtensionM: number;
  depthM: number;
  voxelCount: number;
}

export interface TornadicCondensationResult {
  image: TornadicCondensationImage | null;
  bounds: [number, number, number, number, number, number] | null;
  assessment: TornadicCondensationAssessment;
}

type Bounds = [number, number, number, number, number, number];

const emptyAssessment = (
  reason: TornadicCondensationReason,
): TornadicCondensationAssessment => ({
  present: false,
  visible: false,
  reason,
  lowestCondensateM: null,
  ambientCloudBaseM: null,
  downwardExtensionM: 0,
  depthM: 0,
  voxelCount: 0,
});

function validImage(image: TornadicCondensationImage): boolean {
  return image.shape.length === 3 && image.shape.every(value => Number.isSafeInteger(value) && value > 0)
    && image.spacing.length === 3 && image.spacing.every(value => Number.isFinite(value) && value > 0)
    && image.origin.length === 3 && image.origin.every(Number.isFinite)
    && image.values.length === image.shape.reduce((a, b) => a * b, 1)
    && (!image.validity || image.validity.length === image.values.length);
}

function known(image: TornadicCondensationImage, tuple: number): boolean {
  return image.validity?.[tuple] !== 0 && Number.isFinite(image.values[tuple]);
}

function aligned(a: TornadicCondensationImage, b: TornadicCondensationImage): boolean {
  return a.shape.every((value, axis) => value === b.shape[axis])
    && a.spacing.every((value, axis) => Math.abs(value - b.spacing[axis]!) < 1e-6)
    && a.origin.every((value, axis) => Math.abs(value - b.origin[axis]!) < 1e-4)
    && a.values.length === b.values.length;
}

function finiteBounds(bounds: Bounds | null): bounds is Bounds {
  return bounds != null && bounds.every(Number.isFinite)
    && bounds[1] >= bounds[0] && bounds[3] >= bounds[2] && bounds[5] >= bounds[4];
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1]! + values[middle]!) * 0.5
    : values[middle]!;
}

/**
 * Isolate solved cloud condensate around the connected vortex centreline.
 * `condensateFloor` is kg/kg and `vorticityThreshold` is signed s^-1.
 */
export function deriveTornadicCondensation(
  condensate: TornadicCondensationImage,
  vorticity: TornadicCondensationImage,
  anchor: readonly [number, number, number] | null | undefined,
  coreBounds: Bounds | null,
  vorticityThreshold: number,
  condensateFloor: number,
  tornadoQualified: boolean,
): TornadicCondensationResult {
  if (!validImage(condensate) || !validImage(vorticity)
      || !Number.isFinite(condensateFloor) || condensateFloor <= 0) {
    return { image: null, bounds: null, assessment: emptyAssessment("invalid-input") };
  }
  if (!aligned(condensate, vorticity)) {
    return { image: null, bounds: null, assessment: emptyAssessment("misaligned-fields") };
  }
  if (!anchor || !anchor.every(Number.isFinite) || !finiteBounds(coreBounds)
      || !Number.isFinite(vorticityThreshold) || vorticityThreshold === 0) {
    return { image: null, bounds: null, assessment: emptyAssessment("no-qualified-vortex") };
  }

  const [nx, ny, nz] = condensate.shape;
  const [dx, dy, dz] = condensate.spacing;
  const [ox, oy, oz] = condensate.origin;
  const horizontalSpacing = Math.max(dx, dy);
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const coreSign = Math.sign(vorticityThreshold);
  const coreMagnitude = Math.abs(vorticityThreshold);
  const anchorX = Math.max(0, Math.min(nx - 1, Math.round((anchor[0] - ox) / dx)));
  const anchorY = Math.max(0, Math.min(ny - 1, Math.round((anchor[1] - oy) / dy)));

  // Build one solved centreline from threshold-qualified cells near the anchor.
  const centerX = new Float64Array(nz);
  const centerY = new Float64Array(nz);
  const coreRadius = new Float64Array(nz);
  const coreCounts = new Uint32Array(nz);
  const locatorRadiusM = 3_000;
  const locatorX = Math.ceil(locatorRadiusM / dx);
  const locatorY = Math.ceil(locatorRadiusM / dy);
  for (let z = 0; z < nz; z++) {
    for (let y = Math.max(0, anchorY - locatorY); y <= Math.min(ny - 1, anchorY + locatorY); y++) {
      const worldY = oy + y * dy;
      for (let x = Math.max(0, anchorX - locatorX); x <= Math.min(nx - 1, anchorX + locatorX); x++) {
        const worldX = ox + x * dx;
        if (Math.hypot(worldX - anchor[0], worldY - anchor[1]) > locatorRadiusM) continue;
        const tuple = index(x, y, z);
        if (!known(vorticity, tuple)) continue;
        const value = vorticity.values[tuple]! * coreSign;
        if (!(value >= coreMagnitude)) continue;
        centerX[z] += worldX;
        centerY[z] += worldY;
        coreCounts[z] += 1;
      }
    }
    if (coreCounts[z] > 0) {
      centerX[z] /= coreCounts[z];
      centerY[z] /= coreCounts[z];
    }
  }
  for (let z = 0; z < nz; z++) {
    if (coreCounts[z] === 0) continue;
    for (let y = Math.max(0, anchorY - locatorY); y <= Math.min(ny - 1, anchorY + locatorY); y++) {
      const worldY = oy + y * dy;
      for (let x = Math.max(0, anchorX - locatorX); x <= Math.min(nx - 1, anchorX + locatorX); x++) {
        const tuple = index(x, y, z);
        if (!known(vorticity, tuple)) continue;
        const value = vorticity.values[tuple]! * coreSign;
        if (!(value >= coreMagnitude)) continue;
        const radius = Math.hypot(ox + x * dx - centerX[z], worldY - centerY[z]);
        if (radius > coreRadius[z]) coreRadius[z] = radius;
      }
    }
  }

  const coreLevels = Array.from(coreCounts, (count, z) => count > 0 ? z : -1)
    .filter((z) => z >= 0);
  if (coreLevels.length === 0) {
    return { image: null, bounds: null, assessment: emptyAssessment("no-qualified-vortex") };
  }
  const nearestCoreLevel = new Int32Array(nz);
  for (let z = 0; z < nz; z++) {
    let nearest = coreLevels[0]!;
    let distance = Math.abs(nearest - z);
    for (const candidate of coreLevels) {
      const nextDistance = Math.abs(candidate - z);
      if (nextDistance >= distance) continue;
      nearest = candidate;
      distance = nextDistance;
    }
    nearestCoreLevel[z] = nearest;
  }

  const nominalCoreDiameter = Math.max(
    coreBounds[1] - coreBounds[0],
    coreBounds[3] - coreBounds[2],
  );
  const baseTubeRadiusM = Math.min(
    1_500,
    Math.max(600, nominalCoreDiameter * 0.5 + horizontalSpacing * 4),
  );
  const maxVerticalSeparationM = 2_500;
  const coreTopM = coreBounds[5];
  const candidate = (x: number, y: number, z: number): boolean => {
    const tuple = index(x, y, z);
    if (!known(condensate, tuple) || !known(vorticity, tuple)) return false;
    const nearest = nearestCoreLevel[z]!;
    if (Math.abs(z - nearest) * dz > maxVerticalSeparationM) return false;
    const radius = Math.min(
      1_500,
      Math.max(baseTubeRadiusM, coreRadius[nearest]! + horizontalSpacing * 4),
    );
    return Math.hypot(
      ox + x * dx - centerX[nearest]!,
      oy + y * dy - centerY[nearest]!,
    ) <= radius;
  };

  // Local environmental cloud base comes from an annulus outside the vortex tube.
  const ambientBases: number[] = [];
  const innerAmbientRadius = Math.max(2_000, baseTubeRadiusM * 1.5);
  const outerAmbientRadius = Math.min(6_000, Math.max(4_000, baseTubeRadiusM * 4));
  for (let y = 0; y < ny; y++) {
    const worldY = oy + y * dy;
    for (let x = 0; x < nx; x++) {
      const radial = Math.hypot(ox + x * dx - anchor[0], worldY - anchor[1]);
      if (radial < innerAmbientRadius || radial > outerAmbientRadius) continue;
      for (let z = 0; z < nz; z++) {
        const tuple = index(x, y, z);
        if (!known(condensate, tuple)) break;
        if (condensate.values[tuple]! < condensateFloor) continue;
        ambientBases.push(oz + z * dz);
        break;
      }
    }
  }
  const ambientCloudBaseM = median(ambientBases);

  // Seed the connected condensate component closest to the solved centreline.
  let seed = -1;
  let seedScore = Number.POSITIVE_INFINITY;
  for (let z = 0; z < nz; z++) {
    if (oz + z * dz > coreTopM + maxVerticalSeparationM) break;
    const nearest = nearestCoreLevel[z]!;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const tuple = index(x, y, z);
        if (!candidate(x, y, z) || !(condensate.values[tuple]! >= condensateFloor)) continue;
        const radial = Math.hypot(
          ox + x * dx - centerX[nearest]!,
          oy + y * dy - centerY[nearest]!,
        );
        const score = radial * radial + Math.abs(z - nearest) * dz * dz;
        if (score >= seedScore) continue;
        seedScore = score;
        seed = tuple;
      }
    }
  }
  if (seed < 0) {
    return {
      image: null,
      bounds: null,
      assessment: {
        ...emptyAssessment(tornadoQualified ? "no-condensate" : "no-qualified-vortex"),
        ambientCloudBaseM,
      },
    };
  }

  const visited = new Uint8Array(condensate.values.length);
  const queue = new Int32Array(condensate.values.length);
  const output = new Float32Array(condensate.values.length);
  let validity: Uint8Array | undefined;
  for (let tuple = 0; tuple < output.length; tuple++) {
    if (known(condensate, tuple) && known(vorticity, tuple)) continue;
    validity ??= new Uint8Array(output.length).fill(255);
    validity[tuple] = 0;
  }
  const stepSupported = (x: number, y: number, z: number, nextX: number, nextY: number, nextZ: number) => {
    if (!validity) return true;
    for (let iz = Math.min(z, nextZ); iz <= Math.max(z, nextZ); iz++)
      for (let iy = Math.min(y, nextY); iy <= Math.max(y, nextY); iy++)
        for (let ix = Math.min(x, nextX); ix <= Math.max(x, nextX); ix++)
          if (validity[index(ix, iy, iz)] === 0) return false;
    return true;
  };
  let head = 0;
  let tail = 0;
  queue[tail++] = seed;
  visited[seed] = 1;
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  let voxelCount = 0;
  while (head < tail) {
    const tuple = queue[head++]!;
    const z = Math.floor(tuple / (nx * ny));
    const rem = tuple - z * nx * ny;
    const y = Math.floor(rem / nx);
    const x = rem - y * nx;
    const value = condensate.values[tuple]!;
    if (!(value >= condensateFloor) || !candidate(x, y, z)) continue;

    const nearest = nearestCoreLevel[z]!;
    const radius = Math.min(
      1_500,
      Math.max(baseTubeRadiusM, coreRadius[nearest]! + horizontalSpacing * 4),
    );
    const radial = Math.hypot(
      ox + x * dx - centerX[nearest]!,
      oy + y * dy - centerY[nearest]!,
    );
    // Fade only the outer fifth of the selection tube. This suppresses a hard
    // cylindrical crop while never increasing or inventing condensate.
    const taper = radial <= radius * 0.8 ? 1 : Math.max(0, (radius - radial) / (radius * 0.2));
    output[tuple] = value * taper;
    voxelCount += 1;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);

    for (let dzOffset = -1; dzOffset <= 1; dzOffset++) {
      const nextZ = z + dzOffset;
      if (nextZ < 0 || nextZ >= nz) continue;
      for (let dyOffset = -1; dyOffset <= 1; dyOffset++) {
        const nextY = y + dyOffset;
        if (nextY < 0 || nextY >= ny) continue;
        for (let dxOffset = -1; dxOffset <= 1; dxOffset++) {
          if (dxOffset === 0 && dyOffset === 0 && dzOffset === 0) continue;
          const nextX = x + dxOffset;
          if (nextX < 0 || nextX >= nx) continue;
          const next = index(nextX, nextY, nextZ);
          if (visited[next] || !stepSupported(x, y, z, nextX, nextY, nextZ)) continue;
          visited[next] = 1;
          if (condensate.values[next]! >= condensateFloor && candidate(nextX, nextY, nextZ)) {
            queue[tail++] = next;
          }
        }
      }
    }
  }

  const lowestCondensateM = oz + minZ * dz;
  const highestCondensateM = oz + maxZ * dz;
  const depthM = highestCondensateM - lowestCondensateM;
  const downwardExtensionM = ambientCloudBaseM == null
    ? 0
    : Math.max(0, ambientCloudBaseM - lowestCondensateM);
  const minimumDepthM = Math.max(300, dz * 3);
  const enoughEvidence = voxelCount >= 8 && depthM >= minimumDepthM;
  const descends = ambientCloudBaseM != null
    && downwardExtensionM >= Math.max(300, dz * 3)
    && lowestCondensateM <= coreTopM + dz * 2;
  const present = tornadoQualified && enoughEvidence && descends;
  const reason: TornadicCondensationReason = present
    ? "resolved-funnel"
    : !tornadoQualified
      ? "no-qualified-vortex"
      : !enoughEvidence
        ? "too-small"
        : "rotating-cloud";
  const visible = voxelCount > 0;
  const bounds: Bounds = [
    ox + minX * dx, ox + maxX * dx,
    oy + minY * dy, oy + maxY * dy,
    lowestCondensateM, highestCondensateM,
  ];
  return {
    image: visible ? {
      shape: condensate.shape,
      spacing: condensate.spacing,
      origin: condensate.origin,
      values: output,
      validity,
    } : null,
    bounds: visible ? bounds : null,
    assessment: {
      present,
      visible,
      reason,
      lowestCondensateM,
      ambientCloudBaseM,
      downwardExtensionM,
      depthM,
      voxelCount,
    },
  };
}
