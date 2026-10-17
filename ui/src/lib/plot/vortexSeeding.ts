/* Vortex-line seed placement: ring(s) of seeds at the detected core boundary concentrate streamlines
 * on the feature where vortex lines converge. Seeds in grid-relative metres; `vorticity_z` = ω_z. */

import type { VortexCore } from "./vortexDetection.ts";

/** Domain extent in grid-relative metres. */
export interface DomainBounds {
  xMinM: number;
  xMaxM: number;
  yMinM: number;
  yMaxM: number;
  zMinM: number;
  zMaxM: number;
}

export interface VortexRingSeedOptions {
  /** Azimuthal positions per ring. Default: 16. */
  azimuthCount: number;
  /** Concentric rings per height level. Default: 2. */
  ringCount: number;
  /** Height levels in the seed layer, evenly spaced [zMinM, zSeedTopM]. Default: 3. */
  heightLevelCount: number;
  /** AGL top of seed column (m); default 500 m keeps seeds in surface inflow. */
  zSeedTopM: number;
  /** Multiplier on estimatedCoreRadiusM for the innermost ring. Default: 1.0. */
  coreRadiusFactor: number;
  /** Radial spacing between rings as fraction of inner radius. Default: 0.5. */
  ringRadialSpacingFraction: number;
}

const DEFAULT_OPTIONS: VortexRingSeedOptions = {
  azimuthCount: 16,
  ringCount: 2,
  heightLevelCount: 3,
  zSeedTopM: 500,
  coreRadiusFactor: 1.0,
  ringRadialSpacingFraction: 0.5,
};

export interface VortexRingSeedResult {
  /** Flat xyz triples (m) for vtkPoints.setData(): [x0,y0,z0, x1,y1,z1, …]. */
  points: Float32Array;
  /** Number of seed points emitted (points.length / 3). */
  seedCount: number;
  /** Actual innermost ring radius used (m). */
  innerRingRadiusM: number;
}


/** Build concentric ring seeds centred on the vortex core; out-of-domain seeds are dropped. */
export function buildVortexRingSeeds(
  core: VortexCore,
  domain: DomainBounds,
  options: Partial<VortexRingSeedOptions> = {},
): VortexRingSeedResult {
  const opts: VortexRingSeedOptions = { ...DEFAULT_OPTIONS, ...options };

  const {
    azimuthCount,
    ringCount,
    heightLevelCount,
    zSeedTopM,
    coreRadiusFactor,
    ringRadialSpacingFraction,
  } = opts;

  const innerRadiusM = core.estimatedCoreRadiusM * coreRadiusFactor;
  const radialSpacingM = innerRadiusM * ringRadialSpacingFraction;

  // Candidate buffer: worst case all seeds in bounds.
  const maxCandidates = azimuthCount * ringCount * heightLevelCount;
  const candidates = new Float32Array(maxCandidates * 3);
  let emitted = 0;

  for (let iRing = 0; iRing < ringCount; iRing++) {
    const ringRadiusM = innerRadiusM + iRing * radialSpacingM;

    for (let iHeight = 0; iHeight < heightLevelCount; iHeight++) {
      // Height level: evenly distribute from domain floor to zSeedTopM.
      // When heightLevelCount === 1, place at the midpoint.
      const heightFraction =
        heightLevelCount === 1
          ? 0.5
          : iHeight / (heightLevelCount - 1);
      const z = domain.zMinM + heightFraction * zSeedTopM;

      // Clamp z early: if zSeedTopM pushes above the domain, seeds are
      // concentrated at the domain ceiling, which is fine — don't discard.
      const zClamped = Math.max(domain.zMinM, Math.min(domain.zMaxM, z));

      for (let iAz = 0; iAz < azimuthCount; iAz++) {
        // Distribute azimuths uniformly around the full circle.
        const theta = (2 * Math.PI * iAz) / azimuthCount;
        const x = core.xM + ringRadiusM * Math.cos(theta);
        const y = core.yM + ringRadiusM * Math.sin(theta);

        // Drop any point that falls outside the horizontal domain.
        if (
          x < domain.xMinM ||
          x > domain.xMaxM ||
          y < domain.yMinM ||
          y > domain.yMaxM
        ) {
          continue;
        }

        const base = emitted * 3;
        candidates[base] = x;
        candidates[base + 1] = y;
        candidates[base + 2] = zClamped;
        emitted++;
      }
    }
  }

  return {
    points: candidates.slice(0, emitted * 3),
    seedCount: emitted,
    innerRingRadiusM: innerRadiusM,
  };
}

