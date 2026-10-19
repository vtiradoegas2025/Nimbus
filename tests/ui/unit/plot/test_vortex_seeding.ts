/*
 * Tests for vortexDetection.ts and vortexSeeding.ts.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types \
 *     --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/plot/test_vortex_seeding.ts
 *
 * No test framework — uses the same tiny inline assertion API as test_quiver.ts.
 */

import {
  findInterleavedVerticalRotationMaximum,
  findRotationMaximum,
  type GridShape,
  type GridSpacing,
} from "../../../../ui/src/lib/plot/vortexDetection.ts";
import {
  buildVortexRingSeeds,
  type DomainBounds,
} from "../../../../ui/src/lib/plot/vortexSeeding.ts";

// ---------- Tiny assertion API (matches test_quiver.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL  ${label}`);
  }
}

function approx(a: number, b: number, eps: number): boolean {
  return Math.abs(a - b) <= eps;
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Rankine vortex helper ----------
//
// Matches vortex_seed.hpp / VortexSeedTrigger exactly:
//   v_theta(r) = v_max * (r / r_max)    for r <= r_max   (solid-body interior)
//   v_theta(r) = v_max * (r_max / r)    for r > r_max    (irrotational exterior)
//
// The tangential velocity is projected onto the Cartesian axes and the vertical
// vorticity ω_z = ∂v/∂x - ∂u/∂y is derived analytically:
//   ω_z(r) = 2 * v_max / r_max   for r <= r_max  (constant inside core)
//   ω_z(r) = 0                   for r > r_max   (irrotational)
// (The exterior 1/r profile has ω_z = 0 everywhere outside the core.)

function makeRankineVorticityZ(
  shape: GridShape,
  spacing: GridSpacing,
  centerXM: number,
  centerYM: number,
  coreRadiusM: number,
  vMaxMS: number,
  /** Vorticity sign: +1 cyclonic, -1 anticyclonic. */
  sign: 1 | -1,
): Float32Array {
  const { nx, ny, nz } = shape;
  const { dx, dy } = spacing;
  const data = new Float32Array(nx * ny * nz);

  // Analytical ω_z for the Rankine profile (see above).
  const omegaCore = sign * (2 * vMaxMS) / coreRadiusM;

  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const x = (ix + 0.5) * dx;
        const y = (iy + 0.5) * dy;
        const r = Math.sqrt((x - centerXM) ** 2 + (y - centerYM) ** 2);
        data[ix + nx * (iy + ny * iz)] = r <= coreRadiusM ? omegaCore : 0;
      }
    }
  }
  return data;
}

// ---------- 1. Rankine vortex: known-answer recovery ----------

check("Rankine vortex: findRotationMaximum recovers the planted center within one grid cell", () => {
  // Place the vortex axis exactly at a grid-cell centre so the peak ω_z cell
  // is uniquely determined: ix=19, iy=19 at dx=200 m -> centre at 3900 m.
  // Passing 3900 (a cell centre, not an arbitrary 4000 that straddles cells)
  // ensures the scan-order winner is the cell containing the geometric centre,
  // not some other equally-valued interior cell.
  const shape: GridShape = { nx: 40, ny: 40, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  // ix=19 -> cell centre = (19 + 0.5) * 200 = 3900 m
  const centerXM = 3900;
  const centerYM = 3900;
  const coreRadiusM = 600; // 3 cells
  const vMaxMS = 30;

  const vorticityZ = makeRankineVorticityZ(
    shape, spacing, centerXM, centerYM, coreRadiusM, vMaxMS, 1,
  );

  const core = findRotationMaximum(vorticityZ, shape, spacing, {
    nearSurfaceDepthM: 1000,
    significanceThresholdPerS: 0.01,
  });

  assertTrue(core !== null, "core is detected");
  if (core === null) return;

  const interleaved = new Float32Array(vorticityZ.length * 3);
  for (let i = 0; i < vorticityZ.length; i++) interleaved[i * 3 + 2] = vorticityZ[i];
  const interleavedCore = findInterleavedVerticalRotationMaximum(
    interleaved,
    shape,
    spacing,
    { nearSurfaceDepthM: 1000, significanceThresholdPerS: 0.01 },
  );
  assertTrue(
    JSON.stringify(interleavedCore) === JSON.stringify(core),
    "interleaved VTK vector detection exactly matches the scalar detector",
  );

  // For a discrete Rankine vortex the interior is flat-top — all cells inside
  // r_max carry the same ω_z.  The detector returns the first such cell in
  // scan order; the guaranteed invariant is that the returned cell is within
  // the core, i.e. within coreRadiusM of the planted centre (plus one cell of
  // discretisation error).
  const distFromCenter = Math.sqrt(
    (core.xM - centerXM) ** 2 + (core.yM - centerYM) ** 2,
  );
  const tolerance = coreRadiusM + Math.max(spacing.dx, spacing.dy);
  assertTrue(
    distFromCenter <= tolerance,
    `detected cell is within coreRadius + 1 cell of centre (dist=${distFromCenter.toFixed(0)} m, tol=${tolerance} m)`,
  );

  // Analytical ω_z inside the core = 2*v_max/r_max.
  const expectedOmega = (2 * vMaxMS) / coreRadiusM;
  assertTrue(
    approx(core.peakVorticityPerS, expectedOmega, expectedOmega * 0.01),
    `peak |ζ| within 1% of analytical value (got ${core.peakVorticityPerS.toFixed(4)}, expected ${expectedOmega.toFixed(4)})`,
  );

  // Core radius estimate should be within 2 grid cells of the planted value.
  const radiusTolerance = 2 * Math.max(spacing.dx, spacing.dy);
  assertTrue(
    approx(core.estimatedCoreRadiusM, coreRadiusM, radiusTolerance),
    `estimated core radius within 2 cells (got ${core.estimatedCoreRadiusM.toFixed(0)}, expected ${coreRadiusM})`,
  );
});

// ---------- 2. No significant rotation returns null ----------

check("No significant rotation: findRotationMaximum returns null (no false positive)", () => {
  const shape: GridShape = { nx: 20, ny: 20, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  // Weak background noise well below the default threshold.
  const tiny = 0.001; // s⁻¹, below default 0.01 threshold
  const data = new Float32Array(shape.nx * shape.ny * shape.nz).fill(tiny);

  const core = findRotationMaximum(data, shape, spacing, {
    significanceThresholdPerS: 0.01,
  });
  assertTrue(core === null, "returns null for sub-threshold field (no false positive)");
});

// ---------- 3. Near-surface restriction ----------
// A field with a strong midlevel rotation maximum (iz=8, ~2 km) and no
// near-ground rotation returns null when nearSurfaceDepthM=1000m (4 levels).

check("Near-surface restriction: midlevel-only vortex returns null", () => {
  const shape: GridShape = { nx: 20, ny: 20, nz: 12 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const data = new Float32Array(shape.nx * shape.ny * shape.nz);

  // Plant a strong vortex in the midlevel layer (iz=8, z_centre = 8.5 * 250 = 2125 m).
  // Near-surface limit at 1000 m -> max iz = floor((1000 - 125) / 250) = 3.
  // iz=8 > 3, so the search must miss it.
  const midIz = 8;
  for (let iy = 0; iy < shape.ny; iy++) {
    for (let ix = 0; ix < shape.nx; ix++) {
      data[ix + shape.nx * (iy + shape.ny * midIz)] = 0.5;
    }
  }

  const core = findRotationMaximum(data, shape, spacing, {
    nearSurfaceDepthM: 1000,
    significanceThresholdPerS: 0.01,
  });
  assertTrue(core === null, "midlevel-only rotation is not reported (near-surface restriction works)");
});

// ---------- 4. Near-surface vortex is found when midlevel is also present ----------

check("Near-surface vortex found when midlevel rotation also present", () => {
  const shape: GridShape = { nx: 20, ny: 20, nz: 12 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const data = new Float32Array(shape.nx * shape.ny * shape.nz);

  // Near-surface vortex at iz=1 (z_centre = 375 m)
  const surfIz = 1;
  const surfIx = 10;
  const surfIy = 10;
  data[surfIx + shape.nx * (surfIy + shape.ny * surfIz)] = 0.05;

  // Stronger midlevel vortex at iz=8 — should NOT be selected.
  const midIz = 8;
  for (let iy = 0; iy < shape.ny; iy++) {
    for (let ix = 0; ix < shape.nx; ix++) {
      data[ix + shape.nx * (iy + shape.ny * midIz)] = 0.5;
    }
  }

  const core = findRotationMaximum(data, shape, spacing, {
    nearSurfaceDepthM: 1000,
    significanceThresholdPerS: 0.01,
  });
  assertTrue(core !== null, "near-surface vortex is detected despite stronger midlevel feature");
  if (core !== null) {
    // Should point to iz=1 layer, not iz=8.
    const expectedZ = (surfIz + 0.5) * spacing.dz;
    assertTrue(
      approx(core.zM, expectedZ, spacing.dz),
      `detected vortex is in the near-surface layer (zM=${core.zM.toFixed(0)}, expected ~${expectedZ})`,
    );
  }
});

// ---------- 5. Anticyclonic rotation: correct sign ----------

check("Anticyclonic rotation detected with correct sign", () => {
  const shape: GridShape = { nx: 30, ny: 30, nz: 4 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const vorticityZ = makeRankineVorticityZ(
    shape, spacing, 3000, 3000, 400, 20, -1, // anticyclonic
  );

  const core = findRotationMaximum(vorticityZ, shape, spacing, {
    nearSurfaceDepthM: 1000,
    significanceThresholdPerS: 0.01,
  });

  assertTrue(core !== null, "anticyclonic vortex is detected");
  if (core !== null) {
    assertTrue(core.rotationSign === -1, `rotation sign is -1 (anticyclonic), got ${core.rotationSign}`);
  }
});

// ---------- 6. Cyclonic rotation has sign +1 ----------

check("Cyclonic rotation has sign +1", () => {
  const shape: GridShape = { nx: 30, ny: 30, nz: 4 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const vorticityZ = makeRankineVorticityZ(
    shape, spacing, 3000, 3000, 400, 20, 1,
  );

  const core = findRotationMaximum(vorticityZ, shape, spacing, {
    nearSurfaceDepthM: 1000,
    significanceThresholdPerS: 0.01,
  });
  assertTrue(core !== null, "cyclonic vortex is detected");
  if (core !== null) {
    assertTrue(core.rotationSign === 1, `rotation sign is +1 (cyclonic), got ${core.rotationSign}`);
  }
});

// ---------- 7. buildVortexRingSeeds: correct count ----------

check("buildVortexRingSeeds: emits azimuthCount * ringCount * heightLevelCount seeds (all in domain)", () => {
  const shape: GridShape = { nx: 40, ny: 40, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const vorticityZ = makeRankineVorticityZ(
    shape, spacing, 4000, 4000, 600, 30, 1,
  );

  const core = findRotationMaximum(vorticityZ, shape, spacing, {
    nearSurfaceDepthM: 1000,
  });
  assertTrue(core !== null, "core detected for seeding test");
  if (core === null) return;

  const domain: DomainBounds = {
    xMinM: 0, xMaxM: shape.nx * spacing.dx,
    yMinM: 0, yMaxM: shape.ny * spacing.dy,
    zMinM: 0, zMaxM: shape.nz * spacing.dz,
  };

  const opts = { azimuthCount: 8, ringCount: 2, heightLevelCount: 3, zSeedTopM: 400 };
  const result = buildVortexRingSeeds(core, domain, opts);

  // The core is well away from boundaries, so all seeds should be in domain.
  const expectedMax = opts.azimuthCount * opts.ringCount * opts.heightLevelCount;
  assertTrue(
    result.seedCount > 0 && result.seedCount <= expectedMax,
    `seed count ${result.seedCount} is in range (0, ${expectedMax}]`,
  );
  assertTrue(result.points.length === result.seedCount * 3, "points array length matches seedCount * 3");
});

// ---------- 8. All emitted seeds are within the domain ----------

check("buildVortexRingSeeds: every seed is within domain bounds", () => {
  const shape: GridShape = { nx: 40, ny: 40, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const vorticityZ = makeRankineVorticityZ(
    shape, spacing, 4000, 4000, 600, 30, 1,
  );

  const core = findRotationMaximum(vorticityZ, shape, spacing, { nearSurfaceDepthM: 1000 });
  if (core === null) { assertTrue(false, "core must be detected"); return; }

  const domain: DomainBounds = {
    xMinM: 0, xMaxM: shape.nx * spacing.dx,
    yMinM: 0, yMaxM: shape.ny * spacing.dy,
    zMinM: 0, zMaxM: shape.nz * spacing.dz,
  };

  const result = buildVortexRingSeeds(core, domain, {
    azimuthCount: 16, ringCount: 3, heightLevelCount: 4, zSeedTopM: 600,
  });

  let allInDomain = true;
  for (let i = 0; i + 2 < result.points.length; i += 3) {
    const x = result.points[i];
    const y = result.points[i + 1];
    const z = result.points[i + 2];
    if (
      x < domain.xMinM || x > domain.xMaxM ||
      y < domain.yMinM || y > domain.yMaxM ||
      z < domain.zMinM || z > domain.zMaxM
    ) {
      allInDomain = false;
      break;
    }
  }
  assertTrue(allInDomain, "every emitted seed is within the domain bounds");
});

// ---------- 9. Ring seeds span the full azimuth ----------
// With 8 azimuths the angles should span [0, 2π). No two consecutive seeds
// should point in the same azimuthal direction from the core.

check("buildVortexRingSeeds: seeds are distributed around the full azimuth", () => {
  const shape: GridShape = { nx: 40, ny: 40, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };
  const vorticityZ = makeRankineVorticityZ(
    shape, spacing, 4000, 4000, 600, 30, 1,
  );

  const core = findRotationMaximum(vorticityZ, shape, spacing, { nearSurfaceDepthM: 1000 });
  if (core === null) { assertTrue(false, "core must be detected"); return; }

  const domain: DomainBounds = {
    xMinM: 0, xMaxM: shape.nx * spacing.dx,
    yMinM: 0, yMaxM: shape.ny * spacing.dy,
    zMinM: 0, zMaxM: shape.nz * spacing.dz,
  };

  const azimuthCount = 8;
  const result = buildVortexRingSeeds(core, domain, {
    azimuthCount,
    ringCount: 1,
    heightLevelCount: 1,
    zSeedTopM: 400,
  });

  // Collect azimuthal angles of seeds relative to core.
  const angles: number[] = [];
  for (let i = 0; i + 2 < result.points.length; i += 3) {
    const dx = result.points[i] - core.xM;
    const dy = result.points[i + 1] - core.yM;
    angles.push(Math.atan2(dy, dx));
  }

  // With 8 evenly-spaced azimuths the angular step is 2π/8 = π/4 rad.
  // Consecutive angles (sorted) must differ by roughly that step.
  angles.sort((a, b) => a - b);
  const expectedStep = (2 * Math.PI) / azimuthCount;

  let distributed = true;
  for (let i = 1; i < angles.length; i++) {
    const gap = angles[i] - angles[i - 1];
    if (gap < expectedStep * 0.5 || gap > expectedStep * 1.5) {
      distributed = false;
    }
  }
  assertTrue(
    distributed,
    `${angles.length} ring seeds are evenly distributed around the azimuth (step ≈ ${(expectedStep * 180 / Math.PI).toFixed(0)}°)`,
  );
});

// ---------- 10. Ring radius scales with core radius ----------

check("buildVortexRingSeeds: innerRingRadiusM scales with core's estimatedCoreRadiusM", () => {
  const shape: GridShape = { nx: 40, ny: 40, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };

  // Two vortices with different core radii.
  const vz1 = makeRankineVorticityZ(shape, spacing, 4000, 4000, 400, 20, 1);
  const vz2 = makeRankineVorticityZ(shape, spacing, 4000, 4000, 800, 20, 1);

  const core1 = findRotationMaximum(vz1, shape, spacing, { nearSurfaceDepthM: 1000 });
  const core2 = findRotationMaximum(vz2, shape, spacing, { nearSurfaceDepthM: 1000 });

  assertTrue(core1 !== null && core2 !== null, "both cores detected for radius scaling test");
  if (!core1 || !core2) return;

  const domain: DomainBounds = {
    xMinM: 0, xMaxM: shape.nx * spacing.dx,
    yMinM: 0, yMaxM: shape.ny * spacing.dy,
    zMinM: 0, zMaxM: shape.nz * spacing.dz,
  };

  const r1 = buildVortexRingSeeds(core1, domain, { azimuthCount: 8, ringCount: 1, heightLevelCount: 1 }).innerRingRadiusM;
  const r2 = buildVortexRingSeeds(core2, domain, { azimuthCount: 8, ringCount: 1, heightLevelCount: 1 }).innerRingRadiusM;

  assertTrue(r2 > r1, `larger core radius produces larger ring radius (r1=${r1.toFixed(0)}, r2=${r2.toFixed(0)})`);
});

// ---------- 11. Out-of-domain seeds are dropped ----------

check("buildVortexRingSeeds: out-of-domain seeds dropped when core is near the boundary", () => {
  // Place the core so that some ring seeds (radius 600 m, centred 300 m from the
  // domain edge) fall outside.
  const shape: GridShape = { nx: 30, ny: 30, nz: 5 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 250 };

  // Core at ix=1 -> xM ≈ 300 m, domain ends at 6000 m.  Ring radius ~600 m
  // -> some seeds at xM - 600 < 0 fall out.
  const vorticityZ = makeRankineVorticityZ(shape, spacing, 300, 3000, 400, 20, 1);
  const core = findRotationMaximum(vorticityZ, shape, spacing, {
    nearSurfaceDepthM: 1000,
    significanceThresholdPerS: 0.005,
  });
  if (core === null) { assertTrue(false, "core near boundary must be detected"); return; }

  const domain: DomainBounds = {
    xMinM: 0, xMaxM: shape.nx * spacing.dx,
    yMinM: 0, yMaxM: shape.ny * spacing.dy,
    zMinM: 0, zMaxM: shape.nz * spacing.dz,
  };

  const opts = { azimuthCount: 16, ringCount: 1, heightLevelCount: 1, zSeedTopM: 400, coreRadiusFactor: 1.5 };
  const result = buildVortexRingSeeds(core, domain, opts);

  const maxPossible = opts.azimuthCount;
  assertTrue(
    result.seedCount < maxPossible,
    `some seeds dropped (${result.seedCount} < max ${maxPossible}) when core is near the boundary`,
  );
  assertTrue(result.seedCount > 0, "at least some seeds remain in domain");
});

// ---------- 13. Degenerate inputs ----------

check("findRotationMaximum: returns null on zero-size grid", () => {
  const data = new Float32Array(0);
  const core = findRotationMaximum(data, { nx: 0, ny: 10, nz: 5 }, { dx: 200, dy: 200, dz: 250 });
  assertTrue(core === null, "zero nx -> null");
});

check("findRotationMaximum: returns null when no cell is within the near-surface layer (dz > depth)", () => {
  const shape: GridShape = { nx: 5, ny: 5, nz: 3 };
  const spacing: GridSpacing = { dx: 200, dy: 200, dz: 2000 }; // each cell 2 km thick
  const data = new Float32Array(shape.nx * shape.ny * shape.nz).fill(0.5);
  // kMax = floor((500 - 1000) / 2000) = negative -> layer thinner than half a cell
  const core = findRotationMaximum(data, shape, spacing, {
    nearSurfaceDepthM: 500,
    significanceThresholdPerS: 0.01,
  });
  assertTrue(core === null, "layer shallower than half a cell returns null");
});

// ---------- Report ----------

console.log(`\nvortex-seeding tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
