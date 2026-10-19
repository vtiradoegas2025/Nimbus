/*
 * Regression tests for `src/lib/cartesianInterp.ts`:
 *   - the dy-slot sentinel (isCartesianGrid) + scalar/vector dispatchers, and
 *   - the Cartesian box/vector wrappers incl. the no-rotation invariant and the
 *     non-finite sweep.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_cartesian_interp.ts
 *
 * Bug under test: the Cartesian box passed raw NaN to VTK (a single NaN/Inf
 * scalar handed to marching cubes / the volume mapper / streamlines hangs or
 * corrupts the GPU render), and the streamline vector path spuriously rotated
 * already-Cartesian (u, v). The box paths now sanitize non-finite values and
 * pass vector components through unchanged. isCartesianGrid is the single
 * source of truth for reading dy out of the legacy `dtheta_rad` manifest slot.
 */

import {
  cartesianBoxToImageData,
  cartesianVectorBoxToImageData,
  fieldToCartesianGrid,
  terrainWindowToCartesianGrid,
  vectorFieldToCartesianGrid,
  isCartesianGrid,
  cartesianPreviewStride,
  CARTESIAN_VOXEL_CAP_VOLUME,
  CARTESIAN_VOXEL_CAP_ISOSURFACE,
  CARTESIAN_VOXEL_CAP_STREAMLINES,
  CARTESIAN_VOXEL_CAP_SUPERCELL,
  cartesianGridToVtkScalars,
  cartesianGridToVtkVectors,
  cartesianGridToVtkVectorsInPlace,
  cartesianVectorMaxMagnitude,
  cartesianVectorSpeedQuantile,
  cartesianScalarCoordinate,
  cartesianScalarFractionalIndex,
  type CartesianVoxelGrid,
} from "../../../../ui/src/lib/science/cartesianInterp.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";
import { remapTerrainVerticalSlice } from "../../../../ui/src/lib/science/terrainSliceRemap.ts";

// ---------- Tiny assertion API (matches test_schema_validator.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) {
    passed++;
  } else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

function close(a: number, b: number, eps = 1e-9): boolean {
  return Math.abs(a - b) <= eps;
}

// ---------- Helpers ----------

function makeField(
  nr: number,
  nth: number,
  nz: number,
  fn: (i: number, j: number, k: number) => number,
): Npy3D {
  const data = new Float32Array(nr * nth * nz);
  for (let i = 0; i < nr; i++) {
    for (let j = 0; j < nth; j++) {
      for (let k = 0; k < nz; k++) {
        data[i * nth * nz + j * nz + k] = fn(i, j, k);
      }
    }
  }
  return { shape: [nr, nth, nz], data };
}

const NR = 4;
const NTH = 8;
const NZ = 2;

function allFinite(data: Float32Array): boolean {
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) return false;
  }
  return true;
}

// ---------- 1. isCartesianGrid: explicit metadata with legacy default ----------

check("isCartesianGrid: classifies coordinate metadata correctly", () => {
  assertTrue(isCartesianGrid(null) === true, "null -> legacy Cartesian default");
  assertTrue(isCartesianGrid(undefined) === true, "undefined -> Cartesian");
  assertTrue(isCartesianGrid("cartesian") === true, "explicit Cartesian");
  assertTrue(isCartesianGrid("CARTESIAN") === true, "case-insensitive Cartesian");
  assertTrue(isCartesianGrid("cylindrical") === false, "unsupported explicit coordinate system");
});

// ---------- 2. cartesianBoxToImageData: clean field aliases, no copy ----------

check("cartesian box: a finite field is aliased (zero-copy)", () => {
  const field = makeField(NR, NTH, NZ, (i, j, k) => i * NTH * NZ + j * NZ + k + 10);
  const grid = cartesianBoxToImageData(field, 1000, 600, 1000);
  assertTrue(grid.data === field.data, "clean field: output buffer is the input buffer");
  assertTrue(
    grid.shape[0] === NR && grid.shape[1] === NTH && grid.shape[2] === NZ,
    "shape == input shape (no resample)",
  );
  assertTrue(
    grid.spacing[0] === 1000 && grid.spacing[1] === 600 && grid.spacing[2] === 1000,
    "spacing == (dx, dy, dz)",
  );
  assertTrue(
    grid.origin[0] === -(NR - 1) * 1000 * 0.5 && grid.origin[2] === 0.5 * 1000,
    "box centred on origin, z base at first cell centre (0.5*dz)",
  );
});

// ---------- 8. cartesianBoxToImageData: non-finite -> finite minimum ----------

check("cartesian box: a NaN cell becomes the finite minimum (copy, not alias)", () => {
  const field = makeField(NR, NTH, NZ, (i, j, k) => i * NTH * NZ + j * NZ + k + 10);
  const nanIdx = 1 * NTH * NZ + 1 * NZ + 1;
  field.data[nanIdx] = NaN;
  const grid = cartesianBoxToImageData(field, 1000, 600, 1000);
  assertTrue(allFinite(grid.data), "no NaN in output");
  assertTrue(grid.data !== field.data, "dirty field: output is a sanitized copy");
  assertTrue(grid.data[nanIdx] === 10, "NaN cell replaced with the field minimum (10)");
  assertTrue(grid.sourceNonFiniteCount === 1, "missing-source count survives finite substitution");
});

check("cartesian box: all-NaN field falls back to 0, never NaN", () => {
  const field = makeField(NR, NTH, NZ, () => NaN);
  const grid = cartesianBoxToImageData(field, 1000, 600, 1000);
  assertTrue(allFinite(grid.data), "no NaN in output");
  let allZero = true;
  for (let n = 0; n < grid.data.length; n++) if (grid.data[n] !== 0) allZero = false;
  assertTrue(allZero, "all-NaN field -> 0 fill");
  assertTrue(
    grid.sourceNonFiniteCount === field.data.length,
    "an all-missing source reports every substituted cell",
  );
});

check("cartesian LOD: area mean removes checkerboard aliasing", () => {
  const field = makeField(4, 4, 1, (i, j) => ((i + j) % 2 === 0 ? 1 : -1));
  const grid = cartesianBoxToImageData(field, 1, 1, 1, { stride: 2 });
  assertTrue(grid.shape.join(",") === "2,2,1", "stride 2 produces the bounded lattice");
  assertTrue(
    Array.from(grid.data).every((value) => Math.abs(value) < 1e-7),
    "Nyquist checkerboard averages to zero instead of aliasing to a false constant",
  );
  assertTrue(
    grid.lod?.aggregation === "area_mean_xy" && grid.lod.nativeShape.join(",") === "4,4,1",
    "reduced grid declares native shape and aggregation method",
  );
});

check("cartesian vector LOD: averages components in their Cartesian basis", () => {
  const u = makeField(4, 4, 1, (i, j) => ((i + j) % 2 === 0 ? 2 : -2));
  const v = makeField(4, 4, 1, () => 4);
  const w = makeField(4, 4, 1, () => -1);
  const grid = cartesianVectorBoxToImageData(u, v, w, 1, 1, 1, { stride: 2 });
  let correct = true;
  for (let index = 0; index < grid.data.length; index += 3) {
    if (Math.abs(grid.data[index]) > 1e-7 || grid.data[index + 1] !== 4 || grid.data[index + 2] !== -1) {
      correct = false;
    }
  }
  assertTrue(correct, "each Cartesian component is area-averaged without rotation");
});

// ---------- 9. cartesianVectorBoxToImageData: NO rotation ----------
//
// The Streamlines3D bug rotated already-Cartesian (u, v) by atan2(y, x). The
// Cartesian vector path must pass components through unchanged.

check("cartesian vector: components pass through with no rotation", () => {
  const u = makeField(NR, NTH, NZ, () => 1);
  const v = makeField(NR, NTH, NZ, () => 2);
  const w = makeField(NR, NTH, NZ, () => 3);
  const grid = cartesianVectorBoxToImageData(u, v, w, 1000, 600, 1000);
  assertTrue(grid.data.length === NR * NTH * NZ * 3, "3 interleaved components per voxel");
  let ok = true;
  for (let n = 0; n < grid.data.length; n += 3) {
    if (grid.data[n] !== 1 || grid.data[n + 1] !== 2 || grid.data[n + 2] !== 3) ok = false;
  }
  assertTrue(ok, "every voxel is exactly (1, 2, 3) -- not rotated");
});

check("cartesian vector: NaN component becomes 0 (no flow), others preserved", () => {
  const u = makeField(NR, NTH, NZ, (i, j, k) => (i === 1 && j === 1 && k === 0 ? NaN : 1));
  const v = makeField(NR, NTH, NZ, () => 2);
  const w = makeField(NR, NTH, NZ, () => 3);
  const grid = cartesianVectorBoxToImageData(u, v, w, 1000, 600, 1000);
  assertTrue(allFinite(grid.data), "no NaN in any channel");
  const d = (1 * NTH * NZ + 1 * NZ + 0) * 3;
  assertTrue(
    grid.data[d] === 0 && grid.data[d + 1] === 2 && grid.data[d + 2] === 3,
    "NaN u -> 0, v/w preserved",
  );
});

check("cartesian vector: mismatched component shapes throw", () => {
  const u = makeField(NR, NTH, NZ, () => 1);
  const v = makeField(NR, NTH, NZ, () => 2);
  const wBad = makeField(NR, NTH, NZ + 1, () => 3);
  let threw = false;
  try {
    cartesianVectorBoxToImageData(u, v, wBad, 1000, 600, 1000);
  } catch {
    threw = true;
  }
  assertTrue(threw, "shape mismatch is rejected");
});

// ---------- 10. dispatch on coordinate system ----------

check("dispatch: Cartesian grid routes scalars through the box path", () => {
  const field = makeField(NR, NTH, NZ, (i, j, k) => i + j + k);
  const grid = fieldToCartesianGrid(field, {
    coordinateSystem: "cartesian", drM: 1000, dyM: 5, dthetaRad: 5, dzM: 1000,
  });
  assertTrue(grid.shape[0] === NR && grid.shape[1] === NTH, "box shape == input (no disc resample)");
  assertTrue(grid.spacing[1] === 5, "small dy taken from explicit dy_m metadata");
});

check("dispatch: Cartesian vector field is NOT rotated", () => {
  // u=1, v=0, w=0 everywhere. The components are already Cartesian, so the box
  // path must keep (1, 0, 0) -- no rotation.
  const u = makeField(NR, NTH, NZ, () => 1);
  const v = makeField(NR, NTH, NZ, () => 0);
  const w = makeField(NR, NTH, NZ, () => 0);
  const grid = vectorFieldToCartesianGrid(u, v, w, {
    coordinateSystem: "cartesian", drM: 1000, dyM: 5, dthetaRad: 5, dzM: 1000,
  });
  let ok = true;
  for (let n = 0; n < grid.data.length; n += 3) {
    if (grid.data[n] !== 1 || grid.data[n + 1] !== 0 || grid.data[n + 2] !== 0) ok = false;
  }
  assertTrue(ok, "Cartesian (u,v,w) preserved as (1, 0, 0) everywhere");
});

check("stretched z is linearly resampled onto an equal-count physical VTK lattice", () => {
  const z = [0, 100, 300];
  const field = makeField(2, 2, 3, (_i, _j, k) => z[k]);
  const grid = fieldToCartesianGrid(field, {
    coordinateSystem: "cartesian",
    drM: 1_000,
    dyM: 1_000,
    scalarCoordinatesM: [[0, 1_000], [0, 1_000], z],
  });
  assertTrue(grid.origin[2] === 0 && grid.spacing[2] === 150, "stretched z preserves endpoint span on a uniform VTK lattice");
  assertTrue(
    grid.data[0] === 0 && grid.data[1] === 150 && grid.data[2] === 300,
    "scalar coordinate-linear resampling preserves a field linear in physical height",
  );
  const missing = fieldToCartesianGrid(
    makeField(2, 2, 3, (_i, _j, k) => k === 1 ? Number.NaN : k === 0 ? 1 : 3),
    {
      coordinateSystem: "cartesian",
      drM: 1_000,
      dyM: 1_000,
      scalarCoordinatesM: [[0, 1_000], [0, 1_000], z],
    },
  );
  assertTrue(
    missing.data[0] === 1 && missing.data[2] === 3 && Number.isFinite(missing.data[1]),
    "coordinate resampling preserves finite endpoints and contains missing source levels",
  );

  const u = makeField(2, 2, 3, (_i, _j, k) => z[k]);
  const v = makeField(2, 2, 3, (_i, _j, k) => 2 * z[k]);
  const w = makeField(2, 2, 3, () => 5);
  const vectors = vectorFieldToCartesianGrid(u, v, w, {
    coordinateSystem: "cartesian",
    drM: 1_000,
    dyM: 1_000,
    scalarCoordinatesM: [[0, 1_000], [0, 1_000], z],
  });
  assertTrue(
    vectors.data[3] === 150 && vectors.data[4] === 300 && vectors.data[5] === 5,
    "vector coordinate-linear resampling preserves every Cartesian component",
  );
});

check("3D rendering regularizes stretched x/y/z in physical coordinates", () => {
  const x = [0, 100, 300];
  const y = [0, 200, 600];
  const z = [0, 50, 150];
  const field = makeField(3, 3, 3, (i, j, k) => x[i] + 2 * y[j] + 3 * z[k]);
  const grid = fieldToCartesianGrid(field, {
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    scalarCoordinatesM: [x, y, z],
  });
  assertTrue(grid.origin.join(",") === "0,0,0", "rectilinear regularization preserves first scalar centers");
  assertTrue(grid.spacing.join(",") === "150,300,75", "rectilinear regularization spans each endpoint-center range");
  assertTrue(Math.abs(grid.data[(1 * 3 + 1) * 3 + 1] - 975) < 1e-4, "trilinear physical-coordinate sampling reproduces a linear scalar field exactly");

  const strided = fieldToCartesianGrid(field, {
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    scalarCoordinatesM: [x, y, z],
  }, { stride: 2 });
  assertTrue(strided.shape.join(",") === "2,2,3" && strided.spacing.join(",") === "300,600,75", "bounded LOD preserves stretched-axis endpoint spans");
  assertTrue(
    Math.abs(strided.data[0] - 250) < 1e-4 && Math.abs((strided.data.at(-1) ?? 0) - 1_750) < 1e-4,
    "bounded rectilinear LOD reports physical-area means rather than aliased endpoint samples",
  );

  const withMissing = makeField(3, 3, 3, (i, j, k) => i === 1 && j === 1 && k === 1 ? Number.NaN : x[i] + y[j] + z[k]);
  const contained = fieldToCartesianGrid(withMissing, {
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    scalarCoordinatesM: [x, y, z],
  });
  assertTrue(contained.data.every(Number.isFinite), "rectilinear interpolation contains non-finite source neighborhoods before VTK upload");

  const vectors = vectorFieldToCartesianGrid(
    makeField(3, 3, 3, (i) => x[i]),
    makeField(3, 3, 3, (_i, j) => y[j]),
    makeField(3, 3, 3, (_i, _j, k) => z[k]),
    { coordinateSystem: "cartesian", geometryKind: "physical", scalarCoordinatesM: [x, y, z] },
  );
  const middle = ((1 * 3 + 1) * 3 + 1) * 3;
  assertTrue(
    vectors.data[middle] === 150 && vectors.data[middle + 1] === 300 && vectors.data[middle + 2] === 75,
    "vector regularization applies the same physical trilinear map to every component",
  );
});

check("3D rendering rejects CM1 index geometry explicitly", () => {
  let message = "";
  try {
    fieldToCartesianGrid(makeField(2, 2, 2, () => 1), {
      coordinateSystem: "cartesian",
      geometryKind: "index",
    });
  } catch (error) {
    message = error instanceof Error ? error.message : String(error);
  }
  assertTrue(message.includes("index-based 2D inspection"), "index geometry remains inspectable without masquerading as physical 3D");
});

check("manifest scalar origin drives 3D placement and index round-trips", () => {
  const field = makeField(NR, NTH, NZ, (i, j, k) => i + j + k);
  const metadata = {
    coordinateSystem: "cartesian",
    drM: 1000,
    dyM: 600,
    dzM: 1000,
    scalarOriginM: [500, 300, 500] as [number, number, number],
    physicalBoundsM: {
      lower: [0, 0, 0] as [number, number, number],
      upper: [NR * 1000, NTH * 600, NZ * 1000] as [number, number, number],
    },
  };
  const grid = fieldToCartesianGrid(field, metadata);
  assertTrue(
    grid.origin[0] === 500 && grid.origin[1] === 300 && grid.origin[2] === 500,
    "VTK origin equals manifest scalar_origin_m",
  );

  const spacings = [1000, 600, 1000];
  for (let axis = 0; axis < 3; axis++) {
    for (const index of [0, 1, 3]) {
      const coordinate = cartesianScalarCoordinate(grid.origin[axis], spacings[axis], index);
      const roundTrip = cartesianScalarFractionalIndex(
        grid.origin[axis],
        spacings[axis],
        coordinate,
      );
      assertTrue(close(roundTrip, index), `axis ${axis} index ${index} round-trips`);
    }
  }
});

// ---------- cartesianPreviewStride: bounded voxel count ----------

check("cartesian stride: small grid renders full-res, large grid is capped", () => {
  const CAP = 16_000_000; // mirrors cartesianPreviewStride
  assertTrue(cartesianPreviewStride(10, 10, 10) === 1, "tiny grid -> stride 1");
  // A typical 256^3-class grid (256x256x80 = 5.2M) is now under the cap -> full-res.
  assertTrue(cartesianPreviewStride(256, 256, 80) === 1, "256x256x80 -> stride 1 (native res)");
  // A genuinely huge grid still strides to stay under the cap.
  const s = cartesianPreviewStride(512, 512, 100); // 26.2M voxels > 16M cap
  assertTrue(s >= 2, "huge grid -> stride >= 2");
  const reduced = Math.ceil(512 / s) * Math.ceil(512 / s) * 100;
  assertTrue(reduced <= CAP, "strided voxel count is under the cap");
});

check("cartesian stride: searches beyond the old stride-16 soft ceiling", () => {
  const cap = 1_000;
  const stride = cartesianPreviewStride(10_000, 10_000, 1, cap);
  const reduced = Math.ceil(10_000 / stride) ** 2;
  assertTrue(stride > 16, "large plane needs a stride above 16");
  assertTrue(reduced <= cap, "large plane still honours the hard cap");
});

check("cartesian stride: refuses when unreduced z alone exceeds the cap", () => {
  let message = "";
  try {
    cartesianPreviewStride(1, 1, 2_000, 1_000);
  } catch (error) {
    message = error instanceof Error ? error.message : String(error);
  }
  assertTrue(message.includes("without reducing the vertical axis"), "impossible cap is explicit");
});

check("every 3D renderer keeps a beast-class run within one stride step", () => {
  /*
   * The legacy fallback block-averages x/y. The supercell composite sat at a 2M ceiling, which on
   * a 512x512x160 run kept one column in 25 — and one in 100 inside a 2x2 grid, where the
   * budget is quartered. All four renderers now share the volume ceiling.
   */
  const [nx, ny, nz] = [512, 512, 160];
  for (const cap of [
    CARTESIAN_VOXEL_CAP_VOLUME,
    CARTESIAN_VOXEL_CAP_ISOSURFACE,
    CARTESIAN_VOXEL_CAP_STREAMLINES,
    CARTESIAN_VOXEL_CAP_SUPERCELL,
  ]) {
    assertTrue(cartesianPreviewStride(nx, ny, nz, cap) <= 2, "keeps at least one column in four");
    // A quartered pane budget must still not collapse to a coarse lattice.
    assertTrue(cartesianPreviewStride(nx, ny, nz, cap / 4) <= 4, "survives a 2x2 pane budget");
  }
});

// ---------- 12. VTK memory-order repack: z-fast -> x-fast ----------
//
// CartesianVoxelGrid.data is z-fastest (data[ix*ny*nz + iy*nz + iz]); vtkImageData
// indexes points x-fastest: for setDimensions(nx, ny, nz) the tuple for grid point
// (i, j, k) lives at i + j*nx + k*nx*ny (ImageData.js computeIncrements). Feeding
// the z-fast buffer straight into setScalars/setVectors transposed -- and for
// non-cubic grids scrambled -- every 3D VTK plot. cartesianGridToVtkScalars/
// Vectors bridge the conventions. Non-cubic shape (nx != ny != nz) is deliberate:
// a cubic test would pass even with the axes swapped.

check("vtk repack (scalars): z-fast grid becomes x-fast, non-cubic", () => {
  const nx = 2, ny = 3, nz = 4;
  const data = new Float32Array(nx * ny * nz);
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++)
        data[i * ny * nz + j * nz + k] = 100 * i + 10 * j + k; // unique per voxel
  const grid: CartesianVoxelGrid = {
    shape: [nx, ny, nz], spacing: [1, 1, 1], origin: [0, 0, 0], data,
  };
  const out = cartesianGridToVtkScalars(grid);
  assertTrue(out.length === nx * ny * nz, "length preserved");
  assertTrue(out !== data, "repack copies (does not alias the z-fast buffer)");
  let ok = true;
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++)
        if (out[i + j * nx + k * nx * ny] !== 100 * i + 10 * j + k) ok = false;
  assertTrue(ok, "voxel (i,j,k) lands at x-fast offset i + j*nx + k*nx*ny");
  // The exact transpose this fixes: pre-fix vtk read x-offset 1 (the x-neighbor)
  // and got the z-neighbor's value (1); post-fix it holds voxel (1,0,0) = 100.
  assertTrue(out[1] === 100, "x-offset 1 holds voxel (1,0,0)=100 (was z-neighbor 1 pre-fix)");
});

check("vtk repack (vectors): 3-component z-fast becomes x-fast, components in order", () => {
  const nx = 2, ny = 3, nz = 4;
  const data = new Float32Array(nx * ny * nz * 3);
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        const s = (i * ny * nz + j * nz + k) * 3;
        const tag = 100 * i + 10 * j + k;
        data[s] = tag;            // u tags the voxel
        data[s + 1] = tag + 0.5;  // v
        data[s + 2] = tag + 0.25; // w
      }
  const grid: CartesianVoxelGrid = {
    shape: [nx, ny, nz], spacing: [1, 1, 1], origin: [0, 0, 0], data,
  };
  const out = cartesianGridToVtkVectors(grid);
  const inPlaceData = data.slice();
  const inPlace = cartesianGridToVtkVectorsInPlace({ ...grid, data: inPlaceData });
  assertTrue(out.length === nx * ny * nz * 3, "3 interleaved components per voxel");
  assertTrue(
    inPlace === inPlaceData && inPlace.every((value, index) => value === out[index]),
    "exclusive vector repack preserves exact ordering in its original allocation",
  );
  let ok = true;
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        const d = (i + j * nx + k * nx * ny) * 3;
        const tag = 100 * i + 10 * j + k;
        if (out[d] !== tag || out[d + 1] !== tag + 0.5 || out[d + 2] !== tag + 0.25) ok = false;
      }
  assertTrue(ok, "each velocity tuple lands at its x-fast voxel with (u,v,w) preserved");
});

// ---------- 13. cartesianVectorMaxMagnitude (streamline dt sizing) ----------
//
// vtkImageStreamline integrates xyz += dt*velocity, so dt is a time; the
// streamline widget sizes it as (0.5*minSpacing)/maxSpeed. This helper supplies
// maxSpeed and must (a) find the true max |v| and (b) return 0 for an empty/zero
// field so the caller can guard the divide.

check("vector max magnitude: returns the largest |v| (3-4-5) over the grid", () => {
  const nx = 2, ny = 2, nz = 2;
  const data = new Float32Array(nx * ny * nz * 3);
  data[0] = 1; data[1] = 0; data[2] = 0; // |v| = 1
  const big = (1 * ny * nz + 1 * nz + 1) * 3;
  data[big] = 3; data[big + 1] = 4; data[big + 2] = 0; // |v| = 5
  const grid: CartesianVoxelGrid = {
    shape: [nx, ny, nz], spacing: [1, 1, 1], origin: [0, 0, 0], data,
  };
  assertTrue(Math.abs(cartesianVectorMaxMagnitude(grid) - 5) < 1e-6, "max |v| is 5");
});

check("vector max magnitude: all-zero field returns 0 (caller guards divide)", () => {
  const grid: CartesianVoxelGrid = {
    shape: [1, 1, 1], spacing: [1, 1, 1], origin: [0, 0, 0], data: new Float32Array(3),
  };
  assertTrue(cartesianVectorMaxMagnitude(grid) === 0, "zero field -> 0");
});

// ---------- 14. cartesianVectorSpeedQuantile (outlier-robust streamline dt) ----------
//
// The streamline widget sizes dt off a REPRESENTATIVE fast speed (a high
// quantile), not the domain MAX, so a single clamped near-blow-up cell can't
// shrink dt and collapse the tubes. The quantile must (a) track the median of
// the bulk distribution while being immune to a lone 1000 m/s outlier, and (b)
// return 0 for an all-zero field so the caller can guard the divide. The
// histogram is approximate, so assertions allow a tolerance.

check("vector speed quantile: p50 tracks the median, immune to a lone outlier", () => {
  // 4x4x4 = 64 voxels, velocity (n, 0, 0) so |v| = n for voxel n. Voxel 0 has
  // |v| = 0 (ignored as a zero sample). The last voxel is overridden to a
  // 1000 m/s outlier, so the nonzero magnitudes are {1, 2, ..., 62, 1000} (63
  // values) whose true median is the 32nd = 32.
  const nx = 4, ny = 4, nz = 4;
  const n = nx * ny * nz;
  const data = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) {
    data[v * 3] = v; // u = v -> |vel| = v
  }
  data[(n - 1) * 3] = 1000; // lone outlier replaces |v| = 63
  const grid: CartesianVoxelGrid = {
    shape: [nx, ny, nz], spacing: [1, 1, 1], origin: [0, 0, 0], data,
  };

  const trueMedian = 32;
  const p50 = cartesianVectorSpeedQuantile(grid, 0.5);
  assertTrue(
    Math.abs(p50 - trueMedian) <= 0.15 * trueMedian,
    "p50 within 15% of the true median",
  );
  assertTrue(p50 < 100, "p50 well below the 1000 outlier (not dominated by it)");
});

check("vector speed quantile: all-zero field returns 0 (caller guards divide)", () => {
  const grid: CartesianVoxelGrid = {
    shape: [2, 2, 2], spacing: [1, 1, 1], origin: [0, 0, 0], data: new Float32Array(2 * 2 * 2 * 3),
  };
  assertTrue(cartesianVectorSpeedQuantile(grid, 0.5) === 0, "all-zero field -> 0");
});

// ---------- 15. Manifest valid-domain / halo exclusion ----------

check("scalar VTK path excludes declared lateral halo cells", () => {
  const shape: [number, number, number] = [4, 4, 2];
  const data = new Float32Array(shape[0] * shape[1] * shape[2]);
  for (let i = 0; i < shape[0]; i++)
    for (let j = 0; j < shape[1]; j++)
      for (let k = 0; k < shape[2]; k++)
        data[(i * shape[1] + j) * shape[2] + k] = 100 * i + 10 * j + k;
  const out = fieldToCartesianGrid(
    { shape, data },
    {
      drM: 100,
      dyM: 200,
      dzM: 50,
      scalarOriginM: [50, 100, 25],
      validIndexRange: { lower: [1, 1, 0], upper: [2, 2, 1] },
    },
  );
  assertTrue(out.shape.join("x") === "2x2x2", "valid range determines rendered shape");
  assertTrue(out.origin.join(",") === "150,300,25", "origin advances past cropped low halo");
  assertTrue(out.data[0] === 110 && out.data.at(-1) === 221, "only valid cells remain");
});

check("vector VTK path excludes the same halo from every component", () => {
  const shape: [number, number, number] = [4, 4, 2];
  const make = (offset: number): Npy3D => {
    const data = new Float32Array(shape[0] * shape[1] * shape[2]);
    for (let i = 0; i < data.length; i++) data[i] = offset + i;
    return { shape, data };
  };
  const out = vectorFieldToCartesianGrid(
    make(0), make(1000), make(2000),
    {
      drM: 100,
      dyM: 200,
      dzM: 50,
      scalarOriginM: [50, 100, 25],
      validIndexRange: { lower: [1, 1, 0], upper: [2, 2, 1] },
    },
  );
  const firstSource = (1 * 4 + 1) * 2;
  assertTrue(out.shape.join("x") === "2x2x2", "vector valid shape matches scalar path");
  assertTrue(
    out.data[0] === firstSource &&
      out.data[1] === 1000 + firstSource &&
      out.data[2] === 2000 + firstSource,
    "first valid vector tuple preserves aligned components",
  );
});

check("terrain-following fields never fall through the uniform VTK path", () => {
  let threw = false;
  try {
    fieldToCartesianGrid(makeField(2, 2, 2, () => 1), {
      terrainFollowing: true,
      drM: 100,
      dyM: 100,
      dzM: 50,
    });
  } catch (error) {
    threw = (error as Error).message.includes("Terrain-following");
  }
  assertTrue(threw, "terrain field fails closed instead of using nominal dz");
});

check("terrain windows remap by exact physical-volume overlap", () => {
  const field: Npy3D = {
    shape: [1, 1, 2],
    data: new Float32Array([10, 20]),
    windowGeometry: {
      aggregation: "cell_mean",
      weighting: "physical_cell_volume",
      sampleLocation: "physical_volume_centroid",
      centroidsM: {
        x: new Float64Array([50, 50]),
        y: new Float64Array([50, 50]),
        z: new Float64Array([5, 20]),
      },
      volumesM3: new Float64Array([100_000, 200_000]),
      sourceShape: [1, 1, 2],
      origin: [0, 0, 0],
      extent: [1, 1, 2],
      stride: [1, 1, 1],
    },
  };
  const remapped = terrainWindowToCartesianGrid(field, {
    terrainFollowing: true,
    drM: 100,
    dyM: 100,
  });
  assertTrue(remapped.shape.join("x") === "1x1x2", "terrain remap retains the bounded lattice");
  assertTrue(Math.abs(remapped.spacing[2] - 15) < 1e-6, "terrain remap spans the physical column");
  assertTrue(Math.abs(remapped.data[0] - 13.333333) < 1e-5, "first Cartesian cell is overlap weighted");
  assertTrue(remapped.data[1] === 20, "upper Cartesian cell retains the resolved value");
  const sourceIntegral = 10 * 10 + 20 * 20;
  const targetIntegral = remapped.data[0] * 15 + remapped.data[1] * 15;
  assertTrue(Math.abs(sourceIntegral - targetIntegral) < 1e-4, "vertical scalar integral is conserved");

  const vertical = remapTerrainVerticalSlice(field, {
    terrainFollowing: true,
    nr: 1,
    nth: 1,
    nz: 2,
    drM: 100,
    dyM: 100,
  }, 0);
  assertTrue(vertical.remapped, "terrain vertical plot planes use the physical remap");
  assertTrue(vertical.grid?.terrainFollowing === false, "remapped plot coordinates are separable");
  assertTrue(vertical.grid?.scalarCoordinatesM?.[2].join(",") === "7.5,22.5",
    "vertical plot coordinates are physical cell-centre heights");
  const horizontal = remapTerrainVerticalSlice(field, {
    terrainFollowing: true,
    nr: 1,
    nth: 1,
    nz: 2,
  }, 2);
  assertTrue(!horizontal.remapped && horizontal.field === field,
    "horizontal terrain model-level views retain their native plan field");
});

// ---------- Physical extent is invariant to the LOD stride ----------

check("decimated lattices span the same physical extent as the full-resolution box", () => {
  const nxFull = 65;
  const nyFull = 49;
  const nzFull = 5;
  const dxM = 400;
  const dyM = 500;
  const dzM = 250;
  const field = makeField(nxFull, nyFull, nzFull, (i, j, k) => i + j + k);
  const expectedX = (nxFull - 1) * dxM;
  const expectedY = (nyFull - 1) * dyM;
  const expectedZ = (nzFull - 1) * dzM;

  for (const stride of [1, 2, 3, 4, 6, 8]) {
    const grid = cartesianBoxToImageData(field, dxM, dyM, dzM, { stride });
    const spanX = (grid.shape[0] - 1) * grid.spacing[0];
    const spanY = (grid.shape[1] - 1) * grid.spacing[1];
    const spanZ = (grid.shape[2] - 1) * grid.spacing[2];
    assertTrue(
      Math.abs(spanX - expectedX) < 1e-6 * expectedX,
      `stride ${stride} preserves the x extent (${spanX} vs ${expectedX})`,
    );
    assertTrue(
      Math.abs(spanY - expectedY) < 1e-6 * expectedY,
      `stride ${stride} preserves the y extent (${spanY} vs ${expectedY})`,
    );
    assertTrue(
      Math.abs(spanZ - expectedZ) < 1e-6 * expectedZ,
      `stride ${stride} preserves the z extent (${spanZ} vs ${expectedZ})`,
    );
  }
});

check("decimated lattices keep the full-resolution aspect ratio", () => {
  const field = makeField(33, 33, 4, (i, j, k) => i * j + k);
  const full = cartesianBoxToImageData(field, 300, 900, 200, { stride: 1 });
  const aspect = (g: CartesianVoxelGrid): [number, number] => {
    const x = (g.shape[0] - 1) * g.spacing[0];
    const y = (g.shape[1] - 1) * g.spacing[1];
    const z = (g.shape[2] - 1) * g.spacing[2];
    return [y / x, z / x];
  };
  const [fullYX, fullZX] = aspect(full);
  for (const stride of [2, 3, 5]) {
    const [yx, zx] = aspect(cartesianBoxToImageData(field, 300, 900, 200, { stride }));
    assertTrue(
      Math.abs(yx - fullYX) < 1e-6 && Math.abs(zx - fullZX) < 1e-6,
      `stride ${stride} keeps the domain aspect ratio (${yx}, ${zx} vs ${fullYX}, ${fullZX})`,
    );
  }
});

// ---------- Report ----------

console.log(`\ncartesianInterp tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
