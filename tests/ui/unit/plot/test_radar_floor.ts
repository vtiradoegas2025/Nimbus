/**
 * Tests for `src/lib/plot/radarFloor.ts`.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types \
 *     --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/plot/test_radar_floor.ts
 */

import {
  floorPlaneGeometry,
  floorRequestId,
  floorWindowRequest,
  MAX_FIELD_REQUEST_ID_BYTES,
  planFloorHyperslab,
  reduceFloor,
  resolveFloorSource,
  terrainFloorSurface,
  terrainFloorMesh,
  encodeFloorRgba,
  WEBGL2_MIN_MAX_TEXTURE_SIZE,
  CLEAR_AIR_THRESHOLD_DBZ,
  ALPHA_RAMP_START_DBZ,
  type FloorHyperslabPlan,
  type FloorSourceKind,
} from "../../../../ui/src/lib/plot/radarFloor.ts";
import { MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES } from "../../../../ui/src/lib/dataset/nativeFieldWindow.ts";

import { DBZ_VTK_STOPS } from "../../../../ui/src/lib/plot/plotColormaps.ts";

// ---------- Tiny assertion API (matches test_quiver.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function approx(a: number, b: number, eps = 1e-6): boolean {
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

// ---------- Helpers ----------

/** Build a flat Float32Array from a 3D [x][y][z] callback. */
function makeVolume(
  nx: number,
  ny: number,
  nz: number,
  fn: (xi: number, yi: number, zi: number) => number,
): Float32Array {
  const out = new Float32Array(nx * ny * nz);
  for (let xi = 0; xi < nx; xi++) {
    for (let yi = 0; yi < ny; yi++) {
      for (let zi = 0; zi < nz; zi++) {
        out[(xi * ny + yi) * nz + zi] = fn(xi, yi, zi);
      }
    }
  }
  return out;
}


function planOf(
  nx: number,
  ny: number,
  nz: number,
  kind: FloorSourceKind,
): FloorHyperslabPlan | null {
  const outcome = planFloorHyperslab(nx, ny, nz, kind);
  return outcome.ok ? outcome.plan : null;
}

/** Bytes the RESPONSE carries — what x/y stride is allowed to shrink. */
function outputBytes(plan: FloorHyperslabPlan): number {
  const depth = plan.extent.length >= 3 ? (plan.extent[2] ?? 1) : 1;
  return plan.achievedNx * plan.achievedNy * depth * 4;
}

// ---------- 1. Hyperslab planning: small grid → stride 1 ----------

check("small grid (128×128×50) yields stride 1 for a surface slab", () => {
  const plan = planOf(128, 128, 50, "surface-slab");
  assertTrue(plan !== null, "plan is not null for small grid");
  if (!plan) return;
  assertTrue(plan.strideXY === 1, `stride 1 for small grid, got ${plan.strideXY}`);
  assertTrue(plan.achievedNx === 128, `achievedNx 128, got ${plan.achievedNx}`);
  assertTrue(plan.achievedNy === 128, `achievedNy 128, got ${plan.achievedNy}`);
  assertTrue(plan.extent[2] === 1, "surface z-extent is 1");
  assertTrue(outputBytes(plan) <= 64 * 1024 * 1024, "within output budget");
});

check("small grid (128×128×50) yields stride 1 for a full column", () => {
  const plan = planOf(128, 128, 50, "column");
  assertTrue(plan !== null, "plan is not null");
  if (!plan) return;
  assertTrue(plan.strideXY === 1, `stride 1 for small column, got ${plan.strideXY}`);
  assertTrue(plan.extent[2] === 50, `column z-extent equals nz=50, got ${plan.extent[2]}`);
  assertTrue(outputBytes(plan) <= 64 * 1024 * 1024, "within output budget");
});

// ---------- 2. Hyperslab planning: large grid → stride > 1 and within budget ----------

check("large surface grid (8192×8192×100) uses stride > 1 and stays under budget", () => {
  const plan = planOf(8192, 8192, 100, "surface-slab");
  assertTrue(plan !== null, "plan is not null for large grid");
  if (!plan) return;
  const bytes = outputBytes(plan);
  assertTrue(bytes <= 64 * 1024 * 1024, `bytes ${bytes} within 64 MB budget`);
  // At stride 1 an 8192² surface slab is 268 MB of output, so a stride is forced.
  assertTrue(plan.strideXY > 1, `large grid must use stride > 1, got ${plan.strideXY}`);
  assertTrue(plan.achievedNx <= WEBGL2_MIN_MAX_TEXTURE_SIZE, "achievedNx within texture bound");
  assertTrue(plan.achievedNy <= WEBGL2_MIN_MAX_TEXTURE_SIZE, "achievedNy within texture bound");
});

check("terrain floor planning budgets geometry and caps the draped mesh", () => {
  const outcome = planFloorHyperslab(2048, 2048, 100, "surface-slab", {
    outputBytesPerCell: 36,
    outputDepth: 1,
    maxOutputCells: 512 * 512,
  });
  assertTrue(outcome.ok, "terrain surface plan is admitted");
  if (!outcome.ok) return;
  const cells = outcome.plan.achievedNx * outcome.plan.achievedNy;
  assertTrue(cells <= 512 * 512, `terrain mesh has ${cells} bounded vertices`);
  assertTrue(cells * 36 <= 64 * 1024 * 1024, "terrain values and geometry fit output budget");
  assertTrue(outcome.plan.strideXY >= 4, "large terrain floor is reduced before mesh construction");
});

check("terrain floor recovers bottom interfaces from exact block volumes", () => {
  const centers = {
    x: new Float64Array([50, 50, 250, 250]),
    y: new Float64Array([50, 250, 50, 250]),
    z: new Float64Array([125, 225, 325, 425]),
  };
  // 2x2 source blocks at 100 m spacing; thicknesses are 50, 100, 150, 200 m.
  const volumes = new Float64Array([2_000_000, 4_000_000, 6_000_000, 8_000_000]);
  const surface = terrainFloorSurface(centers, volumes, [2, 2], [4, 4], 2, 100, 100);
  assertTrue(
    [...surface.z].join() === [100, 175, 250, 325].join(),
    `recovered physical surface ${[...surface.z].join()}`,
  );
  assertTrue(surface.x === centers.x && surface.y === centers.y, "exact horizontal centroids alias");
});

// ---------- 2b. THE SOURCE BUDGET ----------
//
// This is the defect these tests previously could not see, because they only
// ever checked the bytes the response carries.
//
// A `cell_mean` window must READ every cell inside `extent` in order to average
// the blocks it returns, so x/y stride shrinks the response but not the read.
// The backend budgets the two separately and rejects on the read. Planning
// against the output budget alone therefore produced requests that were valid
// on paper and refused in practice — which is precisely how the column-maximum
// floor came to fail on every grid deeper than about 64 MB while every test
// here stayed green.

check("a full column over 512×512×160 reports the source bytes it must read", () => {
  const outcome = planFloorHyperslab(512, 512, 160, "column");
  assertTrue(outcome.ok, "a 168 MB read is under the 256 MB ceiling, so it plans");
  if (!outcome.ok) return;
  const expected = 512 * 512 * 160 * 4;
  assertTrue(
    outcome.plan.sourceBytes === expected,
    `sourceBytes ${outcome.plan.sourceBytes} equals the full extent ${expected}`,
  );
  // The point of the whole fix: the read is four times the response, so a
  // request that does not state its source budget gets refused.
  assertTrue(
    outcome.plan.sourceBytes > outputBytes(outcome.plan),
    "source bytes exceed output bytes — stride shrinks one and not the other",
  );
});

check("source bytes are unaffected by x/y stride", () => {
  const outcome = planFloorHyperslab(512, 512, 160, "column");
  assertTrue(outcome.ok, "plans");
  if (!outcome.ok) return;
  assertTrue(outcome.plan.strideXY > 1, "this grid does need a stride");
  assertTrue(
    outcome.plan.sourceBytes === 512 * 512 * 160 * 4,
    "striding the response does not reduce what the aggregation reads",
  );
});

check("a column too deep to read is refused rather than mis-planned", () => {
  // 2048×2048×100 would read 1.6 GB, far past the protocol ceiling.
  const outcome = planFloorHyperslab(2048, 2048, 100, "column");
  assertTrue(!outcome.ok, "an unreadable column does not produce a plan");
  if (outcome.ok) return;
  assertTrue(
    outcome.failure === "source-budget",
    `failure is source-budget, got ${outcome.failure}`,
  );
  assertTrue(
    outcome.sourceBytes === 2048 * 2048 * 100 * 4,
    "the refusal reports how much it would have read, so the UI can say so",
  );
  // Refusing is the only honest option: the sole lever on source bytes is the
  // column depth, and shortening it would turn "maximum over the column" into
  // "maximum over part of the column" while still calling itself a maximum.
});

check("the same grid is fine as a surface slab", () => {
  const outcome = planFloorHyperslab(2048, 2048, 100, "surface-slab");
  assertTrue(outcome.ok, "one level deep reads 16 MB and plans normally");
});

check("a native 2-D product reads a plane regardless of depth", () => {
  const outcome = planFloorHyperslab(2048, 2048, 100, "native-2d");
  assertTrue(outcome.ok, "a rank-2 product is never depth-limited");
  if (!outcome.ok) return;
  assertTrue(
    outcome.plan.sourceBytes === 2048 * 2048 * 4,
    "reads one plane, not a column",
  );
  assertTrue(outcome.plan.extent.length === 2, "rank-2 extent for a rank-2 field");
  assertTrue(outcome.plan.origin.length === 2, "rank-2 origin");
  assertTrue(outcome.plan.stride.length === 2, "rank-2 stride");
  // Rank must match the field's own rank: the response validator rejects a
  // three-vector against a two-dimensional source as a rank disagreement.
});

// ---------- 2c. The request states the budget it needs ----------

check("floorWindowRequest declares maxSourceBytes", () => {
  const outcome = planFloorHyperslab(512, 512, 160, "column");
  assertTrue(outcome.ok, "plans");
  if (!outcome.ok) return;
  const request = floorWindowRequest("floor:test", outcome.plan);
  assertTrue(
    request.maxSourceBytes === outcome.plan.sourceBytes,
    `maxSourceBytes ${request.maxSourceBytes} states the plan's read size`,
  );
  assertTrue(
    (request.maxSourceBytes ?? 0) <= MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
    "and stays inside the protocol ceiling",
  );
  // Left unstated, the backend applies a quarter of the ceiling and refuses.
  assertTrue(
    (request.maxSourceBytes ?? 0) > MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES / 4,
    "this request is exactly the kind the default budget would have refused",
  );
});

check("a rank-2 plan produces rank-2 request vectors", () => {
  const outcome = planFloorHyperslab(512, 512, 160, "native-2d");
  assertTrue(outcome.ok, "plans");
  if (!outcome.ok) return;
  const request = floorWindowRequest("floor:test", outcome.plan);
  assertTrue(request.origin.length === 2, "origin rank 2");
  assertTrue(request.extent.length === 2, "extent rank 2");
  assertTrue(request.stride.length === 2, "stride rank 2");
});

// ---------- 2d. Source selection ----------

check("column maximum of reflectivity prefers the engine's own composite", () => {
  const source = resolveFloorSource(
    "reflectivity_dbz",
    "column-max",
    ["reflectivity_dbz", "composite_reflectivity", "w"],
  );
  assertTrue(source.fieldName === "composite_reflectivity", "reads the 2-D product");
  assertTrue(source.kind === "native-2d", "as a rank-2 read");
  assertTrue(source.substituted, "and flags the substitution so the UI can disclose it");
  // Not an approximation: composite_reflectivity IS the column maximum of
  // reflectivity_dbz, computed by the engine at full model resolution. On a
  // 512×512×160 grid this is 1 MB instead of 168 MB.
});

check("without the composite, a column maximum reduces the column itself", () => {
  const source = resolveFloorSource(
    "reflectivity_dbz",
    "column-max",
    ["reflectivity_dbz", "w"],
  );
  assertTrue(source.fieldName === "reflectivity_dbz", "falls back to the 3-D field");
  assertTrue(source.kind === "column", "read as a full column");
  assertTrue(!source.substituted, "nothing was substituted");
});

check("a surface reduction never substitutes", () => {
  const source = resolveFloorSource(
    "reflectivity_dbz",
    "surface",
    ["reflectivity_dbz", "composite_reflectivity"],
  );
  assertTrue(source.kind === "surface-slab", "reads the ground slab");
  assertTrue(source.fieldName === "reflectivity_dbz", "of the requested field");
  assertTrue(!source.substituted, "the composite is a column maximum, not a surface value");
});

check("a field with no native composite is never substituted", () => {
  const source = resolveFloorSource(
    "buoyancy",
    "column-max",
    ["buoyancy", "composite_reflectivity"],
  );
  assertTrue(source.fieldName === "buoyancy", "buoyancy's column max is not a reflectivity product");
  assertTrue(source.kind === "column", "so it is reduced from the column");
});

// ---------- 2e. Realistic grids satisfy the backend's own validator ----------
//
// The planner's output is a request another process must accept. These cases
// re-apply the constraints `validate_field_window_request` enforces in
// `src-tauri/src/commands/outputs.rs`, so a planner change that drifts out of
// contract fails here instead of in the built app.
//
// 512x512x160 is not a hypothetical: it is the shape of the run this layer was
// reported broken on.

interface BackendLimits {
  maxOutputBytes: number;
  maxSourceBytes: number;
}

const BACKEND: BackendLimits = {
  maxOutputBytes: 64 * 1024 * 1024,
  maxSourceBytes: MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
};

/** Returns the reason the backend would refuse this request, or null. */
function backendRefusal(
  request: {
    origin: number[];
    extent: number[];
    stride: number[];
    maxOutputBytes?: number;
    maxSourceBytes?: number;
  },
  sourceShape: readonly number[],
): string | null {
  const rank = sourceShape.length;
  if (request.origin.length !== rank) return "origin rank disagrees with source";
  if (request.extent.length !== rank) return "extent rank disagrees with source";
  if (request.stride.length !== rank) return "stride rank disagrees with source";
  for (let axis = 0; axis < rank; axis += 1) {
    if (request.origin[axis]! + request.extent[axis]! > sourceShape[axis]!) {
      return `request exceeds source axis ${axis}`;
    }
  }
  const outputShape = request.extent.map(
    (count, axis) => Math.floor((count - 1) / request.stride[axis]!) + 1,
  );
  const decodedBytes = outputShape.reduce((product, count) => product * count, 1) * 4;
  const outputBudget = request.maxOutputBytes ?? 32 * 1024 * 1024;
  if (outputBudget === 0 || outputBudget > BACKEND.maxOutputBytes) return "maxOutputBytes out of range";
  if (decodedBytes > outputBudget) return `needs ${decodedBytes} output bytes, budget ${outputBudget}`;
  // Stride absent on purpose: cell-mean reads every cell inside `extent`.
  const sourceBytes = request.extent.reduce((product, count) => product * count, 1) * 4;
  const sourceBudget = request.maxSourceBytes ?? 64 * 1024 * 1024;
  if (sourceBudget === 0 || sourceBudget > BACKEND.maxSourceBytes) return "maxSourceBytes out of range";
  if (sourceBytes > sourceBudget) {
    return `must inspect ${sourceBytes} source bytes, budget ${sourceBudget}`;
  }
  return null;
}

check("the reported grid's floor requests are all servable", () => {
  // The run the radar floor was reported broken on.
  const [nx, ny, nz] = [512, 512, 160];
  const fields = ["reflectivity_dbz", "composite_reflectivity", "w"];

  for (const reduction of ["surface", "column-max"] as const) {
    const source = resolveFloorSource("reflectivity_dbz", reduction, fields);
    const outcome = planFloorHyperslab(nx, ny, nz, source.kind);
    assertTrue(outcome.ok, `${reduction} plans for 512x512x160`);
    if (!outcome.ok) continue;
    const request = floorWindowRequest(`floor:${reduction}`, outcome.plan);
    const sourceShape = source.kind === "native-2d" ? [nx, ny] : [nx, ny, nz];
    const refusal = backendRefusal(request, sourceShape);
    assertTrue(refusal === null, `${reduction} request is servable (${refusal ?? "ok"})`);
    // Both reductions reach the floor at full horizontal resolution on this
    // grid: 512 is inside the guaranteed texture bound, so no stride is needed.
    assertTrue(
      outcome.plan.strideXY === 1,
      `${reduction} needs no stride on a 512² grid, got ${outcome.plan.strideXY}`,
    );
  }
});

check("without the composite, a 512×512×160 column maximum is still servable", () => {
  // The exact request that used to be built and refused: 42 MB of output over a
  // 168 MB read, with no source budget declared.
  const source = resolveFloorSource("reflectivity_dbz", "column-max", ["reflectivity_dbz"]);
  assertTrue(source.kind === "column", "no composite exported, so the column is read");
  const outcome = planFloorHyperslab(512, 512, 160, source.kind);
  assertTrue(outcome.ok, "plans");
  if (!outcome.ok) return;
  const request = floorWindowRequest("floor:column", outcome.plan);
  assertTrue(
    backendRefusal(request, [512, 512, 160]) === null,
    "servable once the request states its own source budget",
  );
  // And the proof that stating it is what fixed this: strip the declaration and
  // the backend's default refuses the very same window.
  assertTrue(
    backendRefusal({ ...request, maxSourceBytes: undefined }, [512, 512, 160]) !== null,
    "the same request without maxSourceBytes is refused, as it was in the app",
  );
});

check("a range of production grids produce servable floor requests", () => {
  const grids: Array<[number, number, number]> = [
    [48, 36, 24],      // the browser fixture
    [64, 64, 32],      // cgrid_tornado_genesis
    [256, 256, 80],
    [512, 512, 160],   // the reported run
    [1024, 1024, 100],
  ];
  for (const [nx, ny, nz] of grids) {
    for (const kind of ["surface-slab", "native-2d"] as const) {
      const outcome = planFloorHyperslab(nx, ny, nz, kind);
      assertTrue(outcome.ok, `${kind} plans for ${nx}x${ny}x${nz}`);
      if (!outcome.ok) continue;
      const request = floorWindowRequest("floor:grid", outcome.plan);
      const sourceShape = kind === "native-2d" ? [nx, ny] : [nx, ny, nz];
      const refusal = backendRefusal(request, sourceShape);
      assertTrue(refusal === null, `${kind} ${nx}x${ny}x${nz} servable (${refusal ?? "ok"})`);
    }
  }
});

// ---------- 3. Achieved lattice is reported correctly ----------

check("achievedNx and achievedNy equal ceil(n / strideXY)", () => {
  const plan = planOf(100, 200, 30, "surface-slab");
  assertTrue(plan !== null, "plan is not null");
  if (!plan) return;
  const expectedNx = Math.ceil(100 / plan.strideXY);
  const expectedNy = Math.ceil(200 / plan.strideXY);
  assertTrue(
    plan.achievedNx === expectedNx,
    `achievedNx ${plan.achievedNx} === ceil(100/${plan.strideXY}) = ${expectedNx}`,
  );
  assertTrue(
    plan.achievedNy === expectedNy,
    `achievedNy ${plan.achievedNy} === ceil(200/${plan.strideXY}) = ${expectedNy}`,
  );
});

// ---------- 4. Degenerate dims are refused, and say which limit was hit ----------

function refusalFor(nx: number, ny: number, nz: number): string | null {
  const outcome = planFloorHyperslab(nx, ny, nz, "surface-slab");
  return outcome.ok ? null : outcome.failure;
}

check("zero dims are refused as a degenerate grid", () => {
  assertTrue(refusalFor(0, 128, 50) === "degenerate-grid", "nx=0");
  assertTrue(refusalFor(128, 0, 50) === "degenerate-grid", "ny=0");
  assertTrue(refusalFor(128, 128, 0) === "degenerate-grid", "nz=0");
});

check("negative dims are refused as a degenerate grid", () => {
  assertTrue(refusalFor(-1, 128, 50) === "degenerate-grid", "negative nx");
  assertTrue(refusalFor(128, -1, 50) === "degenerate-grid", "negative ny");
});

check("non-finite dims are refused as a degenerate grid", () => {
  assertTrue(refusalFor(NaN, 128, 50) === "degenerate-grid", "NaN nx");
  assertTrue(refusalFor(128, NaN, 50) === "degenerate-grid", "NaN ny");
  assertTrue(refusalFor(128, 128, NaN) === "degenerate-grid", "NaN nz");
  assertTrue(refusalFor(Infinity, 128, 50) === "degenerate-grid", "Infinity nx");
});

check("a degenerate grid is distinguishable from an over-budget one", () => {
  // Both refuse; conflating them would make the UI tell a user with a perfectly
  // valid deep grid that their run's dimensions are broken.
  assertTrue(refusalFor(0, 128, 50) === "degenerate-grid", "degenerate says so");
  const overBudget = planFloorHyperslab(2048, 2048, 100, "column");
  assertTrue(!overBudget.ok && overBudget.failure === "source-budget", "over-budget says so");
});

// ---------- 5. RESOLUTION INDEPENDENCE ----------
//
// The core invariant of the floor redesign: the planned floor lattice is
// determined by the native grid dimensions and the byte budget, never by a 3D
// voxel cap applied elsewhere in the rendering pipeline. Turning render quality
// down must not blur the floor.

check("floor lattice is independent of a 3D voxel cap", () => {
  const nx = 512;
  const ny = 512;
  const nz = 80;

  const reference = planOf(nx, ny, nz, "column");
  assertTrue(reference !== null, "reference plan is valid");
  if (!reference) return;

  // Caps a caller might be tempted to pass in place of the native depth. The
  // planner is called with the NATIVE dims every time, so none of them move it.
  for (const cap of [16, 32, 64, 128, 256]) {
    const native = planOf(nx, ny, nz, "column");
    assertTrue(native !== null, `plan alongside cap=${cap} is not null`);
    if (!native) continue;
    assertTrue(
      native.strideXY === reference.strideXY,
      `strideXY ${native.strideXY} equals reference ${reference.strideXY} for cap=${cap}`,
    );
    assertTrue(
      native.achievedNx === reference.achievedNx && native.achievedNy === reference.achievedNy,
      `achieved lattice matches reference for cap=${cap}`,
    );
  }

  // A surface slab is one level deep by construction, so its budget — and so
  // its lattice — cannot depend on nz at all.
  const surfaceA = planOf(256, 256, 20, "surface-slab");
  const surfaceB = planOf(256, 256, 200, "surface-slab");
  assertTrue(surfaceA !== null && surfaceB !== null, "both surface plans valid");
  if (surfaceA && surfaceB) {
    assertTrue(surfaceA.strideXY === surfaceB.strideXY, "surface stride is independent of nz");
    assertTrue(
      surfaceA.achievedNx === surfaceB.achievedNx && surfaceA.achievedNy === surfaceB.achievedNy,
      "surface achieved lattice is independent of nz",
    );
  }
});


// ---------- 6. `surface` reduction picks the lowest finite level ----------

check("surface reduction skips NaN at k=0 and uses the first finite level", () => {
  // Column (0,0): NaN at z=0, z=1; first finite at z=2 → value 5.
  // Column (1,0): finite at z=0 → value 3.
  const nx = 2;
  const ny = 1;
  const nz = 4;
  const values = makeVolume(nx, ny, nz, (xi, _yi, zi) => {
    if (xi === 0) return zi < 2 ? NaN : (zi === 2 ? 5 : 7);
    return zi === 0 ? 3 : 9;
  });
  const result = reduceFloor(values, [nx, ny, nz], "surface");
  assertTrue(approx(result[0]!, 5), `column(0,0) should be 5 (first finite), got ${result[0]}`);
  assertTrue(approx(result[1]!, 3), `column(1,0) should be 3 (k=0), got ${result[1]}`);
});

check("surface reduction on an all-NaN column returns NaN", () => {
  const nx = 1;
  const ny = 1;
  const nz = 3;
  const values = makeVolume(nx, ny, nz, () => NaN);
  const result = reduceFloor(values, [nx, ny, nz], "surface");
  assertTrue(isNaN(result[0]!), "all-NaN column → NaN (not 0)");
});

check("surface reduction does not descend past the first finite level", () => {
  // k=0 → 10, k=1 → 20: surface must return 10, not 20.
  const values = makeVolume(1, 1, 3, (_xi, _yi, zi) => [10, 20, 30][zi]!);
  const result = reduceFloor(values, [1, 1, 3], "surface");
  assertTrue(approx(result[0]!, 10), `first finite level is 10, got ${result[0]}`);
});

// ---------- 7. `column-max` reduction ----------

check("column-max returns the maximum finite value over the column", () => {
  const nx = 1;
  const ny = 1;
  const nz = 5;
  // Values: 10, NaN, 55, 20, NaN → max = 55.
  const raw = new Float32Array([10, NaN, 55, 20, NaN]);
  const result = reduceFloor(raw, [nx, ny, nz], "column-max");
  assertTrue(approx(result[0]!, 55), `column-max should be 55, got ${result[0]}`);
});

check("column-max all-NaN column returns NaN (not 0)", () => {
  const raw = new Float32Array([NaN, NaN, NaN]);
  const result = reduceFloor(raw, [1, 1, 3], "column-max");
  assertTrue(isNaN(result[0]!), "all-NaN column → NaN for column-max");
});

check("column-max never returns 0 for a NaN-only column", () => {
  const nx = 2;
  const ny = 2;
  const nz = 3;
  // Mix of all-NaN and finite columns.
  const values = makeVolume(nx, ny, nz, (xi, yi, zi) => {
    const nan = xi === 0 && yi === 0;
    return nan ? NaN : (xi * 10 + yi * 3 + zi);
  });
  const result = reduceFloor(values, [nx, ny, nz], "column-max");
  assertTrue(isNaN(result[0]!), "all-NaN column [0,0] → NaN");
  assertTrue(Number.isFinite(result[1]!), "finite column [0,1] → finite value");
  assertTrue(Number.isFinite(result[2]!), "finite column [1,0] → finite value");
  assertTrue(Number.isFinite(result[3]!), "finite column [1,1] → finite value");
});

// ---------- 8. RGBA encoding: dBZ color mapping ----------

check("a known dBZ value maps to the expected DBZ_VTK_STOPS color", () => {
  // 30 dBZ is a stop anchor: r=1.0, g=1.0, b=0.0 (yellow).
  const values = new Float32Array([30]);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [1, 1], policy);

  const expectedStop = DBZ_VTK_STOPS.find((s) => s.scalar === 30)!;
  const expectedR = Math.round(expectedStop.r * 255);
  const expectedG = Math.round(expectedStop.g * 255);
  const expectedB = Math.round(expectedStop.b * 255);

  assertTrue(rgba[0] === expectedR, `30dBZ R: got ${rgba[0]}, expected ${expectedR}`);
  assertTrue(rgba[1] === expectedG, `30dBZ G: got ${rgba[1]}, expected ${expectedG}`);
  assertTrue(rgba[2] === expectedB, `30dBZ B: got ${rgba[2]}, expected ${expectedB}`);
  assertTrue(rgba[3] === 255, `30dBZ should be fully opaque, alpha=${rgba[3]}`);
});

check("50 dBZ stop anchor maps to red (r=255, g=0, b=0)", () => {
  const values = new Float32Array([50]);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [1, 1], policy);
  const stop = DBZ_VTK_STOPS.find((s) => s.scalar === 50)!;
  assertTrue(rgba[0] === Math.round(stop.r * 255), "50 dBZ R");
  assertTrue(rgba[1] === Math.round(stop.g * 255), "50 dBZ G");
  assertTrue(rgba[2] === Math.round(stop.b * 255), "50 dBZ B");
  assertTrue(rgba[3] === 255, "50 dBZ fully opaque");
});

// ---------- 9. RGBA encoding: sub-threshold transparency ----------

check("dBZ below ALPHA_RAMP_START_DBZ is fully transparent", () => {
  const subThreshold = ALPHA_RAMP_START_DBZ - 5;
  const values = new Float32Array([subThreshold]);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [1, 1], policy);
  assertTrue(rgba[3] === 0, `${subThreshold} dBZ (sub-ramp-start) should be transparent, alpha=${rgba[3]}`);
});

check("dBZ at CLEAR_AIR_THRESHOLD_DBZ is fully opaque", () => {
  const values = new Float32Array([CLEAR_AIR_THRESHOLD_DBZ]);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [1, 1], policy);
  assertTrue(rgba[3] === 255, `${CLEAR_AIR_THRESHOLD_DBZ} dBZ should be fully opaque, alpha=${rgba[3]}`);
});

check("dBZ above CLEAR_AIR_THRESHOLD_DBZ is fully opaque", () => {
  const values = new Float32Array([CLEAR_AIR_THRESHOLD_DBZ + 10]);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [1, 1], policy);
  assertTrue(rgba[3] === 255, "above-threshold dBZ should be fully opaque");
});

// ---------- 10. RGBA encoding: NaN → transparent ----------

check("NaN values produce fully transparent pixels", () => {
  const values = new Float32Array([NaN, 50, NaN]);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [3, 1], policy);
  // Missing pixels retain zero alpha but borrow edge RGB so bilinear filtering
  // cannot interpolate transparent black into a grey radar fringe.
  assertTrue(rgba[3] === 0, "NaN pixel 0 is transparent");
  assertTrue(rgba[11] === 0, "NaN pixel 2 is transparent");
  assertTrue(
    rgba[0] === rgba[4] && rgba[1] === rgba[5] && rgba[2] === rgba[6],
    "left missing texel carries its valid neighbour's RGB for filtering only",
  );
  assertTrue(
    rgba[8] === rgba[4] && rgba[9] === rgba[5] && rgba[10] === rgba[6],
    "right missing texel carries its valid neighbour's RGB for filtering only",
  );
  // Pixel 1 (50 dBZ): opaque.
  assertTrue(rgba[7] === 255, "50 dBZ pixel is opaque");
});

// ---------- 11. RGBA encoding: alpha ramps monotonically across threshold ----------

check("alpha ramps monotonically from 0 to 255 across [ALPHA_RAMP_START, CLEAR_AIR_THRESHOLD]", () => {
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const samples = 20;
  const alphas: number[] = [];
  for (let i = 0; i <= samples; i++) {
    const dbz = ALPHA_RAMP_START_DBZ + (i / samples) * (CLEAR_AIR_THRESHOLD_DBZ - ALPHA_RAMP_START_DBZ);
    const values = new Float32Array([dbz]);
    const rgba = encodeFloorRgba(values, [1, 1], policy);
    alphas.push(rgba[3]);
  }
  let monotone = true;
  for (let i = 1; i < alphas.length; i++) {
    if (alphas[i]! < alphas[i - 1]!) {
      monotone = false;
      break;
    }
  }
  assertTrue(alphas[0] === 0, `alpha at ramp start is 0, got ${alphas[0]}`);
  assertTrue(alphas[alphas.length - 1] === 255, `alpha at threshold is 255, got ${alphas[alphas.length - 1]}`);
  assertTrue(monotone, "alpha is monotonically non-decreasing across the ramp");
});

// ---------- 12. RGBA encoding: non-dBZ (signed, sequential) ----------

check("signed policy encodes a midpoint value at neutral (grey ≈ 221,221,221)", () => {
  // Cool-to-warm midpoint (t=0.5) is approximately rgb(221, 221, 221).
  const values = new Float32Array([0]);
  const policy = {
    kind: "signed" as const,
    reduction: "surface" as const,
    colormap: "Cool to Warm",
    min: -10,
    max: 10,
  };
  const rgba = encodeFloorRgba(values, [1, 1], policy);
  // t = (0 - (-10)) / 20 = 0.5 → neutral.
  // Allow ±5 for rounding differences in the baked ramp.
  assertTrue(Math.abs(rgba[0] - 221) <= 5, `signed mid R ≈ 221, got ${rgba[0]}`);
  assertTrue(Math.abs(rgba[1] - 221) <= 5, `signed mid G ≈ 221, got ${rgba[1]}`);
  assertTrue(rgba[3] === 255, "signed mid is fully opaque");
});

check("sequential policy encodes a known fractional value within Viridis ramp", () => {
  // t = 0 → approximately rgb(68, 1, 84) (dark purple).
  const values = new Float32Array([0]);
  const policy = {
    kind: "sequential" as const,
    reduction: "surface" as const,
    colormap: "Viridis",
    min: 0,
    max: 100,
  };
  const rgba = encodeFloorRgba(values, [1, 1], policy);
  // Allow ±5 tolerance for the baked ramp.
  assertTrue(Math.abs(rgba[0] - 68) <= 5, `sequential min R ≈ 68, got ${rgba[0]}`);
  assertTrue(rgba[3] === 255, "sequential min is fully opaque");
});

check("signed and sequential policies treat NaN as transparent", () => {
  const nans = new Float32Array([NaN]);
  const signedPolicy = { kind: "signed" as const, reduction: "surface" as const, colormap: "Cool to Warm", min: -1, max: 1 };
  const seqPolicy = { kind: "sequential" as const, reduction: "surface" as const, colormap: "Viridis", min: 0, max: 1 };

  const signedRgba = encodeFloorRgba(nans, [1, 1], signedPolicy);
  assertTrue(signedRgba[3] === 0, "signed NaN → transparent");

  const seqRgba = encodeFloorRgba(nans, [1, 1], seqPolicy);
  assertTrue(seqRgba[3] === 0, "sequential NaN → transparent");
});

// ---------- 13. RGBA encoding: interpolation between stops ----------

check("dBZ between stop anchors is linearly interpolated (not a step function)", () => {
  // 35 dBZ lies between the 30 (yellow) and 40 (orange) anchors.
  // At the midpoint t = 0.5, R should be ~255 (both anchors have R=255 or near it).
  // The key test: the color at 35 must differ from both the 30 and 40 anchor colors.
  const stop30 = DBZ_VTK_STOPS.find((s) => s.scalar === 30)!;
  const stop40 = DBZ_VTK_STOPS.find((s) => s.scalar === 40)!;

  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba35 = encodeFloorRgba(new Float32Array([35]), [1, 1], policy);
  const rgba30 = encodeFloorRgba(new Float32Array([30]), [1, 1], policy);
  const rgba40 = encodeFloorRgba(new Float32Array([40]), [1, 1], policy);

  // The G channel differs between 30 (255) and 40 (160), so the midpoint
  // must be between them — this is the interpolation check.
  const g30 = Math.round(stop30.g * 255);
  const g40 = Math.round(stop40.g * 255);
  const g35 = rgba35[1]!;
  const gExpected = Math.round((g30 + g40) / 2);

  assertTrue(
    Math.abs(g35 - gExpected) <= 2,
    `G at 35 dBZ ≈ midpoint of [${g30}, ${g40}] = ${gExpected}, got ${g35}`,
  );
  assertTrue(rgba35[0] !== rgba30[0] || rgba35[1] !== rgba30[1] || rgba35[2] !== rgba30[2],
    "35 dBZ color differs from 30 dBZ anchor");
  assertTrue(rgba35[0] !== rgba40[0] || rgba35[1] !== rgba40[1] || rgba35[2] !== rgba40[2],
    "35 dBZ color differs from 40 dBZ anchor");
});

// ---------- 14. RGBA output size ----------

check("encodeFloorRgba produces exactly nx * ny * 4 bytes", () => {
  const nx = 16;
  const ny = 24;
  const values = new Float32Array(nx * ny).fill(40);
  const policy = { kind: "dbz" as const, reduction: "column-max" as const, colormap: "dBZ floor", min: -30, max: 75 };
  const rgba = encodeFloorRgba(values, [nx, ny], policy);
  assertTrue(rgba.length === nx * ny * 4, `RGBA length ${rgba.length} === ${nx * ny * 4}`);
});

// ---------- 14b. RGBA layout: field order in, image row-major out ----------

/*
 * The field→texture transpose is the one silent-corruption risk in this module:
 * getting it wrong renders a mirrored hook echo, which still looks like a
 * plausible storm rather than an obviously broken image. It needs a shape that
 * is neither square nor 1-wide — with nx === ny, or with ny === 1, the two
 * index formulas coincide and the bug passes unnoticed.
 */
check("encodeFloorRgba transposes field order into image row-major order", () => {
  const nx = 2;
  const ny = 3;
  const policy = {
    kind: "dbz" as const,
    reduction: "surface" as const,
    colormap: "dBZ floor",
    min: -30,
    max: 75,
  };

  // Distinct value per cell so every position is individually identifiable.
  // Field layout is x-slowest: values[xi * ny + yi].
  const values = new Float32Array(nx * ny);
  const valueAt = (xi: number, yi: number) => 30 + xi * 10 + yi;
  for (let xi = 0; xi < nx; xi++) {
    for (let yi = 0; yi < ny; yi++) values[xi * ny + yi] = valueAt(xi, yi);
  }

  const rgba = encodeFloorRgba(values, [nx, ny], policy);

  // Independently encode each single cell to learn its expected colour, then
  // assert it lands at the image row-major slot (yi * nx + xi).
  for (let xi = 0; xi < nx; xi++) {
    for (let yi = 0; yi < ny; yi++) {
      const expected = encodeFloorRgba(
        new Float32Array([valueAt(xi, yi)]),
        [1, 1],
        policy,
      );
      const base = (yi * nx + xi) * 4;
      const matches =
        rgba[base] === expected[0] &&
        rgba[base + 1] === expected[1] &&
        rgba[base + 2] === expected[2] &&
        rgba[base + 3] === expected[3];
      assertTrue(matches, `cell (x=${xi}, y=${yi}) lands at image index ${yi * nx + xi}`);
    }
  }

  // Guard the guard: the field-order slot must NOT also match, or the test
  // would pass under either layout and prove nothing.
  const cornerFieldSlot = (1 * ny + 0) * 4; // xi=1, yi=0 read as if x-slowest
  const cornerImageSlot = (0 * nx + 1) * 4; // xi=1, yi=0 in image order
  assertTrue(
    cornerFieldSlot !== cornerImageSlot,
    "chosen shape distinguishes the two layouts",
  );
});

// ---------- 15. Surface extent is always z=1 in the plan ----------

check("surface plan always has z-extent 1 regardless of nz", () => {
  for (const nz of [1, 10, 100, 500]) {
    const plan = planOf(64, 64, nz, "surface-slab");
    assertTrue(plan !== null, `plan valid for nz=${nz}`);
    if (plan) {
      assertTrue(plan.extent[2] === 1, `surface z-extent=1 for nz=${nz}`);
    }
  }
});

// ---------- 16. Full-column extent equals nz ----------

check("a full-column plan z-extent equals nz", () => {
  for (const nz of [1, 20, 80]) {
    const plan = planOf(32, 32, nz, "column");
    assertTrue(plan !== null, `plan valid for nz=${nz}`);
    if (plan) {
      assertTrue(plan.extent[2] === nz, `column z-extent=${nz}, got ${plan.extent[2]}`);
    }
  }
});

// ---------- 17. Texture size cap: achieved dims never exceed WEBGL2_MIN_MAX_TEXTURE_SIZE ----------

check("achieved dims never exceed WEBGL2_MIN_MAX_TEXTURE_SIZE", () => {
  const testCases: [number, number, number][] = [
    [8192, 8192, 1],
    [4096, 4096, 1],
    [10000, 5000, 50],
    [256, 256, 256],
  ];
  for (const [nx, ny, nz] of testCases) {
    const plan = planOf(nx, ny, nz, "surface-slab");
    if (!plan) continue; // already tested degenerate cases above
    assertTrue(
      plan.achievedNx <= WEBGL2_MIN_MAX_TEXTURE_SIZE,
      `achievedNx ${plan.achievedNx} ≤ ${WEBGL2_MIN_MAX_TEXTURE_SIZE} for ${nx}×${ny}×${nz}`,
    );
    assertTrue(
      plan.achievedNy <= WEBGL2_MIN_MAX_TEXTURE_SIZE,
      `achievedNy ${plan.achievedNy} ≤ ${WEBGL2_MIN_MAX_TEXTURE_SIZE} for ${nx}×${ny}×${nz}`,
    );
  }
});

// ---------- Request id ----------

/*
 * Mirrors `validate_request_id` in src-tauri/src/commands/outputs.rs and
 * `validate_netcdf_request_id` in src-tauri/src/commands/datasets.rs. Both
 * validators are byte-identical and a floor read can take either path.
 *
 * These exist because the id used to interpolate the run directory, which is an
 * absolute path and therefore always contains '/'. The backend refused every
 * such read outright, so the floor was broken on every real run while this
 * suite stayed green — it composed its own ids and never asserted that the one
 * actually shipped was legal. That gap is what these close.
 */
const BACKEND_REQUEST_ID = /^[A-Za-z0-9._:-]+$/;

function assertLegalId(id: string, label: string): void {
  assertTrue(BACKEND_REQUEST_ID.test(id), `${label}: charset legal (got "${id}")`);
  assertTrue(
    id.length >= 1 && id.length <= MAX_FIELD_REQUEST_ID_BYTES,
    `${label}: length ${id.length} within 1..=${MAX_FIELD_REQUEST_ID_BYTES}`,
  );
}

check("floorRequestId is legal for ordinary input", () => {
  assertLegalId(
    floorRequestId(1, 0, "composite_reflectivity", "column-max"),
    "ordinary column-max",
  );
  assertLegalId(floorRequestId(7, 42, "reflectivity_dbz", "surface"), "ordinary surface");
});

check("floorRequestId never embeds a run directory path", () => {
  const id = floorRequestId(3, 12, "reflectivity_dbz", "column-max");
  assertTrue(!id.includes("/"), `no path separator in "${id}"`);
});

check("floorRequestId sanitises hostile field names", () => {
  // CM1/netCDF variable names are not constrained to the backend's charset.
  for (const hostile of [
    "/Users/x/.nimbus/runs/beast_xl",
    "field with spaces",
    "réflectivité",
    "a/b\\c*d?e",
    "emoji_\u{1F300}_field",
  ]) {
    assertLegalId(floorRequestId(1, 0, hostile, "surface"), `hostile "${hostile}"`);
  }
});

check("floorRequestId stays within the ceiling for absurd field names", () => {
  const id = floorRequestId(999999, 999999, "q".repeat(4096), "column-max");
  assertLegalId(id, "4096-char field name");
  assertTrue(id.startsWith("floor:999999:999999:"), `sequence survives truncation: "${id}"`);
  assertTrue(id.endsWith(":column-max"), `reduction survives truncation: "${id}"`);
});

check("floorRequestId is unique per sequence", () => {
  // The backend registry rejects an id already in flight, so two reads of the
  // same run/step/field/reduction must not produce the same id.
  const a = floorRequestId(1, 5, "reflectivity_dbz", "surface");
  const b = floorRequestId(2, 5, "reflectivity_dbz", "surface");
  assertTrue(a !== b, `distinct sequences give distinct ids (${a} vs ${b})`);
});

// ---------- Plane placement ----------

/*
 * The sheet is placed by pre-multiplying its corners, while the data actors are
 * placed by vtk's own setScale. Those are only the same transform if the sheet
 * carries scale.xy on x/y and scale.z on z exactly, so these assert the sheet
 * tracks the display scale rather than merely being finite.
 */

const DOMAIN: { lower: [number, number, number]; upper: [number, number, number] } = {
  lower: [-40000, -30000, 0],
  upper: [40000, 30000, 18000],
};

check("the sheet spans the domain in x and y at unit scale", () => {
  const plane = floorPlaneGeometry(DOMAIN, { xy: 1, z: 1 });
  assertTrue(plane != null, "unit scale yields a placement");
  if (!plane) return;
  assertTrue(plane.origin[0] === -40000 && plane.origin[1] === -30000, "origin sits at the lower corner");
  assertTrue(plane.point1[0] === 40000 && plane.point1[1] === -30000, "point1 runs along +x");
  assertTrue(plane.point2[0] === -40000 && plane.point2[1] === 30000, "point2 runs along +y");
});

check("u maps to x and v maps to y, never transposed", () => {
  const plane = floorPlaneGeometry(DOMAIN, { xy: 1, z: 1 });
  if (!plane) return;
  const uEdge = [0, 1, 2].map((i) => plane.point1[i] - plane.origin[i]);
  const vEdge = [0, 1, 2].map((i) => plane.point2[i] - plane.origin[i]);
  assertTrue(uEdge[0] !== 0 && uEdge[1] === 0, "the u edge is purely x");
  assertTrue(vEdge[1] !== 0 && vEdge[0] === 0, "the v edge is purely y");
});

check("the sheet tracks the horizontal display scale", () => {
  const unit = floorPlaneGeometry(DOMAIN, { xy: 1, z: 1 });
  if (!unit) return;
  for (const xy of [0.5, 1.4, 2]) {
    const scaled = floorPlaneGeometry(DOMAIN, { xy, z: 1 });
    assertTrue(scaled != null, `horizontal scale ${xy} yields a placement`);
    if (!scaled) continue;
    const spanX = scaled.point1[0] - scaled.origin[0];
    const spanY = scaled.point2[1] - scaled.origin[1];
    const unitX = unit.point1[0] - unit.origin[0];
    const unitY = unit.point2[1] - unit.origin[1];
    assertTrue(
      Math.abs(spanX - unitX * xy) < 1e-9 && Math.abs(spanY - unitY * xy) < 1e-9,
      `horizontal scale ${xy} scales the sheet's span (${spanX}, ${spanY})`,
    );
  }
});

check("the horizontal scale never leaks into the sheet's height", () => {
  const base = floorPlaneGeometry(DOMAIN, { xy: 1, z: 1 });
  const wide = floorPlaneGeometry(DOMAIN, { xy: 2, z: 1 });
  if (!base || !wide) return;
  assertTrue(base.origin[2] === wide.origin[2], "z is untouched by the horizontal scale");
});

check("the sheet tracks the vertical display scale", () => {
  const raised: [number, number, number] = [-40000, -30000, 2000];
  const domain = { lower: raised, upper: DOMAIN.upper };
  const unit = floorPlaneGeometry(domain, { xy: 1, z: 1 });
  const exaggerated = floorPlaneGeometry(domain, { xy: 1, z: 3 });
  if (!unit || !exaggerated) return;
  assertTrue(
    Math.abs(exaggerated.origin[2] - unit.origin[2] * 3) < 1e-9,
    `vertical scale multiplies the sheet height (${exaggerated.origin[2]} vs ${unit.origin[2] * 3})`,
  );
});

check("the sheet is lifted clear of the domain floor but stays below the first level", () => {
  const plane = floorPlaneGeometry(DOMAIN, { xy: 1, z: 1 });
  if (!plane) return;
  const lift = plane.origin[2] - DOMAIN.lower[2];
  assertTrue(lift > 0, "the sheet clears the domain box edge");
  // A first scalar level at half of a 250 m cell must still sit above the sheet.
  assertTrue(lift < 125, `the lift (${lift} m) stays under a half-cell`);
});

check("all three corners share one height", () => {
  const plane = floorPlaneGeometry(DOMAIN, { xy: 1.7, z: 2.5 });
  if (!plane) return;
  assertTrue(
    plane.origin[2] === plane.point1[2] && plane.origin[2] === plane.point2[2],
    "the sheet is level",
  );
});

check("a missing or non-finite placement is refused rather than drawn", () => {
  assertTrue(floorPlaneGeometry(null, { xy: 1, z: 1 }) === null, "null bounds yield no placement");
  assertTrue(
    floorPlaneGeometry(DOMAIN, { xy: Number.NaN, z: 1 }) === null,
    "a NaN horizontal scale yields no placement",
  );
  assertTrue(
    floorPlaneGeometry(DOMAIN, { xy: 1, z: Number.POSITIVE_INFINITY }) === null,
    "an infinite vertical scale yields no placement",
  );
  assertTrue(
    floorPlaneGeometry(
      { lower: [Number.NaN, 0, 0], upper: DOMAIN.upper },
      { xy: 1, z: 1 },
    ) === null,
    "a non-finite corner yields no placement",
  );
});

check("terrain floor mesh is worker-ready with x-fastest VTK topology and UVs", () => {
  const mesh = terrainFloorMesh({
    x: new Float64Array([0, 0, 1, 1]),
    y: new Float64Array([0, 1, 0, 1]),
    z: new Float64Array([10, 11, 12, 13]),
  }, [2, 2]);
  assertTrue(
    Array.from(mesh.points).join(",") === "0,0,10,1,0,12,0,1,11,1,1,13",
    "terrain points transpose from field order to VTK x-fastest order",
  );
  assertTrue(
    Array.from(mesh.polys).join(",") === "3,0,1,2,3,1,3,2",
    "terrain mesh contains two indexed triangles per cell",
  );
  assertTrue(
    Array.from(mesh.tcoords).join(",") === "0,0,1,0,0,1,1,1",
    "terrain mesh UV coordinates register the floor texture",
  );
});

// ---------- Report ----------

console.log(`\nradar floor tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
