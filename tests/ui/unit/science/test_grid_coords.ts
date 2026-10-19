/*
 * Regression tests for `src/lib/gridCoords.ts` axisCoords().
 *
 * Run via:
 *   cd ui && node --experimental-strip-types \
 *     --import ../tests/ui/register-ts-resolve.mjs scripts/test_grid_coords.ts
 *
 * Bug under test: axisCoords fired the cylindrical theta branch whenever the
 * dtheta slot was truthy, with no Cartesian guard. On a Cartesian run that slot
 * holds the y-spacing in METRES (e.g. 600), so the y-axis was labelled "θ (°)"
 * with a ~34000 deg/cell coordinate (600 * 180/π). The fix routes through
 * isCartesianGrid and labels x/y/z (km) for Cartesian grids.
 */

import {
  axesAreCoLocated,
  axisCoords,
  axisLabel,
  axisOptions,
  heightKmToZIndex,
  LOCKED_ASPECT_AXIS,
  lockedAspectPartnerAxis,
  planeSupportsLockedAspect,
  scalarCoordinateKm,
  type AxisCoords,
} from "../../../../ui/src/lib/science/gridCoords.ts";
import type { SliceAxis } from "../../../../ui/src/lib/platform/npy.ts";
import { fitAxisFragments } from "../../../../ui/src/lib/plot/plotFit.ts";

// ---------- Tiny assertion API (matches test_cartesian_interp.ts) ----------

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

const ax = (n: number): SliceAxis => n as SliceAxis;

// ---------- Cartesian: x / y / z in km (the bug fix) ----------

check("cartesian: x/y/z (km), never a degrees axis", () => {
  const grid = { coordinateSystem: "cartesian", drM: 600, dyM: 5, dthetaRad: 5, dzM: 250 };
  const x = axisCoords(ax(0), 4, grid);
  assertTrue(x.title === "x (km)", "axis0 title 'x (km)' (not 'r')");
  assertTrue(close(x.spacing, 0.6), "axis0 spacing 0.6 km");

  const y = axisCoords(ax(1), 4, grid);
  assertTrue(y.title === "y (km)", "axis1 title 'y (km)' (not 'θ')");
  assertTrue(close(y.spacing, 0.005), "axis1 uses explicit 5 m dy even below 2pi");
  assertTrue(
    !!y.values && close(y.values[0], -0.0075) && close(y.values[3], 0.0075),
    "legacy axis1 is centered consistently with the VTK box",
  );

  const z = axisCoords(ax(2), 4, grid);
  assertTrue(z.title === "z (km)", "axis2 title 'z (km)'");
  assertTrue(close(z.spacing, 0.25), "axis2 spacing 0.25 km");
});

check("cartesian axes consume explicit scalar-origin metadata", () => {
  const grid = {
    coordinateSystem: "cartesian",
    drM: 600,
    dyM: 400,
    dzM: 250,
    scalarOriginM: [900, 700, 375] as [number, number, number],
  };
  const x = axisCoords(ax(0), 3, grid);
  const y = axisCoords(ax(1), 3, grid);
  const z = axisCoords(ax(2), 3, grid);
  assertTrue(!!x.values && close(x.values[0], 0.9) && close(x.values[2], 2.1), "x starts at metadata origin");
  assertTrue(!!y.values && close(y.values[0], 0.7) && close(y.values[2], 1.5), "y starts at metadata origin");
  assertTrue(!!z.values && close(z.values[0], 0.375) && close(z.values[2], 0.875), "z starts at metadata origin");
});

check("physical bounds supply the scalar origin when the explicit field is absent", () => {
  const grid = {
    coordinateSystem: "cartesian",
    drM: 600,
    dyM: 400,
    dzM: 250,
    physicalBoundsM: {
      lower: [100, 200, 300] as [number, number, number],
      upper: [2500, 1800, 1300] as [number, number, number],
    },
  };
  const x = axisCoords(ax(0), 2, grid);
  const z = axisCoords(ax(2), 2, grid);
  assertTrue(!!x.values && close(x.values[0], 0.4), "x origin is lower face plus half dx");
  assertTrue(!!z.values && close(z.values[0], 0.425), "z origin is lower face plus half dz");
});

check("height selector and coordinate display round-trip through manifest origin", () => {
  const grid = {
    coordinateSystem: "cartesian",
    drM: 600,
    dyM: 400,
    dzM: 250,
    scalarOriginM: [300, 200, 375] as [number, number, number],
  };
  for (let k = 0; k < 4; k++) {
    const heightKm = scalarCoordinateKm(ax(2), k, 4, grid);
    assertTrue(heightKm != null, `z index ${k} has a physical coordinate`);
    assertTrue(heightKmToZIndex(heightKm ?? 0, grid, 4) === k, `z index ${k} round-trips`);
  }
  assertTrue(heightKmToZIndex(-10, grid, 4) === 0, "height below domain clamps low");
  assertTrue(heightKmToZIndex(10, grid, 4) === 3, "height above domain clamps high");
});

check("stretched scalar coordinates remain exact across axes and height selection", () => {
  const grid = {
    coordinateSystem: "cartesian",
    drM: 1_000,
    dyM: 1_000,
    scalarCoordinatesM: [
      [-500, 500],
      [-500, 500],
      [100, 400, 1_000],
    ] as [number[], number[], number[]],
  };
  const z = axisCoords(ax(2), 3, grid);
  assertTrue(z.values?.join(",") === "0.1,0.4,1", "z axis exposes native stretched centers in km");
  assertTrue(close(z.spacing, 0.45), "stretched axis reports mean spacing only for display sizing");
  assertTrue(close(scalarCoordinateKm(ax(2), 1, 3, grid) ?? -1, 0.4), "level readout uses the exact center");
  assertTrue(heightKmToZIndex(0.72, grid, 3) === 2, "height selection snaps to the nearest stretched level");
  assertTrue(heightKmToZIndex(0.7, grid, 3) === 1, "nearest-level ties retain the lower source level");
});

/* The level readout sits next to the axis axisCoords() draws. When the exact
 * coordinate array belongs to a different lattice than the plotted field --
 * a halo crop, or a manifest whose coordinate variable is staggered against
 * its data variable -- axisCoords() rejects it on length and falls back to
 * spacing. The readout must reject it on the same test, or the two disagree
 * on screen. */
check("level readout rejects a coordinate array that does not match the plotted axis", () => {
  const grid = {
    coordinateSystem: "cartesian",
    drM: 1_000,
    dyM: 1_000,
    dzM: 500,
    scalarOriginM: [500, 500, 250] as [number, number, number],
    // Sized for a 6-cell axis; the plotted field has 3 cells.
    scalarCoordinatesM: [
      [500, 1_500, 2_500],
      [500, 1_500, 2_500],
      [250, 750, 1_250, 1_750, 2_250, 2_750],
    ] as [number[], number[], number[]],
  };
  const z = axisCoords(ax(2), 3, grid);
  assertTrue(z.values?.join(",") === "0.25,0.75,1.25", "mismatched z coordinates fall back to spacing");
  for (let k = 0; k < 3; k++) {
    assertTrue(
      close(scalarCoordinateKm(ax(2), k, 3, grid) ?? -1, z.values![k]),
      `level readout matches the drawn axis at index ${k}`,
    );
  }
  assertTrue(
    scalarCoordinateKm(ax(2), 3, 3, grid) === undefined,
    "an index past the plotted axis has no coordinate",
  );
  // The matching axis still uses its exact centres.
  assertTrue(
    close(scalarCoordinateKm(ax(0), 2, 3, grid) ?? -1, 2.5),
    "an axis whose coordinates do match keeps the exact centre",
  );
});

check("terrain vertical coordinates remain explicit model levels", () => {
  const grid = {
    coordinateSystem: "cartesian",
    terrainFollowing: true,
    drM: 1_000,
    dyM: 1_000,
    dzM: 250,
    scalarOriginM: [500, 500, 125] as [number, number, number],
  };
  const z = axisCoords(ax(2), 4, grid);
  assertTrue(z.title === "model level k", "terrain vertical axis does not claim physical z");
  assertTrue(z.values?.join(",") === "0,1,2,3", "terrain vertical axis exposes model indices");
  assertTrue(axisLabel(ax(2), grid) === "k", "terrain vertical label is k");
  assertTrue(axisOptions(grid)[2].label === "k", "terrain slice option is k");
  assertTrue(
    scalarCoordinateKm(ax(2), 2, 4, grid) === undefined,
    "terrain level has no fabricated scalar z coordinate",
  );
  assertTrue(
    heightKmToZIndex(1, grid, 4) === 0,
    "physical-height selection is unavailable without the 3-D lattice",
  );
});

// ---------- Cartesian with missing dtheta: square cells (dy = dx) ----------

check("cartesian (no dtheta slot): square cells, still x/y/z", () => {
  const grid = { drM: 600, dzM: 250 };
  const y = axisCoords(ax(1), 4, grid);
  assertTrue(y.title === "y (km)", "axis1 still 'y (km)' with no dtheta metadata");
  assertTrue(close(y.spacing, 0.6), "axis1 reuses dx (square cells)");
});

// ---------- Degenerate: null grid / empty axis fall back to a bare index axis ----

check("degenerate inputs fall back to the index axis", () => {
  const none = axisCoords(ax(1), 4, null);
  assertTrue(none.values === undefined && none.spacing === 1, "null grid -> index fallback");
  const empty = axisCoords(ax(0), 0, { drM: 600, dthetaRad: 600, dzM: 250 });
  assertTrue(empty.values === undefined, "cellCount 0 -> index fallback");
});

// ---------- axisLabel / axisOptions: Cartesian x/y/z on every grid ----
// These back the hover prefixes, index labels, slice-title suffixes and the
// slice-axis dropdown. Nimbus is Cartesian-only, so they are always x/y/z
// regardless of the grid metadata (the `grid` arg is retained for symmetry).

check("axisLabel: x/y/z for any grid", () => {
  const grid = { drM: 100, dthetaRad: 200, dzM: 100 };
  assertTrue(axisLabel(ax(0), grid) === "x", "axis0 -> x");
  assertTrue(axisLabel(ax(1), grid) === "y", "axis1 -> y");
  assertTrue(axisLabel(ax(2), grid) === "z", "axis2 -> z");
});

check("axisLabel: null grid -> x/y/z", () => {
  assertTrue(axisLabel(ax(0), null) === "x", "null grid axis0 -> x");
  assertTrue(axisLabel(ax(1), null) === "y", "null grid axis1 -> y");
});

check("axisOptions: x/y/z for any grid (incl. null)", () => {
  const cart = axisOptions({ drM: 100, dthetaRad: 200, dzM: 100 });
  assertTrue(cart.map((o) => o.label).join("") === "xyz", "labels x/y/z");
  assertTrue(cart[1].value === 1, "option value preserved");
  assertTrue(axisOptions(null).map((o) => o.label).join("") === "xyz", "null grid -> x/y/z");
});

// ---------- axesAreCoLocated / planeSupportsLockedAspect ----------
//
// REGRESSION: the comparison view differenced A and B cell-by-cell after
// checking only the plane identity and the array SHAPES. A resolution
// experiment — two runs on the same cell counts with different spacing — passed
// that check while sampling entirely different physical points, and the result
// was labelled "A−B difference" and quantified to four decimals.

const km = (values: number[]): AxisCoords =>
  ({ values, title: "x (km)", hoverFmt: ":.3f", spacing: 1, unit: "km" });
const indexAxis: AxisCoords = { values: undefined, title: "x", hoverFmt: "", spacing: 1, unit: "index" };
/** Terrain-following vertical: populated `values`, but they are model-level
 *  INDICES. The trap that motivated the `unit` discriminant. */
const levelAxis: AxisCoords = {
  values: [0, 1, 2], title: "model level k", hoverFmt: ":.0f", spacing: 1, unit: "level",
};

check("axesAreCoLocated accepts identical physical sample positions", () => {
  assertTrue(axesAreCoLocated(km([0, 1, 2, 3]), km([0, 1, 2, 3])), "identical coordinates");
  // Reconstructed by different paths (exact array vs origin + spacing), so the
  // comparison has to be tolerant of float noise rather than exact.
  assertTrue(
    axesAreCoLocated(km([0, 1, 2, 3]), km([0, 1 + 1e-12, 2, 3 - 1e-12])),
    "float noise within tolerance",
  );
});

check("axesAreCoLocated rejects same-shape axes at different spacing", () => {
  assertTrue(
    !axesAreCoLocated(km([0, 1, 2, 3]), km([0, 2, 4, 6])),
    "matching cell count with double the spacing is NOT co-located",
  );
  assertTrue(
    !axesAreCoLocated(km([0, 1, 2, 3]), km([10, 11, 12, 13])),
    "same spacing at a different origin is NOT co-located",
  );
  assertTrue(
    !axesAreCoLocated(km([0, 1, 2, 3]), km([0, 1, 2])),
    "different cell counts are not co-located",
  );
  assertTrue(!axesAreCoLocated(km([]), km([])), "empty axes are not co-located");
});

check("axesAreCoLocated tolerance is a fraction of a cell, not of the offset", () => {
  // A domain far from the origin with fine cells. Scaling the tolerance on the
  // coordinate MAGNITUDE would admit a tolerance wider than the entire axis,
  // so grids sampling positions half a kilometre apart would pass.
  const fine = km([1_000_000, 1_000_000.001, 1_000_000.002]);
  const coarse = km([1_000_000, 1_000_000.5, 1_000_000.9]);
  assertTrue(
    !axesAreCoLocated(fine, coarse),
    "a far-from-origin axis is not co-located with one 0.5 km off per cell",
  );
  // ...while genuinely equal far-from-origin axes still compare equal despite
  // float64 representation noise at that magnitude.
  assertTrue(
    axesAreCoLocated(fine, km([1_000_000, 1_000_000.001 + 1e-9, 1_000_000.002])),
    "float64 noise at a large offset is still within tolerance",
  );
});

check("axesAreCoLocated treats index fallbacks as a matched pair, never as mixed", () => {
  assertTrue(
    axesAreCoLocated(indexAxis, indexAxis),
    "two index axes compare index-wise, which is what both plots draw",
  );
  assertTrue(
    !axesAreCoLocated(km([0, 1, 2]), indexAxis),
    "one physical axis against one index axis is a genuine mismatch",
  );
});

/* The `unit` discriminant exists because a model-level axis has POPULATED
 * values that happen to be integers. Testing `values != null` would compare a
 * height in kilometres against a level index and call them co-located. */
check("axesAreCoLocated never compares kilometres against model levels", () => {
  assertTrue(
    !axesAreCoLocated(km([0, 1, 2]), levelAxis),
    "km vs model levels is a mismatch even though BOTH carry values 0,1,2",
  );
  assertTrue(
    !axesAreCoLocated(levelAxis, indexAxis),
    "model levels vs bare indices is a mismatch",
  );
  assertTrue(axesAreCoLocated(levelAxis, levelAxis), "two level axes compare level-wise");
});

check("axisCoords stamps the unit that matches what it returned", () => {
  const cartesian = { coordinateSystem: "cartesian", drM: 600, dyM: 400, dzM: 250 };
  assertTrue(axisCoords(ax(0), 4, cartesian).unit === "km", "uniform-spacing axis is km");
  assertTrue(axisCoords(ax(2), 4, cartesian).unit === "km", "vertical with dz is km");
  assertTrue(axisCoords(ax(0), 4, null).unit === "index", "no grid -> index");
  assertTrue(
    axisCoords(ax(0), 4, { coordinateSystem: "cartesian" }).unit === "index",
    "no declared spacing -> index",
  );
  assertTrue(
    axisCoords(ax(2), 4, { ...cartesian, terrainFollowing: true }).unit === "level",
    "terrain-following vertical is level, NOT km, despite populated values",
  );
  const stretched = {
    coordinateSystem: "cartesian",
    drM: 1_000,
    scalarCoordinatesM: [[100, 400, 1_000], [0, 1, 2], [0, 1, 2]] as [number[], number[], number[]],
  };
  assertTrue(axisCoords(ax(0), 3, stretched).unit === "km", "exact stretched centres are km");
});

check("planeSupportsLockedAspect locks plan views and frees vertical sections", () => {
  const physical = km([0, 1, 2]);
  assertTrue(
    planeSupportsLockedAspect(ax(0), ax(1), physical, physical),
    "x-y plan view locks 1:1",
  );
  assertTrue(
    !planeSupportsLockedAspect(ax(2), ax(0), physical, physical),
    "x-z section stays free so the vertical can be exaggerated",
  );
  assertTrue(
    !planeSupportsLockedAspect(ax(0), ax(2), physical, physical),
    "z on either axis frees the aspect",
  );
  assertTrue(
    !planeSupportsLockedAspect(ax(0), ax(1), physical, indexAxis),
    "an index axis is not commensurable with kilometres",
  );
  assertTrue(
    !planeSupportsLockedAspect(ax(0), ax(1), physical, levelAxis),
    "a model-level axis is not commensurable with kilometres either",
  );
});

// ---------- Locked-aspect axis fragments ----------

/*
 * A square domain drawn in a wide pane used to letterbox: `scaleanchor` was set
 * without `constrain`, so Plotly satisfied the 1:1 ratio by widening the axis
 * RANGE to fill the pane instead of shrinking the plot AREA. On a 512x512 @
 * 100 m grid the axis ran to +/-100 km over +/-25 km of data and most of the
 * pane was empty background. Both halves of the idiom are required, so both are
 * asserted here.
 */

check("locked-aspect anchor axis constrains its domain", () => {
  assertTrue(
    LOCKED_ASPECT_AXIS.constrain === "domain",
    `anchor axis constrain is "domain" (got "${LOCKED_ASPECT_AXIS.constrain}")`,
  );
});

check("locked-aspect partner axis anchors 1:1 AND constrains its domain", () => {
  const partner = lockedAspectPartnerAxis();
  assertTrue(partner.scaleanchor === "x", `default anchor is "x" (got "${partner.scaleanchor}")`);
  assertTrue(partner.scaleratio === 1, `ratio is 1 (got ${partner.scaleratio})`);
  // The regression: scaleratio alone leaves Plotly free to widen the range.
  assertTrue(
    partner.constrain === "domain",
    `partner axis constrain is "domain" (got "${partner.constrain}")`,
  );
});

check("locked-aspect partner axis honours a subplot anchor", () => {
  // A 3-panel comparison figure anchors each subplot to its own x axis; using
  // "x" everywhere would tie all three panels to the first one's scale.
  for (const anchor of ["x2", "x3", "y"] as const) {
    const partner = lockedAspectPartnerAxis(anchor);
    assertTrue(partner.scaleanchor === anchor, `anchor "${anchor}" preserved`);
    assertTrue(partner.constrain === "domain", `anchor "${anchor}" still constrains domain`);
  }
});

// ---------- Fit mode ----------

/*
 * `fitAxisFragments` is the one place that decides whether a plane is drawn at
 * true shape. It has to answer correctly for three inputs, not two: an
 * unlockable plane (a vertical section) must produce no constraint at all
 * regardless of what the Fit control says, or a cross-section would be forced
 * square and become unreadable.
 */

check("fit: true-shape on a lockable plane constrains both axes", () => {
  const fit = fitAxisFragments(true, "true-shape");
  assertTrue("constrain" in fit.x, "x axis carries a constraint");
  assertTrue("scaleanchor" in fit.y, "y axis anchors to x");
  assertTrue("constrain" in fit.y, "y axis carries a constraint");
});

check("fit: fill releases the lock on a lockable plane", () => {
  const fit = fitAxisFragments(true, "fill");
  assertTrue(Object.keys(fit.x).length === 0, "x axis unconstrained under Fill");
  assertTrue(Object.keys(fit.y).length === 0, "y axis unconstrained under Fill");
});

check("fit: an unlockable plane is never constrained, whatever Fit says", () => {
  // A vertical section is conventionally exaggerated; forcing 1:1 would squash
  // a storm-scale x-z plane into an unreadable strip.
  for (const mode of ["true-shape", "fill"] as const) {
    const fit = fitAxisFragments(false, mode);
    assertTrue(Object.keys(fit.x).length === 0, `x free on unlockable plane (${mode})`);
    assertTrue(Object.keys(fit.y).length === 0, `y free on unlockable plane (${mode})`);
  }
});

// ---------- Report ----------

console.log(`\ngridCoords tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
