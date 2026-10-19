/*
 * Tests for src/lib/vtk/vtkResolution.ts — the 3D backing-store pixel budget.
 *
 * The numbers below are anchored to a measured case, not invented: on a
 * 1440x1000 window at devicePixelRatio 2 the rendered-supercell canvas is
 * 1438x510 CSS framed and 1438x941 CSS immersive. Uncapped, vtk.js sized those
 * at 2.93 MP and 5.41 MP, and the median frame cost of a camera drag went from
 * 596 ms to 1880 ms. Those two CSS sizes appear throughout as the reference
 * geometry, so a regression in the policy shows up as a number a reader can
 * check against the report rather than as an abstract ratio.
 *
 * Two properties matter more than any particular budget and are pinned
 * hardest: the scale is UNIFORM (an axis-dependent scale would stretch the
 * scene and quietly falsify the domain box and axis rulers), and it never
 * exceeds the display's own pixel ratio (rendering more device pixels than the
 * display can show is pure cost for no visible return).
 *
 * Run via:
 *   cd ui && npm run test:vtk-resolution
 */

import {
  backingStoreSize,
  MIN_PIXEL_RATIO,
  SURFACE_RESOLUTION_POLICY,
  VOLUME_RESOLUTION_POLICY,
  type ResolutionPolicy,
} from "../../../../ui/src/lib/vtk/vtkResolution.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}
function equal(actual: unknown, expected: unknown, label: string): void {
  check(
    JSON.stringify(actual) === JSON.stringify(expected),
    `${label} (expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)})`,
  );
}

/* The measured immersive canvas, in CSS pixels, on a Retina display. */
const IMMERSIVE: [number, number] = [1438, 941];
const FRAMED: [number, number] = [1438, 510];
const RETINA = 2;

const megapixels = (size: { width: number; height: number }) => size.width * size.height / 1e6;

/* ---------- The measured case is actually bounded ---------- */
{
  const rest = backingStoreSize(...IMMERSIVE, RETINA, VOLUME_RESOLUTION_POLICY)!;

  check(
    megapixels(rest) <= VOLUME_RESOLUTION_POLICY.maxPixels / 1e6,
    `an immersive volume scene fits its budget (was ${megapixels(rest).toFixed(2)} MP)`,
  );
  check(megapixels(rest) < 5.41, "the Retina backing store remains bounded below the uncapped workload");

  // Still supersampled: the point is to bound cost, not to abandon the extra
  // sharpness a Retina display can actually show.
  check(rest.pixelRatio > 1, "a volume scene still renders above CSS resolution");
}

/*
 * The backing store depends on the container and the display, and on NOTHING
 * about what the user is doing.
 *
 * There used to be a second, smaller size that applied for the duration of a
 * camera gesture. vtk.js reports pointer positions in backing-store pixels and
 * rotates against a previous position it keeps private, so resizing the buffer
 * mid-drag rescaled one of the two and threw the camera 150 degrees on the next
 * move. Nothing may reintroduce a size that changes while a gesture is running.
 */
{
  const stable = backingStoreSize(...IMMERSIVE, RETINA, VOLUME_RESOLUTION_POLICY)!;
  for (let repeat = 0; repeat < 3; repeat++) {
    const again = backingStoreSize(...IMMERSIVE, RETINA, VOLUME_RESOLUTION_POLICY)!;
    equal(
      [again.width, again.height],
      [stable.width, stable.height],
      "the backing store is a pure function of container and display",
    );
  }
  check(
    !Object.keys(VOLUME_RESOLUTION_POLICY).some((key) => /interactive/i.test(key)),
    "no policy carries a separate mid-gesture regime",
  );
}

/* ---------- Uniform scale: the scene is never stretched ---------- */
{
  const cases: Array<[number, number]> = [IMMERSIVE, FRAMED, [1000, 1000], [1920, 400]];
  for (const [w, h] of cases) {
    const size = backingStoreSize(w, h, RETINA, VOLUME_RESOLUTION_POLICY)!;
    const sourceAspect = w / h;
    const bufferAspect = size.width / size.height;
    // One device pixel of floor() slack per axis is all that is allowed.
    check(
      Math.abs(sourceAspect - bufferAspect) < 0.01 * sourceAspect,
      `a ${w}x${h} container keeps its aspect ratio`,
    );
  }
}

/* ---------- Never render more device pixels than the display has ---------- */
{
  const policies: Array<[string, ResolutionPolicy]> = [
    ["volume", VOLUME_RESOLUTION_POLICY],
    ["surface", SURFACE_RESOLUTION_POLICY],
  ];
  for (const [name, policy] of policies) {
    for (const dpr of [1, 1.5, 2, 3]) {
      const size = backingStoreSize(800, 600, dpr, policy)!;
      check(
        size.pixelRatio <= dpr + 1e-9,
        `${name} never supersamples past the display at dpr ${dpr}`,
      );
    }
  }
  // A standard-density display is left exactly alone by the volume policy:
  // the budget is generous enough that 1x is never reduced at this size.
  const standard = backingStoreSize(...IMMERSIVE, 1, VOLUME_RESOLUTION_POLICY)!;
  equal(
    [standard.width, standard.height],
    IMMERSIVE,
    "a non-Retina immersive canvas is rendered pixel-for-pixel",
  );
}

/* ---------- Surface scenes keep full resolution ---------- */
{
  const rest = backingStoreSize(...IMMERSIVE, RETINA, SURFACE_RESOLUTION_POLICY)!;
  equal(
    [rest.width, rest.height],
    [IMMERSIVE[0] * 2, IMMERSIVE[1] * 2],
    "a rasterized scene keeps the full Retina backing store — its cost is geometry, not pixels",
  );
}

/* ---------- Small canvases are never touched ---------- */
{
  const small = backingStoreSize(400, 300, RETINA, VOLUME_RESOLUTION_POLICY)!;
  equal([small.width, small.height], [800, 600], "a canvas under budget keeps its full ratio");
}

/* ---------- The legibility floor beats the budget ---------- */
{
  // An 8K canvas is 33 MP; the budget alone would ask for a 0.28 ratio,
  // at which the domain box and tick marks are thinner than a device pixel and
  // drop out of the image entirely rather than softening.
  const huge = backingStoreSize(7680, 4320, RETINA, VOLUME_RESOLUTION_POLICY)!;
  check(
    huge.pixelRatio === MIN_PIXEL_RATIO,
    "an extreme canvas stops at the legibility floor rather than becoming a mosaic",
  );
  check(
    huge.width * huge.height > VOLUME_RESOLUTION_POLICY.maxPixels,
    "the floor is allowed to overrun the pixel budget — legibility outranks it",
  );
}

/* ---------- Degenerate input yields no size, not a broken one ---------- */
{
  const policy = VOLUME_RESOLUTION_POLICY;
  check(backingStoreSize(0, 500, RETINA, policy) === null, "a zero width has no backing store");
  check(backingStoreSize(500, 0, RETINA, policy) === null, "a zero height has no backing store");
  check(backingStoreSize(-10, 500, RETINA, policy) === null, "a negative width has no backing store");
  check(
    backingStoreSize(Number.NaN, 500, RETINA, policy) === null,
    "a non-finite width has no backing store",
  );
  check(
    backingStoreSize(500, Number.POSITIVE_INFINITY, RETINA, policy) === null,
    "an infinite height has no backing store",
  );

  // A missing or nonsensical devicePixelRatio must not poison the geometry:
  // a NaN ratio would reach setSize as a NaN canvas dimension.
  for (const dpr of [Number.NaN, 0, -2, Number.POSITIVE_INFINITY]) {
    const size = backingStoreSize(800, 600, dpr, policy)!;
    check(
      Number.isInteger(size.width) && Number.isInteger(size.height) && size.width > 0,
      `a devicePixelRatio of ${dpr} falls back to a usable 1x buffer`,
    );
  }
  equal(
    backingStoreSize(800, 600, Number.NaN, policy)!.pixelRatio,
    1,
    "an unreadable display ratio is treated as a standard-density display",
  );
}

/* ---------- Whole device pixels ---------- */
{
  // 1439 x 1.3862... is not an integer; a fractional canvas dimension is
  // truncated by the DOM anyway, so it must be decided here rather than there.
  const odd = backingStoreSize(1439, 941, RETINA, VOLUME_RESOLUTION_POLICY)!;
  check(
    Number.isInteger(odd.width) && Number.isInteger(odd.height),
    "a backing store is always a whole number of device pixels",
  );
  const tiny = backingStoreSize(1, 1, 0.1, VOLUME_RESOLUTION_POLICY)!;
  check(tiny.width >= 1 && tiny.height >= 1, "a backing store never rounds down to nothing");
}


const qhd = backingStoreSize(2560,1440,1,VOLUME_RESOLUTION_POLICY)!;
check(qhd.width === 2560 && qhd.height === 1440, "QHD volumes keep one rendered pixel per display pixel");

console.log(`\nvtkResolution tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
