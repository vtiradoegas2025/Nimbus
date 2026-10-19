/* Deterministic scientific plotting contract tests. */

import { readFileSync } from "node:fs";

import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";
import {
  cropAxisReduction,
  cropToValidPlotDomain,
  validAxisLength,
  type PlotGrid,
} from "../../../../ui/src/lib/science/validDomain.ts";
import {
  autoVerticalExag,
  VERTICAL_EXAG_DEFAULT,
  VERTICAL_EXAG_RANGE,
} from "../../../../ui/src/lib/plot/sceneScale.ts";
import { PLOT_TYPE_META } from "../../../../ui/src/lib/plot/plotRegistry.ts";
import {
  coarsestVtkSampling,
  describeVtkSampling,
  describeLayerVtkSampling,
  samplingReductionFactor,
  presentationDetailWeight,
  renderOperationAllowed,
  stableRenderSeed,
  vtkSamplingProvenance,
} from "../../../../ui/src/lib/plot/renderFidelity.ts";
import {
  defaultsToSignedPair,
  defaultsToRotationDominatedSurface,
  defaultsToPositiveQSurface,
  initialPositiveQThreshold,
  initialRotationDominatedThreshold,
  rotationDominatedThresholdRange,
  positiveQThresholdRange,
  signedThresholdRange,
  supportsSignedPair,
} from "../../../../ui/src/lib/plot/coherentStructures.ts";
import {
  DBZ_RANGE,
  DBZ_COLORSCALE,
  DBZ_PHYSICAL_STOPS,
  DBZ_VTK_STOPS,
  DBZ_VTK_STOPS_VOLUME,
  cssColorbarGradient,
  usesFixedAnchorRange,
} from "../../../../ui/src/lib/plot/plotColormaps.ts";
import { axisViewBounds } from "../../../../ui/src/lib/plot/viewSync.ts";
import {
  buildScaledGeometry,
  scalePointTriples,
} from "../../../../ui/src/lib/vtk/vtkScaledPolyData.ts";
import { displayDistanceForPhysicalRay } from "../../../../ui/src/lib/vtk/volumeOpticalDepth.ts";
import { defaultStreamlineTubeRadiusM } from "../../../../ui/src/lib/plot/streamlinePresentation.ts";
import {
  defaultVolumeColormap,
  resolveVolumeTransferPolicy,
  opticalDepthOpacity,
  VOLUME_OPACITY_REFERENCE_M,
  volumeFieldWindowAggregation,
  volumeOpacityClass,
  volumeOpacityReferenceM,
  volumeOpacityPoints,
  volumeUsesOpacityNormals,
  volumeUsesStructuralLighting,
} from "../../../../ui/src/lib/plot/volumeTransferPolicy.ts";
import {
  availableScalarPlanes,
  describeScalarPlane,
  resolveScalarPlanePolicy,
  scalarPlaneReduction,
} from "../../../../ui/src/lib/plot/scalarPlanePolicy.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function equal(actual: unknown, expected: unknown, label: string): void {
  if (actual === expected) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function close(actual: number, expected: number, label: string): void {
  equal(Math.abs(actual - expected) < 1e-5, true, label);
}

function deepEqual(actual: unknown[], expected: unknown[], label: string): void {
  equal(JSON.stringify(actual), JSON.stringify(expected), label);
}

const sourceData = new Float32Array(2 * 3 * 4);
for (let x = 0; x < 2; x++) {
  for (let y = 0; y < 3; y++) {
    for (let z = 0; z < 4; z++) {
      sourceData[(x * 3 + y) * 4 + z] = 100 * x + 10 * y + z;
    }
  }
}
const source: NpyN = { shape: [2, 3, 4], data: sourceData };

const valid = cropToValidPlotDomain(
  source,
  {
    nr: 2,
    nth: 3,
    nz: 4,
    drM: 100,
    dyM: 200,
    dzM: 50,
    scalarOriginM: [50, 100, 25],
    validIndexRange: { lower: [0, 1, 0], upper: [1, 1, 3] },
  },
);
equal(valid.field.shape.join("x"), "2x1x4", "Plotly domain crop removes declared y halos");
equal(valid.field.data[0], 10, "Plotly domain crop starts on first valid cell");
equal(valid.field.data.at(-1), 113, "Plotly domain crop ends on last valid cell");
equal(valid.grid?.scalarOriginM?.join(","), "50,300,25", "cropped Plotly origin shifts by one dy");

const stretchedValid = cropToValidPlotDomain(
  source,
  {
    nr: 2,
    nth: 3,
    nz: 4,
    drM: 100,
    dyM: 200,
    scalarCoordinatesM: [
      [50, 150],
      [0, 200, 500],
      [25, 75, 175, 350],
    ],
    validIndexRange: { lower: [0, 1, 0], upper: [1, 1, 3] },
  },
);
equal(stretchedValid.grid?.scalarCoordinatesM?.[1].join(","), "200", "valid-domain crops preserve exact retained horizontal centers");
equal(stretchedValid.grid?.scalarCoordinatesM?.[2].join(","), "25,75,175,350", "valid-domain crops preserve untouched stretched vertical centers");

/* validAxisLength: controls sized BEFORE a field loads must use the valid
 * extent, not the manifest's nr/nth/nz, or the top of a slider's range indexes
 * halo cells the cropped field does not have and silently redisplays the last
 * valid slice. */
const haloedGrid = {
  nr: 6, nth: 6, nz: 8, drM: 100, dyM: 100, dzM: 50,
  scalarOriginM: [50, 50, 25] as [number, number, number],
  validIndexRange: {
    lower: [1, 1, 2] as [number, number, number],
    upper: [4, 4, 5] as [number, number, number],
  },
};
equal(validAxisLength(2, haloedGrid), 4, "validAxisLength reports the halo-free vertical extent");
equal(validAxisLength(0, haloedGrid), 4, "validAxisLength reports the halo-free x extent");
equal(validAxisLength(2, null), null, "validAxisLength is null without a grid");
equal(validAxisLength(2, { nz: 8 }), null, "validAxisLength is null without validIndexRange");
equal(
  validAxisLength(2, { nz: 8, validIndexRange: { lower: [0, 0, 5], upper: [0, 0, 2] } as PlotGrid["validIndexRange"] }),
  null,
  "validAxisLength rejects an inverted range rather than returning a negative length",
);

/* cropAxisReduction: an axis reduction is computed over the RAW on-disk field,
 * so it must be cropped separately — otherwise halo cells plot as real levels
 * AND the axis is anchored at the uncropped origin. */
const reduction = [0, 1, 2, 3, 4, 5, 6, 7];
const croppedReduction = cropAxisReduction(2, reduction, haloedGrid);
equal(croppedReduction.values.join(","), "2,3,4,5", "cropAxisReduction drops the vertical halo levels");
equal(croppedReduction.grid?.nz, 4, "cropAxisReduction narrows the grid's vertical count");
equal(
  croppedReduction.grid?.scalarOriginM?.join(","),
  "50,50,125",
  "cropAxisReduction shifts the vertical origin to the first valid centre",
);
equal(
  croppedReduction.grid?.validIndexRange?.upper.join(","),
  "4,4,3",
  "cropAxisReduction clears the halo on the reduced axis only",
);
equal(croppedReduction.grid?.nr, 6, "cropAxisReduction leaves the collapsed axes' metadata alone");

const exactReduction = cropAxisReduction(2, reduction, {
  ...haloedGrid,
  scalarCoordinatesM: [[0, 1], [0, 1], [10, 20, 30, 45, 65, 90, 120, 155]],
});
equal(
  exactReduction.grid?.scalarCoordinatesM?.[2].join(","),
  "30,45,65,90",
  "cropAxisReduction keeps the exact stretched centres of the retained levels",
);

equal(
  cropAxisReduction(2, [0, 1, 2, 3], haloedGrid).values.length,
  4,
  "cropAxisReduction leaves an already-cropped reduction untouched",
);
equal(
  cropAxisReduction(2, reduction, { nz: 8 }).values.length,
  8,
  "cropAxisReduction falls back to the input when the grid declares no valid range",
);

/* One unusable axis drops exact coordinates for the whole grid. The cropped
 * grid must then carry NO coordinate array: keeping the manifest's uncropped
 * one would hand every downstream reader halo-indexed centres for a field
 * that no longer has halos. */
const partialCoordinateValid = cropToValidPlotDomain(
  source,
  {
    nr: 2,
    nth: 3,
    nz: 4,
    drM: 100,
    dyM: 200,
    dzM: 50,
    scalarOriginM: [50, 100, 25],
    scalarCoordinatesM: [
      [50, 150],
      [0, 200, 500],
      // Staggered against the data variable: 5 faces for 4 cells.
      [0, 50, 100, 150, 200],
    ],
    validIndexRange: { lower: [0, 1, 0], upper: [1, 1, 3] },
  },
);
equal(partialCoordinateValid.field.shape.join("x"), "2x1x4", "partial-coordinate crop still crops the field");
equal(
  partialCoordinateValid.grid?.scalarCoordinatesM,
  undefined,
  "an uncroppable axis drops the manifest's uncropped coordinates instead of leaking them",
);
equal(
  partialCoordinateValid.grid?.scalarOriginM?.join(","),
  "50,200,25",
  "the per-axis origin still anchors on the axes whose own centres are usable",
);

equal(
  renderOperationAllowed("scientific", "piecewise-linear-reconstruction"),
  true,
  "scientific mode permits convex display reconstruction",
);
equal(
  renderOperationAllowed("scientific", "surface-smoothing"),
  false,
  "scientific mode rejects geometry smoothing",
);
equal(
  renderOperationAllowed("scientific", "support-constrained-subgrid-detail"),
  false,
  "scientific mode rejects invented subgrid detail",
);
equal(
  renderOperationAllowed("presentation", "support-constrained-subgrid-detail"),
  true,
  "presentation mode permits explicitly disclosed subgrid detail",
);

const sampling = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [51, 41, 41], spacing: [200, 200, 100] },
  { shape: [101, 81, 81], spacing: [100, 100, 50] },
  [2, 2, 2],
);
equal(sampling.sampling, "bounded-subsampling", "sampling provenance distinguishes renderer LOD");
equal(sampling.reconstruction, "piecewise-trilinear", "sampling provenance distinguishes display reconstruction");
equal(describeLayerVtkSampling("Cloud", sampling),
  "Cloud: source 100 m · used samples 200×200×100 m · linear display reconstruction 100×100×50 m",
  "layer disclosure distinguishes original, used and reconstructed resolution");
equal(describeLayerVtkSampling("Cloud", { ...sampling, sourceSpacing: [NaN, 100, 100] }), null,
  "unknown source spacing is not replaced by an inferred number");
equal(
  describeVtkSampling(sampling).join(" · "),
  "bounded LOD 2×2×1 · linear 2×2×2 reconstruction",
  "sampling disclosure reports both independent operations",
);
/*
 * A block mean is not a subsample, and the difference decides whether a peak on screen can be
 * trusted: a subsample either keeps a cell's value or misses it, a mean always dilutes it. A
 * native windowed read aggregates with cell_mean, so reporting that as "bounded LOD" would
 * describe an operation that did not happen.
 */
const meanSampling = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [51, 41, 41], spacing: [200, 200, 100] },
  { shape: [51, 41, 41], spacing: [200, 200, 100] },
  [1, 1, 1],
  "cell-mean",
);
equal(meanSampling.sampling, "bounded-cell-mean", "an aggregated read is disclosed as a cell mean");
equal(
  describeVtkSampling(meanSampling).join(" · "),
  "bounded cell-mean 2×2×1",
  "the disclosure names aggregation rather than subsampling",
);
const dbzMeanSampling = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [51, 41, 41], spacing: [200, 200, 100] },
  undefined,
  [1, 1, 1],
  "linear-reflectivity-mean",
);
equal(
  dbzMeanSampling.sampling,
  "bounded-linear-reflectivity-mean",
  "reflectivity aggregation keeps its physical averaging domain in provenance",
);
equal(
  describeVtkSampling(dbzMeanSampling).join(" · "),
  "bounded linear-Z mean 2×2×1",
  "the canvas distinguishes linear-Z averaging from arithmetic dBZ means",
);
const extremaSampling = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [51, 41, 41], spacing: [200, 200, 100] },
  undefined,
  [1, 1, 1],
  "magnitude-extreme",
);
equal(extremaSampling.sampling, "bounded-magnitude-extreme", "signed extrema LOD is explicit provenance");
equal(
  describeVtkSampling(extremaSampling).join(" · "),
  "bounded |value| extreme 2×2×1",
  "surface provenance never describes an extrema envelope as a mean or subsample",
);
equal(
  vtkSamplingProvenance(
    { shape: [101, 81, 41], spacing: [100, 100, 100] },
    { shape: [51, 41, 41], spacing: [200, 200, 100] },
  ).sampling,
  "bounded-subsampling",
  "subsampling stays the default, so existing callers are unchanged",
);

/*
 * A composite scene prepares each layer under its own voxel budget, so one
 * canvas holds several lattices. The badge quoted the cloud SILHOUETTE, which
 * has a fixed cap, while the cloud BODY runs on the pane-scaled one — in a 2x2
 * grid that reported a finer resolution than the thing being read.
 */
const nativeLayer = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
);
const halfLayer = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [51, 41, 41], spacing: [200, 200, 100] },
);
equal(describeLayerVtkSampling("Detail", nativeLayer), "Detail: source 100 m · native samples 100 m",
  "source-native sampling remains explicit");
const quarterLayer = vtkSamplingProvenance(
  { shape: [101, 81, 41], spacing: [100, 100, 100] },
  { shape: [26, 21, 41], spacing: [400, 400, 100] },
);
equal(samplingReductionFactor(nativeLayer), 1, "a native lattice reduces by 1");
equal(samplingReductionFactor(null), 1, "an absent layer cannot lower the floor");
equal(
  samplingReductionFactor(quarterLayer) > samplingReductionFactor(halfLayer),
  true,
  "a coarser lattice ranks higher",
);
equal(
  coarsestVtkSampling([nativeLayer, halfLayer, quarterLayer]),
  quarterLayer,
  "the badge quotes the weakest layer on the canvas",
);
equal(
  coarsestVtkSampling([quarterLayer, nativeLayer]),
  quarterLayer,
  "order does not change which layer is weakest",
);
equal(
  coarsestVtkSampling([null, undefined, halfLayer]),
  halfLayer,
  "layers that are switched off are skipped",
);
equal(coarsestVtkSampling([]), null, "nothing drawn discloses nothing");
equal(coarsestVtkSampling([null, undefined]), null, "no usable provenance discloses nothing");
equal(
  describeVtkSampling(coarsestVtkSampling([nativeLayer, quarterLayer])).join(" · "),
  "bounded LOD 4×4×1",
  "the disclosed text is the coarse layer's, not the native one's",
);
equal(
  vtkSamplingProvenance(
    { shape: [51, 41, 41], spacing: [200, 200, 100] },
    { shape: [51, 41, 41], spacing: [200, 200, 100] },
    undefined,
    [1, 1, 1],
    "cell-mean",
  ).sampling,
  "native",
  "an unreduced lattice is native whatever reduction was nominated",
);

const stretchedSampling = vtkSamplingProvenance(
  {
    shape: [10, 10, 20],
    spacing: [100, 100, 250],
    coordinateResampling: "linear-to-uniform",
    coordinateResamplingAxes: ["z"],
  },
  { shape: [10, 10, 20], spacing: [100, 100, 250] },
);
equal(
  describeVtkSampling(stretchedSampling).join(" · "),
  "linear stretched-z resampling",
  "sampling disclosure names coordinate-linear stretched-z resampling",
);
const stretchedXyzSampling = vtkSamplingProvenance(
  {
    shape: [10, 12, 20],
    spacing: [100, 200, 250],
    coordinateResampling: "linear-to-uniform",
    coordinateResamplingAxes: ["x", "y", "z"],
  },
  { shape: [10, 12, 20], spacing: [100, 200, 250] },
);
equal(describeLayerVtkSampling("Terrain", stretchedXyzSampling),
  "Terrain: nominal source 100×200×250 m · used samples 100×200×250 m",
  "nonuniform geometry is not mislabeled as a constant native physical spacing");
equal(
  describeVtkSampling(stretchedXyzSampling).join(" · "),
  "linear stretched-x/y/z resampling",
  "sampling disclosure names every regularized rectilinear axis",
);
close(presentationDetailWeight([100, 100, 100]), 0, "fine resolved grids receive no presentation detail");
close(presentationDetailWeight([500, 250, 100]), 1, "coarse grids reach the presentation-detail ceiling");
const midDetail = presentationDetailWeight([300, 100, 100]);
equal(midDetail > 0 && midDetail < 1, true, "presentation detail fades smoothly across intermediate spacing");
equal(stableRenderSeed("/run/a"), stableRenderSeed("/run/a"), "render seed is stable within one run");
equal(stableRenderSeed("/run/a") === stableRenderSeed("/run/b"), false, "different runs receive different stable detail seeds");

const signedStats = { min: -0.3, max: 0.5, p2: -0.2, p98: 0.4 };
equal(defaultsToSignedPair("vorticity_z"), true, "signed vorticity components default to paired extraction");
equal(defaultsToSignedPair("vorticity_magnitude"), false, "unsigned vorticity magnitude stays a single contour");
equal(supportsSignedPair(signedStats), true, "paired extraction requires both resolved signs");
equal(supportsSignedPair({ ...signedStats, min: 0 }), false, "one-signed fields cannot advertise paired extraction");
close(signedThresholdRange(signedStats).max, 0.4, "signed threshold uses a robust shared magnitude scale");
const signedVolume = resolveVolumeTransferPolicy("vorticity_z", "Inferno", signedStats);
equal(signedVolume.kind, "signed", "signed volume fields use a two-tailed opacity policy");

/*
 * Vorticity BUDGET terms are signed too, and this is the case that was wrong:
 * they rendered on a sequential ramp, which shows magnitude and silently drops
 * direction. For a tendency the sign IS the finding — whether that term is
 * generating rotation or destroying it — so a budget read off an unsigned ramp
 * is not just less useful, it is unreadable in the way that matters.
 */
for (const term of ["stretching_term", "tilting_term", "baroclinic_term"]) {
  equal(defaultsToSignedPair(term), true, `${term} extracts as a signed pair`);
  equal(
    resolveVolumeTransferPolicy(term, "Inferno", signedStats).kind,
    "signed",
    `${term} volume uses a signed transfer policy`,
  );
  equal(
    defaultVolumeColormap(term),
    "Cool to Warm",
    `${term} defaults to a diverging colormap`,
  );
}
close(signedVolume.min, -0.4, "signed volume range is robust and symmetric below zero");
close(signedVolume.max, 0.4, "signed volume range is robust and symmetric above zero");
equal(defaultVolumeColormap("vorticity_z"), "Cool to Warm", "signed volume fields default to a diverging palette");
equal(defaultVolumeColormap("vorticity_magnitude"), "Inferno", "unsigned magnitude fields retain a sequential palette");
equal(defaultVolumeColormap("okubo_weiss"), "Cool to Warm", "Okubo-Weiss volume separates rotation and strain around zero");
equal(VOLUME_OPACITY_REFERENCE_M, 1000, "generic volume opacity uses one fixed physical reference distance");
equal(volumeOpacityClass("vorticity_magnitude"), "sparse-positive", "vorticity magnitude uses sparse structural opacity");
equal(volumeOpacityClass("buoyancy"), "signed-perturbation", "buoyancy uses signed structural opacity");
equal(volumeOpacityClass("temperature"), "background-stratified", "temperature is not treated as sparse storm material");
equal(volumeFieldWindowAggregation("vorticity_magnitude"), "block_max", "vorticity overview preserves thin positive extrema");
equal(volumeFieldWindowAggregation("buoyancy"), "magnitude_extreme", "signed overview preserves the strongest sign in each block");
equal(volumeFieldWindowAggregation("temperature"), "cell_mean", "background thermodynamics retain conservative overview means");
equal(volumeOpacityReferenceM("sparse-positive"), 100, "sparse structures use the qualified crisp 100 m opacity reference");
equal(volumeOpacityReferenceM("signed-perturbation"), 500, "signed perturbations use a bounded 500 m opacity reference");
equal(volumeUsesStructuralLighting("sparse-positive"), true, "sparse structures retain scalar-gradient definition");
equal(volumeUsesStructuralLighting("signed-perturbation"), false, "signed volumes avoid false opacity-normal striation");
equal(volumeUsesStructuralLighting("background-stratified"), false, "background thermodynamics avoid layered opacity-normal shading");
equal(volumeUsesOpacityNormals("sparse-positive"), false, "sparse structures do not shade transfer-function knees");
equal(volumeUsesOpacityNormals("reflectivity"), false, "reflectivity shades scalar gradients rather than transfer-function knees");

const wholeDomainSparse = resolveVolumeTransferPolicy("vorticity_magnitude", "Inferno", {
  min: 0,
  max: 0.45,
  p2: 0,
  p98: 0.0559,
  p995: 0.082,
});
equal(wholeDomainSparse.max, 0.082, "visible sparse structures retain colour distinctions through the opacity tail");
equal(wholeDomainSparse.opacityMax, 0.082, "sparse opacity uses the bounded structural tail");
equal(
  volumeOpacityPoints(wholeDomainSparse, 1).at(-1)?.scalar,
  0.082,
  "sparse opacity occupancy is independent of the colourbar maximum",
);

deepEqual(
  DBZ_PHYSICAL_STOPS.map((stop) => stop.scalar),
  [...new Set(DBZ_PHYSICAL_STOPS.map((stop) => stop.scalar))],
  "the canonical dBZ table has no duplicate physical anchors",
);
equal(
  DBZ_PHYSICAL_STOPS.every((stop, index) => index === 0
    || stop.scalar > DBZ_PHYSICAL_STOPS[index - 1].scalar),
  true,
  "canonical dBZ anchors are strictly increasing",
);
deepEqual(
  DBZ_COLORSCALE.map((stop) => stop[1]),
  DBZ_PHYSICAL_STOPS.map((stop) => `rgb(${stop.rgb.join(",")})`),
  "Plotly dBZ colors derive from the canonical physical anchors",
);
equal(
  cssColorbarGradient("dBZ").includes("28.5714%")
    && cssColorbarGradient("dBZ").includes("rgb(0,160,255)"),
  true,
  "the 2-D/CSS dBZ legend preserves the physical zero-dBZ anchor",
);
equal(
  cssColorbarGradient("dBZ volume") !== cssColorbarGradient("dBZ"),
  true,
  "the volume legend discloses its compositor-safe optical transform",
);

/*
 * The 3-D dBZ ramp needs headroom the 2-D one must not have. Pinned primaries
 * are correct for a plan view, where a pixel is one dBZ value read off the
 * legend; over an integrated volume ray they saturate to neon instead of
 * averaging. So: same hues, rails pulled in, and the exact stops preserved for
 * every 2-D consumer (plan views, cross-sections, the radar floor).
 */
equal(
  DBZ_VTK_STOPS_VOLUME.length,
  DBZ_VTK_STOPS.length,
  "the volume ramp has one stop per NWS stop",
);
deepEqual(
  DBZ_VTK_STOPS_VOLUME.map((s) => s.scalar),
  DBZ_VTK_STOPS.map((s) => s.scalar),
  "the volume ramp keeps the NWS dBZ anchor values",
);
equal(
  DBZ_VTK_STOPS.some((s) => [s.r, s.g, s.b].some((c) => c === 0 || c === 1)),
  true,
  "the 2-D NWS ramp still uses pinned primaries",
);
equal(
  DBZ_VTK_STOPS_VOLUME.every((s) => [s.r, s.g, s.b].every((c) => c > 0 && c < 1)),
  true,
  "no volume-ramp channel sits on a rail, so the compositor has headroom",
);
{
  // Hue order is what makes the scale readable; compression must not reorder.
  const rails = (stops: ReadonlyArray<{ r: number; g: number; b: number }>) =>
    stops.map((s) => [s.r > s.g, s.g > s.b, s.r > s.b].join(","));
  deepEqual(
    rails(DBZ_VTK_STOPS_VOLUME),
    rails(DBZ_VTK_STOPS),
    "channel ordering is identical at every stop, so hues are unchanged",
  );
}

/*
 * Turning dBZ opacity up used to flatten the curve: `min(1, a * scale)` clamps
 * the anchors from the top down, so past ~2.4x the 40, 55 and 75 dBZ stops were
 * all exactly 1.0 and the storm rendered as an opaque shell of saturated colour
 * with no internal structure. Optical-depth scaling keeps them ordered.
 */
equal(opticalDepthOpacity(0.42, 1), 0.42, "unit scale leaves an anchor untouched");
equal(opticalDepthOpacity(0.42, 0), 0, "zero scale is fully transparent");
equal(opticalDepthOpacity(0, 4), 0, "a transparent anchor stays transparent at any scale");
equal(opticalDepthOpacity(1, 4), 1, "an opaque anchor stays opaque");
equal(opticalDepthOpacity(0.42, 4) < 1, true, "a mid anchor does not saturate at 4x");
equal(
  opticalDepthOpacity(0.1, 4) < opticalDepthOpacity(0.42, 4)
    && opticalDepthOpacity(0.42, 4) < opticalDepthOpacity(0.72, 4),
  true,
  "anchors keep their order at 4x rather than collapsing together",
);

close(
  displayDistanceForPhysicalRay(1_000, { xy: 2, z: 2 }, [1, 1, 1]),
  2_000,
  "isotropic scene scale preserves a one-kilometre physical opacity reference",
);
close(
  displayDistanceForPhysicalRay(1_000, { xy: 1, z: 4 }, [0, 0, -1]),
  4_000,
  "vertical exaggeration scales the opacity reference for a vertical ray",
);
close(
  displayDistanceForPhysicalRay(1_000, { xy: 1, z: 4 }, [1, 0, 0]),
  1_000,
  "vertical exaggeration does not alter a horizontal ray's physical opacity",
);
{
  const sequentialAt1 = volumeOpacityPoints({ kind: "sequential", min: 0, max: 100 }, 1);
  const sequentialAtHigh = volumeOpacityPoints({ kind: "sequential", min: 0, max: 100 }, 2.5);
  deepEqual(
    sequentialAt1.slice(0, 2),
    [
      { scalar: 0, opacity: 0 },
      { scalar: 45, opacity: 0 },
    ],
    "generic volumes keep the lower robust-range background exactly transparent",
  );
  equal(
    sequentialAtHigh.every((point, index) =>
      index === 0 || point.opacity >= sequentialAtHigh[index - 1].opacity
    ),
    true,
    "generic volume opacity remains monotone at the maximum UI density",
  );
  equal(
    sequentialAtHigh.slice(2).every((point) => point.opacity < 1),
    true,
    "generic volume density does not clamp feature anchors into one opaque block",
  );
}
{
  const dbzAt1 = volumeOpacityPoints({ kind: "dbz", min: -30, max: 75 }, 1);
  deepEqual(
    dbzAt1.map((p) => p.opacity),
    [0, 0, 0.1, 0.42, 0.72, 0.9],
    "the default dBZ curve is unchanged",
  );
  const dbzAt4 = volumeOpacityPoints({ kind: "dbz", min: -30, max: 75 }, 4);
  const saturated = dbzAt4.filter((p) => p.opacity >= 1).length;
  equal(saturated <= 1, true, `at 4x at most the top dBZ stop is opaque (got ${saturated})`);
}
{
  const signed = volumeOpacityPoints({ kind: "signed", min: -2, max: 2 }, 1);
  deepEqual(
    signed.map((point) => point.scalar),
    [-2, -1.24, -0.84, -0.56, 0.56, 0.84, 1.24, 2],
    "signed volumes keep a 28-percent transparent deadband around the midpoint",
  );
  equal(
    signed[2]?.opacity === 0.045 && signed[5]?.opacity === 0.045,
    true,
    "signed weak tails remain faint outside the widened structural deadband",
  );
}
{
  const sparse = volumeOpacityPoints({
    kind: "sequential", opacityClass: "sparse-positive", min: 0, max: 100,
  }, 1);
  equal(sparse[1].scalar, 45, "sparse volume keeps the lower 45 percent transparent");
  equal(sparse.at(-1)?.opacity, 0.94, "sparse volume gives resolved peaks defined opacity");
  const background = volumeOpacityPoints({
    kind: "sequential", opacityClass: "background-stratified", min: 200, max: 300,
  }, 1);
  equal(background[1].scalar, 215, "raw background thermodynamics retain the full physical range");
  equal(background.at(-1)?.opacity, 0.1, "raw background thermodynamics remain optically thin through a full column");
}

// Reflectivity straddles zero so `supportsSignedPair` accepts it, but it is a
// log intensity: promoting it made clear air as opaque as a downpour.
const dbzSpanningStats = { min: -30, max: 52.7, p2: -30, p98: 48 };
for (const palette of ["Cool to Warm", "Inferno", "Viridis"]) {
  equal(
    resolveVolumeTransferPolicy("reflectivity_dbz", palette, dbzSpanningStats).kind,
    "sequential",
    `reflectivity under ${palette} stays sequential despite straddling zero`,
  );
}
equal(
  resolveVolumeTransferPolicy("reflectivity_dbz", "dBZ", dbzSpanningStats).kind,
  "dbz",
  "the dBZ palette still selects the fixed -30..75 reflectivity policy",
);

/*
 * The -30..75 window belongs to reflectivity, not to the palette. Keyed on the
 * palette alone it was applied to whatever field was loaded: NWS dBZ on
 * vorticity_magnitude (0..0.004) put every sample below the 5 dBZ opacity
 * anchor and onto the single colour stop nearest zero — a flat transparent
 * smear. The palette must choose colours; the field chooses the policy.
 */
{
  const vorticityStats = { min: 0, max: 0.0038, p2: 0.00004, p98: 0.0031 };
  const policy = resolveVolumeTransferPolicy("vorticity_magnitude", "dBZ", vorticityStats);
  equal(policy.kind, "sequential", "the dBZ palette on a non-reflectivity field is sequential");
  close(policy.min, 0.00004, "its range comes from the field, not from -30 dBZ");
  close(policy.max, 0.0031, "its range comes from the field, not from 75 dBZ");
  // A signed field keeps its own semantics under the dBZ palette too.
  equal(
    resolveVolumeTransferPolicy("vorticity_z", "dBZ", signedStats).kind,
    "signed",
    "a signed field under the dBZ palette stays signed",
  );
}
// The palette must still be able to promote a genuinely diverging field, or the
// fix would have removed a real capability rather than a misapplication.
equal(
  resolveVolumeTransferPolicy("some_unlisted_tendency", "Cool to Warm", signedStats).kind,
  "signed",
  "Cool to Warm still promotes a non-reflectivity field that straddles zero",
);
const emptyOpacity = volumeOpacityPoints(
  resolveVolumeTransferPolicy(
    "vorticity_magnitude",
    "Inferno",
    { min: 0, max: 0, p2: 0, p98: 0 },
  ),
  1,
);
equal(
  emptyOpacity.every((point) => point.opacity === 0),
  true,
  "constant initial volume fields remain fully transparent",
);
equal(defaultsToRotationDominatedSurface("okubo_weiss"), true, "Okubo-Weiss defaults to a rotation-core surface");
const owRange = rotationDominatedThresholdRange({ min: -0.08, max: 0.05, p2: -0.04, p98: 0.03 });
close(owRange.min, -0.04, "Okubo-Weiss slider starts on its robust negative tail");
close(owRange.max, -0.0008, "Okubo-Weiss slider approaches zero without crossing into strain");
close(
  initialRotationDominatedThreshold({ min: -0.08, max: 0.05, p2: -0.04, p98: 0.03 }),
  -0.014,
  "Okubo-Weiss initializes to a negative rotation-dominated contour",
);
equal(defaultsToPositiveQSurface("q_criterion"), true, "Q criterion defaults to its positive rotation-dominated tail");
const qRange = positiveQThresholdRange({ min: -0.05, max: 0.09, p2: -0.03, p98: 0.06 });
close(qRange.min, 0.0012, "Q slider excludes the zero/background surface");
close(qRange.max, 0.06, "Q slider uses its robust positive tail");
close(
  initialPositiveQThreshold({ min: -0.05, max: 0.09, p2: -0.03, p98: 0.06 }),
  0.021,
  "Q criterion initializes to a positive rotation-dominated contour",
);
close(
  positiveQThresholdRange({
    min: -0.05,
    max: 0.09,
    p2: -0.03,
    p98: 1e-8,
    positiveP98: 0.045,
  }).max,
  0.045,
  "compact Q cores use the positive-conditioned tail instead of domain background",
);

{
  const source = readFileSync(
    new URL("../../../../ui/src/components/plots/Isosurface3D.tsx", import.meta.url),
    "utf8",
  );
  equal(
    source.includes("const qFields = useMemo<"),
    true,
    "Q-criterion keeps its coherent u/v/w tuple stable across renders",
  );
  equal(
    source.includes("thresholdFieldName !== fieldName") &&
      source.includes("threshold !== committedThreshold"),
    true,
    "isosurfaces never contour a new field with the prior field's debounced threshold",
  );
  equal(
    source.includes("primarySurfaceVisible")
      && source.includes("negativeSurfaceVisible")
      && source.includes("contourFrameCommitted"),
    true,
    "isosurface legends are derived from committed visible triangle sets",
  );
}

{
  const source = readFileSync(
    new URL("../../../../ui/src/components/plots/VolumeRender3D.tsx", import.meta.url),
    "utf8",
  );
  equal(
    source.includes("overviewPreparation")
      && source.includes("contextVolume")
      && source.includes("VOLUME_CONTEXT_OPACITY_FACTOR"),
    true,
    "native-detail volume retains a deliberately faint full-domain context actor",
  );
  equal(
    source.includes("coarsestVtkSampling"),
    true,
    "hybrid volume disclosure reports the coarser context as well as native detail",
  );
  equal(
    source.includes("currentStep?.publicationGeneration ?? currentStep?.index")
      && source.includes("sparseReducedOverviewActive")
      && source.includes("effectiveMainOpacityScale"),
    true,
    "volume detail is frame-local and an over-wide sparse overview stays faint instead of exposing a cropped box",
  );
  equal(
    source.includes('fieldOpacityClass === "sparse-positive"')
      && source.includes("{ prefetch: !fullNativeInteractionPreview }")
      && source.includes("volumeControlPreviewProps")
      && source.includes("Reduced volume while moving"),
    true,
    "heavy native volumes suppress speculative prefetch and retain a reduced volume during interaction",
  );
}

{
  const source = readFileSync(
    new URL("../../../../ui/src/components/plots/SupercellRender3D.tsx", import.meta.url),
    "utf8",
  );
  equal(
    source.includes("vortexLegendCommitted")
      && source.includes("precipLegendCommitted")
      && source.includes("exactSceneCommitted"),
    true,
    "supercell optional legends require an exact committed scene",
  );
  equal(
    source.includes("result.core?.triangles")
      && source.includes("result.opposite?.triangles")
      && source.includes("result.mesocyclone?.triangles"),
    true,
    "supercell vortex legend items require their own visible triangle sets",
  );
}

equal(
  availableScalarPlanes(["theta_prime", "w", "reflectivity_dbz", "buoyancy"]).join(","),
  "reflectivity_dbz,buoyancy,theta_prime",
  "scalar-plane choices have stable scientific layer ordering",
);
equal(scalarPlaneReduction("reflectivity_dbz"), "column-max", "dBZ plane uses a column maximum");
equal(scalarPlaneReduction("buoyancy"), "surface", "buoyancy plane uses the lowest valid level");
const signedPlane = resolveScalarPlanePolicy("theta_prime", signedStats);
equal(signedPlane.kind, "signed", "two-signed thermodynamic sheets use a diverging mapping");
close(signedPlane.min, -0.4, "signed plane range is symmetric below zero");
close(signedPlane.max, 0.4, "signed plane range is symmetric above zero");
equal(
  describeScalarPlane(signedPlane),
  "registered lowest valid level plane · symmetric diverging",
  "scalar-plane provenance discloses reduction and mapping",
);

/* ---------- dBZ opacity is anchored on reflectivity, not on range fractions ----
 * dBZ is the one policy whose range is FIXED ([-30, 75]) rather than fitted to
 * the data, and its lower half means "no signal". Sharing the fraction-of-range
 * curve with the fitted policies put clear air at a nonzero opacity — hazing the
 * whole domain — while the 20-50 dBZ band a supercell actually occupies stayed
 * near-transparent. */
{
  const dbz = volumeOpacityPoints({ kind: "dbz", min: -30, max: 75 }, 1);
  const opacityAt = (scalar: number): number => {
    const sorted = [...dbz].sort((a, b) => a.scalar - b.scalar);
    if (scalar <= sorted[0].scalar) return sorted[0].opacity;
    for (let i = 1; i < sorted.length; i++) {
      if (scalar <= sorted[i].scalar) {
        const lo = sorted[i - 1];
        const hi = sorted[i];
        const t = (scalar - lo.scalar) / (hi.scalar - lo.scalar);
        return lo.opacity + t * (hi.opacity - lo.opacity);
      }
    }
    return sorted[sorted.length - 1].opacity;
  };

  equal(opacityAt(-30), 0, "the bottom of the dBZ range is fully transparent");
  equal(opacityAt(0), 0, "clear air at 0 dBZ contributes nothing");
  equal(opacityAt(5), 0, "the no-signal threshold is the last fully transparent point");
  equal(opacityAt(20) > 0.05, true, "precipitation at 20 dBZ is visible");
  equal(opacityAt(40) > 0.3, true, "heavy precipitation at 40 dBZ reads clearly");
  equal(opacityAt(55) > 0.6, true, "a hail core at 55 dBZ reads strongly");
  // Monotone: a stronger echo can never render more transparent than a weaker one.
  let previous = -1;
  let monotone = true;
  for (const scalar of [-30, -10, 5, 15, 20, 30, 40, 50, 55, 65, 75]) {
    const value = opacityAt(scalar);
    if (value < previous - 1e-9) monotone = false;
    previous = value;
  }
  equal(monotone, true, "dBZ opacity never decreases as reflectivity rises");
  // The scale control still has authority over the whole curve.
  const dimmed = volumeOpacityPoints({ kind: "dbz", min: -30, max: 75 }, 0.5);
  equal(
    dimmed.every((point, i) => point.opacity <= dbz[i].opacity + 1e-9),
    true,
    "halving the opacity scale cannot raise any dBZ control point",
  );
}

/* ---------- Rendered-cloud volume compositing ----------
 * vtk.js draws translucent geometry before volumes and feeds its depth into the
 * volume ray bounds. A cloud polygon shell therefore clips the volume it was
 * meant to decorate. Independent liquid/frozen volume actors also threshold
 * each phase rather than their physical sum and composite in actor order. */
{
  const supercell = readFileSync(
    new URL("../../../../ui/src/components/plots/SupercellRender3D.tsx", import.meta.url),
    "utf8",
  );

  equal(
    !supercell.includes("cloudSurfaceActor") && !supercell.includes("cloudSurfaceMarching"),
    true,
    "the cloud has no polygon shell that can clip or grey the volume pass",
  );
  equal(
    !supercell.includes("frozenCloudVolume") && !supercell.includes("liquidCloudPreparation"),
    true,
    "liquid and frozen condensate are not rendered as order-biased independent volumes",
  );
  equal(
    supercell.includes("Aggregate suspended condensate owns both visibility and opacity"),
    true,
    "the source documents aggregate condensate as the cloud opacity owner",
  );
  equal(
    ["off", "subtle", "full"].every((value) =>
      supercell.includes(`{ value: "${value}"`)
    ),
    true,
    "volume-native edge emphasis exposes off / subtle / full",
  );
  equal(
    supercell.includes("cloudOpacityPoints(floor, CLOUD_DISPLAY_MAX, peak, edge)"),
    true,
    "the edge control reshapes the volume transfer function",
  );
  equal(
    !supercell.includes("CLOUD_SURFACE_VOXEL_CAP") && !supercell.includes("16_000_000"),
    true,
    "removing the contour shell also removes its 16-million-voxel ceiling",
  );
  equal(
    supercell.includes("mapper: [contextCloudMapper, cloudMapper, condensationMapper, precipMapper]"),
    true,
    "interactive quality controls every active ray-marched volume mapper",
  );
  equal(
    supercell.includes('cloudEdge === "off" ? "no edge emphasis"'),
    true,
    "the active volume-edge setting is disclosed on canvas",
  );
  equal(
    supercell.includes("setCameraAdjustedVolumeRayDistances("),
    true,
    "cloud sampling and its one-kilometre opacity reference are corrected for display scaling",
  );
  equal(
    supercell.includes("setPreparedImageDisplayGeometry(scene.cloudImage") &&
      supercell.includes("scene.cloudVolume.setScale(1, 1, 1)"),
    true,
    "scaled cloud bounds live on ImageData so vtk compiles enough ray-march iterations",
  );
}

/* ---------- Opening geometry is undistorted ----------
 * A scene must open at the domain's true aspect. The old rule auto-seeded 2-6x
 * vertical exaggeration from the domain shape, so the first thing a user saw was
 * a stretched storm — and updraft slope, anvil spread and overshoot depth are
 * all read geometrically. Exaggeration stays available; it just is not the
 * default, and it is disclosed on canvas whenever it is not 1x. */
{
  const wideShallow = {
    nr: 200, nth: 200, nz: 40,
    drM: 250, dyM: 250, dzM: 250,
  } as unknown as Parameters<typeof autoVerticalExag>[0];
  const deep = {
    nr: 40, nth: 40, nz: 200,
    drM: 250, dyM: 250, dzM: 250,
  } as unknown as Parameters<typeof autoVerticalExag>[0];

  equal(VERTICAL_EXAG_DEFAULT, 1, "the default vertical exaggeration is true aspect");
  equal(autoVerticalExag(wideShallow), 1, "a wide shallow domain still opens undistorted");
  equal(autoVerticalExag(deep), 1, "a deep domain still opens undistorted");
  equal(autoVerticalExag(null), 1, "an unknown grid opens undistorted");
  equal(
    VERTICAL_EXAG_RANGE.min <= 1 && VERTICAL_EXAG_RANGE.max > 1,
    true,
    "exaggeration remains reachable from the control in both directions",
  );
}

/* ---------- Soundings are portrait figures ----------
 * A sounding plots height or pressure up the y axis over a narrow x range.
 * Stretched to the full width of a workstation window it becomes a ~4:1
 * letterbox: a Skew-T's isotherms flatten until the 45-degree skew it is named
 * for no longer holds, and an equal-aspect hodograph collapses to a small square
 * pinned against one edge. */
{
  const portraitTypes = PLOT_TYPE_META.filter((m) => m.portrait).map((m) => m.value).sort();
  equal(
    portraitTypes.join(","),
    "hodograph,skewt,vertical",
    "exactly the vertically-read diagrams are marked portrait",
  );
  // A profile along a horizontal axis is a landscape figure and must not be capped.
  equal(
    PLOT_TYPE_META.find((m) => m.value === "radial")?.portrait ?? false,
    false,
    "an x profile is read horizontally and keeps the full width",
  );
  equal(
    PLOT_TYPE_META.find((m) => m.value === "slice")?.portrait ?? false,
    false,
    "a slice heatmap keeps the full width",
  );
  const studioCss = readFileSync(
    new URL("../../../../ui/src/components/studio/PlotStudio.css", import.meta.url),
    "utf8",
  );
  // The default wrapper must generate no box at all, so marking a plot portrait
  // cannot change the layout of any plot that is not.
  equal(
    /\.plot-studio__plot-area\s*\{\s*display:\s*contents;/.test(studioCss),
    true,
    "the non-portrait plot area generates no box",
  );
  equal(
    studioCss.includes(".plot-studio__pane--portrait"),
    true,
    "each portrait pane caps its own figure",
  );
}

/* ---------- A fixed-anchor palette pins its own range ----------
 * The NWS reflectivity ramp's colours are defined AT dBZ values, which is why a
 * radar image is readable without a legend. Selecting it while the separate
 * range mode sat on its "auto" default stretched those stops over the fitted
 * data range: on a [-13, 58] fit the colour meaning 60 dBZ was painted on the
 * 48 dBZ cells. The plot still looked like a radar product while every value it
 * encoded was wrong. */
{
  equal(usesFixedAnchorRange("dBZ"), true, "the NWS reflectivity ramp is anchored");
  for (const scale of ["Viridis", "Inferno", "RdBu", "Greys", "ColdPool"] as const) {
    equal(usesFixedAnchorRange(scale), false, `${scale} is a positional ramp and may be fitted`);
  }
  equal(DBZ_RANGE[0], -30, "the anchored range starts at -30 dBZ");
  equal(DBZ_RANGE[1], 75, "the anchored range ends at 75 dBZ");

  // Every 2D renderer that offers the ramp must consult the guard, not just the
  // range-mode pill — that coupling is the whole defect.
  for (const file of ["SlicePlot.tsx", "CompositePlot.tsx", "ComparisonSlicePlot.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    equal(
      source.includes("usesFixedAnchorRange("),
      true,
      `${file} pins the anchored range independently of the range mode`,
    );
  }
}

/* ---------- Core 2D plots preserve resolved cells ----------
 * Plotly's `best` smoothing is a renderer-owned presentation reconstruction:
 * its exact algorithm is not part of our scientific contract, it can visually
 * bridge missing cells, and it makes vector arrows disagree with their magnitude
 * background. Raw cells keep masks, comparisons, swaths, and hover values honest. */
{
  for (const file of [
    "SlicePlot.tsx",
    "CompositePlot.tsx",
    "SwathPlot.tsx",
    "TimeHeightPlot.tsx",
    "ComparisonSlicePlot.tsx",
    "ComparisonVectorPlot.tsx",
    "ComparisonCompositePlot.tsx",
    "EnsembleAnalysisPlot.tsx",
    "PythonPlot.tsx",
  ]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    const heatmaps = source.split('type: "heatmap"').length - 1;
    const exact = source.split("zsmooth: false").length - 1;
    equal(heatmaps > 0, true, `${file} draws at least one heatmap`);
    equal(
      exact === heatmaps && !source.includes('zsmooth: "best"'),
      true,
      `${file} draws every resolved heatmap cell without implicit reconstruction`,
    );
  }

  const investigation = readFileSync(
    new URL("../../../../ui/src/components/studio/InvestigationWorkspace.tsx", import.meta.url),
    "utf8",
  );
  equal(
    investigation.split("zsmooth: false").length - 1
      === investigation.split('type: "heatmap"').length - 1,
    true,
    "investigation plan and transect heatmaps preserve resolved cells",
  );
}

/* Contour smoothing changes the geometry of field = level after marching
 * contours has located it. The composite renderer must retain exact isolines. */
{
  for (const file of ["CompositePlot.tsx", "ComparisonCompositePlot.tsx", "EnsembleAnalysisPlot.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    const contours = source.split('type: "contour"').length - 1;
    equal(
      contours > 0
        && source.split("smoothing: 0").length - 1 === contours
        && !source.includes("smoothing: 1"),
      true,
      `${file} does not spline-shift its resolved isolines`,
    );
  }
}

/* Skew-T parcel diagnostics must never turn missing moisture into dry air or
 * substitute an aloft qv value at the surface. Either changes virtual
 * temperature, parcel ascent, and CAPE/CIN. */
{
  const source = readFileSync(
    new URL("../../../../ui/src/components/plots/SkewT.tsx", import.meta.url),
    "utf8",
  );
  equal(
    source.includes("levels.some((level) => !Number.isFinite(level.qv) || level.qv < 0)"),
    true,
    "Skew-T parcel diagnostics reject an incomplete moisture column",
  );
  equal(
    source.includes("Number.isFinite(l.qv) ? l.qv : 0") || source.includes("levels.find((l) => l.qv > 0)"),
    false,
    "Skew-T never fabricates dry layers or surface parcel moisture",
  );
}

/* Every renderer that OFFERS the NWS ramp must also pin its range. Offering a
 * fixed-anchor palette while auto-ranging under it is the exact defect. */
{
  for (const file of ["TimeHeightPlot.tsx", "SwathPlot.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    if (!source.includes('"dBZ"')) continue;
    equal(
      source.includes("plotlyColorRange("),
      true,
      `${file} routes the NWS ramp through the shared anchored-range policy`,
    );
  }
}

{
  const volume = readFileSync(
    new URL("../../../../ui/src/components/plots/VolumeRender3D.tsx", import.meta.url),
    "utf8",
  );
  equal(
    volume.includes('refineForDisplay: renderFidelityMode === "presentation"'),
    true,
    "Scientific volume handoff does not densify a reduced lattice for display",
  );
}

{
  for (const file of ["SlicePlot.tsx", "CompositePlot.tsx", "VectorField.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    equal(source.includes("onRelayout={onRelayout}"), true,
      `${file} turns Plotly viewport changes into bounded source refinement`);
    equal(/usePlaneViewport\((planeViewIdentity|viewIdentity)\)/.test(source), true,
      `${file} scopes debounced viewport reads to its displayed plane`);
  }
}

/* ---------- A new frame must not move the camera ----------
 * The scale re-fit effect also depended on the prepared field, which changes on
 * every frame, so scrubbing and playback silently re-framed the camera and threw
 * away the user's zoom and pan. Streamlines already documented this exact bug
 * and depends on scale alone; Volume and Isosurface now gate on a framing guard
 * that reports a change at most once per real scale move. */
{
  for (const file of ["VolumeRender3D.tsx", "Isosurface3D.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    equal(
      source.includes("scaleFraming.consume()"),
      true,
      `${file} re-frames only when the display scale actually moved`,
    );
    // The guard is worthless if the fit is also reachable unguarded.
    const guarded = source.split("cameraInitializedRef.current && scaleFraming.consume()").length - 1;
    equal(guarded >= 1, true, `${file} guards its scale re-fit`);
  }
  // These scenes also rebuild their VTK objects when the active backend changes.
  // Re-fit on scale or that lifecycle boundary, but never on per-frame prepared
  // data (which would overwrite the user's camera during playback).
  for (const file of ["Streamlines3D.tsx", "SupercellRender3D.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    equal(
      source.includes("}, [sceneScale.xy, sceneScale.z, vtkLifecycleGeneration]);"),
      true,
      `${file} keeps its camera re-fit keyed on scale and renderer lifecycle only`,
    );
  }
}

/* ---------- Zoom-out stops at the simulated domain ---------- */
{
  const bounds = axisViewBounds([0, 1, 2, 3, 4]);
  equal(bounds != null, true, "a finite axis yields bounds");
  equal(bounds!.minallowed < 0, true, "the lower bound leaves reachable margin past the data");
  equal(bounds!.maxallowed > 4, true, "the upper bound leaves reachable margin past the data");
  const span = bounds!.maxallowed - bounds!.minallowed;
  equal(span < 4 * 2, true, "the margin is bounded, not unlimited");
  equal(axisViewBounds([]), undefined, "an empty axis falls back to Plotly's default");
  equal(axisViewBounds(null), undefined, "a missing axis falls back to Plotly's default");
  equal(axisViewBounds([2, 2, 2]), undefined, "a degenerate axis falls back to Plotly's default");
  equal(
    axisViewBounds([Number.NaN, 1, 5, Number.POSITIVE_INFINITY])?.maxallowed !== undefined,
    true,
    "non-finite coordinates are skipped rather than poisoning the bound",
  );
}

/* ---------- Time-accumulating plots must consult the motion frame ----------
 * docs/domain.md: "Storm tracks and swaths must accumulate in Earth
 * coordinates. A fixed-size moving window cannot represent a lifetime swath in
 * one local array without overwriting its past." A swath combines many frames
 * by grid CELL, so a translating domain invalidates it silently — the smeared
 * result still looks like a storm swath. */
{
  const swath = readFileSync(
    new URL("../../../../ui/src/components/plots/SwathPlot.tsx", import.meta.url),
    "utf8",
  );
  equal(
    swath.includes("canAccumulateByCell("),
    true,
    "the swath consults the domain-motion frame before accumulating",
  );
  // A translating run with coherent geometry is accumulated for real, in Earth
  // coordinates. The refusal is reserved for frames that cannot be PLACED —
  // an unrecognised mode, a resolution change, a non-finite origin.
  equal(
    swath.includes("earthSwathGrid("),
    true,
    "the swath accumulates in Earth coordinates when the domain moves",
  );
  equal(
    swath.includes("if (!accumulation.allowed && earthGrid == null) {"),
    true,
    "the swath refuses only when frames cannot be placed in Earth coordinates",
  );
  equal(
    swath.includes("earthCellOffset("),
    true,
    "each frame is placed at its own Earth origin",
  );
  equal(
    swath.includes("describeEarthSwath("),
    true,
    "a swath accumulated across a moving domain discloses its binning",
  );
  equal(
    swath.includes("native horizontal cells · exact"),
    true,
    "stationary and moving swaths disclose native spatial sampling and exact reduction",
  );
  equal(
    swath.includes('ownership: "transferable"')
      && swath.includes('cacheRetention: "transient"')
      && swath.includes("[request.data.buffer as ArrayBuffer]"),
    true,
    "swath reads use exclusive transient ownership and transfer the field without cloning",
  );
  equal(
    swath.includes("accumulation.allowed, earthGrid]"),
    true,
    "the refusal is part of the accumulation effect's dependencies",
  );
}

/* ---------- Every bounded multi-source view discloses its reduced lattice ---------- */
for (const [file, marker] of [
  ["ComparisonSlicePlot.tsx", "displayPlans?.samplingDisclosure"],
  ["ComparisonCompositePlot.tsx", "plans?.samplingDisclosure"],
  ["EnsembleAnalysisPlot.tsx", "displayed?.samplingDisclosure"],
] as const) {
  const source = readFileSync(
    new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
    "utf8",
  );
  equal(
    source.includes(marker),
    true,
    `${file} renders bounded sampling provenance beside the figure`,
  );
}

/* ---------- Differencing two runs needs a shared frame, not just shared axes ----
 * Axis co-location compares GRID-relative coordinates. Two runs whose domains
 * translate across the Earth can report identical axis values for entirely
 * different places, so the axis check passes while the difference subtracts one
 * location from another. */
{
  const comparison = readFileSync(
    new URL("../../../../ui/src/components/plots/ComparisonSlicePlot.tsx", import.meta.url),
    "utf8",
  );
  equal(
    comparison.includes("canCompareByCell("),
    true,
    "the comparison consults the domain-motion frame",
  );
  equal(
    comparison.includes("&& frameComparison.allowed"),
    true,
    "the frame check gates the difference alongside axis co-location",
  );
  equal(
    comparison.includes("frameComparison.blocker ??"),
    true,
    "a frame refusal states its own reason rather than the generic axis one",
  );
}

/* ---------- Tubes stay round under vertical exaggeration ----------
 * An anisotropic actor scale stretches a tube's CROSS SECTION as well as its
 * path: at 3.5x vertical exaggeration a horizontally-running tube rendered 3.5x
 * fatter than a vertical one, so apparent line weight encoded direction. Scaling
 * the polyline points ahead of the tube filter keeps every section circular. */
equal(
  defaultStreamlineTubeRadiusM([2000, 2000, 200], [50, 50, 50]),
  250,
  "a 100 km, 50 m grid receives a visible fitted-view streamline radius",
);
equal(
  defaultStreamlineTubeRadiusM([280, 280, 200], [50, 50, 50]),
  50,
  "a storm-detail view retains a grid-scale streamline radius",
);
equal(
  defaultStreamlineTubeRadiusM([4096, 4096, 200], [100, 100, 100]),
  1000,
  "monster domains clamp streamline radius to the exposed control maximum",
);
{
  // The geometry contract: the scale lands on the POINTS, so a tube built from
  // them has a circular cross section whatever the exaggeration.
  const points = Float32Array.from([1000, 2000, 3000, -500, 0, 250]);
  const scaled = scalePointTriples(points, { xy: 2, z: 3.5 });
  deepEqual(
    Array.from(scaled),
    [2000, 4000, 10500, -1000, 0, 875],
    "scalePointTriples scales x/y by xy and z by z",
  );
  equal(
    points[2],
    3000,
    "scalePointTriples leaves the source points in true coordinates",
  );

  // A tube's radius is applied in the space its input points live in, so equal
  // point spacing in every direction is what keeps the cross section round.
  // Verify the scaling is a pure per-axis factor: no cross-axis coupling.
  const unit = scalePointTriples(Float32Array.from([1, 1, 1]), { xy: 1, z: 6 });
  deepEqual(
    Array.from(unit),
    [1, 1, 6],
    "vertical exaggeration does not disturb the horizontal axes",
  );

  equal(
    (() => {
      try {
        scalePointTriples(Float32Array.from([1, 2]), { xy: 1, z: 1 });
        return "accepted";
      } catch {
        return "rejected";
      }
    })(),
    "rejected",
    "a point buffer that is not a multiple of 3 is rejected, not silently bent",
  );

  // Geometry assembly: fresh buffers, and never the source's own arrays. A
  // vtkTransformPolyDataFilter here shared its input's cell arrays and then
  // cleared them in place on its second execution, silently emptying the tubes.
  const sourceLines = Uint32Array.from([3, 0, 1, 2]);
  const sourceScalars = Float32Array.from([10, 20, 30]);
  const source = {
    getPoints: () => ({ getData: () => Float32Array.from([0, 0, 0, 1, 1, 1, 2, 2, 2]) }),
    getLines: () => ({ getData: () => sourceLines }),
    getPointData: () => ({ getScalars: () => ({ getData: () => sourceScalars }) }),
  };
  const geometry = buildScaledGeometry(source, { xy: 1, z: 4 });
  equal(geometry !== null, true, "buildScaledGeometry returns geometry for a valid source");
  deepEqual(
    Array.from(geometry!.lines),
    [3, 0, 1, 2],
    "line topology is carried onto the scaled copy",
  );
  equal(
    (geometry!.lines as Uint32Array) !== sourceLines,
    true,
    "the scaled copy owns its own line buffer, never the source's",
  );
  equal(
    geometry!.scalars !== sourceScalars,
    true,
    "the scaled copy owns its own scalar buffer, never the source's",
  );
  deepEqual(
    Array.from(geometry!.points).slice(0, 3),
    [0, 0, 0],
    "points are scaled, not reordered",
  );
  equal(geometry!.points[8], 8, "the z component carries the vertical exaggeration");

  // A source a filter has already emptied must read as "nothing to draw" so the
  // caller hides the actor, rather than tubing a half-built data set.
  equal(
    buildScaledGeometry(
      { ...source, getLines: () => ({ getData: () => new Uint32Array(0) }) },
      { xy: 1, z: 1 },
    ),
    null,
    "a source with no line topology yields no geometry",
  );
  equal(
    buildScaledGeometry(null, { xy: 1, z: 1 }),
    null,
    "a missing source yields no geometry",
  );

  for (const file of ["Trajectories3D.tsx"]) {
    const source = readFileSync(
      new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url),
      "utf8",
    );
    // vtkTransformPolyDataFilter aliases its input's cell arrays into its output
    // and vtkPolyData.initialize() then clears them in place, so its second
    // execution empties the very polylines it is meant to transform. See the
    // note in lib/vtk/vtkScaledPolyData.ts.
    equal(
      /^import .*TransformPolyDataFilter/m.test(source),
      false,
      `${file} does not tube through vtkTransformPolyDataFilter`,
    );
    equal(
      source.includes("TrajectoryGeometryService"),
      true,
      `${file} delegates path scaling and tube expansion to its worker service`,
    );
    // The actor must NOT also be scaled, or the stretch is applied twice and the
    // cross section is distorted again.
    equal(
      /applySceneScale\(\s*sceneScale\s*,\s*scene\.(backward)?actor/.test(source),
      false,
      `${file} leaves its tube actor at identity`,
    );
  }
  // Trajectories colour their tubes by a scalar ramp, and vtkTubeFilter marks
  // no active scalars on its output, so the mapper MUST select the array by
  // name -- otherwise it silently falls back to the actor's flat colour and the
  // paths render one grey while the legend advertises a ramp. The behaviour
  // behind this is exercised in ../tests/ui/unit/vtk/test_tube_pipeline.ts; this binds
  // the component to it.
  const trajectories = readFileSync(
    new URL("../../../../ui/src/components/plots/Trajectories3D.tsx", import.meta.url),
    "utf8",
  );
  const trajectoryWorkerGeometry = readFileSync(
    new URL("../../../../ui/src/lib/vtk/trajectoryGeometry.ts", import.meta.url),
    "utf8",
  );
  equal(
    trajectoryWorkerGeometry.includes("z * parameters.scale.z"),
    true,
    "trajectory vertical exaggeration is applied to worker-owned points before tubing",
  );
  equal(
    /setColorByArrayName\(\s*VTK_SCALAR_SLOT\s*\)/.test(trajectories),
    true,
    "Trajectories3D colours its tubes by the owned scalar array's name",
  );
  equal(
    /setScalarMode\(\s*ScalarMode\.USE_POINT_FIELD_DATA\s*\)/.test(trajectories),
    true,
    "Trajectories3D selects that array as point field data",
  );
  equal(
    /^import vtkTubeFilter/m.test(trajectories)
      || trajectories.includes("vtkTubeFilter.newInstance")
      || trajectories.includes("function buildPaths")
      || trajectories.includes("MAX_PARCELS"),
    false,
    "Trajectories3D performs neither UI-thread geometry work nor fixed parcel subsampling",
  );
  const radarFloorHook = readFileSync(
    new URL("../../../../ui/src/lib/vtk/useRadarFloor.ts", import.meta.url),
    "utf8",
  );
  equal(
    radarFloorHook.includes("RadarFloorPreparationService")
      && !radarFloorHook.includes("reduceFloor(")
      && !radarFloorHook.includes("floorStats(")
      && !radarFloorHook.includes("terrainFloorSurface("),
    true,
    "radar floor reduction, statistics, and terrain reconstruction stay behind the worker service",
  );
  const radarFloorLayer = readFileSync(
    new URL("../../../../ui/src/components/plots/radarFloorLayer.ts", import.meta.url),
    "utf8",
  );
  equal(
    radarFloorLayer.includes("RadarFloorEncodingService")
      && !radarFloorLayer.includes("encodeFloorRgba("),
    true,
    "radar floor colour encoding and alpha inspection stay behind the worker service",
  );

  // Streamlines must still INTEGRATE in true coordinates: integrating in a
  // stretched metric would bend the trajectories themselves, not just draw them.
  const streamlines = readFileSync(
    new URL("../../../../ui/src/components/plots/Streamlines3D.tsx", import.meta.url),
    "utf8",
  );
  const streamlineWorker = readFileSync(
    new URL("../../../../ui/src/lib/vtk/streamlineExtraction.worker.ts", import.meta.url),
    "utf8",
  );
  equal(
    /const integrated = streamline\.getOutputData\(\);[\s\S]*scalePointTriples\(rawPoints, request\.scale\)/.test(streamlineWorker),
    true,
    "the off-thread scale is applied to the integrator's OUTPUT, after integration",
  );
  equal(
    streamlines.includes("useStreamlineExtraction") && !streamlines.includes("vtkImageStreamline"),
    true,
    "Streamlines3D isolates integration and tube generation from the UI thread",
  );
  equal(
    streamlines.includes("streamlineGeometryCommitted && colorMode === \"magnitude\""),
    true,
    "streamline colorbars require committed integrated geometry",
  );
  // The field being integrated must stay in true coordinates: a scaled lattice
  // would bend the trajectories themselves, not just how they are drawn.
  equal(
    /imageData\.set(Spacing|Origin)\([^)]*sceneScale/.test(streamlines),
    false,
    "the display scale never reaches the lattice the streamlines integrate through",
  );
}

console.log(`\nplotFidelity tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
