import {
  describeBoundedSampling,
  fieldWindowAggregation,
  windowsForFieldAggregation,
  fieldSupportsBoundedWindows,
  gridHorizontalCenterM,
  interactiveFullDomainSourceSafe,
  minimumNominalGridSpacingM,
  MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS,
  nativeDetailRecenteringThresholdM,
  PLOTLY_HEATMAP_CELL_CAP,
  planExactLineWindows,
  planExactValidDomainWindows,
  planInteractiveVolumeWindows,
  planBoundedSliceWindows,
  planBoundedPlaneWindows,
  planBoundedVolumeWindows,
  planNativeStormDetailWindows,
  planNativeStructuralSupportWindows,
} from "../../../../ui/src/lib/plot/fieldLoadPlanning.ts";
import { supercellAuxiliaryVoxelBudget } from "../../../../ui/src/lib/plot/supercellRenderQuality.ts";
import { aggregateDerivedWindow, assembleWindowedField } from "../../../../ui/src/lib/run/windowedField.ts";
import { gridForWindowedField } from "../../../../ui/src/lib/science/windowedGrid.ts";
import {
  MAX_NATIVE_FIELD_WINDOW_BYTES,
  MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
  nativeFieldWindowBodyBytes,
} from "../../../../ui/src/lib/dataset/nativeFieldWindow.ts";
import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
};

for (const [spacing, cap] of [[100, 12_000_000], [30, 3_000_000]] as const) {
  const nz = Math.ceil(16_000 / spacing);
  const cloudColumn = planNativeStormDetailWindows({
    nr: 1024, nth: 1024, nz, drM: spacing, dyM: spacing, dzM: spacing,
  }, [20_000, 20_000], cap, [28_000, 28_000, 10_000], undefined, true);
  check(cloudColumn.ok, `full-height cloud plan at ${spacing} m`);
  if (cloudColumn.ok) {
    check(cloudColumn.plan.sampledShape[2] === nz, "native cloud retains the entire vertical column");
    check(cloudColumn.plan.sampledCells <= cap, "full-height cloud remains within pane allocation");
    check(cloudColumn.plan.stride.every(value => value === 1), "full-height cloud remains source-native");
  }
}

const grid = (nr: number, nth: number, nz: number): GridInfo => ({
  nr, nth, nz,
  coordinateSystem: "cartesian",
  drM: 100,
  dyM: 100,
  dzM: 50,
  scalarOriginM: [50, 50, 25] as [number, number, number],
  validIndexRange: {
    lower: [1, 1, 0] as [number, number, number],
    upper: [nr - 2, nth - 2, nz - 1] as [number, number, number],
  },
});

check(
  describeBoundedSampling({
    sourceShape: [512, 512, 160], sampledShape: [256, 256, 160], stride: [2, 2, 1],
  }, "theta") === "bounded cell-mean 2×2×1 · source 512×512×160 → display 256×256×160",
  "scalar reduced lattices disclose method, stride, source, and display shape",
);
check(
  describeBoundedSampling({
    sourceShape: [512, 512, 160], sampledShape: [256, 256, 160], stride: [2, 2, 1],
  }, "reflectivity_dbz")?.startsWith("bounded linear-Z mean") === true,
  "reflectivity disclosure names its physically correct aggregation",
);
check(
  describeBoundedSampling({
    sourceShape: [64, 64, 20], sampledShape: [64, 64, 20], stride: [1, 1, 1],
  }, "theta") === null,
  "native lattices do not claim a reduction",
);

for (const [name, shape] of [
  ["beast", [512, 512, 160]],
  ["gigacell", [1024, 1024, 256]],
  ["wide-deep", [4096, 2048, 512]],
] as const) {
  const outcome = planBoundedVolumeWindows(grid(shape[0], shape[1], shape[2]), 16_000_000);
  check(outcome.ok, `${name}: bounded plan succeeds`);
  if (!outcome.ok) continue;
  check(outcome.plan.sampledCells <= 16_000_000, `${name}: hard voxel ceiling is honored`);
  check(outcome.plan.windows.length >= 1, `${name}: at least one source tile`);
  check(outcome.plan.windows.every((window) =>
    window.extent.reduce((n, v) => n * v, 1) * 4 <= MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
  ), `${name}: every source tile is inside the source ceiling`);
  check(outcome.plan.windows.every((window) =>
    window.extent.reduce((n, v) => n * v, 1) * 4 <= 16 * 1024 * 1024
  ), `${name}: dependency tiles keep derive-before-reduce bounded`);
  check(outcome.plan.windows.every((window) =>
    window.extent.reduce((n, v, axis) => n * Math.ceil(v / window.stride[axis]), 1) * 4
      <= MAX_NATIVE_FIELD_WINDOW_BYTES
  ), `${name}: every response is inside the output ceiling`);
}

const huge = planBoundedVolumeWindows(grid(1024, 1024, 256), 16_000_000);
check(huge.ok && huge.plan.windows.length > 1, "1 GiB source is tiled rather than read as one volume");
check(huge.ok && huge.plan.stride[2] > 1, "physically finer vertical sampling may aggregate under a hard cap");

const beast = planBoundedVolumeWindows(grid(512, 512, 160), 16_000_000);
check(
  beast.ok && windowsForFieldAggregation(beast.plan.windows, "cloud_condensate", {
    sourceKind: "native",
    terrainFollowing: false,
  }).length === 1,
  "eligible native Beast volume streams through one bounded IPC transaction",
);
check(
  beast.ok && windowsForFieldAggregation(beast.plan.windows, "theta_e", {
    sourceKind: "studio_derived",
    terrainFollowing: false,
  }).length === beast.plan.windows.length,
  "derived Beast volume retains exact dependency tiles",
);

// Exact geometry from the user's 67 GiB Beast run. Rendered-supercell Flow and
// Vortex must share one sign-complete lattice inside this ROI; block max/min
// pairs would be identical at stride 1 and only duplicate I/O/preparation.
const beastManifestGrid: GridInfo = {
  nr: 512,
  nth: 512,
  nz: 160,
  coordinateSystem: "cartesian",
  drM: 100,
  dyM: 100,
  dzM: 100,
  scalarOriginM: [-25_550, -25_550, 50],
};
const beastDiagnostic = planNativeStormDetailWindows(
  beastManifestGrid,
  [0, 0],
  4_000_000,
);
const locatedBeastStructure = planInteractiveVolumeWindows(
  beastManifestGrid,
  16_000_000,
  [6_150, -3_650],
  { preferNativeDetail: true },
);
const beastStrongSupport = {
  lower: [4_050, -17_250, 150] as [number, number, number],
  upper: [24_850, 24_650, 15_550] as [number, number, number],
};
const supportedBeastStructure = planNativeStructuralSupportWindows(
  beastManifestGrid,
  beastStrongSupport,
  16_000_000,
);
const broadUpdraftSupport = {
  lower: [-9_500, -14_500, 250] as [number, number, number],
  upper: [9_500, 14_500, 14_750] as [number, number, number],
};
const supportedBroadUpdraft = planNativeStructuralSupportWindows(
  beastManifestGrid,
  broadUpdraftSupport,
  supercellAuxiliaryVoxelBudget(16_000_000, 100),
);
const legacyTruncatedUpdraft = planNativeStructuralSupportWindows(
  beastManifestGrid,
  broadUpdraftSupport,
  4_000_000,
);
check(
  supportedBroadUpdraft.ok && !legacyTruncatedUpdraft.ok,
  "pre-counted Supercell Flow expands beyond the legacy 4M box to contain known updraft support",
);
check(
  supportedBeastStructure.ok
    && supportedBeastStructure.plan.sampledCells <= 16_000_000
    && supportedBeastStructure.plan.stride.every((stride) => stride === 1),
  "Beast strong-opacity support fits one stride-1 structural ROI",
);
check(
  supportedBeastStructure.ok
    && supportedBeastStructure.plan.detailIndexRange.lower[2] <= 1
    && supportedBeastStructure.plan.detailIndexRange.upper[2] >= 155,
  "structural-support planning retains the upper-level Beast maximum region",
);
const interactiveSupportedBeast = planInteractiveVolumeWindows(
  beastManifestGrid,
  16_000_000,
  [12_550, 9_150],
  { preferNativeDetail: true, structuralSupportBoundsM: beastStrongSupport },
);
check(
  interactiveSupportedBeast.ok
    && supportedBeastStructure.ok
    && interactiveSupportedBeast.mode === "native-detail"
    && interactiveSupportedBeast.plan.sampledCells === supportedBeastStructure.plan.sampledCells,
  "interactive Volume prefers admissible structural support over the fixed peak box",
);
const tooWideBeastSupport = planInteractiveVolumeWindows(
  beastManifestGrid,
  16_000_000,
  [0, 0],
  {
    preferNativeDetail: true,
    structuralSupportBoundsM: {
      lower: [-25_550, -25_550, 50],
      upper: [25_550, 25_550, 15_950],
    },
  },
);
check(
  tooWideBeastSupport.ok && tooWideBeastSupport.mode === "full-domain",
  "bounded ordinary sources keep complete context instead of cropping known support to a fixed ROI",
);
check(
  locatedBeastStructure.ok && locatedBeastStructure.mode === "native-detail",
  "a located Beast structure promotes the volume from overview to native detail",
);
check(
  locatedBeastStructure.ok
    && locatedBeastStructure.plan.stride.every((stride) => stride === 1),
  "a located Beast structural volume retains stride-1 cells",
);
const unlocatedBeastStructure = planInteractiveVolumeWindows(
  beastManifestGrid,
  16_000_000,
  null,
  { preferNativeDetail: true },
);
check(
  unlocatedBeastStructure.ok && unlocatedBeastStructure.mode === "full-domain",
  "structural volume starts with coherent full-domain context until a feature is located",
);
check(beastDiagnostic.ok, "actual Beast diagnostic ROI is admissible under the auxiliary cap");
check(
  beastDiagnostic.ok && beastDiagnostic.plan.sampledCells === 1_960_000,
  "actual Beast diagnostic ROI retains the full 14×14×10 km native lattice",
);
check(
  beastDiagnostic.ok && beastDiagnostic.plan.stride.every((stride) => stride === 1),
  "actual Beast diagnostic ROI needs one sign-complete exact field, not extrema envelopes",
);
check(
  beastDiagnostic.ok && windowsForFieldAggregation(beastDiagnostic.plan.windows, "w", {
    sourceKind: "native",
    terrainFollowing: false,
  }).length === 1,
  "actual Beast Flow lattice crosses IPC once and can be reused by cloud presentation",
);
const beastVortexDiagnostic = planNativeStormDetailWindows(
  beastManifestGrid,
  [6_150, -3_650],
  4_000_000,
);
check(
  beastDiagnostic.ok && beastVortexDiagnostic.ok
    && beastDiagnostic.plan.detailIndexRange.lower.slice(0, 2).join(",")
      !== beastVortexDiagnostic.plan.detailIndexRange.lower.slice(0, 2).join(","),
  "Flow and Vortex can retain separate native ROIs when their solved extrema are displaced",
);

const fiftyMetreGrid: GridInfo = {
  nr: 2002,
  nth: 2002,
  nz: 400,
  coordinateSystem: "cartesian",
  drM: 50,
  dyM: 50,
  dzM: 50,
  scalarOriginM: [25, 25, 25],
  validIndexRange: { lower: [1, 1, 0], upper: [2000, 2000, 399] },
};
const nativeStorm = planNativeStormDetailWindows(
  fiftyMetreGrid,
  [50_000, 50_000],
  16_000_000,
);
check(nativeStorm.ok, "50 m storm detail is planned on a 1.6-billion-cell source");
check(
  nativeStorm.ok && nativeStorm.plan.sampledShape.join("×") === "280×280×200",
  "single-pane 50 m detail retains a 14×14×10 km mesocyclone lattice",
);
check(
  nativeStorm.ok && nativeStorm.plan.stride.every((stride) => stride === 1),
  "storm detail reaches the renderer without cell-mean reduction",
);
check(
  nativeStorm.ok && nativeStorm.plan.sourceCells === nativeStorm.plan.sampledCells,
  "native detail source and sampled cell counts agree",
);
check(
  nativeStorm.ok && nativeStorm.plan.windows.length > 1
    && nativeStorm.plan.windows.every((window) => window.stride.every((stride) => stride === 1)),
  "native storm detail is tiled before IPC without reducing any tile",
);

const edgeStorm = planNativeStormDetailWindows(fiftyMetreGrid, [-50_000, 500_000], 16_000_000);
check(
  edgeStorm.ok && edgeStorm.plan.detailIndexRange.lower[0] === 1
    && edgeStorm.plan.detailIndexRange.upper[1] === 2000,
  "storm detail clamps at domain edges without changing its native shape",
);

const thirtyMetreStorm = planNativeStormDetailWindows({
  ...fiftyMetreGrid,
  nr: 4000,
  nth: 4000,
  nz: 600,
  drM: 30,
  dyM: 30,
  dzM: 30,
  scalarOriginM: [15, 15, 15],
  validIndexRange: { lower: [0, 0, 0], upper: [3999, 3999, 599] },
}, [60_000, 60_000], 16_000_000);
check(
  thirtyMetreStorm.ok && thirtyMetreStorm.plan.sampledCells <= 16_000_000
    && thirtyMetreStorm.plan.stride.every((stride) => stride === 1),
  "sub-50 m grids shrink the physical detail box instead of averaging native cells",
);

const oneMetreGrid: GridInfo = {
  nr: 20_000,
  nth: 20_000,
  nz: 10_000,
  coordinateSystem: "cartesian",
  drM: 1,
  dyM: 1,
  dzM: 1,
  scalarOriginM: [0.5, 0.5, 0.5],
};
const oneMetreStorm = planNativeStormDetailWindows(
  oneMetreGrid,
  [10_000, 10_000],
  16_000_000,
);
check(
  oneMetreStorm.ok
    && oneMetreStorm.plan.sourceCells <= 16_000_000
    && oneMetreStorm.plan.sampledCells === oneMetreStorm.plan.sourceCells
    && oneMetreStorm.plan.stride.every((stride) => stride === 1),
  "1 m detail never scans or transfers more source cells than its native renderer budget",
);
check(
  oneMetreStorm.ok
    && oneMetreStorm.plan.detailSpanM[0] >= 100
    && oneMetreStorm.plan.detailSpanM[1] >= 100
    && oneMetreStorm.plan.detailSpanM[2] >= 600,
  "1 m detail spends its native budget on a tornado-depth column and a resolved horizontal neighborhood",
);
check(
  minimumNominalGridSpacingM(oneMetreGrid) === 1
    && supercellAuxiliaryVoxelBudget(16_000_000, 1)
      === 16_000_000,
  "pre-counted diagnostics use the complete pane-admitted scalar working set",
);
const oneMetreAuxiliaryBudget = supercellAuxiliaryVoxelBudget(16_000_000, 1);
const oneMetreVortexDetail = planNativeStormDetailWindows(
  oneMetreGrid,
  [10_000, 10_000],
  oneMetreAuxiliaryBudget,
);
check(
  oneMetreVortexDetail.ok
    && oneMetreVortexDetail.plan.detailSpanM[0] >= 100
    && oneMetreVortexDetail.plan.detailSpanM[1] >= 100
    && oneMetreVortexDetail.plan.detailSpanM[2] >= 600
    && oneMetreVortexDetail.plan.sampledCells <= oneMetreAuxiliaryBudget,
  "1 m vortex diagnostics retain tornado-depth continuity without exceeding their GPU lattice",
);
check(
  supercellAuxiliaryVoxelBudget(4_000_000, 1) === 4_000_000,
  "multi-pane scaling remains the hard memory boundary for an extreme run",
);
check(
  !interactiveFullDomainSourceSafe(oneMetreGrid)
    && interactiveFullDomainSourceSafe(beastManifestGrid),
  "interactive source-scan guard rejects trillion-cell grids without penalizing ordinary runs",
);
check(
  gridHorizontalCenterM(oneMetreGrid)?.join(",") === "10000.5,10000.5",
  "an extreme run can bootstrap a native ROI from metadata without reading a field",
);
const generalExtremeVolume = planInteractiveVolumeWindows(oneMetreGrid, 16_000_000);
check(
  generalExtremeVolume.ok
    && generalExtremeVolume.mode === "native-detail"
    && generalExtremeVolume.plan.sourceCells <= 16_000_000
    && generalExtremeVolume.plan.stride.every((stride) => stride === 1),
  "every general 3D plot replaces a trillion-cell full-domain scan with one bounded native ROI",
);
const generalBeastVolume = planInteractiveVolumeWindows(beastManifestGrid, 16_000_000);
check(
  generalBeastVolume.ok && generalBeastVolume.mode === "full-domain",
  "ordinary 3D sources retain their established full-domain reduced view",
);
const nativeBeastVolume = planInteractiveVolumeWindows(
  beastManifestGrid,
  MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS,
  [12_550, 9_150],
  { preferNativeDetail: true, structuralSupportBoundsM: beastStrongSupport },
);
check(
  nativeBeastVolume.ok
    && nativeBeastVolume.mode === "full-domain"
    && nativeBeastVolume.plan.sampledShape.join("×") === "512×512×160"
    && nativeBeastVolume.plan.stride.join("×") === "1×1×1",
  "a bounded Beast field that fits the native source envelope is not cropped or averaged",
);
check(
  oneMetreStorm.ok
    && nativeDetailRecenteringThresholdM(oneMetreStorm.plan) < 100,
  "1 m native detail recenters before a resolved vortex can leave its small ROI",
);
check(
  nativeStorm.ok
    && nativeDetailRecenteringThresholdM(nativeStorm.plan) === 1_400,
  "mesoscale native detail retains the established stable 1.4 km hysteresis",
);

const slice = planBoundedSliceWindows(grid(10002, 10002, 200), 2, 50);
check(slice.ok, "large horizontal slice is planned");
check(slice.ok && slice.plan.sourceShape[2] === 1, "slice reads one source level only");
check(slice.ok && slice.plan.sampledCells <= PLOTLY_HEATMAP_CELL_CAP, "slice obeys Plotly cell cap");
check(slice.ok && slice.plan.windows.length > 1, "oversized source plane is tiled");
const plane = planBoundedPlaneWindows(grid(10002, 10002, 1));
check(plane.ok && plane.plan.sampledCells <= PLOTLY_HEATMAP_CELL_CAP, "rank-2 planes obey the Plotly cell cap");
check(plane.ok && plane.plan.windows.every((window) => window.origin.length === 2), "rank-2 transport windows do not invent a z axis");
const zoomedPlane = planBoundedPlaneWindows(grid(10002, 10002, 1), PLOTLY_HEATMAP_CELL_CAP, {
  xMin: 100, xMax: 110, yMin: 200, yMax: 210,
});
check(zoomedPlane.ok && zoomedPlane.plan.stride[0] === 1 && zoomedPlane.plan.stride[1] === 1,
  "zoomed plan view reloads its visible region at native stride");
check(zoomedPlane.ok && zoomedPlane.plan.sourceShape[0] < 200 && zoomedPlane.plan.sourceShape[1] < 200,
  "zoomed plan view reads only its bounded physical viewport");
check(zoomedPlane.ok && zoomedPlane.plan.fullSourceShape?.join("x") === "10000x10000x1",
  "viewport plan retains full-plane provenance");
const zoomedSection = planBoundedSliceWindows(grid(10002, 10002, 200), 1, 5000,
  PLOTLY_HEATMAP_CELL_CAP, { xMin: 100, xMax: 110, yMin: 1, yMax: 3 });
check(zoomedSection.ok
    && zoomedSection.plan.sourceShape[0] < 200
    && zoomedSection.plan.sourceShape[1] === 1
    && zoomedSection.plan.sourceShape[2] < 100,
  "zoomed vertical section bounds horizontal distance and height while retaining one slice cell");
const terrainSlice = planBoundedSliceWindows({ ...grid(512, 512, 80), terrainFollowing: true }, 2, 10);
check(terrainSlice.ok, "terrain-following horizontal slices use bounded physical-volume windows");
check(
  fieldWindowAggregation("reflectivity_dbz") === "linear_reflectivity_mean",
  "dBZ windows average linear reflectivity",
);
check(
  fieldWindowAggregation("w") === "cell_mean",
  "ordinary continuous fields retain cell means",
);
check(fieldSupportsBoundedWindows("native"), "published fields support bounded aggregation");
check(
  fieldSupportsBoundedWindows("studio_derived"),
  "built-in derived fields evaluate exact chunks before display aggregation",
);
check(fieldSupportsBoundedWindows("user_expression"),
  "expression fields evaluate exact chunks before display aggregation");
check(fieldSupportsBoundedWindows("external_provider"),
  "external providers enter the bounded path where chunk-aware capability is enforced");

const stretchedWindow: NpyN = {
  shape: [2, 1, 4],
  data: new Float32Array(8),
  windowGeometry: {
    aggregation: "cell_mean",
    weighting: "uniform_cell_volume",
    sampleLocation: "block_center",
    centroidsM: null,
    sourceShape: [2, 3, 4],
    origin: [0, 1, 0],
    extent: [2, 1, 4],
    stride: [1, 1, 1],
    sourceFormat: "npy",
  },
};
const stretchedWindowGrid = gridForWindowedField(stretchedWindow, {
  ...grid(2, 3, 4),
  dzM: undefined,
  scalarCoordinatesM: [[0, 100], [0, 200, 400], [0, 40, 95, 170]],
});
check(
  stretchedWindowGrid?.scalarCoordinatesM?.[2].join(",") === "0,40,95,170",
  "bounded vertical sections preserve exact stretched source coordinates",
);

const exact = planExactValidDomainWindows(grid(10, 8, 6), 3, {
  lower: [1, 2, 1], upper: [4, 4, 3],
}, 32);
check(exact.ok, "exact comparison region is tileable under a tiny transport budget");
check(exact.ok && exact.plan.shape.join("×") === "4×3×3", "exact plan retains the inclusive region shape");
check(exact.ok && exact.plan.windows.length > 1, "exact plan streams more than one tile when required");
check(exact.ok && exact.plan.windows.every((window) =>
  window.stride.every((stride) => stride === 1)
  && window.aggregation === "cell_mean"
), "exact metric tiles inspect every source cell without reduction");
const exactTerrain = planExactValidDomainWindows({
  ...grid(10, 8, 6), terrainFollowing: true,
}, 3, undefined, 36 * 20);
check(exactTerrain.ok && exactTerrain.plan.windows.every((window) =>
  window.maxOutputBytes === nativeFieldWindowBodyBytes(
    window.extent.reduce((cells, count) => cells * count, 1), 32,
  )
), "exact terrain tiles budget values, x/y/z/volume geometry, and alignment padding");
const singleTerrainCell = {
  ...grid(3, 3, 1), terrainFollowing: true,
};
check(
  !planExactValidDomainWindows(singleTerrainCell, 3, undefined, 36).ok,
  "the old 36-byte single-cell terrain budget rejects because it omitted alignment",
);
const paddedTerrainCell = planExactValidDomainWindows(singleTerrainCell, 3, undefined, 40);
check(
  paddedTerrainCell.ok && paddedTerrainCell.plan.windows[0]?.maxOutputBytes === 40,
  "a single terrain cell is admitted only with its exact 40-byte schema-v5 body",
);

const exactLine = planExactLineWindows(grid(10, 8, 6), 2, [3, 2, 0], 8);
check(exactLine.ok, "exact profile line is tileable under a tiny transport budget");
check(exactLine.ok && exactLine.plan.shape.join("×") === "1×1×6", "profile plan reads one valid-domain column");
check(exactLine.ok && exactLine.plan.windows.length > 1, "long profile lines stream in exact tiles when required");
check(exactLine.ok && exactLine.plan.windows.every((window) =>
  window.stride.every((stride) => stride === 1)
), "profile line tiles never subsample the scientific series");
check(exactLine.ok && exactLine.plan.windows.every((window) =>
  window.origin[0] === 4 && window.origin[1] === 3
), "profile local indices map through the valid-domain halo offset");
check(!planExactLineWindows(grid(10, 8, 6), 2, [8, 2, 0]).ok,
  "profile lines outside the valid domain are refused before field I/O");

const derivedMean = aggregateDerivedWindow(
  { shape: [2], data: new Float32Array([1, 81]) },
  { aggregation: "cell_mean", origin: [0], extent: [2], stride: [2] },
  [2],
);
check(derivedMean.data[0] === 41, "nonlinear derived values are reduced after evaluation");
const extremaSource: NpyN = { shape: [4], data: new Float32Array([3, -8, 5, -4]) };
const reduceDerived = (aggregation: "block_max" | "block_min" | "magnitude_extreme") =>
  aggregateDerivedWindow(
    extremaSource,
    { aggregation, origin: [0], extent: [4], stride: [4] },
    [4],
  ).data[0];
check(reduceDerived("block_max") === 5, "derived block-max retains a compact positive peak");
check(reduceDerived("block_min") === -8, "derived block-min retains a compact negative peak");
check(reduceDerived("magnitude_extreme") === -8, "derived signed LOD retains the strongest sense");
const pointDerived = aggregateDerivedWindow(
  extremaSource,
  { aggregation: "point_sample", origin: [0], extent: [4], stride: [2] },
  [4],
);
check([...pointDerived.data].join(",") === "3,5", "derived sparse windows retain exact point samples");
const terrainDerived = aggregateDerivedWindow({
  shape: [2],
  data: new Float32Array([10, 20]),
  windowGeometry: {
    weighting: "physical_cell_volume",
    sampleLocation: "physical_volume_centroid",
    centroidsM: {
      x: new Float64Array([50, 50]),
      y: new Float64Array([50, 50]),
      z: new Float64Array([5, 20]),
    },
    volumesM3: new Float64Array([1, 3]),
  },
}, { aggregation: "cell_mean", origin: [0], extent: [2], stride: [2] }, [2]);
check(terrainDerived.data[0] === 17.5, "derived terrain means retain physical-volume weighting");
check(terrainDerived.windowGeometry?.volumesM3?.[0] === 4, "derived terrain blocks retain exact volume");
check(terrainDerived.windowGeometry?.centroidsM?.z[0] === 16.25, "derived terrain centroids retain volume moments");

const windows = [
  { aggregation: "cell_mean" as const, origin: [1, 1, 0], extent: [2, 4, 1], stride: [2, 2, 1] },
  { aggregation: "cell_mean" as const, origin: [3, 1, 0], extent: [2, 4, 1], stride: [2, 2, 1] },
];
const geometry = (origin: number[]): NpyN["windowGeometry"] => ({
  weighting: "uniform_cell_volume",
  sampleLocation: "block_center",
  centroidsM: null,
  sourceShape: [6, 6, 1],
  origin,
  extent: [2, 4, 1],
  stride: [2, 2, 1],
});
const assembled = assembleWindowedField([
  { shape: [1, 2, 1], data: new Float32Array([1, 2]), windowGeometry: geometry([1, 1, 0]) },
  { shape: [1, 2, 1], data: new Float32Array([3, 4]), windowGeometry: geometry([3, 1, 0]) },
], windows);
check(assembled.shape.join("×") === "2×2×1", "tiles assemble to the planned shape");
check([...assembled.data].join(",") === "1,2,3,4", "tile values retain C-order placement");

const adjusted = gridForWindowedField(assembled, grid(6, 6, 1));
check(adjusted?.validIndexRange?.lower.join(",") === "0,0,0", "window grid is local and halo-free");
check(adjusted?.scalarCoordinatesM?.[0].join(",") === "200,400", "block centres retain source origin");
check(adjusted?.scalarCoordinatesM?.[1].join(",") === "200,400", "block centres retain cross-axis origin");

const terrainWindow: NpyN = {
  shape: [2, 2, 1],
  data: new Float32Array(4),
  windowGeometry: {
    aggregation: "cell_mean",
    weighting: "physical_cell_volume",
    sampleLocation: "physical_volume_centroid",
    sourceShape: [2, 2, 1],
    origin: [0, 0, 0],
    extent: [2, 2, 1],
    stride: [1, 1, 1],
    centroidsM: {
      x: new Float64Array([100, 100, 200, 200]),
      y: new Float64Array([300, 400, 300, 400]),
      z: new Float64Array([50, 70, 60, 80]),
    },
    volumesM3: new Float64Array([1, 1, 1, 1]),
  },
};
const terrainGrid = gridForWindowedField(terrainWindow, { ...grid(2, 2, 1), terrainFollowing: true });
check(terrainGrid?.scalarCoordinatesM?.[0].join(",") === "100,200", "terrain planes use exact separable x centroids");
check(terrainGrid?.scalarCoordinatesM?.[1].join(",") === "300,400", "terrain planes use exact separable y centroids");

console.log(`\nlarge-domain planning tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  console.log(failures.join("\n"));
  process.exit(1);
}
