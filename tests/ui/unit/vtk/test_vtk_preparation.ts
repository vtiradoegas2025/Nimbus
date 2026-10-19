import { readFileSync } from "node:fs";
import { mock } from "node:test";
import {
  cartesianGridToVtkScalars,
  cartesianGridToVtkVectors,
  fieldToCartesianGrid,
  refineCartesianGridLinear,
  vectorFieldToCartesianGrid,
} from "../../../../ui/src/lib/science/cartesianInterp.ts";
import {
  prepareVtkRequest,
  dominantVortexCenterline,
  qCriterionFromGradient,
  lambda2FromGradient,
  swirlingStrengthFromGradient,
  scalarBoundsAbove,
  scalarMagnitudeBoundsAbove,
  scalarSignedPeak,
  vtkPreparationTransferables,
  velocityQCriterion,
  type VtkPreparationRequest,
  type VtkPreparationResult,
} from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import {
  isPreparationAbort,
  PreparationCoordinator,
  SharedPreparationCoordinator,
  prepareVtkInWorker,
  vtkPreparationIdentity,
  type PreparationWorker,
} from "../../../../ui/src/lib/vtk/vtkPreparationClient.ts";
import {
  combinedContourBudgetViolation,
  contourBudgetViolation,
  measureContourOutput,
} from "../../../../ui/src/lib/vtk/vtkContourBudget.ts";
import {
  CLOUD_OPACITY_REFERENCE_M,
  CLOUD_PEAK_OPACITY,
  CLOUD_DIAGNOSTIC_REVEAL_SCALE,
  PRECIP_PEAK_OPACITY,
  cloudDensityForDiagnostics,
  cloudOpacityPoints,
  scaledCloudPeakOpacity,
  scaledPrecipPeakOpacity,
} from "../../../../ui/src/lib/plot/supercellRenderQuality.ts";
import {
  addCloudPresentationDetail,
  cloudCondensateCentroid,
} from "../../../../ui/src/lib/plot/cloudPresentationDetail.ts";
import { validatePairedCloudDetail } from "../../../../ui/src/lib/plot/cloudDetailValidation.ts";
import {
  LOCAL_VTK_CONTOUR_VOXEL_CAP,
  LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP,
  localVtkContourBudgetError,
  localVtkStreamlineBudgetError,
  vtkGeometryExecutionMode,
} from "../../../../ui/src/lib/vtk/vtkWorkerRuntime.ts";
import {
  handoffVtkExtractionDataset,
  isSharedVtkDataset,
  recordVtkDatasetHandoff,
  resetVtkDatasetHandoffMetrics,
  sharePreparedVtkResult,
  transferablePreparedBuffers,
  uploadPreparedVtkValues,
  vtkDatasetHandoffMetrics,
} from "../../../../ui/src/lib/vtk/vtkDatasetHandoff.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else { failed++; failures.push(`FAIL ${label}`); }
}

{
  const coarse = {
    shape: [2, 2, 2] as [number, number, number],
    spacing: [500, 750, 1000] as [number, number, number],
    origin: [10, 20, 30] as [number, number, number],
    data: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]),
  };
  const refined = refineCartesianGridLinear(coarse, {
    targetSpacingM: 250,
    maxFactor: 4,
    maxVoxels: 1_000,
  });
  check(JSON.stringify(refined.factors) === JSON.stringify([2, 3, 4]), "coarse display refinement is selected independently per axis");
  check(JSON.stringify(refined.grid.shape) === JSON.stringify([3, 4, 5]), "refinement subdivides intervals while retaining both endpoints");
  check(JSON.stringify(refined.grid.spacing) === JSON.stringify([250, 250, 250]), "refinement reaches the target display spacing");
  check(refined.grid.origin === coarse.origin, "refinement preserves the exact world origin");
  check(refined.grid.data[0] === 0 && refined.grid.data.at(-1) === 7, "refinement preserves source endpoint samples");
  check(refined.grid.data.every((value) => value >= 0 && value <= 7), "trilinear reconstruction creates no new scalar extrema");

  const limited = refineCartesianGridLinear(coarse, {
    targetSpacingM: 100,
    maxFactor: 4,
    maxVoxels: 18,
  });
  check(limited.grid.data.length <= 18, "refinement respects its hard voxel ceiling");

  const fine = { ...coarse, spacing: [100, 100, 100] as [number, number, number] };
  const unchanged = refineCartesianGridLinear(fine);
  check(unchanged.grid === fine && unchanged.grid.data === fine.data, "fine grids alias their original display buffer");

  check(CLOUD_OPACITY_REFERENCE_M === 1_000, "cloud extinction uses a fixed one-kilometre world reference");
  check(CLOUD_PEAK_OPACITY === 0.22, "volume-primary cloud transfer preserves fine-grid accumulated opacity");
  check(PRECIP_PEAK_OPACITY === 0.5, "precipitation keeps its denser unit-scale transfer anchor");
  check(
    scaledCloudPeakOpacity(1) === CLOUD_PEAK_OPACITY
      && scaledCloudPeakOpacity(2.5) < CLOUD_PEAK_OPACITY * 2.5
      && scaledCloudPeakOpacity(2.5) < 1,
    "cloud density uses optical-depth scaling without a high-end opaque clamp",
  );
  check(
    scaledPrecipPeakOpacity(0.75) > 0.39
      && scaledPrecipPeakOpacity(0.75) < 0.42
      && scaledPrecipPeakOpacity(1.5) < 0.7,
    "precipitation keeps its default body while preserving structure at maximum density",
  );
  check(
    CLOUD_DIAGNOSTIC_REVEAL_SCALE === 0.35
      && cloudDensityForDiagnostics(1, true) === 0.35
      && cloudDensityForDiagnostics(1, false) === 1,
    "diagnostic reveal reduces display opacity without changing the selected cloud density",
  );
  const offEdge = cloudOpacityPoints(0.0001, 0.0015, CLOUD_PEAK_OPACITY, "off");
  const subtleEdge = cloudOpacityPoints(0.0001, 0.0015, CLOUD_PEAK_OPACITY, "subtle");
  const fullEdge = cloudOpacityPoints(0.0001, 0.0015, CLOUD_PEAK_OPACITY, "full");
  check(
    fullEdge[6].opacity > subtleEdge[6].opacity && subtleEdge[6].opacity > offEdge[6].opacity,
    "cloud edge modes strengthen the same smooth volume transfer without adding a polygon shell",
  );
  check(
    subtleEdge[2].opacity < CLOUD_PEAK_OPACITY * 0.01 &&
      fullEdge[2].opacity < CLOUD_PEAK_OPACITY * 0.01,
    "tenuous condensate remains optically thin instead of accumulating into path haze",
  );
  check(
    fullEdge.every((point, index) => index === 0 || point.scalar > fullEdge[index - 1].scalar),
    "cloud opacity nodes remain strictly increasing for vtk transfer textures",
  );
  const slopes = subtleEdge.slice(1).map((point, index) => (
    (point.opacity - subtleEdge[index].opacity)
      / (point.scalar - subtleEdge[index].scalar)
  ));
  check(
    slopes.every(Number.isFinite)
      && Math.abs(slopes[1] - slopes[0]) < CLOUD_PEAK_OPACITY * 400,
    "cloud opacity leaves the exact floor without an illuminated transfer-function shoulder",
  );
  check(
    JSON.stringify(scalarBoundsAbove(coarse, 6)) === JSON.stringify([10, 510, 20, 770, 30, 1030]),
    "dense-cloud focus bounds preserve physical coordinates and pad to grid edges",
  );
  check(scalarBoundsAbove(coarse, 8) === null, "focus bounds are absent when the physical threshold is not reached");
  check(
    JSON.stringify(scalarMagnitudeBoundsAbove(coarse, 6))
      === JSON.stringify([10, 510, 20, 770, 30, 1030]),
    "absolute feature bounds include robust support of either sign",
  );
  const lowLevelPeak = scalarSignedPeak(coarse, 1, [0, 500]);
  check(
    lowLevelPeak?.value === 6 && lowLevelPeak.position[2] === 30,
    "signed feature location can be constrained to a physical low-level height band",
  );

  const detail = addCloudPresentationDetail(coarse, refined.grid, refined.factors, 42);
  const repeat = addCloudPresentationDetail(coarse, refined.grid, refined.factors, 42);
  const alternate = addCloudPresentationDetail(coarse, refined.grid, refined.factors, 43);
  check(detail.metadata?.supportConstrained === true, "presentation cloud detail declares its resolved-support constraint");
  check(detail.metadata?.coherence === "storm-relative", "presentation cloud detail declares its translational coherence frame");
  check(equal(detail.grid.data, repeat.grid.data), "presentation cloud detail is deterministic for one run seed");
  check(!equal(detail.grid.data, alternate.grid.data), "presentation cloud detail varies across stable run seeds");
  check(detail.grid.data[0] === 0, "presentation cloud detail never creates condensate outside zero resolved support");
  const baseSum = refined.grid.data.reduce((sum, value) => sum + value, 0);
  const detailSum = detail.grid.data.reduce((sum, value) => sum + value, 0);
  check(Math.abs(baseSum - detailSum) < 1e-4, "presentation cloud detail preserves total condensate mass");

  const flowData = new Float32Array(coarse.data.length * 3);
  const [, flowNy, flowNz] = coarse.shape;
  for (let x = 0; x < coarse.shape[0]; x++) {
    for (let y = 0; y < flowNy; y++) {
      for (let z = 0; z < flowNz; z++) {
        const base = ((x * flowNy + y) * flowNz + z) * 3;
        flowData[base] = y * 3;
        flowData[base + 2] = z;
      }
    }
  }
  const flow = { ...coarse, data: flowData };
  const uniformFlowData = new Float32Array(coarse.data.length * 3);
  for (let i = 0; i < uniformFlowData.length; i += 3) {
    uniformFlowData[i] = 12;
    uniformFlowData[i + 1] = -4;
    uniformFlowData[i + 2] = 2;
  }
  const uniformAdvected = addCloudPresentationDetail(
    coarse, refined.grid, refined.factors, 42,
    { grid: { ...coarse, data: uniformFlowData }, simulationTimeS: 45 },
  );
  check(
    equal(detail.grid.data, uniformAdvected.grid.data),
    "bulk uniform wind is removed exactly from storm-relative detail",
  );
  const advected = addCloudPresentationDetail(
    coarse, refined.grid, refined.factors, 42,
    { grid: flow, simulationTimeS: 30 },
  );
  check(advected.metadata?.coherence === "resolved-flow", "presentation detail discloses resolved-flow deformation");
  check(!equal(detail.grid.data, advected.grid.data), "resolved velocity anomalies deform detail with simulation time");
  check(
    Math.abs(advected.grid.data.reduce((sum, value) => sum + value, 0) - baseSum) < 1e-4,
    "flow-deformed detail preserves the same total condensate mass contract",
  );
  const beforeBoundary = addCloudPresentationDetail(
    coarse, refined.grid, refined.factors, 42,
    { grid: flow, simulationTimeS: 59.999 },
  );
  const afterBoundary = addCloudPresentationDetail(
    coarse, refined.grid, refined.factors, 42,
    { grid: flow, simulationTimeS: 60.001 },
  );
  let boundaryDelta = 0;
  for (let i = 0; i < beforeBoundary.grid.data.length; i++) {
    boundaryDelta = Math.max(
      boundaryDelta,
      Math.abs(beforeBoundary.grid.data[i] - afterBoundary.grid.data[i]),
    );
  }
  check(boundaryDelta < 1e-3, "cross-faded advection epochs remain continuous at their boundary");
}

{
  const shape: [number, number, number] = [5, 3, 3];
  const spacing: [number, number, number] = [600, 600, 600];
  const origin: [number, number, number] = [100, 200, 300];
  const translatedSource = (cloudX: number) => {
    const data = new Float32Array(shape[0] * shape[1] * shape[2]);
    for (let y = 0; y < shape[1]; y++) {
      for (let z = 0; z < shape[2]; z++) {
        data[(cloudX * shape[1] + y) * shape[2] + z] = 1 + y * 0.2 + z * 0.1;
      }
    }
    return { shape, spacing, origin, data };
  };
  const firstSource = translatedSource(1);
  const secondSource = translatedSource(2);
  const firstRefined = refineCartesianGridLinear(firstSource, {
    targetSpacingM: 300,
    maxFactor: 2,
    maxVoxels: 10_000,
  });
  const secondRefined = refineCartesianGridLinear(secondSource, {
    targetSpacingM: 300,
    maxFactor: 2,
    maxVoxels: 10_000,
  });
  const firstDetail = addCloudPresentationDetail(
    firstSource,
    firstRefined.grid,
    firstRefined.factors,
    77,
  );
  const secondDetail = addCloudPresentationDetail(
    secondSource,
    secondRefined.grid,
    secondRefined.factors,
    77,
  );
  const firstAnchor = cloudCondensateCentroid(firstSource);
  const secondAnchor = cloudCondensateCentroid(secondSource);
  check(
    Math.abs(secondAnchor[0] - firstAnchor[0] - spacing[0]) < 1e-5,
    "resolved condensate centroid follows an exact one-cell storm translation",
  );
  const [nx, ny, nz] = firstRefined.grid.shape;
  const shift = firstRefined.factors[0];
  let translatedMatch = true;
  for (let x = 0; x < nx - shift; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        const a = firstDetail.grid.data[(x * ny + y) * nz + z];
        const b = secondDetail.grid.data[((x + shift) * ny + y) * nz + z];
        if (Math.abs(a - b) > 1e-5) translatedMatch = false;
      }
    }
  }
  check(translatedMatch, "storm-relative detail translates with resolved support without flicker");
}

{
  // A registered fine/coarse pair with the same resolved envelope. The fine
  // reference contains independent smooth subcell variability; validation is
  // distribution/spectrum based because its phase is unknowable from coarse
  // samples and must never be advertised as reconstructed truth.
  const sourceShape: [number, number, number] = [6, 5, 4];
  const source = {
    shape: sourceShape,
    spacing: [600, 600, 600] as [number, number, number],
    origin: [0, 0, 0] as [number, number, number],
    data: new Float32Array(sourceShape[0] * sourceShape[1] * sourceShape[2]),
  };
  for (let x = 0; x < sourceShape[0]; x++) {
    for (let y = 0; y < sourceShape[1]; y++) {
      for (let z = 0; z < sourceShape[2]; z++) {
        const dx = (x - 2.5) / 2.3;
        const dy = (y - 2) / 1.9;
        const dz = (z - 1.4) / 1.5;
        const envelope = Math.exp(-(dx * dx + dy * dy + dz * dz));
        source.data[(x * sourceShape[1] + y) * sourceShape[2] + z] =
          envelope > 0.12 ? envelope : 0;
      }
    }
  }
  const refined = refineCartesianGridLinear(source, {
    targetSpacingM: 300,
    maxFactor: 2,
    maxVoxels: 100_000,
  });
  const presentation = addCloudPresentationDetail(source, refined.grid, refined.factors, 1201);
  const fineReferenceData = new Float32Array(refined.grid.data.length);
  const [nx, ny, nz] = refined.grid.shape;
  let baselineMass = 0;
  let referenceMass = 0;
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        const index = (x * ny + y) * nz + z;
        const base = refined.grid.data[index];
        baselineMass += base;
        if (!(base > 0)) continue;
        const subcell =
          0.55 * Math.sin(x * 1.37 + y * 0.21) +
          0.35 * Math.cos(y * 1.11 - z * 0.43) +
          0.25 * Math.sin(z * 1.29 + x * 0.31);
        const value = base * Math.exp(0.72 * subcell);
        fineReferenceData[index] = value;
        referenceMass += value;
      }
    }
  }
  const referenceNormalization = baselineMass / referenceMass;
  for (let i = 0; i < fineReferenceData.length; i++) {
    fineReferenceData[i] *= referenceNormalization;
  }
  const validation = validatePairedCloudDetail(
    { ...refined.grid, data: fineReferenceData },
    refined.grid,
    presentation.grid,
  );
  check(validation.supportLeakCount === 0, "paired detail validation observes no support leakage");
  check(Math.abs(validation.relativeMassBias) < 1e-6, "paired detail validation observes conserved condensate mass");
  check(
    validation.variabilityGapAfter < validation.variabilityGapBefore,
    "presentation detail restores coarse-grid variability toward an independent fine reference",
  );
  check(
    validation.gradientGapAfter < validation.gradientGapBefore,
    "presentation detail restores interior gradient energy toward an independent fine reference",
  );
}
function equal(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
}

const shape: [number, number, number] = [3, 2, 4];
const field = {
  shape,
  data: Float32Array.from({ length: 24 }, (_, index) => index === 7 ? Number.NaN : index - 5),
};
const grid = { drM: 100, dyM: 200, dzM: 50, scalarOriginM: [10, 20, 25] as [number, number, number] };

{
  const retained = Float32Array.from([1, 2, 3, 4]);
  const prepared = {
    shape: [2, 2, 1] as [number, number, number],
    spacing: [1, 1, 1] as [number, number, number],
    origin: [0, 0, 0] as [number, number, number],
    values: retained,
    sourceNonFiniteCount: 0,
    samplingProvenance: {
      sourceShape: [2, 2, 1] as [number, number, number],
      sourceSpacing: [1, 1, 1] as [number, number, number],
      sampledShape: [2, 2, 1] as [number, number, number],
      sampledSpacing: [1, 1, 1] as [number, number, number],
      displayShape: [2, 2, 1] as [number, number, number],
      displaySpacing: [1, 1, 1] as [number, number, number],
      sampling: "native" as const,
      reconstruction: "none" as const,
      reconstructionFactors: [1, 1, 1] as [number, number, number],
      coordinateResampling: "none" as const,
      coordinateResamplingAxes: [],
    },
  };
  const renderOnlyValues = Float32Array.from([9, 10]);
  const renderOnly = {
    ...prepared,
    shape: [2, 1, 1] as [number, number, number],
    values: renderOnlyValues,
  };
  const renderOnlyReport = sharePreparedVtkResult(renderOnly, false);
  check(
    renderOnlyReport.mode === "render-only-transfer"
      && renderOnlyReport.copiedBytes === 0
      && renderOnly.values === renderOnlyValues,
    "render-only preparation remains a zero-copy worker-to-UI transfer",
  );
  const preparationReport = sharePreparedVtkResult(prepared, true);
  check(
    preparationReport.mode === "shared-zero-copy"
      && isSharedVtkDataset(prepared.values)
      && retained.byteLength === 16
      && prepared.values.join(",") === "1,2,3,4",
    "prepared worker output is promoted once to shared immutable storage without detaching its source",
  );
  check(
    transferablePreparedBuffers(prepared).length === 0,
    "shared prepared buffers are cloned by reference and never placed in a transfer list",
  );
  const sharedExtraction = handoffVtkExtractionDataset(prepared.values);
  check(
    sharedExtraction.values === prepared.values
      && sharedExtraction.transfer.length === 0
      && sharedExtraction.report.copiedBytes === 0,
    "shared extraction handoff reuses the exact prepared grid with zero copies",
  );
  const secondPaneReport = sharePreparedVtkResult(prepared, true);
  check(
    secondPaneReport.copiedBytes === 0 && secondPaneReport.mode === "shared-zero-copy",
    "a second pane sharing the prepared SAB adds no preparation copy",
  );
  const fallbackValues = Float32Array.from([5, 6, 7, 8]);
  const transferableExtraction = handoffVtkExtractionDataset(fallbackValues);
  recordVtkDatasetHandoff(transferableExtraction.report);
  check(
    transferableExtraction.values !== fallbackValues
      && transferableExtraction.transfer.length === 1
      && fallbackValues.join(",") === "5,6,7,8"
      && transferableExtraction.report.sourceRetained === true
      && transferableExtraction.report.copiedBytes === fallbackValues.byteLength,
    "non-isolated extraction makes exactly one transferable copy while retaining prepared data",
  );
  let uploadAttempts = 0;
  let uploaded: Float32Array | null = null;
  uploadPreparedVtkValues(prepared.values, (values) => {
    uploadAttempts += 1;
    if (uploadAttempts === 1) throw new Error("synthetic VTK SAB refusal");
    uploaded = values;
  });
  check(
    uploadAttempts === 2
      && uploaded != null
      && !isSharedVtkDataset(uploaded)
      && prepared.values.join(",") === "1,2,3,4",
    "VTK upload refusal retries one AB copy without detaching the shared prepared source",
  );
  const sharedSpans = new Float32Array(new SharedArrayBuffer(8 * Float32Array.BYTES_PER_ELEMENT));
  sharedSpans.set([1, 2, 3, 4, 101, 102, 103, 104]);
  const fallbackFor = (source: Float32Array): Float32Array => {
    let result: Float32Array | undefined;
    uploadPreparedVtkValues(source, values => {
      if (isSharedVtkDataset(values)) throw new Error("synthetic shared span refusal");
      result = values;
    });
    return result!;
  };
  const firstSpan = fallbackFor(sharedSpans.subarray(0, 4));
  const secondSpan = fallbackFor(sharedSpans.subarray(4));
  const shorterSpan = fallbackFor(sharedSpans.subarray(0, 2));
  check(firstSpan.join(",") === "1,2,3,4" && secondSpan.join(",") === "101,102,103,104"
    && shorterSpan.join(",") === "1,2", "shared upload fallback distinguishes offsets and lengths within one buffer");
  check(fallbackFor(new Float32Array(sharedSpans.buffer, 16, 4)) === secondSpan,
    "equivalent immutable spans reuse exactly one fallback copy");
  resetVtkDatasetHandoffMetrics();
  // Instrumentation is intentionally exported and mirrored on window in the
  // browser so mounted qualification can measure copy behavior.
  const beforeHandoff = vtkDatasetHandoffMetrics();
  check(
    beforeHandoff.logicalBytes === 0
      && beforeHandoff.copiedBytes === 0
      && beforeHandoff.sourceRetainedDatasets === 0
      && beforeHandoff.downgradeReasons.length === 0,
    "dataset handoff instrumentation resets deterministically",
  );

  check(
    qCriterionFromGradient([0, -2, 0, 2, 0, 0, 0, 0, 0]) === 4,
    "Q criterion is positive for canonical solid-body rotation",
  );
  check(
    qCriterionFromGradient([3, 0, 0, 0, -3, 0, 0, 0, 0]) === -9,
    "Q criterion is negative for canonical pure strain",
  );
  check(
    qCriterionFromGradient([0, 4, 0, 0, 0, 0, 0, 0, 0]) === 0,
    "Q criterion is zero for simple shear with balanced strain and rotation",
  );
  check(
    lambda2FromGradient([0, -2, 0, 2, 0, 0, 0, 0, 0]) === -4,
    "lambda-2 is negative for canonical solid-body rotation",
  );
  check(
    lambda2FromGradient([3, 0, 0, 0, -3, 0, 0, 0, 0]) === 9,
    "lambda-2 rejects canonical planar pure strain",
  );
  check(
    Math.abs(swirlingStrengthFromGradient([0, -2, 0, 2, 0, 0, 0, 0, 0]) - 2) < 1e-12,
    "swirling strength recovers the solid-rotation angular rate",
  );
  check(
    swirlingStrengthFromGradient([0, 4, 0, 0, 0, 0, 0, 0, 0]) === 0,
    "swirling strength rejects simple shear",
  );
  const vectorShape: [number, number, number] = [3, 3, 3];
  const vectors = {
    shape: vectorShape,
    spacing: [100, 100, 100] as [number, number, number],
    origin: [0, 0, 0] as [number, number, number],
    data: new Float32Array(3 * 3 * 3 * 3),
  };
  for (let x = 0; x < 3; x++) {
    for (let y = 0; y < 3; y++) {
      for (let z = 0; z < 3; z++) {
        const tuple = ((x * 3 + y) * 3 + z) * 3;
        vectors.data[tuple] = -2 * y * 100;
        vectors.data[tuple + 1] = 2 * x * 100;
      }
    }
  }
  const q = velocityQCriterion(vectors);
  check(q.data.every((value) => Math.abs(value - 4) < 1e-5), "worker Q field retains full 3D rotational invariance through boundaries");

  const ridgeGrid = {
    shape: [7, 7, 5] as [number, number, number],
    spacing: [100, 100, 250] as [number, number, number],
    origin: [50, 50, 125] as [number, number, number],
    data: new Float32Array(7 * 7 * 5),
  };
  for (let z = 0; z < 5; z++) {
    const x = 2 + Math.floor(z / 2);
    ridgeGrid.data[(x * 7 + 3) * 5 + z] = 10 - z * 0.5;
  }
  const ridge = dominantVortexCenterline(ridgeGrid, "q-criterion");
  check(ridge.length === 15, "dominant vortex ridge retains one connected point per qualified level");
  check(ridge[2] === 125 && ridge.at(-1) === 1125,
    "dominant vortex ridge uses physical vertical coordinates");
  check(ridge[0] < ridge[ridge.length - 3], "dominant vortex ridge follows a tilted core");
  ridgeGrid.data.fill(0);
  check(dominantVortexCenterline(ridgeGrid, "lambda-2").length === 0,
    "ridge extraction refuses a field with no rotation-dominated samples");
}

{
  const expectedCart = fieldToCartesianGrid(field, grid, { preserveMissing: true });
  const expected = cartesianGridToVtkScalars(expectedCart);
  const prepared = prepareVtkRequest({ kind: "scalar", field, grid });
  check(prepared.values.every((value, index) => value === (Number.isFinite(expected[index]) ? expected[index] : 0)), "worker scalar preparation preserves finite values in VTK order");
  check(prepared.validity?.every((value, index) => value === (Number.isFinite(expected[index]) ? 255 : 0)) === true, "missing scalar support has its own mask instead of a physical fill value");
  check(JSON.stringify(prepared.shape) === JSON.stringify(expectedCart.shape), "worker scalar shape matches synchronous baseline");
  check(JSON.stringify(prepared.spacing) === JSON.stringify(expectedCart.spacing), "worker scalar spacing matches synchronous baseline");
  check(JSON.stringify(prepared.origin) === JSON.stringify(expectedCart.origin), "worker scalar origin matches synchronous baseline");
  check(prepared.samplingProvenance.sampling === "native", "worker records native scalar sampling provenance");
  check(prepared.samplingProvenance.coordinateResampling === "none", "uniform grids disclose no coordinate resampling");
}

{
  const signedField = {
    shape: [3, 2, 2] as [number, number, number],
    data: Float32Array.from([0, 1, -2, 0, 3, 0, 0, -9, 7, 0, 0, 0]),
  };
  const positive = prepareVtkRequest({
    kind: "scalar",
    field: signedField,
    grid: { drM: 50, dyM: 100, dzM: 25, scalarOriginM: [25, 50, 12.5] },
    featureSign: 1,
  });
  const negative = prepareVtkRequest({
    kind: "scalar",
    field: signedField,
    grid: { drM: 50, dyM: 100, dzM: 25, scalarOriginM: [25, 50, 12.5] },
    featureSign: -1,
  });
  check(
    positive.featurePeakPosition?.join(",") === "125,50,12.5",
    "worker locates the strongest positive vortex sample in physical coordinates",
  );
  check(
    negative.featurePeakPosition?.join(",") === "75,150,37.5",
    "worker locates the strongest negative pressure-well sample in physical coordinates",
  );
}

{
  const coarseField = {
    shape: [2, 2, 2] as [number, number, number],
    data: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]),
  };
  const prepared = prepareVtkRequest({
    kind: "scalar",
    field: coarseField,
    grid: { drM: 1_000, dyM: 1_000, dzM: 1_000 },
    maxVoxels: 1_000,
    refineForDisplay: true,
  });
  check(
    JSON.stringify(prepared.shape) === JSON.stringify([5, 5, 5]),
    "coarse scalar volumes use the densest 250 m display lattice inside their voxel ceiling",
  );
  check(
    prepared.samplingProvenance.reconstruction === "piecewise-trilinear",
    "coarse scalar volume reconstruction is disclosed",
  );
  check(
    prepared.values.every((value) => value >= 0 && value <= 7),
    "coarse scalar display reconstruction cannot invent extrema",
  );
}

{
  const shape: [number, number, number] = [4, 4, 4];
  const structuralField = {
    shape,
    data: Float32Array.from({ length: 64 }, (_, index) => index / 63),
  };
  const prepared = prepareVtkRequest({
    kind: "scalar",
    field: structuralField,
    grid: { drM: 100, dyM: 100, dzM: 100, scalarOriginM: [50, 50, 50] },
    structuralSupportRangeFraction: 0.9,
  });
  check(
    prepared.structuralSupportBounds != null
      && prepared.structuralSupportBounds[1] === 350
      && prepared.structuralSupportBounds[5] === 350,
    "worker preparation returns physical upper-tail support bounds for adaptive Volume detail",
  );
}

{
  const stretchedField = {
    shape: [2, 2, 3] as [number, number, number],
    data: Float32Array.from([0, 100, 300, 0, 100, 300, 0, 100, 300, 0, 100, 300]),
  };
  const stretched = prepareVtkRequest({
    kind: "scalar",
    field: stretchedField,
    grid: {
      drM: 1_000,
      dyM: 1_000,
      scalarCoordinatesM: [[0, 1_000], [0, 1_000], [0, 100, 300]],
    },
  });
  check(stretched.spacing[2] === 150, "worker preserves stretched-z endpoint span on its VTK lattice");
  check(stretched.samplingProvenance.coordinateResampling === "linear-to-uniform", "worker provenance discloses coordinate-linear resampling");
  check(stretched.samplingProvenance.coordinateResamplingAxes.join(",") === "z", "worker provenance identifies stretched z as the resampled axis");
}

{
  const coordinates = [[0, 100, 300], [0, 200, 600], [0, 50, 150]] as [number[], number[], number[]];
  const stretchedField = {
    shape: [3, 3, 3] as [number, number, number],
    data: Float32Array.from({ length: 27 }, (_, index) => {
      const x = Math.floor(index / 9);
      const y = Math.floor(index / 3) % 3;
      const z = index % 3;
      return coordinates[0][x] + 2 * coordinates[1][y] + 3 * coordinates[2][z];
    }),
  };
  const stretched = prepareVtkRequest({
    kind: "scalar",
    field: stretchedField,
    grid: { geometryKind: "physical", scalarCoordinatesM: coordinates },
  });
  check(stretched.spacing.join(",") === "150,300,75", "worker spans stretched x/y/z endpoint centers with uniform VTK spacing");
  check(stretched.samplingProvenance.coordinateResamplingAxes.join(",") === "x,y,z", "worker provenance identifies every stretched axis");
}

{
  const points = { getData: () => ({ byteLength: 120 }) };
  const polys = { getData: () => ({ byteLength: 80 }) };
  const normals = { getData: () => ({ byteLength: 120 }) };
  const source = {
    getOutputData: () => ({
      getNumberOfPolys: () => 10,
      getPoints: () => points,
      getPolys: () => polys,
      getPointData: () => ({ getArrays: () => [normals] }),
      getCellData: () => ({ getArrays: () => [] }),
    }),
  };
  const usage = measureContourOutput(source);
  check(usage.bytes === 320 && usage.triangles === 10, "contour accounting includes geometry and attribute arrays");
  check(contourBudgetViolation(source, 319, 100)?.includes("bytes") === true, "observed contour bytes enforce a deterministic ceiling");
  check(contourBudgetViolation(source, 1000, 9)?.includes("triangles") === true, "observed contour triangles enforce a deterministic ceiling");
  check(contourBudgetViolation(source, 1000, 10) === null, "contour output at both ceilings is accepted");
  check(
    combinedContourBudgetViolation([source, source], 639, 100)?.includes("bytes") === true,
    "paired surfaces share one combined retained-output ceiling",
  );
  check(
    combinedContourBudgetViolation([source, source], 640, 20) === null,
    "paired surfaces at the combined ceilings are accepted",
  );
}

{
  const subtract = { shape, data: new Float32Array(24).fill(3) };
  const prepared = prepareVtkRequest({
    kind: "scalar",
    field,
    subtractField: subtract,
    grid,
  });
  check(prepared.values.every((value) => Number.isFinite(value) && value >= 0), "rain subtraction clamps and sanitizes before VTK packing");
  check(vtkPreparationTransferables(prepared).length === 2, "values and missing-data support transfer together");
  check(
    vtkPreparationTransferables(prepared)[0] === prepared.values.buffer,
    "the transferred buffer is the values buffer itself, not a copy",
  );

  const missingPrimary = { shape, data: field.data.slice() };
  missingPrimary.data[0] = Number.NaN;
  const missingSubtract = { shape, data: subtract.data.slice() };
  missingSubtract.data[1] = Number.NaN;
  const missingCloud = prepareVtkRequest({
    kind: "cloud",
    field: missingPrimary,
    subtractField: missingSubtract,
    grid,
    volumeMaxVoxels: 24,
    surfaceMaxVoxels: 24,
  });
  check(
    missingCloud.surface.sourceNonFiniteCount === 3
      && missingCloud.surface.values.every(Number.isFinite),
    "cloud preparation retains the pre-sanitization mask count while keeping VTK values finite",
  );

  const cloud = prepareVtkRequest({
    kind: "cloud",
    field,
    subtractField: subtract,
    grid,
    volumeMaxVoxels: 12,
    surfaceMaxVoxels: 24,
  });
  check(cloud.volume.values.length < cloud.surface.values.length, "combined cloud request applies the lower volume resolution ceiling");
  check(cloud.surface.values.length === 24, "combined cloud request preserves the surface resolution ceiling");
  check(JSON.stringify(cloud.surfaceRefinement) === JSON.stringify([1, 1, 1]), "fine cloud preparation avoids invented refinement");
  check(cloud.surface.samplingProvenance.reconstruction === "none", "fine cloud provenance records no display reconstruction");
  check(cloud.surfaceFocusBounds !== null, "cloud preparation publishes dense-cloud camera bounds from physical values");
  check(cloud.presentationDetail === null, "scientific cloud preparation contains no procedural detail");
  check(vtkPreparationTransferables(cloud).length === 4, "combined cloud request transfers both scalar outputs and their support masks");

  const sharedCloud = prepareVtkRequest({
    kind: "cloud",
    field,
    grid,
    volumeMaxVoxels: 24,
    surfaceMaxVoxels: 24,
  });
  check(
    sharedCloud.volume.values.buffer === sharedCloud.surface.values.buffer,
    "identical scientific cloud lattices share one packed scalar buffer",
  );
  check(
    vtkPreparationTransferables(sharedCloud).length === 2 && sharedCloud.volume.validity === sharedCloud.surface.validity,
    "shared cloud output transfers aliased values and support exactly once",
  );
  check(
    sharedCloud.cloudCenter.every(Number.isFinite),
    "cloud preparation returns a finite condensate-weighted storm centre",
  );

  const parentSource = {
    shape: [4, 4, 2] as [number, number, number],
    data: new Float32Array(4 * 4 * 2).fill(1),
  };
  const parentContext = prepareVtkRequest({
    kind: "cloud",
    field: parentSource,
    grid: { drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0.5, 0.5, 0.5] },
    volumeMaxVoxels: parentSource.data.length,
    surfaceMaxVoxels: parentSource.data.length,
    excludePhysicalBoundsM: { lower: [1, 1, 0], upper: [3, 3, 2] },
  });
  check(
    parentContext.volume.values.filter((value) => value === 0).length === 8,
    "coarse context masks exactly the cells whose centres are owned by the fine child",
  );
  check(
    parentSource.data.every((value) => value === 1),
    "fine-level exclusion never mutates the cached parent science field",
  );

  const presentationCloud = prepareVtkRequest({
    kind: "cloud",
    field,
    grid: { ...grid, drM: 600, dyM: 700, dzM: 400 },
    volumeMaxVoxels: 24,
    surfaceMaxVoxels: 1_000_000,
    fidelityMode: "presentation",
    detailSeed: 99,
  });
  check(presentationCloud.presentationDetail === null, "ordinary Presentation cloud preparation adds no procedural detail");
  const resolvedCloud = prepareVtkRequest({ kind: "cloud", field,
    grid: { ...grid, drM: 600, dyM: 700, dzM: 400 }, volumeMaxVoxels: 24,
    surfaceMaxVoxels: 1_000_000, fidelityMode: "scientific" });
  check(presentationCloud.surface.values.every((value, index) => Object.is(value, resolvedCloud.surface.values[index])),
    "Presentation cloud values exactly match the same declared linear reconstruction in Scientific preparation");

  const flowField = { shape, data: new Float32Array(24).fill(2) };
  const flowPresentationCloud = prepareVtkRequest({
    kind: "cloud",
    field,
    grid: { ...grid, drM: 600, dyM: 700, dzM: 400 },
    volumeMaxVoxels: 24,
    surfaceMaxVoxels: 1_000_000,
    fidelityMode: "presentation",
    detailSeed: 99,
    flowFields: [flowField, flowField, flowField],
    simulationTimeS: 30,
  });
  check(
    flowPresentationCloud.presentationDetail === null
      && flowPresentationCloud.surface.values.every((value, index) => Object.is(value, presentationCloud.surface.values[index])),
    "time, seed and velocity fields cannot modify ordinary rendered cloud scalar values",
  );
}

{
  const u = { shape, data: new Float32Array(24).fill(1) };
  const v = { shape, data: new Float32Array(24).fill(2) };
  const wData = new Float32Array(24);
  wData[(2 * shape[1] + 1) * shape[2] + 3] = 10;
  const w = { shape, data: wData };
  const expectedCart = vectorFieldToCartesianGrid(u, v, w, grid);
  const prepared = prepareVtkRequest({
    kind: "vector",
    fields: [u, v, w],
    grid,
    detectVerticalVortexCore: true,
  });
  check(equal(prepared.values, cartesianGridToVtkVectors(expectedCart)), "worker vector preparation preserves exact VTK tuple ordering");
  check(prepared.stormCenter[0] === 210 && prepared.stormCenter[1] === 220, "worker computes the strongest vertical-motion column");
  check(prepared.speed95 > 0, "worker computes the representative streamline speed");
  check(
    Math.abs((prepared.seedColumnSpeedMaximum ?? 0) - Math.hypot(1, 2, 10)) < 1e-6,
    "worker anchors streamline integration speed to the selected storm column",
  );
  check(prepared.featureSeeds.length > 0, "worker produces resolved-feature streamline seeds");
  check(
    prepared.vortexCore?.peakVorticityPerS === 10,
    "worker locates the vertical vortex core without a UI-thread component copy",
  );
  check(
    prepared.featureSeeds[0] === 210
      && prepared.featureSeeds[1] === 220
      && prepared.featureSeeds[2] === 175,
    "feature seeds rank the strongest resolved vector sample first",
  );
  check(vtkPreparationTransferables(prepared).length === 2, "vector values and feature seeds are transferable");
}

class FakeWorker implements PreparationWorker {
  onmessage: ((event: { data:
    | { kind?: "result"; ok: boolean; result?: VtkPreparationResult; error?: string }
    | { kind: "ready" }
  }) => void) | null = null;
  onerror: ((event: { message: string; preventDefault?: () => void }) => void) | null = null;
  terminated = 0;
  posted: VtkPreparationRequest | null = null;
  postMessage(message:
    | VtkPreparationRequest
    | { kind: "prepare"; requestId: number; request: VtkPreparationRequest }
    | { kind: "close" }
  ): void {
    if (message.kind === "prepare") this.posted = message.request;
    else if (message.kind !== "close") this.posted = message;
  }
  terminate(): void { this.terminated++; }
  succeed(result: VtkPreparationResult): void {
    this.onmessage?.({ data: { ok: true, result } });
  }
  fail(message: string, preventDefault?: () => void): void {
    this.onerror?.({ message, preventDefault });
  }
}

class QueueFakeWorker implements PreparationWorker {
  onmessage: PreparationWorker["onmessage"] = null;
  onerror: PreparationWorker["onerror"] = null;
  onmessageerror: PreparationWorker["onmessageerror"] = null;
  terminated = 0;
  throwOnPrepare = false;
  prepares: Array<{ kind: "prepare"; requestId: number; request: VtkPreparationRequest }> = [];
  postMessage(message:
    | VtkPreparationRequest
    | { kind: "prepare"; requestId: number; request: VtkPreparationRequest }
    | { kind: "close" }
  ): void {
    if (message.kind === "prepare") {
      if (this.throwOnPrepare) throw new Error("synchronous worker post failure");
      this.prepares.push(message);
    }
  }
  terminate(): void { this.terminated += 1; }
  ready(): void { this.onmessage?.({ data: { kind: "ready" } }); }
  fail(message: string): void { this.onerror?.({ message }); }
  succeed(requestId: number, result: VtkPreparationResult): void {
    this.onmessage?.({ data: { kind: "result", requestId, ok: true, result } });
  }
}

{
  const saved = new Map(["Worker", "document", "navigator", "crossOriginIsolated"]
    .map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const workers: QueueFakeWorker[] = [];
  let capabilityReads = 0;
  let uploads = 0;
  const gl = {
    ARRAY_BUFFER: 34962, STATIC_DRAW: 35044, NO_ERROR: 0,
    createBuffer: () => ({}), bindBuffer: () => undefined,
    bufferData: (_target: number, values: Float32Array) => {
      check(values.buffer instanceof SharedArrayBuffer, "shared preparation probes an actual shared buffer");
      uploads++;
    },
    getError: () => 0, deleteBuffer: () => undefined,
    getExtension: () => ({ loseContext: () => undefined }),
  };
  class QualifiedWorker extends QueueFakeWorker {
    constructor() {
      super(); workers.push(this); queueMicrotask(() => this.ready());
    }
    override postMessage(message: Parameters<QueueFakeWorker["postMessage"]>[0]): void {
      super.postMessage(message);
      if (message.kind !== "prepare") return;
      const result = prepareVtkRequest(message.request);
      sharePreparedVtkResult(result, message.request.datasetHandoff === "shared-geometry");
      queueMicrotask(() => this.succeed(message.requestId, result));
    }
  }
  Object.defineProperties(globalThis, {
    Worker: { configurable: true, value: QualifiedWorker },
    document: { configurable: true, value: { createElement: () => ({ getContext: () => gl }) } },
    navigator: { configurable: true, value: { get userAgent() {
      // Bound re-entry so a recursion regression cannot starve the test process's timers.
      if (++capabilityReads > 4) throw new Error("Shared preparation admission recursed");
      return "Chromium";
    } } },
    crossOriginIsolated: { configurable: true, value: true },
  });
  try {
    const request = { kind: "velocity-diagnostic" as const, diagnostic: "q-criterion" as const,
      fields: [field, field, field] as [typeof field, typeof field, typeof field], grid,
      datasetHandoff: "shared-geometry" as const };
    let result: VtkPreparationResult | null = null;
    let failure: unknown = null;
    try { result = await prepareVtkInWorker(request); } catch (error) { failure = error; }
    check(failure == null, "qualified shared preparation reaches the worker without recursive admission");
    check(uploads === 1 && capabilityReads === 1, "shared geometry is qualified once for the preparation request");
    check(workers.length === 1 && workers[0].prepares.length === 1,
      "qualified shared preparation submits one diagnostic preparation task");
    check(result != null && "values" in result && result.values.buffer instanceof SharedArrayBuffer,
      "successful preparation retains the qualified shared geometry path");
    capabilityReads = 0;
    const controller = new AbortController();
    const cancelled = prepareVtkInWorker(request, controller.signal);
    controller.abort();
    check(await cancelled.then(() => false, isPreparationAbort),
      "cancellation during shared qualification prevents a preparation dispatch");
    check(workers.length === 1 && workers[0].prepares.length === 1,
      "cancelled shared qualification adds no worker work");
  } finally {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
  }
}

for (const panes of [2, 4]) {
  const worker = new QueueFakeWorker();
  const coordinator = new PreparationCoordinator(() => worker);
  const shared = new SharedPreparationCoordinator(coordinator);
  const controllers = Array.from({ length: panes }, () => new AbortController());
  const request = { kind: "cloud" as const, field, grid, volumeMaxVoxels: 24, surfaceMaxVoxels: 24 };
  const promises = controllers.map((controller, index) => shared.enqueue({ ...request,
    fidelityMode: index % 2 ? "presentation" : "scientific", detailSeed: index,
    simulationTimeS: index * 30, flowFields: [field, field, field],
  }, controller.signal, { scope: `cloud-pane-${index}` }));
  worker.ready();
  check(worker.prepares.length === 1, `${panes} cloud panes share one exact preparation across display metadata`);
  const sent = worker.prepares[0].request;
  check(sent.kind === "cloud" && sent.flowFields == null && sent.fidelityMode == null && sent.simulationTimeS == null,
    "cloud worker transport omits unused wind buffers and display metadata");
  controllers[0].abort();
  check(await promises[0].then(() => false, isPreparationAbort), "one cloud pane cancels without changing shared result ownership");
  check(worker.terminated === 0, "remaining cloud panes retain their single preparation worker");
  const result = prepareVtkRequest(request);
  worker.succeed(worker.prepares[0].requestId, result);
  check((await Promise.all(promises.slice(1))).every(value => value === result),
    "cloud panes receive the same immutable numerical result");
  check(vtkPreparationIdentity(request) !== vtkPreparationIdentity({ ...request, surfaceMaxVoxels: 12 }),
    "different cloud sampling caps retain different preparation identities");
  coordinator.shutdown();
}

{
  const throwing = new QueueFakeWorker();
  throwing.throwOnPrepare = true;
  const retry = new QueueFakeWorker();
  const workers = [throwing, retry];
  const coordinator = new PreparationCoordinator(() => workers.shift()!);
  const resourceBaseline = getVisualizationResourceSnapshot().residentBytes;
  const promise = coordinator.enqueue({ kind: "scalar", field, grid });
  throwing.ready();
  await new Promise((resolve) => setTimeout(resolve, 40));
  retry.ready();
  const result = prepareVtkRequest({ kind: "scalar", field, grid });
  retry.succeed(retry.prepares[0].requestId, result);
  check(await promise === result, "synchronous worker-post failure retries on a fresh worker");
  check(
    getVisualizationResourceSnapshot().residentBytes === resourceBaseline,
    "synchronous worker-post failure rolls back its resource reservation",
  );
  coordinator.shutdown();
}

{
  const first = new QueueFakeWorker();
  const retry = new QueueFakeWorker();
  const workers = [first, retry];
  const coordinator = new PreparationCoordinator(() => workers.shift()!);
  const resourceBaseline = getVisualizationResourceSnapshot().residentBytes;
  const resourceWorkerBaseline = getVisualizationResourceSnapshot()
    .byStage["worker-resident"].residentBytes;
  const promise = coordinator.enqueue({ kind: "scalar", field, grid });
  first.ready();
  check(
    getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
      === resourceWorkerBaseline + field.data.byteLength,
    "active preparation reports its worker-owned input bytes",
  );
  check(
    getVisualizationResourceSnapshot().byStage.handoff.reservedBytes
      >= vtkPreparationTransferables(prepareVtkRequest({ kind: "scalar", field, grid }))[0].byteLength,
    "active preparation reserves its bounded result handoff before allocation",
  );
  first.fail("transient preparation execution failure");
  await new Promise((resolve) => setTimeout(resolve, 40));
  retry.ready();
  const result = prepareVtkRequest({ kind: "scalar", field, grid });
  retry.succeed(retry.prepares[0].requestId, result);
  check(
    await promise === result,
    "coordinated preparation retries an active request after a transient worker failure",
  );
  check(
    getVisualizationResourceSnapshot().residentBytes === resourceBaseline,
    "preparation retry releases worker-input ownership after commit",
  );
  check(first.terminated === 1, "coordinated preparation retires the failed worker before retrying");
  coordinator.shutdown();
}

{
  const first = new QueueFakeWorker();
  const latestWorker = new QueueFakeWorker();
  const workers = [first, latestWorker];
  const coordinator = new PreparationCoordinator(() => workers.shift()!);
  const controller = new AbortController();
  const obsolete = coordinator.enqueue(
    { kind: "scalar", field, grid },
    controller.signal,
    { scope: "volume-pane" },
  );
  first.ready();
  check(first.prepares.length === 1, "coordinator starts one preparation generation");
  controller.abort();
  check(
    await obsolete.then(() => false, isPreparationAbort),
    "active obsolete preparation releases its React consumer immediately",
  );
  check(first.terminated === 1, "active obsolete preparation terminates its synchronous worker");
  const latest = coordinator.enqueue(
    { kind: "scalar", field, grid },
    undefined,
    { scope: "volume-pane" },
  );
  check(latestWorker.prepares.length === 0, "replacement waits for its fresh worker handshake");
  latestWorker.ready();
  check(latestWorker.prepares.length === 1, "latest generation starts without waiting for obsolete arithmetic");
  const result = prepareVtkRequest({ kind: "scalar", field, grid });
  latestWorker.succeed(latestWorker.prepares[0].requestId, result);
  check(await latest === result, "latest serialized preparation resolves normally");
  coordinator.shutdown();
}

{
  const worker = new QueueFakeWorker();
  const coordinator = new PreparationCoordinator(() => worker);
  const shared = new SharedPreparationCoordinator(coordinator);
  const firstController = new AbortController();
  const secondController = new AbortController();
  const request = { kind: "scalar" as const, field, grid, maxVoxels: 24 };
  const equivalentField = {
    ...field,
    shape: [...field.shape] as [number, number, number],
    data: new Float32Array(field.data.buffer, field.data.byteOffset, field.data.length),
  };
  const first = shared.enqueue(request, firstController.signal, { scope: "pane-a" });
  const second = shared.enqueue(
    { ...request, field: equivalentField, grid: { ...grid } },
    secondController.signal,
    { scope: "pane-b" },
  );
  worker.ready();
  check(worker.prepares.length === 1, "identical pane preparations execute once");
  check(
    vtkPreparationIdentity(request) === vtkPreparationIdentity({ ...request }),
    "preparation identity follows immutable field/grid inputs rather than request wrappers",
  );
  check(
    vtkPreparationIdentity(request) === vtkPreparationIdentity({
      ...request,
      grid: { ...grid },
    }),
    "equivalent per-pane grid wrappers share one preparation identity",
  );
  check(
    vtkPreparationIdentity(request) === vtkPreparationIdentity({
      ...request,
      field: equivalentField,
    }),
    "equivalent field wrappers over one immutable backing store share one preparation identity",
  );
  check(
    vtkPreparationIdentity(request) !== vtkPreparationIdentity({
      ...request,
      field: { ...equivalentField, data: equivalentField.data.slice() },
    }),
    "separate field backing stores cannot share a preparation identity",
  );
  check(
    vtkPreparationIdentity(request) !== vtkPreparationIdentity({
      ...request,
      grid: { ...grid, dzM: (grid.dzM ?? 1) + 1 },
    }),
    "scientifically distinct grid spacing cannot share a preparation identity",
  );
  firstController.abort();
  check(
    await first.then(() => false, isPreparationAbort),
    "one shared-preparation consumer can cancel independently",
  );
  check(worker.terminated === 0, "one consumer cannot terminate another pane's shared preparation");
  const result = prepareVtkRequest(request);
  worker.succeed(worker.prepares[0].requestId, result);
  check(await second === result, "remaining shared-preparation consumer receives the result");
  coordinator.shutdown();
}

{
  const worker = new QueueFakeWorker();
  const coordinator = new PreparationCoordinator(() => worker);
  const shared = new SharedPreparationCoordinator(coordinator);
  const firstController = new AbortController();
  const secondController = new AbortController();
  const request = { kind: "scalar" as const, field, grid, maxVoxels: 24 };
  const first = shared.enqueue(request, firstController.signal);
  const second = shared.enqueue({ ...request }, secondController.signal);
  worker.ready();
  firstController.abort();
  secondController.abort();
  check(
    await Promise.all([
      first.then(() => false, isPreparationAbort),
      second.then(() => false, isPreparationAbort),
    ]).then((values) => values.every(Boolean)),
    "all released shared-preparation consumers observe cancellation",
  );
  check(worker.terminated === 1, "the last consumer release terminates obsolete preparation work");
  coordinator.shutdown();
}

{
  const abandonedWorker = new QueueFakeWorker();
  const replacementWorker = new QueueFakeWorker();
  const workers = [abandonedWorker, replacementWorker];
  const coordinator = new PreparationCoordinator(() => workers.shift()!);
  const shared = new SharedPreparationCoordinator(coordinator);
  const request = { kind: "scalar" as const, field, grid, maxVoxels: 24 };
  const abandonedController = new AbortController();
  const abandoned = shared.enqueue(request, abandonedController.signal);
  abandonedWorker.ready();
  abandonedController.abort();

  // Enqueue in the same turn, before the abandoned worker promise's finally
  // callback can run. Context recovery does exactly this as it advances from
  // the lost generation to a fresh renderable generation.
  const replacement = shared.enqueue({ ...request }, new AbortController().signal);
  replacementWorker.ready();
  check(
    replacementWorker.prepares.length === 1,
    "immediate recovery generation starts on a fresh preparation worker",
  );
  const result = prepareVtkRequest(request);
  replacementWorker.succeed(replacementWorker.prepares[0].requestId, result);
  check(
    await abandoned.then(() => false, isPreparationAbort),
    "abandoned preparation generation still rejects with AbortError",
  );
  check(
    await replacement === result,
    "recovery generation does not inherit the abandoned preparation rejection",
  );
  coordinator.shutdown();
}

{
  const fake = new FakeWorker();
  const controller = new AbortController();
  const promise = prepareVtkInWorker(
    { kind: "scalar", field, grid },
    controller.signal,
    () => fake,
  );
  controller.abort();
  let error: unknown;
  try { await promise; } catch (caught) { error = caught; }
  check(isPreparationAbort(error), "aborting a stale generation rejects with AbortError");
  check(fake.terminated === 1, "aborting a stale generation terminates its worker");
  check(fake.posted?.kind === "scalar", "worker receives the bounded preparation request");
}

{
  const fake = new FakeWorker();
  const promise = prepareVtkInWorker({ kind: "scalar", field, grid }, undefined, () => fake);
  const result = prepareVtkRequest({ kind: "scalar", field, grid });
  fake.succeed(result);
  const resolved = await promise;
  check(resolved === result, "successful worker result resolves exactly once");
  check(fake.terminated === 0, "successful one-shot worker is allowed to close itself after transferring its result");
}

{
  const first = new FakeWorker();
  const retry = new FakeWorker();
  const queue = [first, retry];
  let prevented = false;
  const promise = prepareVtkInWorker(
    { kind: "scalar", field, grid },
    undefined,
    () => queue.shift()!,
  );
  first.fail("transient module-worker startup failure", () => { prevented = true; });
  await new Promise((resolve) => setTimeout(resolve, 40));
  const result = prepareVtkRequest({ kind: "scalar", field, grid });
  retry.succeed(result);
  check(await promise === result, "transient worker startup failure retries the identical preparation");
  check(prevented, "handled worker startup failure does not escape as a page error");
  check(first.terminated === 1 && retry.terminated === 0, "retry terminates only the failed worker generation");
}

{
  const first = new FakeWorker();
  const retry = new FakeWorker();
  const queue = [first, retry];
  const promise = prepareVtkInWorker(
    { kind: "scalar", field, grid },
    undefined,
    () => queue.shift()!,
  );
  first.fail("first startup refusal");
  await new Promise((resolve) => setTimeout(resolve, 40));
  retry.fail("second startup refusal");
  const prepared = await promise;
  check(
    equal(prepared.values, prepareVtkRequest({ kind: "scalar", field, grid }).values),
    "two worker startup refusals preserve the bounded preparation through the local fallback",
  );
}

{
  check(
    vtkGeometryExecutionMode(true) === "worker"
      && vtkGeometryExecutionMode(false) === "local",
    "geometry execution follows worker capability instead of a browser user-agent cap",
  );
  check(
    localVtkContourBudgetError(LOCAL_VTK_CONTOUR_VOXEL_CAP, 1024) == null,
    "local contours accept a bounded lattice",
  );
  check(
    localVtkContourBudgetError(LOCAL_VTK_CONTOUR_VOXEL_CAP + 1, 1024) != null,
    "local contours reject an oversized lattice before marching cubes",
  );
  check(
    localVtkStreamlineBudgetError(1024, 64, 300) == null,
    "local streamlines accept the normal interactive workload",
  );
  check(
    localVtkStreamlineBudgetError(LOCAL_VTK_GEOMETRY_INPUT_BYTE_CAP + 1, 64, 300) != null
      && localVtkStreamlineBudgetError(1024, 512, 2_001) != null,
    "local streamlines reject oversized input and integration work",
  );

  const preparationHook = readFileSync(
    new URL("../../../../ui/src/lib/vtk/useVtkPreparation.ts", import.meta.url),
    "utf8",
  );

  check(
    preparationHook.includes("_generation: number")
      && !preparationHook.includes("state.generation")
      && !preparationHook.includes("generation: -1"),
    "GPU lifecycle generation remains API-compatible but is not CPU preparation identity",
  );

  const plotFiles = [
    "VolumeRender3D.tsx", "Isosurface3D.tsx", "Streamlines3D.tsx",
    "SupercellRender3D.tsx",
  ];
  const sources = plotFiles.map((file) =>
    readFileSync(new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url), "utf8")
  );
  check(sources.every((source) => !source.includes("fieldToCartesianGrid(")), "3D scenes do not synchronously build scalar Cartesian grids");
  check(sources.every((source) => !source.includes("vectorFieldToCartesianGrid(")), "3D scenes do not synchronously build vector Cartesian grids");
  check(sources.every((source) => !source.includes("cartesianGridToVtk")), "3D scenes do not synchronously repack VTK arrays");
  const supercell = sources[3];
  const iso = sources[1];
  const streamlines = sources[2];
  const volume = sources[0];
  const contourWorker = readFileSync(
    new URL("../../../../ui/src/lib/vtk/vortexContour.worker.ts", import.meta.url),
    "utf8",
  );
  const streamlineWorker = readFileSync(
    new URL("../../../../ui/src/lib/vtk/streamlineExtraction.worker.ts", import.meta.url),
    "utf8",
  );
  check(supercell.includes("committedCloudFloorGkg") && supercell.includes("volume preview"), "supercell labels and debounces the cloud-floor preview");
  check(supercell.includes("committedUpdraftThreshold"), "supercell debounces updraft contour commits");
  check(
    supercell.includes("supercellAuxiliaryVoxelBudget(")
      && supercell.includes("auxiliaryContourVoxelCap")
      && !supercell.includes("contourVoxelBudget(auxiliaryContourVoxelCap"),
    "composite flow and vortex contours use the pane-bounded native-detail budget without post-extraction coarsening",
  );
  check(
    supercell.includes("flowUsesExactLattice")
      && supercell.includes("vortexUsesExactLattice")
      && supercell.includes('flowUsesExactLattice || flowSupportBounds != null ? "cell_mean" : "block_max"')
      && supercell.includes("opposite: downdraftSurfaceAvailable ? -flowThreshold : null")
      && supercell.includes("opposite: vortexOppositeAvailable")
      && !supercell.includes("downdraftExtraction")
      && !supercell.includes("vortexOppositeExtraction"),
    "supercell exact and complete-support lattices extract both signs from one field and one contour request",
  );
  check(
    supercell.includes("locatedFlowCenter")
      && supercell.includes("locatedVortexCenter")
      && supercell.includes("planNativeStructuralSupportWindows")
      && supercell.includes("diagnosticFrameIdentity")
      && supercell.includes("flowDiagnosticOutcome")
      && supercell.includes("vortexDiagnosticOutcome")
      && supercell.includes("MIN_CONTOUR_VOXEL_BUDGET"),
    "Flow and Vortex use frame-local extrema/support locators before native feature ROIs",
  );
  check(
    !supercell.includes("SUPERCELL_AUXILIARY_CONTOUR_TRIANGLE_CAP")
      && !supercell.includes("SUPERCELL_AUXILIARY_CONTOUR_BYTE_CAP")
      && !supercell.includes("contourBackoff("),
    "composite diagnostics have no renderer-specific byte or triangle ceiling",
  );
  check(
    supercell.includes("cloudDensityForDiagnostics")
      && supercell.includes('useState<DiagnosticVisibility>("auto")')
      && supercell.includes("cloud display density"),
    "embedded diagnostics automatically reveal through cloud display opacity with an override",
  );
  check(iso.includes("committedThreshold") && iso.includes("· preview"), "isosurface labels and debounces threshold previews");
  check(
    iso.includes("negativeContourExtraction")
      && iso.includes("primaryGeometry?.triangles ?? 0")
      && iso.includes("negativeGeometry?.triangles ?? 0")
      && !iso.includes("contourUsageViolation(usage)"),
    "signed isosurfaces publish both admitted worker contours without a downstream fixed ceiling",
  );
  check(
    streamlines.includes("featureSeeds")
      && streamlines.includes("result.backward.path")
      && streamlineWorker.includes("const backward = direction(request, -1)"),
    "vorticity lines use prepared feature seeds and off-thread bidirectional integration",
  );
  check(
    iso.includes("vtkPolyData.newInstance()")
      && supercell.includes("vtkPolyData.newInstance()")
      && contourWorker.includes("imageData.getPointData().setScalars")
      && contourWorker.indexOf("loadImage(request.image)")
        < contourWorker.indexOf("contour(request.thresholds.core)"),
    "marching-cubes workers receive real scalar inputs before extraction and scenes begin as empty geometry",
  );
  const materialStart = volume.indexOf("Transfer-function / material update");
  const materialEnd = volume.indexOf("Vertical exaggeration", materialStart);
  const volumeMaterialEffect = volume.slice(materialStart, materialEnd);
  check(
    materialStart >= 0
      && !volumeMaterialEffect.includes("setPreparedImageScalars(")
      && !volumeMaterialEffect.includes("scene.mapper.modified()"),
    "volume material controls do not invalidate or re-upload the immutable scalar texture",
  );
  check(
    volume.includes("property.setComputeNormalFromOpacity(false)")
      && volume.includes("volumeUsesOpacityNormals(opacityClass)")
      && volume.includes("previewImageSampleDistance: 2")
      && volume.includes("previewDistanceMultiplier: volumeAppearanceQualification?.previewDistanceMultiplier ?? 2")
      && volume.includes('max={2.5}')
      && supercell.includes("const peak = scaledCloudPeakOpacity(scale)")
      && supercell.includes("const peak = scaledPrecipPeakOpacity(opacity)"),
    "volume density controls preserve scalar-gradient boundaries without high-end linear opacity clamps",
  );
  const localContourHook = readFileSync(
    new URL("../../../../ui/src/lib/vtk/useVortexContourExtraction.ts", import.meta.url),
    "utf8",
  );
  const localStreamlineHook = readFileSync(
    new URL("../../../../ui/src/lib/vtk/useStreamlineExtraction.ts", import.meta.url),
    "utf8",
  );
  const streamlineClient = readFileSync(
    new URL("../../../../ui/src/lib/vtk/streamlineExtractionClient.ts", import.meta.url),
    "utf8",
  );
  const workerOwnedStreamlinePreparation = readFileSync(
    new URL("../../../../ui/src/lib/vtk/streamlineWorkerOwnedPreparation.ts", import.meta.url),
    "utf8",
  );
  const contourClient = readFileSync(
    new URL("../../../../ui/src/lib/vtk/vortexContourClient.ts", import.meta.url),
    "utf8",
  );
  const workerOwnedContourPreparation = readFileSync(
    new URL("../../../../ui/src/lib/vtk/contourWorkerOwnedPreparation.ts", import.meta.url),
    "utf8",
  );
  const preparationClient = readFileSync(
    new URL("../../../../ui/src/lib/vtk/vtkPreparationClient.ts", import.meta.url),
    "utf8",
  );
  check(
    [preparationClient, contourClient, streamlineClient].every((source) =>
      source.includes("?worker&url") && source.includes("generation=")),
    "every restartable VTK worker uses a generation-specific WebKit module URL",
  );
  check(
    preparationClient.includes("WebKit uses transferable geometry")
      && preparationClient.includes("transferableGeometryHandoffReason"),
    "WebKit keeps preparation off-thread without exposing vtk.js to SAB-backed geometry arrays",
  );
  check(
    localContourHook.includes("controller.abort()")
      && localStreamlineHook.includes("controller.abort()")
      && !localContourHook.includes('import("./vtkContourFallback")')
      && !localStreamlineHook.includes('import("./vtkStreamlineFallback")'),
    "production geometry hooks cancel workers and cannot enter a UI-thread fallback",
  );
  check(
    [contourClient, streamlineClient].every((source) =>
      !source.includes("prepared.values.slice()")
      && source.includes("handoffVtkExtractionDataset")
      && source.includes("recordWarmVtkDatasetReuse")),
    "geometry hooks avoid ad-hoc grid clones and instrument warm zero-byte reuse",
  );
  check(
    !volume.includes("geometryHandoff")
      && iso.includes("geometryHandoff: true")
      && streamlines.includes("geometryHandoff: true"),
    "render-only volume preparation stays zero-copy while secondary geometry consumers opt into sharing",
  );
  check(
    !iso.includes("preparation.prepared")
      && !streamlines.includes("preparation.prepared")
      && !supercell.includes("updraftPreparation.prepared")
      && !supercell.includes("vortexCorePreparation.prepared")
      && !supercell.includes("tornadicCondensatePreparation.prepared")
      && volume.includes("preparation.prepared"),
    "geometry scenes consume opaque metadata while volume upload retains an explicit resident boundary",
  );
  check(
    streamlineWorker.includes('event.data.kind === "prepare"')
      && streamlineWorker.includes("prepareVtkRequest(event.data.request)")
      && workerOwnedStreamlinePreparation.includes("retainRemotePreparedDataset")
      && workerOwnedStreamlinePreparation.includes("worker-owned-vector-output"),
    "combined streamline worker prepares a remote dataset behind an admitted opaque handle",
  );
  check(
    preparationHook.includes("prepareVectorInWorkerOwnedService")
      && localStreamlineHook.includes("extractStreamlinesFromWorkerOwnedHandle")
      && localStreamlineHook.includes('preparedHandle.residency === "worker"'),
    "ordinary streamline preparation and extraction select the combined worker-owned path",
  );
  check(
    contourWorker.includes('event.data.kind === "prepare"')
      && contourWorker.includes("prepareVtkRequest({ ...event.data.request, includeValidity: false })")
      && workerOwnedContourPreparation.includes("retainRemotePreparedDataset")
      && workerOwnedContourPreparation.includes("worker-owned-contour-output"),
    "combined contour worker prepares a remote scalar behind an admitted opaque handle",
  );
  check(
    preparationHook.includes("prepareScalarInWorkerOwnedService")
      && localContourHook.includes("extractContoursFromWorkerOwnedHandles")
      && !iso.includes("workerOwned: false")
      && supercell.includes("supercell-flow:")
      && supercell.includes("supercell-vortex:")
      && supercell.includes("supercell-condensate:")
      && !supercell.includes("workerOwned: false"),
    "ordinary isosurfaces and all supercell contour layers use worker ownership",
  );
}

{
  const prediction = getVisualizationResourceSnapshot().prediction;
  check(
    prediction.samples > 0 && prediction.underpredictions === 0,
    "prepared-result upper bounds cover every measured preparation result",
  );
}

{
  mock.timers.enable({ apis: ["setTimeout"] });
  const workers: QueueFakeWorker[] = [];
  const coordinator = new PreparationCoordinator(() => {
    const worker = new QueueFakeWorker();
    workers.push(worker);
    return worker;
  });
  try {
    const request = { kind: "scalar" as const, field, grid };
    const result = prepareVtkRequest(request);
    const delayed = coordinator.enqueue(request);
    mock.timers.tick(5_000);
    check(workers.length === 1 && workers[0].terminated === 0,
      "a slow cold module load is not mistaken for a dead preparation worker");
    workers[0].ready();
    workers[0].succeed(workers[0].prepares[0].requestId, result);
    check(await delayed === result, "delayed startup retains the exact requested preparation");
    coordinator.shutdown();

    const canceled = new AbortController();
    const abandoned = coordinator.enqueue(request, canceled.signal).then(() => false, isPreparationAbort);
    canceled.abort();
    check(await abandoned && workers[1].terminated === 1,
      "canceling the last queued request immediately releases a starting worker");

    const dead = coordinator.enqueue(request).then(() => null, error => error as Error);
    mock.timers.tick(30_000);
    mock.timers.tick(25);
    mock.timers.tick(30_000);
    check((await dead)?.message === "VTK preparation worker did not start"
      && workers[2].terminated === 1 && workers[3].terminated === 1,
    "two silent startups fail within the bounded deadline and release both workers");

    const retryController = new AbortController();
    const retryCanceled = coordinator.enqueue(request, retryController.signal).then(() => false, isPreparationAbort);
    workers[4].fail("cold module rejected");
    retryController.abort();
    mock.timers.tick(25);
    check(await retryCanceled && workers.length === 5,
      "a canceled retry cannot create an unowned worker");

    const recovered = coordinator.enqueue(request);
    workers[5].fail("transient new module failure");
    mock.timers.tick(25);
    workers[6].ready();
    workers[6].succeed(workers[6].prepares[0].requestId, result);
    check(await recovered === result, "a later request receives a fresh startup retry budget");
  } finally {
    coordinator.shutdown();
    mock.timers.reset();
  }
}

console.log(`\nvtkPreparation tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
