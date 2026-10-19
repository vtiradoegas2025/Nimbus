/* Shared render fidelity contract. "scientific"=resolved model data; "presentation"=disclosed display detail added.
 * Distinct from GPU quality / ray-step LOD, which is a separate performance axis. */

export type RenderFidelityMode = "scientific" | "presentation";

export const DEFAULT_RENDER_FIDELITY_MODE: RenderFidelityMode = "scientific";

export type RenderOperation =
  | "native-sampling"
  | "bounded-subsampling"
  /* Block mean over the cells it replaces. A convex combination, so it introduces no new
   * extrema and nothing outside the resolved range — but it is not a subsample, and reporting
   * it as one would understate what happened to a peak value. */
  | "bounded-cell-mean-aggregation"
  | "bounded-extrema-aggregation"
  | "piecewise-linear-reconstruction"
  | "linear-volume-interpolation"
  | "threshold-feature-extraction"
  | "surface-smoothing"
  | "support-constrained-subgrid-detail";

const PRESENTATION_ONLY_OPERATIONS = new Set<RenderOperation>([
  "surface-smoothing",
  "support-constrained-subgrid-detail",
]);

/** Whether an operation may run in the selected fidelity mode. */
export function renderOperationAllowed(
  mode: RenderFidelityMode,
  operation: RenderOperation,
): boolean {
  return mode === "presentation" || !PRESENTATION_ONLY_OPERATIONS.has(operation);
}

export interface VtkSamplingProvenance {
  /** Valid (halo-cropped) simulation samples before renderer LOD. */
  sourceShape: [number, number, number];
  sourceSpacing: [number, number, number];
  /** Samples retained after the renderer's bounded preparation LOD. */
  sampledShape: [number, number, number];
  sampledSpacing: [number, number, number];
  /** Lattice actually handed to VTK after any display reconstruction. */
  displayShape: [number, number, number];
  displaySpacing: [number, number, number];
  /**
   * How the sampled lattice was produced from the source one.
   *
   * `bounded-subsampling` keeps one cell per block and discards the rest; `bounded-cell-mean`
   * replaces each block with its mean. They differ in what happens to a peak — a subsample
   * either keeps it or misses it, a mean always dilutes it — so they are disclosed apart.
   */
  sampling: "native" | "bounded-subsampling" | "bounded-cell-mean"
    | "bounded-linear-reflectivity-mean" | "bounded-block-max"
    | "bounded-block-min" | "bounded-magnitude-extreme";
  reconstruction: "none" | "piecewise-trilinear";
  reconstructionFactors: [number, number, number];
  /** VTK image data requires a regular lattice. Stretched imports are sampled
   * linearly onto uniform axes over the same endpoint-center spans. */
  coordinateResampling: "none" | "linear-to-uniform" | "conservative-terrain-to-cartesian";
  coordinateResamplingAxes: Array<"x" | "y" | "z">;
}

type GridGeometry = {
  shape: [number, number, number];
  spacing: [number, number, number];
  coordinateResampling?: "none" | "linear-to-uniform" | "conservative-terrain-to-cartesian";
  coordinateResamplingAxes?: Array<"x" | "y" | "z">;
};

/** Construct immutable display provenance while the preparation worker still
 * knows the native, sampled, and reconstructed grids independently. */
export function vtkSamplingProvenance(
  source: GridGeometry,
  sampled: GridGeometry,
  display: GridGeometry = sampled,
  reconstructionFactors: [number, number, number] = [1, 1, 1],
  /**
   * How the reduction was performed, when one happened. Stated by the caller rather than
   * inferred: the shapes show THAT cells were dropped, never whether the survivors are
   * original values or block means.
   */
  reduction: "subsample" | "cell-mean" | "linear-reflectivity-mean"
    | "block-max" | "block-min" | "magnitude-extreme" = "subsample",
): VtkSamplingProvenance {
  const factors = [...reconstructionFactors] as [number, number, number];
  const reconstructed = factors.some((factor) => factor > 1);
  const subsampled = source.shape.some(
    (count, axis) => count !== sampled.shape[axis],
  );
  return {
    sourceShape: [...source.shape],
    sourceSpacing: [...source.spacing],
    sampledShape: [...sampled.shape],
    sampledSpacing: [...sampled.spacing],
    displayShape: [...display.shape],
    displaySpacing: [...display.spacing],
    sampling: !subsampled
      ? "native"
      : reduction === "linear-reflectivity-mean"
        ? "bounded-linear-reflectivity-mean"
        : reduction === "block-max"
          ? "bounded-block-max"
          : reduction === "block-min"
            ? "bounded-block-min"
            : reduction === "magnitude-extreme"
              ? "bounded-magnitude-extreme"
              : reduction === "cell-mean"
                ? "bounded-cell-mean"
                : "bounded-subsampling",
    reconstruction: reconstructed ? "piecewise-trilinear" : "none",
    reconstructionFactors: factors,
    coordinateResampling: source.coordinateResampling ?? "none",
    coordinateResamplingAxes: [...(source.coordinateResamplingAxes ?? [])],
  };
}

function intervalRatio(sourceCount: number, sampledCount: number): number {
  if (sourceCount <= 1 || sampledCount <= 1) return 1;
  return (sourceCount - 1) / (sampledCount - 1);
}

function formatRatio(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1);
}

/** How far a lattice was reduced from its source; 1 is native. */
export function samplingReductionFactor(
  provenance: VtkSamplingProvenance | null | undefined,
): number {
  if (!provenance) return 1;
  return provenance.sourceShape.reduce(
    (factor, count, axis) =>
      factor * Math.max(1, intervalRatio(count, provenance.sampledShape[axis])),
    1,
  );
}

/**
 * The coarsest lattice among every layer drawn on one canvas.
 *
 * A scene composites several fields, each prepared under its own voxel budget.
 * Reporting any single one claims a resolution the others may never have had,
 * so the badge states the weakest — a floor the whole picture honours.
 */
export function coarsestVtkSampling(
  provenances: ReadonlyArray<VtkSamplingProvenance | null | undefined>,
): VtkSamplingProvenance | null {
  let coarsest: VtkSamplingProvenance | null = null;
  let worst = -Infinity;
  for (const provenance of provenances) {
    if (!provenance) continue;
    const factor = samplingReductionFactor(provenance);
    if (factor > worst) {
      worst = factor;
      coarsest = provenance;
    }
  }
  return coarsest;
}

/** Short, on-canvas provenance fragments. Native sampling is intentionally
 * silent; the mode badge itself still identifies the resolved-only path. */
export function describeVtkSampling(
  provenance: VtkSamplingProvenance | null | undefined,
): string[] {
  if (!provenance) return [];
  const details: string[] = [];
  if (provenance.sampling !== "native") {
    const ratios = provenance.sourceShape.map((count, axis) =>
      formatRatio(intervalRatio(count, provenance.sampledShape[axis]))
    );
    // Named apart: a reader judging whether a peak is trustworthy needs to know whether the
    // value shown is one of the source cells or the mean of a block of them.
    const kind = provenance.sampling === "bounded-cell-mean"
      ? "bounded cell-mean"
      : provenance.sampling === "bounded-linear-reflectivity-mean"
        ? "bounded linear-Z mean"
        : provenance.sampling === "bounded-block-max"
          ? "bounded block-max"
          : provenance.sampling === "bounded-block-min"
            ? "bounded block-min"
            : provenance.sampling === "bounded-magnitude-extreme"
              ? "bounded |value| extreme"
              : "bounded LOD";
    details.push(`${kind} ${ratios.join("×")}`);
  }
  if (provenance.reconstruction === "piecewise-trilinear") {
    details.push(
      `linear ${provenance.reconstructionFactors.join("×")} reconstruction`,
    );
  }
  if (provenance.coordinateResampling === "linear-to-uniform") {
    details.push(`linear stretched-${provenance.coordinateResamplingAxes.join("/")} resampling`);
  } else if (provenance.coordinateResampling === "conservative-terrain-to-cartesian") {
    details.push("physical-volume terrain→Cartesian remap");
  }
  return details;
}

export function describeLayerVtkSampling(
  label: string,
  provenance: VtkSamplingProvenance | null | undefined,
): string | null {
  if (!provenance || [provenance.sourceSpacing, provenance.sampledSpacing, provenance.displaySpacing]
    .some(spacing => spacing.length !== 3 || spacing.some(value => !Number.isFinite(value) || value <= 0))) return null;
  const format = (spacing: readonly number[]) => {
    const values = spacing.map(value => Number(value.toPrecision(4)).toString());
    return `${values.every(value => value === values[0]) ? values[0] : values.join("×")} m`;
  };
  const source = provenance.coordinateResampling === "none" ? "source" : "nominal source";
  const sampled = provenance.sampling === "native" && provenance.coordinateResampling === "none" ? "native samples" : "used samples";
  const details = [`${label}: ${source} ${format(provenance.sourceSpacing)}`,
    `${sampled} ${format(provenance.sampledSpacing)}`];
  if (provenance.reconstruction !== "none") {
    details.push(`linear display reconstruction ${format(provenance.displaySpacing)}`);
  }
  return details.join(" · ");
}

/** Presentation detail is unnecessary on a genuinely fine lattice. The
 * smoothstep avoids a visible mode discontinuity as grid spacing changes. */
export function presentationDetailWeight(
  spacing: [number, number, number],
  fineCutoffM = 100,
  coarseCutoffM = 500,
): number {
  const coarseAxis = Math.max(...spacing);
  if (!(coarseCutoffM > fineCutoffM)) return coarseAxis > fineCutoffM ? 1 : 0;
  const t = Math.max(
    0,
    Math.min(1, (coarseAxis - fineCutoffM) / (coarseCutoffM - fineCutoffM)),
  );
  return t * t * (3 - 2 * t);
}

export function renderFidelityLabel(mode: RenderFidelityMode): string {
  return mode === "scientific" ? "Scientific · resolved data" : "Presentation · disclosed display detail";
}

/** Stable per-run seed. Publication generations intentionally do not enter the
 * hash so movie frames share one coherent detail field instead of flickering. */
export function stableRenderSeed(identity: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < identity.length; i++) {
    hash ^= identity.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}
