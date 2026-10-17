import { sampleReconstruction, type VolumeReconstruction } from "./volumeReconstruction";
import { scalarVolumeBatchStats, scalarVolumeStats } from "./scalarVolumeStats";
import { focusedVolume, type VolumeBackground } from "./volumeBackground";
import { finiteVectorSupport, hasMissingValues } from "./finiteVectorSupport";
import { gridForWindowedField as reconstructionGrid } from "../science/windowedGrid";
import { terrainWindowToCartesianField, cartesianBoxToImageData } from "../science/cartesianInterp";
import {
  cartesianGridToVtkScalars,
  cartesianGridToVtkVectorsInPlace,
  cartesianDyMeters,
  cropFieldToValidRange,
  cartesianVectorSpeedQuantile,
  fieldToCartesianGrid,
  refineCartesianGridLinear,
  vectorFieldToCartesianGrid,
  CARTESIAN_VOXEL_CAP_VOLUME,
  type CartesianVoxelGrid,
  type GridSpacing,
} from "@/lib/science/cartesianInterp";
import type { Npy3D } from "@/lib/platform/npy";
import { assembleWindowedField } from "@/lib/run/windowedField";
import type { FieldLoadWindow } from "@/lib/run/runOutputs";
import {
  vtkSamplingProvenance,
  type VtkSamplingProvenance,
} from "@/lib/plot/renderFidelity";
import {
  cloudCondensateCentroid,
  type CloudPresentationDetailMetadata,
} from "@/lib/plot/cloudPresentationDetail";
import type { RenderFidelityMode } from "@/lib/plot/renderFidelity";
import {
  findInterleavedVerticalRotationMaximum,
  type VortexCore,
} from "@/lib/plot/vortexDetection";

export interface PreparedVtkImage {
  focusEmpty?: boolean;
  focusCropped?: boolean;
  volumeOwnership?: { offset: [number, number, number]; shape: [number, number, number] };
  volumeClipBounds?: [number, number, number, number, number, number];
  volumeUpperOpen?: [number, number, number];
  shape: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  /** X-fastest tuples. CPU geometry retains NaN; volume uploads use finite values plus validity. */
  values: Float32Array;
  /** Binary support in x-fastest order. Interpolation touching missing support stays transparent. */
  validity?: Uint8Array;
  /** Missing/invalid source samples before reconstruction and support filtering. */
  sourceNonFiniteCount: number;
  stats?: {
    min: number;
    max: number;
    p2: number;
    p98: number;
    /** 99.5th percentile over the complete prepared field. */
    p995?: number;
    /** Robust positive tail, conditioned on values above zero. */
    positiveP98?: number;
    negativeP02?: number;
  };
  /** Bounds of robust absolute-value support for signed-volume camera fitting. */
  absoluteFocusBounds?: [number, number, number, number, number, number] | null;
  /** World position of the strongest requested signed feature, computed off-thread. */
  featurePeakPosition?: [number, number, number] | null;
  /** Signed scalar value at featurePeakPosition. */
  featurePeakValue?: number | null;
  /** Dominant vertically connected diagnostic ridge in world metres. */
  vortexCenterline?: Float32Array;
  /** Source-to-display sampling history, rendered on-canvas by every 3D plot. */
  samplingProvenance: VtkSamplingProvenance;
  scalarTransform?: ScalarFieldTransform;
  structuralSupportBounds?: [number, number, number, number, number, number] | null;
}

export type VtkDatasetHandoffIntent = "render-only" | "shared-geometry";
export type ScalarFieldTransform = "none" | "horizontal-level-anomaly";

export interface ScalarPreparationRequest {
  background?: VolumeBackground;
  cropBackground?: boolean;
  /** CPU geometry retains NaN support directly when the GPU mask is omitted. */
  includeValidity?: boolean;
  reconstruction?: VolumeReconstruction;
  kind: "scalar";
  datasetHandoff?: VtkDatasetHandoffIntent;
  field: Npy3D;
  grid: GridSpacing;
  maxVoxels?: number;
  subtractField?: Npy3D;
  /** Re-evaluate the same piecewise-trilinear field on a denser display
   * lattice. This changes presentation sampling, not model resolution. */
  refineForDisplay?: boolean;
  /** Select the positive maximum or negative minimum for storm-detail location. */
  featureSign?: 1 | -1;
  /** Optional physical-height band for the feature extremum. */
  featureZRangeM?: readonly [number, number];
  scalarTransform?: ScalarFieldTransform;
  structuralSupportRangeFraction?: number;
  excludePhysicalBoundsM?: {
    lower: readonly [number, number, number];
    upper: readonly [number, number, number];
  };
}

export interface VectorPreparationRequest {
  kind: "vector";
  datasetHandoff?: VtkDatasetHandoffIntent;
  fields: [Npy3D, Npy3D, Npy3D];
  grid: GridSpacing;
  maxVoxels?: number;
  /** Locate a near-surface core directly in the prepared vertical component. */
  detectVerticalVortexCore?: boolean;
}

export interface VelocityDiagnosticPreparationRequest {
  includeValidity?: boolean;
  kind: "velocity-diagnostic";
  datasetHandoff?: VtkDatasetHandoffIntent;
  diagnostic: VelocityGradientDiagnostic;
  fields: [Npy3D, Npy3D, Npy3D];
  grid: GridSpacing;
  maxVoxels?: number;
  refineForDisplay?: boolean;
}

export type VelocityGradientDiagnostic = "q-criterion" | "lambda-2" | "swirling-strength";

export interface CloudPreparationRequest {
  kind: "cloud";
  datasetHandoff?: VtkDatasetHandoffIntent;
  field: Npy3D;
  subtractField?: Npy3D;
  grid: GridSpacing;
  volumeMaxVoxels: number;
  surfaceMaxVoxels: number;
  fidelityMode?: RenderFidelityMode;
  detailSeed?: number;
  /** Accepted display metadata does not change scalar cloud preparation. */
  flowFields?: [Npy3D, Npy3D, Npy3D];
  simulationTimeS?: number;
  /** Earth-coordinate box owned by a finer level. Samples inside it become
   * transparent in this coarse context volume, preventing double rendering. */
  excludePhysicalBoundsM?: {
    lower: readonly [number, number, number];
    upper: readonly [number, number, number];
  };
}

export interface PreparedCloudImages {
  volume: PreparedVtkImage;
  surface: PreparedVtkImage;
  /** Display-only trilinear subdivisions used for the cloud surface. */
  surfaceRefinement: [number, number, number];
  /** Spacing of the prepared source before display refinement. */
  surfaceSourceSpacing: [number, number, number];
  /** Dense-cloud world bounds used only for the initial camera fit. */
  surfaceFocusBounds: [number, number, number, number, number, number] | null;
  /** Bounds of every cloud sample that the minimum UI floor can make visible.
   * Camera clipping uses this wider envelope; initial framing still uses the
   * denser focus bounds above. */
  surfaceSupportBounds: [number, number, number, number, number, number] | null;
  /** Condensate-weighted world centre used to seed a native storm window when tracker output is absent. */
  cloudCenter: [number, number, number];
  /** Present only when Presentation added a non-scientific surface density field. */
  presentationDetail: CloudPresentationDetailMetadata | null;
}

export interface PreparedVectorImage extends PreparedVtkImage {
  stormCenter: [number, number];
  speed95: number;
  /** ROI-independent integration reference from the selected storm column. */
  seedColumnSpeedMaximum?: number;
  /** Magnitude-ranked, spatially distributed world-space seed positions. */
  featureSeeds: Float32Array;
  /** Grid-relative core; the scene adds the prepared world origin to seed it. */
  vortexCore: VortexCore | null;
}

export type VtkPreparationRequest =
  | ScalarPreparationRequest
  | VectorPreparationRequest
  | VelocityDiagnosticPreparationRequest
  | CloudPreparationRequest;
export type VtkPreparationResult =
  | PreparedVtkImage
  | PreparedVectorImage
  | PreparedCloudImages;

export function numericalCloudPreparationRequest(request: CloudPreparationRequest): CloudPreparationRequest {
  return { kind: "cloud", field: request.field, subtractField: request.subtractField, grid: request.grid,
    volumeMaxVoxels: request.volumeMaxVoxels, surfaceMaxVoxels: request.surfaceMaxVoxels,
    datasetHandoff: request.datasetHandoff, excludePhysicalBoundsM: request.excludePhysicalBoundsM };
}

export function prepareVtkRequest(request: ScalarPreparationRequest): PreparedVtkImage;
export function prepareVtkRequest(request: VectorPreparationRequest): PreparedVectorImage;
export function prepareVtkRequest(request: VelocityDiagnosticPreparationRequest): PreparedVtkImage;
export function prepareVtkRequest(request: CloudPreparationRequest): PreparedCloudImages;
export function prepareVtkRequest(request: VtkPreparationRequest): VtkPreparationResult;
export function prepareVtkRequest(request: VtkPreparationRequest): VtkPreparationResult {
  if (request.kind === "cloud") {
    const field = request.subtractField
      ? subtractNonNegative(request.field, request.subtractField)
      : request.field;
    const volumeSource = fieldToCartesianGrid(field, request.grid, {
      maxVoxels: request.volumeMaxVoxels,
      preserveMissing: true,
    });
    const rawVolumeRefinement = refineCartesianGridLinear(volumeSource, {
      targetSpacingM: 250,
      maxFactor: 8,
      maxVoxels: request.volumeMaxVoxels,
    });
    const volumeRefinement = {
      ...rawVolumeRefinement,
      grid: excludeFinerLevelBounds(rawVolumeRefinement.grid, request.excludePhysicalBoundsM),
    };
    // The common scientific/fine-grid path uses identical volume and surface
    // ceilings. Reuse its Cartesian preparation and packed VTK buffer instead
    // of retaining two 64 MB copies of the same scalar lattice.
    const sharedCloudLattice = request.volumeMaxVoxels === request.surfaceMaxVoxels;
    const surfaceSource = sharedCloudLattice
      ? volumeSource
      : fieldToCartesianGrid(field, request.grid, {
          maxVoxels: request.surfaceMaxVoxels,
          preserveMissing: true,
        });
    const rawSurfaceRefinement = sharedCloudLattice
      ? volumeRefinement
      : refineCartesianGridLinear(surfaceSource, {
          targetSpacingM: 250,
          maxFactor: 8,
          maxVoxels: request.surfaceMaxVoxels,
        });
    const surfaceRefinement = sharedCloudLattice
      ? volumeRefinement
      : {
          ...rawSurfaceRefinement,
          grid: excludeFinerLevelBounds(rawSurfaceRefinement.grid, request.excludePhysicalBoundsM),
        };
    const source = validSourceGeometry(field, request.grid);
    const volumeProvenance = vtkSamplingProvenance(
      source,
      volumeSource,
      volumeRefinement.grid,
      volumeRefinement.factors,
      reductionKind(volumeSource, field),
    );
    const surfaceProvenance = vtkSamplingProvenance(
      source,
      surfaceSource,
      surfaceRefinement.grid,
      surfaceRefinement.factors,
      reductionKind(surfaceSource, field),
    );
    const volumeValues = cartesianGridToVtkScalars(volumeRefinement.grid);
    const surfaceValues = surfaceRefinement.grid === volumeRefinement.grid
      ? volumeValues
      : cartesianGridToVtkScalars(surfaceRefinement.grid);
    const volume = preparedImage(volumeRefinement.grid, volumeValues, volumeProvenance);
    return {
      volume,
      surface: surfaceValues === volumeValues ? { ...volume, samplingProvenance: surfaceProvenance } : preparedImage(
        surfaceRefinement.grid,
        surfaceValues,
        surfaceProvenance,
      ),
      surfaceRefinement: surfaceRefinement.factors,
      surfaceSourceSpacing: surfaceSource.spacing,
      surfaceFocusBounds: scalarBoundsAbove(surfaceRefinement.grid, 0.0005),
      surfaceSupportBounds: scalarBoundsAbove(surfaceRefinement.grid, 0.00005),
      cloudCenter: cloudCondensateCentroid(surfaceSource),
      presentationDetail: null,
    };
  }
  if (request.kind === "scalar") {
    const field = request.subtractField
      ? subtractNonNegative(request.field, request.subtractField)
      : request.field;
    const target = request.reconstruction;
    const source = target && request.grid.terrainFollowing ? terrainWindowToCartesianField(field, request.grid, {
      origin: target.origin[2], spacing: target.spacing[2], count: target.shape[2],
    }) : null;
    const targetValues = target ? sampleReconstruction(source?.field ?? field,
      source?.scalarCoordinatesM ?? reconstructionGrid(field, request.grid)!.scalarCoordinatesM!, target) : null;
    const rawCart = target && targetValues ? cartesianBoxToImageData({ shape: target.shape, data: targetValues },
      ...target.spacing, { originM: target.origin, preserveMissing: true }) : fieldToCartesianGrid(field, request.grid, {
      maxVoxels: request.maxVoxels,
      preserveMissing: true,
    });
    const scalarTransform = request.scalarTransform ?? "none";
    const transformedCart = scalarTransform === "horizontal-level-anomaly"
      ? horizontalLevelAnomaly(rawCart)
      : rawCart;
    const cart = excludeFinerLevelBounds(transformedCart, request.excludePhysicalBoundsM);
    const refinement = request.refineForDisplay
      ? refineCartesianGridLinear(cart, {
          targetSpacingM: 250,
          maxFactor: 8,
          maxVoxels: request.maxVoxels ?? CARTESIAN_VOXEL_CAP_VOLUME,
        })
      : { grid: cart, factors: [1, 1, 1] as [number, number, number] };
    const display = refinement.grid;
    const stats = scalarVolumeStats(cart.data);
    const { p2, p98 } = stats;
    const featurePeak = request.featureSign
      ? scalarSignedPeak(cart, request.featureSign, request.featureZRangeM)
      : null;
    const supportFraction = request.structuralSupportRangeFraction;
    const supportThreshold = supportFraction != null
        && Number.isFinite(supportFraction)
        && supportFraction >= 0
        && supportFraction <= 1
        && Number.isFinite(p2)
        && Number.isFinite(p98)
        && p98 > p2
      ? p2 + supportFraction * (p98 - p2)
      : null;
    const prepared: PreparedVtkImage = {
      ...preparedImage(
        display,
        cartesianGridToVtkScalars(display),
        vtkSamplingProvenance(
          validSourceGeometry(field, request.grid),
          cart,
          display,
          refinement.factors,
          reductionKind(cart, field),
        ),
        request.includeValidity !== false,
      ),
      stats,
      absoluteFocusBounds: scalarMagnitudeBoundsAbove(
        display,
        Math.max(Math.abs(p2), Math.abs(p98)) * 0.2,
      ),
      featurePeakPosition: request.featureSign ? featurePeak?.position ?? null : undefined,
      featurePeakValue: request.featureSign ? featurePeak?.value ?? null : undefined,
      scalarTransform,
      structuralSupportBounds: supportThreshold == null
        ? undefined
        : scalarBoundsAbove(display, supportThreshold),
    };
    return request.background ? focusedVolume(prepared, request.background, request.cropBackground === true) : prepared;
  }
  const cart = vectorFieldToCartesianGrid(
    request.fields[0],
    request.fields[1],
    request.fields[2],
    request.grid,
    { maxVoxels: request.maxVoxels, preserveMissing: true },
  );
  if (request.kind === "velocity-diagnostic") {
    const diagnostic = velocityGradientDiagnostic(cart, request.diagnostic);
    const stats = scalarStats(diagnostic.data);
    const refinement = request.refineForDisplay
      ? refineCartesianGridLinear(diagnostic, {
          targetSpacingM: 250,
          maxFactor: 8,
          maxVoxels: request.maxVoxels ?? CARTESIAN_VOXEL_CAP_VOLUME,
        })
      : { grid: diagnostic, factors: [1, 1, 1] as [number, number, number] };
    const display = refinement.grid;
    return {
      ...preparedImage(
        display,
        cartesianGridToVtkScalars(display),
        vtkSamplingProvenance(
          validSourceGeometry(request.fields[0], request.grid),
          diagnostic,
          display,
          refinement.factors,
          reductionKind(diagnostic, request.fields[0]),
        ),
        request.includeValidity !== false,
      ),
      stats,
      absoluteFocusBounds: scalarMagnitudeBoundsAbove(
        display,
        Math.max(Math.abs(stats.p2), Math.abs(stats.p98)) * 0.2,
      ),
      vortexCenterline: dominantVortexCenterline(display, request.diagnostic),
    };
  }
  const stormCenter = strongestVerticalColumn(cart);
  const speed95 = cartesianVectorSpeedQuantile(cart, 0.95);
  const seedColumnSpeedMaximum = vectorColumnMaximumSpeed(cart, stormCenter);
  const featureSeeds = rankedFeatureSeeds(cart);
  const vtkVectors = cartesianGridToVtkVectorsInPlace(cart);
  return {
    ...preparedImage(
      cart,
      vtkVectors,
      vtkSamplingProvenance(
        validSourceGeometry(request.fields[0], request.grid),
        cart,
        cart,
        [1, 1, 1],
        reductionKind(cart, request.fields[0]),
      ),
      false,
    ),
    stormCenter,
    speed95,
    seedColumnSpeedMaximum,
    featureSeeds,
    vortexCore: request.detectVerticalVortexCore
      ? findInterleavedVerticalRotationMaximum(
          vtkVectors,
          { nx: cart.shape[0], ny: cart.shape[1], nz: cart.shape[2] },
          { dx: cart.spacing[0], dy: cart.spacing[1], dz: cart.spacing[2] },
        )
      : null,
  };
}

export function horizontalLevelAnomaly(source: CartesianVoxelGrid): CartesianVoxelGrid {
  const [nx, ny, nz] = source.shape;
  const means = new Float64Array(nz);
  const counts = new Uint32Array(nz);
  for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y < ny; y += 1) {
      const column = (x * ny + y) * nz;
      for (let z = 0; z < nz; z += 1) {
        const value = source.data[column + z];
        if (!Number.isFinite(value)) continue;
        means[z] += value;
        counts[z] += 1;
      }
    }
  }
  for (let z = 0; z < nz; z += 1) means[z] = counts[z] > 0 ? means[z] / counts[z] : 0;
  const data = new Float32Array(source.data.length);
  for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y < ny; y += 1) {
      const column = (x * ny + y) * nz;
      for (let z = 0; z < nz; z += 1) {
        const value = source.data[column + z];
        data[column + z] = Number.isFinite(value) ? value - means[z] : Number.NaN;
      }
    }
  }
  return { ...source, data };
}

export function applyPreparedGlobalHorizontalLevelAnomaly(
  inputs: readonly PreparedVtkImage[],
): PreparedVtkImage[] {
  if (inputs.length === 0) return [];
  const commonHorizontalSpacing = inputs.every((prepared) =>
    prepared.spacing[0] === inputs[0].spacing[0]
      && prepared.spacing[1] === inputs[0].spacing[1]);
  const levels = new Map<string, Map<string, Array<{
    prepared: PreparedVtkImage;
    x: number;
    z: number;
  }>>>();
  const orderedInputs = [...inputs].sort((left, right) =>
    left.origin[1] - right.origin[1]
      || left.origin[0] - right.origin[0]
      || left.origin[2] - right.origin[2]);
  for (const prepared of orderedInputs) {
    const [nx, ny, nz] = prepared.shape;
    if (prepared.values.length !== nx * ny * nz) {
      throw new Error("Prepared anomaly brick has invalid scalar storage");
    }
    const horizontalWeight = commonHorizontalSpacing
      ? 1
      : prepared.spacing[0] * prepared.spacing[1];
    if (!(horizontalWeight > 0) || !Number.isFinite(horizontalWeight)) {
      throw new Error("Prepared anomaly brick has invalid horizontal support");
    }
    for (let z = 0; z < nz; z += 1) {
      const key = (prepared.origin[2] + z * prepared.spacing[2]).toPrecision(12);
      const columns = levels.get(key) ?? new Map<string, Array<{
        prepared: PreparedVtkImage;
        x: number;
        z: number;
      }>>();
      for (let x = 0; x < nx; x += 1) {
        const xKey = (prepared.origin[0] + x * prepared.spacing[0]).toPrecision(12);
        const segments = columns.get(xKey) ?? [];
        segments.push({ prepared, x, z });
        columns.set(xKey, segments);
      }
      levels.set(key, columns);
    }
  }
  const means = new Map([...levels].map(([key, columns]) => {
    let sum = 0;
    let weight = 0;
    for (const [, segments] of [...columns].sort(([left], [right]) =>
      Number(left) - Number(right))) {
      for (const { prepared, x, z } of segments.sort((left, right) =>
        left.prepared.origin[1] - right.prepared.origin[1])) {
        const [nx, ny] = prepared.shape;
        const horizontalWeight = commonHorizontalSpacing
          ? 1
          : prepared.spacing[0] * prepared.spacing[1];
        for (let y = 0; y < ny; y += 1) {
          const index = x + nx * (y + ny * z);
          if (prepared.validity?.[index] === 0) continue;
          const value = prepared.values[index];
          if (!Number.isFinite(value)) continue;
          sum += value * horizontalWeight;
          weight += horizontalWeight;
        }
      }
    }
    return [key, weight > 0 ? sum / weight : 0];
  }));
  const transformed = inputs.map((prepared) => {
    const [nx, ny, nz] = prepared.shape;
    const values = new Float32Array(prepared.values.length);
    for (let z = 0; z < nz; z += 1) {
      const key = (prepared.origin[2] + z * prepared.spacing[2]).toPrecision(12);
      const mean = means.get(key) ?? 0;
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const index = x + nx * (y + ny * z);
          const value = prepared.values[index];
          values[index] = prepared.validity?.[index] !== 0 && Number.isFinite(value) ? value - mean : 0;
        }
      }
    }
    return { ...prepared, values, scalarTransform: "horizontal-level-anomaly" as const };
  });
  const stats = scalarVolumeBatchStats(transformed);
  const threshold = Math.max(Math.abs(stats.p2), Math.abs(stats.p98)) * 0.2;
  return transformed.map((prepared) => {
    const [nx, ny, nz] = prepared.shape;
    let minX = nx;
    let maxX = -1;
    let minY = ny;
    let maxY = -1;
    let minZ = nz;
    let maxZ = -1;
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          if (Math.abs(prepared.values[x + nx * (y + ny * z)]) < threshold) continue;
          minX = Math.min(minX, x); maxX = Math.max(maxX, x);
          minY = Math.min(minY, y); maxY = Math.max(maxY, y);
          minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }
      }
    }
    const absoluteFocusBounds = maxX < 0 ? null : [
      prepared.origin[0] + minX * prepared.spacing[0],
      prepared.origin[0] + maxX * prepared.spacing[0],
      prepared.origin[1] + minY * prepared.spacing[1],
      prepared.origin[1] + maxY * prepared.spacing[1],
      prepared.origin[2] + minZ * prepared.spacing[2],
      prepared.origin[2] + maxZ * prepared.spacing[2],
    ] as [number, number, number, number, number, number];
    return { ...prepared, stats, absoluteFocusBounds };
  });
}

export function prepareGlobalAnomalyBrickBatch(
  requests: readonly ScalarPreparationRequest[],
): PreparedVtkImage[] | null {
  if (requests.length === 0) return [];
  const geometries = requests.map((request) => request.field.windowGeometry);
  if (geometries.some((geometry) => !geometry?.sourceShape
      || !geometry.origin || !geometry.extent || !geometry.stride)) return null;
  const windows = geometries.map((geometry) => ({
    aggregation: geometry!.aggregation ?? "cell_mean",
    origin: [...geometry!.origin!], extent: [...geometry!.extent!], stride: [...geometry!.stride!],
  })) as FieldLoadWindow[];
  if (windows.some((window) => window.stride.some((value, axis) =>
    value !== windows[0].stride[axis]))) throw new Error("Global anomaly bricks disagree on stride");
  const assembled = assembleWindowedField(requests.map((request) => request.field), windows);
  if (assembled.shape.length !== 3) throw new Error("Global anomaly assembly is not 3-D");
  const targets = requests.map((request) => request.reconstruction);
  const origin = targets[0] ? [0, 1, 2].map((axis) => Math.min(...targets.map((target) => target!.origin[axis]))) as [number, number, number] : null;
  const reconstruction = origin ? { origin, spacing: targets[0]!.spacing,
    shape: origin.map((value, axis) => Math.round((Math.max(...targets.map((target) => target!.origin[axis] + (target!.shape[axis] - 1) * target!.spacing[axis])) - value) / targets[0]!.spacing[axis]) + 1) as [number, number, number] } : undefined;
  const global = prepareVtkRequest({
    ...requests[0], reconstruction, field: assembled as Npy3D, scalarTransform: "horizontal-level-anomaly",
    excludePhysicalBoundsM: undefined, refineForDisplay: false,
  });
  const overall = [0, 1, 2].map((axis) => Math.min(...windows.map((window) => window.origin[axis])));
  const [gnx, gny] = global.shape;
  return requests.map((request, index) => {
    const shape = request.reconstruction?.shape ?? request.field.shape;
    const offset = request.reconstruction ? request.reconstruction.origin.map((value, axis) => Math.round((value - global.origin[axis]) / global.spacing[axis]))
      : windows[index].origin.map((value, axis) => (value - overall[axis]) / windows[0].stride[axis]);
    if (!offset.every(Number.isSafeInteger)) throw new Error("Global anomaly brick is misaligned");
    const values = new Float32Array(shape[0] * shape[1] * shape[2]);
    const validity = global.validity ? new Uint8Array(values.length) : undefined;
    for (let z = 0; z < shape[2]; z += 1) for (let y = 0; y < shape[1]; y += 1) {
      for (let x = 0; x < shape[0]; x += 1) {
        const destination = x + shape[0] * (y + shape[1] * z);
        const source = x + offset[0] + gnx * (y + offset[1] + gny * (z + offset[2]));
        values[destination] = global.values[source];
        if (validity) validity[destination] = global.validity![source];
      }
    }
    return excludePreparedScalarBounds({
      ...global, shape: [...shape], values, validity,
      origin: [global.origin[0] + offset[0] * global.spacing[0],
        global.origin[1] + offset[1] * global.spacing[1],
        global.origin[2] + offset[2] * global.spacing[2]],
    }, request.excludePhysicalBoundsM);
  });
}

export function excludePreparedScalarBounds(
  prepared: PreparedVtkImage,
  bounds: ScalarPreparationRequest["excludePhysicalBoundsM"],
): PreparedVtkImage {
  if (!bounds) return prepared;
  if (![...bounds.lower, ...bounds.upper].every(Number.isFinite)
      || bounds.lower.some((value, axis) => value >= bounds.upper[axis])) {
    throw new Error("Prepared scalar exclusion bounds are invalid");
  }
  const [nx, ny, nz] = prepared.shape;
  const values = prepared.values.slice();
  for (let z = 0; z < nz; z += 1) {
    const worldZ = prepared.origin[2] + z * prepared.spacing[2];
    if (worldZ < bounds.lower[2] || worldZ >= bounds.upper[2]) continue;
    for (let y = 0; y < ny; y += 1) {
      const worldY = prepared.origin[1] + y * prepared.spacing[1];
      if (worldY < bounds.lower[1] || worldY >= bounds.upper[1]) continue;
      for (let x = 0; x < nx; x += 1) {
        const worldX = prepared.origin[0] + x * prepared.spacing[0];
        if (worldX >= bounds.lower[0] && worldX < bounds.upper[0]) {
          values[x + nx * (y + ny * z)] = 0;
        }
      }
    }
  }
  return { ...prepared, values };
}

/** Mask only the display copy owned by the preparation worker. Scientific
 * source arrays remain immutable and the finer level retains exact values. */
function excludeFinerLevelBounds(
  grid: CartesianVoxelGrid,
  bounds: CloudPreparationRequest["excludePhysicalBoundsM"]
    | ScalarPreparationRequest["excludePhysicalBoundsM"],
): CartesianVoxelGrid {
  if (!bounds) return grid;
  if (![...bounds.lower, ...bounds.upper].every(Number.isFinite)
      || bounds.lower.some((value, axis) => value >= bounds.upper[axis])) {
    throw new Error("Fine-level exclusion bounds are invalid");
  }
  const [nx, ny, nz] = grid.shape;
  const x0 = Math.max(0, Math.ceil((bounds.lower[0] - grid.origin[0]) / grid.spacing[0]));
  const x1 = Math.min(nx, Math.ceil((bounds.upper[0] - grid.origin[0]) / grid.spacing[0]));
  const y0 = Math.max(0, Math.ceil((bounds.lower[1] - grid.origin[1]) / grid.spacing[1]));
  const y1 = Math.min(ny, Math.ceil((bounds.upper[1] - grid.origin[1]) / grid.spacing[1]));
  const z0 = Math.max(0, Math.ceil((bounds.lower[2] - grid.origin[2]) / grid.spacing[2]));
  const z1 = Math.min(nz, Math.ceil((bounds.upper[2] - grid.origin[2]) / grid.spacing[2]));
  if (x0 >= x1 || y0 >= y1 || z0 >= z1) return grid;
  const data = grid.data.slice();
  for (let i = x0; i < x1; i += 1) {
    for (let j = y0; j < y1; j += 1) {
      data.fill(0, (i * ny + j) * nz + z0, (i * ny + j) * nz + z1);
    }
  }
  return { ...grid, data };
}

/** Full three-dimensional Q criterion from the physical Cartesian velocity
 * gradient. Positive Q means rotation dominates strain. Derivatives are
 * centered inside the domain and one-sided at its physical boundary. */
export function velocityQCriterion(vectors: CartesianVoxelGrid): CartesianVoxelGrid {
  return velocityGradientDiagnostic(vectors, "q-criterion");
}

export function velocityGradientDiagnostic(
  vectors: CartesianVoxelGrid,
  diagnostic: VelocityGradientDiagnostic,
): CartesianVoxelGrid {
  const [nx, ny, nz] = vectors.shape;
  const spacing = vectors.spacing;
  const output = new Float32Array(nx * ny * nz);
  const sample = (x: number, y: number, z: number, component: number) =>
    vectors.data[((x * ny + y) * nz + z) * 3 + component];
  const derivative = (
    x: number,
    y: number,
    z: number,
    component: number,
    axis: 0 | 1 | 2,
  ) => {
    if (!Number.isFinite(sample(x, y, z, component))) return Number.NaN;
    const coordinates = [x, y, z] as [number, number, number];
    const counts = [nx, ny, nz];
    if (counts[axis] <= 1 || !(spacing[axis] > 0)) return 0;
    const lower = Math.max(0, coordinates[axis] - 1);
    const upper = Math.min(counts[axis] - 1, coordinates[axis] + 1);
    const lowerCoordinates = [...coordinates] as [number, number, number];
    const upperCoordinates = [...coordinates] as [number, number, number];
    lowerCoordinates[axis] = lower;
    upperCoordinates[axis] = upper;
    return (
      sample(upperCoordinates[0], upperCoordinates[1], upperCoordinates[2], component) -
      sample(lowerCoordinates[0], lowerCoordinates[1], lowerCoordinates[2], component)
    ) / ((upper - lower) * spacing[axis]);
  };
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        const gradient = new Float64Array(9);
        for (let component = 0; component < 3; component++) {
          for (let axis = 0; axis < 3; axis++) {
            gradient[component * 3 + axis] = derivative(
              x,
              y,
              z,
              component,
              axis as 0 | 1 | 2,
            );
          }
        }
        output[(x * ny + y) * nz + z] = diagnostic === "q-criterion"
          ? qCriterionFromGradient(gradient)
          : diagnostic === "lambda-2"
            ? lambda2FromGradient(gradient)
            : swirlingStrengthFromGradient(gradient);
      }
    }
  }
  return { ...vectors, data: output };
}

export function qCriterionFromGradient(gradient: ArrayLike<number>): number {
  if (gradient.length < 9) return Number.NaN;
  let strainNormSquared = 0;
  let rotationNormSquared = 0;
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      const a = gradient[row * 3 + column];
      const transpose = gradient[column * 3 + row];
      if (!Number.isFinite(a) || !Number.isFinite(transpose)) return Number.NaN;
      const strain = 0.5 * (a + transpose);
      const rotation = 0.5 * (a - transpose);
      strainNormSquared += strain * strain;
      rotationNormSquared += rotation * rotation;
    }
  }
  return 0.5 * (rotationNormSquared - strainNormSquared);
}

/** Jeong-Hussain lambda-2 criterion: the middle eigenvalue of S²+Ω². */
export function lambda2FromGradient(gradient: ArrayLike<number>): number {
  if (gradient.length < 9 || Array.from({ length: 9 }, (_, i) => gradient[i]).some((v) => !Number.isFinite(v))) {
    return Number.NaN;
  }
  const s = new Float64Array(9);
  const omega = new Float64Array(9);
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      const index = row * 3 + column;
      const transpose = gradient[column * 3 + row];
      s[index] = 0.5 * (gradient[index] + transpose);
      omega[index] = 0.5 * (gradient[index] - transpose);
    }
  }
  const tensor = new Float64Array(9);
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      let value = 0;
      for (let inner = 0; inner < 3; inner++) {
        value += s[row * 3 + inner] * s[inner * 3 + column]
          + omega[row * 3 + inner] * omega[inner * 3 + column];
      }
      tensor[row * 3 + column] = value;
    }
  }
  return symmetricEigenvalues3(tensor)[1];
}

/** Imaginary part of the complex-conjugate velocity-gradient eigenpair. */
export function swirlingStrengthFromGradient(gradient: ArrayLike<number>): number {
  if (gradient.length < 9 || Array.from({ length: 9 }, (_, i) => gradient[i]).some((v) => !Number.isFinite(v))) {
    return Number.NaN;
  }
  const a = gradient;
  const trace = a[0] + a[4] + a[8];
  const invariant2 = a[0] * a[4] + a[0] * a[8] + a[4] * a[8]
    - a[1] * a[3] - a[2] * a[6] - a[5] * a[7];
  const determinant = a[0] * (a[4] * a[8] - a[5] * a[7])
    - a[1] * (a[3] * a[8] - a[5] * a[6])
    + a[2] * (a[3] * a[7] - a[4] * a[6]);
  const p = invariant2 - trace * trace / 3;
  const q = -2 * trace * trace * trace / 27 + trace * invariant2 / 3 - determinant;
  const discriminant = q * q / 4 + p * p * p / 27;
  const tolerance = 1e-18 * Math.max(1, Math.abs(q * q), Math.abs(p * p * p));
  if (!(discriminant > tolerance)) return 0;
  const root = Math.sqrt(discriminant);
  const u = Math.cbrt(-q / 2 + root);
  const v = Math.cbrt(-q / 2 - root);
  return Math.sqrt(3) * 0.5 * Math.abs(u - v);
}

/**
 * Extract one dominant, vertically connected ridge from a velocity-gradient
 * diagnostic. This is deliberately a resolved-grid feature: it begins at the
 * global strongest rotational sample, walks adjacent levels inside a bounded
 * physical search radius, and never bridges a level with no qualifying core.
 */
export function dominantVortexCenterline(
  grid: CartesianVoxelGrid,
  diagnostic: VelocityGradientDiagnostic,
): Float32Array {
  const [nx, ny, nz] = grid.shape;
  if (nx < 1 || ny < 1 || nz < 2) return new Float32Array();
  const support = nx >= 2 && ny >= 2 && hasMissingValues(grid.data)
    ? finiteVectorSupport(grid.data, grid.shape, grid.origin, grid.spacing, 1, "c") : null;
  const physicalPoint = (point: readonly number[]) => point.map((value, axis) => grid.origin[axis] + value * grid.spacing[axis]);
  const score = (index: number) => {
    const value = grid.data[index];
    if (!Number.isFinite(value)) return -Infinity;
    return diagnostic === "lambda-2" ? -value : value;
  };
  let peakScore = 0;
  let peak: [number, number, number] | null = null;
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        const candidate = score((x * ny + y) * nz + z);
        if (candidate > peakScore) {
          peakScore = candidate;
          peak = [x, y, z];
        }
      }
    }
  }
  if (!peak || !(peakScore > 0)) return new Float32Array();
  const minimumScore = peakScore * 0.1;
  const radiusM = Math.max(2_000, 3 * Math.max(grid.spacing[0], grid.spacing[1]));
  const radiusX = Math.max(1, Math.ceil(radiusM / grid.spacing[0]));
  const radiusY = Math.max(1, Math.ceil(radiusM / grid.spacing[1]));
  const points = new Map<number, [number, number, number]>();
  points.set(peak[2], peak);
  const walk = (direction: -1 | 1) => {
    let previous = peak!;
    for (let z = peak![2] + direction; z >= 0 && z < nz; z += direction) {
      let bestScore = minimumScore;
      let best: [number, number, number] | null = null;
      for (let x = Math.max(0, previous[0] - radiusX);
        x <= Math.min(nx - 1, previous[0] + radiusX); x++) {
        for (let y = Math.max(0, previous[1] - radiusY);
          y <= Math.min(ny - 1, previous[1] + radiusY); y++) {
          const dx = (x - previous[0]) * grid.spacing[0];
          const dy = (y - previous[1]) * grid.spacing[1];
          if (dx * dx + dy * dy > radiusM * radiusM) continue;
          const candidate = score((x * ny + y) * nz + z);
          if (candidate > bestScore) {
            if (support && !support.connects(physicalPoint(previous), physicalPoint([x, y, z]))) continue;
            bestScore = candidate;
            best = [x, y, z];
          }
        }
      }
      if (!best) break;
      points.set(z, best);
      previous = best;
    }
  };
  walk(-1);
  walk(1);
  const ordered = [...points.values()].sort((left, right) => left[2] - right[2]);
  if (ordered.length < 2) return new Float32Array();
  const output = new Float32Array(ordered.length * 3);
  ordered.forEach((point, index) => {
    const previous = ordered[Math.max(0, index - 1)];
    const next = ordered[Math.min(ordered.length - 1, index + 1)];
    output[index * 3] = grid.origin[0] + (previous[0] + point[0] + next[0]) / 3 * grid.spacing[0];
    output[index * 3 + 1] = grid.origin[1] + (previous[1] + point[1] + next[1]) / 3 * grid.spacing[1];
    output[index * 3 + 2] = grid.origin[2] + point[2] * grid.spacing[2];
  });
  return output;
}

function symmetricEigenvalues3(input: ArrayLike<number>): [number, number, number] {
  const a = Float64Array.from(input);
  for (let sweep = 0; sweep < 12; sweep++) {
    let p = 0;
    let q = 1;
    let largest = Math.abs(a[1]);
    for (const [row, column] of [[0, 2], [1, 2]] as const) {
      const value = Math.abs(a[row * 3 + column]);
      if (value > largest) { largest = value; p = row; q = column; }
    }
    if (largest <= 1e-14 * Math.max(1, Math.abs(a[0]), Math.abs(a[4]), Math.abs(a[8]))) break;
    const app = a[p * 3 + p];
    const aqq = a[q * 3 + q];
    const apq = a[p * 3 + q];
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    for (let k = 0; k < 3; k++) {
      if (k === p || k === q) continue;
      const akp = a[k * 3 + p];
      const akq = a[k * 3 + q];
      a[k * 3 + p] = a[p * 3 + k] = c * akp - s * akq;
      a[k * 3 + q] = a[q * 3 + k] = s * akp + c * akq;
    }
    a[p * 3 + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q * 3 + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p * 3 + q] = a[q * 3 + p] = 0;
  }
  return [a[0], a[4], a[8]].sort((left, right) => left - right) as [number, number, number];
}

/** Bounds of samples at or above a physical scalar threshold. Kept in the
 * preparation worker so camera framing never adds a full-volume UI scan. */
export function scalarBoundsAbove(
  grid: CartesianVoxelGrid,
  threshold: number,
): [number, number, number, number, number, number] | null {
  return scalarBoundsMatching(grid, (value) => value >= threshold);
}

export function scalarMagnitudeBoundsAbove(
  grid: CartesianVoxelGrid,
  threshold: number,
): [number, number, number, number, number, number] | null {
  return scalarBoundsMatching(grid, (value) => Math.abs(value) >= threshold);
}

/** Strongest signed sample in physical coordinates; stays in the preparation worker. */
export function scalarSignedPeakPosition(
  grid: CartesianVoxelGrid,
  sign: 1 | -1,
  zRangeM?: readonly [number, number],
): [number, number, number] | null {
  return scalarSignedPeak(grid, sign, zRangeM)?.position ?? null;
}

export function scalarSignedPeak(
  grid: CartesianVoxelGrid,
  sign: 1 | -1,
  zRangeM?: readonly [number, number],
): { position: [number, number, number]; value: number } | null {
  let bestIndex = -1;
  let best = sign > 0 ? -Infinity : Infinity;
  const [, , nz] = grid.shape;
  for (let index = 0; index < grid.data.length; index += 1) {
    const z = index % nz;
    const heightM = grid.origin[2] + z * grid.spacing[2];
    if (zRangeM && (heightM < zRangeM[0] || heightM > zRangeM[1])) continue;
    const value = grid.data[index];
    if (!Number.isFinite(value)) continue;
    if ((sign > 0 && value > best) || (sign < 0 && value < best)) {
      best = value;
      bestIndex = index;
    }
  }
  if (bestIndex < 0 || (sign > 0 ? best <= 0 : best >= 0)) return null;
  const [nx, ny] = grid.shape;
  const x = Math.floor(bestIndex / (ny * nz));
  const y = Math.floor(bestIndex / nz) % ny;
  const z = bestIndex % nz;
  if (x >= nx) return null;
  return {
    position: [
      grid.origin[0] + x * grid.spacing[0],
      grid.origin[1] + y * grid.spacing[1],
      grid.origin[2] + z * grid.spacing[2],
    ],
    value: best,
  };
}

function scalarBoundsMatching(
  grid: CartesianVoxelGrid,
  include: (value: number) => boolean,
): [number, number, number, number, number, number] | null {
  const [nx, ny, nz] = grid.shape;
  let minX = nx;
  let maxX = -1;
  let minY = ny;
  let maxY = -1;
  let minZ = nz;
  let maxZ = -1;
  let source = 0;
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++, source++) {
        if (!include(grid.data[source])) continue;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);
      }
    }
  }
  if (maxX < 0) return null;
  const [dx, dy, dz] = grid.spacing;
  const [ox, oy, oz] = grid.origin;
  // One display cell of breathing room avoids fitting exactly through the
  // outermost dense sample before the shared camera padding is applied.
  minX = Math.max(0, minX - 1);
  maxX = Math.min(nx - 1, maxX + 1);
  minY = Math.max(0, minY - 1);
  maxY = Math.min(ny - 1, maxY + 1);
  minZ = Math.max(0, minZ - 1);
  maxZ = Math.min(nz - 1, maxZ + 1);
  return [
    ox + minX * dx,
    ox + maxX * dx,
    oy + minY * dy,
    oy + maxY * dy,
    oz + minZ * dz,
    oz + maxZ * dz,
  ];
}

function preparedImage(
  cart: CartesianVoxelGrid,
  values: Float32Array,
  samplingProvenance: VtkSamplingProvenance,
  includeValidity = true,
): PreparedVtkImage {
  let validity: Uint8Array | undefined;
  if (includeValidity && values.some((value) => !Number.isFinite(value))) {
    validity = Uint8Array.from(values, (value) => Number.isFinite(value) ? 255 : 0);
    values = Float32Array.from(values, (value) => Number.isFinite(value) ? value : 0);
  }
  return {
    shape: cart.shape,
    spacing: cart.spacing,
    origin: cart.origin,
    values,
    validity,
    sourceNonFiniteCount: cart.sourceNonFiniteCount ?? 0,
    samplingProvenance,
  };
}

/**
 * How the sampled lattice was reduced from the source one.
 *
 * The x/y downsample is an area mean over each block, not a kept cell — `cartesianBoxToImageData`
 * records that as `area_mean_xy`. Reporting it as subsampling understates what happened to a
 * peak: a subsample keeps or misses a value, a mean always dilutes it.
 */
function reductionKind(
  sampled: { lod?: { aggregation: "area_mean_xy" } },
  field?: Npy3D,
): "subsample" | "cell-mean" | "linear-reflectivity-mean"
  | "block-max" | "block-min" | "magnitude-extreme" {
  if (field?.windowGeometry?.aggregation === "linear_reflectivity_mean") {
    return "linear-reflectivity-mean";
  }
  if (field?.windowGeometry?.aggregation === "cell_mean") return "cell-mean";
  if (field?.windowGeometry?.aggregation === "block_max") return "block-max";
  if (field?.windowGeometry?.aggregation === "block_min") return "block-min";
  if (field?.windowGeometry?.aggregation === "magnitude_extreme") {
    return "magnitude-extreme";
  }
  return sampled.lod?.aggregation === "area_mean_xy" ? "cell-mean" : "subsample";
}

function validSourceGeometry(field: Npy3D, grid: GridSpacing): {
  shape: [number, number, number];
  spacing: [number, number, number];
  coordinateResampling?: "linear-to-uniform" | "conservative-terrain-to-cartesian";
  coordinateResamplingAxes?: Array<"x" | "y" | "z">;
} {
  /*
   * A windowed read arrives already aggregated, so its own shape is the SAMPLED lattice, not
   * the source one. Reporting it as the source would compare a shape against itself, find no
   * reduction, and disclose a block mean as native resolution. The header's native shape is
   * carried on the array for exactly this comparison.
   */
  const windowSourceShape = field.windowGeometry?.sourceShape;
  const windowSourceExtent = field.windowGeometry?.extent;
  if (windowSourceShape?.length === 3) {
    const sourceShape = windowSourceExtent?.length === 3
      ? windowSourceExtent
      : windowSourceShape;
    const sourceOrigin = field.windowGeometry?.origin;
    const exactSpacing = [0, 1, 2].map((axis) => {
      const coordinates = grid.scalarCoordinatesM?.[axis];
      const start = sourceOrigin?.[axis] ?? 0;
      const count = sourceShape[axis];
      if (!coordinates || count <= 1 || start < 0 || start + count > coordinates.length) {
        return null;
      }
      const first = coordinates[start];
      const last = coordinates[start + count - 1];
      return Number.isFinite(first) && Number.isFinite(last) && last > first
        ? (last - first) / (count - 1)
        : null;
    });
    const dx = exactSpacing[0] ?? grid.drM ?? 1;
    return {
      shape: [sourceShape[0], sourceShape[1], sourceShape[2]],
      spacing: [
        dx,
        exactSpacing[1] ?? grid.dyM ?? cartesianDyMeters(grid, dx),
        exactSpacing[2] ?? grid.dzM ?? 1,
      ],
      ...(grid.terrainFollowing ? {
        coordinateResampling: "conservative-terrain-to-cartesian" as const,
        coordinateResamplingAxes: ["z" as const],
      } : grid.scalarCoordinatesM?.some((values) => values.length > 2 && values.some((value, i) => Math.abs(value - values[0] - i * (values[1] - values[0])) > Math.abs(values[1] - values[0]) * 1e-7)) ? {
        coordinateResampling: "linear-to-uniform" as const,
        coordinateResamplingAxes: (["x", "y", "z"] as const).filter((_, axis) => {
          const values = grid.scalarCoordinatesM![axis];
          return values.length > 2 && values.some((value, i) => Math.abs(value - values[0] - i * (values[1] - values[0])) > Math.abs(values[1] - values[0]) * 1e-7);
        }),
      } : {}),
    };
  }
  const cropped = cropFieldToValidRange(field, grid);
  const valid = cropped.field;
  const declaredSpacing = [grid.drM, grid.dyM, grid.dzM];
  const axes = ["x", "y", "z"] as const;
  const validCoordinates = axes.map((_, axis) => {
    const source = grid.scalarCoordinatesM?.[axis];
    return source?.length === field.shape[axis]
      ? source.slice(cropped.lower[axis], cropped.lower[axis] + valid.shape[axis])
      : null;
  });
  const meanSpacing = validCoordinates.map((coordinates) =>
    coordinates && coordinates.length > 1
      ? (coordinates.at(-1)! - coordinates[0]) / (coordinates.length - 1)
      : null
  );
  const dx = meanSpacing[0] ?? grid.drM ?? 1;
  const dy = meanSpacing[1] ?? grid.dyM ?? cartesianDyMeters(grid, dx);
  const dz = meanSpacing[2] ?? grid.dzM ?? 1;
  const coordinateResamplingAxes = axes.filter((_, axis) =>
    validCoordinates[axis] != null && (declaredSpacing[axis] == null || validCoordinates[axis]!.some((value, index, values) => Math.abs(value - values[0] - index * (meanSpacing[axis] ?? 0)) > Math.abs(meanSpacing[axis] ?? 1) * 1e-7))
  );
  return {
    shape: valid.shape,
    spacing: [dx, dy, dz],
    ...(coordinateResamplingAxes.length > 0
      ? {
          coordinateResampling: "linear-to-uniform" as const,
          coordinateResamplingAxes: [...coordinateResamplingAxes],
        }
      : {}),
  };
}

/** World-space column containing the strongest absolute vertical component. */
export function strongestVerticalColumn(cart: CartesianVoxelGrid): [number, number] {
  const [nx, ny, nz] = cart.shape;
  const [dx, dy] = cart.spacing;
  const [ox, oy] = cart.origin;
  let best = -1;
  let bx = (nx - 1) / 2;
  let by = (ny - 1) / 2;
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const base = (ix * ny + iy) * nz;
      for (let iz = 0; iz < nz; iz++) {
        const value = cart.data[(base + iz) * 3 + 2];
        const magnitude = Math.abs(value);
        if (magnitude > best) {
          best = magnitude;
          bx = ix;
          by = iy;
        }
      }
    }
  }
  return [ox + bx * dx, oy + by * dy];
}

export function vectorColumnMaximumSpeed(
  cart: CartesianVoxelGrid,
  worldColumn: readonly [number, number],
): number {
  const [nx, ny, nz] = cart.shape;
  const [dx, dy] = cart.spacing;
  const [ox, oy] = cart.origin;
  const ix = Math.max(0, Math.min(nx - 1, Math.round((worldColumn[0] - ox) / dx)));
  const iy = Math.max(0, Math.min(ny - 1, Math.round((worldColumn[1] - oy) / dy)));
  const base = (ix * ny + iy) * nz;
  let maximum = 0;
  for (let iz = 0; iz < nz; iz += 1) {
    const offset = (base + iz) * 3;
    const u = cart.data[offset];
    const v = cart.data[offset + 1];
    const w = cart.data[offset + 2];
    if (!Number.isFinite(u) || !Number.isFinite(v) || !Number.isFinite(w)) continue;
    maximum = Math.max(maximum, Math.hypot(u, v, w));
  }
  return maximum;
}

/**
 * Pick the strongest vector sample through each horizontal world-space bin,
 * then rank those column representatives by magnitude. One seed per column
 * avoids launching many duplicate lines at different heights along the same
 * vertically coherent vortex while still covering separated resolved cores.
 */
export function rankedFeatureSeeds(
  cart: CartesianVoxelGrid,
  maxSeeds = 512,
): Float32Array {
  const [nx, ny, nz] = cart.shape;
  if (maxSeeds <= 0 || nx <= 0 || ny <= 0 || nz <= 0) return new Float32Array();
  const binsPerAxis = Math.max(1, Math.floor(Math.sqrt(maxSeeds)));
  const bx = Math.min(nx, binsPerAxis);
  const by = Math.min(ny, binsPerAxis);
  const binCount = bx * by;
  const magnitudes = new Float64Array(binCount);
  magnitudes.fill(-1);
  const positions = new Float32Array(binCount * 3);
  const [dx, dy, dz] = cart.spacing;
  const [ox, oy, oz] = cart.origin;

  for (let ix = 0; ix < nx; ix++) {
    const binX = Math.min(bx - 1, Math.floor((ix * bx) / nx));
    for (let iy = 0; iy < ny; iy++) {
      const binY = Math.min(by - 1, Math.floor((iy * by) / ny));
      for (let iz = 0; iz < nz; iz++) {
        const bin = binX * by + binY;
        const tuple = ((ix * ny + iy) * nz + iz) * 3;
        const x = cart.data[tuple];
        const y = cart.data[tuple + 1];
        const z = cart.data[tuple + 2];
        const magnitude = Math.hypot(x, y, z);
        if (!(magnitude > magnitudes[bin])) continue;
        magnitudes[bin] = magnitude;
        positions[bin * 3] = ox + ix * dx;
        positions[bin * 3 + 1] = oy + iy * dy;
        positions[bin * 3 + 2] = oz + iz * dz;
      }
    }
  }

  const ranked = Array.from({ length: binCount }, (_, bin) => bin)
    .filter((bin) => magnitudes[bin] > 0 && Number.isFinite(magnitudes[bin]))
    .sort((a, b) => magnitudes[b] - magnitudes[a] || a - b)
    .slice(0, maxSeeds);
  const output = new Float32Array(ranked.length * 3);
  for (let i = 0; i < ranked.length; i++) {
    output.set(positions.subarray(ranked[i] * 3, ranked[i] * 3 + 3), i * 3);
  }
  return output;
}

export function vtkPreparationTransferables(result: VtkPreparationResult): ArrayBuffer[] {
  if ("volume" in result) {
    return [...new Set([
      result.volume.values.buffer as ArrayBuffer,
      result.surface.values.buffer as ArrayBuffer,
      ...(result.volume.validity ? [result.volume.validity.buffer as ArrayBuffer] : []),
      ...(result.surface.validity ? [result.surface.validity.buffer as ArrayBuffer] : []),
    ])];
  }
  if ("featureSeeds" in result) {
    return [
      result.values.buffer as ArrayBuffer,
      result.featureSeeds.buffer as ArrayBuffer,
    ];
  }
  return [...new Set([
    result.values.buffer as ArrayBuffer,
    ...(result.validity ? [result.validity.buffer as ArrayBuffer] : []),
    ...(result.vortexCenterline ? [result.vortexCenterline.buffer as ArrayBuffer] : []),
  ])];
}

function subtractNonNegative(field: Npy3D, subtract: Npy3D): Npy3D {
  // Shape, not element count: two arrays can agree on length and disagree on
  // how it splits across x/y/z, which subtracts one cell from another cell's
  // neighbour and scrambles the result without ever failing.
  if (field.shape.some((extent, axis) => extent !== subtract.shape[axis])) return field;
  if (field.data.length !== subtract.data.length) return field;
  const data = new Float32Array(field.data.length);
  for (let i = 0; i < data.length; i++) {
    const primary = field.data[i];
    const removed = subtract.data[i];
    if (!Number.isFinite(primary) || !Number.isFinite(removed)) {
      data[i] = Number.NaN;
      continue;
    }
    const value = primary - removed;
    data[i] = value > 0 ? value : 0;
  }
  return { shape: field.shape, data, windowGeometry: field.windowGeometry };
}

function scalarStats(values: Float32Array): {
  min: number;
  max: number;
  p2: number;
  p98: number;
  positiveP98: number;
} {
  const finite = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return { min: 0, max: 0, p2: 0, p98: 0, positiveP98: 0 };
  const quantile = (samples: readonly number[], q: number) =>
    samples[Math.min(samples.length - 1, Math.floor(q * (samples.length - 1)))];
  const positive = finite.filter((value) => value > 0);
  return {
    min: finite[0],
    max: finite[finite.length - 1],
    p2: quantile(finite, 0.02),
    p98: quantile(finite, 0.98),
    positiveP98: positive.length > 0 ? quantile(positive, 0.98) : 0,
  };
}
