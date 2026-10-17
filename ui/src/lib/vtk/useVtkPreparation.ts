import { useEffect, useMemo, useRef, useState } from "react";
import type { GridSpacing } from "@/lib/science/cartesianInterp";
import type { Npy3D } from "@/lib/platform/npy";
import {
  isPreparationAbort,
  prepareVtkInWorker,
} from "./vtkPreparationClient";
import type {
  PreparedVectorImage,
  PreparedVtkImage,
  PreparedCloudImages,
  VelocityGradientDiagnostic,
  VtkPreparationResult,
} from "./vtkPreparation";
import type { RenderFidelityMode } from "@/lib/plot/renderFidelity";
import {
  resolvePreparedDataset,
  retainPreparedDataset,
  type VtkPreparedDatasetHandle,
  type VtkPreparedDatasetMetadata,
  type VtkPreparedDatasetLease,
} from "./vtkPreparedDatasetRegistry";
import { prepareVectorInWorkerOwnedService } from "./streamlineWorkerOwnedPreparation";
import { prepareScalarInWorkerOwnedService } from "./contourWorkerOwnedPreparation";

export interface PreparationState<T extends VtkPreparationResult> {
  prepared: T | null;
  handle: VtkPreparedDatasetHandle<T> | null;
  metadata: VtkPreparedDatasetMetadata<T> | null;
  loading: boolean;
  error: string | null;
}

let preparationScopeSequence = 0;
function usePreparationScope(kind: string): string {
  const scope = useRef<string>();
  if (!scope.current) scope.current = `vtk:${kind}:${++preparationScopeSequence}`;
  return scope.current;
}

function exposedPreparation<T extends VtkPreparationResult>(
  lease: VtkPreparedDatasetLease<T> | null,
  loading: boolean,
  error: string | null,
  includeResident: boolean,
): PreparationState<T> {
  return {
    handle: lease?.handle ?? null,
    metadata: lease?.handle.metadata ?? null,
    prepared: lease && includeResident ? resolvePreparedDataset(lease.handle) : null,
    loading,
    error,
  };
}

interface ScalarState {
  requestIdentity: object | null;
  field: Npy3D | null;
  grid: GridSpacing | null;
  lease: VtkPreparedDatasetLease<PreparedVtkImage> | null;
  loading: boolean;
  error: string | null;
}

export function usePreparedScalarImage(
  field: Npy3D | null,
  grid: GridSpacing | null,
  maxVoxels: number | undefined,
  _generation: number,
  options: {
    subtractField?: Npy3D | null;
    refineForDisplay?: boolean;
    featureSign?: 1 | -1;
    featureZRangeM?: readonly [number, number];
    geometryHandoff?: boolean;
    workerOwned?: boolean;
    stableWorkerLane?: boolean | string;
    scalarTransform?: import("./vtkPreparation").ScalarFieldTransform;
    background?: import("./volumeBackground").VolumeBackground;
    cropBackground?: boolean;
    structuralSupportRangeFraction?: number;
  } = {},
): PreparationState<PreparedVtkImage> {
  const preparationScope = usePreparationScope("scalar");
  const requestIdentity = useMemo(() => ({}), [
    field,
    grid,
    maxVoxels,
    options.subtractField,
    options.refineForDisplay,
    options.featureSign,
    options.featureZRangeM,
    options.scalarTransform,
    options.background, options.cropBackground,
    options.structuralSupportRangeFraction,
    options.geometryHandoff,
    options.workerOwned,
    options.stableWorkerLane,
    preparationScope,
  ]);
  const [state, setState] = useState<ScalarState>({
    requestIdentity: null,
    field: null,
    grid: null,
    lease: null,
    loading: false,
    error: null,
  });
  useEffect(() => {
    if (!field || !grid) return;
    const controller = new AbortController();
    let lease: VtkPreparedDatasetLease<PreparedVtkImage> | null = null;
    setState({ requestIdentity, field, grid, lease: null, loading: true, error: null });
    const request = {
        kind: "scalar",
        field,
        grid,
        maxVoxels,
        subtractField: options.subtractField ?? undefined,
        refineForDisplay: options.refineForDisplay,
        featureSign: options.featureSign,
        featureZRangeM: options.featureZRangeM,
        scalarTransform: options.scalarTransform,
        background: options.background, cropBackground: options.cropBackground,
        structuralSupportRangeFraction: options.structuralSupportRangeFraction,
        datasetHandoff: options.geometryHandoff ? "shared-geometry" : "render-only",
      } satisfies import("./vtkPreparation").ScalarPreparationRequest;
    const publishLease = (nextLease: VtkPreparedDatasetLease<PreparedVtkImage>) => {
      if (controller.signal.aborted) {
        nextLease.release();
        return;
      }
      lease = nextLease;
      setState({ requestIdentity, field, grid, lease, loading: false, error: null });
    };
    const reject = (error: unknown) => {
      if (controller.signal.aborted || isPreparationAbort(error)) return;
      setState({ requestIdentity,
        field, grid, lease: null, loading: false,
        error: error instanceof Error ? error.message : String(error),
      });
    };
    if (options.geometryHandoff && options.workerOwned !== false && "Worker" in globalThis) {
      void prepareScalarInWorkerOwnedService(
        request,
        controller.signal,
        typeof options.stableWorkerLane === "string"
          ? options.stableWorkerLane
          : options.stableWorkerLane ? preparationScope : undefined,
      ).then(publishLease, reject);
    } else {
      void prepareVtkInWorker(
        request,
        controller.signal,
        undefined,
        { scope: preparationScope },
      ).then(
      (prepared) => {
        if (controller.signal.aborted) return;
        try {
          publishLease(retainPreparedDataset(prepared, "scalar"));
        } catch (error) {
          reject(error);
        }
      },
      reject,
    );
    }
    return () => {
      controller.abort();
      lease?.release();
    };
  }, [
    field,
    grid,
    maxVoxels,
    options.subtractField,
    options.refineForDisplay,
    options.featureSign,
    options.featureZRangeM,
    options.scalarTransform,
    options.background, options.cropBackground,
    options.structuralSupportRangeFraction,
    options.geometryHandoff,
    options.workerOwned,
    options.stableWorkerLane,
    preparationScope,
  ]);

  const current = state.field === field && state.grid === grid;
  return current && state.requestIdentity === requestIdentity && state.lease?.active !== false
    ? exposedPreparation(state.lease, state.loading, state.error, options.geometryHandoff !== true)
    : { prepared: null, handle: null, metadata: null, loading: field != null && grid != null, error: null };
}

interface VectorState {
  requestIdentity: object | null;
  fields: [Npy3D, Npy3D, Npy3D] | null;
  grid: GridSpacing | null;
  detectVerticalVortexCore: boolean;
  lease: VtkPreparedDatasetLease<PreparedVectorImage> | null;
  loading: boolean;
  error: string | null;
}

export function usePreparedVectorImage(
  fields: [Npy3D, Npy3D, Npy3D] | null,
  grid: GridSpacing | null,
  maxVoxels: number | undefined,
  _generation: number,
  options: { detectVerticalVortexCore?: boolean; geometryHandoff?: boolean } = {},
): PreparationState<PreparedVectorImage> {
  const preparationScope = usePreparationScope("vector");
  const detectVerticalVortexCore = options.detectVerticalVortexCore === true;
  const requestIdentity = useMemo(() => ({}), [
    fields,
    grid,
    maxVoxels,
    detectVerticalVortexCore,
    options.geometryHandoff,
    preparationScope,
  ]);
  const [state, setState] = useState<VectorState>({
    requestIdentity: null,
    fields: null,
    grid: null,
    detectVerticalVortexCore: false,
    lease: null,
    loading: false,
    error: null,
  });
  useEffect(() => {
    if (!fields || !grid) return;
    const controller = new AbortController();
    let lease: VtkPreparedDatasetLease<PreparedVectorImage> | null = null;
    setState({ requestIdentity,
      fields,
      grid,
      detectVerticalVortexCore,
      lease: null,
      loading: true,
      error: null,
    });
    const request = {
        kind: "vector",
        fields,
        grid,
        maxVoxels,
        detectVerticalVortexCore,
        datasetHandoff: options.geometryHandoff ? "shared-geometry" : "render-only",
      } satisfies import("./vtkPreparation").VectorPreparationRequest;
    const publishLease = (nextLease: VtkPreparedDatasetLease<PreparedVectorImage>) => {
      if (controller.signal.aborted) {
        nextLease.release();
        return;
      }
      lease = nextLease;
      setState({ requestIdentity, fields, grid, detectVerticalVortexCore, lease, loading: false, error: null });
    };
    const reject = (error: unknown) => {
      if (controller.signal.aborted || isPreparationAbort(error)) return;
      setState({ requestIdentity,
        fields,
        grid,
        detectVerticalVortexCore,
        lease: null,
        loading: false,
        error: error instanceof Error ? error.message : String(error),
      });
    };
    if (options.geometryHandoff && "Worker" in globalThis) {
      void prepareVectorInWorkerOwnedService(request, controller.signal).then(publishLease, reject);
    } else {
      void prepareVtkInWorker(
        request,
        controller.signal,
        undefined,
        { scope: preparationScope },
      ).then(
      (prepared) => {
        if (controller.signal.aborted) return;
        try {
          publishLease(retainPreparedDataset(prepared, "vector"));
        } catch (error) {
          reject(error);
        }
      },
      reject,
    );
    }
    return () => {
      controller.abort();
      lease?.release();
    };
  }, [
    fields,
    grid,
    maxVoxels,
    detectVerticalVortexCore,
    options.geometryHandoff,
    preparationScope,
  ]);

  const current = state.fields === fields &&
    state.grid === grid &&
    state.detectVerticalVortexCore === detectVerticalVortexCore;
  return current && state.requestIdentity === requestIdentity && state.lease?.active !== false
    ? exposedPreparation(state.lease, state.loading, state.error, false)
    : { prepared: null, handle: null, metadata: null, loading: fields != null && grid != null, error: null };
}

interface VelocityDiagnosticState {
  requestIdentity: object | null;
  fields: [Npy3D, Npy3D, Npy3D] | null;
  grid: GridSpacing | null;
  diagnostic: VelocityGradientDiagnostic;
  lease: VtkPreparedDatasetLease<PreparedVtkImage> | null;
  loading: boolean;
  error: string | null;
}

export function usePreparedVelocityDiagnostic(
  fields: [Npy3D, Npy3D, Npy3D] | null,
  grid: GridSpacing | null,
  maxVoxels: number | undefined,
  _generation: number,
  diagnostic: VelocityGradientDiagnostic = "q-criterion",
  options: { refineForDisplay?: boolean; geometryHandoff?: boolean } = {},
): PreparationState<PreparedVtkImage> {
  const preparationScope = usePreparationScope("velocity-diagnostic");
  const requestIdentity = useMemo(() => ({}), [
    fields,
    grid,
    maxVoxels,
    diagnostic,
    options.refineForDisplay,
    options.geometryHandoff,
    preparationScope,
  ]);
  const [state, setState] = useState<VelocityDiagnosticState>({
    requestIdentity: null,
    fields: null,
    grid: null,
    diagnostic,
    lease: null,
    loading: false,
    error: null,
  });
  useEffect(() => {
    if (!fields || !grid) return;
    const controller = new AbortController();
    let lease: VtkPreparedDatasetLease<PreparedVtkImage> | null = null;
    setState({ requestIdentity,
      fields,
      grid,
      diagnostic,
      lease: null,
      loading: true,
      error: null,
    });
    prepareVtkInWorker(
      {
        kind: "velocity-diagnostic",
        diagnostic,
        fields,
        grid,
        maxVoxels,
        refineForDisplay: options.refineForDisplay,
        datasetHandoff: options.geometryHandoff ? "shared-geometry" : "render-only",
      },
      controller.signal,
      undefined,
      { scope: preparationScope },
    ).then(
      (prepared) => {
        if (controller.signal.aborted) return;
        try {
          lease = retainPreparedDataset(prepared, "velocity-diagnostic");
          setState({ requestIdentity, fields, grid, diagnostic, lease, loading: false, error: null });
        } catch (error) {
          setState({ requestIdentity,
            fields, grid, diagnostic, lease: null, loading: false,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      },
      (error) => {
        if (controller.signal.aborted || isPreparationAbort(error)) return;
        setState({ requestIdentity,
          fields,
          grid,
          diagnostic,
          lease: null,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        });
      },
    );
    return () => {
      controller.abort();
      lease?.release();
    };
  }, [
    fields,
    grid,
    maxVoxels,
    diagnostic,
    options.refineForDisplay,
    options.geometryHandoff,
    preparationScope,
  ]);

  const current = state.fields === fields &&
    state.grid === grid &&
    state.diagnostic === diagnostic;
  return current && state.requestIdentity === requestIdentity && state.lease?.active !== false
    ? exposedPreparation(state.lease, state.loading, state.error, false)
    : { prepared: null, handle: null, metadata: null, loading: fields != null && grid != null, error: null };
}

interface CloudState {
  requestIdentity: object | null;
  field: Npy3D | null;
  subtractField: Npy3D | null;
  grid: GridSpacing | null;
  exclusionKey: string;
  lease: VtkPreparedDatasetLease<PreparedCloudImages> | null;
  loading: boolean;
  error: string | null;
}

export function usePreparedCloudImages(
  field: Npy3D | null,
  subtractField: Npy3D | null,
  grid: GridSpacing | null,
  volumeMaxVoxels: number,
  surfaceMaxVoxels: number,
  _generation: number,
  options: {
    fidelityMode?: RenderFidelityMode;
    detailSeed?: number;
    flowFields?: [Npy3D, Npy3D, Npy3D] | null;
    simulationTimeS?: number;
    excludePhysicalBoundsM?: {
      lower: readonly [number, number, number];
      upper: readonly [number, number, number];
    };
  } = {},
): PreparationState<PreparedCloudImages> {
  const preparationScope = usePreparationScope("cloud");
  const excludePhysicalBoundsM = options.excludePhysicalBoundsM;
  const exclusionKey = JSON.stringify(excludePhysicalBoundsM ?? null);
  const requestIdentity = useMemo(() => ({}), [
    field, subtractField, grid, volumeMaxVoxels, surfaceMaxVoxels,
    excludePhysicalBoundsM,
    exclusionKey, preparationScope,
  ]);
  const [state, setState] = useState<CloudState>({
    requestIdentity: null,
    field: null,
    subtractField: null,
    grid: null,
    exclusionKey: "null",
    lease: null,
    loading: false,
    error: null,
  });
  useEffect(() => {
    if (!field || !grid) return;
    const controller = new AbortController();
    let lease: VtkPreparedDatasetLease<PreparedCloudImages> | null = null;
    setState({ requestIdentity,
      field, subtractField, grid, exclusionKey,
      lease: null, loading: true, error: null,
    });
    prepareVtkInWorker(
      {
        kind: "cloud",
        field,
        subtractField: subtractField ?? undefined,
        grid,
        volumeMaxVoxels,
        surfaceMaxVoxels,
        excludePhysicalBoundsM,
      },
      controller.signal,
      undefined,
      { scope: preparationScope },
    ).then(
      (prepared) => {
        if (controller.signal.aborted) return;
        try {
          lease = retainPreparedDataset(prepared, "cloud");
          setState({ requestIdentity,
            field, subtractField, grid, exclusionKey,
            lease, loading: false, error: null,
          });
        } catch (error) {
          setState({ requestIdentity,
            field, subtractField, grid, exclusionKey,
            lease: null, loading: false,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      },
      (error) => {
        if (controller.signal.aborted || isPreparationAbort(error)) return;
        setState({ requestIdentity,
          field, subtractField, grid, exclusionKey,
          lease: null,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        });
      },
    );
    return () => {
      controller.abort();
      lease?.release();
    };
  }, [
    field, subtractField, grid, volumeMaxVoxels, surfaceMaxVoxels,
    excludePhysicalBoundsM,
    exclusionKey, preparationScope,
  ]);
  const current =
    state.field === field && state.subtractField === subtractField &&
    state.grid === grid &&
    state.exclusionKey === exclusionKey;
  return current && state.requestIdentity === requestIdentity && state.lease?.active !== false
    ? exposedPreparation(state.lease, state.loading, state.error, true)
    : { prepared: null, handle: null, metadata: null, loading: field != null && grid != null, error: null };
}
