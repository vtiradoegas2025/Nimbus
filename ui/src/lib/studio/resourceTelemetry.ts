import {
  clearFieldCache,
  fieldCacheStats,
  subscribeFieldCache,
  type FieldCacheStats,
} from "@/lib/core/fieldCache";
import {
  fieldLoadSchedulerStats,
  subscribeFieldLoadScheduler,
  type LoadSchedulerStats,
} from "@/lib/core/loadScheduler";
import {
  getVisualizationResourceSnapshot,
  subscribeVisualizationResources,
  type VisualizationResourceSnapshot,
} from "@/lib/core/visualizationResourceGovernor";
import {
  preparedDatasetRegistrySnapshot,
  subscribePreparedDatasetRegistry,
} from "@/lib/vtk/vtkPreparedDatasetRegistry";
import {
  subscribeVtkExtractionServicePool,
  vtkExtractionServicePool,
} from "@/lib/vtk/vtkExtractionServicePool";

export const RESOURCE_TELEMETRY_SCHEMA = "nimbus.studio.resource-telemetry" as const;
export const RESOURCE_TELEMETRY_VERSION = 6 as const;

export interface ResourceTelemetrySnapshot {
  readonly schema: typeof RESOURCE_TELEMETRY_SCHEMA;
  readonly version: typeof RESOURCE_TELEMETRY_VERSION;
  readonly sequence: number;
  readonly parsedCache: FieldCacheStats;
  readonly loader: LoadSchedulerStats;
  readonly visualization: VisualizationResourceSnapshot;
  readonly preparedRegistry: ReturnType<typeof preparedDatasetRegistrySnapshot>;
  readonly extractionPool: ReturnType<typeof vtkExtractionServicePool.snapshot>;
  readonly boundary: Readonly<{
    parsedCache: string;
    loaderBudget: string;
    nativeIpc: string;
    visualization: string;
  }>;
}

type ResourceTelemetryListener = () => void;
const listeners = new Set<ResourceTelemetryListener>();
let sequence = 0;

// Cancellation is not uniform: NPY/classic-netCDF stop between rows; ZFP v3 and netCDF4 nc_get_vars have no abort hook.
const boundary = Object.freeze({
  parsedCache: "Resident bytes count parsed, decompressed field arrays retained by Studio.",
  loaderBudget: "Reserved bytes are conservative decoded-size estimates for active loader jobs, not measured device or process memory.",
  nativeIpc: "Queued full-field work can be superseded. NPY and classic-netCDF window/LOD reads stop cooperatively between bounded rows, as do NPY terrain-geometry windows. ZFP v3 brick decodes and netCDF4 report their non-interruptible native-call boundaries. Cancelled results are discarded on every path.",
  visualization: "Visualization residency counts exact decoded-cache, prepared, worker-dataset, and retained-geometry backing stores. Native reads and preparation handoffs are capacity reservations, not reported as resident memory. Exposed worker typed-array scratch contributes an observed high water; opaque vtk.js-native, driver, and GPU residency remains unavailable rather than estimated as owned bytes.",
});

function createSnapshot(): ResourceTelemetrySnapshot {
  return Object.freeze({
    schema: RESOURCE_TELEMETRY_SCHEMA,
    version: RESOURCE_TELEMETRY_VERSION,
    sequence,
    parsedCache: fieldCacheStats(),
    loader: fieldLoadSchedulerStats(),
    visualization: getVisualizationResourceSnapshot(),
    preparedRegistry: preparedDatasetRegistrySnapshot(),
    extractionPool: vtkExtractionServicePool.snapshot(),
    boundary,
  });
}

let snapshot = createSnapshot();

function publish(): void {
  sequence += 1;
  snapshot = createSnapshot();
  for (const listener of listeners) listener();
}

// One combined snapshot gives useSyncExternalStore stable identity between transitions.
subscribeFieldCache(publish);
subscribeFieldLoadScheduler(publish);
subscribeVisualizationResources(publish);
subscribePreparedDatasetRegistry(publish);
subscribeVtkExtractionServicePool(publish);

export function getResourceTelemetrySnapshot(): ResourceTelemetrySnapshot {
  return snapshot;
}

export function subscribeResourceTelemetry(listener: ResourceTelemetryListener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Release retained parsed arrays. This deliberately does not cancel native I/O. */
export function clearParsedFieldCache(): void {
  clearFieldCache();
}
