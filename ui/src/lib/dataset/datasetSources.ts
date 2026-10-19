/**
 * Stable identity for data opened in Studio. Plot renderers still consume the
 * existing Nimbus field bridge, but the shell no longer treats an untyped
 * directory string as the product-level data model. New readers attach here.
 */
export const DATASET_SOURCE_SCHEMA = "nimbus.studio.dataset-source" as const;
export const DATASET_SOURCE_VERSION = 1 as const;

export type DatasetSourceKind = "nimbus_run" | "cm1_netcdf";
export type DatasetAdapterAvailability = "available" | "metadata_only" | "unavailable";

export interface DatasetAdapterDefinition {
  kind: DatasetSourceKind;
  title: string;
  description: string;
  actionLabel: string;
  sourceModel: "nimbus" | "cm1";
  nativeFormat: "nimbus_history" | "netcdf";
  availability: DatasetAdapterAvailability;
  unavailableReason: string | null;
  multiresolution: {
    mode: "manifest-hierarchy" | "bounded-native-window";
    label: string;
    detail: string;
  };
}

export interface Cm1CaseInventory {
  schema: "nimbus.studio.cm1-case-inventory";
  schemaVersion: 1 | 2;
  directory: string;
  historyFiles: string[];
  namelistInput: string | null;
  inputSounding: string | null;
  restartFiles: string[];
  statisticsFiles: string[];
  otherFileCount: number;
  nestedDirectoryCount: number;
  scannedDirectoryCount?: number;
  skippedDirectoryCount?: number;
  excludedDirectoryCount?: number;
  unreadableDirectoryCount?: number;
  unreadableEntryCount?: number;
  symlinkCount?: number;
  discoveryMaxDepth?: number;
  discoveryTruncated?: boolean;
  warnings?: string[];
}

export interface Cm1NamelistAssignment {
  key: string;
  rawValue: string;
}

export interface Cm1NamelistGroup {
  name: string;
  assignments: Cm1NamelistAssignment[];
}

export interface Cm1NamelistReport {
  schema: "nimbus.studio.cm1-namelist-report";
  schemaVersion: 1;
  path: string;
  groups: Cm1NamelistGroup[];
  assignmentCount: number;
  values: Record<string, string>;
  warnings: string[];
}

export interface Cm1SoundingSurface {
  pressureHpa: number;
  potentialTemperatureK: number;
  mixingRatioGkg: number;
}

export interface Cm1SoundingLevel {
  sourceLine: number;
  heightM: number;
  potentialTemperatureK: number;
  mixingRatioGkg: number;
  uMs: number;
  vMs: number;
}

export interface Cm1SoundingReport {
  schema: "nimbus.studio.cm1-sounding-report";
  schemaVersion: 1;
  path: string;
  surface: Cm1SoundingSurface;
  levels: Cm1SoundingLevel[];
  topHeightM: number | null;
  warnings: string[];
}

export interface LocalDatasetSource {
  schema: typeof DATASET_SOURCE_SCHEMA;
  schemaVersion: typeof DATASET_SOURCE_VERSION;
  kind: DatasetSourceKind;
  sourceModel: DatasetAdapterDefinition["sourceModel"];
  nativeFormat: DatasetAdapterDefinition["nativeFormat"];
  locator: {
    kind: "local_path" | "local_paths";
    path: string;
    paths?: readonly string[];
  };
  displayName: string;
  readOnly: true;
  caseInventory?: Cm1CaseInventory;
  namelistReport?: Cm1NamelistReport;
  namelistInspectionError?: string;
  soundingReport?: Cm1SoundingReport;
  soundingInspectionError?: string;
}

export const DATASET_ADAPTERS: readonly DatasetAdapterDefinition[] = [
  {
    kind: "nimbus_run",
    title: "Nimbus run",
    description: "Open results from a local Nimbus simulation.",
    actionLabel: "Open Nimbus run…",
    sourceModel: "nimbus",
    nativeFormat: "nimbus_history",
    availability: "available",
    unavailableReason: null,
    multiresolution: {
      mode: "manifest-hierarchy",
      label: "Published hierarchy",
      detail: "NPY, ZFP, and netCDF histories can publish checksummed LODs; otherwise level zero remains a bounded native read.",
    },
  },
  {
    kind: "cm1_netcdf",
    title: "NetCDF data",
    description: "Open one or more atmospheric netCDF files. Compatible model metadata is recognized automatically.",
    actionLabel: "Open netCDF files…",
    sourceModel: "cm1",
    nativeFormat: "netcdf",
    availability: "available",
    unavailableReason: null,
    multiresolution: {
      mode: "bounded-native-window",
      label: "Bounded hyperslabs",
      detail: "Imported netCDF uses native spatial windows and ROI reduction; it does not claim a Nimbus LOD hierarchy.",
    },
  },
] as const;

export function datasetAdapter(
  kind: DatasetSourceKind,
): DatasetAdapterDefinition {
  const adapter = DATASET_ADAPTERS.find((candidate) => candidate.kind === kind);
  if (!adapter) throw new Error(`Unknown dataset source kind: ${kind}`);
  return adapter;
}

export function createLocalDatasetSource(
  kind: DatasetSourceKind,
  selection: string | readonly string[],
): LocalDatasetSource {
  const adapter = datasetAdapter(kind);
  if (adapter.availability === "unavailable") {
    throw new Error(adapter.unavailableReason ?? `${adapter.title} is unavailable`);
  }
  const paths = (typeof selection === "string" ? [selection] : selection)
    .map((path) => path.trim())
    .filter((path, index, values) => path.length > 0 && values.indexOf(path) === index);
  if (paths.length === 0) throw new Error("Dataset path must not be empty");
  if (kind === "nimbus_run" && paths.length !== 1) throw new Error("Nimbus runs must use one directory");
  const path = paths[0];
  return {
    schema: DATASET_SOURCE_SCHEMA,
    schemaVersion: DATASET_SOURCE_VERSION,
    kind,
    sourceModel: adapter.sourceModel,
    nativeFormat: adapter.nativeFormat,
    locator: paths.length === 1
      ? { kind: "local_path", path }
      : { kind: "local_paths", path, paths },
    displayName: paths.length === 1 ? datasetPathLabel(path) : `${paths.length} netCDF history files`,
    readOnly: true,
  };
}

export function datasetSourceIdentity(source: LocalDatasetSource): string {
  return JSON.stringify([
    source.schema,
    source.schemaVersion,
    source.kind,
    source.locator.kind,
    datasetSourcePaths(source),
  ]);
}

export function datasetSourcePaths(source: LocalDatasetSource): readonly string[] {
  return source.locator.paths ?? [source.locator.path];
}

export function datasetPathLabel(path: string): string {
  const parts = path.split(/[/\\]/).filter((part) => part.length > 0);
  if (parts.length === 0) return path;
  if (parts.length <= 2) return path;
  return `…/${parts.slice(-2).join("/")}`;
}
