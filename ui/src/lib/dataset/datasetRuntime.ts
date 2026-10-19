import {
  listSteps,
  openRunDirectory,
  registerNetcdfFrame,
  runHasTrajectories,
  type StepInfo,
} from "@/lib/run/runOutputs";
import {
  createLocalDatasetSource,
  datasetAdapter,
  datasetPathLabel,
  datasetSourcePaths,
  type Cm1CaseInventory,
  type Cm1NamelistReport,
  type Cm1SoundingReport,
  type DatasetSourceKind,
  type LocalDatasetSource,
} from "./datasetSources";
import { parseNpy, type NpyN } from "@/lib/platform/npy";
import {
  canonicalCm1HorizontalShape,
  canonicalCm1ScalarShape,
  cm1HistoryFieldMapping,
  cm1HistoryMappingForCanonical,
  cm1LengthScaleToMeters,
  cm1ScalarCoordinate,
  cm1TimeScaleToSeconds,
} from "@/lib/cm1/cm1HistoryMapping";
import { augmentStepWithDerivedFields } from "@/lib/derived/derivedFieldRegistry";
import { invokeCommand } from "@/lib/platform/appInvoke";

/** Runtime operations are deliberately separate from the serializable source
 * contract. Node-side schema tests can validate identities without importing
 * browser/Tauri globals, and future local/remote readers can implement this
 * boundary without changing workspace serialization. */

export async function reopenCm1CaseFolder(directory: string): Promise<LocalDatasetSource> {
  const inventory = await invokeCommand<Cm1CaseInventory>("inspect_cm1_case_folder", { directory });
  let namelistReport: Cm1NamelistReport | undefined;
  let namelistInspectionError: string | undefined;
  if (inventory.namelistInput) {
    try {
      namelistReport = await invokeCommand<Cm1NamelistReport>("inspect_cm1_namelist", { path: inventory.namelistInput });
    } catch (error) { namelistInspectionError = error instanceof Error ? error.message : String(error); }
  }
  let soundingReport: Cm1SoundingReport | undefined;
  let soundingInspectionError: string | undefined;
  if (inventory.inputSounding) {
    try {
      soundingReport = await invokeCommand<Cm1SoundingReport>("inspect_cm1_sounding", { path: inventory.inputSounding });
    } catch (error) { soundingInspectionError = error instanceof Error ? error.message : String(error); }
  }
  const source = createLocalDatasetSource("cm1_netcdf", inventory.historyFiles.length > 0 ? inventory.historyFiles : directory);
  return {
    ...source,
    displayName: `${datasetPathLabel(directory)} · ${inventory.historyFiles.length} history ${inventory.historyFiles.length === 1 ? "file" : "files"}`,
    caseInventory: inventory,
    ...(namelistReport ? { namelistReport } : {}),
    ...(namelistInspectionError ? { namelistInspectionError } : {}),
    ...(soundingReport ? { soundingReport } : {}),
    ...(soundingInspectionError ? { soundingInspectionError } : {}),
  };
}

export async function openLocalDatasetSource(
  kind: DatasetSourceKind,
  mode: "files" | "folder" = "files",
): Promise<LocalDatasetSource | null> {
  const adapter = datasetAdapter(kind);
  if (adapter.availability === "unavailable") {
    throw new Error(adapter.unavailableReason ?? `${adapter.title} is unavailable`);
  }
  switch (kind) {
    case "nimbus_run": {
      const path = await openRunDirectory();
      return path ? createLocalDatasetSource(kind, path) : null;
    }
    case "cm1_netcdf":
      if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
        return null;
      }
      {
        const { open } = await import("@tauri-apps/plugin-dialog");
        if (mode === "folder") {
          const directory = await open({
            directory: true,
            multiple: false,
            title: "Open CM1 case folder",
          });
          if (typeof directory !== "string") return null;
          return reopenCm1CaseFolder(directory);
        }
        const path = await open({
          directory: false,
          multiple: true,
          title: "Open CM1 / netCDF history files",
          filters: [{ name: "netCDF history", extensions: ["nc", "nc4", "cdf", "netcdf"] }],
        });
        return typeof path === "string" || Array.isArray(path)
          ? createLocalDatasetSource(kind, path)
          : null;
      }
  }
}

export interface NetcdfDimensionMetadata {
  name: string;
  length: number;
  unlimited: boolean;
}

export interface NetcdfVariableMetadata {
  name: string;
  dataType: string;
  dimensions: string[];
  shape: number[];
  units: string | null;
  longName: string | null;
  cfDecode?: {
    status: "identity" | "applied" | "unsupported";
    scaleFactor: number | null;
    addOffset: number | null;
    fillValue: number | null;
    fillValueSource: "attribute" | "netcdf_default" | null;
    missingValues: number[];
    validMin: number | null;
    validMax: number | null;
    validDomain: "stored";
    missingOutput: "nan";
    issue: string | null;
  };
}

export interface NetcdfMetadata {
  schema: "nimbus.studio.netcdf-metadata";
  schemaVersion: 1;
  path: string;
  containerFormat: string;
  recordCount: number;
  dimensions: NetcdfDimensionMetadata[];
  variables: NetcdfVariableMetadata[];
  sourceClassification: "cm1_likely" | "unclassified_netcdf";
  plottingSupported: boolean;
  plotUnavailableReason: string;
}

export async function inspectNetcdfMetadata(
  source: LocalDatasetSource,
): Promise<NetcdfMetadata> {
  if (source.kind !== "cm1_netcdf") throw new Error("Expected a CM1/netCDF source");
  return invokeCommand<NetcdfMetadata>("inspect_netcdf_metadata", { path: source.locator.path });
}

export async function readNetcdfField(
  source: LocalDatasetSource,
  variable: string,
  recordIndex: number,
): Promise<NpyN> {
  if (source.kind !== "cm1_netcdf") throw new Error("Expected a CM1/netCDF source");
  const bytes = await invokeCommand<ArrayBuffer>("read_netcdf_field", {
    path: source.locator.path,
    variable,
    recordIndex,
  });
  return parseNpy(bytes);
}

export async function readNetcdfValues(
  source: LocalDatasetSource,
  variable: string,
): Promise<number[]> {
  if (source.kind !== "cm1_netcdf") throw new Error("Expected a CM1/netCDF source");
  return invokeCommand<number[]>("read_netcdf_values", { path: source.locator.path, variable });
}

async function optionalNetcdfValues(
  source: LocalDatasetSource,
  variable: NetcdfVariableMetadata | undefined,
  reader: (source: LocalDatasetSource, variable: string) => Promise<number[]>,
): Promise<number[] | null> {
  if (!variable) return null;
  try {
    return await reader(source, variable.name);
  } catch {
    return null;
  }
}

export interface NetcdfRuntimeApi {
  inspect(source: LocalDatasetSource): Promise<NetcdfMetadata>;
  readValues(source: LocalDatasetSource, variable: string): Promise<number[]>;
}

const DEFAULT_NETCDF_RUNTIME: NetcdfRuntimeApi = {
  inspect: inspectNetcdfMetadata,
  readValues: readNetcdfValues,
};

async function listNetcdfFileFrames(
  source: LocalDatasetSource,
  path: string,
  runtime: NetcdfRuntimeApi,
  knownMetadata?: NetcdfMetadata,
): Promise<StepInfo[]> {
  const fileSource: LocalDatasetSource = {
    ...source,
    locator: { kind: "local_path", path },
  };
  const metadata = knownMetadata?.path === path
    ? knownMetadata
    : await runtime.inspect(fileSource);
  const mapped = metadata.variables.flatMap((variable) => {
    if (variable.cfDecode?.status === "unsupported") return [];
    const mapping = cm1HistoryFieldMapping(variable.name, variable.units);
    const shape = mapping?.rank === 2
      ? canonicalCm1HorizontalShape(variable.dimensions, variable.shape)
      : canonicalCm1ScalarShape(variable.dimensions, variable.shape);
    return mapping && shape ? [{ mapping, shape, variable }] : [];
  });
  const volumeShape = mapped.find(({ mapping }) => mapping.rank === 3)?.shape;
  const horizontalShape = volumeShape?.slice(0, 2) ?? mapped[0]?.shape;
  if (!horizontalShape) return [];
  const zDimension = metadata.dimensions.find(({ name }) => ["zh", "z", "nk"].includes(name.toLowerCase()));
  const gridShape: [number, number, number] = [
    horizontalShape[0],
    horizontalShape[1],
    volumeShape?.[2] ?? zDimension?.length ?? 1,
  ];
  const compatible = mapped.filter(({ mapping, shape }) => {
    const expected = mapping.rank === 2 ? gridShape.slice(0, 2) : gridShape;
    return shape.length === expected.length && shape.every((length, axis) => length === expected[axis]);
  });
  const nativeFields = [...new Set(compatible.map(({ mapping }) => mapping.canonicalName))];
  const nativeFieldUnits = Object.fromEntries(
    compatible.map(({ mapping }) => [mapping.canonicalName, mapping.canonicalUnit] as const),
  );
  const coordinateNames = [["xh", "x", "ni"], ["yh", "y", "nj"], ["zh", "z", "nk"]] as const;
  const coordinateVariables = coordinateNames.map((names, axis) => {
    const dimension = metadata.dimensions.find(({ name, length }) =>
      names.some((candidate) => candidate === name.toLowerCase()) && length === gridShape[axis],
    )?.name;
    if (!dimension) return undefined;
    return metadata.variables.find((variable) =>
      variable.dimensions.length === 1
      && variable.dimensions[0] === dimension
      && (names.some((name) => name === variable.name.toLowerCase()) || variable.name === dimension),
    );
  });
  const timeVariable = metadata.variables.find((variable) =>
    variable.name.toLowerCase() === "time" && variable.dimensions.length === 1,
  );
  const [xValues, yValues, zValues, timeValues] = await Promise.all([
    optionalNetcdfValues(fileSource, coordinateVariables[0], runtime.readValues),
    optionalNetcdfValues(fileSource, coordinateVariables[1], runtime.readValues),
    optionalNetcdfValues(fileSource, coordinateVariables[2], runtime.readValues),
    optionalNetcdfValues(fileSource, timeVariable, runtime.readValues),
  ]);
  const coordinates = [xValues, yValues, zValues].map((values, axis) => {
    const variable = coordinateVariables[axis];
    return values && values.length === gridShape[axis]
      ? cm1ScalarCoordinate(values, cm1LengthScaleToMeters(variable?.units ?? null))
      : null;
  });
  const hasPhysicalGrid = coordinates.every((coordinate) => coordinate != null);
  const timeScale = cm1TimeScaleToSeconds(timeVariable?.units ?? null);
  const simulationTimes = timeValues && timeScale != null && timeValues.length === metadata.recordCount
    ? timeValues.map((value) => Number.isFinite(value) ? value * timeScale : null)
    : [];
  const grid = {
    nr: gridShape[0], nth: gridShape[1], nz: gridShape[2],
    coordinateSystem: "cartesian", staggering: "cell_centered",
    geometryKind: hasPhysicalGrid ? "physical" as const : "index" as const,
    ...(hasPhysicalGrid ? {
      ...(coordinates[0]!.uniform ? { drM: coordinates[0]!.uniform.spacing } : {}),
      ...(coordinates[1]!.uniform ? { dyM: coordinates[1]!.uniform.spacing } : {}),
      ...(coordinates[2]!.uniform ? { dzM: coordinates[2]!.uniform.spacing } : {}),
      scalarCoordinatesM: coordinates.map((coordinate) => coordinate!.valuesM) as [number[], number[], number[]],
      scalarOriginM: coordinates.map((coordinate) => coordinate!.valuesM[0]) as [number, number, number],
      physicalBoundsM: {
        lower: [coordinates[0]!.lowerBound, coordinates[1]!.lowerBound, coordinates[2]!.lowerBound] as [number, number, number],
        upper: [coordinates[0]!.upperBound, coordinates[1]!.upperBound, coordinates[2]!.upperBound] as [number, number, number],
      },
    } : {}),
  };
  return Array.from({ length: metadata.recordCount }, (_, index) => augmentStepWithDerivedFields({
    index,
    simulationTimeS: simulationTimes[index] ?? null,
    fields: nativeFields,
    grid,
    fieldUnits: nativeFieldUnits,
    fieldOrigins: Object.fromEntries(nativeFields.map((field) => [field, "native" as const])),
    publicationState: "complete" as const,
    publicationGeneration: `netcdf:${path}`,
  }));
}

function compatibleGrid(left: StepInfo["grid"], right: StepInfo["grid"]): boolean {
  if (!left || !right || left.nr !== right.nr || left.nth !== right.nth || left.nz !== right.nz) return false;
  if (left.geometryKind !== right.geometryKind) return false;
  const close = (a: number | undefined, b: number | undefined): boolean => {
    if (a == null || b == null) return a == null && b == null;
    return Math.abs(a - b) <= Math.max(1e-7, Math.abs(a) * 1e-7, Math.abs(b) * 1e-7);
  };
  if (!close(left.drM, right.drM) || !close(left.dyM, right.dyM) || !close(left.dzM, right.dzM)) return false;
  if (left.scalarCoordinatesM == null || right.scalarCoordinatesM == null) {
    if (left.scalarCoordinatesM !== right.scalarCoordinatesM) return false;
  } else if (!left.scalarCoordinatesM.every((values, axis) =>
    values.length === right.scalarCoordinatesM![axis].length
    && values.every((value, index) => close(value, right.scalarCoordinatesM![axis][index]))
  )) return false;
  if (left.scalarOriginM == null || right.scalarOriginM == null) return left.scalarOriginM == null && right.scalarOriginM == null;
  return left.scalarOriginM.every((value, axis) => close(value, right.scalarOriginM?.[axis]));
}

export async function listDatasetFrames(
  source: LocalDatasetSource,
  netcdfRuntime: NetcdfRuntimeApi = DEFAULT_NETCDF_RUNTIME,
  knownMetadata?: NetcdfMetadata,
): Promise<StepInfo[]> {
  switch (source.kind) {
    case "nimbus_run":
      return (await listSteps(source.locator.path)).map(augmentStepWithDerivedFields);
    case "cm1_netcdf":
      {
        const groups: Array<{ path: string; steps: StepInfo[] }> = [];
        for (const path of datasetSourcePaths(source)) {
          groups.push({
            path,
            steps: await listNetcdfFileFrames(source, path, netcdfRuntime, knownMetadata),
          });
        }
        groups.sort((left, right) => {
          const leftTime = left.steps[0]?.simulationTimeS;
          const rightTime = right.steps[0]?.simulationTimeS;
          if (leftTime != null && rightTime != null && leftTime !== rightTime) return leftTime - rightTime;
          return left.path.localeCompare(right.path, undefined, { numeric: true });
        });
        const referenceGrid = groups.flatMap(({ steps }) => steps.slice(0, 1))[0]?.grid;
        if (referenceGrid) {
          for (const { path, steps } of groups) {
            const grid = steps[0]?.grid;
            if (grid && !compatibleGrid(grid, referenceGrid)) {
              throw new Error(`CM1 history grid mismatch in ${path}; open files from one compatible experiment`);
            }
          }
        }
        const result: StepInfo[] = [];
        for (const { path, steps } of groups) {
          for (const localStep of steps) {
            const index = result.length;
            const publicationGeneration = `netcdf:${path}`;
            const step = { ...localStep, index, publicationGeneration };
            result.push(step);
            registerNetcdfFrame(
              source.locator.path,
              index,
              publicationGeneration,
              path,
              localStep.index,
              [step.grid!.nr!, step.grid!.nth!, step.grid!.nz!],
              Object.fromEntries(step.fields.map((field) => {
                const mapping = cm1HistoryMappingForCanonical(field);
                return [field, mapping?.rank === 2
                  ? [step.grid!.nr!, step.grid!.nth!]
                  : [step.grid!.nr!, step.grid!.nth!, step.grid!.nz!]];
              })),
              step.fields.filter((field) => step.fieldOrigins?.[field] !== "studio_derived"),
            );
          }
        }
        return result;
      }
  }
}

export async function datasetHasTrajectories(
  source: LocalDatasetSource,
  expectedGeneration: string | null,
): Promise<boolean> {
  switch (source.kind) {
    case "nimbus_run":
      return runHasTrajectories(source.locator.path, expectedGeneration);
    case "cm1_netcdf":
      return false;
  }
}
