import { readPath, type ConfigObject } from "./configSchema";
import { parseYaml } from "./yaml";
import type { SchemeCatalog } from "./schemeCatalog";
import { invokeCommand } from "@/lib/platform/appInvoke";
import { fnv1a64Text } from "@/lib/cm1/cm1GuidedHandoff";
import type { RemoteComputeTarget } from "@/lib/platform/remoteCompute";

export interface ConfigPreflightRequest {
  schema: "nimbus.config-preflight.v10";
  requestVersion: 10;
  source:
    | { kind: "exact_document"; yamlText: string; yamlPath: string }
    | { kind: "descriptor_only" };
  executionMode: string;
  maturityLimitsAcknowledged: boolean;
  backend: string;
  allowBackendFallback: boolean;
  validationGuardMode: string;
  gpuApplyMicrophysics: boolean;
  gpuApplySlow: boolean;
  gpuApplySgs: boolean;
  outputFormat: string;
  gridMetrics: {
    nx: number;
    ny: number;
    nz: number;
    dx: number;
    dy: number;
    dz: number;
    ztop: number;
  };
  timeControl: {
    dtLarge: number;
    acousticSubsteps: number;
    outputInterval: number;
  };
  output: {
    preset: string;
    asyncIo: boolean;
    zfpMode: string;
    zfpPerFieldTolerances: boolean;
    zfpSparseThreshold: number;
    zfpFloat16Prefilter: boolean;
    zfpFieldExecution: string;
    zfpFieldWorkers: number;
    zfpFieldScratchMib: number;
    zfpFieldTolerances: Record<string, number>;
    keepLastFrames: number;
    minimumFreeSpaceMib: number;
    maximumRetainedHistoryFiles: number;
  };
  coordinateSystem: string;
  staggering: string;
  boundary: {
    west: string;
    east: string;
    south: string;
    north: string;
    bottom: string;
    top: string;
  };
  diagnostics: {
    scientificEverySteps: number;
    deterministicReductions: boolean;
  };
  initialization: {
    soundingSource: string;
    hodographSource: string;
    triggerSource: string;
  };
  interactions: {
    radiationShortwaveEnabled: boolean;
    turbulenceMode: string;
    turbulenceDynamicCoefficient: boolean;
    surfaceScheme: string;
    surfaceApplyFluxes: boolean;
    diffusionTarget: string;
    diffusionActive: boolean;
  };
  availableStateFields: string[];
  selections: Array<{ moduleId: string; schemeId: string }>;
}

export interface ConfigPreflightIssue {
  code: string;
  moduleId: string | null;
  schemeId: string | null;
  message: string;
  expected: string[];
  actual: string | null;
}

export interface ConfigPreflightReport {
  schema: "nimbus.config-preflight-report.v10";
  reportVersion: 10;
  compatible: boolean;
  executionTargetFingerprint?: string | null;
  nativeAdmission: {
    state: "accepted" | "rejected" | "not_checked";
    descriptorProjection: "not_checked";
    sourceSha256: string | null;
    sourceFingerprint: string | null;
    sourcePathFingerprint: string | null;
    validatedBinarySha256: string | null;
    validationExitCode: number | null;
  };
  executable: {
    canonicalPath: string;
    byteLength: number;
    modifiedUnixMs: number | null;
    catalogFingerprint: string;
    binaryFingerprint: string;
  };
  context: Omit<ConfigPreflightRequest, "schema" | "requestVersion" | "source" | "selections">;
  resolvedSelections: Array<{
    moduleId: string;
    requestedSchemeId: string;
    canonicalSchemeId: string;
    maturity: string;
    constructible: boolean;
    runtimeSelectable: boolean;
  }>;
  hardErrors: ConfigPreflightIssue[];
  warnings: ConfigPreflightIssue[];
}

/** Read the authoritative runtimeStateFields from the exact executable's catalog.
 *  Per-scheme requiredStateFields describe needs, not allocations; older binaries fail closed. */
export function authoritativeRuntimeStateFields(
  catalog: SchemeCatalog,
): string[] {
  const raw = (catalog as SchemeCatalog & { runtimeStateFields?: unknown })
    .runtimeStateFields;
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error(
      "the exact simulator catalog does not publish an authoritative runtimeStateFields inventory",
    );
  }
  if (!raw.every((field) => typeof field === "string" && field.trim() !== "")) {
    throw new Error(
      "the exact simulator catalog publishes a malformed runtimeStateFields inventory",
    );
  }
  const normalized = raw.map((field) => field.trim());
  if (new Set(normalized).size !== normalized.length) {
    throw new Error(
      "the exact simulator catalog publishes duplicate runtimeStateFields",
    );
  }
  return [...normalized].sort();
}

/** Build the Rust preflight request from the form config and executable catalog. */
export function buildConfigPreflightRequest(
  config: ConfigObject,
  catalog: SchemeCatalog,
  availableStateFields: readonly string[],
  sourceDocument?: { yamlText: string; yamlPath: string },
): ConfigPreflightRequest {
  if (sourceDocument !== undefined &&
      (sourceDocument.yamlText.trim() === "" ||
       new TextEncoder().encode(sourceDocument.yamlText).byteLength > 8 * 1024 * 1024 ||
       sourceDocument.yamlPath.trim() === "")) {
    throw new Error("exact configuration preflight requires bounded YAML text and a source path");
  }
  const outputConfig = sourceDocument ? parseYaml(sourceDocument.yamlText) : config;
  const optionalValue = (path: string) => readPath(path.startsWith("output.") ? outputConfig : config, path);
  const executionMode = readPath(config, "extensions.execution.mode");
  const required = (path: string): string => {
    const value = readPath(config, path);
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error(`configuration preflight requires '${path}'`);
    }
    return value;
  };
  const requiredBoolean = (path: string): boolean => {
    const value = readPath(config, path);
    if (typeof value !== "boolean") {
      throw new Error(`configuration preflight requires boolean '${path}'`);
    }
    return value;
  };
  const requiredFiniteNumber = (path: string): number => {
    const value = readPath(config, path);
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`configuration preflight requires finite number '${path}'`);
    }
    return value;
  };
  const requiredInteger = (path: string): number => {
    const value = readPath(config, path);
    if (typeof value !== "number" || !Number.isSafeInteger(value)) {
      throw new Error(`configuration preflight requires integer '${path}'`);
    }
    return value;
  };

  const selections = catalog.modules.flatMap((module) => {
    const value = readPath(config, module.selectionPath);
    if (typeof value === "string" && value.trim() !== "") {
      return [{ moduleId: module.id, schemeId: value }];
    }
    if (module.studioSelection.mode === "product_set_v1" && Array.isArray(value)) {
      return value.map((entry) => {
        if (typeof entry !== "string" || entry.trim() === "") {
          throw new Error(
            `configuration preflight requires canonical product tokens at '${module.selectionPath}'`,
          );
        }
        return { moduleId: module.id, schemeId: entry };
      });
    }
    return [];
  });
  if (selections.length === 0) {
    throw new Error("configuration preflight found no selected runtime schemes");
  }
  const zfpFieldTolerances: Record<string, number> = {};
  const configuredTolerances = readPath(outputConfig, "output.zfp_field_tolerance");
  if (configuredTolerances !== undefined) {
    if (
      configuredTolerances === null ||
      typeof configuredTolerances !== "object" ||
      Array.isArray(configuredTolerances)
    ) {
      throw new Error(
        "configuration preflight requires an object at 'output.zfp_field_tolerance'",
      );
    }
    for (const [field, value] of Object.entries(configuredTolerances).sort(
      ([left], [right]) => left.localeCompare(right),
    )) {
      if (
        !field ||
        field !== field.trim().toLowerCase() ||
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        value <= 0
      ) {
        throw new Error(
          `configuration preflight requires canonical positive finite 'output.zfp_field_tolerance.${field}'`,
        );
      }
      zfpFieldTolerances[field] = value;
    }
  }
  const optionalString = (path: string, fallback: string): string => {
    const value = optionalValue(path);
    return typeof value === "string" && value.trim() !== "" ? value : fallback;
  };
  const optionalBoolean = (path: string, fallback: boolean): boolean => {
    const value = optionalValue(path);
    return typeof value === "boolean" ? value : fallback;
  };
  const optionalFiniteNumber = (path: string, fallback: number): number => {
    const value = optionalValue(path);
    if (value === undefined) return fallback;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`configuration preflight requires finite number '${path}'`);
    }
    return value;
  };
  const optionalInteger = (path: string, fallback: number): number => {
    const value = optionalValue(path);
    if (value === undefined) return fallback;
    if (typeof value !== "number" || !Number.isSafeInteger(value)) {
      throw new Error(`configuration preflight requires integer '${path}'`);
    }
    return value;
  };
  const configuredAsyncIo = optionalValue("output.async_io");
  if (configuredAsyncIo !== undefined && typeof configuredAsyncIo !== "boolean") {
    throw new Error("configuration preflight requires boolean 'output.async_io'");
  }
  const asyncIo = configuredAsyncIo ?? false;
  const configuredDynamicCoefficient = readPath(config, "turbulence.dynamic_coefficient");
  const dynamicCoefficient = configuredDynamicCoefficient === undefined
    ? catalog.runtimeCapabilities.interactionPolicy.turbulenceDynamicCoefficientDefault
    : requiredBoolean("turbulence.dynamic_coefficient");
  if (typeof dynamicCoefficient !== "boolean") {
    throw new Error("configuration preflight has no exact-binary dynamic-coefficient default");
  }

  return {
    schema: "nimbus.config-preflight.v10",
    requestVersion: 10,
    source: sourceDocument === undefined
      ? { kind: "descriptor_only" }
      : { kind: "exact_document", yamlText: sourceDocument.yamlText,
          yamlPath: sourceDocument.yamlPath },
    executionMode: required("extensions.execution.mode"),
    maturityLimitsAcknowledged:
      readPath(config, "extensions.execution.acknowledge_maturity_limits") === undefined
        ? executionMode === "qualified"
        : readPath(config, "extensions.execution.acknowledge_maturity_limits") === true,
    backend: required("extensions.compute.backend"),
    allowBackendFallback: optionalBoolean(
      "extensions.compute.allow_fallback",
      required("extensions.execution.mode") !== "qualified",
    ),
    validationGuardMode: optionalString(
      "extensions.validation.guard_mode",
      required("extensions.execution.mode") === "qualified" ? "error" : "warn",
    ),
    gpuApplyMicrophysics: optionalBoolean("extensions.compute.gpu_apply_microphysics", true),
    gpuApplySlow: optionalBoolean("extensions.compute.gpu_apply_slow", false),
    gpuApplySgs: optionalBoolean("extensions.compute.gpu_apply_sgs", false),
    outputFormat: required("output.format"),
    gridMetrics: {
      nx: requiredInteger("grid.nx"),
      ny: requiredInteger("grid.ny"),
      nz: requiredInteger("grid.nz"),
      dx: requiredFiniteNumber("grid.dx"),
      dy: requiredFiniteNumber("grid.dy"),
      dz: requiredFiniteNumber("grid.dz"),
      ztop: requiredFiniteNumber("grid.ztop"),
    },
    timeControl: {
      dtLarge: requiredFiniteNumber("time.dt_large"),
      acousticSubsteps: requiredInteger("time.acoustic_substeps"),
      outputInterval: requiredFiniteNumber("time.output_interval"),
    },
    output: {
      preset: optionalString("output.preset", "minimal"),
      asyncIo,
      zfpMode: optionalString("output.zfp_mode", "accuracy"),
      zfpPerFieldTolerances:
        optionalBoolean("output.zfp_per_field_tolerances", false),
      zfpSparseThreshold:
        optionalFiniteNumber("output.zfp_sparse_threshold", 0),
      zfpFloat16Prefilter:
        optionalBoolean("output.zfp_float16_prefilter", false),
      zfpFieldExecution:
        optionalString("output.zfp_field_execution", "serial"),
      zfpFieldWorkers: optionalInteger("output.zfp_field_workers", 0),
      zfpFieldScratchMib:
        optionalInteger("output.zfp_field_scratch_mib", 64),
      zfpFieldTolerances,
      keepLastFrames: optionalInteger("output.keep_last_frames", 0),
      minimumFreeSpaceMib: optionalInteger("output.minimum_free_space_mib", 0),
      maximumRetainedHistoryFiles:
        optionalInteger("output.maximum_retained_history_files", 0),
    },
    coordinateSystem: required("grid.coordinate_system"),
    staggering: required("grid.staggering"),
    boundary: {
      west: required("boundary_conditions.lateral.west"),
      east: required("boundary_conditions.lateral.east"),
      south: required("boundary_conditions.lateral.south"),
      north: required("boundary_conditions.lateral.north"),
      bottom: required("boundary_conditions.bottom"),
      top: required("boundary_conditions.top"),
    },
    diagnostics: {
      scientificEverySteps:
        optionalInteger("extensions.diagnostics.scientific_every_steps", -1),
      deterministicReductions:
        optionalBoolean("extensions.diagnostics.deterministic_reductions", false),
    },
    initialization: {
      soundingSource: required("initial_conditions.sounding.type"),
      hodographSource: required("initial_conditions.hodograph.type"),
      triggerSource: required("initial_conditions.trigger.type"),
    },
    interactions: {
      radiationShortwaveEnabled: requiredBoolean("radiation.do_sw"),
      turbulenceMode: required("turbulence.mode"),
      turbulenceDynamicCoefficient: dynamicCoefficient,
      surfaceScheme: required("surface.scheme"),
      surfaceApplyFluxes: requiredBoolean("surface.apply_fluxes"),
      diffusionTarget: required("diffusion.apply_to"),
      diffusionActive:
        requiredFiniteNumber("diffusion.K_h") > 0 ||
        requiredFiniteNumber("diffusion.K_v") > 0,
    },
    availableStateFields: [...new Set(availableStateFields.map((field) => field.trim()))]
      .filter(Boolean)
      .sort(),
    selections,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string");
}

function isIssue(value: unknown): value is ConfigPreflightIssue {
  return (
    isRecord(value) &&
    typeof value.code === "string" &&
    (value.moduleId === null || typeof value.moduleId === "string") &&
    (value.schemeId === null || typeof value.schemeId === "string") &&
    typeof value.message === "string" &&
    isStringArray(value.expected) &&
    (value.actual === null || typeof value.actual === "string")
  );
}

function isResolvedSelection(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.moduleId === "string" &&
    typeof value.requestedSchemeId === "string" &&
    typeof value.canonicalSchemeId === "string" &&
    typeof value.maturity === "string" &&
    typeof value.constructible === "boolean" &&
    typeof value.runtimeSelectable === "boolean"
  );
}

function isContext(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.backend === "string" &&
    typeof value.executionMode === "string" &&
    typeof value.maturityLimitsAcknowledged === "boolean" &&
    typeof value.allowBackendFallback === "boolean" &&
    typeof value.validationGuardMode === "string" &&
    typeof value.gpuApplyMicrophysics === "boolean" &&
    typeof value.gpuApplySlow === "boolean" &&
    typeof value.gpuApplySgs === "boolean" &&
    typeof value.outputFormat === "string" &&
    isRecord(value.gridMetrics) &&
    Number.isSafeInteger(value.gridMetrics.nx) &&
    Number.isSafeInteger(value.gridMetrics.ny) &&
    Number.isSafeInteger(value.gridMetrics.nz) &&
    typeof value.gridMetrics.dx === "number" && Number.isFinite(value.gridMetrics.dx) &&
    typeof value.gridMetrics.dy === "number" && Number.isFinite(value.gridMetrics.dy) &&
    typeof value.gridMetrics.dz === "number" && Number.isFinite(value.gridMetrics.dz) &&
    typeof value.gridMetrics.ztop === "number" && Number.isFinite(value.gridMetrics.ztop) &&
    isRecord(value.timeControl) &&
    typeof value.timeControl.dtLarge === "number" && Number.isFinite(value.timeControl.dtLarge) &&
    Number.isSafeInteger(value.timeControl.acousticSubsteps) &&
    typeof value.timeControl.outputInterval === "number" &&
    Number.isFinite(value.timeControl.outputInterval) &&
    isRecord(value.output) &&
    typeof value.output.preset === "string" &&
    typeof value.output.asyncIo === "boolean" &&
    typeof value.output.zfpMode === "string" &&
    typeof value.output.zfpPerFieldTolerances === "boolean" &&
    typeof value.output.zfpSparseThreshold === "number" &&
    Number.isFinite(value.output.zfpSparseThreshold) &&
    typeof value.output.zfpFloat16Prefilter === "boolean" &&
    typeof value.output.zfpFieldExecution === "string" &&
    Number.isSafeInteger(value.output.zfpFieldWorkers) &&
    Number.isSafeInteger(value.output.zfpFieldScratchMib) &&
    Number.isSafeInteger(value.output.keepLastFrames) &&
    Number.isSafeInteger(value.output.minimumFreeSpaceMib) &&
    Number.isSafeInteger(value.output.maximumRetainedHistoryFiles) &&
    isRecord(value.output.zfpFieldTolerances) &&
    Object.entries(value.output.zfpFieldTolerances).every(
      ([field, tolerance]) =>
        field.length > 0 &&
        typeof tolerance === "number" &&
        Number.isFinite(tolerance),
    ) &&
    typeof value.coordinateSystem === "string" &&
    typeof value.staggering === "string" &&
    isRecord(value.boundary) &&
    typeof value.boundary.west === "string" &&
    typeof value.boundary.east === "string" &&
    typeof value.boundary.south === "string" &&
    typeof value.boundary.north === "string" &&
    typeof value.boundary.bottom === "string" &&
    typeof value.boundary.top === "string" &&
    isRecord(value.diagnostics) &&
    typeof value.diagnostics.scientificEverySteps === "number" &&
    Number.isSafeInteger(value.diagnostics.scientificEverySteps) &&
    typeof value.diagnostics.deterministicReductions === "boolean" &&
    isRecord(value.initialization) &&
    typeof value.initialization.soundingSource === "string" &&
    typeof value.initialization.hodographSource === "string" &&
    typeof value.initialization.triggerSource === "string" &&
    isRecord(value.interactions) &&
    typeof value.interactions.radiationShortwaveEnabled === "boolean" &&
    typeof value.interactions.turbulenceMode === "string" &&
    typeof value.interactions.turbulenceDynamicCoefficient === "boolean" &&
    typeof value.interactions.surfaceScheme === "string" &&
    typeof value.interactions.surfaceApplyFluxes === "boolean" &&
    typeof value.interactions.diffusionTarget === "string" &&
    typeof value.interactions.diffusionActive === "boolean" &&
    isStringArray(value.availableStateFields)
  );
}

/** Fail closed at the Tauri IPC boundary before a report can enable Run. */
export function validateConfigPreflightReport(value: unknown): ConfigPreflightReport {
  if (
    !isRecord(value) ||
    value.schema !== "nimbus.config-preflight-report.v10" ||
    value.reportVersion !== 10 ||
    typeof value.compatible !== "boolean" ||
    (value.executionTargetFingerprint != null && (typeof value.executionTargetFingerprint !== "string"
      || !/^sha256:[a-f0-9]{64}$/.test(value.executionTargetFingerprint))) ||
    !isRecord(value.nativeAdmission) ||
    !["accepted", "rejected", "not_checked"].includes(String(value.nativeAdmission.state)) ||
    value.nativeAdmission.descriptorProjection !== "not_checked" ||
    (value.nativeAdmission.sourceSha256 !== null &&
      (typeof value.nativeAdmission.sourceSha256 !== "string" ||
       !/^[a-f0-9]{64}$/.test(value.nativeAdmission.sourceSha256))) ||
    (value.nativeAdmission.sourceFingerprint !== null &&
      (typeof value.nativeAdmission.sourceFingerprint !== "string" ||
       !/^fnv1a64:[a-f0-9]{16}$/.test(value.nativeAdmission.sourceFingerprint))) ||
    (value.nativeAdmission.sourcePathFingerprint !== null &&
      (typeof value.nativeAdmission.sourcePathFingerprint !== "string" ||
       !/^fnv1a64:[a-f0-9]{16}$/.test(value.nativeAdmission.sourcePathFingerprint))) ||
    (value.nativeAdmission.validatedBinarySha256 !== null &&
      (typeof value.nativeAdmission.validatedBinarySha256 !== "string" ||
       !/^sha256:[a-f0-9]{64}$/.test(value.nativeAdmission.validatedBinarySha256))) ||
    (value.nativeAdmission.validationExitCode !== null &&
      !Number.isSafeInteger(value.nativeAdmission.validationExitCode)) ||
    !isRecord(value.executable) ||
    typeof value.executable.canonicalPath !== "string" ||
    typeof value.executable.byteLength !== "number" ||
    (value.executable.modifiedUnixMs !== null &&
      typeof value.executable.modifiedUnixMs !== "number") ||
    typeof value.executable.catalogFingerprint !== "string" ||
    typeof value.executable.binaryFingerprint !== "string" ||
    !isContext(value.context) ||
    !Array.isArray(value.resolvedSelections) ||
    !value.resolvedSelections.every(isResolvedSelection) ||
    !Array.isArray(value.hardErrors) ||
    !value.hardErrors.every(isIssue) ||
    !Array.isArray(value.warnings) ||
    !value.warnings.every(isIssue)
  ) {
    throw new Error("malformed configuration preflight report from Nimbus runtime");
  }
  if (value.compatible !== (value.hardErrors.length === 0)) {
    throw new Error("configuration preflight compatibility contradicts its hard errors");
  }
  if (value.compatible && value.nativeAdmission.state === "rejected") {
    throw new Error("configuration preflight claims compatibility after native rejection");
  }
  return value as unknown as ConfigPreflightReport;
}

/** Bind a structurally valid report to the request that is about to launch. */
function sameJsonValue(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
      left.every((value, index) => sameJsonValue(value, right[index]));
  }
  if (!isRecord(left) || !isRecord(right)) return false;
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every((key) =>
    Object.prototype.hasOwnProperty.call(right, key) && sameJsonValue(left[key], right[key]));
}

export function validateConfigPreflightEvidence(
  report: ConfigPreflightReport,
  request: ConfigPreflightRequest,
): ConfigPreflightReport {
  if (
    report.executable.canonicalPath.trim() === "" ||
    report.executable.catalogFingerprint.trim() === "" ||
    !/^sha256:[a-f0-9]{64}$/.test(report.executable.binaryFingerprint)
  ) {
    throw new Error("configuration preflight report has no executable identity");
  }
  if (request.source.kind === "exact_document") {
    if (report.nativeAdmission.state === "not_checked" ||
        report.nativeAdmission.sourceFingerprint !== fnv1a64Text(request.source.yamlText) ||
        report.nativeAdmission.sourcePathFingerprint !== fnv1a64Text(request.source.yamlPath) ||
        report.nativeAdmission.validatedBinarySha256 !== report.executable.binaryFingerprint ||
        (report.compatible && report.nativeAdmission.state !== "accepted")) {
      throw new Error("configuration preflight is not bound to the exact source and executable");
    }
  } else if (report.nativeAdmission.state !== "not_checked" ||
             report.nativeAdmission.sourceSha256 !== null ||
             report.nativeAdmission.sourceFingerprint !== null ||
             report.nativeAdmission.sourcePathFingerprint !== null ||
             report.nativeAdmission.validatedBinarySha256 !== null ||
             report.nativeAdmission.validationExitCode !== null) {
    throw new Error("descriptor-only configuration preflight invented native admission evidence");
  }
  const normalizeToken = (value: string): string =>
    value.trim().toLowerCase().replaceAll("-", "_");
  const requestedBoundary = Object.fromEntries(
    Object.entries(request.boundary).map(([face, mode]) => [
      face,
      normalizeToken(mode),
    ]),
  );
  if (!sameJsonValue(report.context.boundary, requestedBoundary)) {
    throw new Error("configuration preflight report does not match the requested boundaries");
  }
  if (!sameJsonValue(report.context.diagnostics, request.diagnostics)) {
    throw new Error("configuration preflight report does not match the requested diagnostics");
  }
  const requestedFields = [...request.availableStateFields].sort();
  const reportedFields = [...report.context.availableStateFields].sort();
  if (JSON.stringify(requestedFields) !== JSON.stringify(reportedFields)) {
    throw new Error("configuration preflight report does not match the requested state inventory");
  }
  if (report.context.outputFormat !== request.outputFormat.trim().toLowerCase().replaceAll("-", "_")) {
    throw new Error("configuration preflight report does not match the requested output format");
  }
  if (!sameJsonValue(report.context.gridMetrics, request.gridMetrics) ||
      !sameJsonValue(report.context.timeControl, request.timeControl)) {
    throw new Error("configuration preflight report does not match the requested grid/time envelope");
  }
  const requestedOutput = {
    ...request.output,
    preset: normalizeToken(request.output.preset),
    zfpMode: normalizeToken(request.output.zfpMode),
    zfpFieldExecution: normalizeToken(request.output.zfpFieldExecution),
    zfpFieldTolerances: Object.fromEntries(
      Object.entries(request.output.zfpFieldTolerances)
        .map(([field, tolerance]) => [normalizeToken(field), tolerance] as const)
        .sort(([left], [right]) => left.localeCompare(right)),
    ),
  };
  if (!sameJsonValue(report.context.output, requestedOutput)) {
    throw new Error("configuration preflight report does not match the requested output science");
  }
  if (
    report.context.executionMode !== request.executionMode.trim().toLowerCase() ||
    report.context.maturityLimitsAcknowledged !== request.maturityLimitsAcknowledged ||
    report.context.allowBackendFallback !== request.allowBackendFallback ||
    report.context.validationGuardMode !== normalizeToken(request.validationGuardMode) ||
    report.context.gpuApplyMicrophysics !== request.gpuApplyMicrophysics ||
    report.context.gpuApplySlow !== request.gpuApplySlow ||
    report.context.gpuApplySgs !== request.gpuApplySgs
  ) {
    throw new Error("configuration preflight report does not match the requested execution policy");
  }
  const requestedBackend = ["vulkan", "vk"].includes(normalizeToken(request.backend))
    ? "gpu"
    : normalizeToken(request.backend);
  if (
    report.context.backend !== requestedBackend ||
    report.context.coordinateSystem !== normalizeToken(request.coordinateSystem) ||
    report.context.staggering !== normalizeToken(request.staggering)
  ) {
    throw new Error("configuration preflight report does not match the requested execution geometry");
  }
  const normalizeSounding = (value: string): string => {
    const token = normalizeToken(value);
    if (["cape", "parametriccape"].includes(token)) return "parametric_cape";
    if (["sharpy", "raob", "sounding_file"].includes(token)) return "file";
    if (["targets", "diagnostics", "from_table", "parametrictargets"].includes(token)) {
      return "parametric_targets";
    }
    return token;
  };
  const normalizeHodograph = (value: string): string => {
    const token = normalizeToken(value);
    if (["wk", "wkparam", "custom"].includes(token)) return "wk_param";
    if (["static", "calm"].includes(token)) return "zero";
    return token;
  };
  const normalizeTrigger = (value: string): string => {
    const token = normalizeToken(value);
    if (["bubble", "warmbubble"].includes(token)) return "warm_bubble";
    if (["off", "disabled"].includes(token)) return "none";
    if (["vortex", "vortexseed", "rankine"].includes(token)) return "vortex_seed";
    return token;
  };
  const requestedInitialization = {
    soundingSource: normalizeSounding(request.initialization.soundingSource),
    hodographSource: normalizeHodograph(request.initialization.hodographSource),
    triggerSource: normalizeTrigger(request.initialization.triggerSource),
  };
  if (
    !sameJsonValue(report.context.initialization, requestedInitialization)
  ) {
    throw new Error("configuration preflight report does not match the requested initialization");
  }
  const requestedInteractions = {
    ...request.interactions,
    turbulenceMode: request.interactions.turbulenceMode.trim().toLowerCase().replaceAll("-", "_"),
    surfaceScheme: request.interactions.surfaceScheme.trim().toLowerCase().replaceAll("-", "_"),
    diffusionTarget: request.interactions.diffusionTarget.trim().toLowerCase().replaceAll("-", "_"),
  };
  if (!sameJsonValue(report.context.interactions, requestedInteractions)) {
    throw new Error("configuration preflight report does not match the requested science interactions");
  }
  if (report.compatible) {
    const requestedSelections = request.selections
      .map((selection) => `${selection.moduleId}\u0000${selection.schemeId}`)
      .sort();
    const resolvedSelections = report.resolvedSelections
      .map((selection) => `${selection.moduleId}\u0000${selection.requestedSchemeId}`)
      .sort();
    if (JSON.stringify(requestedSelections) !== JSON.stringify(resolvedSelections)) {
      throw new Error("configuration preflight report does not cover every requested selection");
    }
  }
  return report;
}

/** Invoke the desktop command and validate the entire IPC response. */
export async function preflightConfigWithBinary(
  request: ConfigPreflightRequest,
  simulatorPath?: string,
  executionTarget?: RemoteComputeTarget,
): Promise<ConfigPreflightReport> {
  if (
    typeof window === "undefined" ||
    !("__TAURI_INTERNALS__" in (window as Window & { __TAURI_INTERNALS__?: unknown }))
  ) {
    throw new Error("exact-binary configuration preflight requires Nimbus Studio desktop");
  }
  const response = await invokeCommand<unknown>("preflight_config", {
    simulatorPath,
    executionTarget,
    request,
  });
  const report = validateConfigPreflightEvidence(
    validateConfigPreflightReport(response),
    request,
  );
  if (executionTarget && !report.executionTargetFingerprint) {
    throw new Error("Remote preflight did not bind the compute target and input files.");
  }
  if (request.source.kind === "exact_document") {
    if (typeof crypto === "undefined" || !crypto.subtle) {
      throw new Error("exact configuration preflight requires SHA-256 verification");
    }
    const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(request.source.yamlText));
    const expected = Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (report.nativeAdmission.sourceSha256 !== expected) {
      throw new Error("configuration preflight source SHA-256 differs from the launched YAML");
    }
  }
  return report;
}
