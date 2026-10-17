import {
  SCHEMA_SECTIONS,
  qualifyKey,
  readPath,
  writePath,
  type ConfigObject,
  type EnumField,
  type FieldSchema,
  type SectionSchema,
} from "./configSchema";
import nativeCatalogSnapshot from "./nativeCatalogSnapshot.json" with { type: "json" };

export interface RuntimeSchemeParameter {
  path: string;
  displayName: string;
  kind: "float" | "boolean";
  default: number | boolean;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
  unit: string;
  description: string;
}

export interface RuntimeSchemeCapabilities {
  declared: boolean;
  coordinateSystems: string[];
  staggerings: string[];
  backends: string[];
  requiredStateFields: string[];
  producedStateFields: string[];
  optionalProducts: string[];
  acceptedSoundingSources: string[];
  acceptedHodographSources: string[];
  acceptedTriggerSources: string[];
  recommendedTriggerSource: string;
  configValidationContract: string;
  cpuReferenceCoverage: string;
  gpuImplementationCoverage: string;
  cpuGpuParityCoverage: string;
  storage: {
    coverage: string;
    persistentFullVolumeFields: number;
    peakScratchFullVolumeFields: number;
    persistentFieldsPerIntegratedField: number;
    peakScratchFieldsPerIntegratedField: number;
    note: string;
  };
}

export interface RuntimeScheme {
  id: string;
  aliases: string[];
  displayName: string;
  maturity: string;
  constructible: boolean;
  runtimeSelectable: boolean;
  readinessState: ReadinessState;
  caveat: string;
  parameters: RuntimeSchemeParameter[];
  capabilities: RuntimeSchemeCapabilities;
}

export type ReadinessState =
  | "engineering_incomplete"
  | "ready"
  | "campaign_validated"
  | "externally_reviewed"
  | "fully_qualified";

export interface ReadinessItem {
  id: string;
  area: string;
  scope: string;
  state: ReadinessState;
  boundedEvidence: string;
  campaignEvidence: string;
  externalReviewEvidence: string;
  blocker: string;
}

export interface ReadinessContract {
  version: 1;
  items: ReadinessItem[];
}

export interface RuntimeSchemeModule {
  id: string;
  selectionPath: string;
  studioSelection: {
    /** Exact strict-v1 exposure contract for this descriptor family. */
    mode: "form" | "product_set_v1" | "fixed_v1" | "unavailable_v1";
    /** Canonical identity selected by the v1 bridge when mode is fixed_v1. */
    fixedSchemeId: string;
    /** Required evidence for fixed/unavailable modes; empty for form. */
    reason: string;
  };
  schemes: RuntimeScheme[];
}

export interface RuntimeOutputFormatCapability {
  id: string;
  available: boolean;
  /** Exact-binary reason when unavailable; empty when available. */
  reason: string;
  implementation: string;
}

export const QUALIFIED_FLAT_MIXED_METAL_NPY_SCOPE = {
  "schema": "nimbus.qualified-flat-mixed-metal-npy-scope.v1",
  "version": 1,
  "profileCount": 1,
  "backend": "metal",
  "executionMode": "qualified",
  "familyId": "boundary:ed765facac077ef0b331283d",
  "strictCanonicalYamlSha256": "985c698858be9be4be29dc2adf819f31c20788c13a414747c422e21d7ebe51da",
  "outputFormats": [
    "npy_3d"
  ],
  "preset": "minimal",
  "writeCadenceSeconds": [
    1
  ],
  "asyncIo": false,
  "liveShm": false,
  "trajectories": false,
  "allowFallback": false,
  "memoryAdmissionPolicies": [
    "warn",
    "reject"
  ],
  "memoryPolicyCanonicalValue": "warn",
  "gridShape": [
    18,
    18,
    18
  ],
  "gridSpacingMeters": [
    500,
    500,
    250
  ],
  "topHeightMeters": 4500,
  "dtLargeSeconds": 0.1,
  "acousticSubsteps": 6,
  "adaptiveDt": false,
  "lateralBoundaries": [
    "rigid_slip",
    "rigid_slip",
    "rigid_slip",
    "rigid_slip"
  ],
  "bottomBoundary": "rigid_noslip",
  "topBoundary": "open",
  "scalarTransportOwner": "metal_native_conservative_mc",
  "initialConditionChaosOwner": "host",
  "lateralDampingOwner": "host",
  "lateralSpongeEnabled": true,
  "lateralSpongeWidth": 20,
  "lateralSpongeAlpha": 0.05,
  "wOnlyTopSponge": true,
  "wOnlyDepth": 8,
  "wOnlyAlpha": 0.2,
  "effectiveWOnlySponge": false,
  "sourceProfileOwnerSymbol": "is_public_qualified_flat_mixed_metal_cadence1_profile",
  "manifestSha256": "ea8921dabe62cceac5f95a023bdb92f3f5063dc9cd7aea6e8368925712014402"
};

export interface RuntimeInteractionPolicy {
  version: 9 | 10 | 11 | 12 | 13 | 14;
  qualifiedFlatMixedMetalNpyScope?: typeof QUALIFIED_FLAT_MIXED_METAL_NPY_SCOPE;
  generatedMetalCanonicalScope: {
    version: 2 | 3 | 4;
    backend: "metal";
    executionMode: "qualified";
    tupleCount: number;
    distinctConfigCount: number;
    outputFormats: string[];
    scalarTransportOwner: "native_conservative_mc";
    baseManifestSha256: string;
    manifestSha256: string;
    npyNetcdfManifestSha256?: string;
    zfpTypedOutputManifestSha256?: string;
    writeCadenceSecondsByFormat?: Record<string, number[]>;
    additionalBoundaryNpyScope?: {
      version: 1;
      profileCount: number;
      familyId: string;
      strictCanonicalYamlSha256: string;
      outputFormats: string[];
      writeCadenceSeconds: number[];
      manifestSha256: string;
    };
  };
  inactiveSchemeId: string;
  horizontalOnlyTurbulenceModes: string[];
  knownSurfaceSchemes: string[];
  activeSurfaceSchemeId: string;
  terrainRadiationSchemes: string[];
  terrainLongwaveRadiationSchemeId: string;
  terrainShortwaveAllowed: boolean;
  terrainBoundaryLayerSchemes: string[];
  terrainTurbulenceSchemes: string[];
  terrainChaosSchemes: string[];
  chaosCanonicalTargets: Record<string, {
    initialConditions: string[];
    tendencies: string[];
    nativeGenerationRequired: false;
  }>;
  terrainActiveDiffusionSchemes: string[];
  terrainActiveDiffusionTargets: string[];
  qualifiedZfpTerrainSchemes: string[];
  qualifiedZfpPresetIds: string[];
  qualifiedZfpPresetFields: Record<string, string[]>;
  qualifiedZfpMode: string;
  qualifiedZfpRequiresPerFieldTolerances: boolean;
  qualifiedZfpMinimumKeepLastFrames: number;
  qualifiedZfpMinimumFreeSpaceMib: number;
  qualifiedZfpRequiresRetainedFileCeiling: boolean;
  qualifiedExecutionRequiresMaturityAcknowledgment: false;
  qualifiedZfpMaxSparseThreshold: number;
  qualifiedZfpAllowsFloat16Prefilter: boolean;
  qualifiedZfpFieldBudgets: Record<string, number>;
  qualifiedZfpFieldUnits: Record<string, string>;
  qualifiedZfpReversibleRadarV4?: {
    version: 1;
    admitted: boolean;
    productIds: string[];
    nonemptyProductSubsets: 31;
    coreFields: string[];
    presets: string[];
    namedPresetProductSets: Record<string, string[]>;
    polarimetricExtraFields: string[];
    outputFormatTiers: string[];
    backend: "cpu";
    coordinateSystem: "cartesian";
    staggering: "c_grid";
    terrain: "none";
    domainMotion: "stationary";
    nesting: false;
    reviewedSourceProfileContract: "nimbus.qualified-radar-source-profile.v2";
    reviewedSourceProfileOwnerSymbol: "is_public_qualified_radar_profile";
    reviewedSourceProfileFamilies: string[];
    asyncIoModes: boolean[];
    gridShape: number[];
    gridSpacingMeters: number[];
    dtLargeSeconds: 0.2;
    acousticSubsteps: 6;
    writeCadenceSeconds: number[];
    nonRadarMode: "accuracy";
    nonRadarContainerVersion: 3;
    nonRadarBudgetOwner: "qualifiedZfpFieldBudgets";
    radarMode: "reversible";
    radarContainerVersion: 4;
    radarKeyframeEveryExport: true;
    radarTolerancePolicy: "forbidden";
    manifestContract: "nimbus.zfp.per_field.v2";
    sourceFieldBoundsContract: "registered_field_contract_v1";
  };
  qualifiedVerticalOnlySgs?: {
    version: 1;
    admitted: boolean;
    scheme: "smagorinsky";
    mode: "vertical_only";
    dynamicCoefficient: false;
    executionMode: "qualified";
    backend: "cpu";
    coordinateSystem: "cartesian";
    staggering: "c_grid";
    terrain: "none";
    domainMotion: "stationary";
    nesting: false;
    lateralBoundaryMode: "periodic";
    activeBoundaryLayerAllowed: false;
    gridShape: number[];
    gridSpacingMeters: number[];
    dtLargeSeconds: 0.2;
    acousticSubsteps: 6;
    outputFormat: "npy_3d";
    writeCadenceSeconds: number[];
    sourceProfileContract: "nimbus.static-vertical-sgs-cpu.v1";
    sourceProfileOwnerSymbol: "is_transactional_static_vertical_only_cpu_candidate";
    wholeStepRollback: "state_and_six_science_ledgers";
    dynamicNonThreeDAllowed: false;
  };
  turbulenceDynamicCoefficientDefault?: false;
  milbrandtTripleMoment?: {
    version: 1;
    retained: true;
    admitted: false;
    implementationState: "engineering_incomplete";
    selector: "microphysics.milbrandt.triple_moment";
    strictV1ExplicitKeyAdmitted: false;
    scheme: "milbrandt";
    baselineState: ["q", "N"];
    additionalState: ["Z"];
    additionalMomentOrder: 6;
    additionalMomentSpecies: ["rain", "ice", "snow", "graupel", "hail"];
    privateStorageUnits: "m^6/kg_dry";
  };
  lateralBoundaryModes: string[];
  radiativeLateralStaggerings: string[];
  legacyOpenCgridResolution: string;
  legacyOpenCollocatedResolution: string;
  periodicLateralPairsRequired: boolean;
  bottomBoundaryModes: string[];
  topBoundaryModes: string[];
  scientificDiagnosticsOmittedSentinel: -1;
  scientificDiagnosticsMinimumCadence: 0;
  gpuApplyControlsNoOpOnCpu: boolean;
  gpuApplyMicrophysicsSchemes: string[];
  gpuApplySgsSchemes: string[];
  gpuApplySlowDynamicsSchemes: string[];
  qualifiedGpuApplySlowDynamicsSchemes: string[];
  gpuApplyMicrophysicsDispatchGuardModes: string[];
  gpuApplyMicrophysicsHostCommitGuardModes: string[];
  gpuApplySlowDispatchGuardModes: string[];
  gpuApplySlowHostCommitGuardModes: string[];
  gpuApplySgsDispatchGuardModes: string[];
}

export interface RuntimeCapabilities {
  outputFormats: RuntimeOutputFormatCapability[];
  interactionPolicy: RuntimeInteractionPolicy;
}

export interface SchemeCatalog {
  schema: "nimbus.scheme-catalog.v9";
  catalogVersion: 9;
  /** Logical state/resources supplied by this binary's base runtime. */
  runtimeStateFields: string[];
  readiness: ReadinessContract;
  runtimeCapabilities: RuntimeCapabilities;
  modules: RuntimeSchemeModule[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

/** Validate the IPC boundary before catalog data is allowed to drive controls. */
export function validateSchemeCatalog(value: unknown): SchemeCatalog {
  if (!isRecord(value)) throw new Error("scheme catalog is not an object");
  if (value.schema !== "nimbus.scheme-catalog.v9" || value.catalogVersion !== 9) {
    throw new Error("unsupported simulator scheme catalog version");
  }
  if (!stringArray(value.runtimeStateFields) || value.runtimeStateFields.length === 0) {
    throw new Error("simulator scheme catalog contains no runtime state inventory");
  }
  const runtimeStateFields = new Set<string>();
  for (const field of value.runtimeStateFields) {
    if (!field || field !== field.trim() || runtimeStateFields.has(field)) {
      throw new Error("simulator runtime state inventory contains an empty or duplicate field");
    }
    runtimeStateFields.add(field);
  }
  if (!isRecord(value.readiness) || value.readiness.version !== 1 ||
      !Array.isArray(value.readiness.items) || value.readiness.items.length === 0) {
    throw new Error("simulator scheme catalog contains no readiness contract");
  }
  const readinessStates = new Set<ReadinessState>([
    "engineering_incomplete", "ready", "campaign_validated",
    "externally_reviewed", "fully_qualified",
  ]);
  const readinessIds = new Set<string>();
  for (const item of value.readiness.items) {
    if (!isRecord(item) || typeof item.id !== "string" ||
        typeof item.area !== "string" || typeof item.scope !== "string" ||
        typeof item.state !== "string" ||
        !readinessStates.has(item.state as ReadinessState) ||
        typeof item.boundedEvidence !== "string" ||
        typeof item.campaignEvidence !== "string" ||
        typeof item.externalReviewEvidence !== "string" ||
        typeof item.blocker !== "string" || !item.id || !item.area ||
        !item.scope || item.id !== item.id.trim().toLowerCase() ||
        readinessIds.has(item.id)) {
      throw new Error("simulator scheme catalog contains a malformed readiness item");
    }
    readinessIds.add(item.id);
    if (item.state === "engineering_incomplete") {
      if (!item.blocker.trim()) {
        throw new Error(`readiness item '${item.id}' has no engineering blocker`);
      }
    } else if (item.blocker || !item.boundedEvidence.trim()) {
      throw new Error(`readiness item '${item.id}' contradicts its ready state`);
    }
    if (["campaign_validated", "externally_reviewed", "fully_qualified"].includes(item.state) &&
        !item.campaignEvidence.trim()) {
      throw new Error(`readiness item '${item.id}' has no campaign evidence`);
    }
    if (["externally_reviewed", "fully_qualified"].includes(item.state) &&
        !item.externalReviewEvidence.trim()) {
      throw new Error(`readiness item '${item.id}' has no external review evidence`);
    }
  }
  if (
    !isRecord(value.runtimeCapabilities) ||
    !Array.isArray(value.runtimeCapabilities.outputFormats) ||
    value.runtimeCapabilities.outputFormats.length === 0
  ) {
    throw new Error("simulator scheme catalog contains no output-format capabilities");
  }
  const outputFormatIds = new Set<string>();
  for (const capability of value.runtimeCapabilities.outputFormats) {
    if (
      !isRecord(capability) ||
      typeof capability.id !== "string" ||
      typeof capability.available !== "boolean" ||
      typeof capability.reason !== "string" ||
      typeof capability.implementation !== "string" ||
      !capability.id ||
      capability.id !== capability.id.trim().toLowerCase() ||
      outputFormatIds.has(capability.id) ||
      (!capability.available && !capability.reason.trim()) ||
      (capability.available && !capability.implementation.trim())
    ) {
      throw new Error("simulator scheme catalog contains a malformed output-format capability");
    }
    outputFormatIds.add(capability.id);
  }
  if (!outputFormatIds.has("npy_3d") || !outputFormatIds.has("netcdf")) {
    throw new Error("simulator scheme catalog omits a required output-format capability");
  }
  const interaction = value.runtimeCapabilities.interactionPolicy;
  if (
    !isRecord(interaction) ||
    (interaction.version !== 9 && interaction.version !== 10 &&
      interaction.version !== 11 && interaction.version !== 12 &&
      interaction.version !== 13 && interaction.version !== 14) ||
    !isRecord(interaction.generatedMetalCanonicalScope) ||
    typeof interaction.inactiveSchemeId !== "string" ||
    !stringArray(interaction.horizontalOnlyTurbulenceModes) ||
    !stringArray(interaction.knownSurfaceSchemes) ||
    typeof interaction.activeSurfaceSchemeId !== "string" ||
    !stringArray(interaction.terrainRadiationSchemes) ||
    typeof interaction.terrainLongwaveRadiationSchemeId !== "string" ||
    typeof interaction.terrainShortwaveAllowed !== "boolean" ||
    !stringArray(interaction.terrainBoundaryLayerSchemes) ||
    !stringArray(interaction.terrainTurbulenceSchemes) ||
    !stringArray(interaction.terrainChaosSchemes) ||
    !isRecord(interaction.chaosCanonicalTargets) ||
    !stringArray(interaction.terrainActiveDiffusionSchemes) ||
    !stringArray(interaction.terrainActiveDiffusionTargets) ||
    !stringArray(interaction.qualifiedZfpTerrainSchemes) ||
    !stringArray(interaction.qualifiedZfpPresetIds) ||
    !isRecord(interaction.qualifiedZfpPresetFields) ||
    typeof interaction.qualifiedZfpMode !== "string" ||
    typeof interaction.qualifiedZfpRequiresPerFieldTolerances !== "boolean" ||
    typeof interaction.qualifiedZfpMinimumKeepLastFrames !== "number" ||
    !Number.isSafeInteger(interaction.qualifiedZfpMinimumKeepLastFrames) ||
    interaction.qualifiedZfpMinimumKeepLastFrames <= 0 ||
    typeof interaction.qualifiedZfpMinimumFreeSpaceMib !== "number" ||
    !Number.isSafeInteger(interaction.qualifiedZfpMinimumFreeSpaceMib) ||
    interaction.qualifiedZfpMinimumFreeSpaceMib <= 0 ||
    typeof interaction.qualifiedZfpRequiresRetainedFileCeiling !== "boolean" ||
    interaction.qualifiedExecutionRequiresMaturityAcknowledgment !== false ||
    typeof interaction.qualifiedZfpMaxSparseThreshold !== "number" ||
    !Number.isFinite(interaction.qualifiedZfpMaxSparseThreshold) ||
    typeof interaction.qualifiedZfpAllowsFloat16Prefilter !== "boolean" ||
    !isRecord(interaction.qualifiedZfpFieldBudgets) ||
    !isRecord(interaction.qualifiedZfpFieldUnits) ||
    !stringArray(interaction.lateralBoundaryModes) ||
    !stringArray(interaction.radiativeLateralStaggerings) ||
    typeof interaction.legacyOpenCgridResolution !== "string" ||
    typeof interaction.legacyOpenCollocatedResolution !== "string" ||
    typeof interaction.periodicLateralPairsRequired !== "boolean" ||
    !stringArray(interaction.bottomBoundaryModes) ||
    !stringArray(interaction.topBoundaryModes) ||
    interaction.scientificDiagnosticsOmittedSentinel !== -1 ||
    interaction.scientificDiagnosticsMinimumCadence !== 0 ||
    typeof interaction.gpuApplyControlsNoOpOnCpu !== "boolean" ||
    !stringArray(interaction.gpuApplyMicrophysicsSchemes) ||
    !stringArray(interaction.gpuApplySgsSchemes) ||
    !stringArray(interaction.gpuApplySlowDynamicsSchemes) ||
    !stringArray(interaction.qualifiedGpuApplySlowDynamicsSchemes) ||
    !stringArray(interaction.gpuApplyMicrophysicsDispatchGuardModes) ||
    !stringArray(interaction.gpuApplyMicrophysicsHostCommitGuardModes) ||
    !stringArray(interaction.gpuApplySlowDispatchGuardModes) ||
    !stringArray(interaction.gpuApplySlowHostCommitGuardModes) ||
    !stringArray(interaction.gpuApplySgsDispatchGuardModes)
  ) {
    throw new Error("simulator scheme catalog contains no valid interaction policy");
  }
  const radarZfp = interaction.qualifiedZfpReversibleRadarV4;
  const radarScopePresent = Object.prototype.hasOwnProperty.call(
    interaction, "qualifiedZfpReversibleRadarV4");
  const radarBudgets = interaction.qualifiedZfpFieldBudgets as Record<string, unknown>;
  const same = (actual: unknown, expected: unknown): boolean =>
    JSON.stringify(actual) === JSON.stringify(expected);
  if (interaction.version === 11 || interaction.version === 12 || interaction.version === 13 || interaction.version === 14) {
    const radarKeys = [
      "version", "admitted", "productIds", "nonemptyProductSubsets", "coreFields",
      "presets", "namedPresetProductSets", "polarimetricExtraFields", "outputFormatTiers",
      "backend", "coordinateSystem", "staggering", "terrain", "domainMotion",
      "nesting", "reviewedSourceProfileContract", "reviewedSourceProfileOwnerSymbol",
      "reviewedSourceProfileFamilies",
      "asyncIoModes", "gridShape", "gridSpacingMeters", "dtLargeSeconds",
      "acousticSubsteps", "writeCadenceSeconds", "nonRadarMode",
      "nonRadarContainerVersion", "nonRadarBudgetOwner", "radarMode",
      "radarContainerVersion", "radarKeyframeEveryExport", "radarTolerancePolicy",
      "manifestContract", "sourceFieldBoundsContract",
    ];
    const radarProducts = ["reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"];
    const radarCore = ["u", "v", "w", "theta", "p", "rho", "qv", "qr"];
    if (
      !radarScopePresent || !isRecord(radarZfp) ||
      !same(Object.keys(radarZfp).sort(), radarKeys.sort()) ||
      radarZfp.version !== 1 || typeof radarZfp.admitted !== "boolean" ||
      !same(radarZfp.productIds, radarProducts) ||
      radarZfp.nonemptyProductSubsets !== 31 ||
      !same(radarZfp.coreFields, radarCore) ||
      !same(radarZfp.presets, ["minimal", "radar_ready", "radar_polarimetric_ready"]) ||
      !isRecord(radarZfp.namedPresetProductSets) ||
      !same(Object.keys(radarZfp.namedPresetProductSets).sort(),
        ["radar_ready", "radar_polarimetric_ready"].sort()) ||
      !same(radarZfp.namedPresetProductSets.radar_ready,
        ["reflectivity_dbz", "radial_velocity"]) ||
      !same(radarZfp.namedPresetProductSets.radar_polarimetric_ready,
        ["zdr", "kdp", "rhohv"]) ||
      !same(radarZfp.polarimetricExtraFields, ["qc"]) ||
      !same(radarZfp.outputFormatTiers, ["zfp_low", "zfp_medium", "zfp_high"]) ||
      radarZfp.backend !== "cpu" || radarZfp.coordinateSystem !== "cartesian" ||
      radarZfp.staggering !== "c_grid" || radarZfp.terrain !== "none" ||
      radarZfp.domainMotion !== "stationary" || radarZfp.nesting !== false ||
      radarZfp.reviewedSourceProfileContract !== "nimbus.qualified-radar-source-profile.v2" ||
      radarZfp.reviewedSourceProfileOwnerSymbol !== "is_public_qualified_radar_profile" ||
      !same(radarZfp.reviewedSourceProfileFamilies,
        ["lin_cloudy_radar_ready_v1", "kessler_warm_cloud_radar_v1"]) ||
      !same(radarZfp.asyncIoModes, [false, true]) ||
      !same(radarZfp.gridShape, [12, 12, 48]) ||
      !same(radarZfp.gridSpacingMeters, [1000, 1000, 250]) ||
      radarZfp.dtLargeSeconds !== 0.2 || radarZfp.acousticSubsteps !== 6 ||
      !same(radarZfp.writeCadenceSeconds, [1, 4]) ||
      radarZfp.nonRadarMode !== "accuracy" ||
      radarZfp.nonRadarContainerVersion !== 3 ||
      radarZfp.nonRadarBudgetOwner !== "qualifiedZfpFieldBudgets" ||
      radarZfp.radarMode !== "reversible" || radarZfp.radarContainerVersion !== 4 ||
      radarZfp.radarKeyframeEveryExport !== true ||
      radarZfp.radarTolerancePolicy !== "forbidden" ||
      radarZfp.manifestContract !== "nimbus.zfp.per_field.v2" ||
      radarZfp.sourceFieldBoundsContract !== "registered_field_contract_v1" ||
      radarCore.some((field) => !(field in radarBudgets)) ||
      !("qc" in radarBudgets) ||
      radarProducts.some((field) => field in radarBudgets)
    ) {
      throw new Error("simulator scheme catalog contains a malformed reversible radar ZFP scope");
    }
  } else if (radarScopePresent) {
    throw new Error("historical interaction policy cannot advertise reversible radar ZFP");
  }
  const verticalSgs = interaction.qualifiedVerticalOnlySgs;
  const verticalSgsPresent = Object.prototype.hasOwnProperty.call(
    interaction, "qualifiedVerticalOnlySgs");
  const dynamicDefaultPresent = Object.prototype.hasOwnProperty.call(
    interaction, "turbulenceDynamicCoefficientDefault");
  if (interaction.version === 12 || interaction.version === 13 || interaction.version === 14) {
    const keys = [
      "version", "admitted", "scheme", "mode", "dynamicCoefficient",
      "executionMode", "backend", "coordinateSystem", "staggering", "terrain",
      "domainMotion", "nesting", "lateralBoundaryMode", "activeBoundaryLayerAllowed",
      "gridShape", "gridSpacingMeters", "dtLargeSeconds", "acousticSubsteps",
      "outputFormat", "writeCadenceSeconds", "sourceProfileContract",
      "sourceProfileOwnerSymbol", "wholeStepRollback", "dynamicNonThreeDAllowed",
    ];
    if (!verticalSgsPresent || !dynamicDefaultPresent ||
        interaction.turbulenceDynamicCoefficientDefault !== false ||
        !isRecord(verticalSgs) ||
        !same(Object.keys(verticalSgs).sort(), keys.sort()) ||
        verticalSgs.version !== 1 || typeof verticalSgs.admitted !== "boolean" ||
        verticalSgs.scheme !== "smagorinsky" || verticalSgs.mode !== "vertical_only" ||
        verticalSgs.dynamicCoefficient !== false ||
        verticalSgs.executionMode !== "qualified" || verticalSgs.backend !== "cpu" ||
        verticalSgs.coordinateSystem !== "cartesian" || verticalSgs.staggering !== "c_grid" ||
        verticalSgs.terrain !== "none" || verticalSgs.domainMotion !== "stationary" ||
        verticalSgs.nesting !== false || verticalSgs.lateralBoundaryMode !== "periodic" ||
        verticalSgs.activeBoundaryLayerAllowed !== false ||
        !same(verticalSgs.gridShape, [12, 12, 16]) ||
        !same(verticalSgs.gridSpacingMeters, [1000, 1000, 250]) ||
        verticalSgs.dtLargeSeconds !== 0.2 || verticalSgs.acousticSubsteps !== 6 ||
        verticalSgs.outputFormat !== "npy_3d" ||
        !same(verticalSgs.writeCadenceSeconds, [1]) ||
        verticalSgs.sourceProfileContract !== "nimbus.static-vertical-sgs-cpu.v1" ||
        verticalSgs.sourceProfileOwnerSymbol !==
          "is_transactional_static_vertical_only_cpu_candidate" ||
        verticalSgs.wholeStepRollback !== "state_and_six_science_ledgers" ||
        verticalSgs.dynamicNonThreeDAllowed !== false) {
      throw new Error("simulator scheme catalog contains a malformed static vertical SGS scope");
    }
  } else if (verticalSgsPresent || dynamicDefaultPresent) {
    throw new Error("historical interaction policy cannot advertise vertical-only SGS defaults");
  }
  const triple = interaction.milbrandtTripleMoment;
  const triplePresent = Object.prototype.hasOwnProperty.call(interaction, "milbrandtTripleMoment");
  if (interaction.version === 13 || interaction.version === 14) {
    const keys = [
      "version", "retained", "admitted", "implementationState", "selector",
      "strictV1ExplicitKeyAdmitted", "scheme", "baselineState", "additionalState",
      "additionalMomentOrder", "additionalMomentSpecies", "privateStorageUnits",
    ];
    if (!triplePresent || !isRecord(triple) ||
        !same(Object.keys(triple).sort(), keys.sort()) ||
        triple.version !== 1 || triple.retained !== true || triple.admitted !== false ||
        triple.implementationState !== "engineering_incomplete" ||
        triple.selector !== "microphysics.milbrandt.triple_moment" ||
        triple.strictV1ExplicitKeyAdmitted !== false || triple.scheme !== "milbrandt" ||
        !same(triple.baselineState, ["q", "N"]) ||
        !same(triple.additionalState, ["Z"]) || triple.additionalMomentOrder !== 6 ||
        !same(triple.additionalMomentSpecies, ["rain", "ice", "snow", "graupel", "hail"]) ||
        triple.privateStorageUnits !== "m^6/kg_dry") {
      throw new Error("simulator scheme catalog contains a malformed Milbrandt triple-moment scope");
    }
  } else if (triplePresent) {
    throw new Error("historical interaction policy cannot advertise Milbrandt triple moment");
  }
  const tripleReadiness = value.readiness.items.filter((item: unknown) =>
    isRecord(item) && item.id === "microphysics.milbrandt_triple_moment");
  if (interaction.version === 13 || interaction.version === 14) {
    if (tripleReadiness.length !== 1 ||
        !isRecord(tripleReadiness[0]) ||
        tripleReadiness[0].state !== "engineering_incomplete" ||
        typeof tripleReadiness[0].blocker !== "string" ||
        !tripleReadiness[0].blocker.trim()) {
      throw new Error("simulator scheme catalog omits the incomplete Milbrandt triple-moment readiness row");
    }
  } else if (tripleReadiness.length !== 0) {
    throw new Error("historical interaction policy cannot advertise Milbrandt triple-moment readiness");
  }
  const mixedScope = interaction.qualifiedFlatMixedMetalNpyScope;
  if (interaction.version === 14) {
    const ordered = (record: Record<string, unknown>) =>
      Object.entries(record).sort(([left], [right]) => left.localeCompare(right));
    if (!isRecord(mixedScope) ||
        JSON.stringify(ordered(mixedScope)) !==
          JSON.stringify(ordered(QUALIFIED_FLAT_MIXED_METAL_NPY_SCOPE))) {
      throw new Error("simulator scheme catalog contains an invalid exact mixed-open Metal NPY scope");
    }
  } else if (mixedScope !== undefined) {
    throw new Error("historical interaction policy cannot advertise mixed-open Metal admission");
  }
  const generatedMetalScope = interaction.generatedMetalCanonicalScope;
  if (
    (generatedMetalScope.version !== 2 && generatedMetalScope.version !== 3 &&
      generatedMetalScope.version !== 4) ||
    generatedMetalScope.backend !== "metal" ||
    generatedMetalScope.executionMode !== "qualified" ||
    !Number.isSafeInteger(generatedMetalScope.tupleCount) ||
    (generatedMetalScope.tupleCount as number) <= 0 ||
    !Number.isSafeInteger(generatedMetalScope.distinctConfigCount) ||
    (generatedMetalScope.distinctConfigCount as number) <= 0 ||
    (generatedMetalScope.distinctConfigCount as number) >
      (generatedMetalScope.tupleCount as number) ||
    !stringArray(generatedMetalScope.outputFormats) ||
    generatedMetalScope.outputFormats.length === 0 ||
    new Set(generatedMetalScope.outputFormats).size !== generatedMetalScope.outputFormats.length ||
    generatedMetalScope.outputFormats.some((format) => !outputFormatIds.has(format)) ||
    generatedMetalScope.scalarTransportOwner !== "native_conservative_mc" ||
    typeof generatedMetalScope.baseManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(generatedMetalScope.baseManifestSha256) ||
    typeof generatedMetalScope.manifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(generatedMetalScope.manifestSha256)
  ) {
    throw new Error("simulator scheme catalog contains a malformed generated Metal scope");
  }
  const extension = generatedMetalScope.additionalBoundaryNpyScope;
  const validEastExtension = isRecord(extension) &&
    JSON.stringify(Object.keys(extension).sort()) === JSON.stringify([
      "familyId", "manifestSha256", "outputFormats", "profileCount",
      "strictCanonicalYamlSha256", "version", "writeCadenceSeconds",
    ]) &&
    extension.version === 1 && extension.profileCount === 1 &&
    extension.familyId === "boundary:2b5bfbe6dfa466c6baf991e3" &&
    extension.strictCanonicalYamlSha256 ===
      "a244b58833f7e3cd7060d941e8453d4e5face3dff9e18e87ebf61dd581a24d55" &&
    extension.manifestSha256 ===
      "26712e5d5714b313cc1e17f348d67b9db9eeea60e5b888d56b0b7cf3cc5673a9" &&
    JSON.stringify(extension.outputFormats) === JSON.stringify(["npy_3d"]) &&
    JSON.stringify(extension.writeCadenceSeconds) === JSON.stringify([1]);
  const cadences = generatedMetalScope.writeCadenceSecondsByFormat;
  const cadenceMap = isRecord(cadences) ? cadences : {};
  const validBaseCadences = isRecord(cadences) &&
    JSON.stringify(cadences.npy_3d) === JSON.stringify([1, 301]) &&
    JSON.stringify(cadences.netcdf) === JSON.stringify([1, 301]);
  if (generatedMetalScope.version === 2) {
    if (
      interaction.version !== 9 || extension !== undefined ||
      generatedMetalScope.tupleCount !== 4032 ||
      generatedMetalScope.distinctConfigCount !== 3024 ||
      JSON.stringify(generatedMetalScope.outputFormats) !== JSON.stringify(["npy_3d", "netcdf"])
    ) {
      throw new Error("simulator scheme catalog contains an inconsistent v2 generated Metal scope");
    }
  } else if (generatedMetalScope.version === 3 && interaction.version === 9) {
    if (
      extension !== undefined ||
      generatedMetalScope.tupleCount !== 6048 ||
      generatedMetalScope.distinctConfigCount !== 4536 ||
      JSON.stringify(generatedMetalScope.outputFormats) !==
        JSON.stringify(["npy_3d", "netcdf", "zfp_high"]) ||
      typeof generatedMetalScope.npyNetcdfManifestSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(generatedMetalScope.npyNetcdfManifestSha256) ||
      typeof generatedMetalScope.zfpTypedOutputManifestSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(generatedMetalScope.zfpTypedOutputManifestSha256) ||
      !validBaseCadences ||
      JSON.stringify(Object.keys(cadenceMap).sort()) !==
        JSON.stringify(["netcdf", "npy_3d", "zfp_high"]) ||
      JSON.stringify(cadenceMap.zfp_high) !== JSON.stringify([1])
    ) {
      throw new Error("simulator scheme catalog contains an inconsistent v3 generated Metal scope");
    }
  } else if (generatedMetalScope.version === 3 &&
             (interaction.version === 10 || interaction.version === 11 ||
               interaction.version === 12 || interaction.version === 13 || interaction.version === 14)) {
    if (
      !validEastExtension || !validBaseCadences ||
      generatedMetalScope.tupleCount !== 4033 ||
      generatedMetalScope.distinctConfigCount !== 3025 ||
      JSON.stringify(generatedMetalScope.outputFormats) !== JSON.stringify(["npy_3d", "netcdf"]) ||
      JSON.stringify(Object.keys(cadenceMap).sort()) !== JSON.stringify(["netcdf", "npy_3d"]) ||
      generatedMetalScope.baseManifestSha256 !==
        "85642f094c418fe95eb4c2fa0ba5a2a947aec95f8f1c43b17abcd924b0f068a0" ||
      generatedMetalScope.npyNetcdfManifestSha256 !==
        "30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99" ||
      generatedMetalScope.zfpTypedOutputManifestSha256 !== undefined ||
      generatedMetalScope.manifestSha256 !==
        "4604b059d425fb86690207f69000fd708f153054d739091e0ec2ba00c7d9a4e3"
    ) {
      throw new Error("simulator scheme catalog contains an inconsistent no-ZFP v4 east-NPY scope");
    }
  } else if (generatedMetalScope.version === 4 &&
             (interaction.version === 10 || interaction.version === 11 ||
               interaction.version === 12 || interaction.version === 13 || interaction.version === 14)) {
    if (
      !validEastExtension || !validBaseCadences ||
      generatedMetalScope.tupleCount !== 6049 ||
      generatedMetalScope.distinctConfigCount !== 4537 ||
      JSON.stringify(generatedMetalScope.outputFormats) !==
        JSON.stringify(["npy_3d", "netcdf", "zfp_high"]) ||
      JSON.stringify(Object.keys(cadenceMap).sort()) !==
        JSON.stringify(["netcdf", "npy_3d", "zfp_high"]) ||
      JSON.stringify(cadenceMap.zfp_high) !== JSON.stringify([1]) ||
      generatedMetalScope.baseManifestSha256 !==
        "85642f094c418fe95eb4c2fa0ba5a2a947aec95f8f1c43b17abcd924b0f068a0" ||
      generatedMetalScope.npyNetcdfManifestSha256 !==
        "30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99" ||
      generatedMetalScope.zfpTypedOutputManifestSha256 !==
        "89ef136c9637c6e23d17cd996af20631973286447b783f1978d1471d4982c9fc" ||
      generatedMetalScope.manifestSha256 !==
        "062b609452e090b0b6e5ff1f7bb15cab896700f193796ff4d87a600994f28967"
    ) {
      throw new Error("simulator scheme catalog contains an inconsistent v4 east-NPY Metal scope");
    }
  } else {
    throw new Error("simulator scheme catalog contains an unsupported generated Metal scope version");
  }
  const terrainChaosSchemes = interaction.terrainChaosSchemes as string[];
  const chaosCanonicalTargets = interaction.chaosCanonicalTargets as Record<string, unknown>;
  const canonicalChaosIds = Object.keys(chaosCanonicalTargets).sort();
  if (
    canonicalChaosIds.length !== terrainChaosSchemes.length ||
    canonicalChaosIds.some((id, index) => id !== [...terrainChaosSchemes].sort()[index]) ||
    canonicalChaosIds.some((id) => {
      const targets = chaosCanonicalTargets[id];
      return !isRecord(targets) ||
        !stringArray(targets.initialConditions) ||
        !stringArray(targets.tendencies) ||
        targets.nativeGenerationRequired !== false;
    })
  ) {
    throw new Error("simulator scheme catalog contains malformed canonical chaos targets");
  }
  const interactionLists = [
    interaction.horizontalOnlyTurbulenceModes,
    interaction.knownSurfaceSchemes,
    interaction.terrainRadiationSchemes,
    interaction.terrainBoundaryLayerSchemes,
    interaction.terrainTurbulenceSchemes,
    interaction.terrainChaosSchemes,
    interaction.terrainActiveDiffusionSchemes,
    interaction.terrainActiveDiffusionTargets,
    interaction.qualifiedZfpTerrainSchemes,
    interaction.qualifiedZfpPresetIds,
    interaction.lateralBoundaryModes,
    interaction.radiativeLateralStaggerings,
    interaction.bottomBoundaryModes,
    interaction.topBoundaryModes,
    interaction.gpuApplyMicrophysicsSchemes,
    interaction.gpuApplySgsSchemes,
    interaction.gpuApplySlowDynamicsSchemes,
    interaction.qualifiedGpuApplySlowDynamicsSchemes,
    interaction.gpuApplyMicrophysicsDispatchGuardModes,
    interaction.gpuApplyMicrophysicsHostCommitGuardModes,
    interaction.gpuApplySlowDispatchGuardModes,
    interaction.gpuApplySlowHostCommitGuardModes,
    interaction.gpuApplySgsDispatchGuardModes,
  ];
  const qualifiedZfpFieldBudgets = interaction.qualifiedZfpFieldBudgets;
  const qualifiedZfpFieldUnits = interaction.qualifiedZfpFieldUnits;
  const qualifiedZfpPresetFields = interaction.qualifiedZfpPresetFields;
  const qualifiedZfpPresetIds = interaction.qualifiedZfpPresetIds;
  if (
    !interaction.inactiveSchemeId ||
    !interaction.activeSurfaceSchemeId ||
    !interaction.terrainLongwaveRadiationSchemeId ||
    !interaction.qualifiedZfpMode ||
    !interaction.legacyOpenCgridResolution ||
    !interaction.legacyOpenCollocatedResolution ||
    interactionLists.some((values) =>
      values.length === 0 ||
      values.some((entry) => !entry || entry !== entry.trim().toLowerCase()) ||
      new Set(values).size !== values.length
    ) ||
    !interaction.knownSurfaceSchemes.includes(interaction.inactiveSchemeId) ||
    !interaction.knownSurfaceSchemes.includes(interaction.activeSurfaceSchemeId) ||
    !interaction.terrainRadiationSchemes.includes(interaction.terrainLongwaveRadiationSchemeId) ||
    !interaction.lateralBoundaryModes.includes(interaction.legacyOpenCgridResolution) ||
    !interaction.lateralBoundaryModes.includes(interaction.legacyOpenCollocatedResolution) ||
    !interaction.radiativeLateralStaggerings.includes("c_grid") ||
    !interaction.periodicLateralPairsRequired ||
    Object.keys(qualifiedZfpFieldBudgets).length === 0 ||
    Object.entries(qualifiedZfpFieldBudgets).some(
      ([field, budget]) =>
        !field ||
        field !== field.trim().toLowerCase() ||
        typeof budget !== "number" ||
        !Number.isFinite(budget) ||
        budget <= 0,
    ) ||
    Object.keys(qualifiedZfpFieldUnits).length !==
      Object.keys(qualifiedZfpFieldBudgets).length ||
    Object.entries(qualifiedZfpFieldUnits).some(
      ([field, unit]) =>
        !(field in qualifiedZfpFieldBudgets) ||
        typeof unit !== "string" || unit.trim() === "",
    ) ||
    Object.keys(qualifiedZfpPresetFields).length !== qualifiedZfpPresetIds.length ||
    qualifiedZfpPresetIds.some((preset) => {
      const fields = qualifiedZfpPresetFields[preset];
      return !stringArray(fields) || fields.length === 0 ||
        new Set(fields).size !== fields.length ||
        fields.some((field) =>
          !field || field !== field.trim().toLowerCase() ||
          !(field in qualifiedZfpFieldBudgets)
        );
    }) ||
    Object.keys(qualifiedZfpPresetFields).some(
      (preset) => !qualifiedZfpPresetIds.includes(preset),
    )
  ) {
    throw new Error("simulator scheme catalog contains a malformed interaction policy");
  }
  if (!Array.isArray(value.modules) || value.modules.length === 0) {
    throw new Error("simulator scheme catalog contains no modules");
  }

  const moduleIds = new Set<string>();
  const selectionPaths = new Set<string>();
  for (const rawModule of value.modules) {
    if (
      !isRecord(rawModule) ||
      typeof rawModule.id !== "string" ||
      typeof rawModule.selectionPath !== "string" ||
      !isRecord(rawModule.studioSelection) ||
      !Array.isArray(rawModule.schemes)
    ) {
      throw new Error("simulator scheme catalog contains a malformed module");
    }
    if (moduleIds.has(rawModule.id)) {
      throw new Error(`duplicate scheme module '${rawModule.id}'`);
    }
    if (!rawModule.id || rawModule.id !== rawModule.id.trim().toLowerCase()) {
      throw new Error("simulator scheme catalog contains a non-canonical module id");
    }
    if (!rawModule.selectionPath || rawModule.selectionPath !== rawModule.selectionPath.trim()) {
      throw new Error(`module '${rawModule.id}' contains an invalid selection path`);
    }
    if (selectionPaths.has(rawModule.selectionPath)) {
      throw new Error(`duplicate scheme selection path '${rawModule.selectionPath}'`);
    }
    moduleIds.add(rawModule.id);
    selectionPaths.add(rawModule.selectionPath);
    const studioSelection = rawModule.studioSelection;
    if (
      (studioSelection.mode !== "form" &&
        studioSelection.mode !== "product_set_v1" &&
        studioSelection.mode !== "fixed_v1" &&
        studioSelection.mode !== "unavailable_v1") ||
      typeof studioSelection.fixedSchemeId !== "string" ||
      typeof studioSelection.reason !== "string"
    ) {
      throw new Error(`module '${rawModule.id}' contains a malformed Studio selection policy`);
    }
    if (
      (studioSelection.mode === "form" &&
        (studioSelection.fixedSchemeId !== "" || studioSelection.reason !== "")) ||
      (studioSelection.mode === "product_set_v1" &&
        (studioSelection.fixedSchemeId !== "" || !studioSelection.reason.trim())) ||
      (studioSelection.mode === "fixed_v1" &&
        (!studioSelection.fixedSchemeId || !studioSelection.reason.trim())) ||
      (studioSelection.mode === "unavailable_v1" &&
        (studioSelection.fixedSchemeId !== "" || !studioSelection.reason.trim()))
    ) {
      throw new Error(`module '${rawModule.id}' contains an inconsistent Studio selection policy`);
    }
    const schemeIds = new Set<string>();
    const schemeNames = new Set<string>();
    for (const rawScheme of rawModule.schemes) {
      if (
        !isRecord(rawScheme) ||
        typeof rawScheme.id !== "string" ||
        !stringArray(rawScheme.aliases) ||
        typeof rawScheme.displayName !== "string" ||
        typeof rawScheme.maturity !== "string" ||
        typeof rawScheme.constructible !== "boolean" ||
        typeof rawScheme.runtimeSelectable !== "boolean" ||
        typeof rawScheme.readinessState !== "string" ||
        !readinessStates.has(rawScheme.readinessState as ReadinessState) ||
        typeof rawScheme.caveat !== "string" ||
        !Array.isArray(rawScheme.parameters) ||
        !isRecord(rawScheme.capabilities)
      ) {
        throw new Error(`module '${rawModule.id}' contains a malformed scheme`);
      }
      if (schemeIds.has(rawScheme.id)) {
        throw new Error(`duplicate scheme '${rawModule.id}.${rawScheme.id}'`);
      }
      if (!rawScheme.id || rawScheme.id !== rawScheme.id.trim().toLowerCase()) {
        throw new Error(`module '${rawModule.id}' contains a non-canonical scheme id`);
      }
      if (!rawScheme.displayName.trim() || !rawScheme.maturity.trim()) {
        throw new Error(`scheme '${rawModule.id}.${rawScheme.id}' has incomplete presentation metadata`);
      }
      if (rawScheme.readinessState !== "engineering_incomplete" &&
          (!rawScheme.constructible || !rawScheme.runtimeSelectable ||
           rawScheme.maturity !== "qualified")) {
        throw new Error(`scheme '${rawModule.id}.${rawScheme.id}' contradicts its readiness state`);
      }
      const identities = [rawScheme.id, ...rawScheme.aliases];
      for (const identity of identities) {
        if (!identity || identity !== identity.trim().toLowerCase() || schemeNames.has(identity)) {
          throw new Error(`scheme identity '${rawModule.id}.${identity}' is invalid or duplicated`);
        }
        schemeNames.add(identity);
      }
      schemeIds.add(rawScheme.id);
      const capabilities = rawScheme.capabilities;
      const capabilityArrays = [
        capabilities.coordinateSystems,
        capabilities.staggerings,
        capabilities.backends,
        capabilities.requiredStateFields,
        capabilities.producedStateFields,
        capabilities.optionalProducts,
        capabilities.acceptedSoundingSources,
        capabilities.acceptedHodographSources,
        capabilities.acceptedTriggerSources,
      ];
      if (
        typeof capabilities.declared !== "boolean" ||
        !capabilityArrays.every(stringArray) ||
        capabilityArrays.slice(0, 3).some((entries) => entries.length === 0) ||
        typeof capabilities.recommendedTriggerSource !== "string" ||
        typeof capabilities.configValidationContract !== "string" ||
        typeof capabilities.cpuReferenceCoverage !== "string" ||
        typeof capabilities.gpuImplementationCoverage !== "string" ||
        typeof capabilities.cpuGpuParityCoverage !== "string" ||
        !isRecord(capabilities.storage) ||
        typeof capabilities.storage.coverage !== "string" ||
        typeof capabilities.storage.persistentFullVolumeFields !== "number" ||
        typeof capabilities.storage.peakScratchFullVolumeFields !== "number" ||
        typeof capabilities.storage.persistentFieldsPerIntegratedField !== "number" ||
        typeof capabilities.storage.peakScratchFieldsPerIntegratedField !== "number" ||
        typeof capabilities.storage.note !== "string"
      ) {
        throw new Error(`scheme '${rawModule.id}.${rawScheme.id}' has malformed capabilities`);
      }
      const parameterPaths = new Set<string>();
      const selectionRoot = rawModule.selectionPath.split(".").slice(0, -1).join(".");
      for (const rawParameter of rawScheme.parameters) {
        if (
          !isRecord(rawParameter) ||
          typeof rawParameter.path !== "string" ||
          typeof rawParameter.displayName !== "string" ||
          (rawParameter.kind !== "float" && rawParameter.kind !== "boolean") ||
          (rawParameter.kind === "float"
            ? typeof rawParameter.default !== "number"
            : typeof rawParameter.default !== "boolean") ||
          ![rawParameter.minimum, rawParameter.maximum, rawParameter.step].every(
            (bound) => bound === null || typeof bound === "number",
          ) ||
          typeof rawParameter.unit !== "string" ||
          typeof rawParameter.description !== "string"
        ) {
          throw new Error(`scheme '${rawModule.id}.${rawScheme.id}' has a malformed parameter`);
        }
        if (!rawParameter.path.startsWith(`${selectionRoot}.`)) {
          throw new Error(`parameter '${rawParameter.path}' is outside module '${rawModule.id}'`);
        }
        if (parameterPaths.has(rawParameter.path)) {
          throw new Error(`duplicate parameter '${rawParameter.path}'`);
        }
        parameterPaths.add(rawParameter.path);
      }
    }
    if (studioSelection.mode === "fixed_v1") {
      const fixedScheme = rawModule.schemes.find(
        (candidate) =>
          isRecord(candidate) && candidate.id === studioSelection.fixedSchemeId,
      );
      if (
        !isRecord(fixedScheme) ||
        fixedScheme.constructible !== true ||
        fixedScheme.runtimeSelectable !== true
      ) {
        throw new Error(
          `module '${rawModule.id}' fixes strict v1 to a missing or nonconstructible scheme`,
        );
      }
    }
  }

  const catalog = value as unknown as SchemeCatalog;
  const coverageIssues = catalogStudioCoverageIssues(catalog);
  if (coverageIssues.length > 0) {
    throw new Error(`simulator scheme catalog violates Studio selector coverage: ${coverageIssues.join("; ")}`);
  }
  return catalog;
}

export function schemeModule(
  catalog: SchemeCatalog,
  moduleId: string,
): RuntimeSchemeModule | undefined {
  return catalog.modules.find((module) => module.id === moduleId);
}

export function canonicalSchemeId(
  catalog: SchemeCatalog,
  moduleId: string,
  rawId: string,
): string | undefined {
  const normalized = rawId.trim().toLowerCase();
  return schemeModule(catalog, moduleId)?.schemes.find(
    (scheme) =>
      scheme.id === normalized || scheme.aliases.includes(normalized),
  )?.id;
}

function optionNote(scheme: RuntimeScheme): string {
  const status = [
    `maturity: ${scheme.maturity}`,
    `CPU reference: ${scheme.capabilities.cpuReferenceCoverage}`,
    scheme.capabilities.backends.length > 0
      ? `backends: ${scheme.capabilities.backends.join(", ")}`
      : "no declared backend",
  ].join(" · ");
  return scheme.caveat ? `${status}. ${scheme.caveat}` : status;
}

function parameterField(
  parameter: RuntimeSchemeParameter,
  sectionPath: string,
): FieldSchema {
  const key = parameter.path.slice(sectionPath.length + 1);
  if (parameter.kind === "boolean") {
    return {
      kind: "bool",
      key,
      label: parameter.displayName,
      description: parameter.description,
      default: parameter.default as boolean,
    };
  }
  return {
    kind: "float",
    key,
    label: parameter.displayName,
    description: parameter.description,
    default: parameter.default as number,
    min: parameter.minimum ?? undefined,
    max: parameter.maximum ?? undefined,
    step: parameter.step ?? undefined,
    unit: parameter.unit || undefined,
  };
}

const FORM_CATALOG_MODULES: ReadonlyArray<{
  moduleId: string;
  selectionPath: string;
}> = [
  { moduleId: "dynamics", selectionPath: "dynamics.scheme" },
  { moduleId: "time_stepping", selectionPath: "time.integrator" },
  { moduleId: "advection", selectionPath: "advection.scheme" },
  { moduleId: "microphysics", selectionPath: "microphysics.scheme" },
  { moduleId: "turbulence", selectionPath: "turbulence.scheme" },
  { moduleId: "diffusion", selectionPath: "diffusion.scheme" },
  { moduleId: "boundary_layer", selectionPath: "boundary_layer.scheme" },
  { moduleId: "radiation", selectionPath: "radiation.scheme" },
  { moduleId: "terrain", selectionPath: "terrain.scheme" },
  { moduleId: "chaos", selectionPath: "extensions.chaos.scheme" },
];

const PRODUCT_SET_CATALOG_MODULES: ReadonlyArray<{
  moduleId: string;
  selectionPath: string;
}> = [
  { moduleId: "radar", selectionPath: "radar.products" },
];

function configuredSelection(config: ConfigObject, path: string, fallback: string): string {
  const value = readPath(config, path);
  return typeof value === "string" && value ? value.trim().toLowerCase() : fallback;
}

function humanizeSchemeId(id: string): string {
  return id
    .split(/[_-]+/)
    .map((part) => {
      const upper = part.toUpperCase();
      return ["CPU", "GPU", "LES", "PBL", "RK3", "RK4", "TKE", "TVD", "WENO5", "YSU", "MYNN"].includes(upper)
        ? upper
        : part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

function optionLabel(scheme: RuntimeScheme, baseField: EnumField): string {
  if (scheme.displayName && scheme.displayName !== scheme.id) return scheme.displayName;
  return baseField.options.find((option) => option.value === scheme.id)?.label ??
    humanizeSchemeId(scheme.id);
}

function unavailableOption(
  value: string,
  note: string,
): EnumField["options"][number] {
  return {
    value,
    label: value ? `${humanizeSchemeId(value)} (not available)` : "Unavailable",
    disabled: true,
    note,
  };
}

/** Descriptor-backed selector paths represented by the current v1 form. */
export function formCatalogSelectorPaths(): string[] {
  return FORM_CATALOG_MODULES.map(({ selectionPath }) => selectionPath);
}

/** Form-exposed runtime descriptor paths missing a registered v1 control. */
export function catalogSelectorsOutsideForm(catalog: SchemeCatalog): RuntimeSchemeModule[] {
  const represented = new Set(formCatalogSelectorPaths());
  return catalog.modules.filter(
    (module) =>
      module.studioSelection.mode === "form" &&
      !represented.has(module.selectionPath),
  );
}

/**
 * Audit the bidirectional form/catalog contract. Every exact-binary module
 * must be explicitly form-selectable, fixed by the strict-v1 bridge, or
 * unavailable to v1 with evidence. A newly constructible descriptor family
 * cannot silently fall outside this test by merely choosing a selection path.
 */
export function catalogStudioCoverageIssues(catalog: SchemeCatalog): string[] {
  const issues: string[] = [];
  const formByModule = new Map(FORM_CATALOG_MODULES.map((entry) => [entry.moduleId, entry]));
  const productSetByModule = new Map(
    PRODUCT_SET_CATALOG_MODULES.map((entry) => [entry.moduleId, entry]),
  );
  for (const module of catalog.modules) {
    const form = formByModule.get(module.id);
    const productSet = productSetByModule.get(module.id);
    if (module.studioSelection.mode === "form") {
      if (!form || form.selectionPath !== module.selectionPath) {
        issues.push(`${module.id}: form policy has no matching Studio control`);
      }
    } else if (module.studioSelection.mode === "product_set_v1") {
      if (!productSet || productSet.selectionPath !== module.selectionPath) {
        issues.push(`${module.id}: product-set policy has no matching Studio control`);
      }
    } else if (form) {
      issues.push(`${module.id}: Studio control conflicts with ${module.studioSelection.mode} policy`);
    }
  }
  for (const form of FORM_CATALOG_MODULES) {
    const module = catalog.modules.find((candidate) => candidate.id === form.moduleId);
    if (!module) issues.push(`${form.moduleId}: Studio control has no exact-binary module`);
  }
  for (const productSet of PRODUCT_SET_CATALOG_MODULES) {
    const module = catalog.modules.find((candidate) => candidate.id === productSet.moduleId);
    if (!module) issues.push(`${productSet.moduleId}: Studio product set has no exact-binary module`);
  }
  return issues;
}

/**
 * Overlay the exact executable's descriptor-backed modules onto the form schema.
 * The base schema still owns structure/defaults; selectable identities and
 * availability come only from runtime descriptors.
 */
export function sectionsWithSchemeCatalog(
  catalog: SchemeCatalog | null,
  current: string | ConfigObject,
  unavailableReason = "Waiting for the configured simulator's scheme catalog.",
): SectionSchema[] {
  // Keep the string form as a compatibility bridge for callers that have not
  // yet moved from the original microphysics-only contract.
  const config: ConfigObject = typeof current === "string"
    ? { microphysics: { scheme: current } }
    : current;
  const expectedByPath = new Map(
    FORM_CATALOG_MODULES.map((entry) => [entry.selectionPath, entry]),
  );
  return SCHEMA_SECTIONS.map((section) => {
    const fields: FieldSchema[] = section.fields.map((field) => {
      if (field.kind !== "enum") return field;
      const selectionPath = qualifyKey(section, field);
      if (selectionPath === "output.format") {
        const selected = configuredSelection(config, selectionPath, field.default);
        if (!catalog) {
          return {
            ...field,
            default: selected,
            showDisabledReasons: true,
            disabledWhen: () => true,
            options: field.options.map((option) => ({
              ...option,
              disabled: true,
              note: unavailableReason,
            })),
          } satisfies EnumField;
        }
        const capabilities = new Map(
          catalog.runtimeCapabilities.outputFormats.map((capability) => [capability.id, capability]),
        );
        const options: EnumField["options"] = field.options.map((option) => {
          const capability = capabilities.get(option.value);
          if (!capability) {
            return {
              ...option,
              disabled: true,
              note: `The exact simulator does not publish '${option.value}' output capability.`,
            };
          }
          return {
            ...option,
            disabled: !capability.available,
            note: capability.available
              ? `Provided by ${capability.implementation} in the exact simulator binary.`
              : capability.reason,
          };
        });
        if (!options.some((option) => option.value === selected) && selected) {
          options.unshift(unavailableOption(
            selected,
            "This output format is not represented by the selected simulator binary.",
          ));
        }
        return { ...field, default: selected, options, showDisabledReasons: true } satisfies EnumField;
      }
      const expected = expectedByPath.get(selectionPath);
      if (!expected) return field;
      const selected = configuredSelection(config, selectionPath, field.default);
      const runtimeModule = catalog
        ? schemeModule(catalog, expected.moduleId)
        : undefined;
      if (!runtimeModule || runtimeModule.selectionPath !== selectionPath) {
        const note = catalog
          ? `The exact simulator does not publish the expected '${expected.moduleId}' selector at '${selectionPath}'.`
          : unavailableReason;
        return {
          ...field,
          default: selected,
          disabledWhen: () => true,
          options: [unavailableOption(selected, note)],
        } satisfies EnumField;
      }

      const canonicalCurrent = runtimeModule.schemes.find(
        (scheme) =>
          scheme.id === selected || scheme.aliases.includes(selected),
      )?.id;
      const options: EnumField["options"] = runtimeModule.schemes.map(
        (scheme) => ({
          value: scheme.id,
          label: optionLabel(scheme, field),
          disabled: !scheme.constructible || !scheme.runtimeSelectable,
          note: optionNote(scheme),
        }),
      );
      if (!canonicalCurrent && selected) {
        options.unshift(unavailableOption(
          selected,
          "This value is not exposed by the selected simulator binary.",
        ));
      }
      return { ...field, options } satisfies EnumField;
    });
    if (catalog && section.path) {
      for (const expected of FORM_CATALOG_MODULES) {
        if (!expected.selectionPath.startsWith(`${section.path}.`)) continue;
        const runtimeModule = schemeModule(catalog, expected.moduleId);
        if (!runtimeModule || runtimeModule.selectionPath !== expected.selectionPath) continue;
        const selected = configuredSelection(config, expected.selectionPath, "");
        const activeScheme = runtimeModule.schemes.find(
          (scheme) => scheme.id === selected || scheme.aliases.includes(selected),
        );
        if (activeScheme) {
          fields.push(...activeScheme.parameters.map((parameter) => parameterField(parameter, section.path!)));
        }
      }
    }
    if (catalog && section.id === "output") {
      const existing = new Set(fields.map((field) => field.key));
      const policy = catalog.runtimeCapabilities.interactionPolicy;
      for (const [field, budget] of Object.entries(policy.qualifiedZfpFieldBudgets)
        .sort(([left], [right]) => left.localeCompare(right))) {
        const key = `zfp_field_tolerance.${field}`;
        if (existing.has(key)) continue;
        fields.push({
          kind: "float",
          key,
          label: `${field} error budget`,
          description:
            `Maximum qualified ZFP absolute error for ${field}. The exact simulator ` +
            "publishes and enforces this field-specific ceiling.",
          default: budget,
          min: Number.MIN_VALUE,
          step: budget / 10,
          unit: policy.qualifiedZfpFieldUnits[field],
          visibleWhen: (state: ConfigObject) => {
            const preset = String(readPath(state, "preset"));
            const radarProducts = readPath(config, "radar.products");
            const selectedByRadar = Array.isArray(radarProducts) &&
              radarProducts.includes(field);
            return String(readPath(state, "format")).startsWith("zfp") &&
              readPath(state, "zfp_per_field_tolerances") === true &&
              ((policy.qualifiedZfpPresetFields[preset]?.includes(field) ?? false) ||
                selectedByRadar);
          },
        });
      }
    }
    let disable = section.disable;
    if (disable && catalog) {
      const expected = expectedByPath.get(disable.path);
      const runtimeModule = expected ? schemeModule(catalog, expected.moduleId) : undefined;
      if (runtimeModule?.selectionPath === disable.path) {
        const configuredOn = typeof disable.onValue === "string"
          ? canonicalSchemeId(catalog, expected!.moduleId, disable.onValue)
          : undefined;
        const nextOn = runtimeModule.schemes.find(
          (scheme) => scheme.id === configuredOn && scheme.constructible && scheme.runtimeSelectable && scheme.id !== disable!.offValue,
        ) ?? runtimeModule.schemes.find(
          (scheme) => scheme.constructible && scheme.runtimeSelectable && scheme.id !== disable!.offValue,
        );
        if (nextOn) disable = { ...disable, onValue: nextOn.id };
      }
    }
    return {
      ...section,
      fields,
      disable,
    };
  });
}

function deleteCatalogPath(obj: ConfigObject, path: string): ConfigObject {
  const root = structuredClone(obj);
  const parts = path.split(".");
  let cursor: ConfigObject = root;
  const parents: Array<{ parent: ConfigObject; key: string }> = [];
  for (let index = 0; index < parts.length - 1; index++) {
    const next = cursor[parts[index]];
    if (!isRecord(next)) return root;
    parents.push({ parent: cursor, key: parts[index] });
    cursor = next as ConfigObject;
  }
  delete cursor[parts.at(-1)!];
  for (let index = parents.length - 1; index >= 0; index--) {
    const { parent, key } = parents[index];
    const child = parent[key];
    if (isRecord(child) && Object.keys(child).length === 0) delete parent[key];
    else break;
  }
  return root;
}

function normalizeCatalogModule(
  catalog: SchemeCatalog,
  config: ConfigObject,
  moduleId: string,
): ConfigObject {
  const module = schemeModule(catalog, moduleId);
  if (!module) return config;
  const raw = readPath(config, module.selectionPath);
  if (typeof raw !== "string") return config;
  const normalized = raw.trim().toLowerCase();
  const scheme = module.schemes.find(
    (candidate) => candidate.id === normalized || candidate.aliases.includes(normalized),
  );
  if (!scheme) return config;

  let next = writePath(config, module.selectionPath, scheme.id);
  const activePaths = new Set(scheme.parameters.map((parameter) => parameter.path));
  const declaredPaths = new Set(
    module.schemes.flatMap((candidate) => candidate.parameters.map((parameter) => parameter.path)),
  );
  for (const path of declaredPaths) {
    if (!activePaths.has(path)) next = deleteCatalogPath(next, path);
  }
  for (const parameter of scheme.parameters) {
    const existing = readPath(config, parameter.path);
    const compatible =
      (parameter.kind === "float" && typeof existing === "number") ||
      (parameter.kind === "boolean" && typeof existing === "boolean");
    next = writePath(next, parameter.path, compatible ? existing! : parameter.default);
  }
  return next;
}

/** Canonicalize aliases and reconcile descriptor-declared parameters for all form modules. */
export function normalizeSchemeSelections(
  catalog: SchemeCatalog,
  config: ConfigObject,
): ConfigObject {
  return FORM_CATALOG_MODULES.reduce(
    (next, { moduleId }) => normalizeCatalogModule(catalog, next, moduleId),
    config,
  );
}

/** Canonicalize the selection, retain active values, and prune inactive blocks. */
export function normalizeMicrophysicsParameters(
  catalog: SchemeCatalog,
  config: ConfigObject,
): ConfigObject {
  return normalizeCatalogModule(catalog, config, "microphysics");
}

// Browser-only design fixture. Desktop builds always query the selected binary.
export const MOCK_SCHEME_CATALOG: SchemeCatalog = validateSchemeCatalog(nativeCatalogSnapshot);
