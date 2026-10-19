import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import {
  authoritativeRuntimeStateFields,
  buildConfigPreflightRequest,
  validateConfigPreflightEvidence,
  validateConfigPreflightReport,
} from "../../../../ui/src/lib/config/configPreflight.ts";
import type { SchemeCatalog } from "../../../../ui/src/lib/config/schemeCatalog.ts";
import { fnv1a64Text } from "../../../../ui/src/lib/cm1/cm1GuidedHandoff.ts";

const catalog = {
  schema: "nimbus.scheme-catalog.v9",
  catalogVersion: 9,
  runtimeStateFields: ["w", "rho", "u", "v"],
  readiness: {
    version: 1,
    items: [{
      id: "catalog.runtime_cli_studio", area: "capability_truth",
      scope: "Exact-binary agreement", state: "ready",
      boundedEvidence: "tests", campaignEvidence: "",
      externalReviewEvidence: "", blocker: "",
    }],
  },
  runtimeCapabilities: {
    outputFormats: [
      { id: "npy_3d", available: true, reason: "", implementation: "builtin_npy" },
      { id: "netcdf", available: false, reason: "no libnetcdf", implementation: "" },
    ],
    interactionPolicy: {
      version: 9,
      generatedMetalCanonicalScope: {
        version: 2,
        backend: "metal",
        executionMode: "qualified",
        tupleCount: 4032,
        distinctConfigCount: 3024,
        outputFormats: ["npy_3d", "netcdf"],
        scalarTransportOwner: "native_conservative_mc",
        baseManifestSha256: "85642f094c418fe95eb4c2fa0ba5a2a947aec95f8f1c43b17abcd924b0f068a0",
        manifestSha256: "30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99",
      },
      inactiveSchemeId: "none",
      horizontalOnlyTurbulenceModes: ["2d", "horizontal_only"],
      knownSurfaceSchemes: ["none", "slab"],
      activeSurfaceSchemeId: "slab",
      terrainRadiationSchemes: ["none", "simple_grey", "rrtmg"],
      terrainLongwaveRadiationSchemeId: "rrtmg",
      terrainShortwaveAllowed: true,
      terrainBoundaryLayerSchemes: ["none"],
      terrainTurbulenceSchemes: ["none", "smagorinsky"],
      terrainChaosSchemes: ["none", "initial_conditions", "boundary_layer", "full_stochastic"],
      chaosCanonicalTargets: {
        none: { initialConditions: [], tendencies: [], nativeGenerationRequired: false },
        initial_conditions: {
          initialConditions: ["u", "v", "w", "theta"], tendencies: [],
          nativeGenerationRequired: false,
        },
        boundary_layer: {
          initialConditions: [], tendencies: ["pbl"], nativeGenerationRequired: false,
        },
        full_stochastic: {
          initialConditions: ["u", "v", "w", "theta"],
          tendencies: ["microphysics", "pbl", "diffusion"],
          nativeGenerationRequired: false,
        },
      },
      terrainActiveDiffusionSchemes: ["explicit"],
      terrainActiveDiffusionTargets: ["scalars"],
      qualifiedZfpTerrainSchemes: ["none", "bell", "schar"],
      qualifiedZfpPresetIds: ["minimal"],
      qualifiedZfpPresetFields: {
        minimal: ["p", "qr", "qv", "rho", "theta", "u", "v", "w"],
      },
      qualifiedZfpMode: "accuracy",
      qualifiedZfpRequiresPerFieldTolerances: true,
      qualifiedZfpMinimumKeepLastFrames: 1,
      qualifiedZfpMinimumFreeSpaceMib: 1,
      qualifiedZfpRequiresRetainedFileCeiling: true,
      qualifiedExecutionRequiresMaturityAcknowledgment: false,
      qualifiedZfpMaxSparseThreshold: 0,
      qualifiedZfpAllowsFloat16Prefilter: false,
      qualifiedZfpFieldBudgets: {
        p: 5e-2, qr: 1e-8, qv: 1e-8, rho: 1e-6,
        theta: 1e-3, u: 1e-4, v: 1e-4, w: 1e-4,
      },
      qualifiedZfpFieldUnits: {
        p: "Pa", qr: "kg/kg", qv: "kg/kg", rho: "kg/m^3",
        theta: "K", u: "m/s", v: "m/s", w: "m/s",
      },
      lateralBoundaryModes: [
        "periodic", "radiative", "zero_gradient", "open", "rigid", "rigid_slip",
      ],
      radiativeLateralStaggerings: ["c_grid"],
      legacyOpenCgridResolution: "radiative",
      legacyOpenCollocatedResolution: "zero_gradient",
      periodicLateralPairsRequired: true,
      bottomBoundaryModes: ["rigid_noslip", "rigid_slip"],
      topBoundaryModes: ["rigid", "rigid_with_sponge", "open"],
      scientificDiagnosticsOmittedSentinel: -1,
      scientificDiagnosticsMinimumCadence: 0,
      gpuApplyControlsNoOpOnCpu: true,
      gpuApplyMicrophysicsSchemes: ["kessler", "lin"],
      gpuApplySgsSchemes: ["smagorinsky"],
      gpuApplySlowDynamicsSchemes: ["klemp_wilhelmson_split_cartesian_cgrid"],
      qualifiedGpuApplySlowDynamicsSchemes: [
        "klemp_wilhelmson_split_cartesian_agrid",
        "klemp_wilhelmson_split_cartesian_cgrid",
      ],
      gpuApplyMicrophysicsDispatchGuardModes: ["off"],
      gpuApplyMicrophysicsHostCommitGuardModes: ["warn", "error"],
      gpuApplySlowDispatchGuardModes: ["off"],
      gpuApplySlowHostCommitGuardModes: ["warn", "error"],
      gpuApplySgsDispatchGuardModes: ["off", "warn", "error"],
    },
  },
  modules: [
    {
      id: "dynamics",
      selectionPath: "dynamics.scheme",
      studioSelection: { mode: "form", fixedSchemeId: "", reason: "" },
      schemes: [],
    },
    {
      id: "microphysics",
      selectionPath: "microphysics.scheme",
      studioSelection: { mode: "form", fixedSchemeId: "", reason: "" },
      schemes: [],
    },
  ],
} as SchemeCatalog;

assert.deepEqual(authoritativeRuntimeStateFields(catalog), ["rho", "u", "v", "w"]);
assert.throws(
  () => authoritativeRuntimeStateFields(
    { ...catalog, runtimeStateFields: undefined } as unknown as SchemeCatalog,
  ),
  /does not publish an authoritative/,
);
assert.throws(
  () => authoritativeRuntimeStateFields({ ...catalog, runtimeStateFields: ["u", "u"] } as SchemeCatalog),
  /duplicate/,
);

const requestConfig = {
    grid: {
      coordinate_system: "cartesian", staggering: "c_grid",
      nx: 18, ny: 18, nz: 18, dx: 500, dy: 500, dz: 250, ztop: 4500,
    },
    time: { dt_large: 0.1, acoustic_substeps: 6, output_interval: 1 },
    boundary_conditions: {
      lateral: { west: "open", east: "open", south: "rigid_slip", north: "rigid_slip" },
      bottom: "rigid_noslip",
      top: "rigid_with_sponge",
    },
    dynamics: { scheme: "cgrid" },
    microphysics: { scheme: "kessler" },
    radiation: { do_sw: true },
    turbulence: { mode: "3d", dynamic_coefficient: false },
    diffusion: { K_h: 0, K_v: 0, apply_to: "scalars" },
    surface: { scheme: "slab", apply_fluxes: true },
    extensions: {
      compute: { backend: "gpu" },
      execution: { mode: "exploratory", acknowledge_maturity_limits: false },
      diagnostics: { scientific_every_steps: 5, deterministic_reductions: true },
    },
    output: { format: "npy_3d" },
    initial_conditions: {
      sounding: { type: "parametric_cape" },
      hodograph: { type: "wk_param" },
      trigger: { type: "bubble" },
    },
};
const request = buildConfigPreflightRequest(requestConfig, catalog, ["w", "u", "w", " v "]);
const formWithOutputDefaults = {
  ...requestConfig,
  output: { ...requestConfig.output, zfp_per_field_tolerances: true, zfp_field_tolerance: { theta: 0.001 }, async_io: true },
};
const originalNpyRequest = buildConfigPreflightRequest(formWithOutputDefaults, catalog, ["u", "v", "w"], {
  yamlText: "schema: nimbus.v1\noutput:\n  format: npy_3d\n", yamlPath: "/tmp/original-npy.yaml",
});
assert.equal(originalNpyRequest.output.zfpPerFieldTolerances, false);
assert.deepEqual(originalNpyRequest.output.zfpFieldTolerances, {});
assert.equal(originalNpyRequest.output.asyncIo, false);
const explicitOutputRequest = buildConfigPreflightRequest(formWithOutputDefaults, catalog, ["u", "v", "w"], {
  yamlText: "schema: nimbus.v1\noutput:\n  format: npy_3d\n  async_io: false\n  zfp_per_field_tolerances: true\n  zfp_field_tolerance:\n    theta: 0.002\n",
  yamlPath: "/tmp/explicit-output.yaml",
});
assert.equal(explicitOutputRequest.output.zfpPerFieldTolerances, true);
assert.deepEqual(explicitOutputRequest.output.zfpFieldTolerances, { theta: 0.002 });
assert.equal(explicitOutputRequest.output.asyncIo, false);

assert.deepEqual(request.availableStateFields, ["u", "v", "w"]);
assert.equal(request.requestVersion, 10);
assert.equal(request.interactions.turbulenceDynamicCoefficient, false);
assert.deepEqual(request.source, { kind: "descriptor_only" });
assert.equal(request.outputFormat, "npy_3d");
assert.deepEqual(request.gridMetrics, {
  nx: 18, ny: 18, nz: 18, dx: 500, dy: 500, dz: 250, ztop: 4500,
});
assert.deepEqual(request.timeControl, {
  dtLarge: 0.1, acousticSubsteps: 6, outputInterval: 1,
});
assert.deepEqual(request.output, {
  preset: "minimal",
  asyncIo: false,
  zfpMode: "accuracy",
  zfpPerFieldTolerances: false,
  zfpSparseThreshold: 0,
  zfpFloat16Prefilter: false,
  zfpFieldExecution: "serial",
  zfpFieldWorkers: 0,
  zfpFieldScratchMib: 64,
  zfpFieldTolerances: {},
  keepLastFrames: 0,
  minimumFreeSpaceMib: 0,
  maximumRetainedHistoryFiles: 0,
});
assert.equal(request.executionMode, "exploratory");
assert.equal(request.maturityLimitsAcknowledged, false);
assert.equal(request.allowBackendFallback, true);
assert.equal(request.validationGuardMode, "warn");
assert.equal(request.gpuApplyMicrophysics, true);
assert.equal(request.gpuApplySlow, false);
assert.equal(request.gpuApplySgs, false);
assert.deepEqual(request.boundary, {
  west: "open",
  east: "open",
  south: "rigid_slip",
  north: "rigid_slip",
  bottom: "rigid_noslip",
  top: "rigid_with_sponge",
});
assert.deepEqual(request.diagnostics, {
  scientificEverySteps: 5,
  deterministicReductions: true,
});
assert.deepEqual(request.interactions, {
  radiationShortwaveEnabled: true,
  turbulenceMode: "3d",
  turbulenceDynamicCoefficient: false,
  surfaceScheme: "slab",
  surfaceApplyFluxes: true,
  diffusionTarget: "scalars",
  diffusionActive: false,
});
assert.deepEqual(request.selections, [
  { moduleId: "dynamics", schemeId: "cgrid" },
  { moduleId: "microphysics", schemeId: "kessler" },
]);

const radarCatalog = structuredClone(catalog);
radarCatalog.modules.push({
  id: "radar",
  selectionPath: "radar.products",
  studioSelection: {
    mode: "product_set_v1",
    fixedSchemeId: "",
    reason: "Strict v1 ordered radar product set.",
  },
  schemes: [
    {
      id: "reflectivity",
      aliases: ["reflectivity_dbz"],
    },
    {
      id: "velocity",
      aliases: ["radial_velocity"],
    },
    {
      id: "zdr",
      aliases: ["kdp", "rhohv"],
    },
  ],
} as SchemeCatalog["modules"][number]);
const radarRequest = buildConfigPreflightRequest(
  {
    grid: {
      coordinate_system: "cartesian", staggering: "c_grid",
      nx: 18, ny: 18, nz: 18, dx: 500, dy: 500, dz: 250, ztop: 4500,
    },
    time: { dt_large: 0.1, acoustic_substeps: 6, output_interval: 1 },
    boundary_conditions: {
      lateral: { west: "open", east: "open", south: "open", north: "open" },
      bottom: "rigid_noslip",
      top: "rigid_with_sponge",
    },
    dynamics: { scheme: "cgrid" },
    microphysics: { scheme: "kessler" },
    radar: { enabled: true, products: ["reflectivity_dbz", "radial_velocity", "kdp"] },
    radiation: { do_sw: true },
    turbulence: { mode: "3d", dynamic_coefficient: false },
    diffusion: { K_h: 0, K_v: 0, apply_to: "scalars" },
    surface: { scheme: "slab", apply_fluxes: true },
    extensions: { compute: { backend: "cpu" }, execution: { mode: "exploratory" } },
    output: { format: "npy_3d" },
    initial_conditions: {
      sounding: { type: "parametric_cape" },
      hodograph: { type: "wk_param" },
      trigger: { type: "bubble" },
    },
  },
  radarCatalog,
  ["rho", "u", "v", "w"],
);
assert.deepEqual(
  radarRequest.selections.filter((selection) => selection.moduleId === "radar"),
  [
    { moduleId: "radar", schemeId: "reflectivity_dbz" },
    { moduleId: "radar", schemeId: "radial_velocity" },
    { moduleId: "radar", schemeId: "kdp" },
  ],
);

const report = validateConfigPreflightReport({
  schema: "nimbus.config-preflight-report.v10",
  reportVersion: 10,
  compatible: true,
  nativeAdmission: {
    state: "not_checked",
    descriptorProjection: "not_checked",
    sourceSha256: null,
    sourceFingerprint: null,
    sourcePathFingerprint: null,
    validatedBinarySha256: null,
    validationExitCode: null,
  },
  executable: {
    canonicalPath: "/bin/nimbus_sim",
    byteLength: 10,
    modifiedUnixMs: 1,
    catalogFingerprint: "fnv1a64:1",
    binaryFingerprint: `sha256:${"2".repeat(64)}`,
  },
  context: {
    executionMode: "exploratory",
    maturityLimitsAcknowledged: false,
    backend: "gpu",
    allowBackendFallback: true,
    validationGuardMode: "warn",
    gpuApplyMicrophysics: true,
    gpuApplySlow: false,
    gpuApplySgs: false,
    outputFormat: "npy_3d",
    gridMetrics: request.gridMetrics,
    timeControl: request.timeControl,
    output: request.output,
    coordinateSystem: "cartesian",
    staggering: "c_grid",
    boundary: request.boundary,
    diagnostics: request.diagnostics,
    initialization: {
      soundingSource: "parametric_cape",
      hodographSource: "wk_param",
      triggerSource: "warm_bubble",
    },
    interactions: request.interactions,
    availableStateFields: ["u", "v", "w"],
  },
  resolvedSelections: [
    {
      moduleId: "dynamics",
      requestedSchemeId: "cgrid",
      canonicalSchemeId: "cgrid",
      maturity: "qualified",
      constructible: true,
      runtimeSelectable: true,
    },
    {
      moduleId: "microphysics",
      requestedSchemeId: "kessler",
      canonicalSchemeId: "kessler",
      maturity: "qualified",
      constructible: true,
      runtimeSelectable: true,
    },
  ],
  hardErrors: [],
  warnings: [],
});
assert.throws(
  () => validateConfigPreflightReport({
    ...report,
    schema: "nimbus.config-preflight-report.v9",
    reportVersion: 9,
  }),
  /malformed configuration preflight report/,
);
assert.throws(
  () => validateConfigPreflightReport({
    ...report,
    context: {
      ...report.context,
      interactions: {
        ...report.context.interactions,
        turbulenceDynamicCoefficient: undefined,
      },
    },
  }),
  /malformed configuration preflight report/,
);
assert.throws(
  () => validateConfigPreflightReport({
    ...report,
    context: { ...report.context, output: { ...report.context.output, asyncIo: undefined } },
  }),
  /malformed configuration preflight report/,
);
assert.equal(report.compatible, true);
assert.equal(validateConfigPreflightReport({ ...report, executionTargetFingerprint: `sha256:${"a".repeat(64)}` }).executionTargetFingerprint,
  `sha256:${"a".repeat(64)}`);
assert.throws(() => validateConfigPreflightReport({ ...report, executionTargetFingerprint: "unbound" }),
  /malformed configuration preflight report/);
assert.equal(validateConfigPreflightEvidence(report, request), report);
function reverseObjectKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(reverseObjectKeys);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).reverse().map(([key, child]) => [key, reverseObjectKeys(child)]));
  }
  return value;
}
const reorderedRequest = reverseObjectKeys(request) as typeof request;
const reorderedReport = validateConfigPreflightReport(reverseObjectKeys(report));
assert.equal(validateConfigPreflightEvidence(report, reorderedRequest), report);
assert.equal(validateConfigPreflightEvidence(reorderedReport, request), reorderedReport);
assert.throws(() => validateConfigPreflightEvidence(reorderedReport, {
  ...reorderedRequest, boundary: { ...reorderedRequest.boundary, north: "changed" },
}), /boundaries/);
assert.throws(() => validateConfigPreflightEvidence(reorderedReport, {
  ...reorderedRequest, output: { ...reorderedRequest.output, zfpFieldTolerances: { unrequested: 0.125 } },
}), /output science/);

const exactSource = { kind: "exact_document" as const,
  yamlText: "schema: nimbus.v1\n", yamlPath: "/tmp/preflight.yaml" };
const exactRequest = { ...request, source: exactSource };
const exactReport = validateConfigPreflightReport({
  ...report,
  nativeAdmission: {
    state: "accepted",
    descriptorProjection: "not_checked",
    sourceSha256: createHash("sha256").update(exactSource.yamlText).digest("hex"),
    sourceFingerprint: fnv1a64Text(exactSource.yamlText),
    sourcePathFingerprint: fnv1a64Text(exactSource.yamlPath),
    validatedBinarySha256: report.executable.binaryFingerprint,
    validationExitCode: 0,
  },
});
assert.equal(validateConfigPreflightEvidence(exactReport, exactRequest), exactReport);
assert.throws(
  () => validateConfigPreflightReport({
    ...exactReport,
    nativeAdmission: { ...exactReport.nativeAdmission, descriptorProjection: "checked" },
  }),
  /malformed configuration preflight report/,
);
assert.throws(
  () => validateConfigPreflightReport({
    ...exactReport,
    nativeAdmission: { ...exactReport.nativeAdmission, state: "rejected" },
  }),
  /claims compatibility after native rejection/,
);
assert.throws(
  () => validateConfigPreflightEvidence(exactReport, {
    ...exactRequest,
    source: { ...exactSource, yamlText: "schema: nimbus.v1\n# changed\n" },
  }),
  /not bound to the exact source/,
);
assert.throws(
  () => validateConfigPreflightEvidence(exactReport, {
    ...exactRequest,
    source: { ...exactSource, yamlPath: "/tmp/other.yaml" },
  }),
  /not bound to the exact source/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...exactReport,
    nativeAdmission: { ...exactReport.nativeAdmission,
      validatedBinarySha256: `sha256:${"3".repeat(64)}` },
  }, exactRequest),
  /not bound to the exact source/,
);
assert.throws(
  () => validateConfigPreflightEvidence(report, exactRequest),
  /not bound to the exact source/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      gridMetrics: { ...report.context.gridMetrics, dx: 501 },
    },
  }, request),
  /does not match the requested grid\/time envelope/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      timeControl: { ...report.context.timeControl, outputInterval: 2 },
    },
  }, request),
  /does not match the requested grid\/time envelope/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: { ...report.context, outputFormat: "netcdf" },
  }, request),
  /does not match the requested output format/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      output: {
        ...report.context.output,
        maximumRetainedHistoryFiles: 1,
      },
    },
  }, request),
  /does not match the requested output science/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: { ...report.context, availableStateFields: ["u"] },
  }, request),
  /does not match the requested state inventory/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: { ...report.context, maturityLimitsAcknowledged: true },
  }, request),
  /does not match the requested execution policy/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      boundary: { ...report.context.boundary, west: "zero_gradient" },
    },
  }, request),
  /does not match the requested boundaries/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      diagnostics: { ...report.context.diagnostics, scientificEverySteps: 10 },
    },
  }, request),
  /does not match the requested diagnostics/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: { ...report.context, staggering: "collocated" },
  }, request),
  /does not match the requested execution geometry/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: { ...report.context, backend: "cpu" },
  }, request),
  /does not match the requested execution geometry/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      initialization: {
        ...report.context.initialization,
        triggerSource: "none",
      },
    },
  }, request),
  /does not match the requested initialization/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      interactions: {
        ...report.context.interactions,
        radiationShortwaveEnabled: false,
      },
    },
  }, request),
  /does not match the requested science interactions/,
);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    context: {
      ...report.context,
      interactions: {
        ...report.context.interactions,
        turbulenceDynamicCoefficient: true,
      },
    },
  }, request),
  /does not match the requested science interactions/,
);

const qualifiedConfig = {
    grid: {
      coordinate_system: "cartesian", staggering: "c_grid",
      nx: 18, ny: 18, nz: 18, dx: 500, dy: 500, dz: 250, ztop: 4500,
    },
    time: { dt_large: 0.1, acoustic_substeps: 6, output_interval: 1 },
    boundary_conditions: {
      lateral: { west: "radiative", east: "radiative", south: "radiative", north: "radiative" },
      bottom: "rigid_slip",
      top: "open",
    },
    dynamics: { scheme: "cgrid" },
    microphysics: { scheme: "kessler" },
    radiation: { do_sw: false },
    turbulence: { mode: "3d", dynamic_coefficient: false },
    diffusion: { K_h: 5, K_v: 0, apply_to: "scalars" },
    surface: { scheme: "slab", apply_fluxes: true },
    extensions: {
      compute: { backend: "gpu" },
      execution: { mode: "qualified" },
    },
    output: {
      preset: "minimal",
      format: "zfp_high",
      async_io: true,
      zfp_mode: "accuracy",
      zfp_per_field_tolerances: true,
      zfp_sparse_threshold: 0,
      zfp_float16_prefilter: false,
      keep_last_frames: 3,
      minimum_free_space_mib: 4096,
      maximum_retained_history_files: 256,
      zfp_field_tolerance: {
        p: 5e-2, qr: 1e-8, qv: 1e-8, rho: 1e-6,
        reflectivity_dbz: 1e-2,
        theta: 1e-3, u: 1e-4, v: 1e-4, w: 1e-4,
      },
    },
    initial_conditions: {
      sounding: { type: "parametric_cape" },
      hodograph: { type: "wk_param" },
      trigger: { type: "bubble" },
    },
};
const qualifiedRequest = buildConfigPreflightRequest(
  qualifiedConfig,
  catalog,
  ["u", "v", "w"],
);
assert.equal(qualifiedRequest.executionMode, "qualified");
assert.equal(qualifiedRequest.maturityLimitsAcknowledged, true);
assert.equal(qualifiedRequest.allowBackendFallback, false);
assert.equal(qualifiedRequest.interactions.radiationShortwaveEnabled, false);
assert.equal(qualifiedRequest.interactions.diffusionActive, true);
assert.equal(qualifiedRequest.interactions.turbulenceDynamicCoefficient, false);
const dynamicRequest = buildConfigPreflightRequest({
  ...qualifiedConfig,
  turbulence: { ...qualifiedConfig.turbulence, dynamic_coefficient: true },
}, catalog, ["u", "v", "w"]);
assert.equal(dynamicRequest.interactions.turbulenceDynamicCoefficient, true);
const staticVertical = buildConfigPreflightRequest({
  ...qualifiedConfig,
  turbulence: { mode: "vertical_only", dynamic_coefficient: false },
}, catalog, ["u", "v", "w"]);
const dynamicVertical = buildConfigPreflightRequest({
  ...qualifiedConfig,
  turbulence: { mode: "vertical_only", dynamic_coefficient: true },
}, catalog, ["u", "v", "w"]);
assert.equal(staticVertical.interactions.turbulenceMode, "vertical_only");
assert.equal(staticVertical.interactions.turbulenceDynamicCoefficient, false);
assert.equal(dynamicVertical.interactions.turbulenceDynamicCoefficient, true);
const catalogWithNativeDynamicDefault = structuredClone(catalog);
catalogWithNativeDynamicDefault.runtimeCapabilities.interactionPolicy.version = 12;
catalogWithNativeDynamicDefault.runtimeCapabilities.interactionPolicy
  .turbulenceDynamicCoefficientDefault = false;
const omittedDynamicRequest = buildConfigPreflightRequest({
  ...qualifiedConfig,
  turbulence: { mode: "3d" },
}, catalogWithNativeDynamicDefault, ["u", "v", "w"]);
assert.equal(omittedDynamicRequest.interactions.turbulenceDynamicCoefficient, false);
assert.throws(
  () => buildConfigPreflightRequest({
    ...qualifiedConfig,
    turbulence: { mode: "3d" },
  }, catalog, ["u", "v", "w"]),
  /no exact-binary dynamic-coefficient default/,
);
assert.throws(
  () => buildConfigPreflightRequest({
    ...qualifiedConfig,
    turbulence: { ...qualifiedConfig.turbulence, dynamic_coefficient: "yes" },
  }, catalog, ["u", "v", "w"]),
  /requires boolean.*turbulence.dynamic_coefficient/,
);
assert.equal(qualifiedRequest.outputFormat, "zfp_high");
assert.equal(qualifiedRequest.output.asyncIo, true);
const exactQualifiedRequest = buildConfigPreflightRequest(
  qualifiedConfig, catalog, ["u", "v", "w"],
  { yamlText: "schema: nimbus.v1\n", yamlPath: "/tmp/qualified.yaml" },
);
assert.deepEqual(exactQualifiedRequest.source, {
  kind: "exact_document", yamlText: "schema: nimbus.v1\n", yamlPath: "/tmp/qualified.yaml",
});
assert.throws(
  () => buildConfigPreflightRequest(qualifiedConfig, catalog, ["u"],
    { yamlText: "", yamlPath: "/tmp/qualified.yaml" }),
  /requires bounded YAML text and a source path/,
);
assert.throws(
  () => buildConfigPreflightRequest({
    ...qualifiedConfig,
    output: { ...qualifiedConfig.output, async_io: "false" },
  }, catalog, ["u", "v", "w"]),
  /requires boolean 'output.async_io'/,
);
assert.equal(qualifiedRequest.output.zfpPerFieldTolerances, true);
assert.equal(qualifiedRequest.output.zfpFieldTolerances.rho, 1e-6);
assert.equal(qualifiedRequest.output.zfpFieldTolerances.reflectivity_dbz, 1e-2);
assert.equal(qualifiedRequest.output.keepLastFrames, 3);
assert.equal(qualifiedRequest.output.minimumFreeSpaceMib, 4096);
assert.equal(qualifiedRequest.output.maximumRetainedHistoryFiles, 256);
const { theta: omittedTheta, ...withoutTheta } = qualifiedConfig.output.zfp_field_tolerance;
void omittedTheta;
const missingThetaConfig = {
  ...qualifiedConfig,
  output: { ...qualifiedConfig.output, zfp_field_tolerance: withoutTheta },
};
const missingThetaRequest = buildConfigPreflightRequest(
  missingThetaConfig, catalog, ["u", "v", "w"],
);
assert.equal("theta" in missingThetaRequest.output.zfpFieldTolerances, false);
assert.throws(
  () => validateConfigPreflightEvidence({
    ...report,
    resolvedSelections: [{
      moduleId: "dynamics",
      requestedSchemeId: "cgrid",
      canonicalSchemeId: "cgrid",
      maturity: "qualified",
      constructible: true,
      runtimeSelectable: true,
    }],
  }, request),
  /does not cover every requested selection/,
);

assert.throws(
  () => validateConfigPreflightReport({ ...report, compatible: true, hardErrors: [{
    code: "bad", moduleId: null, schemeId: null, message: "bad", expected: [], actual: null,
  }] }),
  /contradicts/,
);
assert.throws(() => buildConfigPreflightRequest({}, catalog, []), /no selected runtime schemes/);

for (const [key, value] of [
  ["damping.lateral_sponge_enabled", "false"], ["damping.lateral_width", "0"],
  ["damping.lateral_alpha", "0"], ["damping.w_only_top_sponge", "false"],
  ["damping.w_only_depth_cells", "0"], ["damping.w_only_alpha_max", "0"],
  ["extensions.solver.slow_rk2", "false"], ["extensions.solver.lid_bc_gamma", "0.5"],
  ["extensions.solver.axis_u_m1_filter", "true"],
] as const) {
  const parts = key.split(".");
  const originalText = key.startsWith("damping.")
    ? `schema: nimbus.v1\ndamping:\n  ${parts[1]}: ${value}\n`
    : `schema: nimbus.v1\nextensions:\n  solver:\n    ${parts[2]}: ${value}\n`;
  const originalRequest = buildConfigPreflightRequest(requestConfig, catalog, ["w", "u", "w", " v "], {
    yamlText: originalText, yamlPath: "/tmp/mixed-control-original.yaml",
  });
  assert.deepEqual(originalRequest.source, {
    kind: "exact_document", yamlText: originalText, yamlPath: "/tmp/mixed-control-original.yaml",
  });
  const rejected = validateConfigPreflightReport({
    ...report, compatible: false,
    hardErrors: [{ code: "native_strict_admission_rejected", moduleId: null, schemeId: null,
      message: `native control rejection ${key}`, expected: ["exact native admission"], actual: "exit 2" }],
    nativeAdmission: {
      state: "rejected", descriptorProjection: "not_checked",
      sourceSha256: createHash("sha256").update(originalText).digest("hex"),
      sourceFingerprint: fnv1a64Text(originalText),
      sourcePathFingerprint: fnv1a64Text("/tmp/mixed-control-original.yaml"),
      validatedBinarySha256: report.executable.binaryFingerprint, validationExitCode: 2,
    },
  });
  assert.equal(rejected.compatible, false);
  assert.equal(validateConfigPreflightEvidence(rejected, originalRequest), rejected);
}

console.log("config preflight tests passed");
