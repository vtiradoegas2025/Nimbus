import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";

import {
  applyFieldChange,
  defaultsForSchema,
  readPath,
  writePath,
  type EnumField,
  type SectionSchema,
} from "../../../../ui/src/lib/config/configSchema.ts";
import {
  canonicalSchemeId,
  catalogSelectorsOutsideForm,
  catalogStudioCoverageIssues,
  formCatalogSelectorPaths,
  MOCK_SCHEME_CATALOG,
  QUALIFIED_FLAT_MIXED_METAL_NPY_SCOPE,
  normalizeMicrophysicsParameters,
  normalizeSchemeSelections,
  sectionsWithSchemeCatalog,
  validateSchemeCatalog,
} from "../../../../ui/src/lib/config/schemeCatalog.ts";
import { parseYaml, stringifyYaml } from "../../../../ui/src/lib/config/yaml.ts";

let passed = 0;
const failures: string[] = [];

function check(condition: unknown, message: string): void {
  if (condition) passed++;
  else failures.push(message);
}

function microphysicsField(sections: SectionSchema[]): EnumField {
  const field = sections
    .find((section) => section.id === "microphysics")
    ?.fields.find((candidate) => candidate.kind === "enum" && candidate.key === "scheme");
  if (!field || field.kind !== "enum") throw new Error("microphysics enum missing");
  return field;
}

function selectorField(sections: SectionSchema[], selectionPath: string): EnumField {
  for (const section of sections) {
    const field = section.fields.find((candidate) => {
      const path = section.path ? `${section.path}.${candidate.key}` : candidate.key;
      return path === selectionPath;
    });
    if (field?.kind === "enum") return field;
  }
  throw new Error(`selector '${selectionPath}' missing`);
}

const catalog = validateSchemeCatalog(MOCK_SCHEME_CATALOG);
check(catalog.runtimeStateFields.includes("rho"), "catalog must publish exact runtime state");
check(
  catalog.runtimeStateFields.length === new Set(catalog.runtimeStateFields).size,
  "runtime state inventory must be duplicate free",
);
check(
  catalog.readiness.items.length === 93 &&
    catalog.readiness.items.filter((item) => item.state === "ready").length === 87 &&
    ["physics.flat_lin_storm", "compute.strict_acoustic_residency", "compute.nvidia_cuda_assist"]
      .every((id) => catalog.readiness.items.find((item) => item.id === id)?.state === "ready") &&
    JSON.stringify(catalog.readiness.items
      .filter((item) => item.state === "engineering_incomplete")
      .map((item) => item.id).sort()) ===
      JSON.stringify([
        "catalog.runtime_cli_studio",
        "microphysics.milbrandt_triple_moment",
        "nesting.moist_two_way",
        "output.zfp_bounded_history",
        "physics.native_remaining",
        "turbulence.vertical_only_sgs",
    ]) &&
    catalog.readiness.items.find((item) => item.id === "terrain.complete_metal")?.scope ===
      "Exact dry stationary Bell and Schar mapped Cartesian C-grid Heun timesteps on fallback-off Metal",
  "browser fallback readiness must match the native registry, including incomplete scopes",
);
check(
  !catalog.runtimeCapabilities.interactionPolicy.qualifiedZfpPresetIds.includes("radar_ready") &&
    !catalog.runtimeCapabilities.interactionPolicy.qualifiedZfpPresetIds.includes("radar_polarimetric_ready") &&
    catalog.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4?.admitted === true &&
    catalog.readiness.items.find((item) => item.id === "output.zfp_bounded_history")?.state ===
      "engineering_incomplete",
  "qualified radar v4 is separate from lossy budgets and remains Engineering incomplete",
);
check(
  catalog.runtimeCapabilities.outputFormats.some(
    (format) => format.id === "netcdf" && format.available,
  ),
  "catalog must publish exact netCDF output availability",
);
const metalOutputScope = catalog.runtimeCapabilities.interactionPolicy.generatedMetalCanonicalScope;
check(
  catalog.runtimeCapabilities.interactionPolicy.version === 14 &&
    metalOutputScope.version === 4 && metalOutputScope.tupleCount === 6049 &&
    metalOutputScope.distinctConfigCount === 4537 &&
    JSON.stringify(metalOutputScope.outputFormats) ===
      JSON.stringify(["npy_3d", "netcdf", "zfp_high"]) &&
    JSON.stringify(metalOutputScope.writeCadenceSecondsByFormat?.npy_3d) ===
      JSON.stringify([1, 301]) &&
    JSON.stringify(metalOutputScope.writeCadenceSecondsByFormat?.netcdf) ===
      JSON.stringify([1, 301]) &&
    JSON.stringify(metalOutputScope.writeCadenceSecondsByFormat?.zfp_high) ===
      JSON.stringify([1]) &&
    metalOutputScope.manifestSha256 ===
      "062b609452e090b0b6e5ff1f7bb15cab896700f193796ff4d87a600994f28967" &&
    metalOutputScope.additionalBoundaryNpyScope?.familyId ===
      "boundary:2b5bfbe6dfa466c6baf991e3" &&
    metalOutputScope.additionalBoundaryNpyScope?.strictCanonicalYamlSha256 ===
      "a244b58833f7e3cd7060d941e8453d4e5face3dff9e18e87ebf61dd581a24d55" &&
    metalOutputScope.additionalBoundaryNpyScope?.manifestSha256 ===
      "26712e5d5714b313cc1e17f348d67b9db9eeea60e5b888d56b0b7cf3cc5673a9" &&
    JSON.stringify(metalOutputScope.additionalBoundaryNpyScope?.outputFormats) ===
      JSON.stringify(["npy_3d"]) &&
    JSON.stringify(metalOutputScope.additionalBoundaryNpyScope?.writeCadenceSeconds) ===
      JSON.stringify([1]),
  "browser fallback must publish exact v4 Metal scope plus one NPY-only east-boundary extension",
);
check(
  catalog.runtimeCapabilities.interactionPolicy.turbulenceDynamicCoefficientDefault === false &&
    catalog.runtimeCapabilities.interactionPolicy.qualifiedVerticalOnlySgs?.admitted === true &&
    catalog.readiness.items.find((item) => item.id === "turbulence.vertical_only_sgs")?.state ===
      "engineering_incomplete",
  "browser fallback must publish admitted exact static vertical SGS while retaining incomplete readiness",
);
check(
  catalog.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment?.retained === true &&
    catalog.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment?.admitted === false &&
    catalog.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment
      ?.strictV1ExplicitKeyAdmitted === false &&
    catalog.readiness.items.find((item) => item.id === "microphysics.milbrandt_triple_moment")
      ?.state === "engineering_incomplete",
  "browser fallback must retain guarded Milbrandt triple moment separately from ready double moment",
);
check(
  catalog.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4?.admitted === true &&
    catalog.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4
      ?.reviewedSourceProfileContract === "nimbus.qualified-radar-source-profile.v2" &&
    catalog.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4
      ?.nonRadarBudgetOwner === "qualifiedZfpFieldBudgets",
  "browser fallback must retain admitted radar v4 policy separate from lossy field budgets",
);
check(
  catalog.modules.length === 12 && catalogStudioCoverageIssues(catalog).length === 0,
  "all 12 exact-binary modules must have a consistent form/product-set/fixed/unavailable v1 policy",
);
check(
  catalogSelectorsOutsideForm(catalog).length === 0,
  "no form-exposed exact-binary module may lack a Studio control",
);
for (const module of catalog.modules) {
  if (module.studioSelection.mode === "fixed_v1") {
    const fixed = module.schemes.find(
      (scheme) => scheme.id === module.studioSelection.fixedSchemeId,
    );
    check(
      fixed?.constructible === true && fixed.runtimeSelectable === true && Boolean(module.studioSelection.reason),
      `${module.id} fixed-v1 policy must name a canonical runtime-selectable identity with evidence`,
    );
  } else if (module.studioSelection.mode === "unavailable_v1") {
    check(
      module.studioSelection.fixedSchemeId === "" && Boolean(module.studioSelection.reason),
      `${module.id} unavailable-v1 policy must carry explicit evidence and no fixed identity`,
    );
  } else if (module.studioSelection.mode === "product_set_v1") {
    check(
      module.selectionPath === "radar.products" &&
        module.studioSelection.fixedSchemeId === "" &&
        Boolean(module.studioSelection.reason),
      `${module.id} product-set policy must name its ordered strict-v1 path`,
    );
  }
}
const runtimeSchemes = catalog.modules.find((module) => module.id === "microphysics")!.schemes;
const runtimeIds = runtimeSchemes.map((scheme) => scheme.id);
const kessler = runtimeSchemes.find((scheme) => scheme.id === "kessler")!;
check(
  JSON.stringify(kessler.capabilities.staggerings) === JSON.stringify(["collocated", "c_grid"]),
  "offline Kessler capability must match the runtime collocated/C-grid ownership contract",
);
const field = microphysicsField(sectionsWithSchemeCatalog(catalog, "none"));

check(
  JSON.stringify(field.options.map((option) => option.value)) === JSON.stringify(runtimeIds),
  "form options must exactly match runtime canonical ids",
);

const allSections = sectionsWithSchemeCatalog(catalog, defaultsForSchema());
const rkConfig = parseYaml(readFileSync(
  "../tests/fixtures/configs/qualification/cgrid_ssprk3_coupled_ready.yaml",
  "utf8",
));
const rkSections = sectionsWithSchemeCatalog(catalog, rkConfig);
const rkIntegrator = selectorField(rkSections, "time.integrator");
check(
  rkIntegrator.disabledWhen?.(rkConfig) === false &&
    JSON.stringify(rkIntegrator.options.map((option) => option.value)) ===
      JSON.stringify(["rk3", "rk4", "split_explicit"]),
  "the ordinary form exposes exact-binary SSPRK3/RK4 choices in their admitted environment",
);
const rk4Config = applyFieldChange(rkConfig, "time.integrator", "rk4");
check(
  readPath(parseYaml(stringifyYaml(rk4Config)), "time.integrator") === "rk4",
  "the ordinary RK selector serializes the strict-v1 integrator token",
);
const westBoundary = allSections
  .find((section) => section.id === "boundary_conditions")
  ?.fields.find((candidate) =>
    candidate.kind === "enum" && candidate.key === "lateral.west"
  );
check(
  westBoundary?.kind === "enum" &&
    JSON.stringify(westBoundary.options.map((option) => option.value)) ===
      JSON.stringify([
        "periodic", "radiative", "zero_gradient", "open", "rigid", "rigid_slip",
      ]) &&
    westBoundary.options.find((option) => option.value === "open")?.label ===
      "Open (legacy auto)" &&
    westBoundary.description?.includes("radiative on C-grid") === true &&
    westBoundary.description?.includes("zero-gradient on A-grid") === true,
  "Studio boundary controls must expose explicit closures and label legacy open as staggering-dependent",
);
for (const path of formCatalogSelectorPaths()) {
  const module = catalog.modules.find((candidate) => candidate.selectionPath === path)!;
  const selector = selectorField(allSections, path);
  check(
    JSON.stringify(selector.options.map((option) => option.value)) ===
      JSON.stringify(module.schemes.map((scheme) => scheme.id)),
    `${path} options must exactly match runtime canonical ids`,
  );
  check(
    module.schemes.every((scheme) =>
      !scheme.constructible || !scheme.runtimeSelectable || selector.options.some((option) =>
        option.value === scheme.id && option.disabled !== true,
      )),
    `${path} must make every runtime-selectable descriptor reachable`,
  );
  check(
    module.schemes.flatMap((scheme) => scheme.aliases).every((alias) =>
      !selector.options.some((option) => option.value === alias),
    ),
    `${path} aliases must remain import-only`,
  );
}

let aliased = writePath(defaultsForSchema(), "turbulence.scheme", "smag");
aliased = writePath(aliased, "boundary_layer.scheme", "yonsei");
aliased = writePath(aliased, "extensions.chaos.scheme", "full");
const canonicalized = normalizeSchemeSelections(catalog, aliased);
check(
  readPath(canonicalized, "turbulence.scheme") === "smagorinsky" &&
    readPath(canonicalized, "boundary_layer.scheme") === "ysu" &&
    readPath(canonicalized, "extensions.chaos.scheme") === "full_stochastic",
  "generic normalization must canonicalize aliases across form modules",
);
const canonicalRoundTrip = parseYaml(stringifyYaml(canonicalized));
check(
  readPath(canonicalRoundTrip, "turbulence.scheme") === "smagorinsky" &&
    readPath(canonicalRoundTrip, "boundary_layer.scheme") === "ysu",
  "canonical non-microphysics selections must survive YAML round-trip",
);

const missingTurbulenceCatalog = structuredClone(catalog);
missingTurbulenceCatalog.modules = missingTurbulenceCatalog.modules.filter(
  (module) => module.id !== "turbulence",
);
const missingTurbulence = selectorField(
  sectionsWithSchemeCatalog(missingTurbulenceCatalog, defaultsForSchema()),
  "turbulence.scheme",
);

const fallbackEnableCatalog = structuredClone(catalog);
fallbackEnableCatalog.modules
  .find((module) => module.id === "boundary_layer")!
  .schemes.find((scheme) => scheme.id === "ysu")!.constructible = false;
const boundaryLayerSection = sectionsWithSchemeCatalog(
  fallbackEnableCatalog,
  defaultsForSchema(),
).find((section) => section.id === "boundary_layer");
check(
  boundaryLayerSection?.disable?.onValue === "slab",
  "re-enabling a module must choose a constructible runtime-published scheme",
);
check(
  missingTurbulence.disabledWhen?.({}) === true && missingTurbulence.options[0].disabled === true,
  "a missing descriptor-backed module must fail closed instead of using static options",
);

assertCatalogRejected(
  { ...MOCK_SCHEME_CATALOG, schema: "nimbus.scheme-catalog.v8", catalogVersion: 8 },
  "old v8 catalogs must fail visibly after the readiness-contract bump",
);
const missingEngineeringBlocker = structuredClone(MOCK_SCHEME_CATALOG);
missingEngineeringBlocker.readiness.items.push({
  ...missingEngineeringBlocker.readiness.items[0],
  id: "test.engineering_incomplete",
  state: "engineering_incomplete",
  blocker: "",
});
assertCatalogRejected(
  missingEngineeringBlocker,
  "engineering-incomplete readiness items must identify their blocker",
);
const missingBoundedEvidence = structuredClone(MOCK_SCHEME_CATALOG);
missingBoundedEvidence.readiness.items
  .find((item) => item.state === "ready")!.boundedEvidence = "";
assertCatalogRejected(
  missingBoundedEvidence,
  "ready items must carry bounded evidence",
);
const missingCampaignEvidence = structuredClone(MOCK_SCHEME_CATALOG);
missingCampaignEvidence.readiness.items[0].state = "campaign_validated";
assertCatalogRejected(
  missingCampaignEvidence,
  "campaign-validated items must carry campaign evidence",
);
const missingExternalReview = structuredClone(MOCK_SCHEME_CATALOG);
missingExternalReview.readiness.items[0].state = "externally_reviewed";
missingExternalReview.readiness.items[0].campaignEvidence = "campaign.json";
assertCatalogRejected(
  missingExternalReview,
  "externally reviewed items must carry review evidence",
);
const contradictorySchemeReadiness = structuredClone(MOCK_SCHEME_CATALOG);
const nonconstructibleReadyScheme = contradictorySchemeReadiness.modules
  .find((module) => module.id === "microphysics")!
  .schemes.find((scheme) => scheme.id === "kessler")!;
nonconstructibleReadyScheme.constructible = false;
nonconstructibleReadyScheme.readinessState = "ready";
assertCatalogRejected(
  contradictorySchemeReadiness,
  "nonconstructible schemes cannot advertise readiness",
);
const missingStudioPolicy = structuredClone(MOCK_SCHEME_CATALOG) as unknown as {
  modules: Array<Record<string, unknown>>;
};
delete missingStudioPolicy.modules[0].studioSelection;
assertCatalogRejected(
  missingStudioPolicy,
  "a module without explicit Studio exposure policy must fail closed",
);
const nonconstructibleFixed = structuredClone(MOCK_SCHEME_CATALOG);
nonconstructibleFixed.modules
  .find((module) => module.id === "soundings")!
  .schemes.find((scheme) => scheme.id === "auto")!.constructible = false;
assertCatalogRejected(
  nonconstructibleFixed,
  "a fixed-v1 policy cannot point to a nonconstructible descriptor",
);
const unregisteredFormModule = structuredClone(catalog);
unregisteredFormModule.modules.find((module) => module.id === "radar")!.studioSelection = {
  mode: "form",
  fixedSchemeId: "",
  reason: "",
};
check(
  catalogStudioCoverageIssues(unregisteredFormModule).some((issue) => issue.includes("radar")),
  "a newly form-exposed constructible module must fail coverage until Studio registers its selector",
);
assertCatalogRejected(
  unregisteredFormModule,
  "runtime catalog validation must fail closed when a form-exposed module lacks a Studio selector",
);
assertCatalogRejected(
  { ...MOCK_SCHEME_CATALOG, runtimeCapabilities: { outputFormats: [] } },
  "catalogs without output-format capabilities must fail closed",
);
const malformedMetalScope = structuredClone(catalog);
malformedMetalScope.runtimeCapabilities.interactionPolicy.generatedMetalCanonicalScope.manifestSha256 = "stale";
assertCatalogRejected(
  malformedMetalScope,
  "catalogs with a malformed generated Metal manifest identity must fail closed",
);
const malformedMetalCadence = structuredClone(catalog);
malformedMetalCadence.runtimeCapabilities.interactionPolicy
  .generatedMetalCanonicalScope.writeCadenceSecondsByFormat!.zfp_high = [1, 301];
assertCatalogRejected(
  malformedMetalCadence,
  "catalogs that invent an unreviewed ZFP write cadence must fail closed",
);
const radarPolicy11 = structuredClone(catalog);
radarPolicy11.runtimeCapabilities.interactionPolicy.version = 11;
delete radarPolicy11.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
delete radarPolicy11.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment;
radarPolicy11.readiness.items = radarPolicy11.readiness.items.filter(
  (item) => item.id !== "microphysics.milbrandt_triple_moment");
delete radarPolicy11.runtimeCapabilities.interactionPolicy.qualifiedVerticalOnlySgs;
delete radarPolicy11.runtimeCapabilities.interactionPolicy.turbulenceDynamicCoefficientDefault;
radarPolicy11.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4 = {
  version: 1,
  admitted: false,
  productIds: ["reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"],
  nonemptyProductSubsets: 31,
  coreFields: ["u", "v", "w", "theta", "p", "rho", "qv", "qr"],
  presets: ["minimal", "radar_ready", "radar_polarimetric_ready"],
  namedPresetProductSets: {
    radar_ready: ["reflectivity_dbz", "radial_velocity"],
    radar_polarimetric_ready: ["zdr", "kdp", "rhohv"],
  },
  polarimetricExtraFields: ["qc"],
  outputFormatTiers: ["zfp_low", "zfp_medium", "zfp_high"],
  backend: "cpu",
  coordinateSystem: "cartesian",
  staggering: "c_grid",
  terrain: "none",
  domainMotion: "stationary",
  nesting: false,
  reviewedSourceProfileContract: "nimbus.qualified-radar-source-profile.v2",
  reviewedSourceProfileOwnerSymbol: "is_public_qualified_radar_profile",
  reviewedSourceProfileFamilies: ["lin_cloudy_radar_ready_v1", "kessler_warm_cloud_radar_v1"],
  asyncIoModes: [false, true],
  gridShape: [12, 12, 48],
  gridSpacingMeters: [1000, 1000, 250],
  dtLargeSeconds: 0.2,
  acousticSubsteps: 6,
  writeCadenceSeconds: [1, 4],
  nonRadarMode: "accuracy",
  nonRadarContainerVersion: 3,
  nonRadarBudgetOwner: "qualifiedZfpFieldBudgets",
  radarMode: "reversible",
  radarContainerVersion: 4,
  radarKeyframeEveryExport: true,
  radarTolerancePolicy: "forbidden",
  manifestContract: "nimbus.zfp.per_field.v2",
  sourceFieldBoundsContract: "registered_field_contract_v1",
};
check(
  validateSchemeCatalog(radarPolicy11).runtimeCapabilities.interactionPolicy
    .qualifiedZfpReversibleRadarV4?.admitted === false,
  "version 11 must retain the exact guarded reversible radar ZFP contract",
);
const admittedRadarPolicy11 = structuredClone(radarPolicy11);
admittedRadarPolicy11.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4!.admitted = true;
check(
  validateSchemeCatalog(admittedRadarPolicy11).runtimeCapabilities.interactionPolicy
    .qualifiedZfpReversibleRadarV4?.admitted === true,
  "version 11 may signal native-reviewed radar v4 admission without changing its scope",
);
const missingRadarPolicy11 = structuredClone(radarPolicy11);
delete missingRadarPolicy11.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4;
assertCatalogRejected(missingRadarPolicy11, "version 11 cannot omit its radar v4 scope");
const staleRadarPolicy10 = structuredClone(radarPolicy11);
staleRadarPolicy10.runtimeCapabilities.interactionPolicy.version = 10;
delete staleRadarPolicy10.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
assertCatalogRejected(staleRadarPolicy10, "version 10 cannot invent a radar v4 scope");
staleRadarPolicy10.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4 =
  null as unknown as NonNullable<typeof staleRadarPolicy10.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4>;
assertCatalogRejected(staleRadarPolicy10, "version 10 cannot carry a null radar v4 scope");
for (const [label, mutate] of [
  ["admission type", (scope: Record<string, unknown>) => { scope.admitted = "yes"; }],
  ["subset count", (scope: Record<string, unknown>) => { scope.nonemptyProductSubsets = 30; }],
  ["product set", (scope: Record<string, unknown>) => { scope.productIds = ["reflectivity_dbz"]; }],
  ["core fields", (scope: Record<string, unknown>) => { scope.coreFields = ["u", "v"]; }],
  ["named preset", (scope: Record<string, unknown>) => {
    scope.namedPresetProductSets = { radar_ready: ["zdr"], radar_polarimetric_ready: ["zdr", "kdp", "rhohv"] };
  }],
  ["format tiers", (scope: Record<string, unknown>) => { scope.outputFormatTiers = ["zfp_high"]; }],
  ["source owner", (scope: Record<string, unknown>) => { scope.reviewedSourceProfileOwnerSymbol = "unknown"; }],
  ["source families", (scope: Record<string, unknown>) => { scope.reviewedSourceProfileFamilies = []; }],
  ["async modes", (scope: Record<string, unknown>) => { scope.asyncIoModes = [true]; }],
  ["grid shape", (scope: Record<string, unknown>) => { scope.gridShape = [18, 18, 18]; }],
  ["write cadence", (scope: Record<string, unknown>) => { scope.writeCadenceSeconds = [1]; }],
  ["nonradar budget owner", (scope: Record<string, unknown>) => { scope.nonRadarBudgetOwner = "radar"; }],
  ["radar codec", (scope: Record<string, unknown>) => { scope.radarMode = "accuracy"; }],
  ["radar tolerance", (scope: Record<string, unknown>) => { scope.radarTolerancePolicy = "allowed"; }],
  ["extra key", (scope: Record<string, unknown>) => { scope.unreviewed = true; }],
] as const) {
  const invalid = structuredClone(radarPolicy11);
  mutate(invalid.runtimeCapabilities.interactionPolicy.qualifiedZfpReversibleRadarV4 as unknown as Record<string, unknown>);
  assertCatalogRejected(invalid, `policy 11 must reject malformed radar ${label}`);
}
const verticalPolicy12 = structuredClone(admittedRadarPolicy11);
verticalPolicy12.runtimeCapabilities.interactionPolicy.version = 12;
delete verticalPolicy12.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
verticalPolicy12.runtimeCapabilities.interactionPolicy.turbulenceDynamicCoefficientDefault = false;
verticalPolicy12.runtimeCapabilities.interactionPolicy.qualifiedVerticalOnlySgs = {
  version: 1,
  admitted: false,
  scheme: "smagorinsky",
  mode: "vertical_only",
  dynamicCoefficient: false,
  executionMode: "qualified",
  backend: "cpu",
  coordinateSystem: "cartesian",
  staggering: "c_grid",
  terrain: "none",
  domainMotion: "stationary",
  nesting: false,
  lateralBoundaryMode: "periodic",
  activeBoundaryLayerAllowed: false,
  gridShape: [12, 12, 16],
  gridSpacingMeters: [1000, 1000, 250],
  dtLargeSeconds: 0.2,
  acousticSubsteps: 6,
  outputFormat: "npy_3d",
  writeCadenceSeconds: [1],
  sourceProfileContract: "nimbus.static-vertical-sgs-cpu.v1",
  sourceProfileOwnerSymbol: "is_transactional_static_vertical_only_cpu_candidate",
  wholeStepRollback: "state_and_six_science_ledgers",
  dynamicNonThreeDAllowed: false,
};
check(
  validateSchemeCatalog(verticalPolicy12).runtimeCapabilities.interactionPolicy
    .qualifiedVerticalOnlySgs?.admitted === false,
  "policy 12 must retain the exact fail-closed static vertical SGS scope",
);
const admittedVerticalPolicy12 = structuredClone(verticalPolicy12);
admittedVerticalPolicy12.runtimeCapabilities.interactionPolicy.qualifiedVerticalOnlySgs!.admitted = true;
check(
  validateSchemeCatalog(admittedVerticalPolicy12).runtimeCapabilities.interactionPolicy
    .qualifiedVerticalOnlySgs?.admitted === true,
  "policy 12 may advertise the native-reviewed exact static vertical SGS candidate",
);
const missingVerticalPolicy12 = structuredClone(verticalPolicy12);
delete missingVerticalPolicy12.runtimeCapabilities.interactionPolicy.qualifiedVerticalOnlySgs;
assertCatalogRejected(missingVerticalPolicy12, "policy 12 cannot omit its static vertical SGS scope");
const missingDynamicDefault = structuredClone(verticalPolicy12);
delete missingDynamicDefault.runtimeCapabilities.interactionPolicy.turbulenceDynamicCoefficientDefault;
assertCatalogRejected(missingDynamicDefault, "policy 12 cannot omit the native dynamic-coefficient default");
const changedDynamicDefault = structuredClone(verticalPolicy12);
changedDynamicDefault.runtimeCapabilities.interactionPolicy.turbulenceDynamicCoefficientDefault =
  true as unknown as false;
assertCatalogRejected(changedDynamicDefault, "policy 12 cannot change the native dynamic-coefficient default");
const historicalWithVertical = structuredClone(verticalPolicy12);
historicalWithVertical.runtimeCapabilities.interactionPolicy.version = 11;
delete historicalWithVertical.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
assertCatalogRejected(historicalWithVertical, "policy 11 cannot invent a static vertical SGS scope");
for (const [label, mutate] of [
  ["admission type", (scope: Record<string, unknown>) => { scope.admitted = "yes"; }],
  ["dynamic mode", (scope: Record<string, unknown>) => { scope.dynamicCoefficient = true; }],
  ["qualified execution", (scope: Record<string, unknown>) => { scope.executionMode = "exploratory"; }],
  ["backend", (scope: Record<string, unknown>) => { scope.backend = "metal"; }],
  ["grid shape", (scope: Record<string, unknown>) => { scope.gridShape = [18, 18, 18]; }],
  ["write cadence", (scope: Record<string, unknown>) => { scope.writeCadenceSeconds = [2]; }],
  ["source owner", (scope: Record<string, unknown>) => { scope.sourceProfileOwnerSymbol = "unknown"; }],
  ["rollback owner", (scope: Record<string, unknown>) => { scope.wholeStepRollback = "partial"; }],
  ["dynamic non3d", (scope: Record<string, unknown>) => { scope.dynamicNonThreeDAllowed = true; }],
  ["extra key", (scope: Record<string, unknown>) => { scope.unreviewed = true; }],
] as const) {
  const invalid = structuredClone(verticalPolicy12);
  mutate(invalid.runtimeCapabilities.interactionPolicy.qualifiedVerticalOnlySgs as unknown as Record<string, unknown>);
  assertCatalogRejected(invalid, `policy 12 must reject malformed vertical SGS ${label}`);
}
const milbrandtPolicy13 = structuredClone(admittedVerticalPolicy12);
milbrandtPolicy13.runtimeCapabilities.interactionPolicy.version = 13;
delete milbrandtPolicy13.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
milbrandtPolicy13.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment = {
  version: 1,
  retained: true,
  admitted: false,
  implementationState: "engineering_incomplete",
  selector: "microphysics.milbrandt.triple_moment",
  strictV1ExplicitKeyAdmitted: false,
  scheme: "milbrandt",
  baselineState: ["q", "N"],
  additionalState: ["Z"],
  additionalMomentOrder: 6,
  additionalMomentSpecies: ["rain", "ice", "snow", "graupel", "hail"],
  privateStorageUnits: "m^6/kg_dry",
};
milbrandtPolicy13.readiness.items.push({
  id: "microphysics.milbrandt_triple_moment",
  area: "microphysics",
  scope: "Retained Milbrandt-Yau triple-moment selection",
  state: "engineering_incomplete",
  boundedEvidence: "focused implementation evidence",
  campaignEvidence: "",
  externalReviewEvidence: "",
  blocker: "Five reflectivity moment process, transport, and backend owners remain.",
});
check(
  validateSchemeCatalog(milbrandtPolicy13).runtimeCapabilities.interactionPolicy
    .milbrandtTripleMoment?.admitted === false,
  "policy 13 must retain Milbrandt triple moment without admitting the strict-v1 key",
);
const missingMilbrandtTriple = structuredClone(milbrandtPolicy13);
delete missingMilbrandtTriple.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment;
assertCatalogRejected(missingMilbrandtTriple, "policy 13 must not omit retained Milbrandt triple moment");
const missingMilbrandtReadiness = structuredClone(milbrandtPolicy13);
missingMilbrandtReadiness.readiness.items = missingMilbrandtReadiness.readiness.items.filter(
  (item) => item.id !== "microphysics.milbrandt_triple_moment");
assertCatalogRejected(missingMilbrandtReadiness, "policy 13 must not omit incomplete Milbrandt readiness");
const wrongMilbrandtReadiness = structuredClone(milbrandtPolicy13);
const tripleReadiness = wrongMilbrandtReadiness.readiness.items.find(
  (item) => item.id === "microphysics.milbrandt_triple_moment")!;
tripleReadiness.state = "ready";
tripleReadiness.blocker = "";
assertCatalogRejected(wrongMilbrandtReadiness, "policy 13 must not promote retained Milbrandt triple moment");
const emptyMilbrandtBlocker = structuredClone(milbrandtPolicy13);
emptyMilbrandtBlocker.readiness.items.find(
  (item) => item.id === "microphysics.milbrandt_triple_moment")!.blocker = "";
assertCatalogRejected(emptyMilbrandtBlocker, "policy 13 must keep a nonempty Milbrandt engineering blocker");
const duplicateMilbrandtReadiness = structuredClone(milbrandtPolicy13);
duplicateMilbrandtReadiness.readiness.items.push(
  structuredClone(duplicateMilbrandtReadiness.readiness.items.find(
    (item) => item.id === "microphysics.milbrandt_triple_moment")!));
assertCatalogRejected(duplicateMilbrandtReadiness, "policy 13 must reject duplicate Milbrandt readiness");
const historicalMilbrandtTriple = structuredClone(milbrandtPolicy13);
historicalMilbrandtTriple.runtimeCapabilities.interactionPolicy.version = 12;
delete historicalMilbrandtTriple.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
assertCatalogRejected(historicalMilbrandtTriple, "policy 12 cannot invent Milbrandt triple moment");
for (const [label, mutate] of [
  ["retained", (scope: Record<string, unknown>) => { scope.retained = false; }],
  ["public admission", (scope: Record<string, unknown>) => { scope.admitted = true; }],
  ["strict-v1 key", (scope: Record<string, unknown>) => { scope.strictV1ExplicitKeyAdmitted = true; }],
  ["moment order", (scope: Record<string, unknown>) => { scope.additionalMomentOrder = 3; }],
  ["species", (scope: Record<string, unknown>) => { scope.additionalMomentSpecies = ["rain"]; }],
  ["storage unit", (scope: Record<string, unknown>) => { scope.privateStorageUnits = "kg/kg"; }],
  ["extra key", (scope: Record<string, unknown>) => { scope.unreviewed = true; }],
] as const) {
  const invalid = structuredClone(milbrandtPolicy13);
  mutate(invalid.runtimeCapabilities.interactionPolicy.milbrandtTripleMoment as unknown as Record<string, unknown>);
  assertCatalogRejected(invalid, `policy 13 must reject malformed Milbrandt ${label}`);
}
for (const [label, mutate] of [
  ["east netCDF", (scope: typeof metalOutputScope) => { scope.additionalBoundaryNpyScope!.outputFormats = ["npy_3d", "netcdf"]; }],
  ["east ZFP", (scope: typeof metalOutputScope) => { scope.additionalBoundaryNpyScope!.outputFormats = ["zfp_high"]; }],
  ["east cadence", (scope: typeof metalOutputScope) => { scope.additionalBoundaryNpyScope!.writeCadenceSeconds = [301]; }],
  ["east digest", (scope: typeof metalOutputScope) => { scope.additionalBoundaryNpyScope!.strictCanonicalYamlSha256 = "0".repeat(64); }],
  ["east manifest", (scope: typeof metalOutputScope) => { scope.additionalBoundaryNpyScope!.manifestSha256 = "0".repeat(64); }],
  ["aggregate count", (scope: typeof metalOutputScope) => { scope.tupleCount = 6048; }],
] as const) {
  const invalid = structuredClone(catalog);
  mutate(invalid.runtimeCapabilities.interactionPolicy.generatedMetalCanonicalScope);
  assertCatalogRejected(invalid, `catalogs with mismatched ${label} claims must fail closed`);
}
const mixedScope = catalog.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope!;
check(isDeepStrictEqual(mixedScope, QUALIFIED_FLAT_MIXED_METAL_NPY_SCOPE),
  "policy 14 publishes the independent exact mixed-open Metal contract");
const mixedManifest = readFileSync(new URL(
  "../../../fixtures/reference/catalog/flat_mixed_metal_cadence1_scope_v1.json", import.meta.url));
check(createHash("sha256").update(mixedManifest).digest("hex") === mixedScope.manifestSha256,
  "mixed-open Metal contract hash must bind the maintained manifest bytes");
const missingMixedScope = structuredClone(catalog);
delete missingMixedScope.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
assertCatalogRejected(missingMixedScope, "policy 14 requires its exact mixed-open Metal scope");
const historical13WithoutMixed = structuredClone(catalog);
historical13WithoutMixed.runtimeCapabilities.interactionPolicy.version = 13;
delete historical13WithoutMixed.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope;
check(validateSchemeCatalog(historical13WithoutMixed).runtimeCapabilities.interactionPolicy.version === 13,
  "historical policy 13 retains its original scopes without mixed admission");
const historical13WithMixed = structuredClone(catalog);
historical13WithMixed.runtimeCapabilities.interactionPolicy.version = 13;
assertCatalogRejected(historical13WithMixed, "historical policy 13 cannot acquire mixed admission metadata");
for (const [key, value] of [
  ["profileCount", 2], ["familyId", "boundary:orphan"],
  ["strictCanonicalYamlSha256", "0".repeat(64)], ["manifestSha256", "0".repeat(64)],
  ["outputFormats", ["npy_3d", "netcdf"]], ["writeCadenceSeconds", [1, 301]],
  ["gridShape", [18, 18, 20]], ["gridSpacingMeters", [500, 500, 251]],
  ["dtLargeSeconds", 0.2], ["acousticSubsteps", 5], ["asyncIo", true],
  ["memoryAdmissionPolicies", ["warn"]], ["memoryPolicyCanonicalValue", "reject"],
  ["lateralSpongeEnabled", false], ["lateralSpongeWidth", 0], ["lateralSpongeAlpha", 0],
  ["scalarTransportOwner", "cpu"], ["initialConditionChaosOwner", "native"],
  ["lateralDampingOwner", "native"], ["effectiveWOnlySponge", true],
  ["unreviewed", true],
] as const) {
  const invalid = structuredClone(catalog);
  (invalid.runtimeCapabilities.interactionPolicy.qualifiedFlatMixedMetalNpyScope as unknown as
    Record<string, unknown>)[key] = value;
  assertCatalogRejected(invalid, `policy 14 rejects changed mixed-open ${key}`);
}

const noZfpNativeVariant = structuredClone(catalog);
const noZfpScope = noZfpNativeVariant.runtimeCapabilities.interactionPolicy.generatedMetalCanonicalScope;
noZfpScope.version = 3;
noZfpScope.tupleCount = 4033;
noZfpScope.distinctConfigCount = 3025;
noZfpScope.outputFormats = ["npy_3d", "netcdf"];
delete noZfpScope.zfpTypedOutputManifestSha256;
delete noZfpScope.writeCadenceSecondsByFormat!.zfp_high;
noZfpScope.manifestSha256 = "4604b059d425fb86690207f69000fd708f153054d739091e0ec2ba00c7d9a4e3";
check(validateSchemeCatalog(noZfpNativeVariant).runtimeCapabilities.interactionPolicy
  .generatedMetalCanonicalScope.version === 3,
"the no-ZFP native v4 policy retains only NPY/netCDF plus the one east-NPY extension");

const withoutNetcdf = structuredClone(catalog);
const unavailableNetcdf = withoutNetcdf.runtimeCapabilities.outputFormats.find(
  (format) => format.id === "netcdf",
)!;
unavailableNetcdf.available = false;
unavailableNetcdf.implementation = "";
unavailableNetcdf.reason = "Exact fixture binary was built without libnetcdf.";
const outputFormat = selectorField(
  sectionsWithSchemeCatalog(withoutNetcdf, defaultsForSchema()),
  "output.format",
);
const netcdfOption = outputFormat.options.find((option) => option.value === "netcdf");
check(
  netcdfOption?.disabled === true &&
    netcdfOption.note === "Exact fixture binary was built without libnetcdf." &&
    outputFormat.showDisabledReasons === true,
  "Config Editor must keep unavailable netCDF visible, disabled, and paired with the exact reason",
);
assertCatalogRejected(
  { ...MOCK_SCHEME_CATALOG, runtimeStateFields: [] },
  "catalogs without an authoritative runtime-state inventory must fail closed",
);

for (const id of runtimeIds) {
  const selected = applyFieldChange(defaultsForSchema(), "microphysics.scheme", id);
  const roundTripped = parseYaml(stringifyYaml(selected));
  check(
    readPath(roundTripped, "microphysics.scheme") === id,
    `${id} must survive form -> YAML -> config round-trip`,
  );
}

check(
  canonicalSchemeId(catalog, "microphysics", "morrison") === undefined,
  "Morrison must not normalize to the physically distinct Milbrandt scheme",
);

const missing = microphysicsField(
  sectionsWithSchemeCatalog(catalog, "third_party_microphysics"),
).options[0];
check(
  missing.value === "third_party_microphysics" && missing.disabled === true,
  "a config value missing from the selected binary must remain visible and disabled",
);

const unavailable = microphysicsField(
  sectionsWithSchemeCatalog(null, "lin", "catalog probe failed"),
);
check(
  unavailable.disabledWhen?.({}) === true && unavailable.options[0].disabled === true,
  "catalog failure must disable selection instead of using stale options",
);

const withUnavailableScheme = structuredClone(catalog);
withUnavailableScheme.modules
  .find((module) => module.id === "microphysics")!
  .schemes.find((scheme) => scheme.id === "thompson")!.constructible = false;
const thompson = microphysicsField(
  sectionsWithSchemeCatalog(withUnavailableScheme, "none"),
).options.find((option) => option.value === "thompson");
check(
  thompson?.disabled === true,
  "runtime constructible=false must disable the corresponding option",
);

const kesslerConfig = normalizeMicrophysicsParameters(
  catalog,
  writePath(defaultsForSchema(), "microphysics.scheme", "kessler"),
);
check(
  readPath(kesslerConfig, "microphysics.shared.cloud_water_threshold_kgkg") === 1e-3 &&
    readPath(kesslerConfig, "microphysics.kessler.accretion_coefficient") === 2.2,
  "Kessler normalization must install runtime-published defaults",
);
const kesslerEvaporation = sectionsWithSchemeCatalog(catalog, "kessler")
  .find((section) => section.id === "microphysics")
  ?.fields.find((candidate) => candidate.key === "kessler.evaporation_efficiency");
check(
  kesslerEvaporation?.kind === "float" &&
    kesslerEvaporation.default === 1 &&
    kesslerEvaporation.min === 0 &&
    kesslerEvaporation.max === undefined &&
    kesslerEvaporation.step === 0.05 &&
    kesslerEvaporation.unit === undefined,
  "Kessler must expose the dimensionless reference-kernel evaporation control",
);
check(
  readPath(kesslerConfig, "microphysics.kessler.evaporation_efficiency") === 1,
  "Kessler normalization must install reference evaporation efficiency 1",
);

for (const id of ["thompson"]) {
  const evaporation = sectionsWithSchemeCatalog(catalog, id)
    .find((section) => section.id === "microphysics")
    ?.fields.find((candidate) => candidate.key === `${id}.evaporation_rate_s_inv`);
  check(
    evaporation?.kind === "float" &&
      evaporation.default === 1e-3 &&
      evaporation.min === 0 &&
      evaporation.max === undefined &&
      evaporation.step === 1e-4 &&
      evaporation.unit === "s⁻¹",
    `${id} evaporation metadata must generate the exact bounded-lower form control`,
  );
  const normalized = normalizeMicrophysicsParameters(
    catalog,
    writePath(defaultsForSchema(), "microphysics.scheme", id),
  );
  check(
    readPath(normalized, `microphysics.${id}.evaporation_rate_s_inv`) === 1e-3,
    `${id} normalization must install the runtime-published evaporation default`,
  );
}
check(
  runtimeSchemes.find((scheme) => scheme.id === "milbrandt")?.caveat.includes("six transported number moments") === true,
  "Milbrandt Studio metadata must retain its qualified-scope evidence",
);
const milbrandtParameters =
  runtimeSchemes.find((scheme) => scheme.id === "milbrandt")?.parameters ?? [];
check(
  milbrandtParameters.length === 1 &&
    milbrandtParameters[0]?.path === "microphysics.milbrandt.hail_processes",
  "Milbrandt must expose only the physically active hail-production switch",
);

const customizedKessler = writePath(
  kesslerConfig,
  "microphysics.kessler.accretion_coefficient",
  1.6,
);
const linConfig = normalizeMicrophysicsParameters(
  catalog,
  writePath(customizedKessler, "microphysics.scheme", "lin"),
);
check(
  readPath(linConfig, "microphysics.kessler") === undefined &&
    readPath(linConfig, "microphysics.lin.accretion_coefficient") === 2.2,
  "switching variants must prune inactive parameters and install active defaults",
);

const dryConfig = normalizeMicrophysicsParameters(
  catalog,
  writePath(linConfig, "microphysics.scheme", "none"),
);
check(
  JSON.stringify(readPath(dryConfig, "microphysics")) === JSON.stringify({ scheme: "none" }),
  "dry dynamics must contain no inactive wet-scheme parameters",
);

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`scheme catalog: ${passed} checks passed`);

function assertCatalogRejected(value: unknown, message: string): void {
  try {
    validateSchemeCatalog(value);
    check(false, message);
  } catch {
    check(true, message);
  }
}
