/// <reference lib="dom" />

import {
  DATASET_ADAPTERS,
  DATASET_SOURCE_SCHEMA,
  createLocalDatasetSource,
  datasetAdapter,
  datasetPathLabel,
  datasetSourcePaths,
  datasetSourceIdentity,
} from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  assessCm1HistoryVariable,
  canonicalCm1Field,
  canonicalCm1HorizontalShape,
  canonicalCm1Shape,
  canonicalCm1ScalarShape,
  cm1LengthScaleToMeters,
  cm1ScalarCoordinate,
  cm1HistoryFieldMapping,
  cm1HistoryMappingForCanonical,
  cm1TimeScaleToSeconds,
  nativeCm1Field,
  uniformCm1Coordinate,
  transformCm1HistoryValues,
} from "../../../../ui/src/lib/cm1/cm1HistoryMapping.ts";
import { listDatasetFrames, type NetcdfMetadata } from "../../../../ui/src/lib/dataset/datasetRuntime.ts";
import { analyzeDatasetTime } from "../../../../ui/src/lib/dataset/datasetTimeHealth.ts";
import { summarizeCm1Namelist } from "../../../../ui/src/lib/cm1/cm1NamelistSummary.ts";
import { summarizeCm1Sounding } from "../../../../ui/src/lib/cm1/cm1SoundingSummary.ts";
import { buildCm1CompatibilityReport } from "../../../../ui/src/lib/cm1/cm1Compatibility.ts";
import { buildCm1PartialDraft } from "../../../../ui/src/lib/cm1/cm1PartialDraft.ts";
import { buildCm1SoundingProjection } from "../../../../ui/src/lib/cm1/cm1SoundingProjection.ts";
import { buildCm1SoundingProjectDraft } from "../../../../ui/src/lib/cm1/cm1SoundingProject.ts";
import { summarizeNativeCm1Sounding } from "../../../../ui/src/lib/cm1/cm1NativeSounding.ts";
import { validateCm1SoundingProjectEvidence } from "../../../../ui/src/lib/config/exactInitializationValidation.ts";
import { MOCK_SCHEME_CATALOG } from "../../../../ui/src/lib/config/schemeCatalog.ts";
import { availableCm1DerivedFields, cm1DerivedFieldSpec, deriveCm1Field } from "../../../../ui/src/lib/cm1/cm1DerivedFields.ts";

let passed = 0;
const failures: string[] = [];

function check(condition: unknown, message: string): void {
  if (condition) passed += 1;
  else failures.push(message);
}

check(DATASET_ADAPTERS.length === 2, "the initial adapter registry must be explicit");
check(
  DATASET_ADAPTERS.filter((adapter) => adapter.availability !== "unavailable").map(
    (adapter) => adapter.kind,
  ).join(",") === "nimbus_run,cm1_netcdf",
  "Nimbus history and CM1/netCDF must be selectable",
);
check(
  datasetAdapter("cm1_netcdf").availability === "available",
  "the CM1/netCDF adapter must expose its qualified field plotting support",
);
check(
  datasetAdapter("nimbus_run").multiresolution.mode === "manifest-hierarchy"
    && datasetAdapter("nimbus_run").multiresolution.detail.includes("NPY, ZFP, and netCDF histories"),
  "Nimbus history must disclose its maintained three-format hierarchy contract",
);
check(
  datasetAdapter("cm1_netcdf").multiresolution.mode === "bounded-native-window"
    && datasetAdapter("cm1_netcdf").multiresolution.detail.includes("does not claim a Nimbus LOD hierarchy"),
  "imported netCDF must disclose its exact bounded fallback instead of claiming durable LODs",
);

const source = createLocalDatasetSource("nimbus_run", " /research/cases/wk-supercell/run-01 ");
check(source.schema === DATASET_SOURCE_SCHEMA, "sources must carry a versioned schema identity");
check(source.schemaVersion === 1, "sources must carry schema version 1");
check(source.locator.path === "/research/cases/wk-supercell/run-01", "paths must be trimmed");
check(source.displayName === "…/wk-supercell/run-01", "display names must preserve useful context");
check(source.readOnly === true, "opened datasets must be read-only by contract");
check(
  datasetSourceIdentity(source) === datasetSourceIdentity({ ...source }),
  "equivalent sources must have stable identities",
);
check(datasetPathLabel("C:\\cm1\\runs\\control") === "…/runs/control", "Windows paths must label correctly");

const cm1 = createLocalDatasetSource("cm1_netcdf", "/research/cm1out.nc");
check(cm1.nativeFormat === "netcdf" && cm1.sourceModel === "cm1", "CM1 sources must retain native identity");
const cm1Series = createLocalDatasetSource("cm1_netcdf", [
  " /research/cm1out_000002.nc ",
  "/research/cm1out_000001.nc",
  "/research/cm1out_000001.nc",
]);
check(cm1Series.locator.kind === "local_paths", "a CM1 file series must retain multi-file locator identity");
check(datasetSourcePaths(cm1Series).length === 2, "CM1 file selections must be trimmed and deduplicated");
check(cm1Series.displayName === "2 netCDF history files", "multi-file CM1 selections need a concise dataset label");
check(datasetSourceIdentity(cm1Series) !== datasetSourceIdentity(cm1), "multi-file membership must participate in dataset identity");
check(canonicalCm1Field("winterp") === "w", "cell-centered CM1 winterp must map to canonical w");
check(nativeCm1Field("u") === "uinterp", "canonical u must read CM1 uinterp, not staggered u");
check(canonicalCm1Field("u") === null, "native staggered CM1 u must never masquerade as centered u");
check(canonicalCm1Field("th") === "theta", "CM1 total potential temperature must map to canonical theta");
check(canonicalCm1Field("thpert") === "theta_prime", "CM1 perturbation potential temperature must remain distinct from total theta");
check(canonicalCm1Field("prs") === "p" && canonicalCm1Field("prspert") === "p_prime", "CM1 total and perturbation pressure must retain distinct canonical identities");
check(canonicalCm1Field("rho") === "rho" && canonicalCm1Field("qv") === "qv", "CM1 dry-air density and water-vapor mixing ratio must retain their physical identities");
check(["qc", "qr", "qi", "qs", "qg"].every((name) => canonicalCm1Field(name) === name), "stable CM1 hydrometeor species must retain their canonical mixing-ratio identities");
check(cm1HistoryFieldMapping("qc", "kg kg-1")?.canonicalName === "qc" && cm1HistoryFieldMapping("qg", "kg/kg")?.canonicalName === "qg", "CM1 hydrometeors must require qualified dry-air mixing-ratio units");
check(cm1HistoryFieldMapping("qr", "g/kg") === null, "CM1 hydrometeors must fail closed when a conversion-requiring unit is undeclared");
check(assessCm1HistoryVariable({ name: "qh", units: "kg/kg", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2] }).reason.includes("depends on the selected microphysics scheme"), "ambiguous CM1 qh must explain why it remains inspect-only");
check(canonicalCm1Field("zvort") === "vorticity_z", "CM1 vertical vorticity must map to the Cartesian vertical component");
check(canonicalCm1Field("xvort") === null && canonicalCm1Field("yvort") === null, "CM1 Cartesian horizontal vorticity must not masquerade as legacy radial/azimuthal fields");
check(canonicalCm1Field("buoyancy") === "buoyancy", "CM1 buoyancy must retain its physical identity");
check(canonicalCm1Field("uh") === "updraft_helicity", "CM1 2-5 km updraft helicity must retain its diagnostic identity");
check(canonicalCm1Field("pv") === "potential_vorticity", "CM1 Ertel potential vorticity must retain its diagnostic identity");
check(canonicalCm1Field("pwat") === "precipitable_water", "CM1 precipitable water must map to a distinct column diagnostic");
check(canonicalCm1Field("lwp") === "liquid_water_path" && canonicalCm1Field("cwp") === "cloud_water_path", "CM1 liquid and cloud water paths must retain distinct identities");
check(canonicalCm1Field("div") === null, "unqualified CM1 divergence must remain inspect-only");
check(cm1HistoryFieldMapping("prs", "Pa")?.stateSemantic === "total", "qualified CM1 pressure must carry total-state semantics");
check(cm1HistoryFieldMapping("prspert", "Pa")?.stateSemantic === "perturbation", "qualified CM1 pressure perturbation must carry perturbation semantics");
check(cm1HistoryFieldMapping("qv", "g/kg") === null, "CM1 fields requiring unit conversion must remain inspect-only");
check(cm1HistoryFieldMapping("rho", null) === null, "CM1 fields with missing units must remain inspect-only");
check(cm1HistoryFieldMapping("cref", "dBZ")?.rank === 2, "CM1 composite reflectivity must qualify as a horizontal field");
check(cm1HistoryFieldMapping("zvort", "1/s")?.rank === 3, "CM1 vertical vorticity must qualify as a scalar volume");
check(cm1HistoryFieldMapping("buoyancy", "m/s^2")?.rank === 3, "CM1 buoyancy must qualify as a scalar volume");
check(cm1HistoryFieldMapping("uh", "m2/s2")?.rank === 2, "CM1 updraft helicity must qualify as a horizontal field");
check(cm1HistoryFieldMapping("pv", "K m2/kg/s")?.rank === 3, "CM1 potential vorticity must qualify as a scalar volume");
check(cm1HistoryFieldMapping("pwat", "m")?.rank === 2, "CM1 precipitable water must qualify as a horizontal field");
check(cm1HistoryFieldMapping("lwp", "kg/m^2")?.rank === 2 && cm1HistoryFieldMapping("cwp", "kg/m^2")?.rank === 2, "CM1 water paths must qualify as distinct horizontal fields");
check(
  assessCm1HistoryVariable({ name: "zvort", units: "1/s", dimensions: ["time", "nk", "nj", "ni"], shape: [2, 4, 5, 6] }).status === "ready",
  "the variable assessment must expose qualified vertical vorticity as plot-ready",
);
check(
  assessCm1HistoryVariable({
    name: "winterp",
    units: "m/s",
    dimensions: ["time", "nk", "nj", "ni"],
    shape: [2, 4, 5, 6],
    cfDecode: { status: "unsupported", issue: "invalid valid_range", scaleFactor: null, addOffset: null },
  }).reason.includes("CF packed/missing-value metadata is unsupported"),
  "invalid CF decode metadata must make an otherwise familiar CM1 field inspect-only",
);
check(
  assessCm1HistoryVariable({
    name: "winterp",
    units: "m/s",
    dimensions: ["time", "nk", "nj", "ni"],
    shape: [2, 4, 5, 6],
    cfDecode: { status: "applied", issue: null, scaleFactor: 0.1, addOffset: 273.15 },
  }).reason.includes("CF stored-value decoding applied before plotting"),
  "supported CF packing must be disclosed in CM1 field-readiness evidence",
);
check(
  assessCm1HistoryVariable({ name: "xvort", units: "1/s", dimensions: ["time", "nk", "nj", "ni"], shape: [2, 4, 5, 6] }).reason.includes("legacy radial/azimuthal"),
  "the variable assessment must explain the horizontal-vorticity naming blocker",
);
check(
  assessCm1HistoryVariable({ name: "uh", units: null, dimensions: ["time", "nj", "ni"], shape: [2, 5, 6] }).reason.startsWith("Units are missing"),
  "the variable assessment must explain missing units instead of hiding the field",
);
check(
  assessCm1HistoryVariable({ name: "svs", units: "1/s", dimensions: ["time", "nj", "ni"], shape: [2, 5, 6] }).reason.includes("running-extremum"),
  "the variable assessment must distinguish CM1 running extrema from instantaneous fields",
);
check(cm1HistoryMappingForCanonical("accumulated_rainfall")?.valueScale === 10, "CM1 centimetre rainfall must declare the exact millimetre scale");
check(cm1HistoryMappingForCanonical("precip_rate")?.valueScale === 3_600, "CM1 mass-flux precipitation rate must declare the water-equivalent hourly scale");
check(cm1HistoryMappingForCanonical("precipitable_water")?.valueScale === 1_000, "CM1 precipitable-water depth must declare the metre-to-millimetre scale");
check(
  [...transformCm1HistoryValues("accumulated_rainfall", new Float32Array([0.25, 1]))].join(",") === "2.5,10",
  "CM1 accumulated rainfall values must convert from centimetres to millimetres",
);
check(
  [...transformCm1HistoryValues("precip_rate", new Float32Array([0.001, 0.01]))].every((value, index) => Math.abs(value - [3.6, 36][index]) < 1e-5),
  "CM1 precipitation mass flux must convert to millimetres per hour",
);
check(
  [...transformCm1HistoryValues("precipitable_water", new Float32Array([0.025, 0.05]))].join(",") === "25,50",
  "CM1 precipitable-water depth must convert from metres to millimetres",
);
check(
  canonicalCm1Shape(["time", "nk", "nj", "ni"], [4, 2, 3, 5])?.join(",") === "5,3,2",
  "CM1 time/z/y/x variables must expose a canonical x/y/z grid",
);
check(
  canonicalCm1Shape(["time", "nk", "nj", "mystery"], [4, 2, 3, 5]) === null,
  "CM1 variables with an unknown spatial axis must remain inspect-only",
);
check(
  canonicalCm1Shape(["nk", "nj", "nj"], [2, 3, 5]) === null,
  "CM1 variables with duplicate axes must remain inspect-only",
);
check(
  canonicalCm1ScalarShape(["time", "zf", "yh", "xh"], [1, 3, 2, 2]) === null,
  "CM1 face-staggered variables must not qualify as scalar-point fields even when axis order is clear",
);
check(
  canonicalCm1ScalarShape(["time", "nk", "nj", "ni"], [1, 2, 3, 5])?.join(",") === "5,3,2",
  "legacy CM1 scalar index dimensions must remain supported",
);
check(
  canonicalCm1HorizontalShape(["time", "nj", "ni"], [1, 3, 5])?.join(",") === "5,3",
  "CM1 two-dimensional scalar fields must expose canonical x/y shape",
);
check(
  canonicalCm1HorizontalShape(["time", "nj", "nip1"], [1, 3, 6]) === null,
  "CM1 two-dimensional face fields must remain inspect-only",
);
const xCoordinate = uniformCm1Coordinate([-0.5, 0.5, 1.5], cm1LengthScaleToMeters("km"));
check(
  xCoordinate?.origin === -500 && xCoordinate.spacing === 1_000 && xCoordinate.lowerBound === -1_000,
  "uniform CM1 kilometre coordinates must become metre grid metadata",
);
check(
  uniformCm1Coordinate([0, 1, 2.25], 1_000) === null,
  "stretched CM1 coordinates must not masquerade as a uniform Studio grid",
);
const stretchedCoordinate = cm1ScalarCoordinate([0.1, 0.4, 1.0], 1_000);
check(
  stretchedCoordinate?.valuesM.join(",") === "100,400,1000"
    && stretchedCoordinate.uniform === null
    && stretchedCoordinate.lowerBound === -50
    && stretchedCoordinate.upperBound === 1_300,
  "stretched CM1 coordinates must retain exact centers and edge-extrapolated bounds",
);
check(cm1TimeScaleToSeconds("minutes since 2000-01-01") === 60, "CF-style time units must convert to seconds");

const derivedInputs = {
  theta: { shape: [2, 1, 1], data: new Float32Array([300, 300]) },
  p: { shape: [2, 1, 1], data: new Float32Array([100_000, 90_000]) },
  qv: { shape: [2, 1, 1], data: new Float32Array([0.01, -1]) },
};
const derivedTemperature = deriveCm1Field("temperature", derivedInputs);
const derivedDewpoint = deriveCm1Field("dewpoint", derivedInputs);
const derivedRh = deriveCm1Field("relative_humidity", derivedInputs);
const derivedThetaE = deriveCm1Field("theta_e", derivedInputs);
check(derivedTemperature.shape.join(",") === "2,1,1" && derivedTemperature.data[0] === 300, "CM1 derived temperature must preserve shape and satisfy the reference-pressure identity");
check(derivedTemperature.data[1] > 290 && derivedTemperature.data[1] < 292, "CM1 derived temperature must apply the Poisson pressure correction");
check(derivedDewpoint.data[0] > 285 && derivedDewpoint.data[0] < 289, "CM1 derived dewpoint must return Kelvin from vapor mixing ratio and pressure");
check(derivedRh.data[0] > 40 && derivedRh.data[0] < 50, "CM1 derived relative humidity must use vapor and saturation pressure");
check(Math.abs(derivedThetaE.data[0] - 329.631756611) < 0.001, "CM1 derived theta-e must reproduce a fixed Bolton Eq. 15/43 reference parcel");
check(cm1DerivedFieldSpec("theta_e")?.method.includes("Bolton (1980) Eq. 15/43") === true, "CM1 theta-e provenance must name the selected approximation and equations");
check(Number.isNaN(derivedDewpoint.data[1]) && Number.isNaN(derivedRh.data[1]) && Number.isNaN(derivedThetaE.data[1]), "invalid negative CM1 vapor mixing ratio must remain non-finite in derived moisture fields");
const dryThetaE = deriveCm1Field("theta_e", {
  theta: { shape: [1], data: new Float32Array([300]) },
  p: { shape: [1], data: new Float32Array([100_000]) },
  qv: { shape: [1], data: new Float32Array([0]) },
});
check(dryThetaE.data[0] === 300, "Bolton theta-e must retain its finite reference-pressure dry limit");
check(availableCm1DerivedFields(new Set(["theta", "p", "qv"])).map(({ id }) => id).join(",") === "temperature,dewpoint,relative_humidity,theta_e", "qualified total theta, pressure, and vapor must unlock all derived thermodynamics");
check(availableCm1DerivedFields(new Set(["theta", "p"])).map(({ id }) => id).join(",") === "temperature", "missing vapor must expose temperature without fabricating moisture diagnostics");

const fakeMetadata = (path: string): NetcdfMetadata => ({
  schema: "nimbus.studio.netcdf-metadata",
  schemaVersion: 1,
  path,
  containerFormat: "netcdf4",
  recordCount: 1,
  dimensions: [
    { name: "time", length: 1, unlimited: true },
    { name: "nk", length: 2, unlimited: false },
    { name: "nj", length: 2, unlimited: false },
    { name: "ni", length: 2, unlimited: false },
  ],
  variables: [
    { name: "time", dataType: "float", dimensions: ["time"], shape: [1], units: "seconds", longName: null },
    { name: "xh", dataType: "float", dimensions: ["ni"], shape: [2], units: "km", longName: null },
    { name: "yh", dataType: "float", dimensions: ["nj"], shape: [2], units: "km", longName: null },
    { name: "zh", dataType: "float", dimensions: ["nk"], shape: [2], units: "km", longName: null },
    { name: "winterp", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "m s-1", longName: null },
    { name: "th", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "K", longName: "potential temperature" },
    { name: "thpert", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "K", longName: "potential temperature perturbation" },
    { name: "prs", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "Pa", longName: "pressure" },
    { name: "prspert", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "Pa", longName: "pressure perturbation" },
    { name: "rho", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/m^3", longName: "dry-air density" },
    { name: "qv", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "water vapor mixing ratio" },
    { name: "qc", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "cloud-water mixing ratio" },
    { name: "qr", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "rainwater mixing ratio" },
    { name: "qi", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "cloud-ice mixing ratio" },
    { name: "qs", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "snow mixing ratio" },
    { name: "qg", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "graupel mixing ratio" },
    { name: "qh", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "scheme-dependent dense ice" },
    { name: "dbz", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "dBZ", longName: "reflectivity" },
    { name: "u", dataType: "float", dimensions: ["time", "nk", "nj", "nip1"], shape: [1, 2, 2, 3], units: "m/s", longName: "staggered u" },
    { name: "qvpert", dataType: "float", dimensions: ["time", "nk", "nj", "ni"], shape: [1, 2, 2, 2], units: "kg/kg", longName: "water vapor mixing ratio perturbation" },
    { name: "cref", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "dBZ", longName: "composite reflectivity" },
    { name: "cape", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "J/kg", longName: "convective available potential energy" },
    { name: "cin", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "J/kg", longName: "convective inhibition" },
    { name: "lcl", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "m", longName: "lifted condensation level" },
    { name: "lfc", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "m", longName: "level of free convection" },
    { name: "rain", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "cm", longName: "accumulated surface rainfall" },
    { name: "prate", dataType: "float", dimensions: ["time", "nj", "ni"], shape: [1, 2, 2], units: "kg/m2/s", longName: "surface precipitation rate" },
  ],
  sourceClassification: "cm1_likely",
  plottingSupported: true,
  plotUnavailableReason: "qualified subset",
});
let knownMetadataInspections = 0;
await listDatasetFrames(cm1, {
  inspect: async (selected) => {
    knownMetadataInspections += 1;
    return fakeMetadata(selected.locator.path);
  },
  readValues: async (_selected, variable) => {
    if (variable === "time") return [60];
    if (variable === "xh") return [-0.5, 0.5];
    if (variable === "yh") return [-1, 1];
    if (variable === "zh") return [0.25, 0.75];
    throw new Error(`unexpected coordinate ${variable}`);
  },
}, fakeMetadata(cm1.locator.path));
check(knownMetadataInspections === 0, "frame discovery must reuse metadata already inspected for the Studio header");
let activeFileInspections = 0;
let maximumFileInspections = 0;
const multiFrames = await listDatasetFrames(cm1Series, {
  inspect: async (selected) => {
    activeFileInspections += 1;
    maximumFileInspections = Math.max(maximumFileInspections, activeFileInspections);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    activeFileInspections -= 1;
    return fakeMetadata(selected.locator.path);
  },
  readValues: async (selected, variable) => {
    if (variable === "time") return [selected.locator.path.includes("000001") ? 60 : 120];
    if (variable === "xh") return [-0.5, 0.5];
    if (variable === "yh") return [-1, 1];
    if (variable === "zh") return [0.25, 0.75];
    throw new Error(`unexpected coordinate ${variable}`);
  },
});
check(maximumFileInspections === 1, "multi-file netCDF inspection stays sequential until native thread safety is qualified");
check(multiFrames.length === 2, "two CM1 history files must aggregate into one frame table");
check(multiFrames.map((frame) => frame.index).join(",") === "0,1", "multi-file CM1 frames need contiguous global indices");
check(multiFrames.map((frame) => frame.simulationTimeS).join(",") === "60,120", "CM1 files must order by physical time instead of picker order");
check(multiFrames[0].publicationGeneration?.includes("000001") === true, "each aggregate frame must retain its native source-file identity");
check(multiFrames[0].grid?.drM === 1_000 && multiFrames[0].grid?.dyM === 2_000, "aggregate frames must retain physical grid spacing");
check(
  multiFrames[0].grid?.scalarCoordinatesM?.[2].join(",") === "250,750",
  "aggregate frames must retain exact physical scalar-center arrays",
);
check(
  multiFrames.every((frame) => frame.fields.join(",") === "w,theta,theta_prime,p,p_prime,rho,qv,qc,qr,qi,qs,qg,reflectivity_dbz,composite_reflectivity,cape,cin,lcl,lfc,accumulated_rainfall,precip_rate,temperature,dewpoint,relative_humidity,theta_e"),
  "aggregate frames must expose qualified native and dependency-complete Studio-derived fields while excluding staggered or unmapped fields",
);
const unsupportedPackedFrames = await listDatasetFrames(cm1, {
  inspect: async (selected) => {
    const metadata = fakeMetadata(selected.locator.path);
    return {
      ...metadata,
      variables: metadata.variables.map((variable) => variable.name === "winterp"
        ? {
            ...variable,
            cfDecode: {
              status: "unsupported" as const,
              scaleFactor: null,
              addOffset: null,
              fillValue: null,
              fillValueSource: null,
              missingValues: [],
              validMin: null,
              validMax: null,
              validDomain: "stored" as const,
              missingOutput: "nan" as const,
              issue: "invalid valid_range",
            },
          }
        : variable),
    };
  },
  readValues: async (_selected, variable) => {
    if (variable === "time") return [60];
    if (variable === "xh") return [-0.5, 0.5];
    if (variable === "yh") return [-1, 1];
    if (variable === "zh") return [0.25, 0.75];
    throw new Error(`unexpected coordinate ${variable}`);
  },
});
check(
  unsupportedPackedFrames[0]?.fields.includes("w") === false,
  "a CM1 field with unsupported CF metadata must not enter the selectable frame catalog",
);
check(
  multiFrames.every((frame) => frame.fieldUnits.w === "m/s" && frame.fieldUnits.reflectivity_dbz === "dBZ"),
  "CM1 unit spellings must normalize to canonical Nimbus plot labels",
);
check(
  multiFrames.every((frame) => ["qc", "qr", "qi", "qs", "qg"].every((field) => frame.fieldUnits[field] === "kg/kg") && !frame.fields.includes("qh")),
  "CM1 frame catalogs must expose only unit-qualified, unambiguous hydrometeor species",
);
check(
  multiFrames.every((frame) => frame.fieldUnits.accumulated_rainfall === "mm" && frame.fieldUnits.precip_rate === "mm/h"),
  "converted CM1 surface diagnostics must advertise canonical post-transform units",
);
check(
  multiFrames.every((frame) => frame.fieldUnits.temperature === "K" && frame.fieldUnits.dewpoint === "K" && frame.fieldUnits.relative_humidity === "%" && frame.fieldUnits.theta_e === "K"),
  "Studio-derived CM1 thermodynamics must publish canonical plot units",
);
const stretchedFrames = await listDatasetFrames(cm1, {
  inspect: async (selected) => {
    const metadata = fakeMetadata(selected.locator.path);
    return {
      ...metadata,
      dimensions: metadata.dimensions.map((dimension) =>
        dimension.name === "nk" ? { ...dimension, length: 3 } : dimension
      ),
      variables: metadata.variables.map((variable) => ({
        ...variable,
        shape: variable.shape.map((length, axis) =>
          variable.dimensions[axis] === "nk" ? 3 : length
        ),
      })),
    };
  },
  readValues: async (_selected, variable) => {
    if (variable === "time") return [0];
    if (variable === "xh") return [-0.5, 0.5];
    if (variable === "yh") return [-1, 1];
    if (variable === "zh") return [0.1, 0.4, 1.0];
    throw new Error(`unexpected coordinate ${variable}`);
  },
});
check(
  stretchedFrames[0].grid?.dzM == null
    && stretchedFrames[0].grid?.geometryKind === "physical"
    && stretchedFrames[0].grid?.scalarCoordinatesM?.[2].join(",") === "100,400,1000"
    && stretchedFrames[0].grid?.physicalBoundsM?.upper[2] === 1_300,
  "stretched CM1 frames must preserve exact vertical coordinates without inventing uniform dz",
);
const partialCoordinateFrames = await listDatasetFrames(cm1, {
  inspect: async (selected) => fakeMetadata(selected.locator.path),
  readValues: async (_selected, variable) => {
    if (variable === "time") return [0];
    if (variable === "xh") return [-0.5, 0.5];
    if (variable === "yh") return [-1, 1];
    if (variable === "zh") return [0.75, 0.25];
    throw new Error(`unexpected coordinate ${variable}`);
  },
});
check(
  partialCoordinateFrames[0].grid?.geometryKind === "index"
    && partialCoordinateFrames[0].grid?.drM == null
    && partialCoordinateFrames[0].grid?.scalarCoordinatesM == null,
  "partial CM1 coordinate metadata must fall back wholly to index geometry instead of mixing metres and indices",
);
let stretchedMismatch = "";
try {
  await listDatasetFrames(cm1Series, {
    inspect: async (selected) => {
      const metadata = fakeMetadata(selected.locator.path);
      return {
        ...metadata,
        dimensions: metadata.dimensions.map((dimension) =>
          dimension.name === "nk" ? { ...dimension, length: 3 } : dimension
        ),
        variables: metadata.variables.map((variable) => ({
          ...variable,
          shape: variable.shape.map((length, axis) =>
            variable.dimensions[axis] === "nk" ? 3 : length
          ),
        })),
      };
    },
    readValues: async (selected, variable) => {
      if (variable === "time") return [selected.locator.path.includes("000001") ? 60 : 120];
      if (variable === "xh") return [-0.5, 0.5];
      if (variable === "yh") return [-1, 1];
      if (variable === "zh") return selected.locator.path.includes("000001")
        ? [0.1, 0.4, 1.0]
        : [0.1, 0.5, 1.0];
      throw new Error(`unexpected coordinate ${variable}`);
    },
  });
} catch (error) {
  stretchedMismatch = error instanceof Error ? error.message : String(error);
}
check(
  stretchedMismatch.includes("grid mismatch"),
  "multi-file CM1 series must reject different stretched coordinate arrays even when shape and endpoints match",
);
const horizontalOnlyFrames = await listDatasetFrames(cm1, {
  inspect: async (selected) => ({
    ...fakeMetadata(selected.locator.path),
    variables: fakeMetadata(selected.locator.path).variables.filter((variable) =>
      ["time", "xh", "yh", "zh", "cref", "cape", "rain", "prate"].includes(variable.name),
    ),
  }),
  readValues: async (_selected, variable) => {
    if (variable === "time") return [300];
    if (variable === "xh") return [-0.5, 0.5];
    if (variable === "yh") return [-1, 1];
    if (variable === "zh") return [0.25, 0.75];
    throw new Error(`unexpected coordinate ${variable}`);
  },
});
check(horizontalOnlyFrames.length === 1, "a CM1 history file containing only qualified horizontal diagnostics must still open");
check(horizontalOnlyFrames[0].grid?.nz === 2, "horizontal-only CM1 frames must retain the declared vertical domain for workspace compatibility");
check(
  horizontalOnlyFrames[0].fields.join(",") === "composite_reflectivity,cape,accumulated_rainfall,precip_rate",
  "horizontal-only CM1 frames must publish every qualified surface/column diagnostic",
);
const healthyTime = analyzeDatasetTime([
  { simulationTimeS: 0 }, { simulationTimeS: 60 }, { simulationTimeS: 120 },
]);
check(!healthyTime.hasWarnings && healthyTime.expectedCadenceS === 60, "uniform physical times must report a healthy cadence");
const unhealthyTime = analyzeDatasetTime([
  { simulationTimeS: 0 }, { simulationTimeS: 60 }, { simulationTimeS: 60 },
  { simulationTimeS: 240 }, { simulationTimeS: null },
]);
check(unhealthyTime.duplicateCount === 1, "duplicate CM1 output times must be reported");
check(unhealthyTime.gapCount === 1, "likely CM1 output gaps must be reported against median cadence");
check(unhealthyTime.missingTimeCount === 1 && unhealthyTime.hasWarnings, "missing physical times must remain visible as dataset health warnings");

const namelistSummary = summarizeCm1Namelist({
  schema: "nimbus.studio.cm1-namelist-report",
  schemaVersion: 1,
  path: "/research/case/namelist.input",
  groups: [
    { name: "param0", assignments: [] },
    { name: "param1", assignments: [] },
  ],
  assignmentCount: 8,
  values: {
    "param0.nx": "64",
    "param0.ny": "48",
    "param0.nz": "40",
    "param0.dx": "250.0",
    "param0.dy": "250.0",
    "param0.dz": "100.0",
    "param1.run_time": "1800.0",
    "param1.ptype": "3",
  },
  warnings: [],
});
check(namelistSummary.controls.includes("grid 64 × 48 × 40"), "CM1 namelist summaries must expose the source grid without translation");
check(namelistSummary.controls.includes("microphysics ptype 3"), "CM1 namelist summaries must expose raw selectable physics controls");
const ambiguousNamelist = summarizeCm1Namelist({
  schema: "nimbus.studio.cm1-namelist-report",
  schemaVersion: 1,
  path: "/research/case/namelist.input",
  groups: [],
  assignmentCount: 2,
  values: { "param0.ptype": "3", "param9.ptype": "5" },
  warnings: ["duplicate semantic key"],
});
check(!ambiguousNamelist.controls.some((control) => control.includes("ptype")), "ambiguous cross-group namelist controls must not be guessed through");
check(ambiguousNamelist.inspection.includes("1 parser warning"), "namelist parser warnings must remain visible in the summary");

const soundingReport = {
  schema: "nimbus.studio.cm1-sounding-report" as const,
  schemaVersion: 1 as const,
  path: "/research/case/input_sounding",
  surface: { pressureHpa: 1_000, potentialTemperatureK: 300, mixingRatioGkg: 14 },
  levels: [
    { sourceLine: 2, heightM: 0, potentialTemperatureK: 300, mixingRatioGkg: 14, uMs: 3, vMs: 4 },
    { sourceLine: 3, heightM: 17_000, potentialTemperatureK: 410, mixingRatioGkg: 0, uMs: 12, vMs: 5 },
  ],
  topHeightM: 17_000,
  warnings: [],
};
const soundingNamelist = {
  schema: "nimbus.studio.cm1-namelist-report" as const,
  schemaVersion: 1 as const,
  path: "/research/case/namelist.input",
  groups: [],
  assignmentCount: 4,
  values: {
    "param0.nz": "64",
    "param0.dz": "250.0d0",
    "param0.stretch_z": ".false.",
    "param0.isnd": "7",
  },
  warnings: [],
};
const soundingSummary = summarizeCm1Sounding(soundingReport, soundingNamelist);
check(soundingSummary.facts.includes("2 levels"), "CM1 sounding summaries must retain the native level count");
check(soundingSummary.facts.includes("peak wind 13 m/s"), "CM1 sounding summaries must characterize the native u/v wind column");
check(soundingSummary.disposition.includes("isnd=7") && soundingSummary.disposition.includes("extends above"), "CM1 sounding summaries must cross-check source selection and model-top coverage");
const shallowSounding = summarizeCm1Sounding(
  { ...soundingReport, topHeightM: 12_000 },
  { ...soundingNamelist, values: { ...soundingNamelist.values, "param0.isnd": "5" } },
);
check(shallowSounding.attention?.includes("does not select") === true, "inactive external soundings must be called out rather than implied active");
check(shallowSounding.attention?.includes("must exceed") === true, "profiles that do not cover the CM1 model top must be called out");
const nativeSummary = summarizeNativeCm1Sounding(soundingReport, 16_000);
check(nativeSummary.levelCount === 2 && nativeSummary.hasEmbeddedWinds, "native CM1 inspection retains the original level count and embedded u/v ownership evidence");
check(nativeSummary.coverage === "covers" && nativeSummary.coverageMessage.includes("covers"), "native CM1 inspection reports configured model-top coverage without converting the profile");
check(summarizeNativeCm1Sounding(soundingReport, 20_000).coverage === "shallow", "native CM1 inspection warns when the original profile stops below the Nimbus model top");

const soundingProjection = buildCm1SoundingProjection(soundingReport, soundingNamelist);
check(soundingProjection.schemaVersion === 1 && soundingProjection.exactCm1Equivalent === false, "CM1 sounding projections must be versioned without claiming exact equivalence");
check(soundingProjection.copyEligible && soundingProjection.levels.length === 2, "a valid isnd=7 profile must produce a complete copyable projection");
check(soundingProjection.levels[1].pressureHpa < soundingProjection.levels[0].pressureHpa, "projected sounding pressure must decrease hydrostatically");
check(Math.abs(soundingProjection.levels[0].windSpeedMs - 5) < 1e-12, "CM1 u/v must retain wind speed in the projected artifact");
check(Math.abs(soundingProjection.levels[0].windDirectionDeg - 216.86989764584402) < 1e-9, "CM1 u/v must translate to meteorological wind-from direction");
check(soundingProjection.text.includes("TMV_SHARPY_PROFILE_V1") && soundingProjection.text.includes("levels\t2"), "the projection must use Nimbus's actual accepted text contract");
check(!soundingProjection.text.toLowerCase().includes("nan"), "near-dry CM1 levels must retain a finite target-format dewpoint");
const ignoredWindProjection = buildCm1SoundingProjection(soundingReport, {
  ...soundingNamelist,
  values: { ...soundingNamelist.values, "param0.isnd": "17" },
});
check(!ignoredWindProjection.copyEligible && ignoredWindProjection.blockers.some((item) => item.includes("ignores input_sounding winds")), "isnd=17 must block a target artifact that would silently consume file winds");
const ambiguousWindProjection = buildCm1SoundingProjection(soundingReport, {
  ...soundingNamelist,
  values: { ...soundingNamelist.values, "param9.isnd": "17" },
});
check(!ambiguousWindProjection.copyEligible && ambiguousWindProjection.blockers.some((item) => item.includes("isnd is ambiguous")), "ambiguous CM1 isnd ownership must block projection rather than guessing how winds were used");
const malformedProjection = buildCm1SoundingProjection({
  ...soundingReport,
  levels: [soundingReport.levels[1], soundingReport.levels[0]],
});
check(!malformedProjection.copyEligible && malformedProjection.blockers.some((item) => item.includes("strictly in height")), "non-monotonic CM1 profiles must fail projection closed");
const soundingProject = buildCm1SoundingProjectDraft(
  "/research/nimbus-project/cm1-projected-sounding.txt",
  MOCK_SCHEME_CATALOG,
);
const projectSounding = (soundingProject.config.initial_conditions as {
  sounding?: { type?: string; file_path?: string; cape_jkg?: number };
} | undefined)?.sounding;
const projectHodograph = (soundingProject.config.initial_conditions as {
  hodograph?: { type?: string };
} | undefined)?.hodograph;
check(soundingProject.sourceScope === "reviewed_sounding_only", "CM1 sounding project drafts must disclose that no unresolved namelist controls were translated");
check(projectSounding?.type === "file" && projectSounding.file_path === "/research/nimbus-project/cm1-projected-sounding.txt", "the saved reviewed artifact must become the exact Nimbus file-sounding path");
check(projectSounding?.cape_jkg === undefined, "switching the project baseline to a file sounding must prune procedural source controls");
check(projectHodograph?.type === "auto", "a projected CM1 handoff must consume the wind column it carries instead of silently retaining custom default winds");
check(soundingProject.yaml.includes("IMPORT SCOPE: sounding only"), "saved baseline YAML must disclose its intentionally narrow CM1 import scope");
check(soundingProject.preflightRequest.initialization.soundingSource === "file", "exact-binary preflight must evaluate the projected file-sounding source");
const nativeSoundingProject = buildCm1SoundingProjectDraft(
  "/research/case/input_sounding",
  MOCK_SCHEME_CATALOG,
  { sourceMode: "native_cm1" },
);
check(nativeSoundingProject.sourceScope === "native_sounding_only" && nativeSoundingProject.sourceMode === "native_cm1", "native CM1 project drafts retain direct-source provenance without replacing projection support");
check(nativeSoundingProject.yaml.includes("NATIVE SOURCE: theta, qv, u, and v are read directly"), "native CM1 YAML discloses direct columns and Nimbus EOS reconstruction");
check(nativeSoundingProject.yaml.includes("file_path: /research/case/input_sounding"), "extensionless native CM1 paths survive generated YAML unchanged");
const initializationEvidence = validateCm1SoundingProjectEvidence({
  schema: "nimbus.studio.cm1-sounding-project-validation",
  schemaVersion: 1,
  executablePath: "/research/Nimbus/bin/nimbus_sim",
  summary: "[VALIDATE] PASS: strict v1 file/sharpy initialization",
  warnings: [],
});
check(initializationEvidence.summary.includes("file/sharpy"), "CM1 project save must retain exact-binary no-step initialization evidence");
let rejectedMissingInitializationPass = false;
try {
  validateCm1SoundingProjectEvidence({
    ...initializationEvidence,
    summary: "simulator exited successfully",
  });
} catch {
  rejectedMissingInitializationPass = true;
}
check(rejectedMissingInitializationPass, "CM1 project save must fail closed when native validation omits explicit PASS evidence");

const compatibility = buildCm1CompatibilityReport({
  namelist: {
    schema: "nimbus.studio.cm1-namelist-report",
    schemaVersion: 1,
    path: "/research/case/namelist.input",
    groups: [{
      name: "param0",
      assignments: [
        { key: "nx", rawValue: "96" },
        { key: "ptype", rawValue: "3" },
        { key: "isnd", rawValue: "7" },
        { key: "rstfrq", rawValue: "900" },
        { key: "output_format", rawValue: "2" },
        { key: "mystery_control", rawValue: ".true." },
      ],
    }],
    assignmentCount: 6,
    values: {
      "param0.nx": "96",
      "param0.ptype": "3",
      "param0.isnd": "7",
      "param0.rstfrq": "900",
      "param0.output_format": "2",
      "param0.mystery_control": ".true.",
    },
    warnings: [],
  },
  inventory: {
    schema: "nimbus.studio.cm1-case-inventory",
    schemaVersion: 1,
    directory: "/research/case",
    historyFiles: [],
    namelistInput: "/research/case/namelist.input",
    inputSounding: null,
    restartFiles: ["/research/case/cm1rst_000001.dat"],
    statisticsFiles: [],
    otherFileCount: 0,
    nestedDirectoryCount: 0,
  },
});
check(compatibility.schemaVersion === 1 && compatibility.entries.length === 6, "compatibility reports must be versioned and retain every parsed assignment");
check(compatibility.entries.find((entry) => entry.key === "nx")?.status === "ready", "mechanical same-unit CM1 scalars must be marked ready");
check(compatibility.entries.find((entry) => entry.key === "nx")?.target === "grid.nx", "ready mappings must name the exact Nimbus target");
check(compatibility.entries.find((entry) => entry.key === "ptype")?.status === "different", "familiar microphysics names must not imply implementation equivalence");
check(compatibility.entries.find((entry) => entry.key === "rstfrq")?.status === "needs_decision", "CM1 restart cadence must become a review decision now that Nimbus checkpoint cadence exists");
check(compatibility.entries.find((entry) => entry.key === "rstfrq")?.target === "extensions.checkpoint.every_s", "CM1 restart cadence must name the exact Nimbus review target without importing CM1 restart identity");
check(compatibility.entries.find((entry) => entry.key === "output_format")?.status === "needs_decision", "CM1 output format must remain an explicit review decision because CM1 layout codes do not transfer mechanically");
check(compatibility.entries.find((entry) => entry.key === "output_format")?.target === "output.format", "CM1 output format must name the exact Nimbus review target");
check(compatibility.entries.find((entry) => entry.key === "mystery_control")?.status === "unsupported", "unmapped namelist controls must remain visible and blocked");
check(compatibility.assetChecks.some((item) => item.id === "sounding" && item.status === "unsupported"), "an external-sounding mode without input_sounding must fail the case check");
check(compatibility.assetChecks.some((item) => item.id === "restarts" && item.status === "unsupported"), "CM1 restart candidates must not imply Nimbus continuation support");
check(compatibility.draftEligible === false && compatibility.draftBlockers.length > 0, "compatibility reports must block draft generation while translation remains unresolved");
const contextualCompatibility = buildCm1CompatibilityReport({
  namelist: {
    schema: "nimbus.studio.cm1-namelist-report",
    schemaVersion: 1,
    path: "/research/case/namelist.input",
    groups: [{
      name: "param0",
      assignments: [
        { key: "nx", rawValue: "64" },
        { key: "nx", rawValue: "96" },
        { key: "dz", rawValue: "250" },
        { key: "stretch_z", rawValue: ".true." },
        { key: "run_time", rawValue: "1800" },
        { key: "timax", rawValue: "3600" },
      ],
    }],
    assignmentCount: 6,
    values: {
      "param0.nx": "96",
      "param0.dz": "250",
      "param0.stretch_z": ".true.",
      "param0.run_time": "1800",
      "param0.timax": "3600",
    },
    warnings: ["duplicate assignment 'param0.nx'; the last raw value is shown"],
  },
});
check(contextualCompatibility.entries.filter((entry) => entry.key === "nx").every((entry) => entry.status === "needs_decision"), "duplicate source assignments must not remain mechanically ready");
check(contextualCompatibility.entries.find((entry) => entry.key === "dz")?.status === "needs_decision", "stretched CM1 grids must block direct dz transfer");
check(contextualCompatibility.entries.find((entry) => entry.key === "timax")?.status === "needs_decision", "shadowed timax must not compete with CM1 run_time for one Nimbus target");

const partialDraft = buildCm1PartialDraft(compatibility);
check(partialDraft.schemaVersion === 1 && partialDraft.runnable === false, "CM1 mechanical drafts must be versioned and permanently marked non-runnable");
check(partialDraft.includedEntryIds.length === 1, "only Ready assignments that pass value checks may enter a mechanical draft");
check((partialDraft.config.grid as { nx?: number } | undefined)?.nx === 96, "Ready CM1 scalars must populate their exact nested Nimbus targets");
check(partialDraft.omissions.length === compatibility.entries.length - 1, "every excluded source assignment must have an omission record");
check(partialDraft.yaml.includes("NON-RUNNABLE") && !partialDraft.yaml.includes("microphysics:"), "partial YAML must disclose its status and never fill unresolved sections with defaults");
const outOfRangeDraft = buildCm1PartialDraft(buildCm1CompatibilityReport({
  namelist: {
    schema: "nimbus.studio.cm1-namelist-report",
    schemaVersion: 1,
    path: "/research/large/namelist.input",
    groups: [{ name: "param0", assignments: [{ key: "nx", rawValue: "1024" }, { key: "ptype", rawValue: "0" }] }],
    assignmentCount: 2,
    values: { "param0.nx": "1024", "param0.ptype": "0" },
    warnings: [],
  },
}));
check(outOfRangeDraft.valueChecks.some((check) => check.target === "grid.nx" && !check.accepted), "Ready candidates outside current Studio ranges must fail draft value checks");
check((outOfRangeDraft.config.microphysics as { scheme?: string } | undefined)?.scheme === "none", "qualified enum translations such as ptype=0 must enter the partial draft explicitly");

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`dataset sources: ${passed} checks passed`);
