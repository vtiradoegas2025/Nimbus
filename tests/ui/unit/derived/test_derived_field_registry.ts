import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import {
  DERIVED_FIELD_DEFINITIONS,
  DERIVED_FIELD_PROVENANCE_SCHEMA,
  DERIVED_FIELD_REGISTRY_SCHEMA,
  assessDerivedFields,
  augmentStepWithDerivedFields,
  buildDerivedFieldProvenance,
  derivedFieldCacheIdentity,
  evaluateDerivedField,
} from "../../../../ui/src/lib/derived/derivedFieldRegistry.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0; const failures: string[] = [];
const check = (condition: unknown, label: string) => condition ? passed += 1 : failures.push(label);

check(DERIVED_FIELD_DEFINITIONS.length === 5 && DERIVED_FIELD_DEFINITIONS.every(({ schema }) => schema === DERIVED_FIELD_REGISTRY_SCHEMA), "versioned immutable registry inventory");
check(Object.isFrozen(DERIVED_FIELD_DEFINITIONS) && DERIVED_FIELD_DEFINITIONS.every(Object.isFrozen), "registry and definitions are runtime immutable");
check(DERIVED_FIELD_DEFINITIONS.every(({ formula, method, invalidValuePolicy }) => formula.length > 0 && method.length > 0 && invalidValuePolicy.includes("NaN")), "formula method and invalid policy provenance");
check(DERIVED_FIELD_DEFINITIONS.filter(({ id }) => id !== "horizontal_wind_speed").every(({ parameters }) => parameters.length > 0 && parameters.every(({ mutable }) => !mutable)), "fixed scientific parameters are declared");

const canonicalUnits = { theta: "K", p: "Pa", qv: "kg/kg", u: "m/s", v: "m s-1" };
const assessment = assessDerivedFields(new Set(Object.keys(canonicalUnits)), canonicalUnits);
check(assessment.every(({ status }) => status === "available"), "source-neutral canonical dependencies unlock every built-in");
const badUnits = assessDerivedFields(new Set(["theta", "p"]), { theta: "K", p: "hPa" });
check(badUnits.find(({ definition }) => definition.id === "temperature")?.incompatibleUnits[0]?.actual === "hPa", "dependency units fail closed");
const nativeWins = assessDerivedFields(new Set(["temperature", "theta", "p"]), { temperature: "K", theta: "K", p: "Pa" });
check(nativeWins.find(({ definition }) => definition.id === "temperature")?.status === "native_precedence", "native fields take precedence");

const baseStep: StepInfo = {
  index: 4, simulationTimeS: 120, fields: ["theta", "p", "qv", "u", "v"],
  grid: { nr: 2, nth: 1, nz: 1, geometryKind: "physical" }, fieldUnits: canonicalUnits,
  publicationState: "complete", publicationGeneration: "frame-generation",
};
const cm1Step = augmentStepWithDerivedFields(baseStep);
const nimbusStep = augmentStepWithDerivedFields({ ...baseStep });
check(cm1Step.fields.join() === nimbusStep.fields.join() && cm1Step.fields.includes("horizontal_wind_speed"), "CM1 and Nimbus receive the same dependency-complete catalog");
check(cm1Step.fieldOrigins?.temperature === "studio_derived" && cm1Step.fieldUnits.horizontal_wind_speed === "m/s", "augmented frames retain origin and output unit");

const shape = [2, 1, 1];
const inputs = {
  theta: { shape, data: Float32Array.from([300, 300]) },
  p: { shape, data: Float32Array.from([100_000, 90_000]) },
  qv: { shape, data: Float32Array.from([0.01, -1]) },
  u: { shape, data: Float32Array.from([3, Number.NaN]) },
  v: { shape, data: Float32Array.from([4, 4]) },
};
const temperature = evaluateDerivedField("temperature", inputs);
const wind = evaluateDerivedField("horizontal_wind_speed", inputs);
check(temperature.data[0] === 300 && temperature.data[1] > 290 && temperature.data[1] < 292, "versioned temperature evaluator");
check(wind.data[0] === 5 && Number.isNaN(wind.data[1]), "wind magnitude and invalid propagation");
check(derivedFieldCacheIdentity("temperature").includes("reference_pressure_pa") && derivedFieldCacheIdentity("temperature") !== "temperature", "cache identity binds definition and parameters");

const source = createLocalDatasetSource("nimbus_run", "/runs/control");
const provenance = buildDerivedFieldProvenance(source, cm1Step, "temperature");
check(provenance.schema === DERIVED_FIELD_PROVENANCE_SCHEMA && provenance.frameIndex === 4 && provenance.dependencyIdentities.length === 2, "source frame and dependency provenance");

console.log(`derived registry: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) { failures.forEach((failure) => console.error(`FAIL ${failure}`)); process.exit(1); }
