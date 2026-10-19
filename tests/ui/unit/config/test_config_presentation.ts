import { SCHEMA_SECTIONS } from "../../../../ui/src/lib/config/configSchema.ts";
import {
  BASIC_CONFIG_SECTION_IDS,
  advancedConfigSectionCount,
  isConfigDetailLevel,
  loadConfigDetailLevel,
  saveConfigDetailLevel,
  sectionsForConfigDetail,
} from "../../../../ui/src/lib/config/configPresentation.ts";

let checks = 0;
const check = (condition: boolean, message: string) => {
  if (!condition) throw new Error(message);
  checks += 1;
};

const basic = sectionsForConfigDetail(SCHEMA_SECTIONS, "basic");
const advanced = sectionsForConfigDetail(SCHEMA_SECTIONS, "advanced");
check(basic.length === 12, "Basic exposes the twelve maintained core sections");
check(advanced.length === SCHEMA_SECTIONS.length, "Advanced exposes every schema section");
check(advancedConfigSectionCount(SCHEMA_SECTIONS) === SCHEMA_SECTIONS.length - basic.length, "specialist count is exact");
check(basic.every((section) => BASIC_CONFIG_SECTION_IDS.has(section.id)), "Basic contains no specialist section");
check(["grid", "time", "microphysics", "output", "execution", "checkpoint"].every((id) => basic.some((section) => section.id === id)), "Basic retains setup, physics, publication, policy, and continuation");
check(["dynamics", "boundary_conditions", "diffusion", "validation"].every((id) => advanced.some((section) => section.id === id) && !basic.some((section) => section.id === id)), "Advanced restores specialist runtime sections");
check(isConfigDetailLevel("basic") && isConfigDetailLevel("advanced") && !isConfigDetailLevel("expert"), "detail values validate strictly");

const values = new Map<string, string>();
const storage = {
  getItem: (key: string) => values.get(key) ?? null,
  setItem: (key: string, value: string) => { values.set(key, value); },
};
check(loadConfigDetailLevel(storage) === "basic", "missing preference defaults to Basic");
saveConfigDetailLevel(storage, "advanced");
check(loadConfigDetailLevel(storage) === "advanced", "Advanced preference round-trips");
values.set("nimbus-studio:config-detail:v1", "invalid");
check(loadConfigDetailLevel(storage) === "basic", "invalid preference fails safely to Basic");

const diffusion = SCHEMA_SECTIONS.find((section) => section.id === "diffusion");
const horizontalK = diffusion?.fields.find((field) => field.key === "K_h");
check(horizontalK !== undefined, "diffusion presents the horizontal coefficient");
check(horizontalK?.visibleWhen?.({ scheme: "explicit", apply_to: "scalars" }) === true,
  "horizontal K is presented for its qualified explicit scalar target");
check(horizontalK?.visibleWhen?.({ scheme: "explicit", apply_to: "scalars_and_momentum" }) === true,
  "horizontal K is presented for the implemented staggered momentum target");

const turbulence = SCHEMA_SECTIONS.find((section) => section.id === "turbulence");
const turbulentPrandtl = turbulence?.fields.find((field) => field.key === "Pr_t");
const turbulentSchmidt = turbulence?.fields.find((field) => field.key === "Sc_t");
check(turbulentPrandtl?.default === 1 / 3,
  "Studio turbulent Prandtl default matches the strict C++ cloud-LES contract");
check(turbulentSchmidt?.default === 1 / 3,
  "Studio turbulent Schmidt default matches the strict C++ cloud-LES contract");

const hodograph = SCHEMA_SECTIONS.find((section) => section.id === "hodograph");
const hodographType = hodograph?.fields.find((field) => field.key === "type");
check(hodographType?.kind === "enum" && hodographType.default === "custom",
  "adding native embedded winds does not change the established Custom hodograph default");
check(hodographType?.kind === "enum" && hodographType.options.some((option) => option.value === "auto"),
  "the form exposes the engine's embedded-file-wind Auto mode");

console.log(`config presentation: ${checks} checks passed`);
