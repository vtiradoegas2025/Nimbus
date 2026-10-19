/*
 * Tests for src/lib/plotRegistry.ts (+ its interplay with src/lib/plotPresets.ts).
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_plot_registry.ts
 *
 * Guards the data-aware gating contract that drives the Plot Studio picker:
 *   - field-requirement logic (AND across clauses, OR within, "anyField"),
 *   - per-plot and per-preset availability,
 *   - the headline rule "a dry run must not show Radar",
 *   - whole-category hide (only when ALL contents unsupported),
 *   - the run-change fallback helpers,
 *   - registry invariants (one meta row per PlotType, PER_FRAME_PLOTS derives
 *     correctly, every preset resolves, recorder-selector contract).
 *
 * Both modules are pure (no React / Tauri), so the strip-types runner imports
 * them directly.
 */

import {
  reqSatisfied,
  missingRequirementClauses,
  isPlotAvailable,
  isPresetAvailable,
  presetRequirement,
  visiblePlotTypes,
  presetsForCategory,
  visiblePresets,
  categoryHasContent,
  visibleCategories,
  firstAvailableCategory,
  firstAvailablePlotType,
  metaFor,
  PER_FRAME_PLOTS,
  PLOT_TYPE_META,
  CATEGORY_META,
  CROSS_SECTION_VECTOR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLANE_SCALAR_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  TRAJECTORY_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
  type PlotType,
  type PlotCategory,
} from "../../../../ui/src/lib/plot/plotRegistry.ts";
import {
  CM1_FAMILIAR_VIEWS,
  PLOT_PRESETS,
  resolveCm1FamiliarViews,
} from "../../../../ui/src/lib/plot/plotPresets.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

const S = (...names: string[]) => new Set(names);

// A field set with no microphysics / minimal output (the "dry run" the user
// flagged) and a full-physics set exported with the `all` preset.
const empty = S();
const dry = S(
  "u", "v", "w", "p", "theta",
  PLANE_SCALAR_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  CROSS_SECTION_VECTOR_CAPABILITY,
);
// A maximal run: full physics exported with `all`, AND --trajectories (so the
// run-level capability sentinel is present and every plot type is reachable).
const full = S(
  "u", "v", "w", "p", "theta", "qv", "temperature", "dewpoint", "relative_humidity",
  "reflectivity_dbz", "composite_reflectivity", "radar", "total_condensate",
  "qc", "qr", "qg", "qh", "qs", "qi", "theta_e",
  "theta_prime", "p_prime", "divergence",
  "vorticity_r", "vorticity_theta", "vorticity_z", "vorticity_magnitude",
  "dynamic_pressure", "buoyancy", "potential_vorticity", "buoyancy_pressure", "pressure_gradient_force_z",
  "stretching_term", "tilting_term", "baroclinic_term",
  "cape", "cin", "updraft_helicity", "accumulated_rainfall", "precip_rate",
  "precipitable_water", "liquid_water_path", "cloud_water_path",
  PLANE_SCALAR_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  CROSS_SECTION_VECTOR_CAPABILITY,
  TRAJECTORY_CAPABILITY,
);

const ALL_TYPES: PlotType[] = [
  "slice", "vector", "composite", "timeseries", "timeheight", "swath", "vertical", "radial", "hodograph", "skewt",
  "volume", "isosurface", "streamlines", "supercell", "trajectories", "trajectory_budget", "python",
];
const ALL_CATEGORIES = new Set<PlotCategory>(CATEGORY_META.map((c) => c.value));
const findPreset = (id: string) => PLOT_PRESETS.find((p) => p.id === id)!;

// ---------- reqSatisfied: clause logic ----------
assertTrue(reqSatisfied("anyField", empty) === false, "anyField false on empty set");
assertTrue(reqSatisfied("anyField", dry) === true, "anyField true on non-empty set");
assertTrue(reqSatisfied([["p"], ["theta", "temperature"]], S("p", "theta")) === true, "AND+OR: p & theta");
assertTrue(reqSatisfied([["p"], ["theta", "temperature"]], S("p", "temperature")) === true, "AND+OR: p & temperature");
assertTrue(reqSatisfied([["p"], ["theta", "temperature"]], S("p")) === false, "AND+OR: p alone fails");
assertTrue(reqSatisfied([["p"], ["theta", "temperature"]], S("theta")) === false, "AND+OR: theta alone fails");
assertTrue(reqSatisfied([["reflectivity_dbz", "radar"]], S("radar")) === true, "OR within clause: radar alone");
assertTrue(reqSatisfied([["reflectivity_dbz", "radar"]], S("reflectivity_dbz")) === true, "OR within clause: dbz alone");
assertTrue(reqSatisfied([["reflectivity_dbz", "radar"]], S("u")) === false, "OR within clause: neither");
assertTrue(reqSatisfied([["u"], ["v"], ["w"]], S("u", "v", "w")) === true, "all-of velocity satisfied");
assertTrue(reqSatisfied([["u"], ["v"], ["w"]], S("u", "v")) === false, "all-of velocity missing w");
assertTrue(missingRequirementClauses("anyField", empty).length === 1, "readiness: empty anyField reports one gap");
assertTrue(missingRequirementClauses("anyField", dry).length === 0, "readiness: populated anyField has no gap");
{
  const gaps = missingRequirementClauses([["p"], ["theta", "temperature"], ["qv"]], S("p", "temperature"));
  assertTrue(gaps.length === 1 && gaps[0].length === 1 && gaps[0][0] === "qv", "readiness: reports only unsatisfied AND clauses");
}
{
  const gaps = missingRequirementClauses([["reflectivity_dbz", "radar"]], dry);
  assertTrue(gaps.length === 1 && gaps[0].length === 2, "readiness: preserves OR alternatives in a missing clause");
}

// ---------- isPlotAvailable: full vs dry ----------
assertTrue(isPlotAvailable("slice", dry) === true, "slice available on dry");
assertTrue(isPlotAvailable("skewt", dry) === true, "skewt available on dry (p+theta)");
assertTrue(isPlotAvailable("hodograph", dry) === true, "hodograph available on dry (u,v)");
assertTrue(isPlotAvailable("streamlines", dry) === true, "streamlines available on dry (u,v,w)");
assertTrue(isPlotAvailable("vector", dry) === true, "vector available on dry (w + u)");
assertTrue(isPlotAvailable("volume", dry) === true, "volume available on dry (rank-3 capability)");
assertTrue(isPlotAvailable("supercell", dry) === false, "supercell HIDDEN on dry (no cloud)");
assertTrue(isPlotAvailable("supercell", full) === true, "supercell available on full (cloud present)");
assertTrue(
  isPlotAvailable("supercell", S("u", "v", "w", "p", "theta", "cloud_condensate")) === true,
  "supercell available with cloud_condensate alone (gate includes the new field)",
);
assertTrue(
  isPlotAvailable("supercell", S("u", "v", "w", "p", "theta", "qc")) === false,
  "supercell HIDDEN with qc alone (the renderer's cloud layer reads only cloud_condensate/total_condensate)",
);
assertTrue(isPlotAvailable("slice", empty) === false, "slice unavailable on empty (anyField)");
for (const t of ALL_TYPES) assertTrue(isPlotAvailable(t, full) === true, `every plot type available on full: ${t}`);

// Shape/provider sentinels keep the picker aligned with the mounted read path.
const surfaceOnly = S("cape", PLANE_SCALAR_CAPABILITY, NATIVE_REDUCTION_CAPABILITY);
assertTrue(isPlotAvailable("slice", surfaceOnly), "rank-2-only data keeps scalar planes");
assertTrue(isPlotAvailable("swath", surfaceOnly), "rank-2-only data keeps swaths");
assertTrue(isPlotAvailable("timeseries", surfaceOnly), "native rank-2 data keeps scalar reducers");
assertTrue(!isPlotAvailable("volume", surfaceOnly), "rank-2-only data does not advertise volume");
assertTrue(!isPlotAvailable("isosurface", surfaceOnly), "rank-2-only data does not advertise isosurface");
assertTrue(!isPlotAvailable("vertical", surfaceOnly), "rank-2-only data does not advertise vertical profiles");
assertTrue(!isPlotAvailable("timeheight", surfaceOnly), "rank-2-only data does not advertise rank-3 axis reduction");
const providerVolume = S("my_expression", PLANE_SCALAR_CAPABILITY, VOLUME_SCALAR_CAPABILITY);
assertTrue(isPlotAvailable("volume", providerVolume), "provider-backed rank-3 data keeps spatial 3D");
assertTrue(!isPlotAvailable("timeseries", providerVolume), "provider-backed data does not advertise the native temporal reducer");
assertTrue(!isPlotAvailable("timeheight", providerVolume), "provider-backed data does not advertise the native axis reducer");
const planVectorOnly = S("u", "v", PLAN_VECTOR_CAPABILITY, VOLUME_SCALAR_CAPABILITY, PLANE_SCALAR_CAPABILITY);
const crossVectorOnly = S("u", "w", CROSS_SECTION_VECTOR_CAPABILITY, VOLUME_SCALAR_CAPABILITY, PLANE_SCALAR_CAPABILITY);
assertTrue(
  visiblePlotTypes("plan-views", planVectorOnly).some((meta) => meta.value === "vector"),
  "u/v exposes the plan-view vector renderer",
);
assertTrue(
  !visiblePlotTypes("cross-sections", planVectorOnly).some((meta) => meta.value === "vector"),
  "u/v alone does not expose a vertical vector cross-section",
);
assertTrue(
  visiblePlotTypes("cross-sections", crossVectorOnly).some((meta) => meta.value === "vector"),
  "u/w exposes the default x/z vector cross-section",
);
assertTrue(
  !visiblePlotTypes("plan-views", crossVectorOnly).some((meta) => meta.value === "vector"),
  "u/w alone does not expose an empty horizontal vector view",
);
assertTrue(
  !visiblePlotTypes("cross-sections", surfaceOnly).some((meta) => meta.value === "slice" || meta.value === "composite"),
  "rank-2-only maps do not masquerade as vertical cross-sections",
);

// ---------- Radar gating headline ----------
assertTrue(categoryHasContent("radar", dry, PLOT_PRESETS) === false, "HEADLINE: Radar hidden on dry run");
assertTrue(categoryHasContent("radar", S("reflectivity_dbz"), PLOT_PRESETS) === true, "Radar shown with reflectivity_dbz");
assertTrue(categoryHasContent("radar", S("composite_reflectivity"), PLOT_PRESETS) === true, "Radar shown with composite_reflectivity");
assertTrue(categoryHasContent("radar", S("radar"), PLOT_PRESETS) === true, "Radar shown with linear radar (fallback)");
assertTrue(isPresetAvailable(findPreset("surface-dbz"), dry) === false, "surface-dbz hidden on dry");
assertTrue(isPresetAvailable(findPreset("surface-dbz"), S("radar")) === true, "surface-dbz available via radar fallback");
assertTrue(isPresetAvailable(findPreset("composite-dbz"), S("radar")) === false, "composite-dbz needs composite_reflectivity, not radar");
assertTrue(isPresetAvailable(findPreset("composite-dbz"), S("composite_reflectivity")) === true, "composite-dbz available with composite_reflectivity");
assertTrue(isPresetAvailable(findPreset("surface-cape"), S("cape")) === true, "CAPE plan view is available from a horizontal diagnostic alone");
assertTrue(isPresetAvailable(findPreset("updraft-helicity"), S("updraft_helicity")) === true, "CM1 updraft helicity exposes its familiar plan view");
assertTrue(isPresetAvailable(findPreset("lowlevel-buoyancy"), S("buoyancy")) === true, "CM1 buoyancy exposes a familiar low-level plan view");
assertTrue(isPresetAvailable(findPreset("midlevel-pv"), S("potential_vorticity")) === true, "CM1 potential vorticity exposes a familiar mid-level plan view");
assertTrue(isPresetAvailable(findPreset("precipitable-water"), S("precipitable_water")) === true, "CM1 precipitable water exposes a column plan view");
assertTrue(isPresetAvailable(findPreset("liquid-water-path"), S("liquid_water_path")) === true, "CM1 liquid-water path exposes a column plan view");
assertTrue(isPresetAvailable(findPreset("cloud-water-path"), S("cloud_water_path")) === true, "CM1 cloud-water path exposes a distinct column plan view");
assertTrue(isPresetAvailable(findPreset("surface-temperature"), S("temperature")) === true, "derived CM1 temperature exposes a surface plan view");
assertTrue(findPreset("surface-temperature").colormap === "Viridis", "absolute temperature never defaults to a zero-centred diverging palette");
assertTrue(isPresetAvailable(findPreset("surface-dewpoint"), S("dewpoint")) === true, "derived CM1 dewpoint exposes a surface plan view");
assertTrue(isPresetAvailable(findPreset("surface-rh"), S("relative_humidity")) === true, "derived CM1 relative humidity exposes a surface plan view");
assertTrue(isPresetAvailable(findPreset("surface-theta-e"), S("theta_e")) === true, "derived CM1 theta-e exposes a surface plan view");
assertTrue(isPresetAvailable(findPreset("thetae-xsection"), S("theta_e")) === true, "derived CM1 theta-e exposes its familiar cross-section");
for (const [preset, field] of [["xsec-qc", "qc"], ["xsec-qr", "qr"], ["xsec-qi", "qi"], ["xsec-qs", "qs"], ["xsec-qg", "qg"]] as const) {
  assertTrue(isPresetAvailable(findPreset(preset), S(field)) === true, `qualified CM1 ${field} exposes its hydrometeor cross-section`);
}
assertTrue(isPresetAvailable(findPreset("xsec-qh"), S("qg")) === false, "qualified CM1 qg must not masquerade as scheme-ambiguous qh");
assertTrue(isPresetAvailable(findPreset("accumulated-rain"), S("accumulated_rainfall")) === true, "accumulated-rain plan view is data-gated");

// ---------- CM1 familiar-start projection ----------
assertTrue(CM1_FAMILIAR_VIEWS.length === 8, "CM1 familiar start stays intentionally short");
{
  const familiar = resolveCm1FamiliarViews(S("composite_reflectivity", "w", "u", "v", "p", "theta"));
  const reflectivity = familiar.find((view) => view.id === "reflectivity");
  assertTrue(reflectivity?.kind === "preset" && reflectivity.preset?.id === "composite-dbz", "familiar reflectivity falls back to qualified composite dBZ");
  assertTrue(familiar.find((view) => view.id === "updraft")?.available === true, "familiar updraft resolves from w");
  assertTrue(familiar.find((view) => view.id === "sounding")?.available === true, "familiar Skew-T resolves from p and theta");
  assertTrue(familiar.find((view) => view.id === "hodograph")?.available === true, "familiar hodograph resolves from u and v");
  assertTrue(familiar.find((view) => view.id === "cold-pool")?.available === false, "familiar cold pool remains gated without theta prime");
}
{
  const familiar = resolveCm1FamiliarViews(S("reflectivity_dbz"));
  const reflectivity = familiar.find((view) => view.id === "reflectivity");
  assertTrue(reflectivity?.kind === "preset" && reflectivity.preset?.id === "surface-dbz", "familiar reflectivity prefers the lowest-level 3-D product");
}

// ---------- whole-category hide rule ----------
assertTrue(visibleCategories(empty, PLOT_PRESETS).length === 0, "empty run: no categories visible");
assertTrue(categoryHasContent("volumetric", S("u", "v", VOLUME_SCALAR_CAPABILITY), PLOT_PRESETS) === true, "volumetric shown when a rank-3 scalar is supported");
assertTrue(isPlotAvailable("streamlines", S("u", "v")) === false, "streamlines individually hidden without w");
assertTrue(categoryHasContent("radar", S("u", "v"), PLOT_PRESETS) === false, "radar hidden without reflectivity");
assertTrue(categoryHasContent("plan-views", dry, PLOT_PRESETS) === true, "plan-views shown on dry (slice + w)");

// ---------- Trajectories gating (run-level capability sentinel) ----------
const traj = S(TRAJECTORY_CAPABILITY);
assertTrue(categoryHasContent("trajectories", dry, PLOT_PRESETS) === false, "Trajectories hidden without trajectories.npy");
assertTrue(categoryHasContent("trajectories", traj, PLOT_PRESETS) === true, "Trajectories shown with the capability sentinel");
assertTrue(isPlotAvailable("trajectories", dry) === false, "3D paths hidden on dry (no sentinel)");
assertTrue(isPlotAvailable("trajectory_budget", dry) === false, "vorticity budget hidden on dry (no sentinel)");
assertTrue(isPlotAvailable("trajectory_budget", traj) === true, "vorticity budget available with sentinel");
assertTrue(!visibleCategories(dry, PLOT_PRESETS).some((c) => c.value === "trajectories"), "Trajectories category absent from a dry run");
{
  const tt = visiblePlotTypes("trajectories", traj).map((m) => m.value);
  assertTrue(
    tt.includes("trajectories") && tt.includes("trajectory_budget"),
    "Trajectories category offers 3D paths + vorticity budget",
  );
}

// ---------- visiblePlotTypes filtering ----------
{
  const sound = visiblePlotTypes("soundings", S("p", "theta", VOLUME_SCALAR_CAPABILITY)).map((m) => m.value);
  assertTrue(sound.includes("skewt"), "soundings keeps skewt (p+theta)");
  assertTrue(sound.includes("vertical") && sound.includes("radial"), "soundings keeps vertical+radial (rank-3 field)");
  assertTrue(!sound.includes("hodograph"), "soundings drops hodograph without u,v");
}
{
  const radarTypes = visiblePlotTypes("radar", full);
  assertTrue(radarTypes.length === 0, "radar has no direct plot types (product-driven)");
}

// ---------- presets per category ----------
assertTrue(presetsForCategory("radar", PLOT_PRESETS).length === 3, "three radar presets");
{
  const planDry = visiblePresets("plan-views", dry, PLOT_PRESETS).map((p) => p.id);
  assertTrue(planDry.includes("midlevel-updraft"), "plan-views dry keeps midlevel-updraft (w)");
  assertTrue(!planDry.includes("dynamic-pressure"), "plan-views dry drops dynamic-pressure (not exported)");
  assertTrue(!planDry.includes("low-vorticity"), "plan-views dry drops low-vorticity (no vorticity_z)");
}
{
  const planFull = visiblePresets("plan-views", full, PLOT_PRESETS).map((p) => p.id);
  for (const id of ["midlevel-updraft", "low-vorticity", "mid-vorticity", "updraft-helicity", "cold-pool", "p-prime", "lowlevel-buoyancy", "surface-temperature", "surface-dewpoint", "surface-rh", "surface-theta-e", "midlevel-pv", "surface-cape", "surface-cin", "accumulated-rain", "precipitation-rate", "precipitable-water", "liquid-water-path", "cloud-water-path", "surface-convergence", "dynamic-pressure", "buoyancy-pressure", "ppgf-z", "stretching-term", "tilting-term", "baroclinic-term"]) {
    assertTrue(planFull.includes(id), `plan-views full includes ${id}`);
  }
}

// ---------- fallback helpers ----------
assertTrue(firstAvailableCategory(dry, PLOT_PRESETS) === "plan-views", "fallback: first category on dry is plan-views");
assertTrue(firstAvailableCategory(empty, PLOT_PRESETS) === null, "fallback: no category on empty (no throw)");
assertTrue(firstAvailablePlotType("plan-views", dry) === "slice", "fallback: first plan-views type is slice");
assertTrue(firstAvailablePlotType("cross-sections", dry) === "slice", "fallback: first cross-sections type is slice");
assertTrue(firstAvailablePlotType("radar", full) === null, "fallback: radar has no direct plot type");
assertTrue(firstAvailablePlotType("soundings", empty) === null, "fallback: empty soundings -> null (no throw)");

// ---------- registry invariants ----------
assertTrue(PLOT_TYPE_META.length === 17, "17 plot-type meta rows");
assertTrue(new Set(PLOT_TYPE_META.map((m) => m.value)).size === 17, "plot-type meta rows are unique");
/*
 * Parcel paths are a VTK renderer, so they belong on the tab users look at for 3D,
 * as well as under Trajectories beside the 2-D vorticity budget they are read with.
 */
assertTrue(
  visiblePlotTypes("volumetric", full).some((m) => m.value === "trajectories"),
  "trajectories offered on the 3D category",
);
assertTrue(
  visiblePlotTypes("trajectories", full).some((m) => m.value === "trajectories"),
  "trajectories still offered on its own category",
);
assertTrue(
  !visiblePlotTypes("volumetric", dry).some((m) => m.value === "trajectories"),
  "and stays hidden on the 3D category without the capability sentinel",
);
assertTrue(
  !visiblePlotTypes("volumetric", full).some((m) => m.value === "trajectory_budget"),
  "the 2D vorticity budget stays off the 3D category",
);
assertTrue(
  visiblePlotTypes("time-series", dry).some((m) => m.value === "timeheight"),
  "timeheight offered in time-series",
);
assertTrue(
  visiblePlotTypes("swaths", dry).some((m) => m.value === "swath"),
  "swath offered in swaths category",
);
assertTrue(presetsForCategory("swaths", PLOT_PRESETS).length === 4, "four swath presets");
assertTrue(isPlotAvailable("composite", dry) === true, "composite available on dry (rank-2/3 capability)");
assertTrue(
  visiblePlotTypes("plan-views", dry).some((m) => m.value === "composite"),
  "composite offered in plan-views",
);
assertTrue(
  visiblePlotTypes("cross-sections", dry).some((m) => m.value === "composite"),
  "composite offered in cross-sections",
);
for (const t of ALL_TYPES) {
  let ok = true;
  try { metaFor(t); } catch { ok = false; }
  assertTrue(ok, `metaFor resolves ${t}`);
}
// PER_FRAME_PLOTS == everything except timeseries
assertTrue(PER_FRAME_PLOTS.length === 12, "PER_FRAME_PLOTS has 12 entries");
assertTrue(!PER_FRAME_PLOTS.includes("timeseries"), "PER_FRAME_PLOTS excludes timeseries");
assertTrue(!PER_FRAME_PLOTS.includes("timeheight"), "PER_FRAME_PLOTS excludes timeheight");
assertTrue(!PER_FRAME_PLOTS.includes("swath"), "PER_FRAME_PLOTS excludes swath");
assertTrue(!PER_FRAME_PLOTS.includes("trajectories"), "PER_FRAME_PLOTS excludes trajectories");
assertTrue(!PER_FRAME_PLOTS.includes("trajectory_budget"), "PER_FRAME_PLOTS excludes trajectory_budget");
for (const t of ALL_TYPES.filter(
  (t) =>
    t !== "timeseries" &&
    t !== "timeheight" &&
    t !== "swath" &&
    t !== "trajectories" &&
    t !== "trajectory_budget",
)) {
  assertTrue(PER_FRAME_PLOTS.includes(t), `PER_FRAME_PLOTS includes ${t}`);
}
// every category references valid plot types
for (const c of CATEGORY_META) {
  for (const t of c.plotTypes) {
    let ok = true;
    try { metaFor(t); } catch { ok = false; }
    assertTrue(ok, `category ${c.value} references valid plot type ${t}`);
  }
}
// every preset is well-formed
for (const p of PLOT_PRESETS) {
  assertTrue(ALL_CATEGORIES.has(p.category), `preset ${p.id} has a valid category`);
  assertTrue(typeof p.field === "string" && p.field.length > 0, `preset ${p.id} has a field`);
  let typeOk = true;
  try { metaFor(p.plotType); } catch { typeOk = false; }
  assertTrue(typeOk, `preset ${p.id} has a valid plot type`);
  const req = presetRequirement(p);
  assertTrue(req !== "anyField" && Array.isArray(req) && req.length > 0, `preset ${p.id} requirement is concrete`);
}
// recorder-selector contract
const VTK = new Set<PlotType>(["volume", "isosurface", "streamlines", "supercell", "trajectories"]);
for (const m of PLOT_TYPE_META) {
  if (VTK.has(m.value)) {
    assertTrue(m.recorderSelector === ".volume-render__canvas", `${m.value} records from VTK canvas`);
  } else {
    assertTrue(m.recorderSelector === ".plot-canvas", `${m.value} records from Plotly canvas`);
  }
}

console.log(`\nplotRegistry tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
