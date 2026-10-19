/*
 * Tests for `src/lib/plot/volumetricHandoff.ts` — the rule deciding whether a 2-D scalar view can be
 * reproduced as a volume render, and what it carries over.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/plot/test_volumetric_handoff.ts
 */

import {
  canHandOffToVolume,
  describeVolumeHandoffBlock,
  volumeHandoffPreset,
  isVolumeHandoffPresetId,
  planeFocusFromView,
  translatedVolumeCameraFocus,
  VOLUMETRIC_SOURCE_TYPES,
  type ScalarSelection,
} from "../../../../ui/src/lib/plot/volumetricHandoff.ts";
import {
  isPresetAvailable,
  VOLUME_SCALAR_CAPABILITY,
  type PlotType,
} from "../../../../ui/src/lib/plot/plotRegistry.ts";
import { PLOT_PRESETS } from "../../../../ui/src/lib/plot/plotPresets.ts";

// ---------- Tiny assertion API (matches test_pane_layout.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

const S = (...names: string[]) => new Set(names);

function selection(over: Partial<ScalarSelection> = {}): ScalarSelection {
  return {
    field: "w", colormap: "RdBu", rangeMode: "symmetric", rank: 3,
    source: "slice", ...over,
  };
}

const available = S(
  "w",
  "reflectivity_dbz",
  "composite_reflectivity",
  "theta",
  VOLUME_SCALAR_CAPABILITY,
);

// ---------- Which source renderers offer the handoff ----------

check("the scalar-on-a-plane renderers offer it", () => {
  for (const type of ["slice", "composite"] as PlotType[]) {
    assertTrue(canHandOffToVolume(type, selection(), available), `${type} offers the handoff`);
  }
});

check("the quiver plot does not offer it, and says why", () => {
  // It carries a vector pair, not a scalar; a volume render of one component
  // would be a different quantity than the arrows on screen.
  assertTrue(!canHandOffToVolume("vector", selection(), available), "vector is not a source");
  assertTrue(
    (describeVolumeHandoffBlock("vector", selection(), available) ?? "").includes("streamlines"),
    "and it points at the right renderer",
  );
});

check("reducers do not offer it", () => {
  // These collapsed a dimension on purpose; "the same thing in 3-D" is not defined for them.
  const reducers: PlotType[] = [
    "timeseries", "timeheight", "swath", "vertical", "radial", "hodograph", "skewt",
    "trajectory_budget",
  ];
  for (const type of reducers) {
    assertTrue(!canHandOffToVolume(type, selection(), available), `${type} does not offer the handoff`);
  }
});

check("3-D renderers do not offer it to themselves", () => {
  for (const type of ["volume", "isosurface", "streamlines", "supercell", "trajectories"] as PlotType[]) {
    assertTrue(!canHandOffToVolume(type, selection(), available), `${type} does not offer the handoff`);
  }
});

check("the source list and the availability rule agree", () => {
  for (const type of VOLUMETRIC_SOURCE_TYPES) {
    assertTrue(canHandOffToVolume(type, selection(), available), `${type} is listed and allowed`);
  }
});

// ---------- Rank is the gate, not the field name ----------

check("a single-level field is refused even from a slice plot", () => {
  // composite_reflectivity is a column max: it is already a reduction, and a
  // volume render of it would be a flat sheet presented as storm structure.
  const flat = selection({ field: "composite_reflectivity", rank: 2 });
  assertTrue(!canHandOffToVolume("slice", flat, available), "rank 2 is refused");
  assertTrue(
    (describeVolumeHandoffBlock("slice", flat, available) ?? "").includes("composite_reflectivity"),
    "and the reason names the field",
  );
});

check("rank comes from the loaded array, so an unknown field is judged on its data", () => {
  // A studio-derived or CM1 field is absent from the output registry; the rank
  // still decides, which is why the gate reads the array rather than a name list.
  assertTrue(
    canHandOffToVolume("slice", selection({ field: "my_derived_thing", rank: 3 }), available),
    "an unregistered rank-3 field is allowed",
  );
  assertTrue(
    !canHandOffToVolume("slice", selection({ field: "my_derived_thing", rank: 2 }), available),
    "an unregistered rank-2 field is refused",
  );
});

check("an unknown rank is refused rather than assumed", () => {
  assertTrue(!canHandOffToVolume("slice", selection({ rank: null }), available), "null rank refused");
  assertTrue(!canHandOffToVolume("slice", selection({ rank: 1 }), available), "rank 1 refused");
  assertTrue(!canHandOffToVolume("slice", selection({ rank: 4 }), available), "rank 4 refused");
});

check("no selection means no handoff", () => {
  assertTrue(!canHandOffToVolume("slice", null, available), "null selection refused");
  assertTrue(!canHandOffToVolume("slice", selection({ field: "" }), available), "empty field refused");
});

// ---------- The blocked-reason string ----------

check("an available handoff reports no blocking reason", () => {
  assertTrue(describeVolumeHandoffBlock("slice", selection(), available) === null, "null when available");
});

check("each refusal explains itself", () => {
  assertTrue(
    (describeVolumeHandoffBlock("skewt", selection(), available) ?? "").length > 0,
    "a reducer explains itself",
  );
  assertTrue(
    (describeVolumeHandoffBlock("slice", null, available) ?? "").length > 0,
    "a missing selection explains itself",
  );
  assertTrue(
    describeVolumeHandoffBlock("slice", selection(), S()) !== null,
    "an empty run blocks the handoff",
  );
});

// ---------- The synthesized preset ----------

check("the preset carries the field and its display settings", () => {
  const preset = volumeHandoffPreset(
    selection({ field: "theta", colormap: "Inferno", rangeMode: "auto" }), 1);
  assertTrue(preset.plotType === "volume", "targets the volume renderer");
  assertTrue(preset.field === "theta", "carries the field");
  assertTrue(preset.colormap === "Inferno", "carries the colormap");
  assertTrue(preset.rangeMode === "auto", "carries the range mode");
  assertTrue(preset.category === "volumetric", "is filed under the 3D category");
});

check("the preset carries the exact source plane and zoom centre", () => {
  const focus = planeFocusFromView(
    2,
    17,
    4_250,
    0,
    1,
    { xMin: -8, xMax: 4, yMin: 6, yMax: 14 },
  );
  const preset = volumeHandoffPreset(selection({ planeFocus: focus }), 1);
  assertTrue(preset.volumeFocus?.axis === 2, "carries the sliced axis");
  assertTrue(preset.volumeFocus?.sourceIndex === 17, "carries the storage index");
  assertTrue(preset.volumeFocus?.coordinateM === 4_250, "carries the physical plane coordinate");
  assertTrue(
    preset.volumeFocus?.centerM.join(",") === "10000,-2000,4250",
    "maps Plotly row/column zoom to the physical 3D centre",
  );
});

check("camera focus translation preserves orbit direction and clamps to the domain", () => {
  const translated = translatedVolumeCameraFocus(
    [0, 0, 5_000],
    [-20_000, -30_000, 18_000],
    [12_000, null, 30_000],
    [-10_000, 10_000, -12_000, 12_000, 0, 20_000],
  );
  assertTrue(translated.focalPoint.join(",") === "10000,0,20000", "focus is clamped inside visible bounds");
  assertTrue(
    translated.position.join(",") === "-10000,-30000,33000",
    "camera eye translates with the focus and keeps its view vector",
  );
});

check("a composite handoff discloses that only its fill becomes a volume", () => {
  const preset = volumeHandoffPreset(selection({ source: "composite" }), 1);
  assertTrue(preset.description.includes("fill only"), "composite handoff names the retained 2-D overlays");
});

check("a dBZ view keeps its anchored palette and range", () => {
  // The NWS ramp is anchored to physical dBZ values; carrying the field but
  // re-ranging it would show the same storm in a different, wrong palette.
  const preset = volumeHandoffPreset(
    selection({ field: "reflectivity_dbz", colormap: "dBZ", rangeMode: "dBZ" }), 1);
  assertTrue(preset.colormap === "dBZ", "keeps the dBZ ramp");
  assertTrue(preset.rangeMode === "dBZ", "keeps the anchored range");
});

check("the preset gates on the field it carries", () => {
  const preset = volumeHandoffPreset(selection({ field: "theta" }), 1);
  assertTrue(isPresetAvailable(preset, S("theta")), "available when the field is exported");
  assertTrue(!isPresetAvailable(preset, S("w")), "unavailable when it is not");
});

check("each handoff mints a new id so the renderer re-applies it", () => {
  /*
   * The volume renderer re-applies a preset only when `preset.id` changes. A fixed
   * id would leave a second handoff — to a different field — showing the first
   * one, with the 2-D and 3-D panes silently disagreeing about what is on screen.
   */
  const first = volumeHandoffPreset(selection({ field: "w" }), 1);
  const second = volumeHandoffPreset(selection({ field: "theta" }), 2);
  assertTrue(first.id !== second.id, "successive handoffs differ by id");
  assertTrue(second.field === "theta", "and the later one carries the later field");
});

check("re-handing the same field still mints a new id", () => {
  // The action is imperative: clicking it after manually retuning the 3-D pane
  // must put the pane back, which only happens if the id moved.
  const first = volumeHandoffPreset(selection({ field: "w" }), 1);
  const again = volumeHandoffPreset(selection({ field: "w" }), 2);
  assertTrue(first.id !== again.id, "same field, new id");
});

check("handoff ids are recognisable and never collide with the shipped presets", () => {
  const preset = volumeHandoffPreset(selection(), 7);
  assertTrue(isVolumeHandoffPresetId(preset.id), "its own id is recognised");
  for (const shipped of PLOT_PRESETS) {
    assertTrue(!isVolumeHandoffPresetId(shipped.id), `shipped preset ${shipped.id} is not a handoff id`);
    assertTrue(shipped.id !== preset.id, `shipped preset ${shipped.id} does not collide`);
  }
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\nvolumetric-handoff tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
