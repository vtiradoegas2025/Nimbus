/*
 * Domain-motion frame semantics (docs/domain.md).
 *
 * A translating grid breaks every plot that combines frames by CELL INDEX,
 * because the same index is a different Earth location in each frame. The
 * failure is silent and plausible: a smeared updraft-helicity swath still looks
 * like a storm swath. These tests pin the fail-closed policy.
 */

import {
  axisFrame,
  canAccumulateByCell,
  canCompareByCell,
  isStationary,
  motionMode,
  windsAreEarthRelative,
  type MotionBearingStep,
} from "../../../../ui/src/lib/run/frameMotion.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string): void => {
  if (condition) passed++;
  else failures.push(`FAIL ${label}`);
};

const step = (mode?: string | null, extra: Record<string, unknown> = {}): MotionBearingStep =>
  ({ motion: mode === undefined ? undefined : { mode, ...extra } });

/* ---------- Mode resolution ---------- */

// Runs written before the contract carry no motion block, and they were
// stationary. Absence is a fact about those runs, not missing information.
check(motionMode(step()) === "stationary", "a run with no motion block is stationary");
check(motionMode(null) === "stationary", "a missing step is stationary");
check(motionMode(step(null)) === "stationary", "an explicit null mode is stationary");
check(motionMode(step("")) === "stationary", "an empty mode is stationary");
check(motionMode(step("stationary")) === "stationary", "an explicit stationary mode resolves");
check(motionMode(step("constant_velocity")) === "constant_velocity", "translation resolves");
check(motionMode(step("tracked_window")) === "tracked_window", "a tracked window resolves");
check(motionMode(step("moving_nest")) === "moving_nest", "a moving nest resolves");

// Fail closed: a mode from a newer engine must refuse, never fall back to
// stationary. Assuming stationary is exactly how a smeared swath gets published.
check(motionMode(step("adaptive_mesh")) === null, "an unknown future mode is refused");
check(motionMode(step("STATIONARY")) === null, "mode matching is exact, not case-folded");

check(isStationary(step()), "isStationary is true for a contract-free run");
check(!isStationary(step("constant_velocity")), "isStationary is false under translation");
check(!isStationary(step("adaptive_mesh")), "isStationary is false for an unknown mode");

/* ---------- Accumulating across time by cell ---------- */

check(canAccumulateByCell([]).allowed, "an empty run trivially permits accumulation");
check(
  canAccumulateByCell([step(), step("stationary")]).allowed,
  "a stationary run permits accumulation by cell",
);

const translating = canAccumulateByCell([step("stationary"), step("constant_velocity")]);
check(!translating.allowed, "a single translating frame invalidates the whole accumulation");
check(
  (translating.blocker ?? "").includes("constant_velocity"),
  "the blocker names the mode responsible",
);
check(
  (translating.blocker ?? "").includes("smear"),
  "the blocker explains the failure rather than only refusing",
);

for (const mode of ["tracked_window", "moving_nest"]) {
  const verdict = canAccumulateByCell([step(mode)]);
  check(!verdict.allowed, `${mode} refuses accumulation by cell`);
  check((verdict.blocker ?? "").length > 0, `${mode} explains why`);
}

const unknown = canAccumulateByCell([step("adaptive_mesh")]);
check(!unknown.allowed, "an unrecognised mode refuses accumulation");
check(
  (unknown.blocker ?? "").includes("does not recognise"),
  "an unrecognised mode says so rather than blaming motion it cannot name",
);

/* ---------- Comparing two datasets by cell ---------- */

check(canCompareByCell(step(), step()).allowed, "two stationary datasets compare by cell");
check(
  !canCompareByCell(step(), step("constant_velocity")).allowed,
  "a translating dataset cannot be differenced by cell",
);

// Two stationary runs on different Earth origins share no cell either: the
// difference would subtract one place from another.
const shifted = canCompareByCell(
  step("stationary", { earthOriginM: [0, 0] }),
  step("stationary", { earthOriginM: [50_000, 0] }),
);
check(!shifted.allowed, "different Earth origins cannot be differenced by cell");
check(
  (shifted.blocker ?? "").includes("origin"),
  "the origin blocker names the origin as the cause",
);
check(
  canCompareByCell(
    step("stationary", { earthOriginM: [1_000, 2_000] }),
    step("stationary", { earthOriginM: [1_000, 2_000] }),
  ).allowed,
  "a shared Earth origin compares by cell",
);
check(
  canCompareByCell(step("stationary"), step("stationary", { earthOriginM: [10, 20] })).allowed,
  "an undeclared origin does not manufacture a mismatch",
);

/* ---------- Wind frame ---------- */

check(windsAreEarthRelative(step()), "a stationary run's winds are unambiguously Earth-relative");
check(
  !windsAreEarthRelative(step("constant_velocity")),
  "a translating run must DECLARE its wind frame; silence is not Earth-relative",
);
check(
  windsAreEarthRelative(step("constant_velocity", { windsEarthRelative: true })),
  "a translating run that declares Earth-relative winds is trusted",
);
check(
  !windsAreEarthRelative(step("constant_velocity", { windsEarthRelative: false })),
  "declared grid-relative winds are not Earth-relative",
);

/* ---------- Axis frame provenance ---------- */

// The overwhelmingly common case: this must stay SILENT. A notice on every
// stationary run would train users to ignore it, which costs the warning its
// value on the one run that needs it.
check(axisFrame(step()).frame === "earth-fixed", "a run with no motion block has earth-fixed axes");
check(axisFrame(step()).headline === null, "an earth-fixed run raises no axis notice");
check(axisFrame(step()).detail === null, "an earth-fixed run has nothing to disclose");
check(axisFrame(null).frame === "earth-fixed", "a missing step has earth-fixed axes");
check(
  axisFrame(step("stationary")).headline === null,
  "an explicitly stationary run raises no axis notice",
);

for (const mode of ["constant_velocity", "tracked_window", "moving_nest"]) {
  const verdict = axisFrame(step(mode));
  check(verdict.frame === "grid-relative", `${mode} axes are grid-relative`);
  check(Boolean(verdict.headline), `${mode} raises an axis notice`);
  check(
    Boolean(verdict.detail) && verdict.detail!.includes(mode),
    `${mode} names the declared mode in its disclosure`,
  );
}

// Fail closed, exactly as motionMode does: an unrecognised mode must not be
// reported as earth-fixed, because assuming is the failure being guarded.
const unknownFrame = axisFrame(step("adaptive_mesh"));
check(unknownFrame.frame === "unverified", "an unknown motion mode leaves the axis frame unverified");
check(Boolean(unknownFrame.headline), "an unverified axis frame raises a notice");
check(
  axisFrame(step("STATIONARY")).frame === "unverified",
  "axis frame matching is exact, not case-folded",
);

// A reader must be told what the numbers DO mean, not just that something is
// wrong, or the only available action is to distrust the whole figure.
check(
  axisFrame(step("constant_velocity")).detail!.includes("grid"),
  "a grid-relative disclosure says what the coordinates are relative to",
);

const contract = "constant_translation_earth_wind_grid_transport_v1";
for (const mode of ["constant_velocity", "tracked_window", "moving_nest"]) {
  for (const grid of [
    { scalarOriginM: [43050, 16100, 25] as [number, number, number] },
    { horizontalScalarOriginM: [43050, 16100] as [number, number] },
  ]) {
    const registered = { ...step(mode, { horizontalFrameContract: contract }), grid };
    check(axisFrame(registered).frame === "earth-fixed", `${mode} published scalar origins are Earth registered`);
    check(axisFrame(registered).headline === null, `${mode} does not mislabel published Earth axes`);
    check(!canAccumulateByCell([registered]).allowed, `${mode} Earth axes do not permit cell-index accumulation`);
  }
}
check(axisFrame(step("constant_velocity", { horizontalFrameContract: contract })).frame === "unverified",
  "a contract without scalar coordinates does not establish usable Earth axes");
check(axisFrame({ ...step("constant_velocity", { horizontalFrameContract: contract }),
  grid: { scalarOriginM: [NaN, 0, 0] } }).frame === "unverified", "invalid origins remain unverified");
check(axisFrame({ ...step("constant_velocity", { horizontalFrameContract: "future_contract" }),
  grid: { scalarOriginM: [0, 0, 0] } }).frame === "unverified", "unknown contracts remain unverified");

console.log(`\nframeMotion tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  console.log(failures.join("\n"));
  process.exit(1);
}
