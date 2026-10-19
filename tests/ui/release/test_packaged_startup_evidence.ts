import assert from "node:assert/strict";
import { STARTUP_CHECKS, validatePackagedStartupEvidence } from "./validate_packaged_startup_evidence.ts";

const expected = { mode: "release", targetOs: "macos", targetArch: "x86_64" } as const;
const base = {
  schema: "nimbus.studio.packaged-startup-check", schemaVersion: 1, profile: "startup",
  scope: "build-and-open", status: "passed", graphicsValidation: "pending", scienceQualification: "pending",
  elapsedMs: 10, packagedRuntime: { buildMode: "release", targetOs: "macos", targetArch: "x86_64", appVersion: "0.1.0" },
  checks: STARTUP_CHECKS.map(name => ({ name: String(name), passed: true })),
};
assert.equal(validatePackagedStartupEvidence(base, expected).scope, "build-and-open");
for (const mutate of [
  (value: typeof base) => { value.status = "failed"; },
  (value: typeof base) => { value.graphicsValidation = "passed"; },
  (value: typeof base) => { value.scienceQualification = "fully-qualified"; },
  (value: typeof base) => { value.scope = "bounded-packaged-lifecycle"; },
  (value: typeof base) => { value.profile = "production"; },
  (value: typeof base) => { value.elapsedMs = NaN; },
  (value: typeof base) => { value.packagedRuntime.targetArch = "aarch64"; },
  (value: typeof base) => { value.packagedRuntime.buildMode = "debug"; },
  (value: typeof base) => { value.checks.pop(); },
  (value: typeof base) => { value.checks[0] = value.checks[1]; },
  (value: typeof base) => { value.checks[0].passed = false; },
]) {
  const value = structuredClone(base);
  mutate(value);
  assert.throws(() => validatePackagedStartupEvidence(value, expected));
}
console.log("Packaged startup evidence: 12 checks passed; graphics/science promotion rejected.");
