import assert from "node:assert/strict";
import { qualifyPackagedConfigPreflight } from "../../../../ui/src/lib/config/packagedConfigPreflightQualification.ts";
import type { ConfigPreflightRequest, ConfigPreflightReport } from "../../../../ui/src/lib/config/configPreflight.ts";

const hash = "a".repeat(64);
const request = { source: { kind: "exact_document" } } as ConfigPreflightRequest;
const batch = { schema: "nimbus.studio.packaged-config-qualification.v1", binarySha256: hash,
  cases: [{ name: "accepted", expectedAcceptance: true, request },
          { name: "rejected", expectedAcceptance: false, request }] };
const report = (accepted: boolean) => ({ compatible: accepted,
  nativeAdmission: { state: accepted ? "accepted" : "rejected", validatedBinarySha256: `sha256:${hash}` },
  executable: { binaryFingerprint: `sha256:${hash}`, catalogFingerprint: "catalog", canonicalPath: "/bundle/nimbus_sim" },
} as ConfigPreflightReport);
let calls = 0;
const result = await qualifyPackagedConfigPreflight(batch, async () => report(calls++ === 0));
assert.equal(result.checks.length, 2);
assert.equal(result.canonicalPath, "/bundle/nimbus_sim");
assert.equal(calls, 2);
for (const bad of [null, {}, { ...batch, cases: [] }, { ...batch, binarySha256: "bad" },
  { ...batch, cases: Array(129).fill(batch.cases[0]) },
  { ...batch, cases: [batch.cases[0], batch.cases[0]] },
  { ...batch, cases: [{ ...batch.cases[0], request: { source: { kind: "descriptor_only" } } }] }]) {
  await assert.rejects(() => qualifyPackagedConfigPreflight(bad, async () => {
    assert.fail("Malformed batch must fail before native invocation");
  }));
}
for (const mutate of [
  (r: ConfigPreflightReport) => { r.compatible = false; },
  (r: ConfigPreflightReport) => { r.nativeAdmission.state = "not_checked"; },
  (r: ConfigPreflightReport) => { r.nativeAdmission.validatedBinarySha256 = "b".repeat(64); },
  (r: ConfigPreflightReport) => { r.executable.binaryFingerprint = "b".repeat(64); },
]) {
  await assert.rejects(() => qualifyPackagedConfigPreflight(batch, async () => {
    const r = report(true); mutate(r); return r;
  }), /agreement failed/);
}
for (const key of ["canonicalPath", "catalogFingerprint"] as const) {
  calls = 0;
  await assert.rejects(() => qualifyPackagedConfigPreflight(batch, async () => {
    const r = report(calls++ === 0); if (calls > 1) r.executable[key] = "changed"; return r;
  }), /agreement failed/);
}
await assert.rejects(() => qualifyPackagedConfigPreflight(batch, async () => { throw new Error("native process failed"); }), /native process failed/);
console.log("packaged configuration preflight: acceptance, rejection, identity drift and malformed input passed");
