import assert from "node:assert/strict";
import { scalarVolumeBatchStats, scalarVolumeStats } from "../../../../ui/src/lib/vtk/scalarVolumeStats.ts";
import { resolveVolumeTransferPolicy, volumeOpacityPoints } from "../../../../ui/src/lib/plot/volumeTransferPolicy.ts";

const developed = Float32Array.from({ length: 10_000 }, (_, index) => index % 2 ? 4 : -4);
developed[5000] = 238;
developed[5001] = -77;
const before = developed.slice();
const developedStats = scalarVolumeStats(developed);
assert.equal(developedStats.max, 238);
assert.equal(developedStats.min, -77);
assert.equal(developedStats.positiveP98, 4);
assert.equal(developedStats.negativeP02, -4);
assert.deepEqual(developed, before);
for (const transform of ["none", "horizontal-level-anomaly"] as const) {
  const policy = resolveVolumeTransferPolicy("theta_prime", "Cool to Warm", developedStats, transform, true);
  assert.equal(policy.max, 4, "isolated extremes must not hide the resolved signed population");
  assert.equal(policy.min, -4);
}

for (const sign of [-1, 1]) {
  const sparse = Float32Array.from({ length: 10_000 }, (_, index) =>
    index < 100 ? sign * 2 : -sign * (2 / 99));
  const stats = scalarVolumeStats(sparse);
  const policy = resolveVolumeTransferPolicy("theta_prime", "Cool to Warm", stats, "horizontal-level-anomaly");
  assert.equal(policy.max, 0.5, "sign-conditioned statistics retain a compact startup core below the whole-domain tail");
  assert.equal(policy.min, -0.5);
  const opacity = volumeOpacityPoints(policy, 1);
  assert.ok(opacity[3].scalar < -2 / 99 && opacity[4].scalar > 2 / 99,
    "the compensating horizontal background stays transparent for either sign");
}

const masked = scalarVolumeStats(Float32Array.of(0, 1, -2, NaN, Infinity, 999), Uint8Array.of(1, 1, 1, 1, 1, 0));
assert.equal(masked.min, -2);
assert.equal(masked.max, 1);
assert.equal(masked.positiveP98, 1);
assert.equal(masked.negativeP02, -2);
const empty = scalarVolumeStats(Float32Array.of(NaN, Infinity));
assert.equal(empty.min, 0);
assert.equal(empty.max, 0);
assert.equal(empty.positiveP98, 0);
assert.equal(empty.negativeP02, 0);
assert.ok(Number.isNaN(empty.p2) && Number.isNaN(empty.p98));
assert.throws(() => scalarVolumeStats(Float32Array.of(1, 2), Uint8Array.of(1)), /validity length/);
assert.deepEqual(scalarVolumeStats(Float32Array.of(-3, -1, 0, 2, 4)),
  scalarVolumeBatchStats([{ values: Float32Array.of(-3, -1) }, { values: Float32Array.of(0, 2, 4) }]));

const rare = new Float32Array(800_000);
rare[1] = 2;
rare[2] = -3;
const rareStats = scalarVolumeStats(rare);
assert.equal(rareStats.positiveP98, 2, "an unsampled positive sign keeps its full resolved support");
assert.equal(rareStats.negativeP02, -3, "an unsampled negative sign keeps its full resolved support");

console.log("Scalar volume statistics: finite support, signed outlier resistance, sparse startup cores and batch parity passed");
