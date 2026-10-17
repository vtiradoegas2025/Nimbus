import assert from "node:assert/strict";
import { finiteHodographLine, finiteProfileLine } from "../../../../ui/src/lib/plot/finiteProfileLine.ts";

const source = Float32Array.of(1, NaN, -2, Infinity, -Infinity, 3);
assert.deepEqual(finiteProfileLine(source), [1, null, -2, null, null, 3]);
assert.equal(source.length, 6, "invalid levels remain gaps rather than shortening the profile");
const wind = finiteHodographLine(Float32Array.of(2, Infinity, 4, NaN), Float32Array.of(-3, 5, NaN, 10000));
assert.deepEqual(wind, { x: [2, null, null, null], y: [-3, null, null, null], magnitude: 3 });
assert.equal(finiteHodographLine(Float32Array.of(NaN), Float32Array.of(Infinity)).magnitude, 1);
assert.throws(() => finiteHodographLine(Float32Array.of(1), Float32Array.of(1, 2)), /share one profile/);
console.log("Profile gaps, paired wind support, finite chart bounds and invalid profile lengths passed");
