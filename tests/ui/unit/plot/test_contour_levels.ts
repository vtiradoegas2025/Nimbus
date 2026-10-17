import assert from "node:assert/strict";
import { parseContourLevels } from "../../../../ui/src/lib/plot/contourLevels.ts";

assert.deepEqual(parseContourLevels("0, 1e-8, -0.02, 0").levels, [0, 1e-8, -0.02]);
assert.deepEqual(parseContourLevels("3:1:-1").levels, [3, 2, 1]);
assert.deepEqual(parseContourLevels("0:0.3:0.1").levels.map(value => Number(value.toPrecision(12))), [0, 0.1, 0.2, 0.3]);
assert.equal(parseContourLevels("1e-8:3e-8:1e-8").levels[0], 1e-8);
for (const input of ["1,,2", ",", "1,NaN", "1,Infinity", "1:2:0", "2:1:1", "1:2:1:5", "0:100:1", "1e20:1.0000000000000002e20:1", "1:2:"]) {
  const result = parseContourLevels(input);
  assert.equal(result.levels.length, 0, input);
  assert.ok(result.error, input);
}
assert.equal(parseContourLevels("0:11:1").levels.length, 12);
assert.equal(parseContourLevels("0:12:1").levels.length, 0);
assert.deepEqual(parseContourLevels(" "), { levels: [], error: null });
console.log("Contour levels preserve small scientific thresholds and reject malformed or truncated requests.");
