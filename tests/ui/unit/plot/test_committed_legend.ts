import assert from "node:assert/strict";
import { committedLegend } from "../../../../ui/src/components/plots/useCommittedLegend.ts";

const initial = { title: "w (m/s)", min: -12, max: 12, colors: ["blue", "red"] };
const current = committedLegend(null, "run:w", true, initial);
assert.equal(committedLegend(current, "run:w", false, null), current);
assert.equal(committedLegend(current, "run:w", false, { ...initial, max: 99 }), current);
assert.equal(committedLegend(current, "other:w", false, initial), null);
assert.equal(committedLegend(current, "run:theta", false, initial), null);
assert.equal(committedLegend(current, "run:w", true, null)?.value, null);
assert.equal(committedLegend(current, "run:w", true, { ...initial, max: 99 })?.value?.max, 99);
assert.equal(initial.max, 12);
console.log("Committed legends retain displayed values during staging, retire incompatible scopes, and clear only on completed empty scenes.");
