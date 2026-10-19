import assert from "node:assert/strict";
import { canonicalHistoryUnit, historyFrameKey, historySourcesAreAppendOnly, historyUnitMismatch,
  newHistoryReductionCache, reconcileHistoryReductionCache } from "../../../../ui/src/lib/plot/historyReductionCache.ts";

const frame = (index: number, generation = `source-${index}`, fields = ["w"]) => ({ index, publicationGeneration: generation, fields });
const cache = newHistoryReductionCache<number>("run:w:max");
const keys = [0, 1, 2].map(index => historyFrameKey(frame(index), "w"));
reconcileHistoryReductionCache(cache, keys);
keys.forEach((key, index) => cache.values.set(key, index + 10));
reconcileHistoryReductionCache(cache, [...keys, historyFrameKey(frame(3), "w")]);
assert.deepEqual([...cache.values.values()], [10, 11, 12], "append reuses previously reduced immutable frames");
reconcileHistoryReductionCache(cache, keys.slice(1));
assert.deepEqual([...cache.values.values()], [11, 12], "rolling histories retain overlapping reductions without the first frame owning the cache");
const rewritten = historyFrameKey(frame(1, "source-1-rewrite"), "w");
reconcileHistoryReductionCache(cache, [rewritten, keys[2]]);
assert.equal(cache.values.has(keys[1]), false, "an earlier publication cannot contribute after source replacement");
assert.equal(cache.values.get(keys[2]), 12, "unmodified neighbors stay cached after one source rewrite");
assert.notEqual(historyFrameKey(frame(1, "same", []), "w"), historyFrameKey(frame(1, "same"), "w"), "field availability participates in identity");
cache.pending.set(keys[0], Promise.resolve(10));
reconcileHistoryReductionCache(cache, [keys[2]]);
assert.equal(cache.pending.has(keys[0]), false);
assert.equal(cache.activeKeys.has(keys[0]), false, "late results from removed sources cannot re-enter the active reduction cache");
assert.equal(historySourcesAreAppendOnly(keys, new Set([...keys, "next"])), true);
assert.equal(historySourcesAreAppendOnly(keys, new Set(keys.slice(1))), false, "max/min accumulation cannot subtract a removed source without rebuilding");
assert.equal(historySourcesAreAppendOnly(keys, new Set([keys[0], rewritten, keys[2]])), false);

for (const spelling of ["m/s", "m s-1", "m s^-1", "m s**-1", "m·s⁻¹"]) assert.equal(canonicalHistoryUnit(spelling), "m/s");
assert.equal(canonicalHistoryUnit("Kelvin"), "K");
assert.equal(canonicalHistoryUnit("mPa"), "mPa");
assert.equal(canonicalHistoryUnit("MPa"), "MPa", "SI prefixes with different magnitude must retain case");
assert.equal(historyUnitMismatch([{ fields: ["w"], fieldUnits: { w: "m/s" } }, { fields: ["w"], fieldUnits: { w: "m s-1" } }], "w"), null);
assert.ok(historyUnitMismatch([{ fields: ["w"], fieldUnits: { w: "m/s" } }, { fields: ["w"], fieldUnits: { w: "knots" } }], "w"));
assert.ok(historyUnitMismatch([{ fields: ["p"], fieldUnits: { p: "Pa" } }, { fields: ["p"], fieldUnits: { p: "hPa" } }], "p"));
assert.ok(historyUnitMismatch([{ fields: ["w"], fieldUnits: {} }, { fields: ["w"], fieldUnits: { w: "m/s" } }], "w"));
assert.equal(historyUnitMismatch([{ fields: [], fieldUnits: { w: "knots" } }, { fields: ["w"], fieldUnits: { w: "m/s" } }], "w"), null);
console.log("History reductions: append/rolling reuse, source replacement, availability, late-result pruning and unit identity passed");
