import {
  buildIncrementalComparisonSeries,
  comparisonSeriesMatchKey,
  OrderedComparisonSeriesCache,
} from "../../../../ui/src/lib/comparison/comparisonSeriesCache.ts";
import {
  finishPairMetricMoments,
  type PairMetricMoments,
  type PairMetrics,
} from "../../../../ui/src/lib/comparison/comparisonMetrics.ts";

let passed = 0;
const failures: string[] = [];

function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

function equal(actual: unknown, expected: unknown, label: string): void {
  check(JSON.stringify(actual) === JSON.stringify(expected),
    `${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

interface TestMatch {
  index: number;
  generation: string;
  difference: number;
  definition?: string;
  unit?: string;
}

interface CachedValue {
  moments: PairMetricMoments;
  metrics: PairMetrics;
}

function request(match: TestMatch) {
  return {
    key: comparisonSeriesMatchKey({
      left: {
        datasetIdentity: "dataset-a",
        runDir: "/run/a",
        publicationGeneration: match.generation,
        stepIndex: match.index,
        fieldDefinitions: [match.definition ?? "native-theta"],
        fieldUnits: [match.unit ?? "K"],
      },
      right: {
        datasetIdentity: "dataset-b",
        runDir: "/run/b",
        publicationGeneration: match.generation,
        stepIndex: match.index,
        fieldDefinitions: ["native-theta"],
        fieldUnits: ["K"],
      },
      simulationTimeS: match.index * 10,
      analysis: {
        kind: "scalar",
        field: "theta",
        rank: 3,
        leftGeometry: { shape: [8, 6, 4], windows: [{ origin: [0, 0, 0], extent: [8, 6, 4] }] },
        rightGeometry: { shape: [8, 6, 4], windows: [{ origin: [0, 0, 0], extent: [8, 6, 4] }] },
      },
    }),
    request: match,
  };
}

function calculate(match: TestMatch): CachedValue {
  const moments: PairMetricMoments = {
    count: 2,
    weightSum: 3,
    sum: 3 * match.difference,
    absoluteSum: 3 * Math.abs(match.difference),
    squareSum: 3 * match.difference * match.difference,
    maxAbs: Math.abs(match.difference),
  };
  return { moments, metrics: finishPairMetricMoments(moments) };
}

const cache = new OrderedComparisonSeriesCache<CachedValue>();
let reads = 0;
const load = async (match: TestMatch) => {
  reads += 1;
  return calculate(match);
};
const original = [
  { index: 0, generation: "generation-a", difference: 1 },
  { index: 1, generation: "generation-a", difference: -2 },
  { index: 2, generation: "generation-a", difference: 3 },
];
const first = await buildIncrementalComparisonSeries({ cache, requests: original.map(request), load });
equal(reads, 3, "initial series reads each match once");
equal(first, original.map(calculate), "cached series preserves exact moments and finished metrics");

const appended = [...original, { index: 3, generation: "generation-a", difference: -4 }];
equal(cache.matchingPrefix(appended.map(match => request(match).key)), first,
  "an appended request may display only its still-valid completed prefix synchronously");
reads = 0;
const second = await buildIncrementalComparisonSeries({ cache, requests: appended.map(request), load });
equal(reads, 1, "appending one match performs O(new matches) reads");
equal(second, appended.map(calculate), "incremental append is numerically identical to a full rebuild");

const changed = [
  appended[0],
  { index: 1, generation: "generation-b", difference: 20 },
  appended[2],
  appended[3],
];
reads = 0;
equal(cache.matchingPrefix(changed.map(match => request(match).key)), [first[0]],
  "a revised publication cannot relabel obsolete downstream metrics as current");
const third = await buildIncrementalComparisonSeries({ cache, requests: changed.map(request), load });
equal(reads, 3, "a changed match preserves only the exact preceding prefix");
equal(third, changed.map(calculate), "downstream rebuild after a changed match is numerically exact");

const redefined = changed.map((match, index) => index === 1
  ? { ...match, definition: "expression:theta-times-two:v2", difference: 40 } : match);
reads = 0;
equal(cache.matchingPrefix(redefined.map(match => request(match).key)), [third[0]],
  "editing a same-frame derived definition retains only genuinely unchanged metrics");
const fourth = await buildIncrementalComparisonSeries({ cache, requests: redefined.map(request), load });
equal(reads, 3, "a definition revision recomputes its dependent metric suffix");
equal(fourth, redefined.map(calculate), "revised derived metrics equal a fresh full-grid calculation");
equal(cache.matchingPrefix(["different-field-and-units"]), [], "a new analysis cannot borrow old rows under a new caption");
check(request({ ...original[0], unit: "m/s" }).key !== request(original[0]).key,
  "a declared unit revision changes scientific cache ownership");
const leftIdentity = { datasetIdentity: "a", runDir: "/a", publicationGeneration: "g", stepIndex: 0,
  fieldDefinitions: ["u:v1", "v:v1"], fieldUnits: ["m/s", "m/s"] };
const vectorKey = (definition: string) => comparisonSeriesMatchKey({
  left: leftIdentity, right: { ...leftIdentity, runDir: "/b", fieldDefinitions: ["u:v1", definition] }, simulationTimeS: 0,
  analysis: { kind: "vector", components: ["u", "v"], region: { lower: [0, 0, 0], upper: [1, 1, 1] },
    leftGeometry: {}, rightGeometry: {} },
});
check(vectorKey("v:v1") !== vectorKey("v:v2"), "changing either right vector component definition invalidates its metric rows");

const identityBase = request(original[0]).key;
const vectorIdentity = comparisonSeriesMatchKey({
  left: { datasetIdentity: "dataset-a", runDir: "/run/a", publicationGeneration: "generation-a", stepIndex: 0 },
  right: { datasetIdentity: "dataset-b", runDir: "/run/b", publicationGeneration: "generation-a", stepIndex: 0 },
  simulationTimeS: 0,
  analysis: {
    kind: "vector",
    components: ["u", "v"],
    region: { lower: [0, 0, 0], upper: [7, 5, 3] },
    leftGeometry: { shape: [8, 6, 4] },
    rightGeometry: { shape: [8, 6, 4] },
  },
});
check(identityBase !== vectorIdentity, "field/component, region, and geometry definitions are part of match identity");
check(request({ ...original[0], generation: "generation-b" }).key !== identityBase,
  "both frame publication generations are part of match identity");

const generationCache = new OrderedComparisonSeriesCache<string>();
let releaseOld!: (value: string) => void;
const pendingOld = new Promise<string>((resolve) => { releaseOld = resolve; });
const oldRequest = { key: "generation-a:0", request: "old" };
const oldRun = buildIncrementalComparisonSeries({
  cache: generationCache,
  requests: [oldRequest],
  load: () => pendingOld,
});
const newRequest = { key: "generation-b:0", request: "new" };
const newRun = await buildIncrementalComparisonSeries({
  cache: generationCache,
  requests: [newRequest],
  load: async () => "new-value",
});
equal(newRun, ["new-value"], "new publication generation completes independently of a stale read");
releaseOld("old-value");
let staleAborted = false;
try {
  await oldRun;
} catch (error) {
  staleAborted = error instanceof DOMException && error.name === "AbortError";
}
check(staleAborted, "a stale generation cannot commit after a newer cache run begins");
let generationRereads = 0;
const retainedNewRun = await buildIncrementalComparisonSeries({
  cache: generationCache,
  requests: [newRequest],
  load: async () => { generationRereads += 1; return "wrong"; },
});
equal(generationRereads, 0, "stale completion does not evict or overwrite the active generation");
equal(retainedNewRun, ["new-value"], "active generation remains exact after stale cancellation");

console.log(`\ncomparisonSeriesCache tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`  ${failure}`);
  process.exit(1);
}
