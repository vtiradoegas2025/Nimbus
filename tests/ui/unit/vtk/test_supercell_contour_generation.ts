import {
  reduceSupercellContourGeneration,
  type SupercellContourGenerationState,
} from "../../../../ui/src/lib/vtk/useSupercellWorkerOwnedContour.ts";
import type { VortexContourResult } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}
const result = {
  core: null, opposite: null, coreComponentCount: 0, oppositeComponentCount: 0,
  mesocyclone: null, mesocycloneAssessment: null, vortexAssessment: null,
  condensation: null, condensationBounds: null, condensationAssessment: null,
  usage: { triangles: 0, bytes: 0 }, knownScratchBytes: 0, elapsedMs: 1,
} satisfies VortexContourResult;
const empty: SupercellContourGenerationState = {
  generation: null, loading: false, result: null, error: null,
};
const first = reduceSupercellContourGeneration(empty, { type: "begin", generation: "a" });
check(first.loading && first.generation === "a", "begin owns one immutable generation");
check(
  reduceSupercellContourGeneration(first, { type: "resolve", generation: "old", result }) === first,
  "stale result cannot replace the active generation",
);
const resolved = reduceSupercellContourGeneration(first, { type: "resolve", generation: "a", result });
check(!resolved.loading && resolved.result === result, "matching result commits atomically");
const second = reduceSupercellContourGeneration(resolved, { type: "begin", generation: "b" });
check(second.loading && second.result == null, "replacement generation clears old geometry");
const rejected = reduceSupercellContourGeneration(second, { type: "reject", generation: "b", error: "failed" });
check(!rejected.loading && rejected.error === "failed", "matching failure settles the generation");
const cleared = reduceSupercellContourGeneration(rejected, { type: "clear" });
check(cleared.generation == null && !cleared.loading, "clear releases generation ownership");

console.log(`\nsupercellContourGeneration tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
