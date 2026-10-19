import assert from "node:assert/strict";
import {
  finestLevelForFields,
  resolveOutputHierarchyFrame,
  selectOutputHierarchyLevel,
} from "../../../../ui/src/lib/run/outputHierarchy.ts";
import type { GridInfo, NestedHistoryLevel, StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let assertions = 0;
function check(condition: unknown, message: string): asserts condition {
  assertions += 1;
  assert.ok(condition, message);
}

function grid(spacing: number, lower: [number, number, number], upper: [number, number, number]): GridInfo {
  return {
    nr: Math.round((upper[0] - lower[0]) / spacing) + 2,
    nth: Math.round((upper[1] - lower[1]) / spacing) + 2,
    nz: 50,
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    staggering: "cell_centered",
    drM: spacing,
    dyM: spacing,
    dzM: 320,
    scalarOriginM: [lower[0] - spacing / 2, lower[1] - spacing / 2, 160],
    physicalBoundsM: { lower, upper },
  };
}

function step(index: number, time: number, mesh: GridInfo, fields = ["qc", "u", "v", "w"]): StepInfo {
  return {
    index,
    simulationTimeS: time,
    fields,
    grid: mesh,
    fieldUnits: {},
    fieldOrigins: Object.fromEntries(fields.map((field) => [field, "native" as const])),
    publicationState: "complete",
    publicationGeneration: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    motion: { mode: "moving_nest", earthOriginM: [0, 0], windsEarthRelative: true },
  };
}

const rootGrid = grid(1_200, [1_200, 1_200, 0], [160_800, 128_400, 16_000]);
const childGridA = grid(600, [3_600, 2_400, 0], [97_200, 102_000, 16_000]);
const childGridB = grid(600, [15_600, 8_400, 0], [109_200, 108_000, 16_000]);
const root1 = step(1, 450, rootGrid);
const root2 = step(2, 900, rootGrid);
const child: NestedHistoryLevel = {
  levelId: "child_0000",
  parentLevelId: "root",
  depth: 1,
  runDir: "/run/nests/child_0000",
  steps: [step(1, 450, childGridA), step(2, 900, childGridB, ["qc", "u", "v", "w", "vorticity_z"])],
};

const first = resolveOutputHierarchyFrame("/run", root1, [child]);
check(first.ok, "valid child hierarchy should resolve");
check(first.levels.length === 2, "root and child should both remain in the hierarchy");
check(first.finest.levelId === "child_0000", "child should own finest resolved data");
check(first.finest.step.grid?.physicalBoundsM?.lower[0] === 3_600, "first placement must use its own Earth bounds");
check(first.disclosure.includes("600 m"), "disclosure should name native child spacing");

check(selectOutputHierarchyLevel(first, { focusPointM: [150_000, 120_000] }).level.levelId === "root",
  "an explicit focus outside a sole refined child uses root coverage");
check(selectOutputHierarchyLevel(first).level.levelId === "child_0000",
  "a sole child remains the default when no focus is supplied");
check(selectOutputHierarchyLevel(first, { requestedLevelId: "child_0000", focusPointM: [150_000, 120_000] }).level.levelId === "child_0000",
  "an explicit domain choice is respected independently of camera focus");

const relocated = resolveOutputHierarchyFrame("/run", root2, [child]);
check(relocated.ok, "relocated child should remain valid");
check(relocated.finest.step.grid?.physicalBoundsM?.lower[0] === 15_600, "frame selection must follow relocation, not reuse prior bounds");
check(finestLevelForFields(relocated, ["vorticity_z"]).levelId === "child_0000", "fine-only fields should route to the child");
check(finestLevelForFields(first, ["vorticity_z"]).levelId === "root", "missing child fields must fall back instead of being invented");

const staleChild: NestedHistoryLevel = {
  ...child,
  steps: [{ ...child.steps[0], publicationGeneration: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }],
};
const stale = resolveOutputHierarchyFrame("/run", root1, [staleChild]);
check(stale.ok && stale.finest.levelId === "root", "stale child generation must never enter the scene");

const outsideChild: NestedHistoryLevel = {
  ...child,
  steps: [step(1, 450, grid(600, [100_000, 2_400, 0], [193_600, 102_000, 16_000]))],
};
const outside = resolveOutputHierarchyFrame("/run", root1, [outsideChild]);
check(!outside.ok, "child outside parent Earth bounds must fail closed");

const coarseChild: NestedHistoryLevel = {
  ...child,
  steps: [step(1, 450, grid(2_400, [3_600, 2_400, 0], [97_200, 102_000, 16_000]))],
};
const coarse = resolveOutputHierarchyFrame("/run", root1, [coarseChild]);
check(!coarse.ok, "a child that is not finer must fail closed");

const sibling: NestedHistoryLevel = {
  ...child,
  levelId: "child_0001",
  runDir: "/run/nests/child_0001",
  steps: [step(1, 450, grid(400, [80_000, 20_000, 0], [140_000, 90_000, 16_000]))],
};
const siblings = resolveOutputHierarchyFrame("/run", root1, [child, sibling]);
check(siblings.ok, "multiple explicit sibling branches should resolve");
check(siblings.finest.levelId === "root", "ambiguous siblings must not silently pick one without focus");
check(
  selectOutputHierarchyLevel(siblings, { focusPointM: [100_000, 50_000] }).level.levelId === "child_0001",
  "auto selection should follow the finest sibling containing the focus point",
);
check(
  selectOutputHierarchyLevel(siblings, { focusPointM: [90_000, 50_000] }).level.levelId === "child_0001",
  "overlapping siblings should deterministically give focus ownership to the finer lattice",
);
check(
  selectOutputHierarchyLevel(siblings, { requestedLevelId: "child_0000" }).level.levelId === "child_0000",
  "explicit sibling selection should remain stable",
);

const grandchild: NestedHistoryLevel = {
  levelId: "child_0000/child_0002",
  parentLevelId: "child_0000",
  depth: 2,
  runDir: "/run/nests/child_0000/nests/child_0002",
  steps: [step(1, 450, grid(100, [20_000, 20_000, 0], [60_000, 60_000, 16_000]), ["qc", "w"])],
};
const recursive = resolveOutputHierarchyFrame("/run", root1, [child, grandchild]);
check(recursive.ok && recursive.finest.levelId === grandchild.levelId, "recursive explicit parent graph should select its deepest leaf");
check(
  selectOutputHierarchyLevel(recursive, { requiredFields: ["u", "v", "w"] }).level.levelId === "child_0000",
  "missing grandchild fields should fall back only to its declared parent branch",
);

const fineSibling: NestedHistoryLevel = { ...sibling, levelId: "child_0004", runDir: "/run/nests/child_0004",
  steps: [step(1, 450, grid(50, [20_000, 20_000, 0], [60_000, 60_000, 16_000]), ["qc", "w"])] };
const compatibleSibling: NestedHistoryLevel = { ...sibling, levelId: "child_0005", runDir: "/run/nests/child_0005",
  steps: [step(1, 450, grid(200, [20_000, 20_000, 0], [60_000, 60_000, 16_000]))] };
const overlapping = resolveOutputHierarchyFrame("/run", root1, [child, grandchild, fineSibling, compatibleSibling]);
check(overlapping.ok, "overlapping branches with unequal depth and spacing resolve");
check(selectOutputHierarchyLevel(overlapping, { focusPointM: [40_000, 40_000] }).level.levelId === "child_0004",
  "auto focus chooses actual finer spacing rather than tree depth");
check(selectOutputHierarchyLevel(overlapping, { focusPointM: [40_000, 40_000], requiredFields: ["u", "v", "w"] }).level.levelId === "child_0005",
  "auto focus chooses the finest compatible branch before falling back to a coarser parent");

const orphan = resolveOutputHierarchyFrame("/run", root1, [{ ...grandchild, parentLevelId: "child_9999" }]);
check(!orphan.ok, "orphaned recursive levels must fail closed");
const falsePath = resolveOutputHierarchyFrame("/run", root1, [{
  ...grandchild,
  levelId: "child_0002",
}]);
check(!falsePath.ok, "a level identity whose path disagrees with its depth/parent must fail closed");

const noChildFrame = resolveOutputHierarchyFrame("/run", step(0, 0, rootGrid), [child]);
check(noChildFrame.ok && noChildFrame.levels.length === 1, "root-only initialization must remain plottable");

const legacyRoot = resolveOutputHierarchyFrame("/run", { ...root1, publicationGeneration: null }, [child]);
check(!legacyRoot.ok, "nested data may not attach to a legacy root identity");

console.log(`output hierarchy: ${assertions} assertions passed`);
