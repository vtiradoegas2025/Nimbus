import { planVolumeBricks, type FieldBrickHierarchy } from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";
import { NestedVolumeCommitBarrier, planNestedVolumeGeneration } from "../../../../ui/src/lib/vtk/nestedVolumeCoordinator.ts";

const hierarchy = (field: string): FieldBrickHierarchy => ({
  schema: "nimbus.field-brick-hierarchy", schemaVersion: 1, field,
  publicationGeneration: "g", sourceShape: [2, 2, 2], sourceIndexOffset: [0, 0, 0],
  sourceDtype: "float32", sampleLocation: "cell_center", extremaSemantics: "stored_values",
  extremaAbsolutePadding: 0, cullingSafe: true,
  levels: [{ level: 0, stride: [1, 1, 1], shape: [2, 2, 2], brickShape: [2, 2, 2],
    source: { kind: "primary_field" }, bricks: [{ id: "l0:0:0:0", index: [0, 0, 0],
      origin: [0, 0, 0], extent: [2, 2, 2], scalarRange: { minimum: 0, maximum: 1, finiteCount: 8, nonFiniteCount: 0 },
      physicalBoundsM: { lower: [0, 0, 0], upper: [2, 2, 2] }, contentSha256: "a".repeat(64) }] }],
});
const parent = planVolumeBricks(hierarchy("temperature"), [[0, 1], [1, 1]], { maximumResidentBytes: 1024 });
const child = planVolumeBricks(hierarchy("temperature"), [[0, 1], [1, 1]], { maximumResidentBytes: 1024 });
const plan = planNestedVolumeGeneration("frame-a", [
  { levelId: "root", plan: parent }, { levelId: "child_0000", plan: child },
]);
if (plan.claims.length !== 6) throw new Error("Aggregate admission does not cover both layers before reads");
const barrier = new NestedVolumeCommitBarrier<string>();
barrier.reset(plan);
if (barrier.publish("frame-a", "root", "parent") !== null) throw new Error("Parent committed without child");
const complete = barrier.publish("frame-a", "child_0000", "child");
if (complete?.join(",") !== "parent,child") throw new Error("Complete nested generation did not commit atomically");
barrier.reset({ ...plan, identity: "frame-b" });
if (barrier.publish("frame-a", "child_0000", "stale") !== null) throw new Error("Superseded child committed");
console.log("\nnested volume coordinator tests: 3 passed, 0 failed.");
