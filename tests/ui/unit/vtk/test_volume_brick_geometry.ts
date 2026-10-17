import assert from "node:assert/strict";
import { buildVolumeOrder, orderedVolumeIndices, volumeHaloSamples, volumeOwnedBounds, volumeOwnedScalarRows, volumeOpenUpperFaces, withVolumeBrickOwnership } from "../../../../ui/src/lib/vtk/volumeBrickGeometry.ts";
import { compactVolumeBrickPlan, planVolumeBricks, selectVolumeHierarchyPlan, type FieldBrickHierarchy, type VolumeBrickPlan } from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";
import type { PreparedVtkImage } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { scalarMultisetFingerprint } from "../../../../ui/src/lib/vtk/scalarCompositionParity.ts";
import shader from "@kitware/vtk.js/Rendering/OpenGL/glsl/vtkVolumeFS.glsl.js";
import vtkShaderProgram from "@kitware/vtk.js/Rendering/OpenGL/ShaderProgram.js";
import { configureCoherentVolumeSampling } from "../../../../ui/src/lib/vtk/coherentVolumeSampling.ts";
import { volumeShaderSampleCeiling } from "../../../../ui/src/lib/vtk/vtkPhysicalVolumeOpacity.ts";
import { partitionNestedVolume } from "../../../../ui/src/lib/vtk/nestedVolumeGeometry.ts";

let shaderProperties: any = {};
configureCoherentVolumeSampling({ getViewSpecificProperties: () => shaderProperties,
  setViewSpecificProperties: (value: object) => { shaderProperties = value; } }, true);
let replaced = shader;
for (const rule of shaderProperties.OpenGL.ShaderReplacements) {
  const next = vtkShaderProgram.substitute(replaced, rule.originalValue, rule.replacementValue, rule.replaceAll).result;
  assert.notEqual(next, replaced, "each sampling replacement must reach the installed VTK shader");
  replaced = next;
}
assert.equal(replaced.includes("isPointClipped(posIStoVC(offsetedPosIS))"), false);
assert.ok(replaced.includes("float intersection = -distance / direction"), "ownership still clips the ray interval");
for (const count of [1, 127, 128, 129, 511, 512, 513, 3000]) {
  assert.ok(volumeShaderSampleCeiling(count) >= count);
  assert.ok(volumeShaderSampleCeiling(count) <= Math.max(128, count * 2));
}
assert.throws(() => volumeShaderSampleCeiling(Infinity), /sample count/);

const hierarchy: FieldBrickHierarchy = {
  schema: "nimbus.field-brick-hierarchy", schemaVersion: 3, field: "linear",
  publicationGeneration: "one", sourceShape: [8, 4, 4], sourceIndexOffset: [0, 0, 0],
  sourceDtype: "float32", sampleLocation: "cell_center", extremaSemantics: "stored_values",
  extremaAbsolutePadding: 0, cullingSafe: true,
  levels: [{ level: 0, stride: [1, 1, 1], shape: [8, 4, 4], brickShape: [4, 4, 4], source: { kind: "primary_field" },
    bricks: [0, 1].map((x) => ({ id: String(x), index: [x, 0, 0], origin: [x * 4, 0, 0], extent: [4, 4, 4],
      scalarRange: { minimum: x * 4, maximum: x * 4 + 18, finiteCount: 64, nonFiniteCount: 0 },
      physicalBoundsM: { lower: [x * 4 - 0.5, -0.5, -0.5], upper: [x * 4 + 3.5, 3.5, 3.5] }, contentSha256: "0".repeat(64) })),
  }],
};
const plan = planVolumeBricks(hierarchy, [[0, 0.1], [100, 0.1]], { maximumResidentBytes: 1024, haloSamples: 2 });
assert.equal(plan.items.length, 2);
assert.equal(plan.residentBytes, 768);
const sparse = structuredClone(hierarchy);
sparse.levels[0].bricks[0].scalarRange = { minimum: 1, maximum: 1, finiteCount: 64, nonFiniteCount: 0 };
sparse.levels[0].bricks[1].scalarRange = { minimum: 0, maximum: 0, finiteCount: 64, nonFiniteCount: 0 };
const fringe = planVolumeBricks(sparse, [[0, 0], [1, 1]], { maximumResidentBytes: 1024, haloSamples: 2 });
assert.equal(fringe.boundaryOnlyBrickCount, 1);
assert.deepEqual(fringe.items.find((item) => item.boundaryOnly)?.ownedShape, [1, 4, 4]);
assert.ok(fringe.residentBytes < plan.residentBytes, "transparent interiors do not become full retained bricks");
assert.equal(compactVolumeBrickPlan(sparse, fringe, { maximumResidentBytes: 1024 }).items.length, 1);
for (const stride of [1, 2]) for (let pattern = 0; pattern < 4; pattern++) {
  const fixture = structuredClone(hierarchy);
  fixture.sourceShape = [16 * stride, 16 * stride, 12 * stride];
  fixture.sourceIndexOffset = [13, 7, 2];
  const level = fixture.levels[0];
  level.shape = [16, 16, 12];
  level.stride = [stride, stride, stride];
  level.bricks = [];
  for (let z = 0; z < 3; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
    const active = pattern === 0 ? z === 1 : pattern === 1 ? x === y
      : pattern === 2 ? (x + 3 * y + 7 * z) % 5 === 0 : x > 0 && x < 3 && y > 0 && y < 3;
    level.bricks.push({ ...hierarchy.levels[0].bricks[0], id: `${x}:${y}:${z}`,
      index: [x, y, z], origin: [x * 4, y * 4, z * 4], extent: [4, 4, 4],
      scalarRange: { minimum: Number(active), maximum: Number(active), finiteCount: 64, nonFiniteCount: 0 },
    });
  }
  const before = planVolumeBricks(fixture, [[0, 0], [1, 1]], { maximumResidentBytes: 1e7, haloSamples: 2 });
  const after = compactVolumeBrickPlan(fixture, before, { maximumResidentBytes: 1e7, targetShape: [8, 8, 8] });
  const ownership = (value: VolumeBrickPlan) => {
    const cells = new Uint8Array(16 * 16 * 12);
    const boxes: [number, number, number, number, number, number][] = [];
    for (const item of value.items) {
      const origin = item.window.origin.map((v, axis) => (v - fixture.sourceIndexOffset[axis]) / stride + item.ownedOffset![axis]);
      const shape = item.ownedShape!;
      boxes.push([origin[0], origin[0] + shape[0], origin[1], origin[1] + shape[1], origin[2], origin[2] + shape[2]]);
      for (let z = 0; z < shape[2]; z++) for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[0]; x++) {
        const index = origin[0] + x + 16 * (origin[1] + y + 16 * (origin[2] + z));
        assert.equal(cells[index]++, 0, "each scalar has exactly one optical owner");
      }
    }
    assert.equal(buildVolumeOrder(boxes).unresolved, false);
    return cells;
  };
  assert.deepEqual(ownership(after), ownership(before), "compaction preserves occupied cells and culled holes");
  assert.ok(after.residentBytes <= before.residentBytes, "merging does not increase halo storage");
  assert.ok(after.items.length <= before.items.length);
  if (pattern === 0) assert.ok(after.items.length < before.items.length);
}
const missing = structuredClone(hierarchy);
missing.levels[0].bricks[0].scalarRange = { finiteCount: 0, nonFiniteCount: 64, minimum: null, maximum: null };
assert.deepEqual(planVolumeBricks(missing, [[0, 0.1], [100, 0.1]], { maximumResidentBytes: 1024, haloSamples: 2 }).items.map((item) => item.brickId), ["1"]);
const prepared = plan.items.map((item) => {
  const shape = item.window.extent as [number, number, number];
  const origin = item.window.origin as [number, number, number];
  const values = new Float32Array(shape[0] * shape[1] * shape[2]);
  for (let z = 0; z < shape[2]; z++) for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[0]; x++) {
    values[x + shape[0] * (y + shape[1] * z)] = x + origin[0] + 2 * y + 3 * z;
  }
  return withVolumeBrickOwnership({ shape, origin, spacing: [1, 1, 1], values, sourceNonFiniteCount: 0,
    samplingProvenance: {} as PreparedVtkImage["samplingProvenance"] }, item);
});
const full = new Float32Array(128);
for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 8; x++) full[x + 8 * (y + 4 * z)] = x + 2 * y + 3 * z;
assert.equal(scalarMultisetFingerprint(prepared.flatMap(volumeOwnedScalarRows)), scalarMultisetFingerprint([full]));
assert.deepEqual(volumeOwnedBounds(prepared[0]), [-0.5, 3.5, -0.5, 3.5, -0.5, 3.5]);
assert.deepEqual(volumeOwnedBounds(prepared[1]), [3.5, 7.5, -0.5, 3.5, -0.5, 3.5]);
assert.deepEqual(volumeOpenUpperFaces(prepared[0]), [1, 0, 0]);
assert.deepEqual(volumeOpenUpperFaces(prepared[1]), [0, 0, 0]);
for (let x = 2; x < 6; x++) assert.equal(prepared[0].values[x], prepared[1].values[x - 2]);
const tree = buildVolumeOrder(prepared.map(volumeOwnedBounds));
assert.equal(tree.unresolved, false);
assert.deepEqual(orderedVolumeIndices(tree, [-10, 0, 0]), [1, 0]);
assert.deepEqual(orderedVolumeIndices(tree, [10, 0, 0]), [0, 1]);
assert.deepEqual(orderedVolumeIndices(tree, [10, 0, 0], [1, 0, 0]), [1, 0]);
assert.throws(() => selectVolumeHierarchyPlan(hierarchy, [[0, 0.1], [100, 0.1]], { maximumResidentBytes: 512, haloSamples: 2 }), /exceeds/);
assert.throws(() => planVolumeBricks(hierarchy, [], { maximumResidentBytes: 1024, haloSamples: -1 }), /halo/);
assert.throws(() => withVolumeBrickOwnership(prepared[0], { ...plan.items[0], ownedOffset: [100, 0, 0] }), /ownership/);
assert.throws(() => buildVolumeOrder([[0, NaN, 0, 1, 0, 1]]), /bounds/);
assert.equal(volumeHaloSamples({ drM: 1, dyM: 1, dzM: 1 }), 2);
assert.equal(volumeHaloSamples({ scalarCoordinatesM: [[0, 1, 3], [0, 1], [0, 1]] }), 0);
assert.equal(volumeHaloSamples({ terrainFollowing: true }), 0);
const rootBounds = { lower: [0, 0, 0], upper: [8, 8, 8] } as const;
const childBounds = { lower: [1.25, 2.25, 1.75], upper: [6.75, 7.75, 7.25] } as const;
const parentParts: PreparedVtkImage[] = [];
for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
  const parent: PreparedVtkImage = { ...prepared[0], volumeOwnership: undefined,
    shape: [4, 4, 4], origin: [x * 4 + 0.5, y * 4 + 0.5, z * 4 + 0.5], spacing: [1, 1, 1], values: new Float32Array(64).fill(7) };
  const pieces = partitionNestedVolume(parent, rootBounds, childBounds, rootBounds);
  assert.ok(pieces.every((piece) => piece.values === parent.values), "optical partition shares immutable scalar support");
  assert.ok(parent.values.every((value) => value === 7), "exclusion never writes artificial zeros");
  parentParts.push(...pieces);
}
const child: PreparedVtkImage = { ...prepared[0], volumeOwnership: undefined, shape: [11, 11, 11],
  origin: [1.5, 2.5, 2], spacing: [0.5, 0.5, 0.5], values: new Float32Array(1331).fill(9) };
const allParts = [...parentParts, ...partitionNestedVolume(child, childBounds, null, rootBounds)];
const boxes = allParts.map(volumeOwnedBounds);
assert.equal(buildVolumeOrder(boxes).unresolved, false, "parent and child form a separable optical partition");
assert.equal(boxes.reduce((sum, b) => sum + (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]), 0), 512);
for (const x of [0, 1.25 - 1e-6, 1.25, 4, 6.75, 6.75 + 1e-6, 8]) {
  for (const y of [0, 2.25, 4, 7.75, 8]) for (const z of [0, 1.75, 4, 7.25, 8]) {
    const point = [x, y, z];
    const owners = allParts.filter((part) => {
      const b = volumeOwnedBounds(part), open = volumeOpenUpperFaces(part);
      return point.every((value, axis) => value >= b[2 * axis]
        && (open[axis] ? value < b[2 * axis + 1] : value <= b[2 * axis + 1]));
    });
    assert.equal(owners.length, 1, "faces, edges, corners and interiors have one owner");
  }
}
assert.throws(() => partitionNestedVolume(child, { lower: [0, 0, 0], upper: [NaN, 1, 1] }, null, rootBounds), /bounds/);
console.log("Volume brick ownership, halo admission, and camera ordering checks passed");
