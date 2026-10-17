import assert from "node:assert/strict";
import { planVolumeBricks, type FieldBrickHierarchy } from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";
import { reconstructVolumePlan } from "../../../../ui/src/lib/vtk/volumeReconstruction.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { guardVtkContextRendering, protectVtkTraversal } from "../../../../ui/src/lib/vtk/vtkContextRenderGuard.ts";
import { volumeOwnedBounds, withVolumeBrickOwnership } from "../../../../ui/src/lib/vtk/volumeBrickGeometry.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";

const coordinates: [number[], number[], number[]] = [[0, 1, 2, 4, 8, 12, 17, 23], [0, 1, 2, 3], [0, 1, 2, 3]];
const grid = { nr: 8, nth: 4, nz: 4, coordinateSystem: "cartesian", drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0, 0, 0] as [number, number, number], scalarCoordinatesM: coordinates };
const hierarchy: FieldBrickHierarchy = { schema: "nimbus.field-brick-hierarchy", schemaVersion: 3, field: "scalar", publicationGeneration: "test", sourceShape: [8, 4, 4], sourceIndexOffset: [0, 0, 0], sourceDtype: "float32", sampleLocation: "cell_center", extremaSemantics: "stored_values", extremaAbsolutePadding: 0, cullingSafe: true,
  levels: [{ level: 0, stride: [1, 1, 1], shape: [8, 4, 4], brickShape: [4, 4, 4], source: { kind: "primary_field" },
    bricks: [0, 1].map((x) => ({ id: `brick-${x}`, index: [x, 0, 0], origin: [x * 4, 0, 0], extent: [4, 4, 4], scalarRange: { minimum: -3, maximum: 75, finiteCount: 64, nonFiniteCount: 0 }, physicalBoundsM: { lower: [x * 4, 0, 0], upper: [x * 4 + 4, 4, 4] }, contentSha256: "0".repeat(64) })) }],
};
const plan = reconstructVolumePlan(hierarchy, grid, planVolumeBricks(hierarchy, [[-3, 1], [75, 1]], { maximumResidentBytes: 4096, haloSamples: 2 }));
const images = plan.items.map((item) => {
  const [nx, ny, nz] = item.window.extent;
  const [ox, oy, oz] = item.window.origin;
  const field: Npy3D = { shape: [nx, ny, nz], data: Float32Array.from({ length: nx * ny * nz }, (_, at) => {
    const x = Math.floor(at / (ny * nz)), y = Math.floor(at / nz) % ny, z = at % nz;
    return 3 * coordinates[0][ox + x] + 2 * coordinates[1][oy + y] - coordinates[2][oz + z];
  }), windowGeometry: { aggregation: "cell_mean", weighting: "uniform_cell_volume", sampleLocation: "block_center", centroidsM: null, sourceShape: [8, 4, 4], origin: item.window.origin, extent: item.window.extent, stride: [1, 1, 1] } };
  const image = withVolumeBrickOwnership(prepareVtkRequest({ kind: "scalar", field, grid, reconstruction: item.reconstruction, refineForDisplay: false }), item);
  assert.equal(image.values.byteLength, item.preparedScalarBytes);
  const [sx, sy, sz] = image.shape;
  for (let z = 0; z < sz; z++) for (let y = 0; y < sy; y++) for (let x = 0; x < sx; x++) {
    const expected = 3 * (image.origin[0] + x * image.spacing[0]) + 2 * (image.origin[1] + y * image.spacing[1]) - (image.origin[2] + z * image.spacing[2]);
    assert.ok(Math.abs(image.values[x + sx * (y + sy * z)] - expected) < 1e-5);
  }
  return image;
});
assert.equal(images.length, 2);
assert.equal(volumeOwnedBounds(images[0])[1], volumeOwnedBounds(images[1])[0]);
assert.equal(plan.residentBytes, images.reduce((sum, image) => sum + image.values.byteLength, 0));

let lost = false, renders = 0;
const view = { traverseAllPasses: () => { renders++; }, getContext: () => ({ isContextLost: () => lost }) };
protectVtkTraversal(view);
Object.freeze(view);
const retire = guardVtkContextRendering({ getApiSpecificRenderWindow: () => view });
view.traverseAllPasses(); assert.equal(renders, 1);
lost = true; view.traverseAllPasses(); assert.equal(renders, 1);
retire(); lost = false; view.traverseAllPasses(); assert.equal(renders, 1);
console.log("Common physical lattice, brick ownership, admission bytes, and synchronous context-loss checks passed");

const { createVtkVolumeQualityController } = await import("../../../../ui/src/lib/vtk/vtkInteraction.ts");
const { settledVolumeDistance, volumePlanningSampleDistance, flatVolumePreview } = await import("../../../../ui/src/lib/vtk/volumeShaderBudget.ts");
let distance = 0;
const mapper = { setSampleDistance: (value: number) => { distance = value; }, setMaximumSamplesPerRay: () => {} };
const quality = createVtkVolumeQualityController({ mapper, render: () => {}, flatPreview: true });
quality.setFullSampleDistance(25);
quality.begin("interaction");
assert.equal(distance, 50);
assert.equal(flatVolumePreview(mapper), true);
assert.equal(settledVolumeDistance(mapper), 25);
quality.setFullSampleDistanceFor(mapper, 30);
assert.equal(distance, 60);
assert.equal(settledVolumeDistance(mapper), 30);
assert.equal(volumePlanningSampleDistance(mapper, 60), 30);
assert.equal(volumePlanningSampleDistance(mapper, 10), 10);
quality.dispose();
console.log("Preview shader planning retains the settled sample budget");

const { terrainWindowToCartesianField } = await import("../../../../ui/src/lib/science/cartesianInterp.ts");
for (const reverse of [false, true]) {
  const heights = [10, 50, 115, 175], volumes = [20, 60, 70, 50], values = [1, 3, 5, 7];
  if (reverse) { heights.reverse(); volumes.reverse(); values.reverse(); }
  const terrain: Npy3D = { shape: [1, 1, 4], data: Float32Array.from(values), windowGeometry: {
    aggregation: "cell_mean", weighting: "physical_cell_volume", sampleLocation: "physical_volume_centroid",
    centroidsM: { x: new Float64Array(4), y: new Float64Array(4), z: Float64Array.from(heights) }, volumesM3: Float64Array.from(volumes),
    sourceShape: [1, 1, 4], origin: [0, 0, 0], extent: [1, 1, 4], stride: [1, 1, 1],
  } };
  const remapped = terrainWindowToCartesianField(terrain, { terrainFollowing: true, drM: 1, dyM: 1,
    physicalBoundsM: { lower: [0, 0, 0], upper: [1, 1, 200] } });
  assert.deepEqual(remapped.field.data, Float32Array.from([2.2, 3.8, 5, 7]));
  assert.deepEqual(remapped.scalarCoordinatesM[2], [25, 75, 125, 175]);
}
console.log("Terrain overlap means match analytical references, including non-monotone storage");
