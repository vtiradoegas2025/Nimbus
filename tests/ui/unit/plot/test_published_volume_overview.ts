import assert from "node:assert/strict";
import { planPublishedVolumeOverview } from "../../../../ui/src/lib/plot/publishedVolumeOverview.ts";
import { validateFieldBrickHierarchy, type FieldBrickHierarchy, type Triple } from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";
import { MAX_NATIVE_FIELD_WINDOW_BYTES, MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES } from "../../../../ui/src/lib/dataset/nativeFieldWindow.ts";

function hierarchy(shape: Triple): FieldBrickHierarchy {
  const offset: Triple = [1, 2, 3];
  const result: FieldBrickHierarchy = {
    schema: "nimbus.field-brick-hierarchy", schemaVersion: 2, field: "cloud_condensate",
    publicationGeneration: "overview-test", sourceShape: shape, sourceIndexOffset: offset,
    sourceDtype: "float32", sampleLocation: "cell_center", extremaSemantics: "stored_values",
    extremaAbsolutePadding: 0, cullingSafe: true, levels: [],
  };
  for (let l = 0; l < 5; l++) {
    const stride: Triple = [2 ** l, 2 ** l, 2 ** l];
    const size = shape.map((v, axis) => Math.ceil(v / stride[axis])) as Triple;
    const brickShape: Triple = [64, 64, 32];
    const bricks = [];
    for (let i = 0; i < Math.ceil(size[0] / 64); i++) {
      for (let j = 0; j < Math.ceil(size[1] / 64); j++) {
        for (let k = 0; k < Math.ceil(size[2] / 32); k++) {
          const origin: Triple = [i * 64, j * 64, k * 32];
          const extent = origin.map((v, a) => Math.min(brickShape[a], size[a] - v)) as Triple;
          const cells = extent.reduce((n, v) => n * v, 1);
          bricks.push({ id: `l${l}:${i}:${j}:${k}`, index: [i, j, k] as Triple, origin, extent,
            scalarRange: { minimum: 0, maximum: 1, finiteCount: cells, nonFiniteCount: 0 }, contentSha256: "f".repeat(64),
            physicalBoundsM: { lower: origin.map((v, a) => offset[a] + v * stride[a]) as Triple,
              upper: origin.map((v, a) => offset[a] + Math.min(shape[a], (v + extent[a]) * stride[a])) as Triple },
          });
        }
      }
    }
    result.levels.push({ level: l, stride, shape: size, brickShape, bricks,
      source: l === 0 ? { kind: "primary_field" } : { kind: "npy_lod", path: `cloud_condensate.lod${l}.npy`,
        contentSha256: "f".repeat(64), aggregation: "recursive_cell_mean_2x2x2" },
    });
  }
  const validated = validateFieldBrickHierarchy({ ...result, levels: result.levels.map(level => ({ ...level,
    source: level.source.kind === "npy_lod" ? { ...level.source, geometry: null } : level.source,
  })) });
  assert.ok(validated.ok, validated.ok ? "" : validated.reason);
  return result;
}
const source = hierarchy([1027, 771, 259]);
const grid = { nr: 1030, nth: 775, nz: 265, drM: 25, dyM: 50, dzM: 10 };
for (const cap of [1_000_000, 4_000_000, 16_000_000, 64_000_000]) {
  const plan = planPublishedVolumeOverview(source, grid, cap);
  assert.ok(plan.level > 0 && plan.sampledCells <= cap);
  const level = source.levels[plan.level];
  const previous = source.levels[plan.level - 1];
  if (previous.level > 0) assert.ok(previous.shape.reduce((n, v) => n * v, 1) > cap);
  assert.equal(plan.windows.reduce((n, w) => n + w.extent.reduce((m, v, a) => m * Math.ceil(v / w.stride[a]), 1), 0), plan.sampledCells);
  for (const w of plan.windows) {
    assert.deepEqual(w.stride, level.stride);
    assert.equal(w.maxSourceBytes, w.maxOutputBytes);
    assert.ok(w.maxSourceBytes! <= MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES);
    assert.ok(w.maxOutputBytes! <= MAX_NATIVE_FIELD_WINDOW_BYTES);
    for (let a = 0; a < 3; a++) {
      assert.ok(w.origin[a] >= source.sourceIndexOffset[a]);
      assert.equal((w.origin[a] - source.sourceIndexOffset[a]) % w.stride[a], 0);
      assert.ok(w.origin[a] + w.extent[a] <= source.sourceIndexOffset[a] + source.sourceShape[a]);
    }
  }
  assert.match(plan.disclosure, /published cell-mean overview/);
}
const tiled = planPublishedVolumeOverview(source, grid, 64_000_000);
assert.ok(tiled.windows.length > 1);
assert.throws(() => planPublishedVolumeOverview(source, { ...grid, terrainFollowing: true }, 64_000_000), /No published/);
const terrain = structuredClone(source);
terrain.schemaVersion = 3;
for (const level of terrain.levels) if (level.source.kind === "npy_lod") {
  level.source.geometry = { kind: "physical_volume_centroids",
    xM: { path: `physical_x_center_m.lod${level.level}.npy`, contentSha256: "a".repeat(64) },
    yM: { path: `physical_y_center_m.lod${level.level}.npy`, contentSha256: "b".repeat(64) },
    zM: { path: `physical_z_center_m.lod${level.level}.npy`, contentSha256: "c".repeat(64) },
    volumeM3: { path: `physical_volume_m3.lod${level.level}.npy`, contentSha256: "d".repeat(64) },
  };
}
assert.ok(validateFieldBrickHierarchy(terrain).ok);
const physical = planPublishedVolumeOverview(terrain, { ...grid, terrainFollowing: true }, 4_000_000);
assert.ok(physical.windows.every(w => w.maxOutputBytes! > w.extent.reduce((n, v, a) => n * Math.ceil(v / w.stride[a]), 1) * 4));
assert.throws(() => planPublishedVolumeOverview(source, grid, 8), /No published/);
assert.throws(() => planPublishedVolumeOverview(source, grid, NaN), /Invalid/);
console.log("Published overview: finest fitting durable level, complete odd-edge/halo coverage, bounded tiled I/O, terrain geometry and failure cases passed");
