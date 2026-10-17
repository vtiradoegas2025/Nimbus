import {
  compactVolumeBrickPlan,
  planVolumeBricks,
} from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";
import {
  resolveVolumeTransferPolicy,
  volumeOpacityPoints,
} from "../../../../ui/src/lib/plot/volumeTransferPolicy.ts";
import {
  buildTransientVolumeHierarchy,
  sampleVolumeFieldStats,
  TRANSIENT_VOLUME_BRICK_SHAPE,
} from "../../../../ui/src/lib/vtk/transientVolumeHierarchy.ts";
import type { FieldLoadWindow, StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
  passed += 1;
}

const shape = [130, 130, 65] as [number, number, number];
const source = new Float32Array(shape[0] * shape[1] * shape[2]);
for (let x = 0; x < 128; x += 1) {
  for (let y = 0; y < 128; y += 1) {
    for (let z = 0; z < 40; z += 1) {
      source[(x * shape[1] + y) * shape[2] + z] = 0.1;
    }
  }
}

const step: StepInfo = {
  index: 4,
  simulationTimeS: 240,
  fields: ["vorticity_magnitude"],
  grid: {
    nr: shape[0], nth: shape[1], nz: shape[2],
    coordinateSystem: "cartesian",
    geometryKind: "physical",
    drM: 100, dyM: 100, dzM: 100,
    scalarOriginM: [-6_450, -6_450, 50],
  },
  fieldUnits: { vorticity_magnitude: "1/s" },
  publicationState: "complete",
  publicationGeneration: "transient-fixture-generation",
};

function mockReader(calls: FieldLoadWindow[]) {
  return async (window: FieldLoadWindow) => {
    calls.push(window);
    const outputShape = window.extent.map((count, axis) =>
      Math.floor((count - 1) / window.stride[axis]) + 1) as [number, number, number];
    const data = new Float32Array(outputShape[0] * outputShape[1] * outputShape[2]);
    let destination = 0;
    for (let ox = 0; ox < outputShape[0]; ox += 1) {
      for (let oy = 0; oy < outputShape[1]; oy += 1) {
        for (let oz = 0; oz < outputShape[2]; oz += 1) {
          const base = [
            window.origin[0] + ox * window.stride[0],
            window.origin[1] + oy * window.stride[1],
            window.origin[2] + oz * window.stride[2],
          ];
          if (window.aggregation === "point_sample") {
            data[destination++] = source[(base[0] * shape[1] + base[1]) * shape[2] + base[2]];
            continue;
          }
          let selected = window.aggregation === "magnitude_extreme" ? 0 : Number.NEGATIVE_INFINITY;
          const upper = base.map((value, axis) => Math.min(
            window.origin[axis] + window.extent[axis],
            value + window.stride[axis],
          ));
          for (let x = base[0]; x < upper[0]; x += 1) {
            for (let y = base[1]; y < upper[1]; y += 1) {
              for (let z = base[2]; z < upper[2]; z += 1) {
                const value = source[(x * shape[1] + y) * shape[2] + z];
                selected = window.aggregation === "magnitude_extreme"
                  ? Math.abs(value) > Math.abs(selected) ? value : selected
                  : Math.max(selected, value);
              }
            }
          }
          data[destination++] = selected;
        }
      }
    }
    return { shape: outputShape, data };
  };
}

for (const format of ["npy", "zfp", "netcdf"] as const) {
  const calls: FieldLoadWindow[] = [];
  const result = await buildTransientVolumeHierarchy({
    runDir: `/fixture/${format}`,
    step,
    fieldName: "vorticity_magnitude",
    opacityClass: "sparse-positive",
    reader: mockReader(calls),
  });
  const level = result.hierarchy.levels[0];
  check(level.brickShape.join("x") === TRANSIENT_VOLUME_BRICK_SHAPE.join("x"), `${format} uses canonical transient bricks`);
  check(level.bricks.length === 8, `${format} transient index covers every brick`);
  check(level.bricks.at(-1)?.extent.join("x") === "2x2x25", `${format} preserves edge-brick extents`);
  check(calls.some((call) => call.aggregation === "block_max"), `${format} derives conservative maxima`);
  check(calls.some((call) => call.aggregation === "point_sample"), `${format} derives bounded display statistics`);
  check(calls.every((call) => (call.maxSourceBytes ?? 0) <= 256 * 1024 * 1024), `${format} respects the source-read envelope`);
  const transfer = resolveVolumeTransferPolicy("vorticity_magnitude", "Inferno", result.stats);
  const opacity = volumeOpacityPoints(transfer, 1).map((point) =>
    [point.scalar, point.opacity] as const);
  const plan = planVolumeBricks(result.hierarchy, opacity, {
    maximumResidentBytes: source.byteLength,
  });
  check(plan.items.length === 1, `${format} retains only the opacity-bearing native brick`);
  check(plan.transparentBrickCount === 7, `${format} culls only proven-transparent bricks`);
  check(plan.items[0].window.stride.join("x") === "1x1x1", `${format} retained brick values remain native`);
}

const publishedStatsCalls: FieldLoadWindow[] = [];
const publishedStats = await sampleVolumeFieldStats({
  runDir: "/fixture/published-netcdf",
  step,
  fieldName: "vorticity_magnitude",
  reader: mockReader(publishedStatsCalls),
});
check(Math.abs(publishedStats.p98 - 0.1) < 1e-6, "published hierarchies use source-distribution statistics");
check(publishedStatsCalls.every((call) => call.aggregation === "point_sample"), "published statistics stay format-neutral and bounded");

const denseCalls: FieldLoadWindow[] = [];
const denseHierarchy = await buildTransientVolumeHierarchy({
  runDir: "/fixture/dense",
  step,
  fieldName: "vorticity_magnitude",
  opacityClass: "sparse-positive",
  reader: mockReader(denseCalls),
});
const densePlan = planVolumeBricks(denseHierarchy.hierarchy, [[0, 1], [0.1, 1]], {
  maximumResidentBytes: source.byteLength,
});
const compacted = compactVolumeBrickPlan(denseHierarchy.hierarchy, densePlan, {
  maximumResidentBytes: source.byteLength,
});
check(densePlan.items.length === 8 && compacted.items.length === 2, "adjacent native bricks compact into fewer ray-cast actors");
check(compacted.items.every((item) => item.window.stride.join("x") === "1x1x1"), "macro bricks retain native samples");
check(compacted.residentBytes === source.byteLength, "macro-brick residency includes transparent fill exactly");

console.log(`transient volume hierarchy: ${passed} checks passed`);
