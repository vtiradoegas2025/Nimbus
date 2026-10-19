import {
  applyPreparedGlobalHorizontalLevelAnomaly,
  type PreparedVtkImage,
} from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

function brick(originX: number): PreparedVtkImage {
  const shape: [number, number, number] = [2, 2, 3];
  const values = new Float32Array(12);
  for (let z = 0; z < 3; z += 1) {
    for (let y = 0; y < 2; y += 1) {
      for (let x = 0; x < 2; x += 1) {
        values[x + 2 * (y + 2 * z)] = 280 + 10 * z + originX + x + 2 * y;
      }
    }
  }
  return {
    shape,
    spacing: [1, 1, 1],
    origin: [originX, 0, 0],
    values,
    sourceNonFiniteCount: 0,
    samplingProvenance: {} as PreparedVtkImage["samplingProvenance"],
  };
}

const transformed = applyPreparedGlobalHorizontalLevelAnomaly([brick(0), brick(2)]);
for (let z = 0; z < 3; z += 1) {
  let sum = 0;
  for (const prepared of transformed) {
    for (let y = 0; y < 2; y += 1) {
      for (let x = 0; x < 2; x += 1) sum += prepared.values[x + 2 * (y + 2 * z)];
    }
  }
  check(Math.abs(sum) < 1e-6, `global physical level ${z} has one shared zero anomaly mean`);
}
check(transformed.every((prepared) =>
  prepared.scalarTransform === "horizontal-level-anomaly"
    && prepared.stats?.min === transformed[0].stats?.min
    && prepared.stats?.max === transformed[0].stats?.max),
"every transformed brick shares one generation-wide range");
check(transformed.every((prepared) => prepared.stats?.positiveP98 === transformed[0].stats?.positiveP98
    && prepared.stats?.negativeP02 === transformed[0].stats?.negativeP02),
  "global anomaly bricks share conditioned signed tails");
check(transformed[0].values[1] !== transformed[1].values[0],
  "the transform retains resolved horizontal structure instead of zeroing each brick");

console.log(`\nvolume anomaly brick tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
