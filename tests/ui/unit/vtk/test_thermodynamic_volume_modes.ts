import {
  horizontalLevelAnomaly,
  prepareVtkRequest,
} from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { vtkPreparationIdentity } from "../../../../ui/src/lib/vtk/vtkPreparationClient.ts";
import {
  vtkPreparationWorkerScratchUpperBoundBytes,
} from "../../../../ui/src/lib/vtk/vtkPreparationResourcePlanning.ts";
import {
  resolveVolumeTransferPolicy,
  volumeOpacityPoints,
  supportsVolumeStructure,
  volumeFieldWindowAggregation,
} from "../../../../ui/src/lib/plot/volumeTransferPolicy.ts";

let passed = 0;

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
  passed += 1;
}

const shape: [number, number, number] = [2, 2, 3];
for (const field of ["p", "qv", "dewpoint", "p_prime", "theta_prime"]) {
  check(supportsVolumeStructure(field), `${field} offers explicit level departures`);
  check(volumeFieldWindowAggregation(field, "horizontal-level-anomaly") === "cell_mean", `${field} level departures average signed cells without extrema bias`);
}
check(!supportsVolumeStructure("w"), "vertical velocity keeps its physical zero");
check(volumeFieldWindowAggregation("p_prime") === "magnitude_extreme", "raw perturbation keeps its raw signed sampling policy");
const data = Float32Array.from([
  10, 20, 30,
  12, 22, 32,
  8, 18, 28,
  10, 20, 30,
]);
const source = {
  shape,
  spacing: [100, 100, 250] as [number, number, number],
  origin: [50, 50, 125] as [number, number, number],
  data,
};
const transformed = horizontalLevelAnomaly(source);

check(transformed.data !== source.data, "structure mode owns a worker-side transformed buffer");
check(source.data[0] === 10 && source.data[4] === 22, "structure mode leaves decoded science data immutable");
for (let z = 0; z < shape[2]; z += 1) {
  let sum = 0;
  for (let x = 0; x < shape[0]; x += 1) {
    for (let y = 0; y < shape[1]; y += 1) {
      sum += transformed.data[(x * shape[1] + y) * shape[2] + z];
    }
  }
  check(Math.abs(sum) < 1e-6, `structure mode removes the horizontal mean at level ${z}`);
}
check(
  transformed.data[0] === 0 && transformed.data[3] === 2
    && transformed.data[6] === -2 && transformed.data[9] === 0,
  "structure mode retains signed horizontal anomalies",
);

const field = { shape, data };
const grid = { drM: 100, dyM: 100, dzM: 250 };
const rawRequest = { kind: "scalar" as const, field, grid, scalarTransform: "none" as const };
const structureRequest = {
  kind: "scalar" as const,
  field,
  grid,
  scalarTransform: "horizontal-level-anomaly" as const,
};
const prepared = prepareVtkRequest(structureRequest);
check(prepared.scalarTransform === "horizontal-level-anomaly", "prepared metadata discloses the structural transform");
check(prepared.stats?.min === -2 && prepared.stats.max === 2, "prepared statistics describe the transformed values");
check(prepared.stats?.positiveP98 === 2 && prepared.stats.negativeP02 === -2,
  "prepared statistics retain both conditioned signed tails");
check(vtkPreparationIdentity(rawRequest) !== vtkPreparationIdentity(structureRequest), "raw and structure modes cannot alias one prepared cache entry");
check(
  vtkPreparationWorkerScratchUpperBoundBytes(structureRequest) === data.byteLength,
  "resource admission reserves the anomaly output scratch before worker execution",
);
check(vtkPreparationWorkerScratchUpperBoundBytes(rawRequest) === 0, "raw mode allocates no transform scratch");

const anomalyPolicy = resolveVolumeTransferPolicy(
  "temperature",
  "Viridis",
  { min: -2, max: 2, p2: -2, p98: 2 },
  "horizontal-level-anomaly",
);
check(anomalyPolicy.kind === "signed", "structure mode uses a signed transfer independent of the source palette");
check(anomalyPolicy.min === -2 && anomalyPolicy.max === 2, "structure mode keeps a symmetric anomaly range");

const rawPolicy = resolveVolumeTransferPolicy(
  "temperature",
  "Viridis",
  { min: 280, max: 310, p2: 282, p98: 308 },
);
const compactCore = resolveVolumeTransferPolicy("theta", "Viridis",
  { min: -0.07, max: 2, p2: -0.07, p98: 0.01 }, "horizontal-level-anomaly");
check(compactCore.max === 0.5 && compactCore.min === -0.5, "a compact warm core anchors the range above its weak compensating background");
const compactOpacity = volumeOpacityPoints(compactCore, 1);
check(compactOpacity[3].scalar < -0.07 && compactOpacity[4].scalar > 0.07,
  "weak level departures stay in the transparent interval without changing their stored values");
const rawOpacity = volumeOpacityPoints(rawPolicy, 1);
check(rawPolicy.kind === "sequential", "raw mode preserves the physical scalar ordering");
check(rawOpacity[1].scalar < rawPolicy.min + 0.2 * (rawPolicy.max - rawPolicy.min), "raw mode does not discard the lower profile");
check(rawOpacity.at(-1)?.opacity === 0.1, "raw mode remains optically thin at its strongest value");

console.log(`thermodynamic volume modes: ${passed} checks passed`);
