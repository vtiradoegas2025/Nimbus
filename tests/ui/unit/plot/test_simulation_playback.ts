import assert from "node:assert/strict";
import { simulationPlaybackTimeline, simulationPlaybackIntervalMs } from "../../../../ui/src/lib/plot/simulationPlayback.ts";
const timeline = simulationPlaybackTimeline([0, 2, 10, 14].map(simulationTimeS => ({ simulationTimeS })))!;
assert.equal(simulationPlaybackIntervalMs(timeline, 0, 1, 2, 50), 1000);
assert.equal(simulationPlaybackIntervalMs(timeline, 1, 1, 2, 50), 4000);
assert.equal(simulationPlaybackIntervalMs(timeline, 0, 2, 2, 50), 5000);
assert.equal(simulationPlaybackIntervalMs(timeline, 3, 1, 2, 50), 2000);
assert.equal(simulationPlaybackIntervalMs(timeline, 3, 2, 2, 50), 3000);
assert.equal(simulationPlaybackIntervalMs(timeline, 0, 1, 120, 500), 500);
for (const values of [[], [0], [0, 0], [2, 1], [0, NaN], [0, Infinity]]) {
  assert.equal(simulationPlaybackTimeline(values.map(simulationTimeS => ({ simulationTimeS }))), null);
}
assert.equal(simulationPlaybackTimeline([{ simulationTimeS: 0 }, {}]), null);
assert.equal(simulationPlaybackIntervalMs(timeline, -1, 1, 2, 50), null);
assert.equal(simulationPlaybackIntervalMs(timeline, 0, 0, 2, 50), null);
assert.equal(simulationPlaybackIntervalMs(timeline, 0, 1, 0, 50), null);
console.log("Simulation-time playback: irregular cadence, stride, loop dwell, rate cap, and invalid inputs passed.");
