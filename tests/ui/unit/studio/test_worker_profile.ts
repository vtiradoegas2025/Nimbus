import assert from "node:assert/strict";
import { loadWorkerProfile, parseWorkerProfile, saveWorkerProfile, type LoadingWorkerProfile } from "../../../../ui/src/lib/studio/workerProfile.ts";

const storage = new Map<string, string>();
let failWrites = false;
(globalThis as Record<string, unknown>).window = { localStorage: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => {
    if (failWrites) throw new Error("Storage unavailable");
    storage.set(key, value);
  },
} };
const hardware = { cpuBrand: "Synthetic CPU", coreCount: 8, ramBytes: 16 * 1024 ** 3 };
const profile: LoadingWorkerProfile = {
  schema: "nimbus.studio.worker-profile", schemaVersion: 1, benchmark: "npy-load-scan-v1", hardware,
  recommendedWorkers: 2, bytesPerTrial: 32 * 1024 ** 2, elapsedMs: 200,
  trials: [{ workers: 1, medianMs: 20, throughputMibS: 1600 },
    { workers: 2, medianMs: 10.5, throughputMibS: 3047.6 },
    { workers: 4, medianMs: 10, throughputMibS: 3200 }],
};
assert.deepEqual(parseWorkerProfile(profile), profile, "smallest count within 5% of best throughput wins");
for (const invalid of [null, {}, { ...profile, benchmark: "old" }, { ...profile, schemaVersion: 2 },
  { ...profile, recommendedWorkers: 4 }, { ...profile, recommendedWorkers: 8 },
  { ...profile, elapsedMs: Number.NaN }, { ...profile, bytesPerTrial: 0 },
  { ...profile, hardware: { ...hardware, coreCount: 1 } },
  { ...profile, trials: profile.trials.slice(1) },
  { ...profile, trials: [...profile.trials, profile.trials[0]] },
  { ...profile, trials: [{ workers: 1, medianMs: 0, throughputMibS: 1 }] },
  { ...profile, trials: [{ workers: 1, medianMs: 1, throughputMibS: Infinity }] },
]) assert.equal(parseWorkerProfile(invalid), null, JSON.stringify(invalid));
assert.equal(saveWorkerProfile(profile), true);
assert.deepEqual(loadWorkerProfile(hardware), profile);
for (const other of [{ ...hardware, cpuBrand: "Different CPU" }, { ...hardware, coreCount: 4 },
  { ...hardware, ramBytes: hardware.ramBytes / 2 }]) assert.equal(loadWorkerProfile(other), null);
failWrites = true;
assert.equal(saveWorkerProfile(profile), false, "persistence failure is reported");
storage.set("nimbus-studio:worker-profile:v1", "{broken");
assert.equal(loadWorkerProfile(hardware), null);
console.log("Worker profile validation and persistence: passed");
