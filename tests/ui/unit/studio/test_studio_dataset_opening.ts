import assert from "node:assert/strict";
import { admitEnsembleMember, openRequiredDatasetSources } from "../../../../ui/src/lib/studio/studioDatasetOpening.ts";
import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import { MAX_ENSEMBLE_MEMBERS, type EnsembleMemberInput } from "../../../../ui/src/lib/comparison/ensembleAlignment.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let passed = 0;
const failures: string[] = [];
async function test(label: string, run: () => void | Promise<void>): Promise<void> {
  try { await run(); passed++; }
  catch (reason) { failures.push(`${label}: ${reason instanceof Error ? reason.message : String(reason)}`); }
}
const source = (name: string) => createLocalDatasetSource("nimbus_run", `/cases/${name}`);
const frame: StepInfo = { index: 0, simulationTimeS: 0, fields: ["theta"], fieldUnits: { theta: "K" },
  grid: null, publicationState: "complete", publicationGeneration: "test-publication" };
const member = (name: string): EnsembleMemberInput => ({ source: source(name), steps: [frame] });

await test("a cleared group admits restoration independently of the old group's full capacity", () => {
  const previous = Array.from({ length: MAX_ENSEMBLE_MEMBERS - 1 }, (_, index) => member(`old-${index}`));
  let restored: EnsembleMemberInput[] = [];
  for (const name of ["restored-1", "restored-2", "restored-3"]) restored = admitEnsembleMember(source("new-primary"), restored, member(name));
  assert.equal(previous.length, MAX_ENSEMBLE_MEMBERS - 1);
  assert.deepEqual(restored.map(value => value.source.locator.path), ["/cases/restored-1", "/cases/restored-2", "/cases/restored-3"]);
});
await test("current primary and member identities prevent duplicate scientific ensemble weighting", () => {
  const group = admitEnsembleMember(source("primary"), [], member("member"));
  assert.throws(() => admitEnsembleMember(source("primary"), group, member("primary")), /already in/);
  assert.throws(() => admitEnsembleMember(source("primary"), group, member("member")), /already in/);
  assert.equal(group.length, 1);
});
await test("sequential restoration checks the updated capacity for every admitted member", () => {
  let group: EnsembleMemberInput[] = [];
  for (let index = 0; index < MAX_ENSEMBLE_MEMBERS - 1; index++) group = admitEnsembleMember(source("primary"), group, member(`new-${index}`));
  assert.equal(group.length + 1, MAX_ENSEMBLE_MEMBERS);
  assert.throws(() => admitEnsembleMember(source("primary"), group, member("overflow")), /bounded/);
  assert.equal(group.length + 1, MAX_ENSEMBLE_MEMBERS);
});
await test("incomplete members do not alter an already valid group", () => {
  const group = [member("valid")];
  assert.throws(() => admitEnsembleMember(source("primary"), group, { source: source("empty"), steps: [] }), /complete/);
  assert.equal(group[0].source.locator.path, "/cases/valid");
});
const required = [{ role: "primary" as const, source: source("primary") },
  { role: "comparison" as const, source: source("comparison") },
  { role: "ensemble" as const, source: source("member-1") },
  { role: "ensemble" as const, source: source("member-2") }];
await test("restoration reports success only after every declared source opened", async () => {
  const calls: string[] = [];
  const restored = await openRequiredDatasetSources({ sources: required, current: () => true,
    load: async (role, dataset) => { calls.push(`${role}:${dataset.locator.path}`); return true; } });
  assert.equal(restored, true);
  assert.deepEqual(calls, required.map(value => `${value.role}:${value.source.locator.path}`));
});
for (let failure = 0; failure < required.length; failure++) {
  await test(`failed ${required[failure].role} source blocks success and subsequent source opens`, async () => {
    const calls: string[] = [];
    await assert.rejects(openRequiredDatasetSources({ sources: required, current: () => true,
      load: async (_, dataset) => { calls.push(dataset.locator.path); return dataset !== required[failure].source; } }), /could not be reopened/);
    assert.equal(calls.length, failure + 1);
  });
}
await test("a source switch during a delayed restoration prevents stale success and later loads", async () => {
  let current = true;
  let release!: (result: boolean) => void;
  const delayed = new Promise<boolean>(resolve => { release = resolve; });
  const calls: string[] = [];
  const attempt = openRequiredDatasetSources({ sources: required, current: () => current,
    load: async (_, dataset) => { calls.push(dataset.locator.path); return delayed; } });
  current = false;
  release(true);
  assert.equal(await attempt, false);
  assert.deepEqual(calls, ["/cases/primary"]);
});
await test("a superseded failure is cancellation rather than an error on the incoming project", async () => {
  let current = true;
  const restored = await openRequiredDatasetSources({ sources: required, current: () => current,
    load: async () => { current = false; return false; } });
  assert.equal(restored, false);
});
await test("a late rejected source read cannot report failure against a newer project", async () => {
  let current = true;
  const restored = await openRequiredDatasetSources({ sources: required, current: () => current,
    load: async () => { current = false; throw new Error("obsolete native read failed"); } });
  assert.equal(restored, false);
});
console.log(`Studio dataset opening: ${passed} behavioral cases passed, ${failures.length} failed.`);
for (const failure of failures) console.error(failure);
if (failures.length) process.exitCode = 1;
