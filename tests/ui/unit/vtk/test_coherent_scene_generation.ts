import assert from "node:assert/strict";
import {
  CoherentSceneCommit,
  coherentSceneGeneration,
  type SceneGenerationLayer,
} from "../../../../ui/src/lib/vtk/coherentSceneGeneration.ts";

const OWNERS = ["base", "cloud", "precip", "flow", "vortex", "radar"] as const;
type Owner = typeof OWNERS[number];
interface ScientificScene {
  frameIndex: number;
  simulationTimeS: number;
  publication: string;
  settings: string;
  fields: Partial<Record<Owner, readonly number[]>>;
}
let passed = 0;
const failures: string[] = [];
function test(label: string, run: () => void): void {
  try { run(); passed++; }
  catch (error) { failures.push(`${label}: ${error instanceof Error ? error.message : String(error)}`); }
}
function identity(publication: string, settings = "settings-1", frameIndex = 23): string {
  return JSON.stringify(["run-a", frameIndex, publication, settings]);
}
function layers(owners: readonly Owner[], generation: string): SceneGenerationLayer[] {
  return owners.map(layer => ({ layer, identity: generation, ready: true, error: null }));
}
function scene(owners: readonly Owner[], publication: string, settings = "settings-1", frameIndex = 23): ScientificScene {
  return { frameIndex, simulationTimeS: frameIndex === 22 ? 1320.17 : frameIndex === 23 ? 1380.21 : 1440.18, publication, settings,
    fields: Object.fromEntries(owners.map(owner => [owner, [frameIndex, OWNERS.indexOf(owner) + 0.125]])) };
}
function retainedCommit(owners: readonly Owner[]): CoherentSceneCommit<ScientificScene> {
  const state = new CoherentSceneCommit<ScientificScene>();
  const previous = identity("publication-previous", "settings-1", 22);
  state.request(previous);
  assert.equal(state.commit(previous, layers(owners, previous), scene(owners, "publication-previous", "settings-1", 22)), true);
  return state;
}

const current = identity("publication-current");
const obsolete = identity("publication-obsolete");
for (let mask = 0; mask < 2 ** OWNERS.length; mask++) {
  const owners = OWNERS.filter((_, index) => (mask & (1 << index)) !== 0);
  const label = owners.join("+") || "all owners disabled";
  test(`${label}: complete active ownership commits one scientific scene`, () => {
    const state = new CoherentSceneCommit<ScientificScene>();
    const values = scene(owners, "publication-current");
    assert.equal(state.snapshot(), null);
    assert.equal(state.commit(current, layers(owners, current), values), false);
    state.request(current);
    assert.deepEqual(coherentSceneGeneration(current, layers(owners, current)), { ready: true, error: null });
    assert.equal(state.commit(current, layers(owners, current), values), true);
    assert.equal(state.snapshot()?.identity, current);
    assert.strictEqual(state.snapshot()?.value, values);
    assert.deepEqual(Object.keys(state.snapshot()!.value.fields), owners);
  });

  for (const member of owners) {
    test(`${label}: pending ${member} retains the complete previous frame`, () => {
      const state = retainedCommit(owners);
      const retained = state.snapshot();
      const pending = layers(owners, current).map(layer => layer.layer === member ? { ...layer, ready: false } : layer);
      state.request(current);
      assert.deepEqual(coherentSceneGeneration(current, pending), { ready: false, error: null });
      assert.equal(state.commit(current, pending, scene(owners, "publication-current")), false);
      assert.strictEqual(state.snapshot(), retained);
      assert.equal(state.snapshot()?.value.publication, "publication-previous");
      const complete = scene(owners, "publication-current");
      assert.equal(state.commit(current, layers(owners, current), complete), true);
      assert.strictEqual(state.snapshot()?.value, complete);
      assert.equal(state.snapshot()?.value.frameIndex, 23);
      assert.ok(Object.values(state.snapshot()!.value.fields).every(values => values[0] === 23));
    });

    test(`${label}: failed ${member} cannot publish a partly successful scene`, () => {
      const state = retainedCommit(owners);
      const retained = state.snapshot();
      const error = `${member} source or preparation failed`;
      const failed = layers(owners, current).map(layer => layer.layer === member ? { ...layer, error } : layer);
      state.request(current);
      assert.deepEqual(coherentSceneGeneration(current, failed), { ready: false, error });
      assert.equal(state.commit(current, failed, scene(owners, "publication-current")), false);
      assert.strictEqual(state.snapshot(), retained);
      assert.equal(state.commit(current, layers(owners, current), scene(owners, "publication-current")), true);
      assert.equal(state.snapshot()?.identity, current);
    });

    test(`${label}: obsolete ${member} result and error cannot satisfy current ownership`, () => {
      const state = retainedCommit(owners);
      const retained = state.snapshot();
      const mixed = layers(owners, current).map(layer => layer.layer === member
        ? { ...layer, identity: obsolete, error: "obsolete failure" } : layer);
      state.request(current);
      assert.deepEqual(coherentSceneGeneration(current, mixed), { ready: false, error: null });
      assert.equal(state.commit(current, mixed, scene(owners, "publication-current")), false);
      assert.equal(state.commit(obsolete, layers(owners, obsolete), scene(owners, "publication-obsolete")), false);
      assert.strictEqual(state.snapshot(), retained);
      assert.equal(state.commit(current, layers(owners, current), scene(owners, "publication-current")), true);
      const completed = state.snapshot();
      assert.equal(state.commit(obsolete, layers(owners, obsolete), scene(owners, "publication-obsolete")), false);
      assert.strictEqual(state.snapshot(), completed);
    });
  }
}

test("a current failure is visible even while other owners are pending or obsolete", () => {
  const required = layers(OWNERS, current);
  required[0].ready = false;
  required[1] = { ...required[1], identity: obsolete, error: "old cloud failure" };
  required[4].error = "current vortex failure";
  assert.deepEqual(coherentSceneGeneration(current, required), { ready: false, error: "current vortex failure" });
});

for (const revision of ["settings", "publication"] as const) {
  test(`same-frame ${revision} revision requires all owners to replace together`, () => {
    const state = new CoherentSceneCommit<ScientificScene>();
    const first = identity("publication-1", "settings-1");
    const next = revision === "settings" ? identity("publication-1", "settings-2") : identity("publication-2", "settings-1");
    const initial = scene(OWNERS, "publication-1");
    state.request(first);
    assert.equal(state.commit(first, layers(OWNERS, first), initial), true);
    const retained = state.snapshot();
    state.request(next);
    assert.equal(state.commit(first, layers(OWNERS, first), initial), false);
    const partial = layers(OWNERS, next).map(layer => layer.layer === "base" ? layer : { ...layer, identity: first });
    const replacement = scene(OWNERS, revision === "publication" ? "publication-2" : "publication-1",
      revision === "settings" ? "settings-2" : "settings-1");
    assert.equal(coherentSceneGeneration(next, partial).ready, false);
    assert.equal(state.commit(next, partial, replacement), false);
    assert.strictEqual(state.snapshot(), retained);
    assert.equal(state.snapshot()?.value.frameIndex, replacement.frameIndex);
    assert.equal(state.commit(next, layers(OWNERS, next), replacement), true);
    assert.strictEqual(state.snapshot()?.value, replacement);
    assert.equal(state.snapshot()?.identity, next);
  });
}

test("new requests and failures retain complete prior ownership without allowing superseded commits", () => {
  const state = retainedCommit(OWNERS);
  const retained = state.snapshot();
  state.request(current);
  const pending = layers(OWNERS, current).map(layer => layer.layer === "radar" ? { ...layer, ready: false } : layer);
  assert.equal(state.commit(current, pending, scene(OWNERS, "publication-current")), false);
  const newer = identity("publication-newer", "settings-3", 24);
  state.request(newer);
  const failed = layers(OWNERS, newer).map(layer => layer.layer === "precip" ? { ...layer, error: "missing source" } : layer);
  assert.equal(state.commit(newer, failed, scene(OWNERS, "publication-newer", "settings-3", 24)), false);
  assert.equal(state.commit(current, layers(OWNERS, current), scene(OWNERS, "publication-current")), false);
  assert.strictEqual(state.snapshot(), retained);
  assert.equal(state.snapshot()?.value.frameIndex, 22);
  assert.equal(state.snapshot()?.value.publication, "publication-previous");
  const replacement = scene(OWNERS, "publication-newer", "settings-3", 24);
  assert.equal(state.commit(newer, layers(OWNERS, newer), replacement), true);
  assert.strictEqual(state.snapshot()?.value, replacement);
  assert.ok(Object.values(state.snapshot()!.value.fields).every(values => values[0] === 24));
});

for (const owner of OWNERS) {
  test(`duplicate ${owner} ownership is rejected, including an obsolete duplicate`, () => {
    const state = retainedCommit(OWNERS);
    const retained = state.snapshot();
    state.request(current);
    const duplicate = [...layers(OWNERS, current), { layer: owner, identity: obsolete, ready: true, error: null }];
    assert.throws(() => coherentSceneGeneration(current, duplicate), /unique layer owners/);
    assert.throws(() => state.commit(current, duplicate, scene(OWNERS, "publication-current")), /unique layer owners/);
    assert.strictEqual(state.snapshot(), retained);
  });
}

test("an empty generation identity is rejected without destroying the prior snapshot", () => {
  const state = retainedCommit(OWNERS);
  const retained = state.snapshot();
  assert.throws(() => coherentSceneGeneration("", layers(OWNERS, "")), /one identity/);
  assert.throws(() => coherentSceneGeneration("", []), /one identity/);
  state.request("");
  assert.throws(() => state.commit("", layers(OWNERS, ""), scene(OWNERS, "invalid")), /one identity/);
  assert.strictEqual(state.snapshot(), retained);
  state.request(current);
  assert.equal(state.commit(current, layers(OWNERS, current), scene(OWNERS, "publication-current")), true);
});

console.log(`Coherent scene generation: ${passed} behavioral cases passed, ${failures.length} failed (64 owner subsets).`);
for (const failure of failures) console.error(failure);
if (failures.length) process.exitCode = 1;
