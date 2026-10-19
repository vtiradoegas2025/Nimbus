import assert from "node:assert/strict";
import {
  loadRemoteComputeProfiles,
  normalizeRemoteComputeTarget,
  remoteComputeTargetKey,
  saveRemoteComputeProfiles,
  selectedRemoteComputeTarget,
  type RemoteComputeTarget,
} from "../../../../ui/src/lib/platform/remoteCompute.ts";

const values = new Map<string, string>();
const storage = {
  getItem: (key: string) => values.get(key) ?? null,
  setItem: (key: string, value: string) => { values.set(key, value); },
};
const target: RemoteComputeTarget = {
  id: "research", name: "Research workstation", host: "workstation",
  binaryPath: "/opt/nimbus/bin/nimbus_sim", workDirectory: "/home/research/nimbus",
};
assert.deepEqual(loadRemoteComputeProfiles(storage), { profiles: [], error: null }, "existing local-only settings need no migration");
saveRemoteComputeProfiles([target], storage);
assert.deepEqual(loadRemoteComputeProfiles(storage).profiles, [target]);
assert.equal(selectedRemoteComputeTarget([target], "local"), null);
assert.equal(selectedRemoteComputeTarget([target], target.id), target);
assert.equal(selectedRemoteComputeTarget([], target.id), undefined, "a deleted target cannot silently select local execution");
assert.notEqual(remoteComputeTargetKey(target), remoteComputeTargetKey({ ...target, host: "different" }));
assert.notEqual(remoteComputeTargetKey(target), remoteComputeTargetKey({ ...target, binaryPath: "/another/nimbus_sim" }));
assert.notEqual(remoteComputeTargetKey(target), remoteComputeTargetKey({ ...target, workDirectory: "/another/workspace" }));
assert.equal(normalizeRemoteComputeTarget({ ...target, host: " user@workstation ", workDirectory: "/home/research/nimbus/" }).host, "user@workstation");
assert.throws(() => normalizeRemoteComputeTarget({ ...target, host: "-oProxyCommand=other" }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, host: "host\nother" }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, binaryPath: "nimbus_sim" }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, id: "local" }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, id: "a".repeat(129) }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, name: "é".repeat(129) }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, workDirectory: "/" }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, workDirectory: "/home/research/../other" }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, binaryPath: "/opt/../bin/nimbus_sim" }));
assert.throws(() => saveRemoteComputeProfiles([target, target], storage));

const withInput = normalizeRemoteComputeTarget({ ...target, assetMappings: [{ localPath: "/local/sounding.txt", remotePath: "inputs/sounding.txt" }] });
saveRemoteComputeProfiles([withInput], storage);
assert.deepEqual(loadRemoteComputeProfiles(storage).profiles, [withInput]);
assert.notEqual(remoteComputeTargetKey(withInput), remoteComputeTargetKey(target), "input transfers own the launch evidence identity");
assert.throws(() => normalizeRemoteComputeTarget({ ...target, assetMappings: [{ localPath: "/local/data", remotePath: "../outside" }] }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, assetMappings: [{ localPath: "/local/data", remotePath: "/outside/data" }] }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, assetMappings: [{ localPath: "relative/data", remotePath: "inputs/data" }] }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, assetMappings: [{ localPath: "/local/a", remotePath: "inputs/data" }, { localPath: "/local/b", remotePath: "inputs/data" }] }));
assert.throws(() => normalizeRemoteComputeTarget({ ...target, assetMappings: [{ localPath: "/local/a", remotePath: "inputs/./data" }, { localPath: "/local/b", remotePath: "/home/research/nimbus/inputs/data" }] }));
const previous = [...values.values()];
assert.throws(() => saveRemoteComputeProfiles([target], { setItem: () => { throw new Error("write denied"); } }));
assert.deepEqual([...values.values()], previous, "a failed save cannot replace the existing profile data");

storage.setItem("nimbus-studio:remote-compute:v1", JSON.stringify({ version: 2, profiles: [target] }));
assert.match(loadRemoteComputeProfiles(storage).error!, /unsupported/);
storage.setItem("nimbus-studio:remote-compute:v1", "malformed");
assert.equal(loadRemoteComputeProfiles(storage).profiles.length, 0);
assert(loadRemoteComputeProfiles(storage).error, "corrupt connection storage is surfaced instead of fabricating a connected target");
console.log("Remote compute profiles: passed");
