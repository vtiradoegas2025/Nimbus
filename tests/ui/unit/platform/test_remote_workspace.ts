import assert from "node:assert/strict";
import { loadRemoteWorkspaces, normalizeRemoteWorkspace, saveRemoteWorkspaces } from "../../../../ui/src/lib/platform/remoteWorkspace.ts";
const profile = { id: "node", name: "Desktop", host: "user@node", port: 6080, path: "/vnc.html" };
const data = new Map<string, string>();
const storage = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
assert.deepEqual(loadRemoteWorkspaces(storage), { profiles: [], error: null });
saveRemoteWorkspaces([profile], storage);
assert.deepEqual(loadRemoteWorkspaces(storage).profiles, [profile]);
for (const path of ["https://node/", "//node/", "/../x", "/./x", "/vnc.html?password=secret", "/a\r\nHost:x", "/a\\b", "/%2fhost"]) {
  assert.throws(() => normalizeRemoteWorkspace({ ...profile, path }));
}
for (const port of [0, 65536, NaN, Infinity, 1.2, "6080"]) assert.throws(() => normalizeRemoteWorkspace({ ...profile, port }));
assert.equal(normalizeRemoteWorkspace({ ...profile, host: " user@[::1] " }).host, "user@[::1]");
assert.throws(() => saveRemoteWorkspaces([profile, profile], storage));
const before = [...data.values()];
assert.throws(() => saveRemoteWorkspaces([profile], { setItem: () => { throw new Error("storage denied"); } }));
assert.deepEqual([...data.values()], before);
data.set("nimbus-studio:remote-workspaces:v1", "bad");
assert(loadRemoteWorkspaces(storage).error);
console.log("Remote workspace persistence and endpoint validation passed");
