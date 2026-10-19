import {
  CONFIG_DRAFT_LIMIT,
  CONFIG_DRAFT_STORAGE_KEY,
  activateConfigDraft,
  branchConfigDraft,
  configDraftIsDirty,
  configDraftStore,
  createConfigDraft,
  deleteConfigDraft,
  inspectConfigDraftStore,
  loadConfigDraftStore,
  markConfigDraftSaved,
  parseConfigDraftStore,
  saveConfigDraftStore,
  updateConfigDraft,
} from "../../../../ui/src/lib/config/configDrafts.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const at = (minute: number) => `2026-07-20T20:${String(minute).padStart(2, "0")}:00.000Z`;

class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const base = createConfigDraft({
  caseName: "cm1-smoke",
  document: "schema: nimbus.v1\ntime:\n  run_time: 30\n",
}, { id: "draft_baseline-0001", now: at(1) });
let store = configDraftStore(base);
check(store.activeId === base.id && store.drafts.length === 1, "a first draft becomes active");
check(configDraftIsDirty(base), "an unsaved draft is explicitly dirty");

store = markConfigDraftSaved(store, base.id, base.document, "/cases/cm1-smoke.yaml", at(2));
check(!configDraftIsDirty(store.drafts[0]), "matching file fingerprint is clean");
store = updateConfigDraft(store, base.id, { document: `${base.document}# review\n` }, at(3));
check(configDraftIsDirty(store.drafts[0]), "editing after save is dirty");

store = branchConfigDraft(store, base.id, "cm1-smoke-branch", { id: "draft_branch-0002", now: at(4) });
check(store.activeId === "draft_branch-0002" && store.drafts[0].parentId === base.id, "branch records its parent and becomes active");
check(store.drafts[0].saved === null && store.drafts[0].sourcePath === null, "branch never claims the parent file save");
store = activateConfigDraft(store, base.id);
check(store.activeId === base.id, "an existing branch can be activated");

const storage = new MemoryStorage();
saveConfigDraftStore(storage, store);
const restored = loadConfigDraftStore(storage)!;
check(restored.activeId === base.id && restored.drafts.length === 2, "draft branches survive restart");
check(storage.values.has(CONFIG_DRAFT_STORAGE_KEY), "draft ledger uses its isolated storage key");

store = deleteConfigDraft(restored, base.id);
check(store.drafts.length === 1 && store.drafts[0].parentId === null, "deleting a parent keeps and detaches its child");
let refusedLastDelete = false;
try { deleteConfigDraft(store, store.activeId); } catch { refusedLastDelete = true; }
check(refusedLastDelete, "the last recovery draft cannot be deleted");

const serialized = JSON.parse(storage.values.get(CONFIG_DRAFT_STORAGE_KEY)!);
serialized.activeId = "draft_missing-0000";
check(parseConfigDraftStore(serialized) === null, "orphaned active identity fails closed");
serialized.activeId = base.id;
serialized.drafts[0].future = true;
check(parseConfigDraftStore(serialized) === null, "unexpected draft properties fail closed");
storage.values.set(CONFIG_DRAFT_STORAGE_KEY, "broken");
check(loadConfigDraftStore(storage) === null, "corrupt storage is ignored safely");
check(
  inspectConfigDraftStore(storage).kind === "unreadable"
    && storage.values.get(CONFIG_DRAFT_STORAGE_KEY) === "broken",
  "corrupt recovery is reported without overwriting its original bytes",
);

let bounded = configDraftStore(base);
for (let index = 1; index < CONFIG_DRAFT_LIMIT; index += 1) {
  bounded = branchConfigDraft(bounded, bounded.activeId, `branch-${index}`, {
    id: `draft_branch-${String(index).padStart(4, "0")}`,
    now: at(5 + index),
  });
}
let refusedOverflow = false;
try { branchConfigDraft(bounded, bounded.activeId, "overflow"); } catch { refusedOverflow = true; }
check(refusedOverflow, "draft count is bounded");

const throwing = { getItem() { throw new Error("blocked"); } };
check(loadConfigDraftStore(throwing) === null, "blocked storage leaves the editor usable");
check(inspectConfigDraftStore(throwing).kind === "unreadable", "blocked storage is distinguishable from an empty draft ledger");

console.log(`config drafts: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
