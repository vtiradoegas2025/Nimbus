/*
 * Python editor buffers.
 *
 * The property that matters most here is that hand-written code is never lost.
 * The previous design had two ways to lose it — one localStorage buffer as the
 * only copy, and opening a gallery entry overwriting the editor with no dirty
 * check — so the cases below lean on dirty tracking, tab reuse, and what the
 * recovery store is and is not allowed to be the sole copy of.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_workspace.ts
 */

import {
  activateBuffer,
  activeBuffer,
  addScratchBuffer,
  buffersOutsideRecoveryLimit,
  canAddBuffer,
  closeBuffer,
  createWorkspace,
  fileName,
  isBufferDirty,
  markBufferLoaded,
  markBufferSaved,
  WorkspaceOperationOwner,
  MAX_BUFFERS,
  MAX_RECOVERED_BUFFER_BYTES,
  migrateLegacySnippet,
  nextBufferId,
  openPathInWorkspace,
  orphanedPythonWorkspaceKeys,
  parseWorkspace,
  PRIMARY_PANE_ID,
  PYTHON_WORKSPACE_SCHEMA,
  PYTHON_WORKSPACE_STORAGE_KEY,
  PYTHON_WORKSPACE_VERSION,
  pythonWorkspaceKeysForProject,
  pythonWorkspaceSeedKey,
  pythonWorkspaceStorageKey,
  seedProjectWorkspace,
  serializeWorkspace,
  setBufferCode,
  workspaceHasUnsaved,
  workspaceRecoveryWarning,
  type PythonWorkspace,
} from "../../../../ui/src/lib/studio/pythonWorkspace.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const STARTER = "# starter\n";

// ---------- 1. A fresh workspace ----------

const fresh = createWorkspace(STARTER);
check(fresh.buffers.length === 1, "a new workspace has exactly one buffer");
check(fresh.buffers[0].path === null, "the initial buffer is scratch");
check(activeBuffer(fresh).id === fresh.buffers[0].id, "the initial buffer is active");
check(!isBufferDirty(fresh.buffers[0]), "an untouched scratch buffer is clean");
check(!workspaceHasUnsaved(fresh), "a fresh workspace has nothing unsaved");

const edited = setBufferCode(fresh, fresh.activeId, "# edited\n");
check(isBufferDirty(activeBuffer(edited)), "editing marks the buffer dirty");
check(workspaceHasUnsaved(edited), "an edited workspace reports unsaved work");
check(!isBufferDirty(fresh.buffers[0]), "the original workspace object is not mutated");

// ---------- 2. Names and ids stay distinct ----------

check(nextBufferId([]) === "buf_1", "ids start at one");
check(nextBufferId(fresh.buffers) !== fresh.buffers[0].id, "a new id avoids an existing one");

let many = createWorkspace(STARTER);
many = addScratchBuffer(many, STARTER);
many = addScratchBuffer(many, STARTER);
check(many.buffers.length === 3, "scratch buffers accumulate");
check(new Set(many.buffers.map((b) => b.id)).size === 3, "ids are unique");
check(new Set(many.buffers.map((b) => b.name)).size === 3, "scratch names are distinguishable");
check(activeBuffer(many).id === many.buffers[2].id, "a new scratch buffer is focused");

let full = createWorkspace(STARTER);
for (let index = 0; index < MAX_BUFFERS + 5; index += 1) full = addScratchBuffer(full, STARTER);
check(full.buffers.length === MAX_BUFFERS, `the buffer count is capped at ${MAX_BUFFERS}`);
check(!canAddBuffer(full), "a full workspace reports that it cannot take more");

// ---------- 3. Opening a file ----------

check(fileName("/a/b/analysis.py") === "analysis.py", "a POSIX path reduces to its name");
check(fileName("C:\\x\\analysis.py") === "analysis.py", "a Windows path reduces to its name");

const firstOpen = openPathInWorkspace(fresh, "/plugins/vort.py", "# vort\n");
check(firstOpen.opened, "opening an unseen path creates a tab");
check(firstOpen.workspace.buffers.length === 2, "the new tab is added alongside scratch");
check(activeBuffer(firstOpen.workspace).path === "/plugins/vort.py", "the opened file is focused");
check(activeBuffer(firstOpen.workspace).name === "vort.py", "the tab is named after the file");
check(!isBufferDirty(activeBuffer(firstOpen.workspace)), "a freshly opened file is clean");

/*
 * Two tabs over one path would let a user save one tab's text over the other's
 * edits, silently. Reopening focuses the existing tab instead.
 */
const secondOpen = openPathInWorkspace(firstOpen.workspace, "/plugins/vort.py", "# vort\n");
check(!secondOpen.opened, "reopening the same path does not create a second tab");
check(secondOpen.workspace.buffers.length === 2, "the tab count is unchanged");
check(secondOpen.bufferId === firstOpen.bufferId, "the existing tab is the one focused");

// A clean tab adopts what is now on disk; the file changed underneath it.
const refreshed = openPathInWorkspace(firstOpen.workspace, "/plugins/vort.py", "# changed\n");
check(
  activeBuffer(refreshed.workspace).code === "# changed\n",
  "reopening a clean tab picks up the newer disk contents",
);
check(!isBufferDirty(activeBuffer(refreshed.workspace)), "the refreshed tab is clean");

// A dirty tab must NOT be overwritten: its edits are the only unsaved copy.
const dirtyOpen = setBufferCode(firstOpen.workspace, firstOpen.bufferId, "# my unsaved work\n");
const reopenedDirty = openPathInWorkspace(dirtyOpen, "/plugins/vort.py", "# disk version\n");
check(
  activeBuffer(reopenedDirty.workspace).code === "# my unsaved work\n",
  "reopening a dirty tab preserves the unsaved edits",
);
check(isBufferDirty(activeBuffer(reopenedDirty.workspace)), "the dirty tab stays dirty");

// ---------- 4. Saving and loading ----------

const saved = markBufferSaved(dirtyOpen, firstOpen.bufferId, "/plugins/vort.py");
const savedBuffer = saved.buffers.find((b) => b.id === firstOpen.bufferId)!;
check(!isBufferDirty(savedBuffer), "saving clears the dirty flag");
check(savedBuffer.savedCode === "# my unsaved work\n", "saving records the written text");

const writtenCode = activeBuffer(dirtyOpen).code;
const editedDuringSave = setBufferCode(dirtyOpen, firstOpen.bufferId, "# a newer unsaved edit\n");
const completedSave = markBufferSaved(editedDuringSave, firstOpen.bufferId, "/plugins/vort.py", writtenCode);
check(activeBuffer(completedSave).savedCode === writtenCode, "save completion records bytes actually written to disk");
check(activeBuffer(completedSave).code === "# a newer unsaved edit\n" && isBufferDirty(activeBuffer(completedSave)),
  "typing during a delayed save remains visibly unsaved");
const saveAsDuringEdit = markBufferSaved(editedDuringSave, firstOpen.bufferId, "/plugins/renamed.py", writtenCode);
check(activeBuffer(saveAsDuringEdit).path === "/plugins/renamed.py" && isBufferDirty(activeBuffer(saveAsDuringEdit)),
  "Save As adopts its chosen path while preserving newer dirty text");
const ioOwner = new WorkspaceOperationOwner("project-a/primary");
const oldOperation = ioOwner.capture();
check(ioOwner.current(oldOperation), "an I/O operation owns its originating workspace");
ioOwner.select("project-b/primary");
check(!ioOwner.current(oldOperation), "another project cannot accept the old project's reused-buffer-ID completion");
ioOwner.select("project-a/primary");
check(!ioOwner.current(oldOperation), "returning to a project cannot resurrect an obsolete asynchronous I/O result");
const currentOperation = ioOwner.capture();
ioOwner.select("project-a/primary");
check(ioOwner.current(currentOperation), "ordinary rerenders do not cancel current workspace I/O");
ioOwner.retire();
check(!ioOwner.current(currentOperation), "an unmounted workspace cannot accept delayed file or interpreter UI updates");

const renamed = markBufferSaved(edited, edited.activeId, "/plugins/new_name.py");
const renamedBuffer = activeBuffer(renamed);
check(renamedBuffer.path === "/plugins/new_name.py", "save-as attaches the path to the buffer");
check(renamedBuffer.name === "new_name.py", "save-as renames the tab");
check(!isBufferDirty(renamedBuffer), "a scratch buffer saved to a file becomes clean");

const loaded = markBufferLoaded(dirtyOpen, firstOpen.bufferId, "# from disk\n");
const loadedBuffer = loaded.buffers.find((b) => b.id === firstOpen.bufferId)!;
check(!isBufferDirty(loadedBuffer), "a completed disk read leaves the buffer clean");
check(!loadedBuffer.needsReload, "a completed disk read clears the reload flag");

// ---------- 5. Closing ----------

const closedMiddle = closeBuffer(many, many.buffers[1].id, STARTER);
check(closedMiddle.buffers.length === 2, "closing removes one buffer");
check(
  !closedMiddle.buffers.some((b) => b.id === many.buffers[1].id),
  "the closed buffer is the one removed",
);

const closedActive = closeBuffer(many, many.activeId, STARTER);
check(
  closedActive.buffers.some((b) => b.id === closedActive.activeId),
  "closing the active tab focuses a surviving one",
);

const closedLast = closeBuffer(fresh, fresh.activeId, STARTER);
check(closedLast.buffers.length === 1, "closing the only tab leaves a replacement");
check(closedLast.buffers[0].path === null, "the replacement is scratch");
check(closedLast.buffers[0].code === STARTER, "the replacement starts from the starter text");

check(
  closeBuffer(fresh, "no_such_id", STARTER).buffers.length === 1,
  "closing an unknown id is a no-op",
);
check(
  activateBuffer(fresh, "no_such_id").activeId === fresh.activeId,
  "activating an unknown id is a no-op",
);

// ---------- 6. Recovery storage holds only what is not on disk ----------

/*
 * A clean file-backed buffer is already durable; persisting its text would
 * duplicate the file into localStorage for no benefit and grow without bound.
 * Unsaved work has no other copy and must always be persisted.
 */
const mixed: PythonWorkspace = {
  buffers: [
    { id: "buf_1", name: "Scratch", path: null, code: "# scratch\n", savedCode: "# scratch\n", needsReload: false },
    { id: "buf_2", name: "clean.py", path: "/p/clean.py", code: "# clean\n", savedCode: "# clean\n", needsReload: false },
    { id: "buf_3", name: "dirty.py", path: "/p/dirty.py", code: "# NEW\n", savedCode: "# old\n", needsReload: false },
  ],
  activeId: "buf_3",
};
const serialized = JSON.parse(serializeWorkspace(mixed));
check(serialized.schema === PYTHON_WORKSPACE_SCHEMA, "the record is schema-tagged");
check(serialized.schemaVersion === PYTHON_WORKSPACE_VERSION, "the record is versioned");
check(serialized.buffers[0].code === "# scratch\n", "scratch text is persisted");
check(serialized.buffers[1].code === undefined, "a clean file's text is NOT duplicated into storage");
check(serialized.buffers[1].path === "/p/clean.py", "a clean file persists its path so it can be reopened");
check(serialized.buffers[2].code === "# NEW\n", "unsaved edits to a file ARE persisted");

const restored = parseWorkspace(serializeWorkspace(mixed), STARTER);
check(restored.buffers.length === 3, "every buffer survives a round trip");
check(restored.activeId === "buf_3", "the active tab survives a round trip");
check(restored.buffers[0].code === "# scratch\n", "scratch text is restored");
check(!isBufferDirty(restored.buffers[0]), "restored scratch is not falsely marked dirty");
check(restored.buffers[1].needsReload, "a path-only buffer is marked for reload");
check(restored.buffers[1].code === "", "a path-only buffer carries no stale text");
check(
  !isBufferDirty(restored.buffers[1]),
  "a restored path-only buffer is not dirty",
);
/*
 * The reload flag must suppress dirty on its own, not merely because the
 * restored text happens to equal the empty savedCode. If a buffer whose file
 * has not been read yet ever reported dirty, Save would write its empty text
 * over the real file. Constructed directly so the two conditions differ.
 */
check(
  !isBufferDirty({
    id: "buf_x", name: "pending.py", path: "/p/pending.py",
    code: "", savedCode: "# real file contents\n", needsReload: true,
  }),
  "an unread buffer is not dirty even when its text differs from savedCode",
);
check(
  isBufferDirty({
    id: "buf_x", name: "pending.py", path: "/p/pending.py",
    code: "", savedCode: "# real file contents\n", needsReload: false,
  }),
  "the same buffer IS dirty once its text has been read",
);
check(restored.buffers[2].code === "# NEW\n", "unsaved edits are restored");
check(isBufferDirty(restored.buffers[2]), "restored unsaved edits still read as unsaved");

// ---------- 7. Corrupt or hostile recovery state ----------

check(parseWorkspace(null, STARTER).buffers.length === 1, "absent state yields a fresh workspace");
check(parseWorkspace("", STARTER).buffers.length === 1, "empty state yields a fresh workspace");
check(parseWorkspace("{not json", STARTER).buffers.length === 1, "unparseable state is discarded");
check(
  parseWorkspace(JSON.stringify({ schema: "other", schemaVersion: 1, buffers: [] }), STARTER)
    .buffers.length === 1,
  "a foreign schema is discarded",
);
check(
  parseWorkspace(
    JSON.stringify({ schema: PYTHON_WORKSPACE_SCHEMA, schemaVersion: 99, buffers: [] }),
    STARTER,
  ).buffers.length === 1,
  "a future version is discarded rather than guessed at",
);

const duplicateIds = parseWorkspace(
  JSON.stringify({
    schema: PYTHON_WORKSPACE_SCHEMA,
    schemaVersion: PYTHON_WORKSPACE_VERSION,
    activeId: "buf_1",
    buffers: [
      { id: "buf_1", name: "a", path: null, code: "a" },
      { id: "buf_1", name: "b", path: null, code: "b" },
    ],
  }),
  STARTER,
);
check(duplicateIds.buffers.length === 1, "duplicate ids are collapsed");

const shellOnly = parseWorkspace(
  JSON.stringify({
    schema: PYTHON_WORKSPACE_SCHEMA,
    schemaVersion: PYTHON_WORKSPACE_VERSION,
    buffers: [{ id: "buf_1", name: "ghost", path: null }],
  }),
  STARTER,
);
check(
  shellOnly.buffers.length === 1 && shellOnly.buffers[0].name !== "ghost",
  "a scratch record with no text is dropped rather than restored empty",
);

const danglingActive = parseWorkspace(
  JSON.stringify({
    schema: PYTHON_WORKSPACE_SCHEMA,
    schemaVersion: PYTHON_WORKSPACE_VERSION,
    activeId: "buf_99",
    buffers: [{ id: "buf_1", name: "a", path: null, code: "a" }],
  }),
  STARTER,
);
check(danglingActive.activeId === "buf_1", "an active id pointing at nothing falls back");

const tooMany = parseWorkspace(
  JSON.stringify({
    schema: PYTHON_WORKSPACE_SCHEMA,
    schemaVersion: PYTHON_WORKSPACE_VERSION,
    buffers: Array.from({ length: MAX_BUFFERS + 10 }, (_, index) => ({
      id: `buf_${index}`, name: `n${index}`, path: null, code: "x",
    })),
  }),
  STARTER,
);
check(tooMany.buffers.length === MAX_BUFFERS, "an oversized record is trimmed to the cap");

// A single pathological paste must not be able to fill the storage quota.
const huge = "x".repeat(MAX_RECOVERED_BUFFER_BYTES + 1);
const hugeWorkspace: PythonWorkspace = {
  buffers: [{ id: "buf_1", name: "Scratch", path: null, code: huge, savedCode: "", needsReload: false }],
  activeId: "buf_1",
};
check(
  JSON.parse(serializeWorkspace(hugeWorkspace)).buffers[0].code === undefined,
  "a buffer beyond the recovery size cap is not written to storage",
);
check(
  buffersOutsideRecoveryLimit(hugeWorkspace)[0]?.id === "buf_1",
  "an unsaved buffer omitted from recovery is reported before persistence",
);
check(
  workspaceRecoveryWarning(hugeWorkspace)?.includes("Save to disk") === true,
  "an omitted recovery buffer has an actionable persistent warning",
);
const cleanHugeFile: PythonWorkspace = {
  buffers: [{
    id: "buf_1", name: "large.py", path: "/p/large.py",
    code: huge, savedCode: huge, needsReload: false,
  }],
  activeId: "buf_1",
};
check(
  workspaceRecoveryWarning(cleanHugeFile) === null,
  "a large clean file does not warn because disk already holds its text",
);

// ---------- 8. The previous single buffer is not lost on upgrade ----------

/*
 * The two keys being distinct is guaranteed by their literal types — asserting
 * it at runtime is a check that cannot fail, so what is tested here is the
 * behaviour that actually matters: the old text survives the upgrade.
 */
const migrated = migrateLegacySnippet("# my old work\n", STARTER);
check(migrated.buffers.length === 1, "migration yields one buffer");
check(migrated.buffers[0].code === "# my old work\n", "the previous buffer's text is preserved");
check(migrated.buffers[0].path === null, "the migrated buffer is scratch");
check(!isBufferDirty(migrated.buffers[0]), "the migrated buffer is not falsely dirty");
check(
  migrateLegacySnippet(null, STARTER).buffers[0].code === STARTER,
  "no previous buffer yields the starter text",
);

// ---------- 9. One recovery store per project ----------

/*
 * The property is injectivity. Two projects must never resolve to one store
 * (they would show each other's tabs, which is the bug this exists to fix), and
 * no project may resolve to the unscoped store (it would inherit, and then
 * overwrite, the no-project workspace).
 */
check(
  pythonWorkspaceStorageKey(null) === PYTHON_WORKSPACE_STORAGE_KEY,
  "no project uses the unscoped store",
);
check(
  pythonWorkspaceStorageKey("project_a") !== PYTHON_WORKSPACE_STORAGE_KEY,
  "a project does not share the unscoped store",
);
check(
  pythonWorkspaceStorageKey("project_a") !== pythonWorkspaceStorageKey("project_b"),
  "two projects get two stores",
);
check(
  pythonWorkspaceStorageKey("project_a") === pythonWorkspaceStorageKey("project_a"),
  "the same project always resolves to the same store",
);
check(
  pythonWorkspaceStorageKey("") !== PYTHON_WORKSPACE_STORAGE_KEY,
  "an empty project id still does not collide with the unscoped store",
);
const oddIds = ["a b", "a%20b", "a:b", "a/b", "a", "a%3Ab", "a|pane:x"];
check(
  new Set(oddIds.map((id) => pythonWorkspaceStorageKey(id))).size === oddIds.length,
  "ids that differ only in punctuation get distinct stores",
);

/*
 * The encoding is not what makes the map injective -- raw concatenation would
 * be too -- so it is not tested by asserting distinctness. What it is FOR is
 * recovering the id from the key, which is the only reason orphan pruning can
 * work; that is the property asserted here and below.
 */
for (const id of oddIds) {
  const key = pythonWorkspaceStorageKey(id);
  check(
    decodeURIComponent(key.slice(`${PYTHON_WORKSPACE_STORAGE_KEY}:`.length)) === id,
    `the project id round-trips out of its store key (${JSON.stringify(id)})`,
  );
}

// ---------- 9b. Deleting a project must not strand its store ----------

const liveIds = new Set(["project_alpha", "project_beta"]);
const presentKeys = [
  PYTHON_WORKSPACE_STORAGE_KEY,
  pythonWorkspaceStorageKey("project_alpha"),
  pythonWorkspaceStorageKey("project_gone"),
  pythonWorkspaceStorageKey("a b"),
  "nimbus-studio:python-gallery:open",
  "unrelated-app:state",
];
const orphans = orphanedPythonWorkspaceKeys(presentKeys, liveIds);
check(
  orphans.length === 2
    && orphans.includes(pythonWorkspaceStorageKey("project_gone"))
    && orphans.includes(pythonWorkspaceStorageKey("a b")),
  "stores for projects that no longer exist are reported",
);
check(
  !orphans.includes(PYTHON_WORKSPACE_STORAGE_KEY),
  "the unscoped store is never orphaned -- no-project is always reachable",
);
check(
  !orphans.includes(pythonWorkspaceStorageKey("project_alpha")),
  "a live project's store is left alone",
);
check(
  !orphans.some((key) => !key.startsWith(PYTHON_WORKSPACE_STORAGE_KEY)),
  "keys this module did not write are never proposed for deletion",
);
/*
 * A malformed percent sequence makes decodeURIComponent throw. Treating that as
 * an orphan would delete a key on the strength of not being able to read it.
 */
check(
  orphanedPythonWorkspaceKeys(
    [`${PYTHON_WORKSPACE_STORAGE_KEY}:%zz`],
    new Set<string>(),
  ).length === 0,
  "an undecodable key is left alone rather than deleted",
);
check(
  orphanedPythonWorkspaceKeys([], liveIds).length === 0,
  "no stored keys means nothing to prune",
);

// ---------- 9c. One store per pane ----------

/*
 * Two Python tabs open side by side are two editors. Sharing one store meant
 * each pane's debounced write overwrote the other's buffers, so the pane you
 * were not looking at silently lost whatever you had typed in it.
 */
check(
  pythonWorkspaceStorageKey(null, PRIMARY_PANE_ID) === PYTHON_WORKSPACE_STORAGE_KEY,
  "the primary pane's unscoped key is unchanged by panes existing",
);
check(
  pythonWorkspaceStorageKey("project_a", PRIMARY_PANE_ID)
    === pythonWorkspaceStorageKey("project_a"),
  "the primary pane is what the one-argument form means",
);
/*
 * The single most important assertion in this section. The primary pane is
 * every tab that has never been split, so a key that changed shape here would
 * strand the unsaved buffers of every existing user on upgrade -- invisibly,
 * because the new key reads as an empty workspace rather than as an error.
 */
check(
  pythonWorkspaceStorageKey("project_a", PRIMARY_PANE_ID)
    === `${PYTHON_WORKSPACE_STORAGE_KEY}:project_a`,
  "the primary pane's key is byte-identical to the pre-pane key",
);
check(
  pythonWorkspaceStorageKey("project_a", "companion-0")
    !== pythonWorkspaceStorageKey("project_a", PRIMARY_PANE_ID),
  "a companion pane does not share the primary pane's store",
);
check(
  pythonWorkspaceStorageKey("project_a", "companion-0")
    !== pythonWorkspaceStorageKey("project_a", "companion-1"),
  "two companion panes get two stores",
);
check(
  pythonWorkspaceStorageKey("project_a", "companion-0")
    !== pythonWorkspaceStorageKey("project_b", "companion-0"),
  "the same pane in two projects gets two stores",
);
check(
  pythonWorkspaceStorageKey(null, "companion-0") !== PYTHON_WORKSPACE_STORAGE_KEY,
  "a companion pane with no project does not fall back to the unscoped store",
);

/*
 * The separator has to be un-forgeable, because the sweep below decides what to
 * DELETE by parsing it back out. A project literally named `a|pane:companion-0`
 * must not produce the key that project `a`'s companion pane writes to.
 */
check(
  pythonWorkspaceStorageKey("a|pane:companion-0", PRIMARY_PANE_ID)
    !== pythonWorkspaceStorageKey("a", "companion-0"),
  "a project id containing the separator cannot forge another project's pane key",
);

const paneKeys = [
  PYTHON_WORKSPACE_STORAGE_KEY,
  pythonWorkspaceStorageKey(null, "companion-0"),
  pythonWorkspaceStorageKey("project_alpha"),
  pythonWorkspaceStorageKey("project_alpha", "companion-0"),
  pythonWorkspaceStorageKey("project_alpha", "companion-1"),
  pythonWorkspaceStorageKey("project_gone"),
  pythonWorkspaceStorageKey("project_gone", "companion-0"),
  pythonWorkspaceStorageKey("a|pane:companion-0"),
];
const paneOrphans = orphanedPythonWorkspaceKeys(paneKeys, liveIds);
/*
 * The regression this section exists for. Reading everything after the project
 * separator as the project id makes `…:project_alpha|pane:companion-0` decode
 * to a name no live project can match -- so a live project's second pane would
 * be swept away, taking its unsaved text with it.
 */
check(
  !paneOrphans.includes(pythonWorkspaceStorageKey("project_alpha", "companion-0")),
  "a live project's companion store is not mistaken for an orphan",
);
check(
  paneOrphans.includes(pythonWorkspaceStorageKey("project_gone", "companion-0")),
  "a dead project's companion store is swept with the rest of it",
);
check(
  !paneOrphans.includes(pythonWorkspaceStorageKey(null, "companion-0")),
  "an unscoped companion store is never orphaned -- no-project is reachable in any pane",
);
check(
  paneOrphans.includes(pythonWorkspaceStorageKey("a|pane:companion-0")),
  "a dead project whose name contains the separator is still swept",
);

const alphaKeys = pythonWorkspaceKeysForProject(paneKeys, "project_alpha");
check(alphaKeys.length === 3, "every pane of a project is reported for deletion");
check(
  alphaKeys.includes(pythonWorkspaceStorageKey("project_alpha"))
    && alphaKeys.includes(pythonWorkspaceStorageKey("project_alpha", "companion-1")),
  "the primary and the companions are all included",
);
check(
  !alphaKeys.includes(PYTHON_WORKSPACE_STORAGE_KEY)
    && !alphaKeys.includes(pythonWorkspaceStorageKey("project_gone")),
  "no other project's store is caught by a delete",
);

/*
 * Where a store that does not exist yet gets its open files from. Each inherits
 * from the nearest thing it is a variation of, so opening a new pane or a new
 * project reads as opening a view of your work rather than as having closed it.
 */
check(
  pythonWorkspaceSeedKey(null, PRIMARY_PANE_ID) === null,
  "the unscoped primary pane has nothing above it to inherit from",
);
check(
  pythonWorkspaceSeedKey("project_a", PRIMARY_PANE_ID) === PYTHON_WORKSPACE_STORAGE_KEY,
  "a project's first pane inherits from the unscoped store",
);
check(
  pythonWorkspaceSeedKey("project_a", "companion-0")
    === pythonWorkspaceStorageKey("project_a", PRIMARY_PANE_ID),
  "a companion pane inherits from its OWN project, not from the unscoped store",
);
check(
  pythonWorkspaceSeedKey(null, "companion-0") === PYTHON_WORKSPACE_STORAGE_KEY,
  "an unscoped companion pane inherits from the unscoped primary",
);
check(
  pythonWorkspaceSeedKey("project_a", "companion-0")
    !== pythonWorkspaceStorageKey("project_a", "companion-0"),
  "no store ever seeds from itself, which would make seeding a no-op",
);

// ---------- 10. Seeding a project's first workspace ----------

/*
 * What carries into a project used for scripting the first time. Files carry as
 * PATHS; scratch does not carry at all. Copying unsaved text would fork it into
 * two independently editable copies of one script with no way to tell them
 * apart or merge them -- strictly worse than leaving it in the store it was
 * written in, where it is still reachable.
 */
let sourceWorkspace = createWorkspace(STARTER);
const scratchId = sourceWorkspace.activeId;
sourceWorkspace = setBufferCode(sourceWorkspace, scratchId, "# unsaved scratch\n");
sourceWorkspace = openPathInWorkspace(sourceWorkspace, "/w/clean.py", "print('clean')\n").workspace;
sourceWorkspace = openPathInWorkspace(sourceWorkspace, "/w/dirty.py", "print('disk')\n").workspace;
const dirtyId = sourceWorkspace.buffers.find((b) => b.path === "/w/dirty.py")!.id;
sourceWorkspace = setBufferCode(sourceWorkspace, dirtyId, "print('unsaved edit')\n");
/*
 * Leave the SCRATCH buffer focused. Opening a file focuses it, so without this
 * the source's active buffer is always one that survives seeding and the
 * fallback below is never exercised -- which is exactly what a mutation of the
 * fallback proved, by passing.
 */
sourceWorkspace = activateBuffer(sourceWorkspace, scratchId);

const seeded = seedProjectWorkspace(serializeWorkspace(sourceWorkspace), STARTER);
check(seeded.buffers.length === 2, "only the file-backed buffers seed the project");
check(
  seeded.buffers.every((entry) => entry.path !== null),
  "no scratch buffer is copied into a project",
);
check(
  seeded.buffers.map((entry) => entry.path).join() === "/w/clean.py,/w/dirty.py",
  "the seeded paths are the files that were open, in order",
);
check(
  seeded.buffers.every((entry) => entry.needsReload && entry.code === ""),
  "seeded buffers carry no text and are re-read from disk",
);
check(
  !seeded.buffers.some((entry) => entry.code.includes("unsaved edit")),
  "an unsaved edit is not forked into the project's store",
);
check(!workspaceHasUnsaved(seeded), "a seeded workspace reports nothing unsaved");
check(
  seeded.buffers.some((entry) => entry.id === seeded.activeId),
  "the seeded workspace focuses a buffer it actually contains",
);

/*
 * The active buffer in the source is scratch, which does not survive seeding.
 * Carrying its id across would leave `activeId` naming a buffer that is not in
 * the workspace, and `activeBuffer` returns `buffers[0]` in that case -- so the
 * tab strip would render with no tab marked active while the editor showed the
 * first file. Falling back explicitly keeps the two agreeing.
 */
check(
  seeded.activeId !== scratchId,
  "the dropped scratch buffer does not stay named as the active one",
);
check(
  activeBuffer(seeded).id === seeded.activeId && activeBuffer(seeded).path !== null,
  "a dropped active buffer falls back to a surviving one",
);

check(
  seedProjectWorkspace(null, STARTER).buffers[0].code === STARTER,
  "nothing to seed from yields the starter workspace",
);
const scratchOnly = setBufferCode(createWorkspace(STARTER), "buf_1", "# only scratch\n");
const fromScratchOnly = seedProjectWorkspace(serializeWorkspace(scratchOnly), STARTER);
check(
  fromScratchOnly.buffers.length === 1 && fromScratchOnly.buffers[0].path === null
    && fromScratchOnly.buffers[0].code === STARTER,
  "a source holding only scratch seeds a fresh starter buffer, not an empty strip",
);

/*
 * Round-trip: a seeded workspace written back out stores paths only. If seeding
 * ever started carrying text, every project store would grow by the size of the
 * open files and this is where it would show.
 */
const seededRecord = JSON.parse(serializeWorkspace(seeded)) as { buffers: { code?: string }[] };
check(
  seededRecord.buffers.every((entry) => entry.code === undefined),
  "a seeded workspace persists as paths only",
);

console.log(`\npython workspace tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
