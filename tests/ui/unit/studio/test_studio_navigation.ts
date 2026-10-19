import {
  STUDIO_NAVIGATION_STORAGE_KEY,
  defaultArchiveRunId,
  loadStudioNavigation,
  preferredCaseRunId,
  reconcileStudioNavigationRuns,
  saveStudioNavigation,
} from "../../../../ui/src/lib/studio/studioNavigation.ts";

let passed = 0;
const check = (condition: boolean, label: string) => {
  if (!condition) throw new Error(label);
  passed += 1;
};

class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const storage = new MemoryStorage();
check(loadStudioNavigation(storage).page === "config", "missing state opens the safe Configure destination");
check(loadStudioNavigation(storage).selectedRunId === null, "missing state does not invent a run selection");

saveStudioNavigation(storage, { page: "projects", selectedRunId: "run-017" });
const restored = loadStudioNavigation(storage);
check(restored.page === "projects", "Projects is a durable primary destination");
check(restored.selectedRunId === "run-017", "bounded run identity survives restart");

storage.setItem(STUDIO_NAVIGATION_STORAGE_KEY, "not json");
check(loadStudioNavigation(storage).page === "config", "corrupt state fails back to Configure");

storage.setItem(STUDIO_NAVIGATION_STORAGE_KEY, JSON.stringify({
  schema: "nimbus.studio.navigation",
  schemaVersion: 1,
  page: "missing",
  selectedRunId: "run-017",
}));
check(loadStudioNavigation(storage).page === "config", "unknown destinations fail closed");

storage.setItem(STUDIO_NAVIGATION_STORAGE_KEY, JSON.stringify({
  schema: "nimbus.studio.navigation",
  schemaVersion: 1,
  page: "runs",
  selectedRunId: "x".repeat(257),
}));
check(loadStudioNavigation(storage).page === "runs", "a valid destination survives an invalid optional selection");
check(loadStudioNavigation(storage).selectedRunId === null, "oversized run identity is rejected");

const unchanged = reconcileStudioNavigationRuns({ page: "runs", selectedRunId: "run-017" }, ["run-017", "run-018"]);
check(unchanged.selectedRunId === "run-017", "available run selection remains stable");
const removed = reconcileStudioNavigationRuns(unchanged, ["run-018"]);
check(removed.page === "runs" && removed.selectedRunId === null, "missing run clears selection without changing destination");

const throwing = {
  getItem() { throw new Error("blocked"); },
  setItem() { throw new Error("blocked"); },
};
check(loadStudioNavigation(throwing).page === "config", "blocked reads retain a usable default");
saveStudioNavigation(throwing, { page: "plots", selectedRunId: null });
check(true, "blocked writes do not break in-memory navigation");

/* ---------- Default archive selection ----------
 * Arriving in Run Archive with no remembered selection used to render the
 * "No run selected" placeholder even while a simulation was actively running.
 * The chosen run must be the one the user came to watch, and the rule has to
 * stay stable when timestamps are missing on queued/mock records. */

const run = (id: string, status: string, startedAtMs: number | null) => ({ id, status, startedAtMs });

check(defaultArchiveRunId([]) === null, "an empty archive selects nothing");
check(
  defaultArchiveRunId([
    run("done-new", "completed", 900),
    run("live", "running", 100),
    run("queued", "queued", 500),
  ]) === "live",
  "a running run outranks newer queued and completed records",
);
check(
  defaultArchiveRunId([run("done", "completed", 900), run("queued", "queued", 100)]) === "queued",
  "a queued run outranks a completed one",
);
check(
  defaultArchiveRunId([
    run("old", "completed", 100),
    run("new", "failed", 900),
    run("mid", "aborted", 500),
  ]) === "new",
  "a terminal-only archive falls back to the newest record whatever its outcome",
);
check(
  defaultArchiveRunId([run("live-old", "running", 100), run("live-new", "running", 900)]) === "live-new",
  "the newest run wins among equally ranked statuses",
);
check(
  defaultArchiveRunId([run("timeless", "queued", null), run("stamped", "queued", 10)]) === "stamped",
  "a missing start timestamp sorts last instead of poisoning the comparison",
);
check(
  defaultArchiveRunId([run("only", "completed", null)]) === "only",
  "a lone record with no timestamp is still selectable",
);

const archived = (
  id: string,
  caseName: string,
  startedAtMs: number | null,
  libraryHealth: "ready" | "partial" | "invalid",
) => ({ id, caseName, startedAtMs, modifiedAtMs: startedAtMs, libraryHealth, outputDir: `/runs/${id}` });
check(
  preferredCaseRunId([
    archived("other", "manufactured_supercell", 999, "ready"),
    archived("beast-old", "beast_xl", 100, "partial"),
    archived("beast-new", "BEAST_XL", 500, "partial"),
  ], "beast_xl") === "beast-new",
  "cold-start case preference selects the newest matching readable archive",
);
check(
  preferredCaseRunId([
    archived("beast-invalid", "beast_xl", 900, "invalid"),
    archived("beast-readable", "beast_xl", 100, "partial"),
  ], "beast_xl") === "beast-readable",
  "cold-start case preference refuses an invalid matching archive",
);
check(
  preferredCaseRunId([archived("other", "manufactured_supercell", 100, "ready")], "beast_xl") === null,
  "cold-start case preference does not substitute another storm",
);

console.log(`studio navigation: ${passed} checks passed`);
