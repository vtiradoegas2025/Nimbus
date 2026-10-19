/*
 * A run's output, held where the run can still reach it.
 *
 * Studio routes with a switch, so leaving the Plots page unmounts the panel —
 * while the interpreter, which is a module-level worker, carries on. When the
 * results were React state, a cell started before navigating away delivered its
 * figure to a component that no longer existed and coming back showed an empty
 * panel. It looked like switching tabs had killed the run. It had not; only the
 * destination was gone.
 *
 * The properties below are what make the store a valid destination: a write
 * lands whether or not anyone is listening, a reader gets a stable snapshot,
 * and eviction can never take a session whose run is still in flight — that
 * last one would reintroduce the original bug through the back door.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_session.ts
 */

import {
  patchSession,
  readSession,
  resetSessions,
  sessionKey,
  subscribeSession,
} from "../../../../ui/src/lib/studio/pythonSession.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

resetSessions();

// ---------- 1. Identity ----------

check(
  sessionKey("proj", "primary") !== sessionKey("proj", "companion-0"),
  "two panes of one project are two sessions",
);
check(
  sessionKey("a", "primary") !== sessionKey("b", "primary"),
  "the same pane in two projects is two sessions",
);
check(
  sessionKey(null, "primary") === sessionKey(null, "primary"),
  "no project is a stable identity, not a fresh one each call",
);

// ---------- 2. A write lands with nobody listening ----------

const key = sessionKey(null, "primary");
check(readSession(key).running === false, "an unknown session starts idle");
check(readSession(key).completedRuns === 0, "and with no runs behind it");
check(readSession(key).outputBufferId === null, "and no script owns output yet");

patchSession(key, { running: true });
check(readSession(key).running, "a write with no subscriber is still recorded");

// This is the unmounted-run case in miniature: the run finishes long after the
// component that started it is gone, and its result must survive to be read.
patchSession(key, { running: false, completedRuns: 1, ioStatus: "done" });
const afterRun = readSession(key);
check(
  !afterRun.running && afterRun.completedRuns === 1 && afterRun.ioStatus === "done",
  "the result of a run nobody was watching is there to be picked up",
);
patchSession(key, { outputBufferId: "buf_2" });
check(
  readSession(key).outputBufferId === "buf_2",
  "retained output records the buffer that produced it",
);
patchSession(key, {
  stagedFiles: [{
    hostPath: "/data/observations.npy",
    virtualPath: "/mnt/nimbus/observations.npy",
    name: "observations.npy",
    byteLength: 64,
  }],
});
check(
  readSession(key).stagedFiles[0]?.name === "observations.npy",
  "staged-file metadata survives while the pane component is unmounted",
);

// ---------- 3. Snapshots are stable ----------

check(
  readSession(key) === readSession(key),
  "two reads return the same object, or useSyncExternalStore loops forever",
);
const before = readSession(key);
patchSession(key, { ioStatus: "done" });
check(readSession(key) === before, "a patch that changes nothing changes no identity");
patchSession(key, { ioStatus: "different" });
check(readSession(key) !== before, "a patch that changes something does");

// ---------- 4. Notification ----------

let notifications = 0;
const unsubscribe = subscribeSession(key, () => { notifications += 1; });
patchSession(key, { running: true });
check(notifications === 1, "a subscriber hears a real change");
patchSession(key, { running: true });
check(notifications === 1, "and is not woken for a write of the same value");
patchSession(key, (previous) => ({ completedRuns: previous.completedRuns + 1 }));
check(notifications === 2, "an updater form reads the previous value and notifies");
check(readSession(key).completedRuns === 2, "and applies to it");

unsubscribe();
patchSession(key, { running: false });
check(notifications === 2, "an unsubscribed listener stops hearing anything");

// ---------- 5. Eviction never takes a live run ----------

resetSessions();
const live = sessionKey("live-project", "primary");
patchSession(live, { running: true, completedRuns: 7 });
for (let i = 0; i < 40; i += 1) {
  patchSession(sessionKey(`filler-${i}`, "primary"), { completedRuns: i });
}
const survivor = readSession(live);
check(
  survivor.running && survivor.completedRuns === 7,
  "a session with a run in flight survives any amount of churn beside it",
);

// And the store does not grow without bound for the sessions that are idle.
resetSessions();
for (let i = 0; i < 40; i += 1) {
  patchSession(sessionKey(`cold-${i}`, "primary"), { completedRuns: i });
}
let remembered = 0;
for (let i = 0; i < 40; i += 1) {
  if (readSession(sessionKey(`cold-${i}`, "primary")).completedRuns !== 0) remembered += 1;
}
check(remembered < 40, "idle sessions are eventually forgotten rather than accumulating");
check(
  readSession(sessionKey("cold-39", "primary")).completedRuns === 39,
  "and the most recent one is not the one thrown away",
);

resetSessions();
check(readSession(key).completedRuns === 0, "reset clears everything, for the next test");

console.log(`\npython session tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
