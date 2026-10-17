/*
 * What Stop does, for every state it can be pressed in.
 *
 * The property under test is that Stop is never decorative. It used to be
 * exactly one move — write the interrupt byte — which reaches only code that is
 * executing Python bytecode, and on a window without cross-origin isolation
 * reaches nothing at all. Pressed against a numpy kernel, a wheel download or a
 * non-isolated page it did nothing, said nothing, and left Restart disabled
 * because a run was in flight.
 *
 * So each case below asks the same question in a different state: does this
 * press end with the cell stopped? Two answers are acceptable — interrupt now,
 * or take the interpreter — and the cases that answer "interrupt" must also
 * arm the escalation that makes the first answer eventually become the second.
 * A decision that neither stops the cell nor schedules something that will is
 * the bug, in any state.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_stop.ts
 */

import {
  decideStop,
  STOP_GRACE_MS,
  STOP_STATUS_FORCED,
  STOP_STATUS_INTERRUPTED,
  STOP_STATUS_INTERRUPTING,
  STOP_STATUS_NO_INTERRUPT,
  stopButtonLabel,
  stopButtonTitle,
  type StopStage,
} from "../../../../ui/src/lib/studio/pythonStop.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const STAGES: readonly StopStage[] = ["idle", "interrupting"];
const EVERY_CASE = STAGES.flatMap((stage) =>
  [true, false].map((canInterrupt) => ({ stage, canInterrupt })),
);

// ---------- 1. The polite route, when there is one ----------

const first = decideStop({ stage: "idle", canInterrupt: true });
check(first.action === "interrupt", "the first press on an isolated page interrupts");
check(
  first.escalateAfterMs === STOP_GRACE_MS,
  "an interrupt arms the escalation that makes it eventually forceful",
);
check(first.nextStage === "interrupting", "an interrupt moves to the interrupting stage");
check(first.status === STOP_STATUS_INTERRUPTING, "an interrupt says it is waiting");

// ---------- 2. No polite route: skip straight to the effective one ----------

const unisolated = decideStop({ stage: "idle", canInterrupt: false });
check(
  unisolated.action === "terminate",
  "without a shared interrupt byte the first press takes the interpreter",
);
check(
  unisolated.status === STOP_STATUS_NO_INTERRUPT,
  "and says isolation is why, rather than reporting a mystery",
);
check(
  unisolated.escalateAfterMs === null,
  "a terminate schedules nothing further: there is nothing left to escalate to",
);

// ---------- 3. The cell ignored the interrupt ----------

const forced = decideStop({ stage: "interrupting", canInterrupt: true });
check(forced.action === "terminate", "a second press, or the timer, forces");
check(forced.status === STOP_STATUS_FORCED, "and reports that the interrupt was ignored");
check(
  forced.nextStage === "idle",
  "the stage resets with the interpreter it described, so the NEXT run's first "
  + "Stop is polite again rather than inheriting a dead session's impatience",
);

// ---------- 4. The invariant, over every reachable state ----------

for (const input of EVERY_CASE) {
  const decision = decideStop(input);
  const name = `${input.stage}/${input.canInterrupt ? "isolated" : "unisolated"}`;
  check(
    decision.action === "terminate" || decision.escalateAfterMs !== null,
    `${name}: a press either stops the cell now or schedules the stop that will`,
  );
  check(
    decision.action !== "terminate" || decision.escalateAfterMs === null,
    `${name}: a terminate leaves no timer to fire against the next interpreter`,
  );
  check(decision.status.trim().length > 0, `${name}: every outcome is reported`);
}

// A press from idle reaches a terminate in at most two moves, whatever the
// page's isolation. Anything longer is a Stop the user has to keep pressing.
let stage: StopStage = "idle";
let moves = 0;
for (;;) {
  const decision = decideStop({ stage, canInterrupt: true });
  moves += 1;
  if (decision.action === "terminate") break;
  stage = decision.nextStage;
  if (moves > 4) break;
}
check(moves === 2, "an isolated page escalates to terminate on the second move");

// ---------- 5. What the button says is what the press will do ----------

check(stopButtonLabel("idle") === "Stop", "the first press is labelled Stop");
check(
  stopButtonLabel("interrupting") === "Force stop",
  "once interrupting, the label names the harsher thing the next press does",
);
check(
  stopButtonTitle("idle", true) !== stopButtonTitle("idle", false),
  "the tooltip distinguishes an interrupt from a replacement",
);
for (const input of EVERY_CASE) {
  check(
    stopButtonTitle(input.stage, input.canInterrupt).trim().length > 0,
    `${input.stage}/${input.canInterrupt}: the button explains itself`,
  );
}

// ---------- 6. The cost is stated exactly where it is paid ----------

const COST_WORDS = ["Variables", "imports"];
for (const [name, status] of [
  ["forced", STOP_STATUS_FORCED],
  ["unisolated", STOP_STATUS_NO_INTERRUPT],
] as const) {
  check(
    COST_WORDS.every((word) => status.includes(word)),
    `${name}: a status that took the interpreter says what was lost with it`,
  );
}
check(
  !COST_WORDS.some((word) => STOP_STATUS_INTERRUPTED.includes(`${word} are gone`)),
  "a cell that merely took its interrupt is not reported as having lost the namespace",
);
check(
  STOP_STATUS_INTERRUPTING.includes(String(STOP_GRACE_MS / 1000)),
  "the waiting message names the grace period the user is waiting out",
);
check(
  STOP_GRACE_MS > 0 && STOP_GRACE_MS < 10_000,
  "the grace period is long enough to be a grace and short enough to be a wait",
);

console.log(`\npython stop tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
