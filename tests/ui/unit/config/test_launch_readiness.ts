/*
 * Tests for src/lib/config/launchReadiness.ts.
 *
 * The regression: ConfigEditor carried a boolean `launchBlocked` with six
 * disjuncts alongside a tooltip ternary covering three of them. A resume
 * checkpoint that failed to inspect satisfied the boolean but none of the
 * tooltip branches, so the Run button went dead with an EMPTY tooltip and its
 * only explanation sat inside a section that is collapsed by default.
 *
 * The invariant that prevents a recurrence is exhaustive rather than
 * example-based: over the full cross product of inputs, a blocked action must
 * always have a non-empty reason. Several of these states cannot be reached
 * from a browser harness at all, which is why this is a unit test.
 *
 * Run via:
 *   cd ui && npm run test:launch-readiness
 */

import {
  describeCheckpointBlock,
  describeLaunchBlock,
  describeValidationBlock,
  type CheckpointStatus,
  type LaunchReadinessInputs,
} from "../../../../ui/src/lib/config/launchReadiness.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function check(label: string, condition: boolean): void {
  if (condition) {
    passed++;
  } else {
    failed++;
    failures.push(label);
  }
}

/** Everything ready to go; individual tests perturb one axis at a time. */
const READY: LaunchReadinessInputs = {
  desktop: true,
  catalogKind: "ready",
  preflightKind: "required",
  spawnKind: "idle",
  resumeCheckpointPath: "",
  checkpoint: { kind: "empty" },
};

/* ---------- The ready state blocks nothing ---------- */

check("a ready configuration can launch", describeLaunchBlock(READY) === null);
check("a ready configuration can validate", describeValidationBlock(READY) === null);

/* ---------- Each blocking condition produces a reason ---------- */

const BLOCKING: Array<[string, Partial<LaunchReadinessInputs>]> = [
  ["browser runtime", { desktop: false }],
  ["catalog not ready", { catalogKind: "loading" }],
  ["catalog errored", { catalogKind: "error" }],
  ["preflight pending", { preflightKind: "pending" }],
  ["preflight blocked", { preflightKind: "blocked" }],
  ["spawn starting", { spawnKind: "starting" }],
  [
    "checkpoint still inspecting",
    { resumeCheckpointPath: "/runs/a/chk", checkpoint: { kind: "checking", path: "/runs/a/chk" } },
  ],
  [
    "checkpoint unreadable",
    {
      resumeCheckpointPath: "/runs/a/chk",
      checkpoint: { kind: "error", path: "/runs/a/chk", message: "no such file" },
    },
  ],
  [
    "checkpoint never inspected",
    { resumeCheckpointPath: "/runs/a/chk", checkpoint: { kind: "empty" } },
  ],
  [
    "checkpoint accepted for a DIFFERENT path",
    {
      resumeCheckpointPath: "/runs/b/chk",
      checkpoint: { kind: "accepted", requestedPath: "/runs/a/chk" },
    },
  ],
];

for (const [label, patch] of BLOCKING) {
  const reason = describeLaunchBlock({ ...READY, ...patch });
  check(`${label} blocks the launch`, reason !== null);
  check(`${label} states a non-empty reason`, (reason ?? "").trim().length > 0);
}

/* ---------- The exhaustive guard: blocked implies explained ---------- */

const CHECKPOINTS: CheckpointStatus[] = [
  { kind: "empty" },
  { kind: "checking", path: "/runs/a/chk" },
  { kind: "accepted", requestedPath: "/runs/a/chk" },
  { kind: "accepted", requestedPath: "/runs/other/chk" },
  { kind: "error", path: "/runs/a/chk", message: "no such file" },
];

let combinations = 0;
let unexplained = 0;
for (const desktop of [true, false]) {
  for (const catalogKind of ["ready", "loading", "error"]) {
    for (const preflightKind of ["required", "pending", "blocked", "satisfied"]) {
      for (const spawnKind of ["idle", "starting", "running"]) {
        for (const resumeCheckpointPath of ["", "/runs/a/chk"]) {
          for (const checkpoint of CHECKPOINTS) {
            const inputs: LaunchReadinessInputs = {
              desktop,
              catalogKind,
              preflightKind,
              spawnKind,
              resumeCheckpointPath,
              checkpoint,
            };
            combinations += 1;
            for (const describe of [describeLaunchBlock, describeValidationBlock]) {
              const reason = describe(inputs);
              // The whole point: a null reason means "not blocked". There is no
              // way to express "blocked but unexplained" in this API.
              if (reason !== null && reason.trim().length === 0) unexplained += 1;
            }
          }
        }
      }
    }
  }
}
check(`the cross product is non-trivial (${combinations} combinations)`, combinations === 720);
check("no combination yields a blank reason", unexplained === 0);

/* ---------- Ordering and scoping rules ---------- */

// Validation is how the user CLEARS blocking preflight findings, so it must not
// itself be blocked by them, nor by the resume checkpoint.
check(
  "blocking preflight findings do not block validation",
  describeValidationBlock({ ...READY, preflightKind: "blocked" }) === null,
);
check(
  "an unreadable checkpoint does not block validation",
  describeValidationBlock({
    ...READY,
    resumeCheckpointPath: "/runs/a/chk",
    checkpoint: { kind: "error", path: "/runs/a/chk", message: "no such file" },
  }) === null,
);

// A checkpoint the user has since cleared must stop blocking.
check(
  "clearing the resume path releases a stale checkpoint block",
  describeCheckpointBlock("", { kind: "error", path: "/runs/a/chk", message: "boom" }) === null,
);
check(
  "an accepted checkpoint for the requested path does not block",
  describeCheckpointBlock("/runs/a/chk", { kind: "accepted", requestedPath: "/runs/a/chk" }) === null,
);

// The backend's message is the actionable part and must survive into the copy.
const unreadable = describeLaunchBlock({
  ...READY,
  resumeCheckpointPath: "/runs/a/chk",
  checkpoint: { kind: "error", path: "/runs/a/chk", message: "no such file or directory" },
});
check(
  "the checkpoint failure carries the underlying message",
  (unreadable ?? "").includes("no such file or directory"),
);

// Environment first: a browser context explains itself rather than reporting a
// downstream symptom the user cannot act on.
check(
  "browser runtime outranks the catalog state",
  describeLaunchBlock({ ...READY, desktop: false, catalogKind: "loading" })?.includes("desktop") === true,
);

console.log(`\nlaunch-readiness tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
