/*
 * Tests for src/lib/concurrency.ts.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_concurrency.ts
 *
 * Guards the bounded task pool the time-series / time-height plots use to
 * overlap per-step reductions: every index settles, concurrency is actually
 * bounded, an early stop prevents new work and drops late settles, and task
 * failures are delivered (not thrown).
 */

import { mapPool } from "../../../../ui/src/lib/core/concurrency.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function main(): Promise<void> {
  // ---------- every index is processed; values are returned ----------
  {
    const seen: number[] = [];
    await mapPool(
      5,
      2,
      async (i) => i * 10,
      (_i, res) => {
        if (res.ok) seen.push(res.value);
        return true;
      },
    );
    seen.sort((a, b) => a - b);
    assertTrue(
      seen.length === 5 && seen[0] === 0 && seen[4] === 40,
      "all indices processed, values returned",
    );
  }

  // ---------- concurrency is bounded but does overlap ----------
  {
    let inFlight = 0;
    let maxInFlight = 0;
    await mapPool(
      10,
      3,
      async () => {
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await delay(10);
        inFlight--;
      },
      () => true,
    );
    assertTrue(maxInFlight <= 3, `concurrency bounded (max in flight ${maxInFlight} <= 3)`);
    assertTrue(maxInFlight >= 2, `concurrency overlaps (max in flight ${maxInFlight} >= 2)`);
  }

  // ---------- stop early: no new tasks start, late settles dropped ----------
  {
    let started = 0;
    let settles = 0;
    await mapPool(
      20,
      2,
      async (i) => {
        started++;
        await delay(5);
        return i;
      },
      () => {
        settles++;
        return false; // stop on the first settle
      },
    );
    assertTrue(started <= 2, `stop prevents new tasks (started ${started} <= 2)`);
    assertTrue(settles === 1, `stop drops the in-flight worker's late settle (settles ${settles} === 1)`);
  }

  // ---------- task failures are delivered, not thrown ----------
  {
    let okCount = 0;
    let errCount = 0;
    await mapPool(
      4,
      4,
      async (i) => {
        if (i === 2) throw new Error("boom");
        return i;
      },
      (_i, res) => {
        if (res.ok) okCount++;
        else errCount++;
        return true;
      },
    );
    assertTrue(errCount === 1, "task error delivered as { ok: false }");
    assertTrue(okCount === 3, "the other tasks still settle ok");
  }

  // ---------- count <= 0 is a no-op ----------
  {
    let calls = 0;
    await mapPool(
      0,
      4,
      async () => {
        calls++;
      },
      () => true,
    );
    assertTrue(calls === 0, "count <= 0 runs no tasks");
  }

  // ---------- onSettled receives the matching index ----------
  {
    const pairs: Array<[number, number]> = [];
    await mapPool(
      3,
      1,
      async (i) => i * 100,
      (i, res) => {
        if (res.ok) pairs.push([i, res.value]);
        return true;
      },
    );
    assertTrue(
      pairs.length === 3 && pairs.every(([i, v]) => v === i * 100),
      "onSettled receives the matching index",
    );
  }

  console.log(`\nconcurrency tests: ${passed} passed, ${failed} failed.`);
  if (failures.length > 0) {
    console.log("\nFailures:");
    for (const f of failures) console.log("  " + f);
    process.exit(1);
  }
  process.exit(0);
}

main();
