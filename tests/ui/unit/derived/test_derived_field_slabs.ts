/*
 * Tests for slabbed derived-field evaluation.
 *
 * The split is only safe because every built-in derived field is elementwise, so the decisive
 * test is not that the plumbing runs but that slabbing changes no value: a whole-field result
 * and a reassembled slabbed result must agree bit for bit.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/derived/test_derived_field_slabs.ts
 */

import {
  DERIVED_SLAB_CELLS,
  planSlabs,
  sliceInputs,
  slabRequestBytes,
} from "../../../../ui/src/lib/derived/derivedFieldSlabs.ts";
import { evaluateDerivedField } from "../../../../ui/src/lib/derived/derivedFieldRegistry.ts";
import { evaluateDerivedFieldInWorker } from "../../../../ui/src/lib/derived/derivedFieldWorkerClient.ts";
import {
  evaluateDerivedWorkerRequest,
  type DerivedFieldWorkerReply,
  type DerivedFieldWorkerRequest,
} from "../../../../ui/src/lib/derived/derivedFieldWorkerProtocol.ts";
import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";

// ---------- Tiny assertion API (matches test_pane_layout.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

async function check(label: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

/** A small but physically plausible column stack, so the maths exercises real branches. */
function makeInputs(cells: number): Record<string, NpyN> {
  const theta = new Float32Array(cells);
  const p = new Float32Array(cells);
  const qv = new Float32Array(cells);
  const u = new Float32Array(cells);
  const v = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    theta[i] = 300 + (i % 50) * 0.7;
    p[i] = 100_000 - (i % 160) * 500;
    qv[i] = 0.001 + (i % 30) * 0.0004;
    u[i] = -20 + (i % 41);
    v[i] = 15 - (i % 33);
  }
  return {
    theta: { shape: [cells], data: theta },
    p: { shape: [cells], data: p },
    qv: { shape: [cells], data: qv },
    u: { shape: [cells], data: u },
    v: { shape: [cells], data: v },
  };
}

const DERIVED = ["temperature", "dewpoint", "relative_humidity", "theta_e", "horizontal_wind_speed"];

// ---------- Slab planning ----------

await check("slabs tile the field exactly once", () => {
  for (const [length, size] of [[10, 3], [9, 3], [1, 5], [2_000_000, 2_000_000]]) {
    const slabs = planSlabs(length, size);
    assertTrue(slabs[0].start === 0, `${length}/${size} starts at 0`);
    assertTrue(slabs[slabs.length - 1].end === length, `${length}/${size} ends at the length`);
    for (let i = 1; i < slabs.length; i++) {
      assertTrue(slabs[i].start === slabs[i - 1].end, `${length}/${size} slab ${i} is contiguous`);
    }
    const covered = slabs.reduce((sum, s) => sum + (s.end - s.start), 0);
    assertTrue(covered === length, `${length}/${size} covers every cell once`);
  }
});

await check("degenerate lengths and sizes do not hang or explode", () => {
  assertTrue(planSlabs(0).length === 0, "zero length -> no slabs");
  assertTrue(planSlabs(-5).length === 0, "negative length -> no slabs");
  assertTrue(planSlabs(Number.NaN).length === 0, "NaN length -> no slabs");
  // A zero or nonsense slab size must not produce an infinite loop.
  assertTrue(planSlabs(10, 0).length > 0, "zero slab size falls back to the default");
  assertTrue(planSlabs(10, -1).length > 0, "negative slab size falls back to the default");
  assertTrue(planSlabs(10, Number.NaN).length > 0, "NaN slab size falls back to the default");
});

await check("the default slab keeps a three-input request far under a whole-field clone", () => {
  const inputs = makeInputs(64);
  const bytes = slabRequestBytes(
    { theta: inputs.theta, p: inputs.p, qv: inputs.qv },
    { start: 0, end: DERIVED_SLAB_CELLS },
  );
  assertTrue(bytes === DERIVED_SLAB_CELLS * 4 * 3, "three float arrays per slab");
  assertTrue(bytes < 32 * 1024 * 1024, "a slab request stays modest");
});

// ---------- Slicing ----------

await check("a slice copies rather than views its source", () => {
  /*
   * A subarray shares the source buffer, and structured-cloning one can carry the entire
   * 168 MB array to the worker instead of the slab it was meant to send.
   */
  const inputs = makeInputs(16);
  const sliced = sliceInputs({ theta: inputs.theta }, { start: 4, end: 8 });
  assertTrue(sliced.theta.data.length === 4, "the slice has the slab's length");
  assertTrue(
    sliced.theta.data.buffer !== inputs.theta.data.buffer,
    "and owns its buffer",
  );
  assertTrue(sliced.theta.shape.length === 1, "a slab is rank 1");
  for (let i = 0; i < 4; i++) {
    assertTrue(sliced.theta.data[i] === inputs.theta.data[4 + i], `value ${i} carried across`);
  }
});

// ---------- The property that makes slabbing legitimate ----------

for (const fieldName of DERIVED) {
  await check(`${fieldName} is unchanged by slabbing`, async () => {
    const cells = 5_000;
    const inputs = makeInputs(cells);
    const whole = evaluateDerivedField(fieldName, inputs);
    // A slab size that does not divide the length, so boundaries land mid-slab.
    const slabbed = await evaluateDerivedFieldInWorker(fieldName, inputs, { slabCells: 777 });

    assertTrue(slabbed.data.length === whole.data.length, "same length");
    assertTrue(slabbed.shape.join() === whole.shape.join(), "same shape");
    let mismatches = 0;
    for (let i = 0; i < whole.data.length; i++) {
      const a = whole.data[i];
      const b = slabbed.data[i];
      // NaN is a legitimate result (below-ground, unphysical input); it must match as NaN.
      const same = Number.isNaN(a) ? Number.isNaN(b) : a === b;
      if (!same) mismatches++;
    }
    assertTrue(mismatches === 0, `every one of ${cells} values matches exactly`);
  });
}

await check("the reassembled field keeps the source's 3-D shape", () => {
  // The slabs are rank 1; the caller must get the volume shape back, not a flat array.
  const cells = 4 * 5 * 6;
  const flat = makeInputs(cells);
  const volume: Record<string, NpyN> = {};
  for (const [name, input] of Object.entries(flat)) volume[name] = { shape: [4, 5, 6], data: input.data };
  return evaluateDerivedFieldInWorker("temperature", volume, { slabCells: 7 }).then((result) => {
    assertTrue(result.shape.join() === "4,5,6", "the volume shape is restored");
    assertTrue(result.data.length === cells, "with every cell present");
  });
});

// ---------- Windowed reads ----------

await check("window geometry survives derived evaluation", async () => {
  /*
   * A windowed read reports where its aggregated cells physically sit. Losing that on the way
   * through a derived field would draw it on a different geometry than its own dependencies.
   */
  const inputs = makeInputs(64);
  const geometry = {
    weighting: "physical_cell_volume" as const,
    sampleLocation: "physical_volume_centroid" as const,
    centroidsM: null,
  };
  const windowed: Record<string, NpyN> = {};
  for (const [name, input] of Object.entries(inputs)) {
    windowed[name] = { ...input, windowGeometry: geometry };
  }
  const result = await evaluateDerivedFieldInWorker("theta_e", windowed, { slabCells: 16 });
  assertTrue(result.windowGeometry?.weighting === "physical_cell_volume", "weighting carried");
  assertTrue(
    result.windowGeometry?.sampleLocation === "physical_volume_centroid",
    "sample location carried",
  );
});

await check("a full-resolution read carries no window geometry", async () => {
  // Absence is meaningful: it marks the field as being on the native lattice.
  const result = await evaluateDerivedFieldInWorker("temperature", makeInputs(32), { slabCells: 8 });
  assertTrue(result.windowGeometry === undefined, "no geometry is invented");
});

// ---------- Failure handling ----------

await check("a missing dependency is reported, not silently zero-filled", async () => {
  try {
    await evaluateDerivedFieldInWorker("theta_e", { theta: makeInputs(8).theta }, { slabCells: 4 });
    assertTrue(false, "missing dependency should reject");
  } catch (error) {
    assertTrue(error instanceof Error, "rejects with an Error");
  }
});

await check("an unknown field is rejected", async () => {
  try {
    await evaluateDerivedFieldInWorker("not_a_field", makeInputs(8), { slabCells: 4 });
    assertTrue(false, "unknown field should reject");
  } catch (error) {
    assertTrue((error as Error).message.includes("not_a_field"), "and names it");
  }
});

await check("an already-aborted signal stops before any work", async () => {
  const controller = new AbortController();
  controller.abort();
  try {
    await evaluateDerivedFieldInWorker("theta_e", makeInputs(64), {
      slabCells: 8,
      signal: controller.signal,
    });
    assertTrue(false, "an aborted signal should reject");
  } catch (error) {
    assertTrue((error as DOMException).name === "AbortError", "with an AbortError");
  }
});

// ---------- The worker path itself ----------

/*
 * Node has no global Worker, so everything above ran the synchronous fallback. These drive the
 * asynchronous path with a fake worker: one worker must serve every slab and be terminated once,
 * because spawning one per slab would cost more than a slab takes to evaluate.
 */
class FakeDerivedWorker {
  onmessage: ((event: { data: DerivedFieldWorkerReply }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  posts: DerivedFieldWorkerRequest[] = [];
  terminated = 0;
  failNext: string | null = null;

  postMessage(message: DerivedFieldWorkerRequest): void {
    this.posts.push(message);
    // Reply on a later turn, as a real worker does.
    queueMicrotask(() => {
      if (this.failNext) {
        const error = this.failNext;
        this.failNext = null;
        this.onmessage?.({ data: { ok: false, error } });
        return;
      }
      this.onmessage?.({ data: { ok: true, result: evaluateDerivedWorkerRequest(message) } });
    });
  }

  terminate(): void { this.terminated += 1; }
}

await check("one worker serves every slab and is terminated once", async () => {
  const worker = new FakeDerivedWorker();
  const inputs = makeInputs(1_000);
  const result = await evaluateDerivedFieldInWorker("theta_e", inputs, {
    slabCells: 250,
    workerFactory: () => worker,
  });
  assertTrue(worker.posts.length === 4, "one message per slab");
  assertTrue(worker.terminated === 1, "the worker is terminated exactly once");
  assertTrue(result.data.length === 1_000, "the field is fully reassembled");

  const whole = evaluateDerivedField("theta_e", inputs);
  let mismatches = 0;
  for (let i = 0; i < whole.data.length; i++) {
    const a = whole.data[i];
    const b = result.data[i];
    if (!(Number.isNaN(a) ? Number.isNaN(b) : a === b)) mismatches++;
  }
  assertTrue(mismatches === 0, "and matches whole-field evaluation exactly");
});

await check("each slab message carries only its own cells", async () => {
  // The point of slabbing: a message must never carry the whole field.
  const worker = new FakeDerivedWorker();
  await evaluateDerivedFieldInWorker("temperature", makeInputs(900), {
    slabCells: 300,
    workerFactory: () => worker,
  });
  for (const post of worker.posts) {
    assertTrue(post.inputs.theta.data.length === 300, "a slab message holds one slab");
    assertTrue(post.fieldName === "temperature", "and names its field");
  }
});

await check("a worker failure rejects and still terminates the worker", async () => {
  const worker = new FakeDerivedWorker();
  worker.failNext = "worker exploded";
  let error: unknown;
  try {
    await evaluateDerivedFieldInWorker("temperature", makeInputs(100), {
      slabCells: 25,
      workerFactory: () => worker,
    });
  } catch (caught) { error = caught; }
  assertTrue((error as Error)?.message === "worker exploded", "the worker's error surfaces");
  assertTrue(worker.terminated === 1, "and the worker is not leaked");
});

await check("completed workers can close after their final transferable reply", async () => {
  let closed = 0;
  const worker = Object.assign(new FakeDerivedWorker(), { close: () => { closed += 1; } });
  const result = await evaluateDerivedFieldInWorker("temperature", makeInputs(32), {
    slabCells: 8, workerFactory: () => worker,
  });
  assertTrue(result.data.length === 32 && worker.posts.length === 4, "all slabs are received before closing");
  assertTrue(closed === 1 && worker.terminated === 0, "idle completion uses graceful close once");
});

await check("a nonresponsive worker times out and is terminated", async () => {
  const worker = new FakeDerivedWorker();
  worker.postMessage = () => {};
  let error: unknown;
  try {
    await evaluateDerivedFieldInWorker("temperature", makeInputs(32), {
      workerFactory: () => worker, timeoutMs: 5,
    });
  } catch (caught) { error = caught; }
  assertTrue((error as Error)?.message.includes("timeout"), "the caller receives a bounded worker failure");
  assertTrue(worker.terminated === 1, "timed-out workers release their resources");
});

await check("cancellation during a slab rejects without publishing a partial field", async () => {
  const worker = new FakeDerivedWorker();
  const controller = new AbortController();
  worker.postMessage = () => controller.abort();
  let error: unknown;
  try {
    await evaluateDerivedFieldInWorker("temperature", makeInputs(32), {
      workerFactory: () => worker, signal: controller.signal,
    });
  } catch (caught) { error = caught; }
  assertTrue(error instanceof DOMException && error.name === "AbortError", "the active slab observes cancellation");
  assertTrue(worker.terminated === 1 && worker.onmessage === null, "cancelled workers cannot publish late slabs");
});

await check("short worker output cannot silently leave zero-filled cells", async () => {
  const worker = new FakeDerivedWorker();
  worker.postMessage = () => queueMicrotask(() => worker.onmessage?.({
    data: { ok: true, result: { shape: [1], data: Float32Array.of(123) } },
  }));
  let error: unknown;
  try {
    await evaluateDerivedFieldInWorker("temperature", makeInputs(32), { workerFactory: () => worker });
  } catch (caught) { error = caught; }
  assertTrue((error as Error)?.message.includes("incomplete slab"), "incomplete results are rejected");
  assertTrue(worker.terminated === 1, "invalid-result workers are terminated");
});

// ---------- Summary ----------

for (const line of failures) console.error(line);
console.log(`\nderived-field-slab tests: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
