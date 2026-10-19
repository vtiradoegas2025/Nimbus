/*
 * Tests for src/lib/fieldCache.ts.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_field_cache.ts
 *
 * Guards the bounded LRU + in-flight dedup that keeps scrubbing fast and memory
 * capped: a cache hit must not re-load, concurrent identical loads must fetch
 * once, the byte cap must evict the least-recently-used, and an oversized field
 * must be served but not retained.
 */

import {
  getOrLoadField,
  getOrLoadSharedAbortableField,
  makeFieldKey,
  clearFieldCache,
  setFieldCacheLimit,
  fieldCacheStats,
  fieldResidentBytes,
  MAX_FIELD_CACHE_ENTRIES,
} from "../../../../ui/src/lib/core/fieldCache.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

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

// A field of `bytes` bytes (bytes/4 float32 elements). The cache only reads
// data.byteLength, so this stands in for a parsed NpyN.
function field(bytes: number): { shape: number[]; data: Float32Array } {
  return { shape: [bytes / 4], data: new Float32Array(bytes / 4) };
}

const mk = (step: number, f: string, generation = "generation-a") =>
  makeFieldKey("/run dir", step, f, generation);

async function main(): Promise<void> {
  // ---------- publication generation is part of immutable identity ----------
  assertTrue(
    mk(0, "u", "generation-a") !== mk(0, "u", "generation-b"),
    "same directory/step/field in a new generation has a distinct key",
  );

  // ---------- cache hit: load once, reuse ----------
  clearFieldCache();
  setFieldCacheLimit(1000);
  {
    const decodedBaseline = getVisualizationResourceSnapshot()
      .byStage["decoded-field"].residentBytes;
    let calls = 0;
    let cacheHits = 0;
    const loader = () => {
      calls++;
      return Promise.resolve(field(100));
    };
    const a = await getOrLoadField(mk(0, "u"), loader);
    const b = await getOrLoadField(mk(0, "u"), loader, () => cacheHits++);
    assertTrue(calls === 1, "second get is a cache hit (loader called once)");
    assertTrue(a === b, "cache returns the same instance");
    assertTrue(cacheHits === 1, "cache-hit callback advances latest-wins identity");
    assertTrue(
      getVisualizationResourceSnapshot().byStage["decoded-field"].residentBytes
        === decodedBaseline + 100,
      "parsed cache reports exact decoded-field backing stores",
    );
    clearFieldCache();
    assertTrue(
      getVisualizationResourceSnapshot().byStage["decoded-field"].residentBytes
        === decodedBaseline,
      "parsed cache clear releases decoded-field ownership",
    );
  }

  // ---------- in-flight dedup: concurrent identical loads fetch once ----------
  clearFieldCache();
  {
    let calls = 0;
    let cacheHits = 0;
    const slow = () =>
      new Promise<{ shape: number[]; data: Float32Array }>((resolve) => {
        calls++;
        setTimeout(() => resolve(field(100)), 20);
      });
    const [x, y] = await Promise.all([
      getOrLoadField(mk(1, "v"), slow, () => cacheHits++),
      getOrLoadField(mk(1, "v"), slow),
    ]);
    assertTrue(calls === 1, "concurrent identical loads deduped to one fetch");
    assertTrue(x === y, "deduped callers get the same instance");
    assertTrue(cacheHits === 0, "same-key in-flight dedup does not supersede itself");
  }

  // ---------- byte cap evicts; evicted entry reloads ----------
  clearFieldCache();
  setFieldCacheLimit(1000);
  {
    let firstReject: (error: Error) => void = () => {};
    let secondResolve: (value: { shape: number[]; data: Float32Array }) => void = () => {};
    const first = getOrLoadField(
      mk(4, "owned"),
      () => new Promise((_resolve, reject) => { firstReject = reject; }),
      undefined,
      { shareInFlight: false },
    );
    const second = getOrLoadField(
      mk(4, "owned"),
      () => new Promise((resolve) => { secondResolve = resolve; }),
      undefined,
      { shareInFlight: false },
    );
    firstReject(new DOMException("cancelled", "AbortError"));
    secondResolve(field(100));
    const [cancelled, surviving] = await Promise.allSettled([first, second]);
    assertTrue(cancelled.status === "rejected", "one independently owned read can be cancelled");
    assertTrue(
      surviving.status === "fulfilled" && surviving.value.data.byteLength === 100,
      "cancelling one owned read does not reject an identical consumer",
    );
    assertTrue(fieldCacheStats().bytes === 100, "independent completion is cached exactly once");
  }

  // ---------- cancellable consumer may join a prefetch-owned request ----------
  clearFieldCache();
  {
    let calls = 0;
    let release: (value: { shape: number[]; data: Float32Array }) => void = () => {};
    const prefetch = getOrLoadField(
      mk(5, "prefetched"),
      () => new Promise((resolve) => {
        calls += 1;
        release = resolve;
      }),
    );
    const interactive = getOrLoadField(
      mk(5, "prefetched"),
      async () => {
        calls += 1;
        return field(100);
      },
      undefined,
      { shareInFlight: false, joinInFlight: true },
    );
    release(field(100));
    const [warmed, joined] = await Promise.all([prefetch, interactive]);
    assertTrue(calls === 1, "interactive request joins its in-flight prefetch");
    assertTrue(warmed === joined, "prefetch and interactive consumer share the immutable result");
  }

  // ---------- abortable consumers share one platform transaction ----------
  clearFieldCache();
  {
    let calls = 0;
    let platformAborted = false;
    let release: (value: { shape: number[]; data: Float32Array }) => void = () => {};
    const firstController = new AbortController();
    const secondController = new AbortController();
    const loader = (signal: AbortSignal) => new Promise<{
      shape: number[];
      data: Float32Array;
    }>((resolve, reject) => {
      calls += 1;
      release = resolve;
      signal.addEventListener("abort", () => {
        platformAborted = true;
        reject(new DOMException("native cancelled", "AbortError"));
      }, { once: true });
    });
    const first = getOrLoadSharedAbortableField(
      mk(7, "bounded"), loader, firstController.signal,
    );
    const second = getOrLoadSharedAbortableField(
      mk(7, "bounded"), loader, secondController.signal,
    );
    firstController.abort();
    const firstResult = await Promise.allSettled([first]);
    assertTrue(firstResult[0].status === "rejected", "one bounded-read consumer cancels independently");
    assertTrue(!platformAborted, "one remaining consumer keeps the shared native read alive");
    release(field(100));
    const shared = await second;
    assertTrue(calls === 1 && shared.data.byteLength === 100, "identical bounded consumers perform one platform read");
  }

  clearFieldCache();
  {
    let calls = 0;
    let release: (value: ReturnType<typeof field>) => void = () => {};
    const loader = () => new Promise<ReturnType<typeof field>>((resolve) => {
      calls++;
      release = resolve;
    });
    const transient = getOrLoadSharedAbortableField(mk(21, "mixed"), loader, undefined, undefined, { retain: false });
    const resident = getOrLoadSharedAbortableField(mk(21, "mixed"), loader);
    release(field(100));
    const [temporary, retained] = await Promise.all([transient, resident]);
    const warmed = await getOrLoadField(mk(21, "mixed"), loader);
    assertTrue(calls === 1, "a resident consumer joining a transient read prevents a later redundant fetch");
    assertTrue(temporary === retained && warmed === retained, "mixed retention shares the exact immutable values");
    assertTrue(fieldCacheStats().entries === 1, "mixed retention creates one resident entry");
  }

  clearFieldCache();
  {
    let platformAborted = false;
    let release: (value: ReturnType<typeof field>) => void = () => {};
    const controller = new AbortController();
    const loader = (signal: AbortSignal) => new Promise<ReturnType<typeof field>>((resolve) => {
      release = resolve;
      signal.addEventListener("abort", () => { platformAborted = true; });
    });
    const resident = getOrLoadSharedAbortableField(mk(22, "mixed"), loader, controller.signal);
    const transient = getOrLoadSharedAbortableField(mk(22, "mixed"), loader, undefined, undefined, { retain: false });
    controller.abort();
    await Promise.allSettled([resident]);
    release(field(100));
    const surviving = await transient;
    assertTrue(!platformAborted && surviving.data.byteLength === 100, "transient interest survives cancellation of the resident view");
    assertTrue(fieldCacheStats().entries === 0, "an abandoned resident view does not force transient data into the LRU");
  }

  clearFieldCache();
  {
    let calls = 0;
    let release: (value: ReturnType<typeof field>) => void = () => {};
    const loader = () => new Promise<ReturnType<typeof field>>((resolve) => { calls++; release = resolve; });
    const transient = getOrLoadField(mk(23, "mixed"), loader, undefined, { retain: false });
    const resident = getOrLoadField(mk(23, "mixed"), loader);
    release(field(100));
    await Promise.all([transient, resident]);
    await getOrLoadField(mk(23, "mixed"), loader);
    assertTrue(calls === 1 && fieldCacheStats().entries === 1, "unscoped resident interest also promotes a shared transient read");
  }

  clearFieldCache();
  {
    let calls = 0;
    let platformAborted = false;
    let release: (value: ReturnType<typeof field>) => void = () => {};
    const controller = new AbortController();
    const loader = (signal?: AbortSignal) => new Promise<ReturnType<typeof field>>((resolve) => {
      calls++; release = resolve;
      signal?.addEventListener("abort", () => { platformAborted = true; });
    });
    const cancellable = getOrLoadSharedAbortableField(mk(24, "mixed"), loader, controller.signal);
    const unscoped = getOrLoadField(mk(24, "mixed"), loader);
    controller.abort();
    await Promise.allSettled([cancellable]);
    release(field(100));
    const surviving = await unscoped;
    assertTrue(calls === 1, "cancellable and unscoped consumers share a single platform read in either ownership lane");
    assertTrue(!platformAborted && surviving.data.byteLength === 100, "the unscoped consumer keeps a cancelled view's shared read alive");
  }

  // ---------- final consumer cancellation reaches the platform ----------
  clearFieldCache();
  {
    let platformAborted = false;
    const controller = new AbortController();
    const pending = getOrLoadSharedAbortableField(
      mk(8, "bounded"),
      (signal) => new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => {
          platformAborted = true;
          reject(new DOMException("native cancelled", "AbortError"));
        }, { once: true });
      }),
      controller.signal,
    );
    controller.abort();
    await Promise.allSettled([pending]);
    assertTrue(platformAborted, "the final bounded-read consumer cancels the platform transaction");
  }

  // ---------- replacement cannot join an abandoned, aborting transaction ----------
  clearFieldCache();
  {
    let calls = 0;
    let abandonedPlatformAborted = false;
    let rejectAbandoned: (error: Error) => void = () => {};
    const loader = (signal: AbortSignal) => {
      calls += 1;
      if (calls > 1) return Promise.resolve(field(100));
      return new Promise<{ shape: number[]; data: Float32Array }>((_resolve, reject) => {
        rejectAbandoned = reject;
        signal.addEventListener("abort", () => {
          // Model native cancellation acknowledgement arriving later than the
          // consumer's synchronous AbortSignal notification.
          abandonedPlatformAborted = true;
        }, { once: true });
      });
    };
    const abandonedController = new AbortController();
    const abandoned = getOrLoadSharedAbortableField(
      mk(11, "replacement"), loader, abandonedController.signal,
    );
    abandonedController.abort();
    const replacement = getOrLoadSharedAbortableField(
      mk(11, "replacement"), loader, new AbortController().signal,
    );
    const [abandonedResult, replacementResult] = await Promise.all([
      Promise.allSettled([abandoned]).then(([result]) => result),
      replacement,
    ]);
    assertTrue(abandonedResult.status === "rejected", "abandoned consumer rejects immediately");
    assertTrue(abandonedPlatformAborted, "abandoned backing transaction is cancelled");
    assertTrue(calls === 2, "immediate replacement starts a fresh platform transaction");
    assertTrue(
      replacementResult.data.byteLength === 100,
      "replacement does not inherit the abandoned transaction's AbortError",
    );
    rejectAbandoned(new DOMException("native cancellation acknowledged", "AbortError"));
    await Promise.resolve();
  }

  // ---------- byte cap evicts; evicted entry reloads ----------
  clearFieldCache();
  setFieldCacheLimit(250); // holds two 100-byte entries; a third evicts one
  {
    await getOrLoadField(mk(0, "a"), () => Promise.resolve(field(100)));
    await getOrLoadField(mk(1, "b"), () => Promise.resolve(field(100)));
    await getOrLoadField(mk(2, "c"), () => Promise.resolve(field(100)));
    assertTrue(fieldCacheStats().entries === 2, "byte cap holds two entries");
    assertTrue(fieldCacheStats().bytes <= 250, "total bytes within cap");
  }

  // ---------- LRU recency: a hit promotes; the untouched entry is evicted ----------
  clearFieldCache();
  setFieldCacheLimit(250);
  {
    await getOrLoadField(mk(0, "a"), () => Promise.resolve(field(100)));
    await getOrLoadField(mk(1, "b"), () => Promise.resolve(field(100)));
    await getOrLoadField(mk(0, "a"), () => Promise.resolve(field(100))); // promote a
    await getOrLoadField(mk(2, "c"), () => Promise.resolve(field(100))); // evicts b
    let aCalls = 0;
    await getOrLoadField(mk(0, "a"), () => {
      aCalls++;
      return Promise.resolve(field(100));
    });
    assertTrue(aCalls === 0, "promoted entry 'a' survived (cache hit)");
    let bCalls = 0;
    await getOrLoadField(mk(1, "b"), () => {
      bCalls++;
      return Promise.resolve(field(100));
    });
    assertTrue(bCalls === 1, "least-recently-used 'b' was evicted (miss)");
  }

  // ---------- oversized field served but not retained ----------
  clearFieldCache();
  setFieldCacheLimit(250);
  {
    let calls = 0;
    const big = () => {
      calls++;
      return Promise.resolve(field(1000)); // larger than the whole cap
    };
    await getOrLoadField(mk(9, "big"), big);
    assertTrue(fieldCacheStats().entries === 0, "oversized field not retained");
    await getOrLoadField(mk(9, "big"), big);
    assertTrue(calls === 2, "oversized field reloads every time (not cached)");
  }

  // ---------- assembly scratch is served without polluting resident cache ----------
  clearFieldCache();
  setFieldCacheLimit(1000);
  {
    await getOrLoadField(
      mk(10, "tile"),
      () => Promise.resolve(field(100)),
      undefined,
      { retain: false },
    );
    assertTrue(fieldCacheStats().entries === 0, "transient window tile is not retained");
  }

  // ---------- terrain geometry is included exactly once per backing store ----------
  {
    const shared = new Float64Array(12);
    const terrain = {
      ...field(16),
      windowGeometry: {
        weighting: "physical_cell_volume" as const,
        sampleLocation: "physical_volume_centroid" as const,
        centroidsM: {
          x: new Float64Array(shared.buffer, 0, 4),
          y: new Float64Array(shared.buffer, 32, 4),
          z: new Float64Array(shared.buffer, 64, 4),
        },
        volumesM3: new Float64Array(4),
      },
    };
    assertTrue(
      fieldResidentBytes(terrain) === 16 + shared.byteLength + 32,
      "terrain cache accounting counts unique value, centroid, and volume buffers",
    );
  }

  clearFieldCache();
  setFieldCacheLimit(120);
  {
    const shared = field(100);
    await getOrLoadField(mk(25, "direct"), async () => shared);
    await getOrLoadField(mk(25, "assembled"), async () => shared);
    assertTrue(fieldCacheStats().entries === 2 && fieldCacheStats().bytes === 100, "single-tile and direct aliases count their shared physical backing store once");
    let repeated = 0;
    await getOrLoadField(mk(25, "direct"), async () => { repeated++; return shared; });
    await getOrLoadField(mk(25, "assembled"), async () => { repeated++; return shared; });
    assertTrue(repeated === 0, "false double accounting does not evict an alias and force another read");
    await getOrLoadField(mk(26, "fresh"), async () => field(100));
    assertTrue(fieldCacheStats().entries === 1 && fieldCacheStats().bytes === 100, "an alias eviction releases bytes only after the final reference retires");
  }

  clearFieldCache();
  {
    const shared = field(4);
    for (let index = 0; index <= MAX_FIELD_CACHE_ENTRIES; index++) {
      await getOrLoadField(mk(100 + index, "alias"), async () => shared);
    }
    assertTrue(fieldCacheStats().entries === MAX_FIELD_CACHE_ENTRIES && fieldCacheStats().bytes === 4, "shared payload aliases retain bounded cache metadata independently of the byte ceiling");
  }

  // ---------- clearFieldCache during an in-flight load must not repopulate ----------
  clearFieldCache();
  setFieldCacheLimit(1000);
  {
    let release: (f: { shape: number[]; data: Float32Array }) => void = () => {};
    const slow = () =>
      new Promise<{ shape: number[]; data: Float32Array }>((res) => {
        release = res;
      });
    const inflight = getOrLoadField(mk(3, "stale"), slow);
    clearFieldCache(); // epoch++ while the load is still in flight
    release(field(100)); // now let the pre-clear load resolve
    const got = await inflight;
    assertTrue(got.data.byteLength === 100, "caller still receives the loaded field after a mid-load clear");
    assertTrue(
      fieldCacheStats().entries === 0,
      "in-flight load started before clear does not repopulate the cache",
    );
  }

  console.log(`\nfieldCache tests: ${passed} passed, ${failed} failed.`);
  if (failures.length > 0) {
    console.log("\nFailures:");
    for (const f of failures) console.log("  " + f);
    process.exit(1);
  }
  process.exit(0);
}

main();
