import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { configurePlaybackResources, resolvePlaybackResourceBudget } from "../../../../ui/src/lib/core/playbackResourcePolicy.ts";
import { fieldCacheStats } from "../../../../ui/src/lib/core/fieldCache.ts";
import { fieldLoadScheduler } from "../../../../ui/src/lib/core/loadScheduler.ts";
import { visualizationResourceGovernor } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import { NATIVE_WINDOW_IO_BYTES, DEFAULT_NATIVE_WINDOW_OUTPUT_BYTES, DEFAULT_NATIVE_WINDOW_SOURCE_BYTES } from "../../../../ui/src/lib/core/nativeWindowBudget.ts";

function nativeConstant(file: string, name: string): number {
  const source = readFileSync(new URL(`../../../../ui/src-tauri/src/commands/${file}.rs`, import.meta.url), "utf8");
  const expression = source.match(new RegExp(`const ${name}: (?:usize|u64) = ([0-9_ *]+);`))?.[1];
  assert(expression, `${name} must remain an explicit native limit`);
  return expression.split("*").reduce((value, factor) => value * Number(factor.trim().replaceAll("_", "")), 1);
}
assert.equal(NATIVE_WINDOW_IO_BYTES, nativeConstant("io_budget", "DEFAULT_IO_BYTES"));
assert.equal(DEFAULT_NATIVE_WINDOW_SOURCE_BYTES, nativeConstant("outputs", "DEFAULT_FIELD_WINDOW_SOURCE_BYTES"));
assert.equal(DEFAULT_NATIVE_WINDOW_OUTPUT_BYTES, nativeConstant("outputs", "DEFAULT_FIELD_WINDOW_BYTES"));
assert.equal(DEFAULT_NATIVE_WINDOW_SOURCE_BYTES, nativeConstant("datasets", "DEFAULT_NETCDF_SOURCE_BYTES"));
assert.equal(DEFAULT_NATIVE_WINDOW_OUTPUT_BYTES, nativeConstant("datasets", "DEFAULT_NETCDF_OUTPUT_BYTES"));

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const unknown = { source: "unknown", systemRamBytes: null, logicalCores: null } as const;
const fallback = resolvePlaybackResourceBudget(unknown);
assert.equal(fallback.fieldCacheBytes, 256 * MIB);
assert.equal(fallback.fieldLoadBytes, 384 * MIB);
assert.equal(fallback.maxConcurrentLoads, 3);
assert.deepEqual(resolvePlaybackResourceBudget({ ...unknown, source: "browser", systemRamBytes: 128 * GIB, logicalCores: 48 }), fallback,
  "browser hints cannot claim native RAM or unbounded read concurrency");

const small = resolvePlaybackResourceBudget({ source: "tauri", systemRamBytes: 2 * GIB, logicalCores: 2 });
assert.equal(small.maxConcurrentLoads, 1);
assert(small.fieldCacheBytes + small.fieldLoadBytes <= GIB / 8);
assert.equal(small.stagingBytes, small.fieldLoadBytes);

const capacityBefore = visualizationResourceGovernor.stats().capacityBytes;
const workstation = configurePlaybackResources({ source: "tauri", systemRamBytes: 128 * GIB, logicalCores: 24 });
assert.equal(workstation.maxConcurrentLoads, 6);
assert.equal(workstation.fieldLoadBytes, 512 * MIB, "frontend concurrency respects the native shared I/O ceiling");
assert(workstation.fieldCacheBytes + workstation.fieldLoadBytes <= 3 * GIB);
assert.equal(fieldCacheStats().limit, workstation.fieldCacheBytes);
assert.equal(fieldLoadScheduler.stats().maxBytes, workstation.fieldLoadBytes);
assert.equal(visualizationResourceGovernor.stats().capacityBytes, capacityBefore,
  "pipeline tuning cannot assert an aggregate visualization qualification");

let settle!: () => void;
const blocked = new Promise<number>(resolve => { settle = () => resolve(1); });
const reads = Array.from({ length: 6 }, () => fieldLoadScheduler.schedule({ estimatedBytes: 128 * MIB }, () => blocked));
await new Promise<void>(resolve => setTimeout(resolve, 0));
assert.equal(fieldLoadScheduler.stats().active, 4);
assert.equal(fieldLoadScheduler.stats().queued, 2);
assert.equal(visualizationResourceGovernor.stats().reservedBytes, 512 * MIB,
  "excess panes wait without exceeding native read admission");
settle();
await Promise.all(reads);
assert.equal(visualizationResourceGovernor.stats().committedBytes, 0);
configurePlaybackResources(unknown);
const manual = configurePlaybackResources({ source: "tauri", systemRamBytes: 16 * GIB, logicalCores: 8 },
  { loadingWorkers: 4, ioBytes: 1024 * MIB });
assert.equal(manual.maxConcurrentLoads, 4);
assert.equal(manual.fieldLoadBytes, 1024 * MIB);
assert.equal(fieldLoadScheduler.stats().maxBytes, 1024 * MIB);
assert.equal(resolvePlaybackResourceBudget({ source: "tauri", systemRamBytes: 8 * GIB, logicalCores: 2 },
  { loadingWorkers: 64, ioBytes: 64 * MIB }).maxConcurrentLoads, 2, "manual counts respect known hardware cores");
configurePlaybackResources(unknown);
console.log("Playback resource policy: passed");
