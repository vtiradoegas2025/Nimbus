import type { NpyN } from "@/lib/platform/npy";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "./visualizationResourceGovernor";

/* Bounded LRU field cache (256 MB default). Deduplicates concurrent loads for
 * the same key. Callers MUST NOT mutate field.data — the same instance is shared. */

// Two speculative 40 MiB fields plus a four-field 3D scene fit, while old
// frames are evicted before WKWebView enters its multi-gigabyte pressure mode.
const DEFAULT_MAX_BYTES = 256 * 1024 * 1024;
export const MAX_FIELD_CACHE_ENTRIES = 4096;

interface CacheEntry {
  field: NpyN;
  resourceReservationId: number;
}

let maxBytes = DEFAULT_MAX_BYTES;
let totalBytes = 0;
let hits = 0;
let loadsStarted = 0;
let deduplicatedRequests = 0;
let insertions = 0;
let evictions = 0;
let oversizedLoads = 0;
let loadFailures = 0;
let clears = 0;
// Bumped on clear; in-flight loads from before a clear must not repopulate the cache.
let epoch = 0;
// Map insertion order = LRU queue; hit deletes+reinserts to move to most-recent end.
const cache = new Map<string, CacheEntry>();
const retainedBuffers = new Map<ArrayBufferLike, number>();
// Dedup: two concurrent callers for the same key share one in-flight load.
interface PendingFieldLoad {
  promise: Promise<NpyN>;
  retainers: Set<() => boolean>;
}
const inFlight = new Map<string, PendingFieldLoad>();

interface SharedAbortableLoad {
  controller: AbortController;
  promise: Promise<NpyN>;
  consumers: Set<symbol>;
  residentConsumers: number;
  settled: boolean;
}

// Bounded plot reads are cancellable, but cancellation belongs to one view,
// not to the underlying immutable field identity. Keep one native transaction
// alive while any view still needs it and abort only after the final consumer
// releases its interest.
const sharedAbortableLoads = new Map<string, SharedAbortableLoad>();

export interface FieldCacheStats {
  readonly entries: number;
  readonly bytes: number;
  readonly limit: number;
  readonly pendingLoads: number;
  readonly hits: number;
  readonly loadsStarted: number;
  readonly deduplicatedRequests: number;
  readonly insertions: number;
  readonly evictions: number;
  readonly oversizedLoads: number;
  readonly loadFailures: number;
  readonly clears: number;
}

type FieldCacheListener = () => void;
const listeners = new Set<FieldCacheListener>();
let snapshot: FieldCacheStats = createSnapshot();

function createSnapshot(): FieldCacheStats {
  return Object.freeze({
    entries: cache.size,
    bytes: totalBytes,
    limit: maxBytes,
    pendingLoads: inFlight.size + sharedAbortableLoads.size,
    hits,
    loadsStarted,
    deduplicatedRequests,
    insertions,
    evictions,
    oversizedLoads,
    loadFailures,
    clears,
  });
}

function publish(): void {
  snapshot = createSnapshot();
  for (const listener of listeners) listener();
}

/** Build a collision-proof cache key; JSON-encoded tuple avoids bare-concat collisions. */
export function makeFieldKey(
  runDir: string,
  stepIndex: number,
  fieldName: string,
  publicationGeneration: string | null,
): string {
  return JSON.stringify([runDir, publicationGeneration, stepIndex, fieldName]);
}

function retainBuffers(field: NpyN): void {
  for (const buffer of fieldBackingStores(field)) {
    const references = retainedBuffers.get(buffer) ?? 0;
    if (references === 0) totalBytes += buffer.byteLength;
    retainedBuffers.set(buffer, references + 1);
  }
}

function releaseBuffers(field: NpyN): void {
  for (const buffer of fieldBackingStores(field)) {
    const references = retainedBuffers.get(buffer)!;
    if (references === 1) {
      totalBytes -= buffer.byteLength;
      retainedBuffers.delete(buffer);
    } else retainedBuffers.set(buffer, references - 1);
  }
}

function evictToFit(): void {
  while ((totalBytes > maxBytes || cache.size > MAX_FIELD_CACHE_ENTRIES) && cache.size > 0) {
    const oldest = cache.keys().next().value as string;
    const entry = cache.get(oldest);
    cache.delete(oldest);
    if (entry) {
      visualizationResourceGovernor.release(entry.resourceReservationId);
      releaseBuffers(entry.field);
      evictions += 1;
    }
  }
}

/**
 * Return the cached field for `key`, or run `load()` once and cache the result.
 * Concurrent callers for the same key share a single in-flight load. A field
 * larger than the whole cap is returned to the caller but not retained (caching
 * it would immediately evict everything else for no reuse benefit).
 */
export async function getOrLoadField(
  key: string,
  load: () => Promise<NpyN>,
  onCacheHit?: () => void,
  options: {
    shareInFlight?: boolean;
    joinInFlight?: boolean;
    /** Return/deduplicate the value without retaining it in the resident LRU. */
    retain?: boolean;
  } = {},
): Promise<NpyN> {
  const shareInFlight = options.shareInFlight !== false;
  const joinInFlight = options.joinInFlight ?? shareInFlight;
  const hit = cache.get(key);
  if (hit) {
    hits += 1;
    onCacheHit?.();
    // Mark most-recently-used.
    cache.delete(key);
    cache.set(key, hit);
    publish();
    return hit.field;
  }

  // A cancellable interactive read must not publish a promise that another
  // consumer could inherit, but it may safely join a prefetch-owned read. Its
  // AbortSignal then cancels only its interest; the shared prefetch continues.
  const pending = joinInFlight ? inFlight.get(key) : undefined;
  if (pending) {
    pending.retainers.add(() => options.retain !== false);
    deduplicatedRequests += 1;
    publish();
    return pending.promise;
  }
  if (joinInFlight && sharedAbortableLoads.has(key)) {
    return getOrLoadSharedAbortableField(key, () => load(), undefined, onCacheHit, options);
  }

  const startEpoch = epoch;
  const retainers = new Set([() => options.retain !== false]);
  loadsStarted += 1;
  const loadSequence = loadsStarted;
  const p = (async () => {
    try {
      const field = await load();
      const bytes = fieldResidentBytes(field);
      // If the cache was cleared while this load was in flight (epoch bumped),
      // hand the field back to the caller but do NOT repopulate the cache — the
      // clear was meant to drop exactly this kind of now-stale data.
      const retain = [...retainers].some((requested) => requested());
      if (startEpoch === epoch && retain && bytes <= maxBytes) {
        const resources = visualizationResourceGovernor.reserve({
          owner: "parsed-field-cache",
          generation: `${startEpoch}:${loadSequence}`,
          priority: "visible",
          claims: fieldBackingStores(field).map((buffer) =>
            visualizationBufferClaim(buffer, "decoded-field")
          ),
        });
        if (!resources.ok) return field;
        const previous = cache.get(key);
        if (previous) {
          visualizationResourceGovernor.release(previous.resourceReservationId);
          releaseBuffers(previous.field);
        }
        cache.set(key, {
          field,
          resourceReservationId: resources.reservation.id,
        });
        retainBuffers(field);
        insertions += 1;
        evictToFit();
      } else if (startEpoch === epoch && retain) {
        oversizedLoads += 1;
      }
      return field;
    } catch (error) {
      loadFailures += 1;
      // A pre-clear platform read can still fail after its cache identity was
      // invalidated. Record that historical failure without reviving its key.
      if (startEpoch !== epoch) publish();
      throw error;
    } finally {
      // clear() wipes inFlight and bumps the epoch; only delete our own marker
      // if no clear happened since — otherwise a newer load for this key (which
      // re-registered after the wipe) owns the entry and must be left alone.
      if (startEpoch === epoch && shareInFlight) {
        inFlight.delete(key);
        publish();
      } else if (startEpoch === epoch) {
        publish();
      }
    }
  })();
  if (shareInFlight) inFlight.set(key, { promise: p, retainers });
  publish();
  return p;
}

function fieldAbortError(): DOMException {
  return new DOMException("Field read cancelled", "AbortError");
}

/**
 * Reference-counted cancellable variant used by bounded native/netCDF reads.
 * Exact-key consumers share one load even though each owns an AbortSignal.
 * Aborting one consumer rejects only that consumer; the platform operation is
 * cancelled after the last consumer leaves. Completed values still enter the
 * ordinary resident LRU exactly once.
 */
export function getOrLoadSharedAbortableField(
  key: string,
  load: (signal: AbortSignal) => Promise<NpyN>,
  signal?: AbortSignal,
  onCacheHit?: () => void,
  options: { retain?: boolean } = {},
): Promise<NpyN> {
  if (signal?.aborted) return Promise.reject(fieldAbortError());

  let entry = sharedAbortableLoads.get(key);
  if (!entry) {
    const controller = new AbortController();
    entry = {
      controller,
      consumers: new Set(),
      residentConsumers: 0,
      settled: false,
      promise: Promise.resolve(null as unknown as NpyN),
    };
    const owned = entry;
    owned.promise = getOrLoadField(
      key,
      () => load(controller.signal),
      onCacheHit,
      { shareInFlight: false, joinInFlight: true, get retain() { return owned.residentConsumers > 0; } },
    ).finally(() => {
      owned.settled = true;
      if (sharedAbortableLoads.get(key) === owned) sharedAbortableLoads.delete(key);
      publish();
    });
    sharedAbortableLoads.set(key, owned);
    publish();
  } else {
    deduplicatedRequests += 1;
    publish();
  }

  const consumer = Symbol(key);
  const resident = options.retain !== false;
  entry.consumers.add(consumer);
  if (resident) entry.residentConsumers++;
  let finished = false;
  return new Promise<NpyN>((resolve, reject) => {
    const release = () => {
      if (finished) return;
      finished = true;
      signal?.removeEventListener("abort", onAbort);
      entry!.consumers.delete(consumer);
      if (resident) entry!.residentConsumers--;
      if (!entry!.settled && entry!.consumers.size === 0) {
        // Retire the abandoned entry before aborting its backing transaction.
        // Abort completion is asynchronous on the native/netCDF paths; keeping
        // the doomed entry joinable during that window lets an immediate
        // interactive replacement inherit the cancelled request.
        if (sharedAbortableLoads.get(key) === entry) {
          sharedAbortableLoads.delete(key);
        }
        entry!.controller.abort();
        publish();
      }
    };
    const onAbort = () => {
      release();
      reject(fieldAbortError());
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    entry!.promise.then(
      (field) => {
        if (finished) return;
        release();
        resolve(field);
      },
      (error) => {
        if (finished) return;
        release();
        reject(error);
      },
    );
  });
}

/** Count unique backing stores retained by a cached field, including terrain geometry. */
export function fieldResidentBytes(field: NpyN): number {
  return fieldBackingStores(field).reduce((bytes, buffer) => bytes + buffer.byteLength, 0);
}

export function fieldBackingStores(field: NpyN): ArrayBufferLike[] {
  const buffers = new Set<ArrayBufferLike>();
  const add = (view: ArrayBufferView | null | undefined) => {
    if (view) buffers.add(view.buffer);
  };
  add(field.data);
  const geometry = field.windowGeometry;
  add(geometry?.centroidsM?.x);
  add(geometry?.centroidsM?.y);
  add(geometry?.centroidsM?.z);
  add(geometry?.volumesM3);
  return [...buffers];
}

/** Drop retained parsed fields. Shared bounded reads are cancelled; platform
 * operations without cancellation finish behind the bumped epoch and cannot
 * repopulate the cache. */
export function clearFieldCache(): void {
  for (const entry of cache.values()) {
    visualizationResourceGovernor.release(entry.resourceReservationId);
  }
  cache.clear();
  retainedBuffers.clear();
  inFlight.clear();
  for (const entry of sharedAbortableLoads.values()) entry.controller.abort();
  sharedAbortableLoads.clear();
  totalBytes = 0;
  // Invalidate in-flight loads started before this clear so they can't
  // repopulate the cache with stale data after they resolve.
  epoch++;
  clears += 1;
  publish();
}

/** Override the resident byte cap (and evict down to it). For tests/tuning. */
export function setFieldCacheLimit(bytes: number): void {
  maxBytes = Number.isFinite(bytes)
    ? Math.max(0, Math.floor(bytes))
    : DEFAULT_MAX_BYTES;
  evictToFit();
  publish();
}

/** Current cache occupancy — for tests and diagnostics. */
export function fieldCacheStats(): FieldCacheStats {
  return snapshot;
}

/**
 * Subscribe to immutable cache snapshots. Notifications are emitted only on
 * cache/request transitions; consumers never need to poll resident memory.
 */
export function subscribeFieldCache(listener: FieldCacheListener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
