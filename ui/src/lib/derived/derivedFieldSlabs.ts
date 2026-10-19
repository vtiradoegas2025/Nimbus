/*
 * Splitting a derived-field evaluation into slabs. Pure — no React, no Worker.
 * Unit-tested in scripts/derived/test_derived_field_slabs.ts.
 */
import type { NpyN } from "@/lib/platform/npy";

/**
 * Cells per slab.
 *
 * Every built-in derived field is elementwise, so a slab is a valid input on its own and the
 * split cannot change a result. 2M cells is ~8 MB per input array, which keeps a three-input
 * field such as theta_e near 24 MB in flight rather than the 504 MB a whole-field message
 * would clone for a 512x512x160 run.
 */
export const DERIVED_SLAB_CELLS = 2_000_000;

export interface Slab {
  start: number;
  end: number;
}

/** Contiguous [start, end) slabs covering `length`. Empty for a zero-length field. */
export function planSlabs(length: number, slabCells: number = DERIVED_SLAB_CELLS): Slab[] {
  if (!Number.isFinite(length) || length <= 0) return [];
  const size = Number.isFinite(slabCells) && slabCells >= 1 ? Math.floor(slabCells) : DERIVED_SLAB_CELLS;
  const slabs: Slab[] = [];
  for (let start = 0; start < length; start += size) {
    slabs.push({ start, end: Math.min(start + size, length) });
  }
  return slabs;
}

/**
 * The inputs for one slab, as standalone rank-1 fields.
 *
 * `slice` copies rather than views: a subarray shares its source buffer, and structured-cloning
 * one can carry the whole 168 MB array to the worker instead of the slab.
 */
export function sliceInputs(
  inputs: Readonly<Record<string, NpyN>>,
  slab: Slab,
): Record<string, NpyN> {
  const sliced: Record<string, NpyN> = {};
  for (const [name, input] of Object.entries(inputs)) {
    const data = input.data.slice(slab.start, slab.end);
    sliced[name] = { shape: [data.length], data };
  }
  return sliced;
}

/** Total bytes one slab request carries, for admission checks. */
export function slabRequestBytes(
  inputs: Readonly<Record<string, NpyN>>,
  slab: Slab,
): number {
  const cells = slab.end - slab.start;
  let total = 0;
  for (const input of Object.values(inputs)) {
    total += cells * (input.data.BYTES_PER_ELEMENT ?? 4);
  }
  return total;
}
