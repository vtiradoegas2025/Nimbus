import type { FieldLoadWindow } from "./runOutputs";
import { MAX_NATIVE_FIELD_WINDOW_BYTES, nativeFieldWindowBodyBytes } from "@/lib/dataset/nativeFieldWindow";

export function partitionDerivedWindow(
  window: FieldLoadWindow,
  terrain = false,
  maximumBytes = 16 * 1024 * 1024,
): FieldLoadWindow[] {
  const rank = window.extent.length;
  if (rank < 1 || rank > 3 || window.origin.length !== rank || window.stride.length !== rank
      || window.extent.some(value => !Number.isSafeInteger(value) || value < 1)
      || window.stride.some(value => !Number.isSafeInteger(value) || value < 1)
      || window.origin.some(value => !Number.isSafeInteger(value) || value < 0)
      || !Number.isSafeInteger(maximumBytes) || maximumBytes < 1) {
    throw new Error("Invalid derived-field window partition");
  }
  const outputElements = window.extent.reduce((size, count, axis) => size * Math.ceil(count / window.stride[axis]), 1);
  const outputBytes = nativeFieldWindowBodyBytes(outputElements, terrain ? 32 : 0);
  const outputBudget = window.maxOutputBytes ?? MAX_NATIVE_FIELD_WINDOW_BYTES;
  if (!Number.isSafeInteger(outputBudget) || outputBudget < 1 || outputBudget > MAX_NATIVE_FIELD_WINDOW_BYTES
      || outputBytes > outputBudget) throw new Error("Derived-field output exceeds its native window budget");
  const budget = Math.min(maximumBytes, MAX_NATIVE_FIELD_WINDOW_BYTES);
  const result: FieldLoadWindow[] = [];
  const visit = (tile: FieldLoadWindow): void => {
    const elements = tile.extent.reduce((size, count) => size * count, 1);
    const bytes = elements * (terrain ? 36 : 4) + (terrain ? 7 : 0);
    if (!Number.isSafeInteger(bytes)) throw new Error("Derived-field window size overflow");
    if (bytes <= budget) { result.push(tile); return; }
    const counts = tile.extent.map((count, axis) => Math.ceil(count / tile.stride[axis]));
    const axis = counts.indexOf(Math.max(...counts));
    if (counts[axis] < 2) throw new Error("One derived-field aggregation cell exceeds the native dependency budget");
    // Cuts fall between output cells so nonlinear evaluation still precedes
    // every complete aggregation bin, including the final partial bin.
    const cut = Math.floor(counts[axis] / 2) * tile.stride[axis];
    const lower = { ...tile, origin: [...tile.origin], extent: [...tile.extent] };
    const upper = { ...tile, origin: [...tile.origin], extent: [...tile.extent] };
    lower.extent[axis] = cut;
    upper.origin[axis] += cut;
    upper.extent[axis] -= cut;
    visit(lower); visit(upper);
  };
  visit(window);
  return result;
}
