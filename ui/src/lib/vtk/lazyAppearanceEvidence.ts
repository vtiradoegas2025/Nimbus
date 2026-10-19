// Prepared arrays are immutable within a generation. Capturing evidence may
// scan them, but camera updates and frame playback must never pay that cost.
export function lazyAppearanceEvidence<T>(compute: () => T): () => T {
  let cached: { value: T } | undefined;
  return () => (cached ??= { value: compute() }).value;
}
