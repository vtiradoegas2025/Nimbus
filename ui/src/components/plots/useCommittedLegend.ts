import { useRef } from "react";

export interface CommittedLegend<T> { scope: string; value: T | null }

export function committedLegend<T>(previous: CommittedLegend<T> | null, scope: string,
  committed: boolean, candidate: T | null): CommittedLegend<T> | null {
  if (committed) return { scope, value: candidate };
  return previous?.scope === scope ? previous : null;
}

export function useCommittedLegend<T>(scope: string, committed: boolean, candidate: T | null): T | null {
  const snapshot = useRef<CommittedLegend<T> | null>(null);
  snapshot.current = committedLegend(snapshot.current, scope, committed, candidate);
  return snapshot.current?.value ?? null;
}
