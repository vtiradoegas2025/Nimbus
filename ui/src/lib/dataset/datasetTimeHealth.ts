export interface TimedFrame {
  simulationTimeS: number | null;
}

export interface DatasetTimeHealth {
  expectedCadenceS: number | null;
  duplicateCount: number;
  gapCount: number;
  nonMonotonicCount: number;
  missingTimeCount: number;
  hasWarnings: boolean;
  summary: string;
}

function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

export function analyzeDatasetTime(frames: readonly TimedFrame[]): DatasetTimeHealth {
  const times = frames.map((frame) => frame.simulationTimeS);
  const missingTimeCount = times.filter((time) => time == null || !Number.isFinite(time)).length;
  const finite = times.filter((time): time is number => time != null && Number.isFinite(time));
  const deltas = finite.slice(1).map((time, index) => time - finite[index]);
  const scale = Math.max(1, ...finite.map(Math.abs));
  const tolerance = scale * 1e-9;
  const duplicateCount = deltas.filter((delta) => Math.abs(delta) <= tolerance).length;
  const nonMonotonicCount = deltas.filter((delta) => delta < -tolerance).length;
  const positiveDeltas = deltas.filter((delta) => delta > tolerance);
  const positiveMedian = median(positiveDeltas);
  const expectedCadenceS = positiveMedian == null
    ? null
    : median(positiveDeltas.filter((delta) => delta <= positiveMedian + tolerance));
  const gapCount = expectedCadenceS == null
    ? 0
    : positiveDeltas.filter((delta) => delta > expectedCadenceS * 1.5 && delta - expectedCadenceS > tolerance).length;
  const parts: string[] = [];
  if (duplicateCount > 0) parts.push(`${duplicateCount} duplicate ${duplicateCount === 1 ? "time" : "times"}`);
  if (gapCount > 0) parts.push(`${gapCount} likely output ${gapCount === 1 ? "gap" : "gaps"}`);
  if (nonMonotonicCount > 0) parts.push(`${nonMonotonicCount} reversed ${nonMonotonicCount === 1 ? "step" : "steps"}`);
  if (missingTimeCount > 0) parts.push(`${missingTimeCount} ${missingTimeCount === 1 ? "frame" : "frames"} without physical time`);
  return {
    expectedCadenceS,
    duplicateCount,
    gapCount,
    nonMonotonicCount,
    missingTimeCount,
    hasWarnings: parts.length > 0,
    summary: parts.length > 0
      ? `${parts.join("; ")}. Frames are retained; verify the selected history series.`
      : expectedCadenceS == null
        ? "Not enough physical timestamps to assess cadence."
        : `Time coordinates are monotonic with an expected ${Number(expectedCadenceS.toPrecision(6))} s cadence.`,
  };
}
