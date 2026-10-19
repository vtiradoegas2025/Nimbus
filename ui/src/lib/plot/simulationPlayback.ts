export interface SimulationPlaybackTimeline {
  timesS: readonly number[];
  cycleS: number;
}

export function simulationPlaybackTimeline(
  frames: readonly { simulationTimeS?: number | null }[],
): SimulationPlaybackTimeline | null {
  if (frames.length < 2) return null;
  const timesS = frames.map(frame => frame.simulationTimeS);
  if (timesS.some((time, index) => typeof time !== "number" || !Number.isFinite(time)
      || (index > 0 && time <= (timesS[index - 1] as number)))) return null;
  const times = timesS as number[];
  const intervals = times.slice(1).map((time, index) => time - times[index]).sort((a, b) => a - b);
  const middle = Math.floor(intervals.length / 2);
  const loopDwell = intervals.length % 2 ? intervals[middle] : (intervals[middle - 1] + intervals[middle]) / 2;
  const cycleS = times[times.length - 1] - times[0] + loopDwell;
  return Number.isFinite(cycleS) ? { timesS: times, cycleS } : null;
}

export function simulationPlaybackIntervalMs(
  timeline: SimulationPlaybackTimeline,
  index: number,
  stride: number,
  rate: number,
  minimumMs: number,
): number | null {
  const count = timeline.timesS.length;
  if (!Number.isInteger(index) || index < 0 || index >= count
      || !Number.isInteger(stride) || stride < 1 || !Number.isFinite(rate) || rate <= 0
      || !Number.isFinite(minimumMs) || minimumMs < 0) return null;
  const next = (index + Math.min(stride, count - 1)) % count;
  // A loop has no physical forward time across its discontinuity. Hold that
  // boundary for one typical output interval, also counting skipped frames.
  const elapsedS = timeline.timesS[next] - timeline.timesS[index]
    + (next <= index ? timeline.cycleS : 0);
  const interval = elapsedS * 1000 / rate;
  return Number.isFinite(interval) ? Math.max(minimumMs, interval) : null;
}
