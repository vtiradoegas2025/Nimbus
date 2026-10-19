export interface PlaybackScreenResolutionState {
  distance: number;
  recentMs: number[];
  samplesSinceChange: number;
  slowSamples: number;
  fastSamples: number;
}

export function fullPlaybackScreenResolution(): PlaybackScreenResolutionState {
  return { distance: 1, recentMs: [], samplesSinceChange: 0, slowSamples: 0, fastSamples: 0 };
}

const DISTANCES = [1, 1.25, 1.5, 2];

export function observePlaybackGpuDuration(
  state: PlaybackScreenResolutionState,
  gpuMs: number,
  intervalMs: number,
): PlaybackScreenResolutionState {
  if (!(gpuMs > 0) || gpuMs > 10_000 || !Number.isFinite(gpuMs)
      || !(intervalMs > 0) || !Number.isFinite(intervalMs)) return state;
  const recentMs = [...state.recentMs, gpuMs].slice(-5);
  const sorted = [...recentMs].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const index = DISTANCES.indexOf(state.distance);
  const samplesSinceChange = state.samplesSinceChange + 1;
  const slowSamples = median > intervalMs * 1.15 ? state.slowSamples + 1 : 0;
  const nextFiner = DISTANCES[Math.max(0, index - 1)];
  const projectedFinerMs = median * (state.distance / nextFiner) ** 2;
  const fastSamples = projectedFinerMs < intervalMs * 0.55 ? state.fastSamples + 1 : 0;
  const canChange = recentMs.length >= 3 && (state.distance === 1 || samplesSinceChange >= 6);
  const next = canChange && slowSamples >= 3 && index < DISTANCES.length - 1 ? DISTANCES[index + 1]
    : canChange && fastSamples >= 10 && index > 0 ? nextFiner : state.distance;
  return next === state.distance
    ? { distance: next, recentMs, samplesSinceChange, slowSamples, fastSamples }
    : { distance: next, recentMs: [], samplesSinceChange: 0, slowSamples: 0, fastSamples: 0 };
}
