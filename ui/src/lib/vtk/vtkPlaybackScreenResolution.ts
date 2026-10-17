import { fullPlaybackScreenResolution, observePlaybackGpuDuration } from "./playbackScreenResolutionPolicy";
import { createVtkGpuRenderTimer, type GpuRenderTimer, type GpuRenderTimingStatus } from "./vtkGpuRenderTimer";
import { subscribeVtkRenderTraversal } from "./vtkRenderTraversal";

export interface PlaybackScreenMapper {
  getImageSampleDistance(): number;
  setImageSampleDistance(value: number): unknown;
}
export interface PlaybackScreenWindow {
  getRenderWindow(): { render(): unknown };
  getApiSpecificRenderWindow(): { getContext?: () => WebGL2RenderingContext | null };
}
export interface PlaybackScreenResolutionSnapshot {
  playing: boolean;
  distance: number;
  timing: GpuRenderTimingStatus;
  measuredGpuMs: number | null;
  completedSamples: number;
  captureHeld: boolean;
}
export interface VtkPlaybackScreenResolution {
  setPlayback(input: { playing: boolean; recording: boolean; intervalMs: number }): void;
  holdFullResolution(): () => void;
  describe(): PlaybackScreenResolutionSnapshot;
  dispose(): void;
}

const controllers = new WeakMap<object, VtkPlaybackScreenResolution>();
export function lookupVtkPlaybackScreenResolution(window: object | null | undefined): VtkPlaybackScreenResolution | null {
  return window ? controllers.get(window) ?? null : null;
}
export function holdFullVtkPlaybackResolution(window: object): () => void {
  return controllers.get(window)?.holdFullResolution() ?? (() => {});
}

export function attachVtkPlaybackScreenResolution(window: PlaybackScreenWindow, options: {
  getMappers: () => readonly PlaybackScreenMapper[];
  shouldMeasure?: () => boolean;
  onChange?: (snapshot: PlaybackScreenResolutionSnapshot) => void;
  timer?: GpuRenderTimer;
}): VtkPlaybackScreenResolution {
  let state = fullPlaybackScreenResolution();
  let playing = false, intervalMs = 500, captures = 0, disposed = false;
  let measuredGpuMs: number | null = null, completedSamples = 0;
  const timer = options.timer ?? createVtkGpuRenderTimer(() => window.getApiSpecificRenderWindow().getContext?.() ?? null);
  const mappers = new Map<PlaybackScreenMapper, { baseline: number; applied: number }>();
  const distance = () => playing && captures === 0 ? state.distance : 1;
  const describe = (): PlaybackScreenResolutionSnapshot => ({ playing, distance: distance(), timing: timer.status(), measuredGpuMs,
    completedSamples, captureHeld: captures > 0 });
  const notify = () => options.onChange?.(describe());
  const apply = () => {
    for (const [mapper, value] of mappers) {
      value.applied = captures > 0 ? 1 : Math.max(value.baseline, distance());
      if (mapper.getImageSampleDistance() !== value.applied) mapper.setImageSampleDistance(value.applied);
    }
  };
  const synchronizeMappers = () => {
    const current = new Set(options.getMappers());
    for (const mapper of mappers.keys()) if (!current.has(mapper)) mappers.delete(mapper);
    for (const mapper of current) {
      const currentDistance = mapper.getImageSampleDistance();
      const prior = mappers.get(mapper);
      if (!prior) mappers.set(mapper, { baseline: currentDistance, applied: currentDistance });
      else if (currentDistance !== prior.applied) prior.baseline = currentDistance;
    }
    apply();
  };
  const core = window.getRenderWindow();
  const measurements: boolean[] = [];
  const detachTraversal = subscribeVtkRenderTraversal(window.getApiSpecificRenderWindow(), { before() {
    if (disposed) { measurements.push(false); return; }
    synchronizeMappers();
    const measured = measurements.length === 0 && playing && captures === 0 && (options.shouldMeasure?.() ?? true) && timer.begin();
    measurements.push(measured);
    if (playing && !measured && ["unavailable", "disjoint", "failed"].includes(timer.status()) && state.distance !== 1) {
      state = fullPlaybackScreenResolution(); apply();
    }
  }, after() {
    if (measurements.pop()) timer.end(milliseconds => {
      if (disposed || !playing || captures > 0 || !(options.shouldMeasure?.() ?? true)) return;
      measuredGpuMs = milliseconds; completedSamples++;
      const previous = state.distance;
      state = observePlaybackGpuDuration(state, milliseconds, intervalMs);
      if (state.distance !== previous) apply();
      notify();
    });
    if (playing && timer.status() !== "pending") notify();
  } });
  const controller: VtkPlaybackScreenResolution = {
    setPlayback(input) {
      const nextPlaying = input.playing && !input.recording && Number.isFinite(input.intervalMs) && input.intervalMs > 0;
      if (playing === nextPlaying && intervalMs === input.intervalMs) return;
      const continuesPlayback = playing && nextPlaying;
      playing = nextPlaying; intervalMs = input.intervalMs;
      if (continuesPlayback) state = { ...state, slowSamples: 0, fastSamples: 0 };
      else { timer.cancel(); state = fullPlaybackScreenResolution(); }
      synchronizeMappers(); apply(); notify();
      if (!playing && !disposed) core.render();
    },
    holdFullResolution() {
      captures++; timer.cancel(); synchronizeMappers(); apply(); notify();
      let released = false;
      return () => { if (released) return; released = true; captures = Math.max(0, captures - 1); if (!disposed) { apply(); notify(); } };
    },
    describe,
    dispose() {
      if (disposed) return;
      disposed = true; playing = false; timer.dispose();
      const current = new Set(options.getMappers());
      for (const [mapper, value] of mappers) if (current.has(mapper)) mapper.setImageSampleDistance(value.baseline);
      mappers.clear();
      detachTraversal();
      controllers.delete(window);
    },
  };
  controllers.set(window, controller);
  return controller;
}
