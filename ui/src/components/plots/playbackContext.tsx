import { createContext, useContext, type ReactNode } from "react";

/**
 * Playback state passed via context so PlotRouter doesn't need to thread it through all 17 plot props
 * (same reason as PaneContext).
 *
 * The scrub debounce exists to skip frames swept through on a drag. Playback is the opposite intent:
 * every frame is deliberately requested, so debouncing there only delays each read and strobes the
 * frame-transition overlay. The recorder already bypasses the debounce for the same reason.
 */
export interface PlaybackContextValue {
  /** True while the frame slider is auto-advancing. */
  playing: boolean;
  /** True while the recorder is driving exact, full-fidelity frames. */
  recording: boolean;
  interpolationDurationMs?: number;
  frameIntervalMs?: number;
}

const NOT_PLAYING: PlaybackContextValue = { playing: false, recording: false };

const PlaybackContext = createContext<PlaybackContextValue>(NOT_PLAYING);

export function PlaybackProvider({
  value,
  children,
}: {
  value: PlaybackContextValue;
  children: ReactNode;
}) {
  return <PlaybackContext.Provider value={value}>{children}</PlaybackContext.Provider>;
}

export function usePlaybackContext(): PlaybackContextValue {
  return useContext(PlaybackContext);
}
