import { useEffect } from "react";
import { usePlaybackContext } from "@/components/plots/playbackContext";
import { lookupVtkPlaybackScreenResolution } from "./vtkPlaybackScreenResolution";

export function useVtkPlaybackScreenResolution(getWindow: () => object | null, generation: number, enabled = true): void {
  const { playing, recording, interpolationDurationMs = 500, frameIntervalMs = interpolationDurationMs } = usePlaybackContext();
  useEffect(() => {
    lookupVtkPlaybackScreenResolution(getWindow())?.setPlayback({ playing: playing && enabled, recording, intervalMs: frameIntervalMs });
  }, [getWindow, generation, enabled, playing, recording, frameIntervalMs]);
}
