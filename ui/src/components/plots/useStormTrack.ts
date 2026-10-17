import { useEffect, useState } from "react";
import { readStormTrack } from "@/lib/run/runOutputs";
import type { StormTrackSnapshot } from "@/lib/run/stormTrack";
import { stormTrackMatchesFrame } from "@/lib/run/stormTrack";
export { stormTrackCentre } from "@/lib/run/stormTrack";

/**
 * Mesocyclone track snapshot for the current frame. Null when storm_tracking.enabled is false.
 * Read per frame identity (not polled) — scrubbing or live advancement picks it up automatically.
 */
export function useStormTrack(
  runDir: string,
  stepIndex: number | null,
  simulationTimeS: number | null,
  publicationGeneration: string | null,
): StormTrackSnapshot | null {
  const [snapshot, setSnapshot] = useState<StormTrackSnapshot | null>(null);

  useEffect(() => {
    let cancelled = false;
    readStormTrack(runDir, publicationGeneration).then(
      (next) => {
        if (!cancelled) {
          // The command reads run-level `storm_track_latest.json`. While
          // scrubbing, that snapshot may belong to a newer field frame.
          setSnapshot(stormTrackMatchesFrame(next, simulationTimeS) ? next : null);
        }
      },
      () => {
        // `readStormTrack` already folds absence and mid-write reads into null;
        // reaching here at all means something unexpected, and a stale marker
        // is worse than none.
        if (!cancelled) setSnapshot(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [runDir, stepIndex, simulationTimeS, publicationGeneration]);

  return snapshot;
}
