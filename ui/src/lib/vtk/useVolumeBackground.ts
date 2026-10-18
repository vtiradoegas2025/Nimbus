import { useEffect, useState } from "react";
import { useIoBudgetRetry } from "@/lib/core/useIoBudgetRetry";
import { readField, type StepInfo } from "@/lib/run/runOutputs";
import { prepareVtkInWorker } from "./vtkPreparationClient";
import { volumeBackgroundProfile, type VolumeBackground } from "./volumeBackground";
import { domainBoundsM } from "@/lib/plot/domainAxes";

const cache = new Map<string, VolumeBackground>();

export function useVolumeBackground(enabled: boolean, runDir: string, step: StepInfo | null | undefined, field: string | null) {
  const key = JSON.stringify([runDir, step?.index, step?.publicationGeneration, field]);
  const [state, setState] = useState<{ key: string; profile: VolumeBackground | null; error: string | null }>({ key: "", profile: null, error: null });
  const budgetRetry = useIoBudgetRetry(state.error);
  useEffect(() => {
    if (!enabled || !step?.grid || !field) return;
    const cached = cache.get(key);
    if (cached) { setState({ key, profile: cached, error: null }); return; }
    const controller = new AbortController();
    const grid = step.grid;
    void (async () => {
      const counts = [grid.nr!, grid.nth!, grid.nz!];
      const origin = grid.validIndexRange?.lower ?? [0, 0, 0];
      const upper = grid.validIndexRange?.upper ?? counts.map(value => value - 1);
      const extent = upper.map((value, axis) => value - origin[axis] + 1);
      const columns = Math.max(1, Math.min(32, Math.floor(Math.sqrt(1048576 / extent[2]))));
      const stride = [Math.max(1, Math.ceil(extent[0] / columns)), Math.max(1, Math.ceil(extent[1] / columns)), 1];
      const samples = extent.reduce((product, value, axis) => product * Math.ceil(value / stride[axis]), 1);
      if (!Number.isSafeInteger(samples) || samples < 1 || samples > 1048576) throw new Error("Background sampling exceeds its bounded vertical profile budget");
      const source = await readField(runDir, step.index, field, step.publicationGeneration, {
        signal: controller.signal, cacheRetention: "transient", window: {
          origin, extent, stride, aggregation: "point_sample", maxOutputBytes: samples * (grid.terrainFollowing ? 36 : 4) + (grid.terrainFollowing ? 8 : 0), maxSourceBytes: 256 * 1024 * 1024,
        },
      });
      const image = await prepareVtkInWorker({ kind: "scalar", field: source, grid, maxVoxels: 1048576 }, controller.signal);
      const bounds = domainBoundsM(grid);
      const marginX = Math.min((grid.drM ?? image.spacing[0]) * 4, (bounds ? bounds.upper[0] - bounds.lower[0] : Infinity) * 0.1);
      const marginY = Math.min((grid.dyM ?? grid.drM ?? image.spacing[1]) * 4, (bounds ? bounds.upper[1] - bounds.lower[1] : Infinity) * 0.1);
      const profile = volumeBackgroundProfile(image, field === "vorticity_magnitude", bounds ?
        [bounds.lower[0] + marginX, bounds.upper[0] - marginX, bounds.lower[1] + marginY, bounds.upper[1] - marginY] : undefined,
        [marginX, marginY]);
      if (bounds) profile.domain = [bounds.lower[0], bounds.upper[0], bounds.lower[1], bounds.upper[1]];
      if (controller.signal.aborted) return;
      cache.set(key, profile);
      while (cache.size > 50) cache.delete(cache.keys().next().value!);
      setState({ key, profile, error: null });
    })().catch(error => { if (!controller.signal.aborted) setState({ key, profile: null, error: String(error) }); });
    return () => controller.abort();
  }, [enabled, key, budgetRetry]);
  return { profile: enabled && state.key === key ? state.profile : null,
    loading: enabled && (state.key !== key || (!state.profile && !state.error)),
    error: enabled && state.key === key ? state.error : null };
}
