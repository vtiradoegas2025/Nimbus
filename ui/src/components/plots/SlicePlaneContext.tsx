import type { SliceAxis } from "@/lib/platform/npy";
import type { GridInfo } from "@/lib/run/runOutputs";
import { slicePlaneDescription } from "@/lib/plot/slicePlaneDescription";

export function SlicePlaneContext({ grid, axis, index, physicalHeight }: {
  grid: GridInfo | null | undefined; axis: SliceAxis; index: number | null; physicalHeight?: boolean;
}) {
  const description = index == null ? { plane: "Section", location: "Preparing section…" }
    : slicePlaneDescription(grid, axis, index, physicalHeight);
  return <div className="plot-section-context" aria-label="Displayed section position">
    <strong>{description.plane}</strong><span>{description.location}</span>
  </div>;
}
