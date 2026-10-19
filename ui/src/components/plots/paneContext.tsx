import { createContext, useContext, type ReactNode } from "react";
import type { CameraSyncBus } from "@/lib/plot/cameraSyncBus";

/**
 * Pane identity passed via context so PlotRouter doesn't need to thread it through all 17 plot props.
 * Outside a grid the default fires (no bus, no sync, full voxel budget).
 */
export interface PaneContextValue {
  /** Null when this plot is not in a multi-pane grid. */
  bus: CameraSyncBus | null;
  paneId: string;
  /** False when the user has unlocked cameras. */
  cameraLocked: boolean;
  /** Pane that owns programmatic camera commands while cameras are locked. */
  cameraAuthority: boolean;
  /** Fraction of the single-pane voxel budget; scaled down per pane to avoid VRAM exhaustion. */
  voxelBudgetScale: number;
  /** True for exactly one pane in a locked grid — that pane draws the compass / shared overlays. */
  ownsSharedOverlays: boolean;
}

const SINGLE_PANE: PaneContextValue = {
  bus: null,
  paneId: "primary",
  cameraLocked: false,
  cameraAuthority: true,
  voxelBudgetScale: 1,
  ownsSharedOverlays: true,
};

const PaneContext = createContext<PaneContextValue>(SINGLE_PANE);

export function PaneProvider({
  value,
  children,
}: {
  value: PaneContextValue;
  children: ReactNode;
}) {
  return <PaneContext.Provider value={value}>{children}</PaneContext.Provider>;
}

export function usePaneContext(): PaneContextValue {
  return useContext(PaneContext);
}
