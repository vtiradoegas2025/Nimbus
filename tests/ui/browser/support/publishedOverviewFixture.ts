import { makeDerivedBrickHierarchy } from "./fixtures";

export const OVERVIEW_RUN = "/__nimbus_browser_fixture__/bricked-derived";
export type OverviewHierarchyMode = "durable" | "native-only" | "changed-publication" | "invalid-shape";
export const overviewCalls: { command: string; frame: number; publication: unknown; level: unknown; request: unknown }[] = [];
let hierarchyMode: OverviewHierarchyMode = "durable";
let metadataDelayMs = 0;

export function setOverviewHierarchyMode(mode: OverviewHierarchyMode): void { hierarchyMode = mode; }
export function setOverviewMetadataDelay(milliseconds: number): void { metadataDelayMs = milliseconds; }

export async function publishedOverviewInvoke(command: string, payload: unknown): Promise<{ value: unknown } | null> {
  const args = (payload ?? {}) as Record<string, unknown>;
  if (args.runDir !== OVERVIEW_RUN || !["read_field_brick_hierarchy", "read_field_brick_window", "read_field_window", "read_field_bytes"].includes(command)) return null;
  overviewCalls.push({ command, frame: Number(args.stepIndex), publication: args.expectedGeneration,
    level: args.level, request: args.request });
  if (command !== "read_field_brick_hierarchy") return null;
  const hierarchy = structuredClone(makeDerivedBrickHierarchy(Number(args.stepIndex), String(args.fieldName)));
  if (hierarchyMode === "native-only") hierarchy.levels = hierarchy.levels.slice(0, 1);
  if (hierarchyMode === "changed-publication") hierarchy.publicationGeneration += "-changed-after-publication";
  if (hierarchyMode === "invalid-shape") hierarchy.sourceShape[0] += 1;
  if (metadataDelayMs > 0) await new Promise(resolve => setTimeout(resolve, metadataDelayMs));
  return { value: hierarchy };
}
