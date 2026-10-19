export const COMPARISON_WORKSPACE_SCHEMA = "nimbus.studio.comparison-workspace" as const;
export const COMPARISON_WORKSPACE_VERSION = 4 as const;

export interface ComparisonWorkspaceRecipe {
  schema: typeof COMPARISON_WORKSPACE_SCHEMA;
  schemaVersion: typeof COMPARISON_WORKSPACE_VERSION;
  leftIdentity: string;
  rightIdentity: string;
  fieldName: string;
  sliceAxis: 0 | 1 | 2;
  sliceIndex: number;
  rangeMode: "auto" | "symmetric" | "dBZ";
  colormap: "Viridis" | "Plasma" | "Inferno" | "RdBu" | "Greys" | "dBZ" | "ColdPool";
  metricScope: "slice" | "region" | "volume";
  showMetricSeries: boolean;
  showVectorComparison: boolean;
  showVolumeComparison: boolean;
  showCompositeComparison: boolean;
  viewportBounds: { xMin: number; xMax: number; yMin: number; yMax: number } | null;
  savedAt: string;
}

function workspaceHash(leftIdentity: string, rightIdentity: string): string {
  let hash = 0x811c9dc5;
  const text = JSON.stringify([leftIdentity, rightIdentity]);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function comparisonWorkspaceKey(leftIdentity: string, rightIdentity: string): string {
  return `nimbus:comparison-workspace:v4:${workspaceHash(leftIdentity, rightIdentity)}`;
}

export function validComparisonWorkspaceRecipe(
  value: unknown,
  leftIdentity: string,
  rightIdentity: string,
): value is ComparisonWorkspaceRecipe {
  if (!value || typeof value !== "object") return false;
  const recipe = value as Partial<ComparisonWorkspaceRecipe>;
  const bounds = recipe.viewportBounds;
  const validBounds = bounds === null || (
    typeof bounds === "object"
    && [bounds?.xMin, bounds?.xMax, bounds?.yMin, bounds?.yMax].every((coordinate) => Number.isFinite(coordinate))
    && bounds!.xMin < bounds!.xMax
    && bounds!.yMin < bounds!.yMax
  );
  return recipe.schema === COMPARISON_WORKSPACE_SCHEMA
    && recipe.schemaVersion === COMPARISON_WORKSPACE_VERSION
    && recipe.leftIdentity === leftIdentity
    && recipe.rightIdentity === rightIdentity
    && typeof recipe.fieldName === "string"
    && [0, 1, 2].includes(recipe.sliceAxis as number)
    && Number.isInteger(recipe.sliceIndex) && (recipe.sliceIndex ?? -1) >= 0
    && ["auto", "symmetric", "dBZ"].includes(recipe.rangeMode ?? "")
    && ["Viridis", "Plasma", "Inferno", "RdBu", "Greys", "dBZ", "ColdPool"].includes(recipe.colormap ?? "")
    && ["slice", "region", "volume"].includes(recipe.metricScope ?? "")
    && typeof recipe.showMetricSeries === "boolean"
    && typeof recipe.showVectorComparison === "boolean"
    && typeof recipe.showVolumeComparison === "boolean"
    && typeof recipe.showCompositeComparison === "boolean"
    && validBounds
    && typeof recipe.savedAt === "string";
}

export function loadComparisonWorkspace(
  storage: Pick<Storage, "getItem">,
  leftIdentity: string,
  rightIdentity: string,
): ComparisonWorkspaceRecipe | null {
  try {
    const serialized = storage.getItem(comparisonWorkspaceKey(leftIdentity, rightIdentity));
    if (!serialized) return null;
    const parsed: unknown = JSON.parse(serialized);
    return validComparisonWorkspaceRecipe(parsed, leftIdentity, rightIdentity) ? parsed : null;
  } catch {
    return null;
  }
}

export function saveComparisonWorkspace(
  storage: Pick<Storage, "setItem">,
  recipe: ComparisonWorkspaceRecipe,
): boolean {
  // Best-effort: these keys are per dataset pair and accumulate, so quota
  // exhaustion is plausible in a long session and must not fail the caller.
  try {
    storage.setItem(comparisonWorkspaceKey(recipe.leftIdentity, recipe.rightIdentity), JSON.stringify(recipe));
    return true;
  } catch {
    // Storage full or blocked — the in-memory workspace stays authoritative.
    return false;
  }
}
