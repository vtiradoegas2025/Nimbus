import type { PhysicalPoint2D } from "@/lib/plot/transectSampling";

export const INVESTIGATION_WORKSPACE_SCHEMA = "nimbus.studio.investigation-workspace" as const;
export const INVESTIGATION_WORKSPACE_VERSION = 1 as const;

export interface InvestigationAnnotation {
  id: string;
  text: string;
  fieldName: string;
  simulationTimeS: number;
  point: PhysicalPoint2D;
  distanceFraction: number;
  zIndex: number;
  createdAt: string;
}

export interface InvestigationWorkspaceRecipe {
  schema: typeof INVESTIGATION_WORKSPACE_SCHEMA;
  schemaVersion: typeof INVESTIGATION_WORKSPACE_VERSION;
  datasetIdentity: string;
  stepIndex: number;
  fieldName: string;
  zIndex: number;
  start: PhysicalPoint2D;
  end: PhysicalPoint2D;
  probeFraction: number;
  annotations: InvestigationAnnotation[];
  savedAt: string;
}

function hashIdentity(identity: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < identity.length; index += 1) { hash ^= identity.charCodeAt(index); hash = Math.imul(hash, 0x01000193); }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function investigationWorkspaceKey(identity: string): string { return `nimbus:investigation-workspace:v1:${hashIdentity(identity)}`; }

function point(value: unknown): value is PhysicalPoint2D {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<PhysicalPoint2D>;
  return Number.isFinite(candidate.xKm) && Number.isFinite(candidate.yKm);
}

export function validInvestigationWorkspace(value: unknown, identity: string): value is InvestigationWorkspaceRecipe {
  if (!value || typeof value !== "object") return false;
  const recipe = value as Partial<InvestigationWorkspaceRecipe>;
  return recipe.schema === INVESTIGATION_WORKSPACE_SCHEMA
    && recipe.schemaVersion === INVESTIGATION_WORKSPACE_VERSION
    && recipe.datasetIdentity === identity
    && Number.isInteger(recipe.stepIndex) && (recipe.stepIndex ?? -1) >= 0
    && typeof recipe.fieldName === "string" && recipe.fieldName.length > 0
    && Number.isInteger(recipe.zIndex) && (recipe.zIndex ?? -1) >= 0
    && point(recipe.start) && point(recipe.end)
    && Number.isFinite(recipe.probeFraction) && recipe.probeFraction! >= 0 && recipe.probeFraction! <= 1
    && Array.isArray(recipe.annotations) && recipe.annotations.length <= 100
    && recipe.annotations.every((annotation) => annotation && typeof annotation.id === "string"
      && typeof annotation.text === "string" && annotation.text.trim().length > 0 && annotation.text.length <= 500
      && typeof annotation.fieldName === "string" && Number.isFinite(annotation.simulationTimeS)
      && point(annotation.point) && Number.isFinite(annotation.distanceFraction)
      && annotation.distanceFraction >= 0 && annotation.distanceFraction <= 1
      && Number.isInteger(annotation.zIndex) && annotation.zIndex >= 0 && typeof annotation.createdAt === "string")
    && typeof recipe.savedAt === "string";
}

export function loadInvestigationWorkspace(storage: Pick<Storage, "getItem">, identity: string): InvestigationWorkspaceRecipe | null {
  try { const text = storage.getItem(investigationWorkspaceKey(identity)); if (!text) return null; const value: unknown = JSON.parse(text); return validInvestigationWorkspace(value, identity) ? value : null; } catch { return null; }
}

/** Returns false when browser persistence is unavailable or over quota. */
export function saveInvestigationWorkspace(storage: Pick<Storage, "setItem">, recipe: InvestigationWorkspaceRecipe): boolean {
  // Best-effort: these keys are per dataset and accumulate, so quota
  // exhaustion is plausible in a long session and must not fail the caller.
  try {
    storage.setItem(investigationWorkspaceKey(recipe.datasetIdentity), JSON.stringify(recipe));
    return true;
  } catch {
    // Storage full or blocked — the in-memory workspace stays authoritative.
    return false;
  }
}

export function clearInvestigationWorkspace(storage: Pick<Storage, "removeItem">, identity: string): void {
  storage.removeItem(investigationWorkspaceKey(identity));
}
