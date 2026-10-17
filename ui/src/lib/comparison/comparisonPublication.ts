import {
  DATASET_ALIGNMENT_SCHEMA,
  DATASET_ALIGNMENT_VERSION,
  type DatasetAlignmentReport,
} from "@/lib/dataset/datasetAlignment";
import { datasetSourceIdentity, type LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { validComparisonWorkspaceRecipe, type ComparisonWorkspaceRecipe } from "./comparisonWorkspace";
import { isLocalDatasetSource } from "./comparisonSession";

export const COMPARISON_PUBLICATION_SCHEMA = "nimbus.studio.comparison-publication-workspace" as const;
export const COMPARISON_PUBLICATION_VERSION = 1 as const;

export interface ComparisonPublicationWorkspace {
  schema: typeof COMPARISON_PUBLICATION_SCHEMA;
  schemaVersion: typeof COMPARISON_PUBLICATION_VERSION;
  title: string;
  exportedAt: string;
  left: LocalDatasetSource;
  right: LocalDatasetSource;
  alignment: DatasetAlignmentReport;
  recipe: ComparisonWorkspaceRecipe | null;
}

export function createComparisonPublicationWorkspace(args: {
  left: LocalDatasetSource;
  right: LocalDatasetSource;
  alignment: DatasetAlignmentReport;
  recipe: ComparisonWorkspaceRecipe | null;
}): ComparisonPublicationWorkspace {
  return {
    schema: COMPARISON_PUBLICATION_SCHEMA,
    schemaVersion: COMPARISON_PUBLICATION_VERSION,
    title: `${args.left.displayName} versus ${args.right.displayName}`,
    exportedAt: new Date().toISOString(),
    ...args,
  };
}

export function serializeComparisonPublication(workspace: ComparisonPublicationWorkspace): string {
  return `${JSON.stringify(workspace, null, 2)}\n`;
}

export function parseComparisonPublication(text: string): ComparisonPublicationWorkspace {
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch { throw new Error("Comparison workspace is not valid JSON."); }
  if (!parsed || typeof parsed !== "object") throw new Error("Comparison workspace root must be an object.");
  const workspace = parsed as Partial<ComparisonPublicationWorkspace>;
  if (workspace.schema !== COMPARISON_PUBLICATION_SCHEMA || workspace.schemaVersion !== COMPARISON_PUBLICATION_VERSION) {
    throw new Error("Unsupported comparison workspace schema or version.");
  }
  if (!isLocalDatasetSource(workspace.left) || !isLocalDatasetSource(workspace.right)) {
    throw new Error("Comparison workspace contains an invalid dataset source.");
  }
  if (!workspace.alignment
    || workspace.alignment.schema !== DATASET_ALIGNMENT_SCHEMA
    || workspace.alignment.schemaVersion !== DATASET_ALIGNMENT_VERSION
    || workspace.alignment.leftDatasetIdentity !== datasetSourceIdentity(workspace.left)
    || workspace.alignment.rightDatasetIdentity !== datasetSourceIdentity(workspace.right)
    || !["exact", "partial", "blocked"].includes(workspace.alignment.verdict)
    || typeof workspace.alignment.differenceAllowed !== "boolean") {
    throw new Error("Comparison workspace is missing versioned alignment evidence.");
  }
  const recipe = workspace.recipe;
  if (recipe !== null && !validComparisonWorkspaceRecipe(recipe, workspace.left.locator.path, workspace.right.locator.path)) {
    throw new Error("Comparison workspace contains an invalid or mismatched view recipe.");
  }
  if (typeof workspace.title !== "string" || workspace.title.trim().length === 0
    || typeof workspace.exportedAt !== "string" || !Number.isFinite(Date.parse(workspace.exportedAt))) {
    throw new Error("Comparison workspace identity is incomplete.");
  }
  return workspace as ComparisonPublicationWorkspace;
}

export function downloadComparisonPublication(contents: string, filename = "nimbus-comparison-workspace.json"): void {
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
