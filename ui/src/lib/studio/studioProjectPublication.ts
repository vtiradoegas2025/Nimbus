import { datasetSourcePaths, type LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { createStudioProject, parseStudioProject, studioProjectDatasetReference, type StudioProject } from "./studioProjects";
import { packagedAtmosphericExample } from "@/lib/dataset/packagedAtmosphericExamples";

export const STUDIO_PROJECT_PUBLICATION_SCHEMA = "nimbus.studio.project-publication" as const;
export const STUDIO_PROJECT_PUBLICATION_VERSION = 1 as const;
export const STUDIO_PROJECT_PUBLICATION_MAX_BYTES = 750_000 as const;

export type StudioProjectSourceRole =
  | { kind: "primary" }
  | { kind: "comparison" }
  | { kind: "ensemble"; index: number };

export interface StudioProjectSourceRequirement {
  role: "primary" | "comparison" | "ensemble";
  index: number | null;
  kind: LocalDatasetSource["kind"];
  displayName: string;
  referenceCount: number;
  availability: "external_local" | "packaged_immutable";
}

export interface StudioProjectPublication {
  schema: typeof STUDIO_PROJECT_PUBLICATION_SCHEMA;
  schemaVersion: typeof STUDIO_PROJECT_PUBLICATION_VERSION;
  exportedAt: string;
  fieldData: "references_only";
  project: StudioProject;
  sourceRequirements: readonly StudioProjectSourceRequirement[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const expected = new Set(keys);
  const unexpected = Object.keys(value).filter((key) => !expected.has(key));
  const missing = keys.filter((key) => !(key in value));
  if (unexpected.length) throw new Error(`${label} has unsupported properties: ${unexpected.join(", ")}`);
  if (missing.length) throw new Error(`${label} is missing properties: ${missing.join(", ")}`);
}

function requirement(source: LocalDatasetSource, role: StudioProjectSourceRequirement["role"], index: number | null): StudioProjectSourceRequirement {
  return {
    role,
    index,
    kind: source.kind,
    displayName: source.displayName,
    referenceCount: datasetSourcePaths(source).length,
    availability: packagedAtmosphericExample(source.locator.path) ? "packaged_immutable" : "external_local",
  };
}

export function projectSourceRequirements(project: StudioProject): StudioProjectSourceRequirement[] {
  const parsed = parseStudioProject(project);
  return [
    requirement(parsed.primaryDataset, "primary", null),
    ...(parsed.comparisonDataset ? [requirement(parsed.comparisonDataset, "comparison", null)] : []),
    ...parsed.ensembleDatasets.map((source, index) => requirement(source, "ensemble", index)),
  ];
}

export function createStudioProjectPublication(project: StudioProject, exportedAt = new Date().toISOString()): StudioProjectPublication {
  if (!Number.isFinite(Date.parse(exportedAt)) || new Date(exportedAt).toISOString() !== exportedAt) {
    throw new Error("Project export time must be a canonical ISO-8601 timestamp.");
  }
  const parsed = parseStudioProject(project);
  return {
    schema: STUDIO_PROJECT_PUBLICATION_SCHEMA,
    schemaVersion: STUDIO_PROJECT_PUBLICATION_VERSION,
    exportedAt,
    fieldData: "references_only",
    project: parsed,
    sourceRequirements: projectSourceRequirements(parsed),
  };
}

export function serializeStudioProjectPublication(publication: StudioProjectPublication): string {
  const canonical = parseStudioProjectPublication(JSON.stringify(publication));
  const serialized = `${JSON.stringify(canonical, null, 2)}\n`;
  if (new TextEncoder().encode(serialized).byteLength > STUDIO_PROJECT_PUBLICATION_MAX_BYTES) {
    throw new Error("Project bundle exceeds the 750 kB portability limit.");
  }
  return serialized;
}

export function parseStudioProjectPublication(text: string): StudioProjectPublication {
  if (new TextEncoder().encode(text).byteLength > STUDIO_PROJECT_PUBLICATION_MAX_BYTES) {
    throw new Error("Project bundle exceeds the 750 kB import safety limit.");
  }
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("Project bundle is not valid JSON."); }
  if (!isRecord(value)) throw new Error("Project bundle root must be an object.");
  exactKeys(value, ["schema", "schemaVersion", "exportedAt", "fieldData", "project", "sourceRequirements"], "project bundle");
  if (value.schema !== STUDIO_PROJECT_PUBLICATION_SCHEMA || value.schemaVersion !== STUDIO_PROJECT_PUBLICATION_VERSION) {
    throw new Error("Unsupported project bundle schema or version.");
  }
  if (value.fieldData !== "references_only") throw new Error("Project bundle has an unsupported field-data policy.");
  if (typeof value.exportedAt !== "string" || !Number.isFinite(Date.parse(value.exportedAt)) || new Date(value.exportedAt).toISOString() !== value.exportedAt) {
    throw new Error("Project bundle export time is invalid.");
  }
  const project = parseStudioProject(value.project);
  if (!Array.isArray(value.sourceRequirements)) throw new Error("Project bundle source requirements are invalid.");
  const expected = projectSourceRequirements(project);
  if (JSON.stringify(value.sourceRequirements) !== JSON.stringify(expected)) {
    throw new Error("Project bundle source requirements do not match its project references.");
  }
  return {
    schema: STUDIO_PROJECT_PUBLICATION_SCHEMA,
    schemaVersion: STUDIO_PROJECT_PUBLICATION_VERSION,
    exportedAt: value.exportedAt,
    fieldData: "references_only",
    project,
    sourceRequirements: expected,
  };
}

/** Imports as a new local record so opening a bundle can never overwrite an
 * existing project with the same originating id. */
export function materializeStudioProjectPublication(
  publication: StudioProjectPublication,
  now = new Date().toISOString(),
): StudioProject {
  const source = parseStudioProject(publication.project);
  return createStudioProject({
    name: source.name,
    description: source.description,
    primaryDataset: source.primaryDataset,
    comparisonDataset: source.comparisonDataset,
    ensembleDatasets: source.ensembleDatasets,
    analysis: source.analysis,
    workspace: source.workspace,
    expressionDefinitions: source.expressionDefinitions,
  }, { now });
}

export function rebindStudioProjectSource(
  project: StudioProject,
  role: StudioProjectSourceRole,
  replacement: LocalDatasetSource,
  now = new Date().toISOString(),
): StudioProject {
  const parsed = parseStudioProject(project);
  const expected = role.kind === "primary"
    ? parsed.primaryDataset
    : role.kind === "comparison"
      ? parsed.comparisonDataset
      : parsed.ensembleDatasets[role.index];
  if (!expected) throw new Error("The selected project source no longer exists.");
  if (expected.kind !== replacement.kind) {
    throw new Error(`This source requires ${expected.kind === "cm1_netcdf" ? "CM1 / netCDF" : "Nimbus history"} data.`);
  }
  const replacementReference = studioProjectDatasetReference(replacement);
  const rendererState = {...parsed.workspace.rendererState};
  if(role.kind === "primary") delete rendererState.researchRecipe;
  const workspace = {
    ...parsed.workspace,
    rendererState,
    investigation: role.kind === "primary" ? null : parsed.workspace.investigation,
    comparison: role.kind === "primary" || role.kind === "comparison" ? null : parsed.workspace.comparison,
  };
  const candidate: StudioProject = role.kind === "primary"
    ? { ...parsed, primaryDataset: replacementReference, workspace, updatedAt: now }
    : role.kind === "comparison"
      ? { ...parsed, comparisonDataset: replacementReference, workspace, updatedAt: now }
      : {
          ...parsed,
          ensembleDatasets: parsed.ensembleDatasets.map((source, index) => index === role.index ? replacementReference : source),
          workspace,
          updatedAt: now,
        };
  return parseStudioProject(candidate);
}

export function downloadStudioProjectPublication(contents: string, projectName: string): void {
  const stem = projectName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "project";
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `nimbus-${stem}.project.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
