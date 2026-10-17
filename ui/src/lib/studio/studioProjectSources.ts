import { datasetSourcePaths } from "@/lib/dataset/datasetSources";
import { parseStudioProject, type StudioProject } from "./studioProjects";
import type { StudioProjectSourceRole } from "./studioProjectPublication";
import { packagedAtmosphericExample } from "@/lib/dataset/packagedAtmosphericExamples";

export const STUDIO_PROJECT_SOURCE_AUDIT_SCHEMA = "nimbus.studio.project-source-audit" as const;
export const STUDIO_PROJECT_SOURCE_AUDIT_VERSION = 1 as const;

export interface StudioProjectSourceAuditEntry {
  role: StudioProjectSourceRole;
  displayName: string;
  expectedReferences: number;
  reachableReferences: number;
  status: "ready" | "partial" | "missing" | "packaged" | "unverifiable";
  missingPaths: string[];
  probeErrors: string[];
}

export interface StudioProjectSourceAudit {
  schema: typeof STUDIO_PROJECT_SOURCE_AUDIT_SCHEMA;
  schemaVersion: typeof STUDIO_PROJECT_SOURCE_AUDIT_VERSION;
  projectId: string;
  checkedAt: string;
  entries: StudioProjectSourceAuditEntry[];
  reopenReady: boolean;
}

type PathProbe = (path: string) => boolean | Promise<boolean>;

export async function auditStudioProjectSources(
  project: StudioProject,
  probe: PathProbe,
  checkedAt = new Date().toISOString(),
): Promise<StudioProjectSourceAudit> {
  const parsed = parseStudioProject(project);
  if (!Number.isFinite(Date.parse(checkedAt)) || new Date(checkedAt).toISOString() !== checkedAt) {
    throw new Error("Project source audit time must be a canonical ISO-8601 timestamp.");
  }
  const sources = [
    { role: { kind: "primary" } as const, source: parsed.primaryDataset },
    ...(parsed.comparisonDataset ? [{ role: { kind: "comparison" } as const, source: parsed.comparisonDataset }] : []),
    ...parsed.ensembleDatasets.map((source, index) => ({ role: { kind: "ensemble", index } as const, source })),
  ];
  const entries: StudioProjectSourceAuditEntry[] = [];
  for (const { role, source } of sources) {
    const paths = datasetSourcePaths(source);
    if (packagedAtmosphericExample(source.locator.path)) {
      entries.push({
        role,
        displayName: source.displayName,
        expectedReferences: paths.length,
        reachableReferences: paths.length,
        status: "packaged",
        missingPaths: [],
        probeErrors: [],
      });
      continue;
    }
    const missingPaths: string[] = [];
    const probeErrors: string[] = [];
    let reachableReferences = 0;
    for (const path of paths) {
      try {
        if (await probe(path)) reachableReferences += 1;
        else missingPaths.push(path);
      } catch (error) {
        probeErrors.push(`${path}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    const status = probeErrors.length > 0
      ? "unverifiable"
      : reachableReferences === paths.length
        ? "ready"
        : reachableReferences === 0
          ? "missing"
          : "partial";
    entries.push({ role, displayName: source.displayName, expectedReferences: paths.length, reachableReferences, status, missingPaths, probeErrors });
  }
  return {
    schema: STUDIO_PROJECT_SOURCE_AUDIT_SCHEMA,
    schemaVersion: STUDIO_PROJECT_SOURCE_AUDIT_VERSION,
    projectId: parsed.id,
    checkedAt,
    entries,
    reopenReady: entries.every(({ status }) => status === "ready" || status === "packaged"),
  };
}

export async function verifyStudioProjectSources(project: StudioProject): Promise<StudioProjectSourceAudit> {
  if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
    throw new Error("Local project source verification requires Nimbus Studio desktop.");
  }
  const { exists } = await import("@tauri-apps/plugin-fs");
  return auditStudioProjectSources(project, (path) => exists(path));
}

export function summarizeStudioProjectSourceAudit(audit: StudioProjectSourceAudit): string {
  const packaged = audit.entries.filter(({ status }) => status === "packaged").length;
  const ready = audit.entries.filter(({ status }) => status === "ready").length;
  const partial = audit.entries.filter(({ status }) => status === "partial").length;
  const missing = audit.entries.filter(({ status }) => status === "missing").length;
  const unverifiable = audit.entries.filter(({ status }) => status === "unverifiable").length;
  if (audit.reopenReady) {
    return `All ${audit.entries.length} project ${audit.entries.length === 1 ? "source is" : "sources are"} available${packaged ? ` (${packaged} included with Studio)` : ""}.`;
  }
  return [
    `${ready + packaged} of ${audit.entries.length} project sources are ready.`,
    partial ? `${partial} partially reachable.` : "",
    missing ? `${missing} missing.` : "",
    unverifiable ? `${unverifiable} could not be verified.` : "",
    "Relink affected sources before reopening.",
  ].filter(Boolean).join(" ");
}
