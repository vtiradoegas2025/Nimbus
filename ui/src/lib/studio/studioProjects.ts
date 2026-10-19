import {parseResearchRecipe} from "@/lib/research/recipe";
import {
  DATASET_SOURCE_SCHEMA,
  DATASET_SOURCE_VERSION,
  datasetSourceIdentity,
  datasetSourcePaths,
  type DatasetSourceKind,
  type LocalDatasetSource,
} from "@/lib/dataset/datasetSources";
import {
  parseExpressionDerivedField,
  type ExpressionDerivedFieldDefinition,
} from "@/lib/derived/expressionDerivedFields";
import {
  validInvestigationWorkspace,
  type InvestigationWorkspaceRecipe,
} from "./investigationWorkspace";
import {
  validComparisonWorkspaceRecipe,
  type ComparisonWorkspaceRecipe,
} from "@/lib/comparison/comparisonWorkspace";
import type { PlotCategory, PlotType } from "@/lib/plot/plotRegistry";
import { PACKAGED_ATMOSPHERIC_EXAMPLES } from "@/lib/dataset/packagedAtmosphericExamples";

export const STUDIO_PROJECT_SCHEMA = "nimbus.studio.project" as const;
export const STUDIO_PROJECT_VERSION = 1 as const;
export const STUDIO_PROJECT_STORE_SCHEMA = "nimbus.studio.project-store" as const;
export const STUDIO_PROJECT_STORE_VERSION = 1 as const;
export const STUDIO_PROJECT_STORAGE_KEY = "nimbus:studio:projects:v1" as const;
export const STUDIO_PROJECT_STAGING_KEY = "nimbus:studio:projects:v1:staging" as const;
export const STUDIO_PROJECT_BACKUP_KEY = "nimbus:studio:projects:v1:backup" as const;
export const STUDIO_ACTIVE_PROJECT_STORAGE_KEY = "nimbus:studio:active-project:v1" as const;
export const STUDIO_PROJECT_LIMIT = 24 as const;
export const STUDIO_PROJECT_STORE_MAX_BYTES = 750_000 as const;

const MAX_NAME_LENGTH = 96;
const MAX_DESCRIPTION_LENGTH = 480;
const MAX_PATH_LENGTH = 4_096;
const MAX_DISPLAY_NAME_LENGTH = 512;
const MAX_ENSEMBLE_MEMBERS = 32;
const MAX_RENDERER_STATE_BYTES = 64_000;
const MAX_RENDERER_STATE_DEPTH = 6;
const MAX_RENDERER_STATE_NODES = 500;
const PROJECT_ID = /^project_[a-z0-9][a-z0-9-]{7,79}$/;

const PLOT_CATEGORIES: readonly PlotCategory[] = [
  "plan-views", "cross-sections", "soundings", "radar", "volumetric",
  "time-series", "swaths", "trajectories", "scripting",
];
const PLOT_TYPES: readonly PlotType[] = [
  "slice", "vector", "composite", "timeseries", "timeheight", "swath",
  "vertical", "radial", "hodograph", "skewt", "volume", "isosurface",
  "streamlines", "supercell", "trajectories", "trajectory_budget", "python",
];
const PANEL_IDS = [
  "data-health", "recent-datasets", "derived-fields", "resource-budget",
  "investigation", "comparison", "projects", "research",
] as const;

export type StudioPanelId = typeof PANEL_IDS[number];
export type ProjectJsonValue = null | boolean | number | string | ProjectJsonValue[] | { [key: string]: ProjectJsonValue };

export interface StudioProjectAnalysisSnapshot {
  frameIndex: number;
  fieldName: string | null;
  category: PlotCategory | null;
  plotType: PlotType | null;
  presetId: string | null;
}

/** UI state that belongs to the project rather than to a machine. rendererState
 * is a deliberately bounded, JSON-only extension point for plot-specific state. */
export interface StudioProjectWorkspaceSnapshot {
  activePanel: StudioPanelId | null;
  visiblePanels: readonly StudioPanelId[];
  frameRange: readonly [number, number] | null;
  investigation: InvestigationWorkspaceRecipe | null;
  comparison: ComparisonWorkspaceRecipe | null;
  rendererState: Readonly<Record<string, ProjectJsonValue>>;
}

export interface StudioProject {
  schema: typeof STUDIO_PROJECT_SCHEMA;
  schemaVersion: typeof STUDIO_PROJECT_VERSION;
  id: string;
  name: string;
  description: string;
  createdAt: string;
  updatedAt: string;
  lastOpenedAt: string | null;
  primaryDataset: LocalDatasetSource;
  comparisonDataset: LocalDatasetSource | null;
  ensembleDatasets: readonly LocalDatasetSource[];
  analysis: StudioProjectAnalysisSnapshot;
  workspace: StudioProjectWorkspaceSnapshot;
  expressionDefinitions: readonly ExpressionDerivedFieldDefinition[];
}

export interface StudioProjectStore {
  schema: typeof STUDIO_PROJECT_STORE_SCHEMA;
  schemaVersion: typeof STUDIO_PROJECT_STORE_VERSION;
  savedAt: string;
  projects: readonly StudioProject[];
}

export interface StudioProjectLoadResult {
  projects: StudioProject[];
  warnings: string[];
}

export interface StudioProjectDraft {
  name: string;
  description?: string;
  primaryDataset: LocalDatasetSource;
  comparisonDataset?: LocalDatasetSource | null;
  ensembleDatasets?: readonly LocalDatasetSource[];
  analysis?: Partial<StudioProjectAnalysisSnapshot>;
  workspace?: Partial<StudioProjectWorkspaceSnapshot>;
  expressionDefinitions?: readonly ExpressionDerivedFieldDefinition[];
}

export interface StudioProjectCreateOptions {
  id?: string;
  now?: string;
}

export interface BundledProjectExample {
  schema: "nimbus.studio.bundled-project-example";
  schemaVersion: 1;
  id: string;
  title: string;
  summary: string;
  workflow: string;
  sourceModel: "CM1" | "Nimbus";
  acceptedSourceKinds: readonly DatasetSourceKind[];
  openMode: "select_local_dataset" | "open_packaged_dataset";
  availability: "reference_only" | "included";
  availabilityNote: string;
  analysis: StudioProjectAnalysisSnapshot;
}

export const BUNDLED_PROJECT_EXAMPLES: readonly BundledProjectExample[] = Object.freeze([
  Object.freeze({
    schema: "nimbus.studio.bundled-project-example" as const,
    schemaVersion: 1 as const,
    id: "cm1-supercell-review",
    title: "Atmospheric netCDF review",
    summary: "A guided review of compatible netCDF history: reflectivity, vertical velocity, cold-pool structure, soundings, and time evolution.",
    workflow: "Starts in plan view at the first frame, then exposes qualified diagnostics as the selected dataset is inspected.",
    sourceModel: "CM1" as const,
    acceptedSourceKinds: ["cm1_netcdf"] as const,
    openMode: "select_local_dataset" as const,
    availability: "reference_only" as const,
    availabilityNote: "Choose a local case folder or netCDF series; the files remain in place.",
    analysis: Object.freeze({
      frameIndex: 0,
      fieldName: "w",
      category: "plan-views" as const,
      plotType: "composite" as const,
      presetId: null,
    }),
  }),
  Object.freeze({
    schema: "nimbus.studio.bundled-project-example" as const,
    schemaVersion: 1 as const,
    id: "nimbus-storm-evolution",
    title: "Nimbus storm evolution",
    summary: "A compact path through plan views, vertical structure, volumetric rendering, and publication capture for a Nimbus run.",
    workflow: "Opens a local Nimbus result as a named, resumable analysis project.",
    sourceModel: "Nimbus" as const,
    acceptedSourceKinds: ["nimbus_run"] as const,
    openMode: "select_local_dataset" as const,
    availability: "reference_only" as const,
    availabilityNote: "Choose a local Nimbus run directory. The example stores workflow state, never a copy of the run fields.",
    analysis: Object.freeze({
      frameIndex: 0,
      fieldName: "w",
      category: "plan-views" as const,
      plotType: "slice" as const,
      presetId: null,
    }),
  }),
  Object.freeze({
    schema: "nimbus.studio.bundled-project-example" as const,
    schemaVersion: 1 as const,
    id: PACKAGED_ATMOSPHERIC_EXAMPLES[0].id,
    title: PACKAGED_ATMOSPHERIC_EXAMPLES[0].title,
    summary: PACKAGED_ATMOSPHERIC_EXAMPLES[0].summary,
    workflow: "Opens eight immutable manufactured-atmosphere frames directly in Studio, with no local data picker or simulator dependency.",
    sourceModel: "Nimbus" as const,
    acceptedSourceKinds: ["nimbus_run"] as const,
    openMode: "open_packaged_dataset" as const,
    availability: "included" as const,
    availabilityNote: PACKAGED_ATMOSPHERIC_EXAMPLES[0].provenance.scientificUseNotice,
    analysis: Object.freeze({
      frameIndex: 0,
      fieldName: PACKAGED_ATMOSPHERIC_EXAMPLES[0].recommendedView.fieldName,
      category: PACKAGED_ATMOSPHERIC_EXAMPLES[0].recommendedView.category,
      plotType: PACKAGED_ATMOSPHERIC_EXAMPLES[0].recommendedView.plotType,
      presetId: null,
    }),
  }),
]);

type ReadStorage = Pick<Storage, "getItem">;
type ProjectStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const expected = new Set(keys);
  const unexpected = Object.keys(value).filter((key) => !expected.has(key));
  if (unexpected.length > 0) throw new Error(`${label} has unsupported properties: ${unexpected.join(", ")}`);
  const missing = keys.filter((key) => !(key in value));
  if (missing.length > 0) throw new Error(`${label} is missing properties: ${missing.join(", ")}`);
}

function canonicalTimestamp(value: unknown, label: string, nullable = false): string | null {
  if (nullable && value === null) return null;
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString() !== value) {
    throw new Error(`${label} must be a canonical ISO-8601 timestamp${nullable ? " or null" : ""}`);
  }
  return value;
}

function boundedText(value: unknown, label: string, maximum: number, allowEmpty = false): string {
  if (typeof value !== "string" || value.includes("\0") || value !== value.trim()) throw new Error(`${label} must be trimmed text`);
  if ((!allowEmpty && value.length === 0) || value.length > maximum) throw new Error(`${label} must contain ${allowEmpty ? "0" : "1"}-${maximum} characters`);
  return value;
}

function parseProjectId(value: unknown): string {
  if (typeof value !== "string" || !PROJECT_ID.test(value)) throw new Error("project id has an unsupported format");
  return value;
}

function parseDatasetSource(value: unknown, label: string): LocalDatasetSource {
  if (!isRecord(value) || !isRecord(value.locator)) throw new Error(`${label} is not a dataset reference`);
  exactKeys(value, ["schema", "schemaVersion", "kind", "sourceModel", "nativeFormat", "locator", "displayName", "readOnly"], label);
  const locator = value.locator;
  const displayName = boundedText(value.displayName, `${label} display name`, MAX_DISPLAY_NAME_LENGTH);
  const path = boundedText(locator.path, `${label} path`, MAX_PATH_LENGTH);
  let paths: string[] | undefined;
  if (locator.kind === "local_path") {
    exactKeys(locator, ["kind", "path"], `${label} locator`);
  } else if (locator.kind === "local_paths") {
    exactKeys(locator, ["kind", "path", "paths"], `${label} locator`);
    if (!Array.isArray(locator.paths) || locator.paths.length < 2 || locator.paths.length > 1_000) throw new Error(`${label} path series is invalid`);
    paths = locator.paths.map((entry, index) => boundedText(entry, `${label} path ${index + 1}`, MAX_PATH_LENGTH));
    if (paths[0] !== path || new Set(paths).size !== paths.length) throw new Error(`${label} path series is inconsistent`);
  } else {
    throw new Error(`${label} locator kind is unsupported`);
  }
  const kindContract = value.kind === "nimbus_run"
    ? value.sourceModel === "nimbus" && value.nativeFormat === "nimbus_history" && locator.kind === "local_path"
    : value.kind === "cm1_netcdf" && value.sourceModel === "cm1" && value.nativeFormat === "netcdf";
  if (value.schema !== DATASET_SOURCE_SCHEMA || value.schemaVersion !== DATASET_SOURCE_VERSION || value.readOnly !== true || !kindContract) {
    throw new Error(`${label} uses an unsupported dataset contract`);
  }
  return {
    schema: DATASET_SOURCE_SCHEMA,
    schemaVersion: DATASET_SOURCE_VERSION,
    kind: value.kind as DatasetSourceKind,
    sourceModel: value.sourceModel as LocalDatasetSource["sourceModel"],
    nativeFormat: value.nativeFormat as LocalDatasetSource["nativeFormat"],
    locator: paths ? { kind: "local_paths", path, paths } : { kind: "local_path", path },
    displayName,
    readOnly: true,
  };
}

function parseOptionalDataset(value: unknown, label: string): LocalDatasetSource | null {
  return value === null ? null : parseDatasetSource(value, label);
}

export function studioProjectDatasetReference(source: LocalDatasetSource): LocalDatasetSource {
  const paths = datasetSourcePaths(source);
  return parseDatasetSource({
    schema: source.schema,
    schemaVersion: source.schemaVersion,
    kind: source.kind,
    sourceModel: source.sourceModel,
    nativeFormat: source.nativeFormat,
    locator: paths.length > 1
      ? { kind: "local_paths", path: paths[0], paths: [...paths] }
      : { kind: "local_path", path: paths[0] },
    displayName: source.displayName,
    readOnly: true,
  }, "dataset reference");
}

function parseAnalysis(value: unknown): StudioProjectAnalysisSnapshot {
  if (!isRecord(value)) throw new Error("project analysis is not an object");
  exactKeys(value, ["frameIndex", "fieldName", "category", "plotType", "presetId"], "project analysis");
  if (!Number.isInteger(value.frameIndex) || (value.frameIndex as number) < 0 || (value.frameIndex as number) > 10_000_000) {
    throw new Error("project frame index is invalid");
  }
  const nullableText = (candidate: unknown, label: string): string | null => candidate === null
    ? null
    : boundedText(candidate, label, 128);
  const fieldName = nullableText(value.fieldName, "project field name");
  const presetId = nullableText(value.presetId, "project preset id");
  const category = value.category === null ? null : value.category as PlotCategory;
  const plotType = value.plotType === null ? null : value.plotType as PlotType;
  if (category !== null && !PLOT_CATEGORIES.includes(category)) throw new Error("project plot category is unsupported");
  if (plotType !== null && !PLOT_TYPES.includes(plotType)) throw new Error("project plot type is unsupported");
  return { frameIndex: value.frameIndex as number, fieldName, category, plotType, presetId };
}

function cloneJsonValue(value: unknown, state: { nodes: number }, depth = 0): ProjectJsonValue {
  state.nodes += 1;
  if (state.nodes > MAX_RENDERER_STATE_NODES || depth > MAX_RENDERER_STATE_DEPTH) throw new Error("project renderer state exceeds its structural limit");
  if (value === null || typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("project renderer state numbers must be finite");
    return value;
  }
  if (Array.isArray(value)) return value.map((entry) => cloneJsonValue(entry, state, depth + 1));
  if (isRecord(value)) {
    const result: Record<string, ProjectJsonValue> = {};
    for (const [key, entry] of Object.entries(value)) {
      boundedText(key, "renderer state key", 96);
      if (key === "__proto__" || key === "constructor" || key === "prototype") throw new Error("project renderer state contains a reserved key");
      result[key] = cloneJsonValue(entry, state, depth + 1);
    }
    return result;
  }
  throw new Error("project renderer state must contain JSON values only");
}

function parseRendererState(value: unknown): Record<string, ProjectJsonValue> {
  if (!isRecord(value)) throw new Error("project renderer state is not an object");
  const cloned = cloneJsonValue(value, { nodes: 0 });
  if (!isRecord(cloned)) throw new Error("project renderer state is not an object");
  if (new TextEncoder().encode(JSON.stringify(cloned)).byteLength > MAX_RENDERER_STATE_BYTES) throw new Error("project renderer state exceeds 64 kB");
  return cloned as Record<string, ProjectJsonValue>;
}

function parseWorkspace(
  value: unknown,
  primary: LocalDatasetSource,
  comparison: LocalDatasetSource | null,
): StudioProjectWorkspaceSnapshot {
  if (!isRecord(value)) throw new Error("project workspace is not an object");
  exactKeys(value, ["activePanel", "visiblePanels", "frameRange", "investigation", "comparison", "rendererState"], "project workspace");
  const visiblePanels = Array.isArray(value.visiblePanels) ? value.visiblePanels as StudioPanelId[] : [];
  if (!Array.isArray(value.visiblePanels) || visiblePanels.length > PANEL_IDS.length
    || visiblePanels.some((panel) => !PANEL_IDS.includes(panel)) || new Set(visiblePanels).size !== visiblePanels.length) {
    throw new Error("project visible panels are invalid");
  }
  const activePanel = value.activePanel === null ? null : value.activePanel as StudioPanelId;
  if (activePanel !== null && !PANEL_IDS.includes(activePanel)) throw new Error("project active panel is invalid");
  let frameRange: [number, number] | null = null;
  if (value.frameRange !== null) {
    if (!Array.isArray(value.frameRange) || value.frameRange.length !== 2
      || !value.frameRange.every((frame) => Number.isInteger(frame) && frame >= 0)
      || value.frameRange[0] > value.frameRange[1]) throw new Error("project frame range is invalid");
    frameRange = [value.frameRange[0], value.frameRange[1]];
  }
  const primaryIdentity = datasetSourceIdentity(primary);
  const investigation = value.investigation === null ? null : value.investigation;
  if (investigation !== null && !validInvestigationWorkspace(investigation, primaryIdentity)) throw new Error("project investigation workspace does not match its primary dataset");
  const comparisonRecipe = value.comparison === null ? null : value.comparison;
  if (comparisonRecipe !== null) {
    if (!comparison || !validComparisonWorkspaceRecipe(comparisonRecipe, primaryIdentity, datasetSourceIdentity(comparison))) {
      throw new Error("project comparison workspace does not match its dataset references");
    }
  }
  const rendererState=parseRendererState(value.rendererState);
  if(rendererState.researchRecipe!==undefined){if(typeof rendererState.researchRecipe!=="string")throw new Error("Project research recipe must be serialized JSON");parseResearchRecipe(rendererState.researchRecipe,primaryIdentity);}
  return {
    activePanel,
    visiblePanels: [...visiblePanels],
    frameRange,
    investigation: investigation as InvestigationWorkspaceRecipe | null,
    comparison: comparisonRecipe as ComparisonWorkspaceRecipe | null,
    rendererState,
  };
}

function parseExpressions(value: unknown): ExpressionDerivedFieldDefinition[] {
  if (!Array.isArray(value) || value.length > 24) throw new Error("project expression definition set is invalid");
  const identifiers = new Set<string>();
  return value.map((entry) => {
    const definition = parseExpressionDerivedField(entry, { existingDefinitionIds: identifiers });
    identifiers.add(definition.id);
    return definition;
  });
}

export function parseStudioProject(value: unknown): StudioProject {
  if (!isRecord(value)) throw new Error("project is not an object");
  exactKeys(value, [
    "schema", "schemaVersion", "id", "name", "description", "createdAt", "updatedAt",
    "lastOpenedAt", "primaryDataset", "comparisonDataset", "ensembleDatasets", "analysis",
    "workspace", "expressionDefinitions",
  ], "project");
  if (value.schema !== STUDIO_PROJECT_SCHEMA || value.schemaVersion !== STUDIO_PROJECT_VERSION) throw new Error("unsupported project schema or version");
  const primaryDataset = parseDatasetSource(value.primaryDataset, "primary dataset");
  const comparisonDataset = parseOptionalDataset(value.comparisonDataset, "comparison dataset");
  if (!Array.isArray(value.ensembleDatasets) || value.ensembleDatasets.length > MAX_ENSEMBLE_MEMBERS) throw new Error("project ensemble reference set is invalid");
  const ensembleDatasets = value.ensembleDatasets.map((entry, index) => parseDatasetSource(entry, `ensemble dataset ${index + 1}`));
  const identities = ensembleDatasets.map(datasetSourceIdentity);
  if (new Set(identities).size !== identities.length) throw new Error("project ensemble references contain duplicates");
  const createdAt = canonicalTimestamp(value.createdAt, "project created time") as string;
  const updatedAt = canonicalTimestamp(value.updatedAt, "project updated time") as string;
  if (Date.parse(updatedAt) < Date.parse(createdAt)) throw new Error("project update time precedes creation");
  return {
    schema: STUDIO_PROJECT_SCHEMA,
    schemaVersion: STUDIO_PROJECT_VERSION,
    id: parseProjectId(value.id),
    name: boundedText(value.name, "project name", MAX_NAME_LENGTH),
    description: boundedText(value.description, "project description", MAX_DESCRIPTION_LENGTH, true),
    createdAt,
    updatedAt,
    lastOpenedAt: canonicalTimestamp(value.lastOpenedAt, "project last-opened time", true),
    primaryDataset,
    comparisonDataset,
    ensembleDatasets,
    analysis: parseAnalysis(value.analysis),
    workspace: parseWorkspace(value.workspace, primaryDataset, comparisonDataset),
    expressionDefinitions: parseExpressions(value.expressionDefinitions),
  };
}

function generateProjectId(): string {
  const uuid = globalThis.crypto?.randomUUID?.().toLowerCase().replace(/[^a-z0-9-]/g, "");
  if (uuid && uuid.length >= 8) return `project_${uuid}`;
  const entropy = Math.random().toString(36).slice(2, 12);
  return `project_${Date.now().toString(36)}-${entropy}`;
}

function defaultAnalysis(snapshot: Partial<StudioProjectAnalysisSnapshot> | undefined): StudioProjectAnalysisSnapshot {
  return {
    frameIndex: snapshot?.frameIndex ?? 0,
    fieldName: snapshot?.fieldName ?? null,
    category: snapshot?.category ?? null,
    plotType: snapshot?.plotType ?? null,
    presetId: snapshot?.presetId ?? null,
  };
}

function defaultWorkspace(snapshot: Partial<StudioProjectWorkspaceSnapshot> | undefined): StudioProjectWorkspaceSnapshot {
  return {
    activePanel: snapshot?.activePanel ?? null,
    visiblePanels: snapshot?.visiblePanels ?? [],
    frameRange: snapshot?.frameRange ?? null,
    investigation: snapshot?.investigation ?? null,
    comparison: snapshot?.comparison ?? null,
    rendererState: snapshot?.rendererState ?? {},
  };
}

export function createStudioProject(draft: StudioProjectDraft, options: StudioProjectCreateOptions = {}): StudioProject {
  const now = options.now ?? new Date().toISOString();
  const candidate = {
    schema: STUDIO_PROJECT_SCHEMA,
    schemaVersion: STUDIO_PROJECT_VERSION,
    id: options.id ?? generateProjectId(),
    name: draft.name,
    description: draft.description ?? "",
    createdAt: now,
    updatedAt: now,
    lastOpenedAt: null,
    primaryDataset: studioProjectDatasetReference(draft.primaryDataset),
    comparisonDataset: draft.comparisonDataset ? studioProjectDatasetReference(draft.comparisonDataset) : null,
    ensembleDatasets: (draft.ensembleDatasets ?? []).map(studioProjectDatasetReference),
    analysis: defaultAnalysis(draft.analysis),
    workspace: defaultWorkspace(draft.workspace),
    expressionDefinitions: draft.expressionDefinitions ?? [],
  };
  return parseStudioProject(candidate);
}

function replaceProject(projects: readonly StudioProject[], updated: StudioProject): StudioProject[] {
  const parsed = parseStudioProject(updated);
  return projects.map((project) => project.id === parsed.id ? parsed : project);
}

export function renameStudioProject(projects: readonly StudioProject[], id: string, name: string, now = new Date().toISOString()): StudioProject[] {
  const existing = projects.find((project) => project.id === id);
  if (!existing) throw new Error("Project to rename was not found.");
  return replaceProject(projects, { ...existing, name, updatedAt: now });
}

export function updateStudioProjectSnapshot(
  projects: readonly StudioProject[],
  id: string,
  snapshot: { analysis?: StudioProjectAnalysisSnapshot; workspace?: StudioProjectWorkspaceSnapshot; expressionDefinitions?: readonly ExpressionDerivedFieldDefinition[] },
  now = new Date().toISOString(),
): StudioProject[] {
  const existing = projects.find((project) => project.id === id);
  if (!existing) throw new Error("Project to update was not found.");
  const candidate = parseStudioProject({
    ...existing,
    analysis: snapshot.analysis ?? existing.analysis,
    workspace: snapshot.workspace ?? existing.workspace,
    expressionDefinitions: snapshot.expressionDefinitions ?? existing.expressionDefinitions,
    updatedAt: now,
  });
  if (JSON.stringify(candidate.analysis) === JSON.stringify(existing.analysis)
      && JSON.stringify(candidate.workspace) === JSON.stringify(existing.workspace)
      && JSON.stringify(candidate.expressionDefinitions) === JSON.stringify(existing.expressionDefinitions)) {
    return [...projects];
  }
  return replaceProject(projects, candidate);
}

export function markStudioProjectOpened(projects: readonly StudioProject[], id: string, now = new Date().toISOString()): StudioProject[] {
  const existing = projects.find((project) => project.id === id);
  if (!existing) throw new Error("Project to open was not found.");
  return replaceProject(projects, { ...existing, lastOpenedAt: now, updatedAt: now });
}

export function duplicateStudioProject(
  projects: readonly StudioProject[],
  id: string,
  options: StudioProjectCreateOptions & { name?: string } = {},
): StudioProject[] {
  const existing = projects.find((project) => project.id === id);
  if (!existing) throw new Error("Project to duplicate was not found.");
  const duplicate = createStudioProject({
    name: options.name ?? `${existing.name} copy`,
    description: existing.description,
    primaryDataset: existing.primaryDataset,
    comparisonDataset: existing.comparisonDataset,
    ensembleDatasets: existing.ensembleDatasets,
    analysis: existing.analysis,
    workspace: existing.workspace,
    expressionDefinitions: existing.expressionDefinitions,
  }, options);
  return upsertStudioProject(projects, duplicate);
}

export function deleteStudioProject(projects: readonly StudioProject[], id: string): StudioProject[] {
  return projects.filter((project) => project.id !== id);
}

export function upsertStudioProject(projects: readonly StudioProject[], project: StudioProject): StudioProject[] {
  const parsed = parseStudioProject(project);
  const next = [parsed, ...projects.filter((entry) => entry.id !== parsed.id)].map(parseStudioProject);
  if (next.length > STUDIO_PROJECT_LIMIT) throw new Error(`Studio supports at most ${STUDIO_PROJECT_LIMIT} named projects.`);
  return next.sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt) || left.name.localeCompare(right.name));
}

export function instantiateBundledProjectExample(
  example: BundledProjectExample,
  primaryDataset: LocalDatasetSource,
  options: StudioProjectCreateOptions = {},
): StudioProject {
  if (!BUNDLED_PROJECT_EXAMPLES.some((candidate) => candidate.id === example.id)) throw new Error("Unknown bundled project example.");
  if (!example.acceptedSourceKinds.includes(primaryDataset.kind)) {
    throw new Error(`${example.title} requires ${example.sourceModel} data; '${primaryDataset.displayName}' uses ${primaryDataset.sourceModel}.`);
  }
  return createStudioProject({
    name: example.title,
    description: example.summary,
    primaryDataset,
    analysis: example.analysis,
  }, options);
}

export function parseStudioProjectStore(value: unknown): StudioProjectLoadResult {
  if (!isRecord(value)
    || value.schema !== STUDIO_PROJECT_STORE_SCHEMA
    || value.schemaVersion !== STUDIO_PROJECT_STORE_VERSION
    || !Array.isArray(value.projects)
    || value.projects.length > STUDIO_PROJECT_LIMIT) {
    return { projects: [], warnings: ["Saved projects use an unsupported or invalid store contract and were not opened."] };
  }
  try {
    canonicalTimestamp(value.savedAt, "project store save time");
  } catch (error) {
    return { projects: [], warnings: [error instanceof Error ? error.message : "Project store timestamp is invalid."] };
  }
  const projects: StudioProject[] = [];
  const warnings: string[] = [];
  for (let index = 0; index < value.projects.length; index += 1) {
    try {
      projects.push(parseStudioProject(value.projects[index]));
    } catch (error) {
      warnings.push(`Saved project ${index + 1} was quarantined: ${error instanceof Error ? error.message : "invalid project"}`);
    }
  }
  const identities = new Set<string>();
  const unique = projects.filter((project) => {
    if (identities.has(project.id)) {
      warnings.push(`Duplicate saved project '${project.id}' was quarantined.`);
      return false;
    }
    identities.add(project.id);
    return true;
  });
  return { projects: unique.sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt)), warnings };
}

interface StoredProjectCandidate {
  key: string;
  savedAt: string;
  result: StudioProjectLoadResult;
}

function readStoredProjectCandidate(storage: ReadStorage, key: string): StoredProjectCandidate | null {
  const serialized = storage.getItem(key);
  if (!serialized) return null;
  if (new TextEncoder().encode(serialized).byteLength > STUDIO_PROJECT_STORE_MAX_BYTES) return null;
  try {
    const value: unknown = JSON.parse(serialized);
    if (!isRecord(value)
      || value.schema !== STUDIO_PROJECT_STORE_SCHEMA
      || value.schemaVersion !== STUDIO_PROJECT_STORE_VERSION
      || typeof value.savedAt !== "string"
      || !Number.isFinite(Date.parse(value.savedAt))
      || new Date(value.savedAt).toISOString() !== value.savedAt
      || !Array.isArray(value.projects)
      || value.projects.length > STUDIO_PROJECT_LIMIT) return null;
    const result = parseStudioProjectStore(value);
    if (value.projects.length > 0 && result.projects.length === 0) return null;
    return { key, savedAt: value.savedAt, result };
  } catch {
    return null;
  }
}

export function loadStudioProjects(storage: ReadStorage): StudioProjectLoadResult {
  const presentKeys = [STUDIO_PROJECT_STORAGE_KEY, STUDIO_PROJECT_STAGING_KEY, STUDIO_PROJECT_BACKUP_KEY]
    .filter((key) => storage.getItem(key) != null);
  if (presentKeys.length === 0) return { projects: [], warnings: [] };
  const candidates = presentKeys
    .map((key) => readStoredProjectCandidate(storage, key))
    .filter((candidate): candidate is StoredProjectCandidate => candidate != null)
    .sort((left, right) => Date.parse(right.savedAt) - Date.parse(left.savedAt)
      || (left.key === STUDIO_PROJECT_STORAGE_KEY ? -1 : right.key === STUDIO_PROJECT_STORAGE_KEY ? 1 : 0));
  const selected = candidates[0];
  if (!selected) return { projects: [], warnings: ["Saved projects could not be read from the primary store or recovery journal."] };
  const recoveryWarning = selected.key === STUDIO_PROJECT_STORAGE_KEY
    ? []
    : [selected.key === STUDIO_PROJECT_STAGING_KEY
        ? "Recovered named projects from an interrupted-write journal."
        : "Recovered named projects from the last valid backup."];
  return { projects: selected.result.projects, warnings: [...recoveryWarning, ...selected.result.warnings] };
}

export function saveStudioProjects(storage: ProjectStorage, projects: readonly StudioProject[], now = new Date().toISOString()): void {
  if (projects.length > STUDIO_PROJECT_LIMIT) throw new Error(`Studio supports at most ${STUDIO_PROJECT_LIMIT} named projects.`);
  const canonicalProjects = projects.map(parseStudioProject);
  if (new Set(canonicalProjects.map((project) => project.id)).size !== canonicalProjects.length) throw new Error("Named projects must have unique ids.");
  const store: StudioProjectStore = {
    schema: STUDIO_PROJECT_STORE_SCHEMA,
    schemaVersion: STUDIO_PROJECT_STORE_VERSION,
    savedAt: canonicalTimestamp(now, "project store save time") as string,
    projects: canonicalProjects,
  };
  const serialized = JSON.stringify(store);
  if (new TextEncoder().encode(serialized).byteLength > STUDIO_PROJECT_STORE_MAX_BYTES) {
    throw new Error("Named projects exceed the 750 kB local storage safety limit. Remove renderer state or an older project before saving.");
  }
  // Write-ahead log pattern: staging → backup old primary → write primary → remove staging.
  storage.setItem(STUDIO_PROJECT_STAGING_KEY, serialized);
  const previous = storage.getItem(STUDIO_PROJECT_STORAGE_KEY);
  if (previous && readStoredProjectCandidate(storage, STUDIO_PROJECT_STORAGE_KEY)) {
    storage.setItem(STUDIO_PROJECT_BACKUP_KEY, previous);
  }
  storage.setItem(STUDIO_PROJECT_STORAGE_KEY, serialized);
  storage.removeItem(STUDIO_PROJECT_STAGING_KEY);
}

/** The active-project pointer is intentionally separate from the recoverable
 * project store: it is disposable navigation state, not scientific content. */
export function loadActiveStudioProjectId(
  storage: ReadStorage,
  projects: readonly StudioProject[],
): string | null {
  try {
    const id = storage.getItem(STUDIO_ACTIVE_PROJECT_STORAGE_KEY);
    return id && projects.some((project) => project.id === id) ? id : null;
  } catch {
    return null;
  }
}

export function saveActiveStudioProjectId(
  storage: Pick<Storage, "setItem" | "removeItem">,
  id: string | null,
): void {
  try {
    if (id) storage.setItem(STUDIO_ACTIVE_PROJECT_STORAGE_KEY, id);
    else storage.removeItem(STUDIO_ACTIVE_PROJECT_STORAGE_KEY);
  } catch {
    // A blocked preference store must not prevent project navigation in memory.
  }
}

export function projectDatasetSummary(project: StudioProject): string {
  const comparison = project.comparisonDataset ? 1 : 0;
  const ensemble = project.ensembleDatasets.length;
  const additions = [comparison ? "comparison" : "", ensemble ? `${ensemble}-member ensemble` : ""].filter(Boolean);
  return `${project.primaryDataset.sourceModel.toUpperCase()} · ${datasetSourcePaths(project.primaryDataset).length} local reference${datasetSourcePaths(project.primaryDataset).length === 1 ? "" : "s"}${additions.length ? ` · ${additions.join(" · ")}` : ""}`;
}
