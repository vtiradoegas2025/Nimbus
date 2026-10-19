import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";
import { createExpressionDerivedField } from "../../../../ui/src/lib/derived/expressionDerivedFields.ts";
import {
  BUNDLED_PROJECT_EXAMPLES,
  STUDIO_ACTIVE_PROJECT_STORAGE_KEY,
  STUDIO_PROJECT_BACKUP_KEY,
  STUDIO_PROJECT_LIMIT,
  STUDIO_PROJECT_STAGING_KEY,
  STUDIO_PROJECT_STORAGE_KEY,
  createStudioProject,
  deleteStudioProject,
  duplicateStudioProject,
  instantiateBundledProjectExample,
  loadActiveStudioProjectId,
  loadStudioProjects,
  markStudioProjectOpened,
  parseStudioProject,
  parseStudioProjectStore,
  renameStudioProject,
  saveActiveStudioProjectId,
  saveStudioProjects,
  updateStudioProjectSnapshot,
  upsertStudioProject,
} from "../../../../ui/src/lib/studio/studioProjects.ts";
import {
  createStudioProjectPublication,
  materializeStudioProjectPublication,
  parseStudioProjectPublication,
  rebindStudioProjectSource,
  serializeStudioProjectPublication,
} from "../../../../ui/src/lib/studio/studioProjectPublication.ts";
import { auditStudioProjectSources, summarizeStudioProjectSourceAudit } from "../../../../ui/src/lib/studio/studioProjectSources.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);
const at = (minute: number) => `2026-07-19T14:${String(minute).padStart(2, "0")}:00.000Z`;

class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

class FailingPrimaryStorage extends MemoryStorage {
  failPrimary = false;
  override setItem(key: string, value: string) {
    if (this.failPrimary && key === STUDIO_PROJECT_STORAGE_KEY) throw new Error("injected primary write failure");
    super.setItem(key, value);
  }
}

const cm1Base = createLocalDatasetSource("cm1_netcdf", ["/cases/right-mover/cm1out_000001.nc", "/cases/right-mover/cm1out_000002.nc"]);
const cm1 = {
  ...cm1Base,
  caseInventory: {
    schema: "nimbus.studio.cm1-case-inventory" as const,
    schemaVersion: 2 as const,
    directory: "/cases/right-mover",
    historyFiles: [...cm1Base.locator.paths!],
    namelistInput: "/cases/right-mover/namelist.input",
    inputSounding: "/cases/right-mover/input_sounding",
    restartFiles: [],
    statisticsFiles: [],
    otherFileCount: 0,
    nestedDirectoryCount: 0,
  },
};
const nimbus = createLocalDatasetSource("nimbus_run", "/runs/reference-storm");
const expression = createExpressionDerivedField({
  id: "scaled_w",
  label: "Scaled vertical velocity",
  unit: "m s-1",
  dependencies: ["w"],
  parameters: [{ id: "factor", value: 1.5 }],
  formula: "w * factor",
});

const project = createStudioProject({
  name: "Right-mover review",
  description: "CM1 benchmark analysis",
  primaryDataset: cm1,
  comparisonDataset: nimbus,
  ensembleDatasets: [cm1Base, nimbus],
  analysis: {
    frameIndex: 7,
    fieldName: "scaled_w",
    category: "cross-sections",
    plotType: "composite",
    presetId: "updraft-section",
  },
  workspace: {
    activePanel: "data-health",
    visiblePanels: ["data-health", "resource-budget"],
    frameRange: [3, 11],
    rendererState: {
      composite: { sliceAxis: 1, sliceIndex: 32, overlays: ["theta", "dbz"] },
    },
  },
  expressionDefinitions: [expression],
}, { id: "project_alpha-0001", now: at(1) });

check(project.primaryDataset.caseInventory == null, "projects sanitize runtime inspection payloads into reference-only datasets");
check(project.analysis.frameIndex === 7 && project.analysis.plotType === "composite", "analysis selection survives project creation");
check(project.workspace.rendererState.composite != null, "bounded renderer-specific workspace state is retained");
check(project.expressionDefinitions[0]?.id === "scaled_w", "expression definitions are included in the project contract");
check(project.comparisonDataset?.kind === "nimbus_run" && project.ensembleDatasets.length === 2, "comparison and ensemble references are retained");

const parsed = parseStudioProject(JSON.parse(JSON.stringify(project)));
check(parsed.id === project.id && parsed.workspace.frameRange?.[1] === 11, "versioned project records round-trip through strict parsing");

const unchanged = updateStudioProjectSnapshot([project], project.id, {
  analysis: structuredClone(project.analysis),
  workspace: structuredClone(project.workspace),
  expressionDefinitions: structuredClone(project.expressionDefinitions),
}, at(2));
check(unchanged[0] === project && unchanged[0].updatedAt === at(1), "an unchanged snapshot preserves activity time and identity so autosave can skip journaling");
const changedCamera = updateStudioProjectSnapshot([project], project.id, {
  workspace: { ...project.workspace, rendererState: { ...project.workspace.rendererState, camera: { position: [1, 2, 3] } } },
}, at(2));
check(changedCamera[0] !== project && changedCamera[0].updatedAt === at(2), "a changed renderer state remains a durable project edit");
let unchangedInvalidRejected = false;
try { updateStudioProjectSnapshot([project], project.id, { analysis: { ...project.analysis, frameIndex: Number.NaN } }, at(2)); } catch { unchangedInvalidRejected = true; }
check(unchangedInvalidRejected, "snapshot comparison cannot turn invalid numerical state into an ignored edit");

let unsupportedPropertyRejected = false;
try { parseStudioProject({ ...project, futureProperty: true }); } catch { unsupportedPropertyRejected = true; }
check(unsupportedPropertyRejected, "unknown project properties fail closed");

let nonfiniteStateRejected = false;
try {
  createStudioProject({ ...project, workspace: { ...project.workspace, rendererState: { contour: { value: Number.NaN } } } }, { id: "project_alpha-0002", now: at(2) });
} catch { nonfiniteStateRejected = true; }
check(nonfiniteStateRejected, "renderer extension state rejects non-finite JSON values");

let projects = upsertStudioProject([], project);
projects = renameStudioProject(projects, project.id, "Renamed review", at(2));
check(projects[0].name === "Renamed review" && projects[0].updatedAt === at(2), "named projects can be renamed immutably");
projects = duplicateStudioProject(projects, project.id, { id: "project_alpha-0003", now: at(3) });
check(projects.length === 2 && projects.some((entry) => entry.name === "Renamed review copy"), "projects can be duplicated with independent identity");
projects = markStudioProjectOpened(projects, project.id, at(4));
check(projects.find((entry) => entry.id === project.id)?.lastOpenedAt === at(4), "opening records resumable project activity");
projects = updateStudioProjectSnapshot(projects, project.id, {
  analysis: { ...project.analysis, frameIndex: 12, fieldName: "w" },
  workspace: { ...project.workspace, activePanel: "investigation" },
}, at(5));
check(projects.find((entry) => entry.id === project.id)?.analysis.frameIndex === 12, "analysis and workspace snapshots can be refreshed");
projects = deleteStudioProject(projects, "project_alpha-0003");
check(projects.length === 1 && projects[0].id === project.id, "delete removes only the selected named project");

const storage = new MemoryStorage();
saveStudioProjects(storage, projects, at(6));
const restored = loadStudioProjects(storage);
check(restored.projects.length === 1 && restored.warnings.length === 0, "valid named projects persist and restore without warnings");
saveActiveStudioProjectId(storage, project.id);
check(storage.getItem(STUDIO_ACTIVE_PROJECT_STORAGE_KEY) === project.id, "active project navigation is stored separately from scientific records");
check(loadActiveStudioProjectId(storage, restored.projects) === project.id, "active project resumes only when its durable record exists");
check(loadActiveStudioProjectId(storage, []) === null, "orphaned active project identity fails closed");
saveActiveStudioProjectId(storage, null);
check(storage.getItem(STUDIO_ACTIVE_PROJECT_STORAGE_KEY) === null, "clearing project context removes the disposable pointer");
check(!storage.values.has(STUDIO_PROJECT_STAGING_KEY), "successful project saves commit and clear the recovery journal");
check(!storage.values.get(STUDIO_PROJECT_STORAGE_KEY)?.includes("namelist.input"), "stored projects do not copy case inspection or field data");

const firstCommitted = storage.values.get(STUDIO_PROJECT_STORAGE_KEY)!;
saveStudioProjects(storage, renameStudioProject(projects, project.id, "Journal revision", at(7)), at(7));
check(storage.values.get(STUDIO_PROJECT_BACKUP_KEY) === firstCommitted, "a new project commit retains the previous valid primary as backup");
storage.values.set(STUDIO_PROJECT_STORAGE_KEY, "torn primary");
const backupRecovered = loadStudioProjects(storage);
check(backupRecovered.projects[0]?.name === "Renamed review" && backupRecovered.warnings.some((warning) => warning.includes("backup")), "a corrupt primary recovers the last valid backup explicitly");

const interruptedStorage = new MemoryStorage();
interruptedStorage.values.set(STUDIO_PROJECT_STORAGE_KEY, firstCommitted);
const stagedStore = JSON.parse(firstCommitted);
stagedStore.savedAt = at(8);
stagedStore.projects[0].name = "Interrupted revision";
stagedStore.projects[0].updatedAt = at(8);
interruptedStorage.values.set(STUDIO_PROJECT_STAGING_KEY, JSON.stringify(stagedStore));
const stagedRecovered = loadStudioProjects(interruptedStorage);
check(stagedRecovered.projects[0]?.name === "Interrupted revision" && stagedRecovered.warnings.some((warning) => warning.includes("interrupted-write")), "a valid newer staging journal completes recovery after interruption");

const failingStorage = new FailingPrimaryStorage();
saveStudioProjects(failingStorage, projects, at(6));
failingStorage.failPrimary = true;
let injectedWriteFailed = false;
try { saveStudioProjects(failingStorage, renameStudioProject(projects, project.id, "Recoverable revision", at(9)), at(9)); } catch { injectedWriteFailed = true; }
const failedWriteRecovered = loadStudioProjects(failingStorage);
check(injectedWriteFailed && failedWriteRecovered.projects[0]?.name === "Recoverable revision" && failedWriteRecovered.warnings.some((warning) => warning.includes("interrupted-write")), "a failed primary write leaves a recoverable staged transaction");

const saved = JSON.parse(firstCommitted);
saved.projects.push({ ...saved.projects[0], id: "invalid" });
const quarantined = parseStudioProjectStore(saved);
check(quarantined.projects.length === 1 && quarantined.warnings.some((warning) => warning.includes("quarantined")), "one corrupt record is quarantined without hiding valid projects");
check(parseStudioProjectStore({ ...saved, schemaVersion: 99 }).projects.length === 0, "unsupported project-store versions fail closed");

const malformedStorage = new MemoryStorage();
malformedStorage.values.set(STUDIO_PROJECT_STORAGE_KEY, "not json");
const malformed = loadStudioProjects(malformedStorage);
check(malformed.projects.length === 0 && malformed.warnings.length === 1, "malformed storage returns an explicit warning and no projects");

let bounded = [] as typeof projects;
for (let index = 0; index < STUDIO_PROJECT_LIMIT; index += 1) {
  bounded = upsertStudioProject(bounded, createStudioProject({ name: `Project ${index}`, primaryDataset: nimbus }, {
    id: `project_bound-${String(index).padStart(4, "0")}`,
    now: new Date(Date.UTC(2026, 6, 20, 0, index)).toISOString(),
  }));
}
let projectLimitRejected = false;
try {
  upsertStudioProject(bounded, createStudioProject({ name: "One too many", primaryDataset: nimbus }, { id: "project_bound-9999", now: at(7) }));
} catch { projectLimitRejected = true; }
check(bounded.length === STUDIO_PROJECT_LIMIT && projectLimitRejected, "the named-project collection has a hard upper bound");

const cm1Example = BUNDLED_PROJECT_EXAMPLES.find((example) => example.id === "cm1-supercell-review")!;
const exampleProject = instantiateBundledProjectExample(cm1Example, cm1, { id: "project_example-01", now: at(8) });
check(exampleProject.analysis.plotType === "composite" && exampleProject.primaryDataset.kind === "cm1_netcdf", "bundled CM1 workflow becomes a real project after local data selection");
let wrongExampleSourceRejected = false;
try { instantiateBundledProjectExample(cm1Example, nimbus, { id: "project_example-02", now: at(9) }); } catch { wrongExampleSourceRejected = true; }
check(wrongExampleSourceRejected, "bundled examples never pretend an incompatible or absent physical dataset is available");
check(BUNDLED_PROJECT_EXAMPLES.filter((example) => example.availability === "reference_only").every((example) => example.openMode === "select_local_dataset"), "reference examples disclose their local-data requirement");
const includedExample = BUNDLED_PROJECT_EXAMPLES.find((example) => example.availability === "included")!;
check(includedExample.openMode === "open_packaged_dataset" && includedExample.analysis.fieldName === "reflectivity_dbz", "included manufactured example opens directly with a familiar view");
const includedProject = instantiateBundledProjectExample(
  includedExample,
  createLocalDatasetSource("nimbus_run", "nimbus-example://manufactured-supercell-v1"),
  { id: "project_example-03", now: at(9) },
);
check(createStudioProjectPublication(includedProject, at(10)).sourceRequirements[0]?.availability === "packaged_immutable", "portable projects distinguish included immutable data from external local paths");

const publicationText = serializeStudioProjectPublication(createStudioProjectPublication(project, at(10)));
const publication = parseStudioProjectPublication(publicationText);
check(publication.fieldData === "references_only" && publication.sourceRequirements.length === 4 && publication.sourceRequirements.every(({ availability }) => availability === "external_local"), "portable project bundles disclose every external data dependency");
const importedProject = materializeStudioProjectPublication(publication, at(11));
check(importedProject.id !== project.id && importedProject.analysis.fieldName === project.analysis.fieldName, "import creates a collision-safe local identity while retaining analysis state");
let tamperedRequirementRejected = false;
try {
  const tampered = JSON.parse(publicationText);
  tampered.sourceRequirements[0].kind = "nimbus_run";
  parseStudioProjectPublication(JSON.stringify(tampered));
} catch { tamperedRequirementRejected = true; }
check(tamperedRequirementRejected, "project bundles fail closed when source requirements are tampered with");
const relocatedCm1 = createLocalDatasetSource("cm1_netcdf", ["/relocated/cm1out_000001.nc", "/relocated/cm1out_000002.nc"]);
const reboundPrimary = rebindStudioProjectSource(project, { kind: "primary" }, relocatedCm1, at(12));
check(reboundPrimary.primaryDataset.locator.path.startsWith("/relocated/") && reboundPrimary.workspace.investigation === null && reboundPrimary.workspace.comparison === null, "primary relinking invalidates identity-bound recipes without discarding plot state");
const relocatedNimbus = createLocalDatasetSource("nimbus_run", "/relocated/reference-storm");
const reboundComparison = rebindStudioProjectSource(project, { kind: "comparison" }, relocatedNimbus, at(13));
check(reboundComparison.comparisonDataset?.locator.path === "/relocated/reference-storm" && reboundComparison.workspace.comparison === null, "comparison sources can be rebound independently");
let wrongRelinkRejected = false;
try { rebindStudioProjectSource(project, { kind: "primary" }, nimbus, at(14)); } catch { wrongRelinkRejected = true; }
check(wrongRelinkRejected, "relinking cannot silently swap a CM1 requirement for Nimbus data");

const reachablePaths = new Set(["/cases/right-mover/cm1out_000001.nc", "/runs/reference-storm"]);
const sourceAudit = await auditStudioProjectSources(project, (path) => reachablePaths.has(path), at(15));
check(!sourceAudit.reopenReady && sourceAudit.entries.some(({ status }) => status === "partial"), "source audit detects a partially relocated multi-file dataset");
check(summarizeStudioProjectSourceAudit(sourceAudit).includes("Relink affected sources"), "source audit summary gives the next recovery action");
let packagedProbeCalls = 0;
const packagedAudit = await auditStudioProjectSources(includedProject, () => { packagedProbeCalls += 1; return false; }, at(16));
check(packagedAudit.reopenReady && packagedAudit.entries[0]?.status === "packaged" && packagedProbeCalls === 0, "included immutable sources verify without filesystem probing");

console.log(`studio projects: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
