import { useState } from "react";
import { ProjectNavigator } from "@nimbus/components/studio/ProjectNavigator";
import { createLocalDatasetSource } from "@nimbus/lib/dataset/datasetSources";
import { createPackagedAtmosphericExampleSource } from "@nimbus/lib/dataset/packagedAtmosphericExamples";
import { createStudioProject, deleteStudioProject, duplicateStudioProject, loadStudioProjects, renameStudioProject, saveStudioProjects, upsertStudioProject, type StudioProject } from "@nimbus/lib/studio/studioProjects";
import { createStudioProjectPublication, materializeStudioProjectPublication, parseStudioProjectPublication, rebindStudioProjectSource, serializeStudioProjectPublication } from "@nimbus/lib/studio/studioProjectPublication";
import { auditStudioProjectSources, summarizeStudioProjectSourceAudit } from "@nimbus/lib/studio/studioProjectSources";
import "@nimbus/components/studio/PlotStudio.css";

const initial = [
  createStudioProject({ name: "Classic supercell", description: "Linked thermodynamic sections and native vertical levels.", primaryDataset: createLocalDatasetSource("nimbus_run", "/simulations/classic"), analysis: { frameIndex: 23, fieldName: "theta_prime", plotType: "slice" }, workspace: { activePanel: "research" } }, { id: "project_classic-fixture", now: "2026-10-05T09:00:00.000Z" }),
  createStudioProject({ name: "CM1 sounding review", primaryDataset: createLocalDatasetSource("cm1_netcdf", ["/observations/missing/cm1_0001.nc", "/observations/missing/cm1_0002.nc"]), comparisonDataset: createLocalDatasetSource("nimbus_run", `/simulations/${"long-source-directory/".repeat(12)}comparison`), analysis: { fieldName: "w", plotType: "skewt" } }, { id: "project_cm1-fixture", now: "2026-10-05T10:00:00.000Z" }),
  createStudioProject({ name: "Teaching cloud", primaryDataset: createPackagedAtmosphericExampleSource() }, { id: "project_teaching-fixture", now: "2026-10-05T08:00:00.000Z" }),
];

export function ProjectsHarness() {
  const [projects, setProjects] = useState(() => localStorage.getItem("nimbus:studio:projects:v1") ? loadStudioProjects(localStorage).projects : initial);
  const [active, setActive] = useState<string | null>("project_classic-fixture");
  const [events] = useState<string[]>([]);
  const [control] = useState(() => ({ hold: false, release: null as (() => void) | null, fail: false, publication: "" }));
  const update = (next: StudioProject[]) => { saveStudioProjects(localStorage, next); setProjects(next); };
  window.__nimbusProjectsFixture = {
    events: () => events,
    holdVerification: () => { control.hold = true; },
    releaseVerification: () => { control.hold = false; control.release?.(); },
    failNext: () => { control.fail = true; },
    publication: () => control.publication,
  };
  const action = (name: string) => { events.push(name); if (control.fail) { control.fail = false; throw new Error("Fixture storage rejected this change"); } };
  return <main className="plot-studio plot-studio--projects-page" style={{ height: "100vh" }}><ProjectNavigator projects={projects} activeProjectId={active}
    onCreate={() => { action("create"); const project = createStudioProject({ name: "New analysis", primaryDataset: initial[0].primaryDataset }); update(upsertStudioProject(projects, project)); setActive(project.id); }}
    onOpen={project => { action(`open:${project.id}`); setActive(project.id); }}
    onRename={(project, name) => { action(`rename:${project.id}`); update(renameStudioProject(projects, project.id, name)); }}
    onDuplicate={project => { action(`duplicate:${project.id}`); update(duplicateStudioProject(projects, project.id)); }}
    onDelete={project => { action(`delete:${project.id}`); update(deleteStudioProject(projects, project.id)); }}
    onExport={project => { action(`export:${project.id}`); control.publication = serializeStudioProjectPublication(createStudioProjectPublication(project)); }}
    onImport={text => { action("import"); update(upsertStudioProject(projects, materializeStudioProjectPublication(parseStudioProjectPublication(text)))); }}
    onRelink={(project, role) => { action(`relink:${project.id}`); const source = role.kind === "comparison" ? project.comparisonDataset! : project.primaryDataset; update(upsertStudioProject(projects, rebindStudioProjectSource(project, role, createLocalDatasetSource(source.kind, "/relinked/source")))); }}
    onVerify={async project => { action(`verify:${project.id}`); if (control.hold) await new Promise<void>(resolve => { control.release = resolve; }); return summarizeStudioProjectSourceAudit(await auditStudioProjectSources(project, path => !path.includes("missing"))); }}
    onOpenExample={example => { action(`example:${example.id}`); }} /></main>;
}

declare global { interface Window { __nimbusProjectsFixture: { events: () => string[]; holdVerification: () => void; releaseVerification: () => void; failNext: () => void; publication: () => string } } }
