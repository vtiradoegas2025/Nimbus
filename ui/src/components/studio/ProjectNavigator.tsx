import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import {
  BUNDLED_PROJECT_EXAMPLES,
  STUDIO_PROJECT_LIMIT,
  projectDatasetSummary,
  type BundledProjectExample,
  type StudioProject,
} from "@/lib/studio/studioProjects";
import { STUDIO_PROJECT_PUBLICATION_MAX_BYTES, type StudioProjectSourceRole } from "@/lib/studio/studioProjectPublication";
import { datasetSourcePaths } from "@/lib/dataset/datasetSources";
import { packagedAtmosphericExample } from "@/lib/dataset/packagedAtmosphericExamples";
import { metaFor } from "@/lib/plot/plotRegistry";
import "./ProjectNavigator.css";

export interface ProjectNavigatorProps {
  projects: readonly StudioProject[];
  activeProjectId?: string | null;
  examples?: readonly BundledProjectExample[];
  storageWarnings?: readonly string[];
  canCreate?: boolean;
  onCreate: () => void | Promise<void>;
  onOpen: (project: StudioProject) => void | Promise<void>;
  onRename: (project: StudioProject, name: string) => void | Promise<void>;
  onDuplicate: (project: StudioProject) => void | Promise<void>;
  onDelete: (project: StudioProject) => void | Promise<void>;
  onExport: (project: StudioProject) => void | Promise<void>;
  onImport: (contents: string) => void | Promise<void>;
  onRelink: (project: StudioProject, role: StudioProjectSourceRole) => void | Promise<void>;
  onVerify: (project: StudioProject) => string | void | Promise<string | void>;
  onOpenExample: (example: BundledProjectExample) => void | Promise<void>;
  onClose?: () => void;
}

const dateFormat = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const activityTime = (project: StudioProject) => Math.max(Date.parse(project.updatedAt), Date.parse(project.lastOpenedAt ?? project.createdAt));
const formatTime = (timestamp: string) => dateFormat.format(new Date(timestamp));

function projectSources(project: StudioProject) {
  return [
    { label: "Primary", role: { kind: "primary" } as StudioProjectSourceRole, source: project.primaryDataset },
    ...(project.comparisonDataset ? [{ label: "Comparison", role: { kind: "comparison" } as StudioProjectSourceRole, source: project.comparisonDataset }] : []),
    ...project.ensembleDatasets.map((source, index) => ({ label: `Ensemble ${index + 1}`, role: { kind: "ensemble", index } as StudioProjectSourceRole, source })),
  ];
}

export function ProjectNavigator({
  projects, activeProjectId = null, examples = BUNDLED_PROJECT_EXAMPLES, storageWarnings = [], canCreate = true,
  onCreate, onOpen, onRename, onDuplicate, onDelete, onExport, onImport, onRelink, onVerify, onOpenExample, onClose,
}: ProjectNavigatorProps) {
  const [section, setSection] = useState<"saved" | "examples">("saved");
  const [query, setQuery] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [sort, setSort] = useState("recent");
  const [selectedId, setSelectedId] = useState<string | null>(activeProjectId);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const busyRef = useRef(false);
  const mounted = useRef(true);
  const knownIds = useRef(new Set(projects.map(project => project.id)));
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ text: string; failed: boolean } | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const renameButton = useRef<HTMLButtonElement>(null);
  const previousRename = useRef<string | null>(null);
  const deleteButton = useRef<HTMLButtonElement>(null);
  const detailsRef = useRef<HTMLElement>(null);
  const busy = busyKey !== null;
  const atLimit = projects.length >= STUDIO_PROJECT_LIMIT;

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    if (busy) return;
    if (previousRename.current && !renameId) renameButton.current?.focus();
    previousRename.current = renameId;
  }, [renameId, busy]);
  useEffect(() => { if (activeProjectId) setSelectedId(activeProjectId); }, [activeProjectId]);
  useEffect(() => {
    const added = projects.filter(project => !knownIds.current.has(project.id));
    knownIds.current = new Set(projects.map(project => project.id));
    if (added.length === 1) {
      setSelectedId(added[0].id);
      setSection("saved");
      setQuery("");
      setSourceFilter("all");
    }
  }, [projects]);

  const visibleProjects = useMemo(() => {
    const text = query.trim().toLocaleLowerCase();
    return projects.filter(project => (sourceFilter === "all" || project.primaryDataset.kind === sourceFilter)
      && (!text || [project.name, project.description, project.analysis.fieldName ?? "", ...projectSources(project).flatMap(({ source }) => [source.displayName, ...datasetSourcePaths(source)])]
        .join(" ").toLocaleLowerCase().includes(text)))
      .sort((left, right) => sort === "name" ? left.name.localeCompare(right.name)
        : activityTime(right) - activityTime(left) || left.name.localeCompare(right.name));
  }, [projects, query, sourceFilter, sort]);
  const selected = visibleProjects.find(project => project.id === selectedId) ?? visibleProjects[0] ?? null;
  const orderedExamples = useMemo(() => [...examples].sort((left, right) => Number(right.availability === "included") - Number(left.availability === "included")), [examples]);
  const includedExample = orderedExamples.find(example => example.availability === "included");
  useEffect(() => { if (detailsRef.current) detailsRef.current.scrollTop = 0; }, [selected?.id]);

  const run = async (key: string, action: () => unknown | Promise<unknown>, success?: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusyKey(key);
    setFeedback(null);
    try {
      const result = await action();
      if (mounted.current && (typeof result === "string" || success)) setFeedback({ text: typeof result === "string" ? result : success!, failed: false });
    } catch (error) {
      if (mounted.current) setFeedback({ text: error instanceof Error ? error.message : "The project action did not complete.", failed: true });
    } finally {
      busyRef.current = false;
      if (mounted.current) setBusyKey(null);
    }
  };
  const submitRename = (event: FormEvent, project: StudioProject) => {
    event.preventDefault();
    const name = renameValue.trim();
    if (!name) { setFeedback({ text: "Project name cannot be empty.", failed: true }); return; }
    void run(`rename:${project.id}`, async () => {
      await onRename(project, name);
      if (mounted.current) { setRenameId(null); renameButton.current?.focus(); }
    }, `Renamed project to “${name}”.`);
  };
  const resetFilters = () => { setQuery(""); setSourceFilter("all"); };
  const viewLabel = selected?.workspace.activePanel === "research" ? "Research workspace"
    : selected?.workspace.activePanel === "investigation" ? "Investigation workspace"
    : selected?.analysis.plotType ? metaFor(selected.analysis.plotType).label : "View not fixed";

  return <section className="project-navigator" aria-labelledby="project-navigator-title">
    <input ref={importInputRef} type="file" accept="application/json,.json" hidden onChange={event => {
      const file = event.target.files?.[0]; event.currentTarget.value = "";
      if (file) void run("import", async () => {
        if (file.size > STUDIO_PROJECT_PUBLICATION_MAX_BYTES) throw new Error("Project bundle exceeds the 750 kB import limit.");
        await onImport(await file.text());
      }, "Project imported. Select it to review its data sources and saved view.");
    }} />
    <header className="project-navigator__header">
      <div><h2 id="project-navigator-title">Projects</h2><p>Saved views, source data, and research workflows.</p></div>
      <div className="project-navigator__header-actions">
        <button type="button" className="project-navigator__create" disabled={busy || !canCreate || atLimit}
          title={busy ? "Wait for the current project action" : atLimit ? `Studio supports ${STUDIO_PROJECT_LIMIT} named projects` : canCreate ? "Save the current analysis as a named project" : "Open a dataset before creating a project"}
          onClick={() => void run("create", onCreate, "Project saved from the current view.")}>{busyKey === "create" ? "Creating…" : canCreate ? "New from current view" : "Open data to create"}</button>
        <button type="button" disabled={busy || atLimit} title={busy ? "Wait for the current project action" : atLimit ? `Studio supports ${STUDIO_PROJECT_LIMIT} named projects` : "Import a project recipe and review its source references"} onClick={() => importInputRef.current?.click()}>{busyKey === "import" ? "Importing…" : "Import project…"}</button>
        {onClose && <button type="button" onClick={onClose} aria-label="Close projects">Close</button>}
      </div>
    </header>
    <nav className="project-navigator__tabs" aria-label="Project library">
      <button type="button" aria-pressed={section === "saved"} onClick={() => setSection("saved")}>Saved projects <span>{projects.length}</span></button>
      <button type="button" aria-pressed={section === "examples"} onClick={() => setSection("examples")}>Examples and workflows <span>{examples.length}</span></button>
      <details className="project-navigator__privacy"><summary>About project files</summary><p>Projects store local data references and analysis settings. Field arrays remain in their source datasets. Relink those sources when moving a project to another computer.</p></details>
    </nav>
    {(storageWarnings.length > 0 || feedback) && <div className="project-navigator__feedback" role={storageWarnings.length || feedback?.failed ? "alert" : "status"}>
      {[...new Set([...storageWarnings, ...(feedback ? [feedback.text] : [])])].map(message => <p key={message}>{message}</p>)}
      <button type="button" aria-label="Dismiss project message" onClick={() => setFeedback(null)} disabled={storageWarnings.length > 0} title={storageWarnings.length ? "Stored project warnings remain until the underlying issue is resolved" : "Dismiss this action result"}>Dismiss</button>
    </div>}
    {section === "saved" ? <>
      <div className="project-navigator__filters">
        <label className="project-navigator__search">Find projects<input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Name, field or source path" /></label>
        <label>Data source<select value={sourceFilter} onChange={event => setSourceFilter(event.target.value)}><option value="all">All sources</option><option value="nimbus_run">Nimbus history</option><option value="cm1_netcdf">netCDF</option></select></label>
        <label>Sort projects<select value={sort} onChange={event => setSort(event.target.value)}><option value="recent">Recent activity</option><option value="name">Name</option></select></label>
        <span className="project-navigator__count">{visibleProjects.length} of {projects.length}</span>
      </div>
      {!selected ? <div className="project-navigator__empty">
        <h3>{projects.length ? "No matching projects" : "Your next analysis starts here"}</h3>
        <p>{projects.length ? "Search a different name, field or source path, or clear the filters." : "Open a dataset or an included example, arrange your plots, then save a named project to return to that work."}</p>
        <div>{projects.length ? <button type="button" onClick={resetFilters}>Clear filters</button> : <>
          {includedExample && <button type="button" className="project-navigator__open" disabled={busy} title={busy ? "Wait for the current project action" : "Open the included manufactured dataset"} onClick={() => void run(`example:${includedExample.id}`, () => onOpenExample(includedExample))}>Open included example</button>}
          <button type="button" onClick={() => setSection("examples")}>Browse workflows</button>
        </>}</div>
      </div> : <div className="project-navigator__library">
        <ol className="project-navigator__list" aria-label={`${visibleProjects.length} named projects`}>
          {visibleProjects.map(project => <li className="project-navigator__row" key={project.id}>
            <button type="button" aria-label={`Select ${project.name}`} aria-pressed={selected.id === project.id} disabled={busy}
              title={busy ? "Wait for the current project action" : project.name} onClick={() => setSelectedId(project.id)}>
              <span className="project-navigator__status-line"><span>{project.primaryDataset.kind === "cm1_netcdf" ? "netCDF" : "Nimbus history"}</span>{project.id === activeProjectId && <span className="project-navigator__active">Current</span>}</span>
              <strong>{project.name}</strong><span className="project-navigator__source-name">{project.primaryDataset.displayName}</span>
              <time dateTime={new Date(activityTime(project)).toISOString()}>{formatTime(new Date(activityTime(project)).toISOString())}</time>
            </button>
          </li>)}
        </ol>
        <section className="project-navigator__details" ref={detailsRef} aria-label="Project details">
          <header className="project-navigator__detail-heading"><span>{projectDatasetSummary(selected)}</span><h3>{selected.name}</h3>{selected.description && <p>{selected.description}</p>}</header>
          {renameId === selected.id && <form className="project-navigator__rename" onSubmit={event => submitRename(event, selected)}>
            <label htmlFor={`project-name-${selected.id}`}>Project name</label><input id={`project-name-${selected.id}`} value={renameValue} maxLength={96} autoFocus disabled={busy} onChange={event => setRenameValue(event.target.value)} />
            <button type="submit" disabled={busy} title={busy ? "Saving the project name" : "Save this name"}>Save name</button><button type="button" disabled={busy} title={busy ? "Wait for the rename to complete" : "Keep the existing name"} onClick={() => { setRenameId(null); renameButton.current?.focus(); }}>Cancel rename</button>
          </form>}
          <div className="project-navigator__actions">
            <button type="button" className="project-navigator__open" disabled={busy} aria-busy={busyKey === `open:${selected.id}`} title={busy ? "Wait for the current project action" : "Reopen saved sources and analysis settings"} onClick={() => void run(`open:${selected.id}`, () => onOpen(selected))}>{busyKey === `open:${selected.id}` ? "Opening…" : selected.id === activeProjectId ? "Reopen" : "Open"}</button>
            <button type="button" disabled={busy} title={busy ? "Wait for the current project action" : "Export a portable recipe with source references"} onClick={() => void run(`export:${selected.id}`, () => onExport(selected), `Exported “${selected.name}” as an analysis recipe.`)}>Publish recipe</button>
            <button type="button" disabled={busy} title={busy ? "Wait for the current project action" : "Check source paths without reading field arrays"} onClick={() => void run(`verify:${selected.id}`, async () => `${selected.name}: ${await onVerify(selected) ?? "Source check completed."}`)}>{busyKey === `verify:${selected.id}` ? "Checking sources…" : "Verify sources"}</button>
          </div>
          <dl className="project-navigator__summary"><div><dt>Saved view</dt><dd>{viewLabel}</dd></div><div><dt>Field</dt><dd>{selected.analysis.fieldName ?? "Selected when opened"}</dd></div><div><dt>Saved frame</dt><dd>{selected.analysis.frameIndex + 1}</dd></div></dl>
          <div className="project-navigator__sources"><h4>Data sources</h4><p className="project-navigator__hint">Availability is checked when you choose Verify sources or open the project.</p>
            <ul>{projectSources(selected).map(({ label, role, source }) => {
              const paths = datasetSourcePaths(source), included = !!packagedAtmosphericExample(source.locator.path);
              return <li key={label}><div className="project-navigator__source-heading"><strong>{label}</strong><span>{included ? "Included with Studio" : source.kind === "cm1_netcdf" ? "External netCDF" : "External Nimbus history"}</span>
                {!included && <button type="button" disabled={busy} title={busy ? "Wait for the current project action" : "Choose a replacement for this source"} onClick={() => void run(`relink:${label}:${selected.id}`, () => onRelink(selected, role))}>Relink {label.toLowerCase()}…</button>}</div>
                <p>{source.displayName}</p><code>{source.locator.path}</code>
                {paths.length > 1 && <details><summary>All {paths.length} file references</summary><ul>{paths.map(path => <li key={path}><code>{path}</code></li>)}</ul></details>}
              </li>;
            })}</ul>
          </div>
          <footer className="project-navigator__management">
            <div><button type="button" ref={renameButton} disabled={busy} title={busy ? "Wait for the current project action" : "Rename this project"} onClick={() => { setDeleteId(null); setRenameId(selected.id); setRenameValue(selected.name); }}>Rename</button>
              <button type="button" disabled={busy || atLimit} title={busy ? "Wait for the current project action" : atLimit ? `Studio supports ${STUDIO_PROJECT_LIMIT} named projects` : "Create a separate project with these settings"} onClick={() => void run(`duplicate:${selected.id}`, () => onDuplicate(selected), `Duplicated “${selected.name}”.`)}>Duplicate</button>
              <button type="button" ref={deleteButton} className="project-navigator__delete" disabled={busy} title={busy ? "Wait for the current project action" : "Review project deletion"} onClick={() => { setRenameId(null); setDeleteId(selected.id); }}>Delete project</button></div>
            {deleteId === selected.id && <div className="project-navigator__delete-confirm"><p>Delete “{selected.name}” and its scripting recovery data? Source dataset files stay in place.</p><div>
              <button type="button" disabled={busy} title={busy ? "Deleting this project" : "Confirm project deletion"} onClick={() => void run(`delete:${selected.id}`, async () => { await onDelete(selected); if (mounted.current) setDeleteId(null); }, `Deleted “${selected.name}”.`)}>Confirm delete</button>
              <button type="button" disabled={busy} title={busy ? "Wait for deletion to complete" : "Keep this project"} onClick={() => { setDeleteId(null); deleteButton.current?.focus(); }}>Keep project</button></div></div>}
            <p className="project-navigator__hint">Created {formatTime(selected.createdAt)} · Updated {formatTime(selected.updatedAt)}</p>
          </footer>
        </section>
      </div>}
    </> : <div className="project-navigator__examples-page"><header><h3>Start with a workflow</h3><p>The included dataset opens immediately. Other workflows use data you choose from this computer.</p></header>
      <ol className="project-navigator__examples" aria-label={`${orderedExamples.length} bundled project examples`}>{orderedExamples.map(example => <li key={example.id}>
        <div><span className="project-navigator__reference">{example.availability === "included" ? "Included example" : `${example.sourceModel === "CM1" ? "netCDF" : "Nimbus"} data`}</span><h3>{example.title}</h3><p>{example.summary}</p><details><summary>What opens</summary><p>{example.workflow} {example.availabilityNote}</p></details></div>
        <button type="button" disabled={busy} title={busy ? "Wait for the current project action" : example.availabilityNote} aria-busy={busyKey === `example:${example.id}`} onClick={() => void run(`example:${example.id}`, () => onOpenExample(example))}>{busyKey === `example:${example.id}` ? "Opening…" : example.availability === "included" ? "Open included example" : `Choose ${example.sourceModel === "CM1" ? "netCDF" : "Nimbus"} data…`}</button>
      </li>)}</ol>
    </div>}
  </section>;
}
