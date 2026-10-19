import { clearResearchScanCache, researchScanCacheBytes } from "@/lib/research/scanCache";
import { verifyResearchBindings } from "@/lib/research/provenance";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import type { StepInfo } from "@/lib/run/runOutputs";
import type { EnsembleMemberInput } from "@/lib/comparison/ensembleAlignment";
import { ResearchPanelView } from "./ResearchPanel";
import { isTauriRuntime, writeBinaryFile } from "@/lib/platform/tauri";
import { downloadInvestigationArchive } from "@/lib/studio/investigationPublication";
import { parseResearchRecipe, RESEARCH_RECIPE_CHANGE_EVENT, RESEARCH_TABS, type ResearchRecipe, type ResearchTab } from "@/lib/research/recipe";
import { sanitizeResearchViews, type ResearchViews } from "@/lib/research/views";
import { publishResearch, parseResearchArchive } from "@/lib/research/publication";
import { ResearchSettingsContext, type ResearchResult, type ResearchPanel } from "./common";
import { SectionAnalysis } from "./SectionAnalysis";
import { EnvironmentAnalysis } from "./EnvironmentAnalysis";
import { ParcelAnalysis } from "./ParcelAnalysis";
import { ColdPoolAnalysis } from "./ColdPoolAnalysis";
import { HydrometeorAnalysis } from "./HydrometeorAnalysis";
import { ResolutionAnalysis } from "./ResolutionAnalysis";
import "./ResearchWorkspace.css";
const LABELS: Record<ResearchTab, string> = { sections: "Linked sections", environment: "Environment", parcels: "Parcel cohorts", "cold-pools": "Cold pools", pressure: "Pressure", hydrometeors: "Hydrometeors / radar", resolution: "Resolution studies" };
interface Props {
    dataset: LocalDatasetSource;
    steps: readonly StepInfo[];
    stepIdx: number;
    onStepChange: (index: number) => void;
    members: readonly EnsembleMemberInput[];
    onClose: () => void;
}
export function ResearchWorkspace(props: Props) { return <Workspace key={datasetSourceIdentity(props.dataset)} {...props}/>; }
function Workspace({ dataset, steps, stepIdx, onStepChange, members, onClose }: Props) {
    const identity = datasetSourceIdentity(dataset), storageKey = `nimbus:research-workspace:v1:${identity}`;
    const [initial] = useState(() => { try {
        const raw = localStorage.getItem(storageKey);
        return raw ? parseResearchRecipe(raw, identity) : null;
    }
    catch {
        return null;
    } });
    const [controlsOpen, setControlsOpen] = useState(true);
    const [tab, setTab] = useState<ResearchTab>(initial?.tab ?? "sections");
    const [settings, setSettings] = useState<ResearchRecipe["settings"]>(initial?.settings ?? {});
    const [title, setTitle] = useState(initial?.title ?? `${dataset.displayName} research figure`), [citation, setCitation] = useState(initial?.citation ?? "");
    const [bookmarks, setBookmarks] = useState<ResearchRecipe["bookmarks"]>(initial?.bookmarks ?? []), [bookmarkName, setBookmarkName] = useState("");
    const [resultCurrent, setResultCurrent] = useState(false);
    const [renderComplete, setRenderComplete] = useState(false);
    const renderGeneration = useRef(0), renderedPanels = useRef(new Set<string>()), expectedPanels = useRef(0);
    const [result, setResult] = useState<ResearchResult | null>(null), [message, setMessage] = useState<string | null>(null), [publishing, setPublishing] = useState(false), [revision, setRevision] = useState(0);
    const [viewRange, setViewRange] = useState<ResearchViews>(initial?.viewRanges ?? {});
    const viewScope = useRef("");
    const restoringViews = useRef(Boolean(initial?.viewRanges));
    const workspace = useRef<HTMLElement>(null), fileInput = useRef<HTMLInputElement>(null), resultRef = useRef(result), operation = useRef(0);
    resultRef.current = resultCurrent ? result : null;
    const source = useMemo(() => ({ runDir: dataset.locator.path, step: steps[stepIdx] }), [dataset.locator.path, steps, stepIdx]);
    const recipe = useMemo<ResearchRecipe>(() => ({ schema: "nimbus.research-workspace", version: 1, datasetIdentity: identity, tab, stepIndex: source.step?.index ?? 0, settings, title, citation, bookmarks, viewRanges: viewRange }), [identity, tab, source.step?.index, settings, title, citation, bookmarks, viewRange]);
    const recipeRef = useRef(recipe);
    recipeRef.current = recipe;
    const updateSettings = useCallback((patch: Record<string, unknown>) => setSettings(current => ({ ...current, [tab]: { ...current[tab], ...patch } })), [tab]);
    const context = useMemo(() => ({ values: settings[tab] ?? {}, update: updateSettings }), [settings, tab, updateSettings]);
    const present = useCallback((value: ResearchResult | null) => { setResultCurrent(value != null); if (value) {
        renderGeneration.current++;
        renderedPanels.current.clear();
        expectedPanels.current = value.panels.length;
        setRenderComplete(false);
        setResult(value);
        setSettings(current => JSON.stringify(current[tab]) === JSON.stringify(value.parameters) ? current : { ...current, [tab]: value.parameters });
        const scope = JSON.stringify([value.title, { ...value.parameters, probe: undefined }]);
        if (viewScope.current !== scope) {
            if (!restoringViews.current)
                setViewRange({});
            viewScope.current = scope;
            restoringViews.current = false;
        }
    } }, [tab]);
    const pendingRecipe = useRef<ResearchRecipe | null>(null);
    const flushRecipe = useCallback((showError: boolean) => {
        const value = pendingRecipe.current;
        if (!value)
            return;
        pendingRecipe.current = null;
        try {
            const text = JSON.stringify(value);
            if (localStorage.getItem(storageKey) !== text) {
                localStorage.setItem(storageKey, text);
                window.dispatchEvent(new CustomEvent(RESEARCH_RECIPE_CHANGE_EVENT, { detail: identity }));
            }
        }
        catch {
            if (showError)
                setMessage("Local recipe storage is unavailable; export the figure package to retain it.");
        }
    }, [storageKey, identity]);
    useEffect(() => { if (!source.step) return; pendingRecipe.current = recipe; const timer = setTimeout(() => flushRecipe(true), 250); return () => clearTimeout(timer); }, [recipe, flushRecipe, source.step]);
    useEffect(() => () => flushRecipe(false), [flushRecipe]);
    useEffect(() => () => { operation.current++; }, []);
    const restore = useCallback((incoming: ResearchRecipe) => {
        verifyResearchBindings(incoming.sourceBindings, members.flatMap(member => member.steps.map(step => ({ runDir: member.source.locator.path, step }))));
        const index = steps.findIndex(step => step.index === incoming.stepIndex);
        if (index < 0)
            throw new Error("The recipe's saved frame is unavailable");
        restoringViews.current = true;
        setViewRange(incoming.viewRanges ?? {});
        setSettings(incoming.settings);
        setTab(incoming.tab);
        setTitle(incoming.title);
        setCitation(incoming.citation);
        setBookmarks(incoming.bookmarks);
        setRevision(v => v + 1);
        onStepChange(index);
        setMessage("Recipe restored. Run the analysis to read its exact source data.");
    }, [steps, onStepChange, members]);
    const restored = useRef(false);
    useEffect(() => { if (!restored.current && initial && steps.length) {
        restored.current = true;
        const i = steps.findIndex(step => step.index === initial.stepIndex);
        if (i >= 0)
            onStepChange(i);
    } }, [initial, steps, onStepChange]);
    const publish = async () => {
        if (!result || !resultCurrent || !renderComplete || !workspace.current || publishing)
            return;
        const captured = result, root = workspace.current, expected = recipeRef.current, token = ++operation.current;
        const current = () => operation.current === token && resultRef.current === captured && recipeRef.current === expected;
        setPublishing(true);
        setMessage("Capturing completed research panels…");
        try {
            const archive = await publishResearch(root, captured, { ...expected, viewRanges: viewRange }, title, citation, current);
            if (!current())
                throw new Error("Research settings changed during export");
            const filename = `${title.replace(/[^A-Za-z0-9_-]/g, "-").slice(0, 100) || "research"}.zip`;
            if (isTauriRuntime()) {
                const { save } = await import("@tauri-apps/plugin-dialog");
                const path = await save({ title: "Save research figure package", defaultPath: filename, filters: [{ name: "Figure, recipe and numerical tables", extensions: ["zip"] }] });
                if (!current())
                    throw new Error("Research settings changed during export");
                if (path)
                    await writeBinaryFile(path, archive);
                else {
                    setMessage("Export cancelled");
                    return;
                }
            }
            else
                downloadInvestigationArchive(archive, filename);
            setMessage("Figure package saved with PNG, recipe, numerical CSV tables and SHA-256 provenance.");
        }
        catch (error) {
            setMessage(error instanceof Error ? error.message : String(error));
        }
        finally {
            if (operation.current === token)
                setPublishing(false);
        }
    };
    const updateViews = useCallback((panel: ResearchPanel, event: Readonly<Record<string, unknown>>) => {
        panel.onRelayout?.(event);
        const axes: Partial<Plotly.Layout> = {};
        for (const axis of ["xaxis", "yaxis"] as const) {
            const lo = event[`${axis}.range[0]`], hi = event[`${axis}.range[1]`];
            if (typeof lo === "number" && typeof hi === "number")
                axes[axis] = { ...panel.layout?.[axis], range: [lo, hi] };
            else if (event[`${axis}.autorange`] === true)
                axes[axis] = { ...panel.layout?.[axis], autorange: true };
        }
        const camera = event["scene.camera"];
        if (camera && typeof camera === "object")
            axes.scene = { camera };
        if (Object.keys(axes).length)
            setViewRange(current => {
                const next = { ...current, [panel.id]: { ...current[panel.id], ...axes } };
                if (panel.id.startsWith("section-"))
                    for (const other of result?.panels ?? []) {
                        if (other.id.startsWith("section-"))
                            next[other.id] = { ...next[other.id], ...axes };
                        else if (other.id.startsWith("profile") && axes.yaxis)
                            next[other.id] = { ...next[other.id], yaxis: axes.yaxis };
                    }
                const clean = sanitizeResearchViews(next);
                if (JSON.stringify(clean) === JSON.stringify(current))
                    return current;
                return Object.fromEntries(Object.entries(clean).map(([id, value]) => [id, JSON.stringify(value) === JSON.stringify(current[id]) ? current[id] : value]));
            });
    }, [result]);
    const panelRendered = useCallback((id: string, generation: number) => { if (generation !== renderGeneration.current)
        return; renderedPanels.current.add(id); if (renderedPanels.current.size === expectedPanels.current)
        setRenderComplete(true); }, []);
    const sharedProps = { source, steps, onResult: present };
    const changeTab = (next: ResearchTab) => { setResult(null); setResultCurrent(false); setTab(next); setControlsOpen(true); };
    if (!source.step)
        return <section className="research-workspace"><p>Open a dataset with complete saved frames.</p><button className="btn" onClick={onClose}>Close research workspace</button></section>;
    return <section className="research-workspace" ref={workspace} aria-label="Research workspace">
    <header className="research-header"><div><h2>Research workspace</h2><p>{dataset.displayName} · saved t = {source.step.simulationTimeS ?? "unknown"} s · step {source.step.index}</p></div><div className="research-actions"><button type="button" className="btn" aria-expanded={controlsOpen} onClick={() => setControlsOpen(value => !value)}>{controlsOpen ? "Hide analysis controls" : "Show analysis controls"}</button><button type="button" className="btn" onClick={onClose}>Close research workspace</button></div></header>
    <nav aria-label="Research workflows" className="research-tabs">{RESEARCH_TABS.map(name => <button type="button" className="btn" key={name} aria-pressed={tab === name} onClick={() => changeTab(name)}>{LABELS[name]}</button>)}</nav>
    <div className="research-controls"><label>Shared saved frame<select value={stepIdx} onChange={event => onStepChange(Number(event.target.value))}>{steps.map((step, i) => <option value={i} key={step.index}>{step.index} · {step.simulationTimeS ?? "unknown"} s</option>)}</select></label><span>Analysis scans saved native cells. Display sampling and numerical aggregation are recorded separately.</span></div>
    {(tab === "sections" || tab === "pressure") && resultCurrent && <button type="button" className="btn" disabled={!viewRange["section-0"]?.xaxis?.range} onClick={() => {
                const range = viewRange["section-0"]?.xaxis?.range as number[] | undefined, parameters = result?.parameters;
                if (!range || !parameters)
                    return;
                const x0 = Number(parameters.startX), y0 = Number(parameters.startY), x1 = Number(parameters.endX), y1 = Number(parameters.endY), length = Math.hypot(x1 - x0, y1 - y0);
                const low = Math.max(0, Math.min(...range)), high = Math.min(length, Math.max(...range));
                if (!(high > low && length > 0))
                    return;
                context.update({ startX: x0 + (x1 - x0) * low / length, startY: y0 + (y1 - y0) * low / length, endX: x0 + (x1 - x0) * high / length, endY: y0 + (y1 - y0) * high / length });
                setControlsOpen(true);
                setMessage("Section endpoints now match the visible distance range. Load the section to read finer source detail within that region.");
            }}>Refine section to visible distance range</button>}
    <ResearchSettingsContext.Provider value={context}><div key={`${tab}:${revision}`} className="research-configuration" hidden={!controlsOpen}>
      {tab === "sections" && <SectionAnalysis {...sharedProps}/>}{tab === "pressure" && <SectionAnalysis {...sharedProps} pressure/>}
      {tab === "environment" && <EnvironmentAnalysis {...sharedProps}/>}{tab === "parcels" && <ParcelAnalysis {...sharedProps}/>}
      {tab === "cold-pools" && <ColdPoolAnalysis {...sharedProps}/>}{tab === "hydrometeors" && <HydrometeorAnalysis {...sharedProps}/>}
      {tab === "resolution" && <ResolutionAnalysis {...sharedProps} members={members}/>}
    </div></ResearchSettingsContext.Provider>
    {result && <>{!resultCurrent && <p role="status">Showing the previous completed analysis. Run the analysis to apply the selected settings or saved frame.</p>}<div className="research-notes">{result.notes.map((note, i) => <p key={i}>{note}</p>)}</div>{resultCurrent && !renderComplete && <p role="status">Rendering synchronized research panels…</p>}<div className="research-panels" aria-busy={resultCurrent && !renderComplete}>{result.panels.map((panel, i) => <ResearchPanelView key={panel.id} panel={panel} index={i} range={viewRange[panel.id]} identity={`${identity}:${tab}:${panel.id}:${viewScope.current}`} onViewChange={updateViews} generation={renderGeneration.current} onRendered={panelRendered}/>)}</div>
    <details className="research-tables"><summary>Numerical results ({result.tables.length} tables)</summary>{result.tables.map(table => <div key={table.name}><h3>{table.name}</h3><p>{table.rowCount ?? table.rows.length} rows; the figure package includes every row.</p><div className="research-table-scroll"><table><thead><tr>{table.columns.map((column, i) => <th key={i}>{column}</th>)}</tr></thead><tbody>{Array.from({ length: Math.min(12, table.rowCount ?? table.rows.length) }, (_, i) => table.rowAt?.(i) ?? table.rows[i]).map((row, i) => <tr key={i}>{row.map((value, j) => <td key={j}>{value == null || typeof value === "number" && !Number.isFinite(value) ? "Missing" : typeof value === "number" ? Number(value.toPrecision(7)) : value}</td>)}</tr>)}</tbody></table></div></div>)}</details></>}
    <section className="research-publication" aria-label="Research publication"><h3>Figure package and replay recipe</h3><div className="research-controls"><label>Figure title<input value={title} maxLength={300} onChange={e => setTitle(e.target.value)}/></label><label>Citation / study definition<input value={citation} maxLength={2000} onChange={e => setCitation(e.target.value)}/></label><button type="button" className="btn" disabled={!result || !resultCurrent || !renderComplete || publishing} onClick={() => void publish()}>{publishing ? "Exporting…" : "Export research figure package"}</button><button type="button" className="btn" onClick={() => fileInput.current?.click()}>Import research recipe or package</button></div>
      <p>The package includes lettered panels, actual saved times, view ranges, settings and numerical tables. Built-in analyses execute no Python; the provenance states that explicitly. Source arrays remain dependencies.</p>
      <input ref={fileInput} type="file" accept=".json,.zip" hidden onChange={async (event) => { const file = event.target.files?.[0]; event.target.value = ""; if (!file)
        return; try {
        if (file.name.toLowerCase().endsWith(".zip")) {
            if (file.size > 96 * 1024 * 1024)
                throw new Error("Figure package exceeds 96 MiB");
            restore(await parseResearchArchive(new Uint8Array(await file.arrayBuffer()), identity));
        }
        else {
            if (file.size > 1000000)
                throw new Error("Recipe exceeds 1 MB");
            restore(parseResearchRecipe(await file.text(), identity));
        }
    }
    catch (error) {
        setMessage(error instanceof Error ? error.message : String(error));
    } }}/>
      <div className="research-controls"><label>Event bookmark<input value={bookmarkName} maxLength={200} onChange={e => setBookmarkName(e.target.value)}/></label><button type="button" className="btn" disabled={!bookmarkName.trim() || bookmarks.length >= 100} onClick={() => { setBookmarks(current => [...current, { name: bookmarkName.trim(), stepIndex: source.step.index, tab, settings: settings[tab] ?? {} }]); setBookmarkName(""); }}>Bookmark current analysis</button></div>
      {bookmarks.length > 0 && <ul>{bookmarks.map((bookmark, i) => <li key={i}><button type="button" className="btn" onClick={() => { try {
        restore({ ...recipe, stepIndex: bookmark.stepIndex, tab: bookmark.tab, settings: { ...settings, [bookmark.tab]: bookmark.settings } });
    }
    catch (error) {
        setMessage(String(error));
    } }}>{bookmark.name} · step {bookmark.stepIndex}</button><button type="button" className="btn" aria-label={`Remove bookmark ${bookmark.name}`} onClick={() => setBookmarks(current => current.filter((_, j) => j !== i))}>Remove</button></li>)}</ul>}
      <button type="button" className="btn" onClick={() => { const bytes = researchScanCacheBytes(); clearResearchScanCache(); setMessage(`Released ${(bytes / 1024 / 1024).toFixed(1)} MiB of cached analysis payloads. Displayed results remain available.`); }}>Clear analysis cache</button>
      {message && <p role="status">{message}</p>}
    </section>
  </section>;
}
