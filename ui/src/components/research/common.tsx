import { useEffect, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";
import type { ResearchSource } from "@/lib/research/data";
import type { StepInfo } from "@/lib/run/runOutputs";
export interface ResearchTable {
    name: string;
    columns: string[];
    rows: (string | number | null)[][];
    rowCount?: number;
    rowAt?: (index: number) => (string | number | null)[];
}
export interface ResearchPanel {
    id: string;
    title: string;
    data: Data[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    onClick?: (event: Readonly<Record<string, unknown>>) => void;
    onRelayout?: (event: Readonly<Record<string, unknown>>) => void;
}
export interface ResearchResult {
    title: string;
    parameters: Record<string, unknown>;
    sources: ResearchSource[];
    panels: ResearchPanel[];
    tables: ResearchTable[];
    notes: string[];
}
export interface ResearchProps {
    source: ResearchSource;
    steps: readonly StepInfo[];
    onResult: (result: ResearchResult | null) => void;
}
export const RESEARCH_COLORS = ["#78b7ff", "#e9b65f", "#d87ac1", "#7fc5a1", "#ee8c7a", "#a3a2ed"];
export const scientificLayout = (title: string, x: string, y: string): Partial<Layout> => ({ title: { text: title }, xaxis: { title: { text: x }, zeroline: false }, yaxis: { title: { text: y }, zeroline: false }, showlegend: true, margin: { l: 65, r: 30, t: 50, b: 55 } });
export const finite = (values: ArrayLike<number>): (number | null)[] => Array.from(values, v => Number.isFinite(v) ? v : null);
export function useResearchTask<T>(key: string, work: (signal: AbortSignal, progress: (text: string) => void) => Promise<T>) {
    const [state, setState] = useState<{
        key: string;
        value: T | null;
        busy: boolean;
        error: string | null;
        progress: string;
    }>({ key: "", value: null, busy: false, error: null, progress: "" });
    const [owner] = useState(() => ({ sequence: 0, active: null as { key: string; controller: AbortController } | null }));
    const keyRef = useRef(key), workRef = useRef(work);
    keyRef.current = key;
    workRef.current = work;
    useEffect(() => () => {
        if (owner.active?.key === key) {
            owner.sequence++;
            owner.active.controller.abort();
            owner.active = null;
        }
    }, [key, owner]);
    const run = async () => {
        owner.active?.controller.abort();
        const controller = new AbortController();
        owner.active = { key: keyRef.current, controller };
        const token = ++owner.sequence, capturedKey = keyRef.current;
        const current = () => owner.sequence === token && keyRef.current === capturedKey && !controller.signal.aborted;
        setState({ key: capturedKey, value: null, busy: true, error: null, progress: "Preparing exact source reads…" });
        try {
            const value = await workRef.current(controller.signal, text => { if (current())
                setState(s => ({ ...s, progress: text })); });
            if (current())
                setState({ key: capturedKey, value, busy: false, error: null, progress: "Complete" });
        }
        catch (error) {
            if (current())
                setState({ key: capturedKey, value: null, busy: false, error: error instanceof Error ? error.message : String(error), progress: "" });
        }
    };
    const cancel = () => { owner.sequence++; owner.active?.controller.abort(); setState({ key: keyRef.current, value: null, busy: false, error: null, progress: "Cancelled" }); };
    return { ...(state.key === key ? state : { key, value: null, busy: false, error: null, progress: "Settings changed; run the analysis" }), run, cancel };
}
export function TaskControls({ task, label = "Run analysis" }: {
    task: {
        busy: boolean;
        error: string | null;
        progress: string;
        run: () => Promise<void>;
        cancel: () => void;
    };
    label?: string;
}) {
    return <div className="research-actions"><button type="button" className="btn btn--primary" disabled={task.busy} onClick={() => void task.run()}>{label}</button>
    {task.busy && <button type="button" className="btn" onClick={task.cancel}>Cancel analysis</button>}
    <span role={task.error ? "alert" : "status"}>{task.error ?? task.progress}</span></div>;
}
export function NumberControl({ label, value, onChange, min, max, step = "any" }: {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    step?: number | "any";
}) {
    const [draft, setDraft] = useState<string | null>(null);
    const commit = () => { if (draft == null)
        return; const value = Number(draft); if (draft.trim() && Number.isFinite(value) && (step !== 1 || Number.isSafeInteger(value)) && (min == null || value >= min) && (max == null || value <= max))
        onChange(value); setDraft(null); };
    return <label>{label}<input type="number" step={step} min={min} max={max} value={draft ?? value} onChange={event => setDraft(event.target.value)} onBlur={commit} onKeyDown={event => { if (event.key === "Enter") {
        event.preventDefault();
        commit();
    } if (event.key === "Escape")
        setDraft(null); }}/></label>;
}
export function useResearchResult(result: ResearchResult | null, onResult: ResearchProps["onResult"]): void {
    useEffect(() => { onResult(result); return () => onResult(null); }, [result, onResult]);
}
import { createContext, useContext } from "react";
export const ResearchSettingsContext = createContext<{
    values: Record<string, unknown>;
    update: (patch: Record<string, unknown>) => void;
}>({ values: {}, update: () => { } });
export function useResearchSettings<T extends Record<string, unknown>>(defaults: T): [
    T,
    (patch: Partial<T>) => void
] {
    const context = useContext(ResearchSettingsContext);
    const settings = { ...defaults };
    for (const key of Object.keys(defaults)) {
        const value = context.values[key], fallback = defaults[key];
        if (typeof value === typeof fallback && (typeof value !== "number" || Number.isFinite(value)) && (!Array.isArray(fallback) || Array.isArray(value)))
            (settings as Record<string, unknown>)[key] = value;
    }
    return [settings, context.update];
}
