import { useMemo } from "react";
import { nativeFields, physicalAxes, physicalMeasures, scanSource, sourceKey, validShape, type AnalysisOptions } from "@/lib/research/data";
import { readSection } from "@/lib/research/sections";
import { binCenters } from "@/lib/research/statistics";
import { NumberControl, scientificLayout, TaskControls, useResearchResult, useResearchSettings, useResearchTask, type ResearchProps, type ResearchResult } from "./common";
export function ColdPoolAnalysis({ source, steps, onResult }: ResearchProps) {
    const fields = nativeFields(source), candidates = fields.filter(field => /theta_prime|buoyancy|temperature_perturbation/.test(field));
    const [settings, update] = useResearchSettings({ field: candidates[0] ?? fields[0] ?? "", threshold: -2, level: 0, min: -10, max: 4, history: false, frameStride: 1, section: true, sectionFraction: 0.5 });
    const key = JSON.stringify([sourceKey(source, [settings.field]), steps.map(step => [step.index, step.publicationGeneration]), settings]);
    const task = useResearchTask(key, async (signal, progress) => {
        const axes = physicalAxes(source.step.grid), shape = validShape(source.step.grid), lo = source.step.grid?.validIndexRange?.lower ?? [0, 0, 0], measures = physicalMeasures(source.step.grid!, axes);
        if (measures.area == null)
            throw new Error("Physical footprint area requires declared uniform horizontal cell spacing; center coordinates alone do not establish cell-face areas");
        const dx = measures.dx!, dy = measures.dy!;
        const footprint = { x: { min: (axes[0][lo[0]] - .5 * dx) / 1000, max: (axes[0][lo[0] + shape[0] - 1] + .5 * dx) / 1000, bins: Math.min(128, shape[0]) }, y: { min: (axes[1][lo[1]] - .5 * dy) / 1000, max: (axes[1][lo[1] + shape[1] - 1] + .5 * dy) / 1000, bins: Math.min(128, shape[1]) } };
        const options: AnalysisOptions = { field: settings.field, value: { min: settings.min, max: settings.max, bins: 48 }, height: { min: 0, max: 30, bins: 60 }, threshold: settings.threshold, below: true, zIndex: settings.level, footprint };
        const current = await scanSource(source, options, signal, p => progress(`Measuring native footprint ${Math.round(p * 100)}%`));
        const section = settings.section && shape[0] > 1 ? await readSection(source, [settings.field], { startX: axes[0][lo[0]] / 1000, endX: axes[0][lo[0] + shape[0] - 1] / 1000, startY: (axes[1][lo[1]] + (axes[1][lo[1] + shape[1] - 1] - axes[1][lo[1]]) * settings.sectionFraction) / 1000, endY: (axes[1][lo[1]] + (axes[1][lo[1] + shape[1] - 1] - axes[1][lo[1]]) * settings.sectionFraction) / 1000, samples: Math.max(2, Math.min(128, shape[0])) }, signal, progress) : null;
        const history: {
            step: typeof source.step;
            statistics: typeof current;
        }[] = [];
        if (settings.history) {
            if (Math.ceil(steps.length / settings.frameStride) > 2000)
                throw new Error("Choose a frame stride yielding at most 2,000 history frames");
            for (let i = 0; i < steps.length; i += settings.frameStride) {
                const step = steps[i];
                if (step.simulationTimeS == null)
                    throw new Error("Cold-pool histories require saved physical timestamps");
                if (step.fieldUnits[settings.field] !== source.step.fieldUnits[settings.field])
                    throw new Error("History field units differ between saved frames");
                progress(`Measuring saved frame ${i + 1}/${steps.length}`);
                const statistics = step.index === source.step.index && step.publicationGeneration === source.step.publicationGeneration ? current : await scanSource({ runDir: source.runDir, step }, { ...options, footprint: undefined }, signal);
                history.push({ step, statistics: { ...statistics, cfad: [], joint: [], footprint: [], heightCounts: [], heightMeans: [], valueCounts: [] } });
            }
        }
        return { current, history, footprint, options, settings, section };
    });
    const result = useMemo<ResearchResult | null>(() => {
        if (!task.value)
            return null;
        const { current, history, footprint, options, settings: s, section } = task.value;
        const unit = source.step.fieldUnits[s.field] ?? "native units";
        const panels: ResearchResult["panels"] = [{ id: "footprint", title: "Threshold footprint", data: [{ type: "heatmap", x: binCenters(footprint.x), y: binCenters(footprint.y), z: current.footprint, zmin: 0, zmax: 100, colorscale: "Blues", reversescale: true, zsmooth: false, colorbar: { title: { text: `% finite cells ≤ ${s.threshold} ${unit}` } } }], layout: { ...scientificLayout("Threshold footprint", "x (km)", "y (km)"), yaxis: { title: { text: "y (km)" }, scaleanchor: "x", scaleratio: 1 } } },
            { id: "distribution", title: "Native-cell anomaly distribution", data: [{ type: "bar", x: binCenters(options.value), y: current.valueCounts, name: "Native cells" }], layout: scientificLayout("Native-cell anomaly distribution", `${s.field} (${unit})`, "Native cell count") }];
        if (section)
            panels.push({ id: "vertical-section", title: "Linked cold-pool section", data: [{ type: "heatmap", x: section.distanceKm, y: section.heightKm, z: section.values[s.field], colorscale: "RdBu", zmin: -Math.max(1e-12, ...section.values[s.field].map(row => row.reduce<number>((m, v) => Math.max(m, Math.abs(v ?? 0)), 0))), zmax: Math.max(1e-12, ...section.values[s.field].map(row => row.reduce<number>((m, v) => Math.max(m, Math.abs(v ?? 0)), 0))), zsmooth: false, colorbar: { title: { text: `${s.field} (${unit})` } } }, { type: "contour", x: section.distanceKm, y: section.heightKm, z: section.values[s.field], contours: { start: s.threshold, end: s.threshold, size: 1, coloring: "none" }, line: { color: "#ffffff", width: 2 }, showscale: false, connectgaps: false, name: "Cold threshold" }], layout: scientificLayout("Linked cold-pool section", "Distance along x section (km)", "Saved height (km)") });
        if (history.length)
            panels.push({ id: "area-history", title: "Physical area history", data: [{ type: "scatter", mode: "lines+markers", x: history.map(row => row.step.simulationTimeS), y: history.map(row => row.statistics.areaKm2), name: "Area", connectgaps: false }], layout: scientificLayout("Physical area history", "Saved time (s)", "Threshold area (km²)") }, { id: "anomaly-history", title: "Threshold anomaly history", data: [{ type: "scatter", mode: "lines+markers", x: history.map(row => row.step.simulationTimeS), y: history.map(row => row.statistics.min), name: "Domain minimum", connectgaps: false }, { type: "scatter", mode: "lines+markers", x: history.map(row => row.step.simulationTimeS), y: history.map(row => row.statistics.selectedMean), name: "Threshold-cell mean", connectgaps: false }], layout: scientificLayout("Threshold anomaly history", "Saved time (s)", `${s.field} (${unit})`) });
        return { title: "Cold-pool measurements", parameters: s, sources: history.length ? history.map(row => ({ runDir: source.runDir, step: row.step })) : [source], panels,
            notes: [`Threshold definition: ${s.field} ≤ ${s.threshold} ${unit}, native model level ${s.level}. ${current.selected} selected cells; ${current.areaKm2?.toPrecision(7)} km².`,
                `Finite coverage: ${current.finite} cells; ${current.missing} missing. Minimum ${current.min}; threshold mean ${current.selectedMean}. Missing cells are excluded, never treated as clear air.`,
                "The map reports the fraction of native cells selected in each display bin. Area and statistics scan every valid native cell, independent of map resolution.",
                "The reference is the saved field's definition. A theta perturbation threshold is an operational cold-pool definition, not a universal thermodynamic boundary. The linked x section uses the selected horizontal position; interpolation adds no source detail.",
                "This tool measures a model-level footprint. It does not infer cold-pool depth or density-current intensity from theta perturbation alone."],
            tables: [{ name: "footprint-summary", columns: ["time_s", "level_index", "threshold", "selected_cells", "finite_cells", "missing_cells", "area_km2", "minimum", "selected_mean"], rows: (history.length ? history : [{ step: source.step, statistics: current }]).map(({ step, statistics: r }) => [step.simulationTimeS, s.level, s.threshold, r.selected, r.finite, r.missing, r.areaKm2, r.min, r.selectedMean]) },
                { name: "anomaly-distribution", columns: ["bin_center", "native_cell_count"], rows: binCenters(options.value).map((x, i) => [x, current.valueCounts[i]]) }] };
    }, [task.value, source]);
    useResearchResult(result, onResult);
    let levels = 1;
    try {
        levels = validShape(source.step.grid)[2];
    }
    catch { /* Source error is shown by the analysis. */ }
    return <><div className="research-controls"><label>Anomaly field<select value={settings.field} onChange={e => update({ field: e.target.value })}>{fields.map(field => <option key={field}>{field}</option>)}</select></label>
    <NumberControl label="Cold threshold (field units)" value={settings.threshold} onChange={threshold => update({ threshold })}/><NumberControl label="Footprint model level" value={settings.level} onChange={level => update({ level })} min={0} max={levels - 1} step={1}/>
    <NumberControl label="Histogram minimum" value={settings.min} onChange={min => update({ min })}/><NumberControl label="Histogram maximum" value={settings.max} onChange={max => update({ max })}/>
    <label><input type="checkbox" checked={settings.section} onChange={e => update({ section: e.target.checked })}/>Include linked vertical section</label><NumberControl label="Section y fraction" value={settings.sectionFraction} onChange={sectionFraction => update({ sectionFraction })} min={0} max={1}/>
    <label><input type="checkbox" checked={settings.history} onChange={e => update({ history: e.target.checked })}/>Build saved-time histories</label><NumberControl label="History frame stride" value={settings.frameStride} onChange={frameStride => update({ frameStride })} min={1} max={1000} step={1}/></div><TaskControls task={task} label="Measure cold-pool footprint"/></>;
}
