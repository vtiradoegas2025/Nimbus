import { useMemo } from "react";
import type { Data } from "plotly.js";
import { nativeFields, readColumn, sourceKey, validShape, type ResearchColumn } from "@/lib/research/data";
import { layerSummary, profileDifference, windReferenceOffset } from "@/lib/research/profiles";
import { columnTemperatureCelsius, columnDewpointCelsius, pressureUnitFromLabel, toHPa, pToY, skewX } from "@/lib/science/thermo";
import { finite, NumberControl, RESEARCH_COLORS, scientificLayout, TaskControls, useResearchResult, useResearchSettings, useResearchTask, type ResearchProps, type ResearchResult } from "./common";
export function EnvironmentAnalysis({ source, steps, onResult }: ResearchProps) {
    const fields = nativeFields(source);
    const shape = useMemo(() => { try {
        return validShape(source.step.grid);
    }
    catch {
        return [1, 1, 1];
    } }, [source.step.grid]);
    const [settings, update] = useResearchSettings({ field: fields.includes("theta") ? "theta" : fields[0] ?? "", x: 0, y: 0, inflowX: 0, inflowY: 0, reference: 0, stormU: 0, stormV: 0, relative: false, layerBottom: 0, layerTop: 3 });
    const selected = [settings.field, ...["u", "v", "p", "temperature", "theta", "qv"].filter(field => fields.includes(field))];
    const key = JSON.stringify([sourceKey(source, selected), steps.map(step => [step.index, step.publicationGeneration]), settings]);
    const task = useResearchTask(key, async (signal) => {
        if (settings.layerTop <= settings.layerBottom)
            throw new Error("Layer top must be above its bottom");
        const initial = steps[settings.reference];
        if (!initial)
            throw new Error("Reference frame is unavailable");
        if (initial.fieldUnits[settings.field] !== source.step.fieldUnits[settings.field])
            throw new Error("Reference and current profile units differ");
        const profiles: ResearchColumn[] = [];
        profiles.push(await readColumn({ runDir: source.runDir, step: initial }, selected, settings.inflowX, settings.inflowY, signal));
        profiles.push(await readColumn(source, selected, settings.inflowX, settings.inflowY, signal));
        profiles.push(await readColumn(source, selected, settings.x, settings.y, signal));
        return { profiles, settings };
    });
    const result = useMemo<ResearchResult | null>(() => {
        if (!task.value)
            return null;
        const { profiles, settings: s } = task.value;
        const names = [`Reference saved t=${profiles[0].source.step.simulationTimeS ?? "?"} s`, `Inflow selection t=${source.step.simulationTimeS ?? "?"} s`, `Current selection t=${source.step.simulationTimeS ?? "?"} s`];
        const label = `${s.field} (${source.step.fieldUnits[s.field] ?? "native units"})`;
        const profilesData: Data[] = profiles.map((p, i) => ({ type: "scatter", mode: "lines", name: names[i], x: finite(p.values[s.field]), y: p.zM.map(z => z / 1000), connectgaps: false, line: { color: RESEARCH_COLORS[i] } }));
        const hodographs: Data[] = [], soundings: Data[] = [];
        const notes = ["Reference and inflow are user-selected saved columns. First saved time is not assumed to be the initial condition.", "All vertical axes use saved physical heights; no ground elevation is inferred.", "Profile differences interpolate the reference only within adjacent finite samples, without extrapolation."];
        profiles.forEach((p, i) => {
            if (p.values.u && p.values.v) {
                try {
                    if (![p.source.step.fieldUnits.u, p.source.step.fieldUnits.v].every(unit => /^(m\/s|m s-1|m s\^-1|m s\*\*-1)$/.test(unit ?? "")))
                        throw new Error("Hodographs require declared m/s wind units");
                    const offset = windReferenceOffset(p.source.step, s.relative ? [s.stormU, s.stormV] : [0, 0]);
                    hodographs.push({ type: "scatter", mode: "lines+markers", name: names[i], x: p.values.u.map((u, k) => Number.isFinite(u) && Number.isFinite(p.values.v[k]) ? u + offset[0] : null), y: p.values.v.map((v, k) => Number.isFinite(v) && Number.isFinite(p.values.u[k]) ? v + offset[1] : null), customdata: p.zM.map(z => z / 1000), hovertemplate: "u=%{x:.2f}, v=%{y:.2f} m/s<br>z=%{customdata:.3f} km<extra>%{fullData.name}</extra>", line: { color: RESEARCH_COLORS[i] }, connectgaps: false });
                }
                catch (error) {
                    notes.push(String(error instanceof Error ? error.message : error));
                }
            }
            const temp = p.values.temperature ?? p.values.theta, pressure = p.values.p, pUnit = pressureUnitFromLabel(p.source.step.fieldUnits.p);
            if (temp && pressure && pUnit) {
                const temperatureName = p.values.temperature ? "temperature" : "theta";
                if (!/^(K|kelvin)$/i.test(p.source.step.fieldUnits[temperatureName] ?? "")) {
                    notes.push("Skew-T overlays require declared Kelvin temperatures");
                    return;
                }
                const pHPa = pressure.map(p => toHPa(p, pUnit)), pPa = pHPa.map(p => p * 100);
                const tc = columnTemperatureCelsius(temp, pPa, !!p.values.temperature);
                const q = p.values.qv, unit = p.source.step.fieldUnits.qv;
                const factor = unit === "g/kg" ? .001 : /^(kg\/kg|kg kg-1|kg kg\^-1|1)$/.test(unit ?? "") ? 1 : null;
                const td = q && factor != null ? columnDewpointCelsius(q.map(v => v * factor), pPa) : null;
                for (const [values, dew] of [[tc, false], [td, true]] as const)
                    if (values)
                        soundings.push({ type: "scatter", mode: "lines", name: `${names[i]} ${dew ? "Td" : "T"}`, x: Array.from(values, (t, k) => Number.isFinite(t) && pHPa[k] > 0 ? skewX(t, pHPa[k]) : null), y: pHPa.map(p => Number.isFinite(p) && p > 0 ? pToY(p) : null), customdata: Array.from(values), hovertemplate: "T=%{customdata:.2f} °C<extra>%{fullData.name}</extra>", line: { color: RESEARCH_COLORS[i], dash: dew ? "dash" : "solid" }, connectgaps: false });
            }
        });
        const ticks = [1000, 850, 700, 500, 300, 200, 100];
        const layers = profiles.map((p, i) => { const summary = layerSummary(p.zM, p.values[s.field], s.layerBottom * 1000, s.layerTop * 1000); return [names[i], p.xM / 1000, p.yM / 1000, summary.mean, summary.difference, summary.coverage] as (string | number | null)[]; });
        const panels: ResearchResult["panels"] = [{ id: "profiles", title: "Environmental profile overlays", data: profilesData, layout: scientificLayout("Environmental profile overlays", label, "Saved height (km)") },
            { id: "difference", title: "Current minus reference", data: [{ type: "scatter", mode: "lines", x: profileDifference(profiles[2].zM, profiles[2].values[s.field], profiles[0].zM, profiles[0].values[s.field]), y: profiles[2].zM.map(z => z / 1000), name: label, connectgaps: false }], layout: scientificLayout("Current minus reference", `Difference in ${label}`, "Saved height (km)") }];
        if (hodographs.length)
            panels.push({ id: "hodographs", title: s.relative ? "Storm-relative hodographs" : "Earth-relative hodographs", data: hodographs, layout: { ...scientificLayout(s.relative ? "Storm-relative hodographs" : "Earth-relative hodographs", "u (m/s)", "v (m/s)"), yaxis: { title: { text: "v (m/s)" }, scaleanchor: "x", scaleratio: 1 } } });
        if (soundings.length)
            panels.push({ id: "soundings", title: "Saved sounding overlays", data: soundings, layout: { ...scientificLayout("Saved sounding overlays", "Skewed temperature coordinate (°C)", "Pressure (hPa)"), yaxis: { title: { text: "Pressure (hPa)" }, tickvals: ticks.map(pToY), ticktext: ticks.map(String) } } });
        notes.push("Sounding overlays use the existing thermodynamic conversions. Existing Skew-T surface-parcel/CAPE definitions are unchanged.");
        return { title: "Environmental and storm-relative analysis", parameters: s, sources: profiles.map(p => p.source), panels, notes, tables: [{ name: "layer-summaries", columns: ["column", "x_km", "y_km", `height_weighted_mean_${s.field}`, `top_minus_bottom_${s.field}`, "finite_layer_coverage"], rows: layers }, ...profiles.map((p, i) => ({ name: `profile-${i}`, columns: ["height_m", ...Object.keys(p.values)], rows: p.zM.map((z, k) => [z, ...Object.values(p.values).map(v => Number.isFinite(v[k]) ? v[k] : null)]) }))] };
    }, [task.value, source.step]);
    useResearchResult(result, onResult);
    return <><div className="research-controls"><label>Profile field<select value={settings.field} onChange={e => update({ field: e.target.value })}>{fields.map(field => <option key={field}>{field}</option>)}</select></label>
    <label>Reference saved frame<select value={settings.reference} onChange={e => update({ reference: Number(e.target.value) })}>{steps.map((step, i) => <option value={i} key={step.index}>{step.index} · {step.simulationTimeS ?? "?"} s</option>)}</select></label>
    <NumberControl label="Current x index" value={settings.x} onChange={x => update({ x })} min={0} max={shape[0] - 1} step={1}/><NumberControl label="Current y index" value={settings.y} onChange={y => update({ y })} min={0} max={shape[1] - 1} step={1}/>
    <NumberControl label="Inflow x index" value={settings.inflowX} onChange={inflowX => update({ inflowX })} min={0} max={shape[0] - 1} step={1}/><NumberControl label="Inflow y index" value={settings.inflowY} onChange={inflowY => update({ inflowY })} min={0} max={shape[1] - 1} step={1}/>
    <label><input type="checkbox" checked={settings.relative} onChange={e => update({ relative: e.target.checked })}/>Storm-relative winds</label>
    <NumberControl label="Storm u (m/s, Earth)" value={settings.stormU} onChange={stormU => update({ stormU })}/><NumberControl label="Storm v (m/s, Earth)" value={settings.stormV} onChange={stormV => update({ stormV })}/>
    <NumberControl label="Layer bottom (km)" value={settings.layerBottom} onChange={layerBottom => update({ layerBottom })}/><NumberControl label="Layer top (km)" value={settings.layerTop} onChange={layerTop => update({ layerTop })}/></div><TaskControls task={task} label="Load environmental profiles"/></>;
}
