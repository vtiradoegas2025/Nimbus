import { useMemo } from "react";
import { nativeFields, scanSource, sourceKey, reduceResearchSurface } from "@/lib/research/data";
import { binCenters } from "@/lib/research/statistics";
import { NumberControl, RESEARCH_COLORS, scientificLayout, TaskControls, useResearchResult, useResearchSettings, useResearchTask, type ResearchProps, type ResearchResult } from "./common";
const SPECIES = ["qc", "qr", "qi", "qs", "qg", "qh", "cloud_condensate"];
export function HydrometeorAnalysis({ source, steps, onResult }: ResearchProps) {
    const fields = nativeFields(source), species = SPECIES.filter(field => fields.includes(field));
    const [settings, update] = useResearchSettings({ field: fields.includes("reflectivity_dbz") ? "reflectivity_dbz" : species[0] ?? fields[0] ?? "", other: fields.includes("w") ? "w" : "", mask: "", maskThreshold: 5, min: 0, max: 60, otherMin: -20, otherMax: 60, bins: 40, zMin: 0, zMax: 20, heightBins: 40, speciesProfiles: false, dryDensity: "", precipitation: "", frameStride: 1 });
    const selected = [settings.field, settings.other, settings.mask, settings.dryDensity, ...(settings.speciesProfiles ? species : [])].filter(Boolean);
    const key = JSON.stringify([sourceKey(source, selected), steps.map(step => [step.index, step.publicationGeneration]), settings]);
    const task = useResearchTask(key, async (signal, progress) => {
        const value = { min: settings.min, max: settings.max, bins: settings.bins }, height = { min: settings.zMin, max: settings.zMax, bins: settings.heightBins }, joint = settings.other ? { min: settings.otherMin, max: settings.otherMax, bins: settings.bins } : undefined;
        const isMixingRatio = (field: string) => /^(kg\/kg|kg kg-1|kg kg\^-1|g\/kg|1)$/.test(source.step.fieldUnits[field] ?? "");
        if (settings.dryDensity && (!isMixingRatio(settings.field) || !SPECIES.includes(settings.field)))
            throw new Error("Mass integration requires a saved hydrometeor mass mixing ratio with declared kg/kg or g/kg units");
        if (settings.dryDensity && !/^(kg\/m3|kg m-3|kg m\^-3|kg\/m\^3)$/.test(source.step.fieldUnits[settings.dryDensity] ?? ""))
            throw new Error("Dry-air density must declare kg/m³ units");
        const options = { field: settings.field, other: settings.other || undefined, mask: settings.mask || undefined, maskThreshold: settings.maskThreshold, dryDensity: settings.dryDensity || undefined, massFactor: source.step.fieldUnits[settings.field] === "g/kg" ? .001 : 1, value, height, joint, threshold: 0 };
        const current = await scanSource(source, options, signal, p => progress(`Scanning native distributions ${Math.round(p * 100)}%`));
        const profiles: [
            string,
            typeof current
        ][] = [[settings.field, current]];
        if (settings.speciesProfiles)
            for (const field of species) {
                if (field === settings.field)
                    continue;
                progress(`Scanning ${field} native profile`);
                profiles.push([field, await scanSource(source, { ...options, field, other: undefined, joint: undefined, dryDensity: undefined }, signal)]);
            }
        if (settings.precipitation && Math.ceil(steps.length / settings.frameStride) > 2000)
            throw new Error("Choose a precipitation stride yielding at most 2,000 frames");
        const precipitation: {
            time: number;
            mean: number | null;
            max: number | null;
        }[] = [];
        if (settings.precipitation)
            for (let i = 0; i < steps.length; i += settings.frameStride) {
                const step = steps[i];
                if (step.simulationTimeS == null || !step.fields.includes(settings.precipitation))
                    throw new Error("Precipitation history requires the selected field and physical time at every requested saved frame");
                if (step.fieldUnits[settings.precipitation] !== source.step.fieldUnits[settings.precipitation])
                    throw new Error("Precipitation history units change between frames");
                if (signal.aborted)
                    throw new DOMException("Analysis cancelled", "AbortError");
                progress(`Reading precipitation frame ${i + 1}/${steps.length}`);
                const reduction = await reduceResearchSurface({ runDir: source.runDir, step }, settings.precipitation, signal);
                precipitation.push({ time: step.simulationTimeS, mean: reduction.mean, max: reduction.max });
            }
        return { current, profiles, precipitation, settings, value, height, joint };
    });
    const result = useMemo<ResearchResult | null>(() => {
        if (!task.value)
            return null;
        const { current, profiles, precipitation, settings: s, value, height, joint } = task.value;
        const unit = (field: string) => `${field} (${source.step.fieldUnits[field] ?? "native units"})`;
        const panels: ResearchResult["panels"] = [{ id: "cfad", title: "Height–frequency distribution (CFAD)", data: [{ type: "heatmap", x: binCenters(value), y: binCenters(height), z: current.cfad, colorscale: "Viridis", zsmooth: false, colorbar: { title: { text: "% finite cells at height" } } }], layout: scientificLayout("Height–frequency distribution (CFAD)", unit(s.field), "Saved height bin (km)") }];
        const byUnit = new Map<string, typeof profiles>();
        for (const entry of profiles) {
            const key = source.step.fieldUnits[entry[0]] ?? "unknown";
            byUnit.set(key, [...(byUnit.get(key) ?? []), entry]);
        }
        for (const [units, group] of byUnit)
            panels.push({ id: `profiles-${units}`, title: `Conditional profiles (${units})`, data: group.map(([field, r], i) => ({ type: "scatter", mode: "lines", x: r.heightMeans, y: binCenters(height), name: field, connectgaps: false, line: { color: RESEARCH_COLORS[i % RESEARCH_COLORS.length] } })), layout: scientificLayout("Conditional height-bin means", units, "Saved height bin (km)") });
        if (joint)
            panels.push({ id: "joint", title: "Joint native-cell distribution", data: [{ type: "heatmap", x: binCenters(value), y: binCenters(joint), z: current.joint, colorscale: "Inferno", zsmooth: false, colorbar: { title: { text: "Paired native cells" } } }], layout: scientificLayout("Joint native-cell distribution", unit(s.field), unit(s.other)) });
        if (precipitation.length)
            panels.push({ id: "precipitation", title: "Saved precipitation history", data: [{ type: "scatter", mode: "lines+markers", x: precipitation.map(r => r.time), y: precipitation.map(r => r.mean), name: "Valid-domain mean", connectgaps: false }, { type: "scatter", mode: "lines+markers", x: precipitation.map(r => r.time), y: precipitation.map(r => r.max), name: "Valid-domain maximum", connectgaps: false }], layout: scientificLayout("Saved precipitation history", "Saved time (s)", unit(s.precipitation)) });
        return { title: "Hydrometeor and radar analysis", parameters: s, sources: precipitation.length ? steps.filter((_, i) => i % s.frameStride === 0).map(step => ({ runDir: source.runDir, step })) : [source], panels,
            notes: [`Mask: ${s.mask ? `${s.mask} ≥ ${s.maskThreshold} (${source.step.fieldUnits[s.mask] ?? "native units"})` : "all valid native cells"}. ${current.finite} finite primary samples; ${current.missing} missing; ${current.masked} rejected by mask.`,
                "CFAD percentages use every finite masked cell in the height bin as denominator. Values outside the plotted value range remain in that denominator.", `${current.belowRange} samples below the value range; ${current.aboveRange} above. Missing species are omitted, never zero-filled.`,
                "Height/value bin limits control the displayed distributions. Mass integrates every finite masked native cell in the valid domain.",
                "Profiles are native-cell means within physical-height bins. Reflectivity means use linear Z and are reported back in dBZ. Histograms retain raw dBZ bins.",
                current.massKg == null ? "Hydrometeor mass is unavailable without an explicitly saved dry-air density and exact physical cell volumes." : `Selected masked hydrometeor mass: ${current.massKg} kg (saved mass mixing ratio × saved dry-air density × cell volume).`,
                "This is analysis of saved fields, not a radar forward operator. Polarimetry and process budgets require their own saved diagnostics.", "Surface precipitation histories use the selected saved product and its units; no precipitation rate is inferred from cloud appearance."],
            tables: [{ name: "cfad", columns: ["height_bin_km", "value_bin", "percent_of_finite_height_cells"], rows: binCenters(height).flatMap((z, k) => binCenters(value).map((x, j) => [z, x, Number.isFinite(current.cfad[k][j]) ? current.cfad[k][j] : null])) },
                ...profiles.map(([field, r]) => ({ name: `profile-${field}`, columns: ["height_bin_km", "mean", "finite_count"], rows: binCenters(height).map((z, k) => [z, r.heightMeans[k], r.heightCounts[k]]) })),
                ...(joint ? [{ name: "joint-distribution", columns: [s.field, s.other, "native_pair_count"], rows: binCenters(joint).flatMap((y, k) => binCenters(value).map((x, j) => [x, y, current.joint[k][j]])) }] : []),
                ...(precipitation.length ? [{ name: "precipitation-history", columns: ["time_s", "mean", "maximum"], rows: precipitation.map(r => [r.time, r.mean, r.max]) }] : [])] };
    }, [task.value, source, steps]);
    useResearchResult(result, onResult);
    const picker = (label: string, key: "field" | "other" | "mask" | "dryDensity" | "precipitation", available: string[], optional = true) => <label>{label}<select value={settings[key]} onChange={e => update({ [key]: e.target.value })}>{optional && <option value="">Off</option>}{available.map(field => <option key={field}>{field}</option>)}</select></label>;
    return <><div className="research-controls">{picker("Primary distribution field", "field", fields, false)}{picker("Joint distribution field", "other", fields)}{picker("Common mask field", "mask", fields)}
    <NumberControl label="Mask minimum (mask field units)" value={settings.maskThreshold} onChange={maskThreshold => update({ maskThreshold })}/>
    {(["min", "max", "otherMin", "otherMax", "zMin", "zMax"] as const).map((key, i) => <NumberControl key={key} label={["Value minimum", "Value maximum", "Joint minimum", "Joint maximum", "Height minimum (km)", "Height maximum (km)"][i]} value={settings[key]} onChange={v => update({ [key]: v })}/>)}
    <NumberControl label="Distribution bins" value={settings.bins} onChange={bins => update({ bins })} min={2} max={128} step={1}/><NumberControl label="Height bins" value={settings.heightBins} onChange={heightBins => update({ heightBins })} min={2} max={128} step={1}/>
    <label><input type="checkbox" checked={settings.speciesProfiles} onChange={e => update({ speciesProfiles: e.target.checked })}/>All saved species profiles</label>
    {picker("Saved dry-air density for mass", "dryDensity", fields.filter(field => /^(rho_d|dry_air_density|dry_density)$/.test(field)))}
    {picker("Saved precipitation history", "precipitation", source.step.fields.filter(field => /precip_rate|accumulated_rainfall|rain_rate/.test(field)))}<NumberControl label="Precipitation frame stride" value={settings.frameStride} onChange={frameStride => update({ frameStride })} min={1} max={1000} step={1}/>
    </div><TaskControls task={task} label="Build hydrometeor diagnostics"/></>;
}
