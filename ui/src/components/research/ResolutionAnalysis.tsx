import {cartesianDyMeters} from "@/lib/science/cartesianCoordinates";
import { useMemo } from "react";
import type { EnsembleMemberInput } from "@/lib/comparison/ensembleAlignment";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import { canonicalHistoryUnit } from "@/lib/plot/historyReductionCache";
import { axisFrame } from "@/lib/run/frameMotion";
import { nativeFields, scanSource, sourceKey, type ResearchSource } from "@/lib/research/data";
import { matchSavedTime, commonPhysicalBox } from "@/lib/research/resolution";
import { binCenters } from "@/lib/research/statistics";
import { NumberControl, RESEARCH_COLORS, scientificLayout, TaskControls, useResearchSettings, useResearchTask, useResearchResult, type ResearchProps, type ResearchResult } from "./common";
export function ResolutionAnalysis({ source, steps, members, onResult }: ResearchProps & {
    members: readonly EnsembleMemberInput[];
}) {
    const fields = nativeFields(source);
    const [settings, update] = useResearchSettings({ field: fields.includes("w") ? "w" : fields[0] ?? "", threshold: 5, min: -20, max: 60, bins: 40, tolerance: .001, history: false, frameStride: 1, groups: "", customRegion: false, xMin: -20, xMax: 20, yMin: -20, yMax: 20, zMin: 0, zMax: 20 });
    const key = JSON.stringify([sourceKey(source, [settings.field]), members.map(member => [datasetSourceIdentity(member.source), member.steps.map(step => [step.index, step.publicationGeneration])]), { ...settings, groups: undefined }]);
    const task = useResearchTask(key, async (signal, progress) => {
        if (members.length < 2)
            throw new Error("Add a comparison dataset or ensemble member before running a resolution study");
        if (members.length > 12)
            throw new Error("A research comparison supports at most twelve members");
        const times = settings.history ? steps.filter((_, i) => i % settings.frameStride === 0) : [source.step];
        if (times.length * members.length > 2000)
            throw new Error("Choose a frame stride yielding at most 2,000 member-frame scans");
        const rows: {
            label: string;
            source: ResearchSource;
            statistics: Awaited<ReturnType<typeof scanSource>>;
            spacing: number | null;
            region: ReturnType<typeof commonPhysicalBox>;
        }[] = [];
        const unit = canonicalHistoryUnit(source.step.fieldUnits[settings.field]);
        if (!unit)
            throw new Error("Resolution comparisons require declared matching field units");
        for (const reference of times) {
            if (reference.simulationTimeS == null)
                throw new Error("Resolution studies require saved physical times");
            const matched = members.map(member => matchSavedTime(member.steps, reference.simulationTimeS!, settings.tolerance));
            for (const step of matched) {
                const verdict = axisFrame(step);
                if (verdict.frame !== "earth-fixed")
                    throw new Error(verdict.detail ?? "Member coordinate references are unverified");
                if (canonicalHistoryUnit(step.fieldUnits[settings.field]) !== unit)
                    throw new Error("Comparison field units differ between members");
            }
            const intersection = commonPhysicalBox(matched), region = settings.customRegion ? { lower: [settings.xMin * 1000, settings.yMin * 1000, settings.zMin * 1000] as [
                    number,
                    number,
                    number
                ], upper: [settings.xMax * 1000, settings.yMax * 1000, settings.zMax * 1000] as [
                    number,
                    number,
                    number
                ] } : intersection;
            if (region.lower.some((v, i) => v < intersection.lower[i] || region.upper[i] > intersection.upper[i] || v >= region.upper[i]))
                throw new Error("Requested region must be contained within every member's physical coverage");
            for (let i = 0; i < members.length; i++) {
                const member = members[i], step = matched[i], memberSource = { runDir: member.source.locator.path, step };
                progress(`Scanning ${member.source.displayName} · ${step.simulationTimeS} s`);
                const statistics = await scanSource(memberSource, { field: settings.field, value: { min: settings.min, max: settings.max, bins: settings.bins }, height: { min: region.lower[2] / 1000, max: region.upper[2] / 1000, bins: 40 }, threshold: settings.threshold, region }, signal);
                const spacing = step.grid?.drM && step.grid.dzM ? Math.cbrt(step.grid.drM * cartesianDyMeters(step.grid,step.grid.drM) * step.grid.dzM) : null;
                rows.push({ label: member.source.displayName, source: memberSource, statistics: { ...statistics, cfad: [], joint: [], footprint: [], heightCounts: [], heightMeans: [] }, spacing, region });
            }
        }
        return { rows, settings };
    });
    const result = useMemo<ResearchResult | null>(() => {
        if (!task.value)
            return null;
        const { rows, settings: s } = task.value, groups = settings.groups.split(",").map(v => v.trim());
        const labels = members.map((member, i) => `${groups[i] ? `${groups[i]} · ` : ""}${member.source.displayName}`), unit = source.step.fieldUnits[s.field] ?? "native units";
        const atTime = rows.filter(row => row.source.step.simulationTimeS != null && Math.abs(row.source.step.simulationTimeS - (source.step.simulationTimeS ?? NaN)) <= s.tolerance);
        const selected = atTime.length ? atTime : rows.slice(0, members.length), bins = { min: s.min, max: s.max, bins: s.bins };
        const panels: ResearchResult["panels"] = [{ id: "distributions", title: "Native-grid response distributions", data: selected.map((row, i) => ({ type: "scatter", mode: "lines+markers", x: binCenters(bins), y: row.statistics.valueCounts.map(n => row.statistics.finite ? n / row.statistics.finite * 100 : null), name: labels[i], line: { color: RESEARCH_COLORS[i % RESEARCH_COLORS.length] } })), layout: scientificLayout("Native-grid response distributions", `${s.field} (${unit})`, "% finite native cells") },
            { id: "response", title: "Response versus declared resolution", data: [{ type: "scatter", mode: "text+markers", x: selected.map(row => row.spacing), y: selected.map(row => row.statistics.mean), text: labels, textposition: "top center", name: "Member mean" }], layout: scientificLayout("Native-cell mean versus nominal cell length", "Nominal (dx dy dz)⅓ (m)", `Native-cell mean ${s.field} (${unit})`) },
            { id: "threshold", title: "Physical threshold volumes", data: [{ type: "bar", x: labels, y: selected.map(row => row.statistics.volumeKm3), name: `${s.field} ≥ ${s.threshold}` }], layout: scientificLayout("Threshold volume of admitted native cells", "Member", "Volume (km³)") }];
        if (s.history)
            panels.push({ id: "history", title: "Native-grid response histories", data: members.map((member, i) => { const samples = rows.filter(row => row.source.runDir === member.source.locator.path); return { type: "scatter", mode: "lines+markers", x: samples.map(row => row.source.step.simulationTimeS), y: samples.map(row => row.statistics.mean), name: labels[i], connectgaps: false, line: { color: RESEARCH_COLORS[i % RESEARCH_COLORS.length] } }; }), layout: scientificLayout("Native-grid response histories", "Actual saved time (s)", `${s.field} native-cell mean (${unit})`) });
        return { title: "Sensitivity and resolution studies", parameters: { ...s, groups: settings.groups }, sources: rows.map(row => row.source), panels,
            notes: ["Each member is scanned on its own native grid. Histograms and means use finite native-cell counts; no unequal-grid cellwise subtraction or remapping is performed.", "The common physical box admits whole cells by their saved center. Boundary cells are not partially clipped; geometry, counts and physical threshold volumes are included in the numerical table.", `Saved times match within the explicit ${s.tolerance} s tolerance; actual member times are retained. No temporal interpolation is used.`, "Nominal cell length is descriptive for stretched meshes. A response trend alone does not establish convergence; uncertainty and experiment controls require scientific assessment.", "Physical volume is unavailable where exact cell volumes or uniform physical spacing are missing. Missing values are not zero. Group names annotate the member traces."],
            tables: [{ name: "member-responses", columns: ["member", "group", "time_s", "nx", "ny", "nz", "nominal_cell_length_m", "finite_cells", "missing_cells", "native_cell_mean", "threshold_count", "threshold_volume_km3", "region_xmin_m", "region_xmax_m", "region_ymin_m", "region_ymax_m", "region_zmin_m", "region_zmax_m"], rows: rows.map(row => { const i = members.findIndex(member => member.source.locator.path === row.source.runDir), g = row.source.step.grid; return [row.label, groups[i] ?? "", row.source.step.simulationTimeS, g?.nr ?? null, g?.nth ?? null, g?.nz ?? null, row.spacing, row.statistics.finite, row.statistics.missing, row.statistics.mean, row.statistics.selected, row.statistics.volumeKm3, row.region.lower[0], row.region.upper[0], row.region.lower[1], row.region.upper[1], row.region.lower[2], row.region.upper[2]]; }) },
                { name: "member-distributions", columns: ["member", "time_s", "bin_center", "native_cell_count", "fraction_of_finite_cells"], rows: rows.flatMap(row => binCenters(bins).map((x, i) => [row.label, row.source.step.simulationTimeS, x, row.statistics.valueCounts[i], row.statistics.finite ? row.statistics.valueCounts[i] / row.statistics.finite : null])) }] };
    }, [task.value, source, members, settings.groups]);
    useResearchResult(result, onResult);
    return <><p>Add datasets with the existing Compare or Ensemble controls. This workflow accepts unequal native grids when their units, saved times and physical coverage are compatible.</p>
    <div className="research-controls"><label>Comparison field<select value={settings.field} onChange={e => update({ field: e.target.value })}>{fields.map(field => <option key={field}>{field}</option>)}</select></label>
    <NumberControl label="Threshold (field units)" value={settings.threshold} onChange={threshold => update({ threshold })}/><NumberControl label="Distribution minimum" value={settings.min} onChange={min => update({ min })}/><NumberControl label="Distribution maximum" value={settings.max} onChange={max => update({ max })}/>
    <NumberControl label="Distribution bin count" value={settings.bins} onChange={bins => update({ bins })} min={2} max={128} step={1}/><NumberControl label="Saved-time tolerance (s)" value={settings.tolerance} onChange={tolerance => update({ tolerance })} min={0}/>
    <label><input type="checkbox" checked={settings.history} onChange={e => update({ history: e.target.checked })}/>Build member histories</label><NumberControl label="Member history frame stride" value={settings.frameStride} onChange={frameStride => update({ frameStride })} min={1} max={1000} step={1}/>
    <label>Group labels in member order (comma separated)<input value={settings.groups} onChange={e => update({ groups: e.target.value })} maxLength={500}/></label>
    <label><input type="checkbox" checked={settings.customRegion} onChange={e => update({ customRegion: e.target.checked })}/>Choose a common physical region</label>
    {settings.customRegion && (["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"] as const).map(key => <NumberControl key={key} label={`${key} (km)`} value={settings[key]} onChange={value => update({ [key]: value })}/>)}</div>
    <TaskControls task={task} label="Compare native-grid responses"/>
  </>;
}
