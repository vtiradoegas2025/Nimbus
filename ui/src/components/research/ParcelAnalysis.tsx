import { useEffect, useMemo, useRef } from "react";
import { readTrajectories } from "@/lib/run/runOutputs";
import { ParcelClient } from "@/lib/research/parcelClient";
import { NumberControl, scientificLayout, TaskControls, useResearchResult, useResearchSettings, useResearchTask, type ResearchProps, type ResearchResult } from "./common";
export function ParcelAnalysis({ source, steps, onResult }: ResearchProps) {
    const [settings, update] = useResearchSettings({ originMin: 0, originMax: 20, endpointMin: 0, endpointMax: 20, timeMin: 0, timeMax: 86400, selected: 0 });
    const cached = useRef<{
        key: string;
        client: ParcelClient;
    } | null>(null);
    useEffect(() => () => { cached.current?.client.dispose(); cached.current = null; }, []);
    const generation = JSON.stringify([source.runDir, steps[0]?.publicationGeneration, steps.length, steps.at(-1)?.index]);
    const key = JSON.stringify([generation, settings]);
    const task = useResearchTask(key, async (signal, progress) => {
        if (cached.current?.key !== generation) {
            cached.current?.client.dispose();
            cached.current = null;
            progress("Reading saved parcel history…");
            const data = await readTrajectories(source.runDir, steps[0]?.publicationGeneration ?? null);
            if (!data)
                throw new Error("No compatible saved trajectories are available for this publication");
            if (signal.aborted)
                throw new DOMException("Analysis cancelled", "AbortError");
            if (data.data.byteLength > 64 * 1024 * 1024)
                throw new Error("Interactive parcel analysis retains at most 64 MiB of saved values; use a bounded parcel export for larger histories");
            const client = new ParcelClient();
            try {
                await client.load(data.data, data.shape, signal);
                cached.current = { key: generation, client };
            }
            catch (error) {
                client.dispose();
                throw error;
            }
        }
        const owned = cached.current;
        try {
            return { result: await owned.client.analyze(settings, signal), settings };
        }
        catch (error) {
            if (cached.current === owned)
                cached.current = null;
            owned.client.dispose();
            throw error;
        }
    });
    const result = useMemo<ResearchResult | null>(() => {
        if (!task.value)
            return null;
        const { result: r, settings: s } = task.value;
        const select = (event: Readonly<Record<string, unknown>>) => { const point = (event.points as {
            customdata?: number;
        }[] | undefined)?.[0]; if (point && Number.isInteger(point.customdata))
            update({ selected: point.customdata! }); };
        const panels: ResearchResult["panels"] = [{ id: "paths", title: "Parcel cohort · click a path to select", data: r.paths.map(path => ({ type: "scatter3d", mode: "lines", x: path.x, y: path.y, z: path.z, customdata: path.x.map(() => path.id), name: `Parcel ${path.id}`, line: { width: path.id === r.selected ? 7 : 2, color: path.id === r.selected ? "#ffd078" : undefined }, connectgaps: false })), layout: { title: { text: "Parcel cohort · click a path to select" }, scene: { xaxis: { title: { text: "x (km)" } }, yaxis: { title: { text: "y (km)" } }, zaxis: { title: { text: "Saved height (km)" } }, aspectmode: "data" }, showlegend: r.paths.length < 12 }, onClick: select },
            { id: "origins", title: "Cohort origin heights", data: [{ type: "histogram", x: r.cohort.map(p => p.originHeight), name: "All selected parcels" }], layout: scientificLayout("Cohort first-saved heights", "First saved height (km)", "Parcel count") },
            { id: "height", title: `Parcel ${r.selected} height history`, data: [{ type: "scatter", mode: "lines+markers", x: r.times, y: r.heights, name: "Height", connectgaps: false }], layout: scientificLayout("Selected parcel height", "Saved parcel time (s)", "Height (km)"), onRelayout: event => { const lo = event["xaxis.range[0]"], hi = event["xaxis.range[1]"]; if (typeof lo === "number" && typeof hi === "number" && hi > lo)
                    update({ timeMin: lo, timeMax: hi }); } },
            { id: "vertical-velocity", title: "Parcel vertical velocity", data: [{ type: "scatter", mode: "lines", x: r.times, y: r.w, name: "w", connectgaps: false }], layout: scientificLayout("Selected parcel w", "Saved parcel time (s)", "w (m/s)") },
            { id: "vorticity", title: "Parcel vertical vorticity", data: [{ type: "scatter", mode: "lines", x: r.times, y: r.zeta, name: "ζ", connectgaps: false }], layout: scientificLayout("Selected parcel ζ", "Saved parcel time (s)", "ζ (s⁻¹)") },
            { id: "attribution", title: "Saved-term attribution", data: [{ type: "scatter", mode: "lines", x: r.times, y: r.change, name: "Observed ζ change within finite segment", connectgaps: false }, { type: "scatter", mode: "lines", x: r.times, y: r.integrated, name: "Integral of three saved terms", connectgaps: false }], layout: scientificLayout("Limited saved-term attribution", "Saved parcel time (s)", "Vorticity change (s⁻¹)") }];
        return { title: "Parcel cohorts and saved tendencies", parameters: s, sources: [source], panels,
            notes: [`${r.cohort.length} parcels meet the first/last saved height filters. Origin distributions use every selected parcel; the 3D overview shows every ${r.pathParcelStride}th parcel and at most every ${r.pathStride}th saved sample.`,
                "Click a rendered path or enter its ID, then run analysis to update the linked histories. Zoom the height-history time axis to set the next interval.", "Origin means the first finite saved position, not inferred air-mass provenance. Endpoint means the last finite saved position.", "Tilting, stretching and baroclinic terms are trapezoid-integrated at the saved parcel cadence. Gaps and non-increasing times break the integral; a new finite segment starts from zero.", "The three saved terms do not establish a complete vorticity budget or closure. No unsaved force, mixing term or sub-output-cadence behavior is reconstructed."],
            tables: [{ name: "cohort", columns: ["parcel_id", "first_saved_height_km", "last_saved_height_km", "first_time_s", "last_time_s", "finite_position_samples"], rows: r.cohort.map(p => [p.id, p.originHeight, p.endpointHeight, p.firstTime, p.lastTime, p.samples]) },
                { name: "selected-parcel", columns: ["time_s", "height_km", "w_m_s", "zeta_s-1", "tilting_s-2", "stretching_s-2", "baroclinic_s-2", "observed_segment_delta_zeta", "integral_three_terms"], rows: r.times.map((t, i) => [t, r.heights[i], r.w[i], r.zeta[i], r.terms[0][i], r.terms[1][i], r.terms[2][i], r.change[i], r.integrated[i]]) }] };
    }, [task.value, source, update]);
    useResearchResult(result, onResult);
    return <><div className="research-controls">{(["originMin", "originMax", "endpointMin", "endpointMax", "timeMin", "timeMax"] as const).map((key, i) => <NumberControl key={key} label={["Origin minimum (km)", "Origin maximum (km)", "Endpoint minimum (km)", "Endpoint maximum (km)", "Parcel interval start (s)", "Parcel interval end (s)"][i]} value={settings[key]} onChange={value => update({ [key]: value })}/>)}<NumberControl label="Selected parcel ID" value={settings.selected} onChange={selected => update({ selected })} min={0} step={1}/></div><TaskControls task={task} label="Analyze parcel cohort"/></>;
}
