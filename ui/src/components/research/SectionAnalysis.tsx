import { useMemo } from "react";
import type { Data } from "plotly.js";
import { nativeFields, physicalAxes, readPlane, sourceKey, validShape } from "@/lib/research/data";
import { readSection } from "@/lib/research/sections";
import { slice2D, type Npy3D } from "@/lib/platform/npy";
import { axisCoords } from "@/lib/science/gridCoords";
import { DBZ_RANGE, resolveColorscale } from "@/lib/plot/plotColormaps";
import { NumberControl, scientificLayout, TaskControls, useResearchTask, useResearchResult, useResearchSettings, RESEARCH_COLORS, type ResearchProps, type ResearchResult } from "./common";
const PRESSURE_FIELDS = ["p_prime", "dynamic_pressure", "buoyancy_pressure", "buoyancy", "pressure_gradient_force_z", "w"];
export function SectionAnalysis({ source, steps: _, onResult, pressure = false }: {
    pressure?: boolean;
} & ResearchProps) {
    const fields = useMemo(() => nativeFields(source), [source]);
    const coordinates = useMemo(() => { try {
        const axes = physicalAxes(source.step.grid), shape = validShape(source.step.grid), lo = source.step.grid?.validIndexRange?.lower ?? [0, 0, 0];
        return { axes: axes.map((a, i) => a.slice(lo[i], lo[i] + shape[i])), shape };
    }
    catch {
        return { axes: [[0, 1000], [0, 1000], [0, 1000]], shape: [2, 2, 2] };
    } }, [source.step.grid]);
    const pressureFields = useMemo(() => PRESSURE_FIELDS.filter(field => fields.includes(field)), [fields]);
    const [settings, update] = useResearchSettings({ field: pressure ? pressureFields[0] ?? fields[0] ?? "" : fields.includes("w") ? "w" : fields[0] ?? "", contour: pressure ? pressureFields[1] ?? "" : "", extra: pressure ? pressureFields[2] ?? "" : "", vectors: false, allPressure: false,
        startX: coordinates.axes[0][0] / 1000, startY: coordinates.axes[1][0] / 1000, endX: coordinates.axes[0].at(-1)! / 1000, endY: coordinates.axes[1].at(-1)! / 1000,
        samples: Math.min(128, Math.max(2, coordinates.shape[0])), level: 0, probe: 0.5, contourMin: -1, contourMax: 1, contourStep: .2, rangeMin: -10, rangeMax: 10, sharedRange: false, context3d: false });
    const selected = [...new Set([settings.field, settings.contour, settings.extra, ...(settings.vectors ? ["u", "v", "w"] : []), ...(pressure && settings.allPressure ? pressureFields : [])].filter(Boolean))];
    const key = JSON.stringify([sourceKey(source, selected), { ...settings, probe: undefined }]);
    const task = useResearchTask(key, async (signal, progress) => {
        if (settings.sharedRange && settings.rangeMax <= settings.rangeMin)
            throw new Error("Shared color bounds must increase");
        if (settings.contour && (settings.contourMax <= settings.contourMin || settings.contourStep <= 0 || (settings.contourMax - settings.contourMin) / settings.contourStep > 100))
            throw new Error("Contour bounds require a positive interval and at most 100 levels");
        if (settings.sharedRange && [settings.field, settings.contour, settings.extra, ...(pressure && settings.allPressure ? pressureFields : [])].filter(Boolean).some(field => source.step.fieldUnits[field] !== source.step.fieldUnits[settings.field]))
            throw new Error("Shared color ranges require matching field units");
        if (settings.vectors && !["u", "v", "w"].every(field => /^(m\/s|m s-1|m s\^-1|m s\*\*-1)$/.test(source.step.fieldUnits[field] ?? "")))
            throw new Error("Section vectors require saved u/v/w in declared m/s units");
        const section = await readSection(source, selected, settings, signal, progress);
        const plan = await readPlane(source, settings.field, 2, Math.min(settings.level, coordinates.shape[2] - 1), signal);
        return { section, plan, settings };
    });
    const result = useMemo<ResearchResult | null>(() => {
        if (!task.value)
            return null;
        const { section, plan } = task.value;
        const s = { ...task.value.settings, probe: settings.probe };
        const probeIndex = Math.round(s.probe * (section.points.length - 1));
        const profile = { xM: section.points[probeIndex].x * 1000, yM: section.points[probeIndex].y * 1000, zM: section.heightKm.map(z => z * 1000), values: Object.fromEntries(Object.entries(section.values).map(([field, matrix]) => [field, matrix.map(row => row[probeIndex])])) };
        const isDbz = (field: string) => source.step.fieldUnits[field]?.toLowerCase() === "dbz" || field === "reflectivity_dbz";
        const signed = (field: string) => ["u", "v", "w", "theta_prime", "p_prime", "dynamic_pressure", "buoyancy_pressure", "buoyancy", "pressure_gradient_force_z"].includes(field);
        const color = (field: string) => resolveColorscale(isDbz(field) ? "dBZ" : signed(field) ? "RdBu" : "Viridis");
        const range = (field: string) => {
            if (isDbz(field))
                return { zmin: DBZ_RANGE[0], zmax: DBZ_RANGE[1] };
            if (s.sharedRange)
                return { zmin: s.rangeMin, zmax: s.rangeMax };
            if (!signed(field))
                return {};
            const amplitude = section.values[field].reduce((max, row) => row.reduce<number>((m, value) => Math.max(m, Math.abs(value ?? 0)), max), 1e-12);
            return { zmin: -amplitude, zmax: amplitude };
        };
        const unit = (field: string) => `${field} (${source.step.fieldUnits[field] ?? "native units"})`;
        const plane = slice2D(plan.field as Npy3D, 2, 0), x = axisCoords(0, plan.field.shape[0], plan.grid), y = axisCoords(1, plan.field.shape[1], plan.grid);
        const probe = section.points[Math.round(s.probe * (section.points.length - 1))];
        const selectedFields = [...new Set([s.field, s.contour, s.extra, ...(pressure && s.allPressure ? pressureFields : [])].filter(Boolean))];
        const panels: ResearchResult["panels"] = [{ id: "locator", title: "Plan locator", data: [{ type: "heatmap", x: x.values, y: y.values, z: Array.from({ length: plane.cols }, (_, j) => plane.data.map(row => row[j])), colorscale: color(s.field),  ...range(s.field), zsmooth: false, colorbar: { title: { text: unit(s.field) } } },
                    { type: "scatter", mode: "text+markers", x: [s.startX, s.endX], y: [s.startY, s.endY], text: ["A", "B"], line: { color: "#ffffff", width: 2 }, name: "Section A → B" },
                    { type: "scatter", mode: "markers", x: [probe.x], y: [probe.y], name: "Shared probe", marker: { size: 10, color: "#f3be58" } }], config: { editable: true, edits: { shapePosition: true } }, onRelayout: event => { const changes: Record<string, number> = {}; for (const [suffix, key] of [["x0", "startX"], ["y0", "startY"], ["x1", "endX"], ["y1", "endY"]]) {
                    const value = event[`shapes[0].${suffix}`];
                    if (typeof value === "number" && Number.isFinite(value))
                        changes[key] = value;
                } if (Object.keys(changes).length)
                    update(changes); }, layout: { ...scientificLayout("Plan locator", "x (km)", "y (km)"), shapes: [{ type: "line", x0: s.startX, y0: s.startY, x1: s.endX, y1: s.endY, line: { color: "#ffce75", width: 3 } }], yaxis: { title: { text: "y (km)" }, scaleanchor: "x", scaleratio: 1 } }, onClick: event => { const point = (event.points as {
                    x?: number;
                    y?: number;
                }[] | undefined)?.[0]; if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
                    const dx = s.endX - s.startX, dy = s.endY - s.startY;
                    update({ probe: Math.max(0, Math.min(1, ((point.x! - s.startX) * dx + (point.y! - s.startY) * dy) / (dx * dx + dy * dy))) });
                } } }];
        for (const [index, field] of selectedFields.entries()) {
            const data: Data[] = [{ type: "heatmap", x: section.distanceKm, y: section.heightKm, z: section.values[field], colorscale: color(field),  ...range(field), zsmooth: false, colorbar: { title: { text: unit(field) } } }];
            if (index === 0 && s.contour)
                data.push({ type: "contour", x: section.distanceKm, y: section.heightKm, z: section.values[s.contour], contours: { start: s.contourMin, end: s.contourMax, size: s.contourStep, coloring: "none", showlabels: true }, line: { color: "#f0f3f6", width: 1 }, showscale: false, connectgaps: false, name: unit(s.contour) });
            if (index === 0 && s.vectors) {
                const dx = s.endX - s.startX, dy = s.endY - s.startY, l = Math.hypot(dx, dy), xs: (number | null)[] = [], zs: (number | null)[] = [];
                const strideX = Math.max(1, Math.ceil(section.points.length / 24)), strideZ = Math.max(1, Math.ceil(section.heightKm.length / 18));
                const scale = .002;
                for (let k = 0; k < section.heightKm.length; k += strideZ)
                    for (let j = 0; j < section.points.length; j += strideX) {
                        const u = section.values.u[k][j], v = section.values.v[k][j], w = section.values.w[k][j];
                        if (u == null || v == null || w == null)
                            continue;
                        const a = (u * dx + v * dy) / l, b = w, x = section.distanceKm[j], z = section.heightKm[k];
                        const ex = x + a * scale, ez = z + b * scale;
                        xs.push(x, ex, null, ex - .25 * a * scale - .12 * b * scale, ex, ex - .25 * a * scale + .12 * b * scale, null);
                        zs.push(z, ez, null, ez - .25 * b * scale + .12 * a * scale, ez, ez - .25 * b * scale - .12 * a * scale, null);
                    }
                data.push({ type: "scatter", mode: "lines", x: xs, y: zs, line: { color: "#f0c86c", width: 1.5 }, name: "Projected (u,v), w · 2 s displacement" });
            }
            panels.push({ id: `section-${index}`, title: unit(field), data, layout: { ...scientificLayout(unit(field), "Distance A → B (km)", "Saved height (km)"), shapes: [{ type: "line", x0: s.probe * section.distanceKm.at(-1)!, x1: s.probe * section.distanceKm.at(-1)!, y0: section.heightKm[0], y1: section.heightKm.at(-1)!, line: { color: "#f3be58", dash: "dot" } }] }, onClick: event => { const point = (event.points as {
                    x?: number;
                }[] | undefined)?.[0]; if (point && Number.isFinite(point.x))
                    update({ probe: Math.max(0, Math.min(1, point.x! / section.distanceKm.at(-1)!)) }); } });
        }
        const profileData: Data[] = selectedFields.map((field, i) => ({ type: "scatter", mode: "lines", x: profile.values[field], y: profile.zM.map(z => z / 1000), name: unit(field), connectgaps: false, line: { color: RESEARCH_COLORS[i] } }));
        if (new Set(selectedFields.map(field => source.step.fieldUnits[field])).size === 1)
            panels.push({ id: "profiles", title: "Linked section profiles", data: profileData, layout: scientificLayout("Linked section profiles", source.step.fieldUnits[s.field] ?? "native units", "Saved height (km)") });
        else
            for (const [i, field] of selectedFields.entries())
                panels.push({ id: `profile-${i}`, title: `Profile ${field}`, data: [profileData[i]], layout: scientificLayout(`Profile ${field}`, unit(field), "Saved height (km)") });
        if (s.context3d)
            panels.push({ id: "context", title: "Section location in 3D", data: [{ type: "scatter3d", mode: "lines", x: [s.startX, s.endX, s.endX, s.startX, s.startX], y: [s.startY, s.endY, s.endY, s.startY, s.startY], z: [section.heightKm[0], section.heightKm[0], section.heightKm.at(-1)!, section.heightKm.at(-1)!, section.heightKm[0]], name: "Section extent" }], layout: { title: { text: "Section location in 3D" }, scene: { xaxis: { title: { text: "x (km)" } }, yaxis: { title: { text: "y (km)" } }, zaxis: { title: { text: "z (km)" } }, aspectmode: "data" } } });
        const notes = ["All section fields share saved time, endpoints and probe. Drag the plan locator line or edit A/B coordinates, then load the section. Click a plan/section to set the shared probe, to update the linked profiles.", "Sections interpolate horizontally between saved cell centers; missing contributing cells remain gaps.", section.nativeVertical ? "Vertical coordinates retain native scalar levels." : "Terrain columns are displayed on a shared physical-height grid by finite-segment interpolation; this adds no source resolution.", `Profiles interpolate the section at x=${profile.xM / 1000}, y=${profile.yM / 1000} km. ${section.columnsRead} source columns read; moving the probe reuses these values.`, "Projected vectors use the saved wind frame; 2 s arrow displacement is a display scale."];
        notes.push("Reflectivity uses fixed −30 to 75 dBZ palette anchors. Signed fields use symmetric colors unless a shared range is selected; color clipping does not alter exported values.");
        notes.push(`Plan display uses source stride ${(plan.field.windowGeometry?.stride ?? [1, 1, 1]).join(" × ")}; native-column sections and profiles retain their separate sampling contracts.`);
        if (pressure)
            notes.push("p′ is the saved perturbation relative to its model reference. Dynamic/buoyancy pressure potentials use their native gauge (Nimbus: per-level zero-volume-mean). Their sum is not asserted to equal compressible p′. Saved vertical PGF differentiates full pressure; gravity remains separate.");
        return { title: pressure ? "Pressure and vertical-motion analysis" : "Linked multi-field sections", parameters: s, sources: [source], panels, tables: selectedFields.map(field => ({ name: `section-${field}`, columns: ["distance_km", "height_km", unit(field)], rows: [], rowCount: section.heightKm.length * section.distanceKm.length, rowAt: (index: number) => { const k = Math.floor(index / section.distanceKm.length), j = index % section.distanceKm.length; return [section.distanceKm[j], section.heightKm[k], section.values[field][k][j]]; } })), notes };
    }, [task.value, source, settings.probe, pressure, pressureFields, update]);
    useResearchResult(result, onResult);
    return <><div className="research-controls">{["field", "contour", "extra"].map((key, i) => <label key={key}>{["Filled field", "Contour / second field", "Third field"][i]}<select value={settings[key as "field" | "contour" | "extra"]} onChange={e => update({ [key]: e.target.value })}>{i > 0 && <option value="">Off</option>}{fields.map(field => <option key={field}>{field}</option>)}</select></label>)}
    {pressure && <label><input type="checkbox" checked={settings.allPressure} onChange={e => update({ allPressure: e.target.checked })}/>Include every available pressure / buoyancy / PGF / w diagnostic</label>}
    <label><input type="checkbox" checked={settings.vectors} onChange={e => update({ vectors: e.target.checked })}/>Projected section vectors</label>
    {(["startX", "startY", "endX", "endY"] as const).map((key, i) => <NumberControl key={key} label={["A x (km)", "A y (km)", "B x (km)", "B y (km)"][i]} value={settings[key]} onChange={v => update({ [key]: v })}/>)}
    <button type="button" className="btn" disabled={coordinates.shape[0]<2} title={coordinates.shape[0]<2?"X sections require at least two saved x columns":"Use the full saved x extent at A’s y coordinate"} onClick={() => update({startX:coordinates.axes[0][0]/1000,endX:coordinates.axes[0].at(-1)!/1000,endY:settings.startY})}>X section</button><button type="button" className="btn" disabled={coordinates.shape[1]<2} title={coordinates.shape[1]<2?"Y sections require at least two saved y columns":"Use the full saved y extent at A’s x coordinate"} onClick={() => update({startY:coordinates.axes[1][0]/1000,endY:coordinates.axes[1].at(-1)!/1000,endX:settings.startX})}>Y section</button>
    <NumberControl label="Horizontal samples" value={settings.samples} onChange={samples => update({ samples })} min={2} max={512} step={1}/>
    <NumberControl label="Plan level index" value={settings.level} onChange={level => update({ level })} min={0} max={coordinates.shape[2] - 1} step={1}/>
    <NumberControl label="Shared probe fraction" value={settings.probe} onChange={probe => update({ probe })} min={0} max={1}/>
    {(["contourMin", "contourMax", "contourStep"] as const).map((key, i) => <NumberControl key={key} label={["Contour minimum", "Contour maximum", "Contour interval"][i]} value={settings[key]} onChange={v => update({ [key]: v })}/>)}
    <label><input type="checkbox" checked={settings.sharedRange} onChange={e => update({ sharedRange: e.target.checked })}/>Shared color range</label>
    <NumberControl label="Shared color minimum" value={settings.rangeMin} onChange={rangeMin => update({ rangeMin })}/><NumberControl label="Shared color maximum" value={settings.rangeMax} onChange={rangeMax => update({ rangeMax })}/>
    <label><input type="checkbox" checked={settings.context3d} onChange={e => update({ context3d: e.target.checked })}/>3D section context</label></div><TaskControls task={task} label={pressure ? "Load pressure panels" : "Load linked sections"}/></>;
}
