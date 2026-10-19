import type { Layout } from "plotly.js";
export type ResearchViews = Record<string, Partial<Layout>>;
export function sanitizeResearchViews(value: unknown): ResearchViews {
    if (value == null)
        return {};
    if (typeof value !== "object" || Array.isArray(value) || Object.keys(value).length > 24)
        throw new Error("Figure viewport metadata is invalid");
    const result: ResearchViews = {};
    for (const [id, raw] of Object.entries(value)) {
        if (!id || id.length > 100 || id === "__proto__" || !raw || typeof raw !== "object")
            throw new Error("Figure viewport metadata is invalid");
        const source = raw as Record<string, unknown>, layout: Partial<Layout> = {};
        for (const axis of ["xaxis", "yaxis"] as const) {
            const item = source[axis] as {
                range?: unknown;
                autorange?: unknown;
            } | undefined;
            if (!item)
                continue;
            if (Array.isArray(item.range) && item.range.length === 2 && item.range.every(v => typeof v === "number" && Number.isFinite(v)) && item.range[0] !== item.range[1])
                layout[axis] = { range: item.range };
            else if (item.autorange === true)
                layout[axis] = { autorange: true };
            else
                throw new Error("Figure axis range is invalid");
        }
        const scene = source.scene as {
            camera?: Record<string, unknown>;
        } | undefined;
        if (scene?.camera) {
            const camera: Record<string, unknown> = {};
            for (const key of ["eye", "up", "center"]) {
                const v = scene.camera[key] as Record<string, unknown> | undefined;
                if (v) {
                    if (!["x", "y", "z"].every(axis => typeof v[axis] === "number" && Number.isFinite(v[axis])))
                        throw new Error("Figure camera is invalid");
                    camera[key] = { x: v.x, y: v.y, z: v.z };
                }
            }
            const projection = scene.camera.projection as {
                type?: string;
            } | undefined;
            if (projection) {
                if (!["perspective", "orthographic"].includes(projection.type ?? ""))
                    throw new Error("Figure projection is invalid");
                camera.projection = { type: projection.type };
            }
            layout.scene = { camera };
        }
        result[id] = layout;
    }
    return result;
}
