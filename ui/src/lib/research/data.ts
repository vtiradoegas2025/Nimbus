import {cartesianDyMeters} from "@/lib/science/cartesianCoordinates";
import { axisCoords } from "@/lib/science/gridCoords";
import { gridForWindowedField } from "@/lib/science/windowedGrid";
import { readFieldAny, readFieldAnyWindows, registeredFieldRank, fieldDefinitionIdentity, type StepInfo, type GridInfo, type FieldLoadWindow } from "@/lib/run/runOutputs";
import { planExactValidDomainWindows, planExactLineWindows, planBoundedSliceWindows } from "@/lib/plot/fieldLoadPlanning";
import type { NpyN } from "@/lib/platform/npy";
import { cachedResearchScan, cacheResearchScan } from "./scanCache";
import { ResearchScanClient } from "./scanClient";
import type { HistogramSpec, PhysicalBox, ScanChunk, ScanResult } from "./statistics";
function researchWindows(source: ResearchSource, field: string, windows: readonly FieldLoadWindow[]): FieldLoadWindow[] {
    const logarithmic = source.step.fieldUnits[field]?.toLowerCase() === "dbz" || field === "reflectivity_dbz";
    return windows.map(window => ({ ...window, aggregation: logarithmic ? "linear_reflectivity_mean" : "cell_mean" }));
}
export interface ResearchSource {
    runDir: string;
    step: StepInfo;
}
export function sourceKey(source: ResearchSource, fields: readonly string[]): string {
    return JSON.stringify([source.runDir, source.step.index, source.step.publicationGeneration, source.step.grid,
        fields.map(field => [field, source.step.fieldUnits[field], fieldDefinitionIdentity(source.runDir, source.step.index, field, source.step.publicationGeneration)])]);
}
export function physicalAxes(grid: GridInfo | null): [
    number[],
    number[],
    number[]
] {
    if (!grid)
        throw new Error("Analysis requires physical grid metadata");
    return ([0, 1, 2] as const).map(axis => {
        const count = [grid.nr, grid.nth, grid.nz][axis];
        if (!count || !Number.isSafeInteger(count))
            throw new Error("Analysis requires a complete 3D grid");
        if (axis === 2 && grid.terrainFollowing)
            return Array.from({ length: count }, (_, i) => i);
        const coordinates = axisCoords(axis, count, grid);
        if (coordinates.unit !== "km" || !coordinates.values)
            throw new Error("Analysis requires declared physical coordinates");
        return coordinates.values.map(value => value * 1000);
    }) as [
        number[],
        number[],
        number[]
    ];
}
export function validShape(grid: GridInfo | null): [
    number,
    number,
    number
] {
    const axes = physicalAxes(grid), lower = grid?.validIndexRange?.lower ?? [0, 0, 0], upper = grid?.validIndexRange?.upper ?? axes.map(a => a.length - 1);
    return axes.map((a, i) => {
        if (!Number.isSafeInteger(lower[i]) || !Number.isSafeInteger(upper[i]) || lower[i] < 0 || upper[i] >= a.length || upper[i] < lower[i])
            throw new Error("Invalid valid-domain metadata");
        return upper[i] - lower[i] + 1;
    }) as [
        number,
        number,
        number
    ];
}
export function nativeFields(source: ResearchSource): string[] {
    return source.step.fields.filter(field => registeredFieldRank(source.runDir, source.step.index, field, source.step.publicationGeneration) === 3);
}
function uniformSpacing(axis: readonly number[], declared: number | undefined): number | null {
    if (!(declared && declared > 0))
        return null;
    return axis.every((value, i) => i === 0 || Math.abs(value - axis[i - 1] - declared) <= Math.max(1, declared) * 1e-7) ? declared : null;
}
export function physicalMeasures(grid: GridInfo, axes: [
    number[],
    number[],
    number[]
]) {
    const dx = uniformSpacing(axes[0], grid.drM), dy = uniformSpacing(axes[1], grid.drM == null ? grid.dyM : cartesianDyMeters(grid, grid.drM)), dz = grid.terrainFollowing ? null : uniformSpacing(axes[2], grid.dzM);
    return { area: dx && dy ? dx * dy : null, volume: dx && dy && dz ? dx * dy * dz : null, dx, dy, dz };
}
function assertTile(array: NpyN, window: FieldLoadWindow): void {
    if (array.shape.length !== 3 || array.shape.some((n, i) => n !== window.extent[i]) || array.data.length !== window.extent.reduce((a, b) => a * b, 1)
        || array.windowGeometry?.stride?.some(n => n !== 1))
        throw new Error("Analysis requires exact co-located native cells, not display LOD");
}
export interface AnalysisOptions {
    field: string;
    other?: string;
    mask?: string;
    maskThreshold?: number;
    dryDensity?: string;
    value: HistogramSpec;
    height: HistogramSpec;
    joint?: HistogramSpec;
    threshold: number;
    below?: boolean;
    zIndex?: number;
    region?: PhysicalBox;
    footprint?: {
        x: HistogramSpec;
        y: HistogramSpec;
    };
    massFactor?: number;
}
export async function scanSource(source: ResearchSource, options: AnalysisOptions, signal: AbortSignal, progress: (fraction: number) => void = () => { }): Promise<ScanResult> {
    const grid = source.step.grid, axes = physicalAxes(grid), shape = validShape(grid);
    const fields = [...new Set([options.field, options.other, options.mask, options.dryDensity].filter((field): field is string => !!field))];
    if (signal.aborted)
        throw new DOMException("Analysis cancelled", "AbortError");
    const sourceIdentity = sourceKey(source, fields);
    const cacheKey = JSON.stringify([sourceIdentity, options]);
    const cached = cachedResearchScan(cacheKey);
    if (cached) {
        progress(1);
        return cached;
    }
    for (const field of fields)
        if (!nativeFields(source).includes(field))
            throw new Error(`Required 3D field is unavailable: ${field}`);
    if (options.zIndex != null && (!Number.isSafeInteger(options.zIndex) || options.zIndex < 0 || options.zIndex >= shape[2]))
        throw new Error("Model-level index must be an integer within the saved grid");
    const region = options.zIndex == null ? undefined : { lower: [0, 0, options.zIndex], upper: [shape[0] - 1, shape[1] - 1, options.zIndex] };
    const plan = planExactValidDomainWindows(grid, 3, region, 2 * 1024 * 1024);
    if (!plan.ok)
        throw new Error(plan.reason);
    const measures = physicalMeasures(grid!, axes);
    const client = new ResearchScanClient();
    try {
        await client.request({ kind: "start", spec: { value: options.value, height: options.height, joint: options.joint, threshold: options.threshold, below: options.below ?? false,
                maskThreshold: options.mask ? options.maskThreshold : undefined, reflectivity: source.step.fieldUnits[options.field]?.toLowerCase() === "dbz" || options.field === "reflectivity_dbz", region: options.region, axesM: axes,
                cellAreaM2: options.zIndex != null ? measures.area : null, cellVolumeM3: measures.volume, footprint: options.footprint, massFactor: options.massFactor } }, [], signal);
        for (let w = 0; w < plan.plan.windows.length; w++) {
            const window = plan.plan.windows[w];
            const loaded = new Map<string, NpyN>();
            // One tile per field stays resident; worker transfer releases the UI's copy.
            for (const field of fields) {
                const value = await readFieldAny(source.runDir, source.step.index, field, source.step.publicationGeneration, { signal, window: researchWindows(source, field, [window])[0], priority: "background", cacheRetention: "transient", ownership: "transferable" });
                assertTile(value, window);
                loaded.set(field, value);
            }
            const primary = loaded.get(options.field)!;
            if (grid!.terrainFollowing && (!primary.windowGeometry?.centroidsM || !primary.windowGeometry.volumesM3))
                throw new Error("Terrain analysis requires saved physical centroids and cell volumes");
            for (const array of loaded.values()) {
                const a = primary.windowGeometry?.centroidsM, b = array.windowGeometry?.centroidsM;
                if (Boolean(a) !== Boolean(b))
                    throw new Error("Multi-field physical geometry is inconsistent");
                if (a && b)
                    for (const axis of ["x", "y", "z"] as const)
                        if (a[axis].some((v, i) => Math.abs(v - b[axis][i]) > 1e-7 * Math.max(1, Math.abs(v))))
                            throw new Error("Multi-field centroids do not match");
            }
            const chunk: ScanChunk = { origin: window.origin, shape: primary.shape, values: primary.data, other: options.other ? loaded.get(options.other)!.data : undefined,
                mask: options.mask ? loaded.get(options.mask)!.data : undefined, density: options.dryDensity ? loaded.get(options.dryDensity)!.data : undefined,
                centroids: primary.windowGeometry?.centroidsM, volumes: primary.windowGeometry?.volumesM3 };
            const buffers = new Set<ArrayBuffer>();
            for (const data of [chunk.values, chunk.other, chunk.mask, chunk.density, chunk.centroids?.x, chunk.centroids?.y, chunk.centroids?.z, chunk.volumes])
                if (data)
                    buffers.add(data.buffer as ArrayBuffer);
            await client.request({ kind: "append", chunk }, [...buffers], signal);
            progress((w + 1) / plan.plan.windows.length);
        }
        if (sourceIdentity !== sourceKey(source, fields))
            throw new Error("Source field definitions changed during analysis");
        const reply = await client.request({ kind: "finish" }, [], signal);
        if (!reply.result)
            throw new Error("Research scan returned no statistics");
        cacheResearchScan(cacheKey, reply.result);
        return reply.result;
    }
    finally {
        client.dispose();
    }
}
export interface ResearchColumn {
    xM: number;
    yM: number;
    zM: number[];
    values: Record<string, number[]>;
    source: ResearchSource;
    xIndex: number;
    yIndex: number;
}
export async function readColumn(source: ResearchSource, fields: readonly string[], xIndex: number, yIndex: number, signal: AbortSignal): Promise<ResearchColumn> {
    if (!Number.isSafeInteger(xIndex) || !Number.isSafeInteger(yIndex) || xIndex < 0 || yIndex < 0)
        throw new Error("Column indices must be nonnegative integers");
    const plan = planExactLineWindows(source.step.grid, 2, [xIndex, yIndex, 0], 2 * 1024 * 1024);
    if (!plan.ok)
        throw new Error(plan.reason);
    const values: Record<string, number[]> = {};
    let result: ResearchColumn | null = null;
    for (const field of [...new Set(fields)]) {
        if (!nativeFields(source).includes(field))
            throw new Error(`Required profile field is unavailable: ${field}`);
        const data = await readFieldAnyWindows(source.runDir, source.step.index, field, source.step.publicationGeneration, researchWindows(source, field, plan.plan.windows), { signal });
        const grid = gridForWindowedField(data, source.step.grid!);
        const z = data.windowGeometry?.centroidsM?.z ?? axisCoords(2, data.shape[2], grid).values?.map(v => v * 1000);
        const x = data.windowGeometry?.centroidsM?.x[0] ?? axisCoords(0, 1, grid).values?.[0]! * 1000;
        const y = data.windowGeometry?.centroidsM?.y[0] ?? axisCoords(1, 1, grid).values?.[0]! * 1000;
        if (!z || ![x, y, ...z].every(Number.isFinite) || z.some((v, i) => i > 0 && v <= z[i - 1]))
            throw new Error("Profile requires increasing physical heights");
        if (result && (result.zM.length !== z.length || result.zM.some((v, i) => Math.abs(v - z[i]) > 1e-6)))
            throw new Error("Profile fields have different physical heights");
        result ??= { xM: x, yM: y, zM: Array.from(z), values, source, xIndex, yIndex };
        values[field] = Array.from(data.data);
    }
    if (!result)
        throw new Error("Choose at least one profile field");
    return result;
}
export async function readPlane(source: ResearchSource, field: string, axis: 0 | 1 | 2, index: number, signal: AbortSignal): Promise<{
    field: NpyN;
    grid: GridInfo;
}> {
    const plan = planBoundedSliceWindows(source.step.grid, axis, index, 262144);
    if (!plan.ok)
        throw new Error(plan.reason);
    const fieldData = await readFieldAnyWindows(source.runDir, source.step.index, field, source.step.publicationGeneration, researchWindows(source, field, plan.plan.windows), { signal });
    return { field: fieldData, grid: gridForWindowedField(fieldData, source.step.grid!)! };
}
export async function reduceResearchSurface(source: ResearchSource, field: string, signal: AbortSignal) {
    if (registeredFieldRank(source.runDir, source.step.index, field, source.step.publicationGeneration) !== 2)
        throw new Error("Precipitation history requires a saved two-dimensional surface product");
    const plan = planExactValidDomainWindows(source.step.grid, 2, undefined, 2 * 1024 * 1024);
    if (!plan.ok)
        throw new Error(plan.reason);
    const client = new ResearchScanClient(), definition = fieldDefinitionIdentity(source.runDir, source.step.index, field, source.step.publicationGeneration);
    try {
        await client.request({ kind: "surface-start" }, [], signal);
        for (const window of plan.plan.windows) {
            if (definition !== fieldDefinitionIdentity(source.runDir, source.step.index, field, source.step.publicationGeneration))
                throw new Error("Precipitation field definition changed during analysis");
            const data = await readFieldAny(source.runDir, source.step.index, field, source.step.publicationGeneration, { window, signal, ownership: "transferable", cacheRetention: "transient", priority: "background" });
            if (data.shape.length !== 2 || data.shape.some((n, i) => n !== window.extent[i]) || data.windowGeometry?.stride?.some(stride => stride !== 1))
                throw new Error("Surface reduction requires exact native cells");
            await client.request({ kind: "surface-append", values: data.data }, [data.data.buffer as ArrayBuffer], signal);
        }
        const reply = await client.request({ kind: "surface-finish" }, [], signal);
        if (!reply.surface)
            throw new Error("Surface reduction returned no statistics");
        return reply.surface;
    }
    finally {
        client.dispose();
    }
}
