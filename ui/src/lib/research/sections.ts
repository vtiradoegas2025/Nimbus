import { readColumn, physicalAxes, validShape, sourceKey, type ResearchColumn, type ResearchSource } from "./data";
import { sampleFiniteProfile } from "./profiles";
import { mapPool } from "@/lib/core/concurrency";
export interface SectionSettings {
    startX: number;
    startY: number;
    endX: number;
    endY: number;
    samples: number;
}
export interface SectionData {
    distanceKm: number[];
    heightKm: number[];
    values: Record<string, (number | null)[][]>;
    points: {
        x: number;
        y: number;
    }[];
    nativeVertical: boolean;
    columnsRead: number;
}
function bracket(axis: readonly number[], value: number): [
    number,
    number,
    number
] {
    if (value < axis[0] || value > axis[axis.length - 1])
        throw new Error("Section endpoint is outside saved scalar-center coverage");
    let low = 0, high = axis.length - 1;
    while (high - low > 1) {
        const mid = Math.floor((low + high) / 2);
        if (axis[mid] <= value)
            low = mid;
        else
            high = mid;
    }
    if (value === axis[low] || low === high)
        return [low, low, 0];
    if (value === axis[high])
        return [high, high, 0];
    return [low, high, (value - axis[low]) / (axis[high] - axis[low])];
}
export async function readSection(source: ResearchSource, fields: readonly string[], settings: SectionSettings, signal: AbortSignal, progress: (text: string) => void): Promise<SectionData> {
    const definition = sourceKey(source, fields);
    const axes = physicalAxes(source.step.grid), shape = validShape(source.step.grid), origin = source.step.grid?.validIndexRange?.lower ?? [0, 0, 0];
    const x = axes[0].slice(origin[0], origin[0] + shape[0]), y = axes[1].slice(origin[1], origin[1] + shape[1]);
    if (shape[2] * settings.samples * fields.length > 2000000)
        throw new Error(`Section exceeds two million displayed values; choose at most ${Math.floor(2000000 / (shape[2] * Math.max(1, fields.length)))} horizontal samples or fewer fields. Native vertical levels are retained.`);
    if (!Number.isInteger(settings.samples) || settings.samples < 2 || settings.samples > 512)
        throw new Error("Sections require 2–512 horizontal samples");
    if (![settings.startX, settings.startY, settings.endX, settings.endY].every(Number.isFinite))
        throw new Error("Section coordinates must be finite");
    const length = Math.hypot(settings.endX - settings.startX, settings.endY - settings.startY);
    if (length === 0)
        throw new Error("Section endpoints must differ");
    const points = Array.from({ length: settings.samples }, (_, i) => ({ x: settings.startX + (settings.endX - settings.startX) * i / (settings.samples - 1), y: settings.startY + (settings.endY - settings.startY) * i / (settings.samples - 1) }));
    const unique = new Map<string, [
        number,
        number
    ]>();
    const samples = points.map(point => {
        const [x0, x1, wx] = bracket(x, point.x * 1000), [y0, y1, wy] = bracket(y, point.y * 1000);
        const entries: [
            [
                number,
                number
            ],
            number
        ][] = [[[x0, y0], (1 - wx) * (1 - wy)], [[x0, y1], (1 - wx) * wy], [[x1, y0], wx * (1 - wy)], [[x1, y1], wx * wy]];
        return entries.filter(([, weight]) => weight > 1e-12).map(([point, weight]) => { const key = point.join(","); unique.set(key, point); return { key, weight }; });
    });
    const columns = new Map<string, ResearchColumn>(), entries = [...unique.entries()];
    let done = 0, error: unknown = null;
    await mapPool(entries.length, 3, async (i) => { const [key, [x, y]] = entries[i]; return { key, column: await readColumn(source, fields, x, y, signal) }; }, (_index, result) => {
        if (!result.ok) {
            error = result.error;
            return false;
        }
        columns.set(result.value.key, result.value.column);
        progress(`Reading section columns ${++done}/${entries.length}`);
        return !signal.aborted;
    });
    if (signal.aborted)
        throw new DOMException("Section cancelled", "AbortError");
    if (error)
        throw error;
    let heights = columns.values().next().value!.zM as number[];
    const nativeVertical = !source.step.grid?.terrainFollowing;
    if (!nativeVertical) {
        const low = Math.min(...[...columns.values()].map(c => c.zM[0])), high = Math.max(...[...columns.values()].map(c => c.zM[c.zM.length - 1]));
        if (high <= low)
            throw new Error("Section columns have no common physical-height coverage");
        heights = Array.from({ length: shape[2] }, (_, i) => low + (high - low) * i / Math.max(1, shape[2] - 1));
    }
    const values: SectionData["values"] = {};
    for (const field of fields) {
        const matrix: (number | null)[][] = [];
        values[field] = matrix;
        for (let k = 0; k < heights.length; k++) {
            if (k % 32 === 0) {
                if (signal.aborted)
                    throw new DOMException("Section cancelled", "AbortError");
                await new Promise<void>(resolve => setTimeout(resolve, 0));
            }
            const height = heights[k];
            matrix.push(samples.map(sample => {
                let value = 0;
                for (const { key, weight } of sample) {
                    const column = columns.get(key)!;
                    const v = nativeVertical ? column.values[field][k] : sampleFiniteProfile(column.zM, column.values[field], height);
                    if (v == null || !Number.isFinite(v))
                        return null;
                    value += weight * v;
                }
                return value;
            }));
        }
    }
    if (definition !== sourceKey(source, fields))
        throw new Error("Section source definitions changed during analysis");
    return { distanceKm: points.map((_, i) => length * i / (settings.samples - 1)), heightKm: heights.map(z => z / 1000), values, points, nativeVertical, columnsRead: columns.size };
}
