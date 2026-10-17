export interface ParcelFilter {
    originMin: number;
    originMax: number;
    endpointMin: number;
    endpointMax: number;
    timeMin: number;
    timeMax: number;
    selected: number;
}
export interface ParcelSummary {
    id: number;
    originHeight: number;
    endpointHeight: number;
    firstTime: number;
    lastTime: number;
    samples: number;
}
export interface ParcelAnalysis {
    cohort: ParcelSummary[];
    paths: {
        id: number;
        x: (number | null)[];
        y: (number | null)[];
        z: (number | null)[];
    }[];
    selected: number;
    times: (number | null)[];
    heights: (number | null)[];
    w: (number | null)[];
    zeta: (number | null)[];
    integrated: (number | null)[];
    change: (number | null)[];
    terms: (number | null)[][];
    pathStride: number;
    pathParcelStride: number;
}
export function analyzeParcels(values: Float32Array, shape: readonly number[], filter: ParcelFilter): ParcelAnalysis {
    if (shape.length !== 3 || shape[2] < 6 || shape.some(n => !Number.isSafeInteger(n) || n < 1) || shape.reduce((a, b) => a * b, 1) !== values.length)
        throw new Error("Saved trajectories require parcel × sample × at least six variables");
    if (!Object.values(filter).every(Number.isFinite) || filter.originMax < filter.originMin || filter.endpointMax < filter.endpointMin || filter.timeMax < filter.timeMin)
        throw new Error("Parcel filter bounds are invalid");
    const [parcels, samples, vars] = shape, cohort: ParcelSummary[] = [];
    const valid = (base: number) => [0, 1, 2, 3].every(c => Number.isFinite(values[base + c]));
    for (let p = 0; p < parcels; p++) {
        let first = -1, last = -1, count = 0;
        for (let s = 0; s < samples; s++) {
            const b = (p * samples + s) * vars;
            if (valid(b)) {
                if (first < 0)
                    first = b;
                last = b;
                count++;
            }
        }
        if (first < 0)
            continue;
        const z0 = values[first + 3] / 1000, z1 = values[last + 3] / 1000;
        if (z0 < filter.originMin || z0 > filter.originMax || z1 < filter.endpointMin || z1 > filter.endpointMax)
            continue;
        cohort.push({ id: p, originHeight: z0, endpointHeight: z1, firstTime: values[first], lastTime: values[last], samples: count });
    }
    if (cohort.length > 100000)
        throw new Error("The selected cohort exceeds 100,000 parcels; narrow the origin or endpoint filters");
    const selected = cohort.some(p => p.id === filter.selected) ? filter.selected : cohort[0]?.id ?? -1;
    const result: ParcelAnalysis = { cohort, paths: [], selected, times: [], heights: [], w: [], zeta: [], integrated: [], change: [], terms: [[], [], []], pathStride: 1, pathParcelStride: Math.max(1, Math.ceil(cohort.length / 100)) };
    const shown = Math.ceil(cohort.length / result.pathParcelStride);
    result.pathStride = Math.max(1, Math.ceil(shown * samples / 50000));
    for (let p = 0; p < cohort.length; p += result.pathParcelStride) {
        const id = cohort[p].id, path: ParcelAnalysis["paths"][number] = { id, x: [], y: [], z: [] };
        let gap = false;
        for (let s = 0; s < samples; s++) {
            const b = (id * samples + s) * vars, t = values[b];
            if (!valid(b) || t < filter.timeMin || t > filter.timeMax) {
                gap = true;
                continue;
            }
            if (gap && path.x.length) {
                path.x.push(null);
                path.y.push(null);
                path.z.push(null);
            }
            gap = false;
            if (s % result.pathStride !== 0 && s !== samples - 1)
                continue;
            path.x.push(values[b + 1] / 1000);
            path.y.push(values[b + 2] / 1000);
            path.z.push(values[b + 3] / 1000);
        }
        result.paths.push(path);
    }
    let previousTime: number | null = null, previousSum: number | null = null, anchor: number | null = null, integral = 0;
    if (selected >= 0)
        for (let s = 0; s < samples; s++) {
            const b = (selected * samples + s) * vars, t = values[b];
            if (Number.isFinite(t) && (t < filter.timeMin || t > filter.timeMax))
                continue;
            const finite = (c: number) => c < vars && Number.isFinite(values[b + c]) ? values[b + c] : null;
            const zeta = finite(5), terms = [finite(6), finite(7), finite(8)];
            result.times.push(finite(0));
            result.heights.push(finite(3) == null ? null : values[b + 3] / 1000);
            result.w.push(finite(4));
            result.zeta.push(zeta);
            terms.forEach((v, i) => result.terms[i].push(v));
            if (!Number.isFinite(t) || zeta == null || terms.some(v => v == null) || previousTime != null && t <= previousTime) {
                result.integrated.push(null);
                result.change.push(null);
                previousTime = null;
                previousSum = null;
                anchor = null;
                integral = 0;
                continue;
            }
            const sum = terms.reduce<number>((a, v) => a + v!, 0);
            if (previousTime == null) {
                anchor = zeta;
                integral = 0;
            }
            else
                integral += (sum + previousSum!) * .5 * (t - previousTime);
            result.integrated.push(integral);
            result.change.push(zeta - anchor!);
            previousTime = t;
            previousSum = sum;
        }
    return result;
}
