import { windsAreEarthRelative } from "@/lib/run/frameMotion";
import type { StepInfo } from "@/lib/run/runOutputs";
function validateProfile(z: readonly number[], values: readonly number[]): void {
    if (z.length !== values.length || !z.length || z.some((v, i) => !Number.isFinite(v) || i > 0 && v <= z[i - 1]))
        throw new Error("Profile heights must increase strictly");
}
export function sampleFiniteProfile(z: readonly number[], values: readonly number[], target: number): number | null {
    if (!Number.isFinite(target) || target < z[0] || target > z[z.length - 1])
        return null;
    let low = 0, high = z.length - 1;
    while (high - low > 1) {
        const mid = Math.floor((low + high) / 2);
        if (z[mid] <= target)
            low = mid;
        else
            high = mid;
    }
    if (target === z[low])
        return Number.isFinite(values[low]) ? values[low] : null;
    if (target === z[high])
        return Number.isFinite(values[high]) ? values[high] : null;
    if (!Number.isFinite(values[low]) || !Number.isFinite(values[high]))
        return null;
    const f = (target - z[low]) / (z[high] - z[low]);
    return values[low] * (1 - f) + values[high] * f;
}
export function interpolateProfile(z: readonly number[], values: readonly number[], target: number): number | null {
    validateProfile(z, values);
    return sampleFiniteProfile(z, values, target);
}
export function profileDifference(z: readonly number[], values: readonly number[], referenceZ: readonly number[], reference: readonly number[]): (number | null)[] {
    validateProfile(z, values);
    validateProfile(referenceZ, reference);
    return z.map((height, i) => { const value = sampleFiniteProfile(referenceZ, reference, height); return Number.isFinite(values[i]) && value != null ? values[i] - value : null; });
}
export function layerSummary(z: readonly number[], values: readonly number[], lower: number, upper: number): {
    mean: number | null;
    difference: number | null;
    coverage: number;
} {
    validateProfile(z, values);
    if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower)
        throw new Error("Layer bounds must be finite and increasing");
    let integral = 0, covered = 0;
    for (let i = 1; i < z.length; i++) {
        const low = Math.max(lower, z[i - 1]), high = Math.min(upper, z[i]);
        if (high <= low || !Number.isFinite(values[i - 1]) || !Number.isFinite(values[i]))
            continue;
        const span = z[i] - z[i - 1], slope = (values[i] - values[i - 1]) / span;
        const a = values[i - 1] + (low - z[i - 1]) * slope, b = values[i - 1] + (high - z[i - 1]) * slope;
        integral += (a + b) * .5 * (high - low);
        covered += high - low;
    }
    const a = sampleFiniteProfile(z, values, lower), b = sampleFiniteProfile(z, values, upper);
    return { mean: covered ? integral / covered : null, difference: a != null && b != null ? b - a : null, coverage: covered / (upper - lower) };
}
export function windReferenceOffset(step: StepInfo, motion: [
    number,
    number
]): [
    number,
    number
] {
    if (!motion.every(Number.isFinite))
        throw new Error("Storm motion must contain finite Earth-relative components");
    if (windsAreEarthRelative(step))
        return [-motion[0], -motion[1]];
    const frame = step.motion?.frameVelocityMs;
    if (step.motion?.windsEarthRelative !== false || !frame?.every(Number.isFinite))
        throw new Error("The saved wind reference is unknown; storm-relative transformation is unavailable");
    return [frame[0] - motion[0], frame[1] - motion[1]];
}
