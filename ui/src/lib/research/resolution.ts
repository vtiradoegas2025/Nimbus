import type { StepInfo } from "@/lib/run/runOutputs";
import { physicalAxes } from "./data";
import type { PhysicalBox } from "./statistics";
export function matchSavedTime(steps: readonly StepInfo[], time: number, tolerance: number): StepInfo {
    if (!Number.isFinite(time) || !Number.isFinite(tolerance) || tolerance < 0)
        throw new Error("Time alignment requires finite time and nonnegative tolerance");
    const found = steps.filter(step => step.simulationTimeS != null && Math.abs(step.simulationTimeS - time) <= tolerance);
    if (found.length !== 1)
        throw new Error(found.length ? "Multiple saved frames lie within the time tolerance" : "No saved frame lies within the time tolerance");
    return found[0];
}
export function commonPhysicalBox(steps: readonly StepInfo[]): PhysicalBox {
    if (!steps.length)
        throw new Error("No comparison members");
    const boxes = steps.map(step => {
        const axes = physicalAxes(step.grid), lower = step.grid?.validIndexRange?.lower ?? [0, 0, 0], upper = step.grid?.validIndexRange?.upper ?? axes.map(a => a.length - 1);
        if (step.grid?.terrainFollowing && !step.grid.physicalBoundsM)
            throw new Error("Terrain comparisons require physical domain bounds");
        return step.grid?.physicalBoundsM ?? { lower: axes.map((a, i) => a[lower[i]]), upper: axes.map((a, i) => a[upper[i]]) };
    });
    const lower = [0, 1, 2].map(i => Math.max(...boxes.map(box => box.lower[i]))) as [
        number,
        number,
        number
    ];
    const upper = [0, 1, 2].map(i => Math.min(...boxes.map(box => box.upper[i]))) as [
        number,
        number,
        number
    ];
    if (lower.some((v, i) => !Number.isFinite(v) || !Number.isFinite(upper[i]) || v >= upper[i]))
        throw new Error("Members do not share a three-dimensional physical region");
    return { lower, upper };
}
