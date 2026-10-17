import { fieldDefinitionIdentity } from "@/lib/run/runOutputs";
import type { ResearchSource } from "./data";
import type { ResearchBindings } from "./sourceBindings";
export function researchSourceLedger(sources: readonly ResearchSource[]) {
    const grids: Record<string, unknown> = {}, schemas: Record<string, unknown> = {}, definitions: ResearchBindings["definitions"] = {}, frames: (ResearchBindings["frames"][number] & {
        timeS: number | null;
        gridId: string;
    })[] = [];
    const gridKeys = new Map<string, string>(), schemaKeys = new Map<string, string>();
    let bytes = 0;
    for (const { runDir, step } of sources) {
        const gridJson = JSON.stringify(step.grid), definition = Object.fromEntries(step.fields.map(field => [field, JSON.stringify([fieldDefinitionIdentity(runDir, step.index, field, step.publicationGeneration), step.fieldUnits[field] ?? null, step.fieldCompression?.[field] ?? null])]));
        const schema = { fields: step.fields, units: step.fieldUnits, origins: step.fieldOrigins ?? {}, compression: step.fieldCompression ?? {}, definitions: definition };
        const schemaJson = JSON.stringify(schema);
        let gridId = gridKeys.get(gridJson);
        if (!gridId) {
            gridId = `grid-${gridKeys.size}`;
            gridKeys.set(gridJson, gridId);
            grids[gridId] = step.grid;
            bytes += gridJson.length * 2;
        }
        let definitionId = schemaKeys.get(schemaJson);
        if (!definitionId) {
            definitionId = `fields-${schemaKeys.size}`;
            schemaKeys.set(schemaJson, definitionId);
            schemas[definitionId] = schema;
            definitions[definitionId] = definition;
            bytes += schemaJson.length * 2;
        }
        if (bytes > 16 * 1024 * 1024)
            throw new Error("Source provenance exceeds 16 MiB; choose fewer history frames");
        frames.push({ runDir, stepIndex: step.index, generation: step.publicationGeneration, timeS: step.simulationTimeS, gridId, definitionId });
    }
    const bindings: ResearchBindings = { definitions, frames: frames.map(({ runDir, stepIndex, generation, definitionId }) => ({ runDir, stepIndex, generation, definitionId })) };
    return { grids, schemas, frames, bindings };
}
export function verifyResearchBindings(bindings: ResearchBindings | undefined, sources: readonly ResearchSource[]): void {
    if (!bindings)
        return;
    for (const frame of bindings.frames) {
        const source = sources.find(source => source.runDir === frame.runDir && source.step.index === frame.stepIndex);
        if (!source)
            throw new Error(`Reopen the recipe's source frame: ${frame.runDir}, step ${frame.stepIndex}`);
        if (source.step.publicationGeneration !== frame.generation)
            throw new Error("A research source publication changed since export");
        if (JSON.stringify([...source.step.fields].sort()) !== JSON.stringify(Object.keys(bindings.definitions[frame.definitionId]).sort()))
            throw new Error("A research source field catalog changed since export");
        for (const [field, definition] of Object.entries(bindings.definitions[frame.definitionId]))
            if (JSON.stringify([fieldDefinitionIdentity(source.runDir, source.step.index, field, frame.generation), source.step.fieldUnits[field] ?? null, source.step.fieldCompression?.[field] ?? null]) !== definition)
                throw new Error(`The research field definition changed: ${field}`);
    }
}
