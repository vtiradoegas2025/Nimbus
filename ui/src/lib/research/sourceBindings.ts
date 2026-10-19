export interface ResearchBindings {
    definitions: Record<string, Record<string, string>>;
    frames: {
        runDir: string;
        stepIndex: number;
        generation: string | null;
        definitionId: string;
    }[];
}
export function validateResearchBindings(value: unknown): ResearchBindings | undefined {
    if (value == null)
        return undefined;
    const v = value as ResearchBindings;
    if (!v || typeof v !== "object" || !v.definitions || typeof v.definitions !== "object" || Array.isArray(v.definitions) || !Array.isArray(v.frames) || v.frames.length > 2000 || Object.keys(v.definitions).length > 2000)
        throw new Error("Invalid research source bindings");
    for (const schema of Object.values(v.definitions))
        if (!schema || typeof schema !== "object" || Array.isArray(schema) || Object.keys(schema).length > 2000 || Object.entries(schema).some(([field, definition]) => field.length > 256 || typeof definition !== "string" || definition.length > 2048))
            throw new Error("Invalid research field definitions");
    for (const frame of v.frames)
        if (!frame || typeof frame.runDir !== "string" || frame.runDir.length > 4096 || !Number.isSafeInteger(frame.stepIndex) || frame.stepIndex < 0 || !(frame.generation === null || typeof frame.generation === "string") || !Object.hasOwn(v.definitions, frame.definitionId))
            throw new Error("Invalid research source frame");
    return v;
}
