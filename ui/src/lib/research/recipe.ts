export const RESEARCH_RECIPE_CHANGE_EVENT = "nimbus:research-recipe-change";
import { validateResearchBindings, type ResearchBindings } from "./sourceBindings";
import { sanitizeResearchViews, type ResearchViews } from "./views";
export const RESEARCH_TABS = ["sections", "environment", "parcels", "cold-pools", "pressure", "hydrometeors", "resolution"] as const;
export type ResearchTab = typeof RESEARCH_TABS[number];
export interface ResearchRecipe {
    schema: "nimbus.research-workspace";
    version: 1;
    datasetIdentity: string;
    tab: ResearchTab;
    stepIndex: number;
    settings: Partial<Record<ResearchTab, Record<string, unknown>>>;
    title: string;
    citation: string;
    viewRanges?: ResearchViews;
    sourceBindings?: ResearchBindings;
    bookmarks: {
        name: string;
        stepIndex: number;
        tab: ResearchTab;
        settings: Record<string, unknown>;
    }[];
}
function validSettings(value: unknown): value is Record<string, unknown> {
    if (!value || typeof value !== "object" || Array.isArray(value))
        return false;
    const integers = new Set(["x", "y", "inflowX", "inflowY", "reference", "level", "selected", "samples", "bins", "heightBins", "frameStride"]);
    const entries = Object.entries(value);
    return entries.length <= 64 && entries.every(([key, v]) => !integers.has(key) || typeof v === "number" && Number.isSafeInteger(v) && v >= (key === "frameStride" ? 1 : 0))
        && entries.every(([key, v]) => !["probe", "sectionFraction"].includes(key) || typeof v === "number" && v >= 0 && v <= 1)
        && entries.every(([key, v]) => key.length <= 100 && key !== "__proto__" && key !== "constructor" &&
            (typeof v === "string" && v.length <= 2000 || typeof v === "boolean" || typeof v === "number" && Number.isFinite(v)));
}
export function parseResearchRecipe(text: string, identity: string): ResearchRecipe {
    if (text.length > 1000000)
        throw new Error("Research recipe exceeds its size limit");
    const value = JSON.parse(text) as ResearchRecipe;
    if (!value || value.schema !== "nimbus.research-workspace" || value.version !== 1 || value.datasetIdentity !== identity
        || !RESEARCH_TABS.includes(value.tab) || !Number.isSafeInteger(value.stepIndex) || value.stepIndex < 0
        || !value.settings || Object.entries(value.settings).some(([key, v]) => !(RESEARCH_TABS as readonly string[]).includes(key) || !validSettings(v))
        || typeof value.title !== "string" || value.title.length > 300 || typeof value.citation !== "string" || value.citation.length > 2000
        || !Array.isArray(value.bookmarks) || value.bookmarks.length > 100 || value.bookmarks.some(b => !b || typeof b.name !== "string" || b.name.length > 200 || !Number.isSafeInteger(b.stepIndex) || b.stepIndex < 0 || !RESEARCH_TABS.includes(b.tab) || !validSettings(b.settings)))
        throw new Error("Research recipe is invalid or belongs to a different dataset");
    return { ...value, viewRanges: sanitizeResearchViews(value.viewRanges), sourceBindings: validateResearchBindings(value.sourceBindings) };
}
