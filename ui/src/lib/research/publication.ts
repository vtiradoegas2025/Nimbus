import { researchSourceLedger } from "./provenance";
import type { ResearchResult, ResearchTable } from "@/components/research/common";
import { createStoreZip } from "@/lib/platform/storeZip";
import { sha256Hex } from "@/lib/studio/investigationPublication";
import { captureFrameDataUri } from "@/lib/run/frameCapture";
import { isPlotCanvasReceiptCurrent, waitForPlotCanvasReady } from "@/lib/plot/plotCanvasReadiness";
export function tableCsv(columns: readonly string[], rows: readonly (readonly (string | number | null)[])[]): string {
    const cell = (value: string | number | null) => value == null ? "" : typeof value === "number" ? (Number.isFinite(value) ? String(value) : "") : '"' + value.replace(/"/g, '""') + '"';
    return [columns, ...rows].map(row => row.map(cell).join(",")).join("\n") + "\n";
}
async function tableBytes(table: ResearchTable, check: () => void): Promise<Uint8Array> {
    const encode = new TextEncoder(), parts: Uint8Array[] = [encode.encode(tableCsv(table.columns, []))];
    let total = parts[0].length;
    const rows = table.rowCount ?? table.rows.length;
    for (let start = 0; start < rows; start += 4000) {
        check();
        const values = Array.from({ length: Math.min(4000, rows - start) }, (_, i) => table.rowAt?.(start + i) ?? table.rows[start + i]);
        const bytes = encode.encode(tableCsv([], values).slice(1));
        total += bytes.length;
        if (total > 32 * 1024 * 1024)
            throw new Error("A numerical table exceeds 32 MiB; choose a smaller region or fewer samples");
        parts.push(bytes);
        await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    const result = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) {
        result.set(part, offset);
        offset += part.length;
    }
    return result;
}
export async function publishResearch(root: HTMLElement, result: ResearchResult, recipe: Record<string, unknown>, title: string, citation: string, stillCurrent: () => boolean): Promise<Uint8Array> {
    const elements = [...root.querySelectorAll<HTMLElement>(".research-panel .plot-canvas")];
    if (!elements.length || elements.length !== result.panels.length)
        throw new Error("Wait for every research panel to render");
    const receipts = await Promise.all(elements.map(waitForPlotCanvasReady));
    const check = () => { if (!stillCurrent() || !root.isConnected || !receipts.every(isPlotCanvasReceiptCurrent))
        throw new Error("The analysis or figure changed during export"); };
    check();
    const images: HTMLImageElement[] = [];
    for (const element of elements) {
        const bounds = element.getBoundingClientRect();
        const scale = Math.min(window.devicePixelRatio || 1, 2);
        const uri = await captureFrameDataUri(element, { width: Math.max(1, Math.round(bounds.width * scale)), height: Math.max(1, Math.round(bounds.height * scale)) }, { assertCurrent: check });
        const image = new Image();
        await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new Error("Research capture could not be decoded")); image.src = uri; });
        images.push(image);
        check();
    }
    const width = Math.max(...images.map(image => image.width)), height = Math.max(...images.map(image => image.height));
    const rows = Math.ceil(images.length / 2), canvas = document.createElement("canvas");
    canvas.width = 2 * width + 60;
    canvas.height = rows * (height + 45) + 150;
    if (canvas.width * canvas.height > 32000000)
        throw new Error("Figure exceeds the 32 million pixel export budget; reduce the viewport size");
    const ctx = canvas.getContext("2d");
    if (!ctx)
        throw new Error("Figure canvas is unavailable");
    ctx.fillStyle = "#101419";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ecf0f7";
    ctx.font = "24px sans-serif";
    ctx.fillText(title.slice(0, 120), 20, 35);
    ctx.font = "14px sans-serif";
    ctx.fillText(citation.slice(0, 180), 20, 62);
    const times = [...new Set(result.sources.map(source => source.step.simulationTimeS))];
    ctx.fillText(`Saved times: ${times.slice(0, 12).map(time => time == null ? "unknown" : `${time} s`).join(", ")}${times.length > 12 ? " · full list in provenance" : ""}`, 20, 84);
    images.forEach((image, i) => { const x = 20 + (i % 2) * (width + 20), y = 100 + Math.floor(i / 2) * (height + 45); ctx.fillText(`${String.fromCharCode(65 + i)}. ${result.panels[i].title}`, x, y - 8); ctx.drawImage(image, x, y); });
    const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error("Figure PNG encoding failed")), "image/png"));
    const png = new Uint8Array(await blob.arrayBuffer());
    check();
    const encode = (text: string) => new TextEncoder().encode(text);
    const tables: {
        name: string;
        data: Uint8Array;
    }[] = [];
    let tableTotal = 0;
    for (let i = 0; i < result.tables.length; i++) {
        const table = result.tables[i], data = await tableBytes(table, check);
        tableTotal += data.length;
        if (tableTotal > 48 * 1024 * 1024)
            throw new Error("Numerical tables exceed 48 MiB; choose fewer fields or a smaller region");
        tables.push({ name: `tables/${i + 1}-${table.name.replace(/[^A-Za-z0-9_-]/g, "-")}.csv`, data });
    }
    const sourceLedger = researchSourceLedger(result.sources);
    const replayRecipe = { ...recipe, settings: { ...(recipe.settings as Record<string, unknown> ?? {}), [String(recipe.tab)]: result.parameters }, sourceBindings: sourceLedger.bindings };
    const recipeText = JSON.stringify(replayRecipe, null, 2);
    if (recipeText.length > 1000000)
        throw new Error("Replay recipe exceeds 1 MB; choose fewer history frames");
    const provenance = { schema: "nimbus.research-publication", version: 1, title, citation, createdAt: new Date().toISOString(), recipe: replayRecipe, sources: sourceLedger,
        parameters: result.parameters, notes: result.notes, panels: result.panels.map(panel => ({ id: panel.id, title: panel.title, layout: { ...panel.layout, ...(recipe.viewRanges as Record<string, unknown> | undefined)?.[panel.id] as object } })),
        artifacts: await Promise.all([{ name: "figure.png", data: png }, { name: "recipe.json", data: encode(recipeText) }, ...tables].map(async (entry) => ({ filename: entry.name, bytes: entry.data.length, sha256: await sha256Hex(entry.data) }))),
        analysisEngine: "Nimbus frontend native-cell research tools v1", sourceArraysEmbedded: false, executedPython: null };
    const archive = createStoreZip([{ name: "figure.png", data: png }, { name: "recipe.json", data: encode(recipeText) }, { name: "provenance.json", data: encode(JSON.stringify(provenance, null, 2)) }, ...tables]);
    check();
    return archive;
}
export async function parseResearchArchive(bytes: Uint8Array, identity: string) {
    const { readStoreZip } = await import("@/lib/platform/storeZip");
    const { parseResearchRecipe } = await import("./recipe");
    const entries = readStoreZip(bytes), decoder = new TextDecoder();
    const recipeBytes = entries.get("recipe.json"), provenanceBytes = entries.get("provenance.json");
    if (!recipeBytes || !provenanceBytes)
        throw new Error("Figure package is missing its recipe or provenance");
    const recipe = parseResearchRecipe(decoder.decode(recipeBytes), identity), provenance = JSON.parse(decoder.decode(provenanceBytes));
    if (provenance.schema !== "nimbus.research-publication" || provenance.version !== 1 || !Array.isArray(provenance.artifacts) || provenance.artifacts.length > 30)
        throw new Error("Invalid research figure provenance");
    const paired = parseResearchRecipe(JSON.stringify(provenance.recipe), identity);
    if (JSON.stringify(paired) !== JSON.stringify(recipe))
        throw new Error("Figure package recipe and provenance differ");
    for (const artifact of provenance.artifacts) {
        if (!artifact || typeof artifact.filename !== "string" || !Number.isSafeInteger(artifact.bytes) || artifact.bytes < 0 || typeof artifact.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(artifact.sha256))
            throw new Error("Invalid figure artifact metadata");
        const data = entries.get(artifact.filename);
        if (!data || data.length !== artifact.bytes || await sha256Hex(data) !== artifact.sha256)
            throw new Error(`Figure artifact failed its integrity check: ${artifact.filename}`);
    }
    if (!provenance.artifacts.some((artifact: {
        filename: string;
    }) => artifact.filename === "figure.png"))
        throw new Error("Figure package contains no bound image");
    return recipe;
}
