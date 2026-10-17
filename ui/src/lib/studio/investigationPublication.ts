import { datasetSourceIdentity, type LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { isLocalDatasetSource } from "@/lib/comparison/comparisonSession";
import { validInvestigationWorkspace, type InvestigationWorkspaceRecipe } from "./investigationWorkspace";
import type { GridInfo, StepInfo } from "@/lib/run/runOutputs";
import { createStoreZip, readStoreZip } from "@/lib/platform/storeZip";

export const INVESTIGATION_PUBLICATION_SCHEMA = "nimbus.studio.investigation-publication" as const;
export const INVESTIGATION_PUBLICATION_VERSION = 1 as const;
export const INVESTIGATION_RENDERER_ID = "nimbus.studio.linked-investigation" as const;
export const INVESTIGATION_RENDERER_VERSION = 1 as const;

export type InvestigationFigurePanel = "plan" | "vertical_transect" | "vertical_profile" | "probe_history";

export interface InvestigationPublication {
  schema: typeof INVESTIGATION_PUBLICATION_SCHEMA;
  schemaVersion: typeof INVESTIGATION_PUBLICATION_VERSION;
  title: string;
  exportedAt: string;
  dataset: LocalDatasetSource;
  datasetIdentity: string;
  recipe: InvestigationWorkspaceRecipe;
  figure: {
    filename: "figure.png";
    mediaType: "image/png";
    width: number;
    height: number;
    byteLength: number;
    sha256: string;
    panels: InvestigationFigurePanel[];
  };
  provenance: {
    renderer: { id: typeof INVESTIGATION_RENDERER_ID; version: typeof INVESTIGATION_RENDERER_VERSION };
    sourceFrame: {
      stepIndex: number;
      simulationTimeS: number | null;
      publicationGeneration: string | null;
      publicationState: StepInfo["publicationState"];
    };
    field: { canonicalName: string; unit: string; dimensionality: "volume3d" };
    grid: GridInfo | null;
    analysis: {
      horizontalInterpolation: "bilinear-on-scalar-centers";
      verticalSampling: "native-scalar-centers";
      finitePolicy: "all-contributing-horizontal-samples-finite";
    };
    replay: {
      requiresExactDatasetIdentity: true;
      localInteractiveRestore: true;
      headlessBatchRendering: false;
      headlessBlocker: string;
    };
  };
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const PANEL_VALUES = new Set<InvestigationFigurePanel>(["plan", "vertical_transect", "vertical_profile", "probe_history"]);

function finiteInteger(value: unknown, min: number, max: number): value is number {
  return Number.isInteger(value) && Number(value) >= min && Number(value) <= max;
}

export function createInvestigationPublication(args: {
  title: string;
  dataset: LocalDatasetSource;
  recipe: InvestigationWorkspaceRecipe;
  step: StepInfo;
  unit: string;
  figure: InvestigationPublication["figure"];
}): InvestigationPublication {
  const identity = datasetSourceIdentity(args.dataset);
  if (!validInvestigationWorkspace(args.recipe, identity) || args.step.index !== args.recipe.stepIndex) {
    throw new Error("Investigation recipe does not match the exact dataset frame being published.");
  }
  return {
    schema: INVESTIGATION_PUBLICATION_SCHEMA,
    schemaVersion: INVESTIGATION_PUBLICATION_VERSION,
    title: args.title.trim() || `${args.dataset.displayName} investigation`,
    exportedAt: new Date().toISOString(),
    dataset: args.dataset,
    datasetIdentity: identity,
    recipe: args.recipe,
    figure: args.figure,
    provenance: {
      renderer: { id: INVESTIGATION_RENDERER_ID, version: INVESTIGATION_RENDERER_VERSION },
      sourceFrame: {
        stepIndex: args.step.index,
        simulationTimeS: args.step.simulationTimeS,
        publicationGeneration: args.step.publicationGeneration,
        publicationState: args.step.publicationState,
      },
      field: { canonicalName: args.recipe.fieldName, unit: args.unit, dimensionality: "volume3d" },
      grid: args.step.grid,
      analysis: {
        horizontalInterpolation: "bilinear-on-scalar-centers",
        verticalSampling: "native-scalar-centers",
        finitePolicy: "all-contributing-horizontal-samples-finite",
      },
      replay: {
        requiresExactDatasetIdentity: true,
        localInteractiveRestore: true,
        headlessBatchRendering: false,
        headlessBlocker: "Studio does not yet expose a headless desktop dataset adapter and renderer contract.",
      },
    },
  };
}

export function parseInvestigationPublication(text: string): InvestigationPublication {
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("Figure provenance is not valid JSON."); }
  if (!value || typeof value !== "object") throw new Error("Figure provenance root must be an object.");
  const item = value as Partial<InvestigationPublication>;
  if (item.schema !== INVESTIGATION_PUBLICATION_SCHEMA || item.schemaVersion !== INVESTIGATION_PUBLICATION_VERSION) throw new Error("Unsupported investigation publication schema or version.");
  if (!isLocalDatasetSource(item.dataset) || item.datasetIdentity !== datasetSourceIdentity(item.dataset)) throw new Error("Figure provenance contains an invalid dataset identity.");
  if (!validInvestigationWorkspace(item.recipe, item.datasetIdentity)) throw new Error("Figure provenance contains an invalid or mismatched investigation recipe.");
  if (typeof item.title !== "string" || item.title.trim().length === 0 || item.title.length > 300
    || typeof item.exportedAt !== "string" || !Number.isFinite(Date.parse(item.exportedAt))) throw new Error("Figure publication identity is incomplete.");
  const figure = item.figure;
  if (!figure || figure.filename !== "figure.png" || figure.mediaType !== "image/png"
    || !finiteInteger(figure.width, 200, 8192) || !finiteInteger(figure.height, 200, 8192)
    || !finiteInteger(figure.byteLength, 8, 64 * 1024 * 1024)
    || typeof figure.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(figure.sha256)
    || !Array.isArray(figure.panels) || figure.panels.length < 3 || figure.panels.length > 4
    || new Set(figure.panels).size !== figure.panels.length || !figure.panels.every((panel) => PANEL_VALUES.has(panel))) {
    throw new Error("Figure artifact metadata is invalid.");
  }
  const provenance = item.provenance;
  if (!provenance || provenance.renderer?.id !== INVESTIGATION_RENDERER_ID
    || provenance.renderer.version !== INVESTIGATION_RENDERER_VERSION
    || provenance.sourceFrame?.stepIndex !== item.recipe.stepIndex
    || (provenance.sourceFrame.simulationTimeS !== null && !Number.isFinite(provenance.sourceFrame.simulationTimeS))
    || !["complete", "legacy"].includes(provenance.sourceFrame.publicationState)
    || provenance.field?.canonicalName !== item.recipe.fieldName || provenance.field.dimensionality !== "volume3d"
    || provenance.analysis?.horizontalInterpolation !== "bilinear-on-scalar-centers"
    || provenance.analysis.verticalSampling !== "native-scalar-centers"
    || provenance.analysis.finitePolicy !== "all-contributing-horizontal-samples-finite"
    || provenance.replay?.requiresExactDatasetIdentity !== true
    || provenance.replay.localInteractiveRestore !== true
    || provenance.replay.headlessBatchRendering !== false
    || typeof provenance.replay.headlessBlocker !== "string" || provenance.replay.headlessBlocker.length === 0) {
    throw new Error("Figure provenance method or replay boundary is invalid.");
  }
  return item as InvestigationPublication;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes).buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function serializeInvestigationPublication(value: InvestigationPublication): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export function createInvestigationArchive(publication: InvestigationPublication, png: Uint8Array): Uint8Array {
  if (png.length !== publication.figure.byteLength) throw new Error("Figure PNG length does not match its provenance.");
  const readme = [
    "Nimbus Studio investigation figure package",
    "",
    "figure.png       Publication image",
    "provenance.json  Versioned dataset, frame, analysis, and replay recipe",
    "",
    "The source data are not embedded. Open the exact recorded local dataset before importing this package.",
    "Headless batch replay is explicitly not implemented in this schema version.",
    "",
  ].join("\n");
  return createStoreZip([
    { name: "figure.png", data: png },
    { name: "provenance.json", data: encoder.encode(serializeInvestigationPublication(publication)) },
    { name: "README.txt", data: encoder.encode(readme) },
  ]);
}

export async function parseInvestigationArchive(bytes: Uint8Array): Promise<{ publication: InvestigationPublication; png: Uint8Array }> {
  const files = readStoreZip(bytes);
  const png = files.get("figure.png");
  const provenance = files.get("provenance.json");
  if (!png || !provenance || !files.has("README.txt") || files.size !== 3) throw new Error("Figure package must contain only figure.png, provenance.json, and README.txt.");
  if (png.length < 8 || ![137, 80, 78, 71, 13, 10, 26, 10].every((byte, index) => png[index] === byte)) throw new Error("Figure package image is not a PNG.");
  const publication = parseInvestigationPublication(decoder.decode(provenance));
  if (publication.figure.byteLength !== png.length || publication.figure.sha256 !== await sha256Hex(png)) throw new Error("Figure PNG does not match its SHA-256 provenance.");
  return { publication, png };
}

export function downloadInvestigationArchive(bytes: Uint8Array, filename: string): void {
  const blob = new Blob([bytes as BlobPart], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
