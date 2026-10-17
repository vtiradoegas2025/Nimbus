import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import {
  axisCoords,
  LOCKED_ASPECT_AXIS,
  lockedAspectPartnerAxis,
} from "@/lib/science/gridCoords";
import {
  clearInvestigationWorkspace,
  INVESTIGATION_WORKSPACE_SCHEMA,
  INVESTIGATION_WORKSPACE_VERSION,
  loadInvestigationWorkspace,
  saveInvestigationWorkspace,
  type InvestigationAnnotation,
  type InvestigationWorkspaceRecipe,
} from "@/lib/studio/investigationWorkspace";
import { slice2D, type Npy3D } from "@/lib/platform/npy";
import { DBZ_RANGE, resolveColorscale } from "@/lib/plot/plotColormaps";
import { fieldDefinitionIdentity, readFieldAny, registeredFieldRank, type GridInfo, type StepInfo } from "@/lib/run/runOutputs";
import { investigationHistoryAxis, InvestigationHistoryOwner } from "@/lib/studio/investigationHistory";
import { isPlotCanvasReceiptCurrent, waitForPlotCanvasReady } from "@/lib/plot/plotCanvasReadiness";
import {
  buildVerticalTransect,
  sampleHorizontalPoint,
  type PhysicalPoint2D,
  type VerticalTransect,
} from "@/lib/plot/transectSampling";
import { cropToValidPlotDomain } from "@/lib/science/validDomain";
import { PlotCanvas } from "./PlotCanvas";
import { captureFrameDataUri } from "@/lib/run/frameCapture";
import {
  createInvestigationArchive,
  createInvestigationPublication,
  downloadInvestigationArchive,
  parseInvestigationArchive,
  parseInvestigationPublication,
  sha256Hex,
  type InvestigationFigurePanel,
} from "@/lib/studio/investigationPublication";

interface InvestigationWorkspaceProps {
  dataset: LocalDatasetSource;
  steps: readonly StepInfo[];
  stepIdx: number;
  onStepChange: (index: number) => void;
}

interface LoadedFrame {
  identity: string;
  field: Npy3D;
  grid: GridInfo | null;
}

type PlacementMode = "start" | "end" | "probe";

interface HistorySeries {
  identity: string;
  timeUnit: "s" | "index";
  timeTitle: string;
  times: number[];
  values: number[];
  point: PhysicalPoint2D;
  zIndex: number;
  fieldName: string;
}

const PREFERRED_FIELDS = ["reflectivity_dbz", "w", "pressure", "theta", "u", "v"];
const KNOWN_HORIZONTAL_FIELDS = new Set([
  "u10", "v10", "t2", "td2", "skin_temperature", "surface_sensible_heat_flux",
  "surface_latent_heat_flux", "surface_moisture_flux", "surface_pressure_perturbation",
  "surface_friction_velocity", "surface_stress_u", "surface_stress_v", "pbl_height",
  "cold_pool_boundary", "precip_rate", "accumulated_rainfall", "composite_reflectivity",
  "column_max_w", "column_max_vorticity", "vil", "precipitable_water", "liquid_water_path",
  "cloud_water_path", "lcl", "lfc", "el", "cape", "cin", "lifted_index", "k_index",
  "showalter_index", "total_totals", "srh_0_1km", "srh_0_3km", "ehi", "scp", "stp",
  "updraft_helicity", "ppi_sweep", "rhi_sweep", "bwer", "mesocyclone_diagnostic", "vrot",
]);

function finiteRange(values: Iterable<number>): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 0.5, max + 0.5];
  return [min, max];
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function pointAtFraction(start: PhysicalPoint2D, end: PhysicalPoint2D, fraction: number): PhysicalPoint2D {
  return {
    xKm: start.xKm + (end.xKm - start.xKm) * fraction,
    yKm: start.yKm + (end.yKm - start.yKm) * fraction,
  };
}

function closestFraction(start: PhysicalPoint2D, end: PhysicalPoint2D, point: PhysicalPoint2D): number {
  const dx = end.xKm - start.xKm;
  const dy = end.yKm - start.yKm;
  const denominator = dx * dx + dy * dy;
  return denominator > 0
    ? clamp(((point.xKm - start.xKm) * dx + (point.yKm - start.yKm) * dy) / denominator, 0, 1)
    : 0;
}

function plotPoint(event: Readonly<Record<string, unknown>>): PhysicalPoint2D | null {
  const points = event.points;
  if (!Array.isArray(points) || points.length === 0 || !points[0] || typeof points[0] !== "object") return null;
  const point = points[0] as { x?: unknown; y?: unknown };
  return typeof point.x === "number" && Number.isFinite(point.x)
    && typeof point.y === "number" && Number.isFinite(point.y)
    ? { xKm: point.x, yKm: point.y }
    : null;
}

function preferredField(fields: readonly string[]): string {
  return PREFERRED_FIELDS.find((field) => fields.includes(field)) ?? fields[0] ?? "";
}

function formatTime(value: number | null): string { return value != null && Number.isFinite(value) ? `${Number(value.toPrecision(7))} s` : "physical time unavailable"; }

function safeFilename(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 100) || "nimbus-investigation";
}

async function imageFromDataUri(dataUri: string): Promise<HTMLImageElement> {
  const image = new Image(); image.src = dataUri;
  await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new Error("Captured panel PNG could not be decoded.")); });
  return image;
}

async function composeInvestigationFigure(
  root: HTMLElement,
  title: string,
  subtitle: string,
): Promise<{ png: Uint8Array; width: number; height: number; panels: InvestigationFigurePanel[]; current: () => boolean }> {
  const candidates: Array<{ panel: InvestigationFigurePanel; selector: string }> = [
    { panel: "plan", selector: ".investigation__plan-canvas" },
    { panel: "vertical_transect", selector: ".investigation__transect-canvas" },
    { panel: "vertical_profile", selector: ".investigation__profile-canvas" },
    { panel: "probe_history", selector: ".investigation__history-canvas" },
  ];
  const available = candidates.flatMap((candidate) => {
    const element = root.querySelector<HTMLElement>(candidate.selector);
    return element ? [{ ...candidate, element }] : [];
  });
  if (available.length < 3) throw new Error("Wait for the plan, transect, and profile to finish rendering before publishing.");
  const receipts = await Promise.all(available.map(({ element }) => waitForPlotCanvasReady(element)));
  const panelWidth = 720; const panelHeight = 420;
  const captures = await Promise.all(available.map(async ({ panel, element }) => ({
    panel,
    image: await imageFromDataUri(await captureFrameDataUri(element, { width: panelWidth, height: panelHeight })),
  })));
  const width = 1520; const headerHeight = 118; const footerHeight = 46; const gap = 20; const margin = 30;
  const rows = Math.ceil(captures.length / 2);
  const height = headerHeight + rows * panelHeight + Math.max(0, rows - 1) * gap + footerHeight;
  const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d"); if (!context) throw new Error("Publication canvas is unavailable.");
  context.fillStyle = "#111419"; context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(232,227,215,0.2)"; context.lineWidth = 1;
  context.beginPath(); context.moveTo(margin, headerHeight - 18); context.lineTo(width - margin, headerHeight - 18); context.stroke();
  context.fillStyle = "#d7ad67"; context.font = "700 14px ui-sans-serif, sans-serif"; context.fillText("NIMBUS STUDIO · LINKED INVESTIGATION", margin, 32);
  context.fillStyle = "#e8e3d7"; context.font = "32px Georgia, serif"; context.fillText(title.slice(0, 90), margin, 70);
  context.fillStyle = "#96938d"; context.font = "14px ui-sans-serif, sans-serif"; context.fillText(subtitle.slice(0, 180), margin, 94);
  captures.forEach(({ image }, index) => {
    const column = index % 2; const row = Math.floor(index / 2);
    const x = margin + column * (panelWidth + gap); const y = headerHeight + row * (panelHeight + gap);
    context.drawImage(image, x, y, panelWidth, panelHeight);
    context.strokeRect(x + 0.5, y + 0.5, panelWidth - 1, panelHeight - 1);
  });
  context.fillStyle = "#777b82"; context.font = "12px ui-sans-serif, sans-serif";
  context.fillText("Pixels are bound to provenance.json by SHA-256. Source arrays are referenced, not embedded.", margin, height - 17);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error("Publication PNG encoding failed.")), "image/png"));
  const current = () => receipts.every(isPlotCanvasReceiptCurrent);
  if (!current()) throw new Error("A linked plot changed during capture. Retry after the views settle.");
  return { png: new Uint8Array(await blob.arrayBuffer()), width, height, panels: captures.map(({ panel }) => panel), current };
}

export function InvestigationWorkspace(props: InvestigationWorkspaceProps) {
  return <DatasetInvestigationWorkspace key={datasetSourceIdentity(props.dataset)} {...props} />;
}

function DatasetInvestigationWorkspace({ dataset, steps, stepIdx, onStepChange }: InvestigationWorkspaceProps) {
  const identity = datasetSourceIdentity(dataset);
  const currentStep = steps[stepIdx] ?? null;
  const sharedFields = useMemo(() => {
    if (steps.length === 0) return [];
    const remainder = steps.slice(1).map((step) => new Set(step.fields));
    return steps[0].fields.filter((field) => {
      if (!remainder.every((available) => available.has(field))) return false;
      const ranks = steps.map((step) => registeredFieldRank(
        dataset.locator.path, step.index, field, step.publicationGeneration,
      ));
      return ranks.some((rank) => rank != null)
        ? ranks.every((rank) => rank === 3)
        : !KNOWN_HORIZONTAL_FIELDS.has(field);
    });
  }, [dataset.locator.path, steps]);
  const restored = useMemo(
    () => typeof window === "undefined" ? null : loadInvestigationWorkspace(window.localStorage, identity),
    [identity],
  );
  const [fieldName, setFieldName] = useState(() =>
    restored && sharedFields.includes(restored.fieldName) ? restored.fieldName : preferredField(sharedFields),
  );
  const [zIndex, setZIndex] = useState(restored?.zIndex ?? 0);
  const [start, setStart] = useState<PhysicalPoint2D | null>(restored?.start ?? null);
  const [end, setEnd] = useState<PhysicalPoint2D | null>(restored?.end ?? null);
  const [probeFraction, setProbeFraction] = useState(restored?.probeFraction ?? 0.5);
  const [annotations, setAnnotations] = useState<InvestigationAnnotation[]>(restored?.annotations ?? []);
  const [annotationText, setAnnotationText] = useState("");
  const [placementMode, setPlacementMode] = useState<PlacementMode>("probe");
  const [loadedFrame, setFrame] = useState<LoadedFrame | null>(null);
  const frameRequestIdentity = useMemo(() => JSON.stringify([identity, currentStep?.index, currentStep?.publicationGeneration,
    fieldName, fieldDefinitionIdentity(dataset.locator.path, currentStep?.index ?? null, fieldName, currentStep?.publicationGeneration ?? null),
    currentStep?.grid, currentStep?.fieldUnits?.[fieldName]]), [identity, currentStep, fieldName, dataset.locator.path]);
  const frameRequestRef = useRef(frameRequestIdentity);
  frameRequestRef.current = frameRequestIdentity;
  const frame = loadedFrame?.identity === frameRequestIdentity ? loadedFrame : null;
  const transectResult = useMemo<{ value: VerticalTransect | null; error: string | null }>(() => {
    if (!frame || !start || !end) return { value: null, error: null };
    try { return { value: buildVerticalTransect(frame.field, frame.grid, start, end), error: null }; }
    catch (reason) { return { value: null, error: reason instanceof Error ? reason.message : String(reason) }; }
  }, [frame, start, end]);
  const transect = transectResult.value;
  const [historyState, setHistory] = useState<HistorySeries | null>(null);
  const [historyError, setHistoryError] = useState<{ identity: string; error: string } | null>(null);
  const [historyProgress, setHistoryProgress] = useState<string | null>(null);
  // Separate from historyProgress: progress text is kept for the caption; this is the authority on "in flight".
  const [historyBusy, setHistoryBusy] = useState(false);
  const [frameError, setError] = useState<{ identity: string; error: string } | null>(null);
  const error = frameError?.identity === frameRequestIdentity ? frameError.error : null;
  const [saveState, setSaveState] = useState(restored ? "Workspace restored" : "New workspace");
  const [publicationOpen, setPublicationOpen] = useState(false);
  const [publicationTitle, setPublicationTitle] = useState(`${dataset.displayName} investigation`);
  const [publicationState, setPublicationState] = useState<string | null>(null);
  const [publishing, setPublishing] = useState(false);
  const initializedRef = useRef(false);
  const restoredFramePendingRef = useRef(restored?.stepIndex ?? null);
  const loadGenerationRef = useRef(0);
  const historyOwnerRef = useRef(new InvestigationHistoryOwner());
  const workspaceRef = useRef<HTMLElement | null>(null);
  const publicationInputRef = useRef<HTMLInputElement | null>(null);
  const publicationOperationRef = useRef(0);
  const pendingSaveRef = useRef<InvestigationWorkspaceRecipe | null>(null);

  const flushPendingSave = useCallback((updateStatus: boolean) => {
    const pending = pendingSaveRef.current;
    if (!pending || typeof window === "undefined") return;
    pendingSaveRef.current = null;
    const saved = saveInvestigationWorkspace(window.localStorage, pending);
    if (updateStatus) setSaveState(saved ? "Saved locally" : "Local save unavailable");
  }, []);

  // Dataset switches and navigation flush the last debounced edit without
  // turning every drag/probe movement into a synchronous localStorage write.
  useEffect(() => () => flushPendingSave(false), [flushPendingSave, identity]);

  useEffect(() => {
    if (!restored) return;
    const restoredOrdinal = steps.findIndex((step) => step.index === restored.stepIndex);
    if (restoredOrdinal >= 0 && restoredOrdinal !== stepIdx) onStepChange(restoredOrdinal);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity]);

  useEffect(() => {
    const generation = ++loadGenerationRef.current;
    const controller = new AbortController();
    if (!currentStep || !fieldName) { setFrame(null); return; }
    setError(null);
    setFrame(null);
    void readFieldAny(
      dataset.locator.path,
      currentStep.index,
      fieldName,
      currentStep.publicationGeneration,
      { latestScope: `investigation:${identity}`, signal: controller.signal },
    ).then((loaded) => {
      if (controller.signal.aborted || generation !== loadGenerationRef.current || frameRequestRef.current !== frameRequestIdentity) return;
      if (loaded.shape.length !== 3) throw new Error(`${fieldName} is rank ${loaded.shape.length}; arbitrary vertical transects require a three-dimensional field.`);
      const valid = cropToValidPlotDomain(loaded, currentStep.grid);
      if (valid.grid?.terrainFollowing) {
        throw new Error(
          "Terrain investigation transects require the exact non-separable physical height lattice; model-level k is available in ordinary scalar plots, but it is not a physical z coordinate.",
        );
      }
      const physicalField = valid.field as Npy3D;
      const x = axisCoords(0, physicalField.shape[0], valid.grid).values;
      const y = axisCoords(1, physicalField.shape[1], valid.grid).values;
      const z = axisCoords(2, physicalField.shape[2], valid.grid).values;
      if (!x || !y || !z) throw new Error("This field does not provide complete physical x/y/z scalar-center coordinates. Index geometry is intentionally not treated as physical distance.");
      const xMin = x[0]; const xMax = x.at(-1)!;
      const yMin = y[0]; const yMax = y.at(-1)!;
      const defaultStart = { xKm: xMin + 0.2 * (xMax - xMin), yKm: yMin + 0.2 * (yMax - yMin) };
      const defaultEnd = { xKm: xMin + 0.8 * (xMax - xMin), yKm: yMin + 0.8 * (yMax - yMin) };
      setStart((value) => ({ xKm: clamp(value?.xKm ?? defaultStart.xKm, xMin, xMax), yKm: clamp(value?.yKm ?? defaultStart.yKm, yMin, yMax) }));
      setEnd((value) => ({ xKm: clamp(value?.xKm ?? defaultEnd.xKm, xMin, xMax), yKm: clamp(value?.yKm ?? defaultEnd.yKm, yMin, yMax) }));
      setZIndex((value) => clamp(Math.floor(value), 0, physicalField.shape[2] - 1));
      setFrame({ identity: frameRequestIdentity, field: physicalField, grid: valid.grid as GridInfo | null });
      initializedRef.current = true;
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted && generation === loadGenerationRef.current && frameRequestRef.current === frameRequestIdentity) setError({ identity: frameRequestIdentity, error: reason instanceof Error ? reason.message : String(reason) });
    });
    return () => controller.abort();
  }, [currentStep, dataset.locator.path, fieldName, identity, frameRequestIdentity]);

  const restoredStepIndex = restored?.stepIndex ?? null;
  const restoredStepExists = restoredStepIndex != null
    && steps.some((step) => step.index === restoredStepIndex);
  const autosaveStepIndex = currentStep?.index ?? null;

  useEffect(() => {
    if (!initializedRef.current || !start || !end || !fieldName || autosaveStepIndex == null || typeof window === "undefined") return;
    // Guard: a restored component's first render uses the parent's old frame; don't overwrite the saved identity.
    if (restoredFramePendingRef.current != null) {
      if (restoredStepExists && autosaveStepIndex !== restoredFramePendingRef.current) return;
      restoredFramePendingRef.current = null;
    }
    pendingSaveRef.current = {
      schema: INVESTIGATION_WORKSPACE_SCHEMA,
      schemaVersion: INVESTIGATION_WORKSPACE_VERSION,
      datasetIdentity: identity,
      stepIndex: autosaveStepIndex,
      fieldName,
      zIndex,
      start,
      end,
      probeFraction,
      annotations,
      savedAt: new Date().toISOString(),
    };
    setSaveState("Saving locally…");
    const timer = window.setTimeout(() => flushPendingSave(true), 250);
    return () => window.clearTimeout(timer);
  }, [annotations, autosaveStepIndex, end, fieldName, flushPendingSave, identity, probeFraction, restoredStepExists, restoredStepIndex, start, zIndex]);

  const xCoords = useMemo(() => frame ? axisCoords(0, frame.field.shape[0], frame.grid) : null, [frame]);
  const yCoords = useMemo(() => frame ? axisCoords(1, frame.field.shape[1], frame.grid) : null, [frame]);
  const zCoords = useMemo(() => frame ? axisCoords(2, frame.field.shape[2], frame.grid) : null, [frame]);
  const probe = useMemo(() => start && end ? pointAtFraction(start, end, probeFraction) : null, [start, end, probeFraction]);
  const simulationTimeS = currentStep?.simulationTimeS ?? null;
  const unit = currentStep?.fieldUnits[fieldName] ?? "";
  const historyIdentity = useMemo(() => JSON.stringify([identity, fieldName, probe?.xKm, probe?.yKm, zIndex,
    steps.map(step => [step.index, step.publicationGeneration, step.simulationTimeS, step.grid, step.fieldUnits?.[fieldName],
      fieldDefinitionIdentity(dataset.locator.path, step.index, fieldName, step.publicationGeneration)])]),
  [identity, fieldName, probe?.xKm, probe?.yKm, zIndex, steps, dataset.locator.path]);
  historyOwnerRef.current.select(historyIdentity);
  const history = historyState?.identity === historyIdentity ? historyState : null;
  const displayError = error ?? transectResult.error ?? (historyError?.identity === historyIdentity ? historyError.error : null);
  const publicationIdentity = JSON.stringify([frameRequestIdentity, start, end, probeFraction, zIndex, annotations, history?.identity, publicationTitle]);
  const publicationIdentityRef = useRef(publicationIdentity);
  publicationIdentityRef.current = publicationIdentity;
  useEffect(() => {
    const owner = historyOwnerRef.current;
    setHistory(null); setHistoryError(null); setHistoryBusy(false); setHistoryProgress(null);
    return () => owner.cancel();
  }, [historyIdentity]);
  const isDbz = fieldName.includes("reflectivity") || unit.toLowerCase() === "dbz";
  const fieldRange = useMemo<[number, number]>(() => (
    frame ? (isDbz ? [...DBZ_RANGE] : finiteRange(frame.field.data)) : [0, 1]
  ), [frame, isDbz]);

  const handlePlanClick = useCallback((event: Readonly<Record<string, unknown>>) => {
    if (!frame || !start || !end) return;
    const clicked = plotPoint(event);
    const x = axisCoords(0, frame.field.shape[0], frame.grid).values;
    const y = axisCoords(1, frame.field.shape[1], frame.grid).values;
    if (!clicked || !x || !y) return;
    const bounded = { xKm: clamp(clicked.xKm, x[0], x.at(-1)!), yKm: clamp(clicked.yKm, y[0], y.at(-1)!) };
    if (placementMode === "start") setStart(bounded);
    else if (placementMode === "end") setEnd(bounded);
    else setProbeFraction(closestFraction(start, end, bounded));
    setHistory(null);
  }, [end, frame, placementMode, start]);

  const buildHistory = useCallback(async () => {
    if (!probe || !frame) return;
    const request = historyOwnerRef.current.begin();
    setHistory(null);
    setHistoryProgress(`0/${steps.length} frames`);
    setHistoryBusy(true);
    setHistoryError(null);
    const timeAxis = investigationHistoryAxis(steps);
    const values: number[] = [];
    try {
      for (let index = 0; index < steps.length; index += 1) {
        const step = steps[index];
        const loaded = await readFieldAny(dataset.locator.path, step.index, fieldName, step.publicationGeneration,
          { signal: request.signal });
        if (!historyOwnerRef.current.current(request)) return;
        if (loaded.shape.length !== 3) throw new Error(`${fieldName} is not three-dimensional at step ${step.index}.`);
        const valid = cropToValidPlotDomain(loaded, step.grid);
        values.push(sampleHorizontalPoint(valid.field as Npy3D, valid.grid as GridInfo | null, probe, zIndex));
        setHistoryProgress(`${index + 1}/${steps.length} frames`);
      }
      if (!historyOwnerRef.current.current(request)) return;
      setHistory({ identity: request.identity, timeUnit: timeAxis.unit, timeTitle: timeAxis.title,
        times: timeAxis.values, values, point: probe, zIndex, fieldName });
    } catch (reason) {
      if (historyOwnerRef.current.current(request)) setHistoryError({ identity: request.identity, error: reason instanceof Error ? reason.message : String(reason) });
    } finally {
      if (historyOwnerRef.current.current(request)) {
        setHistoryBusy(false);
        setHistoryProgress(null);
      }
    }
  }, [dataset.locator.path, fieldName, frame, probe, steps, zIndex]);

  /** Cancel an in-flight build by bumping generation; bails at the next frame boundary. */
  const cancelHistory = useCallback(() => {
    historyOwnerRef.current.cancel();
    setHistoryBusy(false);
    setHistoryProgress(null);
  }, []);

  const reset = useCallback(() => {
    if (!xCoords?.values || !yCoords?.values) return;
    const x = xCoords.values; const y = yCoords.values;
    setStart({ xKm: x[0] + 0.2 * (x.at(-1)! - x[0]), yKm: y[0] + 0.2 * (y.at(-1)! - y[0]) });
    setEnd({ xKm: x[0] + 0.8 * (x.at(-1)! - x[0]), yKm: y[0] + 0.8 * (y.at(-1)! - y[0]) });
    setProbeFraction(0.5); setZIndex(0); setAnnotations([]); setHistory(null); setSaveState("Reset; saving…");
  }, [xCoords, yCoords]);

  const forget = useCallback(() => {
    if (typeof window !== "undefined") {
      try { clearInvestigationWorkspace(window.localStorage, identity); }
      catch { setSaveState("Local storage unavailable"); return; }
    }
    initializedRef.current = false;
    reset();
    setSaveState("Saved workspace forgotten");
  }, [identity, reset]);

  const addAnnotation = useCallback(() => {
    const text = annotationText.trim();
    if (!text || !probe || simulationTimeS == null || !Number.isFinite(simulationTimeS)) return;
    setAnnotations((current) => current.length >= 100 ? current : [...current, {
      id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${current.length}`,
      text: text.slice(0, 500), fieldName, simulationTimeS, point: probe,
      distanceFraction: probeFraction, zIndex, createdAt: new Date().toISOString(),
    }]);
    setAnnotationText("");
  }, [annotationText, fieldName, probe, probeFraction, simulationTimeS, zIndex]);

  const makeRecipe = useCallback(() => {
    if (!start || !end || !currentStep) return null;
    return {
      schema: INVESTIGATION_WORKSPACE_SCHEMA,
      schemaVersion: INVESTIGATION_WORKSPACE_VERSION,
      datasetIdentity: identity,
      stepIndex: currentStep.index,
      fieldName,
      zIndex,
      start,
      end,
      probeFraction,
      annotations,
      savedAt: new Date().toISOString(),
    } as const;
  }, [annotations, currentStep, end, fieldName, identity, probeFraction, start, zIndex]);

  const exportPublication = useCallback(async () => {
    const recipe = makeRecipe(); const root = workspaceRef.current;
    if (!recipe || !currentStep || !root || !frame || !transect || historyBusy || displayError) return;
    const expectedIdentity = publicationIdentityRef.current;
    const operation = ++publicationOperationRef.current;
    const ownsOperation = () => publicationOperationRef.current === operation && workspaceRef.current === root && root.isConnected;
    setPublishing(true); setPublicationState("Capturing linked panels…");
    try {
      const figure = await composeInvestigationFigure(
        root,
        publicationTitle.trim() || `${dataset.displayName} investigation`,
        `${dataset.displayName} · ${fieldName}${unit ? ` (${unit})` : ""} · t=${formatTime(simulationTimeS)} · level ${zIndex}`,
      );
      if (!ownsOperation()) return;
      if (publicationIdentityRef.current !== expectedIdentity || !figure.current()) throw new Error("The investigation changed during capture. Retry after the views settle.");
      const publication = createInvestigationPublication({
        title: publicationTitle,
        dataset,
        recipe,
        step: currentStep,
        unit,
        figure: {
          filename: "figure.png", mediaType: "image/png", width: figure.width, height: figure.height,
          byteLength: figure.png.length, sha256: await sha256Hex(figure.png), panels: figure.panels,
        },
      });
      const archive = createInvestigationArchive(publication, figure.png);
      if (!ownsOperation()) return;
      if (publicationIdentityRef.current !== expectedIdentity || !figure.current()) throw new Error("The investigation changed before publication. Retry after the views settle.");
      downloadInvestigationArchive(archive, `${safeFilename(publication.title)}.nimbus-figure.zip`);
      setPublicationState(`${figure.panels.length}-panel figure packaged with verified provenance`);
    } catch (reason) {
      if (ownsOperation()) setPublicationState(reason instanceof Error ? reason.message : String(reason));
    } finally { if (ownsOperation()) setPublishing(false); }
  }, [currentStep, dataset, fieldName, makeRecipe, publicationTitle, simulationTimeS, unit, zIndex, frame, transect, historyBusy, displayError]);

  const importPublication = useCallback(async (file: File) => {
    const root = workspaceRef.current;
    if (!root) return;
    const operation = ++publicationOperationRef.current;
    const expectedIdentity = publicationIdentityRef.current;
    const ownsOperation = () => publicationOperationRef.current === operation && workspaceRef.current === root && root.isConnected;
    setPublishing(true);
    setPublicationState("Validating figure package…");
    try {
      const publication = file.name.toLowerCase().endsWith(".json")
        ? parseInvestigationPublication(await file.text())
        : (await parseInvestigationArchive(new Uint8Array(await file.arrayBuffer()))).publication;
      if (!ownsOperation()) return;
      if (publicationIdentityRef.current !== expectedIdentity) throw new Error("The investigation changed while reading the package. Retry after the views settle.");
      if (publication.datasetIdentity !== identity) throw new Error("This figure recipe belongs to a different dataset. Open the exact recorded source first.");
      const recipe = publication.recipe;
      if (!sharedFields.includes(recipe.fieldName)) throw new Error(`The recorded field '${recipe.fieldName}' is not a shared three-dimensional field in this dataset.`);
      const ordinal = steps.findIndex((step) => step.index === recipe.stepIndex);
      if (ordinal < 0) throw new Error(`Recorded source frame ${recipe.stepIndex} is not available.`);
      setFieldName(recipe.fieldName); setZIndex(recipe.zIndex); setStart(recipe.start); setEnd(recipe.end);
      setProbeFraction(recipe.probeFraction); setAnnotations(recipe.annotations); setHistory(null);
      initializedRef.current = true; onStepChange(ordinal); setPublicationTitle(publication.title);
      const saved = saveInvestigationWorkspace(window.localStorage, { ...recipe, savedAt: new Date().toISOString() });
      setSaveState(saved ? "Portable recipe restored" : "Portable recipe restored; local save unavailable");
      setPublicationState("Verified package restored against the exact open dataset");
    } catch (reason) {
      if (ownsOperation()) setPublicationState(reason instanceof Error ? reason.message : String(reason));
    } finally { if (ownsOperation()) setPublishing(false); }
  }, [identity, onStepChange, sharedFields, steps]);

  const currentAnnotations = useMemo(() => simulationTimeS == null ? [] : annotations.filter((annotation) => Math.abs(annotation.simulationTimeS - simulationTimeS) < 1e-6), [annotations, simulationTimeS]);
  const planData = useMemo<Data[]>(() => {
    if (!frame || !start || !end || !probe || !xCoords?.values || !yCoords?.values) return [];
    const slice = slice2D(frame.field, 2, zIndex);
    // Transpose [x][y][z] storage to meteorological screen axes (x horizontal, y vertical).
    const planValues = Array.from({ length: slice.cols }, (_, yIndex) =>
      Array.from({ length: slice.rows }, (_, xIndex) => slice.data[xIndex][yIndex])
    );
    return [
      { type: "heatmap", zsmooth: false, z: planValues, x: xCoords.values, y: yCoords.values, colorscale: resolveColorscale(isDbz ? "dBZ" : "Viridis") as never, zmin: fieldRange[0], zmax: fieldRange[1], colorbar: { title: { text: unit || fieldName, side: "right" }, thickness: 10 }, hovertemplate: "x=%{x:.2f} km<br>y=%{y:.2f} km<br>%{z:.4g}<extra></extra>" },
      { type: "scatter", mode: "text+lines+markers", x: [start.xKm, end.xKm], y: [start.yKm, end.yKm], text: ["A", "B"], textposition: "top center", line: { color: "#f4f0e6", width: 2 }, marker: { color: "#f4f0e6", size: 7 }, hovertemplate: "%{text}<br>x=%{x:.2f} km<br>y=%{y:.2f} km<extra></extra>" },
      { type: "scatter", mode: "markers", x: [probe.xKm], y: [probe.yKm], name: "Linked probe", marker: { color: "#f6bd60", size: 11, symbol: "diamond", line: { color: "#1e2025", width: 1 } }, hovertemplate: "Probe<br>x=%{x:.2f} km<br>y=%{y:.2f} km<extra></extra>" },
      ...(currentAnnotations.length ? [{ type: "scatter", mode: "text+markers", x: currentAnnotations.map((note) => note.point.xKm), y: currentAnnotations.map((note) => note.point.yKm), text: currentAnnotations.map((note) => note.text), textposition: "top right", marker: { color: "#ef8354", size: 8, symbol: "circle-open" }, hovertemplate: "%{text}<extra></extra>" } as Data] : []),
    ];
  }, [currentAnnotations, end, fieldRange, frame, isDbz, probe, start, unit, fieldName, xCoords, yCoords, zIndex]);

  const transectData = useMemo<Data[]>(() => !transect ? [] : [
    { type: "heatmap", zsmooth: false, x: transect.distanceKm, y: transect.zKm, z: transect.values, colorscale: resolveColorscale(isDbz ? "dBZ" : "Viridis") as never, zmin: fieldRange[0], zmax: fieldRange[1], colorbar: { title: { text: unit || fieldName, side: "right" }, thickness: 10 }, hovertemplate: "distance=%{x:.2f} km<br>z=%{y:.2f} km<br>%{z:.4g}<extra></extra>" },
    { type: "scatter", mode: "lines", x: [transect.distanceKm.at(-1)! * probeFraction, transect.distanceKm.at(-1)! * probeFraction], y: [transect.zKm[0], transect.zKm.at(-1)!], line: { color: "#f6bd60", width: 1.5, dash: "dot" }, hoverinfo: "skip", showlegend: false },
  ], [fieldName, fieldRange, isDbz, probeFraction, transect, unit]);

  const profileData = useMemo<Data[]>(() => {
    if (!transect) return [];
    const column = Math.round(probeFraction * (transect.sampleCount - 1));
    return [{ type: "scatter", mode: "lines", x: transect.values.map((row) => row[column]), y: transect.zKm, line: { color: "#f6bd60", width: 2 }, hovertemplate: `${fieldName}=%{x:.4g}<br>z=%{y:.2f} km<extra></extra>` }];
  }, [fieldName, probeFraction, transect]);

  const historyData = useMemo<Data[]>(() => !history ? [] : [{ type: "scatter", connectgaps: false, mode: "lines+markers", x: history.times,
    y: history.values.map(value => Number.isFinite(value) ? value : null), line: { color: "#8ecae6", width: 2 }, marker: { size: 4 },
    hovertemplate: `${history.timeUnit === "s" ? "t=%{x:.3f} s" : "source frame %{x}"}<br>%{y:.4g}<extra></extra>` }], [history]);
  const planLayout = useMemo<Partial<Layout>>(() => ({ margin: { l: 54, r: 52, t: 32, b: 46 }, showlegend: false,
    xaxis: { title: { text: "x (km)" }, ...lockedAspectPartnerAxis("y") }, yaxis: { title: { text: "y (km)" }, ...LOCKED_ASPECT_AXIS }, dragmode: false }), []);
  const transectLayout = useMemo<Partial<Layout>>(() => ({ margin: { l: 54, r: 52, t: 32, b: 46 }, showlegend: false,
    xaxis: { title: { text: "distance from A (km)" } }, yaxis: { title: { text: "z (km)" } } }), []);
  const profileLayout = useMemo<Partial<Layout>>(() => ({ margin: { l: 54, r: 52, t: 32, b: 46 }, showlegend: false,
    xaxis: { title: { text: unit ? `${fieldName} (${unit})` : fieldName } }, yaxis: { title: { text: "z (km)" } } }), [fieldName, unit]);
  const historyLayout = useMemo<Partial<Layout>>(() => {
    const cursor = history?.timeUnit === "s" ? simulationTimeS : currentStep?.index;
    return { margin: { l: 54, r: 52, t: 32, b: 46 }, showlegend: false,
      xaxis: { title: { text: history?.timeTitle ?? "source frame index" } },
      yaxis: { title: { text: unit ? `${fieldName} (${unit})` : fieldName } },
      shapes: cursor == null ? [] : [{ type: "line", x0: cursor, x1: cursor, y0: 0, y1: 1, yref: "paper", line: { color: "#f6bd60", width: 1.5, dash: "dot" } }] };
  }, [history?.timeUnit, history?.timeTitle, simulationTimeS, currentStep?.index, unit, fieldName]);

  return (
    <section ref={workspaceRef} className="investigation" aria-label="Linked transect probe and annotation workspace">
      <input ref={publicationInputRef} hidden type="file" accept="application/zip,.zip,.json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importPublication(file); event.currentTarget.value = ""; }} />
      <header className="investigation__masthead">
        <div><span className="investigation__eyebrow">FIELD NOTEBOOK · READ-ONLY ANALYSIS</span><h3>Linked investigation workspace</h3><p>One physical point and one selected time connect every view. Click the plan after choosing what to place.</p></div>
        <div className="investigation__save"><strong>{saveState}</strong><span>Private to this dataset on this device</span><button type="button" aria-expanded={publicationOpen} onClick={() => setPublicationOpen((open) => !open)}>Publication</button></div>
      </header>
      <div className="investigation__controls">
        <label>Field<select disabled={sharedFields.length === 0} value={fieldName} onChange={(event) => { setFieldName(event.target.value); setHistory(null); }}>{sharedFields.map((field) => <option key={field}>{field}</option>)}</select></label>
        <label className="investigation__range-control">Level<input aria-label="Investigation level" type="range" min={0} max={Math.max(0, (frame?.field.shape[2] ?? 1) - 1)} value={zIndex} onChange={(event) => { setZIndex(Number(event.target.value)); setHistory(null); }} /><span>{zCoords?.values?.[zIndex]?.toFixed(2) ?? zIndex} {zCoords?.values ? "km" : ""}</span></label>
        <label className="investigation__range-control">Time<input aria-label="Linked investigation time" type="range" min={0} max={Math.max(0, steps.length - 1)} value={stepIdx} onChange={(event) => onStepChange(Number(event.target.value))} /><span>{formatTime(simulationTimeS)}</span></label>
        <div className="investigation__placement" aria-label="Plan placement mode">{(["start", "end", "probe"] as const).map((mode) => <button type="button" className={placementMode === mode ? "is-active" : ""} onClick={() => setPlacementMode(mode)} key={mode}>{mode === "start" ? "Place A" : mode === "end" ? "Place B" : "Place probe"}</button>)}</div>
        <button type="button" onClick={reset}>Reset</button><button type="button" onClick={forget}>Forget saved workspace</button>
      </div>
      {publicationOpen && <section className="investigation__publication" aria-label="Investigation figure publication">
        <div><span className="investigation__eyebrow">PORTABLE EVIDENCE</span><h4>Figure and provenance, inseparable</h4><p>Exports one ZIP containing a composed PNG, SHA-256-bound provenance, exact source-frame identity, interpolation method, annotations, and the restorable investigation recipe. Source arrays are referenced rather than copied.</p></div>
        <label>Figure title<input aria-label="Publication figure title" value={publicationTitle} maxLength={300} onChange={(event) => setPublicationTitle(event.target.value)} /></label>
        <div className="investigation__publication-actions"><button type="button" disabled={publishing || !frame || !transect || historyBusy || displayError != null} onClick={() => void exportPublication()}>{publishing ? "Packaging…" : "Export figure package"}</button><button type="button" disabled={publishing} onClick={() => publicationInputRef.current?.click()}>Restore package…</button></div>
        <div className="investigation__publication-boundary"><strong>Replay boundary</strong><span>Exact interactive restore is supported against the recorded dataset identity. Automated headless batch rendering is not yet implemented.</span></div>
        {publicationState && <output role="status">{publicationState}</output>}
      </section>}
      {sharedFields.length === 0 && <div className="investigation__error" role="status">No three-dimensional field is shared by every frame. This workspace will not offer a field that cannot produce the linked transect.</div>}
      {displayError && <div className="investigation__error" role="alert">{displayError}</div>}
      {fieldName && !frame && !displayError && <div className="investigation__loading" role="status">Loading the selected physical field…</div>}
      {frame && start && end && probe && transect && (
        <>
          <div className="investigation__coordinate-strip">
            <span><strong>A</strong> x {start.xKm.toFixed(2)}, y {start.yKm.toFixed(2)} km</span><span><strong>B</strong> x {end.xKm.toFixed(2)}, y {end.yKm.toFixed(2)} km</span><span><strong>Probe</strong> x {probe.xKm.toFixed(2)}, y {probe.yKm.toFixed(2)} km</span><span><strong>Time</strong> {formatTime(simulationTimeS)}</span><span><strong>Transect</strong> {transect.distanceKm.at(-1)!.toFixed(2)} km · {transect.sampleCount} samples</span>
          </div>
          <div className="investigation__primary-grid">
            <figure className="investigation__figure investigation__figure--plan"><figcaption><strong>Plan</strong><span>Click to {placementMode === "start" ? "place A" : placementMode === "end" ? "place B" : "move the linked probe"}</span></figcaption><PlotCanvas className="investigation__plan-canvas" data={planData} onClick={handlePlanClick} layout={planLayout} /></figure>
            <figure className="investigation__figure investigation__figure--transect"><figcaption><strong>A → B vertical transect</strong><span>Bilinear x/y interpolation · native z levels</span></figcaption><PlotCanvas className="investigation__transect-canvas" data={transectData} layout={transectLayout} /></figure>
          </div>
          <div className="investigation__secondary-grid">
            <figure className="investigation__figure"><figcaption><strong>Linked vertical profile</strong><span>{Math.round(probeFraction * 100)}% from A to B</span></figcaption><PlotCanvas className="investigation__profile-canvas" data={profileData} layout={profileLayout} /></figure>
            <figure className="investigation__figure"><figcaption><strong>Linked probe history</strong><span>{historyBusy ? historyProgress ?? "Starting…" : "Load on request to keep browsing responsive"}</span></figcaption>{history ? <PlotCanvas className="investigation__history-canvas" data={historyData} layout={historyLayout} /> : <div className="investigation__history-empty"><p>Build an exact value history at the current probe and level across all {steps.length} frames.</p><div className="investigation__history-actions"><button type="button" onClick={() => void buildHistory()} disabled={historyBusy} title={historyBusy ? "A probe history is already being built" : `Read all ${steps.length} frames at this probe`}>{historyBusy ? "Building…" : "Build probe history"}</button>{historyBusy && <button type="button" className="investigation__history-cancel" onClick={cancelHistory} title="Stop after the frame currently being read">Cancel</button>}</div></div>}</figure>
          </div>
          <section className="investigation__notes" aria-label="Workspace annotations">
            <div className="investigation__notes-compose"><div><span className="investigation__eyebrow">ANNOTATIONS</span><h4>Record what matters here</h4><p>Notes retain field, simulation time, level, and exact physical probe coordinates. {annotations.length}/100 notes.</p></div><div><textarea aria-label="Annotation text" maxLength={500} value={annotationText} onChange={(event) => setAnnotationText(event.target.value)} placeholder="e.g. Rear-flank notch becomes coherent…"/><button type="button" disabled={!annotationText.trim() || annotations.length >= 100 || simulationTimeS == null} title={simulationTimeS == null ? "Timestamped annotations require a declared physical simulation time" : undefined} onClick={addAnnotation}>Add at current probe</button></div></div>
            {annotations.length === 0 ? <p className="investigation__notes-empty">No annotations yet.</p> : <ol>{annotations.map((annotation) => <li key={annotation.id}><div><strong>{annotation.text}</strong><span>{annotation.fieldName} · t={formatTime(annotation.simulationTimeS)} · level {annotation.zIndex} · x {annotation.point.xKm.toFixed(2)}, y {annotation.point.yKm.toFixed(2)} km</span></div><button type="button" aria-label={`Delete annotation ${annotation.text}`} onClick={() => setAnnotations((current) => current.filter((note) => note.id !== annotation.id))}>Delete</button></li>)}</ol>}
          </section>
        </>
      )}
    </section>
  );
}
