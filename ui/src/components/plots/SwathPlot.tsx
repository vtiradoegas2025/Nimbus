import { useEffect, useMemo, useRef, useState } from "react";
import swathAccumulatorWorkerUrl from "@/lib/plot/swathAccumulator.worker.ts?worker&url";
import type { Data, Layout } from "plotly.js";
import { readFieldAny, registeredFieldRank, type GridInfo } from "@/lib/run/runOutputs";
import { labelWithUnit, type FieldUnits } from "@/lib/science/fieldUnits";
import { fieldsAcrossStepsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";
import { axisCoords, axisLabel, planeSupportsLockedAspect } from "@/lib/science/gridCoords";
import { cropToValidPlotDomain } from "@/lib/science/validDomain";
import {
  plotlyColorRange,
  resolveColorscale,
  type ExtendedColorscale,
} from "@/lib/plot/plotColormaps";
import { defaultScalarDisplay } from "@/lib/plot/scalarDisplayDefaults";
import { displayedRange } from "@/lib/plot/displayedRange";
import { canonicalHistoryUnit, historyFrameKey, historySourcesAreAppendOnly, historyUnitMismatch } from "@/lib/plot/historyReductionCache";
import { canAccumulateByCell } from "@/lib/run/frameMotion";
import {
  describeEarthSwath,
  earthCellOffset,
  earthSwathGrid,
  SWATH_3D_SOURCE_CELL_CAP,
  swathAccumulatorBudgetError,
  swathFrameGeometry,
  validGridCellCount,
  type EarthSwathGrid,
} from "@/lib/plot/earthSwath";
import {
  swathProgressSnapshotCount,
  type SwathAccumulatorReply,
  type SwathAccumulatorRequest,
} from "@/lib/plot/swathAccumulatorProtocol";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  ControlGroup,
  FIT_OPTIONS,
  fitAxisFragments,
  PillGroup,
  type PlotChrome,
  type PlotComponentProps,
  type PlotFitMode,
  uniformFieldAnnotation,
  usePlotChrome,
} from "./common";
import { PlotCompressionControls, RootHistoryScopeNotice } from "./PlotSourceDisclosure";

/* Per-cell run-max/min swath. 2D column fields (composite_reflectivity etc.) preferred — 3D fields column-reduce first. */

type SwathReducer = "max" | "min";
type SwathRows = ReadonlyArray<Float64Array>;
const REDUCERS: ReadonlyArray<{ value: SwathReducer; label: string }> = [
  { value: "max", label: "max" },
  { value: "min", label: "min" },
];

interface SwathWorkerClient {
  configKey: string;
  worker: Worker;
  ready: Promise<void>;
  rejectReady: (error: Error) => void;
  control: { resolve: () => void; reject: (error: Error) => void } | null;
  pending: Map<number, { resolve: (snapshot: Float64Array | null) => void; reject: (error: Error) => void }>;
  processing: Promise<void>;
  initialized: boolean;
  closed: boolean;
  accWidth: number;
  accHeight: number;
  earthGrid: EarthSwathGrid | null;
  frameShape: readonly [number, number] | null;
  processed: Set<string>;
  sourceKeys: Set<string>;
  accumulated: number;
  nextRequestId: number;
  latestSnapshot: { values: Float64Array; grid: GridInfo | null; frames: number } | null;
}

let swathWorkerGeneration = 0;

function createSwathWorkerClient(configKey: string): SwathWorkerClient {
  // WebKit can reject restarted COEP module workers served from conditional cache.
  const separator = swathAccumulatorWorkerUrl.includes("?") ? "&" : "?";
  const worker = new Worker(`${swathAccumulatorWorkerUrl}${separator}generation=${++swathWorkerGeneration}`, {
    type: "module",
    name: "nimbus-swath-accumulator",
  });
  let resolveReady!: () => void;
  let rejectReady!: (error: Error) => void;
  const ready = new Promise<void>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  const client: SwathWorkerClient = {
    configKey,
    worker,
    ready,
    rejectReady,
    control: null,
    pending: new Map(),
    processing: Promise.resolve(),
    initialized: false,
    closed: false,
    accWidth: 0,
    accHeight: 0,
    earthGrid: null,
    frameShape: null,
    processed: new Set(),
    sourceKeys: new Set(),
    accumulated: 0,
    nextRequestId: 0,
    latestSnapshot: null,
  };
  worker.onmessage = (event: MessageEvent<SwathAccumulatorReply>) => {
    const reply = event.data;
    if (reply.kind === "ready") {
      resolveReady();
      return;
    }
    if (reply.kind === "initialized" || reply.kind === "resized") {
      const control = client.control;
      client.control = null;
      control?.resolve();
      return;
    }
    if (reply.requestId === -1 && !reply.ok) {
      const control = client.control;
      client.control = null;
      control?.reject(new Error(reply.error));
      return;
    }
    const waiter = client.pending.get(reply.requestId);
    if (!waiter) return;
    client.pending.delete(reply.requestId);
    if (reply.ok) waiter.resolve(reply.snapshot);
    else waiter.reject(new Error(reply.error));
  };
  worker.onerror = (event) => {
    const failure = new Error(event.message || "Swath accumulator worker failed");
    rejectReady(failure);
    client.control?.reject(failure);
    client.control = null;
    for (const waiter of client.pending.values()) waiter.reject(failure);
    client.pending.clear();
  };
  return client;
}

function closeSwathWorkerClient(client: SwathWorkerClient): void {
  if (client.closed) return;
  client.closed = true;
  const failure = new Error("Swath accumulation cancelled");
  client.rejectReady(failure);
  client.control?.reject(failure);
  client.control = null;
  for (const waiter of client.pending.values()) waiter.reject(failure);
  client.pending.clear();
  client.worker.terminate();
}

function sendSwathControl(
  client: SwathWorkerClient,
  request: Extract<SwathAccumulatorRequest, { kind: "initialize" | "resize" }>,
): Promise<void> {
  if (client.closed) return Promise.reject(new Error("Swath accumulator is closed"));
  if (client.control) return Promise.reject(new Error("Swath accumulator control request overlaps another request"));
  return new Promise<void>((resolve, reject) => {
    client.control = { resolve, reject };
    client.worker.postMessage(request);
  });
}

function accumulateSwathOffThread(
  client: SwathWorkerClient,
  request: Omit<Extract<SwathAccumulatorRequest, { kind: "accumulate" }>, "requestId">,
): Promise<Float64Array | null> {
  if (client.closed) return Promise.reject(new Error("Swath accumulator is closed"));
  const requestId = ++client.nextRequestId;
  return new Promise((resolve, reject) => {
    client.pending.set(requestId, { resolve, reject });
    client.worker.postMessage(
      { ...request, requestId },
      [request.data.buffer as ArrayBuffer],
    );
  });
}

const COLORMAPS: ReadonlyArray<{ value: ExtendedColorscale; label: string }> = [
  { value: "Viridis", label: "Viridis" },
  { value: "Plasma", label: "Plasma" },
  { value: "Inferno", label: "Inferno" },
  { value: "RdBu", label: "RdBu" },
  { value: "Greys", label: "Greys" },
  { value: "dBZ", label: "NWS dBZ" },
  { value: "ColdPool", label: "Cold pool (BuRd)" },
];

/** Column fields that make the canonical swaths; preferred as the default. */
const SWATH_PREFER = [
  "composite_reflectivity",
  "column_max_w",
  "column_max_vorticity",
  "reflectivity_dbz",
];

function pickField(available: string[], prefer: string[]): string | null {
  const lower = available.map((f) => f.toLowerCase());
  for (const p of prefer) {
    const i = lower.indexOf(p.toLowerCase());
    if (i >= 0) return available[i];
  }
  return null;
}

function swathColormap(field: string | null, fieldUnits: FieldUnits): ExtendedColorscale {
  return defaultScalarDisplay(field, fieldUnits).colormap;
}

function unitsForField(
  steps: PlotComponentProps["steps"],
  field: string | null,
): FieldUnits {
  return steps.find((step) => field != null && step.fields.includes(field))?.fieldUnits;
}

export function SwathPlot({ runDir, steps, stepIdx, preset, onPresetCleared }: PlotComponentProps) {
  const chrome = usePlotChrome();
  const allFields = useMemo(
    () => fieldsAcrossStepsForPlotRanks(runDir, steps, [2, 3]),
    [runDir, steps],
  );

  const [fieldName, setFieldName] = useState<string | null>(
    preset?.field ?? pickField(allFields, SWATH_PREFER) ?? allFields[0] ?? null,
  );
  const [reducer, setReducer] = useState<SwathReducer>("max");
  const [colormap, setColormap] = useState<ExtendedColorscale>(
    (preset?.colormap as ExtendedColorscale | undefined) ??
      swathColormap(
        preset?.field ?? fieldName,
        unitsForField(steps, preset?.field ?? fieldName),
      ),
  );
  // See PlotFitMode: a swath's tilt is read off the plot, so true shape leads.
  const [fitMode, setFitMode] = useState<PlotFitMode>("true-shape");
  const [acc, setAcc] = useState<SwathRows | null>(null);
  const [accGrid, setAccGrid] = useState<GridInfo | null>(null);
  const [accumulatedFrames, setAccumulatedFrames] = useState(0);
  const [loadingPct, setLoadingPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previousResultPending, setPreviousResultPending] = useState(false);
  const publishedResultRef = useRef<{ configKey: string; values: Float64Array } | null>(null);
  const workerClientRef = useRef<SwathWorkerClient | null>(null);
  const referenceStep = useMemo(
    () => steps.find((step) => fieldName != null && step.fields.includes(fieldName)) ?? null,
    [steps, fieldName],
  );
  const fieldUnits = referenceStep?.fieldUnits;
  const referenceRank = fieldName && referenceStep
    ? registeredFieldRank(
        runDir, referenceStep.index, fieldName, referenceStep.publicationGeneration,
      )
    : null;

  const clearPreset = () => onPresetCleared?.();

  // Apply a preset (field + colormap) on id change.
  useEffect(() => {
    if (!preset) return;
    if (preset.field) setFieldName(preset.field);
    if (preset.colormap) setColormap(preset.colormap as ExtendedColorscale);
  }, [preset]);

  // Reselect the field if the current one disappears (e.g. another run).
  useEffect(() => {
    if (fieldName && allFields.includes(fieldName)) return;
    const next = pickField(allFields, SWATH_PREFER) ?? allFields[0] ?? null;
    setFieldName(next);
    setColormap(swathColormap(next, unitsForField(steps, next)));
  }, [allFields, fieldName, steps]);

  // A swath is the one plot that combines many frames by grid CELL, so it is
  // the one plot a translating domain silently invalidates: the same index is a
  // different Earth location in each frame, and the result would smear the
  // track along the direction of motion while still looking like a storm swath.
  // See docs/domain.md and lib/run/frameMotion.ts.
  const accumulation = useMemo(() => canAccumulateByCell(steps), [steps]);

  // When the domain moves, accumulate into a grid fixed to the EARTH that
  // covers every frame's footprint, and place each frame at its own origin.
  // For a stationary run this resolves to exactly one footprint at offset
  // (0, 0), so the arithmetic below is unchanged. See lib/plot/earthSwath.ts.
  const earthGrid = useMemo(() => {
    const fieldSteps = steps.filter(
      (step) => fieldName != null && step.fields.includes(fieldName),
    );
    const geometry = fieldSteps.map(
      (step) => swathFrameGeometry(step.grid, step.motion?.earthOriginM),
    );
    if (geometry.length === 0 || geometry.some((frame) => frame == null)) return null;
    return earthSwathGrid(geometry as NonNullable<(typeof geometry)[number]>[]);
  }, [steps, fieldName]);
  const earthMoved = !accumulation.allowed && earthGrid != null;

  useEffect(() => () => {
    if (workerClientRef.current) closeSwathWorkerClient(workerClientRef.current);
    workerClientRef.current = null;
  }, []);

  // Append only newly published frames to one persistent worker accumulator.
  // Moving-domain growth resizes and shifts the Earth-fixed grid in place, so
  // old fields are never replayed and historical planes are never retained.
  useEffect(() => {
    if (!accumulation.allowed && earthGrid == null) {
      publishedResultRef.current = null;
      setPreviousResultPending(false);
      // Refuse only when the frames cannot be placed in Earth coordinates
      // either — an unrecognised mode, a resolution change, a non-finite
      // origin. A translating run with coherent geometry is accumulated for
      // real rather than blocked.
      setAcc(null);
      setAccGrid(null);
      setAccumulatedFrames(0);
      setLoadingPct(null);
      if (workerClientRef.current) closeSwathWorkerClient(workerClientRef.current);
      workerClientRef.current = null;
      return;
    }
    if (!fieldName || steps.length === 0) {
      publishedResultRef.current = null;
      setPreviousResultPending(false);
      setAcc(null);
      setAccGrid(null);
      setAccumulatedFrames(0);
      setLoadingPct(null);
      if (workerClientRef.current) closeSwathWorkerClient(workerClientRef.current);
      workerClientRef.current = null;
      return;
    }
    let cancelled = false;
    const isMax = reducer === "max";
    const unitError = historyUnitMismatch(steps, fieldName);
    if (unitError) {
      publishedResultRef.current = null;
      setPreviousResultPending(false);
      if (workerClientRef.current) closeSwathWorkerClient(workerClientRef.current);
      workerClientRef.current = null; setAcc(null); setAccGrid(null); setAccumulatedFrames(0);
      setLoadingPct(null); setError(unitError); return;
    }
    const eligibleSteps = steps.filter((step) => step.fields.includes(fieldName));
    const frameKey = (step: (typeof steps)[number]) => historyFrameKey(step, fieldName);
    const sourceKeys = new Set(eligibleSteps.map(frameKey));
    const configKey = JSON.stringify([runDir, fieldName, reducer, canonicalHistoryUnit(fieldUnits?.[fieldName])]);
    let client = workerClientRef.current;
    let retainPrevious = false;
    if (!client || client.configKey !== configKey || client.closed
        || !historySourcesAreAppendOnly(client.sourceKeys, sourceKeys)) {
      if (client) closeSwathWorkerClient(client);
      client = createSwathWorkerClient(configKey);
      workerClientRef.current = client;
      retainPrevious = publishedResultRef.current?.configKey === configKey;
      setPreviousResultPending(retainPrevious);
      if (!retainPrevious) {
        publishedResultRef.current = null;
        setAcc(null); setAccGrid(null); setAccumulatedFrames(0);
      }
      setError(null);
    }
    client.sourceKeys = sourceKeys;
    const ownedClient = client;
    const publishSnapshot = (snapshot: NonNullable<SwathWorkerClient["latestSnapshot"]>) => {
      if (cancelled) return;
      publishedResultRef.current = { configKey, values: snapshot.values };
      setAcc(rowsFromFlatSnapshot(snapshot.values, ownedClient.accHeight, ownedClient.accWidth));
      setAccGrid(snapshot.grid); setAccumulatedFrames(snapshot.frames); setPreviousResultPending(false);
    };
    const pendingCount = eligibleSteps.filter((step) => !ownedClient.processed.has(frameKey(step))).length;
    setLoadingPct(pendingCount > 0
      ? (ownedClient.processed.size / Math.max(1, eligibleSteps.length)) * 100
      : null);
    setError(null);

    const processNewFrames = async () => {
      if (cancelled || ownedClient.closed) return;
      await ownedClient.ready;
      if (cancelled || ownedClient.closed) return;
      const pendingSteps = eligibleSteps.filter((step) => !ownedClient.processed.has(frameKey(step)));
      if (pendingSteps.length === 0 && ownedClient.latestSnapshot) publishSnapshot(ownedClient.latestSnapshot);

      if (ownedClient.initialized && earthGrid) {
        const previous = ownedClient.earthGrid;
        if (!previous) {
          throw new Error("A swath cannot acquire Earth-grid geometry after accumulation has started.");
        }
        const changed = previous.nx !== earthGrid.nx || previous.ny !== earthGrid.ny
          || previous.originM[0] !== earthGrid.originM[0]
          || previous.originM[1] !== earthGrid.originM[1];
        if (changed) {
          if (previous.dxM !== earthGrid.dxM || previous.dyM !== earthGrid.dyM) {
            throw new Error("A live swath changed grid spacing and cannot be extended exactly.");
          }
          const shiftR = Math.round((previous.originM[0] - earthGrid.originM[0]) / earthGrid.dxM);
          const shiftT = Math.round((previous.originM[1] - earthGrid.originM[1]) / earthGrid.dyM);
          const budgetError = swathAccumulatorBudgetError(earthGrid.nx, earthGrid.ny);
          if (budgetError) throw new Error(budgetError);
          await sendSwathControl(ownedClient, {
            kind: "resize",
            accWidth: earthGrid.ny,
            accHeight: earthGrid.nx,
            shiftR,
            shiftT,
          });
          ownedClient.accWidth = earthGrid.ny;
          ownedClient.accHeight = earthGrid.nx;
          ownedClient.earthGrid = earthGrid;
        }
      }

      const snapshotEvery = Math.max(
        1,
        Math.ceil(pendingSteps.length / swathProgressSnapshotCount(
          Math.max(1, (earthGrid?.nx ?? ownedClient.accHeight) * (earthGrid?.ny ?? ownedClient.accWidth)),
        )),
      );
      for (let pendingIndex = 0; pendingIndex < pendingSteps.length; pendingIndex += 1) {
        if (cancelled) return;
        const step = pendingSteps[pendingIndex];
        const key = frameKey(step);
        const rank = registeredFieldRank(
          runDir,
          step.index,
          fieldName,
          step.publicationGeneration,
        );
        if (rank === 3) {
          const sourceCells = validGridCellCount(step.grid);
          if (sourceCells == null || sourceCells > SWATH_3D_SOURCE_CELL_CAP) {
            if (!cancelled) {
              setError(
                sourceCells == null
                  ? "This 3D swath has no trustworthy source-size metadata. Use a rank-2 column diagnostic."
                  : `This exact 3D swath must inspect ${sourceCells.toLocaleString()} cells per frame; `
                    + `the interactive limit is ${SWATH_3D_SOURCE_CELL_CAP.toLocaleString()}. `
                    + "Use a rank-2 column maximum/minimum diagnostic or export the full-resolution swath.",
              );
              setLoadingPct(null);
            }
            return;
          }
        }
        let rawField;
        try {
          rawField = await readFieldAny(
            runDir,
            step.index,
            fieldName,
            step.publicationGeneration,
            {
              priority: "background",
              cacheRetention: "transient",
              ownership: "transferable",
            },
          );
        } catch (e) {
          if (!cancelled) {
            setError(e instanceof Error ? e.message : String(e));
            setLoadingPct(null);
          }
          return;
        }
        if (cancelled) return;

        const domain = cropToValidPlotDomain(rawField, step.grid);
        const field = domain.field;
        const shape = field.shape;
        if (shape.length !== 2 && shape.length !== 3) continue;
        const lnr = shape[0];
        const lnth = shape[1];
        if (!ownedClient.initialized) {
          const accWidth = earthGrid?.ny ?? lnth;
          const accHeight = earthGrid?.nx ?? lnr;
          const budgetError = swathAccumulatorBudgetError(accHeight, accWidth);
          if (budgetError) {
            if (!cancelled) {
              setError(budgetError);
              setLoadingPct(null);
            }
            return;
          }
          await sendSwathControl(ownedClient, {
            kind: "initialize",
            accWidth,
            accHeight,
            isMax,
          });
          ownedClient.initialized = true;
          ownedClient.accWidth = accWidth;
          ownedClient.accHeight = accHeight;
          ownedClient.earthGrid = earthGrid;
          ownedClient.frameShape = [lnr, lnth];
        } else if (lnr !== ownedClient.frameShape?.[0] || lnth !== ownedClient.frameShape?.[1]) {
          ownedClient.processed.add(key);
          continue;
        }

        const croppedGeometry = earthGrid
          ? swathFrameGeometry(domain.grid, null, [lnr, lnth])
          : null;
        if (earthGrid && croppedGeometry == null) continue;
        const [offsetR, offsetT] = earthGrid && croppedGeometry
          ? earthCellOffset(croppedGeometry, earthGrid)
          : [0, 0];
        const shouldSnapshot = (pendingIndex + 1) % snapshotEvery === 0
          || pendingIndex === pendingSteps.length - 1;
        const flatSnapshot = await accumulateSwathOffThread(ownedClient, {
          kind: "accumulate",
          data: field.data,
          shape,
          nr: lnr,
          nth: lnth,
          offsetR,
          offsetT,
          snapshot: shouldSnapshot,
        });
        ownedClient.processed.add(key);
        ownedClient.accumulated += 1;
        if (flatSnapshot) ownedClient.latestSnapshot = { values: flatSnapshot,
          grid: accumulatorGrid(domain.grid as GridInfo | null, accumulation.allowed ? null : ownedClient.earthGrid),
          frames: ownedClient.accumulated };
        if (!cancelled) {
          setLoadingPct(((pendingIndex + 1) / Math.max(1, pendingSteps.length)) * 100);
          if (flatSnapshot && ownedClient.latestSnapshot
              && (!retainPrevious || pendingIndex === pendingSteps.length - 1)) publishSnapshot(ownedClient.latestSnapshot);
        }
      }
      if (!cancelled) setLoadingPct(null);
    };

    ownedClient.processing = ownedClient.processing.then(processNewFrames).catch((cause: unknown) => {
      if (!cancelled) {
        setError(cause instanceof Error ? cause.message : String(cause));
        setLoadingPct(null);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [runDir, steps, fieldName, fieldUnits, reducer, accumulation.allowed, earthGrid]);

  const plot = useMemo(
    () => buildSwath(acc, fieldName, reducer, colormap, accGrid, fieldUnits, fitMode, chrome),
    [acc, fieldName, reducer, colormap, accGrid, fieldUnits, fitMode, chrome],
  );

  return (
    <>
      <RootHistoryScopeNotice
        runDir={runDir}
        step={steps[stepIdx] ?? null}
        label="Root history · Earth-coordinate swath"
      />
      <div className="plot-controls">
        <ControlGroup label="Field">
          <select
            aria-label="Field"
            value={fieldName ?? ""}
            onChange={(e) => {
              const f = e.target.value || null;
              setFieldName(f);
              setColormap(swathColormap(f, unitsForField(steps, f)));
              clearPreset();
            }}
          >
            {allFields.length === 0 && <option value="">(no fields)</option>}
            {allFields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </ControlGroup>
        <PlotCompressionControls steps={steps} fields={[fieldName]} />

        <ControlGroup label="Over run">
          <PillGroup
            options={REDUCERS}
            value={reducer}
            onChange={(v) => {
              setReducer(v);
              clearPreset();
            }}
          />
        </ControlGroup>

        <ControlGroup label="Colormap">
          <select
            value={colormap}
            onChange={(e) => {
              setColormap(e.target.value as ExtendedColorscale);
              clearPreset();
            }}
          >
            {COLORMAPS.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </ControlGroup>

        {plot?.aspectLockable && (
          <ControlGroup label="Fit">
            <PillGroup options={FIT_OPTIONS} value={fitMode} onChange={setFitMode} />
          </ControlGroup>
        )}

        <ControlGroup label="Frames" stretch>
          <span className="plot-controls__status">
            {loadingPct != null
              ? `${loadingPct.toFixed(0)}%`
              : acc
                ? `${accumulatedFrames} accumulated`
                : "—"}
          </span>
        </ControlGroup>
      </div>

      <div className="plot-body">
        {loadingPct != null && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>{previousResultPending ? "Displaying previous completed swath · recomputing" : "Accumulating"} {loadingPct.toFixed(0)}%</span>
          </div>
        )}
        {error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{error}</span>
          </div>
        )}
        {earthMoved && !acc && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>{describeEarthSwath(earthGrid, true)}</span>
          </div>
        )}
        {acc && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>
              {previousResultPending ? "previous completed swath · " : ""}native horizontal cells · exact {reducer} over {accumulatedFrames} frames
              {referenceRank === 3 ? ` · exact ${reducer} over each vertical column first` : ""}
              {earthMoved ? ` · ${describeEarthSwath(earthGrid, true)}` : ""}
            </span>
          </div>
        )}
        {accumulation.blocker && !earthMoved && (
          <div className="plot-overlay plot-overlay--warn">
            <Icon name="warning" size={12} />
            <span>{accumulation.blocker}</span>
          </div>
        )}
        {plot ? (
          <PlotCanvas data={plot.data} layout={plot.layout} />
        ) : (
          !error &&
          loadingPct == null && (
            <div className="plot-overlay">
              <span>No data to plot.</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

/** Fold one step into the accumulator: a 2D field directly, a 3D field column-
 *  reduced over z with the same reducer. NaN cells are skipped. */
/** Manifest geometry for one frame, in the shape earthSwath expects. */
/**
 * The grid describing the accumulator, which is not the grid of any one frame.
 *
 * A moving-domain swath accumulates into the union of every footprint, anchored at the minimum
 * origin over ALL frames (earthSwathGrid). The first frame processed is first in TIME, not
 * necessarily the one at that minimum corner, so labelling the axes with its origin shifts the whole
 * swath whenever the domain does not simply move away from where it started. Overriding the anchor
 * (and dropping the per-frame exact coordinates, which describe one footprint) keeps the axes honest.
 */
function accumulatorGrid(
  frameGrid: GridInfo | null,
  earthGrid: EarthSwathGrid | null,
): GridInfo | null {
  if (!frameGrid || !earthGrid) return frameGrid;
  const zOrigin = frameGrid.scalarOriginM?.[2];
  return {
    ...frameGrid,
    scalarCoordinatesM: undefined,
    scalarOriginM: [
      earthGrid.originM[0],
      earthGrid.originM[1],
      Number.isFinite(zOrigin) ? zOrigin as number : 0,
    ],
    drM: earthGrid.dxM,
    dyM: earthGrid.dyM,
  };
}

/** Zero-copy row views over the worker's transferred exact snapshot. */
function rowsFromFlatSnapshot(flat: Float64Array, nr: number, nth: number): SwathRows {
  const out: Float64Array[] = new Array(nr);
  for (let r = 0; r < nr; r++) {
    out[r] = flat.subarray(r * nth, (r + 1) * nth);
  }
  return out;
}

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
  /** Whether Fit applies to this plane; see SlicePlot's BuiltPlot. */
  aspectLockable: boolean;
}

function buildSwath(
  acc: SwathRows | null,
  fieldName: string | null,
  reducer: SwathReducer,
  colormap: ExtendedColorscale,
  grid: GridInfo | null,
  fieldUnits: FieldUnits,
  fitMode: PlotFitMode,
  chrome: PlotChrome,
): BuiltPlot | null {
  if (!acc || acc.length === 0 || !fieldName) return null;
  const nr = acc.length;
  const nth = acc[0]?.length ?? 0;
  if (nth === 0) return null;

  // Swath is a plan view: rows = r/x (axis 0), cols = θ/y (axis 1).
  const xCoords = axisCoords(1, nth, grid);
  const yCoords = axisCoords(0, nr, grid);
  // max/min preserve the field's units, so carry them onto the swath label.
  const label = `${reducer}-over-run ${labelWithUnit(fieldName, fieldUnits)}`;
  const range = displayedRange(acc);
  const colorRange = plotlyColorRange(colormap, range);

  const data: Data[] = [
    {
      type: "heatmap",
      // A swath is a cellwise reduction; interpolating it would imply values
      // between those independently reduced cells.
      zsmooth: false,
      z: acc as unknown as number[][],
      x: xCoords.values,
      y: yCoords.values,
      colorscale: resolveColorscale(colormap) as never,
      ...colorRange,
      colorbar: { title: { text: label, side: "right" } },
      hovertemplate:
        `${axisLabel(0, grid)}=%{y${yCoords.hoverFmt}}` +
        `<br>${axisLabel(1, grid)}=%{x${xCoords.hoverFmt}}` +
        `<br>${label}=%{z:.4g}<extra></extra>`,
    },
  ];

  // A swath is always a horizontal plan view (axes 0 and 1), so lock 1:1: the
  // tilt of a swath is read straight off the plot, and a panel-shaped stretch
  // rotates it. See planeSupportsLockedAspect.
  const lockAspect = planeSupportsLockedAspect(0, 1, yCoords, xCoords);
  const fit = fitAxisFragments(lockAspect, fitMode);
  const uniform = uniformFieldAnnotation(range, label, chrome);
  const layout: Partial<Layout> = {
    xaxis: { title: { text: xCoords.title }, ...fit.x },
    yaxis: { title: { text: yCoords.title }, ...fit.y },
    title: { text: `${label} (swath)`, font: { size: 13 } },
    annotations: uniform ? [uniform] : [],
  };

  return { data, layout, aspectLockable: lockAspect };
}
