import { useEffect, useRef, useState } from "react";
import {
  floorRequestId,
  floorWindowRequest,
  planFloorHyperslab,
  resolveFloorSource,
  type FloorReduction,
  type FloorSource,
} from "@/lib/plot/radarFloor";
import { fieldWindowAggregation } from "@/lib/plot/fieldLoadPlanning";
import { MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES } from "@/lib/dataset/nativeFieldWindow";
import type { SignedFieldStats } from "@/lib/plot/coherentStructures";
import type { TerrainFloorMesh } from "@/lib/plot/radarFloor";
import { readFieldAny, fieldDefinitionIdentity } from "@/lib/run/runOutputs";
import { fieldSupportsBoundedWindows } from "@/lib/plot/fieldLoadPlanning";
import {
  RadarFloorPreparationService,
  type RetainedRadarFloorPreparation,
} from "./radarFloorPreparationClient";
import type { RadarFloorTerrainInput } from "./radarFloorPreparationProtocol";
import { selectRadarFloorRequestState } from "./radarFloorOwnership";


export interface RadarFloorLattice {
  /** Floor sample counts actually fetched, after any budget-driven stride. */
  nx: number;
  ny: number;
  /** 1 when the floor is at the model's native horizontal resolution. */
  strideXY: number;
}

export interface RadarFloorState {
  requestedRequestIdentity: string | null;
  loadedRequestIdentity: string | null;
  /**
   * One reduced value per column, x-slowest. Deliberately NOT pre-coloured:
   * the colour domain has to be latched across frames before encoding (see
   * `useRadarFloorLayer`), and a per-frame domain makes the sheet pulse through
   * a movie and stop reading as a quantitative field.
   */
  values: Float32Array | null;
  shape: [number, number] | null;
  /** Robust stats over `values`, for deriving the colour policy. */
  stats: SignedFieldStats | null;
  lattice: RadarFloorLattice | null;
  loading: boolean;
  error: string | null;
  /**
   * The field and shape actually read, once resolved. Surfaced so the control
   * can disclose a substitution: a column maximum served from the engine's own
   * `composite_reflectivity` is the same number by definition, but the user is
   * still entitled to know which array was read.
   */
  source: FloorSource | null;
  /** Physical terrain surface at each returned x/y sample, x-slowest. */
  terrainMesh: TerrainFloorMesh | null;
}

export interface RadarFloorInput {
  runDir: string;
  stepIndex: number | null;
  publicationGeneration: string | null;
  fieldName: string | null;
  /** Model grid dimensions. `nr`/`nth` are the Cartesian x/y counts. */
  grid: {
    nr: number;
    nth: number;
    nz: number;
    terrainFollowing?: boolean;
    drM?: number;
    dyM?: number;
  } | null;
  reduction: FloorReduction;
  /** Every field this step published, for resolving a native 2-D substitute. */
  availableFields: readonly string[];
  fieldOrigins?: Record<string, "native" | "studio_derived" | "user_expression" | "external_provider">;
  /** False when the layer is toggled off — skips the fetch entirely. */
  enabled: boolean;
}

const IDLE: RadarFloorState = {
  requestedRequestIdentity: null,
  loadedRequestIdentity: null,
  values: null,
  shape: null,
  stats: null,
  lattice: null,
  loading: false,
  error: null,
  source: null,
  terrainMesh: null,
};

interface RadarFloorRequestState extends RadarFloorState {
  requestKey: string;
}

/** Stable identity for one floor request, so unrelated re-renders do not refetch. */
function requestKey(input: RadarFloorInput): string {
  const g = input.grid;
  return [
    input.enabled ? "on" : "off",
    input.runDir,
    input.stepIndex ?? "none",
    input.publicationGeneration ?? "legacy",
    input.fieldName ?? "none",
    fieldDefinitionIdentity(input.runDir, input.stepIndex, input.fieldName, input.publicationGeneration),
    fieldDefinitionIdentity(input.runDir, input.stepIndex, "composite_reflectivity", input.publicationGeneration),
    input.reduction,
    g ? `${g.nr}x${g.nth}x${g.nz}:${g.terrainFollowing ? "terrain" : "flat"}`
      + `:${g.drM ?? "dx?"}:${g.dyM ?? "dy?"}` : "nogrid",
    // Only membership matters, and only for the fields that can substitute.
    input.availableFields.includes("composite_reflectivity") ? "comp" : "nocomp",
    input.fieldName && input.availableFields.includes(input.fieldName) ? "available" : "unavailable",
    input.fieldName ? input.fieldOrigins?.[input.fieldName] ?? "native" : "none",
  ].join("|");
}

/** Human-readable cause for a plan that could not be built. */
function describePlanFailure(
  failure: "degenerate-grid" | "source-budget",
  sourceBytes: number,
  reduction: FloorReduction,
): string {
  if (failure === "degenerate-grid") {
    return "Floor unavailable: the run's grid dimensions are degenerate.";
  }
  const megabytes = Math.round(sourceBytes / (1024 * 1024));
  return `Floor unavailable: a ${reduction} reduction over this grid would have to`
    + ` read ${megabytes} MB, past the field-window ceiling. Use Surface, or export`
    + " composite_reflectivity so the engine's own column maximum can be read.";
}

/** One scope per mounted hook; latest-wins so scrubbing supersedes in-flight reads. */
let nextFloorScope = 0;
/** Unique across all in-flight reads in the app; backend registry rejects duplicates. */
let nextFloorRequest = 0;

export function useRadarFloor(input: RadarFloorInput): RadarFloorState {
  const [state, setState] = useState<RadarFloorRequestState>({ ...IDLE, requestKey: "" });
  const [loadScope] = useState(() => `radar-floor:${++nextFloorScope}`);
  const preparationServiceRef = useRef<RadarFloorPreparationService>();
  if (!preparationServiceRef.current) {
    preparationServiceRef.current = new RadarFloorPreparationService();
  }
  const retainedPreparationRef = useRef<RetainedRadarFloorPreparation | null>(null);
  const inputRef = useRef(input);
  inputRef.current = input;
  const key = requestKey(input);

  useEffect(() => {
    const {
      runDir,
      stepIndex,
      publicationGeneration,
      fieldName,
      grid,
      reduction,
      availableFields,
      fieldOrigins,
    } = inputRef.current;
    if (!inputRef.current.enabled || !fieldName || stepIndex == null || !grid) {
      retainedPreparationRef.current?.release();
      retainedPreparationRef.current = null;
      setState({ ...IDLE, requestKey: key });
      return;
    }
    if (!availableFields.includes(fieldName)) {
      retainedPreparationRef.current?.release();
      retainedPreparationRef.current = null;
      setState({ ...IDLE, requestKey: key,
        error: "Floor unavailable: the selected field is not published in this frame." });
      return;
    }

    const source = resolveFloorSource(fieldName, reduction, availableFields);
    if (!fieldSupportsBoundedWindows(fieldOrigins?.[source.fieldName])) {
      retainedPreparationRef.current?.release();
      retainedPreparationRef.current = null;
      setState({
        ...IDLE,
        requestKey: key,
        source,
        error: "Floor LOD requires a published field; derived fields must be evaluated before display aggregation.",
      });
      return;
    }
    const terrain = grid.terrainFollowing === true;
    const outcome = planFloorHyperslab(grid.nr, grid.nth, grid.nz, source.kind, terrain
      ? { outputBytesPerCell: 36, outputDepth: 1, maxOutputCells: 512 * 512 }
      : undefined);
    if (!outcome.ok) {
      retainedPreparationRef.current?.release();
      retainedPreparationRef.current = null;
      setState({
        ...IDLE,
        requestKey: key,
        source,
        error: describePlanFailure(outcome.failure, outcome.sourceBytes, reduction),
      });
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setState((previous) => ({
      ...previous,
      requestKey: key,
      source,
      loading: true,
      error: null,
    }));

    const admittedPlan = terrain
      ? { ...outcome.plan, sourceBytes: MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES }
      : outcome.plan;
    const valuePlan = terrain && source.kind === "column"
      ? { ...admittedPlan, stride: [admittedPlan.stride[0], admittedPlan.stride[1], grid.nz] }
      : admittedPlan;
    const readWindow = (
      readFieldName: string,
      plan: typeof outcome.plan,
      aggregation: ReturnType<typeof fieldWindowAggregation>,
    ) => readFieldAny(runDir, stepIndex, readFieldName, publicationGeneration, {
      latestScope: loadScope,
      window: floorWindowRequest(
        floorRequestId(++nextFloorRequest, stepIndex, readFieldName, reduction),
        plan,
        aggregation,
      ),
    });
    const load = async () => {
      const window = await readWindow(
        source.fieldName,
        valuePlan,
        terrain && source.kind === "column"
          ? "block_max"
          : fieldWindowAggregation(source.fieldName),
      );
      const geometryWindow = terrain && reduction === "column-max"
        ? await readWindow(fieldName, {
          ...admittedPlan,
          origin: [0, 0, 0],
          extent: [grid.nr, grid.nth, 1],
          stride: [outcome.plan.strideXY, outcome.plan.strideXY, 1],
        }, "cell_mean")
        : window;
      return { window, geometryWindow };
    };
    load().then(
      async ({ window, geometryWindow }) => {
        if (cancelled) return;
        // A native 2-D product returns a rank-2 shape; a z-slab or a full
        // column returns rank 3. All three reduce to one value per column, so
        // normalise the depth rather than branching downstream.
        const shape = window.shape;
        const nx = shape[0] ?? 0;
        const ny = shape[1] ?? 0;
        const nz = shape.length >= 3 ? (shape[2] ?? 1) : 1;
        if (!(nx > 0) || !(ny > 0)) {
          retainedPreparationRef.current?.release();
          retainedPreparationRef.current = null;
          setState({
            ...IDLE,
            requestKey: key,
            source,
            error: "Floor unavailable: the field window returned an empty lattice.",
          });
          return;
        }
        let terrainInput: RadarFloorTerrainInput | null = null;
        if (terrain) {
          const geometry = geometryWindow.windowGeometry;
          const centers = geometry?.centroidsM;
          const volumes = geometry?.volumesM3;
          if (!centers || !volumes || geometryWindow.shape[0] !== nx
              || geometryWindow.shape[1] !== ny || geometryWindow.shape[2] !== 1
              || volumes.length !== nx * ny) {
            retainedPreparationRef.current?.release();
            retainedPreparationRef.current = null;
            setState({
              ...IDLE,
              requestKey: key,
              source,
              error: "Floor unavailable: terrain surface geometry is incomplete.",
            });
            return;
          }
          terrainInput = {
            centroids: centers,
            volumesM3: volumes,
            sourceShape: [grid.nr, grid.nth],
            strideXY: outcome.plan.strideXY,
            dxM: grid.drM ?? 1,
            dyM: grid.dyM ?? grid.drM ?? 1,
          };
        }
        const prepared = await preparationServiceRef.current!.prepare(
          window.data,
          [nx, ny, nz],
          reduction,
          terrainInput,
          controller.signal,
        );
        if (cancelled) {
          prepared.release();
          return;
        }
        retainedPreparationRef.current?.release();
        retainedPreparationRef.current = prepared;
        setState({
          requestKey: key,
          requestedRequestIdentity: key,
          loadedRequestIdentity: key,
          // A null `stats` means every sample is non-finite. That is genuinely
          // nothing to draw, not an error — the caller renders no sheet.
          values: prepared.values,
          shape: prepared.stats ? [nx, ny] : null,
          stats: prepared.stats,
          lattice: { nx, ny, strideXY: outcome.plan.strideXY },
          loading: false,
          error: null,
          source,
          terrainMesh: prepared.terrainMesh,
        });
      },
      (error: unknown) => {
        if (cancelled) return;
        retainedPreparationRef.current?.release();
        retainedPreparationRef.current = null;
        setState({
          ...IDLE,
          requestKey: key,
          source,
          error: error instanceof Error ? error.message : String(error),
        });
      },
    ).catch((error: unknown) => {
      if (cancelled || controller.signal.aborted) return;
      retainedPreparationRef.current?.release();
      retainedPreparationRef.current = null;
      setState({
        ...IDLE,
        requestKey: key,
        source,
        error: error instanceof Error ? error.message : String(error),
      });
    });

    return () => {
      cancelled = true;
      controller.abort();
    };
    // Keyed on `key` alone: `input` is a fresh object literal on every render,
    // so depending on it would refetch the floor continuously. `requestKey`
    // folds in every field that actually changes the request.
  }, [key, loadScope]);

  useEffect(() => () => {
    preparationServiceRef.current?.shutdown();
    retainedPreparationRef.current?.release();
    retainedPreparationRef.current = null;
  }, []);

  const expectsRequest = input.enabled && input.fieldName != null
    && input.stepIndex != null && input.grid != null;
  return selectRadarFloorRequestState(state, key, expectsRequest, { ...IDLE, requestKey: key });
}
