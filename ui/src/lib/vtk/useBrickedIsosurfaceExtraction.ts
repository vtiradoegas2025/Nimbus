import { useEffect, useMemo, useRef, useState } from "react";
import type { GridInfo, StepInfo } from "@/lib/run/runOutputs";
import { readField, readFieldBrickHierarchy } from "@/lib/run/runOutputs";
import {
  assertCompatibleBrickHierarchies,
  planAllContourBricks,
  planContourBricks,
  selectContourHierarchyLevel,
  unionContourBrickPlans,
  type ContourBrickPlan,
} from "@/lib/plot/fieldBrickHierarchy";
import type { OwnedBrickSurface } from "./vtkBrickedContour";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";
import { BrickedContourClient } from "./brickedContourClient";
import { BrickedSurfaceCompositionClient } from "./brickedSurfaceCompositionClient";
import type { VelocityGradientDiagnostic } from "./vtkPreparation";
import { MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES } from "@/lib/dataset/nativeFieldWindow";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";

interface BrickedState {
  identity: string;
  geometry: VortexSurfaceGeometry | null;
  oppositeGeometry: VortexSurfaceGeometry | null;
  plan: ContourBrickPlan | null;
  stats: { min: number; max: number; p2: number; p98: number } | null;
  nonFiniteCount: number;
  rawTriangleCounts: { primary: number; opposite: number };
  loading: boolean;
  error: string | null;
}

export interface BrickedIsosurfaceExtraction {
  geometry: VortexSurfaceGeometry | null;
  oppositeGeometry: VortexSurfaceGeometry | null;
  plan: ContourBrickPlan | null;
  stats: { min: number; max: number; p2: number; p98: number } | null;
  nonFiniteCount: number;
  residentBrickCount: number;
  residentBytes: number;
  rawTriangleCounts: { primary: number; opposite: number };
  loading: boolean;
  error: string | null;
}

function axisCoordinate(
  grid: GridInfo,
  sourceOffset: readonly number[],
  sourceShape: readonly number[],
  axis: number,
  localIndex: number,
): number {
  const storageIndex = sourceOffset[axis] + localIndex;
  const coordinates = grid.scalarCoordinatesM?.[axis];
  if (coordinates?.length === sourceShape[axis]) return coordinates[localIndex];
  if (coordinates?.length && storageIndex < coordinates.length) return coordinates[storageIndex];
  const spacing = axis === 0 ? grid.drM ?? 1 : axis === 1 ? grid.dyM ?? grid.drM ?? 1 : grid.dzM ?? 1;
  if (grid.scalarOriginM?.[axis] != null) {
    return grid.scalarOriginM[axis] + storageIndex * spacing;
  }
  const localOrigin = grid.physicalBoundsM?.lower[axis] != null
    ? grid.physicalBoundsM.lower[axis] + 0.5 * spacing
    : axis === 2 ? 0.5 * spacing : -(sourceShape[axis] - 1) * spacing * 0.5;
  return localOrigin + localIndex * spacing;
}

function ownershipFor(
  item: ContourBrickPlan["items"][number],
  hierarchy: Awaited<ReturnType<typeof readFieldBrickHierarchy>>,
  grid: GridInfo,
  levelNumber: number,
): Pick<OwnedBrickSurface, "ownedCellBounds" | "includeUpper"> {
  if (grid.terrainFollowing) {
    return {
      ownedCellBounds: [
        Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY,
      ],
      includeUpper: [true, true, true],
    };
  }
  const level = hierarchy.levels[levelNumber];
  const lower = item.ownedCellOrigin.map((index, axis) =>
    axisCoordinate(
      grid,
      hierarchy.sourceIndexOffset,
      hierarchy.sourceShape,
      axis,
      index * level.stride[axis],
    ));
  const upper = item.ownedCellOrigin.map((index, axis) =>
    axisCoordinate(
      grid,
      hierarchy.sourceIndexOffset,
      hierarchy.sourceShape,
      axis,
      (index + item.ownedCellExtent[axis]) * level.stride[axis],
    ));
  return {
    ownedCellBounds: [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]],
    includeUpper: item.ownedCellOrigin.map((index, axis) =>
      index + item.ownedCellExtent[axis] >= level.shape[axis] - 1) as [boolean, boolean, boolean],
  };
}

function terrainCellBoundsFor(
  item: ContourBrickPlan["items"][number],
  hierarchy: Awaited<ReturnType<typeof readFieldBrickHierarchy>>,
  levelNumber: number,
): [number, number, number, number, number, number] {
  const stride = hierarchy.levels[levelNumber].stride;
  const readOrigin = item.readWindow.origin.map((value, axis) =>
    (value - hierarchy.sourceIndexOffset[axis]) / stride[axis]);
  const lower = item.ownedCellOrigin.map((value, axis) => value - readOrigin[axis]);
  const upper = lower.map((value, axis) => value + item.ownedCellExtent[axis]);
  if (![...readOrigin, ...lower, ...upper].every(Number.isSafeInteger)) {
    throw new Error("Terrain contour ownership is not aligned with its brick window");
  }
  return [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]];
}

let brickedScopeSequence = 0;

export function useBrickedIsosurfaceExtraction(input: {
  enabled: boolean;
  runDir: string;
  step: StepInfo | null | undefined;
  fieldName: string | null;
  threshold: number | null;
  oppositeThreshold?: number | null;
  diagnostic?: VelocityGradientDiagnostic | null;
  componentFields?: readonly [string, string, string] | null;
  maximumPreparedCells: number;
  externalResourceOwnership?: boolean;
  externalWorkerAdmission?: boolean;
  externalGeometryOwnership?: boolean;
}): BrickedIsosurfaceExtraction {
  const {
    enabled,
    runDir,
    step,
    fieldName,
    threshold,
    oppositeThreshold = null,
    diagnostic = null,
    componentFields = null,
    maximumPreparedCells,
    externalResourceOwnership = false,
    externalWorkerAdmission = externalResourceOwnership,
    externalGeometryOwnership = externalResourceOwnership,
  } = input;
  const scopeRef = useRef<string>();
  if (!scopeRef.current) scopeRef.current = `vtk:bricked-isosurface:${++brickedScopeSequence}`;
  const clientRef = useRef<BrickedContourClient | null>(null);
  const compositionClientRef = useRef<BrickedSurfaceCompositionClient | null>(null);
  const componentFieldsKey = componentFields?.join("\0") ?? "";
  const identity = useMemo(() => JSON.stringify([
    runDir,
    step?.index ?? null,
    step?.publicationGeneration ?? null,
    fieldName,
    threshold,
    oppositeThreshold,
    diagnostic,
    componentFieldsKey,
    maximumPreparedCells,
    externalResourceOwnership,
    externalWorkerAdmission,
    externalGeometryOwnership,
  ]), [componentFieldsKey, diagnostic, externalGeometryOwnership, externalResourceOwnership, externalWorkerAdmission, fieldName, maximumPreparedCells, oppositeThreshold, runDir, step?.index, step?.publicationGeneration, threshold]);
  const [state, setState] = useState<BrickedState>({
    identity: "",
    geometry: null,
    oppositeGeometry: null,
    plan: null,
    stats: null,
    nonFiniteCount: 0,
    rawTriangleCounts: { primary: 0, opposite: 0 },
    loading: false,
    error: null,
  });
  useEffect(() => () => {
    clientRef.current?.shutdown();
    clientRef.current = null;
    compositionClientRef.current?.shutdown();
    compositionClientRef.current = null;
  }, []);

  useEffect(() => {
    if (!enabled || !step || !step.grid || !fieldName) return;
    const grid = step.grid;
    const controller = new AbortController();
    setState({ identity, geometry: null, oppositeGeometry: null, plan: null, stats: null, nonFiniteCount: 0, rawTriangleCounts: { primary: 0, opposite: 0 }, loading: true, error: null });
    void (async () => {
      const sourceFields = diagnostic && componentFieldsKey
        ? componentFieldsKey.split("\0")
        : [fieldName];
      const hierarchies = await Promise.all(sourceFields.map((sourceField) =>
        readFieldBrickHierarchy(runDir, step, sourceField)));
      if (hierarchies.length > 1) assertCompatibleBrickHierarchies(hierarchies);
      const hierarchy = hierarchies[0];
      if (!diagnostic && !hierarchy.cullingSafe) {
        throw new Error("Brick extrema are not safe for contour culling.");
      }
      if (threshold == null && !diagnostic) {
        const ranges = hierarchy.levels[0].bricks.map(brick => brick.scalarRange)
          .filter(range => range.finiteCount > 0 && range.minimum != null && range.maximum != null);
        if (ranges.length === 0) throw new Error("Field has no finite samples for an isosurface");
        const minimum = Math.min(...ranges.map(range => range.minimum!));
        const maximum = Math.max(...ranges.map(range => range.maximum!));
        return {
          plan: null,
          stats: { min: minimum, max: maximum, p2: minimum, p98: maximum },
          nonFiniteCount: hierarchy.levels[0].bricks.reduce((sum, brick) => sum + brick.scalarRange.nonFiniteCount, 0),
          rawTriangleCounts: { primary: 0, opposite: 0 },
          geometry: null, oppositeGeometry: null,
        };
      }
      const selectedLevel = selectContourHierarchyLevel(hierarchy, {
        maximumPreparedCells,
        threshold: threshold ?? undefined,
        oppositeThreshold,
        allBricks: diagnostic != null,
        terrainFollowing: grid.terrainFollowing,
      });
      const primaryPlan = diagnostic
        ? planAllContourBricks(hierarchy, {
            level: selectedLevel,
            terrainFollowing: grid.terrainFollowing,
          })
        : planContourBricks(hierarchy, threshold!, {
            level: selectedLevel,
            terrainFollowing: grid.terrainFollowing,
          });
      const oppositePlan = diagnostic || oppositeThreshold == null
        ? null
        : planContourBricks(hierarchy, oppositeThreshold, {
            level: selectedLevel,
            terrainFollowing: grid.terrainFollowing,
          });
      const plan = unionContourBrickPlans(primaryPlan, oppositePlan);
      if ((!clientRef.current || clientRef.current.isClosed()) && "Worker" in globalThis) {
        clientRef.current = new BrickedContourClient(undefined, {
          externallyReserved: externalWorkerAdmission,
        });
      }
      if ((!compositionClientRef.current || compositionClientRef.current.isClosed())
          && "Worker" in globalThis) {
        compositionClientRef.current = new BrickedSurfaceCompositionClient();
      }
      const client = clientRef.current;
      const compositionClient = compositionClientRef.current;
      if (!client || !compositionClient) {
        throw new Error("Progressive brick extraction and composition require module workers");
      }
      const datasetIdentity = JSON.stringify([
        runDir,
        step.index,
        step.publicationGeneration,
        fieldName,
        diagnostic,
        sourceFields,
        selectedLevel,
        hierarchy.levels[selectedLevel].brickShape,
      ]);
      await client.reset(datasetIdentity, controller.signal);
      await client.admitBricks(plan.items.map((item) => {
        const bytes = item.readWindow.maxOutputBytes;
        if (bytes == null) throw new Error(`Brick ${item.brickId} has no admitted output bound`);
        const inputBytes = Math.max(bytes, item.readWindow.maxSourceBytes ?? 0);
        const diagnosticMetadataBytes = diagnostic
          ? (item.readWindow.extent[2] ?? 0) * 3 * Float32Array.BYTES_PER_ELEMENT
          : 0;
        return {
          brickId: item.brickId,
          inputBytes: (inputBytes + MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES)
            * sourceFields.length,
          outputBytes: bytes + diagnosticMetadataBytes,
          sampleUpperBound: item.readWindow.extent.reduce((product, extent) => product * extent, 1),
          contourMethod: grid.terrainFollowing
            ? "terrain-tetrahedra" as const
            : "marching-cubes" as const,
        };
      }), controller.signal);
      const selectedStride = hierarchy.levels[selectedLevel].stride;
      const spacing = Math.min(
        (grid.drM ?? 1) * selectedStride[0],
        (grid.dyM ?? grid.drM ?? 1) * selectedStride[1],
        (grid.dzM ?? 1) * selectedStride[2],
      );
      for (const item of plan.items) {
        if (client.hasBrick(item.brickId)) continue;
        const fields = await Promise.all(sourceFields.map((sourceField) => readField(
          runDir, step.index, sourceField, step.publicationGeneration, {
            signal: controller.signal,
            window: item.readWindow,
            cacheRetention: "transient",
            ownership: "transferable",
            latestScope: `${scopeRef.current}:${sourceField}`,
            brickHierarchyLevel: hierarchy.levels[selectedLevel].source.kind === "npy_lod"
              ? selectedLevel
              : undefined,
          })),
        );
        await client.loadBrick(item.brickId, diagnostic ? {
          kind: "velocity-diagnostic",
          diagnostic,
          fields: fields as [typeof fields[number], typeof fields[number], typeof fields[number]],
          grid,
          refineForDisplay: false,
          datasetHandoff: "shared-geometry",
        } : {
          kind: "scalar",
          field: fields[0],
          grid,
          refineForDisplay: false,
          datasetHandoff: "shared-geometry",
        }, controller.signal, grid.terrainFollowing
          ? terrainCellBoundsFor(item, hierarchy, selectedLevel)
          : undefined);
      }
      const preparedStats = plan.items
        .map((item) => client.brickStats(item.brickId))
        .filter((stats): stats is NonNullable<typeof stats> => stats != null);
      const sourceRanges = hierarchy.levels[0].bricks
        .map((brick) => brick.scalarRange)
        .filter((range) => range.finiteCount > 0 && range.minimum != null && range.maximum != null);
      const stats = diagnostic
        ? preparedStats.length > 0 ? {
            min: Math.min(...preparedStats.map((value) => value.min)),
            max: Math.max(...preparedStats.map((value) => value.max)),
            p2: Math.min(...preparedStats.map((value) => value.p2)),
            p98: Math.max(...preparedStats.map((value) => value.p98)),
            positiveP98: Math.max(...preparedStats.map((value) => value.positiveP98 ?? 0)),
          } : null
        : sourceRanges.length > 0 ? {
            min: Math.min(...sourceRanges.map((range) => range.minimum!)),
            max: Math.max(...sourceRanges.map((range) => range.maximum!)),
            p2: Math.min(...sourceRanges.map((range) => range.minimum!)),
            p98: Math.max(...sourceRanges.map((range) => range.maximum!)),
          } : null;
      const nonFiniteCount = diagnostic
        ? preparedStats.reduce((sum, value) => sum + value.sourceNonFiniteCount, 0)
        : hierarchy.levels[0].bricks.reduce(
            (sum, brick) => sum + brick.scalarRange.nonFiniteCount,
            0,
          );
      if (!controller.signal.aborted) {
        setState({ identity, geometry: null, oppositeGeometry: null, plan, stats, nonFiniteCount, rawTriangleCounts: { primary: 0, opposite: 0 }, loading: threshold != null, error: null });
      }
      if (threshold == null) {
        compositionClient.release("primary");
        compositionClient.release("opposite");
        return {
          plan,
          stats,
          nonFiniteCount,
          rawTriangleCounts: { primary: 0, opposite: 0 },
          geometry: null,
          oppositeGeometry: null,
        };
      }
      const primaryChunks = await client.extract(
        primaryPlan.items.map((item) => item.brickId),
        threshold,
        controller.signal,
        { retentionKey: "primary" },
      );
      const primaryGeometryByBrick = new Map(
        primaryChunks.map((chunk) => [chunk.brickId, chunk.geometry]),
      );
      const oppositeChunks = oppositePlan && oppositeThreshold != null
        ? await client.extract(
            oppositePlan.items.map((item) => item.brickId),
            oppositeThreshold,
            controller.signal,
            { retentionKey: "opposite" },
          )
        : (client.releaseExtraction("opposite"), null);
      const oppositeGeometryByBrick = new Map(
        (oppositeChunks ?? []).map((chunk) => [chunk.brickId, chunk.geometry]),
      );
      const tolerance = Math.max(1e-7, spacing * 1e-6);
      const surfaces = (
        surfacePlan: ContourBrickPlan,
        geometryByBrick: ReadonlyMap<string, VortexSurfaceGeometry | null>,
      ): OwnedBrickSurface[] => surfacePlan.items.flatMap((item) => {
        const geometry = geometryByBrick.get(item.brickId);
        return geometry
          ? [{ brickId: item.brickId, geometry,
              ...ownershipFor(item, hierarchy, grid, selectedLevel) }]
          : [];
      });
      let geometry: VortexSurfaceGeometry | null;
      try {
        geometry = await compositionClient.compose(
          `${identity}:primary`,
          "primary",
          surfaces(primaryPlan, primaryGeometryByBrick),
          tolerance,
          controller.signal,
        );
      } finally {
        client.releaseExtraction("primary");
      }
      let oppositeGeometry: VortexSurfaceGeometry | null = null;
      if (oppositePlan) {
        try {
          oppositeGeometry = await compositionClient.compose(
            `${identity}:opposite`,
            "opposite",
            surfaces(oppositePlan, oppositeGeometryByBrick),
            tolerance,
            controller.signal,
          );
        } finally {
          client.releaseExtraction("opposite");
        }
      } else {
        compositionClient.release("opposite");
      }
      return {
        plan,
        stats,
        nonFiniteCount,
        rawTriangleCounts: {
          primary: primaryChunks.reduce((sum, chunk) => sum + (chunk.geometry?.triangles ?? 0), 0),
          opposite: (oppositeChunks ?? []).reduce((sum, chunk) => sum + (chunk.geometry?.triangles ?? 0), 0),
        },
        geometry,
        oppositeGeometry,
      };
    })().then(
      ({ plan, stats, nonFiniteCount, rawTriangleCounts, geometry, oppositeGeometry }) => {
        if (!controller.signal.aborted) setState({ identity, geometry, oppositeGeometry, plan, stats, nonFiniteCount, rawTriangleCounts, loading: false, error: null });
      },
      (error) => {
        if (!controller.signal.aborted) setState({
          identity,
          geometry: null,
          oppositeGeometry: null,
          plan: null,
          stats: null,
          nonFiniteCount: 0,
          rawTriangleCounts: { primary: 0, opposite: 0 },
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        });
      },
    );
    return () => controller.abort();
  }, [componentFieldsKey, diagnostic, enabled, externalWorkerAdmission, fieldName, identity, maximumPreparedCells, oppositeThreshold, runDir, step, threshold]);

  const current = state.identity === identity;
  const ownedGeometry = current ? state.geometry : null;
  const ownedOppositeGeometry = current ? state.oppositeGeometry : null;
  useEffect(() => {
    if (externalGeometryOwnership || (!ownedGeometry && !ownedOppositeGeometry)) return;
    const geometries = [ownedGeometry, ownedOppositeGeometry]
      .filter((geometry): geometry is VortexSurfaceGeometry => geometry != null);
    const outcome = visualizationResourceGovernor.reserve({
      owner: scopeRef.current!,
      generation: identity,
      priority: "visible",
      claims: geometries.flatMap((geometry) => [geometry.points, geometry.polys, geometry.normals]).map((array) =>
        visualizationBufferClaim(array.buffer, "retained-geometry")),
    });
    if (!outcome.ok) {
      setState((existing) => existing.identity === identity
        ? { ...existing, geometry: null, oppositeGeometry: null, error: outcome.reason }
        : existing);
      return;
    }
    return () => { visualizationResourceGovernor.release(outcome.reservation.id); };
  }, [externalGeometryOwnership, identity, ownedGeometry, ownedOppositeGeometry]);
  return {
    geometry: ownedGeometry,
    oppositeGeometry: ownedOppositeGeometry,
    plan: current ? state.plan : null,
    stats: current ? state.stats : null,
    nonFiniteCount: current ? state.nonFiniteCount : 0,
    residentBrickCount: current ? clientRef.current?.loadedBrickCount() ?? 0 : 0,
    residentBytes: current ? clientRef.current?.residentBytes() ?? 0 : 0,
    rawTriangleCounts: current
      ? state.rawTriangleCounts
      : { primary: 0, opposite: 0 },
    loading: enabled && (!current || state.loading),
    error: current ? state.error : null,
  };
}
