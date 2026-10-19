import { useEffect, useMemo, useRef, useState } from "react";
import type { NestedVolumeOwnershipLayer } from "@/lib/run/outputHierarchy";
import { readFieldBrickHierarchy } from "@/lib/run/runOutputs";
import {
  assertCompatibleBrickHierarchies,
  planAllContourBricks,
  planContourBricks,
  selectContourHierarchyLevel,
  unionContourBrickPlans,
  type ContourBrickPlan,
} from "@/lib/plot/fieldBrickHierarchy";
import { MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES } from "@/lib/dataset/nativeFieldWindow";
import { visualizationBufferClaim, visualizationReservationClaim, visualizationResourceGovernor } from "@/lib/core/visualizationResourceGovernor";
import { useBrickedIsosurfaceExtraction, type BrickedIsosurfaceExtraction } from "./useBrickedIsosurfaceExtraction";
import { NestedSurfaceCompositionClient, type NestedSurfaceCompositionResult } from "./nestedSurfaceCompositionClient";
import type { VelocityGradientDiagnostic } from "./vtkPreparation";

const MAX_NESTED_DEPTH = 8;
const EMPTY: BrickedIsosurfaceExtraction = {
  geometry: null, oppositeGeometry: null, plan: null, stats: null, nonFiniteCount: 0,
  residentBrickCount: 0, residentBytes: 0,
  rawTriangleCounts: { primary: 0, opposite: 0 }, loading: false, error: null,
};

interface NestedAdmission {
  identity: string;
  plans: ContourBrickPlan[];
  reservationId: number;
}

type NestedInput = {
  enabled: boolean;
  layers: readonly NestedVolumeOwnershipLayer[];
  fieldName: string | null;
  threshold: number | null;
  oppositeThreshold?: number | null;
  diagnostic?: VelocityGradientDiagnostic | null;
  componentFields?: readonly [string, string, string] | null;
  maximumPreparedCells: number;
};

function useLayer(input: NestedInput, index: number, admitted: boolean): BrickedIsosurfaceExtraction {
  const layer = input.layers[index];
  return useBrickedIsosurfaceExtraction({
    enabled: admitted && Boolean(layer),
    runDir: layer?.level.runDir ?? "",
    step: layer?.level.step ?? null,
    fieldName: input.fieldName,
    threshold: input.threshold,
    oppositeThreshold: input.oppositeThreshold,
    diagnostic: input.diagnostic,
    componentFields: input.componentFields,
    maximumPreparedCells: Math.max(1, Math.floor(input.maximumPreparedCells / Math.max(1, input.layers.length))),
    externalWorkerAdmission: true,
    externalGeometryOwnership: false,
  });
}

let nestedScopeSequence = 0;

export function useNestedBrickedIsosurfaceExtraction(input: NestedInput): BrickedIsosurfaceExtraction & {
  layerCount: number;
  transitionTriangles: number;
  atomicAdmission: boolean;
} {
  const ownerRef = useRef(`vtk:nested-isosurface:${++nestedScopeSequence}`);
  const componentFieldsKey = input.componentFields?.join("\0") ?? "";
  const layersKey = JSON.stringify(input.layers.map((layer) => ({
    levelId: layer.level.levelId, runDir: layer.level.runDir, step: layer.level.step.index,
    generation: layer.level.step.publicationGeneration, exclusion: layer.excludeChildBoundsM,
  })));
  const identity = useMemo(() => JSON.stringify([
    layersKey, input.fieldName, input.threshold, input.oppositeThreshold ?? null,
    input.diagnostic ?? null, componentFieldsKey, input.maximumPreparedCells,
  ]), [componentFieldsKey, input.diagnostic, input.fieldName, input.maximumPreparedCells, input.oppositeThreshold, input.threshold, layersKey]);
  const stableLayers = useMemo(() => input.layers, [input.layers]);
  const [admission, setAdmission] = useState<NestedAdmission | null>(null);
  const [admissionError, setAdmissionError] = useState<string | null>(null);
  const compositionClientRef = useRef<NestedSurfaceCompositionClient | null>(null);
  const compositionReservationRef = useRef<number | null>(null);
  const [composition, setComposition] = useState<{
    identity: string;
    result: NestedSurfaceCompositionResult | null;
    loading: boolean;
    error: string | null;
  }>({ identity: "", result: null, loading: false, error: null });

  useEffect(() => () => {
    compositionClientRef.current?.shutdown();
    compositionClientRef.current = null;
    if (compositionReservationRef.current != null) {
      visualizationResourceGovernor.release(compositionReservationRef.current);
      compositionReservationRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!input.enabled || !input.fieldName || stableLayers.length < 2) {
      setAdmission(null); setAdmissionError(null); return;
    }
    if (stableLayers.length > MAX_NESTED_DEPTH + 1) {
      setAdmission(null); setAdmissionError(`Nested isosurface depth exceeds ${MAX_NESTED_DEPTH}`); return;
    }
    const controller = new AbortController();
    let reservationId: number | null = null;
    setAdmission(null); setAdmissionError(null);
    void (async () => {
      const sourceFields = input.diagnostic && componentFieldsKey
        ? componentFieldsKey.split("\0") : [input.fieldName!];
      const hierarchySets = await Promise.all(stableLayers.map((layer) => Promise.all(
        sourceFields.map((field) => readFieldBrickHierarchy(layer.level.runDir, layer.level.step, field)),
      )));
      if (controller.signal.aborted) throw new DOMException("Nested contour admission cancelled", "AbortError");
      const perLayerCells = Math.max(1, Math.floor(input.maximumPreparedCells / stableLayers.length));
      const plans = hierarchySets.map((hierarchies, index) => {
        if (hierarchies.length > 1) assertCompatibleBrickHierarchies(hierarchies);
        const hierarchy = hierarchies[0];
        if (!input.diagnostic && !hierarchy.cullingSafe) {
          throw new Error(`${stableLayers[index].level.levelId} contour extrema are not culling-safe`);
        }
        const level = selectContourHierarchyLevel(hierarchy, {
          maximumPreparedCells: perLayerCells, threshold: input.threshold ?? undefined,
          oppositeThreshold: input.oppositeThreshold ?? null, allBricks: input.diagnostic != null,
          terrainFollowing: stableLayers[index].level.step.grid?.terrainFollowing,
        });
        const primary = input.diagnostic
          ? planAllContourBricks(hierarchy, { level, terrainFollowing: stableLayers[index].level.step.grid?.terrainFollowing })
          : planContourBricks(hierarchy, input.threshold!, { level, terrainFollowing: stableLayers[index].level.step.grid?.terrainFollowing });
        const opposite = input.diagnostic || input.oppositeThreshold == null ? null
          : planContourBricks(hierarchy, input.oppositeThreshold, { level, terrainFollowing: stableLayers[index].level.step.grid?.terrainFollowing });
        return unionContourBrickPlans(primary, opposite);
      });
      const claims = plans.flatMap((plan, layerIndex) => plan.items.flatMap((item) => {
        const outputBytes = item.readWindow.maxOutputBytes;
        if (outputBytes == null) throw new Error(`Nested contour brick ${item.brickId} has no output bound`);
        const inputBytes = Math.max(outputBytes, item.readWindow.maxSourceBytes ?? 0);
        const diagnosticMetadataBytes = input.diagnostic
          ? (item.readWindow.extent[2] ?? 0) * 3 * Float32Array.BYTES_PER_ELEMENT : 0;
        const prefix = `${ownerRef.current}:${layerIndex}:${item.brickId}`;
        return [
          visualizationReservationClaim(`${prefix}:input`, "worker-resident",
            (inputBytes + MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES) * sourceFields.length),
          visualizationReservationClaim(`${prefix}:prepared`, "worker-resident", outputBytes + diagnosticMetadataBytes),
        ];
      }));
      const outcome = visualizationResourceGovernor.reserve({
        owner: ownerRef.current, generation: identity, priority: "visible", claims,
      });
      if (!outcome.ok) throw new Error(outcome.reason);
      reservationId = outcome.reservation.id;
      if (controller.signal.aborted) throw new DOMException("Nested contour admission cancelled", "AbortError");
      setAdmission({ identity, plans, reservationId });
    })().catch((error) => {
      if (!controller.signal.aborted) setAdmissionError(error instanceof Error ? error.message : String(error));
    });
    return () => {
      controller.abort();
      if (reservationId != null) visualizationResourceGovernor.release(reservationId);
    };
  }, [componentFieldsKey, identity, input.diagnostic, input.enabled, input.fieldName, input.maximumPreparedCells, input.oppositeThreshold, input.threshold, stableLayers]);

  const admitted = admission?.identity === identity;
  const result0 = useLayer(input, 0, admitted); const result1 = useLayer(input, 1, admitted);
  const result2 = useLayer(input, 2, admitted); const result3 = useLayer(input, 3, admitted);
  const result4 = useLayer(input, 4, admitted); const result5 = useLayer(input, 5, admitted);
  const result6 = useLayer(input, 6, admitted); const result7 = useLayer(input, 7, admitted);
  const result8 = useLayer(input, 8, admitted);
  const results = useMemo(() => [result0, result1, result2, result3, result4, result5, result6, result7, result8],
    [result0, result1, result2, result3, result4, result5, result6, result7, result8]);
  const active = results.slice(0, stableLayers.length);
  const layerError = admissionError ?? active.find((result) => result.error)?.error ?? null;
  const layersLoading = !admitted || active.some((result) => result.loading || result.plan == null);
  const residentBytes = active.reduce((sum, result) => sum + result.residentBytes, 0);
  const compositionInputs = useMemo(() => {
    const primary = [result0.geometry, result1.geometry, result2.geometry, result3.geometry,
      result4.geometry, result5.geometry, result6.geometry, result7.geometry, result8.geometry];
    const opposite = [result0.oppositeGeometry, result1.oppositeGeometry, result2.oppositeGeometry,
      result3.oppositeGeometry, result4.oppositeGeometry, result5.oppositeGeometry,
      result6.oppositeGeometry, result7.oppositeGeometry, result8.oppositeGeometry];
    const childBounds = stableLayers.map((layer) => layer.excludeChildBoundsM ?? null);
    return {
      primary: { geometries: primary.slice(0, stableLayers.length), childBounds },
      opposite: { geometries: opposite.slice(0, stableLayers.length), childBounds },
    };
  }, [
    stableLayers, result0.geometry, result0.oppositeGeometry, result1.geometry, result1.oppositeGeometry,
    result2.geometry, result2.oppositeGeometry, result3.geometry, result3.oppositeGeometry,
    result4.geometry, result4.oppositeGeometry, result5.geometry, result5.oppositeGeometry,
    result6.geometry, result6.oppositeGeometry, result7.geometry, result7.oppositeGeometry,
    result8.geometry, result8.oppositeGeometry,
  ]);

  useEffect(() => {
    if (input.enabled) return;
    compositionClientRef.current?.shutdown();
    compositionClientRef.current = null;
    if (compositionReservationRef.current != null) {
      visualizationResourceGovernor.release(compositionReservationRef.current);
      compositionReservationRef.current = null;
    }
    setComposition({ identity: "", result: null, loading: false, error: null });
  }, [input.enabled]);

  useEffect(() => {
    if (!input.enabled || !admitted || layersLoading || layerError || stableLayers.length < 2) return;
    const controller = new AbortController();
    if (!compositionClientRef.current || compositionClientRef.current.isClosed()) {
      compositionClientRef.current = new NestedSurfaceCompositionClient();
    }
    const client = compositionClientRef.current;
    setComposition({ identity, result: null, loading: true, error: null });
    void client.compose(identity, compositionInputs.primary, compositionInputs.opposite, controller.signal).then(
      (result) => {
        const geometries = [result.primary.geometry, result.opposite.geometry]
          .filter((geometry) => geometry != null);
        const retained = visualizationResourceGovernor.reserve({
          owner: ownerRef.current, generation: `${identity}:composed`, priority: "visible",
          claims: geometries.flatMap((geometry) => [geometry.points, geometry.polys, geometry.normals])
            .map((array) => visualizationBufferClaim(array.buffer, "retained-geometry")),
        });
        client.shutdown();
        if (compositionClientRef.current === client) compositionClientRef.current = null;
        if (!retained.ok) {
          if (!controller.signal.aborted) setComposition({ identity, result: null, loading: false, error: retained.reason });
          return;
        }
        if (compositionReservationRef.current != null) {
          visualizationResourceGovernor.release(compositionReservationRef.current);
        }
        compositionReservationRef.current = retained.reservation.id;
        if (!controller.signal.aborted) setComposition({ identity, result, loading: false, error: null });
      },
      (error) => {
        client.shutdown();
        if (compositionClientRef.current === client) compositionClientRef.current = null;
        if (!controller.signal.aborted) setComposition({ identity, result: null, loading: false, error: error instanceof Error ? error.message : String(error) });
      },
    );
    return () => controller.abort();
  }, [
    admitted, compositionInputs, identity, input.enabled, layerError, layersLoading, stableLayers.length,
  ]);

  const compositionCurrent = composition.identity === identity;
  const combined = {
    ...EMPTY,
    geometry: compositionCurrent ? composition.result?.primary.geometry ?? null : null,
    oppositeGeometry: compositionCurrent ? composition.result?.opposite.geometry ?? null : null,
    plan: admitted ? active.at(-1)?.plan ?? admission?.plans.at(-1) ?? null : null,
    stats: active.reduce<BrickedIsosurfaceExtraction["stats"]>((stats, result) => result.stats
      ? stats ? { min: Math.min(stats.min, result.stats.min), max: Math.max(stats.max, result.stats.max),
          p2: Math.min(stats.p2, result.stats.p2), p98: Math.max(stats.p98, result.stats.p98) }
        : result.stats : stats, null),
    nonFiniteCount: active.reduce((sum, result) => sum + result.nonFiniteCount, 0),
    residentBrickCount: active.reduce((sum, result) => sum + result.residentBrickCount, 0),
    residentBytes,
    rawTriangleCounts: {
      primary: active.reduce((sum, result) => sum + result.rawTriangleCounts.primary, 0),
      opposite: active.reduce((sum, result) => sum + result.rawTriangleCounts.opposite, 0),
    },
    loading: input.enabled && (layersLoading || !compositionCurrent || composition.loading),
    error: layerError ?? (compositionCurrent ? composition.error : null),
    layerCount: input.enabled ? active.length : 0,
    transitionTriangles: compositionCurrent && composition.result
      ? composition.result.primary.transitionTriangles + composition.result.opposite.transitionTriangles : 0,
    atomicAdmission: admitted,
  };

  useEffect(() => {
    if (!admitted || !admission || layersLoading || layerError || residentBytes <= 0) return;
    const transition = visualizationResourceGovernor.transition(admission.reservationId, [{
      allocationId: `${ownerRef.current}:${identity}:resident`,
      stage: "worker-resident",
      bytes: residentBytes,
      accounting: "resident",
    }]);
    if (!transition.ok) setAdmissionError(transition.reason);
  }, [admission, admitted, identity, layerError, layersLoading, residentBytes]);

  return combined;
}
