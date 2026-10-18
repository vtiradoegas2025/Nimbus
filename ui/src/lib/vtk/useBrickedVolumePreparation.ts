import { volumeImageBytes, volumeValidityBudgetFactor, reserveVolumeValidity } from "@/lib/vtk/volumeValidityBudget";
import { reconstructVolumePlan, needsVolumeReconstruction } from "./volumeReconstruction";
import { useEffect, useMemo, useRef, useState } from "react";
import { useIoBudgetRetry } from "@/lib/core/useIoBudgetRetry";
import { readField, readFieldBrickHierarchy, type StepInfo } from "@/lib/run/runOutputs";
import {
  compactVolumeBrickPlan,
  selectVolumeHierarchyPlan,
  VolumeResidencyError,
  type FieldBrickHierarchy,
  type VolumeBrickPlan,
} from "@/lib/plot/fieldBrickHierarchy";
import { MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES } from "@/lib/dataset/nativeFieldWindow";
import {
  visualizationBufferClaim,
  visualizationReservationClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";
import type { PreparedVtkImage } from "./vtkPreparation";
import { volumeHaloSamples, withVolumeBrickOwnership } from "./volumeBrickGeometry";
import { planVolumeTextureLayout } from "./volumeTexturePacking";
import { VolumeBrickPreparationClient } from "./volumeBrickClient";
import type { VolumeBackground } from "./volumeBackground";
import {
  resolveVolumeTransferPolicy,
  volumeOpacityPoints,
  volumeOpacitySelectionKey,
} from "@/lib/plot/volumeTransferPolicy";
import type { ScalarFieldTransform } from "./vtkPreparation";
import type { VolumeOpacityClass } from "@/lib/plot/volumeTransferPolicy";
import {
  buildTransientVolumeHierarchy,
  sampleVolumeFieldStats,
  type TransientVolumeHierarchyResult,
} from "./transientVolumeHierarchy";

export interface PreparedVolumeBrick {
  brickId: string;
  prepared: PreparedVtkImage;
}
const NO_BRICKS: PreparedVolumeBrick[] = [];

interface BrickedVolumeState {
  identity: string;
  commitToken: number;
  reservationId: number | null;
  hierarchy: FieldBrickHierarchy | null;
  plan: VolumeBrickPlan | null;
  stats: { min: number; max: number; p2: number; p98: number; p995?: number } | null;
  bricks: PreparedVolumeBrick[];
  loading: boolean;
  error: string | null;
  requiresBoundedOverview?: boolean;
}

const transientHierarchyCache = new Map<string, TransientVolumeHierarchyResult>();
const sampledStatsCache = new Map<string, TransientVolumeHierarchyResult["stats"]>();

export type BrickedVolumePreparation = BrickedVolumeState & { current: boolean };

let volumeBrickScopeSequence = 0;
let volumeBrickCommitSequence = 0;

export function useBrickedVolumePreparation(input: {
  background?: VolumeBackground | null;
  enabled: boolean;
  runDir: string;
  step: StepInfo | null | undefined;
  fieldName: string | null;
  transfer: {
    colormap: string;
    opacityScale: number;
    scalarTransform: ScalarFieldTransform;
    stormStructure?: boolean;
  };
  maximumPreparedCells: number;
  maximumTextureDimension?: number;
  metadataOnly?: boolean;
  compactBricks?: boolean;
  requestedLevel?: number | null;
  allowTransientHierarchy?: boolean;
  opacityClass?: VolumeOpacityClass;
  excludePhysicalBoundsM?: {
    lower: readonly [number, number, number];
    upper: readonly [number, number, number];
  };
}): BrickedVolumePreparation {
  const {
    enabled,
    runDir,
    step,
    fieldName,
    transfer,
    maximumPreparedCells,
    maximumTextureDimension,
    metadataOnly = false,
    compactBricks = true,
    requestedLevel = null,
    allowTransientHierarchy = false,
    opacityClass = "generic",
    excludePhysicalBoundsM,
  } = input;
  const ownerRef = useRef(`vtk:bricked-volume:${++volumeBrickScopeSequence}`);
  const clientRef = useRef<VolumeBrickPreparationClient | null>(null);
  const residentReservations = useRef(new Set<number>());
  const transferRef = useRef(transfer);
  transferRef.current = transfer;
  const transferKey = volumeOpacitySelectionKey(fieldName, transfer);
  const exclusionKey = JSON.stringify(excludePhysicalBoundsM ?? null);
  const stableExclusion = useMemo(() => JSON.parse(exclusionKey) as typeof excludePhysicalBoundsM, [exclusionKey]);
  const identity = useMemo(() => JSON.stringify([
    runDir,
    step?.index ?? null,
    step?.publicationGeneration ?? null,
    fieldName,
    transferKey,
    maximumPreparedCells,
    maximumTextureDimension,
    metadataOnly,
    compactBricks,
    requestedLevel,
    allowTransientHierarchy,
    input.background,
    opacityClass,
    exclusionKey,
  ]), [input.background, allowTransientHierarchy, compactBricks, exclusionKey, fieldName, maximumPreparedCells, maximumTextureDimension, metadataOnly, opacityClass, requestedLevel, runDir, step?.index, step?.publicationGeneration, transferKey]);
  const [state, setState] = useState<BrickedVolumeState>({
    identity: "",
    commitToken: 0,
    reservationId: null,
    hierarchy: null,
    plan: null,
    stats: null,
    bricks: [],
    loading: false,
    error: null,
  });
  const active = enabled && Boolean(step?.grid && fieldName);
  const budgetRetry = useIoBudgetRetry(state.error);

  useEffect(() => () => {
    clientRef.current?.shutdown();
    clientRef.current = null;
    for (const id of residentReservations.current) visualizationResourceGovernor.release(id);
    residentReservations.current.clear();
  }, []);

  useEffect(() => {
    for (const id of residentReservations.current) {
      if (id === state.reservationId) continue;
      visualizationResourceGovernor.release(id);
      residentReservations.current.delete(id);
    }
  }, [state.reservationId]);

  useEffect(() => {
    if (!enabled || !step || !step.grid || !fieldName) {
      clientRef.current?.shutdown();
      clientRef.current = null;
      setState(current => current.bricks.length || current.plan || current.hierarchy || current.stats || current.loading || current.error
        ? { identity, commitToken: 0, reservationId: null, hierarchy: null, plan: null, stats: null, bricks: [], loading: false, error: null }
        : current);
      return;
    }
    const grid = step.grid;
    const controller = new AbortController();
    let activeClient: VolumeBrickPreparationClient | null = null;
    setState((current) => ({ ...current, loading: true, error: null, requiresBoundedOverview: false }));
    let admissionReservationId: number | null = null;
    void (async () => {
      const transientKey = JSON.stringify([
        runDir, step.index, step.publicationGeneration, fieldName, opacityClass,
      ]);
      const declaredHierarchy = step.fieldBrickHierarchies?.[fieldName];
      let transient = transientHierarchyCache.get(transientKey) ?? null;
      if (!declaredHierarchy && !transient && allowTransientHierarchy) {
        transient = await buildTransientVolumeHierarchy({
          runDir,
          step,
          fieldName,
          opacityClass,
          signal: controller.signal,
        });
        transientHierarchyCache.set(transientKey, transient);
        while (transientHierarchyCache.size > 50) {
          transientHierarchyCache.delete(transientHierarchyCache.keys().next().value!);
        }
      }
      const hierarchy = declaredHierarchy
        ? await readFieldBrickHierarchy(runDir, step, fieldName)
        : transient?.hierarchy
          ?? (() => { throw new Error(`'${fieldName}' has no usable brick hierarchy.`); })();
      if (!hierarchy.cullingSafe) throw new Error("Volume brick skipping requires safe decoded-value bounds");
      const finiteRanges = hierarchy.levels[0].bricks
        .map((brick) => brick.scalarRange)
        .filter((range) => range.finiteCount > 0 && range.minimum != null && range.maximum != null);
      const rangeStats = finiteRanges.length > 0 ? {
        min: Math.min(...finiteRanges.map((range) => range.minimum!)),
        max: Math.max(...finiteRanges.map((range) => range.maximum!)),
        p2: Math.min(...finiteRanges.map((range) => range.minimum!)),
        p98: Math.max(...finiteRanges.map((range) => range.maximum!)),
      } : null;
      let sampledStats = transient?.stats ?? sampledStatsCache.get(transientKey) ?? null;
      if (!sampledStats && declaredHierarchy && !grid.terrainFollowing
          && transferRef.current.scalarTransform === "none") {
        sampledStats = await sampleVolumeFieldStats({
          runDir,
          step,
          fieldName,
          signal: controller.signal,
        });
        sampledStatsCache.set(transientKey, sampledStats);
        while (sampledStatsCache.size > 50) sampledStatsCache.delete(sampledStatsCache.keys().next().value!);
      }
      const stats = sampledStats && rangeStats ? { ...sampledStats, min: rangeStats.min, max: rangeStats.max } : sampledStats ?? rangeStats;
      if (!stats) return { hierarchy, plan: null, stats, bricks: [] };
      const activeTransfer = transferRef.current;
      const policy = resolveVolumeTransferPolicy(
        fieldName,
        activeTransfer.colormap,
        stats,
        activeTransfer.scalarTransform,
        activeTransfer.stormStructure,
      );
      const selectionPoints = activeTransfer.scalarTransform === "horizontal-level-anomaly"
        ? [[stats.min, 1], [stats.max, 1]] as const
        : volumeOpacityPoints(policy, activeTransfer.opacityScale)
            .map((point) => [point.scalar, point.opacity] as const);
      const validityFactor = volumeValidityBudgetFactor(hierarchy, grid.terrainFollowing === true || !declaredHierarchy);
      const maximumResidentBytes = Math.floor(maximumPreparedCells * Float32Array.BYTES_PER_ELEMENT / validityFactor);
      let selectedPlan = selectVolumeHierarchyPlan(hierarchy, selectionPoints, {
        maximumResidentBytes,
        terrainFollowing: grid.terrainFollowing,
        haloSamples: needsVolumeReconstruction(grid) ? 2 : volumeHaloSamples(grid),
        requestedLevel,
        excludePhysicalBoundsM: stableExclusion,
      });
      if (needsVolumeReconstruction(grid)) {
        for (;;) {
          const reconstructed = reconstructVolumePlan(hierarchy, grid, selectedPlan);
          if (reconstructed.residentBytes <= maximumResidentBytes) { selectedPlan = reconstructed; break; }
          if (selectedPlan.level + 1 >= hierarchy.levels.length) throw new Error("Common volume reconstruction exceeds the admitted working set");
          selectedPlan = selectVolumeHierarchyPlan(hierarchy, selectionPoints, {
            maximumResidentBytes, terrainFollowing: grid.terrainFollowing, haloSamples: 2,
            requestedLevel: selectedPlan.level + 1, excludePhysicalBoundsM: stableExclusion,
          });
        }
      }
      const scalarPlan = stableExclusion || !compactBricks || needsVolumeReconstruction(grid) ? selectedPlan : compactVolumeBrickPlan(
        hierarchy,
        selectedPlan,
        { maximumResidentBytes, terrainFollowing: grid.terrainFollowing },
      );
      const plan = reserveVolumeValidity(scalarPlan, validityFactor);
      if (metadataOnly) return { hierarchy, plan, stats, bricks: [] };
      const claims = plan.items.flatMap((item) => {
        const inputBytes = item.window.maxOutputBytes ?? 0;
        const preparedBytes = item.preparedScalarBytes;
        const shape = item.reconstruction?.shape ?? item.window.extent.map((count, axis) =>
          Math.ceil(count / item.window.stride[axis]));
        const gpuBytes = maximumTextureDimension == null ? preparedBytes
          : Math.ceil(planVolumeTextureLayout(shape, maximumTextureDimension).textureCells * 4 * validityFactor);
        return [
          visualizationReservationClaim(
            `${ownerRef.current}:${identity}:${item.brickId}:input`,
            "worker-resident",
            inputBytes + MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES,
          ),
          visualizationReservationClaim(
            `${ownerRef.current}:${identity}:${item.brickId}:prepared`,
            "prepared-main",
            preparedBytes,
          ),
          visualizationReservationClaim(
            `${ownerRef.current}:${identity}:${item.brickId}:gpu`,
            "gpu",
            gpuBytes,
          ),
          ...(activeTransfer.scalarTransform === "horizontal-level-anomaly" ? [
            visualizationReservationClaim(
              `${ownerRef.current}:${identity}:${item.brickId}:batch-scratch`,
              "worker-scratch",
              preparedBytes * 2,
            ),
          ] : []),
        ];
      });
      const admission = visualizationResourceGovernor.reserve({
        owner: ownerRef.current,
        generation: identity,
        priority: "visible",
        claims,
      });
      if (!admission.ok) throw new Error(admission.reason);
      admissionReservationId = admission.reservation.id;
      if (!clientRef.current && "Worker" in globalThis) {
        clientRef.current = new VolumeBrickPreparationClient();
      }
      const readItem = (item: typeof plan.items[number], retry = false) => readField(
        runDir,
        step.index,
        fieldName,
        step.publicationGeneration,
        {
          signal: controller.signal,
          window: item.window,
          cacheRetention: "transient",
          ownership: "transferable",
          latestScope: `${ownerRef.current}:${fieldName}${retry ? ":retry" : ""}`,
          brickHierarchyLevel: hierarchy.levels[plan.level].source.kind === "npy_lod"
            ? plan.level
            : undefined,
        },
      );
      if (activeTransfer.scalarTransform === "horizontal-level-anomaly") {
        let preparedBatch: PreparedVtkImage[] | null = null;
        let lastError: unknown = null;
        for (let attempt = 0; attempt < 2 && !preparedBatch; attempt += 1) {
          const fields = [];
          for (const item of plan.items) fields.push(await readItem(item, attempt > 0));
          if (!clientRef.current && "Worker" in globalThis) {
            clientRef.current = new VolumeBrickPreparationClient();
          }
          const client = clientRef.current;
          if (!client) throw new Error("Progressive volume bricks require module workers");
          activeClient = client;
          try {
            preparedBatch = await client.prepareAnomalyBatch(fields.map((field, index) => ({
              kind: "scalar" as const,
              reconstruction: plan.items[index].reconstruction,
              field,
              grid,
              refineForDisplay: false,
              scalarTransform: "horizontal-level-anomaly" as const,
              datasetHandoff: "render-only" as const,
              excludePhysicalBoundsM: stableExclusion,
            })), controller.signal);
          } catch (error) {
            if (controller.signal.aborted) throw error;
            lastError = error;
            client.shutdown();
            if (clientRef.current === client) clientRef.current = null;
          } finally {
            activeClient = null;
          }
        }
        if (!preparedBatch) throw lastError ?? new Error("Volume anomaly worker failed twice");
        if (preparedBatch.length !== plan.items.length) {
          throw new Error("Volume anomaly batch returned an incomplete brick generation");
        }
        const bricks = preparedBatch.map((prepared, index) => {
          const item = plan.items[index];
          if (volumeImageBytes(prepared) > item.preparedScalarBytes) {
            throw new Error("Prepared anomaly brick exceeded its admitted output bound");
          }
          return { brickId: item.brickId, prepared: withVolumeBrickOwnership(prepared, item) };
        });
        return {
          hierarchy,
          plan,
          stats: preparedBatch[0]?.stats ?? stats,
          bricks,
        };
      }
      const bricks: PreparedVolumeBrick[] = [];
      for (const item of plan.items) {
        let field = await readItem(item);
        let prepared: PreparedVtkImage | null = null;
        let lastError: unknown = null;
        for (let attempt = 0; attempt < 2 && !prepared; attempt += 1) {
          if (attempt > 0) {
            field = await readItem(item, true);
          }
          if (!clientRef.current && "Worker" in globalThis) {
            clientRef.current = new VolumeBrickPreparationClient();
          }
          const client = clientRef.current;
          if (!client) throw new Error("Progressive volume bricks require module workers");
          activeClient = client;
          try {
            prepared = await client.prepare({
              kind: "scalar",
              background: input.background ?? undefined,
              reconstruction: item.reconstruction,
              field,
              grid,
              refineForDisplay: false,
              datasetHandoff: "render-only",
              excludePhysicalBoundsM: stableExclusion,
            }, controller.signal);
          } catch (error) {
            if (controller.signal.aborted) throw error;
            lastError = error;
            client.shutdown();
            if (clientRef.current === client) clientRef.current = null;
          } finally {
            activeClient = null;
          }
        }
        if (!prepared) throw lastError ?? new Error("Volume brick worker failed twice");
        const ceiling = item.preparedScalarBytes;
        if (volumeImageBytes(prepared) > ceiling) {
          throw new Error("Prepared volume brick exceeded its admitted output bound");
        }
        if (!prepared.focusEmpty) bricks.push({ brickId: item.brickId, prepared: withVolumeBrickOwnership(prepared, item) });
      }
      return { hierarchy, plan, stats, bricks };
    })().then(
      (result) => {
        if (controller.signal.aborted) return;
        const commitToken = ++volumeBrickCommitSequence;
        let reservationId: number | null = null;
        if (result.bricks.length > 0 && admissionReservationId != null) {
          const resident = visualizationResourceGovernor.transition(admissionReservationId,
            result.bricks.flatMap(({ brickId, prepared }) => [
              visualizationBufferClaim(prepared.values.buffer, "prepared-main"),
              ...(prepared.validity ? [visualizationBufferClaim(prepared.validity.buffer, "prepared-main")] : []),
              { allocationId: `${ownerRef.current}:${identity}:${commitToken}:${brickId}:gpu`,
                stage: "gpu" as const, bytes: volumeImageBytes(prepared, maximumTextureDimension), accounting: "resident" as const },
            ]));
          if (!resident.ok) throw new Error(resident.reason);
          reservationId = resident.reservation.id;
          residentReservations.current.add(reservationId);
          admissionReservationId = null;
        }
        setState({
          identity,
          commitToken,
          reservationId,
          ...result,
          loading: false,
          error: null,
          requiresBoundedOverview: false,
        });
      },
    ).catch((error) => {
        if (!controller.signal.aborted) setState((current) => ({
          ...current,
          identity,
          reservationId: null,
          bricks: [], hierarchy: null, plan: null, stats: null,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
          requiresBoundedOverview: error instanceof VolumeResidencyError
            && allowTransientHierarchy && !step.fieldBrickHierarchies?.[fieldName],
        }));
      }).finally(() => {
      if (admissionReservationId != null) {
        visualizationResourceGovernor.release(admissionReservationId);
      }
    });
    return () => {
      controller.abort();
      if (activeClient) {
        activeClient.shutdown();
        if (clientRef.current === activeClient) clientRef.current = null;
      }
    };
  }, [allowTransientHierarchy, budgetRetry, compactBricks, enabled, exclusionKey, fieldName, identity, maximumPreparedCells, metadataOnly, opacityClass, requestedLevel, runDir, stableExclusion, step, transferKey]);

  const current = active && state.identity === identity;
  const committed = !active
    ? { ...state, bricks: NO_BRICKS, hierarchy: null, plan: null, stats: null, loading: false, error: null }
    : current ? state : { ...state, loading: true };
  return { ...committed, current };
}
