import { volumeImageBytes, volumeValidityBudgetFactor, reserveVolumeValidity } from "@/lib/vtk/volumeValidityBudget";
import { reconstructVolumePlan, needsVolumeReconstruction } from "./volumeReconstruction";
import { useEffect, useMemo, useRef, useState } from "react";
import type { NestedVolumeOwnershipLayer } from "@/lib/run/outputHierarchy";
import { readField, readFieldBrickHierarchy } from "@/lib/run/runOutputs";
import { selectVolumeHierarchyPlan, type VolumeBrickPlan } from "@/lib/plot/fieldBrickHierarchy";
import { planNestedVolumeGeneration } from "./nestedVolumeCoordinator";
import { partitionNestedVolume } from "./nestedVolumeGeometry";
import { uniqueVolumeImages, volumeHaloSamples, withVolumeBrickOwnership } from "./volumeBrickGeometry";
import { VolumeBrickPreparationClient } from "./volumeBrickClient";
import { planVolumeTextureLayout } from "./volumeTexturePacking";
import type { PreparedVtkImage, ScalarFieldTransform } from "./vtkPreparation";
import { resolveVolumeTransferPolicy, volumeOpacityPoints, volumeOpacitySelectionKey } from "@/lib/plot/volumeTransferPolicy";
import type { VolumeBackground } from "./volumeBackground";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";

export interface NestedPreparedVolumeBrick {
  levelId: string;
  brickId: string;
  prepared: PreparedVtkImage;
}

interface State {
  identity: string;
  plans: Array<{ levelId: string; plan: VolumeBrickPlan }>;
  bricks: NestedPreparedVolumeBrick[];
  stats: { min: number; max: number; p2: number; p98: number } | null;
  loading: boolean;
  error: string | null;
}

let sequence = 0;

export function useNestedBrickedVolumePreparation(input: {
  enabled: boolean;
  layers: readonly NestedVolumeOwnershipLayer[];
  fieldName: string | null;
  transfer: { colormap: string; opacityScale: number; scalarTransform: ScalarFieldTransform; stormStructure?: boolean };
  background?: VolumeBackground | null;
  maximumPreparedCells: number;
  maximumTextureDimension?: number;
  requestedLevel?: number | null;
}): State & { current: boolean } {
  const ownerRef = useRef(`vtk:nested-volume:${++sequence}`);
  const clientsRef = useRef<VolumeBrickPreparationClient[]>([]);
  const layersKey = JSON.stringify(input.layers.map((layer) => ({
    levelId: layer.level.levelId,
    runDir: layer.level.runDir,
    step: layer.level.step.index,
    generation: layer.level.step.publicationGeneration,
    exclusion: layer.excludeChildBoundsM,
  })));
  const transferRef = useRef(input.transfer);
  transferRef.current = input.transfer;
  const transferKey = volumeOpacitySelectionKey(input.fieldName, input.transfer);
  const identity = useMemo(() => JSON.stringify([
    layersKey, input.fieldName, transferKey, input.maximumPreparedCells, input.background,
    input.requestedLevel ?? null, input.maximumTextureDimension,
  ]), [input.background, input.fieldName, input.maximumPreparedCells, input.maximumTextureDimension, input.requestedLevel, layersKey, transferKey]);
  const stableLayers = useMemo(() => input.layers, [input.layers]);
  const [state, setState] = useState<State>({
    identity: "", plans: [], bricks: [], stats: null, loading: false, error: null,
  });

  useEffect(() => () => {
    clientsRef.current.forEach((client) => client.shutdown());
    clientsRef.current = [];
  }, []);

  useEffect(() => {
    if (!input.enabled || !input.fieldName || stableLayers.length < 2) return;
    const controller = new AbortController();
    const transfer = transferRef.current;
    const clients: VolumeBrickPreparationClient[] = [];
    clientsRef.current.forEach((client) => client.shutdown());
    clientsRef.current = clients;
    setState((current) => ({ ...current, loading: true, error: null }));
    let aggregateReservation: number | null = null;
    void (async () => {
      const fieldName = input.fieldName!;
      const hierarchies = await Promise.all(stableLayers.map((layer) =>
        readFieldBrickHierarchy(layer.level.runDir, layer.level.step, fieldName)));
      if (controller.signal.aborted) throw new DOMException("Nested volume cancelled", "AbortError");
      const statsByLayer = hierarchies.map((hierarchy) => {
        const ranges = hierarchy.levels[0].bricks.map((brick) => brick.scalarRange)
          .filter((range) => range.finiteCount > 0 && range.minimum != null && range.maximum != null);
        if (ranges.length === 0) throw new Error(`${hierarchy.field} has no finite nested brick range`);
        return {
          min: Math.min(...ranges.map((range) => range.minimum!)),
          max: Math.max(...ranges.map((range) => range.maximum!)),
          p2: Math.min(...ranges.map((range) => range.minimum!)),
          p98: Math.max(...ranges.map((range) => range.maximum!)),
        };
      });
      const perLayerCells = Math.max(1, Math.floor(input.maximumPreparedCells / stableLayers.length));
      const plans = hierarchies.map((hierarchy, index) => {
        const stats = statsByLayer.at(-1)!;
        const policy = resolveVolumeTransferPolicy(
          fieldName, transfer.colormap, stats, transfer.scalarTransform, transfer.stormStructure);
        const opacity = transfer.scalarTransform === "horizontal-level-anomaly"
          ? [[stats.min, 1], [stats.max, 1]] as const
          : volumeOpacityPoints(policy, transfer.opacityScale)
              .map((point) => [point.scalar, point.opacity] as const);
        const grid = stableLayers[index].level.step.grid!;
        const validityFactor = volumeValidityBudgetFactor(hierarchy, grid.terrainFollowing === true);
        const maximumResidentBytes = Math.floor(perLayerCells * 4 / validityFactor);
        let requestedLevel = input.requestedLevel;
        for (;;) {
          const selected = selectVolumeHierarchyPlan(hierarchy, opacity, {
            maximumResidentBytes, terrainFollowing: grid.terrainFollowing,
            haloSamples: needsVolumeReconstruction(grid) ? 2 : volumeHaloSamples(grid), requestedLevel,
          });
          const plan = reconstructVolumePlan(hierarchy, grid, selected);
          if (plan.residentBytes <= maximumResidentBytes) return reserveVolumeValidity(plan, validityFactor);
          if (plan.level + 1 >= hierarchy.levels.length) throw new Error("Nested reconstruction exceeds the admitted working set");
          requestedLevel = plan.level + 1;
        }
      });
      const generation = planNestedVolumeGeneration(identity, plans.map((plan, index) => ({
        levelId: stableLayers[index].level.levelId,
        plan,
      })));
      if (input.maximumTextureDimension != null) {
        generation.layers.forEach(layer => layer.plan.items.forEach(item => {
          const shape = item.reconstruction?.shape ?? item.window.extent.map((count, axis) => Math.ceil(count / item.window.stride[axis]));
          const layout = planVolumeTextureLayout(shape, input.maximumTextureDimension!);
          const gpu = generation.claims.find(claim => claim.allocationId === `nested:${identity}:${layer.levelId}:${item.brickId}:gpu`);
          if (gpu) gpu.bytes = Math.ceil(item.preparedScalarBytes * layout.textureCells / layout.sourceCells);
        }));
      }
      const aggregateClaims = transfer.scalarTransform === "horizontal-level-anomaly"
        ? [...generation.claims, ...generation.layers.flatMap((layer) =>
            layer.plan.items.map((item) => ({
              allocationId: `nested:${identity}:${layer.levelId}:${item.brickId}:batch-scratch`,
              stage: "worker-scratch" as const,
              bytes: item.preparedScalarBytes * 2,
              accounting: "reserved" as const,
            })))]
        : generation.claims;
      const admission = visualizationResourceGovernor.reserve({
        owner: ownerRef.current,
        generation: identity,
        priority: "visible",
        claims: aggregateClaims,
      });
      if (!admission.ok) throw new Error(admission.reason);
      aggregateReservation = admission.reservation.id;
      if (controller.signal.aborted) throw new DOMException("Nested volume cancelled", "AbortError");

      const bricks: NestedPreparedVolumeBrick[] = [];
      for (let layerIndex = 0; layerIndex < stableLayers.length; layerIndex += 1) {
        const layer = stableLayers[layerIndex];
        const hierarchy = hierarchies[layerIndex];
        const plan = plans[layerIndex];
        if (controller.signal.aborted) throw new DOMException("Nested volume cancelled", "AbortError");
        let prepared: PreparedVtkImage[] | null = null;
        let lastError: unknown = null;
        for (let attempt = 0; attempt < 2 && !prepared; attempt += 1) {
          const fields = [];
          for (const item of plan.items) {
            fields.push(await readField(
              layer.level.runDir,
              layer.level.step.index,
              fieldName,
              layer.level.step.publicationGeneration,
              {
                signal: controller.signal,
                window: item.window,
                ownership: "transferable",
                cacheRetention: "transient",
                latestScope: `${ownerRef.current}:${layer.level.levelId}:${attempt}`,
                brickHierarchyLevel: hierarchy.levels[plan.level].source.kind === "npy_lod"
                  ? plan.level : undefined,
              },
            ));
          }
          if (controller.signal.aborted) throw new DOMException("Nested volume cancelled", "AbortError");
          const client = new VolumeBrickPreparationClient();
          clients.push(client);
          const requests = fields.map((field, index) => ({
            kind: "scalar" as const,
            background: input.background ?? undefined,
            reconstruction: plan.items[index].reconstruction,
            field,
            grid: layer.level.step.grid!,
            refineForDisplay: false,
            scalarTransform: transfer.scalarTransform,
            datasetHandoff: "render-only" as const,
          }));
          try {
            prepared = transfer.scalarTransform === "horizontal-level-anomaly"
              ? await client.prepareAnomalyBatch(requests, controller.signal)
              : await Promise.all(requests.map((request) => client.prepare(request, controller.signal)));
          } catch (error) {
            if (controller.signal.aborted) throw error;
            lastError = error;
            client.shutdown();
          }
        }
        if (!prepared) throw lastError ?? new Error("Nested volume worker failed twice");
        if (prepared.length !== plan.items.length) throw new Error("Nested volume layer prepared incompletely");
        if (transfer.scalarTransform === "horizontal-level-anomaly" && prepared[0]?.stats) statsByLayer[layerIndex] = prepared[0].stats;
        const layerBounds = layer.level.step.grid?.physicalBoundsM;
        const rootBounds = stableLayers[0].level.step.grid?.physicalBoundsM;
        if (!layerBounds || !rootBounds) throw new Error("Nested volume ownership requires physical domain bounds");
        prepared.forEach((value, index) => {
          if (value.focusEmpty) return;
          const owned = withVolumeBrickOwnership(value, plan.items[index]);
          partitionNestedVolume(owned, layerBounds, layer.excludeChildBoundsM ?? null, rootBounds)
            .forEach((piece, part) => bricks.push({
              levelId: layer.level.levelId,
              brickId: `${plan.items[index].brickId}:part:${part}`,
              prepared: piece,
            }));
        });
      }
      return { plans: generation.layers, bricks, stats: statsByLayer.at(-1)! };
    })().then(
      (result) => {
        if (!controller.signal.aborted) setState({ identity, ...result, loading: false, error: null });
      },
      (error) => {
        if (!controller.signal.aborted) setState((current) => ({
          ...current, loading: false, error: error instanceof Error ? error.message : String(error),
        }));
      },
    ).finally(() => {
      if (aggregateReservation != null) visualizationResourceGovernor.release(aggregateReservation);
      if (controller.signal.aborted) clients.forEach((client) => client.shutdown());
    });
    return () => {
      controller.abort();
      clients.forEach((client) => client.shutdown());
    };
  }, [
    identity, input.enabled, input.fieldName, input.maximumPreparedCells,
    input.requestedLevel, stableLayers, transferKey,
  ]);

  const current = state.identity === identity;
  const committed = current ? state : { ...state, loading: input.enabled };
  useEffect(() => {
    if (!current || committed.bricks.length === 0) return;
    const reservation = visualizationResourceGovernor.reserve({
      owner: ownerRef.current,
      generation: identity,
      priority: "visible",
      claims: uniqueVolumeImages(committed.bricks).flatMap((brick) => [
        visualizationBufferClaim(brick.prepared.values.buffer, "prepared-main"),
        ...(brick.prepared.validity ? [visualizationBufferClaim(brick.prepared.validity.buffer, "prepared-main")] : []),
        { allocationId: `${ownerRef.current}:${identity}:${brick.levelId}:${brick.brickId}:gpu`,
          stage: "gpu" as const, bytes: volumeImageBytes(brick.prepared, input.maximumTextureDimension), accounting: "resident" as const },
      ]),
    });
    if (!reservation.ok) return;
    return () => { visualizationResourceGovernor.release(reservation.reservation.id); };
  }, [committed.bricks, current, identity]);
  return { ...committed, current };
}
