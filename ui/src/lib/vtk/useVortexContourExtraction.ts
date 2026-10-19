import { useEffect, useRef, useState } from "react";
import type { PreparedVtkImage } from "./vtkPreparation";
import {
  vortexContourTransferables,
  vortexExtractionInputMode,
  type VortexContourResult,
  type VortexContourThresholds,
} from "./vortexContourProtocol";
import {
  CONTOUR_WORKER_IDLE_RETIRE_MS,
  extractVortexContoursInSharedWorker,
} from "./vortexContourClient";
import {
  resolvePreparedDataset,
  type VtkPreparedDatasetHandle,
} from "./vtkPreparedDatasetRegistry";
import { extractContoursFromWorkerOwnedHandles } from "./contourWorkerOwnedPreparation";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";

export { CONTOUR_WORKER_IDLE_RETIRE_MS };

interface ExtractionInput {
  enabled: boolean;
  preparedHandle: VtkPreparedDatasetHandle<PreparedVtkImage> | null;
  thresholds: VortexContourThresholds;
  anchor?: readonly [number, number, number] | null;
  condensateHandle?: VtkPreparedDatasetHandle<PreparedVtkImage> | null;
  condensateFloor?: number | null;
}

interface ExtractionState {
  preparedHandle: VtkPreparedDatasetHandle<PreparedVtkImage>;
  thresholds: VortexContourThresholds;
  anchorKey: string;
  condensateHandle: VtkPreparedDatasetHandle<PreparedVtkImage> | null;
  condensateFloor: number | null;
  result: VortexContourResult | null;
  error: string | null;
}

export interface VortexContourExtraction {
  result: VortexContourResult | null;
  loading: boolean;
  error: string | null;
}

let contourResourceOwnerSequence = 0;

function contourAnchorFromKey(key: string): [number, number, number] | null {
  if (!key) return null;
  const values = key.split("\u0000").map(Number);
  return values.length === 3 && values.every(Number.isFinite)
    ? [values[0]!, values[1]!, values[2]!]
    : null;
}

export function useVortexContourExtraction(input: ExtractionInput): VortexContourExtraction {
  const {
    enabled,
    preparedHandle,
    thresholds,
    anchor = null,
    condensateHandle = null,
    condensateFloor = null,
  } = input;
  const coreThreshold = thresholds.core;
  const oppositeThreshold = thresholds.opposite;
  const mesocycloneThreshold = thresholds.mesocyclone;
  const anchorKey = anchor?.join("\u0000") ?? "";
  const [state, setState] = useState<ExtractionState | null>(null);
  const resourceOwnerRef = useRef<string>();
  if (!resourceOwnerRef.current) {
    resourceOwnerRef.current = `vtk:contour-result:${++contourResourceOwnerSequence}`;
  }

  useEffect(() => {
    const hasContour = coreThreshold != null
      || oppositeThreshold != null
      || mesocycloneThreshold != null;
    const hasPreparedInputs = preparedHandle != null
      && (condensateFloor == null || condensateHandle != null);
    const inputMode = vortexExtractionInputMode(enabled, hasPreparedInputs, hasContour);
    if (inputMode === "stop") {
      setState(null);
      return;
    }
    if (inputMode === "hold" || !preparedHandle) return;

    const controller = new AbortController();
    const prepared = preparedHandle.residency === "main"
      ? resolvePreparedDataset(preparedHandle)
      : null;
    const condensatePrepared = condensateHandle?.residency === "main"
      ? resolvePreparedDataset(condensateHandle)
      : null;
    const requestThresholds: VortexContourThresholds = {
      core: coreThreshold,
      opposite: oppositeThreshold,
      mesocyclone: mesocycloneThreshold,
    };
    const requestAnchor = contourAnchorFromKey(anchorKey);
    const publish = (result: VortexContourResult | null, error: string | null) => {
      if (controller.signal.aborted) return;
      setState({
        preparedHandle,
        thresholds: requestThresholds,
        anchorKey,
        condensateHandle,
        condensateFloor,
        result,
        error,
      });
    };
    if (preparedHandle.residency === "worker") {
      void extractContoursFromWorkerOwnedHandles(
        preparedHandle,
        condensateHandle,
        { thresholds: requestThresholds, anchor: requestAnchor, condensateFloor },
        controller.signal,
      ).then(
        (result) => publish(result, null),
        (error) => {
          if (!controller.signal.aborted) {
            publish(null, error instanceof Error ? error.message : String(error));
          }
        },
      );
    } else {
      if (typeof Worker !== "function") {
        publish(null, "Contour extraction requires module-worker support");
        return () => controller.abort();
      }
      void extractVortexContoursInSharedWorker(
        prepared!,
        condensatePrepared,
        { thresholds: requestThresholds, anchor: requestAnchor, condensateFloor },
        controller.signal,
      ).then(
        (result) => publish(result, null),
        (error) => {
          if (controller.signal.aborted) return;
          publish(null, error instanceof Error ? error.message : String(error));
        },
      );
    }
    return () => controller.abort();
  }, [
    enabled,
    preparedHandle,
    coreThreshold,
    oppositeThreshold,
    mesocycloneThreshold,
    anchorKey,
    condensateHandle,
    condensateFloor,
  ]);

  const matches = enabled && preparedHandle != null && state != null
    && state.preparedHandle === preparedHandle
    && state.thresholds.core === coreThreshold
    && state.thresholds.opposite === oppositeThreshold
    && state.thresholds.mesocyclone === mesocycloneThreshold
    && state.anchorKey === anchorKey
    && state.condensateHandle === condensateHandle
    && state.condensateFloor === condensateFloor;
  const ownedResult = matches ? state.result : null;
  useEffect(() => {
    if (!ownedResult) return;
    const outcome = visualizationResourceGovernor.reserve({
      owner: resourceOwnerRef.current!,
      generation: JSON.stringify([anchorKey, coreThreshold, oppositeThreshold, mesocycloneThreshold]),
      priority: "visible",
      claims: vortexContourTransferables(ownedResult).map((buffer) =>
        visualizationBufferClaim(buffer, "retained-geometry")
      ),
    });
    if (!outcome.ok) return;
    return () => { visualizationResourceGovernor.release(outcome.reservation.id); };
  }, [anchorKey, coreThreshold, mesocycloneThreshold, oppositeThreshold, ownedResult]);

  return {
    result: matches ? state.result : null,
    loading: enabled && preparedHandle != null
      && (coreThreshold != null || oppositeThreshold != null || mesocycloneThreshold != null)
      && !matches,
    error: enabled && matches ? state.error : null,
  };
}
