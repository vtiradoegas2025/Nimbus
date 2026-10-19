import { useEffect, useReducer, useRef } from "react";
import type { PreparedVtkImage } from "./vtkPreparation";
import type {
  VortexContourResult,
  VortexContourThresholds,
} from "./vortexContourProtocol";
import { vortexContourTransferables } from "./vortexContourProtocol";
import type { VtkPreparedDatasetHandle } from "./vtkPreparedDatasetRegistry";
import { extractContoursFromWorkerOwnedHandles } from "./contourWorkerOwnedPreparation";
import {
  visualizationBufferClaim,
  visualizationResourceGovernor,
} from "@/lib/core/visualizationResourceGovernor";

export interface SupercellWorkerOwnedContourRequest {
  generation: string;
  preparedHandle: VtkPreparedDatasetHandle<PreparedVtkImage>;
  condensateHandle: VtkPreparedDatasetHandle<PreparedVtkImage> | null;
  thresholds: VortexContourThresholds;
  anchor: readonly [number, number, number] | null;
  condensateFloor: number | null;
}

export interface SupercellContourGenerationState {
  generation: string | null;
  loading: boolean;
  result: VortexContourResult | null;
  error: string | null;
}

export type SupercellContourGenerationAction =
  | { type: "clear" }
  | { type: "begin"; generation: string }
  | { type: "resolve"; generation: string; result: VortexContourResult }
  | { type: "reject"; generation: string; error: string };

export function reduceSupercellContourGeneration(
  state: SupercellContourGenerationState,
  action: SupercellContourGenerationAction,
): SupercellContourGenerationState {
  if (action.type === "clear") {
    return { generation: null, loading: false, result: null, error: null };
  }
  if (action.type === "begin") {
    return { generation: action.generation, loading: true, result: null, error: null };
  }
  if (state.generation !== action.generation) return state;
  if (action.type === "resolve") {
    return { generation: action.generation, loading: false, result: action.result, error: null };
  }
  return { generation: action.generation, loading: false, result: null, error: action.error };
}

let resultOwnerSequence = 0;

export function useSupercellWorkerOwnedContour(
  enabled: boolean,
  request: SupercellWorkerOwnedContourRequest | null,
): {
  result: VortexContourResult | null;
  loading: boolean;
  error: string | null;
} {
  const [state, dispatch] = useReducer(reduceSupercellContourGeneration, {
    generation: null,
    loading: false,
    result: null,
    error: null,
  });
  const ownerRef = useRef<string>();
  if (!ownerRef.current) ownerRef.current = `vtk:supercell-worker-contour:${++resultOwnerSequence}`;

  useEffect(() => {
    if (!enabled || !request) {
      dispatch({ type: "clear" });
      return;
    }
    const controller = new AbortController();
    dispatch({ type: "begin", generation: request.generation });
    void extractContoursFromWorkerOwnedHandles(
      request.preparedHandle,
      request.condensateHandle,
      {
        thresholds: request.thresholds,
        anchor: request.anchor ? [...request.anchor] as [number, number, number] : null,
        condensateFloor: request.condensateFloor,
      },
      controller.signal,
    ).then(
      (result) => {
        dispatch({ type: "resolve", generation: request.generation, result });
      },
      (error) => {
        if (controller.signal.aborted) return;
        dispatch({
          type: "reject",
          generation: request.generation,
          error: error instanceof Error ? error.message : String(error),
        });
      },
    );
    return () => {
      controller.abort();
    };
  }, [enabled, request]);

  const current = enabled && request != null && state.generation === request.generation;
  const result = current ? state.result : null;
  useEffect(() => {
    if (!result || !request) return;
    const resources = visualizationResourceGovernor.reserve({
      owner: ownerRef.current!,
      generation: request.generation,
      priority: "visible",
      claims: vortexContourTransferables(result).map((buffer) =>
        visualizationBufferClaim(buffer, "retained-geometry")
      ),
    });
    if (!resources.ok) return;
    return () => { visualizationResourceGovernor.release(resources.reservation.id); };
  }, [request, result]);

  return {
    result,
    loading: current ? state.loading : enabled && request != null,
    error: current ? state.error : null,
  };
}
