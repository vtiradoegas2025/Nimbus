import { useEffect, useMemo, useRef, useState } from "react";
import type { PreparedVectorImage } from "./vtkPreparation";
import { streamlineResultTransferables, type StreamlineExtractionParameters, type StreamlineExtractionResult } from "./streamlineExtractionProtocol";
import { extractStreamlinesInSharedWorker } from "./streamlineExtractionClient";
import { extractStreamlinesFromWorkerOwnedHandle } from "./streamlineWorkerOwnedPreparation";
import {
  resolvePreparedDataset,
  type VtkPreparedDatasetHandle,
} from "./vtkPreparedDatasetRegistry";
import { visualizationBufferClaim, visualizationResourceGovernor } from "@/lib/core/visualizationResourceGovernor";

interface Input {
  enabled: boolean;
  preparedHandle: VtkPreparedDatasetHandle<PreparedVectorImage> | null;
  parameters: StreamlineExtractionParameters | null;
  key: string;
}
interface State {
  preparedHandle: VtkPreparedDatasetHandle<PreparedVectorImage>;
  key: string;
  result: StreamlineExtractionResult | null;
  error: string | null;
}
let streamlineResourceOwnerSequence = 0;

export function useStreamlineExtraction(input: Input): {
  result: StreamlineExtractionResult | null;
  loading: boolean;
  error: string | null;
} {
  const [state, setState] = useState<State | null>(null);
  const key = useMemo(() => JSON.stringify([
    input.key, input.parameters?.integrationStep, input.parameters?.maxSteps,
    input.parameters?.tubeRadius, input.parameters?.scale,
    input.parameters ? Array.from(input.parameters.seeds) : null,
  ]), [input.key, input.parameters]);
  const resourceOwnerRef = useRef<string>();
  if (!resourceOwnerRef.current) resourceOwnerRef.current = `vtk:streamline-result:${++streamlineResourceOwnerSequence}`;

  useEffect(() => {
    if (!input.enabled || !input.preparedHandle || !input.parameters) {
      setState(null);
      return;
    }
    const controller = new AbortController();
    const preparedHandle = input.preparedHandle;
    const prepared = preparedHandle.residency === "main"
      ? resolvePreparedDataset(preparedHandle)
      : null;
    const parameters = input.parameters;
    if (preparedHandle.residency === "worker") {
      void extractStreamlinesFromWorkerOwnedHandle(
        preparedHandle,
        parameters,
        controller.signal,
      ).then(
        (result) => {
          if (!controller.signal.aborted) setState({ preparedHandle, key, result, error: null });
        },
        (error) => {
          if (!controller.signal.aborted) setState({
            preparedHandle,
            key,
            result: null,
            error: error instanceof Error ? error.message : String(error),
          });
        },
      );
    } else {
      if (typeof Worker !== "function") {
        setState({
          preparedHandle,
          key,
          result: null,
          error: "Streamline extraction requires module-worker support",
        });
        return () => controller.abort();
      }
      void extractStreamlinesInSharedWorker(prepared!, parameters, controller.signal).then(
        (result) => {
          if (!controller.signal.aborted) setState({ preparedHandle, key, result, error: null });
        },
        (error) => {
          if (controller.signal.aborted) return;
          setState({
            preparedHandle,
            key,
            result: null,
            error: error instanceof Error ? error.message : String(error),
          });
        },
      );
    }
    return () => controller.abort();
    // Equal seed/parameter objects must not cancel and restart an active worker.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input.enabled, input.preparedHandle, key]);

  const matches = input.enabled && input.preparedHandle != null && state != null
    && state.preparedHandle === input.preparedHandle && state.key === key;
  const ownedResult = matches ? state.result : null;
  useEffect(() => {
    if (!ownedResult) return;
    const outcome = visualizationResourceGovernor.reserve({
      owner: resourceOwnerRef.current!,
      generation: key,
      priority: "visible",
      claims: streamlineResultTransferables(ownedResult).map((buffer) => visualizationBufferClaim(buffer, "retained-geometry")),
    });
    if (!outcome.ok) return;
    return () => { visualizationResourceGovernor.release(outcome.reservation.id); };
  }, [key, ownedResult]);
  return {
    result: matches ? state.result : null,
    loading: input.enabled && input.preparedHandle != null && input.parameters != null && !matches,
    error: input.enabled && matches ? state.error : null,
  };
}
