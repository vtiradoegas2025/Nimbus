export function selectRadarFloorRequestState<T extends {
  requestKey: string;
  loading: boolean;
  error: string | null;
}>(state: T, requestedRequestIdentity: string, expectsRequest: boolean, idle: T): T & {
  requestedRequestIdentity: string;
} {
  if (!expectsRequest) return { ...idle, requestedRequestIdentity };
  if (state.requestKey === requestedRequestIdentity) return { ...state, requestedRequestIdentity };
  return { ...state, requestedRequestIdentity, loading: true, error: null };
}

export function radarFloorPresentationOwnership(input: {
  enabled: boolean;
  requestedRequestIdentity: string | null;
  loadedRequestIdentity: string | null;
  requestLoading: boolean;
  encodingLoading: boolean;
  encodingRequired: boolean;
  encodingReady: boolean;
  gpuRequired: boolean;
  gpuAdmitted: boolean;
  attachedLifecycleGeneration: number | null;
  lifecycleGeneration: number;
  error: string | null;
}): { currentRequest: boolean; attached: boolean; loading: boolean; action: "hide" | "retain" | "commit" } {
  const currentRequest = input.requestedRequestIdentity != null
    && input.loadedRequestIdentity === input.requestedRequestIdentity;
  const attached = input.attachedLifecycleGeneration === input.lifecycleGeneration;
  const loading = input.enabled && input.error == null && (
    !currentRequest || input.requestLoading || input.encodingLoading || !attached
      || input.encodingRequired && !input.encodingReady
      || input.gpuRequired && !input.gpuAdmitted
  );
  return {
    currentRequest,
    attached,
    loading,
    action: !input.enabled ? "hide" : input.error != null || loading ? "retain" : "commit",
  };
}
