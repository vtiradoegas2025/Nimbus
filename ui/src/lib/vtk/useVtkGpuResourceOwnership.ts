import { useEffect, useRef, useState } from "react";
import { visualizationResourceGovernor } from "@/lib/core/visualizationResourceGovernor";

let sequence = 0;

export function useVtkGpuResourceOwnership(input: {
  enabled: boolean;
  owner: string;
  generation: string;
  bytes: number;
}): { admitted: boolean; error: string | null } {
  const namespaceRef = useRef<string>();
  if (!namespaceRef.current) namespaceRef.current = `vtk:gpu:${input.owner}:${++sequence}`;
  const namespace = namespaceRef.current;
  const reservationRef = useRef<number | null>(null);
  const [state, setState] = useState({ identity: "", admitted: false, error: null as string | null });
  const identity = `${input.generation}:${input.bytes}`;

  useEffect(() => () => {
    if (reservationRef.current != null) {
      visualizationResourceGovernor.release(reservationRef.current);
      reservationRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!input.enabled || !Number.isSafeInteger(input.bytes) || input.bytes <= 0) {
      if (reservationRef.current != null) {
        visualizationResourceGovernor.release(reservationRef.current);
        reservationRef.current = null;
      }
      setState({ identity, admitted: false, error: input.enabled
        ? "VTK GPU upload byte count is invalid" : null });
      return;
    }
    const outcome = visualizationResourceGovernor.reserve({
      owner: namespace,
      generation: input.generation,
      priority: "visible",
      claims: [{
        allocationId: `${namespace}:${identity}`,
        stage: "gpu",
        bytes: input.bytes,
        accounting: "resident",
      }],
    });
    if (!outcome.ok) {
      setState({ identity, admitted: false, error: outcome.reason });
      return;
    }
    if (reservationRef.current != null) {
      visualizationResourceGovernor.release(reservationRef.current);
    }
    reservationRef.current = outcome.reservation.id;
    setState({ identity, admitted: true, error: null });
  }, [identity, input.bytes, input.enabled, input.generation, namespace]);

  return state.identity === identity
    ? { admitted: state.admitted, error: state.error }
    : { admitted: false, error: null };
}
