import { evaluateDerivedField } from "./derivedFieldRegistry";
import type { NpyN } from "@/lib/platform/npy";

/** One slab of a built-in derived field. Inputs are rank-1 and already sliced. */
export interface DerivedFieldWorkerRequest {
  fieldName: string;
  inputs: Readonly<Record<string, NpyN>>;
}

export interface DerivedFieldWorkerReply {
  ok: boolean;
  result?: NpyN;
  error?: string;
}

/** Shared by the worker and by the synchronous fallback used where Worker is absent. */
export function evaluateDerivedWorkerRequest(request: DerivedFieldWorkerRequest): NpyN {
  return evaluateDerivedField(request.fieldName, request.inputs);
}
