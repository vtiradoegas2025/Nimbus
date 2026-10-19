import {
  evaluateExpressionDerivedField,
  type ExpressionDerivedFieldDefinition,
} from "./expressionDerivedFields";
import type { NpyN } from "@/lib/platform/npy";

/** Serializable request boundary for dataset-local expression evaluation.
 * Compiled evaluator functions never cross the worker boundary. */
export interface ExpressionDerivedWorkerRequest {
  definition: ExpressionDerivedFieldDefinition;
  inputs: Readonly<Record<string, NpyN>>;
  maxElements?: number;
}

export interface ExpressionDerivedWorkerReply {
  ok: boolean;
  result?: NpyN;
  error?: string;
}

export function evaluateExpressionWorkerRequest(
  request: ExpressionDerivedWorkerRequest,
): NpyN {
  return evaluateExpressionDerivedField(request.definition, request.inputs, {
    maxElements: request.maxElements,
  });
}

