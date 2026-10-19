import { Parser } from "expr-eval";
import type { NpyN } from "@/lib/platform/npy";

/* Voxel-wise math-expression evaluator. `field@step:N` cross-frame syntax is
 * parsed at compile time into frameRefs; expr-eval is sandboxed (no member access). */

const parser = new Parser({
  allowMemberAccess: false,
  operators: {
    add: true,
    subtract: true,
    multiply: true,
    divide: true,
    remainder: true,
    power: true,
    comparison: true,
    logical: true,
    conditional: true,
    in: false,
    assignment: false,
  },
});

/** One cross-frame field reference parsed out of an expression. */
export interface FrameRef {
  /** Field name (manifest id). */
  field: string;
  /** Frame offset relative to the spec's anchor frame. e.g. -1, +2. */
  offset: number;
  /** Synthetic identifier substituted into the rewritten expression. */
  alias: string;
}

export interface CompiledExpression {
  /** Original user-authored expression. */
  source: string;
  /** Identifier order the compiled function expects. */
  paramOrder: string[];
  /** Bare field references (no @step) — to be loaded at the anchor frame. */
  bareFields: string[];
  /** Cross-frame references parsed out of `field@step:N` syntax. */
  frameRefs: FrameRef[];
  /** The compiled JS function. Call as `fn(...values)` with values in
   *  `paramOrder`. Returns NaN if expression evaluates to non-number. */
  fn: (...args: number[]) => number;
}

/** Compile a user expression. Validates declared inputs against parsed free variables. */
export function compileExpression(
  source: string,
  declaredInputs: ReadonlyArray<string>,
): CompiledExpression {
  if (typeof source !== "string" || !source.trim()) {
    throw new Error("expression is empty");
  }

  // Rewrite `field@step:N` to valid identifiers before parsing (@ and : are not valid in expr-eval).
  const frameRefs: FrameRef[] = [];
  const rewritten = source.replace(
    /([A-Za-z_][A-Za-z_0-9]*)@step:(-?\d+)/g,
    (_, field: string, offsetStr: string) => {
      const offset = Number.parseInt(offsetStr, 10);
      const sanitized = offset < 0 ? `m${Math.abs(offset)}` : `p${offset}`;
      const alias = `__${field}_step_${sanitized}`;
      if (!frameRefs.some((r) => r.alias === alias)) {
        frameRefs.push({ field, offset, alias });
      }
      return alias;
    },
  );

  let parsed;
  try {
    parsed = parser.parse(rewritten);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`expression parse failed: ${msg}`);
  }

  const usedVars: string[] = parsed.variables({ withMembers: false });
  // Bare fields = used vars that aren't @step aliases.
  const aliasSet = new Set(frameRefs.map((r) => r.alias));
  const bareFields = usedVars.filter((v) => !aliasSet.has(v));

  if (declaredInputs.length > 0) {
    const declared = new Set(declaredInputs);
    for (const v of bareFields) {
      if (!declared.has(v)) {
        throw new Error(`expression uses field '${v}' that wasn't listed in inputs[]`);
      }
    }
    for (const v of declaredInputs) {
      if (!usedVars.includes(v)) {
        throw new Error(`declared input '${v}' isn't referenced by the expression`);
      }
    }
  }

  const paramOrder = [...bareFields, ...frameRefs.map((r) => r.alias)];
  const fn = parsed.toJSFunction(paramOrder.join(","));

  return {
    source,
    paramOrder,
    bareFields,
    frameRefs,
    fn: fn as (...args: number[]) => number,
  };
}

/** Absolute backstop on a synchronous per-voxel evaluation. Above this the JS
 *  loop would freeze the UI for many seconds (or OOM on the output array), so
 *  fail loud regardless of caller intent. A chunked / worker evaluator with
 *  abort support is the longer-term path. */
const HARD_MAX_ELEMENTS = 50_000_000;

/** Evaluate a compiled expression over per-voxel inputs of identical shape.
 *  Missing inputs throw; opts.maxElements lets callers enforce a per-use ceiling. */
export function evaluateExpression(
  compiled: CompiledExpression,
  inputs: Record<string, NpyN>,
  opts: { maxElements?: number } = {},
): NpyN {
  if (compiled.paramOrder.length === 0) {
    throw new Error("expression has no inputs — evaluate would be a no-op");
  }
  const firstName = compiled.paramOrder[0];
  const first = inputs[firstName];
  if (!first) {
    throw new Error(`missing input '${firstName}' for expression evaluation`);
  }
  const shape = first.shape;
  const n = first.data.length;

  // Hard backstop: synchronous per-voxel loop freezes the UI above this.
  const cap = Math.min(opts.maxElements ?? HARD_MAX_ELEMENTS, HARD_MAX_ELEMENTS);
  if (n > cap) {
    throw new Error(
      `expression input too large (${n} elements > ${cap}); reduce it with a slice or reduce at the source first`,
    );
  }

  const arrs: Float32Array[] = new Array(compiled.paramOrder.length);
  for (let i = 0; i < compiled.paramOrder.length; i++) {
    const name = compiled.paramOrder[i];
    const v = inputs[name];
    if (!v) {
      throw new Error(`missing input '${name}' for expression evaluation`);
    }
    if (v.data.length !== n) {
      throw new Error(
        `input '${name}' has shape ${v.shape.join("×")}, expected ${shape.join("×")}`,
      );
    }
    arrs[i] = v.data;
  }

  const out = new Float32Array(n);
  const args = new Array<number>(arrs.length);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < arrs.length; p++) {
      args[p] = arrs[p][i];
    }
    const v = compiled.fn(...args);
    out[i] = typeof v === "number" ? v : NaN;
  }

  return { shape: [...shape], data: out };
}
