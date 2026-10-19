/*
 * Textual completion: parse the dotted-identifier chain before the caret.
 * Returns null rather than guessing when the prefix requires expression evaluation.
 * Names come from the live interpreter so they stay current with imports.
 */

/** A resolved request for names, ready to hand to the interpreter. */
export interface CompletionRequest {
  /**
   * The dotted chain to resolve, outermost first. Empty for a bare identifier,
   * where the candidates are globals, builtins and keywords.
   */
  readonly path: readonly string[];
  /** The partial name typed so far. Empty right after a dot. */
  readonly partial: string;
  /**
   * How many characters before the caret the replacement starts, i.e. the
   * length of `partial`. The caller turns this into a document offset.
   */
  readonly backtrack: number;
}

function isIdentifierStart(character: string): boolean {
  return /[A-Za-z_]/.test(character);
}

function isIdentifierPart(character: string): boolean {
  return /[A-Za-z0-9_]/.test(character);
}

/** Parse the dotted-identifier chain before the caret. Returns null for numeric literals (e.g. 3.14) or expressions requiring evaluation. */
export function parseCompletionPrefix(textBefore: string): CompletionRequest | null {
  let index = textBefore.length;

  // The partial name being typed, if any.
  let end = index;
  while (index > 0 && isIdentifierPart(textBefore[index - 1])) index -= 1;
  const partial = textBefore.slice(index, end);
  // A run of identifier characters starting with a digit is a number, not a
  // name, and there is nothing to complete against it.
  if (partial.length > 0 && !isIdentifierStart(partial[0])) return null;

  const path: string[] = [];
  while (index > 0 && textBefore[index - 1] === ".") {
    index -= 1; // step over the dot
    end = index;
    while (index > 0 && isIdentifierPart(textBefore[index - 1])) index -= 1;
    const segment = textBefore.slice(index, end);
    // A dot preceded by `)`, `]`, a quote, whitespace, or a number: resolving
    // it would mean evaluating an expression, which this will not do.
    if (segment.length === 0 || !isIdentifierStart(segment[0])) return null;
    path.unshift(segment);
  }

  // Neither a dot nor any typed characters -- the caret is in open space and a
  // popup listing every global would be noise rather than help.
  if (path.length === 0 && partial.length === 0) return null;

  return { path, partial, backtrack: partial.length };
}

/** One candidate, as the interpreter describes it. */
export interface CompletionCandidate {
  readonly label: string;
  /**
   * What the name refers to. Drives the icon CodeMirror shows, so it is worth
   * distinguishing: a user scanning for a method should not have to read past
   * the modules and constants sharing the prefix.
   */
  readonly kind: "function" | "class" | "module" | "keyword" | "variable" | "property";
  /** Signature or short type, shown to the right of the label. */
  readonly detail: string | null;
}

/** Hide dunder/private names unless the user has typed a leading underscore. */
export function candidateIsVisible(name: string, partial: string): boolean {
  if (!name.startsWith("_")) return true;
  return partial.startsWith("_");
}
