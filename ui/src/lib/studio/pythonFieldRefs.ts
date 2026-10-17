/*
 * Detect which fields a script code-references, and enforce a byte budget before injection.
 * Only executable code counts (comments and string literals are stripped).
 */

/**
 * Strip comments and string literals, preserving f-string expressions.
 * Field names in string literals are not injection requests; f-string braces hold real code.
 */
export function stripPythonNonCode(code: string): string {
  let out = "";
  let i = 0;
  while (i < code.length) {
    const char = code[i];

    if (char === "#") {
      while (i < code.length && code[i] !== "\n") i += 1;
      continue;
    }

    if (char === '"' || char === "'") {
      // A prefix immediately before the quote decides whether this is an
      // f-string. Read backwards over the letters that may precede a quote
      // (r, b, u, f in any case and either order) rather than only one.
      let start = out.length;
      while (start > 0 && /[A-Za-z]/.test(out[start - 1])) start -= 1;
      const prefix = out.slice(start);
      const interpolated = /f/i.test(prefix);
      if (/^[rbuf]*$/i.test(prefix) && prefix.length > 0) out = out.slice(0, start);

      const quote = code.startsWith(char.repeat(3), i) ? char.repeat(3) : char;
      i += quote.length;
      let body = "";
      while (i < code.length) {
        if (code[i] === "\\") {
          i += 2;
          continue;
        }
        if (code.startsWith(quote, i)) {
          i += quote.length;
          break;
        }
        // An unterminated single-quoted string ends at the newline, which is
        // the state the editor is in for most of the time anyone is typing.
        if (quote.length === 1 && code[i] === "\n") break;
        body += code[i];
        i += 1;
      }
      // A space, not nothing: two identifiers either side of a string must not
      // be fused into one token.
      out += interpolated ? ` ${expressionsInFString(body)} ` : " ";
      continue;
    }

    out += char;
    i += 1;
  }
  return out;
}

/** The `{...}` expressions of an f-string body, with the literal text dropped. */
function expressionsInFString(body: string): string {
  let out = "";
  let depth = 0;
  for (let i = 0; i < body.length; i += 1) {
    const char = body[i];
    if (char === "{") {
      // `{{` is an escaped brace, not an expression.
      if (body[i + 1] === "{") {
        i += 1;
        continue;
      }
      depth += 1;
      continue;
    }
    if (char === "}") {
      if (depth > 0) depth -= 1;
      else if (body[i + 1] === "}") i += 1;
      continue;
    }
    if (depth > 0) out += char;
  }
  return out;
}

/**
 * Field names the script code-references. Scanned over the whole document so a
 * single-cell run and a full-script run share the same environment.
 */
export function detectReferencedFields(code: string, availableFields: string[]): string[] {
  const available = new Set(availableFields);
  const seen = new Set<string>();
  // Drop attribute accesses (.name) so np.pi doesn't load a field named "pi".
  // Lookbehind is absent on some engines; substitution is used instead.
  const scanned = stripPythonBindingTargets(stripPythonNonCode(code))
    .replace(/\.\s*[A-Za-z_][A-Za-z_0-9]*/g, " ");
  for (const token of scanned.match(/\b[A-Za-z_][A-Za-z_0-9]*\b/g) ?? []) {
    if (available.has(token)) seen.add(token);
  }
  return Array.from(seen).sort();
}

/** Remove the common binding positions that do not read their name. This is
 * intentionally conservative rather than a pretend Python parser: direct and
 * tuple assignment plus for-loop targets are unambiguous after strings and
 * comments have already been stripped. Augmented assignment remains a read. */
function stripPythonBindingTargets(code: string): string {
  const identifier = "[A-Za-z_][A-Za-z_0-9]*";
  const target = `(?:\\(?\\s*${identifier}(?:\\s*,\\s*${identifier})+\\s*\\)?|${identifier})`;
  const blankIdentifiers = (value: string): string => value.replace(
    /[A-Za-z_][A-Za-z_0-9]*/g,
    (name) => " ".repeat(name.length),
  );
  let stripped = code.replace(
    new RegExp(`(^|[;\\n])(\\s*)(${target})(\\s*(?::[^=;\\n]+)?\\s*)=(?!=)`, "gm"),
    (_match, boundary: string, whitespace: string, names: string, suffix: string) =>
      `${boundary}${whitespace}${blankIdentifiers(names)}${suffix}=`,
  );
  stripped = stripped.replace(
    new RegExp(`\\bfor(\\s+)(${target})(\\s+)in\\b`, "g"),
    (_match, before: string, names: string, after: string) =>
      `for${before}${blankIdentifiers(names)}${after}in`,
  );
  return stripped;
}

/**
 * Max bytes auto-injected per Run. 256 MiB fits one 160 MiB field and leaves working room.
 * Empirically derived from WKWebView limits, not the theoretical 4 GiB wasm32 ceiling.
 */
export const MAX_AUTO_INJECT_BYTES = 256 * 1024 * 1024;

export interface SkippedField {
  readonly name: string;
  /**
   * What it would have cost, or null when the budget was already spent and it
   * was never read — the size is only knowable from the data, and reading a
   * field purely to report how big the field we are not using is would spend
   * on the main thread exactly what this is trying to save.
   */
  readonly bytes: number | null;
}

/** Shared empty result. A fresh [] per call is a new identity that triggers spurious re-renders. */
export const NO_SKIPPED_FIELDS: readonly SkippedField[] = Object.freeze([]);

/** Whether one more field of `bytes` still fits alongside what is already bound. */
export function fitsInBudget(usedBytes: number, bytes: number): boolean {
  return usedBytes + bytes <= MAX_AUTO_INJECT_BYTES;
}

/** True when the budget is spent and reading any further field is pure allocation cost. */
export function budgetExhausted(usedBytes: number): boolean {
  return usedBytes >= MAX_AUTO_INJECT_BYTES;
}

function mib(bytes: number): string {
  return `${Math.round(bytes / (1024 * 1024))} MiB`;
}

/** Human-readable notice listing skipped fields and their sizes, with a load_field remedy. */
export function skippedFieldsNotice(skipped: readonly SkippedField[]): string {
  if (skipped.length === 0) return "";
  const list = skipped
    .map((field) => `  ${field.name}${field.bytes === null ? "" : ` (${mib(field.bytes)})`}`)
    .join("\n");
  // stride/box reduce before the data crosses IPC; Python-side slicing materialises the whole field first.
  return `Nimbus: ${skipped.length} field${skipped.length === 1 ? "" : "s"} not bound whole; `
    + `that would have exceeded the ${mib(MAX_AUTO_INJECT_BYTES)} an auto-load may take:\n`
    + `${list}\n`
    + `Ask for as much of one as you like — the reduction happens before it reaches\n`
    + `Python, so any domain works:\n`
    + `  ${skipped[0].name}_coarse = await load_field("${skipped[0].name}", stride=4)\n`
    + `  ${skipped[0].name}_box = await load_field("${skipped[0].name}", `
    + `box=((0, 128), (0, 128), (0, 40)))\n`;
}
