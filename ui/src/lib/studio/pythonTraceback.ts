/*
 * Map traceback line numbers to editor positions.
 * Only File "<exec>" frames are user code; stdlib frames are left as plain text.
 * Works because cellSourceForExecution pads fragments so line numbers match the file.
 */

/** A run of traceback text, optionally pointing at a line of the user's file. */
export interface TracebackSegment {
  readonly text: string;
  /** 1-based line in the user's buffer, or null when this is plain text. */
  readonly line: number | null;
}

// matchAll avoids the stateful lastIndex bug that exec loops carry across calls.
const EXEC_FRAME = /File "<exec>", line (\d+)/g;

/** Split stderr into plain and linkable segments. Concatenating all segment.text reproduces the original exactly. */
export function splitTraceback(stderr: string): TracebackSegment[] {
  const segments: TracebackSegment[] = [];
  let cursor = 0;
  for (const match of stderr.matchAll(EXEC_FRAME)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      segments.push({ text: stderr.slice(cursor, index), line: null });
    }
    segments.push({ text: match[0], line: Number(match[1]) });
    cursor = index + match[0].length;
  }
  if (cursor < stderr.length) {
    segments.push({ text: stderr.slice(cursor), line: null });
  }
  return segments;
}

/** Character offset of a 1-based line. Clamped to file length; a traceback may outlive the edit that invalidated it. */
export function offsetForLine(source: string, oneBasedLine: number): number {
  if (oneBasedLine <= 1) return 0;
  let offset = 0;
  let line = 1;
  while (line < oneBasedLine) {
    const next = source.indexOf("\n", offset);
    if (next === -1) return source.length;
    offset = next + 1;
    line += 1;
  }
  return Math.min(offset, source.length);
}
