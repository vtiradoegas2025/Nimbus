/*
 * # %% cell splitting for the Python scripting tab (VS Code / Spyder / Jupytext convention).
 * The file stays a plain .py; cells are a view over the text, not a storage format.
 * cellSourceForExecution pads with leading newlines so traceback line numbers match the file.
 */

/** A `# %%` delimited region of a script. */
export interface PythonCell {
  /** 0-based position in the document's cell list. */
  readonly index: number;
  /** Text following the `# %%` marker, or null for an unlabelled/implicit cell. */
  readonly title: string | null;
  /**
   * 0-based line where this cell begins. For a marked cell this is the marker
   * line itself -- the marker is a comment, so keeping it inside the cell body
   * is harmless and makes the line arithmetic exact.
   */
  readonly startLine: number;
  /** 0-based line one past the cell's last line. */
  readonly endLine: number;
  /** Character offset of the cell's first character. */
  readonly startOffset: number;
  /** Character offset one past the cell's last character. */
  readonly endOffset: number;
  /** The cell's text, marker line included, with no trailing newline added. */
  readonly code: string;
}

/* Column 0 required: an indented # %% is inside a suite and would split it into a syntax error. #%% (no space) accepted for Spyder. */
const CELL_MARKER = /^#\s*%%(.*)$/;

/** Whether a single line opens a cell. Exported so the editor highlight and the splitter share one pattern. */
export function isCellMarkerLine(line: string): boolean {
  return CELL_MARKER.test(stripCr(line));
}

/** Whether the source uses cell markers at all. */
export function hasCellMarkers(source: string): boolean {
  return source.split("\n").some((line) => isCellMarkerLine(line));
}

/** Split source into cells at column-0 # %% markers. Always returns at least one cell. */
export function splitPythonCells(source: string): PythonCell[] {
  const lines = source.split("\n");
  // Character offset of the start of each line, plus a terminator entry so
  // lineOffsets[i + 1] is always defined for the last line.
  const lineOffsets: number[] = [];
  let running = 0;
  for (const line of lines) {
    lineOffsets.push(running);
    running += line.length + 1; // +1 for the "\n" that split() consumed
  }
  lineOffsets.push(source.length + 1);

  const boundaries: { line: number; title: string | null }[] = [];
  lines.forEach((line, i) => {
    const match = CELL_MARKER.exec(stripCr(line));
    if (match) {
      // A leading "[markdown]"-style tag is Jupytext's cell-type annotation,
      // not a title; keep whatever the user wrote after it.
      const label = match[1].replace(/^\s*\[[^\]]*\]/, "").trim();
      boundaries.push({ line: i, title: label.length > 0 ? label : null });
    }
  });

  // Text above the first marker is a real cell -- imports and setup usually
  // live there -- so it gets an implicit, untitled entry.
  if (boundaries.length === 0 || boundaries[0].line > 0) {
    boundaries.unshift({ line: 0, title: null });
  }

  return boundaries.map((boundary, i) => {
    const nextLine = i + 1 < boundaries.length ? boundaries[i + 1].line : lines.length;
    const startOffset = lineOffsets[boundary.line];
    // One past the last character of the previous line, i.e. excluding the
    // newline that separates this cell from the next marker.
    const endOffset = Math.min(source.length, Math.max(startOffset, lineOffsets[nextLine] - 1));
    return {
      index: i,
      title: boundary.title,
      startLine: boundary.line,
      endLine: nextLine,
      startOffset,
      endOffset,
      code: source.slice(startOffset, endOffset),
    };
  });
}

/** Index of the cell containing `offset`, clamped into range. A caret on a boundary belongs to the cell that starts there. */
export function cellIndexAtOffset(cells: readonly PythonCell[], offset: number): number {
  if (cells.length === 0) return 0;
  for (let i = cells.length - 1; i >= 0; i -= 1) {
    if (offset >= cells[i].startOffset) return i;
  }
  return 0;
}

/**
 * The text to hand the interpreter for one cell, padded so Python's line
 * numbers match the file the user is editing.
 */
export function cellSourceForExecution(cell: PythonCell): string {
  return "\n".repeat(cell.startLine) + cell.code;
}

/** Human label for a cell: its title, or a positional fallback. */
export function cellLabel(cell: PythonCell, total: number): string {
  if (cell.title) return cell.title;
  return total > 1 ? `Cell ${cell.index + 1}` : "Script";
}

/** True when a cell contains only blank lines, the marker, or comments. Run All skips these. */
export function isCellEmpty(cell: PythonCell): boolean {
  return cell.code
    .split("\n")
    .every((line) => {
      const trimmed = line.trim();
      return trimmed.length === 0 || trimmed.startsWith("#");
    });
}

function stripCr(line: string): string {
  return line.endsWith("\r") ? line.slice(0, -1) : line;
}
