/*
 * Cell splitting for the Python tab.
 *
 * Two properties carry the feature and both are easy to break silently:
 *
 *   1. No text is lost or duplicated. The cells partition the document, so
 *      running every cell must be the same as running the file. A splitter
 *      that drops the marker line, or the newline before it, changes what the
 *      user's code does without changing what they see.
 *
 *   2. A fragment reports the file's line numbers. Running one cell means
 *      handing Python a fragment, and an unpadded fragment says "line 3" for
 *      something on line 187 -- which sends the user hunting through the wrong
 *      part of their script. The padding is invisible when it works and
 *      invisible when it stops working, so it is asserted directly.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/studio/test_python_cells.ts
 */

import {
  cellIndexAtOffset,
  cellLabel,
  cellSourceForExecution,
  hasCellMarkers,
  isCellEmpty,
  isCellMarkerLine,
  splitPythonCells,
} from "../../../../ui/src/lib/studio/pythonCells.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

// ---------- 1. A script with no markers is one cell ----------

const plain = "import numpy as np\nresult = np.zeros(3)\n";
const plainCells = splitPythonCells(plain);
check(plainCells.length === 1, "an unmarked script is a single cell");
check(
  plainCells[0].code === plain,
  "the single cell's text is the whole document, byte for byte",
);
check(plainCells[0].startLine === 0, "the single cell starts at line 0");
check(plainCells[0].title === null, "the single cell has no title");
check(
  cellSourceForExecution(plainCells[0]) === plain,
  "running the single cell runs exactly the document, with no padding",
);
check(!hasCellMarkers(plain), "an unmarked script reports no markers");

// The empty document still yields a cell -- callers index cells[0]
// unconditionally, so returning nothing here would be a crash.
const emptyCells = splitPythonCells("");
check(emptyCells.length === 1, "an empty document still yields one cell");
check(emptyCells[0].code === "", "the empty document's cell is empty");

// ---------- 2. Markers split, and titles come through ----------

const marked = [
  "import numpy as np",
  "",
  "# %% Load",
  "w = load()",
  "# %% Plot",
  "display(w)",
].join("\n");
const markedCells = splitPythonCells(marked);
check(markedCells.length === 3, "two markers plus a preamble make three cells");
check(markedCells[0].title === null, "the preamble cell is untitled");
check(markedCells[1].title === "Load", "a marker's trailing text becomes the title");
check(markedCells[2].title === "Plot", "each marker gets its own title");
check(hasCellMarkers(marked), "a marked script reports markers");

check(
  markedCells[0].code === "import numpy as np\n",
  "the preamble cell holds everything above the first marker",
);
check(
  markedCells[1].code === "# %% Load\nw = load()",
  "a cell keeps its own marker line and stops before the next",
);
check(
  markedCells[2].code === "# %% Plot\ndisplay(w)",
  "the last cell runs to the end of the document",
);

// ---------- 3. The cells partition the document ----------

/*
 * The join is with "\n" because each cell excludes the newline that separates
 * it from the next. If that ever changes, this is the assertion that notices:
 * a splitter that swallowed the separator would produce cells that look right
 * individually and silently glue two statements together when rejoined.
 */
const partitionCases = [
  plain,
  marked,
  "",
  "# %%\n",
  "# %%",
  "a\n# %%\nb",
  "\n\n# %% one\n\n# %% two\n\n",
  "x = 1\n# %%\n# %%\ny = 2\n",
];
for (const source of partitionCases) {
  const rejoined = splitPythonCells(source)
    .map((cell) => cell.code)
    .join("\n");
  check(
    rejoined === source,
    `cells partition the document without loss: ${JSON.stringify(source)}`,
  );
}

// ---------- 4. A marker on the first line makes no phantom preamble ----------

const leading = "# %% First\nvalue = 1\n# %% Second\nvalue = 2\n";
const leadingCells = splitPythonCells(leading);
check(leadingCells.length === 2, "a first-line marker does not add an empty preamble cell");
check(leadingCells[0].title === "First", "the first-line marker owns cell 0");
check(leadingCells[0].startLine === 0, "cell 0 starts at line 0");

// ---------- 5. Line numbers survive the fragment ----------

/*
 * The property, stated exactly: every line of a cell sits at the same index in
 * the padded fragment as it does in the file. Asserting the padded fragment's
 * line COUNT alone would pass for a fragment padded at the wrong end, so each
 * line is compared by position.
 */
const lineFidelityCases = [plain, marked, leading, "a\n\n\n# %% late\nz = 1\n"];
for (const source of lineFidelityCases) {
  const fileLines = source.split("\n");
  for (const cell of splitPythonCells(source)) {
    const fragmentLines = cellSourceForExecution(cell).split("\n");
    let aligned = true;
    for (let line = cell.startLine; line < cell.endLine; line += 1) {
      if (fragmentLines[line] !== fileLines[line]) aligned = false;
    }
    check(
      aligned,
      `cell ${cell.index} reports file line numbers in ${JSON.stringify(source)}`,
    );
  }
}

// The padding is newlines and nothing else -- anything with content would
// change what the fragment does, not just where its lines land.
const lateCell = splitPythonCells("a\n\n\n# %% late\nz = 1\n")[1];
check(
  cellSourceForExecution(lateCell).slice(0, lateCell.startLine) === "\n\n\n",
  "the padding is bare newlines",
);
check(
  cellSourceForExecution(lateCell).slice(lateCell.startLine) === lateCell.code,
  "padding is prepended and the cell's own text is untouched",
);

// ---------- 6. What counts as a marker ----------

check(isCellMarkerLine("# %%"), "the bare marker is a marker");
check(isCellMarkerLine("#%%"), "Spyder's unspaced marker is a marker");
check(isCellMarkerLine("#   %% spaced"), "extra space before %% still marks");
check(isCellMarkerLine("# %% titled"), "a titled marker is a marker");
check(!isCellMarkerLine("# % not a cell"), "a single percent is not a marker");
check(!isCellMarkerLine("x = 1  # %% trailing"), "a trailing comment is not a marker");

/*
 * An indented `# %%` is a comment inside a suite. Splitting there would hand
 * Python a fragment beginning mid-block, which is a syntax error rather than a
 * wrong answer -- so this is the case that would be reported as "cells are
 * broken" rather than quietly tolerated.
 */
const indented = "def f():\n    # %% not a boundary\n    return 1\n";
check(!isCellMarkerLine("    # %% not a boundary"), "an indented marker is not a marker");
check(
  splitPythonCells(indented).length === 1,
  "an indented marker does not split a function body",
);
check(!hasCellMarkers(indented), "an indented marker does not count as a marked script");

// Jupytext writes a cell-type tag before the title; the tag is not a title.
const tagged = splitPythonCells("# %% [markdown] Notes\ntext = 1\n");
check(tagged[0].title === "Notes", "a Jupytext type tag is stripped from the title");
check(
  splitPythonCells("# %% [markdown]\ntext = 1\n")[0].title === null,
  "a tag with no title leaves the cell untitled",
);

// Windows line endings arrive from files saved elsewhere; the marker still
// marks, and the text is preserved exactly as written.
const crlf = "a = 1\r\n# %% Second\r\nb = 2\r\n";
const crlfCells = splitPythonCells(crlf);
check(crlfCells.length === 2, "CRLF documents split at markers");
check(crlfCells[1].title === "Second", "a CRLF marker's title drops the carriage return");
check(
  crlfCells.map((cell) => cell.code).join("\n") === crlf,
  "CRLF text survives the split byte for byte",
);

// ---------- 7. Locating the caret ----------

const located = splitPythonCells(marked);
check(cellIndexAtOffset(located, 0) === 0, "offset 0 is in the first cell");
check(
  cellIndexAtOffset(located, located[1].startOffset) === 1,
  "the caret on a cell's first character is in that cell, not the one above",
);
check(
  cellIndexAtOffset(located, located[1].startOffset + 3) === 1,
  "the caret inside a cell is in that cell",
);
check(
  cellIndexAtOffset(located, located[1].endOffset) === 1,
  "the caret at a cell's last character is still in that cell",
);
check(
  cellIndexAtOffset(located, located[2].startOffset) === 2,
  "the caret at the start of the last cell is in the last cell",
);
check(
  cellIndexAtOffset(located, 10_000) === located.length - 1,
  "an offset past the end clamps to the last cell",
);
check(cellIndexAtOffset(located, -5) === 0, "a negative offset clamps to the first cell");
check(cellIndexAtOffset([], 4) === 0, "an empty cell list yields index 0 rather than -1");

// ---------- 8. Cells with nothing to run ----------

/*
 * Running an empty cell is not harmless: it clears the output pane and the
 * produced-files list, so a user who pressed run on the blank cell they were
 * about to write in would lose the result they were looking at.
 */
const emptiness = splitPythonCells("# %% nothing\n\n# comment only\n\n# %% real\nx = 1\n");
check(isCellEmpty(emptiness[0]), "a cell of blanks and comments is empty");
check(!isCellEmpty(emptiness[1]), "a cell containing a statement is not empty");
check(
  isCellEmpty(splitPythonCells("")[0]),
  "the empty document's cell is empty",
);
check(
  !isCellEmpty(splitPythonCells("   \nx = 1\n")[0]),
  "leading whitespace does not make a cell empty",
);
check(
  isCellEmpty(splitPythonCells("   \n\t\n")[0]),
  "whitespace-only lines are empty regardless of what the whitespace is",
);

// ---------- 9. Labels ----------

check(cellLabel(plainCells[0], 1) === "Script", "an only cell is labelled Script");
check(cellLabel(markedCells[1], 3) === "Load", "a titled cell is labelled by its title");
check(
  cellLabel(markedCells[0], 3) === "Cell 1",
  "an untitled cell falls back to its 1-based position",
);

console.log(`\npython cell tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
