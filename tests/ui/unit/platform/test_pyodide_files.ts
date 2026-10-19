/*
 * Staging external data files into the Python interpreter.
 *
 * The budget is the whole point of this module: staging copies a file into
 * Pyodide's wasm32 heap, and overrunning that heap aborts the interpreter
 * rather than raising something catchable. So the checks below care most about
 * the boundaries, and about the check running BEFORE the allocation.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/platform/test_pyodide_files.ts
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  assertStagingBudget,
  basename,
  formatBytes,
  MAX_STAGED_FILE_BYTES,
  MAX_STAGED_TOTAL_BYTES,
  PYODIDE_STAGE_DIR,
  restageFiles,
  stageFile,
  StagingBudgetError,
  uniqueStagedName,
  type StagedFile,
} from "../../../../ui/src/lib/platform/pyodideFiles.ts";
/*
 * The virtual-filesystem half runs inside the interpreter worker, so it is
 * imported from the engine rather than from the staging module. Splitting the
 * two is the point of the worker migration; testing them from one file keeps
 * the seam visible -- budget and naming above, FS effects below.
 */
import {
  bindStagedFiles,
  removeStagedFile,
  writeStagedFile,
} from "../../../../ui/src/lib/platform/pyodideEngine.ts";
import {
  base64ToBytes,
  encodeCsv,
  encodeNpy,
  MAX_CSV_CELLS,
  pngDataUri,
} from "../../../../ui/src/lib/platform/pythonExport.ts";
import { parseNpy } from "../../../../ui/src/lib/platform/npy.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

function throws(run: () => void): Error | null {
  try {
    run();
    return null;
  } catch (error) {
    return error instanceof Error ? error : new Error(String(error));
  }
}

// ---------- 1. Path handling ----------

check(basename("/runs/beast/cm1out.nc") === "cm1out.nc", "POSIX path reduces to its file name");
check(basename("C:\\data\\cm1out.nc") === "cm1out.nc", "Windows path reduces to its file name");
check(basename("cm1out.nc") === "cm1out.nc", "a bare file name is unchanged");
// Not a crash and not an empty name: an empty staged name would produce the
// directory itself as a virtual path and overwrite the staging root.
check(basename("/") === "/", "a path with no final segment falls back to the input");

// ---------- 2. Collisions never silently overwrite ----------

/*
 * Comparing two runs' cm1out.nc is an ordinary thing to do. If the second
 * clobbered the first, user code would read the wrong dataset under the right
 * name, with nothing to indicate the substitution.
 */
check(uniqueStagedName("a.nc", new Set()) === "a.nc", "a free name is used as-is");
check(uniqueStagedName("a.nc", new Set(["a.nc"])) === "a_2.nc", "a collision is suffixed");
check(
  uniqueStagedName("a.nc", new Set(["a.nc", "a_2.nc"])) === "a_3.nc",
  "a chain of collisions keeps counting",
);
check(
  uniqueStagedName("archive.tar.gz", new Set(["archive.tar.gz"])) === "archive.tar_2.gz",
  "only the final extension is preserved when suffixing",
);
check(
  uniqueStagedName("README", new Set(["README"])) === "README_2",
  "an extensionless name is suffixed without inventing a dot",
);
// A leading-dot file is all extension by lastIndexOf, which would produce an
// empty stem; the suffix must still yield a usable, distinct name.
check(
  uniqueStagedName(".env", new Set([".env"])) !== ".env",
  "a dotfile collision still produces a distinct name",
);

// ---------- 3. The budget, at its boundaries ----------

check(throws(() => assertStagingBudget("/a.nc", 1, 0)) === null, "a small file is accepted");
check(
  throws(() => assertStagingBudget("/a.nc", MAX_STAGED_FILE_BYTES, 0)) === null,
  "a file exactly at the per-file limit is accepted",
);
check(
  throws(() => assertStagingBudget("/a.nc", MAX_STAGED_FILE_BYTES + 1, 0)) !== null,
  "one byte over the per-file limit is refused",
);
check(
  throws(() => assertStagingBudget("/a.nc", 1, MAX_STAGED_TOTAL_BYTES)) !== null,
  "a file that would exceed the total is refused even when individually small",
);
check(
  throws(() => assertStagingBudget("/a.nc", 0, MAX_STAGED_TOTAL_BYTES)) === null,
  "a zero-byte file does not trip the total",
);
check(
  MAX_STAGED_FILE_BYTES <= MAX_STAGED_TOTAL_BYTES,
  "the per-file limit cannot exceed the total limit",
);

const overFile = throws(() => assertStagingBudget("/data/big.nc", MAX_STAGED_FILE_BYTES * 2, 0));
check(overFile instanceof StagingBudgetError, "an over-budget file raises StagingBudgetError");
check(
  (overFile as StagingBudgetError)?.hostPath === "/data/big.nc",
  "the error carries the offending path",
);
check(
  (overFile as StagingBudgetError)?.byteLength === MAX_STAGED_FILE_BYTES * 2,
  "the error carries the offending size",
);
// The message has to name the file and both numbers, because the user's only
// remedy is to subset the input and they need to know by how much.
check(overFile?.message.includes("big.nc") === true, "the refusal names the file");
check(
  overFile?.message.includes(formatBytes(MAX_STAGED_FILE_BYTES)) === true,
  "the refusal states the limit in the same units as the panel",
);

const overTotal = throws(() =>
  assertStagingBudget("/data/second.nc", MAX_STAGED_TOTAL_BYTES / 2, MAX_STAGED_TOTAL_BYTES / 2 + 1)
);
check(overTotal instanceof StagingBudgetError, "exceeding the total raises StagingBudgetError");
check(
  overTotal?.message.includes("Release") === true,
  "the total-budget refusal says how to recover",
);

// ---------- 4. Byte formatting matches the panel's units ----------

check(formatBytes(0) === "0 B", "zero formats without a unit prefix");
check(formatBytes(512) === "512 B", "bytes are not scaled");
check(formatBytes(1024) === "1.0 KiB", "kibibytes use one decimal");
check(formatBytes(256 * 1024 * 1024) === "256 MiB", "a large value drops the decimal");
check(formatBytes(MAX_STAGED_TOTAL_BYTES) === "512 MiB", "the total budget reads as 512 MiB");

// ---------- 5. The Python-side view of staged files ----------

/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * A handle whose interpreter is a recorder.
 *
 * `mkdir` throws EEXIST after the first call, matching emscripten's FS: the
 * staging directory is created on every write and the second one must be a
 * no-op rather than an error surfaced to the user.
 */
function fakeHandle() {
  const globals = new Map<string, unknown>();
  const scripts: string[] = [];
  const unlinked: string[] = [];
  const written = new Map<string, Uint8Array>();
  const dirs: string[] = [];
  return {
    pyodide: {
      globals: { set: (key: string, value: unknown) => globals.set(key, value) },
      runPython: (source: string) => scripts.push(source),
      FS: {
        mkdir: (path: string) => {
          if (dirs.includes(path)) throw new Error("EEXIST");
          dirs.push(path);
        },
        writeFile: (path: string, bytes: Uint8Array) => written.set(path, bytes),
        unlink: (path: string) => unlinked.push(path),
      },
    },
    interruptBuffer: new Uint8Array(1),
    inspect: () => ({ globals, scripts, unlinked, written, dirs }),
  } as any;
}

const staged: StagedFile[] = [
  { hostPath: "/runs/a/cm1out.nc", virtualPath: `${PYODIDE_STAGE_DIR}/cm1out.nc`, name: "cm1out.nc", byteLength: 10 },
  { hostPath: "/runs/b/cm1out.nc", virtualPath: `${PYODIDE_STAGE_DIR}/cm1out_2.nc`, name: "cm1out_2.nc", byteLength: 20 },
];

const bindTarget = fakeHandle();
bindStagedFiles(bindTarget, staged);
const bound = bindTarget.inspect().globals.get("_staged_files") as Record<string, string>;
check(Object.keys(bound).length === 2, "every staged file is exposed");
check(bound["cm1out.nc"] === `${PYODIDE_STAGE_DIR}/cm1out.nc`, "the first file maps to its virtual path");
check(
  bound["cm1out_2.nc"] === `${PYODIDE_STAGE_DIR}/cm1out_2.nc`,
  "a de-duplicated file keeps its distinct virtual path",
);
check(
  bindTarget.inspect().scripts.some((s: string) => s.includes("files =")),
  "the dict is materialised as a Python name",
);
// Values cross from JS as a proxy, not a dict; without to_py the user's
// files["x"] would raise instead of indexing.
check(
  bindTarget.inspect().scripts.some((s: string) => s.includes("to_py")),
  "the proxy is converted to a native dict",
);

// Rebinding with fewer files must not leave the released one visible.
const rebindTarget = fakeHandle();
bindStagedFiles(rebindTarget, staged);
bindStagedFiles(rebindTarget, [staged[0]]);
const rebound = rebindTarget.inspect().globals.get("_staged_files") as Record<string, string>;
check(Object.keys(rebound).length === 1, "rebinding drops a released file from the namespace");
check(!("cm1out_2.nc" in rebound), "the released name is specifically gone");

const emptyTarget = fakeHandle();
bindStagedFiles(emptyTarget, []);
check(
  Object.keys(emptyTarget.inspect().globals.get("_staged_files") as object).length === 0,
  "an empty staging set still defines files rather than leaving it undefined",
);

// ---------- 6. Writing into the virtual filesystem ----------

const writeTarget = fakeHandle();
writeStagedFile(writeTarget, PYODIDE_STAGE_DIR, staged[0].virtualPath, new Uint8Array([1, 2, 3]));
check(
  writeTarget.inspect().dirs[0] === PYODIDE_STAGE_DIR,
  "the staging directory is created before the first write",
);
check(
  writeTarget.inspect().written.get(staged[0].virtualPath)?.length === 3,
  "the bytes land at the staged virtual path",
);

/*
 * mkdir throws EEXIST on every call after the first. Staging a second file must
 * succeed anyway -- the directory existing is the desired end state, and an
 * exception here would make every load after the first one fail.
 */
check(
  throws(() =>
    writeStagedFile(writeTarget, PYODIDE_STAGE_DIR, staged[1].virtualPath, new Uint8Array([4]))
  ) === null,
  "a second write tolerates the directory already existing",
);
check(
  writeTarget.inspect().written.size === 2,
  "the second file is written despite the EEXIST",
);

// ---------- 7. Releasing is forgiving ----------

const unlinkTarget = fakeHandle();
removeStagedFile(unlinkTarget, staged[0].virtualPath);
check(
  unlinkTarget.inspect().unlinked[0] === `${PYODIDE_STAGE_DIR}/cm1out.nc`,
  "releasing unlinks the virtual path",
);

/*
 * The user can delete the file from inside Python. Releasing it afterwards must
 * still clear it from the UI's staged set rather than throwing, or the entry
 * becomes permanently stuck.
 */
const hostileTarget = {
  pyodide: { FS: { unlink: () => { throw new Error("ENOENT"); } } },
} as any;
check(
  throws(() => removeStagedFile(hostileTarget, staged[0].virtualPath)) === null,
  "releasing a file already gone from the interpreter does not throw",
);

// ---------- 8. The capability that makes reading possible ----------

/*
 * Reading arbitrary bytes needs an explicit permission. Without it the picker
 * opens, the user chooses a file, and the read fails at the last step — so the
 * grant is pinned here next to the code that depends on it.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
const capabilities = readFileSync(
  resolve(HERE, "../../../../ui/src-tauri/capabilities/default.json"),
  "utf8",
);
check(capabilities.includes("fs:allow-read-file"), "read-file capability is granted");
check(capabilities.includes("fs:allow-stat"), "stat capability is granted for the pre-read size check");

// The size probe must precede the read in the staging path, not follow it.
const stagingSource = readFileSync(
  resolve(HERE, "../../../../ui/src/lib/platform/pyodideFiles.ts"),
  "utf8",
);
const budgetAt = stagingSource.indexOf("assertStagingBudget(hostPath, await (io.stat ?? fileByteLength)");
const readAt = stagingSource.indexOf("await (io.read ?? readBinaryFile)(hostPath)");
check(budgetAt > 0 && readAt > 0, "staging both checks the budget and reads the file");
check(budgetAt < readAt, "the budget is enforced before the file is read into memory");
// stat and read are separate syscalls, so the post-read size is re-checked.
check(
  stagingSource.indexOf("bytes.byteLength > MAX_STAGED_FILE_BYTES") > readAt,
  "the actual read size is re-checked against the limit",
);

// ---------- 9. Export decoding ----------

/*
 * Figures and animation frames cross from Python as base64 PNG. atob yields a
 * binary string whose code units are bytes; writing it out with the wrong
 * conversion (e.g. TextEncoder, which would UTF-8 encode anything above 0x7F)
 * corrupts every non-ASCII byte and produces an unopenable image.
 */
check(
  new TextDecoder().decode(base64ToBytes("SGVsbG8=")) === "Hello",
  "base64 decodes to the original bytes",
);
check(base64ToBytes("").length === 0, "empty base64 decodes to no bytes");

// A PNG signature exercises the high bytes that a naive text conversion breaks.
const pngSignature = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const roundTripped = base64ToBytes(Buffer.from(pngSignature).toString("base64"));
check(
  roundTripped.length === pngSignature.length
    && pngSignature.every((byte, index) => roundTripped[index] === byte),
  "bytes above 0x7F survive the round trip",
);
check(roundTripped instanceof Uint8Array, "decoding yields bytes rather than a string");

check(
  pngDataUri("AAAA") === "data:image/png;base64,AAAA",
  "frames are wrapped as PNG data URIs for the encoder",
);

// ---------- 10. Putting the staged files back after a restart ----------

/*
 * A restart empties the interpreter's filesystem along with everything else.
 * The staged set is the one part that can be rebuilt, because each entry
 * remembers where it came from — and rebuilding it is not optional: the panel
 * would otherwise keep listing files whose paths resolve to nothing, so the
 * failure surfaces later, inside the user's own code.
 *
 * These run against the real `restageFiles`, with the host read injected. The
 * writes it performs go to the interpreter, which does not exist here; what is
 * under test is which files survive and what is reported about the rest.
 */
const restageWrites: string[] = [];
let reboundNames: string[] = [];
const restageIo = {
  write: async (_dir: string, virtualPath: string) => { restageWrites.push(virtualPath); },
  bind: async (files: readonly StagedFile[]) => { reboundNames = files.map((f) => f.name); },
};
const restaged = await restageFiles(staged, {
  ...restageIo,
  read: async (hostPath: string) => {
    if (hostPath.endsWith("/b/cm1out.nc")) throw new Error("ENOENT: no such file");
    return new Uint8Array(64);
  },
});
check(restaged.kept.length === 1, "a file that still reads is staged again");
check(
  restaged.kept[0]?.virtualPath === staged[0].virtualPath,
  "it keeps the virtual path user code already refers to",
);
check(
  restaged.kept[0]?.byteLength === 64,
  "the size is taken from what was actually read, not from the stale record",
);
check(restaged.dropped.length === 1, "a file that has gone is dropped rather than kept broken");
check(
  restaged.dropped[0]?.file.name === "cm1out_2.nc",
  "the drop names the file the user will notice missing",
);
/*
 * The reason travels with it. "Could not reload two files" tells the user
 * nothing they can act on; ENOENT tells them the file moved.
 */
check(
  restaged.dropped[0]?.reason.includes("ENOENT"),
  "the drop carries why, not just that",
);
check(
  restageWrites.length === 1 && restageWrites[0] === staged[0].virtualPath,
  "only the surviving file's bytes are written back",
);
/*
 * Rebound to the survivors, not left alone. `files` still holding the dropped
 * name is the exact failure this function exists to prevent -- a path in the
 * user's namespace with nothing behind it.
 */
check(
  reboundNames.length === 1 && reboundNames[0] === "cm1out.nc",
  "Python's files dict is rebuilt from what survived",
);

/*
 * Both budgets are rechecked from the bytes actually read. Files can change
 * while the interpreter is down, and those new sizes were never admitted.
 */
const grown = await restageFiles([staged[0]], {
  ...restageIo,
  read: async () => new Uint8Array(MAX_STAGED_FILE_BYTES + 1),
});
check(grown.kept.length === 0, "a file that outgrew the per-file limit is not re-staged");
check(
  grown.dropped[0]?.reason.includes(formatBytes(MAX_STAGED_FILE_BYTES)),
  "and the refusal states the limit it broke",
);
check(reboundNames.length === 0, "a restage that saves nothing empties the files dict");

// A stat/read race must re-apply the aggregate limit to the bytes that will
// actually enter wasm, not only to stale metadata observed before the read.
let racedWriteCount = 0;
const fakeBytes = (byteLength: number) => ({ byteLength } as Uint8Array);
let raceRejected = false;
try {
  await stageFile("/data/growing.nc", new Set(), 300 * 1024 * 1024, {
    stat: async () => 200 * 1024 * 1024,
    read: async () => fakeBytes(220 * 1024 * 1024),
    write: async () => { racedWriteCount += 1; },
  });
} catch (error) {
  raceRejected = error instanceof StagingBudgetError;
}
check(raceRejected, "actual bytes are rechecked against the aggregate after a file grows");
check(racedWriteCount === 0, "a stat/read race cannot write an over-budget payload");

const aggregateFiles: StagedFile[] = ["a.nc", "b.nc", "c.nc"].map((name) => ({
  hostPath: `/data/${name}`,
  virtualPath: `${PYODIDE_STAGE_DIR}/${name}`,
  name,
  byteLength: 100,
}));
const aggregateWrites: string[] = [];
const aggregateRestage = await restageFiles(aggregateFiles, {
  read: async () => fakeBytes(200 * 1024 * 1024),
  write: async (_dir, path) => { aggregateWrites.push(path); },
  bind: async () => {},
});
check(
  aggregateRestage.kept.length === 2 && aggregateRestage.dropped.length === 1,
  "restart rechecks the actual aggregate across all surviving staged files",
);
check(aggregateWrites.length === 2, "restart never writes the file that crosses the aggregate cap");

const nothing = await restageFiles([], { ...restageIo, read: async () => new Uint8Array(0) });
check(
  nothing.kept.length === 0 && nothing.dropped.length === 0,
  "restarting with nothing staged is not an error",
);

// ---------- 11. Getting a computed array back out ----------

/*
 * `display(arr)` drew a heatmap and stopped there: the one output a script
 * exists to produce was the one that could not leave. The writer below is the
 * counterpart to the reader every field ENTERS through, so the reader is what
 * checks it — a header this format-sensitive is easy to write in a way only
 * numpy notices, and "our writer agrees with our writer" would notice nothing.
 */
function roundTrip(values: number[], shape: number[]) {
  return parseNpy(encodeNpy(new Float32Array(values), shape));
}

const npy1d = roundTrip([1, 2.5, -3], [3]);
check(npy1d.shape.length === 1 && npy1d.shape[0] === 3, "a 1-D array keeps its shape");
check(
  npy1d.data[0] === 1 && npy1d.data[1] === 2.5 && npy1d.data[2] === -3,
  "a 1-D array keeps its values",
);

const npy2d = roundTrip([1, 2, 3, 4, 5, 6], [2, 3]);
check(
  npy2d.shape.length === 2 && npy2d.shape[0] === 2 && npy2d.shape[1] === 3,
  "a 2-D array keeps both dimensions",
);
// C order, which the header claims: element (1, 0) is index 3, not index 1.
check(npy2d.data[3] === 4, "the elements stay in C order");

const npy3d = roundTrip([...Array(24).keys()], [2, 3, 4]);
check(npy3d.shape.join("x") === "2x3x4", "a 3-D array keeps all three dimensions");
check(npy3d.data[23] === 23, "the last element survives");

/*
 * The padding rule. numpy reads a mis-padded file anyway, so nothing here would
 * fail on it -- but a memory-mapped reader will, and that is exactly the reader
 * someone opens a large exported field with.
 */
for (const shape of [[1], [3], [2, 3], [1024], [2, 3, 4]]) {
  const bytes = encodeNpy(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  const headerLen = bytes[8] | (bytes[9] << 8);
  check(
    (10 + headerLen) % 64 === 0,
    `the header ends on a 64-byte boundary (shape ${shape.join("x")})`,
  );
  check(bytes[9 + headerLen] === 0x0a, `the header ends with a newline (shape ${shape.join("x")})`);
}

/*
 * A 1-D shape must be written `(3,)`. Without the trailing comma Python parses
 * the header's tuple as a plain integer, and numpy rejects the file -- a case
 * no other rank exercises.
 */
const oneDimHeader = new TextDecoder().decode(encodeNpy(new Float32Array(3), [3]).slice(10, 80));
check(oneDimHeader.includes("(3,)"), "a 1-D shape is written as a Python tuple");
check(
  new TextDecoder().decode(encodeNpy(new Float32Array(6), [2, 3]).slice(10, 80)).includes("(2, 3)"),
  "a 2-D shape is written without a stray comma",
);

// ---------- 12. CSV ----------

check(encodeCsv(new Float32Array([1, 2, 3]), [3]) === "1\n2\n3\n", "a 1-D array is one value a line");
check(
  encodeCsv(new Float32Array([1, 2, 3, 4]), [2, 2]) === "1,2\n3,4\n",
  "a 2-D array is rows of the C-order elements",
);
check(
  throws(() => encodeCsv(new Float32Array(8), [2, 2, 2])) !== null,
  "rank 3 is refused rather than flattened into a shape nobody can recover",
);

/*
 * Nine significant digits is the round-trip width for binary32. Printing the
 * float32 nearest 0.1 as a double gives 0.10000000149011612 -- seventeen digits
 * of noise the array never had.
 */
const tenth = encodeCsv(new Float32Array([0.1]), [1]).trim();
check(tenth === "0.100000001", `a float32 prints at its own precision (got ${tenth})`);
check(new Float32Array([Number(tenth)])[0] === new Float32Array([0.1])[0], "and reads back identical");
check(encodeCsv(new Float32Array([2]), [1]).trim() === "2", "a whole number stays whole");

/*
 * numpy's spellings, not JavaScript's. `NaN` and `Infinity` are not values
 * loadtxt or read_csv accept, so a field with a missing region would produce a
 * file that neither of them can open.
 */
const nonFinite = encodeCsv(new Float32Array([NaN, Infinity, -Infinity]), [3]).trim().split("\n");
check(nonFinite[0] === "nan", "NaN is written the way numpy reads it");
check(nonFinite[1] === "inf" && nonFinite[2] === "-inf", "infinities are written the way numpy reads them");

check(MAX_CSV_CELLS > 0, "the CSV cap is a real number the UI can quote");

console.log(`\npyodide file staging tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
