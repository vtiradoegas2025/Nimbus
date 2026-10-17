/*
 * Main-thread half of staging external files into the Pyodide virtual FS.
 * Budget-checks happen here (wasm32 heap abort rather than MemoryError on overflow).
 * Only the FS write crosses to the worker, as bytes not paths.
 */

import { fileByteLength, readBinaryFile } from "./tauri";
import {
  bindStagedFilesInPython,
  removeStagedFile,
  writeStagedFile,
} from "./pyodideRuntime";

export const PYODIDE_STAGE_DIR = "/nimbus";

// Budgets leave headroom for arrays the user's code allocates on the same wasm32 heap.
export const MAX_STAGED_FILE_BYTES = 256 * 1024 * 1024;
export const MAX_STAGED_TOTAL_BYTES = 512 * 1024 * 1024;

export interface StagedFile {
  /** Absolute path on the real filesystem. */
  readonly hostPath: string;
  /** Path inside the Python interpreter — what user code opens. */
  readonly virtualPath: string;
  /** Key in the `files` dict exposed to Python. */
  readonly name: string;
  readonly byteLength: number;
}

/** Human-readable byte count, matching the resource panel's units. */
export function formatBytes(value: number): string {
  if (value === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const unit = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  const scaled = value / 1024 ** unit;
  return `${scaled >= 100 || unit === 0 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[unit]}`;
}

/** Final path segment, tolerating both / and \\ separators. */
export function basename(path: string): string {
  const segments = path.split(/[/\\]/);
  return segments[segments.length - 1] || path;
}

/** Appends _2, _3, … until the name is not in `taken`. Prevents silent overwrites. */
export function uniqueStagedName(desired: string, taken: ReadonlySet<string>): string {
  if (!taken.has(desired)) return desired;
  const dot = desired.lastIndexOf(".");
  const stem = dot > 0 ? desired.slice(0, dot) : desired;
  const extension = dot > 0 ? desired.slice(dot) : "";
  for (let index = 2; ; index += 1) {
    const candidate = `${stem}_${index}${extension}`;
    if (!taken.has(candidate)) return candidate;
  }
}

export class StagingBudgetError extends Error {
  readonly hostPath: string;
  readonly byteLength: number;

  constructor(message: string, hostPath: string, byteLength: number) {
    super(message);
    this.name = "StagingBudgetError";
    this.hostPath = hostPath;
    this.byteLength = byteLength;
  }
}

/** Check per-file and total budgets BEFORE reading. Applied post-read, it defeats the guard. */
export function assertStagingBudget(
  hostPath: string,
  byteLength: number,
  alreadyStagedBytes: number,
): void {
  if (byteLength > MAX_STAGED_FILE_BYTES) {
    throw new StagingBudgetError(
      `${basename(hostPath)} is ${formatBytes(byteLength)}; the per-file limit is `
      + `${formatBytes(MAX_STAGED_FILE_BYTES)} because staging copies the whole file into the `
      + `interpreter's memory. Subset it first, or read it through the dataset picker instead.`,
      hostPath,
      byteLength,
    );
  }
  const total = alreadyStagedBytes + byteLength;
  if (total > MAX_STAGED_TOTAL_BYTES) {
    throw new StagingBudgetError(
      `Staging ${basename(hostPath)} (${formatBytes(byteLength)}) would bring the total to `
      + `${formatBytes(total)}, over the ${formatBytes(MAX_STAGED_TOTAL_BYTES)} limit. `
      + `Release a staged file first.`,
      hostPath,
      byteLength,
    );
  }
}

/** Copy one host file into the interpreter's virtual FS. Caller owns the staged set. */
export async function stageFile(
  hostPath: string,
  taken: ReadonlySet<string>,
  alreadyStagedBytes: number,
  io: {
    stat?: typeof fileByteLength;
    read?: typeof readBinaryFile;
    write?: typeof writeStagedFile;
  } = {},
): Promise<StagedFile> {
  assertStagingBudget(hostPath, await (io.stat ?? fileByteLength)(hostPath), alreadyStagedBytes);
  const name = uniqueStagedName(basename(hostPath), taken);
  const virtualPath = `${PYODIDE_STAGE_DIR}/${name}`;
  const bytes = await (io.read ?? readBinaryFile)(hostPath);
  // stat and read are separate syscalls; re-check in case the file grew between them.
  if (bytes.byteLength > MAX_STAGED_FILE_BYTES) {
    throw new StagingBudgetError(
      `${basename(hostPath)} changed size while loading and no longer fits the `
      + `${formatBytes(MAX_STAGED_FILE_BYTES)} per-file limit.`,
      hostPath,
      bytes.byteLength,
    );
  }
  // The file can grow between stat and read. Recheck the aggregate against the
  // actual payload too; the worker write transfers and detaches this buffer.
  assertStagingBudget(hostPath, bytes.byteLength, alreadyStagedBytes);
  const byteLength = bytes.byteLength;
  await (io.write ?? writeStagedFile)(PYODIDE_STAGE_DIR, virtualPath, bytes);
  return { hostPath, virtualPath, name, byteLength };
}

/** Remove a staged file from the interpreter, freeing its heap. */
export async function unstageFile(
  staged: StagedFile,
  io: { remove?: typeof removeStagedFile } = {},
): Promise<void> {
  await (io.remove ?? removeStagedFile)(staged.virtualPath);
}

/** What survived a re-stage, and what did not. */
export interface RestageResult {
  readonly kept: StagedFile[];
  /** Files whose host path could not be re-read, with the reason. */
  readonly dropped: ReadonlyArray<{ file: StagedFile; reason: string }>;
}

/**
 * Re-populate the virtual FS after the interpreter was replaced.
 * Budgets are not re-checked (already admitted); growth past the per-file limit
 * is caught by the read below.
 */
export async function restageFiles(
  staged: readonly StagedFile[],
  /*
   * The three effects, injectable together. A node suite can exercise which
   * files survive and what is said about the rest without an interpreter to
   * write them into; that the bytes actually land is the engine's own test.
   */
  // Injectable for tests (read/write/bind can be stubbed without a live interpreter).
  io: {
    read?: (hostPath: string) => Promise<Uint8Array>;
    write?: typeof writeStagedFile;
    bind?: typeof bindStagedFiles;
  } = {},
): Promise<RestageResult> {
  const read = io.read ?? readBinaryFile;
  const write = io.write ?? writeStagedFile;
  const bind = io.bind ?? bindStagedFiles;
  const kept: StagedFile[] = [];
  const dropped: { file: StagedFile; reason: string }[] = [];
  let retainedBytes = 0;
  for (const file of staged) {
    try {
      const bytes = await read(file.hostPath);
      assertStagingBudget(file.hostPath, bytes.byteLength, retainedBytes);
      const byteLength = bytes.byteLength;
      await write(PYODIDE_STAGE_DIR, file.virtualPath, bytes);
      retainedBytes += byteLength;
      kept.push({ ...file, byteLength });
    } catch (e) {
      dropped.push({ file, reason: e instanceof Error ? e.message : String(e) });
    }
  }
  // Rebind even when nothing survived so `files` is truthfully empty.
  await bind(kept);
  return { kept, dropped };
}

/** Expose staged files to Python as `files` dict. Rebuilt on every bind so released files disappear. */
export async function bindStagedFiles(
  staged: readonly StagedFile[],
  io: { bind?: typeof bindStagedFilesInPython } = {},
): Promise<void> {
  await (io.bind ?? bindStagedFilesInPython)(
    staged.map((file) => ({ name: file.name, virtualPath: file.virtualPath })),
  );
}
