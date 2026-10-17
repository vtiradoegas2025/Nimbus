/*
 * Import-name lookups against the pinned Pyodide index.
 * Pure table reads — no interpreter, no DOM, no worker — so both sides of the
 * worker boundary and plain-node test suites can import without booting anything.
 */

import {
  PYODIDE_IMPORT_TO_PACKAGE,
  PYODIDE_PACKAGE_NAMES,
} from "./pyodidePackages";

/**
 * Resolve an import name to the loadPackage name. Generated from the lockfile —
 * 61 import names differ from their package name. Missing entries fall through unchanged
 * (correct input for micropip fallback).
 */
export function packageNameForImport(importName: string): string {
  return PYODIDE_IMPORT_TO_PACKAGE[importName] ?? importName;
}

/** Whether the pinned Pyodide index carries this import at all. */
export function isBundledPyodideImport(importName: string): boolean {
  return PYODIDE_PACKAGE_NAMES.has(packageNameForImport(importName));
}

/**
 * Extract the missing module name from a ModuleNotFoundError in stderr.
 * Returns null if not present; strips sub-module paths (loadPackage operates on top-level).
 */
export function detectMissingModule(stderr: string): string | null {
  const match = stderr.match(/ModuleNotFoundError: No module named '([^']+)'/);
  if (!match) return null;
  const full = match[1];
  // Submodule: scipy.signal → scipy.
  const dot = full.indexOf(".");
  return dot >= 0 ? full.slice(0, dot) : full;
}
