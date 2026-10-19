import {
  EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA,
  type ExternalDerivedProvider,
  type ExternalDerivedProviderAdapter,
  type ExternalProviderDiagnostic,
  type ExternalProviderExecutionRequest,
  type ExternalProviderRuntimeStatus,
  type ExternalProviderWireResult,
  type PythonDerivedProvider,
} from "./externalDerivedProviders";
import {
  getDefaultPythonRuntimeClient,
  type PythonRuntimeClient,
} from "@/lib/platform/pyodideRuntime";
import { pythonProviderResult } from "./pythonProviderResult";
import { pythonProviderContext } from "./pythonProviderContext";

export const PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION = "0.26.4" as const;

const PROVIDER_STOP_GRACE_MS = 1_500;

function cancelled(): Error {
  const error = new Error("Python provider execution was cancelled");
  error.name = "AbortError";
  return error;
}

function throwIfCancelled(signal: AbortSignal): void {
  if (signal.aborted) throw cancelled();
}

function runtimeStatus(provider: ExternalDerivedProvider): ExternalProviderRuntimeStatus {
  if (provider.kind !== "python") return { available: false, label: "Pyodide", reason: "This adapter accepts Python providers only." };
  if (provider.entrypoint.runtimeVersion !== PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION) {
    return {
      available: false,
      label: `Pyodide ${PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION}`,
      reason: `Provider pins Pyodide ${provider.entrypoint.runtimeVersion}; Studio ships ${PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION}.`,
    };
  }
  if (typeof window === "undefined" || typeof document === "undefined") {
    return { available: false, label: `Pyodide ${PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION}`, reason: "Pyodide requires the Studio webview." };
  }
  return { available: true, label: `Pyodide ${PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION}`, reason: null };
}

async function sha256(value: string): Promise<string> {
  if (typeof crypto === "undefined" || !crypto.subtle) throw new Error("WebCrypto is unavailable; provider source integrity cannot be verified");
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return `sha256:${Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function diagnostic(severity: ExternalProviderDiagnostic["severity"], code: string, message: string): ExternalProviderDiagnostic {
  return { severity, code, message: message.slice(0, 1_000) };
}

async function executePythonProvider(
  request: ExternalProviderExecutionRequest,
  signal: AbortSignal,
  providerRuntime: PythonRuntimeClient,
): Promise<ExternalProviderWireResult> {
  const provider = request.provider as PythonDerivedProvider;
  if (providerRuntime.isBusy()) throw new Error("The shared Pyodide provider runtime is busy; wait for the active provider to finish or cancel it");
  throwIfCancelled(signal);
  const actualDigest = await sha256(provider.entrypoint.source);
  throwIfCancelled(signal);
  if (actualDigest !== provider.entrypoint.sourceDigest) throw new Error("Python provider source does not match its pinned sha256 digest");

  if (!providerRuntime.tryAcquireExecution()) throw new Error("The shared Pyodide provider runtime is busy; wait for the active provider to finish or cancel it");
  let executionActive = true;
  const startedAt = performance.now();
  let forceStop: ReturnType<typeof setTimeout> | null = null;
  const stop = () => {
    if (!providerRuntime.interruptPython()) {
      providerRuntime.restartPython();
      return;
    }
    forceStop = setTimeout(() => {
      forceStop = null;
      if (executionActive) providerRuntime.restartPython();
    }, PROVIDER_STOP_GRACE_MS);
  };
  signal.addEventListener("abort", stop, { once: true });
  try {
    const handle = await providerRuntime.getPyodide();
    throwIfCancelled(signal);
    await providerRuntime.bindContext(handle, pythonProviderContext(request));
    throwIfCancelled(signal);
    // Reset the conventional result name so a prior provider cannot leak a
    // stale ndarray into this invocation when the current source returns none.
    const outcome = await providerRuntime.runPythonCell(
      handle,
      `result = None\n${provider.entrypoint.source}`,
      { resultOnly: true },
    );
    throwIfCancelled(signal);
    const array = pythonProviderResult(outcome);
    const diagnostics: ExternalProviderDiagnostic[] = [];
    if (outcome.stdout.trim()) diagnostics.push(diagnostic("info", "python_stdout", outcome.stdout.trim()));
    if (outcome.stderr.trim()) diagnostics.push(diagnostic("warning", "python_stderr", outcome.stderr.trim()));
    return {
      schema: EXTERNAL_DERIVED_PROVIDER_RESULT_SCHEMA,
      schemaVersion: 1,
      requestId: request.requestId,
      providerId: provider.id,
      providerVersion: provider.providerVersion,
      fieldId: provider.output.fieldId,
      unit: provider.output.unit,
      shape: array.shape,
      encoding: "float32-le",
      values: array.values,
      diagnostics,
      executionMs: Math.max(0, performance.now() - startedAt),
    };
  } catch (error) {
    if (signal.aborted) throw cancelled();
    throw error;
  } finally {
    if (forceStop !== null) clearTimeout(forceStop);
    signal.removeEventListener("abort", stop);
    executionActive = false;
    providerRuntime.releaseExecution();
  }
}

/** Executable adapter for the already-shipped, pinned Pyodide runtime. It does
 * not grant permissions: the execution gate in externalDerivedProviders must
 * receive the user's grants before this adapter is called. */
export function createPyodideDerivedProviderAdapter(providerRuntime: PythonRuntimeClient = getDefaultPythonRuntimeClient()): ExternalDerivedProviderAdapter {
  return {
    adapterId: `nimbus.pyodide.${PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION}`,
    kind: "python",
    supports: runtimeStatus,
    execute: (request, signal) => executePythonProvider(request, signal, providerRuntime),
  };
}
