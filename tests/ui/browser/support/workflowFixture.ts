import { mockIPC } from "@tauri-apps/api/mocks";
import { MOCK_SCHEME_CATALOG } from "@nimbus/lib/config/schemeCatalog";
import type { CheckpointInspection } from "@nimbus/lib/platform/tauri";
import type { RemoteReceiveSummary } from "@nimbus/lib/platform/remoteCompute";

const calls: { command: string; args: Record<string, unknown> }[] = [];
const inspections: { path: string; resolve: () => void; reject: (message: string) => void }[] = [];
const aborts: { id: string; reject: (message: string) => void }[] = [];
let holdInspection = false;
let saveDialog: ((path: string | null) => void) | null = null;
let receive: ((result: RemoteReceiveSummary) => void) | null = null;

function checkpoint(path: string): CheckpointInspection {
  return { path, sidecarPath: `${path}.json`, schema: "nimbus.checkpoint", schemaVersion: 1,
    checkpointBytes: 1024, checkpointFnv1a64: "0123456789abcdef", runId: "fixture-parent",
    parentRunId: null, parentCheckpoint: null, completedSteps: 10, simulationTimeS: 10,
    boundary: "completed_step", portableAcrossComputeMemoryTopology: true, integrityVerified: true,
    formatCompatible: true, scientificCompatibility: "verified_before_state_restore" };
}

declare global {
  interface Window {
    __nimbusWorkflowFixture?: {
      snapshot: () => { calls: typeof calls; inspections: string[]; aborts: string[]; dialogPending: boolean; receivePending: boolean };
      holdCheckpoint: (hold: boolean) => void;
      settleCheckpoint: (message?: string) => void;
      rejectAbort: (message: string) => void;
      resolveSaveDialog: (path: string | null) => void;
      settleReceive: (state: "complete" | "failed", error?: string) => void;
    };
  }
}

export function installWorkflowFixture(): void {
  mockIPC((command, payload) => {
    const args = (payload ?? {}) as Record<string, unknown>;
    calls.push({ command, args });
    if (command === "project_root") return "/__workflow_fixture__/project";
    if (command === "user_config_dir") return "/__workflow_fixture__/configs";
    if (command === "list_configs" || command === "list_runs" || command === "discover_run_library") return [];
    if (command === "scheme_catalog") return MOCK_SCHEME_CATALOG;
    if (command === "inspect_checkpoint") {
      const path = String(args.path);
      if (!holdInspection) return checkpoint(path);
      return new Promise((resolve, reject) => inspections.push({ path,
        resolve: () => resolve(checkpoint(path)), reject: message => reject(new Error(message)) }));
    }
    if (command === "validate_config_contents") return { valid: false, stdout: "", stderr: "Current exact-binary fixture rejection", exitCode: 2 };
    if (command === "write_config") return null;
    if (command === "plugin:dialog|open") return "/__workflow_fixture__/source.nimbus";
    if (command === "plugin:dialog|save") return new Promise<string | null>(resolve => { saveDialog = resolve; });
    if (command === "abort_sim") return new Promise((_resolve, reject) => aborts.push({ id: String(args.runId), reject: message => reject(new Error(message)) }));
    if (command === "copy_run_provenance") return String(args.destination);
    if (command === "reveal_path") throw new Error(`Cannot reveal ${String(args.path)}`);
    if (command === "probe_remote_compute") throw new Error("Fixture remote host is unavailable");
    if (command === "receive_remote_results") return new Promise<RemoteReceiveSummary>(resolve => { receive = resolve; });
    if (command === "cancel_remote_receive") {
      const resolve = receive; receive = null;
      resolve?.({ state: "cancelled", error: "Receiving cancelled; remote originals retained", retryAvailable: true, updatedUnixMs: Date.now() });
      return null;
    }
    throw new Error(`Unhandled workflow fixture command: ${command}`);
  }, { shouldMockEvents: true });
  window.__nimbusWorkflowFixture = {
    snapshot: () => ({ calls: [...calls], inspections: inspections.map(value => value.path), aborts: aborts.map(value => value.id), dialogPending: saveDialog != null, receivePending: receive != null }),
    holdCheckpoint: hold => { holdInspection = hold; },
    settleCheckpoint: message => { const next = inspections.shift(); if (!next) throw new Error("No pending checkpoint inspection"); if (message) next.reject(message); else next.resolve(); },
    rejectAbort: message => { const next = aborts.shift(); if (!next) throw new Error("No pending abort"); next.reject(message); },
    resolveSaveDialog: path => { const resolve = saveDialog; saveDialog = null; if (!resolve) throw new Error("No pending save dialog"); resolve(path); },
    settleReceive: (state, error) => { const resolve = receive; receive = null; if (!resolve) throw new Error("No pending receive"); resolve({ state, error: error ?? null, retryAvailable: state !== "complete", updatedUnixMs: Date.now() }); },
  };
}
