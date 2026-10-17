import { mockIPC } from "@tauri-apps/api/mocks";
import type { RemoteWorkspaceStatus } from "@nimbus/lib/platform/remoteWorkspace";
import { initializeStudioPerformance, startAutomaticWorkerMeasurement } from "@nimbus/lib/studio/studioPerformance";

const hardware = { cpuBrand: "Synthetic settings CPU", coreCount: 8, ramBytes: 16 * 1024 ** 3 };

export function installSettingsFixture() {
  const calls: { command: string; payload: Record<string, unknown> }[] = [];
  let release: (() => void) | null = null;
  let hold = false;
  let holdBudget = false;
  let releaseBudget: (() => void) | null = null;
  let limitMiB = 512;
  let measurement: { requestId: string; resolve: (value: unknown) => void } | null = null;
  const budget = () => ({ limitBytes: limitMiB * 1024 ** 2, reservedBytes: 0, waitingReads: 0,
    automaticBytes: 512 * 1024 ** 2, maximumBytes: 4096 * 1024 ** 2 });
  let workspace: RemoteWorkspaceStatus = { state: "disconnected", endpoint: null, url: null, error: null };
  mockIPC(async (command, args) => {
    const payload = (args ?? {}) as Record<string, unknown>;
    calls.push({ command, payload });
    if (command === "configure_native_io_budget") {
      const next = Number(payload.limitMib || 512);
      if (next < 64 || next > 4096) throw new Error("I/O memory limit must be 64..=4096 MiB, or 0 for automatic");
      if (holdBudget) await new Promise<void>(resolve => { releaseBudget = resolve; });
      limitMiB = next;
      return budget();
    }
    if (command === "native_io_budget_status") return budget();
    if (command === "profile_loading_workers") return new Promise(resolve => { measurement = { requestId: String(payload.requestId), resolve }; });
    if (command === "loading_worker_profile_status") return measurement
      ? { requestId: measurement.requestId, workers: 2, completed: 1, total: 4 } : null;
    if (command === "cancel_loading_worker_profile") {
      if (!measurement || measurement.requestId !== payload.requestId) return false;
      measurement.resolve(null); measurement = null; return true;
    }
    if (command === "project_root") return "/fixture/Nimbus";
    if (command === "list_configs") return ["custom/rtx5090_lifecycle_50m/01_classic_supercell.yaml"];
    if (command === "simulator_status") {
      if (hold) await new Promise<void>(resolve => { release = resolve; });
      return { state: "ready", binaryPath: payload.simulatorPath || "/fixture/nimbus_sim", gpu: "Fixture GPU" };
    }
    if (command === "ffmpeg_detect") return { path: payload.overridePath || "/fixture/ffmpeg", version: "fixture encoder" };
    if (command === "probe_remote_compute") {
      const target = payload.executionTarget as Record<string, unknown>;
      return { available: true, platform: "linux", binaryPath: target.binaryPath, workDirectory: target.workDirectory, binaryFingerprint: "fixture-binary", catalogFingerprint: "fixture-catalog", catalog: {} };
    }
    if (command === "remote_workspace_status") return workspace;
    if (command === "connect_remote_workspace") {
      workspace = { state: "connected", endpoint: payload.endpoint as RemoteWorkspaceStatus["endpoint"], url: "http://127.0.0.1:34567/vnc.html", error: null };
      return workspace;
    }
    if (command === "disconnect_remote_workspace") {
      workspace = { state: "disconnected", endpoint: null, url: null, error: null };
      return workspace;
    }
    if (command === "open_remote_workspace") return null;
    throw new Error(`Unexpected settings command: ${command}`);
  });
  window.__nimbusSettingsFixture = {
    calls: () => calls,
    holdSimulator: () => { hold = true; },
    releaseSimulator: () => { hold = false; release?.(); release = null; },
    holdBudget: () => { holdBudget = true; },
    releaseBudget: () => { holdBudget = false; releaseBudget?.(); releaseBudget = null; },
    startAutomaticMeasurement: startAutomaticWorkerMeasurement,
    finishMeasurement: () => {
      measurement?.resolve({ schema: "nimbus.studio.worker-profile", schemaVersion: 1, benchmark: "npy-load-scan-v1",
        hardware, recommendedWorkers: 2, bytesPerTrial: 32 * 1024 ** 2, elapsedMs: 200,
        trials: [{ workers: 1, medianMs: 20, throughputMibS: 1600 }, { workers: 2, medianMs: 10, throughputMibS: 3200 }] });
      measurement = null;
    },
  };
  return initializeStudioPerformance(hardware, true);
}

declare global {
  interface Window {
    __nimbusSettingsFixture: {
      calls: () => { command: string; payload: Record<string, unknown> }[];
      holdSimulator: () => void;
      releaseSimulator: () => void;
      holdBudget: () => void;
      releaseBudget: () => void;
      startAutomaticMeasurement: () => void;
      finishMeasurement: () => void;
    };
  }
}
