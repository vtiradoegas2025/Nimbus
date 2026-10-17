import { mockIPC } from "@tauri-apps/api/mocks";
import type { RemoteWorkspaceStatus } from "@nimbus/lib/platform/remoteWorkspace";
let status: RemoteWorkspaceStatus = { state: "disconnected", endpoint: null, url: null, error: null };
let release: (() => void) | null = null;
let epoch = 0;
let opened = 0;
export function installRemoteWorkspaceFixture() {
  mockIPC(async (command, payload) => {
    if (command === "remote_workspace_status") return status;
    if (command === "connect_remote_workspace") {
      const current = ++epoch;
      status = { state: "connecting", endpoint: (payload as { endpoint: RemoteWorkspaceStatus["endpoint"] }).endpoint, url: null, error: null };
      void new Promise<void>(resolve => { release = resolve; }).then(() => {
        if (epoch === current) status = { ...status, state: "connected", url: "http://127.0.0.1:34567/vnc.html" };
      });
      return status;
    }
    if (command === "disconnect_remote_workspace") {
      epoch++;
      status = { state: "disconnected", endpoint: null, url: null, error: null };
      release?.(); release = null;
      return status;
    }
    if (command === "open_remote_workspace") { if (status.state !== "connected") throw new Error("No tunnel"); opened++; return null; }
    throw new Error(`Unexpected remote workspace fixture command: ${command}`);
  });
  window.__nimbusRemoteWorkspaceFixture = {
    release: () => { release?.(); release = null; },
    disconnect: () => { epoch++; status = { ...status, state: "failed", url: null, error: "SSH disconnected" }; },
    opened: () => opened,
  };
}
declare global {
  interface Window { __nimbusRemoteWorkspaceFixture: { release: () => void; disconnect: () => void; opened: () => number } }
}
