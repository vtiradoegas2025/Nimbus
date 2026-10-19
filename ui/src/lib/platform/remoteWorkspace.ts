import { invokeCommand } from "./appInvoke";

export interface RemoteWorkspaceEndpoint {
  id: string;
  name: string;
  host: string;
  port: number;
  path: string;
}
export interface RemoteWorkspaceStatus {
  state: "disconnected" | "connecting" | "connected" | "failed";
  endpoint: RemoteWorkspaceEndpoint | null;
  url: string | null;
  error: string | null;
}
const KEY = "nimbus-studio:remote-workspaces:v1";
export const REMOTE_WORKSPACE_CHANGE = "nimbus-studio:remote-workspaces-change";
export const REMOTE_WORKSPACE_LIMIT = 16;

export function normalizeRemoteWorkspace(value: unknown): RemoteWorkspaceEndpoint {
  if (!value || typeof value !== "object") throw new Error("Invalid remote workspace");
  const v = value as Record<string, unknown>;
  const text = (key: string, limit: number) => {
    const raw = v[key];
    if (typeof raw !== "string" || !raw.trim() || /[\u0000-\u001f\u007f]/.test(raw)
      || new TextEncoder().encode(raw.trim()).length > limit) throw new Error(`Invalid workspace ${key}`);
    return raw.trim();
  };
  const id = text("id", 128), name = text("name", 256), host = text("host", 255), path = text("path", 1024);
  if (!/^[A-Za-z0-9][A-Za-z0-9._@:[\]-]*$/.test(host)) throw new Error("SSH host must be an alias or user@host");
  if (!Number.isInteger(v.port) || Number(v.port) < 1 || Number(v.port) > 65535) throw new Error("Workspace port must be between 1 and 65535");
  if (!/^\/(?!\/)[A-Za-z0-9/_.-]*$/.test(path) || path.split("/").some(part => part === ".." || part === ".")) {
    throw new Error("Workspace path must be an absolute URL path without query parameters or credentials");
  }
  return { id, name, host, path, port: Number(v.port) };
}

function profiles(value: unknown): RemoteWorkspaceEndpoint[] {
  if (!Array.isArray(value) || value.length > REMOTE_WORKSPACE_LIMIT) throw new Error("Invalid saved remote workspaces");
  const result = value.map(normalizeRemoteWorkspace);
  if (new Set(result.map(item => item.id)).size !== result.length) throw new Error("Workspace identifiers must be unique");
  return result;
}
export function loadRemoteWorkspaces(storage?: Pick<Storage, "getItem">): { profiles: RemoteWorkspaceEndpoint[]; error: string | null } {
  try {
    const raw = (storage ?? window.localStorage).getItem(KEY);
    if (!raw) return { profiles: [], error: null };
    const value = JSON.parse(raw);
    if (value.version !== 1) throw new Error("Unsupported remote workspace settings");
    return { profiles: profiles(value.profiles), error: null };
  } catch (error) { return { profiles: [], error: String(error instanceof Error ? error.message : error) }; }
}
export function saveRemoteWorkspaces(value: readonly RemoteWorkspaceEndpoint[], storage?: Pick<Storage, "setItem">): RemoteWorkspaceEndpoint[] {
  const normalized = profiles(value);
  (storage ?? window.localStorage).setItem(KEY, JSON.stringify({ version: 1, profiles: normalized }));
  if (typeof window !== "undefined") window.dispatchEvent(new Event(REMOTE_WORKSPACE_CHANGE));
  return normalized;
}
export const remoteWorkspaceStatus = () => invokeCommand<RemoteWorkspaceStatus>("remote_workspace_status");
export const connectRemoteWorkspace = (endpoint: RemoteWorkspaceEndpoint) => invokeCommand<RemoteWorkspaceStatus>("connect_remote_workspace", { endpoint: normalizeRemoteWorkspace(endpoint) });
export const disconnectRemoteWorkspace = () => invokeCommand<RemoteWorkspaceStatus>("disconnect_remote_workspace");
export const openRemoteWorkspace = () => invokeCommand<void>("open_remote_workspace");
