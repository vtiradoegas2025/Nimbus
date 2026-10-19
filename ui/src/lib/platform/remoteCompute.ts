export interface RemoteComputeTarget {
  id: string;
  name: string;
  host: string;
  binaryPath: string;
  workDirectory: string;
  assetMappings?: { localPath: string; remotePath: string }[];
}

export interface RemoteComputeProbe {
  available: boolean;
  platform: "linux";
  binaryPath: string;
  binaryFingerprint: string;
  catalogFingerprint: string;
  catalog: unknown;
  workDirectory: string;
}

export interface RemoteReceiveSummary {
  state: "available" | "receiving" | "complete" | "failed" | "cancelled" | "invalid";
  error: string | null;
  retryAvailable: boolean;
  updatedUnixMs: number;
}

export const REMOTE_COMPUTE_PROFILES_CHANGE_EVENT = "nimbus-studio:remote-compute-change";
export const REMOTE_RECEIVE_CHANGE_EVENT = "nimbus-studio:remote-receive-change";
export const REMOTE_COMPUTE_PROFILE_LIMIT = 16;
const STORAGE_KEY = "nimbus-studio:remote-compute:v1";

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim() || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new Error(`${label} must contain text without control characters.`);
  }
  return value.trim();
}

export function normalizeRemoteComputeTarget(value: unknown): RemoteComputeTarget {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Remote compute profile is invalid.");
  const record = value as Record<string, unknown>;
  const id = text(record.id, "Profile identifier");
  if (id === "local" || id === "new") throw new Error("Remote compute profile identifier is reserved.");
  const name = text(record.name, "Connection name");
  if (new TextEncoder().encode(id).byteLength > 128 || new TextEncoder().encode(name).byteLength > 256) {
    throw new Error("Profile identifier or connection name is too long.");
  }
  const host = text(record.host, "SSH host");
  const binaryPath = text(record.binaryPath, "Remote simulator path");
  const rawWorkDirectory = text(record.workDirectory, "Remote work directory");
  if (/^\/+$/u.test(rawWorkDirectory)) throw new Error("Remote work directory cannot be the filesystem root.");
  const workDirectory = rawWorkDirectory.replace(/\/+$/, "");
  if (!/^[A-Za-z0-9][A-Za-z0-9._@:[\]-]{0,254}$/.test(host)) {
    throw new Error("SSH host must be an SSH configuration alias or user@host.");
  }
  if (!binaryPath.startsWith("/") || !workDirectory.startsWith("/")) {
    throw new Error("Remote simulator and work directory must be absolute Linux paths.");
  }
  if (binaryPath.split("/").includes("..") || workDirectory.split("/").includes("..")) {
    throw new Error("Remote simulator and work directory cannot contain parent directory references.");
  }
  const target: RemoteComputeTarget = { id, name, host, binaryPath, workDirectory };
  if (record.assetMappings != null) {
    if (!Array.isArray(record.assetMappings) || record.assetMappings.length > 32) throw new Error("Input file mappings are invalid.");
    const destinations = new Set<string>();
    target.assetMappings = record.assetMappings.map((entry: unknown) => {
      if (!entry || typeof entry !== "object") throw new Error("Input file mapping is invalid.");
      const mapping = entry as Record<string, unknown>;
      const localPath = text(mapping.localPath, "Local input file");
      const remotePath = text(mapping.remotePath, "Remote input file");
      if (!localPath.startsWith("/") && !/^[A-Za-z]:[\\/]/.test(localPath)) throw new Error("Local input files must use absolute paths.");
      if (remotePath.split("/").includes("..") || (remotePath.startsWith("/")
          && !remotePath.startsWith(`${workDirectory}/`))) throw new Error("Remote input files must stay inside the work directory.");
      const relative = remotePath.startsWith("/") ? remotePath.slice(workDirectory.length + 1) : remotePath;
      const destination = relative.split("/").filter(part => part && part !== ".").join("/");
      if (!destination) throw new Error("Remote input files must name a file inside the work directory.");
      if (destinations.has(destination)) throw new Error("Input files cannot share a remote destination.");
      destinations.add(destination);
      return { localPath, remotePath };
    });
  }
  return target;
}

function normalizeProfiles(value: unknown): RemoteComputeTarget[] {
  if (!Array.isArray(value) || value.length > REMOTE_COMPUTE_PROFILE_LIMIT) throw new Error("Saved remote compute profiles are invalid.");
  const profiles = value.map(normalizeRemoteComputeTarget);
  if (new Set(profiles.map(profile => profile.id)).size !== profiles.length) throw new Error("Remote compute profile identifiers must be unique.");
  return profiles;
}

export function loadRemoteComputeProfiles(storage?: Pick<Storage, "getItem">): { profiles: RemoteComputeTarget[]; error: string | null } {
  try {
    const source = storage ?? (typeof window === "undefined" ? null : window.localStorage);
    const raw = source?.getItem(STORAGE_KEY);
    if (!raw) return { profiles: [], error: null };
    const value = JSON.parse(raw) as { version?: unknown; profiles?: unknown };
    if (value?.version !== 1) throw new Error("Saved remote compute profiles use an unsupported format.");
    return { profiles: normalizeProfiles(value.profiles), error: null };
  } catch (error) {
    return { profiles: [], error: error instanceof Error ? error.message : String(error) };
  }
}

export function saveRemoteComputeProfiles(profiles: readonly RemoteComputeTarget[], storage?: Pick<Storage, "setItem">): RemoteComputeTarget[] {
  const normalized = normalizeProfiles(profiles);
  const destination = storage ?? (typeof window === "undefined" ? null : window.localStorage);
  if (!destination) throw new Error("Connection storage is unavailable.");
  destination.setItem(STORAGE_KEY, JSON.stringify({ version: 1, profiles: normalized }));
  if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent(REMOTE_COMPUTE_PROFILES_CHANGE_EVENT));
  return normalized;
}

export function remoteComputeTargetKey(target: RemoteComputeTarget | null): string {
  return target == null ? "local" : JSON.stringify(normalizeRemoteComputeTarget(target));
}

export function selectedRemoteComputeTarget(profiles: readonly RemoteComputeTarget[], id: string): RemoteComputeTarget | null | undefined {
  return id === "local" ? null : profiles.find(profile => profile.id === id);
}
