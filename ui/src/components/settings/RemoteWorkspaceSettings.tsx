import { useEffect, useId, useRef, useState } from "react";
import { isTauriRuntime } from "@/lib/platform/tauri";
import { loadRemoteComputeProfiles, REMOTE_COMPUTE_PROFILES_CHANGE_EVENT } from "@/lib/platform/remoteCompute";
import {
  connectRemoteWorkspace, disconnectRemoteWorkspace, openRemoteWorkspace, remoteWorkspaceStatus,
  loadRemoteWorkspaces, saveRemoteWorkspaces, normalizeRemoteWorkspace, REMOTE_WORKSPACE_LIMIT,
  type RemoteWorkspaceEndpoint, type RemoteWorkspaceStatus,
} from "@/lib/platform/remoteWorkspace";
import "./Settings.css";
import "./RemoteWorkspaceSettings.css";

const fresh = (): RemoteWorkspaceEndpoint => ({ id: crypto.randomUUID(), name: "", host: "", port: 6080, path: "/vnc.html" });
const disconnected: RemoteWorkspaceStatus = { state: "disconnected", endpoint: null, url: null, error: null };

export function RemoteWorkspaceSettings({ onClose, visible = true }: { onClose?: () => void; visible?: boolean }) {
  const [initial] = useState(() => loadRemoteWorkspaces());
  const [profiles, setProfiles] = useState(initial.profiles);
  const [draft, setDraft] = useState(() => initial.profiles[0] ?? fresh());
  const [status, setStatus] = useState(disconnected);
  const [message, setMessage] = useState(initial.error);
  const [failed, setFailed] = useState(initial.error !== null);
  const [busy, setBusy] = useState(false);
  const [computeProfiles, setComputeProfiles] = useState(() => loadRemoteComputeProfiles().profiles);
  const sequence = useRef(0);
  const actionPending = useRef(false);
  const mounted = useRef(true);
  const desktop = isTauriRuntime();
  const id = useId();
  const saved = profiles.find(profile => profile.id === draft.id);
  const active = status.state === "connected" || status.state === "connecting";
  const dirty = JSON.stringify(saved) !== JSON.stringify(draft);

  useEffect(() => {
    const refresh = () => setComputeProfiles(loadRemoteComputeProfiles().profiles);
    window.addEventListener(REMOTE_COMPUTE_PROFILES_CHANGE_EVENT, refresh);
    window.addEventListener("storage", refresh);
    return () => {
      window.removeEventListener(REMOTE_COMPUTE_PROFILES_CHANGE_EVENT, refresh);
      window.removeEventListener("storage", refresh);
    };
  }, []);

  useEffect(() => {
    const requests = sequence;
    mounted.current = true;
    return () => { mounted.current = false; requests.current++; };
  }, []);

  useEffect(() => {
    if (!desktop || !visible) return;
    let cancelled = false;
    let polling = false;
    const poll = async () => {
      if (!desktop || polling || actionPending.current) return;
      polling = true;
      const current = sequence.current;
      try { const next = await remoteWorkspaceStatus(); if (!cancelled && mounted.current && current === sequence.current) setStatus(next); }
      catch (error) { if (!cancelled && mounted.current && current === sequence.current) { setMessage(String(error)); setFailed(true); } }
      finally { polling = false; }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [desktop, visible]);

  const run = async (action: () => Promise<RemoteWorkspaceStatus | void>) => {
    const current = ++sequence.current;
    actionPending.current = true; setBusy(true); setMessage(null); setFailed(false);
    try {
      const next = await action();
      if (mounted.current && current === sequence.current && next) setStatus(next);
    } catch (error) {
      if (mounted.current && current === sequence.current) {
        setMessage(error instanceof Error ? error.message : String(error)); setFailed(true);
        if (status.state !== "connected") setStatus(disconnected);
      }
    } finally {
      if (current === sequence.current) { actionPending.current = false; if (mounted.current) setBusy(false); }
    }
  };
  const update = <K extends keyof RemoteWorkspaceEndpoint>(key: K, value: RemoteWorkspaceEndpoint[K]) => {
    setDraft(previous => ({ ...previous, [key]: value })); setMessage(null); setFailed(false);
  };
  const save = () => {
    try {
      const next = normalizeRemoteWorkspace(draft);
      setProfiles(saveRemoteWorkspaces(saved ? profiles.map(profile => profile.id === next.id ? next : profile) : [...profiles, next]));
      setDraft(next); setMessage("Remote workspace saved."); setFailed(false);
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); setFailed(true); }
  };
  const remove = () => {
    try {
      const next = saveRemoteWorkspaces(profiles.filter(profile => profile.id !== draft.id));
      setProfiles(next); setDraft(next[0] ?? fresh()); setMessage("Remote workspace removed."); setFailed(false);
    } catch (error) { setMessage(String(error)); setFailed(true); }
  };

  return <section className="settings-group remote-workspace" aria-labelledby={`${id}-heading`}>
    <header className="remote-workspace__heading">
      <h3 className="settings-group__title" id={`${id}-heading`}>Remote visualization workspace</h3>
      {onClose && <button type="button" onClick={onClose} aria-label="Close remote workspace panel">Close</button>}
    </header>
    <div className="settings-group__body">
      <p className="settings-row__hint">Run Studio in a desktop on the node to use its hardware for 2D and 3D plots, Python, playback and exports. This opens that desktop in your browser through SSH. Local plots continue to use this computer.</p>
      <div className="settings-row"><label htmlFor={`${id}-saved`}>Saved workspace</label><div className="settings-row__control">
        <select id={`${id}-saved`} value={saved?.id ?? "new"} disabled={busy} onChange={event => { setDraft(profiles.find(profile => profile.id === event.target.value) ?? fresh()); setMessage(null); }}>
          <option value="new">New workspace</option>{profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.name} · {profile.host}</option>)}
        </select>
      </div></div>
      {computeProfiles.length > 0 && <div className="settings-row"><label htmlFor={`${id}-compute`}>Copy SSH host from</label><div className="settings-row__control">
        <select id={`${id}-compute`} value="" disabled={busy} onChange={event => {
          const profile = computeProfiles.find(item => item.id === event.target.value);
          if (profile) setDraft(previous => ({ ...previous, name: profile.name, host: profile.host }));
        }}><option value="">Choose a compute connection</option>{computeProfiles.map(profile => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select>
      </div></div>}
      <Field label="Workspace name" value={draft.name} disabled={busy} onChange={value => update("name", value)} placeholder="Research desktop" />
      <Field label="Workspace SSH host" value={draft.host} disabled={busy} onChange={value => update("host", value)} placeholder="workstation or user@host" />
      <Field label="Remote web port" value={draft.port || ""} type="number" disabled={busy} onChange={value => update("port", Number(value))} placeholder="6080" />
      <Field label="Workspace URL path" value={draft.path} disabled={busy} onChange={value => update("path", value)} placeholder="/vnc.html" />
      <p className="settings-row__hint">Requires an existing browser-accessible desktop service such as noVNC, listening on 127.0.0.1 on the node, and Python 3. Nimbus uses your SSH keys and known hosts. Sign in to the desktop in the browser; credentials are not saved here.</p>
      <div className="settings-remote__actions">
        <button type="button" onClick={save} title={busy ? "Wait for the current connection action" : !dirty ? "This workspace is already saved" : !saved && profiles.length >= REMOTE_WORKSPACE_LIMIT ? "Remove a workspace before adding another" : "Save this workspace locally"} disabled={busy || !dirty || (!saved && profiles.length >= REMOTE_WORKSPACE_LIMIT)}>Save workspace</button>
        <button type="button" disabled={!desktop || busy || active} title={!desktop ? "Requires Nimbus Studio desktop" : active ? "Disconnect the current workspace first" : "Check the service and open an SSH tunnel"} onClick={() => void run(async () => {
          const endpoint = normalizeRemoteWorkspace(draft);
          setStatus({ state: "connecting", endpoint, url: null, error: null });
          return connectRemoteWorkspace(endpoint);
        })}>Connect workspace</button>
        <button type="button" disabled={!desktop || busy || status.state !== "connected"} title={!desktop ? "Requires Nimbus Studio desktop" : busy ? "Wait for the current connection action" : status.state !== "connected" ? "Connect a workspace before opening its desktop" : "Open the connected remote desktop in your browser"} onClick={() => void run(openRemoteWorkspace)}>Open remote desktop</button>
        <button type="button" disabled={!desktop || !active || busy} title={!desktop ? "Requires Nimbus Studio desktop" : busy ? "Wait for the connection action to acknowledge" : !active ? "There is no active workspace connection" : "Close the display tunnel; jobs are managed on the node"} onClick={() => void run(disconnectRemoteWorkspace)}>{status.state === "connecting" ? "Cancel connection" : "Disconnect workspace"}</button>
        <button type="button" disabled={busy || !saved || status.endpoint?.id === draft.id && active} title={busy ? "Wait for the current connection action" : !saved ? "Save a workspace before removing it" : status.endpoint?.id === draft.id && active ? "Disconnect this workspace before removing it" : "Remove this saved workspace"} onClick={remove}>Remove workspace</button>
      </div>
      <p className="settings-note" role={status.error ? "alert" : "status"}>
        {status.state === "connecting" ? `Connecting to ${status.endpoint?.host}…` : status.state === "connected" ? `Connected to ${status.endpoint?.name} on ${status.endpoint?.host}. Open the remote desktop to work on that machine.` : status.error ?? "Rendering here uses this computer. Connect a workspace to work on a node."}
      </p>
      <p className="settings-row__hint">Keep this Studio open to retain the tunnel. Disconnect closes the display connection; jobs running in the remote desktop follow that desktop’s lifecycle. This does not move the current local plot or its data. Open node datasets there, and export from the node’s Studio for full-quality images and movies. Display streaming quality is controlled by the desktop service.</p>
      {message && <p className="settings-note" role={failed ? "alert" : "status"}>{message}</p>}
    </div>
  </section>;
}
function Field({ label, value, disabled, onChange, placeholder, type = "text" }: {
  label: string; value: string | number; disabled: boolean; onChange: (value: string) => void; placeholder: string; type?: "text" | "number";
}) {
  const id = useId();
  return <div className="settings-row"><label htmlFor={id}>{label}</label><div className="settings-row__control"><input id={id} type={type} value={value} disabled={disabled} placeholder={placeholder} spellCheck={false} onChange={event => onChange(event.target.value)} /></div></div>;
}
