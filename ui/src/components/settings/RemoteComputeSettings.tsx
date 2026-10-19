import { useEffect, useId, useRef, useState } from "react";
import {
  loadRemoteComputeProfiles,
  normalizeRemoteComputeTarget,
  REMOTE_COMPUTE_PROFILE_LIMIT,
  saveRemoteComputeProfiles,
  type RemoteComputeTarget,
  type RemoteComputeProbe,
} from "@/lib/platform/remoteCompute";
import { isTauriRuntime, probeRemoteCompute } from "@/lib/platform/tauri";

function newProfile(): RemoteComputeTarget {
  return { id: crypto.randomUUID(), name: "", host: "", binaryPath: "", workDirectory: "" };
}

export function RemoteComputeSettings() {
  const [initial] = useState(() => loadRemoteComputeProfiles());
  const [profiles, setProfiles] = useState(initial.profiles);
  const [draft, setDraft] = useState<RemoteComputeTarget>(() => initial.profiles[0] ?? newProfile());
  const [message, setMessage] = useState(initial.error);
  const [failed, setFailed] = useState(initial.error != null);
  const [testing, setTesting] = useState(false);
  const [checked, setChecked] = useState<RemoteComputeProbe | null>(null);
  const [inputsOpen, setInputsOpen] = useState(false);
  const selectedSaved = profiles.find(profile => profile.id === draft.id);
  const dirty = JSON.stringify(draft) !== JSON.stringify(selectedSaved);
  const probeSequence = useRef(0);
  const profileSelectId = useId();
  const importRef = useRef<HTMLInputElement>(null);
  useEffect(() => () => { probeSequence.current++; }, []);

  const edit = (change: (previous: RemoteComputeTarget) => RemoteComputeTarget) => {
    setDraft(change);
    setChecked(null);
    probeSequence.current++;
    setMessage(null);
    setFailed(false);
  };
  const update = (key: keyof RemoteComputeTarget, value: string) => edit(previous => ({ ...previous, [key]: value }));
  const select = (profile: RemoteComputeTarget) => {
    setChecked(null);
    setInputsOpen((profile.assetMappings?.length ?? 0) > 0);
    probeSequence.current++;
    setDraft(profile);
    setMessage(null);
    setFailed(false);
  };
  const save = () => {
    try {
      const normalized = normalizeRemoteComputeTarget(draft);
      const next = profiles.some(profile => profile.id === normalized.id)
        ? profiles.map(profile => profile.id === normalized.id ? normalized : profile)
        : [...profiles, normalized];
      setProfiles(saveRemoteComputeProfiles(next));
      setDraft(normalized);
      setMessage("Connection saved. Select it in Configure to validate and run on that computer.");
      setFailed(false);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
      setFailed(true);
    }
  };
  const remove = () => {
    try {
      const next = saveRemoteComputeProfiles(profiles.filter(profile => profile.id !== draft.id));
      setProfiles(next);
      select(next[0] ?? newProfile());
      setMessage("Connection removed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
      setFailed(true);
    }
  };
  const test = async () => {
    const sequence = ++probeSequence.current;
    setTesting(true);
    setChecked(null);
    setMessage(null);
    try {
      const result = await probeRemoteCompute(normalizeRemoteComputeTarget(draft));
      if (sequence !== probeSequence.current) return;
      if (!result.available) throw new Error("The remote simulator is unavailable.");
      setChecked(result);
      setMessage(`Connected to ${draft.name}: Linux simulator verified at ${result.binaryPath}.`);
      setFailed(false);
    } catch (error) {
      if (sequence !== probeSequence.current) return;
      setMessage(error instanceof Error ? error.message : String(error));
      setFailed(true);
    } finally {
      if (sequence === probeSequence.current) setTesting(false);
    }
  };

  return (
    <section className="settings-group" aria-labelledby="remote-compute-heading">
      <h3 className="settings-group__title" id="remote-compute-heading">Remote compute</h3>
      <div className="settings-group__body">
        <p className="settings-row__hint">Save a Linux SSH connection, then select it as the Compute target in Configure. Each new configuration starts on this computer.</p>
        <div className="settings-row">
          <div className="settings-row__label"><label htmlFor={profileSelectId}>Saved connection</label></div>
          <div className="settings-row__control">
            <select id={profileSelectId} value={selectedSaved?.id ?? "new"} disabled={testing}
              onChange={event => select(profiles.find(profile => profile.id === event.target.value) ?? newProfile())}>
              <option value="new">New connection</option>
              {profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
            </select>
          </div>
        </div>
        <ConnectionField label="Connection name" value={draft.name} disabled={testing} onChange={value => update("name", value)} placeholder="Research workstation" />
        <ConnectionField label="SSH host" value={draft.host} disabled={testing} onChange={value => update("host", value)} placeholder="workstation"
          hint="Use your SSH configuration alias or user@host with existing SSH keys, agent, and known host verification." />
        <ConnectionField label="Remote simulator path" value={draft.binaryPath} disabled={testing} onChange={value => update("binaryPath", value)} placeholder="/opt/nimbus/bin/nimbus_sim" />
        <ConnectionField label="Remote work directory" value={draft.workDirectory} disabled={testing} onChange={value => update("workDirectory", value)} placeholder="/home/you/nimbus"
          hint="A writable directory on the Linux computer for staged inputs and simulation outputs." />
        <details className="settings-remote__inputs" open={inputsOpen} onToggle={event => setInputsOpen(event.currentTarget.open)}>
          <summary>Input files ({draft.assetMappings?.length ?? 0})</summary>
          <p>Map local soundings and other input files to the paths used in your YAML. Relative destinations are staged separately for each run.</p>
        {(draft.assetMappings ?? []).map((mapping, index) => (
          <div className="settings-remote__mapping" key={index}>
            <ConnectionField label={`Local input file ${index + 1}`} value={mapping.localPath} disabled={testing} onChange={value => {
              edit(previous => ({ ...previous, assetMappings: previous.assetMappings!.map((entry, row) => row === index ? { ...entry, localPath: value } : entry) }));
            }} />
            <ConnectionField label={`Remote input file ${index + 1}`} value={mapping.remotePath} disabled={testing} onChange={value => {
              edit(previous => ({ ...previous, assetMappings: previous.assetMappings!.map((entry, row) => row === index ? { ...entry, remotePath: value } : entry) }));
            }} />
            <button type="button" disabled={testing} title={testing ? "Wait for the connection check" : "Remove this input mapping"} onClick={() => edit(previous => ({ ...previous, assetMappings: previous.assetMappings!.filter((_, row) => row !== index) }))}>Remove input file {index + 1}</button>
          </div>
        ))}
        <div className="settings-remote__actions">
          <button type="button" disabled={testing || (draft.assetMappings?.length ?? 0) >= 32} title={testing ? "Wait for the connection check to finish" : (draft.assetMappings?.length ?? 0) >= 32 ? "A connection supports at most 32 input files" : "Add an input file to copy to the remote computer"} onClick={() => edit(previous => ({ ...previous, assetMappings: [...(previous.assetMappings ?? []), { localPath: "", remotePath: "" }] }))}>Add input file</button>
        </div>
        </details>
        <p className="settings-remote__status">{dirty ? "Unsaved connection changes" : "Connection saved"} · {checked ? "checked this session" : "connection not checked for these settings"}</p>
        <div className="settings-remote__actions">
          <button type="button" className="btn btn--primary" onClick={save} disabled={testing || !dirty || (!selectedSaved && profiles.length >= REMOTE_COMPUTE_PROFILE_LIMIT)} title={!dirty ? "This connection is already saved" : "Save this connection locally"}>Save connection</button>
          <button type="button" onClick={() => void test()} disabled={testing || !isTauriRuntime()} title={isTauriRuntime() ? "Check SSH access and the remote simulator" : "Connection checks require Nimbus Studio desktop"}>{testing ? "Checking…" : "Check connection"}</button>
          <button type="button" disabled={testing} title={testing ? "Wait for the connection check" : "Load a connection file for review; saving is a separate action"} onClick={() => importRef.current?.click()}>Import connection…</button>
          <button type="button" onClick={remove} disabled={testing || !selectedSaved} title={selectedSaved ? "Remove this saved connection" : "Save this connection before removing it"}>Remove connection</button>
        </div>
        <input ref={importRef} type="file" accept=".json,application/json" hidden onChange={async event => {
          const file = event.target.files?.[0]; event.target.value = "";
          if (!file) return;
          const sequence = ++probeSequence.current;
          try {
            if (file.size > 1024 * 1024) throw new Error("Connection file exceeds 1 MiB.");
            const target = normalizeRemoteComputeTarget(JSON.parse(await file.text()));
            if (sequence !== probeSequence.current) return;
            select(target); setMessage("Connection loaded for review. Save connection to keep it.");
          } catch (error) {
            if (sequence !== probeSequence.current) return;
            setMessage(error instanceof Error ? error.message : String(error)); setFailed(true);
          }
        }} />
        {checked && <div className="settings-remote__verified"><strong>Connection check passed</strong><dl>
          <dt>Platform</dt><dd>Linux</dd><dt>Simulator</dt><dd>{checked.binaryPath}</dd><dt>Work directory</dt><dd>{checked.workDirectory}</dd>
        </dl><p className="settings-row__hint">This verifies the connection and simulator identity. Validate the selected configuration and complete a short run before a large run.</p></div>}
        <details className="settings-remote__inputs"><summary>Connection requirements and session lifetime</summary>
        <p className="settings-row__hint">Runs use direct SSH on Linux with Python 3.8 or newer. Relative input paths are copied into each run’s private directory. Absolute input paths must already exist inside the work directory with matching bytes. Use the mapped paths in the YAML. Plots and playback use this computer’s hardware. To render on a node, open Settings → Remote rendering.</p>
        <p className="settings-row__hint">Remote simulations launched here are owned by this Studio session. Keep this computer awake, Studio open and SSH connected until completion. Closing Studio or losing the connection requests a remote stop; completed saved results can be received again.</p>
        </details>
        {message && <p className="settings-note" role={failed ? "alert" : "status"}>{message}</p>}
      </div>
    </section>
  );
}

function ConnectionField({ label, hint, value, placeholder, disabled, onChange }: {
  label: string; hint?: string; value: string; placeholder?: string; disabled: boolean; onChange: (value: string) => void;
}) {
  const id = useId();
  const hintId = useId();
  return <div className="settings-row">
    <div className="settings-row__label"><label htmlFor={id}>{label}</label>{hint && <span className="settings-row__hint" id={hintId}>{hint}</span>}</div>
    <div className="settings-row__control"><input id={id} value={value} placeholder={placeholder} disabled={disabled} spellCheck={false} aria-describedby={hint ? hintId : undefined} onChange={event => onChange(event.target.value)} /></div>
  </div>;
}
