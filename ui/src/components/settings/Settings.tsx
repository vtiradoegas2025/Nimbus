import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { AsyncState } from "@/lib/core/hooks";
import type { SimulatorStatus } from "@/lib/core/types";
import {
  effectiveRunsDirectory,
  loadStudioSettings,
  studioSettingsEqual,
  type StudioAppearance,
  type StudioPlotSide,
  type StudioSettings,
} from "@/lib/studio/studioSettings";
import { getProjectRoot, listTemplateConfigs } from "@/lib/platform/tauri";
import { Icon } from "@/components/shell/Icon";
import { RemoteWorkspaceSettings } from "./RemoteWorkspaceSettings";
import { RemoteComputeSettings } from "./RemoteComputeSettings";
import { SettingsPathCheck } from "./SettingsPathCheck";
import "./Settings.css";
import { PerformanceSettings } from "./PerformanceSettings";
import { saveStudioPerformanceSettings } from "@/lib/studio/studioPerformance";

function moveRovingRadioFocus<T extends string>(
  event: React.KeyboardEvent<HTMLButtonElement>,
  onSelect: (value: T) => void,
) {
  const group = event.currentTarget.closest<HTMLElement>('[role="radiogroup"]');
  if (!group) return;
  const radios = Array.from(
    group.querySelectorAll<HTMLButtonElement>('[role="radio"]'),
  );
  const currentIndex = radios.indexOf(event.currentTarget);
  if (currentIndex < 0) return;

  let nextIndex: number;
  switch (event.key) {
    case "ArrowRight":
    case "ArrowDown":
      nextIndex = (currentIndex + 1) % radios.length;
      break;
    case "ArrowLeft":
    case "ArrowUp":
      nextIndex = (currentIndex - 1 + radios.length) % radios.length;
      break;
    case "Home":
      nextIndex = 0;
      break;
    case "End":
      nextIndex = radios.length - 1;
      break;
    default:
      return;
  }

  event.preventDefault();
  const next = radios[nextIndex];
  const value = next?.dataset.radioValue as T | undefined;
  if (!next || !value) return;
  onSelect(value);
  next.focus();
}

const SETTINGS_SECTIONS = [
  { id: "computer", label: "This computer", hint: "Local simulator" },
  { id: "performance", label: "Performance", hint: "Loading workers and I/O memory" },
  { id: "compute", label: "Compute nodes", hint: "SSH connections and inputs" },
  { id: "rendering", label: "Remote rendering", hint: "Node desktops" },
  { id: "runs", label: "Files and defaults", hint: "Run storage and templates" },
  { id: "exports", label: "Exports", hint: "Movie encoding" },
  { id: "appearance", label: "Appearance", hint: "Theme, text and layout" },
] as const;
type SettingsSection = typeof SETTINGS_SECTIONS[number]["id"];

/* Settings page. Schema lives in studioSettings so ConfigEditor's spawn path and this editor share one source. */
export function Settings({ simulator: sim }: { simulator: AsyncState<SimulatorStatus> }) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [section, setSection] = useState<SettingsSection>("computer");
  useEffect(() => { if (bodyRef.current) bodyRef.current.scrollTop = 0; }, [section]);
  const [state, setState] = useState<StudioSettings>(() => loadStudioSettings());
  const [templates, setTemplates] = useState<string[]>([]);
  const [templatesLoaded, setTemplatesLoaded] = useState(false);
  const [templateError, setTemplateError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "failed">("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const editGeneration = useRef(0);
  // Last persisted snapshot; without this, unsaved edits are silently discarded on navigation.
  const [persisted, setPersisted] = useState<StudioSettings>(state);
  const dirty = !studioSettingsEqual(state, persisted);
  // Cleared on unmount so navigating away doesn't fire setSaveState on a dead component.
  const saveTimerRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setTemplateError(null);
        const root = await getProjectRoot();
        const list = await listTemplateConfigs(root);
        if (!cancelled) {
          setTemplates(list);
          setTemplatesLoaded(true);
        }
      } catch (e) {
        if (!cancelled) {
          setTemplatesLoaded(true);
          setTemplateError(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const update = useCallback(<K extends keyof StudioSettings>(key: K, value: StudioSettings[K]) => {
    editGeneration.current += 1;
    setState((prev) => ({ ...prev, [key]: value }));
    setSaveState("idle");
  }, []);

  const save = useCallback(async () => {
    const generation = editGeneration.current;
    const normalized = {
      ...state,
      runsDirectory: effectiveRunsDirectory(state.runsDirectory),
    };
    setSaveState("saving");
    setSaveError(null);
    let stored: boolean;
    try { stored = await saveStudioPerformanceSettings(normalized); }
    catch (error) { setSaveError(error instanceof Error ? error.message : String(error)); setSaveState("failed"); return; }
    setSaveState(stored ? generation === editGeneration.current ? "saved" : "idle" : "failed");
    if (!stored) return;
    // Only clear dirty after a successful write; a failed write must keep showing "Unsaved changes".
    if (generation === editGeneration.current) setState(normalized);
    setPersisted(normalized);
    if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(() => setSaveState("idle"), 1500) as unknown as number;
  }, [state]);

  // Surface the probe error explicitly; "Not connected" would hide a valid-path failure.
  const simProbeFailed = !sim.loading && sim.error != null;
  const resolvedBinary = sim.loading
    ? "Checking…"
    : simProbeFailed
      ? sim.error!
      : sim.data?.binaryPath ?? "Not connected";
  const simReady = sim.data?.state === "ready";
  const simStatus = simReady ? "ok" : simProbeFailed ? "warn" : "muted";
  const simStatusText = simReady
    ? "connected"
    : simProbeFailed
      ? "status check failed"
      : "not connected";
  const gpuDevice = sim.data?.gpu ?? null;
  const gpuLabel = gpuDevice ?? "Not detected";
  const selectedTemplateUnavailable = templatesLoaded
    && state.defaultTemplate.length > 0
    && !templates.includes(state.defaultTemplate);

  return (
    <main className="settings">
      <header className="settings__header">
        <h2>Settings</h2>
        <span className="settings__subtitle">
          Choose where work runs and how Studio looks.
        </span>
      </header>

      <div className="settings__layout">
        <nav className="settings__navigation" aria-label="Settings categories">
          {SETTINGS_SECTIONS.map(item => <button type="button" key={item.id}
            aria-label={item.label}
            aria-current={section === item.id ? "page" : undefined}
            aria-controls={`settings-section-${item.id}`}
            onClick={() => setSection(item.id)}><span>{item.label}</span><small>{item.hint}</small></button>)}
        </nav>
        <div className="settings__body" ref={bodyRef}>
        <div id="settings-section-computer" hidden={section !== "computer"}>
        <Group title="Simulator on this computer">
          <ReadOnlyRow
            label="Resolved binary"
            value={resolvedBinary}
            status={simStatus}
            statusText={simStatusText}
          />
          <Field
            label="Binary path override"
            hint="Optional. Connect a Nimbus simulator when this workspace needs to validate or run configurations."
            value={state.binaryPathOverride}
            placeholder="/usr/local/bin/nimbus_sim"
            onChange={(v) => update("binaryPathOverride", v)}
          />
          <SettingsPathCheck kind="simulator" value={state.binaryPathOverride} />
          <ReadOnlyRow
            label="GPU device"
            value={gpuLabel}
            status={gpuDevice ? "ok" : "muted"}
            statusText={gpuDevice ? "detected" : "not detected"}
          />
          <Note>
            <Icon name="info" size={14} />
            <span>
              Acceleration is selected per simulation with the Compute backend
              control. Vulkan and Metal availability is verified against the
              configured simulator before launch; CPU remains the default.
            </span>
          </Note>
        </Group>

        </div>
        <div id="settings-section-performance" hidden={section !== "performance"}>
          {section === "performance" && <PerformanceSettings settings={state} onChange={update} />}
        </div>
        <div id="settings-section-compute" hidden={section !== "compute"}>
          <RemoteComputeSettings />
        </div>
        <div id="settings-section-rendering" hidden={section !== "rendering"}>
          <RemoteWorkspaceSettings visible={section === "rendering"} />
        </div>
        <div id="settings-section-runs" hidden={section !== "runs"}>
        <Group title="Files and defaults">
          <Field
            label="Runs directory"
            hint="Local run outputs and received remote results. Node-side output uses the connection’s Remote work directory."
            value={state.runsDirectory}
            placeholder="~/.nimbus/runs"
            onChange={(v) => update("runsDirectory", v)}
          />
          <SelectField
            label="Default template"
            hint="Selected automatically when you open the Configure page without a recent run."
            value={state.defaultTemplate}
            options={[
              { value: "", label: "None (start from schema defaults)" },
              ...(selectedTemplateUnavailable
                ? [{
                    value: state.defaultTemplate,
                    label: `Unavailable — ${state.defaultTemplate.replace(/\.ya?ml$/, "")}`,
                  }]
                : []),
              ...templates.map((t) => ({ value: t, label: t.replace(/\.ya?ml$/, "") })),
            ]}
            onChange={(v) => update("defaultTemplate", v)}
          />
          {templateError && (
            <div className="settings-note" role="alert">
              <Icon name="warning" size={14} />
              <span>Templates could not be loaded: {templateError}</span>
            </div>
          )}
          {!templateError && selectedTemplateUnavailable && (
            <div className="settings-note" role="status">
              <Icon name="warning" size={14} />
              <span>The saved default template is unavailable. Choose another template or None.</span>
            </div>
          )}
        </Group>

        </div>
        <div id="settings-section-exports" hidden={section !== "exports"}>
        <Group title="Video export on this computer">
          <Field
            label="ffmpeg path"
            hint="Optional path used for MP4 export. Leave empty to use ffmpeg from the system path."
            value={state.ffmpegPathOverride}
            placeholder="/opt/homebrew/bin/ffmpeg"
            onChange={(v) => update("ffmpegPathOverride", v)}
          />
          <Note>
            <Icon name="info" size={14} />
            <span>
              MP4 export uses the saved ffmpeg path when an export begins. GIF export is
              available without additional setup.
            </span>
          </Note>
        </Group>

          <SettingsPathCheck kind="ffmpeg" value={state.ffmpegPathOverride} />
        </div>
        <div id="settings-section-appearance" hidden={section !== "appearance"}>
        <Group title="Appearance">
          <AppearanceField
            label="Theme"
            hint="System follows your operating-system appearance and updates while Studio is open. Scientific color maps are unchanged."
            value={state.appearance}
            onChange={(value) => update("appearance", value)}
          />
          <label className="settings-row">
            <span className="settings-row__label">Text size<span className="settings-row__hint">Comfortable enlarges controls and scientific plot labels.</span></span>
            <select aria-label="Text size" value={state.textSize} onChange={(event) => update("textSize", event.target.value === "comfortable" ? "comfortable" : "standard")}>
              <option value="standard">Standard</option>
              <option value="comfortable">Comfortable</option>
            </select>
          </label>
          <PlotSideField
            label="Plot side"
            hint="Which side of the window the figure occupies in Plot Studio. The navigation rail and run list move to the other side. Reading order is unchanged."
            value={state.plotSide}
            onChange={(value) => update("plotSide", value)}
          />
        </Group>

        </div>
        </div>
      </div>
        <footer className="settings__actions" hidden={!dirty && (section === "compute" || section === "rendering")}>
          <span className="settings__save-scope">App preferences · stored on this computer</span>
          <button
            type="button"
            className="btn btn--primary"
            onClick={save}
            disabled={!dirty || saveState === "saving"}
            // A disabled control must say why it is disabled; studio-navigation
            // asserts this across every page.
            title={saveState === "saving" ? "Applying settings" : dirty ? "Write these settings to local storage" : "No unsaved changes"}
          >
            <span>{saveState === "saved" ? "Saved" : saveState === "saving" ? "Saving…" : "Save settings"}</span>
          </button>
          {saveState === "failed" ? (
            <span className="settings__save-state settings__save-state--failed" role="alert">
              <Icon name="warning" size={12} />
              <span>{saveError ?? "Could not save these settings. Keep this page open and retry."}</span>
            </span>
          ) : (
            dirty && (
              <span className="settings__save-state" role="status">
                Unsaved app preferences. Save before leaving Settings.
              </span>
            )
          )}
          {dirty && <button type="button" className="settings__discard" onClick={() => { editGeneration.current += 1; setState(persisted); setSaveState("idle"); }}>Discard changes</button>}
        </footer>
    </main>
  );
}

function PlotSideField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: StudioPlotSide;
  onChange: (next: StudioPlotSide) => void;
}) {
  const options: Array<{ value: StudioPlotSide; label: string }> = [
    { value: "right", label: "Right" },
    { value: "left", label: "Left" },
  ];
  const labelId = useId();
  const hintId = useId();
  return (
    <div className="settings-row">
      <div className="settings-row__label">
        <span id={labelId}>{label}</span>
        <span className="settings-row__hint" id={hintId}>{hint}</span>
      </div>
      <div className="settings-row__control">
        <div
          className="settings-appearance settings-appearance--two"
          role="radiogroup"
          aria-labelledby={labelId}
          aria-describedby={hintId}
        >
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={value === option.value}
              tabIndex={value === option.value ? 0 : -1}
              data-radio-value={option.value}
              className={value === option.value
                ? "settings-appearance__option settings-appearance__option--active"
                : "settings-appearance__option"}
              onClick={() => onChange(option.value)}
              onKeyDown={(event) => moveRovingRadioFocus(event, onChange)}
              title={`Put the plot on the ${option.label.toLowerCase()}`}
            >
              <span>{option.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function AppearanceField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: StudioAppearance;
  onChange: (next: StudioAppearance) => void;
}) {
  const options: Array<{ value: StudioAppearance; label: string }> = [
    { value: "system", label: "System" },
    { value: "light", label: "Light" },
    { value: "dark", label: "Dark" },
  ];
  const labelId = useId();
  const hintId = useId();
  return (
    <div className="settings-row">
      <div className="settings-row__label">
        <span id={labelId}>{label}</span>
        <span className="settings-row__hint" id={hintId}>{hint}</span>
      </div>
      <div className="settings-row__control">
        <div
          className="settings-appearance"
          role="radiogroup"
          aria-labelledby={labelId}
          aria-describedby={hintId}
        >
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={value === option.value}
              tabIndex={value === option.value ? 0 : -1}
              data-radio-value={option.value}
              className={value === option.value ? "settings-appearance__option settings-appearance__option--active" : "settings-appearance__option"}
              onClick={() => onChange(option.value)}
              onKeyDown={(event) => moveRovingRadioFocus(event, onChange)}
            >
              <span className={`settings-appearance__swatch settings-appearance__swatch--${option.value}`} aria-hidden="true" />
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

interface GroupProps {
  title: string;
  children: React.ReactNode;
}

function Group({ title, children }: GroupProps) {
  return (
    <section className="settings-group">
      <h3 className="settings-group__title">{title}</h3>
      <div className="settings-group__body">{children}</div>
    </section>
  );
}

interface FieldProps {
  label: string;
  hint?: string;
  value: string;
  placeholder?: string;
  onChange: (next: string) => void;
}

function Field({ label, hint, value, placeholder, onChange }: FieldProps) {
  // The visible label sits in a sibling column, so it needs an explicit
  // htmlFor/id pair — otherwise the input is announced with no name.
  const controlId = useId();
  const hintId = useId();
  return (
    <div className="settings-row">
      <div className="settings-row__label">
        <label htmlFor={controlId}>{label}</label>
        {hint && <span className="settings-row__hint" id={hintId}>{hint}</span>}
      </div>
      <div className="settings-row__control">
        <input
          id={controlId}
          type="text"
          value={value}
          placeholder={placeholder}
          spellCheck={false}
          aria-describedby={hint ? hintId : undefined}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    </div>
  );
}

interface SelectFieldProps {
  label: string;
  hint?: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (next: string) => void;
}

function SelectField({ label, hint, value, options, onChange }: SelectFieldProps) {
  const controlId = useId();
  const hintId = useId();
  return (
    <div className="settings-row">
      <div className="settings-row__label">
        <label htmlFor={controlId}>{label}</label>
        {hint && <span className="settings-row__hint" id={hintId}>{hint}</span>}
      </div>
      <div className="settings-row__control">
        <select
          id={controlId}
          value={value}
          aria-describedby={hint ? hintId : undefined}
          onChange={(e) => onChange(e.target.value)}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

interface ReadOnlyRowProps {
  label: string;
  value: string;
  status: "ok" | "warn" | "info" | "muted";
  statusText: string;
}

function ReadOnlyRow({ label, value, status, statusText }: ReadOnlyRowProps) {
  return (
    <div className="settings-row settings-row--readonly">
      <div className="settings-row__label">
        <span>{label}</span>
      </div>
      <div className="settings-row__control">
        <div className="settings-row__value">
          <code>{value}</code>
          <span className={`settings-row__status settings-row__status--${status}`}>
            {statusText}
          </span>
        </div>
      </div>
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <div className="settings-note">{children}</div>;
}
