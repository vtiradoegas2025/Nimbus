import { useMemo, useState } from "react";
import {
  MAX_EXTERNAL_DERIVED_PROVIDERS,
  importExternalDerivedProvider,
  parseExternalDerivedProvider,
  setExternalDerivedProviderEnabled,
  type ExternalDerivedProvider,
  type ExternalProviderRuntimeStatus,
} from "@/lib/derived/externalDerivedProviders";
import "./ExternalDerivedProviderManager.css";

const PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION = "0.26.4";

export interface ExternalDerivedProviderManagerProps {
  providers: readonly ExternalDerivedProvider[];
  onProvidersChange: (providers: readonly ExternalDerivedProvider[]) => void | Promise<void>;
  runtimeStatus?: (provider: ExternalDerivedProvider) => ExternalProviderRuntimeStatus;
}

function defaultRuntimeStatus(provider: ExternalDerivedProvider): ExternalProviderRuntimeStatus {
  if (provider.kind === "python") {
    const matching = provider.entrypoint.runtimeVersion === PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION;
    return {
      available: matching,
      label: `Pyodide ${provider.entrypoint.runtimeVersion}`,
      reason: matching ? "Loads on demand; execution still requires explicit permission grants." : `Studio ships Pyodide ${PYODIDE_DERIVED_PROVIDER_RUNTIME_VERSION}.`,
    };
  }
  return {
    available: false,
    label: `Plugin host ${provider.entrypoint.hostVersion}`,
    reason: "No native plugin-host adapter is connected. The manifest remains registered but cannot execute.",
  };
}

function providerRuntime(provider: ExternalDerivedProvider): string {
  return provider.kind === "python"
    ? `Pyodide ${provider.entrypoint.runtimeVersion}`
    : `Native host ${provider.entrypoint.hostVersion}`;
}

export function ExternalDerivedProviderManager({
  providers,
  onProvidersChange,
  runtimeStatus = defaultRuntimeStatus,
}: ExternalDerivedProviderManagerProps) {
  const [manifestText, setManifestText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const ids = useMemo(() => new Set(providers.map(({ id }) => id)), [providers]);
  const outputIds = useMemo(() => new Set(providers.map(({ output }) => output.fieldId)), [providers]);

  const commit = async (next: readonly ExternalDerivedProvider[], busy: string, success: string) => {
    setBusyId(busy);
    setError(null);
    setNotice(null);
    try {
      await onProvidersChange(next);
      setNotice(success);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyId(null);
    }
  };

  const importManifest = async () => {
    if (busyId) return;
    try {
      const provider = importExternalDerivedProvider(manifestText);
      if (providers.length >= MAX_EXTERNAL_DERIVED_PROVIDERS) throw new Error(`Provider registry is full (${MAX_EXTERNAL_DERIVED_PROVIDERS} maximum)`);
      if (ids.has(provider.id)) throw new Error(`Provider '${provider.id}' is already registered`);
      if (outputIds.has(provider.output.fieldId)) throw new Error(`Output field '${provider.output.fieldId}' is already supplied by another external provider`);
      await commit([...providers, provider], "import", `${provider.label} was imported disabled. Review its permissions before enabling.`);
      setManifestText("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setNotice(null);
    }
  };

  const toggle = async (provider: ExternalDerivedProvider) => {
    if (busyId) return;
    const nextEnabled = !provider.enabled;
    const next = setExternalDerivedProviderEnabled(providers, provider.id, nextEnabled);
    await commit(next, provider.id, `${provider.label} is ${nextEnabled ? "enabled in the field catalog" : "disabled"}.`);
  };

  const remove = async (provider: ExternalDerivedProvider) => {
    if (busyId) return;
    await commit(providers.filter(({ id }) => id !== provider.id), provider.id, `${provider.label} was removed.`);
  };

  return <section className="provider-console" aria-labelledby="provider-console-title">
    <header className="provider-console__masthead">
      <div>
        <span className="provider-console__eyebrow">Trusted computation boundary</span>
        <h3 id="provider-console-title">Derived-field providers</h3>
        <p>{providers.length}/{MAX_EXTERNAL_DERIVED_PROVIDERS} registered · imported manifests start disabled</p>
      </div>
      <span className="provider-console__schema">contract 1.0</span>
    </header>

    <div className="provider-console__ruler" aria-label="Provider safety model">
      <span><b>01</b> Pin runtime</span>
      <span><b>02</b> Verify artifact</span>
      <span><b>03</b> Grant permissions</span>
      <span><b>04</b> Validate output</span>
    </div>

    <div className="provider-console__ledger" aria-label="Registered external providers">
      {providers.length === 0 ? <div className="provider-console__empty">
        <strong>No external computation is registered.</strong>
        <p>Import a versioned manifest to add a Python field recipe or reserve a native plugin contract. Nothing executes on import.</p>
      </div> : providers.map((provider) => {
        const status = runtimeStatus(parseExternalDerivedProvider(provider));
        const canEnable = status.available || provider.enabled;
        return <article className="provider-console__row" key={provider.id}>
          <div className={`provider-console__lamp provider-console__lamp--${provider.enabled && status.available ? "ready" : status.available ? "standby" : "offline"}`} aria-hidden="true" />
          <div className="provider-console__identity">
            <strong>{provider.label}</strong>
            <code>{provider.id}@{provider.providerVersion}</code>
          </div>
          <div className="provider-console__output">
            <span>Output</span>
            <strong>{provider.output.fieldId} <small>[{provider.output.unit}]</small></strong>
            <code>{provider.output.category} · {provider.output.shapePolicy.kind}</code>
          </div>
          <div className="provider-console__runtime">
            <span>{providerRuntime(provider)}</span>
            <strong className={status.available ? "provider-console__status--ready" : "provider-console__status--offline"}>{status.available ? "Runtime qualified" : "Runtime unavailable"}</strong>
            <small>{status.reason}</small>
          </div>
          <details className="provider-console__review">
            <summary>Review</summary>
            <dl>
              <div><dt>Inputs</dt><dd>{provider.inputs.map(({ fieldId, unit }) => `${fieldId}${unit ? ` [${unit}]` : ""}`).join(", ")}</dd></div>
              <div><dt>Permissions</dt><dd>{provider.permissions.join(", ")}</dd></div>
              <div><dt>Capabilities</dt><dd>{provider.capabilities.join(", ")}</dd></div>
              <div><dt>Integrity</dt><dd><code>{provider.kind === "python" ? provider.entrypoint.sourceDigest : provider.entrypoint.artifactDigest}</code></dd></div>
            </dl>
            <p>{provider.description}</p>
          </details>
          <div className="provider-console__actions">
            <button type="button" className={provider.enabled ? "provider-console__disable" : "provider-console__enable"} disabled={Boolean(busyId) || !canEnable} title={!canEnable ? status.reason ?? undefined : undefined} onClick={() => void toggle(provider)}>
              {busyId === provider.id ? "Applying…" : provider.enabled ? "Disable" : "Enable"}
            </button>
            <button type="button" className="provider-console__remove" disabled={Boolean(busyId)} onClick={() => void remove(provider)}>Remove</button>
          </div>
        </article>;
      })}
    </div>

    <details className="provider-console__import">
      <summary>Import provider manifest</summary>
      <div className="provider-console__import-body">
        <label htmlFor="provider-manifest-json">Versioned JSON manifest</label>
        <textarea id="provider-manifest-json" rows={8} spellCheck={false} value={manifestText} onChange={(event) => setManifestText(event.target.value)} placeholder={'{\n  "schema": "nimbus.studio.external-derived-provider",\n  "schemaVersion": 1,\n  …\n}'} />
        <div>
          <p>Unknown properties, version ranges, undeclared permissions, shape ambiguity, and malformed integrity hashes are rejected.</p>
          <button type="button" disabled={!manifestText.trim() || Boolean(busyId)} onClick={() => void importManifest()}>{busyId === "import" ? "Validating…" : "Validate and register"}</button>
        </div>
      </div>
    </details>

    {(error || notice) && <p className={`provider-console__message ${error ? "provider-console__message--error" : ""}`} role={error ? "alert" : "status"}>{error ?? notice}</p>}
    <footer className="provider-console__boundary">Enabled means catalog-visible. Execution still requires explicit permission grants; Python runs only after sha256 verification. Native plugins remain unavailable until a host adapter is connected.</footer>
  </section>;
}
