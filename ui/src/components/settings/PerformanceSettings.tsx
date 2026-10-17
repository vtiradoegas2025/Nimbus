import { useEffect, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import type { StudioSettings } from "@/lib/studio/studioSettings";
import { cancelLoadingWorkerMeasurement, measureLoadingWorkers, refreshNativeIoBudget,
  studioPerformanceSnapshot, subscribeStudioPerformance } from "@/lib/studio/studioPerformance";
import "./PerformanceSettings.css";

const MIB = 1024 ** 2;
export function PerformanceSettings({ settings, onChange }: {
  settings: StudioSettings;
  onChange: <K extends keyof StudioSettings>(key: K, value: StudioSettings[K]) => void;
}) {
  const performance = useSyncExternalStore(subscribeStudioPerformance, studioPerformanceSnapshot);
  useEffect(() => {
    void refreshNativeIoBudget().catch(() => undefined);
    const timer = setInterval(() => { void refreshNativeIoBudget().catch(() => undefined); }, 1000);
    return () => clearInterval(timer);
  }, []);
  const count = Math.min(64, Math.max(1, performance.hardware?.coreCount ?? 8));
  const counts = Array.from({ length: count }, (_, index) => index + 1);
  if (settings.loadingWorkers > count) counts.push(settings.loadingWorkers);
  const io = performance.io;
  const running = performance.phase !== "idle";
  return <section className="settings-group settings-performance">
    <h3 className="settings-group__title">Loading and 3-D data</h3>
    <div className="settings-group__body">
      <label className="settings-row">
        <span className="settings-row__label">Loading workers<span className="settings-row__hint">Concurrent field reads and decoding. The memory limit can reduce how many run together.</span></span>
        <select aria-label="Loading workers" value={settings.loadingWorkers} onChange={event => onChange("loadingWorkers", Number(event.target.value))}>
          <option value={0}>Automatic{performance.profile ? ` (measured: ${performance.profile.recommendedWorkers})` : " (balanced)"}</option>
          {counts.map(workers => <option key={workers} value={workers}>{workers}</option>)}
        </select>
      </label>
      <label className="settings-row">
        <span className="settings-row__label">I/O memory limit (MiB)<span className="settings-row__hint">Use 0 for automatic. This limits temporary field-reading memory; plot caches and GPU memory are separate. Allowed manual range: 64–{Math.floor((io?.maximumBytes ?? 16384 * MIB) / MIB).toLocaleString()} MiB.</span></span>
        <input aria-label="I/O memory limit (MiB)" type="number" min={0} max={Math.floor((io?.maximumBytes ?? 16384 * MIB) / MIB)} step={1}
          value={settings.ioBudgetMiB} onChange={event => onChange("ioBudgetMiB", Number(event.target.value))} />
      </label>
      <p className="settings-performance__status" aria-live="polite">
        Active limits: {performance.budget?.maxConcurrentLoads ?? "—"} loading workers,
        {" "}{io ? Math.round(io.limitBytes / MIB).toLocaleString() : "—"} MiB for I/O.
        {performance.native && io && <> Currently reserved: {Math.ceil(io.reservedBytes / MIB).toLocaleString()} MiB; {io.waitingReads} reads waiting.</>}
      </p>
      {performance.configError && <p role="alert">{performance.configError}</p>}
      <label className="settings-performance__check"><input type="checkbox" checked={settings.autoProfileWorkers}
        onChange={event => onChange("autoProfileWorkers", event.target.checked)} />Measure automatically on a new computer</label>
      <div className="settings-performance__measure">
        <p>A short local test compares cached NPY reads and scanning with different worker counts. Automatic uses the smallest count close to the best measured result. Storage, file format and other applications can change the result; rerun while this computer is idle.</p>
        <div className="settings-performance__actions">
          <button type="button" className="btn" disabled={!performance.native || running}
            title={!performance.native ? "Available in the native desktop application" : running ? "A measurement is already running" : "Measure loading workers on this computer"}
            onClick={() => void measureLoadingWorkers()}>Measure this computer</button>
          {running && <button type="button" className="btn" disabled={performance.phase === "cancelling"}
            title={performance.phase === "cancelling" ? "Waiting for active read work to stop" : "Stop this measurement"}
            onClick={() => void cancelLoadingWorkerMeasurement()}>{performance.phase === "cancelling" ? "Stopping…" : "Stop measurement"}</button>}
        </div>
        {running && <p role="status">{performance.phase === "cancelling" ? "Waiting for the current read to stop…" : `Testing ${performance.status?.workers || 1} workers · ${performance.status?.completed ?? 0} of ${performance.status?.total || "…"} choices measured`}</p>}
        {performance.error && <p role="alert">{performance.error}</p>}
        {!performance.profileSaved && <p role="alert">The measurement is active for this session but could not be saved.</p>}
        {performance.profile && <details>
          <summary>Measured loading results · recommendation: {performance.profile.recommendedWorkers} workers</summary>
          <table><thead><tr><th>Workers</th><th>Median time</th><th>Cached read / scan</th></tr></thead><tbody>
            {performance.profile.trials.map(trial => <tr key={trial.workers}><td>{trial.workers}</td><td>{trial.medianMs.toFixed(1)} ms</td><td>{trial.throughputMibS.toFixed(0)} MiB/s</td></tr>)}
          </tbody></table>
          <p>Each choice uses a warmup and three equal-work trials. The recommendation applies when Loading workers is Automatic. These measurements do not select simulator threads or a GPU backend.</p>
        </details>}
      </div>
    </div>
  </section>;
}

export function WorkerMeasurementNotice() {
  const performance = useSyncExternalStore(subscribeStudioPerformance, studioPerformanceSnapshot);
  if (performance.phase === "idle" || typeof document === "undefined") return null;
  return createPortal(<aside className="worker-measurement" role="status" aria-live="polite">
    <div><strong>{performance.phase === "cancelling" ? "Stopping loading measurement…" : "Measuring loading workers"}</strong>
      <p>Studio is checking this computer with a small local dataset. You can keep using the app. Details are in Settings → Performance.</p></div>
    <button type="button" disabled={performance.phase === "cancelling"}
      title={performance.phase === "cancelling" ? "Waiting for active read work to stop" : "Stop the loading-worker measurement"}
      onClick={() => void cancelLoadingWorkerMeasurement()}>Stop</button>
  </aside>, document.body);
}
