import { useEffect, useState, useSyncExternalStore } from "react";
import {
  clearParsedFieldCache,
  getResourceTelemetrySnapshot,
  subscribeResourceTelemetry,
  type ResourceTelemetrySnapshot,
} from "@/lib/studio/resourceTelemetry";
import { nativeFieldReadCapabilities, type NativeFieldReadCapabilities } from "@/lib/dataset/nativeFieldWindow";
import { getProcessMemoryInfo, isTauriRuntime } from "@/lib/platform/tauri";
import type { ProcessMemoryInfo } from "@/lib/core/types";
import {
  getVisualizationCapacityQualification,
  resolveVisualizationCapacity,
} from "@/lib/core/visualizationCapacityPolicy";
import "./ResourceBudgetPanel.css";

export interface ResourceBudgetPanelProps {
  onClose?: () => void;
  onClearCache?: (previous: ResourceTelemetrySnapshot) => void;
}

function bytes(value: number): string {
  if (value === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const unit = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  const scaled = value / 1024 ** unit;
  return `${scaled >= 100 || unit === 0 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[unit]}`;
}

function fraction(value: number, limit: number): number {
  return limit <= 0 ? 0 : Math.min(1, value / limit);
}

function cancellationLabel(value: "cooperative" | "boundary_only" | "stale_result_only"): string {
  return value === "cooperative" ? "Cooperative" : value === "boundary_only" ? "Boundary only" : "Stale result only";
}

export function ResourceBudgetPanel({ onClose, onClearCache }: ResourceBudgetPanelProps) {
  const [nativeCapabilities, setNativeCapabilities] = useState<NativeFieldReadCapabilities | null>(null);
  // Separate error state so a failed probe doesn't look like browser mode.
  const [probeError, setProbeError] = useState<string | null>(null);
  const [processMemory, setProcessMemory] = useState<ProcessMemoryInfo | null>(null);
  const desktop = isTauriRuntime();
  useEffect(() => {
    let current = true;
    if (!desktop) return;
    void nativeFieldReadCapabilities().then((capabilities) => {
      if (current) setNativeCapabilities(capabilities);
    }).catch((reason: unknown) => {
      if (current) setProbeError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { current = false; };
  }, [desktop]);
  useEffect(() => {
    if (!desktop) return;
    let current = true;
    let inFlight = false;
    const sample = () => {
      if (inFlight) return;
      inFlight = true;
      void getProcessMemoryInfo().then((memory) => {
        if (current) setProcessMemory(memory);
      }).catch(() => undefined).finally(() => { inFlight = false; });
    };
    sample();
    const timer = window.setInterval(sample, 2_000);
    return () => {
      current = false;
      window.clearInterval(timer);
    };
  }, [desktop]);
  const telemetry = useSyncExternalStore(
    subscribeResourceTelemetry,
    getResourceTelemetrySnapshot,
    getResourceTelemetrySnapshot,
  );
  const { parsedCache: cache, loader, visualization, preparedRegistry, extractionPool } = telemetry;
  const cacheRatio = fraction(cache.bytes, cache.limit);
  const capacityDecision = resolveVisualizationCapacity(
    visualization.hardware,
    getVisualizationCapacityQualification(),
  );

  const clear = () => {
    const previous = telemetry;
    clearParsedFieldCache();
    onClearCache?.(previous);
  };

  return (
    <section className="resource-budget" aria-labelledby="resource-budget-title">
      <header className="resource-budget__header">
        <div>
          <span className="resource-budget__eyebrow">LOCAL RESOURCE LEDGER</span>
          <h3 id="resource-budget-title">Memory &amp; requests</h3>
          <p>Live Studio allocations, updated only when loader state changes.</p>
        </div>
        {onClose && <button className="resource-budget__close" type="button" onClick={onClose}>Close</button>}
      </header>

      <div className="resource-budget__bands">
        <section className="resource-budget__band" aria-labelledby="parsed-cache-heading">
          <div className="resource-budget__band-heading">
            <div><span>01 · PARSED CACHE</span><h4 id="parsed-cache-heading">Resident field arrays</h4></div>
            <strong>{Math.round(cacheRatio * 100)}%</strong>
          </div>
          <meter min={0} max={Math.max(1, cache.limit)} value={Math.min(cache.bytes, Math.max(1, cache.limit))} aria-label="Parsed cache use" />
          <div className="resource-budget__measure"><strong>{bytes(cache.bytes)}</strong><span>of {bytes(cache.limit)}</span></div>
          <dl className="resource-budget__ledger">
            <div><dt>Entries</dt><dd>{cache.entries}</dd></div>
            <div><dt>Loads pending</dt><dd>{cache.pendingLoads}</dd></div>
            <div><dt>Hits</dt><dd>{cache.hits}</dd></div>
            <div><dt>Evictions</dt><dd>{cache.evictions}</dd></div>
          </dl>
          <button className="resource-budget__clear" type="button" onClick={clear} disabled={cache.entries === 0}>
            Release parsed cache
          </button>
          <p className="resource-budget__boundary">{telemetry.boundary.parsedCache} Clearing does not interrupt reads already in progress.</p>
        </section>

        <section className="resource-budget__band" aria-labelledby="loader-budget-heading">
          <div className="resource-budget__band-heading">
            <div><span>02 · LOADER BUDGET</span><h4 id="loader-budget-heading">Reserved decode work</h4></div>
            <strong>{loader.active} / {loader.maxConcurrent}</strong>
          </div>
          <meter min={0} max={Math.max(1, loader.maxBytes)} value={Math.min(loader.activeBytes, Math.max(1, loader.maxBytes))} aria-label="Active loader byte reservations" />
          <div className="resource-budget__measure"><strong>{bytes(loader.activeBytes)}</strong><span>of {bytes(loader.maxBytes)} reserved</span></div>
          <dl className="resource-budget__ledger resource-budget__ledger--loader">
            <div><dt>Active</dt><dd>{loader.active}</dd></div>
            <div><dt>Queued</dt><dd>{loader.queued}</dd></div>
            <div><dt>Queued estimate</dt><dd>{bytes(loader.queuedBytes)}</dd></div>
            <div><dt>Completed</dt><dd>{loader.completed}</dd></div>
            <div><dt>Superseded</dt><dd>{loader.superseded}</dd></div>
            <div><dt>Cancelled</dt><dd>{loader.cancelled}</dd></div>
            <div><dt>Failures</dt><dd>{loader.failures}</dd></div>
          </dl>
          <p className="resource-budget__boundary">{telemetry.boundary.loaderBudget} {telemetry.boundary.nativeIpc}</p>
        </section>
        <section className="resource-budget__band" aria-labelledby="visualization-budget-heading">
          <div className="resource-budget__band-heading">
            <div><span>03 · VISUALIZATION</span><h4 id="visualization-budget-heading">VTK ownership</h4></div>
            <strong>{visualization.reservations} live</strong>
          </div>
          <div className="resource-budget__measure resource-budget__measure--visualization">
            <strong>{bytes(visualization.committedBytes)}</strong>
            <span>{bytes(visualization.residentBytes)} resident + {bytes(visualization.reservedBytes)} reserved · {bytes(visualization.highWaterBytes)} high water</span>
          </div>
          <dl className="resource-budget__ledger resource-budget__ledger--visualization">
            <div><dt>Decoded cache</dt><dd>{bytes(visualization.byStage["decoded-field"].residentBytes)}</dd></div>
            <div><dt>Prepared</dt><dd>{bytes(visualization.byStage["prepared-main"].residentBytes)}</dd></div>
            <div><dt>Prepared handles</dt><dd>{preparedRegistry.handles} / {preparedRegistry.references} refs</dd></div>
            <div><dt>Handle residency</dt><dd>{preparedRegistry.mainHandles} main / {preparedRegistry.workerHandles} worker</dd></div>
            <div><dt>Warm extraction</dt><dd>{extractionPool.warmServices} / {extractionPool.maximumWarmServices}</dd></div>
            <div><dt>Idle evictions</dt><dd>{extractionPool.evictions}</dd></div>
            <div><dt>Worker resident</dt><dd>{bytes(visualization.byStage["worker-resident"].residentBytes)}</dd></div>
            <div><dt>Geometry</dt><dd>{bytes(visualization.byStage["retained-geometry"].residentBytes)}</dd></div>
            <div><dt>Native read</dt><dd>{bytes(visualization.byStage["native-read"].reservedBytes)}</dd></div>
            <div><dt>Handoff</dt><dd>{bytes(visualization.byStage.handoff.reservedBytes)}</dd></div>
            <div><dt>Known scratch high</dt><dd>{bytes(visualization.byStage["worker-scratch"].reservedHighWaterBytes)}</dd></div>
            <div><dt>Predicted / actual</dt><dd>{bytes(visualization.prediction.predictedBytes)} / {bytes(visualization.prediction.actualBytes)}</dd></div>
            <div><dt>Underpredictions</dt><dd>{visualization.prediction.underpredictions}</dd></div>
            <div><dt>System RAM</dt><dd>{visualization.hardware.systemRamBytes == null ? "Unavailable" : bytes(visualization.hardware.systemRamBytes)}</dd></div>
            <div><dt>Logical cores</dt><dd>{visualization.hardware.logicalCores ?? "Unavailable"}</dd></div>
            <div><dt>Host RSS</dt><dd>{processMemory?.currentRssBytes == null ? "Unavailable" : bytes(processMemory.currentRssBytes)}</dd></div>
            <div><dt>Host RSS high</dt><dd>{processMemory?.peakRssBytes == null ? "Unavailable" : bytes(processMemory.peakRssBytes)}</dd></div>
            <div><dt>Process tree RSS</dt><dd>{processMemory?.processTreeRssBytes == null ? "Unavailable" : bytes(processMemory.processTreeRssBytes)}</dd></div>
            <div><dt>Process tree high</dt><dd>{processMemory?.processTreePeakRssBytes == null ? "Unavailable" : bytes(processMemory.processTreePeakRssBytes)}</dd></div>
            <div><dt>Tree processes</dt><dd>{processMemory?.processTreeProcessCount ?? "Unavailable"}</dd></div>
            <div><dt>Admission</dt><dd>{capacityDecision.mode === "enforced" ? "Enforced" : "Observe only"}</dd></div>
            <div><dt>Evidence floor</dt><dd>{bytes(capacityDecision.observedFloorBytes)}</dd></div>
          </dl>
          <p className="resource-budget__boundary">
            {telemetry.boundary.visualization}
            {processMemory
              ? processMemory.processTreeCoverage === "whole_application"
                ? ` Tauri host measurement: ${processMemory.measurement}. Whole-application coverage uses the macOS resource coalition, including associated WebKit/XPC services.`
                : ` Tauri host measurement: ${processMemory.measurement}. Process-tree coverage: ${processMemory.processTreeCoverage}; detached XPC or webview helpers may not be descendants and remain excluded.`
              : ""}
            {capacityDecision.blockers.length > 0
              ? ` Capacity enforcement remains disabled: ${capacityDecision.blockers.join("; ")}.`
              : ""}
          </p>
        </section>
      </div>
      <section className="resource-budget__native" aria-labelledby="native-read-heading">
        <div className="resource-budget__band-heading">
          <div><span>04 · NATIVE READ PATHS</span><h4 id="native-read-heading">Cancellation &amp; resolution contract</h4></div>
          <strong>
            {nativeCapabilities
              ? `${bytes(nativeCapabilities.maxWindowBytes)} cap`
              : !desktop
                ? "Desktop only"
                : probeError
                  ? "Unavailable"
                  : "Reading…"}
          </strong>
        </div>
        {nativeCapabilities ? <dl className="resource-budget__native-grid">
          <div><dt>NPY windows</dt><dd>{cancellationLabel(nativeCapabilities.npy.cancellation)}</dd><span>{nativeCapabilities.npy.limitation ?? "Spatial origin/extent/stride; cancelled at row boundaries."}</span></div>
          <div><dt>ZFP windows</dt><dd>{cancellationLabel(nativeCapabilities.zfp3d.cancellation)}</dd><span>{nativeCapabilities.zfp3d.limitation}</span></div>
          <div><dt>Classic netCDF</dt><dd>{cancellationLabel(nativeCapabilities.netcdfClassic.cancellation)}</dd><span>{nativeCapabilities.netcdfClassic.limitation}</span></div>
          <div><dt>netCDF4</dt><dd>{cancellationLabel(nativeCapabilities.netcdf4.cancellation)}</dd><span>{nativeCapabilities.netcdf4.limitation}</span></div>
        </dl> : !desktop ? (
          <p className="resource-budget__boundary">Native read capabilities are reported by the packaged desktop runtime. Browser mode retains the cache and scheduler ledger but cannot issue filesystem reads.</p>
        ) : probeError ? (
          <p className="resource-budget__boundary resource-budget__boundary--error" role="status">
            The desktop runtime did not report its native read capabilities, so the contract below is
            unknown — reads may still work. {probeError}
          </p>
        ) : (
          <p className="resource-budget__boundary">Reading native capabilities from the desktop runtime…</p>
        )}
        {nativeCapabilities && <p className="resource-budget__boundary">{nativeCapabilities.geometryLimitation}</p>}
      </section>
    </section>
  );
}
