import { useCallback, useEffect, useMemo, useState } from "react";
import { useTicker1Hz, useVelMaxHistory, type VelMaxSample } from "@/lib/core/hooks";
import type { QueuedRun } from "@/lib/core/types";
import {
  abortSim,
  receiveRemoteResults,
  cancelRemoteReceive,
  copyRunProvenance,
  openPath,
  pickProvenanceSavePath,
} from "@/lib/platform/tauri";
import { REMOTE_RECEIVE_CHANGE_EVENT, type RemoteReceiveSummary } from "@/lib/platform/remoteCompute";
import { Icon } from "@/components/shell/Icon";
import { LogStream } from "./LogStream";
import { processExitSummary, runFailureSummary, runPlotBlockReason } from "@/lib/run/runOutcome";
import "./RunDetail.css";

interface RunDetailProps {
  run: QueuedRun | null;
  /** Switches the active page to Plots. Surfaced as a "View plots"
   *  button when the selected run has outputs ready. */
  goToPlots?: () => void;
}

/* Run-detail panel: progress + stats header and live log. Ticks elapsed time locally at 1 Hz to avoid IPC polling. */
export function RunDetail({ run, goToPlots }: RunDetailProps) {
  return <RunDetailSession key={run?.id ?? "empty"} run={run} goToPlots={goToPlots} />;
}

function RunDetailSession({ run, goToPlots }: RunDetailProps) {
  const [receiving, setReceiving] = useState(false);
  const [receiveResult, setReceiveResult] = useState<RemoteReceiveSummary | null>(null);
  const [receiveError, setReceiveError] = useState<string | null>(null);
  const [abortState, setAbortState] = useState<"idle" | "aborting" | "error">("idle");
  const [abortError, setAbortError] = useState<string | null>(null);
  const [openDirError, setOpenDirError] = useState<string | null>(null);
  const [provenanceExport, setProvenanceExport] = useState<
    { kind: "idle" } | { kind: "saving" } | { kind: "saved"; path: string } | { kind: "error"; message: string }
  >({ kind: "idle" });

  const isRunning = run?.status === "running";
  const receive = receiveResult && (!run?.remoteReceive || receiveResult.updatedUnixMs > run.remoteReceive.updatedUnixMs) ? receiveResult : run?.remoteReceive;
  const transferActive = receiving || run?.remoteTransferInProgress || run?.remoteReceive?.state === "receiving";
  const receiveChanged = () => window.dispatchEvent(new CustomEvent(REMOTE_RECEIVE_CHANGE_EVENT));
  const onReceive = async () => {
    if (!run?.outputDir || transferActive) return;
    setReceiving(true); setReceiveError(null); receiveChanged();
    try { const result = await receiveRemoteResults(run.outputDir); setReceiveResult(result); }
    catch (error) { setReceiveError(error instanceof Error ? error.message : String(error)); }
    finally { setReceiving(false); receiveChanged(); }
  };
  const onCancelReceive = async () => {
    if (!run?.outputDir) return;
    setReceiveError(null);
    try { await cancelRemoteReceive(run.outputDir); }
    catch (error) { setReceiveError(error instanceof Error ? error.message : String(error)); }
    receiveChanged();
  };
  // Frozen on last value for completed/failed/aborted so the counter doesn't keep climbing.
  const now = useTicker1Hz(isRunning);
  const velHistory = useVelMaxHistory(run?.origin === "disk" ? null : (run?.id ?? null));

  const onAbort = useCallback(async () => {
    if (!run || abortState === "aborting") return;
    setAbortState("aborting");
    setAbortError(null);
    try {
      // abort_sim returns after SIGINT is sent; stay in "aborting" until status leaves "running".
      await abortSim(run.id);
    } catch (e) {
      setAbortState("error");
      setAbortError(e instanceof Error ? e.message : String(e));
    }
  }, [run, abortState]);

  // The run:exit event flips status once the child is reaped; that is the only
  // honest signal that aborting finished.
  useEffect(() => {
    if (!isRunning || run?.remoteTerminationUnconfirmed) setAbortState((state) => (state === "aborting" ? "idle" : state));
  }, [isRunning, run?.remoteTerminationUnconfirmed]);

  const onOpenDir = useCallback(async () => {
    if (!run?.outputDir) return;
    setOpenDirError(null);
    try {
      await openPath(run.outputDir);
    } catch (e) {
      setOpenDirError(e instanceof Error ? e.message : String(e));
    }
  }, [run]);

  const onExportProvenance = useCallback(async () => {
    if (
      !run?.outputDir ||
      run.provenance?.health !== "verified" ||
      run.provenance.runtimeHealth !== "verified"
    ) return;
    setProvenanceExport({ kind: "saving" });
    try {
      const destination = await pickProvenanceSavePath(
        `${run.caseName.replace(/[^a-zA-Z0-9._-]+/g, "-")}-run-evidence.json`,
      );
      if (!destination) {
        setProvenanceExport({ kind: "idle" });
        return;
      }
      const path = await copyRunProvenance(run.outputDir, destination);
      setProvenanceExport({ kind: "saved", path });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : error && typeof error === "object" && "message" in error && typeof error.message === "string"
            ? error.message
            : String(error);
      setProvenanceExport({ kind: "error", message });
    }
  }, [run]);

  // All useMemo hooks above the early return: hook count must be stable when `run` flips to null.

  // Wall-clock elapsed: prefer frozen endedAtMs; fall back to ticking `now`.
  const elapsedS = useMemo<number | null>(() => {
    if (!run || run.startedAtMs == null) return null;
    const endMs = run.endedAtMs ?? now;
    return Math.max(0, (endMs - run.startedAtMs) / 1000);
  }, [run, now]);

  // Sim-seconds per wall-second; suppressed for the first 3 s to avoid startup transient noise.
  const simRate = useMemo<number | null>(() => {
    if (!run || run.remoteTerminationUnconfirmed || run.remoteTransferInProgress || run.simTimeS == null || elapsedS == null) return null;
    if (elapsedS < 3) return null;
    if (run.simTimeS <= 0) return null;
    return run.simTimeS / elapsedS;
  }, [run, elapsedS]);

  // ETA while running with a positive sim rate.
  const etaS = useMemo<number | null>(() => {
    if (!run || !isRunning) return null;
    if (simRate == null || simRate <= 0) return null;
    if (run.simTimeS == null) return null;
    const remaining = Math.max(0, run.durationS - run.simTimeS);
    return remaining / simRate;
  }, [run, isRunning, simRate]);

  if (!run) {
    return (
      <main className="run-detail run-detail--empty">
        <div className="run-detail__placeholder">
          <Icon name="archive" size={20} />
          <h2>No run selected</h2>
          <p>Pick a run from the queue on the left to view its progress and live log.</p>
        </div>
      </main>
    );
  }

  const progress =
    run.simTimeS != null && run.durationS > 0
      ? Math.min(1, run.simTimeS / run.durationS)
      : 0;
  const pct = (progress * 100).toFixed(1);
  const canAbort = !transferActive && (run.status === "running" || run.status === "queued");
  const displayedRun = { ...run, remoteReceive: receive ?? undefined, remoteTransferInProgress: transferActive };
  const plotBlockReason = runPlotBlockReason(displayedRun, import.meta.env.DEV);
  const failure = runFailureSummary(displayedRun);

  return (
    <main className="run-detail">
      <header className="run-detail__header">
        <div className="run-detail__title-row">
          <h2>{run.caseName}</h2>
          {run.origin === "disk" && run.libraryHealth ? (
            <LibraryHealthPill health={run.libraryHealth} />
          ) : (
            <StatusPill status={run.status} />
          )}
          <div className="run-detail__title-spacer" />
          {canAbort && (
            <button
              type="button"
              className="run-detail__abort"
              onClick={onAbort}
              disabled={abortState === "aborting"}
              title={run.remoteTerminationUnconfirmed ? "Reconnect and confirm the remote simulation has stopped" : run.remoteTransferInProgress ? "Cancel receiving results and wait for the transfer to stop" : "Stop the running simulator and confirm termination"}
            >
              <Icon name="stop" size={12} />
              <span>{abortState === "aborting" ? run.remoteTransferInProgress ? "Cancelling…" : "Aborting…" : run.remoteTerminationUnconfirmed ? "Retry stop" : run.remoteTransferInProgress ? "Cancel transfer" : "Abort"}</span>
            </button>
          )}
          {transferActive && <button type="button" className="run-detail__abort" onClick={() => void onCancelReceive()}
            title="Cancel receiving files without stopping or rerunning the completed remote computation">Cancel transfer</button>}
          {run.outputDir && (
            <button
              type="button"
              className="run-detail__open-dir"
              onClick={onOpenDir}
              title={`Reveal output directory: ${run.outputDir}`}
            >
              <Icon name="folder" size={12} />
              <span>Open dir</span>
            </button>
          )}
          {run.outputDir && goToPlots && (
            <button
              type="button"
              className="run-detail__view-plots"
              onClick={goToPlots}
              disabled={plotBlockReason !== null}
              title={plotBlockReason ?? `Open this run in Plot Studio (${run.outputDir})`}
            >
              <Icon name="plot" size={12} />
              <span>View plots</span>
            </button>
          )}
        </div>
        <div className="run-detail__meta">
          <span className="run-detail__id">{run.id}</span>
          <span className="run-detail__sep">·</span>
          <span>{run.scheme}</span>
          {run.executionTarget && <><span className="run-detail__sep">·</span><span title={run.executionTarget.host}>{run.executionTarget.name} · {run.executionTarget.host}</span></>}
          {run.step != null && (
            <>
              <span className="run-detail__sep">·</span>
              <span>step {run.step.toLocaleString()}</span>
            </>
          )}
          {abortError && (
            <>
              <span className="run-detail__sep">·</span>
              <span className="run-detail__abort-error">{abortError}</span>
            </>
          )}
          {openDirError && (
            <>
              <span className="run-detail__sep">·</span>
              <span className="run-detail__abort-error">{openDirError}</span>
            </>
          )}
        </div>
      </header>

      {run.outputDir && goToPlots && plotBlockReason && <p className="run-detail__plot-notice" role="status">{plotBlockReason}</p>}

      {failure && <div className="run-detail__library-notice run-detail__library-notice--invalid" role="alert">
        <Icon name="warning" size={16} /><div><strong>{failure.title}</strong><span>{failure.detail}</span></div>
      </div>}

      {run.provenance && (
        <details
          className={`run-detail__provenance run-detail__provenance--${run.provenance.health}`}
          open={run.provenance.health === "invalid" || run.provenance.runtimeHealth === "invalid" ? true : undefined}
        >
          <summary>
            <span>Run evidence</span>
            <strong>{`launch ${run.provenance.health} · runtime ${run.provenance.runtimeHealth}`}</strong>
            <span className="run-detail__provenance-summary">
              {run.provenance.health === "verified"
                ? "Requested launch controls and observed runtime completion"
                : run.provenance.health === "missing"
                  ? "No Studio provenance record"
                  : "Record failed validation"}
            </span>
          </summary>
          <div className="run-detail__provenance-body">
            {run.provenance.health === "verified" ? (
              <>
                <dl className="run-detail__provenance-facts">
                  <div><dt>Run identity</dt><dd>{run.provenance.runId ?? "Unavailable"}</dd></div>
                  <div><dt>Created</dt><dd>{fmtDate(run.provenance.createdUnixMs)}</dd></div>
                  <div><dt>Requested policy</dt><dd>{run.provenance.requestedExecutionMode ?? "Unavailable"}</dd></div>
                  <div><dt>Requested backend</dt><dd>{run.provenance.requestedBackend ?? "Unavailable"}</dd></div>
                  <div><dt>Requested output</dt><dd>{run.provenance.requestedOutputFormat ?? "Unavailable"}</dd></div>
                  <div><dt>Requested validation</dt><dd>{run.provenance.requestedValidationMode ?? "Unavailable"}</dd></div>
                  <div><dt>Actual status</dt><dd>{run.provenance.finalStatus ?? "Unavailable"}</dd></div>
                  <div><dt>Process exit</dt><dd>{processExitSummary(run.provenance.exitCode, !!run.executionTarget)}</dd></div>
                  <div><dt>Process started</dt><dd>{fmtDate(run.provenance.processStartedUnixMs)}</dd></div>
                  <div><dt>Process ended</dt><dd>{fmtDate(run.provenance.processEndedUnixMs)}</dd></div>
                  <div><dt>Complete frames</dt><dd>{run.provenance.completeFrameCount ?? "Unavailable"}</dd></div>
                  <div><dt>Verified checkpoints</dt><dd>{run.provenance.verifiedCheckpointCount ?? "Unavailable"}</dd></div>
                </dl>
                <div className="run-detail__provenance-comparison" role="table" aria-label="Requested and observed runtime controls">
                  <div role="row"><span role="columnheader">Control</span><span role="columnheader">Requested</span><span role="columnheader">Observed runtime</span></div>
                  <div role="row"><span role="cell">Compute backend</span><code role="cell">{run.provenance.requestedBackend ?? "Unavailable"}</code><code role="cell">{run.provenance.actualBackend ?? "Unavailable"}</code></div>
                  <div role="row"><span role="cell">Output format</span><code role="cell">{run.provenance.requestedOutputFormat ?? "Unavailable"}</code><code role="cell">{run.provenance.actualOutputFormat ?? "Unavailable"}</code></div>
                  <div role="row"><span role="cell">Backend fallback</span><code role="cell">{run.provenance.requestedFallbackAllowed === null ? "Unavailable" : run.provenance.requestedFallbackAllowed ? "Allowed" : "Disabled"}</code><code role="cell">{run.provenance.fallbackActive === null ? "Unavailable" : run.provenance.fallbackActive ? "Active" : "Off"}</code></div>
                  {Array.from(new Set([...Object.keys(run.provenance.requestedSchemes), ...Object.keys(run.provenance.actualSchemes)])).sort().map((module) => (
                    <div role="row" key={module}><span role="cell">{module}</span><code role="cell">{run.provenance?.requestedSchemes[module] ?? "Unavailable"}</code><code role="cell">{run.provenance?.actualSchemes[module] ?? "Unavailable"}</code></div>
                  ))}
                </div>
                <div className="run-detail__provenance-identities">
                  <span>Executable</span><code title={run.provenance.executablePath ?? undefined}>{run.provenance.executablePath}</code>
                  <span>Executable digest</span><code>{run.provenance.executableDigest}</code>
                  <span>YAML digest</span><code>{run.provenance.configurationDigest}</code>
                  <span>Catalog</span><code>{run.provenance.catalogFingerprint ?? "Unavailable"}</code>
                  <span>Requested schemes</span>
                  <code>{Object.entries(run.provenance.requestedSchemes).map(([module, scheme]) => `${module}=${scheme}`).join(" · ") || "Unavailable"}</code>
                  <span>Initialization source</span>
                  <code title={run.provenance.soundingCanonicalPath ?? run.provenance.soundingRequestedPath ?? undefined}>{run.provenance.soundingCanonicalPath ?? run.provenance.soundingRequestedPath ?? "Unavailable"}</code>
                  <span>Source digest</span><code>{run.provenance.soundingDigest ?? "Unavailable"}</code>
                </div>
                {run.provenance.soundingIssue && (
                  <p className="run-detail__provenance-boundary">Initialization source unavailable: {run.provenance.soundingIssue}</p>
                )}
                <p className="run-detail__provenance-boundary">
                  Launch evidence is immutable before spawn. Runtime evidence is written once after process exit and is linked to the exact launch record; unavailable means Studio did not observe a qualifying fact.
                </p>
                {run.provenance.runtimeHealth !== "verified" && (
                  <p className="run-detail__provenance-problem" role={run.provenance.runtimeHealth === "invalid" ? "alert" : "status"}>
                    {run.provenance.runtimeIssue ?? "No validated runtime completion record is available."}
                  </p>
                )}
                <div className="run-detail__provenance-actions">
                  <button
                    type="button"
                    onClick={() => void onExportProvenance()}
                    disabled={!run.outputDir || provenanceExport.kind === "saving" || run.provenance.runtimeHealth !== "verified"}
                    title={!run.outputDir
                      ? "This run has no local output directory to receive an evidence bundle"
                      : run.provenance.runtimeHealth !== "verified"
                        ? "Evidence export requires a verified runtime completion record"
                        : provenanceExport.kind === "saving"
                          ? "Evidence export is already in progress"
                          : "Write a validated launch and runtime evidence bundle beside this run"}
                  >
                    {provenanceExport.kind === "saving" ? "Exporting evidence…" : "Export evidence bundle"}
                  </button>
                  <span aria-live="polite">
                    {provenanceExport.kind === "saved"
                      ? `Copy saved to ${provenanceExport.path}`
                      : provenanceExport.kind === "error"
                        ? provenanceExport.message
                        : ""}
                  </span>
                </div>
              </>
            ) : (
              <p className="run-detail__provenance-problem">
                {run.provenance.issue ?? "Nimbus Studio could not verify this run's provenance record."}
              </p>
            )}
          </div>
        </details>
      )}

      {run.executionTarget && isRunning && <div className="run-detail__library-notice run-detail__library-notice--partial" role="status">
        <div><strong>Remote simulation belongs to this Studio session</strong><span>Keep this computer awake and SSH connected. Closing Studio or losing the connection requests a remote stop. Plots here use this computer; the node’s desktop can render through a remote visualization workspace.</span></div>
      </div>}
      {run.remoteTerminationUnconfirmed && (
        <div className="run-detail__library-notice run-detail__library-notice--partial" role="alert">
          <Icon name="warning" size={16} />
          <div><strong>Remote termination is unconfirmed</strong><span>Studio lost the connection and could not confirm that the remote simulation stopped. Restore access to {run.executionTarget?.host ?? "the remote computer"}, then choose Retry stop.</span></div>
        </div>
      )}
      {transferActive && (
        <div className="run-detail__library-notice run-detail__library-notice--partial" role="status">
          <Icon name="archive" size={16} />
          <div><strong>Receiving results</strong><span>Studio is checking remote completion and receiving the retained saved output for local plots.</span></div>
        </div>
      )}
      {run.executionTarget && !run.remoteTerminationUnconfirmed && (
        <div className="run-detail__library-notice run-detail__library-notice--partial" role={receiveError || receive?.error ? "alert" : "status"}>
          <Icon name="archive" size={16} />
          <div><strong>{transferActive ? "Receiving saved remote results" : receive?.state === "complete" ? "Remote results received" : "Remote results"}</strong>
            <span>{receiveError ?? receive?.error ?? (receive?.state === "complete"
              ? "The separately recorded receive evidence validates these local files. Original computation evidence remains unchanged."
              : "Receive the retained saved output after remote termination is confirmed. This does not rerun the simulation.")}</span>
            {!transferActive && receive?.state !== "complete" && <button type="button" onClick={() => void onReceive()}
              disabled={isRunning || !run.outputDir || !receive?.retryAvailable}
              title={isRunning ? "Wait for the remote computation to finish or confirm its stop" : !receive?.retryAvailable ? "Receiving requires valid remote launch and recovery evidence" : "Verify and receive retained remote outputs without rerunning compute"}>
              {receive?.state === "available" ? "Receive results" : "Retry receiving results"}
            </button>}
          </div>
        </div>
      )}

      {run.origin === "disk" ? (
        <section className="run-detail__library" aria-label="Run library summary">
          <div
            className={`run-detail__library-notice run-detail__library-notice--${run.libraryHealth}`}
            role={run.libraryHealth === "ready" ? "status" : "alert"}
          >
            <Icon name={run.libraryHealth === "ready" ? "archive" : "warning"} size={16} />
            <div>
              <strong>
                {run.libraryHealth === "ready"
                  ? "Run rebuilt from complete publications"
                  : run.libraryHealth === "partial"
                    ? "Run is only partially published"
                    : "Run outputs need attention"}
              </strong>
              <span>
                {run.libraryIssue ??
                  "Nimbus Studio found this run on disk and verified its published frames."}
              </span>
            </div>
          </div>
          <div className="run-detail__stats-grid run-detail__stats-grid--library">
            <Stat label="Complete frames" value={(run.frameCount ?? 0).toLocaleString()} />
            <Stat label="Latest step" value={run.step?.toLocaleString() ?? "—"} />
            <Stat label="Fields" value={(run.fieldCount ?? 0).toLocaleString()} />
            <Stat label="Format" value={run.outputFormat ?? "Unknown"} />
          </div>
          <div className="run-detail__library-path">
            <span>Last indexed</span>
            <strong>{fmtDate(run.modifiedAtMs)}</strong>
            <span>Directory</span>
            <code title={run.outputDir ?? undefined}>{run.outputDir}</code>
          </div>
          {run.libraryHealth !== "ready" && (
            <p className="run-detail__library-recovery">
              Keep this folder while a simulation may still be writing. Reopen Studio or wait for the automatic refresh after publication completes. Use Open dir to inspect the source files.
            </p>
          )}
        </section>
      ) : (
      <section className="run-detail__progress">
        <div className="run-detail__progress-row">
          <span className="run-detail__progress-label">Simulated time</span>
          <span className="run-detail__progress-value">
            {fmtTime(run.simTimeS)} / {fmtTime(run.durationS)}
            <span className="run-detail__progress-pct"> · {pct}%</span>
          </span>
        </div>
        <div
          className="run-detail__progress-bar"
          role="progressbar"
          aria-label="Simulation progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Number(pct)}
          aria-valuetext={`${pct}% simulated time`}
        >
          <div
            className="run-detail__progress-fill"
            style={{ transform: `scaleX(${progress})` }}
          />
        </div>

        <div className="run-detail__stats-grid">
          <Stat label="Wall clock" value={fmtTime(elapsedS)} />
          <Stat label="ETA" value={fmtTime(etaS)} />
          <Stat
            label="Rate"
            value={simRate != null ? `${simRate.toFixed(2)}× sim/wall` : "—"}
          />
          <Stat
            label="max |v|"
            value={run.velMax != null ? `${run.velMax.toFixed(2)} m/s` : "—"}
          />
        </div>

        {velHistory.length >= 2 && (
          <VelMaxSparkline samples={velHistory} />
        )}
      </section>
      )}

      {run.origin !== "disk" && (
        <section className="run-detail__log">
          <LogStream runId={run.id} />
        </section>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="run-detail__stat">
      <span className="run-detail__stat-label">{label}</span>
      <span className="run-detail__stat-value">{value}</span>
    </div>
  );
}

/**
 * Mini SVG sparkline for max |velocity| over simulated time. Auto-
 * scales to the observed y-range; if min == max, draws a flat line at
 * mid-height. Renders nothing when fewer than two samples are
 * available (see useVelMaxHistory).
 */
function VelMaxSparkline({ samples }: { samples: VelMaxSample[] }) {
  const W = 320;
  const H = 48;
  const PAD = 4;

  const { points, vMin, vMax } = useMemo(() => {
    const xs = samples.map((s) => s.simTimeS);
    const ys = samples.map((s) => s.velMax);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (yMin === yMax) {
      // Pad the y-range so a constant trace renders as a centered flat
      // line rather than collapsing to a degenerate viewBox.
      yMin -= 1;
      yMax += 1;
    }
    const xRange = Math.max(1e-9, xMax - xMin);
    const yRange = Math.max(1e-9, yMax - yMin);
    const pts = samples
      .map((s) => {
        const x = PAD + ((s.simTimeS - xMin) / xRange) * (W - 2 * PAD);
        const y = H - PAD - ((s.velMax - yMin) / yRange) * (H - 2 * PAD);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return { points: pts, vMin: yMin, vMax: yMax };
  }, [samples]);

  return (
    <div className="run-detail__sparkline">
      <div className="run-detail__sparkline-header">
        <span>max |v| trend</span>
        <span className="run-detail__sparkline-range">
          {vMin.toFixed(1)} – {vMax.toFixed(1)} m/s
        </span>
      </div>
      <svg
        className="run-detail__sparkline-svg"
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
      >
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth={1.4}
          strokeLinejoin="round"
          strokeLinecap="round"
          points={points}
        />
      </svg>
    </div>
  );
}

function StatusPill({ status }: { status: QueuedRun["status"] }) {
  return (
    <span className={`status-pill status-pill--${status}`}>
      {status}
    </span>
  );
}

function LibraryHealthPill({ health }: { health: NonNullable<QueuedRun["libraryHealth"]> }) {
  return <span className={`status-pill status-pill--library-${health}`}>{health}</span>;
}

function fmtTime(s: number | null): string {
  if (s == null) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s - m * 60);
  return `${m}m${rem.toString().padStart(2, "0")}s`;
}

function fmtDate(ms: number | null | undefined): string {
  if (ms == null) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(ms));
}
