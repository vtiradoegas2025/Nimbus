import { useEffect, useMemo, useRef, useState } from "react";
import { filterRunLibrary } from "@/lib/run/runLibrary";
import type { RunsError, RunsErrorScope } from "@/lib/core/hooks";
import type { PageId, QueuedRun, SimulatorStatus } from "@/lib/core/types";
import { Icon } from "./Icon";
import { ResizeHandle } from "./ResizeHandle";
import "./QueuePanel.css";

interface QueuePanelProps {
  runs: QueuedRun[];
  simulator: SimulatorStatus;
  activePage: PageId;
  selectedRunId: string | null;
  onSelectRun: (id: string) => void;
  /** Optional delete handler — when omitted (browser mock) the X is
   *  hidden. Provided by App.tsx via useRuns().removeRun. */
  onDeleteRun?: (id: string) => Promise<void>;
  /**
   * Drag-resize delta callback. When omitted (mock/browser mode), the
   * panel stays at its theme-default width and no handle is rendered.
   */
  onResize?: (deltaPx: number) => void;
  /** Most recent run-subsystem failure, carrying which subsystem failed so the
   *  banner can name the right remedy instead of always blaming the archive. */
  runsError?: RunsError | null;
  /** True while the first live+archive sync is in flight. Without it, "still
   *  loading" and "there are no runs" render as the same definitive sentence. */
  loading?: boolean;
  /** Reason the simulator probe itself failed, as opposed to no simulator being
   *  configured. The two are otherwise indistinguishable in the footer. */
  simulatorError?: string | null;
}

/** Banner copy per failure scope. Each names what is actually broken and where
 *  to go; the underlying message is rendered alongside, never in place of it. */
const RUNS_ERROR_COPY: Record<RunsErrorScope, { title: string; remedy: string }> = {
  library: {
    title: "The run library could not be rebuilt.",
    remedy: "Check the Runs directory in Settings. Live runs remain available.",
  },
  live: {
    title: "Live run status is unavailable.",
    remedy: "The list below may be stale. Archived runs are still readable.",
  },
  action: {
    title: "That action was refused.",
    remedy: "The run list has been restored to its previous state.",
  },
};

export function QueuePanel({
  runs,
  simulator,
  activePage,
  selectedRunId,
  onSelectRun,
  onDeleteRun,
  onResize,
  runsError,
  loading = false,
  simulatorError,
}: QueuePanelProps) {
  const [query, setQuery] = useState("");
  const [showAllRecent, setShowAllRecent] = useState(false);
  const heading = HEADING_BY_PAGE[activePage];
  const visibleRuns = useMemo(() => filterRunLibrary(runs, query), [runs, query]);
  const running = visibleRuns.filter((r) => r.status === "running");
  const queued = visibleRuns.filter((r) => r.status === "queued");
  const completed = visibleRuns.filter(
    (r) => r.status === "completed" || r.status === "failed" || r.status === "aborted",
  );
  const empty = running.length === 0 && queued.length === 0 && completed.length === 0;
  const scanLimit = runs.find((run) => run.libraryScanLimited)?.libraryScanLimit;
  // Archive collapses by default; expander is required — this panel is the only run browser.
  const shownRecent = showAllRecent ? completed : completed.slice(0, RECENT_COLLAPSED_COUNT);
  const hiddenRecent = completed.length - shownRecent.length;

  return (
    <aside className="queue-panel">
      <header className="queue-panel__header">
        <h2>{heading}</h2>
      </header>

      <div className="queue-panel__body">
        <div className="queue-panel__search">
          <label htmlFor="run-library-search">Find a run</label>
          <div className="queue-panel__search-control">
            <input
              id="run-library-search"
              type="search"
              value={query}
              // The rail is user-resizable down to a narrow width, so the
              // placeholder has to survive it. The fields it actually matches
              // are named in the tooltip rather than clipped in the box.
              placeholder="Search runs…"
              title="Matches case name, output format, path, and status"
              onChange={(event) => setQuery(event.target.value)}
            />
            {query && (
              <button type="button" onClick={() => setQuery("")} aria-label="Clear run search">
                <Icon name="x" size={12} />
              </button>
            )}
          </div>
          <span className="queue-panel__search-count" aria-live="polite">
            {visibleRuns.length} of {runs.length} runs
          </span>
        </div>
        {runsError && (
          <div className="queue-panel__library-error" role="status">
            <Icon name="warning" size={14} />
            <span>
              <span className="queue-panel__error-title">{RUNS_ERROR_COPY[runsError.scope].title}</span>{" "}
              {RUNS_ERROR_COPY[runsError.scope].remedy}
              <span className="queue-panel__error-detail">{runsError.message}</span>
            </span>
          </div>
        )}
        {scanLimit != null && (
          <div className="queue-panel__library-limit" role="status">
            Showing the latest {scanLimit.toLocaleString()} run folders. Narrow the Runs directory in Settings to index a different collection.
          </div>
        )}
        {empty ? (
          <div className="queue-panel__empty">
            {query
              ? "No runs match this search."
              : loading
                // "No local runs found" is a definitive claim. Making it before
                // the archive scan returns is simply wrong, and the rows then
                // pop in underneath it.
                ? "Scanning the run library…"
                : "No local runs found. New Nimbus runs will appear here automatically."}
          </div>
        ) : (
          <>
            {(running.length > 0 || queued.length > 0) && (
              <div className="queue-panel__section">
                <div className="queue-panel__section-label">
                  <Icon name="queue" size={14} />
                  <span>Queue</span>
                </div>
                <ul className="queue-list">
                  {running.map((run) => (
                    <RunItem
                      key={run.id}
                      run={run}
                      selected={run.id === selectedRunId}
                      onSelect={onSelectRun}
                    />
                  ))}
                  {queued.map((run) => (
                    <RunItem
                      key={run.id}
                      run={run}
                      selected={run.id === selectedRunId}
                      onSelect={onSelectRun}
                    />
                  ))}
                </ul>
              </div>
            )}
            {completed.length > 0 && (
              <div className="queue-panel__section">
                <div className="queue-panel__section-label">
                  <Icon name="archive" size={14} />
                  <span>Recent</span>
                </div>
                <ul className="queue-list">
                  {shownRecent.map((run) => (
                    <RunItem
                      key={run.id}
                      run={run}
                      selected={run.id === selectedRunId}
                      onSelect={onSelectRun}
                      onDelete={run.origin === "disk" || run.libraryHealth ? undefined : onDeleteRun}
                    />
                  ))}
                </ul>
                {completed.length > RECENT_COLLAPSED_COUNT && (
                  <button
                    type="button"
                    className="queue-panel__more"
                    onClick={() => setShowAllRecent((previous) => !previous)}
                  >
                    {showAllRecent
                      ? `Show fewer (${RECENT_COLLAPSED_COUNT} of ${completed.length})`
                      : `Show ${hiddenRecent} more`}
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>

      <footer className="queue-panel__footer">
        <SimulatorStatusRow simulator={simulator} probeError={simulatorError} />
      </footer>

      {onResize && (
        <ResizeHandle
          onResize={onResize}
          label="Resize queue panel"
          minValuePx={180}
          maxValuePx={560}
        />
      )}
    </aside>
  );
}

/** Archived runs shown before the "Show N more" expander. Sized so the live
 *  and queued sections stay visible without scrolling at the minimum window
 *  height. */
const RECENT_COLLAPSED_COUNT = 12;

const HEADING_BY_PAGE: Record<PageId, string> = {
  projects: "Projects",
  config: "Configure",
  runs: "Run Archive",
  plots: "Plot Studio",
  settings: "Settings",
};

interface RunItemProps {
  run: QueuedRun;
  selected: boolean;
  onSelect: (id: string) => void;
  /** When provided, an X button appears on hover; clicking it switches
   *  to an inline confirm state that calls this on Yes. Not provided
   *  for running runs (the user must abort first). */
  onDelete?: (id: string) => Promise<void>;
}

function RunItem({ run, selected, onSelect, onDelete }: RunItemProps) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const openRef = useRef<HTMLButtonElement>(null);
  const confirmRemoveRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (confirming) confirmRemoveRef.current?.focus();
  }, [confirming]);

  const progress =
    run.simTimeS != null && run.durationS > 0
      ? Math.min(1, run.simTimeS / run.durationS)
      : 0;

  // Plots chip gating: in prod only show on fully-completed runs. Partial
  // frames + NaN scalars from running / failed / aborted runs are the
  // primary trigger for 3D GPU hangs (volume render + isosurface). Dev
  // builds stay unguarded so we can debug mid-sim from a hot-reloaded
  // tauri:dev session.
  const diskPlotSafe =
    run.origin === "disk" &&
    run.libraryHealth !== "invalid" &&
    (run.frameCount ?? 0) > 0;
  const showPlotsChip =
    !!run.outputDir &&
    !run.remoteTerminationUnconfirmed &&
    !run.remoteTransferInProgress &&
    (diskPlotSafe ||
      (run.origin !== "disk" && (import.meta.env.DEV || run.status === "completed")));

  const iconName: "dot" | "queue" | "archive" =
    run.status === "running"
      ? "dot"
      : run.status === "queued"
        ? "queue"
        : "archive";

  const handleSelect = () => {
    // Don't surface row selection through confirm clicks — the user is
    // mid-decision, not picking a run.
    if (confirming) return;
    onSelect(run.id);
  };

  const cancelDelete = () => {
    setConfirming(false);
    requestAnimationFrame(() => openRef.current?.focus());
  };

  const handleDelete = async () => {
    if (!onDelete || deleting) return;
    setDeleting(true);
    try {
      await onDelete(run.id);
      // No need to setConfirming(false) — the item unmounts on success.
    } catch {
      // useRuns surfaces the error globally; restore the confirm UI
      // so the user can retry or cancel.
      setDeleting(false);
      cancelDelete();
    }
  };

  return (
    <li
      className={`run-item run-item--${run.status} ${selected ? "run-item--selected" : ""}`}
    >
      <button
        ref={openRef}
        type="button"
        className="run-item__open"
        onClick={handleSelect}
        disabled={confirming}
        aria-current={selected ? "true" : undefined}
        aria-label={`Open ${run.caseName} in Run Archive`}
      />
      <div className="run-item__row">
        <Icon name={iconName} size={10} />
        <span className="run-item__name" title={run.caseName}>{run.caseName}</span>
        {onDelete && !confirming && (
          <button
            type="button"
            className="run-item__delete"
            title="Remove this live run record from the session"
            aria-label={`Remove ${run.caseName} from the session`}
            onClick={() => setConfirming(true)}
          >
            <Icon name="x" size={12} />
          </button>
        )}
      </div>
      {confirming ? (
        <div className="run-item__confirm">
          <span className="run-item__confirm-text">Remove record?</span>
          <button
            type="button"
            className="run-item__confirm-yes"
            ref={confirmRemoveRef}
            onClick={handleDelete}
            disabled={deleting}
          >
            {deleting ? "Removing…" : "Remove"}
          </button>
          <button
            type="button"
            className="run-item__confirm-no"
            onClick={cancelDelete}
            disabled={deleting}
          >
            Keep
          </button>
        </div>
      ) : (
        <div className="run-item__row run-item__row--meta">
          {/* Ellipsised in the 240px rail, so the full value has to stay
            * reachable on hover rather than being silently clipped. */}
          <span
            className="run-item__scheme"
            title={run.origin === "disk" ? run.outputFormat : run.scheme}
          >
            {run.origin === "disk" ? run.outputFormat : run.scheme}
          </span>
          {run.libraryHealth && (
            <span className={`run-item__health run-item__health--${run.libraryHealth}`}>
              {run.libraryHealth}
            </span>
          )}
          {showPlotsChip && (
            <span
              className="run-item__plots-chip"
              title={`Outputs in ${run.outputDir}`}
            >
              plots
            </span>
          )}
          {run.simTimeS != null ? (
            <span className="run-item__time">
              {formatTime(run.simTimeS)} / {formatTime(run.durationS)}
            </span>
          ) : (
            <span className="run-item__time">{formatTime(run.durationS)}</span>
          )}
        </div>
      )}
      {run.status === "running" && (
        <div
          className="run-item__progress"
          role="progressbar"
          aria-label={`${run.caseName} simulation progress`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
        >
          <div
            className="run-item__progress-fill"
            style={{ transform: `scaleX(${progress})` }}
          />
        </div>
      )}
      {run.executionTarget && <div className="run-item__row run-item__row--meta" title={`${run.executionTarget.name} · ${run.executionTarget.host}`}><span className="run-item__scheme">{run.executionTarget.name}</span></div>}
      {run.remoteTerminationUnconfirmed && <div className="run-item__row run-item__row--meta" role="alert"><span title="The remote simulation may still be running. Open this run to retry stopping it.">Remote stop unconfirmed</span></div>}
      {(run.remoteTransferInProgress || run.remoteReceive?.state === "receiving") && <div className="run-item__row run-item__row--meta" role="status"><span title="Saved output is being received and checked for local plots.">Receiving results</span></div>}
      {run.remoteReceive?.error && run.remoteReceive.state !== "receiving" && <div className="run-item__row run-item__row--meta" role="alert"><span title={run.remoteReceive.error}>{run.remoteReceive.retryAvailable ? "Results not received · open to retry" : "Receive evidence invalid"}</span></div>}
    </li>
  );
}

function SimulatorStatusRow({
  simulator,
  probeError,
}: {
  simulator: SimulatorStatus;
  probeError?: string | null;
}) {
  const ready = simulator.state === "ready";
  // A probe that threw is NOT the same as "no simulator configured": telling
  // that user to connect one sends them to re-enter a path that may already be
  // correct. Warn instead of muting, and say what actually happened.
  const label = probeError ? "Simulator status unknown" : ready ? "Simulation ready" : "Analysis only";
  const detail = probeError
    ? probeError
    : simulator.binaryPath ?? "Connect a simulator in Settings to run";
  const tone = probeError ? "warn" : ready ? "ok" : "muted";
  return (
    <div className="sim-status">
      <div className="sim-status__row">
        <span className={`sim-status__dot sim-status__dot--${tone}`} />
        <div className="sim-status__text">
          <div className="sim-status__label">{label}</div>
          <div className="sim-status__sub" title={detail}>
            {detail}
          </div>
        </div>
      </div>
      {simulator.gpu && (
        <div className="sim-status__row">
          <span className="sim-status__dot sim-status__dot--info" />
          <div className="sim-status__text">
            <div className="sim-status__label">GPU detected</div>
            <div className="sim-status__sub" title={simulator.gpu}>
              {simulator.gpu}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatTime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s - m * 60);
  return `${m}m${rem.toString().padStart(2, "0")}s`;
}
