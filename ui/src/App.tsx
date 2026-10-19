import { lazy, memo, Suspense, useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { NavRail } from "@/components/shell/NavRail";
import { QueuePanel } from "@/components/shell/QueuePanel";
import { ConfigEditor } from "@/components/config/ConfigEditor";
import { RunDetail } from "@/components/runs/RunDetail";
import { Settings } from "@/components/settings/Settings";
import { WorkerMeasurementNotice } from "@/components/settings/PerformanceSettings";
import { startAutomaticWorkerMeasurement } from "@/lib/studio/studioPerformance";
import { ErrorBoundary } from "@/components/shell/ErrorBoundary";
import { CommandPalette } from "@/components/shell/CommandPalette";
import type { StudioCommand } from "@/lib/studio/commandRegistry";
import { openDatasetForCommand, type DatasetSourceOpener } from "@/lib/studio/commandActions";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { useResizableWidth, useRuns, useSimulator } from "@/lib/core/hooks";
import type { PageId, QueuedRun } from "@/lib/core/types";
import {
  defaultArchiveRunId,
  loadStudioNavigation,
  preferredCaseRunId,
  reconcileStudioNavigationRuns,
  saveStudioNavigation,
} from "@/lib/studio/studioNavigation";
import "./styles/App.css";

const QUEUE_WIDTH_STORAGE_KEY = "nimbus-studio:layout:queue-width";
const QUEUE_WIDTH_DEFAULT = 240;
const QUEUE_WIDTH_MIN = 180;
const QUEUE_WIDTH_MAX = 560;
const PREFERRED_STARTUP_CASE = "beast_xl";

// PlotStudio pulls in Plotly.js (~3 MB minified). Lazy-load so the main
// bundle stays small and Plotly is only fetched when the user opens the
// Plots tab.
const PlotStudio = lazy(() =>
  import("@/components/studio/PlotStudio").then((m) => ({ default: m.PlotStudio })),
);

interface AppProps {
  openDatasetSource?: DatasetSourceOpener;
}

export function App({ openDatasetSource }: AppProps = {}) {
  useEffect(() => { startAutomaticWorkerMeasurement(); }, []);
  const [initialNavigation] = useState(() =>
    loadStudioNavigation(typeof window === "undefined" ? undefined : window.localStorage),
  );
  const [activePage, setActivePage] = useState<PageId>(initialNavigation.page);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(initialNavigation.selectedRunId);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [initialPlotDataset, setInitialPlotDataset] = useState<LocalDatasetSource | null>(null);
  const [initialPlotDatasetOpenGeneration, setInitialPlotDatasetOpenGeneration] = useState(0);
  const [commandActionError, setCommandActionError] = useState<string | null>(null);
  const [startupRunResolved, setStartupRunResolved] = useState(
    initialNavigation.page !== "runs" || initialNavigation.selectedRunId !== null,
  );
  const persistedNavigationRef = useRef(initialNavigation);

  const runs = useRuns(activePage === "runs");
  const sim = useSimulator();

  // Persisted queue-panel width. Drag handle in QueuePanel calls
  // `adjustQueueWidth(deltaPx)` during pointer-move; the value is exposed
  // to the .app grid via the --queue-panel-width custom property.
  const [queueWidth, adjustQueueWidth] = useResizableWidth(
    QUEUE_WIDTH_STORAGE_KEY,
    QUEUE_WIDTH_DEFAULT,
    QUEUE_WIDTH_MIN,
    QUEUE_WIDTH_MAX,
  );
  const appStyle: CSSProperties = {
    ["--queue-panel-width" as string]: `${queueWidth}px`,
  };
  const showRunLibrary = activePage === "runs";

  const runList: QueuedRun[] = runs.data ?? [];
  const selectedRun = runList.find((r) => r.id === selectedRunId) ?? null;

  useEffect(() => {
    if (startupRunResolved || activePage !== "runs" || runs.loading || !runs.data) return;
    const preferred = preferredCaseRunId(runs.data, PREFERRED_STARTUP_CASE);
    setStartupRunResolved(true);
    if (!preferred) return;
    setSelectedRunId(preferred);
    setInitialPlotDataset(null);
    setActivePage("plots");
  }, [activePage, runs.data, runs.loading, startupRunResolved]);

  useEffect(() => {
    const previous = persistedNavigationRef.current;
    if (previous.page === activePage && previous.selectedRunId === selectedRunId) return;
    const next = { page: activePage, selectedRunId };
    saveStudioNavigation(window.localStorage, next);
    persistedNavigationRef.current = next;
  }, [activePage, selectedRunId]);

  // Only reconcile after the live/archive query completes; a loading state must not erase a valid durable identity.
  useEffect(() => {
    if (activePage !== "runs" || runs.loading || !runs.data || !selectedRunId) return;
    const reconciled = reconcileStudioNavigationRuns(
      { page: activePage, selectedRunId },
      runs.data.map(({ id }) => id),
    );
    if (reconciled.selectedRunId !== selectedRunId) setSelectedRunId(null);
  }, [activePage, runs.data, runs.loading, selectedRunId]);

  // Auto-select the most relevant run in the archive when nothing is selected.
  useEffect(() => {
    if (activePage !== "runs" || selectedRunId || runs.loading || !runs.data) return;
    const preferred = defaultArchiveRunId(runs.data);
    if (preferred) setSelectedRunId(preferred);
  }, [activePage, runs.data, runs.loading, selectedRunId]);

  const onSelectRun = (id: string) => {
    setSelectedRunId(id);
    setInitialPlotDataset(null);
    setActivePage("runs");
  };

  // Drop the run record and clear selection if it was the deleted run.
  const onDeleteRun = useCallback(
    async (id: string) => {
      await runs.removeRun(id);
      setSelectedRunId((curr) => (curr === id ? null : curr));
    },
    [runs],
  );

  const goToPlots = useCallback(() => {
    setInitialPlotDataset(null);
    setActivePage("plots");
  }, []);

  const onProjectOpened = useCallback(() => {
    // Clear older shell intents so the restored project snapshot takes precedence.
    setSelectedRunId(null);
    setInitialPlotDataset(null);
    setActivePage("plots");
  }, []);

  const executeCommand = useCallback((command: StudioCommand) => {
    if (command.page) {
      setActivePage(command.page);
      return;
    }
    if (command.id === "runs.focus-search") {
      setActivePage("runs");
      requestAnimationFrame(() => document.getElementById("run-library-search")?.focus());
      return;
    }
    if (command.id === "dataset.open-cm1-history" || command.id === "dataset.open-nimbus-history") {
      setCommandActionError(null);
      void openDatasetForCommand(command.id, openDatasetSource).then((result) => {
        if (result.status !== "opened") return;
        setSelectedRunId(null);
        setInitialPlotDataset(result.source);
        setInitialPlotDatasetOpenGeneration((generation) => generation + 1);
        setActivePage("plots");
      }).catch((error: unknown) => {
        const subject = command.id === "dataset.open-cm1-history"
          ? "netCDF data"
          : "Nimbus history directory";
        const detail = error instanceof Error ? error.message : String(error);
        setCommandActionError(`Could not open ${subject}. ${detail}`);
      });
    }
  }, [openDatasetSource]);

  return (
    <div className={`app ${showRunLibrary ? "app--with-run-library" : ""}`} style={appStyle}>
      <WorkerMeasurementNotice />
      <NavRail
        activePage={activePage}
        onSelect={setActivePage}
        onOpenCommandPalette={() => setCommandPaletteOpen(true)}
      />
      {showRunLibrary && (
        <QueuePanel
          runs={runList}
          simulator={sim.data ?? { binaryPath: null, state: "unknown", gpu: null }}
          activePage={activePage}
          selectedRunId={selectedRunId}
          onSelectRun={onSelectRun}
          onDeleteRun={onDeleteRun}
          onResize={adjustQueueWidth}
          runsError={runs.error}
          loading={runs.loading}
          simulatorError={sim.error}
        />
      )}
      <ErrorBoundary
        label="This view"
        resetKeys={[
          activePage,
          selectedRunId,
          initialPlotDataset?.locator.path ?? null,
          initialPlotDatasetOpenGeneration,
        ]}
      >
        <PageView
          activePage={activePage}
          selectedRun={activePage === "runs" ? selectedRun : null}
          initialRunDir={selectedRun?.outputDir ?? null}
          initialRunStatus={selectedRun?.status ?? null}
          goToPlots={goToPlots}
          initialDataset={initialPlotDataset}
          initialDatasetOpenGeneration={initialPlotDatasetOpenGeneration}
          onProjectOpened={onProjectOpened}
          simulator={sim}
        />
      </ErrorBoundary>
      <CommandPalette
        open={commandPaletteOpen}
        onOpenChange={setCommandPaletteOpen}
        onExecute={executeCommand}
      />
      {commandActionError && (
        <aside className="app-command-error" role="alert" aria-label="Command error">
          <span>{commandActionError}</span>
          <button type="button" onClick={() => setCommandActionError(null)} aria-label="Dismiss command error">Dismiss</button>
        </aside>
      )}
    </div>
  );
}

interface PageViewProps {
  activePage: PageId;
  selectedRun: QueuedRun | null;
  initialRunDir: string | null;
  initialRunStatus: QueuedRun["status"] | null;
  goToPlots: () => void;
  initialDataset: LocalDatasetSource | null;
  initialDatasetOpenGeneration: number;
  onProjectOpened: () => void;
  simulator: ReturnType<typeof useSimulator>;
}

const PageView = memo(function PageView({
  activePage,
  selectedRun,
  initialRunDir,
  initialRunStatus,
  goToPlots,
  initialDataset,
  initialDatasetOpenGeneration,
  onProjectOpened,
  simulator,
}: PageViewProps) {
  switch (activePage) {
    case "projects":
      return (
        <Suspense fallback={<LazyFallback label="Loading Projects…" />}>
          <PlotStudio
            showProjectNavigator
            onProjectOpened={onProjectOpened}
          />
        </Suspense>
      );
    case "config":
      return <ConfigEditor />;
    case "runs":
      return <RunDetail run={selectedRun} goToPlots={goToPlots} />;
    case "plots":
      return (
        <Suspense fallback={<LazyFallback label="Loading Plot Studio…" />}>
          <PlotStudio
            initialDataset={initialDataset}
            initialDatasetOpenGeneration={initialDatasetOpenGeneration}
            initialRunDir={initialRunDir}
            initialRunStatus={initialRunStatus}
            onProjectOpened={onProjectOpened}
          />
        </Suspense>
      );
    case "settings":
      return <Settings simulator={simulator} />;
  }
});

function LazyFallback({ label }: { label: string }) {
  return (
    <main className="lazy-fallback">
      <span>{label}</span>
    </main>
  );
}
