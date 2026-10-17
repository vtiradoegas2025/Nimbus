import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  clearFieldCache,
  registerExpressionDerivedFields,
  registerExternalDerivedProviders,
  type StepInfo,
} from "@/lib/run/runOutputs";
import { axisFrame } from "@/lib/run/frameMotion";
import {
  DATASET_ADAPTERS,
  createLocalDatasetSource,
  datasetSourceIdentity,
  type Cm1CaseInventory,
  type DatasetSourceKind,
  type LocalDatasetSource,
} from "@/lib/dataset/datasetSources";
import {
  datasetHasTrajectories,
  inspectNetcdfMetadata,
  listDatasetFrames,
  openLocalDatasetSource,
  reopenCm1CaseFolder,
  type NetcdfMetadata,
} from "@/lib/dataset/datasetRuntime";
import { filterCompleteSteps } from "@/lib/run/stepCompleteness";
import { reconcileLiveSteps } from "@/lib/dataset/liveStepDiscovery";
import {
  describeFrameTransition,
  runViewIdentityKey,
} from "@/lib/run/frameIdentity";
import type { RunStatus } from "@/lib/core/types";
import { MovieRecorderDialog } from "./MovieRecorderDialog";
import { ErrorBoundary } from "@/components/shell/ErrorBoundary";
import {
  PANE_LAYOUTS,
  activeThreeDPaneBudgetScale,
  paneCount,
  paneGridTemplate,
  paneLayoutLabel,
  type PaneLayout,
} from "@/lib/plot/paneLayout";
import { createCameraSyncBus, type CameraSyncBus } from "@/lib/plot/cameraSyncBus";
import { PaneProvider } from "@/components/plots/paneContext";
import {
  EMPTY_PRESENTATION_DWELL,
  panesAreLoadingFrame,
  panesReadyForAdvance,
  updatePresentationDwell,
  visiblePaneTypes,
  type PresentationDwellState,
} from "@/lib/plot/playbackGate";
import { simulationPlaybackTimeline, simulationPlaybackIntervalMs } from "@/lib/plot/simulationPlayback";
import { PlaybackProvider } from "@/components/plots/playbackContext";
import { PanePresentationProvider, usePanePresentationSnapshot } from "@/components/plots/panePresentationContext";
import type { PanePresentationSnapshot } from "@/lib/plot/panePresentationBarrier";
import { Icon } from "@/components/shell/Icon";
import { CompositePlot } from "@/components/plots/CompositePlot";
import { Hodograph } from "@/components/plots/Hodograph";
import { RadialProfile } from "@/components/plots/RadialProfile";
import { SkewT } from "@/components/plots/SkewT";
import { SlicePlot } from "@/components/plots/SlicePlot";
import { ComparisonSlicePlot } from "@/components/plots/ComparisonSlicePlot";
import { VectorField } from "@/components/plots/VectorField";
import { TimeSeriesPlot } from "@/components/plots/TimeSeriesPlot";
import { TimeHeightPlot } from "@/components/plots/TimeHeightPlot";
import { SwathPlot } from "@/components/plots/SwathPlot";
import { TrajectoryBudgetPlot } from "@/components/plots/TrajectoryBudgetPlot";
import { VerticalProfile } from "@/components/plots/VerticalProfile";
import { RootHistoryScopeNotice } from "@/components/plots/PlotSourceDisclosure";
import {
  PLOT_PRESETS,
  resolveCm1FamiliarViews,
  type PlotPreset,
  type ResolvedCm1FamiliarView,
} from "@/lib/plot/plotPresets";
import {
  canHandOffToVolume,
  describeVolumeHandoffBlock,
  volumeHandoffPreset,
  VOLUMETRIC_SOURCE_TYPES,
  type ScalarSelection,
  type VolumeHandoffOffer,
} from "@/lib/plot/volumetricHandoff";
import {
  categoryMeta,
  metaFor,
  availableFields,
  TRAJECTORY_CAPABILITY,
  PLOT_TYPE_META,
  visibleCategories,
  visiblePlotTypes,
  visiblePresets,
  isPlotAvailable,
  firstAvailableCategory,
  firstAvailablePlotType,
  type PlotCategory,
  type PlotType,
  type PlotTypeMeta,
  type CategoryMeta,
} from "@/lib/plot/plotRegistry";
import { addPlotFieldCapabilities } from "@/lib/plot/fieldPlotEligibility";
import {
  EMPTY_PLOT_VIEW_HISTORY,
  recordPlotView,
  redoPlotView,
  undoPlotView,
  type PlotViewHistory,
  type PlotViewSelection,
} from "@/lib/plot/plotViewHistory";
import { isEditableTarget } from "@/lib/studio/commandRegistry";
import type { SliceAxis } from "@/lib/platform/npy";
import type { FrameDisplayReport, PlotDatasetHandle } from "@/components/plots/common";
import type { DisplayedFrameIdentity } from "@/lib/run/frameIdentity";
import { setPlaybackPrefetchHint, upcomingSteps } from "@/lib/run/framePrefetch";
import { analyzeDatasetTime } from "@/lib/dataset/datasetTimeHealth";
import { assessCm1HistoryVariable } from "@/lib/cm1/cm1HistoryMapping";
import { summarizeCm1Namelist } from "@/lib/cm1/cm1NamelistSummary";
import { summarizeCm1Sounding } from "@/lib/cm1/cm1SoundingSummary";
import { Cm1CompatibilityReport } from "@/components/cm1/Cm1CompatibilityReport";
import { DatasetAlignmentReport } from "@/components/reports/DatasetAlignmentReport";
import { EnsembleAlignmentReport } from "@/components/reports/EnsembleAlignmentReport";
import { alignDatasets } from "@/lib/dataset/datasetAlignment";
import { alignEnsembleMembers, MAX_ENSEMBLE_MEMBERS, type EnsembleMemberInput } from "@/lib/comparison/ensembleAlignment";
import { admitEnsembleMember, openRequiredDatasetSources } from "@/lib/studio/studioDatasetOpening";
import { EnsembleAnalysisPlot } from "@/components/plots/EnsembleAnalysisPlot";
import { parseResearchRecipe, RESEARCH_RECIPE_CHANGE_EVENT } from "@/lib/research/recipe";
import { ResearchWorkspace } from "@/components/research/ResearchWorkspace";
import { InvestigationWorkspace } from "./InvestigationWorkspace";
import { DatasetHealthReport } from "@/components/reports/DatasetHealthReport";
import { DerivedFieldLedger } from "@/components/derived/DerivedFieldLedger";
import { RecentDatasetsPanel } from "./RecentDatasetsPanel";
import { RemoteWorkspaceSettings } from "@/components/settings/RemoteWorkspaceSettings";
import { ResourceBudgetPanel } from "./ResourceBudgetPanel";
import { useStudioHeaderMenus } from "./useStudioHeaderMenus";
import { ExpressionDerivedFieldEditor } from "@/components/derived/ExpressionDerivedFieldEditor";
import { ProjectNavigator } from "./ProjectNavigator";
import { ExternalDerivedProviderManager } from "@/components/derived/ExternalDerivedProviderManager";
import {
  augmentStepWithExpressionDerivedFields,
  loadExpressionDerivedFields,
  saveExpressionDerivedFields,
  type ExpressionDerivedFieldDefinition,
} from "@/lib/derived/expressionDerivedFields";
import { DERIVED_FIELD_DEFINITIONS } from "@/lib/derived/derivedFieldRegistry";
import {
  forgetRecentDataset,
  loadRecentDatasets,
  rememberRecentDataset,
  saveRecentDatasets,
  setRecentDatasetPinned,
  type RecentDatasetEntry,
} from "@/lib/dataset/recentDatasets";
import {
  clearComparisonSession,
  createComparisonSession,
  loadComparisonSession,
  saveComparisonSession,
} from "@/lib/comparison/comparisonSession";
import {
  cm1HandoffComparisonPair,
  loadCm1GuidedHandoff,
  markCm1ComparisonOpened,
  reconcileCm1SmokeRun,
  saveCm1GuidedHandoff,
} from "@/lib/cm1/cm1GuidedHandoff";
import { discoverRunLibrary } from "@/lib/platform/tauri";
import { isTransientAppError } from "@/lib/platform/appInvoke";
import { effectiveRunsDirectory, loadStudioSettings } from "@/lib/studio/studioSettings";
import {
  createComparisonPublicationWorkspace,
  downloadComparisonPublication,
  parseComparisonPublication,
  serializeComparisonPublication,
} from "@/lib/comparison/comparisonPublication";
import { loadComparisonWorkspace, saveComparisonWorkspace } from "@/lib/comparison/comparisonWorkspace";
import { loadInvestigationWorkspace, saveInvestigationWorkspace } from "@/lib/studio/investigationWorkspace";
import {
  orphanedPythonWorkspaceKeys,
  pythonWorkspaceKeysForProject,
} from "@/lib/studio/pythonWorkspace";
import {
  createStudioProject,
  deleteStudioProject,
  duplicateStudioProject,
  instantiateBundledProjectExample,
  loadActiveStudioProjectId,
  loadStudioProjects,
  markStudioProjectOpened,
  renameStudioProject,
  saveStudioProjects,
  saveActiveStudioProjectId,
  updateStudioProjectSnapshot,
  upsertStudioProject,
  type BundledProjectExample,
  type StudioProject,
  type StudioProjectAnalysisSnapshot,
  type StudioProjectWorkspaceSnapshot,
} from "@/lib/studio/studioProjects";
import {
  createStudioProjectPublication,
  downloadStudioProjectPublication,
  materializeStudioProjectPublication,
  parseStudioProjectPublication,
  rebindStudioProjectSource,
  serializeStudioProjectPublication,
  type StudioProjectSourceRole,
} from "@/lib/studio/studioProjectPublication";
import { createPackagedAtmosphericExampleSource } from "@/lib/dataset/packagedAtmosphericExamples";
import { summarizeStudioProjectSourceAudit, verifyStudioProjectSources } from "@/lib/studio/studioProjectSources";
import {
  loadExternalDerivedProviders,
  saveExternalDerivedProviders,
  type ExternalDerivedProvider,
  type ExternalProviderRuntimeStatus,
} from "@/lib/derived/externalDerivedProviders";
import { augmentStepWithExternalProviders } from "@/lib/derived/externalDerivedCatalog";
import {
  DEFAULT_RENDER_FIDELITY_MODE,
  type RenderFidelityMode,
} from "@/lib/plot/renderFidelity";
import "./PlotStudio.css";

// VTK.js (~3 MB) lazy-loads in its own chunks — only fetched when the
// user selects a 3D plot type. Volume and Isosurface share the VTK
// runtime via Vite chunk-splitting; both stay out of the main bundle.
const VolumeRender3D = lazy(() =>
  import("@/components/plots/VolumeRender3D").then((m) => ({ default: m.VolumeRender3D })),
);

function externalProviderRuntimeStatus(provider: ExternalDerivedProvider): ExternalProviderRuntimeStatus {
  return provider.kind === "python"
    ? provider.entrypoint.runtimeVersion === "0.26.4" && typeof window !== "undefined"
      ? { available: true, label: "Pyodide 0.26.4", reason: null }
      : { available: false, label: "Pyodide 0.26.4", reason: `Provider pins Pyodide ${provider.entrypoint.runtimeVersion}; Studio ships 0.26.4.` }
    : {
        available: false,
        label: `Plugin host ${provider.entrypoint.hostVersion}`,
        reason: "No native plugin-host adapter is connected in this Studio build.",
      };
}

function expressionCatalogSets(steps: readonly StepInfo[]): {
  nativeFields: Set<string>;
  availableFields: Set<string>;
  reservedFieldIds: Set<string>;
} {
  const nativeFields = new Set(steps.flatMap((step) => step.fields.filter((field) => step.fieldOrigins?.[field] === "native")));
  const availableFields = new Set(steps.flatMap((step) => step.fields.filter((field) => step.fieldOrigins?.[field] !== "user_expression")));
  return {
    nativeFields,
    availableFields,
    reservedFieldIds: new Set(DERIVED_FIELD_DEFINITIONS.map(({ id }) => id)),
  };
}
const Isosurface3D = lazy(() =>
  import("@/components/plots/Isosurface3D").then((m) => ({ default: m.Isosurface3D })),
);
const Streamlines3D = lazy(() =>
  import("@/components/plots/Streamlines3D").then((m) => ({ default: m.Streamlines3D })),
);
const SupercellRender3D = lazy(() =>
  import("@/components/plots/SupercellRender3D").then((m) => ({ default: m.SupercellRender3D })),
);
const Trajectories3D = lazy(() =>
  import("@/components/plots/Trajectories3D").then((m) => ({ default: m.Trajectories3D })),
);
// Pyodide is ~6 MB lazy-loaded on first use; the wrapper widget is
// tiny but pulls the runtime when the user clicks the Python tab.
const PythonPlot = lazy(() =>
  import("@/components/plots/PythonPlot").then((m) => ({ default: m.PythonPlot })),
);

/* Plot Studio shell. Owns the run dir + step table; delegates rendering to per-type components. */
export interface PlotStudioProps {
  onPanePresentationChange?: (snapshot: PanePresentationSnapshot) => void;
  /** Show the project ledger instead of the analysis desk. */
  showProjectNavigator?: boolean;
  /** Called after a project opens or the ledger is explicitly closed. */
  onProjectOpened?: () => void;
  /** When false, starts with an empty desk (default: true = restore active project). */
  restoreActiveProject?: boolean;
  /** Preloaded dataset; takes precedence over initialRunDir. */
  initialDataset?: LocalDatasetSource | null;
  /** Incrementing reloads an unchanged source (e.g. after new frames publish). */
  initialDatasetOpenGeneration?: number;
  /** Optional B dataset for comparison/project workspaces. */
  initialComparisonDataset?: LocalDatasetSource | null;
  /** Optional additional members for ensemble workspaces. */
  initialEnsembleDatasets?: readonly LocalDatasetSource[];
  /** Pre-loads this dir and reloads whenever it changes. */
  initialRunDir?: string | null;
  /** Status of the run owning initialRunDir; ignored once the user opens a different dir. */
  initialRunStatus?: RunStatus | null;
}

const PLAY_SPEED_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 1, label: "1 fps" },
  { value: 2, label: "2 fps" },
  { value: 5, label: "5 fps" },
  { value: 10, label: "10 fps" },
  { value: 20, label: "20 fps" },
  { value: 30, label: "30 fps" },
  { value: 60, label: "60 fps" },
];
const MINIMUM_PLAYBACK_INTERVAL_MS = 1000 / 60;
const PLAYBACK_STRIDE_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 1, label: "Every frame" },
  { value: 2, label: "Every 2nd" },
  { value: 5, label: "Every 5th" },
  { value: 10, label: "Every 10th" },
];

/** Longest playback may wait for a missing frame report before pausing. */
const PLAYBACK_GATE_MAX_HOLD_MS = 2_000;
/** Heavy bounded reads/extraction are valid work, not a missing report. Let the
 * active request finish instead of cancelling it every two seconds. */
const PLAYBACK_ACTIVE_LOAD_MAX_HOLD_MS = 60_000;
const PLAYBACK_GATE_INITIAL_RETRY_MS = 50;
const PLAYBACK_GATE_MAX_RETRY_MS = 1_000;

const THREE_D_PLOT_TYPES = new Set<PlotType>([
  "volume",
  "isosurface",
  "streamlines",
  "supercell",
  "trajectories",
]);

const PHYSICAL_GRID_3D_PLOT_TYPES = new Set<PlotType>([
  "volume",
  "isosurface",
  "streamlines",
  "supercell",
]);

function cm1CaseAssetSummary(inventory: Cm1CaseInventory): string {
  const assets = [
    `${inventory.historyFiles.length} history ${inventory.historyFiles.length === 1 ? "file" : "files"}`,
    inventory.namelistInput ? "namelist.input found" : null,
    inventory.inputSounding ? "input_sounding found" : null,
    inventory.restartFiles.length > 0
      ? `${inventory.restartFiles.length} restart ${inventory.restartFiles.length === 1 ? "candidate" : "candidates"}`
      : null,
    inventory.statisticsFiles.length > 0
      ? `${inventory.statisticsFiles.length} statistics ${inventory.statisticsFiles.length === 1 ? "file" : "files"}`
      : null,
    inventory.nestedDirectoryCount > 0
      ? `${inventory.nestedDirectoryCount} nested ${inventory.nestedDirectoryCount === 1 ? "folder" : "folders"} discovered`
      : null,
    (inventory.skippedDirectoryCount ?? 0) > 0
      ? `${inventory.skippedDirectoryCount} ${inventory.skippedDirectoryCount === 1 ? "folder" : "folders"} beyond scan limits`
      : null,
    (inventory.excludedDirectoryCount ?? 0) > 0
      ? `${inventory.excludedDirectoryCount} source/dependency ${inventory.excludedDirectoryCount === 1 ? "folder" : "folders"} excluded`
      : null,
    (inventory.unreadableDirectoryCount ?? 0) > 0
      ? `${inventory.unreadableDirectoryCount} unreadable ${inventory.unreadableDirectoryCount === 1 ? "folder" : "folders"} skipped`
      : null,
    (inventory.unreadableEntryCount ?? 0) > 0
      ? `${inventory.unreadableEntryCount} unreadable ${inventory.unreadableEntryCount === 1 ? "entry" : "entries"} skipped`
      : null,
    (inventory.symlinkCount ?? 0) > 0
      ? `${inventory.symlinkCount} ${inventory.symlinkCount === 1 ? "symlink" : "symlinks"} skipped`
      : null,
  ].filter((value): value is string => value != null);
  return assets.join(" · ");
}

function DatasetErrorNotice({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div className="plot-studio__error" role="alert">
      <Icon name="warning" size={14} />
      <div className="plot-studio__error-copy">
        <strong>Data could not be opened</strong>
        <span>{message}</span>
        <small>
          Nothing in the current workspace was changed. Confirm the selected
          files are readable netCDF history or a published Nimbus run, then
          choose the source again.
        </small>
      </div>
      <button type="button" onClick={onDismiss}>Dismiss message</button>
    </div>
  );
}

function Cm1CaseDetails({ dataset }: { dataset: LocalDatasetSource }) {
  const namelistSummary = dataset.namelistReport
    ? summarizeCm1Namelist(dataset.namelistReport)
    : null;
  const soundingSummary = dataset.soundingReport
    ? summarizeCm1Sounding(dataset.soundingReport, dataset.namelistReport)
    : null;
  return (
    <>
      {dataset.caseInventory && (
        <>
          <div className="plot-studio__case-inventory">
            <Icon name="folder" size={13} />
            <span>{cm1CaseAssetSummary(dataset.caseInventory)}</span>
          </div>
          {(dataset.caseInventory.warnings ?? []).map((warning) => (
            <div className="plot-studio__namelist-warning" role="status" key={warning}>
              <Icon name="warning" size={12} />
              <span>{warning}</span>
            </div>
          ))}
        </>
      )}
      {namelistSummary && (
        <div className="plot-studio__namelist" aria-label="Read-only CM1 namelist summary">
          <div className="plot-studio__namelist-heading">
            <strong>CM1 setup</strong>
            <span>{namelistSummary.inspection}</span>
          </div>
          {namelistSummary.controls.length > 0 && (
            <div className="plot-studio__namelist-controls">
              {namelistSummary.controls.map((control) => (
                <span key={control}>{control}</span>
              ))}
            </div>
          )}
          {dataset.namelistReport && dataset.namelistReport.warnings.length > 0 && (
            <div className="plot-studio__namelist-warning">
              <Icon name="warning" size={12} />
              <span>{dataset.namelistReport.warnings.join(" · ")}</span>
            </div>
          )}
        </div>
      )}
      {dataset.namelistInspectionError && (
        <div className="plot-studio__namelist-warning">
          <Icon name="warning" size={12} />
          <span>namelist.input could not be inspected: {dataset.namelistInspectionError}</span>
        </div>
      )}
      {soundingSummary && dataset.soundingReport && (
        <div className="plot-studio__namelist" aria-label="Read-only CM1 sounding summary">
          <div className="plot-studio__namelist-heading">
            <strong>CM1 sounding</strong>
            <span>source profile · no conversion applied</span>
          </div>
          <div className="plot-studio__namelist-controls">
            {soundingSummary.facts.map((fact) => (
              <span key={fact}>{fact}</span>
            ))}
          </div>
          <div className="plot-studio__case-disposition">
            <Icon name="info" size={12} />
            <span>{soundingSummary.disposition}</span>
          </div>
          {soundingSummary.attention && (
            <div className="plot-studio__namelist-warning" role="status">
              <Icon name="warning" size={12} />
              <span>{soundingSummary.attention}</span>
            </div>
          )}
          {dataset.soundingReport.warnings.length > 0 && (
            <div className="plot-studio__namelist-warning" role="status">
              <Icon name="warning" size={12} />
              <span>{dataset.soundingReport.warnings.join(" · ")}</span>
            </div>
          )}
        </div>
      )}
      {dataset.soundingInspectionError && (
        <div className="plot-studio__namelist-warning" role="status">
          <Icon name="warning" size={12} />
          <span>input_sounding could not be inspected: {dataset.soundingInspectionError}</span>
        </div>
      )}
      <Cm1CompatibilityReport dataset={dataset} />
    </>
  );
}

function Cm1FieldContractDetails({
  metadata,
  defaultOpen = false,
}: {
  metadata: NetcdfMetadata;
  defaultOpen?: boolean;
}) {
  const assessed = metadata.variables.map((variable) => ({
    variable,
    assessment: assessCm1HistoryVariable(variable),
  }));
  const readyCount = assessed.filter(({ assessment }) => assessment.status === "ready").length;
  const supportingCount = assessed.filter(({ assessment }) => assessment.status === "supporting").length;
  const inspectOnlyCount = assessed.length - readyCount - supportingCount;
  const visible = assessed.slice(0, 250);
  return (
    <details className="plot-studio__field-contract" open={defaultOpen}>
      <summary>
        <span className="plot-studio__field-contract-chevron" aria-hidden="true">›</span>
        <strong>CM1 field readiness</strong>
        <span>{readyCount} native-ready · {supportingCount} coordinates · {inspectOnlyCount} inspect-only</span>
      </summary>
      <div className="plot-studio__field-contract-body">
        <p>
          Every source variable remains visible. Plot-ready status requires a qualified name,
          unit, rank, and scalar-grid contract; inspect-only rows explain the exact blocker.
        </p>
        <div className="plot-studio__variable-table" role="table" aria-label="CM1 variable plot readiness">
          {visible.map(({ variable, assessment }) => (
            <div className="plot-studio__variable-row plot-studio__variable-row--contract" role="row" key={variable.name}>
              <strong role="cell">{variable.name}</strong>
              <span role="cell" className={`plot-studio__field-status plot-studio__field-status--${assessment.status}`}>
                {assessment.status === "ready" ? "Plot-ready" : assessment.status === "supporting" ? "Coordinate" : "Inspect-only"}
              </span>
              <span role="cell">{assessment.canonicalName ?? "—"}</span>
              <span role="cell">{variable.dimensions.join(" × ") || "scalar"} · {variable.units ?? "units missing"}</span>
              <span role="cell">{assessment.reason}</span>
            </div>
          ))}
        </div>
        {assessed.length > visible.length && (
          <p>Showing the first {visible.length} of {assessed.length} variables.</p>
        )}
      </div>
    </details>
  );
}

function Cm1FamiliarStart({
  views,
  activePresetId,
  activePlotType,
  activeCategory,
  onSelect,
}: {
  views: ReadonlyArray<ResolvedCm1FamiliarView>;
  activePresetId: string | null;
  activePlotType: PlotType;
  activeCategory: PlotCategory;
  onSelect: (view: ResolvedCm1FamiliarView) => void;
}) {
  const ready = views.filter((view) => view.available);
  const unavailableCount = views.length - ready.length;
  return (
    <section className="plot-studio__familiar" aria-label="CM1 familiar start">
      <div className="plot-studio__familiar-intro">
        <span>CM1 familiar start</span>
        <p>Begin with a recognizable storm-analysis view; the full Studio catalog remains below.</p>
      </div>
      {ready.length > 0 ? (
        <div className="plot-studio__familiar-rail">
          {ready.map((view, index) => {
            const active = view.kind === "preset"
              ? view.preset?.id === activePresetId
              : activePresetId == null
                && view.plotType === activePlotType
                && view.category === activeCategory;
            return (
              <button
                type="button"
                key={view.id}
                className={`plot-studio__familiar-view ${active ? "plot-studio__familiar-view--active" : ""}`}
                aria-pressed={active}
                title={view.description}
                onClick={() => onSelect(view)}
              >
                <span aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                <strong>{view.label}</strong>
              </button>
            );
          })}
        </div>
      ) : (
        <p className="plot-studio__familiar-empty">No familiar starter view is qualified from this history set yet.</p>
      )}
      {unavailableCount > 0 && (
        <span className="plot-studio__familiar-more">
          {unavailableCount} more {unavailableCount === 1 ? "view needs" : "views need"} additional qualified fields
        </span>
      )}
    </section>
  );
}

interface TaggedFrameDisplayReport extends FrameDisplayReport {
  runDir: string;
  plotType: PlotType;
  runPublicationGeneration: string | null;
  openEpoch: number;
}

function PanePresentationObserver({ onChange }: { onChange: (snapshot: PanePresentationSnapshot) => void }) {
  const snapshot = usePanePresentationSnapshot();
  useEffect(() => onChange(snapshot), [snapshot, onChange]);
  return null;
}

export function PlotStudio({
  onPanePresentationChange,
  showProjectNavigator = false,
  onProjectOpened,
  restoreActiveProject = true,
  initialDataset,
  initialDatasetOpenGeneration = 0,
  initialComparisonDataset,
  initialEnsembleDatasets,
  initialRunDir,
  initialRunStatus,
}: PlotStudioProps = {}) {
  const [dataset, setDataset] = useState<LocalDatasetSource | null>(() =>
    initialDataset ?? (initialRunDir ? createLocalDatasetSource("nimbus_run", initialRunDir) : null),
  );
  const initialDatasetIdentity = initialDataset
    ? datasetSourceIdentity(initialDataset)
    : null;
  const initialComparisonDatasetIdentity = initialComparisonDataset
    ? datasetSourceIdentity(initialComparisonDataset)
    : null;
  const initialEnsembleDatasetIdentity = initialEnsembleDatasets?.map(datasetSourceIdentity).join("|") ?? null;
  const [comparisonDataset, setComparisonDataset] = useState<LocalDatasetSource | null>(
    initialComparisonDataset ?? null,
  );
  const [comparisonSteps, setComparisonSteps] = useState<StepInfo[]>([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [compareMenuOpen, setCompareMenuOpen] = useState(false);
  const [comparisonViewActive, setComparisonViewActive] = useState(false);
  const [ensembleMembers, setEnsembleMembers] = useState<EnsembleMemberInput[]>([]);
  const ensembleMembersRef = useRef<EnsembleMemberInput[]>([]);
  const primaryDatasetRef = useRef(dataset);
  const updateEnsembleMembers = useCallback((next: EnsembleMemberInput[] | ((current: EnsembleMemberInput[]) => EnsembleMemberInput[])) => {
    const members = typeof next === "function" ? next(ensembleMembersRef.current) : next;
    ensembleMembersRef.current = members;
    setEnsembleMembers(members);
  }, []);
  const [ensembleLoading, setEnsembleLoading] = useState(false);
  const [ensembleError, setEnsembleError] = useState<string | null>(null);
  const [ensembleMenuOpen, setEnsembleMenuOpen] = useState(false);
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false);
  const [remoteWorkspaceOpen, setRemoteWorkspaceOpen] = useState(false);
  const [viewMenuOpen, setViewMenuOpen] = useState(false);
  const [derivedLedgerOpen, setDerivedLedgerOpen] = useState(false);
  const [sourceDetailsOpen, setSourceDetailsOpen] = useState(false);
  const [familiarViewsOpen, setFamiliarViewsOpen] = useState(false);
  const [ensembleViewActive, setEnsembleViewActive] = useState(false);
  const [investigationViewActive, setInvestigationViewActive] = useState(false);
  const [researchViewActive, setResearchViewActive] = useState(false);
  const [researchRecipeRevision,setResearchRecipeRevision]=useState(0);
  useEffect(()=>{const changed=()=>setResearchRecipeRevision(value=>value+1);window.addEventListener(RESEARCH_RECIPE_CHANGE_EVENT,changed);return()=>window.removeEventListener(RESEARCH_RECIPE_CHANGE_EVENT,changed);},[]);
  const [healthReportOpen, setHealthReportOpen] = useState(false);
  const [recentPanelOpen, setRecentPanelOpen] = useState(false);
  const [resourcePanelOpen, setResourcePanelOpen] = useState(false);
  const [expressionEditorOpen, setExpressionEditorOpen] = useState(false);
  const [projectsPanelOpen, setProjectsPanelOpen] = useState(showProjectNavigator);
  const [providerPanelOpen, setProviderPanelOpen] = useState(false);
  const headerRef = useRef<HTMLElement>(null);
  useStudioHeaderMenus(headerRef, dataset ? `${datasetSourceIdentity(dataset)}:${showProjectNavigator && projectsPanelOpen}` : null);
  const [toolPanelTarget, setToolPanelTarget] = useState<string | null>(null);
  useEffect(() => {
    const body = headerRef.current?.parentElement?.querySelector<HTMLElement>(".plot-studio__tool-panels-body");
    const target = toolPanelTarget ? body?.querySelector<HTMLElement>(toolPanelTarget) : null;
    if (body && target) body.scrollTop += target.getBoundingClientRect().top - body.getBoundingClientRect().top;
  }, [toolPanelTarget, sourceDetailsOpen, healthReportOpen, projectsPanelOpen, recentPanelOpen,
    remoteWorkspaceOpen, resourcePanelOpen, expressionEditorOpen, providerPanelOpen, derivedLedgerOpen]);
  const toolsPanelsOpen = sourceDetailsOpen || healthReportOpen || projectsPanelOpen
    || recentPanelOpen || remoteWorkspaceOpen || resourcePanelOpen || expressionEditorOpen
    || providerPanelOpen || derivedLedgerOpen;
  const closeToolPanels = () => {
    setSourceDetailsOpen(false);
    setHealthReportOpen(false);
    setProjectsPanelOpen(false);
    setRecentPanelOpen(false);
    setRemoteWorkspaceOpen(false);
    setResourcePanelOpen(false);
    setExpressionEditorOpen(false);
    setProviderPanelOpen(false);
    setDerivedLedgerOpen(false);
    headerRef.current?.querySelector<HTMLElement>(".plot-studio__tools-menu summary")?.focus();
  };
  const toggleResearchWorkspace = () => {
    setPlaying(false);
    setResearchViewActive(active => !active);
    setInvestigationViewActive(false);
    setComparisonViewActive(false);
    setEnsembleViewActive(false);
    setToolsMenuOpen(false);
  };
  const [externalProviders, setExternalProviders] = useState<ExternalDerivedProvider[]>(() =>
    typeof window === "undefined" ? [] : loadExternalDerivedProviders(window.localStorage).providers
  );
  const [projectState, setProjectState] = useState(() =>
    typeof window === "undefined" ? { projects: [], warnings: [] } : loadStudioProjects(window.localStorage)
  );
  const projectStateRef = useRef(projectState);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(() =>
    typeof window === "undefined"
      ? null
      : loadActiveStudioProjectId(window.localStorage, projectState.projects),
  );
  const [expressionDefinitions, setExpressionDefinitions] = useState<ExpressionDerivedFieldDefinition[]>([]);
  const [recentDatasets, setRecentDatasets] = useState<RecentDatasetEntry[]>(() =>
    typeof window === "undefined" ? [] : loadRecentDatasets(window.localStorage)
  );
  const recentDatasetsRef = useRef(recentDatasets);
  const [reopeningRecentIdentity, setReopeningRecentIdentity] = useState<string | null>(null);
  const [workspaceMessage, setWorkspaceMessage] = useState<string | null>(null);
  const [cm1Handoff, setCm1Handoff] = useState(() =>
    typeof window === "undefined" ? null : loadCm1GuidedHandoff(window.localStorage),
  );
  const workspaceFileInputRef = useRef<HTMLInputElement | null>(null);
  const sessionRestoreAttemptedRef = useRef(false);
  const ensembleInitialRestoreRef = useRef<string | null>(null);
  const ensembleLoadGenRef = useRef(0);
  const projectRestoreRef = useRef<StudioProject | null>(null);
  const projectAutoRestoreAttemptedRef = useRef(false);
  const runDir = dataset?.locator.path ?? null;
  const [steps, setSteps] = useState<StepInfo[]>([]);
  const [netcdfMetadata, setNetcdfMetadata] = useState<NetcdfMetadata | null>(null);
  const [runPublicationGeneration, setRunPublicationGeneration] =
    useState<string | null>(null);
  const [openEpoch, setOpenEpoch] = useState(0);
  const [hasTrajectories, setHasTrajectories] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [plotType, setPlotType] = useState<PlotType>("slice");
  const [renderFidelityMode, setRenderFidelityMode] =
    useState<RenderFidelityMode>(DEFAULT_RENDER_FIDELITY_MODE);
  const [category, setCategory] = useState<PlotCategory>("plan-views");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // How many steps were dropped as incomplete (torn / still-writing) on the
  // last load — drives the integrity banner.
  const [incompleteCount, setIncompleteCount] = useState(0);
  const [frameDisplay, setFrameDisplay] = useState<TaggedFrameDisplayReport | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(2);
  const [playbackStride, setPlaybackStride] = useState(1);
  const [playbackTiming, setPlaybackTiming] = useState<"frames" | "simulation">("frames");
  const [simulationRate, setSimulationRate] = useState(120);
  const simulationTimeline = useMemo(() => simulationPlaybackTimeline(steps), [steps]);
  useEffect(() => {
    if (playbackTiming === "simulation" && !simulationTimeline) {
      setPlaying(false);
      setPlaybackTiming("frames");
    }
  }, [playbackTiming, simulationTimeline]);
  const [recorderOpen, setRecorderOpen] = useState(false);
  const recorderTriggerRef = useRef<HTMLButtonElement>(null);
  const [recordingAnimation, setRecordingAnimation] = useState(false);
  // Active preset (radar / thermo / dynamics). Set on pill click;
  // cleared when the active widget reports a manual control change.
  const [activePreset, setActivePreset] = useState<PlotPreset | null>(null);
  const applyPreset = useCallback((preset: PlotPreset) => {
    setComparisonViewActive(false);
    setEnsembleViewActive(false);
    setInvestigationViewActive(false);
    setResearchViewActive(false);
    setActivePreset(preset);
    setPlotType(preset.plotType);
  }, []);
  const clearActivePreset = useCallback(() => setActivePreset(null), []);
  const plotViewRef = useRef<PlotViewSelection>({
    category,
    plotType,
    presetId: activePreset?.id ?? null,
  });
  plotViewRef.current = { category, plotType, presetId: activePreset?.id ?? null };
  const plotViewHistoryRef = useRef<PlotViewHistory>(EMPTY_PLOT_VIEW_HISTORY);
  const [plotViewHistoryRevision, setPlotViewHistoryRevision] = useState(0);
  const resetPlotViewHistory = useCallback(() => {
    plotViewHistoryRef.current = EMPTY_PLOT_VIEW_HISTORY;
    setPlotViewHistoryRevision((revision) => revision + 1);
  }, []);
  const applyPlotViewSelection = useCallback((selection: PlotViewSelection) => {
    setPlaying(false);
    setComparisonViewActive(false);
    setEnsembleViewActive(false);
    setInvestigationViewActive(false);
    setResearchViewActive(false);
    setCategory(selection.category);
    setPlotType(selection.plotType);
    setActivePreset(
      selection.presetId
        ? PLOT_PRESETS.find(({ id }) => id === selection.presetId) ?? null
        : null,
    );
  }, []);
  const selectPlotView = useCallback((selection: PlotViewSelection) => {
    const current = plotViewRef.current;
    const nextHistory = recordPlotView(
      plotViewHistoryRef.current,
      current,
      selection,
    );
    if (nextHistory === plotViewHistoryRef.current) return;
    plotViewHistoryRef.current = nextHistory;
    setPlotViewHistoryRevision((revision) => revision + 1);
    applyPlotViewSelection(selection);
  }, [applyPlotViewSelection]);
  const undoPlotViewSelection = useCallback(() => {
    const move = undoPlotView(plotViewHistoryRef.current, plotViewRef.current);
    if (!move.selection) return false;
    plotViewHistoryRef.current = move.history;
    setPlotViewHistoryRevision((revision) => revision + 1);
    applyPlotViewSelection(move.selection);
    return true;
  }, [applyPlotViewSelection]);
  const redoPlotViewSelection = useCallback(() => {
    const move = redoPlotView(plotViewHistoryRef.current, plotViewRef.current);
    if (!move.selection) return false;
    plotViewHistoryRef.current = move.history;
    setPlotViewHistoryRevision((revision) => revision + 1);
    applyPlotViewSelection(move.selection);
    return true;
  }, [applyPlotViewSelection]);
  const canUndoPlotView = plotViewHistoryRevision >= 0 &&
    plotViewHistoryRef.current.past.length > 0;
  const canRedoPlotView = plotViewHistoryRevision >= 0 &&
    plotViewHistoryRef.current.future.length > 0;
  const cm1AutoStartEpochRef = useRef<number | null>(null);
  // Re-read on each tick so the loop sees a growing step list during playback.
  const stepIdxRef = useRef(0);
  const stepsLenRef = useRef(0);
  const stepsRef = useRef<StepInfo[]>([]);
  const runPublicationGenerationRef = useRef<string | null>(null);
  const playingRef = useRef(false);
  // Monotonic token: last-requested load wins; stale inventories cannot clobber newer selections.
  const loadGenRef = useRef(0);
  const comparisonLoadGenRef = useRef(0);
  const frameDisplayRef = useRef<TaggedFrameDisplayReport | null>(null);
  // Companion panes report the frame they have actually drawn, keyed by pane id. Playback waits on
  // every pane that can report, so a fast pane cannot race ahead of a slow one in a split view.
  const companionFrameDisplayRef = useRef(new Map<string, TaggedFrameDisplayReport>());
  const paneGateRef = useRef<{ layout: PaneLayout; companionTypes: readonly PlotType[] }>({
    layout: "single",
    companionTypes: [],
  });
  const gateHeldSinceRef = useRef<number | null>(null);
  const presentationDwellRef = useRef<PresentationDwellState>(EMPTY_PRESENTATION_DWELL);
  // Frame reports wake the adaptive playback gate immediately. While a pane
  // is quiet, the gate backs off instead of rebuilding pane metadata at 20 Hz.
  const playbackGateWakeRef = useRef<(() => void) | null>(null);
  const activeViewRef = useRef({
    runDir,
    plotType,
    runPublicationGeneration,
    openEpoch,
  });
  activeViewRef.current = {
    runDir,
    plotType,
    runPublicationGeneration,
    openEpoch,
  };
  useEffect(() => {
    stepIdxRef.current = stepIdx;
  }, [stepIdx]);
  useEffect(() => {
    stepsLenRef.current = steps.length;
    stepsRef.current = steps;
    // Keyed on the array object: augmentation and live scan replace elements without changing length.
  }, [steps]);
  useEffect(() => {
    runPublicationGenerationRef.current = runPublicationGeneration;
  }, [runPublicationGeneration]);
  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);
  // Lets the plots bypass the scrub debounce while frames auto-advance (see playbackContext).
  // Declared here rather than beside its <PlaybackProvider> because that sits below an early return.
  const playbackContextValue = useMemo(() => {
    const previous = stepIdx - Math.min(playbackStride, Math.max(1, steps.length - 1));
    const duration = playbackTiming === "simulation" && simulationTimeline && previous >= 0
      ? simulationPlaybackIntervalMs(simulationTimeline, previous, playbackStride, simulationRate, MINIMUM_PLAYBACK_INTERVAL_MS)
      : 1000 / playSpeed;
    const frameInterval = playbackTiming === "simulation" && simulationTimeline
      ? simulationPlaybackIntervalMs(simulationTimeline, stepIdx, playbackStride, simulationRate, MINIMUM_PLAYBACK_INTERVAL_MS)
      : 1000 / playSpeed;
    return { playing, recording: recordingAnimation, interpolationDurationMs: duration ?? 500,
      frameIntervalMs: frameInterval ?? 1000 / playSpeed };
  }, [playing, recordingAnimation, stepIdx, steps.length, playbackStride, playbackTiming, simulationTimeline, simulationRate, playSpeed]);

  const updateRecentDatasets = useCallback((update: (current: readonly RecentDatasetEntry[]) => RecentDatasetEntry[]) => {
    const next = update(recentDatasetsRef.current);
    recentDatasetsRef.current = next;
    setRecentDatasets(next);
    if (!saveRecentDatasets(window.localStorage, next)) {
      setWorkspaceMessage("Recent datasets changed for this session, but local save is unavailable.");
    }
  }, []);

  const updateProjects = useCallback((update: (current: readonly StudioProject[]) => StudioProject[]) => {
    const current = projectStateRef.current.projects;
    const projects = update(current);
    if (projects.length === current.length && projects.every((project, index) => project === current[index])) return;
    try {
      saveStudioProjects(window.localStorage, projects);
    } catch (reason) {
      const message = `Could not save project changes: ${reason instanceof Error ? reason.message : String(reason)}`;
      const failed = { projects: [...current], warnings: [message] };
      projectStateRef.current = failed;
      setProjectState(failed);
      setWorkspaceMessage(message);
      throw new Error(message);
    }
    const next = { projects, warnings: [] };
    projectStateRef.current = next;
    setProjectState(next);
  }, []);

  useEffect(() => {
    setProjectsPanelOpen(showProjectNavigator);
  }, [showProjectNavigator]);

  const rememberOpenedDataset = useCallback((source: LocalDatasetSource) => {
    updateRecentDatasets((current) => rememberRecentDataset(current, source));
  }, [updateRecentDatasets]);

  // Auto-advance loop; loops at end of steps (most users compare fields in a cycle).
  useEffect(() => {
    if (!playing) return;
    let cancelled = false;
    let timeoutId: number | null = null;
    let gateRetryMs = PLAYBACK_GATE_INITIAL_RETRY_MS;

    const schedule = (delayMs: number) => {
      if (cancelled) return;
      if (timeoutId != null) window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(checkGate, Math.min(60_000, Math.max(0, Math.ceil(delayMs))));
    };

    const checkGate = () => {
      timeoutId = null;
      if (cancelled) return;
      const len = stepsLenRef.current;
      if (len <= 1) {
        schedule(1000);
        return;
      }
      const view = activeViewRef.current;
      const selected = stepsRef.current[stepIdxRef.current] ?? null;
      const gate = paneGateRef.current;
      const panes = visiblePaneTypes(gate.layout, view.plotType, gate.companionTypes)
        .map((paneType, index) => ({
          plotType: paneType,
          report: index === 0
            ? frameDisplayRef.current
            : companionFrameDisplayRef.current.get(`companion-${index - 1}`),
        }));
      const minimumMs = MINIMUM_PLAYBACK_INTERVAL_MS;
      const intervalMs = playbackTiming === "simulation" && simulationTimeline
        ? simulationPlaybackIntervalMs(simulationTimeline, stepIdxRef.current, playbackStride, simulationRate, minimumMs)
        : Math.max(minimumMs, 1000 / playSpeed);
      if (intervalMs == null) { setPlaying(false); return; }
      const allPanesReady = panesReadyForAdvance(
        panes,
        view,
        selected && {
          stepIndex: selected.index,
          publicationGeneration: selected.publicationGeneration,
        },
      );

      const now = performance.now();
      if (!allPanesReady) {
        presentationDwellRef.current = EMPTY_PRESENTATION_DWELL;
        if (gateHeldSinceRef.current == null) gateHeldSinceRef.current = now;
        const activeLoad = panesAreLoadingFrame(
          panes,
          view,
          selected && {
            stepIndex: selected.index,
            publicationGeneration: selected.publicationGeneration,
          },
        );
        const maxHold = activeLoad
          ? PLAYBACK_ACTIVE_LOAD_MAX_HOLD_MS
          : PLAYBACK_GATE_MAX_HOLD_MS;
        const remainingHoldMs = maxHold - (now - gateHeldSinceRef.current);
        if (remainingHoldMs > 0) {
          schedule(Math.min(gateRetryMs, remainingHoldMs));
          gateRetryMs = Math.min(PLAYBACK_GATE_MAX_RETRY_MS, gateRetryMs * 2);
          return;
        }
        setPlaying(false);
        setWorkspaceMessage("Playback paused while waiting for the selected frame. Resolve the plot error or retry playback.");
        return;
      } else {
        gateRetryMs = PLAYBACK_GATE_INITIAL_RETRY_MS;
        const alreadyDisplayedMs = Math.min(...panes.map(pane => pane.report?.interpolatedDwellMs ?? 0));
        const remainingDwellMs = Math.max(0, intervalMs - Math.max(0, alreadyDisplayedMs));
        const dwell = updatePresentationDwell(
          presentationDwellRef.current,
          selected && {
            stepIndex: selected.index,
            publicationGeneration: selected.publicationGeneration,
          },
          true,
          now,
          remainingDwellMs,
        );
        presentationDwellRef.current = dwell.state;
        if (!dwell.advance) {
          const readySinceMs = dwell.state.readySinceMs ?? now;
          schedule(Math.max(0, remainingDwellMs - (now - readySinceMs)));
          return;
        }
      }
      gateHeldSinceRef.current = null;
      presentationDwellRef.current = EMPTY_PRESENTATION_DWELL;

      const stride = Math.min(playbackStride, len - 1);
      const next = (stepIdxRef.current + stride) % len;
      stepIdxRef.current = next;
      setStepIdx(next);
      schedule(0);
    };

    playbackGateWakeRef.current = () => schedule(0);
    schedule(0);
    return () => {
      cancelled = true;
      if (timeoutId != null) window.clearTimeout(timeoutId);
      playbackGateWakeRef.current = null;
      gateHeldSinceRef.current = null;
      presentationDwellRef.current = EMPTY_PRESENTATION_DWELL;
    };
  }, [playing, playSpeed, playbackStride, playbackTiming, simulationRate, simulationTimeline]);

  // Warm the upcoming steps during playback; cleared on pause so speculative reads don't run at rest.
  useEffect(() => {
    if (!playing || !runDir) {
      setPlaybackPrefetchHint(null);
      return;
    }
    setPlaybackPrefetchHint({
      runDir,
      publicationGeneration: runPublicationGeneration,
      upcoming: upcomingSteps(steps, stepIdx, playbackStride),
    });
  }, [playing, runDir, runPublicationGeneration, steps, stepIdx, playbackStride]);
  useEffect(() => () => setPlaybackPrefetchHint(null), []);

  const loadDataset = useCallback(async (source: LocalDatasetSource, restoringProject: StudioProject | null = null) => {
    const gen = ++loadGenRef.current;
    projectRestoreRef.current = restoringProject;
    resetPlotViewHistory();
    // New view identity even if path/generation is unchanged; clears retained data immediately.
    setOpenEpoch(gen);
    primaryDatasetRef.current = source;
    setDataset(source);
    setRunPublicationGeneration(null);
    setSteps([]);
    setNetcdfMetadata(null);
    setExpressionDefinitions([]);
    frameDisplayRef.current = null;
    setFrameDisplay(null);
    setStepIdx(0);
    setHasTrajectories(false);
    setIncompleteCount(0);
    setPlaying(false);
    setEnsembleViewActive(false);
    setInvestigationViewActive(false);
    setResearchViewActive(false);
    setRecorderOpen(false);
    setLoading(true);
    setError(null);
    runPublicationGenerationRef.current = null;
    stepsRef.current = [];
    stepsLenRef.current = 0;
    stepIdxRef.current = 0;
    // Drop field cache on every deliberate open; frames re-enter only after Rust validates the manifest.
    clearFieldCache();
    try {
      if (
        source.kind === "cm1_netcdf" &&
        source.caseInventory?.historyFiles.length === 0
      ) {
        rememberOpenedDataset(source);
        return true;
      }
      let inspectedNetcdfMetadata: NetcdfMetadata | undefined;
      if (source.kind === "cm1_netcdf") {
        const metadata = await inspectNetcdfMetadata(source);
        if (gen !== loadGenRef.current) return;
        inspectedNetcdfMetadata = metadata;
        setNetcdfMetadata(metadata);
      }
      const found = await listDatasetFrames(source, undefined, inspectedNetcdfMetadata);
      if (gen !== loadGenRef.current) return; // newer load superseded this
      // Drop frames that failed the publication contract; legacy frames stay with a compatibility warning.
      const complete = filterCompleteSteps(found);
      const identity = datasetSourceIdentity(source);
      const expressionOptions = expressionCatalogSets(complete);
      const restoredExpressions = restoringProject
        && datasetSourceIdentity(restoringProject.primaryDataset) === identity
        ? restoringProject.expressionDefinitions
        : null;
      const storedExpressions = restoredExpressions
        ? { definitions: [...restoredExpressions], warnings: [] }
        : loadExpressionDerivedFields(window.localStorage, identity, expressionOptions);
      registerExpressionDerivedFields(source.locator.path, storedExpressions.definitions);
      const expressionCatalogued = complete.map((step) => augmentStepWithExpressionDerivedFields(step, storedExpressions.definitions));
      const catalogued = expressionCatalogued.map((step) => augmentStepWithExternalProviders(step, externalProviders, externalProviderRuntimeStatus));
      registerExternalDerivedProviders(source.locator.path, identity, catalogued, externalProviders.filter(({ enabled }) => enabled));
      const publicationGeneration =
        found.find((step) => step.publicationGeneration != null)
          ?.publicationGeneration ?? null;
      setRunPublicationGeneration(publicationGeneration);
      setSteps(catalogued);
      setExpressionDefinitions(storedExpressions.definitions);
      if (storedExpressions.warnings.length) setWorkspaceMessage(storedExpressions.warnings.join(" "));
      setIncompleteCount(found.length - complete.length);
      rememberOpenedDataset(source);
      setFrameDisplay(null);
      setStepIdx(0);
      runPublicationGenerationRef.current = publicationGeneration;
      stepsRef.current = catalogued;
      stepsLenRef.current = catalogued.length;
      stepIdxRef.current = 0;
      return true;
    } catch (e) {
      if (gen !== loadGenRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      if (gen === loadGenRef.current) setLoading(false);
    }
  }, [externalProviders, rememberOpenedDataset, resetPlotViewHistory]);

  const reopenRecentDataset = useCallback(async (entry: RecentDatasetEntry) => {
    setReopeningRecentIdentity(entry.datasetIdentity);
    setError(null);
    try {
      const source = entry.selection.kind === "cm1_case_folder"
        ? await reopenCm1CaseFolder(entry.selection.paths[0])
        : entry.source;
      await loadDataset(source);
      setRecentPanelOpen(false);
    } finally {
      setReopeningRecentIdentity(null);
    }
  }, [loadDataset]);

  const pinRecentDataset = useCallback((entry: RecentDatasetEntry, pinned: boolean) => {
    updateRecentDatasets((current) => setRecentDatasetPinned(current, entry.datasetIdentity, pinned));
  }, [updateRecentDatasets]);

  const removeRecentDataset = useCallback((entry: RecentDatasetEntry) => {
    updateRecentDatasets((current) => forgetRecentDataset(current, entry.datasetIdentity));
  }, [updateRecentDatasets]);

  const loadComparisonDataset = useCallback(async (source: LocalDatasetSource) => {
    const gen = ++comparisonLoadGenRef.current;
    setComparisonViewActive(false);
    setEnsembleViewActive(false);
    setComparisonDataset(source);
    setComparisonSteps([]);
    setComparisonError(null);
    setComparisonLoading(true);
    try {
      const found = await listDatasetFrames(source);
      if (gen !== comparisonLoadGenRef.current) return false;
      const complete = filterCompleteSteps(found);
      if (complete.length === 0) throw new Error("Comparison dataset has no complete plottable frames.");
      setComparisonSteps(complete);
      return true;
    } catch (error) {
      if (gen !== comparisonLoadGenRef.current) return false;
      setComparisonError(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      if (gen === comparisonLoadGenRef.current) setComparisonLoading(false);
    }
  }, []);

  const loadEnsembleDataset = useCallback(async (source: LocalDatasetSource) => {
    const gen = ++ensembleLoadGenRef.current;
    const primaryGeneration = loadGenRef.current;
    setEnsembleLoading(true);
    setEnsembleError(null);
    try {
      const found = await listDatasetFrames(source);
      if (gen !== ensembleLoadGenRef.current || primaryGeneration !== loadGenRef.current) return false;
      const complete = filterCompleteSteps(found);
      updateEnsembleMembers(current => admitEnsembleMember(primaryDatasetRef.current, current, { source, steps: complete }));
      return true;
    } catch (reason) {
      if (gen === ensembleLoadGenRef.current && primaryGeneration === loadGenRef.current) setEnsembleError(reason instanceof Error ? reason.message : String(reason));
      return false;
    } finally {
      if (gen === ensembleLoadGenRef.current) setEnsembleLoading(false);
    }
  }, [updateEnsembleMembers]);

  const onFrameDisplayChange = useCallback(
    (report: FrameDisplayReport) => {
      if (!runDir) return;
      const next: TaggedFrameDisplayReport = {
        ...report,
        runDir,
        plotType,
        runPublicationGeneration,
        openEpoch,
      };
      frameDisplayRef.current = next;
      playbackGateWakeRef.current?.();
      setFrameDisplay((previous) => {
        if (
          previous?.runDir === next.runDir &&
          previous.plotType === next.plotType &&
          previous.runPublicationGeneration === next.runPublicationGeneration &&
          previous.openEpoch === next.openEpoch &&
          previous.requestedStepIndex === next.requestedStepIndex &&
          previous.requestedPublicationGeneration ===
            next.requestedPublicationGeneration &&
          previous.renderedStepIndex === next.renderedStepIndex &&
          previous.renderedPublicationGeneration ===
            next.renderedPublicationGeneration &&
          previous.loading === next.loading
        ) {
          return previous;
        }
        return next;
      });
    },
    [runDir, plotType, runPublicationGeneration, openEpoch],
  );

  const onOpen = useCallback(async (
    kind: DatasetSourceKind,
    mode: "files" | "folder" = "files",
  ) => {
    setError(null);
    try {
      const source = await openLocalDatasetSource(kind, mode);
      if (!source) return;
      clearComparisonSession(window.localStorage);
      ensembleLoadGenRef.current += 1;
      updateEnsembleMembers([]);
      setEnsembleViewActive(false);
      setEnsembleError(null);
      await loadDataset(source);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [loadDataset, updateEnsembleMembers]);

  const onOpenComparison = useCallback(async (kind: DatasetSourceKind) => {
    setCompareMenuOpen(false);
    setComparisonError(null);
    try {
      const source = await openLocalDatasetSource(kind, "files");
      if (source) await loadComparisonDataset(source);
    } catch (error) {
      setComparisonError(error instanceof Error ? error.message : String(error));
    }
  }, [loadComparisonDataset]);

  const openCm1HandoffComparison = useCallback(async () => {
    const record = loadCm1GuidedHandoff(window.localStorage);
    if (!record || record.run.status !== "complete") {
      setWorkspaceMessage("The CM1 handoff is not comparison-ready. Complete the staged Nimbus smoke run first.");
      return;
    }
    let revalidated = record;
    try {
      const entries = await discoverRunLibrary(effectiveRunsDirectory(loadStudioSettings().runsDirectory));
      const expectedPath = record.run.outputDir.replace(/[\\/]+$/, "");
      const matching = entries.find((entry) => entry.outputDir.replace(/[\\/]+$/, "") === expectedPath) ?? null;
      revalidated = reconcileCm1SmokeRun(record, matching);
    } catch {
      revalidated = reconcileCm1SmokeRun(record, null);
    }
    saveCm1GuidedHandoff(window.localStorage, revalidated);
    setCm1Handoff(revalidated);
    const pair = cm1HandoffComparisonPair(revalidated);
    if (!pair) {
      setWorkspaceMessage(
        revalidated.run.status === "pending"
          ? revalidated.run.pendingReason
          : "Durable CM1 handoff evidence changed; comparison remains blocked.",
      );
      return;
    }
    setWorkspaceMessage("Verifying the CM1 history and matching Nimbus output…");
    const primaryReady = await loadDataset(pair.left);
    if (!primaryReady) return;
    const comparisonReady = await loadComparisonDataset(pair.right);
    if (!comparisonReady) return;
    // Persist and disclose only after both loaders succeed and alignment evidence is recomputed.
    saveComparisonSession(window.localStorage, createComparisonSession(pair.left, pair.right));
    const opened = markCm1ComparisonOpened(revalidated);
    saveCm1GuidedHandoff(window.localStorage, opened);
    setCm1Handoff(opened);
    setWorkspaceMessage("CM1 and Nimbus outputs verified. Review the alignment report before opening synchronized differences.");
  }, [loadComparisonDataset, loadDataset]);

  const onOpenEnsembleMember = useCallback(async (kind: DatasetSourceKind) => {
    setEnsembleMenuOpen(false);
    setEnsembleError(null);
    try {
      const source = await openLocalDatasetSource(kind, "files");
      if (source) await loadEnsembleDataset(source);
    } catch (reason) {
      setEnsembleError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [loadEnsembleDataset]);

  const closeComparison = useCallback(() => {
    comparisonLoadGenRef.current += 1;
    setComparisonDataset(null);
    setComparisonSteps([]);
    setComparisonError(null);
    setComparisonLoading(false);
    setComparisonViewActive(false);
  }, []);

  const closeEnsemble = useCallback(() => {
    ensembleLoadGenRef.current += 1;
    updateEnsembleMembers([]);
    setEnsembleLoading(false);
    setEnsembleError(null);
    setEnsembleViewActive(false);
  }, [updateEnsembleMembers]);

  const removeEnsembleMember = useCallback((identity: string) => {
    updateEnsembleMembers((current) => current.filter((member) => datasetSourceIdentity(member.source) !== identity));
    setEnsembleViewActive(false);
  }, [updateEnsembleMembers]);

  const forgetSavedComparison = useCallback(() => {
    clearComparisonSession(window.localStorage);
    setWorkspaceMessage("Saved comparison pair forgotten; the open analysis is unchanged.");
  }, []);

  const exportComparisonWorkspace = useCallback(() => {
    if (!dataset || !comparisonDataset) return;
    const alignment = alignDatasets(dataset, steps, comparisonDataset, comparisonSteps);
    const recipe = loadComparisonWorkspace(window.localStorage, dataset.locator.path, comparisonDataset.locator.path);
    const workspace = createComparisonPublicationWorkspace({ left: dataset, right: comparisonDataset, alignment, recipe });
    downloadComparisonPublication(serializeComparisonPublication(workspace));
    setWorkspaceMessage("Comparison recipe exported. Source data remain external and must be reachable at the recorded local paths.");
  }, [comparisonDataset, comparisonSteps, dataset, steps]);

  const importComparisonWorkspaceFile = useCallback(async (file: File) => {
    let generation = loadGenRef.current;
    const current = () => generation === loadGenRef.current;
    setWorkspaceMessage(null);
    try {
      const workspace = parseComparisonPublication(await file.text());
      if (!current()) return;
      if (workspace.recipe) {
        saveComparisonWorkspace(window.localStorage, {
          ...workspace.recipe,
          leftIdentity: workspace.left.locator.path,
          rightIdentity: workspace.right.locator.path,
        });
      }
      const restored = await openRequiredDatasetSources({
        sources: [{ role: "primary", source: workspace.left }, { role: "comparison", source: workspace.right }], current,
        load: (role, source) => {
          if (role !== "primary") return loadComparisonDataset(source);
          const pending = loadDataset(source);
          generation = loadGenRef.current;
          return pending;
        },
      });
      if (!restored) return;
      saveComparisonSession(window.localStorage, createComparisonSession(workspace.left, workspace.right));
      setComparisonViewActive(false); // Live report from reopened datasets is the sole authority for differences.
      setWorkspaceMessage(`Imported “${workspace.title}”; alignment is recomputed from the reopened sources before differences can open.`);
    } catch (reason) {
      if (!current()) return;
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [loadComparisonDataset, loadDataset]);

  // Auto-load on source change; identity gates incidental rerenders, generation gates explicit reopens.
  useEffect(() => {
    if (initialDataset) {
      void loadDataset(initialDataset);
      return;
    }
    if (initialRunDir) void loadDataset(createLocalDatasetSource("nimbus_run", initialRunDir));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialDatasetIdentity, initialDatasetOpenGeneration, initialRunDir]);

  useEffect(() => {
    if (initialComparisonDataset) void loadComparisonDataset(initialComparisonDataset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialComparisonDatasetIdentity]);

  useEffect(() => {
    if (!initialEnsembleDatasets?.length || !initialEnsembleDatasetIdentity) return;
    if (ensembleInitialRestoreRef.current === initialEnsembleDatasetIdentity) return;
    ensembleInitialRestoreRef.current = initialEnsembleDatasetIdentity;
    void (async () => {
      for (const source of initialEnsembleDatasets) await loadEnsembleDataset(source);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialEnsembleDatasetIdentity]);

  useEffect(() => {
    if (sessionRestoreAttemptedRef.current) return;
    sessionRestoreAttemptedRef.current = true;
    if (initialDataset || initialRunDir || initialComparisonDataset) return;
    const session = loadComparisonSession(window.localStorage);
    if (!session) return;
    setWorkspaceMessage("Reopening the last saved comparison pair…");
    void (async () => {
      let generation = loadGenRef.current;
      const restored = await openRequiredDatasetSources({
        sources: [{ role: "primary", source: session.left }, { role: "comparison", source: session.right }],
        current: () => generation === loadGenRef.current,
        load: (role, source) => {
          if (role !== "primary") return loadComparisonDataset(source);
          const pending = loadDataset(source);
          generation = loadGenRef.current;
          return pending;
        },
      });
      if (restored) setWorkspaceMessage("Last comparison pair reopened from local source references.");
    })().catch((reason: unknown) => setWorkspaceMessage(`Could not restore the saved comparison: ${reason instanceof Error ? reason.message : String(reason)}`));
  }, [initialComparisonDataset, initialDataset, initialRunDir, loadComparisonDataset, loadDataset]);

  useEffect(() => {
    if (!dataset || !comparisonDataset || comparisonSteps.length === 0) return;
    try { saveComparisonSession(window.localStorage, createComparisonSession(dataset, comparisonDataset)); } catch { /* storage is optional */ }
  }, [comparisonDataset, comparisonSteps.length, dataset]);

  // Clamp stepIdx if the step list shrinks.
  useEffect(() => {
    setStepIdx((prev) => Math.min(prev, Math.max(0, steps.length - 1)));
  }, [steps]);

  // Probe trajectories once per run; adds a sentinel so the Trajectories category gates like any field.
  useEffect(() => {
    let cancelled = false;
    setHasTrajectories(false);
    if (!dataset) return;
    void datasetHasTrajectories(dataset, runPublicationGeneration).then((has) => {
      if (!cancelled) setHasTrajectories(has);
    });
    return () => {
      cancelled = true;
    };
  }, [dataset, runPublicationGeneration, openEpoch]);

  // Union of fields across all steps so categories never flicker out; per-widget "not exported" is the per-step guard.
  const available = useMemo(() => {
    const fields = availableFields(steps);
    if (runDir) addPlotFieldCapabilities(fields, runDir, steps);
    if (hasTrajectories) fields.add(TRAJECTORY_CAPABILITY); // run-level capability treated as a field
    return fields;
  }, [runDir, steps, hasTrajectories]);
  const expressionCatalog = useMemo(() => expressionCatalogSets(steps), [steps]);
  const updateExpressionDefinitions = useCallback(async (definitions: readonly ExpressionDerivedFieldDefinition[]) => {
    if (!dataset) return;
    const identity = datasetSourceIdentity(dataset);
    saveExpressionDerivedFields(window.localStorage, identity, definitions, expressionCatalog);
    registerExpressionDerivedFields(dataset.locator.path, definitions);
    const nextSteps = stepsRef.current.map((step) => augmentStepWithExpressionDerivedFields(step, definitions));
    setExpressionDefinitions([...definitions]);
    setSteps(nextSteps);
    stepsRef.current = nextSteps;
    stepsLenRef.current = nextSteps.length;
    setWorkspaceMessage(`${definitions.length} dataset-local expression ${definitions.length === 1 ? "field is" : "fields are"} registered. Definitions are versioned; source arrays remain external.`);
  }, [dataset, expressionCatalog]);

  const updateExternalProviders = useCallback(async (providers: readonly ExternalDerivedProvider[]) => {
    saveExternalDerivedProviders(window.localStorage, providers);
    const nextProviders = [...providers];
    setExternalProviders(nextProviders);
    if (!dataset) return;
    const baseSteps = stepsRef.current.map((step) => {
      const externalIds = new Set(Object.entries(step.fieldOrigins ?? {})
        .filter(([, origin]) => origin === "external_provider")
        .map(([field]) => field));
      return {
        ...step,
        fields: step.fields.filter((field) => !externalIds.has(field)),
        fieldUnits: Object.fromEntries(Object.entries(step.fieldUnits).filter(([field]) => !externalIds.has(field))),
        fieldOrigins: Object.fromEntries(Object.entries(step.fieldOrigins ?? {}).filter(([, origin]) => origin !== "external_provider")),
      };
    });
    const nextSteps = baseSteps.map((step) => augmentStepWithExternalProviders(step, nextProviders, externalProviderRuntimeStatus));
    registerExternalDerivedProviders(
      dataset.locator.path,
      datasetSourceIdentity(dataset),
      nextSteps,
      nextProviders.filter(({ enabled }) => enabled),
    );
    setSteps(nextSteps);
    stepsRef.current = nextSteps;
    stepsLenRef.current = nextSteps.length;
    setWorkspaceMessage(`${nextProviders.filter(({ enabled }) => enabled).length} external derived provider ${nextProviders.filter(({ enabled }) => enabled).length === 1 ? "is" : "are"} enabled. Python outputs are validated before entering the field cache; native plugins remain unavailable.`);
  }, [dataset]);

  const currentProjectAnalysis = useCallback((): StudioProjectAnalysisSnapshot => ({
    frameIndex: stepIdx,
    fieldName: activePreset?.field ?? null,
    category,
    plotType,
    presetId: activePreset?.id ?? null,
  }), [activePreset, category, plotType, stepIdx]);

  const currentProjectWorkspace = useCallback((): StudioProjectWorkspaceSnapshot => {
    const visiblePanels = [
      projectsPanelOpen ? "projects" : null,
      healthReportOpen ? "data-health" : null,
      recentPanelOpen ? "recent-datasets" : null,
      resourcePanelOpen ? "resource-budget" : null,
      expressionEditorOpen ? "derived-fields" : null,
      investigationViewActive ? "investigation" : null,
      researchViewActive ? "research" : null,
      comparisonViewActive ? "comparison" : null,
    ].filter((panel): panel is StudioProjectWorkspaceSnapshot["visiblePanels"][number] => panel != null);
    const identity = dataset ? datasetSourceIdentity(dataset) : null;
    const comparison = dataset && comparisonDataset
      ? loadComparisonWorkspace(window.localStorage, dataset.locator.path, comparisonDataset.locator.path)
      : null;
    const researchRecipe=identity?window.localStorage.getItem(`nimbus:research-workspace:v1:${identity}`):null;
    return {
      activePanel: visiblePanels.at(-1) ?? null,
      visiblePanels,
      frameRange: steps.length ? [0, steps.length - 1] : null,
      investigation: identity ? loadInvestigationWorkspace(window.localStorage, identity) : null,
      comparison,
      rendererState: researchRecipe?{researchRecipe}:{},
    };
  }, [comparisonDataset, comparisonViewActive, dataset, expressionEditorOpen, healthReportOpen, investigationViewActive, researchViewActive, researchRecipeRevision, projectsPanelOpen, recentPanelOpen, resourcePanelOpen, steps.length]);

  const createCurrentProject = useCallback(() => {
    if (!dataset) throw new Error("Open a dataset before creating a project.");
    const project = createStudioProject({
      name: `${dataset.displayName} analysis`,
      primaryDataset: dataset,
      comparisonDataset,
      ensembleDatasets: ensembleMembers.map(({ source }) => source),
      analysis: currentProjectAnalysis(),
      workspace: currentProjectWorkspace(),
      expressionDefinitions,
    });
    updateProjects((current) => upsertStudioProject(current, project));
    setActiveProjectId(project.id);
    saveActiveStudioProjectId(window.localStorage, project.id);
    setProjectsPanelOpen(true);
    setWorkspaceMessage(`Created project “${project.name}”. Dataset fields remain at their original local paths.`);
  }, [comparisonDataset, currentProjectAnalysis, currentProjectWorkspace, dataset, ensembleMembers, expressionDefinitions, updateProjects]);

  const exportStudioProject = useCallback((project: StudioProject) => {
    const contents = serializeStudioProjectPublication(createStudioProjectPublication(project));
    downloadStudioProjectPublication(contents, project.name);
    setWorkspaceMessage(`Exported “${project.name}” as a portable analysis recipe. Field arrays are not embedded.`);
  }, []);

  const importStudioProject = useCallback((contents: string) => {
    const publication = parseStudioProjectPublication(contents);
    const project = materializeStudioProjectPublication(publication);
    updateProjects((current) => upsertStudioProject(current, project));
    setActiveProjectId(null);
    saveActiveStudioProjectId(window.localStorage, null);
    setProjectsPanelOpen(true);
    setWorkspaceMessage(`Imported “${project.name}” as a new project. Relink any source paths that do not exist on this workstation.`);
  }, [updateProjects]);

  const relinkStudioProject = useCallback(async (project: StudioProject, role: StudioProjectSourceRole) => {
    const expected = role.kind === "primary"
      ? project.primaryDataset
      : role.kind === "comparison"
        ? project.comparisonDataset
        : project.ensembleDatasets[role.index];
    if (!expected) throw new Error("The selected project source no longer exists.");
    const replacement = await openLocalDatasetSource(expected.kind, "files");
    if (!replacement) return;
    const rebound = rebindStudioProjectSource(project, role, replacement);
    updateProjects((current) => upsertStudioProject(current, rebound));
    if (activeProjectId === project.id) {
      setActiveProjectId(null);
      saveActiveStudioProjectId(window.localStorage, null);
    }
    const label = role.kind === "ensemble" ? `ensemble member ${role.index + 1}` : `${role.kind} source`;
    setWorkspaceMessage(`Relinked ${label} for “${project.name}”. Reopen the project to reindex the replacement data.`);
  }, [activeProjectId, updateProjects]);

  const verifyStudioProject = useCallback(async (project: StudioProject) => {
    return summarizeStudioProjectSourceAudit(await verifyStudioProjectSources(project));
  }, []);

  // Delete project record and its orphaned scripting editor recovery stores (up to 512 kB each).
  const deleteProject = useCallback((project: StudioProject) => {
    const remaining = projectState.projects.filter((entry) => entry.id !== project.id);
    updateProjects((current) => deleteStudioProject(current, project.id));
    try {
      for (const key of pythonWorkspaceKeysForProject(Object.keys(window.localStorage), project.id)) {
        window.localStorage.removeItem(key);
      }
      if (projectState.warnings.length === 0) {
        const live = new Set(remaining.map((entry) => entry.id));
        for (const key of orphanedPythonWorkspaceKeys(Object.keys(window.localStorage), live)) {
          window.localStorage.removeItem(key);
        }
      }
    } catch {
      // Storage blocked; project record is gone, unreclaimed store is quota cost only.
    }
    if (activeProjectId === project.id) {
      setActiveProjectId(null);
      saveActiveStudioProjectId(window.localStorage, null);
    }
  }, [activeProjectId, projectState.projects, projectState.warnings.length, updateProjects]);

  const openStudioProject = useCallback(async (project: StudioProject) => {
    projectRestoreRef.current = project;
    let primaryGeneration: number | null = null;
    const current = () => projectRestoreRef.current === project
      && (primaryGeneration == null || primaryGeneration === loadGenRef.current);
    setActiveProjectId(null);
    setPlaying(false);
    setProjectsPanelOpen(false);
    setHealthReportOpen(project.workspace.visiblePanels.includes("data-health"));
    setRecentPanelOpen(project.workspace.visiblePanels.includes("recent-datasets"));
    setResourcePanelOpen(project.workspace.visiblePanels.includes("resource-budget"));
    setExpressionEditorOpen(project.workspace.visiblePanels.includes("derived-fields"));
    setInvestigationViewActive(project.workspace.visiblePanels.includes("investigation"));
    setComparisonViewActive(false);
    try {
      if (typeof project.workspace.rendererState.researchRecipe === "string") {
        const identity = datasetSourceIdentity(project.primaryDataset);
        const recipe = parseResearchRecipe(project.workspace.rendererState.researchRecipe, identity);
        window.localStorage.setItem(`nimbus:research-workspace:v1:${identity}`, JSON.stringify(recipe));
      }
      if (project.workspace.investigation) saveInvestigationWorkspace(window.localStorage, project.workspace.investigation);
      if (project.workspace.comparison) saveComparisonWorkspace(window.localStorage, project.workspace.comparison);
      closeComparison();
      closeEnsemble();
      const primaryLoad = loadDataset(project.primaryDataset, project);
      primaryGeneration = loadGenRef.current;
      const opened = await primaryLoad;
      if (!current()) return;
      if (!opened) throw new Error(`Project source '${project.primaryDataset.locator.path}' could not be reopened. Reconnect or relocate the local data, then try again.`);
      setCategory(project.analysis.category ?? "plan-views");
      setPlotType(project.analysis.plotType ?? "slice");
      setActivePreset(project.analysis.presetId
        ? PLOT_PRESETS.find(({ id }) => id === project.analysis.presetId) ?? null
        : null);
      setStepIdx(project.analysis.frameIndex);
      const restored = await openRequiredDatasetSources({
        sources: [
          ...(project.comparisonDataset ? [{ role: "comparison" as const, source: project.comparisonDataset }] : []),
          ...project.ensembleDatasets.map(source => ({ role: "ensemble" as const, source })),
        ], current,
        load: (role, source) => role === "comparison" ? loadComparisonDataset(source) : loadEnsembleDataset(source),
      });
      if (!restored) return;
      updateProjects((current) => markStudioProjectOpened(current, project.id));
      setActiveProjectId(project.id);
      saveActiveStudioProjectId(window.localStorage, project.id);
      setComparisonViewActive(project.comparisonDataset != null && project.workspace.visiblePanels.includes("comparison"));
      setResearchViewActive(project.workspace.visiblePanels.includes("research"));
      setInvestigationViewActive(!project.workspace.visiblePanels.includes("research") && project.workspace.visiblePanels.includes("investigation"));
      setWorkspaceMessage(`Opened project “${project.name}”. Sources were reindexed and saved analysis state was restored.`);
      onProjectOpened?.();
    } catch (reason) {
      if (!current()) return;
      saveActiveStudioProjectId(window.localStorage, null);
      throw reason;
    } finally {
      if (current()) projectRestoreRef.current = null;
    }
  }, [closeComparison, closeEnsemble, loadComparisonDataset, loadDataset, loadEnsembleDataset, onProjectOpened, updateProjects]);

  // Resume the last named project on return to Analysis, unless a more specific dataset/run intent was supplied.
  useEffect(() => {
    if (projectAutoRestoreAttemptedRef.current || showProjectNavigator || !restoreActiveProject) return;
    if (initialDataset || initialRunDir || dataset || !activeProjectId) {
      projectAutoRestoreAttemptedRef.current = true;
      return;
    }
    projectAutoRestoreAttemptedRef.current = true;
    const project = projectState.projects.find(({ id }) => id === activeProjectId);
    if (!project) {
      setActiveProjectId(null);
      saveActiveStudioProjectId(window.localStorage, null);
      return;
    }
    void openStudioProject(project).catch((reason: unknown) => {
      setWorkspaceMessage(`Project resume stopped: ${reason instanceof Error ? reason.message : String(reason)}`);
    });
  }, [activeProjectId, dataset, initialDataset, initialRunDir, openStudioProject, projectState.projects, restoreActiveProject, showProjectNavigator]);

  const openBundledProjectExample = useCallback(async (example: BundledProjectExample) => {
    const kind = example.acceptedSourceKinds[0];
    const source = example.openMode === "open_packaged_dataset"
      ? createPackagedAtmosphericExampleSource(example.id)
      : await openLocalDatasetSource(kind, "files");
    if (!source) return;
    const project = instantiateBundledProjectExample(example, source);
    updateProjects((current) => upsertStudioProject(current, project));
    await openStudioProject(project);
  }, [openStudioProject, updateProjects]);

  useEffect(() => {
    // Playback position is transient while the clock is running. Persist the
    // final displayed frame when playback pauses instead of rewriting the full
    // recoverable project journal for every animation frame.
    if (!activeProjectId || !dataset || loading || playing) return;
    const timeout = window.setTimeout(() => {
      try {
        updateProjects((current) => updateStudioProjectSnapshot(current, activeProjectId, {
          analysis: currentProjectAnalysis(),
          workspace: currentProjectWorkspace(),
          expressionDefinitions,
        }));
      } catch (reason) {
        setWorkspaceMessage(`Project autosave paused: ${reason instanceof Error ? reason.message : String(reason)}`);
      }
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [activeProjectId, currentProjectAnalysis, currentProjectWorkspace, dataset, expressionDefinitions, loading, playing, updateProjects]);
  const visibleCats = useMemo(
    () => visibleCategories(available, PLOT_PRESETS),
    [available],
  );
  const catPlotTypes = useMemo(
    () => visiblePlotTypes(category, available),
    [category, available],
  );
  const catPresets = useMemo(
    () => visiblePresets(category, available, PLOT_PRESETS),
    [category, available],
  );
  const cm1FamiliarViews = useMemo(
    () => resolveCm1FamiliarViews(available),
    [available],
  );
  const defaultSliceAxis = categoryMeta(category).defaultSliceAxis;

  const onPickType = useCallback(
    (t: PlotType) => {
      selectPlotView({ category, plotType: t, presetId: null });
    },
    [category, selectPlotView],
  );

  const onSelectCm1FamiliarView = useCallback(
    (view: ResolvedCm1FamiliarView, record = true) => {
      if (!view.available) return;
      if (view.kind === "preset" && view.preset) {
        if (record) {
          selectPlotView({
            category: view.preset.category,
            plotType: view.preset.plotType,
            presetId: view.preset.id,
          });
        } else {
          setCategory(view.preset.category);
          applyPreset(view.preset);
        }
        return;
      }
      if (view.kind === "renderer") {
        const selection = {
          category: view.category,
          plotType: view.plotType,
          presetId: null,
        } satisfies PlotViewSelection;
        if (record) selectPlotView(selection);
        else applyPlotViewSelection(selection);
      }
    },
    [applyPlotViewSelection, applyPreset, selectPlotView],
  );

  useEffect(() => {
    if (
      loading ||
      dataset?.kind !== "cm1_netcdf" ||
      steps.length === 0 ||
      cm1AutoStartEpochRef.current === openEpoch
    ) return;
    cm1AutoStartEpochRef.current = openEpoch;
    const firstReady = cm1FamiliarViews.find((view) => view.available);
    if (firstReady) onSelectCm1FamiliarView(firstReady, false);
  }, [cm1FamiliarViews, dataset?.kind, loading, onSelectCm1FamiliarView, openEpoch, steps.length]);

  const onSelectCategory = useCallback(
    (next: PlotCategory) => {
      const types = visiblePlotTypes(next, available);
      if (types.length > 0) {
        // Keep the current renderer if it belongs to the new category, else
        // land on the first supported one.
        const nextType = types.some((m) => m.value === plotType)
          ? plotType
          : types[0].value;
        selectPlotView({ category: next, plotType: nextType, presetId: null });
      } else {
        // Product-only category (Radar): land on its first available product.
        const firstPreset = visiblePresets(next, available, PLOT_PRESETS)[0];
        if (firstPreset) {
          selectPlotView({
            category: firstPreset.category,
            plotType: firstPreset.plotType,
            presetId: firstPreset.id,
          });
        }
      }
    },
    [available, plotType, selectPlotView],
  );

  useEffect(() => {
    if (!dataset || investigationViewActive || researchViewActive) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || isEditableTarget(event.target)) return;
      const modifier = event.metaKey !== event.ctrlKey && (event.metaKey || event.ctrlKey);
      if (!modifier || event.key.toLocaleLowerCase() !== "z") return;
      const changed = event.shiftKey
        ? redoPlotViewSelection()
        : undoPlotViewSelection();
      if (changed) event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [dataset, investigationViewActive, researchViewActive, redoPlotViewSelection, undoPlotViewSelection]);

  // Re-home category/plot type when a run change makes the current selection unsupported.
  useEffect(() => {
    if (loading || steps.length === 0) return;
    let cat = category;
    if (!visibleCats.some((c) => c.value === cat)) {
      const next = firstAvailableCategory(available, PLOT_PRESETS);
      if (next) {
        cat = next;
        setCategory(next);
      }
    }
    if (!isPlotAvailable(plotType, available)) {
      const nextType = firstAvailablePlotType(cat, available);
      if (nextType) {
        setPlotType(nextType);
        clearActivePreset();
      } else {
        const firstPreset = visiblePresets(cat, available, PLOT_PRESETS)[0];
        if (firstPreset) applyPreset(firstPreset);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [available, loading, steps.length]);

  // Queue status drives the informational banner only; computed above the early return to keep hook order stable.
  const inProgress = useMemo(
    () =>
      !!runDir &&
      dataset?.kind === "nimbus_run" &&
      runDir === initialRunDir &&
      (initialRunStatus === "running" || initialRunStatus === "queued"),
    [dataset?.kind, runDir, initialRunDir, initialRunStatus],
  );
  const datasetTimeHealth = useMemo(
    () => dataset?.kind === "cm1_netcdf" ? analyzeDatasetTime(steps) : null,
    [dataset?.kind, steps],
  );
  const datasetAlignment = useMemo(
    () => dataset && comparisonDataset && comparisonSteps.length > 0
      ? alignDatasets(dataset, steps, comparisonDataset, comparisonSteps)
      : null,
    [comparisonDataset, comparisonSteps, dataset, steps],
  );
  // Primary handle is always index 0; comparison appears only after its steps load.
  const plotDatasets = useMemo<PlotDatasetHandle[]>(() => {
    if (!dataset) return [];
    const handles: PlotDatasetHandle[] = [{
      id: datasetSourceIdentity(dataset),
      role: "primary",
      displayName: dataset.displayName,
      runDir: dataset.locator.path,
      steps,
    }];
    if (comparisonDataset && comparisonSteps.length > 0) {
      handles.push({
        id: datasetSourceIdentity(comparisonDataset),
        role: "comparison",
        displayName: comparisonDataset.displayName,
        runDir: comparisonDataset.locator.path,
        steps: comparisonSteps,
      });
    }
    return handles;
  }, [dataset, steps, comparisonDataset, comparisonSteps]);

  const ensembleInputs = useMemo<EnsembleMemberInput[]>(
    () => dataset ? [{ source: dataset, steps }, ...ensembleMembers] : [],
    [dataset, ensembleMembers, steps],
  );
  const researchMembers = useMemo(() => {
    const members = [...ensembleInputs];
    if (comparisonDataset && !members.some(member => datasetSourceIdentity(member.source) === datasetSourceIdentity(comparisonDataset))) members.push({source: comparisonDataset, steps: comparisonSteps});
    return members;
  }, [ensembleInputs, comparisonDataset, comparisonSteps]);
  const ensembleAlignment = useMemo(
    () => ensembleMembers.length > 0 ? alignEnsembleMembers(ensembleInputs) : null,
    [ensembleInputs, ensembleMembers.length],
  );
  const comparisonTimeMatch = datasetAlignment?.time.matches.find(
    (match) => match.leftFrameIndex === stepIdx,
  ) ?? null;

  const getDisplayedFrameIdentity = useCallback(
    (): DisplayedFrameIdentity | null => {
      const report = frameDisplayRef.current;
      const view = activeViewRef.current;
      return report == null ||
        report.runDir !== view.runDir ||
        report.plotType !== view.plotType ||
        report.runPublicationGeneration !== view.runPublicationGeneration ||
        report.openEpoch !== view.openEpoch
        ? null
        : {
            renderedStepIndex: report.renderedStepIndex,
            renderedPublicationGeneration: report.renderedPublicationGeneration,
            loading: report.loading,
          };
    },
    [],
  );

  // Poll for new frames while the run is in progress; freeze while the recorder is capturing.
  useEffect(() => {
    if (!inProgress || !dataset || recorderOpen) return;
    const expectedLoadGeneration = loadGenRef.current;
    let cancelled = false;
    let scanInFlight = false;

    const scan = async () => {
      if (scanInFlight) return;
      scanInFlight = true;
      try {
        const found = await listDatasetFrames(dataset);
        if (
          cancelled ||
          expectedLoadGeneration !== loadGenRef.current
        ) return;
        const result = reconcileLiveSteps(
          stepsRef.current,
          found,
          stepIdxRef.current,
          runPublicationGenerationRef.current,
          !playingRef.current,
        );

        if (result.generationChanged) {
          clearFieldCache();
          setOpenEpoch((epoch) => epoch + 1);
          frameDisplayRef.current = null;
          setFrameDisplay(null);
          setHasTrajectories(false);
        }
        runPublicationGenerationRef.current = result.publicationGeneration;
        stepsRef.current = result.steps;
        stepsLenRef.current = result.steps.length;
        stepIdxRef.current = result.selectedIndex;
        setRunPublicationGeneration(result.publicationGeneration);
        setSteps(result.steps);
        setStepIdx(result.selectedIndex);
        setIncompleteCount(result.incompleteCount);
      } catch (reason) {
        // Transient filesystem errors (queued/writing) retry silently; persistent errors surface and stop polling.
        if (!isTransientAppError(reason) && !cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason));
          stopScanning();
        }
      } finally {
        scanInFlight = false;
      }
    };

    let interval: number | null = window.setInterval(() => void scan(), 3000);
    function stopScanning(): void {
      if (interval != null) {
        window.clearInterval(interval);
        interval = null;
      }
    }
    return () => {
      cancelled = true;
      stopScanning();
    };
  }, [inProgress, dataset, recorderOpen]);

  const captureFrameIdentities = useMemo(() => steps.map(step => ({
    stepIndex: step.index, publicationGeneration: step.publicationGeneration,
  })), [steps]);

  // Per-frame recording timeout: scaled by cell count, clamped [8 s, 60 s].
  const recordMaxFrameWaitMs = useMemo(() => {
    const g = steps[0]?.grid ?? null;
    const cells = g ? (g.nr ?? 0) * (g.nth ?? 0) * (g.nz ?? 0) : 0;
    const BASE_MS = 8000;
    const MS_PER_MCELL = 1500; // ~1.5 s per million cells
    const scaled = BASE_MS + (cells / 1_000_000) * MS_PER_MCELL;
    return Math.max(BASE_MS, Math.min(60_000, Math.round(scaled)));
  }, [steps]);

  // All hooks below MUST stay above the no-run early return; hook count changes across that boundary throw React #310.
  // Multi-pane: pane 0 keeps full history/presets; companions carry only a plot type.
  const [paneLayout, setPaneLayout] = useState<PaneLayout>("single");
  const [cameraLocked, setCameraLocked] = useState(true);
  const [companionPaneTypes, setCompanionPaneTypes] = useState<PlotType[]>(
    () => ["volume", "isosurface", "streamlines"],
  );
  // Presets handed to a companion by "View in 3D", indexed alongside companionPaneTypes.
  const [companionPresets, setCompanionPresets] = useState<(PlotPreset | null)[]>(
    () => [null, null, null],
  );
  // What the primary 2D pane currently shows, published by the widget itself.
  const [scalarSelection, setScalarSelection] = useState<ScalarSelection | null>(null);
  /*
   * Field linking across panes. Off by default: panes exist to compare different quantities,
   * and linking is the narrower case of following one quantity through several renderers.
   * Deliberately independent of the camera lock — one is about what is shown, the other about
   * where it is seen from.
   */
  const [fieldsLinked, setFieldsLinked] = useState(false);
  const [linkedField, setLinkedField] = useState<string | null>(null);
  const onPaneFieldChange = useCallback((field: string | null) => {
    // Only user-initiated changes arrive here; adopting a link never re-broadcasts, so two
    // linked panes cannot drive each other.
    setLinkedField((previous) => (fieldsLinked ? field : previous));
  }, [fieldsLinked]);
  const toggleFieldsLinked = useCallback(() => {
    setFieldsLinked((linked) => {
      // Seed from the pane the user is looking at, so enabling the link pulls the others to it
      // rather than waiting for the next field change.
      if (!linked) setLinkedField(scalarSelection?.field ?? null);
      return !linked;
    });
  }, [scalarSelection]);
  const activeLinkedField = fieldsLinked ? linkedField : null;
  const handoffSequenceRef = useRef(0);

  const setCompanionPaneType = useCallback((index: number, next: PlotType) => {
    setCompanionPaneTypes((previous) => {
      const updated = [...previous];
      updated[index] = next;
      return updated;
    });
    // The handed-over preset described the renderer being replaced, so it does
    // not describe the new one.
    setCompanionPresets((previous) => {
      if (previous[index] == null) return previous;
      const updated = [...previous];
      updated[index] = null;
      return updated;
    });
  }, []);

  /** Open the current 2D scalar as a volume render in the companion pane, splitting if needed. */
  const handOffToVolume = useCallback(() => {
    if (!scalarSelection) return;
    handoffSequenceRef.current += 1;
    const preset = volumeHandoffPreset(scalarSelection, handoffSequenceRef.current);
    setPaneLayout((previous) => (previous === "single" ? "1x2" : previous));
    setCompanionPaneTypes((previous) => {
      const updated = [...previous];
      updated[0] = "volume";
      return updated;
    });
    setCompanionPresets((previous) => {
      const updated = [...previous];
      updated[0] = preset;
      return updated;
    });
  }, [scalarSelection]);

  // Read by the adaptive auto-advance scheduler, which holds no React state of its own.
  paneGateRef.current = { layout: paneLayout, companionTypes: companionPaneTypes };
  useEffect(() => {
    playbackGateWakeRef.current?.();
  }, [paneLayout, plotType, companionPaneTypes]);

  const onCompanionFrameDisplayChange = useCallback(
    (paneId: string, paneType: PlotType, report: FrameDisplayReport) => {
      if (!runDir) return;
      companionFrameDisplayRef.current.set(paneId, {
        ...report,
        runDir,
        plotType: paneType,
        runPublicationGeneration,
        openEpoch,
      });
      playbackGateWakeRef.current?.();
    },
    [runDir, runPublicationGeneration, openEpoch],
  );

  // The gate rejects a report whose plot type no longer matches its pane, so this is about the map
  // itself: without it, every type a pane has ever shown stays resident for the session.
  useEffect(() => {
    const live = paneLayout === "single" ? 0 : paneCount(paneLayout) - 1;
    for (const paneId of [...companionFrameDisplayRef.current.keys()]) {
      const index = Number(paneId.slice("companion-".length));
      const report = companionFrameDisplayRef.current.get(paneId);
      if (index >= live || report?.plotType !== companionPaneTypes[index]) {
        companionFrameDisplayRef.current.delete(paneId);
      }
    }
  }, [paneLayout, companionPaneTypes]);

  useEffect(() => {
    companionFrameDisplayRef.current.clear();
  }, [runDir, runPublicationGeneration, openEpoch]);

  // First 3D pane owns the compass; null when no pane is 3D. Above the early return (hook-count rule).
  const sharedOverlayPaneId = useMemo(() => {
    if (paneLayout === "single") return "primary";
    if (THREE_D_PLOT_TYPES.has(plotType)) return "primary";
    const companions = companionPaneTypes.slice(0, paneCount(paneLayout) - 1);
    const index = companions.findIndex((type) => THREE_D_PLOT_TYPES.has(type));
    return index >= 0 ? `companion-${index}` : null;
  }, [paneLayout, plotType, companionPaneTypes]);

  const cameraSyncBusRef = useRef<{ key: string; bus: CameraSyncBus }>({
    key: "initial",
    bus: createCameraSyncBus(),
  });

  const [immersive, setImmersive] = useState(false);
  const immersiveAvailable = THREE_D_PLOT_TYPES.has(plotType) && !investigationViewActive && !researchViewActive;
  const immersiveActive = immersive && immersiveAvailable;
  useEffect(() => {
    // Leaving the 3D plots (or opening the investigation workspace) must not
    // strand the user in a chrome-less view with no way back to the controls.
    if (!immersiveAvailable && immersive) setImmersive(false);
  }, [immersiveAvailable, immersive]);
  useEffect(() => {
    if (!immersiveActive) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setImmersive(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [immersiveActive]);

  // Panes are disabled in comparison/ensemble/investigation views, which own the full canvas. Both
  // live above the early return: the memo below is a hook, and this flag feeds it.
  const panesActive = paneLayout !== "single"
    && !investigationViewActive && !researchViewActive
    && !comparisonViewActive
    && !ensembleViewActive
    && !immersiveActive;

  // Every lock session starts clean. Reusing the bus's last camera across an
  // unlock, run switch, or authority change can make a historical view win the
  // next synchronization handshake.
  const cameraSyncSessionKey = [
    panesActive,
    cameraLocked,
    sharedOverlayPaneId,
    runDir,
    runPublicationGeneration,
    openEpoch,
  ].join("|");
  if (cameraSyncBusRef.current.key !== cameraSyncSessionKey) {
    cameraSyncBusRef.current = {
      key: cameraSyncSessionKey,
      bus: createCameraSyncBus(),
    };
  }
  const cameraSyncBus = cameraSyncBusRef.current.bus;

  /* Escape returns to one figure. Registered only while immersive is inactive, which owns
   * Escape for its own exit, so a split entered from immersive unwinds a step at a time. */
  useEffect(() => {
    if (!panesActive) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPaneLayout("single");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [panesActive]);

  const liveCompanionTypes = panesActive
    ? companionPaneTypes.slice(0, paneCount(paneLayout) - 1)
    : [];
  const hasFrameDrivenPanes = metaFor(plotType).perFrame || liveCompanionTypes.some(type => metaFor(type).perFrame);
  useEffect(() => { if (!hasFrameDrivenPanes) setPlaying(false); }, [hasFrameDrivenPanes]);
  const threeDBudgetScale = activeThreeDPaneBudgetScale([
    THREE_D_PLOT_TYPES.has(plotType),
    ...liveCompanionTypes.map((type) => THREE_D_PLOT_TYPES.has(type)),
  ]);
  const presentationPaneIds = useMemo(() => {
    if (!panesActive) return [];
    return [plotType, ...companionPaneTypes.slice(0, paneCount(paneLayout) - 1)]
      .flatMap((type, index) => ["volume", "isosurface", "streamlines", "supercell"].includes(type)
        ? [index === 0 ? "primary" : `companion-${index - 1}`] : []);
  }, [panesActive, plotType, companionPaneTypes, paneLayout]);
  const presentationIdentity = JSON.stringify([runDir, runPublicationGeneration, openEpoch,
    steps[stepIdx]?.index, steps[stepIdx]?.publicationGeneration]);

  // Stable identity: a fresh object here re-renders every pane consumer on every studio render,
  // and context updates cut straight through memoized subtrees.
  const primaryPaneValue = useMemo(
    () => ({
      bus: panesActive ? cameraSyncBus : null,
      paneId: "primary",
      cameraLocked: panesActive && cameraLocked,
      cameraAuthority: !panesActive || sharedOverlayPaneId === "primary",
      voxelBudgetScale: panesActive && THREE_D_PLOT_TYPES.has(plotType)
        ? threeDBudgetScale
        : 1,
      ownsSharedOverlays: !panesActive || !cameraLocked || sharedOverlayPaneId === "primary",
    }),
    [panesActive, cameraLocked, plotType, sharedOverlayPaneId, cameraSyncBus, threeDBudgetScale],
  );

  if (!dataset || !runDir) {
    return (
      <main className={`plot-studio plot-studio--empty ${showProjectNavigator ? "plot-studio--projects" : ""}`}>
        <input
          ref={workspaceFileInputRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void importComparisonWorkspaceFile(file);
            event.currentTarget.value = "";
          }}
        />
        <div className="plot-studio__welcome">
          {!showProjectNavigator && <>
          <header className="plot-studio__welcome-header">
            <span className="plot-studio__eyebrow">Analysis workspace</span>
            <h2>Open data</h2>
            <p>
              Inspect an existing simulation or atmospheric dataset. Analysis
              works independently of simulator setup.
            </p>
          </header>

          <div className="plot-studio__source-list" aria-label="Dataset sources">
            {DATASET_ADAPTERS.map((adapter) =>
              adapter.availability !== "unavailable" ? (
                <div
                  key={adapter.kind}
                  className="plot-studio__source-row plot-studio__source-row--available"
                >
                  <span className="plot-studio__source-icon">
                    <Icon name="archive" size={16} />
                  </span>
                  <span className="plot-studio__source-copy">
                    <strong>{adapter.title}</strong>
                    <span>{adapter.description}</span>
                    <small className="plot-studio__source-capability">
                      {adapter.multiresolution.label} · {adapter.multiresolution.detail}
                    </small>
                  </span>
                  <span className="plot-studio__source-actions">
                    <button
                      type="button"
                      className="plot-studio__source-action-button plot-studio__source-action-button--primary"
                      onClick={() => void onOpen(adapter.kind)}
                    >
                      {adapter.kind === "cm1_netcdf" ? "Open files…" : adapter.actionLabel}
                    </button>
                    {adapter.kind === "cm1_netcdf" && (
                      <button
                        type="button"
                        className="plot-studio__source-action-button"
                        onClick={() => void onOpen(adapter.kind, "folder")}
                      >
                        Open folder…
                      </button>
                    )}
                  </span>
                </div>
              ) : (
                <div
                  key={adapter.kind}
                  className="plot-studio__source-row plot-studio__source-row--unavailable"
                  aria-disabled="true"
                >
                  <span className="plot-studio__source-icon">
                    <Icon name="folder" size={16} />
                  </span>
                  <span className="plot-studio__source-copy">
                    <strong>{adapter.title}</strong>
                    <span>{adapter.description}</span>
                    <small className="plot-studio__source-capability">
                      {adapter.multiresolution.label} · {adapter.multiresolution.detail}
                    </small>
                  </span>
                  <span className="plot-studio__source-status">
                    {adapter.unavailableReason}
                  </span>
                </div>
              ),
            )}
          </div>

          <div className="plot-studio__next-workflow">
            <span>Saved comparison</span>
            <span>Continue from a comparison workspace saved on this machine.</span>
            <button type="button" onClick={() => workspaceFileInputRef.current?.click()}>Open workspace…</button>
          </div>

          {cm1Handoff && (
            <div className="plot-studio__next-workflow">
              <span>CM1 guided handoff</span>
              <span>
                {cm1Handoff.run.status === "complete"
                  ? "The matching Nimbus smoke output is recorded. Studio will re-verify both datasets before comparison."
                  : "The reviewed CM1 source is preserved. Complete the 30 s Nimbus smoke run in Configure to unlock comparison."}
              </span>
              <button
                type="button"
                disabled={!cm1HandoffComparisonPair(cm1Handoff)}
                title={cm1HandoffComparisonPair(cm1Handoff)
                  ? "Re-verify both durable datasets and open the comparison"
                  : "Complete the source-bound Nimbus smoke run before comparison"}
                onClick={() => void openCm1HandoffComparison()}
              >
                Verify and compare
              </button>
            </div>
          )}
          </>}

          {showProjectNavigator && <ProjectNavigator
            projects={projectState.projects}
            activeProjectId={activeProjectId}
            storageWarnings={projectState.warnings}
            canCreate={false}
            onCreate={createCurrentProject}
            onOpen={openStudioProject}
            onRename={(project, name) => updateProjects((current) => renameStudioProject(current, project.id, name))}
            onDuplicate={(project) => updateProjects((current) => duplicateStudioProject(current, project.id))}
            onDelete={deleteProject}
            onExport={exportStudioProject}
            onImport={importStudioProject}
            onRelink={relinkStudioProject}
            onVerify={verifyStudioProject}
            onOpenExample={openBundledProjectExample}
          />}

          {!showProjectNavigator && recentDatasets.length > 0 && <RecentDatasetsPanel
            entries={recentDatasets}
            onReopen={reopenRecentDataset}
            onPinChange={pinRecentDataset}
            onForget={removeRecentDataset}
            reopeningIdentity={reopeningRecentIdentity}
          />}

          {error && (
            <DatasetErrorNotice message={error} onDismiss={() => setError(null)} />
          )}
        </div>
      </main>
    );
  }

  if (dataset.kind === "cm1_netcdf" && steps.length === 0) {
    return (
      <main className="plot-studio plot-studio--metadata">
        <input ref={workspaceFileInputRef} type="file" accept="application/json,.json" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void importComparisonWorkspaceFile(file); event.currentTarget.value = ""; }} />
        <header className="plot-studio__header">
          <div className="plot-studio__title-row">
            <div className="plot-studio__dataset-identity">
              <h2>{dataset.displayName}</h2>
              <span className="plot-studio__source-badge">netCDF</span>
            </div>
            <nav className="plot-studio__header-actions" aria-label="Dataset actions">
              <button type="button" className="plot-studio__open-btn" onClick={() => void onOpen("cm1_netcdf")}>Open files…</button>
              <button type="button" className="plot-studio__open-btn" onClick={() => void onOpen("cm1_netcdf", "folder")}>Open case folder…</button>
            </nav>
          </div>
          <div className="plot-studio__meta">Read-only metadata inspection · no simulator required</div>
          <Cm1CaseDetails dataset={dataset} />
        </header>
        {loading && <div className="plot-studio__loading"><Icon name="dot" size={12} /><span>Reading netCDF header…</span></div>}
        {error && <div className="plot-studio__metadata-body"><DatasetErrorNotice message={error} onDismiss={() => setError(null)} /></div>}
        {netcdfMetadata && (
          <div className="plot-studio__metadata-body">
            <section className="plot-studio__metadata-summary">
              <div><span>Container</span><strong>{netcdfMetadata.containerFormat.replaceAll("_", " ")}</strong></div>
              <div><span>Classification</span><strong>{netcdfMetadata.sourceClassification === "cm1_likely" ? "Likely CM1" : "Unclassified netCDF"}</strong></div>
              <div><span>Dimensions</span><strong>{netcdfMetadata.dimensions.length}</strong></div>
              <div><span>Variables</span><strong>{netcdfMetadata.variables.length}</strong></div>
            </section>
            <div className="plot-studio__metadata-notice"><Icon name="info" size={14} /><span>{netcdfMetadata.plotUnavailableReason}</span></div>
            <section className="plot-studio__metadata-section">
              <h3>Dimensions</h3>
              <div className="plot-studio__metadata-chips">{netcdfMetadata.dimensions.map((dimension) => <span key={dimension.name}><strong>{dimension.name}</strong> {dimension.length}{dimension.unlimited ? " records" : ""}</span>)}</div>
            </section>
            <Cm1FieldContractDetails metadata={netcdfMetadata} defaultOpen />
          </div>
        )}
      </main>
    );
  }

  const currentStep = steps[stepIdx] ?? null;
  const axisFrameVerdict = axisFrame(currentStep);
  const showFrameBar = hasFrameDrivenPanes && !researchViewActive;
  /* Offered only by the scalar-on-a-plane renderers, and only to the primary pane. Kept
   * present but disabled when the field has no vertical extent, so the reason is legible. */
  const volumeHandoff: VolumeHandoffOffer | undefined =
    VOLUMETRIC_SOURCE_TYPES.includes(plotType)
      ? {
        available: canHandOffToVolume(plotType, scalarSelection, available),
        blockedReason: describeVolumeHandoffBlock(plotType, scalarSelection, available),
        field: scalarSelection?.field ?? null,
        label: scalarSelection?.source === "composite" ? "View fill in 3D" : "View in 3D",
        onOpen: handOffToVolume,
      }
      : undefined;
  const physical3DBlockedReason = dataset.kind === "cm1_netcdf"
    && PHYSICAL_GRID_3D_PLOT_TYPES.has(plotType)
    ? currentStep?.grid?.geometryKind === "index"
      ? "This file does not provide a complete finite, increasing x/y/z coordinate set with declared length units. Two-dimensional plots remain available on explicit index axes; Studio will not invent metre-scale 3D geometry."
      : null
    : null;
  const physical3DBlocked = physical3DBlockedReason != null;
  const tracksRenderedIdentity =
    !physical3DBlocked
    && !ensembleViewActive
    && !investigationViewActive && !researchViewActive
    && (
      plotType === "slice"
      || plotType === "vector"
      || plotType === "volume"
      || plotType === "isosurface"
      || plotType === "streamlines"
      || plotType === "supercell"
    );
  const activeFrameDisplay =
    frameDisplay?.runDir === runDir &&
    frameDisplay.plotType === plotType &&
    frameDisplay.runPublicationGeneration === runPublicationGeneration &&
    frameDisplay.openEpoch === openEpoch
      ? frameDisplay
      : null;
  const renderedStep =
    activeFrameDisplay?.renderedStepIndex == null
      ? null
      : steps.find(
          (step) =>
            step.index === activeFrameDisplay.renderedStepIndex &&
            step.publicationGeneration ===
              activeFrameDisplay.renderedPublicationGeneration,
        ) ?? null;
  const frameTransition = describeFrameTransition(
    currentStep?.index ?? null,
    tracksRenderedIdentity ? (renderedStep?.index ?? null) : null,
    currentStep?.publicationGeneration ?? null,
    tracksRenderedIdentity
      ? (activeFrameDisplay?.renderedPublicationGeneration ?? null)
      : null,
  );
  const renderedOrdinal = renderedStep
    ? steps.findIndex((step) => step.index === renderedStep.index) + 1
    : 0;
  const exactFrameReady =
    frameTransition.kind === "ready" && activeFrameDisplay?.loading === false;

  // Pad to the run's max width so the readout never changes size while scrubbing.
  const ordinalWidth = String(steps.length).length;
  const timeWidth = Math.max(
    ...steps.map((step) => (step.simulationTimeS != null
      ? step.simulationTimeS.toFixed(1).length
      : 0)),
    0,
  );
  const frameSelection = steps.length === 0
    ? "selected 0 / 0"
    : `selected ${String(stepIdx + 1).padStart(ordinalWidth, " ")} / ${steps.length}`
      + (currentStep?.simulationTimeS != null
        // Repeat time at the scrubber; it may be far from the header which shows the rendered time.
        ? ` · t = ${currentStep.simulationTimeS.toFixed(1).padStart(timeWidth, " ")} s`
        : "");
  // Playback owns a stable status label. Toggling between "reading" and blank
  // on every frame made the controls shimmer even when the scene was coherent.
  const frameStatus = playing
    ? "· synchronized layers"
    : recordingAnimation
      ? "· recording exact frames"
    : tracksRenderedIdentity && renderedOrdinal > 0 && renderedOrdinal !== stepIdx + 1
    ? `· showing ${renderedOrdinal}`
    : "";

  const projectNavigator = <ProjectNavigator projects={projectState.projects} activeProjectId={activeProjectId} storageWarnings={projectState.warnings} canCreate onCreate={createCurrentProject} onOpen={openStudioProject} onRename={(project, name) => updateProjects((current) => renameStudioProject(current, project.id, name))} onDuplicate={(project) => updateProjects((current) => duplicateStudioProject(current, project.id))} onDelete={deleteProject} onExport={exportStudioProject} onImport={importStudioProject} onRelink={relinkStudioProject} onVerify={verifyStudioProject} onOpenExample={openBundledProjectExample} onClose={() => { setProjectsPanelOpen(false); if (showProjectNavigator) onProjectOpened?.(); }} />;

  /*
   * Projects is a destination, not a panel over a figure. Reached from the nav rail it owns the page;
   * rendering it in-flow above a live plot instead donated the plot's height to it inside a fixed
   * `overflow: hidden` column, so the figure and its frame bar were cut off. The Tools-menu toggle
   * still opens it as a panel below, where it sits over a plot the user asked to keep.
   */
  if (showProjectNavigator && projectsPanelOpen) {
    return (
      <main className="plot-studio plot-studio--projects-page">
        {projectNavigator}
      </main>
    );
  }

  return (
    <main className={`plot-studio ${immersiveActive ? "plot-studio--immersive" : ""}`}>
      <input ref={workspaceFileInputRef} type="file" accept="application/json,.json" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void importComparisonWorkspaceFile(file); event.currentTarget.value = ""; }} />
      <header className="plot-studio__header" ref={headerRef}>
        <div className="plot-studio__title-row">
          <div className="plot-studio__dataset-identity">
            <h2>{dataset.displayName}</h2>
            <span className="plot-studio__source-badge">
              {dataset.kind === "cm1_netcdf" ? "netCDF" : "Nimbus history"}
            </span>
          </div>
          <nav className="plot-studio__header-actions" aria-label="Dataset and workspace actions">
            <button type="button" className="plot-studio__open-btn" aria-pressed={researchViewActive}
              title="Research workspace: linked sections, profiles and saved-field diagnostics"
              onClick={toggleResearchWorkspace}>Research</button>
            <button
              type="button"
              className="plot-studio__open-btn"
              onClick={() => void onOpen(dataset.kind)}
            >
              {dataset.kind === "cm1_netcdf" ? "Open files…" : "Open data…"}
            </button>
            {dataset.kind === "cm1_netcdf" && (
              <button
                type="button"
                className="plot-studio__open-btn"
                onClick={() => void onOpen("cm1_netcdf", "folder")}
              >
                Open folder…
              </button>
            )}
          <details
            className="plot-studio__compare-menu"
            open={compareMenuOpen}
            onToggle={(event) => setCompareMenuOpen(event.currentTarget.open)}
          >
            <summary className="plot-studio__open-btn">Compare…</summary>
            <div className="plot-studio__compare-menu-popover">
              <span>Add dataset B</span>
              <button type="button" onClick={() => void onOpenComparison("cm1_netcdf")}>netCDF files…</button>
              <button type="button" onClick={() => void onOpenComparison("nimbus_run")}>Nimbus run…</button>
            </div>
          </details>
          <details
            className="plot-studio__compare-menu"
            open={ensembleMenuOpen}
            onToggle={(event) => setEnsembleMenuOpen(event.currentTarget.open)}
          >
            <summary className="plot-studio__open-btn">Ensemble…</summary>
            <div className="plot-studio__compare-menu-popover">
              <span>{ensembleMembers.length + 1 >= MAX_ENSEMBLE_MEMBERS ? `${MAX_ENSEMBLE_MEMBERS}-member limit reached` : `Add member ${ensembleMembers.length + 2}`}</span>
              <button type="button" disabled={ensembleMembers.length + 1 >= MAX_ENSEMBLE_MEMBERS} title={ensembleMembers.length + 1 >= MAX_ENSEMBLE_MEMBERS ? `The ${MAX_ENSEMBLE_MEMBERS}-member ensemble limit has been reached` : "Add a netCDF member"} onClick={() => void onOpenEnsembleMember("cm1_netcdf")}>netCDF files…</button>
              <button type="button" disabled={ensembleMembers.length + 1 >= MAX_ENSEMBLE_MEMBERS} title={ensembleMembers.length + 1 >= MAX_ENSEMBLE_MEMBERS ? `The ${MAX_ENSEMBLE_MEMBERS}-member ensemble limit has been reached` : "Add a Nimbus run member"} onClick={() => void onOpenEnsembleMember("nimbus_run")}>Nimbus run…</button>
            </div>
          </details>
          {/* Leaving a split was reachable only through the View popover, which is two clicks
            * deep for the one action a user wants the moment the comparison is over. Present
            * only while there is a split to leave; Escape does the same thing. */}
          {panesActive && (
            <button
              type="button"
              className="plot-studio__open-btn"
              onClick={() => setPaneLayout("single")}
              title="Return to a single figure (Esc)"
            >
              Exit split
            </button>
          )}
          {/* How the figure is VIEWED, as opposed to what it shows. A popover rather than a row:
            * these are a small fixed set that is rarely changed, and a permanent row costs the
            * figure height on every frame whether or not anyone touches it. */}
          <details
            className="plot-studio__compare-menu plot-studio__view-menu"
            open={viewMenuOpen}
            onToggle={(event) => setViewMenuOpen(event.currentTarget.open)}
          >
            <summary className="plot-studio__open-btn">
              View{paneLayout === "single" ? "" : ` · ${paneLayoutLabel(paneLayout)}`}
            </summary>
            <div className="plot-studio__compare-menu-popover plot-studio__view-popover">
              <span>Panes</span>
              <div className="plot-studio__panes-options" role="group" aria-label="Pane layout">
                {PANE_LAYOUTS.map((layout) => (
                  <button
                    key={layout}
                    type="button"
                    className={`plot-studio__panes-option ${
                      paneLayout === layout ? "plot-studio__panes-option--active" : ""
                    }`}
                    aria-pressed={paneLayout === layout}
                    // Closes on choice: the result is the grid behind the menu, and leaving it open
                    // parks a popover over the pane headers it just created.
                    onClick={() => { setPaneLayout(layout); setViewMenuOpen(false); }}
                    title={layout === "single"
                      ? "One figure"
                      : `${paneLayoutLabel(layout)} — compare fields at the same frame`}
                  >
                    {paneLayoutLabel(layout)}
                  </button>
                ))}
              </div>
              <button
                type="button"
                aria-pressed={cameraLocked}
                disabled={paneLayout === "single"}
                onClick={() => setCameraLocked((locked) => !locked)}
                title={paneLayout === "single"
                  ? "Camera lock applies once more than one pane is shown"
                  : "Orbiting one 3D pane orbits them all, so the comparison is of the fields rather than of two viewpoints"}
              >
                {cameraLocked ? "Cameras locked" : "Cameras free"}
              </button>
              {/* Separate from the camera lock on purpose: this is about which quantity every
                * pane shows, that one about where it is seen from. */}
              <button
                type="button"
                aria-pressed={fieldsLinked}
                disabled={paneLayout === "single"}
                onClick={toggleFieldsLinked}
                title={paneLayout === "single"
                  ? "Field linking applies once more than one pane is shown"
                  : "Changing the field in any pane moves the others to it. A pane that cannot draw that field keeps its own and says so."}
              >
                {fieldsLinked ? "Fields linked" : "Fields free"}
              </button>
              {THREE_D_PLOT_TYPES.has(plotType) && (
                <>
                  <span>3D fidelity</span>
                  <div className="plot-studio__fidelity-options" role="group" aria-label="3D rendering fidelity">
                    {(["scientific", "presentation"] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={`plot-studio__fidelity-option ${
                          renderFidelityMode === mode ? "plot-studio__fidelity-option--active" : ""
                        }`}
                        aria-pressed={renderFidelityMode === mode}
                        onClick={() => setRenderFidelityMode(mode)}
                      >
                        {mode === "scientific" ? "Scientific" : "Presentation"}
                      </button>
                    ))}
                  </div>
                  <p className="plot-studio__fidelity-hint">
                    {renderFidelityMode === "scientific"
                      ? "Resolved fields only; interpolation and renderer LOD stay disclosed."
                      : "Allows disclosed display shaping; model fields and measurements remain unchanged."}
                  </p>
                  <button
                    type="button"
                    onClick={() => { setImmersive(true); setViewMenuOpen(false); }}
                    title="Give the render the whole window (Esc to exit)"
                  >
                    Immersive
                  </button>
                </>
              )}
            </div>
          </details>
          <details
            className="plot-studio__compare-menu plot-studio__tools-menu"
            open={toolsMenuOpen}
            onToggle={(event) => setToolsMenuOpen(event.currentTarget.open)}
          >
            <summary className="plot-studio__open-btn">Tools</summary>
            <div className="plot-studio__compare-menu-popover plot-studio__tools-popover">
              <span>Analysis</span>
              <button type="button" aria-pressed={researchViewActive} onClick={toggleResearchWorkspace}>Research workspace</button>
              <button type="button" aria-pressed={investigationViewActive} onClick={() => {
                setPlaying(false);
                setComparisonViewActive(false);
                setEnsembleViewActive(false);
                setPlotType("slice");
                setResearchViewActive(false);
                setInvestigationViewActive((active) => !active);
                setToolsMenuOpen(false);
              }}>Investigation workspace</button>
              {dataset.kind === "cm1_netcdf" && <button type="button" aria-pressed={familiarViewsOpen} onClick={() => { setFamiliarViewsOpen((open) => !open); setToolsMenuOpen(false); }}>Model-oriented views</button>}
              <span>Data and fields</span>
              <button type="button" aria-pressed={healthReportOpen} onClick={() => { setHealthReportOpen((open) => !open); setToolPanelTarget(".data-health"); setToolsMenuOpen(false); }}>Data health</button>
              {dataset.kind === "cm1_netcdf" && <button type="button" aria-pressed={sourceDetailsOpen} onClick={() => { setSourceDetailsOpen((open) => !open); setToolPanelTarget(".plot-studio__source-details"); setToolsMenuOpen(false); }}>Source details</button>}
              <button type="button" aria-pressed={derivedLedgerOpen} onClick={() => { setDerivedLedgerOpen((open) => !open); setToolPanelTarget(".derived-ledger"); setToolsMenuOpen(false); }}>Field registry</button>
              <button type="button" aria-pressed={expressionEditorOpen} onClick={() => { setExpressionEditorOpen((open) => !open); setToolPanelTarget(".expression-fields"); setToolsMenuOpen(false); }}>Derived expressions</button>
              <button type="button" aria-pressed={providerPanelOpen} onClick={() => { setProviderPanelOpen((open) => !open); setToolPanelTarget(".provider-console"); setToolsMenuOpen(false); }}>External providers</button>
              <span>Workspace</span>
              <button type="button" onClick={() => { setToolsMenuOpen(false); workspaceFileInputRef.current?.click(); }}>Open workspace…</button>
              <button type="button" aria-pressed={projectsPanelOpen} onClick={() => { setProjectsPanelOpen((open) => !open); setToolPanelTarget(".project-navigator"); setToolsMenuOpen(false); }}>Projects</button>
              <button type="button" aria-pressed={recentPanelOpen} onClick={() => { setRecentPanelOpen((open) => !open); setToolPanelTarget(".recent-datasets"); setToolsMenuOpen(false); }}>Recent data</button>
              <button type="button" aria-pressed={remoteWorkspaceOpen} onClick={() => { setRemoteWorkspaceOpen(open => !open); setToolPanelTarget(".remote-workspace"); setToolsMenuOpen(false); }}>Remote visualization workspace</button>
              <button type="button" aria-pressed={resourcePanelOpen} onClick={() => { setResourcePanelOpen((open) => !open); setToolPanelTarget(".resource-budget"); setToolsMenuOpen(false); }}>Resource use</button>
              {cm1Handoff && cm1HandoffComparisonPair(cm1Handoff) && (
                <button type="button" onClick={() => { setToolsMenuOpen(false); void openCm1HandoffComparison(); }}>Open guided comparison…</button>
              )}
            </div>
          </details>
          </nav>
        </div>
        <div className="plot-studio__meta">
          <span>{steps.length} steps</span>
          {currentStep?.grid?.geometryKind === "index" && (
            <>
              <span className="plot-studio__sep">·</span>
              <span>index-coordinate inspection</span>
            </>
          )}
          {currentStep?.grid?.geometryKind === "physical"
            && currentStep.grid.dzM == null
            && currentStep.grid.scalarCoordinatesM?.[2] && (
            <>
              <span className="plot-studio__sep">·</span>
              <span>native stretched-z coordinates</span>
            </>
          )}
          {datasetTimeHealth?.expectedCadenceS != null && (
            <>
              <span className="plot-studio__sep">·</span>
              <span>{Number(datasetTimeHealth.expectedCadenceS.toPrecision(6))} s expected cadence</span>
            </>
          )}
          {showFrameBar && tracksRenderedIdentity && renderedStep?.simulationTimeS != null && (
            <>
              <span className="plot-studio__sep">·</span>
              <span>
                showing t = {renderedStep.simulationTimeS.toFixed(1)} s
                {frameTransition.kind === "stale" &&
                currentStep?.simulationTimeS != null
                  ? ` while t = ${currentStep.simulationTimeS.toFixed(1)} s loads`
                  : ""}
              </span>
            </>
          )}
          {showFrameBar && tracksRenderedIdentity && renderedStep == null && currentStep && (
            <>
              <span className="plot-studio__sep">·</span>
              <span>{frameTransition.label}</span>
            </>
          )}
          {showFrameBar && !tracksRenderedIdentity && currentStep && (
            <>
              <span className="plot-studio__sep">·</span>
              <span>selected frame {currentStep.index}</span>
            </>
          )}
        </div>
      </header>

      {error && <DatasetErrorNotice message={error} onDismiss={() => setError(null)} />}

      {comparisonLoading && (
        <div className="plot-studio__comparison-loading" role="status">
          <Icon name="dot" size={12} /><span>Indexing dataset B without loading full fields…</span>
        </div>
      )}
      {comparisonError && (
        <div className="plot-studio__comparison-error" role="alert">
          <Icon name="warning" size={12} /><span>{comparisonError}</span>
          <button type="button" onClick={closeComparison}>Dismiss</button>
        </div>
      )}
      {ensembleLoading && (
        <div className="plot-studio__comparison-loading" role="status">
          <Icon name="dot" size={12} /><span>Indexing ensemble member metadata without loading fields…</span>
        </div>
      )}
      {ensembleError && (
        <div className="plot-studio__comparison-error" role="alert">
          <Icon name="warning" size={12} /><span>{ensembleError}</span>
          <button type="button" onClick={() => setEnsembleError(null)}>Dismiss</button>
        </div>
      )}
      {workspaceMessage && (
        <div className="plot-studio__workspace-message" role="status">
          <Icon name="info" size={12} /><span>{workspaceMessage}</span><button type="button" onClick={() => setWorkspaceMessage(null)}>Dismiss</button>
        </div>
      )}
      {toolsPanelsOpen && <section className="plot-studio__tool-panels" aria-label="Tools panels" onKeyDown={event => {
        if (event.key === "Escape") { event.stopPropagation(); closeToolPanels(); }
      }}>
        <header className="plot-studio__tool-panels-heading"><strong>Tools</strong><button type="button" className="plot-studio__open-btn" onClick={closeToolPanels}>Close tools</button></header>
        <div className="plot-studio__tool-panels-body">
      {sourceDetailsOpen && dataset.kind === "cm1_netcdf" && (
        <section className="plot-studio__source-details" aria-label="Dataset source details">
          <Cm1CaseDetails dataset={dataset} />
          {netcdfMetadata && <Cm1FieldContractDetails metadata={netcdfMetadata} />}
        </section>
      )}
      {healthReportOpen && <DatasetHealthReport dataset={dataset} steps={steps} stepIdx={stepIdx} incompleteCount={incompleteCount} onClose={() => setHealthReportOpen(false)} />}
      {projectsPanelOpen && projectNavigator}
      {recentPanelOpen && <RecentDatasetsPanel entries={recentDatasets} onReopen={reopenRecentDataset} onPinChange={pinRecentDataset} onForget={removeRecentDataset} onClose={() => setRecentPanelOpen(false)} reopeningIdentity={reopeningRecentIdentity} />}
      {remoteWorkspaceOpen && <RemoteWorkspaceSettings onClose={() => setRemoteWorkspaceOpen(false)} />}
      {resourcePanelOpen && <ResourceBudgetPanel onClose={() => setResourcePanelOpen(false)} onClearCache={(previous) => setWorkspaceMessage(`Released ${previous.parsedCache.entries} parsed field ${previous.parsedCache.entries === 1 ? "entry" : "entries"}. Native reads already in progress were not interrupted.`)} />}
      {expressionEditorOpen && <ExpressionDerivedFieldEditor datasetLabel={dataset.displayName} definitions={expressionDefinitions} nativeFields={expressionCatalog.nativeFields} reservedFieldIds={expressionCatalog.reservedFieldIds} availableFields={expressionCatalog.availableFields} onDefinitionsChange={updateExpressionDefinitions} />}
      {providerPanelOpen && <ExternalDerivedProviderManager providers={externalProviders} runtimeStatus={externalProviderRuntimeStatus} onProvidersChange={updateExternalProviders} />}
      {derivedLedgerOpen && <DerivedFieldLedger dataset={dataset} steps={steps} stepIdx={stepIdx} />}
        </div>
      </section>}
      {comparisonDataset && datasetAlignment && (
        <DatasetAlignmentReport
          left={dataset}
          right={comparisonDataset}
          report={datasetAlignment}
          onClose={closeComparison}
          comparisonViewActive={comparisonViewActive}
          onOpenSynchronizedSlices={datasetAlignment.differenceAllowed ? () => {
            setPlaying(false);
            setInvestigationViewActive(false);
    setResearchViewActive(false);
            setPlotType("slice");
            setComparisonViewActive(true);
          } : undefined}
          onExportWorkspace={exportComparisonWorkspace}
          onForgetSavedPair={forgetSavedComparison}
        />
      )}
      {ensembleAlignment && (
        <EnsembleAlignmentReport
          sources={ensembleInputs.map((member) => member.source)}
          report={ensembleAlignment}
          loading={ensembleLoading}
          viewActive={ensembleViewActive}
          onOpen={ensembleAlignment.aggregateAllowed ? () => {
            setPlaying(false);
            setComparisonViewActive(false);
            setInvestigationViewActive(false);
    setResearchViewActive(false);
            setPlotType("slice");
            setEnsembleViewActive(true);
          } : undefined}
          onRemove={removeEnsembleMember}
          onClose={closeEnsemble}
        />
      )}

      {datasetTimeHealth?.hasWarnings && (
        <div className="plot-studio__in-progress" role="status">
          <Icon name="warning" size={14} />
          <div className="plot-studio__in-progress-body">
            <strong>CM1 time inventory needs attention</strong>
            <span>{datasetTimeHealth.summary}</span>
          </div>
        </div>
      )}

      {axisFrameVerdict.headline && (
        <div className="plot-studio__in-progress" role="status">
          <Icon name="warning" size={14} />
          <div className="plot-studio__in-progress-body">
            <strong>{axisFrameVerdict.headline}</strong>
            <span>{axisFrameVerdict.detail}</span>
          </div>
        </div>
      )}

      {(inProgress || incompleteCount > 0) && (
        <div className="plot-studio__in-progress" role="status">
          <Icon name="warning" size={14} />
          <div className="plot-studio__in-progress-body">
            <strong>
              {inProgress
                ? initialRunStatus === "running"
                  ? "Run still in progress"
                  : "Run hasn't started yet"
                : "Incomplete frames hidden"}
            </strong>
            <span>
              {incompleteCount > 0
                ? `${incompleteCount} incomplete or invalid frame${incompleteCount === 1 ? "" : "s"} hidden; ${steps.length} validated frame${steps.length === 1 ? "" : "s"} shown.`
                : "Only complete frames are shown; the latest appears once the simulator finishes writing it."}
            </span>
          </div>
        </div>
      )}

      {!investigationViewActive && !researchViewActive && familiarViewsOpen && dataset.kind === "cm1_netcdf" && (
        <Cm1FamiliarStart
          views={cm1FamiliarViews}
          activePresetId={activePreset?.id ?? null}
          activePlotType={plotType}
          activeCategory={category}
          onSelect={onSelectCm1FamiliarView}
        />
      )}

      {immersiveActive && (
        <button
          type="button"
          className="plot-studio__immersive-exit"
          onClick={() => setImmersive(false)}
          title="Exit immersive view (Esc)"
        >
          <Icon name="settings" size={12} />
          <span>Exit immersive</span>
          <kbd>Esc</kbd>
        </button>
      )}

      {/* Navigation stays put; only the preset list below it scrolls. */}
      {!investigationViewActive && !researchViewActive && (
      <>
      <div className="plot-studio__nav">
      <CategoryBar
        categories={visibleCats}
        active={category}
        onSelect={onSelectCategory}
      />

      <PlotTypePicker
        plotTypes={catPlotTypes}
        activePlotType={plotType}
        hasActivePreset={activePreset != null}
        onPickType={onPickType}
        canUndo={canUndoPlotView}
        canRedo={canRedoPlotView}
        onUndo={undoPlotViewSelection}
        onRedo={redoPlotViewSelection}
      />
      </div>

      {/* Toolbelt: scrollable block so the scrubber (below, flex-shrink:0) is never clipped by overflow. */}
      <div className="plot-studio__toolbelt">
      <PlotPresetPicker
        presets={catPresets}
        activePresetId={activePreset?.id ?? null}
        noPlotTypes={catPlotTypes.length === 0}
        onPickPreset={(preset) => selectPlotView({
          /* Stay on the tab the preset was clicked from. A preset listed in a
           * secondary category would otherwise bounce the user to its home
           * tab — picking "3D dBZ volume" from the 3D tab would silently
           * switch them to Radar. Every preset offered here came from
           * `catPresets`, so the active category always contains it. */
          category,
          plotType: preset.plotType,
          presetId: preset.id,
        })}
      />

      </div>

      </>
      )}

      {physical3DBlocked && (
        <div className="plot-studio__in-progress" role="status">
          <Icon name="warning" size={14} />
          <div className="plot-studio__in-progress-body">
            <strong>Physical 3D geometry unavailable</strong>
            <span>{physical3DBlockedReason}</span>
          </div>
        </div>
      )}

      {loading && (
        <div className="plot-studio__loading">
          <Icon name="dot" size={12} />
          <span>Loading step index…</span>
        </div>
      )}

      <PlaybackProvider value={playbackContextValue}>
      <PanePresentationProvider identity={presentationIdentity} paneIds={presentationPaneIds}>
      {onPanePresentationChange && <PanePresentationObserver onChange={onPanePresentationChange} />}
      <div
        className={`plot-studio__plot-area${panesActive ? " plot-studio__plot-area--panes" : ""}`}
        style={panesActive
          ? {
            gridTemplateColumns: paneGridTemplate(paneLayout).columns,
            gridTemplateRows: paneGridTemplate(paneLayout).rows,
          }
          : undefined}
      >
      <PrimaryPaneFrame active={panesActive} portrait={metaFor(plotType).portrait === true}>
      <ErrorBoundary
        key={`${datasetSourceIdentity(dataset)}:${runViewIdentityKey(
          runDir,
          runPublicationGeneration,
          openEpoch,
        )}`}
        label="This plot"
        resetKeys={[
          plotType,
          runDir,
          runPublicationGeneration ?? "legacy",
          openEpoch,
          stepIdx,
        ]}
      >
        {researchViewActive ? (
          <ResearchWorkspace dataset={dataset} steps={steps} stepIdx={stepIdx} onStepChange={setStepIdx} members={researchMembers} onClose={() => setResearchViewActive(false)} />
        ) : investigationViewActive ? (
          <>
            <RootHistoryScopeNotice runDir={runDir} step={currentStep} label="Root history · linked investigation" />
            <InvestigationWorkspace dataset={dataset} steps={steps} stepIdx={stepIdx} onStepChange={setStepIdx} />
          </>
        ) : ensembleViewActive && ensembleAlignment ? (
          <>
            <RootHistoryScopeNotice runDir={runDir} step={currentStep} label="Root histories · ensemble analysis" />
            <EnsembleAnalysisPlot
              members={ensembleInputs}
              alignment={ensembleAlignment}
              referenceFrameIndex={stepIdx}
            />
          </>
        ) : comparisonViewActive && comparisonDataset && datasetAlignment ? (
          <>
            <RootHistoryScopeNotice runDir={runDir} step={currentStep} label="Root histories · comparison analysis" />
            <ComparisonSlicePlot
              leftRunDir={runDir}
              leftDatasetIdentity={datasetAlignment.leftDatasetIdentity}
              leftLabel={dataset.displayName}
              leftSteps={steps}
              leftStepIdx={stepIdx}
              rightRunDir={comparisonDataset.locator.path}
              rightDatasetIdentity={datasetAlignment.rightDatasetIdentity}
              rightLabel={comparisonDataset.displayName}
              rightSteps={comparisonSteps}
              rightStepIdx={comparisonTimeMatch?.rightFrameIndex ?? null}
              eligibleFields={datasetAlignment.differenceEligibleFields}
              timeMatches={datasetAlignment.time.matches}
              initialField={activePreset?.field}
              onFrameDisplayChange={onFrameDisplayChange}
            />
          </>
        ) : physical3DBlocked ? (
          <div className="plot-fallback">
            <Icon name="warning" size={14} />
            <span>Select a 2D plot or provide complete physical coordinates.</span>
          </div>
        ) : (
          <PaneProvider value={primaryPaneValue}>
            <PlotRouter
              plotType={plotType}
              runDir={runDir}
              steps={steps}
              stepIdx={stepIdx}
              datasets={plotDatasets}
              preset={activePreset}
              onPresetCleared={clearActivePreset}
              defaultSliceAxis={defaultSliceAxis}
              onFrameDisplayChange={onFrameDisplayChange}
              onScalarSelectionChange={setScalarSelection}
              volumeHandoff={volumeHandoff}
              linkedField={activeLinkedField}
              onFieldChange={onPaneFieldChange}
              renderFidelityMode={renderFidelityMode}
              projectId={activeProjectId}
            />
          </PaneProvider>
        )}
      </ErrorBoundary>
      </PrimaryPaneFrame>
      {panesActive && companionPaneTypes
        .slice(0, paneCount(paneLayout) - 1)
        .map((companionType, index) => (
          <CompanionPane
            key={`companion-${index}`}
            paneId={`companion-${index}`}
            plotType={companionType}
            preset={companionPresets[index] ?? null}
            onPlotTypeChange={(next) => setCompanionPaneType(index, next)}
            available={available}
            bus={cameraSyncBus}
            cameraLocked={cameraLocked}
            cameraAuthority={sharedOverlayPaneId === `companion-${index}`}
            ownsSharedOverlays={!cameraLocked || sharedOverlayPaneId === `companion-${index}`}
            voxelBudgetScale={THREE_D_PLOT_TYPES.has(companionType) ? threeDBudgetScale : 1}
            runDir={runDir}
            steps={steps}
            stepIdx={stepIdx}
            datasets={plotDatasets}
            renderFidelityMode={renderFidelityMode}
            projectId={activeProjectId}
            onFrameDisplayChange={onCompanionFrameDisplayChange}
            linkedField={activeLinkedField}
            onFieldChange={onPaneFieldChange}
            resetKey={`${runDir}:${runPublicationGeneration ?? "legacy"}:${openEpoch}`}
          />
        ))}
      </div>
      </PanePresentationProvider>
      </PlaybackProvider>

      {showFrameBar && (
        <div className="plot-studio__frame-bar">
          <button
            type="button"
            className="plot-studio__play-btn"
            onClick={() => setPlaying((p) => !p)}
            disabled={steps.length <= 1}
            title={playing ? "Pause" : "Play"}
            aria-label={playing ? "Pause frame playback" : "Play frames"}
          >
            <Icon name={playing ? "pause" : "play"} size={14} />
          </button>
          <span className="plot-studio__playback-options">
            <select
              className="plot-studio__play-speed"
              aria-label="Playback timing"
              value={playbackTiming}
              onChange={event => setPlaybackTiming(event.target.value as "frames" | "simulation")}
              disabled={steps.length <= 1}
              title="Frame timing holds each output equally. Simulation time follows saved timestamps; playback waits for every visible pane."
            >
              <option value="frames">Frames</option>
              <option value="simulation" disabled={!simulationTimeline}>Simulation time</option>
            </select>
            {playbackTiming === "simulation" ? (
              <select
                className="plot-studio__play-speed"
                aria-label="Simulation playback speed"
                value={simulationRate}
                onChange={event => setSimulationRate(Number(event.target.value))}
                title="Target simulation seconds per real second. Playback waits for every visible pane to draw each saved frame."
              >
                {[1, 10, 30, 60, 120, 300].map(rate => <option key={rate} value={rate}>{rate}×</option>)}
              </select>
            ) : (
            <select
              className="plot-studio__play-speed"
              value={playSpeed}
              onChange={(e) => setPlaySpeed(Number(e.target.value))}
              disabled={steps.length <= 1}
              title={THREE_D_PLOT_TYPES.has(plotType)
                ? "Resolved 3D playback cadence; each displayed frame completes before advance"
                : "Playback speed in frames per second"}
              aria-label="Playback speed"
            >
              {PLAY_SPEED_OPTIONS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
            )}
            <select
              className="plot-studio__play-speed"
              value={playbackStride}
              onChange={(event) => setPlaybackStride(Number(event.target.value))}
              disabled={steps.length <= 1}
              title="Skip source frames while preserving full fidelity for every displayed frame"
              aria-label="Frame stride"
            >
              {PLAYBACK_STRIDE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </span>
          <span className="plot-studio__frame-label">Frame</span>
          <span className="plot-studio__frame-scrubber">
            <input
              type="range"
              min={0}
              max={Math.max(0, steps.length - 1)}
              value={stepIdx}
              onChange={(e) => {
                setPlaying(false);
                setStepIdx(Number(e.target.value));
              }}
              disabled={steps.length === 0}
              aria-label="Selected frame"
              aria-valuetext={steps.length === 0 ? "No frames" : `Frame ${stepIdx + 1} of ${steps.length}`}
            />
          </span>
          <span
            className="plot-studio__frame-meta"
            // The box is sized by the layout rather than by this text, so a
            // narrow window can clip it. The full reading stays recoverable.
            title={frameStatus ? `${frameSelection} ${frameStatus}` : frameSelection}
          >
            <span className="plot-studio__frame-selection">{frameSelection}</span>
            {/* Always mounted so mounting/unmounting on each frame doesn't cause layout flicker. */}
            <span className="plot-studio__frame-status">{frameStatus}</span>
          </span>
          <button
            ref={recorderTriggerRef}
            type="button"
            className="plot-studio__rec-btn"
            onClick={() => {
              setPlaying(false);
              setRecorderOpen(true);
            }}
            disabled={
              steps.length === 0 ||
              !tracksRenderedIdentity ||
              !exactFrameReady
            }
            title={
              !tracksRenderedIdentity
                ? "Recording requires a renderer with exact frame-identity reporting"
                : !exactFrameReady
                ? "Wait for the selected frame to render before recording"
                : "Export current frame or record saved frames"
            }
          >
            <Icon name="record" size={10} />
            <span>Rec</span>
          </button>
        </div>
      )}

      {recorderOpen && (
        <MovieRecorderDialog
          totalSteps={steps.length}
          currentStepIdx={stepIdx}
          setStepIdx={setStepIdx}
          maxFrameWaitMs={recordMaxFrameWaitMs}
          frameIdentities={captureFrameIdentities}
          getDisplayedFrameIdentity={getDisplayedFrameIdentity}
          // 2D Plotly (.plot-canvas) vs 3D VTK (.volume-render__canvas) canvas,
          // per the registry.
          plotCanvasSelector={metaFor(plotType).recorderSelector}
          defaultFilename={`${dataset.displayName.replace(/[^a-zA-Z0-9_-]/g, "_")}-${plotType}`}
          onRecordingChange={setRecordingAnimation}
          onClose={() => setRecorderOpen(false)}
          returnFocusRef={recorderTriggerRef}
        />
      )}
    </main>
  );
}

/**
 * Wraps the primary figure in one grid cell when panes are active; transparent otherwise.
 *
 * `flat` rather than returning bare children: swapping the element type at this position makes
 * React unmount the figure and mount a new one, so toggling split screen tore down and rebuilt the
 * plot — seconds of black canvas for a 3D scene. The flat shell keeps the type stable and hands its
 * layout back with `display: contents`, so single-pane geometry is unchanged.
 */
function PrimaryPaneFrame({ active, portrait, children }: {
  active: boolean;
  portrait: boolean;
  children: ReactNode;
}) {
  return (
    <PaneShell
      flat={!active}
      portrait={portrait}
      className="plot-studio__pane--primary"
      label="Primary pane"
      head={<span className="plot-studio__pane-title">Primary</span>}
    >
      {children}
    </PaneShell>
  );
}

/** Shared pane frame: thin header + per-pane controls toggle (open state is local so panes don't interfere). */
function PaneShell({
  className,
  label,
  head,
  children,
  flat = false,
  portrait = false,
}: {
  className: string;
  label: string;
  head: ReactNode;
  children: ReactNode;
  /** Generate no boxes: the figure lays out as though this shell were not here. */
  flat?: boolean;
  portrait?: boolean;
}) {
  const [controlsOpen, setControlsOpen] = useState(false);
  return (
    <section
      className={`${flat
        ? "plot-studio__pane-flat"
        : `plot-studio__pane ${className}${controlsOpen ? " plot-studio__pane--controls" : ""}`}${portrait ? " plot-studio__pane--portrait" : ""}`}
      aria-label={flat ? undefined : label}
    >
      {!flat && (
        <header className="plot-studio__pane-head">
          {head}
          <button
            type="button"
            className={`plot-studio__pane-controls-toggle${
              controlsOpen ? " plot-studio__pane-controls-toggle--active" : ""
            }`}
            aria-expanded={controlsOpen}
            onClick={() => setControlsOpen((open) => !open)}
            title={controlsOpen
              ? "Hide this pane's controls and give the figure the whole pane"
              : "Show this pane's controls"}
          >
            {controlsOpen ? "Hide controls" : "Controls"}
          </button>
        </header>
      )}
      <div className={flat ? "plot-studio__pane-flat-body" : "plot-studio__pane-body"}>
        {children}
      </div>
    </section>
  );
}

/** Companion pane: carries a plot type and an optional handed-over preset (no history); isolated in its own ErrorBoundary. */
function CompanionPane({
  paneId,
  plotType,
  preset,
  onPlotTypeChange,
  available,
  bus,
  cameraLocked,
  cameraAuthority,
  ownsSharedOverlays,
  voxelBudgetScale,
  runDir,
  steps,
  stepIdx,
  datasets,
  renderFidelityMode,
  projectId,
  onFrameDisplayChange,
  linkedField,
  onFieldChange,
  resetKey,
}: {
  paneId: string;
  plotType: PlotType;
  preset: PlotPreset | null;
  onPlotTypeChange: (next: PlotType) => void;
  available: ReadonlySet<string>;
  bus: CameraSyncBus | null;
  cameraLocked: boolean;
  cameraAuthority: boolean;
  ownsSharedOverlays: boolean;
  voxelBudgetScale: number;
  runDir: string;
  steps: StepInfo[];
  stepIdx: number;
  datasets?: readonly PlotDatasetHandle[];
  renderFidelityMode: RenderFidelityMode;
  projectId: string | null;
  onFrameDisplayChange: (paneId: string, paneType: PlotType, report: FrameDisplayReport) => void;
  linkedField: string | null;
  onFieldChange: (field: string | null) => void;
  resetKey: string;
}) {
  const options = PLOT_TYPE_META.filter((meta: PlotTypeMeta) =>
    isPlotAvailable(meta.value, available));

  const paneValue = useMemo(
    () => ({ bus, paneId, cameraLocked, cameraAuthority, ownsSharedOverlays, voxelBudgetScale }),
    [bus, paneId, cameraLocked, cameraAuthority, ownsSharedOverlays, voxelBudgetScale],
  );
  const reportFrameDisplay = useCallback(
    (report: FrameDisplayReport) => onFrameDisplayChange(paneId, plotType, report),
    [onFrameDisplayChange, paneId, plotType],
  );

  return (
    <PaneShell
      className="plot-studio__pane--companion"
      portrait={metaFor(plotType).portrait === true}
      label={`Companion pane ${paneId}`}
      head={(
        <select
          className="plot-studio__pane-select"
          value={plotType}
          onChange={(event) => onPlotTypeChange(event.target.value as PlotType)}
          aria-label="Companion pane plot type"
        >
          {options.map((meta: PlotTypeMeta) => (
            <option key={meta.value} value={meta.value}>{meta.label}</option>
          ))}
        </select>
      )}
    >
      <ErrorBoundary
        key={resetKey}
        label="This pane"
        resetKeys={[plotType, resetKey, stepIdx]}
      >
        <PaneProvider value={paneValue}>
          <PlotRouter
            plotType={plotType}
            runDir={runDir}
            steps={steps}
            stepIdx={stepIdx}
            datasets={datasets}
            preset={preset}
            onPresetCleared={() => undefined}
            onFrameDisplayChange={reportFrameDisplay}
            linkedField={linkedField}
            onFieldChange={onFieldChange}
            renderFidelityMode={renderFidelityMode}
            projectId={projectId}
          />
        </PaneProvider>
      </ErrorBoundary>
    </PaneShell>
  );
}

interface PlotRouterProps {
  plotType: PlotType;
  runDir: string;
  steps: StepInfo[];
  stepIdx: number;
  datasets?: readonly PlotDatasetHandle[];
  preset: PlotPreset | null;
  onPresetCleared: () => void;
  defaultSliceAxis?: SliceAxis;
  onFrameDisplayChange: (report: FrameDisplayReport) => void;
  onScalarSelectionChange?: (selection: ScalarSelection | null) => void;
  volumeHandoff?: VolumeHandoffOffer;
  linkedField?: string | null;
  onFieldChange?: (field: string | null) => void;
  renderFidelityMode: RenderFidelityMode;
  projectId: string | null;
}

function PlotRouter({
  plotType,
  runDir,
  steps,
  stepIdx,
  datasets,
  preset,
  onPresetCleared,
  defaultSliceAxis,
  onFrameDisplayChange,
  onScalarSelectionChange,
  volumeHandoff,
  linkedField,
  onFieldChange,
  renderFidelityMode,
  projectId,
}: PlotRouterProps) {
  const props = {
    runDir,
    steps,
    stepIdx,
    datasets,
    preset,
    onPresetCleared,
    defaultSliceAxis,
    onFrameDisplayChange,
    onScalarSelectionChange,
    volumeHandoff,
    linkedField,
    onFieldChange,
    renderFidelityMode,
    projectId,
  };
  switch (plotType) {
    case "slice":
      return <SlicePlot {...props} />;
    case "vector":
      return <VectorField {...props} />;
    case "composite":
      return <CompositePlot {...props} />;
    case "timeseries":
      return <TimeSeriesPlot {...props} />;
    case "timeheight":
      return <TimeHeightPlot {...props} />;
    case "swath":
      return <SwathPlot {...props} />;
    case "vertical":
      return <VerticalProfile {...props} />;
    case "radial":
      return <RadialProfile {...props} />;
    case "hodograph":
      return <Hodograph {...props} />;
    case "skewt":
      return <SkewT {...props} />;
    case "volume":
      return (
        <Suspense fallback={<VolumeFallback />}>
          <VolumeRender3D {...props} />
        </Suspense>
      );
    case "isosurface":
      return (
        <Suspense fallback={<VolumeFallback />}>
          <Isosurface3D {...props} />
        </Suspense>
      );
    case "streamlines":
      return (
        <Suspense fallback={<VolumeFallback />}>
          <Streamlines3D {...props} />
        </Suspense>
      );
    case "supercell":
      return (
        <Suspense fallback={<VolumeFallback />}>
          <SupercellRender3D {...props} />
        </Suspense>
      );
    case "trajectories":
      return (
        <Suspense fallback={<VolumeFallback />}>
          <Trajectories3D {...props} />
        </Suspense>
      );
    case "trajectory_budget":
      return <TrajectoryBudgetPlot {...props} />;
    case "python":
      return (
        <Suspense fallback={<PythonFallback />}>
          <PythonPlot {...props} />
        </Suspense>
      );
  }
}

function PythonFallback() {
  return (
    <div className="plot-fallback">
      <Icon name="info" size={14} />
      <span>Loading Python tab…</span>
    </div>
  );
}

function VolumeFallback() {
  return (
    <div className="plot-fallback">
      <Icon name="info" size={14} />
      <span>Loading 3D renderer…</span>
    </div>
  );
}

interface CategoryBarProps {
  categories: ReadonlyArray<CategoryMeta>;
  active: PlotCategory;
  onSelect: (c: PlotCategory) => void;
}

/** Researcher-facing category row; categories unavailable for the loaded run are hidden. */
function CategoryBar({ categories, active, onSelect }: CategoryBarProps) {
  return (
    <div className="plot-studio__category-bar" role="group" aria-label="Plot categories">
      {categories.map((c) => (
        <button
          key={c.value}
          type="button"
          className={`plot-category ${active === c.value ? "plot-category--active" : ""}`}
          onClick={() => onSelect(c.value)}
          title={c.hint}
          aria-pressed={active === c.value}
        >
          {c.label}
        </button>
      ))}
    </div>
  );
}

interface PlotTypePickerProps {
  plotTypes: ReadonlyArray<PlotTypeMeta>;
  activePlotType: PlotType;
  hasActivePreset: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => boolean;
  onRedo: () => boolean;
  onPickType: (t: PlotType) => void;
}

/**
 * The navigation half of the picker: view history and the plot-type buttons.
 *
 * Kept out of the scrolling toolbelt. The type list is a fixed, small set, while presets grow with
 * the run — so scrolling the two together pushed a bounded control off screen to make room for an
 * unbounded one.
 */
function PlotTypePicker({
  plotTypes,
  activePlotType,
  hasActivePreset,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onPickType,
}: PlotTypePickerProps) {
  return (
    <div className="plot-studio__picker plot-studio__picker--nav">
      <div className="plot-studio__view-history" role="group" aria-label="Plot view selection history">
        <span>View history</span>
        <button
          type="button"
          disabled={!canUndo}
          onClick={() => onUndo()}
          aria-label="Undo plot view selection"
          aria-keyshortcuts="Meta+Z Control+Z"
          title="Undo the last category, renderer, or preset choice"
        >
          <Icon name="undo" size={13} /> Undo <kbd>⌘/Ctrl Z</kbd>
        </button>
        <button
          type="button"
          disabled={!canRedo}
          onClick={() => onRedo()}
          aria-label="Redo plot view selection"
          aria-keyshortcuts="Meta+Shift+Z Control+Shift+Z"
          title="Redo the next category, renderer, or preset choice"
        >
          <Icon name="redo" size={13} /> Redo <kbd>⇧⌘/Ctrl Z</kbd>
        </button>
      </div>
      {plotTypes.length > 0 && (
        <div className="plot-studio__type-bar" role="group" aria-label="Plot types">
          {plotTypes.map((m) => (
            <button
              key={m.value}
              type="button"
              className={`plot-type ${
                activePlotType === m.value && !hasActivePreset ? "plot-type--active" : ""
              }`}
              onClick={() => onPickType(m.value)}
              title={m.hint}
              aria-pressed={activePlotType === m.value && !hasActivePreset}
            >
              {m.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface PlotPresetPickerProps {
  presets: ReadonlyArray<PlotPreset>;
  activePresetId: string | null;
  /** True when the category also offers no generic renderer, so the run has nothing to plot here. */
  noPlotTypes: boolean;
  onPickPreset: (p: PlotPreset) => void;
}

/** The preset pills for the active category — the part of the picker allowed to scroll. */
function PlotPresetPicker({
  presets,
  activePresetId,
  noPlotTypes,
  onPickPreset,
}: PlotPresetPickerProps) {
  const ungrouped = presets.filter((p) => !p.subgroup);
  const subgroups: string[] = [];
  for (const p of presets) {
    if (p.subgroup && !subgroups.includes(p.subgroup)) subgroups.push(p.subgroup);
  }

  const pill = (p: PlotPreset) => (
    <button
      key={p.id}
      type="button"
      className={`plot-preset-pill ${activePresetId === p.id ? "plot-preset-pill--active" : ""}`}
      title={p.description}
      onClick={() => onPickPreset(p)}
      aria-pressed={activePresetId === p.id}
    >
      {p.label}
    </button>
  );

  return (
    <div className="plot-studio__picker plot-studio__picker--presets">
      {presets.length > 0 && (
        <div className="plot-preset-bar">
          {ungrouped.length > 0 && (
            <div className="plot-preset-group">{ungrouped.map(pill)}</div>
          )}
          {subgroups.map((sg) => (
            <div key={sg} className="plot-preset-group">
              <span className="plot-preset-group__label">{sg}</span>
              {presets.filter((p) => p.subgroup === sg).map(pill)}
            </div>
          ))}
        </div>
      )}

      {noPlotTypes && presets.length === 0 && (
        <div className="plot-studio__picker-empty">
          No plottable fields for this run yet.
        </div>
      )}
    </div>
  );
}
