import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  defaultsForSchema,
  readPath,
  reconcileFormConfig,
  writePath,
  type ConfigObject,
  type SectionSchema,
} from "@/lib/config/configSchema";
import {
  computeDiagnostics,
  type DiagStat,
  type DiagWarning,
} from "@/lib/science/diagnostics";
import { mergeConfig, parseYaml, stringifyYaml } from "@/lib/config/yaml";
import {
  validateSchemaVersion,
  isBlocking as schemaIsBlocking,
  summarize as summarizeSchema,
  type SchemaValidationResult,
} from "@/lib/config/schemaValidator";
import {
  getProjectRoot,
  getUserConfigDirectory,
  getSchemeCatalog,
  discoverRunLibrary,
  inspectCheckpoint,
  isTauriRuntime,
  listRuns,
  listTemplateConfigs,
  onRunExit,
  onRunLine,
  pickYamlFile,
  pickYamlSavePath,
  pickCheckpointFile,
  readConfig as readConfigFile,
  migrateConfig,
  spawnSim,
  writeConfig,
  validateConfigContents,
  type CheckpointInspection,
} from "@/lib/platform/tauri";
import {
  normalizeSchemeSelections,
  sectionsWithSchemeCatalog,
  type SchemeCatalog,
} from "@/lib/config/schemeCatalog";
import {
  authoritativeRuntimeStateFields,
  buildConfigPreflightRequest,
  preflightConfigWithBinary,
  type ConfigPreflightIssue,
  type ConfigPreflightReport,
} from "@/lib/config/configPreflight";
import {
  describeLaunchBlock,
  describeValidationBlock,
  type LaunchReadinessInputs,
} from "@/lib/config/launchReadiness";
import { effectiveRunsDirectory, loadStudioSettings, STUDIO_SETTINGS_CHANGE_EVENT } from "@/lib/studio/studioSettings";
import { processExitSummary } from "@/lib/run/runOutcome";
import {
  loadRemoteComputeProfiles,
  REMOTE_COMPUTE_PROFILES_CHANGE_EVENT,
  remoteComputeTargetKey,
  selectedRemoteComputeTarget,
  type RemoteComputeTarget,
} from "@/lib/platform/remoteCompute";
import {
  advancedConfigSectionCount,
  loadConfigDetailLevel,
  saveConfigDetailLevel,
  sectionsForConfigDetail,
  type ConfigDetailLevel,
} from "@/lib/config/configPresentation";
import {
  CONFIG_DRAFT_LIMIT,
  activateConfigDraft,
  branchConfigDraft,
  configDraftIsDirty,
  configDraftStore,
  createConfigDraft,
  deleteConfigDraft,
  inspectConfigDraftStore,
  markConfigDraftSaved,
  saveConfigDraftStore,
  updateConfigDraft,
  type ConfigDraft,
  type ConfigDraftStore,
} from "@/lib/config/configDrafts";
import { cm1SectionLandmark } from "@/lib/cm1/cm1ConfigCrosswalk";
import {
  consumeCm1SmokeConfig,
  cm1StagedLaunchMismatch,
  loadCm1GuidedHandoff,
  markCm1SmokeFailed,
  markCm1SmokePending,
  markCm1SmokeRunning,
  reconcileCm1SmokeRun,
  saveCm1GuidedHandoff,
  fnv1a64Text,
  type Cm1GuidedHandoff,
} from "@/lib/cm1/cm1GuidedHandoff";
import { Icon } from "@/components/shell/Icon";
import { SchemaForm } from "./SchemaForm";
import { NativeCm1SoundingImport } from "./NativeCm1SoundingImport";
import "./ConfigEditor.css";

const TEMPLATE_DIR = "configs";

/* Form-driven YAML editor. Form mode: schema sections + diagnostics. Raw mode: verbatim textarea (comments preserved; switching to Form re-parses). */

type ViewMode = "form" | "raw";

/** Spawn lifecycle for the Run button, tracked separately from useRuns to surface stderr without cross-referencing the queue panel. */
type SpawnState =
  | { kind: "idle" }
  | { kind: "starting" }
  | { kind: "running"; runId: string }
  | { kind: "failed"; runId: string; exitCode: number | null; stderrTail: string[]; reason: string }
  | { kind: "completed"; runId: string };

type CatalogState =
  | { kind: "loading" }
  | { kind: "ready"; catalog: SchemeCatalog; binaryPath?: string; executionTarget?: RemoteComputeTarget }
  | { kind: "error"; message: string };

type PreflightState =
  | { kind: "required" }
  | { kind: "pending"; evidenceKey: string }
  | { kind: "passed"; evidenceKey: string; report: ConfigPreflightReport }
  | { kind: "blocked"; evidenceKey: string; report: ConfigPreflightReport }
  | { kind: "error"; evidenceKey: string; message: string };

type CheckpointState =
  | { kind: "empty" }
  | { kind: "checking"; path: string }
  | { kind: "accepted"; requestedPath: string; evidence: CheckpointInspection }
  | { kind: "error"; path: string; message: string };

interface MigrationReview {
  sourcePath: string;
  caseName: string;
  originalText: string;
  migratedText: string;
  warnings: string;
}

const STDERR_TAIL_MAX = 40;

// Applied once per session so tab remounts don't clobber in-progress edits.
let autoLoadedDefaultTemplate = false;

function configFromDraft(draft: ConfigDraft): ConfigObject {
  try {
    return reconcileFormConfig(
      mergeConfig(defaultsForSchema(), parseYaml(draft.document)),
    );
  } catch {
    // Invalid raw YAML is retained verbatim and opens in Raw. Structured state
    // stays safe until the user repairs it and deliberately returns to Form.
    return defaultsForSchema();
  }
}

export function ConfigEditor() {
  const [initialDraftRecovery] = useState(() => {
    const inspected = inspectConfigDraftStore(
      typeof window === "undefined" ? undefined : window.localStorage,
    );
    if (inspected.kind === "ready") {
      autoLoadedDefaultTemplate = true;
      return { store: inspected.store, error: null as string | null };
    }
    return {
      store: configDraftStore(createConfigDraft({
        caseName: "cgrid_tornado_genesis",
        document: stringifyYaml(defaultsForSchema()),
      })),
      error: inspected.kind === "unreadable" ? inspected.message : null,
    };
  });
  const initialDraftState = initialDraftRecovery.store;
  const initialDraft = initialDraftState.drafts.find(({ id }) => id === initialDraftState.activeId)!;
  const [draftStore, setDraftStore] = useState<ConfigDraftStore>(initialDraftState);
  const [draftPersistence, setDraftPersistence] = useState<"current" | "saving" | "error">(
    initialDraftRecovery.error ? "error" : "current",
  );
  const [draftMessage, setDraftMessage] = useState<string | null>(initialDraftRecovery.error);
  const [draftRecoveryBlocked, setDraftRecoveryBlocked] = useState(initialDraftRecovery.error);
  const [forgetDraftConfirm, setForgetDraftConfirm] = useState(false);
  const [config, setConfig] = useState<ConfigObject>(() => configFromDraft(initialDraft));
  const [caseName, setCaseName] = useState(initialDraft.caseName);
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(["grid", "dynamics"]),
  );
  const [templates, setTemplates] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>(initialDraft.selectedTemplate);
  const [loadingTemplate, setLoadingTemplate] = useState(false);
  const [savingState, setSavingState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>(initialDraft.viewMode);
  const [detailLevel, setDetailLevel] = useState<ConfigDetailLevel>(() =>
    loadConfigDetailLevel(typeof window === "undefined" ? undefined : window.localStorage),
  );
  const [showCompatibilityNames, setShowCompatibilityNames] = useState(false);
  const [rawYaml, setRawYaml] = useState<string>(initialDraft.document);
  const [saveTargetPath, setSaveTargetPath] = useState<string | null>(initialDraft.sourcePath);
  const [rawError, setRawError] = useState<string | null>(null);
  const [spawn, setSpawn] = useState<SpawnState>({ kind: "idle" });
  const [catalogState, setCatalogState] = useState<CatalogState>({
    kind: "loading",
  });
  const [remoteProfiles, setRemoteProfiles] = useState(() => loadRemoteComputeProfiles());
  const [computeTargetId, setComputeTargetId] = useState("local");
  const [settingsRevision, setSettingsRevision] = useState(0);
  const executionTarget = selectedRemoteComputeTarget(remoteProfiles.profiles, computeTargetId);
  const executionTargetKey = executionTarget === undefined ? `unavailable:${computeTargetId}` : remoteComputeTargetKey(executionTarget);
  const [preflight, setPreflight] = useState<PreflightState>({ kind: "required" });
  const [checkpointState, setCheckpointState] = useState<CheckpointState>({ kind: "empty" });
  const [cm1Handoff, setCm1Handoff] = useState<Cm1GuidedHandoff | null>(() =>
    typeof window === "undefined" ? null : loadCm1GuidedHandoff(window.localStorage),
  );
  const checkpointSequenceRef = useRef(0);
  // Preflight result must be discarded if YAML/catalog evidence changed before it resolved.
  const preflightSequenceRef = useRef(0);
  const preflightEvidenceRef = useRef("");
  // Non-blocking schema info; only surfaces informational warnings (e.g. unknown top-level section).
  const [schemaResult, setSchemaResult] =
    useState<SchemaValidationResult | null>(null);
  const [migrationNotice, setMigrationNotice] = useState<string | null>(null);
  const [migrationReview, setMigrationReview] = useState<MigrationReview | null>(null);

  // stderr ring buffers keyed by runId; ref so high-volume logging doesn't trigger re-renders.
  const stderrBufRef = useRef<Map<string, string[]>>(new Map());
  // Exit snapshots keyed by runId — a fast-failing child can exit before spawn_sim resolves.
  const recentExitsRef = useRef<
    Map<string, { status: string; exitCode: number | null; stderrTail: string[] }>
  >(new Map());
  // Bundled/source root is read-only template storage; new files use the
  // separately resolved application-owned writable directory.
  const [projectRoot, setProjectRoot] = useState<string | null>(null);
  const [userConfigDirectory, setUserConfigDirectory] = useState<string | null>(null);

  // Resolve immutable templates and writable user storage before enabling file-backed actions.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [root, writableConfigs] = await Promise.all([
          getProjectRoot(),
          getUserConfigDirectory(),
        ]);
        if (cancelled) return;
        setProjectRoot(root);
        setUserConfigDirectory(writableConfigs);
        const list = await listTemplateConfigs(root);
        if (!cancelled) setTemplates(list);
      } catch (e) {
        if (!cancelled) setErrorMessage(`Configuration resources are unavailable: ${bridgeErrorMessage(e)}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load scheme catalog from the same executable that will be launched; a missing binary is a visible blocking state.
  useEffect(() => {
    let cancelled = false;
    setCatalogState({ kind: "loading" });
    setPreflight({ kind: "required" });
    if (executionTarget === undefined) {
      setCatalogState({ kind: "error", message: "The selected remote connection is unavailable. Restore it in Settings or select this computer." });
      return;
    }
    const binaryPath =
      executionTarget?.binaryPath ?? (loadStudioSettings().binaryPathOverride.trim() || undefined);
    getSchemeCatalog(binaryPath, executionTarget ?? undefined)
      .then((catalog) => {
        if (!cancelled) setCatalogState({ kind: "ready", catalog, binaryPath, executionTarget: executionTarget ?? undefined });
      })
      .catch((error) => {
        if (!cancelled) {
          setCatalogState({
            kind: "error",
            message: bridgeErrorMessage(error),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [executionTarget, settingsRevision]);

  useEffect(() => {
    const remoteChanged = () => {
      preflightSequenceRef.current++;
      setCatalogState({ kind: "loading" });
      setRemoteProfiles(loadRemoteComputeProfiles());
      setSettingsRevision(revision => revision + 1);
    };
    const localChanged = () => {
      preflightSequenceRef.current++;
      setCatalogState({ kind: "loading" });
      setSettingsRevision(revision => revision + 1);
    };
    window.addEventListener(REMOTE_COMPUTE_PROFILES_CHANGE_EVENT, remoteChanged);
    window.addEventListener(STUDIO_SETTINGS_CHANGE_EVENT, localChanged);
    return () => {
      window.removeEventListener(REMOTE_COMPUTE_PROFILES_CHANGE_EVENT, remoteChanged);
      window.removeEventListener(STUDIO_SETTINGS_CHANGE_EVENT, localChanged);
    };
  }, []);

  // Global run:line + run:exit subscription — catches fast-failing children before a per-runId effect could mount.
  useEffect(() => {
    let cancelled = false;
    let unsubLine: (() => void) | undefined;
    let unsubExit: (() => void) | undefined;
    (async () => {
      unsubLine = await onRunLine((evt) => {
        if (cancelled) return;
        if (evt.stream !== "stderr") return;
        let buf = stderrBufRef.current.get(evt.runId);
        if (!buf) {
          // Evict the oldest entry to bound the concurrent-run tracking size.
          if (stderrBufRef.current.size > 32) {
            const oldestKey = stderrBufRef.current.keys().next().value;
            if (oldestKey !== undefined) stderrBufRef.current.delete(oldestKey);
          }
          buf = [];
          stderrBufRef.current.set(evt.runId, buf);
        }
        buf.push(evt.line);
        if (buf.length > STDERR_TAIL_MAX) buf.splice(0, buf.length - STDERR_TAIL_MAX);
      });
      // Unmounted while onRunLine resolved — unsubscribe to prevent a leak.
      if (cancelled) {
        unsubLine?.();
        return;
      }
      unsubExit = await onRunExit((evt) => {
        if (cancelled) return;
        // Snapshot stderr + status atomically so the runSimulation continuation reads consistent data.
        const stderrTail = stderrBufRef.current.get(evt.runId) ?? [];
        stderrBufRef.current.delete(evt.runId);

        // Cap the snapshot map to avoid accumulating entries from outside this editor.
        if (recentExitsRef.current.size > 32) {
          const oldestKey = recentExitsRef.current.keys().next().value;
          if (oldestKey !== undefined) recentExitsRef.current.delete(oldestKey);
        }
        recentExitsRef.current.set(evt.runId, {
          status: evt.status,
          exitCode: evt.exitCode,
          stderrTail,
        });

        setSpawn((prev) => {
          if (prev.kind !== "running" || prev.runId !== evt.runId) {
            return prev;
          }
          // Consumed — clear the snapshot to keep the map small.
          recentExitsRef.current.delete(evt.runId);
          return buildExitState(evt.runId, evt.status, evt.exitCode, stderrTail);
        });
        void settleCm1GuidedRun(evt.runId, evt.status).then((record) => {
          if (record) setCm1Handoff(record);
        });
      });
      // Unmounted while onRunExit resolved — unsubscribe now.
      if (cancelled) {
        unsubExit?.();
        return;
      }
    })();
    return () => {
      cancelled = true;
      unsubLine?.();
      unsubExit?.();
    };
  }, []);

  const schemeValue = readPath(config, "dynamics.scheme");
  const selectedScheme = catalogState.kind === "ready" && typeof schemeValue === "string"
    ? catalogState.catalog.modules.flatMap((module) => module.schemes).find((scheme) => scheme.id === schemeValue)
    : null;
  const schemeLabel = selectedScheme?.displayName
    ?? (typeof schemeValue === "string"
      ? schemeValue.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase())
      : "—");

  const diagnostics = useMemo(() => computeDiagnostics(config), [config]);

  const schemaSections = useMemo(
    () =>
      sectionsWithSchemeCatalog(
        catalogState.kind === "ready" ? catalogState.catalog : null,
        config,
        catalogState.kind === "error"
          ? catalogState.message
          : "Loading schemes from the configured simulator binary.",
      ),
    [catalogState, config],
  );
  const visibleSchemaSections = useMemo(
    () => sectionsForConfigDetail(schemaSections, detailLevel),
    [schemaSections, detailLevel],
  );
  const specialistSectionCount = useMemo(
    () => advancedConfigSectionCount(schemaSections),
    [schemaSections],
  );
  const changeDetailLevel = useCallback((next: ConfigDetailLevel) => {
    setDetailLevel(next);
    saveConfigDetailLevel(typeof window === "undefined" ? undefined : window.localStorage, next);
  }, []);

  // Normalize aliases and remove inactive variant blocks; equality guard ensures convergence.
  useEffect(() => {
    if (catalogState.kind !== "ready") return;
    setConfig((current) => {
      const normalized = normalizeSchemeSelections(
        catalogState.catalog,
        current,
      );
      return JSON.stringify(normalized) === JSON.stringify(current)
        ? current
        : normalized;
    });
  }, [catalogState]);

  // Normalize aliases inline so Run/YAML cannot be clicked between a scheme change and the reconciliation effect.
  const changeConfig = useCallback(
    (next: ConfigObject) =>
      setConfig(
        catalogState.kind === "ready"
          ? normalizeSchemeSelections(catalogState.catalog, next)
          : next,
      ),
    [catalogState],
  );
  const setResumeCheckpoint = useCallback(
    (path: string) =>
      setConfig((current) => {
        const next = writePath(current, "extensions.checkpoint.resume_from", path);
        return catalogState.kind === "ready"
          ? normalizeSchemeSelections(catalogState.catalog, next)
          : next;
      }),
    [catalogState],
  );

  const resumeCheckpointPath = useMemo(() => {
    try {
      const active = viewMode === "raw" ? parseYaml(rawYaml) : config;
      const value = readPath(active, "extensions.checkpoint.resume_from");
      return typeof value === "string" ? value.trim() : "";
    } catch { return ""; }
  }, [config, rawYaml, viewMode]);

  useEffect(() => {
    checkpointSequenceRef.current++;
    setCheckpointState({ kind: "empty" });
  }, [executionTargetKey]);

  // Checkpoint paths from YAML must earn the same evidence as picker-chosen paths.
  useEffect(() => {
    if (!resumeCheckpointPath) {
      // Guarded: unconditionally setting a fresh literal re-enters this effect forever.
      if (checkpointState.kind !== "empty") {
        checkpointSequenceRef.current += 1;
        setCheckpointState({ kind: "empty" });
      }
      return;
    }
    if (
      checkpointState.kind === "accepted" &&
      checkpointState.requestedPath === resumeCheckpointPath
    ) {
      return;
    }
    if (checkpointState.kind === "checking" && checkpointState.path === resumeCheckpointPath) {
      return;
    }
    if (checkpointState.kind === "error" && checkpointState.path === resumeCheckpointPath) {
      return;
    }
    if (!isTauriRuntime()) {
      setCheckpointState({
        kind: "error",
        path: resumeCheckpointPath,
        message: "Open this configuration in Nimbus Studio desktop to verify the local checkpoint and sidecar.",
      });
      return;
    }
    if (executionTarget === undefined) return;
    const sequence = ++checkpointSequenceRef.current;
    setCheckpointState({ kind: "checking", path: resumeCheckpointPath });
    void inspectCheckpoint(resumeCheckpointPath, executionTarget ?? undefined)
      .then((evidence) => {
        if (sequence === checkpointSequenceRef.current) {
          setCheckpointState({ kind: "accepted", requestedPath: resumeCheckpointPath, evidence });
        }
      })
      .catch((error) => {
        if (sequence === checkpointSequenceRef.current) {
          setCheckpointState({
            kind: "error",
            path: resumeCheckpointPath,
            message: bridgeErrorMessage(error),
          });
        }
      });
  }, [resumeCheckpointPath, checkpointState, executionTarget]);

  const chooseCheckpoint = useCallback(async () => {
    const sequence = ++checkpointSequenceRef.current;
    let path: string | null;
    try {
      if (executionTarget === undefined) throw new Error("Select an available compute target before inspecting a checkpoint.");
      path = executionTarget ? resumeCheckpointPath : await pickCheckpointFile();
      if (executionTarget && !path) throw new Error("Enter the remote checkpoint path in the configuration, then inspect it.");
    } catch (error) {
      setCheckpointState({
        kind: "error",
        path: "",
        message: bridgeErrorMessage(error),
      });
      return;
    }
    if (!path || sequence !== checkpointSequenceRef.current) return;
    setCheckpointState({ kind: "checking", path });
    try {
      const evidence = await inspectCheckpoint(path, executionTarget ?? undefined);
      if (sequence !== checkpointSequenceRef.current) return;
      // Use the canonical path only after all integrity checks pass; a rejected candidate never changes the config.
      setResumeCheckpoint(evidence.path);
      setCheckpointState({ kind: "accepted", requestedPath: evidence.path, evidence });
    } catch (error) {
      if (sequence !== checkpointSequenceRef.current) return;
      setCheckpointState({
        kind: "error",
        path,
        message: bridgeErrorMessage(error),
      });
    }
  }, [setResumeCheckpoint, executionTarget, resumeCheckpointPath]);

  const clearCheckpoint = useCallback(() => {
    checkpointSequenceRef.current += 1;
    setResumeCheckpoint("");
    setCheckpointState({ kind: "empty" });
  }, [setResumeCheckpoint]);

  // New configurations and cloned templates always publish to app-owned storage,
  // never back into the immutable bundled template catalog.
  const defaultYamlPath = useMemo(() => {
    const root = userConfigDirectory ?? (isTauriRuntime() ? "" : "/__mock_user_config__");
    const safeName = (caseName.trim() || "config").replace(/[\\/:*?"<>|]+/g, "-");
    return root ? `${root}/${safeName}.yaml` : "";
  }, [userConfigDirectory, caseName]);
  const yamlPath = saveTargetPath ?? defaultYamlPath;
  const yamlText = useMemo(() => stringifyYaml(config), [config]);
  const draftDocument = viewMode === "raw" ? rawYaml : yamlText;
  const activeDraft = draftStore.drafts.find(({ id }) => id === draftStore.activeId)!;
  const draftDirty = configDraftIsDirty(activeDraft, draftDocument);
  const pendingDraftStoreRef = useRef(draftStore);
  const pendingDraftDirtyRef = useRef(false);
  const draftAutosaveMountedRef = useRef(false);
  const draftRecoveryBlockedRef = useRef(draftRecoveryBlocked);
  draftRecoveryBlockedRef.current = draftRecoveryBlocked;

  const persistDraftStore = useCallback((next: ConfigDraftStore) => {
    setDraftStore(next);
    setDraftPersistence("saving");
    try {
      saveConfigDraftStore(window.localStorage, next);
      pendingDraftStoreRef.current = next;
      pendingDraftDirtyRef.current = false;
      setDraftPersistence("current");
      setDraftMessage(null);
    } catch (reason) {
      setDraftPersistence("error");
      setDraftMessage(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  const commitCurrentDraft = useCallback((store: ConfigDraftStore, now = new Date().toISOString()) =>
    updateConfigDraft(store, store.activeId, {
      caseName,
      document: draftDocument,
      viewMode,
      selectedTemplate,
      sourcePath: saveTargetPath,
    }, now), [caseName, draftDocument, saveTargetPath, selectedTemplate, viewMode]);

  const restoreDraft = useCallback((draft: ConfigDraft) => {
    setConfig(configFromDraft(draft));
    setCaseName(draft.caseName);
    setRawYaml(draft.document);
    setViewMode(draft.viewMode);
    setSelectedTemplate(draft.selectedTemplate);
    setSaveTargetPath(draft.sourcePath);
    setRawError(null);
    setErrorMessage(null);
    setSchemaResult(null);
    setPreflight({ kind: "required" });
    setForgetDraftConfirm(false);
  }, []);

  const selectDraft = useCallback((id: string) => {
    if (id === draftStore.activeId) return;
    const committed = commitCurrentDraft(draftStore);
    const next = activateConfigDraft(committed, id);
    persistDraftStore(next);
    restoreDraft(next.drafts.find((draft) => draft.id === id)!);
  }, [commitCurrentDraft, draftStore, persistDraftStore, restoreDraft]);

  const branchDraft = useCallback(() => {
    const committed = commitCurrentDraft(draftStore);
    const stem = caseName.trim() || "experiment";
    const names = new Set(committed.drafts.map((draft) => draft.caseName));
    let branchName = `${stem}-branch`;
    let suffix = 2;
    while (names.has(branchName)) branchName = `${stem}-branch-${suffix++}`;
    try {
      const next = branchConfigDraft(committed, committed.activeId, branchName);
      persistDraftStore(next);
      restoreDraft(next.drafts.find((draft) => draft.id === next.activeId)!);
    } catch (reason) {
      setDraftMessage(reason instanceof Error ? reason.message : String(reason));
      setDraftPersistence("error");
    }
  }, [caseName, commitCurrentDraft, draftStore, persistDraftStore, restoreDraft]);

  const forgetDraft = useCallback(() => {
    if (!forgetDraftConfirm) {
      setForgetDraftConfirm(true);
      return;
    }
    try {
      const committed = commitCurrentDraft(draftStore);
      const next = deleteConfigDraft(committed, committed.activeId);
      persistDraftStore(next);
      restoreDraft(next.drafts.find((draft) => draft.id === next.activeId)!);
    } catch (reason) {
      setDraftMessage(reason instanceof Error ? reason.message : String(reason));
      setDraftPersistence("error");
    }
  }, [commitCurrentDraft, draftStore, forgetDraftConfirm, persistDraftStore, restoreDraft]);

  // Debounced localStorage draft — keeps working bytes recoverable without claiming a named file save.
  useEffect(() => {
    if (!draftAutosaveMountedRef.current) {
      draftAutosaveMountedRef.current = true;
      return;
    }
    if (draftRecoveryBlocked) {
      setDraftPersistence("error");
      setDraftMessage(draftRecoveryBlocked);
      return;
    }
    setDraftPersistence("saving");
    let next: ConfigDraftStore;
    try {
      next = commitCurrentDraft(draftStore);
      pendingDraftStoreRef.current = next;
      pendingDraftDirtyRef.current = true;
    } catch (reason) {
      setDraftPersistence("error");
      setDraftMessage(reason instanceof Error ? reason.message : String(reason));
      return;
    }
    const timer = window.setTimeout(() => {
      try {
        saveConfigDraftStore(window.localStorage, next);
        pendingDraftDirtyRef.current = false;
        setDraftStore(next);
        setDraftPersistence("current");
        setDraftMessage(null);
      } catch (reason) {
        setDraftPersistence("error");
        setDraftMessage(reason instanceof Error ? reason.message : String(reason));
      }
    }, 220);
    return () => window.clearTimeout(timer);
  // draftStore is intentionally omitted: including the committed timestamp
  // would schedule itself forever. User-visible draft inputs own the cadence.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseName, draftDocument, draftRecoveryBlocked, saveTargetPath, selectedTemplate, viewMode]);

  // A pending debounce must not lose the final keystroke when the page closes
  // or Configure unmounts. The ref is updated synchronously for each render.
  useEffect(() => {
    const flushPendingDraft = () => {
      if (draftRecoveryBlockedRef.current || !pendingDraftDirtyRef.current) return;
      try {
        saveConfigDraftStore(window.localStorage, pendingDraftStoreRef.current);
        pendingDraftDirtyRef.current = false;
      } catch {
        // The editor reports storage failures while mounted; unload cannot safely render.
      }
    };
    window.addEventListener("pagehide", flushPendingDraft);
    return () => {
      window.removeEventListener("pagehide", flushPendingDraft);
      flushPendingDraft();
    };
  }, []);

  const replaceUnreadableDraftRecovery = useCallback(() => {
    const next = commitCurrentDraft(draftStore);
    try {
      saveConfigDraftStore(window.localStorage, next);
      pendingDraftStoreRef.current = next;
      pendingDraftDirtyRef.current = false;
      setDraftStore(next);
      setDraftRecoveryBlocked(null);
      setDraftPersistence("current");
      setDraftMessage(null);
    } catch (reason) {
      setDraftPersistence("error");
      setDraftMessage(reason instanceof Error ? reason.message : String(reason));
    }
  }, [commitCurrentDraft, draftStore]);

  const recordDraftFileEvidence = useCallback((document: string, path: string, nextCaseName = caseName) => {
    try {
      let next = updateConfigDraft(draftStore, draftStore.activeId, {
        caseName: nextCaseName,
        document,
        viewMode,
        selectedTemplate,
        sourcePath: path,
      });
      next = markConfigDraftSaved(next, next.activeId, document, path);
      persistDraftStore(next);
      setSaveTargetPath(path);
    } catch (reason) {
      setDraftPersistence("error");
      setDraftMessage(reason instanceof Error ? reason.message : String(reason));
    }
  }, [caseName, draftStore, persistDraftStore, selectedTemplate, viewMode]);

  const preflightDocument = viewMode === "raw" ? rawYaml : yamlText;
  const preflightEvidenceKey = useMemo(
    () =>
      JSON.stringify({
        executionTarget: executionTargetKey,
        document: preflightDocument,
        binary:
          catalogState.kind === "ready"
            ? catalogState.binaryPath ?? "<auto-resolved>"
            : `<catalog:${catalogState.kind}>`,
        catalog:
          catalogState.kind === "ready"
            ? catalogState.catalog
            : catalogState.kind,
      }),
    [preflightDocument, catalogState, executionTargetKey],
  );
  preflightEvidenceRef.current = preflightEvidenceKey;

  const preflightIsCurrent =
    preflight.kind !== "required" &&
    preflight.evidenceKey === preflightEvidenceKey;
  const currentPreflight = preflightIsCurrent ? preflight : { kind: "required" as const };

  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  /** Parse and validate the original bytes before merging defaults. */
  const populateFromText = useCallback((text: string): boolean => {
    let parsed: ConfigObject;
    try {
      parsed = parseYaml(text);
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : String(e));
      return false;
    }
    const validation = validateSchemaVersion(parsed);
    setSchemaResult(
      validation.errors.length > 0 || validation.warnings.length > 0
        ? validation
        : null,
    );
    if (schemaIsBlocking(validation)) {
      setErrorMessage(summarizeSchema(validation));
      return false;
    }
    setConfig(reconcileFormConfig(mergeConfig(defaultsForSchema(), parsed)));
    setRawYaml(text);
    return true;
  }, []);

  const restoreCm1SmokeConfig = useCallback(() => {
    const record = loadCm1GuidedHandoff(window.localStorage);
    if (!record || !populateFromText(record.smoke.yaml)) return;
    const consumed = consumeCm1SmokeConfig(record);
    saveCm1GuidedHandoff(window.localStorage, consumed);
    setCm1Handoff(consumed);
    setCaseName("cm1-sounding-smoke");
    setSelectedTemplate("");
    recordDraftFileEvidence(record.smoke.yaml, record.smoke.configPath, "cm1-sounding-smoke");
    // Raw mode preserves staged bytes; Form mode would reserialize and invalidate the guided launch.
    setViewMode("raw");
    setErrorMessage(null);
    autoLoadedDefaultTemplate = true;
  }, [populateFromText, recordDraftFileEvidence]);

  // Restore a reviewed CM1 handoff from its versioned record.
  useEffect(() => {
    if (!cm1Handoff || cm1Handoff.run.status === "complete") return;
    restoreCm1SmokeConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cm1Handoff?.id, restoreCm1SmokeConfig]);

  // Reconcile a persisted guided-run id on mount; a cached `complete` flag is revalidated, never trusted blindly.
  useEffect(() => {
    if (cm1Handoff?.run.status !== "running"
      && cm1Handoff?.run.status !== "pending"
      && cm1Handoff?.run.status !== "complete") return;
    void settleCm1GuidedRun(cm1Handoff.run.runId, "reconcile").then((record) => {
      if (record) setCm1Handoff(record);
    });
  }, [cm1Handoff?.run.status, cm1Handoff?.run.runId]);

  const loadTemplate = useCallback(
    async (template: string) => {
      if (!template) return;
      setLoadingTemplate(true);
      setErrorMessage(null);
      setRawError(null);
      setSchemaResult(null);
      try {
        const templatePath = projectRoot
          ? `${projectRoot}/${TEMPLATE_DIR}/${template}`
          : `${TEMPLATE_DIR}/${template}`;
        const text = await readConfigFile(templatePath);
        if (populateFromText(text)) {
          const base = template.split(/[/\\]/).pop() ?? "template";
          setCaseName(base.replace(/\.ya?ml$/, ""));
          setSelectedTemplate(template);
          setSaveTargetPath(null);
          setMigrationNotice(null);
        }
      } catch (e) {
        setErrorMessage(e instanceof Error ? e.message : String(e));
      } finally {
        setLoadingTemplate(false);
      }
    },
    [projectRoot, populateFromText],
  );

  // Apply the "Default template" setting once per session after projectRoot resolves.
  useEffect(() => {
    if (autoLoadedDefaultTemplate) return;
    if (!projectRoot) return;
    const def = loadStudioSettings().defaultTemplate.trim();
    if (!def) return;
    autoLoadedDefaultTemplate = true;
    void loadTemplate(def);
  }, [projectRoot, loadTemplate]);

  /** Native upload — pick a .yaml from anywhere on disk and load it. */
  const onUpload = useCallback(async () => {
    setErrorMessage(null);
    setRawError(null);
    setSchemaResult(null);
    try {
      const path = await pickYamlFile();
      if (!path) return;
      setLoadingTemplate(true);
      const text = await readConfigFile(path);
      const parsed = parseYaml(text);
      const originalValidation = validateSchemaVersion(parsed);
      if (originalValidation.status === "v0_legacy") {
        if (catalogState.kind !== "ready") throw new Error("Wait for the selected simulator catalog before migrating a configuration.");
        const binaryPath = catalogState.kind === "ready" ? catalogState.binaryPath : undefined;
        const migration = await migrateConfig(path, binaryPath, catalogState.executionTarget);
        if (migration.exitCode !== 0 || !migration.migratedYaml.trim()) {
          throw new Error(migration.stderr.trim() || "The simulator could not migrate this legacy configuration.");
        }
        const base = path.split(/[/\\]/).pop() ?? "migrated";
        setMigrationReview({
          sourcePath: path,
          caseName: base.replace(/\.ya?ml$/, ""),
          originalText: text,
          migratedText: migration.migratedYaml,
          warnings: migration.stderr.trim(),
        });
        return;
      }
      setMigrationReview(null);
      setMigrationNotice(null);
      if (populateFromText(text)) {
        // Derive case_name from filename basename, stripped of extension.
        const base = path.split(/[/\\]/).pop() ?? "uploaded";
        const nextCaseName = base.replace(/\.ya?ml$/, "");
        setCaseName(nextCaseName);
        setSelectedTemplate("");
        recordDraftFileEvidence(text, path, nextCaseName);
      }
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingTemplate(false);
    }
  }, [catalogState, populateFromText, recordDraftFileEvidence]);

  const acceptMigration = useCallback(() => {
    if (!migrationReview || !populateFromText(migrationReview.migratedText)) return;
    setCaseName(`${migrationReview.caseName}-v1`);
    setSelectedTemplate("");
    setSaveTargetPath(null);
    setViewMode("raw");
    setMigrationNotice(
      "The reviewed migration is now an unsaved draft. The legacy source file was not modified; Save publishes a new app-owned v1 file.",
    );
    setMigrationReview(null);
    setErrorMessage(null);
  }, [migrationReview, populateFromText]);

  /** Raw mode: write verbatim (comments preserved). Form mode: write canonical re-serialized YAML. */
  const resolveYamlText = useCallback((): { text: string; error: string | null } => {
    if (viewMode === "raw") {
      try {
        const parsed = parseYaml(rawYaml);
        const validation = validateSchemaVersion(parsed);
        setSchemaResult(validation.status === "v1_valid" ? null : validation);
        if (schemaIsBlocking(validation)) {
          return { text: rawYaml, error: summarizeSchema(validation) };
        }
        return { text: rawYaml, error: null };
      } catch (e) {
        return { text: rawYaml, error: e instanceof Error ? e.message : String(e) };
      }
    }
    const validation = validateSchemaVersion(config);
    setSchemaResult(validation.status === "v1_valid" ? null : validation);
    return schemaIsBlocking(validation)
      ? { text: yamlText, error: summarizeSchema(validation) }
      : { text: yamlText, error: null };
  }, [viewMode, rawYaml, yamlText, config]);

  // Cleared on unmount so navigating away doesn't fire setSavingState on a dead component.
  const saveTimerRef = useRef<number | null>(null);
  const saveInFlightRef = useRef(false);
  useEffect(
    () => () => {
      if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
    },
    [],
  );

  const save = useCallback(async () => {
    if (saveInFlightRef.current) return;
    if (!yamlPath) {
      setSavingState("error");
      setErrorMessage("Writable configuration storage is still being prepared.");
      return;
    }
    saveInFlightRef.current = true;
    setSavingState("saving");
    setErrorMessage(null);
    const { text, error } = resolveYamlText();
    if (error) {
      setSavingState("error");
      setErrorMessage(`YAML parse error: ${error}`);
      saveInFlightRef.current = false;
      return;
    }
    try {
      await writeConfig(yamlPath, text);
      recordDraftFileEvidence(text, yamlPath);
      setSavingState("saved");
      if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = window.setTimeout(() => setSavingState("idle"), 1500) as unknown as number;
    } catch (e) {
      setSavingState("error");
      setErrorMessage(e instanceof Error ? e.message : String(e));
    } finally {
      saveInFlightRef.current = false;
    }
  }, [yamlPath, recordDraftFileEvidence, resolveYamlText]);

  /** Open native save dialog; validates YAML before showing the picker. */
  const saveAs = useCallback(async () => {
    if (saveInFlightRef.current) return;
    saveInFlightRef.current = true;
    setErrorMessage(null);
    const { text, error } = resolveYamlText();
    if (error) {
      setSavingState("error");
      setErrorMessage(`YAML parse error: ${error}`);
      saveInFlightRef.current = false;
      return;
    }
    let path: string | null;
    try {
      path = await pickYamlSavePath(`${caseName || "config"}.yaml`);
    } catch (e) {
      setSavingState("error");
      setErrorMessage(e instanceof Error ? e.message : String(e));
      saveInFlightRef.current = false;
      return;
    }
    if (!path) {
      saveInFlightRef.current = false;
      return;
    }
    setSavingState("saving");
    try {
      await writeConfig(path, text);
      recordDraftFileEvidence(text, path);
      setSavingState("saved");
      if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = window.setTimeout(() => setSavingState("idle"), 1500) as unknown as number;
    } catch (e) {
      setSavingState("error");
      setErrorMessage(e instanceof Error ? e.message : String(e));
    } finally {
      saveInFlightRef.current = false;
    }
  }, [caseName, recordDraftFileEvidence, resolveYamlText]);

  /** Preflight the exact binary without writing or spawning. */
  const validateConfiguration = useCallback(async () => {
    setErrorMessage(null);
    if (!isTauriRuntime()) {
      setPreflight({
        kind: "error",
        evidenceKey: preflightEvidenceKey,
        message: "Exact-binary preflight is available only in Nimbus Studio desktop.",
      });
      return;
    }
    if (catalogState.kind !== "ready") {
      setErrorMessage(
        "The configured simulator's scheme catalog is unavailable. Fix the simulator path before validating.",
      );
      return;
    }
    const { text, error } = resolveYamlText();
    if (error) {
      setErrorMessage(`YAML parse error: ${error}`);
      return;
    }

    let candidate: ConfigObject;
    try {
      candidate = viewMode === "raw"
        ? reconcileFormConfig(mergeConfig(defaultsForSchema(), parseYaml(text)))
        : config;
    } catch (parseError) {
      setErrorMessage(`YAML parse error: ${bridgeErrorMessage(parseError)}`);
      return;
    }

    const sequence = ++preflightSequenceRef.current;
    const evidenceKey = preflightEvidenceKey;
    setPreflight({ kind: "pending", evidenceKey });
    if (resumeCheckpointPath) {
      try {
        const evidence = await inspectCheckpoint(resumeCheckpointPath, catalogState.executionTarget);
        if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
        setCheckpointState({ kind: "accepted", requestedPath: resumeCheckpointPath, evidence });
      } catch (checkpointError) {
        if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
        setPreflight({ kind: "error", evidenceKey, message: bridgeErrorMessage(checkpointError) });
        setCheckpointState({
          kind: "error",
          path: resumeCheckpointPath,
          message: bridgeErrorMessage(checkpointError),
        });
        setErrorMessage(`Checkpoint was not accepted: ${bridgeErrorMessage(checkpointError)}`);
        return;
      }
    }

    try {
      const exact = await validateConfigContents(text, yamlPath, catalogState.binaryPath, catalogState.executionTarget);
      if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
      if (!exact.valid) {
        const message = exact.stderr.trim() || exact.stdout.trim()
          || `The exact simulator rejected this configuration (exit ${exact.exitCode ?? "unknown"}).`;
        setPreflight({ kind: "error", evidenceKey, message });
        setErrorMessage(`Exact simulator validation failed: ${message}`);
        return;
      }
      const request = buildConfigPreflightRequest(
        candidate,
        catalogState.catalog,
        authoritativeRuntimeStateFields(catalogState.catalog),
        { yamlText: text, yamlPath },
      );
      const report = await preflightConfigWithBinary(request, catalogState.binaryPath, catalogState.executionTarget);
      if (
        sequence !== preflightSequenceRef.current ||
        evidenceKey !== preflightEvidenceRef.current
      ) return;
      setPreflight(report.compatible
        ? { kind: "passed", evidenceKey, report }
        : { kind: "blocked", evidenceKey, report });
    } catch (error) {
      if (
        sequence !== preflightSequenceRef.current ||
        evidenceKey !== preflightEvidenceRef.current
      ) return;
      setPreflight({ kind: "error", evidenceKey, message: bridgeErrorMessage(error) });
    }
  }, [
    catalogState,
    config,
    preflightEvidenceKey,
    resolveYamlText,
    resumeCheckpointPath,
    viewMode,
    yamlPath,
  ]);

  const runSimulation = useCallback(async () => {
    setErrorMessage(null);
    if (!yamlPath) {
      setErrorMessage("Writable configuration storage is still being prepared.");
      return;
    }
    if (!isTauriRuntime()) {
      setPreflight({
        kind: "error",
        evidenceKey: preflightEvidenceKey,
        message: "Exact-binary preflight is available only in Nimbus Studio desktop.",
      });
      return;
    }
    if (catalogState.kind !== "ready") {
      setErrorMessage(
        "The configured simulator's scheme catalog is unavailable. Fix the simulator path before running.",
      );
      return;
    }
    const { text, error } = resolveYamlText();
    if (error) {
      setErrorMessage(`YAML parse error: ${error}`);
      return;
    }

    let launchConfig: ConfigObject;
    try {
      launchConfig =
        viewMode === "raw"
          ? reconcileFormConfig(mergeConfig(defaultsForSchema(), parseYaml(text)))
          : config;
    } catch (parseError) {
      setErrorMessage(`YAML parse error: ${bridgeErrorMessage(parseError)}`);
      return;
    }

    const launchResumeValue = readPath(
      launchConfig,
      "extensions.checkpoint.resume_from",
    );
    const launchResumePath =
      typeof launchResumeValue === "string" ? launchResumeValue.trim() : "";
    const sequence = ++preflightSequenceRef.current;
    const evidenceKey = preflightEvidenceKey;
    setPreflight({ kind: "pending", evidenceKey });
    if (launchResumePath) {
      try {
        const evidence = await inspectCheckpoint(launchResumePath, catalogState.executionTarget);
        if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
        setCheckpointState({ kind: "accepted", requestedPath: launchResumePath, evidence });
      } catch (checkpointError) {
        if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
        setPreflight({ kind: "error", evidenceKey, message: bridgeErrorMessage(checkpointError) });
        setCheckpointState({
          kind: "error",
          path: launchResumePath,
          message: bridgeErrorMessage(checkpointError),
        });
        setErrorMessage(
          `Checkpoint was not accepted: ${bridgeErrorMessage(checkpointError)}`,
        );
        return;
      }
    }

    try {
      const exact = await validateConfigContents(text, yamlPath, catalogState.binaryPath, catalogState.executionTarget);
      if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
      if (!exact.valid) {
        const message = exact.stderr.trim() || exact.stdout.trim()
          || `The exact simulator rejected this configuration (exit ${exact.exitCode ?? "unknown"}).`;
        setPreflight({ kind: "error", evidenceKey, message });
        setErrorMessage(`Exact simulator validation failed: ${message}`);
        return;
      }
      const availableStateFields = authoritativeRuntimeStateFields(
        catalogState.catalog,
      );
      const request = buildConfigPreflightRequest(
        launchConfig,
        catalogState.catalog,
        availableStateFields,
        { yamlText: text, yamlPath },
      );
      const report = await preflightConfigWithBinary(
        request,
        catalogState.binaryPath,
        catalogState.executionTarget,
      );
      // Stale snapshot: an edit raced the preflight; abort.
      if (
        sequence !== preflightSequenceRef.current ||
        evidenceKey !== preflightEvidenceRef.current
      ) {
        return;
      }
      if (!report.compatible) {
        setPreflight({ kind: "blocked", evidenceKey, report });
        return;
      }
      const stagedHandoff = loadCm1GuidedHandoff(window.localStorage);
      const handoffNeedsLaunch = stagedHandoff
        && stagedHandoff.smoke.consumedAt !== null
        && stagedHandoff.run.status !== "complete";
      if (handoffNeedsLaunch && (stagedHandoff.run.status === "running" || stagedHandoff.run.status === "pending")) {
        setErrorMessage(`Guided run ${stagedHandoff.run.runId} is still awaiting durable Run Library qualification. Resolve that evidence before launching another guided smoke run.`);
        return;
      }
      if (handoffNeedsLaunch) {
        const mismatch = cm1StagedLaunchMismatch(stagedHandoff, {
          yaml: text,
          catalogFingerprint: report.executable.catalogFingerprint,
          executablePath: report.executable.canonicalPath,
        });
        if (mismatch) {
          setErrorMessage(
            `The staged CM1 smoke evidence changed: ${mismatch} Restore the exact staged baseline; any YAML, grid, scheme, catalog, or executable edit is excluded from this guided comparison.`,
          );
          return;
        }
      }
      setPreflight({ kind: "passed", evidenceKey, report });
      setSpawn({ kind: "starting" });

      await writeConfig(yamlPath, text);
      if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) {
        setSpawn({ kind: "idle" });
        return;
      }
      const settings = loadStudioSettings();
      const binaryPath = report.executable.canonicalPath;
      const runsRoot = effectiveRunsDirectory(settings.runsDirectory);
      const schemeVal = readPath(launchConfig, "dynamics.scheme");
      const durationVal = readPath(launchConfig, "time.run_time");
      const validationVal = readPath(launchConfig, "extensions.validation.guard_mode");
      const soundingType = readPath(launchConfig, "initial_conditions.sounding.type");
      const soundingPath = readPath(launchConfig, "initial_conditions.sounding.file_path");
      const runId = await spawnSim(yamlPath, {
        binaryPath,
        executionTarget: catalogState.executionTarget,
        expectedExecutionTargetFingerprint: report.executionTargetFingerprint ?? undefined,
        expectedCatalogFingerprint: report.executable.catalogFingerprint,
        expectedBinaryFingerprint: report.executable.binaryFingerprint,
        expectedConfigFingerprint: fnv1a64Text(text),
        runsRoot,
        scheme: typeof schemeVal === "string" ? schemeVal : undefined,
        durationS: typeof durationVal === "number" ? durationVal : undefined,
        requestedValues: {
          backend: request.backend,
          allowFallback: request.allowBackendFallback,
          outputFormat: request.outputFormat,
          executionMode: request.executionMode,
          validationMode: typeof validationVal === "string" ? validationVal : undefined,
          soundingPath:
            soundingType === "file" && typeof soundingPath === "string"
              ? soundingPath
              : undefined,
          schemes: Object.fromEntries(
            request.selections.map(({ moduleId, schemeId }) => [moduleId, schemeId]),
          ),
        },
      });

      const trackedHandoff = handoffNeedsLaunch === true ? stagedHandoff : null;
      if (trackedHandoff) {
        let outputDir: string | null = null;
        try {
          outputDir = (await listRuns()).find((run) => run.id === runId)?.outputDir ?? null;
        } catch {
          // Live registry binds the launch path only; durable Run Library evidence qualifies output.
        }
        const runningHandoff = markCm1SmokeRunning(trackedHandoff, runId, outputDir);
        saveCm1GuidedHandoff(window.localStorage, runningHandoff);
        setCm1Handoff(runningHandoff);
      }

      // A fast-failing child may exit before spawn_sim resolves; apply the buffered exit directly.
      const earlyExit = recentExitsRef.current.get(runId);
      if (earlyExit) {
        recentExitsRef.current.delete(runId);
        setSpawn(
          buildExitState(runId, earlyExit.status, earlyExit.exitCode, earlyExit.stderrTail),
        );
        if (trackedHandoff) {
          void settleCm1GuidedRun(runId, earlyExit.status).then((record) => {
            if (record) setCm1Handoff(record);
          });
        }
      } else {
        setSpawn({ kind: "running", runId });
      }
    } catch (e) {
      if (sequence !== preflightSequenceRef.current || evidenceKey !== preflightEvidenceRef.current) return;
      const message = e instanceof Error ? e.message : String(e);
      // Preserve a passed report if only file-write/spawn failed; a preflight failure always wins.
      setPreflight((current) => {
        if (current.kind === "passed" && current.evidenceKey === evidenceKey) {
          return current;
        }
        return { kind: "error", evidenceKey, message };
      });
      setErrorMessage(`Could not start simulation: ${message}`);
      setSpawn({ kind: "idle" });
    }
  }, [
    yamlPath,
    resolveYamlText,
    config,
    catalogState,
    viewMode,
    preflightEvidenceKey,
  ]);

  const dismissSpawnResult = useCallback(() => {
    setSpawn({ kind: "idle" });
  }, []);

  const toFormMode = useCallback(() => {
    // Attempt to merge raw edits back into the structured config.
    try {
      const parsed = parseYaml(rawYaml);
      const merged = reconcileFormConfig(mergeConfig(defaultsForSchema(), parsed));
      // Raw edits can change a scheme without touching the old variant block.
      // Reconcile at the raw→form boundary so the structured document never
      // briefly contains parameters for an inactive runtime scheme.
      setConfig(
        catalogState.kind === "ready"
          ? normalizeSchemeSelections(catalogState.catalog, merged)
          : merged,
      );
      setRawError(null);
      setViewMode("form");
    } catch (e) {
      setRawError(
        `Cannot switch to form view — YAML invalid: ${
          e instanceof Error ? e.message : String(e)
        }`,
      );
    }
  }, [catalogState, rawYaml]);

  const toRawMode = useCallback(() => {
    setRawYaml(yamlText);
    setRawError(null);
    setViewMode("raw");
  }, [yamlText]);

  // Derived from reason so disabled binding and tooltip stay in sync (see lib/config/launchReadiness.ts).
  const launchReadiness: LaunchReadinessInputs = {
    desktop: isTauriRuntime(),
    catalogKind: catalogState.kind,
    preflightKind: currentPreflight.kind,
    spawnKind: spawn.kind,
    resumeCheckpointPath,
    checkpoint: checkpointState,
  };
  const launchBlockedReason = !yamlPath
    ? "Waiting for writable configuration storage"
    : describeLaunchBlock(launchReadiness);
  const validationBlockedReason = !yamlPath
    ? "Waiting for writable configuration storage"
    : describeValidationBlock(launchReadiness);
  const launchBlocked = launchBlockedReason != null;
  const validationBlocked = validationBlockedReason != null;
  const draftNavigationBlocked = spawn.kind === "starting" || spawn.kind === "running";

  return (
    <main className="config-editor">
      <header className="config-editor__header">
        <label className="config-editor__identity" htmlFor="experiment-name">
          <span className="config-editor__eyebrow">Experiment configuration</span>
          <input
            id="experiment-name"
            type="text"
            className="config-editor__name-input"
            value={caseName}
            onChange={(e) => setCaseName(e.target.value)}
            spellCheck={false}
          />
        </label>
        <div className="config-editor__header-meta">
          <ViewToggle mode={viewMode} onForm={toFormMode} onRaw={toRawMode} />
          <TemplatePicker
            templates={templates}
            selected={selectedTemplate}
            loading={loadingTemplate}
            onSelect={loadTemplate}
          />
          <button
            type="button"
            className="config-editor__upload-btn"
            onClick={onUpload}
            title="Pick a .yaml file from anywhere on disk"
          >
            <Icon name="plus" size={12} />
            <span>Upload…</span>
          </button>
          <span className="badge">{schemeLabel}</span>
          <span className="config-editor__path" title={yamlPath || "Preparing writable configuration storage"}>
            {yamlPath ? yamlPath.split(/[/\\]/).pop() ?? yamlPath : "Preparing storage…"}
          </span>
        </div>
      </header>

      <section className="config-editor__draft-bar" aria-label="Configuration drafts">
        <div className="config-editor__draft-picker">
          <label htmlFor="configuration-draft">Configuration draft</label>
          <select
            id="configuration-draft"
            value={draftStore.activeId}
            disabled={draftNavigationBlocked}
            title={draftNavigationBlocked ? "Configuration drafts cannot change while a simulation is starting or running" : "Restore a local recovery draft"}
            onChange={(event) => selectDraft(event.target.value)}
          >
            {draftStore.drafts.map((draft) => (
              <option value={draft.id} key={draft.id}>
                {draft.caseName || "Untitled experiment"}{draft.parentId ? " · branch" : ""}
              </option>
            ))}
          </select>
          <span>{draftStore.drafts.length}/{CONFIG_DRAFT_LIMIT} local</span>
        </div>
        <div className="config-editor__draft-actions">
          <button
            type="button"
            disabled={draftNavigationBlocked || draftStore.drafts.length >= CONFIG_DRAFT_LIMIT}
            onClick={branchDraft}
            title={draftNavigationBlocked
              ? "A branch cannot be created while a simulation is starting or running"
              : draftStore.drafts.length >= CONFIG_DRAFT_LIMIT
                ? `The ${CONFIG_DRAFT_LIMIT}-draft local recovery limit has been reached`
                : "Copy the current working document into an independent unsaved branch"}
          >
            Branch draft
          </button>
          <button
            type="button"
            className={forgetDraftConfirm ? "is-confirming" : ""}
            disabled={draftNavigationBlocked || draftStore.drafts.length === 1}
            onClick={forgetDraft}
            title={draftNavigationBlocked
              ? "A draft cannot be forgotten while a simulation is starting or running"
              : draftStore.drafts.length === 1
                ? "Keep at least one local recovery draft"
                : "Forget this local recovery draft after confirmation; saved files are never deleted"}
          >
            {forgetDraftConfirm ? "Confirm forget" : "Forget draft"}
          </button>
          {forgetDraftConfirm && (
            <button type="button" onClick={() => setForgetDraftConfirm(false)}>Keep</button>
          )}
        </div>
        <div
          className={`config-editor__draft-status is-${draftPersistence}${draftDirty ? " is-dirty" : ""}`}
          role={draftPersistence === "error" ? "alert" : "status"}
          title={draftMessage ?? activeDraft.saved?.path ?? "Local recovery draft"}
        >
          <span aria-hidden="true" />
          <div>
            <strong>
              {draftPersistence === "error"
                ? "Local recovery unavailable"
                : draftPersistence === "saving"
                  ? "Saving recovery draft…"
                  : draftDirty
                    ? "File has unsaved changes"
                    : "File save is current"}
            </strong>
            <span>
              {draftMessage
                ?? (draftPersistence === "current"
                  ? draftDirty
                    ? "Working bytes are retained locally. Save writes the named YAML file."
                    : activeDraft.saved?.path ?? "Working bytes are retained locally."
                  : "The active branch remains open while local recovery is updated.")}
            </span>
            {draftRecoveryBlocked && (
              <button type="button" onClick={replaceUnreadableDraftRecovery}>
                Replace unreadable recovery data
              </button>
            )}
          </div>
        </div>
      </section>

      <section className="config-editor__compute-target" aria-label="Compute destination">
        <label htmlFor="compute-target">Compute target</label>
        <select id="compute-target" value={computeTargetId} disabled={spawn.kind === "starting" || spawn.kind === "running"}
          title={spawn.kind === "starting" || spawn.kind === "running" ? "The compute target cannot change while a simulation is starting or running" : "Choose where to validate and run the simulator"}
          onChange={event => {
            preflightSequenceRef.current++;
            setCatalogState({ kind: "loading" });
            setPreflight({ kind: "required" });
            setComputeTargetId(event.target.value);
          }}>
          <option value="local">This computer</option>
          {executionTarget === undefined && <option value={computeTargetId}>Unavailable saved connection</option>}
          {remoteProfiles.profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.name} · {profile.host}</option>)}
        </select>
        <span>{executionTarget ? `Runs on ${executionTarget.host}; plots remain on this computer.` : "Add remote connections in Settings."}</span>
        {executionTarget && <span>Keep this computer awake and Studio connected until the run finishes. Closing Studio or losing SSH requests a stop on the node. For rendering on the node, use Tools → Remote visualization workspace in Plots.</span>}
        {catalogState.kind === "error" && computeTargetId !== "local" && <span role="alert">{catalogState.message}</span>}
        {remoteProfiles.error && <span role="alert">{remoteProfiles.error}</span>}
      </section>

      {cm1Handoff && (
        <section className="config-editor__cm1-handoff" aria-labelledby="cm1-handoff-title">
          <div>
            <span className="config-editor__eyebrow">CM1 guided handoff</span>
            <strong id="cm1-handoff-title">Reviewed sounding · short Nimbus smoke test</strong>
            <span>
              {cm1Handoff.evidence.historyFileCount} CM1 history file{cm1Handoff.evidence.historyFileCount === 1 ? "" : "s"} retained for comparison · {cm1Handoff.evidence.projectedLevelCount} sounding levels reviewed
            </span>
          </div>
          <div className="config-editor__cm1-handoff-state">
            <strong>
              {cm1Handoff.evidence.soundingSourceMode === "native_cm1" ? "Native CM1 source" : "Reviewed projection"}
              {" · "}{cm1Handoff.smoke.durationS} s run · {cm1Handoff.smoke.outputIntervalS} s output
            </strong>
            <span>
              Mapping review: {cm1Handoff.evidence.mappingCounts.ready} ready · {cm1Handoff.evidence.mappingCounts.needsDecision} decisions · {cm1Handoff.evidence.mappingCounts.different} different · {cm1Handoff.evidence.mappingCounts.unsupported} unsupported
            </span>
            {cm1Handoff.evidence.mappingBlockerTotal > 0 && (
              <span>
                {cm1Handoff.evidence.mappingBlockerTotal} mapping blocker{cm1Handoff.evidence.mappingBlockerTotal === 1 ? "" : "s"} retained
                {cm1Handoff.evidence.mappingBlockersTruncated ? ` · showing ${cm1Handoff.evidence.mappingBlockers.length}` : ""}. The smoke baseline imports only the reviewed sounding.
              </span>
            )}
            <span>
              {cm1Handoff.run.status === "complete"
                ? `Durable output verified at ${cm1Handoff.run.outputDir}. Open Plots to re-verify and compare.`
                : cm1Handoff.run.status === "running" || cm1Handoff.run.status === "pending"
                  ? `Run ${cm1Handoff.run.runId}: ${cm1Handoff.run.pendingReason}`
                : cm1Handoff.run.status === "failed"
                    ? `${cm1Handoff.run.pendingReason} Restore the staged baseline and retry after reviewing the failure.`
                    : "Next: run this exact baseline, then open Plots for the qualified CM1 ↔ Nimbus comparison."}
            </span>
          </div>
          <button type="button" onClick={restoreCm1SmokeConfig}>Restore staged baseline</button>
        </section>
      )}

      <div className="config-editor__body">
        {viewMode === "form" ? (
          <>
            <div className="config-editor__decision-heading">
              <div>
                <span className="config-editor__eyebrow">Current evidence</span>
                <h2>Domain and run envelope</h2>
              </div>
              <p>Recomputed from the values below before launch.</p>
            </div>
            <DiagnosticsStrip
              stats={diagnostics.stats}
              warnings={diagnostics.warnings}
            />

            <div className="config-editor__decision-heading config-editor__decision-heading--parameters">
              <div>
                <span className="config-editor__eyebrow">Experiment definition</span>
                <h2>Configuration decisions</h2>
              </div>
              <div className="config-editor__decision-controls">
                <p>
                  {detailLevel === "basic"
                    ? `${visibleSchemaSections.length} core sections shown · ${specialistSectionCount} specialist sections retain their current YAML values.`
                    : `All ${schemaSections.length} configuration sections shown.`}
                </p>
                <ConfigDetailToggle value={detailLevel} onChange={changeDetailLevel} />
                <button
                  type="button"
                  className={`config-compatibility-toggle ${showCompatibilityNames ? "is-active" : ""}`}
                  aria-pressed={showCompatibilityNames}
                  onClick={() => setShowCompatibilityNames((shown) => !shown)}
                >
                  CM1 crosswalk
                </button>
              </div>
            </div>
            {showCompatibilityNames && <Cm1OrientationGuide />}
            <div className="config-editor__sections">
              {visibleSchemaSections.map((section) => (
                <SchemaSection
                  key={section.id}
                  section={section}
                  expanded={expanded.has(section.id)}
                  onToggle={() => toggle(section.id)}
                  config={config}
                  onChange={changeConfig}
                  showCompatibilityNames={showCompatibilityNames}
                  supplement={
                    section.id === "sounding" ? (
                      <NativeCm1SoundingImport
                        configuredPath={String(readPath(config, "initial_conditions.sounding.file_path") ?? "")}
                        hodographType={String(readPath(config, "initial_conditions.hodograph.type") ?? "custom")}
                        modelTopM={typeof readPath(config, "grid.ztop") === "number"
                          ? readPath(config, "grid.ztop") as number
                          : null}
                        desktop={isTauriRuntime()}
                        onAdopt={(path) => {
                          let next = writePath(config, "initial_conditions.sounding.type", "file");
                          next = writePath(next, "initial_conditions.sounding.file_path", path);
                          next = writePath(next, "initial_conditions.hodograph.type", "auto");
                          changeConfig(next);
                        }}
                        onUseEmbeddedWinds={() => changeConfig(
                          writePath(config, "initial_conditions.hodograph.type", "auto"),
                        )}
                      />
                    ) : section.id === "checkpoint" ? (
                      <CheckpointInspector
                        state={checkpointState}
                        configuredPath={resumeCheckpointPath}
                        desktop={isTauriRuntime()}
                        remote={executionTarget != null}
                        onChoose={chooseCheckpoint}
                        onClear={clearCheckpoint}
                      />
                    ) : section.id === "radar" ? (
                      <RadarCatalogControl
                        config={config}
                        catalog={catalogState.kind === "ready" ? catalogState.catalog : null}
                        onChange={changeConfig}
                      />
                    ) : undefined
                  }
                />
              ))}
            </div>
          </>
        ) : (
          <RawYamlView
            value={rawYaml}
            onChange={(next) => {
              setRawYaml(next);
              setRawError(null);
            }}
            parseError={rawError}
          />
        )}

        {errorMessage && (
          <div className="callout callout--error" role="alert">
            <Icon name="warning" size={14} />
            <div className="callout__body">
              <span className="callout__title">Nimbus could not complete that action</span>
              <span className="callout__detail">
                {errorMessage} Your in-memory configuration is unchanged; correct the
                reported issue and try again.
              </span>
            </div>
            <button
              type="button"
              className="callout__dismiss"
              onClick={() => setErrorMessage(null)}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        {schemaResult && (
          <div
            className={
              schemaResult.status === "v1_with_warnings"
                ? "callout callout--warn"
                : "callout callout--error"
            }
            role="alert"
          >
            <Icon
              name={
                schemaResult.status === "v1_with_warnings"
                  ? "info"
                  : "warning"
              }
              size={14}
            />
            <div className="callout__body">
              <span className="callout__title">
                {summarizeSchema(schemaResult)}
              </span>
              {(schemaResult.errors.length > 0 ||
                schemaResult.warnings.length > 0) && (
                <ul className="callout__list">
                  {schemaResult.errors.map((e, i) => (
                    <li key={"e" + i}>
                      {e.path ? <strong>{e.path}: </strong> : null}
                      {e.message}
                    </li>
                  ))}
                  {schemaResult.warnings.map((w, i) => (
                    <li key={"w" + i}>
                      {w.path ? <strong>{w.path}: </strong> : null}
                      {w.message}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <button
              type="button"
              className="callout__dismiss"
              onClick={() => setSchemaResult(null)}
              aria-label="Dismiss schema details"
            >
              ×
            </button>
          </div>
        )}

        {migrationNotice && (
          <div className="callout callout--warn" role="status">
            <Icon name="info" size={14} />
            <div className="callout__body">
              <span className="callout__title">Legacy configuration migrated to an unsaved draft</span>
              <span className="callout__detail">{migrationNotice}</span>
            </div>
            <button
              type="button"
              className="callout__dismiss"
              onClick={() => setMigrationNotice(null)}
              aria-label="Dismiss migration details"
            >
              ×
            </button>
          </div>
        )}

        {migrationReview && (
          <div className="callout callout--warn" role="alert">
            <Icon name="info" size={14} />
            <div className="callout__body">
              <span className="callout__title">Review legacy migration before using it</span>
              <span className="callout__detail">
                {migrationReview.sourcePath} remains untouched. The simulator produced a {migrationReview.migratedText.split("\n").length}-line v1 candidate from {migrationReview.originalText.split("\n").length} legacy lines.
              </span>
              {migrationReview.warnings && (
                <span className="callout__detail">{migrationReview.warnings}</span>
              )}
              <details>
                <summary>Compare source and migrated YAML</summary>
                <strong>Legacy source</strong>
                <pre>{migrationReview.originalText.slice(0, 12_000)}</pre>
                <strong>Migrated nimbus.v1 candidate</strong>
                <pre>{migrationReview.migratedText.slice(0, 12_000)}</pre>
              </details>
              <div>
                <button type="button" onClick={acceptMigration}>Use migrated draft</button>
                <button type="button" onClick={() => setMigrationReview(null)}>Cancel</button>
              </div>
            </div>
          </div>
        )}

        <SpawnStatusCallout spawn={spawn} onDismiss={dismissSpawnResult} />

        {catalogState.kind === "ready" && currentPreflight.kind !== "required" && (
          <PreflightReadiness
            state={currentPreflight}
            desktop={isTauriRuntime()}
            catalogReady
          />
        )}
      </div>

      <footer className="config-editor__launch-dock" aria-label="Experiment readiness and actions">
        <div className="config-editor__launch-dock-inner">
          <ActionDockEvidence
            state={currentPreflight}
            desktop={isTauriRuntime()}
            catalogReady={catalogState.kind === "ready"}
            catalogMessage={catalogState.kind === "error" ? catalogState.message : null}
            stats={diagnostics.stats}
          />
          <div className="config-editor__actions">
          <button
            type="button"
            className="btn btn--secondary"
            onClick={save}
            disabled={savingState === "saving" || !yamlPath}
            title={!yamlPath ? "Waiting for writable configuration storage" : savingState === "saving" ? "A configuration save is already in progress" : "Save this configuration"}
          >
            <Icon name="plus" size={14} />
            <span>
              {savingState === "saving"
                ? "Saving…"
                : savingState === "saved"
                  ? "Saved ✓"
                  : "Save"}
            </span>
          </button>
          <button
            type="button"
            className="btn btn--secondary"
            onClick={saveAs}
            disabled={savingState === "saving"}
            title={savingState === "saving" ? "A configuration save is already in progress" : "Choose a new configuration file"}
          >
            <Icon name="folder" size={14} />
            <span>Save As…</span>
          </button>
          <button
            type="button"
            className="btn btn--secondary"
            onClick={() => void validateConfiguration()}
            disabled={validationBlocked}
            title={
              !isTauriRuntime()
                ? "Exact-binary validation is available in Nimbus Studio desktop"
                : validationBlockedReason ?? "Validate without saving or starting a simulation"
            }
          >
            <Icon name="info" size={14} />
            <span>{currentPreflight.kind === "pending" ? "Validating…" : "Validate"}</span>
          </button>
          <button
            type="button"
            className="btn btn--primary"
            onClick={runSimulation}
            disabled={launchBlocked}
            title={launchBlockedReason ?? "Preflight the exact binary and start this configuration"}
          >
            <Icon name="play" size={14} />
            <span>
              {spawn.kind === "starting"
                ? "Starting…"
                : currentPreflight.kind === "pending"
                  ? "Checking exact binary…"
                : spawn.kind === "running"
                  ? "Running — start another"
                  : "Run simulation"}
            </span>
          </button>
          </div>
        </div>
      </footer>
    </main>
  );
}

async function settleCm1GuidedRun(runId: string, status: string): Promise<Cm1GuidedHandoff | null> {
  const record = loadCm1GuidedHandoff(window.localStorage);
  if (!record
    || (record.run.status !== "running"
      && record.run.status !== "pending"
      && record.run.status !== "complete")
    || record.run.runId !== runId) return record;
  if (status !== "completed" && status !== "reconcile") {
    const failed = markCm1SmokeFailed(record, `Process exit reported ${status}; no comparison output was qualified.`);
    saveCm1GuidedHandoff(window.localStorage, failed);
    return failed;
  }
  // Re-scan durable archive; process memory is never completion evidence.
  try {
    const runsRoot = effectiveRunsDirectory(loadStudioSettings().runsDirectory);
    const entries = await discoverRunLibrary(runsRoot);
    const expectedPath = record.run.outputDir?.replace(/[\\/]+$/, "") ?? null;
    let matching = expectedPath
      ? entries.find((entry) => entry.outputDir.replace(/[\\/]+$/, "") === expectedPath) ?? null
      : null;
    if (!expectedPath) {
      const byDurableRunId = entries.filter((entry) => entry.provenance.runId === runId);
      if (byDurableRunId.length > 1) {
        const pending = markCm1SmokePending(
          record,
          "Durable Run Library scan found multiple outputs with this run identity; comparison remains blocked.",
        );
        saveCm1GuidedHandoff(window.localStorage, pending);
        return pending;
      }
      matching = byDurableRunId[0] ?? null;
    }
    const reconciled = reconcileCm1SmokeRun(record, matching);
    if (reconciled !== record) saveCm1GuidedHandoff(window.localStorage, reconciled);
    return reconciled;
  } catch (error) {
    const pending = markCm1SmokePending(
      record,
      `Durable Run Library scan failed: ${bridgeErrorMessage(error)}. Retry by reopening Configure.`,
    );
    saveCm1GuidedHandoff(window.localStorage, pending);
    return pending;
  }
}

/** Map a run:exit event to SpawnState; shared by the subscription handler and early-exit path. */
function buildExitState(
  runId: string,
  status: string,
  exitCode: number | null,
  stderrTail: string[],
): SpawnState {
  if (status === "completed") {
    return { kind: "completed", runId };
  }
  const narrowed = status === "aborted" ? "aborted" : "failed";
  const reason = summarizeFailure(narrowed, exitCode, stderrTail);
  return { kind: "failed", runId, exitCode, stderrTail, reason };
}

function bridgeErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (
    error !== null &&
    typeof error === "object" &&
    "message" in error &&
    typeof error.message === "string"
  ) {
    return error.message;
  }
  return String(error);
}

/** Build a short failure summary; prefers the most recent [FATAL/ERROR/CONFIG] tag from stderr. */
function summarizeFailure(
  status: "completed" | "failed" | "aborted" | "queued" | "running",
  exitCode: number | null,
  stderrTail: string[],
): string {
  if (status === "aborted") {
    return "Simulation was aborted before completion.";
  }
  const tagged = [...stderrTail].reverse().find((line) =>
    /\[(FATAL|ERROR|CONFIG)\]/.test(line),
  );
  if (tagged) return tagged.trim();
  if (exitCode != null && exitCode < 0 && exitCode >= -64) return processExitSummary(exitCode);
  const lastNonEmpty = [...stderrTail].reverse().find((l) => l.trim().length > 0);
  if (lastNonEmpty) return lastNonEmpty.trim();
  return exitCode != null
    ? `${processExitSummary(exitCode)} No stderr output was captured.`
    : "Simulator exited without producing stderr output.";
}

interface SpawnStatusCalloutProps {
  spawn: SpawnState;
  onDismiss: () => void;
}

function SpawnStatusCallout({ spawn, onDismiss }: SpawnStatusCalloutProps) {
  if (spawn.kind === "idle" || spawn.kind === "starting") return null;

  if (spawn.kind === "running") {
    return (
      <div className="callout callout--info" role="status">
        <Icon name="dot" size={14} />
        <div className="callout__body">
          <span className="callout__title">Simulation running</span>
          <span className="callout__detail">
            Run id <code>{spawn.runId}</code> · open the Runs tab for live logs.
          </span>
        </div>
        <button
          type="button"
          className="callout__dismiss"
          onClick={onDismiss}
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    );
  }

  if (spawn.kind === "completed") {
    return (
      <div className="callout callout--ok" role="status">
        <Icon name="dot" size={14} />
        <div className="callout__body">
          <span className="callout__title">Simulation completed</span>
          <span className="callout__detail">
            Run id <code>{spawn.runId}</code> · outputs ready in the Plots tab.
          </span>
        </div>
        <button
          type="button"
          className="callout__dismiss"
          onClick={onDismiss}
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    );
  }

  // Failed (or aborted) — show as much detail as we have.
  return (
    <div className="callout callout--error" role="alert">
      <Icon name="warning" size={14} />
      <div className="callout__body">
        <span className="callout__title">
          Simulation failed
          {spawn.exitCode != null && (
            <span className="callout__code"> (exit {spawn.exitCode})</span>
          )}
        </span>
        <span className="callout__detail">{spawn.reason}</span>
        {spawn.stderrTail.length > 0 && (
          <details className="callout__details">
            <summary>
              Show stderr ({spawn.stderrTail.length} line
              {spawn.stderrTail.length === 1 ? "" : "s"})
            </summary>
            <pre className="callout__pre">{spawn.stderrTail.join("\n")}</pre>
          </details>
        )}
      </div>
      <button
        type="button"
        className="callout__dismiss"
        onClick={onDismiss}
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
}

interface PreflightReadinessProps {
  state: PreflightState;
  desktop: boolean;
  catalogReady: boolean;
}

function PreflightReadiness({
  state,
  desktop,
  catalogReady,
}: PreflightReadinessProps) {
  if (!desktop) {
    return (
      <section className="preflight-readiness preflight-readiness--blocked" role="alert">
        <PreflightMarker label="Unavailable" />
        <div className="preflight-readiness__body">
          <strong>Simulation launch unavailable</strong>
          <span>
            Continue editing or saving this configuration. Validation and simulation launch
            require a connected desktop simulator.
          </span>
        </div>
      </section>
    );
  }

  if (!catalogReady) {
    return (
      <section className="preflight-readiness" role="status">
        <PreflightMarker label="Waiting" />
        <div className="preflight-readiness__body">
          <strong>Launch evidence unavailable</strong>
          <span>The configured simulator catalog must load before compatibility can be checked.</span>
        </div>
      </section>
    );
  }

  if (state.kind === "required") {
    return (
      <section className="preflight-readiness" role="status">
        <PreflightMarker label="Required" />
        <div className="preflight-readiness__body">
          <strong>Exact-binary launch check</strong>
          <span>
            Run will first verify every selected scheme, backend, grid, initialization source,
            and allocated state field against the executable that will be spawned.
          </span>
        </div>
      </section>
    );
  }

  if (state.kind === "pending") {
    return (
      <section className="preflight-readiness preflight-readiness--pending" role="status" aria-live="polite">
        <PreflightMarker label="Checking" />
        <div className="preflight-readiness__body">
          <strong>Reading the exact simulator</strong>
          <span>Launch is held while Nimbus verifies the current configuration.</span>
        </div>
      </section>
    );
  }

  if (state.kind === "error") {
    return (
      <section className="preflight-readiness preflight-readiness--blocked" role="alert">
        <PreflightMarker label="Unavailable" />
        <div className="preflight-readiness__body">
          <strong>Preflight could not establish evidence</strong>
          <span>
            No run was started. {state.message} Check the simulator path in Settings,
            then run the check again.
          </span>
        </div>
      </section>
    );
  }

  const report = state.report;
  const qualified = report.resolvedSelections.filter(
    (selection) => selection.maturity === "qualified",
  ).length;
  const binaryName = report.executable.canonicalPath.split(/[/\\]/).pop() || "nimbus_sim";

  return (
    <section
      className={`preflight-readiness preflight-readiness--${state.kind}`}
      role={state.kind === "blocked" ? "alert" : "status"}
      aria-live="polite"
    >
      <PreflightMarker label={state.kind === "blocked" ? "Blocked" : "Passed"} />
      <div className="preflight-readiness__body">
        <strong>
          {state.kind === "blocked"
            ? `${report.hardErrors.length} launch blocker${report.hardErrors.length === 1 ? "" : "s"}`
            : "Configuration accepted by the exact binary"}
        </strong>
        {state.kind === "blocked" && (
          <span>
            Nothing was launched. Correct the findings below, then run this configuration
            again to produce fresh evidence.
          </span>
        )}
        <span className="preflight-readiness__evidence" title={report.executable.canonicalPath}>
          {binaryName} · {report.executable.catalogFingerprint} · {qualified}/
          {report.resolvedSelections.length} selections qualified
        </span>
        {report.hardErrors.length > 0 && (
          <PreflightIssueList label="Blocking findings" issues={report.hardErrors} />
        )}
        {report.warnings.length > 0 && (
          <PreflightIssueList label="Qualification warnings" issues={report.warnings} />
        )}
        {state.kind === "passed" && report.warnings.length > 0 && (
          <span className="preflight-readiness__disclosure">
            Warnings do not block launch, but these selections are not represented as fully
            qualified.
          </span>
        )}
      </div>
    </section>
  );
}

function PreflightMarker({ label }: { label: string }) {
  return (
    <div className="preflight-readiness__marker" aria-hidden="true">
      <span />
      {label}
    </div>
  );
}

function PreflightIssueList({
  label,
  issues,
}: {
  label: string;
  issues: ConfigPreflightIssue[];
}) {
  return (
    <details className="preflight-readiness__issues" open={label === "Blocking findings"}>
      <summary>{label} · {issues.length}</summary>
      <ul>
        {issues.map((issue, index) => (
          <li key={`${issue.code}:${issue.moduleId ?? "runtime"}:${index}`}>
            {issue.moduleId && <code>{issue.moduleId}</code>}
            <span>{issue.message}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

interface ViewToggleProps {
  mode: ViewMode;
  onForm: () => void;
  onRaw: () => void;
}

/* role="radiogroup" (no tabpanel, no arrow-key nav). Mirrors the theme switcher in Settings.tsx. */
function ViewToggle({ mode, onForm, onRaw }: ViewToggleProps) {
  return (
    <div className="view-toggle" role="radiogroup" aria-label="Editor view">
      <button
        type="button"
        role="radio"
        className={`view-toggle__btn ${mode === "form" ? "view-toggle__btn--active" : ""}`}
        aria-checked={mode === "form"}
        onClick={onForm}
      >
        Form
      </button>
      <button
        type="button"
        role="radio"
        className={`view-toggle__btn ${mode === "raw" ? "view-toggle__btn--active" : ""}`}
        aria-checked={mode === "raw"}
        onClick={onRaw}
      >
        YAML
      </button>
    </div>
  );
}

function ConfigDetailToggle({
  value,
  onChange,
}: {
  value: ConfigDetailLevel;
  onChange: (next: ConfigDetailLevel) => void;
}) {
  return (
    <div className="config-detail-toggle" role="group" aria-label="Configuration detail">
      {(["basic", "advanced"] as const).map((detail) => (
        <button
          key={detail}
          type="button"
          className={value === detail ? "config-detail-toggle__button config-detail-toggle__button--active" : "config-detail-toggle__button"}
          aria-pressed={value === detail}
          onClick={() => onChange(detail)}
        >
          {detail === "basic" ? "Basic" : "Advanced"}
        </button>
      ))}
    </div>
  );
}

function ActionDockEvidence({
  state,
  desktop,
  catalogReady,
  catalogMessage,
  stats,
}: {
  state: PreflightState;
  desktop: boolean;
  catalogReady: boolean;
  catalogMessage: string | null;
  stats: DiagStat[];
}) {
  let tone: "ready" | "required" | "checking" | "blocked" = "required";
  let title = "Validation required";
  let detail = "Run performs an exact-binary check before launch.";
  if (!desktop) {
    tone = "required";
    title = "Configuration editing ready";
    detail = "Validation and simulation runs are available in the desktop app.";
  } else if (!catalogReady) {
    tone = "required";
    title = "Simulator not connected";
    detail = catalogMessage
      ? "Editing is available. Connect a Nimbus simulator in Settings when you are ready to validate or run."
      : "Looking for a Nimbus simulator. Editing remains available.";
  } else if (state.kind === "pending") {
    tone = "checking";
    title = "Checking exact binary";
    detail = "Launch is held until current evidence returns.";
  } else if (state.kind === "error") {
    tone = "blocked";
    title = "Validation unavailable";
    detail = state.message;
  } else if (state.kind === "blocked") {
    tone = "blocked";
    title = `${state.report.hardErrors.length} blocking finding${state.report.hardErrors.length === 1 ? "" : "s"}`;
    detail = "Review the readiness evidence above; no run can start.";
  } else if (state.kind === "passed") {
    tone = "ready";
    title = state.report.warnings.length > 0
      ? `Validation passed with ${state.report.warnings.length} warning${state.report.warnings.length === 1 ? "" : "s"}`
      : "Validation passed";
    detail = "Evidence matches the current document and executable. Use a short run to check GPU startup and runtime stability.";
  }

  return (
    <div
      className={`action-dock-evidence action-dock-evidence--${tone}`}
      role={tone === "blocked" ? "alert" : "status"}
      aria-live="polite"
    >
      <div className="action-dock-evidence__readiness">
        <span className="action-dock-evidence__marker" aria-hidden="true" />
        <div>
          <strong>{title}</strong>
          <span>{detail}</span>
        </div>
      </div>
      <dl className="action-dock-evidence__resources" aria-label="Estimated resources">
        {stats.slice(0, 3).map((stat) => (
          <div key={stat.label} title={stat.hint} data-tone={stat.tone}>
            <dt>{stat.label}</dt>
            <dd>{stat.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

interface RawYamlViewProps {
  value: string;
  onChange: (next: string) => void;
  parseError: string | null;
}

function RawYamlView({ value, onChange, parseError }: RawYamlViewProps) {
  return (
    <div className="raw-yaml">
      {parseError && (
        <div className="callout callout--warn">
          <Icon name="warning" size={14} />
          <span>{parseError}</span>
        </div>
      )}
      <textarea
        className="raw-yaml__doc"
        aria-label="Config YAML editor"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={false}
      />
      <p className="raw-yaml__hint">
        Full YAML editing is enabled. Add, remove, or paste any configuration;
        the selected simulator binary validates the exact document before launch.
      </p>
    </div>
  );
}

interface TemplatePickerProps {
  templates: string[];
  selected: string;
  loading: boolean;
  onSelect: (name: string) => void;
}

function TemplatePicker({
  templates,
  selected,
  loading,
  onSelect,
}: TemplatePickerProps) {
  return (
    <select
      className="config-editor__template-picker"
      value={selected}
      disabled={loading}
      title={loading ? "Waiting for the template catalog" : "Replace the working document with a packaged template"}
      onChange={(e) => onSelect(e.target.value)}
    >
      <option value="">Load template…</option>
      {templates.map((t) => (
        <option key={t} value={t}>
          {t.replace(/\.ya?ml$/, "")}
        </option>
      ))}
    </select>
  );
}

interface DiagnosticsStripProps {
  stats: DiagStat[];
  warnings: DiagWarning[];
}

function DiagnosticsStrip({ stats, warnings }: DiagnosticsStripProps) {
  return (
    <section className="diagnostics" aria-label="Computed configuration evidence">
      <div className="diagnostics__stats">
        {stats.map((s) => (
          <div
            key={s.label}
            className={`diagnostics__stat diagnostics__stat--${s.tone}`}
            title={s.hint}
          >
            <span className="diagnostics__stat-label">{s.label}</span>
            <span className="diagnostics__stat-value">{s.value}</span>
          </div>
        ))}
      </div>
      {warnings.length > 0 && (
        <ul className="diagnostics__warnings">
          {warnings.map((w, i) => (
            <li
              key={i}
              className={`diagnostics__warning diagnostics__warning--${w.tone}`}
            >
              <Icon name="warning" size={12} />
              <span>{w.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

interface SchemaSectionProps {
  section: SectionSchema;
  expanded: boolean;
  onToggle: () => void;
  config: ConfigObject;
  onChange: (next: ConfigObject) => void;
  showCompatibilityNames?: boolean;
  supplement?: ReactNode;
}

function SchemaSection({
  section,
  expanded,
  onToggle,
  config,
  onChange,
  showCompatibilityNames = false,
  supplement,
}: SchemaSectionProps) {
  const cm1Landmark = cm1SectionLandmark(section.id);
  const moduleOff =
    !!section.disable &&
    readPath(config, section.disable.path) === section.disable.offValue;
  return (
    <div
      className={`collapsible ${expanded ? "collapsible--open" : ""} ${
        moduleOff ? "collapsible--module-off" : ""
      }`}
    >
      <button
        type="button"
        className="collapsible__head"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <Icon name={expanded ? "chevron-down" : "chevron-right"} size={12} />
        <span className="collapsible__identity">
          {showCompatibilityNames && cm1Landmark && (
            <span className="collapsible__cm1-landmark">
              CM1 {cm1Landmark.familiarConcept}
            </span>
          )}
          <span className="collapsible__label">{section.label}</span>
          {showCompatibilityNames && cm1Landmark && section.path && (
            <code className="collapsible__canonical-id">{section.path}</code>
          )}
        </span>
        <span className="collapsible__desc">{section.description}</span>
        {section.locked && (
          <span className="collapsible__pill collapsible__pill--locked">
            <Icon name="lock" size={10} />
            Locked
          </span>
        )}
        {moduleOff && (
          <span className="collapsible__pill collapsible__pill--off">
            Disabled
          </span>
        )}
      </button>
      {expanded && (
        <div className="collapsible__body">
          {showCompatibilityNames && cm1Landmark && (
            <div className="cm1-section-context">
              <div>
                <span className="cm1-section-context__label">CM1 landmarks</span>
                <span className="cm1-section-context__keys">
                  {cm1Landmark.cm1Keys.join(" · ")}
                </span>
              </div>
              <p>{cm1Landmark.note}</p>
            </div>
          )}
          {section.locked && (
            <p className="section-locked-note">
              <Icon name="lock" size={12} />
              <span>
                Config — set by the template or a preset. To change these, load
                a different template, apply a preset, or edit a value in the
                YAML view.
              </span>
            </p>
          )}
          {section.disable && (
            <ModuleDisableControl
              section={section}
              config={config}
              onChange={onChange}
            />
          )}
          <SchemaForm
            section={
              section.id === "checkpoint"
                ? { ...section, fields: section.fields.filter((field) => field.key !== "resume_from") }
                : section
            }
            config={config}
            onChange={onChange}
            showCompatibilityNames={showCompatibilityNames}
          />
          {supplement}
        </div>
      )}
    </div>
  );
}

function Cm1OrientationGuide() {
  return (
    <details className="cm1-orientation-guide">
      <summary>
        <span className="cm1-orientation-guide__index" aria-hidden="true">CM1</span>
        <span>
          <strong>Coming from CM1?</strong>
          <small>Familiar controls now lead each maintained crosswalk.</small>
        </span>
        <span className="cm1-orientation-guide__chevron" aria-hidden="true">›</span>
      </summary>
      <div className="cm1-orientation-guide__body">
        <p>
          Use the CM1 names as landmarks, then make the Nimbus decision shown beneath them.
          These labels orient the workflow; they do not claim solver or scheme equivalence.
        </p>
        <dl>
          <div>
            <dt>Same scalar</dt>
            <dd>A maintained value-and-unit transfer for the stated condition.</dd>
          </div>
          <div>
            <dt>Review mapping</dt>
            <dd>A related Nimbus control exists, but context must be reconciled.</dd>
          </div>
          <div>
            <dt>Different formulation</dt>
            <dd>The concept is familiar; values and implementation identity do not transfer.</dd>
          </div>
        </dl>
        <p className="cm1-orientation-guide__boundary">
          Exact Nimbus paths remain visible as secondary evidence. CM1 execution, restart
          consumption, and HPC orchestration are outside Studio’s configuration workflow.
        </p>
      </div>
    </details>
  );
}

interface CheckpointInspectorProps {
  state: CheckpointState;
  configuredPath: string;
  desktop: boolean;
  remote?: boolean;
  onChoose: () => void;
  onClear: () => void;
}

const RADAR_PRODUCT_OPTIONS = [
  { value: "reflectivity_dbz", label: "Reflectivity (dBZ)" },
  { value: "radial_velocity", label: "Radial velocity" },
  { value: "zdr", label: "Differential reflectivity (ZDR)" },
  { value: "kdp", label: "Specific differential phase (KDP)" },
  { value: "rhohv", label: "Copolar correlation (rhoHV)" },
] as const;

interface RadarCatalogControlProps {
  config: ConfigObject;
  catalog: SchemeCatalog | null;
  onChange: (next: ConfigObject) => void;
}

function RadarCatalogControl({ config, catalog, onChange }: RadarCatalogControlProps) {
  const radarModule = catalog?.modules.find((module) => module.id === "radar");
  const advertised = new Set(
    radarModule?.schemes.flatMap((scheme) => [scheme.id, ...scheme.aliases]) ?? [],
  );
  const options = RADAR_PRODUCT_OPTIONS.filter((option) => advertised.has(option.value));
  const enabled = readPath(config, "radar.enabled") === true;
  const rawProducts = readPath(config, "radar.products");
  const selected = Array.isArray(rawProducts)
    ? rawProducts.filter((product): product is string => typeof product === "string")
    : [];

  const setEnabled = (nextEnabled: boolean) => {
    let next = writePath(config, "radar.enabled", nextEnabled);
    if (!nextEnabled) {
      next = writePath(next, "radar.products", []);
    } else if (selected.length === 0) {
      const defaults = ["reflectivity_dbz", "radial_velocity"].filter((product) =>
        advertised.has(product));
      next = writePath(next, "radar.products", defaults.length > 0 ? defaults : [options[0]?.value ?? ""]);
    }
    onChange(next);
  };

  const toggleProduct = (product: string) => {
    const active = selected.includes(product);
    if (active && selected.length === 1) return;
    const products = RADAR_PRODUCT_OPTIONS
      .map((option) => option.value)
      .filter((candidate) => candidate === product ? !active : selected.includes(candidate));
    let next = writePath(config, "radar.products", products);
    if (!active && (product === "kdp" || product === "rhohv")) {
      next = writePath(next, "radar.zdr_tier", "zhang2021_rain_mp");
    }
    onChange(next);
  };

  return (
    <section className="radar-catalog-control" aria-label="Radar product selection">
      <div className="radar-catalog-control__header">
        <div>
          <strong>Diagnostic product set</strong>
          <span>Published by the connected simulator's exact catalog.</span>
        </div>
        <button
          type="button"
          className={`toggle ${enabled ? "toggle--on" : ""}`}
          disabled={!radarModule || options.length === 0}
          onClick={() => setEnabled(!enabled)}
          aria-pressed={enabled}
          aria-label="Enable radar diagnostics"
        >
          <span className="toggle__thumb" />
        </button>
      </div>
      {enabled && (
        <div className="radar-catalog-control__products" role="group" aria-label="Radar products">
          {options.map((option) => {
            const checked = selected.includes(option.value);
            return (
              <label key={option.value}>
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={checked && selected.length === 1}
                  onChange={() => toggleProduct(option.value)}
                />
                <span>{option.label}</span>
              </label>
            );
          })}
        </div>
      )}
      {!radarModule && (
        <p>The connected simulator has not published its radar product set.</p>
      )}
    </section>
  );
}

function CheckpointInspector({
  state,
  configuredPath,
  desktop,
  remote = false,
  onChoose,
  onClear,
}: CheckpointInspectorProps) {
  const evidence = state.kind === "accepted" ? state.evidence : null;
  const status = evidence
    ? "Accepted"
    : state.kind === "checking"
      ? "Inspecting"
      : state.kind === "error"
        ? "Not accepted"
        : "No source";
  return (
    <section
      className={`checkpoint-inspector checkpoint-inspector--${state.kind}`}
      aria-labelledby="checkpoint-inspector-title"
      aria-live="polite"
    >
      <div className="checkpoint-inspector__header">
        <div>
          <span className="config-editor__eyebrow">Continuation source</span>
          <h3 id="checkpoint-inspector-title">Checkpoint evidence</h3>
        </div>
        <span className="checkpoint-inspector__status">
          <span aria-hidden="true" />
          {status}
        </span>
      </div>

      {evidence ? (
        <>
          <dl className="checkpoint-inspector__ledger">
            <div className="checkpoint-inspector__path-evidence">
              <dt>Checkpoint</dt>
              <dd title={evidence.path}>{checkpointBasename(evidence.path)}</dd>
            </div>
            <div>
              <dt>Source branch</dt>
              <dd title={evidence.runId}>{evidence.runId}</dd>
            </div>
            <div>
              <dt>Parent branch</dt>
              <dd title={evidence.parentRunId ?? undefined}>{evidence.parentRunId || "Root run"}</dd>
            </div>
            <div>
              <dt>Completed step</dt>
              <dd>{evidence.completedSteps.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Simulation time</dt>
              <dd>{formatCheckpointTime(evidence.simulationTimeS)}</dd>
            </div>
            <div>
              <dt>Integrity</dt>
              <dd>{formatBytes(evidence.checkpointBytes)} · {evidence.checkpointFnv1a64}</dd>
            </div>
            <div>
              <dt>Restart boundary</dt>
              <dd>Completed timestep · format v{evidence.schemaVersion}</dd>
            </div>
            <div>
              <dt>Current config</dt>
              <dd>Exact match required before state restore</dd>
            </div>
          </dl>
          <p className="checkpoint-inspector__assurance">
            Studio reproduced the sidecar digest and accepted the portable format. Run creates
            a new branch; the source files are never modified.
          </p>
        </>
      ) : state.kind === "checking" ? (
        <p className="checkpoint-inspector__message" role="status">
          Reading the checkpoint and matching sidecar… The current continuation source is
          unchanged until all evidence passes.
        </p>
      ) : state.kind === "error" ? (
        <div className="checkpoint-inspector__failure" role="alert">
          <strong>Checkpoint was not accepted</strong>
          <span>
            {state.message} Choose the original checkpoint and its matching <code>.nimbus.json</code>
            sidecar, or continue without a checkpoint.
          </span>
          {configuredPath && (
            <span>Your loaded configuration still names this path; launch remains blocked.</span>
          )}
        </div>
      ) : (
        <p className="checkpoint-inspector__message">
          Start a new run, or inspect a local Nimbus checkpoint to continue from a completed
          timestep. CM1 restart files are not opened here.
        </p>
      )}

      <div className="checkpoint-inspector__actions">
        <button
          type="button"
          className="btn btn--secondary btn--small"
          onClick={onChoose}
          disabled={!desktop || state.kind === "checking" || (remote && !configuredPath)}
          title={!desktop
            ? "Checkpoint inspection requires Nimbus Studio desktop"
            : state.kind === "checking"
              ? "Checkpoint inspection is already in progress"
              : remote && !configuredPath
                ? "Enter a remote checkpoint path in the configuration first"
              : "Inspect a Nimbus checkpoint and its integrity sidecar"}
        >
          <Icon name="folder" size={13} />
          {remote ? "Inspect remote checkpoint" : evidence ? "Inspect another…" : "Inspect checkpoint…"}
        </button>
        {configuredPath && (
          <button
            type="button"
            className="checkpoint-inspector__clear"
            onClick={onClear}
          >
            Continue without checkpoint
          </button>
        )}
      </div>
      <p className="checkpoint-inspector__retention">
        Retention is manual in this release. Studio never deletes checkpoint or sidecar files.
      </p>
    </section>
  );
}

function checkpointBasename(path: string): string {
  return path.split(/[/\\]/).pop() || path;
}

function formatCheckpointTime(seconds: number): string {
  return `${seconds.toLocaleString(undefined, { maximumFractionDigits: 3 })} s`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`;
}

interface ModuleDisableControlProps {
  section: SectionSchema;
  config: ConfigObject;
  onChange: (next: ConfigObject) => void;
}

/** Per-module on/off toggle; disabling requires typing "disable" to prevent accidents. */
function ModuleDisableControl({
  section,
  config,
  onChange,
}: ModuleDisableControlProps) {
  const spec = section.disable!;
  const [confirming, setConfirming] = useState(false);
  const [typed, setTyped] = useState("");

  const isOff = readPath(config, spec.path) === spec.offValue;
  const setOff = () => onChange(writePath(config, spec.path, spec.offValue));
  const setOn = () => onChange(writePath(config, spec.path, spec.onValue));

  if (isOff) {
    return (
      <div className="module-toggle module-toggle--off">
        <span className="module-toggle__badge module-toggle__badge--off">
          Disabled
        </span>
        <span className="module-toggle__note">
          {capitalizeFirst(spec.noun)} is off — the simulation still runs
          without it.
        </span>
        <button
          type="button"
          className="btn btn--secondary btn--small"
          onClick={setOn}
        >
          Enable
        </button>
      </div>
    );
  }

  if (!confirming) {
    return (
      <div className="module-toggle">
        <span className="module-toggle__badge module-toggle__badge--on">
          Enabled
        </span>
        <button
          type="button"
          className="btn btn--secondary btn--small"
          onClick={() => {
            setTyped("");
            setConfirming(true);
          }}
        >
          Disable {spec.noun}…
        </button>
      </div>
    );
  }

  const canConfirm = typed.trim().toLowerCase() === "disable";
  return (
    <div className="module-toggle module-toggle--confirming">
      <span className="module-toggle__note">
        Type <code>disable</code> to turn off {spec.noun}. The simulation will
        still run.
      </span>
      <input
        className="module-toggle__input"
        type="text"
        value={typed}
        placeholder="disable"
        autoFocus
        spellCheck={false}
        onChange={(e) => setTyped(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && canConfirm) {
            setOff();
            setConfirming(false);
          } else if (e.key === "Escape") {
            setConfirming(false);
          }
        }}
      />
      <button
        type="button"
        className="btn btn--secondary btn--small"
        disabled={!canConfirm}
        title={canConfirm ? "Disable this module" : "Type disable to confirm"}
        onClick={() => {
          setOff();
          setConfirming(false);
        }}
      >
        Confirm
      </button>
      <button
        type="button"
        className="btn btn--secondary btn--small"
        onClick={() => setConfirming(false)}
      >
        Cancel
      </button>
    </div>
  );
}

function capitalizeFirst(s: string): string {
  return s.length ? s[0].toUpperCase() + s.slice(1) : s;
}
