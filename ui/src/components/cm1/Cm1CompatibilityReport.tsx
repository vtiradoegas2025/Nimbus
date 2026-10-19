import { useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  buildCm1CompatibilityReport,
  type Cm1CompatibilityStatus,
} from "@/lib/cm1/cm1Compatibility";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { buildCm1PartialDraft } from "@/lib/cm1/cm1PartialDraft";
import { uniqueCm1NamelistValue } from "@/lib/cm1/cm1NamelistSummary";
import { buildCm1SoundingProjectDraft } from "@/lib/cm1/cm1SoundingProject";
import { buildCm1SoundingProjection } from "@/lib/cm1/cm1SoundingProjection";
import {
  createCm1GuidedHandoff,
  saveCm1GuidedHandoff,
} from "@/lib/cm1/cm1GuidedHandoff";
import { preflightConfigWithBinary } from "@/lib/config/configPreflight";
import { validateCm1SoundingProjectWithBinary } from "@/lib/config/exactInitializationValidation";
import { loadStudioSettings } from "@/lib/studio/studioSettings";
import {
  getSchemeCatalog,
  isTauriRuntime,
  pickSoundingSavePath,
  pickYamlSavePath,
  saveCm1SoundingProjection,
  writeConfig,
} from "@/lib/platform/tauri";
import { Icon } from "@/components/shell/Icon";
import { Cm1DecisionWorkbook } from "./Cm1DecisionWorkbook";

const STATUS_LABELS: Record<Cm1CompatibilityStatus, string> = {
  ready: "Ready",
  needs_decision: "Needs decision",
  different: "Different",
  unsupported: "Unsupported / unmapped",
};

type Filter = "all" | Cm1CompatibilityStatus;

type SoundingProjectState =
  | { kind: "idle" }
  | { kind: "saving"; message: string }
  | { kind: "saved"; message: string }
  | { kind: "blocked"; message: string }
  | { kind: "failed"; message: string };

function finiteNamelistNumber(raw: string | null): number | null {
  if (raw == null) return null;
  const value = Number(raw.trim().replace(/[dD]/g, "e"));
  return Number.isFinite(value) ? value : null;
}

function actionableErrorMessage(error: unknown): string {
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

export function Cm1CompatibilityReport({ dataset }: { dataset: LocalDatasetSource }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const [soundingCopyState, setSoundingCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const [soundingReviewed, setSoundingReviewed] = useState(false);
  const [nativeSoundingReviewed, setNativeSoundingReviewed] = useState(false);
  const [soundingProjectState, setSoundingProjectState] = useState<SoundingProjectState>({ kind: "idle" });
  const [nativeProjectState, setNativeProjectState] = useState<SoundingProjectState>({ kind: "idle" });
  const sourceGeneration = useRef(0);
  useLayoutEffect(() => {
    sourceGeneration.current += 1;
    setSoundingReviewed(false);
    setNativeSoundingReviewed(false);
    setCopyState("idle");
    setSoundingCopyState("idle");
    setSoundingProjectState({ kind: "idle" });
    setNativeProjectState({ kind: "idle" });
    return () => { sourceGeneration.current += 1; };
  }, [dataset]);
  const report = useMemo(
    () => dataset.namelistReport
      ? buildCm1CompatibilityReport({
          namelist: dataset.namelistReport,
          inventory: dataset.caseInventory,
          sounding: dataset.soundingReport,
          namelistInspectionError: dataset.namelistInspectionError,
          soundingInspectionError: dataset.soundingInspectionError,
        })
      : null,
    [dataset],
  );
  if (!report) return null;
  const draft = buildCm1PartialDraft(report);
  const soundingProjection = dataset.soundingReport
    ? buildCm1SoundingProjection(dataset.soundingReport, dataset.namelistReport)
    : null;
  const sourceIsnd = finiteNamelistNumber(
    dataset.namelistReport
      ? uniqueCm1NamelistValue(dataset.namelistReport, "isnd")
      : null,
  );

  const copyDraft = async () => {
    const generation = sourceGeneration.current;
    try {
      if (!navigator.clipboard) throw new Error("Clipboard API unavailable");
      await navigator.clipboard.writeText(draft.yaml);
      if (generation === sourceGeneration.current) setCopyState("copied");
    } catch {
      if (generation === sourceGeneration.current) setCopyState("failed");
    }
  };

  const copySounding = async () => {
    const generation = sourceGeneration.current;
    if (!soundingProjection?.copyEligible) return;
    try {
      if (!navigator.clipboard) throw new Error("Clipboard API unavailable");
      await navigator.clipboard.writeText(soundingProjection.text);
      if (generation === sourceGeneration.current) setSoundingCopyState("copied");
    } catch {
      if (generation === sourceGeneration.current) setSoundingCopyState("failed");
    }
  };

  const stageResolvedSoundingProject = async ({
    soundingPath,
    sourcePath,
    levelCount,
    sourceMode,
    defaultConfigName,
    setState,
  }: {
    soundingPath: string;
    sourcePath: string;
    levelCount: number;
    sourceMode: "reviewed_projection" | "native_cm1";
    defaultConfigName: string;
    setState: (state: SoundingProjectState) => void;
  }) => {
    const generation = sourceGeneration.current;
    const current = () => generation === sourceGeneration.current;
    const sourceLabel = sourceMode === "native_cm1" ? "Native CM1 sounding" : "Reviewed sounding";
    try {
      setState({ kind: "saving", message: "Checking a Nimbus baseline against the configured simulator…" });
      const binaryPath = loadStudioSettings().binaryPathOverride.trim() || undefined;
      const catalog = await getSchemeCatalog(binaryPath);
      if (!current()) return;
      const project = buildCm1SoundingProjectDraft(soundingPath, catalog, { sourceMode });
      const preflight = await preflightConfigWithBinary(project.preflightRequest, binaryPath);
      if (!current()) return;
      if (!preflight.compatible) {
        const reason = preflight.hardErrors[0]?.message ?? "the exact simulator rejected the baseline selections";
        setState({ kind: "blocked", message: `${sourceLabel} was not staged: ${reason}.` });
        return;
      }

      setState({
        kind: "saving",
        message: "Catalog preflight passed. Building the sounding through the configured simulator…",
      });
      let initialization;
      try {
        initialization = await validateCm1SoundingProjectWithBinary(
          project.yaml,
          preflight.executable.catalogFingerprint,
          binaryPath,
        );
        if (!current()) return;
      } catch (error) {
        if (!current()) return;
        setState({
          kind: "blocked",
          message: `${sourceLabel} was not staged because no-step initialization failed: ${actionableErrorMessage(error)}.`,
        });
        return;
      }

      const configPath = await pickYamlSavePath(defaultConfigName);
      if (!current()) return;
      if (!configPath) {
        setState({ kind: "idle" });
        return;
      }
      await writeConfig(configPath, project.yaml);
      if (!current()) return;
      const handoff = createCm1GuidedHandoff({
        cm1Source: dataset,
        soundingSourcePath: sourcePath,
        projectedLevelCount: levelCount,
        soundingSourceMode: sourceMode,
        mappingCounts: {
          ready: report.counts.ready,
          needsDecision: report.counts.needs_decision,
          different: report.counts.different,
          unsupported: report.counts.unsupported,
        },
        mappingBlockers: report.draftBlockers,
        soundingPath,
        configPath,
        yaml: project.yaml,
        catalogFingerprint: preflight.executable.catalogFingerprint,
        executablePath: preflight.executable.canonicalPath,
        initializationExecutablePath: initialization.executablePath,
      });
      saveCm1GuidedHandoff(window.localStorage, handoff);
      setState({
        kind: "saved",
        message: `Handoff staged. Open Configure: Studio will load the ${sourceMode === "native_cm1" ? "original native profile" : "reviewed projection"} as a 30 s smoke test with 10 s output. Exact executable ${preflight.executable.catalogFingerprint} passed catalog preflight and no-step initialization.`,
      });
    } catch (error) {
      if (current()) setState({ kind: "failed", message: actionableErrorMessage(error) });
    }
  };

  const saveSoundingProject = async () => {
    const generation = sourceGeneration.current;
    if (!soundingProjection?.copyEligible || !soundingReviewed) return;
    if (!isTauriRuntime()) {
      setSoundingProjectState({
        kind: "failed",
        message: "Reviewed projection and exact-binary preflight require Nimbus Studio desktop.",
      });
      return;
    }
    try {
      setSoundingProjectState({ kind: "saving", message: "Choose where to save the reviewed sounding…" });
      const soundingPath = await pickSoundingSavePath();
      if (generation !== sourceGeneration.current) return;
      if (!soundingPath) {
        setSoundingProjectState({ kind: "idle" });
        return;
      }
      const saved = await saveCm1SoundingProjection(
        soundingPath,
        soundingProjection.text,
        soundingProjection.sourcePath,
        sourceIsnd,
        true,
      );
      if (generation !== sourceGeneration.current) return;
      await stageResolvedSoundingProject({
        soundingPath: saved.canonicalPath,
        sourcePath: soundingProjection.sourcePath,
        levelCount: soundingProjection.levels.length,
        sourceMode: "reviewed_projection",
        defaultConfigName: "cm1-sounding-nimbus-baseline.yaml",
        setState: setSoundingProjectState,
      });
    } catch (error) {
      if (generation !== sourceGeneration.current) return;
      setSoundingProjectState({
        kind: "failed",
        message: actionableErrorMessage(error),
      });
    }
  };

  const stageNativeSoundingProject = async () => {
    const native = dataset.soundingReport;
    if (!native || sourceIsnd !== 7 || !nativeSoundingReviewed) return;
    if (!isTauriRuntime()) {
      setNativeProjectState({ kind: "failed", message: "Native handoff and exact-binary validation require Nimbus Studio desktop." });
      return;
    }
    await stageResolvedSoundingProject({
      soundingPath: native.path,
      sourcePath: native.path,
      levelCount: native.levels.length,
      sourceMode: "native_cm1",
      defaultConfigName: "cm1-native-sounding-nimbus-baseline.yaml",
      setState: setNativeProjectState,
    });
  };

  const assetRows = report.assetChecks.filter((item) => filter === "all" || item.status === filter);
  const assignmentRows = report.entries.filter((item) => filter === "all" || item.status === filter);
  const visibleAssignments = assignmentRows.slice(0, 500);
  const hiddenCount = assignmentRows.length - visibleAssignments.length;
  const filters: Array<{ value: Filter; label: string; count: number }> = [
    { value: "all", label: "All", count: report.entries.length + report.assetChecks.length },
    ...Object.entries(STATUS_LABELS).map(([value, label]) => ({
      value: value as Cm1CompatibilityStatus,
      label,
      count: report.counts[value as Cm1CompatibilityStatus],
    })),
  ];

  return (
    <details className="cm1-compatibility">
      <summary className="cm1-compatibility__summary">
        <span className="cm1-compatibility__chevron" aria-hidden="true">›</span>
        <strong>CM1 → Nimbus compatibility</strong>
        <span className="cm1-compatibility__summary-counts">
          {report.counts.ready} ready · {report.counts.needs_decision} decision · {report.counts.different} different · {report.counts.unsupported} unsupported
        </span>
        <span className="cm1-compatibility__blocked">Draft blocked</span>
      </summary>
      <div className="cm1-compatibility__body">
        <div className="cm1-compatibility__intro">
          <div>
            <strong>Compatibility evidence, not runnable configuration</strong>
            <span>Every parsed assignment remains visible. “Ready” means mechanically transferable; it does not claim numerical equivalence.</span>
          </div>
          <div className="cm1-compatibility__blockers" role="status">
            <Icon name="warning" size={13} />
            <span>{report.draftBlockers.join(" · ")}</span>
          </div>
        </div>

        <Cm1DecisionWorkbook dataset={dataset} report={report} />

        <details className="cm1-draft">
          <summary className="cm1-draft__summary">
            <span className="cm1-draft__chevron" aria-hidden="true">›</span>
            <strong>Mechanical YAML preview</strong>
            <span>{draft.includedEntryIds.length} included · {draft.omissions.length} omitted</span>
            <span className="cm1-draft__non-runnable">Non-runnable</span>
          </summary>
          <div className="cm1-draft__body">
            <div className="cm1-draft__source">
              <div className="cm1-draft__toolbar">
                <div>
                  <strong>Ready values only</strong>
                  <span>No defaults or scientific substitutions added</span>
                </div>
                <button type="button" onClick={() => void copyDraft()}>
                  {copyState === "copied" ? "Copied" : copyState === "failed" ? "Copy unavailable" : "Copy partial YAML"}
                </button>
              </div>
              <pre tabIndex={0}><code>{draft.yaml}</code></pre>
            </div>
            <aside className="cm1-draft__accounting" aria-label="Partial draft accounting">
              <div>
                <strong>{draft.includedEntryIds.length}</strong>
                <span>source assignments included</span>
              </div>
              <div>
                <strong>{draft.omissions.length}</strong>
                <span>source assignments omitted with reasons</span>
              </div>
              <div>
                <strong>{draft.valueChecks.filter((check) => !check.accepted).length}</strong>
                <span>current descriptor check failures</span>
              </div>
              <ul>
                {draft.blockers.slice(0, 5).map((blocker) => <li key={blocker}>{blocker}</li>)}
              </ul>
              <span className="cm1-draft__copy-status" aria-live="polite">
                {copyState === "copied"
                  ? "Partial YAML copied to the clipboard."
                  : copyState === "failed"
                    ? "Clipboard access is unavailable in this environment."
                    : ""}
              </span>
            </aside>
          </div>
        </details>

        {dataset.soundingReport && (
          <details className="cm1-draft cm1-native-sounding">
            <summary className="cm1-draft__summary">
              <span className="cm1-draft__chevron" aria-hidden="true">›</span>
              <strong>Native CM1 sounding handoff</strong>
              <span>{dataset.soundingReport.levels.length} original levels · no conversion</span>
              <span className="cm1-draft__non-runnable">
                {sourceIsnd === 7 ? "Review required" : "Source behavior blocked"}
              </span>
            </summary>
            <div className="cm1-draft__body">
              <div className="cm1-draft__source cm1-native-sounding__source">
                <div className="cm1-draft__toolbar">
                  <div>
                    <strong>Original input_sounding</strong>
                    <span>Direct θ, qv, u, and v ingestion · Nimbus EOS reconstruction</span>
                  </div>
                  <button
                    type="button"
                    disabled={
                      sourceIsnd !== 7
                      || !nativeSoundingReviewed
                      || nativeProjectState.kind === "saving"
                    }
                    title={sourceIsnd !== 7
                      ? "Direct handoff is offered only when the CM1 case explicitly selects input_sounding thermodynamics and winds with isnd=7"
                      : !nativeSoundingReviewed
                        ? "Review the native-source scope and wind ownership before staging"
                        : nativeProjectState.kind === "saving"
                          ? "The exact simulator compatibility check is already running"
                          : "Validate and stage the original native CM1 profile without conversion"}
                    onClick={() => void stageNativeSoundingProject()}
                  >
                    {nativeProjectState.kind === "saving" ? "Checking…" : "Stage native 30 s smoke test"}
                  </button>
                </div>
                <div className="cm1-native-sounding__contract">
                  <code title={dataset.soundingReport.path}>{dataset.soundingReport.path}</code>
                  <p>
                    This imports one atmospheric column only. Nimbus reads the original profile,
                    consumes its embedded u/v through hodograph Auto, and reconstructs pressure and
                    temperature using Nimbus thermodynamics. No CM1 grid, physics, perturbation, or
                    runtime control is implied.
                  </p>
                </div>
              </div>
              <aside className="cm1-draft__accounting" aria-label="Native sounding handoff accounting">
                <div><strong>{dataset.soundingReport.levels.length}</strong><span>native levels retained</span></div>
                <div><strong>{sourceIsnd ?? "?"}</strong><span>CM1 isnd source behavior</span></div>
                <p className="cm1-sounding-projection__method">
                  {sourceIsnd === 7
                    ? "CM1 isnd=7 selects input_sounding thermodynamics and winds, so direct embedded-wind ownership is coherent."
                    : sourceIsnd === 17
                      ? "CM1 isnd=17 ignores input_sounding winds; direct Auto would change wind ownership and is therefore blocked."
                      : "The case does not unambiguously select the native file for both thermodynamics and winds; direct staging is blocked."}
                </p>
                <label className="cm1-sounding-projection__review">
                  <input
                    type="checkbox"
                    checked={nativeSoundingReviewed}
                    disabled={sourceIsnd !== 7}
                    onChange={(event) => setNativeSoundingReviewed(event.target.checked)}
                  />
                  <span>I reviewed the native-source scope, Nimbus EOS reconstruction, and embedded wind ownership.</span>
                </label>
                {dataset.soundingReport.warnings.length > 0 && (
                  <ul>{dataset.soundingReport.warnings.slice(0, 6).map((warning) => <li key={warning}>{warning}</li>)}</ul>
                )}
                {nativeProjectState.kind !== "idle" && (
                  <span
                    className={`cm1-sounding-projection__save-status is-${nativeProjectState.kind}`}
                    role={nativeProjectState.kind === "failed" || nativeProjectState.kind === "blocked" ? "alert" : "status"}
                  >
                    {nativeProjectState.message}
                  </span>
                )}
              </aside>
            </div>
          </details>
        )}

        {soundingProjection && (
          <details className="cm1-draft cm1-sounding-projection">
            <summary className="cm1-draft__summary">
              <span className="cm1-draft__chevron" aria-hidden="true">›</span>
              <strong>Optional compatibility projection</strong>
              <span>{soundingProjection.levels.length} projected levels</span>
              <span className="cm1-draft__non-runnable">
                {soundingProjection.copyEligible ? "Review required" : "Projection blocked"}
              </span>
            </summary>
            <div className="cm1-draft__body">
              <div className="cm1-draft__source">
                <div className="cm1-draft__toolbar">
                  <div>
                    <strong>Projected Nimbus sounding text</strong>
                    <span>Hydrostatic projection · not an exact CM1 model state</span>
                  </div>
                  <div className="cm1-sounding-projection__actions">
                    <button
                      type="button"
                      disabled={!soundingProjection.copyEligible}
                      title={soundingProjection.copyEligible
                        ? "Copy the reviewed projection text without changing the CM1 source"
                        : "Resolve the listed sounding projection blockers before copying"}
                      onClick={() => void copySounding()}
                    >
                      {soundingCopyState === "copied"
                        ? "Copied"
                        : soundingCopyState === "failed"
                          ? "Copy unavailable"
                          : "Copy sounding text"}
                    </button>
                    <button
                      type="button"
                      disabled={
                        !soundingProjection.copyEligible
                        || !soundingReviewed
                        || soundingProjectState.kind === "saving"
                      }
                      title={!soundingProjection.copyEligible
                        ? "Resolve the listed sounding projection blockers before staging"
                        : !soundingReviewed
                          ? "Review the projected levels, caveats, and wind behavior before staging"
                          : soundingProjectState.kind === "saving"
                            ? "The exact simulator compatibility check is already running"
                            : "Create a source-bound 30 second Nimbus smoke-test draft"}
                      onClick={() => void saveSoundingProject()}
                    >
                      {soundingProjectState.kind === "saving" ? "Checking…" : "Stage 30 s Nimbus smoke test"}
                    </button>
                  </div>
                </div>
                <pre tabIndex={0}><code>{soundingProjection.text}</code></pre>
              </div>
              <aside className="cm1-draft__accounting" aria-label="Sounding projection accounting">
                <div>
                  <strong>{soundingProjection.levels.length}</strong>
                  <span>CM1 source levels projected</span>
                </div>
                <div>
                  <strong>{soundingProjection.blockers.length}</strong>
                  <span>blocking source or behavior conflicts</span>
                </div>
                <p className="cm1-sounding-projection__method">{soundingProjection.method}</p>
                <label className="cm1-sounding-projection__review">
                  <input
                    type="checkbox"
                    checked={soundingReviewed}
                    disabled={!soundingProjection.copyEligible}
                    title={soundingProjection.copyEligible
                      ? "Record that the projected profile and caveats were reviewed"
                      : "Review is unavailable until the sounding projection blockers are resolved"}
                    onChange={(event) => setSoundingReviewed(event.target.checked)}
                  />
                  <span>I reviewed the projected levels, caveats, and wind behavior.</span>
                </label>
                <ul>
                  {[...soundingProjection.blockers, ...soundingProjection.caveats].slice(0, 6).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                <span className="cm1-draft__copy-status" aria-live="polite">
                  {soundingCopyState === "copied"
                    ? "Sounding projection copied to the clipboard."
                    : soundingCopyState === "failed"
                      ? "Clipboard access is unavailable in this environment."
                      : ""}
                </span>
                {soundingProjectState.kind !== "idle" && (
                  <span
                    className={`cm1-sounding-projection__save-status is-${soundingProjectState.kind}`}
                    role={soundingProjectState.kind === "failed" || soundingProjectState.kind === "blocked" ? "alert" : "status"}
                  >
                    {soundingProjectState.message}
                  </span>
                )}
              </aside>
            </div>
          </details>
        )}

        <div className="cm1-compatibility__filters" aria-label="Filter compatibility items">
          {filters.map((item) => (
            <button
              type="button"
              key={item.value}
              aria-pressed={filter === item.value}
              onClick={() => setFilter(item.value)}
            >
              {item.label} <span>{item.count}</span>
            </button>
          ))}
        </div>

        <div className="cm1-compatibility__table" role="table" aria-label="CM1 compatibility disposition report">
          <div className="cm1-compatibility__table-head" role="row">
            <span role="columnheader">Source</span>
            <span role="columnheader">Disposition</span>
            <span role="columnheader">Nimbus target and evidence</span>
          </div>
          {assetRows.map((item) => (
            <div className="cm1-compatibility__row" role="row" key={`asset:${item.id}`}>
              <div role="cell"><strong>{item.label}</strong><span>case asset</span></div>
              <div role="cell"><Status status={item.status} /></div>
              <div role="cell"><strong>Case workflow</strong><span>{item.detail}</span></div>
            </div>
          ))}
          {visibleAssignments.map((item) => (
            <div className="cm1-compatibility__row" role="row" key={item.id}>
              <div role="cell">
                <strong>{item.group}.{item.key}</strong>
                <code title={item.rawValue}>{item.rawValue}</code>
              </div>
              <div role="cell"><Status status={item.status} /></div>
              <div role="cell">
                <strong>{item.target ?? "No qualified Nimbus target"}</strong>
                <span>{item.reason}</span>
              </div>
            </div>
          ))}
          {assetRows.length === 0 && visibleAssignments.length === 0 && (
            <div className="cm1-compatibility__empty">No items have this disposition.</div>
          )}
        </div>
        {hiddenCount > 0 && (
          <div className="cm1-compatibility__limit">
            Showing the first 500 matching assignments; {hiddenCount} remain in the report.
          </div>
        )}
      </div>
    </details>
  );
}

function Status({ status }: { status: Cm1CompatibilityStatus }) {
  return (
    <span className="cm1-compatibility__status" data-status={status}>
      {STATUS_LABELS[status]}
    </span>
  );
}
