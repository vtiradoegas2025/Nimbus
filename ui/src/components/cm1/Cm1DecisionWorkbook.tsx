import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { Cm1CompatibilityReport } from "@/lib/cm1/cm1Compatibility";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import {
  CM1_DECISION_NOTE_MAX_CHARS,
  CM1_DECISION_WORKBOOK_MAX_JSON_BYTES,
  CM1_MANUAL_VALUE_MAX_CHARS,
  clearCm1MappingDecision,
  cm1DecisionOptions,
  cm1DecisionProgress,
  createCm1DecisionWorkbook,
  decideCm1MappingItem,
  importCm1DecisionWorkbook,
  loadCm1DecisionWorkbook,
  saveCm1DecisionWorkbook,
  serializeCm1DecisionWorkbook,
  type Cm1DecisionKind,
  type Cm1DecisionWorkbook as Workbook,
  type Cm1DecisionWorkbookItem,
} from "@/lib/cm1/cm1DecisionWorkbook";

const DECISION_LABELS: Record<Cm1DecisionKind, string> = {
  keep_nimbus_default: "Keep documented Nimbus default",
  manual_resolution: "Resolve to the supported Nimbus target",
  acknowledge_difference: "Acknowledge different formulation",
  omit_unsupported: "Omit unsupported CM1 control",
};

const DISPOSITION_LABELS: Record<Cm1DecisionWorkbookItem["status"], string> = {
  needs_decision: "Needs decision",
  different: "Different",
  unsupported: "Unsupported / unmapped",
};

interface WorkbookBaseline {
  workbook: Workbook | null;
  error: string | null;
}

function buildBaseline(dataset: LocalDatasetSource, report: Cm1CompatibilityReport): WorkbookBaseline {
  try {
    return { workbook: createCm1DecisionWorkbook(dataset, report), error: null };
  } catch (error) {
    return { workbook: null, error: error instanceof Error ? error.message : String(error) };
  }
}

function downloadWorkbook(workbook: Workbook): void {
  const blob = new Blob([serializeCm1DecisionWorkbook(workbook)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `cm1-review-${workbook.source.mappingFingerprint.replaceAll(":", "-")}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function Cm1DecisionWorkbook({
  dataset,
  report,
}: {
  dataset: LocalDatasetSource;
  report: Cm1CompatibilityReport;
}) {
  const baseline = useMemo(() => buildBaseline(dataset, report), [dataset, report]);
  const [workbook, setWorkbook] = useState<Workbook | null>(() => baseline.workbook
    ? loadCm1DecisionWorkbook(window.localStorage, baseline.workbook) ?? baseline.workbook
    : null);
  const [visibleCount, setVisibleCount] = useState(100);
  const [message, setMessage] = useState<{ kind: "status" | "error"; text: string } | null>(null);
  const importRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setVisibleCount(100);
    setMessage(null);
    setWorkbook(baseline.workbook
      ? loadCm1DecisionWorkbook(window.localStorage, baseline.workbook) ?? baseline.workbook
      : null);
  }, [baseline]);

  const persist = useCallback((next: Workbook) => {
    saveCm1DecisionWorkbook(window.localStorage, next);
    setWorkbook(next);
    setMessage({ kind: "status", text: "Review decision saved locally." });
  }, []);

  const importFile = useCallback(async (file: File) => {
    if (!baseline.workbook) return;
    setMessage(null);
    try {
      if (file.size > CM1_DECISION_WORKBOOK_MAX_JSON_BYTES) {
        throw new Error("This workbook exceeds the portable JSON size limit.");
      }
      const imported = importCm1DecisionWorkbook(await file.text(), baseline.workbook);
      saveCm1DecisionWorkbook(window.localStorage, imported);
      setWorkbook(imported);
      setMessage({ kind: "status", text: "Source-matched review workbook imported." });
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    }
  }, [baseline.workbook]);

  if (!workbook) {
    return (
      <div className="cm1-workbook__blocked" role="alert">
        Decision workbook unavailable. {baseline.error ?? "The source-bound report could not be verified."}
      </div>
    );
  }

  const progress = cm1DecisionProgress(workbook);
  const visibleItems = workbook.items.slice(0, visibleCount);
  return (
    <details className="cm1-workbook">
      <summary className="cm1-workbook__summary">
        <span className="cm1-draft__chevron" aria-hidden="true">›</span>
        <strong>Mapping decision workbook</strong>
        <span>{progress.total === 0 ? "No decisions required" : `${progress.complete} reviewed · ${progress.remaining} remaining`}</span>
        <span className="cm1-workbook__scope">Review record only</span>
      </summary>
      <div className="cm1-workbook__body">
        <header className="cm1-workbook__header">
          <div>
            <strong>{workbook.source.displayName}</strong>
            <span>
              Every non-Ready assignment mapping remains source-bound. Decisions do not enter YAML, make a draft runnable, or establish scientific equivalence.
            </span>
          </div>
          <div className="cm1-workbook__progress" aria-label={`${progress.complete} of ${progress.total} mapping decisions complete`}>
            <span>{progress.total === 0 ? "Complete · no decisions" : `${progress.complete} / ${progress.total}`}</span>
            <progress max={Math.max(1, progress.total)} value={progress.total === 0 ? 1 : progress.complete} />
          </div>
          <div className="cm1-workbook__publication">
            <button type="button" tabIndex={0} onClick={() => downloadWorkbook(workbook)}>Export review JSON</button>
            <button type="button" tabIndex={0} onClick={() => importRef.current?.click()}>Import matching review…</button>
            <input
              ref={importRef}
              hidden
              type="file"
              accept="application/json,.json"
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                if (file) void importFile(file);
                event.currentTarget.value = "";
              }}
            />
          </div>
        </header>

        {message && (
          <div className={`cm1-workbook__message is-${message.kind}`} role={message.kind === "error" ? "alert" : "status"}>
            {message.text}
          </div>
        )}

        {workbook.items.length === 0 ? (
          <div className="cm1-workbook__empty">All assignment mappings are Ready; no review decisions are required.</div>
        ) : (
          <div className="cm1-workbook__items" aria-label="Non-Ready CM1 mapping decisions">
            {visibleItems.map((item) => (
              <DecisionRow
                key={`${item.id}:${item.decision?.decidedAt ?? "open"}`}
                item={item}
                onSave={(input) => {
                  try { persist(decideCm1MappingItem(workbook, item.id, input)); }
                  catch (error) { setMessage({ kind: "error", text: error instanceof Error ? error.message : String(error) }); }
                }}
                onClear={() => persist(clearCm1MappingDecision(workbook, item.id))}
              />
            ))}
          </div>
        )}
        {visibleCount < workbook.items.length && (
          <button
            type="button"
            className="cm1-workbook__show-more"
            onClick={() => setVisibleCount((count) => Math.min(workbook.items.length, count + 100))}
          >
            Show next {Math.min(100, workbook.items.length - visibleCount)} mappings
          </button>
        )}
      </div>
    </details>
  );
}

function DecisionRow({
  item,
  onSave,
  onClear,
}: {
  item: Cm1DecisionWorkbookItem;
  onSave: (input: { kind: Cm1DecisionKind; manualValue?: string; note?: string }) => void;
  onClear: () => void;
}) {
  const options = cm1DecisionOptions(item);
  const [kind, setKind] = useState<Cm1DecisionKind | "">(item.decision?.kind ?? "");
  const [manualValue, setManualValue] = useState(item.decision?.manualValue ?? "");
  const [note, setNote] = useState(item.decision?.note ?? "");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!kind) return;
    onSave({ kind, manualValue, note });
  };
  return (
    <form className={`cm1-workbook__item ${item.decision ? "is-complete" : ""}`} onSubmit={submit}>
      <div className="cm1-workbook__identity">
        <strong>{item.group}.{item.key}</strong>
        <code title={item.rawValue}>{item.rawValue}</code>
        <span className="cm1-compatibility__status" data-status={item.status}>{DISPOSITION_LABELS[item.status]}</span>
      </div>
      <div className="cm1-workbook__evidence">
        <strong>{item.target ?? "No qualified Nimbus target"}</strong>
        <span>{item.reason}</span>
      </div>
      <div className="cm1-workbook__decision">
        <label>
          <span>Review decision</span>
          <select value={kind} required onChange={(event) => setKind(event.target.value as Cm1DecisionKind | "")}>
            <option value="">Record a decision…</option>
            {options.map((option) => <option key={option} value={option}>{DECISION_LABELS[option]}</option>)}
          </select>
        </label>
        {kind === "manual_resolution" && (
          <label>
            <span>Candidate value for {item.target}</span>
            <input
              value={manualValue}
              maxLength={CM1_MANUAL_VALUE_MAX_CHARS}
              required
              spellCheck={false}
              onChange={(event) => setManualValue(event.target.value)}
            />
          </label>
        )}
        {kind === "keep_nimbus_default" && (
          <div className="cm1-workbook__default-evidence">
            <span>Exact default recorded</span>
            <code>
              {item.targetDescriptor
                ? `${item.target} = ${JSON.stringify(item.targetDescriptor.defaultValue)}`
                : "No single Nimbus target or descriptor; no CM1 value transfers."}
            </code>
            {item.targetDescriptor && <small>{item.targetDescriptor.fingerprint}</small>}
          </div>
        )}
        <label>
          <span>Review note <small>optional · {note.length}/{CM1_DECISION_NOTE_MAX_CHARS}</small></span>
          <textarea
            value={note}
            maxLength={CM1_DECISION_NOTE_MAX_CHARS}
            rows={2}
            onChange={(event) => setNote(event.target.value)}
          />
        </label>
        <div className="cm1-workbook__actions">
          <button type="submit" disabled={!kind}>{item.decision ? "Update decision" : "Record decision"}</button>
          {item.decision && <button type="button" onClick={onClear}>Clear decision</button>}
        </div>
      </div>
    </form>
  );
}
