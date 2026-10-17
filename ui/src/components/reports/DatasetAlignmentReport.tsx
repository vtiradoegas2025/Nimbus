import type { DatasetAlignmentReport as AlignmentReport } from "@/lib/dataset/datasetAlignment";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { Icon } from "@/components/shell/Icon";

function verdictLabel(verdict: AlignmentReport["verdict"]): string {
  if (verdict === "exact") return "Exact alignment";
  if (verdict === "partial") return "Matched subset";
  return "Difference blocked";
}

export function DatasetAlignmentReport({
  left,
  right,
  report,
  onClose,
  comparisonViewActive,
  onOpenSynchronizedSlices,
  onExportWorkspace,
  onForgetSavedPair,
}: {
  left: LocalDatasetSource;
  right: LocalDatasetSource;
  report: AlignmentReport;
  onClose: () => void;
  comparisonViewActive: boolean;
  onOpenSynchronizedSlices?: () => void;
  onExportWorkspace?: () => void;
  onForgetSavedPair?: () => void;
}) {
  return (
    <section className="dataset-alignment" aria-label="Dataset alignment report">
      <header className="dataset-alignment__header">
        <div className="dataset-alignment__identity">
          <span>A</span><strong title={left.locator.path}>{left.displayName}</strong>
          <span className="dataset-alignment__arrow" aria-hidden="true">↔</span>
          <span>B</span><strong title={right.locator.path}>{right.displayName}</strong>
        </div>
        <span className={`dataset-alignment__verdict dataset-alignment__verdict--${report.verdict}`}>
          {verdictLabel(report.verdict)}
        </span>
        <button type="button" className="dataset-alignment__close" onClick={onClose} aria-label="Close comparison">
          ×
        </button>
      </header>

      <div className="dataset-alignment__ledger">
        <div>
          <span>Physical time</span>
          <strong>{report.time.matches.length} matched</strong>
          <p>{report.time.detail}</p>
        </div>
        <div>
          <span>Scalar grid</span>
          <strong>{report.grid.status === "exact" ? report.grid.shape?.join(" × ") : "Blocked"}</strong>
          <p>{report.grid.detail}</p>
        </div>
        <div>
          <span>Canonical fields</span>
          <strong>{report.differenceEligibleFields.length} difference-ready</strong>
          <p>{report.commonFields.length} common · {report.leftOnlyFields.length} A-only · {report.rightOnlyFields.length} B-only</p>
        </div>
      </div>

      <details className="dataset-alignment__details" open={report.verdict === "blocked"}>
        <summary>Alignment evidence</summary>
        <div className="dataset-alignment__details-body">
          {report.differenceEligibleFields.length > 0 && (
            <div className="dataset-alignment__field-list">
              <span>Eligible A−B fields</span>
              <p>{report.differenceEligibleFields.join(" · ")}</p>
            </div>
          )}
          {report.unitMismatches.length > 0 && (
            <div className="dataset-alignment__mismatches">
              <span>Unit blockers</span>
              {report.unitMismatches.map((mismatch) => (
                <p key={mismatch.field}>
                  <strong>{mismatch.field}</strong>
                  <span>{mismatch.leftUnit ?? "A unit missing"} ↔ {mismatch.rightUnit ?? "B unit missing"}</span>
                  <span>{mismatch.reason}</span>
                </p>
              ))}
            </div>
          )}
          {report.blockers.map((blocker) => (
            <div className="dataset-alignment__message dataset-alignment__message--blocked" key={blocker}>
              <Icon name="warning" size={12} /><span>{blocker}</span>
            </div>
          ))}
          {report.warnings.map((warning) => (
            <div className="dataset-alignment__message" key={warning}>
              <Icon name="info" size={12} /><span>{warning}</span>
            </div>
          ))}
        </div>
      </details>

      <footer className="dataset-alignment__footer">
        <span>
          {report.differenceAllowed
            ? "Matched frames and listed fields can open as synchronized A, B, and A−B plan slices. Studio performs no interpolation or unit conversion."
            : "A−B remains disabled. Resolve the blockers above; Studio will not interpolate, convert units, or pair frames silently."}
        </span>
        {report.differenceAllowed && onOpenSynchronizedSlices && (
          <button type="button" onClick={onOpenSynchronizedSlices} aria-pressed={comparisonViewActive}>
            {comparisonViewActive ? "A/B/difference slices open" : "Open A/B/difference slices"}
          </button>
        )}
        {onExportWorkspace && <button type="button" onClick={onExportWorkspace}>Export workspace</button>}
        {onForgetSavedPair && <button type="button" className="dataset-alignment__forget" onClick={onForgetSavedPair}>Forget saved pair</button>}
      </footer>
    </section>
  );
}
