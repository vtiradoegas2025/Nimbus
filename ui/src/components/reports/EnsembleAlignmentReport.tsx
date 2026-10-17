import type { EnsembleAlignmentReport as EnsembleReport } from "@/lib/comparison/ensembleAlignment";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { Icon } from "@/components/shell/Icon";

export function EnsembleAlignmentReport({
  sources,
  report,
  loading,
  viewActive,
  onOpen,
  onRemove,
  onClose,
}: {
  sources: readonly LocalDatasetSource[];
  report: EnsembleReport;
  loading: boolean;
  viewActive: boolean;
  onOpen?: () => void;
  onRemove: (identity: string) => void;
  onClose: () => void;
}) {
  const verdict = report.status === "exact" ? "Exact group" : report.status === "partial" ? "Matched subset" : "Aggregation blocked";
  return (
    <section className="ensemble-alignment" aria-label="Ensemble alignment report">
      <header>
        <div><span>Ensemble group</span><strong>{sources.length} atmospheric histories</strong><p>Member 01 is the reference for canonical field, grid, and physical-time qualification.</p></div>
        <span className={`ensemble-alignment__verdict ensemble-alignment__verdict--${report.status}`}>{verdict}</span>
        <button type="button" aria-label="Close ensemble" onClick={onClose}>×</button>
      </header>
      <div className="ensemble-alignment__members" role="list" aria-label="Ensemble members">
        {sources.map((source, index) => {
          const identity = report.memberIdentities[index];
          const member = report.members[index];
          return (
            <div role="listitem" key={identity}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <div><strong>{source.displayName}</strong><small title={source.locator.path}>{source.kind === "cm1_netcdf" ? "netCDF data" : "Nimbus history"}{index === 0 ? " · reference" : ""}</small></div>
              <em>{index === 0 ? "Reference" : member?.report?.differenceAllowed ? `${member.report.time.matches.length} matched times` : "Blocked"}</em>
              {index > 0 && <button type="button" onClick={() => onRemove(identity)}>Remove</button>}
            </div>
          );
        })}
      </div>
      <div className="ensemble-alignment__ledger">
        <div><span>Common time</span><strong>{report.timeMatches.length}</strong><p>physical frames across every member</p></div>
        <div><span>Scalar grid</span><strong>{report.gridShape?.join(" × ") ?? "Blocked"}</strong><p>exact scalar-center coordinates</p></div>
        <div><span>Canonical fields</span><strong>{report.eligibleFields.length}</strong><p>stable units across every member</p></div>
      </div>
      {(report.blockers.length > 0 || report.warnings.length > 0) && (
        <details open={report.status === "blocked"}>
          <summary>Qualification evidence</summary>
          <div>
            {report.blockers.map((blocker) => <p className="ensemble-alignment__blocked" key={blocker}><Icon name="warning" size={11} />{blocker}</p>)}
            {report.warnings.map((warning) => <p key={warning}><Icon name="info" size={11} />{warning}</p>)}
          </div>
        </details>
      )}
      <footer>
        <span>{loading ? "Indexing an additional member…" : report.aggregateAllowed ? "Every aggregate uses only the listed all-member field/time/grid intersection." : "Mean, spread, probability, and spaghetti remain disabled until every blocker is resolved."}</span>
        {onOpen && <button type="button" onClick={onOpen} aria-pressed={viewActive}>{viewActive ? "Ensemble board open" : "Open ensemble board"}</button>}
      </footer>
    </section>
  );
}
