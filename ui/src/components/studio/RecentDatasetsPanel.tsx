import type { RecentDatasetEntry } from "@/lib/dataset/recentDatasets";
import "./RecentDatasetsPanel.css";

export interface RecentDatasetsPanelProps {
  entries: readonly RecentDatasetEntry[];
  onReopen: (entry: RecentDatasetEntry) => void | Promise<void>;
  onPinChange?: (entry: RecentDatasetEntry, pinned: boolean) => void;
  onForget?: (entry: RecentDatasetEntry) => void;
  onClose?: () => void;
  reopeningIdentity?: string | null;
}

function sourceLabel(entry: RecentDatasetEntry): string {
  if (entry.selection.kind === "nimbus_folder") return "Nimbus run";
  if (entry.selection.kind === "cm1_case_folder") return "NetCDF folder";
  return entry.selection.paths.length === 1 ? "NetCDF file" : `NetCDF series · ${entry.selection.paths.length} files`;
}

function referenceLabel(entry: RecentDatasetEntry): string {
  if (entry.selection.kind === "cm1_case_folder") return entry.selection.paths[0];
  if (entry.selection.paths.length === 1) return entry.selection.paths[0];
  const first = entry.selection.paths[0];
  return `${first} + ${entry.selection.paths.length - 1} more`;
}

function openedLabel(timestamp: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

export function RecentDatasetsPanel({
  entries,
  onReopen,
  onPinChange,
  onForget,
  onClose,
  reopeningIdentity = null,
}: RecentDatasetsPanelProps) {
  return <section className="recent-datasets" aria-labelledby="recent-datasets-title">
    <header className="recent-datasets__header">
      <div>
        <span className="recent-datasets__eyebrow">LOCAL OBSERVATION LOG · REFERENCES ONLY</span>
        <h2 id="recent-datasets-title">Recent datasets</h2>
        <p>Reopen a netCDF dataset or Nimbus run without rebuilding the selection.</p>
      </div>
      {onClose && <button className="recent-datasets__close" type="button" onClick={onClose} aria-label="Close recent datasets">Close</button>}
    </header>

    <div className="recent-datasets__notice" role="note">
      <strong>No field data is stored here.</strong>
      <span>Studio keeps local file and folder references only. Moved files, changed permissions, or disconnected volumes must be resolved when reopened.</span>
    </div>

    {entries.length === 0
      ? <div className="recent-datasets__empty">
          <span aria-hidden="true">00</span>
          <div><h3>No local references yet</h3><p>Open a dataset once and its reference will appear here.</p></div>
        </div>
      : <ol className="recent-datasets__list" aria-label={`${entries.length} recent datasets`}>
          {entries.map((entry, index) => {
            const reopening = reopeningIdentity === entry.datasetIdentity;
            return <li className="recent-datasets__row" key={entry.datasetIdentity}>
              <span className="recent-datasets__index" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
              <div className="recent-datasets__identity">
                <div className="recent-datasets__kindline">
                  <span>{sourceLabel(entry)}</span>
                  {entry.pinned && <span className="recent-datasets__pinmark">Pinned</span>}
                </div>
                <h3>{entry.source.displayName}</h3>
                <p title={entry.selection.paths.join("\n")}>{referenceLabel(entry)}</p>
                <time dateTime={entry.lastOpenedAt}>Opened {openedLabel(entry.lastOpenedAt)}</time>
              </div>
              <div className="recent-datasets__actions">
                <button
                  className="recent-datasets__open"
                  type="button"
                  disabled={reopening}
                  aria-busy={reopening}
                  onClick={() => void onReopen(entry)}
                >{reopening ? "Opening…" : "Open"}</button>
                {onPinChange && <button
                  type="button"
                  aria-pressed={entry.pinned}
                  onClick={() => onPinChange(entry, !entry.pinned)}
                >{entry.pinned ? "Unpin" : "Pin"}</button>}
                {onForget && <button className="recent-datasets__forget" type="button" onClick={() => onForget(entry)}>Forget</button>}
              </div>
            </li>;
          })}
        </ol>}
  </section>;
}
