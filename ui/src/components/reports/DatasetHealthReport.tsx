import { useMemo, useRef, useState } from "react";
import type { LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { datasetSourceIdentity } from "@/lib/dataset/datasetSources";
import {
  analyzeDatasetStructure,
  analyzeFieldValues,
  createDatasetFieldHealthReport,
  serializeDatasetFieldHealth,
  type DatasetFieldHealthReport,
  type FieldFrameHealth,
} from "@/lib/dataset/dataHealth";
import { readFieldAny, type StepInfo } from "@/lib/run/runOutputs";
import { cropToValidPlotDomain } from "@/lib/science/validDomain";

interface DatasetHealthReportProps {
  dataset: LocalDatasetSource;
  steps: readonly StepInfo[];
  stepIdx: number;
  incompleteCount: number;
  onClose: () => void;
}

function number(value: number | null): string { return value == null ? "—" : Number(value.toPrecision(7)).toLocaleString(); }
function safeName(value: string): string { return value.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 100) || "dataset"; }

function downloadJson(contents: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename;
  document.body.appendChild(anchor); anchor.click(); anchor.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function DatasetHealthReport({ dataset, steps, stepIdx, incompleteCount, onClose }: DatasetHealthReportProps) {
  const identity = datasetSourceIdentity(dataset);
  const structure = useMemo(() => analyzeDatasetStructure(steps), [steps]);
  const fields = useMemo(() => [...new Set(steps.flatMap((step) => step.fields))].sort(), [steps]);
  const [fieldName, setFieldName] = useState(() => fields.includes("reflectivity_dbz") ? "reflectivity_dbz" : fields[0] ?? "");
  const [scope, setScope] = useState<"current_frame" | "all_available_frames">("current_frame");
  const [report, setReport] = useState<DatasetFieldHealthReport | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const generationRef = useRef(0);

  const scan = async () => {
    const generation = ++generationRef.current;
    setScanning(true); setError(null); setReport(null);
    const requested = scope === "current_frame" ? steps.slice(stepIdx, stepIdx + 1) : [...steps];
    const available = requested.filter((step) => step.fields.includes(fieldName));
    const frames: FieldFrameHealth[] = [];
    try {
      for (let index = 0; index < available.length; index += 1) {
        if (generation !== generationRef.current) return;
        const step = available[index]; setProgress(`${index}/${available.length} frames`);
        const loaded = await readFieldAny(dataset.locator.path, step.index, fieldName, step.publicationGeneration, { latestScope: `data-health:${identity}` });
        if (generation !== generationRef.current) return;
        const valid = cropToValidPlotDomain(loaded, step.grid);
        frames.push({ ...analyzeFieldValues(valid.field, fieldName), shape: [...valid.field.shape], stepIndex: step.index, simulationTimeS: step.simulationTimeS });
        setProgress(`${index + 1}/${available.length} frames`);
      }
      const unit = available.find((step) => step.fieldUnits[fieldName]?.trim())?.fieldUnits[fieldName] ?? "";
      setReport(createDatasetFieldHealthReport({ dataset, fieldName, unit, requestedScope: scope, requestedFrameCount: requested.length, missingFieldFrameCount: requested.length - available.length, structure, frames }));
    } catch (reason) { if (generation === generationRef.current) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (generation === generationRef.current) setScanning(false); }
  };

  const cancel = () => { generationRef.current += 1; setScanning(false); setProgress("Cancelled"); };
  const nonfinite = report ? report.nanCount + report.positiveInfinityCount + report.negativeInfinityCount : 0;

  return <section className="data-health" aria-label="Dataset health report">
    <header className="data-health__header"><div><span>DATASET INTEGRITY · SOURCE-NEUTRAL</span><h3>Data health</h3><p>Metadata is immediate. Value scans are explicit and read only the selected field and frames.</p></div><button type="button" onClick={onClose}>Close</button></header>
    <div className="data-health__ledger" aria-label="Dataset structure health">
      <div><span>Frames</span><strong>{structure.frameCount}</strong><small>{incompleteCount} hidden incomplete</small></div>
      <div><span>Geometry</span><strong>{structure.physicalFrameCount === structure.frameCount ? "Physical" : structure.indexFrameCount > 0 ? "Mixed / index" : "Incomplete"}</strong><small>{structure.exactCoordinateFrameCount}/{structure.frameCount} exact coordinate sets</small></div>
      <div><span>Time</span><strong>{structure.time.hasWarnings ? "Needs review" : structure.time.expectedCadenceS == null ? "Limited" : "Qualified"}</strong><small>{structure.time.summary}</small></div>
      <div><span>Field coverage</span><strong>{structure.sharedFieldCount} shared</strong><small>{structure.unionFieldCount} total · {structure.varyingFieldCount} varying</small></div>
      <div><span>Units</span><strong>{structure.fieldsMissingUnits.length === 0 ? "Declared" : `${structure.fieldsMissingUnits.length} missing`}</strong><small>{structure.fieldsMissingUnits.length ? structure.fieldsMissingUnits.slice(0, 5).join(", ") : "Every published field has a declared unit"}</small></div>
      <div><span>Unit families</span><strong>{structure.unitCompatibilityIssues.length === 0 ? "Consistent" : `${structure.unitCompatibilityIssues.length} conflict${structure.unitCompatibilityIssues.length === 1 ? "" : "s"}`}</strong><small>{structure.unitCompatibilityIssues.length ? structure.unitCompatibilityIssues.map(({ family }) => family).join(", ") : "Wind, temperature, and moisture families agree"}</small></div>
    </div>
    <div className="data-health__scan">
      <div><span className="data-health__kicker">EXACT VALUE SCAN</span><h4>Inspect finite coverage and extrema</h4><p>No sampling: every value in each valid-domain field payload is classified.</p></div>
      <label>Field<select value={fieldName} onChange={(event) => { setFieldName(event.target.value); setReport(null); }}>{fields.map((field) => <option key={field}>{field}</option>)}</select></label>
      <fieldset><legend>Scope</legend><label><input type="radio" checked={scope === "current_frame"} onChange={() => { setScope("current_frame"); setReport(null); }} />Current frame</label><label><input type="radio" checked={scope === "all_available_frames"} onChange={() => { setScope("all_available_frames"); setReport(null); }} />All frames</label></fieldset>
      <div className="data-health__actions">{scanning ? <button type="button" onClick={cancel}>Cancel scan</button> : <button type="button" disabled={!fieldName} onClick={() => void scan()}>Scan exact values</button>}{progress && <span>{progress}</span>}</div>
    </div>
    {error && <div className="data-health__error" role="alert">{error}</div>}
    {report && <div className="data-health__result" aria-label="Exact field health results">
      <div className={nonfinite > 0 ? "is-warning" : "is-clean"}><span>Finite</span><strong>{report.finiteCount.toLocaleString()} / {report.elementCount.toLocaleString()}</strong><small>{nonfinite === 0 ? "Complete finite coverage" : `${nonfinite.toLocaleString()} non-finite values`}</small></div>
      <div><span>Range</span><strong>{number(report.minimum)} → {number(report.maximum)}</strong><small>{report.unit || "unit not declared"}</small></div>
      <div><span>Mean</span><strong>{number(report.mean)}</strong><small>{report.constantFiniteField ? "Constant finite field" : "Across finite values only"}</small></div>
      <div><span>Non-finite</span><strong>{nonfinite.toLocaleString()}</strong><small>{report.nanCount} NaN · {report.positiveInfinityCount} +∞ · {report.negativeInfinityCount} −∞</small></div>
      <div><span>Frames</span><strong>{report.scannedFrameCount} / {report.requestedFrameCount}</strong><small>{report.missingFieldFrameCount} without this field</small></div>
      <div className={report.physicalViolationCount > 0 ? "is-warning" : "is-clean"}><span>Physical domain</span><strong>{report.physicalRule ? report.physicalViolationCount.toLocaleString() : "Not defined"}</strong><small>{report.physicalRule ?? "No canonical scalar-domain rule for this field"}</small></div>
      <button type="button" onClick={() => downloadJson(serializeDatasetFieldHealth(report), `${safeName(dataset.displayName)}-${safeName(fieldName)}-health.json`)}>Export health report</button>
    </div>}
  </section>;
}
