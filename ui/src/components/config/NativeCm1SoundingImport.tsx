import { useState } from "react";
import type { Cm1SoundingReport } from "@/lib/dataset/datasetSources";
import {
  formatHeight,
  summarizeNativeCm1Sounding,
} from "@/lib/cm1/cm1NativeSounding";
import {
  inspectNativeCm1Sounding,
  pickNativeCm1SoundingFile,
} from "@/lib/platform/tauri";
import { Icon } from "@/components/shell/Icon";

type NativeSoundingState =
  | { kind: "idle" }
  | { kind: "checking"; path: string }
  | { kind: "accepted"; report: Cm1SoundingReport }
  | { kind: "error"; path: string; message: string };

interface NativeCm1SoundingImportProps {
  configuredPath: string;
  hodographType: string;
  modelTopM: number | null;
  desktop: boolean;
  onAdopt: (path: string) => void;
  onUseEmbeddedWinds: () => void;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return String(error);
}

function basename(path: string): string {
  return path.split(/[/\\]/).pop() || path;
}

export function NativeCm1SoundingImport({
  configuredPath,
  hodographType,
  modelTopM,
  desktop,
  onAdopt,
  onUseEmbeddedWinds,
}: NativeCm1SoundingImportProps) {
  const [state, setState] = useState<NativeSoundingState>({ kind: "idle" });
  const evidence = state.kind === "accepted" && state.report.path === configuredPath
    ? state.report
    : null;
  const summary = evidence ? summarizeNativeCm1Sounding(evidence, modelTopM) : null;

  const inspect = async (path: string) => {
    if (!path.trim()) return;
    setState({ kind: "checking", path });
    try {
      const report = await inspectNativeCm1Sounding(path);
      setState({ kind: "accepted", report });
      // Adoption is atomic from the form's perspective: only a successfully
      // parsed native profile can select File + embedded-wind ownership.
      onAdopt(report.path);
    } catch (error) {
      setState({ kind: "error", path, message: errorMessage(error) });
    }
  };

  const choose = async () => {
    const selected = await pickNativeCm1SoundingFile();
    if (selected) await inspect(selected);
  };

  const status = evidence
    ? "Native CM1 accepted"
    : state.kind === "checking"
      ? "Inspecting"
      : state.kind === "error"
        ? "Not accepted"
        : "Optional native path";

  return (
    <section
      className={`native-cm1-sounding native-cm1-sounding--${state.kind}`}
      aria-labelledby="native-cm1-sounding-title"
      aria-live="polite"
    >
      <div className="native-cm1-sounding__header">
        <div>
          <span className="config-editor__eyebrow">Additional source format</span>
          <h3 id="native-cm1-sounding-title">Native CM1 input_sounding</h3>
          <p>Use the original CM1 θ, qv, u, and v columns directly—no SHARPy projection or rewritten copy.</p>
        </div>
        <span className="native-cm1-sounding__status"><span aria-hidden="true" />{status}</span>
      </div>

      {evidence && summary ? (
        <>
          <dl className="native-cm1-sounding__ledger">
            <div className="native-cm1-sounding__wide">
              <dt>Source</dt>
              <dd title={evidence.path}>{basename(evidence.path)}</dd>
            </div>
            <div><dt>Native levels</dt><dd>{summary.levelCount.toLocaleString()}</dd></div>
            <div><dt>Vertical extent</dt><dd>{formatHeight(summary.bottomHeightM)}–{formatHeight(summary.topHeightM)}</dd></div>
            <div><dt>Surface</dt><dd>{evidence.surface.pressureHpa.toFixed(1)} hPa · θ {evidence.surface.potentialTemperatureK.toFixed(1)} K · qv {evidence.surface.mixingRatioGkg.toFixed(2)} g/kg</dd></div>
            <div><dt>Embedded winds</dt><dd>u/v present · peak {summary.peakWindMs.toFixed(1)} m/s</dd></div>
            <div><dt>Wind owner</dt><dd>{hodographType === "auto" ? "CM1 u/v column" : "Nimbus hodograph override"}</dd></div>
          </dl>
          <p className={`native-cm1-sounding__coverage is-${summary.coverage}`}>
            {summary.coverageMessage}
          </p>
          {evidence.warnings.length > 0 && (
            <ul className="native-cm1-sounding__warnings">
              {evidence.warnings.map((warning) => <li key={warning}>{warning}</li>)}
            </ul>
          )}
        </>
      ) : state.kind === "checking" ? (
        <p className="native-cm1-sounding__message">Parsing the native profile and checking its bounded scientific contract…</p>
      ) : state.kind === "error" ? (
        <div className="native-cm1-sounding__failure" role="alert">
          <strong>This file was not adopted</strong>
          <span>{state.message}</span>
        </div>
      ) : (
        <p className="native-cm1-sounding__message">
          Existing Nimbus sounding formats remain available through the file path above. Use this inspector only for native CM1 files.
        </p>
      )}

      <div className="native-cm1-sounding__actions">
        <button
          type="button"
          className="btn btn--secondary btn--small"
          onClick={() => void choose()}
          disabled={!desktop || state.kind === "checking"}
          title={desktop ? "Choose and validate an extensionless or named CM1 input_sounding" : "Native file inspection requires Nimbus Studio desktop"}
        >
          <Icon name="folder" size={13} />
          Choose native CM1 sounding…
        </button>
        {configuredPath && (
          <button
            type="button"
            className="btn btn--secondary btn--small"
            onClick={() => void inspect(configuredPath)}
            disabled={!desktop || state.kind === "checking"}
          >
            Inspect configured file
          </button>
        )}
        {evidence && hodographType !== "auto" && (
          <button type="button" className="native-cm1-sounding__wind-action" onClick={onUseEmbeddedWinds}>
            Use embedded u/v
          </button>
        )}
      </div>
      <p className="native-cm1-sounding__retention">
        Nimbus reads the source in place and never modifies it. Exact EOS reconstruction and model-top validation still occur in the simulator.
      </p>
    </section>
  );
}
