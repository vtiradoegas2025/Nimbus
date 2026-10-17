import { useEffect, useRef, useState } from "react";
import { ffmpegDetect, getSimulatorStatus, isTauriRuntime } from "@/lib/platform/tauri";

export function SettingsPathCheck({ kind, value }: { kind: "simulator" | "ffmpeg"; value: string }) {
  const [result, setResult] = useState<{ failed: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const generation = useRef(0);
  useEffect(() => {
    const requests = generation;
    requests.current++;
    setResult(null);
    setBusy(false);
    return () => { requests.current++; };
  }, [value]);
  const check = async () => {
    const current = ++generation.current;
    setBusy(true);
    setResult(null);
    try {
      const path = value.trim() || undefined;
      let text: string;
      if (kind === "simulator") {
        const info = await getSimulatorStatus(path);
        if (info.state !== "ready" || !info.binaryPath) throw new Error("No usable simulator found at this path.");
        text = `Simulator found: ${info.binaryPath}. Configuration validation still runs before launch.`;
      } else {
        const info = await ffmpegDetect(path);
        if (!info.path) throw new Error("ffmpeg was not found. Choose an installed executable or leave the path empty to search PATH.");
        text = `Encoder found: ${info.path}${info.version ? ` · ${info.version}` : ""}.`;
      }
      if (current === generation.current) setResult({ failed: false, text });
    } catch (error) {
      if (current === generation.current) setResult({ failed: true, text: error instanceof Error ? error.message : String(error) });
    } finally {
      if (current === generation.current) setBusy(false);
    }
  };
  return <div className="settings-path-check">
    <button type="button" disabled={busy || !isTauriRuntime()} onClick={() => void check()}
      title={!isTauriRuntime() ? "Path checks require Nimbus Studio desktop" : busy ? "Checking this path" : "Check the current field without saving it"}>
      {busy ? "Checking path…" : kind === "simulator" ? "Check simulator path" : "Check ffmpeg path"}
    </button>
    <span className="settings-row__hint">Checks the current value. Save settings to apply changes.</span>
    {result && <p className="settings-note" role={result.failed ? "alert" : "status"}>{result.text}</p>}
  </div>;
}
