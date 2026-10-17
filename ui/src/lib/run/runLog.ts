import { useEffect, useRef, useState } from "react";
import type { LineEvent } from "@/lib/core/types";
import { onRunLine } from "@/lib/platform/tauri";

/* Ring-buffered log lines per run. The component renders a smaller bounded window. */

export const MAX_LOG_LINES = 4096;

/* Classification for LogStream syntax highlights. Matches simulator log conventions (include/util/log.hpp). */
export type LineKind =
  | "info"
  | "debug"
  | "warn"
  | "error"
  | "fatal"
  | "velmax"
  | "budget"
  | "progress"
  | "default";

export interface LogLine {
  id: number;
  stream: "stdout" | "stderr";
  line: string;
  ts: number;
  /** Classified once at ingestion; rendering never rescans the line text. */
  kind: LineKind;
}

export interface UseRunLogResult {
  lines: LogLine[];
  warnCount: number;
  errorCount: number;
  clear: () => void;
}

export interface RunLogSnapshot {
  lines: LogLine[];
  warnCount: number;
  errorCount: number;
}

const EMPTY_LOG: RunLogSnapshot = { lines: [], warnCount: 0, errorCount: 0 };

function severityDelta(line: LogLine): readonly [warn: number, error: number] {
  if (line.kind === "warn") return [1, 0];
  if (line.kind === "error" || line.kind === "fatal") return [0, 1];
  return [0, 0];
}

/** Append a classified batch and update retained-line counters incrementally. */
export function appendLogBatch(
  previous: RunLogSnapshot,
  batch: readonly LogLine[],
  maxLines = MAX_LOG_LINES,
): RunLogSnapshot {
  if (batch.length === 0) return previous;
  let warnCount = previous.warnCount;
  let errorCount = previous.errorCount;
  for (const line of batch) {
    const [warn, error] = severityDelta(line);
    warnCount += warn;
    errorCount += error;
  }

  const combined = previous.lines.concat(batch);
  const dropCount = Math.max(0, combined.length - Math.max(0, maxLines));
  for (let index = 0; index < dropCount; index++) {
    const [warn, error] = severityDelta(combined[index]);
    warnCount -= warn;
    errorCount -= error;
  }
  return {
    lines: dropCount === 0 ? combined : combined.slice(dropCount),
    warnCount,
    errorCount,
  };
}

export function useRunLog(runId: string | null): UseRunLogResult {
  const [snapshot, setSnapshot] = useState<RunLogSnapshot>(EMPTY_LOG);
  // Monotonically increasing line ID so React keys stay stable even if a
  // line repeats verbatim.
  const nextIdRef = useRef(1);

  useEffect(() => {
    if (!runId) {
      setSnapshot(EMPTY_LOG);
      return;
    }
    setSnapshot(EMPTY_LOG);
    let cancelled = false;
    let unsubscribe: (() => void) | undefined;
    // One IPC event per stdout line; buffer and commit once per frame to avoid a React commit per line.
    let pending: LogLine[] = [];
    let flushHandle: number | null = null;

    const flush = () => {
      flushHandle = null;
      if (cancelled || pending.length === 0) return;
      const batch = pending;
      pending = [];
      setSnapshot((previous) => appendLogBatch(previous, batch));
    };

    (async () => {
      const unsub = await onRunLine((evt: LineEvent) => {
        if (cancelled) return;
        if (evt.runId !== runId) return;
        const line = {
          id: nextIdRef.current++,
          stream: evt.stream,
          line: evt.line,
          ts: evt.ts,
        };
        pending.push({ ...line, kind: classifyLine(line) });
        // rAF does not fire while the window is hidden, so cap the buffer to
        // the same bound the committed list uses.
        if (pending.length > MAX_LOG_LINES) pending = pending.slice(-MAX_LOG_LINES);
        if (flushHandle == null) flushHandle = requestAnimationFrame(flush);
      });
      // Unmounted while listen() was resolving -> unsubscribe now, else the
      // cleanup already ran with `unsubscribe` undefined and this leaks.
      if (cancelled) unsub();
      else unsubscribe = unsub;
    })();

    return () => {
      cancelled = true;
      if (flushHandle != null) cancelAnimationFrame(flushHandle);
      pending = [];
      unsubscribe?.();
    };
  }, [runId]);

  const clear = () => setSnapshot(EMPTY_LOG);
  return { ...snapshot, clear };
}

export function classifyLine(line: Pick<LogLine, "stream" | "line">): LineKind {
  const t = line.line;
  if (line.stream === "stderr") {
    if (t.includes("[FATAL]")) return "fatal";
    if (t.includes("[ERROR]")) return "error";
    if (t.includes("[WARN]")) return "warn";
    return "error";
  }
  if (t.includes("[FATAL]")) return "fatal";
  if (t.includes("[ERROR]")) return "error";
  if (t.includes("[WARN]")) return "warn";
  if (t.includes("[VEL-MAX]")) return "velmax";
  if (t.includes("[PHYSICS BUDGET]")) return "budget";
  if (t.includes("[TIME STEP DEBUG]") || t.includes("[EXPORT DEBUG]")) return "progress";
  if (t.includes("[DEBUG]")) return "debug";
  if (t.includes("[INFO]")) return "info";
  return "default";
}
