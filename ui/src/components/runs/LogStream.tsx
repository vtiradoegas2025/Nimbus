import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useRunLog, type LineKind, type LogLine } from "@/lib/run/runLog";
import { Icon } from "@/components/shell/Icon";
import "./LogStream.css";

interface LogStreamProps {
  runId: string | null;
}

/* Live-tail log viewer. Auto-scroll pauses on user scroll-up; collapse state persists in localStorage. */
const COLLAPSED_STORAGE_KEY = "nimbus-studio:log-stream:collapsed";
/** Keep the live DOM cheap while retaining a deeper in-memory diagnostic buffer. */
export const MAX_RENDERED_LOG_LINES = 600;

export function LogStream({ runId }: LogStreamProps) {
  const { lines, warnCount, errorCount, clear } = useRunLog(runId);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem(COLLAPSED_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(COLLAPSED_STORAGE_KEY, next ? "1" : "0");
      } catch {
        // Persistence best-effort; the toggle still works in-session.
      }
      return next;
    });
  };

  // Maintain bottom-anchor when autoScroll is true. useLayoutEffect to avoid
  // a frame of mid-list state being visible to the user.
  useLayoutEffect(() => {
    if (collapsed) return;
    const el = scrollerRef.current;
    if (!el) return;
    if (autoScroll) el.scrollTop = el.scrollHeight;
  }, [lines, autoScroll, collapsed]);

  // Detect user scroll-up: if scrollTop drifts >100 px from the bottom, the
  // user is reading history — pause autoscroll.
  useEffect(() => {
    if (collapsed) return;
    const el = scrollerRef.current;
    if (!el) return;
    const handler = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      setAutoScroll(distFromBottom < 100);
    };
    el.addEventListener("scroll", handler, { passive: true });
    return () => el.removeEventListener("scroll", handler);
  }, [collapsed]);

  const visibleLines = useMemo(
    () => lines.length > MAX_RENDERED_LOG_LINES
      ? lines.slice(-MAX_RENDERED_LOG_LINES)
      : lines,
    [lines],
  );

  const jumpToLatest = () => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setAutoScroll(true);
  };

  if (!runId) {
    return (
      <div className="log-stream log-stream--empty">
        <div className="log-stream__empty-message">
          <Icon name="info" size={14} />
          <span>Select a run to view its log.</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`log-stream ${collapsed ? "log-stream--collapsed" : ""}`}>
      <header className="log-stream__header">
        <button
          type="button"
          className="log-stream__toggle"
          onClick={toggleCollapsed}
          title={collapsed ? "Show log" : "Hide log"}
          aria-expanded={!collapsed}
        >
          <Icon name={collapsed ? "chevron-right" : "chevron-down"} size={12} />
          <span className="log-stream__title">Log</span>
        </button>
        <span className="log-stream__meta">
          {lines.length} lines
          {lines.length > visibleLines.length ? ` · latest ${visibleLines.length} shown` : ""}
        </span>
        {warnCount > 0 && (
          <span className="log-stream__counter log-stream__counter--warn">
            {warnCount} warn
          </span>
        )}
        {errorCount > 0 && (
          <span className="log-stream__counter log-stream__counter--error">
            {errorCount} err
          </span>
        )}
        {!collapsed && (
          <button
            type="button"
            className="log-stream__btn"
            onClick={clear}
            title="Clear log buffer (doesn't affect the running sim)"
          >
            Clear
          </button>
        )}
      </header>
      {!collapsed && (
        <div className="log-stream__body" ref={scrollerRef}>
          {lines.length === 0 ? (
            <div className="log-stream__placeholder">Waiting for output…</div>
          ) : (
            visibleLines.map((line) => <LogRow key={line.id} line={line} />)
          )}
        </div>
      )}
      {!collapsed && !autoScroll && (
        <button
          type="button"
          className="log-stream__resume"
          onClick={jumpToLatest}
        >
          Jump to latest ↓
        </button>
      )}
    </div>
  );
}

function LogRow({ line }: { line: LogLine }) {
  return <div className={`log-row log-row--${line.kind}`}>{line.line}</div>;
}

export type { LineKind };
