import { useCallback, useEffect, useRef, useState } from "react";
import "./ResizeHandle.css";

/** One arrow press, in CSS pixels. */
const STEP_PX = 16;
/** One arrow press with Shift held. */
const COARSE_STEP_PX = 64;

interface ResizeHandleProps {
  /** Pointer-X delta in CSS pixels; positive = widen. */
  onResize: (deltaPx: number) => void;
  /** Optional label for assistive tech. */
  label?: string;
  /** `overlay` (default): absolutely positioned at the right edge. `inline`: in-flow for grid/flex splitter tracks. */
  placement?: "overlay" | "inline";
  /** Lower and upper pixel bounds announced for the adjustable separator. */
  minValuePx?: number;
  maxValuePx?: number;
}

/** Vertical drag-to-resize splitter. Arrow keys for a11y; setPointerCapture so drag survives leaving the hit region. */
export function ResizeHandle({
  onResize,
  label,
  placement = "overlay",
  minValuePx = 0,
  maxValuePx,
}: ResizeHandleProps) {
  const [dragging, setDragging] = useState(false);
  const [valuePx, setValuePx] = useState(minValuePx);
  const handleRef = useRef<HTMLDivElement>(null);
  const lastXRef = useRef<number>(0);
  const valueRef = useRef(valuePx);
  valueRef.current = valuePx;
  const resolvedMax = maxValuePx ?? Math.max(
    minValuePx + 1,
    typeof window === "undefined" ? 1_920 : window.innerWidth,
  );

  // The handle does not own the pane width, so observe the element whose edge
  // it moves and expose that real rendered size through separator semantics.
  useEffect(() => {
    const handle = handleRef.current;
    if (!handle) return;
    const target = placement === "overlay"
      ? handle.parentElement
      : handle.previousElementSibling;
    if (!(target instanceof HTMLElement)) return;
    const updateValue = () => {
      const measured = Math.round(target.getBoundingClientRect().width);
      setValuePx(Math.min(resolvedMax, Math.max(minValuePx, measured)));
    };
    updateValue();
    const observer = new ResizeObserver(updateValue);
    observer.observe(target);
    return () => observer.disconnect();
  }, [minValuePx, placement, resolvedMax]);

  const emitResize = useCallback((deltaPx: number) => {
    onResize(deltaPx);
    setValuePx((current) =>
      Math.min(resolvedMax, Math.max(minValuePx, current + deltaPx)),
    );
  }, [minValuePx, onResize, resolvedMax]);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      lastXRef.current = e.clientX;
      setDragging(true);
    },
    [],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return;
      const dx = e.clientX - lastXRef.current;
      if (dx === 0) return;
      lastXRef.current = e.clientX;
      emitResize(dx);
    },
    [dragging, emitResize],
  );

  const endDrag = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Pointer capture may already have been released by the browser
      // (e.g. on pointercancel). Safe to ignore.
    }
    setDragging(false);
  }, [dragging]);

  // Arrow keys: STEP_PX nudge or COARSE_STEP_PX with Shift; required for keyboard a11y on role="separator".
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const step = e.shiftKey ? COARSE_STEP_PX : STEP_PX;
      if (e.key === "ArrowLeft") emitResize(-step);
      else if (e.key === "ArrowRight") emitResize(step);
      else if (e.key === "Home") emitResize(minValuePx - valueRef.current);
      else if (e.key === "End") emitResize(resolvedMax - valueRef.current);
      else return;
      e.preventDefault();
    },
    [emitResize, minValuePx, resolvedMax],
  );

  return (
    <div
      ref={handleRef}
      className={`resize-handle resize-handle--${placement} ${
        dragging ? "resize-handle--active" : ""
      }`}
      role="separator"
      tabIndex={0}
      aria-orientation="vertical"
      aria-label={label ?? "Resize panel"}
      aria-valuemin={minValuePx}
      aria-valuemax={resolvedMax}
      aria-valuenow={valuePx}
      aria-valuetext={`${valuePx} pixels wide`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onKeyDown={onKeyDown}
    />
  );
}
