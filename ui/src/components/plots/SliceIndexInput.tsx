import { useState } from "react";

export function SliceIndexInput({ value, max, disabled, label, onCommit }: {
  value: number; max: number; disabled?: boolean; label: string; onCommit: (value: number) => void;
}) {
  const [edit, setEdit] = useState<{ source: number; text: string } | null>(null);
  const text = edit?.source === value ? edit.text : String(value);
  const commit = () => {
    const parsed = text.trim() === "" ? NaN : Number(text);
    const next = Math.max(0, Math.min(max, parsed));
    if (Number.isSafeInteger(parsed) && !disabled && next !== value) onCommit(next);
    setEdit(null);
  };
  return <input className="slice-index-input" type="number" inputMode="numeric" step={1}
    aria-label={label} title="Enter a model index, then press Enter" min={0} max={max}
    disabled={disabled} value={text}
    onChange={event => setEdit({ source: value, text: event.target.value })}
    onBlur={commit}
    onKeyDown={event => {
      if (event.key === "Enter") { event.preventDefault(); commit(); }
      if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); setEdit(null); }
    }} />;
}
