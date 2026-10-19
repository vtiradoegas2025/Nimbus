import type { OutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { useRef } from "react";
import { ControlGroup } from "./common";

function levelLabel(levelId: string, depth: number, spacingM: number | undefined): string {
  if (levelId === "root") return "Root domain";
  const name = levelId.split("/").at(-1)?.replace("_", " ") ?? levelId;
  const spacing = spacingM != null ? ` · ${spacingM.toLocaleString()} m` : "";
  return `${"↳ ".repeat(Math.max(0, depth - 1))}${name}${spacing}`;
}

/** Compact branch selector shared by every hierarchy-aware field plot. */
export function OutputHierarchyControls({ source }: { source: OutputHierarchySource }) {
  const visibility = useRef({ runDir: source.rootRunDir, shown: false });
  if (visibility.current.runDir !== source.rootRunDir) visibility.current = { runDir: source.rootRunDir, shown: false };
  if (source.availableLevels.length > 1 || source.error) visibility.current.shown = true;
  if (!visibility.current.shown) return null;
  const selectedExists = source.requestedLevelId === "auto"
    || source.availableLevels.some((level) => level.levelId === source.requestedLevelId);
  const detail = source.selection?.fallbackReason ?? source.error ?? source.selection?.disclosure ?? "Root history";
  return (
    <ControlGroup label="Domain detail" stretch>
      <select
        className="volume-render__select"
        value={source.requestedLevelId}
        onChange={(event) => source.setRequestedLevelId(event.target.value)}
        disabled={source.loading}
        title="Auto follows the finest published nest containing the storm focus; explicit choices stay on that declared branch"
      >
        <option value="auto">{source.loading ? "Discovering domains…" : "Auto · follow finest"}</option>
        {!selectedExists && <option value={source.requestedLevelId} disabled>{source.requestedLevelId} · not published in this frame</option>}
        {source.availableLevels.map((level) => (
          <option key={level.levelId} value={level.levelId}>
            {levelLabel(level.levelId, level.depth, level.spacingM)}
          </option>
        ))}
      </select>
      <span className="volume-render__slider-value output-hierarchy__detail" title={detail}>
        {detail}
      </span>
    </ControlGroup>
  );
}
