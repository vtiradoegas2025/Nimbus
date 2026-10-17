import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FloorReduction } from "@/lib/plot/radarFloor";
import { availableScalarPlanes } from "@/lib/plot/scalarPlanePolicy";
import { ControlGroup, PillGroup } from "./common";
import {
  publishShared3DSetting,
  readLocal3DSetting,
  readShared3DSetting,
  storeLocal3DSetting,
  subscribeShared3DSetting,
  type ThreeDControlScope,
} from "@/lib/plot/threeDControlScope";

/* Shared ground-plane radar floor state — off by default, opt-in layer. */

export interface RadarFloorSelection {
  enabled: boolean;
  fieldName: string | null;
  reduction: FloorReduction;
  opacity: number;
  /** Fields in this run that can be drawn as a ground plane. */
  planeFields: string[];
  setEnabled: (next: boolean) => void;
  setFieldName: (next: string | null) => void;
  setReduction: (next: FloorReduction) => void;
  setOpacity: (next: number) => void;
}

type SharedFloorSelection = Pick<
  RadarFloorSelection,
  "enabled" | "fieldName" | "reduction" | "opacity"
>;
let liveSharedFloorSelection: { datasetKey: string; selection: SharedFloorSelection } | null = null;

function isSharedFloorSelection(value: unknown): value is SharedFloorSelection {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<SharedFloorSelection>;
  return typeof candidate.enabled === "boolean"
    && (candidate.fieldName == null || typeof candidate.fieldName === "string")
    && (candidate.reduction === "surface" || candidate.reduction === "column-max")
    && typeof candidate.opacity === "number"
    && Number.isFinite(candidate.opacity)
    && candidate.opacity >= 0.1
    && candidate.opacity <= 1;
}

export function useRadarFloorSelection(
  fieldsForStep: readonly string[],
  scope: ThreeDControlScope = "independent",
  viewKey = "default",
): RadarFloorSelection {
  const datasetKey = viewKey.includes(":") ? viewKey.slice(viewKey.indexOf(":") + 1) : viewKey;
  const planeFields = useMemo(
    () => availableScalarPlanes(fieldsForStep),
    [fieldsForStep],
  );
  const planeFieldsRef = useRef(planeFields);
  planeFieldsRef.current = planeFields;
  const [enabled, setEnabledState] = useState(false);
  const [selectionViewKey, setSelectionViewKey] = useState(viewKey);
  const [fieldName, setFieldNameState] = useState<string | null>(planeFields[0] ?? null);
  // Default "surface": matches the lowest radar tilt; "column-max" available for aloft signatures.
  const [reduction, setReductionState] = useState<FloorReduction>("surface");
  const [opacity, setOpacityState] = useState(1);

  const applyShared = useCallback((next: SharedFloorSelection) => {
    setEnabledState(next.enabled);
    setFieldNameState(next.fieldName);
    setReductionState(next.reduction);
    setOpacityState(next.opacity);
  }, []);
  useEffect(() => {
    if (scope === "independent") {
      const local = readLocal3DSetting(viewKey, "radar-floor", isSharedFloorSelection);
      applyShared(local
        ? { ...local, enabled: false }
        : { enabled: false, fieldName: planeFieldsRef.current[0] ?? null, reduction: "surface", opacity: 1 });
      setSelectionViewKey(viewKey);
      return;
    }
    const stored = liveSharedFloorSelection?.datasetKey === datasetKey
      ? liveSharedFloorSelection.selection
      : readShared3DSetting("radar-floor", isSharedFloorSelection);
    const initial = stored
      ? { ...stored, enabled: liveSharedFloorSelection?.datasetKey === datasetKey && stored.enabled }
      : { enabled: false, fieldName: planeFieldsRef.current[0] ?? null, reduction: "surface" as const, opacity: 1 };
    applyShared(initial);
    liveSharedFloorSelection = { datasetKey, selection: initial };
    setSelectionViewKey(viewKey);
    return subscribeShared3DSetting("radar-floor", isSharedFloorSelection, (next) => {
      if (liveSharedFloorSelection?.datasetKey !== datasetKey) return;
      liveSharedFloorSelection = { datasetKey, selection: next };
      applyShared(next);
    });
  }, [applyShared, datasetKey, scope, viewKey]);
  const publish = useCallback((next: SharedFloorSelection) => {
    if (scope === "shared") {
      liveSharedFloorSelection = { datasetKey, selection: next };
      publishShared3DSetting("radar-floor", next);
    }
    else storeLocal3DSetting(viewKey, "radar-floor", next);
  }, [datasetKey, scope, viewKey]);
  const setEnabled = useCallback((next: boolean) => {
    setEnabledState(next);
    publish({ enabled: next, fieldName, reduction, opacity });
  }, [fieldName, opacity, publish, reduction]);
  const setFieldName = useCallback((next: string | null) => {
    setFieldNameState(next);
    publish({ enabled, fieldName: next, reduction, opacity });
  }, [enabled, opacity, publish, reduction]);
  const setReduction = useCallback((next: FloorReduction) => {
    setReductionState(next);
    publish({ enabled, fieldName, reduction: next, opacity });
  }, [enabled, fieldName, opacity, publish]);
  const setOpacity = useCallback((next: number) => {
    const bounded = Math.max(0.1, Math.min(1, next));
    setOpacityState(bounded);
    publish({ enabled, fieldName, reduction, opacity: bounded });
  }, [enabled, fieldName, publish, reduction]);

  // A run change can retire the selected field.
  useEffect(() => {
    if (fieldName && planeFields.includes(fieldName)) return;
    setFieldNameState(planeFields[0] ?? null);
  }, [fieldName, planeFields]);

  return {
    enabled: selectionViewKey === viewKey ? enabled : false,
    fieldName,
    reduction,
    opacity,
    planeFields,
    setEnabled,
    setFieldName,
    setReduction,
    setOpacity,
  };
}

export function RadarFloorControls({
  selection,
  lattice,
  paintedCells,
  error,
  source,
  loading,
}: {
  selection: RadarFloorSelection;
  /** Reported so the displayed resolution is never overstated. */
  lattice: { nx: number; ny: number; strideXY: number } | null;
  /** Non-zero-alpha cells painted; zero means echo is fully below the cutoff. */
  paintedCells?: number | null;
  /** Load error message, shown explicitly (not swallowed). */
  error?: string | null;
  /** Field actually read; substituted=true means a fallback was used. */
  source?: { fieldName: string; substituted: boolean } | null;
  /** True while a read is in flight (used to suppress stale empty verdict). */
  loading?: boolean;
}) {
  const { enabled, fieldName, reduction, opacity, planeFields } = selection;
  const unavailable = planeFields.length === 0 || !fieldName;
  const failed = enabled && !unavailable && error != null;

  // empty: sheet loaded but all cells below clear-air cutoff (common with surface reduction + aloft echo).
  const empty = enabled && !unavailable && !failed && !loading
    && lattice != null && paintedCells === 0;
  // unencoded: lattice loaded but no finite samples encoded (missing/masked data — column-max won't help).
  const unencoded = enabled && !unavailable && !failed && !loading
    && lattice != null && paintedCells == null;

  return (
    <>
      <ControlGroup label="Radar floor">
        <button
          type="button"
          className={`pill ${enabled && !unavailable && !failed ? "pill--active" : ""}${
            failed ? " pill--error" : ""
          }`}
          onClick={() => selection.setEnabled(!enabled)}
          disabled={unavailable}
          aria-busy={enabled && loading ? true : undefined}
          title={unavailable
            ? "No ground-plane field exported in this run"
            : failed
            ? error ?? ""
            : empty
            ? `No ${fieldName} at or above the clear-air cutoff on this plane, so the sheet is fully transparent. Try Column max if the echo is aloft.`
            : unencoded
            ? `The ${fieldName} window loaded but held no finite samples, so no sheet was built. This is missing or masked data rather than a threshold — Column max will not help.`
            : source?.substituted
            ? `Ground-plane sheet, read from the engine's own ${source.fieldName}${
              paintedCells != null ? ` · ${paintedCells.toLocaleString()} cells painted` : ""
            }`
            : `Ground-plane sheet, loaded at native resolution${
              paintedCells != null ? ` · ${paintedCells.toLocaleString()} cells painted` : ""
            }`}
        >
          {unavailable
            ? "n/a"
            : !enabled
            ? "off"
            : failed
            ? "on · failed"
            : !lattice
            ? "on · loading"
            : empty
            ? `on · ${lattice.nx}×${lattice.ny} · no echo`
            : unencoded
            ? `on · ${lattice.nx}×${lattice.ny} · no sheet`
            : `on · ${lattice.nx}×${lattice.ny}`}
        </button>
      </ControlGroup>
      {failed && (
        <ControlGroup label="Floor error" stretch>
          <span className="volume-render__control-error" role="status">{error}</span>
        </ControlGroup>
      )}
      <ControlGroup label="Floor field">
        <select
          className="volume-render__select"
          value={fieldName ?? ""}
          disabled={unavailable}
          onChange={(event) => selection.setFieldName(event.target.value || null)}
        >
          {planeFields.length === 0 && <option value="">n/a</option>}
          {planeFields.map((name) => (
            <option value={name} key={name}>{name}</option>
          ))}
        </select>
      </ControlGroup>
      <ControlGroup label="Floor reduction">
        <PillGroup
          options={[
            { value: "surface", label: "Surface" },
            { value: "column-max", label: "Column max" },
          ]}
          value={reduction}
          onChange={(next) => selection.setReduction(next)}
        />
      </ControlGroup>
      <ControlGroup label="Floor opacity">
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.05}
          value={opacity}
          disabled={!enabled || unavailable}
          aria-label="Radar floor opacity"
          onChange={(event) => selection.setOpacity(Number.parseFloat(event.target.value))}
          className="volume-render__slider"
        />
        <span className="volume-render__slider-value">{Math.round(opacity * 100)}%</span>
      </ControlGroup>
    </>
  );
}
