import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  DEFAULT_SCENE_APPEARANCE,
  resolveSceneAppearance,
  normalizeShadowDepth,
  normalizeFillTint,
  sceneAppearanceCssVariables,
  type ResolvedSceneAppearance,
  type SceneAppearanceSelection,
  type SceneBackgroundMode,
  type SceneLightingPreset,
} from "@/lib/plot/sceneAppearance";
import {
  currentStudioAppearance,
  STUDIO_APPEARANCE_APPLIED_EVENT,
} from "@/lib/studio/studioAppearance";
import { ControlGroup } from "./common";
import {
  readLocal3DSetting,
  storeLocal3DSetting,
  type ThreeDControlScope,
} from "@/lib/plot/threeDControlScope";
import {
  type VtkBackendPreference,
  type VtkBackendState,
} from "@/lib/vtk/vtkBackend";

const BACKGROUND_OPTIONS: ReadonlyArray<{ value: SceneBackgroundMode; label: string }> = [
  { value: "theme", label: "Follow theme" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
  { value: "custom", label: "Custom" },
];

const LIGHTING_OPTIONS: ReadonlyArray<{ value: SceneLightingPreset; label: string }> = [
  { value: "neutral", label: "Neutral" },
  { value: "soft", label: "Soft" },
  { value: "contrast", label: "High contrast" },
  { value: "flat", label: "Flat / exact color" },
];
const STORAGE_KEY = "nimbus-studio:scene-appearance:v1";
const CHANGE_EVENT = "nimbus-studio:scene-appearance-change";

function normalizeSelection(value: unknown): SceneAppearanceSelection {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_SCENE_APPEARANCE };
  }
  const candidate = value as Partial<SceneAppearanceSelection>;
  const backgroundMode = BACKGROUND_OPTIONS.some((option) => option.value === candidate.backgroundMode)
    ? candidate.backgroundMode!
    : DEFAULT_SCENE_APPEARANCE.backgroundMode;
  const lighting = LIGHTING_OPTIONS.some((option) => option.value === candidate.lighting)
    ? candidate.lighting!
    : DEFAULT_SCENE_APPEARANCE.lighting;
  const customBackground = typeof candidate.customBackground === "string"
    && /^#[0-9a-f]{6}$/i.test(candidate.customBackground)
    ? candidate.customBackground.toLowerCase()
    : DEFAULT_SCENE_APPEARANCE.customBackground;
  return { backgroundMode, customBackground, lighting,
    shadowDepth: normalizeShadowDepth(candidate.shadowDepth), fillTint: normalizeFillTint(candidate.fillTint) };
}

function loadSelection(): SceneAppearanceSelection {
  if (typeof window === "undefined") return { ...DEFAULT_SCENE_APPEARANCE };
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? normalizeSelection(JSON.parse(stored)) : { ...DEFAULT_SCENE_APPEARANCE };
  } catch {
    return { ...DEFAULT_SCENE_APPEARANCE };
  }
}

function localSelection(viewKey: string): SceneAppearanceSelection | null {
  const stored = readLocal3DSetting(
    viewKey,
    "scene-appearance",
    (value): value is Record<string, unknown> => Boolean(value && typeof value === "object"),
  );
  return stored ? normalizeSelection(stored) : null;
}

export interface SceneAppearanceState {
  selection: SceneAppearanceSelection;
  setSelection: React.Dispatch<React.SetStateAction<SceneAppearanceSelection>>;
  resolved: ResolvedSceneAppearance;
  hostStyle: CSSProperties;
}

export function useSceneAppearance(
  scope: ThreeDControlScope = "shared",
  viewKey = "default",
): SceneAppearanceState {
  const [selection, setSelectionState] = useState<SceneAppearanceSelection>(() =>
    scope === "independent" ? localSelection(viewKey) ?? loadSelection() : loadSelection()
  );
  const [theme, setTheme] = useState<"dark" | "light">(() => currentStudioAppearance());
  useEffect(() => {
    const onAppearance = () => setTheme(currentStudioAppearance());
    window.addEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
    return () => window.removeEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
  }, []);
  useEffect(() => {
    const onSceneAppearance = (event: Event) => {
      if (scope !== "shared" || !(event instanceof CustomEvent)) return;
      setSelectionState(normalizeSelection(event.detail));
    };
    window.addEventListener(CHANGE_EVENT, onSceneAppearance);
    return () => window.removeEventListener(CHANGE_EVENT, onSceneAppearance);
  }, [scope]);
  useEffect(() => {
    if (scope === "shared") {
      setSelectionState(loadSelection());
      return;
    }
    const local = localSelection(viewKey);
    if (local) setSelectionState(local);
  }, [scope, viewKey]);
  const setSelection = useCallback<React.Dispatch<React.SetStateAction<SceneAppearanceSelection>>>(
    (update) => {
      const next = normalizeSelection(typeof update === "function" ? update(selection) : update);
      setSelectionState(next);
      if (scope !== "shared") {
        storeLocal3DSetting(viewKey, "scene-appearance", next);
        return;
      }
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // The active scene still updates; persistence is best-effort.
      }
      window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: next }));
    },
    [scope, selection, viewKey],
  );
  const resolved = useMemo(
    () => resolveSceneAppearance(selection, theme),
    [selection, theme],
  );
  const hostStyle = useMemo(
    () => sceneAppearanceCssVariables(resolved) as CSSProperties,
    [resolved],
  );
  return { selection, setSelection, resolved, hostStyle };
}

export function SceneAppearanceControls({
  state,
  backend,
  lightingEnabled = true,
}: {
  state: SceneAppearanceState;
  backend?: VtkBackendState;
  lightingEnabled?: boolean;
}) {
  const { selection, setSelection } = state;
  const controlsDisabled = !lightingEnabled || selection.lighting === "flat";
  const shadowDepth = normalizeShadowDepth(selection.shadowDepth);
  const shadowDepthLabel = `${Math.round(shadowDepth * 50)}%`;
  return (
    <>
      <ControlGroup label="3D background">
        <select
          className="volume-render__select"
          value={selection.backgroundMode}
          onChange={(event) => setSelection((current) => ({
            ...current,
            backgroundMode: event.target.value as SceneBackgroundMode,
          }))}
          title="Scene clear color; scientific data colormaps are not changed"
        >
          {BACKGROUND_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        {selection.backgroundMode === "custom" && (
          <input
            type="color"
            className="scene-color-input"
            value={selection.customBackground}
            aria-label="Custom 3D background color"
            title="Custom 3D background color"
            onChange={(event) => setSelection((current) => ({
              ...current,
              customBackground: event.target.value,
            }))}
          />
        )}
      </ControlGroup>
      <ControlGroup label="3D lighting">
        <select
          className="volume-render__select"
          value={selection.lighting}
          onChange={(event) => setSelection((current) => ({
            ...current,
            lighting: event.target.value as SceneLightingPreset,
          }))}
          title="Lighting changes geometry cues; Flat preserves lookup-table colors"
        >
          {LIGHTING_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </ControlGroup>
      <ControlGroup label="Shadow depth">
        <input type="range" min={0} max={2} step={0.05}
          aria-label="Shadow depth" aria-valuetext={shadowDepthLabel} value={shadowDepth}
          disabled={controlsDisabled}
          title={lightingEnabled ? "Adjust shaded depth without changing scalar values or opacity" : "This field renders unlit to preserve its scalar structure"}
          onChange={event => setSelection(current => ({ ...current, shadowDepth: Number(event.target.value) }))} />
        <span className="volume-render__slider-value">{shadowDepthLabel}</span>
      </ControlGroup>
      <ControlGroup label="Shadow / fill tint">
        <select className="volume-render__select" aria-label="Shadow tint mode" disabled={controlsDisabled}
          value={selection.fillTint ? "custom" : "preset"}
          onChange={event => setSelection(current => ({ ...current, fillTint: event.target.value === "custom" ? "#ffffff" : null }))}>
          <option value="preset">Preset light color</option>
          <option value="custom">Custom fill tint</option>
        </select>
        {selection.fillTint && <input type="color" className="scene-color-input" aria-label="Shadow fill tint"
          value={selection.fillTint} disabled={controlsDisabled}
          onChange={event => setSelection(current => ({ ...current, fillTint: event.target.value }))} />}
      </ControlGroup>
      <button type="button" className="plot-reset-btn" onClick={() => setSelection(current => ({ ...current,
        lighting: DEFAULT_SCENE_APPEARANCE.lighting, shadowDepth: 1, fillTint: null }))}>Reset lighting</button>
      {(!lightingEnabled || selection.fillTint) && <span className="scene-lighting-notice">
        {!lightingEnabled ? "This field is unlit; shadow controls do not apply."
          : "Tint changes rendered hues. Flat / exact color removes lighting tint."}
      </span>}
      {backend && (
        <ControlGroup label="3D renderer" stretch>
          <select
            className="volume-render__select"
            value={backend.preference}
            onChange={(event) => backend.setPreference(
              event.target.value as VtkBackendPreference,
            )}
            title="Auto qualifies WebGPU and falls back to WebGL if the device or webview cannot sustain it"
          >
            <option value="auto">Auto (GPU preferred)</option>
            <option value="webgpu">WebGPU</option>
            <option value="webgl">WebGL compatibility</option>
          </select>
          <span className="volume-render__slider-value">
            {backend.qualifying
              ? "qualifying GPU"
              : backend.backend === "webgpu"
                ? "WebGPU active"
                : backend.volumeSafeFallback
                  ? "WebGL volume-safe"
                : backend.fallbackReason
                  ? "WebGL fallback"
                  : "WebGL active"}
          </span>
          {backend.fallbackReason && (
            <span className="sr-only" role="status">{backend.fallbackReason}</span>
          )}
        </ControlGroup>
      )}
    </>
  );
}
