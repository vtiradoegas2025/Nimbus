import { sceneBackgroundFor, type SceneRgb } from "./sceneBackground";

export type SceneBackgroundMode = "theme" | "dark" | "light" | "custom";
export type SceneLightingPreset = "neutral" | "soft" | "contrast" | "flat";
export type SceneLightingSettings = {
  preset: SceneLightingPreset;
  shadowDepth: number;
  fillTint: string | null;
};
export type SceneLightingInput = SceneLightingPreset | SceneLightingSettings;

export function normalizeShadowDepth(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(2, value)) : 1;
}

export function normalizeFillTint(value: unknown): string | null {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value) ? value.toLowerCase() : null;
}

export function lightingSettings(input: SceneLightingInput): SceneLightingSettings {
  return typeof input === "string"
    ? { preset: input, shadowDepth: 1, fillTint: null }
    : {
      preset: input.preset,
      shadowDepth: normalizeShadowDepth(input.shadowDepth),
      fillTint: normalizeFillTint(input.fillTint),
    };
}

export function shadowMaterial(
  ambient: number,
  diffuse: number,
  input: SceneLightingInput,
): { ambient: number; diffuse: number } {
  const depth = lightingSettings(input).shadowDepth;
  if (depth === 1) return { ambient, diffuse };
  return depth <= 1
    ? { ambient: 1 + (ambient - 1) * depth, diffuse: diffuse * depth }
    : { ambient: ambient * (2 - depth), diffuse: Math.min(1, diffuse + ambient * (depth - 1)) };
}

export interface SceneAppearanceSelection {
  backgroundMode: SceneBackgroundMode;
  customBackground: string;
  lighting: SceneLightingPreset;
  shadowDepth?: number;
  fillTint?: string | null;
}

export interface ResolvedSceneAppearance {
  background: SceneRgb;
  backgroundHex: string;
  isLight: boolean;
  luminance: number;
  ink: SceneRgb;
  mutedInk: SceneRgb;
  reference: SceneRgb;
  scrim: string;
  scrimStrong: string;
  lighting: SceneLightingPreset;
  lightingOptions: SceneLightingSettings;
}

export const DEFAULT_SCENE_APPEARANCE: SceneAppearanceSelection = {
  backgroundMode: "theme",
  customBackground: "#202734",
  lighting: "neutral",
};

const DARK_INK: SceneRgb = [0.91, 0.93, 0.97];
const DARK_MUTED_INK: SceneRgb = [0.66, 0.7, 0.78];
const LIGHT_INK: SceneRgb = [0.11, 0.13, 0.17];
const LIGHT_MUTED_INK: SceneRgb = [0.34, 0.38, 0.45];

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function normalizeHexColor(value: string, fallback = "#202734"): string {
  const candidate = value.trim();
  if (/^#[0-9a-f]{6}$/i.test(candidate)) return candidate.toLowerCase();
  if (/^#[0-9a-f]{3}$/i.test(candidate)) {
    return `#${candidate.slice(1).split("").map((channel) => channel + channel).join("")}`
      .toLowerCase();
  }
  return fallback;
}

export function hexToSceneRgb(value: string): SceneRgb {
  const hex = normalizeHexColor(value).slice(1);
  return [
    Number.parseInt(hex.slice(0, 2), 16) / 255,
    Number.parseInt(hex.slice(2, 4), 16) / 255,
    Number.parseInt(hex.slice(4, 6), 16) / 255,
  ];
}

export function sceneRgbToHex(rgb: SceneRgb): string {
  const channel = (value: number) => Math.round(clamp01(value) * 255)
    .toString(16)
    .padStart(2, "0");
  return `#${channel(rgb[0])}${channel(rgb[1])}${channel(rgb[2])}`;
}

function linearChannel(channel: number): number {
  return channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4;
}

export function sceneRelativeLuminance(rgb: SceneRgb): number {
  return 0.2126 * linearChannel(rgb[0])
    + 0.7152 * linearChannel(rgb[1])
    + 0.0722 * linearChannel(rgb[2]);
}

export function sceneContrastRatio(a: SceneRgb, b: SceneRgb): number {
  const bright = Math.max(sceneRelativeLuminance(a), sceneRelativeLuminance(b));
  const dark = Math.min(sceneRelativeLuminance(a), sceneRelativeLuminance(b));
  return (bright + 0.05) / (dark + 0.05);
}

function rgbCss(rgb: SceneRgb): string {
  const channels = rgb.map((value) => Math.round(clamp01(value) * 255));
  return `rgb(${channels[0]} ${channels[1]} ${channels[2]})`;
}

export function resolveSceneAppearance(
  selection: SceneAppearanceSelection,
  theme: "dark" | "light",
): ResolvedSceneAppearance {
  const background = selection.backgroundMode === "theme"
    ? sceneBackgroundFor(theme)
    : selection.backgroundMode === "custom"
      ? hexToSceneRgb(selection.customBackground)
      : sceneBackgroundFor(selection.backgroundMode);
  const luminance = sceneRelativeLuminance(background);
  const isLight = luminance >= 0.36;
  const ink = isLight ? LIGHT_INK : DARK_INK;
  const mutedInk = isLight ? LIGHT_MUTED_INK : DARK_MUTED_INK;
  const reference: SceneRgb = isLight ? [0.28, 0.33, 0.41] : [0.62, 0.68, 0.77];
  const scrimRgb = isLight ? [1, 1, 1] as SceneRgb : [0.04, 0.05, 0.07] as SceneRgb;
  return {
    background,
    backgroundHex: sceneRgbToHex(background),
    isLight,
    luminance,
    ink,
    mutedInk,
    reference,
    scrim: `color-mix(in srgb, ${rgbCss(scrimRgb)} 72%, transparent)`,
    scrimStrong: `color-mix(in srgb, ${rgbCss(scrimRgb)} 84%, transparent)`,
    lighting: selection.lighting,
    lightingOptions: lightingSettings({
      preset: selection.lighting,
      shadowDepth: normalizeShadowDepth(selection.shadowDepth),
      fillTint: normalizeFillTint(selection.fillTint),
    }),
  };
}

export function contrastAdjustedColor(
  color: SceneRgb,
  background: SceneRgb,
  minimumRatio = 2.2,
): SceneRgb {
  if (sceneContrastRatio(color, background) >= minimumRatio) return color;
  const toward: SceneRgb = sceneRelativeLuminance(background) >= 0.36
    ? [0.035, 0.04, 0.055]
    : [1, 1, 1];
  for (let step = 1; step <= 20; step++) {
    const amount = step / 20;
    const candidate: SceneRgb = [
      color[0] + (toward[0] - color[0]) * amount,
      color[1] + (toward[1] - color[1]) * amount,
      color[2] + (toward[2] - color[2]) * amount,
    ];
    if (sceneContrastRatio(candidate, background) >= minimumRatio) return candidate;
  }
  return toward;
}

export function sceneAppearanceCssVariables(
  appearance: ResolvedSceneAppearance,
): Record<string, string> {
  return {
    "--scene-bg": appearance.backgroundHex,
    "--scene-ink": rgbCss(appearance.ink),
    "--scene-ink-muted": rgbCss(appearance.mutedInk),
    "--scene-scrim": appearance.scrim,
    "--scene-scrim-strong": appearance.scrimStrong,
    "--scene-accent": appearance.isLight ? "oklch(48% 0.11 70)" : "oklch(80% 0.09 82)",
    "--scene-warn": appearance.isLight ? "oklch(49% 0.115 66)" : "oklch(77% 0.12 78)",
    "--scene-error": appearance.isLight ? "oklch(51% 0.16 25)" : "oklch(70% 0.14 22)",
    "--scene-hairline": appearance.isLight ? "rgb(0 0 0 / 0.18)" : "rgb(255 255 255 / 0.18)",
    "--scene-chip": appearance.isLight ? "rgb(0 0 0 / 0.08)" : "rgb(255 255 255 / 0.10)",
  };
}
