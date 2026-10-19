import {
  contrastAdjustedColor,
  hexToSceneRgb,
  type ResolvedSceneAppearance,
} from "./sceneAppearance";
import type { SceneRgb } from "./sceneBackground";
import type { CloudEdgeMode } from "./supercellRenderQuality";

export type CloudTone = "auto" | "white" | "neutral" | "storm" | "custom";
export type PrecipitationMode = "total" | "qr" | "qg" | "qh";
export type VortexColorMode = "neutral" | "cloud" | "diagnostic";

export interface VortexSceneColors {
  core: SceneRgb;
  opposite: SceneRgb;
  mesocyclone: SceneRgb;
}

const CLOUD_BASE: Record<Exclude<CloudTone, "auto" | "custom">, SceneRgb> = {
  white: [0.98, 0.985, 1],
  neutral: [0.69, 0.73, 0.79],
  storm: [0.37, 0.42, 0.5],
};

export const PRECIPITATION_COLORS: Record<PrecipitationMode, SceneRgb> = {
  total: [0.16, 0.66, 0.86],
  qr: [0.12, 0.7, 0.9],
  qg: [0.94, 0.59, 0.18],
  qh: [0.82, 0.3, 0.74],
};

function scaleColor(color: SceneRgb, brightness: number): SceneRgb {
  const scale = Math.min(1.2, Math.max(0.3, brightness));
  return color.map((channel) => Math.min(1, Math.max(0, channel * scale))) as unknown as SceneRgb;
}

function mix(a: SceneRgb, b: SceneRgb, amount: number): SceneRgb {
  return [
    a[0] + (b[0] - a[0]) * amount,
    a[1] + (b[1] - a[1]) * amount,
    a[2] + (b[2] - a[2]) * amount,
  ];
}

export function cloudSceneColors(
  tone: CloudTone,
  customColor: string,
  brightness: number,
  edge: CloudEdgeMode,
  appearance: ResolvedSceneAppearance,
): { edge: SceneRgb; body: SceneRgb } {
  const selected = tone === "auto"
    ? appearance.isLight
      ? [0.46, 0.51, 0.59] as SceneRgb
      : [0.78, 0.82, 0.88] as SceneRgb
    : tone === "custom"
      ? hexToSceneRgb(customColor)
      : CLOUD_BASE[tone];
  const body = contrastAdjustedColor(
    scaleColor(selected, brightness),
    appearance.background,
    1.75,
  );
  if (edge === "off") return { edge: body, body };
  const contrastTarget: SceneRgb = appearance.isLight
    ? [0.05, 0.07, 0.1]
    : [0.8, 0.9, 1];
  const edgeAmount = edge === "full" ? 0.28 : 0.14;
  return {
    edge: contrastAdjustedColor(mix(body, contrastTarget, edgeAmount), appearance.background, 1.9),
    body,
  };
}

/** Quiet neutral geometry is the default. Signed diagnostic colors remain
 * available when polarity needs to read without consulting the legend, while
 * cloud matching lets presentation views integrate the extracted surfaces. */
export function vortexSceneColors(
  mode: VortexColorMode,
  cloud: { edge: SceneRgb; body: SceneRgb },
  appearance: ResolvedSceneAppearance,
): VortexSceneColors {
  if (mode === "diagnostic") {
    return {
      core: [0.9, 0.29, 0.32],
      opposite: [0.62, 0.44, 0.86],
      mesocyclone: [0.28, 0.68, 0.64],
    };
  }
  if (mode === "cloud") {
    return {
      core: cloud.body,
      opposite: mix(cloud.body, appearance.background, 0.34),
      mesocyclone: mix(cloud.edge, appearance.background, 0.18),
    };
  }
  return appearance.isLight
    ? {
        core: [0.25, 0.28, 0.33],
        opposite: [0.47, 0.5, 0.55],
        mesocyclone: [0.36, 0.39, 0.44],
      }
    : {
        core: [0.68, 0.71, 0.76],
        opposite: [0.44, 0.47, 0.52],
        mesocyclone: [0.56, 0.59, 0.64],
      };
}

export function precipitationSceneColor(
  mode: PrecipitationMode,
  customColor: string | null,
  appearance: ResolvedSceneAppearance,
): SceneRgb {
  return contrastAdjustedColor(
    customColor ? hexToSceneRgb(customColor) : PRECIPITATION_COLORS[mode],
    appearance.background,
    2,
  );
}
