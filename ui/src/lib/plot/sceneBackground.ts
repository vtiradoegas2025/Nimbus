import { currentStudioAppearance } from "@/lib/studio/studioAppearance";

/* Clear colour for a vtk.js scene; follows the user's light/dark theme.
 * Light uses soft paper grey (not white) to preserve cloud-volume silhouette. */

export type SceneRgb = readonly [number, number, number];

export const SCENE_BACKGROUND_DARK: SceneRgb = [0.055, 0.063, 0.075];
export const SCENE_BACKGROUND_LIGHT: SceneRgb = [0.902, 0.910, 0.925];

export function sceneBackgroundFor(appearance: "dark" | "light"): SceneRgb {
  return appearance === "light" ? SCENE_BACKGROUND_LIGHT : SCENE_BACKGROUND_DARK;
}

/** Clear colour for the appearance currently applied to the document. */
export function currentSceneBackground(): SceneRgb {
  return sceneBackgroundFor(currentStudioAppearance());
}
