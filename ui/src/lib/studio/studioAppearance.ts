import {
  STUDIO_SETTINGS_CHANGE_EVENT,
  loadStudioSettings,
  type StudioAppearance,
  type StudioSettings,
} from "./studioSettings";

export type ResolvedStudioAppearance = Exclude<StudioAppearance, "system">;
export const STUDIO_APPEARANCE_APPLIED_EVENT = "nimbus-studio:appearance-applied";

export function resolveStudioAppearance(
  appearance: StudioAppearance,
  systemPrefersDark: boolean,
): ResolvedStudioAppearance {
  return appearance === "system" ? (systemPrefersDark ? "dark" : "light") : appearance;
}

export function currentStudioAppearance(): ResolvedStudioAppearance {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

/** Apply semantic attributes to the root. Keeping preference and resolved
 * theme separate lets diagnostics say “System · Light” without freezing the
 * operating-system choice into a stored value. */
export function applyStudioAppearance(
  appearance: StudioAppearance,
  systemPrefersDark: boolean,
  root: HTMLElement = document.documentElement,
): ResolvedStudioAppearance {
  const resolved = resolveStudioAppearance(appearance, systemPrefersDark);
  root.dataset.appearance = appearance;
  root.dataset.theme = resolved;
  root.style.colorScheme = resolved;
  return resolved;
}

/**
 * Publish the plot-side preference as a root data attribute.
 *
 * Layout, not colour, but it rides the same settings subscription: it is a
 * document-level preference with exactly one consumer (the app grid), so
 * threading it through React state and prop-drilling it to the shell would buy
 * nothing over a CSS attribute selector, and would re-render every page on a
 * settings save.
 */
export function applyStudioPlotSide(
  side: StudioSettings["plotSide"],
  root: HTMLElement = document.documentElement,
): void {
  root.dataset.plotSide = side;
}

/** Install the app-lifetime controller. It responds both to Settings saves and
 * to live OS appearance changes while the persisted preference is `system`. */
export function installStudioAppearance(): () => void {
  if (typeof window === "undefined" || typeof document === "undefined") return () => undefined;
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  const apply = (settings: StudioSettings = loadStudioSettings()) => {
    applyStudioPlotSide(settings.plotSide);
    document.documentElement.dataset.textSize = settings.textSize;
    const resolved = applyStudioAppearance(settings.appearance, media.matches);
    window.dispatchEvent(new CustomEvent(STUDIO_APPEARANCE_APPLIED_EVENT, {
      detail: { preference: settings.appearance, resolved },
    }));
  };
  const onSettings = (event: Event) => {
    const settings = event instanceof CustomEvent ? event.detail as StudioSettings : loadStudioSettings();
    apply(settings);
  };
  const onSystemChange = () => {
    const settings = loadStudioSettings();
    if (settings.appearance === "system") apply(settings);
  };

  apply();
  window.addEventListener(STUDIO_SETTINGS_CHANGE_EVENT, onSettings);
  media.addEventListener?.("change", onSystemChange);
  return () => {
    window.removeEventListener(STUDIO_SETTINGS_CHANGE_EVENT, onSettings);
    media.removeEventListener?.("change", onSystemChange);
  };
}
