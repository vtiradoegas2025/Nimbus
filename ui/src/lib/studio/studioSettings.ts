/* Studio-wide preferences in localStorage. Synchronous + best-effort: losing a preference is better than blocking on I/O. */

const STORAGE_KEY = "nimbus-studio:settings:v1";
export const STUDIO_SETTINGS_CHANGE_EVENT = "nimbus-studio:settings-change";

export type StudioAppearance = "system" | "light" | "dark";

/** Which side of the Plot Studio window the figure occupies. Per-user; depends on screen layout and trackpad hand. */
export type StudioPlotSide = "left" | "right";

export interface StudioSettings {
  /** Color appearance. `system` follows the operating-system preference and
   * updates live while Studio is open. */
  appearance: StudioAppearance;
  textSize: "standard" | "comfortable";
  /** A curated template path under configs/ (for example,
   *  "500meterResolution/kessler/baseline.yaml"), as listed by listTemplateConfigs(). Empty string
   *  means "no preferred template — start from schema defaults". */
  defaultTemplate: string;
  /** Root directory where runs land. The simulator writes a
   *  timestamped subdirectory under this root per spawn. Supports the
   *  `~/` prefix; the Rust side expands it. */
  runsDirectory: string;
  /** Absolute path to a nimbus_sim binary that overrides PATH /
   *  auto-resolution. Empty string means "auto". */
  binaryPathOverride: string;
  /** Deprecated persistence-only key retained so existing settings records
   *  deserialize without migration loss. No shipped feature consults it and
   *  it is intentionally absent from Settings. */
  pythonSidecarPath: string;
  /** Absolute path to an `ffmpeg` binary that overrides PATH resolution.
   *  Used by the Plot Studio MP4 export path. Empty string falls back
   *  to the system `ffmpeg`. */
  ffmpegPathOverride: string;
  /** Which side of the window the plot occupies; panels take the other. */
  plotSide: StudioPlotSide;
  loadingWorkers: number;
  ioBudgetMiB: number;
  autoProfileWorkers: boolean;
}

export const STUDIO_SETTINGS_DEFAULTS: StudioSettings = {
  appearance: "system",
  textSize: "standard",
  defaultTemplate: "",
  runsDirectory: "~/.nimbus/runs",
  binaryPathOverride: "",
  pythonSidecarPath: "",
  ffmpegPathOverride: "",
  plotSide: "right",
  loadingWorkers: 0,
  ioBudgetMiB: 0,
  autoProfileWorkers: true,
};

/** One canonical run-library root for launch, discovery, and persisted settings.
 * An empty field means "use the Nimbus default", never the process working directory. */
export function effectiveRunsDirectory(value: string | null | undefined): string {
  const trimmed = value?.trim() ?? "";
  return trimmed || STUDIO_SETTINGS_DEFAULTS.runsDirectory;
}

/** Shallow equality over all persisted keys. Key list is derived from defaults to stay in sync with the interface. */
export function studioSettingsEqual(a: StudioSettings, b: StudioSettings): boolean {
  const keys = Object.keys(STUDIO_SETTINGS_DEFAULTS) as Array<keyof StudioSettings>;
  return keys.every((key) => a[key] === b[key]);
}

export function isStudioAppearance(value: unknown): value is StudioAppearance {
  return value === "system" || value === "light" || value === "dark";
}

export function isStudioPlotSide(value: unknown): value is StudioPlotSide {
  return value === "left" || value === "right";
}

/** Normalize persisted input instead of trusting arbitrary localStorage JSON.
 * Older v1 records have no appearance key and migrate to `system`. */
export function normalizeStudioSettings(value: unknown): StudioSettings {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...STUDIO_SETTINGS_DEFAULTS };
  }
  const parsed = value as Partial<Record<keyof StudioSettings, unknown>>;
  const stringValue = (key: keyof StudioSettings, fallback: string) =>
    typeof parsed[key] === "string" ? parsed[key] as string : fallback;
  return {
    appearance: isStudioAppearance(parsed.appearance)
      ? parsed.appearance
      : STUDIO_SETTINGS_DEFAULTS.appearance,
    textSize: parsed.textSize === "comfortable" ? "comfortable" : "standard",
    defaultTemplate: stringValue("defaultTemplate", STUDIO_SETTINGS_DEFAULTS.defaultTemplate),
    runsDirectory: effectiveRunsDirectory(
      stringValue("runsDirectory", STUDIO_SETTINGS_DEFAULTS.runsDirectory),
    ),
    binaryPathOverride: stringValue("binaryPathOverride", STUDIO_SETTINGS_DEFAULTS.binaryPathOverride),
    pythonSidecarPath: stringValue("pythonSidecarPath", STUDIO_SETTINGS_DEFAULTS.pythonSidecarPath),
    ffmpegPathOverride: stringValue("ffmpegPathOverride", STUDIO_SETTINGS_DEFAULTS.ffmpegPathOverride),
    plotSide: isStudioPlotSide(parsed.plotSide)
      ? parsed.plotSide
      : STUDIO_SETTINGS_DEFAULTS.plotSide,
    loadingWorkers: Number.isSafeInteger(parsed.loadingWorkers)
      && Number(parsed.loadingWorkers) >= 0 && Number(parsed.loadingWorkers) <= 64
      ? Number(parsed.loadingWorkers) : 0,
    ioBudgetMiB: Number.isSafeInteger(parsed.ioBudgetMiB)
      && (parsed.ioBudgetMiB === 0 || Number(parsed.ioBudgetMiB) >= 64 && Number(parsed.ioBudgetMiB) <= 16384)
      ? Number(parsed.ioBudgetMiB) : 0,
    autoProfileWorkers: typeof parsed.autoProfileWorkers === "boolean" ? parsed.autoProfileWorkers : true,
  };
}

export function loadStudioSettings(): StudioSettings {
  if (typeof window === "undefined") return { ...STUDIO_SETTINGS_DEFAULTS };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...STUDIO_SETTINGS_DEFAULTS };
    return normalizeStudioSettings(JSON.parse(raw));
  } catch {
    return { ...STUDIO_SETTINGS_DEFAULTS };
  }
}

/** Persist settings. Returns false when storage is unavailable; the UI must not show "Saved" on false. */
export function saveStudioSettings(s: StudioSettings): boolean {
  if (typeof window === "undefined") return false;
  try {
    const normalized = normalizeStudioSettings(s);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
    window.dispatchEvent(new CustomEvent(STUDIO_SETTINGS_CHANGE_EVENT, { detail: normalized }));
    return true;
  } catch {
    // The in-memory copy in the Settings component remains correct for the
    // rest of the session; only durability is lost.
    return false;
  }
}
