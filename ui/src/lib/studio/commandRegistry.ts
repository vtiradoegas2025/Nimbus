import type { PageId } from "@/lib/core/types";

export type StudioCommandId =
  | "navigate.projects"
  | "navigate.configure"
  | "navigate.runs"
  | "navigate.plots"
  | "navigate.settings"
  | "runs.focus-search"
  | "dataset.open-cm1-history"
  | "dataset.open-nimbus-history";

export interface StudioCommand {
  id: StudioCommandId;
  label: string;
  description: string;
  group: "Navigate" | "Open data" | "Run archive";
  keywords: readonly string[];
  shortcut?: { key: string; shift?: boolean };
  page?: PageId;
}

export const STUDIO_COMMANDS: readonly StudioCommand[] = [
  { id: "navigate.projects", label: "Open Projects", description: "Resume named analysis or publish a portable project recipe.", group: "Navigate", keywords: ["project", "workspace", "publication", "recipe"], shortcut: { key: "0" }, page: "projects" },
  { id: "navigate.configure", label: "Open Configure", description: "Build and review a Nimbus experiment.", group: "Navigate", keywords: ["config", "yaml", "experiment"], shortcut: { key: "1" }, page: "config" },
  { id: "navigate.runs", label: "Open Run Archive", description: "Inspect live and durable local runs.", group: "Navigate", keywords: ["history", "library", "queue"], shortcut: { key: "2" }, page: "runs" },
  { id: "navigate.plots", label: "Open Plot Studio", description: "Analyze the selected run or a netCDF dataset.", group: "Navigate", keywords: ["analysis", "netcdf", "visualization"], shortcut: { key: "3" }, page: "plots" },
  { id: "navigate.settings", label: "Open Settings", description: "Review simulator and storage paths.", group: "Navigate", keywords: ["binary", "directory", "preferences"], shortcut: { key: "," }, page: "settings" },
  { id: "dataset.open-cm1-history", label: "Open netCDF data…", description: "Select atmospheric netCDF files for read-only analysis.", group: "Open data", keywords: ["cm1", "netcdf", "history", "files", "plot"] },
  { id: "dataset.open-nimbus-history", label: "Open Nimbus history / run directory…", description: "Select a published Nimbus run for read-only analysis in Plot Studio.", group: "Open data", keywords: ["nimbus", "history", "run", "directory", "folder", "plot"] },
  { id: "runs.focus-search", label: "Find a run", description: "Focus the Run Archive search field.", group: "Run archive", keywords: ["filter", "case", "status", "path"], shortcut: { key: "/" } },
];

export function isMacPlatform(platform = ""): boolean {
  return /Mac|iPhone|iPad|iPod/i.test(platform);
}

export function platformModifierLabel(mac: boolean): string {
  return mac ? "⌘" : "Ctrl";
}

export function shortcutLabel(command: StudioCommand, mac: boolean): string | null {
  if (!command.shortcut) return null;
  const modifier = platformModifierLabel(mac);
  const key = command.shortcut.key === "," ? "," : command.shortcut.key.toUpperCase();
  return `${modifier}${command.shortcut.shift ? mac ? "⇧" : "+Shift" : mac ? "" : "+"}${key}`;
}

export function ariaShortcut(command: StudioCommand, mac: boolean): string | undefined {
  if (!command.shortcut) return undefined;
  return `${mac ? "Meta" : "Control"}+${command.shortcut.shift ? "Shift+" : ""}${command.shortcut.key}`;
}

export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  const editable = target.closest("input, textarea, select, [contenteditable='true'], [role='textbox']");
  if (!editable) return false;
  // WebKit can keep or restore activeElement inside a dialog after close().
  // That hidden control cannot receive user editing, and treating it as live
  // creates a short interval in which every global command is swallowed.
  const dialog = editable.closest("dialog");
  return !(dialog && !dialog.hasAttribute("open"));
}

export function matchesCommandShortcut(
  event: Pick<KeyboardEvent, "key" | "metaKey" | "ctrlKey" | "altKey" | "shiftKey">,
  command: StudioCommand,
  mac: boolean,
): boolean {
  if (!command.shortcut) return false;
  const modifier = mac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
  return modifier && !event.altKey && event.shiftKey === Boolean(command.shortcut.shift)
    && event.key.toLocaleLowerCase() === command.shortcut.key.toLocaleLowerCase();
}

export function filterCommands(commands: readonly StudioCommand[], query: string): StudioCommand[] {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [...commands];
  return commands.filter((command) => {
    const text = [command.label, command.description, command.group, ...command.keywords]
      .join(" ").toLocaleLowerCase();
    return terms.every((term) => text.includes(term));
  });
}

export function commandForPage(page: PageId): StudioCommand {
  const command = STUDIO_COMMANDS.find((candidate) => candidate.page === page);
  if (!command) throw new Error(`No navigation command registered for '${page}'`);
  return command;
}
