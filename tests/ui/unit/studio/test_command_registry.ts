import {
  STUDIO_COMMANDS,
  ariaShortcut,
  commandForPage,
  filterCommands,
  matchesCommandShortcut,
  shortcutLabel,
} from "../../../../ui/src/lib/studio/commandRegistry.ts";
import { openDatasetForCommand } from "../../../../ui/src/lib/studio/commandActions.ts";
import { createLocalDatasetSource } from "../../../../ui/src/lib/dataset/datasetSources.ts";

let checks = 0;
const check = (condition: boolean, message: string) => {
  if (!condition) throw new Error(message);
  checks += 1;
};

check(new Set(STUDIO_COMMANDS.map(({ id }) => id)).size === STUDIO_COMMANDS.length, "command ids are unique");
check(["projects", "config", "runs", "plots", "settings"].every((page) => commandForPage(page as never)), "every durable page has a command");
check(filterCommands(STUDIO_COMMANDS, "netcdf analysis")[0]?.id === "navigate.plots", "search covers descriptions and keywords");
check(filterCommands(STUDIO_COMMANDS, "archive status")[0]?.id === "runs.focus-search", "multi-term search is conjunctive");
check(filterCommands(STUDIO_COMMANDS, "cm1 netcdf history")[0]?.id === "dataset.open-cm1-history", "CM1 history action is discoverable");
check(filterCommands(STUDIO_COMMANDS, "nimbus run directory")[0]?.id === "dataset.open-nimbus-history", "Nimbus history action is discoverable");
const configure = commandForPage("config");
check(matchesCommandShortcut({ key: "1", metaKey: true, ctrlKey: false, altKey: false, shiftKey: false }, configure, true), "macOS shortcut uses Command");
check(!matchesCommandShortcut({ key: "1", metaKey: false, ctrlKey: true, altKey: false, shiftKey: false }, configure, true), "macOS shortcut rejects Control substitution");
check(matchesCommandShortcut({ key: "1", metaKey: false, ctrlKey: true, altKey: false, shiftKey: false }, configure, false), "non-macOS shortcut uses Control");
check(shortcutLabel(configure, true) === "⌘1" && shortcutLabel(configure, false) === "Ctrl+1", "visual shortcut labels follow platform convention");
check(ariaShortcut(configure, true) === "Meta+1" && ariaShortcut(configure, false) === "Control+1", "ARIA shortcuts follow platform convention");
const cm1Open = STUDIO_COMMANDS.find(({ id }) => id === "dataset.open-cm1-history")!;
check(shortcutLabel(cm1Open, true) === null && ariaShortcut(cm1Open, true) === undefined, "palette-only actions do not claim shortcuts");
check(!matchesCommandShortcut({ key: "o", metaKey: true, ctrlKey: false, altKey: false, shiftKey: false }, cm1Open, true), "palette-only actions cannot match a global shortcut");
check(STUDIO_COMMANDS.every(({ id }) => !/launch|abort|delete|remove/.test(id)), "global registry contains no destructive or launch command");

const cm1Source = createLocalDatasetSource("cm1_netcdf", ["/case/cm1out_000001.nc", "/case/cm1out_000002.nc"]);
const requests: Array<{ kind: string; mode: string | undefined }> = [];
const opened = await openDatasetForCommand("dataset.open-cm1-history", async (kind, mode) => {
  requests.push({ kind, mode });
  return cm1Source;
});
check(requests[0]?.kind === "cm1_netcdf" && requests[0]?.mode === "files", "CM1 command reuses the multi-file picker contract");
check(opened.status === "opened" && opened.source === cm1Source, "selected CM1 source is returned unchanged");

const cancelled = await openDatasetForCommand("dataset.open-nimbus-history", async (kind, mode) => {
  requests.push({ kind, mode });
  return null;
});
check(requests[1]?.kind === "nimbus_run" && requests[1]?.mode === "folder", "Nimbus command reuses the run-directory picker contract");
check(cancelled.status === "cancelled", "picker cancellation is explicit and non-exceptional");
check((await openDatasetForCommand("navigate.plots", async () => cm1Source)).status === "not-handled", "navigation commands never invoke a dataset picker");

let propagated = false;
try {
  await openDatasetForCommand("dataset.open-cm1-history", async () => { throw new Error("picker unavailable"); });
} catch (error) {
  propagated = error instanceof Error && error.message === "picker unavailable";
}
check(propagated, "picker failures propagate for shell-level non-destructive reporting");

console.log(`command registry: ${checks} checks passed`);
